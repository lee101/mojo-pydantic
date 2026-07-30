from __future__ import annotations

import inspect
import math
import re
import types
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    ForwardRef,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from uuid import UUID

from .errors import ValidationError, issue
from .fields import FieldInfo
from .functional_validators import (
    AfterValidator,
    BeforeValidator,
    PlainValidator,
    ValidationInfo,
    WrapValidator,
)


class _ValidationFailure(Exception):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors


@dataclass
class ValidationState:
    strict: bool = False
    context: Any = None
    mode: str = "python"
    config: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    field_name: str | None = None

    def info(self) -> ValidationInfo:
        return ValidationInfo(
            context=self.context,
            config=self.config,
            mode=self.mode,
            data=self.data,
            field_name=self.field_name,
        )


def fail(
    kind: str,
    loc: tuple[Any, ...],
    value: Any,
    *,
    msg: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> None:
    raise _ValidationFailure([issue(kind, loc, value, msg=msg, ctx=ctx)])


def call_with_info(function: Any, value: Any, state: ValidationState, owner: Any = None) -> Any:
    function = function.__func__ if isinstance(function, (classmethod, staticmethod)) else function
    try:
        params = list(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return function(value)
    if owner is not None and params and params[0] in {"cls", "self"}:
        if len(params) >= 3:
            return function(owner, value, state.info())
        return function(owner, value)
    if len(params) >= 2:
        return function(value, state.info())
    return function(value)


def _validator_error(
    exc: Exception, loc: tuple[Any, ...], value: Any
) -> _ValidationFailure:
    if isinstance(exc, _ValidationFailure):
        return exc
    if isinstance(exc, ValidationError):
        return _ValidationFailure(exc.errors(include_url=False))
    kind = "assertion_error" if isinstance(exc, AssertionError) else "value_error"
    return _ValidationFailure(
        [issue(kind, loc, value, ctx={"error": str(exc) or exc.__class__.__name__})]
    )


def _strict_from_metadata(strict: bool, metadata: list[Any]) -> bool:
    for marker in metadata:
        if marker.__class__.__name__ in {"Strict", "StringConstraints"} and getattr(
            marker, "strict", False
        ):
            return True
    return strict


def _field_from_annotated(annotation: Any) -> tuple[Any, FieldInfo | None, list[Any]]:
    if get_origin(annotation) is not Annotated:
        return annotation, None, []
    base, *metadata = get_args(annotation)
    field: FieldInfo | None = None
    validators: list[Any] = []
    for marker in metadata:
        if isinstance(marker, FieldInfo):
            field = marker if field is None else field.merged(marker)
        else:
            validators.append(marker)
    return base, field, validators


def _apply_constraints(
    value: Any,
    field: FieldInfo | None,
    metadata: list[Any],
    loc: tuple[Any, ...],
    original: Any,
) -> Any:
    constraints: dict[str, Any] = {}
    if field is not None:
        for name in (
            "gt",
            "ge",
            "lt",
            "le",
            "multiple_of",
            "min_length",
            "max_length",
            "pattern",
        ):
            candidate = getattr(field, name)
            if candidate is not None:
                constraints[name] = candidate
    for marker in metadata:
        for name in (
            "gt",
            "ge",
            "lt",
            "le",
            "multiple_of",
            "min_length",
            "max_length",
            "pattern",
        ):
            candidate = getattr(marker, name, None)
            if candidate is not None:
                constraints[name] = candidate
    if "gt" in constraints and not value > constraints["gt"]:
        fail("greater_than", loc, original, ctx={"gt": constraints["gt"]})
    if "ge" in constraints and not value >= constraints["ge"]:
        fail("greater_than_equal", loc, original, ctx={"ge": constraints["ge"]})
    if "lt" in constraints and not value < constraints["lt"]:
        fail("less_than", loc, original, ctx={"lt": constraints["lt"]})
    if "le" in constraints and not value <= constraints["le"]:
        fail("less_than_equal", loc, original, ctx={"le": constraints["le"]})
    if "multiple_of" in constraints and value % constraints["multiple_of"] != 0:
        fail(
            "multiple_of",
            loc,
            original,
            msg="Input should be a multiple of {multiple_of}",
            ctx={"multiple_of": constraints["multiple_of"]},
        )
    if "min_length" in constraints and len(value) < constraints["min_length"]:
        kind = "string_too_short" if isinstance(value, str) else "too_short"
        fail(
            kind,
            loc,
            original,
            msg=(
                "String should have at least {min_length} characters"
                if isinstance(value, str)
                else None
            ),
            ctx={"min_length": constraints["min_length"]},
        )
    if "max_length" in constraints and len(value) > constraints["max_length"]:
        kind = "string_too_long" if isinstance(value, str) else "too_long"
        fail(
            kind,
            loc,
            original,
            msg=(
                "String should have at most {max_length} characters"
                if isinstance(value, str)
                else None
            ),
            ctx={"max_length": constraints["max_length"]},
        )
    if "pattern" in constraints and re.search(constraints["pattern"], value) is None:
        fail(
            "string_pattern_mismatch",
            loc,
            original,
            ctx={"pattern": constraints["pattern"]},
        )
    return value


def _validate_int(value: Any, strict: bool, loc: tuple[Any, ...]) -> int:
    if type(value) is int:
        return value
    if isinstance(value, int) and (not strict or not isinstance(value, bool)):
        return int(value)
    if strict:
        fail("int_type", loc, value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, Decimal)):
        if not math.isfinite(float(value)) or value != int(value):
            fail("int_from_float", loc, value)
        return int(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            text = bytes(value).decode() if not isinstance(value, str) else value
            return int(text)
        except (ValueError, UnicodeDecodeError):
            fail("int_parsing", loc, value)
    fail("int_type", loc, value)


def _validate_float(value: Any, strict: bool, loc: tuple[Any, ...]) -> float:
    if type(value) is float:
        return value
    if type(value) is int:
        return float(value)
    if isinstance(value, (int, float, Decimal)) and (
        not strict or not isinstance(value, bool)
    ):
        return float(value)
    if strict:
        fail("float_type", loc, value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            text = bytes(value).decode() if not isinstance(value, str) else value
            return float(text)
        except (ValueError, UnicodeDecodeError):
            fail("float_parsing", loc, value)
    fail("float_type", loc, value)


_TRUE = {"1", "on", "t", "true", "y", "yes"}
_FALSE = {"0", "off", "f", "false", "n", "no"}
_NO_FAST_ITEMS = object()


def _validate_bool(value: Any, strict: bool, loc: tuple[Any, ...]) -> bool:
    if type(value) is bool:
        return value
    if strict:
        fail("bool_type", loc, value)
    if isinstance(value, (int, float, Decimal)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (int, float, Decimal)):
        fail("bool_parsing", loc, value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            text = bytes(value).decode() if not isinstance(value, str) else value
        except UnicodeDecodeError:
            fail("bool_parsing", loc, value)
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        fail("bool_parsing", loc, value)
    fail("bool_type", loc, value)


def _validate_str(
    value: Any, strict: bool, loc: tuple[Any, ...], config: dict[str, Any]
) -> str:
    if isinstance(value, str):
        result = value
    elif not strict and isinstance(value, (bytes, bytearray)):
        try:
            result = bytes(value).decode()
        except UnicodeDecodeError:
            fail("string_unicode", loc, value, msg="Input should be valid UTF-8")
    elif not strict and config.get("coerce_numbers_to_str") and isinstance(
        value, (int, float, Decimal)
    ):
        result = str(value)
    else:
        fail("string_type", loc, value)
    if config.get("str_strip_whitespace"):
        result = result.strip()
    if config.get("str_to_lower"):
        result = result.lower()
    if config.get("str_to_upper"):
        result = result.upper()
    min_length = config.get("str_min_length")
    max_length = config.get("str_max_length")
    if min_length is not None and len(result) < min_length:
        fail("string_too_short", loc, value, msg=f"String should have at least {min_length} characters")
    if max_length is not None and len(result) > max_length:
        fail("string_too_long", loc, value, msg=f"String should have at most {max_length} characters")
    return result


def validate_value(
    annotation: Any,
    value: Any,
    state: ValidationState,
    loc: tuple[Any, ...] = (),
    field: FieldInfo | None = None,
) -> Any:
    if field is None:
        if annotation is int:
            return _validate_int(value, state.strict, loc)
        if annotation is float:
            return _validate_float(value, state.strict, loc)
        if annotation is bool:
            return _validate_bool(value, state.strict, loc)
        if annotation is str:
            return _validate_str(value, state.strict, loc, state.config or {})
    elif (
        field.gt is None
        and field.ge is None
        and field.lt is None
        and field.le is None
        and field.multiple_of is None
        and field.min_length is None
        and field.max_length is None
        and field.pattern is None
        and not field.metadata
    ):
        strict = state.strict if field.strict is None else field.strict
        if annotation is int:
            return _validate_int(value, strict, loc)
        if annotation is float:
            return _validate_float(value, strict, loc)
        if annotation is bool:
            return _validate_bool(value, strict, loc)

    original = value
    annotation, annotated_field, metadata = _field_from_annotated(annotation)
    if annotated_field is not None:
        field = annotated_field if field is None else field.merged(annotated_field)
    strict = _strict_from_metadata(
        field.strict if field is not None and field.strict is not None else state.strict,
        metadata,
    )

    for marker in metadata:
        try:
            if isinstance(marker, BeforeValidator):
                value = call_with_info(marker.func, value, state)
            elif isinstance(marker, PlainValidator):
                return _apply_constraints(
                    call_with_info(marker.func, value, state), field, metadata, loc, original
                )
            elif isinstance(marker, WrapValidator):
                handler = lambda item: validate_value(annotation, item, state, loc, field)
                value = marker.func(value, handler, state.info())
        except Exception as exc:
            raise _validator_error(exc, loc, original) from exc

    origin = get_origin(annotation)
    args = get_args(annotation)
    config = state.config or {}

    if annotation in (Any, object) or isinstance(annotation, TypeVar):
        result = value
    elif annotation is None or annotation is type(None):
        if value is not None:
            fail("none_required", loc, value)
        result = None
    elif annotation is int:
        result = _validate_int(value, strict, loc)
    elif annotation is float:
        result = _validate_float(value, strict, loc)
    elif annotation is bool:
        result = _validate_bool(value, strict, loc)
    elif annotation is str:
        result = _validate_str(value, strict, loc, config)
        for marker in metadata:
            if marker.__class__.__name__ == "StringConstraints":
                if getattr(marker, "strip_whitespace", False):
                    result = result.strip()
                if getattr(marker, "to_upper", False):
                    result = result.upper()
                if getattr(marker, "to_lower", False):
                    result = result.lower()
    elif annotation is bytes:
        if isinstance(value, bytes):
            result = value
        elif (not strict or state.mode == "json") and isinstance(value, (str, bytearray)):
            result = value.encode() if isinstance(value, str) else bytes(value)
        else:
            fail("bytes_type", loc, value)
    elif annotation is Decimal:
        try:
            if strict and state.mode != "json" and not isinstance(value, Decimal):
                raise InvalidOperation
            result = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError):
            fail("decimal_parsing", loc, value)
    elif annotation is UUID:
        try:
            if strict and state.mode != "json" and not isinstance(value, UUID):
                raise ValueError
            result = value if isinstance(value, UUID) else UUID(str(value))
        except (ValueError, AttributeError):
            fail("uuid_parsing", loc, value)
    elif annotation in (date, datetime, time):
        if isinstance(value, annotation):
            result = value
        elif (not strict or state.mode == "json") and isinstance(value, str):
            try:
                result = annotation.fromisoformat(value)
            except ValueError:
                kind = "date_from_datetime_parsing" if annotation is date else "datetime_from_date_parsing"
                fail(kind, loc, value)
        else:
            fail(f"{annotation.__name__}_type", loc, value, msg=f"Input should be a valid {annotation.__name__}")
    elif annotation is timedelta:
        if isinstance(value, timedelta):
            result = value
        elif not strict and isinstance(value, (int, float)):
            result = timedelta(seconds=value)
        else:
            fail("time_delta_type", loc, value, msg="Input should be a valid timedelta")
    elif annotation is Path:
        if isinstance(value, Path):
            result = value
        elif not strict and isinstance(value, (str, bytes)):
            result = Path(value)
        else:
            fail("path_type", loc, value, msg="Input should be a valid path")
    elif origin is Literal:
        if not any(value == candidate and type(value) is type(candidate) for candidate in args):
            expected = ", ".join(repr(candidate) for candidate in args)
            fail("literal_error", loc, value, msg=f"Input should be {expected}", ctx={"expected": expected})
        result = value
    elif origin in (Union, types.UnionType):
        exact = [
            branch
            for branch in args
            if isinstance(branch, type) and isinstance(value, branch)
        ]
        ordered = [*exact, *(branch for branch in args if branch not in exact)]
        failures: list[dict[str, Any]] = []
        for branch in ordered:
            try:
                result = validate_value(branch, value, state, loc)
                break
            except _ValidationFailure as exc:
                label = getattr(branch, "__name__", str(branch))
                failures.extend(
                    [{**item, "loc": (*loc, label, *item["loc"][len(loc):])} for item in exc.errors]
                )
        else:
            raise _ValidationFailure(failures)
    elif origin in (list, Sequence) or annotation is list:
        if isinstance(value, list):
            source = value
        elif not strict and isinstance(value, (tuple, set, frozenset, deque)):
            source = list(value)
        else:
            fail("list_type", loc, value)
        item_type = args[0] if args else Any
        result = _validate_items(source, item_type, state, loc)
    elif origin is tuple or annotation is tuple:
        if not isinstance(value, tuple):
            if strict or not isinstance(value, (list, set, deque)):
                fail("tuple_type", loc, value)
            value = tuple(value)
        if not args:
            result = value
        elif len(args) == 2 and args[1] is Ellipsis:
            result = tuple(_validate_items(value, args[0], state, loc))
        else:
            if len(value) != len(args):
                fail(
                    "tuple_type",
                    loc,
                    value,
                    msg=f"Tuple should have {len(args)} items",
                )
            result = tuple(
                validate_value(item_type, item, state, (*loc, index))
                for index, (item_type, item) in enumerate(zip(args, value))
            )
    elif origin in (set, frozenset) or annotation in (set, frozenset):
        expected = origin or annotation
        if not isinstance(value, expected):
            if strict or not isinstance(value, (list, tuple, set, frozenset, deque)):
                fail("set_type", loc, value)
        item_type = args[0] if args else Any
        validated = _validate_items(list(value), item_type, state, loc)
        result = expected(validated)
    elif origin is dict or annotation is dict:
        if not isinstance(value, Mapping):
            fail("dict_type", loc, value)
        key_type, item_type = args if len(args) == 2 else (Any, Any)
        result = {}
        failures = []
        for key, item in value.items():
            try:
                new_key = validate_value(key_type, key, state, (*loc, key, "[key]"))
                result[new_key] = validate_value(item_type, item, state, (*loc, key))
            except _ValidationFailure as exc:
                failures.extend(exc.errors)
        if failures:
            raise _ValidationFailure(failures)
    elif inspect.isclass(annotation) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            result = value
        elif strict:
            fail("is_instance_of", loc, value, msg=f"Input should be an instance of {annotation.__name__}")
        else:
            try:
                result = annotation(value)
            except ValueError:
                fail("enum", loc, value)
        if config.get("use_enum_values"):
            result = result.value
    elif (
        inspect.isclass(annotation)
        and annotation.__name__ == "SecretStr"
        and hasattr(annotation, "get_secret_value")
    ):
        if isinstance(value, annotation):
            result = value
        else:
            result = annotation(_validate_str(value, strict, loc, config))
    elif inspect.isclass(annotation) and hasattr(annotation, "model_validate"):
        try:
            if hasattr(annotation, "_validate_model"):
                result = annotation._validate_model(
                    value,
                    strict=strict,
                    context=state.context,
                    _mode=state.mode,
                )
            else:
                result = annotation.model_validate(
                    value, strict=strict, context=state.context
                )
        except ValidationError as exc:
            raise _ValidationFailure(
                [{**item, "loc": (*loc, *item["loc"])} for item in exc.errors(include_url=False)]
            ) from exc
    elif inspect.isclass(annotation) and is_dataclass(annotation):
        if isinstance(value, annotation):
            result = value
        elif isinstance(value, Mapping):
            try:
                result = annotation(**value)
            except (TypeError, ValueError) as exc:
                fail("dataclass_type", loc, value, msg=str(exc))
        else:
            fail("dataclass_type", loc, value, msg="Input should be a dictionary or dataclass")
    elif inspect.isclass(annotation):
        if not isinstance(value, annotation):
            fail("is_instance_of", loc, value, msg=f"Input should be an instance of {annotation.__name__}")
        result = value
    elif isinstance(annotation, ForwardRef):
        result = value
    else:
        result = value

    result = _apply_constraints(result, field, metadata, loc, original)
    for marker in metadata:
        if (
            marker.__class__.__name__ == "AllowInfNan"
            and not getattr(marker, "allow_inf_nan", True)
            and isinstance(result, float)
            and not math.isfinite(result)
        ):
            fail("finite_number", loc, original, msg="Input should be a finite number")
    for marker in metadata:
        if isinstance(marker, AfterValidator):
            try:
                result = call_with_info(marker.func, result, state)
            except Exception as exc:
                raise _validator_error(exc, loc, original) from exc
    return result


def _validate_items(
    values: Sequence[Any], item_type: Any, state: ValidationState, loc: tuple[Any, ...]
) -> list[Any]:
    fast = _fast_primitive_items(values, item_type, state.strict)
    if fast is not _NO_FAST_ITEMS:
        return fast
    result = []
    failures = []
    if item_type is int:
        validator = _validate_int
    elif item_type is float:
        validator = _validate_float
    elif item_type is bool:
        validator = _validate_bool
    else:
        validator = None
    if validator is not None:
        for index, item in enumerate(values):
            try:
                result.append(validator(item, state.strict, (*loc, index)))
            except _ValidationFailure as exc:
                failures.extend(exc.errors)
        if failures:
            raise _ValidationFailure(failures)
        return result
    for index, item in enumerate(values):
        try:
            result.append(validate_value(item_type, item, state, (*loc, index)))
        except _ValidationFailure as exc:
            failures.extend(exc.errors)
    if failures:
        raise _ValidationFailure(failures)
    return result


def _fast_primitive_items(
    values: Sequence[Any], item_type: Any, strict: bool
) -> list[Any] | object:
    result: list[Any] = []
    append = result.append
    try:
        if item_type is int:
            for item in values:
                item_class = type(item)
                if item_class is int:
                    append(item)
                elif not strict and item_class is bool:
                    append(int(item))
                elif not strict and item_class in (str, bytes, bytearray):
                    append(int(item))
                elif not strict and item_class in (float, Decimal):
                    if not math.isfinite(float(item)) or item != int(item):
                        return _NO_FAST_ITEMS
                    append(int(item))
                else:
                    return _NO_FAST_ITEMS
            return result
        if item_type is float:
            for item in values:
                item_class = type(item)
                if item_class in (int, float, Decimal) and item_class is not bool:
                    append(float(item))
                elif not strict and item_class is bool:
                    append(float(item))
                elif not strict and item_class in (str, bytes, bytearray):
                    append(float(item))
                else:
                    return _NO_FAST_ITEMS
            return result
        if item_type is bool:
            for item in values:
                item_class = type(item)
                if item_class is bool:
                    append(item)
                elif strict:
                    return _NO_FAST_ITEMS
                elif item_class in (int, float, Decimal):
                    if item not in (0, 1):
                        return _NO_FAST_ITEMS
                    append(bool(item))
                elif item_class in (str, bytes, bytearray):
                    text = (
                        bytes(item).decode()
                        if item_class is not str
                        else item
                    ).lower()
                    if text in _TRUE:
                        append(True)
                    elif text in _FALSE:
                        append(False)
                    else:
                        return _NO_FAST_ITEMS
                else:
                    return _NO_FAST_ITEMS
            return result
    except (TypeError, ValueError, UnicodeDecodeError, OverflowError):
        return _NO_FAST_ITEMS
    return _NO_FAST_ITEMS


def validation_error(title: str, exc: _ValidationFailure) -> ValidationError:
    return ValidationError(title, exc.errors)


def dump_value(value: Any, mode: str = "python", by_alias: bool = False) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode=mode, by_alias=by_alias)
    if isinstance(value, Enum):
        return value.value if mode == "json" else value
    if value.__class__.__name__ == "SecretStr" and hasattr(value, "get_secret_value"):
        return str(value) if mode == "json" else value
    if isinstance(value, Mapping):
        return {
            dump_value(key, mode, by_alias): dump_value(item, mode, by_alias)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        items = [dump_value(item, mode, by_alias) for item in value]
        return items if mode == "json" else type(value)(items)
    if mode == "json":
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, timedelta):
            return value.total_seconds()
        if isinstance(value, (UUID, Path, Decimal)):
            return str(value)
        if isinstance(value, bytes):
            return value.decode()
    return value


def schema_for(annotation: Any) -> dict[str, Any]:
    annotation, field, metadata = _field_from_annotated(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is int:
        schema: dict[str, Any] = {"type": "integer"}
    elif annotation is float:
        schema = {"type": "number"}
    elif annotation is bool:
        schema = {"type": "boolean"}
    elif annotation is str:
        schema = {"type": "string"}
    elif annotation is bytes:
        schema = {"type": "string", "format": "binary"}
    elif annotation is None or annotation is type(None):
        schema = {"type": "null"}
    elif annotation in (date, datetime, time):
        schema = {"type": "string", "format": annotation.__name__}
    elif annotation is UUID:
        schema = {"type": "string", "format": "uuid"}
    elif annotation is Decimal:
        schema = {"anyOf": [{"type": "number"}, {"type": "string"}]}
    elif origin in (list, set, frozenset, Sequence) or annotation in (list, set, frozenset):
        schema = {"type": "array", "items": schema_for(args[0] if args else Any)}
        if origin in (set, frozenset) or annotation in (set, frozenset):
            schema["uniqueItems"] = True
    elif origin is tuple or annotation is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            schema = {"type": "array", "items": schema_for(args[0])}
        else:
            schema = {"type": "array", "prefixItems": [schema_for(arg) for arg in args]}
    elif origin is dict or annotation is dict:
        schema = {"type": "object", "additionalProperties": schema_for(args[1] if args else Any)}
    elif origin in (Union, types.UnionType):
        schema = {"anyOf": [schema_for(arg) for arg in args]}
    elif origin is Literal:
        schema = {"enum": list(args)}
    elif inspect.isclass(annotation) and issubclass(annotation, Enum):
        schema = {"enum": [member.value for member in annotation]}
    elif inspect.isclass(annotation) and hasattr(annotation, "model_json_schema"):
        schema = annotation.model_json_schema()
    else:
        schema = {}
    constraints = field or FieldInfo()
    mapping = {
        "gt": "exclusiveMinimum",
        "ge": "minimum",
        "lt": "exclusiveMaximum",
        "le": "maximum",
        "multiple_of": "multipleOf",
        "min_length": "minLength" if annotation is str else "minItems",
        "max_length": "maxLength" if annotation is str else "maxItems",
        "pattern": "pattern",
    }
    for source, target in mapping.items():
        value = getattr(constraints, source)
        if value is not None:
            schema[target] = value
    return schema
