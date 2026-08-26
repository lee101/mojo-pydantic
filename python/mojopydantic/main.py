from __future__ import annotations

import inspect
import json
import sys
from copy import copy, deepcopy
from typing import Any, ClassVar, get_args, get_origin

from ._core import (
    _NO_FAST_ITEMS,
    _ValidationFailure,
    ValidationState,
    _fast_primitive_items,
    _validate_bool,
    _validate_float,
    _validate_int,
    _validate_items,
    dump_value,
    schema_for,
    validate_value,
    validation_error,
)
from .errors import ValidationError, issue
from .fields import (
    AliasChoices,
    AliasPath,
    FieldInfo,
    PydanticUndefined,
)


class ConfigDict(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kwargs)


def _metadata(target: Any, name: str) -> Any:
    function = target.__func__ if isinstance(target, (classmethod, staticmethod)) else target
    return getattr(function, name, None)


def _invoke(function: Any, value: Any, state: ValidationState, handler: Any = None) -> Any:
    parameters = list(inspect.signature(function).parameters)
    if handler is not None:
        if len(parameters) >= 3:
            return function(value, handler, state.info())
        return function(value, handler)
    if len(parameters) >= 2:
        return function(value, state.info())
    if len(parameters) == 1:
        return function(value)
    return function()


def _validator_failure(exc: Exception, loc: tuple[Any, ...], value: Any) -> _ValidationFailure:
    if isinstance(exc, _ValidationFailure):
        return exc
    if isinstance(exc, ValidationError):
        return _ValidationFailure(exc.errors(include_url=False))
    kind = "assertion_error" if isinstance(exc, AssertionError) else "value_error"
    return _ValidationFailure(
        [issue(kind, loc, value, ctx={"error": str(exc) or exc.__class__.__name__})]
    )


def _simple_model_plan(
    fields: dict[str, FieldInfo],
    config: dict[str, Any],
    field_validators: list[tuple[str, tuple[str, ...], str]],
    model_validators: list[tuple[str, str]],
) -> tuple[tuple[str, FieldInfo, Any, Any], ...] | None:
    if (
        field_validators
        or model_validators
        or config.get("extra", "ignore") != "ignore"
        or config.get("from_attributes", False)
    ):
        return None
    plan = []
    for name, field in fields.items():
        annotation = field.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)
        if (
            not field.is_required()
            or field.alias is not None
            or field.validation_alias is not None
            or field.strict is not None
            or field.gt is not None
            or field.ge is not None
            or field.lt is not None
            or field.le is not None
            or field.multiple_of is not None
            or field.min_length is not None
            or field.max_length is not None
            or field.pattern is not None
            or field.metadata
        ):
            return None
        if annotation is int:
            plan.append((name, field, _validate_int, None))
        elif annotation is float:
            plan.append((name, field, _validate_float, None))
        elif annotation is bool:
            plan.append((name, field, _validate_bool, None))
        elif origin is list and len(args) == 1 and args[0] in (int, float, bool):
            plan.append((name, field, None, args[0]))
        else:
            return None
    return tuple(plan)


class ModelMetaclass(type):
    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **class_config: Any,
    ) -> type:
        inherited_fields: dict[str, FieldInfo] = {}
        inherited_config: dict[str, Any] = {}
        inherited_field_validators: list[tuple[str, tuple[str, ...], str]] = []
        inherited_model_validators: list[tuple[str, str]] = []
        inherited_serializers: list[tuple[str, tuple[str, ...], str, str]] = []
        inherited_computed: dict[str, dict[str, Any]] = {}
        for base in bases:
            inherited_fields.update(
                {key: copy(value) for key, value in getattr(base, "model_fields", {}).items()}
            )
            inherited_config.update(getattr(base, "model_config", {}))
            inherited_field_validators.extend(getattr(base, "__field_validators__", []))
            inherited_model_validators.extend(getattr(base, "__model_validators__", []))
            inherited_serializers.extend(getattr(base, "__field_serializers__", []))
            inherited_computed.update(getattr(base, "__computed_fields__", {}))

        supplied_config = namespace.get("model_config", {})
        if isinstance(supplied_config, dict):
            inherited_config.update(supplied_config)
        inherited_config.update(class_config)

        annotations = dict(namespace.get("__annotations__", {}))
        fields = inherited_fields
        for field_name, annotation in annotations.items():
            if (
                get_origin(annotation) is ClassVar
                or (isinstance(annotation, str) and annotation.startswith("ClassVar["))
                or field_name.startswith("_")
                or field_name in {"model_fields", "model_config"}
            ):
                continue
            default = namespace.get(field_name, PydanticUndefined)
            if isinstance(default, FieldInfo):
                info = copy(default)
            else:
                info = FieldInfo(default=default)
            info.annotation = annotation
            fields[field_name] = info
            namespace.pop(field_name, None)

        field_validators = inherited_field_validators
        model_validators = inherited_model_validators
        serializers = inherited_serializers
        computed = inherited_computed
        for attr_name, target in namespace.items():
            marker = _metadata(target, "__mp_field_validator__")
            if marker is not None:
                validator_fields, mode, _ = marker
                field_validators.append((attr_name, validator_fields, mode))
            marker = _metadata(target, "__mp_model_validator__")
            if marker is not None:
                model_validators.append((attr_name, marker))
            marker = _metadata(target, "__mp_field_serializer__")
            if marker is not None:
                serializer_fields, mode, when_used = marker
                serializers.append((attr_name, serializer_fields, mode, when_used))
            if isinstance(target, property):
                marker = getattr(target.fget, "__mp_computed_field__", None)
                if marker is not None:
                    computed[attr_name] = marker

        namespace["model_fields"] = fields
        namespace["model_config"] = inherited_config
        namespace["__field_validators__"] = field_validators
        namespace["__model_validators__"] = model_validators
        namespace["__field_serializers__"] = serializers
        namespace["__computed_fields__"] = computed
        cls = super().__new__(mcls, name, bases, namespace)
        if name != "BaseModel":
            cls.model_rebuild(raise_errors=False)
        cls.__simple_model_plan__ = _simple_model_plan(
            cls.model_fields,
            cls.model_config,
            cls.__field_validators__,
            cls.__model_validators__,
        )
        return cls


class BaseModel(metaclass=ModelMetaclass):
    model_fields: ClassVar[dict[str, FieldInfo]] = {}
    model_config: ClassVar[dict[str, Any]] = {}
    __field_validators__: ClassVar[list[tuple[str, tuple[str, ...], str]]] = []
    __model_validators__: ClassVar[list[tuple[str, str]]] = []
    __field_serializers__: ClassVar[list[tuple[str, tuple[str, ...], str, str]]] = []
    __computed_fields__: ClassVar[dict[str, dict[str, Any]]] = {}
    __simple_model_plan__: ClassVar[
        tuple[tuple[str, FieldInfo, Any, Any], ...] | None
    ] = None

    def __init__(self, /, **data: Any) -> None:
        validated = self.__class__._validate_model(data)
        self.__dict__.update(validated.__dict__)

    @classmethod
    def _validate_model(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: Any = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
        _mode: str = "python",
    ) -> "BaseModel":
        if isinstance(obj, cls):
            if cls.model_config.get("revalidate_instances") == "always":
                obj = obj.model_dump()
            else:
                return obj
        if (
            cls.__simple_model_plan__ is not None
            and isinstance(obj, dict)
            and extra is None
            and from_attributes is None
            and context is None
            and by_alias is None
            and by_name is None
            and _mode == "python"
        ):
            return cls._validate_simple_dict(
                obj,
                cls.model_config.get("strict", False) if strict is None else strict,
            )
        state = ValidationState(
            strict=cls.model_config.get("strict", False) if strict is None else strict,
            context=context,
            mode=_mode,
            config=cls.model_config,
        )
        for validator_name, mode in cls.__model_validators__:
            if mode != "before":
                continue
            try:
                obj = _invoke(getattr(cls, validator_name), obj, state)
            except Exception as exc:
                raise validation_error(cls.__name__, _validator_failure(exc, (), obj)) from exc

        use_attributes = (
            cls.model_config.get("from_attributes", False)
            if from_attributes is None
            else from_attributes
        )
        if isinstance(obj, dict):
            source = obj
        elif hasattr(obj, "items"):
            source = dict(obj.items())
        elif use_attributes:
            source = {
                name: getattr(obj, name)
                for name in cls.model_fields
                if hasattr(obj, name)
            }
        else:
            raise ValidationError(
                cls.__name__,
                [
                    issue(
                        "model_type",
                        (),
                        obj,
                        msg=f"Input should be a valid dictionary or instance of {cls.__name__}",
                    )
                ],
            )

        values: dict[str, Any] = {}
        fields_set: set[str] = set()
        failures: list[dict[str, Any]] = []
        extra_mode = extra or cls.model_config.get("extra", "ignore")
        consumed: set[str] | None = set() if extra_mode != "ignore" else None
        allow_names = (
            cls.model_config.get("populate_by_name", False)
            or cls.model_config.get("validate_by_name", False)
            or by_name is True
        )
        allow_aliases = by_alias is not False
        for field_name, field in cls.model_fields.items():
            found = False
            raw = PydanticUndefined
            validation_alias = field.validation_alias or field.alias
            aliases: list[str | AliasPath] = []
            if allow_aliases and isinstance(validation_alias, AliasChoices):
                aliases.extend(validation_alias.choices)
            elif allow_aliases and validation_alias is not None:
                aliases.append(validation_alias)
            if not aliases or allow_names:
                aliases.append(field_name)
            used_location: Any = field_name
            if aliases == [field_name]:
                if field_name in source:
                    raw, found = source[field_name], True
                    if consumed is not None:
                        consumed.add(field_name)
            else:
                for alias in aliases:
                    if isinstance(alias, AliasPath):
                        candidate = alias.search_dict_for_path(source)
                        if candidate is not PydanticUndefined:
                            raw, found, used_location = candidate, True, alias.path[0]
                            if consumed is not None:
                                consumed.add(str(alias.path[0]))
                            break
                    elif alias in source:
                        raw, found, used_location = source[alias], True, alias
                        if consumed is not None:
                            consumed.add(alias)
                        break
            if not found:
                if field.default_factory is not None:
                    raw = field.default_factory()
                elif field.default is not PydanticUndefined:
                    raw = deepcopy(field.default)
                else:
                    failures.append(issue("missing", (field.alias or field_name,), source))
                    continue
                if not (field.validate_default or cls.model_config.get("validate_default")):
                    values[field_name] = raw
                    continue
            else:
                fields_set.add(field_name)

            field_state = ValidationState(
                strict=state.strict,
                context=context,
                mode=_mode,
                config=cls.model_config,
                data=values,
                field_name=field_name,
            )
            try:
                for validator_name, validator_fields, mode in cls.__field_validators__:
                    if field_name not in validator_fields and "*" not in validator_fields:
                        continue
                    if mode == "before":
                        raw = _invoke(getattr(cls, validator_name), raw, field_state)
                    elif mode == "plain":
                        raw = _invoke(getattr(cls, validator_name), raw, field_state)
                        break
                    elif mode == "wrap":
                        handler = lambda value: validate_value(
                            field.annotation, value, field_state, (used_location,), field
                        )
                        raw = _invoke(
                            getattr(cls, validator_name), raw, field_state, handler
                        )
                        break
                else:
                    raw = validate_value(
                        field.annotation, raw, field_state, (used_location,), field
                    )
                for validator_name, validator_fields, mode in cls.__field_validators__:
                    if (
                        mode == "after"
                        and (field_name in validator_fields or "*" in validator_fields)
                    ):
                        raw = _invoke(getattr(cls, validator_name), raw, field_state)
                values[field_name] = raw
            except Exception as exc:
                failures.extend(_validator_failure(exc, (used_location,), raw).errors)

        extras = (
            {
                key: value
                for key, value in source.items()
                if consumed is not None and key not in consumed
            }
            if extra_mode != "ignore"
            else {}
        )
        if extra_mode == "forbid":
            failures.extend(
                issue("extra_forbidden", (key,), value) for key, value in extras.items()
            )
        elif extra_mode == "allow":
            values.update(extras)
        if failures:
            raise ValidationError(cls.__name__, failures)

        instance = cls.__new__(cls)
        object.__setattr__(instance, "__dict__", values)
        object.__setattr__(instance, "__pydantic_fields_set__", fields_set)
        object.__setattr__(
            instance, "__pydantic_extra__", extras if extra_mode == "allow" else None
        )
        for validator_name, mode in cls.__model_validators__:
            if mode == "after":
                try:
                    candidate = _invoke(getattr(instance, validator_name), instance, state)
                    if candidate is not None:
                        instance = candidate
                except Exception as exc:
                    raise validation_error(
                        cls.__name__, _validator_failure(exc, (), source)
                    ) from exc
            elif mode == "wrap":
                try:
                    instance = _invoke(
                        getattr(cls, validator_name),
                        source,
                        state,
                        lambda _: instance,
                    )
                except Exception as exc:
                    raise validation_error(
                        cls.__name__, _validator_failure(exc, (), source)
                    ) from exc
        return instance

    @classmethod
    def _validate_simple_dict(
        cls, source: dict[str, Any], strict: bool
    ) -> "BaseModel":
        plan = cls.__simple_model_plan__ or ()
        values: dict[str, Any] = {}
        try:
            for name, _, scalar_validator, item_type in plan:
                raw = source[name]
                raw_type = type(raw)
                if scalar_validator is _validate_int:
                    if raw_type is int:
                        values[name] = raw
                    elif not strict and raw_type in (str, bytes, bytearray, bool):
                        values[name] = int(raw)
                    else:
                        return cls._validate_simple_dict_diagnostic(source, strict)
                elif scalar_validator is _validate_float:
                    if raw_type is float:
                        values[name] = raw
                    elif raw_type is int or (
                        not strict and raw_type in (str, bytes, bytearray, bool)
                    ):
                        values[name] = float(raw)
                    else:
                        return cls._validate_simple_dict_diagnostic(source, strict)
                elif scalar_validator is _validate_bool:
                    if raw_type is bool:
                        values[name] = raw
                    elif not strict and raw_type in (str, bytes, bytearray):
                        text = raw if raw_type is str else bytes(raw).decode()
                        lowered = text.lower()
                        if lowered in ("1", "on", "t", "true", "y", "yes"):
                            values[name] = True
                        elif lowered in ("0", "off", "f", "false", "n", "no"):
                            values[name] = False
                        else:
                            return cls._validate_simple_dict_diagnostic(source, strict)
                    else:
                        return cls._validate_simple_dict_diagnostic(source, strict)
                elif raw_type is list:
                    parsed = _fast_primitive_items(raw, item_type, strict)
                    if parsed is _NO_FAST_ITEMS:
                        return cls._validate_simple_dict_diagnostic(source, strict)
                    values[name] = parsed
                else:
                    return cls._validate_simple_dict_diagnostic(source, strict)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, OverflowError):
            return cls._validate_simple_dict_diagnostic(source, strict)
        instance = cls.__new__(cls)
        object.__setattr__(instance, "__dict__", values)
        object.__setattr__(instance, "__pydantic_fields_set__", set(values))
        object.__setattr__(instance, "__pydantic_extra__", None)
        return instance

    @classmethod
    def _validate_simple_dict_diagnostic(
        cls, source: dict[str, Any], strict: bool
    ) -> "BaseModel":
        values: dict[str, Any] = {}
        failures: list[dict[str, Any]] = []
        plan = cls.__simple_model_plan__ or ()
        for name, field, scalar_validator, item_type in plan:
            try:
                raw = source[name]
            except KeyError:
                failures.append(issue("missing", (name,), source))
                continue
            try:
                if scalar_validator is not None:
                    values[name] = scalar_validator(raw, strict, (name,))
                elif isinstance(raw, list):
                    parsed = _fast_primitive_items(raw, item_type, strict)
                    values[name] = (
                        _validate_items(raw, item_type, ValidationState(strict=strict), (name,))
                        if parsed is _NO_FAST_ITEMS
                        else parsed
                    )
                else:
                    values[name] = validate_value(
                        field.annotation,
                        raw,
                        ValidationState(strict=strict),
                        (name,),
                        field,
                    )
            except _ValidationFailure as exc:
                failures.extend(exc.errors)
        if failures:
            raise ValidationError(cls.__name__, failures)
        instance = cls.__new__(cls)
        object.__setattr__(instance, "__dict__", values)
        object.__setattr__(
            instance, "__pydantic_fields_set__", {item[0] for item in plan}
        )
        object.__setattr__(instance, "__pydantic_extra__", None)
        return instance

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: Any = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "BaseModel":
        return cls._validate_model(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "BaseModel":
        try:
            value = json.loads(json_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValidationError(
                cls.__name__,
                [issue("json_invalid", (), json_data, ctx={"error": str(exc)})],
            ) from exc
        return cls._validate_model(
            value,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
            _mode="json",
        )

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "BaseModel":
        return cls._validate_model(
            obj,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
            _mode="strings",
        )

    @classmethod
    def model_construct(
        cls, _fields_set: set[str] | None = None, **values: Any
    ) -> "BaseModel":
        instance = cls.__new__(cls)
        data = {}
        for name, field in cls.model_fields.items():
            if name in values:
                data[name] = values.pop(name)
            elif field.default_factory is not None:
                data[name] = field.default_factory()
            elif field.default is not PydanticUndefined:
                data[name] = deepcopy(field.default)
        if cls.model_config.get("extra") == "allow":
            data.update(values)
        object.__setattr__(instance, "__dict__", data)
        object.__setattr__(
            instance, "__pydantic_fields_set__", _fields_set or set(data)
        )
        object.__setattr__(instance, "__pydantic_extra__", values or None)
        return instance

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: dict[str, Any] | None = None,
    ) -> bool | None:
        try:
            module_globals = vars(sys.modules[cls.__module__])
            namespace = dict(module_globals)
            namespace.update(_types_namespace or {})
            namespace[cls.__name__] = cls
            for field in cls.model_fields.values():
                if isinstance(field.annotation, str):
                    field.annotation = eval(field.annotation, namespace, namespace)
            cls.__simple_model_plan__ = _simple_model_plan(
                cls.model_fields,
                cls.model_config,
                cls.__field_validators__,
                cls.__model_validators__,
            )
            return True
        except (NameError, TypeError, SyntaxError):
            if raise_errors:
                raise
            return False

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        context: Any = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | str = True,
        fallback: Any = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        aliases = bool(by_alias)
        included = set(include) if include is not None and not isinstance(include, dict) else None
        excluded = set(exclude) if exclude is not None and not isinstance(exclude, dict) else set()
        output: dict[str, Any] = {}
        for name, field in self.model_fields.items():
            if name not in self.__dict__ or field.exclude:
                continue
            if included is not None and name not in included:
                continue
            if name in excluded:
                continue
            if exclude_unset and name not in self.__pydantic_fields_set__:
                continue
            value = self.__dict__[name]
            if exclude_none and value is None:
                continue
            if (
                exclude_defaults
                and field.default is not PydanticUndefined
                and value == field.default
            ):
                continue
            if (
                exclude_defaults
                and field.default_factory is not None
                and value == field.default_factory()
            ):
                continue
            for serializer_name, fields, serializer_mode, when_used in self.__field_serializers__:
                if name in fields or "*" in fields:
                    if when_used == "json" and mode != "json":
                        continue
                    value = _invoke(
                        getattr(self, serializer_name),
                        value,
                        ValidationState(mode=mode, context=context, config=self.model_config),
                    )
            key = (
                field.serialization_alias
                or field.alias
                or name
                if aliases
                else name
            )
            output[key] = dump_value(value, mode, aliases)
        if self.__pydantic_extra__:
            for key, value in self.__pydantic_extra__.items():
                if key not in excluded:
                    output[key] = dump_value(value, mode, aliases)
        if not exclude_computed_fields:
            for name, metadata in self.__computed_fields__.items():
                if name in excluded or (included is not None and name not in included):
                    continue
                key = metadata.get("alias") if aliases and metadata.get("alias") else name
                output[key] = dump_value(getattr(self, name), mode, aliases)
        return output

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include: Any = None,
        exclude: Any = None,
        context: Any = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | str = True,
        fallback: Any = None,
        serialize_as_any: bool = False,
    ) -> str:
        data = self.model_dump(
            mode="json",
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_computed_fields=exclude_computed_fields,
            round_trip=round_trip,
        )
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            data, indent=indent, ensure_ascii=ensure_ascii, separators=separators
        )

    def model_copy(
        self, *, update: dict[str, Any] | None = None, deep: bool = False
    ) -> "BaseModel":
        instance = deepcopy(self) if deep else copy(self)
        object.__setattr__(instance, "__dict__", deepcopy(self.__dict__) if deep else dict(self.__dict__))
        if update:
            instance.__dict__.update(update)
            instance.__pydantic_fields_set__ = {
                *self.__pydantic_fields_set__,
                *update,
            }
        return instance

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: Any = None,
        mode: str = "validation",
        *,
        union_format: str = "any_of",
    ) -> dict[str, Any]:
        properties = {}
        required = []
        for name, field in cls.model_fields.items():
            key = field.alias if by_alias and field.alias else name
            item = schema_for(field.annotation)
            if field.title is not None:
                item["title"] = field.title
            else:
                item.setdefault("title", name.replace("_", " ").title())
            if field.description is not None:
                item["description"] = field.description
            if field.default is not PydanticUndefined:
                item["default"] = dump_value(field.default, "json")
            elif field.default_factory is None:
                required.append(key)
            properties[key] = item
        result: dict[str, Any] = {
            "properties": properties,
            "title": cls.__name__,
            "type": "object",
        }
        if required:
            result["required"] = required
        if cls.model_config.get("extra") == "forbid":
            result["additionalProperties"] = False
        return result

    @property
    def model_fields_set(self) -> set[str]:
        return self.__pydantic_fields_set__

    @property
    def model_extra(self) -> dict[str, Any] | None:
        return self.__pydantic_extra__

    def __setattr__(self, name: str, value: Any) -> None:
        if self.model_config.get("frozen"):
            raise ValidationError(
                self.__class__.__name__,
                [issue("frozen_instance", (name,), value, msg="Instance is frozen")],
            )
        field = self.model_fields.get(name)
        if field is not None and field.frozen:
            raise ValidationError(
                self.__class__.__name__,
                [issue("frozen_field", (name,), value, msg="Field is frozen")],
            )
        if field is not None and self.model_config.get("validate_assignment"):
            state = ValidationState(
                strict=self.model_config.get("strict", False),
                config=self.model_config,
                data=self.__dict__,
                field_name=name,
            )
            try:
                value = validate_value(field.annotation, value, state, (name,), field)
            except _ValidationFailure as exc:
                raise validation_error(self.__class__.__name__, exc) from exc
        object.__setattr__(self, name, value)
        if field is not None and hasattr(self, "__pydantic_fields_set__"):
            self.__pydantic_fields_set__.add(name)

    def __repr_args__(self):
        for name, field in self.model_fields.items():
            if field.repr and name in self.__dict__:
                yield name, self.__dict__[name]
        for name, metadata in self.__computed_fields__.items():
            if metadata.get("repr", True):
                yield name, getattr(self, name)

    def __repr__(self) -> str:
        args = " ".join(f"{name}={value!r}" for name, value in self.__repr_args__())
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        return " ".join(f"{name}={value!r}" for name, value in self.__repr_args__())

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.model_dump() == other.model_dump()
        )

    def __iter__(self):
        yield from self.model_dump().items()

    @classmethod
    def parse_obj(cls, obj: Any) -> "BaseModel":
        return cls.model_validate(obj)

    @classmethod
    def parse_raw(cls, data: str | bytes, **kwargs: Any) -> "BaseModel":
        return cls.model_validate_json(data)

    def dict(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump(**kwargs)

    def json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    def copy(self, **kwargs: Any) -> "BaseModel":
        return self.model_copy(**kwargs)


def create_model(
    model_name: str,
    /,
    *,
    __config__: ConfigDict | None = None,
    __doc__: str | None = None,
    __base__: type[BaseModel] | tuple[type[BaseModel], ...] | None = None,
    __module__: str = __name__,
    __validators__: dict[str, Any] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any,
) -> type[BaseModel]:
    annotations = {}
    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        "__module__": __module__,
        "__doc__": __doc__,
    }
    if __config__ is not None:
        namespace["model_config"] = dict(__config__)
    if __validators__:
        namespace.update(__validators__)
    for name, definition in field_definitions.items():
        if isinstance(definition, tuple):
            annotation, default = definition
        else:
            annotation, default = definition, PydanticUndefined
        if default is Ellipsis:
            default = PydanticUndefined
        annotations[name] = annotation
        if default is not PydanticUndefined:
            namespace[name] = default
    bases = (
        __base__
        if isinstance(__base__, tuple)
        else (__base__ or BaseModel,)
    )
    return ModelMetaclass(model_name, bases, namespace, **(__cls_kwargs__ or {}))
