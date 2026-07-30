from __future__ import annotations

import json
from typing import Any, get_args, get_origin

import numpy as np

from . import _lib
from ._core import (
    _ValidationFailure,
    ValidationState,
    dump_value,
    schema_for,
    validate_value,
    validation_error,
)
from .errors import ValidationError, issue


class TypeAdapter:
    def __init__(
        self,
        type: Any,
        *,
        config: dict[str, Any] | None = None,
        _parent_depth: int = 2,
        module: str | None = None,
    ) -> None:
        self._type = type
        self._config = dict(config or {})

    def validate_python(
        self,
        object: Any,
        /,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: Any = None,
        experimental_allow_partial: bool | str = False,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Any:
        state = ValidationState(
            strict=self._config.get("strict", False) if strict is None else strict,
            context=context,
            config=self._config,
        )
        try:
            return validate_value(self._type, object, state)
        except _ValidationFailure as exc:
            raise validation_error(self._display_name(), exc) from exc

    def validate_json(
        self,
        data: str | bytes | bytearray,
        /,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any = None,
        experimental_allow_partial: bool | str = False,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Any:
        raw = data.encode() if isinstance(data, str) else bytes(data)
        effective_strict = (
            self._config.get("strict", False) if strict is None else strict
        )
        origin = get_origin(self._type)
        args = get_args(self._type)
        if origin is list and len(args) == 1 and args[0] in (int, float, bool):
            parsed = _lib.typed_json_array(raw, args[0], effective_strict)
            if parsed is not None:
                return parsed
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValidationError(
                self._display_name(),
                [issue("json_invalid", (), data, ctx={"error": str(exc)})],
            ) from exc
        state = ValidationState(
            strict=effective_strict,
            context=context,
            mode="json",
            config=self._config,
        )
        try:
            return validate_value(self._type, value, state)
        except _ValidationFailure as exc:
            raise validation_error(self._display_name(), exc) from exc

    def validate_strings(
        self,
        obj: Any,
        /,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any = None,
        experimental_allow_partial: bool | str = False,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Any:
        return self.validate_python(obj, strict=strict, context=context)

    def validate_json_array(
        self,
        data: str | bytes | bytearray,
        /,
        *,
        strict: bool | None = None,
    ) -> np.ndarray:
        """Validate a homogeneous numeric JSON list into an unboxed NumPy array."""
        raw = data.encode() if isinstance(data, str) else bytes(data)
        effective_strict = (
            self._config.get("strict", False) if strict is None else strict
        )
        origin = get_origin(self._type)
        args = get_args(self._type)
        if origin is list and len(args) == 1 and args[0] in (int, float, bool):
            parsed = _lib.typed_json_ndarray(raw, args[0], effective_strict)
            if parsed is not None:
                return parsed
        return np.asarray(self.validate_json(raw, strict=effective_strict))

    def get_default_value(
        self, *, strict: bool | None = None, context: Any = None
    ) -> None:
        return None

    def dump_python(
        self,
        instance: Any,
        /,
        *,
        mode: str = "python",
        include: Any = None,
        exclude: Any = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | str = True,
        fallback: Any = None,
        serialize_as_any: bool = False,
        context: Any = None,
    ) -> Any:
        return dump_value(instance, mode, bool(by_alias))

    def dump_json(
        self,
        instance: Any,
        /,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include: Any = None,
        exclude: Any = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | str = True,
        fallback: Any = None,
        serialize_as_any: bool = False,
        context: Any = None,
    ) -> bytes:
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.dump_python(instance, mode="json", by_alias=by_alias),
            indent=indent,
            ensure_ascii=ensure_ascii,
            separators=separators,
        ).encode()

    def json_schema(
        self,
        *,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        union_format: str = "any_of",
        schema_generator: Any = None,
        mode: str = "validation",
    ) -> dict[str, Any]:
        return schema_for(self._type)

    def _display_name(self) -> str:
        return getattr(self._type, "__name__", str(self._type))
