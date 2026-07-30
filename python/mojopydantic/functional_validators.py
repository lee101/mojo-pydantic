from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BeforeValidator:
    func: Callable[..., Any]


@dataclass(frozen=True)
class AfterValidator:
    func: Callable[..., Any]


@dataclass(frozen=True)
class PlainValidator:
    func: Callable[..., Any]


@dataclass(frozen=True)
class WrapValidator:
    func: Callable[..., Any]


@dataclass(frozen=True)
class ValidationInfo:
    context: Any = None
    config: dict[str, Any] | None = None
    mode: str = "python"
    data: dict[str, Any] | None = None
    field_name: str | None = None


def _tag(target: Any, name: str, value: Any) -> Any:
    function = target.__func__ if isinstance(target, (classmethod, staticmethod)) else target
    setattr(function, name, value)
    return target


def field_validator(
    *fields: str,
    mode: str = "after",
    check_fields: bool | None = None,
    json_schema_input_type: Any = None,
) -> Callable[[Any], Any]:
    if mode not in {"before", "after", "plain", "wrap"}:
        raise ValueError(f"unsupported validator mode: {mode}")

    def decorator(function: Any) -> Any:
        return _tag(function, "__mp_field_validator__", (fields, mode, check_fields))

    return decorator


def model_validator(*, mode: str) -> Callable[[Any], Any]:
    if mode not in {"before", "after", "wrap"}:
        raise ValueError(f"unsupported validator mode: {mode}")

    def decorator(function: Any) -> Any:
        return _tag(function, "__mp_model_validator__", mode)

    return decorator


def field_serializer(
    *fields: str,
    mode: str = "plain",
    return_type: Any = None,
    when_used: str = "always",
    check_fields: bool | None = None,
) -> Callable[[Any], Any]:
    def decorator(function: Any) -> Any:
        return _tag(function, "__mp_field_serializer__", (fields, mode, when_used))

    return decorator


def computed_field(
    func: Any = None,
    *,
    alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    repr: bool | None = None,
    return_type: Any = None,
) -> Any:
    def decorate(target: Any) -> Any:
        prop = target if isinstance(target, property) else property(target)
        setattr(prop.fget, "__mp_computed_field__", {"alias": alias, "repr": repr})
        return prop

    return decorate(func) if func is not None else decorate


validator = field_validator
root_validator = model_validator
