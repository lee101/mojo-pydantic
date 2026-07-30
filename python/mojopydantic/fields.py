from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable


class _Undefined:
    def __repr__(self) -> str:
        return "PydanticUndefined"


PydanticUndefined = _Undefined()


@dataclass
class AliasPath:
    path: tuple[str | int, ...]

    def __init__(self, first_arg: str, *args: str | int) -> None:
        self.path = (first_arg, *args)

    def search_dict_for_path(self, data: Any) -> Any:
        current = data
        try:
            for part in self.path:
                current = current[part]
            return current
        except (KeyError, IndexError, TypeError):
            return PydanticUndefined


@dataclass
class AliasChoices:
    choices: tuple[str | AliasPath, ...]

    def __init__(self, first_choice: str | AliasPath, *choices: str | AliasPath) -> None:
        self.choices = (first_choice, *choices)


@dataclass
class FieldInfo:
    annotation: Any = None
    default: Any = PydanticUndefined
    default_factory: Callable[[], Any] | None = None
    alias: str | None = None
    validation_alias: str | AliasPath | AliasChoices | None = None
    serialization_alias: str | None = None
    title: str | None = None
    description: str | None = None
    examples: list[Any] | None = None
    exclude: bool | None = None
    frozen: bool | None = None
    repr: bool = True
    validate_default: bool | None = None
    strict: bool | None = None
    gt: Any = None
    ge: Any = None
    lt: Any = None
    le: Any = None
    multiple_of: Any = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    metadata: list[Any] = field(default_factory=list)

    def is_required(self) -> bool:
        return self.default is PydanticUndefined and self.default_factory is None

    def get_default(self, *, call_default_factory: bool = False) -> Any:
        if self.default_factory is not None:
            return self.default_factory() if call_default_factory else None
        return self.default

    def merged(self, other: "FieldInfo") -> "FieldInfo":
        result = copy(self)
        for name in self.__dataclass_fields__:
            value = getattr(other, name)
            if name == "metadata":
                result.metadata = [*self.metadata, *value]
            elif value is not None and not (name == "default" and value is PydanticUndefined):
                setattr(result, name, value)
        return result


def Field(
    default: Any = PydanticUndefined,
    *,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    alias_priority: int | None = None,
    validation_alias: str | AliasPath | AliasChoices | None = None,
    serialization_alias: str | None = None,
    title: str | None = None,
    description: str | None = None,
    examples: list[Any] | None = None,
    exclude: bool | None = None,
    discriminator: str | None = None,
    deprecated: bool | str | None = None,
    frozen: bool | None = None,
    validate_default: bool | None = None,
    repr: bool = True,
    init: bool | None = None,
    init_var: bool | None = None,
    kw_only: bool | None = None,
    pattern: str | None = None,
    strict: bool | None = None,
    coerce_numbers_to_str: bool | None = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    multiple_of: Any = None,
    allow_inf_nan: bool | None = None,
    max_digits: int | None = None,
    decimal_places: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    union_mode: str = "smart",
    fail_fast: bool | None = None,
    **extra: Any,
) -> FieldInfo:
    return FieldInfo(
        default=default,
        default_factory=default_factory,
        alias=alias,
        validation_alias=validation_alias,
        serialization_alias=serialization_alias,
        title=title,
        description=description,
        examples=examples,
        exclude=exclude,
        frozen=frozen,
        validate_default=validate_default,
        repr=repr,
        pattern=pattern,
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        min_length=min_length,
        max_length=max_length,
        metadata=[value for value in (allow_inf_nan, max_digits, decimal_places) if value is not None],
    )
