from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any

from .fields import Field
from .functional_validators import AfterValidator


@dataclass(frozen=True)
class Strict:
    strict: bool = True


@dataclass(frozen=True)
class StringConstraints:
    strip_whitespace: bool | None = None
    to_upper: bool | None = None
    to_lower: bool | None = None
    strict: bool | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


@dataclass(frozen=True)
class AllowInfNan:
    allow_inf_nan: bool = True


StrictBool = Annotated[bool, Strict()]
StrictBytes = Annotated[bytes, Strict()]
StrictFloat = Annotated[float, Strict()]
StrictInt = Annotated[int, Strict()]
StrictStr = Annotated[str, Strict()]
PositiveInt = Annotated[int, Field(gt=0)]
NegativeInt = Annotated[int, Field(lt=0)]
NonPositiveInt = Annotated[int, Field(le=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NegativeFloat = Annotated[float, Field(lt=0)]
NonPositiveFloat = Annotated[float, Field(le=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
FiniteFloat = Annotated[float, AllowInfNan(False)]


def conint(
    *,
    strict: bool | None = None,
    gt: int | None = None,
    ge: int | None = None,
    lt: int | None = None,
    le: int | None = None,
    multiple_of: int | None = None,
) -> Any:
    return Annotated[
        int,
        Field(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of),
    ]


def confloat(
    *,
    strict: bool | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    allow_inf_nan: bool | None = None,
) -> Any:
    return Annotated[
        float,
        Field(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of),
    ]


def constr(
    *,
    strip_whitespace: bool | None = None,
    to_upper: bool | None = None,
    to_lower: bool | None = None,
    strict: bool | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
) -> Any:
    def transform(value: str) -> str:
        if strip_whitespace:
            value = value.strip()
        if to_upper:
            value = value.upper()
        if to_lower:
            value = value.lower()
        return value

    return Annotated[
        str,
        Strict(strict) if strict else StringConstraints(),
        Field(min_length=min_length, max_length=max_length, pattern=pattern),
        AfterValidator(transform),
    ]


def conbytes(
    *, min_length: int | None = None, max_length: int | None = None, strict: bool | None = None
) -> Any:
    return Annotated[bytes, Field(min_length=min_length, max_length=max_length, strict=strict)]


def conlist(
    item_type: Any,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    unique_items: None = None,
) -> Any:
    return Annotated[list[item_type], Field(min_length=min_length, max_length=max_length)]


def conset(
    item_type: Any, *, min_length: int | None = None, max_length: int | None = None
) -> Any:
    return Annotated[set[item_type], Field(min_length=min_length, max_length=max_length)]


def confrozenset(
    item_type: Any, *, min_length: int | None = None, max_length: int | None = None
) -> Any:
    return Annotated[frozenset[item_type], Field(min_length=min_length, max_length=max_length)]


def condecimal(
    *,
    strict: bool | None = None,
    gt: Decimal | int | None = None,
    ge: Decimal | int | None = None,
    lt: Decimal | int | None = None,
    le: Decimal | int | None = None,
    multiple_of: Decimal | int | None = None,
    max_digits: int | None = None,
    decimal_places: int | None = None,
    allow_inf_nan: bool | None = None,
) -> Any:
    return Annotated[
        Decimal,
        Field(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of),
    ]


def _email(value: str) -> str:
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value) is None:
        raise ValueError("value is not a valid email address")
    return value


EmailStr = Annotated[str, AfterValidator(_email)]


class SecretStr:
    def __init__(self, secret_value: str) -> None:
        self._secret_value = secret_value

    def get_secret_value(self) -> str:
        return self._secret_value

    def __str__(self) -> str:
        return "**********" if self._secret_value else ""

    def __repr__(self) -> str:
        return f"SecretStr({str(self)!r})"
