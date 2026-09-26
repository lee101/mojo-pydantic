from __future__ import annotations

import json
import ctypes
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

import pydantic as pd
import pytest
import numpy as np

import mojopydantic as mp
from mojopydantic import _lib


def error_signature(exc: Exception):
    return [(item["type"], item["loc"]) for item in exc.errors()]


@pytest.mark.parametrize(
    ("annotation", "values"),
    [
        (int, [0, True, 1.0, " 42 ", b"-7"]),
        (float, [0, True, 1.5, " 2.25 ", b"-7e2"]),
        (bool, [True, 0, 1.0, "yes", "OFF", b"false"]),
        (str, ["x", b"bytes"]),
        (bytes, [b"x", "text"]),
    ],
)
def test_scalar_coercion_matches_pydantic(annotation, values):
    ours, theirs = mp.TypeAdapter(annotation), pd.TypeAdapter(annotation)
    assert [ours.validate_python(value) for value in values] == [
        theirs.validate_python(value) for value in values
    ]


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        (int, 1.5),
        (int, "abc"),
        (float, "abc"),
        (bool, 2),
        (bool, "maybe"),
        (str, 1),
        (bytes, object()),
    ],
)
def test_scalar_error_types_match_pydantic(annotation, value):
    with pytest.raises(mp.ValidationError) as ours:
        mp.TypeAdapter(annotation).validate_python(value)
    with pytest.raises(pd.ValidationError) as theirs:
        pd.TypeAdapter(annotation).validate_python(value)
    assert error_signature(ours.value) == error_signature(theirs.value)


def test_primitive_list_fast_path_accumulates_errors():
    value = ["1", "bad", "3", "also-bad"]
    with pytest.raises(mp.ValidationError) as ours:
        mp.TypeAdapter(list[int]).validate_python(value)
    with pytest.raises(pd.ValidationError) as theirs:
        pd.TypeAdapter(list[int]).validate_python(value)
    assert error_signature(ours.value) == error_signature(theirs.value)


def test_string_integer_list_fast_path_parity():
    values = [" 1 ", "+2", "-3", "004"]
    assert mp.TypeAdapter(list[int]).validate_python(values) == pd.TypeAdapter(
        list[int]
    ).validate_python(values)


@pytest.mark.parametrize(
    ("annotation", "payload"),
    [
        (list[int], b'[1, "2", 3.0, true, -4, 1e2]'),
        (list[float], b'[1, "2.5", 3.0, true, -4e-2]'),
        (list[bool], b'[true, "yes", 0, 1.0, "OFF"]'),
        (list[int], b"[]"),
    ],
)
def test_native_json_arrays_match_pydantic(annotation, payload):
    assert mp.TypeAdapter(annotation).validate_json(payload) == pd.TypeAdapter(
        annotation
    ).validate_json(payload)


@pytest.mark.parametrize(
    ("annotation", "payload", "dtype"),
    [
        (list[int], b'[1, "2", 3.0, true]', "int64"),
        (list[float], b'[1, "2.5", -3e2, true]', "float64"),
        (list[bool], b'[true, "yes", 0, 1.0]', "bool"),
    ],
)
def test_unboxed_json_array_extension(annotation, payload, dtype):
    ours = mp.TypeAdapter(annotation).validate_json_array(payload)
    theirs = np.asarray(pd.TypeAdapter(annotation).validate_json(payload), dtype=dtype)
    assert ours.dtype == theirs.dtype
    assert np.array_equal(ours, theirs)


def test_native_abi_rejects_invalid_addresses_and_sizes():
    library = _lib.lib()
    source = ctypes.create_string_buffer(b"[]")
    destination = (ctypes.c_int64 * 1)()
    source_address = ctypes.addressof(source)
    destination_address = ctypes.addressof(destination)
    for name in (
        "mp_json_i64_array",
        "mp_json_f64_array",
        "mp_json_bool_array",
    ):
        function = getattr(library, name)
        assert function(0, 2, destination_address, 1, 0) == -1
        assert function(source_address, 2, 0, 1, 0) == -1
        assert function(source_address, 0, destination_address, 1, 0) == -1
        assert function(source_address, 2, destination_address, -1, 0) == -1


def test_float_array_simd_tail_across_delimiter_widths():
    adapter = mp.TypeAdapter(list[float])
    for size in (99_999, 100_003):
        expected = np.arange(size, dtype=np.float64) * 0.125 - 10_000
        payload = json.dumps(expected.tolist(), separators=(",", ":")).encode()
        ours = adapter.validate_json_array(payload)
        assert np.array_equal(ours, expected)


def test_float_array_failure_matches_upstream_error():
    values = [index * 0.125 for index in range(100_003)]
    values[50_001] = "not-a-float"
    payload = json.dumps(values, separators=(",", ":")).encode()
    with pytest.raises(mp.ValidationError) as ours:
        mp.TypeAdapter(list[float]).validate_json_array(payload)
    with pytest.raises(pd.ValidationError) as theirs:
        pd.TypeAdapter(list[float]).validate_json(payload)
    assert error_signature(ours.value) == error_signature(theirs.value)


@pytest.mark.parametrize(
    ("annotation", "payload"),
    [
        (list[int], b'[1, "nan"]'),
        (list[int], b"[1, 2.5]"),
        (list[float], b'[1, "nope"]'),
        (list[bool], b'[true, 2, "maybe"]'),
    ],
)
def test_native_json_array_fallback_errors_match(annotation, payload):
    with pytest.raises(mp.ValidationError) as ours:
        mp.TypeAdapter(annotation).validate_json(payload)
    with pytest.raises(pd.ValidationError) as theirs:
        pd.TypeAdapter(annotation).validate_json(payload)
    assert error_signature(ours.value) == error_signature(theirs.value)


def test_large_native_integer_falls_back_without_losing_precision():
    payload = b"[9223372036854775808, -9223372036854775809]"
    assert mp.TypeAdapter(list[int]).validate_json(payload) == pd.TypeAdapter(
        list[int]
    ).validate_json(payload)


def test_native_checked_integer_boundaries():
    payload = b"[-9007199254740991,0,9007199254740991]"
    parsed = _lib.typed_json_ndarray(payload, int, False)
    assert parsed is not None
    assert parsed.tolist() == [-9007199254740991, 0, 9007199254740991]
    assert _lib.typed_json_ndarray(b"[9007199254740992]", int, False) is None


@pytest.mark.parametrize("payload", [b"[+1]", b"[01]", b"[1.]", b"[.1]", b"[1,]"])
def test_invalid_json_is_not_accepted_by_native_fast_path(payload):
    with pytest.raises(mp.ValidationError) as ours:
        mp.TypeAdapter(list[float]).validate_json(payload)
    with pytest.raises(pd.ValidationError) as theirs:
        pd.TypeAdapter(list[float]).validate_json(payload)
    assert ours.value.errors()[0]["type"] == theirs.value.errors()[0]["type"] == "json_invalid"


def test_strict_typed_json_arrays_match():
    good = {
        list[int]: b"[1, -2]",
        list[float]: b"[1, 2.5]",
        list[bool]: b"[true, false]",
    }
    for annotation, payload in good.items():
        assert mp.TypeAdapter(annotation).validate_json(
            payload, strict=True
        ) == pd.TypeAdapter(annotation).validate_json(payload, strict=True)
    for annotation, payload in (
        (list[int], b"[1.0]"),
        (list[int], b'["1"]'),
        (list[float], b'["1"]'),
        (list[bool], b"[1]"),
    ):
        with pytest.raises(mp.ValidationError) as ours:
            mp.TypeAdapter(annotation).validate_json(payload, strict=True)
        with pytest.raises(pd.ValidationError) as theirs:
            pd.TypeAdapter(annotation).validate_json(payload, strict=True)
        assert error_signature(ours.value) == error_signature(theirs.value)


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class MPAddress(mp.BaseModel):
    city: str
    postal_code: Annotated[int, mp.Field(ge=10_000, le=99_999)]


class PDAddress(pd.BaseModel):
    city: str
    postal_code: Annotated[int, pd.Field(ge=10_000, le=99_999)]


class MPUser(mp.BaseModel):
    model_config = mp.ConfigDict(extra="forbid")
    id: int
    name: str = mp.Field(alias="fullName", min_length=2)
    active: bool = True
    tags: list[str] = mp.Field(default_factory=list)
    address: MPAddress | None = None
    role: Literal["admin", "user"] = "user"
    color: Color = Color.RED


class PDUser(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")
    id: int
    name: str = pd.Field(alias="fullName", min_length=2)
    active: bool = True
    tags: list[str] = pd.Field(default_factory=list)
    address: PDAddress | None = None
    role: Literal["admin", "user"] = "user"
    color: Color = Color.RED


def user_input():
    return {
        "id": "42",
        "fullName": "Ada",
        "active": "yes",
        "tags": ("math", "mojo"),
        "address": {"city": "London", "postal_code": "12345"},
        "role": "admin",
        "color": "blue",
    }


def test_nested_model_validation_and_dump_parity():
    ours = MPUser.model_validate(user_input())
    theirs = PDUser.model_validate(user_input())
    assert ours.model_dump() == theirs.model_dump()
    assert ours.model_dump(by_alias=True) == theirs.model_dump(by_alias=True)
    assert json.loads(ours.model_dump_json()) == json.loads(theirs.model_dump_json())
    assert ours.model_fields_set == theirs.model_fields_set


def test_required_primitive_model_fast_plan_parity():
    class Ours(mp.BaseModel):
        identifier: int
        score: float
        enabled: bool
        tags: list[int]

    class Theirs(pd.BaseModel):
        identifier: int
        score: float
        enabled: bool
        tags: list[int]

    value = {
        "identifier": "42",
        "score": "2.5",
        "enabled": "yes",
        "tags": ["1", 2, 3.0],
    }
    assert Ours.model_validate(value).model_dump() == Theirs.model_validate(
        value
    ).model_dump()
    invalid = {**value, "identifier": "bad", "tags": ["bad", "also-bad"]}
    with pytest.raises(mp.ValidationError) as ours:
        Ours.model_validate(invalid)
    with pytest.raises(pd.ValidationError) as theirs:
        Theirs.model_validate(invalid)
    assert error_signature(ours.value) == error_signature(theirs.value)


def test_model_validate_json_parity():
    payload = json.dumps(user_input(), default=list)
    assert MPUser.model_validate_json(payload).model_dump() == PDUser.model_validate_json(
        payload
    ).model_dump()


def test_model_collects_field_and_extra_errors_like_pydantic():
    data = {"id": "wrong", "fullName": "x", "role": "root", "extra": 1}
    with pytest.raises(mp.ValidationError) as ours:
        MPUser.model_validate(data)
    with pytest.raises(pd.ValidationError) as theirs:
        PDUser.model_validate(data)
    assert error_signature(ours.value) == error_signature(theirs.value)
    assert ours.value.error_count() == 4
    assert json.loads(ours.value.json())


def test_defaults_are_per_instance():
    first, second = MPUser(id=1, fullName="one"), MPUser(id=2, fullName="two")
    first.tags.append("x")
    assert second.tags == []


def test_alias_choices_and_alias_path():
    class Ours(mp.BaseModel):
        first: str = mp.Field(validation_alias=mp.AliasChoices("first", "fname"))
        code: int = mp.Field(validation_alias=mp.AliasPath("meta", "code"))

    class Theirs(pd.BaseModel):
        first: str = pd.Field(validation_alias=pd.AliasChoices("first", "fname"))
        code: int = pd.Field(validation_alias=pd.AliasPath("meta", "code"))

    data = {"fname": "Ada", "meta": {"code": "7"}}
    assert Ours.model_validate(data).model_dump() == Theirs.model_validate(
        data
    ).model_dump()


def test_from_attributes_parity():
    class Source:
        id = "4"
        name = "item"

    class Ours(mp.BaseModel):
        model_config = mp.ConfigDict(from_attributes=True)
        id: int
        name: str

    class Theirs(pd.BaseModel):
        model_config = pd.ConfigDict(from_attributes=True)
        id: int
        name: str

    assert Ours.model_validate(Source()).model_dump() == Theirs.model_validate(
        Source()
    ).model_dump()


def test_validate_assignment():
    class Ours(mp.BaseModel):
        model_config = mp.ConfigDict(validate_assignment=True)
        count: int

    ours = Ours(count=1)
    ours.count = "2"
    assert ours.count == 2
    with pytest.raises(mp.ValidationError):
        ours.count = "bad"


def test_frozen_model_and_field():
    class Frozen(mp.BaseModel):
        model_config = mp.ConfigDict(frozen=True)
        value: int

    with pytest.raises(mp.ValidationError):
        Frozen(value=1).value = 2

    class OneField(mp.BaseModel):
        value: int = mp.Field(frozen=True)

    with pytest.raises(mp.ValidationError):
        OneField(value=1).value = 2


def test_field_validators_match():
    class Ours(mp.BaseModel):
        name: str
        score: int

        @mp.field_validator("name", mode="before")
        @classmethod
        def strip_name(cls, value):
            return value.strip()

        @mp.field_validator("score")
        @classmethod
        def cap_score(cls, value):
            if value > 100:
                raise ValueError("too high")
            return value

    class Theirs(pd.BaseModel):
        name: str
        score: int

        @pd.field_validator("name", mode="before")
        @classmethod
        def strip_name(cls, value):
            return value.strip()

        @pd.field_validator("score")
        @classmethod
        def cap_score(cls, value):
            if value > 100:
                raise ValueError("too high")
            return value

    data = {"name": " Ada ", "score": "99"}
    assert Ours(**data).model_dump() == Theirs(**data).model_dump()
    with pytest.raises(mp.ValidationError) as ours:
        Ours(name="x", score=101)
    with pytest.raises(pd.ValidationError) as theirs:
        Theirs(name="x", score=101)
    assert error_signature(ours.value) == error_signature(theirs.value)


def test_model_validators_match():
    class Ours(mp.BaseModel):
        low: int
        high: int

        @mp.model_validator(mode="before")
        @classmethod
        def split_range(cls, value):
            if "range" in value:
                low, high = value["range"].split(":")
                return {"low": low, "high": high}
            return value

        @mp.model_validator(mode="after")
        def ordered(self):
            if self.low > self.high:
                raise ValueError("range is reversed")
            return self

    class Theirs(pd.BaseModel):
        low: int
        high: int

        @pd.model_validator(mode="before")
        @classmethod
        def split_range(cls, value):
            if "range" in value:
                low, high = value["range"].split(":")
                return {"low": low, "high": high}
            return value

        @pd.model_validator(mode="after")
        def ordered(self):
            if self.low > self.high:
                raise ValueError("range is reversed")
            return self

    assert Ours(range="1:5").model_dump() == Theirs(range="1:5").model_dump()
    with pytest.raises(mp.ValidationError) as ours:
        Ours(low=5, high=1)
    with pytest.raises(pd.ValidationError) as theirs:
        Theirs(low=5, high=1)
    assert error_signature(ours.value) == error_signature(theirs.value)


def test_annotated_functional_validators():
    def double(value):
        return value * 2

    OursType = Annotated[int, mp.BeforeValidator(int), mp.AfterValidator(double)]
    TheirsType = Annotated[int, pd.BeforeValidator(int), pd.AfterValidator(double)]
    assert mp.TypeAdapter(OursType).validate_python("4") == pd.TypeAdapter(
        TheirsType
    ).validate_python("4")


def test_field_serializer_and_computed_field():
    class Item(mp.BaseModel):
        price: Decimal
        quantity: int

        @mp.field_serializer("price")
        def serialize_price(self, value):
            return f"{value:.2f}"

        @mp.computed_field
        @property
        def total(self) -> Decimal:
            return self.price * self.quantity

    item = Item(price="2.5", quantity=3)
    assert item.model_dump() == {"price": "2.50", "quantity": 3, "total": Decimal("7.5")}
    assert json.loads(item.model_dump_json()) == {
        "price": "2.50",
        "quantity": 3,
        "total": "7.5",
    }


def test_dump_json_extended_types_matches():
    class Ours(mp.BaseModel):
        when: datetime
        day: date
        identifier: UUID
        amount: Decimal
        color: Color

    class Theirs(pd.BaseModel):
        when: datetime
        day: date
        identifier: UUID
        amount: Decimal
        color: Color

    data = {
        "when": "2025-02-03T04:05:06",
        "day": "2025-02-03",
        "identifier": "12345678-1234-5678-1234-567812345678",
        "amount": "12.50",
        "color": "red",
    }
    assert json.loads(Ours(**data).model_dump_json()) == json.loads(
        Theirs(**data).model_dump_json()
    )


def test_container_union_literal_and_enum_parity():
    annotation = dict[str, tuple[int, bool | None, Literal["x"], Color]]
    value = {"row": ["4", "yes", "x", "blue"]}
    assert mp.TypeAdapter(annotation).validate_python(value) == pd.TypeAdapter(
        annotation
    ).validate_python(value)


def test_constraint_factories():
    Ours = mp.TypeAdapter(mp.conlist(mp.conint(gt=0), min_length=2, max_length=3))
    Theirs = pd.TypeAdapter(pd.conlist(pd.conint(gt=0), min_length=2, max_length=3))
    assert Ours.validate_python(["1", 2]) == Theirs.validate_python(["1", 2])
    for value in ([1], [1, -2]):
        with pytest.raises(mp.ValidationError):
            Ours.validate_python(value)
        with pytest.raises(pd.ValidationError):
            Theirs.validate_python(value)


def test_context_reaches_validator():
    class Model(mp.BaseModel):
        value: int

        @mp.field_validator("value")
        @classmethod
        def multiply(cls, value, info: mp.ValidationInfo):
            return value * info.context["factor"]

    assert Model.model_validate({"value": 3}, context={"factor": 4}).value == 12


def test_create_model_root_model_construct_copy():
    Dynamic = mp.create_model("Dynamic", value=(int, ...), label=(str, "x"))
    item = Dynamic(value="3")
    assert item.model_dump() == {"value": 3, "label": "x"}
    assert item.model_copy(update={"value": 4}).value == 4
    assert Dynamic.model_construct(value="not validated").value == "not validated"
    IntList = mp.RootModel[list[int]]
    root = IntList.model_validate_json('[1, "2", 3.0]')
    assert root.root == [1, 2, 3]
    assert root.model_dump() == [1, 2, 3]


def test_model_json_schema_has_pydantic_shape():
    ours, theirs = MPUser.model_json_schema(), PDUser.model_json_schema()
    assert ours["type"] == theirs["type"] == "object"
    assert ours["required"] == theirs["required"]
    assert set(ours["properties"]) == set(theirs["properties"])
    assert ours["additionalProperties"] is theirs["additionalProperties"] is False
    assert ours["properties"]["id"]["type"] == theirs["properties"]["id"]["type"]


def test_model_dump_exclusion_options():
    item = MPUser(id=1, fullName="Ada")
    assert item.model_dump(exclude_unset=True) == {"id": 1, "name": "Ada"}
    assert item.model_dump(exclude_defaults=True) == {"id": 1, "name": "Ada"}
    assert item.model_dump(include={"id"}) == {"id": 1}
    assert "active" not in item.model_dump(exclude={"active"})


def test_repr_and_equality():
    first = MPUser(id=1, fullName="Ada")
    second = MPUser(id=1, fullName="Ada")
    assert first == second
    assert "id=1" in repr(first)
    assert dict(first)["name"] == "Ada"


def test_strict_json_representation_types_match():
    class Ours(mp.BaseModel):
        day: date
        identifier: UUID
        payload: bytes

    class Theirs(pd.BaseModel):
        day: date
        identifier: UUID
        payload: bytes

    raw = (
        b'{"day":"2025-01-02","identifier":'
        b'"12345678-1234-5678-1234-567812345678","payload":"abc"}'
    )
    assert Ours.model_validate_json(raw, strict=True).model_dump() == (
        Theirs.model_validate_json(raw, strict=True).model_dump()
    )


def test_string_constraints_metadata_matches():
    OursType = Annotated[
        str,
        mp.StringConstraints(strip_whitespace=True, to_lower=True, min_length=2),
    ]
    TheirsType = Annotated[
        str,
        pd.StringConstraints(strip_whitespace=True, to_lower=True, min_length=2),
    ]
    assert mp.TypeAdapter(OursType).validate_python("  ADA  ") == pd.TypeAdapter(
        TheirsType
    ).validate_python("  ADA  ")


def test_finite_float_rejects_special_values():
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(mp.ValidationError):
            mp.TypeAdapter(mp.FiniteFloat).validate_python(value)
        with pytest.raises(pd.ValidationError):
            pd.TypeAdapter(pd.FiniteFloat).validate_python(value)


def test_secret_string_validation_and_json_masking():
    class Credentials(mp.BaseModel):
        password: mp.SecretStr

    credentials = Credentials(password="correct horse")
    assert credentials.password.get_secret_value() == "correct horse"
    assert json.loads(credentials.model_dump_json()) == {"password": "**********"}
