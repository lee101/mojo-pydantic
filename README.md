# mojo-pydantic

`mojo-pydantic` is a standalone Pydantic-style validation library with
compute-heavy typed JSON parsing implemented in
[Mojo](https://www.modular.com/mojo). Its Python package is named
`mojopydantic`, so it can be parity-tested alongside the real `pydantic`
package:

```python
from typing import Annotated

from mojopydantic import BaseModel, Field, TypeAdapter


class Event(BaseModel):
    id: int
    score: Annotated[float, Field(ge=0, le=100)]
    labels: list[str] = Field(default_factory=list)


event = Event.model_validate(
    {"id": "42", "score": "98.5", "labels": ("nightly", "mojo")}
)
assert event.id == 42
assert event.model_dump_json() == (
    '{"id":42,"score":98.5,"labels":["nightly","mojo"]}'
)

scores = TypeAdapter(list[float]).validate_json_array(b"[1, 2.5, -3e2]")
assert scores.tolist() == [1.0, 2.5, -300.0]
```

The regular `TypeAdapter.validate_json()` method returns normal Python lists,
as Pydantic does. `validate_json_array()` is an additional zero-boxing API for
homogeneous `list[int]`, `list[float]`, and `list[bool]` schemas; it returns a
contiguous NumPy array and is where the Mojo implementation is most useful.

## Covered subset

The implementation targets Pydantic 2's model and type-validation API:

| area | covered |
| --- | --- |
| Models | `BaseModel`, nested models, defaults and factories, aliases and alias paths, `extra="forbid"`, validation from attributes, assignment validation, frozen models/fields |
| Validation | `model_validate`, `model_validate_json`, `TypeAdapter.validate_python`, `TypeAdapter.validate_json`, strict JSON validation, accumulated `ValidationError` details |
| Types | `int`, `float`, `bool`, `str`, `bytes`, `Decimal`, UUID, date and datetime, enums, literals, optional/union types, `SecretStr` |
| Containers | typed lists, tuples, dictionaries, and tested nested combinations |
| Constraints | numeric `Field` bounds, minimum lengths, `Annotated`, `conint`, `conlist`, `StringConstraints`, and finite floats |
| Hooks | before/after field and model validators, before/after functional annotated validators, field serializers, computed fields, validation context |
| Output | Python and JSON model dumps, aliases and top-level inclusion/exclusion flags, basic JSON Schema, `model_copy`, `model_construct`, `create_model`, `RootModel` |
| Native extension | direct typed JSON arrays to Python lists or unboxed NumPy arrays |

This is a useful subset, not a claim to cover all of Pydantic. Not implemented
are `pydantic-core` custom schema hooks, discriminated unions, generic model
specialization beyond `RootModel`, dataclass decoration, recursive JSON Schema
`$defs`, URL/network families, settings management, plugin support, and all
of Pydantic's specialized string and decimal constraints. Nested
`include`/`exclude` selector dictionaries and assignment-time custom
validators are also outside the current subset.

The parity suite compares the behaviors in this table with upstream Pydantic;
a few extension-only behaviors, such as unboxed result dtypes and native ABI
input rejection, have direct regression tests instead.
Fallback from a Mojo typed-array parser is intentional: syntax errors,
special floating-point values, integers beyond binary64's exact range, and
unsupported coercions are reprocessed by the general validator rather than
accepted approximately.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi` installs the pinned Mojo nightly, Python, NumPy, pytest, and upstream
Pydantic used by the parity suite. The build produces
`dist/libmojo-pydantic.so`. The library can be loaded from another location by
setting `MOJOPYDANTIC_LIB=/absolute/path/to/libmojo-pydantic.so`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, and Pydantic 2.13.4. Times are the best of five
warmed runs.

For the three unboxed rows, the Pydantic reference is
`np.asarray(TypeAdapter(...).validate_json(...))`, so both sides return the
same contiguous dtype and values.

| case | mojopydantic | pydantic | relative |
| --- | ---: | ---: | ---: |
| `validate_json_array` int64 (500k) | 43.09 ms | 65.44 ms | 1.52x faster |
| `validate_json_array` float64 (300k) | 18.91 ms | 51.28 ms | 2.71x faster |
| `validate_json_array` bool (750k) | 17.69 ms | 127.72 ms | 7.22x faster |
| `validate_json list[int]` (500k) | 68.09 ms | 48.58 ms | 1.40x slower |
| `validate_json list[float]` (300k) | 43.23 ms | 36.40 ms | 1.19x slower |
| `validate_json list[bool]` (750k) | 23.57 ms | 67.17 ms | 2.85x faster |
| `validate_python list[int]` strings (200k) | 79.20 ms | 25.40 ms | 3.12x slower |
| `BaseModel.model_validate` (50k) | 333.49 ms | 156.01 ms | 2.14x slower |

Pydantic's Rust core is faster when the contract requires Python objects, and
the Python schema engine remains slower for ordinary models. Mojo wins for all
three large unboxed arrays in this measurement. Large float arrays use a SIMD
delimiter scan followed by eight independent CPU parser tasks; arrays below
100,000 elements or 512 KiB remain serial to avoid thread-launch overhead.
Float conversion still calls correctly rounded libc `strtod`.

No GPU path is provided. Typed JSON parsing is branch-heavy and moves at least
one input byte for very little arithmetic, well below the roughly two
flops-per-byte threshold where host/device transfer could pay off.

## How it works

`src/pydantic.mojo` is one compilation unit with three exported C-ABI
functions. Python calls it through `ctypes`; each byte buffer and output array
crosses the ABI as an integer address plus a length. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside the exported wrapper,
validates the JSON grammar and Pydantic coercion subset, then writes directly
into caller-owned contiguous `int64`, `float64`, or one-byte boolean memory.
There are no Mojo-side allocations and no ownership transfer.

The native parser returns a negative status for any case it cannot prove
correct. `TypeAdapter.validate_json()` then uses the general JSON and schema
engine, preserving error locations and coercion behavior. The unboxed
`validate_json_array()` path returns the NumPy allocation that Mojo filled,
while the drop-in path converts that same buffer to a Python list.

The Python layer builds field schemas from annotations, caches primitive model
validation plans, recursively validates complex models and containers, runs
validators, and owns object construction, serialization, errors, defaults,
and aliases. Homogeneous primitive lists use direct coercion loops and fall
back to the generic error-accumulating validator on the first unsupported
value.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds a machine-wide lock. Run it through `pixi`; invoking
`bench/bench.py` directly can produce distorted numbers when other repository
jobs are active.

## License

MIT
