from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import pydantic as pd  # noqa: E402

import mojopydantic as mp  # noqa: E402


def best_time(function, repeat: int = 5) -> float:
    best = math.inf
    function()
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


class MojoRecord(mp.BaseModel):
    id: int
    score: float
    enabled: bool
    tags: list[int]


class PydanticRecord(pd.BaseModel):
    id: int
    score: float
    enabled: bool
    tags: list[int]


def cases():
    ints_raw = json.dumps(list(range(500_000)), separators=(",", ":")).encode()
    float_values = [index * 0.125 - 10_000 for index in range(300_000)]
    floats_raw = json.dumps(float_values, separators=(",", ":")).encode()
    bools_raw = (
        b"[" + b",".join(b"true" if index & 1 else b"false" for index in range(750_000)) + b"]"
    )
    python_values = [str(index) for index in range(200_000)]
    record = {
        "id": "42",
        "score": "98.5",
        "enabled": "yes",
        "tags": ["1", 2, 3.0, True],
    }

    ours_int = mp.TypeAdapter(list[int])
    theirs_int = pd.TypeAdapter(list[int])
    ours_float = mp.TypeAdapter(list[float])
    theirs_float = pd.TypeAdapter(list[float])
    ours_bool = mp.TypeAdapter(list[bool])
    theirs_bool = pd.TypeAdapter(list[bool])

    assert ours_int.validate_json(ints_raw) == theirs_int.validate_json(ints_raw)
    assert ours_float.validate_json(floats_raw) == theirs_float.validate_json(floats_raw)
    assert ours_bool.validate_json(bools_raw) == theirs_bool.validate_json(bools_raw)
    assert ours_int.validate_python(python_values) == theirs_int.validate_python(
        python_values
    )
    assert MojoRecord.model_validate(record).model_dump() == PydanticRecord.model_validate(
        record
    ).model_dump()
    assert np.array_equal(
        ours_int.validate_json_array(ints_raw),
        np.asarray(theirs_int.validate_json(ints_raw), dtype=np.int64),
    )
    assert np.array_equal(
        ours_float.validate_json_array(floats_raw),
        np.asarray(theirs_float.validate_json(floats_raw), dtype=np.float64),
    )
    assert np.array_equal(
        ours_bool.validate_json_array(bools_raw),
        np.asarray(theirs_bool.validate_json(bools_raw), dtype=np.bool_),
    )

    return [
        (
            "validate_json_array int64 (500k)",
            lambda: ours_int.validate_json_array(ints_raw),
            lambda: np.asarray(theirs_int.validate_json(ints_raw), dtype=np.int64),
        ),
        (
            "validate_json_array float64 (300k)",
            lambda: ours_float.validate_json_array(floats_raw),
            lambda: np.asarray(
                theirs_float.validate_json(floats_raw), dtype=np.float64
            ),
        ),
        (
            "validate_json_array bool (750k)",
            lambda: ours_bool.validate_json_array(bools_raw),
            lambda: np.asarray(
                theirs_bool.validate_json(bools_raw), dtype=np.bool_
            ),
        ),
        (
            "validate_json list[int] (500k)",
            lambda: ours_int.validate_json(ints_raw),
            lambda: theirs_int.validate_json(ints_raw),
        ),
        (
            "validate_json list[float] (300k)",
            lambda: ours_float.validate_json(floats_raw),
            lambda: theirs_float.validate_json(floats_raw),
        ),
        (
            "validate_json list[bool] (750k)",
            lambda: ours_bool.validate_json(bools_raw),
            lambda: theirs_bool.validate_json(bools_raw),
        ),
        (
            "validate_python list[int] strings (200k)",
            lambda: ours_int.validate_python(python_values),
            lambda: theirs_int.validate_python(python_values),
        ),
        (
            "BaseModel.model_validate (50k)",
            lambda: [MojoRecord.model_validate(record) for _ in range(50_000)],
            lambda: [PydanticRecord.model_validate(record) for _ in range(50_000)],
        ),
    ]


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    print(
        f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; "
        f"Python {platform.python_version()}; Pydantic {pd.__version__}"
    )
    print()
    print("| case | mojopydantic | pydantic | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases():
        ours_s = best_time(ours)
        theirs_s = best_time(theirs)
        ratio = theirs_s / ours_s
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {ours_s * 1e3:.2f} ms | "
            f"{theirs_s * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
