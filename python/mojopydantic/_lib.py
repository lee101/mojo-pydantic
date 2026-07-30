from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(os.environ.get("MOJOPYDANTIC_LIB", ROOT / "dist" / "libmojo-pydantic.so"))

I = ctypes.c_int64
_SIGNATURES = {
    "mp_json_i64_array": ([I, I, I, I, I], I),
    "mp_json_f64_array": ([I, I, I, I, I], I),
    "mp_json_bool_array": ([I, I, I, I, I], I),
}
_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIB.exists():
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` or set "
                "MOJOPYDANTIC_LIB"
            )
        _library = ctypes.CDLL(str(LIB))
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def typed_json_ndarray(
    raw: bytes, item_type: type, strict: bool
) -> np.ndarray | None:
    if not raw:
        return None
    capacity = raw.count(b",") + 1
    # Both arrays remain strongly referenced until the synchronous ctypes call
    # returns. Their dtypes are the exact C-ABI element widths and np.empty
    # produces writable, C-contiguous storage.
    source = np.frombuffer(raw, dtype=np.uint8)
    if item_type is int:
        result = np.empty(capacity, dtype=np.int64)
        name = "mp_json_i64_array"
    elif item_type is float:
        result = np.empty(capacity, dtype=np.float64)
        name = "mp_json_f64_array"
    elif item_type is bool:
        result = np.empty(capacity, dtype=np.bool_)
        name = "mp_json_bool_array"
    else:
        return None
    count = getattr(lib(), name)(
        source.ctypes.data, len(raw), result.ctypes.data, capacity, int(strict)
    )
    if count < 0 or count > capacity:
        return None
    return result[:count]


def typed_json_array(raw: bytes, item_type: type, strict: bool) -> list | None:
    result = typed_json_ndarray(raw, item_type, strict)
    if result is None:
        return None
    return result.tolist()
