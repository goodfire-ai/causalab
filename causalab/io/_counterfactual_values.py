"""Version-two JSON values: typed numbers and shared acyclic value contents.

References are local to one trace. They preserve sharing in supported values,
not links to model configuration, other traces, or arbitrary Python objects.
NumPy is imported only when a NumPy value or tag is encountered.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache


def _hex_bytes(value, length):
    if (
        type(value) is not str
        or len(value) != 2 * length
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(
            f"Invalid saved numeric bits: expected {2 * length} hex digits"
        )
    return bytes.fromhex(value)


@lru_cache(maxsize=1)
def _numpy_types():
    import numpy as np

    # dtype.str alone conflates int64/longlong on some platforms. Canonical
    # scalar-class names retain that distinction without importing saved names.
    names = (
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "longlong",
        "ulonglong",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
    return {np.dtype(name).type.__name__: np.dtype(name).type for name in names}


def _numpy_info(value):
    import numpy as np

    kind = type(value)
    if kind is not np.ndarray and not isinstance(value, np.generic):
        raise ValueError(f"Unsupported saved value type: {kind.__name__}")
    dtype = value.dtype
    if (
        dtype.metadata is not None
        or dtype.fields is not None
        or dtype.subdtype is not None
        or _numpy_types().get(dtype.type.__name__) is not dtype.type
        or (kind is not np.ndarray and kind is not dtype.type)
    ):
        raise ValueError(
            "Saved NumPy values require a supported numeric dtype without metadata"
        )
    return "array" if kind is np.ndarray else "scalar", dtype


class _Encoder:
    def __init__(self):
        self.counts = {}
        self.active = set()
        self.references = {}

    def inspect(self, value):
        kind = type(value)
        if kind in (type(None), bool, int, str) or (
            kind is float and math.isfinite(value)
        ):
            return
        if kind not in (float, complex, list, tuple, dict):
            if not kind.__module__.startswith("numpy"):
                raise ValueError(f"Unsupported saved value type: {kind.__name__}")
            _numpy_info(value)
        identity = id(value)
        if identity in self.active:
            raise ValueError("Cannot save cyclic values; use acyclic containers")
        self.counts[identity] = self.counts.get(identity, 0) + 1
        if self.counts[identity] > 1:
            return
        self.active.add(identity)
        try:
            if kind is dict:
                for key, item in value.items():
                    self.inspect(key)
                    self.inspect(item)
            elif kind in (list, tuple):
                for item in value:
                    self.inspect(item)
        finally:
            self.active.remove(identity)

    def encode(self, value):
        identity = id(value)
        if self.counts.get(identity, 0) > 1:
            if identity in self.references:
                return {"ref": self.references[identity]}
            reference = len(self.references)
            self.references[identity] = reference
            return {"id": reference, "value": self._encode(value)}
        return self._encode(value)

    def _encode(self, value):
        kind = type(value)
        if kind is float and not math.isfinite(value):
            return {"float64": struct.pack("!d", value).hex()}
        if kind is complex:
            return {"complex128": struct.pack("!dd", value.real, value.imag).hex()}
        if kind is tuple:
            return {"tuple": [self.encode(item) for item in value]}
        if kind is list:
            return [self.encode(item) for item in value]
        if kind is dict:
            return {
                "mapping": [
                    [self.encode(key), self.encode(item)] for key, item in value.items()
                ]
            }
        if kind.__module__.startswith("numpy"):
            numpy_kind, dtype = _numpy_info(value)
            return {
                "numpy": {
                    "kind": numpy_kind,
                    "scalar_type": dtype.type.__name__,
                    "dtype": dtype.str,
                    "shape": list(value.shape),
                    "data": value.tobytes(order="C").hex(),
                }
            }
        return value


class _Decoder:
    def __init__(self):
        self.references = {}
        self.definitions = {}
        self.active = set()

    def _reference(self, value):
        if type(value) is not int or value < 0:
            raise ValueError("Saved value references must be nonnegative integers")
        return value

    def inspect(self, value):
        # Index declarations before decoding: JSON object members may be
        # reordered without changing their meaning, including trace variables.
        if type(value) is dict:
            if set(value) == {"id", "value"}:
                reference = self._reference(value["id"])
                if reference in self.definitions:
                    raise ValueError("Duplicate saved value reference")
                self.definitions[reference] = value["value"]
            for item in value.values():
                self.inspect(item)
        elif type(value) is list:
            for item in value:
                self.inspect(item)

    def _resolve(self, reference):
        if reference in self.references:
            return self.references[reference]
        if reference not in self.definitions or reference in self.active:
            raise ValueError("Unknown or cyclic saved value reference")
        self.active.add(reference)
        try:
            decoded = self.decode(self.definitions[reference])
        finally:
            self.active.remove(reference)
        self.references[reference] = decoded
        return decoded

    def decode(self, value):
        kind = type(value)
        if kind in (type(None), bool, int, str):
            return value
        if kind is float and math.isfinite(value):
            return value
        if kind is list:
            return [self.decode(item) for item in value]
        if kind is not dict:
            raise ValueError("Unknown saved value format")
        fields = set(value)
        if fields == {"ref"}:
            return self._resolve(self._reference(value["ref"]))
        if fields == {"id", "value"}:
            return self._resolve(self._reference(value["id"]))
        if fields == {"float64"}:
            return struct.unpack("!d", _hex_bytes(value["float64"], 8))[0]
        if fields == {"complex128"}:
            return complex(*struct.unpack("!dd", _hex_bytes(value["complex128"], 16)))
        if fields == {"tuple"} and type(value["tuple"]) is list:
            return tuple(self.decode(item) for item in value["tuple"])
        if fields == {"mapping"} and type(value["mapping"]) is list:
            result = {}
            for pair in value["mapping"]:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError("Saved mapping entries must be key/value pairs")
                key, item = map(self.decode, pair)
                try:
                    if key in result:
                        raise ValueError("Duplicate saved mapping key")
                    result[key] = item
                except TypeError as exc:
                    raise ValueError("Saved mapping keys must be hashable") from exc
            return result
        if fields == {"numpy"}:
            return self._numpy(value["numpy"])
        raise ValueError("Unknown saved value format")

    def _numpy(self, value):
        import numpy as np

        expected = {"kind", "scalar_type", "dtype", "shape", "data"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("Invalid saved NumPy value fields")
        name, dtype_string, shape = value["scalar_type"], value["dtype"], value["shape"]
        if (
            type(name) is not str
            or name not in _numpy_types()
            or type(dtype_string) is not str
            or not dtype_string
            or dtype_string[0] not in "<>|"
        ):
            raise ValueError("Unsupported saved NumPy dtype")
        dtype = np.dtype(_numpy_types()[name]).newbyteorder(dtype_string[0])
        if dtype.str != dtype_string or dtype.type.__name__ != name:
            raise ValueError("Saved NumPy dtype and scalar type disagree")
        if (
            value["kind"] not in ("scalar", "array")
            or type(shape) is not list
            or any(type(size) is not int or size < 0 for size in shape)
            or (value["kind"] == "scalar" and shape)
        ):
            raise ValueError("Invalid saved NumPy kind or shape")
        data = _hex_bytes(value["data"], math.prod(shape) * dtype.itemsize)
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        if value["kind"] == "array":
            return array.copy()
        scalar = array[()]
        if scalar.dtype.str != dtype_string:
            raise ValueError("Saved NumPy scalar must use its native byte order")
        return scalar


def encode_values(values):
    """Encode one trace's values without merging distinct equal numeric atoms."""
    if type(values) is not dict or any(type(name) is not str for name in values):
        raise ValueError("Saved trace values require string variable names")
    encoder = _Encoder()
    for value in values.values():
        encoder.inspect(value)
    return {name: encoder.encode(value) for name, value in values.items()}


def decode_values(values):
    """Decode a trace with a fresh reference scope; reject cycles/dangling refs."""
    if type(values) is not dict or any(type(name) is not str for name in values):
        raise ValueError("Saved trace values require string variable names")
    decoder = _Decoder()
    for value in values.values():
        decoder.inspect(value)
    return {name: decoder.decode(value) for name, value in values.items()}
