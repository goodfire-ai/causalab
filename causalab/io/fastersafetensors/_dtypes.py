"""The torch dtype <-> header dtype table.

The header spells element types the way the reference library does
(``"F32"``, ``"BF16"``, ``"F8_E4M3"`` ...). Only dtypes the installed torch
has are in the table; ``uint16``/``uint32``/``uint64`` and the float8 family
arrived in different torch releases, so the table is built by lookup rather
than written out.
"""

from __future__ import annotations

import torch

from .errors import FormatError

# (header name, torch attribute), every pair the format and torch both know
_PAIRS: tuple[tuple[str, str], ...] = (
    ("BOOL", "bool"),
    ("U8", "uint8"),
    ("I8", "int8"),
    ("I16", "int16"),
    ("U16", "uint16"),
    ("I32", "int32"),
    ("U32", "uint32"),
    ("I64", "int64"),
    ("U64", "uint64"),
    ("F16", "float16"),
    ("BF16", "bfloat16"),
    ("F32", "float32"),
    ("F64", "float64"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
    ("F8_E4M3FNUZ", "float8_e4m3fnuz"),
    ("F8_E5M2FNUZ", "float8_e5m2fnuz"),
    ("C64", "complex64"),
)


def _build() -> tuple[dict[torch.dtype, str], dict[str, torch.dtype]]:
    to_header: dict[torch.dtype, str] = {}
    to_torch: dict[str, torch.dtype] = {}
    for header_name, attr in _PAIRS:
        dtype = getattr(torch, attr, None)
        if isinstance(dtype, torch.dtype):
            to_header[dtype] = header_name
            to_torch[header_name] = dtype
    return to_header, to_torch


TORCH_TO_HEADER, HEADER_TO_TORCH = _build()


def header_dtype(dtype: torch.dtype) -> str:
    """The header's name for a torch dtype, or [`FormatError`][] naming it."""
    try:
        return TORCH_TO_HEADER[dtype]
    except KeyError:
        supported = ", ".join(str(d) for d in TORCH_TO_HEADER)
        raise FormatError(
            f"dtype {dtype} cannot be stored; supported: {supported}"
        ) from None


def torch_dtype(name: str) -> torch.dtype:
    """The torch dtype for a header name, or [`FormatError`][] naming it."""
    try:
        return HEADER_TO_TORCH[name]
    except KeyError:
        raise FormatError(
            f"header dtype {name!r} has no torch equivalent in this build"
        ) from None
