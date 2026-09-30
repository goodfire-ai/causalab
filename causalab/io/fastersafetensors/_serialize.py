"""Building the on-disk object without joining it: header plus parts.

``write_file`` hands every tensor to the write engine where it lives: CPU
tensors as host parts written straight from their memory, CUDA tensors as
device parts the engine drains through its pinned staging ring with the
device-to-host copy of the next chunk overlapped against the write of the
current one. Nothing is joined and no tensor is copied in Python. ``save()``
has no file to stream to, so it copies device tensors to the host first and
joins once (`Payload`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

import causalab.io.fastersafetensors._core as _core
from ._buffers import as_memoryview
from ._dtypes import header_dtype
from ._files import sync_torch
from .errors import FormatError

PathLike = str | os.PathLike[str]
SpecRow = tuple[str, str, list[int], int]
"""``(name, dtype, shape, nbytes)`` as ``_core`` describes a tensor to write."""


@dataclass(frozen=True, slots=True)
class Part:
    """One tensor ready to write: its spec and the contiguous tensor that
    owns the bytes (on the host, or on a CUDA device)."""

    spec: SpecRow
    tensor: torch.Tensor

    @property
    def name(self) -> str:
        return self.spec[0]

    @property
    def nbytes(self) -> int:
        return self.spec[3]

    @property
    def on_device(self) -> bool:
        return self.tensor.device.type == "cuda"

    @property
    def raw(self) -> tuple[int, int, bool]:
        """``(address, nbytes, on_device)`` for ``_core.write_object``."""
        return (
            self.tensor.data_ptr() if self.nbytes else 0,
            self.nbytes,
            self.on_device,
        )

    def to_host(self) -> Part:
        """The same part with its bytes on the host (a copy when it was not)."""
        if self.tensor.device.type == "cpu":
            return self
        return Part(self.spec, self.tensor.cpu())


def prepare(name: str, tensor: torch.Tensor, *, host_only: bool) -> Part:
    """A tensor as a part. Non-contiguous tensors are packed (a copy). CUDA
    tensors stay where they are unless ``host_only``; any other accelerator
    (mps, ...) is copied to the host, because the engine's device path is the
    CUDA runtime."""
    # a caller's mapping is untyped at run time; the check is the refusal
    if not isinstance(tensor, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise FormatError(
            f"tensor {name!r} is a {type(tensor).__name__}, not a torch.Tensor"
        )
    contiguous = tensor.contiguous()
    nbytes = contiguous.numel() * contiguous.element_size()
    part = Part(
        (name, header_dtype(contiguous.dtype), list(contiguous.shape), nbytes),
        contiguous,
    )
    if host_only or contiguous.device.type not in ("cpu", "cuda"):
        return part.to_host()
    return part


def _metadata_rows(metadata: Mapping[str, str] | None) -> list[tuple[str, str]] | None:
    return None if metadata is None else list(metadata.items())


def _one_device(parts: list[Part]) -> tuple[list[Part], int | None]:
    """The engine drains device parts through one runtime, so one CUDA device
    per write: the lowest ordinal present keeps its parts on the device, any
    other CUDA device's parts are copied to the host first."""
    ordinals = {p.tensor.device.index for p in parts if p.on_device}
    if not ordinals:
        return parts, None
    chosen = min(ordinals)
    kept = [
        p if not p.on_device or p.tensor.device.index == chosen else p.to_host()
        for p in parts
    ]
    return kept, chosen


def write_file(
    tensors: Mapping[str, torch.Tensor],
    filename: PathLike,
    metadata: Mapping[str, str] | None,
    *,
    durable: bool,
    atomic: bool,
) -> int:
    """Write ``tensors`` to ``filename`` through the write engine; returns the
    bytes written. Torch's stream is synchronized before device pointers go
    down (see `sync_torch`)."""
    names = list(tensors)
    parts, device = _one_device(
        [prepare(n, tensors[n], host_only=False) for n in names]
    )
    if device is not None:
        sync_torch(torch.device("cuda", device))
    return _core.write_object(
        os.fspath(filename),
        [p.spec for p in parts],
        [p.raw for p in parts],
        _metadata_rows(metadata),
        durable,
        atomic,
        device,
    )


@dataclass(frozen=True, slots=True)
class Payload:
    """A safetensors object as a header and the tensors that follow it, in
    data-section order, never joined into one buffer.

    ``header`` is the length prefix, the JSON and its padding; ``parts()`` are
    byte views over the tensors' memory in the order they follow the header.
    Everything the views point at is owned by ``tensors``, which the payload
    holds; they are on the host (a device tensor was copied once to get here).
    """

    header: bytes
    tensors: tuple[torch.Tensor, ...] = field(repr=False)
    specs: tuple[SpecRow, ...] = field(repr=False)
    metadata: tuple[tuple[str, str], ...] | None = field(repr=False)

    @property
    def nbytes(self) -> int:
        """Bytes the whole object occupies."""
        return len(self.header) + sum(spec[3] for spec in self.specs)

    def parts(self) -> list[memoryview]:
        """Read-only byte views of the tensors, in the order they are written."""
        return [as_memoryview(t) for t in self.tensors]

    def to_bytes(self) -> bytes:
        """The object as one ``bytes`` (one copy)."""
        return b"".join([self.header, *self.parts()])

    def write(
        self, filename: PathLike, *, durable: bool = False, atomic: bool = True
    ) -> int:
        """Stream the object to ``filename`` through the write engine; returns
        the bytes written."""
        parts = [
            Part(spec, t) for spec, t in zip(self.specs, self.tensors, strict=True)
        ]
        return _core.write_object(
            os.fspath(filename),
            [p.spec for p in parts],
            [p.raw for p in parts],
            None if self.metadata is None else list(self.metadata),
            durable,
            atomic,
            None,
        )


def serialize(
    tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None
) -> Payload:
    """Build the header for ``tensors`` and gather their bytes as host parts.

    ``metadata=None`` writes no ``__metadata__`` key; ``{}`` writes an empty
    one; keys keep their insertion order. Device tensors are copied to the
    host: a payload is for callers who want the bytes, and there is no file
    to stream them to.
    """
    names = list(tensors)
    parts = [prepare(name, tensors[name], host_only=True) for name in names]
    rows = _metadata_rows(metadata)
    header, order = _core.build_header([p.spec for p in parts], rows)
    ordered = [parts[i] for i in order]
    return Payload(
        header,
        tuple(p.tensor for p in ordered),
        tuple(p.spec for p in ordered),
        None if rows is None else tuple(rows),
    )
