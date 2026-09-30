"""The ``safetensors.torch`` API, over the Rust core.

Same names, same signatures, same bytes on disk. ``save``/``save_file`` build
the header in Rust and hand the tensors' own memory to the write engine —
CUDA tensors drained through pinned staging, nothing joined; ``load_file``,
``load_files`` and ``safe_open`` allocate every destination with torch (on
the host or on the CUDA device asked for) and have the read engine fill it,
with as many files in flight and readers per file as the planner chooses for
this machine (``explain()`` prints the choice).

Differences from the reference, all deliberate: non-contiguous tensors are
packed rather than refused; tensors that share storage are each written whole
rather than refused (they come back as separate tensors); device-resident
tensors are accepted by ``save``/``save_file``; ``save_file`` takes
``durable`` and ``atomic``;
``load_files``, ``serialize`` and ``explain`` exist; ``load_files`` takes a
``select`` mapping (a slice per tensor, or a `causalab.neural.engines.pytorch_hooks.weights.Shard`, see
`select_shards`) and ``safe_open`` has ``get_sharded``, so a tensor
parallel rank reads only its part.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING

import torch

from . import _files
from ._files import Device, PathLike, as_device
from ._header import Layout, TensorEntry
from ._select import Selection, SelectSpec, Shard, resolve, select_shards
from ._serialize import Payload, serialize, write_file
from ._slices import SafeSlice
from .errors import FormatError, SelectError

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

__all__ = [
    "Payload",
    "Shard",
    "explain",
    "load",
    "load_file",
    "load_files",
    "safe_open",
    "save",
    "save_file",
    "select_shards",
    "serialize",
    "stream_files",
]


def save(
    tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None
) -> bytes:
    """Serialize ``tensors`` to ``bytes``, identical to ``safetensors.torch.save``.

    Non-contiguous tensors are packed (``.contiguous()``, a copy) rather than
    refused; device tensors are copied to the host first (there is no file to
    stream them to). ``metadata`` ``None`` writes no ``__metadata__`` key,
    ``{}`` an empty one; keys keep their insertion order.
    """
    return serialize(tensors, metadata).to_bytes()


def save_file(
    tensors: Mapping[str, torch.Tensor],
    filename: PathLike,
    metadata: Mapping[str, str] | None = None,
    *,
    durable: bool = False,
    atomic: bool = True,
) -> None:
    """Write ``tensors`` to ``filename``: header, then each tensor's bytes
    straight from its own memory, GIL released. CUDA tensors are drained
    through the write engine's pinned staging ring, the copy of the next
    chunk overlapped with the write of the current one. See [`save`][] for
    what else is accepted.

    ``durable`` fsyncs the file and its directory before returning;
    ``atomic`` (the default) writes a sibling temporary file and renames it
    into place, so ``filename`` never holds a half-written object.
    """
    write_file(tensors, filename, metadata, durable=durable, atomic=atomic)


def load(data: bytes) -> dict[str, torch.Tensor]:
    """Deserialize an in-memory object into CPU tensors (one copy each)."""
    layout = Layout.from_bytes(data)
    view = memoryview(data)
    out: dict[str, torch.Tensor] = {}
    for entry in layout.in_data_order():
        dtype = entry.torch_dtype
        if entry.nbytes == 0:
            out[entry.name] = torch.empty(entry.shape, dtype=dtype)
            continue
        raw = torch.frombuffer(
            bytearray(view[entry.start : entry.end]), dtype=torch.uint8
        )
        out[entry.name] = raw.view(dtype).reshape(entry.shape)
    return {name: out[name] for name in sorted(out)}


def load_file(filename: PathLike, device: Device = "cpu") -> dict[str, torch.Tensor]:
    """Read every tensor of ``filename`` onto ``device``."""
    return _files.load_files([filename], as_device(device), None)


def load_files(
    filenames: Sequence[PathLike],
    device: Device = "cpu",
    keys: Sequence[str] | None = None,
    select: Mapping[str, SelectSpec] | None = None,
    *,
    group: ProcessGroup | None = None,
    shards: Mapping[str, Shard] | None = None,
) -> dict[str, torch.Tensor]:
    """Read a sharded checkpoint as one call, many files in flight.

    ``keys`` restricts the read to those tensors (only their bytes are
    touched); a name present in two files is a [`FormatError`][], a
    requested name in none a ``KeyError``.

    ``select`` cuts tensors before they are read: per name, an index — ints,
    slices, ``Ellipsis``, as ``t[index]`` would take — or a `causalab.neural.engines.pytorch_hooks.weights.Shard`
    (see `select_shards`). The tensor comes back at the selected shape
    and only its runs are read — exactly the bytes wanted for a cut along the
    outer dimensions; for a cut along inner ones, runs whose gaps are cheaper
    to read through than to skip are read as one (the profile's rule;
    ``explain`` shows the amplification). Stepped slices read their covering
    box. A selection that does not fit its tensor is a [`SelectError`][].

    ``group`` opts into a collective load (CPU/Gloo or CUDA/NCCL). Every
    member must call with identical files, keys and selections; those tensors
    are replicated: divided among group ranks, read once per group, then
    broadcast in bounded byte batches. Set the current CUDA device to
    ``device`` first.

    ``shards`` names the tensors of which each rank wants only its own
    `causalab.neural.engines.pytorch_hooks.weights.Shard` — tensor-parallel narrowing. The shard may differ per rank
    (its ``rank``; ``dim`` and ``world`` must agree across the group) and is
    what the rank receives: read directly by that rank, never broadcast, and
    counted at shard size in every memory bound. A cut along the outer
    dimension is one contiguous read; a cut along an inner one is read as
    blocks of whole rows, one per rank, and redistributed with an all-to-all.
    A name in both ``select`` and ``shards`` is a [`SelectError`][].
    Without ``group``, ``shards`` are ordinary selections.
    """
    if group is not None:
        from . import _distributed

        return _distributed.load_files(
            filenames, as_device(device), keys, select, shards or {}, group
        )
    return _files.load_files(
        filenames, as_device(device), keys, merge_shards(select, shards or {})
    )


def merge_shards(
    select: Mapping[str, SelectSpec] | None, shards: Mapping[str, Shard]
) -> dict[str, SelectSpec]:
    """``select`` with ``shards`` added, refusing a name in both."""
    both = sorted(set(select or {}) & set(shards))
    if both:
        raise SelectError(f"tensors in both select and shards: {both}")
    return {**(select or {}), **shards}


def stream_files(
    requests: Sequence[tuple[Sequence[PathLike], Sequence[str] | None]],
    device: Device = "cpu",
    *,
    group: ProcessGroup,
    shards: Mapping[str, Shard] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Several coordinated [`load_files`][] requests, streamed and pipelined:
    yields ``(name, tensor)`` for every tensor of every ``(filenames, keys)``
    request, in request order, while the next request is already being read.
    A rank holds its own slice of a request (about ``1/world`` of its bytes)
    plus a few tensors in flight, never the whole request, so requests may be
    ``world`` times larger than a ``load_files`` call for the same device
    memory; that is what keeps each rank's reads long enough to run at the
    storage's rate. Every member of ``group`` must pass identical requests;
    see [`load_files`][] for what ``group`` requires.

    ``shards`` applies across the requests, as in [`load_files`][]; a name
    that no request loads is a [`SelectError`][] once the last request has
    streamed.

    Nothing runs until the first ``next``: the errors [`load_files`][]
    documents for a request surface from the iteration, not from this call.
    Consume the iterator to exhaustion. Every yield sits between collectives
    that every member must reach, so a rank that stops early (a ``break``, an
    exception in the consumer) leaves the others waiting for it until the
    process group times out.
    """
    from . import _distributed

    return _distributed.stream_files(requests, as_device(device), shards or {}, group)


def explain(
    filenames: PathLike | Sequence[PathLike],
    device: Device = "cpu",
    keys: Sequence[str] | None = None,
    select: Mapping[str, SelectSpec] | None = None,
) -> str:
    """How this machine would read ``filenames`` to ``device``, and why; with
    ``select``, per selection the bytes wanted against the bytes read."""
    if isinstance(filenames, str | os.PathLike):
        filenames = [filenames]
    return _files.explain(filenames, as_device(device), keys, select)


class safe_open:  # the reference's name, so it is not CapWords
    """Open a file for reading tensors one at a time, or in slices.

    Only the header is read on open; ``get_tensor`` and ``get_slice`` read
    exactly the bytes they return.
    """

    __slots__ = ("_device", "_layout", "_path", "_plan")

    def __init__(
        self, filename: PathLike, framework: str = "pt", device: Device = "cpu"
    ) -> None:
        if framework != "pt":
            raise ValueError(f"only framework='pt' is supported, got {framework!r}")
        self._path = os.fspath(filename)
        self._device = as_device(device)
        self._layout = Layout.from_file(self._path)
        self._plan: _files.Plan | None = None

    def __enter__(self) -> safe_open:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def keys(self) -> list[str]:
        """Tensor names, sorted."""
        return sorted(self._layout.tensors)

    def offset_keys(self) -> list[str]:
        """Tensor names in the order their bytes appear in the file."""
        return [t.name for t in self._layout.in_data_order()]

    def metadata(self) -> dict[str, str] | None:
        return None if self._layout.metadata is None else dict(self._layout.metadata)

    def _file_plan(self) -> _files.Plan:
        """The plan for this file to the open device, made once. It fixes how
        a tensor's range is split across readers and the staging a device
        destination needs; the fit check is off because each read is a
        subset of the file the plan is made for."""
        if self._plan is None:
            self._plan = _files.Plan.for_read(
                [self._path],
                [self._layout.wanted_bytes()],
                self._device,
                check_fit=False,
            )
        return self._plan

    def _read(self, entry: TensorEntry, selection: Selection) -> torch.Tensor:
        return _files.read_selection(
            self._path, entry, selection, self._file_plan(), self._device
        )

    def _entry(self, name: str) -> TensorEntry:
        try:
            return self._layout.tensors[name]
        except KeyError:
            raise FormatError(
                f"file {self._path} does not contain tensor {name!r}"
            ) from None

    def get_tensor(self, name: str) -> torch.Tensor:
        """Read one tensor, whole, onto the open device."""
        entry = self._entry(name)
        target = _files.engine_device(self._device)
        tensor = entry.empty(target)
        if entry.nbytes:
            _files.sync_torch(target)
            self._file_plan().execute(
                [(self._path, [_files.transfer(entry.start, tensor)])], target
            )
        return tensor.to(self._device)

    def get_slice(self, name: str) -> SafeSlice:
        """A lazy view: index it to read just those bytes."""
        return SafeSlice(self._entry(name), self._read, self._device)

    def get_sharded(self, name: str, dim: int, rank: int, world: int) -> torch.Tensor:
        """Shard ``rank`` of ``world`` equal shards of ``name`` along ``dim``
        — ``torch.chunk(t, world, dim)[rank]`` when ``dim`` divides, read
        without the rest of the tensor. [`SelectError`][] when it does
        not divide or ``dim``/``rank`` are out of range."""
        entry = self._entry(name)
        selection = resolve(name, entry.shape, Shard(dim, rank, world))
        return self._read(entry, selection).to(self._device)
