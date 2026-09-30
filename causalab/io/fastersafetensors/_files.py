"""Reading files: the plan, the destinations, the job.

Every destination tensor is allocated by torch here, then filled by one
``_core.read_job`` call that runs the read engine over every file of the
request — ``files_in_flight`` files at once, ``readers_per_file`` pieces of
``split_bytes`` each, device destinations through the engine's pinned
staging ring. Rust never allocates a tensor.

A tensor is read whole or as a `Selection`
(a box plus the view that narrows it). The reads for a box come from
``_core.select_reads`` — the core's runs, coalesced under the profile's
policy for the file's storage, each read with the placements that put its
wanted bytes into the destination — and `stage` turns them into
``read_job`` rows: the read shifted by the tensor's start, landing in the
result tensor when the result is the box's bytes (whole tensors, shards,
inner cuts) or in a scratch box that torch's ``copy_`` narrows when the
index had a step.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import torch

import causalab.io.fastersafetensors._core as _core
from ._header import Layout, TensorEntry
from ._select import Item, Selection, SelectSpec, resolve
from .errors import FormatError, SelectError

PathLike = str | os.PathLike[str]
Device = str | int | torch.device

Placement = tuple[int, int, int]
"""``(offset within the read, offset within the destination, nbytes)``: one
piece of a coalesced read and where it lands."""
Transfer = tuple[int, int, int, int | None, list[Placement] | None]
"""``(file offset, nbytes, address, destination offset, placements)``. The
destination offset is ``None`` for a host destination and an offset into
the buffer at ``address`` for a device one; ``placements`` is ``None`` when
the range lands whole, else the pieces that land, packed from there."""
FileJob = tuple[str, list[Transfer]]

Pick = tuple[TensorEntry, Selection]
"""One tensor of a file and the part of it wanted."""
ReadRow = tuple[int, int, int, list[Placement] | None]
"""``(offset within the tensor, nbytes, dst within the destination,
placements)``: one read as ``_core.select_reads`` returned it."""
Reads = list[ReadRow]
"""The reads of one pick."""

HEADER_READERS = 16
"""Headers are read concurrently: each is a small read, and on a network mount
the round trip, not the bytes, is the cost."""
CONCURRENCY_PIECE = 64 << 20
"""Piece size when a caller asks for several readers per file (``Plan.with_concurrency``)."""
CONCURRENCY_STAGING_MAX = 128
"""Cap on staging buffers for a caller-overridden plan."""


def as_device(device: Device) -> torch.device:
    """The reference accepts ``"cpu"``, ``"cuda:1"`` or a bare ordinal (CUDA)."""
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def cuda_index(device: torch.device) -> int | None:
    """The CUDA ordinal the engine lands bytes on, ``None`` for anything else."""
    if device.type != "cuda":
        return None
    # torch's stubs type ``index`` as ``int``; ``torch.device("cuda").index`` is None
    return (
        device.index
        if device.index is not None  # pyright: ignore[reportUnnecessaryComparison]
        else torch.cuda.current_device()
    )


def engine_device(device: torch.device) -> torch.device:
    """Where the engine can land bytes for ``device``: the host and CUDA
    directly; any other accelerator (mps, ...) goes through a host tensor and
    a ``copy_`` afterwards, because the engine's device path is the CUDA
    runtime."""
    index = cuda_index(device)
    if index is None:
        return torch.device("cpu")
    return torch.device("cuda", index)


def sync_torch(device: torch.device) -> None:
    """Order torch's work before the engine's: the runtime copies on its own
    non-blocking stream, which does not wait for torch's default stream, so
    memory torch has just freed or written must be quiet before a pointer to
    it goes down. After the engine returns the copies are complete (it
    synchronizes), so nothing is needed on the way back."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def available_bytes(free: int, reserved: int, allocated: int) -> int:
    """Bytes a destination on a CUDA device can still take: what the driver
    reports free plus what torch has reserved in its cache but not handed out.
    This is aggregate headroom, not a guarantee that a contiguous allocation fits.
    Only the first is visible to ``cudaMemGetInfo``; a caller that has just
    warmed torch's allocator (transformers does, with an allocation the size
    of the model) shows almost nothing free there while the cache holds
    everything the load needs."""
    return free + max(0, reserved - allocated)


def device_headroom(index: int) -> int:
    """`available_bytes` for CUDA device ``index``."""
    free, _total = torch.cuda.mem_get_info(index)
    return available_bytes(
        free, torch.cuda.memory_reserved(index), torch.cuda.memory_allocated(index)
    )


def _plan_device(
    device: torch.device, *, check_fit: bool
) -> tuple[str | None, int | None]:
    """What the planner needs to know about the destination: the CUDA
    ordinal and, when ``check_fit``, its free memory."""
    index = cuda_index(device)
    if index is None:
        return None, None
    if not check_fit:
        return f"cuda:{index}", None
    return f"cuda:{index}", device_headroom(index)


@dataclass(frozen=True, slots=True)
class Plan:
    """``plan::plan_read``'s answer for one request."""

    files_in_flight: int
    split_bytes: int
    readers_per_file: int
    transport: str
    staging: tuple[int, int] | None
    reasons: tuple[str, ...]

    @classmethod
    def for_read(
        cls,
        paths: Sequence[str],
        wanted: Sequence[int],
        device: torch.device,
        *,
        check_fit: bool = True,
        coalesced: _core.CoalesceDict | None = None,
        allocation_bytes: int | None = None,
    ) -> Plan:
        """The plan for reading ``wanted[i]`` bytes of ``paths[i]`` to
        ``device``. ``check_fit`` compares ``allocation_bytes`` (or the read
        total when omitted) with aggregate device headroom; ``safe_open``
        turns it off because its reads are a subset of the file being planned. ``coalesced`` is what
        ``_core.select_reads`` summarised for the request, so the reasons can
        say what coalescing did."""
        dev, free = _plan_device(device, check_fit=check_fit)
        summary = (
            None
            if coalesced is None
            else (
                coalesced["runs"],
                coalesced["reads"],
                coalesced["wanted_bytes"],
                coalesced["read_bytes"],
            )
        )
        raw = _core.plan_read(
            list(paths), list(wanted), dev, free, summary, allocation_bytes
        )
        return cls(
            raw["files_in_flight"],
            raw["split_bytes"],
            raw["readers_per_file"],
            raw["transport"],
            raw["staging"],
            tuple(raw["reasons"]),
        )

    def with_concurrency(self, readers_per_file: int, files_in_flight: int) -> Plan:
        """This plan with explicit read concurrency: ``readers_per_file`` pieces
        of ``CONCURRENCY_PIECE`` bytes per file and ``files_in_flight`` files at
        once, staging sized to the new geometry. For callers that know their
        request is one slice of a larger concurrent load (a coordinated rank
        reads a few files' worth of tensors while fifteen others do the same),
        where the profile's single-request plan leaves the link idle."""
        staging = (
            None
            if self.staging is None
            else (
                min(
                    CONCURRENCY_STAGING_MAX,
                    max(2, files_in_flight * readers_per_file * 2),
                ),
                self.staging[1],
            )
        )
        return replace(
            self,
            files_in_flight=files_in_flight,
            readers_per_file=readers_per_file,
            split_bytes=CONCURRENCY_PIECE if readers_per_file > 1 else self.split_bytes,
            staging=staging,
            reasons=(
                *self.reasons,
                f"caller override: {files_in_flight} files in flight, "
                f"{readers_per_file} readers per file",
            ),
        )

    def execute(
        self, files: Sequence[FileJob], device: torch.device
    ) -> _core.ReadReport:
        """Run ``files`` through the engine as this plan says; every
        destination is filled when this returns."""
        return _core.read_job(
            list(files),
            self.files_in_flight,
            self.split_bytes,
            self.readers_per_file,
            self.transport,
            self.staging,
            cuda_index(device),
        )


def transfer(offset: int, tensor: torch.Tensor) -> Transfer:
    """Fill the contiguous ``tensor`` with its byte count from ``offset``."""
    nbytes = tensor.numel() * tensor.element_size()
    address = tensor.data_ptr() if nbytes else 0
    return (offset, nbytes, address, 0 if tensor.device.type == "cuda" else None, None)


def read_transfer(entry: TensorEntry, read: ReadRow, dest: torch.Tensor) -> Transfer:
    """One read of ``entry`` landing at its offset in the contiguous ``dest``."""
    offset, nbytes, dst, placements = read
    if dest.device.type == "cuda":
        return (entry.start + offset, nbytes, dest.data_ptr(), dst, placements)
    return (entry.start + offset, nbytes, dest.data_ptr() + dst, None, placements)


@dataclass(slots=True)
class Staged:
    """A selection in flight: the result tensor and, when the bytes land in a
    scratch box first, the box and the index that narrows it."""

    result: torch.Tensor
    scratch: torch.Tensor | None = None
    view: tuple[Item, ...] | None = None

    def finish(self) -> torch.Tensor:
        if self.scratch is not None and self.view is not None:
            self.result.copy_(self.scratch[self.view])
        return self.result


def stage(
    entry: TensorEntry, selection: Selection, reads: Reads, target: torch.device
) -> tuple[Staged, list[Transfer]]:
    """Allocate the result of ``selection`` on ``target`` and the transfers
    that fill it from ``reads``: straight into the result when it is the
    box's bytes, else into a scratch box the result is narrowed from. The
    caller issues the transfers and calls ``Staged.finish`` once they have
    landed."""
    dtype = entry.torch_dtype
    result = torch.empty(selection.result_shape, dtype=dtype, device=target)
    if not reads or selection.is_empty:
        return Staged(result), []
    if selection.is_plain:
        return Staged(result), [read_transfer(entry, read, result) for read in reads]
    scratch = torch.empty(selection.box_shape, dtype=dtype, device=target)
    transfers = [read_transfer(entry, read, scratch) for read in reads]
    return Staged(result, scratch, selection.view), transfers


def select_reads(
    picks: Sequence[tuple[str, Pick]],
) -> tuple[list[Reads], _core.CoalesceDict]:
    """The core's reads for every ``(path, pick)``, and what coalescing did."""
    items = [
        (path, entry.name, list(entry.shape), selection.ranges, entry.dtype)
        for path, (entry, selection) in picks
    ]
    return _core.select_reads(items)


def bytes_read(reads: Reads) -> int:
    return sum(nbytes for _, nbytes, _, _ in reads)


def read_layouts(paths: Sequence[str]) -> list[Layout]:
    """Parse every header, several at a time (the GIL is released per read)."""
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=min(HEADER_READERS, len(paths))) as pool:
        return list(pool.map(Layout.from_file, paths))


def read_selection(
    path: str,
    entry: TensorEntry,
    selection: Selection,
    plan: Plan,
    device: torch.device,
) -> torch.Tensor:
    """``selection`` of ``entry`` in ``path`` as a fresh tensor of the
    result's shape on ``device`` (or on the host, if the engine cannot land
    there)."""
    target = engine_device(device)
    (reads,), _ = select_reads([(path, (entry, selection))])
    staged, transfers = stage(entry, selection, reads, target)
    if transfers:
        sync_torch(target)
        plan.execute([(path, transfers)], target)
    return staged.finish()


def check_unique(paths: Sequence[str], layouts: Sequence[Layout]) -> None:
    """A checkpoint names every tensor once across its files."""
    seen: dict[str, str] = {}
    for path, layout in zip(paths, layouts, strict=True):
        for name in layout.tensors:
            if name in seen:
                raise FormatError(
                    f"tensor {name!r} appears in both {seen[name]} and {path}"
                )
            seen[name] = path


@dataclass(frozen=True, slots=True)
class Resolved:
    """A request's picks with their reads: per file, the picks and each
    pick's reads side by side, plus the coalescing summary."""

    files: list[tuple[str, list[tuple[Pick, Reads]]]]
    summary: _core.CoalesceDict

    @classmethod
    def of(cls, wanted: Sequence[tuple[str, Sequence[Pick]]]) -> Resolved:
        flat = [(path, pick) for path, picks in wanted for pick in picks]
        reads, summary = select_reads(flat)
        it = iter(reads)
        files = [(path, [(pick, next(it)) for pick in picks]) for path, picks in wanted]
        return cls(files, summary)

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self.files]

    @property
    def bytes_per_file(self) -> list[int]:
        return [sum(bytes_read(reads) for _, reads in picks) for _, picks in self.files]

    @property
    def wanted_bytes(self) -> int:
        return sum(
            s.wanted_bytes(e.itemsize) for _, picks in self.files for (e, s), _ in picks
        )

    @property
    def allocation_bytes(self) -> int:
        return sum(
            s.allocation_bytes(e.itemsize)
            for _, picks in self.files
            for (e, s), _ in picks
        )


def fill(
    wanted: Sequence[tuple[str, Sequence[Pick]]],
    device: torch.device,
    *,
    readers_per_file: int | None = None,
) -> dict[str, torch.Tensor]:
    """Allocate a tensor on ``device`` for every pick and fill them all with
    one engine job: resolve the reads, plan, allocate,
    synchronize torch, read, narrow what landed in a scratch box. Returns
    the tensors by name, unsorted. ``readers_per_file`` overrides the plan's
    concurrency (every file in flight, that many readers each)."""
    resolved = Resolved.of(wanted)
    plan = Plan.for_read(
        resolved.paths,
        resolved.bytes_per_file,
        device,
        coalesced=resolved.summary,
        allocation_bytes=resolved.allocation_bytes,
    )
    if readers_per_file is not None:
        plan = plan.with_concurrency(readers_per_file, len(resolved.paths))
    if os.environ.get("FASTERSAFETENSORS_TRACE", "") == "1":
        print(
            f"fastersafetensors fill: {len(resolved.paths)} files, "
            f"{resolved.wanted_bytes / 1e9:.2f} GB, "
            f"plan {plan.files_in_flight} in flight x {plan.readers_per_file} readers, "
            f"pieces {plan.split_bytes >> 20} MiB, staging {plan.staging}",
            file=sys.stderr,
            flush=True,
        )
    target = engine_device(device)
    staged: dict[str, Staged] = {}
    files: list[FileJob] = []
    for path, picks in resolved.files:
        transfers: list[Transfer] = []
        for (entry, selection), reads in picks:
            staged[entry.name], more = stage(entry, selection, reads, target)
            transfers.extend(more)
        files.append((path, transfers))
    sync_torch(target)
    plan.execute(files, target)
    out = {name: s.finish() for name, s in staged.items()}
    if target != device:
        out = {name: tensor.to(device) for name, tensor in out.items()}
    return out


def gather(
    paths: Sequence[str],
    layouts: Sequence[Layout],
    keys: Sequence[str] | None,
    select: Mapping[str, SelectSpec] | None,
) -> list[tuple[str, list[Pick]]]:
    """Which tensors of which files a request touches, in data order per
    file, each with its selection resolved against the header. ``keys``
    restricts the request; ``select`` names must be loaded (a name outside
    ``keys`` is a [`SelectError`][], one in no file a ``KeyError``, as
    for ``keys``). Names are not checked for uniqueness across files here:
    ``load_files`` refuses a duplicate, ``explain`` describes it."""
    wanted = None if keys is None else set(keys)
    select = dict(select or {})
    if wanted is not None:
        extra = sorted(set(select) - wanted)
        if extra:
            raise SelectError(f"selections for tensors not in keys: {extra}")
    present = {name for layout in layouts for name in layout.tensors}
    missing = sorted(((set() if wanted is None else wanted) | set(select)) - present)
    if missing:
        raise KeyError(f"tensors not in any file: {missing}")
    out: list[tuple[str, list[Pick]]] = []
    for path, layout in zip(paths, layouts, strict=True):
        picks: list[Pick] = []
        for entry in layout.in_data_order():
            if wanted is not None and entry.name not in wanted:
                continue
            spec = select.get(entry.name)
            selection = (
                Selection.full(entry.shape)
                if spec is None
                else resolve(entry.name, entry.shape, spec)
            )
            picks.append((entry, selection))
        if picks:
            out.append((path, picks))
    return out


def load_files(
    filenames: Sequence[PathLike],
    device: torch.device,
    keys: Sequence[str] | None,
    select: Mapping[str, SelectSpec] | None = None,
) -> dict[str, torch.Tensor]:
    """Read ``keys`` (or every tensor) from ``filenames`` into fresh tensors on
    ``device``, each cut to its ``select`` entry, with as many files in
    flight as the plan says."""
    paths = [os.fspath(f) for f in filenames]
    layouts = read_layouts(paths)
    check_unique(paths, layouts)
    wanted = gather(paths, layouts, keys, select)
    if not wanted:
        return {}
    out = fill(wanted, device)
    return {name: out[name] for name in sorted(out)}


def explain(
    filenames: Sequence[PathLike],
    device: torch.device,
    keys: Sequence[str] | None = None,
    select: Mapping[str, SelectSpec] | None = None,
) -> str:
    """The plan for reading ``filenames`` (cut to ``keys`` and ``select``) to
    ``device``, in words; per selection, the bytes wanted against the bytes
    read; and what CUDA the machine has."""
    paths = [os.fspath(f) for f in filenames]
    if not paths:
        raise ValueError("explain needs at least one file")
    layouts = read_layouts(paths)
    resolved = Resolved.of(gather(paths, layouts, keys, select))
    per_path = dict(zip(resolved.paths, resolved.bytes_per_file, strict=True))
    to_read = [per_path.get(path, 0) for path in paths]
    total, wanted = sum(to_read), resolved.wanted_bytes
    classes = _core.storage_classes(paths)
    plan = Plan.for_read(
        paths,
        to_read,
        device,
        coalesced=resolved.summary,
        allocation_bytes=resolved.allocation_bytes,
    )
    staging = (
        "none"
        if plan.staging is None
        else f"{plan.staging[0]} x {plan.staging[1] >> 20} MiB pinned buffers"
    )
    volume = f"{total / 1e9:.3f} GB to read"
    if wanted != total:
        volume += f" for {wanted / 1e9:.3f} GB wanted"
    lines = [
        f"fastersafetensors: {len(paths)} file(s), {volume}, destination {device}",
        f"tensor allocations: {resolved.allocation_bytes} bytes peak "
        f"({wanted} result, {resolved.allocation_bytes - wanted} scratch)",
    ]
    for path, cls in zip(paths, classes, strict=True):
        lines.append(f"  {path}: storage class {cls}")
    if select:
        lines.append("selections:")
        for _, picks in resolved.files:
            for (entry, selection), reads in picks:
                if entry.name in select:
                    lines.append(f"  {_describe(entry, selection, reads)}")
    pieces = f"{plan.split_bytes >> 20} MiB pieces" if plan.split_bytes else "unsplit"
    lines += [
        f"plan: {plan.files_in_flight} files in flight, {plan.readers_per_file} reader(s) "
        f"per file, {pieces}, transport {plan.transport}, staging {staging}",
        "because:",
    ]
    lines += [f"  - {reason}" for reason in plan.reasons]
    cuda = _core.probe_cuda()
    lines.append("cuda:")
    for library in ("cudart", "cufile"):
        available, detail = cuda[library]
        lines.append(
            f"  {library}: {detail}" if available else f"  {library}: absent ({detail})"
        )
    lines.append(f"  devices: {cuda['devices']}")
    return "\n".join(lines)


def _describe(entry: TensorEntry, selection: Selection, reads: Reads) -> str:
    """One ``explain`` line: the cut, the bytes wanted, the bytes read."""
    runs = sum(1 if placements is None else len(placements) for *_, placements in reads)
    wanted = selection.wanted_bytes(entry.itemsize)
    read = bytes_read(reads)
    amplification = read / wanted if wanted else 1.0
    return (
        f"{entry.name}: {entry.shape} -> {selection.result_shape}, "
        f"{selection.wanted_bytes(entry.itemsize)} bytes wanted, {bytes_read(reads)} read "
        f"in {len(reads)} read(s) over {runs} run(s), "
        f"{amplification:.3f}x amplification, "
        f"{read / len(reads) if reads else 0:.0f} mean bytes/read"
    )
