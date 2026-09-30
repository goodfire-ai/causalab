"""Explicit replicated loads: stripe tensors over a group, then broadcast bytes.

Every member requests the same tensors and selections. Each tensor has one
reader (its owner) in the group: the request's bytes are cut into one
contiguous slice per rank, so every rank reads one long run per file (what
network storage readahead wants) and the owners of a request sit on every
node of a multi-node group. Owners hold only their slice; every other tensor
is allocated on a rank as its broadcast arrives and handed to the consumer at
once, so a request can be as large as the group is wide for the same device
memory, which is what keeps each rank's read job long enough to run at the
mount's rate. Tensors of at least ``DIRECT_BYTES`` are broadcast from their
own memory with several collectives in flight; smaller ones are packed per
owner into one bounded byte buffer, which also carries quantized/float8 dtypes
that a collective backend may not support directly.

A tensor named in ``shards`` is not replicated: each rank receives only its
own `Shard` of it, so the memory bounds count
it at shard size and no rank ever holds it whole. A shard along the outer
dimension is one contiguous run, read by the rank that wants it inside its
chunk jobs (`Direct`). A shard along an inner dimension is one short run
per leading index — on network storage many short reads, or reading every
rank through the whole tensor — so instead every rank reads a block of whole
rows (``1/world`` of the bytes, one run) and one all-to-all hands each rank
the piece of every row it wants (`Exchange`).
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import prod
from typing import Any

import torch
import torch.distributed as dist

from . import _files
from ._files import PathLike, Pick
from ._header import TensorEntry
from ._select import Selection, SelectSpec, Shard
from .errors import PlanError, ReadError, SelectError


def _env_optional_int(name: str, *, minimum: int = 0) -> int | None:
    """An integer tuning knob from the environment; unset (or empty) is
    ``None``. A value that is not an integer or is below ``minimum`` is a
    [`PlanError`][] naming the variable and the value, raised when the
    module is imported: a bare ``int()`` error names neither, and for the
    knobs in the request signature a wrong value on one node is a refusal
    someone has to diagnose from the message. ``minimum`` is 1 for counts and
    for sizes the code divides by; the two byte thresholds
    (``FASTERSAFETENSORS_INFLIGHT_GIB``, ``FASTERSAFETENSORS_COOPERATIVE_GIB``)
    accept 0 as a meaningful extreme (drain the window before every
    broadcast; read every whole tensor cooperatively)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise PlanError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise PlanError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = _env_optional_int(name, minimum=minimum)
    return default if value is None else value


def _env_gib(name: str, default: int, *, minimum: int = 0) -> int:
    return _env_int(name, default, minimum=minimum) << 30


def _env_optional_gib(name: str, *, minimum: int = 0) -> int | None:
    value = _env_optional_int(name, minimum=minimum)
    return None if value is None else value << 30


BROADCAST_BYTES: int = 64 << 20
"""Pack buffer for tensors below ``DIRECT_BYTES``, per owner."""
DIRECT_BYTES: int = 8 << 20
"""A tensor at least this large is broadcast from its own memory, no packing."""
IN_FLIGHT: int = _env_int("FASTERSAFETENSORS_INFLIGHT", 8, minimum=1)
"""Direct broadcasts outstanding before the oldest is waited for."""
IN_FLIGHT_BYTES: int = _env_gib("FASTERSAFETENSORS_INFLIGHT_GIB", 4)
"""Bytes of tensors outstanding in direct broadcasts before the oldest is
waited for. A receiving rank holds every tensor in flight whole (the consumer
narrows it afterwards), so the window is bounded by bytes as well as by count.

Whole tensors above ``COOPERATIVE_BYTES`` are not broadcast at all: every rank
allocates the destination, reads one contiguous ``1/world`` of its bytes
straight into it, and an in-place all-gather completes it (the last
``nbytes % world`` bytes each rank reads itself). No rank holds such a tensor
before its turn, the read is spread over every rank's storage link, and the
only memory beyond the destination is what the consumer still references.
A selection narrowed or stepped above that size is not the file's bytes in
order, so its owner reads it and broadcasts it: it enters this window, and a
request whose window does not fit beside the model is a [`PlanError`][]
from the fit check rather than a load that runs.
Nemotron-Ultra-253B has 14 FFN weights of 13-14 GB before tensor-parallel
narrowing beside 30 GB of parameters per rank on 80 GB GPUs; owner-broadcast
with any read-ahead of those did not fit, and reading them one owner at a
time cost 47 s of a 75 s load."""
CHUNK_TARGET_BYTES: int = _env_gib("FASTERSAFETENSORS_CHUNK_GIB", 4, minimum=1)
"""Bytes a rank's sequential read job aims for. Round ``c`` of a request
(every rank's chunk ``c``) is streamed while chunks ``c+1..`` are still being
read, so only the last round's broadcasts wait on the storage. More chunks
hide more of the stream but shorten each job, and a job ramps on network
storage: a 9 GB slice (Llama-70B, two nodes) was fastest as 4 jobs and slower
as 8; a 31 GB slice (Nemotron-Ultra per-rank shards) was fastest as 8 jobs
(29 s against 35 s as 4). ``chunk_count`` turns the target into a count."""
CHUNKS_MIN, CHUNKS_MAX = 4, 16
CHUNKS_OVERRIDE: int | None = _env_optional_int("FASTERSAFETENSORS_CHUNKS", minimum=1)
"""A fixed job count instead of the byte target (``FASTERSAFETENSORS_CHUNKS``)."""


def chunk_count(slice_bytes: int) -> int:
    """Sequential read jobs for a slice of ``slice_bytes``: one per
    ``CHUNK_TARGET_BYTES``, at least ``CHUNKS_MIN`` (an exposed stream is
    never worth less) and at most ``CHUNKS_MAX``; ``CHUNKS_OVERRIDE`` wins."""
    if CHUNKS_OVERRIDE is not None:
        return CHUNKS_OVERRIDE
    return max(CHUNKS_MIN, min(CHUNKS_MAX, -(-slice_bytes // CHUNK_TARGET_BYTES)))


READ_WORKERS: int = _env_int("FASTERSAFETENSORS_READ_WORKERS", 2, minimum=1)
COOPERATIVE_BYTES: int = _env_gib("FASTERSAFETENSORS_COOPERATIVE_GIB", 4)
"""A tensor selected whole above this size is not owned by any rank but
read cooperatively (below); it is also the floor of the derived read-ahead
budget."""
RESIDENT_OVERRIDE_BYTES: int | None = _env_optional_gib(
    "FASTERSAFETENSORS_RESIDENT_GIB", minimum=1
)
"""A fixed read-ahead budget instead of the one derived from the device's
headroom (``resident_budget``); on a device without measurable headroom
(CPU) it replaces ``RESIDENT_MAX_BYTES``. Parsed here with the other knobs,
so a bad value is refused at import on every backend."""
RESIDENT_MAX_BYTES: int = 16 << 30
"""Ceiling of the read-ahead budget: this rank's own chunks that may be read
but not yet streamed. A chunk is resident from the moment its job is
submitted until its round has been handed to the consumer, so read-ahead
(the next chunks, the next request) is bounded by bytes, not only by worker
count. The budget itself is set per load from the device's headroom (see
``resident_budget``): what is left after two of the largest tensors (the one
the consumer still references and the next one) and the in-flight window,
halved. Llama-70B on 80 GB gets the ceiling (three 2.2 GB chunks keep two
workers busy: cold 7.5 s against 9.5 s at 4 GiB); Nemotron-Ultra-253B, with
30 GB of parameters and 13 GB tensors, gets about 6 GiB."""
COOPERATIVE_READERS: int = 16
"""Readers per file for a rank's piece of a cooperatively read tensor: every
rank reads at once, and the pieces are one contiguous run each."""
TRACE: bool = os.environ.get("FASTERSAFETENSORS_TRACE", "") == "1"
"""Print per-phase timings of each coordinated call from group rank 0 to stderr."""
READERS_PER_FILE: int = _env_int("FASTERSAFETENSORS_COORDINATED_READERS", 4, minimum=1)
"""Readers per file for each rank's own reads. A coordinated rank reads a few
files' worth of tensors while the other ranks read theirs, so the profile's
single-request plan (one reader per file) leaves the storage link idle; four
readers per file on a 16-rank group kept an NFS mount at its measured ceiling."""


def exchange(group: dist.ProcessGroup, value: Any) -> list[Any]:
    values: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def check_errors(group: dist.ProcessGroup, error: Exception | None, phase: str) -> None:
    errors = exchange(
        group, None if error is None else f"{type(error).__name__}: {error}"
    )
    for rank, detail in enumerate(errors):
        if detail is not None:
            raise ReadError(
                f"coordinated {phase} failed on group rank {rank}: {detail}"
            ) from error


def fingerprint(
    wanted: Sequence[tuple[str, Sequence[Pick]]],
    device: torch.device,
    layouts: Mapping[str, ShardLayout],
) -> str:
    """What every member must agree on: the files, the tensors and their
    selections, for a sharded tensor the cut and the world, and the knobs
    that shape the collectives — the chunking ones, since the round count
    (one collective per round) comes from them, and ``COOPERATIVE_BYTES``,
    which decides whether a tensor is all-gathered or broadcast — not the
    rank, which is each member's own."""
    digest = hashlib.sha256(device.type.encode())
    knobs = (
        CHUNK_TARGET_BYTES,
        CHUNKS_OVERRIDE,
        CHUNKS_MIN,
        CHUNKS_MAX,
        COOPERATIVE_BYTES,
    )
    digest.update(repr(knobs).encode())
    for path, picks in wanted:
        digest.update(os.fsencode(os.path.abspath(path)))
        for entry, selection in picks:
            layout = layouts.get(entry.name)
            cut = None if layout is None else (layout.axis, layout.shard.world)
            digest.update(repr((entry, selection, cut)).encode())
    return digest.hexdigest()


def assign_owners(nbytes: Sequence[int], world: int) -> list[int]:
    """Owner rank per item, items given in file and data order: the request's
    bytes are cut into ``world`` contiguous slices and an item goes to the
    slice holding its midpoint. Each rank then reads one long sequential run
    per file rather than scattered tensors, which is what network storage
    readahead wants (0.5 GB/s per rank for scattered 190 MB tensors against
    1.5 GB/s for one run on the same mount), and the slices are balanced by
    bytes up to one item. Deterministic, so every member computes the same
    assignment from the same request."""
    total = sum(nbytes)
    if total == 0:
        return [min(i, world - 1) for i in range(len(nbytes))]
    owners: list[int] = []
    start = 0
    for size in nbytes:
        midpoint = start + size / 2
        owners.append(min(world - 1, int(midpoint * world // total)))
        start += size
    return owners


class _Packer:
    """Small tensors of one owner, packed into a shared buffer and broadcast
    together once the buffer is full, the owner changes or the items end."""

    def __init__(
        self, buffer: torch.Tensor, group: dist.ProcessGroup, rank: int
    ) -> None:
        self.buffer = buffer
        self.group = group
        self.rank = rank
        self.owner: int | None = None
        self.pending: list[tuple[str, torch.Tensor]] = []
        self.used = 0

    def add(
        self, owner: int, name: str, tensor: torch.Tensor
    ) -> Iterator[tuple[str, torch.Tensor]]:
        flat = tensor.reshape(-1).view(torch.uint8)
        if flat.numel() > self.buffer.numel():
            raise PlanError("packed tensor larger than the broadcast buffer")
        if self.owner != owner or self.used + flat.numel() > self.buffer.numel():
            yield from self.flush()
        self.owner = owner
        self.pending.append((name, tensor))
        self.used += flat.numel()

    def flush(self) -> Iterator[tuple[str, torch.Tensor]]:
        if not self.pending:
            return
        owner = self.owner
        assert owner is not None
        pending, used = self.pending, self.used
        self.pending, self.used, self.owner = [], 0, None
        source = dist.get_global_rank(self.group, owner)
        if self.rank == owner:
            at = 0
            for _, tensor in pending:
                flat = tensor.reshape(-1).view(torch.uint8)
                self.buffer[at : at + flat.numel()].copy_(flat)
                at += flat.numel()
        dist.broadcast(self.buffer[:used], src=source, group=self.group)
        if self.rank != owner:
            at = 0
            for _, tensor in pending:
                flat = tensor.reshape(-1).view(torch.uint8)
                flat.copy_(self.buffer[at : at + flat.numel()])
                at += flat.numel()
        yield from pending


def assign_chunks(nbytes: Sequence[int], chunks: int) -> list[int]:
    """Chunk index per item of one rank's slice, items in order: consecutive
    items are grouped until a group reaches the slice's bytes divided by
    ``chunks``, and an item larger than that target is a chunk of its own.
    The count of chunks therefore varies with the slice (at least one, more
    than ``chunks`` when oversized tensors are present), so that no chunk
    holds two tensors that each dwarf the target."""
    total = sum(nbytes)
    if not nbytes:
        return []
    target = max(1, -(-total // chunks))
    out: list[int] = []
    chunk, filled = 0, 0
    for size in nbytes:
        if filled and filled + size > target:
            chunk += 1
            filled = 0
        out.append(chunk)
        filled += size
    return out


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """Where a shard's bytes lie. The tensor is ``rows`` x ``cols`` elements
    (``rows`` the product of the dimensions before the cut, ``cols`` the
    product from the cut on), every row holds ``world`` pieces of ``piece``
    elements, and the shard is piece ``shard.rank`` of every row. With one
    row (a cut along the outer dimension, or leading dimensions of extent
    one) the shard is one contiguous run."""

    shard: Shard
    axis: int
    """``shard.dim`` as a non-negative index."""
    rows: int
    cols: int
    piece: int
    itemsize: int

    @classmethod
    def of(cls, shape: Sequence[int], itemsize: int, shard: Shard) -> ShardLayout:
        """The layout of ``shard`` over a tensor of ``shape``; a cut that does
        not divide, or a ``dim`` outside the shape, is a [`SelectError`][]."""
        shard.ranges(shape)  # the core's validation: dim, rank, divisibility
        axis = shard.axis(len(shape))
        cols = prod(shape[axis:])
        return cls(shard, axis, prod(shape[:axis]), cols, cols // shard.world, itemsize)

    @property
    def direct(self) -> bool:
        return self.rows <= 1

    @property
    def row_bytes(self) -> int:
        return self.cols * self.itemsize

    @property
    def piece_bytes(self) -> int:
        return self.piece * self.itemsize

    @property
    def nbytes(self) -> int:
        """Bytes of the shard itself."""
        return self.rows * self.piece_bytes

    def block_bytes(self, block: tuple[int, int]) -> int:
        lo, hi = block
        return (hi - lo) * self.row_bytes


def row_blocks(rows: int, world: int) -> list[tuple[int, int]]:
    """``world`` contiguous row ranges covering ``[0, rows)``, the first
    ``rows % world`` of them one row longer (``torch.tensor_split``)."""
    base, extra = divmod(rows, world)
    out: list[tuple[int, int]] = []
    start = 0
    for q in range(world):
        end = start + base + (1 if q < extra else 0)
        out.append((start, end))
        start = end
    return out


def exchange_input(
    block: torch.Tensor, layout: ShardLayout, wanted: Sequence[int]
) -> torch.Tensor:
    """What a rank holding ``block`` (whole rows of the tensor, as bytes)
    sends in the all-to-all: for every group rank in order, piece
    ``wanted[i]`` of each of the block's rows, packed. One gather copy."""
    if block.numel() == 0:
        return block.new_empty(0)
    rows = block.numel() // layout.row_bytes
    pieces = block.view(rows, layout.shard.world, layout.piece_bytes)
    return pieces.permute(1, 0, 2)[list(wanted)].reshape(-1)


@dataclass(frozen=True, slots=True)
class Broadcast:
    """Replicated: ``owner`` reads it whole and broadcasts it."""

    owner: int


@dataclass(frozen=True, slots=True)
class Cooperative:
    """Replicated, selected whole and above ``COOPERATIVE_BYTES``: every
    rank reads ``1/world`` of the bytes into its own copy at its turn, an
    all-gather completes it. ``turn`` is the owner sequence whose chunking
    places it."""

    turn: int


@dataclass(frozen=True, slots=True)
class Direct:
    """Sharded, one contiguous run: this rank reads its shard in its chunk jobs."""

    layout: ShardLayout


@dataclass(frozen=True, slots=True)
class Exchange:
    """Sharded along an inner dimension: group rank ``q`` reads rows
    ``blocks[q]`` whole in its chunk jobs; an all-to-all then delivers to
    each rank ``i`` piece ``wanted[i]`` of every row."""

    layout: ShardLayout
    blocks: tuple[tuple[int, int], ...]
    wanted: tuple[int, ...]


Delivery = Broadcast | Cooperative | Direct | Exchange


@dataclass(frozen=True, slots=True)
class Item:
    """One tensor of a coordinated request: what this rank receives
    (``selection`` of ``entry``) and how it gets there."""

    path: str
    entry: TensorEntry
    selection: Selection
    delivery: Delivery

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def nbytes(self) -> int:
        """Bytes delivered to this rank."""
        return self.selection.wanted_bytes(self.entry.itemsize)

    def pick(self, rank: int) -> Pick | None:
        """What group rank ``rank`` reads for this item in its chunk jobs, if anything."""
        match self.delivery:
            case Broadcast(owner):
                return (self.entry, self.selection) if owner == rank else None
            case Cooperative():
                return None
            case Direct():
                return (self.entry, self.selection)
            case Exchange(layout, blocks, _):
                shape = (layout.rows, layout.cols)
                block = Selection.from_ranges(shape, [blocks[rank], (0, layout.cols)])
                return (replace(self.entry, shape=shape), block)
        raise AssertionError(self.delivery)

    def job_bytes(self, rank: int) -> int:
        pick = self.pick(rank)
        return 0 if pick is None else pick[1].wanted_bytes(self.entry.itemsize)

    @property
    def nominal_job_bytes(self) -> int:
        """Bytes a rank reads for this item, the same number on every rank:
        what the rank-invariant chunk assignment is made from."""
        match self.delivery:
            case Broadcast() | Direct():
                return self.nbytes
            case Cooperative():
                return 0
            case Exchange(layout, blocks, _):
                return layout.block_bytes(blocks[0])
        raise AssertionError(self.delivery)


def plan_items(
    wanted: Sequence[tuple[str, Sequence[Pick]]],
    shards: Mapping[str, Shard],
    exchanged: Mapping[str, Sequence[int]],
    world: int,
    *,
    cooperative_bytes: int,
) -> list[Item]:
    """The items of a request, in file and data order; a pure function of the
    agreed request, so every group rank plans the same items. A name in
    ``shards`` is delivered as that shard; ``exchanged`` gives, for each such
    name whose shard is not one run, the shard index every group rank wants.
    Owners of the replicated tensors are balanced over the replicated bytes
    alone: sharded and cooperative tensors cost every rank the same."""
    picks = [(path, entry, selection) for path, ps in wanted for entry, selection in ps]
    layouts = {
        entry.name: ShardLayout.of(entry.shape, entry.itemsize, shards[entry.name])
        for _, entry, _ in picks
        if entry.name in shards
    }
    replicated = [
        i for i, (_, entry, _) in enumerate(picks) if entry.name not in layouts
    ]
    sizes = [picks[i][2].wanted_bytes(picks[i][1].itemsize) for i in replicated]
    owners = dict(zip(replicated, assign_owners(sizes, world), strict=True))
    items: list[Item] = []
    for i, (path, entry, selection) in enumerate(picks):
        layout = layouts.get(entry.name)
        delivery: Delivery
        if layout is None:
            nbytes = selection.wanted_bytes(entry.itemsize)
            owner = owners[i]
            # A cooperative read lands each rank's piece by file offset, so it
            # needs the result to be the tensor's bytes in storage order; a
            # narrowed or stepped selection is read by its owner and broadcast.
            whole = selection.is_plain and nbytes == entry.nbytes
            delivery = (
                Cooperative(owner)
                if whole and nbytes > cooperative_bytes
                else Broadcast(owner)
            )
        else:
            selection = Selection.from_ranges(
                entry.shape, layout.shard.ranges(entry.shape)
            )
            if layout.direct:
                delivery = Direct(layout)
            else:
                delivery = Exchange(
                    layout,
                    tuple(row_blocks(layout.rows, world)),
                    tuple(exchanged[entry.name]),
                )
        items.append(Item(path, entry, selection, delivery))
    return items


class _Prepared:
    """One coordinated request after the group agreed on it: the items in a
    fixed order, the chunk each belongs to, the stream order (round by
    round), and this rank's chunk jobs."""

    __slots__ = (
        "broadcast_largest",
        "buffer_bytes",
        "chunk_bytes",
        "chunk_of",
        "device",
        "exchange_bytes",
        "items",
        "largest",
        "n_rounds",
        "owned_chunks",
        "rank",
        "rounds",
        "sizes",
        "total",
        "world",
    )

    def __init__(
        self, items: list[Item], device: torch.device, rank: int, world: int
    ) -> None:
        self.device = device
        self.rank = rank
        self.world = world
        self.items = items
        self.sizes = [item.nbytes for item in items]
        # Each rank's slice is cut into sequential jobs of about
        # CHUNK_TARGET_BYTES (chunk_count); round c streams every rank's chunk
        # c, so the order is the same on every member. Ranks whose slice holds
        # oversized tensors have more chunks; the others' later rounds are
        # empty for them. The replicated tensors are chunked per owner
        # (cooperative ones take their turn in their owner's sequence); the
        # sharded ones, read by every rank alike, form one more sequence with
        # the same chunking on every rank (their bytes are the same everywhere).
        self.chunk_of = [0] * len(items)
        self.n_rounds = 1
        sequences = [
            [
                i
                for i, item in enumerate(items)
                if isinstance(item.delivery, Direct | Exchange)
            ]
        ]
        for owner in range(world):
            sequences.append(
                [i for i, item in enumerate(items) if _turn(item) == owner]
            )
        for indices in sequences:
            job_bytes = [items[i].nominal_job_bytes for i in indices]
            chunks = assign_chunks(job_bytes, chunk_count(sum(job_bytes)))
            for i, chunk in zip(indices, chunks, strict=True):
                self.chunk_of[i] = chunk
            self.n_rounds = max(self.n_rounds, max(chunks, default=0) + 1)
        self.rounds: list[list[int]] = [
            [i for i in range(len(items)) if self.chunk_of[i] == c]
            for c in range(self.n_rounds)
        ]
        self.owned_chunks: list[list[tuple[str, list[Pick]]]] = []
        self.chunk_bytes: list[int] = []
        for c in range(self.n_rounds):
            job: list[tuple[str, list[Pick]]] = []
            for i in self.rounds[c]:
                pick = items[i].pick(rank)
                if pick is None:
                    continue
                if job and job[-1][0] == items[i].path:
                    job[-1][1].append(pick)
                else:
                    job.append((items[i].path, [pick]))
            self.owned_chunks.append(job)
            self.chunk_bytes.append(
                sum(items[i].job_bytes(rank) for i in self.rounds[c])
            )
        self.total = sum(self.sizes)
        self.largest = max(self.sizes, default=0)
        self.broadcast_largest = max(
            (item.nbytes for item in items if isinstance(item.delivery, Broadcast)),
            default=0,
        )
        self.exchange_bytes = max(
            (
                item.delivery.layout.block_bytes(item.delivery.blocks[rank])
                for item in items
                if isinstance(item.delivery, Exchange)
            ),
            default=0,
        )
        self.buffer_bytes = min(BROADCAST_BYTES, max(self.total, 1))

    @property
    def owned_bytes(self) -> int:
        return sum(self.chunk_bytes)

    @property
    def owned_files(self) -> int:
        return len({path for job in self.owned_chunks for path, _ in job})

    @property
    def sharded_names(self) -> set[str]:
        return {
            item.name
            for item in self.items
            if isinstance(item.delivery, Direct | Exchange)
        }


def _turn(item: Item) -> int | None:
    """The owner sequence a replicated item is chunked in; ``None`` for a sharded one."""
    match item.delivery:
        case Broadcast(owner):
            return owner
        case Cooperative(turn):
            return turn
        case _:
            return None


def check_shards(
    shards: Mapping[str, Shard],
    keys: Sequence[str] | None,
    select: Mapping[str, SelectSpec] | None,
    present: set[str],
) -> dict[str, Shard]:
    """The entries of ``shards`` this request loads: a name that is present
    but outside ``keys``, or also in ``select``, is a [`SelectError`][];
    a name in no file of the request is left to the caller."""
    both = sorted(set(shards) & set(select or {}))
    if both:
        raise SelectError(f"tensors in both select and shards: {both}")
    loaded = present if keys is None else present & set(keys)
    outside = sorted((set(shards) & present) - loaded)
    if outside:
        raise SelectError(f"shards for tensors not in keys: {outside}")
    return {name: shard for name, shard in shards.items() if name in loaded}


def _prepare(
    filenames: Sequence[PathLike],
    device: torch.device,
    keys: Sequence[str] | None,
    select: Mapping[str, SelectSpec] | None,
    shards: Mapping[str, Shard],
    group: dist.ProcessGroup,
    *,
    all_shards: bool,
) -> _Prepared:
    """Resolve the request and agree on it across the group (one collective).
    ``all_shards`` requires every name of ``shards`` to be in the request
    (``load_files``); a stream checks that over all its requests."""
    if not dist.is_initialized() or dist.get_rank(group) < 0:
        raise PlanError("group must be initialized and this process must be a member")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    wanted: list[tuple[str, list[Pick]]] = []
    layouts: dict[str, ShardLayout] = {}
    mine: dict[str, Shard] = {}
    error: Exception | None = None
    try:
        backend = dist.get_backend(group)
        if (device.type, backend) not in (("cpu", "gloo"), ("cuda", "nccl")):
            raise PlanError("coordinated loads support CPU/Gloo or CUDA/NCCL")
        if device.type == "cuda":
            index = _files.cuda_index(device)
            if index != torch.cuda.current_device():
                raise PlanError(
                    "set the current CUDA device to the load destination before using group"
                )
            # Pin the ordinal now: the current device is per thread, and the
            # read may run on a helper thread where a bare ``cuda`` means GPU 0.
            device = torch.device("cuda", index)
        paths = sorted(os.fspath(path) for path in filenames)
        file_layouts = _files.read_layouts(paths)
        _files.check_unique(paths, file_layouts)
        present = {name for layout in file_layouts for name in layout.tensors}
        if all_shards:
            missing = sorted(set(shards) - present)
            if missing:
                raise KeyError(f"shards for tensors not in any file: {missing}")
        mine = check_shards(shards, keys, select, present)
        wanted = _files.gather(paths, file_layouts, keys, select)
        layouts = {
            entry.name: ShardLayout.of(entry.shape, entry.itemsize, mine[entry.name])
            for _, picks in wanted
            for entry, _ in picks
            if entry.name in mine
        }
    except Exception as exc:
        error = exc
    # One collective carries the error report, the request signature and the
    # shard index this rank wants of every tensor that is redistributed.
    exchanged = [layout.shard.rank for layout in layouts.values() if not layout.direct]
    reports = exchange(
        group,
        (
            None if error is None else f"{type(error).__name__}: {error}",
            fingerprint(wanted, device, layouts),
            exchanged,
        ),
    )
    for other, (detail, _, _) in enumerate(reports):
        if detail is not None:
            raise ReadError(
                f"coordinated preparation failed on group rank {other}: {detail}"
            ) from error
    if len({signature for _, signature, _ in reports}) != 1:
        raise PlanError(
            "coordinated loads require identical files, keys, selections and shard cuts "
            "on every rank"
        )
    names = [name for name, layout in layouts.items() if not layout.direct]
    wanted_by_rank = {
        name: tuple(report[2][i] for report in reports) for i, name in enumerate(names)
    }
    items = plan_items(
        wanted, mine, wanted_by_rank, world, cooperative_bytes=COOPERATIVE_BYTES
    )
    return _Prepared(items, device, rank, world)


def _read_chunk(
    prepared: _Prepared, chunk: int
) -> tuple[dict[str, torch.Tensor], float]:
    """Read one of this rank's chunk jobs with the ordinary engine. No
    collectives, so it runs on a helper thread while earlier rounds stream."""
    t0 = time.perf_counter()
    device = prepared.device
    index = _files.cuda_index(device)
    if index is not None:
        torch.cuda.set_device(index)  # the current device is per thread
    job = prepared.owned_chunks[chunk]
    out = _files.fill(job, device, readers_per_file=READERS_PER_FILE) if job else {}
    return out, time.perf_counter() - t0


def _all_gather_pieces(
    flat: torch.Tensor, piece: int, rank: int, world: int, group: dist.ProcessGroup
) -> None:
    """Complete ``flat[: world * piece]`` from each rank's ``piece`` bytes at
    ``rank * piece``: in place over NCCL; Gloo has no in-place all-gather, so
    there the pieces travel through a list (tests run on Gloo)."""
    mine = flat[rank * piece : (rank + 1) * piece]
    if dist.get_backend(group) == "nccl":
        dist.all_gather_into_tensor(flat[: world * piece], mine, group=group)
        return
    pieces = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(pieces, mine.clone(), group=group)
    for r, got in enumerate(pieces):
        flat[r * piece : (r + 1) * piece].copy_(got)


def _exchange(
    tensor: torch.Tensor,
    block: torch.Tensor,
    layout: ShardLayout,
    blocks: Sequence[tuple[int, int]],
    wanted: Sequence[int],
    rank: int,
    group: dist.ProcessGroup,
) -> None:
    """Fill ``tensor`` (this rank's shard) from every rank's row ``block``
    with one all-to-all: rank ``q`` sends each rank ``i`` piece ``wanted[i]``
    of its rows, and the pieces arrive in row order because the blocks are."""
    flat = tensor.reshape(-1).view(torch.uint8)
    rows = blocks[rank][1] - blocks[rank][0]
    sent = exchange_input(block.reshape(-1).view(torch.uint8), layout, wanted)
    dist.all_to_all_single(
        flat,
        sent,
        [layout.piece_bytes * (hi - lo) for lo, hi in blocks],
        [layout.piece_bytes * rows] * len(blocks),
        group=group,
    )


def _read_piece(path: str, transfers: list, device: torch.device) -> None:
    """Land byte ranges of ``path`` at the addresses in ``transfers`` (this
    rank's share of a cooperatively read tensor) with every reader the plan
    allows for one file."""
    target = _files.engine_device(device)
    nbytes = sum(t[1] for t in transfers)
    plan = _files.Plan.for_read(
        [path], [nbytes], device, check_fit=False
    ).with_concurrency(COOPERATIVE_READERS, 1)
    _files.sync_torch(target)
    plan.execute([(path, transfers)], target)


def _check_fit(prepared: _Prepared) -> None:
    """Beside the model the device must hold the resident chunks (or one
    oversized chunk), the copy an exchange packs beside its row block, the
    tensors in flight (or the largest broadcast tensor), the largest
    delivered tensor twice (the one the consumer still references while the
    next one is allocated), and the pack buffer. Nemotron-Ultra-253B at TP16
    on 80 GB: 30 GB of parameters, 6 GB outside torch, then 4 + 13 + 13 GiB
    of this with whole tensors; sharded, the 13 GiB terms become the shards.
    The resident term is what the scheduler will enforce (``_resident_floor``):
    the derived budget's floor, or an explicit ``RESIDENT_OVERRIDE_BYTES``,
    which bypasses the derived clamp and so must be reserved here rather than
    discovered as an OOM mid-load. A small override therefore also makes the
    check more permissive: it reserves only what the scheduler will hold."""
    index = _files.cuda_index(prepared.device)
    if index is None:
        return
    largest = prepared.largest
    resident = max(max(prepared.chunk_bytes, default=0), _resident_floor())
    in_flight = _in_flight_bytes(prepared.broadcast_largest)
    needed = (
        resident
        + prepared.exchange_bytes
        + in_flight
        + 2 * largest
        + prepared.buffer_bytes
    )
    if needed > _files.device_headroom(index):
        term = (
            "resident chunks"
            if RESIDENT_OVERRIDE_BYTES is None
            else "the FASTERSAFETENSORS_RESIDENT_GIB read-ahead budget"
        )
        raise PlanError(
            f"{term} ({resident >> 30} GiB), the tensors in flight "
            f"({in_flight >> 30} GiB) and two of the largest tensor ({largest >> 30} GiB: "
            f"one held by the consumer, one arriving) do not fit on the device"
        )


def _resident_floor() -> int:
    """The least the scheduler may hold resident: the derived budget's floor
    (``COOPERATIVE_BYTES``), or the explicit override, which wins outright."""
    return (
        COOPERATIVE_BYTES
        if RESIDENT_OVERRIDE_BYTES is None
        else RESIDENT_OVERRIDE_BYTES
    )


def _budget_for(prepared: _Prepared) -> int:
    index = _files.cuda_index(prepared.device)
    if index is None:
        # no headroom to derive from, but an explicit budget needs no derivation
        return (
            RESIDENT_MAX_BYTES if RESIDENT_OVERRIDE_BYTES is None else _resident_floor()
        )
    return resident_budget(
        _files.device_headroom(index),
        prepared.largest,
        prepared.broadcast_largest,
        prepared.buffer_bytes + prepared.exchange_bytes,
    )


def resident_budget(
    headroom: int, largest: int, broadcast_largest: int, buffer_bytes: int
) -> int:
    """Read-ahead budget for a device with ``headroom`` bytes beside the
    model: half of what two ``largest`` delivered tensors, the in-flight
    window (sized by the largest broadcast tensor) and ``buffer_bytes`` leave,
    clamped to ``[COOPERATIVE_BYTES, RESIDENT_MAX_BYTES]``.
    ``RESIDENT_OVERRIDE_BYTES`` (``FASTERSAFETENSORS_RESIDENT_GIB``) overrides it."""
    if RESIDENT_OVERRIDE_BYTES is not None:
        return RESIDENT_OVERRIDE_BYTES
    left = headroom - 2 * largest - _in_flight_bytes(broadcast_largest) - buffer_bytes
    return max(COOPERATIVE_BYTES, min(RESIDENT_MAX_BYTES, left // 2))


def _in_flight_bytes(broadcast_largest: int) -> int:
    """What the direct-broadcast window can hold: the byte budget, or the
    largest broadcast tensor when that travels alone above it. Cooperative
    and sharded tensors never enter the window."""
    return max(broadcast_largest, IN_FLIGHT_BYTES)


class _Reads:
    """This rank's chunk jobs, submitted in order as workers free up and the
    resident budget allows. Owned by the streaming thread; the done callbacks
    of the jobs run on worker threads, hence the lock."""

    def __init__(self, pool: ThreadPoolExecutor, budget: int) -> None:
        self.pool = pool
        self.budget = budget
        self.queue: deque[tuple[_Prepared, int]] = deque()
        self.futures: dict[tuple[_Prepared, int], Future] = {}
        self.resident = 0
        self.running = 0
        self.failure: Exception | None = None
        """What ``pump`` raised off the streaming thread; ``take`` raises it."""
        self.changed = threading.Condition()

    def add(self, prepared: _Prepared) -> None:
        for c in range(prepared.n_rounds):
            self.queue.append((prepared, c))
        self.pump()

    def tighten(self, budget: int) -> None:
        """Lower the read-ahead budget; it never grows, as headroom only
        shrinks while the caller keeps the tensors it is handed."""
        with self.changed:
            self.budget = min(self.budget, budget)

    def _finished(self, _: Future) -> None:
        with self.changed:
            self.running -= 1
            self.changed.notify_all()
        # Refill the freed worker at once (on the worker's thread): waiting
        # for the streaming thread to pump would idle it through the next
        # round's collective and stream.
        self._pump_recording()

    def _pump_recording(self) -> None:
        """``pump`` where an exception would be lost or land between two
        collectives: the executor drops what a done callback raises, and a
        raise from ``release`` would leave this rank out of the next round.
        A failure is kept for ``take`` instead, which every rank reports
        through the round's error collective."""
        try:
            self.pump()
        except Exception as exc:
            with self.changed:
                self.failure = exc
                self.changed.notify_all()

    def pump(self) -> None:
        """Submit queued jobs while a worker is free and the chunk fits the
        resident budget (a chunk always fits when nothing is resident)."""
        with self.changed:
            while self.queue and self.running < READ_WORKERS:
                prepared, c = self.queue[0]
                nbytes = prepared.chunk_bytes[c]
                if self.resident and self.resident + nbytes > self.budget:
                    return
                future = self.pool.submit(_read_chunk, prepared, c)
                # Account only for a job that was submitted: a raise above
                # leaves the queue and the counters as they were.
                self.queue.popleft()
                self.resident += nbytes
                self.running += 1
                self.futures[(prepared, c)] = future
                future.add_done_callback(self._finished)

    def take(self, prepared: _Prepared, c: int) -> Future:
        """The job for chunk ``c``, submitting it first if it is still queued.
        Raises what ``pump`` raised on a worker thread, so a scheduling
        failure ends the load like a read failure instead of stalling it."""
        key = (prepared, c)
        while True:
            self.pump()
            with self.changed:
                if self.failure is not None:
                    raise self.failure
                future = self.futures.pop(key, None)
                if future is not None:
                    return future
                # a worker or the budget is holding the queue; both change
                # when a job finishes or a round is released
                self.changed.wait(timeout=1.0)

    def release(self, prepared: _Prepared, c: int) -> None:
        """Chunk ``c`` has been handed to the consumer: its bytes are no longer resident."""
        with self.changed:
            self.resident -= prepared.chunk_bytes[c]
            self.changed.notify_all()
        self._pump_recording()


def _stream(
    prepared: _Prepared,
    reads: _Reads | None,
    submit_error: Exception | None,
    group: dist.ProcessGroup,
    prepare_s: float,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Hand out every tensor of the request round by round: round ``c`` is
    every rank's chunk ``c``. Before a round, every member reports whether its
    chunk landed (one collective), so a read error on any rank surfaces on all
    before anyone enters a broadcast that could never complete. The owner's
    tensors come from its chunk; everyone else allocates a tensor and the
    broadcast fills it. Up to ``IN_FLIGHT`` direct broadcasts are outstanding;
    a tensor is yielded once its bytes have landed."""
    t0 = time.perf_counter()
    device, rank = prepared.device, prepared.rank
    allocated_before = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    buffer = torch.empty(prepared.buffer_bytes, dtype=torch.uint8, device=device)
    packer = _Packer(buffer, group, rank)
    inflight: deque[tuple[str, torch.Tensor, dist.Work]] = deque()
    inflight_bytes = 0
    read_s: list[float] = []
    round_s: list[float] = []
    coop_s: list[float] = []
    exchange_s: list[float] = []
    sharded = 0
    for c in range(prepared.n_rounds):
        error = submit_error
        mine: dict[str, torch.Tensor] = {}
        if error is None:
            assert reads is not None
            try:
                mine, seconds = reads.take(prepared, c).result()
                read_s.append(seconds)
            except Exception as exc:
                error = exc
        check_errors(group, error, f"read (chunk {c})")
        t_round = time.perf_counter()
        for i in prepared.rounds[c]:
            item = prepared.items[i]
            path, entry, selection = item.path, item.entry, item.selection
            name = entry.name
            match item.delivery:
                case Cooperative():
                    # Cooperative read: every rank lands its 1/world of the bytes in
                    # its own destination, then an in-place all-gather completes it.
                    # ``plan_items`` chose this only for a whole, plain selection,
                    # so the destination's bytes are the file's from ``entry.start``.
                    # The blocks of the previous such tensors sit freed but cached in
                    # pieces that need not add up to this one, so the cache goes first.
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    t_coop = time.perf_counter()
                    tensor = torch.empty(
                        selection.result_shape, dtype=entry.torch_dtype, device=device
                    )
                    flat = tensor.reshape(-1).view(torch.uint8)
                    nbytes = flat.numel()
                    piece = nbytes // prepared.world
                    transfers = []
                    if piece:
                        transfers.append(
                            _files.transfer(
                                entry.start + rank * piece,
                                flat[rank * piece : (rank + 1) * piece],
                            )
                        )
                    if nbytes % prepared.world:
                        tail = prepared.world * piece
                        transfers.append(
                            _files.transfer(entry.start + tail, flat[tail:])
                        )
                    error = None
                    try:
                        _read_piece(path, transfers, device)
                    except Exception as exc:
                        error = exc
                    check_errors(group, error, f"read (tensor {name})")
                    if piece:
                        _all_gather_pieces(flat, piece, rank, prepared.world, group)
                    coop_s.append(time.perf_counter() - t_coop)
                    yield name, tensor
                    del tensor, flat
                    continue
                case Direct():
                    # This rank's shard, read into its final tensor by the chunk job.
                    sharded += 1
                    yield name, mine.pop(name)
                    continue
                case Exchange(layout, blocks, wanted):
                    sharded += 1
                    t_exchange = time.perf_counter()
                    block = mine.pop(name)
                    tensor = torch.empty(
                        selection.result_shape, dtype=entry.torch_dtype, device=device
                    )
                    if tensor.numel():
                        _exchange(tensor, block, layout, blocks, wanted, rank, group)
                    del block
                    exchange_s.append(time.perf_counter() - t_exchange)
                    yield name, tensor
                    del tensor
                    continue
                case Broadcast(owner):
                    pass
            tensor = (
                mine.pop(name)
                if owner == rank
                else torch.empty(
                    selection.result_shape, dtype=entry.torch_dtype, device=device
                )
            )
            flat = tensor.reshape(-1).view(torch.uint8)
            nbytes = flat.numel()
            if nbytes == 0:
                yield name, tensor
                del tensor, flat
                continue
            if nbytes < DIRECT_BYTES:
                del flat
                yield from packer.add(owner, name, tensor)
                del tensor
                continue
            # Make room first: a receiving rank allocated this tensor whole, so the
            # window is bounded by bytes as well as by count. References to a
            # yielded tensor are dropped at once: the consumer's loop variable
            # already keeps the previous tensor alive while this one is
            # allocated, and a third copy of a 14 GB tensor is what does not fit.
            while inflight and (
                len(inflight) >= IN_FLIGHT or inflight_bytes + nbytes > IN_FLIGHT_BYTES
            ):
                done_name, done_tensor, work = inflight.popleft()
                inflight_bytes -= done_tensor.numel() * done_tensor.element_size()
                work.wait()
                yield done_name, done_tensor
                del done_tensor
            source = dist.get_global_rank(group, owner)
            inflight.append(
                (
                    name,
                    tensor,
                    dist.broadcast(flat, src=source, group=group, async_op=True),
                )
            )
            inflight_bytes += nbytes
            del tensor, flat
        round_s.append(time.perf_counter() - t_round)
        if reads is not None:
            reads.release(prepared, c)
    yield from packer.flush()
    while inflight:
        name, tensor, work = inflight.popleft()
        work.wait()
        yield name, tensor
        del tensor
    _files.sync_torch(device)
    if TRACE and rank == 0:
        allocated_now = (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        )
        print(
            f"fastersafetensors coordinated: {len(prepared.items)} tensors, "
            f"{prepared.total / 1e9:.2f} GB total, rank 0 read {prepared.owned_bytes / 1e9:.2f} GB "
            f"of {prepared.owned_files} files in {prepared.n_rounds} rounds "
            f"[{' '.join(f'{x:.2f}' for x in read_s)}] s; prepare {prepare_s:.2f} s, "
            f"rounds [{' '.join(f'{x:.2f}' for x in round_s)}] s, "
            f"{len(coop_s)} cooperative tensors {sum(coop_s):.2f} s, "
            f"{sharded} sharded tensors ({len(exchange_s)} exchanged, {sum(exchange_s):.2f} s), "
            f"budget {(reads.budget >> 30) if reads is not None else 0} GiB, "
            f"total stream {time.perf_counter() - t0:.2f} s, done at {time.time():.2f}; "
            f"torch allocated {allocated_before / 2**30:.1f} -> {allocated_now / 2**30:.1f} GiB",
            file=sys.stderr,
            flush=True,
        )


def stream_files(
    requests: Sequence[tuple[Sequence[PathLike], Sequence[str] | None]],
    device: torch.device,
    shards: Mapping[str, Shard],
    group: dist.ProcessGroup,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Coordinated loads of several requests, streamed and pipelined: each
    rank's slice of a request is read as ``chunk_count`` jobs, ``READ_WORKERS`` at
    a time within the read-ahead budget (``resident_budget``), and the request streams
    round by round as the chunks land; the next request's jobs queue behind.
    Every member passes the same requests; the collectives stay on the
    calling thread, in the same order on every rank, and a read error on any
    rank surfaces on all. ``shards`` applies to whichever request loads each
    name; a name no request loads is a [`SelectError`][] at the end."""
    unseen = set(shards)
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        reads: _Reads | None = None
        pending: tuple[_Prepared, Exception | None, float] | None = None
        for filenames, keys in requests:
            t0 = time.perf_counter()
            prepared = _prepare(
                filenames, device, keys, None, shards, group, all_shards=False
            )
            unseen -= prepared.sharded_names
            prepare_s = time.perf_counter() - t0
            error: Exception | None = None
            if prepared.items:
                try:
                    _check_fit(prepared)
                    if reads is None:
                        reads = _Reads(pool, _budget_for(prepared))
                    else:
                        # Headroom only shrinks as the model fills (the caller may
                        # allocate parameters as tensors arrive), so the budget is
                        # re-derived per request and never grows.
                        reads.tighten(_budget_for(prepared))
                    reads.add(prepared)
                except Exception as exc:
                    error = exc
            if pending is not None:
                yield from _stream(pending[0], reads, pending[1], group, pending[2])
            pending = (prepared, error, prepare_s) if prepared.items else None
        if pending is not None:
            yield from _stream(pending[0], reads, pending[1], group, pending[2])
    if unseen:
        raise SelectError(f"shards for tensors no request loaded: {sorted(unseen)}")


def load_files(
    filenames: Sequence[PathLike],
    device: torch.device,
    keys: Sequence[str] | None,
    select: Mapping[str, SelectSpec] | None,
    shards: Mapping[str, Shard],
    group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    t0 = time.perf_counter()
    prepared = _prepare(filenames, device, keys, select, shards, group, all_shards=True)
    if not prepared.items:
        return {}
    prepare_s = time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        reads: _Reads | None = None
        error: Exception | None = None
        try:
            _check_fit(prepared)
            reads = _Reads(pool, _budget_for(prepared))
            reads.add(prepared)
        except Exception as exc:
            error = exc
        out = dict(_stream(prepared, reads, error, group, prepare_s))
    return {name: out[name] for name in sorted(out)}
