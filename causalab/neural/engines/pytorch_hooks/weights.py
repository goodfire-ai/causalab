"""Read checkpoint tensors onto the model device.

The default FastersafetensorsReader uses one planned Rust read with
concurrent shards and the GIL released. An explicit SafetensorsReader
uses sixteen ``safe_open`` threads.

``load_pretrained`` identifies required keys through Transformers' renamer
on a meta-device model, then passes tensors through ``state_dict`` so its
conversion rules run. Unused vision and MTP weights stay unread. Missing
required keys raise an error. Device tensors become parameters without
an additional copy.

A tensor stored in another dtype than the model holds it in is cast on the
host: ``conversions`` names the converting keys, and ``Prefetch`` reads them
in batches no larger than the largest converting tensor, casts, and copies
each to its device already in its dtype, so the device allocates the resident
weights only (model_parallelism.md §5.3). A same-dtype checkpoint takes the
straight path.

``device`` may be a comma list (``DeviceMap``): wanted keys are grouped by
the device their parameter lands on and read there, transformers receives a
real ``device_map``, accelerate's offload hooks are stripped, and the
engine's own crossings run a block where its weights are. A head tied to
its embedding refuses a list by name.

Under a ``Sharding``, ``load_planned`` applies the registry's plan to the
meta instance first, ``shard_read.read_plan`` decides each rank's byte
ranges before any read, and one planned read per device group fetches
exactly those ranges. A pipeline stage reads its own layers alone.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from causalab.io.fastersafetensors._dtypes import (  # pyright: ignore[reportPrivateUsage]
    torch_dtype,  # the reader's own header → torch dtype table
)
from causalab.neural.engines.pytorch_hooks.checkpoint import (
    TensorHeader,
    checkpoint_files,
    read_header,
)
from causalab.neural.engines.pytorch_hooks.crossings import place_crossings
from causalab.neural.engines.pytorch_hooks.residency import Census, Residency
from causalab.neural.engines.pytorch_hooks.shard_read import (
    LoadReport,
    Piece,
    ShardReadError,
    read_plan,
    model_table,
)
from causalab.neural.engines.pytorch_hooks.sharding import Sharding, apply_plan
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.parallel.memory import preflight
from causalab.protocol.parallel import format_geometry
from causalab.protocol.parallel_memory import DTYPE_ITEMSIZES
from causalab.protocol.registry import (
    TreeAddress,
    family_for,
    model_info_from_hf_config,
)
from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "CheckpointReader",
    "FastersafetensorsReader",
    "Loaded",
    "Prefetch",
    "ReadGroup",
    "SafetensorsReader",
    "Shard",
    "ShardReader",
    "TensorHeader",
    "checkpoint_files",
    "conversions",
    "default_reader",
    "group_shards",
    "load_planned",
    "load_pretrained",
    "read_header",
    "renamed_keys",
    "shard_plan",
    "target_dtypes",
    "wanted_keys",
]

logger = logging.getLogger(__name__)

#: The document's precision word of each torch dtype a load takes — the
#: inverse of ``loading._DTYPES``, for the memory pre-flight's arithmetic
#: and its refusal text (``protocol/parallel_memory.py``).
_PRECISION_OF: Mapping[torch.dtype, str] = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}
assert set(_PRECISION_OF.values()) == set(DTYPE_ITEMSIZES)


# ---------------------------------------------------------------------------
# What the model wants, and where it is
# ---------------------------------------------------------------------------


def renamed_keys(meta_model: Any, keys: Iterable[str]) -> dict[str, str]:
    """The checkpoint keys ``meta_model`` will consume, each with the
    parameter name it lands on — decided with transformers' own renamer, so
    the answer is the one its loading loop reaches
    (``convert_and_load_state_dict_in_model``, transformers 5.16): every
    ``WeightRenaming`` in turn, at most one ``WeightConverter``, the
    ``base_model_prefix`` added or stripped, then membership in the meta
    state dict; a key that already names a parameter is kept as is.

    A multimodal checkpoint's vision tower and MTP head fall out here: neither
    renames onto a text-model parameter, and the loader never reads them. The
    parameter name is what places a key on a device ([`load_pretrained`][]):
    the block index is in the model's name, not necessarily the checkpoint's.
    """
    from transformers.core_model_loading import (
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )
    from transformers.conversion_mapping import get_model_conversion_mapping

    mapping = get_model_conversion_mapping(meta_model, None, None)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    meta_state = meta_model.state_dict()
    prefix = meta_model.base_model_prefix
    wanted: dict[str, str] = {}
    for key in keys:
        renamed, _ = rename_source_key(key, renamings, converters, prefix, meta_state)
        if renamed not in meta_state and key in meta_state:
            renamed, _ = rename_source_key(key, [], [], prefix, meta_state)
        if renamed in meta_state:
            wanted[key] = renamed
    return wanted


def wanted_keys(meta_model: Any, keys: Iterable[str]) -> frozenset[str]:
    """The checkpoint keys ``meta_model`` will consume ([`renamed_keys`][])."""
    return frozenset(renamed_keys(meta_model, keys))


@dataclasses.dataclass(frozen=True)
class Shard:
    """One file and the keys the model wants out of it, in header order.
    ``select`` names, per key, the one piece a reader call cuts — set by
    [`Prefetch`][] on the shards it hands a reader under a read plan;
    ``None`` reads every key whole."""

    path: Path
    keys: tuple[str, ...]
    headers: Mapping[str, TensorHeader]
    select: Mapping[str, Piece] | None = None


def shard_plan(
    tables: Sequence[tuple[Path, Mapping[str, TensorHeader]]],
    wanted: frozenset[str],
) -> tuple[tuple[Shard, ...], frozenset[str]]:
    """Partition ``wanted`` over the files whose headers ``tables`` are: the
    shards that carry at least one wanted key (a file the model wants nothing
    from is never opened again), and the wanted keys no file carries — the
    caller's refusal, by name.

    A key in two files is refused: the format promises uniqueness across a
    sharded checkpoint, and a loader that picked one silently would load
    whichever file sorted first.
    """
    seen: dict[str, Path] = {}
    shards: list[Shard] = []
    for path, headers in tables:
        keys = tuple(name for name in headers if name in wanted)
        for name in keys:
            if name in seen:
                raise ProtocolError(
                    "P2",
                    f"tensor {name!r} appears in both {seen[name]} and {path}; "
                    "a sharded checkpoint carries each tensor once",
                )
            seen[name] = path
        if keys:
            shards.append(Shard(path=path, keys=keys, headers=headers))
    return tuple(shards), wanted - frozenset(seen)


@dataclasses.dataclass(frozen=True)
class ReadGroup:
    """One device's share of the plan: every shard carrying a key placed on
    the device, narrowed to those keys (header order kept)."""

    device: torch.device
    shards: tuple[Shard, ...]


def group_shards(
    shards: Sequence[Shard], placement: Mapping[str, torch.device]
) -> tuple[ReadGroup, ...]:
    """Split the plan by target device: for each device, in first-appearance
    order over the plan, the shards that carry a key ``placement`` puts on
    it, each shard narrowed to those keys. On a single-device map this is one
    group of the very same shards. Every key of every shard must be placed."""
    groups: dict[torch.device, list[Shard]] = {}
    for shard in shards:
        by_device: dict[torch.device, list[str]] = {}
        for key in shard.keys:
            by_device.setdefault(placement[key], []).append(key)
        for device, keys in by_device.items():
            groups.setdefault(device, []).append(
                Shard(path=shard.path, keys=tuple(keys), headers=shard.headers)
            )
    return tuple(
        ReadGroup(device=device, shards=tuple(members))
        for device, members in groups.items()
    )


# ---------------------------------------------------------------------------
# Readers: one shard's wanted tensors onto the device
# ---------------------------------------------------------------------------


@runtime_checkable
class ShardReader(Protocol):
    """Read the named tensors of one shard onto ``device``, each as a tensor
    the caller owns outright (torch-allocated, nothing to release later).

    ``concurrency`` is how many shards the loader may hand a reader at once —
    the reader's own memory story decides it. A reader that serves
    shard-on-read takes a fourth argument, ``select`` (per key the one piece
    to cut, [`Piece`][]);
    [`Prefetch`][] passes it only under a read plan, so a three-argument
    reader still serves every world-1 load.
    """

    @property
    def concurrency(self) -> int: ...

    def read(
        self, path: Path, keys: Sequence[str], device: torch.device
    ) -> dict[str, torch.Tensor]: ...


@runtime_checkable
class CheckpointReader(Protocol):
    """Read every wanted tensor of every shard in one call — for a reader that
    plans its own concurrency across files ([`FastersafetensorsReader`][]).
    Each shard's ``select`` names the piece to cut per key, when any."""

    def read_all(
        self, shards: Sequence["Shard"], device: torch.device
    ) -> dict[str, torch.Tensor]: ...


@dataclasses.dataclass(frozen=True)
class SafetensorsReader:
    """``safe_open(device=…).get_tensor`` per key: one device allocation per
    tensor, the copy outside the GIL, so sixteen shards in flight cost no
    memory beyond the tensors themselves. The reference reader, taken only when
    a caller passes it; the default is [`FastersafetensorsReader`][]. Under
    a read plan a selected key is read through ``get_slice`` as the piece's
    covering box, narrowed in torch."""

    concurrency: int = 16

    def read(
        self,
        path: Path,
        keys: Sequence[str],
        device: torch.device,
        select: Mapping[str, Piece] | None = None,
    ) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        select = select or {}
        with safe_open(str(path), framework="pt", device=str(device)) as handle:
            out: dict[str, torch.Tensor] = {}
            for key in keys:
                piece = select.get(key)
                if piece is None:
                    out[key] = handle.get_tensor(key)
                else:
                    out[key] = handle.get_slice(key)[piece.box][piece.view]
            return out


@dataclasses.dataclass(frozen=True)
class FastersafetensorsReader:
    """[`causalab.io.fastersafetensors.torch.load_files`][] over the whole
    checkpoint: one call, the wanted keys only, files in flight and the
    transport decided by its planner for this machine, the reads in Rust with
    the GIL released. Same memory story as [`SafetensorsReader`][] — every
    tensor is allocated by torch on the device and filled in place. Under a
    read plan the shards' ``select`` becomes the call's ``select``, so only
    the pieces' bytes are read. What [`default_reader`][] returns."""

    def read_all(
        self, shards: Sequence["Shard"], device: torch.device
    ) -> dict[str, torch.Tensor]:
        from causalab.io.fastersafetensors.torch import load_files

        select = {
            key: piece.spec
            for shard in shards
            if shard.select
            for key, piece in shard.select.items()
        }
        return load_files(
            [shard.path for shard in shards],
            device=str(device),
            keys=[key for shard in shards for key in shard.keys],
            select=select or None,
        )


def default_reader() -> ShardReader | CheckpointReader:
    """The reader [`load_pretrained`][] takes when none is passed: the
    planned Rust read. [`SafetensorsReader`][] is the explicit alternative."""
    return FastersafetensorsReader()


# ---------------------------------------------------------------------------
# The lazy stand-ins transformers materializes
# ---------------------------------------------------------------------------


#: Where a converting batch is read to and cast (module docstring).
_HOST = torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class _Job:
    """One reader call: the shards it reads — each narrowed to the job's
    keys, ``select`` set under a plan, every key of one round — onto
    ``read_device``, landing on ``device``. A straight job reads onto the
    device it lands on and hands the tensors out as read; a converting job
    reads onto the host and ``cast`` names, per key, the dtype it is cast to
    there before the copy to ``device`` (module docstring, "Converting")."""

    device: torch.device
    read_device: torch.device
    round: int
    shards: tuple[Shard, ...]
    cast: Mapping[str, torch.dtype]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for shard in self.shards for key in shard.keys)

    @property
    def converting(self) -> bool:
        return bool(self.cast)

    @property
    def disk_bytes(self) -> int:
        """The on-disk bytes the reader lands for this job."""
        return sum(
            _disk_bytes(shard, key) for shard in self.shards for key in shard.keys
        )


def _narrow(shard: Shard, keys: Sequence[str]) -> Shard:
    """``shard`` restricted to ``keys`` (their order kept) — the shard itself
    when nothing is dropped."""
    if tuple(keys) == shard.keys:
        return shard
    select = (
        None
        if shard.select is None
        else {key: shard.select[key] for key in keys if key in shard.select}
    )
    return Shard(
        path=shard.path, keys=tuple(keys), headers=shard.headers, select=select
    )


def _disk_bytes(shard: Shard, key: str) -> int:
    """The on-disk bytes the reader lands for ``key`` of ``shard``: its
    piece's under a plan, the whole tensor's else."""
    header = shard.headers[key]
    piece = shard.select.get(key) if shard.select else None
    elements = header.elements if piece is None else piece.elements
    return elements * header.itemsize


class Prefetch:
    """Every group's reads submitted at once — per shard, bounded by a
    [`ShardReader`][]'s ``concurrency``; or one whole-group read per device
    for a [`CheckpointReader`][] — each tensor handed out exactly once. A
    single-device load is one group; a placed load one per device, each
    reading that device's shards onto it ([`group_shards`][]).

    Under a read plan (``pieces``: per key the pieces this rank reads) the
    reads go in **rounds**: round ``r`` reads, of every shard, the keys with
    more than ``r`` pieces, each cut to its piece ``r`` — so one piece per
    key is the one planned read as before, and a strided placement's second
    range is one more call, not a whole read. [`take`][] is then asked for
    a key *and* a piece.

    **Converting** (``cast``: per key the dtype the model holds it in, for
    the keys stored in another — [`conversions`][]). A converting key is
    never read onto the device in its on-disk dtype. Per group and round its
    keys are taken from the shards in turn, so every file stays in flight,
    and packed into **batches** of at most [`budget`][] on-disk bytes —
    the largest converting tensor's, so a batch is one tensor at the least
    and the small ones travel together; each batch is one reader call onto
    the host, each tensor cast there and copied to the device already in
    its dtype, the on-disk copy dropped as the batch is handed on. The
    device's peak is the resident weights and nothing else; the host stages
    one batch per device group at a time under a [`CheckpointReader`][]
    (a group's jobs run on one worker, in order) and up to ``concurrency``
    under a [`ShardReader`][]. Every other key is read straight onto its
    device, as before. [`jobs`][] is the plan, decided at construction.

    The reads start on the first [`take`][], not on construction: transformers
    warms its caching allocator with one allocation the size of the model right
    before it materializes anything, and reads that started earlier would fill
    the device before that allocation.
    """

    def __init__(
        self,
        groups: Sequence[ReadGroup],
        reader: ShardReader | CheckpointReader,
        pieces: Mapping[str, tuple[Piece, ...]] | None = None,
        *,
        cast: Mapping[str, torch.dtype] | None = None,
    ) -> None:
        self._groups = tuple(group for group in groups if group.shards)
        self._reader = reader
        self._pieces = pieces
        self._cast = dict(cast or {})
        self._keys = frozenset(
            key
            for group in self._groups
            for shard in group.shards
            for key in shard.keys
        )
        self._lock = threading.Lock()
        self._pools: list[ThreadPoolExecutor] = []
        self._futures: dict[tuple[str, int], Future[dict[str, torch.Tensor]]] = {}
        #: The on-disk bytes of the largest converting tensor (its piece,
        #: under a plan) any group reads — a converting batch's ceiling;
        #: ``0`` with nothing to convert.
        self.budget: int = max(
            (
                _disk_bytes(narrowed, key)
                for group in self._groups
                for shard in group.shards
                for narrowed in self._rounds(shard)
                for key in narrowed.keys
                if key in self._cast
            ),
            default=0,
        )
        #: The reader calls, in submission order (class docstring).
        self.jobs: tuple[_Job, ...] = self._plan_jobs()

    def __enter__(self) -> "Prefetch":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _rounds(self, shard: Shard) -> list[Shard]:
        """The shards one reader call takes for ``shard``, one per round:
        the shard itself with no plan; else round ``r`` is its keys with
        more than ``r`` pieces, each selected to piece ``r``."""
        if self._pieces is None:
            return [shard]
        rounds: list[Shard] = []
        depth = max((len(self._pieces.get(key, ())) for key in shard.keys), default=0)
        for r in range(depth):
            keys = tuple(
                key for key in shard.keys if len(self._pieces.get(key, ())) > r
            )
            select = {key: self._pieces[key][r] for key in keys}
            rounds.append(
                Shard(path=shard.path, keys=keys, headers=shard.headers, select=select)
            )
        return rounds

    def _plan_jobs(self) -> tuple[_Job, ...]:
        """The reader calls (class docstring): per group and round, the
        straight keys as one whole-group call (a [`CheckpointReader`][])
        or one call per shard (a [`ShardReader`][]), then the converting
        keys in batches (`_batches`)."""
        jobs: list[_Job] = []
        whole_group = isinstance(self._reader, CheckpointReader)
        for group in self._groups:
            by_round: dict[int, list[Shard]] = {}
            for shard in group.shards:
                for r, narrowed in enumerate(self._rounds(shard)):
                    by_round.setdefault(r, []).append(narrowed)
            for r, shards in sorted(by_round.items()):
                straight = [
                    narrowed
                    for narrowed in (
                        _narrow(s, [k for k in s.keys if k not in self._cast])
                        for s in shards
                    )
                    if narrowed.keys
                ]
                converting = [
                    narrowed
                    for narrowed in (
                        _narrow(s, [k for k in s.keys if k in self._cast])
                        for s in shards
                    )
                    if narrowed.keys
                ]
                if whole_group:
                    if straight:
                        jobs.append(
                            _Job(group.device, group.device, r, tuple(straight), {})
                        )
                else:
                    jobs.extend(
                        _Job(group.device, group.device, r, (s,), {}) for s in straight
                    )
                jobs.extend(self._batches(group.device, r, converting))
        return tuple(jobs)

    def _batches(
        self, device: torch.device, r: int, shards: Sequence[Shard]
    ) -> list[_Job]:
        """The converting jobs of one group and round: the shards' keys taken
        in turn (so a batch spans the files and every one stays in flight),
        packed greedily up to [`budget`][], one job per batch, the shards
        narrowed to the batch's keys."""
        queues = [list(shard.keys) for shard in shards]
        order: list[tuple[int, str]] = []
        while any(queues):
            for index, queue in enumerate(queues):
                if queue:
                    order.append((index, queue.pop(0)))
        jobs: list[_Job] = []
        taken: dict[int, list[str]] = {}
        total = 0
        for index, key in order:
            nbytes = _disk_bytes(shards[index], key)
            if taken and total + nbytes > self.budget:
                jobs.append(self._batch(device, r, shards, taken))
                taken, total = {}, 0
            taken.setdefault(index, []).append(key)
            total += nbytes
        if taken:
            jobs.append(self._batch(device, r, shards, taken))
        return jobs

    def _batch(
        self,
        device: torch.device,
        r: int,
        shards: Sequence[Shard],
        taken: Mapping[int, Sequence[str]],
    ) -> _Job:
        members = tuple(
            _narrow(shards[index], keys) for index, keys in sorted(taken.items())
        )
        cast = {key: self._cast[key] for shard in members for key in shard.keys}
        return _Job(device, _HOST, r, members, cast)

    def _run(self, job: _Job) -> dict[str, torch.Tensor]:
        """One job on a worker: the reader call onto ``job.read_device``,
        then — a converting job — each tensor cast on the host and copied to
        the device in its dtype, the on-disk copy dropped here."""
        reader = self._reader
        if isinstance(reader, CheckpointReader):
            out = reader.read_all(job.shards, job.read_device)
        else:
            out = {}
            for shard in job.shards:
                if shard.select is None:
                    out.update(reader.read(shard.path, shard.keys, job.read_device))
                else:
                    # the protocol's fourth argument (its docstring)
                    selecting: Any = reader
                    out.update(
                        selecting.read(
                            shard.path, shard.keys, job.read_device, shard.select
                        )
                    )
        for key, dtype in job.cast.items():
            # the cast on the host, then the copy: the device allocates the
            # target dtype and nothing else (a combined ``.to(device, dtype)``
            # is left to torch's choice of where the cast runs)
            staged = out.pop(key)
            out[key] = staged.to(dtype).to(job.device)
            del staged
        return out

    def _start(self) -> None:
        with self._lock:
            if self._pools:
                return
            reader = self._reader
            pools: dict[torch.device, ThreadPoolExecutor]
            if isinstance(reader, CheckpointReader):
                # one worker per device group: a group's jobs run in order —
                # its whole-group read, then one converting batch at a time —
                # while the groups are in flight together
                pools = {
                    group.device: ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="causalab-weights"
                    )
                    for group in self._groups
                }
                self._pools = list(pools.values())
            else:
                shared = ThreadPoolExecutor(
                    max_workers=max(1, reader.concurrency),
                    thread_name_prefix="causalab-weights",
                )
                self._pools = [shared]
                pools = {group.device: shared for group in self._groups}
            for job in self.jobs:
                future = pools[job.device].submit(self._run, job)
                for key in job.keys:
                    self._futures[(key, job.round)] = future

    def take(self, key: str, piece: Piece | None = None) -> torch.Tensor:
        """The tensor for ``key`` — under a read plan, its pre-read ``piece``
        — ownership included: the reference leaves this object, so a fully
        taken shard holds nothing.

        Raises:
            ShardReadError: ``key`` is read by no group (this rank reads
                nothing of it), or ``piece`` is not one the plan pre-read.
        """
        if key not in self._keys:
            raise ShardReadError(
                f"{key!r}: this rank reads nothing of it (no piece planned), "
                "and the loader asked for it anyway"
            )
        r = 0
        if self._pieces is not None:
            planned = self._pieces.get(key, ())
            if piece is None or piece not in planned:
                raise ShardReadError(
                    f"{key!r}: the loader asked for {piece.spec if piece else '[...]'}"
                    f", which the read plan did not pre-read (planned: "
                    f"{[p.spec for p in planned]}); nothing is read whole behind "
                    "the plan's back"
                )
            r = planned.index(piece)
        self._start()
        tensors = self._futures[(key, r)].result()
        with self._lock:
            return tensors.pop(key)

    def close(self) -> None:
        with self._lock:
            pools, self._pools = self._pools, []
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


class _LazyWeight:
    """What transformers sees in the ``state_dict``: the slice surface it
    materializes through (``[...]``, ``get_shape``, ``get_dtype``), backed by
    a [`Prefetch`][]. ``__getitem__`` is the one call that touches data.
    With ``pieces`` (a read plan), the index is resolved to its
    [`Piece`][] and the pre-read piece handed out at the result's shape;
    without, ``[index]`` is plain indexing of the whole read — today's path."""

    __slots__ = ("_key", "_header", "_prefetch", "_pieces")

    def __init__(
        self,
        key: str,
        header: TensorHeader,
        prefetch: Prefetch,
        pieces: tuple[Piece, ...] | None = None,
    ) -> None:
        self._key = key
        self._header = header
        self._prefetch = prefetch
        self._pieces = pieces

    def get_shape(self) -> list[int]:
        return list(self._header.shape)

    def get_dtype(self) -> str:
        return self._header.dtype

    def __getitem__(self, index: Any) -> torch.Tensor:
        if self._pieces is None:
            return self._prefetch.take(self._key)[index]
        piece = Piece.of(self._header.shape, index)
        return self._prefetch.take(self._key, piece)


# ---------------------------------------------------------------------------
# What each key is cast to
# ---------------------------------------------------------------------------


def target_dtypes(
    meta_model: Any, targets: Mapping[str, str], dtype: torch.dtype
) -> dict[str, torch.dtype]:
    """Per checkpoint key the dtype transformers lands it in — its own rule
    (``core_model_loading.convert_and_load_state_dict_in_model``,
    transformers 5.16, "Handle dtype casting"): the dtype plan's entry when
    one of its patterns matches the parameter (``_keep_in_fp32_modules``
    under fp16, ``_keep_in_fp32_modules_strict`` under either half
    precision; ``PreTrainedModel._get_dtype_plan``), else the meta
    parameter's own dtype — the model's ``dtype`` for a float parameter, an
    integer buffer's its own. ``targets`` is the key → parameter map
    ([`renamed_keys`][], ``ReadPlan.targets``)."""
    from transformers.core_model_loading import build_glob_alternation

    plan: Mapping[str, torch.dtype] = meta_model._get_dtype_plan(dtype)  # pyright: ignore[reportPrivateUsage]
    meta_state = meta_model.state_dict()
    alternation = by_group = None
    if plan:
        alternation, by_group, _ = build_glob_alternation(list(plan))
    out: dict[str, torch.dtype] = {}
    for key, parameter in targets.items():
        want = dtype
        matched = alternation.search(parameter) if alternation is not None else None
        group = matched.lastgroup if matched is not None else None
        if group is not None and by_group is not None:
            want = plan[by_group[group]]
        elif meta_state[parameter].dtype != want:
            want = meta_state[parameter].dtype
        out[key] = want
    return out


def conversions(
    meta_model: Any,
    targets: Mapping[str, str],
    headers: Mapping[str, TensorHeader],
    dtype: torch.dtype,
) -> dict[str, torch.dtype]:
    """The checkpoint keys a load of ``meta_model`` as ``dtype`` casts, each
    with the dtype it lands in ([`target_dtypes`][]): those whose header
    dtype is another — what [`Prefetch`][] stages on the host (module
    docstring, "Converting"). Empty for a checkpoint stored as the model
    holds it."""
    wanted = target_dtypes(meta_model, targets, dtype)
    return {
        key: want
        for key, want in wanted.items()
        if torch_dtype(headers[key].dtype) != want
    }


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------


def load_pretrained(
    key: str,
    revision: str,
    *,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str | None,
    reader: ShardReader | CheckpointReader | None = None,
) -> Any:
    """``AutoModelForCausalLM.from_pretrained(key, revision, dtype,
    attn_implementation)`` with its weights placed by ``device`` — one device,
    or a comma list spreading the layers over the devices of this process
    ([`DeviceMap.parse`][]) — the stock result, read as the module docstring
    describes. A placed model runs a forward as is: its crossings are
    installed (``crossings.py``).

    The stock loader takes over when the checkpoint ships no safetensors.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # ``None`` leaves the attention backend to transformers' default
    attention = (
        {"attn_implementation": attn_implementation}
        if attn_implementation is not None
        else {}
    )
    config = AutoConfig.from_pretrained(key, revision=revision)
    devices = DeviceMap.parse(device, model_info_from_hf_config(key, config).num_layers)
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(config, dtype=dtype, **attention)
    tree = _placement_tree(meta, devices, key=key)
    # one device: exactly the stock ``{"": device}``; several: the map over
    # module prefixes transformers places parameters by
    device_map = (
        devices.module_map(tree) if tree is not None else {"": str(devices.single)}
    )
    files = checkpoint_files(key, revision)
    if files is None:
        logger.info(
            "weights: stock loader (no safetensors) key=%s revision=%s", key, revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            key,
            revision=revision,
            dtype=dtype,
            **attention,
            device_map=device_map,
        )
        return _placed(model, devices, tree)

    tables = [(path, read_header(path)) for path in files]
    wanted = wanted_keys(meta, (name for _, table in tables for name in table))
    shards, absent = shard_plan(tables, wanted)
    if absent:
        raise ProtocolError(
            "P2",
            f"{key}@{revision}: the model wants {len(absent)} tensor(s) no shard "
            f"carries, first {sorted(absent)[0]!r}",
        )
    # where each checkpoint key's parameter lands: the one device, or the
    # device of the module prefix of the parameter name the key renames onto
    renamed = renamed_keys(meta, wanted)
    if tree is None:
        placement = dict.fromkeys(wanted, devices.embedding)
    else:
        placement = {
            source: devices.device_for(renamed.get(source, source), tree)
            for source in wanted
        }
    groups = group_shards(shards, placement)
    headers = {
        name: header
        for _, table in tables
        for name, header in table.items()
        if name in wanted
    }
    cast = conversions(meta, renamed, headers, dtype)
    chosen = reader if reader is not None else default_reader()
    with Prefetch(groups, chosen, cast=cast) as prefetch:
        logger.info(
            "weights: reader=%s shards=%d tensors=%d converting=%d batch_bytes=%d "
            "device=%s groups=%d key=%s",
            type(chosen).__name__,
            len(shards),
            len(wanted),
            len(cast),
            prefetch.budget,
            devices.spelling,
            len(groups),
            key,
        )
        state = {
            name: _LazyWeight(name, shard.headers[name], prefetch)
            for shard in shards
            for name in shard.keys
        }
        model, info = type(meta).from_pretrained(
            None,
            config=meta.config,
            state_dict=state,
            dtype=dtype,
            **attention,
            device_map=device_map,
            output_loading_info=True,
        )
    _check_complete(info, key=key, revision=revision)
    # what the stock loader records; nothing hashed reads it, the receipt's
    # identity is the bundle's ``key``, but the two paths should not differ
    model.config.name_or_path = key
    return _placed(model, devices, tree)


@dataclasses.dataclass(frozen=True)
class Loaded:
    """A sharded load's result: the model on this rank, what the reader was
    asked for against what is on disk, per parameter, and what the rank
    holds against it (``residency.py``)."""

    model: Any
    report: LoadReport
    residency: Residency


def load_planned(
    key: str,
    revision: str,
    *,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str | None,
    sharding: Sharding,
    reader: ShardReader | CheckpointReader | None = None,
    census: bool = False,
) -> Loaded:
    """This rank's share of ``key`` under ``sharding`` (module docstring,
    "Shard-on-read"): the registry's plan applied to a meta instance, the
    rank's ranges of the wanted tensors decided and read — one planned read
    per device group — and transformers' loader run over them. Every rank
    of the world calls this with its own [`Sharding`][]. The result
    carries the rank's [`Residency`][], measured right after
    the load — with the live-tensor census when ``census`` is set: a
    [`Census`][] of the device taken here, before anything
    is allocated, is what the load's residency is measured against.

    Raises:
        ProtocolError: ``P4`` — a device list (one device per rank under a
            geometry above world 1), or the plan's own refusals
            (``sharding.apply_plan``); ``P2`` — a checkpoint without
            safetensors (the stock loader has no shard-on-read), a wanted
            tensor no shard carries, or a parameter the reader did not
            deliver.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    attention = (
        {"attn_implementation": attn_implementation}
        if attn_implementation is not None
        else {}
    )
    config = AutoConfig.from_pretrained(key, revision=revision)
    info = model_info_from_hf_config(key, config)
    devices = DeviceMap.parse(device, info.num_layers)
    target = devices.single
    if target is None:
        raise ProtocolError(
            "P4",
            f"device {device!r} places the layers across several devices, and a "
            f"geometry above world 1 ({format_geometry(sharding.geometry)}) places "
            "one rank on one device; give each rank its device",
        )
    assert info.parallel_plan is not None  # the adapter always derives one
    # what the device held before the load: never the load's
    baseline = Census.take(target) if census else None
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype, **attention)
    files = checkpoint_files(key, revision)
    if files is None:
        raise ProtocolError(
            "P2",
            f"{key}@{revision} ships no safetensors, and a sharded load reads "
            "ranges of a safetensors checkpoint; the stock loader has no "
            "shard-on-read",
        )
    tables = [(path, read_header(path)) for path in files]
    names = [name for _, table in tables for name in table]
    all_headers = {
        name: header for _, table in tables for name, header in table.items()
    }
    # the whole model's placement table — the memory pre-flight's input —
    # taken before the plan places a pipeline stage on the meta model, after
    # which the model lists its stage's keys alone (``shard_read.model_table``)
    whole = model_table(
        model,
        wanted_keys(model, names),
        all_headers,
        plan=info.parallel_plan,
        info=info,
    )
    # the registry's plan as applied under this geometry: the K/V projections
    # replicated when the tensor axis exceeds the KV heads (§6.6)
    apply_plan(
        model, info.parallel_plan.for_geometry(sharding.geometry, info), sharding
    )
    wanted = wanted_keys(model, names)
    shards, absent = shard_plan(tables, wanted)
    if absent:
        raise ProtocolError(
            "P2",
            f"{key}@{revision}: the model wants {len(absent)} tensor(s) no shard "
            f"carries, first {sorted(absent)[0]!r}",
        )
    headers = {name: header for name, header in all_headers.items() if name in wanted}
    plan = read_plan(
        model, wanted, headers, plan=info.parallel_plan, info=info, sharding=sharding
    )
    # the memory pre-flight (docs/model_parallelism.md §2, §11): what this
    # rank's share of the weights and the run's headroom come to on its
    # device, refused by name here — before the first read is issued —
    # rather than inside NCCL after the load; the table is the whole
    # model's, so a pipeline stage is split by the tower's height and the
    # headroom is the rule's fraction of the model, not of the stage
    preflight(
        geometry=sharding.geometry,
        rank=sharding.rank,
        device=str(target),
        dtype=_PRECISION_OF[dtype],
        table=whole,
        info=info,
    )
    read = frozenset(k for k, pieces in plan.pieces.items() if pieces)
    narrowed = [
        Shard(
            path=shard.path,
            keys=tuple(k for k in shard.keys if k in read),
            headers=shard.headers,
        )
        for shard in shards
    ]
    groups = group_shards(
        [shard for shard in narrowed if shard.keys], dict.fromkeys(read, target)
    )
    chosen = reader if reader is not None else default_reader()
    cast = conversions(model, plan.targets, headers, dtype)
    with Prefetch(groups, chosen, pieces=plan.pieces, cast=cast) as prefetch:
        logger.info(
            "weights: reader=%s shards=%d tensors=%d read=%d converting=%d "
            "batch_bytes=%d device=%s parallel=%s rank=%d key=%s",
            type(chosen).__name__,
            len(shards),
            len(wanted),
            len(read),
            sum(1 for k in cast if k in read),
            prefetch.budget,
            target,
            format_geometry(sharding.geometry),
            sharding.rank,
            key,
        )
        state = {
            name: _LazyWeight(name, shard.headers[name], prefetch, plan.pieces[name])
            for shard in shards
            for name in shard.keys
        }
        loading = _load_into(model, state, dtype=dtype, device=target)
    _check_complete(loading, key=key, revision=revision)
    model.config.name_or_path = key
    residency = Residency.of(model, target, since=baseline)
    return Loaded(model=model, report=plan.report(headers), residency=residency)


def _load_into(
    model: Any, state: Mapping[str, Any], *, dtype: torch.dtype, device: torch.device
) -> dict[str, Any]:
    """Run transformers' loading machinery over ``state`` into ``model`` — a
    meta instance with the plan applied — the way ``from_pretrained`` does
    after constructing its own: the conversion mapping, the dtype plan, the
    load, the finalization (missing keys off meta, ties, the report), eval
    mode. The loading info, as ``output_loading_info`` returns it."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.modeling_utils import LoadStateDictConfig

    # typed as the transforms' base class; transformers' own call site hands
    # the same list to the same field
    mapping: Any = get_model_conversion_mapping(model, None, None)
    load_config = LoadStateDictConfig(
        pretrained_model_name_or_path=None,
        device_map={"": device},
        dtype=dtype,
        dtype_plan=model._get_dtype_plan(dtype),  # pyright: ignore[reportPrivateUsage]
        weight_mapping=mapping,
    )
    cls = type(model)
    loading_info, _offload_index = cls._load_pretrained_model(  # pyright: ignore[reportPrivateUsage]
        model, dict(state), None, load_config
    )
    loading_info = cls._finalize_model_loading(  # pyright: ignore[reportPrivateUsage]
        model, load_config, loading_info
    )
    model.eval()
    return loading_info.to_dict()


def _placement_tree(meta: Any, devices: DeviceMap, *, key: str) -> TreeAddress | None:
    """The family's module-tree addresses a placed load is keyed by — ``None``
    on a single device, where nothing is placed by prefix and an unregistered
    family loads as it always has. A model whose head is tied to its
    embedding is refused a list: the two are one tensor, and the map wants
    it on the first device and on the last."""
    if devices.single is not None:
        return None
    if getattr(meta.config, "tie_word_embeddings", False):
        raise ProtocolError(
            "P4",
            f"{key}: the model ties its head to its embedding "
            "(tie_word_embeddings), one tensor the device map would put on "
            f"{devices.embedding} and on {devices.head}. A tied model runs on "
            "one device.",
        )
    return family_for(meta).tree


def _placed(model: Any, devices: DeviceMap, tree: TreeAddress | None) -> Any:
    """A model transformers loaded under a map over several devices, made to
    run there: accelerate's dispatch hooks stripped (module docstring —
    their ``cpu`` entry is offload, not placement; every parameter is back
    on the device the map placed it on) and the engine's crossings installed
    (``crossings.py``). A single-device model is returned untouched."""
    if tree is None:
        return model
    from accelerate.hooks import (  # pyright: ignore[reportMissingTypeStubs]
        remove_hook_from_module,
    )

    remove_hook_from_module(model, recurse=True)
    place_crossings(model, devices, tree)
    return model


def _check_complete(info: Mapping[str, Any], *, key: str, revision: str) -> None:
    """A fast path that left a parameter at its random initialization would
    be the worst kind of wrong — numbers that run. transformers raises on a
    shape mismatch itself; what it only *reports* is a parameter it never saw
    (``missing``) and a tensor it had no parameter for (``unexpected``, which
    [`wanted_keys`][] should have excluded). Both are refused here, naming
    the first key. A DTensor parameter is compared by transformers to its
    local shard's shape, so a sharded load whose slice is the slice passes,
    and one whose slice is not is a ``mismatched`` entry — refused too."""
    missing = sorted(info.get("missing_keys", ()))
    unexpected = sorted(info.get("unexpected_keys", ()))
    mismatched = sorted(str(entry) for entry in info.get("mismatched_keys", ()))
    problems = [
        (label, entries)
        for label, entries in (
            ("missing", missing),
            ("unexpected", unexpected),
            ("mismatched", mismatched),
        )
        if entries
    ]
    if problems:
        detail = "; ".join(
            f"{len(entries)} {label} (first {entries[0]!r})"
            for label, entries in problems
        )
        raise ProtocolError(
            "P2",
            f"{key}@{revision}: the weight reader and the model disagree about "
            f"the checkpoint — {detail}. Nothing was loaded.",
        )
