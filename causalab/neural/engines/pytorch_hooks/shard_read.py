"""Shard-on-read: which range of each checkpoint tensor this rank reads
(``docs/model_parallelism.md`` §5.3).

After the plan is applied (``sharding.py``) transformers' loader slices each
checkpoint tensor down to the rank's local shard by indexing the lazy weight
— ``DtensorShardOperation.shard_tensor(source)`` computes the slices from
the parameter's DTensor placements and runs ``source[slices]``. The loader
here decides those slices **before** the read, from the same facts the
placeholders were built from, so one planned read per device group fetches
exactly the ranges the rank will be asked for, and nothing else:

* [`Piece`][] is one basic index (ints, slices, ``Ellipsis``) over a
  tensor's shape, normalised to the ranges it covers — the key a read-ahead
  is stored under and the ``select`` spec the Rust reader is handed.
* [`read_plan`][] walks the wanted keys in the loader's order, renames
  each onto the parameter it lands on through the model's conversion
  mapping (the per-expert count ``tensor_idx`` included), and reads the
  parameter's placement off the **placement table**
  (``protocol/parallel_memory.placement_table`` — the torch-free table
  ``dry-run`` estimates from, whose per-rank byte counts already reproduce
  every recorded load report) and its cut off the style's
  [`Partition`][] (``styles.partition_of``, the one owner of
  transformers' ``Shard`` / ``_StridedShard`` chunk arithmetic). A
  parameter the table holds whole — no row, a replicated style, a K/V
  projection above the KV heads — is read whole; a sharded one at this
  rank's ranges of the partitioned dimension; an unowned expert's
  per-expert tensor gets no piece at all. No DTensor is consulted, so the
  plan runs in one process for any geometry and rank, and the transformers
  tier's ``shard_tensor`` is held to it by the ``gloo`` loads
  (``test_sharded_load.py``: the local shard is the chunk the plan read).
* [`LoadReport`][] counts the bytes requested against the bytes on disk,
  per model parameter, so a test can hold a sharded parameter to
  ``1 / world`` and a replicated one to the whole — and the elements
  requested, which ``residency.py`` holds the loaded parameter to.

An index the plan did not pre-read, an index that is not basic indexing, or
a key this rank reads nothing of is a [`ShardReadError`][]: an internal
invariant between transformers' arithmetic and the plan broke, and the
answer is never a whole read behind the plan's back.
"""

from __future__ import annotations

import dataclasses
from math import prod
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from causalab.neural.engines.pytorch_hooks.checkpoint import TensorHeader
from causalab.neural.engines.pytorch_hooks.styles import Partition, partition_of
from causalab.protocol.parallel_memory import (
    PlacementTable,
    placement_table,
    row_for,
)
from causalab.protocol.registry import ModelInfo, ParallelPlan, family_for

if TYPE_CHECKING:
    from causalab.neural.engines.pytorch_hooks.sharding import Sharding

__all__ = [
    "LoadReport",
    "Piece",
    "ReadPlan",
    "ShardReadError",
    "model_table",
    "read_plan",
]


class ShardReadError(RuntimeError):
    """The read plan and the index the loader was handed disagree (module
    docstring). Not a [`ProtocolError`][causalab.protocol.rules.errors.ProtocolError]: no
    document rule was broken, an internal contract was."""


# --------------------------------------------------------------------------- #
# Piece — one basic index over a tensor, as the reader reads it
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Piece:
    """A basic index over a tensor of ``shape``: per dimension the covered
    range ``(lo, hi, step)`` — ``hi`` exclusive, the last element at
    ``hi - 1`` — and whether an int dropped the dimension from the result."""

    shape: tuple[int, ...]
    ranges: tuple[tuple[int, int, int], ...]
    squeezed: tuple[bool, ...]

    @classmethod
    def of(cls, shape: Sequence[int], index: Any) -> "Piece":
        """Normalise ``index`` — an int, a slice, ``Ellipsis`` or a tuple of
        them, as ``tensor[index]`` takes — over ``shape``, with torch's own
        bounds and wrap-around. Anything else (``None``, a tensor, a
        string, a bool) is refused: transformers hands the lazy weight basic
        indexing and nothing more."""
        shape = tuple(int(n) for n in shape)
        items: list[Any] = list(index) if isinstance(index, tuple) else [index]
        for item in items:
            if isinstance(item, bool) or not isinstance(
                item, (int, slice, EllipsisType)
            ):
                raise ShardReadError(
                    f"index {item!r} is not basic indexing (an int, a slice or "
                    f"'...'); the loader's lazy weight over shape {shape} takes "
                    "nothing else"
                )
        consuming = sum(not isinstance(item, EllipsisType) for item in items)
        if consuming > len(shape):
            raise ShardReadError(
                f"too many indices for a tensor of shape {shape}: {index!r}"
            )
        ellipses = [i for i, item in enumerate(items) if isinstance(item, EllipsisType)]
        if len(ellipses) > 1:
            raise ShardReadError(f"an index may have one '...', got {index!r}")
        fill: list[Any] = [slice(None)] * (len(shape) - consuming)
        if ellipses:
            at = ellipses[0]
            items = items[:at] + fill + items[at + 1 :]
        else:
            items = items + fill
        ranges: list[tuple[int, int, int]] = []
        squeezed: list[bool] = []
        for dim, (item, size) in enumerate(zip(items, shape, strict=True)):
            if isinstance(item, int):
                if item < -size or item >= size:
                    raise ShardReadError(
                        f"index {item} is out of bounds for dimension {dim} of "
                        f"size {size} (shape {shape})"
                    )
                at = item + size if item < 0 else item
                ranges.append((at, at + 1, 1))
                squeezed.append(True)
            else:
                start, stop, step = item.indices(size)
                if step < 1:
                    raise ShardReadError(
                        f"slice step must be positive, got {item!r} on dimension {dim}"
                    )
                count = len(range(start, stop, step))
                hi = start + (count - 1) * step + 1 if count else start
                ranges.append((start, hi, step))
                squeezed.append(False)
        return cls(shape, tuple(ranges), tuple(squeezed))

    @classmethod
    def whole(cls, shape: Sequence[int]) -> "Piece":
        """The whole tensor — what ``[...]`` names."""
        return cls.of(shape, Ellipsis)

    @classmethod
    def along(cls, shape: Sequence[int], dim: int, span: range) -> "Piece":
        """The rows ``span`` of dimension ``dim``, every other dimension
        whole — the chunk a partition names; an empty span is the
        zero-length slice at the origin, as transformers indexes it."""
        index: list[Any] = [slice(None)] * len(shape)
        index[dim] = slice(span.start, span.stop) if len(span) else slice(0, 0)
        return cls.of(shape, tuple(index))

    @property
    def is_whole(self) -> bool:
        return self == Piece.whole(self.shape)

    @property
    def spec(self) -> tuple[int | slice, ...]:
        """The index as ``load_files(select=…)`` and ``tensor[…]`` take it:
        an int where a dimension is squeezed, ``slice(lo, hi, step)`` else."""
        return tuple(
            lo if squeeze else slice(lo, hi, step)
            for (lo, hi, step), squeeze in zip(self.ranges, self.squeezed, strict=True)
        )

    @property
    def box(self) -> tuple[slice, ...]:
        """The unit-step slices covering the piece — for a reader that cuts
        contiguous ranges only; [`view`][] then narrows the box."""
        return tuple(slice(lo, hi) for lo, hi, _ in self.ranges)

    @property
    def view(self) -> tuple[int | slice, ...]:
        """The index that turns a tensor of the [`box`][]'s shape into the
        result: ``0`` on a squeezed dimension, a stepped slice from 0 else."""
        return tuple(
            0 if squeeze else slice(0, hi - lo, step)
            for (lo, hi, step), squeeze in zip(self.ranges, self.squeezed, strict=True)
        )

    @property
    def result_shape(self) -> tuple[int, ...]:
        return tuple(
            len(range(lo, hi, step))
            for (lo, hi, step), squeeze in zip(self.ranges, self.squeezed, strict=True)
            if not squeeze
        )

    @property
    def elements(self) -> int:
        """Elements of the result."""
        return prod(self.result_shape)

    def nbytes(self, itemsize: int) -> int:
        """Bytes of the result — what the reader is asked for."""
        return self.elements * itemsize


# --------------------------------------------------------------------------- #
# the plan and its report
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class LoadReport:
    """Per model parameter, the bytes the reader was asked for against the
    bytes its checkpoint tensors hold on disk. A parameter sharded over
    ``world`` ranks requests ``1 / world``; a replicated one the whole; a
    pipeline stage's report covers its own parameters and nothing else."""

    bytes_requested: Mapping[str, int]
    bytes_on_disk: Mapping[str, int]
    #: Elements requested per parameter — what its local tensor must hold
    #: after the load, whatever dtype it lands in (``residency.py``).
    elements_requested: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: Per parameter the safetensors dtype its checkpoint tensors are stored
    #: in (``F32``; two dtypes joined ``BF16/F32``) — against the residency's
    #: ``dtype_resident`` this is what names a converting load (§5.3).
    dtype_on_disk: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def fraction(self, parameter: str) -> float:
        """``requested / on_disk`` for one parameter."""
        return self.bytes_requested[parameter] / self.bytes_on_disk[parameter]

    @property
    def requested_total(self) -> int:
        return sum(self.bytes_requested.values())

    @property
    def on_disk_total(self) -> int:
        return sum(self.bytes_on_disk.values())


@dataclasses.dataclass(frozen=True)
class ReadPlan:
    """Per checkpoint key the pieces this rank reads — the empty tuple for a
    tensor it reads nothing of — the model parameter each key lands on, and
    the placement table the pieces were decided from (the memory
    pre-flight's input, so it is built once)."""

    pieces: Mapping[str, tuple[Piece, ...]]
    targets: Mapping[str, str]
    table: PlacementTable = dataclasses.field(default_factory=dict)

    def report(self, headers: Mapping[str, TensorHeader]) -> LoadReport:
        requested: dict[str, int] = {}
        on_disk: dict[str, int] = {}
        elements: dict[str, int] = {}
        dtypes: dict[str, set[str]] = {}
        for key, pieces in self.pieces.items():
            header = headers[key]
            target = self.targets[key]
            on_disk[target] = (
                on_disk.get(target, 0) + prod(header.shape) * header.itemsize
            )
            requested[target] = requested.get(target, 0) + sum(
                piece.nbytes(header.itemsize) for piece in pieces
            )
            elements[target] = elements.get(target, 0) + sum(
                piece.elements for piece in pieces
            )
            dtypes.setdefault(target, set()).add(header.dtype)
        return LoadReport(
            bytes_requested=requested,
            bytes_on_disk=on_disk,
            elements_requested=elements,
            dtype_on_disk={
                target: "/".join(sorted(words)) for target, words in dtypes.items()
            },
        )


@dataclasses.dataclass(frozen=True)
class _Landing:
    """Where a checkpoint key lands: the parameter, and — for a tensor a
    ``MergeModulelist`` converter stacks — its index along the stacked axis
    and how many are stacked."""

    target: str
    tensor_idx: int | None
    slot: tuple[str, str] | None


def _landings(meta_model: Any, keys: Iterable[str]) -> dict[str, _Landing]:
    """Mirrors ``convert_and_load_state_dict_in_model`` (transformers 5.16):
    the keys in its natural order, each renamed by the model's conversion
    mapping onto the parameter it lands on; a key a ``MergeModulelist``
    converter stacks (one tensor per expert) carries the running count of
    its source pattern as ``tensor_idx``, which is how the loader tells an
    owned expert from an unowned one."""
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import (
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
        dot_natural_key,
        rename_source_key,
    )

    mapping = get_model_conversion_mapping(meta_model, None, None)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    by_pattern = {
        pattern: converter
        for converter in converters
        for pattern in converter.source_patterns
    }
    meta_state = meta_model.state_dict()
    prefix = meta_model.base_model_prefix
    counts: dict[tuple[str, str], int] = {}
    landings: dict[str, _Landing] = {}
    for key in sorted(keys, key=dot_natural_key):
        renamed, source_pattern = rename_source_key(
            key, renamings, converters, prefix, meta_state
        )
        if renamed not in meta_state and key in meta_state:
            renamed, source_pattern = rename_source_key(key, [], [], prefix, meta_state)
        if renamed not in meta_state:
            raise ShardReadError(
                f"checkpoint key {key!r} renames onto {renamed!r}, which the model "
                "has no parameter for; the wanted keys were decided against the "
                "same model, so the two disagree"
            )
        tensor_idx: int | None = None
        slot: tuple[str, str] | None = None
        if source_pattern is not None:
            converter = by_pattern[source_pattern]
            if any(isinstance(op, MergeModulelist) for op in converter.operations):
                slot = (renamed, source_pattern)
                tensor_idx = counts.get(slot, 0)
                counts[slot] = tensor_idx + 1
        landings[key] = _Landing(renamed, tensor_idx, slot)
    return landings


def model_table(
    meta_model: Any,
    keys: Iterable[str],
    headers: Mapping[str, TensorHeader],
    *,
    plan: ParallelPlan,
    info: ModelInfo,
) -> PlacementTable:
    """The placement table of ``meta_model``'s parameters that ``keys`` land
    on, under the family's ``plan`` — the memory pre-flight's input. Built
    on the meta model **before** the plan places a pipeline stage on it: a
    placed model lists only its stage's keys, and a table of those is the
    stage's slice, which cannot determine the full model's layer count or
    memory requirements."""
    landings = _landings(meta_model, keys)
    return _table(meta_model, landings, headers, plan, info)[1]


def _table(
    meta_model: Any,
    landings: Mapping[str, _Landing],
    headers: Mapping[str, TensorHeader],
    plan: ParallelPlan,
    info: ModelInfo,
) -> tuple[dict[str, str], PlacementTable]:
    """Per key its landing parameter, and the placement table of those
    parameters (``placement_table`` reads the K/V projections'
    ``shard_limit`` off ``info``)."""
    targets = {key: landing.target for key, landing in landings.items()}
    tree = family_for(meta_model).tree
    table = placement_table(
        {key: headers[key].elements for key in landings},
        targets,
        plan,
        tree,
        info,
        dtypes={key: headers[key].dtype for key in landings},
    )
    return targets, table


def read_plan(
    meta_model: Any,
    keys: Iterable[str],
    headers: Mapping[str, TensorHeader],
    *,
    plan: ParallelPlan,
    info: ModelInfo,
    sharding: Sharding,
) -> ReadPlan:
    """What this rank reads of each wanted checkpoint key of ``meta_model``
    (module docstring): the placement table of the keys' landing parameters
    under the **family's** ``plan`` (``placement_table`` reads the K/V
    projections' ``shard_limit`` off ``info``), then per key its pieces —
    whole for a parameter the table holds whole under the geometry, this
    rank's ranges of the style's partition for a sharded one, none for a
    per-expert tensor of an expert another rank owns. ``meta_model`` is
    read for its conversion mapping and state-dict keys alone: the plan
    needs no placeholder, DTensor or otherwise.
    """
    landings = _landings(meta_model, keys)
    stacked: dict[tuple[str, str], int] = {}
    for landing in landings.values():
        if landing.slot is not None:
            stacked[landing.slot] = stacked.get(landing.slot, 0) + 1
    targets, table = _table(meta_model, landings, headers, plan, info)
    prefix = family_for(meta_model).tree.blocks.split(".", 1)[0]
    geometry = sharding.geometry
    pieces: dict[str, tuple[Piece, ...]] = {}
    for key, landing in landings.items():
        shape = tuple(headers[key].shape)
        placement = table[landing.target]
        shards = placement.shards(geometry)
        if shards == 1 or placement.axis is None:
            pieces[key] = (Piece.whole(shape),)
            continue
        row = row_for(plan, landing.target, prefix)
        assert row is not None  # the table found the row that set the axis
        rank = sharding.layout.rank_in(sharding.rank, placement.axis)
        leaf = landing.target.rsplit(".", 1)[-1]
        if landing.tensor_idx is None:
            partition = partition_of(row.style, len(shape), leaf)
            pieces[key] = _dense_pieces(partition, shape, rank, shards)
            continue
        assert landing.slot is not None
        partition = partition_of(row.style, len(shape) + 1, leaf)
        pieces[key] = _stacked_pieces(
            partition, shape, landing.tensor_idx, stacked[landing.slot], rank, shards
        )
    return ReadPlan(pieces=pieces, targets=targets, table=table)


def _dense_pieces(
    partition: Partition, shape: tuple[int, ...], rank: int, size: int
) -> tuple[Piece, ...]:
    """One stacked tensor: this rank's ranges of the partitioned dimension,
    one piece each (an interleaved partition reads its halves apart)."""
    if partition.whole:
        return (Piece.whole(shape),)
    dim = partition.axis(len(shape))
    return tuple(
        Piece.along(shape, dim, span)
        for span in partition.ranges(shape[dim], rank, size)
    )


def _stacked_pieces(
    partition: Partition,
    shape: tuple[int, ...],
    tensor_idx: int,
    stacked: int,
    rank: int,
    size: int,
) -> tuple[Piece, ...]:
    """One tensor per expert, ``tensor_idx`` of ``stacked``: nothing when
    the partition cuts the stacked axis and this rank does not own the
    expert; else the whole tensor, or — a partition of an inner dimension —
    this rank's contiguous chunk of the matching source dimension, the
    stacked axis shifted off (transformers' per-expert path chunks
    contiguously whatever the placement)."""
    if partition.whole:
        return (Piece.whole(shape),)
    dim = partition.axis(len(shape) + 1)
    if dim == 0:
        (owned,) = Partition(0).ranges(stacked, rank, size)
        return (Piece.whole(shape),) if tensor_idx in owned else ()
    (span,) = Partition(dim - 1).ranges(shape[dim - 1], rank, size)
    return (Piece.along(shape, dim - 1, span),)
