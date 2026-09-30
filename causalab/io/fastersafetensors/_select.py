"""Selections: an index over a tensor, as the box the engine reads.

An index — ints, slices, ``Ellipsis``, ``None`` — becomes a
`Selection`: per dimension the ``[lo, hi)`` range the box covers (a
stepped slice covers its first through its last element, an int covers one)
and the ``view`` that turns a tensor of the box's shape into the result.
That is all Python decides. Which bytes the box is, how the runs are
coalesced under the profile's policy for the file's storage, and where each
piece lands are ``fst_core::select``'s, reached through
``_core.select_reads`` with the box's ranges; a [`Shard`][causalab.neural.engines.pytorch_hooks.weights.Shard] is
``Selection::shard`` through ``_core.shard_ranges``. A stepped slice reads
its box and is narrowed in torch afterwards; ints and ``None`` only reshape.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import TypeAlias

import causalab.io.fastersafetensors._core as _core
from .errors import SelectError

Item: TypeAlias = int | slice | EllipsisType | None
"""One component of an index."""

SelectItem: TypeAlias = int | slice | EllipsisType
"""One component of a ``load_files(select=...)`` selection."""

_INDEX_TYPES_MESSAGE = (
    "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
    "Variables are valid indices (got {})"
)


def _expand(index: object, ndim: int) -> list[Item]:
    """Validate an index and expand ``Ellipsis`` into full slices, so that the
    result has exactly ``ndim`` consuming items (ints and slices) plus any
    ``None``s. The error classes and messages are the reference's."""
    raw: list[object] = list(index) if isinstance(index, tuple) else [index]
    items: list[Item] = []
    for item in raw:
        if isinstance(item, str):
            raise TypeError(f"new(): invalid data type '{type(item).__name__}'")
        if isinstance(item, bool) or not isinstance(
            item, int | slice | EllipsisType | None
        ):
            raise IndexError(_INDEX_TYPES_MESSAGE.format(type(item).__name__))
        items.append(item)
    consuming = sum(isinstance(i, int | slice) for i in items)
    if consuming > ndim:
        raise IndexError(f"too many indices for tensor of dimension {ndim}")
    ellipses = [i for i, item in enumerate(items) if item is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    fill: list[Item] = [slice(None)] * (ndim - consuming)
    if ellipses:
        at = ellipses[0]
        return items[:at] + fill + items[at + 1 :]
    return items + fill


def _check_int(index: int, dim: int, size: int) -> int:
    if index < -size or index >= size:
        raise IndexError(
            f"index {index} is out of bounds for dimension {dim} with size {size}"
        )
    return index + size if index < 0 else index


@dataclass(frozen=True, slots=True)
class Axis:
    """One dimension of a box: the elements ``[lo, hi)`` it covers, the step
    the index walks them with, and whether the index was an int (the
    dimension is dropped from the result)."""

    lo: int
    hi: int
    step: int = 1
    squeeze: bool = False

    @property
    def extent(self) -> int:
        return self.hi - self.lo


@dataclass(frozen=True, slots=True)
class Selection:
    """A box over a tensor and the index that turns the box into the result."""

    shape: tuple[int, ...]
    axes: tuple[Axis, ...]
    """One per dimension of ``shape``."""
    view: tuple[Item, ...]
    """The index to apply to a tensor of `box_shape` to get the result:
    ``0`` for a squeezed dimension, a slice relative to the box for the
    others, ``None`` where a new axis is inserted."""

    @classmethod
    def full(cls, shape: Sequence[int]) -> Selection:
        """The whole tensor."""
        return select(tuple(shape), Ellipsis)

    @classmethod
    def from_ranges(
        cls, shape: Sequence[int], ranges: Sequence[tuple[int, int]]
    ) -> Selection:
        """A plain box: the ranges as they are, nothing squeezed or stepped."""
        axes = tuple(Axis(lo, hi) for lo, hi in ranges)
        return cls(tuple(shape), axes, tuple(slice(0, axis.extent, 1) for axis in axes))

    @property
    def ranges(self) -> list[tuple[int, int]]:
        """``(lo, hi)`` per dimension: what ``_core.select_reads`` takes."""
        return [(axis.lo, axis.hi) for axis in self.axes]

    @property
    def box_shape(self) -> tuple[int, ...]:
        return tuple(axis.extent for axis in self.axes)

    @property
    def result_shape(self) -> tuple[int, ...]:
        out: list[int] = []
        dim = 0
        for item in self.view:
            if item is None:
                out.append(1)
                continue
            if isinstance(item, slice):
                out.append(len(range(*item.indices(self.axes[dim].extent))))
            dim += 1
        return tuple(out)

    def wanted_bytes(self, itemsize: int) -> int:
        """Bytes of the result: what the caller asked for."""
        return prod(self.result_shape) * itemsize

    def allocation_bytes(self, itemsize: int) -> int:
        """Peak tensor storage: result plus the scratch box for stepped slices."""
        scratch = (
            0 if self.is_plain or self.is_empty else prod(self.box_shape) * itemsize
        )
        return self.wanted_bytes(itemsize) + scratch

    @property
    def is_plain(self) -> bool:
        """Whether the result's bytes are the box's bytes in the box's order
        (no stepped slice): ints and ``None`` only reshape."""
        return all(axis.step == 1 for axis in self.axes)

    @property
    def is_empty(self) -> bool:
        return any(axis.extent == 0 for axis in self.axes)


def select(shape: tuple[int, ...], index: object) -> Selection:
    """Turn an index over ``shape`` into a `Selection`, raising the
    reference's ``IndexError`` / ``TypeError`` / ``ValueError`` for what the
    reference refuses."""
    items = _expand(index, len(shape))
    axes: list[Axis] = []
    view: list[Item] = []
    dim = 0
    for item in items:
        if item is None:
            view.append(None)
            continue
        size = shape[dim]
        if isinstance(item, int):
            at = _check_int(item, dim, size)
            axes.append(Axis(at, at + 1, 1, True))
            view.append(0)
        else:
            assert isinstance(item, slice)
            start, stop, step = item.indices(size)  # a zero step raises here
            if step < 0:
                raise ValueError("step must be greater than zero")
            count = len(range(start, stop, step))
            lo = start
            hi = start + (count - 1) * step + 1 if count else start
            axes.append(Axis(lo, hi, step))
            view.append(slice(0, hi - lo, step))
        dim += 1
    return Selection(shape, tuple(axes), tuple(view))


@dataclass(frozen=True, slots=True)
class Shard:
    """Shard ``rank`` of ``world`` equal shards along ``dim``; resolved
    against a tensor's shape when the header is known (``Selection::shard``
    in the core: a dimension that does not divide is a [`SelectError`][])."""

    dim: int
    rank: int
    world: int

    def __post_init__(self) -> None:
        # ``dim`` is checked against a shape; ``rank`` and ``world`` stand alone.
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise SelectError(
                f"shard rank {self.rank} is not within a world of {self.world}"
            )

    def axis(self, ndim: int) -> int:
        """``dim`` as a non-negative index into an ``ndim``-d shape."""
        dim = self.dim + ndim if self.dim < 0 else self.dim
        if not 0 <= dim < ndim:
            raise SelectError(
                f"shard dim {self.dim} is out of range for a {ndim}-d shape"
            )
        return dim

    def ranges(self, shape: Sequence[int]) -> list[tuple[int, int]]:
        """``(lo, hi)`` per dimension of this shard of a tensor of ``shape``."""
        return _core.shard_ranges(
            list(shape), self.axis(len(shape)), self.rank, self.world
        )


SelectSpec: TypeAlias = SelectItem | Sequence[SelectItem] | Shard
"""What ``load_files(select=...)`` accepts per tensor name."""


def select_shards(
    names: Iterable[str], dim: int, rank: int, world: int
) -> dict[str, Shard]:
    """A ``select`` mapping giving every name its shard ``rank`` of ``world``
    along ``dim`` — for ``load_files(select=select_shards(...))``."""
    return dict.fromkeys(names, Shard(dim, rank, world))


def resolve(name: str, shape: tuple[int, ...], spec: SelectSpec) -> Selection:
    """The selection ``spec`` names on tensor ``name`` of ``shape``; anything
    the index machinery refuses becomes a [`SelectError`][] naming the
    tensor."""
    if isinstance(spec, Shard):
        return Selection.from_ranges(shape, spec.ranges(shape))
    index: object = spec
    if isinstance(spec, Sequence) and not isinstance(spec, str):
        index = tuple(spec)
    try:
        return select(shape, index)
    except (IndexError, TypeError, ValueError) as err:
        raise SelectError(f"selection for {name!r} over shape {shape}: {err}") from err
