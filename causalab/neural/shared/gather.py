"""Gather positions with backward behavior derived from the index.

``dense_index`` and ``flat_index`` determine whether every ``(row, position)``
pair is distinct. A distinct index uses a non-accumulating scatter backward;
repeated pairs use autograd's accumulating backward. The decision travels
with ``PositionIndex`` so callers cannot mismatch the index and operation.
Padded rows that repeat their final position require accumulation.

Index tensors are cached by position table and evicted by count. CUDA's
non-accumulating ``index_put_`` is refused under Torch deterministic mode,
even for a distinct index. Supporting that mode requires a fallback.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Sequence

import torch

__all__ = [
    "PositionIndex",
    "dense_index",
    "flat_index",
    "gather_positions",
    "rows_are_distinct",
    "splice_features",
]

#: How many distinct position tables' index tensors stay resident per
#: process. A table is a few hundred bytes at eval width (a fit's tables are
#: larger and churn — module docstring); an eval workflow addresses a few
#: dozen.
_INDEX_CACHE_SIZE = 1024


@dataclasses.dataclass(frozen=True, eq=False)
class PositionIndex:
    """The index tensors a position table gathers with, and what that table
    can hold: ``tensor[row_ids, idx]`` is the gather, ``distinct`` says no two
    entries of it name the same ``(row, position)`` element — decided on the
    host, from the lists, when the index was built. Compared by identity
    (``eq=False``): one object per cached table, and a field-wise ``__eq__``
    over tensors would raise on its own truth value."""

    row_ids: torch.Tensor
    idx: torch.Tensor
    distinct: bool

    @property
    def pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The advanced index itself — ``tensor[index.pair]``."""
        return self.row_ids, self.idx


def rows_are_distinct(
    per_row: Sequence[Sequence[int]], rows: Sequence[int] | None = None
) -> bool:
    """Whether no two entries of a position table name the same ``(row,
    position)`` element — the condition under which a gather at those
    positions has one source per element. ``rows`` names the batch row each
    table row belongs to (the table's own order by default); a batch row
    named twice makes overlapping positions collide even when every table
    row is distinct on its own."""
    row_names = range(len(per_row)) if rows is None else rows
    pairs = [(r, p) for r, row in zip(row_names, per_row) for p in row]
    return len(set(pairs)) == len(pairs)


@functools.lru_cache(maxsize=_INDEX_CACHE_SIZE)
def _dense_index(
    rows: tuple[int, ...], per_row: tuple[tuple[int, ...], ...], device: str
) -> PositionIndex:
    return PositionIndex(
        row_ids=torch.tensor(rows, dtype=torch.long, device=device).unsqueeze(1),
        idx=torch.tensor(per_row, dtype=torch.long, device=device),
        distinct=rows_are_distinct(per_row, rows),
    )


@functools.lru_cache(maxsize=_INDEX_CACHE_SIZE)
def _flat_index(
    rows: tuple[int, ...], per_row: tuple[tuple[int, ...], ...], device: str
) -> PositionIndex:
    return PositionIndex(
        row_ids=torch.tensor(
            [r for r, row in zip(rows, per_row) for _ in row],
            dtype=torch.long,
            device=device,
        ),
        idx=torch.tensor(
            [p for row in per_row for p in row], dtype=torch.long, device=device
        ),
        distinct=rows_are_distinct(per_row, rows),
    )


def dense_index(
    per_row: Sequence[Sequence[int]],
    device: torch.device | str,
    *,
    rows: Sequence[int] | None = None,
) -> PositionIndex:
    """The index for a position table whose rows all have one width:
    ``row_ids (rows, 1)`` against ``idx (rows, width)``. ``rows`` names the
    batch rows the table's entries belong to (the table's own order by
    default).

    Built once per distinct ``(rows, table, device)`` and reused: the
    executor resolves the same positions for the same batch on every forward
    (each read and write, every hook call), and each ``torch.tensor(list,
    device="cuda")`` is a host→device copy the launch queue waits on. The
    tensors are only ever read from, so sharing them is safe.
    """
    rows_key = tuple(range(len(per_row))) if rows is None else tuple(rows)
    return _dense_index(rows_key, tuple(tuple(r) for r in per_row), str(device))


def flat_index(
    per_row: Sequence[Sequence[int]],
    device: torch.device | str,
    *,
    rows: Sequence[int] | None = None,
) -> PositionIndex:
    """The index as two flat vectors, one entry per position of every row —
    the ragged form of [`dense_index`][], the same caching."""
    rows_key = tuple(range(len(per_row))) if rows is None else tuple(rows)
    return _flat_index(rows_key, tuple(tuple(r) for r in per_row), str(device))


class _DistinctGather(torch.autograd.Function):
    """``tensor[row_ids, idx]`` for an index with one source per element:
    the backward scatters the gradient without accumulating."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, row_ids: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(row_ids, idx)
        ctx.source_shape = tensor.shape
        return tensor[row_ids, idx]

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        row_ids, idx = ctx.saved_tensors
        out = grad.new_zeros(ctx.source_shape)
        out.index_put_((row_ids, idx), grad, accumulate=False)
        return out, None, None


def gather_positions(tensor: torch.Tensor, index: PositionIndex) -> torch.Tensor:
    """``tensor[index.pair]``, with the sort-free backward when the index
    repeats no element ([`PositionIndex.distinct`][]).

    The forward is the plain advanced index either way, so the value is the
    same object autograd would produce; only the backward differs, and only
    when a gradient can flow (a tensor that needs none, or a no-grad forward,
    takes the plain index and pays nothing).
    """
    if not index.distinct or not torch.is_grad_enabled() or not tensor.requires_grad:
        return tensor[index.pair]
    return _DistinctGather.apply(tensor, index.row_ids, index.idx)


def splice_features(
    landed: torch.Tensor, fslice: slice, value: torch.Tensor
) -> torch.Tensor:
    """``landed`` with ``value`` in its feature slice ``fslice`` — what a
    landing writes back over the positions it gathered ``landed`` from.

    When the slice is the whole feature axis the written value *is* the
    whole slice, and ``value`` is returned as it is: no copy, and no second
    gather of the positions to splice into (the landing used to re-gather
    them, paying the sorted backward a second time).
    """
    if fslice == slice(None):
        return value
    out = landed.clone()
    out[..., fslice] = value
    return out
