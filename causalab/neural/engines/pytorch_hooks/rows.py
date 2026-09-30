"""Data parallelism over rows (``docs/model_parallelism.md`` §7, §8.3): this
replica's share of every fit minibatch, and the two agreements that make the
split fit equal the unsplit one.

Under ``--parallel dp=N:rows`` every replica runs every point, and inside a
fit each optimizer step's minibatch — ``train.batch.pairs`` rows, the
contiguous index blocks ``train._prepare_fit`` cuts — is split into ``N``
contiguous row slices, one per replica, sizes differing by at most one
([`RowSplit.slice_for`][]). Each replica runs its slice's forward and
backward; what has to be agreed is then exactly two things:

- **the loss.** The unsplit minibatch's objective is ``Σ_terms w · mean over
  the N rows`` (a regularizer term is a function of the stages alone, the
  same on every replica). Replica ``r`` computes the same objective over its
  ``n_r`` rows and **weighs** it by its share ``n_r / N``
  ([`RowSplit.weigh`][]); the shares sum to one, so the replicas' weighed
  losses sum to the unsplit loss exactly in real arithmetic — for the metric
  terms because ``Σ_r (n_r / N) · mean_r = mean``, for a regularizer because
  ``Σ_r n_r / N = 1``. The record of the update (``last_loss``, the
  ``term.*`` values a trajectory checkpoint carries) is agreed the same way
  ([`RowSplit.agree_record`][]).
- **the gradient.** Backward runs on the weighed local loss, so each
  replica's ``.grad`` is its share of the unsplit gradient and the
  ``all_reduce_sum`` over the ``data`` axis ([`RowSplit.reduce_gradients`][])
  is the unsplit gradient — up to the order the reduction adds in, which is
  the one place the split fit and the world-1 fit differ (§7's training
  band). No division follows the reduction: the ``1 / N`` is already in the
  shares.

Evaluation splits the eval split's rows the same way and agrees each
metric's ``(sum, count)`` over the replicas ([`RowSplit.agree_means`][]),
so the score — and every early-stop decision made from it — is the same
number on every replica; a divergent early stop would be a deadlock (§3).

*Inactive is the identity.* A split built for the ``points`` mode, or at a
data axis of one, returns every argument untouched and never reaches the
collective — the world-1 suites run this code with a refusing collective.
Every collective here runs **after** a budget window's body, never inside
one (§3: a body stays free of collectives so an out-of-memory rank and a
completing rank reach the same next collective).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from causalab.neural.engines.pytorch_hooks.budget import LOCKSTEP_AXES
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.placement import Axis
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry

__all__ = ["WHOLE", "RowSplit"]

#: The axis the replicas of a rows split live on.
_AXIS: Axis = "data"


class RowSplit:
    """This replica's place in a rows split (module docstring): ``active``
    when the geometry's data mode is ``rows`` and the data axis has more
    than one member; the identity otherwise.

    ``replica`` is this rank's index on the data axis, of ``replicas``.
    ``budget_axes`` is what a [`RowBudget`][causalab.neural.engines.pytorch_hooks.budget.RowBudget] agrees over: the
    lockstep axes (``budget.LOCKSTEP_AXES`` — model, pipeline, context),
    and the data axis too under an active split (every replica runs the
    same windows in lockstep, so a probe's bound and an out-of-memory window
    are agreed across the replicas too).
    """

    __slots__ = ("active", "collective", "geometry", "replica", "replicas")

    def __init__(self, collective: Collective, geometry: ParallelGeometry) -> None:
        self.collective = collective
        self.geometry = geometry
        size = collective.size(_AXIS)
        self.active: bool = geometry.data_mode == "rows" and size > 1
        self.replicas: int = size if self.active else 1
        self.replica: int = collective.rank(_AXIS) if self.active else 0

    @property
    def budget_axes(self) -> tuple[Axis, ...]:
        return (*LOCKSTEP_AXES, _AXIS) if self.active else LOCKSTEP_AXES

    # -- the rows ---------------------------------------------------------------

    def slice_for(self, indices: Sequence[int]) -> list[int]:
        """This replica's contiguous slice of ``indices`` — the ``replica``-th
        of ``replicas`` slices in order, sizes differing by at most one with
        the first slices longer, so the slices partition ``indices`` and no
        replica idles. Every index at an inactive split.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.data`` — fewer indices
                than replicas, so a replica would hold no rows; decided from
                the minibatch alone, identically on every rank.
        """
        rows = list(indices)
        if not self.active:
            return rows
        if len(rows) < self.replicas:
            raise ProtocolError(
                "P4",
                f"dp={self.replicas}:rows over a minibatch of {len(rows)} row"
                f"{'s' if len(rows) != 1 else ''}: a replica would hold no rows; "
                "every replica holds at least one, so every minibatch — an "
                "epoch's remainder included — needs at least dp rows (choose "
                "train.batch.pairs dividing the row count, or fewer replicas)",
                path="--parallel.data",
            )
        base, extra = divmod(len(rows), self.replicas)
        start = self.replica * base + min(self.replica, extra)
        return rows[start : start + base + (1 if self.replica < extra else 0)]

    # -- the loss ---------------------------------------------------------------

    def weigh(self, loss: torch.Tensor, rows: int, total: int) -> torch.Tensor:
        """``loss`` weighed by this replica's share ``rows / total`` of the
        minibatch (module docstring), so the replicas' weighed losses — and
        their gradients — sum to the unsplit minibatch's. ``loss`` itself,
        untouched, at an inactive split."""
        if not self.active:
            return loss
        return loss * (rows / total)

    def reduce_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Replace each parameter's ``.grad`` by its sum over the replicas —
        the unsplit minibatch's gradient, since each replica's loss carried
        its share; a parameter without a gradient is skipped. Nothing at an
        inactive split."""
        if not self.active:
            return
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            parameter.grad = self.collective.all_reduce_sum(grad, _AXIS)

    def agree_record(
        self, values: Sequence[float], rows: int, total: int, device: torch.device
    ) -> list[float]:
        """The replicas' ``values`` — each a mean over its own rows — as the
        means over every replica's rows: ``Σ_r (rows_r / total) · value_r``,
        one float64 reduction over the data axis on ``device``. ``values``
        themselves at an inactive split."""
        if not self.active:
            return list(values)
        local = torch.tensor(values, dtype=torch.float64, device=device) * (
            rows / total
        )
        return [float(v) for v in self.collective.all_reduce_sum(local, _AXIS)]

    # -- evaluation -------------------------------------------------------------

    def agree_means(
        self, sums: Sequence[float], counts: Sequence[int], device: torch.device
    ) -> list[float]:
        """Each metric's mean over every replica's eval rows from the local
        ``(sum, count)`` pairs — one float64 reduction over the data axis on
        ``device`` — ``0.0`` for a metric no replica scored a row of. The
        local ``sum / count`` at an inactive split."""
        if len(sums) != len(counts):
            raise ValueError(f"{len(sums)} sums but {len(counts)} counts")
        if self.active:
            local = torch.tensor(
                [*sums, *(float(c) for c in counts)], dtype=torch.float64, device=device
            )
            agreed = self.collective.all_reduce_sum(local, _AXIS).tolist()
            sums = agreed[: len(counts)]
            counts = [int(round(c)) for c in agreed[len(counts) :]]
        return [s / c if c else 0.0 for s, c in zip(sums, counts, strict=True)]


#: The inactive split every executor carries unless the engine set another:
#: world 1, the identity.
WHOLE = RowSplit(SOLO, ONE)
