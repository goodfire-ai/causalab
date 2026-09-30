"""The one autograd function two styles share (``docs/model_parallelism.md``
§6.3, §6.6, §7): the identity forward whose backward all-reduce-**sums** the
gradient over a group of the collective.

A style that runs a module replicated and keeps a rank-local slice of its
output — transformers' ``ep_router`` (this rank's experts' scores) and the
engine's ``kv_replicated`` (the one KV head this rank's query heads read) —
is forward-only slicing: in backward the gradient that flows through the
slice into the module's **input** is this rank's share alone, a partial sum
over the group that nothing sums. At inference nothing reads it; a fit of a
featurizer below the module does, and the §7 guard's mean over the group of
``n`` partials is ``1 / n`` of the whole. [`summed_over`][] on the module's
input makes every rank's input gradient the whole — the same shape as
transformers' ``_AllReduceBackward``, which the colwise styles get from
DTensor's ``from_local`` and the experts' style applies to its own input.
Every rank's partial added, nothing divided. The forward is the identity,
bit for bit.

The sum runs through the [`Collective`][]
— ``TorchCollective`` in production, the tests' ``SimulatedWorld`` in the
simulated tiers — so the function is held to the collective contract like
every other backward collective (``parallel/autograd.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis

__all__ = ["SumGradient", "summed_over"]


class SumGradient(torch.autograd.Function):
    """The identity forward; in backward the gradient all-reduce-summed over
    the group of ``axis``."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, collective: Collective, axis: Axis
    ) -> torch.Tensor:
        ctx.collective = collective
        ctx.axis = axis
        return tensor

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return ctx.collective.all_reduce_sum(grad.contiguous(), ctx.axis), None, None


def summed_over(
    tensor: torch.Tensor, axis: Axis, collective: Collective
) -> torch.Tensor:
    """``tensor`` unchanged, its gradient summed over the group of ``axis``
    on backward."""
    return SumGradient.apply(tensor, collective, axis)
