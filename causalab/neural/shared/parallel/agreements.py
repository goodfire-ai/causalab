"""The three host-side agreements and the training guard (``docs/model_parallelism.md`` §3, §6.5, §7).

A host-side decision that reads device state must be agreed before it steers
control flow, or ranks drift into different collective sequences and
deadlock. The three such decisions, and their rule:

- the ``RowBudget`` probe's bound — ``min`` over the model group
  ([`Agreements.min`][], called by ``budget.RowBudget.run``);
- the OOM retry — ``any`` over the model group before ``can_shrink`` /
  ``shrink``, so a rank that did not fail abandons the window too
  ([`Agreements.any`][], ``budget.RowBudget.out_of_memory``);
- fire counts — each member's tally summed over the pipeline stages before
  ``check_fires`` compares it to the declaration ([`summed_fires`][]); and,
  under context parallelism (§8.4), a **state** writer's steps — the
  positions it fired at, which the context ranks partition — made whole
  over the context group ([`whole_steps`][]), so the count compared to
  the declaration and the receipt's record are the whole frame's. A
  module-kind writer fires once on every context rank and is not summed.

And the §7 guard: after ``loss.backward()`` every trained featurizer
parameter's gradient is the all-reduce mean over the model group
([`average_gradients`][]). The invariant it rests on — at **every** site,
replicated or sharded — is that every rank's gradient is already the full
gradient: at a replicated site because the featurizer's forward and backward
see the same tensors everywhere, at a sharded site because the tap's
``fragment`` all-gathers (or all-reduce-sums) the ranks' gradient slices in
backward before they reach the write math (``parallel/autograd.py``), so
the mean of identical gradients is the gradient, exact at a power-of-two
group (``(g + g) / 2 == g``). The ``agreement`` check holds the invariant at
runtime — the gradients gathered and compared across the ranks before the
mean, a disagreement refused by name as [`GradientDisagreement`][] — and
is switched on by one environment variable, [`AGREEMENT_VARIABLE`][]
(``CAUSALAB_GRADIENT_AGREEMENT``), read once per fit by
``train.run_cohort_training`` ([`configured_agreement`][]); unset, the
check costs nothing. Its value is the tolerance, **relative to the largest
entry of the gathered gradients** ([`relative_disagreement`][]): the
largest difference between a rank's gradient and the first rank's, divided
by the largest absolute entry any rank holds. Relative, not absolute,
because the failure the check exists for scales with the gradient: a pairing
that hands a rank a *partial* — its own slice scattered into zeros, or a
gradient ``1 / size`` of the whole — differs from the whole by at least half
the whole's largest entry (``1 − 1/size ≥ 1/2`` for a group of two or
more), at every magnitude, while a real backend's reduction order leaves fp32
rounding between ranks — a few ulps, ``1e-7`` relative — whatever the
gradient's scale. An absolute tolerance would pass a broken pairing on a
small gradient and refuse rounding on a large one. So a tolerance is a float
in ``[0, 1/2)`` — ``0`` asks for bit identity, what the simulator delivers;
one half or more could not see a group of two averaging one partial and is
refused as a setting — and a malformed or negative value is refused by name
([`AgreementSetting`][]), never silently ignored. The training smokes and
the simulated training scenarios run with the variable set
(``tests/neural/engines/pytorch_hooks/conftest.py``: ``0`` on the simulator,
the measured band on ``gloo``), so a placement that breaks the pairing fails
loudly there rather than scaling the gradient quietly.

Under a pipeline (§6.5, §8.3) the featurizer's gradient is made on the stage
owning its site alone, so after the optimizer step the owner's parameters
are broadcast over the pipeline axis ([`sync_parameters`][]) and every
rank holds the trained copy — the publisher, stage 0, included.

*A group of one is the identity.* Every method returns its argument
untouched when the axis has one member, with the sizes read once at
construction — at world 1 none of this reaches the collective, which is what
lets today's single-device suites exercise the same code paths.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Iterable, Mapping

import torch

from causalab.neural.shared.fires import FireTally
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.environment import AGREEMENT_VARIABLE
from causalab.neural.shared.parallel.placement import AXES, Axis

__all__ = [
    "AGREEMENT_CEILING",
    "AGREEMENT_VARIABLE",
    "AgreementSetting",
    "Agreements",
    "GradientDisagreement",
    "average_gradients",
    "configured_agreement",
    "parse_agreement",
    "relative_disagreement",
    "summed_fires",
    "sync_parameters",
    "whole_steps",
]

#: Debug check, opt in (module docstring, §7): the relative tolerance of the
#: gradient agreement check — a float in ``[0, 1/2)`` — read once per fit
#: from the environment by ``train.run_cohort_training`` and handed to
#: [`average_gradients`][] as ``agreement``. Unset or empty, the check is
#: off and the guard is the plain mean; a value outside the range or not a
#: number is refused by name. ``0`` is bit identity across the ranks. Its
#: raw text is agreed across the world at the rendezvous
#: (``parallel/environment.py``, where the name is defined torch-free).

#: A tolerance of one half or more could not see a group of two averaging
#: one partial (``|g − g/2| / |g| = 1/2``), so the setting stops below it.
AGREEMENT_CEILING = 0.5


class AgreementSetting(ValueError):
    """[`AGREEMENT_VARIABLE`][] holds something that is not a tolerance: not
    a number, negative, not finite, or at least [`AGREEMENT_CEILING`][].
    Named after the variable so the operator's fix is one line."""


class GradientDisagreement(ValueError):
    """The ranks' gradients of one parameter differed before the mean — the
    §7 invariant broken: a tap whose ``fragment`` hands a rank only its own
    slice, or a ``whole`` off the graph. Names the parameter's index in the
    guard's list, the relative disagreement and the tolerance; raised on
    every rank alike, since every rank compares the same gathered tensors."""


def parse_agreement(text: str | None) -> float | None:
    """The tolerance [`AGREEMENT_VARIABLE`][] spells: ``None`` for an unset
    or empty variable (the check off), else the float in ``[0, 1/2)``.

    Raises:
        AgreementSetting: the text is not a finite number in the range.
    """
    if text is None or not text.strip():
        return None
    try:
        tolerance = float(text)
    except ValueError:
        raise AgreementSetting(
            f"{AGREEMENT_VARIABLE}={text!r} is not a number; the gradient agreement "
            f"tolerance is a float in [0, {AGREEMENT_CEILING}) relative to the "
            "gradient's largest entry (0 asks for bit identity), or unset for no "
            "check (docs/model_parallelism.md §7)"
        ) from None
    if (
        not math.isfinite(tolerance)
        or tolerance < 0.0
        or tolerance >= AGREEMENT_CEILING
    ):
        raise AgreementSetting(
            f"{AGREEMENT_VARIABLE}={text!r} is outside [0, {AGREEMENT_CEILING}): the "
            "gradient agreement tolerance is relative to the gradient's largest "
            "entry, and a partial gradient — one rank's slice, or 1/size of the "
            f"whole — differs from the whole by at least {AGREEMENT_CEILING} of it, "
            "so a tolerance that high would see nothing (docs/model_parallelism.md §7)"
        )
    return tolerance


def configured_agreement(environ: Mapping[str, str] = os.environ) -> float | None:
    """The tolerance the environment asks for ([`AGREEMENT_VARIABLE`][]),
    ``None`` when it asks for none; read once per fit by the training loop.

    Raises:
        AgreementSetting: a malformed value ([`parse_agreement`][]).
    """
    return parse_agreement(environ.get(AGREEMENT_VARIABLE))


class Agreements:
    """The three agreements bound to one collective, with the fast path."""

    __slots__ = ("_collective", "_sizes")

    def __init__(self, collective: Collective) -> None:
        self._collective = collective
        self._sizes: dict[Axis, int] = {axis: collective.size(axis) for axis in AXES}

    @property
    def collective(self) -> Collective:
        return self._collective

    def size(self, axis: Axis) -> int:
        return self._sizes[axis]

    def min(self, value: int, axis: Axis) -> int:
        if self._sizes[axis] == 1:
            return value
        return self._collective.agree_min(value, axis)

    def any(self, value: bool, axis: Axis) -> bool:
        if self._sizes[axis] == 1:
            return value
        return self._collective.agree_any(value, axis)

    def sum(self, value: int, axis: Axis) -> int:
        if self._sizes[axis] == 1:
            return value
        return self._collective.agree_sum(value, axis)


def average_gradients(
    parameters: Iterable[torch.nn.Parameter],
    collective: Collective,
    *,
    axis: Axis = "model",
    agreement: float | None = None,
) -> None:
    """Replace each parameter's ``.grad`` by its mean over ``axis`` (§7);
    a parameter without a gradient is skipped. The identity at a group of
    one, with no call on the collective.

    ``agreement`` is the runtime check (module docstring): the ranks'
    gradients are gathered and each compared to the first rank's before the
    mean, a [`relative_disagreement`][] above it refused as
    [`GradientDisagreement`][] — ``0.0`` asks for bit identity, what the
    simulator delivers; a real backend's reduction order can leave fp32
    rounding between ranks. One extra collective per parameter; ``None``
    (the default, and an unset [`AGREEMENT_VARIABLE`][]) costs nothing.
    """
    size = collective.size(axis)
    if size == 1:
        return
    for index, parameter in enumerate(parameters):
        grad = parameter.grad
        if grad is None:
            continue
        if agreement is not None:
            _check_agreement(grad, collective, axis, agreement, index)
        parameter.grad = collective.all_reduce_sum(grad, axis).div_(size)


def sync_parameters(
    tensors: Iterable[torch.Tensor],
    owner: int,
    collective: Collective,
    *,
    axis: Axis = "pipeline",
) -> None:
    """Overwrite each of ``tensors``, in order, with the ``owner`` rank's
    copy broadcast over ``axis`` (§7, §8.3): the trained featurizer's
    parameters and the buffers derived from them, after the optimizer step,
    so every rank of the pipeline holds what the owning stage fitted. The
    order is the caller's and must be the same on every rank (a stage's
    ``state_dict`` is); the identity at a group of one, with no call on the
    collective."""
    if collective.size(axis) == 1:
        return
    mine = collective.rank(axis) == owner
    with torch.no_grad():
        for tensor in tensors:
            received = collective.broadcast(tensor if mine else None, owner, axis)
            if not mine:
                tensor.copy_(received)


def relative_disagreement(gathered: torch.Tensor) -> float:
    """How far the ranks' gradients are from one another, relative to their
    scale (module docstring): ``gathered`` holds one rank's flattened
    gradient per row, and the result is the largest ``|g_r − g_0|`` over the
    largest ``|entry|`` any rank holds — ``0.0`` when every rank holds the
    same tensor, an empty one, or zeros everywhere (nothing to disagree
    about, and no scale to divide by). Identical gradients are ``0.0`` to
    the bit; a rank holding ``1 / size`` of a non-zero gradient is at least
    ``1 − 1/size``, half or more."""
    if gathered.numel() == 0:
        return 0.0
    scale = float(gathered.abs().max())
    if scale == 0.0:
        return 0.0
    worst = float((gathered - gathered[0]).abs().max())
    return worst / scale


def _check_agreement(
    grad: torch.Tensor, collective: Collective, axis: Axis, tolerance: float, index: int
) -> None:
    ranks = collective.all_gather(grad.detach().reshape(1, -1), 0, axis)
    worst = relative_disagreement(ranks)
    if worst > tolerance:
        raise GradientDisagreement(
            f"parameter {index}: the ranks' gradients differ by {worst:.3e} of their "
            f"largest entry before the mean over {axis!r} (tolerance {tolerance:.1e}, "
            f"{AGREEMENT_VARIABLE}); every rank's gradient should already be the "
            "full gradient (docs/model_parallelism.md §7)"
        )


def summed_fires(
    tally: FireTally, agreements: Agreements, *, axis: Axis = "pipeline"
) -> FireTally:
    """The tally as the whole pipeline saw it: every member's count summed
    over the stages, in member order (the same order on every rank — the
    declaration is the document's, identical everywhere), the declaration
    and the state steps kept. A hook on a stage that does not own the module
    never fires, so the sum is what the declared count is compared to (§6.5)."""
    members = sorted(set(tally.expected) | set(tally.counts))
    return dataclasses.replace(
        tally,
        expected=dict(tally.expected),
        counts={
            member: agreements.sum(tally.counts.get(member, 0), axis)
            for member in members
        },
        steps={member: set(steps) for member, steps in tally.steps.items()},
    )


def whole_steps(
    tally: FireTally,
    collective: Collective,
    members: Iterable[str],
    *,
    padded_len: int,
    axis: Axis = "context",
) -> FireTally:
    """The tally with each of ``members``' fired steps unioned over ``axis``
    and its count set to the union's size (§8.4): a state writer fires at
    the steps of its rank's chunk, and the chunks partition the frame's
    positions, so the union — a 0/1 vector over the ``padded_len``
    positions summed over the group — is the whole forward's firing. Other
    members are untouched; a group of one is the identity."""
    named = tuple(members)
    if collective.size(axis) == 1 or not named:
        return tally
    steps = {member: set(steps) for member, steps in tally.steps.items()}
    counts = dict(tally.counts)
    # the vector crosses the collective, so it is born on the collective's
    # device (``Collective.device``: NCCL refuses a CPU tensor)
    device = collective.device
    for member in named:
        fired = torch.zeros(padded_len, dtype=torch.int64, device=device)
        for step in steps.get(member, ()):
            fired[step] = 1
        union = collective.all_reduce_sum(fired, axis)
        whole = {int(step) for step in torch.nonzero(union).flatten().tolist()}
        steps[member] = whole
        counts[member] = len(whole)
    return dataclasses.replace(
        tally, expected=dict(tally.expected), counts=counts, steps=steps
    )
