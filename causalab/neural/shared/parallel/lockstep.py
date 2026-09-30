"""The workflow lockstep over the mesh (``docs/model_parallelism.md`` §3, §11):
the production [`Lockstep`][causalab.protocol.lockstep.Lockstep], carrying the
joiner's [`Outcome`][] to every rank through
the one [`Collective`][] seam — so the ``SimulatedWorld``
runs the same code under a seeded schedule, and a real world runs it over
the mesh's process groups without carving a new one.

**The chain.** The joiner is the rank at local index 0 on every mesh axis.
Its outcome, encoded as bytes, is broadcast from local index 0 along the
[`CHAIN`][] of axes in order — ``data``, ``pipeline``, ``context``,
``model`` — and a rank joins the round on an axis only when its coordinates
on every *later* axis are all zero: those are exactly the groups whose
source already holds the value (it received it in the earlier rounds, or
is the joiner), and after the round every rank with zeros on the later
axes holds it. The last round covers the world. An axis of size one is no
round. A rank that joined a round it did not qualify for would seat a
non-holder as its group's source with nothing to send — the simulator
refuses that as a misuse; a real backend would broadcast garbage — which
is why the rule is not "every rank, every axis" (its test is the mutation).
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.neural.shared.parallel.collective import Collective
from causalab.protocol.lockstep import LockstepError, Outcome, decode, encode
from causalab.protocol.parallel import Axis

__all__ = ["CHAIN", "CollectiveLockstep"]

#: The mesh axes the outcome is broadcast along, outermost first.
CHAIN: tuple[Axis, ...] = ("data", "pipeline", "context", "model")


class CollectiveLockstep:
    """A [`Lockstep`][causalab.protocol.lockstep.Lockstep] over ``collective``.

    The payload travels on the collective's device (``Collective.device``:
    ``TorchCollective`` refuses a tensor elsewhere; the simulator's and
    ``Solo`` are CPU-side).
    """

    def __init__(self, collective: Collective) -> None:
        self.collective = collective
        self.device: torch.device = collective.device

    def _joins(self) -> bool:
        return all(self.collective.rank(axis) == 0 for axis in CHAIN)

    def payload(self, outcome: Outcome) -> torch.Tensor:
        """``outcome`` as a byte tensor on the collective's device."""
        data = encode(outcome)
        return torch.tensor(list(data), dtype=torch.uint8, device=self.device)

    def outcome(self, payload: torch.Tensor) -> Outcome:
        """The outcome a received byte tensor carries.

        Raises:
            LockstepError: the bytes are not an encoded outcome.
        """
        return decode(bytes(payload.to("cpu").tolist()))

    def agree(self, outcome: Outcome | None) -> Outcome:
        """The joiner's ``outcome`` on every rank (module docstring).

        Raises:
            LockstepError: a rank other than the joiner passed an outcome, or
                the joiner passed none.
        """
        joins = self._joins()
        coordinates = {axis: self.collective.rank(axis) for axis in CHAIN}
        if joins and outcome is None:
            raise LockstepError(
                "the joiner — the rank at local index 0 on every mesh axis — "
                "passes its outcome to agree, not None"
            )
        if not joins and outcome is not None:
            raise LockstepError(
                f"a rank at mesh coordinates {coordinates} is not the joiner and "
                "follows: it passes None to agree and receives the joiner's outcome"
            )
        payload: Any = self.payload(outcome) if outcome is not None else None
        for index, axis in enumerate(CHAIN):
            if self.collective.size(axis) == 1:
                continue
            if any(self.collective.rank(later) != 0 for later in CHAIN[index + 1 :]):
                continue  # not this round: the group's source holds nothing yet
            payload = self.collective.broadcast(payload, 0, axis)
        return self.outcome(payload)
