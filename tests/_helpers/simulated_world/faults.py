"""Scripted device memory: the simulated ``budget.Meter`` (§10.1).

A `SimulatedMeter` holds one script per rank — a ``(peak, available)``
reading per probe — and the ``(rank, step)`` pairs at which a probe runs out
of memory. ``for_rank`` hands a rank the [`Meter`][causalab.neural.engines.pytorch_hooks.budget.Meter] its ``RowBudget``
takes; ``check`` lets a scenario's window body raise the same OOM at a train
step the meter never sees (a resolved budget runs its windows unmeasured).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch

from tests._helpers.simulated_world.errors import MeterScriptExhausted


class SimulatedMeter:
    def __init__(
        self,
        readings: Mapping[int, Sequence[tuple[int, int]]],
        oom_at: Iterable[tuple[int, int]] = (),
    ) -> None:
        self._readings = {rank: tuple(script) for rank, script in readings.items()}
        self._oom = frozenset(oom_at)
        self._probes: dict[int, int] = defaultdict(int)

    def for_rank(self, rank: int) -> RankMeter:
        return RankMeter(self, rank)

    def probes(self, rank: int) -> int:
        """How many times ``rank`` has probed so far."""
        return self._probes[rank]

    def check(self, rank: int, step: int) -> None:
        """Raise the scripted OOM for ``(rank, step)``, if there is one."""
        if (rank, step) in self._oom:
            raise torch.OutOfMemoryError(
                f"simulated out of memory on rank {rank} at step {step}"
            )

    def reading(self, rank: int) -> tuple[int, int]:
        """The next scripted reading for ``rank``, advancing its probe count;
        the scripted OOM for that probe takes precedence over the reading."""
        step = self._probes[rank]
        self._probes[rank] = step + 1
        self.check(rank, step)
        script = self._readings.get(rank, ())
        if step >= len(script):
            raise MeterScriptExhausted(
                f"rank {rank} probed {step + 1} times; its script has {len(script)} readings"
            )
        return script[step]


@dataclass(frozen=True)
class RankMeter:
    """``budget.Meter`` for one rank: the probe is scripted, not measured. A
    probe that runs out of memory raises **instead of** running the window —
    the window produced nothing, as a real OOM mid-forward leaves nothing."""

    meter: SimulatedMeter
    rank: int

    def measure(self, run: Callable[[], None]) -> tuple[int, int]:
        reading = self.meter.reading(self.rank)
        run()
        return reading
