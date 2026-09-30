"""The schedule: what decides which rank runs next (``docs/model_parallelism.md`` §10.2, §10.5).

A `Schedule` is one of two things. A **seed** is a whole run's worth of
randomness in one integer: at every switch ``random.Random(seed).choice`` over
the sorted candidates. A **tape** is a sequence of non-negative integers read
one entry per *choice*: the ``i``-th switch with more than one candidate picks
``candidates[tape[i] % len(candidates)]``; a switch with a single candidate
takes it and keeps the tape's entry — the baton parks ranks one by one at a
rendezvous, so most hand-offs are forced, and a tape spending an entry on
each would name few real decisions — and a choice past the tape's end picks
the lowest candidate, so the empty tape is rank order and a tape of length
``k`` is "these ``k`` decisions, then rank order". A tape is what
hypothesis draws (``tests/_helpers/parallel_strategies.py:schedules``): it
shrinks towards the empty tape, so a failing interleaving is reported as the
few early choices that provoke it, and two tapes that share a prefix make the
same choices over that prefix — the property that lets the shrinker cut a
tape without changing what came before the cut.

Every driver of a simulated interleaving — the baton scheduler of
`.scheduler`, the heartbeat clock of `.heartbeat` — asks one
`Picker` for its choices and records each as a `Pick`, so a
test can assert on the order it got.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from tests._helpers.simulated_world.errors import ScheduleError

#: A seed, or a tape of non-negative integers (module docstring).
Schedule = int | Sequence[int]


@dataclass(frozen=True)
class Pick:
    """One switch: the ranks that could have run, sorted, and the one chosen."""

    runnable: tuple[int, ...]
    rank: int


class Picker(Protocol):
    def pick(self, candidates: Sequence[int]) -> int:
        """One of ``candidates`` — sorted, non-empty — advancing the schedule."""
        ...


class Seeded:
    """``random.Random(seed).choice`` at every switch."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def pick(self, candidates: Sequence[int]) -> int:
        return self._rng.choice(candidates)


class Taped:
    """``candidates[tape[i] % len(candidates)]`` at the ``i``-th choice — a
    switch with more than one candidate; the lowest candidate once the tape
    is spent. A forced switch takes its one candidate and reads nothing."""

    def __init__(self, tape: tuple[int, ...]) -> None:
        self.tape = tape
        self._switch = 0

    def pick(self, candidates: Sequence[int]) -> int:
        if len(candidates) == 1:
            return candidates[0]
        switch = self._switch
        self._switch = switch + 1
        if switch < len(self.tape):
            return candidates[self.tape[switch] % len(candidates)]
        return candidates[0]


def check_schedule(schedule: Schedule) -> int | tuple[int, ...]:
    """The schedule as a seed or a tuple tape; anything else — a bool, a
    tape with a negative or non-integer entry — is refused."""
    if isinstance(schedule, bool):
        raise ScheduleError(f"a schedule is a seed or a tape, not {schedule!r}")
    if isinstance(schedule, int):
        return schedule
    tape = tuple(schedule)
    for index, entry in enumerate(tape):
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
            raise ScheduleError(
                f"tape entry {index} is {entry!r}; a tape holds non-negative integers"
            )
    return tape


def picker(schedule: Schedule) -> Picker:
    """The `Picker` a schedule names."""
    checked = check_schedule(schedule)
    if isinstance(checked, int):
        return Seeded(checked)
    return Taped(checked)
