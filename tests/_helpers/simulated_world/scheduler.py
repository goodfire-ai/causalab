"""The baton: one runnable rank at a time, chosen by the schedule.

Ranks are threads, but the scheduler hands a single baton to exactly one of
them; a rank yields it only inside a collective (``yield_baton``) or when its
program ends (``finish``). Between collectives a rank's code is therefore the
only code running, and the only nondeterminism left is which parked rank
resumes next — the `Picker` of the world's schedule
decides, and every choice is recorded as a `Pick`.

Every wait carries a timeout so a bug in the simulator itself can never hang
the test process: the main thread's deadline reports a `Hang`.
"""

from __future__ import annotations

import threading
from typing import Callable

from tests._helpers.simulated_world.errors import Hang, Refusal, Waiting
from tests._helpers.simulated_world.schedule import Pick, Schedule, picker


class Abort(BaseException):
    """Unwinds a rank thread after a refusal; not an ``Exception`` so a
    program's ``except Exception`` cannot swallow it."""


class Baton:
    def __init__(
        self,
        world: int,
        *,
        schedule: Schedule,
        slow: frozenset[int],
        timeout: float,
    ) -> None:
        self.world = world
        self._picker = picker(schedule)
        self._slow = slow
        self._timeout = timeout
        self.cv = threading.Condition()
        self.holder: int | None = None
        self.runnable: set[int] = set(range(world))
        self.done: set[int] = set()
        self.parked: dict[int, Waiting] = {}
        self.failure: Refusal | None = None
        self.aborted = False
        #: every hand-off, in order: the runnable ranks and the one chosen
        self.picks: list[Pick] = []

    # -- under ``cv`` -----------------------------------------------------------

    def _pick(self) -> int:
        """The schedule's choice among the runnable ranks; a slow rank runs
        only when no other rank can, so it arrives last at every rendezvous."""
        runnable = sorted(self.runnable)
        eager = [rank for rank in runnable if rank not in self._slow]
        rank = self._picker.pick(eager or runnable)
        self.picks.append(Pick(tuple(runnable), rank))
        return rank

    def _hand_off(self) -> None:
        """Give the baton to the next runnable rank, or detect a hang."""
        if self.runnable:
            self.holder = self._pick()
        elif len(self.done) == self.world:
            self.holder = None
        else:
            self.fail(Hang(tuple(self.parked[r] for r in sorted(self.parked))))
        self.cv.notify_all()

    def fail(self, refusal: Refusal) -> None:
        """Record the first refusal and wake every thread so it can unwind."""
        if self.failure is None:
            self.failure = refusal
        self.aborted = True
        self.cv.notify_all()

    def wake(self, ranks: tuple[int, ...]) -> None:
        """The members of a completed rendezvous become runnable again."""
        for rank in ranks:
            self.parked.pop(rank, None)
            if rank not in self.done:
                self.runnable.add(rank)

    def _wait_for(self, rank: int) -> None:
        while self.holder != rank and not self.aborted:
            self.cv.wait(self._timeout)
        if self.aborted:
            raise Abort()
        self.runnable.discard(rank)

    # -- from rank threads ------------------------------------------------------

    def start(self, rank: int) -> None:
        """Block until the scheduler first hands ``rank`` the baton."""
        with self.cv:
            self._wait_for(rank)

    def yield_baton(self, rank: int, parked: Waiting | None) -> None:
        """Give up the baton — parked at a rendezvous when ``parked`` is given,
        runnable otherwise — and block until it comes back."""
        with self.cv:
            if parked is None:
                self.runnable.add(rank)
            else:
                self.parked[rank] = parked
            self._hand_off()
            self._wait_for(rank)

    def finish(self, rank: int) -> None:
        with self.cv:
            self.done.add(rank)
            self._hand_off()

    # -- from the main thread ---------------------------------------------------

    def run_all(self, bodies: dict[int, Callable[[], None]]) -> None:
        """Start one daemon thread per rank, hand out the first baton and wait
        for every rank to finish or the first refusal, under the deadline."""
        threads = [
            threading.Thread(target=body, name=f"rank-{rank}", daemon=True)
            for rank, body in bodies.items()
        ]
        for thread in threads:
            thread.start()
        with self.cv:
            self._hand_off()
            while self.failure is None and len(self.done) < self.world:
                if not self.cv.wait(self._timeout):
                    parked = tuple(self.parked[r] for r in sorted(self.parked))
                    self.fail(Hang(parked, timed_out=True))
        for thread in threads:
            thread.join(self._timeout)
        if self.failure is not None:
            raise self.failure
