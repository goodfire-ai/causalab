"""The replay deadline (``docs/cuda_graphs.md`` "Hung replays"): a bound on
device work that no backend watchdog tracks. Torch-free, so the tests run the
rule without a device.

**Why.** An eager collective is a ``Work`` object the process group's watchdog
times against ``CAUSALAB_COLLECTIVE_TIMEOUT``. A collective recorded into a
CUDA graph is not, so a rank whose peer skipped the replay or went to a
different collective waits forever, and the peer is alive or finished cleanly,
so the heartbeat has nobody lost to name. The replay site enqueues a
completion marker after each replay ([`Outstanding.enqueue`][]); the heartbeat
thread asks [`Outstanding.overdue`][] every tick and refuses an overdue replay
as [`ReplayOverdue`][].

**The rule.** Work is queued per stream in enqueue order, so an entry
completes no earlier than those before it. Each queue drains from the front
while its head reports completion, and only heads are timed. The oldest head
outstanding for the timeout or longer is overdue. Enqueues drain too, so a
queue holds only the replays still in flight.

A [`Completion`][] ``query`` must never wait on the device: the asking thread
is the one that ends a rank stuck there. A query that raises ``RuntimeError``
(a sticky device error) counts as complete; the main thread raises it at its
next synchronization.

**Captures.** Nothing is asked while this rank captures a graph
([`Outstanding.capturing`][]): querying a completed event from another thread
during a global-mode capture invalidates the capture. The capture site opens
the window after draining the device, so a replay stuck before the capture is
still refused. Queries run under the window's lock, so a capture never begins
during one.

**Release.** The heartbeat thread retires completed work but never drops the
last reference; the enqueuing thread releases it on its next enqueue or
capture window. Destroying a CUDA event waits on a driver lock that a
``cuGraphLaunch`` stuck behind a peer holds, and the destroying thread keeps
the GIL, which would stop the heartbeat too.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import threading
from typing import Hashable, Iterator, Protocol

from causalab.protocol.parallel import ParallelGeometry, format_geometry
from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "REPLAY",
    "Completion",
    "Outstanding",
    "Overdue",
    "ReplayOverdue",
    "describe_overdue",
]

#: What a CUDA graph replay is called in the refusal.
REPLAY = "a CUDA graph replay"


class Completion(Protocol):
    """Device work's completion, asked without waiting: ``True`` once done."""

    def query(self) -> bool: ...


@dataclasses.dataclass(frozen=True)
class Overdue:
    """``op`` was enqueued ``waited`` seconds ago and has not completed."""

    op: str
    waited: float


@dataclasses.dataclass(frozen=True)
class _Entry:
    op: str
    work: Completion
    since: float


def _done(work: Completion) -> bool:
    try:
        return work.query()
    except RuntimeError:
        # a device error: the main thread raises it at its next sync
        return True


class Outstanding:
    """The work this rank has enqueued that only the deadline bounds, one
    queue per stream (module docstring). Written by the main thread
    ([`enqueue`][]), read by the heartbeat thread ([`overdue`][]); one lock
    over both, and a query never waits, so neither holds the other up."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        self._queues: dict[Hashable, collections.deque[_Entry]] = {}
        #: open capture windows ([`capturing`][]); nothing is asked while > 0
        self._captures = 0
        #: drained work awaiting release by the enqueuing thread (module docstring)
        self._retired: list[_Entry] = []

    def enqueue(self, stream: Hashable, op: str, work: Completion, now: float) -> None:
        """``op`` was enqueued on ``stream`` at ``now``; ``work`` reports
        when it is done."""
        with self._lock:
            queue = self._queues.setdefault(stream, collections.deque())
            self._drain(queue)
            queue.append(_Entry(op, work, now))
            retired, self._retired = self._retired, []
        # released here, on the enqueuing thread and outside the lock
        del retired

    def overdue(self, now: float) -> Overdue | None:
        """The oldest queue head outstanding for the timeout or longer at
        ``now``; ``None`` when every head is younger or nothing is pending."""
        # plain values only past the lock: a local holding an entry would let
        # this (the heartbeat's) thread drop an event's last reference once
        # the enqueuing thread has released it (module docstring)
        oldest_op: str | None = None
        oldest_since = float("inf")
        with self._lock:
            if self._captures:
                return None
            for stream, queue in list(self._queues.items()):
                self._drain(queue)
                if not queue:
                    del self._queues[stream]
                    continue
                if queue[0].since < oldest_since:
                    oldest_op, oldest_since = queue[0].op, queue[0].since
        if oldest_op is None or now - oldest_since < self.timeout:
            return None
        return Overdue(oldest_op, now - oldest_since)

    @contextlib.contextmanager
    def capturing(self) -> Iterator[None]:
        """A window in which this rank captures a CUDA graph: completed work
        is drained on entry, by the capturing thread before its capture
        begins, and [`overdue`][] asks nothing until the window closes
        (module docstring). Windows nest."""
        with self._lock:
            for queue in self._queues.values():
                self._drain(queue)
            self._captures += 1
            retired, self._retired = self._retired, []
        del retired
        try:
            yield
        finally:
            with self._lock:
                self._captures -= 1

    def pending(self) -> int:
        """How many entries are queued, completed or not (for the tests)."""
        with self._lock:
            return sum(len(queue) for queue in self._queues.values())

    def _drain(self, queue: collections.deque[_Entry]) -> None:
        """Retire the completed heads; the caller holds the lock."""
        while queue and _done(queue[0].work):
            self._retired.append(queue.popleft())


class ReplayOverdue(ProtocolError):
    """Work only the deadline bounds did not complete within the collective
    timeout ([`Overdue`][]): ``P4`` at ``--parallel``, naming the work, how
    long it has waited, this rank, the geometry and the timeout."""

    def __init__(self, overdue: Overdue, message: str) -> None:
        self.overdue = overdue
        super().__init__("P4", message, path="--parallel")


def describe_overdue(
    overdue: Overdue,
    *,
    rank: int,
    world: int,
    geometry: ParallelGeometry,
    timeout: float,
    variable: str,
) -> ReplayOverdue:
    """The refusal for ``overdue`` on ``rank``; ``variable`` names the
    timeout's environment variable."""
    return ReplayOverdue(
        overdue,
        f"{overdue.op} enqueued {overdue.waited:.0f} s ago on rank {rank} of {world} "
        "has not completed; the backend's watchdog does not track collectives "
        "inside a CUDA graph, so a peer that left the lockstep would leave this "
        f"rank waiting forever (--parallel {format_geometry(geometry)}; "
        f"bounded by {variable}={timeout:g}; docs/cuda_graphs.md, "
        "docs/model_parallelism.md §3)",
    )
