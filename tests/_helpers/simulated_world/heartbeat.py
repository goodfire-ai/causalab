"""The heartbeat protocol, simulated (``docs/model_parallelism.md`` §3 "when a
rank dies", §10.2): ``world`` [`Heartbeat`][causalab.neural.shared.parallel.heartbeat.Heartbeat]
monitors over one in-process store and one clock, ticked every interval in
the order the schedule picks.

A rank *dies* by ceasing to tick (a kill, an OOM), *fails* by writing its
non-zero status (a refusal of its own), *finishes* by writing status zero,
*stops* (``stops``: a ``SIGSTOP``) by ceasing to tick without ever exiting
— the survivors cannot tell it from a death, and the driver reports it in
`HeartbeatSimulation.stopped` rather than among the exits —, is
*wedged* (``wedges``: its main thread stuck while its heartbeat thread
beats on) by ticking on while never finishing, its peers' collectives
failing the collective timeout after it (`HeartbeatSimulation.timeout`)
unless the scenario scripts them, or is *unreached* (``unreached``: dead before
it connected) by never ticking at all, so its peers never see a key of
its and name it ``unreached`` once the timeout has passed. **Every exit of
rank 0 takes the store with it** — its death, failure, finish and its
refusal of a lost peer alike, as the ``TCPStore`` it hosts goes with its
process — and so does its stop, black-holed: the fake fails at once where
the real tick stalls against a host that never replies and is counted
unreachable from its start once a grace has passed (``heartbeat.STALL_S``),
so the verdict lands at the same tick here and in production. A survivor
ticking after the host's exit
finds the store gone: a peer whose silence had reached the grace before
that is still named by its silence, the host is named as unreachable a
grace later otherwise (the watchdog module docstring, "when the store
goes with rank 0"). Rank 0 cannot be unreached here: without the store no
heartbeat runs anywhere, and the launcher's own refusal of an unreachable
store is the production path (``launcher.rendezvous``).
A rank whose *collective fails*
(``collective_failures``: the backend error a dead peer's closed socket
raises at once under gloo) asks its heartbeat at every tick from then on,
as the production ``hold`` does between the thread's ticks
([`Heartbeat.consult`][causalab.neural.shared.parallel.heartbeat.Heartbeat.consult]): a lost peer is refused by name, the
collective's own refusal (`OwnRefusal`, the rank finishing with
status 1 as the CLI does) is taken once every peer has beaten twice after
the failure or the bound has passed. Within one tick the ranks are ticked
one at a time in the order a `Picker` chooses among those
still running — the same seed-or-tape `Schedule` the baton
scheduler takes — and every choice is recorded in
`HeartbeatSimulation.picks`. The schedule decides from the tick
**before the first scripted event** (`HeartbeatSimulation.deciding_from`);
every tick before it is rank order and consumes nothing of the schedule.
Nothing a rank observes there outlives the next tick: a live peer's beat
advances every interval, and what a survivor holds entering the tick of a
death, a failure or a finish — the last beat it saw, the status it read —
is decided by the order of that tick alone. So a tape's entries land on the
ticks the verdicts turn on, and a heartbeat scenario is drawn and shrunk
like a collective one rather than spending its tape on the steady state.

The seams the production ``Heartbeat`` exposes are the ones driven here: the
clock is the driver's, the exit raises `Exit` so the driver sees the
status and the rank stops, the stderr write is captured per rank. The thread
and the real ``os._exit`` are the one seam this simulation does not run; the
spawned world is ``tests/neural/engines/pytorch_hooks/test_rank_watchdog_run.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from causalab.neural.shared.parallel.heartbeat import Heartbeat
from causalab.neural.shared.parallel.watchdog import (
    LOST_STATUS,
    Lost,
    RankLost,
    Settings,
)
from causalab.protocol.parallel import parse_geometry
from tests._helpers.simulated_world.errors import Unfinished
from tests._helpers.simulated_world.schedule import (
    Pick,
    Schedule,
    check_schedule,
    picker,
)

__all__ = [
    "Exit",
    "FakeStore",
    "HeartbeatSimulation",
    "OwnRefusal",
    "StoreGone",
    "UnsimulatedHost",
]


class UnsimulatedHost(ValueError):
    """Rank 0 was scripted as unreached: the store it hosts never exists, so
    no heartbeat runs and the simulation has nothing to drive."""


class StoreGone(RuntimeError):
    """The fake store's host has exited: every operation fails, as a
    ``TCPStore`` whose host exited raises a ``RuntimeError``."""


class FakeStore:
    """The three store verbs the heartbeat uses over a dict; ``gone`` makes
    every verb raise."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.gone = False

    def _check_alive(self) -> None:
        if self.gone:
            raise StoreGone("connection reset by peer")

    def set(self, key: str, value: str) -> None:
        self._check_alive()
        self.values[key] = value.encode()

    def get(self, key: str) -> bytes:
        self._check_alive()
        return self.values[key]

    def check(self, keys: Sequence[str]) -> bool:
        self._check_alive()
        return all(key in self.values for key in keys)


@dataclass
class Exit(Exception):
    """The seam standing in for ``os._exit``: raised so the driver sees the
    status and the rank stops."""

    status: int


@dataclass(frozen=True)
class OwnRefusal:
    """A rank raised its collective's own refusal at ``decided``: the
    heartbeat found every peer alive after the failure at ``failed_at`` (or
    the bound passed with nobody lost), so ``CollectiveFailed`` stood and
    the rank finished with status 1 (`HeartbeatSimulation.own_refusals`)."""

    failed_at: float
    decided: float


@dataclass
class HeartbeatSimulation:
    """``world`` monitors over one store and one clock, ticked every
    ``interval`` in the order ``schedule`` picks (module docstring).
    ``deaths`` maps a rank to the time it stops ticking, ``failures`` to
    ``(time, status)`` when it finishes with that status, ``finishes`` to the
    time it finishes cleanly, ``collective_failures`` to the time its
    collective fails on it, ``stops`` to the time it stops ticking without
    exiting, ``wedges`` to the time its main thread sticks while its
    heartbeat beats on; ``unreached`` names the ranks that never tick.
    ``timeout`` is the collective timeout (a hundred graces unless given:
    out of every scenario's way, as it is in production).

    Raises:
        UnsimulatedHost: rank 0 among ``unreached``.
    """

    world: int
    grace: float
    schedule: Schedule
    deaths: dict[int, float] = field(default_factory=dict)
    failures: dict[int, tuple[float, int]] = field(default_factory=dict)
    finishes: dict[int, float] = field(default_factory=dict)
    collective_failures: dict[int, float] = field(default_factory=dict)
    stops: dict[int, float] = field(default_factory=dict)
    wedges: dict[int, float] = field(default_factory=dict)
    unreached: frozenset[int] = frozenset()
    timeout: float | None = None
    geometry: str = "tp=2"
    #: the last run's choices of which running rank ticks next
    picks: list[Pick] = field(default_factory=list, init=False)
    #: the last run's ranks that raised their collective's own refusal — an
    #: exit of the rank's own in `run`, ``None`` there like a finish
    own_refusals: dict[int, OwnRefusal] = field(default_factory=dict, init=False)
    #: the last run's stopped ranks and the tick they stopped at: never among
    #: the exits, never unfinished — a process left behind for the launcher
    stopped: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.schedule = check_schedule(self.schedule)
        self.unreached = frozenset(self.unreached)
        if 0 in self.unreached:
            raise UnsimulatedHost(
                "rank 0 hosts the store: unreached, no heartbeat runs and the "
                "launcher's refusal of an unreachable store is the path (§11)"
            )

    @property
    def settings(self) -> Settings:
        timeout = self.grace * 100 if self.timeout is None else self.timeout
        return Settings(timeout=timeout, grace=self.grace)

    def survivors_collective_failures(self) -> dict[int, float]:
        """When each rank's collective fails: as scripted, and for every
        rank not wedged the collective timeout after the earliest wedge —
        a wedged peer never arrives, and the timeout is what ends the
        wait (module docstring)."""
        failures = dict(self.collective_failures)
        if self.wedges:
            earliest = min(self.wedges.values()) + self.settings.timeout
            for rank in range(self.world):
                if rank not in self.wedges and rank not in self.unreached:
                    failures.setdefault(rank, earliest)
        return failures

    @property
    def interval(self) -> float:
        return self.settings.interval

    @property
    def deciding_from(self) -> float | None:
        """The clock time from which the schedule decides the tick order:
        one interval before the first scripted event (module docstring);
        ``None`` when nothing is scripted, every tick then rank order."""
        events = [
            *self.deaths.values(),
            *(when for when, _ in self.failures.values()),
            *self.finishes.values(),
            *self.survivors_collective_failures().values(),
            *self.stops.values(),
            *self.wedges.values(),
        ]
        if self.unreached:
            # the unreached ranks' peers judge them at the timeout: the tick
            # order matters from one interval before it
            events.append(self.settings.timeout)
        if not events:
            return None
        return min(events) - self.interval

    def survivors(self, gone: int) -> list[int]:
        """Every rank but ``gone`` and the unreached."""
        return [
            rank
            for rank in range(self.world)
            if rank != gone and rank not in self.unreached
        ]

    def run(self, until: float) -> dict[int, tuple[float, RankLost | None]]:
        """Every rank's exit: ``(time, refusal)`` — the refusal a lost peer
        drew, ``None`` for a death or a finish of its own (a clean finish, a
        failure, or the collective's own refusal, the last recorded in
        `own_refusals`). A stopped rank (`stopped`) and an
        unreached one never exit and are not among them; any other rank still
        running when the clock reaches ``until`` is `Unfinished`."""
        choose = picker(self.schedule)
        self.picks = []
        self.own_refusals = {}
        self.stopped = {}
        store = FakeStore()
        config = self.settings
        collective_failures = self.survivors_collective_failures()
        stderr: dict[int, list[str]] = {rank: [] for rank in range(self.world)}

        def exit_(status: int) -> None:
            raise Exit(status)

        clock = [0.0]  # the driver's clock, read at construction
        monitors = {
            rank: Heartbeat(
                store,
                rank=rank,
                world=self.world,
                settings=config,
                geometry=parse_geometry(self.geometry),
                clock=lambda: clock[0],
                exit=exit_,
                write=stderr[rank].append,
            )
            for rank in range(self.world)
        }
        exits: dict[int, tuple[float, RankLost | None]] = {}

        def refused(rank: int, exited: Exit) -> RankLost:
            # the protocol's exit contract, checked where it fires
            assert exited.status == LOST_STATUS, exited.status
            text = "".join(stderr[rank])
            assert text.startswith("refused: [P4] at --parallel "), text
            refusal = monitors[rank].refusal
            assert refusal is not None
            return refusal

        deciding_from = self.deciding_from
        ticking = [rank for rank in range(self.world) if rank not in self.unreached]
        now = 0.0
        while now <= until and len(exits) + len(self.stopped) < len(ticking):
            clock[0] = now
            deciding = deciding_from is not None and now >= deciding_from - 1e-9
            remaining = [
                rank
                for rank in ticking
                if rank not in exits and rank not in self.stopped
            ]
            while remaining:
                rank = choose.pick(remaining) if deciding else remaining[0]
                self.picks.append(Pick(tuple(remaining), rank))
                remaining.remove(rank)
                monitor = monitors[rank]
                if rank in self.deaths and now >= self.deaths[rank]:
                    exits[rank] = (now, None)
                elif rank in self.stops and now >= self.stops[rank]:
                    # every thread stops with the process: no beat, no exit
                    self.stopped[rank] = now
                elif rank in self.failures and now >= self.failures[rank][0]:
                    monitor.finish(self.failures[rank][1])
                    exits[rank] = (now, None)
                elif rank in self.finishes and now >= self.finishes[rank]:
                    try:
                        lost = monitor.tick(now)
                        if lost is not None:
                            monitor.refuse(lost)
                        if not monitor.completion_ready(now):
                            continue
                        monitor.finish(0)
                        exits[rank] = (now, None)
                    except Exit as exited:
                        exits[rank] = (now, refused(rank, exited))
                else:
                    try:
                        lost = monitor.tick(now)
                        if lost is not None:
                            monitor.refuse(lost)
                        failed_at = collective_failures.get(rank)
                        if failed_at is not None and now >= failed_at:
                            # the main thread, held in the failed collective,
                            # asks after each tick as ``hold`` does
                            blame = monitor.consult(failed_at, now)
                            if isinstance(blame, Lost):
                                monitor.refuse(blame)
                            elif blame == "collective":
                                # ``CollectiveFailed`` propagates to the
                                # CLI, whose ``leave`` finishes with status 1
                                monitor.finish(1)
                                self.own_refusals[rank] = OwnRefusal(failed_at, now)
                                exits[rank] = (now, None)
                    except Exit as exited:
                        # the refusal's exit — rank 0's takes the store too,
                        # as every exit of the host does (module docstring)
                        exits[rank] = (now, refused(rank, exited))
                    if rank not in exits:
                        continue
                if rank == 0:
                    store.gone = True
            now = round(now + config.interval, 9)
        running = tuple(
            rank for rank in ticking if rank not in exits and rank not in self.stopped
        )
        if running:
            raise Unfinished(running, until)
        return exits
