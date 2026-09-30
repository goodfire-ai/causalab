"""The rank watchdog's heartbeat (``docs/model_parallelism.md`` §3 "when a
rank dies"): one daemon thread per launched rank over the rendezvous store,
beating and watching, that turns a lost peer into the named refusal of
[`.watchdog`][causalab.neural.shared.parallel.watchdog] and ends the process.

**The store.** ``torch.distributed`` rendezvous through a ``TCPStore`` rank
0 hosts and every rank connects to (``launcher.join_group`` builds it, as
``env://`` would, but without the store's wait for workers: the heartbeat
is the watch over who arrives); it outlives the process group and needs
no collective, so a rank blocked inside an NCCL kernel can still be watched
from a thread, and a rank that died stops writing to it. A peer never seen
is bounded by the collective timeout, not the grace (``watchdog.Liveness``
``arrival``): before its first beat it may still be starting. The heartbeat takes its own **client**
connection with the grace as its timeout — the rendezvous store keeps the
collective timeout, so a slow rank's arrival at ``new_group`` or at NCCL's
communicator exchange is not cut short by the shorter watch — and uses three
verbs of it ([`Store`][]): ``set``, ``get``, ``check``. A dict behind the
same three verbs is the deterministic simulation's store.

**The loop.** Every ``settings.interval`` seconds [`Heartbeat.tick`][]
writes ``beat:<n>`` under this rank's key, reads every peer's key into the
[`Liveness`][] and takes its verdict; a store operation that
fails is the host's death or the network's, recorded as such — and so is
one that does not return within a grace (a stopped host acknowledges and
never replies, and the client's own timeout does not fire; ``_round``). A verdict is
[`refuse`][causalab.neural.shared.parallel.heartbeat.Heartbeat.refuse]: the refusal printed to stderr as the CLI prints every other
(``refused: [P4] …``), then ``os._exit(LOST_STATUS)`` — the one call that
ends a process whose main thread is inside a collective that will never
return; ``sys.exit`` from a thread would not. `finish` writes
``done:<status>`` so the peers read a clean finish as clean and a refusal
of this rank's own as its status, and stops the thread.

**A collective that fails** on the main thread (``TorchCollective``, the
watchdog module docstring) is held in [`Heartbeat.hold`][] until the
thread's ticks can say whose it is ([`consult`][causalab.neural.shared.parallel.heartbeat.Heartbeat.consult], [`Blame`][]):
a lost peer is refused by name — once, whichever thread sees it first —
and the process ends; every peer proven alive after the failure, or the
bound passed with nobody lost, returns to the caller, whose
``CollectiveFailed`` then stands. [`running`][] is the one heartbeat
this process has started and not finished, registered by `start`
for the collective to find.

**Replays.** Collectives inside a replayed CUDA graph are invisible to the
process group's watchdog. Each replay registers its completion through
[`enqueued`][causalab.neural.shared.parallel.heartbeat.Heartbeat.enqueued]; a
round with no lost peer asks the replay deadline
([`.deadline`][causalab.neural.shared.parallel.deadline]) for overdue work and
refuses it as
[`ReplayOverdue`][causalab.neural.shared.parallel.deadline.ReplayOverdue] with
the same exit. The check runs on the heartbeat thread, not the store's worker,
so a stalled store cannot hide it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Hashable, Protocol, Sequence

from causalab.neural.shared.parallel import watchdog
from causalab.neural.shared.parallel.deadline import (
    Completion,
    Outstanding,
    Overdue,
    ReplayOverdue,
    describe_overdue,
)
from causalab.neural.shared.parallel.watchdog import (
    CompletionTimeout,
    MalformedSignal,
    LOST_STATUS,
    Beat,
    Blame,
    Done,
    Liveness,
    Lost,
    RankLost,
    Settings,
    Signal,
    StoreHost,
    decode,
    encode,
)
from causalab.protocol.parallel import ParallelGeometry

__all__ = [
    "KEY_PREFIX",
    "STALL_S",
    "Heartbeat",
    "Store",
    "done_key_for",
    "key_for",
    "running",
]

#: Every rank's liveness key is ``KEY_PREFIX/<rank>``.
KEY_PREFIX = "causalab/liveness"

#: The fraction of a grace a tick's store round trips may take before the
#: store counts as unreachable since the tick began (``Heartbeat._round``):
#: a whole grace, so the verdict on a stopped host lands a grace after the
#: first stalled tick — the bound a dead host has.
STALL_S = 1.0


def done_key_for(rank: int) -> str:
    """A terminal status cannot be overwritten by an in-flight beat."""
    return f"{KEY_PREFIX}/done/{rank}"


def key_for(rank: int) -> str:
    return f"{KEY_PREFIX}/{rank}"


class Store(Protocol):
    """The three verbs of ``torch.distributed.TCPStore`` the heartbeat uses."""

    def set(self, key: str, value: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def check(self, keys: Sequence[str]) -> bool: ...


def _write_stderr(text: str) -> None:
    sys.stderr.write(text)
    sys.stderr.flush()


#: The heartbeat this process has started and not finished (module
#: docstring) — one per launched rank, none at world 1 or in a process that
#: never joined a group.
_running: Heartbeat | None = None


def running() -> Heartbeat | None:
    """The process's started heartbeat, for a failing collective to consult."""
    return _running


class Heartbeat:
    """This rank's beat and watch over ``store`` (module docstring).

    ``clock``, ``exit`` and ``write`` are the seams: the wall clock, the
    process exit and the stderr write, replaced by the simulation with a
    driven clock, a raised exit and a captured line. [`refusal`][] is the
    [`RankLost`][] or [`ReplayOverdue`][] [`refuse`][] drew, for the reader;
    [`outstanding`][] the work only the replay deadline bounds (module
    docstring).
    """

    def __init__(
        self,
        store: Store,
        *,
        rank: int,
        world: int,
        settings: Settings,
        geometry: ParallelGeometry,
        clock: Callable[[], float] = time.monotonic,
        exit: Callable[[int], None] = os._exit,
        write: Callable[[str], None] = _write_stderr,
        store_host: StoreHost = "rank",
    ) -> None:
        self._store = store
        self.store_host: StoreHost = store_host
        self.rank = rank
        self.world = world
        self.settings = settings
        self.geometry = geometry
        self._clock = clock
        self._exit = exit
        self._write = write
        self._count = 0
        # a peer never seen may still be starting: its bound is the timeout
        self._liveness = Liveness(
            rank=rank,
            world=world,
            grace=settings.grace,
            now=clock(),
            arrival=settings.timeout,
        )
        #: the liveness is written by the thread's ticks and read by a held
        #: main thread: every tick notifies, ``hold`` waits
        self._changed = threading.Condition()
        #: one refusal per process, whichever thread sees the loss first
        self._refusing = threading.Lock()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self.refusal: RankLost | ReplayOverdue | None = None
        self._completion_started: float | None = None
        #: device work only the deadline bounds: CUDA graph replays
        self.outstanding = Outstanding(settings.timeout)

    # -- one round --------------------------------------------------------------

    def tick(self, now: float) -> Lost | None:
        """Beat, read every peer, and take the verdict at ``now``."""
        self._count += 1
        # the store round trips run outside the lock: a read stuck in a
        # half-open socket must not hold up ``finish`` or a held main thread
        observed: list[tuple[int, Signal | None]] = []
        reachable = True
        try:
            self._store.set(key_for(self.rank), encode(Beat(self._count)))
            for peer in range(self.world):
                if peer == self.rank:
                    continue
                terminal = done_key_for(peer)
                key = terminal if self._store.check([terminal]) else key_for(peer)
                signal = (
                    self._signal(self._store.get(key))
                    if self._store.check([key])
                    else None
                )
                observed.append((peer, signal))
        except (RuntimeError, OSError):
            # ``DistStoreError`` and the store's socket errors are both
            # RuntimeErrors; a refused connection is an OSError: the host
            # is gone, or the network is
            reachable = False
        with self._changed:
            for peer, signal in observed:
                self._liveness.observe(peer, signal, now)
            if not reachable:
                self._liveness.unreachable(now)
            verdict = self._liveness.verdict(now)
            self._changed.notify_all()
        return verdict

    @staticmethod
    def _signal(raw: bytes) -> Signal | None:
        """A peer's key decoded; a byte that is neither a beat nor a done
        ([`MalformedSignal`][]) reads as no signal this tick —
        the peer advances nothing, and is named silent past the grace —
        rather than ending the watch, which must outlive a bad byte on
        the store."""
        try:
            return decode(raw)
        except MalformedSignal:
            return None

    def enqueued(self, stream: Hashable, op: str, work: Completion) -> None:
        """``op`` was just enqueued on ``stream``, ``work`` reporting its
        completion: bounded by the collective timeout from now (module
        docstring)."""
        self.outstanding.enqueue(stream, op, work, self._clock())

    def refuse(self, lost: Lost | Overdue) -> None:
        """Print the refusal for ``lost`` (a lost peer or an overdue replay)
        and end the process, once: a second thread arriving with the same
        loss finds the refusal drawn and leaves (in the rank the first never
        returns from the exit). The exit is unconditional: the decision is
        taken before the words are, and a failure rendering or writing them
        must not leave a process alive that has decided not to be."""
        with self._refusing:
            if self.refusal is not None:
                return
            try:
                if isinstance(lost, Overdue):
                    self.refusal = describe_overdue(
                        lost,
                        rank=self.rank,
                        world=self.world,
                        geometry=self.geometry,
                        timeout=self.settings.timeout,
                        variable=watchdog.COLLECTIVE_TIMEOUT_VARIABLE,
                    )
                else:
                    self.refusal = watchdog.describe(
                        lost,
                        rank=self.rank,
                        world=self.world,
                        geometry=self.geometry,
                        settings=self.settings,
                        where=watchdog.current(),
                        store_host=self.store_host,
                    )
                self._write(f"refused: {self.refusal}\n")
            finally:
                self._exit(LOST_STATUS)

    # -- a collective that failed -----------------------------------------------

    def consult(self, failed_at: float, now: float) -> Blame | None:
        """What the watch says, at ``now``, of a collective that failed at
        ``failed_at`` ([`Blame`][]): the peer whose loss
        explains it; ``"collective"`` when every peer is proven alive after
        the failure ([`every_peer_alive_since`][causalab.neural.shared.parallel.watchdog.Liveness.every_peer_alive_since]) or
        the bound has passed with nobody lost **while the watch could see**
        — a store gone since the failure has hidden the peers, and the
        verdict it leads to (the host unreachable, a grace after it went)
        is what ends the hold instead, so a dead peer behind a dead store
        is never mistaken for a hang; ``None`` while undecided."""
        with self._changed:
            lost = self._liveness.verdict(now)
            if lost is not None:
                return lost
            if self._liveness.every_peer_alive_since(failed_at):
                return "collective"
            watching = self._liveness.reachable or self._liveness.settled
        if watching and now - failed_at >= self.settings.bound:
            return "collective"
        return None

    def hold(self) -> None:
        """A collective failed on the calling thread just now: wait on the
        watch until [`consult`][] decides. A lost peer is [`refuse`][]d
        — the process ends, here or on the thread — and the collective's own
        failure returns to the caller to raise. A finished heartbeat, or one
        that has refused already, holds nobody."""
        failed_at = self._clock()
        blame: Blame | None = None
        with self._changed:
            while not self._stopped.is_set() and self.refusal is None:
                blame = self.consult(failed_at, self._clock())
                if blame is not None:
                    break
                self._changed.wait(self.settings.interval)
        if isinstance(blame, Lost):
            self.refuse(blame)

    # -- the thread -------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stopped.wait(self.settings.interval):
            if self._round():
                return

    def _round(self) -> bool:
        """One tick of the thread: ``True`` once it has refused. A verdict
        taken by a tick already in flight when [`finish`][] was called is
        dropped — the run has returned its status, and the watch has no
        say after it. The tick itself must not take the watch down: a
        failure inside it that is not the store's (a bug in the rules, an
        unforeseen store error) is written once and read as the host
        unreachable for this tick, so the grace still bounds the silence
        instead of a dead daemon thread leaving the rank to the timeout.

        Nor may it hang the watch: the tick's store round trips run on a
        worker joined for a grace ([`STALL_S`][] of it), and a tick that
        has not returned by then reads as the store unreachable since the
        tick began. The client's own timeout may not fire when a host
        acknowledges a request but never replies, so the watch bounds the
        round trip independently. The
        stalled worker is a daemon: the process ends with the refusal, or,
        should the store answer after all, the worker's late observations
        land at the tick's own time."""
        now = self._clock()
        outcome: list[Lost | None] = []

        def attempt() -> None:
            try:
                outcome.append(self.tick(now))
            except Exception as err:  # noqa: BLE001 — the watch outlives its own failure
                self._write(
                    f"heartbeat: tick failed ({type(err).__name__}: {err}); watching on\n"
                )
                with self._changed:
                    self._liveness.unreachable(now)
                    outcome.append(self._liveness.verdict(now))
                    self._changed.notify_all()

        worker = threading.Thread(
            target=attempt, name=f"causalab-heartbeat-{self.rank}-tick", daemon=True
        )
        worker.start()
        worker.join(self.settings.grace * STALL_S)
        if worker.is_alive():
            # the store answers nothing: unreachable since the tick began
            with self._changed:
                self._liveness.unreachable(now)
                lost = self._liveness.verdict(self._clock())
                self._changed.notify_all()
        else:
            lost = outcome[0] if outcome else None
        if self._stopped.is_set():
            return False
        verdict: Lost | Overdue | None = lost
        if verdict is None:
            verdict = self._overdue()
        if verdict is None:
            return False
        self.refuse(verdict)
        return True

    def _overdue(self) -> Overdue | None:
        """The replay deadline's verdict this round; a failure of the check
        itself is written and read as nothing overdue, so the watch outlives
        it as it outlives a failed tick."""
        try:
            return self.outstanding.overdue(self._clock())
        except Exception as err:  # noqa: BLE001 — the watch outlives its own failure
            self._write(
                f"heartbeat: replay check failed ({type(err).__name__}: {err}); "
                "watching on\n"
            )
            return None

    def start(self) -> None:
        """Take one bounded store round, then beat on a daemon thread; register
        as the process's heartbeat ([`running`][])."""
        global _running
        if self._round():
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"causalab-heartbeat-{self.rank}", daemon=True
        )
        _running = self
        self._thread.start()

    def completion_ready(self, now: float) -> bool:
        """Keep rank 0's store alive until every peer has completed successfully.

        The poll is shared with deterministic simulation. Healthy peers may
        finish more than a grace apart; a live peer that never finishes is
        bounded by the collective timeout, while the heartbeat still detects
        dead peers within the grace.
        """
        if self.rank != 0:
            return True
        with self._changed:
            if self._completion_started is None:
                self._completion_started = now
            if self._liveness.all_finished:
                return True
            if now - self._completion_started >= self.settings.timeout:
                raise CompletionTimeout(self.settings.timeout)
            return False

    def complete(self) -> None:
        """Wait for clean peer completion while the heartbeat keeps running."""
        with self._changed:
            while not self._stopped.is_set() and self.refusal is None:
                if self.completion_ready(self._clock()):
                    return
                self._changed.wait(self.settings.interval)

    def finish(self, status: int) -> None:
        """Stop watching and tell the peers this rank is done with ``status``."""
        global _running
        if _running is self:
            _running = None
        self._stopped.set()
        with self._changed:
            self._changed.notify_all()  # a held main thread returns
        if self._thread is not None:
            # a tick blocked in the store is bounded by the store's timeout
            # (the grace); the process is exiting either way
            self._thread.join(timeout=self.settings.grace)
            self._thread = None

        def publish() -> None:
            try:
                self._store.set(done_key_for(self.rank), encode(Done(status)))
                self._store.set(key_for(self.rank), encode(Done(status)))
            except (RuntimeError, OSError):
                pass  # the store is gone; shutdown must still complete

        # Even set() can block on a half-open store socket. Failure shutdown
        # must not wait for a backend timeout or a peer's completion.
        writer = threading.Thread(
            target=publish, name=f"causalab-heartbeat-{self.rank}-done", daemon=True
        )
        writer.start()
        writer.join(timeout=self.settings.grace)
