"""The heartbeat's thread (``docs/model_parallelism.md`` §3 "when a rank
dies"): the one seam the deterministic simulation of ``test_watchdog.py``
does not run, run here over the simulation's own fake store with the same
clock, exit and stderr seams.

``unit``: a fresh heartbeat has drawn no refusal; [`Heartbeat.refuse`][causalab.neural.shared.parallel.heartbeat.Heartbeat.refuse]
names the collective the process is inside ([`watchdog.inside`][causalab.neural.shared.parallel.watchdog.inside]), writes
the ``refused: [P4] at --parallel …`` line and exits through the seam with
[`LOST_STATUS`][causalab.neural.shared.parallel.watchdog.LOST_STATUS]; `start` beats once synchronously and then from a
daemon thread named for the rank, which ticks every interval, refuses on the
first lost verdict and returns; `finish` stops the thread, waits no
longer than the grace for a tick stuck in the store, and writes
``done:<status>`` for the peers. [`consult`][causalab.neural.shared.parallel.heartbeat.Heartbeat.consult] is the rule a collective
failure asks — the peer whose loss explains it, ``"collective"`` once every
peer has beaten twice after the failure or the bound has passed, ``None``
while undecided — and `hold` the main thread's wait on it, which ends
in the thread's refusal (once, however many threads see the loss) or
returns for the collective's own. [`running`][causalab.neural.shared.parallel.heartbeat.running] is the process's one
started heartbeat, what the collective consults. The clock is driven from
the test, so the verdict is the grace rule and not the wall clock; only the
thread's own interval is real time.
"""

from __future__ import annotations

import threading
import time
from typing import Sequence

import pytest

from causalab.neural.shared.parallel import watchdog
from causalab.neural.shared.parallel.heartbeat import Heartbeat, key_for, running
from causalab.neural.shared.parallel.watchdog import (
    LOST_STATUS,
    Beat,
    Done,
    Lost,
    Settings,
    decode,
    encode,
)
from causalab.protocol.parallel import parse_geometry
from tests._helpers.simulated_world.heartbeat import FakeStore

pytestmark = pytest.mark.unit

#: A tenth of a second's grace: the thread ticks every 10 ms.
QUICK = Settings(timeout=10.0, grace=0.1)
GEOMETRY = parse_geometry("tp=2")
#: How long a test waits on the thread before calling it hung.
PATIENCE = 5.0


class _Seams:
    """The exit and stderr seams, recorded."""

    def __init__(self) -> None:
        self.statuses: list[int] = []
        self.lines: list[str] = []
        self.exited = threading.Event()

    def exit(self, status: int) -> None:
        self.statuses.append(status)
        self.exited.set()

    def write(self, text: str) -> None:
        self.lines.append(text)


def _heartbeat(store: FakeStore, seams: _Seams, clock: list[float]) -> Heartbeat:
    return Heartbeat(
        store,
        rank=0,
        world=2,
        settings=QUICK,
        geometry=GEOMETRY,
        clock=lambda: clock[0],
        exit=seams.exit,
        write=seams.write,
    )


def _thread_named(name: str) -> threading.Thread | None:
    found = [thread for thread in threading.enumerate() if thread.name == name]
    return found[0] if found else None


class TestRefuse:
    def test_a_fresh_heartbeat_has_drawn_no_refusal(self) -> None:
        heartbeat = _heartbeat(FakeStore(), _Seams(), [0.0])
        assert heartbeat.refusal is None

    def test_the_refusal_names_the_collective_the_process_is_inside(self) -> None:
        seams = _Seams()
        heartbeat = _heartbeat(FakeStore(), seams, [0.0])
        lost = Lost(1, "silent", silence=2.0)
        with watchdog.inside("tensor", "all_gather"):
            heartbeat.refuse(lost)
        assert seams.statuses == [LOST_STATUS]
        assert heartbeat.refusal is not None and heartbeat.refusal.lost == lost
        text = str(heartbeat.refusal)
        assert "rank 0 is still running, waiting in all_gather on axis 'tensor'" in text
        assert seams.lines == [f"refused: {text}\n"]
        assert text.startswith("[P4] at --parallel ")

    def test_outside_every_collective_the_refusal_says_running_and_no_more(
        self,
    ) -> None:
        assert watchdog.current() is None
        seams = _Seams()
        heartbeat = _heartbeat(FakeStore(), seams, [0.0])
        heartbeat.refuse(Lost(1, "exited", status=3))
        text = str(heartbeat.refusal)
        assert "rank 1 of 2 exited with status 3 and rank 0 is still running:" in text
        assert "waiting in" not in text
        assert text.endswith("docs/model_parallelism.md §3)")


class TestTheWatchOutlivesItsOwnFailures:
    """The daemon thread is the last thing standing between a rank and the
    backend timeout, so nothing on it may end the watch quietly."""

    def test_a_malformed_signal_on_a_peers_key_reads_as_no_signal(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))  # seen once: the grace's from here
        heartbeat.tick(0.0)
        store.values[key_for(1)] = b"garbage"
        assert heartbeat.tick(0.0) is None, "a bad byte is not a verdict"
        clock[0] = 1.0  # past the grace with nothing readable from the peer
        lost = heartbeat.tick(1.0)
        assert lost == Lost(1, "silent", silence=1.0)
        # a beat after the garbage is read as the beat it is
        store.set(key_for(1), encode(Beat(1)))
        assert heartbeat.tick(1.0) is None

    def test_refuse_exits_even_when_the_words_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from causalab.neural.shared.parallel import watchdog

        def broken(*args, **kwargs):
            raise RuntimeError("no words")

        monkeypatch.setattr(watchdog, "describe", broken)
        seams = _Seams()
        heartbeat = _heartbeat(FakeStore(), seams, [0.0])
        with pytest.raises(RuntimeError, match="no words"):
            heartbeat.refuse(Lost(1, "silent", silence=1.0))
        assert seams.statuses == [LOST_STATUS], "the exit is unconditional"

    def test_a_verdict_taken_after_finish_refuses_nothing(self) -> None:
        """``finish`` sets the stop and joins, but a tick in flight completes:
        its verdict must not exit a rank whose run already returned 0."""
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.tick(0.0)  # the peer was seen at 0.0 …
        clock[0] = 1.0  # … and has been silent past the grace
        heartbeat._stopped.set()  # pyright: ignore[reportPrivateUsage]
        assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
        assert seams.statuses == [] and heartbeat.refusal is None
        heartbeat._stopped.clear()  # pyright: ignore[reportPrivateUsage]
        assert heartbeat._round() is True  # pyright: ignore[reportPrivateUsage]
        assert seams.statuses == [LOST_STATUS]

    def test_a_tick_that_does_not_return_within_the_grace_is_the_host_unreachable(
        self,
    ) -> None:
        """The store acknowledges and never replies (a stopped host): the
        client's own timeout does not fire, so the round joins its tick for
        a grace and then counts the store unreachable since the tick began —
        the verdict lands at once, a grace after the tick, and the process
        refuses the host. The stalled worker is released afterwards."""
        store, seams, clock = FakeStore(), _Seams(), [0.5]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.tick(0.5)  # the peer seen at 0.5: nothing is lost by silence
        release = threading.Event()

        class _BlackHole(FakeStore):
            def check(self, keys: Sequence[str]) -> bool:
                release.wait()
                return True

        heartbeat._store = _BlackHole()  # pyright: ignore[reportPrivateUsage]
        # the stalled tick begins at 0.5; the driven clock stands still
        started = time.monotonic()
        try:
            assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
            waited = time.monotonic() - started
            assert QUICK.grace <= waited < PATIENCE, "joined for a grace, no longer"
            liveness = heartbeat._liveness  # pyright: ignore[reportPrivateUsage]
            # unreachable since 0.5: at 0.59 the grace has not passed; at 0.7 it has
            assert liveness.verdict(0.59) is None
            assert liveness.verdict(0.7) == Lost(0, "unreachable")
            clock[0] = 0.7
            assert heartbeat._round() is True  # pyright: ignore[reportPrivateUsage]
            assert heartbeat.refusal is not None
            assert heartbeat.refusal.lost == Lost(0, "unreachable")
            assert seams.statuses == [LOST_STATUS]
        finally:
            release.set()

    def test_a_tick_that_raises_is_written_and_the_watch_goes_on(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)

        class _Odd(FakeStore):
            def check(self, keys):
                raise ValueError("an unforeseen store error")

        heartbeat._store = _Odd()  # pyright: ignore[reportPrivateUsage]
        assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
        assert seams.lines == [
            "heartbeat: tick failed (ValueError: an unforeseen store error); watching on\n"
        ]
        clock[0] = 1.0  # the host has been unreachable for a grace
        assert heartbeat._round() is True  # pyright: ignore[reportPrivateUsage]
        assert heartbeat.refusal is not None
        assert heartbeat.refusal.lost == Lost(0, "unreachable")


class TestTheThread:
    def test_it_beats_every_interval_and_refuses_the_lost_peer_once(self) -> None:
        """The peer beat once and never again: once the driven clock has
        passed the grace the thread's next tick names it silent, exits
        through the seam once and returns — the thread ends. (A peer never
        seen at all is the timeout's, ``unreached``: ``test_watchdog.py``.)"""
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.start()
        try:
            # the first beat is synchronous, before the thread exists
            assert decode(store.get(key_for(0))) == Beat(1)
            thread = _thread_named("causalab-heartbeat-0")
            assert thread is not None, "no thread named for the rank"
            assert thread.daemon, "the heartbeat must not keep the process alive"
            clock[0] = 1.0  # past the grace: the peer has been silent since 0.0
            assert seams.exited.wait(PATIENCE), "the thread never refused"
            assert seams.statuses == [LOST_STATUS]
            assert heartbeat.refusal is not None
            assert heartbeat.refusal.lost == Lost(1, "silent", silence=1.0)
            thread.join(PATIENCE)
            assert not thread.is_alive(), "the loop went on after refusing"
            # the refusing tick beat before it looked: at least one more beat
            assert decode(store.get(key_for(0))).count >= 2
            assert seams.lines and seams.lines[0].startswith(
                "refused: [P4] at --parallel "
            )
        finally:
            heartbeat.finish(LOST_STATUS)

    def test_finish_stops_the_thread_and_writes_done(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))  # a live peer: nothing is lost
        heartbeat.start()
        thread = _thread_named("causalab-heartbeat-0")
        assert thread is not None
        heartbeat.finish(0)
        assert not thread.is_alive()
        assert decode(store.get(key_for(0))) == Done(0)
        assert seams.statuses == [] and heartbeat.refusal is None

    def test_finish_waits_no_longer_than_the_grace_for_a_tick_stuck_in_the_store(
        self,
    ) -> None:
        """A store read that never returns — the host's socket half-open —
        holds the thread inside ``tick``; ``finish`` joins it for a grace at
        most and still writes ``done``. The store's ``set`` is not stuck, as
        the write side of a stuck read is not."""

        class _StuckStore(FakeStore):
            def __init__(self) -> None:
                super().__init__()
                self.stuck = False
                self.entered = threading.Event()
                self.release = threading.Event()

            def check(self, keys: Sequence[str]) -> bool:
                return True

            def get(self, key: str) -> bytes:
                if self.stuck:
                    self.entered.set()
                    self.release.wait()  # until the test lets go, however long
                return encode(Beat(1)).encode()

        store, seams, clock = _StuckStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        heartbeat.start()  # the synchronous first tick reads the peer at once
        store.stuck = True
        try:
            assert store.entered.wait(PATIENCE), "the thread never ticked again"
            finisher = threading.Thread(target=heartbeat.finish, args=(0,))
            finisher.start()
            finisher.join(PATIENCE)
            assert not finisher.is_alive(), (
                "finish waited past the grace on a stuck tick"
            )
            assert decode(store.values[key_for(0)]) == Done(0)
        finally:
            store.release.set()
        assert seams.statuses == []


def _writer(
    store: FakeStore, peer: int, stop: threading.Event, clock: list[float]
) -> threading.Thread:
    """A live peer: a thread writing an advancing beat under ``peer``'s key
    every millisecond until ``stop``, the driven clock advancing with it."""

    def beat() -> None:
        count = 0
        while not stop.is_set():
            count += 1
            store.set(key_for(peer), encode(Beat(count)))
            clock[0] += 0.001
            stop.wait(0.001)

    thread = threading.Thread(target=beat, name=f"peer-{peer}", daemon=True)
    thread.start()
    return thread


class TestConsult:
    def test_a_peer_beating_twice_after_the_failure_clears_the_collective(
        self,
    ) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.tick(0.0)
        failed_at = 0.01
        assert heartbeat.consult(failed_at, 0.01) is None
        store.set(key_for(1), encode(Beat(2)))
        heartbeat.tick(0.02)
        assert heartbeat.consult(failed_at, 0.02) is None, "one change is no proof"
        store.set(key_for(1), encode(Beat(3)))
        heartbeat.tick(0.03)
        assert heartbeat.consult(failed_at, 0.03) == "collective"

    def test_a_peer_gone_silent_is_the_blame(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.tick(0.0)
        failed_at = 0.01
        for now in (0.02, 0.05, 0.09):
            heartbeat.tick(now)
            assert heartbeat.consult(failed_at, now) is None
        heartbeat.tick(0.1)
        assert heartbeat.consult(failed_at, 0.1) == Lost(1, "silent", silence=0.1)

    def test_a_peer_that_exited_with_a_status_is_the_blame(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Done(3)))
        heartbeat.tick(0.0)
        heartbeat.tick(QUICK.grace)
        assert heartbeat.consult(0.0, QUICK.grace) == Lost(1, "exited", status=3)

    def test_the_bound_passing_with_nobody_lost_is_the_collectives(self) -> None:
        """A peer that beat once, half a grace after the failure, and never
        again: not lost until 1.5 graces, not confirmed alive either; at
        the bound (1.3 graces) the failure is the collective's own."""
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.tick(0.0)
        failed_at = 0.0
        store.set(key_for(1), encode(Beat(2)))
        heartbeat.tick(0.05)
        just_short = round(QUICK.bound - QUICK.interval, 9)
        heartbeat.tick(just_short)
        assert heartbeat.consult(failed_at, just_short) is None
        heartbeat.tick(QUICK.bound)
        assert heartbeat.consult(failed_at, QUICK.bound) == "collective"


class TestHold:
    def test_hold_returns_once_every_peer_has_beaten_twice_since_the_failure(
        self,
    ) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        stop = threading.Event()
        writer = _writer(store, 1, stop, clock)
        heartbeat.start()
        try:
            holder = threading.Thread(target=heartbeat.hold)
            holder.start()
            holder.join(PATIENCE)
            assert not holder.is_alive(), "hold never returned with a live peer"
            assert seams.statuses == [] and heartbeat.refusal is None
        finally:
            stop.set()
            writer.join(PATIENCE)
            heartbeat.finish(1)

    def test_hold_ends_in_one_refusal_when_the_peer_is_lost(self) -> None:
        """The peer beat once, is silent since, and the driven clock is
        past the grace: the thread's tick and the holder both see the loss;
        the refusal is written and the exit taken exactly once, and
        ``hold`` returns (through the exit seam here; ``os._exit`` in the
        rank)."""
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.start()
        try:
            clock[0] = 1.0
            holder = threading.Thread(target=heartbeat.hold)
            holder.start()
            holder.join(PATIENCE)
            assert not holder.is_alive(), "hold never returned on a lost peer"
            assert seams.exited.wait(PATIENCE)
            thread = _thread_named("causalab-heartbeat-0")
            if thread is not None:
                thread.join(PATIENCE)
            assert seams.statuses == [LOST_STATUS]
            assert len(seams.lines) == 1
            assert heartbeat.refusal is not None
            assert heartbeat.refusal.lost.rank == 1
            assert heartbeat.refusal.lost.why == "silent"
        finally:
            heartbeat.finish(LOST_STATUS)

    def test_hold_returns_at_once_on_a_finished_heartbeat(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        heartbeat.start()
        heartbeat.finish(0)
        holder = threading.Thread(target=heartbeat.hold)
        holder.start()
        holder.join(PATIENCE)
        assert not holder.is_alive()
        assert seams.statuses == []


class TestRunning:
    def test_start_registers_the_process_heartbeat_and_finish_clears_it(
        self,
    ) -> None:
        assert running() is None
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.start()
        try:
            assert running() is heartbeat
        finally:
            heartbeat.finish(0)
        assert running() is None
