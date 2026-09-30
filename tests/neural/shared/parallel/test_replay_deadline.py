"""The replay deadline (``docs/model_parallelism.md`` §3, ``docs/cuda_graphs.md``
"Hung replays"): device work no backend watchdog tracks, bounded by the
collective timeout.

``unit``: [`Outstanding`][causalab.neural.shared.parallel.deadline.Outstanding]
times the head of each stream's queue from its enqueue, drains completed heads
on every enqueue and check, and counts a query that raises as complete; the
refusal names the work, the wait, the rank and the timeout. ``property``: over
drawn enqueue times and completion times that respect stream order, the
verdict is exactly the oldest incomplete entry once it has waited the timeout.
The heartbeat: its round refuses an overdue replay the way it refuses a lost
peer — one ``refused: [P4] …`` line, ``LOST_STATUS`` through the exit seam —
a lost peer takes precedence, a finished heartbeat refuses nothing, and the
thread itself ends a rank whose peer finished cleanly while its replay hangs
(a case that would otherwise need an outside kill).
"""

from __future__ import annotations

import dataclasses
import threading
import weakref

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.deadline import (
    REPLAY,
    Outstanding,
    Overdue,
    ReplayOverdue,
    describe_overdue,
)
from causalab.neural.shared.parallel.heartbeat import Heartbeat, done_key_for, key_for
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    LOST_STATUS,
    Beat,
    Done,
    Lost,
    Settings,
    encode,
)
from causalab.protocol.parallel import format_geometry, parse_geometry
from tests._helpers.simulated_world.heartbeat import FakeStore

GEOMETRY = parse_geometry("tp=2")
#: the thread ticks every 10 ms; the deadline is half a second
QUICK = Settings(timeout=0.5, grace=0.1)
PATIENCE = 5.0


@dataclasses.dataclass
class _Work:
    """A completion the test sets; ``asked`` counts the queries."""

    done: bool = False
    asked: int = 0

    def query(self) -> bool:
        self.asked += 1
        return self.done


class _Broken:
    def query(self) -> bool:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestOutstanding:
    def test_nothing_enqueued_is_never_overdue(self) -> None:
        assert Outstanding(1.0).overdue(1e9) is None

    def test_pending_work_is_overdue_exactly_once_the_timeout_has_passed(self) -> None:
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, _Work(), now=5.0)
        assert outstanding.overdue(14.9) is None
        assert outstanding.overdue(15.0) == Overdue(REPLAY, 10.0)
        assert outstanding.overdue(20.0) == Overdue(REPLAY, 15.0)

    def test_completed_work_is_drained_and_never_overdue(self) -> None:
        outstanding = Outstanding(10.0)
        work = _Work()
        outstanding.enqueue("s", REPLAY, work, now=0.0)
        work.done = True
        assert outstanding.overdue(100.0) is None
        assert outstanding.pending() == 0

    def test_the_head_is_timed_from_its_own_enqueue_not_its_predecessors(
        self,
    ) -> None:
        """A healthy stream that keeps replaying is never overdue: each head
        is timed from its own enqueue once the one before it completed."""
        outstanding = Outstanding(10.0)
        first, second = _Work(), _Work()
        outstanding.enqueue("s", "first", first, now=0.0)
        outstanding.enqueue("s", "second", second, now=8.0)
        first.done = True
        assert outstanding.overdue(12.0) is None
        assert outstanding.overdue(18.0) == Overdue("second", 10.0)

    def test_an_incomplete_head_holds_the_later_entries(self) -> None:
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", "first", _Work(), now=0.0)
        outstanding.enqueue("s", "second", _Work(), now=9.0)
        assert outstanding.overdue(10.0) == Overdue("first", 10.0)

    def test_enqueue_drains_so_a_queue_holds_only_work_in_flight(self) -> None:
        outstanding = Outstanding(10.0)
        for step in range(1000):
            outstanding.enqueue("s", REPLAY, _Work(done=True), now=float(step))
        assert outstanding.pending() == 1

    def test_each_stream_is_its_own_queue_and_the_oldest_head_is_named(
        self,
    ) -> None:
        outstanding = Outstanding(10.0)
        outstanding.enqueue("a", "on a", _Work(), now=3.0)
        outstanding.enqueue("b", "on b", _Work(), now=1.0)
        assert outstanding.overdue(12.0) == Overdue("on b", 11.0)

    def test_a_query_that_raises_counts_as_complete(self) -> None:
        """The device error is the main thread's to raise at its next sync."""
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, _Broken(), now=0.0)
        assert outstanding.overdue(100.0) is None
        assert outstanding.pending() == 0


# (stream, enqueue time, completion time or None) — per stream, enqueues and
# completions are both nondecreasing: the order a stream runs its work in
_ENTRIES = st.lists(
    st.tuples(
        st.sampled_from(["a", "b", "c"]),
        st.floats(0.0, 50.0),
        st.one_of(st.none(), st.floats(0.0, 100.0)),
    ),
    max_size=12,
)


@pytest.mark.unit
class TestTheCaptureWindow:
    """While this rank captures a CUDA graph the deadline asks nothing: a
    completed event queried from another thread during a global-mode capture
    fails and invalidates the capture."""

    def test_nothing_is_queried_while_a_capture_is_open(self) -> None:
        outstanding = Outstanding(10.0)
        work = _Work()
        outstanding.enqueue("s", REPLAY, work, now=0.0)
        with outstanding.capturing():
            asked = work.asked
            assert outstanding.overdue(1e9) is None
            assert work.asked == asked
        # the window closed: the deadline sees the same work again
        assert outstanding.overdue(1e9) == Overdue(REPLAY, 1e9)

    def test_opening_the_window_drains_completed_work(self) -> None:
        outstanding = Outstanding(10.0)
        work = _Work(done=True)
        outstanding.enqueue("s", REPLAY, work, now=0.0)
        with outstanding.capturing():
            assert outstanding.pending() == 0

    def test_nested_windows_resume_only_when_the_outer_one_closes(self) -> None:
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, _Work(), now=0.0)
        with outstanding.capturing():
            with outstanding.capturing():
                pass
            assert outstanding.overdue(1e9) is None
        assert outstanding.overdue(1e9) is not None

    def test_a_window_waits_for_a_query_in_flight(self) -> None:
        """The query and the window share one lock, so a capture never
        begins while the heartbeat thread is inside a query."""
        release = threading.Event()
        querying = threading.Event()

        class _Slow:
            def query(self) -> bool:
                querying.set()
                release.wait(PATIENCE)
                return False

        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, _Slow(), now=0.0)
        checker = threading.Thread(target=outstanding.overdue, args=(1.0,))
        checker.start()
        assert querying.wait(PATIENCE)
        opened = threading.Event()

        def open_window() -> None:
            with outstanding.capturing():
                opened.set()

        capture = threading.Thread(target=open_window)
        capture.start()
        assert not opened.wait(0.05)
        release.set()
        assert opened.wait(PATIENCE)
        checker.join(PATIENCE)
        capture.join(PATIENCE)


@pytest.mark.unit
class TestReleaseStaysWithTheEnqueuingThread:
    """Completed work is released by the thread that enqueues, never by the
    heartbeat's: destroying a CUDA event takes the driver's write lock, which
    a ``cuGraphLaunch`` blocked behind a stuck collective holds, and the
    destroying thread holds the GIL meanwhile — every Python thread, the
    heartbeat included, then stops."""

    def _tracked(self, released: list[str]) -> _Work:
        work = _Work(done=True)
        weakref.finalize(work, lambda: released.append(threading.current_thread().name))
        return work

    def test_the_checking_thread_retires_but_does_not_release(self) -> None:
        released: list[str] = []
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, self._tracked(released), now=0.0)
        checker = threading.Thread(
            target=outstanding.overdue, args=(1.0,), name="checker"
        )
        checker.start()
        checker.join(PATIENCE)
        assert outstanding.pending() == 0
        assert released == []
        outstanding.enqueue("s", REPLAY, _Work(), now=1.0)
        assert released == [threading.current_thread().name]

    def test_opening_a_capture_window_releases_on_the_opening_thread(self) -> None:
        released: list[str] = []
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, self._tracked(released), now=0.0)
        with outstanding.capturing():
            pass
        assert released == [threading.current_thread().name]

    def test_the_checking_thread_keeps_no_entry_past_the_lock(self) -> None:
        """The interleaving the check must survive: the heartbeat leaves the
        lock holding the oldest pending entry; that work completes and the
        enqueuing thread drains and releases it before the check returns.
        The last reference must go with that release, not with the
        checking thread's frame."""
        released: list[str] = []
        outstanding = Outstanding(10.0)
        work = _Work()
        weakref.finalize(work, lambda: released.append(threading.current_thread().name))
        outstanding.enqueue("s", REPLAY, work, now=0.0)
        del work
        left, resume = threading.Event(), threading.Event()
        lock = outstanding._lock  # pyright: ignore[reportPrivateUsage]

        class _Pausing:
            """The real lock; the checking thread pauses just after leaving it."""

            def __enter__(self) -> None:
                lock.acquire()

            def __exit__(self, *_exc: object) -> None:
                lock.release()
                if threading.current_thread().name == "checker":
                    left.set()
                    resume.wait(PATIENCE)

        outstanding._lock = _Pausing()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
        checker = threading.Thread(
            target=outstanding.overdue, args=(1.0,), name="checker"
        )
        checker.start()
        assert left.wait(PATIENCE)
        outstanding._lock = lock  # pyright: ignore[reportPrivateUsage]
        (entry,) = outstanding._queues["s"]  # pyright: ignore[reportPrivateUsage]
        entry.work.done = True  # type: ignore[attr-defined]
        del entry
        outstanding.enqueue("s", REPLAY, _Work(), now=1.0)
        resume.set()
        checker.join(PATIENCE)
        assert released == [threading.current_thread().name]

    def test_work_drained_by_its_own_enqueue_is_released_there(self) -> None:
        released: list[str] = []
        outstanding = Outstanding(10.0)
        outstanding.enqueue("s", REPLAY, self._tracked(released), now=0.0)
        outstanding.enqueue("s", REPLAY, _Work(), now=1.0)
        assert released == [threading.current_thread().name]


@pytest.mark.property
@settings(deadline=None, max_examples=200)
@given(entries=_ENTRIES, now=st.floats(0.0, 120.0), timeout=st.floats(0.5, 60.0))
def test_the_verdict_is_the_oldest_incomplete_entry_past_the_timeout(
    entries: list[tuple[str, float, float | None]], now: float, timeout: float
) -> None:
    # lay each stream's entries in stream order: enqueue times sorted, and a
    # completion no earlier than its predecessor's (an incomplete entry holds
    # every later one incomplete)
    by_stream: dict[str, list[tuple[float, float | None]]] = {}
    for stream, since, done_at in entries:
        by_stream.setdefault(stream, []).append((since, done_at))
    laid: list[tuple[str, str, float, float | None]] = []
    for stream, items in by_stream.items():
        floor: float | None = 0.0
        for index, (since, done_at) in enumerate(sorted(items, key=lambda x: x[0])):
            if floor is None or done_at is None:
                floor = None
            else:
                floor = max(floor, done_at, since)
            laid.append((stream, f"{stream}{index}", since, floor))
    laid.sort(key=lambda entry: entry[2])  # the main thread enqueues in time order

    outstanding = Outstanding(timeout)
    clock = [0.0]

    class _At:
        def __init__(self, done_at: float | None) -> None:
            self.done_at = done_at

        def query(self) -> bool:
            return self.done_at is not None and self.done_at <= clock[0]

    for stream, op, since, done_at in laid:
        clock[0] = since
        outstanding.enqueue(stream, op, _At(done_at), now=since)
    clock[0] = max(now, clock[0])

    incomplete = [
        (since, op)
        for _stream, op, since, done_at in laid
        if done_at is None or done_at > clock[0]
    ]
    verdict = outstanding.overdue(clock[0])
    if not incomplete or clock[0] - min(incomplete)[0] < timeout:
        assert verdict is None
    else:
        since, _op = min(incomplete)
        assert verdict is not None
        assert verdict.waited == clock[0] - since
        assert verdict.op in {op for s, op in incomplete if s == since}


@pytest.mark.unit
def test_the_refusal_names_the_work_the_wait_the_rank_and_the_timeout() -> None:
    refusal = describe_overdue(
        Overdue(REPLAY, 612.4),
        rank=0,
        world=2,
        geometry=GEOMETRY,
        timeout=600.0,
        variable=COLLECTIVE_TIMEOUT_VARIABLE,
    )
    assert isinstance(refusal, ReplayOverdue)
    assert refusal.overdue == Overdue(REPLAY, 612.4)
    text = str(refusal)
    assert text.startswith("[P4] at --parallel ")
    assert (
        "a CUDA graph replay enqueued 612 s ago on rank 0 of 2 has not completed"
        in text
    )
    assert f"--parallel {format_geometry(GEOMETRY)};" in text
    assert "CAUSALAB_COLLECTIVE_TIMEOUT=600" in text


# --------------------------------------------------------------------------- #
# the heartbeat
# --------------------------------------------------------------------------- #


class _Seams:
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


def _peer_finished(store: FakeStore) -> None:
    """The peer finished cleanly: it is never lost, whatever the clock says."""
    store.set(done_key_for(1), encode(Done(0)))
    store.set(key_for(1), encode(Done(0)))


@pytest.mark.unit
class TestTheHeartbeatBoundsReplays:
    def test_an_overdue_replay_is_refused_by_name_and_exits_lost(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        _peer_finished(store)
        heartbeat.enqueued("stream", REPLAY, _Work())
        clock[0] = 0.4
        assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
        assert seams.statuses == []
        clock[0] = 0.5
        assert heartbeat._round() is True  # pyright: ignore[reportPrivateUsage]
        assert seams.statuses == [LOST_STATUS]
        refusal = heartbeat.refusal
        assert isinstance(refusal, ReplayOverdue)
        assert refusal.overdue == Overdue(REPLAY, 0.5)
        assert seams.lines == [f"refused: {refusal}\n"]

    def test_a_replay_that_completes_is_never_refused(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        _peer_finished(store)
        work = _Work()
        heartbeat.enqueued("stream", REPLAY, work)
        work.done = True
        clock[0] = 100.0
        assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
        assert seams.statuses == [] and heartbeat.refusal is None

    def test_a_lost_peer_is_named_before_an_overdue_replay(self) -> None:
        """The death explains the hang: the peer is the refusal, not the replay."""
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        store.set(key_for(1), encode(Beat(1)))
        heartbeat.tick(0.0)
        heartbeat.enqueued("stream", REPLAY, _Work())
        clock[0] = 1.0
        assert heartbeat._round() is True  # pyright: ignore[reportPrivateUsage]
        assert heartbeat.refusal is not None
        assert getattr(heartbeat.refusal, "lost", None) == Lost(
            1, "silent", silence=1.0
        )

    def test_a_finished_heartbeat_refuses_no_replay(self) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        _peer_finished(store)
        heartbeat.enqueued("stream", REPLAY, _Work())
        clock[0] = 100.0
        heartbeat._stopped.set()  # pyright: ignore[reportPrivateUsage]
        assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
        assert seams.statuses == []

    def test_a_replay_check_that_fails_is_written_and_the_watch_goes_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        _peer_finished(store)

        def broken(now: float) -> Overdue | None:
            raise ValueError("a bug in the rule")

        monkeypatch.setattr(heartbeat.outstanding, "overdue", broken)
        assert heartbeat._round() is False  # pyright: ignore[reportPrivateUsage]
        assert seams.lines == [
            "heartbeat: replay check failed (ValueError: a bug in the rule); watching on\n"
        ]
        assert seams.statuses == []

    def test_the_thread_ends_a_rank_whose_replay_hangs_behind_a_finished_peer(
        self,
    ) -> None:
        """The peer skipped the replay and finished cleanly,
        so nobody is lost — only the deadline ends this rank."""
        store, seams, clock = FakeStore(), _Seams(), [0.0]
        heartbeat = _heartbeat(store, seams, clock)
        _peer_finished(store)
        heartbeat.start()
        try:
            heartbeat.enqueued("stream", REPLAY, _Work())
            clock[0] = 0.6
            assert seams.exited.wait(PATIENCE), "the thread never refused"
            assert seams.statuses == [LOST_STATUS]
            assert isinstance(heartbeat.refusal, ReplayOverdue)
        finally:
            heartbeat.finish(LOST_STATUS)
