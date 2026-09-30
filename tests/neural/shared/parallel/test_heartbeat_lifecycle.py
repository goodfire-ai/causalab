"""Startup and clean shutdown stay bounded without losing healthy peers."""

from __future__ import annotations

import threading
from typing import Sequence

import pytest
from hypothesis import given, settings as hypothesis_settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.heartbeat import Heartbeat, key_for
from causalab.neural.shared.parallel.watchdog import Beat, Done, Settings, encode
from causalab.protocol.parallel import parse_geometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world.heartbeat import FakeStore, HeartbeatSimulation
from tests._helpers.simulated_world.schedule import Schedule


@pytest.mark.unit
def test_start_bounds_the_first_store_read() -> None:
    release = threading.Event()
    returned = threading.Event()

    class BlockedStore(FakeStore):
        def check(self, keys: Sequence[str]) -> bool:
            release.wait()
            return super().check(keys)

    watch = Heartbeat(
        BlockedStore(),
        rank=1,
        world=2,
        settings=Settings(timeout=1.0, grace=0.05),
        geometry=parse_geometry("tp=2"),
        exit=lambda _: None,
        write=lambda _: None,
    )

    def start() -> None:
        watch.start()
        returned.set()

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    try:
        assert returned.wait(0.5), "the initial tick bypassed the bounded store round"
    finally:
        release.set()
        thread.join(1)
        watch.finish(1)


@pytest.mark.unit
def test_final_status_write_is_bounded() -> None:
    release = threading.Event()

    class BlockedStore(FakeStore):
        def set(self, key: str, value: str) -> None:
            release.wait()
            super().set(key, value)

    watch = Heartbeat(
        BlockedStore(),
        rank=1,
        world=2,
        settings=Settings(timeout=1.0, grace=0.05),
        geometry=parse_geometry("tp=2"),
        exit=lambda _: None,
        write=lambda _: None,
    )
    thread = threading.Thread(target=watch.finish, args=(1,), daemon=True)
    thread.start()
    try:
        thread.join(0.5)
        assert not thread.is_alive(), "final Done publication hung shutdown"
    finally:
        release.set()
        thread.join(1)


@pytest.mark.property
@given(schedule=ps.schedules(), delay=st.integers(min_value=4, max_value=20))
@hypothesis_settings(max_examples=30, deadline=None)
def test_clean_store_host_waits_for_arbitrarily_ordered_finishers(
    schedule: Schedule,
    delay: int,
) -> None:
    sim = HeartbeatSimulation(
        world=3,
        grace=2.0,
        schedule=schedule,
        finishes={0: 5.0, 1: 5.0 + delay, 2: 6.0},
    )
    exits = sim.run(until=40.0)
    assert all(refusal is None for _, refusal in exits.values())
    assert exits[0][0] >= exits[1][0], "the store host left before its healthy peer"


@pytest.mark.unit
def test_host_completion_keeps_beating_until_the_peer_finishes() -> None:
    store = FakeStore()
    store.set(key_for(1), encode(Beat(1)))
    statuses: list[int] = []
    watch = Heartbeat(
        store,
        rank=0,
        world=2,
        settings=Settings(timeout=1.0, grace=0.1),
        geometry=parse_geometry("tp=2"),
        exit=statuses.append,
        write=lambda _: None,
    )
    watch.start()
    completed = threading.Event()

    def finish() -> None:
        watch.complete()
        watch.finish(0)
        completed.set()

    thread = threading.Thread(target=finish, daemon=True)
    thread.start()
    try:
        for count in range(2, 8):
            store.set(key_for(1), encode(Beat(count)))
            assert not completed.wait(0.02)
        store.set(key_for(1), encode(Done(0)))
        assert completed.wait(1.0)
        assert not statuses
    finally:
        store.set(key_for(1), encode(Done(0)))
        thread.join(1)
        watch.finish(1)


@pytest.mark.unit
def test_live_peer_that_never_finishes_has_a_bounded_completion_wait() -> None:
    from causalab.neural.shared.parallel.watchdog import CompletionTimeout

    watch = Heartbeat(
        FakeStore(),
        rank=0,
        world=2,
        settings=Settings(timeout=1.0, grace=0.1),
        geometry=parse_geometry("tp=2"),
        exit=lambda _: None,
        write=lambda _: None,
    )
    assert not watch.completion_ready(3.0)
    assert not watch.completion_ready(3.99)
    with pytest.raises(CompletionTimeout, match="clean shutdown is bounded"):
        watch.completion_ready(4.0)


@pytest.mark.property
@given(schedule=ps.schedules())
@hypothesis_settings(max_examples=30, deadline=None)
def test_host_still_detects_a_dead_peer_while_waiting_to_complete(
    schedule: Schedule,
) -> None:
    sim = HeartbeatSimulation(
        world=2,
        grace=2.0,
        schedule=schedule,
        finishes={0: 1.0},
        deaths={1: 3.0},
    )
    exits = sim.run(until=10.0)
    when, refusal = exits[0]
    assert refusal is not None and refusal.lost.rank == 1
    assert when <= 3.0 + sim.settings.bound


@pytest.mark.unit
def test_late_inflight_beat_cannot_hide_a_terminal_status() -> None:
    entered, release = threading.Event(), threading.Event()

    class DelayedBeat(FakeStore):
        def set(self, key: str, value: str) -> None:
            if key == key_for(1) and value.startswith("beat:"):
                entered.set()
                release.wait()
            super().set(key, value)

    store = DelayedBeat()

    def watch(rank: int) -> Heartbeat:
        return Heartbeat(
            store,
            rank=rank,
            world=2,
            settings=Settings(timeout=1.0, grace=0.05),
            geometry=parse_geometry("tp=2"),
            exit=lambda _: None,
            write=lambda _: None,
        )

    peer = watch(1)
    old_tick = threading.Thread(target=lambda: peer.tick(0.0), daemon=True)
    old_tick.start()
    assert entered.wait(1)
    try:
        peer.finish(0)
    finally:
        release.set()
        old_tick.join(1)
    observer = watch(0)
    assert observer.tick(0.0) is None
    assert observer.tick(10.0) is None, "late Beat hid the peer's completed status"
