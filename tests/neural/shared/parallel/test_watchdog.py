"""The rank watchdog (``docs/model_parallelism.md`` §3 "when a rank dies",
§11): the settings, the liveness verdict and the heartbeat protocol.

``unit``: the two settings (``CAUSALAB_COLLECTIVE_TIMEOUT``,
``CAUSALAB_RANK_GRACE``) parsed once, every malformed value refused by name;
the signal codec; the refusal's text. ``property``: the verdict never names
a rank that beats or finished cleanly, and names a silent one exactly once
the grace has passed. **Deterministic simulation** of the protocol itself
(``tests/_helpers/simulated_world/heartbeat.py``, the driver every heartbeat
scenario shares): ``world`` [`Heartbeat`][causalab.neural.shared.parallel.heartbeat.Heartbeat] monitors over one in-process
store and one clock, ticked in the order a drawn schedule picks — the same
seed-or-tape schedule the collective simulator takes — a rank *dies* by
ceasing to tick (a kill, an OOM), *fails* by writing its non-zero status
(a refusal of its own), and every survivor's verdict must name it within
``grace + interval`` of the event and never before ``grace``; rank 0's death
takes the store with it and is named as the store's host; a world that
finishes cleanly names nobody; a rank still running when the clock runs out
is a typed `Unfinished`, never a missing key. **A collective that
fails** on a rank (``collective_failures``; gloo's read fails the instant a
peer's socket closes, before any beat is missed) is the heartbeat's to
explain: the rank exits with the named refusal when a peer is lost within
the bound, and with the collective's own refusal (`OwnRefusal`) only
once every peer has beaten twice after the failure — or the bound has
passed with nobody lost. **The three cases that never ran on hardware**
(§3): a rank *stopped* (``stops``: a ``SIGSTOP``) is named silent exactly
as a dead one and never exits; a rank *wedged* (``wedges``: its main
thread stuck, its heartbeat beating) is named by nobody — its peers take
the collective's own refusal at the timeout, and it names *them* a grace
later; *rank 0's every exit takes the store*, so a survivor that ticks
after the host's refusal names the victim when its silence had reached the
grace before the store went, else the host a grace later; a rank *unreached*
(dead before it connected) is named ``never reached the rendezvous`` once
the timeout has passed, never before. The thread and the ``os._exit`` are
the one seam the simulation does not run; the real worlds are
``tests/neural/engines/pytorch_hooks/test_rank_watchdog_run.py`` and
``test_rank_watchdog_cases_run.py``.
"""

from __future__ import annotations

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel import watchdog
from causalab.neural.shared.parallel.watchdog import (
    BEATS_PER_GRACE,
    BOUND_BEATS,
    COLLECTIVE_TIMEOUT_S,
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_S,
    RANK_GRACE_VARIABLE,
    Beat,
    Done,
    Liveness,
    Lost,
    RankLost,
    Settings,
    WatchdogSetting,
    decode,
    describe,
    encode,
    parse_seconds,
)
from causalab.protocol.parallel import parse_geometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, Unfinished
from tests._helpers.simulated_world.schedule import Pick
from tests._helpers.simulated_world.heartbeat import (
    HeartbeatSimulation,
    OwnRefusal,
    UnsimulatedHost,
)

_PROPERTY = settings(deadline=None, max_examples=60)
#: the drawn-schedule scenarios: the repository's ``max_examples=30`` idiom
_SCHEDULED = settings(deadline=None, max_examples=30)


# --------------------------------------------------------------------------- #
# the settings
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSettings:
    def test_the_defaults_are_the_documented_ones(self) -> None:
        parsed = Settings.from_environment({})
        assert parsed == Settings(timeout=COLLECTIVE_TIMEOUT_S, grace=RANK_GRACE_S)
        assert parsed.timeout == 600.0 and parsed.grace == 30.0
        assert parsed.interval == RANK_GRACE_S / BEATS_PER_GRACE
        # the documented bound on a refusal after a death: the grace plus
        # three beats, 1.3 × grace
        assert BOUND_BEATS == 3
        assert parsed.bound == RANK_GRACE_S + 3 * parsed.interval == 39.0

    def test_the_environment_names_both(self) -> None:
        parsed = Settings.from_environment(
            {COLLECTIVE_TIMEOUT_VARIABLE: "90", RANK_GRACE_VARIABLE: "4.5"}
        )
        assert parsed == Settings(timeout=90.0, grace=4.5)
        assert parsed.interval == 0.45

    def test_an_empty_value_is_the_default(self) -> None:
        assert Settings.from_environment({RANK_GRACE_VARIABLE: "  "}).grace == 30.0

    @pytest.mark.parametrize("text", ["ten", "-1", "0", "nan", "inf", "1e400"])
    def test_a_malformed_value_is_refused_by_name(self, text: str) -> None:
        for variable in (COLLECTIVE_TIMEOUT_VARIABLE, RANK_GRACE_VARIABLE):
            with pytest.raises(WatchdogSetting) as refused:
                Settings.from_environment({variable: text})
            assert variable in str(refused.value) and text in str(refused.value)
            assert refused.value.code == "P4" and refused.value.path == "--parallel"

    def test_the_parser_itself_refuses_a_zero_and_points_at_the_document(self) -> None:
        """``Settings`` re-checks its fields, so ``from_environment`` would
        refuse a zero either way; the public parser refuses it first, as not
        positive, and every refusal of its own points at §11."""
        with pytest.raises(WatchdogSetting, match="not a positive, finite") as refused:
            parse_seconds(RANK_GRACE_VARIABLE, "0", RANK_GRACE_S)
        assert str(refused.value).endswith("(docs/model_parallelism.md §11)")
        with pytest.raises(WatchdogSetting, match="is not a number of seconds") as bad:
            parse_seconds(COLLECTIVE_TIMEOUT_VARIABLE, "ten", COLLECTIVE_TIMEOUT_S)
        assert str(bad.value).endswith("(docs/model_parallelism.md §11)")
        assert parse_seconds(RANK_GRACE_VARIABLE, "1e-9", RANK_GRACE_S) == 1e-9

    def test_a_grace_at_or_above_the_timeout_is_refused_by_name(self) -> None:
        with pytest.raises(WatchdogSetting) as refused:
            Settings.from_environment(
                {COLLECTIVE_TIMEOUT_VARIABLE: "20", RANK_GRACE_VARIABLE: "20"}
            )
        message = str(refused.value)
        assert COLLECTIVE_TIMEOUT_VARIABLE in message and RANK_GRACE_VARIABLE in message

    @_PROPERTY
    @given(
        timeout=st.floats(min_value=1.0, max_value=1e6, allow_nan=False),
        fraction=st.floats(min_value=1e-3, max_value=0.99),
    )
    def test_any_positive_pair_with_grace_below_timeout_parses(
        self, timeout: float, fraction: float
    ) -> None:
        grace = timeout * fraction
        parsed = Settings.from_environment(
            {
                COLLECTIVE_TIMEOUT_VARIABLE: repr(timeout),
                RANK_GRACE_VARIABLE: repr(grace),
            }
        )
        assert parsed.timeout == timeout and parsed.grace == grace
        assert 0 < parsed.interval < parsed.grace


# --------------------------------------------------------------------------- #
# the signal codec and the refusal
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSignals:
    @_PROPERTY
    @given(count=st.integers(min_value=0, max_value=10**12))
    def test_a_beat_round_trips(self, count: int) -> None:
        assert decode(encode(Beat(count))) == Beat(count)
        assert decode(encode(Beat(count)).encode()) == Beat(count)

    @_PROPERTY
    @given(status=st.integers(min_value=0, max_value=255))
    def test_a_done_round_trips(self, status: int) -> None:
        assert decode(encode(Done(status))) == Done(status)

    @pytest.mark.parametrize("text", ["", "beat", "beat:x", "done:", "alive:3", "3"])
    def test_anything_else_is_refused_by_name(self, text: str) -> None:
        with pytest.raises(watchdog.MalformedSignal) as refused:
            decode(text)
        assert repr(text) in str(refused.value)

    def test_the_refusal_names_the_rank_the_status_and_the_collective(self) -> None:
        geometry = parse_geometry("ep=4")
        config = Settings(timeout=600.0, grace=30.0)
        exited = describe(
            Lost(3, "exited", status=3),
            rank=1,
            world=4,
            geometry=geometry,
            settings=config,
            where=("expert", "all_reduce_sum"),
        )
        assert isinstance(exited, RankLost) and isinstance(exited, ValueError)
        text = str(exited)
        assert text.startswith("[P4] at --parallel ")
        assert "rank 3 of 4 exited with status 3" in text
        assert "rank 1" in text and "all_reduce_sum" in text and "'expert'" in text
        assert "ep=4" in text and f"{RANK_GRACE_VARIABLE}=30" in text
        silent = describe(
            Lost(2, "silent", silence=31.2),
            rank=0,
            world=4,
            geometry=geometry,
            settings=config,
            where=None,
        )
        assert "rank 2 of 4 has sent no heartbeat for 31 s" in str(silent)
        assert "waiting in" not in str(silent)
        host = describe(
            Lost(0, "unreachable"),
            rank=2,
            world=4,
            geometry=geometry,
            settings=config,
            where=("pipeline", "recv"),
        )
        assert "rank 0 of 4, the rendezvous store's host, is unreachable" in str(host)
        assert exited.lost.rank == 3 and host.lost.why == "unreachable"
        agent = describe(
            Lost(0, "unreachable"),
            rank=2,
            world=4,
            geometry=geometry,
            settings=config,
            where=None,
            store_host="agent",
        )
        assert (
            "the rendezvous store, hosted by rank 0's torchrun agent, is unreachable "
            "(the agent exits when rank 0 fails, or its node is gone) and rank 2 is "
            "still running"
        ) in str(agent)
        unreached = describe(
            Lost(3, "unreached", silence=600.4),
            rank=0,
            world=4,
            geometry=geometry,
            settings=config,
            where=None,
        )
        text = str(unreached)
        assert "rank 3 of 4 never reached the rendezvous in 600 s" in text
        assert f"the wait for it is {COLLECTIVE_TIMEOUT_VARIABLE}'s" in text
        assert "rank 0 is still running:" in text and "waiting in" not in text

    def test_current_is_one_read_under_a_churning_main_thread(self) -> None:
        """The heartbeat thread reads ``current()`` while the main thread
        pushes and pops: a test-then-index would raise ``IndexError`` in
        the gap, and the thread that raised is the watch."""
        import threading

        stop = threading.Event()
        seen: list[BaseException] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    watchdog.current()
            except BaseException as err:  # noqa: BLE001 — the failure under test
                seen.append(err)

        thread = threading.Thread(target=reader)
        thread.start()
        for _ in range(20000):
            with watchdog.inside("tensor", "all_gather"):
                pass
        stop.set()
        thread.join(5.0)
        assert seen == [] and watchdog.current() is None

    def test_the_current_collective_is_registered_around_a_call(self) -> None:
        assert watchdog.current() is None
        with watchdog.inside("tensor", "all_gather"):
            assert watchdog.current() == ("tensor", "all_gather")
            with watchdog.inside("pipeline", "send"):
                assert watchdog.current() == ("pipeline", "send")
            assert watchdog.current() == ("tensor", "all_gather")
        assert watchdog.current() is None
        with pytest.raises(ValueError), watchdog.inside("model", "broadcast"):
            raise ValueError("the register is cleared on the way out too")
        assert watchdog.current() is None


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #


GRACE = 10.0


@pytest.mark.unit
class TestLiveness:
    def test_a_beating_world_names_nobody(self) -> None:
        liveness = Liveness(rank=0, world=3, grace=GRACE, now=0.0)
        for step in range(50):
            now = step * 1.0
            for peer in (1, 2):
                liveness.observe(peer, Beat(step), now)
            assert liveness.verdict(now) is None

    def test_a_silent_peer_is_lost_once_the_grace_has_passed(self) -> None:
        liveness = Liveness(rank=0, world=2, grace=GRACE, now=0.0)
        liveness.observe(1, Beat(1), 1.0)
        liveness.observe(1, Beat(2), 2.0)
        for now in (2.0, 5.0, 11.9):
            liveness.observe(1, Beat(2), now)
            assert liveness.verdict(now) is None
        liveness.observe(1, Beat(2), 12.0)
        assert liveness.verdict(12.0) == Lost(1, "silent", silence=10.0)

    def test_a_peer_never_seen_is_absent_once_the_arrival_bound_has_passed(
        self,
    ) -> None:
        """Without an arrival bound of its own the grace is it (a watch that
        begins once every peer is known to be there); the production watch
        hands the collective timeout, since a peer not yet seen may still
        be starting — and a peer seen once is the grace's from then on."""
        liveness = Liveness(rank=1, world=2, grace=GRACE, now=100.0)
        liveness.observe(0, None, 105.0)
        assert liveness.verdict(105.0) is None
        liveness.observe(0, None, 110.0)
        assert liveness.verdict(110.0) == Lost(0, "unreached", silence=10.0)
        starting = Liveness(rank=1, world=2, grace=GRACE, now=100.0, arrival=50.0)
        for now in (110.0, 149.9):
            starting.observe(0, None, now)
            assert starting.verdict(now) is None, "it may still be starting"
        assert starting.verdict(150.0) == Lost(0, "unreached", silence=50.0)
        arrived = Liveness(rank=1, world=2, grace=GRACE, now=100.0, arrival=50.0)
        arrived.observe(0, Beat(1), 120.0)
        assert arrived.verdict(129.9) is None
        assert arrived.verdict(130.0) == Lost(0, "silent", silence=10.0)
        # a key that reads as nothing after a beat (a malformed byte) is the
        # silence of a peer that arrived — the grace's, never an absence
        garbled = Liveness(rank=1, world=2, grace=GRACE, now=100.0, arrival=50.0)
        garbled.observe(0, Beat(1), 101.0)
        garbled.observe(0, None, 102.0)
        assert garbled.verdict(111.9) is None
        assert garbled.verdict(112.0) == Lost(0, "silent", silence=10.0)

    def test_a_peer_silent_for_the_grace_before_the_store_went_is_named_by_its_silence(
        self,
    ) -> None:
        """The store goes with rank 0's exit (its refusal of the same loss
        included): what this rank read before still counts. Rank 2's silence
        had reached the grace when the store went — named at that tick;
        rank 3's had not — its silence is the store's, and the host is
        named a grace after the store went."""
        liveness = Liveness(rank=1, world=4, grace=GRACE, now=0.0)
        liveness.observe(0, Beat(5), 9.0)
        liveness.observe(2, Beat(3), 0.0)
        liveness.observe(3, Beat(3), 1.0)
        liveness.unreachable(10.0)
        assert liveness.verdict(10.0) == Lost(2, "silent", silence=10.0)
        younger = Liveness(rank=1, world=4, grace=GRACE, now=0.0)
        younger.observe(0, Beat(5), 9.0)
        younger.observe(3, Beat(3), 1.0)
        younger.observe(2, Beat(0), 0.5)
        younger.unreachable(10.0)
        for now in (10.0, 11.0, 19.9):
            assert younger.verdict(now) is None, "a silence the store explains"
        assert younger.verdict(20.0) == Lost(0, "unreachable")
        # a failed exit read before the store went is a fact, not a silence
        exited = Liveness(rank=1, world=3, grace=GRACE, now=0.0)
        exited.observe(2, Done(3), 5.0)
        exited.unreachable(6.0)
        assert exited.verdict(14.9) is None
        assert exited.verdict(15.0) == Lost(2, "exited", status=3)

    @_PROPERTY
    @given(
        world=st.integers(min_value=3, max_value=6),
        grace=st.floats(min_value=0.5, max_value=60.0),
        data=st.data(),
    )
    def test_once_the_store_is_gone_only_a_silence_older_than_the_grace_names_a_peer(
        self, world: int, grace: float, data: st.DataObject
    ) -> None:
        """After the store went at ``gone``: a peer whose last change lies a
        grace or more before ``gone`` is named (the lowest such); otherwise
        the host is named once ``gone + grace`` has passed, and nobody
        before that."""
        me = data.draw(st.integers(min_value=1, max_value=world - 1))
        peers = [r for r in range(world) if r != me]
        last = {
            p: data.draw(st.floats(min_value=0.0, max_value=100.0), label=f"t{p}")
            for p in peers
        }
        liveness = Liveness(rank=me, world=world, grace=grace, now=0.0)
        for peer in peers:
            liveness.observe(peer, Beat(1), last[peer])
        gone = data.draw(st.floats(min_value=0.0, max_value=200.0), label="gone")
        liveness.unreachable(gone)
        now = gone + data.draw(st.floats(min_value=0.0, max_value=200.0), label="dt")
        older = [p for p in peers if gone - last[p] >= grace and now - last[p] >= grace]
        verdict = liveness.verdict(now)
        if older:
            assert verdict == Lost(older[0], "silent", silence=now - last[older[0]])
        elif now - gone >= grace:
            assert verdict == Lost(0, "unreachable")
        else:
            assert verdict is None

    def test_a_clean_finish_is_never_lost(self) -> None:
        liveness = Liveness(rank=0, world=2, grace=GRACE, now=0.0)
        liveness.observe(1, Done(0), 1.0)
        for now in (1.0, 50.0, 1e6):
            liveness.observe(1, Done(0), now)
            assert liveness.verdict(now) is None

    def test_a_failed_peer_is_lost_after_the_grace_with_its_status(self) -> None:
        liveness = Liveness(rank=0, world=2, grace=GRACE, now=0.0)
        liveness.observe(1, Beat(1), 1.0)
        liveness.observe(1, Done(3), 4.0)
        assert liveness.verdict(13.9) is None, "its own refusal may be on the way"
        assert liveness.verdict(14.0) == Lost(1, "exited", status=3)

    def test_an_unreachable_store_names_its_host_unless_the_host_finished(
        self,
    ) -> None:
        liveness = Liveness(rank=2, world=3, grace=GRACE, now=0.0)
        liveness.observe(0, Beat(4), 1.0)
        liveness.unreachable(2.0)
        assert liveness.verdict(11.9) is None
        assert liveness.verdict(12.0) == Lost(0, "unreachable")
        finished = Liveness(rank=2, world=3, grace=GRACE, now=0.0)
        finished.observe(0, Done(0), 1.0)
        finished.unreachable(2.0)
        assert finished.verdict(1e6) is None
        assert finished.settled, "the host finished: nothing left to watch"

    def test_the_lowest_lost_rank_is_named(self) -> None:
        liveness = Liveness(rank=1, world=4, grace=GRACE, now=0.0)
        liveness.observe(3, Done(2), 0.0)
        liveness.observe(2, Done(0), 0.0)
        liveness.observe(0, Beat(1), 5.0)
        assert liveness.verdict(10.0) == Lost(3, "exited", status=2)
        # rank 0 has been silent since 5.0: at 15.0 both are lost, 0 is named
        assert liveness.verdict(15.0) == Lost(0, "silent", silence=10.0)

    def test_a_peer_is_alive_since_a_moment_once_its_beat_advanced_twice_after_it(
        self,
    ) -> None:
        """The rule a collective failure consults: one change observed after
        the moment is no proof — the beat may have been written before the
        peer died and read after — two are, since the second was written
        after the first was read. A clean finish is alive (nothing is left
        to lose); a failed exit is not (it is lost once the grace passes)."""
        liveness = Liveness(rank=0, world=3, grace=GRACE, now=0.0)
        liveness.observe(1, Beat(1), 1.0)
        liveness.observe(2, Beat(1), 1.0)
        failed_at = 2.0
        assert not liveness.every_peer_alive_since(failed_at)
        liveness.observe(1, Beat(2), 3.0)
        liveness.observe(2, Beat(2), 3.0)
        assert not liveness.every_peer_alive_since(failed_at), "one change each"
        liveness.observe(1, Beat(3), 4.0)
        assert not liveness.every_peer_alive_since(failed_at), "rank 2 unconfirmed"
        liveness.observe(2, Beat(3), 4.0)
        assert liveness.every_peer_alive_since(failed_at)
        # an unchanged read is not a change; the rule still holds afterwards
        liveness.observe(1, Beat(3), 5.0)
        assert liveness.every_peer_alive_since(failed_at)
        # a change at the moment itself does not count: strictly after
        assert not liveness.every_peer_alive_since(3.0)
        assert liveness.every_peer_alive_since(2.999)

    def test_a_clean_finish_is_alive_and_a_failed_exit_is_not(self) -> None:
        liveness = Liveness(rank=0, world=3, grace=GRACE, now=0.0)
        liveness.observe(1, Done(0), 1.0)
        liveness.observe(2, Done(3), 1.0)
        assert not liveness.every_peer_alive_since(0.0)
        liveness.observe(2, Beat(1), 2.0)
        liveness.observe(2, Beat(2), 3.0)
        assert liveness.every_peer_alive_since(0.0)
        # a peer never seen (its key unwritten) is not alive
        unseen = Liveness(rank=0, world=2, grace=GRACE, now=0.0)
        unseen.observe(1, None, 1.0)
        unseen.observe(1, None, 2.0)
        assert not unseen.every_peer_alive_since(0.0)

    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=6),
        grace=st.floats(min_value=0.5, max_value=60.0),
        data=st.data(),
    )
    def test_the_verdict_is_exactly_the_grace_rule(
        self, world: int, grace: float, data: st.DataObject
    ) -> None:
        """Over a drawn schedule of observations, a peer is lost iff its
        last change (or the start) lies at least ``grace`` back, a clean
        finish never, and the verdict is the lowest such rank."""
        me = data.draw(st.integers(min_value=0, max_value=world - 1))
        peers = [r for r in range(world) if r != me]
        finished = {p for p in peers if data.draw(st.booleans(), label=f"done{p}")}
        last_change = {
            p: data.draw(st.floats(min_value=0.0, max_value=100.0), label=f"t{p}")
            for p in peers
        }
        liveness = Liveness(rank=me, world=world, grace=grace, now=0.0)
        for peer in peers:
            liveness.observe(peer, Beat(1), last_change[peer])
            if peer in finished:
                liveness.observe(peer, Done(0), last_change[peer])
        now = data.draw(st.floats(min_value=0.0, max_value=200.0), label="now")
        expected = [
            p for p in peers if p not in finished and now - last_change[p] >= grace
        ]
        verdict = liveness.verdict(now)
        if expected:
            assert verdict is not None and verdict.rank == expected[0]
            assert verdict.why == "silent"
        else:
            assert verdict is None


# --------------------------------------------------------------------------- #
# the protocol, simulated
# --------------------------------------------------------------------------- #


class TestProtocol:
    @pytest.mark.property
    @pytest.mark.parametrize("geometry", ["tp=2", "pp=2", "ep=2"])
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SCHEDULED
    def test_a_rank_that_dies_mid_forward_is_named_by_the_survivor(
        self, geometry: str, schedule: Schedule
    ) -> None:
        """Rank 1 stops beating at 3.0 (a kill inside the forward); rank 0
        refuses at the first tick at or after 3.0 + grace, naming the rank
        as silent, and exits with [`LOST_STATUS`][causalab.neural.shared.parallel.watchdog.LOST_STATUS]."""
        sim = HeartbeatSimulation(
            world=2, grace=2.0, schedule=schedule, deaths={1: 3.0}, geometry=geometry
        )
        exits = sim.run(until=20.0)
        assert set(exits) == {0, 1}
        when, refusal = exits[0]
        assert refusal is not None
        assert refusal.lost.rank == 1 and refusal.lost.why == "silent"
        # the victim's last beat is the tick before its death
        assert 3.0 + sim.grace - sim.interval <= when <= 3.0 + sim.grace + sim.interval
        assert "rank 1 of 2 has sent no heartbeat" in str(refusal)
        assert "rank 0" in str(refusal) and geometry in str(refusal)

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SCHEDULED
    def test_a_world_of_four_names_the_one_dead_rank_on_every_survivor(
        self, schedule: Schedule
    ) -> None:
        """Every survivor names rank 2 within the bound — or, when rank 0
        refused first and took the store with it and this survivor's last
        look at rank 2 was one beat younger than rank 0's, the host: both
        are gone, both refusals are true (`_named_the_victim_or_the_host`)."""
        sim = HeartbeatSimulation(
            world=4, grace=3.0, schedule=schedule, deaths={2: 5.0}, geometry="ep=4"
        )
        exits = sim.run(until=30.0)
        for rank in sim.survivors(2):
            _named_the_victim_or_the_host(sim, exits, rank, victim=2, death=5.0)

    @pytest.mark.unit
    def test_the_hosts_refusal_takes_the_store_and_a_younger_look_names_the_host(
        self,
    ) -> None:
        """The interleaving the completed model exposes: at the deciding
        tick, 4.8 (rank 2 dies at 5.0; the tape decides from 4.7), the tape
        orders the ticks 1, 2, 0, 3, so rank 1 reads rank 2's last beat only
        at 5.1 while ranks 0 and 3 read it at 4.8; rank 0 refuses at 7.8 and
        its store goes with it; rank 3, ticking after, still names rank 2
        (its silence had reached the grace when the store went); rank 1
        holds a silence of 2.7 < 3 that the store explains, and names the
        host at 10.8 — a grace after the store went."""
        sim = HeartbeatSimulation(
            world=4, grace=3.0, schedule=[1, 1, 0], deaths={2: 5.0}, geometry="ep=4"
        )
        exits = sim.run(until=30.0)
        assert sim.deciding_from == pytest.approx(4.7)
        # sixteen rank-order ticks (0.0 .. 4.5) of four picks, then the tape
        assert [p.rank for p in sim.picks[64:68]] == [1, 2, 0, 3]
        for rank in (0, 3):
            when, refusal = exits[rank]
            assert refusal is not None and refusal.lost == Lost(
                2, "silent", silence=3.0
            )
            assert when == pytest.approx(7.8)
        when, refusal = exits[1]
        assert refusal is not None and refusal.lost == Lost(0, "unreachable")
        assert when == pytest.approx(10.8)
        assert "the rendezvous store's host, is unreachable" in str(refusal)

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SCHEDULED
    def test_rank_zero_dying_takes_the_store_and_is_named_as_its_host(
        self, schedule: Schedule
    ) -> None:
        sim = HeartbeatSimulation(
            world=3, grace=2.0, schedule=schedule, deaths={0: 4.0}
        )
        exits = sim.run(until=20.0)
        for rank in (1, 2):
            when, refusal = exits[rank]
            assert refusal is not None
            assert refusal.lost.rank == 0 and refusal.lost.why == "unreachable"
            assert "the rendezvous store's host" in str(refusal)
            assert 6.0 - sim.interval <= when <= 6.0 + sim.interval

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SCHEDULED
    def test_a_rank_that_refuses_on_its_own_is_named_with_its_status(
        self, schedule: Schedule
    ) -> None:
        sim = HeartbeatSimulation(
            world=2, grace=2.0, schedule=schedule, failures={1: (3.0, 3)}
        )
        exits = sim.run(until=20.0)
        when, refusal = exits[0]
        assert refusal is not None and refusal.lost == Lost(1, "exited", status=3)
        assert "rank 1 of 2 exited with status 3" in str(refusal)
        assert 5.0 - sim.interval <= when <= 5.0 + sim.interval

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SCHEDULED
    def test_a_world_that_finishes_names_nobody(self, schedule: Schedule) -> None:
        """Every rank finishes cleanly in some order — rank 0, the store's
        host, first even — and no survivor refuses: a clean ``done`` is
        never lost, and a store whose host finished is settled."""
        sim = HeartbeatSimulation(
            world=3, grace=2.0, schedule=schedule, finishes={0: 5.0, 1: 6.0, 2: 6.9}
        )
        exits = sim.run(until=30.0)
        assert {rank: refusal for rank, (_, refusal) in exits.items()} == {
            0: None,
            1: None,
            2: None,
        }

    @pytest.mark.unit
    def test_a_late_clean_finisher_is_not_refused_when_the_host_finishes_first(
        self,
    ) -> None:
        """The host keeps its store alive while a healthy peer is still writing."""
        quick = HeartbeatSimulation(
            world=2, grace=2.0, schedule=[], finishes={0: 5.0, 1: 6.5}
        )
        assert quick.run(until=30.0)[1][1] is None
        slow = HeartbeatSimulation(
            world=2, grace=2.0, schedule=[], finishes={0: 5.0, 1: 9.0}
        )
        when, refusal = slow.run(until=30.0)[1]
        assert refusal is None and when >= 9.0

    @pytest.mark.unit
    def test_the_schedule_decides_from_the_tick_before_the_first_event(self) -> None:
        """The ticks before the one preceding the first scripted event are
        rank order and read nothing of the tape (the driver's module
        docstring): a one-entry tape lands its deviation at 2.8, the tick
        before the death at 3.0, and not at 0.0."""
        sim = HeartbeatSimulation(world=2, grace=2.0, schedule=[1], deaths={1: 3.0})
        sim.run(until=20.0)
        assert sim.deciding_from == pytest.approx(2.8)
        # fourteen ticks (0.0 .. 2.6) of two picks each, every one rank order
        steady = sim.picks[:28]
        assert steady == [Pick((0, 1), 0), Pick((1,), 1)] * 14
        # the tick at 2.8: the tape's entry picks rank 1 first
        assert sim.picks[28:30] == [Pick((0, 1), 1), Pick((0,), 0)]
        # nothing scripted: rank order throughout, and no deciding tick
        assert HeartbeatSimulation(world=2, grace=2.0, schedule=7).deciding_from is None

    @pytest.mark.unit
    def test_a_rank_still_running_at_the_horizon_is_unfinished_by_name(self) -> None:
        """Nobody dies, nobody finishes: the clock runs out on both ranks and
        the driver says so, typed, instead of a missing key."""
        sim = HeartbeatSimulation(world=2, grace=2.0, schedule=[])
        with pytest.raises(Unfinished) as unfinished:
            sim.run(until=1.0)
        assert unfinished.value.ranks == (0, 1) and unfinished.value.until == 1.0
        assert [p.rank for p in sim.picks][:2] == [0, 1], "the empty tape is rank order"

    @pytest.mark.property
    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=5),
        victim=st.integers(min_value=0, max_value=4),
        # after its first beat: a rank dead before it (0.0) is unreached, the
        # timeout's case (``test_a_rank_that_never_arrives…``)
        death=st.floats(min_value=0.01, max_value=20.0),
        grace=st.floats(min_value=0.5, max_value=5.0),
        schedule=ps.schedules(),
    )
    def test_whichever_rank_dies_whenever_every_survivor_names_it_in_time(
        self, world: int, victim: int, death: float, grace: float, schedule: Schedule
    ) -> None:
        victim %= world
        sim = HeartbeatSimulation(
            world=world, grace=grace, schedule=schedule, deaths={victim: death}
        )
        exits = sim.run(until=death + 3 * grace + 5.0)
        for rank in sim.survivors(victim):
            assert rank in exits, f"rank {rank} never exited"
            _named_the_victim_or_the_host(sim, exits, rank, victim=victim, death=death)

    @pytest.mark.property
    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=4),
        victim=st.integers(min_value=0, max_value=3),
        death=st.floats(min_value=0.01, max_value=10.0),
        lag=st.floats(min_value=0.0, max_value=0.99),
        grace=st.floats(min_value=0.5, max_value=5.0),
        schedule=ps.schedules(),
    )
    def test_a_collective_that_fails_under_a_dead_peer_is_refused_by_name_in_time(
        self,
        world: int,
        victim: int,
        death: float,
        lag: float,
        grace: float,
        schedule: Schedule,
    ) -> None:
        """Linux gloo: the survivor's read fails the instant the victim's
        socket closes — ``lag`` graces after the death, anywhere inside the
        grace — before a single beat is missed. The survivor's exit must
        still be the refusal naming the victim, never the collective's own,
        and inside the documented bound of the death (the heartbeat may even
        speak first, when the failure lands late in the grace)."""
        victim %= world
        survivor = (victim + 1) % world
        failed_at = death + lag * grace
        sim = HeartbeatSimulation(
            world=world,
            grace=grace,
            schedule=schedule,
            deaths={victim: death},
            collective_failures={survivor: failed_at},
        )
        exits = sim.run(until=death + 3 * grace + 5.0)
        assert sim.own_refusals == {}, "the collective's refusal was never taken"
        for rank in sim.survivors(victim):
            _named_the_victim_or_the_host(sim, exits, rank, victim=victim, death=death)

    @pytest.mark.property
    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=4),
        failing=st.integers(min_value=0, max_value=3),
        failed_at=st.floats(min_value=0.0, max_value=10.0),
        grace=st.floats(min_value=0.5, max_value=5.0),
        schedule=ps.schedules(),
    )
    def test_a_collective_that_fails_with_every_peer_alive_is_its_own_refusal(
        self,
        world: int,
        failing: int,
        failed_at: float,
        grace: float,
        schedule: Schedule,
    ) -> None:
        """Nobody dies: the failing rank's peers each beat twice after the
        failure — within four beats: two advances strictly after it, and a
        flip in the ranks' tick order can hide every other advance for a
        beat — and the rank raises the collective's refusal, finishing with
        status 1; its peers then name *it*, exited with that status (or, for
        rank 0, as the store's host gone)."""
        failing %= world
        sim = HeartbeatSimulation(
            world=world,
            grace=grace,
            schedule=schedule,
            collective_failures={failing: failed_at},
        )
        exits = sim.run(until=failed_at + 3 * grace + 5.0)
        when, refusal = exits[failing]
        assert refusal is None, "not a lost peer's refusal"
        assert sim.own_refusals == {failing: OwnRefusal(failed_at, when)}
        assert failed_at - 1e-6 <= when <= failed_at + 4 * sim.interval + 1e-6
        for rank in sim.survivors(failing):
            later, refusal = exits[rank]
            assert refusal is not None, (rank, "exited without a refusal")
            assert refusal.lost.rank == failing
            if failing == 0:
                assert refusal.lost.why == "unreachable"
            else:
                assert refusal.lost == Lost(failing, "exited", status=1)
            assert later >= when + grace - 1e-6

    @pytest.mark.unit
    def test_a_collective_failure_at_the_moment_of_the_death_is_still_named(
        self,
    ) -> None:
        """The Linux case exactly: the victim dies at 3.0 and the survivor's
        collective fails at 3.0; the tape is rank order. The survivor names
        rank 1 as silent at the first tick past 3.0 + grace."""
        sim = HeartbeatSimulation(
            world=2,
            grace=2.0,
            schedule=[],
            deaths={1: 3.0},
            collective_failures={0: 3.0},
        )
        exits = sim.run(until=20.0)
        when, refusal = exits[0]
        assert refusal is not None and sim.own_refusals == {}
        assert refusal.lost.rank == 1 and refusal.lost.why == "silent"
        assert "rank 1 of 2 has sent no heartbeat" in str(refusal)
        assert 5.0 - sim.interval <= when <= 5.0 + sim.interval

    # -- the three cases that never ran on hardware (§3) -----------------------

    @pytest.mark.property
    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=5),
        victim=st.integers(min_value=0, max_value=4),
        stop=st.floats(min_value=0.01, max_value=20.0),
        grace=st.floats(min_value=0.5, max_value=5.0),
        schedule=ps.schedules(),
    )
    def test_a_stopped_rank_is_named_like_a_dead_one_and_never_exits(
        self, world: int, victim: int, stop: float, grace: float, schedule: Schedule
    ) -> None:
        """``SIGSTOP``: every thread stops, the heartbeat's included, so the
        survivors read a death and refuse inside the same bound — silent,
        or the store's host unreachable when rank 0 is the one stopped (the
        fake fails at once; the real tick stalls and is counted unreachable
        from its start a grace later, the same verdict at the same tick) —
        while the victim itself never exits: it is
        `HeartbeatSimulation.stopped`, never unfinished, and the
        launcher's to reap."""
        victim %= world
        sim = HeartbeatSimulation(
            world=world, grace=grace, schedule=schedule, stops={victim: stop}
        )
        exits = sim.run(until=stop + 3 * grace + 5.0)
        assert victim not in exits and set(sim.stopped) == {victim}
        assert stop <= sim.stopped[victim] < stop + sim.interval + 1e-6
        for rank in sim.survivors(victim):
            _named_the_victim_or_the_host(sim, exits, rank, victim=victim, death=stop)
            refusal = exits[rank][1]
            assert refusal is not None
            if refusal.lost.rank == victim and victim != 0:
                assert f"rank {victim} of {world} has sent no heartbeat" in str(refusal)

    @pytest.mark.property
    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=4),
        wedged=st.integers(min_value=0, max_value=3),
        wedge=st.floats(min_value=0.01, max_value=10.0),
        grace=st.floats(min_value=0.5, max_value=5.0),
        schedule=ps.schedules(),
    )
    def test_a_wedged_rank_is_named_by_nobody_and_its_peers_take_the_collectives_refusal(
        self, world: int, wedged: int, wedge: float, grace: float, schedule: Schedule
    ) -> None:
        """The hang without a death: the wedged rank's heartbeat beats on,
        so **nobody ever names it**. Its peers' collectives fail the timeout
        after it; the first to prove every peer alive takes its own
        ``CollectiveFailed`` (status 1) within four beats of the timeout —
        never before it — and so does every peer that proved the others
        alive before reading that exit, or whose hold ran out its bound
        (the grace and three beats) with nobody lost. A peer that read the
        first decider's ``done:1`` first can no longer prove it alive and,
        when that exit reaches the grace before its hold's bound does,
        names *it*, exited with status 1 (or the host unreachable): the
        cascade, a true refusal of an exit that followed the hang. The
        wedged rank then names a peer that left, a grace after the first of
        them went, so the world empties."""
        wedged %= world
        timeout = 3.0 * grace
        sim = HeartbeatSimulation(
            world=world,
            grace=grace,
            timeout=timeout,
            schedule=schedule,
            wedges={wedged: wedge},
        )
        exits = sim.run(until=wedge + timeout + 3 * grace + 5.0)
        failed_at = wedge + timeout
        peers = sim.survivors(wedged)
        first = min(exits[rank][0] for rank in peers)
        assert failed_at - 1e-6 <= first <= failed_at + 4 * sim.interval + 1e-6
        assert sim.own_refusals, "the first to decide takes the collective's refusal"
        for rank in peers:
            when, refusal = exits[rank]
            if refusal is None:
                assert sim.own_refusals[rank] == OwnRefusal(failed_at, when)
                assert failed_at - 1e-6 <= when, "never before the timeout"
                assert when <= failed_at + sim.settings.bound + sim.interval + 1e-6
                continue
            _named_a_peer_that_left(sim, exits, refusal, when, peers, wedged)
        if world == 2:
            assert exits[peers[0]][1] is None, "one peer: nobody to cascade from"
        when, refusal = exits[wedged]
        assert refusal is not None, "the wedged rank names the peers that left"
        _named_a_peer_that_left(sim, exits, refusal, when, peers, wedged)
        assert when >= first + grace - 1e-6

    @pytest.mark.property
    @_PROPERTY
    @given(
        world=st.integers(min_value=2, max_value=5),
        missing=st.integers(min_value=1, max_value=4),
        grace=st.floats(min_value=0.5, max_value=5.0),
        factor=st.floats(min_value=2.0, max_value=6.0),
        schedule=ps.schedules(),
    )
    def test_a_rank_that_never_arrives_is_named_absent_at_the_timeout_and_not_before(
        self, world: int, missing: int, grace: float, factor: float, schedule: Schedule
    ) -> None:
        """Dead before it connected: its peers never see a key of its and
        cannot tell it from a slow starter, so they wait the collective
        timeout — then every one of them names it ``never reached the
        rendezvous``, within one beat of the timeout, never before."""
        missing = 1 + (missing - 1) % (world - 1)
        timeout = factor * grace
        sim = HeartbeatSimulation(
            world=world,
            grace=grace,
            timeout=timeout,
            schedule=schedule,
            unreached=frozenset({missing}),
        )
        exits = sim.run(until=timeout + grace + 5.0)
        assert missing not in exits and set(exits) == set(sim.survivors(missing))
        for rank in sim.survivors(missing):
            when, refusal = exits[rank]
            assert refusal is not None
            assert refusal.lost.rank == missing and refusal.lost.why == "unreached"
            assert f"rank {missing} of {world} never reached the rendezvous" in str(
                refusal
            )
            assert timeout - 1e-6 <= when <= timeout + sim.interval + 1e-6

    @pytest.mark.unit
    def test_rank_zero_cannot_be_absent_here(self) -> None:
        """Without the store no heartbeat runs: the launcher's own refusal
        of an unreachable store is that path (§11), not a simulation."""
        with pytest.raises(UnsimulatedHost, match="rank 0 hosts the store"):
            HeartbeatSimulation(
                world=2, grace=1.0, schedule=[], unreached=frozenset({0})
            )

    @pytest.mark.unit
    def test_a_stopped_host_black_holes_the_store(self) -> None:
        """Rank 0 stopped at 3.0: rank 1's next tick finds the store gone
        and names the host unreachable a grace later — 5.0 here and in
        production alike, where the tick stalls against a host that never
        replies and the watch counts the store unreachable from the tick's
        start once a grace has passed (``heartbeat.STALL_S``)."""
        sim = HeartbeatSimulation(world=2, grace=2.0, schedule=[], stops={0: 3.0})
        exits = sim.run(until=20.0)
        assert sim.stopped == {0: 3.0} and set(exits) == {1}
        when, refusal = exits[1]
        assert refusal is not None and refusal.lost == Lost(0, "unreachable")
        assert when == pytest.approx(5.0)

    @pytest.mark.unit
    def test_a_wedge_scripts_the_peers_collective_failures_at_the_timeout(
        self,
    ) -> None:
        sim = HeartbeatSimulation(
            world=3, grace=1.0, timeout=4.0, schedule=[], wedges={1: 2.0}
        )
        assert sim.survivors_collective_failures() == {0: 6.0, 2: 6.0}
        assert sim.deciding_from == pytest.approx(1.9)
        scripted = HeartbeatSimulation(
            world=3,
            grace=1.0,
            timeout=4.0,
            schedule=[],
            wedges={1: 2.0},
            collective_failures={2: 3.0},
        )
        assert scripted.survivors_collective_failures() == {0: 6.0, 2: 3.0}


def _named_a_peer_that_left(
    sim: HeartbeatSimulation,
    exits: dict[int, tuple[float, RankLost | None]],
    refusal: RankLost,
    when: float,
    peers: list[int],
    wedged: int,
) -> None:
    """``refusal`` names a peer (never ``wedged``) that had exited by its
    own refusal: exited with status 1, or the host unreachable — a grace
    after that exit, within three ticks."""
    named = refusal.lost.rank
    assert named in peers and named != wedged, refusal
    if named == 0:
        assert refusal.lost.why == "unreachable"
    else:
        assert refusal.lost == Lost(named, "exited", status=1)
    left, _ = exits[named]
    assert left + sim.grace - 1e-6 <= when <= left + sim.grace + 3 * sim.interval + 1e-6


def _named_the_victim_or_the_host(
    sim: HeartbeatSimulation,
    exits: dict[int, tuple[float, RankLost | None]],
    rank: int,
    *,
    victim: int,
    death: float,
) -> None:
    """Survivor ``rank`` of ``victim``'s death (or stop) at ``death`` exited
    with a refusal naming the victim inside the documented bound — never
    before the grace less two ticks (its last beat is the tick before the
    death), never later than the grace plus three ticks — or, when the
    victim is not the host, naming the host as unreachable: rank 0 refused
    first, its exit took the store, and this rank's last look at the victim
    was a beat younger than rank 0's; then within a grace and three ticks of
    rank 0's exit. Every survivor is gone within two graces and three ticks."""
    when, refusal = exits[rank]
    assert refusal is not None, f"rank {rank} exited without a refusal"
    grace, interval = sim.grace, sim.interval
    if refusal.lost.rank == victim:
        assert refusal.lost.why == ("unreachable" if victim == 0 else "silent")
        assert when >= death + grace - 2 * interval - 1e-6
        assert when <= death + grace + 3 * interval + 1e-6
        return
    assert victim != 0 and refusal.lost == Lost(0, "unreachable"), refusal
    host_when, host_refusal = exits[0]
    assert host_refusal is not None and host_refusal.lost.rank == victim
    assert host_when <= when
    assert when <= host_when + grace + 3 * interval + 1e-6
    assert when <= death + 2 * grace + 3 * interval + 1e-6
