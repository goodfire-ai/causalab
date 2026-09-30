"""The watchdog cases' rules (``tests/_helpers/watchdog_cases.py``), held
without a process: each case's phrase is what the watchdog's own
``describe`` prints for the loss it stands for, the bounds are the
documented ones in terms of the settings, and `check` names every
way an observed world falls short — a late exit, the wrong status, a
missing or forbidden word, a traceback, a victim that exited when it
should have stayed or stayed when it should have exited.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings, strategies as st

from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    LOST_STATUS,
    Lost,
    Settings,
    describe,
)
from causalab.protocol.parallel import parse_geometry
from tests._helpers import watchdog_cases as wc
from tests._helpers.watchdog_cases import Case, Observed, check

_PROPERTY = settings(deadline=None, max_examples=30)
GEOMETRY = parse_geometry("tp=2")


def _refusal(lost: Lost, rank: int, config: Settings) -> str:
    return str(
        describe(
            lost, rank=rank, world=2, geometry=GEOMETRY, settings=config, where=None
        )
    )


def _passing(case: Case) -> Observed:
    """An observation that satisfies ``case`` on gloo, built from the
    watchdog's own words for the loss the case stands for."""
    s = case.settings
    survivor = case.survivor
    if case.mode == "wedge":
        err = (
            f"refused: [P4] at --parallel all_gather on axis 'tensor' failed on rank "
            f"{survivor} of 2 (group ranks [0, 1]): RuntimeError: Timed out — no peer "
            f"went silent for the grace after the failure, so the rank watchdog names "
            f"none; a hang without a death is bounded by {COLLECTIVE_TIMEOUT_VARIABLE}\n"
        )
        victim_err = "refused: " + _refusal(Lost(0, "unreachable"), 1, s) + "\n"
        return Observed(
            LOST_STATUS, err, s.timeout + 1.0, LOST_STATUS, victim_err, s.grace + 0.5
        )
    if case.mode == "exit-before-join":
        lost = Lost(1, "unreached", silence=s.timeout + 0.1)
    elif case.host_gone:
        lost = Lost(0, "unreachable")
    else:
        lost = Lost(case.victim, "silent", silence=s.grace)
    err = "refused: " + _refusal(lost, survivor, s) + "\n"
    lag = s.timeout + 0.5 if case.mode == "exit-before-join" else s.grace + 0.2
    return Observed(LOST_STATUS, err, lag, case.victim_status, "dying rank …\n")


@pytest.mark.unit
class TestTheTable:
    def test_the_seven_cases_by_name(self) -> None:
        assert [c.name for c in wc.CASES] == [
            "stop",
            "wedge",
            "host-stop",
            "host-exit",
            "host-kill",
            "exit-before-join",
            "exit-after-join",
        ]
        assert wc.case_named("wedge").mode == "wedge"
        with pytest.raises(KeyError, match="no watchdog case named 'hang'"):
            wc.case_named("hang")
        with pytest.raises(ValueError, match="outside a world of 2"):
            Case("bad", "exit", 2, wc.QUICK)

    def test_every_case_passes_its_own_words(self) -> None:
        for case in wc.CASES:
            assert check(case, _passing(case)) == [], case.name

    def test_every_phrase_is_the_watchdogs_own(self) -> None:
        """A case's phrase is a substring of what ``describe`` prints for the
        loss it stands for — the table cannot drift from the words."""
        s = wc.QUICK
        assert wc.case_named("stop").phrases() == (
            ("rank 1 of 2 has sent no heartbeat",),
        )
        assert "rank 1 of 2 has sent no heartbeat" in _refusal(
            Lost(1, "silent", silence=3.0), 0, s
        )
        host = "rank 0 of 2, the rendezvous store's host, is unreachable"
        for name in ("host-stop", "host-exit", "host-kill"):
            assert wc.case_named(name).phrases() == ((host,),)
        assert host in _refusal(Lost(0, "unreachable"), 1, s)
        unreached = wc.case_named("exit-before-join").phrases()
        assert unreached == (("rank 1 of 2 never reached the rendezvous",),)
        assert unreached[0][0] in _refusal(
            Lost(1, "unreached", silence=10.0), 0, wc.STARTUP
        )
        assert wc.case_named("exit-after-join").forbidden() == (
            wc.TRACEBACK,
            "waiting in",
        )
        wedge = wc.case_named("wedge")
        assert wedge.phrases("gloo") == (
            (
                "failed on rank 0 of 2",
                "so the rank watchdog names none",
                COLLECTIVE_TIMEOUT_VARIABLE,
            ),
        )
        assert wedge.phrases("nccl") == (
            (wc.NCCL_TEARDOWN, "taking the entire process down"),
        )
        assert (
            wedge.survivor_status("nccl") == -6 and wedge.survivor_status("gloo") == 1
        )
        assert wedge.forbidden("nccl") == () and wedge.forbidden("gloo") == (
            wc.TRACEBACK,
        )
        assert wc.case_named("stop").survivor_status("nccl") == 1, (
            "only the wedge differs"
        )
        assert wedge.victim_phrases() == (
            ("rank 0 of 2 exited with status 1",),
            (host,),
        )
        assert "rank 0 of 2 exited with status 1" in _refusal(
            Lost(0, "exited", status=1), 1, s
        )

    def test_the_victims_fate_per_mode(self) -> None:
        by = {c.name: c for c in wc.CASES}
        assert by["stop"].victim_status is None and not by["stop"].victim_exits
        assert by["host-stop"].victim_status is None
        assert (
            by["host-exit"].victim_status == 3
            and by["exit-after-join"].victim_status == 3
        )
        assert (
            by["host-kill"].victim_status == wc.KILLED and by["host-kill"].mode is None
        )
        assert (
            by["wedge"].victim_status == LOST_STATUS
            and by["wedge"].victim_after_survivor
        )
        assert all(c.survivor_status() == LOST_STATUS for c in wc.CASES)
        assert by["host-kill"].act.startswith("SIGKILL of rank 0")
        assert by["stop"].act == "rank 1 stop (its mark)"

    def test_the_bounds_are_the_documented_ones(self) -> None:
        q, t, u = wc.QUICK, wc.TIMED, wc.STARTUP
        slack = wc.EXIT_SLACK_S
        assert wc.case_named("stop").bound() == q.bound + slack
        assert wc.case_named("host-exit").bound() == q.bound + slack
        assert wc.case_named("host-kill").bound() == q.bound + slack
        assert wc.case_named("exit-after-join").bound() == q.bound + slack
        # a stopped host: its store never answers, the tick stalls a grace and
        # the watch counts the store unreachable from the tick's start — the
        # same bound as a dead host
        assert wc.case_named("host-stop").bound() == q.bound + slack
        # a wedge: the survivor runs ahead to its next collective before the
        # timeout's clock starts (RUN_AHEAD_S), then the hold under gloo
        assert wc.case_named("wedge").bound() == (
            t.timeout + t.bound + wc.RUN_AHEAD_S + slack
        )
        # NCCL: the watchdog's abort, four dump waits and a poll after the timeout
        assert wc.case_named("wedge").bound("nccl") == pytest.approx(
            t.timeout + 2.1 + wc.RUN_AHEAD_S + slack
        )
        assert wc.case_named("wedge").victim_bound() == t.bound + slack
        assert wc.case_named("exit-before-join").bound() == (
            u.timeout + u.interval + wc.STARTUP_SKEW_S + slack
        )
        for case in wc.CASES:
            assert case.settings.grace < case.settings.timeout


@pytest.mark.unit
class TestCheckNamesEveryShortfall:
    def _mutate(self, case: Case, **fields: object) -> list[str]:
        return check(case, dataclasses.replace(_passing(case), **fields))

    def test_a_late_survivor(self) -> None:
        case = wc.case_named("stop")
        (problem,) = self._mutate(case, lag=case.bound() + 0.01)
        assert problem.startswith("the survivor exited ") and "the bound is" in problem
        assert case.act in problem
        (before,) = self._mutate(case, lag=-1.0)
        assert "before the act" in before

    def test_the_wrong_status(self) -> None:
        (problem,) = self._mutate(wc.case_named("host-exit"), survivor_status=0)
        assert problem == "the survivor exited 0, not 1"

    def test_a_missing_or_forbidden_word_and_a_traceback(self) -> None:
        case = wc.case_named("exit-after-join")
        (problem,) = self._mutate(case, survivor_stderr="refused: something else\n")
        assert problem.startswith("the survivor's stderr names none of ")
        passing = _passing(case)
        (clause,) = self._mutate(
            case, survivor_stderr=passing.survivor_stderr + ", waiting in all_gather"
        )
        assert clause == "the survivor's stderr contains 'waiting in'"
        (trace,) = self._mutate(
            case, survivor_stderr=passing.survivor_stderr + wc.TRACEBACK + "\n"
        )
        assert trace == f"the survivor's stderr contains {wc.TRACEBACK!r}"

    def test_the_victims_fate(self) -> None:
        (exited,) = self._mutate(wc.case_named("stop"), victim_status=-9)
        assert exited == "the victim exited -9; it should never exit"
        (stayed,) = self._mutate(wc.case_named("host-exit"), victim_status=None)
        assert stayed == "the victim is still there; it should exit 3"
        wedge = wc.case_named("wedge")
        (words,) = self._mutate(wedge, victim_stderr="")
        assert words.startswith("the victim's stderr names none of ")
        # the survivor's ``done:1`` read before its store went: as good
        exited = "refused: " + _refusal(Lost(0, "exited", status=1), 1, wedge.settings)
        assert self._mutate(wedge, victim_stderr=exited) == []
        (late,) = self._mutate(wedge, victim_lag=wedge.victim_bound() + 1.0)
        assert "the victim exited" in late and "after the survivor" in late

    def test_the_wedge_under_nccl_is_the_watchdogs_teardown_and_under_gloo_our_refusal(
        self,
    ) -> None:
        """NCCL: the survivor dies ``-6`` with NCCL's words, no refusal of
        its own, within the timeout plus the abort delay — the words alone,
        the status alone, or 76.5 s (torch's 60 s sleep) each fail by name;
        gloo: the same observation fails, and ``BackendFailed``'s words with
        status 1 pass."""
        wedge = wc.case_named("wedge")
        torn = dataclasses.replace(
            _passing(wedge),
            survivor_status=-6,
            survivor_stderr=(
                f"[rank0]:[E ProcessGroupNCCL.cpp:683] [PG ID 0 PG GUID 0(default_pg) Rank 0] "
                f"{wc.NCCL_TEARDOWN}: WorkNCCL(SeqNum=7, OpType=ALLREDUCE) ran for 15053 "
                "milliseconds before timing out.\n"
                "[rank0]:[E ProcessGroupNCCL.cpp:760] To avoid data inconsistency, we are "
                "taking the entire process down.\n"
                "terminate called after throwing an instance of 'c10::DistBackendError'\n"
            ),
            lag=17.4,
        )
        assert check(wedge, torn, "nccl") == []
        assert any("exited -6, not 1" in p for p in check(wedge, torn, "gloo"))
        slow = dataclasses.replace(torn, lag=76.5)
        assert any("the bound is" in p for p in check(wedge, slow, "nccl"))
        refused_instead = dataclasses.replace(torn, survivor_status=1)
        assert check(wedge, refused_instead, "nccl") == [
            "the survivor exited 1, not -6"
        ]
        outside = dataclasses.replace(
            _passing(wedge),
            survivor_stderr=(
                "refused: [P4] at --parallel a collective outside this package's "
                "calls failed on rank 0 of 2: RuntimeError: [gloo] Timed out waiting "
                "15000ms for recv operation to complete — no peer went silent for the "
                "grace after the failure (CAUSALAB_RANK_GRACE=3), so the rank watchdog "
                "names none; a hang without a death is bounded by "
                f"{COLLECTIVE_TIMEOUT_VARIABLE}\n"
            ),
        )
        assert check(wedge, outside, "gloo") == []
        assert any("names none of" in p for p in check(wedge, outside, "nccl"))

    def test_the_record_is_the_observation_with_the_tails(self) -> None:
        observed = _passing(wc.case_named("wedge"))
        record = observed.record()
        assert record["survivor_status"] == LOST_STATUS
        assert record["lag_s"] == round(observed.lag, 3)
        assert record["victim_lag_s"] == round(observed.victim_lag or 0.0, 3)
        assert isinstance(record["survivor_stderr_tail"], str)


@pytest.mark.property
class TestBounds:
    @_PROPERTY
    @given(
        timeout=st.floats(min_value=2.0, max_value=1000.0),
        fraction=st.floats(min_value=0.05, max_value=0.9),
    )
    def test_every_bound_holds_the_heartbeats_and_orders_the_cases(
        self, timeout: float, fraction: float
    ) -> None:
        """Over any settings: the heartbeat's cases are bounded by its own
        bound plus the exit slack, a stopped host among them; the wedge and the startup case are the timeout's —
        above it, whatever the grace."""
        s = Settings(timeout=timeout, grace=timeout * fraction)
        floor = s.bound + wc.EXIT_SLACK_S
        cases = {c.name: dataclasses.replace(c, settings=s) for c in wc.CASES}
        for name in ("stop", "host-stop", "host-exit", "host-kill", "exit-after-join"):
            assert cases[name].bound() == floor, name
        assert cases["wedge"].bound() > s.timeout + s.bound
        assert cases["exit-before-join"].bound() == (
            s.timeout + s.interval + wc.STARTUP_SKEW_S + wc.EXIT_SLACK_S
        )
