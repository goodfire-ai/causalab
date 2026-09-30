"""The simulator's own contract (``docs/model_parallelism.md`` §10.2, §10.5).

Every collective matches ``collective.py``'s protocol contract; a
rank-uniform program is schedule independent while a rank-dependent branch is
refused as a `Divergence` naming the rank and both call sites; a group
that never completes is a `Hang`, an early exit `Abandoned`, a
kill reported by rank — none of them a hang of the process, under any drawn
schedule; two runs at one schedule are byte-identical. The schedule itself
is held to its rule: every hand-off went to a runnable rank, never a slow one
while an eager rank could run, a tape's ``i``-th switch is its ``i``-th
entry modulo the candidates and the switches past its end are rank order, a
seed replays ``random.Random(seed).choice``, and a tape agrees with its
extensions over the common prefix. The hand-written mutations the repository
convention asks for close the file: a descending-order sum fails the
fixed-order test, a signature that ignores the call site fails the divergence
test, a tape that cycles instead of falling back to rank order and a picker
that forgets the slow ranks both fail the schedule rule.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Sequence

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.budget import RowBudget
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Abandoned,
    Divergence,
    Hang,
    LayoutError,
    MeterScriptExhausted,
    Misuse,
    RankFailed,
    RankKilled,
    Schedule,
    ScheduleError,
    SimulatedMeter,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.simulated_world import rendezvous, scheduler
from tests._helpers.simulated_world.schedule import Picker, Seeded, Taped

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_THIS = "tests/_helpers/test_simulated_world.py"

#: world 2 with both multi-member axes over the same pair
_PAIR: dict[Axis, tuple[tuple[int, ...], ...]] = {
    "tensor": ((0, 1),),
    "pipeline": ((0, 1),),
}

#: tapes over the four-rank worlds below, with room to extend
_TAPES = st.lists(st.integers(min_value=0, max_value=3), max_size=40)


def _world(schedule: Schedule = 0, **geometry: int) -> SimulatedWorld:
    """World 4, ``pipeline=2 × tensor=2`` unless told otherwise: tensor groups
    ``(0, 1), (2, 3)``, pipeline groups ``(0, 2), (1, 3)``."""
    geometry = geometry or {"tensor": 2, "pipeline": 2}
    return SimulatedWorld(groups_for(4, **geometry), world=4, schedule=schedule)


def _uniform(rank: int, c: Collective) -> tuple[list[float], list[float], int]:
    """A rank-uniform program touching three axes and four collectives."""
    own = torch.full((2,), float(rank))
    gathered = c.all_gather(own, 0, "tensor")
    summed = c.all_reduce_sum(gathered, "pipeline")
    least = c.agree_min(rank, "model")
    c.barrier("tensor")
    return gathered.tolist(), summed.tolist(), least


def _divergent(rank: int, c: Collective) -> int:
    """The same collective at two call sites, chosen by rank: §3's forbidden branch."""
    if rank == 1:
        c.barrier("tensor")
    else:
        c.barrier("tensor")
    return rank


def _assert_divergence_caught(schedule: Schedule) -> Divergence:
    try:
        _world(schedule).run(_divergent)
    except Divergence as divergence:
        return divergence
    raise AssertionError("the rank-dependent branch was not refused")


#: fp32 addends whose sequential sum depends on the order: ascending gives
#: ``((1 + 1e8) - 1e8) + 0.5 = 0.5`` (the 1 is absorbed), descending gives 1.0
_ORDER_SENSITIVE = (1.0, 1e8, -1e8, 0.5)


def _assert_fixed_order(schedule: Schedule = 0) -> None:
    world = SimulatedWorld(groups_for(4, tensor=4), world=4, schedule=schedule)

    def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
        own = torch.tensor([_ORDER_SENSITIVE[rank]], dtype=torch.float32)
        return own, c.all_reduce_sum(own, "tensor")

    expected = torch.tensor([_ORDER_SENSITIVE[0]], dtype=torch.float32)
    for value in _ORDER_SENSITIVE[1:]:
        expected = expected + torch.tensor([value], dtype=torch.float32)
    for rank, (own, total) in enumerate(world.run(program)):
        assert torch.equal(total, expected), f"rank {rank}: {total} != {expected}"
        assert total is not own
        assert own.item() == _ORDER_SENSITIVE[rank], "the argument was mutated"


def _assert_schedule_rule(world: SimulatedWorld) -> None:
    """Every recorded hand-off obeyed the schedule: a runnable rank, never a
    slow one while an eager rank could run; a forced hand-off — one eager
    candidate — takes it and reads nothing of a tape; a tape's ``i``-th
    *choice* is its ``i``-th entry modulo the eager candidates and the
    choices past its end the lowest eager rank; a seed replays
    ``random.Random(seed).choice`` at every hand-off."""
    schedule = world.schedule
    rng = random.Random(schedule) if isinstance(schedule, int) else None
    assert world.picks, "no hand-off was recorded"
    choice = 0
    for switch, pick in enumerate(world.picks):
        assert pick.runnable and list(pick.runnable) == sorted(set(pick.runnable))
        eager = [r for r in pick.runnable if r not in world.slow_ranks]
        candidates = eager or list(pick.runnable)
        where = f"switch {switch}: {pick} with slow ranks {sorted(world.slow_ranks)}"
        assert pick.rank in candidates, where
        if rng is not None:
            assert pick.rank == rng.choice(candidates), where
        elif len(candidates) == 1:
            assert pick.rank == candidates[0], f"{where}: forced, the tape untouched"
        elif choice < len(schedule):
            expected = candidates[schedule[choice] % len(candidates)]
            assert pick.rank == expected, f"{where}: choice {choice}"
            choice += 1
        else:
            assert pick.rank == candidates[0], f"{where}: past the tape, rank order"
            choice += 1


class TestCollectives:
    @pytest.mark.unit
    def test_the_rank_view_is_a_collective(self) -> None:
        assert all(_world().run(lambda rank, c: isinstance(c, Collective)))

    @pytest.mark.unit
    def test_rank_and_size_per_axis(self) -> None:
        results = _world().run(
            lambda rank, c: (
                c.rank("tensor"),
                c.size("tensor"),
                c.rank("pipeline"),
                c.size("data"),
            )
        )
        assert results == [(0, 2, 0, 1), (1, 2, 0, 1), (0, 2, 1, 1), (1, 2, 1, 1)]

    @pytest.mark.unit
    def test_all_gather_concatenates_in_rank_order(self) -> None:
        results = _world().run(
            lambda rank, c: c.all_gather(torch.full((1, 2), float(rank)), 1, "tensor")
        )
        assert results[0].tolist() == [[0.0, 0.0, 1.0, 1.0]]
        assert results[2].tolist() == [[2.0, 2.0, 3.0, 3.0]]
        assert torch.equal(results[0], results[1]) and torch.equal(
            results[2], results[3]
        )

    @pytest.mark.unit
    def test_all_reduce_sum_is_the_fixed_order_sum_equal_on_every_rank(self) -> None:
        _assert_fixed_order()

    @pytest.mark.unit
    def test_broadcast_carries_shape_and_dtype_from_the_source(self) -> None:
        def program(rank: int, c: Collective) -> torch.Tensor:
            own = torch.arange(6, dtype=torch.int64).reshape(2, 3) * (rank + 1)
            return c.broadcast(own if c.rank("pipeline") == 1 else None, 1, "pipeline")

        results = _world().run(program)
        # pipeline groups (0, 2) and (1, 3): the sources are ranks 2 and 3
        for receiver, source in ((0, 2), (1, 3), (2, 2), (3, 3)):
            got = results[receiver]
            assert got.shape == (2, 3) and got.dtype == torch.int64
            assert torch.equal(got, torch.arange(6).reshape(2, 3) * (source + 1))
        events = {(e.rank, e.shape) for e in _world_events(program)}
        assert events == {(r, (2, 3)) for r in range(4)}, "receivers learn the shape"

    @pytest.mark.unit
    def test_send_recv_point_to_point(self) -> None:
        def program(rank: int, c: Collective) -> torch.Tensor | None:
            if c.rank("pipeline") == 0:
                c.send(torch.tensor([rank, -rank], dtype=torch.int32), 1, "pipeline")
                return None
            return c.recv((2,), torch.int32, torch.device("cpu"), 0, "pipeline")

        results = _world().run(program)
        assert results[0] is None and results[1] is None
        second, third = results[2], results[3]
        assert second is not None and third is not None
        assert second.tolist() == [0, 0] and second.dtype == torch.int32
        assert third.tolist() == [1, -1]

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_pairs_in_two_groups_of_one_axis_are_two_rendezvous(
        self, schedule: Schedule
    ) -> None:
        """Ranks 0 and 1 are both local 0 of their pipeline groups and send to
        local 1; keyed by local indices alone the two pairs would share one
        rendezvous, which completes or hangs by schedule. The conformance
        suite found this against ``TorchCollective`` (``tests/neural/shared/
        parallel/collective_contract.py``); a slow rank 2 pins the order that
        hung."""

        def program(rank: int, c: Collective) -> list[int] | None:
            if c.rank("pipeline") == 0:
                c.send(torch.tensor([rank]), 1, "pipeline")
                return None
            return c.recv(
                (1,), torch.int64, torch.device("cpu"), 0, "pipeline"
            ).tolist()

        world = _world(schedule)
        world.slow(2)
        assert world.run(program) == [None, None, [0], [1]]

    @pytest.mark.unit
    def test_the_three_agreements_are_python_scalars(self) -> None:
        def program(rank: int, c: Collective) -> tuple[int, bool, int]:
            return (
                c.agree_min(10 - rank, "tensor"),
                c.agree_any(rank == 3, "tensor"),
                c.agree_sum(rank, "pipeline"),
            )

        results = _world().run(program)
        assert results == [(9, False, 2), (9, False, 4), (7, True, 2), (7, True, 4)]
        assert all(
            type(m) is int and type(a) is bool and type(s) is int for m, a, s in results
        )

    @pytest.mark.unit
    def test_barrier_holds_every_member(self) -> None:
        world = _world()
        assert world.run(lambda rank, c: c.barrier("model")) == [None] * 4
        assert [e.op for e in world.transcript] == ["barrier"] * 4
        assert sorted(e.rank for e in world.transcript) == [0, 1, 2, 3]

    @pytest.mark.unit
    def test_misuse_surfaces_as_rank_failed(self) -> None:
        def program(rank: int, c: Collective) -> torch.Tensor:
            return c.broadcast(torch.zeros(1), 0, "tensor")  # non-sources pass a tensor

        with pytest.raises(RankFailed) as failed:
            _world().run(program)
        assert isinstance(failed.value.__cause__, Misuse)
        assert failed.value.rank in (1, 3)

    @pytest.mark.unit
    def test_an_axis_outside_the_layout_is_misuse(self) -> None:
        world = SimulatedWorld(_PAIR, world=2, schedule=0)
        with pytest.raises(RankFailed) as failed:
            world.run(lambda rank, c: c.barrier("expert"))
        assert isinstance(failed.value.__cause__, Misuse)
        assert "expert" in str(failed.value)


def _world_events(
    program: Callable[[int, Collective], object],
) -> list[rendezvous.Event]:
    world = _world()
    world.run(program)
    return world.transcript


class TestLayouts:
    pytestmark = pytest.mark.unit

    def test_groups_for_is_the_mesh_of_section_2(self) -> None:
        layout = groups_for(8, pipeline=2, tensor=2, expert=4)
        assert layout["model"] == ((0, 1, 2, 3), (4, 5, 6, 7))
        assert layout["tensor"] == ((0, 1), (2, 3), (4, 5), (6, 7))
        assert layout["expert"] == ((0, 1, 2, 3), (4, 5, 6, 7))
        assert layout["pipeline"] == ((0, 4), (1, 5), (2, 6), (3, 7))
        assert layout["data"] == tuple((r,) for r in range(8))

    @pytest.mark.parametrize(
        "groups",
        [
            {"tensor": ((0, 1), (1, 2))},
            {"tensor": ((0,), (2,))},
            {"tensor": ((1, 0), (2,))},
            {"tensor": ((0, 1, 3),)},
            {"sideways": ((0, 1, 2),)},
        ],
    )
    def test_a_layout_that_is_not_a_partition_is_refused(
        self, groups: dict[Axis, tuple[tuple[int, ...], ...]]
    ) -> None:
        with pytest.raises(LayoutError):
            SimulatedWorld(groups, world=3, schedule=0)

    def test_a_geometry_that_does_not_multiply_out_is_refused(self) -> None:
        with pytest.raises(LayoutError):
            groups_for(4, tensor=3)
        with pytest.raises(LayoutError):
            groups_for(4, tensor=4, expert=3)


class TestRefusals:
    """Every refusal is detected under any schedule, never as a hang of the
    process (the simulated world's guarantee, drawn rather than enumerated)."""

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_a_rank_dependent_branch_is_a_divergence_naming_both_call_sites(
        self, schedule: Schedule
    ) -> None:
        divergence = _assert_divergence_caught(schedule)
        assert {divergence.rank, divergence.first_rank} == {0, 1}
        assert divergence.fields == ("call_site",)
        assert divergence.call_site.startswith(_THIS)
        assert divergence.first_call_site.startswith(_THIS)
        assert divergence.call_site != divergence.first_call_site
        assert divergence.call_site in str(divergence)
        assert divergence.first_call_site in str(divergence)
        assert divergence.group == (0, 1) and divergence.axis == "tensor"

    @pytest.mark.unit
    def test_shape_dtype_and_op_divergences_name_the_field(self) -> None:
        def shape(rank: int, c: Collective) -> torch.Tensor:
            return c.all_gather(torch.zeros(3 if rank == 1 else 2), 0, "tensor")

        def dtype(rank: int, c: Collective) -> torch.Tensor:
            own = torch.zeros(2, dtype=torch.float64 if rank == 1 else torch.float32)
            return c.all_reduce_sum(own, "tensor")

        def op(rank: int, c: Collective) -> object:
            own = torch.zeros(2)
            return (
                c.all_gather(own, 0, "tensor")
                if rank == 1
                else c.all_reduce_sum(own, "tensor")
            )

        for program, field in ((shape, "shape"), (dtype, "dtype"), (op, "op")):
            with pytest.raises(Divergence) as caught:
                _world().run(program)
            assert field in caught.value.fields, program.__name__
            assert {caught.value.rank, caught.value.first_rank} == {0, 1}

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_group_that_never_completes_is_a_hang(self, schedule: Schedule) -> None:
        def program(rank: int, c: Collective) -> None:
            c.barrier("tensor" if rank == 0 else "pipeline")

        started = time.monotonic()
        with pytest.raises(Hang) as hang:
            SimulatedWorld(_PAIR, world=2, schedule=schedule).run(program)
        assert time.monotonic() - started < 5.0, (
            "detected by the scheduler, not the guard"
        )
        assert not hang.value.timed_out
        assert [(w.rank, w.axis, w.op) for w in hang.value.waiting] == [
            (0, "tensor", "barrier"),
            (1, "pipeline", "barrier"),
        ]
        assert all(w.call_site.startswith(_THIS) for w in hang.value.waiting)

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_an_early_exit_is_abandoned(self, schedule: Schedule) -> None:
        def program(rank: int, c: Collective) -> int:
            if rank == 0:
                c.barrier("tensor")
            return rank

        with pytest.raises(Abandoned) as abandoned:
            SimulatedWorld(_PAIR, world=2, schedule=schedule).run(program)
        assert abandoned.value.finished == 1
        assert [(w.rank, w.op) for w in abandoned.value.waiting] == [(0, "barrier")]

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_kill_is_reported_by_rank_and_never_hangs(
        self, schedule: Schedule
    ) -> None:
        world = _world(schedule)
        world.kill(1, at_call=2)

        def program(rank: int, c: Collective) -> int:
            for _ in range(3):
                c.barrier("tensor")
            return rank

        started = time.monotonic()
        with pytest.raises(RankKilled) as killed:
            world.run(program)
        assert time.monotonic() - started < 5.0
        assert killed.value.rank == 1 and killed.value.call == 2
        assert killed.value.call_site.startswith(_THIS)
        assert all(w.rank == 0 and w.op == "barrier" for w in killed.value.abandoned)
        assert len(world.transcripts[1]) == 1, "the killed rank's transcript stops"

    @pytest.mark.unit
    def test_a_program_error_is_rank_failed_with_the_cause(self) -> None:
        def program(rank: int, c: Collective) -> None:
            c.barrier("tensor")
            if rank == 2:
                raise ValueError("rank-local trouble")

        with pytest.raises(RankFailed) as failed:
            _world().run(program)
        assert failed.value.rank == 2
        assert isinstance(failed.value.__cause__, ValueError)

    @pytest.mark.unit
    def test_the_wall_clock_guard_reports_a_hang(self) -> None:
        def program(rank: int, c: Collective) -> None:
            if rank == 0:
                time.sleep(0.8)  # stuck outside any collective
            c.barrier("tensor")

        with pytest.raises(Hang) as hang:
            SimulatedWorld(_PAIR, world=2, schedule=0, timeout=0.2).run(program)
        assert hang.value.timed_out


class TestFaults:
    @pytest.mark.unit
    def test_meter_readings_are_scripted_per_rank_and_probe(self) -> None:
        meter = SimulatedMeter({0: [(100, 1000), (50, 900)], 1: [(7, 70)]})
        ran: list[int] = []
        first = meter.for_rank(0)
        assert first.measure(lambda: ran.append(1)) == (100, 1000)
        assert first.measure(lambda: ran.append(2)) == (50, 900)
        assert meter.for_rank(1).measure(lambda: ran.append(3)) == (7, 70)
        assert ran == [1, 2, 3]
        assert meter.probes(0) == 2 and meter.probes(1) == 1
        with pytest.raises(MeterScriptExhausted):
            meter.for_rank(1).measure(lambda: None)

    @pytest.mark.unit
    def test_meter_oom_at_rank_and_step(self) -> None:
        meter = SimulatedMeter(
            {0: [(1, 10), (1, 10)], 1: [(1, 10), (1, 10)]}, oom_at=[(1, 1)]
        )
        ran: list[str] = []
        assert meter.for_rank(1).measure(lambda: ran.append("first")) == (1, 10)
        with pytest.raises(torch.OutOfMemoryError):
            meter.for_rank(1).measure(lambda: ran.append("second"))
        assert ran == ["first"], "the failing window produced nothing"
        assert meter.for_rank(0).measure(lambda: None) == (1, 10)
        assert meter.for_rank(0).measure(lambda: None) == (1, 10)
        meter.check(0, 5)
        with pytest.raises(torch.OutOfMemoryError):
            meter.check(1, 1)

    @pytest.mark.unit
    def test_a_row_budget_probed_per_rank_agrees_to_one_bound(self) -> None:
        """§3's first agreement: differing free memory at the probe, one bound."""
        meter = SimulatedMeter({0: [(400, 10_000)], 1: [(400, 4_000)]})

        def program(rank: int, c: Collective) -> tuple[int | None, int | None]:
            budget = RowBudget.of(None, meter.for_rank(rank))
            budget.run(2, lambda: None)
            assert budget.bound is not None
            return c.agree_min(budget.bound, "tensor"), budget.bound

        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=3)
        assert world.run(program) == [(18, 44), (18, 18)]

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_slow_rank_arrives_last_at_every_rendezvous(
        self, schedule: Schedule
    ) -> None:
        world = SimulatedWorld(_PAIR, world=2, schedule=schedule)
        world.slow(0)

        def program(rank: int, c: Collective) -> None:
            for _ in range(4):
                c.barrier("tensor")

        world.run(program)
        order = [e.rank for e in world.transcript]
        assert order == [1, 0] * 4


class TestDeterminism:
    pytestmark = pytest.mark.unit

    def test_two_runs_at_one_seed_are_byte_identical(self) -> None:
        first, second = _world(schedule=11), _world(schedule=11)
        assert first.run(_uniform) == second.run(_uniform)
        assert first.transcript == second.transcript
        assert first.transcripts == second.transcripts
        assert first.picks == second.picks
        assert first.run(_uniform) == second.run(_uniform), "and the world reruns"
        assert first.transcript == second.transcript

    def test_tapes_name_interleavings_a_seed_only_happens_upon(self) -> None:
        """The empty tape is rank order; a tape of ones takes the second
        runnable rank at every switch it covers — two different transcripts of
        one program, each chosen, neither drawn."""
        rank_order, second_first = _world(schedule=[]), _world(schedule=[1] * 8)
        assert rank_order.run(_uniform) == second_first.run(_uniform)
        assert rank_order.transcript != second_first.transcript
        # rank 0 runs first and parks at its all-gather, then rank 1; under
        # the ones rank 1 runs first, then — 1 parked — the next candidate, 2
        assert [p.rank for p in rank_order.picks][:2] == [0, 1]
        assert [p.rank for p in second_first.picks][:2] == [1, 2]

    def test_transcript_lines_carry_step_rank_axis_op_call_site_shape(self) -> None:
        world = _world()
        world.run(_uniform)
        for rank in range(4):
            steps = [
                (e.step, e.rank, e.axis, e.op, e.shape) for e in world.transcripts[rank]
            ]
            assert steps == [
                (1, rank, "tensor", "all_gather", (2,)),
                (2, rank, "pipeline", "all_reduce_sum", (4,)),
                (3, rank, "model", "agree_min", ()),
                (4, rank, "tensor", "barrier", None),
            ]
            assert all(e.call_site.startswith(_THIS) for e in world.transcripts[rank])
        assert sorted(world.transcript, key=lambda e: (e.rank, e.step)) == [
            e for rank in range(4) for e in world.transcripts[rank]
        ]

    @pytest.mark.parametrize("schedule", [True, [1, -1], [0, 1.5], ["0"]])
    def test_a_schedule_that_is_neither_seed_nor_tape_is_refused(
        self, schedule: object
    ) -> None:
        with pytest.raises(ScheduleError):
            _world(schedule)  # type: ignore[arg-type]

    def test_the_schedule_is_kept_normalised(self) -> None:
        assert _world(schedule=[2, 0]).schedule == (2, 0)
        assert _world(schedule=7).schedule == 7


class TestSchedules:
    """The schedule's rule (``schedule.py``), over drawn tapes and seeds."""

    pytestmark = pytest.mark.property

    def test_a_forced_switch_keeps_the_tapes_entry(self) -> None:
        """One candidate is no choice: the tape's entry waits for the next
        switch that has one, so a tape names decisions, not hand-offs."""
        taped = Taped((1,))
        assert taped.pick([5]) == 5
        assert taped.pick([3, 4]) == 4, "the entry lands on the first real choice"
        assert taped.pick([3, 4]) == 3, "past the tape: rank order"
        # the seeded form draws at every switch, forced or not
        seeded = Seeded(0)
        assert seeded.pick([5]) == 5

    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_one_schedule_replays_the_same_picks_transcripts_and_results(
        self, schedule: Schedule
    ) -> None:
        first, second = _world(schedule), _world(schedule)
        assert first.run(_uniform) == second.run(_uniform)
        assert first.picks == second.picks and first.picks
        assert first.transcript == second.transcript
        assert first.transcripts == second.transcripts

    @given(
        schedule=ps.schedules(),
        slow=st.one_of(st.none(), st.integers(min_value=0, max_value=3)),
    )
    @_SETTINGS
    def test_every_hand_off_follows_the_schedule_over_the_eager_runnable_ranks(
        self, schedule: Schedule, slow: int | None
    ) -> None:
        world = _world(schedule)
        if slow is not None:
            world.slow(slow)
        world.run(_uniform)
        _assert_schedule_rule(world)
        assert {p.rank for p in world.picks} == set(range(4)), "every rank ran"

    @given(
        tape=_TAPES,
        extension=st.lists(st.integers(min_value=0, max_value=3), min_size=1),
    )
    @_SETTINGS
    def test_a_tape_agrees_with_its_extensions_over_the_common_prefix(
        self, tape: list[int], extension: list[int]
    ) -> None:
        short, long = _world(tape), _world(tape + extension)
        assert short.run(_uniform) == long.run(_uniform)
        assert short.picks[: len(tape)] == long.picks[: len(tape)]
        assert len(short.picks) == len(long.picks)

    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_rank_uniform_program_is_schedule_independent(
        self, schedule: Schedule
    ) -> None:
        reference = _world(schedule=[])
        results = reference.run(_uniform)
        world = _world(schedule)
        assert world.run(_uniform) == results
        assert world.transcripts == reference.transcripts
        assert results[0] == ([0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 4.0, 4.0], 0)
        assert results[2] == ([2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0], 2)

    @given(schedule=ps.schedules(), layout=ps.group_layouts(6))
    @_SETTINGS
    def test_a_uniform_program_over_any_layout_is_schedule_independent(
        self, layout: ps.Layout, schedule: Schedule
    ) -> None:
        def program(rank: int, c: Collective) -> tuple[list[float], int]:
            gathered = c.all_gather(torch.tensor([float(rank)]), 0, "context")
            return gathered.tolist(), c.agree_sum(rank, "data")

        reference = SimulatedWorld(layout, world=6, schedule=[])
        expected = reference.run(program)
        world = SimulatedWorld(layout, world=6, schedule=schedule)
        assert world.run(program) == expected
        assert world.transcripts == reference.transcripts
        _assert_schedule_rule(world)


class TestProperties:
    pytestmark = pytest.mark.property

    @given(layout=ps.group_layouts(4), schedule=ps.schedules(), data=st.data())
    @_SETTINGS
    def test_all_gather_restores_the_whole(
        self, layout: ps.Layout, schedule: Schedule, data: st.DataObject
    ) -> None:
        size = len(layout["tensor"][0])
        spec = data.draw(ps.sharded_tensors(size))
        whole = spec.tensor()

        def program(rank: int, c: Collective) -> torch.Tensor:
            fragment = whole.chunk(size, dim=spec.axis)[c.rank("tensor")]
            return c.all_gather(fragment, spec.axis, "tensor")

        results = SimulatedWorld(layout, world=4, schedule=schedule).run(program)
        for group in layout["tensor"]:
            expected = torch.cat(
                [whole.chunk(size, dim=spec.axis)[i] for i in range(len(group))],
                dim=spec.axis,
            )
            assert torch.equal(expected, whole)
            for rank in group:
                assert torch.equal(results[rank], whole)

    @given(layout=ps.group_layouts(4), schedule=ps.schedules(), data=st.data())
    @_SETTINGS
    def test_all_reduce_sum_is_the_sequential_sum(
        self, layout: ps.Layout, schedule: Schedule, data: st.DataObject
    ) -> None:
        size = len(layout["expert"][0])
        base = data.draw(ps.sharded_tensors(size)).tensor()
        own = [base * (rank + 1) for rank in range(4)]

        def program(rank: int, c: Collective) -> torch.Tensor:
            return c.all_reduce_sum(own[rank], "expert")

        results = SimulatedWorld(layout, world=4, schedule=schedule).run(program)
        for group in layout["expert"]:
            expected = own[group[0]].clone()
            for rank in group[1:]:
                expected = expected + own[rank]
            for rank in group:
                assert torch.equal(results[rank], expected)
                assert torch.equal(own[rank], base * (rank + 1)), "argument untouched"

    @given(
        layout=ps.group_layouts(4),
        schedule=ps.schedules(),
        values=st.lists(st.integers(-5, 5), min_size=4, max_size=4),
    )
    @_SETTINGS
    def test_agreements_are_the_group_aggregates(
        self, layout: ps.Layout, schedule: Schedule, values: list[int]
    ) -> None:
        def program(rank: int, c: Collective) -> tuple[int, bool, int]:
            return (
                c.agree_min(values[rank], "model"),
                c.agree_any(values[rank] > 0, "model"),
                c.agree_sum(values[rank], "model"),
            )

        results = SimulatedWorld(layout, world=4, schedule=schedule).run(program)
        for group in layout["model"]:
            members = [values[rank] for rank in group]
            for rank in group:
                assert results[rank] == (
                    min(members),
                    any(v > 0 for v in members),
                    sum(members),
                )


class _Cycling(Taped):
    """The mutation: a spent tape starts over instead of falling back to rank order."""

    def pick(self, candidates: Sequence[int]) -> int:
        switch = self._switch
        self._switch = switch + 1
        if not self.tape:
            return candidates[0]
        return candidates[self.tape[switch % len(self.tape)] % len(candidates)]


class TestMutations:
    """The hand-written mutations: each names the change and shows the test go red."""

    pytestmark = pytest.mark.unit

    def test_a_descending_sum_fails_the_fixed_order_test(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def descending(tensors: list[torch.Tensor]) -> torch.Tensor:
            total = tensors[-1].clone()
            for tensor in reversed(tensors[:-1]):
                total = total + tensor
            return total

        _assert_fixed_order()
        monkeypatch.setattr(rendezvous, "reduce_sum_in_rank_order", descending)
        with pytest.raises(AssertionError, match="!="):
            _assert_fixed_order()

    def test_skipping_the_call_site_fails_the_divergence_test(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _assert_divergence_caught(0)
        without = tuple(f for f in rendezvous.COMPARED_FIELDS if f != "call_site")
        monkeypatch.setattr(rendezvous, "COMPARED_FIELDS", without)
        with pytest.raises(AssertionError, match="not refused"):
            _assert_divergence_caught(0)

    def test_a_cycling_tape_fails_the_schedule_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tape = [1, 2]  # shorter than the run: the switches past it must be rank order

        def run() -> SimulatedWorld:
            world = _world(schedule=tape)
            world.run(_uniform)
            return world

        _assert_schedule_rule(run())

        def cycling(schedule: Schedule) -> Picker:
            return (
                _Cycling(tuple(schedule))
                if not isinstance(schedule, int)
                else Seeded(schedule)
            )

        monkeypatch.setattr(scheduler, "picker", cycling)
        with pytest.raises(AssertionError, match="past the tape"):
            _assert_schedule_rule(run())

    def test_forgetting_the_slow_ranks_fails_the_schedule_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def run() -> SimulatedWorld:
            world = _world(schedule=[1, 1, 1, 1, 1, 1])
            world.slow(1)
            world.run(_uniform)
            return world

        _assert_schedule_rule(run())

        def forgetful(self: scheduler.Baton) -> int:
            runnable = sorted(self.runnable)
            rank = self._picker.pick(runnable)
            self.picks.append(scheduler.Pick(tuple(runnable), rank))
            return rank

        monkeypatch.setattr(scheduler.Baton, "_pick", forgetful)
        with pytest.raises(AssertionError, match="slow ranks"):
            _assert_schedule_rule(run())
