"""Window membership, probe failure and distributed OOM safety."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks import budget as budget_module
from causalab.neural.engines.pytorch_hooks.budget import RowBudget
from causalab.neural.engines.pytorch_hooks.train import (
    _advance_eval_budget,
    _run_windows,
)
from causalab.neural.shared.parallel import heartbeat
from causalab.neural.shared.parallel.collective import Collective
from causalab.protocol.parallel import ParallelGeometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Abandoned,
    RankFailed,
    Schedule,
    SimulatedMeter,
    SimulatedWorld,
    groups_for,
)


@dataclass
class _Rows:
    count: int

    def rows_for_metrics(self) -> list[None]:
        return [None] * self.count


def _items(sizes: list[int]) -> list[Any]:
    return [(None, _Rows(size)) for size in sizes]


@st.composite
def _rank_sizes(draw: st.DrawFn) -> list[list[int]]:
    ranks = draw(st.integers(2, 4))
    members = draw(st.integers(1, 6))
    return draw(
        st.lists(
            st.lists(st.integers(1, 8), min_size=members, max_size=members),
            min_size=ranks,
            max_size=ranks,
        )
    )


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(sizes=_rank_sizes(), bound=st.integers(1, 20), schedule=ps.schedules())
@example(sizes=[[2, 2], [1, 1]], bound=2, schedule=[])
def test_every_rank_packs_the_same_members(
    sizes: list[list[int]], bound: int, schedule: Schedule
) -> None:
    def program(rank: int, collective: Collective) -> list[list[int]]:
        budget = RowBudget.of(bound, None, collective, axes=("data",))
        pending = list(range(len(sizes[rank])))
        windows: list[list[int]] = []
        while pending:
            window, pending = budget.take(pending, lambda i: sizes[rank][i])
            assert len(window) == 1 or sum(sizes[rank][i] for i in window) <= bound
            windows.append(window)
        return windows

    ranks = len(sizes)
    results = SimulatedWorld(
        groups_for(ranks, data=ranks), world=ranks, schedule=schedule
    ).run(program)
    assert all(windows == results[0] for windows in results)
    assert [i for window in results[0] for i in window] == list(range(len(sizes[0])))


@pytest.mark.unit
def test_uneven_rows_complete_identical_budget_agreements() -> None:
    def program(rank: int, collective: Collective) -> list[int]:
        budget = RowBudget.of(2, None, collective, axes=("data",))
        windows: list[int] = []
        _run_windows(
            _items([2, 2] if rank == 0 else [1, 1]),
            budget,
            lambda window: windows.append(len(window)),
            lambda window: None,
            None,
        )
        return windows

    assert SimulatedWorld(groups_for(2, data=2), world=2, schedule=0).run(program) == [
        [1, 1],
        [1, 1],
    ]


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(failed_rank=st.integers(0, 1), schedule=ps.schedules())
def test_probe_oom_is_agreed_before_any_bound_reduction(
    failed_rank: int, schedule: Schedule
) -> None:
    meter = SimulatedMeter(
        {0: [(10, 1000)], 1: [(10, 2000)]}, oom_at={(failed_rank, 0)}
    )

    def program(rank: int, collective: Collective) -> tuple[bool, int | None]:
        budget = RowBudget.of(None, meter.for_rank(rank), collective)
        with pytest.raises(torch.OutOfMemoryError):
            _run_windows(
                _items([1]), budget, lambda window: None, lambda window: None, 1
            )
        return budget.resolved, budget.bound

    world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=schedule)
    assert world.run(program) == [(False, None), (False, None)]
    assert world.transcript[0].op == "agree_any"


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(schedule=ps.schedules(), failed_rank=st.integers(0, 1))
def test_uneven_retry_keeps_window_membership_and_bound_agreed(
    schedule: Schedule, failed_rank: int
) -> None:
    def program(rank: int, collective: Collective) -> tuple[list[int], int | None]:
        budget = RowBudget.of(None, None, collective, axes=("data",))
        budget.bound = 8
        attempts = 0
        completed: list[int] = []

        def body(window: Sequence[Any]) -> None:
            nonlocal attempts
            attempts += 1
            if rank == failed_rank and attempts == 1:
                raise torch.OutOfMemoryError("injected local allocation failure")
            completed.append(len(window))

        def abandon(window: Sequence[Any]) -> None:
            if rank != failed_rank:
                completed.pop()

        _run_windows(
            _items([2, 2, 2] if rank == 0 else [1, 1, 1]), budget, body, abandon, 1
        )
        return completed, budget.bound

    results = SimulatedWorld(groups_for(2, data=2), world=2, schedule=schedule).run(
        program
    )
    assert results[0] == results[1]
    assert sum(results[0][0]) == 3


@pytest.mark.unit
@pytest.mark.parametrize("probing", [False, True])
def test_collective_body_oom_does_not_enter_retry_agreement(probing: bool) -> None:
    def program(rank: int, collective: Collective) -> None:
        meter = SimulatedMeter({rank: [(10, 1000)]}).for_rank(rank) if probing else None
        budget = RowBudget.of(
            None, meter, collective, oom_policy=budget_module.OOMPolicy.ABORT
        )

        def body(window: Sequence[Any]) -> None:
            if rank == 1:
                raise torch.OutOfMemoryError("allocation before a sharded layer")
            collective.all_reduce_sum(torch.ones(1), "model")

        _run_windows(_items([1]), budget, body, lambda window: None, 1)

    world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=[0, 1])
    with pytest.raises(RankFailed) as failure:
        world.run(program)
    assert isinstance(failure.value.__cause__, budget_module.DistributedOutOfMemory)
    assert not any(event.op == "agree_any" for event in world.transcript)


@pytest.mark.unit
def test_unrecoverable_oom_publishes_failure_before_a_caller_can_catch_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[int] = []

    class Watch:
        def finish(self, status: int) -> None:
            statuses.append(status)

    monkeypatch.setattr(heartbeat, "running", lambda: Watch())
    budget = RowBudget.of(None, None, oom_policy=budget_module.OOMPolicy.ABORT)

    def fail() -> None:
        raise torch.OutOfMemoryError("injected")

    with pytest.raises(budget_module.DistributedOutOfMemory):
        budget.run(1, fail)
    assert statuses == [1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "geometry,abort",
    [
        (ParallelGeometry(), False),
        (ParallelGeometry(data=2, data_mode="rows"), False),
        (ParallelGeometry(tensor=2), True),
        (ParallelGeometry(expert=2), True),
        (ParallelGeometry(pipeline=2), True),
        (ParallelGeometry(context=2), True),
    ],
)
def test_loaded_geometry_selects_the_oom_policy(
    geometry: ParallelGeometry, abort: bool
) -> None:
    expected = budget_module.OOMPolicy.ABORT if abort else budget_module.OOMPolicy.RETRY
    assert budget_module.OOMPolicy.for_geometry(geometry) is expected


@pytest.mark.unit
@pytest.mark.parametrize("batch_rows", [None, 3])
def test_eval_inherits_the_distributed_oom_policy(batch_rows: int | None) -> None:
    budget = RowBudget.of(None, None, oom_policy=budget_module.OOMPolicy.ABORT)
    assert (
        _advance_eval_budget(batch_rows, budget, None).oom_policy
        is budget_module.OOMPolicy.ABORT
    )


@pytest.mark.unit
def test_local_window_membership_mutation_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RowBudget, "_min", lambda self, value: value)
    with pytest.raises(Abandoned):
        test_uneven_rows_complete_identical_budget_agreements()


@pytest.mark.unit
def test_skipping_probe_failure_agreement_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RowBudget, "out_of_memory", lambda self, failed: failed)
    meter = SimulatedMeter({0: [(10, 1000)], 1: [(10, 1000)]}, oom_at={(1, 0)})

    def program(rank: int, collective: Collective) -> None:
        budget = RowBudget.of(None, meter.for_rank(rank), collective)
        try:
            budget.run(1, lambda: None)
        except torch.OutOfMemoryError:
            return

    with pytest.raises(Abandoned):
        SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0).run(program)
