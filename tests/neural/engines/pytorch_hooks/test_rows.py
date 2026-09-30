"""``rows.RowSplit`` (``docs/model_parallelism.md`` §7, §8.3): this replica's
slice of a minibatch, the weighed loss whose replica sum is the unsplit loss,
the summed gradient that is the unsplit gradient, and the agreed eval means.

``unit``: the inactive split is the identity and never reaches the
collective (a refusing one proves it — the ``points`` mode with two replicas
included); ``slice_for`` on a small minibatch, its refusal beside its twin.
``property``: over ``(rows, data)`` the slices partition every minibatch in
order with sizes differing by at most one. And the **fp64 twin** of §8.3's
band, on the loss and gradient functions alone since the loader carries no
fp64 dtype: under ``SimulatedWorld`` the replicas' weighed losses and summed
gradients equal the unsplit minibatch's to fp64 precision (``rtol=1e-12`` —
the two sums add the same terms in a different order, so bit equality is not
a property of any split reduction) and within ``1e-6`` in fp32; the
mutation the convention asks for — a loss weighed by nothing, the mean over
the replica's rows rather than the minibatch's — fails both.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.budget import LOCKSTEP_AXES
from causalab.neural.engines.pytorch_hooks.rows import WHOLE, RowSplit
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, SimulatedWorld, groups_for
from tests._helpers.refusing_collective import RefusingCollective

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: fp64: the reduction order is the only difference; fp32: §7's band on the
#: functions alone.
FP64_RTOL = 1e-12
FP32_ATOL = 1e-6

CPU = torch.device("cpu")


def _rows(data: int) -> ParallelGeometry:
    return ParallelGeometry(data=data, data_mode="rows")


class _Position(RefusingCollective):
    """A rank's place on the data axis; every collective refused."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self, axis: str) -> int:  # type: ignore[override]
        return self._rank if axis == "data" else 0

    def size(self, axis: str) -> int:  # type: ignore[override]
        return self._size if axis == "data" else 1


# --------------------------------------------------------------------------- #
# the inactive split
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestInactive:
    @pytest.mark.parametrize(
        "split",
        [
            WHOLE,
            RowSplit(RefusingCollective(), ONE),
            RowSplit(RefusingCollective(), _rows(1)),
            # the points mode over two replicas: the replicas hold different
            # points, and nothing here may reach the collective
            RowSplit(_Position(1, 2), ParallelGeometry(data=2)),
        ],
        ids=["whole", "solo", "rows-at-one", "points-at-two"],
    )
    def test_is_the_identity_and_calls_nothing(self, split: RowSplit) -> None:
        assert not split.active
        assert (split.replica, split.replicas) == (0, 1)
        # the lockstep axes alone (model, pipeline, context), never data
        assert split.budget_axes == LOCKSTEP_AXES and "data" not in LOCKSTEP_AXES
        assert split.slice_for([3, 4, 5]) == [3, 4, 5]
        assert split.slice_for([]) == []
        loss = torch.tensor(1.5)
        assert split.weigh(loss, 1, 3) is loss
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.full((2,), 2.0)
        split.reduce_gradients([parameter])
        assert torch.equal(parameter.grad, torch.full((2,), 2.0))
        assert split.agree_record([0.25, 4.0], 1, 3, CPU) == [0.25, 4.0]
        assert split.agree_means([3.0, 0.0], [2, 0], CPU) == [1.5, 0.0]

    def test_agree_means_at_world_one_is_the_local_mean_to_the_bit(self) -> None:
        values = [0.1, 0.2, 0.7]
        assert WHOLE.agree_means([sum(values)], [3], CPU) == [sum(values) / 3]

    def test_mismatched_sums_and_counts_are_refused(self) -> None:
        with pytest.raises(ValueError):
            WHOLE.agree_means([1.0], [1, 2], CPU)


# --------------------------------------------------------------------------- #
# slice_for
# --------------------------------------------------------------------------- #


def _splits(data: int) -> list[RowSplit]:
    return [RowSplit(_Position(rank, data), _rows(data)) for rank in range(data)]


@pytest.mark.unit
class TestSliceFor:
    def test_two_replicas_take_the_two_halves_the_first_longer(self) -> None:
        first, second = _splits(2)
        assert first.active and second.active
        assert (first.replica, first.replicas) == (0, 2)
        assert first.budget_axes == (*LOCKSTEP_AXES, "data")
        assert first.slice_for([10, 11, 12, 13, 14]) == [10, 11, 12]
        assert second.slice_for([10, 11, 12, 13, 14]) == [13, 14]

    def test_the_slices_keep_the_minibatchs_order_not_a_range(self) -> None:
        first, second = _splits(2)
        assert first.slice_for([7, 3, 9]) == [7, 3]
        assert second.slice_for([7, 3, 9]) == [9]

    def test_a_minibatch_smaller_than_the_replicas_is_refused_beside_its_twin(
        self,
    ) -> None:
        first, second, third = _splits(3)
        assert [s.slice_for([0, 1, 2]) for s in (first, second, third)] == [
            [0],
            [1],
            [2],
        ]
        with pytest.raises(ProtocolError) as err:
            second.slice_for([0, 1])
        assert err.value.code == "P4" and err.value.path == "--parallel.data"
        assert "minibatch of 2 rows" in str(err.value)
        assert "dp=3:rows" in str(err.value)


@pytest.mark.property
@_SETTINGS
@given(
    data=st.integers(min_value=1, max_value=8),
    rows=st.integers(min_value=1, max_value=64),
)
def test_the_slices_partition_every_minibatch_with_sizes_differing_by_at_most_one(
    data: int, rows: int
) -> None:
    indices = list(range(100, 100 + rows))
    if rows < data:
        with pytest.raises(ProtocolError):
            _splits(data)[data - 1].slice_for(indices)
        return
    slices = [split.slice_for(indices) for split in _splits(data)]
    assert [i for s in slices for i in s] == indices
    sizes = [len(s) for s in slices]
    assert all(size >= 1 for size in sizes)
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)  # the first slices are the longer


# --------------------------------------------------------------------------- #
# the loss and the gradient: the fp64 twin under SimulatedWorld
# --------------------------------------------------------------------------- #

Weigh = Callable[[RowSplit, torch.Tensor, int, int], torch.Tensor]


def _per_row(rows: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 3, generator=generator).to(dtype)


def _loss(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """A per-row loss with a non-linear dependence on the parameter."""
    return ((x @ theta).tanh() - x[:, 0]) ** 2


def _theta(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([0.3, -0.7, 1.1], dtype=dtype, requires_grad=True)


def _world(data: int, schedule: Schedule = ()) -> SimulatedWorld:
    return SimulatedWorld(groups_for(data, data=data), world=data, schedule=schedule)


def _split_program(
    x: torch.Tensor, weigh: Weigh
) -> Callable[[int, Collective], tuple[torch.Tensor, list[float], list[float]]]:
    """One replica of a rows split: its slice's weighed loss backward, the
    gradient summed, the loss record agreed, the eval means agreed."""

    def program(
        rank: int, collective: Collective
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        split = RowSplit(collective, _rows(collective.size("data")))
        assert split.active and split.replica == rank
        rows = split.slice_for(range(x.shape[0]))
        local = x[rows]
        theta = _theta(x.dtype)
        per_row = _loss(local, theta)
        weigh(split, per_row.mean(), len(rows), x.shape[0]).backward()
        split.reduce_gradients([theta])  # type: ignore[list-item]
        assert theta.grad is not None
        record = split.agree_record([float(per_row.mean())], len(rows), x.shape[0], CPU)
        means = split.agree_means(
            [float(per_row.sum()), float(local[:, 1].sum())],
            [len(rows), len(rows)],
            CPU,
        )
        return theta.grad.detach(), record, means

    return program


def _unsplit(x: torch.Tensor) -> tuple[torch.Tensor, float, list[float]]:
    theta = _theta(x.dtype)
    per_row = _loss(x, theta)
    per_row.mean().backward()
    assert theta.grad is not None
    return (
        theta.grad.detach(),
        float(per_row.mean()),
        [
            float(per_row.mean()),
            float(x[:, 1].mean()),
        ],
    )


def _assert_twin(
    x: torch.Tensor, data: int, *, rtol: float, atol: float, schedule: Schedule = ()
) -> None:
    results = _world(data, schedule).run(_split_program(x, RowSplit.weigh))
    grad, loss, means = _unsplit(x)
    for rank_grad, rank_record, rank_means in results:
        torch.testing.assert_close(rank_grad, grad, rtol=rtol, atol=atol)
        assert rank_record[0] == pytest.approx(loss, rel=rtol, abs=atol)
        for a, b in zip(rank_means, means, strict=True):
            assert a == pytest.approx(b, rel=rtol, abs=atol)
    # every replica holds the same reduced tensors: identical inputs to the step
    for rank_grad, rank_record, rank_means in results[1:]:
        assert torch.equal(rank_grad, results[0][0])
        assert rank_record == results[0][1] and rank_means == results[0][2]


@pytest.mark.numerical_unit
class TestTheTwin:
    @pytest.mark.parametrize("data", [2, 3, 4])
    @pytest.mark.parametrize("rows", [4, 5, 7, 12])
    def test_fp64_equals_the_unsplit_minibatch_to_fp64_precision(
        self, rows: int, data: int
    ) -> None:
        _assert_twin(_per_row(rows, torch.float64), data, rtol=FP64_RTOL, atol=0.0)

    @pytest.mark.parametrize("data", [2, 3, 4])
    @pytest.mark.parametrize("rows", [4, 5, 7, 12])
    def test_fp32_is_within_the_band(self, rows: int, data: int) -> None:
        _assert_twin(_per_row(rows, torch.float32), data, rtol=0.0, atol=FP32_ATOL)

    @_SETTINGS
    @given(schedule=ps.schedules())
    def test_the_twin_does_not_depend_on_the_schedule(self, schedule: Schedule) -> None:
        x = _per_row(7, torch.float64)

        def grad(schedule: Schedule) -> torch.Tensor:
            return _world(3, schedule).run(_split_program(x, RowSplit.weigh))[0][0]

        assert torch.equal(grad(schedule), grad(()))


@pytest.mark.property
@_SETTINGS
@given(
    data=st.integers(min_value=2, max_value=4),
    rows=st.integers(min_value=4, max_value=40),
    seed=st.integers(min_value=0, max_value=1000),
    schedule=ps.schedules(),
)
@example(data=2, rows=4, seed=0, schedule=[])
def test_fp64_twin_holds_over_drawn_minibatches(
    data: int, rows: int, seed: int, schedule: Schedule
) -> None:
    _assert_twin(
        _per_row(rows, torch.float64, seed),
        data,
        rtol=FP64_RTOL,
        atol=0.0,
        schedule=schedule,
    )


# --------------------------------------------------------------------------- #
# mutations
# --------------------------------------------------------------------------- #


def _unweighed(
    split: RowSplit, loss: torch.Tensor, rows: int, total: int
) -> torch.Tensor:
    """The forbidden mean: over the replica's rows, not the minibatch's."""
    del split, rows, total
    return loss


@pytest.mark.numerical_unit
class TestMutations:
    def test_a_loss_weighed_by_nothing_fails_the_fp64_twin(self) -> None:
        x = _per_row(6, torch.float64)
        (grad, _, _), *_rest = _world(2).run(_split_program(x, _unweighed))
        expected, _, _ = _unsplit(x)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(grad, expected, rtol=FP64_RTOL, atol=0.0)
        # equal halves: the summed local means are twice the mean
        torch.testing.assert_close(grad, 2 * expected, rtol=FP64_RTOL, atol=0.0)

    def test_a_replica_skipping_the_reduce_holds_only_its_share(self) -> None:
        x = _per_row(6, torch.float64)
        expected, _, _ = _unsplit(x)

        def program(rank: int, collective: Collective) -> torch.Tensor:
            split = RowSplit(collective, _rows(2))
            rows = split.slice_for(range(6))
            theta = _theta(torch.float64)
            split.weigh(_loss(x[rows], theta).mean(), len(rows), 6).backward()
            assert theta.grad is not None
            collective.barrier("data")  # the same collective on every rank
            return theta.grad.detach()

        shares = _world(2).run(program)
        for share in shares:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(share, expected, rtol=FP64_RTOL, atol=0.0)
        torch.testing.assert_close(sum(shares), expected, rtol=FP64_RTOL, atol=0.0)


@pytest.mark.unit
def test_the_split_reads_its_place_off_the_collective() -> None:
    world = _world(3)

    def program(rank: int, collective: Collective) -> tuple[int, int, bool]:
        split = RowSplit(collective, _rows(3))
        return split.replica, split.replicas, split.active

    assert world.run(program) == [(0, 3, True), (1, 3, True), (2, 3, True)]
    assert RowSplit(SOLO, _rows(1)).active is False


@pytest.mark.unit
class TestSurvivors:
    """Gradient reduction and inactive row-split edge cases."""

    def test_a_parameter_without_a_gradient_does_not_stop_the_reduction(self) -> None:
        """``reduce_gradients`` skips a gradient-less parameter and still sums
        the ones after it."""
        world = SimulatedWorld(groups_for(2, data=2), world=2, schedule=0)

        def program(
            rank: int, c: Collective
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
            split = RowSplit(c, _rows(2))
            unused = torch.nn.Parameter(torch.zeros(2))
            used = torch.nn.Parameter(torch.zeros(3))
            used.grad = torch.full((3,), float(rank + 1))
            split.reduce_gradients([unused, used])
            return unused.grad, used.grad

        for unused_grad, used_grad in world.run(program):
            assert unused_grad is None
            assert torch.equal(used_grad, torch.full((3,), 3.0))

    def test_an_inactive_split_is_replica_zero_of_one(self) -> None:
        """The documented reading of ``replica`` / ``replicas`` off an active
        split: world 1, and the ``points`` mode with two replicas alike."""
        assert (WHOLE.replicas, WHOLE.replica) == (1, 0)
        points = RowSplit(_Position(1, 2), ParallelGeometry(data=2))
        assert not points.active
        assert (points.replicas, points.replica) == (1, 0)
