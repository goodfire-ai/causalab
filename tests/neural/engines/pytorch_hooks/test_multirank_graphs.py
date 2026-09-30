"""CUDA graphs across ranks (``docs/cuda_graphs.md`` "Multi-rank execution").

Which worlds capture — tensor and expert groups and data replicas over
points, never a pipeline or context group or a rows split — and what a
graph holder does with an out-of-memory failure under each: a geometry whose
windows carry collectives (``OOMPolicy.ABORT``) ends the run by name, since
one rank falling back to the eager path while its peers replay would issue
a different collective sequence; every other geometry keeps the eager
fallback. The collectives' own capture rules are in
``tests/neural/shared/parallel/test_collective.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.budget import (
    DistributedOutOfMemory,
    OOMPolicy,
)
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    GraphPool,
    TrainingGraphs,
    make_executor,
    unsupported_reason,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.graph_cohort import (
    CohortGraphs,
    EvaluationGraphs,
    Member,
    WindowItem,
    _LayoutMismatch,  # pyright: ignore[reportPrivateUsage]
)
from causalab.neural.shared.devices import DeviceMap
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.schema import parse_document
from tests._helpers.geometries import mesh_geometries
from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda
from tests.neural.engines.pytorch_hooks.test_cuda_graphs import metadata
from tests.neural.engines.pytorch_hooks.test_train import das_doc
from tests.protocol._docs import in_order


class _Sized:
    """The questions `unsupported_reason` and `Fragments` ask a collective —
    its group size and position on each axis — answered for ``geometry`` as
    the mesh lays it out."""

    def __init__(self, geometry: ParallelGeometry) -> None:
        self._sizes = {
            "data": geometry.data,
            "pipeline": geometry.pipeline,
            "context": geometry.context,
            "model": geometry.model,
            "tensor": geometry.tensor,
            "expert": geometry.expert,
        }

    def size(self, axis: str) -> int:
        return self._sizes[axis]

    def rank(self, axis: str) -> int:
        return 0


def _world(geometry: ParallelGeometry) -> tuple[Any, Any]:
    """A bundle loaded under ``geometry`` and a collective of its sizes."""
    bundle = metadata()
    bundle.geometry = geometry
    return bundle, _Sized(geometry)


#: The refusal of a rows split: the captured step's loss, not a shape.
ROWS_REASON = (
    "CUDA graphs do not serve data parallelism over rows: the captured training "
    "step lacks each replica's share of the loss"
)


@functools.cache
def _eligible_doc() -> Any:
    return parse_document(in_order(das_doc()))


def _point(geometry: ParallelGeometry) -> PointExecutor:
    bundle, collective = _world(geometry)
    return make_executor(
        _eligible_doc(),
        bundle,
        cuda_graphs=True,
        collective=collective,
        role_rows={"base": [{"input": "a"}]},
        role_fields={"base": "input"},
        load_tensors=lambda _: {},
    )


@pytest.mark.property
class TestWhichWorldsCapture:
    @settings(max_examples=60, deadline=None)
    @given(geometry=mesh_geometries(bound=3), rows=st.booleans())
    @example(geometry=ParallelGeometry(tensor=2), rows=False)
    @example(geometry=ParallelGeometry(expert=2), rows=False)
    @example(geometry=ParallelGeometry(tensor=2, expert=4), rows=False)
    @example(geometry=ParallelGeometry(data=2), rows=False)
    @example(geometry=ParallelGeometry(data=2), rows=True)
    @example(geometry=ParallelGeometry(data=2, tensor=2), rows=True)
    @example(geometry=ParallelGeometry(pipeline=2, context=2), rows=False)
    def test_tensor_expert_and_point_replicas_capture_and_nothing_else(
        self, geometry: ParallelGeometry, rows: bool
    ) -> None:
        """Tensor and expert groups and data replicas over points capture; a
        pipeline or context group, or a data split over rows, runs eagerly
        with a reason naming it."""
        geometry = dataclasses.replace(geometry, data_mode="rows" if rows else "points")
        bundle, collective = _world(geometry)
        reason = unsupported_reason(_eligible_doc(), bundle, collective)
        refused = [a for a in ("pipeline", "context") if getattr(geometry, a) > 1]
        if refused:
            assert reason is not None
            assert all(axis in reason for axis in refused), reason
            assert reason.endswith("axes" if len(refused) == 2 else "axis"), reason
        elif rows and geometry.data > 1:
            assert reason == ROWS_REASON
        else:
            assert reason is None


@pytest.mark.unit
class TestTheExecutorOfAWorld:
    def test_a_refused_world_logs_its_reason_and_builds_the_eager_executor(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            point = _point(ParallelGeometry(pipeline=2))
        assert type(point) is PointExecutor
        assert (
            "CUDA graphs disabled: CUDA graphs run under tensor, expert and data "
            "parallelism; the collective spans more than one rank on the pipeline "
            "axis"
        ) in caplog.text

    @pytest.mark.parametrize(
        "geometry", [ParallelGeometry(tensor=2), ParallelGeometry(expert=2)]
    )
    def test_a_model_parallel_world_builds_the_graph_executor(
        self, geometry: ParallelGeometry
    ) -> None:
        point = _point(geometry)
        assert type(point) is GraphExecutor
        # the executor's whole / fragment run over the world's collective
        assert point.fragments.collective.size("model") == 2
        point.close()


# --------------------------------------------------------------------------- #
# out of memory in a graph holder
# --------------------------------------------------------------------------- #


def _executor() -> Any:
    return SimpleNamespace(
        # the bundle's placement, one device (`graph_device` reads it)
        bundle=SimpleNamespace(devices=DeviceMap.parse("cuda:0", 1)),
        reset_reads=lambda: None,
    )


def _out_of_memory(*_args: Any, **_kwargs: Any) -> None:
    raise torch.OutOfMemoryError("no memory")


POLICIES = pytest.mark.parametrize(
    "policy, aborts", [(OOMPolicy.ABORT, True), (OOMPolicy.RETRY, False)]
)


@pytest.mark.unit
class TestOutOfMemoryInAGraph:
    def test_a_graph_abort_inside_a_window_is_raised_once_by_its_own_name(
        self,
    ) -> None:
        """A holder's abort runs inside a window's body; the window's own
        abort passes it through rather than wrapping it a second time."""
        from causalab.neural.engines.pytorch_hooks.budget import (
            RowBudget,
            abort_distributed,
        )
        from causalab.neural.engines.pytorch_hooks.cuda_graphs import IN_GRAPH

        def body() -> None:
            abort_distributed(torch.OutOfMemoryError("no memory"), IN_GRAPH)

        budget = RowBudget.of(None, None, oom_policy=OOMPolicy.ABORT)
        with pytest.raises(DistributedOutOfMemory) as aborted:
            budget.run(1, body)
        assert str(aborted.value).startswith(f"out of memory {IN_GRAPH};")
        assert not isinstance(aborted.value.__cause__, DistributedOutOfMemory)

    @POLICIES
    def test_the_training_bank(
        self, monkeypatch: pytest.MonkeyPatch, policy: OOMPolicy, aborts: bool
    ) -> None:
        FakeCuda().install(monkeypatch)
        pool = GraphPool()
        pool.handle(torch.device("cuda", 0))
        bank = TrainingGraphs([], pool=pool, oom_policy=policy)
        monkeypatch.setattr(bank, "_backward", _out_of_memory)
        if aborts:
            with pytest.raises(DistributedOutOfMemory, match="captured CUDA graph"):
                bank.backward(_executor(), objective=None)
        else:
            assert bank.backward(_executor(), objective=None) is False
            assert bank.disabled
        bank.close()
        pool.close()

    @staticmethod
    def _cohort(policy: OOMPolicy) -> tuple[CohortGraphs, list[WindowItem]]:
        executor = _executor()
        members = [
            Member(
                key=key,
                executor=executor,
                stages={},
                parameters=[],
                objective_reads=(),
                pairs=1,
            )
            for key in (1, 2)
        ]
        bank = CohortGraphs(
            members, make_objective=lambda *_a, **_k: None, oom_policy=policy
        )
        window = [
            WindowItem(key=key, indices=[0], minibatch=executor, objective=None)  # pyright: ignore[reportArgumentType]
            for key in (1, 2)
        ]
        return bank, window

    @POLICIES
    def test_the_cohort_step(
        self, monkeypatch: pytest.MonkeyPatch, policy: OOMPolicy, aborts: bool
    ) -> None:
        FakeCuda().install(monkeypatch)
        bank, window = self._cohort(policy)
        monkeypatch.setattr(bank, "_capture", _out_of_memory)
        if aborts:
            with pytest.raises(DistributedOutOfMemory, match="captured CUDA graph"):
                bank.backward(window)
        else:
            assert bank.backward(window) is False
            assert bank.disabled
        bank.close()

    def test_a_layout_mismatch_keeps_the_eager_cohort_under_every_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mismatch follows from the staged rows, which every rank of a
        model group holds alike, so every rank takes the eager cohort on
        the same step: it is not a rank-local decision."""
        FakeCuda().install(monkeypatch)
        bank, window = self._cohort(OOMPolicy.ABORT)

        def mismatch(*_args: Any, **_kwargs: Any) -> None:
            raise _LayoutMismatch("mask layout")

        monkeypatch.setattr(bank, "_capture", mismatch)
        assert bank.backward(window) is False
        assert bank.disabled
        bank.close()

    @POLICIES
    def test_the_evaluation_graph(
        self, monkeypatch: pytest.MonkeyPatch, policy: OOMPolicy, aborts: bool
    ) -> None:
        from causalab.neural.engines.pytorch_hooks import graph_cohort

        FakeCuda().install(monkeypatch)
        evaluation = EvaluationGraphs(oom_policy=policy)
        evaluation.pool.handle(torch.device("cuda", 0))
        members = [(GraphExecutor.__new__(GraphExecutor), ["logits"]) for _ in range(2)]
        evaluation.layout = tuple((id(ex), tuple(reads)) for ex, reads in members)
        monkeypatch.setattr(evaluation, "_capture", _out_of_memory)
        monkeypatch.setattr(
            graph_cohort,
            "cohort_entries",
            lambda members: {
                "base": [SimpleNamespace(executor=ex) for ex, _ in members]
            },
        )
        if aborts:
            with pytest.raises(DistributedOutOfMemory, match="captured CUDA graph"):
                evaluation.forward(members)  # pyright: ignore[reportArgumentType]
        else:
            assert evaluation.forward(members) is False  # pyright: ignore[reportArgumentType]
            assert evaluation.disabled
        evaluation.close()


# --------------------------------------------------------------------------- #
# a captured cohort step is reduced like an eager one
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_captured_cohort_step_reaches_the_model_group_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replayed cohort step leaves each member's gradient on its
    parameters like an eager step, so the step's reductions follow it the
    same way: the mean over the model group, and with it the §7 gradient
    agreement check, run after a replay as after eager windows."""
    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.engines.pytorch_hooks.budget import RowBudget
    from causalab.neural.shared.parallel.collective import SOLO

    averaged: list[tuple[list[torch.nn.Parameter], Any]] = []
    monkeypatch.setattr(
        train_module,
        "average_gradients",
        lambda parameters, _collective, **kwargs: averaged.append(
            (list(parameters), kwargs.get("agreement"))
        ),
    )
    current = []
    for _ in range(2):
        parameter = torch.nn.Parameter(torch.zeros(1))
        minibatch = _Minibatch(SOLO)
        fit = SimpleNamespace(
            rows=SimpleNamespace(active=False, reduce_gradients=lambda _p: None),
            optimizer=SimpleNamespace(param_groups=[{"params": [parameter]}]),
            graph_objectives={minibatch: object()},
            slices=[[0]],
            order=[0],
            position=0,
        )
        current.append((fit, minibatch))

    class _Replayed:
        disabled = False

        def backward(self, _items: Any) -> bool:
            for fit, _ in current:
                for p in fit.optimizer.param_groups[0]["params"]:
                    p.grad = torch.ones(1)
            return True

    def refuse(*_args: Any) -> None:
        raise AssertionError("a replayed step ran the eager windows")

    monkeypatch.setattr(train_module, "_run_windows", refuse)
    train_module._run_step_windows(  # pyright: ignore[reportPrivateUsage]
        current,  # type: ignore[arg-type]
        RowBudget.of(None, None),
        None,
        cohort_graphs=_Replayed(),  # type: ignore[arg-type]
        agreement=0.0,
    )
    assert averaged == [
        (fit.optimizer.param_groups[0]["params"], 0.0) for fit, _ in current
    ]


class _Minibatch:
    """A minibatch executor's surface for the step's reductions; hashable,
    as the fit's objective table keys it."""

    def __init__(self, collective: Any) -> None:
        self.fragments = SimpleNamespace(collective=collective)
