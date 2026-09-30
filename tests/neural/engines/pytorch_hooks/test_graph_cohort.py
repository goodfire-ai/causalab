"""The fixed-layout cohort capture's torch-only pieces (``graph_cohort.py``):
padding a minibatch into its slot, the padding weights that silence the
padded rows, and the eligibility rule. The capture itself needs CUDA and is
pinned in ``tests/golden/test_graph_cohort.py``."""

from __future__ import annotations

# the capture's internals are the subject
# pyright: reportPrivateUsage=false

import dataclasses
import gc
import logging
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    graph_device,
)
from causalab.neural.engines.pytorch_hooks.graph_cohort import (
    CohortGraphs,
    EvaluationGraphs,
    Member,
    _LayoutMismatch,
    _mask_copies,
    _single_role,
    cohort_graph_reason,
    pad_rows,
    padded_indices,
    slot_weights,
    slotted_frame,
)
from causalab.neural.engines.pytorch_hooks.train import TrainingObjective, _slot_rows
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.encoding import EncodedBatch, first_real_indices
from causalab.protocol.schema import ReadRef

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
)

pytestmark = pytest.mark.unit


class TestPadding:
    def test_a_whole_minibatch_is_its_own_slot(self) -> None:
        assert padded_indices([3, 4, 5], 3) == [3, 4, 5]

    def test_a_remainder_repeats_its_last_row(self) -> None:
        assert padded_indices([6, 7], 5) == [6, 7, 7, 7, 7]

    def test_an_empty_or_oversized_minibatch_is_refused(self) -> None:
        with pytest.raises(ValueError):
            padded_indices([], 2)
        with pytest.raises(ValueError):
            padded_indices([0, 1, 2], 2)

    @given(
        real=st.integers(min_value=1, max_value=6),
        extra=st.integers(min_value=0, max_value=6),
        width=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=60, deadline=None)
    def test_pad_rows_keeps_the_real_rows_and_repeats_the_last(
        self, real: int, extra: int, width: int
    ) -> None:
        tensor = torch.randn(real, width, 3)
        padded = pad_rows(tensor, real, real + extra)
        assert padded.shape == (real + extra, width, 3)
        assert torch.equal(padded[:real], tensor)
        assert torch.equal(padded[real:], tensor[-1:].expand(extra, width, 3))
        # already the slot's size: the same tensor, not a copy
        assert pad_rows(padded, real + extra, real + extra) is padded

    def test_pad_rows_refuses_a_row_count_it_was_not_told(self) -> None:
        with pytest.raises(ValueError):
            pad_rows(torch.zeros(3, 2), 2, 4)

    def test_slot_weights_are_one_on_real_rows_and_zero_on_padding(self) -> None:
        assert slot_weights(2, 5, "cpu").tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
        assert slot_weights(4, 4, "cpu").tolist() == [1.0] * 4


class TestWeightedObjective:
    """``TrainingObjective.weight``: the padded rows contribute nothing, and
    with every weight one the objective is the plain mean."""

    @pytest.fixture
    def bundle(self):
        from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
        from causalab.neural.engines.pytorch_hooks.loading import load_model

        return load_model(TINY_LLAMA)

    def _executor(self, bundle, rows: list[int], grad: bool = True):
        raw = das_doc()
        raw["method"]["sites"]["tgt"]["layers"] = [1]
        return executor_for(
            raw,
            bundle,
            base_texts=[BASES[i] for i in rows],
            counterfactual_texts=[COUNTERFACTUALS[i] for i in rows],
            extra_columns={"label": [ANSWERS[i] for i in rows]},
            grad_enabled=grad,
        )

    def test_unit_weights_are_the_plain_mean(self, bundle) -> None:
        plain = self._executor(bundle, [0, 1, 2])
        weighted = self._executor(bundle, [0, 1, 2])
        weighted.stage_cache = plain.stage_cache  # the same parameters
        stages = {"rot": plain.stage("rot")}
        loss_plain = TrainingObjective(plain, stages)()
        loss_weighted = TrainingObjective(
            weighted, stages, weight=torch.ones(3, device=bundle.devices.head)
        )()
        torch.testing.assert_close(loss_weighted, loss_plain, rtol=1e-6, atol=1e-7)

    def test_padding_rows_carry_no_loss_and_no_gradient(self, bundle) -> None:
        real = self._executor(bundle, [0, 1])
        padded = self._executor(bundle, [0, 1, 1, 1])  # the last row repeated
        padded.stage_cache = real.stage_cache
        stages = {"rot": real.stage("rot")}
        parameter = next(p for p in stages["rot"].parameters() if p.requires_grad)

        loss_real = TrainingObjective(real, stages)()
        loss_real.backward()
        grad_real = parameter.grad.detach().clone()
        parameter.grad = None

        weight = slot_weights(2, 4, bundle.devices.head)
        loss_padded = TrainingObjective(padded, stages, weight=weight)()
        loss_padded.backward()
        grad_padded = parameter.grad.detach().clone()

        # the same rows, so the same batch shape up to the padding: on CPU in
        # fp32 the per-row values agree and the weighted mean matches
        torch.testing.assert_close(loss_padded, loss_real, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grad_padded, grad_real, rtol=1e-4, atol=1e-6)

    def test_a_zero_weight_row_changes_nothing_when_its_value_moves(
        self, bundle
    ) -> None:
        # two executors over the same rows whose padding row differs: the
        # padded row is weighted out, so the objectives agree
        a = self._executor(bundle, [0, 1, 1])
        b = self._executor(bundle, [0, 1, 2])
        b.stage_cache = a.stage_cache
        stages = {"rot": a.stage("rot")}
        weight = slot_weights(2, 3, bundle.devices.head)
        torch.testing.assert_close(
            TrainingObjective(a, stages, weight=weight)(),
            TrainingObjective(b, stages, weight=weight)(),
            rtol=1e-5,
            atol=1e-6,
        )


class TestEligibility:
    @pytest.fixture
    def bundle(self):
        from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
        from causalab.neural.engines.pytorch_hooks.loading import load_model

        return load_model(TINY_LLAMA)

    def _graph_executor(self, bundle) -> GraphExecutor:
        raw = das_doc()
        raw["method"]["sites"]["tgt"]["layers"] = [1]
        reference = executor_for(
            raw,
            bundle,
            base_texts=BASES[:2],
            counterfactual_texts=COUNTERFACTUALS[:2],
            extra_columns={"label": ANSWERS[:2]},
        )
        return GraphExecutor(
            reference.doc,
            bundle,
            role_rows=reference.role_rows,
            role_fields=reference.role_fields,
            load_tensors=reference.load_tensors,
        )

    def test_one_member_is_not_a_cohort(self, bundle) -> None:
        executor = self._graph_executor(bundle)
        assert "two members" in (cohort_graph_reason([executor], None, [2]) or "")

    def test_every_member_must_be_graph_eligible(self, bundle) -> None:
        graph = self._graph_executor(bundle)
        raw = das_doc()
        raw["method"]["sites"]["tgt"]["layers"] = [1]
        plain = executor_for(
            raw, bundle, base_texts=BASES[:2], counterfactual_texts=COUNTERFACTUALS[:2]
        )
        reason = cohort_graph_reason([graph, plain], None, [2, 2])
        assert reason is not None and "graph-eligible" in reason

    def test_off_cuda_there_is_no_graph(self, bundle) -> None:
        members = [self._graph_executor(bundle), self._graph_executor(bundle)]
        reason = cohort_graph_reason(members, None, [2, 2])
        assert reason is not None and "CUDA" in reason

    def test_an_authored_bound_below_the_layout_refuses(
        self, bundle, monkeypatch
    ) -> None:
        members = [self._graph_executor(bundle), self._graph_executor(bundle)]
        # the device check comes first off CUDA; pretend the bundle is on one
        # (the map is the placement contract, so that is the thing to fake)
        on_cuda = dataclasses.replace(
            bundle, devices=DeviceMap.parse("cuda", len(bundle.blocks))
        )
        for member in members:
            monkeypatch.setattr(member, "bundle", on_cuda)
        reason = cohort_graph_reason(members, 3, [2, 2])
        assert reason is not None and "fit_rows=3" in reason
        assert cohort_graph_reason(members, 4, [2, 2]) is None
        assert cohort_graph_reason(members, None, [2, 2]) is None

    def test_a_bundle_spanning_devices_has_no_graph(self, bundle, monkeypatch) -> None:
        members = [self._graph_executor(bundle), self._graph_executor(bundle)]
        spanning = dataclasses.replace(
            bundle, devices=DeviceMap.parse("cuda:0,cuda:1", len(bundle.blocks))
        )
        for member in members:
            monkeypatch.setattr(member, "bundle", spanning)
        reason = cohort_graph_reason(members, None, [2, 2])
        assert reason is not None and "single-device" in reason


@pytest.mark.parametrize(
    "layout,expected",
    [
        ({"base": [0, 1]}, "base"),
        ({}, None),
        ({"base": [0], "counterfactual": [1]}, None),
        ({"base": [0]}, None),
        ({"base": [0, 0, 1]}, None),
        ({"base": [0, 2]}, None),
    ],
)
def test_capture_requires_one_role_with_each_member_once(monkeypatch, layout, expected):
    from causalab.neural.engines.pytorch_hooks import graph_cohort

    executors = [object() for _ in range(3)]
    monkeypatch.setattr(
        graph_cohort,
        "cohort_entries",
        lambda members: {
            role: [SimpleNamespace(executor=executors[i]) for i in indices]
            for role, indices in layout.items()
        },
    )
    assert _single_role([(ex, []) for ex in executors[:2]]) == expected


@pytest.mark.parametrize("rows,expected", [(2, 2), (4, 4), (6, 4)])
def test_slot_rows_never_exceed_the_largest_real_minibatch(rows, expected):
    doc = SimpleNamespace(train=SimpleNamespace(batch={"pairs": 4}))
    executor = SimpleNamespace(rows_for_metrics=lambda: [None] * rows)
    assert _slot_rows(doc, executor) == expected


@pytest.mark.parametrize(
    "failure", [_LayoutMismatch("mask layout"), torch.OutOfMemoryError("OOM")]
)
def test_failed_capture_releases_storage_and_stays_eager(failure, monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones(1)
    executor = SimpleNamespace(
        bundle=SimpleNamespace(devices=DeviceMap.parse("cuda", 2))
    )
    members = [Member(i, executor, {}, [parameter], [], 1) for i in range(2)]
    bank = CohortGraphs(members, make_objective=Mock())
    bank.replay = object()
    step = Mock(side_effect=failure)
    monkeypatch.setattr(bank, "_step", step)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    assert not bank.backward([])
    assert bank.disabled
    assert bank.replay is None
    assert parameter.grad is None
    assert not bank.backward([])
    assert step.call_count == 1


def test_prepared_mask_copies_only_replace_the_members_rows():
    target = torch.zeros(5, 1, 3, 3)
    source = torch.ones(2, 1, 3, 3)
    copies = []
    _mask_copies(copies, target, source, slice(1, 3))
    assert not target.any()  # preparation does not alter captured storage
    for held, new in copies:
        held.copy_(new)
    assert torch.equal(target[1:3], source)
    assert not target[[0, 3, 4]].any()
    with pytest.raises(_LayoutMismatch):
        _mask_copies([], None, source, slice(1, 3))


def test_an_unchanged_mask_does_not_rewrite_captured_storage():
    target = torch.zeros(2, 3)
    source = torch.ones(2, 3)
    slot = SimpleNamespace(staged_masks={})
    prepared = SimpleNamespace(masks={"base": ((3, 3), [(target, source)])})
    CohortGraphs._stage_masks(slot, prepared)
    version = target._version
    CohortGraphs._stage_masks(slot, prepared)
    assert target._version == version
    prepared.masks["base"] = ((2, 3), [(target, torch.zeros_like(source))])
    CohortGraphs._stage_masks(slot, prepared)
    assert not target.any()


@pytest.mark.parametrize("remaining", [1, 2])
def test_evaluation_replays_its_slots_for_the_members_still_due(monkeypatch, remaining):
    """A captured layout serves any subset of its members — the ones still
    due after an early stop — from one replay: the due members' reads are
    copied out, a stopped member's are left alone, nothing is released. A
    set disjoint from the layout — another split's members — runs eagerly
    and leaves it be; a set that overlaps the layout with a member the
    capture holds no slot for releases it."""
    from causalab.neural.engines.pytorch_hooks import graph_cohort

    executors = [_bare_graph_executor() for _ in range(4)]
    monkeypatch.setattr(
        graph_cohort,
        "cohort_entries",
        lambda members: {"base": [SimpleNamespace(executor=ex) for ex, _ in members]},
    )
    bank = EvaluationGraphs()
    members = [(ex, ["logits"]) for ex in executors[:3]]
    bank.layout = bank._signature(members)
    bank.members = executors[:3]
    bank.groups = [{("patched", "base")}] * 3
    logits = ReadRef("logits", "patched")
    values = tuple({logits: torch.full((1,), float(i))} for i in range(3))
    replay = Mock(return_value=values)
    bank.bank = replay
    assert bank.forward(members[:remaining])
    assert bank.bank is not None and not bank.disabled
    for i, ex in enumerate(executors[:3]):
        if i < remaining:
            got = ex._read_values[logits]
            assert isinstance(got, torch.Tensor) and torch.equal(got, values[i][logits])
            assert ex._groups_run == {("patched", "base")}
        else:
            assert not ex._read_values and not ex._groups_run
    assert bank.forward(members[remaining - 1 : remaining])  # one member, its slot
    assert replay.call_count == 2
    # a set disjoint from the layout: another split's, served eagerly, the
    # capture kept and still serving its own
    assert not bank.forward([(executors[3], ["logits"])])
    assert bank.bank is replay and not bank.disabled
    assert bank.forward(members[:remaining]) and replay.call_count == 3
    # a layout member beside one the capture holds no slot for: released
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    assert not bank.forward([members[0], (executors[3], ["logits"])])
    assert bank.bank is None and bank.disabled
    assert not bank.forward(members)


def test_evaluation_says_which_members_its_capture_holds(monkeypatch):
    """``holds`` is true for a set that shares a member with a captured
    layout and false for a disjoint one, an uncaptured layout, or a released
    bank — what ``_evaluate`` asks before releasing a capture whose group the
    row bound no longer runs as one window."""
    executors = [_bare_graph_executor() for _ in range(4)]
    bank = EvaluationGraphs()
    members = [(ex, ["logits"]) for ex in executors[:3]]
    assert not bank.holds(members)  # nothing laid out
    bank.layout = bank._signature(members)
    bank.members = executors[:3]
    assert not bank.holds(members)  # laid out, not captured: nothing to release
    bank.bank = Mock()
    assert bank.holds(members)
    assert bank.holds(members[1:2])  # one slot member is enough
    assert not bank.holds([(executors[3], ["logits"])])  # another split's set
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    bank.invalidate()
    assert bank.bank is None and bank.disabled and not bank.holds(members)


def test_evaluation_lays_out_on_the_first_pass_and_captures_on_the_second(monkeypatch):
    """The first pass over a layout is eager (it warms the store) on the
    layout's prepared frame; the second captures; a different set before any
    capture is a new layout, laid out afresh."""
    from causalab.neural.engines.pytorch_hooks import graph_cohort

    executors = [_bare_graph_executor() for _ in range(3)]
    monkeypatch.setattr(
        graph_cohort,
        "cohort_entries",
        lambda members: {"base": [SimpleNamespace(executor=ex) for ex, _ in members]},
    )
    laid_out: list[tuple[int, ...]] = []
    captured: list[int] = []

    def lay_out(self, members):
        laid_out.append(tuple(id(ex) for ex, _ in members))
        self.layout = self._signature(members)
        self.members = [ex for ex, _ in members]
        self.frames = {"base": object()}

    def capture(self):
        captured.append(len(self.members))
        self.bank = Mock(return_value=tuple({} for _ in self.members))
        self.groups = [set() for _ in self.members]

    monkeypatch.setattr(EvaluationGraphs, "_lay_out", lay_out)
    monkeypatch.setattr(EvaluationGraphs, "_capture", capture)
    bank = EvaluationGraphs()
    members = [(ex, ["logits"]) for ex in executors]
    assert not bank.forward(members)  # laid out, eager
    assert bank.frames_for(members) == {"base": bank.frames["base"]}
    assert bank.frames_for(members[:2]) is None  # not the layout: per-forward
    assert not bank.forward(members[:2])  # a member stopped before any capture
    assert laid_out == [tuple(id(ex) for ex in executors)] + [
        tuple(id(ex) for ex in executors[:2])
    ]
    assert not captured
    assert bank.forward(members[:2])  # the new layout's second pass captures
    assert captured == [2]
    assert bank.forward(members[:1])  # and serves its subsets
    assert isinstance(bank.bank, Mock) and bank.bank.call_count == 2


def _bare_graph_executor() -> GraphExecutor:
    """A graph executor with only what ``EvaluationGraphs`` touches on it —
    no model, no document."""
    executor = GraphExecutor.__new__(GraphExecutor)
    executor._read_values = {}
    executor._groups_run = set()
    executor._masks = {}
    executor._position_ids = {}
    executor.bundle = SimpleNamespace(devices=DeviceMap.parse("cpu", 1))  # type: ignore[assignment]
    executor.reset_reads = lambda: None  # type: ignore[method-assign]
    return executor


class TestSlottedFrame:
    """A minibatch laid into its slot is a **row selection** of its frame
    (``slotted_frame``), so every per-row field — the first-real cache that
    position resolution and the mask signature read — follows the padded
    rows. The previous form replaced the two tensors on the frame and would
    have carried the short minibatch's cache onto the slot's mask, which the
    frame now refuses; the slot's mask signature is the cache itself, so
    preparing a minibatch reads nothing back from the device."""

    @staticmethod
    def _frame(mask: torch.Tensor) -> EncodedBatch:
        rows, width = mask.shape
        return EncodedBatch(
            texts=tuple(f"r{i}" for i in range(rows)),
            input_ids=torch.arange(rows * width).reshape(rows, width),
            attention_mask=mask,
            offset_mapping=tuple(
                tuple((j, j + 1) for j in range(width)) for _ in range(rows)
            ),
            prefix_lengths=(0,) * rows,
        )

    @given(
        pads=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=4),
        extra=st.integers(min_value=0, max_value=4),
    )
    @settings(max_examples=100, deadline=None)
    def test_the_slot_is_pad_rows_on_every_tensor_with_the_cache_in_step(
        self, pads: list[int], extra: int
    ) -> None:
        width = 6
        mask = torch.tensor([[0] * pad + [1] * (width - pad) for pad in pads])
        frame = self._frame(mask)
        real, total = len(pads), len(pads) + extra
        slotted = slotted_frame(frame, real, total)
        assert torch.equal(slotted.input_ids, pad_rows(frame.input_ids, real, total))
        assert torch.equal(
            slotted.attention_mask, pad_rows(frame.attention_mask, real, total)
        )
        assert slotted.first_reals == first_real_indices(slotted.attention_mask)
        # the mask signature the slot is staged under: the row lengths the
        # sum over the mask would give, off the cache
        assert [width - first for first in slotted.first_reals] == (
            slotted.attention_mask.sum(dim=1).tolist()
        )

    def test_a_replaced_mask_with_the_short_cache_is_what_the_frame_refuses(
        self,
    ) -> None:
        frame = self._frame(torch.tensor([[0, 1, 1], [0, 0, 1]]))
        with pytest.raises(ValueError, match="first_reals"):
            dataclasses.replace(
                frame,
                input_ids=pad_rows(frame.input_ids, 2, 4),
                attention_mask=pad_rows(frame.attention_mask, 2, 4),
            )

    def test_a_minibatch_of_another_size_is_refused(self) -> None:
        frame = self._frame(torch.tensor([[0, 1, 1], [0, 0, 1]]))
        with pytest.raises(ValueError, match="got 2 rows"):
            slotted_frame(frame, 3, 4)


class TestSharedPool:
    """A cohort's step graph and its evaluation graph are captured into the
    one pool the fit hands them (``GraphPool``), never into private pools;
    ``_fake_cuda`` stands in for the driver so the real executors, frames
    and objectives run here on the CPU fixture."""

    @pytest.fixture
    def bundle(self):
        from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
        from causalab.neural.engines.pytorch_hooks.loading import load_model

        return load_model(TINY_LLAMA)

    def _point(self, bundle, layer: int, rows: list[int]) -> GraphExecutor:
        raw = das_doc()
        raw["method"]["sites"]["tgt"]["layers"] = [layer]
        reference = executor_for(
            raw,
            bundle,
            base_texts=[BASES[i] for i in rows],
            counterfactual_texts=[COUNTERFACTUALS[i] for i in rows],
            extra_columns={"label": [ANSWERS[i] for i in rows]},
        )
        return GraphExecutor(
            reference.doc,
            bundle,
            role_rows=reference.role_rows,
            role_fields=reference.role_fields,
            load_tensors=reference.load_tensors,
        )

    def test_step_and_evaluation_graphs_share_the_pool(
        self, bundle, monkeypatch, caplog
    ):
        from causalab.neural.engines.pytorch_hooks import train
        from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool
        from causalab.neural.engines.pytorch_hooks.graph_cohort import WindowItem
        from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

        cuda = FakeCuda().install(monkeypatch)
        # the loop's minibatch executors, graph-eligible here (off CUDA the
        # factory would refuse; the capture itself is faked)
        monkeypatch.setattr(
            train,
            "make_executor",
            lambda doc, bundle, **kwargs: GraphExecutor(
                doc, bundle, **{k: v for k, v in kwargs.items() if k != "cuda_graphs"}
            ),
        )
        points = [
            self._point(bundle, 1, [0, 1, 2, 3]),
            self._point(bundle, 0, [0, 1, 2, 3]),
        ]
        fits = [
            train._prepare_fit(point.doc, point)  # pyright: ignore[reportPrivateUsage]
            for point in points
        ]
        members = [
            Member(
                key=id(fit),
                executor=fit.executor,
                stages=fit.stages,
                parameters=[p for g in fit.optimizer.param_groups for p in g["params"]],
                objective_reads=fit.objective_reads,
                pairs=_slot_rows(fit.doc, fit.executor),
            )
            for fit in fits
        ]
        window = [
            WindowItem(
                key=id(fit),
                indices=fit.batches[0],
                minibatch=fit.minibatch_executors[0],
                objective=TrainingObjective(fit.minibatch_executors[0], fit.stages),
            )
            for fit in fits
        ]
        pool = GraphPool()
        device = graph_device(bundle)
        bank = CohortGraphs(members, make_objective=TrainingObjective, pool=pool)
        assert bank.backward(window)
        assert bank.replay is not None
        assert cuda.pool_ids == [pool.handle(device)]
        for member in members:
            assert all(p.grad is not None for p in member.parameters)

        evaluators = [
            GraphExecutor(
                point.doc,
                bundle,
                role_rows={role: rows[:2] for role, rows in point.role_rows.items()},
                role_fields=point.role_fields,
                load_tensors=point.load_tensors,
                stage_cache=point.stage_cache,
            )
            for point in points
        ]
        evaluation = EvaluationGraphs(pool=pool)
        entries = [(ex, ["logits"]) for ex in evaluators]
        assert not evaluation.forward(entries)  # the first use is eager
        for ex in evaluators:
            ex.reset_reads()
        assert evaluation.forward(entries)
        assert evaluation.bank is not None
        assert cuda.pool_ids == [pool.handle(device)] * 2
        assert len(cuda.pools) == 1
        for ex in evaluators:
            assert ex.dense_value("logits").device.type == "cpu"
        evaluation.close()
        bank.close()
        assert not pool.closed  # neither holder owns the pool it was handed
        # both holders released their graphs (the fake's own record of them
        # is the last reference): in that order the pool's release finds no
        # live graph, warns about nothing and resets nothing
        graphs = [weakref.ref(graph) for graph in cuda.captures]
        cuda.captures.clear()
        gc.collect()
        assert all(ref() is None for ref in graphs)
        with caplog.at_level(logging.WARNING):
            pool.close()  # released with the fit, after both graph holders
        assert caplog.text == ""
        assert pool.handle(device) is None


@pytest.mark.parametrize("held_elsewhere", [False, True])
# the exception types, not instances: a raised instance kept alive by the
# parametrization would keep its traceback, and so the stub's graph, alive
@pytest.mark.parametrize("failure", [torch.OutOfMemoryError, _LayoutMismatch])
def test_a_cohort_fallback_releases_the_pool_only_after_an_oom_no_graph_holds(
    monkeypatch, failure, held_elsewhere
):
    """The eager cohort that takes over after an OOM needs the working set
    the step graph held: the fallback releases the pool, unless the fit's
    evaluation graph or inference replays still capture into it. A layout
    mismatch is not memory pressure: the pool stays for the evaluation graph."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool
    from causalab.neural.engines.pytorch_hooks.graph_cohort import WindowItem
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda, FakeGraph

    FakeCuda().install(monkeypatch)
    device = torch.device("cuda", 0)
    pool = GraphPool()
    pool.handle(device)
    other = FakeGraph()  # e.g. the cohort's evaluation graph
    if held_elsewhere:
        pool.captured(other)
    executor = SimpleNamespace(
        # the bundle's placement, one device (`graph_device` reads it)
        bundle=SimpleNamespace(devices=DeviceMap.parse("cuda:0", 1)),
        reset_reads=lambda: None,
    )
    members = [
        Member(
            key=key,
            executor=executor,  # pyright: ignore[reportArgumentType]
            stages={},
            parameters=[],
            objective_reads=(),
            pairs=1,
        )
        for key in (1, 2)
    ]
    bank = CohortGraphs(members, make_objective=lambda *_a, **_k: None, pool=pool)

    def out_of_memory(*_args, **_kwargs):
        # a real capture OOM: the graph is registered with the pool before
        # the capture raises, and the traceback keeps this frame — and so this
        # local — alive for as long as the handler runs
        graph = FakeGraph()
        pool.captured(graph)
        raise failure("failed")

    monkeypatch.setattr(bank, "_capture", out_of_memory)
    window = [
        WindowItem(key=key, indices=[0], minibatch=executor, objective=None)  # pyright: ignore[reportArgumentType]
        for key in (1, 2)
    ]
    assert bank.backward(window) is False
    assert bank.disabled
    out_of_memory_failure = failure is torch.OutOfMemoryError
    assert pool.closed is (out_of_memory_failure and not held_elsewhere)
    assert other.resets == 0


@pytest.mark.parametrize("held_elsewhere", [False, True])
def test_an_evaluation_oom_fallback_releases_the_pool_unless_another_graph_holds_it(
    monkeypatch, held_elsewhere
):
    """The third fallback, same contract as the training bank's and the step
    graph's: the pool goes back after the failed frame unwinds, unless the
    step graph or inference replays still hold it."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda, FakeGraph

    FakeCuda().install(monkeypatch)
    pool = GraphPool()
    pool.handle(torch.device("cuda", 0))
    other = FakeGraph()  # e.g. the cohort's step graph
    if held_elsewhere:
        pool.captured(other)
    evaluation = EvaluationGraphs(pool=pool)
    members = [(GraphExecutor.__new__(GraphExecutor), ["logits"]) for _ in range(2)]
    evaluation.layout = tuple((id(ex), tuple(reads)) for ex, reads in members)

    def out_of_memory(*_args, **_kwargs):
        graph = FakeGraph()  # a local: the traceback keeps it alive in the handler
        pool.captured(graph)
        raise torch.OutOfMemoryError("failed")

    monkeypatch.setattr(evaluation, "_capture", out_of_memory)
    # the members are bare executors: give the (real) eligibility rule their
    # entries, as test_evaluation_releases_the_graph_when_membership_changes does
    from causalab.neural.engines.pytorch_hooks import graph_cohort

    monkeypatch.setattr(
        graph_cohort,
        "cohort_entries",
        lambda members: {"base": [SimpleNamespace(executor=ex) for ex, _ in members]},
    )
    assert evaluation.forward(members) is False  # pyright: ignore[reportArgumentType]
    assert evaluation.disabled
    assert pool.closed is (not held_elsewhere)
    assert other.resets == 0


def test_a_layout_change_drops_the_evaluation_graph_but_keeps_the_pool(
    monkeypatch, caplog
):
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda, FakeGraph

    FakeCuda().install(monkeypatch)
    evaluation = EvaluationGraphs()  # a pool of its own, released with it
    evaluation.pool.handle(torch.device("cuda", 0))
    evaluation.bank = Mock()
    graph = FakeGraph()
    evaluation.pool.captured(graph)  # the captured layout's graph
    evaluation.invalidate()
    assert evaluation.bank is None and evaluation.disabled
    assert not evaluation.pool.closed  # a layout change is not the end of the fit
    assert graph.resets == 0  # and never a reset: that would break a later replay
    del graph  # the bank's replay goes with the layout; here it was this local
    with caplog.at_level(logging.WARNING):
        evaluation.close()
    assert evaluation.pool.closed
    assert caplog.text == ""  # released cleanly, no holder outlived the pool
