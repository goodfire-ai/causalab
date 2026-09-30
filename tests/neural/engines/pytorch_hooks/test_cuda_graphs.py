"""Eligibility and CPU fallback for the opt-in production replay path."""

from __future__ import annotations

import dataclasses
import gc
import logging
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.neural.shared.engine_router import load_engine
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    make_executor,
    unsupported_reason,
    graph_device,
)
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.featurizers import Gate
from causalab.protocol.schema import PositionSpec, parse_document
from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.test_train import (
    BASES,
    COUNTERFACTUALS,
    das_doc,
)
from tests.protocol._docs import in_order


pytestmark = pytest.mark.unit


def metadata():
    # Only the placement/config boundary is synthetic; no GPU execution here.
    model: Any = torch.nn.Linear(2, 2).requires_grad_(False).eval()
    model.config = SimpleNamespace(model_type="qwen3", _attn_implementation="eager")
    return SimpleNamespace(
        devices=DeviceMap.parse("cuda", 2), model=model, quantization=None
    )


@pytest.mark.parametrize(
    "change",
    [
        "cpu",
        "two_devices",
        "hybrid",
        "eager_experts",
        "quantized",
        "model_train",
        "model_grad",
        "sdpa",
        "sliding",
        "dims",
        "position",
        "generated",
        "head",
        "matrix_exp",
        "multiple_writes",
        "band",
    ],
)
def test_unsupported_workloads_are_not_captured(change):
    doc = parse_document(in_order(das_doc()))
    bundle = metadata()
    assert unsupported_reason(doc, bundle) is None
    if change == "cpu":
        bundle.devices = DeviceMap.parse("cpu", 2)
    elif change == "two_devices":
        # a bundle spanning devices is refused by name; the run falls back eagerly
        bundle.devices = DeviceMap.parse("cuda:0,cuda:1", 2)
    elif change == "hybrid":
        bundle.model.config.model_type = "qwen3_5_moe"
    elif change == "eager_experts":
        bundle.model.config.model_type = "qwen3_5_moe_text"
        bundle.model.config._experts_implementation = "eager"
    elif change == "quantized":
        bundle.quantization = {"scheme": "int8"}
    elif change == "model_train":
        bundle.model.train()
    elif change == "model_grad":
        bundle.model.requires_grad_(True)
    elif change == "sdpa":
        bundle.model.config._attn_implementation = "sdpa"
    elif change == "sliding":
        bundle.model.config.use_sliding_window = True
    elif change in {"dims", "position", "generated"}:
        read = doc.reads["logits"]
        fields = {
            "dims": {"dims": (0,)},
            "position": {"pos": PositionSpec(index=0)},
            "generated": {
                "pos": PositionSpec(index=-1, generated={"max_new_tokens": 1})
            },
        }[change]
        doc = dataclasses.replace(
            doc, reads=dict(doc.reads, logits=dataclasses.replace(read, **fields))
        )
    elif change == "head":
        doc = dataclasses.replace(
            doc,
            sites=dict(doc.sites, tgt=dataclasses.replace(doc.sites["tgt"], head=0)),
        )
    elif change == "matrix_exp":
        doc = dataclasses.replace(
            doc,
            featurizers=dict(
                doc.featurizers,
                rot=dataclasses.replace(
                    doc.featurizers["rot"], parametrization="matrix_exp"
                ),
            ),
        )
    elif change == "band":
        doc = dataclasses.replace(
            doc,
            sites=dict(
                doc.sites, tgt=dataclasses.replace(doc.sites["tgt"], layers=(0, 1))
            ),
        )
    elif change == "multiple_writes":
        doc = dataclasses.replace(
            doc, writes=dict(doc.writes, second=doc.writes["patch"])
        )
    assert unsupported_reason(doc, bundle) is not None


def test_qwen36_grouped_experts_are_eligible():
    bundle = metadata()
    bundle.model.config.model_type = "qwen3_5_moe_text"
    bundle.model.config._experts_implementation = "grouped_mm"
    assert unsupported_reason(parse_document(in_order(das_doc())), bundle) is None


def test_hybrid_preparation_preserves_values_and_gradients(qwen35moe_bundle):
    raw = das_doc()
    raw["model"]["key"] = qwen35moe_bundle.key
    reference = executor_for(
        raw,
        qwen35moe_bundle,
        base_texts=BASES[:2],
        counterfactual_texts=["cold mountains", COUNTERFACTUALS[1]],
        grad_enabled=True,
    )
    stage = reference.stage("rot")
    prepared = GraphExecutor(
        reference.doc,
        reference.bundle,
        role_rows=reference.role_rows,
        role_fields=reference.role_fields,
        load_tensors=reference.load_tensors,
        stage_cache=reference.stage_cache,
        grad_enabled=True,
    )
    parameter = next(stage.parameters())
    expected = reference.dense_value("logits")
    expected.sum().backward()
    assert parameter.grad is not None
    gradient = parameter.grad.clone()
    parameter.grad = None
    actual = prepared.dense_value("logits")
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


def test_opt_in_on_cpu_preserves_eager_execution(llama_bundle):
    reference = executor_for(
        das_doc(), llama_bundle, base_texts=BASES, counterfactual_texts=COUNTERFACTUALS
    )
    actual = make_executor(
        reference.doc,
        llama_bundle,
        cuda_graphs=True,
        role_rows=reference.role_rows,
        role_fields=reference.role_fields,
        load_tensors=reference.load_tensors,
        stage_cache=reference.stage_cache,
    )
    assert type(actual) is PointExecutor
    torch.testing.assert_close(
        actual.dense_value("logits"), reference.dense_value("logits"), rtol=0, atol=0
    )


def test_first_inference_use_is_eager_and_exact(qwen35moe_bundle):
    """A one-shot read must never attempt CUDA capture, even on this CPU fixture."""

    raw = das_doc()
    raw["model"]["key"] = qwen35moe_bundle.key
    reference = executor_for(
        raw,
        qwen35moe_bundle,
        base_texts=BASES[:2],
        counterfactual_texts=["cold mountains", COUNTERFACTUALS[1]],
    )
    actual = GraphExecutor(
        reference.doc,
        reference.bundle,
        role_rows=reference.role_rows,
        role_fields=reference.role_fields,
        load_tensors=reference.load_tensors,
        stage_cache=reference.stage_cache,
    )
    torch.testing.assert_close(
        actual.dense_value("logits"), reference.dense_value("logits"), rtol=0, atol=0
    )
    actual.close()


def test_engine_option_and_default():
    eager = load_engine("pytorch_hooks", device="cpu")
    graphs = load_engine("pytorch_hooks", device="cuda:1", cuda_graphs=True)
    assert isinstance(eager, PytorchHooksEngine) and isinstance(
        graphs, PytorchHooksEngine
    )
    assert not eager.cuda_graphs and graphs.cuda_graphs


def test_capture_temperature_does_not_change_checkpoint_or_public_anneal():
    gate = Gate(3)
    with torch.no_grad():
        gate.theta.copy_(torch.tensor([-0.1, 0.0, 0.2]))
    gate.temperature = 0.3
    expected = gate.soft_mask()
    gate.capture_temperature = gate.theta.new_tensor(gate.temperature)
    assert "capture_temperature" not in gate.state_dict()
    assert gate.temperature == 0.3
    gate.capture_temperature = None
    torch.testing.assert_close(gate.soft_mask(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["das", "dbm"])
@pytest.mark.parametrize("answers", ["text", "id"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_capture_objective_matches_cohort_loss(
    llama_bundle, method, answers, reduction
):
    from causalab.neural.engines.pytorch_hooks import train
    from tests.neural.engines.pytorch_hooks.test_train import ANSWERS, dbm_doc

    raw = dbm_doc() if method == "dbm" else das_doc()
    if answers == "id":
        # the cross-entropy is spelled where it is consumed (§2.10): on the
        # objective term the fit minimizes and on the save entry that tables it
        for entry in [t[1] for t in raw["method"]["train"]["objective"]] + raw[
            "method"
        ]["save"]:
            if entry.get("aggregation", {}).get("kind") == "cross_entropy":
                entry["aggregation"]["token_form"] = "id"
    point = executor_for(
        raw,
        llama_bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": [1, 2, 3, 4] if answers == "id" else ANSWERS},
    )
    assert point.doc.train is not None
    point.doc = dataclasses.replace(
        point.doc,
        train=dataclasses.replace(
            point.doc.train,
            objective=tuple(
                dataclasses.replace(term, reduce=reduction)
                if term.regularizer is not None
                else term
                for term in point.doc.train.objective
            ),
        ),
    )
    fit = train._prepare_fit(point.doc, point)
    batch = fit.minibatch_executors[0]
    actual = train.TrainingObjective(batch, fit.stages)()
    expected = train._loss(fit, batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = tuple(p for stage in fit.stages.values() for p in stage.parameters())
    a = torch.autograd.grad(actual, parameters, retain_graph=True)
    b = torch.autograd.grad(expected, parameters)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "feature",
    ["phases", "hard_concrete", "budget", "dead", "weight_anneal", "soft_accuracy"],
)
def test_new_staging_training_features_fall_back(feature):
    from causalab.protocol.schema import PhaseSpec
    from tests.neural.engines.pytorch_hooks.test_train import dbm_doc

    doc = parse_document(in_order(dbm_doc()))
    assert doc.train is not None
    if feature == "phases":
        doc = dataclasses.replace(
            doc,
            train=dataclasses.replace(
                doc.train, phases=(PhaseSpec(until={"frac": 1.0}, params=("gate",)),)
            ),
        )
    elif feature in {"hard_concrete", "budget", "dead"}:
        doc = dataclasses.replace(
            doc,
            featurizers={
                "gate": dataclasses.replace(
                    doc.featurizers["gate"],
                    **(
                        {"dead": {"freeze_after": 2}}
                        if feature == "dead"
                        else {"parametrization": feature}
                    ),
                )
            },
        )
    elif feature == "weight_anneal":
        doc = dataclasses.replace(
            doc,
            train=dataclasses.replace(
                doc.train,
                anneal={
                    "train.objective.sparsity.weight": next(
                        iter(doc.train.anneal.values())
                    )
                },
            ),
        )
    else:
        # the aggregation lives on the objective term that consumes it (§2.11)
        doc = dataclasses.replace(
            doc,
            train=dataclasses.replace(
                doc.train,
                objective=tuple(
                    dataclasses.replace(
                        term,
                        aggregation=dataclasses.replace(
                            term.aggregation, kind="soft_accuracy"
                        ),
                    )
                    if term.aggregation is not None
                    else term
                    for term in doc.train.objective
                ),
            ),
        )
    point = make_executor(
        doc,
        metadata(),
        cuda_graphs=True,
        role_rows={"base": [{"input": "a"}]},
        role_fields={"base": "input"},
        load_tensors=lambda _: {},
    )
    assert type(point) is PointExecutor


@pytest.mark.parametrize("feature", ["control", "trajectory"])
def test_stateful_training_diagnostics_use_eager(feature):
    from causalab.protocol.schema import SaveEntry

    doc = parse_document(in_order(das_doc()))
    if feature == "control":
        assert doc.train is not None
        doc = dataclasses.replace(
            doc, train=dataclasses.replace(doc.train, control={"schedule": {}})
        )
    else:
        doc = dataclasses.replace(
            doc,
            save=(*doc.save, SaveEntry("trajectory", "trajectory", kind="trajectory")),
        )
    assert unsupported_reason(doc, metadata()) == (
        "training controllers and trajectories require eager execution"
    )


def test_a_constraint_term_runs_eager_rather_than_refusing():
    """§2.11 ``constraint``: the duals step on the eager loop, so a run asked
    to capture fit graphs sends a constrained point down the eager path —
    the same decision `control` and `phases` get — instead of a P4."""
    from causalab.protocol.schema import ConstraintSpec, ObjectiveTerm

    doc = parse_document(in_order(das_doc()))
    assert doc.train is not None
    assert unsupported_reason(doc, metadata()) is None
    constrained = ObjectiveTerm(
        weight=None,
        regularizer=("l1", ("gate",)),
        name="density",
        constraint=ConstraintSpec(target=0.1, dual_lr=0.5),
    )
    doc = dataclasses.replace(
        doc,
        train=dataclasses.replace(
            doc.train, objective=(*doc.train.objective, constrained)
        ),
    )
    assert unsupported_reason(doc, metadata()) == (
        "Lagrangian constraint duals require eager execution"
    )
    # the claim is about the class, not the label: the constrained point gets an
    # executor that runs (the earlier head refused the pairing at prepare)
    point = make_executor(
        doc,
        metadata(),
        cuda_graphs=True,
        role_rows={"base": [{"input": "a"}]},
        role_fields={"base": "input"},
        load_tensors=lambda _: {},
    )
    assert type(point) is PointExecutor


def test_a_drawn_role_runs_eager_rather_than_refusing():
    """§2.2 ``draw``: the fit rebuilds its minibatch executors each epoch,
    which a captured fit graph cannot follow — so a run asked to capture
    graphs runs a drawn point eager, as `control` and a `constraint` do."""
    raw = das_doc()
    raw["data"]["counterfactual"] = {
        **raw["data"]["counterfactual"],
        "field": "counterfactual_inputs",
        "draw": {"kind": "uniform"},
    }
    doc = parse_document(in_order(raw))
    assert unsupported_reason(doc, metadata()) == (
        "a drawn role rebuilds its minibatch executors each epoch"
    )
    # the reason is the fit's, like `phases`, `control` and `constraint`: an
    # apply document reads the fixed `eval` member at one width and rebuilds
    # nothing, so it keeps capture
    assert unsupported_reason(dataclasses.replace(doc, train=None), metadata()) is None
    # the claim is about the class, not the label — and it is the premise of
    # `_Drawn.of`'s assert: a drawn point is the eager executor
    point = make_executor(
        doc,
        metadata(),
        cuda_graphs=True,
        role_rows={"base": [{"input": "a"}]},
        role_fields={"base": "input"},
        load_tensors=lambda _: {},
    )
    assert type(point) is PointExecutor


def test_row_bound_preserves_windowed_eager_executor():
    doc = parse_document(in_order(das_doc()))
    point = make_executor(
        doc,
        metadata(),
        cuda_graphs=True,
        role_rows={"base": [{"input": "a"}, {"input": "b"}]},
        role_fields={"base": "input"},
        load_tensors=lambda _: {},
        batch_rows=1,
    )
    assert type(point) is PointExecutor
    assert point.batch_rows == 1


@pytest.mark.parametrize("method", ["das", "dbm"])
@pytest.mark.parametrize("layer", [0, 1])
def test_frozen_work_preserves_gradients_and_invalidates_changed_inputs(
    llama_bundle, method, layer
):
    from causalab.neural.engines.pytorch_hooks.train import TrainingObjective
    from tests.neural.engines.pytorch_hooks.test_train import ANSWERS, dbm_doc

    raw = das_doc() if method == "das" else dbm_doc()
    raw["method"]["sites"]["tgt"]["layers"] = [layer]
    reference = executor_for(
        raw,
        llama_bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    reference.grad_enabled = True
    name = next(iter(reference.doc.featurizers))
    stages = {name: reference.stage(name)}
    actual = GraphExecutor(
        reference.doc,
        llama_bundle,
        role_rows=reference.role_rows,
        role_fields=reference.role_fields,
        load_tensors=reference.load_tensors,
        stage_cache=reference.stage_cache,
        grad_enabled=True,
    )
    objective = TrainingObjective(actual, stages)
    parameters = tuple(stages[name].parameters())
    actual.prepare_frozen(objective)
    assert actual._frozen["sources"]
    assert bool(actual._frozen["prefixes"]) == bool(layer)
    calls = []
    hook = llama_bundle.blocks[0].register_forward_hook(lambda *args: calls.append(1))
    try:
        for _ in range(2):
            with torch.no_grad():
                for parameter in parameters:
                    parameter.add_(0.01)
            actual.reset_reads()
            calls.clear()
            loss = objective()
            if layer:
                assert not calls  # Neither source nor base prefix runs again.
            grads = torch.autograd.grad(loss, parameters)
            reference.reset_reads()
            expected = TrainingObjective(reference, stages)()
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                grads,
                torch.autograd.grad(expected, parameters),
                rtol=0,
                atol=0,
            )
        frozen = actual._frozen
        for role in actual.role_rows:
            actual._batch(role).input_ids.copy_(
                actual._batch(role).input_ids.roll(1, 0)
            )
            reference._batch(role).input_ids.copy_(actual._batch(role).input_ids)
        actual.prepare_frozen(objective)
        assert actual._frozen is not frozen
        actual.reset_reads()
        reference.reset_reads()
        torch.testing.assert_close(
            objective(),
            TrainingObjective(reference, stages)(),
            rtol=0,
            atol=0,
        )
        frozen = actual._frozen
        for role in actual.role_rows:
            actual._batch(role).attention_mask[0, 0] = 0
            reference._batch(role).attention_mask.copy_(
                actual._batch(role).attention_mask
            )
        actual.prepare_frozen(objective)
        assert actual._frozen is not frozen
        actual.reset_reads()
        reference.reset_reads()
        torch.testing.assert_close(
            objective(), TrainingObjective(reference, stages)(), rtol=0, atol=0
        )
    finally:
        hook.remove()
        actual.close()
    assert not any(actual._frozen.values())


def test_campaign_prefix_plan_still_skips_blocks(llama_bundle):
    from tests.neural.engines.pytorch_hooks.test_prefix_resume import (
        _campaign,
        _executor,
        _reads,
        _block_fires,
        swap_doc,
    )

    raw = swap_doc()
    other = swap_doc()
    other["method"]["featurizers"] = {"gate": {"kind": "gate", "init": {"fill": 0.3}}}
    other["method"]["writes"]["patch"]["featurizer"] = "gate"
    _, handles, cache = _campaign([raw, other])
    eager = _executor(raw, llama_bundle, interning=None)
    source = _executor(raw, llama_bundle, interning=handles[0])
    point = GraphExecutor(
        source.doc,
        llama_bundle,
        role_rows=source.role_rows,
        role_fields=source.role_fields,
        load_tensors=source.load_tensors,
        interning=handles[0],
    )
    torch.testing.assert_close(_reads(point), _reads(eager), rtol=0, atol=0)
    eager = _executor(other, llama_bundle, interning=None)
    source = _executor(other, llama_bundle, interning=handles[1])
    point = GraphExecutor(
        source.doc,
        llama_bundle,
        role_rows=source.role_rows,
        role_fields=source.role_fields,
        load_tensors=source.load_tensors,
        interning=handles[1],
    )
    expected = _reads(eager)
    with _block_fires(llama_bundle) as fires:
        actual = _reads(point)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.resumed
    assert len(fires[0]) < len(fires[1])
    assert not point._frozen["prefixes"]


def test_decoding_request_uses_eager():
    point = make_executor(
        parse_document(in_order(das_doc())),
        metadata(),
        cuda_graphs=True,
        decoding=object(),
        role_rows={"base": [{"input": "a"}]},
        role_fields={"base": "input"},
        load_tensors=lambda _: {},
    )
    assert type(point) is PointExecutor


def test_graph_labels_are_prepared_once_per_minibatch(llama_bundle, monkeypatch):
    from causalab.neural.engines.pytorch_hooks import train
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import TrainingGraphs
    from causalab.neural.engines.pytorch_hooks.budget import RowBudget
    from tests.neural.engines.pytorch_hooks.test_train import ANSWERS

    point = executor_for(
        das_doc(),
        llama_bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    fit = train._prepare_fit(point.doc, point)
    minibatch = fit.minibatch_executors[0]
    bank = TrainingGraphs([])
    seen = []
    original = train.TrainingObjective

    def create(*args):
        seen.append(args[0])
        return original(*args)

    monkeypatch.setattr(train, "TrainingObjective", create)
    for _ in range(2):
        minibatch.reset_reads()
        fit.optimizer.zero_grad()
        train._run_step_windows(
            [(fit, minibatch)], RowBudget.of(None, None), None, graphs=bank
        )
    assert seen == [minibatch]
    bank.disabled = True
    other = fit.minibatch_executors[1]
    train._run_step_windows([(fit, other)], RowBudget.of(None, None), None, graphs=bank)
    assert seen == [minibatch]


# --------------------------------------------------------------------------- #
# one allocator pool per engine


def test_graph_pool_is_one_mempool_until_closed(monkeypatch, caplog):
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda, FakeGraph

    cuda = FakeCuda().install(monkeypatch)
    pool = GraphPool()
    assert not cuda.pools  # nothing is allocated before the first capture
    device = torch.device("cuda", 1)
    handle = pool.handle(device)
    assert pool.handle(device) == handle
    assert len(cuda.pools) == 1 and cuda.pools[0].id == handle
    # free blocks serve eager allocations on OOM: the fallback runs beside it
    assert cuda.pools[0].kwargs == {"use_on_oom": True}
    # blocks are cached per stream: every capture into the pool shares one
    assert pool.stream(device) is pool.stream(device)
    with pytest.raises(ValueError):
        pool.handle(torch.device("cuda", 0))
    # a graph still alive when the pool goes is a closing-order bug: warned
    # about, and reset first — the allocator aborts the process if the pool
    # object dies while a graph holds it
    graph = FakeGraph()
    pool.captured(graph)
    with caplog.at_level(logging.WARNING):
        pool.close()
    assert "1 live graph" in caplog.text
    assert graph.resets == 1
    with pytest.raises(RuntimeError):
        graph.replay()
    assert pool.handle(device) is None  # a closed pool hands out no handle
    with pytest.raises(ValueError):
        pool.stream(device)
    assert len(cuda.pools) == 1
    pool.close()  # idempotent


@pytest.mark.parametrize("shared", [False, True])
def test_replay_captures_into_the_given_pool(monkeypatch, shared):
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool, Replay
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    cuda = FakeCuda().install(monkeypatch)
    pool = GraphPool() if shared else None
    device = torch.device("cpu")
    replay = Replay(lambda: torch.ones(1), {}, device=device, pool=pool)
    assert cuda.pool_ids == [None if pool is None else pool.handle(device)]
    assert replay.graph is cuda.captures[0]
    # the replay holds its pool: the pool object cannot be collected first
    assert replay.pool is pool
    if pool is not None:
        pool.close()
        assert cuda.captures[0].resets == 1


def test_a_pool_closed_in_order_resets_no_graph(monkeypatch, caplog):
    """The correct order — every holder releases its graphs, then the pool —
    warns about nothing and resets nothing; only a leaked graph does."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool, Replay
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    cuda = FakeCuda().install(monkeypatch)
    pool = GraphPool()
    device = torch.device("cpu")
    replays = [
        Replay(lambda: torch.ones(1), {}, device=device, pool=pool) for _ in range(3)
    ]
    graphs = [weakref.ref(replay.graph) for replay in replays]
    del replays  # the holders release their graphs ...
    cuda.captures.clear()  # ... and the fake's own record of them
    gc.collect()
    assert all(ref() is None for ref in graphs)  # nothing outlived its holder
    with caplog.at_level(logging.WARNING):
        pool.close()
    assert caplog.text == ""


@pytest.mark.parametrize("held_elsewhere", [False, True])
def test_oom_fallback_releases_the_pool_unless_another_graph_holds_it(
    monkeypatch, held_elsewhere
):
    """The eager remainder of a fit needs the working set the graphs held:
    the bank's OOM fallback releases the pool — unless the fit's inference
    replays still capture into it, which the fallback must not reset."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
        GraphPool,
        TrainingGraphs,
    )
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda, FakeGraph

    FakeCuda().install(monkeypatch)
    pool = GraphPool()
    device = torch.device("cuda", 0)
    pool.handle(device)
    other = FakeGraph()  # e.g. the fit's held-out inference replay
    if held_elsewhere:
        pool.captured(other)
    bank = TrainingGraphs([], pool=pool)

    def out_of_memory(*_args, **_kwargs):
        # what a real capture OOM looks like: Replay registered its graph
        # with the pool, then the capture raised — and the traceback keeps
        # this frame, and so this local, alive for as long as the handler runs
        graph = FakeGraph()
        pool.captured(graph)
        raise torch.OutOfMemoryError("no memory")

    monkeypatch.setattr(bank, "_backward", out_of_memory)
    executor = SimpleNamespace(
        # the bundle's placement, one device (`graph_device` reads it)
        bundle=SimpleNamespace(devices=DeviceMap.parse("cuda:0", 1)),
        reset_reads=lambda: None,
    )
    assert bank.backward(executor, objective=None) is False
    assert bank.disabled
    assert pool.closed is (not held_elsewhere)
    assert other.resets == 0
    assert bool(pool.handle(device)) is held_elsewhere


def test_a_bank_that_opened_its_own_pool_releases_it_on_close(monkeypatch):
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
        GraphPool,
        TrainingGraphs,
    )
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    FakeCuda().install(monkeypatch)
    device = torch.device("cuda", 0)
    own = TrainingGraphs([])
    own.pool.handle(device)
    own.close()
    assert own.pool.closed  # nobody else could have released it
    pool = GraphPool()
    pool.handle(device)
    TrainingGraphs([], pool=pool).close()
    assert not pool.closed  # the fit's owner closes it after every holder
    pool.close()


def _graph_minibatches(bundle, texts, counterfactuals, answers):
    """A point's stage cache and one grad-enabled GraphExecutor per row —
    the fit's minibatches at ``pairs=1``, each its own shape/mask bucket
    when the rows' token counts differ."""
    reference = executor_for(
        das_doc(),
        bundle,
        base_texts=texts,
        counterfactual_texts=counterfactuals,
        extra_columns={"label": answers},
    )
    reference.grad_enabled = True
    stages = {"rot": reference.stage("rot")}
    minibatches = [
        GraphExecutor(
            reference.doc,
            bundle,
            role_rows={
                role: rows[i : i + 1] for role, rows in reference.role_rows.items()
            },
            role_fields=reference.role_fields,
            load_tensors=reference.load_tensors,
            stage_cache=reference.stage_cache,
            grad_enabled=True,
        )
        for i in range(len(texts))
    ]
    return reference, stages, minibatches


def test_training_buckets_share_the_fits_pool_without_a_cap(llama_bundle, monkeypatch):
    """Three minibatches of three distinct padded lengths are three buckets,
    every one captured into the fit's one pool: no second-pool memory guard,
    no cap at two, and each captured step's gradients are the eager ones."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
        GraphPool,
        TrainingGraphs,
    )
    from causalab.neural.engines.pytorch_hooks.train import TrainingObjective
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    cuda = FakeCuda().install(monkeypatch)
    texts = ["the fox", BASES[0], BASES[1] + " under a bright red autumn moon"]
    counterfactuals = ["cold hills", COUNTERFACTUALS[0], COUNTERFACTUALS[1]]
    from tests.neural.engines.pytorch_hooks.test_train import ANSWERS

    _, stages, minibatches = _graph_minibatches(
        llama_bundle, texts, counterfactuals, ANSWERS[:3]
    )
    parameters = list(stages["rot"].parameters())
    bank = TrainingGraphs(parameters, pool=GraphPool())  # the fit's, as in train.py
    device = graph_device(llama_bundle)
    for minibatch in minibatches:
        objective = TrainingObjective(minibatch, stages)
        expected = torch.autograd.grad(objective(), parameters)
        minibatch.reset_reads()
        for parameter in parameters:
            parameter.grad = None
        assert bank.backward(minibatch, objective)
        torch.testing.assert_close(
            [p.grad for p in parameters], list(expected), rtol=0, atol=0
        )
    assert len(bank.buckets) == 3
    assert not bank.disabled
    handle = bank.pool.handle(device)
    assert cuda.pool_ids == [handle] * 3
    assert len(cuda.pools) == 1
    # one capture stream, or the allocator would cache each graph's blocks apart
    assert len({id(graph.stream) for graph in cuda.captures}) == 1
    assert cuda.captures[0].stream is bank.pool.stream(device)
    # a repeated minibatch replays its bucket rather than capturing again
    assert bank.backward(minibatches[0], TrainingObjective(minibatches[0], stages))
    assert len(cuda.captures) == 3
    assert cuda.captures[0].replays == 2
    bank.close()
    assert not bank.buckets
    assert all(p.grad is None for p in parameters)
    # the pool outlives the bank — the fit's inference replays share it —
    # until the fit's owner closes it
    assert bank.pool.handle(device) == handle
    bank.pool.close()
    assert bank.pool.handle(device) is None


def test_a_fit_graph_cache_keeps_the_banks_pool_across_compatible_fits(llama_bundle):
    from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache
    from tests.neural.engines.pytorch_hooks.test_graph_reuse import executor

    cache = FitGraphCache()
    first, second = executor(llama_bundle), executor(llama_bundle, seed=1)
    bank = cache.begin(first, list(first.stage("rot").parameters()))
    pool = bank.pool
    assert cache.pool is pool  # the cache owns it, not the bank
    assert cache.begin(second, list(second.stage("rot").parameters())).pool is pool
    assert not pool.closed
    cache.close()
    assert pool.closed  # released with the bank and its held-out executor
    assert cache.pool is None


def test_a_fit_graph_cache_opens_a_fresh_pool_for_an_incompatible_fit(llama_bundle):
    from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache
    from tests.neural.engines.pytorch_hooks.test_graph_reuse import executor

    cache = FitGraphCache()
    first = executor(llama_bundle)
    pool = cache.begin(first, list(first.stage("rot").parameters())).pool
    other = executor(llama_bundle, rank=8)  # a different featurizer layout
    fresh = cache.begin(other, list(other.stage("rot").parameters())).pool
    assert fresh is not pool
    assert pool.closed and not fresh.closed  # the evicted bank's pool went with it
    cache.close()
    assert fresh.closed


def test_eval_executor_joins_the_fits_pool(llama_bundle, monkeypatch):
    from causalab.neural.engines.pytorch_hooks import train
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool
    from tests.neural.engines.pytorch_hooks.test_graph_reuse import executor
    from tests.neural.engines.pytorch_hooks.test_prefix_resume import (
        EVAL_SPLIT,
        _train_request,  # pyright: ignore[reportPrivateUsage]
    )

    point = executor(llama_bundle)
    pool = GraphPool()
    monkeypatch.setattr(
        train,
        "make_executor",
        lambda doc, bundle, **kwargs: GraphExecutor(
            doc, bundle, **{k: v for k, v in kwargs.items() if k != "cuda_graphs"}
        ),
    )
    built = train._eval_executor(  # pyright: ignore[reportPrivateUsage]
        point.doc, point, _train_request(), EVAL_SPLIT, pool=pool
    )
    assert isinstance(built, GraphExecutor)
    assert built.graph_pool is pool
    # the pool travels to the eval executor explicitly; the point executor,
    # which never captures during a fit, carries none
    assert point.graph_pool is None


@pytest.mark.parametrize("shared", [False, True])
def test_the_warmup_allocates_into_the_pool(monkeypatch, shared):
    """The warm-up pass grows the pool the capture allocates from: for
    exactly that pass every allocation on the device is routed into the
    pool (``GraphPool.allocating`` — autograd's thread included, which
    ``torch.cuda.use_mem_pool`` would miss), the ordinary cache emptied
    first, so the capture finds its working set cached on its stream. A
    private capture (no pool) routes nothing and the capture pass itself
    never runs routed."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool, Replay
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    cuda = FakeCuda().install(monkeypatch)
    emptied: list[tuple[int, ...] | None] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(cuda.routing))
    pool = GraphPool() if shared else None
    device = torch.device("cpu")
    seen: list[tuple[int, int] | None] = []

    def work():
        seen.append(cuda.routing)  # the pool allocations go to, if any
        return torch.ones(1)

    Replay(work, {}, device=device, pool=pool)
    handle = None if pool is None else pool.handle(device)
    # one warm-up pass (routed to the pool when there is one), then the capture
    assert seen == ([handle, None] if shared else [None, None])
    if shared:
        assert cuda.routed == [
            ("begin", handle),
            ("end", handle),
            ("release", handle),
        ]
        # the ordinary cache is emptied exactly once here, before the routing
        # begins (the allocator releases nothing on pressure while
        # allocations are routed); the real torch.cuda.graph empties it again
        # before the capture, the fake one does not
        assert emptied == [None]
        assert pool is not None
        pool.close()
    else:
        assert cuda.routed == []
        assert emptied == []  # the private path empties nothing


def test_a_capture_without_a_warmup_records_the_pass_once(monkeypatch):
    """``warmup=False`` — for a pass the caller has just run eagerly on the
    storage the capture records — runs the work once, in the capture, and
    routes nothing."""
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool, Replay
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    cuda = FakeCuda().install(monkeypatch)
    pool = GraphPool()
    calls = 0

    def work():
        nonlocal calls
        calls += 1
        return torch.ones(1)

    replay = Replay(work, {}, device=torch.device("cpu"), pool=pool, warmup=False)
    assert calls == 1
    assert cuda.routed == []
    assert cuda.pool_ids == [pool.handle(torch.device("cpu"))]
    assert replay.pool is pool
    pool.close()


@pytest.mark.skipif(
    torch.version.cuda is None,
    reason="a CPU-only torch need not have the CUDA allocator bindings",
)
def test_the_pool_routing_bindings_exist():
    """``GraphPool.allocating`` reaches for torch's private allocator surface
    (``torch.cuda.use_mem_pool`` is built on the same three bindings). The
    fakes install whether or not the build has them (the darwin wheel does
    carry them; a ``+cpu`` wheel need not), so this is the CPU gate's notice,
    on the CUDA build CI installs, when a torch release moves them."""
    for name in (
        "_cuda_beginAllocateToPool",
        "_cuda_endAllocateToPool",
        "_cuda_releasePool",
    ):
        assert hasattr(torch._C, name), name


def test_a_closed_pool_routes_no_allocation(monkeypatch):
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphPool
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    cuda = FakeCuda().install(monkeypatch)
    pool = GraphPool()
    device = torch.device("cpu")
    pool.handle(device)
    pool.close()
    with pool.allocating(device):
        assert cuda.routing is None
    assert cuda.routed == []
