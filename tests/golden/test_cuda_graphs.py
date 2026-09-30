"""Production CUDA replay against eager on real Qwen3, including the fit loop."""

from __future__ import annotations

# The assertions inspect replay buffers/cache ownership, not a second executor.
# pyright: reportPrivateUsage=false

import copy
import gc
import json
import os
from unittest.mock import patch

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import train
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    graph_device,
    GraphExecutor,
    Replay,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.executor import ForwardCache, Interning
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.featurizers import Gate
from causalab.protocol.engine import RunContext
from causalab.io.env import FileArtifacts, ResolutionEnv
from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.protocol._docs import saved
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
    dbm_doc,
    ce_term,
)

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

MODEL = os.environ.get("CAUSALAB_CUDA_GRAPH_MODEL", "Qwen/Qwen3.6-35B-A3B")


@pytest.fixture(scope="module")
def bundle():
    # Match the engine's cache key so the CLI round trip shares this backbone.
    return load_model(MODEL, "main", dtype="bf16", device="cuda", quantization=None)


def document(method):
    raw = dbm_doc() if method == "dbm" else das_doc()
    raw["model"].update(key=MODEL, dtype="bf16")
    raw["method"]["sites"]["tgt"]["layers"] = [18]
    # the target activation on base, un-intervened and patched (§2.9): the
    # network on base is a model of its own, declared here
    models = raw["method"]["intervened_models"]
    models["original_base"] = {"input": "base", "reads": []}
    for name, model in (("base_act", "original_base"), ("patched_act", "patched")):
        raw["method"]["reads"][name] = {"site": "tgt", "pos": {"index": -1}}
        models[model]["reads"].append(name)
        raw["method"]["save"].append(saved(name, model, name + ".safetensors"))
    return raw


def executor(
    raw,
    bundle,
    *,
    graphs,
    base=BASES,
    cf=COUNTERFACTUALS,
    answers=ANSWERS,
    interning=None,
):
    reference = executor_for(
        raw,
        bundle,
        base_texts=base,
        counterfactual_texts=cf,
        extra_columns={"label": answers},
    )
    return make_executor(
        reference.doc,
        bundle,
        cuda_graphs=graphs,
        role_rows=reference.role_rows,
        role_fields=reference.role_fields,
        load_tensors=reference.load_tensors,
        interning=interning,
    )


@pytest.mark.parametrize("method", ["dbm", "das"])
def test_inference_replay_preserves_outputs_and_forward_cache(bundle, method):
    raw = document(method)
    torch.manual_seed(0)
    actual = executor(
        raw,
        bundle,
        graphs=True,
        base=BASES[:2],
        cf=COUNTERFACTUALS[:2],
        answers=ANSWERS[:2],
    )
    assert isinstance(actual, GraphExecutor)
    stage = actual.stage(next(iter(actual.doc.featurizers)))
    parameter = next(stage.parameters())
    with torch.no_grad():
        parameter.add_(torch.randn_like(parameter) * 0.02)
    reference = executor(
        raw,
        bundle,
        graphs=False,
        base=BASES[:2],
        cf=COUNTERFACTUALS[:2],
        answers=ANSWERS[:2],
    )
    reference.stage_cache = actual.stage_cache
    # First run captures, subsequent runs exercise changed tokens, parameters,
    # temperature, and train/eval mode transitions on the same executor.
    held = None
    held_copy = None
    for step in range(4):
        stage.train(step == 2)
        if isinstance(stage, Gate):
            stage.temperature = 0.3 + step * 0.2
        with torch.no_grad():
            parameter.add_(0.001)
        for role in actual.role_rows:
            actual._batch(role).input_ids.copy_(
                actual._batch(role).input_ids.roll(1, 0)
            )
            reference._batch(role).input_ids.copy_(actual._batch(role).input_ids)
        actual.reset_reads()
        reference.reset_reads()
        for name in actual.doc.reads:
            value = actual.dense_value(name)
            torch.testing.assert_close(
                value, reference.dense_value(name), rtol=0, atol=0
            )
            if name == "patched_act" and step == 0:
                held, held_copy = value, value.clone()
        if held is not None:
            torch.testing.assert_close(held, held_copy, rtol=0, atol=0)
    assert all(replay.replays >= 1 for replay, _ in actual._inference_graphs.values())

    cache = ForwardCache()
    keys = {
        (name, str(m.input)): f"{name}/{m.input}"
        for name, m in actual.doc.intervened_models.items()
    }
    interning = Interning(keys=keys, cache=cache)
    first = executor(
        raw,
        bundle,
        graphs=True,
        base=BASES[:2],
        cf=COUNTERFACTUALS[:2],
        answers=ANSWERS[:2],
        interning=interning,
    )
    first.stage_cache = actual.stage_cache
    first.run_all()
    assert len(cache.executed) == 3  # warmup/capture must not publish extra passes
    frozen = {
        d: {k: v.clone() for k, v in taps.items()} for d, taps in cache.captured.items()
    }
    first.reset_reads()
    first.run_all()
    assert len(cache.executed) == 3  # actual cache hits, not merely same outputs
    for digest, taps in frozen.items():
        for key, value in taps.items():
            torch.testing.assert_close(
                cache.captured[digest][key], value, rtol=0, atol=0
            )


class Datasets:
    def digest(self, ref):
        return "0" * 64

    def columns(self, ref):
        return ("input", "counterfactual_inputs", "label")

    def rows(self, ref):
        return [
            {"input": b, "counterfactual_inputs": [c], "label": a}
            for b, c, a in zip(BASES[:2], COUNTERFACTUALS[:2], ANSWERS[:2])
        ]


@pytest.mark.parametrize("method", ["dbm", "das"])
@pytest.mark.parametrize("oom_on_third_capture", [False, True])
@pytest.mark.parametrize("warmup_steps", [1, 3])
def test_production_fit_matches_eager_updates_and_eval(
    bundle, method, oom_on_third_capture, tmp_path, warmup_steps, monkeypatch
):
    """The production loop against eager on a dataset whose minibatches take
    three shape/mask buckets: every step replays, three graphs on the fit's one
    pool. With ``oom_on_third_capture`` the third bucket's capture runs the
    allocator out of memory mid-fit: the bank is released, the rest of the fit
    runs eagerly, the held-out inference graphs on the same pool keep
    replaying — and the updates, parameters and eval scores are the eager
    fit's bit for bit either way."""
    monkeypatch.setattr(Replay, "warmup_steps", warmup_steps)
    raw = document(method)
    raw["method"]["train"]["steps"] = {"epochs": 3}
    raw["method"]["train"]["eval"] = {
        "split": "eval",
        "every": {"epochs": 1},
        "aggregations": {"ce": ce_term()},
    }
    raw["method"]["train"]["early_stop"] = {
        "on": "ce",
        "mode": "min",
        "patience": 1,
    }
    # Two matching batches with different tokens/labels, a changed mask, and
    # a partial batch: three buckets.
    base = (
        BASES[:2]
        + [BASES[0].replace("quick", "slow"), BASES[1].replace("green", "brown")]
        + ["the fox", BASES[1], BASES[0]]
    )
    cf = (
        COUNTERFACTUALS[:2]
        + [
            COUNTERFACTUALS[0].replace("cold", "warm"),
            COUNTERFACTUALS[1].replace("bright", "cold"),
        ]
        + ["cold mountains", COUNTERFACTUALS[1], COUNTERFACTUALS[0]]
    )
    for texts in (base, cf):
        encoded = bundle.tokenizer(texts[:4], padding=True, return_tensors="pt")
        torch.testing.assert_close(
            encoded.attention_mask[:2], encoded.attention_mask[2:]
        )
        assert not torch.equal(encoded.input_ids[:2], encoded.input_ids[2:])
    answers = ANSWERS + ANSWERS[:3]
    request = RunContext(
        env=ResolutionEnv(datasets=Datasets(), artifacts=FileArtifacts(tmp_path)),
        output_dir=tmp_path,
    )
    build_optimizer = train._build_optimizer
    replay_call = Replay.__call__
    replay_init = Replay.__init__
    replay_count = 0
    training_replays = 0
    training_graphs = set()
    training_captures = 0
    inference_captures = 0
    inference_reuses = 0

    def counted_init(self, work, *args, **kwargs):
        nonlocal training_captures
        if not kwargs.get("parameters"):
            return replay_init(self, work, *args, **kwargs)
        training_captures += 1
        if not (oom_on_third_capture and training_captures == 3):
            return replay_init(self, work, *args, **kwargs)

        def failing_work():
            result = work()
            if torch.cuda.is_current_stream_capturing():
                # Larger than physical VRAM: a real allocator OOM inside the
                # capture, without allocating the rest of the device.
                device = graph_device(bundle)
                total = torch.cuda.get_device_properties(device).total_memory
                torch.empty(total + 1, device=device, dtype=torch.uint8)
            return result

        replay_init(self, failing_work, *args, **kwargs)

    def counted_replay(self):
        nonlocal replay_count, training_replays
        nonlocal inference_captures, inference_reuses
        replay_count += 1
        if self.parameters:
            training_replays += 1
            training_graphs.add(id(self))
        elif self.replays == 0:
            inference_captures += 1
        else:
            inference_reuses += 1
        return replay_call(self)

    def fit(graphs):
        actual = executor(raw, bundle, graphs=graphs, base=base, cf=cf, answers=answers)
        updates = []
        eval_scores = []
        evaluate = train._score

        def traced_eval(doc, worker):
            score = evaluate(doc, worker)
            eval_scores.append(dict(score))
            return score

        def traced_optimizer(spec, parameters):
            optimizer = build_optimizer(spec, parameters)
            parameters = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            step = optimizer.step

            def traced_step(*args, **kwargs):
                grads = [
                    p.grad.detach().cpu().clone() if p.grad is not None else None
                    for p in parameters
                ]
                result = step(*args, **kwargs)
                updates.append(
                    (
                        grads,
                        [p.detach().cpu().clone() for p in parameters],
                        copy.deepcopy(optimizer.state_dict()),
                    )
                )
                return result

            optimizer.step = traced_step
            return optimizer

        with (
            patch.object(train, "_build_optimizer", traced_optimizer),
            patch.object(Replay, "__call__", counted_replay),
            patch.object(Replay, "__init__", counted_init),
            patch.object(train, "_score", traced_eval),
        ):
            outcome = train.run_training(actual.doc, actual, request)
        return outcome, updates, eval_scores

    expected, eager_updates, eager_scores = fit(False)
    assert replay_count == 0
    actual, graph_updates, graph_scores = fit(True)
    assert inference_captures > 0
    assert (
        inference_reuses == inference_captures
    )  # first epoch eager, second captures, third replays
    assert graph_scores == eager_scores
    assert replay_count >= 4
    assert training_captures == 3  # the third bucket is captured, or attempted
    if oom_on_third_capture:
        # the third bucket is first drawn at the third or fourth step of the
        # first epoch (a permutation of two matching batches and two others);
        # the steps before it replayed, nothing replays after the fallback
        assert len(training_graphs) == 2
        assert 2 <= training_replays <= 3
    else:
        assert training_replays == 12  # every step of every epoch
        assert len(training_graphs) == 3
    assert len(graph_updates) == len(eager_updates) == 12
    for graph_update, eager_update in zip(graph_updates, eager_updates):
        torch.testing.assert_close(graph_update, eager_update, rtol=0, atol=0)
    for name, stage in actual.stages.items():
        torch.testing.assert_close(
            stage.state_dict(), expected.stages[name].state_dict(), rtol=0, atol=0
        )
        assert not stage.training
        if hasattr(stage, "temperature"):
            assert stage.temperature == expected.stages[name].temperature
            assert stage.capture_temperature is None
    assert actual.eval_score == expected.eval_score


def test_cli_fit_saves_the_same_artifacts_with_cuda_graphs(bundle, tmp_path):
    from causalab.cli import main
    from safetensors.torch import load_file
    from tests.protocol._docs import in_order

    raw = document("dbm")
    raw["method"]["train"]["steps"] = {"epochs": 2}
    for role in raw["data"].values():
        role["dataset"] = "rows.json"
    rows = [
        {"input": b, "counterfactual_inputs": [c], "label": a, "split": "all"}
        for b, c, a in zip(BASES, COUNTERFACTUALS, ANSWERS)
    ]
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(in_order(raw)))
    replays = []
    original = Replay.__call__

    def record(self):
        replays.append(bool(self.parameters))
        return original(self)

    for graphs in (False, True):
        args = [
            "run",
            str(path),
            "--data-root",
            str(tmp_path),
            "--device",
            "cuda",
            "--engine",
            "pytorch_hooks",
            "--out",
            str(tmp_path / str(graphs)),
        ]
        if graphs:
            args.append("--cuda-graphs")
        with patch.object(Replay, "__call__", record):
            assert main(args) == 0
    assert replays and all(replays)  # training replays; one-shot saved reads stay eager
    for entry in raw["method"]["save"]:
        name = entry["file_path"]
        first, second = tmp_path / "False" / name, tmp_path / "True" / name
        if name.endswith(".safetensors"):
            torch.testing.assert_close(
                load_file(str(first)), load_file(str(second)), rtol=0, atol=0
            )
        else:
            assert json.loads(first.read_text()) == json.loads(second.read_text())


@pytest.mark.parametrize("failure_phase", ["warmup", "capture"])
def test_first_capture_oom_falls_back_with_exact_gradients(
    bundle, failure_phase, monkeypatch
):
    from causalab.neural.engines.pytorch_hooks import cuda_graphs

    raw = document("dbm")
    point = executor(raw, bundle, graphs=True)
    fit = train._prepare_fit(point.doc, point)
    batch = fit.minibatch_executors[0]
    objective = train.TrainingObjective(batch, fit.stages)
    parameters = [p for group in fit.optimizer.param_groups for p in group["params"]]
    expected = torch.autograd.grad(objective(), parameters)
    batch.reset_reads()
    bank = cuda_graphs.TrainingGraphs(parameters)
    original = Replay.__init__

    def fail(self, work, *args, **kwargs):
        def failing_work():
            result = work()
            if torch.cuda.is_current_stream_capturing() == (failure_phase == "capture"):
                # Larger than physical VRAM: a real allocator OOM without
                # allocating the rest of the device or affecting other jobs.
                total = torch.cuda.get_device_properties(
                    graph_device(bundle)
                ).total_memory
                torch.empty(total + 1, device=graph_device(bundle), dtype=torch.uint8)
            return result

        original(self, failing_work, *args, **kwargs)

    monkeypatch.setattr(Replay, "__init__", fail)
    assert not bank.backward(batch, objective)
    assert bank.disabled and not bank.buckets
    assert all(p.grad is None for p in parameters)
    objective().backward()
    torch.testing.assert_close(
        [p.grad for p in parameters], list(expected), rtol=0, atol=0
    )
    bank.close()


@pytest.mark.parametrize("temperature", [0.1, 0.3, 0.7, 1.3, 2.0, 10.0])
def test_captured_gate_temperature_matches_eager_scalar_division(temperature):
    gate = Gate(10000).cuda()
    with torch.no_grad():
        gate.theta.copy_(torch.linspace(-6, 6, gate.theta.numel(), device="cuda"))
    gate.temperature = temperature
    expected = gate.soft_mask().detach()
    replay = Replay(
        lambda: gate.soft_mask().detach(), {"gate": gate}, device=torch.device("cuda")
    )
    torch.testing.assert_close(replay(), expected, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# one pool per fit's graphs (docs/cuda_graphs.md "One pool per engine")

THREE_LENGTHS = (
    ["the fox", BASES[0], BASES[1] + " under a bright red autumn moon"],
    ["cold hills", COUNTERFACTUALS[0], COUNTERFACTUALS[1] + " beside the lake"],
    ANSWERS[:3],
)


def _pool_bytes(pool_ids) -> int:
    """The segments the allocator holds for ``pool_ids``."""
    wanted = {tuple(pool) for pool in pool_ids}
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if tuple(segment.get("segment_pool_id", ())) in wanted
    )


def _three_bucket_bank(bundle, method, *, pool_kind):
    """A fit's three one-row minibatches of three padded lengths, captured
    into a ``TrainingGraphs`` bank: on the fit's one shared pool, or one
    private pool per graph (``Replay`` without a pool) for comparison."""
    from causalab.neural.engines.pytorch_hooks import cuda_graphs

    raw = document(method)
    raw["method"]["train"]["batch"] = {"pairs": 1}
    base, cf, answers = THREE_LENGTHS
    point = executor(raw, bundle, graphs=True, base=base, cf=cf, answers=answers)
    assert isinstance(point, GraphExecutor)
    fit = train._prepare_fit(point.doc, point)
    assert len(fit.minibatch_executors) == 3
    # the minibatches select rows of the point's one padded frame, so the
    # buckets differ by attention mask (the rows' lengths), not by width
    masks = {
        tuple(m._batch("base").attention_mask.flatten().tolist())
        for m in fit.minibatch_executors
    }
    assert len(masks) == 3, "the rows must bring three different masks"
    parameters = [p for g in fit.optimizer.param_groups for p in g["params"]]
    bank = cuda_graphs.TrainingGraphs(parameters)
    if pool_kind == "private":
        bank.pool.close()  # a closed pool: every capture keeps a private one
    objectives = [
        train.TrainingObjective(batch, fit.stages) for batch in fit.minibatch_executors
    ]
    return fit, bank, objectives, parameters


@pytest.mark.parametrize("method", ["dbm", "das"])
def test_three_buckets_share_one_pool_and_replay_out_of_capture_order(bundle, method):
    """Three buckets on one pool, replayed in an order other than the one
    they were captured in: a graph replayed after a later capture may write
    blocks the later graph's outputs sit in, so every replay's gradients are
    read right after it — as the loop's optimizer step does — and must be
    the eager ones, bit for bit, with no OOM fallback."""
    fit, bank, objectives, parameters = _three_bucket_bank(
        bundle, method, pool_kind="shared"
    )
    expected = []
    for batch, objective in zip(fit.minibatch_executors, objectives, strict=True):
        expected.append(
            [g.detach().clone() for g in torch.autograd.grad(objective(), parameters)]
        )
        batch.reset_reads()
    try:
        order = [0, 1, 2, 2, 1, 0, 0, 2, 1, 1, 0, 2]
        for step, index in enumerate(order):
            for parameter in parameters:
                parameter.grad = None
            assert bank.backward(fit.minibatch_executors[index], objectives[index]), (
                f"step {step} (bucket {index}) fell back to eager"
            )
            torch.testing.assert_close(
                [p.grad for p in parameters], expected[index], rtol=0, atol=0
            )
        assert not bank.disabled
        assert len(bank.buckets) == 3
        replays = [replay for _, _, replay in bank.buckets.values()]
        assert [r.replays for r in replays] == [4, 4, 4]
        pool_ids = {tuple(r.graph.pool()) for r in replays}
        assert len(pool_ids) == 1, "three buckets, one pool"
        handle = bank.pool.handle(graph_device(bundle))
        assert handle is not None and pool_ids == {tuple(handle)}
    finally:
        bank.close()
        bank.pool.close()


def test_three_buckets_on_one_pool_reserve_less_than_three_private_pools(bundle):
    """The point of the shared pool, measured: the allocator segments the
    three buckets' graphs hold on one pool against the sum of what the same
    three graphs hold on a private pool each. Printed for the record."""
    reserved = {}
    for pool_kind in ("private", "shared"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_reserved()
        fit, bank, objectives, parameters = _three_bucket_bank(
            bundle, "das", pool_kind=pool_kind
        )
        replays = []
        try:
            for batch, objective in zip(
                fit.minibatch_executors, objectives, strict=True
            ):
                assert bank.backward(batch, objective)
            replays = [replay for _, _, replay in bank.buckets.values()]
            pool_ids = [tuple(r.graph.pool()) for r in replays]
            assert len(set(pool_ids)) == (3 if pool_kind == "private" else 1)
            torch.cuda.synchronize()
            reserved[pool_kind] = {
                "pools": _pool_bytes(pool_ids),
                "each": [_pool_bytes([pool]) for pool in pool_ids],
                "process_delta": torch.cuda.memory_reserved() - before,
            }
        finally:
            bank.close()
            bank.pool.close()
            del fit, bank, objectives, parameters, replays
    gc.collect()
    torch.cuda.empty_cache()
    for kind, numbers in reserved.items():
        print(
            f"\n{kind} pools: {numbers['pools'] / 2**20:.0f} MiB in graph pools "
            f"({', '.join(f'{b / 2**20:.0f}' for b in numbers['each'])} MiB each), "
            f"process reserved +{numbers['process_delta'] / 2**20:.0f} MiB"
        )
    shared, private = reserved["shared"], reserved["private"]
    # one working set plus the other two graphs' live outputs, not three
    # working sets: fewer segments in the graphs' pools than three private
    # pools hold
    assert 0 < shared["pools"] < private["pools"]
    # and less device memory held by the process as a whole (the line this
    # test prints read +122 MiB shared, +370 MiB for three private pools on
    # an H100). The shared pool also
    # keeps the warm-up's allocations (the cuBLAS workspaces of the main and
    # autograd threads among them), which a private-pool capture leaves
    # outside any pool, so the pool's own size is not the comparison.
    assert shared["process_delta"] < private["process_delta"]


def test_a_fit_with_three_buckets_matches_eager_without_fallback(bundle, tmp_path):
    """The production loop on a dataset whose minibatches take three padded
    shapes (two matching pairs, a changed mask, a partial batch): every one
    of the twelve steps replays — three graphs on one pool, no eager
    fallback for a third bucket — and the updates, parameters and eval
    scores are the eager fit's bit for bit."""
    raw = document("dbm")
    raw["method"]["train"]["steps"] = {"epochs": 3}
    raw["method"]["train"]["eval"] = {
        "split": "eval",
        "every": {"epochs": 1},
        "aggregations": {"ce": ce_term()},
    }
    base = (
        BASES[:2]
        + [BASES[0].replace("quick", "slow"), BASES[1].replace("green", "brown")]
        + ["the fox", BASES[1], BASES[0]]
    )
    cf = (
        COUNTERFACTUALS[:2]
        + [
            COUNTERFACTUALS[0].replace("cold", "warm"),
            COUNTERFACTUALS[1].replace("bright", "cold"),
        ]
        + ["cold mountains", COUNTERFACTUALS[1], COUNTERFACTUALS[0]]
    )
    answers = ANSWERS + ANSWERS[:3]
    request = RunContext(
        env=ResolutionEnv(datasets=Datasets(), artifacts=FileArtifacts(tmp_path)),
        output_dir=tmp_path,
    )
    build_optimizer = train._build_optimizer
    score_fit = train._score
    replay_call = Replay.__call__

    def fit(graphs):
        point = executor(raw, bundle, graphs=graphs, base=base, cf=cf, answers=answers)
        updates, scores, training_graphs = [], [], []

        def traced_optimizer(spec, parameters):
            optimizer = build_optimizer(spec, parameters)
            held = [p for group in optimizer.param_groups for p in group["params"]]
            step = optimizer.step

            def traced_step(*args, **kwargs):
                grads = [p.grad.detach().cpu().clone() for p in held]
                result = step(*args, **kwargs)
                updates.append((grads, [p.detach().cpu().clone() for p in held]))
                return result

            optimizer.step = traced_step
            return optimizer

        def traced_score(doc, worker):
            score = score_fit(doc, worker)
            scores.append(dict(score))
            return score

        def counted_replay(self):
            if self.parameters:
                training_graphs.append((id(self), tuple(self.graph.pool())))
            return replay_call(self)

        with (
            patch.object(train, "_build_optimizer", traced_optimizer),
            patch.object(train, "_score", traced_score),
            patch.object(Replay, "__call__", counted_replay),
        ):
            outcome = train.run_training(point.doc, point, request)
        return outcome, updates, scores, training_graphs

    expected, eager_updates, eager_scores, none = fit(False)
    assert not none
    actual, graph_updates, graph_scores, training = fit(True)
    assert len(training) == 12, "every step of every epoch replays"
    assert len({graph for graph, _ in training}) == 3, "three buckets"
    assert len({pool for _, pool in training}) == 1, "one pool"
    assert graph_scores == eager_scores
    assert len(graph_updates) == len(eager_updates) == 12
    for graph_update, eager_update in zip(graph_updates, eager_updates, strict=True):
        torch.testing.assert_close(graph_update, eager_update, rtol=0, atol=0)
    for name, stage in actual.stages.items():
        torch.testing.assert_close(
            stage.state_dict(), expected.stages[name].state_dict(), rtol=0, atol=0
        )
    assert actual.eval_score == expected.eval_score
