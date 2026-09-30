"""Compatibility and ownership of request-scoped capture reuse."""

import copy
from types import SimpleNamespace

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.cuda_graphs import GraphExecutor, GraphPool
from causalab.neural.engines.pytorch_hooks.graph_reuse import (
    FitGraphCache,
    fit_signature,
)
from causalab.protocol.engine import RunContext
from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
)

pytestmark = pytest.mark.unit


def handoff(tmp_path):
    """What ``Engine.execute(compiled, run)`` is handed here: a compiled
    stand-in and a real ``RunContext``. Under a monkeypatched
    ``execute_request`` the engine reads only ``run.execution`` (its fit
    bound) and ``run.decoding`` (``None``: no continuations), never the
    compiled object."""
    compiled = SimpleNamespace(points=SimpleNamespace(points=()))
    run = RunContext(output_dir=tmp_path, env=None)  # pyright: ignore[reportArgumentType]
    return compiled, run


def executor(bundle, *, seed=0, rank=4, rows=None):
    raw = das_doc(seed=seed)
    raw["method"]["featurizers"]["rot"]["k"] = rank
    eager = executor_for(
        raw,
        bundle,
        base_texts=rows or BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    # Real executor/state on CPU: these tests never request a CUDA capture.
    return GraphExecutor(
        eager.doc,
        bundle,
        role_rows=eager.role_rows,
        role_fields=eager.role_fields,
        load_tensors=eager.load_tensors,
    )


def test_seed_only_compatibility_and_fresh_parameter_ownership(llama_bundle):
    first, second = executor(llama_bundle), executor(llama_bundle, seed=1)
    assert fit_signature(first) == fit_signature(second)
    cache = FitGraphCache()
    a = list(first.stage("rot").parameters())
    b = list(second.stage("rot").parameters())
    bank = cache.begin(first, a)
    assert cache.begin(second, b) is bank
    assert bank.parameters is b
    assert a[0] is not b[0]
    assert cache.reused_training_fits == 1
    cache.close()
    assert cache.training is None


@pytest.mark.parametrize("change", ["rank", "rows", "labels", "model", "dtype"])
def test_incompatible_fit_evicts_previous_bank(llama_bundle, change):
    first = executor(llama_bundle)
    second = executor(llama_bundle, seed=1, rank=2 if change == "rank" else 4)
    if change in {"rows", "labels"}:
        second.role_rows = copy.deepcopy(second.role_rows)
        second.role_rows["base"][0]["input" if change == "rows" else "label"] = (
            "different"
        )
    elif change == "model":
        import dataclasses

        second.bundle = dataclasses.replace(
            llama_bundle, model=copy.deepcopy(llama_bundle.model)
        )
    elif change == "dtype":
        second.stage("rot").double()
    cache = FitGraphCache()
    params = list(first.stage("rot").parameters())
    old = cache.begin(first, params)
    params[0].grad = torch.ones_like(params[0])
    new = cache.begin(second, list(second.stage("rot").parameters()))
    assert old is not new
    assert params[0].grad is None
    assert cache.reused_training_fits == 0
    cache.close()


@pytest.mark.parametrize(
    "success,disabled", [(False, False), (True, True), (True, False)]
)
def test_failed_fallback_and_empty_fits_release_both_caches(
    llama_bundle, success, disabled
):
    point = executor(llama_bundle)
    cache = FitGraphCache()
    parameters = list(point.stage("rot").parameters())
    bank = cache.begin(point, parameters)
    bank.disabled = disabled
    held_out = cache.evaluation("held-out", lambda: executor(llama_bundle))
    assert (
        cache.evaluation("held-out", lambda: pytest.fail("unexpected new executor"))
        is held_out
    )
    parameters[0].grad = torch.ones_like(parameters[0])
    cache.finish(success=success)
    assert cache.training is cache.eval_executor is cache.signature is None
    assert parameters[0].grad is None
    cache.close()  # idempotent


def test_request_failure_releases_cache_without_affecting_another_request(
    llama_bundle, monkeypatch, tmp_path
):
    from causalab.neural.engines.pytorch_hooks import engine

    other = FitGraphCache()
    point = executor(llama_bundle)
    other_bank = other.begin(point, list(point.stage("rot").parameters()))
    observed = []

    def fail_request(compiled, run, **kwargs):
        cache = kwargs["train_runner"].keywords["graph_cache"]
        observed.append(cache)
        cache.begin(point, list(point.stage("rot").parameters()))
        cache.evaluation("held-out", lambda: executor(llama_bundle))
        raise OSError("forced output failure")

    monkeypatch.setattr(engine, "execute_request", fail_request)
    with pytest.raises(OSError, match="forced output failure"):
        engine.PytorchHooksEngine(cuda_graphs=True).execute(
            *handoff(tmp_path)  # type: ignore[arg-type]  # forced boundary failure
        )
    assert observed[0].training is observed[0].eval_executor is None
    assert other.training is other_bank
    other.close()


def test_signature_uses_parsed_fields_even_when_raw_is_stale(llama_bundle):
    import dataclasses

    first = executor(llama_bundle)
    second = executor(llama_bundle, seed=1)
    second.doc = dataclasses.replace(second.doc, raw={})
    assert fit_signature(first) == fit_signature(second)
    second.doc = dataclasses.replace(
        second.doc,
        sites={
            **second.doc.sites,
            "tgt": dataclasses.replace(second.doc.sites["tgt"], layers=(1,)),
        },
    )
    assert fit_signature(first) != fit_signature(second)


def test_eval_reuse_does_not_build_or_copy_again(llama_bundle, monkeypatch):
    from causalab.neural.engines.pytorch_hooks import train
    from tests.neural.engines.pytorch_hooks.test_prefix_resume import (
        _train_request,
        EVAL_SPLIT,
    )

    cache = FitGraphCache()
    first, second = executor(llama_bundle), executor(llama_bundle, seed=1)
    for point in (first, second):
        point.graph_cache = cache
        point.stage("rot").train()
    # Only CUDA eligibility is bypassed: no forward/capture is requested.
    monkeypatch.setattr(
        train,
        "make_executor",
        lambda doc, bundle, **kwargs: GraphExecutor(
            doc, bundle, **{k: v for k, v in kwargs.items() if k != "cuda_graphs"}
        ),
    )
    built = train._eval_executor(first.doc, first, _train_request(), EVAL_SPLIT)
    assert built.stage_cache["rot"] is not first.stage("rot")
    monkeypatch.setattr(
        train.copy, "deepcopy", lambda *args: pytest.fail("unused stage copy")
    )
    monkeypatch.setattr(
        train, "make_executor", lambda *args, **kwargs: pytest.fail("unused executor")
    )
    assert (
        train._eval_executor(second.doc, second, _train_request(), EVAL_SPLIT) is built
    )
    cache.close()


def test_a_fit_graph_cache_borrows_a_pool_and_leaves_it_open(llama_bundle):
    """Handed the engine's pool, the cache captures its bank into it and its
    close releases the bank but not the pool; a closed pool is not borrowed
    — the cache opens one of its own, released with the bank as before."""
    point = executor(llama_bundle)
    parameters = list(point.stage("rot").parameters())
    shared = GraphPool()
    cache = FitGraphCache(pool=shared)
    assert cache.pool is None  # nothing until a fit begins
    assert cache.begin(point, parameters).pool is shared
    assert cache.pool is shared
    cache.close()
    assert not shared.closed
    assert cache.pool is None
    shared.close()
    cache = FitGraphCache(pool=shared)
    own = cache.begin(point, parameters).pool
    assert own is not shared and not own.closed
    cache.close()
    assert own.closed


def test_the_engine_keeps_one_pool_across_requests_and_replaces_a_closed_one(
    llama_bundle, monkeypatch, tmp_path
):
    """One ``GraphPool`` per engine: every request's fit cache and training
    runner are handed the same pool, which the request's teardown leaves
    open; a pool closed by an out-of-memory fallback is replaced for the
    next request. With graphs off there is no pool and no cache."""
    from causalab.neural.engines.pytorch_hooks import engine

    seen = []

    def record(compiled, run, **kwargs):
        keywords = kwargs["train_runner"].keywords
        seen.append((keywords["graph_pool"], keywords["graph_cache"]))
        return SimpleNamespace(files={})

    monkeypatch.setattr(engine, "execute_request", record)
    hooks = engine.PytorchHooksEngine(cuda_graphs=True)
    compiled, run = handoff(tmp_path)
    hooks.execute(compiled, run)  # type: ignore[arg-type]
    hooks.execute(compiled, run)  # type: ignore[arg-type]
    (first, cache), (second, other) = seen
    assert isinstance(first, GraphPool) and first is second
    assert cache is not other  # the graphs are the request's; the pool is not
    assert not first.closed
    point = executor(llama_bundle)
    assert cache.begin(point, list(point.stage("rot").parameters())).pool is first
    cache.close()
    assert not first.closed
    first.close()
    hooks.execute(compiled, run)  # type: ignore[arg-type]
    third = seen[2][0]
    assert isinstance(third, GraphPool) and third is not first and not third.closed
    seen.clear()
    engine.PytorchHooksEngine(cuda_graphs=False).execute(compiled, run)  # type: ignore[arg-type]
    assert seen == [(None, None)]
