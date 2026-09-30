"""Fit-constant forwards run once inside a fit (spec §3, §4 "Fits").

The shipped DAS/DBM methods read their operand off ``original`` on the
counterfactual rows: frozen weights, no writes, and the trained featurizer is
applied *after* the capture. That forward's activations cannot change between
optimizer steps, yet the train loop re-ran it every step, every epoch, every
eval pass, and again for every point of a swept campaign. This file pins the
new arithmetic — with ``B`` minibatches and ``E`` epochs (eval every epoch) a
fit pays ``B + B·E + 1 + E`` model forwards instead of ``2·B·E + 2·E`` — and
the two properties that make it safe: the fitted weights are bit-identical to
the un-interned path, and nothing a trained parameter can reach is ever
served from the cache.

Ground truth for "a forward happened" is a pre-hook on the loaded model, as in
``test_forward_interning.py``; the batch size it sees tells a minibatch pass
(2 rows) from an eval pass (3 rows) apart.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.train import run_training
from causalab.neural.shared.execution import campaign_plans
from causalab.neural.shared.executor import ForwardCache, Interning
from causalab.protocol.engine import RunContext
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import interned_groups, plan_point
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import Document, parse_document

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
    dbm_doc,
)
from tests.protocol._docs import in_order, term
from tests.protocol._env import (
    CORPUS_DIR,
    FIXTURES,
    build_env,
    write_rot_fixture,
    steps_of,
)

EVAL_SPLIT = "inline#eval"
#: Three rows, so an eval forward (batch 3) is distinguishable from a
#: minibatch forward (batch 2) and from the whole-role forward (batch 4).
EVAL_ROWS = [
    {
        "input": "the tallest tree in the forest is",
        "counterfactual_inputs": ["the deepest lake in the valley is"],
        "label": " one",
    },
    {
        "input": "nine purple kites drift over",
        "counterfactual_inputs": ["two rusty bicycles lean against"],
        "label": " two",
    },
    {
        "input": "her grandmother's kitchen always smelled of",
        "counterfactual_inputs": ["his uncle's workshop always sounded like"],
        "label": " three",
    },
]
B = 2  # minibatches: 4 training rows at pairs=2
DATA_IDENTITY = {
    "base": "inline#input",
    "counterfactual": "inline#counterfactual_inputs[0]",
}


class _InlineDatasets:
    """The eval split, served in memory: ``rows`` is what the train loop
    reads; the other two are the resolver contract's pure-verb half."""

    def __init__(self, splits: dict[str, list[dict[str, Any]]]) -> None:
        self._splits = splits

    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return tuple(self._splits[ref][0]) if self._splits.get(ref) else ()

    def rows(self, ref: str) -> list[dict[str, Any]]:
        return self._splits[ref]


def _request() -> RunContext:
    return RunContext(
        env=ResolutionEnv(
            datasets=_InlineDatasets({EVAL_SPLIT: EVAL_ROWS}), artifacts=None
        ),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )


def _train_doc(
    kind: str = "das", *, seed: int = 0, epochs: int = 3, pairs: int = 2
) -> dict:
    raw = das_doc(seed=seed, epochs=epochs) if kind == "das" else dbm_doc()
    if kind == "dbm":
        raw["method"]["train"]["seed"] = seed
        raw["method"]["train"]["steps"] = {"epochs": epochs}
    raw["method"]["train"]["batch"] = {"pairs": pairs}
    # the eval scores the objective's cross-entropy, under its saved label
    (ce,) = [e for e in raw["method"]["save"] if e.get("file_path") == "ce.json"]
    raw["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": EVAL_SPLIT,
        "aggregations": {"ce": term(ce["read"], ce["model"], ce["aggregation"])},
    }
    return raw


def _campaign(
    raw: dict, cache: ForwardCache | None = None
) -> tuple[Document, Interning]:
    """One point's handle on a campaign cache, built the way
    ``execute_request`` builds it: plan digests plus the tap union."""
    doc = parse_document(in_order(raw))
    plan = plan_point(doc, data_identity=DATA_IDENTITY)
    if cache is None:
        cache = ForwardCache(
            wanted={
                g.key: tuple(doc.sites[t.site] for t in g.taps) for g in plan.groups
            }
        )
    return doc, Interning(
        keys={(g.model, g.input): g.key for g in plan.groups}, cache=cache
    )


@contextlib.contextmanager
def _batch_sizes(bundle: ModelBundle) -> Iterator[list[int]]:
    """Every top-level forward of the model, by the batch it saw."""
    seen: list[int] = []
    handle = bundle.model.register_forward_pre_hook(
        lambda _m, _args, kwargs: seen.append(int(kwargs["input_ids"].shape[0])),
        with_kwargs=True,
    )
    try:
        yield seen
    finally:
        handle.remove()


def _fit(
    raw: dict, bundle: ModelBundle, *, interning: Interning | None
) -> tuple[Any, PointExecutor, list[int]]:
    executor = executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        interning=interning,
    )
    with _batch_sizes(bundle) as sizes:
        outcome = run_training(executor.doc, executor, _request())
    return outcome, executor, sizes


def _weights(outcome: Any) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in outcome.stages.items()
        for slot, param in stage.slot_params().items()
    }


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.mark.unit
class TestFitInterning:
    def test_a_fit_runs_each_constant_group_once_per_row_slice(self, bundle):
        """``B + B·E + 1 + E``: the source forward once per minibatch and once
        for the eval split; the patched forward every step and every eval."""
        epochs = 3
        _doc, interning = _campaign(_train_doc(epochs=epochs))
        _outcome, _ex, sizes = _fit(
            _train_doc(epochs=epochs), bundle, interning=interning
        )
        assert len(sizes) == B + B * epochs + 1 + epochs
        assert sizes.count(len(EVAL_ROWS)) == 1 + epochs  # eval: one source + E patched
        assert sizes.count(B) == B + B * epochs  # minibatch: B source + B·E patched
        cache = interning.cache
        # the passes the fit's inner executors paid for constant groups; the
        # campaign's own tally is untouched — inner passes are not `forwards`
        assert len(cache.inner_executed) == B + 1
        assert cache.executed == []

    def test_without_interning_every_step_reruns_both_groups(self, bundle):
        """The reference path, pinned so the saving above is measured against
        something: ``interning=None`` means no reuse at all."""
        epochs = 3
        _outcome, _ex, sizes = _fit(_train_doc(epochs=epochs), bundle, interning=None)
        assert len(sizes) == 2 * B * epochs + 2 * epochs

    def test_a_second_point_pays_only_the_groups_the_fit_changes(self, bundle):
        """Another seed on the same cache: its source slices and eval split are
        already captured, so it pays ``B·E + E`` — the patched forwards alone."""
        epochs = 2
        _doc, first = _campaign(_train_doc(seed=0, epochs=epochs))
        _fit(_train_doc(seed=0, epochs=epochs), bundle, interning=first)
        _doc, second = _campaign(_train_doc(seed=1, epochs=epochs), first.cache)
        _outcome, _ex, sizes = _fit(
            _train_doc(seed=1, epochs=epochs), bundle, interning=second
        )
        assert len(sizes) == B * epochs + epochs
        assert sizes.count(len(EVAL_ROWS)) == epochs

    def test_a_different_batch_size_adds_new_slices_and_runs_them(self, bundle):
        """Slices are part of the key: ``pairs=1`` on a cache filled at
        ``pairs=2`` finds no ``(digest, (i,))`` captures and runs those four,
        while the eval split — the same rows — is still served."""
        epochs = 2
        doc, first = _campaign(_train_doc(epochs=epochs))
        _fit(_train_doc(epochs=epochs), bundle, interning=first)
        _doc, single = _campaign(_train_doc(epochs=epochs, pairs=1), first.cache)
        _outcome, _ex, sizes = _fit(
            _train_doc(epochs=epochs, pairs=1), bundle, interning=single
        )
        source = first.keys[("original_counterfactual", "counterfactual")]
        assert len(sizes) == 4 + 4 * epochs + epochs
        assert sizes.count(len(EVAL_ROWS)) == epochs
        for i in range(4):
            assert (source, (i,)) in first.cache.captured
        assert (source, (0, 1)) in first.cache.captured
        assert (source, (2, 3)) in first.cache.captured

    @pytest.mark.parametrize("kind", ["das", "dbm"])
    def test_interning_changes_no_weight(self, bundle, kind):
        """Serving the source capture from the cache is exactly what re-running
        the frozen forward would produce, so the fitted weights — and the eval
        score the loop selected on — are bit-identical."""
        _doc, interning = _campaign(_train_doc(kind))
        interned, _ex, _s = _fit(_train_doc(kind), bundle, interning=interning)
        plain, _ex, _s = _fit(_train_doc(kind), bundle, interning=None)
        a, b = _weights(interned), _weights(plain)
        assert set(a) == set(b) == ({"rot.weight"} if kind == "das" else {"gate.theta"})
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0.0, rtol=0.0)
        assert interned.eval_score is not None and plain.eval_score is not None
        assert interned.eval_score.metrics == plain.eval_score.metrics
        assert interned.eval_score.passes == plain.eval_score.passes

    def test_the_trained_model_is_never_cached(self, bundle):
        """Negative space. After a fit no key carries the ``patched`` digest;
        a grad-enabled executor handed the *unfiltered* campaign handle still
        publishes nothing for ``patched``; and a grad-free inner executor
        (the eval pass) is gated the same way — grad alone is not the test."""
        raw = _train_doc(epochs=1)
        _doc, interning = _campaign(raw)
        _outcome, executor, _s = _fit(raw, bundle, interning=interning)
        cache = interning.cache
        patched = interning.keys[("patched", "base")]
        source = interning.keys[("original_counterfactual", "counterfactual")]

        def key_of(capture_key: Any) -> str:
            """The group key of a bare or sliced ``CaptureKey``."""
            return capture_key if isinstance(capture_key, str) else capture_key[0]

        assert all(key_of(k) != patched for k in cache.captured)
        assert any(key_of(k) == source for k in cache.captured)

        grad = PointExecutor(
            executor.doc,
            bundle,
            role_rows=executor.role_rows,
            role_fields=executor.role_fields,
            load_tensors=executor.load_tensors,
            stage_cache=executor.stage_cache,
            grad_enabled=True,
            interning=interning,  # unfiltered: carries the patched key
        )
        assert grad._may_intern("original_counterfactual")
        assert not grad._may_intern("patched")
        grad.run_all()
        assert all(key_of(k) != patched for k in cache.captured)

        inner = PointExecutor(
            executor.doc,
            bundle,
            role_rows=executor.role_rows,
            role_fields=executor.role_fields,
            load_tensors=executor.load_tensors,
            stage_cache=executor.stage_cache,
            grad_enabled=False,
            interning=Interning(
                keys=interning.keys, cache=cache, rows="probe", counted=False
            ),
        )
        assert inner._may_intern("original_counterfactual")
        assert not inner._may_intern("patched")
        inner.run_all()
        assert all(key_of(k) != patched for k in cache.captured)
        assert (source, "probe") in cache.captured
        # neither inner executor counted toward the campaign's forwards
        assert patched not in cache.executed

    def test_the_eval_split_has_its_own_key_and_its_own_rows(self, bundle):
        """``(key, split)`` is distinct from every minibatch slice, holds a
        capture over the split's rows, and is read once — the score it yields
        is the un-interned score."""
        epochs = 2
        _doc, interning = _campaign(_train_doc(epochs=epochs))
        interned, _ex, _s = _fit(_train_doc(epochs=epochs), bundle, interning=interning)
        plain, _ex, _s = _fit(_train_doc(epochs=epochs), bundle, interning=None)
        source = interning.keys[("original_counterfactual", "counterfactual")]
        captured = interning.cache.captured
        assert {(source, EVAL_SPLIT), (source, (0, 1)), (source, (2, 3))} <= set(
            captured
        )
        for tensor in captured[(source, EVAL_SPLIT)].values():
            assert tensor.shape[0] == len(EVAL_ROWS)
        for tensor in captured[(source, (0, 1))].values():
            assert tensor.shape[0] == B
        assert interned.eval_score is not None and plain.eval_score is not None
        assert interned.eval_score.metrics == plain.eval_score.metrics

    def test_one_executor_per_minibatch_and_one_for_eval(self, bundle, monkeypatch):
        """The eval executor is built on the first pass and kept for the fit
        — not rebuilt per pass — so the split is read and tokenized once, and
        its source forward runs once across the whole fit."""
        epochs = 3
        constructed: list[Interning | None] = []
        real_init = PointExecutor.__init__

        def spy(self: PointExecutor, *args: Any, **kwargs: Any) -> None:
            constructed.append(kwargs.get("interning"))
            real_init(self, *args, **kwargs)

        _doc, interning = _campaign(_train_doc(epochs=epochs))
        executor = executor_for(
            _train_doc(epochs=epochs),
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
            interning=interning,
        )
        monkeypatch.setattr(PointExecutor, "__init__", spy)
        with _batch_sizes(bundle) as sizes:
            run_training(executor.doc, executor, _request())
        assert len(constructed) == B + 1
        assert all(inner is not None and not inner.counted for inner in constructed)
        assert {inner.rows for inner in constructed if inner is not None} == {
            (0, 1),
            (2, 3),
            EVAL_SPLIT,
        }
        assert sizes.count(len(EVAL_ROWS)) == 1 + epochs


# --------------------------------------------------------------------------- #
# end to end: corpus 08 through the engine
# --------------------------------------------------------------------------- #

#: Corpus 08 at tiny scale: one ``k`` (so the sweep is over seed alone, three
#: points), two epochs, one pair per batch. The fixture's train split has two
#: rows, so ``B = 2``; its test split is the eval split.
SWEEP_OVERRIDES = {
    "model.key": TINY_LLAMA,
    "model.dtype": "fp32",
    "sites.target.layers": 1,
    "featurizers.rot.k": 4,
    "train.steps": {"epochs": 2},
    "train.batch": {"pairs": 1},
}


@pytest.fixture()
def counted_forwards() -> Iterator[list[int]]:
    """Every top-level call of the model the engine will run — ``load_model``
    is memoized on this exact call form (see ``test_forward_interning``)."""
    bundle = load_model(
        TINY_LLAMA, "main", dtype="fp32", device="cpu", quantization=None
    )
    calls: list[int] = []
    handle = bundle.model.register_forward_pre_hook(lambda _m, _a: calls.append(1))
    try:
        yield calls
    finally:
        handle.remove()


@pytest.mark.smoke
def test_a_swept_fit_shares_its_constant_forwards_across_points(
    tmp_path: Path, counted_forwards: list[int]
) -> None:
    root = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
    env = build_env(root)
    loaded = compile_protocol(
        CORPUS_DIR / "08_weekdays_das_sweep_im.json", env=env, overrides=SWEEP_OVERRIDES
    )
    docs = steps_of(loaded, env).documents
    assert len(docs) == 3
    train_rows = len(env.datasets.rows("weekdays/data#train"))
    pairs = SWEEP_OVERRIDES["train.batch"]["pairs"]
    b = -(-train_rows // pairs)
    e = SWEEP_OVERRIDES["train.steps"]["epochs"]

    out = tmp_path / "out"
    result = PytorchHooksEngine().execute(loaded, RunContext(output_dir=out, env=env))

    # The three points fit as one cohort (§4 "Cohorts"): every optimizer step
    # is ONE forward over the three minibatches together, every eval pass one
    # forward over the split three times. The source slices still run once
    # each (b distinct slices, plus the eval split's), whichever member asks
    # first; then each point's own two whole-role groups, the source shared.
    cohort_steps = b * e
    cohort_evals = e
    assert len(counted_forwards) == (b + 1) + cohort_steps + cohort_evals + 1 + len(
        docs
    )
    # the campaign's own tally is unchanged: inner passes of a fit are not
    # forward groups (RunResult.forwards)
    assert result.forwards == len(
        interned_groups(campaign_plans(docs, steps_of(loaded, env).canonical))
    )
    # ...but the saving is visible per point in the run's summaries: what
    # the fit's inner executors ran for constant groups, and what they were
    # served instead of running. Which member pays for a shared slice depends
    # on the seeds' minibatch orders, so the cohort's totals are the claim:
    # b + 1 runs across the cohort, and every other constant pass served
    tallies = [summary["fit_forwards"] for summary in result.summaries]
    assert sum(t["run"] for t in tallies) == b + 1
    assert sum(t["served"] for t in tallies) == len(docs) * (b * e + e) - (b + 1)

    fitted = load_file(str(out / "rot.safetensors"))
    assert sorted(fitted) == [f"weight[seed={s}]" for s in (0, 1, 2)]
    names = sorted(fitted)
    for i, a in enumerate(names):
        for b_name in names[i + 1 :]:
            qa, qb = fitted[a], fitted[b_name]
            assert float((qa @ qa.T - qb @ qb.T).norm()) > 0.5
