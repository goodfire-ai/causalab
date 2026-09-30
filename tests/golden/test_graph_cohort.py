"""A cohort captured as one CUDA graph against the eager cohort, on real Qwen.

The fixed-layout capture (``graph_cohort.py``) is pinned three ways: with
every slot full it is the eager cohort's arithmetic exactly; with a padded
slot the padding rows change nothing but bf16 rounding; and the loop's
bookkeeping — one replay per step, no recapture when a member stops early,
the eager cohort after an out-of-memory capture — holds.
"""

from __future__ import annotations

# The assertions inspect replay buffers, not a second executor.
# pyright: reportPrivateUsage=false

import copy
import math
import os
import sys
from unittest.mock import patch

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import graph_cohort, train
from causalab.neural.engines.pytorch_hooks.cuda_graphs import Replay, make_executor
from causalab.neural.engines.pytorch_hooks.graph_cohort import CohortGraphs
from causalab.neural.engines.pytorch_hooks.graph_reuse import FitGraphCache
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.execution import campaign_cache
from causalab.neural.shared.executor import Interning
from causalab.protocol.engine import RunContext
from causalab.neural.shared.plan import plan_point
from causalab.io.env import FileArtifacts, ResolutionEnv
from tests.golden.test_cuda_graphs import Datasets, document
from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.protocol._docs import saved
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    ce_term,
)

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

MODEL = os.environ.get("CAUSALAB_CUDA_GRAPH_MODEL", "Qwen/Qwen3.6-35B-A3B")

# six rows: at pairs=3 two whole minibatches, at pairs=4 one whole and a
# remainder of two
BASE6 = BASES + [BASES[0].replace("quick", "slow"), BASES[1].replace("green", "brown")]
CF6 = COUNTERFACTUALS + [
    COUNTERFACTUALS[0].replace("cold", "warm"),
    COUNTERFACTUALS[1].replace("bright", "cold"),
]
ANSWERS6 = ANSWERS + ANSWERS[:2]


@pytest.fixture(scope="module")
def bundle():
    return load_model(MODEL, "main", dtype="bf16", device="cuda", quantization=None)


def _docs(pairs: int, epochs: int = 3):
    """A DAS point at layer 18, a DAS point at layer 12 and a DBM point at
    layer 18 — members at different layers, DAS beside DBM."""
    docs = []
    for method, layer in (("das", 18), ("das", 12), ("dbm", 18)):
        raw = document(method)
        raw["method"]["sites"]["tgt"]["layers"] = [layer]
        raw["method"]["train"]["batch"] = {"pairs": pairs}
        raw["method"]["train"]["steps"] = {"epochs": epochs}
        raw["method"]["train"]["eval"] = {
            "split": "eval",
            "every": {"epochs": 1},
            "aggregations": {"ce": ce_term()},
        }
        docs.append(raw)
    return docs


# The early-stop fixture. The two tests it serves assert the loop's
# bookkeeping around a stop — no recapture, stale slots, one eval capture —
# so the stop is arranged to follow from the code rather than from a loss
# curve: ``_evaluate`` counts the first eval as an improvement (``best`` is
# None) and every later eval that does not beat it as stale, so ``mode: max``
# on ``ce`` — which only falls — is stale from the second eval on and stops a
# member at pass ``patience + 2`` for any curve whose later evals never beat
# the first (pinned on CPU by ``test_train.py::
# test_max_mode_on_a_falling_metric_stops_at_patience_plus_two``). The first
# eval is after one epoch, so the premise is
# ``max(ce_2, ce_3, ce_4) <= ce_1``: here the DAS members' ``ce`` goes
# 8.85 → 2.68 → 0.068 → 0.028 (layer 18) and 8.96 → 3.99 → 1.23 → 0.049
# (layer 12) in the captured frame, and 8.95 → 2.73 and 8.98 → 4.01 over the
# first two evals of the eager cohort under an auto budget (one CUDA run of
# this fixture on ``MODEL`` in bf16) — the comparison
# that matters is the second eval's against the first, in the steepest part
# of the descent. The DBM member authors no ``early_stop`` and runs its whole
# budget by construction. Both facts hold in the captured frame and in the
# eager cohort's whatever their bf16 scores, so the two cohorts stop on the
# same pass. (An earlier fixture stopped on a real ``ce`` rise; a later kernel
# change moved it onto the budget's last eval.)
EARLY_STOP_BUDGET = 6
EARLY_STOP_PATIENCE = 2
EARLY_STOP = {"on": "ce", "mode": "max", "patience": EARLY_STOP_PATIENCE}
# the stop has to land strictly inside the budget, or the tests below assert
# their identities over a cohort in which nothing ever stopped
assert EARLY_STOP_PATIENCE + 2 < EARLY_STOP_BUDGET
#: ``_docs`` orders the members DAS, DAS, DBM; the DAS members stop, the DBM
#: runs its budget
EARLY_STOP_PASSES = [EARLY_STOP_PATIENCE + 2] * 2 + [EARLY_STOP_BUDGET]


def _early_stop_docs():
    raws = _docs(pairs=3, epochs=EARLY_STOP_BUDGET)
    for raw in raws[:2]:
        raw["method"]["train"]["early_stop"] = EARLY_STOP
    return raws


DATA_IDENTITY = {
    "base": "inline#input",
    "counterfactual": "inline#counterfactual_inputs[0]",
}


def _executors(raws, bundle, *, graphs: bool, store: bool = False):
    """The points' executors; with ``store`` they share one campaign cache,
    built the way ``execute_request`` builds it (as the cohort tests do)."""
    references = [
        executor_for(
            raw,
            bundle,
            base_texts=BASE6,
            counterfactual_texts=CF6,
            extra_columns={"label": ANSWERS6},
        )
        for raw in raws
    ]
    handles = [None] * len(raws)
    if store:
        docs = [reference.doc for reference in references]
        plans = [plan_point(doc, data_identity=DATA_IDENTITY) for doc in docs]
        cache = campaign_cache(docs, plans)
        handles = [
            Interning(
                keys={(g.model, g.input): g.key for g in plan.groups},
                cache=cache,
            )
            for plan in plans
        ]
    return [
        make_executor(
            reference.doc,
            bundle,
            cuda_graphs=graphs,
            role_rows=reference.role_rows,
            role_fields=reference.role_fields,
            load_tensors=reference.load_tensors,
            interning=handle,
        )
        for reference, handle in zip(references, handles, strict=True)
    ]


def _request(tmp_path) -> RunContext:
    return RunContext(
        env=ResolutionEnv(datasets=Datasets(), artifacts=FileArtifacts(tmp_path)),
        output_dir=tmp_path,
    )


class _Trace:
    """Every optimizer update's gradients and parameters, every eval score,
    every model call's batch size, and the captures/replays."""

    def __init__(self, bundle):
        self.updates: list[list[torch.Tensor]] = []
        self.scores: list[dict[str, float]] = []
        self.batch_sizes: list[int] = []
        self.replays = 0
        self.captures = 0
        self.eval_captures = 0
        self.eval_replays = 0
        self.handle = bundle.model.register_forward_pre_hook(
            lambda _m, args, kwargs: self.batch_sizes.append(
                int(
                    (
                        kwargs.get("input_ids")
                        if kwargs.get("input_ids") is not None
                        else args[0]
                    ).shape[0]
                )
            ),
            with_kwargs=True,
        )

    def __enter__(self):
        build_optimizer = train._build_optimizer
        score = train._score
        replay_call = Replay.__call__
        replay_init = Replay.__init__
        trace = self

        def traced_optimizer(spec, groups):
            optimizer = build_optimizer(spec, groups)
            parameters = [p for g in optimizer.param_groups for p in g["params"]]
            step = optimizer.step

            def traced_step(*a, **k):
                grads = [
                    p.grad.detach().cpu().clone() if p.grad is not None else None
                    for p in parameters
                ]
                result = step(*a, **k)
                trace.updates.append(
                    grads + [p.detach().cpu().clone() for p in parameters]
                )
                return result

            optimizer.step = traced_step
            return optimizer

        def traced_score(doc, worker):
            result = score(doc, worker)
            trace.scores.append(dict(result))
            return result

        def counted_call(self):
            if self.parameters:
                trace.replays += 1
            else:
                trace.eval_replays += 1
            return replay_call(self)

        def counted_init(self, *a, **k):
            replay_init(self, *a, **k)
            if self.parameters:
                trace.captures += 1
            else:
                trace.eval_captures += 1

        self._patches = [
            patch.object(train, "_build_optimizer", traced_optimizer),
            patch.object(train, "_score", traced_score),
            patch.object(Replay, "__call__", counted_call),
            patch.object(Replay, "__init__", counted_init),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        self.handle.remove()


def _fit(
    raws,
    bundle,
    tmp_path,
    *,
    graphs: bool,
    fit_rows: int | None = None,
    store: bool = False,
):
    """``fit_rows`` pins the eager cohort's windows: an auto budget probes its
    first step with the first member alone, a different batch shape from the
    captured frame, so an exact comparison authors the bound. ``store`` gives
    the points a shared campaign cache (shared sources, prefix resume)."""
    executors = _executors(raws, bundle, graphs=graphs, store=store)
    cache = FitGraphCache() if graphs else None
    with _Trace(bundle) as trace:
        try:
            outcomes = train.run_cohort_training(
                [e.doc for e in executors],
                executors,
                _request(tmp_path),
                fit_rows=fit_rows,
                graph_cache=cache,
            )
        finally:
            if cache is not None:
                cache.close()
    return outcomes, trace


def _assert_stages_close(actual, expected, **tolerance):
    for got, want in zip(actual, expected, strict=True):
        for name, stage in got.stages.items():
            torch.testing.assert_close(
                stage.state_dict(), want.stages[name].state_dict(), **tolerance
            )


def test_full_slots_replay_the_eager_cohort_exactly(bundle, tmp_path):
    """Six rows at pairs=3: every minibatch is whole, so the captured frame
    is the eager cohort's frame and its arithmetic — updates, parameters and
    eval scores bit for bit."""
    raws = _docs(pairs=3)
    expected, eager = _fit(raws, bundle, tmp_path / "eager", graphs=False, fit_rows=9)
    assert eager.replays == 0
    actual, graphs = _fit(raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=9)
    assert graphs.captures == 1, "one capture for the whole cohort"
    # three members × two minibatches × three epochs = six steps, every one a replay
    assert graphs.replays == 6
    assert len(graphs.updates) == len(eager.updates) == 18
    for got, want in zip(graphs.updates, eager.updates, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert graphs.scores == eager.scores
    _assert_stages_close(actual, expected, rtol=0, atol=0)
    # the eager cohort called the model once per step and once per eval; the
    # capture's warm-up and recording are the only model calls of the graph
    # run's fits (frozen sources aside), and replays are not model calls
    cohort_rows = 3 * 3
    assert eager.batch_sizes.count(cohort_rows) == 6
    assert graphs.batch_sizes.count(cohort_rows) == 2  # warm-up + capture


@pytest.mark.parametrize("variable_lengths", [False, True])
def test_with_the_store_full_slots_still_replay_the_eager_cohort_exactly(
    bundle, tmp_path, monkeypatch, variable_lengths
):
    """The production shape: the points share a campaign cache. The eager
    cohort then serves every member's source from one pass per row slice and
    resumes below the shallowest write; the captured cohort copies those
    constants out of the store and resumes from a buffer of its own — and is
    still the eager cohort's arithmetic bit for bit."""
    if variable_lengths:
        monkeypatch.setattr(
            sys.modules[__name__],
            "BASE6",
            [BASE6[0] + " next to the very tall tree", *BASE6[1:]],
        )
        monkeypatch.setattr(
            sys.modules[__name__],
            "CF6",
            [*CF6[:2], CF6[2] + " beside the lake", *CF6[3:]],
        )
    raws = _docs(pairs=3)
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=9, store=True
    )
    actual, graphs = _fit(
        raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=9, store=True
    )
    assert graphs.captures == 1
    assert graphs.replays == 6
    for got, want in zip(graphs.updates, eager.updates, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert graphs.scores == eager.scores
    _assert_stages_close(actual, expected, rtol=0, atol=0)
    # the store shares the source pass across the three members: one 3-row
    # source forward per slice in either run, so the graph run's 3-row
    # forwards equal the eager run's
    assert graphs.batch_sizes.count(3) == eager.batch_sizes.count(3)
    # 9-row model calls: warm-up and capture, plus one un-intervened pass per
    # composition whose prefix the store lacked (the two slices, epoch one),
    # against the eager cohort's six steps
    assert eager.batch_sizes.count(9) == 6
    assert graphs.batch_sizes.count(9) == 4


def test_with_the_store_the_captured_forward_resumes(bundle, tmp_path):
    """Members at layers 12 and 18 resume at block 12: the blocks below never
    run in a replay, and the prefix passes stop at block 12."""
    raws = _docs(pairs=3)
    executors = _executors(raws, bundle, graphs=True, store=True)
    fires = [[] for _ in bundle.blocks]
    handles = [
        block.register_forward_pre_hook(
            lambda _m, args, kwargs, i=i: fires[i].append(
                int((args[0] if args else kwargs["hidden_states"]).shape[0])
            ),
            with_kwargs=True,
        )
        for i, block in enumerate(bundle.blocks)
    ]
    try:
        with _Trace(bundle) as trace:
            train.run_cohort_training(
                [e.doc for e in executors], executors, _request(tmp_path), fit_rows=9
            )
    finally:
        for handle in handles:
            handle.remove()
    assert trace.replays == 6
    # A replay is not a model call, so no hook fires in one. Of the 9-row
    # calls: the two prefix passes run blocks 0..11 and stop at block 12 (the
    # stop hook is registered after this test's, so block 12 still counts
    # them); the warm-up and the capture resume at block 12 — blocks 0..11
    # are swapped out for them — and run 12..last.
    assert fires[0].count(9) == 2
    assert fires[11].count(9) == 2
    assert fires[12].count(9) == 4
    assert fires[13].count(9) == 2
    assert fires[-1].count(9) == 2


def _distance(left, right) -> float:
    """The largest absolute difference over two lists of tensors."""
    return max(
        float((a.float() - b.float()).abs().max())
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    )


def _solo_fits(raws, bundle, tmp_path):
    """Each point fitted alone, eagerly — the reference the eager cohort is
    itself measured against (spec §4: equal to the rounding of a different
    batch shape)."""
    outcomes, traces = [], []
    for i, raw in enumerate(raws):
        (executor,) = _executors([raw], bundle, graphs=False)
        with _Trace(bundle) as trace:
            outcomes.append(
                train.run_training(executor.doc, executor, _request(tmp_path / str(i)))
            )
        traces.append(trace)
    return outcomes, traces


def test_padded_slots_change_only_rounding(bundle, tmp_path):
    """Six rows at pairs=4: the second minibatch has two rows in a slot of
    four. The padding rows carry zero weight, so what they change is the batch
    shape the MoE layers dispatch — the same kind of drift the eager cohort
    already has against solo fits (its documented contract). The bound pinned
    here is that one: the padded replay strays from the eager cohort by no
    more than a small multiple of what the eager cohort strays from the solo
    fits, on the first step's gradients, on the fitted parameters and on the
    first epoch's eval scores."""
    raws = _docs(pairs=4)
    expected, eager = _fit(raws, bundle, tmp_path / "eager", graphs=False, fit_rows=12)
    actual, graphs = _fit(raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=12)
    solo, alone = _solo_fits(raws, bundle, tmp_path / "solo")
    assert graphs.captures == 1
    assert graphs.replays == 6
    assert len(graphs.updates) == len(eager.updates) == 18
    # the first optimizer step: `current` steps the members in cohort order,
    # so updates[i] is member i's first update in either cohort run
    for i in range(3):
        cohort_vs_solo = _distance(eager.updates[i], alone[i].updates[0])
        graph_vs_cohort = _distance(graphs.updates[i], eager.updates[i])
        assert graph_vs_cohort <= 3 * cohort_vs_solo + 1e-6, (
            f"member {i}: padded replay strays {graph_vs_cohort:.3e} from the eager "
            f"cohort, which strays {cohort_vs_solo:.3e} from its solo fit"
        )
    for got, want, ref in zip(actual, expected, solo, strict=True):
        for name, stage in got.stages.items():
            state = [t for t in stage.state_dict().values()]
            want_state = [t for t in want.stages[name].state_dict().values()]
            ref_state = [t for t in ref.stages[name].state_dict().values()]
            cohort_vs_solo = _distance(want_state, ref_state)
            graph_vs_cohort = _distance(state, want_state)
            assert graph_vs_cohort <= 3 * cohort_vs_solo + 1e-6, (
                f"{name}: padded replay strays {graph_vs_cohort:.3e} from the eager "
                f"cohort, which strays {cohort_vs_solo:.3e} from the solo fit"
            )
    # eval scores: `_evaluate` scores the due members in cohort order, one
    # eval per epoch, so score e*3+i is member i's e-th eval in either cohort
    # run and alone[i].scores[e] the solo fit's. The rounding bound is pinned
    # on the first epoch's scores only: they score the parameters the bound
    # above holds. Later epochs compound bf16 rounding through a three-row fit
    # whose loss is far from converged. Comparing ratios of later drifts
    # cannot isolate the replay's rounding error.
    assert len(graphs.scores) == len(eager.scores) == 9
    for k, (got, want) in enumerate(zip(graphs.scores, eager.scores, strict=True)):
        ref = alone[k % 3].scores[k // 3]
        assert got.keys() == want.keys() == ref.keys()
        for name in got:
            for label, series in (("graphs", got), ("eager", want), ("solo", ref)):
                assert math.isfinite(series[name]), (
                    f"eval {k}: {label} {name} is {series[name]!r}"
                )
    for k in range(3):
        got, want, ref = graphs.scores[k], eager.scores[k], alone[k].scores[0]
        for name in got:
            graph_vs_cohort = abs(got[name] - want[name])
            cohort_vs_solo = abs(want[name] - ref[name])
            assert graph_vs_cohort <= 3 * cohort_vs_solo + 1e-3, (
                f"eval {k}: {name} strays {graph_vs_cohort:.3e} from the eager cohort, "
                f"which strays {cohort_vs_solo:.3e} from the solo fit"
            )
    # the captured frame is every slot: 3 × 4 rows, whatever the minibatches held
    assert graphs.batch_sizes.count(12) == 2


def test_a_member_that_stops_early_keeps_its_slot(bundle, tmp_path):
    """The DAS members drop out at pass four while the DBM goes on
    (``_early_stop_docs``). No recapture; the loop never steps a dropped
    member again — its slot replays stale until the cohort's budget ends.
    (The returned stages are another matter: under ``mode: max`` the first
    eval is the best, so ``_finish`` restores the epoch-1 snapshot.)"""
    raws = _early_stop_docs()
    actual, graphs = _fit(raws, bundle, tmp_path / "graphs", graphs=True)
    assert graphs.captures == 1
    passes = [outcome.eval_score.passes for outcome in actual]
    assert passes == EARLY_STOP_PASSES, f"passes {passes}"
    assert graphs.replays == 2 * max(passes)
    # a stopped member's slot replays stale: the loop never steps it again
    assert len(graphs.updates) == 2 * sum(passes)


def test_an_unsupported_capture_layout_falls_back_to_the_eager_cohort(bundle, tmp_path):
    raws = _docs(pairs=3)
    expected, eager = _fit(raws, bundle, tmp_path / "eager", graphs=False, fit_rows=9)
    with patch.object(graph_cohort, "_single_role", return_value=None):
        actual, graphs = _fit(
            raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=9
        )
    assert graphs.captures == graphs.replays == 0
    assert graphs.scores == eager.scores
    for got, want in zip(graphs.updates, eager.updates, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_slots_are_capped_at_the_dataset_size(bundle, tmp_path):
    # Six real rows per member fit in 18 rows, despite authored pairs=100.
    raws = _docs(pairs=100)
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=18, store=True
    )
    actual, graphs = _fit(
        raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=18, store=True
    )
    assert graphs.captures == 1
    assert graphs.replays == 3
    assert graphs.scores == eager.scores
    for got, want in zip(graphs.updates, eager.updates, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_an_oom_capture_falls_back_to_the_eager_cohort(bundle, tmp_path):
    raws = _docs(pairs=3)
    expected, eager = _fit(raws, bundle, tmp_path / "eager", graphs=False, fit_rows=9)
    real_capture = CohortGraphs._capture

    def failing_capture(self, window):
        real_capture(self, window)
        raise torch.OutOfMemoryError("simulated")

    with patch.object(CohortGraphs, "_capture", failing_capture):
        actual, graphs = _fit(
            raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=9
        )
    assert graphs.replays == 0
    # the cohort still steps together: one 9-row forward per step
    assert graphs.batch_sizes.count(9) >= 6
    for got, want in zip(graphs.updates, eager.updates, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_a_partial_mask_stage_is_released_before_eager_fallback(bundle, tmp_path):
    raws = _docs(pairs=3)
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=9, store=True
    )
    stage = CohortGraphs._stage
    failed_banks = []

    def mismatched_stage(self, slot, prepared):
        stage(self, slot, prepared)
        failed_banks.append(self)
        raise graph_cohort._LayoutMismatch("simulated after partially staging masks")

    with patch.object(CohortGraphs, "_stage", mismatched_stage):
        actual, graphs = _fit(
            raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=9, store=True
        )
    assert len(failed_banks) == 1  # the next step must not retry the broken bank
    assert failed_banks[0].disabled and failed_banks[0].replay is None
    assert graphs.replays == 0
    assert graphs.scores == eager.scores
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_a_mixed_cohort_keeps_eager_batching(bundle, tmp_path):
    """A member the graph path refuses (a JS objective) keeps the whole
    cohort on the eager batched path."""
    raws = _docs(pairs=3)
    # a JS aggregation is enough to keep a document off the graph path: the
    # clean logits are read on the un-intervened model on base (§2.9)
    raws[1]["method"]["reads"]["base_logits"] = {
        "site": "lm_head",
        "pos": {"index": -1},
    }
    raws[1]["method"]["intervened_models"]["original_base"]["reads"].append(
        "base_logits"
    )
    raws[1]["method"]["save"].append(
        saved(
            "logits",
            "patched",
            "js.json",
            {"kind": "js", "target": {"read": "base_logits", "model": "original_base"}},
        )
    )
    executors = _executors(raws, bundle, graphs=True)
    kinds = {type(e).__name__ for e in executors}
    assert kinds == {"GraphExecutor", "PointExecutor"}
    assert graph_cohort.cohort_graph_reason(executors, None, [3, 3, 3]) is not None
    with _Trace(bundle) as trace:
        train.run_cohort_training(
            [e.doc for e in executors], executors, _request(tmp_path), fit_rows=9
        )
    assert trace.captures == 0
    assert trace.batch_sizes.count(9) == 6


def test_a_bounded_cohort_keeps_eager_windows(bundle, tmp_path):
    raws = _docs(pairs=3)
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=6, store=True
    )
    actual, graphs = _fit(
        raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=6, store=True
    )
    assert graphs.captures == 0
    assert graphs.batch_sizes == eager.batch_sizes
    assert graphs.scores == eager.scores
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_repeated_eval_is_captured_with_exact_scores(bundle, tmp_path):
    raws = _docs(pairs=3, epochs=5)
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=100, store=True
    )
    actual, graphs = _fit(
        raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=100, store=True
    )
    assert graphs.eval_captures == 1
    assert graphs.eval_replays == 4
    assert graphs.scores == eager.scores
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_an_eval_capture_oom_keeps_eager_scores(bundle, tmp_path):
    raws = _docs(pairs=3)
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=9, store=True
    )
    capture = graph_cohort.EvaluationGraphs._capture
    failed = []

    def oom(self):
        capture(self)
        failed.append(self)
        raise torch.OutOfMemoryError("simulated eval capture OOM")

    with patch.object(graph_cohort.EvaluationGraphs, "_capture", oom):
        actual, graphs = _fit(
            raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=9, store=True
        )
    assert len(failed) == 1
    assert failed[0].disabled and failed[0].bank is None
    assert not failed[0].frames
    assert graphs.eval_replays == 0
    assert graphs.scores == eager.scores
    _assert_stages_close(actual, expected, rtol=0, atol=0)


def test_repeated_padded_steps_only_copy_prepared_inputs(bundle, tmp_path):
    step = CohortGraphs._step
    repeated = 0

    def checked_step(self, window):
        nonlocal repeated
        ready = all(
            id(item.minibatch) in self.slots[self.by_key[item.key]].prepared
            for item in window
        )
        if ready:
            repeated += 1
            with (
                patch.object(
                    graph_cohort,
                    "pad_rows",
                    side_effect=AssertionError("padding during replay"),
                ),
                patch.object(
                    graph_cohort.GraphExecutor,
                    "prepare_batch",
                    side_effect=AssertionError("mask preparation during replay"),
                ),
            ):
                return step(self, window)
        return step(self, window)

    with patch.object(CohortGraphs, "_step", checked_step):
        _fit(_docs(pairs=4), bundle, tmp_path, graphs=True, fit_rows=100, store=True)
    assert repeated >= 4


def test_copy_of_the_eager_cohort_is_not_disturbed_by_padding_code(bundle):
    """Off the graph path nothing here runs: the eager loop never imports a
    weight into its objective."""
    raws = _docs(pairs=4)
    executors = _executors(raws, bundle, graphs=False)
    objective = train.TrainingObjective(executors[0], {})
    assert objective.weight is None
    assert copy.copy(objective).weight is None


def test_an_early_stop_keeps_the_eval_capture(bundle, tmp_path):
    """The DAS members drop out at pass four while the DBM goes on
    (``_early_stop_docs``): the eval layout is captured once and replayed for
    the members still due — a stopped member's slot replays stale and is
    never scored again — and while every member is due the replayed scores
    are the eager cohort's exactly. After the stops the eager cohort
    evaluates a smaller frame, so from there the two agree only to bf16
    rounding."""
    raws = _early_stop_docs()
    expected, eager = _fit(
        raws, bundle, tmp_path / "eager", graphs=False, fit_rows=100, store=True
    )
    actual, graphs = _fit(
        raws, bundle, tmp_path / "graphs", graphs=True, fit_rows=100, store=True
    )
    passes = [
        outcome.eval_score.passes
        for outcome in actual
        if outcome.eval_score is not None
    ]
    assert passes == EARLY_STOP_PASSES, f"passes {passes}"
    eager_passes = [
        outcome.eval_score.passes
        for outcome in expected
        if outcome.eval_score is not None
    ]
    assert eager_passes == EARLY_STOP_PASSES, f"eager passes {eager_passes}"
    assert graphs.eval_captures == 1
    assert graphs.eval_replays == max(passes) - 1
    assert len(graphs.scores) == sum(passes)
    # the same frame while every member is due: the same scores, bit for bit
    full = 3 * min(passes)
    assert graphs.scores[:full] == eager.scores[:full]
