"""The eval passes of a captured cohort as one graph, replayed for whichever
members are still due (``graph_cohort.EvaluationGraphs``).

Off CUDA there is no capture, so the graph is stood in for: ``Replay`` runs
its work eagerly on every call, so the "captured" computation *is* the eager
one and what these tests hold is the loop's bookkeeping around it — the
layout laid out and prepared once, captured once on its second full pass,
replayed for the due members after a member early-stops (never a recapture,
never a fall back to eager, never a score for the stopped member), the eager
first pass on the prepared frame rather than a transient one (no transient
frame ever prepared), a single member still due served from its slot, and the
fit's pool — the training graph's — handed to the capture, every eval
executor's reads released once the round scored them. The scores are held to the plain
eager loop's: exactly while every member is due, and to the eager cohort's
rounding once the eager frame has shrunk and the captured one has not. A
cohort evaluating on two splits is the one layout's limit: the first
split's set is captured and served, the second's runs eagerly per forward,
and both are the eager loop to the bit. A row bound that falls below the
layout's group gives the capture back: the remaining passes are the eager
windowed ones, exact, with no recapture. The capture itself is pinned on
CUDA in ``tests/golden/test_graph_cohort.py``.
"""

# the loop's internals and the cohort suite's builders are used directly
# pyright: reportPrivateUsage=false

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Sequence

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import cuda_graphs as cuda_graphs_module
from causalab.neural.engines.pytorch_hooks import graph_cohort
from causalab.neural.engines.pytorch_hooks import train as train_module
from causalab.neural.engines.pytorch_hooks.cuda_graphs import (
    GraphExecutor,
    GraphPool,
    make_executor,
)
from causalab.neural.engines.pytorch_hooks.graph_cohort import EvaluationGraphs
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.protocol.engine import RunContext
from causalab.io.env import ResolutionEnv

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_fit_cohort import (
    B,
    EVAL_ROWS,
    EVAL_SPLIT,
    _assert_same_fit,
    _batch_sizes,
    _campaign,
    _InlineDatasets,
    _request,
    _train_doc,
)
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
)

pytestmark = pytest.mark.unit

EVAL = len(EVAL_ROWS)
SECOND_SPLIT = "inline#eval_b"


def _two_split_request() -> RunContext:
    """The eval split beside a second table of the same rows, reversed."""
    return RunContext(
        env=ResolutionEnv(
            datasets=_InlineDatasets(
                {EVAL_SPLIT: EVAL_ROWS, SECOND_SPLIT: list(reversed(EVAL_ROWS))}
            ),
            artifacts=None,  # type: ignore[arg-type]
        ),
        output_dir=None,  # type: ignore[arg-type]
    )


class _FakeReplay:
    """A capture that runs its work eagerly on every call."""

    inits: list[Any] = []
    #: whether each capture asked for a warm-up pass, in ``inits`` order
    warmups: list[bool] = []
    calls: int = 0
    #: the fit's pool, as the training step saw it (``graphs_on``)
    training_pool: Any = None

    def __init__(
        self,
        work: Callable[[], Any],
        stages: dict[str, Any],
        *,
        device: torch.device,
        parameters: list[torch.nn.Parameter] | None = None,
        pool: Any = None,
        warmup: bool = True,
    ) -> None:
        type(self).inits.append(pool)
        type(self).warmups.append(warmup)
        self.work = work
        self.parameters = list(parameters or [])
        self.output = work()
        self.gradients = [p.grad for p in self.parameters]
        self.graph = SimpleNamespace(pool=lambda: "eval-pool")

    def __call__(self) -> Any:
        type(self).calls += 1
        self.output = self.work()
        return self.output


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture
def graphs_on(monkeypatch: pytest.MonkeyPatch) -> type[_FakeReplay]:
    """Graph mode on the CPU: every executor graph-eligible, the cohort
    captured (its training step declining every window, so the steps are
    the eager cohort's and only the eval path differs), the capture faked."""
    _FakeReplay.inits = []
    _FakeReplay.warmups = []
    _FakeReplay.calls = 0
    _FakeReplay.training_pool = None
    monkeypatch.setattr(
        cuda_graphs_module, "unsupported_reason", lambda doc, bundle, collective: None
    )
    monkeypatch.setattr(
        train_module, "cohort_graph_reason", lambda executors, fit_rows, pairs: None
    )

    def declined(self: Any, window: Any) -> bool:
        # the pool the training step would capture into: the fit's GraphPool,
        # the one the eval capture must share
        _FakeReplay.training_pool = self.pool
        return False

    monkeypatch.setattr(graph_cohort.CohortGraphs, "backward", declined)
    monkeypatch.setattr(graph_cohort, "Replay", _FakeReplay)
    return _FakeReplay


def _executors(raws: Sequence[dict[str, Any]], bundle: ModelBundle) -> list[Any]:
    _docs, handles = _campaign(raws)
    out = []
    for raw, handle in zip(raws, handles, strict=True):
        reference = executor_for(
            raw,
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
        )
        out.append(
            make_executor(
                reference.doc,
                bundle,
                cuda_graphs=True,
                role_rows=reference.role_rows,
                role_fields=reference.role_fields,
                load_tensors=reference.load_tensors,
                interning=handle,
            )
        )
    return out


@contextlib.contextmanager
def _scored(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[int, list[dict]]]:
    """Every score the loop takes, by the document's seed, in order."""
    score = train_module._score
    scores: dict[int, list[dict]] = {}

    def recording(doc: Any, executor: Any) -> dict[str, float]:
        result = score(doc, executor)
        scores.setdefault(int(doc.raw["method"]["train"]["seed"]), []).append(
            dict(result)
        )
        return result

    monkeypatch.setattr(train_module, "_score", recording)
    try:
        yield scores
    finally:
        monkeypatch.setattr(train_module, "_score", score)


def _fit(
    raws: Sequence[dict[str, Any]],
    bundle: ModelBundle,
    monkeypatch: pytest.MonkeyPatch,
    *,
    eval_graphs: bool,
    request: RunContext | None = None,
) -> tuple[list[Any], list[Any], list[int], dict[int, list[dict]]]:
    executors = _executors(raws, bundle)
    if not eval_graphs:
        monkeypatch.setattr(train_module, "EvaluationGraphs", lambda **kw: None)
    with _scored(monkeypatch) as scores, _batch_sizes(bundle) as sizes:
        outcomes = train_module.run_cohort_training(
            [ex.doc for ex in executors],
            executors,
            _request() if request is None else request,
        )
    # the round released every eval executor's reads once it scored them
    assert all(not ex.eval_executor._read_values for ex in executors)
    return outcomes, executors, sizes, scores


def _docs(epochs: int = 6) -> list[dict[str, Any]]:
    """A member that stops at its first non-improving eval (seed 0 at rank 4,
    the configuration ``test_fit_cohort`` pins as stopping) beside two that
    run their budget; three seeds, so the scores are keyed per member."""
    return [
        _train_doc(
            seed=0,
            k=4,
            epochs=epochs,
            early_stop={"on": "ce", "patience": 0, "mode": "min"},
        ),
        _train_doc(seed=1, k=2, epochs=epochs),
        _train_doc(seed=2, k=8, epochs=epochs),
    ]


class TestEvalGraphs:
    def test_the_eval_capture_skips_the_warmup_and_the_loop_borrows_a_pool(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        """The evaluation graph is captured on the layout's second pass,
        after the first ran the same forward on the same frame: no warm-up
        pass. Handed the engine's pool (``graph_pool``), the loop captures
        the step and the eval graphs into it and leaves it open for the next
        fit; a closed pool is not borrowed — the loop opens one of its own
        for the fit and releases it after."""
        raws = _docs()
        pool = GraphPool()
        executors = _executors(raws, bundle)
        train_module.run_cohort_training(
            [ex.doc for ex in executors], executors, _request(), graph_pool=pool
        )
        assert graphs_on.training_pool is pool
        assert graphs_on.inits == [pool]
        assert graphs_on.warmups == [False]
        assert not pool.closed
        pool.close()
        graphs_on.inits.clear()
        graphs_on.warmups.clear()
        executors = _executors(raws, bundle)
        train_module.run_cohort_training(
            [ex.doc for ex in executors], executors, _request(), graph_pool=pool
        )
        own = graphs_on.training_pool
        assert isinstance(own, GraphPool) and own is not pool
        assert graphs_on.inits == [own]
        assert own.closed  # opened by the loop, released with the fit

    def test_scores_are_the_eager_passes_and_a_stopped_member_keeps_its_slot(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        epochs = 6
        raws = _docs(epochs)
        eager, _e, eager_sizes, eager_scores = _fit(
            raws, bundle, monkeypatch, eval_graphs=False
        )
        monkeypatch.setattr(train_module, "EvaluationGraphs", EvaluationGraphs)
        actual, _a, sizes, scores = _fit(raws, bundle, monkeypatch, eval_graphs=True)
        passes = [o.eval_score.passes for o in actual]  # type: ignore[union-attr]
        stopped = passes[0]
        assert stopped < epochs, "the stopper never stopped — nothing to hold"
        assert passes == [o.eval_score.passes for o in eager]  # type: ignore[union-attr]
        # one capture, on the second pass; a replay on every pass after it,
        # including the ones with a member gone
        assert isinstance(graphs_on.training_pool, GraphPool)
        assert graphs_on.inits == [graphs_on.training_pool]
        assert graphs_on.calls == max(passes) - 1
        # every score is the eager pass's: exactly while every member is due
        # (the same frame), to the eager cohort's rounding once the eager
        # frame has shrunk and the captured one has not
        for seed, got in scores.items():
            want = eager_scores[seed]
            assert len(got) == len(want)
            for epoch, (a, b) in enumerate(zip(got, want, strict=True)):
                if epoch < stopped:
                    assert a == b
                else:
                    for name, value in a.items():
                        assert value == pytest.approx(b[name], rel=1e-4)
        # the stopped member was scored exactly as many times as it passed
        assert len(scores[0]) == stopped
        # the fits are the eager loop's, to the byte
        for got, want in zip(actual, eager, strict=True):
            _assert_same_fit(got, want, atol=0.0, rtol=0.0)
        # the eval frame never shrinks: the layout's first pass, the capture's
        # own run and every replay are the full three members' rows, while
        # the eager loop ran the two remaining members once the first stopped
        # (a two-member eval frame is 2·EVAL rows — so are the three members'
        # grad steps before the stop, B per epoch, in both runs)
        assert sizes.count(3 * EVAL) == max(passes) + 1
        assert sizes.count(2 * EVAL) == B * stopped
        assert eager_sizes.count(2 * EVAL) == B * stopped + (max(passes) - stopped)

    def test_two_splits_capture_the_first_and_run_the_second_eagerly(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        """One layout per cohort: members evaluating on two splits are two
        due sets every round. The first split's set is the layout — laid
        out on its first pass (an eager pass on the prepared frame), captured
        on its second, replayed on every later one — and the second split's
        set, disjoint from it, is never laid out: every one of its passes is
        the eager one, as an eager cohort runs it, the layout untouched.
        Scores and fits are the plain eager loop's to the bit."""
        epochs = 4
        raws = [
            _train_doc(seed=0, k=4, epochs=epochs),
            _train_doc(seed=1, k=2, epochs=epochs),
            _train_doc(seed=2, k=8, epochs=epochs),
        ]
        raws[2]["method"]["train"]["eval"]["split"] = SECOND_SPLIT
        eager, _e, _es, eager_scores = _fit(
            raws, bundle, monkeypatch, eval_graphs=False, request=_two_split_request()
        )
        served: list[tuple[int, bool]] = []
        framed: list[tuple[int, bool]] = []
        forward, frames_for = EvaluationGraphs.forward, EvaluationGraphs.frames_for

        def watched_forward(self: EvaluationGraphs, members: Any) -> bool:
            result = forward(self, members)
            served.append((len(members), result))
            return result

        def watched_frames(self: EvaluationGraphs, members: Any) -> Any:
            result = frames_for(self, members)
            framed.append((len(members), result is not None))
            return result

        monkeypatch.setattr(EvaluationGraphs, "forward", watched_forward)
        monkeypatch.setattr(EvaluationGraphs, "frames_for", watched_frames)
        monkeypatch.setattr(train_module, "EvaluationGraphs", EvaluationGraphs)
        actual, _a, _s, scores = _fit(
            raws, bundle, monkeypatch, eval_graphs=True, request=_two_split_request()
        )
        # one capture, of the first split's two members; a replay per later pass
        assert isinstance(graphs_on.training_pool, GraphPool)
        assert graphs_on.inits == [graphs_on.training_pool]
        assert graphs_on.calls == epochs - 1
        # per round the two-member set then the one-member set: the first
        # is eager once (laid out) and replayed after, the second eager always
        assert served == [(2, False), (1, False)] + [(2, True), (1, False)] * (
            epochs - 1
        )
        # the eager passes' frames: the layout's own for its first pass, none
        # for the other split's, ever
        assert framed == [(2, True)] + [(1, False)] * epochs
        passes = [o.eval_score.passes for o in actual]  # type: ignore[union-attr]
        assert passes == [epochs] * 3
        assert passes == [o.eval_score.passes for o in eager]  # type: ignore[union-attr]
        assert scores == eager_scores
        for got, want in zip(actual, eager, strict=True):
            _assert_same_fit(got, want, atol=0.0, rtol=0.0)

    def test_a_bound_fallen_below_the_layout_gives_the_capture_back(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        """From round ``shrink_at`` the eval budget's bound is what a device
        that ran short leaves: two members' rows, not three. The layout no
        longer runs as one window, so the capture is released and the loop
        stays eager and windowed to the end — no recapture at the smaller
        size — and every score is the eager loop's under the same bound."""
        epochs, shrink_at = 6, 4
        raws = [
            _train_doc(seed=0, k=4, epochs=epochs),
            _train_doc(seed=1, k=2, epochs=epochs),
            _train_doc(seed=2, k=8, epochs=epochs),
        ]
        evaluate = train_module._evaluate
        rounds = {"n": 0}

        def shrinking(due: Any, request: Any, budget: Any, **kwargs: Any) -> None:
            rounds["n"] += 1
            if rounds["n"] >= shrink_at:
                budget.bound = 2 * EVAL  # re-seeded down after a training OOM
            return evaluate(due, request, budget, **kwargs)

        monkeypatch.setattr(train_module, "_evaluate", shrinking)
        eager, _e, _es, eager_scores = _fit(
            raws, bundle, monkeypatch, eval_graphs=False
        )
        rounds["n"] = 0
        built: list[EvaluationGraphs] = []

        def factory(**kwargs: Any) -> EvaluationGraphs:
            built.append(EvaluationGraphs(**kwargs))
            return built[-1]

        monkeypatch.setattr(train_module, "EvaluationGraphs", factory)
        actual, _a, _s, scores = _fit(raws, bundle, monkeypatch, eval_graphs=True)
        # one capture (eval round 2), replayed until the bound fell; then given
        # back, disabled, and never recaptured
        assert isinstance(graphs_on.training_pool, GraphPool)
        assert graphs_on.inits == [graphs_on.training_pool]
        assert graphs_on.calls == shrink_at - 2
        assert len(built) == 1 and built[0].bank is None and built[0].disabled
        passes = [o.eval_score.passes for o in actual]  # type: ignore[union-attr]
        assert passes == [epochs] * 3
        assert passes == [o.eval_score.passes for o in eager]  # type: ignore[union-attr]
        assert scores == eager_scores
        for got, want in zip(actual, eager, strict=True):
            _assert_same_fit(got, want, atol=0.0, rtol=0.0)

    def test_the_first_pass_runs_on_the_prepared_frame_without_a_signature(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        """The layout's frame is prepared once, host-side, and registered on
        the lead: no eval executor ever prepares a transient frame
        (``_model_forward``'s per-shape path for a batch it did not prepare
        — its record is read as each one closes, since ``close()`` clears
        it), and no pass rebuilds the frame — the memory the eval holds is
        fixed at the first pass."""
        prepared: list[int] = []
        prepare_batch = GraphExecutor.prepare_batch
        transient_at_close: dict[int, int] = {}
        close = GraphExecutor.close

        def counted(self: Any, batch: Any) -> Any:
            prepared.append(int(batch.input_ids.shape[0]))
            return prepare_batch(self, batch)

        def closing(self: Any) -> None:
            transient_at_close[id(self)] = len(self._transient)
            close(self)

        monkeypatch.setattr(GraphExecutor, "prepare_batch", counted)
        monkeypatch.setattr(GraphExecutor, "close", closing)
        _outcomes, executors, _sizes, _scores = _fit(
            _docs(), bundle, monkeypatch, eval_graphs=True
        )
        evaluators = [ex.eval_executor for ex in executors]
        assert all(isinstance(ev, GraphExecutor) for ev in evaluators)
        assert [transient_at_close[id(ev)] for ev in evaluators] == [0, 0, 0]
        # and the layout's frame itself once: a transient frame of the same
        # rows would have prepared them again
        assert prepared.count(3 * EVAL) == 1

    def test_one_member_still_due_is_served_from_its_slot(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        """Two stoppers beside a runner: once both have stopped the runner is
        due alone, and its pass is still the replay — never a pass of its own
        (``_run_eval``) — and every eval forward is the full frame."""
        epochs = 8
        raws = [
            _train_doc(
                seed=0,
                k=4,
                epochs=epochs,
                early_stop={"on": "ce", "patience": 0, "mode": "min"},
            ),
            _train_doc(
                seed=1,
                k=4,
                epochs=epochs,
                early_stop={"on": "ce", "patience": 0, "mode": "min"},
            ),
            _train_doc(seed=2, k=8, epochs=epochs),
        ]
        solo: list[Any] = []
        run_eval = train_module._run_eval

        def counted(*args: Any, **kwargs: Any) -> Any:
            solo.append(args)
            return run_eval(*args, **kwargs)

        monkeypatch.setattr(train_module, "_run_eval", counted)
        actual, _e, sizes, scores = _fit(raws, bundle, monkeypatch, eval_graphs=True)
        passes = [o.eval_score.passes for o in actual]  # type: ignore[union-attr]
        assert max(passes[:2]) < epochs, "a stopper never stopped — nobody due alone"
        assert passes[2] == epochs
        assert not solo
        assert isinstance(graphs_on.training_pool, GraphPool)
        assert graphs_on.inits == [graphs_on.training_pool]
        assert graphs_on.calls == epochs - 1
        assert sizes.count(3 * EVAL) == epochs + 1
        assert sizes.count(EVAL) == 1  # the split's one source forward
        assert [len(scores[seed]) for seed in (0, 1, 2)] == passes

    def test_the_replay_is_entered_outside_the_round_s_featurizer_scope(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, graphs_on
    ) -> None:
        """A capture's passes evaluate every stage inside the graph
        (``cuda_graphs.captured_pass``, an isolated scope); the loop must
        not hand it a value shared in from eager code, so the eval round
        enters the replay before it opens its own ``featurizer_cache``
        scope — every ``forward`` call sees no open scope."""
        from causalab.neural.shared.featurizers import sharing

        open_scopes: list[int] = []
        forward = EvaluationGraphs.forward

        def watched(self: EvaluationGraphs, members: Any) -> bool:
            open_scopes.append(sharing._SCOPE.depth)  # pyright: ignore[reportPrivateUsage]
            return forward(self, members)

        monkeypatch.setattr(EvaluationGraphs, "forward", watched)
        actual, _e, _sizes, _scores = _fit(
            _docs(), bundle, monkeypatch, eval_graphs=True
        )
        assert open_scopes, "the eval round never reached the graph"
        assert set(open_scopes) == {0}
        assert graphs_on.calls == max(o.eval_score.passes for o in actual) - 1  # type: ignore[union-attr]
