"""Points of one campaign fit **together** (spec §4, "Cohorts").

The DAS and DBM steps of a swept study run one fit per point, and each fit's
optimizer step is one intervened forward over one minibatch. The rows of
those forwards are independent — nothing in the model couples them — so the
points of a cohort (one realization, one row set, one frame) step in
lockstep, and every step is **one** model call over the concatenation of
their minibatches, each member's writes on its own rows at its own address.
This file pins the arithmetic — ``P`` members pay one forward per step, not
``P`` — and the two properties that make it safe: the fitted weights equal
the sequential fits (to the rounding of a different batch shape), and every
member keeps its own seed, order, schedule and early-stop decision.

Ground truth for "a forward happened" is a pre-hook on the loaded model, as
in ``test_train_interning.py``; the batch size it sees tells a cohort step
(``P·pairs`` rows) from a solo one (``pairs`` rows).
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.cohort import Entry, batchable
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.train import (
    run_cohort_training,
    run_training,
)
from causalab.neural.shared.execution import campaign_cache
from causalab.neural.shared.executor import ForwardCache, Interning
from causalab.protocol.engine import RunContext
from causalab.neural.shared.plan import plan_point
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import Document, parse_document

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    _drawn_executor,
    das_doc,
    dbm_doc,
)
from tests.protocol._docs import in_order, term

EVAL_SPLIT = "inline#eval"
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
PAIRS = 2
B = 2  # minibatches: 4 training rows at pairs=2
DATA_IDENTITY = {
    "base": "inline#input",
    "counterfactual": "inline#counterfactual_inputs[0]",
}


class _InlineDatasets:
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
            datasets=_InlineDatasets({EVAL_SPLIT: EVAL_ROWS}),
            artifacts=None,  # type: ignore[arg-type]
        ),
        output_dir=None,  # type: ignore[arg-type]
    )


def _train_doc(
    kind: str = "das",
    *,
    seed: int = 0,
    epochs: int = 3,
    layer: int = 1,
    k: int = 4,
    eval_every: int | None = 1,
    early_stop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = das_doc(seed=seed, epochs=epochs) if kind == "das" else dbm_doc()
    method = raw["method"]
    if kind == "dbm":
        method["train"]["seed"] = seed
        method["train"]["steps"] = {"epochs": epochs}
    else:
        method["featurizers"]["rot"]["k"] = k
    method["sites"]["tgt"]["layers"] = [layer]
    method["train"]["batch"] = {"pairs": PAIRS}
    if eval_every is None:
        method["train"].pop("eval", None)
        method["train"].pop("early_stop", None)
    else:
        method["train"]["eval"] = {
            "every": {"epochs": eval_every},
            "split": EVAL_SPLIT,
            "aggregations": {"ce": _ce_term(raw)},
        }
        if early_stop is not None:
            method["train"]["early_stop"] = early_stop
        else:
            method["train"].pop("early_stop", None)
    return raw


def _ce_term(raw: dict[str, Any]) -> dict[str, Any]:
    """The document's cross-entropy as an objective or eval term: the
    aggregation its ``ce.json`` save entry tabulates, over the same read."""
    (ce,) = [e for e in raw["method"]["save"] if e.get("file_path") == "ce.json"]
    return term(ce["read"], ce["model"], ce["aggregation"])


def _campaign(
    raws: Sequence[dict[str, Any]], cache: ForwardCache | None = None
) -> tuple[list[Document], list[Interning]]:
    """The points' handles on one campaign cache, built the way
    ``execute_request`` builds them."""
    docs = [parse_document(in_order(raw)) for raw in raws]
    plans = [plan_point(doc, data_identity=DATA_IDENTITY) for doc in docs]
    if cache is None:
        cache = campaign_cache(docs, plans)
    return docs, [
        Interning(keys={(g.model, g.input): g.key for g in plan.groups}, cache=cache)
        for plan in plans
    ]


def _executor(
    raw: dict[str, Any], bundle: ModelBundle, *, interning: Interning | None
) -> PointExecutor:
    return executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        interning=interning,
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


@contextlib.contextmanager
def _block_fires(bundle: ModelBundle) -> Iterator[list[list[int]]]:
    """Per decoder block, the batch size of every forward it ran."""
    fires: list[list[int]] = [[] for _ in bundle.blocks]
    handles = []
    for index, block in enumerate(bundle.blocks):

        def hook(_m: Any, args: tuple[Any, ...], _kw: Any, *, i: int = index) -> None:
            hidden = args[0] if args else _kw["hidden_states"]
            fires[i].append(int(hidden.shape[0]))

        handles.append(block.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield fires
    finally:
        for handle in handles:
            handle.remove()


def _fit_cohort(
    raws: Sequence[dict[str, Any]],
    bundle: ModelBundle,
    *,
    interned: bool = True,
    fit_rows: int | None = None,
) -> tuple[list[Any], list[PointExecutor], list[int], ForwardCache | None]:
    _docs, handles = _campaign(raws)
    executors = [
        _executor(raw, bundle, interning=handle if interned else None)
        for raw, handle in zip(raws, handles)
    ]
    with _batch_sizes(bundle) as sizes:
        outcomes = run_cohort_training(
            [ex.doc for ex in executors], executors, _request(), fit_rows=fit_rows
        )
    return outcomes, executors, sizes, handles[0].cache if interned else None


def _fit_alone(
    raw: dict[str, Any], bundle: ModelBundle, *, interned: bool = True
) -> tuple[Any, list[int]]:
    _docs, (handle,) = _campaign([raw])
    executor = _executor(raw, bundle, interning=handle if interned else None)
    with _batch_sizes(bundle) as sizes:
        outcome = run_training(executor.doc, executor, _request())
    return outcome, sizes


def _weights(outcome: Any) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in outcome.stages.items()
        for slot, param in stage.slot_params().items()
    }


def _assert_same_fit(a: Any, b: Any, *, atol: float, rtol: float) -> None:
    wa, wb = _weights(a), _weights(b)
    assert set(wa) == set(wb)
    for name in wa:
        torch.testing.assert_close(wa[name], wb[name], atol=atol, rtol=rtol)
    assert (a.eval_score is None) == (b.eval_score is None)
    if a.eval_score is not None:
        assert a.eval_score.passes == b.eval_score.passes
        assert a.eval_score.selected == b.eval_score.selected
        for metric, value in a.eval_score.metrics.items():
            assert value == pytest.approx(b.eval_score.metrics[metric], rel=1e-4)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.mark.unit
class TestCohortArithmetic:
    def test_a_cohort_pays_one_forward_per_step_and_per_eval(self, bundle):
        """Three DAS points at one layer with three ranks, ``B`` minibatches
        and ``E`` epochs: the source slices run once each and the eval source
        once (§4 "Fits"); then every step is ONE forward of ``3·pairs`` rows
        and every eval pass ONE forward of ``3·|split|`` rows."""
        epochs = 3
        raws = [_train_doc(k=k, epochs=epochs) for k in (2, 4, 8)]
        _outcomes, _executors, sizes, cache = _fit_cohort(raws, bundle)
        assert cache is not None
        assert sizes.count(3 * PAIRS) == B * epochs
        assert sizes.count(3 * len(EVAL_ROWS)) == epochs
        # the constant passes: B source slices + the eval split's source
        assert sizes.count(PAIRS) == B
        assert sizes.count(len(EVAL_ROWS)) == 1
        assert len(sizes) == B + 1 + B * epochs + epochs
        assert len(cache.inner_executed) == B + 1
        # the cohort's tallies attribute every constant pass to a member
        tallies = [outcome.fit_forwards for outcome in _outcomes]
        assert all(t is not None for t in tallies)
        assert sum(t["run"] for t in tallies if t) == B + 1
        assert sum(t["served"] for t in tallies if t) == 3 * (B * epochs + epochs) - (
            B + 1
        )

    def test_a_cohort_of_one_is_the_solo_fit(self, bundle):
        raw = _train_doc(epochs=2)
        outcomes, _executors, sizes, _cache = _fit_cohort([raw], bundle)
        solo, solo_sizes = _fit_alone(raw, bundle)
        assert sizes == solo_sizes
        _assert_same_fit(outcomes[0], solo, atol=0.0, rtol=0.0)

    def test_members_at_different_layers_share_the_forward(self, bundle):
        """A write at layer 0 beside a write at layer 1: each member's writer
        lands on its own rows at its own block, and the step is still one
        forward. Every block runs once per step, for both members' rows."""
        epochs = 2
        raws = [_train_doc(layer=0, epochs=epochs), _train_doc(layer=1, epochs=epochs)]
        with _block_fires(bundle) as fires:
            _outcomes, _executors, sizes, _cache = _fit_cohort(raws, bundle)
        assert sizes.count(2 * PAIRS) == B * epochs
        # block 1 sees every forward; a cohort step is one of them, with both
        # members' rows
        assert fires[1].count(2 * PAIRS) == B * epochs

    def test_members_with_different_seeds_bring_their_own_rows(self, bundle):
        """Seeds 0 and 1 draw different minibatch orders, so at one step the
        two members hold *different* rows of the shared frame; the forward is
        still one, over both members' rows."""
        epochs = 2
        raws = [_train_doc(seed=0, epochs=epochs), _train_doc(seed=1, epochs=epochs)]
        outcomes, _executors, sizes, _cache = _fit_cohort(raws, bundle)
        assert sizes.count(2 * PAIRS) == B * epochs
        # the orders really differ (the premise of the test)
        orders = []
        for raw in raws:
            rng = torch.Generator().manual_seed(int(raw["method"]["train"]["seed"]))
            orders.append(
                [torch.randperm(B, generator=rng).tolist() for _ in range(epochs)]
            )
        assert orders[0] != orders[1]
        # and each member is the fit its seed alone would have produced
        for raw, outcome in zip(raws, outcomes):
            solo, _sizes = _fit_alone(raw, bundle)
            _assert_same_fit(outcome, solo, atol=1e-5, rtol=1e-4)

    def test_drawn_members_at_one_seed_and_two_objectives_take_the_same_members(
        self, bundle
    ):
        """The draw stream is ``train.seed``'s alone: two points at one seed
        that differ in something ``cohort_key`` leaves out — the objective
        here — are one cohort whose members take the identical member
        sequence, epoch for epoch. That is what makes an objective sweep over
        a drawn set a *paired* comparison (the objective differs, the draws do
        not), and it pins the seed as the stream's only key: folding the
        objective into ``_Drawn.rng`` fails here, folding it into
        ``cohort_key`` fails the one-forward count."""
        epochs = 3
        members = [[text, text + " again"] for text in COUNTERFACTUALS]
        raws = []
        for l1_weight in (0.01, 0.02):
            raw = _train_doc("dbm", seed=0, epochs=epochs, eval_every=None)
            raw["data"]["counterfactual"] = {
                **raw["data"]["counterfactual"],
                "field": "counterfactual_inputs",
                "draw": {"kind": "uniform"},
            }
            raw["method"]["train"]["objective"] = [
                [1.0, _ce_term(raw)],
                [l1_weight, {"l1": "gate"}],
            ]
            raws.append(raw)
        _docs, handles = _campaign(raws)
        executors = [
            _drawn_executor(raw, bundle, members, interning=handle)
            for raw, handle in zip(raws, handles)
        ]
        with _batch_sizes(bundle) as sizes:
            outcomes = run_cohort_training(
                [ex.doc for ex in executors], executors, _request()
            )
        assert sizes.count(2 * PAIRS) == B * epochs  # one cohort forward
        draws = [outcome.draws["counterfactual"]["members"] for outcome in outcomes]
        assert all(len(epoch_draws) == epochs for epoch_draws in draws)
        assert draws[0] == draws[1]

    def test_drawn_members_at_different_seeds_fit_as_one_cohort(self, bundle):
        """§2.2 ``draw`` in a cohort — reachable, since ``cohort_key`` leaves
        the seed out: a ``train.seed`` sweep over a drawn document is one
        cohort of drawn members, each drawing on its own stream. Their
        expanded frames are one width (every member encodes the same texts and
        ``EncodedBatch.select`` keeps it), so the forward is still one over
        both members' rows; the draws differ (a fact about seeds 0 and 1, not
        a probability), and each member's are the ones its seed alone would
        have taken. The campaign store is *on* here, and bypassed: a drawn
        minibatch's source slice is re-run every step — a cached source
        forward from epoch 1 would silently serve epoch 2."""
        epochs = 3
        members = [[text, text + " again"] for text in COUNTERFACTUALS]
        raws = []
        for seed in (0, 1):
            raw = _train_doc("dbm", seed=seed, epochs=epochs, eval_every=None)
            raw["data"]["counterfactual"] = {
                **raw["data"]["counterfactual"],
                "field": "counterfactual_inputs",
                "draw": {"kind": "uniform"},
            }
            raws.append(raw)
        # `DATA_IDENTITY` is a constant here, and since `resolved_field` it is
        # also what `_data_identity` stamps for these roles: a drawn role's
        # identity is `counterfactual_inputs[0]`, its `eval` member
        _docs, handles = _campaign(raws)
        executors = [
            _drawn_executor(raw, bundle, members, interning=handle)
            for raw, handle in zip(raws, handles)
        ]
        with _batch_sizes(bundle) as sizes:
            outcomes = run_cohort_training(
                [ex.doc for ex in executors], executors, _request()
            )
        assert sizes.count(2 * PAIRS) == B * epochs
        # the store bypass: every member re-runs its own source slice every
        # step despite the store (the store-less cohort's count, with a store)
        assert sizes.count(PAIRS) == 2 * B * epochs
        draws = [outcome.draws["counterfactual"]["members"] for outcome in outcomes]
        assert all(len(epoch_draws) == epochs for epoch_draws in draws)
        assert draws[0] != draws[1]
        for raw, taken, outcome in zip(raws, draws, outcomes):
            executor = _drawn_executor(raw, bundle, members)
            solo = run_training(executor.doc, executor, _request())
            assert solo.draws["counterfactual"]["members"] == taken
            # ...and fits the same weights: the concatenated batch and the
            # shared expanded-frame width perturb nothing
            _assert_same_fit(outcome, solo, atol=1e-5, rtol=1e-4)

    def test_fit_rows_packs_members_into_bounded_forwards(self, bundle):
        """Four members at ``pairs=2`` under ``fit_rows=4``: two forwards of
        four rows per step, never one of eight; under ``fit_rows=1`` (below a
        member's minibatch) every member runs alone — a minibatch is never
        split."""
        epochs = 1
        raws = [_train_doc(k=k, epochs=epochs, eval_every=None) for k in (1, 2, 3, 4)]
        _o, _e, sizes, _c = _fit_cohort(raws, bundle, fit_rows=4)
        assert sizes.count(2 * PAIRS) == 2 * B * epochs
        assert sizes.count(4 * PAIRS) == 0
        _o, _e, sizes, _c = _fit_cohort(raws, bundle, fit_rows=1)
        assert sizes.count(PAIRS) == B + 4 * B * epochs  # sources + solo steps
        assert sizes.count(2 * PAIRS) == 0

    def test_a_member_that_stops_early_leaves_the_batch(self, bundle):
        """One member has patience 0 and stops after its second eval; the
        other runs its full budget. The cohort forwards shrink to one member's
        rows once the first has dropped out, and the stopped member keeps the
        weights its early stop selected."""
        epochs = 4
        stopper = _train_doc(
            seed=0,
            epochs=epochs,
            early_stop={"on": "ce", "patience": 0, "mode": "min"},
        )
        runner = _train_doc(seed=1, epochs=epochs)
        outcomes, _executors, sizes, _cache = _fit_cohort([stopper, runner], bundle)
        solo_stopper, _s = _fit_alone(stopper, bundle)
        solo_runner, _s = _fit_alone(runner, bundle)
        _assert_same_fit(outcomes[0], solo_stopper, atol=1e-5, rtol=1e-4)
        _assert_same_fit(outcomes[1], solo_runner, atol=1e-5, rtol=1e-4)
        assert outcomes[0].eval_score is not None
        assert outcomes[0].eval_score.passes < epochs
        assert outcomes[1].eval_score is not None
        assert outcomes[1].eval_score.passes == epochs
        # steps after the stop are one member's: the tail of the run has
        # `pairs`-row forwards that are not source slices
        after_stop = epochs * B - outcomes[0].eval_score.passes * B
        assert sizes.count(PAIRS) == B + after_stop
        assert sizes.count(2 * PAIRS) == outcomes[0].eval_score.passes * B

    def test_members_with_different_budgets_drop_out_in_turn(self, bundle):
        raws = [
            _train_doc(k=2, epochs=1, eval_every=None),
            _train_doc(k=4, epochs=3, eval_every=None),
        ]
        _outcomes, _executors, sizes, _cache = _fit_cohort(raws, bundle)
        assert sizes.count(2 * PAIRS) == B  # one epoch together
        assert sizes.count(PAIRS) == B + 2 * B  # sources, then two solo epochs


@pytest.mark.unit
class TestCohortParity:
    @pytest.mark.parametrize("kind", ["das", "dbm"])
    def test_a_cohort_fits_what_each_member_would_have_fitted(self, bundle, kind):
        """Two members of one kind (different seeds; for DAS also different
        ranks) fitted together equal the two solo fits: the batch shape is the
        only difference, so the weights agree to rounding and the eval scores
        and selections agree exactly."""
        raws = [
            _train_doc(kind, seed=0, k=2),
            _train_doc(kind, seed=1, k=4),
        ]
        outcomes, _executors, _sizes, _cache = _fit_cohort(raws, bundle)
        for raw, outcome in zip(raws, outcomes):
            solo, _s = _fit_alone(raw, bundle)
            _assert_same_fit(outcome, solo, atol=1e-5, rtol=1e-4)

    def test_a_das_and_a_dbm_point_fit_together(self, bundle):
        """Different featurizer kinds, objectives and anneals are each
        member's own; the forward they share is the same."""
        raws = [_train_doc("das", epochs=2), _train_doc("dbm", epochs=2)]
        outcomes, _executors, sizes, _cache = _fit_cohort(raws, bundle)
        assert sizes.count(2 * PAIRS) == B * 2
        assert set(_weights(outcomes[0])) == {"rot.weight"}
        assert set(_weights(outcomes[1])) == {"gate.theta"}
        for raw, outcome in zip(raws, outcomes):
            solo, _s = _fit_alone(raw, bundle)
            _assert_same_fit(outcome, solo, atol=1e-5, rtol=1e-4)

    def test_without_a_store_the_cohort_still_batches(self, bundle):
        """``interning=None`` means no reuse of constant groups — the reference
        path — but the cohort forward is independent of the store."""
        epochs = 2
        raws = [_train_doc(k=2, epochs=epochs), _train_doc(k=4, epochs=epochs)]
        outcomes, _executors, sizes, cache = _fit_cohort(raws, bundle, interned=False)
        assert cache is None
        assert sizes.count(2 * PAIRS) == B * epochs
        # every member re-runs its own source slice every step (no store)
        assert sizes.count(PAIRS) == 2 * B * epochs
        for raw, outcome in zip(raws, outcomes):
            solo, _s = _fit_alone(raw, bundle, interned=False)
            _assert_same_fit(outcome, solo, atol=1e-5, rtol=1e-4)


@pytest.mark.unit
class TestCohortResume:
    def test_the_cohort_forward_resumes_below_the_shallowest_write(self, bundle):
        """Two members writing at layer 1: block 0 runs for the source slices
        and once per slice for the first cohort step, then never again — the
        cohort forward starts at block 1 from both members' prefixes
        concatenated."""
        epochs = 3
        raws = [_train_doc(k=2, epochs=epochs), _train_doc(k=4, epochs=epochs)]
        with _block_fires(bundle) as fires:
            _outcomes, _executors, sizes, cache = _fit_cohort(raws, bundle)
        assert cache is not None
        # block 1 sees every forward
        assert len(fires[1]) == len(sizes)
        # block 0: B source slices + eval source, and the first cohort pass
        # per slice and for the eval split — nothing after that
        assert sorted(fires[0]) == sorted(
            [PAIRS] * B + [len(EVAL_ROWS)] + [2 * PAIRS] * B + [2 * len(EVAL_ROWS)]
        )
        assert len(cache.resumed) == (B + 1) * (epochs - 1)
        assert all(depth == 1 for depth in cache.resumed)
        # each member's outcome saw every resumed cohort forward
        for outcome in _outcomes:
            assert len(outcome.resumed) == (B + 1) * (epochs - 1)

    def test_mixed_layers_resume_at_the_shallowest(self, bundle):
        """A layer-0 member beside a layer-1 member: block 0 must run for the
        first, so the cohort never resumes (there is no block below 0), and
        both members' prefixes are still stored for the point's own passes."""
        epochs = 2
        raws = [_train_doc(layer=0, epochs=epochs), _train_doc(layer=1, epochs=epochs)]
        with _block_fires(bundle) as fires:
            _outcomes, _executors, sizes, cache = _fit_cohort(raws, bundle)
        assert cache is not None
        assert len(fires[0]) == len(sizes)
        assert not cache.resumed


@pytest.mark.unit
class TestCohortBackend:
    def test_prefix_keys_carry_the_backend_the_forward_ran_under(self) -> None:
        """A cohort on an ``sdpa`` document whose members write at an
        attention-function interior runs its forwards under eager (the taps
        need it), so its prefixes must be keyed as eager — the 4-tuple — not
        tagged with the document's backend. Resolved outside the forward's
        stack they would carry ``"sdpa"``: a residual computed under eager
        filed where a genuine sdpa pass looks it up."""
        sdpa = load_model(TINY_LLAMA, attn_implementation="sdpa")
        assert sdpa.model.config._attn_implementation == "sdpa"
        raws = []
        for k in (2, 4):
            raw = _train_doc(k=k, epochs=2, layer=1)
            raw["model"]["attn_implementation"] = "sdpa"
            raw["method"]["sites"]["tgt"]["component"] = "attention_query"
            raws.append(raw)
        _docs, handles = _campaign(raws)
        executors = [_executor(raw, sdpa, interning=h) for raw, h in zip(raws, handles)]
        run_cohort_training([ex.doc for ex in executors], executors, _request())
        cache = handles[0].cache
        assert cache.resumed, "the cohort never resumed — the test measures nothing"
        assert cache.prefixes, "nothing stored"
        assert all(len(key) == 4 for key in cache.prefixes), sorted(
            len(key) for key in cache.prefixes
        )
        # the document's backend is back once the forwards have returned
        assert sdpa.model.config._attn_implementation == "sdpa"

    def test_an_interior_member_never_shares_a_forward_with_a_boundary_one(
        self,
    ) -> None:
        """One member at ``attention_query`` (needs eager) beside one at
        ``block_output`` (runs the document's backend): the plan puts them in
        separate cohorts, so the boundary member's forwards are never dragged
        onto eager by its neighbour."""
        from causalab.neural.shared.plan import fit_cohorts

        interior = _train_doc(k=2, epochs=1, layer=1)
        interior["method"]["sites"]["tgt"]["component"] = "attention_query"
        boundary = _train_doc(k=4, epochs=1, layer=1)
        docs = [parse_document(in_order(raw)) for raw in (interior, boundary)]
        assert fit_cohorts(docs, [DATA_IDENTITY, DATA_IDENTITY]) == ((0,), (1,))


@pytest.mark.unit
def test_batchable_names_what_the_cohort_forward_admits(bundle):
    raw = _train_doc()
    executor = _executor(raw, bundle, interning=None)
    assert batchable(Entry(executor, "patched", "base"))
    assert not batchable(Entry(executor, "original", "counterfactual"))
    # a grad-free executor outside a fit may be *served* the trained model's
    # capture, which is the store's business, not the cohort's
    _docs, (handle,) = _campaign([raw])
    point = _executor(raw, bundle, interning=handle)
    assert not batchable(Entry(point, "patched", "base"))
