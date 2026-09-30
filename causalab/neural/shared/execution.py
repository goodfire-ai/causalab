"""Execute protocol points and assemble metrics and output tables.

The driver uses the common executor surface for reads, row windows,
generated tokens, and model bundles. Engines supply an executor factory
and an optional training runner.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from causalab.causal.scoring import ScoringCheck
from causalab.neural.shared.executor import ForwardCache, Interning, PrefixPlan
from causalab.neural.shared.fires import group_label
from causalab.neural.shared.join import ShardOutput, join_shards
from causalab.neural.shared.featurizers import featurizer_cache
from causalab.neural.shared.metrics import score_metric
from causalab.io.results_io import write_outputs
from causalab.neural.shared.results import (
    MetricTable,
    TensorFile,
    _metric_cell,  # pyright: ignore[reportPrivateUsage]
    _row_exclusions,  # pyright: ignore[reportPrivateUsage]
    _summary_stat,  # pyright: ignore[reportPrivateUsage]
    _Windowed,  # pyright: ignore[reportPrivateUsage]
    _windowed_eligibility,  # pyright: ignore[reportPrivateUsage]
    rank_records,
)
from causalab.neural.shared.receipt import emit_run_events, run_events, write_run_record
from causalab.neural.shared.head import capture_spec
from causalab.protocol.identity import site_identity, spec_identity
from causalab.protocol.positions.encoding import generated_budget
from causalab.protocol.positions.ledger import ledger_records, wants_ledger
from causalab.protocol.positions.resolve import StepResolution, positions_key
from causalab.protocol.positions.roles import input_roles
from causalab.neural.shared.step_rules import check_steps
from causalab.neural.shared.sweep import enumerate_steps, sign_steps
from causalab.protocol.schema.explicit import canonical_model
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult, StepRecord
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.results import example_labels
from causalab.io.env import build_artifact_identity
from causalab.protocol.rules.data import check_fit_splits
from causalab.protocol.results import (
    Eligibility,
    Resolution,
    Unavailable,
    available,
    cell_key,
    cell_record,
    unavailable,
)
from causalab.protocol.publish import is_joiner, point_shard
from causalab.neural.shared.plan import (
    _data_identity,  # pyright: ignore[reportPrivateUsage]
    campaign_plans,
    fit_cohorts,
    GroupKey,
    interned_groups,
    PointPlan,
)
from causalab.protocol.lowering import lower_bands
from causalab.protocol.receipt import (
    RAGGED_KEY,
    RUN_RECORD_NAME,
    record_fires,
    record_measured_bounds,
    record_models,
    record_ragged_geometry,
)
from causalab.provenance import runtime_identity
from causalab.protocol.estimand import metric_record_identity
from causalab.protocol.schema import (
    WHOLE_WINDOW_METRIC_KINDS,
    Document,
    parse_document,
    read_is_vocabulary,
    ReadRef,
    SiteSpec,
)

__all__ = [
    "ExecutorSurface",
    "TrainEvalScore",
    "TrainOutcome",
    "TrainRunner",
    "campaign_cache",
    "campaign_plans",
    "execute_request",
    "featurizer_identity",
]

#: The run's progress lines, at INFO: point selection, each point's model
#: load, cohort fits, each point's run, the output write. Silent unless a
#: caller enables this logger — ``causalab run --verbose`` attaches a stderr
#: handler to this name alone ([`causalab.cli`][]). Nothing here reaches the
#: receipt, the event stream or any digest.
_log = logging.getLogger(__name__)


class ExecutorSurface(Protocol):
    """What [`execute_request`][] needs from a point executor."""

    bundle: Any

    def run_all(self) -> None: ...
    def read_value(self, ref: ReadRef) -> Any: ...
    def resolution(self, ref: ReadRef) -> Resolution: ...
    def row_resolutions(self, ref: ReadRef) -> list[Unavailable | None]: ...
    def dense_value(self, ref: ReadRef) -> Any: ...
    def dense_rows(self, ref: ReadRef, rows: Sequence[int]) -> Any: ...
    def windowed_value(self, ref: ReadRef) -> list[Any]: ...
    def generated_metric(self, metric: Any) -> list[list[Any]]: ...
    def is_generated(self, ref: ReadRef) -> bool: ...
    def addressed_steps(self, ref: ReadRef) -> list[list[int]]: ...
    def generated_ids(self, name: str) -> list[list[int]]: ...
    def rows_for_metrics(self) -> list[dict[str, Any]]: ...
    def check_scoring(self) -> ScoringCheck: ...
    def location_ledger(self) -> Any: ...


#: The run receipt's block for the table's string mode (spec §2.2): per base
#: dataset ref, what [`ExecutorSurface.check_scoring`][] found before the
#: point's first forward — ``{"string_mode", "result"}``, the result
#: one of ``causalab.causal.scoring.SCORING_RESULTS``. A sibling of the
#: ``execution`` block: recorded, never gated, because the refusal has already
#: happened by the time anything is written.
SCORING_KEY = "scoring"


def record_scoring(output_dir: Path, ref: str, check: ScoringCheck) -> Path | None:
    """Write ``check`` into the run receipt (``protocol.json`` under
    ``output_dir``) as ``scoring.<ref>``, merged per ref, and return the
    receipt's path.

    The receipt is written before execution by
    [`write_run_record`][]; this adds to it what the
    pre-forward check found. A caller that wrote no receipt (a workflow step
    records its run in ``_step.json``; an engine test drives
    ``execute_request`` directly) has nothing to amend: ``None``.
    """
    return record_scoring_records(output_dir, {ref: check.as_record()})


def record_scoring_records(
    output_dir: Path, records: Mapping[str, Mapping[str, Any]]
) -> Path | None:
    """[`record_scoring`][] over already-rendered records, ``{ref:
    check.as_record()}`` — what the data-parallel joiner writes for the
    points the other replicas ran (``neural/shared/join.py``). Merged per
    ref; the block is sorted on write, so the order of arrival is not in the
    bytes. ``None`` when there is no receipt to amend."""
    receipt = output_dir / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text())
    block = dict(record.get(SCORING_KEY) or {})
    for ref, rendered in records.items():
        block[ref] = dict(rendered)
    record[SCORING_KEY] = block
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


@dataclasses.dataclass(frozen=True)
class TrainEvalScore:
    """One point's ``train.eval`` result: the split it was measured on, the
    mean per eval metric, and how many eval passes ran.

    ``passes`` is there so a reader can tell "evaluated once at the end" from
    "evaluated every epoch and early-stopped", which decides whether the score
    describes the returned weights or merely the last pass over them.
    """

    split: str
    metrics: Mapping[str, float]
    passes: int
    #: The trained featurizers this score describes — the join back to the
    #: saved bundle, which stamps the same point digest.
    featurizers: tuple[str, ...] = ()
    #: How the returned weights were chosen: ``"early_stop.best"`` when the
    #: loop restored the best-scoring snapshot, ``"last"`` when nothing
    #: selected. Without it a reader cannot tell whether this score describes
    #: the saved weights or merely the final pass over them.
    selected: str = "last"

    def as_record(self, *, point: str, coords: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "point": point,
            "coords": dict(coords),
            "split": self.split,
            "metrics": dict(self.metrics),
            "passes": self.passes,
            "featurizers": list(self.featurizers),
            "selected": self.selected,
        }


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """One photograph of a fit (§2.12 ``trajectory``): the trained
    featurizers' slots after ``step`` updates, detached and on the CPU, and
    what the fit says about itself there — ``step``, ``epoch``, the loss and
    every objective term's value at that update, every trained gate's
    ``hard_mask_size`` / ``decisive_fraction``, every controlled
    hyperparameter's live value. The record rides on the bundle entry as
    numbers, so a reader has the scalar trace from the header alone."""

    step: int
    epoch: int
    slots: Mapping[str, Mapping[str, Any]]
    record: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class TrainOutcome:
    """What an engine's train loop produced.

    ``stages`` is the fitted stage per trained featurizer name — what the save
    manifest writes. ``eval_score`` is the held-out score from ``train.eval``,
    and it is deliberately *not* a metric-table column: it is measured on a
    different split, i.e. a different population from the point's metric rows,
    and folding it in under the same metric name is exactly the confusion that
    made a fit document's ``iia.json`` read as an eval score when it was the
    train score (spec §2.12).
    """

    stages: Mapping[str, Any]
    eval_score: TrainEvalScore | None = None
    #: Per trained featurizer, whatever the fit can say about *itself* — see
    #: [`fit_diagnostics`][causalab.neural.engines.pytorch_hooks.train.fit_diagnostics].
    #: Written beside the bundle, because a fit that produced a meaningless
    #: parameter and a perfect score is otherwise indistinguishable from a
    #: good one.
    diagnostics: Mapping[str, Mapping[str, float]] = dataclasses.field(
        default_factory=dict
    )
    #: What the fit's inner passes paid for its constant groups (``run``) and
    #: what the campaign store handed them instead (``served``) — the saving
    #: §4 "Fits" promises, tallied per point by the loop, since a cohort's
    #: passes interleave several points' (§4 "Cohorts"). ``None`` when the
    #: loop ran against no store.
    fit_forwards: Mapping[str, int] | None = None
    #: The block each of this fit's forwards that resumed started at (§4
    #: "Resume"): one entry per forward the point took part in, cohort
    #: forwards included, in the shape of ``ForwardCache.resumed``.
    resumed: tuple[int, ...] = ()
    #: The rows-per-forward bound every window of the fit's cohort ran under
    #: (§8 ``fit_rows``): the authored bound, or the one the engine measured
    #: when none was authored — after any shrink, of a grad window or of an
    #: eval window packing under it, so pinning it is safe (a run that shrank
    #: says so in ``fit_rows_shrinks``; only when the shrink was an eval
    #: window's does the pinned re-run pack its grad windows smaller than
    #: this one did); ``None`` when the cohort ran unbounded.
    fit_rows: int | None = None
    #: How many windows packed under the fit's bound — grad forwards, and the
    #: batched eval passes when ``batch_rows`` is unauthored — were re-packed
    #: after running out of memory: the measured bound was too loose by that
    #: many. A cohort's count, reported on each of its members.
    fit_rows_shrinks: int = 0
    #: Per ``train.control`` target (§2.11): where the controlled value
    #: started and ended, the last signal and setpoint, the update count —
    #: written into ``fit_diagnostics.json`` beside the featurizer records.
    controls: Mapping[str, Mapping[str, float]] = dataclasses.field(
        default_factory=dict
    )
    #: The same, per update: ``{"step", "value", "signal", "setpoint"}`` rows
    #: — what a checkpoint records about the controller at its step.
    control_trace: Mapping[str, Sequence[Mapping[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: Per ``constraint`` term (§2.11), by name: the target density, the dual
    #: pair where it started and ended (``lambda1_initial`` … ``lambda2_final``),
    #: the density the last update saw (``value_final``) and the update count
    #: — written into ``fit_diagnostics.json`` under ``constraints``.
    constraints: Mapping[str, Mapping[str, float]] = dataclasses.field(
        default_factory=dict
    )
    #: The same, per update: ``{"step", "value", "target", "lambda1",
    #: "lambda2"}`` rows — the duals *after* that update's ascent.
    constraint_trace: Mapping[str, Sequence[Mapping[str, float]]] = dataclasses.field(
        default_factory=dict
    )
    #: Per drawn counterfactual role (§2.2 ``draw``): the kind, the ``eval``
    #: member the non-training forwards read, and ``members`` — one list per
    #: epoch of the member index each row took — written into
    #: ``fit_diagnostics.json`` under ``draws``, so a reader sees which member
    #: each row trained against (under ``train.steps.updates`` a partial
    #: final epoch lists a member for every row, including rows whose
    #: minibatch never ran).
    draws: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    #: Per ``train.anneal`` target (§2.11): the schedule's endpoints, shape
    #: and the value the last update used — beside ``controls`` in
    #: ``fit_diagnostics.json``, so a reader sees what moved the weight it
    #: finds in the trajectory without re-deriving the schedule.
    anneals: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    #: Per ``train.phases`` entry (§2.11), in order: the update window it
    #: covered (``start``, ``end``), what trained in it (``params``), which
    #: masks it pinned (``freeze_masks``) — written under ``phases`` in
    #: ``fit_diagnostics.json``; empty for a one-phase fit.
    phases: Sequence[Mapping[str, Any]] = ()
    #: The fit's checkpoints, when the document saves a ``trajectory``
    #: (§2.12); empty otherwise. The last one, when present, is the fit the
    #: returned ``stages`` hold — unless ``early_stop`` restored an earlier
    #: best, which the record's ``step`` lets a reader see.
    checkpoints: Sequence[Checkpoint] = ()


#: An engine's train loop: fit these points **together** where it can (§4
#: "Cohorts") and return one outcome per point, in order. The points are one
#: cohort — same realization, same rows, same frame (``plan.fit_cohorts``) —
#: so the loop may run their optimizer steps as one forward each.
TrainRunner = Callable[
    [Sequence[Document], Sequence[Any], RunContext], Sequence[TrainOutcome]
]


def _tap_union(
    docs: Sequence[Document], plans: Sequence[PointPlan]
) -> dict[GroupKey, tuple[SiteSpec, ...]]:
    """Forward-group key → every site the campaign taps in that group.

    The union *is* the interning. Taps are deliberately absent from a group's
    key, so the single pass a shared key earns has to capture every
    address any point will ask of it — for a 32-layer scan that is one
    counterfactual forward with 32 taps instead of 32 forwards with one each.

    Continuation reads are excluded: those are served by the decode's
    per-step accumulation, not by the prefill capture this store holds, so a
    decoding group contributes only its prompt-frame taps (and can therefore
    still hand its prefill to a non-decoding point that shares the key).

    A site enters the union as the read **captures** it
    ([`capture_spec`][]): an ``lm_head`` read
    at named positions is served from ``ln_final``, so that is what the
    shared pass stores for it — ``[rows, seq, d_model]``, not the whole
    vocabulary — and what a later point's lookup asks for.
    """
    union: dict[GroupKey, dict[str, SiteSpec]] = {}
    for doc, plan in zip(docs, plans):
        for group in plan.groups:
            wanted = union.setdefault(group.key, {})
            for tap in group.taps:
                read = doc.reads[tap.read]
                if generated_budget(doc, read.pos) is not None:
                    continue
                spec = capture_spec(doc, group.model, group.input, tap.read)
                wanted[json.dumps(spec_identity(spec), sort_keys=True)] = spec
    return {key: tuple(specs.values()) for key, specs in union.items()}


def campaign_cache(
    docs: Sequence[Document], plans: Sequence[PointPlan]
) -> ForwardCache:
    """The one [`ForwardCache`][causalab.neural.shared.executor.cache.ForwardCache] a campaign runs against: the tap union
    per group key (§3) and the prefix plans per group key (§4 "Resume").

    The prefix arithmetic is read off the **interned** groups, whose taps are
    the union over every sharer: the block a shared pass may start at is the
    shallowest any of them taps, exactly as the block it may stop after is
    the deepest. ``wanted_prefix_depths`` inverts that map by prefix
    identity, so the first pass over an input's rows — whichever model runs
    it — knows every depth a later intervened model will want.
    ``prefix_owed`` is that prefix's lifetime, in the shape of ``owed``: per
    (identity, depth), how many group instances across the points may still
    start from it — every one whose interned ``resume_at`` reaches the depth.
    """
    prefix_plans: dict[GroupKey, PrefixPlan] = {}
    wanted: dict[GroupKey, set[int]] = {}
    for group in interned_groups(plans):
        depth = group.resume_at
        prefix_plans[group.key] = PrefixPlan(
            base_key=group.base_key,
            resume_at=depth,
            write_depth=group.write_depth,
        )
        if depth > 0:
            wanted.setdefault(group.base_key, set()).add(depth)
    prefix_owed: Counter[tuple[GroupKey, int]] = Counter()
    for plan in plans:
        for group in plan.groups:
            reach = prefix_plans[group.key].resume_at
            for depth in wanted.get(group.base_key, ()):
                if depth <= reach:
                    prefix_owed[(group.base_key, depth)] += 1
    return ForwardCache(
        wanted=_tap_union(docs, plans),
        prefix_plans=prefix_plans,
        wanted_prefix_depths={base: frozenset(v) for base, v in wanted.items()},
        prefix_owed=dict(prefix_owed),
        owed=dict(Counter(group.key for plan in plans for group in plan.groups)),
    )


def execute_request(
    compiled: CompiledProtocol,
    run: RunContext,
    *,
    engine_name: str,
    executor_factory: Callable[
        [
            Document,
            RunContext,
            Mapping[str, Any],
            "Interning | None",
            StepResolution | None,
        ],
        ExecutorSurface,
    ],
    train_runner: TrainRunner | None = None,
    intern_forwards: bool = False,
    engine: Engine | None = None,
) -> RunResult:
    """Run the points ``run`` selects of ``compiled`` through one engine's
    executors — the engine-neutral body of every ``Engine.execute``.

    ``compiled`` is the identity: the axes and the lowered tree they index
    into, and the campaign digest that stamps what is written. **The sweep is
    the engine's**: the steps are enumerated here in the canonical
    order ([`enumerate_steps`][]), the ones
    ``run`` selects are parsed once and held to the whole §5 checklist — a
    violation two axes make together is no representative's of the
    compiler's pass ([`check_steps`][])
    — and signed
    with the protocol's one hasher ([`sign_steps`][]), all before anything is planned and before any model
    loads; what was signed comes back as [`RunResult.steps`][]. ``run`` is
    the context: the shard (``points``, ``None`` for every point), the
    resolution environment, where outputs land, the execution and decoding
    blocks, and ``record``: when set (opt-in at every door), the run receipt
    (``protocol.json``) is written and the event stream opened here, after
    the signing and before the first forward
    ([`causalab.neural.shared.receipt`][]), and the stream is finished once
    the campaign has run; ``engine`` is then the
    [`Engine`][] whose execution block the
    receipt records. The amenders below add what execution observed to a
    receipt on disk, and are no-ops without one, so the ``fires`` counts of
    an unrecorded run live in the returned summaries alone.

    ``train_runner`` is the engine's train loop, cohort-shaped
    ([`TrainRunner`][]); an engine without one (its ``grad`` capability
    absent, so routing never sends it a ``train`` document) refuses loudly if
    a train document reaches it anyway.

    ``intern_forwards`` says this engine's executor consults the shared
    [`ForwardCache`][causalab.neural.shared.executor.cache.ForwardCache] (§3), so
    [`forwards`][causalab.protocol.engine.RunResult.forwards] reports what the run
    paid. An engine that has not claimed the interning leaves it False and
    reports 0 — "not measured" rather than a number it did not count.

    **Order** (§4 "Cohorts"). The campaign runs in two phases. First, cohort
    by cohort ([`fit_cohorts`][]), every point of
    the cohort is prepared — its executor built, its ledger checked — and the
    cohort's fits run together through ``train_runner``. Then every point is
    finished in **point order**: its own whole-role passes, metrics and saved
    entries. Nothing a point writes depends on which cohort it fitted in, so
    the outputs are the per-point loop's; only the fits' inner passes are
    shared.
    """
    # The whole campaign is planned before anything runs, because §3's
    # interning is a property of the point *set*: a forward group can only be
    # shared once you know which other points share it, and the union of taps
    # it must capture only exists across all of them.
    # Planned and run in the execution form: a band site (§2.4 ``layers``) is
    # lowered to its per-layer members here, once, so the plans' taps, the
    # campaign's tap union and every executor name the same reads and sites
    # (`lower_bands` is idempotent; the executor lowers again for callers that
    # build one directly). What a band read has no member for — a save, a
    # metric — is refused here, before any forward.
    expansion = enumerate_steps(compiled)
    indices = run.indices(len(expansion.points))
    # the checklist over every selected step first: a violation two axes make
    # together (a collision, a depth inversion, a layer the swept model lacks)
    # is refused before anything is signed or planned
    parsed = tuple(parse_document(expansion.points[index].raw) for index in indices)
    check_steps(
        parsed,
        run.env,
        coords=[expansion.points[index].coords for index in indices],
    )
    campaign = sign_steps(expansion, run.env, indices=indices)
    # the campaign's signed steps, on every rank: the receipt lists them, the
    # result returns them, and the data-parallel join places shards by them
    records = tuple(step.record for step in campaign)
    campaign_digests = tuple(step.digest for step in campaign)
    # this replica's contiguous shard of the selection under data parallelism
    # over points (docs/model_parallelism.md §8.3; `publish.point_shard`) — the
    # whole selection at world 1 and under the rows mode, where every replica
    # runs every point and the engine splits each fit minibatch instead
    publisher = run.publisher
    geometry = getattr(engine, "parallel", None)
    over_rows = (
        geometry is not None and getattr(geometry, "data_mode", "points") == "rows"
    )
    if publisher.replicas > 1 and not over_rows:
        shard = point_shard(range(len(indices)), publisher.replica, publisher.replicas)
        selected = tuple(campaign[position] for position in shard)
        # the parsed steps narrow with the signed ones: everything planned
        # and executed below is this replica's shard, indexed alike
        parsed = tuple(parsed[position] for position in shard)
    else:
        selected = campaign
    _log.info(
        "%d of %d points selected; engine %s",
        len(selected),
        len(expansion.points),
        engine_name,
    )
    log = None
    if run.record and is_joiner(publisher):
        # `engine` None (a caller driving the shared body directly) records
        # the block an engine with no bounds and a loaded model would; the
        # joiner alone writes the receipt and the stream (§3)
        write_run_record(compiled, run, engine, records)
        log = run_events(compiled, run, records)
    digests = [step.digest for step in selected]
    canonical = [step.canonical for step in selected]
    # the positions the protocol layer resolved before any weights loaded
    # (`pipeline.resolve_positions`), by the key a step shares with the
    # representative that stands for it; a step no representative covers —
    # or a door that skipped the resolution — leaves the executor to resolve
    # its own through the same functions
    resolutions = compiled.positions or {}
    resolved = [resolutions.get(positions_key(doc)) for doc in parsed]
    docs = tuple(lower_bands(doc) for doc in parsed)
    plans = campaign_plans(docs, canonical) if intern_forwards else ()
    # `owed` (inside `campaign_cache`) is the same count `interned_groups`
    # merges: how many point groups key into each group key. It bounds a
    # capture's lifetime to its sharers.
    cache = campaign_cache(docs, plans) if intern_forwards else ForwardCache()
    # which points may fit together: the same rows on the same realization
    # (the data identity the group keys carry); without a planned
    # campaign there is no such identity and every point stands alone
    cohorts = (
        fit_cohorts(
            docs,
            [_data_identity(doc, form) for doc, form in zip(docs, canonical)],
        )
        if intern_forwards
        else tuple((i,) for i in range(len(docs)))
    )

    def prepare(i: int) -> _Prepared:
        """Point ``i``'s executor: the model load, if this point's is not
        already cached, sits inside ``executor_factory``."""
        doc = docs[i]
        _log.info(
            "point %d/%d %s: loading model %s@%s",
            i + 1,
            len(docs),
            digests[i][:12],
            doc.model.key,
            doc.model.revision,
        )
        started = time.perf_counter()
        member = _prepare_point(
            doc,
            run,
            coords=selected[i].coords,
            point_digest=digests[i],
            executor_factory=executor_factory,
            interning=(
                Interning(
                    keys={
                        (group.model, group.input): group.key
                        for group in plans[i].groups
                    },
                    cache=cache,
                )
                if intern_forwards
                else None
            ),
            resolution=resolved[i],
        )
        _log.info(
            "point %d/%d %s: ready in %.1fs",
            i + 1,
            len(docs),
            digests[i][:12],
            time.perf_counter() - started,
        )
        return member

    prepared: list[_Prepared | None] = [None] * len(docs)
    for cohort in cohorts:
        if all(docs[i].train is None for i in cohort):
            # nothing to fit: built when its turn comes, in point order below,
            # so the store it starts against is what the points before it left
            continue
        members = [prepare(i) for i in cohort]
        fits = [member for member in members if member.doc.train is not None]
        if fits:
            if train_runner is None:
                raise ProtocolError(
                    "P4",
                    f"this document declares a train section, which the "
                    f"{engine_name!r} engine does not implement — its 'grad' "
                    "capability is absent, so routing should not have sent it "
                    "here",
                )
            _log.info("fitting a cohort of %d point(s)", len(fits))
            started = time.perf_counter()
            outcomes = train_runner(
                [member.doc for member in fits],
                [member.executor for member in fits],
                run,
            )
            _log.info("cohort fit in %.1fs", time.perf_counter() - started)
            if len(outcomes) != len(fits):
                raise AssertionError(
                    f"the train loop returned {len(outcomes)} outcomes for a "
                    f"cohort of {len(fits)} points"
                )
            for member, outcome in zip(fits, outcomes):
                member.outcome = outcome
        for i, member in zip(cohort, members):
            prepared[i] = member

    tensor_files: dict[str, TensorFile] = {}
    metric_files: dict[str, MetricTable] = {}
    train_evals: list[Mapping[str, Any]] = []
    fit_diagnostics: list[Mapping[str, Any]] = []
    routing_mismatch: list[Mapping[str, Any]] = []
    summaries: list[Mapping[str, Any]] = []
    cells: list[Resolution] = []
    scoring: dict[str, Mapping[str, Any]] = {}
    for i in range(len(prepared)):
        member = prepared[i]
        if member is None:  # a point that fits nothing
            member = prepare(i)
        _log.info("point %d/%d %s: running", i + 1, len(docs), digests[i][:12])
        started = time.perf_counter()
        summaries.append(
            _execute_point(
                member,
                run,
                tensor_files=tensor_files,
                metric_files=metric_files,
                train_evals=train_evals,
                fit_diagnostics=fit_diagnostics,
                routing_mismatch=routing_mismatch,
                cells=cells,
            )
        )
        if member.scoring is not None:
            ref, check = member.scoring
            scoring[ref] = check.as_record()
        _log.info(
            "point %d/%d %s: done in %.1fs",
            i + 1,
            len(docs),
            digests[i][:12],
            time.perf_counter() - started,
        )
        # the point is finished: its executor — the eval executor, the read
        # values, the frames it holds — dies here as it did in the per-point
        # loop, not at the end of the run
        prepared[i] = None
    first_doc = docs[0]
    first_realization = canonical_model(first_doc.raw["model"])
    shard = ShardOutput(
        digests=tuple(digests),
        tensor_files=tensor_files,
        metric_files=metric_files,
        train_evals=train_evals,
        fit_diagnostics=fit_diagnostics,
        routing_mismatch=routing_mismatch,
        summaries=summaries,
        cells=cells,
        scoring=scoring,
        model={
            "key": str(first_doc.model.key),
            "revision": str(first_doc.model.revision),
            "dtype": str(first_realization["dtype"]),
            "quantization": first_realization.get("quantization"),
            "attn_implementation": first_realization.get("attn_implementation"),
        },
        attention_backends=frozenset(doc.model.attn_implementation for doc in docs),
        forwards=len(cache.executed),
    )
    # the one seam (docs/model_parallelism.md §3, §8.3): a rank that does not
    # publish computed and discards; a publishing rank hands its shard to the
    # joiner, which places every replica's shard by point digest and writes
    # the campaign once — at world 1 the gather is the identity and the
    # joined shard is this one, today's path to the byte
    if not publisher.publish:
        return RunResult(files={}, forwards=shard.forwards, steps=records)
    gathered = publisher.gather(shard)
    if gathered is None:
        return RunResult(files={}, forwards=shard.forwards, steps=records)
    joined = join_shards(gathered, campaign_digests)
    if len(gathered) > 1:
        # the pre-forward scoring checks of the points other replicas ran:
        # the joiner's own were recorded as they were made
        record_scoring_records(run.output_dir, joined.scoring)
    return _publish(run, joined, engine_name=engine_name, steps=records, log=log)


def _publish(
    run: RunContext,
    joined: ShardOutput,
    *,
    engine_name: str,
    steps: tuple[StepRecord, ...],
    log: Any,
) -> RunResult:
    """Write what the campaign accumulated ([`ShardOutput`][]) under the
    run's output directory — the save files with their identity stamp,
    then the receipt's ``fires``, measured bounds and ragged geometry — and
    return the result carrying the campaign's signed ``steps``. The only
    writer of a campaign's outputs; the joiner is the only rank that reaches
    it. ``log`` is the run's open event stream, or ``None``."""
    # implementation requirements the points' addresses imposed (§7.3, e.g.
    # "attn_eager") — execution metadata beside the engine name, never
    # canonical form: the documents and their digests are implementation-blind
    applied = sorted(
        {
            requirement
            for summary in joined.summaries
            for requirement in summary.get("implementations", ())
        }
    )
    identity_base = {
        "model_key": joined.model["key"],
        "model_revision": joined.model["revision"],
        "model_dtype": joined.model["dtype"],
        "model_quantization": joined.model["quantization"],
        # A backend sweep has no single file-level selection. Each tensor
        # entry carries its own choice, just as fitted entries do below.
        "model_attn_implementation": (
            joined.model["attn_implementation"]
            if len(joined.attention_backends) == 1
            else None
        ),
        "engine": engine_name,
        **({"implementations": ",".join(applied)} if applied else {}),
        # `runtime_identity().short_revision`, not a `git rev-parse` here: the
        # first 12 hex of the running package's tree digest, for every install
        # kind. The field keeps its shape — a short hex string — and loses its
        # ability to say "unknown": the value always identifies content.
        "commit": runtime_identity().short_revision,
    }
    files = write_outputs(
        run.output_dir,
        joined.tensor_files,
        joined.metric_files,
        identity_base=identity_base,
        train_evals=joined.train_evals,
        fit_diagnostics=joined.fit_diagnostics,
        routing_mismatch=joined.routing_mismatch,
    )
    _log.info("wrote %d file(s) to %s", len(files), run.output_dir)
    # every write member of every point fired the count its kind declares
    # (§4 "Fires") — the executor refused the run otherwise, before this
    # line; the counts go into the receipt once, after the whole campaign,
    # so a refused run's receipt records no subset. A run without a receipt
    # (the default) keeps them in the summaries and writes nothing.
    record_fires(
        run.output_dir,
        {
            str(summary["point"]): summary["fires"]
            for summary in joined.summaries
            if summary.get("fires")
        },
    )
    # the bounds the run measured when none was authored (§8 `fit_rows`,
    # `batch_rows`): into the receipt's `execution` block, as the numbers to pin
    record_measured_bounds(run.output_dir, joined.summaries)
    # the ragged-write geometry the points landed under (§5 rule 19):
    # into the same block, only when some write ran under a non-`refuse`
    # policy — a receipt that authors none is byte-identical to before
    record_ragged_geometry(run.output_dir, joined.summaries)
    # the commit each model's revision resolved to, read at load: beside the
    # `execution` block, so a document that names `main` still says which
    # snapshot produced its tables
    record_models(run.output_dir, joined.summaries)
    result = RunResult(
        files=files,
        summaries=tuple(joined.summaries),
        forwards=joined.forwards,
        cells=tuple(joined.cells),
        steps=steps,
    )
    if log is not None:
        emit_run_events(log, result)
    return result


@dataclasses.dataclass
class _Prepared:
    """One point between its two phases (see [`execute_request`][]): built
    and, if it trains, fitted — its own passes not yet run."""

    doc: Document
    coords: Mapping[str, Any]
    point_digest: str
    executor: ExecutorSurface
    interning: "Interning | None"
    #: the location ledger (§6), only when the document saves one
    ledger: Any
    #: the attention backend the loaded model runs — observed at load, stamped
    #: on what the point saves beside the authored one
    runtime_attention: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    #: the point's model as it loaded: the document's ``key`` and ``revision``
    #: and the commit the loader resolved the revision to (§8; the receipt's
    #: ``models``). Execution provenance, never stamped
    model: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    outcome: TrainOutcome | None = None
    #: the base ref and what the pre-forward scoring check found — recorded
    #: in the receipt as it was made when this process holds one, and carried
    #: to the data-parallel joiner otherwise
    scoring: tuple[str, ScoringCheck] | None = None


def _prepare_point(
    doc: Document,
    run: RunContext,
    *,
    coords: Mapping[str, Any],
    point_digest: str,
    executor_factory: Callable[
        [
            Document,
            RunContext,
            Mapping[str, Any],
            "Interning | None",
            StepResolution | None,
        ],
        ExecutorSurface,
    ],
    interning: "Interning | None",
    resolution: StepResolution | None,
) -> _Prepared:
    if doc.train is not None:
        # §5 rule 22's cross-table refusal, before an executor exists: a fit
        # whose training rows and `train.eval.split` rows share a prompt is
        # refused here, so no engine's forward — and no minibatch — runs on it
        check_fit_splits(doc, run.env.datasets)
    executor = executor_factory(doc, run, coords, interning, resolution)
    config = getattr(
        getattr(getattr(executor, "bundle", None), "model", None), "config", None
    )
    loaded_backend = getattr(config, "_attn_implementation", None)
    # Runtime observation is inherited, but compatibility remains authored.
    runtime_attention = build_artifact_identity(
        loaded_attn_implementation=loaded_backend
    )
    # transformers keeps the snapshot a config was read from as
    # `_commit_hash` (None for a local directory); the measurement layer
    # reads the same attribute (causalab/measurement/runtime/worker.py)
    commit = getattr(config, "_commit_hash", None)
    model = {
        "key": str(doc.model.key),
        "revision": str(doc.model.revision),
        "resolved_revision": commit if isinstance(commit, str) else None,
    }
    # the base table's recorded string_mode against the document's `match` modes
    # (§2.2, §2.10): refused before any forward, recorded in the run receipt
    # by the joiner alone — the one process that writes (docs/model_parallelism.md
    # §3); every other rank's check travels in its shard, and the joiner writes
    # it at the join. Two ranks amending one receipt would race on the file.
    base_ref = doc.data["base"].dataset
    scoring: tuple[str, ScoringCheck] | None = None
    if isinstance(base_ref, str):
        check = executor.check_scoring()
        if is_joiner(run.publisher):
            record_scoring(run.output_dir, base_ref, check)
        scoring = (base_ref, check)
    # the location ledger (§6), only when the document saves one — the
    # protocol layer's when it resolved this step (`pipeline.resolve_positions`),
    # else built here, before any forward, from the executor's frames through
    # the same builder (`protocol/positions/resolve.py`)
    ledger = None
    if wants_ledger(doc):
        ledger = (
            resolution.ledger
            if resolution is not None and resolution.ledger is not None
            else executor.location_ledger()
        )
    return _Prepared(
        doc=doc,
        coords=coords,
        point_digest=point_digest,
        executor=executor,
        interning=interning,
        ledger=ledger,
        runtime_attention=runtime_attention,
        model=model,
        scoring=scoring,
    )


def _execute_point(
    member: _Prepared,
    run: RunContext,
    *,
    tensor_files: dict[str, TensorFile],
    metric_files: dict[str, MetricTable],
    train_evals: list[Mapping[str, Any]],
    fit_diagnostics: list[Mapping[str, Any]],
    routing_mismatch: list[Mapping[str, Any]],
    cells: list[Resolution],
) -> Mapping[str, Any]:
    doc, executor, interning, ledger = (
        member.doc,
        member.executor,
        member.interning,
        member.ledger,
    )
    coords, point_digest = member.coords, member.point_digest
    runtime_attention = member.runtime_attention
    # one result cell per save entry (spec §4.1): what this point's run
    # resolved and what it could not. Appended to the campaign's list here,
    # and the unavailable ones repeated in this point's summary.
    point_cells: list[Resolution] = []
    trained_stages: dict[str, Any] = {}
    fit_forwards: Mapping[str, int] | None = None
    # what this point's passes resumed from a cached prefix instead of
    # recomputing (§4 "Resume"): the fit's, tallied per point by the loop,
    # plus its own, a difference over the campaign store
    fit_resumed: tuple[int, ...] = ()
    fit_rows: int | None = None
    fit_rows_shrinks = 0
    resumed_before = len(interning.cache.resumed) if interning is not None else 0
    if doc.train is not None:
        outcome = member.outcome
        if outcome is None:
            raise AssertionError("a train point reached its own passes unfitted")
        # what the fit's inner passes paid for its constant groups, and what
        # the store handed them instead — the saving §4 "Fits" promises, per
        # point. Reported in this point's summary, which `RunResult` returns
        # to the caller and nothing writes to disk.
        fit_forwards = outcome.fit_forwards
        fit_resumed = outcome.resumed
        fit_rows = outcome.fit_rows
        fit_rows_shrinks = outcome.fit_rows_shrinks
        trained_stages = dict(outcome.stages)
        if outcome.eval_score is not None:
            train_evals.append(
                outcome.eval_score.as_record(point=point_digest, coords=coords)
            )
        if (
            outcome.diagnostics
            or outcome.controls
            or outcome.anneals
            or outcome.phases
            or outcome.constraints
            or outcome.draws
        ):
            fit_diagnostics.append(
                {
                    "point": point_digest,
                    "coords": dict(coords),
                    "featurizers": {
                        name: dict(values)
                        for name, values in outcome.diagnostics.items()
                    },
                    **(
                        {
                            "controls": {
                                target: dict(values)
                                for target, values in outcome.controls.items()
                            }
                        }
                        if outcome.controls
                        else {}
                    ),
                    **(
                        {
                            "anneals": {
                                target: dict(values)
                                for target, values in outcome.anneals.items()
                            }
                        }
                        if outcome.anneals
                        else {}
                    ),
                    **(
                        {"phases": [dict(phase) for phase in outcome.phases]}
                        if outcome.phases
                        else {}
                    ),
                    **(
                        {
                            "constraints": {
                                name: dict(values)
                                for name, values in outcome.constraints.items()
                            }
                        }
                        if outcome.constraints
                        else {}
                    ),
                    **(
                        {
                            "draws": {
                                role: dict(values)
                                for role, values in outcome.draws.items()
                            }
                        }
                        if outcome.draws
                        else {}
                    ),
                }
            )
    # Order is load-bearing. The point executor is counted and grad-free, so
    # `_may_intern` lets it read from and publish to the store for every
    # model, the trained one included; that is safe only because no forward
    # of it runs until the fit above has finished and the stages are final.
    # One featurizer-cache scope for the point's own passes: they run grad-free
    # over stages that no longer move, so a rotation or a mask evaluated for
    # the first read is the value every later featurize and every write hook
    # of the point would recompute — the same computation, read several times
    # (`featurizers.featurizer_cache`)
    with featurizer_cache():
        executor.run_all()
    # what every write through an expert-keyed gate found about the pair's
    # routing (executor.writes._align_by_expert): the base slots whose expert
    # the operand's side never activated, per layer and example. Filled by
    # the writes the full-data pass above landed, so it describes the rows
    # the metric tables describe
    for (write, layer, example), (mismatched, slots) in sorted(
        (getattr(executor, "routing_mismatch", None) or {}).items()
    ):
        routing_mismatch.append(
            {
                "point": point_digest,
                "coords": dict(coords),
                "write": write,
                "layer": layer,
                "example": example,
                "mismatched": mismatched,
                "slots": slots,
            }
        )
    metric_values: dict[str, list[Any]] = {}
    windowed: dict[str, _Windowed] = {}
    #: metrics whose read is an unavailable cell, and so are they (§4.1)
    inherited: dict[str, Unavailable] = {}
    saved_aggregations = {agg.owner: agg for agg in doc.saved_aggregations()}
    for agg in saved_aggregations.values():
        qname, metric = agg.label, agg.spec
        if qname in metric_values or qname in windowed:
            continue  # one reduction, however many entries restate it
        of_name = agg.read
        target_name = agg.target
        if executor.is_generated(of_name):
            # a continuation read addresses as many positions as the row
            # generated, so its metric reduces per step and reports which
            # steps it saw (§2.3, §2.10)
            windowed[qname] = _Windowed(
                values=executor.generated_metric(agg),
                steps=(
                    None
                    if str(metric.kind) in WHOLE_WINDOW_METRIC_KINDS
                    else executor.addressed_steps(of_name)
                ),
                matched=[bool(steps) for steps in executor.addressed_steps(of_name)],
            )
            continue
        resolved_of = executor.resolution(of_name)
        key = cell_key(qname, coords)
        # the rows of the read(s) this metric reduces that aligned on nothing
        # (§4.1): each is an excluded measurement of *this* metric, under the
        # read's reason, keyed under the metric's cell (§2.10 "Eligibility")
        per_row = _row_exclusions(executor, qname, of_name, target_name, key)
        eligible = [i for i, cell in enumerate(per_row) if cell is None]
        if isinstance(resolved_of, Unavailable) and (not any(per_row) or not eligible):
            # the whole cell is unavailable — for a reason that is not any
            # row's (an `expert:` face the router sent no token), or because
            # every row failed to align: the metric inherits the cell — same
            # reason, the read's detail — rather than reducing a gather with
            # empty rows into a number
            inherited[qname] = unavailable(
                resolved_of.reason,
                f"metric {qname!r} reduces read {of_name.read!r}, which is unavailable: "
                + resolved_of.detail,
                key,
            )
            metric_values[qname] = []
            continue
        rows = executor.rows_for_metrics()
        if any(per_row):
            # some rows aligned and some did not: score the rows that did,
            # over exactly their positions, and put the typed `unavailable`
            # in each excluded row's place — the value is computed over the
            # eligible rows only, and the excluded ones are never averaged in
            of_dense = executor.dense_rows(of_name, eligible)
            target_dense = (
                executor.dense_rows(target_name, eligible)
                if target_name is not None
                else None
            )
            rows = [rows[i] for i in eligible]
        else:
            of_dense = executor.dense_value(of_name)
            target_dense = (
                executor.dense_value(target_name) if target_name is not None else None
            )
        # scored where the value sits: a read only these metrics consume was
        # left on its device (`executor.base.device_scored_reads`), and the
        # scorer copies the answer columns or the argmax, not the vocabulary
        values = score_metric(
            metric,
            of_dense,
            rows,
            executor.bundle.tokenizer,
            target_value=target_dense,
            vocab_axis=read_is_vocabulary(doc, of_name.read),
            denominator_key=key,
        )
        if any(per_row):
            scored = iter(values)
            values = [cell if cell is not None else next(scored) for cell in per_row]
        metric_values[qname] = values
    # every metric cell's eligibility record (§2.10): how many rows its
    # decision rule was evaluated over, of how many considered, the excluded
    # ones by reason — derived from the rows, and repeated on the cell and in
    # this point's summary so no consumer has to count the table again
    eligibility: dict[str, Eligibility] = {
        **{
            qname: Eligibility.of(values)
            for qname, values in metric_values.items()
            if qname not in inherited
        },
        **{qname: _windowed_eligibility(window) for qname, window in windowed.items()},
    }
    # every metric row is a base row (§2.2), so its label is the base row's
    labels = example_labels(executor.rows_for_metrics())
    for index, entry in enumerate(doc.save):
        # every entry's cell goes by its label — a read entry's file stem, so
        # one read saved on two models is two cells under two stems (§2.12)
        key = cell_key(entry.label, coords)
        agg = saved_aggregations.get(f"save[{index}]")
        if entry.kind == "trajectory":  # §2.12: the fit's checkpoints, one bundle
            point_cells.append(
                available({"file_path": entry.file_path, "kind": entry.kind}, key)
            )
            trajectory_file = tensor_files.setdefault(entry.file_path, TensorFile())
            for checkpoint in outcome.checkpoints:
                for fname, slots in checkpoint.slots.items():
                    # the same identity the featurizer's own bundle carries, so
                    # a checkpoint reloads through the same checks. `featurizer`
                    # and `step` are coordinates of the entry, not of the
                    # document (not in the digest): a fit that trains several
                    # featurizers photographs each of them at every step, and
                    # the name is what keeps `gate_3`'s theta from overwriting
                    # `gate_7`'s under one `theta[step=n]` key — a consumer
                    # names its own (`entry: {"featurizer": "gate_3", "step":
                    # 40}`); a one-featurizer consumer's `{"step": 40}` is
                    # already unique
                    stage = trained_stages[fname]
                    identity = featurizer_identity(
                        doc,
                        fname,
                        _featurizer_site(doc, fname),
                        stage=stage,
                        group_map=getattr(stage, "groups", None),
                    )
                    for slot, tensor in slots.items():
                        trajectory_file.add(
                            slot,
                            tensor,
                            {**coords, "featurizer": fname, "step": checkpoint.step},
                            label_entry=fname,
                            identity=identity,
                            record=checkpoint.record,
                        )
                    trajectory_file.record_common(identity)
            continue
        if entry.kind == "rank":  # §2.12: every gate's units ordered by theta
            point_cells.append(
                available({"file_path": entry.file_path, "kind": entry.kind}, key)
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            table.rows.extend(
                rank_records(
                    {**executor.stage_cache, **trained_stages}, point_digest, coords
                )
            )
            continue
        if entry.kind is not None:  # `location_ledger` (§2.12): a JSON table
            assert ledger is not None
            point_cells.append(
                available({"file_path": entry.file_path, "kind": entry.kind}, key)
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            table.rows.extend(ledger_records(ledger, point_digest, coords))
            continue
        if agg is not None:
            label, spec = agg.label, agg.spec
            point_cells.append(
                inherited.get(label)
                or _metric_cell(
                    label,
                    entry.file_path,
                    eligibility[label],
                    metric_values.get(label, []),
                    key,
                )
            )
            table = metric_files.setdefault(entry.file_path, MetricTable())
            # the record's identity (§2.10): authored on the aggregation, or
            # the kind's own — repeated on every row, never a digest field
            identity = metric_record_identity(
                str(spec.kind), unit=spec.unit, estimand_version=spec.estimand_version
            )
            if label in windowed:
                window = windowed[label]
                table.add_windowed(
                    label,
                    window.values,
                    coords,
                    identity=identity,
                    steps=window.steps,
                    matched=window.matched,
                    labels=labels,
                )
            else:
                table.add(
                    label,
                    metric_values[label],
                    coords,
                    identity=identity,
                    labels=labels,
                )
        elif entry.read is not None:
            # the site and the data go on the entry too: a harvested
            # activation is bound to where it was read and to what was read,
            # and a consumer (a script step fitting a basis on it, then a
            # document starting a fit from that basis) has no other way to
            # prove the site agrees, or to record which data the basis saw
            ref = entry.read
            read = doc.reads[ref.read]
            read_site = site_identity(doc, str(read.site))
            dataset = str(input_roles(doc)[doc.group_of(ref)[1]].dataset)
            resolved = executor.resolution(ref)
            cell: Resolution = (
                resolved
                if isinstance(resolved, Unavailable)
                else available({"file_path": entry.file_path, "key": key}, key)
            )
            point_cells.append(cell)
            tensor_files.setdefault(entry.file_path, TensorFile()).add(
                ref.read,
                executor.read_value(ref),
                coords,
                reduce=entry.reduce,
                identity={
                    **runtime_attention,
                    **build_artifact_identity(
                        model_attn_implementation=doc.model.attn_implementation,
                    ),
                    "trained_on": dataset,
                    **(
                        {"site": json.dumps(read_site, sort_keys=True)}
                        if read_site
                        else {}
                    ),
                    # nothing for an available cell; the four status fields
                    # for an unavailable one — so a result written before
                    # the value existed is byte-identical (spec §4.1)
                    **cell_record(cell),
                },
            )
        else:  # a trained featurizer bundle
            assert entry.value is not None
            stage = trained_stages.get(entry.value)
            if stage is None:
                raise ProtocolError(
                    "P2", f"featurizer {entry.value!r} was not trained this run"
                )
            point_cells.append(
                available(
                    {"file_path": entry.file_path, "featurizer": entry.value}, key
                )
            )
            bundle_file = tensor_files.setdefault(entry.file_path, TensorFile())
            identity = {
                **featurizer_identity(
                    doc,
                    entry.value,
                    entry.site,
                    stage=stage,
                    group_map=getattr(stage, "groups", None),
                ),
                **runtime_attention,
            }
            for slot, param in stage.slot_params().items():
                # per entry, not per file: a swept fit writes one file from
                # many points, and only the entry table can say which point
                # produced which rotation (§8)
                bundle_file.add(
                    slot,
                    param.detach(),
                    coords,
                    label_entry=entry.value,
                    identity=identity,
                )
            bundle_file.record_common(identity)
    # the windows this point's passes resumed from a cached prefix (§4
    # "Resume"), by the block they started at: the fit's forwards this point
    # took part in, then its own
    resumed = [
        *fit_resumed,
        *(interning.cache.resumed[resumed_before:] if interning is not None else []),
    ]
    # per forward group with writes, each member's fire count (§4 "Fires") —
    # what this point's own pass counted, or the counts of the pass that
    # produced the captures it was served; an engine whose executor keeps no
    # tally records nothing rather than zeroes
    fires = {
        group_label(model, input_role): dict(counts)
        for (model, input_role), counts in sorted(
            (getattr(executor, "fires", None) or {}).items()
        )
        if counts
    }
    ragged = {
        f"{model}/{write}": dict(geometry)
        for (model, write), geometry in sorted(
            (getattr(executor, "ragged_geometry", None) or {}).items()
        )
    }
    cells.extend(point_cells)
    excluded = {
        cell.denominator_key: cell.record()
        for cell in point_cells
        if isinstance(cell, Unavailable)
    }
    return {
        "point": point_digest,
        **runtime_attention,
        # the receipt's `models` entry for this point (`model_records`)
        **({"model": dict(member.model)} if member.model else {}),
        "coords": dict(coords),
        "metrics": {
            name: _summary_stat(values) for name, values in metric_values.items()
        },
        # every metric cell's `n_eligible` / `n_considered` (§2.10), the
        # excluded rows by reason when there were any — the aggregate cell's
        # denominator, beside the aggregate
        **(
            {"eligibility": {name: e.as_record() for name, e in eligibility.items()}}
            if eligibility
            else {}
        ),
        **({"fit_forwards": fit_forwards} if fit_forwards is not None else {}),
        # the rows-per-grad-forward bound the fit ran under (§8 `fit_rows`):
        # what an author pins to reproduce a measured one
        **({"fit_rows": fit_rows} if fit_rows is not None else {}),
        **({"fit_rows_shrinks": fit_rows_shrinks} if fit_rows_shrinks else {}),
        **(
            {"prefix_reuse": {"resumed": len(resumed), "blocks_skipped": sum(resumed)}}
            if resumed
            else {}
        ),
        **({"fires": fires} if fires else {}),
        # per write that landed ragged under a declared policy (§5 rule 19):
        # the policy, the per-row widths and the width buckets the executor
        # recorded before any forward — the receipt's `execution.ragged`
        **({RAGGED_KEY: ragged} if ragged else {}),
        **(
            {"implementations": sorted(getattr(executor, "applied_requirements", ()))}
            if getattr(executor, "applied_requirements", None)
            else {}
        ),
        # only when something was excluded: a point whose every cell measured
        # summarizes exactly as before
        **({"unavailable": excluded} if excluded else {}),
    }


def _featurizer_site(doc: Document, name: str) -> str | None:
    """The site a trained featurizer's own ``save`` entry restates (§2.12) —
    what its bundle's identity is stamped with, and so what a checkpoint of
    it is stamped with too. Every trained featurizer has one (rule 10)."""
    for entry in doc.save:
        if entry.kind is None and entry.value == name and entry.site is not None:
            return entry.site
    return None


def featurizer_identity(
    doc: Document,
    name: str,
    site_name: str | None,
    *,
    stage: Any = None,
    group_map: tuple[int, int] | None = None,
) -> dict[str, str]:
    """The ArtifactIdentity a trained featurizer bundle stamps (§8): what the
    document implies about the fit, plus whatever the fitted ``stage`` knows
    that the document only pointed at — a ``subspace`` seeded from a saved
    basis records that basis (``Stage.identity_fields``).

    ``group_map`` is the ``(groups, group_width)`` a grouped gate was built
    over — read off the trained stage, since the document authors only the
    group *kind* and the map is derived from the site (§2.5).

    ``trained_on`` is the ref the fit read on ``base`` — a human-readable
    name, recorded rather than expected at load: an apply document
    legitimately reads a different split than the fit trained on. Which
    bytes that ref resolved to is the run's record (the receipt's canonical
    form carries the data digests), not a header stamp.
    """
    spec = doc.featurizers[name]
    site = site_identity(doc, site_name)
    base = doc.data["base"]
    trained_on = base.dataset if not isinstance(base, tuple) else base[0].dataset
    realization = canonical_model(doc.raw["model"])
    return build_artifact_identity(
        **(stage.identity_fields() if stage is not None else {}),
        model_key=str(doc.model.key),
        model_revision=str(doc.model.revision),
        model_dtype=str(realization["dtype"]),
        model_quantization=realization.get("quantization"),
        model_attn_implementation=realization.get("attn_implementation"),
        site=site,
        k=spec.k if isinstance(spec.k, int) else None,
        # a gate stamps its effective map, `sigmoid` when unauthored: the hard
        # split a replay must reproduce depends on it (§2.5), and a bundle
        # fitted before the field existed is read as sigmoid by both checks
        parametrization=spec.parametrization
        if isinstance(spec.parametrization, str)
        else ("sigmoid" if spec.kind == "gate" else None),
        group=spec.group if isinstance(spec.group, str) else None,
        group_map=list(group_map) if group_map is not None else None,
        dtype=spec.dtype if isinstance(spec.dtype, str) else "fp32",
        trained_on=str(trained_on),
    )
