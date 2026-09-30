"""Define run receipt fields and update recorded execution metadata.

The receipt records the document and engine used, execution bounds, and write
counts. Helpers update an existing record with measured row bounds or ragged
geometry. The neural layer writes the initial receipt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.publish import LAUNCHERS
from causalab.protocol.rules.errors import ProtocolError

if TYPE_CHECKING:
    from causalab.protocol.compiled import CompiledProtocol

__all__ = [
    "FIRES_KEY",
    "LAUNCHERS",
    "MODEL_SOURCES",
    "MODELS_KEY",
    "RUN_RECORD_NAME",
    "execution_record",
    "model_records",
    "record_models",
    "resolved_fit_rows",
    "FIT_ROWS_RESOLVED_KEY",
    "FIT_ROWS_SHRINKS_KEY",
    "RAGGED_KEY",
    "ragged_geometry",
    "record_ragged_geometry",
    "MEASURED_BOUNDS",
    "bound_shrinks",
    "fit_rows_shrinks",
    "measured_bounds",
    "record_measured_bounds",
    "resolved_bound",
    "parse_points",
    "record_fires",
]

#: The run receipt's filename inside the output directory. The exported
#: name keeps `record`: it is public API (``causalab.protocol.__all__``),
#: and §11.1 does not spend a compatibility break on a word.
RUN_RECORD_NAME = "protocol.json"


#: The receipt's fire-count block (§4 "Fires"): per point digest, per forward
#: group (``<model> on <input>``), how many times each write member fired per
#: forward — a state write, at how many distinct steps. An **observation of
#: the run**, so it sits beside the ``execution`` block rather than in it:
#: that block is the execution parameters the run was declared with, and its
#: contract closes it to anything else ([`execution_record`][]).
FIRES_KEY = "fires"


#: The receipt's list of the models the run ran (§8, execution provenance),
#: beside the ``execution`` block because it is observed at load: one entry
#: per distinct ``{"key", "revision", "resolved_revision"}``. The
#: ``resolved_revision`` is the commit the loader resolved the document's
#: ``revision`` to, read off the loaded config (``None`` when the weights did
#: not come through the Hub cache, such as a local directory). A document
#: names ``main``; this names the snapshot that ran. Each point's summary
#: carries its own entry under the singular ``model``
#: ([`model_records`][]).
MODELS_KEY = "models"


#: The closed vocabulary of ``execution.model_source`` (§8): ``loaded`` when
#: the engine loaded the document's model itself, ``caller`` when it ran a
#: caller-owned bundle handed to its constructor (§9, the ownership contract).
MODEL_SOURCES = frozenset({"loaded", "caller"})

# The closed vocabulary of ``execution.parallel.launcher`` is
# [`causalab.protocol.publish.LAUNCHERS`][] — ``solo`` (this process alone,
# no ``torch.distributed`` initialised, every world-1 run), ``spawned`` (one
# of the ``world`` local children the CLI's parent started) and ``joined`` (a
# process launched into a ``WORLD_SIZE`` group by ``torchrun`` or Slurm;
# ``docs/model_parallelism.md`` §3, §9) — re-exported here as public API.
assert LAUNCHERS == ("solo", "spawned", "joined")


def parallel_record(
    geometry: ParallelGeometry, launcher: str = LAUNCHERS[0]
) -> dict[str, Any]:
    """The receipt's ``execution.parallel`` block (§9): the five axes, the
    mode of the data axis (``points`` or ``rows``, §8.3), the ``world`` they
    multiply to, and the launcher — the word the process's
    [`Publisher`][causalab.protocol.publish.Publisher] carries, ``solo`` for a
    run in one process.

    Raises:
        AssertionError: ``launcher`` is not one of [`LAUNCHERS`][].
    """
    if launcher not in LAUNCHERS:
        raise AssertionError(f"launcher {launcher!r} is not one of {LAUNCHERS}")
    return {
        "data": geometry.data,
        "data_mode": geometry.data_mode,
        "pipeline": geometry.pipeline,
        "context": geometry.context,
        "tensor": geometry.tensor,
        "expert": geometry.expert,
        "world": geometry.world,
        "launcher": launcher,
    }


def parse_points(spec: str, n_points: int) -> range:
    """The ``--points`` shard selector: a half-open ``[start, stop)`` range.

    Refused rather than clamped when it falls outside ``[0, n_points]`` or
    selects nothing — a shard that silently became a different shard is worse
    than one that failed, because its artifacts would still stamp as members
    of the campaign.
    """
    try:
        start_text, stop_text = spec.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError:
        raise ProtocolError("P4", f"--points {spec!r} is not START:STOP") from None
    if not (0 <= start < stop <= n_points):
        raise ProtocolError(
            "P4",
            f"--points {spec!r} is outside the campaign's {n_points} points "
            "or selects none",
        )
    return range(start, stop)


def execution_record(
    engine: Any, request: Any = None, *, launcher: str = LAUNCHERS[0]
) -> dict[str, Any]:
    """The ``execution`` block of a run receipt: the batch geometry the chosen
    engine will run under, and where its model came from (§8, execution
    scale; §9, the ownership contract).

    Execution provenance has exactly one recorder, and this is it.
    ``batch_rows`` is the engine's microbatch bound — ``None`` when the engine
    runs every forward group whole, and for an engine that has no such bound
    at all (the nnsight engine). ``fit_rows`` is its rows-per-grad-forward
    bound for a fit — the members of a fit cohort are packed into forwards
    under it — ``None`` when the engine measures it, and for an engine with
    no grad path. With a ``request``, an engine that reads the request's
    ``execution`` block (a workflow step's overrides) reports the value it
    will run under through its ``effective_<bound>`` methods; an engine that
    never reads the block keeps its own value, so the record never claims an
    override that was not applied.
    ``model_source`` is ``"caller"`` when the
    engine was built around a caller-owned bundle and ``"loaded"`` when it
    loads the document's model itself ([`MODEL_SOURCES`][]; an engine that
    declares nothing loads). Both are read off the engine rather than declared
    by the document because they are execution, not identity: they enter no
    canonical document, no digest and no artifact stamp, so two layouts of one
    document — or the same weights loaded and handed in — differ in their
    receipts here and nowhere else. **Recorded, not gated**: a
    layout-dependent flip in a top-1 token is something a reader of two
    receipts can see and attribute, not something a run refuses over.
    ``device`` is the placement the engine was built with, its ``device``
    argument as given (the CLI's ``--device``: ``cpu``, ``mps``, ``cuda:1``,
    or a comma list spreading the layers), and ``None`` for an engine that
    declares none. Above world 1 every rank of a CUDA world runs on
    ``cuda:LOCAL_RANK`` (``docs/model_parallelism.md`` §3), so the block
    records the world's word ``cuda`` rather than the writing rank's own
    ordinal; the geometry is in ``parallel``. Like the row bounds it is a
    layout, so two placements of one document share every digest and differ
    in their receipts here.
    ``parallel`` is the geometry the engine was built with
    (``docs/model_parallelism.md`` §9; [`parallel_record`][]): the five
    axes, their ``world`` and the launcher, read off the engine's
    ``parallel`` — an engine declaring none runs at world 1 — so a document
    run at ``tp=4`` and the same document run on one device share every
    digest and differ in their receipts here; ``launcher`` is the word the
    process's [`Publisher`][causalab.protocol.publish.Publisher] carries
    (``solo`` unless a launcher set another). A later
    execution parameter adds its own key beside these; nothing else belongs
    in the block — what the run *observed* (the ``fires`` block,
    [`record_fires`][]; the ``scoring`` block) is recorded beside it,
    not in it.
    """
    model_source = getattr(engine, "model_source", "loaded")
    if model_source not in MODEL_SOURCES:
        raise AssertionError(
            f"engine {getattr(engine, 'name', engine)!r} reports model_source "
            f"{model_source!r}; expected one of {sorted(MODEL_SOURCES)}"
        )
    parallel = engine_geometry(engine)

    def bound(name: str) -> Any:
        # an engine that reads a request's `execution` block says what it
        # will run under (`effective_<name>`); one that never reads it keeps
        # its own value, so the receipt never claims an override nobody applied
        effective = getattr(engine, f"effective_{name}", None)
        if request is not None and callable(effective):
            return effective(request)
        return getattr(engine, name, None)

    return {
        "batch_rows": bound("batch_rows"),
        "device": _placement(engine, parallel),
        "fit_rows": bound("fit_rows"),
        "model_source": model_source,
        "parallel": parallel_record(parallel, launcher),
    }


def _placement(engine: Any, geometry: ParallelGeometry) -> str | None:
    """The ``execution.device`` word ([`execution_record`][]): the engine's
    ``device`` as given at world 1, and ``cuda`` for a CUDA world above it,
    whose ranks each run on ``cuda:LOCAL_RANK`` whatever ordinal the writing
    rank holds."""
    device = getattr(engine, "device", None)
    if device is None:
        return None
    word = str(device)
    if geometry.world > 1 and word.startswith("cuda"):
        return "cuda"
    return word


def engine_geometry(engine: Any) -> ParallelGeometry:
    """The geometry ``engine`` was built with, read off its ``parallel`` — an
    engine declaring none runs at world 1 (``docs/model_parallelism.md`` §9).

    Raises:
        AssertionError: the engine reports something that is not a geometry.
    """
    parallel = getattr(engine, "parallel", ONE)
    if not isinstance(parallel, ParallelGeometry):
        raise AssertionError(
            f"engine {getattr(engine, 'name', engine)!r} reports parallel "
            f"{parallel!r}; expected a ParallelGeometry"
        )
    return parallel


def fit_pairs(compiled: CompiledProtocol) -> tuple[int | None, ...]:
    """One entry per point of ``compiled``: its ``train.batch.pairs``, or
    ``None`` for a point with no ``train`` — what
    [`check_rows`][causalab.protocol.parallel.check_rows] reads of a document
    (``docs/model_parallelism.md`` §8.3). Torch-free; a point's ``pairs`` is
    concrete once the sweep is expanded."""
    from causalab.protocol.schema import concrete_int

    return tuple(
        None
        if point.train is None
        else concrete_int(point.train.batch["pairs"], "train.batch.pairs")
        for point in compiled.representatives
    )


def decodes(compiled: CompiledProtocol) -> bool:
    """Whether any point of ``compiled`` reads a ``generated`` position — a
    greedy continuation — what
    [`check_context`][causalab.protocol.parallel.check_context] reads of a document
    (``docs/model_parallelism.md`` §8.4). Torch-free."""
    from causalab.protocol.positions.encoding import generated_budget

    return any(
        generated_budget(point, read.pos) is not None
        for point in compiled.representatives
        for read in point.reads.values()
    )


#: The receipt's keys for a bound the run measured rather than authored (§8):
#: beside the ``null`` request, the number to pin as the bound to reproduce the
#: run — the bound every window ran under, after any shrink — and, only when
#: non-zero, the most windows any one cohort re-packed after running out of
#: memory. The pin is still safe: a grad shrink lowered the bound in place,
#: so a re-run at it packs as this run did; only an eval shrink leaves the
#: report below the grad bound, and a re-run pinned there packs its grad
#: windows smaller, which the drift tier's author should know.
FIT_ROWS_RESOLVED_KEY = "fit_rows_resolved"
FIT_ROWS_SHRINKS_KEY = "fit_rows_shrinks"


#: The measured bounds, one row each: the requested key (also the summaries'
#: key for the bound a fit ran under), the summaries' shrink key, the
#: receipt's resolved key and the receipt's shrink key. One row today —
#: ``fit_rows`` bounds a cohort's grad forwards and, unless ``batch_rows`` is
#: authored, its batched eval passes too.
MEASURED_BOUNDS: tuple[tuple[str, str, str, str], ...] = (
    ("fit_rows", "fit_rows_shrinks", FIT_ROWS_RESOLVED_KEY, FIT_ROWS_SHRINKS_KEY),
)


def resolved_bound(summaries: Sequence[Mapping[str, Any]], key: str) -> int | None:
    """The bound ``key`` (``fit_rows`` or ``batch_rows``) a run's fits
    reported, or ``None`` when none did: the **smallest** over the points. A
    run's cohorts measure independently (different frames, different
    parametrizations) and an authored bound is never shrunk, so the number
    safe to pin is the one every cohort of the run ran at or above."""
    bounds = [
        int(summary[key]) for summary in summaries if isinstance(summary.get(key), int)
    ]
    return min(bounds) if bounds else None


def resolved_fit_rows(summaries: Sequence[Mapping[str, Any]]) -> int | None:
    """[`resolved_bound`][] for ``fit_rows`` — what a receipt records as
    ``execution.fit_rows_resolved`` when the run authored no bound."""
    return resolved_bound(summaries, "fit_rows")


def bound_shrinks(summaries: Sequence[Mapping[str, Any]], key: str) -> int:
    """The most windows any one cohort of the run re-packed after running out
    of memory under the bound whose shrink key is ``key`` — a run that shrank
    is one whose measured bound was too loose, which the author reading the
    resolved number should know. The **largest** over the points, not a sum:
    a cohort's count is reported on each of its members, so a sum would
    multiply it by the cohort's size, and what the number answers is whether,
    and how badly, the measurement missed."""
    return max(
        (
            int(summary[key])
            for summary in summaries
            if isinstance(summary.get(key), int)
        ),
        default=0,
    )


def fit_rows_shrinks(summaries: Sequence[Mapping[str, Any]]) -> int:
    """[`bound_shrinks`][] for ``fit_rows`` — the grad windows and the eval
    windows packed under it."""
    return bound_shrinks(summaries, "fit_rows_shrinks")


def measured_bounds(
    execution: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """The keys a receipt's ``execution`` block gains from what the run's
    fits measured ([`MEASURED_BOUNDS`][]): for each requested bound that is
    ``null`` and that some fit reported, the resolved number, and its shrink
    count when non-zero. An authored bound is already the receipt's; a run
    with no fit, or off CUDA where a bound stays unbounded, adds nothing."""
    out: dict[str, int] = {}
    for requested, summary_shrinks, resolved_key, shrinks_key in MEASURED_BOUNDS:
        if execution.get(requested) is not None:
            continue
        resolved = resolved_bound(summaries, requested)
        if resolved is None:
            continue
        out[resolved_key] = resolved
        shrinks = bound_shrinks(summaries, summary_shrinks)
        if shrinks:
            out[shrinks_key] = shrinks
    return out


def record_measured_bounds(
    output_directory: Path, summaries: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Write the bounds the run measured ([`measured_bounds`][]) into the
    receipt's ``execution`` block and return the receipt's path — the same
    amend-after-execution as [`record_fires`][]. A caller that wrote no
    receipt has nothing to amend: ``None``."""
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text())
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return receipt
    added = measured_bounds(execution, summaries)
    if not added:
        return receipt
    execution.update(added)
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


#: The receipt's ``execution`` key for the ragged-write geometry a run landed
#: under (intervention protocol spec §5 rule 19, §8): present **only**
#: when some write ran under a non-``refuse`` ``ragged`` policy, so a receipt
#: of a document that authors none is byte-identical to before the key.
RAGGED_KEY = "ragged"


def ragged_geometry(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``execution.ragged`` block from the points' summaries: per
    ``"<intervened_model>/<write>"`` that landed ragged, the policy, the
    per-row widths and the ``[width, rows]`` buckets — the geometry the
    executor recorded at its pre-forward width check (``check_write_widths``).

    Points of one document address the same rows, so a write's geometry is
    ordinarily one value across them; a sweep over the write's position can
    make it differ per point, and then ``widths`` / ``buckets`` are ``None``
    at the top and each point's own are recorded under ``by_point``, keyed by
    point digest — the block keeps one shape either way. Empty when no point
    landed a ragged write."""
    seen: dict[str, dict[str, Any]] = {}
    by_point: dict[str, dict[str, dict[str, Any]]] = {}
    for summary in summaries:
        entries = summary.get(RAGGED_KEY)
        if not isinstance(entries, Mapping):
            continue
        point = str(summary.get("point"))
        for key, geometry in entries.items():
            by_point.setdefault(key, {})[point] = dict(geometry)
            seen.setdefault(key, dict(geometry))
    out: dict[str, Any] = {}
    for key, geometry in seen.items():
        variants = by_point[key]
        shapes = {json.dumps(g, sort_keys=True) for g in variants.values()}
        if len(shapes) == 1:
            out[key] = geometry
        else:
            out[key] = {
                "policy": geometry["policy"],
                "widths": None,
                "buckets": None,
                "by_point": {
                    point: {"widths": g["widths"], "buckets": g["buckets"]}
                    for point, g in sorted(variants.items())
                },
            }
    return out


def record_ragged_geometry(
    output_directory: Path, summaries: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Write the ragged-write geometry the run landed under
    ([`ragged_geometry`][]) into the receipt's ``execution`` block as
    [`RAGGED_KEY`][] and return the receipt's path — the same
    amend-after-execution as [`record_measured_bounds`][], and like it a
    fact **recorded, not gated**. Nothing is written, and no key appears, when
    no write landed ragged; a caller that wrote no receipt has nothing to
    amend: ``None``."""
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    geometry = ragged_geometry(summaries)
    if not geometry:
        return receipt
    record = json.loads(receipt.read_text())
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return receipt
    execution[RAGGED_KEY] = geometry
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


def model_records(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The [`MODELS_KEY`][] list from the points' summaries: each distinct
    ``{"key", "revision", "resolved_revision"}`` a point reported under
    ``model``, sorted by key, revision and commit.

    Points of one campaign usually load one model. A swept ``model.key``
    gives one entry per model. Two entries with one ``key`` and ``revision``
    mean the ref moved between two loads of the run, which the list keeps
    visible rather than choosing one. ``resolved_revision`` is null when the
    loaded model has no Hub commit, such as a local directory or an executor
    without a loaded config; a null entry sorts before a resolved one. The
    list is empty only when no summary carries a ``model``: the shared
    execution reports one for every point, so only an engine that skips it
    (a test stub) gives none."""
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for summary in summaries:
        model = summary.get("model")
        if not isinstance(model, Mapping):
            continue
        entry = {
            "key": model.get("key"),
            "revision": model.get("revision"),
            "resolved_revision": model.get("resolved_revision"),
        }
        order = (
            str(entry["key"]),
            str(entry["revision"]),
            str(entry["resolved_revision"] or ""),
        )
        seen.setdefault(order, entry)
    return [seen[order] for order in sorted(seen)]


def record_models(
    output_directory: Path, summaries: Sequence[Mapping[str, Any]]
) -> Path | None:
    """Write the models the run ran ([`model_records`][]) into the receipt
    as its [`MODELS_KEY`][] list and return the receipt's path. The same
    amend-after-execution as [`record_fires`][]: the commit is known once a
    point has loaded its model, so a run that fails before that keeps a
    receipt without the key. Nothing is written when no point reported a
    model; a caller that wrote no receipt has nothing to amend: ``None``."""
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    models = model_records(summaries)
    if not models:
        return receipt
    record = json.loads(receipt.read_text())
    record[MODELS_KEY] = models
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt


def record_fires(
    output_directory: Path, fires: Mapping[str, Mapping[str, Mapping[str, int]]]
) -> Path | None:
    """Write the run's fire counts into the receipt as its ``fires`` block
    ([`FIRES_KEY`][]) and return the receipt's path.

    ``fires`` is ``{point digest: {"<model> on <input>": {write: count}}}``
    — per forward group with writes, how many times each member fired per
    forward (§4 "Fires"; ``neural/shared/fires.py``). The engine writes it
    **once the whole campaign has run**, never per point: a point whose member
    fired other than its declared count refuses the run before this is
    called, so a refused run's receipt carries no ``fires`` key at all rather
    than the counts of the points before it — the same all-or-nothing the
    tables keep. A caller that wrote no receipt (a workflow step records its
    run in ``_step.json``; an engine test drives ``execute_request``
    directly) has nothing to amend: ``None``.

    The counts are layout-invariant — the same under ``--batch-rows`` as
    whole — so two layouts of one document still differ in their receipts at
    ``execution.batch_rows`` and nowhere else (§8).
    """
    receipt = output_directory / RUN_RECORD_NAME
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text())
    record[FIRES_KEY] = {
        point: {group: dict(counts) for group, counts in groups.items()}
        for point, groups in fires.items()
    }
    receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return receipt
