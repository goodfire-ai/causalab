"""Provide CLI reports for intervention specifications.

The commands share the compiler in ``pipeline``. Validation reports rule
violations; explanation describes the plan and required capabilities.
``dry_run`` reports the checks available before weights load. The run command
calls ``pipeline.run_protocol``.

This module handles argument dispatch, display, and exit codes."""

from __future__ import annotations

import argparse
import os
import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, get_args

from causalab.io.sources import Diagnostic
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.rules.errors import (
    ProtocolError,
    ValidationError,
    ValidationErrors,
)
from causalab.protocol.lowering import DEFAULT_POINT_CAP, point_count
from causalab.protocol.parallel import (
    ONE,
    ParallelGeometry,
    check_rows,
    context_refusals,
    format_geometry,
)
from causalab.protocol.parallel import check as check_geometry
from causalab.protocol.parallel_memory import Conversion, format_bytes
from causalab.protocol.pipeline import (
    build,
    check_engine,
    resolve_answers,
    resolve_positions,
    run_protocol,
    tokenizer_service,
    validate,
)
from causalab.protocol.publish import SOLO
from causalab.protocol.registry import (
    COMPONENT_STREAMS,
    ENGINES,
    INTERIOR_ROWS,
    Inventory,
    ModelInfo,
    capability,
    component_shape,
    component_width,
    effective_capabilities,
    inventory,
    predicate_holds,
    unavailable_at_load,
)
from causalab.protocol.schema import (
    LAYERLESS_COMPONENTS,
    MODEL_DTYPE_DEFAULT,
    Document,
    PositionSpec,
)

if TYPE_CHECKING:
    from causalab.io.env import ResolutionEnv

__all__ = [
    "SITE_STATUSES",
    "UNDECIDED_TOPICS",
    "CompositionReport",
    "DataReport",
    "DryRunReport",
    "EngineReport",
    "MemoryEstimate",
    "ModelReport",
    "OutputReport",
    "ParallelReport",
    "PointsReport",
    "ReadoutReport",
    "Refusal",
    "SiteReport",
    "SiteStatus",
    "TokenizationReport",
    "Undecided",
    "UndecidedTopic",
    "dry_run",
    "main",
    "site_report",
]


# --------------------------------------------------------------------------- #
# the two closed vocabularies of the report (tabulated in spec §9)
# --------------------------------------------------------------------------- #

#: What the registry entry can say about one site the document names.
#: ``available`` — the tensor exists, its shape and width are known and the
#: capability row says who reads and writes it; ``undecided`` — the entry
#: cannot decide a fact the run will (which mixer the layer carries, a
#: module-tree predicate, a measured address); ``refused`` — the row or the
#: entry says there is no such tensor (``V4``, ``component_unavailable``) or
#: the site's ``head`` names an axis the component has none of.
SiteStatus = Literal["available", "undecided", "refused"]
SITE_STATUSES: tuple[SiteStatus, ...] = get_args(SiteStatus)

#: The facts a dry run leaves to the run, by name — each is a line of the
#: report, never an omission. ``tokenization``: window counts, token widths
#: and answer tokens (sec. 2.3, sec. 2.10); ``pair_validity``: the
#: counterfactual pair's checks over rows and tokenizer (sec. 2.2);
#: ``controls``: declared one layer up, on the workflow; ``stream_at_layer``:
#: which mixer a layer carries on an entry without ``layer_types``;
#: ``module_tree``: a predicate or an address only the loaded modules decide;
#: ``inventory``: the per-layer inventory needs ``layer_types``; ``engines``:
#: which engine serves, when none was handed in; ``model``: the model key is
#: swept and the site facts are the first point's; ``memory``: the
#: ``--parallel`` memory estimate could not be computed (the checkpoint is
#: not in the local cache, the entry declares no plan, no tree matches) and
#: each rank measures its own card before its load.
UndecidedTopic = Literal[
    "tokenization",
    "pair_validity",
    "controls",
    "stream_at_layer",
    "module_tree",
    "inventory",
    "engines",
    "model",
    "memory",
]
UNDECIDED_TOPICS: tuple[UndecidedTopic, ...] = get_args(UndecidedTopic)


# --------------------------------------------------------------------------- #
# the records
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Refusal:
    """One refusal as a record: the code, the §5 rule (number and slug) when
    it is one, the field, the [`REASON_CODES`][causalab.protocol.rules.errors.REASON_CODES]
    entry when the refusal carries one, and the rendered text."""

    code: str
    rule: int | None
    rule_id: str | None
    path: str | None
    reason: str | None
    message: str

    @classmethod
    def from_error(cls, err: ProtocolError) -> Refusal:
        return cls(
            code=err.code,
            rule=getattr(err, "rule", None),
            rule_id=getattr(err, "rule_id", None),
            path=err.path,
            reason=err.reason,
            message=str(err),
        )

    def render(self) -> str:
        """The record on one line, after the rendered text: what the CLI
        prints so the reason code is visible, not only the rule."""
        where = f" at {self.path}" if self.path else ""
        rule = f" ({self.rule_id})" if self.rule_id else ""
        reason = f", reason {self.reason}" if self.reason else ""
        return f"code {self.code}{rule}{where}{reason}"


@dataclasses.dataclass(frozen=True)
class CompositionReport:
    """Dry-run fact 1 ([`DryRunReport`][]) — method/application composition: the campaign
    digest, the title, and the overrides that were applied on the way
    (``--set`` / a step's ``set``)."""

    digest: str
    title: str | None
    overrides: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class DataReport:
    """Dry-run fact 2 ([`DryRunReport`][]) — one resolved dataset ref (a split is a ref): the
    roles that read it, its content digest and its columns."""

    ref: str
    roles: tuple[str, ...]
    digest: str
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ModelReport:
    """Dry-run fact 5 ([`DryRunReport`][]) — the model configuration, from the registry entry
    (no config fetched): the realization the document names and the entry's
    static widths. ``layer_pattern`` is the entry's ``layer_types`` when it
    declares one — what decides the stream at a layer offline."""

    key: str
    revision: str
    dtype: str
    quantization: str | None
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    family: str | None
    layer_pattern: tuple[str, ...] | None


@dataclasses.dataclass(frozen=True)
class PointsReport:
    """Dry-run fact 7 ([`DryRunReport`][]) — sweep expansion: the point count and the axes
    (id, number of values) it came from."""

    n: int
    axes: tuple[tuple[str, int], ...]


@dataclasses.dataclass(frozen=True)
class EngineReport:
    """Dry-run fact 11 ([`DryRunReport`][]), for the named engine — whether it serves the
    campaign, what it lacks, and [`check_engine`][]'s
    refusal as a ``capability_shortfall`` diagnostic (reported, not raised).
    One per report since capability routing between engines was retired:
    ``--engine`` names one, and the report answers for it."""

    name: str
    serves: bool
    lacks: tuple[str, ...]
    shortfall: Diagnostic | None
    refusal: Refusal | None


@dataclasses.dataclass(frozen=True)
class SiteReport:
    """The dry run's site report — one site the document names, resolved from the registry
    entry alone: availability, tensor shape, width, head space, and who reads
    and writes it. ``layers`` is every layer the site takes across the points
    (a swept ``layer`` lists them all; a layer-less component lists none).
    ``writes`` is the mechanisms a write may use, ``None`` for a read-only
    row, with the row's ``why``."""

    name: str
    component: str
    layers: tuple[int, ...]
    head: int | None
    expert: Any
    stream: str | None
    shape: str | None
    width: int | None
    head_space: int | None
    reads: tuple[str, ...]
    writes: tuple[str, ...] | None
    why: str
    status: SiteStatus
    undecided: tuple[str, ...]
    refusal: Refusal | None

    def __post_init__(self) -> None:
        if self.status not in SITE_STATUSES:
            raise AssertionError(
                f"unknown site status {self.status!r}; expected one of {SITE_STATUSES}"
            )


@dataclasses.dataclass(frozen=True)
class ReadoutReport:
    """Dry-run fact 9 ([`DryRunReport`][]) — one read: where it taps, and the metrics that
    reduce it."""

    name: str
    site: str
    model: str
    input: str
    metrics: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class OutputReport:
    """Dry-run fact 10 ([`DryRunReport`][]) — one ``save`` entry: the value, its binding, the
    file it produces and, for a non-value entry, its kind."""

    value: str
    binding: str
    file_path: str
    kind: str | None


@dataclasses.dataclass(frozen=True)
class TokenizationReport:
    """What the model's tokenizer decided before any weights, under
    ``--tokenizer``
    ([`resolve_positions`][causalab.protocol.pipeline.resolve_positions],
    [`resolve_answers`][causalab.protocol.pipeline.resolve_answers]).

    ``tokenizers`` are the tokenizers loaded (``key@revision``) and
    ``positions`` the number of distinct position resolutions the
    representatives needed. ``metrics`` are the labels of the metrics whose
    answers resolved. ``when_scored`` are the labels of the metrics whose
    answers the pass left to the score: a saved metric over a continuation
    read, whose rows are known only after the decode. Their answers are not
    checked before the weights, and a value the tokenizer cannot score is
    refused when the run scores it."""

    tokenizers: tuple[str, ...]
    positions: int
    metrics: tuple[str, ...]
    when_scored: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Undecided:
    """One fact the dry run leaves to the run, by topic, with the reason."""

    topic: UndecidedTopic
    detail: str

    def __post_init__(self) -> None:
        if self.topic not in UNDECIDED_TOPICS:
            raise AssertionError(
                f"unknown undecided topic {self.topic!r}; expected one of "
                f"{UNDECIDED_TOPICS}"
            )


@dataclasses.dataclass(frozen=True)
class MemoryEstimate:
    """What the geometry places on each rank (``docs/model_parallelism.md``
    §11; [`causalab.protocol.parallel_memory`][]): the model's whole bytes
    at the document's ``dtype``, per rank the resident weights shard-on-read
    places and the card footprint the measured rule expects, the rule in
    words, and where the facts came from — the cached checkpoint's headers,
    read torch-free with no device in sight. ``conversion`` is what the
    load casts when the checkpoint is stored in another dtype than the
    document's (§5.3 "the load's peak under a dtype conversion": the host
    bytes staged per rank; the card footprint carries no conversion term);
    ``None`` when every tensor lands as stored."""

    dtype: str
    whole: int
    resident: tuple[int, ...]
    footprint: tuple[int, ...]
    rule: str
    source: str
    conversion: Conversion | None = None


@dataclasses.dataclass(frozen=True)
class ParallelReport:
    """The ``--parallel`` geometry a run was asked for, and every §2 refusal
    [`check`][causalab.protocol.parallel.check] makes of it against the entry
    the compile resolved (``docs/model_parallelism.md`` §2) — decided here,
    torch-free, before any weights. Present only when a geometry was
    passed; each refusal is also a ``refusals`` entry of the report.
    ``memory`` is the [`MemoryEstimate`][] for an accepted geometry whose
    checkpoint is in the local cache; otherwise ``memory_undecided`` says
    why there is none (the ranks check their own cards before loading, so
    an undecided estimate here is a status, never a green)."""

    geometry: ParallelGeometry
    refusals: tuple[str, ...]
    memory: MemoryEstimate | None = None
    memory_undecided: str | None = None
    #: The plan rows the derivation declined and this geometry would have
    #: applied — ``"embed_tokens (embedding_rowwise)"`` under ``tp > 1`` —
    #: so a reader sees that the embedding and the head are whole on every
    #: rank rather than inferring it (§6.1, §11); empty at ``tp=1``.
    unapplied: tuple[str, ...] = ()


def memory_estimate(
    info: ModelInfo, revision: str, dtype: str, geometry: ParallelGeometry
) -> MemoryEstimate | str:
    """The estimate off the cached checkpoint's headers (§11), or the reason
    there is none: the checkpoint is not cached, no registered tree matches
    its keys, the text tower cannot be told apart, or the entry has no plan.
    Torch-free; reads headers only, never a tensor."""
    from causalab.protocol.checkpoint_census import (
        cached_checkpoint_files,
        checkpoint_targets,
        read_headers,
        tree_of,
    )
    from causalab.protocol.parallel_memory import (
        DTYPE_ITEMSIZES,
        RULE,
        conversion,
        estimate_resident,
        placement_table,
        whole_bytes,
    )

    files = cached_checkpoint_files(info.key, revision)
    if files is None:
        return (
            f"the checkpoint {info.key}@{revision} is not in the local Hub cache; "
            "each rank checks its own device before loading"
        )
    if info.parallel_plan is None:
        return f"the registry entry for {info.key!r} declares no parallel plan"
    itemsize = DTYPE_ITEMSIZES.get(dtype)
    if itemsize is None:
        return f"model.dtype {dtype!r} has no byte size"
    headers = read_headers(files)
    tree = tree_of(headers, info.num_layers)
    if tree is None:
        return (
            f"no registered family tree matches the checkpoint's keys with "
            f"{info.num_layers} blocks"
        )
    targets = checkpoint_targets(headers, tree, info.num_layers)
    if targets is None:
        return "the checkpoint's text tower cannot be told apart from its other towers"
    table = placement_table(
        {name: header.elements for name, header in headers.items()},
        targets,
        info.parallel_plan,
        tree,
        info,
        dtypes={name: header.dtype for name, header in headers.items()},
    )
    return MemoryEstimate(
        dtype=dtype,
        whole=whole_bytes(table, itemsize),
        resident=estimate_resident(table, geometry, itemsize),
        footprint=RULE.footprint(table, geometry, itemsize),
        rule=RULE.describe(geometry),
        source=f"the headers of {len(files)} cached safetensors file(s)",
        conversion=conversion(table, geometry, dtype),
    )


@dataclasses.dataclass(frozen=True)
class DryRunReport:
    """Everything a run decides before weights load — twelve pre-flight
    facts and the per-site report, as one frozen record.

    The twelve facts, and where each is answered:

    1. method/application composition — ``composition``;
    2. datasets and splits — ``data``;
    3. counterfactual invariants — ``undecided`` (``pair_validity``): the
       pair's checks need rows and a tokenizer; the ``--data`` pass checks
       column existence and row roles here (rules 4, 20, 25), a refusal of
       which is a ``refusals`` entry;
    4. tokenization and semantic positions — ``undecided``
       (``tokenization``), never a green, unless ``--tokenizer`` loaded the
       tokenizer and resolved them: then ``tokenization``, and a refusal is a
       ``refusals`` entry;
    5. model configuration — ``model``;
    6. site names and expected tensor shapes — ``sites``, ``inventory``;
    7. sweep expansion — ``points``;
    8. controls — ``undecided`` (``controls``): declared on the workflow;
    9. readouts — ``readouts``;
    10. output schema — ``outputs``;
    11. required engine capabilities — ``capabilities``, ``engines``;
    12. estimated point and shard counts — ``points`` (the count is decided
        from the axes, [`point_count`][]; the
        forwards a point owes and a shard plan are the run's, which plans
        after it has enumerated — a dry run stops before weights or
        artifacts).

    ``refusals`` are the refusals the dry run *reports* rather than raises
    (no candidate engine serves; the ``--data`` pass; a ``--parallel``
    geometry the entry refuses); ``diagnostics`` are the compile's own;
    ``ok`` is "no refusals". ``parallel`` is the geometry fact
    (``docs/model_parallelism.md`` §2), present when one was asked for. A
    document that does not compile never becomes a report — [`dry_run`][]
    re-raises.
    """

    composition: CompositionReport
    data: tuple[DataReport, ...]
    model: ModelReport
    points: PointsReport
    capabilities: tuple[str, ...]
    engines: tuple[EngineReport, ...]
    sites: tuple[SiteReport, ...]
    inventory: Inventory | None
    readouts: tuple[ReadoutReport, ...]
    outputs: tuple[OutputReport, ...]
    refusals: tuple[Refusal, ...]
    undecided: tuple[Undecided, ...]
    diagnostics: tuple[Diagnostic, ...]
    parallel: ParallelReport | None = None
    tokenization: TokenizationReport | None = None

    @property
    def ok(self) -> bool:
        return not self.refusals

    @property
    def undecided_topics(self) -> tuple[UndecidedTopic, ...]:
        """The topics left to the run, each once, in report order."""
        seen: list[UndecidedTopic] = []
        for item in self.undecided:
            if item.topic not in seen:
                seen.append(item.topic)
        return tuple(seen)


# --------------------------------------------------------------------------- #
# the derivations
# --------------------------------------------------------------------------- #


def site_report(
    name: str,
    component: str,
    info: ModelInfo,
    *,
    layers: Sequence[int] = (),
    head: int | None = None,
    expert: Any = None,
    stream: str | None = None,
) -> SiteReport:
    """The dry run's report for one site, from the registry entry alone.

    Availability is what the row's predicates and the entry decide
    ([`unavailable_at_load`][causalab.protocol.registry.components.unavailable_at_load]); shape, width
    and head space are [`component_shape`][causalab.protocol.registry.components.component_shape]'s
    (a ``head`` narrows the width to one head's slice, or refuses on a
    component without a head axis); read and write support is the row's.
    What the entry cannot decide is named under ``undecided``: the stream at
    a layer for a stream-bound component on an entry without
    ``layer_types``, a predicate only the module tree answers, an
    attention-interior address the per-family tap table has not measured.
    A compiled document's sites are all ``available`` or ``undecided`` —
    the compile refused the rest — so ``refused`` is what a caller sees who
    asks about a site the compile has not yet seen.
    """
    row = capability(component)
    undecided: list[str] = []

    def refused(err: ProtocolError) -> SiteReport:
        return SiteReport(
            name=name,
            component=component,
            layers=tuple(layers),
            head=head,
            expert=expert,
            stream=stream,
            shape=None,
            width=None,
            head_space=None,
            reads=tuple(sorted(row.reads)),
            writes=None if row.writes is None else tuple(sorted(row.writes)),
            why=row.why,
            status="refused",
            undecided=(),
            refusal=Refusal.from_error(err),
        )

    unavailable = unavailable_at_load(info, component)
    if unavailable is not None:
        return refused(
            ValidationError(
                4,
                f"site {name!r}: {unavailable}",
                path=f"sites.{name}.component",
                reason="component_unavailable",
            )
        )
    try:
        shape = component_shape(info, component)
        width = shape.width
        if head is not None:
            width = component_width(info, component, head=head)
    except ValidationError as err:
        return refused(err)

    if stream is None:
        bound = COMPONENT_STREAMS.get(component)
        if bound is not None:
            stream = bound
        elif info.layer_types is not None and layers:
            at = {info.layer_types[layer] for layer in layers}
            stream = next(iter(at)) if len(at) == 1 else None
    if (
        component in COMPONENT_STREAMS
        and component not in LAYERLESS_COMPONENTS
        and info.layer_types is None
    ):
        undecided.append(
            f"exists only on a {COMPONENT_STREAMS[component]!r} mixer, and "
            f"model {info.key!r} declares no layer pattern (layer_types): which "
            "mixer each layer carries is decided against the loaded module"
        )
    open_predicates = [p for p in row.requires if predicate_holds(info, p) is None]
    if open_predicates:
        undecided.append(
            f"requires {sorted(open_predicates)}, which the entry cannot decide — "
            "the module tree does, at load"
        )
    if component in INTERIOR_ROWS and row.address_on(info) is None:
        undecided.append(
            "an attention-interior address the per-family tap table has not "
            f"measured for family {info.family!r}: served by measurement at "
            "load where that is unambiguous, refused where it is not"
        )
    return SiteReport(
        name=name,
        component=component,
        layers=tuple(layers),
        head=head,
        expert=expert,
        stream=stream,
        shape=shape.describe(),
        width=width,
        head_space=shape.head_space,
        reads=tuple(sorted(row.reads)),
        writes=None if row.writes is None else tuple(sorted(row.writes)),
        why=row.why,
        status="undecided" if undecided else "available",
        undecided=tuple(undecided),
        refusal=None,
    )


def _sites(docs: Sequence[Document], info: ModelInfo) -> tuple[SiteReport, ...]:
    """One report per distinct site the points name — a swept ``layer``
    collects into one report's ``layers``."""
    keyed: dict[tuple[str, str, Any, Any, Any], set[int]] = {}
    for doc in docs:
        for name, site in doc.sites.items():
            key = (name, str(site.component), site.head, site.expert, site.stream)
            layers = keyed.setdefault(key, set())
            if isinstance(site.layers, tuple):
                layers.update(site.layers)
    return tuple(
        site_report(
            name,
            component,
            info,
            layers=sorted(layers),
            head=head if isinstance(head, int) else None,
            expert=expert,
            stream=stream if isinstance(stream, str) else None,
        )
        for (name, component, head, expert, stream), layers in keyed.items()
    )


def _mentions_window(value: Any) -> bool:
    """Whether a position spelling has a tokenizer-decided window (a
    ``variable`` or ``column`` anchor — bare, or as the ``scope`` of an index
    — or ``all``) anywhere in it. A parsed [`PositionSpec`][causalab.protocol.schema.positions.PositionSpec] carries a
    scoped anchor as ``scope`` plus ``anchor_source``; a raw spelling nests
    it as ``{"scope": {"variable": …}}``."""
    if isinstance(value, PositionSpec):
        return (
            value.variable is not None
            or value.column is not None
            or bool(value.all)
            or (
                value.scope is not None
                and value.anchor_source in ("variable", "column")
            )
        )
    if isinstance(value, Mapping):
        return any(
            (key in ("variable", "column", "all") and item is not None)
            or _mentions_window(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_mentions_window(item) for item in value)
    return False


def _windowed(doc: Document) -> tuple[str, ...]:
    """The reads and writes whose position the tokenizer decides."""
    out: list[str] = []
    for kind, table in (("reads", doc.reads), ("writes", doc.writes)):
        for name, entry in table.items():
            pos = entry.pos
            if isinstance(pos, str) and pos in doc.positions:
                pos = doc.positions[pos]
            if _mentions_window(pos):
                out.append(f"{kind}.{name}")
    return tuple(out)


def _engine_report(compiled: CompiledProtocol, engine: str) -> EngineReport:
    """For the named engine, what [`check_engine`][] would refuse — as a
    ``capability_shortfall`` diagnostic, never a raise. The capability set is
    the registry's for the name; no engine is constructed. ``check_engine``
    orders the refusal as the run's own check does: a shortfall the
    *document* authored (a training field rule 30 decides) names the field,
    and the generated rule-13 text is the answer only when no rule has a
    narrower one."""
    offered = effective_capabilities(engine)
    lacks = tuple(sorted(compiled.capabilities - offered))
    try:
        check_engine(compiled, offered)
    except ValidationError as err:
        return EngineReport(
            name=engine,
            serves=False,
            lacks=lacks,
            shortfall=Diagnostic(
                "capability_shortfall", f"{engine}: {err}", path=err.path
            ),
            refusal=Refusal.from_error(err),
        )
    return EngineReport(engine, True, lacks, None, None)


def dry_run(
    document: CompiledProtocol | Path | Mapping[str, Any],
    env: ResolutionEnv,
    *,
    engine: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    check_data: bool = False,
    parallel: ParallelGeometry | None = None,
    tokenizer: bool = False,
) -> DryRunReport:
    """Resolve everything a run decides before weights load, and report it.

    ``document`` is a compiled result, a path or a tree; a path or a tree is
    built and validated here ([`build`][], then
    [`validate`][] without the data rules — the
    facade's pass) with ``overrides`` applied, and a compiled result is taken
    as is (then ``overrides`` only says what the caller applied). ``engine``
    is the registered engine *name* to ask
    [`check_engine`][] about — its capability
    set is the registry's, so nothing is constructed; ``None`` leaves the
    engine question ``undecided``. ``check_data`` runs the ``validate
    --data`` pass (column and prompt-variable existence at every axis value,
    rule 25's row roles) and reports its refusal instead of raising it.
    ``parallel`` is the ``--parallel`` geometry to check against the resolved
    entry (``docs/model_parallelism.md`` §2), its refusals reported, not
    raised. ``tokenizer`` loads the model's tokenizer and runs the run's two
    tokenizer passes before the weights, positions and every metric's
    answers (``--tokenizer``): a refusal is reported, and on success the
    ``tokenization`` topic is decided (``tokenization`` in the report).

    Never calls ``load_model``, ``execute`` or a config fetch: the model
    facts are the registry entry's, and an unregistered ``model.key`` is
    the compile's ``V4`` refusal, re-raised. Only ``tokenizer`` reads
    anything of the model's own, and only its tokenizer files.

    Raises what the compile raises ([`ParseError`][causalab.protocol.rules.errors.ParseError],
    [`ValidationError`][]); returns a
    [`DryRunReport`][] for every document that compiles, ``ok`` when
    nothing it reports is a refusal.
    """
    if engine is not None and engine not in ENGINES:
        # the same door-side check pipeline._offered runs for validate(engine=…)
        # — a name (``"auto"`` included: a routing policy the caller resolves
        # first) the registry does not know is refused here, before the
        # compile, rather than as the registry's own assertion
        raise ValueError(
            f"dry_run(engine={engine!r}) — not a registered engine's name; "
            f"expected one of {ENGINES}"
        )
    applied: Mapping[str, Any] = dict(overrides or {})
    if isinstance(document, CompiledProtocol):
        compiled = document
    else:
        base_directory = document.parent if isinstance(document, Path) else None
        compiled = validate(
            build(
                document, base_dir=base_directory, overrides=applied or None, env=env
            ),
            env=env,
            data=False,
        )
    # the checklist's domain: one concrete step per axis value, the first
    # the campaign's first step — every fact below that reads "every point"
    # reads every value every axis takes
    docs = compiled.representatives
    first = docs[0]
    undecided: list[Undecided] = []
    refusals: list[Refusal] = []

    # (5) the model, from the entry the compile already resolved
    info = env.model_info(str(first.model.key))
    keys = sorted({str(doc.model.key) for doc in docs})
    if len(keys) > 1:
        undecided.append(
            Undecided(
                "model",
                f"model.key is swept over {keys}; the model and site facts below "
                f"are the first point's ({info.key!r})",
            )
        )
    quantization = first.model.quantization
    model = ModelReport(
        key=info.key,
        revision=str(first.model.revision),
        dtype=str(first.model.dtype or MODEL_DTYPE_DEFAULT),
        quantization=(
            None
            if quantization is None
            else f"{quantization.scheme} ({quantization.method})"
        ),
        hidden_size=info.hidden_size,
        num_layers=info.num_layers,
        num_heads=info.num_heads,
        num_kv_heads=info.num_kv_heads,
        head_dim=info.head_dim,
        vocab_size=info.vocab_size,
        family=info.family,
        layer_pattern=None if info.layer_types is None else tuple(info.layer_types),
    )

    # (1) composition
    composition = CompositionReport(
        digest=compiled.digests.document,
        title=first.title,
        overrides=applied,
    )

    # (2) datasets and splits — every ref the points name, with its roles
    roles_of: dict[str, list[str]] = {}
    for doc in docs:
        for role, spec in doc.data.items():
            members = spec if isinstance(spec, tuple) else (spec,)
            for member in members:
                roles = roles_of.setdefault(str(member.dataset), [])
                if role not in roles:
                    roles.append(role)
    data = tuple(
        DataReport(
            ref=ref,
            roles=tuple(roles_of.get(ref, ())),
            digest=identity.digest,
            columns=tuple(identity.columns),
        )
        for ref, identity in sorted(compiled.data.items())
    )

    # (7), (12) points — the count from the axes, nothing enumerated
    points = PointsReport(
        n=point_count(compiled.axes),
        axes=tuple((axis.id, len(axis.values)) for axis in compiled.axes),
    )

    # (11) capabilities and, for the named engine, the shortfall
    engine_reports: tuple[EngineReport, ...] = ()
    if engine is None:
        undecided.append(
            Undecided(
                "engines",
                "whether the engine serves the campaign: none was named (pass "
                "--engine; the engines' capability sets are the registry's rows)",
            )
        )
    else:
        engine_reports = (_engine_report(compiled, engine),)
        if not engine_reports[0].serves:
            assert engine_reports[0].refusal is not None
            refusals.append(engine_reports[0].refusal)

    # (6) sites and the inventory
    sites = _sites(docs, info)
    for site in sites:
        for why in site.undecided:
            topic: UndecidedTopic = (
                "stream_at_layer" if "layer pattern" in why else "module_tree"
            )
            undecided.append(Undecided(topic, f"site {site.name!r}: {why}"))
    try:
        inv: Inventory | None = inventory(info)
    except ValidationError as err:
        inv = None
        undecided.append(Undecided("inventory", str(err)))

    # (9), (10) readouts and outputs
    readouts = tuple(
        ReadoutReport(
            name=ref.read,
            site=str(first.reads[ref.read].site),
            model=str(ref.model),
            input=first.group_of(ref)[1],
            metrics=tuple(
                agg.label
                for agg in first.aggregations()
                if agg.read == ref or agg.target == ref
            ),
        )
        for ref in first.read_refs()
    )
    outputs = tuple(
        OutputReport(
            value=entry.label,
            binding=(
                f"site={entry.site}"
                if entry.site is not None
                else f"model={entry.read.model}"
                if entry.read is not None
                else f"kind={entry.kind}"
            ),
            file_path=entry.file_path,
            kind=entry.kind,
        )
        for entry in first.save
    )

    # (3), (4), (8) — the run's, named here; (4) decided with --tokenizer
    tokenization: TokenizationReport | None = None
    if tokenizer:
        try:
            tokenization = _tokenizer_passes(compiled, env)
        except ProtocolError as err:
            refusals.append(Refusal.from_error(err))
    if tokenization is None:
        windowed = sorted({name for doc in docs for name in _windowed(doc)})
        positions = (
            f"the windows of {windowed} (a variable, column or all position) "
            if windowed
            else "every position's token index "
        )
        undecided.append(
            Undecided(
                "tokenization",
                positions + "and the rows' token widths are decided when the run "
                "encodes its inputs (sec. 2.3), as are the answer tokens a metric "
                "matches (sec. 2.10)",
            )
        )
    if check_data:
        try:
            validate(compiled, env=env, data=True)  # the `validate --data` pass
        except ProtocolError as err:
            refusals.append(Refusal.from_error(err))
        pair = (
            "column and prompt-variable existence and the declared row roles "
            "were checked at every point (the --data pass); "
        )
    else:
        pair = (
            "column and prompt-variable existence and the declared row roles "
            "are checked under --data; "
        )
    undecided.append(
        Undecided(
            "pair_validity",
            pair + "the counterfactual pair's validity (answer change, intended "
            "edit, no unintended edit, tokenizer stability) is checked over the "
            "rows and the tokenizer at run (sec. 2.2)",
        )
    )
    undecided.append(
        Undecided(
            "controls",
            "controls are declared one layer up, on the workflow that applies "
            "this document; a document dry run decides none",
        )
    )

    # the geometry (docs/model_parallelism.md §2): the §2 divisibility facts
    # against the entry the compile resolved — what the run would refuse,
    # reported here with no weights in sight
    parallel_report: ParallelReport | None = None
    if parallel is not None:
        # the rows mode's two document rules (§8.3) beside the model facts:
        # a train section to split, and pairs at least the replica count
        from causalab.protocol.receipt import decodes, fit_pairs

        geometry_refusals = (
            check_geometry(parallel, info)
            + check_rows(parallel, fit_pairs(compiled))
            + context_refusals(parallel, decodes(compiled), (info,), os.environ)
        )
        # the memory estimate (§11) for an accepted geometry, off the cached
        # headers; a refused geometry gets none, an uncached checkpoint the
        # reason — each rank measures its own card before its load, so the
        # reason is an ``undecided`` fact of the report, never a green
        memory: MemoryEstimate | None = None
        memory_undecided: str | None = None
        if not geometry_refusals:
            estimated = memory_estimate(info, model.revision, model.dtype, parallel)
            if isinstance(estimated, MemoryEstimate):
                memory = estimated
            else:
                memory_undecided = estimated
                undecided.append(Undecided("memory", estimated))
        # the declined vocabulary rows (§6.1, §11), off the plan the entry
        # declares — an entry declaring none has nothing to have declined
        plan = info.parallel_plan
        parallel_report = ParallelReport(
            geometry=parallel,
            refusals=geometry_refusals,
            memory=memory,
            memory_undecided=memory_undecided,
            unapplied=tuple(
                f"{pattern} ({style})" for pattern, style in plan.unapplied.items()
            )
            if parallel.tensor > 1 and plan is not None
            else (),
        )
        refusals.extend(
            Refusal(
                code="P4", rule=None, rule_id=None, path=None, reason=None, message=text
            )
            for text in geometry_refusals
        )

    return DryRunReport(
        composition=composition,
        data=data,
        model=model,
        points=points,
        capabilities=tuple(sorted(compiled.capabilities)),
        engines=engine_reports,
        sites=sites,
        inventory=inv,
        readouts=readouts,
        outputs=outputs,
        refusals=tuple(refusals),
        undecided=tuple(undecided),
        diagnostics=compiled.diagnostics,
        parallel=parallel_report,
        tokenization=tokenization,
    )


def _when_scored(labels: Sequence[str]) -> str:
    """The clause of a ``--tokenizer`` line that names the metrics whose
    answers the pass left to the score, or ``""`` when there are none."""
    if not labels:
        return ""
    return (
        f"; the answers of {list(labels)} are checked when scored, because "
        "they are read over generated tokens"
    )


def _tokenizer_passes(
    compiled: CompiledProtocol, env: ResolutionEnv
) -> TokenizationReport:
    """The run's two tokenizer passes before the weights, for
    ``--tokenizer``: positions, then every metric's answers, with one load
    per model. Raises what they raise."""
    tokenizers = tokenizer_service(env)
    resolved = resolve_positions(compiled, env=env, tokenizers=tokenizers)
    answers = resolve_answers(resolved, env=env, tokenizers=tokenizers)
    docs = resolved.representatives
    return TokenizationReport(
        tokenizers=tuple(sorted({f"{d.model.key}@{d.model.revision}" for d in docs})),
        positions=len(resolved.positions or {}),
        metrics=answers.resolved,
        when_scored=answers.when_scored,
    )


# --------------------------------------------------------------------------- #
# the CLI verbs (the body of protocol/cli.py)
# --------------------------------------------------------------------------- #


def _compile(
    args: argparse.Namespace,
    env: ResolutionEnv,
    *,
    data: bool = False,
    engine: str | None = None,
) -> CompiledProtocol:
    """The one pipeline, with the CLI's inputs: the file, its own directory
    for relative references, ``--set``, the environment, and — for
    ``validate`` — the named engine, held to by its *registered* capability
    set (no engine is constructed; ``run`` checks the built engine itself
    through ``route_engine``). ``data`` runs the rules that read the resolved
    tables too — the ``validate`` verb's pass."""
    compiled = build(
        args.document,
        base_dir=args.document.parent,
        overrides=dict(args.parsed_set),
        env=env,
        point_cap=args.max_points if args.max_points is not None else DEFAULT_POINT_CAP,
    )
    return validate(compiled, engine, env=env, data=data)


def main(args: argparse.Namespace, env: ResolutionEnv) -> int:
    """Run one verb against an **intervention specification**.

    Dispatch between document types lives in [`causalab.cli`][], so this module
    — and the whole ``protocol/`` package — links against nothing in the
    workflow layer. That is what lets someone use the intervention protocol on
    its own."""
    try:
        from causalab.cli import ensure_model_registered, wants_hf_registration

        if args.verb == "dry-run":
            # before the registration hook: a dry run never fetches a config
            return _dry_run(args, env)
        if wants_hf_registration(args):
            ensure_model_registered(args)
        # `validate` runs the rules that read the resolved tables too — the
        # pass `--data` used to opt into and now names the default; the flag
        # stays accepted and is never read (test_pipeline pins the no-op) —
        # and holds the document to the named engine (§5 rules 13 and 30, the
        # §8 shortfall), which is where a document the engine cannot serve
        # is refused now that routing between engines is retired
        compiled = _compile(
            args,
            env,
            data=args.verb == "validate",
            engine=args.engine if args.verb == "validate" else None,
        )
        if args.verb == "validate":
            # --tokenizer: the run's tokenizer passes before the weights
            tokenization = (
                _tokenizer_passes(compiled, env)
                if getattr(args, "tokenizer", False)
                else None
            )
            n = point_count(compiled.axes)
            print(
                f"OK: {args.document} — {n} point{'s' if n != 1 else ''}, "
                f"digest {compiled.digests.document[:16]}…"
            )
            if tokenization is not None:
                answers = (
                    f"and the answers of {list(tokenization.metrics)} resolve"
                    if tokenization.metrics
                    else "resolve"
                )
                if not tokenization.metrics and not tokenization.when_scored:
                    answers += "; no metric names answers"
                print(
                    f"tokenizer {', '.join(tokenization.tokenizers)}: positions "
                    + answers
                    + _when_scored(tokenization.when_scored)
                )
            return 0
        if args.verb == "digest":
            print(compiled.digests.document)
            return 0
        if args.verb == "explain":
            _explain(compiled)
            _explain_engine(compiled, args.engine)
            return 0
        # run — the one verb that constructs the engine: a lazily-imported
        # extra so the pure verbs stay torch-free; --engine named it
        from causalab.neural.shared.engine_router import route

        result = run_protocol(
            compiled,
            env,
            route(
                args.engine,
                device=args.device,
                cuda_graphs=getattr(args, "cuda_graphs", False),
                batch_rows=getattr(args, "batch_rows", None),
                fit_rows=getattr(args, "fit_rows", None),
                parallel=getattr(args, "parallel_geometry", ONE),
            ),
            args.out,
            points=args.points,
            record=getattr(args, "record", False),
            # this process's place in a launched world (causalab.cli sets it
            # for a world above 1); SOLO — world 1 — otherwise
            publisher=getattr(args, "publisher", SOLO),
        )
        for manifest_path, disk_path in sorted(result.files.items()):
            print(f"saved {manifest_path} -> {disk_path}")
        if result.cells:
            # the denominator is data (§4.1): how many cells measured, and
            # which were excluded and why — read from the result, not kept
            # by the campaign
            print(f"cells {result.denominator.render()}")
        return 0
    except ProtocolError as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1


def _dry_run(args: argparse.Namespace, env: ResolutionEnv) -> int:
    """``dry-run``: everything a run decides before weights load, resolved and
    reported ([`dry_run`][]).

    Exit ``0`` when the document compiles and every fact is resolved or
    explicitly undecided, with no shortfall for the named engine; ``1`` on
    any refusal — a compile refusal (printed as ``refused: …`` with its
    reason-coded record, exactly as ``validate`` refuses), the named
    engine's shortfall, or a ``--data`` refusal. No engine is constructed:
    the named engine's capabilities are the registry's rows.

    ``--register-from-hf`` is refused rather than inherited: the flag's one
    effect is a config fetch, and a dry run's contract is that it fetches no
    config — an unregistered ``model.key`` is the registry's ``V4``
    refusal. Only ``--tokenizer`` reads anything of the model's own: its
    tokenizer files, as a run loads them.
    """
    if getattr(args, "register_from_hf", False):
        raise ProtocolError(
            "P4",
            "--register-from-hf does not apply to dry-run: a dry run resolves "
            "the model from the registry alone and never fetches a config. An "
            "unregistered model.key is refused [V4]; register its static entry "
            "(causalab.protocol.registry.register_model), or pre-flight with "
            "'validate --register-from-hf'",
        )
    try:
        compiled = _compile(args, env)
    except ProtocolError as err:
        # the compile's refusal, plus the record: the rule's slug, the field
        # and the reason code — the reason is what the rendered text lacks
        print(f"refused: {err}", file=sys.stderr)
        each = err.errors if isinstance(err, ValidationErrors) else (err,)
        for violation in each:
            print(f"  {Refusal.from_error(violation).render()}", file=sys.stderr)
        return 1
    # the geometry fact (docs/model_parallelism.md §2): only when asked for,
    # so a report without the flag is byte-for-byte the report before it
    asked = getattr(args, "parallel", None) is not None
    report = dry_run(
        compiled,
        env,
        engine=getattr(args, "engine", None),
        overrides=dict(args.parsed_set),
        check_data=bool(getattr(args, "data", False)),
        parallel=getattr(args, "parallel_geometry", ONE) if asked else None,
        tokenizer=bool(getattr(args, "tokenizer", False)),
    )
    _print_dry_run(report, args.document)
    for refusal in report.refusals:
        print(f"refused: {refusal.message}", file=sys.stderr)
        print(f"  {refusal.render()}", file=sys.stderr)
    return 0 if report.ok else 1


def _print_dry_run(report: DryRunReport, document: Any) -> None:
    """The report, one block per fact, ending with the ``undecided`` line —
    so a run's tokenizer-time refusal is never mistaken for a green."""
    print(f"dry-run   {document}")
    print(f"digest    {report.composition.digest}")
    if report.composition.title:
        print(f"title     {report.composition.title}")
    if report.composition.overrides:
        applied = ", ".join(f"{k}={v}" for k, v in report.composition.overrides.items())
        print(f"overrides {applied}")
    model = report.model
    realization = f"{model.key}@{model.revision} {model.dtype}"
    if model.quantization is not None:
        realization += f" + {model.quantization}"
    print(f"model     {realization}")
    pattern = (
        "declares no layer pattern"
        if model.layer_pattern is None
        else "layer pattern "
        + ", ".join(
            f"{model.layer_pattern.count(s)} {s}"
            for s in sorted(set(model.layer_pattern))
        )
    )
    print(
        f"  {model.num_layers} layers, hidden {model.hidden_size}, "
        f"{model.num_heads} heads ({model.num_kv_heads} kv) x {model.head_dim}, "
        f"vocab {model.vocab_size}, family {model.family or 'unknown'}; {pattern}"
    )
    if report.parallel is not None:
        geometry = report.parallel.geometry
        verdict = (
            "accepted by the registry entry"
            if not report.parallel.refusals
            else f"refused ({len(report.parallel.refusals)})"
        )
        print(
            f"parallel  {format_geometry(geometry)} (world {geometry.world}): {verdict}"
        )
        for text in report.parallel.refusals:
            print(f"  {text}")
        if report.parallel.unapplied:
            # a vocabulary-sharding row the derivation declined (§6.1, §11):
            # the embedding and the head stay whole on every rank
            print(
                "  not applied: "
                + ", ".join(report.parallel.unapplied)
                + " — the embedding and the head are whole on every rank "
                "(docs/model_parallelism.md §6.1, §11)"
            )
        memory = report.parallel.memory
        if memory is not None:
            # the memory estimate (docs/model_parallelism.md §11): per rank
            # the weights shard-on-read places and the card footprint the
            # measured rule expects, off the cached headers, no device needed
            resident = ", ".join(
                f"rank{rank} {format_bytes(count)}"
                for rank, count in enumerate(memory.resident)
            )
            footprint = ", ".join(
                f"rank{rank} {format_bytes(count)}"
                for rank, count in enumerate(memory.footprint)
            )
            print(
                f"  resident weights ({memory.dtype}; the model is "
                f"{format_bytes(memory.whole)}): {resident}"
            )
            print(f"  estimated card footprint: {footprint}")
            if memory.conversion is not None:
                # the checkpoint is stored in another dtype than the
                # document's: the loader stages the cast on the host
                # (docs/model_parallelism.md §5.3), the card is unchanged
                print(f"  {memory.conversion.describe()}")
            print(f"  headroom rule: {memory.rule}")
        elif report.parallel.memory_undecided is not None:
            print(f"  memory: undecided — {report.parallel.memory_undecided}")
    print("data")
    for entry in report.data:
        roles = ", ".join(entry.roles) or "(no role)"
        print(
            f"  {entry.ref} ({roles}): digest {entry.digest[:16]}… "
            f"{len(entry.columns)} columns"
        )
    if report.points.axes:
        axes = ", ".join(f"{axis} ({n} values)" for axis, n in report.points.axes)
        print(f"axes      {axes}")
    print(f"points    {report.points.n}")
    print(f"requires  {list(report.capabilities) or 'nothing beyond a forward pass'}")
    for engine in report.engines:
        if engine.serves:
            print(f"engine    {engine.name}: serves")
        else:
            assert engine.shortfall is not None
            print(
                f"engine    {engine.name}: {engine.shortfall.kind} {engine.shortfall.message}"
            )
    print("sites")
    for site in report.sites:
        where = site.component
        if site.layers:
            layers = (
                f"layer {site.layers[0]}"
                if len(site.layers) == 1
                else f"layers {site.layers[0]}..{site.layers[-1]} ({len(site.layers)})"
            )
            where += f" {layers}"
        if site.head is not None:
            where += f" head {site.head}"
        if site.expert is not None:
            where += f" expert {site.expert}"
        if site.stream is not None:
            where += f" [{site.stream}]"
        print(f"  {site.name}: {where}: {site.status}")
        if site.refusal is not None:
            print(f"    {site.refusal.message}")
            print(f"    {site.refusal.render()}")
            continue
        heads = (
            "no head axis"
            if site.head_space is None
            else f"head space {site.head_space}"
        )
        print(f"    shape {site.shape}, width {site.width}, {heads}")
        writes = (
            f"read-only ({site.why})"
            if site.writes is None
            else "writes " + ", ".join(site.writes)
        )
        print(f"    reads {', '.join(site.reads)}; {writes}")
        for why in site.undecided:
            print(f"    undecided: {why}")
    if report.inventory is not None:
        streams = ", ".join(
            f"{report.inventory.count(s)} {s}"
            for s in sorted({layer.stream for layer in report.inventory.layers})
        )
        print(
            f"inventory {len(report.inventory.layers)} layers ({streams}); "
            f"layerless {', '.join(report.inventory.layerless)}"
        )
    else:
        print("inventory undecided (see below)")
    print("readouts")
    for read in report.readouts:
        metrics = ", ".join(read.metrics) or "(saved or operand only)"
        print(
            f"  {read.name}: {read.model} on {read.input} at {read.site} -> {metrics}"
        )
    print("save")
    for out in report.outputs:
        kind = f" [{out.kind}]" if out.kind else ""
        print(f"  {out.value} ({out.binding}) -> {out.file_path}{kind}")
    if report.tokenization is not None:
        # --tokenizer: what the tokenizer decided, before any weights
        tokenization = report.tokenization
        print(
            f"tokenizer {', '.join(tokenization.tokenizers)}: "
            f"{tokenization.positions} position resolution(s); answers of "
            f"{list(tokenization.metrics) or 'no metric'} resolve"
            + _when_scored(tokenization.when_scored)
        )
    for diagnostic in report.diagnostics:
        print(f"diagnostic {diagnostic.kind}: {diagnostic.message}")
    for refusal in report.refusals:
        print(f"refusal   {refusal.message}")
        print(f"  {refusal.render()}")
    for item in report.undecided:
        print(f"  {item.topic}: {item.detail}")
    print(
        "undecided (decided when the run encodes its inputs): "
        + ", ".join(report.undecided_topics)
    )


def _explain_engine(compiled: CompiledProtocol, engine: str) -> None:
    """Print the named engine's capability verdict: its name when it serves
    the document, or the §8 refusal.

    ``explain`` printed ``requires`` and stopped there, so the engine question
    could not be pre-flighted at all — and it is exactly what is not obvious
    on a model where one family of components is hooks-only and another is
    nnsight-only. The refusal is the *more* useful answer of the two, so it is
    printed rather than raised (``validate`` refuses; ``dry-run`` reports).

    No engine is constructed: the verdict is [`check_engine`][] against the
    registry's capability set for the name, so ``explain`` stays torch-free
    (``test_load_is_torch_free``).
    """
    try:
        check_engine(compiled, effective_capabilities(engine))
    except ValidationError as err:
        print(f"engine    refused: {err}")
    else:
        print(f"engine    {engine}")


def _explain(compiled: CompiledProtocol) -> None:
    """Print what the compile decided: the digest, the header's title and
    description, the realization, the axes and the step count (from the
    axes, nothing enumerated), the derived ``requires``, the first step's
    forward plan (its groups and what a decode obliges) and what ``save``
    produces.
    No per-step digest: the steps are the engine's to enumerate and sign,
    so ``explain`` stops where a compile does."""
    doc = compiled.representatives[0]
    axes = compiled.axes
    print(f"digest    {compiled.digests.document}")
    if doc.title:
        print(f"title     {doc.title}")
    # the author's intent (§1): authoring metadata the canonical form drops,
    # so this is the one place a reader of the compile sees it
    if doc.description:
        print(f"note      {doc.description}")
    model = doc.model
    realization = f"{model.key}@{model.revision} {model.dtype or MODEL_DTYPE_DEFAULT}"
    if model.quantization is not None:
        realization += f" + {model.quantization.scheme} ({model.quantization.method})"
    print(f"model     {realization}")
    if axes:
        print(
            f"axes      {', '.join(f'{a.id} ({len(a.values)} values)' for a in axes)}"
        )
    print(f"points    {point_count(axes)}")
    # the compiler's required-capability set (§8), read from the registry rows
    print(
        f"requires  {sorted(compiled.capabilities) or 'nothing beyond a forward pass'}"
    )
    # the first step's forward plan: its groups, and what a decode obliges.
    # Planning is the engine's; the planner is torch-free and imported
    # here the way `run` imports the router — the pure verb stays torch-free
    from causalab.neural.shared.plan import plan_point

    plan = plan_point(doc)
    print("plan")
    for group in plan.groups:
        taps = ", ".join(t.read for t in group.taps) or "(no reads — operands only)"
        print(f"  {group.model} on {group.input}: {taps}")
        if group.decode_depth:
            # print what the decode obliges, so the bill of a document is
            # readable before it runs — the mechanism stays the engine's
            print(f"    decode {group.decode_depth} tokens (greedy)")
            for item in group.materialize:
                needs = (
                    "distribution per addressed position"
                    if item.needs_distribution
                    else "no distribution — ids and activations only"
                )
                print(f"    {item.read} at {item.site}: {needs}")
    print("save")
    for entry in doc.save:
        if entry.site is not None:
            binding = f"site={entry.site}"
        elif entry.read is not None:
            binding = f"model={entry.read.model}" + (
                f", aggregation={entry.aggregation.kind}"
                if entry.aggregation is not None
                else ""
            )
        else:
            binding = f"kind={entry.kind}"
        print(f"  {entry.label} ({binding}) -> {entry.file_path}")


if __name__ == "__main__":
    raise SystemExit(main())
