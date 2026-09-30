"""Compile, validate, and execute intervention specifications.

``build`` resolves sources and overrides, lowers document forms, creates the
canonical form, and records identities. ``validate`` checks the compiled object
against the document rules, data tables, and selected engine capabilities.
``compile_protocol`` composes those operations for Python callers.

``resolve_positions`` uses the configured tokenizer before weights load, and
``resolve_answers`` resolves every metric's answers with it.
``handoff`` validates a run, resolves positions and answers, and calls the
engine.
``run_protocol`` builds the run context and serves both Python and CLI callers.
The engine enumerates points, executes them, and writes the run receipt.

Imports remain independent of tensor libraries until a run needs their services."""

from __future__ import annotations

import os

import dataclasses
import json
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Literal,
    Mapping,
    Sequence,
    cast,
    get_args,
)

from causalab.io.env import ArtifactStore, DatasetResolver, ResolutionEnv
from causalab.io.sources import (
    DataIdentity,
    Diagnostic,
    ResolvedArtifact,
    apply_overrides,
    check_json_values,
    identify,
    load_text,
    resolve_artifact_fields,
)
from causalab.protocol import identity as _identity
from causalab.protocol.answers import metric_token_ids, names_answers
from causalab.protocol.compiled import Authored, CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult, check_steps_signed
from causalab.protocol.lowering import (
    CAP_MESSAGE,
    DEFAULT_POINT_CAP,
    Axes,
    Axis,
    axes_of,
    canonical_axes,
    expand_families,
    has_axes,
    lower_axes,
    parse_axes,
    point_count,
    representative_trees,
)
from causalab.protocol.registry import ENGINES
from causalab.protocol.registry.engines import effective_capabilities
from causalab.protocol.rules.capability import check_engine_support, refuse_shortfall
from causalab.protocol.publish import SOLO, Publisher
from causalab.protocol.receipt import parse_points
from causalab.protocol.rules.data import (
    check_data_columns,
    check_loaded_featurizers,
    check_row_roles,
)
from causalab.protocol.rules.document import validate_document
from causalab.protocol.rules.errors import (
    ProtocolError,
    ProtocolWarning,
    ValidationError,
    raise_distinct,
)
from causalab.protocol.schema import (
    BoundAggregation,
    Document,
    PositionSpec,
    SpanSpec,
    check_protocol_version,
    metric_column_fields,
    parse_document,
)
from causalab.protocol.schema import explicit as _explicit

__all__ = [
    "STAGES",
    "AnswerCheck",
    "Stage",
    "build",
    "check_engine",
    "compile_protocol",
    "handoff",
    "read_document",
    "resolve_answers",
    "resolve_positions",
    "route_engine",
    "run_protocol",
    "tokenizer_service",
    "validate",
]


# --------------------------------------------------------------------------- #
# the stages, in order
# --------------------------------------------------------------------------- #

#: The build, in the order it runs. Each name is a stage function below;
#: [`build`][] walks this tuple and nothing else decides the order. Spec §9
#: prints the same table (with ``validate`` and ``route`` beside them, which
#: are [`validate`][]'s), and ``test_compile_protocol.py`` holds the two
#: together — a new stage is one entry here and one row there.
Stage = Literal[
    "read",
    "override",
    "resolve",
    "families",
    "axes",
    "gate",
    "canonicalize",
    "digest",
    "identify",
]
STAGES: tuple[Stage, ...] = get_args(Stage)


@dataclasses.dataclass
class _Build:
    """The build in progress: the inputs, then what each stage adds."""

    source: Path | Mapping[str, Any] | Authored
    base_directory: Path | None
    overrides: Mapping[str, Any] | None
    env: ResolutionEnv
    point_cap: int | None
    # read
    authored: dict[str, Any] = dataclasses.field(default_factory=dict)
    # resolve + families + axes + gate
    explicit: dict[str, Any] = dataclasses.field(default_factory=dict)
    axes: Axes | None = None
    document: Document | None = None
    found: tuple[Axis, ...] = ()
    # the checklist's domain (one representative per axis value), parsed on
    # first need: every one parsed so far is kept for the guard in `build`
    representatives: tuple[Document, ...] | None = None
    parsed: tuple[Document, ...] = ()
    # canonicalize + digest (the campaign alone)
    canonical: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    digest: str | None = None
    # identify
    data: dict[str, DataIdentity] = dataclasses.field(default_factory=dict)
    artifacts: list[ResolvedArtifact] = dataclasses.field(default_factory=list)
    diagnostics: list[Diagnostic] = dataclasses.field(default_factory=list)


def _read(build: _Build) -> None:
    """Read the authored source and check it is an intervention specification
    of the version this compiler reads (§1).

    The version check comes first so that a workflow document, or a v1
    document, is refused as that — before the override stage addresses the
    tree by path and would refuse it as a path that does not exist. An
    [`Authored`][] source has been through this stage already and is taken
    as is."""
    source = build.source
    if isinstance(source, Authored):
        build.authored = dict(source.raw)
        return
    raw = dict(load_text(source)) if isinstance(source, Path) else dict(source)
    check_protocol_version(raw)
    build.authored = raw


def _override(build: _Build) -> None:
    """``--set`` / a step's ``set`` (§9), on the authored tree, by
    section-rooted path (§1) — and the digest is the overridden document's."""
    if build.overrides:
        build.authored = apply_overrides(build.authored, build.overrides)


def _resolve(build: _Build) -> None:
    """Artifact-valued fields resolve first (§1: legal anywhere a value is),
    then the tree is held to the JSON object model whatever surface produced
    it."""
    build.explicit = resolve_artifact_fields(build.authored, build.env)
    check_json_values(build.explicit)


def _families(build: _Build) -> None:
    """``at_once`` families materialize into the entries they denote (§3.1,
    rule 26).

    Here, and not later, is the whole of why they are *sugar*: every stage
    after this one — the shape gate, the checklist, sweep expansion, the
    canonical form, the digest — sees exactly the document the author would
    have written out by hand. A no-op on any specification that declares no
    family, which is every one written before §3.1. Families live in the
    ``method`` group (§1), which is where the expansion looks; a tree without
    one is the shape gate's refusal to make, next."""
    build.explicit = expand_families(build.explicit)


def _axes(build: _Build) -> None:
    """Named axes — correlated row tuples and dependent axes (§3.2) — parse,
    and the document lowers to its **display form**: every ``{"axis": …}``
    wrapper becomes the ``{"sweep": [column]}`` it stands for and the ``axes``
    group is removed, so the gate and ``CompiledProtocol.tree`` see a
    document any swept one could be. After families, so a wrapper a family
    entry carries has been copied to every member before the references are
    found; before the gate, which knows no fifth group. Any stage that
    rewrites the tree runs no later than this one: the parsed axes are kept,
    and the engine builds every point from the tree they were parsed against.

    The parsed axes are kept: the engine's enumeration walks *them* — the
    correlated rows as the rows they are, never as the display form's cross
    product — and ``canonicalize`` writes the block into the campaign's
    canonical form. A no-op on any document without the group, which is
    every one written before §3.2."""
    if not has_axes(build.explicit):
        return
    build.axes = parse_axes(build.explicit, build.env.model_info)
    build.explicit = lower_axes(build.explicit, build.axes)


def _gate(build: _Build) -> None:
    """The strict parse of the explicit form, sweep wrappers intact (rules 1
    and 2): a shape gate on what the engine is about to enumerate. Then the
    axes the steps are indexed by ([`axes_of`][] — the named axes' rows slowest, the sweep axes the first row's
    tree carries, §3.2), and rule 14's cap decided from their sizes alone
    ([`point_count`][]): a campaign over
    ``point_cap`` is refused without an explicit override, with the text the
    expansion always gave, and nothing is enumerated to decide it."""
    build.document = parse_document(build.explicit)
    build.found = axes_of(build.explicit, build.axes)
    total = point_count(build.found)
    if build.point_cap is not None and total > build.point_cap:
        raise ValidationError(14, CAP_MESSAGE.format(total=total, cap=build.point_cap))


def _representatives(build: _Build) -> tuple[Document, ...]:
    """The checklist's domain, parsed once: one representative step per axis
    value ([`representative_trees`][]), each
    through the same gate the campaign passed, so a representative that is
    not a well-formed document is a could-not-build
    ([`ParseError`][causalab.protocol.rules.errors.ParseError]) here, before any rule.

    The representatives parsed before a failing one are kept on the build
    state: the guard in [`build`][] consults the checklist over them, so a
    step that does not parse after a step that broke a rule reports the rule
    — the compiler stopped its pass there and reported the earlier refusal,
    as it always had."""
    if build.representatives is not None:
        return build.representatives
    parsed: list[Document] = []
    try:
        for tree in representative_trees(build.explicit, build.found, build.axes):
            parsed.append(parse_document(tree))
    finally:
        build.parsed = tuple(parsed)
    build.representatives = tuple(parsed)
    return build.representatives


def _canonicalize(build: _Build) -> None:
    """The canonical document (wrappers intact — the campaign; the ``axes``
    block beside them when one was authored, §3.2), §7 — and every
    representative's canonical form, materialised and discarded: the
    provenance units are the engine's to sign, but the materialisation
    refuses too — a featurizer fitting a basis on a component that is not a
    feature space, a gate group the site's map cannot carry (rule 23) — and
    it decides per step, from the model's static metadata. Those refusals
    are collected across the representatives exactly as the checklist's are:
    distinct ones together, identical ones once. (A site's layer, stream,
    component and head are the checklist's since rule 4's address half moved
    into it; [`build`][]'s guard consults the checklist before a refusal
    raised here gets out, so an illegal site is reported as the rule 4 it
    is.) The representatives are parsed first, so the guard has them."""
    _representatives(build)
    build.canonical = _explicit.canonicalize(
        build.explicit,
        build.env,
        axes=None if build.axes is None else canonical_axes(build.axes),
    )
    refused: list[ValidationError] = []
    for tree in representative_trees(build.explicit, build.found, build.axes):
        try:
            _explicit.canonicalize(tree, build.env)
        except ValidationError as err:
            refused.append(err)
    raise_distinct(refused)


def _digest(build: _Build) -> None:
    """``sha256`` of the canonical bytes — the campaign (§7). The step
    digests are the engine's, signed with the same hasher as each step is
    enumerated ([`causalab.protocol.identity.sign_step`][])."""
    build.digest = _identity.digest(build.canonical)


def _identify(build: _Build) -> None:
    """What the document reached outside itself, as resolved: every dataset
    ref's identity and schema, every artifact reference and whether the store
    deferred its check — [`causalab.io.sources.identify`][] over the
    authored tree (where the value references are) and the representatives
    (where the data roles and the ``file_path`` loads are: every axis value
    once, so every ref and every load some step names is reached)."""
    build.data, build.artifacts, build.diagnostics = identify(
        build.authored, _representatives(build), build.env
    )


_STAGE: Mapping[Stage, Callable[[_Build], None]] = {
    "read": _read,
    "override": _override,
    "resolve": _resolve,
    "families": _families,
    "axes": _axes,
    "gate": _gate,
    "canonicalize": _canonicalize,
    "digest": _digest,
    "identify": _identify,
}


# --------------------------------------------------------------------------- #
# the entry points
# --------------------------------------------------------------------------- #


def read_document(
    authored_document: Path | Mapping[str, Any],
    base_directory: Path | None,
    overrides: Mapping[str, Any] | None,
) -> Authored:
    """The read prefix — [`STAGES`][] up to and including ``override`` — on
    its own.

    For the caller that has to see the authored tree before it can choose a
    resolver (a workflow, deciding whether an inner document depends on a
    step), and for the ``--register-from-hf`` pre-pass, which needs the
    overridden ``model.key`` before anything resolves. The result goes back
    into [`compile_protocol`][] (or [`build`][]) as the authored document, so reading here
    and compiling there is one implementation run in two halves."""
    build = _Build(
        source=authored_document,
        base_directory=base_directory,
        overrides=overrides,
        env=ResolutionEnv(
            datasets=cast(DatasetResolver, _NoResolver()),
            artifacts=cast(ArtifactStore, _NoResolver()),
        ),
        point_cap=None,
    )
    _STAGE["read"](build)
    _STAGE["override"](build)
    return Authored(raw=build.authored)


class _NoResolver:
    """Stands where a resolver would in [`read_document`][], which never
    resolves. Reaching it is a bug in the stage order, and says so."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"read_document() resolves nothing (asked for {name!r})")


def build(
    source: Path | Mapping[str, Any] | Authored,
    *,
    base_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: ResolutionEnv,
    point_cap: int | None = DEFAULT_POINT_CAP,
    offered: Collection[str] | None = None,
) -> CompiledProtocol:
    """Build one intervention specification into the object every door reads:
    [`STAGES`][] in order — read, override, resolve, families, paths, axes,
    gate, canonicalize, digest, identify.

    ``source`` is a path, the document as a tree, or an [`Authored`][]
    prefix. ``base_dir`` is the directory a caller says the document's
    relative references resolve from, kept on the build state for the stage
    that will read it (today the doors hand the resolvers the document's own
    directory themselves, as the compiler's callers did). ``overrides`` are
    ``--set`` / a step's ``set`` (section-rooted dotted paths, §1, §9).
    ``env`` is the resolution environment — the dataset and artifact services
    and the static model metadata — none of which the result keeps: it
    records the *identities* the environment resolved to, never the
    environment. ``point_cap`` is ``--max-points`` / a step's ``max_points``
    (rule 14).

    Refuses only what cannot be built: a source that is not a document
    ([`ParseError`][causalab.protocol.rules.errors.ParseError]), a dependency that is not
    there — a dataset, an artifact, a registry row (the ``V4`` / ``V15``
    refusals the resolvers raise). No §5 rule decides whether a document
    builds; that is [`validate`][]'s job, and a document with a rule
    violation and nothing else *builds*. The one impurity is the canonical
    form's own rule-23 and derived-width refusals during materialisation (the
    module docstring says why it is left there).

    **Which refusal, when there are two.** A stage that refuses after the
    representatives are parsed — ``canonicalize`` on a representative that
    does not parse or on the impurity, ``identify`` (a dataset or a bundle
    that is not there) — would pre-empt a rule the checklist reports on the
    same document, and the compiler, which ran the checklist between
    ``expand`` and ``canonicalize``, reported the rule. So the checklist over
    the representatives parsed so far is consulted *before such a refusal
    gets out*, and its violation is the report when there is one; a document
    that builds never enters the checklist here. ``offered`` is the engine's capability set for
    that consultation alone — the compiler decided rules 13 and 30 in the same
    pass when it was handed the offered set, so [`compile_protocol`][]
    hands it on and a two-defect document keeps the rule it always reported.
    It decides nothing about whether a document builds, and no other door
    passes it.
    """
    guard = None if offered is None else frozenset(offered)
    state = _Build(
        source=source,
        base_directory=base_dir,
        overrides=overrides,
        env=env,
        point_cap=point_cap,
    )
    for stage in STAGES:
        try:
            _STAGE[stage](state)
        except ProtocolError:
            _checklist(state.parsed, state.env, guard)
            raise
    if state.document is None or state.digest is None:
        raise AssertionError("the stage list did not run to completion")
    return CompiledProtocol(
        document=state.document,
        explicit=state.canonical,
        tree=state.explicit,
        axes=state.found,
        named_axes=state.axes,
        campaign_digest=state.digest,
        data=dict(state.data),
        artifacts=tuple(state.artifacts),
        diagnostics=tuple(state.diagnostics),
    )


def _checklist(
    documents: Sequence[Document],
    env: ResolutionEnv,
    offered: frozenset[str] | None,
) -> None:
    """The §5 checklist over every representative and every ``file_path``
    load against the document that names it (§2.5/§8) — every
    representative, not the first failing one, distinct violations together
    ([`raise_distinct`][]). Rules 13 and 30
    are decided exactly when ``offered`` is given; rule 29 reads the
    operands' widths from ``env.model_info``, never from weights.

    [`validate`][]'s first group, and the guard [`build`][] consults
    before a stage's refusal gets out."""
    refused: list[ValidationError] = []
    for pdoc in documents:
        try:
            validate_document(
                pdoc, engine_capabilities=offered, model_info=env.model_info
            )
            check_loaded_featurizers(pdoc, env)
        except ValidationError as err:
            refused.append(err)
    raise_distinct(refused)


def _offered(engine: str | Engine | Collection[str] | None) -> frozenset[str] | None:
    """What the engine offers, as a capability set: ``None`` for no engine; a
    registered engine's *name* ([`effective_capabilities`][] — never ``"auto"``, which is not an engine but a
    routing policy the caller resolves first); an [`Engine`][] (its ``effective_capabilities``); or the offered set
    itself, for the caller that knows only what an engine offers (the
    keyword-only [`compile_protocol`][]'s ``engine``). A name the
    registry does not know is a ``ValueError`` (the registry's own assertion
    is for its callers, not a door's user); anything else — an ``Engine``
    *class*, a number, a collection of anything but names — is a
    ``TypeError``, never a capability set by accident."""
    if engine is None:
        return None
    if isinstance(engine, str):
        if engine == "auto":
            raise ValueError(
                "validate(engine='auto') — 'auto' is a routing policy, not an "
                "engine; choose the engine first and pass its name or the "
                "Engine instance"
            )
        if engine not in ENGINES:
            raise ValueError(
                f"validate(engine={engine!r}) — not a registered engine's name; "
                f"expected one of {ENGINES}"
            )
        return effective_capabilities(engine)
    if isinstance(engine, Engine):
        return frozenset(engine.effective_capabilities)
    # the runtime check reaches past the static type on purpose: a class, a
    # number, a collection of anything but names is refused, not made a set
    candidate = cast(object, engine)
    if isinstance(candidate, Collection):
        members = cast(Collection[object], candidate)
        if all(isinstance(member, str) for member in members):
            return frozenset(cast(Collection[str], members))
    raise TypeError(
        "validate(engine=…) takes a registered engine's name, an Engine, or "
        f"the offered capability set; got {type(engine).__name__}"
    )


def validate(
    compiled: CompiledProtocol,
    engine: str | Engine | Collection[str] | None = None,
    *,
    env: ResolutionEnv,
    data: bool = True,
) -> CompiledProtocol:
    """Hold a built document to the rules, and return it.

    Three groups of rules, in the order the doors always ran them:

    1. **The checklist over the axis domain, each axis value once** (§5) —
       a step is exactly as valid as the same document written by hand, so
       the checklist runs over the [`representatives`][causalab.protocol.compiled.CompiledProtocol.representatives]: one concrete step per axis value,
       every other axis at its first value, each a step the engine also
       enumerates (a document with no axes validates exactly once) — and
       every ``file_path`` load against the document that names it
       (§2.5/§8). Every representative, not the first failing one: steps are
       independent documents, so distinct violations across them are raised
       together as a [`ValidationErrors`][causalab.protocol.rules.errors.ValidationErrors];
       identical ones (a sweep whose every step breaks the same rule the
       same way) collapse to the one the loader always reported, so a
       single-rule refusal keeps its text. Rules 13 and 30 need to know the
       engine (§2.8, §2.11): they are evaluated exactly when ``engine`` was
       given. Rule 29 reads the operands' widths from the model's static
       config (``env.model_info``), never from weights. A violation only a
       *combination* of axis values makes (two swept writes colliding at one
       pair of positions, a swept model crossed with a swept layer it lacks)
       is no representative's: the engine runs the whole checklist again
       over every enumerated step before it plans
       (``neural/shared/step_rules.py``), so ``run`` refuses it before the
       first forward where ``validate`` said OK.
    2. **The rules that read the resolved tables**, when ``data`` — every
       column, prompt-variable and position reference against the base
       table (rule 20 and rule 4's column half), a metric's ``minimum_count``
       against the rows it can make eligible (rule 4's threshold half), the
       declared row roles (rule 25) and a fit's split disjointness (rule 22)
       — once per distinct table, at every representative
       ([`check_data_columns`][]).
    3. **The engine**, when ``engine`` was given: a shortfall against what
       the campaign requires is refused (rule 13, the routing text), with
       rules 13 and 30 already decided per representative under (1).
       ``engine`` is a registered engine's name, an
       [`Engine`][] instance, or the offered
       capability set itself — never ``"auto"``, which the caller resolves to
       an engine first.

    ``env`` is the resolution environment the rules read against — the same
    one the document was built against; the compiled object carries no
    environment because it is identity and the environment is not.

    Returns ``compiled`` itself: validation carries no state on the object.
    Raises [`ValidationError`][]
    (several as [`ValidationErrors`][causalab.protocol.rules.errors.ValidationErrors]).
    """
    offered = _offered(engine)
    _checklist(compiled.representatives, env, offered)
    if data:
        check_data_columns(compiled, env)
    if offered is not None:
        refuse_shortfall(compiled.capabilities, offered)
    return compiled


def check_engine(
    compiled: CompiledProtocol, engine_capabilities: Collection[str]
) -> None:
    """The engine-aware rules of a compile, for the engine the caller chose —
    the seam the invariant of spec §5 hangs on: **a configuration accepted
    by preflight must either execute or fail with a narrower runtime
    condition that preflight could not know.**

    [`validate`][] handed an engine decides rule 13 (``pytorch_fn`` needs a
    local engine), rule 30 (the fit as authored needs the engine's training
    verbs) and the capability shortfall (rule 13) itself. The run doors —
    [`run_protocol`][] and the workflow runner —
    compile with no engine and call this with the chosen engine's
    ``effective_capabilities`` **before** the engine loads a model. It
    re-enters the checklist's own rule functions
    ([`check_engine_support`][], then
    [`refuse_shortfall`][]) per
    representative, collecting distinct violations as [`validate`][] does;
    it is not a second pipeline. A dry run calls it once, for the named engine,
    and reports instead of raising.

    Raises [`ValidationError`][] (several as
    [`ValidationErrors`][causalab.protocol.rules.errors.ValidationErrors]); returns ``None`` when
    the engine can execute every point as authored.
    """
    offered = frozenset(engine_capabilities)
    refused: list[ValidationError] = []
    for pdoc in compiled.representatives:
        try:
            check_engine_support(pdoc, offered)
        except ValidationError as err:
            refused.append(err)
    raise_distinct(refused)
    refuse_shortfall(compiled.capabilities, offered)


def resolve_positions(
    compiled: CompiledProtocol,
    *,
    env: ResolutionEnv,
    tokenizers: Callable[[str, str], Any] | None = None,
) -> CompiledProtocol:
    """Resolve every position of every representative with the model's
    tokenizer, hold them to the encode-time rules, and return the compiled
    object carrying the result ([`positions`][causalab.protocol.compiled.CompiledProtocol.positions]) — the tokenizer-dependent half of the
    protocol layer's work, run where a model is about to
    load and nowhere else.

    Per representative (one per axis value, as the checklist and the data
    rules run): the roles' rows ([`resolve_roles`][causalab.protocol.positions.roles.resolve_roles]), one frame per role ([`encode_roles`][causalab.protocol.positions.resolve.encode_roles] — the plain frame, or the document's
    ``segments`` frame through the tokenizer's own chat template), every
    address of every read and write on every row ([`resolve_positions`][causalab.protocol.positions.resolve.resolve_positions]), each declared ``alignment``
    against the pair's observed cardinality, and the refusals
    ([`check_positions`][causalab.protocol.positions.resolve.check_positions]: a write on
    a row its address cannot align — ``alignment_missing`` /
    ``alignment_ambiguous`` — and an out-of-bounds index, with the text the
    executor used to raise at its first forward). When the document saves a
    ``location_ledger``, the ledger is built here too (§6). Representatives
    sharing a positions key (a layer sweep) resolve once.

    The tokenizer comes from [`tokenizer_service`][]: ``tokenizers`` —
    [`handoff`][] passes a caller-owned bundle's own tokenizer when the
    engine holds one (spec §9) — else ``env.tokenizers``, else
    [`causalab.io.tokenizer`][] (the reference engine's own loader), so what
    is resolved here is what the engine encodes. Loading one imports torch,
    which is why this is its own verb and not a group of [`validate`][]: the
    pure verbs stay torch-free and call [`validate`][] alone. The metrics'
    answers are the other tokenizer-dependent check before the weights,
    [`resolve_answers`][].
    """
    from causalab.protocol.positions.ledger import wants_ledger
    from causalab.protocol.positions.resolve import (
        StepResolution,
        build_ledger,
        check_positions,
        encode_roles,
        positions_key,
    )
    from causalab.protocol.positions.resolve import (
        resolve_positions as resolve_step_positions,
    )
    from causalab.protocol.lowering import lower_bands
    from causalab.protocol.positions.roles import resolve_roles
    from causalab.protocol.registry import component_shape

    resolve = tokenizer_service(env, tokenizers)
    resolved: dict[str, StepResolution] = dict(compiled.positions or {})
    for pdoc in compiled.representatives:
        key = positions_key(pdoc)
        if key in resolved:
            continue
        tokenizer = resolve(str(pdoc.model.key), str(pdoc.model.revision))
        # the execution form: a band site (§2.4 `layers`) lowered to its
        # per-layer members, as every executor lowers its document — so the
        # ledger's constituent labels (`reads.r[layers=3].pos`) are the ones
        # the executor would have written
        ldoc = lower_bands(pdoc)
        role_rows, role_fields = resolve_roles(ldoc, env)
        frames = encode_roles(tokenizer, ldoc, role_rows, role_fields)
        positions = resolve_step_positions(ldoc, frames, role_rows, role_fields)
        check_positions(ldoc, positions)
        ledger = None
        if wants_ledger(ldoc):
            info = env.model_info(str(ldoc.model.key))

            def has_positions(site_name: str, doc: Document = ldoc) -> bool:
                component = str(doc.sites[site_name].component)
                return component_shape(info, component).has_contract_form

            ledger = build_ledger(
                ldoc, positions, tokenizer, has_positions=has_positions
            )
        resolved[key] = StepResolution(positions=positions, ledger=ledger)
    return dataclasses.replace(compiled, positions=resolved)


def tokenizer_service(
    env: ResolutionEnv,
    tokenizers: Callable[[str, str], Any] | None = None,
    *,
    engine: Engine | None = None,
) -> Callable[[str, str], Any]:
    """The ``(model key, revision) -> tokenizer`` service the checks before
    the weights resolve with, one load per key and revision.

    The first that applies serves: the tokenizer of a caller-owned bundle
    the ``engine`` holds (spec §9), which is what that engine encodes with;
    ``tokenizers``; ``env.tokenizers``; the reference engine's loader
    ([`causalab.io.tokenizer`][]). The loader reads the tokenizer files, never
    the weights: from the Hugging Face cache, or a download of those files
    when the Hub is reachable.

    Raises:
        ProtocolError: ``P4`` when the tokenizer cannot be loaded, naming the
            key and revision: its files are not in the cache and the Hub is
            unreachable or offline, or the repository is gated and no token
            grants access.
    """
    bundle = getattr(engine, "bundle", None)
    if bundle is not None and hasattr(bundle, "tokenizer"):

        def from_bundle(key: str, revision: str) -> Any:
            return bundle.tokenizer

        tokenizers = from_bundle
    resolve = tokenizers or env.tokenizers
    if resolve is None:
        # the default loader, imported here and not by `io/env.py`: the
        # environment module is in every step script's layering closure
        from causalab.io import tokenizer as tokenizer_module

        resolve = tokenizer_module.load_tokenizer
    loaded: dict[tuple[str, str], Any] = {}

    def service(key: str, revision: str) -> Any:
        if (key, revision) not in loaded:
            try:
                loaded[key, revision] = resolve(key, revision)
            except OSError as err:
                # transformers raises OSError both for files that are not
                # cached while offline and for a gated repository without a
                # token; its first line says which
                reason = str(err).strip().splitlines()[0] if str(err).strip() else ""
                raise ProtocolError(
                    "P4",
                    f"the tokenizer of {key}@{revision} could not be loaded "
                    f"({reason or type(err).__name__}). The checks before the "
                    "weights load read the tokenizer files only: have them in "
                    "the Hugging Face cache (with HF_HUB_OFFLINE=1 they must "
                    "already be cached), or, for a gated repository, accept its "
                    "license and set HF_TOKEN",
                ) from err
        return loaded[key, revision]

    return service


@dataclasses.dataclass(frozen=True)
class AnswerCheck:
    """What [`resolve_answers`][] did with each metric that names answers,
    by label in document order. ``resolved``: its answers resolved with the
    tokenizer before the weights. ``when_scored``: a saved metric over a
    continuation read, whose answers the pass left to the score, because
    which rows address a generated step is known only after the decode."""

    resolved: tuple[str, ...]
    when_scored: tuple[str, ...]


def resolve_answers(
    compiled: CompiledProtocol,
    *,
    env: ResolutionEnv,
    tokenizers: Callable[[str, str], Any] | None = None,
) -> AnswerCheck:
    """Resolve every metric's answers with the model's tokenizer, before the
    weights load (§2.10 "Token forms").

    For every representative, each aggregation that names answers (a
    ``save`` entry, an objective term, an eval entry) is resolved through the
    score path's own function
    ([`metric_token_ids`][causalab.protocol.answers.metric_token_ids])
    over the rows its score reduces: every row of the
    base table for an objective term, every held-out row
    (``train.eval.split``) for an eval entry, and for a saved metric the
    rows its read aligns on. A saved metric's row whose read does not align
    is an excluded measurement (§4.1), and the score never tokenizes its
    answer, so neither does this pass. A value the tokenizer cannot score
    is refused here, where the score would have refused it after the model
    loaded. The refusal names every such value: one line per failing field
    of each aggregation, with the aggregation's owner and label, the table,
    the tokenizer, how many rows fail and the first of them. Aggregations
    that name no answer (``kl``, an unrestricted ``js``, ``top_k``,
    ``decode``) load no tokenizer. A saved metric over a continuation read
    is left to the score: which rows address a generated step is known only
    after the decode.

    One case is legal and still likely wrong, and it warns
    (``ProtocolWarning``) rather than refuses: a *glued* answer. A read at
    the last prompt token scores a bare string answer (it starts with a
    letter or digit) after framed text that ends in a letter or digit, and
    both the answer and its space-prefixed form are single tokens with
    different ids. After such text a model usually emits the spaced form.
    A word continuation can be the real answer, so this is not a refusal.
    The text tested is the one the tokenizer receives: under
    ``segments.frame: chat`` the rendered chat, which ends with the
    assistant header, so a bare answer there is quiet. A ``match`` forms
    list is not checked, because it may credit the bare spelling on
    purpose; nor is an id column. ``tokenizers`` is
    [`tokenizer_service`][]'s argument; [`handoff`][] and the workflow runner
    pass the service they resolve positions with.

    Returns:
        The labels of the metrics whose answers resolved, and of those left
        to the score ([`AnswerCheck`][]), so a report can tell them apart.

    Raises:
        ProtocolError: ``P2`` (the score path's code) naming every value the
            tokenizer cannot score; ``P4`` when the tokenizer cannot load.
    """
    resolve = tokenizer_service(env, tokenizers)
    refusals: list[ProtocolError] = []
    glued: list[str] = []
    seen: set[str] = set()
    # label -> None: document order, each label once
    resolved: dict[str, None] = {}
    when_scored: dict[str, None] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    # the rows each read aligns on, when the compiled object carries no
    # positions (a workflow's check before step 1): per frame and address
    alignment = _Alignment()
    for pdoc in compiled.representatives:
        model = f"{pdoc.model.key}@{pdoc.model.revision}"
        for agg in pdoc.aggregations():
            if not names_answers(agg.spec):
                continue
            ref = _scored_ref(pdoc, agg)
            if ref is None:
                continue
            # a sweep's representatives repeat most aggregations unchanged:
            # the frame and the reads' addresses decide which rows align
            key = json.dumps(
                [
                    ref,
                    agg.owner,
                    str(agg.spec.kind),
                    dict(agg.spec.fields),
                    agg.spec.token_form,
                    _frame_key(pdoc),
                    _addresses(pdoc, agg),
                ],
                sort_keys=True,
                default=str,
            )
            if key in seen:
                continue
            seen.add(key)
            tokenizer = resolve(str(pdoc.model.key), str(pdoc.model.revision))
            if ref not in tables:
                tables[ref] = env.datasets.rows(ref)
            rows = tables[ref]
            scored = _scored_rows(
                compiled, pdoc, agg, len(rows), env, tokenizer, alignment
            )
            if scored is None:
                when_scored[agg.label] = None
                continue
            try:
                metric_token_ids(
                    agg.spec, [rows[i] for i in scored], tokenizer, row_numbers=scored
                )
                resolved[agg.label] = None
            except ProtocolError as err:
                # the path names the owner; the text the label, table and
                # tokenizer, then the score path's own words, one line per
                # failing field
                refusals.extend(
                    ProtocolError(
                        err.code,
                        f"{agg.label!r} over table {ref!r}, with the tokenizer "
                        f"of {model}: {line}",
                        path=f"{agg.owner}.aggregation",
                    )
                    for line in err.message.splitlines()
                )
            # a glued table usually splits some names too: the warning says
            # how many rows share the cause of the refusal beside it
            where = f"{agg.owner} ({agg.label!r}) over table {ref!r}"
            glued.extend(_glued_answers(pdoc, agg, rows, where, tokenizer, env))
    for text in glued:
        warnings.warn(text, ProtocolWarning, stacklevel=2)
    if refusals:
        # one refusal as itself; several as one, each line its own code,
        # path and text
        first, *rest = refusals
        raise ProtocolError(
            first.code,
            "\n".join([first.message, *(str(err) for err in rest)]),
            path=first.path,
        )
    return AnswerCheck(resolved=tuple(resolved), when_scored=tuple(when_scored))


def _frame_key(doc: Document) -> str:
    """What decides a representative's token frames: the model (its
    tokenizer), the data section and the ``segments`` frame. A layer sweep
    shares one."""
    return json.dumps(
        [
            str(doc.model.key),
            str(doc.model.revision),
            doc.raw.get("data"),
            repr(doc.segments),
        ],
        sort_keys=True,
        default=str,
    )


def _addresses(doc: Document, agg: BoundAggregation) -> list[Any]:
    """The address (position spec and input role) of each read ``agg``
    reduces: its read, and a ``js`` target."""
    from causalab.protocol.positions.resolve import spec_of

    return [
        None
        if ref is None
        else [repr(spec_of(doc, doc.reads[ref.read].pos)), doc.group_of(ref)[1]]
        for ref in (agg.read, agg.target)
    ]


def _scored_ref(doc: Document, agg: BoundAggregation) -> str | None:
    """The table whose rows ``agg`` scores: the held-out split for an eval
    entry (§2.11), the base role's table otherwise (§2.2: a metric row is a
    base row)."""
    if agg.owner.startswith("train.eval.aggregations."):
        held_out = doc.train.eval if doc.train is not None else None
        split = held_out.get("split") if held_out is not None else None
        return split if isinstance(split, str) else None
    base = doc.data.get("base")
    ref = getattr(base, "dataset", None)
    return ref if isinstance(ref, str) else None


@dataclasses.dataclass
class _Alignment:
    """The alignment [`resolve_answers`][] resolved itself: per frame key the
    roles' rows, per frame key and role the encoded frame, and per address
    the rows it cannot align on."""

    roles: dict[str, Any] = dataclasses.field(default_factory=dict)
    frames: dict[tuple[str, str], Any] = dataclasses.field(default_factory=dict)
    unaligned: dict[str, frozenset[int]] = dataclasses.field(default_factory=dict)


def _scored_rows(
    compiled: CompiledProtocol,
    doc: Document,
    agg: BoundAggregation,
    count: int,
    env: ResolutionEnv,
    tokenizer: Any,
    alignment: _Alignment,
) -> list[int] | None:
    """The rows of ``agg``'s table (``count`` of them) whose answers its
    score resolves, as the score path picks them, or ``None`` when that is
    known only after the decode.

    An objective term and an eval entry reduce a dense read, so they score
    every row (a row their read cannot align on makes the read ragged, which
    they refuse). A saved metric scores the rows its read, and a ``js``
    target, align on (``neural/shared/execution.py``): a row whose address
    matches nothing or several times is an excluded measurement (§2.10
    "Eligibility", §4.1). A read that names answers taps ``lm_head`` (rule
    4), which has a position axis, so every such read gathers at its
    address. A continuation read addresses the steps the decode generated:
    ``None``."""
    from causalab.protocol.positions.encoding import generated_budget
    from causalab.protocol.positions.resolve import spec_of

    every = list(range(count))
    if not agg.owner.startswith("save["):
        return every
    if generated_budget(doc, doc.reads[agg.read.read].pos) is not None:
        return None
    excluded: set[int] = set()
    for ref in (agg.read, agg.target):
        if ref is None:
            continue
        pos = doc.reads[ref.read].pos
        spec = spec_of(doc, pos)
        if spec.generated is not None:
            continue
        _model, role = doc.group_of(ref)
        excluded |= _unaligned(
            compiled, doc, pos, spec, role, env, tokenizer, alignment
        )
    return [i for i in every if i not in excluded]


def _unaligned(
    compiled: CompiledProtocol,
    doc: Document,
    pos: Any,
    spec: PositionSpec,
    role: str,
    env: ResolutionEnv,
    tokenizer: Any,
    alignment: _Alignment,
) -> frozenset[int]:
    """The rows of ``role`` that ``pos`` cannot align on: read off
    ``compiled.positions`` when [`resolve_positions`][] ran, else resolved
    here through the same functions, over the frame it would encode. Each
    frame is encoded once, and each address resolved once, however many
    representatives share them. A plain ``index``, ``span`` or ``all`` spec
    names no value to find in the row, so it aligns on every row and nothing
    is encoded for it."""
    from causalab.protocol.positions.resolve import (
        encode_role,
        positions_key,
        resolve_address,
    )
    from causalab.protocol.positions.roles import resolve_roles

    if not isinstance(spec, SpanSpec) and all(
        anchor is None
        for anchor in (spec.variable, spec.column, spec.scope, spec.relative_to)
    ):
        return frozenset()
    held = (compiled.positions or {}).get(positions_key(doc))
    if held is not None:
        return frozenset(held.positions.address(pos, spec, role).problems)
    frame = _frame_key(doc)
    address = json.dumps([frame, repr(spec), role])
    if address not in alignment.unaligned:
        if frame not in alignment.roles:
            alignment.roles[frame] = resolve_roles(doc, env)
        role_rows, role_fields = alignment.roles[frame]
        if (frame, role) not in alignment.frames:
            alignment.frames[frame, role] = encode_role(
                tokenizer, doc, role_rows[role], role_fields[role]
            )
        resolved = resolve_address(
            spec, alignment.frames[frame, role], role_rows[role], role_fields[role]
        )
        alignment.unaligned[address] = frozenset(resolved.problems)
    return alignment.unaligned[address]


def _at_last_prompt_token(doc: Document, read: str) -> bool:
    """Whether ``read`` taps the last token of the prompt (``index: -1``
    and nothing else), where the model predicts the answer that follows the
    prompt."""
    pos = doc.reads[read].pos
    if isinstance(pos, str):
        pos = doc.positions.get(pos)
    return (
        isinstance(pos, PositionSpec)
        and pos.index == -1
        and pos.scope is None
        and pos.relative_to is None
        and pos.generated is None
    )


def _glued_answers(
    doc: Document,
    agg: BoundAggregation,
    rows: Sequence[Mapping[str, Any]],
    where: str,
    tokenizer: Any,
    env: ResolutionEnv,
) -> list[str]:
    """One warning per answer column of ``agg`` that holds a glued answer
    ([`resolve_answers`][] says when one is), with the count and the first
    row. Such a read aligns on every row, so every row is scored."""
    from causalab.protocol.positions.encoding import select_field
    from causalab.protocol.positions.framing import frame_rows
    from causalab.protocol.positions.roles import input_roles, resolve_roles

    if agg.spec.token_form == "id" or agg.read.model is None:
        return []
    if not _at_last_prompt_token(doc, agg.read.read):
        return []
    fields = {
        field: column
        for field, column in metric_column_fields(agg.spec).items()
        if field != "restrict"
    }
    if not fields:
        return []
    role = doc.input_of(agg.read.model)
    if agg.owner.startswith("train.eval.aggregations."):
        # an eval pass reads every role from the held-out rows
        prompts: Sequence[Mapping[str, Any]] = rows
        field_of_role = input_roles(doc)[role].resolved_field
    else:
        role_rows, role_fields = resolve_roles(doc, env)
        prompts, field_of_role = role_rows[role], role_fields[role]
    # the text the tokenizer receives: under a `segments` section its frame
    # (under `chat`, the rendering, which ends with the assistant header)
    texts: Sequence[Any] = (
        frame_rows(tokenizer, prompts, field_of_role, doc.segments).texts
        if doc.segments is not None
        else [select_field(row, field_of_role) for row in prompts]
    )
    encoded: dict[str, list[int]] = {}

    def ids(text: str) -> list[int]:
        if text not in encoded:
            encoded[text] = list(tokenizer.encode(text, add_special_tokens=False))
        return encoded[text]

    out: list[str] = []
    for field, column in fields.items():
        count, first = 0, None
        for index, (row, prompt) in enumerate(zip(rows, texts)):
            value = row.get(column)
            if not (isinstance(value, str) and isinstance(prompt, str)):
                continue
            if not (prompt[-1:].isalnum() and value[:1].isalnum()):
                continue
            bare, spaced = ids(value), ids(" " + value)
            if len(bare) == 1 and len(spaced) == 1 and bare != spaced:
                count += 1
                if first is None:
                    first = (index, prompt, value, bare[0], spaced[0])
        if first is None:
            continue
        index, prompt, value, bare_id, spaced_id = first
        out.append(
            f"{where}: metric {agg.spec.kind}.{field} column {column!r}: {count} of "
            f"{len(rows)} rows glue a bare answer to a prompt that ends in a letter "
            f"or digit (row {index}: ...{prompt[-24:]!r} + {value!r}). {value!r} is "
            f"token {bare_id} and {' ' + value!r} is token {spaced_id}. After such "
            "a prompt a model usually emits the spaced form, so the metric likely "
            "scores a token the model does not emit there. Put the space in the "
            "string, or ignore this warning where the bare word continuation is "
            "the answer (§2.10 'Token forms')"
        )
    return out


def handoff(compiled: CompiledProtocol, engine: Engine, run: RunContext) -> RunResult:
    """Hand a compiled document to the engine routing chose: [`validate`][]
    against that engine with the data rules — every rule the pure verbs can
    decide, decided before a model loads — then [`resolve_positions`][]
    and [`resolve_answers`][] with the model's tokenizer (the encode-time
    rules and every metric's answers, still before any weights), then
    ``engine.execute(compiled, run)``. Nothing else: the run receipt and the event stream are the
    engine's to write ([`causalab.neural.shared.receipt`][], after it has
    enumerated and signed the steps and before the first forward) exactly
    when ``run.record`` says so, so a workflow step — whose receipt is its
    ``_step.json`` and whose stream is the runner's — executes without them.
    ``engine`` is an [`Engine`][] —
    [`validate`][] reads its ``effective_capabilities`` — not a duck-typed
    stand-in. What comes back is held to the other half of the contract
    ([`check_steps_signed`][]): one signed step
    per selected index.
    """
    validate(compiled, engine, env=run.env, data=True)
    # a caller-owned bundle (spec §9) is what runs: its tokenizer resolves the
    # positions and the answers, and no tokenizer is loaded by the model's
    # key; otherwise one load serves both passes
    tokenizers = tokenizer_service(run.env, engine=engine)
    compiled = resolve_positions(compiled, env=run.env, tokenizers=tokenizers)
    resolve_answers(compiled, env=run.env, tokenizers=tokenizers)
    check_parallel(compiled, engine, env=run.env)
    result = engine.execute(compiled, run)
    check_steps_signed(result, run.indices(point_count(compiled.axes)), engine)
    return result


def check_parallel(
    compiled: CompiledProtocol,
    engine: Engine,
    *,
    env: ResolutionEnv,
    decoding: bool = False,
) -> None:
    """The two document rules of the engine's ``--parallel`` geometry
    (``docs/model_parallelism.md`` §8.3, §8.4), decided here, before any
    weights and on every rank alike: context parallelism refuses a decoding
    document and a hybrid tower at any point of a swept model key (the
    waiver's text is agreed across the world at the rendezvous, so every rank
    reads the same switch); data parallelism over rows refuses a document
    with no fit to split or with fewer ``train.batch.pairs`` than replicas.
    An engine declaring no geometry runs at world 1 and nothing is checked.

    Args:
        compiled: The document the engine is about to run.
        engine: The routed engine; its ``parallel`` is the geometry.
        env: Resolves each swept model key's registry row.
        decoding: The run decodes whatever the document reads, as a
            behavioral step does through ``RunContext.decoding``. A document
            decodes on its own when a read addresses a ``generated`` position.

    Raises:
        ProtocolError: ``P4`` with the refusals, one per line.
    """
    from causalab.protocol.parallel import check_rows, context_refusals
    from causalab.protocol.receipt import decodes, engine_geometry, fit_pairs

    geometry = engine_geometry(engine)
    if geometry.world == 1:
        return
    keys = sorted({str(doc.model.key) for doc in compiled.representatives})
    refusals = context_refusals(
        geometry,
        decoding or decodes(compiled),
        (env.model_info(key) for key in keys),
        os.environ,
    )
    if refusals:
        raise ProtocolError("P4", "\n".join(refusals))
    if geometry.data_mode == "rows":
        refusals = check_rows(geometry, fit_pairs(compiled))
        if refusals:
            raise ProtocolError("P4", "\n".join(refusals))


def compile_protocol(
    source: Path | Mapping[str, Any] | Authored,
    *,
    env: ResolutionEnv,
    base_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    point_cap: int | None = DEFAULT_POINT_CAP,
    engine: str | Engine | Collection[str] | None = None,
    data: bool = False,
) -> CompiledProtocol:
    """[`build`][] then [`validate`][], as one call — the door every
    non-CLI caller takes (a workflow's inner document at load and at run, a
    test's fixture document, a script). Keyword-only over the resolution
    environment; the six-positional ``compile_protocol`` facade of the
    pre-refactor loader is gone. ``engine`` and ``data`` are
    [`validate`][]'s; a path resolves its relative references from its own
    directory unless ``base_dir`` says otherwise."""
    # the offered set goes into `build` too, so a document with a rule-13/30
    # defect *and* a missing dependency reports the rule — the precedence the
    # loader always had (`build`'s docstring, "Which refusal, when there are two")
    offered = _offered(engine)
    compiled = build(
        source,
        base_dir=base_dir
        if base_dir is not None
        else (source.parent if isinstance(source, Path) else None),
        overrides=overrides,
        env=env,
        point_cap=point_cap,
        offered=offered,
    )
    return validate(compiled, offered, env=env, data=data)


def run_protocol(
    document: CompiledProtocol | Path | Mapping[str, Any],
    environment: ResolutionEnv,
    engine: Engine,
    output_directory: Path,
    *,
    points: str | None = None,
    record: bool = False,
    sink: Callable[[Mapping[str, Any]], None] | None = None,
    publisher: Publisher = SOLO,
) -> RunResult:
    """Execute one intervention specification and return what it produced.

    ``document`` is a compiled document, a path, or the document itself as a
    tree; a path or a tree is compiled here against ``environment`` (a path's
    relative references resolve from its own directory). Pass a
    [`CompiledProtocol`][] when the compile
    needs options (``overrides``, a non-default ``point_cap``), or when the
    caller has already validated it and does not want a second parse.

    ``engine`` is the one implementation of the
    [`Engine`][] contract the caller chose (the
    CLI's ``--engine``, through [`causalab.neural.shared.engine_router`][]);
    [`route_engine`][] holds the document to it, or refuses (§8). Supplying
    it is the caller's job precisely so this module imports none.

    ``points`` shards the campaign: ``"START:STOP"`` over the steps in the
    engine's canonical enumeration order (the count is decided from the axes,
    [`point_count`][]), carried to the engine
    as the [`RunContext`][]'s point indices —
    the engine enumerates the steps and reads them by index, and the
    **campaign digest is untouched**, so a shard's artifacts still stamp and
    dedup as members of the whole campaign — which is what lets an external
    scheduler fan shards out and recombine them by digest.

    ``record`` opts into the run's two sidecar files, both off by default: a
    run writes the saved tables and nothing else. Set, the engine writes the
    run receipt, ``protocol.json``, before the first forward pass, once it
    has enumerated and signed the steps (``record=True`` on the run context;
    [`causalab.neural.shared.receipt.write_run_record`][]), and amends it
    after the campaign with what execution observed (the ``fires`` counts,
    the ``scoring`` check, measured row bounds). Beside it the engine appends
    the run's **event stream**, ``events.jsonl`` ([`causalab.io.events`][];
    workflow spec §4.3): ``phase_started`` before it runs, then — once it has
    — one ``progress`` line per point, one ``metric`` line per summarized
    metric, ``result_committed`` naming the files, ``phase_completed`` and
    ``campaign_terminal``. A run that raises leaves a stream without a
    terminal line. ``sink`` is the optional adapter handed each line after
    its local write; its failure becomes a ``warning`` line and changes
    nothing else. The stream is what carries a line to a sink, so a ``sink``
    without ``record`` is refused rather than silently delivering nothing.
    Neither file enters a digest or a stamp, and the ``fires`` counts live
    only in the returned [`RunResult`][]'s
    summaries when no receipt is asked for.

    ``publisher`` is this process's place in a launched world
    (``docs/model_parallelism.md`` §3, §8.3; [`causalab.protocol.publish`][])
    — [`SOLO`][], the default, is world 1. It
    rides on the run context: under data parallelism over points the engine
    runs this replica's contiguous shard of the selected points and only the
    joiner — the publishing rank of replica 0 — writes the receipt, the
    stream and the joined outputs; every other rank's result carries the
    campaign's signed steps and no files. The receipt's
    ``execution.parallel.launcher`` is the publisher's word.

    Rule 25 is checked here rather than only in ``validate --data``: a
    declared row convention that does not describe the resolved batch is
    refused before the engine is checked, so no weights load (§2.8.1). The
    engine is held to the rules that needed to know it by
    [`route_engine`][], likewise before any weights: what the document
    decides is refused at load, and the engine is left only the conditions
    the load could not know (§5). The handoff itself is
    [`handoff`][]: the compiled document is
    validated against the chosen engine with the data rules — rule 20's
    columns, rule 4's threshold and scoring halves, rule 22's fit splits, the
    pass ``causalab validate`` runs — then executed. Those rules run before
    the engine is entered, so a document they refuse writes nothing at all —
    no receipt, no stream (the receipt is the engine's, and a refused
    document never reaches it); the text is ``causalab validate``'s.
    """
    compiled = (
        document
        if isinstance(document, CompiledProtocol)
        else compile_protocol(document, env=environment)
    )
    if sink is not None and not record:
        raise ValueError(
            "run_protocol: a sink receives the lines of events.jsonl, which only "
            "record=True writes; pass record=True with the sink"
        )
    for point in compiled.representatives:
        check_row_roles(point, environment)  # §5 rule 25, before any weights
    chosen = route_engine(compiled, engine)
    n_points = point_count(compiled.axes)
    run = RunContext(
        output_dir=output_directory,
        env=environment,
        points=tuple(parse_points(points, n_points)) if points is not None else None,
        sink=sink,
        record=record,
        publisher=publisher,
    )
    return handoff(compiled, chosen, run)


def route_engine(compiled: CompiledProtocol, engine: Engine | None) -> Engine:
    """Hold the chosen engine to the rules that needed to know it — §5 rules
    13 and 30 and the §8 capability shortfall, through
    [`check_engine`][] — before it loads a
    model, and return it. The one engine-check step both doors share: this
    module's [`run_protocol`][] and the workflow runner's protocol step.

    The name survives the retirement of capability routing: there is
    no choice left to make here — ``--engine`` made it — only the check. The
    order inside ``check_engine`` is what routing used to arrange by hand:
    when the shortfall is a fact the *document* authored (a training field
    rule 30 decides) the refusal names the field; the generated rule-13 text
    naming the missing verbs is the answer only when no rule has a narrower
    one. ``None`` — a workflow run with no engine at all — is refused under
    rule 13 before any weights, naming what the document requires: no engine
    offers nothing, so the text is the generated shortfall every other
    refusal gets ([`refuse_shortfall`][]),
    never a hand-written sentence.
    """
    if engine is None:
        refuse_shortfall(compiled.capabilities, frozenset())
        raise AssertionError(
            "a compiled document requires at least one capability (every read "
            "contributes its component entry); none was derived"
        )
    check_engine(compiled, engine.effective_capabilities)
    return engine
