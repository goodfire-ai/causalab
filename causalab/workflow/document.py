"""Parse and validate workflow documents.

A workflow declares steps and their inputs, outputs, and dependencies. The
loader derives scheduling edges from references, resolves nested workflows,
and computes step identities. Control declarations add qualification requirements
and check whether control sites match their targets.

See ``docs/workflow_protocol.md`` for the format and validation rules."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from causalab.protocol.bundles import (
    entry_key,
    entry_selection,
    select_entry,
    selector_slot,
)
from causalab.protocol.schema.explicit import canonical_model_ref
from causalab.protocol.identity import (
    ResolvedCode,
    canonical_bytes,
    closure_sha256,
    import_closure,
    is_installed_module,
    source_root,
    source_sha256,
)
from causalab.protocol.compiled import Authored
from causalab.protocol.pipeline import compile_protocol, read_document
from causalab.protocol.equivalence import (
    EQUIVALENCE_FIELDS,
    SITE_FIELDS,
    SiteTuple,
    compare,
    coverage,
    explain,
)
from causalab.protocol.equivalence import sharing as coordinate_sharing
from causalab.protocol.rules.errors import (
    ParseError,
    ProtocolError,
    ProtocolWarning,
    ValidationError,
    suggest,
)
from causalab.io.sources import apply_overrides, load_text
from causalab.protocol.registry import ModelInfo
from causalab.io.env import ArtifactStore, ResolutionEnv
from causalab.protocol.schema import (
    FEATURIZER_SLOTS,
    Document,
    FeaturizerSpec,
    IMSpec,
    operand_reads,
    SiteSpec,
    Sweep,
)
from causalab.protocol.lowering import (
    DEFAULT_POINT_CAP,
    coordinate_label,
    short_coords,
)
from causalab.io.tables import TABLE_SUFFIX
from causalab.workflow.reduction import (
    REDUCE_MODULE,
    REDUCTION_INPUT,
    ReductionSpecError,
    parse_reduction,
)
from causalab.workflow.steps import InnerProtocol

__all__ = [
    "COLUMN_DTYPES",
    "CONTROLS_FILE",
    "CONTROL_INPUT",
    "CONTROL_KINDS",
    "CONTROL_RULE",
    "CONTROL_SEAMS",
    "CONTROL_STATUSES",
    "COVERAGE_KINDS",
    "DEFAULT_MIN_DRAWS",
    "DEFAULT_STOP_AFTER_FAILURE_RATE",
    "EQUIVALENCE_FIELDS",
    "EQUIVALENCE_RULE",
    "INHERITED_STATUSES",
    "POST_HOC_CONTROL_KINDS",
    "QUALIFICATION_RULE",
    "REQUIRED_CONTROL_KINDS",
    "WAIVER_REASONS",
    "BehavioralStep",
    "ConditionalStep",
    "DecisionStep",
    "DeferredArtifacts",
    "LoadedWorkflow",
    "OutputDecl",
    "ProtocolStep",
    "Reference",
    "STEP_TYPES",
    "ScriptStep",
    "Step",
    "WorkflowDocument",
    "WorkflowError",
    "WorkflowStep",
    "certifier_subject",
    "is_workflow",
    "load_workflow",
    "parse_workflow",
    "producer_of",
]

STEP_TYPES: tuple[str, ...] = (
    "intervention_protocol",
    "script",
    "behavioral",
    "decision",
    "conditional",
    "workflow",
)

#: Column dtypes a table output may declare. Deliberately narrow: the types
#: that survive a JSON round-trip and a strict re-parse by a consuming step.
COLUMN_DTYPES: tuple[str, ...] = ("int64", "float64", "bool", "string")

#: The two *record* formats: structured data and dense numerics (§2.5).
RECORD_SUFFIXES: tuple[str, ...] = (TABLE_SUFFIX, ".safetensors")

#: Visualization formats. These carry no record — a figure is a rendering of an
#: artifact rather than one itself — so they are legal outputs but may declare
#: no `columns`/`keys`. `png` is preferred over `pdf` unless a document asks for
#: pdf explicitly (``causalab.io.plots.figure_format``).
VISUALIZATION_SUFFIXES: tuple[str, ...] = (".png", ".pdf", ".html")

OUTPUT_SUFFIXES: tuple[str, ...] = RECORD_SUFFIXES + VISUALIZATION_SUFFIXES

#: The control kinds a protocol step may declare itself to be (§2.2). Closed:
#: the spec's ``kind`` table lists exactly these and
#: ``tests/workflow/test_controls.py`` holds the two together. Four, not
#: seven: the no-op collapses into ``self_swap`` (the same document); the
#: full-component and untouched-layer controls are documents any author writes
#: (a featurizer omitted, a site swept) and materializing them is forbidden —
#: ``full_component`` *declares* such a document so its site can be held
#: equivalent to its target's (rule 16) and its measured ceiling recorded, the
#: untouched-layer control is a site swept and needs no kind; and
#: ``carry_stratified`` was dropped as task-specific. ``shuffled_source``
#: is the label control: its document is the target's with one difference —
#: a counterfactual role authoring the data verb ``shuffle: {seed}`` (IM spec
#: §2.2), checked at load against the target's canonical form with ``shuffle``
#: masked, so a declaration names a real permutation of the target's pairing
#: and nothing else.
CONTROL_KINDS: tuple[str, ...] = (
    "self_swap",
    "matched_random",
    "shuffled_source",
    "full_component",
)

#: The kinds whose semantics demand *coverage* of the target's site — every
#: layer, head, expert, stream and coordinate the target's points write, the
#: control's points write too — and which rule 16 therefore holds
#: site-equivalent to their target ([`causalab.protocol.equivalence`][]).
#: Not ``self_swap``: a self-swap certifies per point, and per-point
#: *agreement* by coordinates (§8) is the right semantics there — a control
#: pinned to one layer says exactly what it says at that layer. Not
#: ``shuffled_source``: its document is the target's own, so the sites agree
#: by construction and rule 14 holds the whole canonical form instead.
COVERAGE_KINDS: tuple[str, ...] = ("full_component", "matched_random")

#: The kinds rule 14 holds every fit and every named control target to:
#: each is the ``kind`` of a step declaring ``control.of`` = that step, or is
#: named in that step's ``waive`` — never silently absent.
REQUIRED_CONTROL_KINDS: tuple[str, ...] = ("self_swap", "matched_random")

#: The kinds whose document may legitimately depend on the target by
#: construction — a ``matched_random`` draws its mask from the fit's bundle
#: through ``causalab.analysis.random_mask`` — so an authored path from the
#: control to its target keeps its direction (rule 15, §2.2). Every other kind
#: is certifiable and has no post-hoc direction: its qualification is what the
#: target's whole fanout inherits, so a ``self_swap`` or ``shuffled_source``
#: control authored to run after its target is refused, never scheduled.
POST_HOC_CONTROL_KINDS: tuple[str, ...] = ("matched_random",)

#: Why a control may be waived (§2.2). Small and closed on purpose — seven
#: ``not_applicable`` entries would be silence with extra steps. ``no_fit``
#: waives ``matched_random`` on a target that trains nothing; ``single_role``
#: waives ``shuffled_source`` on a document with no counterfactual role;
#: ``external`` says the control lives elsewhere and **must carry a
#: reference** to it.
WAIVER_REASONS: tuple[str, ...] = ("no_fit", "single_role", "external")

#: A control's status per point (§2.2, §8): certified, not certified, waived
#: on the target, or not (yet) run.
CONTROL_STATUSES: tuple[str, ...] = ("passed", "failed", "waived", "not_run")

#: The words a *dependent* point inherits (§8): ``instrument_invalid`` on a
#: point whose control ``failed``; ``instrument_failure`` is the control
#: point's own word on the event stream (a ``warning`` reason, §4.3). Both are
#: per-point words inside a step record, never a step status (`manifest.py`).
INHERITED_STATUSES: tuple[str, ...] = ("instrument_invalid", "instrument_failure")

#: The seam a replay control's failure rate is measured on (§2.2): in-process
#: re-execution, artifact round-trip, batch order, or a parametrization
#: re-materialization. Recorded beside the rate; nothing on this tree checks a
#: tolerance against it.
CONTROL_SEAMS: tuple[str, ...] = ("A", "B", "C", "R1")

#: How many seeds a ``matched_random`` control draws unless it says otherwise
#: (twenty draws, recorded seeds). Authored lower it is
#: recorded lower — never silently.
DEFAULT_MIN_DRAWS = 20

#: The failure rate above which a certified control is a **failed** step and
#: its dependents are blocked (§8). Zero: the first failure stops, and a
#: campaign that can justify a non-zero bound declares one.
DEFAULT_STOP_AFTER_FAILURE_RATE = 0.0

#: The table a certifying script step writes: one row per control point with
#: its ``status`` and the certification legs (``causalab.analysis.certify_control``).
CONTROLS_FILE = "controls.json"

#: The input name under which the runner hands a certifying script the
#: declaration it certifies (rule 14 refuses a step authoring one).
CONTROL_INPUT = "control"

#: The checklist rule the control layer refuses under (§5).
CONTROL_RULE = 14

#: The checklist rule qualify-once refuses under (§2.2, §5): a control runs
#: before its target's whole fanout (the schedule derives the edge), and a
#: control and its target are one realization — the compiled ``model`` as
#: ``canonical_model`` materializes it (key, revision, dtype, quantization, and
#: the attention backend when authored), equal field for field.
QUALIFICATION_RULE = 15

#: The checklist rule site equivalence refuses under (§5): a control and its
#: target are site-equivalent, or the control declares why not.
EQUIVALENCE_RULE = 16

_STEP_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")

#: Top-level sections in the recommended order (§1) — recommended, not
#: required, exactly as for an intervention specification
#: (intervention_protocol §5 rule 2): a workflow's canonical form walks this
#: tuple, so order changes neither the digest nor the schedule. No `save`:
#: everything a step declares is published where it lands (§0).
SECTION_ORDER: tuple[str, ...] = (
    "version",
    "description",
    "output_dir",
    "steps",
    "measurement",
)

#: The highest checklist rule number (§5). The census guard
#: (``tests/workflow/test_reduction_census.py``) holds the spec's numbered list
#: to exactly this many items. 15 is the qualify-once rule, 16 is
#: site equivalence, 17 is the behavioral step's
#: ([`causalab.workflow.behavioral.BEHAVIORAL_RULE`][]); 18 is the
#: decision / conditional / receipt layer's
#: ([`causalab.workflow.conditional.CONDITIONAL_RULE`][], §2.8); 19 is the
#: declared fan-out's ([`causalab.workflow.fan_out.FAN_OUT_RULE`][], §2.9);
#: 20 is the nested workflow's ([`causalab.workflow.nested.NESTED_RULE`][],
#: §2.10).
MAX_RULE = 20


class WorkflowError(ProtocolError):
    """A workflow document violates checklist rule ``rule``
    (docs/workflow_protocol.md §5); code ``W<rule>``."""

    def __init__(self, rule: int, message: str, *, path: str | None = None) -> None:
        if not 1 <= rule <= MAX_RULE:
            raise AssertionError(f"workflow checklist rule out of range: {rule}")
        self.rule = rule
        super().__init__(f"W{rule}", message, path=path)


# --------------------------------------------------------------------------- #
# object model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Reference:
    """One resolved-at-run-time input: a locator plus an optional selector (§3).

    Exactly one locator is set. ``step``+``file`` names a file in the run tree;
    ``path`` names a file on disk (absolute if it starts with ``/``, otherwise
    relative to the directory of the workflow document that names it — the
    base a ``{"path": …}`` script locator resolves against, so a document and
    the files it points at travel together and resolve the same from a
    checkout, a wheel, or any working directory). ``key`` selects a scalar out
    of a JSON values
    object; ``entry`` selects one tensor of a safetensors bundle. At most one
    selector, and each requires the matching format."""

    step: str | None = None
    file: str | None = None
    path: str | None = None
    key: str | None = None
    entry: Mapping[str, Any] | None = None
    #: which named tensor of a multi-slot bundle, when ``entry`` alone is ambiguous
    slot: str | None = None

    @property
    def target(self) -> str:
        """What this reference names, for an error message."""
        return f"{self.step}/{self.file}" if self.step is not None else str(self.path)

    @property
    def suffix(self) -> str:
        name = self.file if self.step is not None else self.path
        return Path(str(name)).suffix


@dataclasses.dataclass(frozen=True)
class OutputDecl:
    """One declared output: a filename, plus at most one shape promise.

    ``columns`` says "an array of row objects" and maps column name to a
    [`COLUMN_DTYPES`][] entry. ``keys`` says "one object mapping these names
    to values" and maps each name to a **representative value** — not a type,
    because a step-dependent inner document validates against it and a position
    spec has to type-check as a position spec (§2.3)."""

    file: str
    columns: Mapping[str, str] | None = None
    keys: Mapping[str, Any] | None = None

    @property
    def suffix(self) -> str:
        return Path(self.file).suffix


@dataclasses.dataclass(frozen=True)
class ProtocolStep:
    type: str
    document: str
    set: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    max_points: int | None = None
    #: the step's row bounds (§2.2): ``batch_rows`` / ``fit_rows``, each a
    #: positive integer or ``None`` for "unbounded for this step". Execution,
    #: not identity (IM spec §8): absent from the canonical form and the
    #: digest, handed to the engine on the request and recorded in the step's
    #: receipt
    execution: Mapping[str, int | None] = dataclasses.field(default_factory=dict)
    #: the step's declaration that it *is* a control of another step (§2.2):
    #: ``{of, kind, seam?, seeds?, min_draws?}`` in its parsed form — like
    #: ``runtime``, absent when unauthored, so it enters the digest only for a
    #: step that declares it
    control: Mapping[str, Any] | None = None
    #: the controls this step waives, ``{kind: {reason, reference?}}`` (§2.2);
    #: absent when unauthored
    waive: Mapping[str, Mapping[str, str]] | None = None
    #: the failure rate above which this control's certification fails the
    #: step (§8); absent when unauthored, [`DEFAULT_STOP_AFTER_FAILURE_RATE`][]
    #: at run time
    stop_after_failure_rate: float | None = None
    #: the receipt this step's allocation requires (§2.8): ``{step, outcome}``
    #: naming a decision producer and the outcome its ``decision.json`` must
    #: carry; absent when unauthored, so it enters the digest only for a step
    #: that declares it
    requires_receipt: Mapping[str, str] | None = None
    #: the step's declared fan-out (§2.9): ``{"over": {"axis": A} | {"shards":
    #: N}, "join": {"require": "all" | "selected"}}`` in its parsed form;
    #: absent when unauthored, so it enters the digest only for a step that
    #: declares it. The step's name is then the join of its children
    fan_out: Mapping[str, Any] | None = None
    #: set on a **derived child** only (§2.9, §6 — never authored, never
    #: canonical): ``{index, of, over, value | range, points}``, the parent's
    #: compiled point indices this child runs
    shard: Mapping[str, Any] | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class ScriptStep:
    type: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, OutputDecl]
    #: exactly one of these is set — the script locator (§2.3)
    module: str | None = None
    path: str | None = None
    #: sha256 of the script's bytes — in the digest, so ``--resume`` is correct
    script_sha256: str = ""
    #: the script's declared import closure (§4.2): ``{path: sha256}`` of every
    #: module *beside* a ``{"path": …}`` script that it reaches through its
    #: imports, and one hash over the manifest — in the digest beside
    #: ``script_sha256`` when non-empty, so an edit to a sibling helper busts
    #: ``--resume`` the way an edit to the script does. Empty for a
    #: ``{"module": …}`` script into the package (its imports are runtime
    #: identity, the ``tree_digest`` every record carries) or into an installed
    #: module (third-party); a user package's siblings on ``sys.path`` enter.
    closure: Mapping[str, str] = dataclasses.field(default_factory=dict)
    closure_sha256: str = ""
    runtime: Mapping[str, Any] | None = None
    #: the authored reduction contract in its canonical form (§2.6), or
    #: ``None`` — like ``runtime``, absent when unauthored, so it enters the
    #: digest only for a step that declares it
    reduction: Mapping[str, Any] | None = None
    is_deterministic: bool = True
    #: the receipt this step's allocation requires (§2.8); absent when unauthored
    requires_receipt: Mapping[str, str] | None = None
    after: tuple[str, ...] = ()
    description: str | None = None

    @property
    def script(self) -> str:
        """What the document said, for an error message or a manifest."""
        return self.module if self.module is not None else str(self.path)


@dataclasses.dataclass(frozen=True)
class BehavioralStep:
    """A ``behavioral`` step (§2.7): a no-intervention document under the
    ``generated`` frame — ``document``, ``set`` and ``max_points`` exactly as
    a protocol step has them — plus what the document does not carry, all in
    their parsed form and all in the step's canonical entry (§7): the
    ``decoding`` spec, the ``checker`` binding, the ``split`` purpose, the
    ``thresholds``, the ``retain`` bound (absent when unauthored — bounded by
    default at run time) and the typed ``decision``. Parsed and checked by
    [`causalab.workflow.behavioral`][], run by it too."""

    type: str
    document: str
    set: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    max_points: int | None = None
    decoding: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    checker: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    split: str = ""
    thresholds: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    retain: Mapping[str, Any] | None = None
    decision: Mapping[str, str] = dataclasses.field(default_factory=dict)
    #: the receipt this step's allocation requires (§2.8); absent when unauthored
    requires_receipt: Mapping[str, str] | None = None
    #: the step's declared fan-out (§2.9), as on a protocol step; absent when
    #: unauthored
    fan_out: Mapping[str, Any] | None = None
    #: set on a derived child only (§2.9); never authored, never canonical
    shard: Mapping[str, Any] | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class DecisionStep:
    """A ``decision`` step (§2.8): a typed decision record over a script
    step's values object, read through the §3 reference grammar. ``values``
    names the producer's ``.json`` values file; ``rule`` maps each declared
    key to exactly one comparator and a JSON literal; ``decision`` maps pass
    and fail to the decision vocabulary, as on a behavioral step. Parsed and
    checked by [`causalab.workflow.conditional`][], run by it too; every
    field is in the step's canonical entry (§7)."""

    type: str
    values: Reference
    rule: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    decision: Mapping[str, str] = dataclasses.field(default_factory=dict)
    requires_receipt: Mapping[str, str] | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class ConditionalStep:
    """A ``conditional`` step (§2.8): a closed predicate over one field of a
    producer's ``decision.json`` — ``{"decision": {"step"}, "field",
    "eq" | "ne" | "in": literal}`` — and the two disjoint, non-empty sets of
    step names its verdict decides between: ``true`` skips every ``on_false``
    step (and their dependents), ``false`` the ``on_true`` side. ``scope`` is
    from the closed set: ``global`` decides for the run; ``per_target`` and
    ``per_variable`` decide per child of a declared fan-out (§2.9) and expand
    with it at load."""

    type: str
    predicate: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    on_true: tuple[str, ...] = ()
    on_false: tuple[str, ...] = ()
    scope: str = "global"
    requires_receipt: Mapping[str, str] | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


@dataclasses.dataclass(frozen=True)
class WorkflowStep:
    """A ``workflow`` step (§2.10): its ``document`` is another workflow
    document, relative to the workflow file, loaded once at the outer's load
    through the same [`load_workflow`][]; its steps join the run as
    ``<step>/<inner>`` and keep their own identities. ``set`` is the nested
    form — inner step name → that step's own ``set`` map — laid over the named
    inner steps before the inner parse. A container: it publishes no file,
    writes no receipt, is in no schedule and has no status; its canonical
    entry carries the inner document's own digest as ``workflow_digest`` (§7).
    Parsed and mounted by [`causalab.workflow.nested`][]."""

    type: str
    document: str
    set: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    #: the receipt every step of the nested workflow waits on (§2.8, §2.10);
    #: absent when unauthored
    requires_receipt: Mapping[str, str] | None = None
    after: tuple[str, ...] = ()
    description: str | None = None


Step = (
    ProtocolStep
    | ScriptStep
    | BehavioralStep
    | DecisionStep
    | ConditionalStep
    | WorkflowStep
)

#: The step types that compile an inner intervention specification at load
#: (rule 8) and whose outputs are its ``save`` manifest (rule 4).
_DOCUMENT_STEPS = (ProtocolStep, BehavioralStep)


@dataclasses.dataclass(frozen=True)
class WorkflowDocument:
    version: str
    output_dir: str
    steps: Mapping[str, Step]
    description: str | None = None
    measurement: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class LoadedWorkflow:
    """One loaded workflow: the parsed document, the derived schedule, the
    inner protocol loads (or authored-form info for step-dependent ones), the
    canonical form, and the digest."""

    document: WorkflowDocument
    workflow_dir: Path
    order: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    dependencies: Mapping[str, tuple[str, ...]]
    inner: Mapping[str, InnerProtocol]
    inner_digest_kind: Mapping[str, str]  # "campaign" | "authored" | "workflow"
    inner_digests: Mapping[str, str]
    canonical: Mapping[str, Any]
    digest: str
    #: ``{step: digest of its canonical entry}`` — the provenance unit a tensor
    #: a script step writes is stamped with (§7)
    step_digests: Mapping[str, str] = dataclasses.field(default_factory=dict)
    #: absolute `path` references, which load cannot existence-check (rule 4)
    unchecked_paths: tuple[str, ...] = ()
    #: ``{fanned-out step: its children, in child order}`` (§2.9) — the
    #: expansion a declared ``fan_out`` derived at load; the children are in
    #: ``document.steps``, ``order``, ``dependencies`` and ``step_digests``,
    #: and in ``canonical`` never
    children: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    #: ``{workflow step: the inner workflow it loaded}`` (§2.10) — the inner's
    #: steps are in ``document.steps``, ``order``, ``dependencies``, ``inner``
    #: and ``step_digests`` under ``<step>/<inner>``, and in ``canonical``
    #: never: the entry carries the inner's digest as ``workflow_digest``
    nested: Mapping[str, "LoadedWorkflow"] = dataclasses.field(default_factory=dict)
    #: ``{control step: {status, fields, sharing}}`` — the rule-16 verdict of
    #: every coverage-kind control against its target (§2.2, §8): derived at
    #: load, recorded on the control step's record, never canonical
    equivalence: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def nondeterministic(self) -> tuple[str, ...]:
        """Steps that declared themselves not replayable (§7)."""
        return tuple(
            name
            for name, step in self.document.steps.items()
            if isinstance(step, ScriptStep) and not step.is_deterministic
        )


def is_workflow(raw: Mapping[str, Any]) -> bool:
    """A workflow document is distinguished by its ``steps`` section (§1)."""
    return "steps" in raw


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _check_keys(obj: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    for key in obj:
        if key not in allowed:
            raise WorkflowError(
                1, f"unknown key {key!r}{suggest(key, allowed)}", path=path
            )


def _need(obj: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    for field in fields:
        if field not in obj:
            raise WorkflowError(1, f"missing required key {field!r}", path=path)


def _str_field(obj: Mapping[str, Any], field: str, path: str) -> str:
    value = obj[field]
    if not isinstance(value, str) or not value:
        raise WorkflowError(
            1, f"{field!r} is a non-empty string, got {value!r}", path=path
        )
    return value


def _after(obj: Mapping[str, Any], path: str) -> tuple[str, ...]:
    raw = obj.get("after", ())
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise WorkflowError(1, "'after' is a list of step names", path=path)
    return tuple(str(name) for name in raw)


def _bool_field(obj: Mapping[str, Any], field: str, default: bool, path: str) -> bool:
    if field not in obj:
        return default
    value = obj[field]
    if not isinstance(value, bool):
        raise WorkflowError(11, f"{field!r} is a boolean, got {value!r}", path=path)
    return value


def _contained(value: str, rule: int, path: str) -> None:
    """A relative path that cannot escape its directory (rules 2, 6, 7)."""
    if not isinstance(value, str) or not value:
        raise WorkflowError(rule, f"expected a path, got {value!r}", path=path)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise WorkflowError(
            rule,
            f"{value!r} must be relative and stay inside its directory",
            path=path,
        )


def _parse_reference(value: Mapping[str, Any], path: str) -> Reference | None:
    """One ``inputs`` value as a reference, or ``None`` if it is a literal.

    References are recognized **only at the top level** of an inputs entry, so
    a nested object is always a literal and the loader never guesses (§3)."""
    locators = [key for key in ("step", "path") if key in value]
    if not locators:
        return None
    if len(locators) > 1:
        raise WorkflowError(
            1,
            "a reference has one locator: 'step'+'file', or 'path'",
            path=path,
        )
    allowed = ("step", "file", "path", "key", "entry", "slot")
    _check_keys(value, allowed, path)
    if "step" in value:
        _need(value, ("file",), path)
        ref = Reference(
            step=_str_field(value, "step", path),
            file=_str_field(value, "file", path),
        )
        _contained(str(ref.file), 4, path)
    else:
        ref = Reference(path=_str_field(value, "path", path))
    selectors = [key for key in ("key", "entry") if key in value]
    if len(selectors) > 1:
        raise WorkflowError(
            4,
            "a reference carries at most one selector: 'key' (JSON) or "
            "'entry' (safetensors)",
            path=path,
        )
    if "key" in value:
        ref = dataclasses.replace(ref, key=_str_field(value, "key", path))
    if "entry" in value:
        entry = value["entry"]
        if not isinstance(entry, Mapping):
            raise WorkflowError(1, "'entry' maps coordinate names to values", path=path)
        ref = dataclasses.replace(ref, entry=dict(entry))
    if "slot" in value:
        ref = dataclasses.replace(ref, slot=_str_field(value, "slot", path))
    # rule 4: a selector must match its locator's format, decidable from the
    # filename alone — which is exactly what having only two formats buys
    if ref.key is not None and ref.suffix != TABLE_SUFFIX:
        raise WorkflowError(
            4,
            f"'key' reads a {TABLE_SUFFIX} file; {ref.target!r} is "
            f"{ref.suffix or 'extensionless'}",
            path=path,
        )
    if (ref.entry is not None or ref.slot is not None) and ref.suffix != ".safetensors":
        raise WorkflowError(
            4,
            f"'entry'/'slot' reads a .safetensors bundle; {ref.target!r} is "
            f"{ref.suffix or 'extensionless'}",
            path=path,
        )
    return ref


def _parse_script_locator(raw: Any, path: str) -> tuple[str | None, str | None]:
    """``script`` as a locator: ``{"module": …}`` or ``{"path": …}`` (§2.3)."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(
            6,
            '\'script\' is a locator: {"module": "causalab.analysis.fit_pca"} '
            'or {"path": "scripts/probe.py"}',
            path=path,
        )
    _check_keys(raw, ("module", "path"), path)
    present = [key for key in ("module", "path") if key in raw]
    if len(present) != 1:
        raise WorkflowError(
            6,
            "'script' names exactly one of 'module' or 'path'",
            path=path,
        )
    if "module" in raw:
        module = _str_field(raw, "module", path)
        if not all(part.isidentifier() for part in module.split(".")):
            raise WorkflowError(
                6, f"script module {module!r} is not a dotted identifier", path=path
            )
        return module, None
    return None, _str_field(raw, "path", path)


def _parse_output(slot: str, raw: Any, path: str) -> OutputDecl:
    if isinstance(raw, str):
        decl = OutputDecl(file=raw)
    elif isinstance(raw, Mapping):
        _check_keys(raw, ("file", "columns", "keys"), path)
        _need(raw, ("file",), path)
        columns = raw.get("columns")
        keys = raw.get("keys")
        if columns is not None and keys is not None:
            raise WorkflowError(
                7,
                "'columns' and 'keys' are mutually exclusive: the first says "
                "an array of row objects, the second one values object",
                path=path,
            )
        if columns is not None:
            if not isinstance(columns, Mapping) or not columns:
                raise WorkflowError(
                    7, "'columns' maps column names to dtypes", path=path
                )
            for column, dtype in columns.items():
                if dtype not in COLUMN_DTYPES:
                    raise WorkflowError(
                        7,
                        f"column {column!r} has unknown dtype {dtype!r}"
                        f"{suggest(str(dtype), COLUMN_DTYPES)}",
                        path=path,
                    )
        if keys is not None and (not isinstance(keys, Mapping) or not keys):
            raise WorkflowError(
                7,
                "'keys' maps emitted names to a representative value each",
                path=path,
            )
        decl = OutputDecl(
            file=_str_field(raw, "file", path),
            columns=dict(columns) if columns is not None else None,
            keys=dict(keys) if keys is not None else None,
        )
    else:
        raise WorkflowError(
            1, f"output {slot!r} is a filename or an object with 'file'", path=path
        )
    _contained(decl.file, 7, path)
    if decl.suffix not in OUTPUT_SUFFIXES:
        raise WorkflowError(
            7,
            f"output {slot!r} is {decl.file!r} — every output ends in "
            f"{' or '.join(OUTPUT_SUFFIXES)} (§2.5)",
            path=path,
        )
    if decl.suffix != TABLE_SUFFIX and (
        decl.columns is not None or decl.keys is not None
    ):
        raise WorkflowError(
            7,
            f"output {slot!r} declares columns/keys but is not {TABLE_SUFFIX} — "
            "a shape promise only means something for a structured file",
            path=path,
        )
    return decl


def _parse_runtime(raw: Any, path: str) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise WorkflowError(10, "'runtime' is an object", path=path)
    _check_keys(raw, ("isolate", "deps", "env"), path)
    isolate = raw.get("isolate", False)
    if not isinstance(isolate, bool):
        raise WorkflowError(10, "'runtime.isolate' is a boolean", path=path)
    deps = raw.get("deps", ())
    if isinstance(deps, str) or not isinstance(deps, (list, tuple)):
        raise WorkflowError(10, "'runtime.deps' is a list of requirements", path=path)
    env = raw.get("env", ())
    if isinstance(env, str) or not isinstance(env, (list, tuple)):
        raise WorkflowError(
            10, "'runtime.env' is a list of variable NAMES, never values", path=path
        )
    if isolate and not deps:
        raise WorkflowError(
            10,
            "an isolated step declares its 'deps' — otherwise isolation buys "
            "nothing and the subprocess would import a different environment "
            "than the document says",
            path=path,
        )
    out: dict[str, Any] = {"isolate": isolate, "deps": [str(d) for d in deps]}
    if env:
        out["env"] = [str(name) for name in env]
    return out


def _parse_reduction(
    raw: Any, inputs: Mapping[str, Any], path: str, *, module: str | None
) -> Mapping[str, Any] | None:
    """Rule 12: an authored ``reduction`` is complete and in vocabulary (§2.6);
    rule 13: an authored ``estimand_version`` is one the block computes.

    The block is parsed by [`causalab.workflow.reduction.parse_reduction`][]
    — the same parser the built-in script re-runs at execution — and the
    refusal names the field. Column existence is data and is checked at run
    time against the real table, never here. The parsed form is what enters
    the canonical entry, so ``1`` and ``1.0`` and key order cannot make two
    identical declarations digest differently."""
    if raw is None:
        return None
    try:
        parsed = parse_reduction(raw)
    except ReductionSpecError as err:
        where = f"{path}.{err.field}" if err.field else path
        raise WorkflowError(err.rule, str(err), path=where) from err
    if REDUCTION_INPUT in inputs:
        raise WorkflowError(
            12,
            f"the step authors 'reduction' and also declares an input named "
            f"{REDUCTION_INPUT!r} — the runner hands the authored block to the "
            "script under that name, so the two would collide",
            path=path,
        )
    if module == REDUCE_MODULE and parsed.estimator.y is not None and "value" in inputs:
        # both operands are authored, so the contradiction is the parser's to
        # refuse (the script keeps the same refusal as its run-time backstop).
        # Only for the built-in: `value` is its convention, and a user script
        # step's input of that name means what its author says
        value = inputs["value"]
        # a `Reference` object's repr reads badly in a sentence about a column
        # name; a literal — string, number, list, plain mapping (the built-in
        # takes `str()` of it) — is shown as what it is
        spelled = "a reference" if isinstance(value, Reference) else repr(value)
        raise WorkflowError(
            12,
            f"inputs.value ({spelled}) and reduction.estimator.y "
            f"({parsed.estimator.y!r}) both name the column the built-in "
            f"{REDUCE_MODULE} reduces — declare one (a field that governs "
            "nothing may not be declared)",
            path=f"{path}.estimator.y",
        )
    return parsed.canonical()


#: The keys a protocol step's ``execution`` block may carry (§2.2): the
#: reference engine's two row bounds, by the names its constructor and the
#: run receipt use.
EXECUTION_KEYS = ("batch_rows", "fit_rows")


def _execution_block(raw: Any, path: str) -> dict[str, int | None]:
    """A protocol step's ``execution`` block: each of [`EXECUTION_KEYS`][] a
    positive integer, or ``null`` for "unbounded for this step" — a value,
    kept, since it overrides an engine-wide bound. Absent means empty."""
    if raw is None:
        return {}
    where = f"{path}.execution"
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, "'execution' is an object of row bounds", path=where)
    _check_keys(raw, EXECUTION_KEYS, where)
    block: dict[str, int | None] = {}
    for key, value in raw.items():
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise WorkflowError(
                1, f"'execution.{key}' is a positive integer or null", path=where
            )
        block[key] = value
    return block


def _parse_control(raw: Any, path: str) -> dict[str, Any]:
    """Rule 14: a step's ``control`` declaration is well-formed (§2.2) —
    ``kind`` from [`CONTROL_KINDS`][], ``seam`` from [`CONTROL_SEAMS`][],
    ``seeds`` distinct integers, ``min_draws`` a positive integer. What the
    declaration *claims* about another step — the self-swap predicate, the
    matched-random pairing, the site equivalence rule 16 holds a coverage kind
    to — is checked against the compiled inner documents in
    [`load_workflow`][], not here. ``non_equivalence`` (rule 16) names the
    [`EQUIVALENCE_FIELDS`][] the control knowingly differs from its target
    in, with a reason; parsed with its fields sorted so two spellings digest
    identically."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(
            CONTROL_RULE,
            "'control' is an object: {of, kind, seam?, seeds?, min_draws?, "
            "non_equivalence?} (§2.2)",
            path=path,
        )
    _check_keys(
        raw, ("of", "kind", "seam", "seeds", "min_draws", "non_equivalence"), path
    )
    _need(raw, ("of", "kind"), path)
    kind = raw["kind"]
    if kind not in CONTROL_KINDS:
        raise WorkflowError(
            CONTROL_RULE,
            f"unknown control kind {kind!r}{suggest(str(kind), CONTROL_KINDS)}",
            path=f"{path}.kind",
        )
    out: dict[str, Any] = {"of": _str_field(raw, "of", path), "kind": kind}
    if "seam" in raw:
        seam = raw["seam"]
        if seam not in CONTROL_SEAMS:
            raise WorkflowError(
                CONTROL_RULE,
                f"unknown seam {seam!r}{suggest(str(seam), CONTROL_SEAMS)} — a "
                "replay control names the seam its failure rate is measured on",
                path=f"{path}.seam",
            )
        out["seam"] = seam
    if "seeds" in raw:
        seeds = raw["seeds"]
        if isinstance(seeds, str) or not isinstance(seeds, (list, tuple)) or not seeds:
            raise WorkflowError(
                CONTROL_RULE,
                "'seeds' is a non-empty list of integers — the recorded draws "
                "of the control",
                path=f"{path}.seeds",
            )
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise WorkflowError(
                    CONTROL_RULE,
                    f"seed {seed!r} is not an integer — a recorded seed is what "
                    "reproduces a recorded control, so it is not coerced",
                    path=f"{path}.seeds",
                )
        if len(set(seeds)) != len(seeds):
            raise WorkflowError(
                CONTROL_RULE,
                f"'seeds' repeats a value ({list(seeds)}) — every draw of a "
                "control is its own",
                path=f"{path}.seeds",
            )
        out["seeds"] = [int(seed) for seed in seeds]
    if "min_draws" in raw:
        min_draws = raw["min_draws"]
        if (
            isinstance(min_draws, bool)
            or not isinstance(min_draws, int)
            or min_draws < 1
        ):
            raise WorkflowError(
                CONTROL_RULE,
                "'min_draws' is a positive integer",
                path=f"{path}.min_draws",
            )
        out["min_draws"] = min_draws
    if "non_equivalence" in raw:
        out["non_equivalence"] = _parse_non_equivalence(
            raw["non_equivalence"], f"{path}.non_equivalence"
        )
    return out


def _parse_non_equivalence(raw: Any, path: str) -> dict[str, Any]:
    """Rule 16: ``{"fields": [...], "reason": "..."}`` — ``fields`` a non-empty
    list of distinct names from [`EQUIVALENCE_FIELDS`][], ``reason`` a
    non-empty string. A declaration is how a comparison across a known
    difference stays on the record instead of being silently accepted."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(
            EQUIVALENCE_RULE,
            "'non_equivalence' is an object: {fields: [<field>, …], reason: …} "
            "(§2.2) — the fields the control knowingly differs from its target "
            "in, and why the comparison is still wanted",
            path=path,
        )
    _check_keys(raw, ("fields", "reason"), path)
    _need(raw, ("fields", "reason"), path)
    fields = raw["fields"]
    if isinstance(fields, str) or not isinstance(fields, (list, tuple)) or not fields:
        raise WorkflowError(
            EQUIVALENCE_RULE,
            "'non_equivalence.fields' is a non-empty list of equivalence fields",
            path=f"{path}.fields",
        )
    names: list[str] = []
    for field in fields:
        if not isinstance(field, str) or field not in EQUIVALENCE_FIELDS:
            raise WorkflowError(
                EQUIVALENCE_RULE,
                f"unknown equivalence field {field!r}"
                f"{suggest(str(field), EQUIVALENCE_FIELDS)} — the fields are "
                f"{list(EQUIVALENCE_FIELDS)}",
                path=f"{path}.fields",
            )
        if field in names:
            raise WorkflowError(
                EQUIVALENCE_RULE,
                f"'non_equivalence.fields' names {field!r} twice",
                path=f"{path}.fields",
            )
        names.append(field)
    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise WorkflowError(
            EQUIVALENCE_RULE,
            "'non_equivalence.reason' is a non-empty string — a declared "
            "difference says why the comparison is still wanted",
            path=f"{path}.reason",
        )
    return {
        "fields": sorted(names, key=EQUIVALENCE_FIELDS.index),
        "reason": reason,
    }


def _parse_waive(raw: Any, path: str) -> dict[str, dict[str, str]]:
    """Rule 14: a waiver names a kind from [`CONTROL_KINDS`][] and a reason
    from [`WAIVER_REASONS`][] — a bare word, or ``{reason, reference}``;
    ``external`` must carry its ``reference`` and no other reason may. The
    parsed form is the object form, so ``"no_fit"`` and ``{"reason":
    "no_fit"}`` digest identically."""
    if not isinstance(raw, Mapping) or not raw:
        raise WorkflowError(
            CONTROL_RULE, "'waive' maps a control kind to its reason (§2.2)", path=path
        )
    out: dict[str, dict[str, str]] = {}
    for kind, reason in raw.items():
        where = f"{path}.{kind}"
        if kind not in CONTROL_KINDS:
            raise WorkflowError(
                CONTROL_RULE,
                f"unknown control kind {kind!r}{suggest(str(kind), CONTROL_KINDS)}",
                path=where,
            )
        if isinstance(reason, str):
            entry: dict[str, Any] = {"reason": reason}
        elif isinstance(reason, Mapping):
            _check_keys(reason, ("reason", "reference"), where)
            _need(reason, ("reason",), where)
            entry = dict(reason)
        else:
            raise WorkflowError(
                CONTROL_RULE,
                "a waiver is a reason from the closed set, or {reason, reference}",
                path=where,
            )
        word = entry["reason"]
        if not isinstance(word, str) or word not in WAIVER_REASONS:
            raise WorkflowError(
                CONTROL_RULE,
                f"waiver reason {word!r} is not one of {list(WAIVER_REASONS)}"
                f"{suggest(str(word), WAIVER_REASONS)} — a control is waived for a "
                "reason from the closed set, never silently",
                path=where,
            )
        reference = entry.get("reference")
        if word == "external":
            if not isinstance(reference, str) or not reference:
                raise WorkflowError(
                    CONTROL_RULE,
                    "an 'external' waiver carries a 'reference' — the run, artifact "
                    "or document the control lives in",
                    path=where,
                )
            out[str(kind)] = {"reason": word, "reference": reference}
        elif "reference" in entry:
            raise WorkflowError(
                CONTROL_RULE,
                f"'reference' belongs to reason 'external', not {word!r}",
                path=where,
            )
        else:
            out[str(kind)] = {"reason": word}
    return out


def _parse_failure_rate(raw: Any, path: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0 <= raw <= 1:
        raise WorkflowError(
            CONTROL_RULE,
            "'stop_after_failure_rate' is a number in [0, 1] — the fraction of "
            "control points that may fail before the control is a failed step (§8)",
            path=path,
        )
    return float(raw)


def _parse_step(name: str, raw: Any, path: str) -> Step:
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, f"step {name!r} is an object", path=path)
    _need(raw, ("type",), path)
    kind = raw["type"]
    if kind not in STEP_TYPES:
        raise WorkflowError(
            1,
            f"unknown step type {kind!r}{suggest(str(kind), STEP_TYPES)}",
            path=path,
        )
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise WorkflowError(1, "'description' is free text", path=path)
    if kind == "intervention_protocol":
        _check_keys(
            raw,
            (
                "type",
                "document",
                "set",
                "max_points",
                "execution",
                "control",
                "waive",
                "stop_after_failure_rate",
                "requires_receipt",
                "fan_out",
                "after",
                "description",
            ),
            path,
        )
        _need(raw, ("document",), path)
        overrides = raw.get("set", {})
        if not isinstance(overrides, Mapping):
            raise WorkflowError(1, "'set' maps dotted paths to values", path=path)
        max_points = raw.get("max_points")
        if max_points is not None and (
            not isinstance(max_points, int)
            or isinstance(max_points, bool)
            or max_points < 1
        ):
            raise WorkflowError(1, "'max_points' is a positive integer", path=path)
        control = (
            _parse_control(raw["control"], f"{path}.control")
            if "control" in raw
            else None
        )
        if "stop_after_failure_rate" in raw and control is None:
            raise WorkflowError(
                CONTROL_RULE,
                "'stop_after_failure_rate' bounds a control's certification — "
                "only a step declaring 'control' authors it",
                path=f"{path}.stop_after_failure_rate",
            )
        return ProtocolStep(
            type="intervention_protocol",
            document=_str_field(raw, "document", path),
            set=dict(overrides),
            max_points=max_points,
            execution=_execution_block(raw.get("execution"), path),
            control=control,
            waive=_parse_waive(raw["waive"], f"{path}.waive")
            if "waive" in raw
            else None,
            stop_after_failure_rate=(
                _parse_failure_rate(
                    raw["stop_after_failure_rate"], f"{path}.stop_after_failure_rate"
                )
                if "stop_after_failure_rate" in raw
                else None
            ),
            requires_receipt=_receipt(raw, path),
            fan_out=_fan_out(raw, path),
            after=_after(raw, path),
            description=description,
        )
    if kind == "behavioral":
        _check_keys(
            raw,
            (
                "type",
                "document",
                "set",
                "max_points",
                "decoding",
                "checker",
                "split",
                "thresholds",
                "retain",
                "decision",
                "requires_receipt",
                "fan_out",
                "after",
                "description",
            ),
            path,
        )
        _need(raw, ("document",), path)
        overrides = raw.get("set", {})
        if not isinstance(overrides, Mapping):
            raise WorkflowError(1, "'set' maps dotted paths to values", path=path)
        max_points = raw.get("max_points")
        if max_points is not None and (
            not isinstance(max_points, int)
            or isinstance(max_points, bool)
            or max_points < 1
        ):
            raise WorkflowError(1, "'max_points' is a positive integer", path=path)
        # imported here, not at module level: behavioral.py imports this
        # module's WorkflowError and BehavioralStep
        from causalab.workflow.behavioral import parse_behavioral

        step = parse_behavioral(
            raw,
            path,
            document=_str_field(raw, "document", path),
            overrides=overrides,
            max_points=max_points,
            after=_after(raw, path),
            description=description,
        )
        # the receipt is this module's field on every kind (§2.8), and the
        # fan-out this module's on both document kinds (§2.9), so the
        # behavioral parser, unchanged, never sees either
        return dataclasses.replace(
            step, requires_receipt=_receipt(raw, path), fan_out=_fan_out(raw, path)
        )
    if kind == "decision":
        _check_keys(
            raw,
            (
                "type",
                "values",
                "rule",
                "decision",
                "requires_receipt",
                "after",
                "description",
            ),
            path,
        )
        _need(raw, ("values", "rule", "decision"), path)
        # imported here, not at module level: conditional.py imports this
        # module's WorkflowError, Reference and the two step classes
        from causalab.workflow.conditional import parse_decision

        return parse_decision(
            raw,
            path,
            values=_parse_values_reference(raw["values"], f"{path}.values"),
            requires_receipt=_receipt(raw, path),
            after=_after(raw, path),
            description=description,
        )
    if kind == "conditional":
        _check_keys(
            raw,
            (
                "type",
                "predicate",
                "on_true",
                "on_false",
                "scope",
                "requires_receipt",
                "after",
                "description",
            ),
            path,
        )
        from causalab.workflow.conditional import parse_conditional

        return parse_conditional(
            raw,
            path,
            requires_receipt=_receipt(raw, path),
            after=_after(raw, path),
            description=description,
        )
    if kind == "workflow":
        # §2.10: a nested workflow document — `fan_out` is not a key here (left
        # for a later version), so strict keys refuse it (rule 1)
        _check_keys(
            raw,
            ("type", "document", "set", "requires_receipt", "after", "description"),
            path,
        )
        _need(raw, ("document",), path)
        from causalab.workflow.nested import parse_workflow_step

        return parse_workflow_step(
            raw,
            path,
            document=_str_field(raw, "document", path),
            requires_receipt=_receipt(raw, path),
            after=_after(raw, path),
            description=description,
        )

    _check_keys(
        raw,
        (
            "type",
            "script",
            "inputs",
            "outputs",
            "runtime",
            "reduction",
            "is_deterministic",
            "requires_receipt",
            "after",
            "description",
        ),
        path,
    )
    _need(raw, ("script", "inputs", "outputs"), path)
    module, script_path = _parse_script_locator(raw["script"], f"{path}.script")
    inputs = raw["inputs"]
    if not isinstance(inputs, Mapping):
        raise WorkflowError(1, "'inputs' maps names to values", path=path)
    outputs = raw["outputs"]
    if not isinstance(outputs, Mapping) or not outputs:
        raise WorkflowError(
            7, "'outputs' is a non-empty map of slot to file", path=path
        )
    parsed_outputs = {
        slot: _parse_output(slot, decl, f"{path}.outputs.{slot}")
        for slot, decl in outputs.items()
    }
    files = [decl.file for decl in parsed_outputs.values()]
    duplicate = next((f for f in files if files.count(f) > 1), None)
    if duplicate is not None:
        raise WorkflowError(
            7,
            f"two outputs both write {duplicate!r} — one file, one slot",
            path=f"{path}.outputs",
        )
    parsed_inputs = {
        key: (
            _parse_reference(value, f"{path}.inputs.{key}") or value
            if isinstance(value, Mapping)
            else value
        )
        for key, value in inputs.items()
    }
    return ScriptStep(
        type="script",
        module=module,
        path=script_path,
        inputs=parsed_inputs,
        outputs=parsed_outputs,
        runtime=_parse_runtime(raw.get("runtime"), f"{path}.runtime"),
        reduction=_parse_reduction(
            raw.get("reduction"), parsed_inputs, f"{path}.reduction", module=module
        ),
        is_deterministic=_bool_field(raw, "is_deterministic", True, path),
        requires_receipt=_receipt(raw, path),
        after=_after(raw, path),
        description=description,
    )


def _receipt(raw: Mapping[str, Any], path: str) -> dict[str, str] | None:
    """A step's ``requires_receipt`` block (§2.8), parsed, or ``None`` when
    unauthored — never a materialized default (§7)."""
    if "requires_receipt" not in raw:
        return None
    from causalab.workflow.conditional import parse_requires_receipt

    return parse_requires_receipt(raw["requires_receipt"], f"{path}.requires_receipt")


def _fan_out(raw: Mapping[str, Any], path: str) -> dict[str, Any] | None:
    """A document step's ``fan_out`` block (§2.9), parsed, or ``None`` when
    unauthored — never a materialized default (§7)."""
    if "fan_out" not in raw:
        return None
    from causalab.workflow.fan_out import parse_fan_out

    return parse_fan_out(raw["fan_out"], f"{path}.fan_out")


def _child_target(target: str, steps: Mapping[str, Step]) -> str | None:
    """The fanned-out step ``target`` names a child of (``<step>@<i>``), or
    ``None``: a reference names the join, never a child (§2.9, rule 19)."""
    from causalab.workflow.manifest import CHILD_SEPARATOR

    head, separator, _ = target.partition(CHILD_SEPARATOR)
    if not separator:
        return None
    parent = steps.get(head)
    if isinstance(parent, _DOCUMENT_STEPS) and parent.fan_out is not None:
        return head
    return None


def _refuse_child_reference(target: Any, steps: Mapping[str, Step], path: str) -> None:
    from causalab.workflow.fan_out import FAN_OUT_RULE

    parent = _child_target(str(target), steps) if isinstance(target, str) else None
    if parent is not None:
        raise WorkflowError(
            FAN_OUT_RULE,
            f"{str(target)!r} is a child of {parent!r}; a reference names the join "
            f"— read {parent!r}, whose published files are the children's joined "
            "in point order (§2.9)",
            path=path,
        )


def _walk_child_references(node: Any, steps: Mapping[str, Step]) -> str | None:
    """The first ``artifact`` or ``file_path`` string inside an inner document
    that names a child (§2.9), or ``None``."""
    if isinstance(node, Mapping):
        for key in ("artifact", "file_path"):
            value = node.get(key)
            if isinstance(value, str) and _child_target(
                producer_of(value, steps) or value.split("/", 1)[0], steps
            ):
                return value
        for value in node.values():
            found = _walk_child_references(value, steps)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _walk_child_references(item, steps)
            if found is not None:
                return found
    return None


def _parse_values_reference(value: Any, path: str) -> Reference:
    """A decision step's ``values`` (§2.8): a §3 step reference to a ``.json``
    values object, with no selector — the ``rule`` names the keys it reads."""
    from causalab.workflow.conditional import CONDITIONAL_RULE

    if not isinstance(value, Mapping):
        raise WorkflowError(
            CONDITIONAL_RULE,
            '\'values\' is a step reference {"step": S, "file": F} to a values '
            "object (§3)",
            path=path,
        )
    ref = _parse_reference(value, path)
    if ref is None or ref.step is None:
        raise WorkflowError(
            CONDITIONAL_RULE,
            '\'values\' names a step\'s values file ({"step": S, "file": F}) — '
            "a decision is made over what a step of this workflow measured, "
            "never over a path outside the run tree",
            path=path,
        )
    if ref.key is not None or ref.entry is not None or ref.slot is not None:
        raise WorkflowError(
            CONDITIONAL_RULE,
            "'values' carries no selector — the 'rule' names the keys it reads",
            path=path,
        )
    if ref.suffix != TABLE_SUFFIX:
        raise WorkflowError(
            CONDITIONAL_RULE,
            f"'values' reads a {TABLE_SUFFIX} values object; {ref.target!r} is "
            f"{ref.suffix or 'extensionless'}",
            path=path,
        )
    return ref


def parse_workflow(raw: Mapping[str, Any]) -> WorkflowDocument:
    """Parse and structurally validate one workflow document (rules 1-3)."""
    if not isinstance(raw, Mapping):
        raise WorkflowError(1, "a workflow document is an object")
    _check_keys(raw, SECTION_ORDER, "")
    _need(raw, ("version", "output_dir", "steps"), "")

    present = [key for key in raw if key in SECTION_ORDER]
    recommended = [key for key in SECTION_ORDER if key in raw]
    if present != recommended:
        # order is a reading convention, not content — same rule, and the same
        # reasoning, as the intervention specification's (§5 rule 2 over there)
        warnings.warn(
            f"sections are not in the recommended docs/workflow_protocol.md §1 "
            f"order: got {present}, recommended {recommended} — this parses, "
            f"digests and runs identically either way",
            ProtocolWarning,
            stacklevel=3,
        )
    version = _str_field(raw, "version", "")
    if version != "1":
        raise WorkflowError(1, f"unsupported version {version!r} (expected '1')")

    output_dir = _str_field(raw, "output_dir", "")
    if not _SEGMENT.match(output_dir) or output_dir in {".", ".."}:
        raise WorkflowError(
            2,
            f"'output_dir' is one filesystem-safe path segment, got "
            f"{output_dir!r} — the CLI supplies the root it sits under (§1.1)",
        )

    steps_raw = raw["steps"]
    if not isinstance(steps_raw, Mapping) or not steps_raw:
        raise WorkflowError(1, "'steps' is a non-empty object", path="steps")
    steps: dict[str, Step] = {}
    for name, step_raw in steps_raw.items():
        if not _STEP_NAME.match(str(name)):
            raise WorkflowError(
                3,
                f"step name {name!r} is not filesystem-safe ([A-Za-z0-9_-]+)",
                path="steps",
            )
        if name == "workflow.json" or name.startswith("_"):
            raise WorkflowError(
                3,
                f"step name {name!r} is reserved — step directories sit beside "
                "the run manifest",
                path="steps",
            )
        steps[str(name)] = _parse_step(str(name), step_raw, f"steps.{name}")

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise WorkflowError(1, "'description' is free text")
    measurement = None
    if "measurement" in raw:
        from causalab.measurement.spec import (
            MeasurementSpecError,
            parse_measurement,
        )

        try:
            measurement = parse_measurement(raw["measurement"], steps)
        except MeasurementSpecError as exc:
            raise WorkflowError(1, str(exc), path="measurement") from exc
    return WorkflowDocument(
        version=version,
        output_dir=output_dir,
        steps=steps,
        description=description,
        measurement=measurement,
    )


# --------------------------------------------------------------------------- #
# script resolution — the module hashed, never imported; its parents imported,
# so they must be numerics-free (§4.2)
# --------------------------------------------------------------------------- #


def resolve_script(step: ScriptStep, workflow_dir: Path, path: str) -> Path:
    """The file a step's ``script`` locator names — found, never imported (rule 6).

    Two locators, the same shape an ``inputs`` reference uses (§3):

    * ``{"module": "causalab.analysis.fit_pca"}`` — an importable module, found
      with `importlib.util.find_spec`, which resolves a dotted name to a
      file without executing **that module**. That is what lets a shipped
      script live wherever it belongs by subject (``causalab.analysis``,
      ``causalab.io.plots``, ``causalab.workflow.scripts``) instead of in one
      flat namespace with a search order.

      It is not, however, import-*free*, and the stdlib says so: *"If the name
      is for a submodule (contains a dot), the parent package is automatically
      imported."* So the torch-free guarantee (§4.2) rests on a second
      obligation, weaker than "nothing is imported" and enough to buy it:
      **every package that can hold a shipped script must be importable without
      numerics.** ``causalab/io/plots/__init__.py`` is lazy (PEP 562) for
      exactly that reason — a shipped script lives under it, and the eager
      version made ``validate`` of the shipped ``weekdays.json`` pay for the
      plotting stack. Both obligations are checked in
      ``tests/protocol/test_load_is_torch_free.py``: the ``{"path": …}`` case
      for the module never being imported, and a per-package import probe plus
      a real ``validate`` of that workflow for the parents.
    * ``{"path": "scripts/probe.py"}`` — a file beside the workflow document,
      contained, no parent escapes.

    v1 spelled a shipped script ``causalab:<name>``. That needed a registry —
    exactly the thing this layer removes — and it hid *which* code ran behind a
    lookup. A module path says it."""
    import importlib.util

    if step.module is not None:
        try:
            spec = importlib.util.find_spec(step.module)
        except (ImportError, ValueError) as err:
            # a missing PARENT package raises rather than returning None
            raise WorkflowError(
                6, f"script module {step.module!r} not found: {err}", path=path
            ) from err
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            raise WorkflowError(
                6,
                f"script module {step.module!r} does not resolve to a Python file",
                path=path,
            )
        return Path(spec.origin)
    _contained(str(step.path), 6, path)
    target = (workflow_dir / str(step.path)).resolve()
    if not target.is_file():
        raise WorkflowError(6, f"script {step.path!r} not found", path=path)
    return target


@dataclasses.dataclass(frozen=True)
class ScriptIdentity:
    """What rule 6 derives from a script's source: its own hash, and — for a
    script with sibling imports beside it — the manifest and hash of its
    declared import closure (§4.2, §7). Both closure fields are empty for a
    ``{"module": …}`` script into the ``causalab`` package or an installed
    module."""

    script_sha256: str
    closure: Mapping[str, str]
    closure_sha256: str


def check_script(target: Path, step: ScriptStep, path: str) -> ScriptIdentity:
    """Rule 6: the script parses and declares ``main``. Returns its identity —
    its own sha256 and the manifest of its declared import closure.

    Deliberately shallow, and deliberately not an import: ``validate`` and
    ``digest`` must stay runnable without torch, and importing a user script
    would pull in whatever it links against (§4.2). Hashing needs no import,
    which is what lets the hash reach the digest.

    The hash itself is [`causalab.protocol.identity.source_sha256`][], and the
    closure [`causalab.protocol.identity.import_closure`][] — both shared with an
    intervention specification's ``code`` references (§2.8.1) so a module that
    is both a script step and a referenced function's home cannot acquire two
    identities. The closure walk is as static as the hash: every member is
    read and parsed, never imported. It is the *identity* walk
    (``repository=False``): only the module's own root is probed — a
    ``{"path": …}`` script's directory, a ``{"module": …}`` locator's package
    root — and nothing under the ``causalab`` package is admitted, because the
    package's bytes are runtime identity, the ``tree_digest`` every step record
    carries and ``--resume`` compares (§7). So a shipped script declares no
    closure, a locator into an installed module declares none (its imports are
    third-party), and a user package's siblings on ``sys.path`` enter exactly
    as a ``{"path": …}`` script's do.

    Hashing is only half of it: resolving a dotted locator imports the
    target's parent packages, so the guarantee also needs those to be
    numerics-free — see [`resolve_script`][]."""
    source = target.read_bytes()
    try:
        tree = ast.parse(source, filename=str(target))
    except SyntaxError as err:
        raise WorkflowError(
            6, f"script {step.script!r} does not parse: {err}", path=path
        ) from err
    has_main = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
        for node in tree.body
    )
    if not has_main:
        raise WorkflowError(
            6,
            f"script {step.script!r} declares no module-level 'main' — the "
            "runner calls main(inputs, outputs) (§4)",
            path=path,
        )
    # one identity walk for both locator forms (§4.2), the same `_canon_code`
    # runs: a `{"module": …}` locator into an installed module (the stdlib,
    # site-packages) declares no closure — its imports are third-party; any
    # other module is walked from its own root with the `causalab` package
    # excluded, so a package module yields no closure (its imports are the
    # `tree_digest`) while a user package's siblings on `sys.path`, like a
    # `{"path": …}` script's, are covered by nothing else and enter.
    if step.module is not None:
        closure = (
            {}
            if is_installed_module(target)
            else import_closure(
                target,
                root=source_root(
                    ResolvedCode(module=step.module, path=target, attr=())
                ),
                repository=False,
            )
        )
    else:
        closure = import_closure(target, root=target.parent, repository=False)
    return ScriptIdentity(
        script_sha256=source_sha256(target),
        closure=closure,
        closure_sha256=closure_sha256(closure),
    )


# --------------------------------------------------------------------------- #
# load-time artifact store for step-dependent inner documents
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class DeferredArtifacts:
    """Wraps an outer store for workflow-load-time inner validation:
    step-refs answer with declared representatives, and every ``file_path``
    check defers to run time — a representative-substituted document may carry
    representative values in exactly the fields an ArtifactIdentity is checked
    against (the site record), so a load-time identity comparison would be
    against the wrong document."""

    outer: ArtifactStore
    step_names: frozenset[str]
    representatives: Mapping[tuple[str, str], Any]

    def _head(self, ref: str) -> str:
        # the longest step name the path starts with (§2.10): a nested step
        # `tail/best` owns `tail/best/values.json`, not its container `tail`
        return producer_of(ref, self.step_names) or ref.split("/", 1)[0]

    def defers(self, file_path: str) -> bool:
        del file_path
        return True  # all file checks re-run with real values at run time

    def read_value(self, artifact: str, key: str) -> Any:
        if self._head(artifact) in self.step_names:
            try:
                return self.representatives[(self._head(artifact), key)]
            except KeyError as err:
                raise KeyError(
                    f"step {artifact!r} declares no emitted key {key!r} — a "
                    "step whose values another document reads declares them "
                    "in outputs.<slot>.keys (workflow §2.3)"
                ) from err
        return self.outer.read_value(artifact, key)

    def file_digest(self, file_path: str) -> str:
        del file_path
        return "0" * 64  # placeholder; the digest of a deferred doc is discarded

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        del file_path
        return None  # loader skips the check for deferring stores


# --------------------------------------------------------------------------- #
# loading: dependencies, schedule, inner documents, checklist, digest
# --------------------------------------------------------------------------- #


def _walk_step_refs(node: Any, step_names: frozenset[str]) -> set[tuple[str, str]]:
    """Every ``{"artifact": <step…>, "key": …}`` reference into a step."""
    refs: set[tuple[str, str]] = set()
    if isinstance(node, Mapping):
        artifact = node.get("artifact")
        if isinstance(artifact, str) and producer_of(artifact, step_names) is not None:
            key = node.get("key")
            if isinstance(key, str):
                refs.add((artifact, key))
        for value in node.values():
            refs |= _walk_step_refs(value, step_names)
    elif isinstance(node, list):
        for item in node:
            refs |= _walk_step_refs(item, step_names)
    return refs


def _walk_run_tree_paths(node: Any, step_names: frozenset[str]) -> set[str]:
    """Every ``file_path`` the inner document LOADS from a step's run tree.

    The IM spec has exactly three file_path *load* sites — featurizer specs,
    params entries, and a featurizer's ``init`` (§2.5, §2.6): the basis or
    theta a fit **starts** from is loaded exactly as a fitted bundle is, so a
    fit initialised from a PCA basis another step produces depends on that
    step (harvest → ``fit_pca`` → fit is one workflow, not two). An inner
    document's ``save`` section also carries ``file_path`` keys, but those are
    outputs, never loads — walking them would fabricate dependency edges (and
    even self-cycles) out of a step's own products."""
    paths: set[str] = set()
    method = node.get("method") if isinstance(node, Mapping) else None
    if not isinstance(method, Mapping):
        return paths

    def note(value: Any) -> None:
        if isinstance(value, str) and producer_of(value, step_names) is not None:
            paths.add(value)

    for section in ("featurizers", "params"):
        table = method.get(section)
        if not isinstance(table, Mapping):
            continue
        for entry in table.values():
            if not isinstance(entry, Mapping):
                continue
            note(entry.get("file_path"))
            init = entry.get("init")
            if isinstance(init, Mapping):
                note(init.get("file_path"))
    return paths


def producer_of(path: str, step_names: Iterable[str]) -> str | None:
    """The step a run-tree path belongs to: the **longest** name in
    ``step_names`` that is ``path`` itself or a prefix of it at a ``/``
    (§2.10). A nested workflow's steps are named ``<step>/<inner>``, so the
    head of ``tail/best/values.json`` is the container ``tail`` and its
    producer is ``tail/best`` — every site that once split a path at its
    first ``/`` asks here instead. ``None`` when no step owns the path."""
    best: str | None = None
    for name in step_names:
        if path == name or path.startswith(name + "/"):
            if best is None or len(name) > len(best):
                best = name
    return best


def _authored_digest(raw: Mapping[str, Any]) -> str:
    """The digest of a deferred step's document as authored — the tree before
    the run supplies what an earlier step produces. The header's ``title`` and
    ``description`` are dropped first, as the canonical form drops them (IM
    spec §7): renaming a step's document or rewording its intent moves no
    step identity, deferred or not."""
    tree = dict(raw)
    header = tree.get("header")
    if isinstance(header, Mapping):
        tree["header"] = {
            key: value
            for key, value in header.items()
            if key not in ("title", "description")
        }
    return hashlib.sha256(canonical_bytes(tree)).hexdigest()


def load_workflow(
    source: Path | Mapping[str, Any],
    env: ResolutionEnv,
    *,
    workflow_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    _including: tuple[Path, ...] = (),
    _step_set: Mapping[str, Mapping[str, Any]] | None = None,
) -> LoadedWorkflow:
    """Load one workflow document through the full pipeline (§5).

    ``_including`` and ``_step_set`` are the recursion's (§2.10): the resolved
    paths of every document being loaded above this one — a document already
    among them is refused as including itself — and the ``set`` an outer
    ``workflow`` step lays over this document's steps before they are parsed.
    [`causalab.workflow.nested.load_nested`][] passes both; a caller does
    not."""
    if isinstance(source, Path):
        raw: dict[str, Any] = dict(load_text(source))
        workflow_dir = source.parent if workflow_dir is None else workflow_dir
    else:
        raw = dict(source)
        workflow_dir = Path(".") if workflow_dir is None else workflow_dir
    if overrides:
        raw = apply_overrides(raw, overrides)
    if _step_set:
        from causalab.workflow.nested import lay_over_set

        raw = lay_over_set(raw, _step_set)
    document = parse_workflow(raw)
    steps = document.steps
    step_names = frozenset(steps)

    # ---- rule 6: scripts resolve, parse, declare main, and hash ------------ #
    identities: dict[str, ScriptIdentity] = {}
    for name, step in steps.items():
        if isinstance(step, ScriptStep):
            target = resolve_script(step, workflow_dir, f"steps.{name}.script")
            identities[name] = check_script(target, step, f"steps.{name}.script")
    steps = {
        name: (
            dataclasses.replace(
                step,
                script_sha256=identities[name].script_sha256,
                closure=identities[name].closure,
                closure_sha256=identities[name].closure_sha256,
            )
            if isinstance(step, ScriptStep)
            else step
        )
        for name, step in steps.items()
    }
    document = dataclasses.replace(document, steps=steps)

    # ---- rule 4 (references) + derived dependency edges (§3) --------------- #
    #: per step, the compiler's read prefix (IM spec §9): the authored document
    #: with `set` applied. The tree is walked for step references below, and
    #: then goes back into the compiler as the authored document — one read,
    #: run in two halves, never a second implementation.
    composed: dict[str, Authored] = {}
    inner_dirs: dict[str, Path] = {}
    deps: dict[str, set[str]] = {name: set() for name in steps}
    #: the DATA half of `deps`: the edges a values reference, a script input
    #: reference or a run-tree load adds — never an `after` entry, which orders
    #: and reads nothing. Rule 15 reads it: only a route of data hops makes a
    #: `matched_random` control post-hoc (it draws from its target's bundle).
    data_deps: dict[str, set[str]] = {name: set() for name in steps}
    step_refs: dict[str, set[tuple[str, str]]] = {}
    run_tree_loads: dict[str, set[str]] = {name: set() for name in steps}
    unchecked_paths: list[str] = []

    # ---- rule 20: nested workflows (§2.10) --------------------------------- #
    # A `workflow` step's document is loaded here — the whole of this function,
    # recursively — before the loop below derives edges: `dependency_edges`,
    # `_walk_step_refs`, `outputs_of` and `_check_declares_key` all index
    # `steps` by name, so an inner step must be in the table, under
    # `<step>/<inner>`, before an outer step can name it. The inner steps keep
    # their identities and their compiled documents; the inner document's own
    # digest is what the outer entry carries as `workflow_digest` (§7).
    # Nothing mounted here enters `canonical`.
    nested: dict[str, LoadedWorkflow] = {}
    mounted: dict[str, str] = {}
    inner: dict[str, InnerProtocol] = {}
    inner_digests: dict[str, str] = {}
    inner_digest_kind: dict[str, str] = {}
    mounted_digests: dict[str, str] = {}
    children: dict[str, tuple[str, ...]] = {}
    if any(isinstance(step, WorkflowStep) for step in steps.values()):
        from causalab.workflow.nested import load_nested, mount

        chain: tuple[Path, ...] = tuple(_including)
        if isinstance(source, Path) and source.resolve() not in chain:
            chain = (*chain, source.resolve())
        flattened: dict[str, Step] = dict(steps)
        for name, step in steps.items():
            if not isinstance(step, WorkflowStep):
                continue
            nested[name] = load_nested(name, step, workflow_dir, env, chain)
            inner_digests[name] = nested[name].digest
            inner_digest_kind[name] = "workflow"
            mount(
                name,
                nested[name],
                steps=flattened,
                deps=deps,
                inner=inner,
                inner_digests=inner_digests,
                inner_digest_kind=inner_digest_kind,
                step_digests=mounted_digests,
                children=children,
                unchecked_paths=unchecked_paths,
                mounted=mounted,
            )
        steps = flattened
        step_names = frozenset(steps)

    for name, step in steps.items():
        if name in mounted:
            continue  # its edges came with it, and its own load checked it (§2.10)
        # §2.9 (rule 19): a child `<step>@<i>` is addressed through its join —
        # refused before rule 4 would call it unknown, naming the parent
        for other in step.after:
            _refuse_child_reference(other, steps, f"steps.{name}.after")
        if step.requires_receipt is not None:
            _refuse_child_reference(
                step.requires_receipt.get("step"),
                steps,
                f"steps.{name}.requires_receipt.step",
            )
        if isinstance(step, DecisionStep):
            _refuse_child_reference(
                step.values.step, steps, f"steps.{name}.values.step"
            )
        if isinstance(step, ConditionalStep):
            _refuse_child_reference(
                step.predicate.get("decision", {}).get("step"),
                steps,
                f"steps.{name}.predicate.decision.step",
            )
            for side in ("on_true", "on_false"):
                for gated in getattr(step, side):
                    _refuse_child_reference(gated, steps, f"steps.{name}.{side}")
        if isinstance(step, ScriptStep):
            for slot, value in step.inputs.items():
                if isinstance(value, Reference) and value.step is not None:
                    _refuse_child_reference(
                        value.step, steps, f"steps.{name}.inputs.{slot}"
                    )
        for other in step.after:
            if other not in steps:
                raise WorkflowError(
                    4, f"'after' names unknown step {other!r}", path=f"steps.{name}"
                )
            deps[name].add(other)

        # §2.8: the edges a decision, a conditional or a receipt derive — a
        # receipt-bearing step after its receipt's producer, a decision after
        # the step it measures, a conditional after its predicate's producer,
        # and every step a conditional gates after the conditional. Schedule
        # only, never canonical (§7). The two new kinds carry no `document`,
        # so they leave the loop here, before the inner document is read.
        if step.requires_receipt is not None or isinstance(
            step, (DecisionStep, ConditionalStep)
        ):
            from causalab.workflow.conditional import dependency_edges

            for dependent, upstream in dependency_edges(name, step, steps):
                deps[dependent].add(upstream)
        if isinstance(step, (DecisionStep, ConditionalStep, WorkflowStep)):
            # a `workflow` step's own edges are its `after` and its receipt
            # (§2.10); its steps' edges came in with the mount
            continue

        if isinstance(step, ScriptStep):
            for slot, value in step.inputs.items():
                if not isinstance(value, Reference):
                    continue
                where = f"steps.{name}.inputs.{slot}"
                if value.step is not None:
                    if value.step not in steps:
                        raise WorkflowError(
                            4,
                            f"input {slot!r} names unknown step {value.step!r}"
                            f"{suggest(value.step, sorted(steps))}",
                            path=where,
                        )
                    deps[name].add(value.step)
                    data_deps[name].add(value.step)
                    continue
                target = str(value.path)
                if target.startswith("/"):
                    # rule 4: an absolute path is NOT existence-checked —
                    # validation and execution routinely run on different
                    # hosts, so checking here would fail a path that is
                    # perfectly good where the run happens
                    unchecked_paths.append(f"{name}.{slot}: {target}")
                    continue
                # rule 4: a relative path resolves against the document's own
                # directory — the same base as a script's `path` locator, and
                # the only one an installed package can offer (a wheel has no
                # repo root; site-packages is not one)
                if not (workflow_dir / target).is_file():
                    raise WorkflowError(
                        4,
                        f"input {slot!r} names {target!r}, which does not exist "
                        f"beside the workflow document (under {str(workflow_dir)!r}; "
                        "an absolute path would defer to run time, a relative "
                        "one must be here now)",
                        path=where,
                    )
            continue

        doc_path = (workflow_dir / step.document).resolve()
        if not doc_path.is_file():
            raise WorkflowError(
                4, f"document {step.document!r} not found", path=f"steps.{name}"
            )
        try:
            # read first — the step's `set` lands on the authored tree by
            # section-rooted path (IM spec §1), before anything resolves
            composed[name] = read_document(doc_path, doc_path.parent, step.set)
        except ParseError as err:
            raise WorkflowError(
                8,
                f"'set' override failed on {step.document!r}: {err}",
                path=f"steps.{name}",
            ) from err
        except ProtocolError as err:
            raise WorkflowError(
                8,
                f"document {step.document!r} does not load: {err}",
                path=f"steps.{name}",
            ) from err
        inner_raw = composed[name].raw
        inner_dirs[name] = doc_path.parent
        child_ref = _walk_child_references(inner_raw, steps)
        if child_ref is not None:
            _refuse_child_reference(
                producer_of(child_ref, steps) or child_ref.split("/", 1)[0],
                steps,
                f"steps.{name}",
            )
        refs = _walk_step_refs(inner_raw, step_names)
        step_refs[name] = refs
        run_tree_loads[name] = _walk_run_tree_paths(inner_raw, step_names)
        for artifact, key in refs:
            producer = producer_of(artifact, step_names)
            assert producer is not None  # `_walk_step_refs` admitted it
            deps[name].add(producer)
            data_deps[name].add(producer)
            # rule 4: a values reference must name a step that declares the
            # key. With `select` a script, the declaration is the contract —
            # and it is what the representative substitution below reads.
            _check_declares_key(steps[producer], producer, key, path=f"steps.{name}")
        for run_path in run_tree_loads[name]:
            producer = producer_of(run_path, step_names)
            assert producer is not None  # `_walk_run_tree_paths` admitted it
            deps[name].add(producer)
            data_deps[name].add(producer)

    # ---- rule 15: a control runs before its target --------------------------- #
    # Every `control.of` adds an implicit edge target -> (the control's
    # certifier when one exists, else the control), so the target's whole
    # rank/seed fanout runs after the qualification and inherits its statuses
    # by coordinates (§2.2, §8) without an authored `after`. An authored path
    # the other way — the control (or its certifier) depending on its target
    # through `after` or a reference — splits by kind: for a kind in
    # `POST_HOC_CONTROL_KINDS` (`matched_random` only: the `random_mask` chain
    # draws from the fit's bundle) the control is post-hoc by construction and
    # keeps its direction — when the route is one of DATA hops (`data_deps`:
    # a values reference, a script input reference or a run-tree load); an
    # `after` entry orders and reads nothing, so a route that needs one to
    # reach the target is refused naming it — the skip would otherwise drop
    # the derived edge on a bare ordering; for every other kind (`self_swap`,
    # `shuffled_source`) there is no post-hoc direction — the qualification is
    # what the target's fanout inherits, and letting the authored edge stand
    # would silently drop it (`inherit` finds no ancestor, the target's record
    # gets no `controls` block) — so it is refused naming the route. The edge
    # lives in the schedule only, never in the canonical form (§7), so a
    # workflow that declares a control has the digest it had.
    authored = {name: set(edges) for name, edges in deps.items()}
    for name, step in steps.items():
        if not isinstance(step, ProtocolStep) or step.control is None:
            continue
        of = str(step.control["of"])
        if of not in steps or of == name or not isinstance(steps[of], ProtocolStep):
            continue  # rule 14 refuses each of these, after the schedule
        head = _certifier_or_none(steps, name) or name
        if _reaches(authored, head, of):
            # `_parse_control` (rule 14) already held `kind` to CONTROL_KINDS,
            # so the read needs no re-validation here (as in `_check_controls`)
            kind = str(step.control["kind"])
            post_hoc = kind in POST_HOC_CONTROL_KINDS
            if post_hoc and _reaches(data_deps, head, of):
                continue  # post-hoc: the control (or its certifier) reads its target
            route = _path(authored, head, of)
            via = " -> ".join(repr(node) for node in route)
            last_hop = _hop(steps, data_deps, route[-2], of)
            if post_hoc:
                raise WorkflowError(
                    QUALIFICATION_RULE,
                    f"control {name!r} (kind {kind!r}) is authored to run after "
                    f"its target {of!r} — {via}, the last hop {last_hop}; ordering "
                    f"alone does not make a control post-hoc; a post-hoc {kind!r} "
                    f"reads its target's bundle — remove the 'after' entry or draw "
                    f"from {of!r} (§2.2, rule 15).",
                    path=f"steps.{name}.control",
                )
            raise WorkflowError(
                QUALIFICATION_RULE,
                f"control {name!r} (kind {kind!r}) is authored to run after its "
                f"target {of!r} — {via}, the last hop {last_hop}; a {kind!r} "
                f"control has no post-hoc direction: its qualification is what "
                f"the whole fanout of {of!r} inherits, so the schedule derives the "
                f"edge target -> control and an authored dependency on the "
                f"target silently drops the qualification. Remove the 'after' "
                f"entry or the reference into {of!r} (§2.2, rule 15).",
                path=f"steps.{name}.control",
            )
        if _reaches(deps, head, of):
            raise WorkflowError(
                QUALIFICATION_RULE,
                f"control {name!r} runs before its target {of!r}, but {of!r} "
                f"already runs before {name!r} through another control's edge — "
                "two steps cannot each qualify the other (§2.2)",
                path=f"steps.{name}.control",
            )
        deps[of].add(head)

    # ---- rule 20: the containers leave the edge map (§2.10, §6) ----------- #
    # an edge to a `workflow` step becomes edges to each of its steps; the
    # container's own edges are inherited by its inner roots; the schedule is
    # over work alone (a container is never attempted and has no status)
    container_edges: dict[str, set[str]] = {}
    if nested:
        from causalab.workflow.nested import flatten_edges

        container_edges = flatten_edges(steps, deps)

    # ---- rule 5: acyclicity + the schedule --------------------------------- #
    order, levels = _schedule(deps)

    # ---- representatives, then the inner loads (rule 4 tail, rule 8) ------- #
    representatives: dict[tuple[str, str], Any] = {}
    for name, step in steps.items():
        if not isinstance(step, ScriptStep):
            continue
        for decl in step.outputs.values():
            for key, value in (decl.keys or {}).items():
                representatives[(name, key)] = value

    # `inner`, `inner_digests` and `inner_digest_kind` were bound above, with
    # every mounted step's entries already in them (§2.10)
    for name in order:
        step = steps[name]
        if name in mounted or not isinstance(step, _DOCUMENT_STEPS):
            continue
        deferred = bool(step_refs[name]) or bool(run_tree_loads[name])
        load_env = env
        if deferred:
            load_env = ResolutionEnv(
                datasets=env.datasets,
                artifacts=DeferredArtifacts(
                    outer=env.artifacts,
                    step_names=step_names,
                    representatives=representatives,
                ),
                model_info=env.model_info,
            )
        try:
            compiled = compile_protocol(
                composed[name],
                env=load_env,
                base_dir=inner_dirs.get(name),
                # `set` is already in the composition; no engine at load
                point_cap=step.max_points
                if step.max_points is not None
                else DEFAULT_POINT_CAP,
            )
        except ProtocolError as err:
            raise WorkflowError(
                8,
                f"document {step.document!r} does not load: {err}",
                path=f"steps.{name}",
            ) from err
        # the steps, enumerated and signed against the load-time environment
        # (the one spelling of the engine's sweep in this layer)
        inner[name] = InnerProtocol.enumerate(compiled, load_env)
        if deferred:
            inner_digests[name] = _authored_digest(composed[name].raw)
            inner_digest_kind[name] = "authored"
        else:
            inner_digests[name] = compiled.digests.document
            inner_digest_kind[name] = "campaign"
        if isinstance(step, BehavioralStep):
            # rule 17 (§2.7): the document decodes, its base ref's fragment is
            # the step's split, and the checker binds to the task's ScoringSpec
            # by digest — before any model exists
            from causalab.workflow.behavioral import check_behavioral

            check_behavioral(name, step, compiled, env)

    # ---- rule 19 + the expansion (§2.9): a declared fan-out's children ----- #
    # The width is a pure function of the compiled document (9·8), so the
    # children exist here, after the inner loads and before every check that
    # walks the step table — rule 4's `outputs_of`, rule 14, rule 18 — and the
    # schedule is recomputed over the expanded graph, so rule 5 holds over the
    # children too. The children share the parent's compiled document, its
    # digest and its composition; they enter `order`, `dependencies`,
    # `step_digests` and `document.steps`, and the canonical form never (§6).
    authored = document
    # `children` was bound above, with the mounted fan-outs' (§2.10); the
    # expansion below is over the outer's own steps — a mounted step's
    # children came in with it, under their flattened names
    own = {name: step for name, step in steps.items() if name not in mounted}
    if any(
        isinstance(step, _DOCUMENT_STEPS) and step.fan_out is not None
        for step in own.values()
    ):
        from causalab.workflow.conditional import dependency_edges
        from causalab.workflow.fan_out import (
            check_fan_out,
            expand,
            expand_conditional,
        )

        check_fan_out(steps, inner)
        expanded: dict[str, Step] = dict(steps)
        for name, step in own.items():
            if not isinstance(step, _DOCUMENT_STEPS) or step.fan_out is None:
                continue
            kids = expand(name, step, inner[name])
            children[name] = tuple(child for child, _ in kids)
            for child, child_step in kids:
                expanded[child] = child_step
                deps[child] = set(deps[name])  # a child after the parent's deps
                inner[child] = inner[name]
                inner_digests[child] = inner_digests[name]
                inner_digest_kind[child] = inner_digest_kind[name]
                composed[child] = composed[name]
                step_refs[child] = step_refs[name]
                run_tree_loads[child] = run_tree_loads[name]
            deps[name] |= set(children[name])  # the join after its children
        for name, step in own.items():
            if not isinstance(step, ConditionalStep) or step.scope == "global":
                continue
            producer = str(step.predicate["decision"]["step"])
            kids = expand_conditional(name, step, len(children[producer]))
            children[name] = tuple(child for child, _ in kids)
            for child, child_step in kids:
                expanded[child] = child_step
                deps[child] = set(deps[name]) - {producer}
                for dependent, upstream in dependency_edges(
                    child, child_step, expanded
                ):
                    deps[dependent].add(upstream)
            deps[name] |= set(children[name])
        steps = expanded
        step_names = frozenset(steps)
        order, levels = _schedule(deps)

    # ---- rule 4: every `file` names an output its producer really writes --- #
    def outputs_of(name: str, path: str | None = None) -> set[str]:
        step = steps[name]
        if isinstance(step, WorkflowStep):
            # §2.10 (rule 20): a nested workflow publishes nothing of its own;
            # its steps do, under `<step>/<inner>`
            from causalab.workflow.nested import NESTED_RULE

            raise WorkflowError(
                NESTED_RULE,
                f"{name!r} is a workflow step and publishes no file; name one of "
                f"its steps, '{name}/<step>' (§2.10)",
                path=path,
            )
        if isinstance(step, BehavioralStep):
            from causalab.workflow.behavioral import BEHAVIORAL_FILES

            saved = {entry.file_path for entry in inner[name].compiled.document.save}
            return saved | set(BEHAVIORAL_FILES)
        if isinstance(step, ProtocolStep):
            return {entry.file_path for entry in inner[name].compiled.document.save}
        if isinstance(step, DecisionStep):
            from causalab.workflow.behavioral import DECISION_FILE

            return {DECISION_FILE}
        if isinstance(step, ConditionalStep):
            # a conditional publishes its record and nothing a §3 reference
            # could read: a reference to one is refused below as "writes no"
            return set()
        return {decl.file for decl in step.outputs.values()}

    for name, step in steps.items():
        if not isinstance(step, ScriptStep):
            continue
        for slot, value in step.inputs.items():
            if not isinstance(value, Reference) or value.step is None:
                continue
            produced = outputs_of(value.step, f"steps.{name}.inputs.{slot}")
            if value.file not in produced:
                raise WorkflowError(
                    4,
                    f"input {slot!r} reads {value.target}, but "
                    f"{value.step!r} writes no {value.file!r} "
                    f"(has {sorted(produced)})",
                    path=f"steps.{name}.inputs.{slot}",
                )
            if value.key is not None:
                _check_declares_key(
                    steps[value.step],
                    value.step,
                    value.key,
                    path=f"steps.{name}.inputs.{slot}",
                    file=value.file,
                )
            if value.entry is not None or value.slot is not None:
                _check_script_entry(
                    name, slot, value, steps=steps, inner=inner, outputs_of=outputs_of
                )

    # ---- rule 4 for a decision's `values` (§2.8): the file is one its ----- #
    # producer writes, and every key the rule reads is one it declares
    for name, step in steps.items():
        if not isinstance(step, DecisionStep):
            continue
        ref = step.values
        produced = outputs_of(str(ref.step), f"steps.{name}.values.step")
        if ref.file not in produced:
            raise WorkflowError(
                4,
                f"'values' reads {ref.target}, but {ref.step!r} writes no "
                f"{ref.file!r} (has {sorted(produced)})",
                path=f"steps.{name}.values",
            )
        for key in step.rule:
            _check_declares_key(
                steps[str(ref.step)],
                str(ref.step),
                key,
                path=f"steps.{name}.rule.{key}",
                file=ref.file,
            )

    # ---- rule 9: run-tree loads inside intervention specifications ------- #
    for name, paths in run_tree_loads.items():
        for run_path in sorted(paths):
            producer = producer_of(run_path, step_names) or run_path.partition("/")[0]
            rest = run_path[len(producer) + 1 :]
            produced = outputs_of(producer, f"steps.{name}")
            if rest not in produced:
                raise WorkflowError(
                    4,
                    f"{name!r} loads {run_path!r}, but {producer!r} writes no "
                    f"{rest!r} (has {sorted(produced)}) — a run-tree file_path "
                    "must name a file the step actually saves",
                    path=f"steps.{name}",
                )
            if isinstance(steps[producer], _DOCUMENT_STEPS):
                _check_entry_selection(
                    consumer=inner[name],
                    producer=inner[producer],
                    run_path=run_path,
                    rest=rest,
                    step=name,
                )

    # ---- rule 20: what crosses the nesting boundary, and what does not ----- #
    if nested:
        from causalab.workflow.nested import check_nested

        check_nested(steps, nested, mounted)

    # ---- rule 14: controls declared or waived, and true to their kind ------ #
    # over the outer's own steps: a nested workflow's controls layer is its
    # own — and empty in this version — so the coverage clause examines no
    # fit inside a nested workflow; `check_nested` (rule 20, above) already
    # refused a nested fit under an engaged outer, so none loads unheld
    # (§2.10). The seed-provenance lookup still needs the flattened names:
    # a featurizer's `file_path` may name a nested step's file, and `<step>`
    # there is `tail/draw` (§2.10)
    equivalence = _check_controls(
        {n: s for n, s in steps.items() if n not in mounted},
        inner,
        env.model_info,
        steps,
    )

    # ---- rule 18: decisions, conditionals and receipts typed and bound ----- #
    if any(
        isinstance(step, (DecisionStep, ConditionalStep))
        or step.requires_receipt is not None
        for step in steps.values()
    ):
        from causalab.workflow.conditional import check_conditional
        from causalab.workflow.nested import expand_skips

        # a global conditional's side may name a `workflow` step (§2.10), and
        # `flatten_edges` took every container out of `deps` — so the rule-18
        # disjointness walk sees each side as the members it would skip
        sided = {
            n: dataclasses.replace(
                s,
                on_true=tuple(expand_skips(steps, s.on_true)),
                on_false=tuple(expand_skips(steps, s.on_false)),
            )
            if isinstance(s, ConditionalStep) and s.scope == "global"
            else s
            for n, s in steps.items()
        }
        check_conditional(sided, deps)

    # ---- canonical form + digest (§7) ------------------------------------- #
    # over the authored steps: a fan-out's children are derived and never
    # canonical (§2.9, §6) — the parent's entry carries `fan_out`
    canonical = _canonicalize(authored, inner_digests)
    digest = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
    # a script step's identity: the digest of its own canonical entry, which
    # is a pure function of script hash + inputs + outputs + runtime (+
    # reduction when authored). It is what `--resume` compares (the
    # `_step.json` identity) — the analogue of a compiled intervention's
    # digest.
    # A behavioral step's identity is its canonical entry too: the decoding
    # (seed included), checker, split, thresholds and decision are in it, so a
    # changed seed or split is a changed identity (§2.7, §7).
    # A decision's and a conditional's identity is its canonical entry too:
    # the rule, the predicate and the gated sides are what the step *is* (§2.8).
    step_digests = {
        name: hashlib.sha256(canonical_bytes(canonical["steps"][name])).hexdigest()
        for name, step in authored.steps.items()
        if isinstance(
            step,
            (ScriptStep, BehavioralStep, DecisionStep, ConditionalStep, WorkflowStep),
        )
        or (isinstance(step, ProtocolStep) and step.fan_out is not None)
    }
    # a nested workflow's steps keep their own identities (§2.10): the inner
    # load's `step_digests`, under their flattened names — which cannot be
    # authored names: `/` is outside rule 3's alphabet, so no key is overwritten
    for flat, mounted_digest in mounted_digests.items():
        step_digests[flat] = mounted_digest
    # a fan-out's children (§2.9): a child's identity is its parent's entry
    # digest plus its shard — a changed width or a changed parent is a child
    # that runs again, and `--resume` never reuses `fit@0` from a run with
    # another selection; a fanned-out protocol parent has an entry digest too
    # (its `fan_out` is in it), where an unfanned one keeps its inner digest
    for parent, kids in children.items():
        if parent in mounted:
            continue  # a mounted fan-out's children came in with their identities
        for index, child in enumerate(kids):
            child_step = steps[child]
            facts: dict[str, Any] = {
                "type": child_step.type,
                "parent": step_digests[parent],
                "document_digest": inner_digests.get(parent),
            }
            shard = getattr(child_step, "shard", None)
            facts["shard"] = dict(shard) if shard is not None else {"index": index}
            step_digests[child] = hashlib.sha256(canonical_bytes(facts)).hexdigest()

    # the loaded table is the flattened one (§2.10): the outer's steps, a
    # fan-out's children (§2.9) and every nested workflow's steps under
    # `<step>/<inner>`; `canonical` above was over the authored steps alone
    document = dataclasses.replace(document, steps=steps)
    return LoadedWorkflow(
        document=document,
        workflow_dir=workflow_dir,
        order=tuple(order),
        levels=levels,
        dependencies={
            # a container's edges are its own outer ones, for the record
            name: tuple(sorted(deps[name] if name in deps else container_edges[name]))
            for name in steps
        },
        inner=inner,
        inner_digest_kind=inner_digest_kind,
        inner_digests=inner_digests,
        canonical=canonical,
        digest=digest,
        step_digests=step_digests,
        unchecked_paths=tuple(sorted(unchecked_paths)),
        children=children,
        nested=nested,
        equivalence=equivalence,
    )


def _schedule(
    deps: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Rule 5 and the derived schedule (§6): a topological order over
    ``deps`` (``{step: the steps it depends on}``), refusing a cycle with its
    trail, and the levels of independent steps ``explain`` reports. Run once
    over the authored graph and again over the expanded one when a fan-out
    added children (§2.9), so a cycle through a child is refused with the
    child's name in the trail."""
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(node: str, trail: tuple[str, ...]) -> None:
        mark = state.get(node)
        if mark == 1:
            return
        if mark == 0:
            cycle = " -> ".join((*trail[trail.index(node) :], node))
            raise WorkflowError(5, f"the step graph has a cycle: {cycle}")
        state[node] = 0
        for dep in sorted(deps[node]):
            visit(dep, trail + (node,))
        state[node] = 1
        order.append(node)

    for name in deps:
        visit(name, ())

    depth: dict[str, int] = {}
    for name in order:
        depth[name] = 1 + max((depth[d] for d in deps[name]), default=-1)
    n_levels = 1 + max(depth.values(), default=0)
    levels = tuple(
        tuple(n for n in order if depth[n] == level) for level in range(n_levels)
    )
    return tuple(order), levels


def _check_declares_key(
    producer: Step,
    producer_name: str,
    key: str,
    *,
    path: str,
    file: str | None = None,
) -> None:
    """Rule 4: a ``key`` reference names a step that declares it.

    v1 could only ask this of a `select` step, whose `emit` table the loader
    read. v2 asks it of *any* step, because outputs are declared — which is why
    the check is stronger than the rule it replaces."""
    if not isinstance(producer, ScriptStep):
        raise WorkflowError(
            4,
            f"values reference reads a key of {producer_name!r}, which is a "
            f"{producer.type} step — only a script step declares a values "
            "object (outputs.<slot>.keys); an intervention specification's "
            "outputs are tables and tensors, and a decision record is read by "
            "a conditional's predicate, never by key (§2.8)",
            path=path,
        )
    declared: dict[str, Any] = {}
    for decl in producer.outputs.values():
        if file is not None and decl.file != file:
            continue
        declared.update(decl.keys or {})
    if key not in declared:
        where = f"{producer_name}/{file}" if file else producer_name
        raise WorkflowError(
            4,
            f"{where} declares no emitted key {key!r} "
            f"({sorted(declared) or 'no keys declared'}) — a step whose values "
            "another step reads declares them in outputs.<slot>.keys (§2.3)"
            f"{suggest(key, sorted(declared))}",
            path=path,
        )


# --------------------------------------------------------------------------- #
# controls: declared or waived, and true to their kind (rule 14)
# --------------------------------------------------------------------------- #


def certifier_subject(steps: Mapping[str, Step], name: str) -> str | None:
    """The control step a script step certifies, or ``None`` when ``name``
    certifies nothing (§2.2, §8).

    A certifying step is recognized by its shape, not its module: it declares
    an output whose file is [`CONTROLS_FILE`][] and reads, through its
    inputs, exactly one step that declares ``control``. Half the shape is
    refused (rule 14): a ``controls.json`` that reads no control, or two, has
    no subject to attach its statuses to. The runner hands such a step the
    control's declaration under [`CONTROL_INPUT`][], so a step authoring an
    input of that name is refused the way rule 12 refuses ``reduction``."""
    step = steps[name]
    if not isinstance(step, ScriptStep):
        return None
    if not any(decl.file == CONTROLS_FILE for decl in step.outputs.values()):
        return None
    subjects = sorted(
        {
            str(value.step)
            for value in step.inputs.values()
            if isinstance(value, Reference)
            and value.step is not None
            and isinstance(steps.get(value.step), ProtocolStep)
            and steps[value.step].control is not None  # type: ignore[union-attr]
        }
    )
    if len(subjects) != 1:
        raise WorkflowError(
            CONTROL_RULE,
            f"step {name!r} writes {CONTROLS_FILE!r}, so it certifies a control — "
            "its inputs must read exactly one step declaring 'control' "
            f"(it reads {subjects or 'none'})",
            path=f"steps.{name}",
        )
    if CONTROL_INPUT in step.inputs:
        raise WorkflowError(
            CONTROL_RULE,
            f"step {name!r} certifies a control and also declares an input named "
            f"{CONTROL_INPUT!r} — the runner hands the control's declaration to "
            "the script under that name, so the two would collide",
            path=f"steps.{name}.inputs.{CONTROL_INPUT}",
        )
    return subjects[0]


def _certifier_or_none(steps: Mapping[str, Step], control: str) -> str | None:
    """The script step certifying ``control`` for the schedule's implicit
    edge (rule 15), or ``None``. A half-shaped certifier (rule 14) is not an
    answer here — `_check_controls` refuses it after the schedule."""
    for name in steps:
        try:
            if certifier_subject(steps, name) == control:
                return name
        except WorkflowError:
            continue
    return None


def _reaches(graph: Mapping[str, set[str]], start: str, goal: str) -> bool:
    """Whether ``start`` depends on ``goal``, transitively, in ``graph``
    (``{step: the steps it depends on}``)."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        node = pending.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph.get(node, ()))
    return False


def _path(graph: Mapping[str, set[str]], start: str, goal: str) -> list[str]:
    """One dependency path from ``start`` to ``goal`` in ``graph`` (``{step:
    the steps it depends on}``), ``[start, ..., goal]``, for naming an authored
    route in a refusal — the caller has established with `_reaches` that
    one exists. Edges are walked in sorted order, so the route is stable."""
    parents: dict[str, str | None] = {start: None}
    pending = [start]
    while pending:
        node = pending.pop(0)
        if node == goal:
            route = [goal]
            parent = parents[goal]
            while parent is not None:
                route.append(parent)
                parent = parents[parent]
            return route[::-1]
        for dep in sorted(graph.get(node, ())):
            if dep not in parents:
                parents[dep] = node
                pending.append(dep)
    raise AssertionError(f"no path from {start!r} to {goal!r}")


def _hop(
    steps: Mapping[str, Step],
    data_deps: Mapping[str, set[str]],
    tail: str,
    head: str,
) -> str:
    """How the authored edge ``tail -> head`` was written, for naming a
    route's last hop in a refusal: ``an 'after' entry``, ``a reference`` (a
    values reference, a script input reference or a run-tree load — the edges
    ``data_deps`` holds), or both when the step authors the two."""
    after = head in steps[tail].after
    reference = head in data_deps.get(tail, set())
    if after and reference:
        return "both an 'after' entry and a reference"
    return "an 'after' entry" if after else "a reference"


def _check_controls(
    steps: Mapping[str, Step],
    inner: Mapping[str, InnerProtocol],
    model_info: Callable[[str], ModelInfo],
    flattened: Mapping[str, Step],
) -> dict[str, dict[str, Any]]:
    """Rule 14 (§2.2, §5): every control declaration names a protocol step
    other than itself and is true to its kind — a ``self_swap`` document really
    holds a self-swap model and a script step certifies it; a
    ``matched_random`` pairs the fit's featurizer kind, rank and site, records
    at least ``min_draws`` distinct seeds and those are the draws the
    document makes; a ``shuffled_source`` document is the target's with a
    counterfactual role's ``shuffle`` as its one difference; a
    ``full_component`` writes the whole component, through no featurizer and
    no ``dims`` — every waiver applies to its kind; and, in a workflow that
    engages the layer at all, every fit and every named target has each of
    [`REQUIRED_CONTROL_KINDS`][] declared by some step or waived.

    Rule 16 (§2.2, §5) rides on the same pass: every [`COVERAGE_KINDS`][]
    control is site-equivalent to its target over the expanded points of both
    — [`causalab.protocol.equivalence.compare`][] — or declares the differing
    fields in ``non_equivalence``. The verdicts are returned, one per such
    control, for the record (§8).

    The last clause is gated on *engagement* — some step authoring ``control``
    or ``waive`` — on purpose: the shipped fit campaigns predate the layer and
    declare nothing, and refusing them would fail legitimate work (fail-closed
    discipline). A workflow that declares one control commits to the
    requirement for every fit it runs in its own steps — a nested workflow's
    fits are held by that document's own declarations, none in this version,
    so rule 20 refuses a nested fit under an engaged outer before this pass
    runs (§2.10).

    ``steps`` is the table the checks run over — the outer's own (§2.10);
    ``flattened`` is the whole loaded table, nested steps included, which the
    seed-provenance lookup reads: a featurizer's ``file_path`` is a run-tree
    path, and its producer may be a step inside a nested workflow."""
    protocol = {
        name: step for name, step in steps.items() if isinstance(step, ProtocolStep)
    }
    certifiers: dict[str, str] = {}
    for name in steps:
        subject = certifier_subject(steps, name)
        if subject is None:
            continue
        if subject in certifiers:
            raise WorkflowError(
                CONTROL_RULE,
                f"control {subject!r} is certified by both {certifiers[subject]!r} "
                f"and {name!r} — one control, one certification",
                path=f"steps.{name}",
            )
        certifiers[subject] = name

    declared: dict[str, dict[str, list[str]]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for name, step in protocol.items():
        if step.control is None:
            continue
        of, kind = str(step.control["of"]), str(step.control["kind"])
        where = f"steps.{name}.control"
        if of not in steps:
            raise WorkflowError(
                CONTROL_RULE,
                f"control 'of' names unknown step {of!r}{suggest(of, sorted(steps))}",
                path=where,
            )
        if of == name:
            raise WorkflowError(
                CONTROL_RULE,
                f"step {name!r} declares itself its own control",
                path=where,
            )
        if not isinstance(steps[of], ProtocolStep):
            raise WorkflowError(
                CONTROL_RULE,
                f"a control is of an intervention_protocol step; {of!r} is a "
                "script step",
                path=where,
            )
        declared.setdefault(of, {}).setdefault(kind, []).append(name)
        if kind == "self_swap":
            _check_self_swap(name, step, inner[name], certifiers)
            continue
        if kind == "shuffled_source":
            _check_shuffled_source(name, step, inner[name], of, inner[of])
            continue  # the target's own document: not a coverage kind (§2.2)
        pairs: list[tuple[str, FeaturizerSpec]] = []
        if kind == "matched_random":
            pairs = _check_matched_random(
                name, step, inner[name], of, inner[of], model_info
            )
        elif kind == "full_component":
            _check_full_component(name, step, inner[name], of)
        else:
            # unreachable from an authored document — `_parse_control` holds
            # `kind` to the closed vocabulary — but a kind added to the tuple
            # without its load-time check is refused here, never waved through
            raise WorkflowError(
                CONTROL_RULE,
                f"control kind {kind!r} has no load-time check — the kinds are "
                f"{list(CONTROL_KINDS)}",
                path=where,
            )
        if kind in COVERAGE_KINDS:
            verdicts[name] = _check_equivalence(
                name, step, inner[name], of, inner[of], model_info
            )
        seeds = [int(seed) for seed in (step.control or {}).get("seeds", [])]
        for cname, cspec in pairs:
            _check_seed_provenance(name, cname, cspec, seeds, flattened)

    engaged = any(
        step.control is not None or step.waive is not None for step in protocol.values()
    )
    for name, step in protocol.items():
        is_fit = inner[name].compiled.document.train is not None
        waived = step.waive or {}
        for kind, waiver in waived.items():
            if kind in declared.get(name, {}):
                raise WorkflowError(
                    CONTROL_RULE,
                    f"step {name!r} waives {kind!r} and step "
                    f"{declared[name][kind][0]!r} declares it — a control is "
                    "declared or waived, never both",
                    path=f"steps.{name}.waive.{kind}",
                )
            _check_waiver_applies(
                name, kind, waiver, inner[name].compiled.document, is_fit
            )
        if not engaged or not (is_fit or name in declared):
            continue
        for kind in REQUIRED_CONTROL_KINDS:
            if kind in declared.get(name, {}) or kind in waived:
                continue
            what = (
                "declares a fit"
                if is_fit
                else "is the target of control "
                + ", ".join(
                    repr(c) for controls in declared[name].values() for c in controls
                )
            )
            raise WorkflowError(
                CONTROL_RULE,
                f"step {name!r} {what}; control {kind!r} is neither declared by a "
                "step nor waived",
                path=f"steps.{name}",
            )

    # rule 15's second clause, after every rule-14 refusal: a pair that is
    # true to its kind and declared-or-waived is then held to one realization
    for name, step in protocol.items():
        if step.control is not None:
            of = str(step.control["of"])
            _check_realization(name, inner[name], of, inner[of])
    return verdicts


def _info_for(
    loaded: InnerProtocol, model_info: Callable[[str], ModelInfo]
) -> ModelInfo | None:
    """The registry entry a compiled document was sized against — the compile
    already looked it up, so a miss here means a swept model key, and the
    predicate compares what it can without one."""
    key = loaded.compiled.document.model.key
    if not isinstance(key, str):
        return None
    try:
        return model_info(key)
    except ValidationError:
        return None


def _check_full_component(
    name: str, step: ProtocolStep, loaded: InnerProtocol, of: str
) -> None:
    """A ``full_component`` control writes the **whole** component: every
    write its intervened models list goes through no featurizer (or only the
    identity) and names no ``dims`` — what makes its score the measured
    ceiling at that site. Its site is held to the
    target's by rule 16, with the featurizer (and so the coordinates) allowed
    to differ: that difference is the kind."""
    where = f"steps.{name}.control"
    for point in loaded.point_documents:
        for im_name, im in point.intervened_models.items():
            names = im.writes if isinstance(im.writes, tuple) else ()
            for wname in names:
                write = point.writes.get(wname)
                if write is None:
                    continue
                chain = write.featurizer
                stages = (
                    chain if isinstance(chain, tuple) else (chain,) if chain else ()
                )
                for fname in stages:
                    spec = point.featurizers.get(str(fname))
                    kind = spec.kind if spec is not None else "identity"
                    if kind != "identity":
                        raise WorkflowError(
                            CONTROL_RULE,
                            f"control {name!r} declares kind 'full_component' of "
                            f"{of!r}, but intervened_models.{im_name}'s write "
                            f"{wname!r} goes through featurizer {fname!r} "
                            f"({kind!r}) — a full-component control swaps the "
                            "whole component; drop the featurizer",
                            path=where,
                        )
                if write.dims is not None:
                    raise WorkflowError(
                        CONTROL_RULE,
                        f"control {name!r} declares kind 'full_component' of "
                        f"{of!r}, but intervened_models.{im_name}'s write "
                        f"{wname!r} selects dims {write.dims!r} — a "
                        "full-component control covers every coordinate; drop "
                        "'dims'",
                        path=where,
                    )


#: What a ``full_component`` control may differ from its target in without a
#: declaration: the featurizer is the kind (identity against the fit's), and
#: ``dims`` index the featurized value, so they follow it.
_KIND_ALLOWS: Mapping[str, frozenset[str]] = {
    "full_component": frozenset({"featurizer", "dims"}),
    "matched_random": frozenset(),
}


def _check_equivalence(
    name: str,
    step: ProtocolStep,
    loaded: InnerProtocol,
    of: str,
    target: InnerProtocol,
    model_info: Callable[[str], ModelInfo],
) -> dict[str, Any]:
    """Rule 16 (§2.2, §5): the control's writes and its target's, over the
    expanded points of both, are site-equivalent — every field of
    [`EQUIVALENCE_FIELDS`][] agrees, and the two share no coordinate system
    — or every differing field is named in the control's ``non_equivalence``.
    Refused naming the first undeclared field, with what separates the two
    and what that means for the comparison; a declared field the pair does
    not differ in is refused too, since a declaration records a real
    difference. Returns the record block ``{status, fields, sharing}``."""
    control = dict(step.control or {})
    kind = str(control.get("kind"))
    ctl = coverage(loaded.point_documents, _info_for(loaded, model_info), owner=name)
    tgt = coverage(target.point_documents, _info_for(target, model_info), owner=of)
    fields = compare(ctl, tgt)
    allowed = _KIND_ALLOWS.get(kind, frozenset())
    declaration = control.get("non_equivalence") or {}
    declared = [str(f) for f in declaration.get("fields", [])]
    for field in fields:
        if field in declared or field in allowed:
            continue
        raise WorkflowError(
            EQUIVALENCE_RULE,
            f"control {name!r} and its target {of!r} differ in {field}: "
            f"{explain(field, ctl, tgt)}. Declare `non_equivalence: {{fields: "
            f"[{field}], reason: …}}` to compare them anyway",
            path=f"steps.{name}.control",
        )
    for field in declared:
        if field not in fields:
            raise WorkflowError(
                EQUIVALENCE_RULE,
                f"control {name!r} declares a non-equivalence in {field!r} with "
                f"its target {of!r}, but the two are equivalent there — a "
                "declaration records a real difference; drop the field",
                path=f"steps.{name}.control.non_equivalence.fields",
            )
    undeclared = [f for f in fields if f not in allowed]
    return {
        "status": "declared" if undeclared else "equivalent",
        "fields": list(fields),
        "sharing": coordinate_sharing(ctl, tgt),
    }


def _writes_through(doc: Document, featurizer: str) -> list[str]:
    """The writes whose featurizer chain includes ``featurizer``."""
    out: list[str] = []
    for wname, write in doc.writes.items():
        chain = write.featurizer
        names = chain if isinstance(chain, tuple) else (chain,)
        if featurizer in names:
            out.append(wname)
    return out


def _check_realization(
    name: str, control: InnerProtocol, of: str, target: InnerProtocol
) -> None:
    """Rule 15's second clause (§2.2, §5): one realization per control/target
    pair. A control qualifies its target under the target's exact production
    fingerprint, so the two compiled documents' ``model`` blocks are equal as
    [`canonical_model`][causalab.protocol.schema.explicit.canonical_model] materializes them —
    key, revision, dtype, quantization, and the attention backend when authored;
    the list is ``canonical_model``'s, not this module's, so a field added there
    is compared here without anyone re-listing it. Refused naming the first
    field (in sorted order) that differs. A field whose omission
    ``canonical_model`` preserves (the backend; a quantization block), authored
    on one side only, is a difference — the omission is the engine's default
    and the two documents digest differently — and the refusal says so. A
    campaign that changes the realization on **both** steps is re-qualified,
    not refused."""
    mine = canonical_model_ref(control.compiled.document.model)
    theirs = canonical_model_ref(target.compiled.document.model)
    if mine == theirs:
        return
    field = next(
        f for f in sorted(set(mine) | set(theirs)) if mine.get(f) != theirs.get(f)
    )
    message = (
        f"control {name!r} runs the model at model.{field} = "
        f"{mine.get(field)!r}; its target {of!r} runs it at "
        f"{theirs.get(field)!r} — a control qualifies its target under "
        "the target's realization, so the two compiled `model` blocks are "
        "equal as `canonical_model` materializes them (`set` the field on the "
        "control, or change it on both)"
    )
    if (field in mine) != (field in theirs):
        authored, unauthored = (
            (f"its target {of!r}", f"control {name!r}")
            if field in theirs
            else (f"control {name!r}", f"its target {of!r}")
        )
        value = (theirs if field in theirs else mine)[field]
        message += (
            f". model.{field} is authored as {value!r} on {authored} and not "
            f"authored on {unauthored} — an omitted model.{field} is the "
            "engine's default, which is a different model identity, so author "
            "it on both or on neither"
        )
    raise WorkflowError(QUALIFICATION_RULE, message, path=f"steps.{name}.control")


def _check_waiver_applies(
    name: str, kind: str, waiver: Mapping[str, str], doc: Document, is_fit: bool
) -> None:
    reason = waiver["reason"]
    where = f"steps.{name}.waive.{kind}"
    if reason == "no_fit":
        if kind != "matched_random":
            raise WorkflowError(
                CONTROL_RULE,
                f"'no_fit' waives 'matched_random' — the control that needs a fit "
                f"to match — not {kind!r}",
                path=where,
            )
        if is_fit:
            raise WorkflowError(
                CONTROL_RULE,
                f"step {name!r} waives 'matched_random' as 'no_fit' but declares a "
                "fit ('train') — a fit's matched-random control is declared, or "
                "waived as 'external' with a reference",
                path=where,
            )
    elif reason == "single_role":
        if kind != "shuffled_source":
            raise WorkflowError(
                CONTROL_RULE,
                f"'single_role' waives 'shuffled_source' — the control that "
                f"permutes the counterfactual role — not {kind!r}",
                path=where,
            )
        if "counterfactual" in doc.data:
            raise WorkflowError(
                CONTROL_RULE,
                f"step {name!r} waives 'shuffled_source' as 'single_role' but its "
                "document declares a counterfactual role",
                path=where,
            )


def _check_self_swap(
    name: str,
    step: ProtocolStep,
    loaded: InnerProtocol,
    certifiers: Mapping[str, str],
) -> None:
    """A ``self_swap`` control's document holds a self-swap model — one whose
    every write swaps in a read taken from ``original`` on the model's own
    input at the write's own address (site, pos, featurizer, dims) — and a
    script step certifies it. Rule 21 of the IM spec admits equal depth, so
    the document loads; this is the classification the loader adds."""
    doc = loaded.compiled.document
    of = str(step.control["of"]) if step.control else ""
    failures: list[str] = []
    for im_name, im in doc.intervened_models.items():
        failure = _self_swap_failure(doc, im_name, im)
        if failure is None:
            break
        failures.append(failure)
    else:
        detail = "; ".join(failures) if failures else "it declares no intervened model"
        raise WorkflowError(
            CONTROL_RULE,
            f"control {name!r} declares kind 'self_swap' of {of!r}, but no "
            f"intervened model of {step.document!r} is a self-swap: {detail}",
            path=f"steps.{name}.control",
        )
    if name not in certifiers:
        raise WorkflowError(
            CONTROL_RULE,
            f"control {name!r} declares kind 'self_swap' but no script step "
            "certifies it — a self-swap's per-point status is the rows of a "
            f"script step that reads its saved reads and writes {CONTROLS_FILE!r} "
            "(causalab.analysis.certify_control); a no-op nobody certifies "
            "records nothing",
            path=f"steps.{name}.control",
        )


def _self_swap_failure(doc: Document, im_name: str, im: IMSpec) -> str | None:
    """Why ``im`` is not a self-swap model, naming the field — or ``None``."""
    prefix = f"intervened_models.{im_name}"
    writes = im.writes
    if not isinstance(writes, tuple):
        return (
            f"{prefix}.writes is swept or artifact-valued, so it names no fixed write"
        )
    if not writes:
        return f"{prefix} lists no write"
    for wname in writes:
        write = doc.writes.get(wname)
        if write is None:
            return f"{prefix} names unknown write {wname!r}"
        if write.do.mechanism != "swap":
            return f"{prefix}: write {wname!r} is a {write.do.mechanism!r}, not a swap"
        (ref,) = operand_reads(doc, write.do) or (None,)
        operand = ref.read if ref is not None else write.do.payload
        read = doc.reads.get(ref.read) if ref is not None else None
        if read is None or ref is None:
            return (
                f"{prefix}: write {wname!r} swaps in {operand!r}, which is not a read"
            )
        if ref.model is None or not doc.is_unwritten(ref.model):
            return (
                f"{prefix}: operand read {operand!r} is taken from model "
                f"{ref.model!r}, which lands writes — a self-swap reads the "
                "un-intervened model"
            )
        _model, operand_input = doc.group_of(ref)
        if operand_input != im.input:
            return (
                f"{prefix}: operand read {operand!r} has input {operand_input!r}, not "
                f"the model's input {im.input!r}"
            )
        for field in ("site", "pos", "featurizer", "dims"):
            if getattr(read, field) != getattr(write, field):
                return (
                    f"{prefix}: operand read {operand!r} and write {wname!r} differ "
                    f"at {field}: {getattr(read, field)!r} vs {getattr(write, field)!r}"
                )
    return None


def _check_shuffled_source(
    name: str,
    step: ProtocolStep,
    loaded: InnerProtocol,
    of: str,
    target: InnerProtocol,
) -> None:
    """A ``shuffled_source`` control's document is its target's with **one**
    difference: at least one counterfactual role authors ``shuffle: {seed}``
    (IM spec §2.2) and the target's does not. Everything else — every stamped
    dataset digest, every materialized default, every ``set`` override — must
    agree, so the two canonical forms (§7) are compared with ``shuffle`` masked
    and the first differing field is named, the way the self-swap predicate
    names its failing field. The rows are then the same rows in a different
    pairing, the base role untouched; the permutation itself is applied by the
    run (``protocol/positions/roles.resolve_roles``) from the seed alone."""
    where = f"steps.{name}.control"
    head = f"control {name!r} declares kind 'shuffled_source' of {of!r}, but "
    shuffled = _shuffled_roles(loaded.compiled.canonical)
    if not shuffled:
        raise WorkflowError(
            CONTROL_RULE,
            head + f"no counterfactual role of {step.document!r} authors 'shuffle' "
            "— a shuffled-source control is the target's document with "
            "'data.counterfactual.shuffle: {seed}' as its one difference "
            "(IM spec §2.2)",
            path=where,
        )
    if target_shuffled := _shuffled_roles(target.compiled.canonical):
        raise WorkflowError(
            CONTROL_RULE,
            head
            + f"the target itself authors 'shuffle' at {', '.join(target_shuffled)}"
            " — the target is the unshuffled pairing the control permutes",
            path=where,
        )
    difference = _first_difference(
        _mask_shuffle(loaded.compiled.canonical),
        _mask_shuffle(target.compiled.canonical),
    )
    if difference is not None:
        field, mine, theirs = difference
        raise WorkflowError(
            CONTROL_RULE,
            head + f"its document differs from the target's beyond 'shuffle' — "
            f"at {field}: {mine!r} vs {theirs!r}",
            path=where,
        )


def _shuffled_roles(canonical: Mapping[str, Any]) -> list[str]:
    """The counterfactual role paths of a canonical form that author
    ``shuffle`` — ``data.counterfactual`` or ``data.counterfactual[j]``."""
    data = canonical.get("data", {})
    cf = data.get("counterfactual") if isinstance(data, Mapping) else None
    if isinstance(cf, list):
        return [
            f"data.counterfactual[{j}]"
            for j, role in enumerate(cf)
            if isinstance(role, Mapping) and "shuffle" in role
        ]
    if isinstance(cf, Mapping) and "shuffle" in cf:
        return ["data.counterfactual"]
    return []


def _mask_shuffle(canonical: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical form with every counterfactual role's ``shuffle`` removed
    — what a shuffled-source control and its target must agree on."""
    out = json.loads(json.dumps(canonical, sort_keys=True))
    cf = out.get("data", {}).get("counterfactual")
    roles = cf if isinstance(cf, list) else [cf] if isinstance(cf, dict) else []
    for role in roles:
        if isinstance(role, dict):
            role.pop("shuffle", None)
    return out


def _first_difference(
    mine: Any, theirs: Any, path: str = ""
) -> tuple[str, Any, Any] | None:
    """The first leaf at which two JSON trees differ, as ``(field, mine,
    theirs)`` with the field named the way ``--set`` names it (section-rooted:
    ``sites.target.layers``, never ``method.sites…``; §1) — or ``None``. Keys
    are walked in sorted order, list entries by index; a key one side lacks is
    a difference at that key."""
    if isinstance(mine, Mapping) and isinstance(theirs, Mapping):
        for key in sorted(set(mine) | set(theirs)):
            here = f"{path}.{key}" if path else str(key)
            if key not in mine or key not in theirs:
                return _section_rooted(here), mine.get(key), theirs.get(key)
            found = _first_difference(mine[key], theirs[key], here)
            if found is not None:
                return found
        return None
    if isinstance(mine, list) and isinstance(theirs, list):
        for index, (a, b) in enumerate(zip(mine, theirs)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found is not None:
                return found
        if len(mine) != len(theirs):
            return _section_rooted(path), len(mine), len(theirs)
        return None
    if mine != theirs:
        return _section_rooted(path), mine, theirs
    return None


def _section_rooted(path: str) -> str:
    return path[len("method.") :] if path.startswith("method.") else path


def _check_matched_random(
    name: str,
    step: ProtocolStep,
    loaded: InnerProtocol,
    of: str,
    target: InnerProtocol,
    model_info: Callable[[str], ModelInfo],
) -> list[tuple[str, FeaturizerSpec]]:
    """A ``matched_random`` control pairs its target fit: for every featurizer
    the fit trains, the control holds one of the same kind, rank (``k``),
    group and axis, written at the same site; and it declares at least ``min_draws``
    distinct seeds. Returns the ``(control featurizer name, spec)`` pairs, so
    the caller can check — after rule 16 has held the two sites equivalent —
    that the seeds are the draws the document makes (its featurizer's own
    ``seed``, or the ``seed`` of the script step that drew the bundle it
    loads, ``causalab.analysis.random_mask``).

    The rank, group and site clauses are the site-equivalence predicate
    ([`causalab.protocol.equivalence`][]) read over the writes through the
    paired featurizers: a site field the two differ in — unless the control
    declares it in ``non_equivalence`` — or a differing ``k`` / ``group`` /
    ``axis`` is
    refused here under rule 14 in the pairing's own words; every other
    difference is rule 16's."""
    control = dict(step.control or {})
    where = f"steps.{name}.control"
    if target.compiled.document.train is None:
        raise WorkflowError(
            CONTROL_RULE,
            f"control {name!r} declares kind 'matched_random' of {of!r}, which "
            "declares no fit (no 'train') — a matched-random control matches a "
            "fit's rank, site and count; a target that trains nothing waives it "
            "with 'no_fit'",
            path=where,
        )
    if "seeds" not in control:
        raise WorkflowError(
            CONTROL_RULE,
            f"control {name!r} declares kind 'matched_random' without 'seeds' — "
            "the recorded draws are what make the control reproducible",
            path=where,
        )
    seeds = [int(seed) for seed in control["seeds"]]
    min_draws = int(control.get("min_draws", DEFAULT_MIN_DRAWS))
    if len(seeds) < min_draws:
        raise WorkflowError(
            CONTROL_RULE,
            f"control {name!r} declares {len(seeds)} seeds, fewer than min_draws "
            f"{min_draws} — author 'min_draws' to record a smaller draw "
            "deliberately, never silently",
            path=f"{where}.seeds",
        )
    pairs: list[tuple[str, FeaturizerSpec]] = []
    trained = [
        fname
        for fname in target.compiled.document.train.params
        if fname in target.compiled.document.featurizers
    ]
    if not trained:
        raise WorkflowError(
            CONTROL_RULE,
            f"control {name!r} declares kind 'matched_random' of {of!r}, whose fit "
            "trains no featurizer — there is no rank or mask to match",
            path=where,
        )
    for fname in trained:
        fit_spec = target.compiled.document.featurizers[fname]
        candidates = {
            cname: cspec
            for cname, cspec in loaded.compiled.document.featurizers.items()
            if cspec.kind == fit_spec.kind
        }
        if fname in candidates:
            cname = fname
        elif len(candidates) == 1:
            cname = next(iter(candidates))
        else:
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r} has no {fit_spec.kind!r} featurizer to pair with "
                f"fit {of!r}'s {fname!r} (has "
                f"{sorted(loaded.compiled.document.featurizers) or 'none'}"
                f"{'; name it ' + repr(fname) if candidates else ''})",
                path=where,
            )
        cspec = candidates[cname]
        ctl: frozenset[SiteTuple] = coverage(
            loaded.point_documents,
            _info_for(loaded, model_info),
            owner=name,
            writes=_writes_through(loaded.compiled.document, cname),
        )
        fit: frozenset[SiteTuple] = coverage(
            target.point_documents,
            _info_for(target, model_info),
            owner=of,
            writes=_writes_through(target.compiled.document, fname),
        )
        fields = compare(ctl, fit)
        if "featurizer" in fields and cspec.k != fit_spec.k:
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r} pairs featurizer {cname!r} (k={cspec.k!r}) with "
                f"fit {of!r}'s {fname!r} (k={fit_spec.k!r}) — a matched_random "
                "control matches the fit's rank",
                path=f"{where}",
            )
        if "featurizer" in fields and cspec.group != fit_spec.group:
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r} pairs featurizer {cname!r} (group={cspec.group!r}) "
                f"with fit {of!r}'s {fname!r} (group={fit_spec.group!r}) — the unit "
                "one parameter covers must match",
                path=where,
            )
        if "featurizer" in fields and cspec.axis != fit_spec.axis:
            # §2.5 `axis`: the same fact as `group` — what one parameter
            # indexes, a token position or a coordinate — so it has the same
            # named refusal rather than rule 16's generic one
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r} pairs featurizer {cname!r} (axis={cspec.axis!r}) "
                f"with fit {of!r}'s {fname!r} (axis={fit_spec.axis!r}) — what one "
                "parameter indexes, a token position or a coordinate, must match",
                path=where,
            )
        declared = {
            str(f) for f in (control.get("non_equivalence") or {}).get("fields", [])
        }
        if any(f in SITE_FIELDS and f not in declared for f in fields):
            csite, fsite = (
                _written_site(loaded.compiled.document, cname),
                _written_site(target.compiled.document, fname),
            )
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r} writes featurizer {cname!r} at {csite!r}, fit "
                f"{of!r} writes {fname!r} at {fsite!r} — a matched_random control "
                "is drawn at the fit's site",
                path=where,
            )
        pairs.append((cname, cspec))
    return pairs


def _written_site(doc: Document, featurizer: str) -> SiteSpec | None:
    """The site of the first write through ``featurizer``, or ``None``."""
    for write in doc.writes.values():
        chain = write.featurizer
        names = chain if isinstance(chain, tuple) else (chain,)
        if featurizer in names:
            return doc.sites.get(str(write.site))
    return None


def _check_seed_provenance(
    name: str,
    cname: str,
    cspec: FeaturizerSpec,
    seeds: list[int],
    flattened: Mapping[str, Step],
) -> None:
    """The seed a control's featurizer is drawn at is recorded: its own
    ``seed``, or the ``seed`` on the inputs of the script step that drew the
    bundle it loads. ``flattened`` is the whole loaded table (§2.10): a
    ``file_path`` is a run-tree path whose producer is its longest step name —
    ``tail/draw/gate.safetensors`` is drawn by ``tail/draw``, a script step of
    a nested workflow, never by its container ``tail``."""
    where = f"steps.{name}.control.seeds"
    if cspec.seed is not None:
        values = (
            set(cspec.seed.values) if isinstance(cspec.seed, Sweep) else {cspec.seed}
        )
        if values != set(seeds):
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r} declares seeds {seeds} but featurizers.{cname}."
                f"seed draws {sorted(values, key=str)} — the declaration records "
                "the draws the document makes",
                path=where,
            )
        return
    if isinstance(cspec.file_path, str):
        producer = producer_of(cspec.file_path, flattened)
        pstep = flattened.get(producer) if producer is not None else None
        if isinstance(pstep, ScriptStep):
            seed = pstep.inputs.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                if seed not in seeds:
                    raise WorkflowError(
                        CONTROL_RULE,
                        f"control {name!r} declares seeds {seeds} but featurizer "
                        f"{cname!r} is drawn by step {producer!r} at seed {seed} — "
                        "record every draw",
                        path=where,
                    )
                return
            raise WorkflowError(
                CONTROL_RULE,
                f"control {name!r}: featurizer {cname!r} is drawn by step "
                f"{producer!r}, whose inputs carry no integer 'seed' — a control "
                "is reproducible from a recorded seed",
                path=where,
            )
    raise WorkflowError(
        CONTROL_RULE,
        f"control {name!r}: featurizer {cname!r} has no seed to record — author "
        f"featurizers.{cname}.seed (a sweep over the declared seeds) or load it "
        "from a script step that draws at a recorded seed",
        path=where,
    )


# --------------------------------------------------------------------------- #
# bundle-entry checking (rule 9)
# --------------------------------------------------------------------------- #


def _bundle_entries(
    producer: InnerProtocol, file_path: str
) -> dict[str, dict[str, Any]] | None:
    """The tensor keys a producing document will write into ``file_path``, with
    their coordinates — derivable at load because sweeps expand
    deterministically (§3), which is what lets a wrong selection fail here
    instead of after the producing step has run."""
    entries: dict[str, dict[str, Any]] = {}
    document = producer.compiled.document
    for index, save_entry in enumerate(document.save):
        if save_entry.file_path != file_path:
            continue
        if document.aggregation_at(f"save[{index}]") is not None:
            return None  # an aggregation's table, not a tensor bundle
        if save_entry.read is not None:
            slots: tuple[str, ...] = (save_entry.read.read,)
        else:
            spec = producer.compiled.document.featurizers.get(str(save_entry.value))
            kind = spec.kind if spec is not None and isinstance(spec.kind, str) else ""
            slots = FEATURIZER_SLOTS.get(kind, ())
        if not slots:
            return None
        for point in producer.expansion.points:
            short = short_coords(point.coords, entry=save_entry.label)
            label = coordinate_label(point.coords, entry=save_entry.label)
            for slot in slots:
                entries[entry_key(slot, label)] = {"slot": slot, "coords": short}
    return entries or None


def _sole_bundle_slot(entries: Mapping[str, Mapping[str, Any]]) -> str | None:
    """The one slot a bundle holds, or ``None`` when it holds several."""
    slots = {str(entry.get("slot")) for entry in entries.values()}
    return slots.pop() if len(slots) == 1 else None


def _check_script_entry(
    name: str,
    slot: str,
    ref: Reference,
    *,
    steps: Mapping[str, Step],
    inner: Mapping[str, InnerProtocol],
    outputs_of: Any,
) -> None:
    """Rule 9 for a script input: the entry it selects is one the producer will
    write. Checkable only against a *protocol* producer, whose expansion is
    deterministic at load; against a script producer it is a run-time check."""
    producer = steps[str(ref.step)]
    if not isinstance(producer, _DOCUMENT_STEPS):
        return
    entries = _bundle_entries(inner[str(ref.step)], str(ref.file))
    if entries is None:
        return
    bundle_slot = ref.slot or _sole_bundle_slot(entries)
    if bundle_slot is None:
        held = sorted({str(e.get("slot")) for e in entries.values()})
        raise WorkflowError(
            9,
            f"input {slot!r} reads a bundle holding several slots ({held}) — "
            "name one with 'slot'",
            path=f"steps.{name}.inputs.{slot}",
        )
    try:
        select_entry(
            entries.keys(),
            bundle_slot,
            ref.entry,
            what=f"step {name!r}: input {slot!r} reads {ref.target}",
            coords_by_key=entries,
            implicit=False,
        )
    except ValidationError as err:
        raise WorkflowError(9, str(err), path=f"steps.{name}.inputs.{slot}") from err


def _check_entry_selection(
    *,
    consumer: InnerProtocol,
    producer: InnerProtocol,
    run_path: str,
    rest: str,
    step: str,
) -> None:
    """Rule 9, protocol-document half: the entry a load selects must be one the
    producer will write, for every point of the consuming document."""
    entries = _bundle_entries(producer, rest)
    if entries is None:
        return
    for expanded, point in zip(consumer.expansion.points, consumer.point_documents):
        loads: list[tuple[str, str, Any]] = []
        for fname, spec in point.featurizers.items():
            if spec.file_path == run_path:
                kind = spec.kind if isinstance(spec.kind, str) else "identity"
                slots = FEATURIZER_SLOTS.get(kind, ())
                if slots:
                    loads.append((fname, slots[0], spec.entry))
        for pname, pspec in point.params.items():
            if pspec.file_path == run_path:
                loads.append((pname, selector_slot(pspec.entry, "value"), pspec.entry))
        for name, slot, authored in loads:
            want, implicit = entry_selection(authored, expanded.coords, name)
            try:
                select_entry(
                    entries.keys(),
                    slot,
                    want,
                    what=f"step {step!r}: {name!r} loads {run_path!r}",
                    coords_by_key=entries,
                    implicit=implicit,
                )
            except ValidationError as err:
                raise WorkflowError(9, str(err), path=f"steps.{step}") from err


# --------------------------------------------------------------------------- #
# canonical form (§7)
# --------------------------------------------------------------------------- #


def _canon_reference(ref: Reference) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if ref.step is not None:
        entry["step"] = ref.step
        entry["file"] = ref.file
    else:
        entry["path"] = ref.path
    if ref.key is not None:
        entry["key"] = ref.key
    if ref.slot is not None:
        entry["slot"] = ref.slot
    if ref.entry:
        entry["entry"] = dict(ref.entry)
    return entry


def _canon_output(decl: OutputDecl) -> dict[str, Any]:
    entry: dict[str, Any] = {"file": decl.file}
    if decl.columns is not None:
        entry["columns"] = dict(decl.columns)
    if decl.keys is not None:
        entry["keys"] = dict(decl.keys)
    return entry


def _canonicalize(
    document: WorkflowDocument,
    inner_digests: Mapping[str, str],
) -> dict[str, Any]:
    """The canonical form: every default materialized, `after` sorted, each
    protocol step stamped with its document's digest, each script step with
    its script's content hash and the manifest and hash of its declared import
    closure (§4.2, §7), and each behavioral step with its document's digest
    and every authored behavioral field (§2.7).

    ``output_dir`` is **absent**: it names where the run lands, not what the run
    is, so moving a run tree must not change the workflow's identity (§1.1)."""
    canon_steps: dict[str, Any] = {}
    for name in sorted(document.steps):
        step = document.steps[name]
        entry: dict[str, Any] = {"type": step.type}
        if step.description is not None:
            entry["description"] = step.description
        if isinstance(step, ProtocolStep):
            entry["document"] = step.document
            if step.set:
                entry["set"] = dict(step.set)
            if step.max_points is not None:
                entry["max_points"] = step.max_points
            # only when authored (§2.2, §7) — the `runtime`/`reduction` rule: a
            # materialized default here would move every protocol step's
            # identity in the repo to say nothing new about any of them
            if step.control is not None:
                entry["control"] = json.loads(json.dumps(dict(step.control)))
            if step.waive is not None:
                entry["waive"] = {kind: dict(w) for kind, w in step.waive.items()}
            if step.stop_after_failure_rate is not None:
                entry["stop_after_failure_rate"] = step.stop_after_failure_rate
            if step.fan_out is not None:
                # only when authored (§2.9, §7): the width and the join are the
                # step's identity; the children it derives are never here (§6)
                entry["fan_out"] = json.loads(json.dumps(dict(step.fan_out)))
            entry["document_digest"] = inner_digests[name]
        elif isinstance(step, BehavioralStep):
            # every behavioral field is identity (§2.7, §7): the decode spec
            # with its seed, the checker's task binding, the split purpose, the
            # thresholds and the decision. Written on behavioral entries
            # **only** — a key here on any other type would move every step
            # identity in the repo (the per-kind key censuses are the guard)
            entry["document"] = step.document
            if step.set:
                entry["set"] = dict(step.set)
            if step.max_points is not None:
                entry["max_points"] = step.max_points
            entry["document_digest"] = inner_digests[name]
            entry["decoding"] = dict(step.decoding)
            entry["checker"] = json.loads(json.dumps(dict(step.checker)))
            entry["split"] = step.split
            entry["thresholds"] = dict(step.thresholds)
            if step.retain is not None:
                entry["retain"] = json.loads(json.dumps(dict(step.retain)))
            entry["decision"] = dict(step.decision)
            if step.fan_out is not None:
                # only when authored (§2.9, §7), as on a protocol entry
                entry["fan_out"] = json.loads(json.dumps(dict(step.fan_out)))
        elif isinstance(step, DecisionStep):
            # every field is identity (§2.8, §7): the values reference, the
            # rule and the decision mapping. Written on decision entries only
            entry["values"] = _canon_reference(step.values)
            entry["rule"] = json.loads(
                json.dumps({key: dict(clause) for key, clause in step.rule.items()})
            )
            entry["decision"] = dict(step.decision)
        elif isinstance(step, ConditionalStep):
            # the predicate, both gated sides (sorted) and the declared scope
            # (§2.8, §7). Written on conditional entries only
            entry["predicate"] = json.loads(json.dumps(dict(step.predicate)))
            entry["on_true"] = sorted(step.on_true)
            entry["on_false"] = sorted(step.on_false)
            entry["scope"] = step.scope
        elif isinstance(step, WorkflowStep):
            # the nesting is identity (§2.10, §7): the document, the `set` laid
            # over it (only when authored, as on a protocol entry) and the
            # inner document's own digest — so the outer digest moves exactly
            # when the inner one does. Written on workflow entries **only**;
            # the inner's steps are never entries here (both pins are the guard)
            entry["document"] = step.document
            if step.set:
                entry["set"] = json.loads(
                    json.dumps({inner: dict(over) for inner, over in step.set.items()})
                )
            entry["workflow_digest"] = inner_digests[name]
        else:
            entry["script"] = (
                {"module": step.module}
                if step.module is not None
                else {"path": step.path}
            )
            entry["script_sha256"] = step.script_sha256
            # only when a `{"path": …}` script imports a sibling (§4.2, §7): a
            # `{"module": …}` step never carries the keys, so no edit to the
            # package moves a shipped workflow's digest
            if step.closure:
                entry["closure"] = dict(step.closure)
                entry["closure_sha256"] = step.closure_sha256
            entry["inputs"] = {
                key: (
                    _canon_reference(value) if isinstance(value, Reference) else value
                )
                for key, value in sorted(step.inputs.items())
            }
            entry["outputs"] = {
                slot: _canon_output(decl) for slot, decl in sorted(step.outputs.items())
            }
            if step.runtime is not None:
                entry["runtime"] = dict(step.runtime)
            if step.reduction is not None:
                # only when authored (§2.6, §7): a materialized default here
                # would move every script step's identity in the repo
                entry["reduction"] = dict(step.reduction)
            entry["is_deterministic"] = step.is_deterministic
        if step.requires_receipt is not None:
            # only when authored (§2.8, §7), on any kind: a materialized
            # default here would move every step identity in the repo
            entry["requires_receipt"] = dict(step.requires_receipt)
        if step.after:
            entry["after"] = sorted(step.after)
        canon_steps[name] = entry

    canonical: dict[str, Any] = {"version": document.version}
    if document.description is not None:
        canonical["description"] = document.description
    canonical["steps"] = canon_steps
    if document.measurement is not None:
        canonical["measurement"] = dict(document.measurement)
    return canonical
