"""Expand declared workflow fan-out and join its results.

A document step can run over an axis or a fixed number of shards. Children
have distinct names and point assignments. The join checks point coverage and
combines results in the parent's order. Conditional selection determines which
children are required for a selected join.

The parent records the complete point set and join status for downstream steps."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.io.step_record import SIDECAR
from causalab.protocol.rules.errors import ProtocolError, suggest
from causalab.protocol.lowering import short_coords
from causalab.io.tables import TABLE_SUFFIX, write_table
from causalab.workflow.behavioral import (
    DECISION_FILE,
    OUTCOMES,
    OUTCOMES_FILE,
    write_decision,
)
from causalab.workflow.steps import InnerProtocol
from causalab.workflow.document import (
    BehavioralStep,
    ConditionalStep,
    LoadedWorkflow,
    ProtocolStep,
    Step,
    WorkflowError,
    certifier_subject,
)
from causalab.workflow.manifest import CHILD_SEPARATOR

__all__ = [
    "CHILD_SEPARATOR",
    "FAN_OUT_RULE",
    "JOIN_POLICIES",
    "OVER_KEYS",
    "SCOPE_ROOTS",
    "check_fan_out",
    "check_scope",
    "child_name",
    "describe",
    "evidence_holds",
    "expand",
    "expand_conditional",
    "parent_of",
    "parse_fan_out",
    "run_conditional_join",
    "run_join_step",
    "selective_joins",
]

#: The checklist rule this layer refuses under (§5): a fan-out is declared,
#: finite and joined (§2.9).
FAN_OUT_RULE = 19

#: What a fan-out may be declared over (§2.9's table): an axis the compiled
#: document expands, or a literal shard count.
OVER_KEYS: tuple[str, ...] = ("axis", "shards")

#: The join policies (§2.9's table): every child, or the children a per-child
#: verdict left unskipped.
JOIN_POLICIES: tuple[str, ...] = ("all", "selected")

#: The axis root each per-child scope is declared over: a target is a site,
#: a variable is a position (IM spec §2.4). Neither is a shard.
SCOPE_ROOTS: Mapping[str, str] = {"per_target": "sites", "per_variable": "positions"}

_FAN_OUT_KEYS = ("over", "join")
_JOIN_KEYS = ("require",)
_DOCUMENT_STEPS = (ProtocolStep, BehavioralStep)


def _refuse(message: str, path: str) -> WorkflowError:
    return WorkflowError(FAN_OUT_RULE, message, path=path)


def child_name(parent: str, index: int) -> str:
    """``<parent>@<index>`` — the derived name of one child (§1.1)."""
    return f"{parent}{CHILD_SEPARATOR}{index}"


def parent_of(name: str) -> str | None:
    """The parent a child name derives from, or ``None`` for an authored name."""
    head, separator, _ = name.partition(CHILD_SEPARATOR)
    return head if separator else None


# --------------------------------------------------------------------------- #
# the authored grammar
# --------------------------------------------------------------------------- #


def parse_fan_out(raw: Any, path: str) -> dict[str, Any]:
    """The ``fan_out`` block in its parsed form (§2.9): ``over`` exactly one
    of [`OVER_KEYS`][] — ``axis`` a non-empty string, ``shards`` an integer
    of at least 2 (a ``bool`` refused) — and ``join.require`` from
    [`JOIN_POLICIES`][]. ``join`` is required when ``fan_out`` is authored:
    no default is materialized, so nothing is written a reader did not see."""
    grammar = (
        '{"over": {"axis": A} | {"shards": N}, "join": {"require": "all" | "selected"}}'
    )
    if not isinstance(raw, Mapping):
        raise _refuse(f"'fan_out' is an object: {grammar}", path)
    for key in raw:
        if key not in _FAN_OUT_KEYS:
            raise _refuse(
                f"unknown key {key!r} in 'fan_out'{suggest(str(key), _FAN_OUT_KEYS)} "
                f"— the block is {grammar}; a fan-out declares its width and its "
                "join and nothing about devices",
                path,
            )
    if "over" not in raw:
        raise _refuse(
            "'fan_out' declares 'over' — what the step's compiled points are "
            'partitioned by: {"axis": A} or {"shards": N}',
            f"{path}.over",
        )
    over = raw["over"]
    where = f"{path}.over"
    if not isinstance(over, Mapping):
        raise _refuse('\'over\' is {"axis": A} or {"shards": N}', where)
    for key in over:
        if key not in OVER_KEYS:
            raise _refuse(
                f"'over' is one of {list(OVER_KEYS)}, got {key!r}"
                f"{suggest(str(key), OVER_KEYS)} — a fan-out is declared over an "
                "axis the document expands or a literal shard count, never over "
                "what a run reads",
                where,
            )
    if len(over) != 1:
        raise _refuse(
            f"'over' carries exactly one of {list(OVER_KEYS)}, got {sorted(over)}",
            where,
        )
    parsed_over: dict[str, Any]
    if "axis" in over:
        axis = over["axis"]
        if not isinstance(axis, str) or not axis:
            raise _refuse(
                "'over.axis' names an axis of the step's compiled document — a "
                f"non-empty string such as 'sites.target.layers', got {axis!r}",
                f"{where}.axis",
            )
        parsed_over = {"axis": axis}
    else:
        shards = over["shards"]
        if not isinstance(shards, int) or isinstance(shards, bool) or shards < 2:
            raise _refuse(
                f"'over.shards' is a literal integer of at least 2, got {shards!r} — "
                "the width is a pure function of the document, declared when the "
                "workflow is written and never read at run time",
                f"{where}.shards",
            )
        parsed_over = {"shards": shards}
    if "join" not in raw:
        raise _refuse(
            "'fan_out' declares 'join' — how the children are joined: "
            '{"require": "all" | "selected"}; no default is materialized',
            f"{path}.join",
        )
    join = raw["join"]
    where = f"{path}.join"
    if not isinstance(join, Mapping):
        raise _refuse('\'join\' is {"require": "all" | "selected"}', where)
    for key in join:
        if key not in _JOIN_KEYS:
            raise _refuse(
                f"unknown key {key!r} in 'join'{suggest(str(key), _JOIN_KEYS)}", where
            )
    if "require" not in join:
        raise _refuse(
            f"'join' declares 'require' — one of {list(JOIN_POLICIES)}",
            f"{where}.require",
        )
    require = join["require"]
    if not isinstance(require, str) or require not in JOIN_POLICIES:
        raise _refuse(
            f"join.require {require!r} is not one of {list(JOIN_POLICIES)}"
            f"{suggest(str(require), JOIN_POLICIES)}",
            f"{where}.require",
        )
    return {"over": parsed_over, "join": {"require": require}}


# --------------------------------------------------------------------------- #
# load time: the expansion
# --------------------------------------------------------------------------- #


def expand(
    name: str, step: ProtocolStep | BehavioralStep, compiled: InnerProtocol
) -> tuple[tuple[str, Step], ...]:
    """The children of one fanned-out step, ``((name, step), …)`` in child
    order (§2.9): ``over.axis`` gives one child per value of that axis in
    compiled coordinate order, its ``points`` the indices whose coordinate on
    the axis equals the value; ``over.shards: N`` gives ``N`` contiguous
    ranges of the compiled index list. Each child is the parent step with
    ``fan_out`` cleared and ``shard`` set: ``{index, of, over, value | range,
    points}``. Refused: an axis the document does not expand (naming the ids
    it does), an axis with one value, more shards than points."""
    assert step.fan_out is not None
    over = step.fan_out["over"]
    expansion = compiled.expansion
    n_points = len(expansion.points)
    where = f"steps.{name}.fan_out.over"
    selections: list[tuple[dict[str, Any], list[int]]] = []
    if "axis" in over:
        axis = str(over["axis"])
        ids = [a.id for a in expansion.axes]
        if axis not in ids:
            raise _refuse(
                f"'over.axis' names {axis!r}, which is not an axis the document "
                f"{step.document!r} expands (has {ids}){suggest(axis, ids)}",
                f"{where}.axis",
            )
        groups: dict[str, list[int]] = {}
        values: dict[str, Any] = {}
        for index, point in enumerate(expansion.points):
            value = point.coords[axis]
            token = json.dumps(value, sort_keys=True)
            groups.setdefault(token, []).append(index)
            values.setdefault(token, value)
        if len(groups) < 2:
            raise _refuse(
                f"'over.axis' {axis!r} takes one value on {step.document!r} — a "
                "fan-out of one child is the step itself; declare no fan_out",
                f"{where}.axis",
            )
        for token, indices in groups.items():
            selections.append(({"value": values[token]}, indices))
    else:
        shards = int(over["shards"])
        if shards > n_points:
            raise _refuse(
                f"'over.shards' is {shards}, but the document {step.document!r} "
                f"compiles {n_points} point(s) — a shard holds at least one point",
                f"{where}.shards",
            )
        base, extra = divmod(n_points, shards)
        start = 0
        for index in range(shards):
            size = base + (1 if index < extra else 0)
            selections.append(
                ({"range": [start, start + size]}, list(range(start, start + size)))
            )
            start += size
    width = len(selections)
    children: list[tuple[str, Step]] = []
    for index, (label, indices) in enumerate(selections):
        shard = {
            "index": index,
            "of": width,
            "over": json.loads(json.dumps(dict(over))),
            **json.loads(json.dumps(label)),
            "points": list(indices),
        }
        children.append(
            (
                child_name(name, index),
                dataclasses.replace(step, fan_out=None, shard=shard),
            )
        )
    return tuple(children)


def expand_conditional(
    name: str, step: ConditionalStep, width: int
) -> tuple[tuple[str, ConditionalStep], ...]:
    """A per-child conditional's children (§2.9): ``gate@i`` reads
    ``P@i``'s decision and gates ``S@i`` on each side."""
    producer = str(step.predicate["decision"]["step"])
    children: list[tuple[str, ConditionalStep]] = []
    for index in range(width):
        predicate = json.loads(json.dumps(dict(step.predicate)))
        predicate["decision"] = {
            **predicate["decision"],
            "step": child_name(producer, index),
        }
        children.append(
            (
                child_name(name, index),
                dataclasses.replace(
                    step,
                    predicate=predicate,
                    on_true=tuple(child_name(s, index) for s in step.on_true),
                    on_false=tuple(child_name(s, index) for s in step.on_false),
                ),
            )
        )
    return tuple(children)


# --------------------------------------------------------------------------- #
# load time: rule 19
# --------------------------------------------------------------------------- #


def check_fan_out(
    steps: Mapping[str, Step], inner: Mapping[str, InnerProtocol]
) -> None:
    """Rule 19 over the authored steps, before the expansion (§2.9, §5): a
    step that declares ``control``, is named by a ``control.of`` or is a
    certifier's subject declares no ``fan_out`` (the controls layer joins by
    coordinates and knows nothing of children); a step whose document saves
    a ``.safetensors`` bundle declares no ``fan_out`` (a bundle's entries are
    written by one engine call; ``protocol/bundles.py`` addresses entries and
    has no entry-wise writer that round-trips byte-identically, so the join
    re-assembles tables and refuses bundles rather than rebuilding a writer
    outside the shared closure); a step whose document saves a non-value
    entry — ``kind: location_ledger`` (IM spec §2.12; ``SAVE_KINDS``) —
    declares no ``fan_out`` (a ledger row names its point in a ``point``
    column holding the digest string, and the ledger is the run's per-point
    audit table of resolved positions: the join re-assembles measurement
    tables, not audit tables); a step whose
    document saves a file whose name
    (the path's final component) is one of `_SPARSE_FILES` declares no
    ``fan_out`` (those names are the engine's sparse side tables, which the
    join declares sparse and never holds to the missing-file check — a
    fanned-out step's save files are the join's, so a document cannot claim
    one of those names for a dense metric table); ``join.require: selected``
    is declared only
    on a step a per-child conditional gates; and every per-child conditional
    satisfies [`check_scope`][]."""
    controls_of: dict[str, str] = {}
    for other, candidate in steps.items():
        if isinstance(candidate, ProtocolStep) and candidate.control is not None:
            controls_of.setdefault(str(candidate.control.get("of")), other)
    subjects: dict[str, str] = {}
    for other in steps:
        try:
            subject = certifier_subject(steps, other)
        except WorkflowError:
            continue
        if subject is not None:
            subjects.setdefault(subject, other)
    per_child = {
        other: candidate
        for other, candidate in steps.items()
        if isinstance(candidate, ConditionalStep) and candidate.scope != "global"
    }
    for name, step in steps.items():
        if not isinstance(step, _DOCUMENT_STEPS) or step.fan_out is None:
            continue
        where = f"steps.{name}.fan_out"
        if isinstance(step, ProtocolStep) and step.control is not None:
            raise _refuse(
                f"'fan_out' on {name!r}, which declares 'control' — the controls "
                "layer joins a control onto its target by coordinates and knows "
                "nothing of children, so a control step declares no fan_out",
                where,
            )
        if name in controls_of:
            raise _refuse(
                f"'fan_out' on {name!r}, which {controls_of[name]!r} declares a "
                "control of — a step named by a 'control.of' declares no fan_out",
                where,
            )
        if name in subjects:
            raise _refuse(
                f"'fan_out' on {name!r}, the control {subjects[name]!r} certifies — "
                "a certifier's subject declares no fan_out",
                where,
            )
        bundles = sorted(
            str(entry.file_path)
            for entry in inner[name].compiled.document.save
            if str(entry.file_path).endswith(".safetensors")
        )
        if bundles:
            raise _refuse(
                f"'fan_out' on {name!r}, whose document saves the bundle(s) "
                f"{bundles} — a join re-assembles tables by point; a bundle's "
                "entries are written by one engine call and no shared writer "
                "merges them byte-identically, so a step that saves a bundle "
                "declares no fan_out in this version",
                where,
            )
        ledgers = sorted(
            f"{entry.kind} ({entry.file_path})"
            for entry in inner[name].compiled.document.save
            if entry.kind is not None
        )
        if ledgers:
            raise _refuse(
                f"'fan_out' on {name!r}, whose document saves the non-value "
                f"entry(ies) {ledgers} — a location_ledger row names its point in "
                "a 'point' column holding the digest string (IM spec §2.12), and "
                "the ledger is the run's "
                "per-point audit table of resolved positions; the join "
                "re-assembles measurement and behavioral tables by point, not "
                "audit tables, so a step whose document saves an entry "
                "of a kind in SAVE_KINDS declares no fan_out in this version",
                where,
            )
        # the sparse side tables are named, and the join's density decision
        # (`rel not in _SPARSE_FILES`) keys on the name alone: a document
        # whose `save[].file_path` is a free string (IM spec §2.12) could
        # claim one for a dense metric table, which the engine would
        # overwrite with its side table and the join would exempt from the
        # missing-file check. Compared by the path's final component, so a
        # directory prefix does not slip the name past the check
        claimed = sorted(
            str(entry.file_path)
            for entry in inner[name].compiled.document.save
            if Path(str(entry.file_path)).name in _SPARSE_FILES
        )
        if claimed:
            raise _refuse(
                f"'fan_out' on {name!r}, whose document saves {claimed} — a file "
                "whose name (the path's final component) is one of "
                f"{list(_SPARSE_FILES)}, which the engine writes as a sparse side "
                "table and the join declares sparse, joining it from the children "
                "that have it and never holding it to the missing-file check; a "
                "fanned-out step's save files are the join's, so it cannot claim "
                "that name and declares no fan_out in this version",
                where,
            )
        require = step.fan_out["join"]["require"]
        if require == "selected":
            gated_by = sorted(
                other
                for other, gate in per_child.items()
                if name in (*gate.on_true, *gate.on_false)
            )
            if not gated_by:
                raise _refuse(
                    f"join.require 'selected' on {name!r}, which no per_target or "
                    "per_variable conditional gates — 'selected' names the children "
                    "a per-child verdict left unskipped; declare 'all', or gate the "
                    "step",
                    f"{where}.join.require",
                )
    for name, step in per_child.items():
        check_scope(name, step, steps)


def check_scope(name: str, step: ConditionalStep, steps: Mapping[str, Step]) -> None:
    """A per-child scope's conditions (§2.8, §2.9; rule 19): the predicate's
    producer is a fanned-out ``behavioral`` step, its fan-out is over an axis
    under the scope's root ([`SCOPE_ROOTS`][] — never over ``shards``), and
    every gated step is fanned out over the same ``over``, so verdict ``i``
    gates child ``i``."""
    scope = step.scope
    if scope == "global":
        return
    root = SCOPE_ROOTS[scope]
    where = f"steps.{name}.scope"
    producer_name = str(step.predicate["decision"]["step"])
    producer = steps.get(producer_name)
    if not isinstance(producer, BehavioralStep) or producer.fan_out is None:
        raise _refuse(
            f"scope {scope!r} names {producer_name!r} as its producer, which is not "
            "a fanned-out behavioral step — a per-child verdict reads one "
            f"{DECISION_FILE} per child, so the producer declares fan_out (a "
            "decision step has no points to partition; a protocol step writes "
            "no decision)",
            where,
        )
    over = producer.fan_out["over"]
    if "shards" in over:
        raise _refuse(
            f"scope {scope!r} over {producer_name!r}'s fan-out {over} — a shard is "
            "a range of points, neither a target nor a variable; a per-child scope "
            f"is declared over an axis under '{root}.'",
            where,
        )
    axis = str(over["axis"])
    if axis.split(".", 1)[0] != root:
        raise _refuse(
            f"scope {scope!r} over axis {axis!r} — per_target is declared over an "
            "axis under 'sites.', per_variable over an axis under 'positions.'",
            where,
        )
    for side in ("on_true", "on_false"):
        for gated in getattr(step, side):
            candidate = steps[gated]
            if not isinstance(candidate, _DOCUMENT_STEPS) or candidate.fan_out is None:
                raise _refuse(
                    f"{side!r} names {gated!r}, which declares no fan_out — under "
                    f"scope {scope!r} every gated step is fanned out over the "
                    f"producer's axis {axis!r}, so verdict i gates child i",
                    f"steps.{name}.{side}",
                )
            if candidate.fan_out["over"] != over:
                raise _refuse(
                    f"{side!r} names {gated!r}, fanned out over "
                    f"{candidate.fan_out['over']} while the producer {producer_name!r} "
                    f"is over {over} — the two are equal, so verdict i gates child i",
                    f"steps.{name}.{side}",
                )


def selective_joins(steps: Mapping[str, Step]) -> frozenset[str]:
    """The joins declaring ``require: selected`` — the steps a skipped child
    does not skip (§2.9; ``manifest.classify_unreached``, ``conditional.fold_skips``)."""
    return frozenset(
        name
        for name, step in steps.items()
        if isinstance(step, _DOCUMENT_STEPS)
        and step.fan_out is not None
        and step.fan_out["join"]["require"] == "selected"
    )


def describe(step: ProtocolStep | BehavioralStep, children: Sequence[str]) -> str:
    """One ``explain`` clause: ``fan-out over sites.target.layers → 4
    children, join all``."""
    assert step.fan_out is not None
    over = step.fan_out["over"]
    what = over["axis"] if "axis" in over else f"{over['shards']} shards"
    return (
        f"fan-out over {what} → {len(children)} children, "
        f"join {step.fan_out['join']['require']}"
    )


# --------------------------------------------------------------------------- #
# run time: the join
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


#: the row keys whose writers promise **one row per point**: a continuation
#: row's ``point_digest`` (the engine's ``continuations.json``). A metric row
#: carries no digest key — it is placed by its coordinate columns — and is
#: dense by construction (``MetricTable`` writes one row per point per
#: example, ``neural/shared/results.py``); a file placed by either rule
#: enters the join's per-file completeness check
_DENSE_KEYS: tuple[str, ...] = ("point_digest",)

#: the save files whose writers are **sparse** — written per occurrence and,
#: per child, only when there is something to write (``write_outputs`` in
#: ``causalab/io/results_io.py``): the engine's three side tables, each
#: naming its point in a digest-valued ``point`` column. A file some child did
#: not publish is missing unless it is named here — density is the writer's
#: property, so it is declared, never inferred from the rows the join found.
#: Restated rather than imported: ``results.py`` imports torch at module level
#: and loading a fanned-out workflow must not
#: (``test_loading_a_fanned_out_workflow_imports_no_torch``); the census in
#: ``tests/workflow/test_fan_out.py`` holds this tuple to ``TRAIN_EVAL_FILE``,
#: ``FIT_DIAGNOSTICS_FILE`` and ``ROUTING_MISMATCH_FILE``
_SPARSE_FILES: tuple[str, ...] = (
    "train_eval.json",
    "fit_diagnostics.json",
    "routing_mismatch.json",
)


def _plain(value: Any) -> Any:
    """The writer's serialization of a coordinate value as a table column
    (``neural/shared/results.py`` ``_plain``): scalars verbatim, anything
    else as sorted JSON. Restated rather than imported — ``results.py``
    imports torch at module level and loading a fanned-out workflow must not
    (``test_loading_a_fanned_out_workflow_imports_no_torch``)."""
    if isinstance(value, (int, float, str, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _row_digest(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """The point digest a table row names, and the key it named it by:
    ``(digest, key)`` for a continuation row's ``point_digest``, or a
    digest-valued ``point`` — the engine's side tables (``train_eval.json``,
    ``fit_diagnostics.json``, ``routing_mismatch.json``) name their point
    that way; ``None`` when the row names no digest. A metric row names no
    digest at all: the caller places it by its coordinate columns first.

    The key matters to the caller: ``point_digest`` **promises a row per
    point** (the ``continuations.json`` row contract: the
    ``CONTINUATIONS_FILE`` docstring in ``protocol/engine.py``), so a file placed by it
    is held to the per-file completeness check; a digest-valued ``point``
    **only places the row** — the side tables are written per occurrence, not
    per point (``write_outputs`` in ``causalab/io/results_io.py`` writes a side
    table only when there is something to write), so a point with no row in
    such a file is not missing; whether a file some child did not publish is
    missing is the writer's to say, and `_SPARSE_FILES` says it. A
    behavioral ``outcomes.json`` row's ``point`` is an *index* into the
    child's own list and is re-based by the caller; a row with none of these
    cannot be placed and the caller refuses it."""
    for key in (*_DENSE_KEYS, "point"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value, key
    return None


def run_join_step(
    name: str,
    step: ProtocolStep | BehavioralStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    attempt_dir: Path,
    implementation: Mapping[str, Any],
    *,
    skipped: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join the children of ``name`` into ``attempt_dir`` and return the
    parent's record (§2.9, §8): read each child's ``_step.json`` and tables,
    refuse a **missing** point and a **duplicate** point by digest as two
    distinct rule-19 refusals (the join is ``failed``, its dependents
    ``blocked``, never ``skipped``), re-assemble every save file in the
    parent's point order, and — for a behavioral parent — write one
    ``decision.json`` over the summed counts through
    [`write_decision`][]. ``skipped`` is the
    run's ``skipped_by`` map: under ``require: selected`` a skipped child's
    points are not missing and are named under ``join.skipped``.

    The parent's point list — digests **and coordinates** — is
    **reconstructed from the children's own records** (``point_digests`` and
    ``coords``, aligned, in shard order), never read from the load-time
    compile: a deferred document's run-time compile legitimately moves the
    digests *and the coordinate values* (a sweep fed by a step reference
    resolves to the declared representative at load and to the emitted value
    at run time), which is why the join consults the children at all. The
    load-time expansion contributes only the parent's point count (the runner
    has already held the run-time compile to it) and, for a point no child
    published (a skipped child's, under ``selected``), the coordinates a
    message names. So every check here is against what the children
    published — a **foreign** point is a digest a table row names that no
    child of the join declared as its own compiled point, or coordinates that
    no child published a point at, and a child declaring one digest twice is
    a **duplicate** — and a row naming no point (no coordinate columns, no
    ``point_digest``, no ``point``) is refused rather than appended out of
    order. A metric row names its point by its coordinate columns: every axis
    id of the document (the children's ``axes``) is a column (``results.py``
    splats them unconditionally), and the tuple of their values maps to the
    point some child published at those coordinates and thence to the digest
    that child declared there — row provenance is the child's record, not a
    per-row stamp."""
    assert step.fan_out is not None
    fan = step.fan_out
    require = str(fan["join"]["require"])
    children = tuple(loaded.children.get(name, ()))
    skipped = skipped or {}
    what = f"join {name!r} (require {require})"
    where = f"steps.{name}.fan_out.join"
    child_steps = {child: loaded.document.steps[child] for child in children}
    expansion = loaded.inner[name].expansion
    n_points = len(expansion.points)
    # each point's coordinates as the child that ran it published them; the
    # load-time expansion's only for a point no child published (a skipped
    # child's, under `selected`) — a message names those, nothing is placed
    # by them
    published_coords: list[Mapping[str, Any] | None] = [None] * n_points

    def coords_of(index: int) -> dict[str, Any]:
        coords = published_coords[index]
        return short_coords(
            coords if coords is not None else expansion.points[index].coords
        )

    owner: dict[int, str] = {}
    for child in children:
        shard = getattr(child_steps[child], "shard", None) or {}
        for index in shard.get("points", ()):
            owner[int(index)] = child

    records: dict[str, dict[str, Any]] = {}
    skipped_children: list[dict[str, Any]] = []
    for child in children:
        if child in skipped:
            if require != "selected":
                raise _refuse(
                    f"{what}: child {child!r} was skipped by a verdict — under "
                    "'all' every child publishes and a skipped child skips the "
                    "join with it; only 'selected' joins the children a verdict "
                    "left",
                    where,
                )
            skipped_children.append(
                {"child": child, "skipped_by": dict(skipped[child])}
            )
            continue
        record = _read_json(run_root / child / SIDECAR)
        if record is not None:
            records[child] = record

    def missing(index: int, child: str, detail: str) -> WorkflowError:
        digest = loaded.inner[name].point_digests[index]
        return _refuse(
            f"{what}: point {digest} (coords {coords_of(index)}) was published by "
            f"no child — {child!r} {detail}",
            where,
        )

    # the parent's point list, reconstructed from the children by index: a
    # child's record names its points in shard order, so the full list is the
    # union placed at each child's indices
    full: list[str | None] = [None] * n_points
    published_by: dict[str, set[str]] = {}
    # a metric row's coordinate columns → the parent's index: the document's
    # axis ids (the children's `axes`, one document, so one list), in the
    # children's order, serialized as the writer serializes them — built from
    # the coordinates the children published, so a run-time value the
    # load-time compile never saw places its rows
    axis_ids: list[str] = []
    index_by_coords: dict[tuple[Any, ...], int] = {}
    for child in children:
        if child in skipped:
            continue
        shard = getattr(child_steps[child], "shard", None) or {}
        indices = [int(i) for i in shard.get("points", ())]
        record = records.get(child)
        if record is None:
            raise missing(
                indices[0], child, "is failed or absent (no published record)"
            )
        digests = record.get("point_digests")
        if not isinstance(digests, list) or len(digests) != len(indices):
            n_published = len(digests) if isinstance(digests, list) else 0
            raise missing(
                indices[0],
                child,
                f"published {n_published} of its {len(indices)} points",
            )
        coords_list = record.get("coords")
        if (
            not isinstance(coords_list, list)
            or len(coords_list) != len(indices)
            or not all(isinstance(coords, Mapping) for coords in coords_list)
        ):
            n_published = (
                sum(isinstance(coords, Mapping) for coords in coords_list)
                if isinstance(coords_list, list)
                else 0
            )
            raise missing(
                indices[0],
                child,
                f"published the coordinates of {n_published} of its "
                f"{len(indices)} points",
            )
        axes = record.get("axes")
        if not isinstance(axes, list) or not axes:
            # two points need an axis, and a fan-out needs two points
            raise missing(indices[0], child, "published no axes for its points")
        if not axis_ids:
            axis_ids = [str(axis) for axis in axes]
        elif [str(axis) for axis in axes] != axis_ids:
            raise _refuse(
                f"{what}: the children disagree on the document's axes "
                f"({axis_ids} against {child!r}'s {axes}) — every child runs the "
                "parent's compiled document",
                where,
            )
        own_digests: set[str] = set()
        for index, digest, coords in zip(indices, digests, coords_list):
            if str(digest) in own_digests:
                raise _refuse(
                    f"join {name!r}: point {digest} is declared twice in the "
                    f"point_digests of {child!r} — one point, one child",
                    where,
                )
            own_digests.add(str(digest))
            absent_axes = [axis for axis in axis_ids if axis not in coords]
            if absent_axes:
                raise missing(
                    index,
                    child,
                    f"published the point's coordinates without {absent_axes}",
                )
            key = tuple(_plain(coords[axis]) for axis in axis_ids)
            if key in index_by_coords:
                raise _refuse(
                    f"{what}: points {index_by_coords[key]} and {index} share the "
                    f"coordinates {short_coords(coords)} — a metric row is placed "
                    "by its coordinates, which must name one point",
                    where,
                )
            index_by_coords[key] = index
            published_coords[index] = coords
            full[index] = str(digest)
            published_by.setdefault(str(digest), set()).add(child)

    document_digests = {
        str(record.get("document_digest")) for record in records.values()
    }
    if len(document_digests) > 1:
        raise _refuse(
            f"{what}: the children ran different documents ({sorted(document_digests)}) "
            "— every child runs the parent's compiled document",
            where,
        )

    # every file each child published, and every table row's point
    files: list[str] = sorted(
        {str(rel) for record in records.values() for rel in record.get("files", [])}
    )
    rows_by_file: dict[str, list[tuple[int, int, int, dict[str, Any]]]] = {}
    digests_by_file: dict[str, set[str]] = {}
    index_of = {
        digest: index for index, digest in enumerate(full) if digest is not None
    }
    for rel in files:
        if rel == DECISION_FILE and isinstance(step, BehavioralStep):
            continue  # written by the join itself, over the summed counts
        if not rel.endswith(TABLE_SUFFIX):
            raise _refuse(
                f"{what}: {rel!r} is not a JSON table — the join re-assembles tables "
                "by point and nothing else",
                where,
            )
        rows_by_file[rel] = []
        digests_by_file[rel] = set()
        # a file some child did not publish is missing unless its writer is
        # declared sparse: `_SPARSE_FILES` names the engine's side tables,
        # written per occurrence and per child only when there is something
        # to write (`write_outputs`, `io/results_io.py`), and those are joined from the
        # children that have them. Every other file is dense, and an absent
        # one is missing whatever rows the children that have it published —
        # a dense table published empty beside a child that omitted it is
        # missing, never joined empty; density is the writer's, not the data's
        absent = [
            child
            for child, record in records.items()
            if rel not in record.get("files", [])
        ]
        if absent and rel not in _SPARSE_FILES:
            shard = getattr(child_steps[absent[0]], "shard", None) or {}
            raise missing(int(shard["points"][0]), absent[0], f"published no {rel!r}")
        for order, (child, record) in enumerate(records.items()):
            if rel not in record.get("files", []):
                continue
            path = run_root / child / rel
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as err:
                raise ProtocolError(
                    "P2", f"{what}: {path} is not a readable JSON table: {err}"
                ) from err
            if not isinstance(payload, list) or not all(
                isinstance(row, Mapping) for row in payload
            ):
                raise _refuse(
                    f"{what}: {rel!r} of {child!r} is not a table (a JSON array of "
                    "row objects) — the join re-assembles tables by point",
                    where,
                )
            shard = getattr(child_steps[child], "shard", None) or {}
            own = [int(i) for i in shard.get("points", ())]
            # four row shapes, three placement rules — by coordinates, by
            # digest, by `own[point]` — with a re-base attached to the two
            # shapes that carry an int `point`: a metric table row carries
            # every axis id as a column and is placed by that coordinate
            # tuple, through the coordinates the children published, onto
            # the digest the owning child declared at that point; a behavioral
            # `outcomes.json` row carries an int `point` only — the child's
            # own index — and is placed by `own[point]`, then re-based; a
            # `continuations.json` row carries BOTH (the row contract on
            # `CONTINUATIONS_FILE`, `protocol/engine.py`: `point` the index
            # into the child's request, `point_digest` the point itself) — the
            # digest places it and, beside that pairing
            # only, its index is re-based to the parent's, so the joined
            # `continuations.json` and the `outcomes.json` derived from it
            # (`behavioral.py:619`) name the same points, and a row whose two
            # keys disagree about which point it describes is refused; a side
            # table row carries a digest-valued `point` and is placed by it.
            # A metric row's columns are open — the point's coordinates splat
            # in as columns (`results.py`) and nothing reserves a name — so
            # the coordinate rule dispatches FIRST, and an int `point` on such
            # a row is a coordinate named `point` the join never touches
            for position, row in enumerate(payload):
                row = dict(row)
                placed = _row_digest(row)
                raw_point = row.get("point")
                point: int | None = (
                    raw_point
                    if isinstance(raw_point, int) and not isinstance(raw_point, bool)
                    else None
                )
                # a metric row carries every axis id as a column (`axis_ids`
                # is non-empty: a fan-out needs two points, two points an axis)
                if all(axis in row for axis in axis_ids):
                    key = tuple(_plain(row[axis]) for axis in axis_ids)
                    index = index_by_coords.get(key)
                    if index is None:
                        # coordinates no child published a point at: a value
                        # outside the sweep, or a skipped child's point under
                        # 'selected' — either way foreign to the join, like a
                        # digest no child owns
                        named = {axis: row[axis] for axis in axis_ids}
                        raise _refuse(
                            f"join {name!r}: row {position} of {rel!r} in {child!r} "
                            f"names coordinates {named}, which no child of {name!r} "
                            "published — the join's point list is the children's "
                            "point_digests and coordinates, and a metric row is "
                            "placed by its coordinates onto a point of the child "
                            "that wrote it, never appended out of order",
                            where,
                        )
                    digest = full[index]
                    assert digest is not None  # index_by_coords holds published points
                    published_by.setdefault(digest, set()).add(child)
                    digests_by_file[rel].add(digest)
                    parent_index = index
                elif placed is not None:
                    digest, key = placed
                    published_by.setdefault(digest, set()).add(child)
                    if key in _DENSE_KEYS:
                        digests_by_file[rel].add(digest)
                    # a digest no child declared is refused below, as foreign
                    parent_index = index_of.get(digest, n_points)
                    if point is not None and key == "point_digest":
                        if not 0 <= point < len(own):
                            raise _refuse(
                                f"{what}: row {position} of {rel!r} in {child!r} "
                                f"names point {point} beside point_digest {digest}, "
                                f"and {point} is not an index of the child's "
                                f"{len(own)} point(s) — a row's two keys name one "
                                "point",
                                where,
                            )
                        if digest in index_of and own[point] != parent_index:
                            raise _refuse(
                                f"{what}: row {position} of {rel!r} in {child!r} "
                                f"names point {point} (the child's point at that "
                                f"index is the parent's point {own[point]}) beside "
                                f"point_digest {digest} (the parent's point "
                                f"{parent_index}) — the row's two keys disagree "
                                "about which point it describes; a row's two keys "
                                "name one point",
                                where,
                            )
                        row["point"] = parent_index
                elif point is not None:
                    if not 0 <= point < len(own):
                        raise _refuse(
                            f"{what}: row {position} of {rel!r} in {child!r} names "
                            f"point {point}, which is not an index of the child's "
                            f"{len(own)} point(s) — a row is placed by its point, "
                            "never appended out of order",
                            where,
                        )
                    parent_index = own[point]
                    row["point"] = parent_index
                else:
                    raise _refuse(
                        f"{what}: row {position} of {rel!r} in {child!r} names no "
                        "point — no coordinate columns, no 'point_digest', no "
                        f"'point' (its keys: {sorted(row)}); a row the join cannot place "
                        "is refused, never appended out of order",
                        where,
                    )
                rows_by_file[rel].append((parent_index, order, position, row))

    # one point, one child — by digest
    for digest, owners in sorted(published_by.items()):
        if len(owners) > 1:
            first, second = sorted(owners)[:2]
            raise _refuse(
                f"join {name!r}: point {digest} was published by {first!r} and "
                f"{second!r} — one point, one child",
                where,
            )
        if digest not in index_of:
            (child,) = owners
            raise _refuse(
                f"join {name!r}: point {digest}, named by a row {child!r} "
                f"published, is no point any child of {name!r} declared as its "
                "own — the join's point list is the children's point_digests, "
                "and a row is placed by a point of the child that wrote it",
                where,
            )
    # every point of every unskipped child, in every table that names points
    for index in range(n_points):
        child = owner.get(index)
        if child is None or child in skipped:
            continue
        if full[index] is None:
            raise missing(index, child, "is failed or absent (no published record)")
    for rel, seen in digests_by_file.items():
        if not seen:
            continue
        for index in range(n_points):
            child = owner.get(index)
            digest = full[index]
            if child is None or child in skipped or digest is None:
                continue
            if digest not in seen:
                shard = getattr(child_steps[child], "shard", None) or {}
                raise _refuse(
                    f"{what}: point {digest} (coords {coords_of(index)}) was "
                    f"published by no child in {rel!r} — {child!r} published "
                    f"{len(seen & {full[i] for i in shard.get('points', ())})} of "
                    f"its {len(shard.get('points', ()))} points",
                    where,
                )

    # re-assemble, in the parent's point order
    written: list[str] = []
    for rel, entries in rows_by_file.items():
        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        write_table(attempt_dir / rel, [row for _, _, _, row in entries])
        written.append(rel)

    point_digests: list[str | None] = [
        digest if digest is not None else loaded.inner[name].point_digests[index]
        for index, digest in enumerate(full)
    ]
    # the parent's full coordinate list, as `point_digests`: a published
    # point's as its child published it, a skipped slot's from the load-time
    # expansion (the same policy as its digest)
    joined_coords: list[dict[str, Any]] = [
        dict(coords) if coords is not None else dict(expansion.points[index].coords)
        for index, coords in enumerate(published_coords)
    ]
    consumed = {
        child: {
            "identity": record.get("identity"),
            "points": list(record.get("point_digests", [])),
            "digests": dict(record.get("digests", {})),
        }
        for child, record in records.items()
    }
    first_record = next(iter(records.values()), {})
    join: dict[str, Any] = {
        "require": require,
        "consumed": consumed,
        "n_points": sum(len(entry["points"]) for entry in consumed.values()),
        "n_missing": 0,
        "n_duplicate": 0,
    }
    if require == "selected":
        join["skipped"] = skipped_children
    record: dict[str, Any] = {
        "type": step.type,
        "status": "completed",
        "identity": loaded.step_digests[name],
        "implementation": dict(implementation),  # the code that joined (§7)
        "document": step.document,
        "document_digest": next(iter(document_digests), loaded.inner_digests[name]),
        "points": n_points,
        "point_digests": point_digests,
        "coords": joined_coords,
        "axes": list(first_record.get("axes") or [axis.id for axis in expansion.axes]),
        "files": sorted(written),
        "fan_out": {
            "over": json.loads(json.dumps(dict(fan["over"]))),
            "width": len(children),
            "children": list(children),
        },
        "join": join,
    }
    if isinstance(step, BehavioralStep):
        counts = {"n": 0, **{outcome: 0 for outcome in OUTCOMES}, "correct": 0}
        retained = 0
        n_generations = 0
        for child_record in records.values():
            for key, value in (child_record.get("outcomes") or {}).items():
                counts[str(key)] = counts.get(str(key), 0) + int(value)
            retain = child_record.get("retain") or {}
            retained += int(retain.get("retained", 0))
            n_generations += int(retain.get("n", 0))
        if OUTCOMES_FILE not in written:
            raise _refuse(
                f"{what}: no child published {OUTCOMES_FILE} — a behavioral join "
                "decides over the joined outcomes",
                where,
            )
        write_decision(
            attempt_dir,
            step=name,
            split=step.split,
            identity=loaded.step_digests[name],
            thresholds=step.thresholds,
            decision=step.decision,
            counts=counts,
        )
        record["files"] = sorted({*written, DECISION_FILE})
        authored_retain = dict(first_record.get("retain") or {})
        cohort = dict(first_record.get("cohort") or {})
        default_min = int(cohort.get("default_min_examples", 0))
        record.update(
            {
                "checker": dict(first_record.get("checker") or step.checker),
                "split": step.split,
                "outcomes": counts,
                "thresholds": dict(step.thresholds),
                "retain": {
                    **{
                        k: v
                        for k, v in authored_retain.items()
                        if k not in ("retained", "n")
                    },
                    "retained": retained,
                    "n": n_generations,
                },
                "cohort": {
                    "default_min_examples": default_min,
                    "n": counts["n"],
                    "below_default": counts["n"] < default_min,
                },
                "decision": DECISION_FILE,
            }
        )
    return record


def run_conditional_join(
    name: str,
    step: ConditionalStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """The record of a per-child conditional's parent (§2.9): one verdict per
    child, read from the children's published records; ``evidence`` is the
    joined producer's ``decision.json`` (what ``--resume`` holds the record
    to); ``skipped`` is the union of what the children skipped, already folded
    into the run by each child. No data file."""
    producer = str(step.predicate["decision"]["step"])
    verdicts: dict[str, bool | None] = {}
    skipped: set[str] = set()
    for child in loaded.children.get(name, ()):
        record = _read_json(run_root / child / SIDECAR)
        if record is None:
            raise ProtocolError(
                "P2",
                f"step {name!r}: child {child!r} published no record — a per-child "
                "conditional is joined after every child decided",
            )
        verdicts[child] = record.get("verdict")
        skipped |= {str(s) for s in record.get("skipped") or ()}
    decision = _read_json(run_root / producer / DECISION_FILE) or {}
    return {
        "type": "conditional",
        "status": "completed",
        "identity": loaded.step_digests[name],
        "implementation": dict(implementation),
        "predicate": json.loads(json.dumps(dict(step.predicate))),
        "scope": step.scope,
        "verdicts": verdicts,
        "evidence": {
            "step": producer,
            "decision_type": decision.get("decision_type"),
            "outcome": decision.get("outcome"),
            "evidence_identity": decision.get("evidence_identity"),
        },
        "skipped": sorted(skipped),
        "axes": [],
        "files": [],
    }


# --------------------------------------------------------------------------- #
# --resume: the join's evidence
# --------------------------------------------------------------------------- #


def evidence_holds(step: Step, step_dir: Path, record: Mapping[str, Any]) -> bool:
    """Whether a published join still rests on the children in the run tree
    now (§2.9, §8): every consumed child's current ``_step.json`` carries the
    identity, points and digests the join recorded, and every child the join
    named as skipped is still absent. Otherwise the join re-runs. Any other
    kind holds trivially."""
    if not isinstance(step, _DOCUMENT_STEPS) or step.fan_out is None:
        return True
    if step.shard is not None:
        return True
    fan = record.get("fan_out")
    join = record.get("join")
    if not isinstance(fan, Mapping) or not isinstance(join, Mapping):
        return False
    consumed = join.get("consumed")
    if not isinstance(consumed, Mapping):
        return False
    run_root = step_dir.parent
    for child in fan.get("children") or ():
        current = _read_json(run_root / str(child) / SIDECAR)
        want = consumed.get(str(child))
        if isinstance(want, Mapping):
            if current is None:
                return False
            if current.get("identity") != want.get("identity"):
                return False
            if list(current.get("point_digests") or []) != list(
                want.get("points") or []
            ):
                return False
            if dict(current.get("digests") or {}) != dict(want.get("digests") or {}):
                return False
        elif current is not None:
            return False
    return True
