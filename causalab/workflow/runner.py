"""Execute a workflow in dependency order.

The runner resolves each step's inputs and gives it an attempt directory.
Declared outputs are verified before publication. Control qualification and
conditional decisions determine which dependent steps can run.

``--resume`` reuses a step when its identity, runtime, and verified products
match the record. The event stream supplies the statuses written to the manifest."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Sequence

from causalab.protocol.pipeline import (
    AnswerCheck,
    check_parallel,
    compile_protocol,
    resolve_answers,
    resolve_positions,
    route_engine,
    tokenizer_service,
)
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, check_steps_signed
from causalab.protocol.lockstep import SOLO as SOLO_LOCKSTEP
from causalab.protocol.lockstep import Lockstep, decide
from causalab.protocol.pipeline import check_engine
from causalab.protocol.publish import SOLO, Publisher, is_joiner
from causalab.protocol.receipt import (
    MODELS_KEY,
    execution_record,
    measured_bounds,
    model_records,
)
from causalab.protocol.rules.errors import (
    ProtocolError,
    ProtocolWarning,
    ValidationError,
)
from causalab.io.env import ArtifactStore, ResolutionEnv
from causalab.protocol.schema import inline_train_saves, tree_path
from causalab.protocol.lowering import DEFAULT_POINT_CAP, point_count, short_coords
from causalab.io.events import EVENTS_FILE, EventLog, EventSink, read_events
from causalab.io.step_record import SIDECAR, write_sidecar
from causalab.io.tables import TABLE_SUFFIX, read_table
from causalab.provenance import runtime_identity
from causalab.workflow import behavioral, conditional, fan_out, nested
from causalab.workflow.derived import derive_statuses
from causalab.workflow.document import (
    CONTROL_INPUT,
    CONTROLS_FILE,
    DEFAULT_STOP_AFTER_FAILURE_RATE,
    BehavioralStep,
    ConditionalStep,
    DecisionStep,
    LoadedWorkflow,
    OutputDecl,
    ProtocolStep,
    Reference,
    ScriptStep,
    certifier_subject,
    resolve_script,
)
from causalab.workflow.manifest import (
    ATTEMPTS_DIR,
    RETAINED_FAILED_ATTEMPTS,
    STDERR_TAIL_BYTES,
    AttemptFailure,
    classify_unreached,
    displaced_attempt,
    new_attempt_dir,
    prune_attempts,
    publish_attempt,
    retain_superseded,
    superseded_units,
    remove_if_empty,
    restore_displaced,
    write_attempt_record,
    write_manifest,
)
from causalab.workflow.reduction import REDUCTION_INPUT

__all__ = [
    "CERTIFIER_STATUSES",
    "ControlFailure",
    "IMPLEMENTATION_FIELDS",
    "QUALIFICATION_IDENTITY_FIELDS",
    "INSTRUMENT_FAILURE",
    "OverlayArtifacts",
    "SIDECAR",
    "ScriptCall",
    "WRITE_BOUNDARIES",
    "WorkflowRunResult",
    "check_tokenization",
    "coords_token",
    "run_workflow",
    "script_call",
    "verify_output",
]

#: What identity a script-written tensor is stamped as coming from.
SCRIPT_ENGINE = "script"

#: The fields of a step record's ``implementation`` block (§8): what
#: [`causalab.provenance.runtime_identity`][] said about the ``causalab``
#: package that ran the step. Closed — the spec's §8 ``stamping`` row lists
#: exactly these and ``tests/workflow/test_resume_implementation.py`` holds
#: the row, this tuple and the written record together.
#:
#: ``tree_digest`` is compared on ``--resume`` (§7): it is the digest of
#: every byte that executed, so it is "the same code" and nothing else is.
IMPLEMENTATION_FIELDS: tuple[str, ...] = ("tree_digest",)

#: The runner's write boundaries — the instants at which a crash leaves the
#: run tree in a state ``--resume`` has to cope with (§8). Closed: each name
#: is a ``_boundary(...)`` site (``outputs_partial`` is raised from inside a
#: step's own writes, which the runner cannot interpose), and the interruption
#: test enumerates them so a new boundary cannot appear without a recovery
#: case.
WRITE_BOUNDARIES: tuple[str, ...] = (
    "attempt_created",  # the attempt dir exists, nothing written yet
    "outputs_partial",  # some declared outputs written, not all
    "outputs_written",  # every output written, none verified
    "verified",  # outputs verified and digested, no step record yet
    "recorded",  # step record written into the attempt, not published
    "displaced",  # a stale published unit moved aside, new one not yet in place
    "committed",  # the step is published on disk, not yet narrated on the stream: the manifest is refused
    "published",  # the step is published and narrated, the manifest not yet written
    "superseded",  # a displaced prior unit is about to be marked and retained as superseded (§8)
    "manifest",  # the manifest is written to its temp file, not renamed
)


class ScriptFailure(ProtocolError):
    """An isolated script exited non-zero: its stderr travels with the error so
    the failed attempt can retain a bounded tail of it (§8)."""

    def __init__(self, message: str, *, stderr: str) -> None:
        super().__init__("P2", message)
        self.stderr = stderr


class ControlFailure(ProtocolError):
    """A certified control's failure rate exceeded its declared bound, or its
    certifier's rows left a point of the control without a row (§8) — a
    control that did not run on a point cannot certify it. Either way the
    certifying step is ``failed``, so every step downstream of it is
    ``blocked`` by the manifest's own rule. Points that failed certification
    are on the stream as ``warning`` lines with ``reason: instrument_failure``;
    an uncertified point is named in the message and nowhere else."""

    def __init__(self, message: str) -> None:
        super().__init__("P2", message)


#: The ``warning`` reason a control point that failed certification carries on
#: the stream (§4.3). Not a status: ``derive_statuses`` reads only
#: ``attempt_failed`` warnings, so these lines move no step word.
INSTRUMENT_FAILURE = "instrument_failure"

#: What a certifying script's rows may say about a point. The other two words
#: of the status vocabulary (``waived``, ``not_run``) are the layer's, never a
#: certifier's.
CERTIFIER_STATUSES: tuple[str, ...] = ("passed", "failed")

#: The identity of a qualification as a dependent's record carries it (§8,
#: ``controls.identity.<control>``): the control's fully resolved document
#: digest (which document qualified), the ``implementation.tree_digest`` of the
#: code that ran it (the same digest ``--resume`` compares,
#: so a control record from another tree is re-run, never reused) and the
#: engine. Closed on purpose: nothing from the ``execution`` block (``batch_rows``,
#: ``model_source``), no device and no install path — facts a run observes,
#: which legitimately vary between two runs of one campaign — can enter it.
QUALIFICATION_IDENTITY_FIELDS: tuple[str, ...] = (
    "document_digest",
    "tree_digest",
    "engine",
)

_MISSING = object()


def coords_token(coords: Mapping[str, Any], *, entry: str | None = None) -> str:
    """One point's coordinates spelled the way a saved bundle's header spells
    them (``causalab.neural.shared.results.TensorFile.add``): the short axis
    names against ``entry`` — an axis on the saved read's own entity drops the
    entity prefix (``pos``, not ``recv_original.pos``) — and every non-scalar
    value as its sorted JSON text; the whole dict as sorted JSON, so two
    spellings compare as strings.

    The one function the ledger tokenizes with, on both sides of the join
    (§8): a certifying script copies its rows' ``coords`` from a header, so
    the control's points must be spelled the same way or valid work — a
    control swept on an axis of its own saved read, or on a non-scalar
    coordinate — is refused as matching no point."""
    return json.dumps(
        {
            short: _plain(value)
            for short, value in short_coords(coords, entry=entry).items()
        },
        sort_keys=True,
    )


def _plain(value: Any) -> Any:
    """The header writer's value spelling, mirrored byte for byte: a scalar
    as it is, anything else as its sorted JSON text. Mirrored rather than
    imported — the writer lives beside torch, and this module must load for
    ``validate`` without it."""
    if isinstance(value, (int, float, str, bool)):
        return value
    return json.dumps(value, sort_keys=True)


@dataclasses.dataclass
class _ControlLedger:
    """What one run knows about each control step's points (§8): the
    declaration, the control's points (digest, coordinates) with each point's
    status once known, the explicit document a coordinate is looked up in
    when the control did not sweep it, and the certifying step if any.

    ``matched_random``, ``shuffled_source`` and ``full_component`` points are
    ``passed`` when they ran — the pairing (the permutation, the swap) was
    checked at load and the comparison against the target is a reduction, not
    a status. ``self_swap`` points are ``None`` until their certifier publishes
    ``controls.json`` (``not_run`` on the control's own record, which carries
    every point's coordinates so a ``--resume`` re-seats them whether or not
    the certifier is reused)."""

    entries: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def declare(
        self,
        name: str,
        step: ProtocolStep,
        loaded: LoadedWorkflow,
        digests: Sequence[str],
        coords: Sequence[Mapping[str, Any]],
        *,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record a control step's points as it runs; returns the record block.

        ``identity`` is the qualification's identity every dependent inherits
        (§8): [`QUALIFICATION_IDENTITY_FIELDS`][] — the control's resolved
        document digest, the ``tree_digest`` of the code that ran it and the
        engine. Nothing the run merely *observed* (``execution``: the row
        bound, the model source) is in it, so a dependent can never be keyed
        to a fact that legitimately varies between two identical runs."""
        control = dict(step.control or {})
        kind = str(control.get("kind"))
        # a draw that ran, a permutation that ran, or a full-component swap
        # that ran, is `passed`: the comparison against the target is a
        # reduction, not a status (§2.2)
        status = (
            "passed"
            if kind in ("matched_random", "shuffled_source", "full_component")
            else None
        )
        points = [
            {"digest": digest, "coords": dict(point), "status": status}
            for digest, point in zip(digests, coords)
        ]
        self.entries[name] = {
            **control,
            "points": points,
            "raw": loaded.inner[name].compiled.tree,
            "certifier": _certifier_of(loaded, name),
            "identity": {
                field: identity[field] for field in QUALIFICATION_IDENTITY_FIELDS
            },
        }
        block: dict[str, Any] = dict(control)
        # every kind records its points with their coordinates (§8) — the
        # `--resume` path re-seats a swept control from here; a `self_swap`
        # point is `not_run` on its own record until its certifier says
        block["by_point"] = {
            p["digest"]: {"coords": p["coords"], "status": status or "not_run"}
            for p in points
        }
        block["n_points"] = len(points)
        # the load-time site-equivalence verdict against the target (rule 16,
        # §8): derived, so recorded here and never canonical
        if name in loaded.equivalence:
            block["equivalence"] = dict(loaded.equivalence[name])
        if status is not None:
            block["n_failed"] = 0
        return block

    def restore(
        self, name: str, record: Mapping[str, Any], loaded: LoadedWorkflow
    ) -> None:
        """Re-seat what a reused step's record says (``--resume``), so a
        dependent that runs again inherits the same statuses."""
        step = loaded.document.steps.get(name)
        control = record.get("control")
        if isinstance(step, ProtocolStep) and isinstance(control, Mapping):
            by_point = control.get("by_point")
            # a record written before every kind carried `by_point` names its
            # points by digest only: their coordinates are the loader's, when
            # its load-time enumeration signed the same digests
            inner = loaded.inner[name]
            known = dict(
                zip(
                    inner.point_digests,
                    (dict(p.coords) for p in inner.expansion.points),
                )
            )
            points = [
                {
                    "digest": digest,
                    "coords": dict(entry.get("coords", {})),
                    "status": None
                    if entry.get("status") == "not_run"
                    else entry.get("status"),
                }
                for digest, entry in (by_point or {}).items()
            ] or [
                {
                    "digest": digest,
                    "coords": dict(known.get(digest, {})),
                    "status": None,
                }
                for digest in record.get("point_digests", [])
            ]
            implementation = record.get("implementation")
            self.entries[name] = {
                **{
                    k: v
                    for k, v in control.items()
                    if k not in ("by_point", "n_points", "n_failed")
                },
                "points": points,
                "raw": loaded.inner[name].compiled.tree,
                "certifier": _certifier_of(loaded, name),
                # the reused record's own identity (§8): `--resume` already
                # held its tree digest to the running package's
                "identity": {
                    "document_digest": record.get("document_digest"),
                    "tree_digest": implementation.get("tree_digest")
                    if isinstance(implementation, Mapping)
                    else None,
                    "engine": record.get("engine"),
                },
            }
        certifies = record.get("certifies")
        if isinstance(certifies, Mapping):
            subject = str(certifies.get("control"))
            entry = self.entries.get(subject)
            if entry is not None:
                for point in entry["points"]:
                    seen = certifies.get("by_point", {}).get(point["digest"])
                    if isinstance(seen, Mapping):
                        point["status"] = seen.get("status")

    def certify(
        self,
        certifier: str,
        subject: str,
        rows: Sequence[Mapping[str, Any]],
        bound: float,
        emit: Any,
    ) -> dict[str, Any]:
        """Join a certifier's rows onto the control's points by coordinates,
        narrate every failed point, and hold the rate to ``bound``."""
        entry = self.entries.get(subject)
        if entry is None:
            raise ProtocolError(
                "P2",
                f"step {certifier!r} certifies {subject!r}, which has not run in "
                "this process — a control is certified after it publishes",
            )
        if not rows:
            raise ProtocolError(
                "P2",
                f"step {certifier!r} wrote an empty {CONTROLS_FILE} — a certification "
                "with no points decides nothing",
            )
        by_token = self._spellings(entry)
        certified: dict[str, dict[str, Any]] = {}
        for row in rows:
            status = row.get("status")
            if status not in CERTIFIER_STATUSES:
                raise ProtocolError(
                    "P2",
                    f"step {certifier!r}: {CONTROLS_FILE} row says status {status!r}; "
                    f"a certifier says one of {list(CERTIFIER_STATUSES)}",
                )
            coords = row.get("coords")
            token = json.dumps(
                coords if isinstance(coords, Mapping) else {}, sort_keys=True
            )
            point = by_token.get(token)
            if point is None:
                raise ProtocolError(
                    "P2",
                    f"step {certifier!r}: {CONTROLS_FILE} row at coordinates {coords} "
                    f"matches no point of control {subject!r} (has "
                    f"{[json.loads(coords_token(p['coords'])) for p in entry['points']]})",
                )
            if point["digest"] in certified:
                raise ProtocolError(
                    "P2",
                    f"step {certifier!r}: {CONTROLS_FILE} holds two rows at "
                    f"coordinates {coords} — one point, one row",
                )
            certified[point["digest"]] = point
            point["status"] = str(status)
        # every point the control expanded needs a row: a control that did not
        # run on a point cannot certify it, and filling the point in would
        # only dilute the bounded rate (§8)
        uncovered = [p for p in entry["points"] if p["digest"] not in certified]
        if uncovered:
            first = uncovered[0]
            raise ControlFailure(
                f"step {certifier!r}: {CONTROLS_FILE} has no row for "
                f"{len(uncovered)} of {len(entry['points'])} points of control "
                f"{subject!r} — first {first['digest']} at coordinates "
                f"{first['coords']}; a control that did not run on a point cannot "
                f"certify it, so every step downstream of {certifier!r} is blocked"
            )
        failed = [p for p in entry["points"] if p["status"] == "failed"]
        for point in failed:
            emit(
                "warning",
                {
                    "step": subject,
                    "reason": INSTRUMENT_FAILURE,
                    "point": point["digest"],
                    "coords": point["coords"],
                    "certified_by": certifier,
                },
            )
        n_points = len(entry["points"])
        rate = len(failed) / n_points
        block = {
            "control": subject,
            "of": entry.get("of"),
            "kind": entry.get("kind"),
            "by_point": {
                p["digest"]: {"coords": p["coords"], "status": p["status"]}
                for p in entry["points"]
            },
            "n_failed": len(failed),
            "n_points": n_points,
            "stop_after_failure_rate": bound,
        }
        if rate > bound:
            raise ControlFailure(
                f"control {subject!r} ({entry.get('kind')} of {entry.get('of')!r}): "
                f"{len(failed)} of {n_points} points failed certification — rate "
                f"{rate:.3g} exceeds stop_after_failure_rate {bound:g}; the failing "
                "points are instrument_failure on the stream and every step "
                f"downstream of {certifier!r} is blocked"
            )
        return block

    @staticmethod
    def _spellings(entry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Every header spelling of every point of a control, to the point:
        [`coords_token`][] against no entity and against each value the
        control's document saves — a certifier copies its rows' ``coords``
        from one of those bundles' headers. Two points of one expansion
        differ in some axis value, and a spelling keeps every value, so no
        token names two points."""
        raw = entry.get("raw") or {}
        # a `train` save is labelled as the aggregation it names (§2.12)
        saves = (
            inline_train_saves(raw.get("method", {}))
            if isinstance(raw, Mapping)
            else []
        )
        entities: list[str | None] = [None]
        for save in saves:
            if not isinstance(save, Mapping):
                continue
            # the entity a saved value's header is labelled by (§2.12): the
            # read of a tensor entry, the file stem of an aggregation, the
            # featurizer of a bundle
            if "aggregation" in save:
                value: Any = str(save.get("file_path", "")).rsplit("/", 1)[-1]
                value = value.rsplit(".", 1)[0] if "." in value else value
            elif "read" in save:
                value = save.get("read")
            else:
                value = save.get("value")
            if isinstance(value, str) and value not in entities:
                entities.append(value)
        return {
            coords_token(point["coords"], entry=entity): point
            for point in entry["points"]
            for entity in entities
        }

    def inherit(
        self,
        name: str,
        loaded: LoadedWorkflow,
        digests: Sequence[str],
        coords: Sequence[Mapping[str, Any]],
        raw: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """The ``controls`` block of a dependent step (§8): for every point,
        the status of each upstream control whose points agree with it —
        ``instrument_invalid`` where a control ``failed``, ``not_run`` where
        no control covers the point or one covering it has not been
        certified, ``passed`` otherwise — or ``None`` when no control is
        upstream of the step. Beside the statuses, ``identity`` names the
        qualification each status came from, per control (§8): the same
        triple for every point of the fanout, by construction."""
        upstream = _ancestors(loaded.dependencies, name)
        controls = sorted(
            c
            for c, entry in self.entries.items()
            if c in upstream or entry.get("certifier") in upstream
        )
        if not controls:
            return None
        by_point: dict[str, Any] = {}
        n_invalid = 0
        for digest, point in zip(digests, coords):
            per: dict[str, str] = {}
            for control in controls:
                entry = self.entries[control]
                matched = [
                    q
                    for q in entry["points"]
                    if _agree(point, raw, q["coords"], entry["raw"])
                ]
                if not matched:
                    continue  # pinned elsewhere: this control says nothing here
                if any(q["status"] in (None, "not_run") for q in matched):
                    per[control] = "not_run"
                elif any(q["status"] == "failed" for q in matched):
                    per[control] = "failed"
                else:
                    per[control] = "passed"
            if "failed" in per.values():
                status = "instrument_invalid"
                n_invalid += 1
            elif not per or "not_run" in per.values():
                status = "not_run"  # no control covers the point, or none has run
            else:
                status = "passed"
            by_point[digest] = {
                "coords": dict(point),
                "controls": per,
                "status": status,
            }
        return {
            "inherited_from": controls,
            "identity": {
                control: dict(self.entries[control].get("identity", {}))
                for control in controls
            },
            "by_point": by_point,
            "n_invalid": n_invalid,
            "n_points": len(by_point),
        }


def _certifier_of(loaded: LoadedWorkflow, control: str) -> str | None:
    steps = loaded.document.steps
    for name in steps:
        if certifier_subject(steps, name) == control:
            return name
    return None


def _ancestors(dependencies: Mapping[str, tuple[str, ...]], name: str) -> set[str]:
    """Every step upstream of ``name``, transitively."""
    seen: set[str] = set()
    pending = list(dependencies.get(name, ()))
    while pending:
        upstream = pending.pop()
        if upstream in seen:
            continue
        seen.add(upstream)
        pending.extend(dependencies.get(upstream, ()))
    return seen


def _agree(
    coords_a: Mapping[str, Any],
    raw_a: Mapping[str, Any],
    coords_b: Mapping[str, Any],
    raw_b: Mapping[str, Any],
) -> bool:
    """Two points of two documents agree when, on every axis either sweeps,
    the other point has the same value — as its own coordinate, or as the
    value its document authored there (a control pinned to one layer by
    ``set`` agrees with the dependent's point at that layer). Values are
    compared as canonical values (`_same`): ``layers: 18`` and
    ``layers: [18]`` are one. An axis the other document does not have
    constrains nothing."""
    for axis, value in coords_a.items():
        other = coords_b[axis] if axis in coords_b else _lookup(raw_b, axis)
        if other is not _MISSING and not _same(other, value):
            return False
    for axis, value in coords_b.items():
        other = coords_a[axis] if axis in coords_a else _lookup(raw_a, axis)
        if other is not _MISSING and not _same(other, value):
            return False
    return True


def _same(a: Any, b: Any) -> bool:
    """Equal as canonical values. A bare index is the one-layer band ``[n]``
    (IM spec §2.4) — the fold ``schema._band`` and ``explicit._canon_site``
    make, so a sweep value or ``set`` override ``layers: 18`` and a document's
    ``layers: [18]`` are one value here as they are one digest there; a
    longer band stays a list (``18`` and ``[18, 19]`` differ). Mirrored, not
    imported: ``_band`` refuses whatever is not a band, and an axis here may
    carry any value."""
    return _as_band(a) == _as_band(b)


def _as_band(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return value


def _lookup(raw: Mapping[str, Any], axis: str) -> Any:
    """The value an explicit document authors at a section-rooted dotted
    path, or `_MISSING`; a sweep wrapper there is missing too (the
    value would be a coordinate)."""
    node: Any = raw
    for part in tree_path(axis):
        key, index = part, None
        if part.endswith("]") and "[" in part:
            key, rest = part[:-1].split("[", 1)
            index = int(rest) if rest.isdigit() else None
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
        if index is not None:
            if not isinstance(node, list) or index >= len(node):
                return _MISSING
            node = node[index]
    if isinstance(node, Mapping) and "sweep" in node:
        return _MISSING
    return node


def _boundary(name: str, step: str | None) -> None:
    """Fault-injection seam. A no-op in production; the interruption test
    replaces it with one that raises at a chosen ``(name, step)``."""
    assert name in WRITE_BOUNDARIES, name


@dataclasses.dataclass(frozen=True)
class OverlayArtifacts:
    """The §3 overlay: step outputs in the run tree shadow the external
    artifacts root. Every check is real here — this is the run-time store the
    load-time [`DeferredArtifacts`][causalab.workflow.document.DeferredArtifacts] defers
    to."""

    run_root: Path
    outer: ArtifactStore
    step_names: frozenset[str]

    def _local(self) -> Any:
        from causalab.io.env import FileArtifacts

        return FileArtifacts(root=self.run_root)

    def _in_run_tree(self, ref: str) -> bool:
        # STEP NAMES shadow the external root (§3) — exactly the names, never
        # mere directory existence: a rerun into a used output dir, or an
        # external ref matching a stray directory, must not change resolution
        return ref.split("/", 1)[0] in self.step_names

    def read_value(self, artifact: str, key: str) -> Any:
        if self._in_run_tree(artifact):
            return self._local().read_value(artifact, key)
        return self.outer.read_value(artifact, key)

    def file_digest(self, file_path: str) -> str:
        if self._in_run_tree(file_path):
            return self._local().file_digest(file_path)
        return self.outer.file_digest(file_path)

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        if self._in_run_tree(file_path):
            return self._local().read_identity(file_path)
        return self.outer.read_identity(file_path)

    def resolve_path(self, file_path: str) -> Path:
        if self._in_run_tree(file_path):
            return self.run_root / file_path
        outer_resolve = getattr(self.outer, "resolve_path", None)
        if outer_resolve is None:
            raise ProtocolError(
                "P2", f"outer artifact store cannot resolve {file_path!r} to a file"
            )
        return outer_resolve(file_path)


@dataclasses.dataclass(frozen=True)
class WorkflowRunResult:
    """What one workflow run produced: the manifest (also on disk as
    ``workflow.json``) and the run tree it landed in."""

    manifest: Mapping[str, Any]
    run_root: Path


def run_workflow(
    loaded: LoadedWorkflow,
    env: ResolutionEnv,
    out_root: Path,
    engine: Engine | None,
    *,
    resume: bool = False,
    reuse_nondeterministic: bool = False,
    sink: EventSink | None = None,
    publisher: Publisher = SOLO,
    lockstep: Lockstep = SOLO_LOCKSTEP,
) -> WorkflowRunResult:
    """Execute one loaded workflow into ``<out_root>/<output_dir>/``.

    Each step is attempted in its own directory, verified, then published by
    one rename (§8). The manifest is written in the ``finally`` — a failed or
    interrupted run still classifies every step, unless the derived status
    and the runner's memory disagree or the stream cannot be read (§4.3), then
    no manifest is written rather than a wrong one — and if writing it fails
    while a step failure is in flight, the step failure is what propagates.

    **A launched world** (``docs/model_parallelism.md`` §3, §11). ``publisher``
    is this process's place in it — [`SOLO`][],
    world 1, unless a launcher set another — and rides on every protocol
    and behavioral step's request, so the engine's collectives meet on every
    rank and one rank writes each step's outputs. Every rank runs this loop
    over the same steps; the **joiner** ([`is_joiner`][])
    alone touches the run tree — the attempt directories, the records, the
    stream, the manifest, the reuse decision ``--resume`` reads from it —
    and every decision it makes there is **agreed** through ``lockstep``
    ([`causalab.protocol.lockstep`][]) before the ranks move on: whether a
    step's turn reuses its published unit or attempts it (after the receipt
    checks), the record once the attempt is verified and published (a rank
    that does not publish has by then run the engine's half of the step and
    nothing else), and the manifest. A refusal on the joiner is agreed
    before it propagates, so it is a refusal on every rank and no rank waits
    on a collective the joiner never reaches. The ranks share the run tree
    (a spawn is one node; a ``torchrun`` group mounts one), since a later
    step's compile on any rank reads the joiner's published outputs.

    Beside the manifest the run appends its **event stream**, ``events.jsonl``
    (§4.3; [`causalab.io.events`][]): ``phase_started`` as each step's turn
    begins, ``result_committed`` at the publish moment, ``phase_completed``
    when a step is published or reused, ``warning`` for a retained failed
    attempt, and ``campaign_terminal`` once the manifest is written by a run
    that ran to its end (every step completed, or a step failed) — so a run
    that never got there, or one interrupted (``KeyboardInterrupt``,
    ``SystemExit``: the manifest is written, no terminal line), leaves a
    stream without one (`_terminal`). ``sink`` is the optional
    adapter handed each line after its local write; its failure becomes a
    ``warning`` line and changes nothing else. The stream is a sidecar: it is
    an input to no reuse decision and no identity, and ``workflow.json`` is
    byte for byte what a run with no sink writes.

    **The stream is the authority for status** (§4.3, §8). Each step's
    ``status`` in ``workflow.json`` is [`derive_statuses`][]
    over the lines this run appended, not the runner's memory of what it did;
    the memory supplies the other fields (files, digests, ``error``,
    ``blocked_by``) and must agree on the word — if it does not (an emitter
    bug, or an interrupt that landed between a memory assignment and its
    emit), no manifest is written: a ``ProtocolError`` naming the step and
    both words is raised *before* the write, or the interrupt propagates as
    itself beside a ``ProtocolWarning``. A stream this run cannot read back
    at write time likewise leaves no manifest and the failure in flight as
    it was — the manifest is derived from the stream, never from memory in
    its place (`_derived_statuses`). A stream this run cannot open
    (a torn ``events.jsonl`` in the run tree, a ``seq`` gap) is a
    ``ProtocolError`` here, before any step, chaining the read error that
    names ``path:line``; nothing is written — move the sidecar aside to start
    a fresh stream, or restore it byte-for-byte."""
    # Once per run, before anything is written: the identity every record of
    # this run carries and every reuse decision compares (§7). A run that
    # cannot say what code it is (`ProvenanceError`) leaves no tree claiming
    # it can.
    if loaded.document.measurement is not None:
        raise ProtocolError(
            "MEASUREMENT",
            "a workflow with measurement settings requires 'causalab measure'",
        )
    implementation = _implementation()
    # one tokenizer load per model for the whole run: the checks before step
    # 1 and every step's check at its turn share this service
    env = dataclasses.replace(env, tokenizers=tokenizer_service(env))
    # every static inner document's metric answers, with its model's
    # tokenizer (IM spec §2.10), before step 1 and before anything is
    # written: a table the tokenizer cannot score refuses the run here, not
    # at the step that scores it after every earlier step and its weights.
    # Deterministic, so every rank of a launched world refuses alike. Under
    # --resume a step may be reused and never run, and its tokenizer may not
    # load here (an offline node, a gated repository), so each attempted
    # step resolves its answers at its turn instead (`covered`)
    covered: set[str] = (
        set() if resume else set(check_tokenization(loaded, env, engine=engine))
    )
    # the one rank that writes the run tree (docs/model_parallelism.md §3);
    # every other rank computes each protocol step's engine half and follows
    # the joiner's agreed decisions for the rest
    joins = is_joiner(publisher)
    run_root = out_root / loaded.document.output_dir
    if joins:
        run_root.mkdir(parents=True, exist_ok=True)
    overlay = OverlayArtifacts(
        run_root=run_root,
        outer=env.artifacts,
        step_names=frozenset(loaded.document.steps),
    )
    run_env = ResolutionEnv(
        datasets=env.datasets,
        artifacts=overlay,
        model_info=env.model_info,
        tokenizers=env.tokenizers,
    )
    step_manifest: dict[str, Any] = {}
    failure: BaseException | None = None
    manifest: dict[str, Any] | None = None
    # the controls layer's memory for this run (§8): what each control step's
    # points are, and their statuses once a certifier says
    ledger = _ControlLedger()
    # beside the manifest, never inside a step directory (§4.3); a step name
    # has no dot (§5 rule 3), so the two cannot collide. Opening reads the
    # existing stream back to continue its `seq`: one it cannot read (a torn
    # tail, a foreign line, a gap) refuses the run before anything is written
    stream = run_root / EVENTS_FILE
    log: EventLog | _FollowerLog
    if joins:
        try:
            log = EventLog(stream, identity={}, sink=sink)
        except (ValueError, OSError) as err:
            raise ProtocolError(
                "P2",
                f"{stream} cannot be read ({err}); the run was not started — move "
                "the sidecar aside to start a fresh stream or restore it "
                "byte-for-byte",
            ) from err
    else:
        log = _FollowerLog()

    # the conditional layer's memory for this run (§2.8): every step a verdict
    # took out of the run, directly or through a step it depends on, with the
    # decision that did it — filled as each conditional runs or is reused,
    # read when the skipped step's turn comes
    skipped_by: dict[str, dict[str, Any]] = {}
    # the joins a skipped child does not skip (§2.9, `require: selected`)
    selective = fan_out.selective_joins(loaded.document.steps)
    # a nested workflow's steps (§2.10) execute rooted at `<run_root>/<step>/`,
    # against their own document's table and an overlay over that sub-root —
    # the inner document's `artifact: "<inner>/file"` strings resolve there —
    # while the stream, the manifest and `skipped_by` carry the flattened
    # names; one overlay per sub-root, built as its first step comes up — the
    # sub-root IS the key (it is `step_root`), so one document nested twice
    # gets two overlays whichever objects the loader hands back
    envs: dict[str, ResolutionEnv] = {"": run_env}
    try:
        for name in loaded.order:
            owner, local, rel = nested.locate(loaded, name)
            step = owner.document.steps[local]
            step_root = run_root / rel if rel else run_root
            step_dir = step_root / local
            if rel not in envs:
                envs[rel] = ResolutionEnv(
                    datasets=env.datasets,
                    artifacts=OverlayArtifacts(
                        run_root=step_root,
                        outer=env.artifacts,
                        step_names=frozenset(owner.document.steps),
                    ),
                    model_info=env.model_info,
                    tokenizers=env.tokenizers,
                )
            step_env = envs[rel]
            if joins:
                restore_displaced(step_root, local, step_dir)
            log.emit("phase_started", {"step": name, "type": step.type})
            if name in skipped_by:
                # the third outcome (§2.8, §8): no attempt, no directory, no
                # `result_committed`; memory before the emit, as for a reuse
                skipped = conditional.skipped_entry(step.type, skipped_by[name])
                step_manifest[name] = skipped
                log.emit(
                    "phase_completed",
                    {
                        "step": name,
                        "status": "skipped",
                        "skipped_by": skipped["skipped_by"],
                    },
                )
                if joins and step_dir.exists():
                    # a rerun's skip supersedes the step's earlier published
                    # unit (§8): retained and marked, never left where a
                    # reader beside its files would take it for accepted
                    _boundary("superseded", name)
                    retain_superseded(
                        step_dir,
                        step_root / ATTEMPTS_DIR / local,
                        superseded_by={
                            "attempt": None,
                            "identity": None,
                            "published": None,
                            "skipped_by": skipped["skipped_by"],
                        },
                    )
                if joins:
                    _attach_superseded(step_root, local, skipped)
                continue
            try:
                # the step's turn (docs/model_parallelism.md §11): reused, or
                # to be attempted — decided by the joiner, who reads the tree,
                # and agreed before any rank's engine runs. §2.8: a required
                # receipt is checked before the step is scheduled — before an
                # attempt directory, before any engine is chosen
                # (`route_engine` is inside `_attempt_step`), before any
                # device. A refusal is a failed attempt like any other, and
                # it changes no file under `step_dir`: an earlier published
                # unit stays as published (`disposition: accepted`) while the
                # manifest's `failed` is the authority (§2.8) — the skip path
                # above retains and marks because a skip is a decision about
                # the run; a refusal is an attempt that produced nothing
                turn = decide(
                    lockstep,
                    joins,
                    "turn",
                    name,
                    lambda: _turn(
                        loaded,
                        owner,
                        name,
                        local,
                        step,
                        step_dir,
                        run_root,
                        resume,
                        reuse_nondeterministic,
                        implementation,
                        engine,
                        run_env=step_env,
                    ),
                )
                reused = turn["reused"]
                if reused is None and name not in covered:
                    # attempted, and not checked before step 1 (--resume):
                    # a static document's answers resolve now, before its
                    # engine; a step-dependent one resolves in `_protocol_run`
                    covered.update(
                        check_tokenization(loaded, env, engine=engine, steps=(name,))
                    )
                if reused is not None:
                    step_manifest[name] = reused
                    ledger.restore(local, reused, owner)
                    if isinstance(step, ConditionalStep):
                        # a reused conditional re-seats its verdict from its
                        # record
                        conditional.fold_skips(
                            name,
                            _flattened_skips(loaded, rel, reused),
                            loaded,
                            skipped_by,
                        )
                    log.emit("phase_completed", _completed_payload(name, reused))
                    # the retained prior units are the tree's, not the
                    # record's: a reused entry lists them as a fresh run's
                    # does (§8)
                    if joins:
                        _attach_superseded(step_root, local, reused)
                    continue
                displaced_unit: list[Path | None] = [None]

                def attempt() -> dict[str, Any]:
                    # the joiner's attempt → verify → publish; the unit it
                    # displaced is kept aside for the narration below
                    record, displaced_unit[0] = _attempt_step(
                        local,
                        step,
                        owner,
                        step_env,
                        step_root,
                        engine,
                        implementation,
                        ledger=ledger,
                        emit=log.emit
                        if owner is loaded
                        else _emit_as(log.emit, local, name),
                        skipped=nested.local_skips(rel, skipped_by),
                        publisher=publisher,
                    )
                    return record

                entry = decide(
                    lockstep,
                    joins,
                    "attempt",
                    name,
                    attempt,
                    # a rank that does not publish: the engine's half of the
                    # step, whose collectives the joiner's engine meets
                    follow=lambda: _follow_step(
                        local, step, owner, step_env, step_dir, engine, publisher
                    ),
                )
                displaced = displaced_unit[0]
            except BaseException as err:
                step_manifest[name] = {
                    "type": step.type,
                    "status": "failed",
                    "error": {"type": type(err).__name__, "message": str(err)},
                }
                # the attempt is retained under `.attempts/` (§8); the stream
                # says so, and then the failure propagates as before
                log.emit(
                    "warning",
                    {
                        "step": name,
                        "reason": "attempt_failed",
                        "error": step_manifest[name]["error"],
                    },
                )
                raise
            step_manifest[name] = entry
            if isinstance(step, ConditionalStep):
                conditional.fold_skips(
                    name, _flattened_skips(loaded, rel, entry), loaded, skipped_by
                )
            # the publish moment: the verified attempt is `<step>/` now (§8),
            # and memory says so before the stream does. A run dying at the
            # `committed` seam leaves the two disagreeing (`completed` vs
            # `pending`), so the `finally` refuses the manifest and `--resume`
            # finds the unit. Past the two emits the stream and the memory
            # agree, so a failure at `published` — after the publish, before
            # the manifest — gets a manifest whose `completed` is derived
            _boundary("committed", name)
            log.emit("result_committed", {"step": name, "files": list(entry["files"])})
            log.emit("phase_completed", _completed_payload(name, entry))
            _boundary("published", name)
            if not joins:
                # the tree is the joiner's: nothing to retain or list here
                continue
            if displaced is not None:
                # supersession preserves (§8): the unit this publish displaced
                # is marked and retained, never deleted — after the publish is
                # narrated, so a death here leaves a published step and a
                # displaced unit the next run's `restore_displaced` retains
                _boundary("superseded", name)
                retain_superseded(
                    displaced,
                    step_root / ATTEMPTS_DIR / local,
                    superseded_by={
                        "attempt": displaced_attempt(displaced),
                        "identity": entry["identity"],
                        "published": name,
                    },
                )
            _attach_superseded(step_root, local, entry)
    except BaseException as err:
        failure = err
        raise
    finally:
        # the manifest: the joiner derives and writes it, then agrees it —
        # or its failure to — with every rank, whatever brought the ranks
        # here, so a follower never waits on a joiner that has left
        # (docs/model_parallelism.md §11)
        manifest = _agreed_manifest(
            lockstep,
            joins,
            failure,
            lambda: _write_run_manifest(
                loaded, run_root, log, step_manifest, selective, failure
            )
            if joins
            else None,
        )
    if joins:
        for sub_root in nested.sub_roots(loaded):
            # a nested sub-root (§2.10) a clean run left nothing in: its
            # `.attempts/`, then the directory itself when every step was
            # skipped
            remove_if_empty(run_root / sub_root / ATTEMPTS_DIR)
            remove_if_empty(run_root / sub_root)
        remove_if_empty(run_root / ATTEMPTS_DIR)
    if manifest is None:
        # unreachable by construction — the manifest is withheld only while a
        # failure propagates out of the `finally` — but the returned
        # scientific record is guarded by a refusal, not by an `assert` that
        # `python -O` drops
        raise ProtocolError(
            "P2",
            "workflow.json was not written and no failure is in flight; the "
            "run has no manifest to return",
        )
    return WorkflowRunResult(manifest=manifest, run_root=run_root)


class _FollowerLog:
    """The event log of a rank that does not publish: nothing is written,
    the stream is the joiner's (§4.3; ``docs/model_parallelism.md`` §3)."""

    def emit(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        del event, payload


def _turn(
    loaded: LoadedWorkflow,
    owner: LoadedWorkflow,
    name: str,
    local: str,
    step: Any,
    step_dir: Path,
    run_root: Path,
    resume: bool,
    reuse_nondeterministic: bool,
    implementation: Mapping[str, Any],
    engine: Engine | None,
    *,
    run_env: ResolutionEnv | None = None,
) -> dict[str, Any]:
    """The joiner's decision at a step's turn (``docs/model_parallelism.md``
    §11): the published unit ``--resume`` reuses, or ``None`` once every
    required receipt (§2.8) has been checked against the run tree — both
    read from files only the joiner holds, so decided here and agreed.
    ``run_env`` is the step's run-tree environment, against which a step that
    loads an earlier step's output is recompiled (`_reusable`)."""
    reused = _reusable(
        owner,
        local,
        step,
        step_dir,
        resume,
        reuse_nondeterministic,
        implementation,
        engine,
        run_env=run_env,
    )
    if reused is not None:
        return {"reused": reused}
    for _, container, at in nested.containers(loaded, name):
        # a receipt on the `workflow` step itself (§2.10): checked before any
        # of its steps is allocated. The rebased step against the run root —
        # a flattened producer name is the path to its receipt, so both names
        # in the refusal are the ones the manifest and the stream carry
        # (`a/b` requires `a/gate_k`; never `b`, never `gate_k`)
        flat = nested.qualified(at, container)
        conditional.check_receipt(flat, loaded.document.steps[flat], run_root)
    conditional.check_receipt(name, loaded.document.steps[name], run_root)
    return {"reused": None}


def _write_run_manifest(
    loaded: LoadedWorkflow,
    run_root: Path,
    log: Any,
    step_manifest: Mapping[str, Any],
    selective: Any,
    failure: BaseException | None,
) -> dict[str, Any] | None:
    """The joiner's end of a run (§4.3, §8): every step classified, the
    manifest derived from the stream, written, and the stream's terminal
    line once it is on disk. ``None`` when no manifest may be written and a
    failure is propagating as itself (a ``ProtocolWarning`` has said why);
    on a clean run the same conditions are a ``ProtocolError`` instead."""
    entries: dict[str, Any] = {
        **step_manifest,
        **classify_unreached(
            loaded.order, loaded.dependencies, step_manifest, selective=selective
        ),
    }
    # §4.3: the stream is the authority for status. `derived` is the word
    # each step carries — or None when no manifest may be written
    derived = _derived_statuses(log, loaded, entries, failure, selective)
    if derived is None:
        return None
    manifest: dict[str, Any] = {
        "output_dir": loaded.document.output_dir,
        "steps": {
            name: {**entry, "status": derived[name]} for name, entry in entries.items()
        },
    }
    if loaded.nondeterministic:
        manifest["nondeterministic"] = list(loaded.nondeterministic)
    if loaded.nested:
        # record-only (§2.10, §8), like `nondeterministic`: which steps each
        # nested workflow contributed and the inner digest its entry carries
        # — never canonical, never compared on --resume
        manifest["nested"] = nested.manifest_block(loaded)
    written = False
    try:
        write_manifest(run_root, manifest, between=lambda: _boundary("manifest", None))
        written = True
    except BaseException as manifest_err:
        if failure is None:
            raise
        # the step failure is the finding; a manifest that could not be
        # written is reported beside it, never in its place
        warnings.warn(
            f"workflow.json could not be written ({manifest_err!r}); "
            f"the step failure {failure!r} is re-raised",
            ProtocolWarning,
            stacklevel=2,
        )
    if written:
        # the manifest is on disk, so the stream may say the run ran to its
        # end — if it did: an interrupt writes no terminal line
        # (`_terminal`). Only after the manifest (§4.3) — a stream with a
        # terminal line and no manifest would be a lie.
        _terminal(log, manifest, failure)
    return manifest


def _agreed_manifest(
    lockstep: Lockstep,
    joins: bool,
    failure: BaseException | None,
    write: Any,
) -> dict[str, Any] | None:
    """The run's manifest on every rank (``docs/model_parallelism.md`` §11):
    the joiner's ``write`` — its value, or its own failure — agreed as the
    run's last decision. A rank whose step failure is already propagating
    takes a refusal here as the joiner's twin of it and stays with its own;
    the joiner's ``write`` raising propagates on the joiner after the
    agreement, as [`decide`][] does."""
    if failure is None:
        return decide(lockstep, joins, "manifest", None, write)
    # a failure is in flight on this rank: the agreement still runs, so no
    # rank is left waiting, but a refusal it carries is not raised over the
    # failure already propagating
    try:
        return decide(lockstep, joins, "manifest", None, write)
    except ProtocolError:
        if joins:
            raise
        return None


def _follow_step(
    name: str,
    step: Any,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engine: Engine | None,
    publisher: Publisher,
) -> None:
    """A rank that does not publish (``docs/model_parallelism.md`` §3, §11):
    the engine's half of a step — a protocol or behavioral step's compile,
    routing and request, run through the engine whose collectives the
    joiner's engine meets, its outputs discarded by the engine through
    ``publisher``. Every other kind of step runs on the joiner alone; the
    record of any step is the joiner's, agreed through the lockstep."""
    shard = getattr(step, "shard", None)
    selection = None if shard is None else shard["points"]
    if isinstance(step, ProtocolStep) and step.fan_out is None:
        inner, chosen, run = _protocol_run(
            name, step, loaded, run_env, step_dir, engine, publisher, selection
        )
        chosen.execute(inner, run)
    elif isinstance(step, BehavioralStep) and step.fan_out is None:
        behavioral.follow_behavioral_step(
            name,
            step,
            loaded,
            run_env,
            step_dir,
            engine,
            publisher,
            selection=selection,
        )


def _emit_as(emit: Any, local: str, name: str) -> Any:
    """``_attempt_step`` narrates under the name it was handed — an inner
    step's local one — while the stream carries flattened names (§2.10,
    §4.3): a payload whose ``step`` is *this* step is rewritten to ``name``;
    one naming another step (a control's subject) is left for its own owner
    to spell, never renamed to the caller."""

    def wrapped(event: str, payload: Mapping[str, Any]) -> Any:
        return emit(
            event,
            {**payload, "step": name} if payload.get("step") == local else payload,
        )

    return wrapped


def _flattened_skips(
    loaded: LoadedWorkflow, rel: str, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """A conditional's record as ``fold_skips`` reads it (§2.10): its
    ``skipped`` names and its evidence's ``step`` — the producer, which
    ``fold_skips`` copies into every ``skipped_by`` block as ``decision_step``
    — spelled under the conditional's sub-root, so the block's two names are
    both manifest keys; and every ``workflow`` step among the skipped
    replaced by its steps — ``on_false: ["tail"]`` skips every ``tail/*``,
    each its own entry (§8). The record itself keeps what the conditional
    wrote."""
    skipped = nested.prefix_skips(rel, entry.get("skipped") or ())
    evidence = dict(entry.get("evidence") or {})
    if rel and evidence.get("step") is not None:
        evidence["step"] = nested.qualified(rel, str(evidence["step"]))
    return {
        **entry,
        "evidence": evidence,
        "skipped": nested.expand_skips(loaded.document.steps, skipped),
    }


def _completed_payload(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The ``phase_completed`` payload (§4.3): the step, its word and — for a
    protocol step — ``forwards``, the forward groups its engine ran, so a
    reader of the stream can see a qualification ran once for its target's
    whole fanout without opening the step record."""
    payload: dict[str, Any] = {"step": name, "status": entry["status"]}
    if "forwards" in entry:
        payload["forwards"] = entry["forwards"]
    return payload


def _derived_statuses(
    log: EventLog,
    loaded: LoadedWorkflow,
    entries: Mapping[str, Mapping[str, Any]],
    failure: BaseException | None,
    selective: frozenset[str] = frozenset(),
) -> dict[str, str] | None:
    """§4.3: the word each step carries in ``workflow.json``, derived from the
    lines this run appended to the stream — or ``None`` when no manifest may
    be written and the ``failure`` in flight is left to propagate as itself.

    The in-memory ``entries`` supply every other field and must agree on the
    word. Two things stop the stream from deciding:

    * **it cannot be read back** (a torn tail, a foreign line — a sidecar IO
      problem, not a status). The read of a sidecar is held to the rule its
      writes follow ([`write_manifest`][], `_terminal`): it may not
      mask a step failure in flight. With one propagating this warns and
      withholds the manifest; on a clean run it is a ``ProtocolError`` chaining
      the read error. The manifest is derived from the stream and is not
      written without it — never from memory in its place, which would be the
      memory-sourced manifest §4.3 retires.
    * **it disagrees with memory** — an emitter bug, or an interrupt that
      landed between a memory assignment and its emit. An emitter bug is
      refused: a ``ProtocolError`` naming the step and both words, chaining an
      ``Exception`` in flight so neither finding is lost. An interrupt
      (``KeyboardInterrupt``, ``SystemExit`` — a ``BaseException`` that is not
      an ``Exception``) stays an interrupt: this warns, withholds the manifest
      and lets it propagate unchanged, exit code included.
    """
    try:
        derived = derive_statuses(
            (r for r in read_events(log.path) if r["seq"] >= log.opened_at),
            order=loaded.order,
            dependencies=loaded.dependencies,
            selective=selective,
        )
    except (ValueError, KeyError, TypeError, OSError) as read_err:
        if failure is None:
            raise ProtocolError(
                "P2",
                "workflow.json was not written: events.jsonl could not be read "
                f"({read_err}); the manifest is derived from the stream and is "
                "not written without it",
            ) from read_err
        warnings.warn(
            "workflow.json was not written: events.jsonl could not be read "
            f"({read_err!r}); the step failure {failure!r} is re-raised",
            ProtocolWarning,
            stacklevel=3,
        )
        return None
    for name, entry in entries.items():
        if entry["status"] == derived[name]:
            continue
        if failure is not None and not isinstance(failure, Exception):
            warnings.warn(
                f"workflow.json was not written: step {name!r} is "
                f"{entry['status']!r} in memory and {derived[name]!r} on the "
                f"stream because {failure!r} landed between the two",
                ProtocolWarning,
                stacklevel=3,
            )
            return None
        raise ProtocolError(
            "P2",
            f"workflow.json would say step {name!r} is {entry['status']!r} "
            f"but events.jsonl records {derived[name]!r}; the manifest is "
            "derived from the stream and is not written disagreeing with it",
        ) from failure
    return derived


def _terminal(
    log: EventLog, manifest: Mapping[str, Any], failure: BaseException | None
) -> None:
    """The stream's last line, for a run that ran to its end: every step
    completed, or a step failed and the runner still finished (the manifest is
    written). An interrupt — ``KeyboardInterrupt``, ``SystemExit``: a
    ``BaseException`` that is not an ``Exception`` — also gets its manifest
    but is not an end the run reached: no line is appended, so ``terminal()``
    reads False exactly as it does for a run that died before its ``finally``
    (§4.3: the absence is what "did not finish" reads as). A sidecar write may
    not mask a step failure in flight — the same rule the manifest write
    follows — so while one is propagating, a failure here is a warning beside
    it; on a clean run it is raised like any other write."""
    if failure is not None and not isinstance(failure, Exception):
        return
    payload = {
        "outcome": "failed" if failure is not None else "completed",
        "steps": {name: entry["status"] for name, entry in manifest["steps"].items()},
    }
    try:
        log.emit("campaign_terminal", payload)
    except BaseException as stream_err:
        if failure is None:
            raise
        warnings.warn(
            f"events.jsonl could not be appended ({stream_err!r}); "
            f"the step failure {failure!r} is re-raised",
            ProtocolWarning,
            stacklevel=3,
        )


def _implementation() -> dict[str, Any]:
    """The running package's identity as a step record carries it (§7, §8):
    the [`IMPLEMENTATION_FIELDS`][] of ``runtime_identity()``.

    Asked once per run, not per step. ``runtime_identity()`` hashes every file
    of the installed package the first time it is called (a few hundred
    files) and is cached for the process,
    so every step of one run carries the same answer by construction. Imported
    as a module attribute so a test can stand in a different identity.
    ``location`` — an absolute path — is deliberately not recorded: a record
    that carried it would change when the tree moved, and moving a run tree
    alone must not bust reuse."""
    identity = runtime_identity()
    return {"tree_digest": identity.tree_digest}


def _attempt_step(
    name: str,
    step: Any,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    run_root: Path,
    engine: Engine | None,
    implementation: Mapping[str, Any],
    *,
    ledger: _ControlLedger | None = None,
    emit: Any = None,
    skipped: Mapping[str, Mapping[str, Any]] | None = None,
    publisher: Publisher = SOLO,
) -> tuple[dict[str, Any], Path | None]:
    """Attempt → verify → publish for one step (§8): the step's record, and
    the prior unit this publish displaced (or ``None``), which the caller
    retains as superseded once the publish is narrated. ``publisher`` is
    this process's place in a launched world (``docs/model_parallelism.md``
    §3) — the joiner's, since only the joiner attempts — and rides on the
    step's engine request.

    A declared fan-out (§2.9) runs through the same path: a child (``shard``
    set) is its parent's document over a point selection; the parent
    (``fan_out`` set) is the join, an arm here and nothing in the run loop —
    ``skipped`` is the run's ``skipped_by`` map, which a ``selected`` join
    reads to name the children a verdict left out.

    Every write goes to a fresh attempt directory. On any failure —
    ``KeyboardInterrupt`` included — the attempt keeps bounded metadata
    (``attempt.json``) and the exception propagates; the published step
    directory, if any, is untouched.

    The controls layer sits inside the attempt (§8): a control step's points
    are recorded as it runs; a certifying step's ``controls.json`` is joined
    onto its control's points **before** the step is verified and published,
    so a failure rate over the declared bound is a failed attempt like any
    other; and a dependent protocol step's record carries the statuses it
    inherits, written by its own run into its own directory."""
    step_dir = run_root / name
    prune_attempts(run_root, name, keep=RETAINED_FAILED_ATTEMPTS - 1)
    attempt_dir = new_attempt_dir(run_root, name)
    started = time.time()
    ledger = ledger if ledger is not None else _ControlLedger()
    emit = emit if emit is not None else (lambda event, payload: None)
    try:
        _boundary("attempt_created", name)
        shard = getattr(step, "shard", None)
        if (
            isinstance(step, (ProtocolStep, BehavioralStep))
            and step.fan_out is not None
        ):
            # the join (§2.9): the children's records and tables re-assembled
            # by point digest into one receipt; no engine runs here
            entry = fan_out.run_join_step(
                name,
                step,
                loaded,
                run_root,
                attempt_dir,
                implementation,
                skipped=skipped,
            )
        elif isinstance(step, ProtocolStep):
            entry = _run_protocol_step(
                name,
                step,
                loaded,
                run_env,
                attempt_dir,
                engine,
                implementation,
                ledger=ledger,
                selection=None if shard is None else shard["points"],
                publisher=publisher,
            )
        elif isinstance(step, BehavioralStep):
            # the declarative behavioral runner (§2.7): the same attempt →
            # verify → publish path, its record built in behavioral.py
            entry = behavioral.run_behavioral_step(
                name,
                step,
                loaded,
                run_env,
                attempt_dir,
                engine,
                implementation,
                selection=None if shard is None else shard["points"],
                publisher=publisher,
            )
        elif isinstance(step, DecisionStep):
            # a typed decision over a values object (§2.8): decision.json
            # written into the attempt, verified and published like any file
            entry = conditional.run_decision_step(
                name, step, loaded, run_root, attempt_dir, implementation
            )
        elif isinstance(step, ConditionalStep) and name in loaded.children:
            # a per-child conditional's parent (§2.9): one verdict per child,
            # read from the children's records — the children folded the skips
            entry = fan_out.run_conditional_join(
                name, step, loaded, run_root, implementation
            )
        elif isinstance(step, ConditionalStep):
            # the verdict over a producer's decision.json (§2.8): a record and
            # no data file; the skips it names are folded in by the caller
            entry = conditional.run_conditional_step(
                name, step, loaded, run_root, implementation
            )
        else:
            entry = _run_script_step(
                name, step, loaded, run_root, attempt_dir, implementation
            )
            subject = certifier_subject(loaded.document.steps, name)
            if subject is not None:
                control_step = loaded.document.steps[subject]
                bound = (
                    control_step.stop_after_failure_rate
                    if isinstance(control_step, ProtocolStep)
                    and control_step.stop_after_failure_rate is not None
                    else DEFAULT_STOP_AFTER_FAILURE_RATE
                )
                entry["certifies"] = ledger.certify(
                    name,
                    subject,
                    read_table(attempt_dir / CONTROLS_FILE),
                    bound,
                    emit,
                )
        if shard is not None:
            # a child's record (§2.9): its own identity — the parent's entry
            # digest plus its selection, what `--resume` compares — and the
            # shard it ran, beside the sliced `points` / `point_digests`
            entry["identity"] = _step_identity(loaded, name, step)
            entry["shard"] = json.loads(json.dumps(dict(shard)))
        _boundary("outputs_written", name)
        entry["digests"], entry["checks"] = _verify_outputs(
            name, step, attempt_dir, entry["files"]
        )
        _boundary("verified", name)
        # the record's disposition (§8, a closed vocabulary): `candidate`
        # while it sits in the attempt, `accepted` as it is published — a
        # reader never infers acceptance from absence
        entry["disposition"] = "candidate"
        write_sidecar(attempt_dir, entry)
        _boundary("recorded", name)
        entry["disposition"] = "accepted"
        write_sidecar(attempt_dir, entry)
        displaced = publish_attempt(
            attempt_dir, step_dir, between=lambda: _boundary("displaced", name)
        )
    except BaseException as err:
        stderr = err.stderr if isinstance(err, ScriptFailure) else None
        write_attempt_record(
            attempt_dir,
            step=name,
            started=started,
            failure=AttemptFailure.from_exception(err, stderr),
            declared=_declared_files(step),
        )
        raise
    remove_if_empty(attempt_dir.parent)
    return entry, displaced


def _attach_superseded(run_root: Path, name: str, entry: dict[str, Any]) -> None:
    """The step's retained prior units on its manifest entry (§8) — read from
    the run tree, the same on the skip, the reuse and the publish path, so one
    tree lists one `superseded` whichever path a run took to the step."""
    retained = superseded_units(run_root, name)
    if retained:
        entry["superseded"] = retained


def _declared_files(step: Any) -> dict[str, str]:
    """``{relative file: slot}`` for a script step, the decision file for a
    decision step; a protocol step's files are known only after its engine
    ran."""
    if isinstance(step, ScriptStep):
        return {decl.file: slot for slot, decl in step.outputs.items()}
    if isinstance(step, DecisionStep):
        return {behavioral.DECISION_FILE: "decision"}
    return {}


def _reusable(
    loaded: LoadedWorkflow,
    name: str,
    step: Any,
    step_dir: Path,
    resume: bool,
    reuse_nondeterministic: bool,
    implementation: Mapping[str, Any],
    engine: Engine | None,
    *,
    run_env: ResolutionEnv | None = None,
) -> dict[str, Any] | None:
    """The prior run's record for ``name`` if ``--resume`` may reuse it (§8).

    The digest comparison is what makes this correct: a script step's digest
    carries its script's content hash, so editing the script busts the reuse.
    A step that declared itself non-deterministic is never reused silently —
    replaying it is exactly what it said it cannot guarantee.

    A protocol step whose document loads a run-tree path (a fit's
    ``init.file_path``, an apply's ``file_path``) records the digest of its
    run-time compile, and the upstream bytes it read are in that digest. The
    load-time digest keeps the declared path. Such a step is compared with a
    compile against the current run tree, ``run_env`` (`_run_tree_digest`),
    after the cheaper checks below have passed. An unfanned step's record
    carries that digest as its ``identity``; a fanned-out child's identity is
    its parent's entry digest plus its shard, so the child is held to it
    through its ``document_digest``. The step is reused while the bytes it
    loaded and its document are unchanged, and runs again otherwise.

    The **implementation** must be the same code too (§7): the record's
    ``implementation.tree_digest`` — the bytes of the ``causalab`` package
    that ran the step — must equal the running package's, and a record with
    no such block (one written before it was recorded) is not trusted, like a
    record without digests. Refusal here is silent re-execution, exactly as a
    digest mismatch is.

    A protocol step's record carries its **engine** — the third member of the
    qualification identity ([`QUALIFICATION_IDENTITY_FIELDS`][]) — and it
    must be the engine the step would run under now (`_engine_for`): a
    same-tree run under another engine re-runs the step rather than
    inheriting the old engine's identity into its dependents, and a record
    without the key is not trusted, as one without an ``implementation``
    block is not. When no configured engine covers the step on this host
    (`_engine_for` is ``None``) there is no engine to compare against,
    and a record that carries its own is reused: a content-digest-verified
    record of work done is not invalidated by a host that could not redo it
    (fail-closed on a valid ``--resume`` under ``auto``, where the install may
    have changed). A script step records no engine (its outputs' stamps carry
    [`SCRIPT_ENGINE`][]), so nothing is compared for it — and neither is
    for a fanned-out parent (§2.9): its record is the join, which names no
    engine because no engine ran it; its children's records carry theirs,
    and [`fan_out.evidence_holds`][] below binds the join to them.

    And the record's **content digests** must match the published bytes.
    Existence is never enough: a truncated output, a file overwritten in
    place, or a record from before digests were recorded all mean the unit is
    not reusable, and the step runs again.

    A script step's referenced inputs must still be the bytes it read.
    Both external paths and upstream step files enter the canonical entry as
    locators, so ``input_digests`` records their sha256 by slot. Every file
    must match, including a selected value or tensor's containing file. This
    also holds after an interrupted run publishes a changed upstream output.
    Missing or incomplete input digests force execution again.

    And for a **conditional or a decision** the evidence must hold (§2.8): a
    conditional's record names the ``evidence_identity`` it read;
    if the producer's current ``decision.json`` carries another, the numbers
    behind the verdict moved and the conditional is re-evaluated. A decision
    whose values file or producer identity moved is re-made the same way.
    ``evidence_identity`` is a run-time value, so this clause — not the
    load-time digest — is where it binds.

    And a step of **any kind** that declares ``requires_receipt`` is reusable
    only while the producer's current ``decision.json`` still carries the
    required outcome (§2.8): a receipt that flipped since the step ran is
    never reused — the step goes back through ``check_receipt``, which
    refuses."""
    if not resume:
        return None
    record_path = step_dir / SIDECAR
    if not record_path.is_file():
        return None
    try:
        with record_path.open() as handle:
            record = json.load(handle)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    identity = record.get("identity")
    # an unfanned step that loads a run-tree path records its run-time digest
    # as its identity, which the load-time one never equals: it is compared
    # with a compile against the run tree at the end, after the cheap checks
    run_tree = _loads_run_tree(loaded, name, step)
    resolved = run_tree and step.fan_out is None and step.shard is None
    if identity != _step_identity(loaded, name, step) and not resolved:
        return None
    if isinstance(step, ProtocolStep) and step.fan_out is None:
        want_engine = _engine_for(loaded, name, engine)
        recorded_engine = record.get("engine")
        if recorded_engine is None:
            return None
        if want_engine is not None and recorded_engine != want_engine:
            return None
    recorded = record.get("implementation")
    if not isinstance(recorded, dict):
        return None
    if recorded.get("tree_digest") != implementation["tree_digest"]:
        return None
    if isinstance(step, ScriptStep) and not step.is_deterministic:
        if not reuse_nondeterministic:
            return None
    files = record.get("files")
    digests = record.get("digests")
    if not isinstance(files, list) or not isinstance(digests, dict):
        return None
    if set(digests) != {str(rel) for rel in files}:
        return None
    for rel, want_digest in digests.items():
        target = step_dir / str(rel)
        if not target.is_file() or _sha256(target) != want_digest:
            return None
    if isinstance(step, ScriptStep):
        file_inputs = _file_inputs(step, step_dir.parent, loaded.workflow_dir)
        if file_inputs:
            input_digests = record.get("input_digests")
            if not isinstance(input_digests, dict):
                return None
            if set(input_digests) != set(file_inputs):
                return None
            for slot, target in file_inputs.items():
                if not target.is_file() or _sha256(target) != input_digests[slot]:
                    return None
    if not conditional.evidence_holds(step, step_dir, record):
        return None
    if not fan_out.evidence_holds(step, step_dir, record):
        return None
    if run_tree and isinstance(step, ProtocolStep) and step.fan_out is None:
        compiled = identity if step.shard is None else record.get("document_digest")
        if compiled is None or compiled != _run_tree_digest(loaded, step, run_env):
            return None
    return {**record, "status": "reused"}


def _loads_run_tree(loaded: LoadedWorkflow, name: str, step: Any) -> bool:
    """Whether ``step`` is a protocol step whose document loads a run-tree
    path or reads an upstream step's value: the steps the loader keeps in
    their authored form, because their inputs exist only at run time
    (``inner_digest_kind`` is ``"authored"``). A fanned-out child shares its
    parent's kind."""
    return (
        isinstance(step, ProtocolStep)
        and loaded.inner_digest_kind.get(name) == "authored"
    )


def _run_tree_digest(
    loaded: LoadedWorkflow, step: ProtocolStep, run_env: ResolutionEnv | None
) -> str | None:
    """The document digest of the compile `_protocol_run` makes for
    ``step``, against the run tree as it stands now (`_compile_step`).
    ``None`` without a run tree, or when that compile refuses (an upstream
    output is missing or no longer matches): nothing then matches a record,
    the step runs again, and its own attempt reports the refusal."""
    if run_env is None:
        return None
    try:
        return _compile_step(loaded, step, run_env).digests.document
    except ProtocolError:
        return None


def _step_identity(loaded: LoadedWorkflow, name: str, step: Any) -> str:
    """What `--resume` compares: the step's own digest — a behavioral step's
    canonical entry carries its decoding, checker, split and decision, so a
    changed seed or split is a step that runs again (§2.7, §7); a decision's
    and a conditional's carry the rule, the predicate and the gated sides
    (§2.8). A protocol step's is its inner document's digest, as before — except
    a fanned-out one and a child (§2.9): the join's identity is its entry
    digest (the `fan_out` block is in it) and a child's is its parent's plus
    its shard, so a changed width is a step that runs again."""
    if isinstance(step, (ScriptStep, BehavioralStep, DecisionStep, ConditionalStep)):
        return loaded.step_digests[name]
    if isinstance(step, ProtocolStep) and (
        step.fan_out is not None or step.shard is not None
    ):
        return loaded.step_digests[name]
    return loaded.inner_digests[name]


def _engine_for(loaded: LoadedWorkflow, name: str, engine: Engine | None) -> str | None:
    """The ``engine`` a protocol step's record carries when it runs now — what
    a reused record must match (§8): the name of the run's engine when
    [`route_engine`][] would accept it for the step (``check_engine``, the
    same capability match) — computed here over the load-time compile, whose
    verbs are the run-time compile's (a document's needs are authored, never
    resolved from an artifact). ``None`` when no engine was supplied or the
    supplied one does not cover the step: nothing is compared, a record
    carrying its own ``engine`` is reused, and a step that does run has its
    own check refuse with the real message."""
    if engine is None:
        return None
    compiled = loaded.inner[name].compiled
    try:
        check_engine(compiled, engine.effective_capabilities)
    except ValidationError:
        return None
    return engine.name


# --------------------------------------------------------------------------- #
# protocol steps
# --------------------------------------------------------------------------- #


def _run_protocol_step(
    name: str,
    step: ProtocolStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engine: Engine | None,
    implementation: Mapping[str, Any],
    *,
    ledger: _ControlLedger | None = None,
    selection: Sequence[int] | None = None,
    publisher: Publisher = SOLO,
) -> dict[str, Any]:
    """One protocol step's attempt (§8). ``selection`` — a fanned-out child's
    point indices (§2.9) — is the run context's ``points``, as
    ``run_protocol``'s ``--points`` is; the engine enumerates the steps and
    reads them by index and the document digest is untouched, so a child's
    artifacts stamp as members of the whole campaign. ``None`` runs every
    point. The digests and coordinates the record carries are the ones the
    engine signed (``result.steps``). ``publisher`` is this process's place in
    a launched world (``docs/model_parallelism.md`` §3, §9): it rides on the
    run context, and its word is the record's ``execution.parallel.launcher``."""
    inner, engine, run = _protocol_run(
        name, step, loaded, run_env, step_dir, engine, publisher, selection
    )
    result = engine.execute(inner, run)
    # one signed step per index; under data parallelism the campaign's, on
    # every rank
    check_steps_signed(result, run.indices(point_count(inner.axes)), engine)
    # the provenance units (§7), as the engine signed them, in run order
    digests = [step_record.digest for step_record in result.steps]
    coords = [dict(step_record.coords) for step_record in result.steps]
    record: dict[str, Any] = {
        "type": "intervention_protocol",
        "status": "completed",
        "identity": inner.digests.document,
        "implementation": dict(implementation),  # the code that ran it (§7)
        "document": step.document,
        "engine": engine.name,
        "document_digest": inner.digests.document,  # fully resolved (§7)
        "points": len(digests),
        "point_digests": list(digests),  # the provenance units (§7)
        # each point's coordinates (axis id → value), aligned with
        # `point_digests`: what a fan-out's join places a metric row by (§2.9)
        "coords": coords,
        # the sweep axes a downstream script groups by (§6)
        "axes": [axis.id for axis in inner.axes],
        "files": sorted(result.files),
        # the row bounds this step ran under — the engine's, overridden by the
        # step's own `execution` block; declared before execution, `null` when
        # unbounded (IM spec §8) — and the one recorder of them, the same block
        # `protocol.json` carries for a document run; execution, so it enters
        # no digest and no stamp
        "execution": execution_record(engine, run, launcher=publisher.launcher),
        # the forward groups the engine actually ran for this step (§4.3,
        # §8) — a record field, never an identity field: `--resume` compares
        # `identity`, `implementation` and the content digests, not this
        "forwards": result.forwards,
    }
    # the bounds the step measured rather than authored (IM spec §8): the
    # numbers to pin in the step's `execution` block to reproduce the step
    record["execution"].update(measured_bounds(record["execution"], result.summaries))
    # the commit each model's revision resolved to, the list a document
    # run's receipt carries (IM spec §8): execution, never identity
    if models := model_records(result.summaries):
        record[MODELS_KEY] = models
    # the controls layer (§2.2, §8) — like `reduction`, recorded only when
    # authored: the step's own declaration and its waivers, and the statuses
    # it inherits from every control upstream of it, joined by coordinates
    if step.waive is not None:
        record["waive"] = {kind: dict(w) for kind, w in step.waive.items()}
    if ledger is not None:
        if step.control is not None:
            record["control"] = ledger.declare(
                name,
                step,
                loaded,
                digests,
                coords,
                identity={
                    "document_digest": inner.digests.document,
                    "tree_digest": implementation["tree_digest"],
                    "engine": engine.name,
                },
            )
        inherited = ledger.inherit(name, loaded, digests, coords, inner.tree)
        if inherited is not None:
            record["controls"] = inherited
    return record


def _protocol_run(
    name: str,
    step: ProtocolStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engine: Engine | None,
    publisher: Publisher,
    selection: Sequence[int] | None,
) -> tuple[CompiledProtocol, Engine, RunContext]:
    """Compile, route and build a protocol step's run context — the half of
    the step every rank of a launched world runs identically
    (``docs/model_parallelism.md`` §3): the same document, the same ``set``,
    the same points, so the engines' collectives meet; a refusal here is the
    same on every rank."""
    inner = _compile_step(loaded, step, run_env)
    # the rules that needed the engine (IM spec §5 rules 13, 30) against the
    # run's one engine — before it loads a model, as `run_protocol`
    chosen = route_engine(inner, engine)
    # the engine geometry's document rules, before any weights and on every
    # rank alike, as `run_protocol` decides them (docs/model_parallelism.md
    # §8.3, §8.4): a rank that refused alone would leave the others waiting
    # in a collective
    check_parallel(inner, chosen, env=run_env)
    if loaded.inner_digest_kind.get(name) != "campaign":
        # a step-dependent document compiled at load against a deferring
        # store, so `check_tokenization` left it out: its answers resolve
        # now, with the earlier steps' outputs in place, before the engine
        # is handed it (a static one was resolved before step 1)
        with _in_step(name, step.document):
            resolve_answers(
                inner, env=run_env, tokenizers=tokenizer_service(run_env, engine=chosen)
            )
    n_points = point_count(inner.axes)
    indices = range(n_points) if selection is None else tuple(selection)
    if any(i < 0 or i >= n_points for i in indices):
        raise ProtocolError(
            "P2",
            f"step {name!r}: its shard selects point indices {list(indices)} of a "
            f"document that compiled {n_points} point(s) — the "
            "run-time compile and the load-time expansion disagree",
        )
    run = RunContext(
        output_dir=step_dir,
        env=run_env,
        points=tuple(indices),
        # the step's own row bounds (§2.2) — execution, so they ride on the
        # run context and never on the compiled document
        execution=step.execution,
        # the step's record is its receipt and this runner's stream is its
        # stream (§4.3): the engine records nothing beside the outputs
        record=False,
        # this process's place in a launched world (docs/model_parallelism.md
        # §3): the engine publishes through it, so one rank writes the step
        publisher=publisher,
    )
    return inner, chosen, run


def _compile_step(
    loaded: LoadedWorkflow, step: ProtocolStep, run_env: ResolutionEnv
) -> CompiledProtocol:
    """A protocol step's document compiled for execution. `_protocol_run`
    runs it, and `--resume` compares its digest with a record
    (`_run_tree_digest`), so the two cannot compile the step differently."""
    # The one compiler (IM spec §9), with the step's inputs: the document, its
    # own directory for relative artifact paths, the step's `set` as the
    # overrides (section-rooted, IM spec §1), and the run-tree overlay
    # as the artifact store — real resolution now, since earlier steps' outputs
    # exist. Validation compiled the same document through the same function
    # against the deferring store, so the two cannot resolve it differently.
    return compile_protocol(
        (loaded.workflow_dir / step.document).resolve(),
        env=run_env,
        overrides=step.set,
        point_cap=step.max_points if step.max_points is not None else DEFAULT_POINT_CAP,
    )


def check_tokenization(
    loaded: LoadedWorkflow,
    env: ResolutionEnv,
    *,
    engine: Engine | None = None,
    positions: bool = False,
    steps: Collection[str] | None = None,
) -> dict[str, AnswerCheck]:
    """Resolve, with each model's tokenizer, the metric answers of every
    inner document that compiles at load (IM spec §2.10;
    [`resolve_answers`][causalab.protocol.pipeline.resolve_answers]), and
    return the steps whose documents resolved, in schedule order, each with
    its document's [`AnswerCheck`][causalab.protocol.pipeline.AnswerCheck]:
    the metrics whose answers resolved and those left to the score.

    A static document (``inner_digest_kind`` ``"campaign"``) names no earlier
    step's output, so its tables and fields are the ones its step will run.
    A step-dependent document compiled against a deferring store, whose
    placeholders are not what its step runs; it is resolved at its own step,
    before the engine is handed it, and is never in the result. A document
    several steps share (the children of a fan-out) is resolved once, and
    every step that shares it is in the result. A fan-out parent is its
    children's join and scores nothing, so it is left out. ``steps`` limits the check to
    those steps (every step when ``None``): [`run_workflow`][] checks every
    step before step 1, or under ``--resume`` each attempted step at its
    turn. ``validate --tokenizer`` calls this with ``positions``, which
    resolves each static document's token positions first
    ([`resolve_positions`][causalab.protocol.pipeline.resolve_positions]).
    The tokenizer is ``engine``'s caller-owned bundle's when it holds one,
    else ``env``'s service ([`tokenizer_service`][causalab.protocol.pipeline.tokenizer_service]),
    one load per model for the call.

    Raises:
        ProtocolError: the pass's refusal, at ``steps.<name>`` and naming the
            document.
    """
    tokenizers = tokenizer_service(env, engine=engine)
    # a fan-out parent is its children's join (§2.9): it hands no engine the
    # document, and its children resolve that document
    static = [
        name
        for name in loaded.order
        if name in loaded.inner
        and loaded.inner_digest_kind.get(name) == "campaign"
        and getattr(loaded.document.steps[name], "fan_out", None) is None
    ]
    resolved: dict[str, AnswerCheck] = {}
    for name in static:
        if steps is not None and name not in steps:
            continue
        inner = loaded.inner[name]
        digest = inner.compiled.digests.document
        if digest in resolved:
            continue
        step = loaded.document.steps[name]
        with _in_step(name, getattr(step, "document", None)):
            compiled = inner.compiled
            if positions:
                compiled = resolve_positions(compiled, env=env, tokenizers=tokenizers)
            resolved[digest] = resolve_answers(compiled, env=env, tokenizers=tokenizers)
    return {
        name: resolved[digest]
        for name in static
        if (digest := loaded.inner[name].compiled.digests.document) in resolved
    }


@contextlib.contextmanager
def _in_step(name: str, document: str | None) -> Iterator[None]:
    """Re-raise a protocol refusal as the step's: at ``steps.<name>``,
    naming the document, with the refusal's own code, path and text."""
    try:
        yield
    except ProtocolError as err:
        at = f"at {err.path}: " if err.path else ""
        raise ProtocolError(
            err.code, f"document {document!r}: {at}{err.message}", path=f"steps.{name}"
        ) from err


# --------------------------------------------------------------------------- #
# script steps
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ScriptCall:
    """One script step's call, resolved before the script runs (§3, §4).

    [`script_call`][] builds it. The runner calls the script's
    ``main(inputs, outputs)``, in process or across the isolation boundary.
    [`stamp`][causalab.workflow.runner.ScriptCall.stamp] then finishes the
    outputs, and [`verify_output`][] checks each one before the step is
    published.
    """

    #: The step's name in the workflow.
    name: str
    #: The step as the workflow loaded it.
    step: ScriptStep
    #: What ``main`` receives as ``inputs``: each literal, each resolved
    #: reference, and the ``reduction`` and ``control`` inputs when the step
    #: has them.
    inputs: dict[str, Any]
    #: What ``main`` receives as ``outputs``: the path of each declared slot
    #: in the step's directory.
    outputs: dict[str, Path]
    #: The identity of every tensor input. A ``.safetensors`` output inherits
    #: the fields they agree on.
    identities: tuple[Mapping[str, Any], ...]
    #: The sha256 of every referenced file by slot, taken before the script
    #: reads it (§7, §8).
    input_digests: dict[str, str]

    def stamp(self) -> None:
        """Refuse a declared output the script did not write, and stamp every
        ``.safetensors`` output with the identity its tensor inputs agree on.

        Stamping rewrites a bundle, so it comes before the outputs are
        verified and digested. The digest is then of the bytes that get
        published.

        Raises:
            ProtocolError: ``P2`` when a declared output does not exist.
        """
        from causalab.io import step_io

        identity = step_io.inherited_identity(self.identities)
        identity["engine"] = SCRIPT_ENGINE
        for slot, decl in self.step.outputs.items():
            target = self.outputs[slot]
            what = f"step {self.name!r}: output {slot!r} ({decl.file})"
            if not target.is_file():
                raise ProtocolError(
                    "P2",
                    f"{what} was not written — a script step must create every "
                    "output it declares",
                )
            if decl.suffix == ".safetensors":
                step_io.stamp_tensor(target, identity, what=what)


def script_call(
    name: str, loaded: LoadedWorkflow, run_root: Path, step_dir: Path
) -> ScriptCall:
    """Resolve a script step's inputs and outputs as the runner hands them to
    the script (§3, §4).

    A reference becomes a path, and a ``key``, ``slot`` or ``entry`` selector
    reads through it. An authored ``reduction`` travels as the ``reduction``
    input (§2.6). A certifying step gets the declaration it certifies as the
    ``control`` input (§2.2). The function is public so that a test can run
    a script on sample inputs through the runner's own resolution;
    ``tests/demos/test_papers.py`` does this for every paper package.

    Args:
        name: The script step's name in ``loaded``.
        loaded: The loaded workflow.
        run_root: The run tree. Each step's files sit under its name.
        step_dir: Where the script writes its outputs.

    Returns:
        The resolved call.

    Raises:
        ProtocolError: ``P2`` when ``name`` is not a script step, a
            referenced file does not exist, or a selector finds nothing in it.
    """
    step = loaded.document.steps.get(name)
    if not isinstance(step, ScriptStep):
        raise ProtocolError("P2", f"step {name!r} is not a script step")
    resolved, tensor_identities, input_digests = _resolve_inputs(
        name, step, run_root, loaded.workflow_dir
    )
    if step.reduction is not None:
        # the authored reduction contract travels the channel inputs already
        # travel (§2.6): the script reads it as `inputs["reduction"]`, in
        # process or across the isolation boundary alike. Rule 12 has refused
        # any authored input of the same name, so nothing is shadowed here.
        resolved[REDUCTION_INPUT] = json.loads(json.dumps(dict(step.reduction)))
    subject = certifier_subject(loaded.document.steps, name)
    if subject is not None:
        # a certifying step gets the declaration it certifies the same way
        # (§2.2); rule 14 has refused an authored input of this name
        control_step = loaded.document.steps[subject]
        declaration = (
            dict(control_step.control or {})
            if isinstance(control_step, ProtocolStep)
            else {}
        )
        resolved[CONTROL_INPUT] = json.loads(
            json.dumps({"step": subject, **declaration})
        )
    return ScriptCall(
        name=name,
        step=step,
        inputs=resolved,
        outputs={slot: step_dir / decl.file for slot, decl in step.outputs.items()},
        identities=tuple(tensor_identities),
        input_digests=input_digests,
    )


def _run_script_step(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    step_dir: Path,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve inputs, run the script, verify and stamp its outputs (§4)."""
    call = script_call(name, loaded, run_root, step_dir)
    if step.runtime and step.runtime.get("isolate"):
        _run_isolated(name, step, loaded, call.inputs, call.outputs)
    else:
        _run_in_process(name, step, loaded, call.inputs, call.outputs)
    call.stamp()
    input_digests = call.input_digests

    return {
        "type": "script",
        "status": "completed",
        "identity": loaded.step_digests[name],
        "implementation": dict(implementation),  # the code that ran it (§7)
        "script": step.script,
        "script_sha256": step.script_sha256,
        # which sibling files the identity hashed beside a `{"path": …}` script
        # (§4.2, §7) — the keys the canonical entry carries, when it does
        **(
            {"closure": dict(step.closure), "closure_sha256": step.closure_sha256}
            if step.closure
            else {}
        ),
        "digest": loaded.step_digests[name],
        "is_deterministic": step.is_deterministic,
        "inputs": {
            key: (value.target if isinstance(value, Reference) else value)
            for key, value in step.inputs.items()
        },
        # Referenced file bytes, hashed before the script reads them. Step
        # references need the same check as external paths: the producer can
        # publish new bytes without changing this consumer's identity.
        **({"input_digests": input_digests} if input_digests else {}),
        "axes": [],  # a script step carries no sweep coordinates of its own
        "files": sorted(decl.file for decl in step.outputs.values()),
        **({"runtime": dict(step.runtime)} if step.runtime else {}),
        # the declaration a review reads from the tree (§2.6) — like
        # `runtime`, recorded only when authored
        **({"reduction": dict(step.reduction)} if step.reduction else {}),
    }


def _file_inputs(
    step: ScriptStep, run_root: Path, workflow_dir: Path
) -> dict[str, Path]:
    """Resolve each file reference by slot, before reading any selectors.

    Literal inputs are already covered by the step identity. References name
    either a published step file or an external path; both need byte checks.
    """
    out: dict[str, Path] = {}
    for slot, value in step.inputs.items():
        if not isinstance(value, Reference):
            continue
        if value.step is not None:
            out[slot] = run_root / value.step / str(value.file)
        else:
            candidate = Path(str(value.path))
            out[slot] = (
                candidate
                if candidate.is_absolute()
                else (workflow_dir / candidate).resolve()
            )
    return out


def _resolve_inputs(
    name: str, step: ScriptStep, run_root: Path, workflow_dir: Path
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, str]]:
    """The §3 grammar, resolved: a locator becomes a path, a selector reads
    through it. Also returns the identity of every tensor input, which is what
    a safetensors output inherits (§4), and the sha256 of every referenced
    file by slot — digested here, before the script runs, so the record
    names the bytes the step read and not whatever the file holds after (§7,
    §8). A relative ``path`` resolves against ``workflow_dir``, exactly as
    rule 4 checked it at load."""
    from causalab.io import step_io

    resolved: dict[str, Any] = {}
    identities: list[Mapping[str, Any]] = []
    input_digests: dict[str, str] = {}
    file_inputs = _file_inputs(step, run_root, workflow_dir)
    for slot, value in step.inputs.items():
        if not isinstance(value, Reference):
            resolved[slot] = value
            continue
        what = f"step {name!r}: input {slot!r} ({value.target})"
        target = file_inputs[slot]
        if not target.is_file():
            raise ProtocolError("P2", f"{what} does not exist at {str(target)!r}")
        input_digests[slot] = _sha256(target)
        if value.key is not None:
            values = step_io.read_values(target)
            if value.key not in values:
                raise ProtocolError(
                    "P2",
                    f"{what}: no key {value.key!r} in {target.name} "
                    f"(has {sorted(values)})",
                )
            resolved[slot] = values[value.key]
            continue
        if value.entry is not None or value.slot is not None:
            tensor, identity = step_io.read_tensor_with_identity(
                target, slot=value.slot, entry=value.entry, what=what
            )
            resolved[slot] = tensor
            identities.append(identity)
            continue
        resolved[slot] = target
        if target.suffix == ".safetensors":
            # an unselected bundle is handed over as a path, but its identity
            # still flows: a fit over one harvest is bound to that harvest
            try:
                _, identity = step_io.read_tensor_with_identity(target, what=what)
                identities.append(identity)
            except ProtocolError:
                pass  # multi-slot bundle: nothing unambiguous to inherit
    return resolved, identities, input_digests


def _verify_outputs(
    name: str, step: Any, attempt_dir: Path, files: Sequence[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Every file the step's record lists, verified for its format and
    content-digested (§8): ``({file: sha256}, {file: check})``.

    For a script step the files are its declared outputs, each checked against
    its declaration; a protocol step's files are what its engine reported
    having written, checked by format alone."""
    declared = (
        {decl.file: decl for decl in step.outputs.values()}
        if isinstance(step, ScriptStep)
        else {}
    )
    digests: dict[str, str] = {}
    checks: dict[str, str] = {}
    for rel in sorted(files):
        target = attempt_dir / rel
        what = f"step {name!r}: output {rel!r}"
        checks[rel] = verify_output(target, declared.get(rel), what=what)
        digests[rel] = _sha256(target)
    return digests, checks


def verify_output(
    target: Path, decl: OutputDecl | None = None, *, what: str | None = None
) -> str:
    """Check one step output against its format and its declaration (§2.5, §8).

    The runner calls this on every file the step's record lists, before it
    publishes the step: a script step's declared outputs, and the files a
    protocol step's engine reports. A file a script writes without declaring
    it is published with the step and is not checked. The function is public
    so that a test can hold a script's outputs to the same check without a
    run: ``tests/demos/test_papers.py`` runs every paper package's script
    steps through it.

    Every format must exist and be non-empty. A ``.json`` output must parse.
    Every declared key must be in a values object, and every declared column
    must be in the first row of a table. A ``.safetensors`` header must
    promise the file's own length, and a ``.png`` or ``.pdf`` must start
    with its signature. An ``.html`` output has no structure to check and is
    recorded as ``non-empty``, so the weakness of that check is visible in
    the step record.

    The check reads the real file, so it runs after the script. A consuming
    step was validated against the declaration at load, which is why a
    missing declared column refuses here. The check has three limits. An
    empty table satisfies any column declaration. Only the first row of a
    table is read, so a later row may lack a declared column. Column dtypes
    are not checked.

    Args:
        target: The written file.
        decl: The step's declaration of the file. ``None`` checks the format
            alone, as for a file a protocol step's engine reports.
        what: How a refusal names the file; ``output '<file name>'`` by
            default.

    Returns:
        The name of the check that passed: ``json``, ``json-values``,
        ``json-table``, ``safetensors-header``, ``png-signature``,
        ``pdf-signature`` or ``non-empty``.

    Raises:
        ProtocolError: ``P2`` when the file is missing or empty, does not
            parse under its format, or misses a declared key or column.
    """
    what = what if what is not None else f"output {target.name!r}"
    if not target.is_file():
        raise ProtocolError(
            "P2",
            f"{what} was not written"
            + (" — every declared output must exist" if decl is not None else ""),
        )
    if target.stat().st_size == 0:
        raise ProtocolError("P2", f"{what} is empty (0 bytes)")
    suffix = target.suffix
    if suffix == TABLE_SUFFIX:
        return _verify_json_output(target, decl, what)
    if suffix == ".safetensors":
        _verify_safetensors(target, what)
        return "safetensors-header"
    if suffix in _SIGNATURES:
        with target.open("rb") as handle:
            head = handle.read(len(_SIGNATURES[suffix]))
        if head != _SIGNATURES[suffix]:
            raise ProtocolError(
                "P2", f"{what} does not start with the {suffix} signature"
            )
        return f"{suffix[1:]}-signature"
    return "non-empty"  # .html and anything else: no structure to check


#: Magic bytes of the figure formats that have them (§2.5).
_SIGNATURES: dict[str, bytes] = {".png": b"\x89PNG\r\n\x1a\n", ".pdf": b"%PDF"}


def _verify_json_output(target: Path, decl: OutputDecl | None, what: str) -> str:
    """A JSON output parses, and matches its declared shape when it has one
    (§2.3): declared ``keys`` against the values object written, declared
    ``columns`` against the rows written."""
    try:
        with target.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ProtocolError("P2", f"{what} is not valid JSON: {err}") from err
    if decl is None or (decl.keys is None and decl.columns is None):
        return "json"
    if decl.keys is not None:
        if not isinstance(payload, dict):
            raise ProtocolError(
                "P2",
                f"{what} declares keys, so it must be a values object (a JSON "
                "mapping of name to value), not a "
                f"{type(payload).__name__}",
            )
        missing = sorted(set(decl.keys) - set(payload))
        if missing:
            raise ProtocolError(
                "P2",
                f"{what}: declares keys {sorted(decl.keys)} but wrote "
                f"{sorted(payload)} — missing {missing}",
            )
        return "json-values"
    rows = read_table(target)  # a JSON array of row objects, or a P2
    if not rows:
        return "json-table"  # an empty table satisfies any column declaration
    present = set(rows[0])
    missing = sorted(set(decl.columns or {}) - present)
    if missing:
        raise ProtocolError(
            "P2",
            f"{what}: declares columns {sorted(decl.columns or {})} but wrote "
            f"{sorted(present)} — missing {missing}",
        )
    return "json-table"


def _verify_safetensors(target: Path, what: str) -> None:
    """The bundle's header parses and its data section is exactly as long as
    the header says — a truncated or padded file fails here, with no tensor
    library involved (format: https://github.com/huggingface/safetensors)."""
    size = target.stat().st_size
    with target.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ProtocolError("P2", f"{what} is not a safetensors file (no header)")
        (header_len,) = struct.unpack("<Q", prefix)
        raw = handle.read(header_len)
    if len(raw) != header_len:
        raise ProtocolError("P2", f"{what} is truncated inside its safetensors header")
    try:
        header = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ProtocolError(
            "P2", f"{what} has an unreadable safetensors header: {err}"
        ) from err
    if not isinstance(header, dict):
        raise ProtocolError("P2", f"{what} has a malformed safetensors header")
    data_end = 0
    for key, spec in header.items():
        if key == "__metadata__" or not isinstance(spec, dict):
            continue
        offsets = spec.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            data_end = max(data_end, int(offsets[1]))
    if size != 8 + header_len + data_end:
        raise ProtocolError(
            "P2",
            f"{what}: safetensors header promises {8 + header_len + data_end} "
            f"bytes, the file has {size}",
        )


def _sha256(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_in_process(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    """Import the script and call ``main`` (§4).

    The import happens *here*, not at load: ``validate``/``digest`` must not
    pull a script's dependencies in (§4.2). By the time we are running, the
    process is already committed to executing."""
    import importlib.util

    target = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    spec = importlib.util.spec_from_file_location(f"_causalab_step_{name}", target)
    if spec is None or spec.loader is None:
        raise ProtocolError("P2", f"step {name!r}: cannot import {str(target)!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise ProtocolError(
            "P2", f"step {name!r}: {step.script!r} has no callable 'main'"
        )
    main(inputs, dict(outputs))


def _run_isolated(
    name: str,
    step: ScriptStep,
    loaded: LoadedWorkflow,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> None:
    """Run the script in a subprocess with its own dependency set (§4.1).

    Tensor-valued inputs cannot cross the process boundary, so an isolated step
    takes its tensors as *paths* — which means it must not use an ``entry``
    selector. Refused here rather than silently pickling a tensor into JSON."""
    runtime = dict(step.runtime or {})
    payload: dict[str, Any] = {}
    for slot, value in inputs.items():
        if isinstance(value, Path):
            payload[slot] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
            payload[slot] = value
        else:
            raise ProtocolError(
                "P2",
                f"step {name!r}: input {slot!r} is a {type(value).__name__}, "
                "which cannot cross a process boundary — an isolated step takes "
                "tensors as paths, so drop the 'entry'/'slot' selector and read "
                "the bundle inside the script",
            )
    target = resolve_script(step, loaded.workflow_dir, f"steps.{name}.script")
    request = {
        "script": str(target),
        "inputs": payload,
        "outputs": {slot: str(path) for slot, path in outputs.items()},
    }
    # The runner's own environment with `deps` layered on top: `--python` names
    # the interpreter this process runs under, `--no-project` keeps uv from
    # re-syncing whatever project the working directory happens to be in, and
    # `--with` builds an ephemeral overlay whose site-packages precede the
    # interpreter's — the declared deps win, the runner's environment is not
    # modified, and `causalab` is imported from the same bytes that are running
    # here. That holds for an editable checkout and an installed wheel alike;
    # the former `uv run` from the package's parent directory needed a
    # pyproject there, which only a checkout has.
    command = ["uv", "run", "--no-project", "--python", _python()]
    for dep in runtime.get("deps", ()):
        command += ["--with", str(dep)]
    command += ["python", "-m", "causalab.workflow.isolate"]

    environ = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "VIRTUAL_ENV")
        if key in os.environ
    }
    for passthrough in runtime.get("env", ()):
        if str(passthrough) in os.environ:
            environ[str(passthrough)] = os.environ[str(passthrough)]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=environ,
        )
    except FileNotFoundError as err:
        raise ProtocolError(
            "P2",
            f"step {name!r}: isolation needs 'uv' on PATH ({err})",
        ) from err
    if completed.returncode != 0:
        raise ScriptFailure(
            f"step {name!r}: isolated script failed (exit "
            f"{completed.returncode})\n{completed.stderr.strip()[-STDERR_TAIL_BYTES:]}",
            stderr=completed.stderr,
        )


def _python() -> str:
    return sys.executable
