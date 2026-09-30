"""Write workflow manifests and manage step attempts.

Each attempt has a record under ``.attempts``. Publication verifies its products
and renames the complete attempt into the step directory. Superseded attempts
retain their records. The manifest reports the status of every scheduled step."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from causalab.io.step_record import SIDECAR

__all__ = [
    "ATTEMPTS_DIR",
    "ATTEMPT_RECORD",
    "CHILD_SEPARATOR",
    "DISPLACED_SUFFIX",
    "DISPOSITIONS",
    "MANIFEST",
    "RETAINED_FAILED_ATTEMPTS",
    "SKIPPED_BY_FIELDS",
    "STDERR_TAIL_BYTES",
    "STEP_STATUSES",
    "SUPERSEDED_SUFFIX",
    "AttemptFailure",
    "StepStatus",
    "classify_unreached",
    "new_attempt_dir",
    "propagates",
    "skipped_via",
    "prune_attempts",
    "publish_attempt",
    "remove_if_empty",
    "displaced_attempt",
    "restore_displaced",
    "retain_superseded",
    "superseded_units",
    "write_attempt_record",
    "write_manifest",
]

#: The run manifest, beside the step directories (§1.1).
MANIFEST = "workflow.json"
#: Where attempts live inside the run tree (§1.1). A step name matches
#: ``[A-Za-z0-9_-]+`` (rule 3), so this leading-dot name can never be a step.
ATTEMPTS_DIR = ".attempts"
#: The bounded failure metadata a failed attempt keeps (§8).
ATTEMPT_RECORD = "attempt.json"
#: What joins a fanned-out step's name to a child's index — ``<step>@<i>``
#: (spec §1.1, §2.9). Outside rule 3's authored alphabet, so a child can never
#: collide with an authored step, and neither a leading dot nor an underscore,
#: so nothing here or in ``events.py`` reserves it twice. Published by
#: ``causalab.workflow.fan_out`` as ``CHILD_SEPARATOR``; defined here because
#: the manifest's skip rule below reads it.
CHILD_SEPARATOR = "@"
#: A previously published step directory, moved aside inside the step's
#: attempts directory for the instant between the two renames of a publish.
DISPLACED_SUFFIX = ".previous"
#: Where a superseded unit is retained (§8): ``.attempts/<step>/<n>.superseded/``
#: — the prior complete unit, its record marked, never deleted. Not a digit
#: name, so it is outside the attempt counter, the pruning and the restore.
SUPERSEDED_SUFFIX = ".superseded"
#: What a step record's ``disposition`` may say (§8), closed: ``candidate`` —
#: an attempt's record before its publish; ``accepted`` — the published unit,
#: the one a reader beside its files finds and analysis accepts by default;
#: ``inadmissible`` — a failed attempt's ``attempt.json``; ``superseded`` — a
#: unit a rerun displaced, retained under [`SUPERSEDED_SUFFIX`][].
DISPOSITIONS: tuple[str, ...] = ("candidate", "accepted", "inadmissible", "superseded")
#: How many failed attempts a step keeps; older ones are deleted, partial
#: outputs included.
RETAINED_FAILED_ATTEMPTS = 3
#: How much of an isolated script's stderr a failed attempt retains.
STDERR_TAIL_BYTES = 64 * 1024

#: The six words a step's status can be. Closed: the spec's §8 table lists
#: exactly these, and the census test holds the two together. ``skipped`` is
#: a decision's word (§2.8), never a failure's: the manifest entry names the
#: decision that skipped the step.
STEP_STATUSES: tuple[str, ...] = (
    "completed",
    "reused",
    "failed",
    "blocked",
    "pending",
    "skipped",
)
StepStatus = Literal["completed", "reused", "failed", "blocked", "pending", "skipped"]

#: A skipped step's manifest entry names the decision that skipped it (§8):
#: the conditional, the producer whose ``decision.json`` it read, what that
#: record said, the evidence identity binding it to the numbers, and — for a
#: step skipped because a step it depends on was — the steps it followed.
#: Manifest vocabulary, like [`STEP_STATUSES`][]: the two emitters — the
#: runner's reached skip (``conditional.skipped_entry``) and this module's
#: unreached skip ([`classify_unreached`][]) — write the same six keys, absent
#: ones ``None``, so a reader never meets two shapes in one manifest.
SKIPPED_BY_FIELDS: tuple[str, ...] = (
    "conditional",
    "decision_step",
    "decision_type",
    "outcome",
    "evidence_identity",
    "transitive_from",
)


@dataclasses.dataclass(frozen=True)
class AttemptFailure:
    """What a failed attempt records about why (§8): the exception, and for an
    isolated script the tail of its stderr — bounded, so a chatty script cannot
    turn failure metadata into a log archive."""

    error_type: str
    message: str
    stderr_tail: str | None = None

    @classmethod
    def from_exception(cls, err: BaseException, stderr: str | None) -> AttemptFailure:
        tail = None
        if stderr is not None:
            encoded = stderr.encode("utf-8", errors="replace")[-STDERR_TAIL_BYTES:]
            tail = encoded.decode("utf-8", errors="replace")
        return cls(
            error_type=type(err).__name__,
            message=str(err)[:STDERR_TAIL_BYTES],
            stderr_tail=tail,
        )


# --------------------------------------------------------------------------- #
# attempt directories
# --------------------------------------------------------------------------- #


def _attempt_dirs(attempts_root: Path) -> list[Path]:
    """Existing attempt directories of one step, oldest first."""
    if not attempts_root.is_dir():
        return []
    return sorted(
        (p for p in attempts_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def new_attempt_dir(run_root: Path, step: str) -> Path:
    """Create ``<run_root>/.attempts/<step>/<id>/`` and return it.

    The id is a **monotonic counter** (zero-padded so a listing sorts): the
    runner is the only writer of its run tree, so a counter is unambiguous,
    and it reads better in a failure report than a timestamp+pid. It also
    makes "the last three attempts" a plain sort."""
    attempts_root = run_root / ATTEMPTS_DIR / step
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = _attempt_dirs(attempts_root)
    next_id = int(existing[-1].name) + 1 if existing else 1
    attempt_dir = attempts_root / f"{next_id:04d}"
    attempt_dir.mkdir()
    return attempt_dir


def prune_attempts(run_root: Path, step: str, *, keep: int) -> None:
    """Keep the newest ``keep`` attempt directories of ``step``; delete the
    rest, partial outputs and all (§8, bounded failure metadata)."""
    existing = _attempt_dirs(run_root / ATTEMPTS_DIR / step)
    for stale in existing[: len(existing) - keep]:
        shutil.rmtree(stale)


def remove_if_empty(path: Path) -> None:
    """Remove ``path`` when it is an empty directory — so a clean run leaves
    no ``.attempts/`` behind."""
    try:
        path.rmdir()
    except OSError:
        pass  # not empty, or not there: either is fine


def write_attempt_record(
    attempt_dir: Path,
    *,
    step: str,
    started: float,
    failure: AttemptFailure,
    declared: Mapping[str, str],
) -> None:
    """``attempt.json`` for a failed attempt: when, what died, what it had
    written (§8). Never raises — metadata about a failure must not hide the
    failure."""
    present = sorted(rel for rel in declared if (attempt_dir / rel).is_file())
    record: dict[str, Any] = {
        "step": step,
        "attempt": attempt_dir.name,
        "status": "failed",
        "disposition": "inadmissible",
        "started": _iso(started),
        "ended": _iso(time.time()),
        "pid": os.getpid(),
        "error": {"type": failure.error_type, "message": failure.message},
        "stderr_tail": failure.stderr_tail,
        "outputs_written": present,
        "outputs_declared": sorted(declared),
    }
    try:
        (attempt_dir / ATTEMPT_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    except OSError as err:  # pragma: no cover — the disk is the failure here
        print(f"could not write {ATTEMPT_RECORD}: {err}", file=sys.stderr)


def _iso(stamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


# --------------------------------------------------------------------------- #
# publish
# --------------------------------------------------------------------------- #


def publish_attempt(
    attempt_dir: Path, step_dir: Path, *, between: Any = None
) -> Path | None:
    """Make a verified attempt the published step directory by rename.

    ``attempt_dir`` and ``step_dir`` sit in the same run tree, so
    `os.replace` is one atomic ``rename(2)``. A stale ``step_dir`` from
    a previous complete unit is moved aside *inside the attempts directory*
    first — until the new unit is in place it is still a complete unit, and
    [`restore_displaced`][] puts it back if the process dies between the two
    renames. It is **never deleted** (§8, supersession preserves): the
    displaced path is returned for the runner to retain through
    [`retain_superseded`][] once the publish is narrated, and a displaced
    unit the runner never got to is retained by the next run's
    [`restore_displaced`][].

    ``between`` is the runner's fault-injection seam for the instant between
    the two renames.
    """
    displaced: Path | None = None
    if step_dir.exists():
        displaced = attempt_dir.parent / f"{attempt_dir.name}{DISPLACED_SUFFIX}"
        os.replace(step_dir, displaced)
    if between is not None:
        between()
    os.replace(attempt_dir, step_dir)
    return displaced


def _superseded_dirs(attempts_root: Path) -> list[Path]:
    """The retained superseded units of one step, oldest first."""
    if not attempts_root.is_dir():
        return []
    cut = -len(SUPERSEDED_SUFFIX)
    return sorted(
        (
            p
            for p in attempts_root.iterdir()
            if p.is_dir()
            and p.name.endswith(SUPERSEDED_SUFFIX)
            and p.name[:cut].isdigit()
        ),
        key=lambda p: int(p.name[:cut]),
    )


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def displaced_attempt(displaced: Path) -> str | None:
    """The attempt id a displaced path carries (``0001.previous`` → ``0001``)."""
    name = displaced.name
    if name.endswith(DISPLACED_SUFFIX):
        return name[: -len(DISPLACED_SUFFIX)]
    return None


def retain_superseded(
    unit: Path, attempts_root: Path, *, superseded_by: Mapping[str, Any]
) -> Path:
    """Retain a complete unit as **superseded** (§8) — the unit a publish
    displaced (its ``.previous`` path), or a published step directory a skip
    took out of the run. The unit moves to ``<attempts_root>/<n>.superseded/``,
    ``n`` counting the step's retained units, and its ``_step.json`` is then
    rewritten with ``status: superseded``, ``disposition: superseded`` and the
    given ``superseded_by`` — for a publish, the replacing attempt, the
    identity of the record now published and the step directory's name; for
    a skip, the decision (``skipped_by``). The prior bytes are readable there
    for audit; nothing under ``.attempts/`` is addressable through the §3
    grammar, and a reader beside a published file (``read_sidecar``) finds
    only the ``accepted`` record."""
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = _superseded_dirs(attempts_root)
    cut = -len(SUPERSEDED_SUFFIX)
    next_id = int(existing[-1].name[:cut]) + 1 if existing else 1
    retained = attempts_root / f"{next_id:04d}{SUPERSEDED_SUFFIX}"
    os.replace(unit, retained)
    marked = {
        **_read_record(retained / SIDECAR),
        "status": "superseded",
        "disposition": "superseded",
        "superseded_by": dict(superseded_by),
    }
    (retained / SIDECAR).write_text(json.dumps(marked, indent=2) + "\n")
    return retained


def superseded_units(run_root: Path, step: str) -> list[str]:
    """The retained superseded units of ``step``, as paths relative to the run
    root, oldest first — what the manifest entry lists (§8)."""
    return [
        str(p.relative_to(run_root))
        for p in _superseded_dirs(run_root / ATTEMPTS_DIR / step)
    ]


def restore_displaced(run_root: Path, step: str, step_dir: Path) -> None:
    """Finish a publish that died after its first rename: if the new unit
    never landed, the previous complete unit is back where §3 references find
    it; if it did land, the displaced unit is retained as superseded
    ([`retain_superseded`][]) — either way nothing is lost.

    Only a displaced directory left by [`publish_attempt`][] qualifies."""
    attempts_root = run_root / ATTEMPTS_DIR / step
    if not attempts_root.is_dir():
        return
    displaced = sorted(attempts_root.glob(f"*{DISPLACED_SUFFIX}"))
    for candidate in displaced:
        if not candidate.is_dir():
            continue
        if not step_dir.exists():
            os.replace(candidate, step_dir)
        else:
            published = _read_record(step_dir / SIDECAR)
            retain_superseded(
                candidate,
                attempts_root,
                superseded_by={
                    "attempt": displaced_attempt(candidate),
                    "identity": published.get("identity"),
                    "published": step,
                },
            )


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #


def propagates(upstream: str, name: str, selective: frozenset[str]) -> bool:
    """Whether ``upstream``'s skip skips ``name`` (§2.8, §2.9): always, except
    from a child onto its own join when that join declared ``require:
    selected`` — such a join publishes the children a verdict left. A skip
    from any other upstream (an ``after`` step, an input's producer) skips a
    selective join like any step; and a selective join every child of which
    is skipped has nothing to publish and is skipped with them
    ([`skipped_via`][])."""
    return not (name in selective and upstream.startswith(f"{name}{CHILD_SEPARATOR}"))


def skipped_via(
    name: str,
    upstream: Sequence[str],
    is_skipped: Callable[[str], bool],
    selective: frozenset[str],
) -> list[str]:
    """The skipped upstream steps whose skip reaches ``name`` (§2.8, §2.9),
    sorted: every skipped upstream that [`propagates`][] — and, for a
    selective join, its children when **all** of them are skipped."""
    via = sorted(
        u for u in upstream if is_skipped(u) and propagates(u, name, selective)
    )
    if name in selective:
        children = [u for u in upstream if u.startswith(f"{name}{CHILD_SEPARATOR}")]
        if children and all(is_skipped(u) for u in children):
            via = sorted({*via, *children})
    return via


def classify_unreached(
    order: tuple[str, ...],
    dependencies: Mapping[str, tuple[str, ...]],
    steps: Mapping[str, Mapping[str, Any]],
    *,
    selective: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Status entries for every step the run did not get to (§8).

    ``skipped`` when any upstream step is ``skipped`` — a decision already
    took the step out of this run, and its dependents follow it
    (``skipped_by.transitive_from`` names them; §2.8); else ``blocked`` when
    any upstream step is ``failed`` or ``blocked`` — the step could not have
    run; else ``pending`` — the run stopped before reaching it. Walked in
    schedule order so a skip or a block propagates down a chain. ``selective``
    names the joins declaring ``require: selected`` (§2.9): a skipped child
    does not skip such a join ([`propagates`][]). An unreached skip carries
    every [`SKIPPED_BY_FIELDS`][] key, as a reached one does: the decision
    fields inherited from the first skipped upstream's ``skipped_by``,
    ``transitive_from`` the skipped upstreams."""
    entries: dict[str, dict[str, Any]] = {}
    statuses = {name: str(entry.get("status", "")) for name, entry in steps.items()}
    skipped_by: dict[str, Mapping[str, Any]] = {
        name: entry.get("skipped_by") or {}
        for name, entry in steps.items()
        if statuses[name] == "skipped"
    }
    for name in order:
        if name in steps:
            continue
        skipped_from = skipped_via(
            name,
            dependencies.get(name, ()),
            lambda upstream: statuses.get(upstream) == "skipped",
            selective,
        )
        if skipped_from:
            first = skipped_by.get(skipped_from[0], {})
            entries[name] = {
                "status": "skipped",
                "skipped_by": {
                    **{
                        field: first.get(field)
                        for field in SKIPPED_BY_FIELDS
                        if field != "transitive_from"
                    },
                    "transitive_from": skipped_from,
                },
            }
            skipped_by[name] = entries[name]["skipped_by"]
            statuses[name] = "skipped"
            continue
        blocked_by = sorted(
            upstream
            for upstream in dependencies.get(name, ())
            if statuses.get(upstream) in ("failed", "blocked")
        )
        if blocked_by:
            entries[name] = {"status": "blocked", "blocked_by": blocked_by}
            statuses[name] = "blocked"
        else:
            entries[name] = {"status": "pending"}
            statuses[name] = "pending"
    return entries


def write_manifest(
    run_root: Path, manifest: Mapping[str, Any], *, between: Any = None
) -> Path:
    """``workflow.json``, written to a sibling temp file and renamed into place
    so a reader never sees a half-written manifest. ``between`` is the
    fault-injection seam between the write and the rename."""
    target = run_root / MANIFEST
    temp = run_root / f"{MANIFEST}.tmp"
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    if between is not None:
        between()
    os.replace(temp, target)
    return target
