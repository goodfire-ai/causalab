"""A step is attempted, verified, then published by one rename; reuse trusts
content digests, never existence; the manifest is always written (spec §8).

The strong test here is the interruption one: a failure is injected at
**every** write boundary the runner has, and after ``--resume`` the run tree
must expose either the previous complete unit (reused, byte-identical) or no
reusable unit (a fresh, complete execution) — never a half-written step
directory, never a published directory with fewer files than declared. The
boundaries are enumerated here *and* counted against the runner's source, so a
new write boundary cannot be added without a recovery case.

The rest pins the fail-closed half (a truncated or in-place-overwritten output
is re-executed; a record without digests is not trusted), the always-written
manifest with its six-word status vocabulary (census against the spec's §8
table), the bounded failure metadata, and that valid work still
passes: a clean run followed by ``--resume`` reuses every step with identical
bytes and leaves no attempt directory behind. A rerun without ``--resume`` is
the supersession twin: the prior unit is retained byte for byte
under ``.attempts/<step>/0001.superseded/``, marked, and listed.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import typing
from pathlib import Path
from typing import Any

import pytest

from causalab.io.step_record import SIDECAR
from causalab.protocol.rules.errors import ProtocolError, ProtocolWarning
from causalab.io.tables import read_table
from causalab.workflow import manifest as mf
from causalab.workflow import runner
from causalab.workflow.document import OutputDecl, load_workflow
from causalab.workflow.runner import (
    WRITE_BOUNDARIES,
    ScriptFailure,
    _verify_safetensors,  # pyright: ignore[reportPrivateUsage]
    run_workflow,
    script_call,
    verify_output,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"

#: A valid 1×1 PNG signature plus a few bytes — enough for the signature check.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class InjectedFailure(RuntimeError):
    """What the fault-injection hook raises."""


# --------------------------------------------------------------------------- #
# the tiny three-step chain
# --------------------------------------------------------------------------- #

#: Every script writes its outputs one after another and calls the runner's
#: fault-injection seam *between* them — that is the one boundary
#: (``outputs_partial``) the runner itself cannot interpose, because a
#: script's writes are its own.
FIRST = """
import json
from pathlib import Path
from causalab.workflow import runner

def main(inputs, outputs):
    Path(outputs["a"]).write_text(json.dumps([{"n": 1}, {"n": 2}]))
    runner._boundary("outputs_partial", "first")
    Path(outputs["b"]).write_text(json.dumps({"k": 2}))
"""

SECOND = (
    """
import json
from pathlib import Path
from causalab.workflow import runner
from causalab.io.tables import read_table

def main(inputs, outputs):
    rows = read_table(inputs["a"])
    Path(outputs["c"]).write_text(json.dumps([{"total": sum(r["n"] for r in rows)}]))
    runner._boundary("outputs_partial", "second")
    Path(outputs["d"]).write_bytes(%r)
"""
    % PNG
)

THIRD = """
import json
from pathlib import Path
from causalab.workflow import runner

def main(inputs, outputs):
    Path(outputs["e"]).write_text(json.dumps({"k": inputs["k"], "seen": True}))
    runner._boundary("outputs_partial", "third")
    Path(outputs["f"]).write_text("<html></html>")
"""

RAISING = """
def main(inputs, outputs):
    raise RuntimeError("the step died")
"""

INTERRUPTED = """
def main(inputs, outputs):
    raise KeyboardInterrupt()
"""

#: ``{step: {relative file}}`` — what a complete unit of each step holds.
DECLARED = {
    "first": {"a.json", "b.json"},
    "second": {"c.json", "d.png"},
    "third": {"e.json", "f.html"},
}


def _document(*, aside: bool = False, scripts: dict[str, str] | None = None) -> dict:
    scripts = {"first": "first.py", "second": "second.py", "third": "third.py"} | (
        scripts or {}
    )
    steps: dict[str, Any] = {
        "first": {
            "type": "script",
            "script": {"path": f"scripts/{scripts['first']}"},
            "inputs": {"seed": {"value": 1}},
            "outputs": {
                "a": {"file": "a.json", "columns": {"n": "int64"}},
                "b": {"file": "b.json", "keys": {"k": 2}},
            },
        },
        "second": {
            "type": "script",
            "script": {"path": f"scripts/{scripts['second']}"},
            "inputs": {"a": {"step": "first", "file": "a.json"}},
            "outputs": {
                "c": {"file": "c.json", "columns": {"total": "int64"}},
                "d": {"file": "d.png"},
            },
        },
        "third": {
            "type": "script",
            "script": {"path": f"scripts/{scripts['third']}"},
            "inputs": {
                "k": {"step": "first", "file": "b.json", "key": "k"},
                "c": {"step": "second", "file": "c.json"},
            },
            "outputs": {
                "e": {"file": "e.json", "keys": {"k": 2, "seen": True}},
                "f": {"file": "f.html"},
            },
        },
    }
    if aside:
        # independent of the chain and listed last, so it is scheduled last
        steps["aside"] = {
            "type": "script",
            "script": {"path": "scripts/first.py"},
            "inputs": {"seed": {"value": 2}},
            "outputs": {
                "a": {"file": "a.json", "columns": {"n": "int64"}},
                "b": {"file": "b.json", "keys": {"k": 2}},
            },
        }
    return {"version": "1", "output_dir": "chain", "steps": steps}


@pytest.fixture()
def wf_dir(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "first.py").write_text(FIRST)
    (scripts / "second.py").write_text(SECOND)
    (scripts / "third.py").write_text(THIRD)
    (scripts / "raising.py").write_text(RAISING)
    (scripts / "interrupted.py").write_text(INTERRUPTED)
    return tmp_path


def _load(wf_dir: Path, env: Any, **kwargs: Any) -> Any:
    return load_workflow(_document(**kwargs), env, workflow_dir=wf_dir)


def _run(loaded: Any, env: Any, out: Path, *, resume: bool = False) -> dict[str, str]:
    result = run_workflow(loaded, env, out, None, resume=resume)
    return {name: entry["status"] for name, entry in result.manifest["steps"].items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(run_root: Path) -> dict[str, Any]:
    return json.loads((run_root / mf.MANIFEST).read_text())


def _statuses(run_root: Path) -> dict[str, str]:
    return {
        name: entry["status"] for name, entry in _manifest(run_root)["steps"].items()
    }


def _snapshot(run_root: Path) -> dict[str, dict[str, bytes]]:
    """``{step: {relative file: bytes}}`` for every published step directory."""
    out: dict[str, dict[str, bytes]] = {}
    for step in DECLARED:
        step_dir = run_root / step
        if step_dir.is_dir():
            out[step] = {
                str(p.relative_to(step_dir)): p.read_bytes()
                for p in step_dir.rglob("*")
                if p.is_file()
            }
    return out


def _record_digests_match(step_dir: Path) -> bool:
    record = json.loads((step_dir / SIDECAR).read_text())
    return all(
        (step_dir / rel).is_file() and _sha256(step_dir / rel) == digest
        for rel, digest in record["digests"].items()
    )


def _assert_complete_unit(step: str, step_dir: Path) -> None:
    """A published step directory holds exactly its declared files plus the
    record, and the record's digests match the bytes."""
    assert (step_dir / SIDECAR).is_file(), f"{step}: published without a record"
    record = json.loads((step_dir / SIDECAR).read_text())
    files = {p.name for p in step_dir.iterdir()}
    assert files == DECLARED[step] | {SIDECAR}, f"{step}: published {files}"
    assert set(record["files"]) == DECLARED[step]
    assert set(record["digests"]) == DECLARED[step]
    assert set(record["checks"]) == DECLARED[step]
    assert _record_digests_match(step_dir), f"{step}: digests do not match"


def _truncate(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:-1])


def _flip_a_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


# --------------------------------------------------------------------------- #
# T1 — interruption at every write boundary
# --------------------------------------------------------------------------- #

#: The runner's write boundaries, spelled out. The census below holds this
#: list equal to the runner's and to its hook sites, so adding a boundary
#: without adding it here — and so without a recovery case — fails.
BOUNDARIES = [
    "attempt_created",  # (a) the attempt dir exists, nothing written yet
    "outputs_partial",  # (b) some declared outputs written, not all
    "outputs_written",  # (c) every output written, none verified
    "verified",  # (d) verified and digested, no step record yet
    "recorded",  # (e) step record written, not published
    "displaced",  # (e') stale unit moved aside, new one not yet renamed in
    "committed",  # (f') published on disk, not yet narrated: the manifest is refused
    "published",  # (f) published and narrated, manifest not yet written
    "superseded",  # (f'') published and narrated, the displaced prior unit not yet retained
    "manifest",  # (g) manifest written to its temp file, not renamed in
]

#: ``fresh``: the interrupted run is the first into the tree. ``republish``: a
#: clean run exists but one output per step was truncated, so every step
#: re-executes and publishes *over* a stale unit — the only way ``displaced``
#: and ``superseded`` fire. ``committed`` is the one boundary where the interrupted run writes
#: no manifest at all (§4.3: memory says ``completed``, the stream ``pending``,
#: and the disagreement is refused), so in ``republish`` the previous manifest
#: must survive byte for byte.
CASES = [
    (scenario, boundary, step)
    for scenario in ("fresh", "republish")
    for boundary in BOUNDARIES
    for step in (["first", "second", "third"] if boundary != "manifest" else [None])
    if not (boundary in ("displaced", "superseded") and scenario == "fresh")
]


def test_the_boundary_list_is_a_census_of_the_runner():
    """The list above is the runner's, and every boundary except the one a
    script raises from inside its own writes is a ``_boundary(...)`` site in
    the runner's source."""
    assert tuple(BOUNDARIES) == WRITE_BOUNDARIES
    source = (REPO / "causalab" / "workflow" / "runner.py").read_text()
    sites = set(re.findall(r'_boundary\(\s*"([a-z_]+)"', source))
    assert sites, "no fault-injection sites found in runner.py"
    assert sites == set(WRITE_BOUNDARIES) - {"outputs_partial"}
    assert len(set(BOUNDARIES)) == len(BOUNDARIES)


@pytest.mark.parametrize(("scenario", "boundary", "step"), CASES)
def test_interruption_at_every_write_boundary_recovers(
    wf_dir, tmp_path, env, monkeypatch, scenario, boundary, step
):
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"

    manifest_before: bytes | None = None
    if scenario == "republish":
        assert set(_run(loaded, env, out).values()) == {"completed"}
        for name, files in DECLARED.items():
            _truncate(run_root / name / sorted(files)[0])
        manifest_before = (run_root / mf.MANIFEST).read_bytes()

    fired: list[tuple[str, str | None]] = []

    def inject(name: str, at: str | None) -> None:
        assert name in WRITE_BOUNDARIES, name
        if name == boundary and at == step:
            fired.append((name, at))
            raise InjectedFailure(f"{name} in {at}")

    monkeypatch.setattr(runner, "_boundary", inject)
    if boundary == "committed":
        # published on disk and `completed` in memory, but the stream has not
        # narrated it: the manifest is refused (§4.3) naming the step and both
        # words, with the injected failure chained as the cause
        with pytest.raises(
            ProtocolError, match=rf"'{step}'.*'completed'.*'pending'"
        ) as info:
            run_workflow(loaded, env, out, None)
        assert isinstance(info.value.__cause__, InjectedFailure)
    else:
        with pytest.raises(InjectedFailure):
            run_workflow(loaded, env, out, None)
    assert fired == [(boundary, step)], "the injection did not fire exactly once"
    monkeypatch.undo()

    # ---- the interrupted tree: no half-written published unit --------------
    for name in DECLARED:
        step_dir = run_root / name
        if step_dir.exists():
            record = json.loads((step_dir / SIDECAR).read_text())
            present = {p.name for p in step_dir.iterdir()}
            assert present == DECLARED[name] | {SIDECAR}, f"{name}: {present}"
            assert set(record["digests"]) == DECLARED[name]
    if boundary == "committed":
        # no manifest rather than a wrong one: nothing new on disk, the
        # previous manifest (if any) intact, no temp file, the unit published
        if manifest_before is None:
            assert not (run_root / mf.MANIFEST).exists(), "written disagreeing"
        else:
            assert (run_root / mf.MANIFEST).read_bytes() == manifest_before
        assert not (run_root / f"{mf.MANIFEST}.tmp").exists()
        _assert_complete_unit(step, run_root / step)
    elif boundary != "manifest":
        # the finally wrote a manifest classifying every step
        statuses = _statuses(run_root)
        assert set(statuses) == set(DECLARED)
        assert set(statuses.values()) <= set(mf.STEP_STATUSES)
        # `superseded` fires after the publish is narrated (§8): the step is
        # published and `completed`; only the prior unit's retention is undone
        expected = "completed" if boundary in ("published", "superseded") else "failed"
        assert statuses[step] == expected, statuses
        downstream = [
            n for n in loaded.order if loaded.order.index(n) > loaded.order.index(step)
        ]
        for name in downstream:
            assert statuses[name] == (
                "pending" if expected == "completed" else "blocked"
            )
    before = _snapshot(run_root)
    reusable_before = {
        name for name in before if _record_digests_match(run_root / name)
    }

    # ---- resume: the previous complete unit, or a fresh complete one ---------
    after = _run(loaded, env, out, resume=True)
    assert set(after.values()) <= {"completed", "reused"}, after
    for name in DECLARED:
        step_dir = run_root / name
        _assert_complete_unit(name, step_dir)
        if after[name] == "reused":
            assert name in reusable_before, f"{name} reused without a complete unit"
            assert _snapshot(run_root)[name] == before[name], f"{name} rewritten"
        else:
            assert name not in reusable_before, f"{name} re-executed a complete unit"
    if boundary in ("committed", "published", "superseded"):
        # the interrupted run published this unit; `--resume` found it
        assert after[step] == "reused", after
        assert (run_root / mf.MANIFEST).is_file()
    if scenario == "republish":
        # supersession preserves (§8): every stale unit was displaced once —
        # by the interrupted run or by the resume — and is retained exactly
        # once (the one interrupted at `superseded` by the resume run's
        # `restore_displaced`); nothing lingers under a `.previous` name
        for name in DECLARED:
            attempts = run_root / mf.ATTEMPTS_DIR / name
            retained = attempts / f"0001{mf.SUPERSEDED_SUFFIX}"
            assert retained.is_dir(), f"{name}: the displaced unit was not retained"
            # exactly one retained unit; a failed attempt of the interrupted
            # step may sit beside it (bounded failure metadata, unchanged)
            assert [
                p.name
                for p in attempts.iterdir()
                if p.name.endswith(mf.SUPERSEDED_SUFFIX)
            ] == [retained.name], name
            marked = json.loads((retained / SIDECAR).read_text())
            assert marked["status"] == "superseded"
            assert marked["disposition"] == "superseded"
    assert read_table(run_root / "second" / "c.json") == [{"total": 3}]
    assert json.loads((run_root / "third" / "e.json").read_text()) == {
        "k": 2,
        "seen": True,
    }
    assert set(_statuses(run_root).values()) <= {"completed", "reused"}
    # nothing of the interrupted attempt lingers where it could be mistaken for
    # a unit: no displaced copy, no attempt directory inside a published step,
    # no half-renamed manifest
    assert not list(run_root.rglob(f"*{mf.DISPLACED_SUFFIX}"))
    assert not any((run_root / name / mf.ATTEMPTS_DIR).exists() for name in DECLARED)
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()


# --------------------------------------------------------------------------- #
# T2 — a truncated output is not reused
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("damage", [_truncate, _flip_a_byte])
def test_a_damaged_output_is_re_executed_not_reused(wf_dir, tmp_path, env, damage):
    """Cut the last byte, or overwrite one byte keeping the size: either way
    the content digest disagrees with the record and the step runs again,
    publishing a verified unit. Steps downstream follow the identity rules as
    they stand (their own digest)."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    assert set(_run(loaded, env, out).values()) == {"completed"}
    intact = _snapshot(run_root)

    damage(run_root / "second" / "c.json")
    assert _run(loaded, env, out, resume=True) == {
        "first": "reused",
        "second": "completed",
        "third": "reused",
    }
    _assert_complete_unit("second", run_root / "second")
    assert _snapshot(run_root)["second"] == intact["second"], "not republished"
    assert read_table(run_root / "second" / "c.json") == [{"total": 3}]
    # supersession preserves (§8): the damaged unit is retained as the audit
    # trail of what was replaced — marked, never deleted; the reused steps
    # displaced nothing and keep no attempts
    retained = run_root / mf.ATTEMPTS_DIR / "second" / f"0001{mf.SUPERSEDED_SUFFIX}"
    assert retained.is_dir()
    assert (retained / "c.json").read_bytes() != intact["second"]["c.json"]
    assert json.loads((retained / SIDECAR).read_text())["status"] == "superseded"
    assert [p.name for p in (run_root / mf.ATTEMPTS_DIR / "second").iterdir()] == [
        retained.name
    ]
    for name in ("first", "third"):
        assert not (run_root / mf.ATTEMPTS_DIR / name).exists(), name


def test_a_record_without_digests_is_not_reused(wf_dir, tmp_path, env):
    """The pre-digest record shape — files listed, nothing about their bytes
    — is exactly the "existence is enough" reuse this PR removes."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    _run(loaded, env, out)
    record_path = run_root / "first" / SIDECAR
    record = json.loads(record_path.read_text())
    del record["digests"]
    record_path.write_text(json.dumps(record))
    statuses = _run(loaded, env, out, resume=True)
    assert statuses["first"] == "completed"
    assert "digests" in json.loads(record_path.read_text())


def test_a_missing_file_is_not_reused(wf_dir, tmp_path, env):
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    _run(loaded, env, out)
    (run_root / "third" / "f.html").unlink()
    assert _run(loaded, env, out, resume=True)["third"] == "completed"
    _assert_complete_unit("third", run_root / "third")


# --------------------------------------------------------------------------- #
# T3 — a failed run still writes a manifest classifying every step
# --------------------------------------------------------------------------- #


def test_a_failed_run_classifies_every_step(wf_dir, tmp_path, env):
    """Step 2 raises: 1 completed, 2 failed, 3 blocked — and the manifest is
    on disk although the run raised, which only a ``finally`` does."""
    loaded = _load(wf_dir, env, scripts={"second": "raising.py"})
    out = tmp_path / "runs"
    run_root = out / "chain"
    with pytest.raises(RuntimeError, match="the step died"):
        run_workflow(loaded, env, out, None)
    manifest = _manifest(run_root)
    assert {n: e["status"] for n, e in manifest["steps"].items()} == {
        "first": "completed",
        "second": "failed",
        "third": "blocked",
    }
    assert manifest["steps"]["second"]["error"] == {
        "type": "RuntimeError",
        "message": "the step died",
    }
    assert manifest["steps"]["third"]["blocked_by"] == ["second"]
    assert "workflow_digest" not in manifest  # no run-level identity (§7)
    assert not (run_root / "second").exists(), "a failed step must not publish"
    attempts = sorted((run_root / mf.ATTEMPTS_DIR / "second").iterdir())
    assert [p.name for p in attempts] == ["0001"]
    assert (attempts[0] / mf.ATTEMPT_RECORD).is_file()


def test_an_interrupt_in_step_one_blocks_the_chain_and_leaves_the_rest_pending(
    wf_dir, tmp_path, env
):
    """``KeyboardInterrupt`` is a ``BaseException``: the ``finally`` still
    writes the manifest. The interrupted step is ``failed`` (its attempt did
    not complete), the chain behind it ``blocked``, and an independent step
    the run never reached ``pending`` — the distinction §8 draws."""
    loaded = _load(wf_dir, env, aside=True, scripts={"first": "interrupted.py"})
    assert loaded.order == ("first", "second", "third", "aside")
    out = tmp_path / "runs"
    with pytest.raises(KeyboardInterrupt):
        run_workflow(loaded, env, out, None)
    assert _statuses(out / "chain") == {
        "first": "failed",
        "second": "blocked",
        "third": "blocked",
        "aside": "pending",
    }
    assert _manifest(out / "chain")["steps"]["first"]["error"]["type"] == (
        "KeyboardInterrupt"
    )


def test_a_manifest_that_cannot_be_written_does_not_mask_the_step_failure(
    wf_dir, tmp_path, env, monkeypatch
):
    loaded = _load(wf_dir, env, scripts={"second": "raising.py"})

    def refuse(*args: Any, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_manifest", refuse)
    with pytest.warns(ProtocolWarning, match="workflow.json could not be written"):
        with pytest.raises(RuntimeError, match="the step died"):
            run_workflow(loaded, env, tmp_path / "runs", None)


def test_a_manifest_failure_on_a_clean_run_is_raised(
    wf_dir, tmp_path, env, monkeypatch
):
    loaded = _load(wf_dir, env)

    def refuse(*args: Any, **kwargs: Any) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(runner, "write_manifest", refuse)
    with pytest.raises(OSError, match="disk full"):
        run_workflow(loaded, env, tmp_path / "runs", None)


# --------------------------------------------------------------------------- #
# T4 — valid work is not refused
# --------------------------------------------------------------------------- #


def test_a_clean_run_then_resume_reuses_every_step_byte_identically(
    wf_dir, tmp_path, env
):
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    assert not (run_root / mf.ATTEMPTS_DIR).exists(), "a clean run leaves no attempts"
    assert not (run_root / f"{mf.MANIFEST}.tmp").exists()
    for name in DECLARED:
        _assert_complete_unit(name, run_root / name)
    before = _snapshot(run_root)
    record = json.loads((run_root / "second" / SIDECAR).read_text())
    assert record["checks"] == {"c.json": "json-table", "d.png": "png-signature"}
    first = json.loads((run_root / "first" / SIDECAR).read_text())
    assert first["checks"] == {"a.json": "json-table", "b.json": "json-values"}
    third = json.loads((run_root / "third" / SIDECAR).read_text())
    assert third["checks"] == {"e.json": "json-values", "f.html": "non-empty"}

    assert _run(loaded, env, out, resume=True) == {n: "reused" for n in DECLARED}
    assert _snapshot(run_root) == before
    assert _statuses(run_root) == {n: "reused" for n in DECLARED}
    assert not (run_root / mf.ATTEMPTS_DIR).exists()


def test_without_resume_every_step_is_re_executed_and_the_prior_unit_is_retained(
    wf_dir, tmp_path, env
):
    """No ``--resume``: every step re-executes and publishes over the stale
    unit — and **supersession preserves** (§8): the prior unit is
    retained byte for byte at ``.attempts/<step>/0001.superseded/``, its
    record says ``superseded`` and points at the record that replaced it, the
    manifest lists it, and the published unit is the ``accepted`` one. The
    mutation — deleting the displaced unit, as the runner once did — fails
    here on the missing file, not on a status word."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    _run(loaded, env, out)
    before = _snapshot(run_root)
    first_records = {
        name: json.loads((run_root / name / SIDECAR).read_text()) for name in DECLARED
    }
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    manifest = _manifest(run_root)
    for name in DECLARED:
        _assert_complete_unit(name, run_root / name)
        published = json.loads((run_root / name / SIDECAR).read_text())
        assert published["disposition"] == "accepted"
        retained = run_root / mf.ATTEMPTS_DIR / name / f"0001{mf.SUPERSEDED_SUFFIX}"
        assert retained.is_dir(), f"{name}: the prior unit was deleted"
        # the prior bytes, readable and equal to the first run's — every
        # declared file; the record is the one file rewritten (marked)
        for rel, data in before[name].items():
            if rel == SIDECAR:
                continue
            assert (retained / rel).read_bytes() == data, f"{name}/{rel} changed"
        marked = json.loads((retained / SIDECAR).read_text())
        assert marked["status"] == "superseded"
        assert marked["disposition"] == "superseded"
        assert marked["superseded_by"]["identity"] == published["identity"]
        assert marked["superseded_by"]["published"] == name
        assert marked["superseded_by"]["attempt"] == "0001"
        # everything else the prior record said is still there
        for key, value in first_records[name].items():
            if key not in ("status", "disposition"):
                assert marked[key] == value, (name, key)
        assert manifest["steps"][name]["superseded"] == [
            f"{mf.ATTEMPTS_DIR}/{name}/0001{mf.SUPERSEDED_SUFFIX}"
        ]
        # nothing under a `.previous` name, no numeric attempt left behind
        assert not list(
            (run_root / mf.ATTEMPTS_DIR / name).glob(f"*{mf.DISPLACED_SUFFIX}")
        )
        assert not [
            p for p in (run_root / mf.ATTEMPTS_DIR / name).iterdir() if p.name.isdigit()
        ]
    # a third run retains a second unit beside the first
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    assert _manifest(run_root)["steps"]["first"]["superseded"] == [
        f"{mf.ATTEMPTS_DIR}/first/0001{mf.SUPERSEDED_SUFFIX}",
        f"{mf.ATTEMPTS_DIR}/first/0002{mf.SUPERSEDED_SUFFIX}",
    ]


@pytest.mark.parametrize(
    ("script", "match"),
    [
        (
            "def main(i, o):\n    from pathlib import Path\n"
            "    Path(o['a']).write_text('[]')\n",
            "was not written",
        ),
        (
            "def main(i, o):\n    from pathlib import Path\n"
            "    Path(o['a']).write_text('')\n    Path(o['b']).write_text('{}')\n",
            "is empty",
        ),
        (
            "def main(i, o):\n    from pathlib import Path\n"
            "    Path(o['a']).write_text('[{')\n    Path(o['b']).write_text('{}')\n",
            "not valid JSON",
        ),
        (
            "def main(i, o):\n    from pathlib import Path\n"
            "    Path(o['a']).write_text('[{\"m\": 1}]')\n"
            "    Path(o['b']).write_text('{\"k\": 1}')\n",
            "missing \\['n'\\]",
        ),
        (
            "def main(i, o):\n    from pathlib import Path\n"
            "    Path(o['a']).write_text('[]')\n    Path(o['b']).write_text('[]')\n",
            "must be a values object",
        ),
    ],
)
def test_verification_refuses_before_publish(wf_dir, tmp_path, env, script, match):
    (wf_dir / "scripts" / "first.py").write_text(script)
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    with pytest.raises(ProtocolError, match=match):
        run_workflow(loaded, env, out, None)
    assert not (out / "chain" / "first").exists(), "an unverified unit was published"
    assert _statuses(out / "chain")["first"] == "failed"


def test_a_figure_without_its_signature_is_refused(wf_dir, tmp_path, env):
    (wf_dir / "scripts" / "second.py").write_text(
        SECOND.replace(repr(PNG), repr(b"not a png at all"))
    )
    loaded = _load(wf_dir, env)
    with pytest.raises(ProtocolError, match="png signature"):
        run_workflow(loaded, env, tmp_path / "runs", None)


def test_verify_output_holds_a_file_to_its_declaration(tmp_path):
    """The public check is the one the runner publishes by: declared columns
    against a table's first row, declared keys against a values object, the
    format alone without a declaration. It returns the name of the check that
    passed, and a refusal names the file and what it misses."""
    table = tmp_path / "direct_effect.json"
    table.write_text(json.dumps([{"example_id": "0", "delta": 0.5}]))
    columns = {"example_id": "string", "delta": "float64"}
    assert verify_output(table, OutputDecl(file=table.name, columns=columns)) == (
        "json-table"
    )
    assert verify_output(table) == "json"
    stale = OutputDecl(
        file=table.name, columns={"example": "int64", "delta": "float64"}
    )
    with pytest.raises(
        ProtocolError, match=r"output 'direct_effect\.json'.*missing \['example'\]"
    ):
        verify_output(table, stale)
    # the documented limit: only the first row is held to the declaration
    ragged = tmp_path / "ragged.json"
    ragged.write_text(json.dumps([{"example_id": "0", "delta": 0.5}, {"delta": 0.1}]))
    assert verify_output(ragged, OutputDecl(file=ragged.name, columns=columns)) == (
        "json-table"
    )
    # without a declaration a missing file is only that
    with pytest.raises(
        ProtocolError, match=r"^\[P2\] output 'gone\.json' was not written$"
    ):
        verify_output(tmp_path / "gone.json")
    values = tmp_path / "values.json"
    values.write_text(json.dumps({"ablate_layer": 3}))
    keys = OutputDecl(file=values.name, keys={"ablate_layer": 18})
    assert verify_output(values, keys) == "json-values"
    with pytest.raises(
        ProtocolError, match=r"^\[P2\] step 'scale': .*missing \['noise_scale'\]"
    ):
        verify_output(
            values,
            OutputDecl(file=values.name, keys={"noise_scale": 0.1}),
            what="step 'scale': output 'values'",
        )


def test_script_call_resolves_what_the_runner_hands_a_script(wf_dir, tmp_path, env):
    """The public resolution is the runner's own: a step reference becomes
    the producer's file, a ``key`` selector reads through it, and each
    declared output sits in the step directory. ``stamp`` refuses a declared
    output the script did not write, and a name that is no script step is
    refused before anything is read."""
    loaded = _load(wf_dir, env)
    root = tmp_path / "runs"
    (root / "first").mkdir(parents=True)
    (root / "first" / "b.json").write_text(json.dumps({"k": 2}))
    (root / "second").mkdir()
    (root / "second" / "c.json").write_text(json.dumps([{"total": 3}]))
    call = script_call("third", loaded, root, root / "third")
    assert call.inputs == {"k": 2, "c": root / "second" / "c.json"}
    assert call.outputs == {
        "e": root / "third" / "e.json",
        "f": root / "third" / "f.html",
    }
    assert call.input_digests == {
        "k": _sha256(root / "first" / "b.json"),
        "c": _sha256(root / "second" / "c.json"),
    }
    with pytest.raises(
        ProtocolError, match=r"step 'third': output 'e' \(e\.json\) was not written"
    ):
        call.stamp()
    with pytest.raises(ProtocolError, match=r"step 'fourth' is not a script step"):
        script_call("fourth", loaded, root, root / "fourth")


def _safetensors(tmp_path: Path, *, data_bytes: int, claimed: int) -> Path:
    header = {
        "w": {"dtype": "F32", "shape": [claimed // 4], "data_offsets": [0, claimed]}
    }
    raw = json.dumps(header).encode()
    target = tmp_path / "w.safetensors"
    target.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(data_bytes))
    return target


def test_safetensors_verification_checks_the_promised_length(tmp_path):
    _verify_safetensors(_safetensors(tmp_path, data_bytes=16, claimed=16), "ok")
    with pytest.raises(ProtocolError, match="promises"):
        _verify_safetensors(_safetensors(tmp_path, data_bytes=15, claimed=16), "cut")
    with pytest.raises(ProtocolError, match="promises"):
        _verify_safetensors(_safetensors(tmp_path, data_bytes=17, claimed=16), "pad")
    (tmp_path / "w.safetensors").write_bytes(b"\x00" * 4)
    with pytest.raises(ProtocolError, match="no header"):
        _verify_safetensors(tmp_path / "w.safetensors", "short")


# --------------------------------------------------------------------------- #
# T5 — the closed status vocabulary
# --------------------------------------------------------------------------- #


def _spec_status_table() -> list[str]:
    section = SPEC.read_text().split("## 8. Runner contract", 1)[1]
    section = re.split(r"^## ", section, maxsplit=1, flags=re.M)[0]
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.strip().startswith("|")
    ]
    header = next(i for i, row in enumerate(rows) if row[0] == "status")
    statuses: list[str] = []
    for row in rows[header + 2 :]:
        if not row[0].startswith("`"):
            break
        statuses.append(row[0].strip("`"))
    return statuses


def test_the_status_vocabulary_matches_the_spec_table():
    documented = _spec_status_table()
    assert documented, "no status table under §8"
    assert documented == list(mf.STEP_STATUSES)
    assert set(typing.get_args(mf.StepStatus)) == set(mf.STEP_STATUSES)
    assert len(mf.STEP_STATUSES) == 6
    assert mf.STEP_STATUSES[-1] == "skipped"  # the skip outcome (§2.8)


def test_classify_unreached_distinguishes_blocked_from_pending():
    order = ("a", "b", "c", "d")
    deps = {"a": (), "b": ("a",), "c": ("b",), "d": ()}
    entries = mf.classify_unreached(order, deps, {"a": {"status": "failed"}})
    assert {n: e["status"] for n, e in entries.items()} == {
        "b": "blocked",
        "c": "blocked",
        "d": "pending",
    }
    assert entries["c"]["blocked_by"] == ["b"]


# --------------------------------------------------------------------------- #
# T6 — bounded failure metadata
# --------------------------------------------------------------------------- #


def test_failed_attempts_are_retained_up_to_the_cap(wf_dir, tmp_path, env):
    loaded = _load(wf_dir, env, scripts={"second": "raising.py"})
    out = tmp_path / "runs"
    for _ in range(4):
        with pytest.raises(RuntimeError):
            run_workflow(loaded, env, out, None)
    attempts_root = out / "chain" / mf.ATTEMPTS_DIR / "second"
    kept = sorted(p.name for p in attempts_root.iterdir())
    assert kept == ["0002", "0003", "0004"]
    assert len(kept) == mf.RETAINED_FAILED_ATTEMPTS
    for name in kept:
        record = json.loads((attempts_root / name / mf.ATTEMPT_RECORD).read_text())
        assert record["step"] == "second"
        assert record["attempt"] == name
        assert record["status"] == "failed"
        assert record["error"] == {"type": "RuntimeError", "message": "the step died"}
        assert record["stderr_tail"] is None  # in-process: no captured stderr
        assert record["outputs_declared"] == ["c.json", "d.png"]
        assert record["outputs_written"] == []
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["started"])
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["ended"])
    # the step before it published four times over its own stale unit (no
    # `--resume`): no failed attempt is kept, the three displaced units are
    # retained as superseded (§8) and nothing else sits beside them
    first = out / "chain" / mf.ATTEMPTS_DIR / "first"
    assert sorted(p.name for p in first.iterdir()) == [
        f"{n:04d}{mf.SUPERSEDED_SUFFIX}" for n in (1, 2, 3)
    ]
    assert not [p for p in first.iterdir() if p.name.isdigit()]


def test_an_isolated_scripts_stderr_tail_is_bounded(wf_dir, tmp_path, env, monkeypatch):
    loaded = _load(wf_dir, env)
    chatty = "e" * (3 * mf.STDERR_TAIL_BYTES) + "THE END"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ScriptFailure("isolated script failed", stderr=chatty)

    monkeypatch.setattr(runner, "_run_in_process", fail)
    with pytest.raises(ScriptFailure):
        run_workflow(loaded, env, tmp_path / "runs", None)
    attempt = tmp_path / "runs" / "chain" / mf.ATTEMPTS_DIR / "first" / "0001"
    record = json.loads((attempt / mf.ATTEMPT_RECORD).read_text())
    tail = record["stderr_tail"]
    assert len(tail.encode()) <= mf.STDERR_TAIL_BYTES
    assert tail.endswith("THE END"), "the tail keeps the end, not the start"
    assert record["error"]["type"] == "ScriptFailure"


def test_partial_outputs_of_a_failed_attempt_are_kept_only_within_the_cap(
    wf_dir, tmp_path, env, monkeypatch
):
    """A failure after the first output: the attempt keeps that partial file
    (within the cap) and names it; the published tree never sees it."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"

    def inject(name: str, at: str | None) -> None:
        if name == "outputs_partial" and at == "first":
            raise InjectedFailure(name)

    monkeypatch.setattr(runner, "_boundary", inject)
    for _ in range(mf.RETAINED_FAILED_ATTEMPTS + 2):
        with pytest.raises(InjectedFailure):
            run_workflow(loaded, env, out, None)
    attempts_root = out / "chain" / mf.ATTEMPTS_DIR / "first"
    kept = sorted(attempts_root.iterdir())
    assert len(kept) == mf.RETAINED_FAILED_ATTEMPTS
    for attempt in kept:
        record = json.loads((attempt / mf.ATTEMPT_RECORD).read_text())
        assert record["outputs_written"] == ["a.json"]
        assert (attempt / "a.json").is_file()
    assert not (out / "chain" / "first").exists()
