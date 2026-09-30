"""``--resume`` refuses reuse when the implementation identity differs
(workflow spec §7, §8).

Two capabilities are tested here: resuming a workflow, and
refusing to resume under code that is not the code that produced the step.
"Same code" has two halves. [`causalab.provenance.runtime_identity`][]
gives the bytes of the ``causalab`` package that ran — its ``tree_digest`` is
what a step record now carries in its ``implementation`` block and what the
reuse check compares. The step identity covers a step's *user* code by
content: a script step's ``script_sha256`` and an intervention step's §2.8.1 ``code``
references' ``source_sha256`` are already inside the identity the record
compares, so that half is proved here rather than re-implemented.

Every refusal is bound to a proof that valid work still passes: only
``tree_digest`` is compared, and equal digests are the same bytes whatever
tree they came from. The guardrail — moving the run tree alone must not bust
reuse — is what stops the record from hashing paths.

Without the change, T1 and T2 fail the same way: every step is ``reused``,
because nothing compared the implementation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.provenance import RuntimeIdentity, runtime_identity
from causalab.workflow import runner
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import IMPLEMENTATION_FIELDS

from tests.protocol.test_code_identity import CLEAN_SOURCE, doc_with_code, write_module
from tests.workflow.test_attempt_publish import (
    DECLARED,
    FIRST,
    SECOND,
    SIDECAR,
    SPEC,
    THIRD,
    _assert_complete_unit,  # pyright: ignore[reportPrivateUsage]
    _document,  # pyright: ignore[reportPrivateUsage]
    _load,  # pyright: ignore[reportPrivateUsage]
    _run,  # pyright: ignore[reportPrivateUsage]
    _snapshot,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def wf_dir(tmp_path: Path) -> Path:
    """The tiny three-step chain of ``test_attempt_publish``, in a fresh
    directory: the same scripts, so the same units and the same records."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name, source in (("first", FIRST), ("second", SECOND), ("third", THIRD)):
        (scripts / f"{name}.py").write_text(source)
    return tmp_path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _record(run_root: Path, step: str) -> dict[str, Any]:
    return json.loads((run_root / step / SIDECAR).read_text())


def _implementations(run_root: Path) -> dict[str, dict[str, Any]]:
    return {step: _record(run_root, step)["implementation"] for step in DECLARED}


def _stand_in(monkeypatch: pytest.MonkeyPatch, identity: RuntimeIdentity) -> None:
    """Make the runner see ``identity`` as the running package. The runner
    imports ``runtime_identity`` as a module attribute for exactly this."""
    monkeypatch.setattr(runner, "runtime_identity", lambda: identity)


@pytest.fixture(autouse=True)
def fresh_identity() -> None:
    # each test reads the process for itself (tests/test_provenance.py does the
    # same): the cache would otherwise let an earlier test fix the value
    runtime_identity.cache_clear()


# --------------------------------------------------------------------------- #
# T1 — refused when the implementation differs
# --------------------------------------------------------------------------- #


def test_a_different_tree_digest_re_executes_every_step(
    wf_dir, tmp_path, env, monkeypatch
):
    """Same workflow, same bytes on disk, a different package: nothing is
    reused, everything is re-executed and republished as a verified unit, and
    the new records carry the new digest."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    real = runtime_identity()
    assert {i["tree_digest"] for i in _implementations(run_root).values()} == {
        real.tree_digest
    }

    other = dataclasses.replace(real, tree_digest="0" * 64)
    _stand_in(monkeypatch, other)
    assert _run(loaded, env, out, resume=True) == {n: "completed" for n in DECLARED}
    for name in DECLARED:
        _assert_complete_unit(name, run_root / name)
    assert {i["tree_digest"] for i in _implementations(run_root).values()} == {"0" * 64}
    assert not (run_root / "workflow.json.tmp").exists()

    # and back under the other package, the units it wrote are its to reuse
    assert _run(loaded, env, out, resume=True) == {n: "reused" for n in DECLARED}


# --------------------------------------------------------------------------- #
# T2 — a record without an implementation block is not reused
# --------------------------------------------------------------------------- #


def test_a_record_without_an_implementation_block_is_not_reused(wf_dir, tmp_path, env):
    """A record from before the implementation block: files, digests, checks,
    no ``implementation``. It cannot say what code wrote it, so it is
    re-executed — and only it; the other units still carry the running
    package's digest and are reused."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    record_path = run_root / "first" / SIDECAR
    record = json.loads(record_path.read_text())
    del record["implementation"]
    record_path.write_text(json.dumps(record))

    assert _run(loaded, env, out, resume=True) == {
        "first": "completed",
        "second": "reused",
        "third": "reused",
    }
    _assert_complete_unit("first", run_root / "first")
    assert _record(run_root, "first")["implementation"]["tree_digest"] == (
        runtime_identity().tree_digest
    )


def test_a_malformed_implementation_block_is_not_reused(wf_dir, tmp_path, env):
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    _run(loaded, env, out)
    record_path = run_root / "second" / SIDECAR
    record = json.loads(record_path.read_text())
    record["implementation"] = "not a block"
    record_path.write_text(json.dumps(record))
    assert _run(loaded, env, out, resume=True)["second"] == "completed"


# --------------------------------------------------------------------------- #
# T3 — valid work: the same process reuses everything
# --------------------------------------------------------------------------- #


def test_a_clean_run_then_resume_reuses_every_step_and_records_the_implementation(
    wf_dir, tmp_path, env
):
    """The twin of the clean-run test, extended: every record carries the
    running package's identity in the closed field set, the reused entries in
    the manifest are the earlier records, and no byte moves."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    real = runtime_identity()
    for name in DECLARED:
        block = _record(run_root, name)["implementation"]
        assert tuple(block) == IMPLEMENTATION_FIELDS
        assert block == {"tree_digest": real.tree_digest}
    before = _snapshot(run_root)

    result = runner.run_workflow(loaded, env, out, None, resume=True)
    statuses = {n: e["status"] for n, e in result.manifest["steps"].items()}
    assert statuses == {n: "reused" for n in DECLARED}
    assert _snapshot(run_root) == before
    for name in DECLARED:
        entry = result.manifest["steps"][name]
        assert entry["implementation"]["tree_digest"] == real.tree_digest


def test_the_identity_is_asked_once_per_run(wf_dir, tmp_path, env, monkeypatch):
    """``runtime_identity()`` hashes the whole package on its first call; the
    runner asks once per run, not once per step, and every record of the run
    carries that one answer."""
    loaded = _load(wf_dir, env)
    real = runtime_identity()
    calls: list[int] = []

    def counted() -> RuntimeIdentity:
        calls.append(1)
        return real

    monkeypatch.setattr(runner, "runtime_identity", counted)
    out = tmp_path / "runs"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    assert len(calls) == 1
    assert _run(loaded, env, out, resume=True) == {n: "reused" for n in DECLARED}
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# T4 — the guardrail: moving the run tree alone does not bust reuse
# --------------------------------------------------------------------------- #


def test_moving_the_run_tree_alone_does_not_bust_reuse(wf_dir, tmp_path, env):
    """The guardrail. The tree is moved to a
    different parent *and* a different directory name; ``--resume`` under the
    new ``output_dir`` reuses every step byte-identically. This is what stops
    the record from carrying a path — ``runtime_identity().location`` is an
    absolute path and is deliberately not recorded."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    before = _snapshot(out / "chain")

    elsewhere = tmp_path / "archive" / "nested"
    elsewhere.mkdir(parents=True)
    shutil.move(str(out / "chain"), str(elsewhere / "relocated"))
    assert not (out / "chain").exists()

    relocated = load_workflow(
        _document() | {"output_dir": "relocated"}, env, workflow_dir=wf_dir
    )
    assert relocated.digest == loaded.digest, "output_dir is not in the digest (§7)"
    assert _run(relocated, env, elsewhere, resume=True) == {
        n: "reused" for n in DECLARED
    }
    assert _snapshot(elsewhere / "relocated") == before
    for name in DECLARED:
        text = (elsewhere / "relocated" / name / SIDECAR).read_text()
        assert str(out) not in text and str(elsewhere) not in text
        assert runtime_identity().location not in text


# --------------------------------------------------------------------------- #
# T5 — the user-code half: user code by content
# --------------------------------------------------------------------------- #


def test_editing_one_byte_of_a_script_re_executes_that_step(wf_dir, tmp_path, env):
    """(a) A script step: ``script_sha256`` is in the step's canonical entry,
    so a comment moves ``loaded.step_digests[name]`` — the identity the record
    compares — and that step alone runs again. Its output bytes are the same,
    so the step downstream still reuses."""
    loaded = _load(wf_dir, env)
    out = tmp_path / "runs"
    run_root = out / "chain"
    assert _run(loaded, env, out) == {n: "completed" for n in DECLARED}
    before = _record(run_root, "second")

    script = wf_dir / "scripts" / "second.py"
    script.write_text(script.read_text() + "\n# one more byte of comment\n")
    edited = _load(wf_dir, env)
    assert edited.step_digests["second"] != loaded.step_digests["second"]
    assert edited.step_digests["first"] == loaded.step_digests["first"]
    assert (
        edited.document.steps["second"].script_sha256
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )

    assert _run(edited, env, out, resume=True) == {
        "first": "reused",
        "second": "completed",
        "third": "reused",
    }
    after = _record(run_root, "second")
    assert after["identity"] == edited.step_digests["second"] != before["identity"]
    assert after["implementation"] == before["implementation"]


def test_editing_a_code_reference_moves_a_protocol_steps_compared_identity(
    tmp_path, env, monkeypatch
):
    """(b) An intervention step whose document carries a §2.8.1 ``code``
    reference: the referenced module's ``source_sha256`` is in the canonical
    form, so it is in ``inner.compiled.digests.document`` — which is
    ``_step_identity`` for a protocol step, the value the record's
    ``identity`` compares. Proved at the identity level: running the step
    needs an engine, and this tier has none."""
    module = write_module(tmp_path, monkeypatch, "pr30_code_under_test", CLEAN_SOURCE)
    (tmp_path / "inner.json").write_text(
        json.dumps(doc_with_code("pr30_code_under_test.corrupt", args={"scale": 0.5}))
    )
    document = {
        "version": "1",
        "output_dir": "probe",
        "steps": {"probe": {"type": "intervention_protocol", "document": "inner.json"}},
    }

    loaded = load_workflow(document, env, workflow_dir=tmp_path)
    step = loaded.document.steps["probe"]
    before = runner._step_identity(loaded, "probe", step)  # pyright: ignore[reportPrivateUsage]
    inner = loaded.inner["probe"]
    assert before == loaded.inner_digests["probe"] == inner.compiled.digests.document
    assert inner.compiled.canonical["method"]["code"]["corrupt"]["source_sha256"] == (
        hashlib.sha256(module.read_bytes()).hexdigest()
    )
    # the guard against the guard: the same tree loads to the same identity
    assert (
        runner._step_identity(  # pyright: ignore[reportPrivateUsage]
            load_workflow(document, env, workflow_dir=tmp_path), "probe", step
        )
        == before
    )

    module.write_text(module.read_text().replace("* scale", "+ scale"))
    importlib.invalidate_caches()
    edited = load_workflow(document, env, workflow_dir=tmp_path)
    after = runner._step_identity(edited, "probe", step)  # pyright: ignore[reportPrivateUsage]
    assert after != before, "an edited referenced function left the identity alone"
    assert (
        edited.inner["probe"].compiled.canonical["method"]["code"]["corrupt"][
            "source_sha256"
        ]
        == hashlib.sha256(module.read_bytes()).hexdigest()
    )


# --------------------------------------------------------------------------- #
# T6 — the spec and the record shape agree
# --------------------------------------------------------------------------- #


#: The spec's published companion page on the runner's implementation.
INTERNALS = SPEC.with_name("workflow_protocol_internals.md")


def _section(heading: str, spec: Path = SPEC) -> str:
    text = spec.read_text().split(heading, 1)[1]
    return re.split(r"^## ", text, maxsplit=1, flags=re.M)[0]


def _row(section: str, service: str) -> str:
    rows = [
        line
        for line in section.splitlines()
        if line.strip().startswith(f"| {service} |")
    ]
    assert len(rows) == 1, f"expected one `{service}` row, found {len(rows)}"
    return rows[0]


def test_the_spec_names_the_check_and_the_recorded_fields():
    """§7's ``--resume`` paragraph and §8's ``resume`` and ``stamping`` rows
    state the implementation comparison, and §8 lists exactly the fields the
    writer records — the census that holds the record and the spec together.
    (The spec tabulates no ``_step.json`` field list, so none is invented.)"""
    seven = _section("## 7. Canonical form, digests, and `--resume`")
    resume_paragraph = next(
        p for p in seven.split("\n\n") if p.startswith("Reuse requires ")
    )
    assert "`tree_digest`" in resume_paragraph
    assert "SHA-256" in resume_paragraph and "`input_digests`" in resume_paragraph
    assert (
        "Missing identity or digest records cause execution again" in resume_paragraph
    )

    # The service rows are the runner contract; the status table stays in the spec.
    eight = _section("## 8. Runner contract", INTERNALS)
    assert "`implementation.tree_digest`" in _row(eight, "resume")
    stamping = _row(eight, "stamping")
    assert "`implementation`" in stamping
    for field in IMPLEMENTATION_FIELDS:
        assert f"`{field}`" in stamping, f"§8 stamping row does not list {field!r}"
    statuses = _section("## 8. Runner contract")
    assert "`implementation`" in _row(statuses, "`reused`")
    assert IMPLEMENTATION_FIELDS[0] == "tree_digest"
    assert len(set(IMPLEMENTATION_FIELDS)) == len(IMPLEMENTATION_FIELDS) == 1
