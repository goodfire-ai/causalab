"""What ``--resume`` compares is pinned, and it is a closed set (workflow
spec §7, §8).

``runner._reusable`` is the whole of the ``--resume`` decision. It reads a
step's ``_step.json`` and compares exactly: ``identity`` (the step's own
digest), ``engine`` (a protocol step's), ``implementation.tree_digest``,
``files`` / ``digests`` (every published file's content digest against the
bytes on disk), ``input_digests`` (a script step's ``{"path": …}`` inputs,
each against the file's current bytes — the chain here has none, so its twin
lives in ``tests/workflow/test_resume_inputs.py``), ``document_digest`` (a
fanned-out child's, when its document loads a run-tree path, against a
compile over the current run tree; its twin lives in
``tests/workflow/test_resume_run_tree_references.py``), and the
``conditional`` / ``fan_out`` evidence clauses. Everything else a record
carries is written for a reader and never consulted.

Three tests hold that line:

* the **pin** — the ``identity``, ``files`` and ``digests`` of the tiny
  three-step chain (``test_attempt_publish``), generated once by
  ``update_resume_identity_pins.py`` and compared byte for byte. A PR that
  drops recorded fields (stamps, provenance columns, a document section) can
  prove it left ``--resume`` alone by leaving this file's diff empty;
* the **census** — the record keys ``_reusable`` reads, counted against its
  source, so a compared field cannot be added or removed silently;
* the **twin** — the decision itself, on a real run tree: a mutated compared
  field refuses, a mutated recorded-only field does not.

The chain's identities are pure functions of the canonical form (relative
script paths, ``script_sha256``, fixed inputs and outputs; the run directory
is not in them — ``test_moving_the_run_tree`` proves it), which is what
makes a committed pin portable.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.io.step_record import SIDECAR
from causalab.workflow import runner
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

from tests.workflow.test_attempt_publish import DECLARED, _document
from tests.workflow.update_resume_identity_pins import (
    PINNED_FIELDS,
    PINS_PATH,
    chain_dir,
    pins_of,
)

pytestmark = pytest.mark.unit

REGENERATE = (
    "regenerate with `uv run python tests/workflow/update_resume_identity_pins.py` "
    "and review the diff — a moved identity is a canonical-form change (spec §7)"
)

#: Every ``record.get("<key>")`` in ``_reusable``: the closed set of
#: top-level record fields the reuse decision reads.
COMPARED_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "identity",
        "engine",
        "implementation",
        "files",
        "digests",
        "input_digests",
        "document_digest",
    }
)

#: Every ``recorded.get("<key>")`` — the one ``implementation`` field compared.
COMPARED_IMPLEMENTATION_FIELDS: tuple[str, ...] = ("tree_digest",)


@pytest.fixture()
def wf_dir(tmp_path: Path) -> Path:
    return chain_dir(tmp_path)


def _run(wf_dir: Path, env: Any, out: Path) -> Path:
    loaded = load_workflow(_document(), env, workflow_dir=wf_dir)
    run_workflow(loaded, env, out, None)
    return out / "chain"


def _pins() -> dict[str, dict[str, Any]]:
    return json.loads(PINS_PATH.read_text())


# --------------------------------------------------------------------------- #
# the pin
# --------------------------------------------------------------------------- #


def test_the_pin_covers_every_step_of_the_chain():
    pins = _pins()
    assert set(pins) == set(DECLARED), REGENERATE
    for step, entry in pins.items():
        assert set(entry) == {*PINNED_FIELDS, "record_keys"}, (step, REGENERATE)
        assert re.fullmatch(r"[0-9a-f]{64}", entry["identity"]), (step, REGENERATE)
        assert set(entry["digests"]) == set(entry["files"]) == DECLARED[step], step


def test_a_clean_run_records_the_pinned_identities_files_and_digests(
    wf_dir, tmp_path, env
):
    """The chain run now records exactly what the pin says it recorded: the
    same step identities, the same published files, the same content digests
    and the same record key set."""
    run_root = _run(wf_dir, env, tmp_path / "runs")
    assert pins_of(run_root) == _pins(), REGENERATE


# --------------------------------------------------------------------------- #
# the census
# --------------------------------------------------------------------------- #


def test_reusable_reads_a_closed_set_of_record_fields():
    """``_reusable`` consults these record fields and no others; a compared
    field added or removed shows up here before it shows up in a run tree."""
    source = inspect.getsource(runner._reusable)  # pyright: ignore[reportPrivateUsage]
    assert set(re.findall(r'record\.get\("(\w+)"\)', source)) == COMPARED_RECORD_FIELDS
    assert (
        tuple(re.findall(r'recorded\.get\("(\w+)"\)', source))
        == COMPARED_IMPLEMENTATION_FIELDS
    )
    # the two evidence clauses are the only other reads of the record
    assert "conditional.evidence_holds(step, step_dir, record)" in source
    assert "fan_out.evidence_holds(step, step_dir, record)" in source


# --------------------------------------------------------------------------- #
# the twin
# --------------------------------------------------------------------------- #


def _reusable(loaded: Any, name: str, run_root: Path) -> dict[str, Any] | None:
    return runner._reusable(  # pyright: ignore[reportPrivateUsage]
        loaded,
        name,
        loaded.document.steps[name],
        run_root / name,
        True,
        False,
        runner._implementation(),  # pyright: ignore[reportPrivateUsage]
        None,
    )


def _rewrite(run_root: Path, step: str, edit: Any) -> None:
    path = run_root / step / SIDECAR
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record, indent=2))


def test_the_decision_holds_on_compared_fields_and_ignores_the_rest(
    wf_dir, tmp_path, env
):
    """On the clean run tree every step is reusable. A compared field that
    moves — the identity, the tree digest, a published byte — refuses; a
    recorded-only field that moves — ``inputs``, ``digest``, ``checks`` —
    changes nothing, because nothing compares it."""
    loaded = load_workflow(_document(), env, workflow_dir=wf_dir)
    out = tmp_path / "runs"
    run_workflow(loaded, env, out, None)
    run_root = out / "chain"
    for name in DECLARED:
        reused = _reusable(loaded, name, run_root)
        assert reused is not None and reused["status"] == "reused", name

    # recorded, never compared
    def touch_recorded(record: dict[str, Any]) -> None:
        record["inputs"] = {"seed": {"value": 99}}
        record["digest"] = "f" * 64
        record["checks"] = {}

    _rewrite(run_root, "first", touch_recorded)
    assert _reusable(loaded, "first", run_root) is not None

    # compared
    def move_identity(record: dict[str, Any]) -> None:
        record["identity"] = "0" * 64

    _rewrite(run_root, "first", move_identity)
    assert _reusable(loaded, "first", run_root) is None

    def move_tree(record: dict[str, Any]) -> None:
        record["implementation"]["tree_digest"] = "0" * 64

    _rewrite(run_root, "second", move_tree)
    assert _reusable(loaded, "second", run_root) is None

    (run_root / "third" / "f.html").write_text("<html>moved</html>")
    assert _reusable(loaded, "third", run_root) is None
