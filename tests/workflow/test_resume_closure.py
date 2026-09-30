"""``--resume`` re-runs a ``{"path": …}`` script step when a sibling module it
*imports* changed (workflow spec §4.2, §7).

The reuse check compares the step's digest, and the digest carries the
script's declared import closure — its siblings — beside its own hash, so
nothing in the runner had to change for this to hold: ``_step_identity`` reads
``step_digests``, which already absorbs any change to the canonical entry.
This is the one case only the closure covers: the sibling lives beside the
workflow, outside the ``causalab`` package, so the ``implementation.tree_digest``
the reuse check also compares never sees it. Without the closure the third run
below is ``reused``, and a step whose arithmetic lives in a sibling module is
skipped as up to date after that module changed.

The valid-work twin rides along: the unchanged tree is reused, and the step record names
the files the identity hashed so a reader can see why a step ran again.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.io.step_record import SIDECAR
from causalab.io.tables import read_table
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

pytestmark = pytest.mark.unit

HELPER_NAME = "closure_probe_helper"

#: A script whose number comes from a sibling module. The script puts its own
#: directory on ``sys.path`` before the import, which is where the closure walk
#: resolves the sibling too (the script's own root, §4.2).
SCRIPT = f"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import {HELPER_NAME}  # noqa: E402


def main(inputs, outputs):
    Path(outputs["out"]).write_text(json.dumps([{{"n": {HELPER_NAME}.value()}}]))
"""

HELPER_V1 = "def value():\n    return 1\n"
#: Longer than V1 on purpose: the interpreter validates a cached ``.pyc`` by
#: source size and mtime, and two edits within one second of equal length
#: would hand the re-run the *old* bytecode — the digest would have moved and
#: the step re-executed, but with yesterday's arithmetic. Not this test's
#: subject, so it is kept out of the way rather than asserted on.
HELPER_V2 = "def value():\n    return 2  # the sibling changed\n"


def _document() -> dict[str, Any]:
    return {
        "version": "1",
        "output_dir": "closure",
        "steps": {
            "count": {
                "type": "script",
                "script": {"path": "scripts/count.py"},
                "inputs": {"seed": 1},
                "outputs": {"out": {"file": "count.json", "columns": {"n": "int64"}}},
            }
        },
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_resume_reruns_a_step_whose_imported_module_changed(
    tmp_path: Path, env: Any
) -> None:
    wf_dir = tmp_path / "wf"
    scripts = wf_dir / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "count.py").write_text(SCRIPT)
    helper = scripts / f"{HELPER_NAME}.py"
    helper.write_text(HELPER_V1)
    out = tmp_path / "runs"

    try:
        loaded = load_workflow(_document(), env, workflow_dir=wf_dir)
        first = run_workflow(loaded, env, out, None)
        assert first.manifest["steps"]["count"]["status"] == "completed"
        # the record names what the identity hashed beside the script
        record = json.loads((first.run_root / "count" / SIDECAR).read_text())
        assert record["closure"] == {f"{HELPER_NAME}.py": _sha256(HELPER_V1)}
        assert (
            record["closure_sha256"]
            == loaded.canonical["steps"]["count"]["closure_sha256"]
        )

        # the twin: nothing changed, the step is reused
        again = run_workflow(
            load_workflow(_document(), env, workflow_dir=wf_dir),
            env,
            out,
            None,
            resume=True,
        )
        assert again.manifest["steps"]["count"]["status"] == "reused"

        helper.write_text(HELPER_V2)
        sys.modules.pop(HELPER_NAME, None)  # the re-run must see the new bytes
        edited = load_workflow(_document(), env, workflow_dir=wf_dir)
        assert edited.step_digests["count"] != loaded.step_digests["count"]
        third = run_workflow(edited, env, out, None, resume=True)
        assert third.manifest["steps"]["count"]["status"] == "completed", (
            "a step whose imported module changed was reused as up to date"
        )
        assert read_table(third.run_root / "count" / "count.json")[0]["n"] == 2
        record = json.loads((third.run_root / "count" / SIDECAR).read_text())
        assert record["closure"] == {f"{HELPER_NAME}.py": _sha256(HELPER_V2)}
    finally:
        sys.modules.pop(HELPER_NAME, None)
