"""``--resume`` holds a script step to the bytes of its ``path`` inputs
(workflow spec §7, §8).

A ``{"path": …}`` input enters a script step's canonical entry as the path
string alone, so editing the file it names moves no step identity. What holds
the step to the bytes it read is the **record**: ``_step.json`` carries
``input_digests`` — the sha256 of each ``path`` input by slot — and
``_reusable`` compares each against the file on ``--resume``. A step with no
``path`` inputs records no such key, so its record's key set is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.workflow import run_workflow
from causalab.workflow.document import load_workflow
from causalab.io.step_record import SIDECAR

pytestmark = pytest.mark.unit

COUNT = """
import json
from pathlib import Path
from causalab.io.tables import read_table

def main(inputs, outputs):
    rows = read_table(inputs["rows"])
    Path(outputs["out"]).write_text(json.dumps([{"n": len(rows)}]))
"""

TOTAL = """
import json
from pathlib import Path
from causalab.io.tables import read_table

def main(inputs, outputs):
    rows = read_table(inputs["counts"])
    Path(outputs["out"]).write_text(json.dumps([{"total": sum(r["n"] for r in rows)}]))
"""


#: reads its `path` input, writes its output, then rewrites the input file
REWRITE = """
import json
from pathlib import Path
from causalab.io.tables import read_table

def main(inputs, outputs):
    rows = read_table(inputs["rows"])
    Path(outputs["out"]).write_text(json.dumps([{"n": len(rows)}]))
    Path(inputs["rows"]).write_text(json.dumps(rows + [{"i": len(rows)}]))
"""


def _document() -> dict[str, Any]:
    return {
        "version": "1",
        "output_dir": "chain",
        "steps": {
            "count": {
                "type": "script",
                "script": {"path": "scripts/count.py"},
                "inputs": {"rows": {"path": "data/rows.json"}},
                "outputs": {"out": {"file": "count.json", "columns": {"n": "int64"}}},
            },
            "total": {
                "type": "script",
                "script": {"path": "scripts/total.py"},
                "inputs": {"counts": {"step": "count", "file": "count.json"}},
                "outputs": {
                    "out": {"file": "total.json", "columns": {"total": "int64"}}
                },
            },
        },
    }


@pytest.fixture()
def wf_dir(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "count.py").write_text(COUNT)
    (tmp_path / "scripts" / "total.py").write_text(TOTAL)
    (tmp_path / "data").mkdir()
    _write_rows(tmp_path, 2)
    return tmp_path


def _write_rows(wf_dir: Path, n: int) -> None:
    (wf_dir / "data" / "rows.json").write_text(json.dumps([{"i": i} for i in range(n)]))


def _run(wf_dir: Path, env: Any, out: Path, *, resume: bool = False) -> dict[str, str]:
    loaded = load_workflow(_document(), env, workflow_dir=wf_dir)
    result = run_workflow(loaded, env, out, None, resume=resume)
    return {name: entry["status"] for name, entry in result.manifest["steps"].items()}


def _record(run_root: Path, step: str) -> dict[str, Any]:
    return json.loads((run_root / step / SIDECAR).read_text())


def test_a_path_input_is_recorded_and_reused_while_its_bytes_hold(
    wf_dir, tmp_path, env
):
    out = tmp_path / "runs"
    assert _run(wf_dir, env, out) == {"count": "completed", "total": "completed"}
    record = _record(out / "chain", "count")
    rows = wf_dir / "data" / "rows.json"
    assert record["input_digests"] == {
        "rows": hashlib.sha256(rows.read_bytes()).hexdigest()
    }
    # An upstream step reference records the bytes consumed as well.
    counts = out / "chain" / "count" / "count.json"
    assert _record(out / "chain", "total")["input_digests"] == {
        "counts": hashlib.sha256(counts.read_bytes()).hexdigest()
    }
    assert _run(wf_dir, env, out, resume=True) == {"count": "reused", "total": "reused"}


def test_editing_a_path_inputs_bytes_reruns_the_step_on_resume(wf_dir, tmp_path, env):
    """The identity does not move — the path string is unchanged — and the
    record's ``input_digests`` is what re-runs the step."""
    out = tmp_path / "runs"
    assert _run(wf_dir, env, out) == {"count": "completed", "total": "completed"}
    before = _record(out / "chain", "count")

    _write_rows(wf_dir, 3)
    assert _run(wf_dir, env, out, resume=True)["count"] == "completed"
    after = _record(out / "chain", "count")
    assert after["identity"] == before["identity"]
    assert after["input_digests"] != before["input_digests"]
    assert json.loads((out / "chain" / "count" / "count.json").read_text()) == [
        {"n": 3}
    ]
    assert _run(wf_dir, env, out, resume=True)["count"] == "reused"


def test_input_digests_are_of_the_bytes_the_step_read(wf_dir, tmp_path, env):
    """A script that rewrites its own input during the run: the record names
    the bytes the step *read*, digested before the script ran — so the file
    as it stands after no longer matches, and `--resume` re-runs the step
    rather than reusing outputs computed from other bytes."""
    (wf_dir / "scripts" / "count.py").write_text(REWRITE)
    rows = wf_dir / "data" / "rows.json"
    before = hashlib.sha256(rows.read_bytes()).hexdigest()
    out = tmp_path / "runs"
    assert _run(wf_dir, env, out) == {"count": "completed", "total": "completed"}
    after = hashlib.sha256(rows.read_bytes()).hexdigest()
    assert after != before  # the script rewrote it
    assert _record(out / "chain", "count")["input_digests"] == {"rows": before}
    assert _run(wf_dir, env, out, resume=True)["count"] == "completed"


def test_a_record_without_input_digests_is_not_reused(wf_dir, tmp_path, env):
    """A record from before the field existed says nothing about the bytes
    the step read, so it is not trusted — like a record without digests."""
    out = tmp_path / "runs"
    assert _run(wf_dir, env, out) == {"count": "completed", "total": "completed"}
    path = out / "chain" / "count" / SIDECAR
    record = json.loads(path.read_text())
    del record["input_digests"]
    path.write_text(json.dumps(record))
    assert _run(wf_dir, env, out, resume=True)["count"] == "completed"
