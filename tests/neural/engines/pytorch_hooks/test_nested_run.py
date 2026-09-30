"""A nested workflow whose inner step is a **behavioral** step on the tiny
fixture (workflow spec §2.10; the engine variant).

`tests/workflow/test_nested.py` proves the layer on a CPU chain of scripts and
decisions. This file nests the behavioral fixture (`qualify.json`) as
`tail`, retargets its `model.key` through the nested `set` form — one level
deeper than a document step's `set` — and gates a script step on the real
`decision.json` the inner step writes under the sub-root, by
`requires_receipt: {"step": "tail/qualify", …}`. Two runs differing in one
inner threshold — `min_correct_rate` 0.0, which four rows of a random-weight
model pass, and 1.0, which they do not — allocate the gated step or refuse it
before any engine is chosen.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.workflow import manifest as mf
from causalab.workflow.behavioral import DECISION_FILE

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
BEHAVIORAL = REPO / "tests" / "workflow" / "fixtures" / "behavioral"
CONDITIONAL = REPO / "tests" / "workflow" / "fixtures" / "conditional"
OUTPUT_DIR = "nested_weekdays"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _tree(tmp: Path) -> Path:
    """The behavioral fixture tree (workflow, document, table) plus the
    conditional fixtures' two scripts, copied so the documents can be rewritten."""
    root = tmp / "nested"
    shutil.copytree(BEHAVIORAL, root)
    shutil.copytree(CONDITIONAL / "scripts", root / "scripts")
    return root


def _workflow(root: Path, min_correct_rate: float) -> Path:
    inner = json.loads((root / "qualify.json").read_text())
    inner["steps"]["qualify"]["thresholds"]["min_correct_rate"] = min_correct_rate
    # the inner names gpt2 (a registered key, so it loads torch-free); the
    # outer's `set` retargets it, one level deeper
    inner["steps"]["qualify"]["set"].pop("model.key")
    (root / "qualify.json").write_text(json.dumps(inner, indent=2))
    raw = {
        "version": "1",
        "output_dir": OUTPUT_DIR,
        "steps": {
            "tail": {
                "type": "workflow",
                "document": "qualify.json",
                "set": {"qualify": {"model.key": TINY}},
            },
            "advance_probe": {
                "type": "script",
                "script": {"path": "scripts/writer.py"},
                "inputs": {},
                "outputs": {"out": {"file": "advance.json", "keys": {"ran": True}}},
                "requires_receipt": {"step": "tail/qualify", "outcome": "pass"},
            },
        },
    }
    target = root / "gated.json"
    target.write_text(json.dumps(raw, indent=2))
    return target


def _run_cli(root: Path, workflow: Path, out: Path) -> int:
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(workflow),
            "--data-root",
            str(root),
            "--artifacts-root",
            str(root),
            "--out",
            str(out),
        ]
    )


@pytest.mark.parametrize(
    ("min_correct_rate", "outcome", "code", "gated"),
    [(0.0, "pass", 0, "completed"), (1.0, "fail", 1, "failed")],
    ids=["pass-allocates", "fail-refuses"],
)
def test_a_script_gated_on_a_nested_behavioral_receipt(
    tmp_path: Path, min_correct_rate: float, outcome: str, code: int, gated: str
) -> None:
    root = _tree(tmp_path)
    workflow = _workflow(root, min_correct_rate)
    out = tmp_path / "runs"
    assert _run_cli(root, workflow, out) == code
    run_root = out / OUTPUT_DIR
    manifest = json.loads((run_root / mf.MANIFEST).read_text())
    steps = manifest["steps"]
    assert set(steps) == {"tail/qualify", "advance_probe"}  # the container has no entry
    decision = json.loads((run_root / "tail" / "qualify" / DECISION_FILE).read_text())
    assert decision["outcome"] == outcome and decision["split"] == "development"
    record = json.loads((run_root / "tail" / "qualify" / "_step.json").read_text())
    assert record["type"] == "behavioral" and record["status"] == "completed"
    assert steps["tail/qualify"]["status"] == "completed"
    assert steps["advance_probe"]["status"] == gated
    assert manifest["nested"]["tail"]["steps"] == ["tail/qualify"]
    assert manifest["nested"]["tail"]["document"] == "qualify.json"
    if gated == "completed":
        assert (run_root / "advance_probe" / "advance.json").is_file()
    else:
        assert "W18" in steps["advance_probe"]["error"]["message"]
        assert "'tail/qualify'" in steps["advance_probe"]["error"]["message"]
        assert not (run_root / "advance_probe").exists()
        assert not (run_root / mf.ATTEMPTS_DIR / "advance_probe").exists()
