"""A conditional over a **behavioral** step's real decision record, on the
tiny fixture (workflow spec §2.8; the engine variant of T10).

`tests/workflow/test_conditional.py` proves the layer on a CPU chain whose
decision comes from a `decision` step over a script's values object. This
file is the other producer: the behavioral fixture `qualify` writes
`decision.json` from its outcome counts, and a conditional gates two script
steps on its `outcome`. Two runs differing in one threshold —
`min_correct_rate` 0.0, which four rows of a random-weight model pass, and
1.0, which they do not — publish opposite sides, and each skipped entry names
the record by its `evidence_identity`.
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
OUTPUT_DIR = "gated_weekdays"


def _tree(tmp: Path) -> Path:
    """The behavioral fixture tree (workflow, document, table) plus the
    conditional fixtures' two scripts, copied so the workflow can be rewritten."""
    root = tmp / "gated"
    shutil.copytree(BEHAVIORAL, root)
    shutil.copytree(CONDITIONAL / "scripts", root / "scripts")
    return root


def _workflow(root: Path, min_correct_rate: float) -> Path:
    raw = json.loads((root / "qualify.json").read_text())
    raw["output_dir"] = OUTPUT_DIR
    raw["steps"]["qualify"]["thresholds"]["min_correct_rate"] = min_correct_rate
    raw["steps"]["gate"] = {
        "type": "conditional",
        "predicate": {
            "decision": {"step": "qualify"},
            "field": "outcome",
            "eq": "pass",
        },
        "on_true": ["advance_probe"],
        "on_false": ["narrow_probe"],
        "scope": "global",
    }
    for name, file in (
        ("advance_probe", "advance.json"),
        ("narrow_probe", "narrow.json"),
    ):
        raw["steps"][name] = {
            "type": "script",
            "script": {"path": "scripts/writer.py"},
            "inputs": {},
            "outputs": {"out": {"file": file, "keys": {"ran": True}}},
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
    ("min_correct_rate", "outcome", "published", "skipped"),
    [
        (0.0, "pass", "advance_probe", "narrow_probe"),
        (1.0, "fail", "narrow_probe", "advance_probe"),
    ],
    ids=["pass-advances", "fail-narrows"],
)
def test_t10_the_conditional_launches_or_skips_from_the_behavioral_decision(
    tmp_path: Path, min_correct_rate: float, outcome: str, published: str, skipped: str
) -> None:
    root = _tree(tmp_path)
    workflow = _workflow(root, min_correct_rate)
    out = tmp_path / "runs"
    assert _run_cli(root, workflow, out) == 0
    run_root = out / OUTPUT_DIR
    steps = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    decision = json.loads((run_root / "qualify" / DECISION_FILE).read_text())
    assert decision["outcome"] == outcome and decision["split"] == "development"
    assert steps["qualify"]["status"] == "completed"
    assert steps["gate"]["status"] == "completed"
    assert steps[published]["status"] == "completed"
    assert (run_root / published).is_dir()
    assert steps[skipped]["status"] == "skipped"
    assert steps[skipped]["skipped_by"] == {
        "conditional": "gate",
        "decision_step": "qualify",
        "decision_type": decision["decision_type"],
        "outcome": outcome,
        "evidence_identity": decision["evidence_identity"],
        "transitive_from": [],
    }
    assert not (run_root / skipped).exists()
    gate = json.loads((run_root / "gate" / "_step.json").read_text())
    assert gate["verdict"] is (outcome == "pass")
    assert gate["evidence"]["evidence_identity"] == decision["evidence_identity"]
    assert gate["skipped"] == [skipped]
    assert set(json.loads((run_root / "qualify" / "_step.json").read_text())) >= {
        "disposition",
        "decision",
    }
