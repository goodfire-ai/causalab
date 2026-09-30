"""The supplied fixed-work and normal-policy studies share a maximum budget."""

import json
from pathlib import Path

import pytest

from causalab.io.sources import apply_overrides
from causalab.protocol.schema import parse_document
from causalab.workflow.document import parse_workflow

pytestmark = pytest.mark.unit


def test_presets_separate_fixed_work_from_natural_stopping():
    root = Path(__file__).resolve().parents[2] / "examples/measurements/qwen"
    documents = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in ("fixed-work", "normal-workflow")
    }
    parsed = {name: parse_workflow(doc) for name, doc in documents.items()}
    for value in parsed.values():
        assert value.measurement["profile"]["cases"] == []
        assert all(
            arm["execution"]["cuda_graphs"]
            for arm in value.measurement["arms"].values()
        )
    assert {
        case["kind"] for case in parsed["fixed-work"].measurement["cases"].values()
    } == {"operation"}
    assert {
        case["kind"] for case in parsed["normal-workflow"].measurement["cases"].values()
    } == {"workflow"}
    pairs = sum(
        row["split"] == "train"
        for row in json.loads((root / "data/pairs.json").read_text())
    )
    for value in parsed.values():
        assert all(
            arm["execution"]["batch_rows"] >= pairs
            for arm in value.measurement["arms"].values()
        )  # A smaller bound selects eager row-window executors before fitting.
    for method in ("das", "dbm"):
        resolved = {}
        for name, doc in documents.items():
            step = doc["steps"][method]
            raw = apply_overrides(
                json.loads((root / step["document"]).read_text()), step["set"]
            )
            parse_document(raw)
            resolved[name] = raw
        fixed = resolved["fixed-work"]["method"]["train"]
        normal = resolved["normal-workflow"]["method"]["train"]
        assert fixed.pop("steps")["updates"] == normal.pop("steps")["epochs"] * (
            (pairs + normal["batch"]["pairs"] - 1) // normal["batch"]["pairs"]
        )
        assert "early_stop" not in fixed
        assert normal.pop("early_stop") == {
            "on": "iia",
            "patience": 3,
            "mode": "max",
        }
        assert resolved["fixed-work"] == resolved["normal-workflow"]
