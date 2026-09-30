"""Offline single-source example authors real inputs without extra observations."""

import json

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("observations", [False, True])
def test_example_preparation_is_offline_and_authors_one_source(tmp_path, observations):
    from examples.measurements.single import prepare
    from causalab.workflow.document import parse_workflow

    document, bindings = prepare(tmp_path / "example", observations=observations)
    raw = json.loads(document.read_text())
    parsed = parse_workflow(raw)
    assert parsed.measurement["mode"] == "single"
    assert ("observations" in parsed.measurement) is observations
    assert parsed.measurement["profile"]["backends"].keys() == {"torch"}
    assert set(parsed.measurement["cases"]) == {"resident", "cold"}
    binding = json.loads(bindings.read_text())
    assert "source" in binding and "arms" not in binding
    protocol = json.loads((document.parent / "inference.json").read_text())
    assert protocol["method"]["save"][0]["file_path"] == "logits.safetensors"
    assert (document.parent / "checkpoint/model.safetensors").is_file()


def test_qwen_single_preset_is_a_valid_timing_and_profiling_plan():
    from pathlib import Path
    from causalab.workflow.document import parse_workflow

    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root / "examples/measurements/qwen/single.json").read_text())
    plan = parse_workflow(raw).measurement
    assert plan["mode"] == "single"
    assert "observations" not in plan
    assert "evaluation" not in plan
    assert set(plan["cases"]) == {"resident", "cold"}
    assert plan["profile"]["cases"] == ["resident", "cold"]
