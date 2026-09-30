"""Single measurement semantics enter the full loaded workflow's identity."""

import copy
import json
from pathlib import Path
import shutil
import tempfile

from hypothesis import given, settings, strategies as st
import pytest

from causalab.measurement.deployment.bindings import target_bindings
from causalab.measurement.census import collect_pins
from causalab.measurement.plan import execution_plan
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.workflow.document import load_workflow
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.unit


def authored_workflow():
    return {
        "version": "1",
        "output_dir": "outputs",
        "steps": {
            "harvest": {"type": "intervention_protocol", "document": "harvest.json"}
        },
        "measurement": {
            "version": 1,
            "mode": "single",
            "source": {"revision": "a" * 40},
            "cases": {"workflow": {"kind": "workflow", "cold_process": False}},
            "seeds": [0],
            "repeats": 2,
        },
    }


def observations():
    return {
        "activations": {
            "step": "harvest",
            "file": "acts_L8_ans.safetensors",
            "kind": "tensor",
        }
    }


def deployment(root, device="cpu"):
    root.mkdir()
    shutil.copyfile(CORPUS_DIR / "01_harvest_im.json", root / "harvest.json")
    shutil.copytree(FIXTURES / "data", root / "datasets")
    (root / "artifacts").mkdir()
    return {
        "source": {
            "repository": str(root / "checkout"),
            "python": str(root / "venv/bin/python"),
        },
        "device": device,
        "data_root": str(root / "datasets"),
        "artifacts_root": str(root / "artifacts"),
    }


def loaded(raw, directory, bindings):
    env = ResolutionEnv(
        datasets=FileDatasets(root=Path(bindings["data_root"])),
        artifacts=FileArtifacts(root=Path(bindings["artifacts_root"])),
    )
    return load_workflow(raw, env, workflow_dir=directory)


@settings(max_examples=12, deadline=None)
@given(
    seeds=st.lists(st.integers(0, 1000), min_size=1, max_size=4, unique=True),
    observed=st.booleans(),
    cold=st.booleans(),
)
def test_loaded_single_measurement_round_trip_preserves_canonical_and_digest(
    seeds, observed, cold
):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "workflow"
        bindings = deployment(directory)
        raw = authored_workflow()
        raw["measurement"]["seeds"] = seeds
        raw["measurement"]["cases"]["workflow"]["cold_process"] = cold
        if observed:
            raw["measurement"]["observations"] = observations()
        first = loaded(raw, directory, bindings)
        canonical = json.loads(json.dumps(first.canonical))
        measurement = canonical["measurement"]
        assert measurement == first.document.measurement
        assert measurement["mode"] == "single"
        assert ("observations" in measurement) == observed
        assert "arms" not in measurement
        assert measurement["source"]["execution"] == {
            "engine": "pytorch_hooks",
            "batch_rows": None,
            "cuda_graphs": False,
        }
        assert measurement["profile"]["backends"] == {
            "torch": {"record_shapes": False, "with_stack": False},
        }
        # Canonical steps have derived hashes; re-author only the normalized
        # measurement block to exercise the real workflow load again.
        normalized = {**raw, "measurement": measurement}
        second = loaded(normalized, directory, bindings)
        assert second.canonical == first.canonical
        assert second.digest == first.digest
        assert second.inner_digests == first.inner_digests


@pytest.mark.parametrize(
    "change", ["revision", "observations", "cold_scope", "operation", "mode"]
)
def test_loaded_identity_changes_with_scientific_measurement_semantics(
    tmp_path, change
):
    bindings = deployment(tmp_path / "workflow")
    original = authored_workflow()
    before = loaded(original, tmp_path / "workflow", bindings)
    changed = copy.deepcopy(original)
    measurement = changed["measurement"]
    if change == "revision":
        measurement["source"]["revision"] = "b" * 40
    elif change == "observations":
        measurement["observations"] = observations()
    elif change == "cold_scope":
        measurement["cases"]["workflow"]["cold_process"] = True
    elif change == "operation":
        measurement["cases"] = {"workflow": {"kind": "operation", "step": "harvest"}}
    else:
        del measurement["mode"]
        source = measurement.pop("source")
        measurement["arms"] = {"before": source, "after": {"revision": "b" * 40}}
        measurement["observations"] = observations()
    after = loaded(changed, tmp_path / "workflow", bindings)
    assert after.inner_digests == before.inner_digests
    assert after.canonical["steps"] == before.canonical["steps"]
    assert after.canonical["measurement"] != before.canonical["measurement"]
    assert after.digest != before.digest


def test_deployment_paths_and_device_are_excluded_from_loaded_scientific_identity(
    tmp_path,
):
    raw = authored_workflow()
    left = tmp_path / "local-workstation"
    right = tmp_path / "remote-accelerator"
    first_bindings = deployment(left, "cpu")
    second_bindings = deployment(right, "cuda:0")
    first = loaded(raw, left, first_bindings)
    second = loaded({**raw, "output_dir": "elsewhere"}, right, second_bindings)
    internal = execution_plan(first.document.measurement)
    assert target_bindings(first_bindings, internal) != target_bindings(
        second_bindings, internal
    )
    assert first.canonical == second.canonical
    assert first.digest == second.digest
    first_pins = collect_pins(
        first, FileDatasets(root=Path(first_bindings["data_root"]))
    )
    second_pins = collect_pins(
        second, FileDatasets(root=Path(second_bindings["data_root"]))
    )
    assert first_pins == second_pins
    canonical_text = json.dumps(first.canonical)
    assert "output_dir" not in first.canonical
    for binding in (first_bindings, second_bindings):
        for path in [
            *binding["source"].values(),
            binding["data_root"],
            binding["artifacts_root"],
        ]:
            assert path not in canonical_text
    assert "cuda:0" not in canonical_text
    assert set(first.canonical["measurement"]) >= {"mode", "source", "cases", "profile"}
    assert not (
        {
            "bindings",
            "data_root",
            "artifacts_root",
            "device",
            "source_pin_anchor",
            "observation_policy",
        }
        & set(first.canonical["measurement"])
    )
