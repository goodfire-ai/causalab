import json
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from causalab.protocol.bundles import RAGGED_SUFFIX
from causalab.measurement.runtime.observations import observation_specs, observations

pytestmark = pytest.mark.unit


def test_gate_temperature_binds_to_saved_fit_not_an_assumed_default(tmp_path):
    root = tmp_path / "fit"
    root.mkdir()
    save_file(
        {"theta": torch.tensor([-0.2, 0.2])},
        str(root / "gate.safetensors"),
        metadata={
            "entries": json.dumps(
                {
                    "theta": {
                        "slot": "theta",
                        "coords": {},
                        "produced_by": "point-1",
                    }
                }
            )
        },
    )
    diagnostic = root / "fit_diagnostics.json"
    diagnostic.write_text(
        json.dumps(
            [
                {
                    "point": "point-1",
                    "coords": {},
                    "featurizers": {"gate": {"temperature": 0.02}},
                }
            ]
        )
    )
    spec: dict[str, Any] = {
        "gate": {"step": "fit", "file": "gate.safetensors", "kind": "gate"}
    }
    resolved = observation_specs(tmp_path, spec)
    key = next(key for key in resolved if "/" in key)
    assert resolved[key]["temperature"] == 0.02
    assert resolved[key]["temperature_source"] == "fit_diagnostics"
    spec["gate"]["temperature"] = 1
    with pytest.raises(ValueError, match="disagrees with fit"):
        observation_specs(tmp_path, spec)
    diagnostic.unlink()
    assert observation_specs(tmp_path, spec)[key]["temperature_source"] == "declared"
    del spec["gate"]["temperature"]
    with pytest.raises(ValueError, match="no finite positive fit temperature"):
        observation_specs(tmp_path, spec)


def test_swept_gate_temperatures_follow_bundle_coordinates_and_restored_state(tmp_path):
    from causalab.neural.shared.results import TensorFile

    root = tmp_path / "fit"
    root.mkdir()
    bundle = TensorFile()
    rows = []
    # Synthetic values: one temperature per layer, and an annealed final value
    # that differs from it, so a resolver that read the anneal would fail.
    for layer, temperature in [(12, 0.4), (20, 0.25), (28, 0.05)]:
        coords = {"sites.target.layers": layer, "featurizers.gate.init": 0.5}
        point = f"point-{layer}"
        bundle.add(
            "theta",
            torch.tensor([-0.2, 0.2]),
            coords,
            label_entry="gate",
            identity={"produced_by": point},
        )
        rows.append(
            {
                "point": point,
                "coords": coords,
                "featurizers": {"gate": {"temperature": temperature}},
                "anneals": {"gate.theta.temperature": {"final": temperature - 0.01}},
            }
        )
    save_file(
        bundle.entries,
        str(root / "gate.safetensors"),
        metadata={"entries": json.dumps(bundle.entry_meta)},
    )
    diagnostics = root / "fit_diagnostics.json"
    diagnostics.write_text(json.dumps(rows))
    specs = {"gate": {"step": "fit", "file": "gate.safetensors", "kind": "gate"}}
    resolved = observation_specs(tmp_path, specs)
    assert {
        json.loads(key.split("/", 1)[1])["coords"]["target.layers"]: value[
            "temperature"
        ]
        for key, value in resolved.items()
        if "/" in key
    } == {12: 0.4, 20: 0.25, 28: 0.05}

    # A matching point digest cannot excuse conflicting coordinates.
    rows[0]["coords"]["sites.target.layers"] = 99
    diagnostics.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="one matching fit temperature"):
        observation_specs(tmp_path, specs)


def test_gate_temperature_rejects_ambiguous_features_even_at_same_temperature(tmp_path):
    root = tmp_path / "fit"
    root.mkdir()
    save_file(
        {"theta": torch.tensor([0.2])},
        str(root / "gate.safetensors"),
        metadata={
            "entries": json.dumps(
                {"theta": {"slot": "theta", "coords": {}, "produced_by": "point"}}
            )
        },
    )
    (root / "fit_diagnostics.json").write_text(
        json.dumps(
            [
                {
                    "point": "point",
                    "coords": {},
                    "featurizers": {
                        "first": {"temperature": 0.02},
                        "second": {"temperature": 0.02},
                    },
                }
            ]
        )
    )
    specs = {"gate": {"step": "fit", "file": "gate.safetensors", "kind": "gate"}}
    with pytest.raises(ValueError, match="featurizer selector"):
        observation_specs(tmp_path, specs)
    specs["gate"]["featurizer"] = "second"
    resolved = observation_specs(tmp_path, specs)
    assert (
        next(value for key, value in resolved.items() if "/" in key)["temperature"]
        == 0.02
    )


def test_ragged_reads_keep_logical_rows_and_only_real_positions(tmp_path):
    root = tmp_path / "inference"
    root.mkdir()
    save_file(
        {
            "logits": torch.arange(6).reshape(3, 2),
            "logits" + RAGGED_SUFFIX: torch.tensor([1, 2]),
        },
        str(root / "values.safetensors"),
        metadata={
            "entries": json.dumps(
                {"logits": {"slot": "logits", "coords": {"layer": 0}}}
            )
        },
    )
    specs = {
        "read": {"step": "inference", "file": "values.safetensors", "kind": "tensor"}
    }
    values = observations(tmp_path, specs)
    assert len(values) == 2
    assert [v.shape for _, v in sorted(values.items())] == [
        torch.Size([1, 2]),
        torch.Size([2, 2]),
    ]
    assert all("layer" in key and "/row_" in key for key in values)


def test_table_alignment_is_semantic_not_file_order(tmp_path):
    root = tmp_path / "eval"
    root.mkdir()
    path = root / "metric.json"
    rows = [
        {"example": "second", "metric": "effect", "value": 2.0},
        {"example": "first", "metric": "effect", "value": 1.0},
    ]
    spec = {
        "effect": {
            "step": "eval",
            "file": "metric.json",
            "kind": "table",
            "row_keys": ["example", "metric"],
            "value": "value",
        }
    }
    path.write_text(json.dumps(rows))
    first = observations(tmp_path, spec)
    path.write_text(json.dumps(list(reversed(rows))))
    second = observations(tmp_path, spec)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)
    path.write_text(json.dumps(rows + rows[:1]))
    with pytest.raises(ValueError, match="duplicate logical"):
        observations(tmp_path, spec)


def test_ineligible_observations_do_not_become_zero(tmp_path):
    root = tmp_path / "eval"
    root.mkdir()
    (root / "metric.json").write_text(
        json.dumps([{"example": 1, "value": None, "eligible": False}])
    )
    spec = {
        "effect": {
            "step": "eval",
            "file": "metric.json",
            "kind": "table",
            "row_keys": ["example"],
            "value": "value",
        }
    }
    with pytest.raises(ValueError, match="no eligible"):
        observations(tmp_path, spec)
