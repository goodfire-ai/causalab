"""Native captures retain pairing and compare against the correct process state."""

import copy

import pytest
import torch
from safetensors.torch import save_file

from causalab.measurement.analysis.profiles import (
    capture_pairs,
    compare_captures,
    render_captures,
    validate_captures,
)
from causalab.measurement.collection import file_hash

pytestmark = pytest.mark.unit


def collection(tmp_path):
    def tensor(name, value):
        path = tmp_path / name
        save_file({"score": torch.tensor([value])}, str(path))
        return {"file": name, "sha256": file_hash(path)}

    cold = tensor("cold.safetensors", 2.0)
    warm = tensor("warm.safetensors", 1.0)
    samples = {
        (7, 0): {
            "unobserved_observations": cold,
            "unobserved_observation_specs": {"score": {"kind": "tensor"}},
            "resident_unobserved_observations": warm,
            "resident_unobserved_observation_specs": {"score": {"kind": "tensor"}},
            "tensors": {"score": torch.tensor([2.0])},
        }
    }
    captures = []
    for mode, observations in (("cold", cold), ("warm", warm)):
        path = tmp_path / f"{mode}.nsys-rep"
        path.write_bytes(b"opaque native trace")
        captures.append(
            {
                "id": f"nsys:{mode}",
                "pair_id": "nsys",
                "backend": "nsys",
                "mode": mode,
                "seed": 7,
                "repeat": 0,
                "status": "completed",
                "artifact_format": "nsys-rep",
                "artifacts": [{"file": path.name, "sha256": file_hash(path)}],
                "observations": observations,
                "observation_specs": {"score": {"kind": "tensor"}},
                "coverage": {"scope": mode},
            }
        )
    record = {
        "captures": captures,
        "capture_plan": {"backends": ["nsys"], "modes": ["cold", "warm"]},
        "observation_specs": {"score": {"kind": "tensor"}},
    }
    return record, samples


def test_cold_and_warm_use_distinct_clean_references(tmp_path):
    record, samples = collection(tmp_path)
    validate_captures(record, tmp_path, samples)
    results = compare_captures(record, tmp_path / "measurement.json", samples)
    for capture in results:
        assert capture["observation_check"]["exactly_equal"]
        assert capture["observation_check"]["drift"]["score"]["rms"] == 0
        assert capture["artifact_paths"]
    assert results[1]["comparison_reference"] == "clean resident timing-pass outputs"
    assert capture_pairs(record, results)[0]["status"] == "completed"
    page = render_captures(
        {
            "captures": {"before": results},
            "capture_pairs": {"before": capture_pairs(record, results)},
        }
    )
    assert "cold" in page and "warm" in page and ".nsys-rep" in page
    assert "not clean benchmark timings" in page


def test_missing_or_failed_pair_member_is_visible(tmp_path):
    record, samples = collection(tmp_path)
    record["captures"].pop()
    validate_captures(record, tmp_path, samples)
    pair = capture_pairs(record, record["captures"])[0]
    assert pair["modes"]["warm"] == "missing"
    assert pair["status"] == "incomplete"
    record["captures"].append(
        {
            "id": "nsys:warm",
            "pair_id": "nsys",
            "backend": "nsys",
            "mode": "warm",
            "seed": 7,
            "repeat": 0,
            "status": "unavailable",
            "error": "missing tool",
        }
    )
    validate_captures(record, tmp_path, samples)
    results = compare_captures(record, tmp_path / "measurement.json", samples)
    assert results[1]["error"] == "missing tool"
    assert capture_pairs(record, results)[0]["status"] == "incomplete"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["captures"].append(copy.deepcopy(r["captures"][0])), "duplicate"),
        (lambda r: r["captures"][0].update(seed=9), "clean sample"),
        (lambda r: r["captures"][0].update(artifacts=[]), "needs native artifacts"),
        (
            lambda r: r["captures"][0]["artifacts"][0].update(file="../escape"),
            "missing or escaping",
        ),
    ],
)
def test_invalid_capture_records_refused(tmp_path, mutation, match):
    record, samples = collection(tmp_path)
    mutation(record)
    with pytest.raises(ValueError, match=match):
        validate_captures(record, tmp_path, samples)


def test_native_artifact_corruption_is_refused(tmp_path):
    record, samples = collection(tmp_path)
    (tmp_path / "cold.nsys-rep").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_captures(record, tmp_path, samples)


def test_warm_capture_never_falls_back_to_cold_reference(tmp_path):
    record, samples = collection(tmp_path)
    del samples[7, 0]["resident_unobserved_observations"]
    results = compare_captures(record, tmp_path / "measurement.json", samples)
    assert results[1]["observation_check"]["status"] == "missing_clean_warm_reference"
    assert "exactly_equal" not in results[1]["observation_check"]


def test_dropped_observation_is_not_reported_as_equal(tmp_path):
    record, samples = collection(tmp_path)
    reference = tmp_path / "more.safetensors"
    save_file(
        {"score": torch.tensor([2.0]), "row2": torch.tensor([3.0])}, str(reference)
    )
    samples[7, 0]["unobserved_observations"] = {
        "file": reference.name,
        "sha256": file_hash(reference),
    }
    results = compare_captures(record, tmp_path / "measurement.json", samples)
    assert results[0]["observation_check"]["status"] == "unaligned"
    assert "exactly_equal" not in results[0]["observation_check"]


def test_cold_warm_pair_must_share_sample_identity(tmp_path):
    record, samples = collection(tmp_path)
    samples[8, 0] = samples[7, 0]
    record["captures"][1]["seed"] = 8
    with pytest.raises(ValueError, match="share a seed and repeat"):
        validate_captures(record, tmp_path, samples)
