"""Timing-only receipts remain distinct from damaged numerical evidence."""

import json

import pytest

from causalab.measurement.collection import write_record

pytestmark = pytest.mark.unit


def timing_receipt(root):
    root.mkdir()
    path = root / "measurement.json"
    write_record(
        path,
        {
            "schema_version": 1,
            "mode": "single",
            "observation_policy": "not_requested",
            "status": "completed",
            "case": "workflow",
            "scope": "resident",
            "input_identity": "fixture",
            "reset_policy": "fresh",
            "alignment": "logical_keys_and_caller_aligned_tensor_coordinates",
            "provenance": {"implementation": {"tree_digest": "fixture"}},
            "plan": {"seeds": [0], "repeats": 1},
            "samples": [
                {
                    "seed": 0,
                    "repeat": 0,
                    "seconds": 1.0,
                    "observation_status": "not_requested",
                }
            ],
        },
    )
    return path


def test_generic_loader_accepts_timing_only_but_comparison_refuses(tmp_path):
    from causalab.measurement.analysis.receipts import load_measurement
    from causalab.measurement.analysis.compare import compare

    path = timing_receipt(tmp_path / "run")
    record, samples = load_measurement(path, require_observations=False)
    assert record["observation_policy"] == "not_requested"
    assert "tensors" not in samples[0, 0]
    with pytest.raises(ValueError, match="observations"):
        compare(path, path, bootstrap_draws=100)


@pytest.mark.parametrize("field", ["mode", "observation_policy"])
def test_missing_policy_cannot_hide_missing_observations(tmp_path, field):
    from causalab.measurement.analysis.receipts import load_measurement

    path = timing_receipt(tmp_path / "run")
    record = json.loads(path.read_text())
    del record[field]
    write_record(path, record)
    with pytest.raises(ValueError):
        load_measurement(path, require_observations=False)


def test_timing_only_rejects_conflicting_sample_evidence(tmp_path):
    from causalab.measurement.analysis.receipts import load_measurement

    path = timing_receipt(tmp_path / "run")
    record = json.loads(path.read_text())
    record["samples"][0]["observations"] = {"file": "fake", "sha256": "fake"}
    write_record(path, record)
    with pytest.raises(ValueError):
        load_measurement(path, require_observations=False)


def test_saved_output_hashes_are_verified_without_observations(tmp_path):
    from causalab.measurement.analysis.receipts import load_measurement
    from causalab.measurement.collection import file_hash

    path = timing_receipt(tmp_path / "run")
    output = path.parent / "output.json"
    output.write_text("{}")
    record = json.loads(path.read_text())
    record["samples"][0]["output_files"] = {output.name: file_hash(output)}
    write_record(path, record)
    load_measurement(path, require_observations=False)
    output.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        load_measurement(path, require_observations=False)
