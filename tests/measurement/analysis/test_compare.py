"""Hand-computed drift/variance, alignment, and provenance integrity oracles."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from causalab.measurement.analysis.compare import compare, main, render_report
from causalab.measurement.collection import file_hash, write_record

pytestmark = pytest.mark.unit


def receipt(root: Path, values, *, seconds=None):
    root.mkdir()
    samples = []
    for seed, repeats in enumerate(values):
        for repeat, value in enumerate(repeats):
            file = root / f"s{seed}r{repeat}.safetensors"
            tensors = (
                value
                if isinstance(value, dict)
                else {"example/site": torch.tensor(value, dtype=torch.float64)}
            )
            save_file(tensors, str(file))
            samples.append(
                {
                    "seed": seed,
                    "repeat": repeat,
                    "seconds": seconds[seed][repeat] if seconds else 1.0,
                    "observations": {"file": file.name, "sha256": file_hash(file)},
                }
            )
    path = root / "measurement.json"
    write_record(
        path,
        {
            "schema_version": 1,
            "status": "completed",
            "case": "test",
            "input_identity": "logical-inputs-v1",
            "scope": "operation",
            "reset_policy": "fresh",
            "alignment": "logical_keys_and_caller_aligned_tensor_coordinates",
            "provenance": {"implementation": {"tree_digest": "fixture"}},
            "plan": {"seeds": list(range(len(values))), "repeats": len(values[0])},
            "samples": samples,
            "trace": {"status": "not_requested"},
        },
    )
    return path


def test_subspace_report_uses_projection_operators(tmp_path):
    q = torch.eye(4, dtype=torch.float64)[:, :2].contiguous()
    rotated = q[:, [1, 0]].contiguous()
    before = receipt(tmp_path / "b", [[{"basis/weight": q}, {"basis/weight": q}]])
    after = receipt(tmp_path / "a", [[{"basis/weight": -q}, {"basis/weight": rotated}]])
    for path in (before, after):
        record = json.loads(path.read_text())
        record["observation_specs"] = {"basis": {"kind": "subspace"}}
        write_record(path, record)
    report = compare(before, after, bootstrap_draws=100)
    value = report["observations"]["basis/weight"]
    assert value["kind"] == "subspace"
    assert value["within_seed"][0]["dispersion_change"] == pytest.approx(0)
    assert all(pair["overlap"] == pytest.approx(1) for pair in value["paired_drift"])
    assert "projector dispersion" in render_report(report)


def test_gate_report_separates_confidence_from_mask_agreement(tmp_path):
    theta = torch.tensor([-1.0, 1.0])
    values = [[{"gate/theta": theta}, {"gate/theta": -theta}]]
    before, after = receipt(tmp_path / "b", values), receipt(tmp_path / "a", values)
    for path, temperature in ((before, 2.0), (after, 0.1)):
        record = json.loads(path.read_text())
        record["observation_specs"] = {
            "gate/theta": {"kind": "gate", "temperature": temperature}
        }
        write_record(path, record)
    report = compare(before, after, bootstrap_draws=100)
    value = report["observations"]["gate/theta"]
    assert value["variance_units"] == "gate probabilities"
    assert all(pair["mask_jaccard"] == 1 for pair in value["paired_drift"])
    assert all(pair["probability_rms_difference"] > 0 for pair in value["paired_drift"])
    assert value["selection_frequency"] == {"before": [0.5, 0.5], "after": [0.5, 0.5]}
    assert value["within_seed"][0]["variance_ratio"] > 1
    assert "effectiveness" in value


def test_gate_variance_uses_each_repetition_temperature(tmp_path):
    values = [[{"gate/theta": torch.tensor([1.0])}] * 2] * 2
    before, after = receipt(tmp_path / "b", values), receipt(tmp_path / "a", values)
    for path in (before, after):
        record = json.loads(path.read_text())
        for sample in record["samples"]:
            sample["observation_specs"] = {
                "gate/theta": {"kind": "gate", "temperature": sample["repeat"] + 1.0}
            }
        write_record(path, record)
    value = compare(before, after, bootstrap_draws=100)["observations"]["gate/theta"]
    expected = float(
        (
            torch.sigmoid(torch.tensor(1.0, dtype=torch.float64))
            - torch.sigmoid(torch.tensor(0.5, dtype=torch.float64))
        )
        ** 2
        / 2
    )
    assert value["within_seed"][0]["before_mean_coordinate_variance"] == pytest.approx(
        expected
    )
    assert value["within_seed"][0]["variance_ratio"] == 1
    assert all(row["probability_rms_difference"] == 0 for row in value["paired_drift"])


def test_variance_increase_with_unchanged_mean(tmp_path):
    before = receipt(tmp_path / "b", [[[1.0], [3.0]], [[11.0], [13.0]]])
    after = receipt(tmp_path / "a", [[[0.0], [4.0]], [[10.0], [14.0]]])
    result = compare(before, after, bootstrap_draws=100)
    obs = result["observations"]["example/site"]
    for row in obs["within_seed"]:
        assert row["before_mean_coordinate_variance"] == 2
        assert row["after_mean_coordinate_variance"] == 8
        assert row["variance_change"] == 6
        assert row["variance_ratio"] == 4
        assert row["mean_output_drift"]["rms"] == 0
        assert row["before_mean"] == row["after_mean"]
        assert row["before_standard_deviation"] == pytest.approx(2**0.5)
        assert row["after_standard_deviation"] == pytest.approx(8**0.5)
        assert len(row["before_values"]) == len(row["after_values"]) == 2
    assert obs["across_seed_means"]["count"] == 2
    assert obs["across_seed_means"]["before_mean_coordinate_variance"] == 50
    assert obs["mean_within_seed_variance_change_ci"]["interval"] == [6, 6]
    assert all(p["rms"] == 1 for p in obs["paired_drift"])


def test_profiled_gate_uses_the_unobserved_pass_temperature(tmp_path):
    paths = [
        receipt(tmp_path / arm, [[[0.5], [0.5]], [[0.5], [0.5]]])
        for arm in ("before", "after")
    ]
    for path in paths:
        record = json.loads(path.read_text())
        record["observation_specs"] = {
            "example/site": {"kind": "gate", "temperature": 3.0}
        }
        for sample in record["samples"]:
            sample["unobserved_observations"] = dict(sample["observations"])
            sample["unobserved_observation_specs"] = {
                "example/site": {"kind": "gate", "temperature": 1.0}
            }
        trace = path.parent / "trace.json"
        trace.write_text("{}")
        record["trace"] = {
            "status": "completed",
            "file": trace.name,
            "sha256": file_hash(trace),
            "seed": 0,
            "repeat": 0,
            "observations": record["samples"][0]["observations"],
            "observation_specs": {"example/site": {"kind": "gate", "temperature": 2.0}},
        }
        write_record(path, record)
    result = compare(*paths, bootstrap_draws=100)
    expected = float(
        torch.sigmoid(torch.tensor(0.5, dtype=torch.float64))
        - torch.sigmoid(torch.tensor(0.25, dtype=torch.float64))
    )
    for trace in result["traces"].values():
        assert (
            trace["comparison_reference"]
            == "required outputs of the uninstrumented timing pass"
        )
        assert trace["observation_check"]["drift"]["example/site"][
            "probability_rms_difference"
        ] == pytest.approx(expected)


@pytest.mark.parametrize(
    "field",
    [
        "unobserved_observations",
        "diagnostic_observations",
        "resident_diagnostic_observations",
    ],
)
def test_unobserved_reference_bytes_are_verified(tmp_path, field):
    before = receipt(tmp_path / "before", [[[1.0]]])
    after = receipt(tmp_path / "after", [[[1.0]]])
    record = json.loads(before.read_text())
    record["samples"][0][field] = {
        **record["samples"][0]["observations"],
        "sha256": "changed",
    }
    write_record(before, record)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        compare(before, after, bootstrap_draws=100)


def test_shift_is_not_variance_and_argmax_is_not_equivalence(tmp_path):
    before = receipt(tmp_path / "b", [[[10.0, 1], [10.0, 1]]])
    after = receipt(tmp_path / "a", [[[12.0, 3], [12.0, 3]]])
    result = compare(before, after)
    row = result["observations"]["example/site"]["within_seed"][0]
    assert row["variance_change"] == 0
    assert row["variance_ratio"] is None
    assert row["mean_output_drift"]["rms"] == 2
    assert result["timing"]["mean_paired_seconds_change_ci"]["interval"] is None


def test_single_repeat_and_zero_vectors_are_explicit(tmp_path):
    before = receipt(tmp_path / "b", [[[0.0, 0]]])
    after = receipt(tmp_path / "a", [[[1.0, 1]]])
    result = compare(before, after)
    row = result["observations"]["example/site"]["within_seed"][0]
    assert row["before_mean_coordinate_variance"] is None
    assert row["variance_change"] is None
    assert row["mean_output_drift"]["relative_rms"] is None
    assert row["mean_output_drift"]["cosine_distance"] is None
    json.dumps(result, allow_nan=False)


def test_semantic_key_order_does_not_change_pairing(tmp_path):
    one = {"example_a/site": torch.tensor([1.0]), "example_b/site": torch.tensor([2.0])}
    two = dict(reversed(list(one.items())))
    before = receipt(tmp_path / "b", [[one]])
    after = receipt(tmp_path / "a", [[two]])
    result = compare(before, after)
    assert all(
        o["paired_drift"][0]["max_abs"] == 0 for o in result["observations"].values()
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["samples"].append(r["samples"][0]), "duplicate"),
        (lambda r: r["samples"].pop(), "missing"),
        (lambda r: r.update(scope="different boundary"), "incomparable scope"),
        (
            lambda r: r.update(input_identity="different data"),
            "incomparable input_identity",
        ),
        (lambda r: r.update(status="failed"), "completed"),
        (lambda r: r["samples"][0].update(seconds=float("nan")), "finite"),
    ],
)
def test_invalid_evidence_is_refused(tmp_path, mutation, match):
    before = receipt(tmp_path / "b", [[[1.0], [2.0]]])
    after = receipt(tmp_path / "a", [[[1.0], [2.0]]])
    record = json.loads(after.read_text())
    mutation(record)
    after.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        compare(before, after)


@pytest.mark.parametrize(
    "value,match",
    [
        ([float("nan")], "nonfinite"),
        ([1.0, 2.0], "shape"),
        ({"wrong-key": torch.tensor([1.0])}, "logical observation keys"),
    ],
)
def test_invalid_tensor_alignment_or_values(tmp_path, value, match):
    before = receipt(tmp_path / "b", [[[1.0]]])
    after = receipt(tmp_path / "a", [[value]])
    with pytest.raises(ValueError, match=match):
        compare(before, after)


def test_artifact_tampering_is_detected(tmp_path):
    before = receipt(tmp_path / "b", [[[1.0]]])
    after = receipt(tmp_path / "a", [[[1.0]]])
    (after.parent / "s0r0.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare(before, after)


def test_bootstrap_uses_paired_seed_effects_and_runtime_variance(tmp_path):
    values = [[[1.0], [2.0]], [[3.0], [4.0]]]
    before = receipt(tmp_path / "b", values, seconds=[[1, 3], [1, 3]])
    after = receipt(tmp_path / "a", values, seconds=[[2, 6], [2, 6]])
    result = compare(before, after, bootstrap_draws=100)
    assert result["timing"]["mean_paired_seconds_change_ci"]["interval"] == [2, 2]
    for row in result["timing"]["per_seed"]:
        assert row["runtime_variance_ratio"] == 4
        assert row["median_speedup"] == 0.5


def test_workflow_script_writes_standalone_report(tmp_path):
    before = receipt(tmp_path / "b", [[[1.0], [2.0]]])
    after = receipt(tmp_path / "a", [[[2.0], [3.0]]])
    outputs = {"summary": tmp_path / "summary.json", "report": tmp_path / "report.html"}
    main(
        {
            "before": before,
            "after": after,
            "before_sha256": file_hash(before),
            "after_sha256": file_hash(after),
        },
        outputs,
    )
    assert json.loads(outputs["summary"].read_text())["acceptance"] == "not_evaluated"
    assert "<!doctype html>" in outputs["report"].read_text()


def test_workflow_input_hash_pins_refuse_replaced_receipt(tmp_path):
    before = receipt(tmp_path / "b", [[[1.0]]])
    after = receipt(tmp_path / "a", [[[2.0]]])
    inputs = {
        "before": before,
        "after": after,
        "before_sha256": file_hash(before),
        "after_sha256": file_hash(after),
    }
    after.write_text(after.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        main(
            inputs,
            {"summary": tmp_path / "summary.json", "report": tmp_path / "report.html"},
        )
    assert not (tmp_path / "summary.json").exists()
