"""Different fit configurations retain paired inputs and crossed saved-fit replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from causalab.measurement.analysis.compare import load_measurement
from causalab.measurement.study.controller import run
from causalab.protocol.schema import inline_train_saves
from tests.measurement.study.test_controller import _local_inputs

# `measurement_study`: this module builds and installs source arms and runs the
# study controller with cold worker processes — minutes each.
# `-m "not measurement_study"` deselects it for a quick run (docs/TESTS.md lists
# the markers).
pytestmark = [pytest.mark.smoke, pytest.mark.measurement_study]


def test_workflow_training_contrast_preserves_crossed_evaluation(tmp_path, monkeypatch):
    checkpoint, data = _local_inputs(tmp_path, monkeypatch)
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    rows = [
        {
            "input": base,
            "counterfactual_inputs": [source],
            "split": split,
            "label": source[-1],
            "base_answer": base[-1],
            "cf_answer": source[-1],
            "id": index,
        }
        for index, (base, source, split) in enumerate(
            [
                ("a b", "a c", "train"),
                ("b c", "b b", "train"),
                ("c b", "c c", "test"),
                ("c a", "c c", "test"),
            ]
        )
    ]
    (data / "pairs.json").write_text(json.dumps(rows))
    protocol = json.loads((repository / "demos/methods/protocols/das.json").read_text())
    protocol["model"] = {"key": str(checkpoint), "revision": "local", "dtype": "fp32"}
    for role in protocol["data"].values():
        role["dataset"] = "pairs#train"
    protocol["method"]["sites"]["target"]["layers"] = [0]
    protocol["method"]["featurizers"]["rot"]["k"] = 2
    train = protocol["method"]["train"]
    train["steps"] = {"updates": 1}
    train["batch"] = {"pairs": 2}
    train["eval"]["split"] = "pairs#test"
    train.pop("early_stop", None)
    train.pop("anneal", None)
    (tmp_path / "das-before.json").write_text(json.dumps(protocol))
    candidate = deepcopy(protocol)
    candidate["method"]["train"]["steps"]["updates"] = 2
    (tmp_path / "das-after.json").write_text(json.dumps(candidate))

    replay = deepcopy(protocol)
    replay["method"]["save"] = [
        value
        for value in inline_train_saves(replay["method"])
        if value.get("value") != "rot"
    ]
    replay["method"].pop("train")
    replay["method"]["featurizers"]["rot"]["file_path"] = "das/rot.safetensors"
    for role in replay["data"].values():
        role["dataset"] = "pairs#test"
    (tmp_path / "evaluate.json").write_text(json.dumps(replay))
    before = {
        "version": "1",
        "output_dir": "results",
        "steps": {
            "das": {"type": "intervention_protocol", "document": "das-before.json"},
            "evaluate": {"type": "intervention_protocol", "document": "evaluate.json"},
        },
        "measurement": {
            "version": 1,
            "comparison": "workflow",
            "arms": {
                "before": {"revision": revision},
                "after": {"revision": revision, "workflow": "after.json"},
            },
            "seeds": [7],
            "repeats": 1,
            "warmups": 0,
            "bootstrap_draws": 100,
            "profile": {"cases": []},
            "cases": {"training": {"kind": "workflow", "cold_process": False}},
            "observations": {
                "rotation": {
                    "step": "das",
                    "file": "rot.safetensors",
                    "kind": "subspace",
                },
                "effect": {
                    "step": "evaluate",
                    "file": "iia.json",
                    "kind": "table",
                    "row_keys": ["example_id", "metric"],
                    "value": "value",
                },
            },
            "evaluation": {
                "arm": "before",
                "crossed": True,
                "cases": {"training": ["evaluate"]},
            },
        },
    }
    after = deepcopy(before)
    after.pop("measurement")
    after["steps"]["das"]["document"] = "das-after.json"
    document = tmp_path / "before.json"
    document.write_text(json.dumps(before))
    (tmp_path / "after.json").write_text(json.dumps(after))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {"repository": str(repository), "python": sys.executable}
                    for arm in ("before", "after")
                },
                "device": "cpu",
                "data_root": str(data),
                "artifacts_root": str(tmp_path),
            }
        )
    )
    reports = run(document, bindings, tmp_path / "run")
    report = json.loads((reports / "training.json").read_text())
    paired = report["training"]["per_sample"][0]
    assert paired["status"] == "compared"
    paired_fit = paired["fits"][0]
    assert paired_fit["before"]["optimizer_steps"] == 1
    assert paired_fit["after"]["optimizer_steps"] == 2
    assert paired_fit["initial_parameters_match"]
    assert not paired_fit["optimizer_steps_match"]
    summary = report["comparison_summary"]
    assert summary["diagnostic_work"][0]["before_updates"] == 1
    assert summary["diagnostic_work"][0]["after_updates"] == 2
    assert any("changed training work" in caveat for caveat in summary["caveats"])
    receipts, fits, tensors = {}, {}, {}
    for arm, expected_updates in (("before", 1), ("after", 2)):
        receipt, samples = load_measurement(Path(report["sources"][arm]["file"]))
        receipts[arm] = receipt
        sample = samples[7, 0]
        fits[arm] = sample["numerics_context"]["fits"][0]
        assert fits[arm]["optimizer_steps"] == expected_updates
        assert len(fits[arm]["batches"]) == expected_updates
        assert (
            sample["evaluated_fit_origin"]
            == sample["observation_origin"]
            == "timing_pass"
        )
        assert sample["evaluated_fit_tree_sha256"]
        assert receipt["context"]["evaluation"]["crossed"]
        assert set(receipt["context"]["evaluation"]["evaluators"]) == {
            "before",
            "after",
        }
        tensors[arm] = sample["tensors"]
        for evaluator in ("before", "after"):
            assert any(
                name.startswith(f"evaluation__{evaluator}__effect/")
                for name in tensors[arm]
            )
    assert (
        fits["before"]["identity"]["protocol"] != fits["after"]["identity"]["protocol"]
    )
    assert (
        fits["before"]["initial"]["parameters"]
        == fits["after"]["initial"]["parameters"]
    )
    rotation = next(name for name in tensors["before"] if name.startswith("rotation/"))
    assert not tensors["before"][rotation].equal(tensors["after"][rotation])
    assert receipts["before"]["input_identity"] == receipts["after"]["input_identity"]
    workers = {arm: receipt["context"]["worker"] for arm, receipt in receipts.items()}
    assert (
        workers["before"]["source_commit"]
        == workers["after"]["source_commit"]
        == revision
    )
    assert (
        workers["before"]["benchmark_identity"]
        != workers["after"]["benchmark_identity"]
    )
    assert (
        workers["before"]["comparison_identity"]
        == workers["after"]["comparison_identity"]
    )
    assert (
        workers["before"]["shared_pins"]["datasets"]
        == workers["after"]["shared_pins"]["datasets"]
    )
