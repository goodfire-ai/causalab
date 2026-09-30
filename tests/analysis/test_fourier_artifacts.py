"""Verify native Fourier files before downstream analysis consumes them."""

import json

import numpy as np
import pytest
import torch

from causalab.analysis.fit_fourier_probe import main
from causalab.analysis.fourier_artifacts import load_fit
from causalab.io.step_io import (
    StepError,
    read_tensor,
    stamp_tensor,
    write_table,
    write_tensor,
)
from tests.analysis.test_fourier_probe import fixture

pytestmark = pytest.mark.unit


@pytest.fixture
def saved(tmp_path):
    inputs = fixture()
    inputs["acts"] = np.concatenate((inputs["acts"], inputs["acts"] * 2), axis=1)
    inputs["position_labels"] = ["operand", "answer"]
    outputs = {
        name: tmp_path / f"{name}.safetensors"
        for name in ("weight", "bias", "plane", "calibration", "predictions")
    }
    outputs.update({name: tmp_path / f"{name}.json" for name in ("metadata", "scores")})
    main(inputs, outputs)
    write_table(tmp_path / "rows.json", inputs["rows"])
    write_tensor(
        tmp_path / "acts.safetensors", torch.from_numpy(inputs["acts"]), slot="acts"
    )
    return tmp_path


def load(directory):
    return load_fit(directory, directory / "acts.safetensors", directory / "rows.json")


def test_complete_native_fit(saved):
    loaded = load(saved)
    assert loaded["acts"].shape == (300, 2, 8)
    assert loaded["truth"].shape == (300, 3, 2)
    assert loaded["predictions"].shape == (300, 2, 3, 2)
    assert len(loaded["scores"]) == 6
    assert len(loaded["rows"]) == 300
    assert loaded["identity"] == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("example_ids", []),
        ("split_ids", {}),
        ("positions", 1),
        ("feature_width", 9),
        ("position_labels", ["operand"]),
        ("position_labels", ["operand", "operand"]),
        ("frequencies", [{"period": 10}]),
    ],
)
def test_refuses_inconsistent_metadata(saved, field, value):
    path = saved / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(StepError):
        load(saved)


def test_refuses_reordered_rows(saved):
    path = saved / "rows.json"
    rows = json.loads(path.read_text())
    rows[0], rows[1] = rows[1], rows[0]
    write_table(path, rows)
    with pytest.raises(StepError, match="rows differ"):
        load(saved)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "frequency", "rank"])
def test_refuses_inconsistent_scores(saved, problem):
    path = saved / "scores.json"
    scores = json.loads(path.read_text())
    if problem == "missing":
        scores.pop()
    elif problem == "duplicate":
        scores[-1] = scores[0]
    elif problem == "frequency":
        scores[0]["frequency"] = 99
    else:
        scores[0]["rank"] = 3
    write_table(path, scores)
    with pytest.raises(StepError):
        load(saved)


@pytest.mark.parametrize(
    "problem", ["shape", "replay", "calibration", "orthonormality", "identity", "model"]
)
def test_refuses_inconsistent_tensor_files(saved, problem):
    if problem == "identity":
        stamp_tensor(
            saved / "bias.safetensors", {"model_key": "test/another-fit"}, what="test"
        )
    elif problem == "model":
        for name in ("weight", "bias", "plane", "calibration", "predictions"):
            stamp_tensor(
                saved / f"{name}.safetensors",
                {"model_key": "test/model-a"},
                what="test",
            )
        stamp_tensor(
            saved / "acts.safetensors", {"model_key": "test/model-b"}, what="test"
        )
    else:
        name = {
            "shape": "weight",
            "replay": "acts",
            "calibration": "calibration",
            "orthonormality": "plane",
        }[problem]
        path = saved / f"{name}.safetensors"
        value = read_tensor(path)
        value = value[0] if problem == "shape" else value + 1
        write_tensor(path, value, slot=name)
    with pytest.raises(StepError):
        load(saved)
