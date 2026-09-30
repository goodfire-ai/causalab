"""Recover known frequencies and preserve a frozen readout across populations."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge

from causalab.analysis.apply_fourier_probe import apply
from causalab.analysis.fit_fourier_probe import fit, frequencies, main, targets
from causalab.io.step_io import StepError, read_tensor, read_values


def fixture() -> dict[str, Any]:
    rng = np.random.default_rng(8)
    value = rng.permutation(np.arange(300))
    angle = 2 * np.pi * value / 10
    x = np.stack([np.cos(angle), np.sin(angle), (-1.0) ** value, value / 100], axis=1)
    x = x @ np.linalg.qr(rng.normal(size=(8, 4)))[0].T + 3
    rows = [
        {
            "id": str(i),
            "split": "train" if i < 180 else "validation" if i < 240 else "evaluation",
            "number": int(v),
        }
        for i, v in enumerate(value)
    ]
    return {
        "acts": x[:, None, :],
        "rows": rows,
        "target": "number",
        "periods": [2, 7, 10],
        "alphas": [0.001, 0.1, 10],
        "layer": 3,
    }


@pytest.mark.numerical_unit
def test_first_ten_natural_harmonics_preserve_both_periods():
    _, scores, metadata = fit(
        {**fixture(), "periods": [10, 7], "harmonics": list(range(1, 11))}
    )
    aliases = {
        (a["period"], a["harmonic"])
        for spec in metadata["frequencies"]
        for a in spec["aliases"]
    }
    assert aliases == {(float(p), h) for p in (10, 7) for h in range(1, 11)}
    assert len(metadata["frequencies"]) == 19
    decimal_third = next(s for s in scores if s["period"] == 10 and s["harmonic"] == 3)
    assert decimal_third["frequency"] == 3 / 10
    constant = next(s for s in scores if s["frequency"] == 1)
    assert constant["reason"] == "constant_training_targets"
    assert constant["evaluation"]["r2"] is None


@pytest.mark.numerical_unit
def test_known_frequency_rank_one_and_sklearn_oracle():
    inputs = fixture()
    tensors, scores, metadata = fit(inputs)
    assert scores[2]["evaluation"]["r2"] > 0.999
    assert scores[2]["evaluation"]["phase_mae"] < 0.001
    assert scores[2]["shuffled"]["r2"] < 0.2
    assert scores[1]["evaluation"]["r2"] < 0.2
    assert scores[0]["target_rank"] == scores[0]["rank"] == 1
    assert scores[0]["evaluation"]["sin_r2"] is None
    np.testing.assert_allclose(
        tensors["plane"] @ tensors["calibration"], tensors["weight"], atol=1e-12
    )
    y = targets([r["number"] for r in inputs["rows"]], metadata["frequencies"], 0)
    reference = Ridge(alpha=scores[2]["alpha"]).fit(inputs["acts"][:180, 0], y[:180, 2])
    np.testing.assert_allclose(
        reference.predict(inputs["acts"][:, 0]),
        tensors["predictions"][:, 0, 2],
        atol=1e-11,
    )


@pytest.mark.property
def test_evaluation_cannot_change_fit_and_frozen_apply_matches():
    inputs = fixture()
    original, scores, _ = fit(inputs)
    for row in inputs["rows"][240:]:
        row["number"] += 3
    inputs["acts"][240:] += 20
    changed, changed_scores, _ = fit(inputs)
    for key in ("weight", "bias", "plane", "calibration"):
        np.testing.assert_array_equal(original[key], changed[key])
    assert [s["alpha"] for s in scores] == [s["alpha"] for s in changed_scores]
    replay = apply(inputs["acts"], original["weight"], original["bias"])
    np.testing.assert_allclose(replay["predictions"], changed["predictions"])
    assert replay["phase"].min() >= 0
    assert replay["phase"].max() < 2 * np.pi


@pytest.mark.property
def test_das_coordinates_and_reconstruction_agree():
    inputs = fixture()
    q = np.linalg.qr(np.random.default_rng(2).normal(size=(8, 3)))[0]
    coordinates = inputs["acts"] @ q
    small, scores, _ = fit({**inputs, "acts": coordinates, "representation": "das"})
    full, full_scores, _ = fit({**inputs, "acts": coordinates @ q.T})
    np.testing.assert_allclose(small["predictions"], full["predictions"], atol=1e-10)
    assert [s["alpha"] for s in scores] == [s["alpha"] for s in full_scores]
    np.testing.assert_allclose(q @ small["weight"], full["weight"], atol=1e-10)


@pytest.mark.property
def test_origin_rotation_preserves_selection_and_joint_scores():
    inputs = fixture()
    _, scores, _ = fit(inputs)
    _, rotated, _ = fit({**inputs, "origin": 0.37})
    for a, b in zip(scores, rotated):
        assert a["alpha"] == b["alpha"]
        assert a["evaluation"]["r2"] == pytest.approx(b["evaluation"]["r2"], abs=1e-12)


@pytest.mark.numerical_unit
def test_short_arc_plane_preserves_the_fitted_readout():
    values = np.linspace(-1e-5, 1e-5, 90)
    x = targets(values, frequencies([10], [1]), 0)[:, 0]
    rows = [
        {
            "id": str(i),
            "split": ("train", "validation", "evaluation")[i % 3],
            "value": float(value),
        }
        for i, value in enumerate(values)
    ]
    tensors, scores, _ = fit(
        {"acts": x, "rows": rows, "target": "value", "periods": [10], "alphas": [1e-30]}
    )
    assert scores[0]["target_rank"] == 1
    assert scores[0]["rank"] == 2
    restored = tensors["plane"] @ tensors["calibration"]
    np.testing.assert_allclose(restored, tensors["weight"], atol=1e-12)
    np.testing.assert_allclose(
        apply(x, restored, tensors["bias"])["predictions"],
        tensors["predictions"],
        atol=1e-12,
    )


@pytest.mark.numerical_unit
@pytest.mark.parametrize("width", [1, 45])
def test_each_position_matches_an_independent_ridge_fit(width):
    rng = np.random.default_rng(42)
    values = rng.permutation(np.arange(60))
    x = rng.normal(size=(60, 3, width))
    x[:, 0, 0] = np.cos(2 * np.pi * values / 10)
    x[:, 1, 0] = np.sin(2 * np.pi * values / 7)
    x[:, 2] = 0
    rows = [
        {
            "id": str(i),
            "split": "train" if i < 36 else "validation" if i < 48 else "evaluation",
            "value": int(value),
        }
        for i, value in enumerate(values)
    ]
    tensors, scores, metadata = fit(
        {"acts": x, "rows": rows, "target": "value", "periods": [7, 10], "origin": 0.37}
    )
    y = targets(values, metadata["frequencies"], metadata["origin"])
    for score in scores:
        p, f = score["position"], score["frequency_index"]
        candidates = [
            Ridge(alpha=alpha, solver="svd").fit(x[:36, p], y[:36, f])
            for alpha in metadata["alphas"]
        ]
        reference = min(
            candidates,
            key=lambda model: np.mean((model.predict(x[36:48, p]) - y[36:48, f]) ** 2),
        )
        assert score["alpha"] == reference.alpha
        np.testing.assert_allclose(
            tensors["predictions"][:, p, f], reference.predict(x[:, p]), atol=1e-11
        )
    np.testing.assert_allclose(
        apply(x, tensors["weight"], tensors["bias"])["predictions"],
        tensors["predictions"],
    )


@pytest.mark.numerical_unit
def test_broad_scan_and_saved_fit_replay(tmp_path):
    inputs = fixture()
    del inputs["periods"]
    out = {
        key: tmp_path / f"{key}.safetensors"
        for key in ("weight", "bias", "plane", "calibration", "predictions")
    }
    out.update({key: tmp_path / f"{key}.json" for key in ("scores", "metadata")})
    main(inputs, out)
    assert len(read_values(out["metadata"])["frequencies"]) == 149
    readout = apply(inputs["acts"], out["weight"], out["bias"])
    np.testing.assert_allclose(
        readout["predictions"], read_tensor(out["predictions"]).numpy()
    )


@pytest.mark.unit
def test_aliases_constants_and_undefined_phase():
    specs = frequencies([10, 20], [1, 2])
    assert len(specs) == 3
    assert {r["period"] for r in specs[0]["aliases"]} == {10, 20}
    inputs = fixture()
    _, scores, _ = fit({**inputs, "periods": [1]})
    assert scores[0]["reason"] == "constant_training_targets"
    assert scores[0]["evaluation"]["r2"] is None
    output = apply(np.ones((2, 3)), np.zeros((1, 1, 3, 2)), np.zeros((1, 1, 2)))
    assert not output["phase_defined"].any()


@pytest.mark.unit
@pytest.mark.parametrize(
    "problem",
    ["ids", "split", "labels", "nan", "period", "harmonic", "alpha", "positions"],
)
def test_invalid_inputs(problem):
    inputs = deepcopy(fixture())
    if problem == "ids":
        inputs["rows"][0]["id"] = inputs["rows"][1]["id"]
    elif problem == "split":
        inputs["rows"][0]["split"] = "test"
    elif problem == "labels":
        inputs["rows"][0]["number"] = "ten"
    elif problem == "nan":
        inputs["acts"][0, 0, 0] = np.nan
    else:
        inputs.update(
            {
                "period": {"periods": [0]},
                "harmonic": {"harmonics": [0]},
                "alpha": {"alphas": [-1]},
                "positions": {"position_labels": []},
            }[problem]
        )
    with pytest.raises(StepError):
        fit(inputs)


@pytest.mark.unit
def test_apply_refuses_mismatched_axes():
    with pytest.raises(StepError):
        apply(torch.zeros(2, 3), torch.zeros(1, 4, 8, 2), torch.zeros(1, 4, 2))
