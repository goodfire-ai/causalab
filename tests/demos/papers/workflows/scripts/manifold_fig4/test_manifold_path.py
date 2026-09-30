"""``manifold_path.py``: the paper code's centroid pairs on the activation manifold."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from tests.demos.papers._scripts import load_script
from tests.demos.papers.workflows.scripts.manifold_fig4._fixtures import (
    DAYS,
    TWO_PI,
    synthetic_activations,
)

pytestmark = pytest.mark.numerical_unit

manifold_path = load_script("manifold_fig4", "manifold_path")
STEPS = 5


def _run(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    inputs = {**synthetic_activations(tmp_path), "steps": STEPS}
    out = tmp_path / "out"
    outputs = {}
    for name in manifold_path.path_names():
        for prefix in ("waypoints", "chord"):
            outputs[f"{prefix}_{name}"] = out / f"{prefix}_{name}.safetensors"
    for key in ("centroids", "points", "spline", "path"):
        outputs[key] = out / f"{key}.json"
    manifold_path.main(inputs, outputs)
    return inputs, outputs


def _day_means(
    inputs: dict[str, Path],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Each day's mean activation, the PCA basis and its mean, from the inputs."""
    from causalab.io.tensor_files import load_file

    acts = load_file(str(inputs["acts"]))["acts"].double().numpy()[:, 0, :]
    labels = [row["result"] for row in json.loads(inputs["table"].read_text())]
    means = {d: acts[[label == d for label in labels]].mean(axis=0) for d in DAYS}
    basis = load_file(str(inputs["weight"]))["weight"].double().numpy()
    mean = load_file(str(inputs["mean"]))["mean"].double().numpy()
    return means, basis, mean


def test_the_step_writes_the_codes_centroid_pairs(tmp_path: Path) -> None:
    """The step writes the 21 centroid pairs in ``itertools.combinations``
    order, the default path set of the paper's code, and each manifold path
    runs from one knot to the other."""
    from causalab.io.step_io import read_table

    names = manifold_path.path_names()
    assert names == [
        f"{DAYS[i]}_{DAYS[j]}" for i, j in itertools.combinations(range(7), 2)
    ]
    _, outputs = _run(tmp_path)
    centroids = {r["day"]: r for r in read_table(outputs["centroids"])}
    path = read_table(outputs["path"])
    assert [r["pair"] for r in path if r["step"] == 0] == names
    first = {r["pair"]: r for r in path if r["step"] == 0}
    last = {r["pair"]: r for r in path if r["step"] == STEPS - 1}
    for pair in names:
        a, b = pair.split("_")
        for row, day in ((first[pair], a), (last[pair], b)):
            gap = ((row["theta"] - centroids[day]["theta"] + np.pi) % TWO_PI) - np.pi
            assert abs(gap) < 1e-9


def test_the_chord_is_one_point_per_waypoint(tmp_path: Path) -> None:
    """The chord bundle holds the point ``(1 - t) c_a + t c_b`` of the full
    residual stream at each waypoint, ``c`` a day's mean activation, so the
    linear write is one swap and one cast to the model's dtype, as the
    paper's code writes its chord (``path_mode._build_linear_path_kd``)."""
    from causalab.io.tensor_files import load_file

    inputs, outputs = _run(tmp_path)
    means, _, _ = _day_means(inputs)
    for pair in manifold_path.path_names():
        a, b = pair.split("_")
        bundle = load_file(str(outputs[f"chord_{pair}"]))
        assert sorted(bundle) == sorted(f"value[step={i}]" for i in range(STEPS))
        for step in range(STEPS):
            t = step / (STEPS - 1)
            expected = (1.0 - t) * means[a] + t * means[b]
            point = bundle[f"value[step={step}]"]
            assert str(point.dtype) == "torch.float32"
            assert point.double().numpy() == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_the_manifold_waypoints_are_feature_vectors(tmp_path: Path) -> None:
    """A manifold waypoint is the spline point in the ``pca`` featurizer's
    coordinates ``Pᵀ x``: at ``t = 0`` and ``t = 1`` it is ``Pᵀ`` of the
    endpoint's centroid, projected into the subspace."""
    from causalab.io.tensor_files import load_file

    inputs, outputs = _run(tmp_path)
    means, basis, mean = _day_means(inputs)
    for pair in ("Monday_Tuesday", "Tuesday_Friday", "Friday_Sunday"):
        a, b = pair.split("_")
        bundle = load_file(str(outputs[f"waypoints_{pair}"]))
        for step, day in ((0, a), (STEPS - 1, b)):
            expected = basis.T @ mean + (means[day] - mean) @ basis
            point = bundle[f"value[step={step}]"].double().numpy()
            assert point == pytest.approx(expected, rel=1e-6, abs=1e-6)
