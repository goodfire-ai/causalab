"""``fig4_figure.py``: the drawn values and the 3D views."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.demos.papers._scripts import load_script
from tests.demos.papers.workflows.scripts.manifold_fig4._fixtures import (
    CLASSES,
    behavior_tables,
    steer_tables,
    synthetic_activations,
)

pytestmark = pytest.mark.numerical_unit

figure = load_script("manifold_fig4", "fig4_figure")
behavior = load_script("manifold_fig4", "behavior_manifold")
manifold_path = load_script("manifold_fig4", "manifold_path")
energy = load_script("manifold_fig4", "energy")
PAIR = "Tuesday_Friday"


def _run_tree(tmp_path: Path) -> dict[str, Path]:
    """The figure's inputs: the manifold_path, behavior and energy steps run
    on synthetic activations, the oracle baseline p and synthetic steering
    tables."""
    act = tmp_path / "manifold_path"
    act.mkdir()
    act_outputs = {
        f"{prefix}_{name}": act / f"{prefix}_{name}.safetensors"
        for name in manifold_path.path_names()
        for prefix in ("waypoints", "chord")
    } | {key: act / f"{key}.json" for key in ("centroids", "points", "spline", "path")}
    manifold_path.main({**synthetic_activations(act), "steps": 3}, act_outputs)
    beh = tmp_path / "behavior"
    beh.mkdir()
    fitted = behavior_tables(beh)
    steer = tmp_path / "steer"
    steer.mkdir()
    energy_outputs = {
        key: tmp_path / f"{key}.json" for key in ("trajectory", "energy", "summary")
    }
    energy.main(
        {
            **steer_tables(steer, [PAIR], np.random.default_rng(5)),
            "behavior_centroids": fitted["centroids"],
            "behavior_spline": fitted["spline"],
        },
        energy_outputs,
    )
    return {
        "pair": PAIR,
        **{key: act_outputs[key] for key in ("centroids", "points", "spline", "path")},
        "behavior_centroids": fitted["centroids"],
        "behavior_points": fitted["points"],
        "behavior_spline": fitted["spline"],
        "trajectory": energy_outputs["trajectory"],
        "summary": energy_outputs["summary"],
    }


def test_the_plotted_values_are_the_papers_panels(tmp_path: Path) -> None:
    """``fig4_plotted.json`` holds what the figure draws. The behavior panel
    is ``sqrt(p)`` projected on the top three principal components of the
    prompts' ``sqrt(p)`` (the paper's ``plot_3d`` with
    ``feature_kind="hellinger"``), and each trajectory row carries the band
    half-widths of the energy step."""
    from causalab.io.step_io import read_table

    inputs = _run_tree(tmp_path)
    plotted = tmp_path / "fig4_plotted.json"
    figure.main(inputs, {"plotted": plotted})
    drawn = json.loads(plotted.read_text())

    p = np.array(
        [[r[c] for c in CLASSES] for r in read_table(inputs["behavior_points"])]
    )
    mean, components = behavior.hellinger_pca(p, 3)
    points = np.array([[r["x"], r["y"], r["z"]] for r in drawn["behavior"]["points"]])
    assert points == pytest.approx((np.sqrt(p) - mean) @ components.T, abs=1e-12)
    b = np.array(
        [[r[c] for c in CLASSES] for r in read_table(inputs["behavior_centroids"])]
    )
    centroids = np.array(
        [[r["x"], r["y"], r["z"]] for r in drawn["behavior"]["centroids"]]
    )
    assert centroids == pytest.approx((np.sqrt(b) - mean) @ components.T, abs=1e-12)

    trajectory = [r for r in read_table(inputs["trajectory"]) if r["pair"] == PAIR]
    assert len(drawn["trajectory"]) == len(trajectory) == 6
    key = lambda r: (r["method"], r["step"])  # noqa: E731
    for ours, theirs in zip(
        sorted(drawn["trajectory"], key=key), sorted(trajectory, key=key)
    ):
        for c in CLASSES:
            assert ours[c] == theirs[c]
            assert ours[f"band_{c}"] == theirs[f"band_{c}"]


def test_the_3d_panels_have_equal_units() -> None:
    """Each axis spans the drawn points, and the box aspect is proportional
    to the spans, so one unit is as long on every axis."""
    drawn = np.array([[0.0, -1.0, 2.0], [4.0, 1.0, 2.5], [1.0, 0.0, 3.0]])
    low, high, box = figure.equal_units(drawn)
    assert low == pytest.approx([0.0, -1.0, 2.0])
    assert high == pytest.approx([4.0, 1.0, 3.0])
    assert box / box[0] == pytest.approx((high - low) / (high - low)[0])


def test_the_view_looks_across_the_chord() -> None:
    """The 3D view's azimuth is perpendicular to the chord, so the chord is
    drawn at its full horizontal length."""
    start, stop = np.array([1.0, 2.0, 0.5]), np.array([4.0, -1.0, 0.0])
    azim = np.deg2rad(figure.view_azimuth(start, stop))
    sight = np.array([np.cos(azim), np.sin(azim)])
    assert sight @ (stop - start)[:2] == pytest.approx(0.0, abs=1e-12)
