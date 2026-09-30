"""``behavior_manifold.py``: p(x), the behavior manifold and its distance."""

from __future__ import annotations

import numpy as np
import pytest

from tests.demos.papers._scripts import load_script
from tests.demos.papers.workflows.scripts.manifold_fig4._fixtures import (
    DAYS,
    ORACLES,
    TWO_PI,
    ring_gaps,
)

pytestmark = pytest.mark.numerical_unit

behavior = load_script("manifold_fig4", "behavior_manifold")
splines = load_script("manifold_fig4", "splines")


def _fitted():
    p = np.array(ORACLES["baseline_p"])
    return p, behavior.BehaviorManifold.fit(p, ORACLES["baseline_labels"])


def test_the_behavior_angle_is_the_paper_codes() -> None:
    """The behavior knots, from a Hellinger PCA of the 49 per-prompt
    ``sqrt(p)`` and the normalized angle, equal the paper pipeline's
    behavior checkpoint to 1e-4 rad (``fit_belief_tps_pca``)."""
    _, fitted = _fitted()
    ours = ring_gaps(fitted.theta)
    paper = ring_gaps(np.array(ORACLES["behavior_checkpoint_control_points"]))
    assert np.abs(ours - paper).max() < 1e-4


def test_the_manifold_passes_through_the_centroids() -> None:
    """Decoding a knot gives its centroid, and a point on the manifold is at
    distance zero under the code's Gauss-Newton projection."""
    _, fitted = _fitted()
    decoded = fitted.decode(fitted.theta) ** 2
    assert np.abs(decoded - fitted.centroids).max() < 1e-7
    on = fitted.decode(np.linspace(0.05, TWO_PI - 0.05, 40)) ** 2
    assert fitted.distance(on).max() < 1e-8


def test_the_projection_distance_is_the_bhattacharyya_distance() -> None:
    """``D_B = -log(1 - d_H^2)`` equals ``-log(sum sqrt(p q))`` at the
    projected point ``q``. The projection is a local minimum: it is never
    nearer than the nearest of 10500 dense samples (the global minimum up to
    their spacing), and it equals that minimum for most points."""
    p0, fitted = _fitted()
    rng = np.random.default_rng(1)
    p = 0.2 * rng.dirichlet(np.ones(8), size=30) + 0.8 * p0[rng.integers(0, 49, 30)]
    ours = fitted.distance(p)
    dense = splines.bhattacharyya_to_set(p, fitted.sample(10500))
    assert np.all(ours >= dense - 1e-6)
    assert np.median(ours - dense) < 1e-6
    projected = fitted.decode(fitted.nearest(p))
    h = np.sqrt(p) / np.linalg.norm(np.sqrt(p), axis=1, keepdims=True)
    assert ours == pytest.approx(-np.log((h * projected).sum(axis=1)), abs=1e-9)


def test_p_sums_every_spelling() -> None:
    """``p(x)`` sums the space-prefixed, the bare and the lowercase mass of
    each day (App. A.2), and ``other`` takes the rest. The lowercase table
    names only the days whose spelling is one token."""
    key = "example_id"
    space = [{key: "0", "value": {d: 0.1 for d in DAYS}}]
    bare = [{key: "0", "value": {d: 0.01 for d in DAYS}}]
    lower = [{key: "0", "value": {"Monday": 0.05, "Friday": 0.02, "Sunday": 0.01}}]
    p = behavior.probabilities_by_example([space, bare, lower])[("0",)]
    expected = np.array([0.16, 0.11, 0.11, 0.11, 0.13, 0.11, 0.12])
    assert p[:7] == pytest.approx(expected)
    assert p[7] == pytest.approx(1 - expected.sum())


def test_the_hellinger_pca_is_the_pca_of_sqrt_p() -> None:
    """``hellinger_pca`` is the PCA of the per-prompt ``sqrt(p)`` that both the
    knots and the figure's behavior panel use (the paper's ``plot_3d`` with
    ``feature_kind="hellinger"``)."""
    p = np.array(ORACLES["baseline_p"])
    mean, components = behavior.hellinger_pca(p, 3)
    x = np.sqrt(p) - np.sqrt(p).mean(axis=0)
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    assert mean == pytest.approx(np.sqrt(p).mean(axis=0))
    for k in range(3):
        assert abs(components[k] @ vh[k]) == pytest.approx(1.0)
    projected = (np.sqrt(p) - mean) @ components.T
    assert projected.var(axis=0, ddof=1) == pytest.approx(s[:3] ** 2 / 48)
