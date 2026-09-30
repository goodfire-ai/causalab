"""``splines.py``: the knot and direction rules of the paper's code."""

from __future__ import annotations

import numpy as np
import pytest

from tests.demos.papers._scripts import load_script
from tests.demos.papers.workflows.scripts.manifold_fig4._fixtures import (
    ORACLES,
    TWO_PI,
    ring_gaps,
)

pytestmark = pytest.mark.numerical_unit

splines = load_script("manifold_fig4", "splines")


def test_the_activation_angle_is_the_paper_codes() -> None:
    """The knots of the seven layer-28 centroids equal the paper pipeline's
    checkpoint to 1e-4 rad (``remap_periodic_to_angle``). Plain
    ``atan2(PC2, PC1)``, the text's rule (App. A.3), misses them by more
    than 0.1 rad."""
    centroids = np.array(ORACLES["activation_centroids_pc12"])
    ours = ring_gaps(splines.periodic_angle(centroids))
    paper = ring_gaps(np.array(ORACLES["activation_checkpoint_control_points"]))
    assert np.abs(ours - paper).max() < 1e-4
    plain = ring_gaps(np.arctan2(centroids[:, 1], centroids[:, 0]))
    assert np.abs(plain - paper).max() > 0.1


def test_the_path_direction_is_the_wrapped_angle_difference() -> None:
    """``path_mode._build_geodesic_path`` wraps the difference into
    ``[-pi, pi)``, so a path takes the shorter way round the ring."""
    assert splines.wrapped_delta(0.1, 0.4) == pytest.approx(0.3)
    assert splines.wrapped_delta(0.1, TWO_PI - 0.2) == pytest.approx(-0.3)
    assert splines.wrapped_delta(TWO_PI - 0.2, 0.1) == pytest.approx(0.3)
    assert splines.wrapped_delta(0.0, np.pi) == pytest.approx(-np.pi)
