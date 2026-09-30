"""Shared knee selection preserves scientific boundary decisions."""

import pytest

from causalab.analysis.selection import smallest_near_best

pytestmark = pytest.mark.unit


def test_smallest_rank_at_boundary_and_tied_fits():
    assert smallest_near_best([0.70, 0.72, 0.70], [2, 8, 2]) == [0, 2]
    assert smallest_near_best([0.69, 0.72, 0.71], [2, 8, 4]) == [2]


def test_unmeasured_scores_cannot_select_a_rank():
    assert smallest_near_best([float("nan"), 0.6], [1, 8]) == [1]
    with pytest.raises(ValueError, match="finite scores"):
        smallest_near_best([float("nan")], [1])


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan")])
def test_invalid_tolerance_is_refused(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        smallest_near_best([0.5], [2], tolerance=tolerance)
