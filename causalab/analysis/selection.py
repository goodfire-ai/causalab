"""Numerical selection rules shared by workflows and saved-result consumers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def smallest_near_best(
    scores: Sequence[float], costs: Sequence[float], *, tolerance: float = 0.02
) -> list[int]:
    """Return all least-cost indices within tolerance of the best finite score.

    Missing scores cannot select a candidate. The absolute boundary tolerance is
    1e-12, so floating-point subtraction does not change a boundary decision.
    Input order determines the order of tied indices.
    """
    if len(scores) != len(costs):
        raise ValueError("scores and costs must have the same length")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    valid = [
        i
        for i, score in enumerate(scores)
        if score is not None and math.isfinite(score)
    ]
    if not valid or any(not math.isfinite(costs[i]) for i in valid):
        raise ValueError("selection requires finite scores and costs")
    boundary = max(scores[i] for i in valid) - tolerance
    near = [
        i
        for i in valid
        if scores[i] >= boundary
        or math.isclose(scores[i], boundary, abs_tol=1e-12, rel_tol=0)
    ]
    cost = min(costs[i] for i in near)
    return [i for i in near if costs[i] == cost]
