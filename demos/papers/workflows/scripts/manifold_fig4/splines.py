"""The two curve fits behind the paper's manifolds (arXiv:2605.05115, App. A.3-A.5).

* `periodic_cubic_spline`: a periodic cubic interpolant (Reinsch, 1967)
  through W knots on a circle of circumference ``2*pi``, one curve for every
  ambient dimension at once. The cyclic tasks (weekdays, months) use it.
* `periodic_angle` and `wrapped_delta`: the knot of each centroid and the
  direction of a path between two knots, as the paper's code computes them
  (the text says plain ``atan2(PC2, PC1)``; the code normalizes first).
* `log_map` / `exp_map`: the sphere's tangent-plane charts at a base point,
  which is how the behavior manifold is fitted in Hellinger space
  (``p -> sqrt(p)`` puts a distribution on the unit sphere; a spline fitted in
  ambient coordinates would leave it, so the fit is taken in the tangent plane
  at the centroids' mean and lifted back).

The paper's code is goodfire-ai/causalab, branch ``manifold_steering``, commit
``1b6f43a509d2a4ae49fd1bc4252e13889d57e38b``; the file names below are its.

Pure numpy, float64, no scipy: the periodic system is 7 x 7 for weekdays and
solving it densely is the whole cost. A workflow script imports this as a
sibling module (``sys.path`` insert), which the workflow digest records as its
import closure.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

TWO_PI = 2.0 * np.pi


def periodic_cubic_spline(
    theta: np.ndarray, values: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """A periodic cubic spline through ``(theta_j, values_j)``.

    ``theta`` is ``(W,)`` in radians, any order, all distinct modulo ``2*pi``;
    ``values`` is ``(W, d)``. The knots are sorted, the last interval wraps
    from ``theta_max`` back to ``theta_min + 2*pi``, and the curve is C2
    everywhere including across the wrap. Returns ``s(q)`` taking ``(...,)``
    angles to ``(..., d)`` points; ``s(theta_j) == values_j`` to rounding.

    The second derivatives ``M_j`` solve the standard cyclic tridiagonal system
    ``h_{j-1} M_{j-1} + 2 (h_{j-1} + h_j) M_j + h_j M_{j+1} = 6 (Δ_j − Δ_{j-1})``
    with ``Δ_j = (y_{j+1} − y_j) / h_j`` and every index taken modulo ``W``.
    """
    theta = np.asarray(theta, dtype=np.float64) % TWO_PI
    values = np.asarray(values, dtype=np.float64)
    if theta.ndim != 1 or values.ndim != 2 or values.shape[0] != theta.shape[0]:
        raise ValueError("periodic_cubic_spline: theta is (W,), values is (W, d)")
    w = theta.shape[0]
    if w < 3:
        raise ValueError("periodic_cubic_spline: a closed curve needs at least 3 knots")
    order = np.argsort(theta)
    knots = theta[order]
    y = values[order]
    gaps = np.diff(np.append(knots, knots[0] + TWO_PI))  # h_j, j = 0..W-1 (cyclic)
    if np.any(gaps <= 1e-12):
        raise ValueError("periodic_cubic_spline: two knots coincide modulo 2*pi")
    y_next = np.roll(y, -1, axis=0)
    slopes = (y_next - y) / gaps[:, None]  # Δ_j
    rhs = 6.0 * (slopes - np.roll(slopes, 1, axis=0))
    system = np.zeros((w, w))
    for j in range(w):
        system[j, (j - 1) % w] += gaps[(j - 1) % w]
        system[j, j] += 2.0 * (gaps[(j - 1) % w] + gaps[j])
        system[j, (j + 1) % w] += gaps[j]
    second = np.linalg.solve(system, rhs)  # M_j, (W, d)
    second_next = np.roll(second, -1, axis=0)

    def evaluate(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        flat = ((q.reshape(-1) - knots[0]) % TWO_PI) + knots[0]
        j = np.clip(np.searchsorted(knots, flat, side="right") - 1, 0, w - 1)
        h = gaps[j]
        left = knots[j]
        a = (left + h) - flat  # distance to the right knot
        b = flat - left  # distance to the left knot
        out = (
            second[j] * (a**3)[:, None] / (6.0 * h)[:, None]
            + second_next[j] * (b**3)[:, None] / (6.0 * h)[:, None]
            + (y[j] / h[:, None] - second[j] * h[:, None] / 6.0) * a[:, None]
            + (y_next[j] / h[:, None] - second_next[j] * h[:, None] / 6.0) * b[:, None]
        )
        return out.reshape(*q.shape, values.shape[1])

    return evaluate


def periodic_angle(coords: np.ndarray) -> np.ndarray:
    """The knot of each of ``(W, >=2)`` centroids on a closed loop, in
    ``[0, 2*pi)``: PC1 and PC2 centred on the centroids' mean and divided by
    their standard deviation over the centroids, then ``atan2``.

    This is ``remap_periodic_to_angle`` of the paper's code
    (``causalab/methods/spline/builders.py``), with the eigenvalues it
    divides by taken as ``control_points.var(dim=0)``
    (``causalab/methods/spline/train.py``). The division maps an elliptical
    loop to a circle, so the knots are spaced by the loop's shape and not by
    its aspect ratio. The text's rule, plain ``atan2(PC2, PC1)`` (App. A.3),
    puts the weekday knots up to 0.13 rad elsewhere. The two columns are
    divided by the same ``ddof`` correction, so the angle does not depend on
    it.
    """
    coords = np.asarray(coords, dtype=np.float64)[:, :2]
    centred = coords - coords.mean(axis=0)
    scale = centred.std(axis=0, ddof=1)
    return np.arctan2(centred[:, 1] / scale[1], centred[:, 0] / scale[0]) % TWO_PI


def wrapped_delta(start: float, stop: float) -> float:
    """The signed angle from ``start`` to ``stop`` the shorter way round,
    in ``[-pi, pi)``: ``_build_geodesic_path`` of the paper's code
    (``causalab/analyses/path_steering/path_mode.py``), which wraps each
    periodic dimension's difference by its period."""
    return float(((stop - start + np.pi) % TWO_PI) - np.pi)


def unit(v: np.ndarray) -> np.ndarray:
    """Each row of ``v`` scaled to unit Euclidean norm."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def log_map(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Tangent vectors at ``base`` (unit) pointing along the great circle to
    each unit ``points`` row, with length the geodesic (angular) distance."""
    base = unit(np.asarray(base, dtype=np.float64))
    points = unit(np.asarray(points, dtype=np.float64))
    cos = np.clip(points @ base, -1.0, 1.0)
    angle = np.arccos(cos)
    direction = points - cos[:, None] * base[None, :]
    norm = np.linalg.norm(direction, axis=1)
    safe = np.where(norm > 1e-12, norm, 1.0)
    out = direction / safe[:, None] * angle[:, None]
    out[norm <= 1e-12] = 0.0
    return out


def exp_map(base: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Inverse of `log_map`: walk ``|t|`` along the great circle from
    ``base`` in direction ``t``. Unit-norm by construction."""
    base = unit(np.asarray(base, dtype=np.float64))
    tangents = np.asarray(tangents, dtype=np.float64)
    tangents = tangents - (tangents @ base)[:, None] * base[None, :]
    norm = np.linalg.norm(tangents, axis=1)
    safe = np.where(norm > 1e-12, norm, 1.0)
    out = np.cos(norm)[:, None] * base[None, :] + np.sin(norm)[:, None] * (
        tangents / safe[:, None]
    )
    return unit(out)


def bhattacharyya_to_set(p: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """``min_q -log(sum_i sqrt(p_i q_i))`` of each ``p`` row against every
    ``candidates`` row: the distance to the nearest sampled point of a
    manifold (App. A.7). ``p`` is ``(n, c)`` probabilities, ``candidates``
    ``(m, c)`` probabilities."""
    coefficient = (
        np.sqrt(np.clip(p, 0.0, None)) @ np.sqrt(np.clip(candidates, 0.0, None)).T
    )
    return -np.log(np.clip(coefficient.max(axis=1), 1e-300, 1.0))
