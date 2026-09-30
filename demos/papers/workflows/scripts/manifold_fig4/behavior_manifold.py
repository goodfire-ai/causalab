"""The behavior manifold M_y (arXiv:2605.05115, §2.2 and App. A.4).

```json
"behavior": {
  "type": "script", "script": {"path": "scripts/manifold_fig4/behavior_manifold.py"},
  "inputs": {"space": {"step": "baseline", "file": "day_probs_space.json"},
             "bare": {"step": "baseline", "file": "day_probs_bare.json"},
             "lower": {"step": "baseline", "file": "day_probs_lower.json"},
             "table": {"path": "../artifacts/data/manifold_fig4/data.json"}, "samples": 1050},
  "outputs": {"centroids": "behavior_centroids.json", "points": "behavior_points.json",
              "spline": "behavior_spline.json"}
}
```

The output distribution ``p(x)`` of App. A.2 is the softmax mass on each
concept value's spellings plus one ``other`` bin. The baseline document saves
the weekdays' mass three times: space-prefixed (`` Monday``), bare
(``Monday``) and lowercase (`` monday``, only for the three days whose
lowercase spelling is one token). ``p`` is their sum and ``other`` is one
minus the total, clamped at zero. The paper's code keeps every single-token
spelling (``tokenize_variable_values`` in ``causalab/methods/metric.py``).
**Behavior centroids** ``b_i`` are the means of ``p`` over the prompts whose
ground-truth answer is day ``i``.

The manifold is fitted in Hellinger coordinates, ``p -> sqrt(p)``, which puts
every distribution on the unit sphere. A spline through the ambient
coordinates would leave the sphere, so the fit is taken in the **tangent
plane** at ``b* = normalize(mean_i sqrt(b_i))``: log-map every ``sqrt(b_i)``
to a tangent vector, fit the periodic cubic spline to the tangent vectors,
and decode through the exponential map (App. A.4, and ``sphere_project`` of
the code's ``SplineManifold``). Each centroid's knot follows the paper's code
(``fit_belief_tps_pca`` in ``causalab/methods/spline/belief_fit.py``): a PCA
of the 49 per-prompt ``sqrt(p)`` (`hellinger_pca`), the centroids'
``sqrt(max(b_i, 1e-8))`` projected onto its first two components, and
`splines.periodic_angle`.

`BehaviorManifold.distance` is the code's distance to ``M_y``
(``hellinger_distance_to_manifold`` in ``causalab/methods/distances.py``):
Gauss-Newton from the nearest centroid, then ``D_B = -log(1 - d_H^2)``. The
step also writes a dense sampling of ``M_y``, which gives the nearest point by
brute force (the text's ``inf`` over ``M_y``, App. A.7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splines import (  # noqa: E402  (a sibling module, hashed into the step's closure)
    TWO_PI,
    exp_map,
    log_map,
    periodic_angle,
    periodic_cubic_spline,
    unit,
)

from causalab.io.step_io import StepError, read_table, write_table  # noqa: E402
from causalab.io.step_record import EXAMPLE_COLUMN  # noqa: E402
from causalab.protocol.results import example_labels  # noqa: E402

__all__ = [
    "main",
    "BehaviorManifold",
    "CLASSES",
    "DAYS",
    "SPELLINGS",
    "class_probs",
    "hellinger_pca",
    "probabilities_by_example",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLASSES = DAYS + ["other"]
#: The saved spellings of each day, one ``class_probs`` table each.
SPELLINGS = ("space", "bare", "lower")
#: The code's floor under ``p`` before a square root (``belief_fit.EPS``).
EPS = 1e-8


def class_probs(value: Any) -> dict[str, float]:
    """One ``class_probs`` value: the table stores a mapping (or its JSON
    text, when the writer serialized it)."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise StepError(f"class_probs value is not a mapping: {value!r}")
    return {str(k): float(v) for k, v in value.items()}


def probabilities_by_example(
    tables: Sequence[list[dict[str, Any]]],
    key: tuple[str, ...] = (EXAMPLE_COLUMN,),
) -> dict[tuple[Any, ...], np.ndarray]:
    """``p`` per key (the metric table's example label, `EXAMPLE_COLUMN`,
    plus any sweep coordinates): each day's mass summed over the spelling
    ``tables``, then ``other = max(0, 1 - sum)``, as an ``(8,)`` vector in
    `CLASSES` order.

    The first table names all seven days. A later table may name only some
    (the lowercase spelling is one token for three days), and a day it does
    not name adds nothing. Every table must cover the same keys."""
    if not tables:
        raise StepError("behavior: no spelling tables")
    first, *rest = tables
    others = [
        {tuple(row[k] for k in key): class_probs(row["value"]) for row in table}
        for table in rest
    ]
    by_key: dict[tuple[Any, ...], np.ndarray] = {}
    for row in first:
        k = tuple(row[c] for c in key)
        values = class_probs(row["value"])
        missing = sorted(set(DAYS) - set(values))
        if missing:
            raise StepError(f"behavior: row {k} of the first table lacks {missing}")
        days = np.array([values[d] for d in DAYS], dtype=np.float64)
        for table in others:
            if k not in table:
                raise StepError(f"behavior: row {k} is missing from a spelling table")
            days = days + np.array([table[k].get(d, 0.0) for d in DAYS])
        total = float(days.sum())
        if total > 1.0 + 1e-6:
            raise StepError(f"behavior: row {k} sums to {total:.6f} > 1")
        by_key[k] = np.append(days, max(0.0, 1.0 - total))
    if any(len(table) != len(by_key) for table in others):
        raise StepError("behavior: the spelling tables cover different rows")
    return by_key


def hellinger_pca(p: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the top ``k`` principal directions ``(k, classes)`` of
    ``sqrt(max(p, 0))`` over the rows of ``p``, the code's ``PCA(n_components
    =3)`` of the per-prompt ``sqrt(p)`` (``output_manifold/main.py``). Each
    direction's largest entry is positive, scikit-learn's sign rule."""
    x = np.sqrt(np.clip(np.asarray(p, dtype=np.float64), 0.0, None))
    mean = x.mean(axis=0)
    _, _, vh = np.linalg.svd(x - mean, full_matrices=False)
    components = vh[:k].copy()
    signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
    return mean, components * signs[:, None]


class BehaviorManifold:
    """``M_y``: a periodic spline in the tangent plane of the Hellinger
    sphere through seven centroids at given knots."""

    def __init__(self, theta: np.ndarray, centroids: np.ndarray) -> None:
        """``theta`` is ``(W,)`` knots in radians, ``centroids`` the ``(W, C)``
        centroid distributions."""
        self.theta = np.asarray(theta, dtype=np.float64) % TWO_PI
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.sqrt_centroids = np.sqrt(np.clip(self.centroids, EPS, None))
        self.base = unit(self.sqrt_centroids.mean(axis=0))
        self.tangent = periodic_cubic_spline(
            self.theta, log_map(self.base, self.sqrt_centroids)
        )

    @classmethod
    def fit(cls, p: np.ndarray, labels: Sequence[str]) -> "BehaviorManifold":
        """The manifold of per-prompt distributions ``p`` ``(n, C)`` whose
        ground-truth days are ``labels``, with the code's knots."""
        p = np.asarray(p, dtype=np.float64)
        centroids = np.stack(
            [p[[label == d for label in labels]].mean(axis=0) for d in DAYS]
        )
        mean, components = hellinger_pca(p, 3)
        projected = (np.sqrt(np.clip(centroids, EPS, None)) - mean) @ components.T
        return cls(periodic_angle(projected[:, :2]), centroids)

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Unit ``sqrt(q)`` vectors of ``M_y`` at knots ``u`` ``(n,)``."""
        return exp_map(self.base, self.tangent(np.atleast_1d(np.asarray(u, float))))

    def sample(self, n: int) -> np.ndarray:
        """``n`` distributions of ``M_y`` at equally spaced knots."""
        return self.decode(np.linspace(0.0, TWO_PI, n, endpoint=False)) ** 2

    def nearest(
        self,
        p: np.ndarray,
        iterations: int = 5,
        tol: float = 1e-6,
        damping: float = 1e-6,
        step: float = 1e-5,
    ) -> np.ndarray:
        """The knot of the point of ``M_y`` nearest each row of ``p``, by the
        code's ``encode_to_nearest_point`` (``causalab/methods/spline/
        manifold.py``): start at the nearest centroid in Hellinger space, then
        at most ``iterations`` damped Gauss-Newton steps with a central
        finite-difference Jacobian, stopping when every row's step is below
        ``tol``. It finds a local minimum, which can lie above the global one
        that dense sampling finds."""
        h = self._sphere(p)
        distances = ((h[:, None, :] - self.sqrt_centroids[None]) ** 2).sum(axis=-1)
        u = self.theta[distances.argmin(axis=1)]
        for _ in range(iterations):
            residual = self.decode(u) - h
            jacobian = (self.decode(u + step) - self.decode(u - step)) / (2 * step)
            delta = (jacobian * residual).sum(axis=1) / (
                (jacobian * jacobian).sum(axis=1) + damping
            )
            u = (u - delta) % TWO_PI
            if np.abs(delta).max() < tol:
                break
        return u

    def distance(self, p: np.ndarray) -> np.ndarray:
        """``D_B(p, M_y) = -log(1 - d_H^2)`` per row of ``p``, with ``d_H`` the
        Hellinger distance to the `nearest` point, floored at ``1e-7`` inside
        the log as the code does (``distance_from_behavior_manifold.py``)."""
        h = self._sphere(p)
        d_h = np.linalg.norm(h - self.decode(self.nearest(p)), axis=1) / np.sqrt(2.0)
        return -np.log(np.clip(1.0 - d_h**2, 1e-7, None))

    @staticmethod
    def _sphere(p: np.ndarray) -> np.ndarray:
        h = np.sqrt(np.clip(np.asarray(p, dtype=np.float64), EPS, None))
        return h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), EPS, None)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> "BehaviorManifold":
        """The manifold the ``behavior`` step fitted, from its
        ``behavior_centroids.json`` rows (knot and distribution per day)."""
        by_day = {row["day"]: row for row in rows}
        if set(by_day) != set(DAYS):
            raise StepError(f"behavior: centroid rows name {sorted(by_day)}")
        theta = np.array([float(by_day[d]["theta"]) for d in DAYS])
        centroids = np.array([[float(by_day[d][c]) for c in CLASSES] for d in DAYS])
        return cls(theta, centroids)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    tables = [read_table(Path(inputs[spelling])) for spelling in SPELLINGS]
    rows = json.loads(Path(inputs["table"]).read_text())
    samples = int(inputs.get("samples", 1050))
    probs = probabilities_by_example(tables)
    n = len(rows)
    # the metric tables label each row the way the run did: the dataset's
    # example_id, else the row index as a string
    ids = example_labels(rows)
    if set(probs) != {(label,) for label in ids}:
        raise StepError(
            f"behavior: the baseline tables cover examples {sorted(k[0] for k in probs)[:5]}… "
            f"but the table has {n} rows labelled {ids[:5]}…"
        )
    labels = [str(row["result"]) for row in rows]
    p = np.stack([probs[(label,)] for label in ids])  # (n, 8)
    manifold = BehaviorManifold.fit(p, labels)
    counts = [int(sum(label == d for label in labels)) for d in DAYS]
    decoded = manifold.decode(manifold.theta)
    if np.abs(decoded - unit(manifold.sqrt_centroids)).max() > 1e-6:
        raise StepError("behavior: the decoded spline misses a centroid")

    grid = np.linspace(0.0, TWO_PI, samples, endpoint=False)
    dense = manifold.decode(grid) ** 2  # unit sqrt vectors square to distributions

    write_table(
        Path(outputs["centroids"]),
        [
            {"day": d, "n": counts[i], "theta": float(manifold.theta[i])}
            | {c: float(manifold.centroids[i, j]) for j, c in enumerate(CLASSES)}
            for i, d in enumerate(DAYS)
        ],
    )
    write_table(
        Path(outputs["points"]),
        [
            {EXAMPLE_COLUMN: ids[i], "day": labels[i]}
            | {c: float(p[i, j]) for j, c in enumerate(CLASSES)}
            for i in range(n)
        ],
    )
    write_table(
        Path(outputs["spline"]),
        [
            {"theta": float(g)} | {c: float(q[j]) for j, c in enumerate(CLASSES)}
            for g, q in zip(grid, dense)
        ],
    )
