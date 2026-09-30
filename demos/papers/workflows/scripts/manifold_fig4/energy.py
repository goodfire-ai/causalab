"""Behavioral trajectories and their cumulative output energy (arXiv:2605.05115, §3.2, App. A.7).

```json
"energy": {
  "type": "script", "script": {"path": "scripts/manifold_fig4/energy.py"},
  "inputs": {"linear_space": {"step": "steer", "file": "linear_space.json"}, …  (space, bare, lower per strategy)
             "behavior_centroids": {"step": "behavior", "file": "behavior_centroids.json"},
             "behavior_spline": {"step": "behavior", "file": "behavior_spline.json"},
             "pair_axis": "axes.pair", "step_axis": "axes.path"},
  "outputs": {"trajectory": "trajectory.json", "energy": "energy.json", "summary": "summary.json"}
}
```

For each steering strategy, path and waypoint, the steering document saved
the weekdays' mass per base prompt in three spellings; ``p`` is their sum
with ``other = 1 - total`` (`behavior_manifold.probabilities_by_example`).

* The **trajectory** table is the pointwise mean of ``p`` over the base
  prompts (the curve the paper plots) and the half-width of each band the
  paper draws around it (``path_visualization.py``): per day, the standard
  deviation over prompts (``ddof = 1``, as ``torch.std``); for ``other``,
  the square root of the summed day variances. ``bc_to_manifold`` is the
  distance of the mean curve to ``M_y``.
* The **energy** of one prompt's trajectory is the sum over the ``K``
  waypoints of ``D_B(p, M_y)`` (App. A.7), averaged over the base prompts to
  one scalar per (strategy, path). ``energy_sum`` takes ``D_B`` from the
  code's Gauss-Newton projection (`behavior_manifold.BehaviorManifold`),
  over all paths of a strategy in one batch as the code does.
  ``energy_sum_dense`` takes the nearest of the behavior step's dense samples,
  the text's ``inf`` over ``M_y``.
* The **summary** follows the paper's reported statistic (§3.2, App. A.7):
  the mean and standard error over the centroid pairs, one value per pair,
  and a paired t test of linear against manifold over them. The
  dense-sampling means sit beside them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_manifold import (  # noqa: E402
    CLASSES,
    DAYS,
    SPELLINGS,
    BehaviorManifold,
    probabilities_by_example,
)
from splines import bhattacharyya_to_set  # noqa: E402

from causalab.io.step_io import StepError, read_table, write_table, write_values  # noqa: E402
from causalab.io.step_record import EXAMPLE_COLUMN  # noqa: E402

__all__ = ["main", "METHODS", "bands", "unordered_pair"]

METHODS = ("manifold", "linear")


def unordered_pair(name: str) -> frozenset[str] | None:
    """The two days of a centroid-pair path ``<a>_<b>``, or ``None`` for a
    name that is not one."""
    parts = name.split("_")
    if len(parts) == 2 and parts[0] != parts[1] and set(parts) <= set(DAYS):
        return frozenset(parts)
    return None


def bands(cube: np.ndarray) -> np.ndarray:
    """``(K, 8)`` band half-widths of a ``(K, n, 8)`` cube over ``n`` prompts:
    the day columns' standard deviation (``ddof = 1``), and for ``other`` the
    square root of the summed day variances (``path_visualization.py``)."""
    if cube.shape[1] < 2:
        return np.zeros((cube.shape[0], cube.shape[2]))
    std = cube.std(axis=1, ddof=1)
    out = std.copy()
    out[:, -1] = np.sqrt((std[:, :-1] ** 2).sum(axis=1))
    return out


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(values.mean()), se


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    from scipy import stats

    pair_axis = str(inputs.get("pair_axis", "axes.pair"))
    step_axis = str(inputs.get("step_axis", "axes.path"))
    manifold = BehaviorManifold.from_rows(
        read_table(Path(inputs["behavior_centroids"]))
    )
    dense_q = np.array(
        [
            [row[c] for c in CLASSES]
            for row in read_table(Path(inputs["behavior_spline"]))
        ],
        dtype=np.float64,
    )

    trajectory_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    per_path: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    per_path_dense: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    for method in METHODS:
        tables = [read_table(Path(inputs[f"{method}_{s}"])) for s in SPELLINGS]
        probs = probabilities_by_example(
            tables, key=(EXAMPLE_COLUMN, pair_axis, step_axis)
        )
        names = {k[1] for k in probs}
        unknown = sorted(p for p in names if unordered_pair(p) is None)
        if unknown:
            raise StepError(
                f"energy: {method} steers {unknown}, which are not centroid "
                "pairs <day>_<day>"
            )
        # combinations order, as the pair axis lists them
        paths = sorted(names, key=lambda p: [DAYS.index(d) for d in p.split("_")])
        if len({unordered_pair(p) for p in paths}) != len(paths):
            raise StepError(
                f"energy: {method} steers a pair in both orientations; steer "
                "each centroid pair once, as the paper's code does"
            )
        steps = sorted({int(k[2]) for k in probs})
        examples = sorted({k[0] for k in probs})
        if len(steps) < 2:
            raise StepError(
                f"energy: {method} has {len(steps)} waypoint(s); need a path"
            )
        last_step = max(steps)
        cube = np.stack(
            [
                np.stack(
                    [
                        np.stack([probs[(ex, path, step)] for ex in examples])
                        for step in steps
                    ]
                )
                for path in paths
            ]
        )  # (paths, K, n, 8)
        flat = cube.reshape(-1, len(CLASSES))
        # one Gauss-Newton batch over every path, prompt and waypoint, as the
        # code projects its whole pair_distributions tensor at once
        distance = manifold.distance(flat).reshape(cube.shape[:3])
        # the dense minimum one path at a time: all paths at once would hold
        # a (paths * K * n) x samples matrix
        distance_dense = np.stack(
            [
                bhattacharyya_to_set(c.reshape(-1, len(CLASSES)), dense_q).reshape(
                    c.shape[:2]
                )
                for c in cube
            ]
        )
        mean_curves = cube.mean(axis=2)  # (paths, K, 8)
        curve_distance = manifold.distance(mean_curves.reshape(-1, len(CLASSES)))
        curve_distance = curve_distance.reshape(mean_curves.shape[:2])
        n = cube.shape[2]
        for index, path in enumerate(paths):
            half_widths = bands(cube[index])
            for i, step in enumerate(steps):
                trajectory_rows.append(
                    {
                        "method": method,
                        "pair": path,
                        "step": step,
                        "t": step / last_step,
                        "n": n,
                        **{
                            c: float(mean_curves[index, i, j])
                            for j, c in enumerate(CLASSES)
                        },
                        **{
                            f"band_{c}": float(half_widths[i, j])
                            for j, c in enumerate(CLASSES)
                        },
                        "bc_to_manifold": float(curve_distance[index, i]),
                    }
                )
            record = {
                "energy_sum": float(distance[index].sum(axis=0).mean()),
                "energy_sum_dense": float(distance_dense[index].sum(axis=0).mean()),
            }
            per_path[method][path] = record["energy_sum"]
            per_path_dense[method][path] = record["energy_sum_dense"]
            energy_rows.append(
                {"method": method, "pair": path, "n_prompts": n, **record}
            )

    pairs = [p for p in per_path["manifold"] if p in per_path["linear"]]
    if not pairs:
        raise StepError("energy: the two strategies share no centroid pair")
    manifold_values = np.array([per_path["manifold"][p] for p in pairs])
    linear_values = np.array([per_path["linear"][p] for p in pairs])
    m_mean, m_se = _mean_se(manifold_values)
    l_mean, l_se = _mean_se(linear_values)
    if len(pairs) > 1:
        test = stats.ttest_rel(linear_values, manifold_values)
        t_stat, p_value = float(test.statistic), float(test.pvalue)
    else:
        t_stat, p_value = float("nan"), float("nan")

    write_table(Path(outputs["trajectory"]), trajectory_rows)
    write_table(Path(outputs["energy"]), energy_rows)
    write_values(
        Path(outputs["summary"]),
        {
            "n_pairs": len(pairs),
            "manifold_energy": m_mean,
            "manifold_energy_se": m_se,
            "linear_energy": l_mean,
            "linear_energy_se": l_se,
            "paired_t": t_stat,
            "paired_p": p_value,
            "manifold_energy_dense": float(
                np.mean([per_path_dense["manifold"][p] for p in pairs])
            ),
            "linear_energy_dense": float(
                np.mean([per_path_dense["linear"][p] for p in pairs])
            ),
        },
    )
