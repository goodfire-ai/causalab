"""The activation manifold and the two steering paths (arXiv:2605.05115, App. A.3, A.6).

```json
"manifold_path": {
  "type": "script", "script": {"path": "scripts/manifold_fig4/manifold_path.py"},
  "inputs": {"acts": {"step": "harvest_l28", "file": "acts.safetensors"},
             "weight": {"step": "pca48", "file": "basis.safetensors"},
             "mean": {"step": "pca48", "file": "mean.safetensors"},
             "table": {"path": "../artifacts/data/manifold_fig4/data.json"},
             "steps": 50},
  "outputs": {"waypoints_Monday_Tuesday": "waypoints_Monday_Tuesday.safetensors", …  (one per pair)
              "chord_Monday_Tuesday": "chord_Monday_Tuesday.safetensors", …  (one per pair)
              "centroids": …, "points": …, "spline": …, "path": …}
}
```

From the harvested layer-28 activations of every prompt (one row each, the
answer slot) and the PCA basis fitted over them:

1. project every activation into the PCA subspace, average by the prompt's
   ground-truth day into seven **concept centroids**, and give each centroid
   its knot `splines.periodic_angle` (the paper's code normalizes PC1 and
   PC2 before ``atan2``; its text says plain ``atan2``);
2. fit a **periodic cubic spline** through the centroids as a function of the
   knot, the activation manifold ``M_h``, which passes through every centroid
   exactly;
3. build a path for each of the 21 **centroid pairs** ``(a, b)``, ``a``
   before ``b`` in ``itertools.combinations`` order, the pairs over which
   App. A.7 reports the energy and the default path set of the paper's code
   (``configs/analysis/path_steering.yaml``, ``n_extra_pairs: 0``). The
   manifold path is the spline at ``theta_a + t * wrapped_delta(theta_a,
   theta_b)``, the linear path the chord ``(1 - t) c_a + t c_b`` between the
   raw full-space centroids, with ``t = step / (steps - 1)``. The text (App.
   A.6) counts all W(W - 1) = 42 ordered pairs; the code steers the 21
   unordered ones, and each reversed path is the same waypoints in reverse
   order.

The bundles are what the steering document writes into the network. Each
holds one ``value`` entry per step keyed ``{"step": i}``; the steering
document's pair axis names the file and its path axis names the entry, so
every point's selection is authored in its own document. (One bundle keyed by
both axes and selected implicitly from the point's coordinates does not work:
the planner interns forward groups by document content, and the coordinates
are not content, so every point would be served the first point's waypoint.)

* ``waypoints_<pair>.safetensors``: the manifold waypoint **in the
  featurizer's coordinates** ``Pᵀ x`` (``P`` the basis). The ``pca``
  featurizer of the steering document reads and writes ``Pᵀ x`` with no
  centering, so the waypoint ``mean + P u`` on the manifold is the feature
  vector ``Pᵀ mean + u``;
* ``chord_<pair>.safetensors``: the chord point ``(1 - t) c_a + t c_b`` in
  the full residual stream, computed here in float64 and stored in float32.
  The linear write swaps it in, so it is cast to the model's dtype once, as
  the paper's code casts its float32 chord point once
  (``path_mode._build_linear_path_kd``).

The tables are the figure's material: the centroids and every prompt in the
first three principal components, a dense sampling of the spline, and both
paths per (pair, step) in the same coordinates.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splines import (  # noqa: E402  (a sibling module, hashed into the step's closure)
    TWO_PI,
    periodic_angle,
    periodic_cubic_spline,
    wrapped_delta,
)

from causalab.io.step_io import StepError, read_tensor, write_table  # noqa: E402
from causalab.io.step_record import EXAMPLE_COLUMN  # noqa: E402
from causalab.protocol.results import example_labels  # noqa: E402

__all__ = ["main", "DAYS", "centroid_pairs", "path_names"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SPLINE_SAMPLES_PER_SEGMENT = 150


def centroid_pairs(n: int = len(DAYS)) -> list[tuple[int, int]]:
    """The code's centroid pairs: ``itertools.combinations(range(n), 2)``."""
    return list(itertools.combinations(range(n), 2))


def path_names() -> list[str]:
    """The steering document's pair axis, in order: ``<a>_<b>`` for each
    centroid pair."""
    return [f"{DAYS[i]}_{DAYS[j]}" for i, j in centroid_pairs()]


def _rows(tensor: Any) -> np.ndarray:
    array = tensor.detach().cpu().to(__import__("torch").float64).numpy()
    if array.ndim == 3:
        if array.shape[1] != 1:
            raise StepError(
                f"manifold_path: 'acts' has {array.shape[1]} positions per row; "
                "harvest one position (the answer slot)"
            )
        array = array[:, 0, :]
    if array.ndim != 2:
        raise StepError(
            f"manifold_path: 'acts' has shape {array.shape}; expected (n, d)"
        )
    return array


def _save_bundle(
    path: Path, entries: Mapping[str, tuple[np.ndarray, Mapping[str, Any]]]
) -> None:
    """One ``.safetensors`` with the ``entries`` table a swept producer
    writes (key -> slot + coords), which is what the consuming point's
    coordinates are matched against (``causalab.protocol.bundles``)."""
    import torch

    from causalab.io.tensor_files import save_file

    tensors = {
        key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
        for key, (value, _) in entries.items()
    }
    table = {
        key: {"slot": "value", "coords": dict(coords)}
        for key, (_, coords) in entries.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors, str(path), metadata={"entries": json.dumps(table, sort_keys=True)}
    )


def _label(coords: Mapping[str, Any]) -> str:
    return "[" + ",".join(f"{k}={v}" for k, v in coords.items()) + "]"


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    acts = _rows(read_tensor(Path(inputs["acts"]), what="manifold_path: 'acts'"))
    basis = read_tensor(Path(inputs["weight"]), what="manifold_path: 'weight'")
    mean = read_tensor(Path(inputs["mean"]), what="manifold_path: 'mean'")
    basis = basis.detach().cpu().double().numpy()  # (d, k)
    mean = mean.detach().cpu().double().numpy()  # (d,)
    rows = json.loads(Path(inputs["table"]).read_text())
    steps = int(inputs.get("steps", 50))
    if steps < 2:
        raise StepError("manifold_path: 'steps' must be at least 2")
    if len(rows) != acts.shape[0]:
        raise StepError(
            f"manifold_path: the table has {len(rows)} rows but 'acts' has "
            f"{acts.shape[0]}; the harvest must cover the whole table in order"
        )
    if basis.shape[0] != acts.shape[1] or mean.shape[0] != acts.shape[1]:
        raise StepError(
            "manifold_path: basis / mean width does not match the activations"
        )
    k = basis.shape[1]
    # the ground-truth day of each prompt (App. A.3), not the model's answer
    labels = [str(row["result"]) for row in rows]
    # the label the metric tables give each row, so the two point tables join
    ids = example_labels(rows)
    unknown = sorted(set(labels) - set(DAYS))
    if unknown:
        raise StepError(f"manifold_path: unknown answer days in the table: {unknown}")

    # 1. the PCA subspace, the centroids and their knots
    coords = (acts - mean) @ basis  # (n, k)
    centroid_u = np.stack(
        [coords[[label == d for label in labels]].mean(axis=0) for d in DAYS]
    )
    centroid_x = np.stack(
        [acts[[label == d for label in labels]].mean(axis=0) for d in DAYS]
    )
    counts = [int(sum(label == d for label in labels)) for d in DAYS]
    theta = periodic_angle(centroid_u)

    # 2. the activation manifold: a periodic cubic spline through the centroids
    spline = periodic_cubic_spline(theta, centroid_u)
    reconstruction = np.abs(spline(theta) - centroid_u).max()
    if reconstruction > 1e-6:
        raise StepError(
            f"manifold_path: the spline misses a centroid by {reconstruction:.2e}"
        )
    order = [DAYS[i] for i in np.argsort(theta)]
    ring = DAYS + DAYS
    forward = any(order == ring[i : i + 7] for i in range(7))
    backward = any(order == ring[i : i + 7][::-1] for i in range(7))
    if not (forward or backward):
        print(
            f"manifold_path: WARNING: the centroids' knot order {order} is not the "
            "weekday ring; the manifold path interpolates the loop as fitted"
        )

    # 3. one manifold path and one chord per centroid pair
    pca_mean_feature = basis.T @ mean  # Pᵀ mean, (k,)
    waypoints: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
    chords: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
    path_rows: list[dict[str, Any]] = []
    for name, (i, j) in zip(path_names(), centroid_pairs()):
        delta_theta = wrapped_delta(theta[i], theta[j])
        waypoints[name], chords[name] = {}, {}
        for step in range(steps):
            t = step / (steps - 1)
            angle = theta[i] + t * delta_theta
            u = spline(np.array([angle]))[0]  # (k,)
            manifold_x = mean + basis @ u
            linear_x = (1.0 - t) * centroid_x[i] + t * centroid_x[j]
            linear_u = (linear_x - mean) @ basis
            step_coords = {"step": step}
            key = f"value{_label(step_coords)}"
            waypoints[name][key] = (pca_mean_feature + u, step_coords)
            chords[name][key] = (linear_x, step_coords)
            path_rows.append(
                {
                    "pair": name,
                    "step": step,
                    "t": t,
                    "theta": float(angle % TWO_PI),
                    "direction": 1 if delta_theta > 0 else -1,
                    "manifold_pc1": float(u[0]),
                    "manifold_pc2": float(u[1]),
                    "manifold_pc3": float(u[2]) if k > 2 else 0.0,
                    "linear_pc1": float(linear_u[0]),
                    "linear_pc2": float(linear_u[1]),
                    "linear_pc3": float(linear_u[2]) if k > 2 else 0.0,
                    "manifold_linear_distance": float(
                        np.linalg.norm(manifold_x - linear_x)
                    ),
                }
            )

    for prefix, bundles in (("waypoints", waypoints), ("chord", chords)):
        for name, entries in bundles.items():
            slot = f"{prefix}_{name}"
            if slot not in outputs:
                raise StepError(
                    f"manifold_path: the workflow declares no output {slot!r}"
                )
            _save_bundle(Path(outputs[slot]), entries)

    grid = np.linspace(0.0, TWO_PI, 7 * SPLINE_SAMPLES_PER_SEGMENT, endpoint=False)
    dense = spline(grid)
    write_table(
        Path(outputs["centroids"]),
        [
            {
                "day": d,
                "n": counts[i],
                "theta": float(theta[i]),
                "pc1": float(centroid_u[i, 0]),
                "pc2": float(centroid_u[i, 1]),
                "pc3": float(centroid_u[i, 2]) if k > 2 else 0.0,
                "norm": float(np.linalg.norm(centroid_x[i])),
            }
            for i, d in enumerate(DAYS)
        ],
    )
    write_table(
        Path(outputs["points"]),
        [
            {
                EXAMPLE_COLUMN: ids[i],
                "day": labels[i],
                "pc1": float(coords[i, 0]),
                "pc2": float(coords[i, 1]),
                "pc3": float(coords[i, 2]) if k > 2 else 0.0,
            }
            for i in range(acts.shape[0])
        ],
    )
    write_table(
        Path(outputs["spline"]),
        [
            {
                "theta": float(g),
                "pc1": float(p[0]),
                "pc2": float(p[1]),
                "pc3": float(p[2]) if k > 2 else 0.0,
            }
            for g, p in zip(grid, dense)
        ],
    )
    write_table(Path(outputs["path"]), path_rows)
