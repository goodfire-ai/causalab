"""Shared inputs of the manifold_fig4 script tests.

The package reproduces Figure 4 (weekdays column) of Wurgaft et al. 2026
(arXiv:2605.05115). Where the paper's text and the code that drew its figure
differ, its scripts follow the code (goodfire-ai/causalab, branch
``manifold_steering``, commit ``1b6f43a5``). ``paper_pipeline_oracles.json``
holds the oracles the tests hold those rules to; its ``description`` names
the run each value comes from. The helpers below write small synthetic step
inputs in the layout of a run tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.demos.papers._scripts import load_script

ORACLES = json.loads(
    (Path(__file__).resolve().parent / "paper_pipeline_oracles.json").read_text()
)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLASSES = DAYS + ["other"]
TWO_PI = 2.0 * np.pi


def ring_gaps(theta: np.ndarray) -> np.ndarray:
    """``|wrapped(theta[d+1] - theta[d])|`` for Monday->Tuesday .. Sunday->Monday.
    The gaps do not depend on the ring's rotation or orientation, which a
    principal component's sign fixes arbitrarily."""
    theta = np.asarray(theta, dtype=np.float64)
    step = np.roll(theta, -1) - theta
    return np.abs(((step + np.pi) % TWO_PI) - np.pi)


def synthetic_activations(tmp_path: Path) -> dict[str, Path]:
    """Seven noisy clusters on an ellipse in a 12-dimensional space, seven
    prompts per day in ``data.json`` order, a PCA basis of the 49 rows, and
    ``data.json``: the inputs of the ``manifold_path`` step."""
    import torch

    from causalab.io.tensor_files import save_file

    rows = load_script("manifold_fig4", "build_dataset").build()["data"].rows
    labels = [row["result"] for row in rows]
    rng = np.random.default_rng(0)
    angle = {d: TWO_PI * i / 7 for i, d in enumerate(DAYS)}
    acts = np.zeros((49, 12))
    for n, day in enumerate(labels):
        acts[n, 0] = 6.0 * np.cos(angle[day])
        acts[n, 1] = 3.0 * np.sin(angle[day])
        acts[n] += rng.normal(scale=0.3, size=12)
    mean = acts.mean(axis=0)
    _, _, vh = np.linalg.svd(acts - mean, full_matrices=False)
    paths = {
        "acts": tmp_path / "acts.safetensors",
        "weight": tmp_path / "basis.safetensors",
        "mean": tmp_path / "mean.safetensors",
        "table": tmp_path / "data.json",
    }
    save_file({"acts": torch.from_numpy(acts[:, None, :].copy())}, str(paths["acts"]))
    save_file({"weight": torch.from_numpy(vh[:6].T.copy())}, str(paths["weight"]))
    save_file({"mean": torch.from_numpy(mean.copy())}, str(paths["mean"]))
    paths["table"].write_text(json.dumps(rows))
    return paths


def steer_tables(
    tmp_path: Path, pairs: list[str], rng: np.random.Generator
) -> dict[str, Path]:
    """The steering step's three spelling tables per strategy for two prompts
    and three waypoints per pair: one row per (prompt, pair, waypoint)."""
    from causalab.io.step_io import write_table

    paths = {}
    for method in ("linear", "manifold"):
        tables: dict[str, list] = {"space": [], "bare": [], "lower": []}
        for pair in pairs:
            for step in range(3):
                for example in ("0", "1"):
                    mass = rng.dirichlet(np.ones(8))[:7] * 0.9
                    coords = {
                        "example_id": example,
                        "axes.pair": pair,
                        "axes.path": step,
                    }
                    tables["space"].append(
                        {**coords, "value": {d: m * 0.8 for d, m in zip(DAYS, mass)}}
                    )
                    tables["bare"].append(
                        {**coords, "value": {d: m * 0.15 for d, m in zip(DAYS, mass)}}
                    )
                    lower = {
                        d: mass[DAYS.index(d)] * 0.05
                        for d in ("Monday", "Friday", "Sunday")
                    }
                    tables["lower"].append({**coords, "value": lower})
        for spelling, rows in tables.items():
            path = tmp_path / f"{method}_{spelling}.json"
            write_table(path, rows)
            paths[f"{method}_{spelling}"] = path
    return paths


def behavior_tables(tmp_path: Path) -> dict[str, Path]:
    """The ``behavior`` step run on the oracle baseline ``p``: all the day mass
    in the space-prefixed table, none in the other two."""
    from causalab.io.step_io import write_table

    behavior = load_script("manifold_fig4", "behavior_manifold")
    p = np.array(ORACLES["baseline_p"])
    rows = [{"result": d} for d in ORACLES["baseline_labels"]]
    table = tmp_path / "data.json"
    table.write_text(json.dumps(rows))
    inputs: dict[str, object] = {"table": table, "samples": 1050}
    for spelling in ("space", "bare", "lower"):
        path = tmp_path / f"day_probs_{spelling}.json"
        values = [
            {
                d: (float(p[i, j]) if spelling == "space" else 0.0)
                for j, d in enumerate(DAYS)
            }
            for i in range(len(rows))
        ]
        write_table(
            path, [{"example_id": str(i), "value": v} for i, v in enumerate(values)]
        )
        inputs[spelling] = path
    outputs = {
        "centroids": tmp_path / "behavior_centroids.json",
        "points": tmp_path / "behavior_points.json",
        "spline": tmp_path / "behavior_spline.json",
    }
    behavior.main(inputs, outputs)
    return outputs
