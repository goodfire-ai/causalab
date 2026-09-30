"""Figure 4, weekdays column: this replication's four rows, together and one by one.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/manifold_fig4/fig4_figure.py                      # Tuesday -> Friday, the paper's example
    python workflows/scripts/manifold_fig4/fig4_figure.py --pair Monday_Thursday

It reads the run tree under ``artifacts/output/manifold_fig4/``: the
activation manifold tables of the ``manifold_path`` step, the behavior
manifold tables of the ``behavior`` step, and the mean trajectories, bands and
energies of the ``energy`` step. It writes, under
``artifacts/figures/manifold_fig4/``:

* ``fig4_replication.png``, the four rows of the paper's column (the page
  shows the paper's crop separately);
* ``fig4_behavior.png``, ``fig4_activation.png`` and ``fig4_steering.png``,
  each panel alone (`PANEL_FILES`); the steering panel holds both P(token)
  rows, the two rows the steering document draws;
* ``fig4_plotted.json``, every value drawn: the energy summary, both mean
  trajectories with their band half-widths, and for each 3D panel its view
  and the projected prompts, centroids, manifold curve and both paths.

The rows are the paper's:

* **Behavior space.** ``sqrt(p)`` projected onto the top three principal
  components of the 49 prompts' ``sqrt(p)``, as the paper's ``plot_3d`` does
  for ``feature_kind="hellinger"`` (``path_steering/path_visualization.py``,
  ``plot_paths_in_belief_space``): the prompts, the centroids ``sqrt(b_i)``,
  the behavior manifold, and both trajectories as ``sqrt`` of their mean
  distribution.
* **Activation space.** The first three principal components of the
  harvested layer-28 activations, with the prompts, the centroids, the
  activation manifold and both steering paths.
* **P(token)** along the manifold path and along the linear path, the mean
  over the 16 base prompts with the paper's bands: plus or minus one
  standard deviation over prompts per day, and the square root of the summed
  day variances for ``other``.

Both 3D panels use equal units on their three axes. They look down at
``ELEVATION`` (55 degrees), from an azimuth perpendicular to the pair's chord
(`view_azimuth`). So the chord's horizontal part is not foreshortened, and
the chord does not hide behind the loop.

The figure is a command rather than a workflow step because a step's outputs
must stay inside its own directory (workflow spec §2.3) and the committed
figure sits under ``artifacts/figures/manifold_fig4/``; ``main(inputs,
outputs)`` keeps the step signature anyway.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_manifold import CLASSES, DAYS, hellinger_pca  # noqa: E402

from causalab.io.step_io import StepError, read_table, read_values  # noqa: E402

__all__ = [
    "main",
    "cli",
    "load",
    "draw_panels",
    "equal_units",
    "view_azimuth",
    "PANEL_FILES",
    "ELEVATION",
]

#: ``demos/papers/``: this script sits in ``workflows/scripts/manifold_fig4/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "manifold_fig4"
FIGURES = PAPERS / "artifacts" / "figures" / "manifold_fig4"
#: The paper's legend colours, Monday → Sunday, and the dashed `other`.
COLORS = {
    "Monday": "#7b2cbf",
    "Tuesday": "#2f6fe4",
    "Wednesday": "#19c6c6",
    "Thursday": "#4fdc8c",
    "Friday": "#b9d24a",
    "Saturday": "#f08a3c",
    "Sunday": "#e0201e",
    "other": "#e0201e",
}
MANIFOLD = "manifold"
LINEAR = "linear"
#: panel -> the file stem of its image alone, in the order of the paper's rows.
PANEL_FILES = {
    "behavior": "fig4_behavior",
    "activation": "fig4_activation",
    "steering": "fig4_steering",
}
#: panel -> rows of the grid it takes (the steering panel is two P(token) rows).
PANEL_ROWS = {"behavior": 1, "activation": 1, "steering": 2}
#: The 3D views' elevation in degrees. At 15 and 35 the Tuesday -> Friday
#: linear trajectory crosses the Sunday centroid in the behavior panel's
#: projection; at 55 both panels show it inside the loop, clear of every
#: centroid (compared by eye at 15, 35, 55 and 75).
ELEVATION = 55.0
PCS = ("pc1", "pc2", "pc3")


def view_azimuth(start: np.ndarray, stop: np.ndarray) -> float:
    """The azimuth in degrees (matplotlib's convention) at which the line of
    sight, seen from above, is perpendicular to the segment ``start -> stop``.

    The segment's horizontal part then lies parallel to the image plane at
    any elevation. The figure looks down at ``ELEVATION``."""
    step = np.asarray(stop, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    return float(np.degrees(np.arctan2(step[1], step[0])) + 90.0)


def equal_units(drawn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(low, high, box)`` for a 3D panel of the ``(n, 3)`` points ``drawn``:
    each axis spans the points, and the box aspect is proportional to the
    spans, so one unit has the same length on all three axes and the loop
    keeps its shape."""
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    return low, high, high - low


def _vec(rows: list[dict[str, Any]], cols: tuple[str, ...]) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)


def _xyz(points: np.ndarray, **extra: list[Any]) -> list[dict[str, Any]]:
    """Rows of drawn 3D coordinates, for ``fig4_plotted.json``."""
    rows = []
    for i, point in enumerate(points):
        row = {key: values[i] for key, values in extra.items()}
        rows.append(
            row | {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
        )
    return rows


def _panel(
    points: np.ndarray,
    point_days: list[str],
    centroids: np.ndarray,
    curve: np.ndarray,
    manifold: np.ndarray,
    linear: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    """One 3D panel's drawn values and its view."""
    return {
        "view": {
            "elevation": ELEVATION,
            "azimuth": view_azimuth(linear[0], linear[-1]),
        },
        "points": _xyz(points, day=point_days),
        "centroids": _xyz(centroids, day=list(DAYS)),
        "manifold_curve": _xyz(curve),
        "manifold_path": _xyz(manifold),
        "linear_path": _xyz(linear),
        **extra,
    }


def load(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The drawn material for one pair: both 3D panels in the coordinates
    they plot, and both mean trajectories with their bands."""
    pair = str(inputs["pair"])
    act_centroids = read_table(Path(inputs["centroids"]))
    act_points = read_table(Path(inputs["points"]))
    act_spline = read_table(Path(inputs["spline"]))
    act_path = [r for r in read_table(Path(inputs["path"])) if r["pair"] == pair]
    beh_centroids = read_table(Path(inputs["behavior_centroids"]))
    beh_points = read_table(Path(inputs["behavior_points"]))
    beh_spline = read_table(Path(inputs["behavior_spline"]))
    trajectory = [
        r for r in read_table(Path(inputs["trajectory"])) if r["pair"] == pair
    ]
    summary = read_values(Path(inputs["summary"]))
    if not act_path or not trajectory:
        raise StepError(f"fig4_figure: no rows for pair {pair!r}")
    act_path.sort(key=lambda r: r["step"])
    man_rows = sorted(
        (r for r in trajectory if r["method"] == MANIFOLD), key=lambda r: r["step"]
    )
    lin_rows = sorted(
        (r for r in trajectory if r["method"] == LINEAR), key=lambda r: r["step"]
    )
    by_day = lambda rows: sorted(rows, key=lambda r: DAYS.index(r["day"]))  # noqa: E731

    # behavior space: the paper's plot_3d on sqrt(p) of the 49 prompts
    classes = tuple(CLASSES)
    prompts_p = _vec(beh_points, classes)
    mean, components = hellinger_pca(prompts_p, 3)
    project = lambda p: (np.sqrt(np.clip(p, 0.0, None)) - mean) @ components.T  # noqa: E731
    variance = ((np.sqrt(prompts_p) - mean) @ components.T).var(axis=0, ddof=1)
    total = np.sqrt(prompts_p).var(axis=0, ddof=1).sum()
    behavior = _panel(
        project(prompts_p),
        [r["day"] for r in beh_points],
        project(_vec(by_day(beh_centroids), classes)),
        project(_vec(beh_spline, classes)),
        project(_vec(man_rows, classes)),
        project(_vec(lin_rows, classes)),
        explained_variance_ratio=[float(v / total) for v in variance],
    )
    activation = _panel(
        _vec(act_points, PCS),
        [r["day"] for r in act_points],
        _vec(by_day(act_centroids), PCS),
        _vec(act_spline, PCS),
        _vec(act_path, ("manifold_pc1", "manifold_pc2", "manifold_pc3")),
        _vec(act_path, ("linear_pc1", "linear_pc2", "linear_pc3")),
    )
    return {
        "pair": pair,
        "summary": summary,
        "man_rows": man_rows,
        "lin_rows": lin_rows,
        "behavior": behavior,
        "activation": activation,
    }


def _array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([[r["x"], r["y"], r["z"]] for r in rows], dtype=np.float64)


def _draw_3d(ax: Any, title: str, panel: Mapping[str, Any], label_paths: bool) -> None:
    points = _array(panel["points"])
    days = [r["day"] for r in panel["points"]]
    centroids = _array(panel["centroids"])
    curve = _array(panel["manifold_curve"])
    manifold = _array(panel["manifold_path"])
    linear = _array(panel["linear_path"])
    for day in DAYS:
        mask = np.array([d == day for d in days])
        if mask.any():
            ax.scatter(
                *points[mask].T, s=6, alpha=0.18, color=COLORS[day], depthshade=False
            )
    ax.plot(*np.vstack([curve, curve[:1]]).T, color="0.55", lw=1.0, alpha=0.9)
    for i, day in enumerate(DAYS):
        ax.scatter(
            *centroids[i],
            marker="D",
            s=46,
            color=COLORS[day],
            edgecolor="k",
            linewidth=0.4,
            depthshade=False,
            zorder=5,
        )
    ax.plot(
        *linear.T,
        color="0.45",
        lw=3.0,
        alpha=0.9,
        label="Linear" if label_paths else None,
    )
    ax.plot(
        *manifold.T,
        color="k",
        lw=1.0,
        marker="o",
        ms=2.5,
        label="Manifold" if label_paths else None,
    )
    low, high, box = equal_units(
        np.vstack([points, centroids, curve, manifold, linear])
    )
    for setter, lo, hi in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), low, high):
        setter(lo, hi)
    ax.set_box_aspect(tuple(box))
    ax.view_init(elev=panel["view"]["elevation"], azim=panel["view"]["azimuth"])
    ax.set_title(title, fontsize=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("PC1", fontsize=7, labelpad=-8)
    ax.set_ylabel("PC2", fontsize=7, labelpad=-8)
    ax.set_zlabel("PC3", fontsize=7, labelpad=-8)
    if label_paths:
        ax.legend(loc="upper left", fontsize=7, frameon=False)


def _draw_probs(
    ax: Any, rows: list[dict[str, Any]], title: str, xlabel: str | None
) -> None:
    t = np.array([r["t"] for r in rows])
    for cls in CLASSES:
        y = np.array([r[cls] for r in rows])
        band = np.array([r[f"band_{cls}"] for r in rows])
        style = dict(ls="--", lw=1.6) if cls == "other" else dict(ls="-", lw=1.6)
        ax.plot(t, y, color=COLORS[cls], label=cls, **style)
        ax.fill_between(
            t,
            np.clip(y - band, 0, None),
            np.clip(y + band, 0, 1),
            color=COLORS[cls],
            alpha=0.2,
            lw=0,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("P(token)", fontsize=9)
    ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)


def draw_panels(
    target: Path, material: Mapping[str, Any], panels: list[str], caption: str
) -> None:
    """The replication alone: ``panels`` stacked in the paper's row order,
    without the paper's crop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = {"behavior": [1.35], "activation": [1.35], "steering": [1.0, 1.0]}
    ratios = [h for panel in panels for h in heights[panel]]
    figure = plt.figure(figsize=(6.4, 3.4 * sum(ratios) + 0.6))
    gs = figure.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.35)
    a, b = str(material["pair"]).split("_")
    row = 0
    for panel in panels:
        if panel in ("behavior", "activation"):
            ax = figure.add_subplot(gs[row, 0], projection="3d")
            title = (
                "Behavior space (√p, 3D PCA)"
                if panel == "behavior"
                else "Activation space (layer 28, PC1–3)"
            )
            _draw_3d(ax, title, material[panel], label_paths=panel == "activation")
        else:
            top = figure.add_subplot(gs[row, 0])
            _draw_probs(top, material["man_rows"], "Manifold steering", None)
            top.legend(loc="upper right", fontsize=6.5, ncol=2, frameon=False)
            bottom = figure.add_subplot(gs[row + 1, 0], sharex=top)
            _draw_probs(bottom, material["lin_rows"], "Linear steering", f"{a} to {b}")
        row += PANEL_ROWS[panel]
    figure.suptitle(caption, fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    material = load(inputs)
    summary = material["summary"]
    man_rows = material["man_rows"]
    caption = (
        f"causalab: Llama-3.1-8B bf16, layer 28, mean over {man_rows[0]['n']} base prompts, "
        f"{len(man_rows)} waypoints\nE_BC over {summary['n_pairs']} centroid pairs: "
        f"manifold {summary['manifold_energy']:.2f} ± {summary['manifold_energy_se']:.2f}, "
        f"linear {summary['linear_energy']:.2f} ± {summary['linear_energy_se']:.2f}"
    )
    if "replication" in outputs:
        draw_panels(Path(outputs["replication"]), material, list(PANEL_FILES), caption)
    for panel in PANEL_FILES:
        if panel in outputs:
            draw_panels(Path(outputs[panel]), material, [panel], caption)
    if "plotted" in outputs:
        Path(outputs["plotted"]).parent.mkdir(parents=True, exist_ok=True)
        Path(outputs["plotted"]).write_text(
            json.dumps(
                {
                    "pair": material["pair"],
                    "summary": summary,
                    "trajectory": man_rows + material["lin_rows"],
                    "behavior": material["behavior"],
                    "activation": material["activation"],
                },
                indent=1,
            )
            + "\n"
        )


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--pair", default="Tuesday_Friday")
    parser.add_argument("--artifacts", default=str(RUN))
    parser.add_argument("--figures", default=str(FIGURES))
    args = parser.parse_args(argv)
    root = Path(args.artifacts)
    figures = Path(args.figures)
    main(
        {
            "pair": args.pair,
            "centroids": root / "manifold_path" / "centroids.json",
            "points": root / "manifold_path" / "points.json",
            "spline": root / "manifold_path" / "spline.json",
            "path": root / "manifold_path" / "path.json",
            "behavior_centroids": root / "behavior" / "behavior_centroids.json",
            "behavior_points": root / "behavior" / "behavior_points.json",
            "behavior_spline": root / "behavior" / "behavior_spline.json",
            "trajectory": root / "energy" / "trajectory.json",
            "summary": root / "energy" / "summary.json",
        },
        {
            "replication": figures / "fig4_replication.png",
            **{panel: figures / f"{stem}.png" for panel, stem in PANEL_FILES.items()},
            "plotted": figures / "fig4_plotted.json",
        },
    )
    print(figures / "fig4_replication.png")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
