"""Figure 2a: this replication's addition curve, and the curve of each rank.

A workflow ``script`` step, ``main(inputs, outputs)``, over the ``apply``
steps' ``iia.json``; it is also a command. Run after the workflow, from
``demos/papers/``::

    python workflows/scripts/arithmetic_fig2a/fig2a_figure.py            # reads artifacts/output/arithmetic_fig2a/, writes artifacts/figures/arithmetic_fig2a/

Each table has one row per (layer, component, k, example), a sublayer being
one (layer, component) of the fit's two sweeps. The replication figure
reports, per sublayer, the mean test IIA of the best ``k``, which is what the
paper's "Best Test IIA" plots (Section 2: "reporting results for the
dimension with the best performance"). The ranks figure adds the curve of
every ``k``, so the reader sees which rank the best-of selection takes.

Inputs
    ``iia_<i>``      the apply steps' per-example IIA tables, one per fit shard
                     (any input whose name starts with ``iia``)

Outputs
    ``replication``  the best-k curve alone, in the paper's style, ``.png``;
                     the page shows the paper's crop separately
    ``ranks``        the best-k curve over the curve of each k, ``.png``
    ``plotted``      the drawn rows: sublayer, layer, component, best_k,
                     test_iia, and ``iia_k<k>`` for each k

The rows are aggregated through [`causalab.io.step_record.aggregate`][], the
same reduction the shipped ``select`` and ``workflow_figures`` scripts use, so
a value quoted from ``plotted`` and one chosen from the table by a later step
cannot disagree about what a row is.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import aggregate, axes_for

__all__ = ["main", "cli"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/arithmetic_fig2a/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "arithmetic_fig2a"
FIGURES = PAPERS / "artifacts" / "figures" / "arithmetic_fig2a"
#: The workflow's apply steps, one per fit shard of eight layers.
APPLY_STEPS = ("apply_0", "apply_1", "apply_2", "apply_3")

#: The two sublayer components the fit sweeps, in depth order:
#: ``sublayer = 2 * layer + half``, half 0 before the MLP and 1 after it.
COMPONENTS = ("block_mid", "block_output")
LAYER_AXIS = "sites.target.layers"
COMPONENT_AXIS = "sites.target.component"
#: The derived column this script keys the curve by.
SITE_AXIS = "sublayer"
K_AXIS = "featurizers.rot.k"
#: The paper draws the addition curve in matplotlib's default red.
RED = "#d62728"
#: The drop the paper labels "Layer 18 MLP": x = 18 is the residual stream
#: before the layer-18 MLP, x = 18.5 the one after it.
MLP18 = (18.0, 18.5)


def best_k_curve(iia_paths: Sequence[Path]) -> Any:
    """Per sublayer: the mean test IIA at the best k, and that k.

    ``iia_paths`` are the apply steps' tables — one per fit shard, each over
    its own sublayers — aggregated per (sublayer, k) and concatenated."""
    import pandas as pd

    parts = []
    for iia_path in iia_paths:
        df = frame(iia_path)
        axes = [axis for axis in axes_for(iia_path) if axis in df.columns]
        for axis in (LAYER_AXIS, COMPONENT_AXIS, K_AXIS):
            if axis not in axes:
                raise StepError(
                    f"{iia_path.name} does not carry the axis {axis!r} (has {axes}); "
                    "this script reads the apply steps of workflow.json"
                )
        table = aggregate(df, iia_path, "value")[0]
        unknown = set(table[COMPONENT_AXIS].unique()) - set(COMPONENTS)
        if unknown:
            raise StepError(f"{iia_path.name}: unexpected components {sorted(unknown)}")
        table[SITE_AXIS] = 2 * table[LAYER_AXIS].astype(int) + table[
            COMPONENT_AXIS
        ].map(COMPONENTS.index)
        parts.append(table)
    per_point = pd.concat(parts, ignore_index=True)
    best = (
        per_point.sort_values(
            [SITE_AXIS, "value", K_AXIS], ascending=[True, False, True]
        )
        .groupby(SITE_AXIS, sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    out = best.rename(
        columns={SITE_AXIS: "sublayer", K_AXIS: "best_k", "value": "test_iia"}
    )
    out["sublayer"] = out["sublayer"].astype(int)
    out["best_k"] = out["best_k"].astype(int)
    out["layer"] = out[LAYER_AXIS].astype(int)
    out["component"] = out[COMPONENT_AXIS]
    return out[["sublayer", "layer", "component", "best_k", "test_iia"]], per_point


def with_ranks(curve: Any, per_point: Any) -> Any:
    """The best-k rows with one ``iia_k<k>`` column per rank, the values the
    ranks figure draws."""
    wide = per_point.pivot(index=SITE_AXIS, columns=K_AXIS, values="value")
    wide.columns = [f"iia_k{int(k)}" for k in wide.columns]
    return curve.merge(wide, left_on="sublayer", right_index=True, how="left")


def _axes_like_paper(ax: Any) -> None:
    """The paper's frame: dashed grid, no top or right spine, 0 to 1."""
    ax.set_xlim(-1.5, 32.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best Test IIA")
    ax.grid(True, color="0.85", linestyle="--", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def draw_replication(target: Path, curve: Any, caption: str) -> None:
    """The best-k addition curve alone, drawn as the paper draws its red
    series, with the paper's "Layer 18 MLP" arrow at the same drop."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    x = curve["sublayer"] / 2.0  # sublayer 2L is layer L, 2L+1 is L.5
    ax.plot(x, curve["test_iia"], color=RED, linewidth=1.6, label="Addition")
    by_x = dict(zip(x, curve["test_iia"]))
    if all(point in by_x for point in MLP18):
        drop = (sum(MLP18) / 2, (by_x[MLP18[0]] + by_x[MLP18[1]]) / 2)
        ax.annotate(
            "Layer 18 MLP",
            xy=drop,
            xytext=(21.0, 0.8),
            fontsize=10,
            style="italic",
            arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.2},
        )
    _axes_like_paper(ax)
    ax.set_title("(a) Input Concept (first operand of a+b=)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    figure.suptitle(caption, fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def draw_ranks(target: Path, curve: Any, per_point: Any, caption: str) -> None:
    """The curve of each k in greys, light to dark with k, under the best-k
    curve in red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    ks = sorted(int(k) for k in per_point[K_AXIS].unique())
    for rank, k in enumerate(ks):
        rows = per_point[per_point[K_AXIS] == k].sort_values(SITE_AXIS)
        # Rank is ordered, so one grey ramp from light (k = 1) to dark.
        shade = 0.8 - 0.6 * rank / max(len(ks) - 1, 1)
        ax.plot(
            rows[SITE_AXIS] / 2.0,
            rows["value"],
            color=str(round(shade, 3)),
            linewidth=1.0,
            label=f"k = {k}",
            zorder=2,
        )
    ax.plot(
        curve["sublayer"] / 2.0,
        curve["test_iia"],
        color=RED,
        linewidth=1.6,
        linestyle=(0, (4, 2)),
        label="best k",
        zorder=3,
    )
    _axes_like_paper(ax)
    ax.set_ylabel("Test IIA")
    ax.set_title("Addition, every rank k of the DAS subspace", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    figure.suptitle(caption, fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    iia_paths = [
        Path(inputs[name]) for name in sorted(inputs) if name.startswith("iia")
    ]
    if not iia_paths:
        raise StepError(
            "declare at least one input named iia or iia_<i>: an apply step's iia.json"
        )
    curve, per_point = best_k_curve(iia_paths)
    n_ranks = per_point[K_AXIS].nunique()
    caption = (
        f"causalab: Llama-3.1-8B bf16, held-out split; best of {n_ranks} ranks "
        "per sublayer, one seed each"
    )
    if "replication" in outputs:
        draw_replication(Path(outputs["replication"]), curve, caption)
    if "ranks" in outputs:
        draw_ranks(Path(outputs["ranks"]), curve, per_point, caption)
    if "plotted" in outputs:
        write_frame(with_ranks(curve, per_point), Path(outputs["plotted"]))


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: apply_<i>/iia.json is read; the figures go to artifacts/figures/",
    )
    args = parser.parse_args(argv)
    main(
        {
            f"iia_{i}": args.artifacts / step / "iia.json"
            for i, step in enumerate(APPLY_STEPS)
        },
        {
            "replication": FIGURES / "fig2a_replication.png",
            "ranks": FIGURES / "fig2a_ranks.png",
            "plotted": FIGURES / "fig2a_plotted.json",
        },
    )
    print(FIGURES / "fig2a_replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
