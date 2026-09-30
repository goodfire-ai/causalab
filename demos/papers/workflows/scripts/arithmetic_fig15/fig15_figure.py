"""Figure 15: this replication's three heatmaps and the values drawn.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/arithmetic_fig15/fig15_figure.py            # reads artifacts/output/arithmetic_fig15/, writes artifacts/figures/arithmetic_fig15/

It reads the ``scan`` step's three IIA tables (one patched forward per point,
scored against each variable's label) from ``artifacts/output/arithmetic_fig15/scan/`` and writes,
under ``artifacts/figures/arithmetic_fig15/``, ``fig15_replication.png`` (the three heatmaps; the
page shows the paper's crop separately) and
``fig15_plotted.json``, the exact cells drawn: variable, depth, layer,
position, iia, n. Each table has one row per (depth, position, example); each
heatmap cell is the mean IIA of one (depth, position) for one variable over
every pair of the table, which is what the paper's cells report (Appendix C).
The table holds the paper's own 4096 pairs (``build_dataset.py``). A null
value is left out of its cell's mean and of its ``n``. One specification
draws the whole figure, so no panel is drawn alone
(``docs/paper_replications.md``, The figures).

The figure is a command rather than a workflow step because a step's outputs
must stay inside its own directory (workflow spec §2.3), and the committed
figure sits under ``artifacts/figures/arithmetic_fig15/``. ``main(inputs, outputs)`` is still the step
signature (inputs ``iia_output_day``, ``iia_offset``, ``iia_input_day``;
outputs ``replication``, ``plotted``), so the same function serves either
way.

Rows are aggregated through [`causalab.io.step_record.aggregate`][], the
same reduction the shipped ``select`` and ``workflow_figures`` scripts use, so
a value quoted from ``plotted`` and one chosen from the table by a later step
cannot disagree about what a row is.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import EXAMPLE_COLUMN, aggregate, axes_for

__all__ = ["main", "cli"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/arithmetic_fig15/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "arithmetic_fig15"
FIGURES = PAPERS / "artifacts" / "figures" / "arithmetic_fig15"

DEPTH_AXIS = "axes.depth"
POSITION_AXIS = "positions.tap"
#: The paper's panel order and titles; each variable is one input table.
VARIABLES = ("output_day", "offset", "input_day")
TITLES = {
    "output_day": "(a) Output Day",
    "offset": "(b) Offset",
    "input_day": "(c) Input Day",
}
#: The paper's x-axis labels, in its order: the offset word, the day word, the
#: last token.
POSITIONS = ("number", "input", "last_token")
KEYS = ["variable", "depth", "position"]
#: The outputs ``main`` writes.
OUTPUTS = ("replication", "plotted")


def position_name(coordinate: Any) -> str:
    """The paper's name for one ``positions.tap`` coordinate — whichever
    rendering the table carries, the spec's distinguishing pair is in it."""
    text = (
        coordinate
        if isinstance(coordinate, str)
        else json.dumps(coordinate, sort_keys=True)
    )
    if "offset" in text:
        return "number"
    if "input_day" in text:
        return "input"
    if "index" in text:
        return "last_token"
    raise StepError(f"unrecognised position coordinate {coordinate!r}")


def depth_label(depth: int) -> str:
    """The paper's y tick: depth 0 is the embedding output, depth d the
    output of block d-1."""
    return "Embed" if depth == 0 else f"L{depth - 1}"


def load_scan(inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Path]]:
    """The three IIA tables as one frame with a ``variable`` column, plus
    each variable's path (the sidecar beside it names the axes)."""
    import pandas as pd

    frames = []
    paths: dict[str, Path] = {}
    for variable in VARIABLES:
        path = Path(inputs[f"iia_{variable}"])
        df = frame(path)
        axes = [axis for axis in axes_for(path) if axis in df.columns]
        for axis in (DEPTH_AXIS, POSITION_AXIS):
            if axis not in axes:
                raise StepError(
                    f"{path.name} does not carry the axis {axis!r} (has {axes}); "
                    "this script reads the scan step of workflow.json"
                )
        if EXAMPLE_COLUMN not in df.columns:
            raise StepError(f"{path.name} has no {EXAMPLE_COLUMN!r} column")
        df = df.assign(variable=variable)
        frames.append(df)
        paths[variable] = path
    return pd.concat(frames, ignore_index=True), paths


def grid(df: Any, paths: Mapping[str, Path]) -> Any:
    """Per (variable, depth, position): the mean IIA and the number of rows
    it is a mean over — each variable's rows reduced through ``aggregate``
    against its own table's axes."""
    import pandas as pd

    parts = []
    for variable, path in paths.items():
        rows = df[df["variable"] == variable]
        if rows.empty:
            continue
        per_point, _ = aggregate(rows, path, "value")
        counts = (
            rows[rows["value"].notna()]
            .groupby([DEPTH_AXIS, POSITION_AXIS])
            .size()
            .rename("n")
            .reset_index()
        )
        parts.append(
            per_point.merge(counts, on=[DEPTH_AXIS, POSITION_AXIS], how="left").assign(
                variable=variable
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={DEPTH_AXIS: "depth", "value": "iia"})
    out["position"] = out[POSITION_AXIS].map(position_name)
    out["depth"] = out["depth"].astype(int)
    out["n"] = out["n"].fillna(0).astype(int)
    return out[KEYS + ["iia", "n"]]


def draw_panels(target: Path, table: Any, caption: str) -> None:
    """The replication alone: the three heatmaps side by side, without the
    paper's crop, on one colour scale from 0 to 1 as in the paper."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = sorted(int(d) for d in table["depth"].unique())
    figure, axes = plt.subplots(
        1,
        len(VARIABLES),
        figsize=(3.4 * len(VARIABLES) + 1.0, 8.6),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for column, (ax, variable) in enumerate(zip(axes[0], VARIABLES)):
        cells = (
            table[table["variable"] == variable]
            .pivot(index="depth", columns="position", values="iia")
            .reindex(index=depths, columns=list(POSITIONS))
        )
        matrix = cells.to_numpy(dtype=float)
        image = ax.imshow(
            matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis"
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value == value:  # not NaN
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value < 0.6 else "black",
                    )
        ax.set_xticks(
            range(len(POSITIONS)), POSITIONS, rotation=30, ha="right", fontsize=8
        )
        ax.set_yticks(
            range(len(depths)), [depth_label(d) for d in depths], fontsize=6.5
        )
        ax.set_title(TITLES[variable], fontsize=11)
        ax.set_xlabel("Token Position")
        if column == 0:
            ax.set_ylabel("Layer")
    figure.colorbar(image, ax=list(axes[0]), label="Interchange Intervention Accuracy")
    figure.suptitle(textwrap.fill(caption, 140), fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    """Draw the outputs named in ``outputs`` from the three tables in
    ``inputs``: ``replication``, ``plotted``."""
    import pandas as pd

    unknown = sorted(set(outputs) - set(OUTPUTS))
    if unknown:
        raise StepError(
            f"unknown outputs {unknown}; this script writes {list(OUTPUTS)}"
        )
    df, paths = load_scan(inputs)
    n_pairs = int(df[EXAMPLE_COLUMN].nunique())
    table = grid(df, paths)
    table["layer"] = table["depth"].map(depth_label)
    table["variable"] = pd.Categorical(table["variable"], categories=list(VARIABLES))
    table["position"] = pd.Categorical(table["position"], categories=list(POSITIONS))
    table = table.sort_values(KEYS).reset_index(drop=True)
    table["variable"] = table["variable"].astype(str)
    table["position"] = table["position"].astype(str)
    table = table[["variable", "depth", "layer", "position", "iia", "n"]]

    caption = (
        "causalab: Llama-3.1-8B bf16, whole residual stream interchanged; "
        f"n = {n_pairs} pairs"
    )
    if "replication" in outputs:
        draw_panels(Path(outputs["replication"]), table, caption)
    if "plotted" in outputs:
        write_frame(table, Path(outputs["plotted"]))


def cli(argv: list[str] | None = None) -> int:
    """The command line: read ``--artifacts``, write the committed figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: scan/ is read; the figure goes to artifacts/figures/",
    )
    args = parser.parse_args(argv)
    root = args.artifacts
    main(
        {f"iia_{v}": root / "scan" / f"iia_{v}.json" for v in VARIABLES},
        {
            "replication": FIGURES / "fig15_replication.png",
            "plotted": FIGURES / "fig15_plotted.json",
        },
    )
    print(FIGURES / "fig15_replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
