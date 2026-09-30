"""The knockout extension: p(Seattle) after zeroing a ten-layer window of one
component at one token, over window centre x token position, for the
residual stream, the MLP outputs and the attention outputs.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/rome_fig1/knockout_figure.py        # reads artifacts/output/rome_fig1/, writes artifacts/figures/rome_fig1/

It reads the ``knockout_residual``, ``knockout`` (MLP) and
``knockout_attention`` steps from ``artifacts/output/rome_fig1/`` and writes,
under ``artifacts/figures/rome_fig1/``, ``knockout_all.png`` (the three
heatmaps side by side, in the colours of the paper's Figure 1: purple for
the residual stream, green for MLP, red for attention),
``knockout_<component>.png`` (each heatmap alone) and
``knockout_all_plotted.json``, the cells drawn: component, window centre,
token position, the token, the answer's probability under the knockout, and
the number of rows each mean is over. The knockout document runs no clean
forward, so the page gives the clean probability and the figure does not
draw it.

A cell is the ``Seattle`` entry of the document's ``class_probs`` save: the
softmax probability of `` Seattle`` at the last position, the mean over the
inline table's rows (one prompt, so one row). The token labels come from the
run's own location ledger. The inline prompt names no subject, so no token is
starred, while the tracing panels star the tokens they corrupt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import EXAMPLE_COLUMN, aggregate, axes_for

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig1_figure import POSITION_AXIS, position_index, token_labels  # noqa: E402

__all__ = ["main", "cli"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/rome_fig1/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "rome_fig1"
DATA = PAPERS / "artifacts" / "data" / "rome_fig1"
FIGURES = PAPERS / "artifacts" / "figures" / "rome_fig1"
CENTER_AXIS = "axes.center"
COLUMNS = ["layer", "position", "token", "p_ablated", "n"]
#: The knockout document's ``class_probs`` save and the group it names.
TABLE = "p_ablated.json"
GROUP = "Seattle"
#: Component -> (workflow step, colour map, panel title), in the order of the
#: paper's Figure 1 panels (e), (f), (g).
COMPONENTS: dict[str, tuple[str, str, str]] = {
    "residual": ("knockout_residual", "Purples_r", "residual stream"),
    "mlp": ("knockout", "Greens_r", "MLP"),
    "attention": ("knockout_attention", "Reds_r", "attention"),
}


def group_probability(value: Any, table: str) -> float:
    """The ``GROUP`` entry of one ``class_probs`` cell. The table stores the
    per-group mapping as a JSON object string (``causalab.neural.shared.results``);
    a cell that is already a mapping is accepted too.

    An excluded measurement has no value (``value: null`` with ``eligible:
    false``, written by ``causalab.neural.shared.results``). It gives NaN, so
    the cell's mean skips it and its row count leaves it out."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping) or GROUP not in value:
        raise StepError(
            f"{table}: a class_probs cell without a {GROUP!r} group: {value!r}"
        )
    return float(value[GROUP])


def load(step: Path) -> tuple[Any, dict[int, str]]:
    """Per (centre, position): mean p(Seattle) under the knockout and its row
    count, plus the token labels."""
    ablated = frame(step / TABLE)
    if "value" not in ablated.columns or EXAMPLE_COLUMN not in ablated.columns:
        raise StepError(f"{step.name}/{TABLE} is not a per-example metric table")
    axes = [a for a in axes_for(step / TABLE) if a in ablated.columns]
    for axis in (CENTER_AXIS, POSITION_AXIS):
        if axis not in axes:
            raise StepError(
                f"{step.name}/{TABLE} does not carry the axis {axis!r} (has {axes})"
            )
    ablated = ablated.assign(
        p=[group_probability(v, f"{step.name}/{TABLE}") for v in ablated["value"]]
    )
    per_cell, _ = aggregate(ablated, step / TABLE, "p")
    counts = (
        ablated[ablated["p"].notna()].groupby(axes).size().rename("n").reset_index()
    )
    table = per_cell.merge(counts, on=axes, how="left")
    table = table.rename(columns={CENTER_AXIS: "layer", "p": "p_ablated"})
    table["position"] = table[POSITION_AXIS].map(position_index)
    table["layer"] = table["layer"].astype(int)
    labels = token_labels(step / "location_ledger.json")
    table["token"] = table["position"].map(labels)
    table["n"] = table["n"].fillna(0).astype(int)
    return (
        table[COLUMNS].sort_values(["position", "layer"]).reset_index(drop=True),
        labels,
    )


def _panel(
    figure: Any, ax: Any, table: Any, labels: dict[int, str], component: str
) -> None:
    """One heatmap: p(Seattle) over window centre (x) and token (y); dark is
    the fact lost."""
    positions = sorted(labels)
    cells = table.pivot(index="position", columns="layer", values="p_ablated").reindex(
        index=positions
    )
    layers = [int(c) for c in cells.columns]
    _, cmap, title = COMPONENTS[component]
    image = ax.imshow(
        cells.to_numpy(dtype=float),
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=(min(layers) - 0.5, max(layers) + 0.5, len(positions) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(positions)), [labels[p] for p in positions], fontsize=9)
    ax.set_xticks([layer for layer in layers if layer % 5 == 0])
    ax.set_title(f"zeroing 10 {title} layers at one token", fontsize=11)
    ax.set_xlabel(f"center of interval of 10 zeroed {title} layers", fontsize=9)
    figure.colorbar(image, ax=ax).set_label("p(Seattle) after knockout", fontsize=9)


def draw(
    target: Path,
    tables: Mapping[str, Any],
    labels: dict[int, str],
    caption: str,
) -> None:
    """The components of ``tables`` side by side, in ``COMPONENTS`` order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = [c for c in COMPONENTS if c in tables]
    figure, axes = plt.subplots(
        1,
        len(components),
        figsize=(7 * len(components), 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, component in zip(axes[0], components):
        _panel(figure, ax, tables[component], labels, component)
    figure.suptitle(caption, fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    """``inputs``: one step directory per component name of ``COMPONENTS``.
    ``outputs``: ``figure`` (all components), optional ``<component>`` (one
    each) and ``plotted`` (the cells of every component)."""
    import pandas as pd

    tables: dict[str, Any] = {}
    labels: dict[int, str] = {}
    for component in COMPONENTS:
        if component in inputs:
            tables[component], labels = load(Path(inputs[component]))
    caption = "causalab: GPT-2 XL fp32, one intervened forward per cell"
    draw(Path(outputs["figure"]), tables, labels, caption)
    for component, table in tables.items():
        if component in outputs:
            draw(Path(outputs[component]), {component: table}, labels, caption)
    if "plotted" in outputs:
        plotted = pd.concat(
            [t.assign(component=c) for c, t in tables.items()], ignore_index=True
        )
        write_frame(plotted[["component", *COLUMNS]], Path(outputs["plotted"]))


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: the three knockout steps are read; the figures go to artifacts/figures/",
    )
    args = parser.parse_args(argv)
    root = args.artifacts
    main(
        {component: root / step for component, (step, _, _) in COMPONENTS.items()},
        {
            "figure": FIGURES / "knockout_all.png",
            **{c: FIGURES / f"knockout_{c}.png" for c in COMPONENTS},
            "plotted": FIGURES / "knockout_all_plotted.json",
        },
    )
    print(FIGURES / "knockout_all.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
