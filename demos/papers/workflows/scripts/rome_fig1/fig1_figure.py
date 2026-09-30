"""Figure 1 (e, f, g): this replication's three heatmaps, together and one by one.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/rome_fig1/fig1_figure.py            # reads artifacts/output/rome_fig1/, writes artifacts/figures/rome_fig1/

It reads the ``state`` step (panel e: one restored residual-stream state) and
the ``window`` step (panels f and g: a ten-layer window of MLP or attention
outputs) from ``artifacts/output/rome_fig1/`` and writes, under
``artifacts/figures/rome_fig1/``, ``fig1_replication.png`` (the three
heatmaps, which the page shows below the paper's crop), ``fig1_state.png``,
``fig1_mlp.png`` and ``fig1_attention.png`` (each heatmap alone), and
``fig1_plotted.json``, the exact values drawn: panel, layer (or window
centre), token position, the token, the mean probability of the answer under
restoration, and the number of noise samples each mean is over.

A value is the paper's ``P_{*, clean h}[o]`` (Section 2.1): the metric tables
hold the cross-entropy of the answer token per row, and ``exp(-ce)`` is its
probability; the mean over the table's ten rows is the mean over the paper's
ten noise samples. The token labels come from the run's own location ledger
(what the engine decoded at each swept index), and a token the ``subject``
position covered (the corrupted ones) is starred, as in the paper.

Restoring the last layer's state at a token before the last cannot reach the
last position, so panel (e) at the last layer holds the corrupted run's
probability at every earlier token, and restoring it at the last token gives
the clean run's. Each panel's colour scale runs from that corrupted
probability to the panel's own maximum, as in ROME's ``plot_trace_heatmap``
(``vmin=low_score``, line 554 of
<https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/experiments/causal_trace.py>). A value below the corrupted probability
draws in the lightest colour.

Rows are aggregated through [`causalab.io.step_record.aggregate`][], the
reduction the shipped ``select`` and ``workflow_figures`` scripts use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import EXAMPLE_COLUMN, aggregate, axes_for

__all__ = ["main", "cli"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/rome_fig1/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "rome_fig1"
FIGURES = PAPERS / "artifacts" / "figures" / "rome_fig1"

#: The ledger's row column: ``example``, where the metric tables say ``example_id``.
LEDGER_EXAMPLE = "example"
POSITION_AXIS = "positions.tap.index"
#: (panel, step, the axis that is the x coordinate, the component it restores)
PANELS = (
    ("e", "state", "sites.restore.layers", "block_output"),
    ("f", "window", "axes.center", "mlp_output"),
    ("g", "window", "axes.center", "attention_output"),
)
TITLES = {
    "e": "(e) restoring one state h(l)",
    "f": "(f) restoring 10 MLP layers",
    "g": "(g) restoring 10 Attn layers",
}
CMAPS = {"e": "Purples", "f": "Greens", "g": "Reds"}
#: The file stem of each panel drawn alone.
PANEL_FILES = {"e": "fig1_state", "f": "fig1_mlp", "g": "fig1_attention"}
XLABELS = {
    "e": "single restored layer within GPT-2 XL",
    "f": "center of interval of 10 restored MLP layers",
    "g": "center of interval of 10 restored Attn layers",
}
COLUMNS = [
    "panel",
    "layer",
    "position",
    "token",
    "p_restored",
    "n",
]
#: The largest spread allowed between the (e) last-layer values at the tokens
#: before the last. Each is the corrupted run, recomputed in its own forward of
#: the same shape, so they agree to the bit on one device; a restored state
#: that leaked to the last position would move them by orders more.
CORRUPTED_SPREAD = 1e-6


def position_index(coordinate: Any) -> int:
    """The token index a ``positions.tap.index`` coordinate names: the swept
    integer itself, or an ``{"index": i}`` object from an earlier run tree."""
    if isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool):
        return int(coordinate)
    if isinstance(coordinate, str):
        try:
            coordinate = json.loads(coordinate)
        except json.JSONDecodeError:
            pass
    if isinstance(coordinate, Mapping) and "index" in coordinate:
        return int(coordinate["index"])
    if isinstance(coordinate, (int, float)):
        return int(coordinate)
    raise StepError(f"unrecognised position coordinate {coordinate!r}")


def token_labels(ledger: Path) -> dict[int, str]:
    """token index -> decoded token, starred where the ``subject`` position
    (the corrupted tokens) covered it; from the ledger's first example."""
    rows = json.loads(Path(ledger).read_text())
    if not isinstance(rows, list) or not rows:
        raise StepError(f"{ledger.name} holds no ledger rows")
    example = min(int(r[LEDGER_EXAMPLE]) for r in rows)
    decoded: dict[int, str] = {}
    corrupted: set[int] = set()
    for r in rows:
        if int(r[LEDGER_EXAMPLE]) != example or r.get("side") != "base":
            continue
        index = int(r["token_index"])
        decoded[index] = str(r["decoded_token"])
        if str(r["constituent"]).startswith("subject"):
            corrupted.add(index)
    if not decoded:
        raise StepError(f"{ledger.name} has no base-side rows")
    return {
        i: f"{t.replace(chr(288), chr(32)).strip() or repr(t)}{'*' if i in corrupted else ''}"
        for i, t in decoded.items()
    }


def probabilities(step: Path, x_axis: str, component: str | None) -> Any:
    """Per (x, position): mean exp(-ce) under restoration and its row count."""
    import numpy as np

    restored = frame(step / "ce_restored.json")
    if "value" not in restored.columns or EXAMPLE_COLUMN not in restored.columns:
        raise StepError(
            f"{step.name}/ce_restored.json is not a per-example metric table"
        )
    axes = [a for a in axes_for(step / "ce_restored.json") if a in restored.columns]
    for axis in (x_axis, POSITION_AXIS):
        if axis not in axes:
            raise StepError(
                f"{step.name}/ce_restored.json does not carry the axis {axis!r} (has {axes})"
            )
    if component is not None:
        if "axes.component" not in restored.columns:
            raise StepError(
                f"{step.name}/ce_restored.json has no axes.component column"
            )
        restored = restored[restored["axes.component"] == component]
        if restored.empty:
            raise StepError(f"{step.name}: no rows for component {component!r}")
    restored = restored.assign(p=np.exp(-restored["value"].astype(float)))
    per_cell, _ = aggregate(restored, step / "ce_restored.json", "p")
    counts = (
        restored[restored["p"].notna()].groupby(axes).size().rename("n").reset_index()
    )
    return per_cell.merge(counts, on=axes, how="left")


def load(root: Path) -> tuple[Any, dict[int, str]]:
    """The cells of every panel, sorted by panel, position and layer, and
    the token labels."""
    import pandas as pd

    labels = token_labels(root / "state" / "location_ledger.json")
    parts = []
    for panel, step, x_axis, component in PANELS:
        table = probabilities(
            root / step, x_axis, component if step == "window" else None
        )
        table = table.rename(columns={x_axis: "layer", "p": "p_restored"})
        table["position"] = table[POSITION_AXIS].map(position_index)
        table["layer"] = table["layer"].astype(int)
        table["panel"] = panel
        table["token"] = table["position"].map(labels)
        table["n"] = table["n"].fillna(0).astype(int)
        parts.append(table[COLUMNS])
    out = pd.concat(parts, ignore_index=True)
    return (
        out.sort_values(["panel", "position", "layer"]).reset_index(drop=True),
        labels,
    )


def corrupted_score(table: Any) -> float:
    """The corrupted run's answer probability: panel (e) at the last layer, at
    the tokens before the last.

    Raises:
        StepError: the table has no such values, or they spread by more than
            ``CORRUPTED_SPREAD``.
    """
    state = table[table["panel"] == "e"]
    if state.empty:
        raise StepError("the table has no panel (e) values")
    last = state[state["layer"] == state["layer"].max()]
    earlier = last[last["position"] < last["position"].max()]["p_restored"]
    earlier = earlier[earlier.notna()]
    if earlier.empty:
        raise StepError("panel (e) has no last-layer value before the last token")
    spread = float(earlier.max() - earlier.min())
    if spread > CORRUPTED_SPREAD:
        raise StepError(
            f"panel (e) at the last layer should equal the corrupted run at every "
            f"token before the last, but its values spread by {spread:.3g}"
        )
    return float(earlier.iloc[0])


def colour_scale(table: Any, panel: str) -> tuple[float, float]:
    """``(vmin, vmax)`` of one panel: the corrupted run's probability, shared
    by every panel, and the panel's own maximum (ROME's ``plot_trace_heatmap``,
    ``vmin=low_score`` and an automatic ``vmax``)."""
    vmin = corrupted_score(table)
    values = table[table["panel"] == panel]["p_restored"]
    values = values[values.notna()]
    top = float(values.max()) if not values.empty else vmin
    return vmin, max(top, vmin + 1e-3)


def draw_panels(
    target: Path, table: Any, labels: dict[int, str], panels: list[str], caption: str
) -> None:
    """The replication alone: ``panels`` side by side, without the paper's
    crop, each panel on its own ``colour_scale``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = sorted(labels)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 3.6),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, panel in zip(axes[0], panels):
        cells = (
            table[table["panel"] == panel]
            .pivot(index="position", columns="layer", values="p_restored")
            .reindex(index=positions)
        )
        layers = [int(c) for c in cells.columns]
        matrix = cells.to_numpy(dtype=float)
        vmin, vmax = colour_scale(table, panel)
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap=CMAPS[panel],
            vmin=vmin,
            vmax=vmax,
            extent=(min(layers) - 0.5, max(layers) + 0.5, len(positions) - 0.5, -0.5),
        )
        ax.set_yticks(range(len(positions)), [labels[p] for p in positions], fontsize=9)
        ax.set_xticks([layer for layer in layers if layer % 5 == 0])
        ax.set_title(TITLES[panel], fontsize=11)
        ax.set_xlabel(XLABELS[panel], fontsize=9)
        figure.colorbar(image, ax=ax).set_label("p(Seattle)", fontsize=9)
    figure.suptitle(caption, fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    root = Path(inputs["artifacts"])
    table, labels = load(root)
    n = int(table["n"].max())
    caption = f"causalab: GPT-2 XL fp32, mean over {n} noise samples"
    if "replication" in outputs:
        draw_panels(
            Path(outputs["replication"]), table, labels, list(PANEL_FILES), caption
        )
    for panel in PANEL_FILES:
        if panel in outputs:
            draw_panels(Path(outputs[panel]), table, labels, [panel], caption)
    if "plotted" in outputs:
        write_frame(table, Path(outputs["plotted"]))


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: state/ and window/ are read; the figure goes to artifacts/figures/",
    )
    args = parser.parse_args(argv)
    root = args.artifacts
    main(
        {
            "artifacts": root,
        },
        {
            "replication": FIGURES / "fig1_replication.png",
            **{panel: FIGURES / f"{stem}.png" for panel, stem in PANEL_FILES.items()},
            "plotted": FIGURES / "fig1_plotted.json",
        },
    )
    print(FIGURES / "fig1_replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
