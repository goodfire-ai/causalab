"""Draw the component scan and the two fitted DBM masks of the package.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/mcqa_components_dbm/figures.py        # reads artifacts/output/mcqa_components_dbm/, writes artifacts/figures/mcqa_components_dbm/

It reads the steps of ``artifacts/output/mcqa_components_dbm/`` and
``artifacts/data/mcqa_components_dbm/component_iia_onboarding09_original.json``,
the values of onboarding 09's scan that ``copy_original.py`` copies, and
writes, under ``artifacts/figures/mcqa_components_dbm/``:

* ``components_original.png``: onboarding 09's scan, drawn from the copy in
  the style of the replication;
* ``components_replication.png``: the ``scan`` step, IIA over layer with one
  line per component, in the colours of the ROME figures (red for attention
  outputs, green for MLP outputs, purple for the residual stream);
* ``components_mask.png``: the held-out IIA of the ``apply``, ``apply_mid``
  and ``apply_hi`` steps over the number of gates their ``fit`` step keeps,
  and the 56 gates of the ringed ``CHOSEN`` weight, one tile per attention
  output and MLP output of each layer;
* ``components_heads.png``: the same for the head gate of layer 22, its
  ``head_fit`` and ``head_apply`` steps, with the 12 heads of the ringed mask
  in a layer by head matrix of the whole model;
* ``components_plotted.json``: every value drawn, one row per cell of the
  Original and of the scan, per gate and per head of each fit (its fitted
  ``theta`` and whether it is kept), and per held-out score.

A scan cell is the mean of the ``match`` table over its 64 held-out pairs. A
gate is kept when its hard mask is on (``theta > 0``, the ``hard`` column of
the fit's ``rank.json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import EXAMPLE_COLUMN, aggregate, axes_for

__all__ = ["main", "cli", "dbm_figure"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/mcqa_components_dbm/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "mcqa_components_dbm"
DATA = PAPERS / "artifacts" / "data" / "mcqa_components_dbm"
FIGURES = PAPERS / "artifacts" / "figures" / "mcqa_components_dbm"
COMPONENT_AXIS = "sites.target.component"
LAYER_AXIS = "sites.target.layers"
#: component -> (colour, label), in the order the legend lists them. The
#: ``mcqa_symbol`` and ``mcqa_pointer`` figures take the residual stream's
#: purple from here.
COMPONENTS: dict[str, tuple[str, str]] = {
    "block_output": ("#6a3d9a", "residual stream (block_output)"),
    "attention_output": ("#d62728", "attention output"),
    "mlp_output": ("#2ca02c", "MLP output"),
}
#: gate name prefix -> the component its sites name.
GATE_COMPONENT = {"g_attn": "attention_output", "g_mlp": "mlp_output"}
#: fit step suffix -> l1 weight: onboarding 09's three, as the workflow sets them.
ARMS = {"": 0.01, "_mid": 0.3, "_hi": 3.0}
#: The l1 weight whose mask both DBM figures draw: the fit the captions of
#: Figures 2 and 3 report, six components at held-out IIA 0.922 and heads 7,
#: 9 and 11 of layer 22 at 0.531.
CHOSEN = 3.0
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
#: The layer of the head gate (``protocols/mcqa_components_dbm_head_fit.json``).
HEAD_LAYER = 22
#: gate component -> the row of ``plot_dbm``'s component mask.
MASK_ROW: dict[str, Literal["attention", "mlp"]] = {
    "attention_output": "attention",
    "mlp_output": "mlp",
}
COLUMNS = ["panel", "l1", "component", "layer", "head", "value", "kept", "n"]
#: the integer columns, nullable: a scan cell has no head, a gate no pair count
INTEGERS = {"layer": "Int64", "head": "Int64", "n": "Int64"}


def mean_iia(table: Path) -> float:
    """The mean of a per-example ``match`` table, over its eligible rows."""
    rows = json.loads(table.read_text())
    values = [float(r["value"]) for r in rows if r.get("eligible", True)]
    return sum(values) / len(values)


def scan(step: Path) -> Any:
    """Per (component, layer): mean IIA and the number of pairs."""
    table = frame(step / "iia.json")
    if "value" not in table.columns or EXAMPLE_COLUMN not in table.columns:
        raise StepError(f"{step.name}/iia.json is not a per-example metric table")
    axes = [a for a in axes_for(step / "iia.json") if a in table.columns]
    if set(axes) != {COMPONENT_AXIS, LAYER_AXIS}:
        raise StepError(f"{step.name}/iia.json has axes {axes}, not component x layer")
    table = table.assign(value=table["value"].astype(float))
    cells, _ = aggregate(table, step / "iia.json", "value")
    counts = (
        table[table["value"].notna()].groupby(axes).size().rename("n").reset_index()
    )
    cells = cells.merge(counts, on=axes, how="left")
    cells = cells.rename(columns={COMPONENT_AXIS: "component", LAYER_AXIS: "layer"})
    cells["layer"] = cells["layer"].astype(int)
    return cells.assign(panel="scan", l1=None, head=None, kept=None)


def original(copy: Path) -> Any:
    """Per (component, layer): onboarding 09's IIA, from the committed copy."""
    import pandas as pd

    records = json.loads(copy.read_text())["records"]
    return pd.DataFrame(
        [
            {
                "panel": "original",
                "l1": None,
                "component": r["component"],
                "layer": int(r["layer"]),
                "head": None,
                "value": float(r["iia"]),
                "kept": None,
                "n": None,
            }
            for r in records
        ]
    )


def gates(step: Path, l1: float) -> Any:
    """Per component gate: its fitted theta and whether the hard mask keeps it."""
    import pandas as pd

    rows = []
    for row in json.loads((step / "rank.json").read_text()):
        name = row["featurizer"]
        prefix = name.rstrip("0123456789")
        rows.append(
            {
                "panel": "mask",
                "l1": l1,
                "component": GATE_COMPONENT[prefix],
                "layer": int(name[len(prefix) :]),
                "head": None,
                "value": float(row["theta"]),
                "kept": bool(row["hard"]),
                "n": None,
            }
        )
    return pd.DataFrame(rows).sort_values(["component", "layer"])


def heads(step: Path, l1: float, layer: int = 22) -> Any:
    """Per head of the head gate: its fitted theta and whether it is kept."""
    import pandas as pd

    rows = [
        {
            "panel": "heads",
            "l1": l1,
            "component": "attention_premix",
            "layer": layer,
            "head": int(row["unit"]),
            "value": float(row["theta"]),
            "kept": bool(row["hard"]),
            "n": None,
        }
        for row in json.loads((step / "rank.json").read_text())
    ]
    return pd.DataFrame(rows).sort_values("head")


def _save(figure: Any, target: Path) -> None:
    import matplotlib.pyplot as plt

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def draw_scan(
    table: Any,
    target: Path,
    title: str = "swap one component at the answer slot",
    ylabel: str = "IIA (answer letter), 64 held-out pairs",
) -> None:
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for component, (colour, label) in COMPONENTS.items():
        cells = table[table["component"] == component].sort_values("layer")
        ax.plot(
            cells["layer"], cells["value"], marker="o", ms=4, color=colour, label=label
        )
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xticks(range(0, 28, 3))
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=11)
    _save(figure, target)


def model_layers() -> list[Any]:
    """Every layer of the model with its head count, from the model registry."""
    from causalab.io.plots.dbm_figure import DbmLayer
    from causalab.protocol.registry import get_model_info

    info = get_model_info(MODEL)
    return [DbmLayer(index, info.num_heads) for index in range(info.num_layers)]


def dbm_figure(rows: list[dict[str, Any]], panel: str) -> Any:
    """One DBM figure from the rows of ``components_plotted.json``.

    ``panel`` is ``mask`` for the 56 component gates or ``heads`` for the
    head gate of layer 22. The sweep has one point per l1 weight of ``ARMS``:
    the units its fit keeps against its apply step's held-out IIA. The mask
    is the one of the ``CHOSEN`` weight, and every unit of it must have a row.
    """
    from causalab.io.plots.dbm_figure import (
        ComponentMask,
        HeadMask,
        SweepPoint,
        plot_dbm,
    )

    if panel not in ("mask", "heads"):
        raise ValueError(f"panel {panel!r} is neither 'mask' nor 'heads'")
    units = [r for r in rows if r["panel"] == panel]
    held_out = {
        float(r["l1"]): float(r["value"])
        for r in rows
        if r["panel"] == "held_out" and r["component"] == panel
    }
    weights = list(ARMS.values())
    sweep = [
        SweepPoint(
            sum(bool(r["kept"]) for r in units if float(r["l1"]) == weight),
            held_out[weight],
            f"{weight:g}",
        )
        for weight in weights
    ]
    chosen = [r for r in units if float(r["l1"]) == CHOSEN]
    layers = model_layers()
    short = MODEL.split("/")[-1]
    if panel == "mask":
        keys = [(MASK_ROW[r["component"]], int(r["layer"])) for r in chosen]
        expected = [(row, layer.index) for row in MASK_ROW.values() for layer in layers]
        title = (
            f"DBM over the {len(expected)} attention and MLP outputs "
            f"at the answer slot · {short} bf16"
        )
    else:
        keys = [(int(r["layer"]), int(r["head"])) for r in chosen]
        expected = [(HEAD_LAYER, h) for h in range(layers[HEAD_LAYER].heads)]
        title = (
            f"DBM over the {len(expected)} heads of layer {HEAD_LAYER} "
            f"at the answer slot · {short} bf16"
        )
    if sorted(keys) != sorted(expected):
        raise ValueError(
            f"{panel}: the rows at l1 {CHOSEN:g} do not give one value per unit"
        )
    kept = {key: int(bool(r["kept"])) for key, r in zip(keys, chosen)}
    if panel == "mask":
        mask: Any = ComponentMask(
            layers,
            {
                row: [kept[(row, layer.index)] for layer in layers]
                for row in MASK_ROW.values()
            },
        )
    else:
        mask = HeadMask(layers, {HEAD_LAYER: [kept[key] for key in expected]})
    return plot_dbm(mask, sweep, chosen=weights.index(CHOSEN), title=title)


def main(run: Path, data: Path, figures: Path) -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    scanned = scan(run / "scan")
    copied = original(data / "component_iia_onboarding09_original.json")
    masks = {l1: gates(run / f"fit{suffix}", l1) for suffix, l1 in ARMS.items()}
    head_masks = {
        l1: heads(run / f"head_fit{suffix}", l1) for suffix, l1 in ARMS.items()
    }
    iia = {
        l1: mean_iia(run / f"apply{suffix}" / "iia.json") for suffix, l1 in ARMS.items()
    }
    head_iia = {
        l1: mean_iia(run / f"head_apply{suffix}" / "iia.json")
        for suffix, l1 in ARMS.items()
    }
    draw_scan(
        copied,
        figures / "components_original.png",
        title="onboarding 09: swap one component at the answer slot",
        ylabel="IIA (answer letter), 64 other pairs",
    )
    draw_scan(scanned, figures / "components_replication.png")
    held_out = pd.DataFrame(
        [
            {"panel": "held_out", "l1": l1, "component": "mask", "value": v}
            for l1, v in iia.items()
        ]
        + [
            {"panel": "held_out", "l1": l1, "component": "heads", "value": v}
            for l1, v in head_iia.items()
        ]
    )
    # one frame from the rows of every part, so no column's dtype is decided
    # by a part that leaves it empty
    parts = (copied, scanned, *masks.values(), *head_masks.values(), held_out)
    rows = [
        {column: row.get(column) for column in COLUMNS}
        for part in parts
        for row in part.to_dict("records")
    ]
    # the DBM figures draw the rows the JSON records, so the two cannot differ
    for panel, name in (("mask", "components_mask"), ("heads", "components_heads")):
        dbm_figure(rows, panel).savefig(figures / f"{name}.png", dpi=200)
    write_frame(
        pd.DataFrame(rows, columns=COLUMNS).astype(INTEGERS),
        figures / "components_plotted.json",
    )


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--artifacts", type=Path, default=RUN, help="the workflow's output directory"
    )
    args = parser.parse_args(argv)
    main(args.artifacts, DATA, FIGURES)
    print(FIGURES / "components_replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
