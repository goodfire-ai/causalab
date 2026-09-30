"""Figures 4b, 5b, 6b and 13: this replication's curves, one image per panel.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/lookbacks/lookbacks_figure.py            # reads artifacts/output/lookbacks/, writes artifacts/figures/lookbacks/

It reads the per-example ``match`` tables the twelve protocol steps wrote under
``artifacts/output/lookbacks/``: the two baselines (is the un-intervened model
right on the original and on the counterfactual prompt?), ``answer_lookback``
(Figure 4b: ``iia_pointer``, ``iia_payload``), ``binding_lookback`` (Figure
5b: ``iia_binding``) and ``binding_source`` (Figure 6b: ``iia_late``, the
drink tokens frozen in the order the authors' script applies its writes, and
``iia_frozen``, the intended freeze; Figure 13: ``iia_unfrozen``). It writes,
under ``artifacts/figures/lookbacks/``, ``lookbacks_4b.png``,
``lookbacks_5b.png``, ``lookbacks_6b.png`` and ``lookbacks_13.png`` (each
panel alone, which the page shows beside the paper's crop of it) and
``lookbacks_plotted.json``, the points drawn: figure, curve, layer, the mean
IIA over every pair and over the pairs the model answers correctly on both
prompts, with both counts, and the paper's value at that layer where the
paper samples it
(``artifacts/data/lookbacks/lookbacks_prakash2025_values.json``, written by
``paper_values.py``). It prints what the two ``screen_*`` steps found: how
many of candidates 0 to 79 of each design the model answers correctly on both
prompts. No curve reads them.

The paper reports each curve over 80 pairs the model answers correctly on
both prompts (Section 3.1). The tables here hold the paper's 80 validation
pairs unfiltered, so this script makes the restriction as a join. Every
``match`` table carries one row per (point, example), and the baseline step
over the same split carries the same ``example`` ids. The correct-pair curve
is drawn solid; the curve over every pair is drawn dotted only where it
differs. Each step runs once per 40-row split of its table
(``<step>_first``, ``<step>_second``; see ``build_dataset.py``), and a curve
is the mean over the rows of both splits.

The figure is a command and not a workflow step, because a step's outputs
must stay inside its own directory (workflow spec §2.3) and the committed
figures are under ``artifacts/figures/lookbacks/``. ``main(inputs, outputs)``
keeps the step signature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import EXAMPLE_COLUMN, axes_for

__all__ = [
    "PANELS",
    "curve_rows",
    "curve_table",
    "answered",
    "correct_pairs",
    "screen_report",
    "load",
    "paper_values",
    "with_paper",
    "draw_panels",
    "main",
    "cli",
]

#: ``demos/papers/``: this script sits in ``workflows/scripts/lookbacks/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "lookbacks"
FIGURES = PAPERS / "artifacts" / "figures" / "lookbacks"
VALUES = (
    PAPERS / "artifacts" / "data" / "lookbacks" / "lookbacks_prakash2025_values.json"
)

#: panel -> its title, x label, baseline step, legend place (with the point
#: it is anchored at, where the place alone would cover a curve) and curves
#: ``(curve, step, table, legend, colour)``. The paper draws every
#: full-residual curve in dark grey. The two 4b curves keep grey in two
#: shades so that they read apart where they cross. The intended freeze has
#: no counterpart in the paper and takes a colour of its own.
PANELS: dict[str, dict[str, Any]] = {
    "4b": {
        "title": "Figure 4b: answer lookback",
        "xlabel": "layer: last token patched",
        "baseline": "baseline_restate",
        "legend": "center left",
        "curves": [
            (
                "pointer",
                "answer_lookback",
                "iia_pointer",
                "pointer → other drink",
                "0.15",
            ),
            (
                "payload",
                "answer_lookback",
                "iia_payload",
                "payload → fresh drink",
                "0.6",
            ),
        ],
    },
    "5b": {
        "title": "Figure 5b: binding address + payload",
        "xlabel": "layer: both drink tokens swapped by word",
        "baseline": "baseline_reorder",
        "legend": "upper right",
        "curves": [
            (
                "binding",
                "binding_lookback",
                "iia_binding",
                "drink tokens swapped by word",
                "0.45",
            ),
        ],
    },
    "6b": {
        "title": "Figure 6b: binding source",
        "xlabel": "L: character and container tokens patched at 0..L",
        "baseline": "baseline_restate",
        "legend": "center right",
        # between the intended freeze at 0.975 and the late tail below 0.3
        "legend_anchor": (1.0, 0.66),
        "curves": [
            (
                "late",
                "binding_source",
                "iia_late",
                "frozen at L..79 (authors' order)",
                "0.45",
            ),
            (
                "frozen",
                "binding_source",
                "iia_frozen",
                "frozen at every layer (intended)",
                "C0",
            ),
        ],
    },
    "13": {
        "title": "Figure 13: binding source, no freeze",
        "xlabel": "L: character and container tokens patched at 0..L",
        "baseline": "baseline_restate",
        "legend": "upper right",
        "curves": [
            (
                "unfrozen",
                "binding_source",
                "iia_unfrozen",
                "drinks not frozen",
                "0.45",
            ),
        ],
    },
}
#: The file stem of each panel drawn alone.
PANEL_FILES = {name: f"lookbacks_{name}" for name in PANELS}
KEYS = ["figure", "curve", "layer"]
COLUMNS = KEYS + ["iia_correct", "n_correct", "iia_all", "n_all", "paper"]
SPLITS = ("first", "second")
#: The baseline over candidates 0 to 79 of each design (``build_dataset.py``
#: ``SCREEN``): the authors keep the first 160 candidates the model answers,
#: so the paper's pairs are candidates 80 to 159 when it answers all of them.
SCREEN_STEPS = ("screen_restate", "screen_reorder")
BASELINE_TABLES = ("correct_base", "correct_cf")
MODEL = "Meta-Llama-3-70B-Instruct (fp16)"


def layer_axis(path: Path, df: Any) -> str:
    axes = [axis for axis in axes_for(path) if axis in df.columns]
    if len(axes) != 1:
        raise StepError(
            f"{path.name} carries the axes {axes}; this script reads a one-axis layer scan"
        )
    return axes[0]


def answered(inputs: Mapping[str, Any], step: str) -> tuple[set[str], int]:
    """The examples of one baseline step the un-intervened model answers
    correctly on both prompts, as strings, and the number of examples. A run
    tree may write ``example_id`` as ``"0"`` or as ``0``, and the join in
    `curve_rows` compares the two spellings as one."""
    both: set[str] | None = None
    seen: set[str] = set()
    for table in BASELINE_TABLES:
        df = frame(Path(inputs[f"{step}/{table}"]))
        seen |= set(str(e) for e in df[EXAMPLE_COLUMN])
        right = set(str(e) for e in df.loc[df["value"] == 1.0, EXAMPLE_COLUMN])
        both = right if both is None else both & right
    assert both is not None
    return both, len(seen)


def correct_pairs(inputs: Mapping[str, Any], baseline: str, split: str) -> set[str]:
    """The examples of one split the un-intervened model answers correctly on
    both prompts (`answered`)."""
    return answered(inputs, f"{baseline}_{split}")[0]


def screen_report(inputs: Mapping[str, Any]) -> list[str]:
    """One line per screening step: the pairs the model answers correctly
    on both prompts, of the pairs it screens."""
    lines = []
    for step in SCREEN_STEPS:
        both, n = answered(inputs, step)
        lines.append(f"{step}: {len(both)} of {n} pairs answered on both prompts")
    return lines


def curve_rows(inputs: Mapping[str, Any], step: str, table: str, baseline: str) -> Any:
    """One row per (split, layer, example) with a ``correct`` flag, both
    splits concatenated."""
    import pandas as pd

    parts = []
    for split in SPLITS:
        path = Path(inputs[f"{step}_{split}/{table}"])
        df = frame(path)
        axis = layer_axis(path, df)
        if EXAMPLE_COLUMN not in df.columns:
            raise StepError(f"{path.name} has no {EXAMPLE_COLUMN!r} column")
        correct = correct_pairs(inputs, baseline, split)
        parts.append(
            pd.DataFrame(
                {
                    "layer": df[axis].astype(int),
                    "value": df["value"],
                    "correct": df[EXAMPLE_COLUMN].astype(str).isin(correct),
                    "split": split,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def curve_table(rows: Any) -> Any:
    """Per layer: mean IIA and count over every pair, and over the correct pairs."""

    out = None
    for label, part in (("all", rows), ("correct", rows[rows["correct"]])):
        part = part[part["value"].notna()]
        agg = part.groupby("layer")["value"].agg(["mean", "size"]).reset_index()
        agg = agg.rename(columns={"mean": f"iia_{label}", "size": f"n_{label}"})
        out = agg if out is None else out.merge(agg, on="layer", how="outer")
    for label in ("all", "correct"):
        out[f"n_{label}"] = out[f"n_{label}"].fillna(0).astype(int)
    return out.sort_values("layer").reset_index(drop=True)


def paper_values(path: Path = VALUES) -> dict[tuple[str, str, int], float]:
    """The paper's value per (figure, curve, layer), from the values file
    ``paper_values.py`` writes."""
    records = json.loads(path.read_text())["records"]
    return {(r["figure"], r["curve"], int(r["layer"])): r["accuracy"] for r in records}


def with_paper(table: Any, paper: Mapping[tuple[str, str, int], float]) -> Any:
    """``table`` with a ``paper`` column: the paper's value at a (figure,
    curve, layer) it samples, else ``None``. A paper value whose curve and
    layer the table lacks is an error, since the comparison would drop it."""
    keys = set(zip(table["figure"], table["curve"], table["layer"].astype(int)))
    missing = sorted(set(paper) - keys)
    if missing:
        raise StepError(f"the run has no point for the paper's {missing[:5]}")
    out = table.copy()
    out["paper"] = [
        paper.get((f, c, int(layer)))
        for f, c, layer in zip(out["figure"], out["curve"], out["layer"])
    ]
    out["paper"] = out["paper"].astype(object).where(out["paper"].notna(), None)
    return out


def load(inputs: Mapping[str, Any]) -> Any:
    """Every curve of every panel, in the columns of ``lookbacks_plotted.json``."""
    import pandas as pd

    parts = []
    for name, spec in PANELS.items():
        for curve, step, table_name, _legend, _color in spec["curves"]:
            rows = curve_rows(inputs, step, table_name, spec["baseline"])
            parts.append(curve_table(rows).assign(figure=name, curve=curve))
    table = pd.concat(parts, ignore_index=True)
    table = with_paper(table, paper_values(Path(inputs.get("paper", VALUES))))
    return table[COLUMNS].sort_values(KEYS).reset_index(drop=True)


def population(table: Any, panel: str) -> str:
    """``n = <correct> of <all>`` for one panel: the pairs its solid curves
    are over, of every pair in its table."""
    rows = table[table["figure"] == panel]
    return f"n = {int(rows['n_correct'].max())} of {int(rows['n_all'].max())}"


def draw_panels(
    target: Path, table: Any, panels: Sequence[str], model: str = MODEL
) -> None:
    """The replication alone: ``panels`` side by side, without the paper's
    crops, from a table in the columns of ``lookbacks_plotted.json``. The
    committed figures draw one panel each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.6 * len(panels), 4.0),
        constrained_layout=True,
        squeeze=False,
    )
    dotted = False
    for column, (ax, name) in enumerate(zip(axes[0], panels)):
        spec = PANELS[name]
        for curve, _step, _table, legend, color in spec["curves"]:
            rows = table[(table["figure"] == name) & (table["curve"] == curve)]
            ax.plot(
                rows["layer"], rows["iia_correct"], color=color, lw=2.2, label=legend
            )
            if (rows["n_correct"] != rows["n_all"]).any():
                dotted = True
                ax.plot(
                    rows["layer"],
                    rows["iia_all"],
                    color=color,
                    lw=1.2,
                    ls=":",
                    alpha=0.8,
                )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, max(table["layer"]))
        ax.set_xticks(range(0, int(max(table["layer"])) + 1, 10))
        ax.set_title(f"{spec['title']} ({population(table, name)})", fontsize=10)
        ax.set_xlabel(spec["xlabel"], fontsize=9)
        if column == 0:
            ax.set_ylabel("Intervention accuracy (IIA)")
        ax.grid(alpha=0.25)
        ax.legend(
            fontsize=8, loc=spec["legend"], bbox_to_anchor=spec.get("legend_anchor")
        )
    note = "pairs the model answers correctly on both prompts"
    if dotted:
        note = f"solid: {note}; dotted: every pair"
    # one panel alone is too narrow for the note on one line
    sep = "; " if len(panels) > 1 else "\n"
    figure.suptitle(f"causalab: {model}{sep}{note}", fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    table = load(inputs)
    model = str(inputs.get("model", MODEL))
    for name in PANELS:
        if name in outputs:
            draw_panels(Path(outputs[name]), table, [name], model)
    if "plotted" in outputs:
        write_frame(table, Path(outputs["plotted"]))
    if all(f"{step}/{t}" in inputs for step in SCREEN_STEPS for t in BASELINE_TABLES):
        print("\n".join(screen_report(inputs)))


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: the step directories are read; the figures go to --figures",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=FIGURES,
        help="where the figures and lookbacks_plotted.json are written",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="the model name the figure title states",
    )
    args = parser.parse_args(argv)
    root = args.artifacts
    inputs: dict[str, Any] = {"model": args.model, "paper": VALUES}
    for split in SPLITS:
        for step in ("baseline_restate", "baseline_reorder"):
            for table in BASELINE_TABLES:
                inputs[f"{step}_{split}/{table}"] = (
                    root / f"{step}_{split}" / f"{table}.json"
                )
        for spec in PANELS.values():
            for _curve, step, table, _legend, _color in spec["curves"]:
                inputs[f"{step}_{split}/{table}"] = (
                    root / f"{step}_{split}" / f"{table}.json"
                )
    for step in SCREEN_STEPS:
        for table in BASELINE_TABLES:
            inputs[f"{step}/{table}"] = root / step / f"{table}.json"
    figures = args.figures
    main(
        inputs,
        {
            **{name: figures / f"{stem}.png" for name, stem in PANEL_FILES.items()},
            "plotted": figures / "lookbacks_plotted.json",
        },
    )
    print(figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
