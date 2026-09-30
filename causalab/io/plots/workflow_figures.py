"""Figures over a metric table, as a workflow step script.

```json
"scan_heatmap": {
  "type": "script",
  "script": {"module": "causalab.io.plots.workflow_figures"},
  "inputs": {"table": {"step": "locate", "file": "iia.json"},
             "plot": "heatmap", "x": "sites.target.layers", "y": "positions.tap"},
  "outputs": {"figure": "scan_iia.png", "plotted": {"file": "scan_iia.json"}}
}
```

Two kinds cover the pipeline: scan grids (``heatmap``) and metric-vs-axis curves
(``lines``). In v1 that was a closed spec enum; here it is just what this script
does, so a third kind is another script rather than a spec change.

**Structured values.** A ``class_probs`` or ``token_logits`` row stores its
value as a JSON object with one entry per group or token, which no axis can
draw. ``key`` names the entry to plot; the drawn column is that entry's number
and the colorbar or y label says which entry it was.

**Output format.** ``figure`` is the rendered image — ``.png`` by default and
preferred, ``.pdf`` when a vector figure is actually wanted
([`causalab.io.plots.figure_format`][]). ``.html`` is a legal output *format*
but not for this script: it is matplotlib, so an interactive figure needs a
plotly-based script instead. ``plotted`` is optional and holds the
**exact rows that were drawn**: a figure carries no record, so declaring the
numbers beside it is what makes the picture checkable and lets a later step
reference what it showed.

Rows are aggregated exactly as `causalab.workflow.scripts.select` does —
both call [`causalab.io.step_record.aggregate`][] — so a figure and a value
chosen from the same table can never disagree about what a row is.

**A figure must cover every axis of what it renders.** An uncovered axis
silently collapses into duplicate cells, averaging over a dimension the reader
cannot see. v1 refused that at load; here it is refused at the step, which is
the honest place now that the axes are read from published data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from causalab.io.plots.figure_format import normalize_figure_format
from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import aggregate, axes_for

__all__ = ["main"]

KINDS = ("heatmap", "lines")


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table_path = Path(inputs["table"])
    kind = str(inputs.get("plot", "heatmap"))
    if kind not in KINDS:
        raise StepError(f"'plot' is one of {list(KINDS)}, got {kind!r}")
    value_column = str(inputs.get("value", "value"))
    key = inputs.get("key")
    if key is not None and not isinstance(key, str):
        raise StepError("'key' names one entry of a structured value column")
    x = inputs.get("x")
    if not isinstance(x, str):
        raise StepError("'x' names the horizontal-axis column")
    y = inputs.get("y")
    series = inputs.get("series")
    if kind == "heatmap" and not isinstance(y, str):
        raise StepError("a heatmap needs 'y'")
    if "figure" not in outputs:
        raise StepError(
            "declare a 'figure' output (.png preferred, .pdf for vector) — the "
            "image is what this step renders"
        )
    target = Path(outputs["figure"])
    # validates the suffix and pins the png-over-pdf preference in one place
    fmt = normalize_figure_format(target.suffix)
    if fmt == "html":
        # `.html` is a legal *document* output (§2.5) — it just needs an
        # interactive renderer. This one is matplotlib, so say so rather than
        # letting savefig raise about supported formats.
        raise StepError(
            "this renderer is matplotlib and writes .png or .pdf; .html needs "
            "an interactive script (see causalab.io.plots.distance_plots for "
            "the plotly idiom)"
        )

    df = frame(table_path)
    if value_column not in df.columns:
        raise StepError(
            f"{table_path.name} has no column {value_column!r} "
            f"(has {sorted(map(str, df.columns))})"
        )
    if key is not None:
        df[value_column] = df[value_column].map(
            lambda entry: _entry(entry, key, table_path.name, value_column)
        )
    label = value_column if key is None else f"{value_column}[{key}]"
    axes = [axis for axis in axes_for(table_path) if axis in df.columns]
    covered = {c for c in (x, y, series) if isinstance(c, str)}
    for column in covered:
        if column not in df.columns:
            raise StepError(
                f"column {column!r} is not in {table_path.name} "
                f"(has {sorted(map(str, df.columns))})"
            )
    uncovered = [axis for axis in axes if axis not in covered]
    if uncovered:
        raise StepError(
            f"the plot leaves the axes {sorted(uncovered)} uncovered — every "
            "sweep axis must be x, y or series, or the figure silently averages "
            "over a dimension the reader cannot see"
        )
    table, _ = aggregate(df, table_path, value_column)

    figure, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if kind == "heatmap":
        grid = table.pivot_table(
            index=str(y), columns=x, values=value_column, aggfunc="mean"
        )
        image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
        ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
        ax.set_xlabel(x)
        ax.set_ylabel(str(y))
        figure.colorbar(image, ax=ax, label=label)
    else:
        if isinstance(series, str):
            for value, group in table.groupby(series, sort=True):
                ordered = group.sort_values(x)
                ax.plot(
                    ordered[x],
                    ordered[value_column],
                    marker="o",
                    label=f"{series}={value}",
                )
            ax.legend()
        else:
            ordered = table.sort_values(x)
            ax.plot(ordered[x], ordered[value_column], marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(label)
    ax.set_title(f"{table_path.name} — {label}")

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)

    if "plotted" in outputs:
        write_frame(table, Path(outputs["plotted"]))


def _entry(entry: Any, key: str, table: str, column: str) -> float:
    """One number out of a structured metric value.

    ``class_probs`` and ``token_logits`` serialise their per-group values as a
    JSON object string in the ``value`` column (``causalab.neural.shared.results``). A row
    that is already a mapping is accepted too, so a table another script
    rewrote does not have to re-serialise."""
    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except ValueError as error:
            raise StepError(
                f"{table}: column {column!r} is not a JSON object, so 'key' "
                "does not apply"
            ) from error
    if not isinstance(entry, dict):
        raise StepError(
            f"{table}: column {column!r} holds a plain number, so 'key' does not apply"
        )
    if key not in entry:
        raise StepError(
            f"{table}: no entry {key!r} in column {column!r} "
            f"(has {sorted(map(str, entry))})"
        )
    return float(entry[key])
