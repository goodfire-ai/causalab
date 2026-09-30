"""``causalab.workflow.scripts.select`` — reduce a metric table to named values.

The stage-1 → stage-2 seam: turn a swept metric table into the scalar(s) the
next protocol needs (the locate → DAS handoff), as data instead of a notebook.

```json
"best": {
  "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
  "inputs": {
    "table": {"step": "locate", "file": "iia.json"},
    "choose": "max",
    "emit": {"best_layer": "sites.target.layers", "best_pos": "positions.tap"}
  },
  "outputs": {"values": {"file": "values.json",
                         "keys": {"best_layer": 0, "best_pos": {"index": -1}}}}
}
```

Rows are grouped by the producing document's **sweep-coordinate columns** — read
from the step's ``_step.json`` (`._sidecar`), not authored — and aggregated
by mean over examples. ``choose`` then picks the best group and ``emit`` reads
that group's columns. The exact rule, including when a table is ranked *as
written*, is `._sidecar.aggregate`.

``choose`` is ``"max"``, ``"min"`` or ``"knee"``. The last is for a saturating
curve — IIA against subspace rank — where the highest score is *not* the answer:
it takes the **cheapest** group within ``tolerance`` (default 0.02) of the best,
ordered by a cost axis:

```json
"inputs": {
  "table": {"step": "fit", "file": "iia.json"},
  "choose": "knee", "order": "featurizers.rot.k", "tolerance": 0.02,
  "emit": {"best_k": "featurizers.rot.k"}
}
```

``order`` defaults to the run's sole numeric sweep axis and is required when
there is more than one. See `_knee`.

Two behaviours v1 had as spec rules and this has as script behaviour, on
purpose: the axes come from published data rather than from the document model,
and the as-written case is decided by the data rather than by the producing
step's type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_record import aggregate
from causalab.io.step_io import StepError, frame, write_values

__all__ = ["main"]

CHOICES = ("max", "min", "knee")

#: Default half-width of the "as good as the best" band for ``choose: "knee"``.
#: 0.02 of an IIA-style score: a two-point difference is inside the run-to-run
#: noise of a fit, so it is not evidence that a larger k bought anything.
DEFAULT_KNEE_TOLERANCE = 0.02


def _decode(value: Any) -> Any:
    """Coordinate cells round-trip through a metric table as JSON when they are
    structured (a swept position spec); decode them back so what lands in the
    values object is what a consuming document can parse."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if hasattr(value, "item"):
        return value.item()
    return value


def _knee(
    grouped: Any,
    axes: tuple[str, ...],
    value_column: str,
    *,
    order: Any,
    tolerance: Any,
) -> Any:
    """The index of the *cheapest* group that is as good as the best one.

    ``max`` answers "which cell scored highest", which is not the question a
    rank sweep asks. The causal protocol's own instruction is to *"choose rank
    from the IIA-versus-k curve, not the highest score"*, and on a saturated
    curve `idxmax` returns whichever near-tied group the table happens to list
    first — one A3B run got k=2 that way, the right answer by luck.

    So: among the groups within ``tolerance`` of the best value, take the one
    with the smallest ``order`` coordinate. ``order`` defaults to the run's
    sole numeric sweep axis, and is **required** when there is more than one —
    "the knee" is meaningless until someone says which axis is the cost.

    Ranks upward like ``max``, because a knee is a saturation point: the curve
    this exists for climbs with k and then flattens. A metric where lower is
    better has no knee in this sense — use ``min``.
    """
    from pandas.api.types import is_numeric_dtype

    numeric = [
        axis
        for axis in axes
        if axis in grouped.columns and is_numeric_dtype(grouped[axis])
    ]
    if order is None:
        if len(numeric) != 1:
            raise StepError(
                "'knee' needs to know which axis is the cost: give 'order' "
                f"(the run's numeric axes are {numeric or 'none'})"
            )
        order = numeric[0]
    order = str(order)
    if order not in grouped.columns:
        raise StepError(
            f"'order' column {order!r} is not in the aggregated table "
            f"({sorted(map(str, grouped.columns))}) — the producing run "
            "carried no such axis"
        )
    if not is_numeric_dtype(grouped[order]):
        raise StepError(
            f"'order' column {order!r} is not numeric, so 'smallest' has no "
            "meaning — a knee needs a cost axis that can be ordered"
        )
    try:
        band = float(tolerance)
    except (TypeError, ValueError):
        raise StepError(f"'tolerance' is a number, got {tolerance!r}") from None
    if band < 0:
        raise StepError(f"'tolerance' is not negative, got {band}")

    from causalab.analysis.selection import smallest_near_best

    try:
        indices = smallest_near_best(
            grouped[value_column].tolist(), grouped[order].tolist(), tolerance=band
        )
    except ValueError as error:
        raise StepError(str(error)) from error
    return grouped.index[indices[0]]


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    table_path = Path(inputs["table"])
    choose = str(inputs.get("choose", "max"))
    if choose not in CHOICES:
        raise StepError(f"'choose' is one of {list(CHOICES)}, got {choose!r}")
    value_column = str(inputs.get("value", "value"))
    emit = inputs.get("emit")
    if not isinstance(emit, Mapping) or not emit:
        raise StepError("'emit' maps output key to the column it reads")

    df = frame(table_path)
    if df.empty:
        raise StepError(f"{table_path.name} has no rows to select from")
    if value_column not in df.columns:
        raise StepError(
            f"{table_path.name} has no column {value_column!r} "
            f"(has {sorted(map(str, df.columns))})"
        )

    grouped, axes = aggregate(df, table_path, value_column)

    if choose == "knee":
        index = _knee(
            grouped,
            axes,
            value_column,
            order=inputs.get("order"),
            tolerance=inputs.get("tolerance", DEFAULT_KNEE_TOLERANCE),
        )
    else:
        index = (
            grouped[value_column].idxmax()
            if choose == "max"
            else grouped[value_column].idxmin()
        )
    # single-row FRAME indexing: a row Series would upcast mixed dtypes and
    # emit integer coordinates as floats, which a consuming document's strict
    # parse then refuses
    row = grouped.loc[[index]]

    values: dict[str, Any] = {}
    for key, column in emit.items():
        if column not in row.columns:
            raise StepError(
                f"emit column {column!r} is not in the aggregated table "
                f"({sorted(map(str, row.columns))}) — the producing run "
                "carried no such axis"
            )
        values[str(key)] = _decode(row[column].iloc[0])
    # only the emitted keys land here: the values object is what a consuming
    # document reads by name, and a winning score nobody declared would be an
    # undeclared key in a file whose shape is part of the record
    write_values(outputs["values"], values)
