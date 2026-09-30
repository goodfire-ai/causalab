"""Read and write a workflow step's ``_step.json`` record.

The record identifies the step and its declared products. Shared aggregation
rules let selection and plotting scripts read metric rows consistently."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "EXAMPLE_COLUMN",
    "SIDECAR",
    "aggregate",
    "axes_for",
    "implied_reduction",
    "read_sidecar",
    "write_sidecar",
]

SIDECAR = "_step.json"


def read_sidecar(table_path: Path) -> dict[str, Any]:
    """The record for the step that wrote ``table_path``, or ``{}``.

    Absent is not an error: a script may be pointed at a table nobody's runner
    produced (a pinned file under the repo root, a fixture), and the reduction
    still has to work — it just has no axes to group by."""
    candidate = Path(table_path).parent / SIDECAR
    if not candidate.is_file():
        return {}
    try:
        with candidate.open() as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def axes_for(table_path: Path) -> tuple[str, ...]:
    """The sweep-axis column ids of the step that wrote ``table_path``."""
    record = read_sidecar(table_path)
    axes = record.get("axes")
    if not isinstance(axes, list):
        return ()
    return tuple(str(axis) for axis in axes)


#: The column a protocol run stamps per example (the base row's label, spec
#: §2.2, ``protocol/examples.py``). Its presence is what makes "mean over
#: examples" a meaningful reduction.
EXAMPLE_COLUMN = "example_id"


def implied_reduction(axes: tuple[str, ...]) -> dict[str, Any]:
    """The reduction [`aggregate`][] performs, **declared** in the workflow
    spec's §2.6 vocabulary — what a step that authors nothing implicitly does
    in cases 1 and 2 below:

    ``{estimator: mean, unit: row, group_by: <the sidecar's axes>,
    weight: null, missing: exclude, uncertainty: none}``

    The unit is ``row``, not ``example`` — and that is the finding, not a
    typo. ``select`` describes itself as "mean over examples", and the two
    coincide exactly when a table holds **one row per example per group**,
    which every non-windowed metric table does. A windowed metric writes
    several rows per example (``add_windowed``), and there the arithmetic is a
    mean over rows: an example with more positions weighs more. The
    declaration says what is computed; whether the unit *should* be
    ``example`` is a numbers-moving change and is not made here.

    Written as data rather than imported from the workflow layer because
    ``io/`` sits below it (docs/CODEBASE.md §1). The equality of this
    declaration, run through the built-in ``causalab.workflow.scripts.reduce``,
    with [`aggregate`][]'s output on the same table is a test
    (``tests/workflow/test_reduction.py``): if today's behaviour could not be
    written in the vocabulary, the vocabulary would be wrong. ``aggregate``'s
    own arithmetic is deliberately untouched — a table an unauthored step
    reduces today reduces to the same bytes tomorrow — because this module is
    imported by hashed script modules without being hashed itself (spec §7)."""
    return {
        "estimator": {"kind": "mean"},
        "unit": {"kind": "row"},
        "group_by": list(axes),
        "weight": None,
        "missing": "exclude",
        "uncertainty": {"kind": "none"},
    }


def aggregate(
    df: Any, table_path: Path, value_column: str
) -> tuple[Any, tuple[str, ...]]:
    """The table a reduction should work on, plus the axes it grouped by.

    Three cases, and the discriminator is the **data** rather than the kind of
    step that produced it (v1 special-cased a ``transform`` producer by type):

    1. the producer published sweep axes → group by them, mean over the rest;
    2. no axes but an ``example_id`` column → the whole table is one group, so the
       mean over examples is the single row to rank;
    3. no axes and no ``example_id`` column → the rows **are** the unit. A script
       that wrote one row per principal component already decided what a row
       means, and re-aggregating would collapse exactly the rows a consumer
       wants to choose between.

    Shared by ``select`` and ``plot`` on purpose: a figure and the value chosen
    from the same table must never disagree about what a row is.

    Cases 1 and 2 are one reduction in the workflow spec's §2.6 vocabulary —
    [`implied_reduction`][] spells it out — and pandas' ``skipna`` default is
    its ``missing: exclude`` with the excluded count unrecorded. Case 3 is not a
    reduction at all. A step that wants the statistical unit, the grouping, the
    missing policy or an interval *declared* authors a ``reduction`` block on
    the built-in ``causalab.workflow.scripts.reduce`` instead; nothing here
    changes for a step that does not.
    """
    import pandas as pd

    axes = tuple(axis for axis in axes_for(table_path) if axis in df.columns)
    if axes:
        return (
            df.groupby(list(axes), sort=True)[value_column].mean().reset_index(),
            axes,
        )
    if EXAMPLE_COLUMN in df.columns:
        return pd.DataFrame([{value_column: df[value_column].mean()}]), ()
    return df, ()


def write_sidecar(step_dir: Path, entry: Any) -> None:
    """Publish one step's record beside its outputs (workflow spec §4).

    ``axes`` is the load-bearing field: it is how a downstream script groups a
    swept table by its coordinate columns without the document model having to
    derive it (§6)."""
    (Path(step_dir) / SIDECAR).write_text(json.dumps(dict(entry), indent=2) + "\n")
