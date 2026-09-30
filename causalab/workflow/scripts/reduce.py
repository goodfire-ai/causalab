"""``causalab.workflow.scripts.reduce`` — reduce a metric table under a
declared reduction contract (docs/workflow_protocol.md §2.6).

The post-hoc twin of ``save.reduce``: where that verb runs in the forward
pass over a read's gathered rows, this step runs after the rows are on disk
and under a declaration of what the number *is* — statistical unit, grouping,
weighting, missing-value policy, uncertainty procedure, resampling unit,
repetitions and seed — authored on the step and carried in the workflow
digest:

```json
"facts": {
  "type": "script", "script": {"module": "causalab.workflow.scripts.reduce"},
  "inputs": {"table": {"step": "trace", "file": "aie.json"}},
  "reduction": {
    "estimator": {"kind": "mean"},
    "unit": {"kind": "example", "columns": ["fact"]},
    "group_by": ["sites.target.layers"],
    "weight": null,
    "missing": "exclude",
    "uncertainty": {"kind": "percentile_bootstrap",
                    "resample_unit": {"kind": "example", "columns": ["fact"]},
                    "repetitions": 2000, "seed": 42}
  },
  "outputs": {"table": {"file": "aie_by_layer.json",
                        "columns": {"value": "float64", "n": "int64"}}}
}
```

Inputs: ``table`` (the metric table), optional ``value`` (the column to
reduce, default ``"value"``; a curve estimator's ``estimator.y`` names the
same thing, so declaring both is refused), and ``reduction`` — the authored
block, which the runner delivers under that name. Output: one ``table`` slot, one row per group,
carrying the group coordinates, ``value``, the counts (``n``, ``n_rows``,
``n_missing``, ``n_unmatched``, ``n_excluded``), the record's identity
(``unit``, ``estimand_version`` — what the number *is*; the group coordinates
say which point it came from, so a report claim can bind to the row) and
``lower``/``upper`` when an uncertainty procedure is declared.

The block is re-validated here with the same parser the loader used (rule
12), so a direct call — a test, a notebook — is refused exactly as a document
would be. Column existence is checked against the real table and refused
naming the column and the dimension that named it. Everything outside the
closed estimator set is a user script step that authors the same block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, frame, write_table
from causalab.workflow.reduction import (
    REDUCTION_INPUT,
    ReductionSpecError,
    parse_reduction,
    reduce_frame,
)

__all__ = ["main"]


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    table_path = Path(inputs["table"])
    value_column = str(inputs.get("value", "value"))
    if REDUCTION_INPUT not in inputs:
        raise StepError(
            "no reduction declared — author a 'reduction' block on the step "
            "(workflow spec §2.6); the runner hands it to this script as "
            f"inputs[{REDUCTION_INPUT!r}]"
        )
    try:
        spec = parse_reduction(inputs[REDUCTION_INPUT])
    except ReductionSpecError as err:
        raise StepError(str(err)) from err
    if "table" not in outputs:
        raise StepError("declare a 'table' output — the reduced rows land there")
    if "value" in inputs and spec.estimator.y is not None:
        raise StepError(
            f"inputs.value ({value_column!r}) and reduction.estimator.y "
            f"({spec.estimator.y!r}) both name the column to reduce — declare one "
            "(a field that governs nothing may not be declared)"
        )

    df = frame(table_path)
    if df.empty:
        raise StepError(f"{table_path.name} has no rows to reduce")
    rows = reduce_frame(df, spec, value_column, what=table_path.name)
    write_table(outputs["table"], rows)
