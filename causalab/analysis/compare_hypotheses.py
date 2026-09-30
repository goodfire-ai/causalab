"""Compare saved neural outputs with saved symbolic hypothesis predictions.

This module deliberately performs no model forward or fitting.  The public
function consumes ordinary row mappings so it can be used by a workflow script
or directly in a CPU test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

from causalab.io.step_io import write_table

# the metric-row schema (neural/shared/results.py ``MetricTable._row``); every
# other column on a neural row is a sweep coordinate, keyed by its full axis id
METRIC_COLUMNS = frozenset(
    {
        "example_id",
        "metric",
        "value",
        "unit",
        "estimand_version",
        "eligible",
        "reason_code",
        "step",
        "matched",
    }
)


def coordinate_key(row: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """The point a neural row came from: its sorted coordinate columns."""
    import json

    return tuple(
        sorted(
            (
                column,
                value
                if isinstance(value, (int, float, str, bool))
                else json.dumps(value, sort_keys=True),
            )
            for column, value in row.items()
            if column not in METRIC_COLUMNS
        )
    )


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    """Workflow adapter for native saved-output comparisons."""
    from transformers import AutoTokenizer

    from causalab.io.tables import read_table

    tokenizer = AutoTokenizer.from_pretrained(
        inputs["tokenizer"],
        revision=inputs["tokenizer_revision"],
        local_files_only=True,
    )
    rows = compare_saved_outputs(
        read_table(Path(inputs["pairs"])),
        read_table(Path(inputs["predictions"])),
        read_table(Path(inputs["neural"])),
        tokenizer=tokenizer,
        target=inputs["target"],
        alternatives=inputs["alternatives"],
        metric=inputs["metric"],
        split=inputs.get("split"),
    )
    write_table(Path(outputs["comparisons"]), rows)


def compare_saved_outputs(
    pairs: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    neural: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    target: str,
    alternatives: Sequence[str],
    metric: str,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Join native top-1 rows to exact pairs and frozen symbolic predictions.

    Returns one scalar difference per alternative and pair, with both agreement
    scores. A neural row's point is its coordinate columns (every column outside
    the metric-row schema); each point must cover the exact pair table, and the
    coordinates pass through to the result rows. Reduce separately by the sweep
    coordinates, family, split and alternative. A prediction's answer forms
    are tokenized as written, as the intervention's exact-match metric
    tokenizes its column (§2.10).
    """
    import json

    from causalab.protocol.answers import column_token_ids

    names = [target, *alternatives]
    if isinstance(alternatives, str) or len(set(names)) != len(names):
        raise ValueError("target and alternatives must be distinct names")
    if not alternatives:
        raise ValueError("at least one alternative is required")

    def identity(row):
        value = row.get("example_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("example_id must be a nonempty string")
        return value

    by_id = {}
    for pair in pairs:
        if split is not None and pair.get("split") != split:
            continue
        key = identity(pair)
        if key in by_id:
            raise ValueError(f"duplicate pair example_id: {key}")
        for field in ("pair_id", "family", "split"):
            if not pair.get(field):
                raise ValueError(f"pair {key} is missing {field}")
        by_id[key] = pair
    if not by_id:
        raise ValueError("cannot compare an empty pair table")
    symbolic = {}
    for prediction in predictions:
        if split is not None and prediction.get("split") != split:
            continue
        key = identity(prediction)
        name = prediction["hypothesis_id"]
        if name not in names:
            continue
        if key not in by_id:
            raise ValueError(f"prediction has unknown example_id: {key}")
        if (key, name) in symbolic:
            raise ValueError(f"duplicate prediction: {key}, {name}")
        for field in ("pair_id", "family", "split"):
            if prediction.get(field) != by_id[key][field]:
                raise ValueError(f"prediction {key} disagrees on {field}")
        forms = prediction.get("answer_forms")
        if not isinstance(forms, list) or not forms:
            raise ValueError(f"prediction {key}, {name} has no answer forms")
        symbolic[key, name] = set(column_token_ids(tokenizer, forms))
    if set(symbolic) != {(key, name) for key in by_id for name in names}:
        raise ValueError("predictions must cover every pair and hypothesis")

    result = []
    seen = set()
    for output in neural:
        if output.get("metric") != metric:
            continue
        key = identity(output)
        if key not in by_id:
            raise ValueError(f"neural output has unknown example_id: {key}")
        point = coordinate_key(output)
        if (point, key) in seen:
            raise ValueError(f"duplicate neural output: {point}, {key}")
        seen.add((point, key))
        eligible = output.get("eligible")
        if not isinstance(eligible, bool):
            raise ValueError("neural output must declare eligibility")
        token = None
        if eligible:
            value = output["value"]
            if isinstance(value, str):
                value = json.loads(value)
            indices = value.get("indices") if isinstance(value, dict) else None
            if not isinstance(indices, list) or len(indices) != 1:
                raise ValueError("expected native top_k with k=1")
            token = indices[0]
            if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                raise ValueError("top-1 index must be a nonnegative integer")
        elif not output.get("reason_code"):
            raise ValueError("ineligible output must record a reason_code")
        for alternative in alternatives:
            target_ids, alternative_ids = (
                symbolic[key, target],
                symbolic[key, alternative],
            )
            if target_ids != alternative_ids and target_ids & alternative_ids:
                raise ValueError("distinct answers have overlapping token forms")
            target_score = float(token in target_ids) if eligible else None
            alternative_score = float(token in alternative_ids) if eligible else None
            result.append(
                {
                    **output,
                    **{
                        field: by_id[key][field]
                        for field in ("pair_id", "family", "split")
                    },
                    "target": target,
                    "alternative": alternative,
                    "target_score": target_score,
                    "alternative_score": alternative_score,
                    "distinguishing": target_ids != alternative_ids,
                    "neural_token_id": token,
                    "target_token_ids": sorted(target_ids),
                    "alternative_token_ids": sorted(alternative_ids),
                    "source_metric": output["metric"],
                    "metric": "hypothesis_accuracy_difference",
                    "unit": "fraction",
                    "estimand_version": "hypothesis_accuracy_difference/v1",
                    "value": target_score - alternative_score if eligible else None,
                }
            )
    points = {point for point, _ in seen}
    if not points or seen != {(point, key) for point in points for key in by_id}:
        raise ValueError("each neural point must cover the exact pair table")
    return result


__all__ = ["compare_saved_outputs", "main"]
