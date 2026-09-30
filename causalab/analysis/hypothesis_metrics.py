"""Read native answer logits and reduce the declared hypothesis populations."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from causalab.protocol.schema import inline_train_saves


def _missing_logits(pair, reason):
    changing = (
        pair["label"] != pair["base_answer"]
        if pair.get("base_answer") is not None
        else None
    )
    if changing is False:
        reason = "answer_unchanged"
    return {
        "value": None,
        "eligible": False,
        "population_eligible": changing,
        "missing": changing is not False,
        "reason_code": reason,
    }


def _labelled_saves(method):
    """A run's read save entries by label (§2.12): the file stem, which is
    the ``metric`` column of the table the entry writes. A ``train`` entry
    counts as the term it names."""
    out = {}
    for entry in inline_train_saves(method):
        if entry.get("read") is None:
            continue
        stem = str(entry["file_path"]).rsplit("/", 1)[-1]
        label = stem.rsplit(".", 1)[0] if "." in stem else stem
        out.setdefault(label, []).append(entry)
    return out


def _row_coords(coords):
    """A point's coordinates as a metric row spells them: scalars as they are,
    anything else as sorted-key JSON."""
    return {
        axis: value
        if isinstance(value, (str, int, float, bool))
        else json.dumps(value, sort_keys=True)
        for axis, value in coords.items()
    }


def read_token_logits(native, item, tokenizer, paths):
    """Read target-minus-original logits from the same frozen output as top-1.

    The run's method is a protocol-4 document: an aggregation lives on the
    save entry that tables it, over the entry's ``read`` on its ``model``.
    The token-logit table has to reduce the same bound read the compared
    top-1 output does.
    """
    from causalab.protocol.answers import column_token_ids

    method = native["method"]
    saves = _labelled_saves(method)
    (source,) = saves[item["source_metric"]]
    output = (source["read"], source["model"])
    names = [
        label
        for label, entries in saves.items()
        if any(
            entry.get("aggregation", {}).get("kind") == "token_logits"
            and (entry["read"], entry["model"]) == output
            for entry in entries
        )
    ]
    name = item.get("logit_metric")
    if name is None:
        if len(names) > 1:
            raise ValueError(
                "Specify logit_metric when several output token-logit metrics exist"
            )
        name = next(iter(names), None)
    if name is not None and name not in names:
        raise ValueError(
            "logit_metric must save token_logits from the compared intervened output"
        )
    pairs = {pair["example_id"]: pair for pair in native["rows"]}
    result = {
        key: _missing_logits(pair, "token_logits_not_saved")
        for key, pair in pairs.items()
    }
    source = {
        "metric": name,
        "definition": "logit(label) - logit(base_answer)",
        "population": "answer-changing pairs",
        "status": "unavailable",
    }
    if name is None:
        return result, source
    entries = saves.get(name, [])
    if len(entries) != 1:
        raise ValueError("Save exactly one table for the hypothesis token-logit metric")
    (entry,) = entries
    spec = entry["aggregation"]
    accuracy = [e["aggregation"] for e in saves.get("iia", []) if "aggregation" in e]
    token_form = accuracy[0].get("token_form") if accuracy else None
    if spec.get("token_form") != token_form:
        raise ValueError(
            "Token logits and exact hypothesis accuracy must use the same token_form"
        )
    path = Path(native["paths"]["run_dir"]) / entry["file_path"]
    source["path"] = str(path)
    if not path.exists():
        return result, source
    paths.append(path)
    rows = json.loads(path.read_text())
    # A row carries no digest: its coordinate columns place it on a point of
    # the receipt (`causalab.neural.shared.results.MetricTable._row`).
    recorded = [_row_coords(point["coords"]) for point in native["receipt"]["points"]]
    axes = {axis for coords in recorded for axis in coords}
    if any(
        {axis: row.get(axis) for axis in axes if axis in row} not in recorded
        or row.get("metric") != name
        for row in rows
    ):
        raise ValueError(
            "Token-logit table contains an unrecorded point or another metric"
        )
    point = _row_coords(native["coords"])
    rows = [row for row in rows if all(row.get(axis) == point[axis] for axis in point)]
    by_id = {}
    for row in rows:
        key = row.get("example_id")
        if key not in pairs or key in by_id:
            raise ValueError(
                "Token-logit rows must name unique pairs from the native dataset"
            )
        if type(row.get("eligible")) is not bool or (
            not row["eligible"] and not row.get("reason_code")
        ):
            raise ValueError(
                "Token-logit rows must declare eligibility and excluded reasons"
            )
        by_id[key] = row

    def token_ids(values):
        return column_token_ids(
            tokenizer,
            values,
            token_form=token_form,
            vocabulary_size=len(tokenizer) if token_form == "id" else None,
        )

    declared = token_ids(spec["tokens"])
    if len(set(declared)) != len(declared):
        raise ValueError("The saved token-logit vocabulary repeats a token identity")
    for key, pair in pairs.items():
        row = by_id.get(key)
        if pair.get("base_answer") is None:
            result[key] = _missing_logits(pair, "original_answer_not_saved")
            continue
        changing = pair["label"] != pair["base_answer"]
        if not changing:
            continue
        if row is None:
            continue
        if not row["eligible"]:
            result[key] = {
                "value": None,
                "eligible": False,
                "population_eligible": False,
                "missing": False,
                "reason_code": row["reason_code"],
            }
            continue
        value = (
            json.loads(row["value"])
            if isinstance(row.get("value"), str)
            else row.get("value")
        )
        indices = value.get("indices") if isinstance(value, dict) else None
        logits = value.get("values") if isinstance(value, dict) else None
        if (
            indices != declared
            or any(type(index) is not int for index in indices)
            or not isinstance(logits, list)
            or len(logits) != len(declared)
            or any(
                type(logit) not in (int, float) or not math.isfinite(logit)
                for logit in logits
            )
        ):
            raise ValueError(
                "Token-logit values must match the declared token IDs and contain finite logits"
            )
        answers = [pair["label"], pair["base_answer"]]
        if token_form != "id":
            answers = [str(answer) for answer in answers]
        target, original = token_ids(answers)
        if target not in declared or original not in declared:
            result[key] = _missing_logits(pair, "answer_token_not_saved")
            continue
        lookup = dict(zip(indices, logits, strict=True))
        result[key] = {
            "value": lookup[target] - lookup[original],
            "eligible": True,
            "population_eligible": True,
            "missing": False,
            "reason_code": None,
        }
    source["status"] = (
        "partial" if any(row["missing"] for row in result.values()) else "available"
    )
    return result, source


def score_rows(rows):
    """Summarize one family and pair filter, keeping absent logits explicit."""
    valid = [row for row in rows if row["eligible"]]
    n = len(valid)
    logits = [
        row.get(
            "logit_diff",
            {
                "value": None,
                "eligible": False,
                "population_eligible": None,
                "missing": True,
                "reason_code": "token_logits_not_saved",
            },
        )
        for row in rows
    ]
    scored = [row["value"] for row in logits if row["eligible"]]
    logit = {
        "value": sum(scored) / len(scored) if scored else None,
        "n": len(scored),
        "total": len(rows),
        "eligible_total": sum(row["population_eligible"] is True for row in logits),
        "unknown_eligibility": sum(
            row["population_eligible"] is None for row in logits
        ),
        "missing": sum(row["missing"] for row in logits),
        "excluded": dict(
            Counter(row["reason_code"] for row in logits if not row["eligible"])
        ),
        "population": "answer-changing pairs",
    }
    if not scored:
        logit["reason"] = (
            "Required token-logit measurements are unavailable"
            if logit["missing"]
            else "No eligible answer-changing pairs"
        )
    result = {
        "target": sum(row["target_score"] for row in valid) / n if n else None,
        "alternative": sum(row["alternative_score"] for row in valid) / n
        if n
        else None,
        "n": n,
        "total": len(rows),
        "excluded": dict(
            Counter(row["reason_code"] for row in rows if not row["eligible"])
        ),
        "logit_diff": logit,
    }
    if not n:
        result["reason"] = "No eligible pairs"
    return result


def mix_scores(family_scores, family_weights, population_counts):
    """Condition the protocol mixture on the requested pairs and metric eligibility."""
    active = {family: weight for family, weight in family_weights.items() if weight > 0}
    missing_population = [
        family for family in active if not population_counts.get(family)
    ]
    families = {family: family_scores[family] for family in active}
    masses = {
        family: weight * families[family]["n"] / population_counts[family]
        if population_counts.get(family)
        else 0
        for family, weight in active.items()
    }

    def normalized(values):
        total = sum(values.values())
        return {
            family: value / total if total else 0 for family, value in values.items()
        }

    effective = normalized(masses)
    result = {
        "target": None,
        "alternative": None,
        "n": sum(score["n"] for score in families.values()),
        "total": sum(score["total"] for score in families.values()),
        "excluded": dict(
            sum((Counter(score["excluded"]) for score in families.values()), Counter())
        ),
        "family_weights": dict(family_weights),
        "effective_family_weights": effective,
        "family_counts": {
            family: {
                "population": population_counts.get(family, 0),
                "selected": score["total"],
                "scored": score["n"],
            }
            for family, score in families.items()
        },
    }
    if missing_population:
        result["reason"] = "No saved pairs for weighted families: " + ", ".join(
            missing_population
        )
    elif not sum(masses.values()):
        result["reason"] = "No eligible pairs in the protocol mixture"
    elif any(
        score[field] is None and effective[family] > 0
        for family, score in families.items()
        for field in ("target", "alternative")
    ):
        result["reason"] = "Required accuracy measurements are unavailable"
    else:
        for field in ("target", "alternative"):
            result[field] = sum(
                effective[family] * score[field]
                for family, score in families.items()
                if effective[family] > 0
            )

    logits = {
        family: score.get(
            "logit_diff",
            {
                "value": None,
                "n": 0,
                "total": score["total"],
                "eligible_total": 0,
                "unknown_eligibility": score["total"],
                "missing": score["total"],
                "excluded": {"token_logits_not_saved": score["total"]},
            },
        )
        for family, score in families.items()
    }
    logit_masses = {
        family: active[family]
        * score.get("eligible_total", score["n"])
        / population_counts[family]
        if population_counts.get(family)
        else 0
        for family, score in logits.items()
    }
    logit_weights = normalized(logit_masses)
    logit = {
        "value": None,
        "n": sum(score["n"] for score in logits.values()),
        "total": sum(score["total"] for score in logits.values()),
        "eligible_total": sum(
            score.get("eligible_total", score["n"]) for score in logits.values()
        ),
        "unknown_eligibility": sum(
            score.get("unknown_eligibility", 0) for score in logits.values()
        ),
        "missing": sum(score.get("missing", 0) for score in logits.values()),
        "excluded": dict(
            sum((Counter(score["excluded"]) for score in logits.values()), Counter())
        ),
        "population": "answer-changing pairs",
        "family_weights": dict(family_weights),
        "effective_family_weights": logit_weights,
        "family_counts": {
            family: {
                "population": population_counts.get(family, 0),
                "selected": score["total"],
                "eligible": score.get("eligible_total", score["n"]),
                "scored": score["n"],
                "missing": score.get("missing", 0),
            }
            for family, score in logits.items()
        },
    }
    if missing_population:
        logit["reason"] = result["reason"]
    elif logit["missing"] or logit["unknown_eligibility"]:
        logit["reason"] = (
            "Required token-logit measurements are unavailable in the protocol mixture"
        )
    elif not sum(logit_masses.values()):
        logit["reason"] = "No eligible answer-changing pairs in the protocol mixture"
    elif any(
        score["value"] is None and logit_weights[family] > 0
        for family, score in logits.items()
    ):
        logit["reason"] = (
            "Required token-logit measurements are unavailable in the protocol mixture"
        )
    else:
        logit["value"] = sum(
            logit_weights[family] * score["value"]
            for family, score in logits.items()
            if logit_weights[family] > 0
        )
    result["logit_diff"] = logit
    return result
