"""The reduction contract (workflow spec §2.6, checklist rule 12).

Without the contract, a number published from a table carries eight undeclared
decisions — statistical unit, grouping, weighting, missing-value policy,
uncertainty procedure, resampling unit, repetitions, seed. These tests hold the
contract that makes them declared, digest-covered and reproducible:

* **T1** (a ROME-style reduction): a declared fact-level percentile bootstrap, 2,000
  repetitions, seed 42, is **byte identical** across two independent runs, and
  removing any one of the eight fields fails **at load naming the field**.
* **T2** (reused prompts): `unit: row` and `unit: pair`, where
  counterfactual roles reuse prompts, give **different intervals** and
  **different workflow digests** — a test, not a warning.
* **T3**: `missing: error` refuses a `null`; `exclude` reduces without it and
  **records the count**; a `null` from `matched: false` is distinguishable from
  a non-finite one. The *count* of contributing rows is asserted, not only the
  mean — today's drop is a pandas default and only the count catches it.
* the implied reduction `aggregate` performs, run through the built-in, equals
  `aggregate`'s output on the same table (the honesty test of §2.6);
* the run-time refusals (a column the table lacks, `missing: error`) each ship
  with the legitimate case beside them (fail-closed discipline).

Every test here fails without the change: the module under test does not
exist on the base, `reduction` is an unknown step key (rule 1), and
`MAX_RULE` is 11. The specific behavioural failure each test would show on a
partial implementation is in its docstring.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalab.io.step_io import StepError
from causalab.io.step_record import aggregate, implied_reduction
from causalab.io.tables import read_table
from causalab.workflow.document import (
    MAX_RULE,
    WorkflowError,
    load_workflow,
    parse_workflow,
)
from causalab.workflow.reduction import (
    CURVE_ESTIMATORS,
    ESTIMATORS,
    MIB_GRID,
    REDUCTION_INPUT,
    X_SCALES,
    ReductionSpecError,
    parse_reduction,
    reduce_frame,
)
from causalab.workflow.runner import run_workflow
from causalab.workflow.scripts import reduce
from tests.step_scripts import put_sidecar, put_table, run_step

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# generated tables — arithmetic only, no model, no tokenizer
# --------------------------------------------------------------------------- #


def fact_table(n_facts: int = 12, layers: tuple[int, ...] = (3, 5)) -> list[dict]:
    """ROME's shape: one row per (fact, token position), swept over a layer.
    Facts hold 1, 2 or 3 token rows, so a fact-level and a row-level unit
    genuinely differ; values are dyadic so sums are exact in any order."""
    rows: list[dict] = []
    for layer in layers:
        for fact in range(n_facts):
            for token in range(1 + fact % 3):
                rows.append(
                    {
                        "sites.target.layers": layer,
                        "fact": fact,
                        "token": token,
                        "value": 0.25 * ((fact + token) % 4) + 0.125 * layer,
                    }
                )
    return rows


def reused_prompt_table() -> list[dict]:
    """Counterfactual pairs whose roles **reuse
    prompts**. Prompt 0 is the base of four pairs, prompt 4 the counterfactual
    of one; every pair holds a base row and a counterfactual row. Mean over
    rows, over pairs and over prompts are three different numbers."""
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]
    rows: list[dict] = []
    for pair_id, (base, counterfactual) in enumerate(pairs):
        for role, prompt in (("base", base), ("counterfactual", counterfactual)):
            rows.append(
                {
                    "pair": pair_id,
                    "role": role,
                    "prompt": prompt,
                    "value": 0.125 * prompt + (0.5 if role == "base" else 0.0),
                }
            )
    return rows


ROME_REDUCTION: dict[str, Any] = {
    "estimator": {"kind": "mean"},
    "unit": {"kind": "example", "columns": ["fact"]},
    "group_by": ["sites.target.layers"],
    "weight": None,
    "missing": "exclude",
    "uncertainty": {
        "kind": "percentile_bootstrap",
        "resample_unit": {"kind": "example", "columns": ["fact"]},
        "repetitions": 2000,
        "seed": 42,
    },
}

#: The eight dimensions, as the dotted fields an author would remove.
EIGHT_DIMENSIONS = (
    "unit",
    "group_by",
    "weight",
    "missing",
    "uncertainty.kind",
    "uncertainty.resample_unit",
    "uncertainty.repetitions",
    "uncertainty.seed",
)


def _without(block: dict[str, Any], dotted: str) -> dict[str, Any]:
    out = copy.deepcopy(block)
    *parents, leaf = dotted.split(".")
    node = out
    for key in parents:
        node = node[key]
    del node[leaf]
    return out


def reduce_workflow(table: Path, reduction: dict[str, Any] | None) -> dict[str, Any]:
    """A one-step workflow over the built-in, reading a table by absolute
    path — so it loads, digests and runs with no protocol step and no engine."""
    step: dict[str, Any] = {
        "type": "script",
        "script": {"module": "causalab.workflow.scripts.reduce"},
        "inputs": {"table": {"path": str(table)}},
        "outputs": {
            "table": {
                "file": "reduced.json",
                "columns": {"value": "float64", "n": "int64"},
            }
        },
    }
    if reduction is not None:
        step["reduction"] = reduction
    return {"version": "1", "output_dir": "run", "steps": {"facts": step}}


def _run(raw: dict[str, Any], env, out_root: Path) -> Path:
    loaded = load_workflow(raw, env, workflow_dir=out_root)
    result = run_workflow(loaded, env, out_root, engine=None)
    return result.run_root / "facts" / "reduced.json"


# --------------------------------------------------------------------------- #
# T1 — a ROME-style fact-level bootstrap
# --------------------------------------------------------------------------- #


def test_t1_two_independent_runs_are_byte_identical(env, tmp_path):
    """Fails without the change: the step does not load (rule 1, unknown key
    `reduction`). Fails under the mutation "seed the resampler from `random`
    rather than the declaration": two runs disagree in `lower`/`upper`."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    first = _run(reduce_workflow(table, ROME_REDUCTION), env, tmp_path / "a")
    second = _run(reduce_workflow(table, ROME_REDUCTION), env, tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()
    rows = read_table(first)
    assert [row["sites.target.layers"] for row in rows] == [3, 5]
    for row in rows:
        assert row["n"] == 12  # facts, not the 24 rows
        assert row["n_rows"] == 24
        assert row["lower"] < row["value"] < row["upper"]


def test_t1_the_declaration_is_in_the_run_record(env, tmp_path):
    """A review reads the declaration from the tree: `_step.json` and
    `workflow.json` both carry it, in canonical form."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    reduced = _run(reduce_workflow(table, ROME_REDUCTION), env, tmp_path / "a")
    record = json.loads((reduced.parent / "_step.json").read_text())
    manifest = json.loads((reduced.parent.parent / "workflow.json").read_text())
    want = parse_reduction(ROME_REDUCTION).canonical()
    assert record["reduction"] == want
    assert manifest["steps"]["facts"]["reduction"] == want


@pytest.mark.parametrize("dimension", EIGHT_DIMENSIONS)
def test_t1_removing_one_dimension_fails_at_load_naming_it(env, tmp_path, dimension):
    """Fails without the change: `reduction` is refused as an unknown key
    under rule 1 without naming any dimension. Fails on a parser that fills a
    default for a missing field: the document would load."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    raw = reduce_workflow(table, _without(ROME_REDUCTION, dimension))
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == 12
    assert dimension in str(err.value), str(err.value)


def test_t1_the_estimator_is_required_too(env, tmp_path):
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    raw = reduce_workflow(table, _without(ROME_REDUCTION, "estimator"))
    with pytest.raises(WorkflowError) as err:
        load_workflow(raw, env, workflow_dir=tmp_path)
    assert err.value.rule == 12 and "estimator" in str(err.value)


def test_t1_the_seed_governs_the_interval():
    """The determinism is the *declaration's*: a different seed is a
    different (equally valid) interval, and the same seed the same one."""
    df = pd.DataFrame(fact_table())
    a = reduce_frame(df, parse_reduction(ROME_REDUCTION), "value", what="t")
    b = reduce_frame(df, parse_reduction(ROME_REDUCTION), "value", what="t")
    other = copy.deepcopy(ROME_REDUCTION)
    other["uncertainty"]["seed"] = 7
    c = reduce_frame(df, parse_reduction(other), "value", what="t")
    assert a == b
    assert (a[0]["lower"], a[0]["upper"]) != (c[0]["lower"], c[0]["upper"])
    assert a[0]["value"] == c[0]["value"]  # the estimate does not depend on it


# --------------------------------------------------------------------------- #
# T2 — reused prompts: row, pair and prompt are not interchangeable
# --------------------------------------------------------------------------- #


def _pattern8(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimator": {"kind": "mean"},
        "unit": unit,
        "group_by": [],
        "weight": None,
        "missing": "exclude",
        "uncertainty": {
            "kind": "percentile_bootstrap",
            "resample_unit": unit,
            "repetitions": 400,
            "seed": 42,
        },
    }


def test_t2_row_and_pair_give_different_intervals_and_digests(env, tmp_path):
    """Fails without the change (no `reduction` loads). Fails under the
    mutation "drop `unit` from the canonical entry": the two digests collide."""
    table = put_table(tmp_path / "pairs" / "scores.json", reused_prompt_table())
    by_row = _pattern8({"kind": "row"})
    by_pair = _pattern8({"kind": "pair", "columns": ["pair"]})
    row_loaded = load_workflow(
        reduce_workflow(table, by_row), env, workflow_dir=tmp_path
    )
    pair_loaded = load_workflow(
        reduce_workflow(table, by_pair), env, workflow_dir=tmp_path
    )
    assert row_loaded.digest != pair_loaded.digest
    assert row_loaded.step_digests["facts"] != pair_loaded.step_digests["facts"]

    df = pd.DataFrame(reused_prompt_table())
    row_out = reduce_frame(df, parse_reduction(by_row), "value", what="t")[0]
    pair_out = reduce_frame(df, parse_reduction(by_pair), "value", what="t")[0]
    assert (row_out["lower"], row_out["upper"]) != (
        pair_out["lower"],
        pair_out["upper"],
    )
    assert row_out["n"] == 12 and pair_out["n"] == 6


def test_t2_prompt_as_the_unit_changes_the_estimate_itself():
    """Where roles reuse prompts, the prompt-level mean weighs prompt 0 once
    rather than four times: not just a different interval, a different
    number."""
    df = pd.DataFrame(reused_prompt_table())
    by_row = reduce_frame(
        df, parse_reduction(_pattern8({"kind": "row"})), "value", what="t"
    )
    by_prompt = reduce_frame(
        df,
        parse_reduction(_pattern8({"kind": "prompt", "columns": ["prompt"]})),
        "value",
        what="t",
    )
    assert by_row[0]["value"] != by_prompt[0]["value"]
    assert by_prompt[0]["n"] == 5  # five distinct prompts


def test_t2_every_dimension_moves_the_step_digest(env, tmp_path):
    """All eight sit in the canonical entry, so each one is a different
    computation to `--resume` and to a reviewer."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    base = load_workflow(
        reduce_workflow(table, ROME_REDUCTION), env, workflow_dir=tmp_path
    ).digest
    variants: dict[str, dict[str, Any]] = {}
    for name, edit in {
        "unit": lambda b: b["unit"].update(columns=["token"]),
        "group_by": lambda b: b.update(group_by=[]),
        "weight": lambda b: (
            b.update(weight="token"),
            b["estimator"].update(kind="weighted_mean"),
        ),
        "missing": lambda b: b.update(missing="zero"),
        "uncertainty.kind": lambda b: b.update(
            uncertainty={
                "kind": "normal_approx",
                "resample_unit": b["uncertainty"]["resample_unit"],
            }
        ),
        "uncertainty.resample_unit": lambda b: b["uncertainty"]["resample_unit"].update(
            columns=["token"]
        ),
        "uncertainty.repetitions": lambda b: b["uncertainty"].update(repetitions=1999),
        "uncertainty.seed": lambda b: b["uncertainty"].update(seed=43),
    }.items():
        block = copy.deepcopy(ROME_REDUCTION)
        edit(block)
        variants[name] = block
    digests = {
        name: load_workflow(
            reduce_workflow(table, block), env, workflow_dir=tmp_path
        ).digest
        for name, block in variants.items()
    }
    assert all(d != base for d in digests.values()), digests
    assert len(set(digests.values())) == len(digests)


# --------------------------------------------------------------------------- #
# T3 — the missing-value policy, by count
# --------------------------------------------------------------------------- #

WITH_NULLS = [
    {"example_id": "0", "value": 1.0, "matched": True},
    {"example_id": "1", "value": None, "matched": False},  # the model never said it
    {"example_id": "2", "value": None, "matched": True},  # it computed nothing finite
    {"example_id": "3", "value": 3.0, "matched": True},
]


def _policy(missing: str) -> dict[str, Any]:
    return {**implied_reduction(()), "missing": missing}


def test_t3_error_refuses_a_table_with_a_null():
    with pytest.raises(StepError, match="null"):
        reduce_frame(
            pd.DataFrame(WITH_NULLS),
            parse_reduction(_policy("error")),
            "value",
            what="t",
        )


def test_t3_error_has_a_legitimate_twin():
    """The refusal must not fire on a table with no null."""
    clean = [row for row in WITH_NULLS if row["value"] is not None]
    out = reduce_frame(
        pd.DataFrame(clean), parse_reduction(_policy("error")), "value", what="t"
    )
    assert out == [
        {
            "value": 2.0,
            "n": 2,
            "n_rows": 2,
            "n_missing": 0,
            "n_unmatched": 0,
            "n_excluded": 0,
            # the record's identity: an unlabelled table has an
            # unknown unit; the estimator names itself
            "unit": None,
            "estimand_version": "mean/v1",
        }
    ]


def test_t3_exclude_reduces_without_the_null_and_records_the_count():
    """Assert the **count**: the mean alone (2.0) is what pandas' silent
    `skipna` already gave, and only `n`/`n_excluded` show the two rows went."""
    out = reduce_frame(
        pd.DataFrame(WITH_NULLS), parse_reduction(_policy("exclude")), "value", what="t"
    )
    assert out == [
        {
            "value": 2.0,
            "n": 2,
            "n_rows": 2,
            "n_missing": 2,
            "n_unmatched": 1,
            "n_excluded": 2,
            "unit": None,
            "estimand_version": "mean/v1",
        }
    ]


def test_t3_zero_and_exclude_are_different_numbers():
    """The mutation "make both spellings produce the same number" fails here."""
    exclude = reduce_frame(
        pd.DataFrame(WITH_NULLS), parse_reduction(_policy("exclude")), "value", what="t"
    )[0]
    zero = reduce_frame(
        pd.DataFrame(WITH_NULLS), parse_reduction(_policy("zero")), "value", what="t"
    )[0]
    assert exclude["value"] == 2.0 and zero["value"] == 1.0
    assert exclude["n"] == 2 and zero["n"] == 4
    assert zero["n_excluded"] == 0 and zero["n_missing"] == 2


def test_t3_a_matched_false_null_is_distinguishable_from_a_non_finite_one():
    """Two tables, one null each, same mean, same `n_excluded` — and a
    different `n_unmatched`, which is the only thing that tells "never said
    it" from "computed NaN" once both are written as `null`."""
    unmatched = [WITH_NULLS[0], WITH_NULLS[1], WITH_NULLS[3]]
    nonfinite = [WITH_NULLS[0], WITH_NULLS[2], WITH_NULLS[3]]
    spec = parse_reduction(_policy("exclude"))
    a = reduce_frame(pd.DataFrame(unmatched), spec, "value", what="t")[0]
    b = reduce_frame(pd.DataFrame(nonfinite), spec, "value", what="t")[0]
    assert a["value"] == b["value"] and a["n_excluded"] == b["n_excluded"] == 1
    assert a["n_unmatched"] == 1 and b["n_unmatched"] == 0


def test_t3_a_table_with_no_matched_column_counts_no_unmatched():
    """A non-windowed metric never writes `matched` (`_row` emits it only when
    not None); its nulls are all non-finite values."""
    rows = [{"example_id": "0", "value": 1.0}, {"example_id": "1", "value": None}]
    out = reduce_frame(
        pd.DataFrame(rows), parse_reduction(_policy("exclude")), "value", what="t"
    )[0]
    assert out["n_missing"] == 1 and out["n_unmatched"] == 0


def test_t3_a_null_weight_is_a_missing_row():
    rows = [
        {"example_id": "0", "value": 1.0, "w": 1.0},
        {"example_id": "1", "value": 3.0, "w": None},
    ]
    block = {
        **_policy("exclude"),
        "estimator": {"kind": "weighted_mean"},
        "weight": "w",
    }
    out = reduce_frame(pd.DataFrame(rows), parse_reduction(block), "value", what="t")[0]
    assert out["value"] == 1.0 and out["n"] == 1 and out["n_excluded"] == 1


# --------------------------------------------------------------------------- #
# the honesty test — `aggregate` is the implied reduction
# --------------------------------------------------------------------------- #

SWEPT_DYADIC = [
    {
        "featurizers.rot.k": k,
        "train.seed": seed,
        "example_id": str(ex),
        "value": 0.125 * ((k + 3 * seed + 7 * ex) % 9),
    }
    for k in (2, 8, 16)
    for seed in (0, 1)
    for ex in range(5)
]


def test_the_implied_reduction_equals_aggregate_grouped(tmp_path):
    """Case 1 of `aggregate`: group by the sidecar's axes, mean over the rest.
    Dyadic values, so every summation order gives the same bits. Fails without
    the change (`implied_reduction` does not exist); fails on a vocabulary that
    cannot spell `aggregate` — e.g. a unit that collapsed rows `aggregate`
    does not collapse."""
    fit = tmp_path / "fit"
    table = put_table(fit / "iia.json", SWEPT_DYADIC)
    put_sidecar(fit, ["featurizers.rot.k", "train.seed"])
    df = pd.DataFrame(SWEPT_DYADIC)
    grouped, axes = aggregate(df, table, "value")
    out = reduce_frame(df, parse_reduction(implied_reduction(axes)), "value", what="t")
    assert [(r["featurizers.rot.k"], r["train.seed"]) for r in out] == list(
        zip(grouped["featurizers.rot.k"], grouped["train.seed"])
    )
    assert [r["value"] for r in out] == list(grouped["value"])
    assert all(r["n"] == 5 for r in out)


def test_the_implied_reduction_equals_aggregate_ungrouped(tmp_path):
    """Case 2: no axes but an `example` column — one row, the mean over the
    whole table. Exact: both sides are numpy's pairwise sum over the same
    values."""
    apply_dir = tmp_path / "apply"
    rows = [
        {"example_id": str(i), "value": float(v)}
        for i, v in enumerate(np.random.default_rng(1).random(41))
    ]
    table = put_table(apply_dir / "iia.json", rows)
    put_sidecar(apply_dir, [])
    df = pd.DataFrame(rows)
    single, axes = aggregate(df, table, "value")
    out = reduce_frame(df, parse_reduction(implied_reduction(axes)), "value", what="t")
    assert axes == () and len(out) == 1
    assert out[0]["value"] == single["value"].iloc[0]
    assert out[0]["n"] == 41


def test_the_implied_reduction_matches_aggregate_on_random_data_to_summation_order(
    tmp_path,
):
    """Grouped, random values with nulls: pandas' cython group-mean sums with
    Kahan compensation and numpy pairwise, so the two can differ in the last
    bit. Same estimand, same rows, same policy — asserted to 1e-12 relative,
    and the *counts* exactly. `aggregate` itself is untouched, so an unauthored
    step's bytes do not move (spec §2.6, §7)."""
    rng = np.random.default_rng(0)
    rows = [
        {
            "featurizers.rot.k": k,
            "train.seed": seed,
            "example_id": str(ex),
            "value": float(rng.random()) if rng.random() > 0.1 else None,
        }
        for k in (2, 8, 16, 32)
        for seed in (0, 1, 2)
        for ex in range(37)
    ]
    fit = tmp_path / "fit"
    table = put_table(fit / "iia.json", rows)
    put_sidecar(fit, ["featurizers.rot.k", "train.seed"])
    df = pd.DataFrame(rows)
    grouped, axes = aggregate(df, table, "value")
    out = reduce_frame(df, parse_reduction(implied_reduction(axes)), "value", what="t")
    assert len(out) == len(grouped) == 12
    for want, got in zip(grouped["value"], out):
        assert math.isclose(want, got["value"], rel_tol=1e-12)
    null_counts = (
        df["value"].isna().groupby([df["featurizers.rot.k"], df["train.seed"]]).sum()
    )
    assert [r["n_excluded"] for r in out] == list(null_counts)
    assert all(r["n"] + r["n_excluded"] == 37 for r in out)


def test_the_implied_unit_is_row_not_example():
    """The finding §2.6 records: `aggregate` means over **rows**. On a table
    with one row per example the two units agree; on a windowed one — several
    positions per example — they do not, and `aggregate` follows the rows."""
    windowed = [
        {"example_id": "0", "step": 0, "value": 0.0},
        {"example_id": "0", "step": 1, "value": 0.0},
        {"example_id": "0", "step": 2, "value": 0.0},
        {"example_id": "1", "step": 0, "value": 1.0},
    ]
    df = pd.DataFrame(windowed)
    by_row = reduce_frame(
        df, parse_reduction(implied_reduction(())), "value", what="t"
    )[0]
    by_example = reduce_frame(
        df,
        parse_reduction(
            {
                **implied_reduction(()),
                "unit": {"kind": "example", "columns": ["example_id"]},
            }
        ),
        "value",
        what="t",
    )[0]
    assert by_row["value"] == df["value"].mean() == 0.25
    assert by_example["value"] == 0.5
    assert by_row["n"] == 4 and by_example["n"] == 2


# --------------------------------------------------------------------------- #
# the built-in script, called as the runner calls it
# --------------------------------------------------------------------------- #


def test_the_script_reduces_and_writes_one_row_per_group(tmp_path):
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    out = tmp_path / "reduced.json"
    run_step(reduce, {"table": table, REDUCTION_INPUT: ROME_REDUCTION}, {"table": out})
    rows = read_table(out)
    assert len(rows) == 2 and set(rows[0]) == {
        "sites.target.layers",
        "value",
        "n",
        "n_rows",
        "n_missing",
        "n_unmatched",
        "n_excluded",
        "unit",
        "estimand_version",
        "lower",
        "upper",
    }


def test_the_script_refuses_a_column_the_table_lacks_naming_it(tmp_path):
    """Column existence is data (§2.6): refused at run time, naming the column
    **and** the dimension that named it."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    block = copy.deepcopy(ROME_REDUCTION)
    block["group_by"] = ["sites.ghost.layers"]
    with pytest.raises(StepError) as err:
        run_step(
            reduce,
            {"table": table, REDUCTION_INPUT: block},
            {"table": tmp_path / "r.json"},
        )
    assert "sites.ghost.layers" in str(err.value) and "group_by" in str(err.value)


def test_the_column_refusal_has_a_legitimate_twin(tmp_path):
    """The same declaration over a table that has the column runs."""
    rows = [{**row, "sites.ghost.layers": 1} for row in fact_table()]
    table = put_table(tmp_path / "trace" / "aie.json", rows)
    block = copy.deepcopy(ROME_REDUCTION)
    block["group_by"] = ["sites.ghost.layers"]
    block["uncertainty"]["repetitions"] = 20
    run_step(
        reduce, {"table": table, REDUCTION_INPUT: block}, {"table": tmp_path / "r.json"}
    )
    assert read_table(tmp_path / "r.json")[0]["sites.ghost.layers"] == 1


@pytest.mark.parametrize("dimension", ("unit", "uncertainty.resample_unit", "weight"))
def test_every_binding_dimension_is_checked(tmp_path, dimension):
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    block = copy.deepcopy(ROME_REDUCTION)
    if dimension == "weight":
        block["estimator"] = {"kind": "weighted_mean"}
        block["weight"] = "ghost"
    else:
        node = block
        *parents, leaf = dimension.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = {"kind": "example", "columns": ["ghost"]}
    with pytest.raises(StepError, match="ghost"):
        run_step(
            reduce,
            {"table": table, REDUCTION_INPUT: block},
            {"table": tmp_path / "r.json"},
        )


def test_the_script_refuses_an_undeclared_reduction(tmp_path):
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    with pytest.raises(StepError, match="reduction"):
        run_step(reduce, {"table": table}, {"table": tmp_path / "r.json"})


def test_the_script_revalidates_the_block(tmp_path):
    """A direct call — no loader in front of it — is refused exactly as a
    document would be (S-3: the hatch declares the same dimensions)."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    with pytest.raises(StepError, match="uncertainty.seed"):
        run_step(
            reduce,
            {
                "table": table,
                REDUCTION_INPUT: _without(ROME_REDUCTION, "uncertainty.seed"),
            },
            {"table": tmp_path / "r.json"},
        )


# --------------------------------------------------------------------------- #
# rule 12 — the vocabulary is closed, and the refusal names the field
# --------------------------------------------------------------------------- #


def _step(
    reduction: dict[str, Any], inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "r": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.reduce"},
                "inputs": inputs
                if inputs is not None
                else {"table": {"path": "/x/t.json"}},
                "outputs": {"table": "r.json"},
                "reduction": reduction,
            }
        },
    }


def _refused(
    reduction: dict[str, Any], field: str, inputs: dict[str, Any] | None = None
) -> None:
    with pytest.raises(WorkflowError) as err:
        parse_workflow(_step(reduction, inputs))
    assert err.value.rule == 12, str(err.value)
    assert field in str(err.value), str(err.value)


def test_max_rule_is_at_least_12():
    """Rule 12 is this contract's; later rules (13, estimand identity) are
    their own PR's and censused in `test_reduction_census.py`."""
    assert MAX_RULE >= 12


def test_rule_12_off_vocabulary_unit_suggests():
    block = copy.deepcopy(ROME_REDUCTION)
    block["unit"]["kind"] = "exmaple"
    with pytest.raises(WorkflowError) as err:
        parse_workflow(_step(block))
    assert (
        err.value.rule == 12
        and "unit.kind" in str(err.value)
        and "example" in str(err.value)
    )


@pytest.mark.parametrize(
    "edit, field",
    [
        (lambda b: b.update(missing="skip"), "missing"),
        (lambda b: b["uncertainty"].update(kind="bootstrap"), "uncertainty.kind"),
        (lambda b: b["estimator"].update(kind="average"), "estimator.kind"),
        (lambda b: b["uncertainty"].update(repetitions=0), "uncertainty.repetitions"),
        (lambda b: b["uncertainty"].update(repetitions=2.5), "uncertainty.repetitions"),
        (lambda b: b["uncertainty"].update(seed=-1), "uncertainty.seed"),
        (lambda b: b["uncertainty"].update(seed="42"), "uncertainty.seed"),
        (lambda b: b.update(weight="token"), "weight"),  # mean takes no weight
        (lambda b: b["estimator"].update(kind="weighted_mean"), "weight"),  # needs one
        (lambda b: b["unit"].update(kind="row"), "unit.columns"),  # row names no column
        (lambda b: b["unit"].update(columns=[]), "unit.columns"),  # example needs one
        (lambda b: b["estimator"].update(kind="quantile"), "estimator.q"),
        (lambda b: b["estimator"].update(kind="quantile", q=1.0), "estimator.q"),
        (lambda b: b.update(group_by="sites.target.layers"), "group_by"),
        (lambda b: b.update(group_by=["a", "a"]), "group_by"),
        (lambda b: b.update(minimum_count=5), "minimum_count"),  # not a ninth dimension
    ],
)
def test_rule_12_refuses_naming_the_field(edit, field):
    block = copy.deepcopy(ROME_REDUCTION)
    edit(block)
    _refused(block, field)


def test_rule_12_none_takes_no_resampling_fields():
    """A field that governs nothing may not be declared."""
    block = copy.deepcopy(ROME_REDUCTION)
    block["uncertainty"]["kind"] = "none"
    _refused(block, "uncertainty.resample_unit")
    block["uncertainty"] = {"kind": "none"}
    parse_workflow(_step(block))  # the legitimate form loads


def test_rule_12_normal_approx_is_for_a_mean():
    block = copy.deepcopy(ROME_REDUCTION)
    block["estimator"] = {"kind": "median"}
    block["uncertainty"] = {
        "kind": "normal_approx",
        "resample_unit": {"kind": "example", "columns": ["fact"]},
    }
    _refused(block, "uncertainty.kind")
    block["estimator"] = {"kind": "mean"}
    parse_workflow(_step(block))


def test_rule_12_normal_approx_takes_no_repetitions():
    block = copy.deepcopy(ROME_REDUCTION)
    block["uncertainty"]["kind"] = "normal_approx"
    _refused(block, "uncertainty.repetitions")


def test_rule_12_an_input_named_reduction_collides():
    inputs = {"table": {"path": "/x/t.json"}, REDUCTION_INPUT: {"anything": 1}}
    _refused(ROME_REDUCTION, REDUCTION_INPUT, inputs)


def test_rule_12_has_a_legitimate_twin_for_every_estimator():
    """Each estimator, correctly declared, parses — the refusals above are
    about the wrong shape, never about the vocabulary member."""
    for kind in ("mean", "sum", "count", "median"):
        block = copy.deepcopy(ROME_REDUCTION)
        block["estimator"] = {"kind": kind}
        parse_workflow(_step(block))
    block = copy.deepcopy(ROME_REDUCTION)
    block["estimator"] = {"kind": "quantile", "q": 0.9}
    parse_workflow(_step(block))
    block = copy.deepcopy(ROME_REDUCTION)
    block["estimator"] = {"kind": "weighted_mean"}
    block["weight"] = "token"
    parse_workflow(_step(block))
    block = copy.deepcopy(ROME_REDUCTION)
    block["estimator"] = {"kind": "auc", "x": "axes.cut", "x_scale": "log"}
    parse_workflow(_step(block))
    block = copy.deepcopy(ROME_REDUCTION)
    block["estimator"] = {
        "kind": "cpr",
        "x": "axes.cut",
        "x_scale": "linear",
        "normalize": 156,
    }
    parse_workflow(_step(block))
    assert set(CURVE_ESTIMATORS) < set(ESTIMATORS)


def test_protocol_steps_take_no_reduction():
    """Their in-forward reduction is `save.reduce` (§2.6): `reduction` on a
    protocol step is an unknown key, rule 1."""
    raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "a": {
                "type": "intervention_protocol",
                "document": "x.json",
                "reduction": ROME_REDUCTION,
            }
        },
    }
    with pytest.raises(WorkflowError) as err:
        parse_workflow(raw)
    assert err.value.rule == 1 and "reduction" in str(err.value)


def test_parse_reduction_names_the_field_directly():
    with pytest.raises(ReductionSpecError) as err:
        parse_reduction(_without(ROME_REDUCTION, "uncertainty.seed"))
    assert err.value.field == "uncertainty.seed"


# --------------------------------------------------------------------------- #
# canonical form — present when authored, absent otherwise
# --------------------------------------------------------------------------- #


def test_the_canonical_entry_carries_reduction_only_when_authored(env, tmp_path):
    """The `is_deterministic` trap, avoided: an unauthored step's canonical
    entry has no `reduction` key at all — not `null`, not a default."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    with_block = load_workflow(
        reduce_workflow(table, ROME_REDUCTION), env, workflow_dir=tmp_path
    )
    without = load_workflow(reduce_workflow(table, None), env, workflow_dir=tmp_path)
    assert (
        with_block.canonical["steps"]["facts"]["reduction"]
        == parse_reduction(ROME_REDUCTION).canonical()
    )
    assert "reduction" not in without.canonical["steps"]["facts"]
    assert with_block.digest != without.digest


def test_the_canonical_form_is_the_parsed_form(env, tmp_path):
    """`2000` and `2000.0`, key order, and the `row` unit's omitted `columns`
    all canonicalize to one entry — two identical declarations cannot digest
    differently."""
    table = put_table(tmp_path / "trace" / "aie.json", fact_table())
    a = copy.deepcopy(ROME_REDUCTION)
    b = {key: a[key] for key in reversed(list(a))}
    b["uncertainty"] = dict(b["uncertainty"], repetitions=2000)
    assert (
        load_workflow(reduce_workflow(table, a), env, workflow_dir=tmp_path).digest
        == load_workflow(reduce_workflow(table, b), env, workflow_dir=tmp_path).digest
    )
    row_a = parse_reduction({**a, "unit": {"kind": "row"}}).canonical()
    row_b = parse_reduction({**a, "unit": {"kind": "row", "columns": []}}).canonical()
    assert row_a == row_b


# --------------------------------------------------------------------------- #
# the estimators and procedures, against hand-computed values
# --------------------------------------------------------------------------- #

UNITS = [
    {"u": 0, "value": 1.0, "w": 1.0},
    {"u": 0, "value": 3.0, "w": 1.0},
    {"u": 1, "value": 5.0, "w": 2.0},
    {"u": 2, "value": 7.0, "w": 0.0},
]


def _est(kind: str, unit: str = "row", **extra: Any) -> dict[str, Any]:
    estimator: dict[str, Any] = {"kind": kind}
    if "q" in extra:
        estimator["q"] = extra.pop("q")
    return {
        "estimator": estimator,
        "unit": {"kind": "row"} if unit == "row" else {"kind": unit, "columns": ["u"]},
        "group_by": [],
        "weight": extra.pop("weight", None),
        "missing": "exclude",
        "uncertainty": {"kind": "none"},
    }


def _value(block: dict[str, Any]) -> Any:
    return reduce_frame(pd.DataFrame(UNITS), parse_reduction(block), "value", what="t")[
        0
    ]["value"]


def test_estimators_over_rows():
    assert _value(_est("mean")) == 4.0
    assert _value(_est("sum")) == 16.0
    assert _value(_est("count")) == 4
    assert (
        _value(_est("median")) == 3.0
    )  # lower of the two middle values, no interpolation
    assert _value(_est("quantile", q=0.5)) == 4.0  # interpolated, so not the median
    assert _value(_est("weighted_mean", weight="w")) == (1 + 3 + 10) / 4.0


def test_estimators_over_units_collapse_first():
    """`unit` collapses rows to observations: the mean of unit means, the sum
    of unit sums, the count of units — never the row-level figures."""
    assert _value(_est("mean", unit="example")) == (2.0 + 5.0 + 7.0) / 3
    assert _value(_est("sum", unit="example")) == 16.0
    assert _value(_est("count", unit="example")) == 3
    assert _value(_est("median", unit="example")) == 5.0
    # unit 2 has zero total weight: it contributes nothing, and does not poison
    assert (
        _value(_est("weighted_mean", unit="example", weight="w"))
        == (2.0 * 2 + 5.0 * 2) / 4.0
    )


def test_normal_approx_is_the_sample_standard_error_across_resample_units():
    rows = [{"u": u, "value": float(v)} for u, v in enumerate((1.0, 2.0, 3.0, 4.0))]
    block = _est("mean")
    block["uncertainty"] = {
        "kind": "normal_approx",
        "resample_unit": {"kind": "example", "columns": ["u"]},
    }
    out = reduce_frame(pd.DataFrame(rows), parse_reduction(block), "value", what="t")[0]
    se = np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2.0
    assert out["lower"] == pytest.approx(2.5 - 1.959963984540054 * se)
    assert out["upper"] == pytest.approx(2.5 + 1.959963984540054 * se)


def test_normal_approx_with_one_resample_unit_has_null_bounds(tmp_path):
    rows = [{"u": 0, "value": 1.0}, {"u": 0, "value": 3.0}]
    block = _est("mean")
    block["uncertainty"] = {
        "kind": "normal_approx",
        "resample_unit": {"kind": "example", "columns": ["u"]},
    }
    out = reduce_frame(pd.DataFrame(rows), parse_reduction(block), "value", what="t")[0]
    assert math.isnan(out["lower"]) and math.isnan(out["upper"])
    table = put_table(tmp_path / "t.json", rows)
    run_step(
        reduce, {"table": table, REDUCTION_INPUT: block}, {"table": tmp_path / "r.json"}
    )
    assert read_table(tmp_path / "r.json")[0]["lower"] is None  # written as null


def test_a_group_interval_does_not_depend_on_other_groups():
    """Each group's resampler is seeded from the declaration plus the group's
    own coordinates, so adding a group leaves the others' intervals still."""
    both = pd.DataFrame(fact_table(layers=(3, 5)))
    one = pd.DataFrame(fact_table(layers=(3,)))
    spec = parse_reduction(ROME_REDUCTION)
    assert (
        reduce_frame(both, spec, "value", what="t")[0]
        == reduce_frame(one, spec, "value", what="t")[0]
    )


def test_the_bootstrap_resamples_clusters_not_rows():
    """`unit: row`, `resample_unit: fact`: the estimate is the row mean and
    the interval is a cluster bootstrap over facts. Compared with resampling
    rows, the two intervals differ — same estimate, different uncertainty."""
    df = pd.DataFrame(fact_table(layers=(3,)))
    cluster = copy.deepcopy(ROME_REDUCTION)
    cluster["unit"] = {"kind": "row"}
    rows = copy.deepcopy(cluster)
    rows["uncertainty"]["resample_unit"] = {"kind": "row"}
    a = reduce_frame(df, parse_reduction(cluster), "value", what="t")[0]
    b = reduce_frame(df, parse_reduction(rows), "value", what="t")[0]
    assert a["value"] == b["value"] and a["n"] == b["n"] == 24
    assert (a["lower"], a["upper"]) != (b["lower"], b["upper"])


# --------------------------------------------------------------------------- #
# curve estimators — `auc` and `cpr` (workflow §2.6 "Curve estimators")
# --------------------------------------------------------------------------- #

#: gpt2's MIB node count with the `input` node: 12·12 heads + 12 MLPs + 1.
N = 157


def _curve(kind: str, **fields: Any) -> dict[str, Any]:
    """A curve block over `axes.cut` / `value`, `unit: row`, no interval."""
    estimator: dict[str, Any] = {"kind": kind, "x": "axes.cut", "x_scale": "linear"}
    estimator.update(fields)
    return {
        "estimator": estimator,
        "unit": {"kind": "row"},
        "group_by": [],
        "weight": None,
        "missing": "exclude",
        "uncertainty": {"kind": "none"},
    }


def sweep_table(
    means: dict[int, float], *, examples: int = 3, method: str | None = None
) -> list[dict]:
    """A `top_k` sweep's metric table: one row per (example, cut), values
    spread ±1 around the per-cut mean so the mean is what is asserted."""
    rows = []
    for cut, mean in means.items():
        for i in range(examples):
            row = {
                "example": i,
                "metric": "ld",
                "value": mean + (i - 1),
                "axes.cut": cut,
                "units": N,
            }
            if method is not None:
                row["method"] = method
            rows.append(row)
    return rows


def _mib_cut(p: float, units: int = N) -> int:
    return units - int(p * units)


def _trapezoid(xs, ys) -> float:
    return sum(
        (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)
    )


def _curve_out(block: dict[str, Any], rows: list[dict], value: str = "value") -> Any:
    return reduce_frame(pd.DataFrame(rows), parse_reduction(block), value, what="t")[0]


def test_the_vocabulary_grew_by_exactly_the_two_curve_estimators():
    assert ESTIMATORS[-2:] == ("auc", "cpr") == CURVE_ESTIMATORS
    assert X_SCALES == ("linear", "log")
    assert MIB_GRID == (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def test_cpr_matches_mibs_hand_computed_curve():
    """MIB's `evaluate_area_under_curve`: faith = (m − C)/(B − C) with B at cut
    0 and C at cut N, the grid point p read at cut N − int(p·N), trapezoid over
    the raw p. A metric linear in the kept fraction gives faith(p) = the
    fraction actually kept at that cut — which is *not* p, because int()
    floors: the three smallest p keep 0 of 157 units."""
    base, corrupted = 10.0, 2.0
    means: dict[int, float] = {}
    for p in MIB_GRID:
        cut = _mib_cut(p)
        means[cut] = corrupted + ((N - cut) / N) * (base - corrupted)
    means[0], means[N] = base, corrupted
    out = _curve_out(_curve("cpr", normalize=N), sweep_table(means))
    kept = [(N - _mib_cut(p)) / N for p in MIB_GRID]
    assert kept[:3] == [0.0, 0.0, 0.0]  # the polarity trap, kept as MIB keeps it
    assert out["value"] == pytest.approx(_trapezoid(list(MIB_GRID), kept))
    assert out["n"] == out["n_rows"] == 3 * len(means)
    assert out["unit"] == "dimensionless" and out["estimand_version"] == "cpr/v1"


def test_a_perfect_ranking_scores_the_full_area_and_a_flat_one_zero():
    """Every kept set but the empty one recovers the clean metric: faith = 1
    except at the three grid points that keep nothing → the area is the
    trapezoid of [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]. A ranking that recovers
    nothing until everything is kept scores the [0, …, 0, 1] tail only."""
    perfect = {_mib_cut(p): 10.0 for p in MIB_GRID}
    perfect[0], perfect[N] = 10.0, 2.0
    out = _curve_out(_curve("cpr", normalize=N), sweep_table(perfect))
    assert out["value"] == pytest.approx(_trapezoid(list(MIB_GRID), [0] * 3 + [1] * 7))
    flat = {_mib_cut(p): 2.0 for p in MIB_GRID}
    flat[0] = 10.0
    out = _curve_out(_curve("cpr", normalize=N), sweep_table(flat))
    assert out["value"] == pytest.approx(_trapezoid(list(MIB_GRID), [0] * 9 + [1]))


def test_cpr_reads_n_from_a_column_and_takes_an_authored_grid():
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    rows = sweep_table(means)
    by_number = _curve_out(_curve("cpr", normalize=N), rows)
    by_column = _curve_out(_curve("cpr", normalize="units"), rows)
    assert (
        by_number["value"]
        == by_column["value"]
        == pytest.approx(_trapezoid(list(MIB_GRID), [0.0] * 3 + [0.5] * 6 + [1.0]))
    )
    # an authored grid reads other cuts: p = 0.5 and 1.0 only
    half = _curve_out(_curve("cpr", normalize=N, grid=[0.5, 1.0]), rows)
    assert half["value"] == pytest.approx(0.5 * (0.5 + 1.0) / 2)
    # the log abscissa is log p — the same faithfulnesses, another integral
    logged = _curve_out(_curve("cpr", normalize=N, x_scale="log"), rows)
    assert logged["value"] == pytest.approx(
        _trapezoid([math.log(p) for p in MIB_GRID], [0.0] * 3 + [0.5] * 6 + [1.0])
    )


def test_cpr_refuses_a_missing_anchor_a_missing_cut_and_equal_anchors():
    means = {_mib_cut(p): 5.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    rows = sweep_table(means)
    block = _curve("cpr", normalize=N)
    with pytest.raises(StepError, match="both anchors"):
        _curve_out(block, [r for r in rows if r["axes.cut"] != N])
    with pytest.raises(StepError, match=r"no rows at cuts 142 \(p = 0\.1\) — x must"):
        _curve_out(block, [r for r in rows if r["axes.cut"] != N - 15])
    with pytest.raises(StepError, match="anchors are equal"):
        _curve_out(block, sweep_table({**means, N: 10.0}))
    with pytest.raises(StepError, match="integer cuts"):
        _curve_out(
            block, rows + [{"example": 0, "value": 1.0, "axes.cut": 3.5, "units": N}]
        )
    # a `units` column that disagrees within the group is two curves
    torn = rows + [{"example": 0, "value": 5.0, "axes.cut": 0, "units": N + 1}]
    with pytest.raises(StepError, match="one ceiling per curve"):
        _curve_out(_curve("cpr", normalize="units"), torn)
    # the legitimate twin of each refusal
    assert math.isfinite(_curve_out(block, rows)["value"])
    # under a declared `unit` a cut *no* unit holds is still the table's shape:
    # the anchor refusal speaks, not the panel's (which would advise dropping
    # a unit that cannot make the anchor appear)
    paneled = _curve("cpr", normalize=N)
    paneled["unit"] = {"kind": "example", "columns": ["example"]}
    with pytest.raises(StepError, match="both anchors"):
        _curve_out(paneled, [r for r in rows if r["axes.cut"] != N])
    # ...while a cut one unit lacks is the panel's refusal
    with pytest.raises(StepError, match="balanced panel"):
        _curve_out(
            paneled, [r for r in rows if not (r["axes.cut"] == N and r["example"] == 0)]
        )
    # x may not be the column being reduced (the default y walks past the
    # parse gate)
    with pytest.raises(StepError, match="the column being reduced"):
        _curve_out(block, rows, value="axes.cut")
    # ...and neither may the `normalize` column
    with pytest.raises(StepError, match="estimator.normalize names 'units'"):
        _curve_out(_curve("cpr", normalize="units"), rows, value="units")


def test_cpr_is_not_normalised_by_the_metric_scale():
    """faith is a ratio of differences: scaling every value by 7 and shifting
    by 3 leaves the CPR unchanged — the curve is about the ranking, not the
    metric's unit."""
    means = {_mib_cut(p): 4.0 + (N - _mib_cut(p)) / N for p in MIB_GRID}
    means[0], means[N] = 9.0, 4.0
    a = _curve_out(_curve("cpr", normalize=N), sweep_table(means))
    b = _curve_out(
        _curve("cpr", normalize=N),
        sweep_table({c: 7 * m + 3 for c, m in means.items()}),
    )
    assert a["value"] == pytest.approx(b["value"])


def test_auc_is_the_trapezoid_of_per_x_means_and_normalize_divides_x():
    rows = [
        {"example": e, "value": v + e, "axes.cut": x}
        for x, v in ((0, 0.0), (10, 1.0), (30, 2.0))
        for e in (-1, 0, 1)
    ]
    out = _curve_out(_curve("auc"), rows)
    assert out["value"] == pytest.approx(10 * 0.5 + 20 * 1.5)
    assert out["n"] == 9 and out["unit"] is None and out["estimand_version"] == "auc/v1"
    scaled = _curve_out(_curve("auc", normalize=30), rows)
    assert scaled["value"] == pytest.approx((10 * 0.5 + 20 * 1.5) / 30)
    # y names another column than the step's value column
    for row in rows:
        row["acc"] = 2 * row["value"]
    assert _curve_out(_curve("auc", y="acc"), rows)["value"] == pytest.approx(
        2 * (10 * 0.5 + 20 * 1.5)
    )


def test_auc_under_log_integrates_over_log_x_and_refuses_a_zero_cut():
    rows = [{"value": v, "axes.cut": x} for x, v in ((1, 1.0), (10, 1.0), (100, 1.0))]
    out = _curve_out(_curve("auc", x_scale="log"), rows)
    assert out["value"] == pytest.approx(math.log(100))  # a flat 1 over two decades
    with pytest.raises(StepError, match="no logarithm"):
        _curve_out(_curve("auc", x_scale="log"), rows + [{"value": 1.0, "axes.cut": 0}])


def test_a_curve_observation_is_one_unit_at_one_x():
    """`unit: example` collapses rows sharing (example, x) — not rows sharing
    the example across the curve. An example with three rows at one cut
    weighs as one there and as nothing elsewhere."""
    rows = [
        {"example": 0, "value": 0.0, "axes.cut": 0},
        {"example": 0, "value": 0.0, "axes.cut": 0},
        {"example": 0, "value": 0.0, "axes.cut": 0},
        {"example": 1, "value": 3.0, "axes.cut": 0},
        {"example": 0, "value": 1.0, "axes.cut": 10},
        {"example": 1, "value": 1.0, "axes.cut": 10},
    ]
    by_row = _curve_out(_curve("auc"), rows)
    by_example = _curve("auc")
    by_example["unit"] = {"kind": "example", "columns": ["example"]}
    by_example_out = _curve_out(by_example, rows)
    # row mean at cut 0 is 0.75; the example mean is 1.5
    assert by_row["value"] == pytest.approx(10 * (0.75 + 1.0) / 2)
    assert by_example_out["value"] == pytest.approx(10 * (1.5 + 1.0) / 2)
    assert by_row["n"] == 6 and by_example_out["n"] == 4


def test_a_curve_group_is_one_curve_per_group():
    means_a = {_mib_cut(p): 10.0 for p in MIB_GRID}
    means_a[0], means_a[N] = 10.0, 2.0
    means_b = {_mib_cut(p): 2.0 for p in MIB_GRID}
    means_b[0] = 10.0
    block = _curve("cpr", normalize="units")
    block["group_by"] = ["method"]
    rows = sweep_table(means_a, method="dbm") + sweep_table(means_b, method="random")
    out = reduce_frame(pd.DataFrame(rows), parse_reduction(block), "value", what="t")
    by_method = {row["method"]: row["value"] for row in out}
    assert by_method["dbm"] > by_method["random"]
    assert by_method["dbm"] == pytest.approx(
        _trapezoid(list(MIB_GRID), [0] * 3 + [1] * 7)
    )


def test_missing_applies_to_the_ordinate_and_a_null_abscissa_has_no_zero():
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    rows = sweep_table(means)
    rows.append({"example": 9, "value": None, "axes.cut": 0, "units": N})
    block = _curve("cpr", normalize=N)
    out = _curve_out(block, rows)
    assert out["n_missing"] == out["n_excluded"] == 1
    assert out["value"] == pytest.approx(
        _trapezoid(list(MIB_GRID), [0.0] * 3 + [0.5] * 6 + [1.0])
    )
    strict = {**block, "missing": "error"}
    with pytest.raises(StepError, match="reduction.missing is 'error'"):
        _curve_out(strict, rows)
    # a null x under `zero` is refused: an abscissa cannot be scored 0.0
    rows_x = sweep_table(means) + [
        {"example": 9, "value": 1.0, "axes.cut": None, "units": N}
    ]
    with pytest.raises(StepError, match="null x"):
        _curve_out({**block, "missing": "zero"}, rows_x)
    assert _curve_out({**block, "missing": "exclude"}, rows_x)["n_excluded"] == 1


def test_the_bootstrap_redraws_units_and_recomputes_the_curve():
    """A percentile bootstrap over examples: the interval brackets the point
    estimate, is seeded (two runs identical), and a table whose examples all
    agree has a zero-width interval."""
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    block = _curve("cpr", normalize=N)
    block["unit"] = {"kind": "example", "columns": ["example"]}
    block["uncertainty"] = {
        "kind": "percentile_bootstrap",
        "resample_unit": {"kind": "example", "columns": ["example"]},
        "repetitions": 200,
        "seed": 7,
    }
    rows = sweep_table(means, examples=5)
    a = _curve_out(block, rows)
    b = _curve_out(block, rows)
    assert a == b
    assert a["lower"] <= a["value"] <= a["upper"]
    assert a["n"] == 5 * len(means)
    identical = [dict(r, value=means[r["axes.cut"]]) for r in rows]
    c = _curve_out(block, identical)
    assert c["lower"] == c["upper"] == pytest.approx(c["value"])


@pytest.mark.parametrize(
    "edit, field",
    [
        (lambda e: e.pop("x"), "estimator.x"),
        (lambda e: e.update(x=""), "estimator.x"),
        (lambda e: e.update(y="axes.cut"), "estimator.y"),
        (lambda e: e.pop("x_scale"), "estimator.x_scale"),
        (lambda e: e.update(x_scale="lin"), "estimator.x_scale"),
        (lambda e: e.update(normalize=0), "estimator.normalize"),
        (lambda e: e.update(normalize=1.5), "estimator.normalize"),  # cpr: integer N
        (lambda e: e.update(normalize=True), "estimator.normalize"),
        (lambda e: e.update(normalize="axes.cut"), "estimator.normalize"),
        (lambda e: e.update(grid=[0.5]), "estimator.grid"),
        (lambda e: e.update(grid=[0.5, 0.2]), "estimator.grid"),
        (lambda e: e.update(grid=[0.0, 1.0]), "estimator.grid"),
        (lambda e: e.update(grid=[0.5, 1.5]), "estimator.grid"),
        (lambda e: e.update(q=0.5), "estimator.q"),
    ],
)
def test_rule_12_refuses_a_malformed_curve_estimator_naming_the_field(edit, field):
    block = _curve("cpr", normalize=N)
    edit(block["estimator"])
    _refused(block, field)


def test_rule_12_curve_fields_belong_to_curve_estimators_only():
    """`x` on a mean, `grid` on an `auc`, `normalize` missing on a `cpr`, a
    weight on a curve — each refused naming the field; the twins parse."""
    block = copy.deepcopy(ROME_REDUCTION)
    block["estimator"] = {"kind": "mean", "x": "axes.cut"}
    _refused(block, "estimator.x")
    _refused(_curve("auc", grid=[0.5, 1.0]), "estimator.grid")
    _refused(_curve("cpr"), "estimator.normalize")
    weighted = _curve("cpr", normalize=N)
    weighted["weight"] = "w"
    _refused(weighted, "weight")
    approx = _curve("auc")
    approx["uncertainty"] = {
        "kind": "normal_approx",
        "resample_unit": {"kind": "example", "columns": ["example"]},
    }
    _refused(approx, "uncertainty.kind")
    parse_workflow(_step(_curve("auc")))
    parse_workflow(_step(_curve("auc", normalize=0.5, x_scale="log")))
    parse_workflow(_step(_curve("cpr", normalize=N, grid=list(MIB_GRID))))


def test_the_curve_canonical_form_carries_only_what_was_authored():
    """An unauthored `y`, `normalize` (auc) or `grid` (cpr) is absent — not
    defaulted in — so the digest of a block that says less is the digest of
    what it says; `156` and `156.0` are one N."""
    minimal = parse_reduction(_curve("auc")).canonical()["estimator"]
    assert minimal == {"kind": "auc", "x": "axes.cut", "x_scale": "linear"}
    cpr = parse_reduction(_curve("cpr", normalize=156)).canonical()["estimator"]
    assert cpr == {
        "kind": "cpr",
        "x": "axes.cut",
        "x_scale": "linear",
        "normalize": 156,
    }
    assert (
        parse_reduction(_curve("cpr", normalize=156.0)).canonical()
        == parse_reduction(_curve("cpr", normalize=156)).canonical()
    )
    full = parse_reduction(
        _curve("cpr", normalize="units", y="acc", grid=[0.5, 1.0])
    ).canonical()["estimator"]
    assert (
        full["grid"] == [0.5, 1.0]
        and full["y"] == "acc"
        and full["normalize"] == "units"
    )


def test_the_curve_binding_columns_are_checked_against_the_table():
    rows = [{"value": 1.0, "cut": 0}, {"value": 0.0, "cut": 5}]
    with pytest.raises(StepError, match="estimator.x names column 'axes.cut'"):
        _curve_out(_curve("auc"), rows)
    with pytest.raises(StepError, match="estimator.normalize names column 'units'"):
        _curve_out(_curve("auc", x="cut", normalize="units"), rows)
    with pytest.raises(StepError, match="estimator.y names column 'acc'"):
        _curve_out(_curve("auc", x="cut", y="acc"), rows)
    assert _curve_out(_curve("auc", x="cut"), rows)["value"] == pytest.approx(2.5)


def test_the_built_in_runs_a_cpr_block_end_to_end(env, tmp_path):
    """A one-step workflow over the built-in: loads (rule 12), digests with
    the curve estimator in its canonical entry, runs, and writes MIB's number
    with its identity."""
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    table = put_table(tmp_path / "apply" / "ld.json", sweep_table(means))
    block = _curve("cpr", normalize="units")
    block["unit"] = {"kind": "example", "columns": ["example"]}
    loaded = load_workflow(reduce_workflow(table, block), env, workflow_dir=tmp_path)
    assert loaded.canonical["steps"]["facts"]["reduction"]["estimator"]["kind"] == "cpr"
    out = read_table(_run(reduce_workflow(table, block), env, tmp_path / "out"))
    assert len(out) == 1
    assert out[0]["value"] == pytest.approx(
        _trapezoid(list(MIB_GRID), [0.0] * 3 + [0.5] * 6 + [1.0])
    )
    assert out[0]["estimand_version"] == "cpr/v1" and out[0]["unit"] == "dimensionless"
    assert out[0]["n"] == 3 * len(means)


# --------------------------------------------------------------------------- #
# what the parser guarantees, the run time need not
# assume — and the one hunk on the scalar path, pinned to a number
# --------------------------------------------------------------------------- #


def test_the_scalar_bootstrap_interval_is_pinned():
    """A characterization test for the one hunk that reaches the scalar path
    (the estimator became a callable on the resamplers): ROME's fact-level
    percentile bootstrap over `fact_table()` — 2,000 repetitions, seed 42 —
    gave these bounds before the estimator became a callable and must give
    them after. The relational assertions (two intervals differ, the seed
    governs) would let a drifted draw sequence through; a number does not."""
    df = pd.DataFrame(fact_table())
    out = reduce_frame(df, parse_reduction(ROME_REDUCTION), "value", what="t")
    by_layer = {row["sites.target.layers"]: row for row in out}
    assert by_layer[3]["value"] == pytest.approx(0.75, abs=1e-12)
    assert by_layer[3]["lower"] == pytest.approx(0.6388020833333334, abs=1e-12)
    assert by_layer[3]["upper"] == pytest.approx(0.8646701388888884, abs=1e-12)
    assert by_layer[5]["value"] == pytest.approx(1.0, abs=1e-12)
    assert by_layer[5]["lower"] == pytest.approx(0.888888888888889, abs=1e-12)
    assert by_layer[5]["upper"] == pytest.approx(1.1145833333333333, abs=1e-12)


def test_a_normalize_column_meets_the_parsers_predicate():
    """`auc` over `x ÷ normalize`: an authored `0` or `-4` is refused at load;
    a column holding one is refused at run time by the same predicate — a
    column is not a way around it. Before the change a zero column gave
    infinite abscissae and a NaN area, a negative one a negative area, and no
    refusal fired. An all-null column is a missing ceiling, not "0 distinct
    values"."""
    rows = [{"value": v, "axes.cut": x} for x, v in ((0, 0.0), (10, 1.0), (30, 2.0))]
    for ceiling in (0, -4):
        with pytest.raises(StepError, match="must be positive"):
            _curve_out(
                _curve("auc", normalize="units"), [dict(r, units=ceiling) for r in rows]
            )
    with pytest.raises(StepError, match="holds no number"):
        _curve_out(
            _curve("auc", normalize="units"), [dict(r, units=None) for r in rows]
        )
    # finite first: +inf passed `> 0` and divided every x to 0 — a zero area
    # published under `auc/v1`; under `cpr` `int(inf)` raised inside the check
    with pytest.raises(StepError, match="must be finite"):
        _curve_out(
            _curve("auc", normalize="units"), [dict(r, units=math.inf) for r in rows]
        )
    with pytest.raises(StepError, match="must be finite"):
        _curve_out(
            _curve("cpr", normalize="units"),
            [dict(r, units=math.inf) for r in sweep_table({0: 10.0, N: 2.0})],
        )
    good = _curve_out(
        _curve("auc", normalize="units"), [dict(r, units=30) for r in rows]
    )
    assert good["value"] == pytest.approx((10 * 0.5 + 20 * 1.5) / 30)
    # `cpr` already held its column to the parser's rule; still does
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    with pytest.raises(StepError, match="integer ≥ 2"):
        _curve_out(
            _curve("cpr", normalize="units"),
            [dict(r, units=1.5) for r in sweep_table(means)],
        )


def test_group_by_may_not_name_the_abscissa():
    """Grouped by `x`, every curve is one point and every area 0.0 — a
    declaration that can only publish zeros is refused at load naming
    `group_by`; the twin (grouping by another column) parses."""
    block = _curve("auc")
    block["group_by"] = ["axes.cut"]
    _refused(block, "group_by")
    twin = _curve("auc")
    twin["group_by"] = ["method"]
    parse_workflow(_step(twin))
    # the same error in the two other dimensions that would key on x
    keyed = _curve("auc")
    keyed["unit"] = {"kind": "example", "columns": ["axes.cut"]}
    _refused(keyed, "unit.columns")
    drawn = _curve("auc")
    drawn["uncertainty"] = {
        "kind": "percentile_bootstrap",
        "resample_unit": {"kind": "example", "columns": ["axes.cut"]},
        "repetitions": 20,
        "seed": 0,
    }
    _refused(drawn, "uncertainty.resample_unit.columns")


def test_auc_reads_a_balanced_panel_under_a_unit_too():
    """The panel guarantee is a curve's, not `cpr`'s: an `auc` over units with
    a hole at one x integrates a mean over fewer units there, and a cluster
    bootstrap draw missing the only unit at that x would integrate a grid
    without it — the same silent number `cpr` refuses. Under `unit: row`
    there is no panel, and a hole no unit fills is `auc`'s own shape."""
    rows = [
        {"example": e, "value": float(x + e), "axes.cut": x}
        for e in range(3)
        for x in (0, 10, 30)
    ]
    block = _curve("auc")
    block["unit"] = {"kind": "example", "columns": ["example"]}
    assert math.isfinite(_curve_out(block, rows)["value"])
    holed = [r for r in rows if not (r["example"] == 2 and r["axes.cut"] == 10)]
    with pytest.raises(StepError, match=r"'auc' reads a balanced panel.*x 10 \(1 unit"):
        _curve_out(block, holed)
    # a hole every unit shares is not a hole
    assert math.isfinite(
        _curve_out(block, [r for r in rows if r["axes.cut"] != 10])["value"]
    )
    assert math.isfinite(_curve_out(_curve("auc"), holed)["value"])


def test_a_panel_refusal_lists_five_holes_and_counts_the_rest():
    """The listing is capped as the module's other listings are; an `auc`
    reads every distinct x, so the holes can be many."""
    rows = [
        {"example": e, "value": 1.0 + x, "axes.cut": x}
        for e in range(2)
        for x in range(9)
    ]
    block = _curve("auc")
    block["unit"] = {"kind": "example", "columns": ["example"]}
    holed = [r for r in rows if not (r["example"] == 1 and r["axes.cut"] >= 2)]
    with pytest.raises(
        StepError,
        match=r"x 2 \(1 unit\(s\)\), 3 \(1 unit\(s\)\), .*6 \(1 unit\(s\)\) and 2 more",
    ):
        _curve_out(block, holed)


def test_a_row_resample_unit_is_refused_at_load_and_a_splitting_one_on_the_table():
    """Under a declared `unit` a curve's draw must be whole curves: each unit
    lies in one resample cluster. `row` is inside every unit and is refused
    at load, for both estimators; whether a *column* resample unit splits a
    unit is the table's to say — a coarser unit declared by its own column
    (examples nested in templates) is the two-level cluster bootstrap and
    passes, a finer one (a method inside an example) is refused at run time
    naming how many units it splits. Column containment would have refused
    the nested case as "finer". Rows are no panel; a non-curve estimator is
    not held to it."""

    def block(kind: str, resample_unit: dict[str, Any]) -> dict[str, Any]:
        out = _curve(kind, normalize=N) if kind == "cpr" else _curve(kind)
        out["unit"] = {"kind": "example", "columns": ["example"]}
        out["uncertainty"] = {
            "kind": "percentile_bootstrap",
            "resample_unit": resample_unit,
            "repetitions": 5,
            "seed": 0,
        }
        return out

    for kind in ("auc", "cpr"):
        with pytest.raises(ReductionSpecError, match="is 'row' under unit") as err:
            parse_reduction(block(kind, {"kind": "row"}))
        assert err.value.field == "uncertainty.resample_unit"
        parse_reduction(block(kind, {"kind": "example", "columns": ["example"]}))
    rows_block = _curve("auc")
    rows_block["uncertainty"] = {
        "kind": "percentile_bootstrap",
        "resample_unit": {"kind": "row"},
        "repetitions": 5,
        "seed": 0,
    }
    parse_reduction(rows_block)  # rows are no panel
    # the table decides: examples nested in templates (coarser) pass, a
    # method inside an example (finer) splits every unit and is refused
    rows = [
        {
            "example": e,
            "template": e // 2,
            "method": "A" if x < 20 else "B",
            "value": float(x + e),
            "axes.cut": x,
        }
        for e in range(4)
        for x in (0, 10, 30)
    ]
    nested = block("auc", {"kind": "source_family", "columns": ["template"]})
    out = _curve_out(nested, rows)
    # a proper interval: with seed 0 the five draws over the two templates
    # mix them (a draw of one template twice gives that template's curve),
    # so the assertion is about the estimator over a mixed draw, not a tautology
    assert math.isfinite(out["value"]) and out["lower"] <= out["value"] <= out["upper"]
    split = block("auc", {"kind": "pair", "columns": ["example", "method"]})
    with pytest.raises(StepError, match=r"splits 4 of 4 unit\(s\) across resample"):
        _curve_out(split, rows)
    # a column disjoint from the unit's that the data does not nest: the
    # table's refusal, at run time (column containment refused it at load)
    disjoint = block("auc", {"kind": "pair", "columns": ["method"]})
    with pytest.raises(StepError, match=r"splits 4 of 4 unit\(s\) across resample"):
        _curve_out(disjoint, rows)
    # and the same rule under `cpr`: the check is kind-agnostic
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    cpr_rows = [
        dict(r, method="A" if r["axes.cut"] < N // 2 else "B")
        for r in sweep_table(means, examples=4)
    ]
    with pytest.raises(StepError, match=r"splits 4 of 4 unit\(s\) across resample"):
        _curve_out(block("cpr", {"kind": "pair", "columns": ["method"]}), cpr_rows)
    mean = block("auc", {"kind": "pair", "columns": ["example", "method"]})
    mean["estimator"] = {"kind": "mean"}
    parse_reduction(mean)  # not a curve's concern
    assert math.isfinite(_curve_out(mean, rows)["value"])


def test_an_infinite_x_is_refused_on_both_estimators():
    """A null `x` is `missing`'s (refused or excluded); an infinite one is a
    value nothing else refused — under `auc` the last trapezoid had infinite
    width and ±inf/nan published, under `cpr` the integer check could not
    format it — so the table refuses it once, before any group is read and
    before `missing` is applied, naming the rows and their values. `x_scale:
    log`'s x ≤ 0 refusal has its companion."""
    rows = [
        {"value": 1.0, "axes.cut": 0},
        {"value": 2.0, "axes.cut": 5},
        {"value": 3.0, "axes.cut": math.inf},
    ]
    with pytest.raises(
        StepError,
        match=r"x \('axes.cut'\) is infinite in 1 row\(s\) \(row 2: inf\).*"
        r"\(`missing` governs a null value, not an infinite x\)$",
    ):
        _curve_out(_curve("auc"), rows)
    assert math.isfinite(_curve_out(_curve("auc"), rows[:2])["value"])
    # before `missing`: a null ordinate beside the infinite abscissa is still
    # refused under `exclude` — the table's defect, not a missing observation
    rows[2]["value"] = None
    with pytest.raises(StepError, match="is infinite in 1 row"):
        _curve_out(_curve("auc"), rows)
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    cpr_rows = sweep_table(means) + [
        {"example": 0, "metric": "ld", "value": 1.0, "axes.cut": -math.inf, "units": N}
    ]
    with pytest.raises(StepError, match=r"is infinite in 1 row\(s\) \(row \d+: -inf\)"):
        _curve_out(_curve("cpr", normalize=N), cpr_rows)


def test_the_missing_anchor_refusal_names_the_anchor_and_the_cut_range():
    """The one listing that was unbounded: a `cpr` table lacking `x = N`
    printed every distinct cut; capped to the five lowest it hid the maximum,
    the number that diagnoses a wrong `N`. It now names the absent anchor and
    the table's range — a bounded message whose content no longer depends on
    which five cuts sort lowest (`min`/`max` still walk the cuts; on an error
    path that walk is free)."""
    rows = [{"value": float(c), "axes.cut": c} for c in range(10)]
    with pytest.raises(
        StepError, match=r"no row at x = 100 — its 10 cut\(s\) run 0 … 9"
    ):
        _curve_out(_curve("cpr", normalize=100), rows)
    # both anchors absent: named together, the range unchanged
    with pytest.raises(
        StepError, match=r"no row at x = 0 or 100 — its 9 cut\(s\) run 1 … 9"
    ):
        _curve_out(_curve("cpr", normalize=100), rows[1:])


def test_a_one_point_auc_curve_is_refused_not_zero():
    """One distinct `x` has no area. Before the change `_trapezoid` over one
    element summed nothing and the row published `0.0`."""
    rows = [{"value": 1.0, "axes.cut": 5}, {"value": 3.0, "axes.cut": 5}]
    with pytest.raises(StepError, match="at least two distinct x"):
        _curve_out(_curve("auc"), rows)
    two = rows + [{"value": 1.0, "axes.cut": 7}]
    assert _curve_out(_curve("auc"), two)["value"] == pytest.approx(2 * (2.0 + 1.0) / 2)


def test_cpr_reads_a_balanced_panel_under_a_unit():
    """Under `unit: example`, a unit with no row at a cut the curve reads is
    refused naming the cut: otherwise `B`, `C` and `m(cut)` would average
    different examples (MIB's evaluator does; a declared estimand refuses).
    `missing: exclude` on a null row is how the hole arises. A hole at a cut
    the grid does *not* read is no hole, and `unit: row` has no panel."""
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    rows = sweep_table(means, examples=4)
    block = _curve("cpr", normalize=N)
    block["unit"] = {"kind": "example", "columns": ["example"]}
    half = _mib_cut(0.5)
    holed = [r for r in rows if not (r["example"] == 3 and r["axes.cut"] == half)]
    with pytest.raises(StepError, match=rf"balanced panel.*cut {half} \(1 unit"):
        _curve_out(block, holed)
    nulled = [
        dict(r, value=None) if (r["example"] == 3 and r["axes.cut"] == 0) else r
        for r in rows
    ]
    with pytest.raises(StepError, match=r"cut 0 \(1 unit"):
        _curve_out(block, nulled)
    # the same table under `unit: row` is rows, not a panel — it reduces
    assert math.isfinite(_curve_out(_curve("cpr", normalize=N), holed)["value"])
    # a cut no grid point reads may be ragged
    extra = rows + [{"example": 0, "value": 5.0, "axes.cut": 77, "units": N}]
    whole = _curve_out(block, rows)["value"]
    assert math.isfinite(whole)
    assert _curve_out(block, extra)["value"] == pytest.approx(whole)


def test_a_refusal_inside_a_bootstrap_draw_names_the_repetition():
    """With `resample_unit: row` over a `cpr` table of one row per cut, a draw
    of rows with replacement rarely carries every cut, while the table's own
    estimate is fine. The refusal says it came from a draw, and which."""
    n = 4  # the cuts read: 4 − int(p·4) over MIB's grid → {4, 2, 0}
    rows = [{"value": v, "axes.cut": c} for c, v in ((0, 10.0), (2, 6.0), (4, 2.0))]
    assert math.isfinite(_curve_out(_curve("cpr", normalize=n), rows)["value"])
    block = _curve("cpr", normalize=n)
    block["uncertainty"] = {
        "kind": "percentile_bootstrap",
        "resample_unit": {"kind": "row"},
        "repetitions": 50,
        "seed": 0,
    }
    with pytest.raises(
        StepError,
        # the repetition, and advice that points away from the failure: under
        # a declared unit a draw is whole curves, so this refusal is `unit:
        # row`'s (as here) or coinciding anchors
        match=r"inside bootstrap repetition \d+ of 50.*under a declared unit a draw "
        r"is whole curves",
    ):
        _curve_out(block, rows)


def test_an_authored_grid_point_selects_the_cut_int_selects():
    """`N − int(p·N)` in floating point, as MIB reads it: `0.29 × 100` is
    `28.999…`, so `p = 0.29` on 100 units reads cut 72 (28 kept), not 71.
    The arithmetic is kept — bit parity with `evaluate_area_under_curve` is
    the point of `cpr/v1` — and §2.6 says to check an authored point against
    the cut it selects; this pins which."""
    assert 0.29 * 100 == 28.999999999999996
    means = {0: 10.0, 100: 2.0, 71: 9.0, 72: 5.0}
    rows = [{"value": m, "axes.cut": c} for c, m in means.items()]
    out = _curve_out(_curve("cpr", normalize=100, grid=[0.29, 1.0]), rows)
    faith_at_72 = (5.0 - 2.0) / (10.0 - 2.0)
    faith_at_71 = (9.0 - 2.0) / (10.0 - 2.0)
    assert out["value"] == pytest.approx((1.0 - 0.29) * (faith_at_72 + 1.0) / 2)
    assert out["value"] != pytest.approx((1.0 - 0.29) * (faith_at_71 + 1.0) / 2)


def test_the_anchor_floor_is_scale_free():
    """`B = C` is judged against the curve's own magnitude: a curve of means
    near 1e-12 whose anchors differ by 2e-12 is as well defined as one near
    10 (an absolute `1e-6` floor refused it), while anchors equal to one part
    in 1e9 of the curve are refused at any scale."""
    means = {_mib_cut(p): 4.0 + (N - _mib_cut(p)) / N for p in MIB_GRID}
    means[0], means[N] = 9.0, 4.0
    block = _curve("cpr", normalize=N)
    unit_scale = _curve_out(block, sweep_table(means, examples=1))["value"]
    tiny = [
        {"example": 0, "value": (m - 4.0) * 4e-13, "axes.cut": c}
        for c, m in means.items()
    ]
    assert _curve_out(block, tiny)["value"] == pytest.approx(unit_scale)
    near = [
        {"example": 0, "value": 1e6 + (1e-4 if c == 0 else 0.0), "axes.cut": c}
        for c in means
    ]
    with pytest.raises(StepError, match="anchors are equal"):
        _curve_out(block, near)


def test_the_two_ordinate_refusal_binds_the_built_in_alone():
    """`inputs.value` is the built-in's convention: a step running it that
    declares `value` beside `estimator.y` is refused at load naming the
    module; a user script step with an input of that name means what its
    author says and parses."""
    block = _curve("cpr", normalize=N, y="ld")
    inputs = {"table": {"path": "/x/t.json"}, "value": "ld"}
    with pytest.raises(WorkflowError, match="both name the column the built-in"):
        parse_workflow(_step(block, inputs))
    user = _step(block, inputs)
    user["steps"]["r"]["script"] = {"path": "scripts/mine.py"}
    parse_workflow(user)
    # a `value` authored as a reference is described, not interpolated
    referenced = _step(
        block,
        {"table": {"path": "/x/t.json"}, "value": {"step": "a", "file": "v.json"}},
    )
    with pytest.raises(WorkflowError, match=r"inputs.value \(a reference\)"):
        parse_workflow(referenced)
    # a literal that is not a string is shown as itself, not as a reference —
    # the built-in takes `str()` of whatever `value` holds
    literal = _step(block, {"table": {"path": "/x/t.json"}, "value": {"column": "ld"}})
    with pytest.raises(WorkflowError, match=r"inputs.value \(\{'column': 'ld'\}\)"):
        parse_workflow(literal)


def test_the_script_refuses_two_spellings_of_the_ordinate(tmp_path):
    """`inputs.value` and `estimator.y` name the column to reduce; declaring
    both is refused (a field that governs nothing may not be declared), and
    each alone runs to the same number."""
    means = {_mib_cut(p): 6.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    rows = [dict(r, ld=r["value"]) for r in sweep_table(means)]
    table = put_table(tmp_path / "apply" / "ld.json", rows)
    block = _curve("cpr", normalize=N, y="ld")
    with pytest.raises(StepError, match="both name the column"):
        run_step(
            reduce,
            {"table": table, "value": "ld", REDUCTION_INPUT: block},
            {"table": tmp_path / "r.json"},
        )
    run_step(
        reduce, {"table": table, REDUCTION_INPUT: block}, {"table": tmp_path / "a.json"}
    )
    run_step(
        reduce,
        {"table": table, "value": "ld", REDUCTION_INPUT: _curve("cpr", normalize=N)},
        {"table": tmp_path / "b.json"},
    )
    a, b = read_table(tmp_path / "a.json")[0], read_table(tmp_path / "b.json")[0]
    assert a["value"] == pytest.approx(b["value"])


def test_the_missing_cut_refusal_names_every_grid_point_that_selected_the_cut():
    """Several ``p`` may floor to one cut — the flat segment ``cpr`` keeps —
    and a missing cut's fix list names each of them, not the last one: under
    ``grid: [0.3, 0.302]`` at ``N = 157`` both select cut 110 (``int(47.1)``
    and ``int(47.414)``), so an author who deleted one point would otherwise
    meet the same refusal for the other. Unreachable on ``MIB_GRID``: with
    consecutive ratios ≥ 2 a shared cut forces ``int(p·N) = 0`` at any ``N``,
    so the only cut two of its points share is ``N`` itself — MIB's three
    smallest, on 157 units — and an absent ``N`` is the anchor refusal's,
    which runs first."""
    means = {_mib_cut(p): 5.0 for p in MIB_GRID}
    means[0], means[N] = 10.0, 2.0
    block = _curve("cpr", normalize=N, grid=[0.3, 0.302])
    with pytest.raises(
        StepError, match=r"no rows at cuts 110 \(p = 0\.3, 0\.302\) — x must"
    ):
        _curve_out(block, sweep_table(means))
    # with the cut present both points read it: one flat segment of width 0.002
    out = _curve_out(block, sweep_table({**means, 110: 6.0}))
    assert out["value"] == pytest.approx(0.5 * (0.302 - 0.3))


def test_the_missing_cut_refusal_caps_the_grid_points_it_lists_too():
    """The per-cut listing is `_listed`'s as the cut listing is: an authored
    grid may put many ``p`` in one ``1/N``-wide window (here seven in
    ``[0.5, 1)`` at ``N = 2``, all cut 1), and a missing cut names the first
    five and how many more, not every float."""
    grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    block = _curve("cpr", normalize=2, grid=grid)
    with pytest.raises(
        StepError,
        match=r"no rows at cuts 1 \(p = 0\.5, 0\.55, 0\.6, 0\.65, 0\.7 and 2 more\) — x",
    ):
        _curve_out(block, sweep_table({0: 10.0, 2: 2.0}))
