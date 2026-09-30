"""Describe what a metric measures and identify its arithmetic.

Metric records carry a ``unit`` and ``estimand_version``. Their identity also
records the model read, answer forms, scoring mode, and relevant options.
Reduction code uses this information to check that measurements can be combined.
Keep definitions synchronized with the metric implementation and reference tables."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Any, Mapping, Sequence

__all__ = [
    "Claim",
    "Comparison",
    "EstimandError",
    "IDENTIFIER",
    "IDENTITY_COLUMNS",
    "METRIC_UNITS",
    "REDUCTION_ESTIMANDS",
    "Record",
    "ReductionEstimand",
    "UNITS",
    "admissible_reduction_identifiers",
    "check_claim",
    "compare",
    "metric_identity",
    "metric_record_identity",
    "parse_identifier",
    "reduction_identity",
    "reduction_unit",
    "table_record",
]

#: The closed unit vocabulary (§2.10). ``fraction`` is a probability or a
#: proportion in [0, 1]; ``percentage_points`` is the same quantity × 100 and
#: is a *different* unit, which is the whole point; ``count`` is a number of
#: things; ``logit`` a raw (or differenced) pre-softmax score; ``nat`` and
#: ``bit`` information in base e and base 2; ``dimensionless`` a ratio of two
#: like-unit quantities that is not a proportion. Grown by PR, with
#: ``tests/protocol/test_estimand.py`` holding the spec table to this tuple.
UNITS: tuple[str, ...] = (
    "fraction",
    "percentage_points",
    "count",
    "logit",
    "nat",
    "bit",
    "dimensionless",
)

#: The unit each metric kind's per-example value is in (``metrics.py``), or
#: ``None`` for a kind whose value is not a scalar. Keyed by kind name rather
#: than importing ``METRIC_KINDS`` so ``schema.py`` can import this module;
#: the census holds the two key sets equal.
METRIC_UNITS: dict[str, str | None] = {
    "logit_diff": "logit",  # logits[a] − logits[b]: a difference of logits
    "soft_accuracy": "fraction",  # σ(logits[a] − logits[b]): a margin squashed to (0, 1)
    "token_logit": "logit",  # the raw logit of one token
    "cross_entropy": "nat",  # −log_softmax(target): natural log
    "kl": "nat",  # Σ p (log p − log q): natural log
    "js": "nat",  # ½ KL(p‖m) + ½ KL(q‖m), m = ½(p+q): natural log, ≤ ln 2
    "class_probs": "fraction",  # softmax mass per group
    "token_logits": "logit",  # the raw logit of each listed token
    "top_k": None,  # a structure; its `values` carry the read's own unit
    "match": "fraction",  # a 0/1 indicator; its mean is the accuracy
    "decode": None,  # text
}

#: ``<estimand>/v<n>``: a snake_case arithmetic name and a positive version.
IDENTIFIER = re.compile(
    r"^(?P<estimand>[a-z][a-z0-9]*(?:_[a-z0-9]+)*)/v(?P<n>[1-9][0-9]*)$"
)

#: The two identity columns a metric row and a reduced row carry, in order.
IDENTITY_COLUMNS: tuple[str, ...] = ("unit", "estimand_version")


class EstimandError(ValueError):
    """A unit, an identifier or a claim is refused. Plain ``ValueError`` on
    purpose (the shape ``ReductionSpecError`` set): each caller wraps it into
    the error class of its own boundary — ``ParseError`` at the document,
    ``WorkflowError`` at a workflow step, ``StepError`` inside a script."""


# --------------------------------------------------------------------------- #
# the grammar
# --------------------------------------------------------------------------- #


def parse_identifier(text: Any) -> tuple[str, int]:
    """``(estimand, n)`` from ``<estimand>/v<n>``; anything else is refused
    with the grammar in the message."""
    if not isinstance(text, str) or (found := IDENTIFIER.match(text)) is None:
        raise EstimandError(
            f"{text!r} is not an estimand identifier — the grammar is "
            "'<estimand>/v<n>', a snake_case name for the arithmetic and a "
            "positive version (e.g. 'mean_of_eligible_row_ratios/v1')"
        )
    return found.group("estimand"), int(found.group("n"))


# --------------------------------------------------------------------------- #
# metric kinds — one arithmetic each, so the identity is derived
# --------------------------------------------------------------------------- #


def metric_identity(kind: str) -> str:
    """The identifier a metric kind's per-example value carries: ``<kind>/v1``.
    Every kind is exactly one arithmetic once its fields are fixed (``match``'s
    ``mode`` and ``top_k``'s ``by`` are fields, in the canonical form), so no
    kind admits a second identifier and none requires an authored one."""
    if kind not in METRIC_UNITS:
        raise EstimandError(f"unknown metric kind {kind!r}")
    return f"{kind}/v1"


def metric_record_identity(
    kind: str, *, unit: Any = None, estimand_version: Any = None
) -> dict[str, Any]:
    """The identity columns one metric row carries (``neural/shared/results.py``): the
    authored ``unit`` / ``estimand_version`` when the document states them,
    the kind's own otherwise. Authored values are checked against the kind at
    parse (``schema.py``), so the two never disagree here."""
    return {
        "unit": unit if unit is not None else METRIC_UNITS[kind],
        "estimand_version": (
            estimand_version if estimand_version is not None else metric_identity(kind)
        ),
    }


# --------------------------------------------------------------------------- #
# reductions — the estimator names the arithmetic; a campaign may name it too
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ReductionEstimand:
    """One admissible identifier for a ``reduction`` block (workflow §2.6):
    the estimator that computes it and the constraints on the block under
    which the name is honest. ``None`` is "any"."""

    identifier: str
    estimator: str
    arithmetic: str
    #: ``reduction.unit.kind`` the name presumes (``row`` for the two
    #: row-ratio identifiers), or any
    unit_kind: str | None = None
    #: ``"null"`` — no weight; ``"column"`` — a weight column; or any
    weight: str | None = None
    #: ``reduction.missing`` the name presumes, or any
    missing: str | None = None

    def admits(self, block: Mapping[str, Any]) -> bool:
        estimator = block.get("estimator")
        kind = estimator.get("kind") if isinstance(estimator, Mapping) else None
        if kind != self.estimator:
            return False
        unit = block.get("unit")
        unit_kind = unit.get("kind") if isinstance(unit, Mapping) else None
        if self.unit_kind is not None and unit_kind != self.unit_kind:
            return False
        if self.weight == "null" and block.get("weight") is not None:
            return False
        if self.weight == "column" and block.get("weight") is None:
            return False
        if self.missing is not None and block.get("missing") != self.missing:
            return False
        return True


#: Every identifier a reduction block may declare. The first eight are the
#: estimators' own — what an unauthored block is recorded as. The last two are
#: a pair: the same value column, the same rows, two
#: different questions, so they may not share a name — and each is admissible
#: only for the block that computes it, which is what rule 13 checks.
REDUCTION_ESTIMANDS: tuple[ReductionEstimand, ...] = (
    ReductionEstimand("mean/v1", "mean", "arithmetic mean of the observations"),
    ReductionEstimand(
        "weighted_mean/v1", "weighted_mean", "Σ w·v / Σ w over the observations"
    ),
    ReductionEstimand("sum/v1", "sum", "sum of the observations"),
    ReductionEstimand("count/v1", "count", "the number of observations"),
    ReductionEstimand("median/v1", "median", "the lower middle observation"),
    ReductionEstimand("quantile/v1", "quantile", "the q-quantile, interpolated"),
    ReductionEstimand(
        "auc/v1",
        "auc",
        "the trapezoid area under the per-x means of y over the sorted x, "
        "against the block's declared x_scale (x ÷ normalize when given; "
        "log x under x_scale: log) — the grid and scale are the block's, "
        "carried by its digest, not by the name",
    ),
    ReductionEstimand(
        "cpr/v1",
        "cpr",
        "MIB's CPR arithmetic over the block's declared kept-fraction grid p "
        "(MIB's ten points unless a grid is authored) and x_scale: the "
        "trapezoid of (m(N − int(p·N)) − C)/(B − C), B the mean at cut 0, C "
        "at cut N — MIB's number on MIB's grid, the block's otherwise",
    ),
    ReductionEstimand(
        "mean_of_eligible_row_ratios/v1",
        "mean",
        "per row, numerator ÷ denominator (the value column); then the mean "
        "over the eligible rows — the rows `missing: exclude` kept",
        unit_kind="row",
        weight="null",
        missing="exclude",
    ),
    ReductionEstimand(
        "ratio_of_sums/v1",
        "weighted_mean",
        "Σ numerator ÷ Σ denominator over the same rows: the value column is "
        "the per-row ratio and the weight column its denominator",
        unit_kind="row",
        weight="column",
    ),
)

_REDUCTION_ESTIMANDS_BY_ID: dict[str, ReductionEstimand] = {
    entry.identifier: entry for entry in REDUCTION_ESTIMANDS
}


def admissible_reduction_identifiers(block: Mapping[str, Any]) -> list[str]:
    """The identifiers a canonical ``reduction`` block may honestly declare,
    the estimator's own first."""
    return [entry.identifier for entry in REDUCTION_ESTIMANDS if entry.admits(block)]


def reduction_identity(block: Mapping[str, Any], authored: Any = None) -> str:
    """The identifier a reduction's output rows carry: the authored one when
    the block admits it, the estimator's own (``mean/v1``) otherwise. An
    authored identifier the block does not admit is refused naming what was
    declared, what is computed, and what the block may declare."""
    admissible = admissible_reduction_identifiers(block)
    if authored is None:
        return admissible[0]
    parse_identifier(authored)
    if authored in admissible:
        return str(authored)
    declared = _REDUCTION_ESTIMANDS_BY_ID.get(str(authored))
    what = (
        f"names {declared.arithmetic} (estimator {declared.estimator!r})"
        if declared is not None
        else "is not an identifier any reduction computes"
    )
    raise EstimandError(
        f"estimand_version {authored!r} {what}, but this block computes "
        f"{admissible[0]!r} — admissible here: {admissible}. A document may "
        "name the arithmetic it runs, not another one"
    )


def reduction_unit(estimator: str, table_unit: str | None) -> str | None:
    """The unit of a reduced value: ``count`` counts things whatever the table
    holds; ``cpr`` is a ratio of two like-unit differences integrated over a
    fraction axis — ``dimensionless`` whatever the metric was; ``auc`` is an
    x·y product no vocabulary member names — ``None``, unknown; every other
    estimator is in the table's unit (``None`` when the table carries none)."""
    if estimator == "count":
        return "count"
    if estimator == "cpr":
        return "dimensionless"
    if estimator == "auc":
        return None
    return table_unit


# --------------------------------------------------------------------------- #
# records, comparisons, claims
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Record:
    """The identity of one metric record as a comparison sees it: a label
    (a file, a step slot), its unit and its estimand. ``None`` is
    *undeclared*, never a unit of its own."""

    name: str
    unit: str | None = None
    estimand_version: str | None = None


@dataclasses.dataclass(frozen=True)
class Comparison:
    """A permitted comparison: ``arm`` when both records carry the same
    estimand (or neither declares one) — two arms of one campaign; ``version``
    when both declare and the estimands differ — a comparison *between
    arithmetics*, allowed and labelled so a reader knows which it was."""

    left: Record
    right: Record
    kind: str

    @property
    def unit(self) -> str | None:
        return self.left.unit if self.left.unit is not None else self.right.unit


def compare(left: Record, right: Record) -> Comparison:
    """Refuse two records whose units differ, naming both units and both
    records; otherwise say what kind of comparison this is.

    An undeclared unit on either side is not a mismatch — two tables written
    before units existed, or a script's own table, compare as they always did.
    The refusal is for two records that both say what they are and disagree,
    such as ``fraction`` against ``percentage_points``."""
    if left.unit is not None and right.unit is not None and left.unit != right.unit:
        raise EstimandError(
            f"unit mismatch: {left.name} is in {left.unit!r} and {right.name} is in "
            f"{right.unit!r} — a fraction cannot be compared to percentage points, "
            "and no unit is rescaled into another silently. Reduce one record into "
            "the other's unit and declare that reduction"
        )
    both = left.estimand_version is not None and right.estimand_version is not None
    kind = (
        "version" if both and left.estimand_version != right.estimand_version else "arm"
    )
    return Comparison(left=left, right=right, kind=kind)


def _column_values(rows: Sequence[Mapping[str, Any]], column: str) -> set[Any]:
    return {row.get(column) for row in rows}


def table_record(rows: Sequence[Mapping[str, Any]], *, name: str) -> Record:
    """The identity a table's rows carry — read from the repeated
    ``unit`` / ``estimand_version`` columns. A table whose rows disagree on
    their unit is refused: its rows cannot be reduced or compared as one
    record. Rows disagreeing on the estimand are one record with an
    undeclared estimand (a table mixing kinds is legal; a claim about it
    binds a reduction, not the table)."""
    units = _column_values(rows, "unit") - {None}
    if len(units) > 1:
        raise EstimandError(
            f"unit mismatch inside {name}: its rows are in "
            f"{' and '.join(repr(u) for u in sorted(units))} — a fraction cannot be "
            "reduced together with percentage points; reduce each unit on its own"
        )
    estimands = _column_values(rows, "estimand_version") - {None}
    return Record(
        name=name,
        unit=next(iter(units)) if units else None,
        estimand_version=next(iter(estimands)) if len(estimands) == 1 else None,
    )


@dataclasses.dataclass(frozen=True)
class Claim:
    """A number a report quotes, bound to the record it came from: the file,
    the row in it (``where``: a column → value mapping that selects the row —
    a reduced row's ``group_by`` coordinates, say; empty for a one-row
    table), the estimand and the unit, and the value. The binding is to the
    record's *content*: a recomputation that changes the value, the unit or
    the estimand at that row is refused naming file and coordinates. A rerun
    that lands the same number at the same row is, by design, the same
    record."""

    file: str
    where: Mapping[str, Any]
    estimand_version: str
    unit: str
    value: float


def check_claim(
    claim: Claim, rows: Sequence[Mapping[str, Any]], *, tolerance: float = 0.0
) -> None:
    """Refuse a claim its record no longer supports.

    ``rows`` are the claim's file as it is *now*. The record is the row (one)
    the claim's ``where`` mapping selects; it must carry the claim's unit and
    estimand, and its ``value`` must equal the claim's within ``tolerance``
    (exact by default: a recomputation that reproduces writes the same JSON
    number). Every refusal names the record — its file and the selecting
    columns — so the stale sentence can be found from the message alone."""
    selection = json.dumps(dict(claim.where), sort_keys=True, default=str)
    where = f"{claim.file} (where {selection})"
    # a column the file no longer carries is a different table, not a match:
    # a `where` entry binds a *present* column's value (a row without the
    # column is not a row where it is None)
    unknown = [k for k in claim.where if not any(k in row for row in rows)]
    if unknown:
        raise EstimandError(
            f"stale claim: {where} names column(s) {unknown} that no row of the "
            f"file carries — the document that wrote it changed, so the quoted "
            "value binds to nothing"
        )
    bound = [
        row
        for row in rows
        if all(k in row and row[k] == v for k, v in claim.where.items())
    ]
    if not bound:
        raise EstimandError(
            f"stale claim: {where} holds no row at those coordinates — the file "
            f"now carries {len(rows)} row(s), none matching; the document that "
            "wrote it changed, so the quoted value binds to nothing"
        )
    if len(bound) != 1:
        raise EstimandError(
            f"claim binds {len(bound)} rows in {where}; a claim binds one record — "
            "reduce the table first (workflow spec §2.6) and bind the reduced row, "
            "or name every column that singles it out"
        )
    record = table_record(bound, name=where)
    if record.unit != claim.unit:
        raise EstimandError(
            f"stale claim: {where} is in {record.unit!r}, the claim says {claim.unit!r}"
        )
    if record.estimand_version != claim.estimand_version:
        raise EstimandError(
            f"stale claim: {where} is {record.estimand_version!r}, the claim says "
            f"{claim.estimand_version!r}"
        )
    value = bound[0].get("value")
    try:
        current = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise EstimandError(
            f"stale claim: {where} holds value {value!r}, which is not a number"
        ) from None
    if not math.isclose(current, claim.value, rel_tol=0.0, abs_tol=tolerance):
        raise EstimandError(
            f"stale claim: {where} now produces {current!r}, the claim quotes "
            f"{claim.value!r} — the record was recomputed and the prose was not"
        )
