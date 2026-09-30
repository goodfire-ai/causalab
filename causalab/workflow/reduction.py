"""Define reductions over saved metric tables.

A reduction declares its grouping, estimator, eligibility rules, and uncertainty
procedure. Metric units and arithmetic versions are checked before rows are
combined. The built-in estimators include scalar summaries and curve measures.
Seeded resampling gives reproducible uncertainty estimates.

Numerical imports occur inside the operations that use them so workflow
validation can read the vocabulary before numerical libraries load."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from causalab.io.step_io import StepError
from causalab.protocol.estimand import (
    IDENTITY_COLUMNS,
    EstimandError,
    reduction_identity,
    reduction_unit,
    table_record,
)

__all__ = [
    "CONFIDENCE",
    "CURVE_ESTIMATORS",
    "ESTIMATORS",
    "Estimator",
    "IDENTITY_RULE",
    "MIB_GRID",
    "MISSING_POLICIES",
    "OUTPUT_COLUMNS",
    "REDUCE_MODULE",
    "REDUCTION_INPUT",
    "Reduction",
    "ReductionSpecError",
    "UNCERTAINTY_KINDS",
    "UNIT_KINDS",
    "Uncertainty",
    "Unit",
    "X_SCALES",
    "parse_reduction",
    "reduce_frame",
]

#: The workflow checklist rule the estimand-identity refusal fires under
#: (rule 12 is the block's shape and vocabulary).
IDENTITY_RULE = 13

#: What one independent observation is. ``row`` is the table's own row and
#: names no column; every other member names the column(s) that identify one
#: observation in *this* table — only the campaign knows its key.
UNIT_KINDS: tuple[str, ...] = (
    "row",
    "example",
    "pair",
    "prompt",
    "source_family",
    "component",
)

#: What happens to a ``null`` value (or weight) before the estimator runs.
MISSING_POLICIES: tuple[str, ...] = ("error", "exclude", "zero")

#: The uncertainty procedures.
UNCERTAINTY_KINDS: tuple[str, ...] = ("none", "percentile_bootstrap", "normal_approx")

#: The estimators. Where a verb is also a ``save.reduce`` verb it means the
#: same thing: ``median`` is the lower of the two middle values, no
#: interpolation; ``count`` is a denominator. ``auc`` and ``cpr`` are the two
#: curve estimators ([`CURVE_ESTIMATORS`][]): one area from a table of
#: ``(x, y)`` rows.
ESTIMATORS: tuple[str, ...] = (
    "mean",
    "weighted_mean",
    "sum",
    "count",
    "median",
    "quantile",
    "auc",
    "cpr",
)

#: The estimators whose observation is a point on a curve rather than a
#: scalar. They take ``x`` (the abscissa column), an optional ``y``,
#: ``x_scale``, ``normalize`` and (``cpr``) ``grid`` on the estimator
#: (spec §2.6, "Curve estimators").
CURVE_ESTIMATORS: tuple[str, ...] = ("auc", "cpr")

#: The abscissa a curve estimator's trapezoid runs over: the x values as they
#: are, or their natural logarithm.
X_SCALES: tuple[str, ...] = ("linear", "log")

#: MIB's kept-fraction grid (``MIB_circuit_track/evaluation.py``,
#: ``percentages``) — ``cpr``'s ``grid`` when none is authored. Raw fractions:
#: the trapezoid runs over these numbers, never over counts or percentages.
MIB_GRID: tuple[float, ...] = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
)

#: The ``inputs`` key the runner hands an authored block under. A step that
#: authors ``reduction`` may not also declare an input by this name (rule 12).
REDUCTION_INPUT = "reduction"

#: The built-in reduce step, by module — the one script whose conventions
#: this module fixes: it reads the authored block under
#: [`REDUCTION_INPUT`][], and its optional ``value`` input names the column
#: to reduce, so the workflow parser holds ``inputs.value`` against
#: ``estimator.y`` for this module alone (a user script's input of that name
#: means what its author says).
REDUCE_MODULE = "causalab.workflow.scripts.reduce"

#: The confidence level of both interval procedures. Not a dimension: the
#: eight are what a review needs to answer *what* the interval is across; its
#: level is one convention, stated here and in §2.6.
CONFIDENCE = 0.95

#: The columns every reduced row carries beside its group coordinates, in
#: order: the value, the counts, then the record's identity (spec §2.6).
#: ``lower``/``upper`` appear only when uncertainty is declared.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "value",
    "n",
    "n_rows",
    "n_missing",
    "n_unmatched",
    "n_excluded",
    *IDENTITY_COLUMNS,
)
INTERVAL_COLUMNS: tuple[str, ...] = ("lower", "upper")

#: The column a windowed metric writes to say an example addressed nothing
#: (``causalab/neural/shared/results.py`` ``add_windowed``).
MATCHED_COLUMN = "matched"

#: z for a two-sided 95% normal interval. Stated to the digits used rather
#: than computed, so the interval is a function of the declaration alone.
_Z_95 = 1.959963984540054


class ReductionSpecError(ValueError):
    """An authored ``reduction`` block is incomplete or off-vocabulary.

    ``field`` names the offending dimension (dotted, e.g.
    ``uncertainty.seed``) so the loader can refuse *naming the field* — which
    is the whole of what rule 12 promises. ``rule`` is the checklist rule the
    loader raises under: 12 for shape and vocabulary, [`IDENTITY_RULE`][]
    for an ``estimand_version`` the block does not compute."""

    def __init__(self, field: str, message: str, *, rule: int = 12) -> None:
        self.field = field
        self.rule = rule
        super().__init__(f"reduction.{field}: {message}")


# --------------------------------------------------------------------------- #
# the declaration
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Unit:
    kind: str
    columns: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {"kind": self.kind, "columns": list(self.columns)}


@dataclasses.dataclass(frozen=True)
class Estimator:
    kind: str
    #: the quantile, for ``quantile`` only
    q: float | None = None
    #: curve estimators only (``auc``, ``cpr``): the abscissa column
    x: str | None = None
    #: the ordinate column, when it is not the step's value column
    y: str | None = None
    #: ``linear`` | ``log`` — what the trapezoid runs over
    x_scale: str | None = None
    #: the x ceiling ``N``: a number, or a column constant within each group.
    #: ``cpr`` requires it (its anchors are the cuts ``0`` and ``N``); ``auc``
    #: divides x by it when given
    normalize: str | int | float | None = None
    #: ``cpr`` only: the kept-fraction grid, when it is not [`MIB_GRID`][]
    grid: tuple[float, ...] | None = None

    @property
    def is_curve(self) -> bool:
        return self.kind in CURVE_ESTIMATORS

    def canonical(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        for name in ("q", "x", "y", "x_scale", "normalize"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.grid is not None:
            out["grid"] = list(self.grid)
        return out


@dataclasses.dataclass(frozen=True)
class Uncertainty:
    kind: str
    resample_unit: Unit | None = None
    repetitions: int | None = None
    seed: int | None = None

    def canonical(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        if self.resample_unit is not None:
            out["resample_unit"] = self.resample_unit.canonical()
        if self.repetitions is not None:
            out["repetitions"] = self.repetitions
        if self.seed is not None:
            out["seed"] = self.seed
        return out


@dataclasses.dataclass(frozen=True)
class Reduction:
    """One complete declaration — every field validated against its closed
    vocabulary, ready to run or to canonicalize."""

    estimator: Estimator
    unit: Unit
    group_by: tuple[str, ...]
    weight: str | None
    missing: str
    uncertainty: Uncertainty
    #: the authored estimand identifier, or ``None`` — like the block itself,
    #: in the canonical form only when authored; [`identity`][] is what the
    #: output rows carry either way
    estimand_version: str | None = None

    def canonical(self) -> dict[str, Any]:
        """The form that enters ``steps.<name>.reduction`` of the canonical
        entry, and therefore the step's identity (§7)."""
        out: dict[str, Any] = {
            "estimator": self.estimator.canonical(),
            "unit": self.unit.canonical(),
            "group_by": list(self.group_by),
            "weight": self.weight,
            "missing": self.missing,
            "uncertainty": self.uncertainty.canonical(),
        }
        if self.estimand_version is not None:
            out["estimand_version"] = self.estimand_version
        return out

    @property
    def identity(self) -> str:
        """The identifier this reduction's rows carry: authored, or the
        estimator's own (``estimand.reduction_identity``)."""
        return reduction_identity(self.canonical(), self.estimand_version)

    def y_column(self, default: str) -> str:
        """The column the estimator reduces: a curve estimator's authored
        ``y``, else the step's value column."""
        return self.estimator.y if self.estimator.y is not None else default

    @property
    def columns_named(self) -> dict[str, str]:
        """Every column the declaration binds → the field that binds it, so a
        run-time refusal can say which dimension named a column the table
        does not have."""
        named: dict[str, str] = {}
        for column in self.unit.columns:
            named.setdefault(column, "unit")
        if self.estimator.x is not None:
            named.setdefault(self.estimator.x, "estimator.x")
        if self.estimator.y is not None:
            named.setdefault(self.estimator.y, "estimator.y")
        if isinstance(self.estimator.normalize, str):
            named.setdefault(self.estimator.normalize, "estimator.normalize")
        for column in self.group_by:
            named.setdefault(column, "group_by")
        if self.weight is not None:
            named.setdefault(self.weight, "weight")
        if self.uncertainty.resample_unit is not None:
            for column in self.uncertainty.resample_unit.columns:
                named.setdefault(column, "uncertainty.resample_unit")
        return named


def _suggest(value: str, allowed: Sequence[str]) -> str:
    from causalab.protocol.rules.errors import suggest

    return suggest(value, allowed)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_unit(raw: Any, field: str) -> Unit:
    if not isinstance(raw, Mapping):
        raise ReductionSpecError(
            field, 'is an object {"kind": <vocabulary member>, "columns": [...]}'
        )
    unknown = sorted(set(raw) - {"kind", "columns"})
    if unknown:
        raise ReductionSpecError(field, f"unknown key(s) {unknown}")
    if "kind" not in raw:
        raise ReductionSpecError(f"{field}.kind", "is required")
    kind = raw["kind"]
    if kind not in UNIT_KINDS:
        raise ReductionSpecError(
            f"{field}.kind",
            f"{kind!r} is not one of {list(UNIT_KINDS)}"
            f"{_suggest(str(kind), UNIT_KINDS)}",
        )
    columns_raw = raw.get("columns", [])
    if isinstance(columns_raw, str) or not isinstance(columns_raw, (list, tuple)):
        raise ReductionSpecError(f"{field}.columns", "is a list of column names")
    columns = tuple(str(c) for c in columns_raw)
    if any(not c for c in columns):
        raise ReductionSpecError(f"{field}.columns", "holds an empty column name")
    if len(set(columns)) != len(columns):
        raise ReductionSpecError(f"{field}.columns", f"names a column twice: {columns}")
    if kind == "row" and columns:
        raise ReductionSpecError(
            f"{field}.columns",
            "'row' is the table's own row and names no column — drop 'columns'",
        )
    if kind != "row" and not columns:
        raise ReductionSpecError(
            f"{field}.columns",
            f"{kind!r} names the column(s) that identify one {kind} in this "
            "table — only the campaign knows its key",
        )
    return Unit(kind=str(kind), columns=columns)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: Every key an estimator object may carry; which apply is the kind's.
_ESTIMATOR_KEYS: tuple[str, ...] = (
    "kind",
    "q",
    "x",
    "y",
    "x_scale",
    "normalize",
    "grid",
)
_CURVE_KEYS: tuple[str, ...] = ("x", "y", "x_scale", "normalize", "grid")


def _parse_estimator(raw: Any) -> Estimator:
    if not isinstance(raw, Mapping):
        raise ReductionSpecError(
            "estimator",
            'is an object {"kind": <ESTIMATORS member>, …} — "q" on quantile; '
            '"x", "y", "x_scale", "normalize", "grid" on a curve estimator',
        )
    unknown = sorted(set(raw) - set(_ESTIMATOR_KEYS))
    if unknown:
        raise ReductionSpecError("estimator", f"unknown key(s) {unknown}")
    if "kind" not in raw:
        raise ReductionSpecError("estimator.kind", "is required")
    kind = raw["kind"]
    if kind not in ESTIMATORS:
        raise ReductionSpecError(
            "estimator.kind",
            f"{kind!r} is not one of {list(ESTIMATORS)}{_suggest(str(kind), ESTIMATORS)}",
        )
    kind = str(kind)
    q = raw.get("q")
    if kind == "quantile":
        if not _is_number(q):
            raise ReductionSpecError("estimator.q", "a quantile declares q in (0, 1)")
        if not 0.0 < float(q) < 1.0:
            raise ReductionSpecError("estimator.q", f"{q!r} is not in (0, 1)")
    elif q is not None:
        raise ReductionSpecError(
            "estimator.q", f"only 'quantile' takes q, not {kind!r}"
        )
    if kind in CURVE_ESTIMATORS:
        return _parse_curve_estimator(kind, raw)
    for name in _CURVE_KEYS:
        if name in raw:
            raise ReductionSpecError(
                f"estimator.{name}",
                f"only a curve estimator ({' | '.join(CURVE_ESTIMATORS)}) takes it, "
                f"not {kind!r}",
            )
    if kind == "quantile":
        return Estimator(kind="quantile", q=float(q))
    return Estimator(kind=kind)


def _parse_curve_estimator(kind: str, raw: Mapping[str, Any]) -> Estimator:
    """``auc`` / ``cpr``: the abscissa column, the optional ordinate, the
    scale, the ceiling and (``cpr`` only) the kept-fraction grid."""
    x = raw.get("x")
    if not isinstance(x, str) or not x:
        raise ReductionSpecError(
            "estimator.x",
            f"{kind!r} names the column holding the curve's abscissa (a cut, a "
            "kept count) — required",
        )
    y = raw.get("y")
    if y is not None and (not isinstance(y, str) or not y):
        raise ReductionSpecError(
            "estimator.y",
            "is the ordinate column — a column name, or absent for the step's "
            "value column",
        )
    if y == x:
        raise ReductionSpecError("estimator.y", f"names the same column as x ({x!r})")
    if "x_scale" not in raw:
        raise ReductionSpecError(
            "estimator.x_scale",
            f"is required by {kind!r}: one of {list(X_SCALES)} — the abscissa the "
            "trapezoid runs over",
        )
    x_scale = raw["x_scale"]
    if x_scale not in X_SCALES:
        raise ReductionSpecError(
            "estimator.x_scale",
            f"{x_scale!r} is not one of {list(X_SCALES)}"
            f"{_suggest(str(x_scale), X_SCALES)}",
        )
    normalize: str | int | float | None = raw.get("normalize")
    if kind == "cpr" and normalize is None:
        raise ReductionSpecError(
            "estimator.normalize",
            "'cpr' needs the unit count N — an integer ≥ 2, or a column that holds "
            "it: its anchors are the cuts 0 (clean) and N (fully corrupted)",
        )
    if isinstance(normalize, str):
        if not normalize:
            raise ReductionSpecError(
                "estimator.normalize", "holds an empty column name"
            )
        if normalize in (x, y):
            raise ReductionSpecError(
                "estimator.normalize", f"names the curve's own column {normalize!r}"
            )
    elif normalize is not None:
        if not _is_number(normalize) or not math.isfinite(float(normalize)):
            raise ReductionSpecError(
                "estimator.normalize",
                f"is a number (the x ceiling) or a column name, got {normalize!r}",
            )
        if kind == "cpr":
            if float(normalize) != int(normalize) or int(normalize) < 2:
                raise ReductionSpecError(
                    "estimator.normalize",
                    f"'cpr' takes an integer unit count ≥ 2, got {normalize!r}",
                )
            normalize = int(normalize)
        else:
            if not float(normalize) > 0.0:
                raise ReductionSpecError(
                    "estimator.normalize",
                    f"is a positive number (the x ceiling), got {normalize!r}",
                )
            normalize = float(normalize)
    grid: tuple[float, ...] | None = None
    if "grid" in raw:
        if kind != "cpr":
            raise ReductionSpecError(
                "estimator.grid",
                f"only 'cpr' takes a grid — {kind!r} integrates over the x values "
                "the table holds",
            )
        grid_raw = raw["grid"]
        if (
            isinstance(grid_raw, str)
            or not isinstance(grid_raw, (list, tuple))
            or len(grid_raw) < 2
        ):
            raise ReductionSpecError(
                "estimator.grid",
                "is a list of at least two kept fractions in (0, 1], increasing",
            )
        points: list[float] = []
        for p in grid_raw:
            if not _is_number(p) or not 0.0 < float(p) <= 1.0:
                raise ReductionSpecError(
                    "estimator.grid", f"holds {p!r}, not a fraction in (0, 1]"
                )
            points.append(float(p))
        if any(b <= a for a, b in zip(points, points[1:])):
            raise ReductionSpecError(
                "estimator.grid", f"is not strictly increasing: {points}"
            )
        grid = tuple(points)
    return Estimator(
        kind=kind, x=x, y=y, x_scale=str(x_scale), normalize=normalize, grid=grid
    )


def _parse_uncertainty(raw: Any, estimator: Estimator) -> Uncertainty:
    field = "uncertainty"
    if not isinstance(raw, Mapping):
        raise ReductionSpecError(field, 'is an object {"kind": …}')
    allowed = {"kind", "resample_unit", "repetitions", "seed"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ReductionSpecError(field, f"unknown key(s) {unknown}")
    if "kind" not in raw:
        raise ReductionSpecError(f"{field}.kind", "is required")
    kind = raw["kind"]
    if kind not in UNCERTAINTY_KINDS:
        raise ReductionSpecError(
            f"{field}.kind",
            f"{kind!r} is not one of {list(UNCERTAINTY_KINDS)}"
            f"{_suggest(str(kind), UNCERTAINTY_KINDS)}",
        )
    takes = {
        "none": (),
        "percentile_bootstrap": ("resample_unit", "repetitions", "seed"),
        "normal_approx": ("resample_unit",),
    }[str(kind)]
    for name in ("resample_unit", "repetitions", "seed"):
        if name in takes and name not in raw:
            raise ReductionSpecError(
                f"{field}.{name}", f"is required by {kind!r} and missing"
            )
        if name not in takes and name in raw:
            raise ReductionSpecError(
                f"{field}.{name}",
                f"{kind!r} does not take it — a field that governs nothing "
                "may not be declared",
            )
    if kind == "none":
        return Uncertainty(kind="none")
    resample_unit = _parse_unit(raw["resample_unit"], f"{field}.resample_unit")
    if kind == "normal_approx":
        if estimator.kind not in ("mean", "weighted_mean"):
            raise ReductionSpecError(
                f"{field}.kind",
                "'normal_approx' is the standard error of a mean — with "
                f"estimator {estimator.kind!r} use 'percentile_bootstrap'",
            )
        return Uncertainty(kind="normal_approx", resample_unit=resample_unit)
    repetitions = raw["repetitions"]
    if not _is_int(repetitions) or repetitions < 1:
        raise ReductionSpecError(
            f"{field}.repetitions", f"is a positive integer, got {repetitions!r}"
        )
    seed = raw["seed"]
    if not _is_int(seed) or seed < 0:
        raise ReductionSpecError(
            f"{field}.seed",
            f"is a non-negative integer (what seeds a numpy Generator), got {seed!r}",
        )
    return Uncertainty(
        kind="percentile_bootstrap",
        resample_unit=resample_unit,
        repetitions=int(repetitions),
        seed=int(seed),
    )


def parse_reduction(raw: Any) -> Reduction:
    """Parse and validate one authored ``reduction`` block.

    Every refusal is a [`ReductionSpecError`][] naming the field. Column
    *existence* is not checked here — that is data, and [`reduce_frame`][]
    refuses it at run time against the real table."""
    if not isinstance(raw, Mapping):
        raise ReductionSpecError("", "is an object")
    required = ("estimator", "unit", "group_by", "weight", "missing", "uncertainty")
    allowed = (*required, "estimand_version")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ReductionSpecError(
            "", f"unknown key(s) {unknown}{_suggest(unknown[0], allowed)}"
        )
    for name in required:
        if name not in raw:
            raise ReductionSpecError(
                name,
                "is required — an authored reduction declares every dimension, "
                f"including {name!r}" + (" as null" if name == "weight" else ""),
            )
    estimator = _parse_estimator(raw["estimator"])
    unit = _parse_unit(raw["unit"], "unit")
    group_raw = raw["group_by"]
    if isinstance(group_raw, str) or not isinstance(group_raw, (list, tuple)):
        raise ReductionSpecError("group_by", "is a list of column names (may be empty)")
    group_by = tuple(str(c) for c in group_raw)
    if any(not c for c in group_by):
        raise ReductionSpecError("group_by", "holds an empty column name")
    if len(set(group_by)) != len(group_by):
        raise ReductionSpecError("group_by", f"names a column twice: {group_by}")
    if estimator.is_curve and estimator.x in group_by:
        raise ReductionSpecError(
            "group_by",
            f"names the curve's abscissa {estimator.x!r} — grouped by x, every "
            "curve is one point and every area 0.0",
        )
    if estimator.is_curve and estimator.x in unit.columns:
        raise ReductionSpecError(
            "unit.columns",
            f"names the curve's abscissa {estimator.x!r} — a unit keyed on x holds "
            "one cut each, so no unit holds a curve",
        )
    weight = raw["weight"]
    if weight is not None and (not isinstance(weight, str) or not weight):
        raise ReductionSpecError("weight", f"is a column name or null, got {weight!r}")
    if estimator.kind == "weighted_mean" and weight is None:
        raise ReductionSpecError("weight", "'weighted_mean' needs a weight column")
    if estimator.kind != "weighted_mean" and weight is not None:
        raise ReductionSpecError(
            "weight",
            f"{estimator.kind!r} takes no weight — declare null, or use "
            "'weighted_mean'",
        )
    missing = raw["missing"]
    if missing not in MISSING_POLICIES:
        raise ReductionSpecError(
            "missing",
            f"{missing!r} is not one of {list(MISSING_POLICIES)}"
            f"{_suggest(str(missing), MISSING_POLICIES)}",
        )
    uncertainty = _parse_uncertainty(raw["uncertainty"], estimator)
    if (
        estimator.is_curve
        and uncertainty.resample_unit is not None
        and estimator.x in uncertainty.resample_unit.columns
    ):
        # resampling cuts resamples the abscissa: each draw integrates a
        # different grid, and the interval is a spread over grids, not over
        # the population — the same error as `group_by` naming x, in the
        # dimension that decides what the interval means
        raise ReductionSpecError(
            "uncertainty.resample_unit.columns",
            f"names the curve's abscissa {estimator.x!r} — resampling cuts "
            "resamples the grid, so the interval would be a spread over grids, "
            "not over the population",
        )
    if (
        estimator.is_curve
        and unit.kind != "row"
        and uncertainty.resample_unit is not None
        and uncertainty.resample_unit.kind == "row"
    ):
        # the panel guarantee (`_check_panel`) makes a draw of whole units a
        # draw of whole curves only when each unit lies in one resample
        # cluster. Rows are inside every unit, so a row draw splits every
        # unit's curve — decidable here. Whether a *column* resample unit
        # splits a unit is the table's to say, not the column names' (a
        # coarser unit declared by its own column — examples nested in
        # templates — is the textbook cluster bootstrap and passes), so that
        # half is `_check_clusters_hold_units`, at run time
        raise ReductionSpecError(
            "uncertainty.resample_unit",
            f"is 'row' under unit {list(unit.columns)} — rows are inside every "
            "unit, so a row draw splits each unit's curve across draws and a "
            "draw can integrate a grid the table does not have (a lost cut is "
            "refused under 'cpr'; under 'auc' the shorter grid integrates "
            "silently); resample whole units, or coarser ones",
        )
    parsed = Reduction(
        estimator=estimator,
        unit=unit,
        group_by=group_by,
        weight=weight,
        missing=str(missing),
        uncertainty=uncertainty,
    )
    authored = raw.get("estimand_version")
    if authored is None:
        return parsed
    # rule 13: the authored identifier must be one this block computes —
    # `mean_of_eligible_row_ratios/v1` on a `weighted_mean` block is a document
    # declaring one arithmetic while running another
    try:
        reduction_identity(parsed.canonical(), authored)
    except EstimandError as err:
        raise ReductionSpecError(
            "estimand_version", str(err), rule=IDENTITY_RULE
        ) from err
    return dataclasses.replace(parsed, estimand_version=str(authored))


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #


def _observations(
    values: Any, weights: Any, unit_ids: Any, estimator: Estimator
) -> tuple[Any, Any]:
    """Collapse rows to observations (§2.6, "what ``unit`` does").

    ``unit_ids`` is ``None`` for ``unit: row`` (no collapse) or an int array
    labelling each row's unit. Returns ``(obs_values, obs_weights)``;
    ``obs_weights`` is ``None`` when no weight is declared."""
    import numpy as np

    if unit_ids is None:
        return values, weights
    n_units = int(unit_ids.max()) + 1 if len(unit_ids) else 0
    counts = np.bincount(unit_ids, minlength=n_units).astype(float)
    if estimator.kind in ("sum", "count"):
        obs = np.bincount(unit_ids, weights=values, minlength=n_units)
        return obs, None
    if weights is None:
        sums = np.bincount(unit_ids, weights=values, minlength=n_units)
        return sums / counts, None
    weight_sums = np.bincount(unit_ids, weights=weights, minlength=n_units)
    weighted = np.bincount(unit_ids, weights=values * weights, minlength=n_units)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs = weighted / weight_sums
    return obs, weight_sums


def _estimate(values: Any, weights: Any, estimator: Estimator) -> float:
    """The estimator over observations. Empty → NaN (written as ``null``)."""
    import numpy as np

    n = len(values)
    if estimator.kind == "count":
        return float(n)
    if n == 0:
        return float("nan")
    if estimator.kind == "mean":
        return float(np.mean(values))
    if estimator.kind == "weighted_mean":
        total = float(np.sum(weights))
        if total == 0.0:
            return float("nan")
        # a zero-weight observation contributes nothing — including the NaN a
        # zero-total-weight unit collapsed to, which 0 * NaN would not remove
        return float(np.sum(np.where(weights == 0.0, 0.0, values * weights)) / total)
    if estimator.kind == "sum":
        return float(np.sum(values))
    if estimator.kind == "median":
        # the lower of the two middle values at even n — the same choice as
        # save.reduce's median, so the one verb means one thing at both times
        return float(np.sort(values)[(n - 1) // 2])
    return float(np.quantile(values, float(estimator.q or 0.5)))


def _curve_observations(xs: Any, values: Any, unit_ids: Any) -> tuple[Any, Any]:
    """Collapse rows to observations on a curve (§2.6, "Curve estimators"):
    rows sharing the unit key *and* an x value are one observation, by their
    mean. ``unit: row`` collapses nothing. Returns ``(obs_xs, obs_values)``."""
    import numpy as np

    if unit_ids is None or len(xs) == 0:
        return xs, values
    keys = np.stack([np.asarray(unit_ids, dtype=float), xs], axis=1)
    distinct, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = np.bincount(inverse, minlength=len(distinct)).astype(float)
    sums = np.bincount(inverse, weights=values, minlength=len(distinct))
    return distinct[:, 1], sums / counts


def _means_by_x(xs: Any, values: Any) -> tuple[Any, Any]:
    """The curve ``m(x)``: the distinct x, sorted, and the mean of the
    observations at each."""
    import numpy as np

    distinct, inverse = np.unique(xs, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    counts = np.bincount(inverse, minlength=len(distinct)).astype(float)
    sums = np.bincount(inverse, weights=values, minlength=len(distinct))
    return distinct, sums / counts


def _trapezoid(xs: Sequence[float], ys: Sequence[float]) -> float:
    """MIB's integral: consecutive trapezoids over the raw ``xs``. The
    abscissae are distinct by construction (``auc``'s from ``np.unique``,
    ``cpr``'s a strictly increasing grid); a repeated *ordinate* — two grid
    points reading one cut — is a flat segment of nonzero width."""
    return float(
        sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0 for i in range(len(xs) - 1))
    )


def _curve_value(
    xs: Any, values: Any, estimator: Estimator, ceiling: float | None, *, what: str
) -> float:
    """``auc`` / ``cpr`` over curve observations. Empty → NaN."""
    import numpy as np

    if len(values) == 0:
        return float("nan")
    grid_x, means = _means_by_x(xs, values)
    if estimator.kind == "auc":
        if len(grid_x) < 2:
            raise StepError(
                f"{what}: 'auc' needs at least two distinct x ({estimator.x!r}) — "
                f"the curve is one point, x = {float(grid_x[0])!r}, and has no area"
            )
        # the ceiling is positive (parsed, or checked by `_ceiling`), so the
        # sign test on the scaled abscissa is the sign test on x
        abscissa = grid_x / ceiling if ceiling is not None else grid_x
        if estimator.x_scale == "log":
            if bool(np.any(abscissa <= 0.0)):
                raise StepError(
                    f"{what}: estimator.x_scale is 'log' but x ({estimator.x!r}) "
                    f"holds {float(grid_x.min())!r} — every x must be positive (a "
                    "cut of 0 has no logarithm; exclude the row or use 'linear')"
                )
            abscissa = np.log(abscissa)
        return _trapezoid(list(abscissa), list(means))
    # cpr — MIB's `evaluate_area_under_curve`, cut by cut
    assert ceiling is not None
    units = int(ceiling)
    fractional = grid_x[np.mod(grid_x, 1.0) != 0.0]
    if len(fractional):
        raise StepError(
            f"{what}: 'cpr' reads integer cuts from x ({estimator.x!r}), but it "
            f"holds "
            f"{_listed([repr(float(v)) for v in fractional[:5]], len(fractional))}"
        )
    by_cut = {int(x): float(m) for x, m in zip(grid_x, means)}
    absent = [c for c in (0, units) if c not in by_cut]
    if absent:
        # evidence about the table's shape, not a list of things to fix: the
        # absent anchor is nearly always `units` (an authored or read `N`), and
        # the number that diagnoses it is the table's largest cut — so the
        # message names the anchor and the range, never an elided listing
        raise StepError(
            f"{what}: 'cpr' needs both anchors — x = 0 (the clean model, MIB's B) "
            f"and x = {units} (fully corrupted, MIB's C); the table has no row at "
            f"x = {' or '.join(str(c) for c in absent)} — its {len(by_cut)} cut(s) run "
            f"{min(by_cut)} … {max(by_cut)}"
        )
    baseline, corrupted = by_cut[0], by_cut[units]
    grid = _cpr_grid(estimator)
    cuts = _cpr_cuts(estimator, units)
    missing = sorted({c for c in cuts if c not in by_cut})
    if missing:
        # the fix list names every grid point that selected each missing cut —
        # what a reader would otherwise derive from the grid, whose elided
        # tail (the large p) is exactly the half a truncated sweep lacks —
        # grouped per cut: several p may floor to one cut (the flat segment the
        # estimator keeps), and a dict over the pairs would keep the last only;
        # both listings are capped (`_listed`), the per-cut one too, since an
        # authored grid may put many p in one 1/N-wide window
        by_point: dict[int, list[float]] = {}
        for p, c in zip(grid, cuts):
            by_point.setdefault(c, []).append(p)
        listed = _listed(
            [
                f"{c} (p = "
                + _listed([repr(p) for p in by_point[c][:5]], len(by_point[c]))
                + ")"
                for c in missing[:5]
            ],
            len(missing),
        )
        raise StepError(
            f"{what}: 'cpr' has no rows at cuts {listed} — x must include every "
            f"grid point's N − int(p·N) for N = {units}"
        )
    # a floor relative to the curve's own magnitude — the anchors and the cuts
    # it integrates, not whatever else the table carries — and to no metric
    # unit: faith is scale-free, so a curve of means near 1e-12 with anchors
    # 2e-12 apart is as well defined as one near 10, and B == C exactly is
    # refused whatever the scale (a zero floor when every read mean is 0)
    scale = max(abs(by_cut[c]) for c in (0, units, *cuts))
    if abs(baseline - corrupted) <= 1e-9 * scale:
        raise StepError(
            f"{what}: the clean and corrupted anchors are equal ({baseline!r}, "
            f"{corrupted!r}) — (B − C) is zero, so the faithfulness curve is "
            "undefined"
        )
    faith = [(by_cut[c] - corrupted) / (baseline - corrupted) for c in cuts]
    abscissa = [math.log(p) for p in grid] if estimator.x_scale == "log" else list(grid)
    return _trapezoid(abscissa, faith)


def _cpr_grid(estimator: Estimator) -> tuple[float, ...]:
    """The kept-fraction grid a ``cpr`` curve is read on: the authored one, or
    MIB's ten points. The one place it is resolved, so the abscissa and the
    cuts cannot come from different grids."""
    return tuple(estimator.grid) if estimator.grid is not None else MIB_GRID


def _cpr_cuts(estimator: Estimator, units: int) -> list[int]:
    """The cuts a ``cpr`` curve reads, one per grid point: MIB's ``N − int(p·N)``
    in floating point — ``0.29 × 100`` is ``28.999…``, cut 72 — so an authored
    ``p`` selects the cut this function says, which is what §2.6 documents."""
    return [units - int(p * units) for p in _cpr_grid(estimator)]


def _check_panel(
    obs_xs: Any, n_units: int, cuts: Sequence[float] | None, *, kind: str, what: str
) -> None:
    """A curve presumes a balanced panel: every unit has a row at every ``x``
    the curve reads — ``cpr``'s two anchors and grid cuts, ``auc``'s every
    distinct ``x``. A unit missing one would let the per-``x`` means (``B``,
    ``C`` and ``m(cut)`` under ``cpr``) average different populations, and a
    bootstrap draw over an ``auc`` panel with a hole could lose the ``x``
    outright and integrate a different grid, silently — MIB's evaluator
    averages whatever is there; a declared estimand refuses, naming the
    ``x``. Only under a ``unit`` other than ``row``: rows are not a panel.
    Balanced in *units*, not in rows: a second axis a ``group_by`` left out
    (a ``method`` absent at one cut) leaves every unit a row at every cut and
    still averages different populations — §2.6 marks that limit.

    Read off ``obs_xs``, the ``x`` of every ``(unit, x)`` observation
    `_curve_observations` collapsed the group to — so the panel checked
    is the one the curve is integrated from, decided once, and the check is
    one ``np.unique`` whatever the abscissa's cardinality (an ``auc`` reads
    every distinct ``x``, so a per-``x`` scan of the rows would be quadratic
    on a fine sweep). ``cuts`` is the ``x`` the curve reads — ``cpr``'s
    anchors and grid cuts — or ``None`` for every distinct ``x`` seen, which
    is what ``auc`` reads and what the collapse already holds."""
    import numpy as np

    seen, present = np.unique(obs_xs, return_counts=True)
    at = dict(zip(seen.tolist(), present.tolist()))
    # a cut *no* unit holds is the table's shape, not a ragged panel: leave
    # it to `_curve_value`, whose anchor / missing-cut / integer refusals
    # name it precisely
    holes = {
        cut: n_units - at[cut]
        for cut in (dict.fromkeys(cuts) if cuts is not None else at)
        if 0 < at.get(cut, 0) < n_units
    }
    if holes:
        # `cpr` reads integer cuts; `auc` any x — say which the table lacks,
        # the first five, each exactly as the table holds it (an integral x
        # without a decimal point at any magnitude, a fractional one by its
        # repr; the `isfinite` guard is defensive — an infinite x is refused
        # on the table and a NaN one never reaches a collapse)
        word = "cut" if kind == "cpr" else "x"
        ordered = sorted(holes.items())
        listed = _listed(
            [
                (
                    f"{cut:.0f}"
                    if math.isfinite(cut) and float(cut).is_integer()
                    else repr(float(cut))
                )
                + f" ({n} unit(s))"
                for cut, n in ordered[:5]
            ],
            len(ordered),
        )
        raise StepError(
            f"{what}: {kind!r} reads a balanced panel — every unit at every {word} "
            f"it reads — but of {n_units} units some have no row at {word} {listed}; "
            f"a curve over different units at different {word}s is not one curve "
            "(drop the unit, or fill its rows)"
        )


def _check_clusters_hold_units(
    unit_ids: Any, cluster_ids: Any, n_units: int, *, what: str
) -> None:
    """A curve's draw is a draw of whole curves only if every unit's rows fall
    in one resample cluster: the panel (`_check_panel`) gives each unit
    every ``x`` the curve reads, and one cluster per unit keeps those rows
    together in a draw. Decided on the table, not on column names —
    containment of ``resample_unit.columns`` in ``unit.columns`` is sufficient
    for it and not necessary: a coarser unit declared by its own column
    (examples nested in templates, the two-level cluster bootstrap) passes,
    a finer one (a ``method`` inside an ``example``) or one the data splits
    (a null in a unit column) is refused, naming how many units span
    clusters. One ``np.unique`` over the two label arrays already in hand,
    compacted to one dense label as `_bootstrap` compacts its draws.
    ``resample_unit: row`` never reaches here — the parser refuses it under a
    declared unit."""
    import numpy as np

    unit_ids, cluster_ids = np.asarray(unit_ids), np.asarray(cluster_ids)
    n_clusters = int(cluster_ids.max()) + 1 if len(cluster_ids) else 1
    pairs = np.unique(unit_ids * n_clusters + cluster_ids)
    per_unit = np.bincount(pairs // n_clusters, minlength=n_units)
    split = int((per_unit > 1).sum())
    if split:
        raise StepError(
            f"{what}: uncertainty.resample_unit splits {split} of {n_units} "
            "unit(s) across resample clusters — a draw of clusters would draw part "
            "of a unit's curve and could integrate a grid the table does not have; "
            "resample whole units (resample_unit equal to unit) or coarser ones "
            "(a column each unit lies within)"
        )


def _group_seed(seed: int, key: Sequence[Any]) -> list[int]:
    """The generator entropy for one group: the declared seed plus a stable
    word derived from the group's coordinates, so a group's draws do not
    depend on which other groups the table happens to hold."""
    payload = json.dumps(list(key), sort_keys=True, default=str).encode()
    word = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return [int(seed), word]


#: The estimator over a subset of one group's rows: ``(rows, draw_of_row)``
#: → value, where ``draw_of_row`` (or ``None``) labels which bootstrap draw
#: each row came from, so a unit drawn twice is two observations.
_EstimateOf = Callable[[Any, Any], float]


def _bootstrap(
    estimate_of: _EstimateOf,
    cluster_rows: list[Any],
    *,
    repetitions: int,
    seed: int,
    key: Sequence[Any],
) -> tuple[float, float]:
    """Percentile bootstrap over resample units (§2.6). Each drawn cluster
    keeps its rows together and is a *fresh* observation-holder: a unit drawn
    twice is two observations, which ``draw * n_units + unit`` encodes."""
    import numpy as np

    rng = np.random.default_rng(_group_seed(seed, key))
    k = len(cluster_rows)
    if k == 0:
        return float("nan"), float("nan")
    sizes = np.array([len(rows) for rows in cluster_rows])
    stats = np.empty(repetitions)
    for rep in range(repetitions):
        draw = rng.integers(0, k, size=k)
        rows = np.concatenate([cluster_rows[c] for c in draw])
        try:
            stats[rep] = estimate_of(rows, np.repeat(np.arange(k), sizes[draw]))
        except StepError as err:
            # the table's own estimate passed; the refusal is the *draw's* —
            # the drawn resample units lack a point of the curve, or their
            # anchors coincide. Under a declared `unit` a draw is whole curves
            # (the panel is balanced, `row` is refused at load and a resample
            # unit that splits a unit is refused on the table), so a lost
            # point is `unit: row`'s
            raise StepError(
                f"{err} — inside bootstrap repetition {rep + 1} of {repetitions}, "
                "over the drawn resample units (the table's own estimate passed): "
                "a draw that lacks a point of the curve, or whose anchors coincide; "
                "under a declared unit a draw is whole curves and cannot lose a "
                "point"
            ) from err
    alpha = 1.0 - CONFIDENCE
    lower, upper = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _normal_approx(
    estimate_of: _EstimateOf, cluster_rows: list[Any], estimate: float
) -> tuple[float, float]:
    """``estimate ± z · sd(cluster estimates) / sqrt(k)`` with the sample
    standard deviation (``ddof=1``) over the ``k`` resample units. Fewer than
    two clusters → NaN bounds: a spread over one observation is undefined."""
    import numpy as np

    k = len(cluster_rows)
    if k < 2:
        return float("nan"), float("nan")
    per_cluster = np.empty(k)
    for index, rows in enumerate(cluster_rows):
        per_cluster[index] = estimate_of(rows, None)
    se = float(np.std(per_cluster, ddof=1)) / (k**0.5)
    return estimate - _Z_95 * se, estimate + _Z_95 * se


def _listed(items: Sequence[str], total: int) -> str:
    """The first five of a listing and how many more — the one idiom for the
    refusals that list what to go and fix. ``items`` is the first five, already
    formatted, and ``total`` the whole count, so a caller formats only what it
    shows (and cannot, by omitting the count, report five as all)."""
    return ", ".join(items[:5]) + (f" and {total - 5} more" if total > 5 else "")


def _ceiling(group: Any, estimator: Estimator, key: Sequence[Any], what: str) -> Any:
    """A curve estimator's x ceiling for one group: the authored number, or
    the one value its ``normalize`` column holds across the group's rows —
    two values is two curves, refused. ``None`` when nothing was authored or
    the group is empty."""
    import pandas as pd

    normalize = estimator.normalize
    if normalize is None or len(group) == 0:
        return None
    if not isinstance(normalize, str):
        return float(normalize)
    distinct = pd.unique(pd.to_numeric(group[normalize], errors="coerce").dropna())
    if len(distinct) == 0:
        raise StepError(
            f"{what}: reduction.estimator.normalize names column {normalize!r}, "
            f"which holds no number in group {[_plain(c) for c in key]} — the "
            "ceiling is missing"
        )
    if len(distinct) != 1:
        raise StepError(
            f"{what}: reduction.estimator.normalize names column {normalize!r}, "
            f"which holds {len(distinct)} distinct values in group "
            f"{[_plain(c) for c in key]} — one ceiling per curve"
        )
    value = float(distinct[0])
    # the same predicate the parser applies to an authored number — finite,
    # then the kind's own bound: a column is not a way around it. A table
    # causalab wrote holds no inf (`write_table` maps a non-finite float to
    # null), so an infinite ceiling is a foreign table's or a direct caller's;
    # unrefused, +inf divided every `auc` abscissa to 0 and published a zero
    # area, and `int(value)` below raised OverflowError under `cpr`
    if not math.isfinite(value):
        raise StepError(
            f"{what}: reduction.estimator.normalize names column {normalize!r}, "
            f"which holds {value!r} in group {[_plain(c) for c in key]} — the "
            "ceiling must be finite"
        )
    if estimator.kind == "cpr" and (value != int(value) or value < 2):
        raise StepError(
            f"{what}: 'cpr' reads the unit count N from column {normalize!r}, "
            f"which holds {value!r} — an integer ≥ 2 is needed"
        )
    if estimator.kind == "auc" and not value > 0.0:
        raise StepError(
            f"{what}: 'auc' divides x by column {normalize!r}, which holds "
            f"{value!r} — the ceiling must be positive (0 gives no abscissa, a "
            "negative one a negative area)"
        )
    return value


def _plain(value: Any) -> Any:
    """A group coordinate as a JSON scalar (numpy scalars → Python)."""
    if hasattr(value, "item"):
        return value.item()
    return value


def reduce_frame(
    df: Any, spec: Reduction, value_column: str, *, what: str
) -> list[dict[str, Any]]:
    """Run one declaration over a table: one output row per group (§2.6).

    ``what`` labels refusals (the step and table). Every column the
    declaration names must exist — refused with a [`StepError`][] naming
    the column and the dimension that named it; that check is data, so it
    lives here rather than at load."""
    import numpy as np
    import pandas as pd

    curve = spec.estimator.is_curve
    value_column = spec.y_column(value_column)
    present = set(map(str, df.columns))
    # the declaration's bindings first, so an authored `estimator.y` the
    # table lacks is refused naming that field rather than as "no column"
    for column, field in spec.columns_named.items():
        if column not in present:
            raise StepError(
                f"{what}: reduction.{field} names column {column!r}, which the "
                f"table does not have (has {sorted(present)})"
            )
    if value_column not in present:
        raise StepError(
            f"{what}: no column {value_column!r} to reduce (has {sorted(present)})"
        )
    if curve and spec.estimator.x == value_column:
        # the parse gate compares `x` against an authored `y`; the default `y`
        # is the step's value column, known only here
        raise StepError(
            f"{what}: reduction.estimator.x names {value_column!r}, the column "
            "being reduced — a curve of y against itself has no area to publish; "
            "name the abscissa column, or declare estimator.y"
        )
    if curve and spec.estimator.normalize == value_column:
        # the same blind spot for the ceiling column
        raise StepError(
            f"{what}: reduction.estimator.normalize names {value_column!r}, the "
            "column being reduced — the ceiling is a coordinate of the curve, not "
            "its ordinate; name the column that holds it, or declare estimator.y"
        )

    # the table's own identity (spec §2.6): rows in two units are refused
    # before anything is summed — the mismatch refusal, at the one site where
    # rows are combined
    try:
        source = table_record(df.to_dict(orient="records"), name=what)
    except EstimandError as err:
        raise StepError(str(err)) from err
    identity = spec.identity

    frame = df.copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    missing_mask = frame[value_column].isna()
    x_column = spec.estimator.x
    if curve:
        assert x_column is not None
        frame[x_column] = pd.to_numeric(frame[x_column], errors="coerce")
        missing_mask = missing_mask | frame[x_column].isna()
        # a null x is `missing`'s (refused or excluded below); an infinite one
        # is a value nothing else refuses — under `auc` the last trapezoid has
        # infinite width and ±inf/nan publishes, under `cpr` the integer check
        # cannot even format it — so it is the table's: refused once over the
        # whole table, before any group is read and before `missing` is
        # applied (a row with a null ordinate and an infinite abscissa is a
        # malformed table, not a missing observation). A table causalab wrote
        # holds no inf — `write_table` maps a non-finite float to null — so
        # this is a foreign table's or a direct caller's; the rows are named
        # by the frame's labels (positions, for a table the built-in read)
        table_x = frame[x_column].to_numpy(dtype=float, na_value=np.nan)
        infinite = np.flatnonzero(np.isinf(table_x))
        if len(infinite):
            rows = [
                f"row {frame.index[i]}: {float(table_x[i])!r}" for i in infinite[:5]
            ]
            raise StepError(
                f"{what}: x ({x_column!r}) is infinite in {len(infinite)} row(s) "
                f"({_listed(rows, len(infinite))}) — a curve is integrated over "
                "finite x; drop the row upstream or use a finite cut (`missing` "
                "governs a null value, not an infinite x)"
            )
    if spec.weight is not None:
        frame[spec.weight] = pd.to_numeric(frame[spec.weight], errors="coerce")
        missing_mask = missing_mask | frame[spec.weight].isna()
    if MATCHED_COLUMN in frame.columns:
        unmatched_mask = missing_mask & frame[MATCHED_COLUMN].eq(False)
    else:
        unmatched_mask = missing_mask & False
    frame["__missing"] = missing_mask
    frame["__unmatched"] = unmatched_mask

    if spec.missing == "error" and bool(missing_mask.any()):
        raise StepError(
            f"{what}: {int(missing_mask.sum())} row(s) hold a null value"
            + (" or x" if curve else "")
            + (" or weight" if spec.weight is not None else "")
            + f" ({int(unmatched_mask.sum())} from matched=false) and "
            "reduction.missing is 'error' — declare 'exclude' or 'zero' to "
            "reduce without them"
        )

    if spec.group_by:
        groups = frame.groupby(list(spec.group_by), sort=True, dropna=False)
        items = [(key if isinstance(key, tuple) else (key,), g) for key, g in groups]
    else:
        items = [((), frame)]

    out: list[dict[str, Any]] = []
    for key, group in items:
        n_missing = int(group["__missing"].sum())
        n_unmatched = int(group["__unmatched"].sum())
        if spec.missing == "exclude":
            group = group[~group["__missing"]]
            n_excluded = n_missing
        else:
            n_excluded = 0
        values = group[value_column].to_numpy(dtype=float)
        weights = None
        if spec.weight is not None:
            weights = group[spec.weight].to_numpy(dtype=float)
        xs = None
        ceiling = None
        if curve:
            xs = group[x_column].to_numpy(dtype=float)
            if spec.missing == "zero" and bool(np.isnan(xs).any()):
                raise StepError(
                    f"{what}: {int(np.isnan(xs).sum())} row(s) hold a null x "
                    f"({x_column!r}) — an abscissa has no zero; declare "
                    "reduction.missing 'exclude' or 'error'"
                )
            ceiling = _ceiling(group, spec.estimator, key, what)
        if spec.missing == "zero":
            values = np.nan_to_num(values, nan=0.0)
            if weights is not None:
                weights = np.nan_to_num(weights, nan=0.0)

        unit_ids = None
        if spec.unit.kind != "row":
            unit_ids = _factorize(group, spec.unit.columns)
        n_units = (
            int(unit_ids.max()) + 1 if unit_ids is not None and len(unit_ids) else 1
        )

        # the whole group's observations, collapsed once: the panel check,
        # the point estimate and `n` all read them
        if curve:
            collapsed = _curve_observations(xs, values, unit_ids)
            if unit_ids is not None:
                # the x the curve reads: `cpr` its anchors and grid cuts, `auc`
                # every distinct x (no null x reaches here — `missing_mask`
                # covers the abscissa — and an infinite one was refused above)
                read_xs: list[float] | None
                if spec.estimator.kind == "cpr":
                    read_xs = (
                        [0, int(ceiling), *_cpr_cuts(spec.estimator, int(ceiling))]
                        if ceiling is not None
                        else []
                    )
                else:
                    # every distinct x seen: `_check_panel` reads them off
                    # the collapse rather than a second `np.unique` here
                    read_xs = None
                if read_xs is None or read_xs:
                    _check_panel(
                        collapsed[0],
                        n_units,
                        read_xs,
                        kind=spec.estimator.kind,
                        what=what,
                    )
        else:
            collapsed = _observations(values, weights, unit_ids, spec.estimator)
        obs = collapsed[0]

        def estimate_of(rows: Any, draw_of_row: Any) -> float:
            """The estimator over ``rows`` of this group (all of them when
            ``rows`` is ``None``); the resamplers call it per draw."""
            if rows is None:
                if curve:
                    return _curve_value(*collapsed, spec.estimator, ceiling, what=what)
                return _estimate(*collapsed, spec.estimator)
            ids = None
            if unit_ids is not None:
                ids = unit_ids[rows]
                if draw_of_row is not None:
                    ids = draw_of_row * n_units + ids
                # compact to a dense label space so bincount stays small
                _, ids = np.unique(ids, return_inverse=True)
                ids = np.asarray(ids).reshape(-1)
            v = values[rows]
            w = None if weights is None else weights[rows]
            x = None if xs is None else xs[rows]
            if curve:
                obs_x, obs_v = _curve_observations(x, v, ids)
                return _curve_value(obs_x, obs_v, spec.estimator, ceiling, what=what)
            obs_v, obs_w = _observations(v, w, ids, spec.estimator)
            return _estimate(obs_v, obs_w, spec.estimator)

        estimate = estimate_of(None, None)

        row: dict[str, Any] = {
            column: _plain(coord) for column, coord in zip(spec.group_by, key)
        }
        row["value"] = int(estimate) if spec.estimator.kind == "count" else estimate
        row["n"] = int(len(obs))
        row["n_rows"] = int(len(values))
        row["n_missing"] = n_missing
        row["n_unmatched"] = n_unmatched
        row["n_excluded"] = n_excluded
        row["unit"] = reduction_unit(spec.estimator.kind, source.unit)
        row["estimand_version"] = identity

        procedure = spec.uncertainty
        if procedure.kind != "none":
            assert procedure.resample_unit is not None
            if procedure.resample_unit.kind == "row":
                cluster_rows = [np.array([i]) for i in range(len(values))]
            else:
                cluster_ids = _factorize(group, procedure.resample_unit.columns)
                if curve and unit_ids is not None:
                    # a draw is whole curves only if each unit lies in one
                    # cluster — the table decides, not the column names
                    _check_clusters_hold_units(
                        unit_ids, cluster_ids, n_units, what=what
                    )
                cluster_rows = [
                    np.flatnonzero(cluster_ids == c)
                    for c in range(
                        int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
                    )
                ]
            if procedure.kind == "percentile_bootstrap":
                lower, upper = _bootstrap(
                    estimate_of,
                    cluster_rows,
                    repetitions=int(procedure.repetitions or 0),
                    seed=int(procedure.seed or 0),
                    key=[_plain(c) for c in key],
                )
            else:
                lower, upper = _normal_approx(estimate_of, cluster_rows, estimate)
            row["lower"] = lower
            row["upper"] = upper
        out.append(row)
    return out


def _factorize(group: Any, columns: Sequence[str]) -> Any:
    """Dense int labels for the distinct values of ``columns`` within a group,
    in first-appearance order. A null key is its own label rather than dropped:
    a row whose unit is unknown is still a row."""
    import pandas as pd

    if len(group) == 0:
        import numpy as np

        return np.zeros(0, dtype=int)
    if len(columns) == 1:
        codes, _ = pd.factorize(group[columns[0]], use_na_sentinel=False)
        return codes
    index = pd.MultiIndex.from_frame(group[list(columns)])
    codes, _ = pd.factorize(index, use_na_sentinel=False)
    return codes
