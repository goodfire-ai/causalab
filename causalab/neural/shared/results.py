"""Assemble metric rows and tensors produced by protocol points.

Tensor reads and fitted bundles use safetensors. Per-example metrics use
JSON row arrays. Sweeps retain the authored file path and add coordinates
to metric rows or tensor keys, such as ``rot[k=8,seed=0]``.

A tensor file stores shared identity fields at file level and varying
fields in an ``entries`` metadata table keyed by tensor name. This supports
selection and provenance checks for each entry. ``TensorFile`` stays in the
engine layer because it handles ragged values and save reductions.
``causalab.io.results_io`` writes the assembled outputs.
"""

from __future__ import annotations

import dataclasses
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from causalab.neural.shared.values import RaggedValue
from causalab.protocol.bundles import RAGGED_SUFFIX, entry_key
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.estimand import IDENTITY_COLUMNS
from causalab.protocol.lowering import coordinate_label, short_coords
from causalab.protocol.results import (
    EXAMPLE_ID_COLUMN,
    Eligibility,
    Resolution,
    Unavailable,
    available,
    unavailable,
)

if TYPE_CHECKING:
    from causalab.neural.shared.execution import ExecutorSurface

__all__ = [
    "ELIGIBLE_COLUMN",
    "MASK_DECISIVE_MARGIN",
    "MetricTable",
    "REASON_CODE_COLUMN",
    "TensorFile",
    "rank_records",
]


#: The eligibility record on every metric row (spec §2.10 "Eligibility"):
#: ``eligible`` is ``true`` on a row the metric's decision rule was evaluated
#: over and ``false`` on an excluded measurement, which alone also carries
#: ``reason_code`` — the [`ReasonCode`][causalab.protocol.rules.errors.ReasonCode] of the
#: ``unavailable`` the row became. Both derived, never authored (§6).
ELIGIBLE_COLUMN = "eligible"
REASON_CODE_COLUMN = "reason_code"


class TensorFile:
    """Accumulates tensor entries for one save file across points."""

    def __init__(self) -> None:
        self.entries: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, str] = {}
        #: key -> {"slot", "coords", …identity}; serialized as ``entries``
        self.entry_meta: dict[str, dict[str, Any]] = {}
        self._common_seen = False

    def add(
        self,
        name: str,
        value: Any,
        coords: Mapping[str, Any],
        *,
        label_entry: str | None = None,
        reduce: str | None = None,
        identity: Mapping[str, Any] | None = None,
        record: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one point's value under ``name``.

        ``label_entry`` is the *declared* entity the coordinates belong to,
        which for a featurizer bundle is the featurizer, not the slot: the
        axis ``featurizers.rot.k`` shortens to ``k`` against ``rot`` and to
        ``rot.k`` against ``weight``, and only the former is a name a
        consuming document can write in an ``entry`` selector.

        ``identity`` fields are stamped as strings (the ArtifactIdentity
        contract); ``record`` fields ride on the entry **as JSON values** —
        what a ``trajectory`` checkpoint says about itself (its step, the
        controlled weight, the kept count), which a reader wants as numbers."""
        entity = label_entry or name
        key = entry_key(name, coordinate_label(coords, entry=entity) if coords else "")
        self.entry_meta[key] = {
            "slot": name,
            "coords": {
                short: _plain(coord)
                for short, coord in short_coords(coords, entry=entity).items()
            },
            **{k: str(v) for k, v in (identity or {}).items()},
            **{k: _plain(v) for k, v in (record or {}).items()},
        }
        if reduce is not None:
            self.entries[key] = _reduce_rows(value, reduce)
            return
        if isinstance(value, RaggedValue):
            # ragged reads persist as the flat gather + per-row widths
            self.entries[key] = value.flat.detach().to("cpu").contiguous()
            self.entries[f"{key}{RAGGED_SUFFIX}"] = torch.tensor(
                value.widths, dtype=torch.long
            )
            return
        self.entries[key] = value.detach().to("cpu").contiguous()

    def record_common(self, identity: Mapping[str, Any]) -> None:
        """Fold one point's identity into the file-level stamp, keeping only
        the fields every point so far agrees on.

        A single-point document therefore stamps exactly what it always did;
        a swept one drops the fields that differ (``k``, a swept site)
        rather than letting the last point speak for the file. The dropped
        fields are still provable per entry via the ``entries`` table."""
        stamped = {key: str(value) for key, value in identity.items()}
        if not self._common_seen:
            self.metadata.update(stamped)
            self._common_seen = True
            return
        for key in list(self.metadata):
            if self.metadata[key] != stamped.get(key):
                del self.metadata[key]

    def extend(self, other: TensorFile) -> None:
        """Append ``other``'s entries after this file's, in ``other``'s
        order, and fold its file-level stamp in as [`record_common`][]
        folds a point's: the result is what one file accumulates over the
        points of both, in that order — the data-parallel join's fold
        (``neural/shared/join.py``)."""
        self.entries.update(other.entries)
        self.entry_meta.update(other.entry_meta)
        if not other._common_seen:
            return
        if not self._common_seen:
            self.metadata.update(other.metadata)
            self._common_seen = True
            return
        for key in list(self.metadata):
            if self.metadata[key] != other.metadata.get(key):
                del self.metadata[key]


def _reduce_rows(value: Any, reduce: str) -> torch.Tensor:
    """§2.12 ``reduce``: a statistic over a read's gathered rows instead of
    the rows themselves — ``(…, width)`` collapses to ``(width,)``, the
    broadcast form a write operand takes.

    One branch per verb in [`SAVE_REDUCTIONS`][causalab.protocol.schema.parse.SAVE_REDUCTIONS],
    and the vocabulary is closed: a new verb is a PR that adds a branch here,
    a §2.12 row, and a test. The docs↔code guard
    (``tests/protocol/test_vocabulary_census.py``) fails if the two drift.

    Reducing here rather than downstream is the point: the un-reduced
    harvest never reaches disk, which for an ablation grid is the difference
    between gigabytes of activations and kilobytes of means. The
    accumulation is fp32 regardless of the run's dtype — a bf16 sum over
    thousands of rows loses the low bits it is meant to average.
    """
    rows = value.flat if isinstance(value, RaggedValue) else value
    flat = (
        rows.detach().to(device="cpu", dtype=torch.float32).reshape(-1, rows.shape[-1])
    )
    if flat.shape[0] == 0 and reduce in ("mean", "std", "median"):
        # an unavailable cell (a scoped slice that selected no rows, spec
        # §4.1): the statistic of no observations is undefined, and NaN is the
        # honest `(width,)` answer — torch already says so for `mean` and
        # `std`, but `median` raises on an empty axis. `sum` (0) and `count`
        # (0) fall through: both are right, and together they compose.
        return torch.full((flat.shape[-1],), float("nan"), dtype=torch.float32)
    if reduce == "mean":
        out = flat.mean(dim=0)
    elif reduce == "sum":
        # the numerator half of a weighted mean across points or shards: a
        # mean of means is wrong whenever the point row counts differ
        out = flat.sum(dim=0)
    elif reduce == "std":
        # the *sample* standard deviation (torch's default correction=1): the
        # rows are a sample of examples drawn from a table, not the population.
        # One row therefore gives NaN, which is the honest answer — the spread
        # of a single observation is undefined, and a 0.0 would read as "no
        # variation".
        out = flat.std(dim=0)
    elif reduce == "median":
        # the lower of the two middle values at even row counts, which is what
        # torch.median does; no interpolation, so the saved value is one that
        # a row actually held
        out = flat.median(dim=0).values
    elif reduce == "count":
        # how many rows were reduced, as a width-vector so every reduction has
        # the one shape §2.12 promises. It is the denominator that makes `sum`
        # composable across points, and it records a truncated or ragged
        # harvest that a `mean` alone would hide.
        out = torch.full((flat.shape[-1],), float(flat.shape[0]), dtype=torch.float32)
    else:
        raise ProtocolError("P2", f"unknown save reduction {reduce!r}")
    return out.contiguous()


class MetricTable:
    """Accumulates per-example metric rows for one save file across points.

    A value may be an [`Unavailable`][] —
    the row is a structurally unobservable measurement (its address aligned
    on nothing, its answer column is empty; spec §4.1) — and is then written
    with a ``null`` value, ``eligible: false`` and its ``reason_code``, so an
    excluded row and a row that scored ``null`` for another reason never look
    alike after a group-by (§2.10 "Eligibility")."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        values: list[Any],
        coords: Mapping[str, Any],
        *,
        identity: Mapping[str, Any],
        labels: Sequence[str] | None = None,
    ) -> None:
        """One row per value. ``labels`` are the rows' ``example_id``s
        (``protocol/examples.py``, the base role's); ``None`` labels each row
        by its index, which is what a table without the column resolves to."""
        for label, value in zip(_labels(labels, len(values)), values):
            self.rows.append(self._row(name, label, value, coords, identity=identity))

    def add_windowed(
        self,
        name: str,
        values: list[list[Any]],
        coords: Mapping[str, Any],
        *,
        identity: Mapping[str, Any],
        steps: list[list[int]] | None,
        matched: list[bool],
        labels: Sequence[str] | None = None,
    ) -> None:
        """Rows for a metric over a read that addresses several positions.

        One row per (example, position), carrying the ``step`` it scored and
        whether the example addressed anything at all. An example that
        addressed **nothing** — a row that stopped generating, or never said
        the value a ``variable`` anchor looks for — still gets exactly one
        row, with a null value and ``matched=false``: "the model never said
        it" has to be distinguishable from "it said it and scored 0", and a
        missing row would make the two look identical after a group-by.

        ``steps`` is ``None`` for a kind that reduces the whole window to
        one value (``decode``): there is no single step such a value belongs
        to, so the column stays null rather than lying about one.
        """
        row_labels = _labels(labels, len(values))
        for example, row_values in enumerate(values):
            if not row_values:
                self.rows.append(
                    self._row(
                        name,
                        row_labels[example],
                        None,
                        coords,
                        identity=identity,
                        step=None,
                        matched=matched[example],
                    )
                )
                continue
            for offset, value in enumerate(row_values):
                self.rows.append(
                    self._row(
                        name,
                        row_labels[example],
                        value,
                        coords,
                        identity=identity,
                        step=steps[example][offset] if steps is not None else None,
                        matched=matched[example],
                    )
                )

    def _row(
        self,
        name: str,
        label: str,
        value: Any,
        coords: Mapping[str, Any],
        *,
        identity: Mapping[str, Any],
        step: int | None = None,
        matched: bool | None = None,
    ) -> dict[str, Any]:
        """One metric row: ``{example_id, metric, value, [step, matched],
        …coords, unit, estimand_version, eligible, [reason_code]}``.
        ``example_id`` is the base row's label (spec §2.2,
        ``protocol/examples.py``): the author's, or the row index as a string
        for a table without the column. The coordinate columns (one per axis
        of the sweep, keyed by the full axis id) are what places the row on
        its point — a consumer joins them to the receipt's ``points[].coords``
        or to the parent expansion (workflow spec §2.9); the row carries no
        digest of its own. ``identity`` is the record's ``unit`` /
        ``estimand_version`` (spec §2.10) — authored on the metric or derived
        from its kind (``estimand.metric_record_identity``) — repeated on
        every row, because a table has no envelope to carry it once
        (``tables.py``). ``null`` for a kind with no scalar value.

        ``eligible`` is the row's eligibility record (§2.10 "Eligibility"):
        ``false`` — with the ``reason_code`` of the ``Unavailable`` the value
        is, and a ``null`` value — for an excluded measurement; ``false`` too,
        under ``alignment_missing``, for a continuation row that addressed
        nothing (``matched: false`` — the anchor's value occurred nowhere in
        what the row generated); ``true`` otherwise, with no ``reason_code``
        column, as an available cell records nothing (§4.1)."""
        row: dict[str, Any] = {EXAMPLE_ID_COLUMN: label, "metric": name}
        excluded: Unavailable | None = value if isinstance(value, Unavailable) else None
        if excluded is not None:
            row["value"] = None
        elif isinstance(value, dict):
            import json

            row["value"] = json.dumps(value, sort_keys=True)
        else:
            row["value"] = value
        if matched is not None:
            row["step"] = step
            row["matched"] = matched
        row.update({axis: _plain(coord) for axis, coord in coords.items()})
        row.update({column: identity[column] for column in IDENTITY_COLUMNS})
        if excluded is not None:
            row[ELIGIBLE_COLUMN] = False
            row[REASON_CODE_COLUMN] = excluded.reason
        elif matched is False:
            row[ELIGIBLE_COLUMN] = False
            row[REASON_CODE_COLUMN] = "alignment_missing"
        else:
            row[ELIGIBLE_COLUMN] = True
        return row


def _labels(labels: Sequence[str] | None, count: int) -> list[str]:
    if labels is None:
        return [str(index) for index in range(count)]
    if len(labels) != count:
        raise ValueError(f"{len(labels)} labels for {count} rows")
    return list(labels)


def _plain(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    import json

    return json.dumps(value, sort_keys=True)


# --------------------------------------------------------------------------- #
# the assembly of one point's results (formerly the tail of execution.py)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _Windowed:
    """One continuation metric's per-example results, plus what the rows need
    to stay legible: the steps each value scored, and whether the example
    addressed anything at all."""

    values: list[list[Any]]
    steps: list[list[int]] | None
    matched: list[bool]


#: A soft mask is "decisive" at a dimension when σ(θ) is outside
#: ``[0.5 - MASK_DECISIVE_MARGIN, 0.5 + MASK_DECISIVE_MARGIN]`` — i.e. outside
#: [0.1, 0.9] at the default. Chosen to match the measure the DBM preset's
#: description uses (``demos/methods/protocols/dbm.json``); its committed run
#: (``demos/methods/results/protocols/dbm.json``) records no dimension that
#: cleared it.
MASK_DECISIVE_MARGIN = 0.4


def rank_records(
    stages: Mapping[str, Any], point_digest: str, coords: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The rows a ``rank`` save (§2.12) carries for one point: one per unit of
    every gate the point built — trained or loaded — with the unit's ``theta``,
    its position in the gate's ranking (``0`` = kept first), whether the
    eval-mode split keeps it, the map that split is read through, and the
    ``top_k`` the cut was made at (``null`` under the map's own threshold).
    Units are flat indices into ``theta``, so on a grouped gate a row is a head
    or an ``(expert, neuron)`` entry, never a coordinate; on a position gate
    (§2.5 ``axis``) a row is an addressed token position, and ``axis``
    records which (``null`` for a feature gate). A gate in a budget
    pool (§2.5 ``pool``) also carries the pool's name and the unit's position in
    the **pooled** ranking (``pool_rank``), the order the pooled cut is made in;
    both are ``null`` otherwise. The point's provenance and coordinates ride
    along as on every table."""
    import torch

    from causalab.neural.shared.featurizers import Gate

    rows: list[dict[str, Any]] = []
    for name in sorted(stages):
        stage = stages[name]
        if not isinstance(stage, Gate):
            continue
        with torch.no_grad():
            theta = stage.theta.detach().float().view(-1).tolist()
            rank = stage.rank().view(-1).tolist()
            hard = stage.hard_mask().view(-1).tolist()
            pooled = (
                stage.pool.member_rank(stage).view(-1).tolist()
                if stage.pool is not None
                else [None] * len(theta)
            )
        for unit, (value, position, kept, pool_rank) in enumerate(
            zip(theta, rank, hard, pooled)
        ):
            rows.append(
                {
                    "featurizer": name,
                    "unit": unit,
                    "theta": value,
                    "rank": int(position),
                    "hard": bool(kept),
                    "parametrization": stage.parametrization,
                    # §2.5 axis: which axis `unit` indexes — as `pool` says what
                    # `pool_rank` ranks against
                    "axis": stage.axis,
                    "top_k": stage.top_k,
                    "pool": stage.pool.name if stage.pool is not None else None,
                    "pool_rank": int(pool_rank) if pool_rank is not None else None,
                    "point": point_digest,
                    "coords": dict(coords),
                }
            )
    return rows


def _summary_stat(values: list[Any]) -> Any:
    """The aggregate: the mean over the **eligible** numeric rows. An excluded
    row is an ``Unavailable``, not a number, so it is never in the
    denominator here (§2.10 "Eligibility")."""
    numeric = [v for v in values if isinstance(v, (int, float))]
    if numeric:
        return sum(numeric) / len(numeric)
    return f"{len(values)} rows"


def metric_label(entry: Any) -> str:
    """The label a saved reduction goes by: the ``metric`` column of its
    table, its key in the point summary and the name on its ``metric`` event
    — the save entry's ``file_path`` stem (``iia.json`` → ``iia``)."""
    return PurePosixPath(str(entry.file_path)).stem


def _row_exclusions(
    executor: ExecutorSurface,
    qname: str,
    of_name: Any,
    target_name: Any | None,
    key: str,
) -> list[Unavailable | None]:
    """Per base row, the ``unavailable`` a metric's row is when the read it
    reduces (or, for ``kl``, the read it compares against) aligned on nothing
    for that row (§4.1) — re-keyed under the metric's own cell and saying
    which read — else ``None``. The row-level form of "a metric over an
    unavailable read inherits the cell"."""
    per_row = list(executor.row_resolutions(of_name))
    if target_name is not None:
        per_row = [
            of_cell or target_cell
            for of_cell, target_cell in zip(
                per_row, executor.row_resolutions(target_name), strict=True
            )
        ]
    return [
        None
        if cell is None
        else unavailable(
            cell.reason,
            f"metric {qname!r} reduces a read unavailable on this row: {cell.detail}",
            key,
        )
        for cell in per_row
    ]


def _windowed_eligibility(window: _Windowed) -> Eligibility:
    """A continuation metric's eligibility, per example: a row that addressed
    nothing (``matched: false`` — the anchor's value occurred nowhere in what
    it generated) is excluded under ``alignment_missing``; a row whose every
    position scored is eligible; a row with an excluded position is counted
    under that position's reason."""
    per_example: list[Any] = []
    for values, matched in zip(window.values, window.matched, strict=True):
        if not matched:
            per_example.append(
                unavailable("alignment_missing", "the row addressed nothing", "")
            )
            continue
        excluded = next((v for v in values if isinstance(v, Unavailable)), None)
        per_example.append(excluded if excluded is not None else values)
    return Eligibility.of(per_example)


def _metric_cell(
    name: str,
    file_path: str,
    counts: Eligibility,
    values: list[Any],
    key: str,
) -> Resolution:
    """The result cell of one metric at one point (§4.1): available — with
    its ``n_eligible`` / ``n_considered`` in the mapping — when at least one
    row was eligible, else the ``unavailable`` its rows all are, under the
    first excluded row's reason. A metric over zero rows is available: nothing
    was excluded."""
    first = next((v for v in values if isinstance(v, Unavailable)), None)
    if counts.n_eligible == 0 and first is not None:
        return unavailable(
            first.reason,
            f"metric {name!r}: all {counts.n_considered} rows are excluded "
            f"measurements — {first.detail}",
            key,
        )
    return available(
        {"file_path": file_path, "metric": name, **counts.as_record()}, key
    )
