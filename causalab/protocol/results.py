"""Represent availability, eligibility, and row labels.

``Available`` carries a value. ``Unavailable`` records a supported measurement
that cannot be made for the current cell. ``Invalid`` describes a validation
error. Denominators report eligible and total cells with reasons for exclusions.
``Eligibility`` supplies the corresponding row counts for a metric.

Example labels identify the original row in saved results."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Sequence

from causalab.protocol.bundles import entry_key
from causalab.protocol.rules.errors import REASON_CODES, ReasonCode
from causalab.protocol.lowering import coordinate_label

__all__ = [
    "STATUS_KEY",
    "UNAVAILABLE",
    "Available",
    "Denominator",
    "Eligibility",
    "Invalid",
    "Resolution",
    "Unavailable",
    "available",
    "cell_key",
    "cell_record",
    "invalid",
    "unavailable",
    "EXAMPLE_ID_COLUMN",
    "example_labels",
    "example_id_defect",
]

#: The result-cell field that marks an unavailable cell. An available cell
#: records **nothing** — no ``status: "available"`` — so every result written
#: before this value existed is byte-identical to one written after.
STATUS_KEY = "status"
#: The one value [`STATUS_KEY`][] takes.
UNAVAILABLE = "unavailable"


@dataclasses.dataclass(frozen=True)
class Available:
    """A resolution that produced something: ``mapping`` names what (the
    saved key, the rows, the address — whatever the resolving site knows)."""

    mapping: Mapping[str, Any]
    denominator_key: str


@dataclasses.dataclass(frozen=True)
class Unavailable:
    """A legal cell with nothing to measure, and why.

    ``reason`` is a [`ReasonCode`][]; ``detail`` is
    the fact in prose (which expert, at which positions); ``denominator_key``
    is what the aggregator counts this cell under ([`cell_key`][]).
    """

    reason: ReasonCode
    detail: str
    denominator_key: str

    def __post_init__(self) -> None:
        if self.reason not in REASON_CODES:
            raise AssertionError(
                f"unknown reason code {self.reason!r}; expected one of {REASON_CODES}"
            )

    def record(self) -> dict[str, Any]:
        """The four result-cell fields (spec §4.1)."""
        return {
            STATUS_KEY: UNAVAILABLE,
            "reason": self.reason,
            "detail": self.detail,
            "denominator_key": self.denominator_key,
        }


@dataclasses.dataclass(frozen=True)
class Invalid:
    """A document-decidable defect: what a validator raises from.

    ``error_code`` is the existing rule or parse code (``V15``, ``P4``) — the
    triple adds no rule numbers. An ``Invalid`` never enters a result; it is
    not a member of [`Resolution`][], and [`cell_record`][] refuses it.
    """

    error_code: str
    detail: str


#: The value-carrying pair. ``Invalid`` is deliberately not in it.
Resolution = Available | Unavailable


def available(mapping: Mapping[str, Any], denominator_key: str) -> Available:
    return Available(mapping=dict(mapping), denominator_key=denominator_key)


def unavailable(reason: ReasonCode, detail: str, denominator_key: str) -> Unavailable:
    return Unavailable(reason=reason, detail=detail, denominator_key=denominator_key)


def invalid(error_code: str, detail: str) -> Invalid:
    return Invalid(error_code=error_code, detail=detail)


def cell_key(value: str, coords: Mapping[str, Any]) -> str:
    """The denominator key of one saved value at one point.

    The value's name plus the point's coordinate label — exactly the key the
    value's tensor entry takes in a saved bundle (``entry_key``), so a
    reader can go from the ``cells`` line to the entry it names.
    """
    return entry_key(value, coordinate_label(coords, entry=value))


def cell_record(cell: Resolution | Invalid) -> dict[str, Any]:
    """What a result cell records about its resolution: nothing when
    available, the four fields when unavailable. Refuses an ``Invalid`` —
    a defect stops validation and never becomes a cell — which is why the
    signature admits one: so the refusal is the contract, not a type hint."""
    if isinstance(cell, Available):
        return {}
    if isinstance(cell, Unavailable):
        return cell.record()
    raise TypeError(
        f"{type(cell).__name__} is not a result value: `invalid` stops "
        "validation and never enters a result"
    )


@dataclasses.dataclass(frozen=True)
class Denominator:
    """``eligible`` of ``total`` cells, with the excluded ones by reason.

    The numbers a summary reads instead of keeping its own books: an
    aggregator that means over ``eligible`` cells and names the excluded
    ones is honest about both.
    """

    total: int
    eligible: int
    #: reason → the excluded cells' denominator keys, sorted
    unavailable: Mapping[ReasonCode, tuple[str, ...]]

    @classmethod
    def of(cls, cells: Iterable[Resolution]) -> "Denominator":
        total = eligible = 0
        excluded: dict[ReasonCode, list[str]] = {}
        for cell in cells:
            cell_record(cell)  # refuses an Invalid before it is counted
            total += 1
            if isinstance(cell, Available):
                eligible += 1
            else:
                excluded.setdefault(cell.reason, []).append(cell.denominator_key)
        return cls(
            total=total,
            eligible=eligible,
            unavailable={
                reason: tuple(sorted(keys)) for reason, keys in sorted(excluded.items())
            },
        )

    @property
    def excluded(self) -> int:
        return self.total - self.eligible

    def as_record(self) -> dict[str, Any]:
        """The denominator as numbers: ``eligible``, ``total``, and per reason
        the count and the excluded keys."""
        return {
            "eligible": self.eligible,
            "total": self.total,
            "unavailable": {
                reason: {"count": len(keys), "cells": list(keys)}
                for reason, keys in self.unavailable.items()
            },
        }

    def render(self) -> str:
        """One line: ``155 / 157 eligible; 2 excluded: empty_selector ×2``."""
        head = f"{self.eligible} / {self.total} eligible"
        if not self.excluded:
            return head
        by_reason = ", ".join(
            f"{reason} ×{len(keys)}" for reason, keys in self.unavailable.items()
        )
        return f"{head}; {self.excluded} excluded: {by_reason}"


@dataclasses.dataclass(frozen=True)
class Eligibility:
    """How many of a metric cell's rows its decision rule was evaluated over
    (spec §2.10 "Eligibility"): ``n_eligible`` of ``n_considered``, with the
    excluded rows counted by reason.

    The row-level twin of [`Denominator`][], and a **third** denominator
    named apart from the other two on purpose: ``n_eligible`` counts the rows
    a metric's *decision rule* saw; ``save.reduce: "count"`` (§2.12) counts
    the rows a saved read's reduction collapsed; a workflow reduction's
    ``unit`` (workflow §2.6) is the statistical unit a table is later reduced
    over. Derived from the rows, never authored (§6): ``of`` reads a metric's
    per-example values, where an excluded row is an [`Unavailable`][].
    """

    n_eligible: int
    n_considered: int
    #: reason → how many rows it excluded, sorted by reason
    excluded: Mapping[ReasonCode, int]

    @classmethod
    def of(cls, values: Iterable[Any]) -> "Eligibility":
        considered = eligible = 0
        by_reason: dict[ReasonCode, int] = {}
        for value in values:
            considered += 1
            if isinstance(value, Unavailable):
                by_reason[value.reason] = by_reason.get(value.reason, 0) + 1
            else:
                eligible += 1
        return cls(
            n_eligible=eligible,
            n_considered=considered,
            excluded=dict(sorted(by_reason.items())),
        )

    def as_record(self) -> dict[str, Any]:
        """The aggregate cell's two counts, plus ``excluded`` by reason only
        when a row was excluded — a cell whose every row was eligible records
        exactly the two numbers."""
        record: dict[str, Any] = {
            "n_eligible": self.n_eligible,
            "n_considered": self.n_considered,
        }
        if self.excluded:
            record["excluded"] = dict(self.excluded)
        return record


# --------------------------------------------------------------------------- #
# example labels (formerly protocol/examples.py)
# --------------------------------------------------------------------------- #

#: The optional dataset column, and the column every per-example table writes
#: (spec §2.2). A dataset may carry an ``example_id`` column: the author's
#: name for the row. When it does, every row must carry one, non-empty and
#: unique within the table. When it does not, a row's label is its zero-based
#: index as a string. Either way one label names one example, and the base
#: row's label names the pair — rows are paired by index and the base role is
#: never permuted, so the label a metric row carries is the base row's.
EXAMPLE_ID_COLUMN = "example_id"


def example_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """One label per row: its ``example_id`` as a string, or its index.

    Assumes [`example_id_defect`][] returned ``None`` for ``rows``.
    """
    if not any(EXAMPLE_ID_COLUMN in row for row in rows):
        return [str(index) for index in range(len(rows))]
    return [str(row[EXAMPLE_ID_COLUMN]) for row in rows]


def example_id_defect(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the table's ``example_id`` column cannot label its rows, or ``None``.

    A table without the column has no defect. One that has it must carry it
    on every row, non-empty, and no two rows may share a label.
    """
    carrying = [index for index, row in enumerate(rows) if EXAMPLE_ID_COLUMN in row]
    if not carrying:
        return None
    if len(carrying) != len(rows):
        missing = [i for i in range(len(rows)) if i not in set(carrying)]
        return (
            f"rows {missing[:3]}{'…' if len(missing) > 3 else ''} carry no "
            f"{EXAMPLE_ID_COLUMN} while other rows do"
        )
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        value = row[EXAMPLE_ID_COLUMN]
        label = "" if value is None else str(value)
        if not label.strip():
            return f"row {index} has an empty {EXAMPLE_ID_COLUMN}"
        if label in seen:
            return (
                f"{EXAMPLE_ID_COLUMN} {label!r} labels rows {seen[label]} and {index}"
            )
        seen[label] = index
    return None
