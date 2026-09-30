"""Record resolved token positions in a location ledger.

The ledger gives the token indices used for each address and row, along with
alignment information. It is a run artifact for inspecting the experiment's
locations. Document identity is computed from the authored specification and data."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from causalab.protocol.schema import Document

__all__ = [
    "LEDGER_COLUMNS",
    "LedgerRow",
    "LocationLedger",
    "ledger_records",
    "wants_ledger",
]

#: The seven columns of one ledger row (§6), in this order.
LEDGER_COLUMNS: tuple[str, ...] = (
    "example",
    "edit_group",
    "constituent",
    "side",
    "token_index",
    "token_id",
    "decoded_token",
)


@dataclasses.dataclass(frozen=True)
class LedgerRow:
    """One addressed token: which example, in which forward group (``model``
    on ``input``), by which constituent (a position's name, or ``name[k]`` for
    the k-th member of a non-atomic set), on which side (the input role), at
    which index of the row's own token sequence (0 = the row's first real
    token, chat prefix included — padding never enters), with what token id
    and decoded piece."""

    example: int
    edit_group: str
    constituent: str
    side: str
    token_index: int
    token_id: int
    decoded_token: str

    def as_record(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in LEDGER_COLUMNS}


class LocationLedger:
    """The rows one point's resolution produced, keyed so that resolving the
    same address twice (the width pre-flight, then the write) records it once.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[int, str, str, str, int], LedgerRow] = {}

    def add(self, row: LedgerRow) -> None:
        key = (row.example, row.edit_group, row.constituent, row.side, row.token_index)
        previous = self._rows.get(key)
        if previous is not None and previous != row:
            raise AssertionError(
                f"ledger row {key} recorded twice with different tokens: "
                f"{previous} vs {row}"
            )
        self._rows[key] = row

    def rows(self) -> tuple[LedgerRow, ...]:
        """Every row, by (example, edit group, constituent, side, token
        index) — a reading order."""
        return tuple(self._rows[key] for key in sorted(self._rows))

    def records(self) -> list[dict[str, Any]]:
        return [row.as_record() for row in self.rows()]

    def __len__(self) -> int:
        return len(self._rows)


def wants_ledger(doc: Document) -> bool:
    """Whether the document opts into the ledger: a ``save`` entry of kind
    ``location_ledger`` (§2.12). Nothing else makes a run write one."""
    return any(entry.kind == "location_ledger" for entry in doc.save)


def ledger_records(
    ledger: LocationLedger, point_digest: str, coords: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The rows the saved table carries for one point: the seven ledger
    columns plus the point's provenance (``point``) and coordinates
    (``coords``), which a swept document needs to tell its points apart."""
    return [
        {**record, "point": point_digest, "coords": dict(coords)}
        for record in ledger.records()
    ]
