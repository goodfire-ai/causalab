"""Read and write metric tables as JSON arrays of row objects.

Writes preserve row order and convert nonfinite floats to JSON null. Callers
use this format for datasets, saved metrics, and workflow script inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.protocol.rules.errors import ProtocolError

__all__ = ["TABLE_SUFFIX", "read_table", "write_table"]

#: The one extension a metric table may carry.
TABLE_SUFFIX = ".json"


def write_table(target: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ``rows`` as a metric table.

    Non-finite floats become ``null``: ``json.dumps`` would otherwise emit the
    bare tokens ``NaN``/``Infinity``, which Python reads back but no other JSON
    parser accepts — and a metric that computed nothing is exactly the "no
    value" a ``null`` means (the same choice the per-step ``matched`` flag
    encodes for continuation reads)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [{key: _finite(value) for key, value in row.items()} for row in rows]
    target.write_text(json.dumps(payload, indent=2) + "\n")


def read_table(path: Path) -> list[dict[str, Any]]:
    """One metric table back as a list of row dicts."""
    if not path.is_file():
        raise ProtocolError("P2", f"table {str(path)!r} does not exist")
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ProtocolError(
            "P2",
            f"{path.name} is not a metric table — expected a JSON array of row objects",
        )
    return rows


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
