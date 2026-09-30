"""Reading and writing metric tables in tests.

Tables are JSON on disk (``causalab.io.tables``) — an array of row
objects. Tests that want to reduce one reach for pandas, so this is the two
lines of glue, in one place, rather than in every assertion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from causalab.io.tables import read_table, write_table

__all__ = ["frame", "write_frame", "write_rows"]


def frame(path: Path) -> pd.DataFrame:
    """One saved metric table as a DataFrame."""
    return pd.DataFrame(read_table(path))


def write_frame(df: pd.DataFrame, path: Path) -> None:
    """A DataFrame as a metric table on disk."""
    write_table(path, df.to_dict(orient="records"))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Row dicts as a metric table on disk."""
    write_table(path, rows)
