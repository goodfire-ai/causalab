"""``causalab.analysis.head_stats`` against a hand-computed oracle.

Ported from ``head_stats@1``; assertions unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.io.tables import read_table
from causalab.analysis import head_stats
from causalab.io.step_io import StepError
from tests.step_scripts import put_table, run_step

pytestmark = pytest.mark.numerical_unit

ROWS = [
    {"sites.target.layers": 0, "sites.target.head": 0, "value": 1.0},
    {"sites.target.layers": 0, "sites.target.head": 0, "value": 3.0},
    {"sites.target.layers": 1, "sites.target.head": 0, "value": 5.0},
    {"sites.target.layers": 1, "sites.target.head": 1, "value": 2.0},
    {"sites.target.layers": 1, "sites.target.head": 1, "value": 4.0},
]


def _stats(tmp_path: Path, rows=ROWS, tag: str = "a", **extra) -> list[dict]:
    table = put_table(tmp_path / tag / "in.json", list(rows))
    out = tmp_path / tag / "stats.json"
    run_step(head_stats, {"table": table, **extra}, {"stats": out})
    return read_table(out)


def test_cells_match_the_oracle(tmp_path):
    """(0,0): {1,3} -> mean 2, population std 1. (1,0): {5} -> 5, 0.
    (1,1): {2,4} -> mean 3, population std 1."""
    stats = _stats(tmp_path)
    assert list(stats[0]) == ["layer", "head", "n", "mean", "std"]
    assert stats == [
        {"layer": 0, "head": 0, "n": 2, "mean": 2.0, "std": 1.0},
        {"layer": 1, "head": 0, "n": 1, "mean": 5.0, "std": 0.0},
        {"layer": 1, "head": 1, "n": 2, "mean": 3.0, "std": 1.0},
    ]


def test_a_single_row_cell_has_zero_spread_not_nan(tmp_path):
    """ddof=0 on purpose: NaN here would poison every downstream mean."""
    lonely = next(
        row for row in _stats(tmp_path) if row["layer"] == 1 and row["head"] == 0
    )
    assert lonely["std"] == 0.0


def test_rows_are_sorted_so_the_output_is_deterministic(tmp_path):
    shuffled = [ROWS[i] for i in (4, 0, 3, 1, 2)]
    assert _stats(tmp_path, tag="one") == _stats(tmp_path, shuffled, tag="two")


def test_a_missing_column_names_what_the_table_has(tmp_path):
    with pytest.raises(StepError) as err:
        _stats(tmp_path, head_column="sites.target.expert")
    assert "sites.target.expert" in str(err.value)
    assert "sites.target.head" in str(err.value)  # what it does have
