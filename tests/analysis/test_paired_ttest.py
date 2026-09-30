"""``causalab.analysis.paired_ttest`` against a hand-computed oracle.

Ported from ``paired_ttest@1``; assertions unchanged. Differences 0.5, 1.0, 0.5,
1.0 → mean 0.75, sample sd sqrt(0.25/3) = 0.288675, so
t = 0.75 / (0.288675 / 2) = 5.196152 on 3 degrees of freedom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.io.tables import read_table
from causalab.analysis import paired_ttest
from causalab.io.step_io import StepError
from tests.step_scripts import put_table, run_step

pytestmark = pytest.mark.numerical_unit

A = [{"example_id": str(i), "value": v} for i, v in enumerate([1.0, 2.0, 3.0, 4.0])]
B = [{"example_id": str(i), "value": v} for i, v in enumerate([0.5, 1.0, 2.5, 3.0])]


def _ttest(tmp_path: Path, a=A, b=B, tag: str = "a") -> dict:
    root = tmp_path / tag
    left = put_table(root / "a.json", list(a))
    right = put_table(root / "b.json", list(b))
    out = root / "stats.json"
    run_step(paired_ttest, {"a": left, "b": right}, {"stats": out})
    rows = read_table(out)
    assert len(rows) == 1
    return rows[0]


def test_statistics_match_the_oracle(tmp_path):
    row = _ttest(tmp_path)
    assert list(row) == [
        "n_pairs",
        "df",
        "mean_difference",
        "t_statistic",
        "p_value",
        "unit",
        "comparison",
    ]
    # two tables that declare nothing compare as two arms, in no stated unit
    assert row["unit"] is None and row["comparison"] == "arm"
    assert row["n_pairs"] == 4 and row["df"] == 3
    assert row["mean_difference"] == pytest.approx(0.75)
    assert row["t_statistic"] == pytest.approx(5.196152, abs=1e-6)
    assert row["p_value"] == pytest.approx(0.013847, abs=1e-6)


def test_the_test_is_two_sided(tmp_path):
    """Swapping the arms flips the statistic's sign and leaves p alone."""
    forward = _ttest(tmp_path, tag="fwd")
    reverse = _ttest(tmp_path, a=B, b=A, tag="rev")
    assert reverse["t_statistic"] == pytest.approx(-forward["t_statistic"])
    assert reverse["p_value"] == pytest.approx(forward["p_value"])


def test_rows_are_averaged_within_a_pair_before_the_test(tmp_path):
    """A swept document writes one row per (example, point); a bare join would
    multiply rows and inflate n."""
    once = _ttest(tmp_path, tag="once")
    twice = _ttest(tmp_path, a=A + A, tag="twice")
    assert twice["n_pairs"] == once["n_pairs"] == 4
    assert twice["t_statistic"] == pytest.approx(once["t_statistic"])


def test_only_shared_pairs_are_compared(tmp_path):
    partial = [{"example_id": "0", "value": 0.5}, {"example_id": "1", "value": 1.0}]
    assert _ttest(tmp_path, b=partial, tag="partial")["n_pairs"] == 2


def test_a_constant_difference_is_reported_rather_than_fudged(tmp_path):
    """Zero spread makes the statistic degenerate; inf/0.0 is honest, an
    epsilon in the denominator would not be. Note the JSON round-trip: a
    non-finite float is written as null, so the sign is what survives."""
    shifted = [{"example_id": r["example_id"], "value": r["value"] + 1.0} for r in A]
    row = _ttest(tmp_path, a=shifted, b=A, tag="const")
    assert row["t_statistic"] is None  # inf is not JSON; null is the honest cell
    assert row["p_value"] == 0.0


def test_too_few_shared_pairs_is_refused(tmp_path):
    with pytest.raises(StepError) as err:
        _ttest(tmp_path, b=[{"example_id": "0", "value": 0.5}], tag="lonely")
    assert "at least 2" in str(err.value)
