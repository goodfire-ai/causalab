"""The receipt carries the bound a run measured, never one it authored twice
(spec §8, ``fit_rows``).

``execution.fit_rows`` is what the run was asked for — ``null`` when nothing
bounded the grad forwards and the engine measured a bound on its first window.
That measured number is the one an author pins, so it lands beside the request
as ``fit_rows_resolved``: only when the request was ``null``, only when some
fit reported one, the smallest over the points, and with ``fit_rows_shrinks``
beside it when a window had to be re-packed. A cohort's eval passes pack under
the same bound unless ``batch_rows`` is authored, so ``batch_rows`` is never
measured on its own and ``null`` there stays ``null``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol.receipt import (
    FIT_ROWS_RESOLVED_KEY,
    FIT_ROWS_SHRINKS_KEY,
    RUN_RECORD_NAME,
    bound_shrinks,
    fit_rows_shrinks,
    measured_bounds,
    record_measured_bounds,
    resolved_bound,
    resolved_fit_rows,
)

pytestmark = pytest.mark.unit


def _receipt(
    tmp_path: Path, fit_rows: int | None, batch_rows: int | None = None
) -> Path:
    path = tmp_path / RUN_RECORD_NAME
    path.write_text(
        json.dumps(
            {
                "execution": {
                    "batch_rows": batch_rows,
                    "fit_rows": fit_rows,
                    "model_source": "loaded",
                }
            }
        )
    )
    return path


def test_the_smallest_reported_bound_is_the_one_to_pin() -> None:
    """Cohorts measure independently and an authored bound never shrinks, so
    the pin every cohort of the run survived is the smallest."""
    summaries = [{"fit_rows": 32}, {"metrics": {}}, {"fit_rows": 48}]
    assert resolved_fit_rows(summaries) == 32
    assert resolved_bound(summaries, "fit_rows") == 32
    assert resolved_fit_rows([{"metrics": {}}]) is None
    assert resolved_fit_rows([{"fit_rows": None}]) is None
    assert resolved_bound([{"other": 12}, {"other": 9}], "other") == 9


def test_a_measured_bound_lands_beside_a_null_request(tmp_path: Path) -> None:
    path = _receipt(tmp_path, None, None)
    summaries = [{"fit_rows": 96}]
    assert measured_bounds({"fit_rows": None, "batch_rows": None}, summaries) == {
        FIT_ROWS_RESOLVED_KEY: 96
    }
    assert record_measured_bounds(tmp_path, summaries) == path
    execution = json.loads(path.read_text())["execution"]
    assert execution["fit_rows"] is None and execution["batch_rows"] is None
    assert execution[FIT_ROWS_RESOLVED_KEY] == 96
    assert "batch_rows_resolved" not in execution


def test_a_run_that_shrank_says_so_beside_the_bound(tmp_path: Path) -> None:
    path = _receipt(tmp_path, None)
    # a cohort's count is reported on each of its members: the largest, never the sum
    summaries = [
        {"fit_rows": 32, "fit_rows_shrinks": 1},
        {"fit_rows": 32, "fit_rows_shrinks": 1},
        {"fit_rows": 48},
    ]
    assert fit_rows_shrinks(summaries) == 1
    assert bound_shrinks(summaries, "fit_rows_shrinks") == 1
    # two cohorts that shrank: the worse one's count, not their sum
    assert (
        bound_shrinks(
            [{"fit_rows_shrinks": 1}, {"fit_rows_shrinks": 1}], "fit_rows_shrinks"
        )
        == 1
    )
    assert (
        bound_shrinks(
            [{"fit_rows_shrinks": 1}, {"fit_rows_shrinks": 2}], "fit_rows_shrinks"
        )
        == 2
    )
    record_measured_bounds(tmp_path, summaries)
    execution = json.loads(path.read_text())["execution"]
    assert execution[FIT_ROWS_RESOLVED_KEY] == 32
    assert execution[FIT_ROWS_SHRINKS_KEY] == 1
    # a run that never shrank carries no such key
    path = _receipt(tmp_path, None)
    record_measured_bounds(tmp_path, [{"fit_rows": 96}])
    assert FIT_ROWS_SHRINKS_KEY not in json.loads(path.read_text())["execution"]


def test_an_authored_bound_is_not_repeated(tmp_path: Path) -> None:
    path = _receipt(tmp_path, 16, 6)
    record_measured_bounds(tmp_path, [{"fit_rows": 16}])
    assert FIT_ROWS_RESOLVED_KEY not in json.loads(path.read_text())["execution"]


def test_a_run_without_a_bound_adds_nothing(tmp_path: Path) -> None:
    path = _receipt(tmp_path, None)
    before = path.read_text()
    record_measured_bounds(tmp_path, [{"metrics": {}}])
    assert path.read_text() == before
    assert record_measured_bounds(tmp_path / "elsewhere", [{"fit_rows": 8}]) is None
