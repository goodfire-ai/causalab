"""The receipt's ``models`` list (§8): which model revisions a run loaded and
the commit each resolved to, built from the points' summaries by
`model_records` and written by `record_models`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.receipt import (
    MODELS_KEY,
    RUN_RECORD_NAME,
    model_records,
    record_models,
)

pytestmark = pytest.mark.unit

#: Synthetic snapshot commits, ordered so the sort is visible.
EARLIER = "1" * 40
LATER = "2" * 40
QWEN = "3" * 40


def _point(key: str, revision: str, resolved: str | None) -> dict[str, Any]:
    """One point's summary with the model it loaded."""
    return {
        "point_digest": f"{key}@{revision}",
        "model": {"key": key, "revision": revision, "resolved_revision": resolved},
    }


def _entry(key: str, revision: str, resolved: str | None) -> dict[str, Any]:
    return {"key": key, "revision": revision, "resolved_revision": resolved}


def test_one_entry_per_distinct_model_sorted_by_key() -> None:
    """A swept ``model.key`` gives one entry per model, sorted, and the
    points that loaded one model collapse to one entry."""
    summaries = [
        _point("gpt2", "main", EARLIER),
        _point("Qwen/Qwen3-8B", "main", QWEN),
        _point("gpt2", "main", EARLIER),
    ]
    assert model_records(summaries) == [
        _entry("Qwen/Qwen3-8B", "main", QWEN),
        _entry("gpt2", "main", EARLIER),
    ]


def test_a_ref_that_moved_between_loads_keeps_both_commits() -> None:
    """Two commits for one ``key`` and ``revision`` mean the ref moved during
    the run. The list keeps both, sorted by commit, and does not choose."""
    summaries = [_point("gpt2", "main", LATER), _point("gpt2", "main", EARLIER)]
    assert model_records(summaries) == [
        _entry("gpt2", "main", EARLIER),
        _entry("gpt2", "main", LATER),
    ]


def test_an_unresolved_commit_is_kept_as_null() -> None:
    """A model that did not load through the Hub cache, such as a local
    directory, reports no commit. Its entry stays, with a null commit, and
    sorts before a resolved entry of the same key and revision."""
    summaries = [_point("gpt2", "main", EARLIER), _point("gpt2", "main", None)]
    assert model_records(summaries) == [
        _entry("gpt2", "main", None),
        _entry("gpt2", "main", EARLIER),
    ]


def test_a_summary_without_a_model_is_skipped() -> None:
    """Only an engine that skips the shared execution, such as a test stub,
    reports no model. Such a summary adds nothing, and a run of only such
    points gives an empty list."""
    summaries = [{"point_digest": "a"}, {"point_digest": "b", "model": None}]
    assert model_records(summaries) == []
    assert model_records([*summaries, _point("gpt2", "main", EARLIER)]) == [
        _entry("gpt2", "main", EARLIER)
    ]


def test_record_models_amends_the_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / RUN_RECORD_NAME
    receipt.write_text(json.dumps({"digest": "d", "execution": {"device": "cpu"}}))
    assert record_models(tmp_path, [_point("gpt2", "main", EARLIER)]) == receipt
    record = json.loads(receipt.read_text())
    assert record == {
        "digest": "d",
        "execution": {"device": "cpu"},
        MODELS_KEY: [_entry("gpt2", "main", EARLIER)],
    }


def test_record_models_writes_nothing_without_a_model_or_a_receipt(
    tmp_path: Path,
) -> None:
    """A run whose points report no model keeps its receipt as it was. A
    caller that wrote no receipt has nothing to amend."""
    assert record_models(tmp_path, [_point("gpt2", "main", EARLIER)]) is None
    receipt = tmp_path / RUN_RECORD_NAME
    receipt.write_text('{"digest": "d"}')
    assert record_models(tmp_path, [{"point_digest": "a"}]) == receipt
    assert receipt.read_text() == '{"digest": "d"}'
