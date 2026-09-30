"""The report distinguishes fixed benchmark inputs from each arm's source."""

from copy import deepcopy
import html
import json

import pytest
from hypothesis import given, strategies as st

from causalab.measurement.analysis.summary import comparison_summary, render_summary
from tests.measurement.analysis.test_summary import evidence

pytestmark = pytest.mark.unit


def _summary(
    before_worker,
    after_worker,
    *,
    before_identity="receipt-sha256",
    after_identity="receipt-sha256",
):
    before, samples = evidence()
    after = deepcopy(before)
    for record, identity in ((before, before_identity), (after, after_identity)):
        if identity is not None:
            record["input_identity"] = identity
    for record, worker in ((before, before_worker), (after, after_worker)):
        record["context"]["worker"].update(worker)
    return comparison_summary(
        before, after, samples, samples, [(7, 0)], {"per_sample": []}
    )


def _worker(commit="a" * 40):
    return {
        "benchmark_identity": "benchmark-sha256",
        "source_commit": commit,
        "source_pins": {"code": {"causalab.metric": commit}, "scripts": {}},
        "shared_pins": {
            "documents": {"protocol.json": "document-sha256"},
            "datasets": {"train": "dataset-sha256"},
            "files": {"basis.pt": "artifact-sha256"},
        },
    }


def test_report_identifies_benchmark_and_each_committed_source_without_warning():
    before, after = _worker(), _worker("b" * 40)
    summary = _summary(before, after)
    assert summary["benchmark"] == {
        "shared_identity": "receipt-sha256",
        "identity_status": "matched",
        "arms": {"before": before, "after": after},
    }
    assert summary["caveats"] == []
    rendered = render_summary(summary)
    assert "Shared benchmark identity" in rendered
    assert "equality enforced" in rendered
    assert "receipt-sha256" in rendered
    assert "benchmark-sha256" in rendered
    assert "Baseline (before)" in rendered and "Candidate (after)" in rendered
    assert "Source commit" in rendered and "Source hashes" in rendered
    for worker in (before, after):
        assert worker["source_commit"] in rendered
        assert (
            html.escape(json.dumps(worker["source_pins"], sort_keys=True)) in rendered
        )
        assert (
            html.escape(json.dumps(worker["shared_pins"], sort_keys=True)) in rendered
        )
    assert rendered.index("Shared benchmark identity") < rendered.index(
        "Declared execution"
    )


@pytest.mark.parametrize(
    "field", ["benchmark_identity", "source_commit", "source_pins", "shared_pins"]
)
def test_unrecorded_evidence_is_distinct_from_recorded_empty_pins(field):
    before, after = _worker(), _worker()
    before.pop(field)
    after["source_pins"] = {"code": {}, "scripts": {}}
    summary = _summary(before, after)
    assert summary["benchmark"]["arms"]["before"][field] is None
    assert summary["benchmark"]["arms"]["after"]["source_pins"] == {
        "code": {},
        "scripts": {},
    }
    assert "not recorded" in render_summary(summary)
    assert summary["benchmark"]["shared_identity"] == "receipt-sha256"


def test_disagreeing_benchmark_identities_are_never_labeled_shared():
    before, after = _worker(), _worker()
    summary = _summary(before, after, after_identity="different-benchmark")
    assert summary["benchmark"]["shared_identity"] is None
    assert summary["benchmark"]["identity_status"] == "mismatch"
    assert "mismatch" in render_summary(summary)
    assert "equality enforced" not in render_summary(summary)


@pytest.mark.parametrize("missing", ["before", "after", "both"])
def test_missing_receipt_identity_is_not_inferred_from_worker_evidence(missing):
    summary = _summary(
        _worker(),
        _worker(),
        before_identity=None if missing in ("before", "both") else "receipt-sha256",
        after_identity=None if missing in ("after", "both") else "receipt-sha256",
    )
    assert summary["benchmark"]["shared_identity"] is None
    assert summary["benchmark"]["identity_status"] == "missing"
    rendered = render_summary(summary)
    assert "Shared benchmark identity: not recorded" in rendered
    assert "equality enforced" not in rendered


def test_missing_worker_identity_does_not_hide_enforced_receipt_identity():
    before, after = _worker(), _worker()
    before.pop("benchmark_identity")
    after.pop("benchmark_identity")
    summary = _summary(before, after)
    assert summary["benchmark"]["shared_identity"] == "receipt-sha256"
    assert summary["benchmark"]["identity_status"] == "matched"
    assert (
        "Shared benchmark identity (equality enforced): receipt-sha256"
        in render_summary(summary)
    )


@given(
    st.dictionaries(st.text(min_size=1, max_size=30), st.text(max_size=64), max_size=5),
    st.dictionaries(st.text(min_size=1, max_size=30), st.text(max_size=64), max_size=5),
)
def test_each_arms_pin_census_is_preserved_and_html_escaped(before_pins, after_pins):
    before, after = _worker(), _worker("b" * 40)
    before["source_pins"] = {"code": before_pins, "scripts": {}}
    after["source_pins"] = {"code": {}, "scripts": after_pins}
    summary = _summary(before, after)
    rendered = render_summary(summary)
    for arm, worker in (("before", before), ("after", after)):
        assert summary["benchmark"]["arms"][arm]["source_pins"] == worker["source_pins"]
        assert (
            html.escape(json.dumps(worker["source_pins"], sort_keys=True)) in rendered
        )


def test_source_provenance_cannot_inject_html():
    worker = _worker('<script>alert("source")</script>')
    worker["benchmark_identity"] = '<img src=x onerror="alert(1)">'
    rendered = render_summary(_summary(worker, worker))
    assert "<script>" not in rendered and "<img" not in rendered
    assert html.escape(worker["source_commit"]) in rendered
    assert html.escape(worker["benchmark_identity"]) in rendered
