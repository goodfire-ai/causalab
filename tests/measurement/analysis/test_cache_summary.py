"""Report cache capabilities without claiming observed hits or warm occupancy."""

from copy import deepcopy

import pytest
from hypothesis import given, strategies as st

from causalab.measurement.analysis.summary import comparison_summary, render_summary
from tests.measurement.analysis.test_summary import evidence

pytestmark = pytest.mark.unit

_memo_capabilities = st.dictionaries(
    st.sampled_from(["tokenization", "metric_encoding", "table_validation"]),
    st.sampled_from(["available", "absent", "module_not_loaded"]),
)


def _policy(availability="available", *, lifetime="resident_worker"):
    return {
        "lifetime": lifetime,
        "observed_at": "after_execution",
        "host_memos": {"causalab.neural.shared.encoding._TOKENIZED": availability},
        "external_caches": "OS/HF caches not flushed",
    }


def _summary(before_policy, after_policy):
    before, samples = evidence()
    after = deepcopy(before)
    for record, policy in ((before, before_policy), (after, after_policy)):
        if policy is not None:
            record["context"]["cache_policy"] = policy
        # Cold measurements also retain resident provenance; it is not the
        # primary measured process and must not determine the visible policy.
        record["context"]["resident_cache_policy"] = _policy()
    return comparison_summary(
        before, after, samples, samples, [(7, 0)], {"per_sample": []}
    )


def test_different_memo_capabilities_are_visible_in_summary_and_html():
    before, after = _policy("absent"), _policy("available")
    summary = _summary(before, after)
    assert summary["cache_policy"] == {"before": before, "after": after}
    caveat = next(
        note for note in summary["caveats"] if "memo capabilities differ" in note
    )
    text = render_summary(summary)
    assert caveat in text
    assert "_TOKENIZED" in text
    assert "absent" in text and "available" in text
    assert "not cache hits or warmed occupancy" in text
    assert "OS/HF caches not flushed" in text


def test_identical_memo_capabilities_do_not_add_caveats():
    policy = _policy()
    summary = _summary(policy, deepcopy(policy))
    assert summary["caveats"] == []
    assert summary["status"] == "no_detected_caveats"
    assert "Host memo capabilities" in render_summary(summary)


@given(_memo_capabilities, _memo_capabilities)
def test_memo_caveat_tracks_recorded_capability_differences(before_memos, after_memos):
    before, after = _policy(), _policy()
    before["host_memos"] = before_memos
    after["host_memos"] = after_memos
    summary = _summary(before, after)
    assert any("memo capabilities differ" in note for note in summary["caveats"]) == (
        before_memos != after_memos
    )


def test_cold_policy_uses_primary_process_lifetime():
    lifetime = "fresh_process; may warm during preparation and execution"
    summary = _summary(_policy(lifetime=lifetime), _policy(lifetime=lifetime))
    text = render_summary(summary)
    assert lifetime in text
    assert "resident_worker" not in text
    assert "not cache hits or warmed occupancy" in text


@pytest.mark.parametrize("missing_arm", ["before", "after", "both"])
def test_missing_cache_evidence_is_visible_without_claiming_a_mismatch(missing_arm):
    summary = _summary(
        None if missing_arm in ("before", "both") else _policy(),
        None if missing_arm in ("after", "both") else _policy(),
    )
    for arm in ("before", "after"):
        if missing_arm in (arm, "both"):
            assert summary["cache_policy"][arm] is None
    assert summary["caveats"] == []
    assert "not recorded" in render_summary(summary)
