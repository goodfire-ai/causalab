"""Workflow contrasts bind both definitions to one resolved source revision."""

from copy import deepcopy

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.study.contrast import WorkflowContrast, ComparisonError

pytestmark = pytest.mark.unit


def test_different_resolved_commits_are_refused():
    with pytest.raises(ComparisonError, match="same resolved code commit"):
        WorkflowContrast.create(
            {"before": "a" * 40, "after": "b" * 40},
            {"before": "first", "after": "second"},
        )


def test_eager_reference_also_uses_the_same_commit():
    with pytest.raises(ComparisonError, match="same resolved code commit"):
        WorkflowContrast.create(
            {"before": "a", "after": "a", "eager": "b"},
            {"before": "first", "after": "second", "eager": "first"},
        )


@given(st.text(min_size=1))
def test_every_arm_definition_participates_in_the_shared_identity(candidate):
    commits = {"before": "a" * 40, "after": "a" * 40}
    original = WorkflowContrast.create(
        commits, {"before": "baseline", "after": candidate}
    )
    changed = WorkflowContrast.create(
        commits, {"before": "baseline", "after": candidate + "!"}
    )
    assert original.identity != changed.identity
    assert (
        original.identity
        == WorkflowContrast.create(
            dict(reversed(list(commits.items()))),
            {"after": candidate, "before": "baseline"},
        ).identity
    )


def test_receipt_is_independent_of_mutable_caller_maps():
    commits = {"before": "a", "after": "a"}
    definitions = {"before": "first", "after": "second"}
    contrast = WorkflowContrast.create(commits, definitions)
    expected = deepcopy(contrast.receipt())
    commits["after"] = "b"
    definitions["after"] = "changed"
    receipt = contrast.receipt()
    receipt["definitions"]["after"] = "changed"
    assert contrast.receipt() == expected


@pytest.mark.parametrize(
    "commits,definitions",
    [
        ({"before": "a"}, {"before": "first"}),
        ({"before": "a", "after": "a"}, {"before": "first"}),
        ({"before": "", "after": ""}, {"before": "first", "after": "second"}),
    ],
)
def test_incomplete_contrasts_are_refused(commits, definitions):
    with pytest.raises(ComparisonError):
        WorkflowContrast.create(commits, definitions)


def test_definition_diff_retains_added_nulls_and_order_changes():
    from causalab.measurement.study.contrast import definition_changes

    before = {"removed": None, "rows": [1, 2], "same": {"lr": 0.01}}
    after = {"added": None, "rows": [2, 1], "same": {"lr": 0.01}}
    assert definition_changes(before, after) == [
        {"path": ["added"], "kind": "added", "before": None, "after": None},
        {"path": ["removed"], "kind": "removed", "before": None, "after": None},
        {"path": ["rows"], "kind": "changed", "before": [1, 2], "after": [2, 1]},
    ]


@given(st.dictionaries(st.text(min_size=1, max_size=20), st.integers(), max_size=8))
def test_definition_diff_is_empty_for_equal_objects_with_different_key_order(value):
    from causalab.measurement.study.contrast import definition_changes

    assert definition_changes(value, dict(reversed(list(value.items())))) == []
