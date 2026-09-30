"""Configuration comparisons expose intentional differences without changing code studies."""

from copy import deepcopy
from typing import Any

import pytest

from causalab.measurement.analysis.summary import comparison_summary, render_summary
from tests.measurement.analysis.test_summary import evidence

pytestmark = pytest.mark.unit


def _summary(changes):
    before, samples = evidence()
    after = deepcopy(before)
    comparison = {
        "kind": "workflow",
        "source_commit": "a" * 40,
        "definitions": {"before": "first", "after": "second"},
    }
    for arm, record in (("before", before), ("after", after)):
        record["input_identity"] = "shared-study"
        record["context"]["worker"].update(
            comparison_kind="workflow",
            comparison_contract=comparison,
            configuration_changes=changes,
            benchmark_identity=comparison["definitions"][arm],
            source_commit="a" * 40,
        )
    return comparison_summary(
        before, after, samples, samples, [(7, 0)], {"per_sample": []}
    )


def test_report_distinguishes_shared_contract_from_different_workflows():
    changes = [
        {
            "path": ["protocols", "fit", "method", "train", "epochs"],
            "before": 1,
            "after": 2,
            "kind": "changed",
        }
    ]
    summary = _summary(changes)
    assert summary["configuration_comparison"]["changes"] == changes
    assert summary["benchmark"]["identity_status"] == "matched"
    assert any("configured outputs" in warning for warning in summary["caveats"])
    rendered = render_summary(summary)
    assert "Shared comparison contract" in rendered
    assert "Workflow definition" in rendered
    assert "Configuration differences" in rendered
    assert "epochs" in rendered


def test_configuration_diff_html_is_escaped():
    rendered = render_summary(
        _summary(
            [
                {
                    "path": ["<script>"],
                    "before": None,
                    "after": '<img src=x onerror="alert(1)">',
                    "kind": "added",
                }
            ]
        )
    )
    assert "<script>" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;" in rendered


@pytest.mark.parametrize("reference_mode", [None, "code", "workflow", "invalid"])
@pytest.mark.parametrize("candidate_mode", [None, "code", "workflow", "invalid"])
def test_direct_summary_requires_matching_known_comparison_modes(
    reference_mode: str | None, candidate_mode: str | None
) -> None:
    before: dict[str, Any]
    before, samples = evidence()
    after = deepcopy(before)
    for record, mode in ((before, reference_mode), (after, candidate_mode)):
        if mode is not None:
            record["context"]["worker"]["comparison_kind"] = mode
    normalized = (reference_mode or "code", candidate_mode or "code")
    if normalized not in (("code", "code"), ("workflow", "workflow")):
        with pytest.raises(ValueError, match="comparison modes"):
            comparison_summary(
                before, after, samples, samples, [(7, 0)], {"per_sample": []}
            )
    else:
        result = comparison_summary(
            before, after, samples, samples, [(7, 0)], {"per_sample": []}
        )
        assert ("configuration_comparison" in result) == (
            normalized == ("workflow", "workflow")
        )


def test_eager_reference_retains_its_changes_from_the_study_baseline() -> None:
    eager: dict[str, Any]
    eager, samples = evidence()
    candidate = deepcopy(eager)
    changes = {
        "eager": [{"path": ["execution", "compile"], "before": True, "after": False}],
        "after": [{"path": ["train", "steps", "epochs"], "before": 1, "after": 2}],
    }
    for arm, record in (("eager", eager), ("after", candidate)):
        record["context"]["worker"].update(
            comparison_kind="workflow",
            arm=arm,
            configuration_changes=changes[arm],
        )
    result = comparison_summary(
        eager, candidate, samples, samples, [(7, 0)], {"per_sample": []}
    )
    assert result["configuration_comparison"]["reference_changes"] == changes["eager"]
    assert result["configuration_comparison"]["changes"] == changes["after"]
    rendered = render_summary(result)
    assert "relative to the study baseline" in rendered
    assert "compile" in rendered and "epochs" in rendered
