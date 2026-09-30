"""Acceptance cannot turn absent evidence into a pass; eager stays a control."""

import json

import pytest

from causalab.measurement.analysis.reports import evaluate_acceptance, write_reports
from tests.measurement.analysis.test_compare import receipt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "0", []])
def test_non_numeric_or_missing_evidence_is_not_accepted(value):
    criteria = [{"name": "drift", "path": ["value"], "maximum": 1}]
    result = evaluate_acceptance({"value": value}, criteria)
    assert result["status"] == "insufficient_evidence"
    assert result["criteria"][0]["value"] is None
    assert evaluate_acceptance({}, criteria)["status"] == "insufficient_evidence"


def test_bounds_include_equality_and_failures_dominate_missing_evidence():
    criteria = [
        {"name": "lower", "path": ["values", "0"], "minimum": 2},
        {"name": "upper", "path": ["values", "1"], "maximum": 4},
    ]
    assert evaluate_acceptance({"values": [2, 4]}, criteria)["status"] == "passed"
    failed = evaluate_acceptance({"values": [1]}, criteria)
    assert failed["status"] == "failed"
    assert [row["status"] for row in failed["criteria"]] == [
        "failed",
        "insufficient_evidence",
    ]
    assert evaluate_acceptance({}, [])["status"] == "not_evaluated"


def test_study_reports_eager_variability_and_authored_acceptance(tmp_path):
    arms = {
        "eager": receipt(tmp_path / "eager", [[[1.0], [3.0]], [[11.0], [13.0]]]),
        "before": receipt(tmp_path / "before", [[[1.0], [3.0]], [[11.0], [13.0]]]),
        "after": receipt(tmp_path / "after", [[[0.0], [4.0]], [[10.0], [14.0]]]),
    }
    plan = {
        "bootstrap_draws": 100,
        "order_seed": 0,
        "acceptance": [
            {
                "name": "variance",
                "path": [
                    "cases",
                    "test",
                    "comparisons",
                    "before_after",
                    "observations",
                    "example/site",
                    "within_seed",
                    "0",
                    "variance_ratio",
                ],
                "maximum": 2,
            }
        ],
    }
    out = tmp_path / "reports"
    study = write_reports({"test": arms}, plan, out)
    comparisons = study["cases"]["test"]["comparisons"]
    assert set(comparisons) == {"before_after", "eager_before", "eager_after"}
    assert comparisons["eager_before"]["contrast"] == {
        "reference": "eager",
        "candidate": "before",
    }
    eager = comparisons["eager_after"]["observations"]["example/site"]
    assert eager["within_seed"][0]["before_mean_coordinate_variance"] == 2
    assert eager["across_seed_means"]["before_mean_coordinate_variance"] == 50
    assert study["acceptance"]["status"] == "failed"
    assert study["acceptance"]["criteria"][0]["value"] == 4
    assert json.loads((out / "study.json").read_text()) == study
    assert (out / "index.html").is_file()
    for files in study["cases"]["test"]["files"].values():
        assert (out / files["report"]).is_file()
    assert "Reference: eager" in (out / "test.eager_after.html").read_text()
    rendered = (out / "test.eager_after.html").read_text()
    assert "Across-seed variation" in rendered
    assert "Within-seed numerical drift and variance" in rendered
    assert "50.0" in rendered
    assert "Comparison evidence: caveats_present" in rendered
    assert "primary outputs are not attested" in (out / "index.html").read_text()
    assert "measurement/analysis/compare.py" in study["analysis"]["files_sha256"]
