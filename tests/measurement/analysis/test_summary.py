from copy import deepcopy

import pytest

from causalab.measurement.analysis.summary import comparison_summary, render_summary
from causalab.measurement.analysis.training import compare_training
from tests.measurement.analysis.test_training import sample

pytestmark = pytest.mark.unit


def evidence():
    record = {
        "case": "fit",
        "context": {
            "worker": {
                "execution": {"cuda_graphs": True},
                "execution_probe": {
                    "cases": {
                        "fit": {
                            "python_dispatch": ["torch.cuda.graphs.CUDAGraph.replay"],
                            "native_profile": {"status": "not_requested"},
                        }
                    }
                },
            }
        },
    }
    value = sample()
    value["observation_origin"] = "timing_pass"
    value["observer_check"]["observation_specs_match"] = True
    return record, {(7, 0): value}


def test_summary_reports_work_observer_and_execution_confounds_separately():
    record, before = evidence()
    after = deepcopy(before)
    after[7, 0]["numerics_context"]["fits"][0]["optimizer_steps"] = 2
    after[7, 0]["observer_check"]["exactly_equal"] = False
    candidate = deepcopy(record)
    candidate["context"]["worker"]["execution_probe"]["cases"]["fit"][
        "python_dispatch"
    ] = []
    summary = comparison_summary(
        record,
        candidate,
        before,
        after,
        [(7, 0)],
        compare_training(before, after, [(7, 0)]),
    )
    assert summary["diagnostic_work"][0]["before_updates"] == 1
    assert summary["diagnostic_work"][0]["after_updates"] == 2
    assert summary["outputs"]["after"]["diagnostic_checks"] == {"different": 1}
    assert summary["execution"]["after"]["graph_replay_observed"] is False
    text = render_summary(summary)
    assert "same-work implementation speedup" in text
    assert "stochastic variation" in text
    assert "graph replay was not observed" in text
    assert "do not prove identical timed work" in text


def test_equal_evidence_is_not_a_scientific_acceptance_verdict():
    record, samples = evidence()
    result = comparison_summary(
        record,
        record,
        samples,
        samples,
        [(7, 0)],
        compare_training(samples, samples, [(7, 0)]),
    )
    assert result["status"] == "no_detected_caveats"
    assert result["caveats"] == []
    assert "not acceptance" in result["policy"]
    assert result["replication"] == {
        "seed_count": 1,
        "repeats_per_seed": [{"seed": 7, "repeats": 1}],
    }
    assert "Independent seeds: 1" in render_summary(result)
    assert "not calibrated performance regression gates" in render_summary(result)


def test_missing_evidence_is_visible():
    record = {"case": "fit"}
    samples = {(7, 0): {}}
    result = comparison_summary(
        record,
        record,
        samples,
        samples,
        [(7, 0)],
        compare_training(samples, samples, [(7, 0)]),
    )
    assert result["status"] == "caveats_present"
    assert result["execution"]["before"]["graph_replay_observed"] is None
    assert result["outputs"]["before"]["diagnostic_checks"] == {"unavailable": 1}
