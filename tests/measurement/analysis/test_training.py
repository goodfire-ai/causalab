"""Seed labels and unchanged outputs cannot hide changed experimental starts."""

from copy import deepcopy

import pytest

from causalab.measurement.analysis.training import compare_training

pytestmark = pytest.mark.unit


def sample():
    return {
        "numerics_context": {
            "scope": "separate numerical pass",
            "fits": [
                {
                    "identity": {
                        "step": "fit",
                        "protocol": "fixed",
                        "coords": {},
                        "train_params": ["gate.theta"],
                    },
                    "seed": 7,
                    "initial": {
                        "parameters": "initial bytes",
                        "stages": {},
                        "optimizer_class": "AdamW",
                        "optimizer_state": {},
                    },
                    "rng_after_initialization": "start",
                    "rng_at_return": "end",
                    "batches": [
                        {
                            "update": 0,
                            "epoch": 0,
                            "logical_indices": [0, 1],
                            "role_rows_sha256": {"base": "rows"},
                            "order_rng_seed": 7,
                            "order_rng_state": "order",
                            "rng_before_update": "update",
                        }
                    ],
                    "optimizer_steps": 1,
                }
            ],
        },
        "observer_check": {"status": "compared", "exactly_equal": True},
    }


def test_changed_initial_state_is_reported_even_when_observer_outputs_match():
    before = {(7, 0): sample(), (7, 1): sample()}
    after = deepcopy(before)
    after[7, 1]["numerics_context"]["fits"][0]["initial"]["parameters"] = (
        "different bytes"
    )
    report = compare_training(before, after, sorted(before))
    fit = report["per_sample"][1]["fits"][0]
    assert not fit["initial_parameters_match"]
    assert fit["logical_schedule_matches"]
    assert fit["observed_rng_states_match"]
    assert report["per_sample"][1]["after"]["observer_check"]["exactly_equal"]
    after_start = next(row for row in report["within_seed"] if row["arm"] == "after")
    assert after_start["initial_parameters_match"] is False


@pytest.mark.parametrize(
    "change,flag",
    [
        ("schedule", "logical_schedule_matches"),
        ("rng", "observed_rng_states_match"),
        ("mask_rng", "observed_rng_states_match"),
        ("optimizer", "initial_optimizer_match"),
    ],
)
def test_separate_training_confounds(change, flag):
    before, after = sample(), sample()
    fit = after["numerics_context"]["fits"][0]
    if change == "schedule":
        fit["batches"][0]["logical_indices"] = [1, 0]
        fit["batches"][0]["role_rows_sha256"]["base"] = "reversed rows"
    elif change == "rng":
        fit["batches"][0]["rng_before_update"] = "different consumption"
    elif change == "mask_rng":
        before["numerics_context"]["fits"][0]["batches"][0]["mask_rng_state"] = "before"
        fit["batches"][0]["mask_rng_state"] = "after"
    else:
        fit["initial"]["optimizer_state"] = {"existing momentum": "bytes"}
    report = compare_training({(7, 0): before}, {(7, 0): after}, [(7, 0)])
    assert report["per_sample"][0]["fits"][0][flag] is False
    if change == "mask_rng":
        assert report["per_sample"][0]["fits"][0]["logical_schedule_matches"] is True
    assert all(row["initial_parameters_match"] is None for row in report["within_seed"])


def test_missing_training_evidence_is_not_a_match():
    report = compare_training({(7, 0): sample()}, {(7, 0): {}}, [(7, 0)])
    assert report["per_sample"][0]["status"] == "incomplete"
    assert report["per_sample"][0]["fits"] == []
