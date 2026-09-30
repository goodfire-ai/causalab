"""Gate reports require exact metadata and identify the rejected observation."""

import pytest
import torch
from hypothesis import given, strategies as st

from causalab.measurement.analysis.compare import observation_spec, paired_difference
from causalab.measurement.analysis.stability import (
    MissingGateObservationSpecError,
    UnsupportedGateObservationError,
)

pytestmark = pytest.mark.unit


@given(st.text(min_size=1))
def test_suffixed_gate_cannot_fall_back_to_base_metadata(suffix):
    key = "gate/" + suffix
    record = {"observation_specs": {"gate": {"kind": "gate", "temperature": 1.0}}}
    with pytest.raises(MissingGateObservationSpecError) as error:
        observation_spec(record, key)
    assert error.value.observation == key


def test_exact_gate_spec_preserves_entry_temperature_and_provenance():
    entry = {
        "kind": "gate",
        "temperature": 0.02,
        "temperature_source": "fit_diagnostics",
        "produced_by": "point-3",
        "parametrization": "sigmoid",
    }
    record = {"observation_specs": {"gate": {"kind": "gate", "temperature": 1.0}}}
    sample = {"observation_specs": {"gate/theta": entry}}
    assert observation_spec(record, "gate/theta", sample) == entry


@pytest.mark.parametrize("include_base", [False, True])
def test_ragged_gate_row_without_exact_metadata_fails_closed(include_base):
    entry = {"kind": "gate", "temperature": 0.02}
    specs = {"gate/theta": entry}
    if include_base:
        specs["gate"] = entry
    record = {"observation_specs": specs}
    with pytest.raises(MissingGateObservationSpecError):
        observation_spec(record, "gate/theta/row_0")


@pytest.mark.parametrize("arm", ["before", "after"])
def test_paired_gate_failure_identifies_observation_not_arm(arm):
    before = {"kind": "gate", "temperature": 1.0}
    after = dict(before)
    (before if arm == "before" else after)["parametrization"] = "clamp"
    with pytest.raises(UnsupportedGateObservationError) as error:
        paired_difference(
            torch.tensor([0.1]),
            torch.tensor([0.9]),
            before,
            after,
            observation="gate/theta",
        )
    assert error.value.observation == "gate/theta"
    assert error.value.parametrization == "clamp"
