"""Public comparison APIs must reject explicit unsupported gate maps."""

import pytest
import torch

from causalab.measurement.analysis.compare import observation_spec, paired_difference

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "parametrization", ["clamp", "hard_concrete", "budget", "boundary"]
)
def test_external_observation_specs_refuse_unsupported_gate_maps(parametrization):
    spec = {"kind": "gate", "temperature": 1.0, "parametrization": parametrization}
    with pytest.raises(ValueError, match="supports only sigmoid"):
        observation_spec({"observation_specs": {"g": spec}}, "g")


@pytest.mark.parametrize("arm", ["before", "after"])
def test_direct_paired_comparison_refuses_unsupported_gate_map(arm):
    before = {"kind": "gate", "temperature": 1.0}
    after = dict(before)
    (before if arm == "before" else after)["parametrization"] = "clamp"
    with pytest.raises(ValueError, match="supports only sigmoid"):
        paired_difference(
            torch.tensor([0.1]), torch.tensor([0.9]), before, after, observation="g"
        )
