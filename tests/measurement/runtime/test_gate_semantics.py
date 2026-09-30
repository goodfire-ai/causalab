"""Gate measurements must not reinterpret a saved map as sigmoid logits."""

import json

import pytest
import torch
from hypothesis import given, strategies as st
from safetensors.torch import save_file

from causalab.measurement.runtime.observations import observation_specs

pytestmark = pytest.mark.unit


def _write_gate(root, parametrization, *, common=False):
    fit = root / "fit"
    fit.mkdir(exist_ok=True)
    entry = {"slot": "theta", "coords": {}, "produced_by": "fit-point"}
    header = {}
    if parametrization is not None:
        (header if common else entry)["parametrization"] = parametrization
    header["entries"] = json.dumps({"theta": entry})
    save_file(
        {"theta": torch.tensor([0.1, 0.9])},
        str(fit / "gate.safetensors"),
        metadata=header,
    )
    return {
        "gate": {
            "kind": "gate",
            "step": "fit",
            "file": "gate.safetensors",
            "temperature": 1.0,
        }
    }


@pytest.mark.parametrize(
    "parametrization", ["clamp", "hard_concrete", "budget", "boundary"]
)
@pytest.mark.parametrize("common", [False, True])
def test_unsupported_saved_gate_maps_are_refused(tmp_path, parametrization, common):
    specs = _write_gate(tmp_path, parametrization, common=common)
    with pytest.raises(ValueError, match="supports only sigmoid") as error:
        observation_specs(tmp_path, specs)
    assert error.value.parametrization == parametrization
    assert error.value.observation == "gate"


@pytest.mark.parametrize("parametrization", [None, "sigmoid"])
def test_sigmoid_semantics_are_recorded_in_resolved_spec(tmp_path, parametrization):
    specs = _write_gate(tmp_path, parametrization)
    resolved = observation_specs(tmp_path, specs)
    assert (
        next(value for key, value in resolved.items() if "/" in key)["parametrization"]
        == "sigmoid"
    )


@given(st.text(min_size=1).filter(lambda value: value != "sigmoid"))
def test_unknown_gate_maps_never_silently_use_sigmoid(parametrization):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        specs = _write_gate(root, parametrization)
        with pytest.raises(ValueError, match="supports only sigmoid"):
            observation_specs(root, specs)


def test_observation_helpers_import_from_controller_not_historical_arm():
    import subprocess
    import sys
    from pathlib import Path

    controller = Path(__file__).resolve().parents[3] / "causalab"
    script = """
import importlib
import sys
import types

# Historical arms may have no intervention-stability helpers at all.
legacy = types.ModuleType('causalab.measurement.analysis.stability')
sys.modules[legacy.__name__] = legacy
runtime = types.ModuleType('_measurement_runtime')
runtime.__path__ = [sys.argv[1]]
sys.modules[runtime.__name__] = runtime
observations = importlib.import_module('_measurement_runtime.measurement.runtime.observations')
assert observations.require_sigmoid_gate.__module__ == '_measurement_runtime.measurement.analysis.stability'
try:
    observations.require_sigmoid_gate('gate', 'clamp')
except ValueError as error:
    assert error.parametrization == 'clamp'
else:
    raise AssertionError('controller guard did not reject clamp')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(controller)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
