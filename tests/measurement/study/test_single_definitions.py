"""Single-source studies share definition loading with workflow comparisons."""

import copy
import json

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.runtime.benchmark import benchmark_identity
from causalab.measurement.spec import MeasurementSpecError, parse_measurement
from causalab.measurement.study.definitions import load_definitions
from tests.measurement.study.test_definitions import _files

pytestmark = pytest.mark.unit


@given(observed=st.booleans())
def test_single_definition_preserves_authorship_and_observation_identity(
    tmp_path_factory, observed
):
    root = tmp_path_factory.mktemp("single-definitions")
    document, _ = _files(root)
    raw = json.loads(document.read_text())
    plan = raw["measurement"]
    del plan["comparison"]
    plan["mode"] = "single"
    plan["source"] = plan.pop("arms")["before"]
    if not observed:
        del plan["observations"]
    document.write_text(json.dumps(raw))
    authored = parse_measurement(plan, raw["steps"])
    original = copy.deepcopy(authored)

    definitions = load_definitions(document, authored)

    assert set(definitions) == {"source"}
    definition = definitions["source"]
    assert definition.path == document
    assert definition.raw == raw
    assert definition.benchmark_identity == benchmark_identity(
        raw, definition.protocols, authored.get("observations", {})
    )
    assert authored == original


@pytest.mark.parametrize("comparison", ["code", "workflow"])
def test_single_mode_refuses_comparison_selector(tmp_path, comparison):
    document, _ = _files(tmp_path)
    raw = json.loads(document.read_text())
    plan = raw["measurement"]
    plan.update(mode="single", comparison=comparison)
    plan["source"] = plan.pop("arms")["before"]
    with pytest.raises(MeasurementSpecError, match="comparison"):
        parse_measurement(plan, raw["steps"])
