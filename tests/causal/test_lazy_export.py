"""Labels and saved examples must not force unselected lazy equations."""

import json

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.causal.counterfactuals import label_data_with_variables
from causalab.causal.model import CausalTrace
from causalab.io.counterfactuals import (
    load_counterfactual_examples,
    save_counterfactual_examples,
)

pytestmark = pytest.mark.unit


def _lazy_model():
    @mechanism
    def equations(enabled: Dom(bool), divisor: Dom([0, 1])):
        bad = V(1 // divisor, domain=Dom(int), lazy=True)
        raw_input = V(str(enabled), domain=Dom(str))  # noqa: F841
        raw_output = V(str(bad if enabled else 5), domain=Dom(str))
        return raw_output

    return CausalModel(equations)


def test_counterfactual_labels_preserve_unselected_lazy_branches():
    model = _lazy_model()
    base = model.new_trace({"enabled": False, "divisor": 0})
    donor = model.new_trace({"enabled": False, "divisor": 1})
    labeled = model.label_counterfactual_data(
        [{"input": base, "counterfactual_inputs": [donor]}], ["enabled"]
    )
    assert labeled[0]["label"] == "5"
    assert labeled[0]["setting"]["raw_output"] == "5"
    assert "bad" not in labeled[0]["setting"]
    assert "bad" not in base
    assert "bad" not in donor


def test_variable_labels_do_not_materialize_unrelated_lazy_equations():
    model = _lazy_model()
    trace = model.new_trace({"enabled": False, "divisor": 0})
    labeled, classes = label_data_with_variables(
        model, [{"input": trace}], ["raw_output"]
    )
    assert classes == {"5": 0}
    assert labeled[0]["label"] == 0
    assert labeled[0]["input"]["raw_output"] == "5"
    assert "bad" not in labeled[0]["input"]


def test_saving_preserves_lazy_branches_and_deleted_override_caches(tmp_path):
    model = _lazy_model()
    trace = model.new_trace({"enabled": False, "divisor": 0})
    trace["enabled"] = False
    del trace["enabled"]
    path = str(tmp_path / "lazy.json")
    save_counterfactual_examples(
        [{"input": trace, "counterfactual_inputs": [trace.copy()]}], path
    )
    assert "enabled" not in trace
    assert "bad" not in trace
    saved = json.loads((tmp_path / "lazy.json").read_text())[0]["input"]
    assert saved["values"]["enabled"] is False
    assert saved["values"]["divisor"] == 0
    assert "bad" not in saved["values"]
    loaded = load_counterfactual_examples(path, model)[0]
    for restored in (loaded["input"], *loaded["counterfactual_inputs"]):
        assert restored["raw_output"] == "5"
        assert "bad" not in restored
        del restored["enabled"]
        assert restored["enabled"] is False


def test_snapshot_recovers_only_deleted_interventions_and_copies_values():
    model = _lazy_model()
    trace = model.new_trace({"enabled": False, "divisor": 0})
    trace["bad"] = 7
    del trace["bad"]
    snapshot = trace.snapshot()
    assert snapshot["bad"] == 7
    assert "bad" not in trace
    snapshot["enabled"] = True
    assert trace["enabled"] is False


def test_value_only_traces_snapshot_and_save_deleted_overrides(tmp_path):
    trace = CausalTrace.from_values({"raw_input": "prompt", "tokens": [1, 2]})
    del trace["tokens"]
    snapshot = trace.snapshot()
    snapshot["tokens"].append(3)
    assert trace["tokens"] == [1, 2]
    del trace["tokens"]
    path = str(tmp_path / "values.json")
    save_counterfactual_examples([{"input": trace, "counterfactual_inputs": []}], path)
    saved = json.loads((tmp_path / "values.json").read_text())[0]["input"]
    assert saved["values"] == {"raw_input": "prompt", "tokens": [1, 2]}
    assert set(saved["interventions"]) == {"raw_input", "tokens"}
    assert "tokens" not in trace


def test_explicit_materialization_and_selected_branches_still_evaluate():
    model = _lazy_model()
    trace = model.new_trace({"enabled": False, "divisor": 0})
    with pytest.raises(ZeroDivisionError):
        trace.to_dict()
    donor = model.new_trace({"enabled": True, "divisor": 1})
    with pytest.raises(ZeroDivisionError):
        model.label_counterfactual_data(
            [{"input": trace, "counterfactual_inputs": [donor]}], ["enabled"]
        )
    assert trace["enabled"] is False
