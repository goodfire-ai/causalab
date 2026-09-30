"""Graph views use simultaneous interventions and leave unused lazy nodes alone."""

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism, require
from causalab.io.plots.causal_graph import (
    build_forward_pass_app,
    build_interchange_app,
    build_interchange_subgraphs,
    build_variable_nodes,
    make_interchange_onload,
)

pytestmark = pytest.mark.unit


def _model():
    @mechanism
    def equations(x: Dom([0, 1]), y: Dom([0, 1])):
        bad = V(1 // (x + y - 1), domain=Dom(int), lazy=True)  # noqa: F841
        require(x + y == 1, error="must remain one-hot")
        raw_input = V(f"{x},{y}", domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))
        return raw_output

    return CausalModel(equations)


@pytest.mark.parametrize("viewer", ["forward", "interchange"])
def test_viewers_apply_joint_changes_without_reading_unused_lazy_nodes(viewer):
    model = _model()
    base = model.new_trace({"x": 0, "y": 1})
    donor = model.new_trace({"x": 1, "y": 0})
    if viewer == "forward":
        app = build_forward_pass_app(model, {"x": 0, "y": 1}, {"x": 1, "y": 0})
    else:
        app = build_interchange_app(model, base, {"x": donor, "y": donor})
    assert app is not None
    assert base["raw_output"] == "0"
    assert "bad" not in base
    assert "bad" not in donor


def test_interchange_onload_does_not_materialize_lazy_source_nodes():
    model = _model()
    base = model.new_trace({"x": 0, "y": 1})
    donor = model.new_trace({"x": 1, "y": 0})
    sources = {"x": donor, "y": donor}
    result = model.run_interchange(base, sources)
    source_nodes, source_edges = build_interchange_subgraphs(model, sources)
    onload = make_interchange_onload(
        model,
        outputs=result.snapshot(),
        inputs=base,
        counterfactual_inputs=sources,
        cf_traces=sources,
    )
    _, elements = onload("", build_variable_nodes(model) + source_nodes + source_edges)
    labels = {
        element["data"]["id"]: element["data"].get("label") for element in elements
    }
    assert labels["raw_output-value"] == "1"
    assert labels["raw_output-source-0-value"] == "1"
    # the source DAG computes its missing values on a copy: bad raises there
    # and is labelled with the error, while the donor's own cache is untouched
    assert labels["bad-source-0-value"] == "error: ZeroDivisionError"
    assert "bad" not in donor


def _lazy_model():
    @mechanism
    def equations(x: Dom([0, 1]), y: Dom([0, 1])):
        bad = V(1 // (x + y - 1), domain=Dom(int), lazy=True)  # noqa: F841
        raw_input = V(f"{x},{y}", domain=Dom(str), lazy=True)  # noqa: F841
        raw_output = V(str(x), domain=Dom(str), lazy=True)
        return raw_output

    return CausalModel(equations)


def test_views_label_lazy_nodes_that_compute_and_name_the_ones_that_raise():
    """graph_walk renders its prompt and answer through lazy nodes; a view
    that shows only cached values would leave both unlabelled."""
    from causalab.io.plots.causal_graph import displayed_values

    model = _lazy_model()
    trace = model.new_trace({"x": 1, "y": 0})
    values = displayed_values(model, trace)
    assert values["raw_input"] == "1,0"
    assert values["raw_output"] == "1"
    assert values["bad"] == "error: ZeroDivisionError"
    # computed on a copy: the viewed trace keeps its cache as it was
    assert "raw_input" not in trace
    assert "raw_output" not in trace


def test_interchange_onload_computes_source_values_once(monkeypatch):
    """Every page load or trigger calls onload; the source DAGs' values are
    computed when the callback is built, as the base DAG's outputs are."""
    import causalab.io.plots.causal_graph as causal_graph

    calls = []
    real = causal_graph.displayed_values

    def counting(model, trace):
        calls.append(trace)
        return real(model, trace)

    monkeypatch.setattr(causal_graph, "displayed_values", counting)
    model = _lazy_model()
    base = model.new_trace({"x": 0, "y": 1})
    donor = model.new_trace({"x": 1, "y": 0})
    sources = {"x": donor}
    source_nodes, source_edges = build_interchange_subgraphs(model, sources)
    onload = make_interchange_onload(
        model,
        outputs=model.run_interchange(base, sources).snapshot(),
        inputs=base,
        counterfactual_inputs=sources,
        cf_traces=sources,
    )
    for _ in range(3):
        _, elements = onload(
            "", build_variable_nodes(model) + source_nodes + source_edges
        )
    assert len(calls) == 1
    labels = {e["data"]["id"]: e["data"].get("label") for e in elements}
    assert labels["raw_output-source-0-value"] == "1"
