"""Tests for ``causalab.io.plots.causal_graph``.

Developer/debug visualization helpers for ``CausalModel``. Two output families:

- Dash/Cytoscape interactive graphs (``build_*_app`` / ``display_*``)
- Matplotlib/NetworkX static figures (``build_*_figure`` / ``print_*``)

Each family is factored into pure builders (which return data/objects) and
thin ``display_*`` / ``print_*`` wrappers that actually launch a server or
call ``plt.show``. The tests below exercise the builders directly without
binding a port or opening a window.

Nothing in ``runner`` or ``analyses`` imports this module — call sites are
the IOI / MCQA demo notebooks. A wrong value here breaks interactive visual
inspection, not the pipeline.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
from dash import Dash

from causalab.io.plots.causal_graph import (
    DEFAULT_COLORS,
    _get_descendants,
    build_edges,
    build_forward_pass_app,
    build_interchange_app,
    build_interchange_subgraphs,
    build_setting_figure,
    build_structure_app,
    build_structure_figure,
    build_stylesheet,
    build_variable_nodes,
    classify_forward_node,
    display_forward_pass,
    display_interchange,
    display_structure,
    make_forward_pass_onload,
    make_general_on_moving_nodes,
    make_interchange_onload,
    make_simple_on_moving_nodes,
    print_setting,
    print_structure,
)

from tests._helpers.tiny import tiny_chain_model

# Force Agg backend so plt.show never opens a window in headless CI.
matplotlib.use("Agg")


@pytest.fixture
def model():
    """3-node A→B→C CausalModel; ``print_pos`` is auto-populated by __init__."""
    return tiny_chain_model()


@pytest.fixture(autouse=True)
def _no_window():
    """Close all matplotlib figures after each test to keep state clean."""
    yield
    plt.close("all")


@pytest.fixture
def no_dash_run(monkeypatch):
    """Replace ``Dash.run`` with a no-op so display_*() never binds a port."""
    monkeypatch.setattr(Dash, "run", lambda self, *args, **kwargs: None)


class TestDefaultColorsUnit:
    """``DEFAULT_COLORS`` palette constants — callers depend on these names."""

    pytestmark = pytest.mark.unit

    def test_five_hex_colors_present(self):
        for name in (
            "BASE_INPUT",
            "BASE_OUTPUT",
            "SOURCE_INPUT",
            "SOURCE_OUTPUT",
            "COUNTERFACTUAL_OUTPUT",
        ):
            value = getattr(DEFAULT_COLORS, name)
            assert isinstance(value, str)
            assert value.startswith("#")
            assert len(value) == 7


class TestGetDescendantsUnit:
    """``_get_descendants`` BFS — backs intervention-only and counterfactual node sets."""

    pytestmark = pytest.mark.unit

    def test_intervention_descendants_strict(self, model):
        d = _get_descendants(model, {"B": 1}, strict=True)
        assert "B" in d
        assert "C" in d
        assert "raw_output" in d
        # A is an ancestor of B — must not appear.
        assert "A" not in d

    def test_intervention_descendants_non_strict(self, model):
        d = _get_descendants(model, {"B": 1}, strict=False)
        assert "B" in d
        assert "C" in d


class TestBuildVariableNodesUnit:
    """``build_variable_nodes`` — Cytoscape node list, one per variable."""

    pytestmark = pytest.mark.unit

    def test_one_node_per_variable(self, model):
        nodes = build_variable_nodes(model)
        ids = {n["data"]["id"] for n in nodes}
        assert ids == set(model.variables)

    def test_suffix_applied_to_every_id(self, model):
        nodes = build_variable_nodes(model, suffix="-source-0")
        for n in nodes:
            assert n["data"]["id"].endswith("-source-0")

    def test_class_string_is_variable(self, model):
        nodes = build_variable_nodes(model)
        for n in nodes:
            assert n["classes"] == "variable"


class TestBuildEdgesUnit:
    """``build_edges`` — Cytoscape edges directly mirroring ``model.parents``."""

    pytestmark = pytest.mark.unit

    def test_edge_set_matches_model_parents(self, model):
        edges = build_edges(model)
        pairs = {(e["data"]["source"], e["data"]["target"]) for e in edges}
        expected = {
            (parent, child)
            for child in model.variables
            for parent in model.parents[child]
        }
        assert pairs == expected

    def test_suffix_applied_to_source_and_target(self, model):
        edges = build_edges(model, suffix="-source-0")
        for e in edges:
            assert e["data"]["source"].endswith("-source-0")
            assert e["data"]["target"].endswith("-source-0")


class TestBuildInterchangeSubgraphsUnit:
    """``build_interchange_subgraphs`` — N source DAGs + N interchange edges."""

    pytestmark = pytest.mark.unit

    def test_subgraph_count_matches_keys(self, model):
        cf = model.new_trace({"A": 1})
        nodes, edges = build_interchange_subgraphs(model, {"B": cf, "C": cf})
        assert len(nodes) == len(model.variables) * 2  # 2 cf keys
        base_edge_count = sum(len(model.parents[v]) for v in model.variables)
        assert len(edges) == (base_edge_count + 1) * 2

    def test_interchange_edges_classified(self, model):
        cf = model.new_trace({"A": 1})
        _, edges = build_interchange_subgraphs(model, {"B": cf})
        ic = [e for e in edges if e.get("classes") == "interchange_edge"]
        assert len(ic) == 1
        # Interchange edge: source is the suffixed key, target is the base key.
        assert ic[0]["data"]["target"] == "B"
        assert ic[0]["data"]["source"].endswith("-source-0")


class TestBuildStylesheetUnit:
    """``build_stylesheet`` — shared Cytoscape stylesheet, flag-driven branches."""

    pytestmark = pytest.mark.unit

    def test_structure_view_variable_has_no_white_background(self):
        sheet = build_stylesheet(
            DEFAULT_COLORS,
            hide_variable_when_value_present=False,
            include_interchange_edge=False,
        )
        variable_style = next(s for s in sheet if s["selector"] == ".variable")["style"]
        assert "background-color" not in variable_style

    def test_forward_view_variable_is_white(self):
        sheet = build_stylesheet(
            DEFAULT_COLORS,
            hide_variable_when_value_present=True,
            include_interchange_edge=False,
        )
        variable_style = next(s for s in sheet if s["selector"] == ".variable")["style"]
        assert variable_style.get("background-color") == "white"

    def test_interchange_edge_selector_conditional(self):
        with_edge = build_stylesheet(
            DEFAULT_COLORS,
            hide_variable_when_value_present=True,
            include_interchange_edge=True,
        )
        without_edge = build_stylesheet(
            DEFAULT_COLORS,
            hide_variable_when_value_present=True,
            include_interchange_edge=False,
        )
        selectors_with = {s["selector"] for s in with_edge}
        selectors_without = {s["selector"] for s in without_edge}
        assert ".interchange_edge" in selectors_with
        assert ".interchange_edge" not in selectors_without

    def test_custom_palette_propagated_to_style_entries(self):
        class CustomColors:
            BASE_INPUT = "#000001"
            BASE_OUTPUT = "#000002"
            SOURCE_INPUT = "#000003"
            SOURCE_OUTPUT = "#000004"
            COUNTERFACTUAL_OUTPUT = "#000005"

        sheet = build_stylesheet(
            CustomColors,
            hide_variable_when_value_present=True,
            include_interchange_edge=True,
        )
        rendered = str(sheet)
        for name in (
            "BASE_INPUT",
            "BASE_OUTPUT",
            "SOURCE_INPUT",
            "SOURCE_OUTPUT",
            "COUNTERFACTUAL_OUTPUT",
        ):
            assert getattr(CustomColors, name) in rendered


class TestClassifyForwardNodeUnit:
    """``classify_forward_node`` — class string per the precedence rule."""

    pytestmark = pytest.mark.unit

    def test_intervention_beats_base(self):
        assert (
            classify_forward_node(
                "B",
                inputs={"A": 0},
                intervention={"B": 99},
                intervention_only=["B", "C"],
                counterfactual=["B", "C"],
            )
            == "source_input"
        )

    def test_base_beats_intervention_only(self):
        assert (
            classify_forward_node(
                "A",
                inputs={"A": 0},
                intervention={"B": 99},
                intervention_only=["B"],
                counterfactual=["B"],
            )
            == "base_input"
        )

    def test_intervention_only_beats_counterfactual(self):
        assert (
            classify_forward_node(
                "C",
                inputs={"A": 0},
                intervention={"B": 99},
                intervention_only=["B", "C"],
                counterfactual=["B", "C"],
            )
            == "source_value"
        )

    def test_counterfactual_beats_default(self):
        assert (
            classify_forward_node(
                "C",
                inputs={"A": 0},
                intervention={"B": 99},
                intervention_only=[],
                counterfactual=["C"],
            )
            == "counterfactual_value"
        )

    def test_default_base_value(self):
        assert (
            classify_forward_node(
                "C",
                inputs={"A": 0},
                intervention={},
                intervention_only=[],
                counterfactual=[],
            )
            == "base_value"
        )


class TestMakeForwardPassOnloadUnit:
    """``make_forward_pass_onload`` — callback that paints value nodes onto the DAG."""

    pytestmark = pytest.mark.unit

    def test_appends_one_value_node_per_variable(self, model):
        inputs = {"A": 0}
        outputs = model.new_trace(inputs).to_dict()
        onload = make_forward_pass_onload(
            model, outputs=outputs, inputs=inputs, intervention={}
        )
        elements = build_variable_nodes(model) + build_edges(model)
        _, new_elements = onload("ignored", list(elements))

        value_ids = {
            e["data"]["id"] for e in new_elements if e["data"]["id"].endswith("-value")
        }
        assert value_ids == {f"{v}-value" for v in model.variables}

    def test_no_intervention_only_emits_base_classes(self, model):
        inputs = {"A": 0}
        outputs = model.new_trace(inputs).to_dict()
        onload = make_forward_pass_onload(
            model, outputs=outputs, inputs=inputs, intervention={}
        )
        elements = build_variable_nodes(model) + build_edges(model)
        _, new_elements = onload("ignored", list(elements))
        for e in new_elements:
            if e["data"]["id"].endswith("-value"):
                assert e["classes"] in {"base_input", "base_value"}

    def test_mid_dag_intervention_marks_source_node(self, model):
        inputs = {"A": 0}
        intervention = {"B": 1}
        outputs = model.new_trace(inputs)
        outputs.intervene("B", 1)
        outputs_dict = outputs.to_dict()

        onload = make_forward_pass_onload(
            model, outputs=outputs_dict, inputs=inputs, intervention=intervention
        )
        elements = build_variable_nodes(model) + build_edges(model)
        _, new_elements = onload("ignored", list(elements))

        b_value = next(e for e in new_elements if e["data"]["id"] == "B-value")
        assert b_value["classes"] == "source_input"


class TestMakeInterchangeOnloadUnit:
    """``make_interchange_onload`` — labels base + per-key source DAGs."""

    pytestmark = pytest.mark.unit

    def test_per_source_value_nodes_carry_suffix(self, model):
        cf = model.new_trace({"A": 1})
        cf_traces = {"B": cf}
        counterfactual_inputs = {"B": {"A": 1}}
        inputs = {"A": 0}
        outputs = model.new_trace(inputs).to_dict()
        onload = make_interchange_onload(
            model,
            outputs=outputs,
            inputs=inputs,
            counterfactual_inputs=counterfactual_inputs,
            cf_traces=cf_traces,
        )

        base_nodes = build_variable_nodes(model)
        base_edges = build_edges(model)
        src_nodes, src_edges = build_interchange_subgraphs(model, counterfactual_inputs)
        elements = base_nodes + src_nodes + base_edges + src_edges
        _, new_elements = onload("ignored", list(elements))

        suffix_value_ids = [
            e["data"]["id"]
            for e in new_elements
            if e["data"]["id"].endswith("-source-0-value")
        ]
        assert len(suffix_value_ids) == len(model.variables)

    def test_two_source_dags(self, model):
        cf = model.new_trace({"A": 1})
        cf_traces = {"B": cf, "C": cf}
        counterfactual_inputs = {"B": {"A": 1}, "C": {"A": 1}}
        inputs = {"A": 0}
        outputs = model.new_trace(inputs).to_dict()
        onload = make_interchange_onload(
            model,
            outputs=outputs,
            inputs=inputs,
            counterfactual_inputs=counterfactual_inputs,
            cf_traces=cf_traces,
        )

        base_nodes = build_variable_nodes(model)
        base_edges = build_edges(model)
        src_nodes, src_edges = build_interchange_subgraphs(model, counterfactual_inputs)
        elements = base_nodes + src_nodes + base_edges + src_edges
        _, new_elements = onload("ignored", list(elements))

        # One value node per variable per source DAG (2 DAGs).
        suffix0 = [
            e for e in new_elements if e["data"]["id"].endswith("-source-0-value")
        ]
        suffix1 = [
            e for e in new_elements if e["data"]["id"].endswith("-source-1-value")
        ]
        assert len(suffix0) == len(model.variables)
        assert len(suffix1) == len(model.variables)


class TestMakeOnMovingNodesUnit:
    """``make_simple_on_moving_nodes`` / ``make_general_on_moving_nodes`` —
    position-sync callbacks that pin variable nodes to their value-node partners."""

    pytestmark = pytest.mark.unit

    def test_simple_syncs_variable_to_value_position(self, model):
        elements = [
            {"data": {"id": "A", "label": "A"}, "position": {"x": 0, "y": 0}},
            {"data": {"id": "A-value", "label": 0}, "position": {"x": 50, "y": 50}},
        ]
        cb = make_simple_on_moving_nodes(model)
        result = cb(list(elements))
        a_node = next(e for e in result if e["data"]["id"] == "A")
        assert a_node["position"] == {"x": 50, "y": 50}

    def test_simple_leaves_variable_without_value_alone(self, model):
        elements = [
            {"data": {"id": "A", "label": "A"}, "position": {"x": 0, "y": 0}},
        ]
        cb = make_simple_on_moving_nodes(model)
        result = cb(list(elements))
        a_node = next(e for e in result if e["data"]["id"] == "A")
        assert a_node["position"] == {"x": 0, "y": 0}

    def test_general_returns_callable(self, model):
        assert callable(make_general_on_moving_nodes())


class TestBuildStructureAppUnit:
    """``build_structure_app`` — no callbacks; layout has only the Cytoscape element."""

    pytestmark = pytest.mark.unit

    def test_returns_dash_app(self, model):
        assert isinstance(build_structure_app(model), Dash)

    def test_no_hidden_onload_trigger_in_layout(self, model):
        app = build_structure_app(model)
        layout = app.layout
        ids = [child.id for child in layout.children if hasattr(child, "id")]
        assert "hidden-onload-trigger" not in ids


class TestBuildForwardPassAppUnit:
    """``build_forward_pass_app`` — wires onload + on-moving callbacks."""

    pytestmark = pytest.mark.unit

    def test_layout_has_onload_trigger(self, model):
        app = build_forward_pass_app(model, {"A": 0})
        layout = app.layout
        ids = [child.id for child in layout.children if hasattr(child, "id")]
        assert "hidden-onload-trigger" in ids

    def test_callbacks_registered(self, model):
        app = build_forward_pass_app(model, {"A": 0})
        # Two callbacks: onload and on_moving_nodes.
        assert len(app.callback_map) == 2


class TestBuildInterchangeAppUnit:
    """``build_interchange_app`` — wires onload + on-moving callbacks across base+source DAGs."""

    pytestmark = pytest.mark.unit

    def test_returns_dash_app(self, model):
        cf = model.new_trace({"A": 1})
        assert isinstance(build_interchange_app(model, {"A": 0}, {"B": cf}), Dash)

    def test_layout_has_onload_trigger(self, model):
        cf = model.new_trace({"A": 1})
        app = build_interchange_app(model, {"A": 0}, {"B": cf})
        layout = app.layout
        ids = [child.id for child in layout.children if hasattr(child, "id")]
        assert "hidden-onload-trigger" in ids

    def test_callbacks_registered(self, model):
        cf = model.new_trace({"A": 1})
        app = build_interchange_app(model, {"A": 0}, {"B": cf})
        assert len(app.callback_map) == 2


class TestBuildStructureFigureUnit:
    """``build_structure_figure`` — matplotlib Figure for the DAG."""

    pytestmark = pytest.mark.unit

    def test_returns_figure(self, model):
        fig = build_structure_figure(model)
        assert isinstance(fig, plt.Figure)


class TestBuildSettingFigureUnit:
    """``build_setting_figure`` — DAG figure with values relabeled into node strings."""

    pytestmark = pytest.mark.unit

    def test_returns_figure_for_full_setting(self, model):
        setting = model.new_trace({"A": 0}).to_dict()
        fig = build_setting_figure(model, setting)
        assert isinstance(fig, plt.Figure)


class TestDisplayWrappersUnit:
    """``display_*`` wrappers — call ``build_*_app(...).run()``."""

    pytestmark = pytest.mark.unit

    def test_display_structure_delegates_to_dash_run(self, model, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            Dash, "run", lambda self, *a, **k: called.__setitem__("n", called["n"] + 1)
        )
        display_structure(model)
        assert called["n"] == 1

    def test_display_forward_pass_delegates_to_dash_run(self, model, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            Dash, "run", lambda self, *a, **k: called.__setitem__("n", called["n"] + 1)
        )
        display_forward_pass(model, {"A": 0})
        assert called["n"] == 1

    def test_display_interchange_delegates_to_dash_run(self, model, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            Dash, "run", lambda self, *a, **k: called.__setitem__("n", called["n"] + 1)
        )
        cf = model.new_trace({"A": 1})
        display_interchange(model, {"A": 0}, {"B": cf})
        assert called["n"] == 1


class TestPrintWrappersUnit:
    """``print_*`` wrappers — call ``build_*_figure(...)`` then ``plt.show()``."""

    pytestmark = pytest.mark.unit

    def test_print_structure_calls_plt_show(self, model, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            plt, "show", lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        )
        print_structure(model)
        assert called["n"] == 1

    def test_print_setting_calls_plt_show(self, model, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            plt, "show", lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        )
        setting = model.new_trace({"A": 1}).to_dict()
        print_setting(model, setting)
        assert called["n"] == 1
