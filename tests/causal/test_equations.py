"""Test model definitions and interventions, including live notebook source."""

import random

import pytest

from causalab.causal import (
    DefinitionError,
    Dom,
    DomainError,
    Exo,
    FamilyDom,
    V,
    family,
    mechanism,
    require,
    submodel,
)
from causalab.causal.model import CausalModel

pytestmark = pytest.mark.unit


def test_gate_union_and_whole_equation_override():
    @mechanism
    def equations(enabled: Dom(bool), x: Dom(range(3)), y: Dom(range(3))):
        if enabled:
            optional = V(x)
        else:
            optional = V(None)
        result = V(y if optional is None else optional + y)
        raw_input = V(f"{enabled}:{x}:{y}")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["optional"]) == {"enabled", "x"}
    assert set(model.parents["result"]) == {"optional", "y"}
    trace = model.new_trace({"enabled": False, "x": 1, "y": 2})
    assert trace["optional"] is None
    trace["optional"] = 1
    assert trace["result"] == 3
    trace["enabled"] = True
    trace["x"] = 2
    assert trace["optional"] == 1
    assert model.new_trace({"enabled": False, "x": 1, "y": 2})["result"] == 2
    with pytest.raises(DomainError, match="optional"):
        trace["optional"] = 42


def test_input_and_computed_validation():
    @mechanism
    def equations(x: Dom(range(3))):
        result = V(x * 2, domain=Dom(range(4)))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    with pytest.raises(DomainError, match="x"):
        model.new_trace({"x": 4})
    with pytest.raises(DomainError, match="result"):
        model.new_trace({"x": 2})
    trace = model.new_trace({"x": 1})
    with pytest.raises(DomainError):
        trace["x"] = 2
    assert trace["x"] == 1  # failed intervention did not corrupt the trace


def test_bounded_steps_and_exact_bound():
    bound = 3

    @mechanism
    def equations(start: Dom(range(5))):
        steps = family(size=bound + 1, domain=Dom(range(5)))
        steps[0] = start
        for i in range(bound):
            steps[i + 1] = steps[i] - 1 if steps[i] > 0 else steps[i]
        require(steps[bound] == 0, error="Step bound exceeded")
        raw_input = V(str(start))  # noqa: F841
        raw_output = V(str(steps[bound]))
        return raw_output

    model = CausalModel(equations)
    for start in range(4):
        trace = model.new_trace({"start": start})
        assert trace["steps[3]"] == 0
        assert [trace[f"steps[{i}]"] for i in range(4)] == [
            max(0, start - i) for i in range(4)
        ]
    with pytest.raises(ValueError, match="Step bound exceeded"):
        model.new_trace({"start": 4})
    assert model.parents["steps[2]"] == ["steps[1]"]


def test_indexed_inputs_and_union_of_selector_edges():
    @mechanism
    def equations(xs: FamilyDom(Dom(range(3)), size=3), index: Dom([0, 2])):
        selected = V(xs[index])
        raw_input = V(str(xs), domain=Dom(str))  # noqa: F841
        raw_output = V(str(selected))  # noqa: F841
        return selected

    model = CausalModel(equations)
    assert set(model.parents["selected"]) == {"index", "xs[0]", "xs[2]"}
    trace = model.new_trace({"xs": [1, 0, 2], "index": 2})
    assert trace["selected"] == 2
    trace["xs[2]"] = 1
    assert trace["selected"] == 1


def test_nested_namespaces_and_aliases():
    @submodel
    def compare(a, b):
        equal = V(a == b)
        return equal

    @submodel
    def inner(a, b):
        comparison = compare(a, b)
        return comparison

    @mechanism
    def equations(a: Dom(range(2)), b: Dom(range(2)), c: Dom(range(2))):
        left = inner(a, b)
        right = compare(b, c)
        result = V(left == right)
        raw_input = V(f"{a}{b}{c}")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.variables) == {
        "a",
        "b",
        "c",
        "left.comparison.equal",
        "right.equal",
        "result",
        "raw_input",
        "raw_output",
    }
    assert set(model.parents["left.comparison.equal"]) == {"a", "b"}
    trace = model.new_trace({"a": 0, "b": 0, "c": 1})
    assert trace["result"] is False
    trace["left.comparison.equal"] = False
    assert trace["result"] is True


def test_noise_is_explicit_and_fixed_for_interventions_and_enumeration():
    @mechanism
    def equations(x: Dom(range(2)), noise: Exo(Dom(range(10)))):
        result = V(x + noise)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    with pytest.raises(ValueError, match="noise"):
        model.new_trace({"x": 1})
    with pytest.raises(ValueError, match="fixed noise"):
        model.enumerate_inputs()
    assert model.count_inputs(noise={"noise": 3}) == 2
    assert [t["result"] for t in model.enumerate_inputs(noise={"noise": 3})] == [3, 4]
    rng = random.Random(6)
    a, b = model.sample_input(rng=rng), model.sample_input(rng=rng)
    result = model.run_interchange(a, {"x": b})
    assert result["noise"] == a["noise"]
    assert result["result"] == b["x"] + a["noise"]


def test_private_configuration_snapshot_and_explicit_domain_for_interventions():
    lookup = {"a": 0, "b": 1}

    def helper(x):
        return lookup[x]

    @mechanism
    def equations(x: Dom(["a", "b"])):
        result = V(helper(x), domain=Dom(range(4)))
        raw_input = V(x)  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    lookup["a"] = 3
    assert model.new_trace({"x": "a"})["result"] == 0
    trace = model.new_trace({"x": "b"})
    trace["result"] = 3
    assert trace["raw_output"] == "3"


def test_lazy_descendants_invalidate_and_copies_are_independent():
    @mechanism
    def equations(x: Dom(range(3))):
        slow = V(
            [x], domain=Dom.sequence(Dom(range(3)), length=1, container=list), lazy=True
        )
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(slow), lazy=True)  # noqa: F841
        return slow

    model = CausalModel(equations)
    trace = model.new_trace({"x": 0})
    assert "slow" not in trace
    assert trace["raw_output"] == "[0]"
    other = trace.copy()
    other["slow"].append(2)
    assert trace["slow"] == [0]
    trace["x"] = 1
    assert trace["raw_output"] == "[1]"


def test_conditional_missing_definition_is_rejected():
    @mechanism
    def equations(x: Dom(bool)):
        if x:
            result = V(1)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="both branches"):
        CausalModel(equations)


def test_undefined_family_and_data_dependent_bound_are_rejected():
    @mechanism
    def missing(x: Dom(range(3))):
        steps = family(size=2)
        steps[0] = x
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(steps[0]))
        return raw_output

    @mechanism
    def unbounded(x: Dom(range(3))):
        steps = family(size=3)
        for i in range(x):
            steps[i] = x
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(steps[0]))
        return raw_output

    with pytest.raises(DefinitionError, match="not defined"):
        CausalModel(missing)
    with pytest.raises(DefinitionError, match="fixed build"):
        CausalModel(unbounded)


def test_jupyter_cell_redefinition_keeps_old_model_and_configuration():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell()
    setup = """from causalab.causal import mechanism, V, Dom
from causalab.causal.model import CausalModel
settings = {'offset': 1}
@mechanism
def equations(x: Dom(range(3))):
    y = V(x + settings['offset'])
    raw_input = V(str(x))
    raw_output = V(str(y))
    return y
model = CausalModel(equations)
"""
    result = shell.run_cell(setup, store_history=True)
    assert result.success
    first = shell.user_ns["model"]
    result = shell.run_cell(
        setup.replace("'offset': 1", "'offset': 2"), store_history=True
    )
    assert result.success
    assert first.new_trace({"x": 0})["y"] == 1
    assert shell.user_ns["model"].new_trace({"x": 0})["y"] == 2


def test_atomic_intervention_checks_execution_constraint_after_all_overrides():
    @mechanism
    def equations(x: Dom(range(3)), y: Dom(range(3))):
        require(x + y == 2, error="sum must remain two")
        raw_input = V(f"{x}:{y}", domain=Dom(str))  # noqa: F841
        raw_output = V(x + y)
        return raw_output

    model = CausalModel(equations)
    trace = model.new_trace({"x": 1, "y": 1})
    with pytest.raises(ValueError, match="sum must remain two"):
        trace["x"] = 0
    assert trace["x"] == 1
    donor = model.new_trace({"x": 0, "y": 2})
    result = model.run_interchange(trace, {"x": donor, "y": donor})
    assert result["x"] == 0 and result["y"] == 2


def test_selector_edges_use_allowed_indices_even_with_huge_member_domains():
    @mechanism
    def equations(xs: FamilyDom(Dom(range(10**6)), size=3), index: Dom([0, 2])):
        result = V(xs[index], domain=Dom(range(10**6)))
        raw_input = V(str(index), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"index", "xs[0]", "xs[2]"}
    assert model.new_trace({"xs": [10, 20, 30], "index": 2})["result"] == 30


def test_large_helper_needs_sound_explicit_domain():
    def helper(x):
        return x // 2

    @mechanism
    def equations(x: Dom(range(10**6))):
        result = V(helper(x))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Cannot infer a sound domain"):
        CausalModel(equations)


def test_never_reached_exposed_node_remains_and_is_not_an_input():
    @mechanism
    def equations(x: Dom(range(2))):
        diagnostic = V(42)  # noqa: F841 — retained as an exposed graph node
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(x)
        return raw_output

    model = CausalModel(equations)
    assert "diagnostic" in model.variables
    assert model.inputs == ["x"]
    assert model.new_trace({"x": 1})["diagnostic"] == 42


def test_cycle_and_duplicate_equations_are_rejected():
    @mechanism
    def cycle(x: Dom(range(2))):
        nodes = family(size=2)
        nodes[0] = nodes[1]
        nodes[1] = nodes[0]
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(nodes[1]), domain=Dom(str))
        return raw_output

    @mechanism
    def duplicate(x: Dom(range(2))):
        nodes = family(size=1)
        nodes[0] = x
        nodes[0] = x + 1
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(nodes[0]), domain=Dom(str))
        return raw_output

    with pytest.raises(DefinitionError, match="acyclic"):
        CausalModel(cycle)
    with pytest.raises(DefinitionError, match="more than once"):
        CausalModel(duplicate)


def test_direct_hidden_randomness_is_rejected():
    @mechanism
    def equations(x: Dom(range(2))):
        result = V(x + random.randint(0, 1))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="explicit Exo input"):
        CausalModel(equations)


def test_walkthrough_notebook_cells_execute():
    import json
    from pathlib import Path

    from IPython.core.interactiveshell import InteractiveShell

    notebook = Path(__file__).parents[2] / "demos/causal_models/defining_models.ipynb"
    shell = InteractiveShell()
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            result = shell.run_cell("".join(cell["source"]), store_history=True)
            assert result.success


def test_arithmetic_examples_have_same_observations_and_different_interventions():
    from demos.causal_models.arithmetic_demos import make_addition

    left, right = make_addition("left"), make_addition("right")
    for trace in left.enumerate_inputs():
        inputs = {k: trace[k] for k in left.inputs}
        assert trace["Y"] == right.new_trace(inputs)["Y"]
    base = {"A": 1, "B": 2, "C": 3}
    donor = {"A": 5, "B": 2, "C": 3}
    assert (
        left.run_interchange(left.new_trace(base), {"S": left.new_trace(donor)})["Y"]
        == 0
    )
    assert (
        right.run_interchange(right.new_trace(base), {"S": right.new_trace(donor)})["Y"]
        == 6
    )


def test_arithmetic_examples_take_the_modulus():
    """The onboarding tutorial's ``demos/onboarding_tutorial/03_causal_model.md``
    builds the two models mod 12; the digits and the sums follow the modulus."""
    from demos.causal_models.arithmetic_demos import make_addition

    left = make_addition("left", modulus=12)
    assert left.count_inputs() == 12**3
    assert left.new_trace({"A": 11, "B": 11, "C": 11})["Y"] == 33 % 12
    assert sorted(left.values["S"]) == list(range(12))


def test_unproven_conditional_edges_are_refused_instead_of_added():
    @mechanism
    def equations(x: Dom(range(10**8)), y: Dom(range(2))):
        result = V(y if x < 0 else 0, domain=Dom(range(2)))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="cannot establish whether it reads"):
        CausalModel(equations)


_HELPER_LOOKUP = {"a": 1, "b": 2}


def _helper_with_nested_global_read(values):
    return sum(_HELPER_LOOKUP[v] for v in values)


def test_snapshot_includes_helper_globals_used_in_nested_comprehensions():
    @mechanism
    def equations(x: Dom(["a", "b"])):
        result = V(_helper_with_nested_global_read([x]))
        raw_input = V(x, domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    try:
        _HELPER_LOOKUP["a"] = 99
        assert model.new_trace({"x": "a"})["result"] == 1
    finally:
        _HELPER_LOOKUP["a"] = 1


def test_notebook_capture_does_not_retain_unrelated_objects():
    import gc
    import weakref

    class LargeNotebookObject:
        pass

    def capture():
        unrelated = LargeNotebookObject()
        reference = weakref.ref(unrelated)

        @mechanism
        def equations(x: Dom(range(2))):
            raw_input = V(str(x), domain=Dom(str))  # noqa: F841
            raw_output = V(x)
            return raw_output

        return equations, reference

    equations, reference = capture()
    gc.collect()
    assert reference() is None
    assert CausalModel(equations).new_trace({"x": 1})["raw_output"] == 1


@pytest.mark.parametrize("values", [[1], [1, 2, 3], {0: 1, 2: 2}])
def test_family_inputs_reject_missing_or_extra_members(values):
    @mechanism
    def equations(xs: FamilyDom(Dom(range(4)), size=2)):
        raw_input = V(str(xs), domain=Dom(str))  # noqa: F841
        raw_output = V(xs[0] + xs[1])
        return raw_output

    model = CausalModel(equations)
    with pytest.raises(ValueError, match="requires exactly these indices"):
        model.new_trace({"xs": values})
