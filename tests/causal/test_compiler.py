"""Compiler regressions: intervention domains, ownership, and Python semantics."""

import gc
import linecache
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, V, family, mechanism
from causalab.causal.compiler import ConfigurationCopier

pytestmark = pytest.mark.unit


def test_comprehension_keeps_filters_before_a_constant_false_filter():
    @mechanism
    def equations(x: Dom([0, 1])):
        result = V([i for i in range(1) if 1 // x if False], domain=Dom(list))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == ["x"]
    assert model.new_trace({"x": 1})["result"] == []
    with pytest.raises(ZeroDivisionError):
        model.new_trace({"x": 0})


def test_comprehension_keeps_filters_before_an_empty_nested_loop():
    @mechanism
    def equations(x: Dom([0, 1])):
        result = V([j for i in range(1) if 1 // x for j in range(0)], domain=Dom(list))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == ["x"]
    assert model.new_trace({"x": 1})["result"] == []
    with pytest.raises(ZeroDivisionError):
        model.new_trace({"x": 0})


def test_mutating_methods_require_a_pure_helper():
    @mechanism
    def equations(x: Dom([3])):
        items = [10, x]
        first = items.pop()
        second = items.pop()
        result = V((first, second), domain=Dom(tuple))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="[Mm]utat.*pure helper"):
        CausalModel(equations)


def test_unknown_comparison_result_type_requires_an_explicit_domain():
    import numpy as np

    @mechanism
    def equations(x: Dom(np.int64)):
        result = V(x > 0)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Cannot infer a sound domain"):
        CausalModel(equations)


@pytest.mark.parametrize("finite", [False, True])
def test_numpy_comparisons_keep_their_scalar_type(finite):
    import numpy as np

    inputs = Dom([np.int64(-1), np.int64(1)]) if finite else Dom(np.int64)
    result_domain = None if finite else Dom(np.bool_)

    @mechanism
    def equations(x: inputs):
        result = V(x > 0, domain=result_domain)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    trace = model.new_trace({"x": np.int64(-1)})
    assert type(trace["result"]) is np.bool_
    assert not trace["result"]
    trace["x"] = np.int64(1)
    assert trace["result"]


@pytest.mark.parametrize("expression", ["items.pop()", "pop()", "list.pop(items)"])
def test_mutating_method_aliases_are_rejected(expression):
    source = f"""@mechanism
def equations(x: Dom([3])):
    items = [10, x]
    pop = items.pop
    result = V({expression}, domain=Dom(int))
    raw_input = V(str(x))
    raw_output = V(str(result))
    return result
"""
    with pytest.raises(DefinitionError, match=r"Mutating method.*pure helper"):
        _from_source(source)


def test_pure_helper_can_mutate_its_own_container():
    def take_two(x):
        items = [10, x]
        return items.pop(), items.pop()

    @mechanism
    def equations(x: Dom([3])):
        result = V(take_two(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 3})["result"] == (3, 10)


def test_numpy_mutation_requires_a_pure_helper():
    import numpy as np

    @mechanism
    def equations(x: Dom(np.ndarray)):
        result = V(x.fill(0), domain=Dom([None]))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Mutating method.*pure helper"):
        CausalModel(equations)


def test_setattr_requires_a_pure_helper():
    settings = SimpleNamespace(value=0)

    @mechanism
    def equations(x: Dom([1])):
        result = V(setattr(settings, "value", x), domain=Dom([None]))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Mutation.*pure helper"):
        CausalModel(equations)


def test_pure_module_functions_can_share_container_method_names():
    import operator

    addition = operator.add

    @mechanism
    def equations(x: Dom(range(3))):
        result = V(operator.add(x, 1) + addition(x, 1))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 2})["result"] == 6


def test_capped_filtered_count_streams_sequence_domains():
    @mechanism
    def equations(pair: Dom.sequence(Dom(range(2**32)), length=2)):
        raw_input = V(str(pair))  # noqa: F841
        raw_output = V(str(pair))
        return raw_output

    model = CausalModel(equations)
    visited = []

    def accepts(trace):
        visited.append(trace["pair"])
        return trace["pair"][1] % 2 == 0

    model.input_filter = accepts
    assert model.count_inputs(limit=3) == 4
    assert visited == [(0, y) for y in range(7)]


def test_capped_filtered_count_streams_large_ranges():
    @mechanism
    def equations(x: Dom(range(2**32)), y: Dom(range(2**32))):
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(y))
        return raw_output

    model = CausalModel(equations)
    visited = []

    def accepts(trace):
        visited.append((trace["x"], trace["y"]))
        return trace["y"] % 2 == 0

    model.input_filter = accepts
    assert model.count_inputs(limit=3) == 4
    assert visited == [(0, y) for y in range(7)]


def test_snapshot_memo_retains_source_identities_until_discarded():
    @dataclass
    class Container:
        items: list

    freeze = ConfigurationCopier()
    source = Container([1, 2])
    reference = weakref.ref(source)
    assert freeze(source).items == [1, 2]
    del source
    gc.collect()
    assert reference() is not None
    del freeze
    gc.collect()
    assert reference() is None


def test_helpers_in_the_model_module_copy_their_configuration():
    offsets = [1]

    def shift(value):
        return value + offsets[0]

    shift.__module__ = "causalab.causal.model"

    @mechanism
    def equations(x: Dom(range(4))):
        result = V(shift(x), domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    offsets[0] = 10
    assert model.new_trace({"x": 2})["result"] == 3


def test_temporary_loop_constants_and_default_metadata_are_independent():
    @mechanism
    def equations():
        items = family(size=5, domain=Dom(tuple))
        for i in range(5):
            for pair in [(i, i + 10)]:
                items[i] = pair
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(items), domain=Dom(str))
        return raw_output

    for _ in range(100):
        model = CausalModel(equations)
        trace = model.new_trace()
        assert [trace[f"items[{i}]"] for i in range(5)] == [
            (i, i + 10) for i in range(5)
        ]
        assert model.embeddings is not model.periods


def test_branch_domains_are_merged_before_specializing_consumers():
    @mechanism
    def equations(enabled: Dom(bool)):
        if enabled:
            gate = V(0, domain=Dom([0]))
            result = V(10 if gate == 1 else 20)
        else:
            gate = V(1, domain=Dom([1]))
            result = V(10 if gate == 1 else 20)
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert "gate" in model.parents["result"]
    for enabled in (False, True):
        for gate in (0, 1):
            trace = model.new_trace({"enabled": enabled})
            trace["gate"] = gate
            assert trace["result"] == (10 if gate == 1 else 20)


def test_branch_domains_are_merged_before_selecting_family_members():
    @mechanism
    def equations(enabled: Dom(bool)):
        items = family(size=2)
        items[0] = 10
        items[1] = 20
        if enabled:
            index = V(0, domain=Dom([0]))
            result = V(items[index])
        else:
            index = V(1, domain=Dom([1]))
            result = V(items[index])
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert {"items[0]", "items[1]", "index"} <= set(model.parents["result"])
    for enabled in (False, True):
        for index in (0, 1):
            trace = model.new_trace({"enabled": enabled})
            trace["index"] = index
            assert trace["result"] == [10, 20][index]


def test_equation_results_do_not_expose_compiler_constants():
    config = [{"items": [1]}]

    @mechanism
    def equations():
        payload = V(config, domain=Dom(list))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(payload), domain=Dom(str))  # noqa: F841
        return payload

    model = CausalModel(equations)
    first, second = model.new_trace(), model.new_trace()
    first["payload"][0]["items"].append(2)
    assert second["payload"] == [{"items": [1]}]
    assert model.new_trace()["payload"] == [{"items": [1]}]


def test_metadata_and_saved_definition_do_not_expose_equation_constants():
    periods = {"x": 1}

    @mechanism
    def equations():
        result = V(periods["x"], domain=Dom(int))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations, periods=periods)
    model.periods["x"] = 10
    model.definition.environment["periods"]["x"] = 20
    periods["x"] = 30
    assert model.new_trace()["result"] == 1


@pytest.mark.parametrize("shape", [SimpleNamespace, "dataclass"])
def test_configuration_objects_freeze_nested_helpers(shape):
    lookup = {"a": 1}

    def helper(x):
        return lookup[x]

    @dataclass
    class Config:
        helper: object

    config = (Config if shape == "dataclass" else shape)(helper=helper)

    @mechanism
    def equations(x: Dom(["a"])):
        result = V(config.helper(x), domain=Dom(int))
        raw_input = V(x)  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    lookup["a"] = 99
    assert model.new_trace({"x": "a"})["result"] == 1


def test_inferred_selector_domain_removes_impossible_self_cycle():
    @mechanism
    def equations(x: Dom(range(3))):
        index = V(0)
        steps = family(size=2)
        steps[0] = x
        steps[1] = steps[index]
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(steps[1]))  # noqa: F841
        return steps[1]

    model = CausalModel(equations)
    assert set(model.parents["steps[1]"]) == {"index", "steps[0]"}
    assert model.new_trace({"x": 2})["steps[1]"] == 2


def test_allowed_selector_self_cycle_is_still_rejected():
    @mechanism
    def equations(x: Dom(range(3))):
        index = V(0, domain=Dom([0, 1]))
        steps = family(size=2, domain=Dom(range(3)))
        steps[0] = x
        steps[1] = steps[index]
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(steps[1]))  # noqa: F841
        return steps[1]

    with pytest.raises(DefinitionError, match="acyclic"):
        CausalModel(equations)


def test_specializing_many_impossible_indices_does_not_duplicate_branches():
    @mechanism
    def equations():
        index = V(39)
        items = family(size=40, domain=Dom(int))
        for i in range(40):
            items[i] = i
        selected = V(items[index])
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(selected), domain=Dom(str))  # noqa: F841
        return selected

    model = CausalModel(equations)
    assert set(model.parents["selected"]) == {"index", "items[39]"}
    assert model.new_trace()["selected"] == 39


@pytest.mark.parametrize("domain", [Dom(range(3)), Dom(int)])
def test_chained_comparison_reads_follow_python_short_circuiting(domain):
    @mechanism
    def equations(x: domain):
        result = V(1 < 0 < x)
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == []
    assert model.new_trace({"x": 2})["result"] is False


def _from_source(source, **values):
    filename = "<compiler-test-cell>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(
        CausalModel=CausalModel, Dom=Dom, V=V, mechanism=mechanism, **values
    )
    try:
        exec(compile(source, filename, "exec"), namespace)
        return CausalModel(namespace["equations"])
    finally:
        linecache.cache.pop(filename, None)


@pytest.mark.parametrize(
    "expression",
    [
        "any(x // (1-i) for i in range(2))",
        "all((x-1) // (1-i) for i in range(2))",
        "next(x // (1-i) for i in range(2))",
    ],
)
def test_generators_require_a_pure_helper_instead_of_eager_lowering(expression):
    source = f"""@mechanism
def equations(x: Dom([1])):
    result = V({expression}, domain=Dom(int))
    raw_input = V(str(x))
    raw_output = V(str(result))
    return result
"""
    with pytest.raises(DefinitionError, match="[Gg]enerator.*pure helper"):
        _from_source(source)


def test_generator_helpers_keep_any_all_and_next_short_circuiting():
    def helper(x):
        return (
            any(x // (1 - i) for i in range(2)),
            all((x - 1) // (1 - i) for i in range(2)),
            next(x // (1 - i) for i in range(2)),
        )

    @mechanism
    def equations(x: Dom([1])):
        result = V(helper(x), domain=Dom(tuple))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 1})["result"] == helper(1)


def test_large_eager_comprehensions_use_flat_construction():
    @mechanism
    def equations(x: Dom([0, 1])):
        result = V([x + i for i in range(1000) if i % 2 == 0], domain=Dom(list))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"x": 1})["result"] == [
        1 + i for i in range(1000) if i % 2 == 0
    ]


def test_total_comprehension_expansion_has_a_located_limit():
    @mechanism
    def equations():
        result = V([i + j for i in range(101) for j in range(100)], domain=Dom(list))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    with pytest.raises(
        DefinitionError,
        match=r"test_compiler.py:\d+: .*(expansion|complexity)",
    ):
        CausalModel(equations)


def test_comprehension_filters_run_before_nested_iterables():
    @mechanism
    def equations():
        result = V([j for i in range(2) if i != 0 for j in range(1 // i)])
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace()["result"] == [0]


@pytest.mark.parametrize("use_helper", [False, True])
@pytest.mark.parametrize("collision", [False, True])
def test_captured_definitions_release_unrelated_source_namespace_objects(
    use_helper, collision
):
    class LargeObject:
        pass

    def build():
        unrelated = LargeObject()
        reference = weakref.ref(unrelated)
        expression = "helper(x)" if use_helper else "x + 1"
        model = _from_source(
            f"""lookup = {{0: 1}}
def helper(x):
    return lookup[x]
@mechanism
def equations(x: Dom([0])):
    result = V({expression})
    raw_input = V(str(x))
    raw_output = V(str(result))
    return result
""",
            **{"result" if collision else "unrelated": unrelated},
        )
        return model, reference

    model, reference = build()
    gc.collect()
    assert reference() is None
    assert model.new_trace({"x": 0})["result"] == 1


@pytest.mark.parametrize(
    "assignment", ["x, other = (1, 2)", "for x, other in [(1, 2)]:\n        pass"]
)
def test_recursive_assignment_targets_cannot_rebind_exposed_inputs(assignment):
    source = f"""@mechanism
def equations(x: Dom([0, 1])):
    {assignment}
    raw_input = V(str(x))
    raw_output = V(x)
    return raw_output
"""
    with pytest.raises(DefinitionError, match="cannot be reassigned"):
        _from_source(source)


@pytest.mark.parametrize(
    "expression",
    [
        "x < 0 < y < z",
        "x > 0 and (y or z)",
        "y if x else z",
        "[y + i for i in range(3) if x > i]",
        "[j + y for i in range(2) if i for j in range(1 // i)]",
    ],
)
def test_accepted_expressions_match_python_values_and_union_of_reads(expression):
    import itertools

    source = f"""@mechanism
def equations(x: Dom([-1, 0, 1]), y: Dom([-1, 0, 1]), z: Dom([-1, 0, 1])):
    result = V({expression}, domain=Dom(object))
    raw_input = V(str((x, y, z)), domain=Dom(str))
    raw_output = V(str(result), domain=Dom(str))
    return result
"""
    model = _from_source(source)
    reads = set()

    # Substitute only model inputs with observable reads in the native Python
    # reference. Comprehensions keep their original scopes and evaluation order.
    import ast

    class Observe(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in ("x", "y", "z"):
                return ast.Call(
                    ast.Name("read", ast.Load()), [ast.Constant(node.id)], []
                )
            return node

    native = compile(
        ast.fix_missing_locations(Observe().visit(ast.parse(expression, mode="eval"))),
        "<native>",
        "eval",
    )
    for values in itertools.product([-1, 0, 1], repeat=3):
        inputs = dict(zip(("x", "y", "z"), values))

        def read(name):
            reads.add(name)
            return inputs[name]

        assert model.new_trace(inputs)["result"] == eval(native, {"read": read})
    assert set(model.parents["result"]) == reads


def test_callable_resolution_does_not_execute_a_factory():
    calls = []

    def factory():
        calls.append("called")
        return str

    @mechanism
    def equations(x: Dom([1])):
        result = V(factory()(x), domain=Dom(str))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(result)  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="dynamic callable"):
        CausalModel(equations)
    assert calls == []


def test_hierarchical_equality_notebook_introductory_cells_execute():
    import json
    from pathlib import Path

    notebook = (
        Path(__file__).parents[2] / "causalab/tasks/hierarchical_equality/demo.ipynb"
    )
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {}
    for cell in cells[:6]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    trace = namespace["trace"]
    assert trace["icl_seed"] >= 0
    assert trace["result_equality"] == (
        (trace["var_1"] == trace["var_2"]) == (trace["var_3"] == trace["var_4"])
    )


def test_large_domains_have_bounded_consumer_operations():
    import random

    domain = Dom(range(2**32))
    assert domain.cardinality() == 2**32
    assert domain.cardinality(limit=10) == 11
    assert 0 <= domain.sample(random.Random(1)) < 2**32
    with pytest.raises(ValueError, match="explicit candidates or use sampling"):
        domain.require_enumerated()
    assert Dom(int).cardinality() is None


def test_capped_filtered_count_stops_early_without_computing_traces(monkeypatch):
    @mechanism
    def equations(x: Dom(range(1000))):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(x)
        return raw_output

    model = CausalModel(equations)
    visited = []

    def accepts(trace):
        visited.append(trace["x"])
        return trace["x"] % 2 == 0

    model.input_filter = accepts

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("Counting must not compute traces")

    monkeypatch.setattr(model, "new_trace", unexpected_execution)
    assert model.count_inputs(limit=5) == 6
    assert visited == list(range(11))


def test_container_subclasses_are_rejected_instead_of_losing_their_behavior():
    from collections import defaultdict, namedtuple

    Config = namedtuple("Config", ["threshold"])
    for config in (Config(2), defaultdict(int, {"threshold": 2})):

        @mechanism
        def equations():
            raw_input = V("input")  # noqa: F841
            raw_output = V(str(config), domain=Dom(str))
            return raw_output

        with pytest.raises(
            DefinitionError, match="container subclass.*plain container"
        ):
            CausalModel(equations)


def test_deleting_an_intervened_value_only_invalidates_its_cache():
    @mechanism
    def equations(x: Dom([0, 1])):
        result = V(x)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    trace = CausalModel(equations).new_trace({"x": 0})
    trace["result"] = 1
    del trace["result"]
    assert trace["result"] == 1
    assert trace["raw_output"] == "1"
