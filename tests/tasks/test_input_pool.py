"""Split enumeration uses the valid input count, not the raw domain size."""

from types import SimpleNamespace

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.tasks.loader import Task
from causalab.tasks.splits import _input_pool

pytestmark = pytest.mark.unit


def test_finiteness_is_independent_of_size_and_capped_lower_bounds():
    assert Dom(range(2**64)).is_finite
    assert Dom.sequence(Dom(str), length=0).is_finite
    assert not Dom.sequence(Dom(str), max_length=1).is_finite
    assert Dom.union(Dom(range(100000)), Dom([-1])).is_finite
    assert not Dom.union(Dom([0]), Dom(int)).is_finite


def test_small_filtered_pool_is_enumerated_from_large_finite_domain():
    @mechanism
    def equations(x: Dom(range(65537))):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))
        return raw_output

    model = CausalModel(equations, input_filter=lambda trace: trace["x"] < 2)
    task = Task("finite_inline", model, lambda a, b: a == b, "x")
    pool = _input_pool(task, seed=0, max_inputs=2, generator="generate_dataset")
    assert [trace["x"] for trace in pool] == [0, 1]


def test_open_union_still_uses_task_generator(monkeypatch):
    @mechanism
    def equations(x: Dom.union(Dom([0]), Dom(int))):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    task = Task("open_inline", model, lambda a, b: a == b, "x")

    def generate_dataset(model, n, seed):
        return [
            {
                "input": model.new_trace({"x": 7}),
                "counterfactual_inputs": [model.new_trace({"x": 8})],
            }
        ]

    monkeypatch.setattr(
        "causalab.tasks.splits.load_task_counterfactuals",
        lambda name: SimpleNamespace(generate_dataset=generate_dataset),
    )
    pool = _input_pool(task, seed=0, max_inputs=2, generator="generate_dataset")
    assert [trace["x"] for trace in pool] == [7, 8]


def test_open_domain_without_max_inputs_asks_for_a_pool_size():
    """An open domain has no input count to default the pool size to; the
    refusal names the input instead of failing inside n_unique_inputs."""

    @mechanism
    def equations(x: Dom.union(Dom([0]), Dom(int))):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))
        return raw_output

    task = Task("open_inline", CausalModel(equations), lambda a, b: a == b, "x")
    with pytest.raises(ValueError, match=r"Specify max_inputs: inputs \['x'\]"):
        _input_pool(task, seed=0, max_inputs=None, generator="generate_dataset")
