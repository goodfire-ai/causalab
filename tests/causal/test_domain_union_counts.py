"""Capped finite counts remain useful beyond the enumeration budget."""

import random
from types import SimpleNamespace

import pytest

from causalab.causal import CausalModel, Dom, Exo, V, mechanism
from causalab.causal.counterfactuals import sample_intervention

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("limit", [0, 1, 10, 4099, 4100, 4101, 5000])
def test_large_finite_union_counts_saturate(limit):
    domain = Dom.union(Dom(range(4100)), Dom([None]))
    assert domain.cardinality() == 4101
    assert domain.cardinality(limit=limit) == min(4101, limit + 1)


def test_overlapping_members_do_not_inflate_capped_count():
    domain = Dom.union(Dom(range(4100)), Dom(range(4000, 4200)))
    assert domain.cardinality(limit=4199) == 4200
    assert domain.cardinality(limit=4200) == 4200
    assert domain.cardinality() == 4200


def test_repeated_finite_entries_do_not_prove_a_false_union_lower_bound():
    domain = Dom.union(Dom([0] * 4100), Dom([1]))
    assert domain.cardinality() == domain.cardinality(limit=10) == 2
    nested = Dom.union(Dom.sequence(Dom([0, 0]), length=20), Dom([None]))
    assert nested.cardinality() == nested.cardinality(limit=10) == 2


def test_large_member_proves_saturation_without_enumerating(monkeypatch):
    huge = Dom(range(2**32))
    domain = Dom.union(Dom(int), huge)

    def unexpected_enumeration(*args):
        raise AssertionError("The finite lower bound already exceeds the cap")

    monkeypatch.setattr(huge, "enumerated", unexpected_enumeration)
    assert domain.cardinality(limit=10) == 11
    assert Dom.union(Dom(int), Dom([None])).cardinality(limit=10) is None


def test_dataset_input_pool_reaches_its_generator_fallback(monkeypatch):
    from causalab.tasks import splits

    @mechanism
    def equations(x: Dom.union(Dom(range(4100)), Dom([None]))):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    assert model.count_inputs(limit=10) == 11
    calls = []

    def generate(model, n, seed):
        calls.append((n, seed))
        return [
            {"input": model.new_trace({"x": i}), "counterfactual_inputs": []}
            for i in range(n)
        ]

    monkeypatch.setattr(
        splits,
        "load_task_counterfactuals",
        lambda name: SimpleNamespace(generate=generate),
    )
    task = SimpleNamespace(name="inline", causal_model=model)
    pool = splits._input_pool(task, seed=7, max_inputs=10, generator="generate")
    assert calls == [(10, 7)]
    assert [trace["x"] for trace in pool] == list(range(10))


def test_numpy_scalar_classes_remain_distinct_in_domains_and_inference():
    np = pytest.importorskip("numpy")
    if np.int64 is np.longlong:
        pytest.skip("This platform aliases int64 and longlong")
    native = np.array([1], dtype=np.int64)
    alternate = np.array([1], dtype=np.longlong)
    assert native.dtype.str == alternate.dtype.str
    assert not Dom([native]).contains(alternate)
    assert Dom.union(Dom([native]), Dom([alternate])).cardinality() == 2

    @mechanism
    def equations(use_native: Dom(bool), y: Dom([10, 20])):
        packed = V(
            np.array([1], dtype=np.int64)
            if use_native
            else np.array([1], dtype=np.longlong)
        )
        result = V(y if type(packed[0]) is np.longlong else 99)
        raw_input = V(str(use_native))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"packed", "y"}
    trace = model.new_trace({"use_native": True, "y": 10})
    assert trace["result"] == 99
    trace["use_native"] = False
    assert trace["result"] == 10
    trace["y"] = 20
    assert trace["result"] == 20


@pytest.mark.parametrize("container", [tuple, list])
def test_empty_sequences_have_one_value_without_a_finite_element_domain(container):
    domain = Dom.sequence(Dom(str), length=0, container=container)
    assert domain.cardinality() == 1
    assert domain.cardinality(limit=0) == 1
    assert domain.enumerated() == (container(),)
    assert domain.sample(random.Random(0)) == container()

    @mechanism
    def equations(x: domain):
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))
        return raw_output

    model = CausalModel(equations)
    assert model.count_inputs() == 1
    assert [trace["x"] for trace in model.enumerate_inputs()] == [container()]


@pytest.mark.parametrize(
    "values", [range(2**64), range(2**64, -1, -3), range(-(2**64), 2**64, 2)]
)
def test_large_ranges_count_and_sample_without_platform_sized_lengths(values):
    domain = Dom(values)
    size = (abs(values.stop - values.start) + abs(values.step) - 1) // abs(values.step)
    assert domain.cardinality() == size
    assert domain.cardinality(limit=10) == 11
    assert domain.public_values() is values
    assert domain.enumerated() is None
    rng = random.Random(1)
    samples = [domain.sample(rng) for _ in range(20)]
    assert all(sample in values for sample in samples)
    assert len(set(samples)) == len(samples)


def test_unsigned_64_bit_noise_constructs_and_conditional_reads_have_witnesses():
    @mechanism
    def equations(seed: Exo(Dom(range(2**64))), y: Dom([10, 20])):
        result = V(y if seed else 99, domain=Dom(int))
        raw_input = V(str(seed), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    trace = model.sample_input(rng=random.Random(0))
    assert 0 <= trace["seed"] < 2**64
    assert trace["result"] == trace["y"]
    trace["seed"] = 0
    assert trace["result"] == 99
    assert model.count_inputs(noise={"seed": 1}) == 2


def test_conditional_finite_unions_are_sampleable_intervention_domains():
    @mechanism
    def equations(x: Dom(range(5000)), enabled: Dom(bool)):
        result = V(x if enabled else -1)
        raw_input = V(str((x, enabled)), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    domain = model.domains["result"]
    assert domain.cardinality() == 5001
    rng = random.Random(0)
    assert all(domain.contains(domain.sample(rng)) for _ in range(20))
    state = random.getstate()
    try:
        random.seed(1)
        assert domain.contains(sample_intervention(model)["result"])
    finally:
        random.setstate(state)
