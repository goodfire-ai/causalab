"""Family key shapes retain separate names, values, and interventions."""

import pytest

from causalab.causal import (
    CausalModel,
    DefinitionError,
    Dom,
    FamilyDom,
    V,
    family,
    mechanism,
)
from causalab.io.counterfactuals import (
    load_counterfactual_examples,
    save_counterfactual_examples,
)

pytestmark = pytest.mark.unit

KEYS = (0, (0,), (), "x", ("x",), (0, "x"), "0,", "()")
SUFFIXES = ("[0]", "[0,]", "[()]", "['x']", "['x',]", "[0,'x']", "['0,']", "['()']")


def _input_model():
    @mechanism
    def equations(xs: FamilyDom({key: Dom(range(10)) for key in KEYS})):
        result = V(tuple([xs[key] for key in KEYS]), domain=Dom(tuple))
        raw_input = V(str(xs), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    return CausalModel(equations)


def _computed_model():
    @mechanism
    def equations(x: Dom(range(10))):
        xs = family(keys=KEYS, domain=Dom(range(10)))
        for key in KEYS:
            xs[key] = x
        result = V(tuple([xs[key] for key in KEYS]), domain=Dom(tuple))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    return CausalModel(equations)


@pytest.mark.parametrize("build_model", [_input_model, _computed_model])
def test_distinct_family_key_shapes_keep_independent_interventions(build_model):
    model = build_model()
    assert model.families["xs"] == {
        key: "xs" + suffix for key, suffix in zip(KEYS, SUFFIXES)
    }
    inputs = (
        {"xs": dict(zip(KEYS, range(len(KEYS))))}
        if build_model is _input_model
        else {"x": 2}
    )
    trace = model.new_trace(inputs)
    expected = (
        list(range(len(KEYS))) if build_model is _input_model else [2] * len(KEYS)
    )
    assert trace["result"] == tuple(expected)
    for index, suffix in enumerate(SUFFIXES):
        trace["xs" + suffix] = 9
        expected[index] = 9
        assert trace["result"] == tuple(expected)


def test_input_family_accepts_distinct_canonical_member_keys():
    model = _input_model()
    trace = model.new_trace(
        {"xs" + suffix: value for value, suffix in enumerate(SUFFIXES)}
    )
    assert trace["result"] == tuple(range(len(KEYS)))


@pytest.mark.parametrize("build_model", [_input_model, _computed_model])
def test_family_key_shapes_and_interventions_survive_json(tmp_path, build_model):
    model = build_model()
    inputs = {"xs": dict.fromkeys(KEYS, 2)} if build_model is _input_model else {"x": 2}
    original = model.new_trace(inputs)
    original.intervene_many({"xs[0]": 3, "xs[0,]": 4, "xs[()]": 5})
    path = str(tmp_path / "family-keys.json")
    save_counterfactual_examples(
        [{"input": original, "counterfactual_inputs": []}], path
    )
    restored = load_counterfactual_examples(path, model)[0]["input"]
    assert restored.to_dict() == original.to_dict()
    assert restored._overrides == original._overrides
    restored["xs[0,]"] = 6
    assert restored["result"] == (3, 6, 5, 2, 2, 2, 2, 2)
    if build_model is _computed_model:
        restored["x"] = 7
        assert restored["result"] == (3, 6, 5, 7, 7, 7, 7, 7)


def test_invalid_input_family_keys_have_a_source_location():
    @mechanism
    def equations(xs: FamilyDom({0.5: Dom([0])})):
        raw_input = V("input")  # noqa: F841
        raw_output = V("output")
        return raw_output

    with pytest.raises(
        DefinitionError,
        match=r"test_family_keys.py:\d+: Input family 'xs': Family indices",
    ):
        CausalModel(equations)


def test_duplicate_computed_family_keys_have_a_source_location():
    @mechanism
    def equations():
        xs = family(keys=[(0,), (0,)])
        xs[(0,)] = 0
        raw_input = V("input")  # noqa: F841
        raw_output = V("output")
        return raw_output

    with pytest.raises(
        DefinitionError, match=r"test_family_keys.py:\d+: Family keys must be unique"
    ):
        CausalModel(equations)
