"""Task conveniences retain the model's construction-time configuration."""

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.tasks.loader import Task

pytestmark = pytest.mark.unit


def test_intervention_candidates_own_nested_domain_values():
    @mechanism
    def equations(x: Dom([0, 1])):
        result = V([x], domain=Dom([[0], [1]]))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result[0]), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    task = Task("inline", model, lambda a, b: a == b, "result")
    candidates = task.intervention_values
    candidates[0][0] = 9
    assert task.intervention_values == [[0], [1]]
    assert model.new_trace({"x": 0})["raw_output"] == "0"


def test_relation_factory_snapshots_generator_and_equation_configuration():
    from causalab.tasks.subject_object_relations.causal_models import (
        create_causal_model,
    )
    from causalab.tasks.subject_object_relations.config import (
        SubjectObjectRelationsConfig,
    )
    from causalab.tasks.subject_object_relations.counterfactuals import generate_dataset

    config = SubjectObjectRelationsConfig(
        relation="word_first_letter",
        subjects=["a", "b", "c"],
        subject_to_object={"a": "red", "b": "blue", "c": "red"},
        objects=["red", "blue"],
        templates=["Object for {subject}:"],
    )
    model = create_causal_model(config)
    config.subject_to_object.update({"b": "red", "c": "blue"})
    config.templates[0] = "Changed {subject}"
    config.subjects.append("unknown")
    rows = generate_dataset(model, n=10, seed=1)
    for row in rows:
        base, donor = row["input"], row["counterfactual_inputs"][0]
        result = model.run_interchange(base, {"object": donor})
        assert result["raw_output"] != base["raw_output"]
        assert base["raw_input"].startswith("Object for ")


def test_identity_factory_snapshots_generator_templates():
    from causalab.tasks.identity_naming.causal_models import create_causal_model
    from causalab.tasks.identity_naming.config import IdentityNamingConfig
    from causalab.tasks.identity_naming.counterfactuals import generate_dataset

    config = IdentityNamingConfig(
        domain_type="pitch_midi",
        entities=["C4", "D4"],
        entity_to_result={"C4": "60", "D4": "62"},
        templates=["Name {entity}:"],
    )
    model = create_causal_model(config)
    config.templates[0] = "Changed {entity}"
    config.entity_to_result["C4"] = "62"
    row = generate_dataset(model, n=1, seed=1)[0]
    assert row["input"]["raw_input"] == "Name C4:"
    assert row["input"]["result"] == "60"


@pytest.mark.parametrize("random_baseline", [False, True])
def test_arithmetic_factory_snapshots_metadata_and_helpers(
    monkeypatch, random_baseline
):
    from causalab.tasks.natural_domains_arithmetic import causal_models
    from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

    config = NaturalDomainConfig(
        domain_type="weekdays",
        entities=["Monday", "Tuesday"],
        numbers=["one"],
        number_to_int={"one": 1},
        modulus=2,
        template=["After {entity} by {number}:"],
    )
    monkeypatch.setattr(causal_models, "get_random_words", lambda n: ["alpha", "beta"])
    factory = (
        causal_models.create_random_causal_model
        if random_baseline
        else causal_models.create_causal_model
    )
    model = factory(config)
    config.template[0] = "Changed {entity} {number}"
    config.modulus = 1
    config.number_to_int["one"] = 0
    assert causal_models.GET_TEMPLATE(model) == ["After {entity} by {number}:"]
    assert causal_models.GET_PERIODIC_INFO(model) == {"entity": 2, "result": 2}
    entity = "alpha" if random_baseline else "Monday"
    expected = "beta" if random_baseline else "Tuesday"
    trace = model.new_trace(
        {"entity": entity, "number": "one", "template": "After {entity} by {number}:"}
    )
    assert trace["result"] == expected
