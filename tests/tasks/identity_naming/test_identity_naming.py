"""Property-tier invariants for the identity_naming factory task.

``identity_naming`` parametrises an entity -> canonical-name mapping via
``IdentityNamingConfig``; ``DOMAIN_PRESETS`` currently ships a single preset
(``pitch_midi``: note-name -> MIDI number over C2..C6). The mechanisms are
pure-symbolic, so these tests run sub-second and hypothesis-driven sweeps
catch contract drift the runner-level golden in
``tests/end_to_end/goldens/identity_naming_pitch_midi.json`` would only flag indirectly.

Numerical pinning of specific symbolic outputs would live in
``tests/tasks/identity_naming/pinned_samples.json`` (with a sibling
``test_identity_naming_numerical.py``), but
``tests/_helpers/task_pins.py::walk_task_samples`` currently raises
``NotImplementedError`` for factory tasks. The numerical-direct cell
stays red on the dashboard until that walker learns how to serialise
``IdentityNamingConfig``. The runner golden defends MIDI math
end-to-end in the meantime.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke comes from
``baseline/identity_naming_pitch_midi.yaml`` via the smoke harness.
This file satisfies property-direct.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.causal.model import CausalModel
from causalab.tasks.identity_naming.causal_models import (
    CYCLIC_VARIABLES,
    EMBEDDINGS,
    GET_CYCLIC_VARIABLES,
    GET_EMBEDDINGS,
    GET_PERIODIC_INFO,
    GET_TEMPLATE,
    GET_VARIABLE_VALUES,
    TARGET_VARIABLE,
    create_causal_model,
)
from causalab.tasks.identity_naming.config import (
    DOMAIN_PRESETS,
    IdentityNamingConfig,
    _note_to_midi,  # pyright: ignore[reportPrivateUsage]
)
from causalab.tasks.identity_naming.counterfactuals import generate_dataset

# Fixture return type: (CausalModel, IdentityNamingConfig). Aliased so per-test
# annotations stay readable.
ModelAndConfig = tuple[CausalModel, IdentityNamingConfig]


# Hypothesis settings used across this file:
#   - deadline=None: causal-model evaluation is pure-symbolic and not
#     timing-sensitive, but hypothesis' default 200ms deadline can flake
#     on cold caches.
#   - function_scoped_fixture: the per-preset fixture below is
#     function-scoped, and hypothesis surfaces a health-check warning
#     otherwise.
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture(params=list(DOMAIN_PRESETS))
def model_and_config(request: pytest.FixtureRequest) -> ModelAndConfig:
    """Factory fixture: one (model, config) pair per preset in DOMAIN_PRESETS.

    The fixture parametrisation gives every property class automatic
    coverage when a new preset is added to ``DOMAIN_PRESETS``.
    """
    config = IdentityNamingConfig(domain_type=request.param)
    return create_causal_model(config), config


class TestIdentityNamingConfigProperty:
    """``IdentityNamingConfig`` validation and the MIDI math in ``config.py``."""

    pytestmark = pytest.mark.property

    def test_invalid_domain_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="domain_type must be one of"):
            IdentityNamingConfig(domain_type="nonexistent")

    def test_preset_fills_entities(self, model_and_config: ModelAndConfig) -> None:
        """Empty ``entities`` is back-filled from ``DOMAIN_PRESETS``."""
        _, config = model_and_config
        assert len(config.entities) > 0

    def test_preset_fills_entity_to_result(
        self, model_and_config: ModelAndConfig
    ) -> None:
        _, config = model_and_config
        assert len(config.entity_to_result) == len(config.entities)

    def test_preset_fills_templates(self, model_and_config: ModelAndConfig) -> None:
        _, config = model_and_config
        assert len(config.templates) >= 1

    def test_midi_c4(self) -> None:
        assert _note_to_midi("C4") == 60

    def test_midi_a0(self) -> None:
        assert _note_to_midi("A0") == 21

    def test_midi_csharp4(self) -> None:
        assert _note_to_midi("C#4") == 61

    def test_midi_b_minus_1_boundary(self) -> None:
        """``B-1`` is the lowest standard MIDI note (number 11)."""
        assert _note_to_midi("B-1") == 11

    def test_pitch_midi_preset_entity_count(self) -> None:
        """Range C2..C6 inclusive = 49 entities."""
        cfg = IdentityNamingConfig(domain_type="pitch_midi")
        assert len(cfg.entities) == 49

    def test_pitch_midi_preset_round_trip(self) -> None:
        """``entity_to_result[note] == str(_note_to_midi(note))`` for every entity."""
        cfg = IdentityNamingConfig(domain_type="pitch_midi")
        assert all(
            cfg.entity_to_result[note] == str(_note_to_midi(note))
            for note in cfg.entities
        )


class TestIdentityNamingCausalModelStructureProperty:
    """Static structural invariants of the identity_naming causal DAG."""

    pytestmark = pytest.mark.property

    def test_required_variables_present(self, model_and_config: ModelAndConfig) -> None:
        model, _ = model_and_config
        expected = {"entity", "result", "raw_input", "raw_output"}
        assert expected <= set(model.variables)

    def test_result_parents(self, model_and_config: ModelAndConfig) -> None:
        model, _ = model_and_config
        assert model.parents["result"] == ["entity"]

    def test_raw_input_parents(self, model_and_config: ModelAndConfig) -> None:
        model, _ = model_and_config
        assert model.parents["raw_input"] == ["entity"]

    def test_raw_output_parents(self, model_and_config: ModelAndConfig) -> None:
        model, _ = model_and_config
        assert model.parents["raw_output"] == ["result"]

    def test_target_variable_constant(self) -> None:
        assert TARGET_VARIABLE == "result"

    def test_cyclic_variables_constant_is_empty(self) -> None:
        assert CYCLIC_VARIABLES == set()

    def test_embeddings_module_constant_is_empty(self) -> None:
        """Per-model embeddings come from the factory; the module-level
        constant must stay empty so the loader doesn't double-bind."""
        assert EMBEDDINGS == {}


class TestIdentityNamingSampleInputProperty:
    """Well-formedness invariants for ``CausalModel.sample_input()`` traces."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_entity_is_in_entities(
        self, model_and_config: ModelAndConfig, seed: int
    ) -> None:
        model, config = model_and_config
        random.seed(seed)
        trace = model.sample_input()
        assert trace["entity"] in config.entities

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_result_matches_entity_to_result(
        self, model_and_config: ModelAndConfig, seed: int
    ) -> None:
        model, config = model_and_config
        random.seed(seed)
        trace = model.sample_input()
        assert trace["result"] == config.entity_to_result[trace["entity"]]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_input_uses_default_template(
        self, model_and_config: ModelAndConfig, seed: int
    ) -> None:
        """The causal model's ``raw_input`` mechanism uses ``templates[0]``;
        phrasing variation only happens in ``generate_dataset``."""
        model, config = model_and_config
        random.seed(seed)
        trace = model.sample_input()
        assert trace["raw_input"] == config.templates[0].format(entity=trace["entity"])

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_is_prefixed_result(
        self, model_and_config: ModelAndConfig, seed: int
    ) -> None:
        model, config = model_and_config
        random.seed(seed)
        trace = model.sample_input()
        assert trace["raw_output"] == config.output_prefix + trace["result"]


class TestIdentityNamingCounterfactualGeneratorProperty:
    """Invariants ``generate_dataset`` must preserve."""

    pytestmark = pytest.mark.property

    @given(n=st.integers(min_value=1, max_value=8))
    @_HYPOTHESIS_SETTINGS
    def test_dataset_length_matches_n(
        self, model_and_config: ModelAndConfig, n: int
    ) -> None:
        model, _ = model_and_config
        assert len(generate_dataset(model, n=n, seed=42)) == n

    def test_template_cycling(self, model_and_config: ModelAndConfig) -> None:
        """Example ``i`` uses ``templates[i % len(templates)]``."""
        model, config = model_and_config
        examples = generate_dataset(model, n=2 * len(config.templates), seed=42)
        expected = [
            config.templates[i % len(config.templates)].format(
                entity=ex["input"]["entity"]
            )
            for i, ex in enumerate(examples)
        ]
        actual = [ex["input"]["raw_input"] for ex in examples]
        assert actual == expected

    def test_generator_is_deterministic_under_fixed_seed(
        self, model_and_config: ModelAndConfig
    ) -> None:
        """Two ``generate_dataset`` invocations under the same seed produce
        byte-equal ``input`` / ``counterfactual_inputs[0]`` dicts."""
        model, _ = model_and_config
        first = generate_dataset(model, n=4, seed=123)
        second = generate_dataset(model, n=4, seed=123)
        pairs = [
            (f["input"].to_dict(), s["input"].to_dict()) for f, s in zip(first, second)
        ] + [
            (
                f["counterfactual_inputs"][0].to_dict(),
                s["counterfactual_inputs"][0].to_dict(),
            )
            for f, s in zip(first, second)
        ]
        assert all(a == b for a, b in pairs)

    def test_input_and_counterfactual_share_template(
        self, model_and_config: ModelAndConfig
    ) -> None:
        """Within a single example, ``_sample_trace`` reuses the cycle's
        ``template``, so the input and the counterfactual share the same
        template prefix."""
        model, config = model_and_config
        examples = generate_dataset(model, n=len(config.templates), seed=42)
        for i, ex in enumerate(examples):
            template = config.templates[i % len(config.templates)]
            entity_cf = ex["counterfactual_inputs"][0]["entity"]
            assert ex["counterfactual_inputs"][0]["raw_input"] == template.format(
                entity=entity_cf
            )


class TestIdentityNamingAccessorProperty:
    """The ``GET_*`` loader hooks exported by ``causal_models.py``."""

    pytestmark = pytest.mark.property

    def test_get_variable_values_returns_entity_and_result_keys(
        self, model_and_config: ModelAndConfig
    ) -> None:
        model, _ = model_and_config
        assert set(GET_VARIABLE_VALUES(model).keys()) == {"entity", "result"}

    def test_get_variable_values_entity_matches_config(
        self, model_and_config: ModelAndConfig
    ) -> None:
        model, config = model_and_config
        assert GET_VARIABLE_VALUES(model)["entity"] == config.entities

    def test_get_cyclic_variables_is_empty(
        self, model_and_config: ModelAndConfig
    ) -> None:
        model, _ = model_and_config
        assert GET_CYCLIC_VARIABLES(model) == set()

    def test_get_embeddings_entity_is_list_of_floats(
        self, model_and_config: ModelAndConfig
    ) -> None:
        """``GET_EMBEDDINGS(model)["entity"](e)`` must be a non-empty
        ``list[float]`` so downstream embedders can stack rows."""
        model, config = model_and_config
        vec = GET_EMBEDDINGS(model)["entity"](config.entities[0])
        assert isinstance(vec, list) and vec and all(isinstance(x, float) for x in vec)

    def test_get_embeddings_pitch_midi_matches_note_to_midi(self) -> None:
        """For ``pitch_midi``, the entity embedding equals
        ``[float(_note_to_midi(entity))]``."""
        cfg = IdentityNamingConfig(domain_type="pitch_midi")
        model = create_causal_model(cfg)
        assert GET_EMBEDDINGS(model)["entity"](cfg.entities[0]) == [
            float(_note_to_midi(cfg.entities[0]))
        ]

    def test_get_periodic_info_is_none(self, model_and_config: ModelAndConfig) -> None:
        model, _ = model_and_config
        assert GET_PERIODIC_INFO(model) is None

    def test_get_template_returns_default(
        self, model_and_config: ModelAndConfig
    ) -> None:
        model, config = model_and_config
        assert GET_TEMPLATE(model) == config.templates[0]

    def test_output_tokens_prepend_prefix(
        self, model_and_config: ModelAndConfig
    ) -> None:
        """``output_tokens["result"]`` carries the ``output_prefix + name`` form."""
        model, config = model_and_config
        ot = model.output_tokens["result"]
        for v in model.values["result"]:
            assert ot[v] == [config.output_prefix + str(v)]
