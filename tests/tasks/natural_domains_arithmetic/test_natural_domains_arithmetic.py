"""Property-tier invariants for the natural_domains_arithmetic factory task.

This task wraps six natural-domain arithmetic variants — ``weekdays``,
``months``, ``hours``, ``integer``, ``alphabet``, ``age`` — into a
single ``(entity, number) → result → raw_output`` DAG over a
domain-specific modular or bounded mechanism. A Hydra task YAML is
unpacked into a [`NaturalDomainConfig`][causalab.tasks.natural_domains_arithmetic.config.NaturalDomainConfig] by
``causalab.runner.helpers.resolve_task`` and handed to
``load_task("natural_domains_arithmetic", task_cfg=...)``; the factory
returns the ``CausalModel`` plus dynamic getters consumed downstream.

The factory has the largest branch matrix of any shipped task — six
domains × cyclic/non-cyclic × 1D/2D (``number_groups``) ×
single/multi-template × bounded/unbounded ``input_filter`` — so the
property classes below enumerate the reachable cells explicitly.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke is satisfied transitively by the six
NDA-flavoured baselines under ``baseline/{weekdays,months,
hours,integer,alphabet,age}.yaml`` driven by
``tests/end_to_end/test_smoke.py``. This file satisfies
property-direct. Numerical-direct is *currently blocked*:
``tests/_helpers/task_pins.py::walk_task_samples`` raises
``NotImplementedError`` for factory tasks until a serialisable
``task_cfg`` shim lands; ``numerical-direct`` stays red as the honest
signal.
"""

from __future__ import annotations

import random
import re
import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.tasks.natural_domains_arithmetic import causal_models as nda_cm
from causalab.tasks.natural_domains_arithmetic.causal_models import (
    CREATE_CAUSAL_MODEL,
    CREATE_RANDOM_CAUSAL_MODEL,
    GET_CYCLIC_VARIABLES,
    GET_EMBEDDINGS,
    GET_PERIODIC_INFO,
    GET_TEMPLATE,
    GET_VARIABLE_VALUES,
    TARGET_VARIABLE,
    create_causal_model,
    create_random_causal_model,
)
from causalab.tasks.natural_domains_arithmetic.config import (
    DOMAIN_PRESETS,
    NaturalDomainConfig,
)
from causalab.tasks.natural_domains_arithmetic.counterfactuals import (
    generate_dataset,
)
from causalab.tasks.natural_domains_arithmetic.token_positions import (
    create_token_positions,
)


# Pilot's Hypothesis settings:
#   - deadline=None: causal-model evaluation is pure-symbolic, but
#     hypothesis' default 200ms deadline can flake on cold caches.
#   - function_scoped_fixture: we don't use them, but suppressing the
#     health check is cheaper than asserting it.
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Six shipped domains. Order = parametrisation order.
ALL_DOMAINS = ("weekdays", "months", "hours", "integer", "alphabet", "age")
CYCLIC_DOMAINS = ("weekdays", "months", "hours")
NONCYCLIC_DOMAINS = ("integer", "alphabet", "age")
# Domains where create_random_causal_model fits within RANDOM_WORD_POOL (≤24).
RANDOM_BASELINE_DOMAINS = ("weekdays", "months", "hours", "integer")

# Per-domain expected cyclic-variable sets (the task README's domain table
# and its note on `periods`).
EXPECTED_CYCLIC: dict[str, set[str]] = {
    "weekdays": {"entity", "number", "result"},
    "months": {"entity", "result"},
    "hours": {"entity", "result"},
    "integer": set(),
    "alphabet": set(),
    "age": set(),
}

# Per-domain expected periodic info (the task README's note on `periods`).
EXPECTED_PERIODIC: dict[str, dict[str, int] | None] = {
    "weekdays": {"entity": 7, "number": 7, "result": 7},
    "months": {"entity": 12, "result": 12},
    "hours": {"entity": 24, "result": 24},
    "integer": None,
    "alphabet": None,
    "age": None,
}


def _make_model(domain: str):
    """Build a fresh ``CausalModel`` for ``domain`` from its preset."""
    cfg = NaturalDomainConfig(domain_type=domain)
    return create_causal_model(cfg), cfg


def _grouped_weekdays_model():
    """Build a 2D (grouped) weekdays model used by branch-matrix tests."""
    cfg = NaturalDomainConfig(domain_type="weekdays", number_groups=[[1, 3], [4, 7]])
    return create_causal_model(cfg), cfg


def _multi_template_weekdays_model():
    """Build a multi-template weekdays model used by branch-matrix tests."""
    templates = [
        "Q1: What day comes {number} days after {entity}?\nA:",
        "Q2: Starting on {entity} and adding {number} days, the day is:\nA:",
    ]
    # Pass explicit entities so __post_init__'s preset auto-fill (which
    # is gated on ``not self.entities``) doesn't overwrite our template.
    cfg = NaturalDomainConfig(
        domain_type="weekdays",
        entities=DOMAIN_PRESETS["weekdays"]["entities"],
        numbers=["one", "two", "three", "four", "five", "six", "seven"],
        number_to_int={
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
        },
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=templates,
        output_prefix=" ",
    )
    return create_causal_model(cfg), cfg


class TestNaturalDomainsArithmeticCausalModelStructureProperty:
    """DAG-structure invariants of ``create_causal_model`` across six domains."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_required_variables_present(self, domain: str) -> None:
        """Every domain's DAG names ``entity``, ``number``, ``result``,
        ``raw_input``, ``raw_output``."""
        model, _ = _make_model(domain)
        assert {"entity", "number", "result", "raw_input", "raw_output"} <= set(
            model.variables
        )

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_result_parents_are_entity_and_number(self, domain: str) -> None:
        """``result`` depends on ``entity`` and ``number`` for every domain."""
        model, _ = _make_model(domain)
        assert set(model.parents["result"]) == {"entity", "number"}

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_raw_output_parent_is_result(self, domain: str) -> None:
        """``raw_output`` is a 1-parent mechanism reading ``result``."""
        model, _ = _make_model(domain)
        assert model.parents["raw_output"] == ["result"]

    @pytest.mark.parametrize("domain", RANDOM_BASELINE_DOMAINS)
    def test_random_baseline_id_suffix(self, domain: str) -> None:
        """``create_random_causal_model`` tags the model id with ``_random``."""
        cfg = NaturalDomainConfig(domain_type=domain)
        model = create_random_causal_model(cfg)
        assert model.id.endswith("_random")

    def test_module_level_cyclic_variables_is_empty(self) -> None:
        """Static stub: ``CYCLIC_VARIABLES`` is empty; dynamic getter is authoritative."""
        assert nda_cm.CYCLIC_VARIABLES == set()

    def test_module_level_embeddings_is_empty(self) -> None:
        """Static stub: ``EMBEDDINGS`` is empty; dynamic getter is authoritative."""
        assert nda_cm.EMBEDDINGS == {}

    def test_target_variable_is_result(self) -> None:
        """The factory's intervention target is the ``result`` variable."""
        assert TARGET_VARIABLE == "result"

    def test_create_causal_model_alias_points_at_factory(self) -> None:
        """The ``CREATE_CAUSAL_MODEL`` export is the factory function."""
        assert CREATE_CAUSAL_MODEL is create_causal_model

    def test_create_random_causal_model_alias_points_at_factory(self) -> None:
        """The ``CREATE_RANDOM_CAUSAL_MODEL`` export is the random factory."""
        assert CREATE_RANDOM_CAUSAL_MODEL is create_random_causal_model


class TestNaturalDomainsArithmeticSampleInputProperty:
    """Well-formedness invariants for ``model.sample_input()`` per domain."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_sample_input_has_required_variables(self, domain: str, seed: int) -> None:
        """Every sampled trace populates the required variable set."""
        random.seed(seed)
        model, _ = _make_model(domain)
        trace = model.sample_input()
        assert {"entity", "number", "result", "raw_input", "raw_output"} <= set(
            trace.to_dict().keys()
        )

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_input_has_no_unsubstituted_placeholders(
        self, domain: str, seed: int
    ) -> None:
        """``raw_input`` contains no remaining ``{...}`` placeholders."""
        random.seed(seed)
        model, _ = _make_model(domain)
        trace = model.sample_input()
        assert not re.search(r"\{[a-zA-Z_]+\}", trace["raw_input"])

    @pytest.mark.parametrize("domain", CYCLIC_DOMAINS)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_cyclic_mechanism_is_modular(self, domain: str, seed: int) -> None:
        """Cyclic domains satisfy ``(entity_idx + number_int) % modulus == result_idx``."""
        random.seed(seed)
        model, cfg = _make_model(domain)
        trace = model.sample_input()
        entity_idx = cfg.entities.index(trace["entity"])
        number_int = cfg.number_to_int[trace["number"]]
        result_idx = cfg.entities.index(trace["result"])
        assert (entity_idx + number_int) % cfg.modulus == result_idx

    @pytest.mark.parametrize("domain", NONCYCLIC_DOMAINS)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_noncyclic_result_stays_in_result_entities(
        self, domain: str, seed: int
    ) -> None:
        """Non-cyclic domains produce results bounded by ``result_entities``."""
        random.seed(seed)
        model, cfg = _make_model(domain)
        trace = model.sample_input()
        assert trace["result"] in cfg.result_entities

    def test_age_enumerate_inputs_bounded_to_2_100(self) -> None:
        """``age`` ``input_filter`` drops over-range pairs: results ⊆ {"2",…,"100"}."""
        model, _ = _make_model("age")
        valid = {str(i) for i in range(2, 101)}
        assert all(t["result"] in valid for t in model.enumerate_inputs())

    def test_alphabet_enumerate_inputs_bounded_to_letters(self) -> None:
        """``alphabet`` ``input_filter`` drops overflow past Z: results ⊆ A–Z."""
        model, _ = _make_model("alphabet")
        assert all(
            t["result"] in string.ascii_uppercase for t in model.enumerate_inputs()
        )


class TestNaturalDomainsArithmeticCounterfactualGeneratorProperty:
    """Invariants for ``counterfactuals.generate_dataset(model, n, seed)``."""

    pytestmark = pytest.mark.property

    def test_generate_dataset_returns_list_of_length_n(self) -> None:
        """Output length matches the requested count."""
        model, _ = _make_model("weekdays")
        examples = generate_dataset(model, n=5, seed=42)
        assert len(examples) == 5

    def test_generate_dataset_examples_have_required_keys(self) -> None:
        """Each example has ``input`` and ``counterfactual_inputs`` keys."""
        model, _ = _make_model("weekdays")
        examples = generate_dataset(model, n=3, seed=42)
        assert all(
            set(ex.keys()) == {"input", "counterfactual_inputs"} for ex in examples
        )

    def test_generate_dataset_counterfactual_inputs_is_single_trace(self) -> None:
        """Per the factory contract, each entry pairs one base with one counterfactual."""
        model, _ = _make_model("weekdays")
        examples = generate_dataset(model, n=3, seed=42)
        assert all(len(ex["counterfactual_inputs"]) == 1 for ex in examples)

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_generate_dataset_examples_are_well_formed(self, domain: str) -> None:
        """Both input and counterfactual traces have the required variables."""
        model, _ = _make_model(domain)
        ex = generate_dataset(model, n=1, seed=42)[0]
        required = {"entity", "number", "result", "raw_input", "raw_output"}
        assert required <= set(ex["input"].to_dict().keys()) and required <= set(
            ex["counterfactual_inputs"][0].to_dict().keys()
        )

    def test_generate_dataset_is_deterministic_under_fixed_seed(self) -> None:
        """Two calls with the same seed produce byte-equal output."""
        model, _ = _make_model("weekdays")
        first = generate_dataset(model, n=5, seed=42)
        second = generate_dataset(model, n=5, seed=42)
        assert [ex["input"].to_dict() for ex in first] == [
            ex["input"].to_dict() for ex in second
        ]

    def test_generate_dataset_preserves_global_rng_state(self) -> None:
        """External ``random.random()`` is unaffected by ``generate_dataset``."""
        model, _ = _make_model("weekdays")
        random.seed(99)
        state = random.getstate()
        generate_dataset(model, n=3, seed=42)
        after = random.random()
        random.setstate(state)
        assert after == random.random()


class TestNaturalDomainsArithmeticDynamicGettersProperty:
    """Contracts for the ``GET_*`` dynamic getters consumed by ``load_task``."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_variable_values_has_required_keys(self, domain: str) -> None:
        """``GET_VARIABLE_VALUES`` exposes ``entity``, ``number``, ``result``."""
        model, _ = _make_model(domain)
        assert set(GET_VARIABLE_VALUES(model).keys()) == {
            "entity",
            "number",
            "result",
        }

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_variable_values_match_model_values(self, domain: str) -> None:
        """The getter's values agree with the model's underlying ``.values`` dict."""
        model, _ = _make_model(domain)
        vv = GET_VARIABLE_VALUES(model)
        assert (
            vv["entity"] == model.values["entity"]
            and vv["number"] == model.values["number"]
            and vv["result"] == model.values["result"]
        )

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_cyclic_variables_matches_expected(self, domain: str) -> None:
        """Cyclic-variable set is the documented per-domain set."""
        model, _ = _make_model(domain)
        assert GET_CYCLIC_VARIABLES(model) == EXPECTED_CYCLIC[domain]

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_embeddings_returns_callable_per_required_key(
        self, domain: str
    ) -> None:
        """``GET_EMBEDDINGS`` returns callables for ``entity``, ``number``, ``result``."""
        model, _ = _make_model(domain)
        emb = GET_EMBEDDINGS(model)
        assert all(
            k in emb and callable(emb[k]) for k in ("entity", "number", "result")
        )

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_embeddings_entity_returns_list_of_floats(self, domain: str) -> None:
        """Calling the entity embedding on a real entity returns ``list[float]``."""
        model, cfg = _make_model(domain)
        vec = GET_EMBEDDINGS(model)["entity"](cfg.entities[0])
        assert isinstance(vec, list) and all(isinstance(x, float) for x in vec)

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_periodic_info_matches_expected(self, domain: str) -> None:
        """Periodic info matches the documented per-domain spec."""
        model, _ = _make_model(domain)
        assert GET_PERIODIC_INFO(model) == EXPECTED_PERIODIC[domain]

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_get_template_is_str_for_single_template_presets(self, domain: str) -> None:
        """Every preset ships a single-string template."""
        model, _ = _make_model(domain)
        assert isinstance(GET_TEMPLATE(model), str)

    def test_output_tokens_include_canonical_and_bare_forms(self) -> None:
        """``output_tokens["result"]["Monday"]`` declares both BPE spacings."""
        model, _ = _make_model("weekdays")
        assert model.output_tokens["result"]["Monday"] == [" Monday", "Monday"]

    def test_output_tokens_are_case_sensitive(self) -> None:
        """Declared forms carry no lowercase variant — case-sensitive class columns.

        The lowercase tolerance lives in the probability grader's
        ``answer_token_forms`` pass, not the declaration.
        """
        model, _ = _make_model("weekdays")
        assert " monday" not in model.output_tokens["result"]["Monday"]

    def test_output_tokens_bare_and_spaced_regardless_of_prefix(self) -> None:
        """Hours has ``output_prefix=""``; forms of ``"3"`` still carry both spacings."""
        model, _ = _make_model("hours")
        assert model.output_tokens["result"]["3"] == [" 3", "3"]

    def test_output_tokens_forms_deduplicate(self) -> None:
        """Each value's form list contains no duplicates."""
        model, _ = _make_model("hours")
        forms = model.output_tokens["result"]["3"]
        assert len(forms) == len(set(forms))

    def test_output_tokens_keys_match_result_values_1d(self) -> None:
        """1D (non-grouped) tasks key ``output_tokens["result"]`` by each result value."""
        model, _ = _make_model("weekdays")
        assert set(model.output_tokens["result"]) == set(model.values["result"])


class TestNaturalDomainsArithmeticFactoryBranchProperty:
    """Branch-matrix coverage: ``number_groups``, multi-template, ``input_filter``,
    random baseline."""

    pytestmark = pytest.mark.property

    def test_grouped_result_values_are_tuples(self) -> None:
        """When ``number_groups`` has >1 bin, ``result`` values are ``(entity, group)`` tuples."""
        model, _ = _grouped_weekdays_model()
        assert all(isinstance(v, tuple) and len(v) == 2 for v in model.values["result"])

    def test_grouped_raw_output_strips_group(self) -> None:
        """The grouped ``raw_output`` mechanism prepends the prefix to the entity (index 0)."""
        model, _ = _grouped_weekdays_model()
        trace = model.sample_input()
        assert trace["raw_output"] == " " + trace["result"][0]

    def test_grouped_periodic_info_uses_result_0_axis(self) -> None:
        """Grouped cyclic domains expose ``result_0`` (cyclic axis); ``result_1`` (linear) is absent."""
        model, _ = _grouped_weekdays_model()
        info = GET_PERIODIC_INFO(model)
        assert "result_0" in info and "result_1" not in info

    def test_grouped_embedding_adds_group_coordinate(self) -> None:
        """The grouped result embedding appends one float (the group index)."""
        model, cfg = _grouped_weekdays_model()
        emb = GET_EMBEDDINGS(model)
        # 1D embedding for entity is length 1; grouped result is length 2.
        assert (
            len(emb["result"](model.values["result"][0]))
            == len(emb["entity"](cfg.entities[0])) + 1
        )

    def test_grouped_output_tokens_dedup_via_form_groups(self) -> None:
        """Grouped forms collapse the (entity, group) fan-out via form-groups.

        The 2D ``result`` has ``N_entities × N_groups`` tuple keys, but all groups
        of one entity share its forms — so the distinct form-groups reduce back to
        the ``N_entities`` score tokens the removed ``output_token_values`` used to
        encode. The groups are the spec's own derivation
        (``ScoringSpec.form_groups``), so they cannot drift from the grader.
        """
        model, cfg = _grouped_weekdays_model()
        entities = list(cfg.result_entities or cfg.entities)
        var_map = model.output_tokens["result"]
        assert all(isinstance(k, tuple) and len(k) == 2 for k in var_map)
        assert model.scoring is not None
        groups = model.scoring.form_groups("result")
        assert len(groups) == len(entities)
        assert groups == [[f" {e}", e] for e in entities]

    def test_multi_template_makes_template_a_variable(self) -> None:
        """Multi-template factory adds ``template`` to ``model.variables``."""
        model, _ = _multi_template_weekdays_model()
        assert "template" in model.variables

    def test_multi_template_raw_input_parents_include_template(self) -> None:
        """``raw_input`` mechanism reads ``template`` alongside ``entity`` / ``number``."""
        model, _ = _multi_template_weekdays_model()
        assert "template" in model.parents["raw_input"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_multi_template_fill_template_uses_trace_template(self, seed: int) -> None:
        """A sampled trace's ``raw_input`` is built from its own ``template`` value."""
        random.seed(seed)
        model, _ = _multi_template_weekdays_model()
        trace = model.sample_input()
        expected = trace["template"].format(
            entity=trace["entity"], number=trace["number"]
        )
        assert trace["raw_input"] == expected

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_age_input_filter_drops_over_sum(self, seed: int) -> None:
        """``age`` sampled traces never have ``int(entity)+int(number) > 100``."""
        random.seed(seed)
        model, _ = _make_model("age")
        trace = model.sample_input()
        assert int(trace["entity"]) + int(trace["number"]) <= 100

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_alphabet_input_filter_drops_overflow_past_z(self, seed: int) -> None:
        """``alphabet`` sampled traces never produce ``result`` past ``Z``."""
        random.seed(seed)
        model, _ = _make_model("alphabet")
        trace = model.sample_input()
        assert ord(trace["result"]) <= ord("Z")

    @pytest.mark.parametrize("domain", RANDOM_BASELINE_DOMAINS)
    def test_random_baseline_entities_disjoint_from_preset(self, domain: str) -> None:
        """Random baselines replace real entities with random words (disjoint pool)."""
        cfg = NaturalDomainConfig(domain_type=domain)
        rmodel = create_random_causal_model(cfg)
        assert set(rmodel.values["entity"]).isdisjoint(set(cfg.entities))

    @pytest.mark.parametrize("domain", RANDOM_BASELINE_DOMAINS)
    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_baseline_is_modular(self, domain: str, seed: int) -> None:
        """Random baselines wrap modular arithmetic over the random alphabet."""
        random.seed(seed)
        cfg = NaturalDomainConfig(domain_type=domain)
        rmodel = create_random_causal_model(cfg)
        trace = rmodel.sample_input()
        entities = rmodel.values["entity"]
        n = len(entities)
        idx = entities.index(trace["entity"])
        k = cfg.number_to_int[trace["number"]]
        assert entities[(idx + k) % n] == trace["result"]


class TestNaturalDomainsArithmeticConfigProperty:
    """``NaturalDomainConfig.__post_init__`` and ``DOMAIN_PRESETS`` invariants."""

    pytestmark = pytest.mark.property

    def test_invalid_domain_type_raises(self) -> None:
        """Unknown ``domain_type`` raises ``ValueError`` with a helpful message."""
        with pytest.raises(ValueError, match="domain_type must be"):
            NaturalDomainConfig(domain_type="not_a_real_domain")

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_preset_auto_fill_populates_entities(self, domain: str) -> None:
        """Bare ``NaturalDomainConfig(domain_type=X)`` auto-fills ``entities``."""
        cfg = NaturalDomainConfig(domain_type=domain)
        assert len(cfg.entities) > 0

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_preset_auto_fill_populates_numbers_and_template(self, domain: str) -> None:
        """Auto-fill populates ``numbers``, ``number_to_int``, and ``template``."""
        cfg = NaturalDomainConfig(domain_type=domain)
        assert (
            len(cfg.numbers) > 0 and len(cfg.number_to_int) > 0 and cfg.template != ""
        )

    def test_number_range_overflow_raises(self) -> None:
        """``number_range`` past ``_ALL_NUMBER_WORDS`` length raises ``ValueError``."""
        # weekdays preset uses ``number_range`` rather than explicit ``numbers``.
        with pytest.raises(ValueError, match="exceeds available"):
            NaturalDomainConfig(domain_type="weekdays", number_range=999)

    def test_explicit_result_entities_override_skips_preset(self) -> None:
        """An explicit ``result_entities`` skips the preset's auto-fill for that field."""
        custom = ["X", "Y", "Z"]
        cfg = NaturalDomainConfig(domain_type="weekdays", result_entities=custom)
        assert cfg.result_entities == custom

    def test_domain_presets_keys_match_expected_six(self) -> None:
        """``DOMAIN_PRESETS`` covers exactly the six shipped variants."""
        assert set(DOMAIN_PRESETS.keys()) == set(ALL_DOMAINS)

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_every_preset_has_entities_and_template(self, domain: str) -> None:
        """Every preset ships ``entities`` and a ``template`` field."""
        preset = DOMAIN_PRESETS[domain]
        assert "entities" in preset and "template" in preset

    @pytest.mark.parametrize("domain", ALL_DOMAINS)
    def test_every_preset_has_number_range_or_explicit_numbers(
        self, domain: str
    ) -> None:
        """Every preset specifies either ``number_range`` or explicit ``numbers``."""
        preset = DOMAIN_PRESETS[domain]
        assert "number_range" in preset or "numbers" in preset


class TestNaturalDomainsArithmeticTokenPositionsProperty:
    """``create_token_positions`` defensive contract (LM-free).

    Multi-template dispatch and pipeline-coupled keys require a real
    ``LMPipeline`` and so live in ``tests/tasks/test_loader.py`` /
    runner smoke. We only pin the ``ValueError`` branch here.
    """

    pytestmark = pytest.mark.property

    def test_both_template_and_templates_none_raises(self) -> None:
        """Calling with neither ``template`` nor ``templates`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="template is required"):
            create_token_positions(pipeline=None)
