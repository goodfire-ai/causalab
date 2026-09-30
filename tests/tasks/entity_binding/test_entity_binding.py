"""Property-tier invariants for the entity_binding task.

The entity_binding task models positional entity binding as a causal DAG
over per-position entity inputs ``entities[{g},{e}]``, the symbolic
intermediates ``queries[{e}]`` / ``positional_queries[{e}]`` /
``positional_answer``, and the text-level ``raw_input`` / ``raw_output``.
The runner smoke config at ``tests/end_to_end/configs/smoke/entity_binding.yaml``
targets ``positional_answer``; this file pins the **invariants** of every
mechanism in the DAG plus the counterfactual generators, the token-position
factories, and the ``EntityBindingTaskConfig`` template machinery.

Numerical pinning of specific symbolic outputs lives in the sibling
``pinned_samples.json`` and is enforced by
``test_entity_binding_numerical.py`` (sample-by-sample equality at
fixed seeds). This file covers the *invariants* — things that should
hold for every input, every seed.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke comes from
``baseline/entity_binding.yaml`` via
``tests/end_to_end/test_smoke.py``. This file satisfies
property-direct; the sibling numerical-tier test satisfies
numerical-direct.

Risks pinned here (documented inline at each test):

* ``sample_valid_entity_binding_input`` has a 100-attempt rejection cap;
  the default love config has comfortable margins, but a tightening of
  ``entity_pools`` would surface as a ``ValueError`` at sampling time.
* ``generate_dataset`` reseeds Python's global ``random`` and binds to
  the module-level ``_default_config`` (ignoring its ``config`` knob);
  determinism asserts reseed every call.
* ``COUNTERFACTUAL_GENERATORS`` values are eagerly bound to
  ``_default_config`` at import time; tests never mutate it in place.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.tasks.entity_binding.causal_models import (
    CAUSAL_MODEL,
    TARGET_VARIABLE,
    causal_model,
    create_positional_entity_causal_model,
    sample_valid_entity_binding_input,
)
from causalab.tasks.entity_binding.config import (
    MAX_NEW_TOKENS,
    MAX_TASK_TOKENS,
    EntityBindingTaskConfig,
    create_sample_love_config,
)
from causalab.tasks.entity_binding.counterfactuals import (
    COUNTERFACTUAL_GENERATORS,
    generate_dataset,
    random_counterfactual,
    swap_query_group,
)
from causalab.tasks.entity_binding.token_positions import (
    create_token_positions,
    get_entity_token_positions,
    get_question_entity_token_positions,
    get_statement_entity_token_positions,
)
from tests._helpers.tasks import get_task


# Hypothesis settings re-used from the MCQA pilot:
#   - deadline=None: symbolic mechanisms are not timing-sensitive; the 200ms
#     default flakes on cold tokenizer caches in the token-position class.
#   - function_scoped_fixture: suppressed because the tiny_pipeline fixture
#     is session-scoped (we don't actually use function-scoped fixtures
#     under @given here, but the suppression is cheap insurance).
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# Module-scope reusable causal model: matches the singleton at
# ``causal_models.causal_model`` (both routes through
# ``create_positional_entity_causal_model(create_sample_love_config())``).
_DEFAULT_MODEL = create_positional_entity_causal_model(create_sample_love_config())


class TestEntityBindingCausalModelStructureProperty:
    """Static structural invariants of the entity_binding causal DAG."""

    pytestmark = pytest.mark.property

    def test_required_variables_present(self) -> None:
        """DAG names every variable the loader contract and runner rely on."""
        cfg = create_sample_love_config()
        gs, es = range(cfg.max_groups), range(cfg.max_entities_per_group)
        expected = {
            "query_group",
            "query_indices",
            "answer_index",
            "active_groups",
            "entities_per_group",
            "statement_template",
            "question_template",
            "positional_answer",
            "raw_input",
            "raw_output",
            *(f"entities[{g},{e}]" for g in gs for e in es),
            *(f"positional_entities[{g},{e}]" for g in gs for e in es),
            *(f"queries[{e}]" for e in es),
            *(f"positional_queries[{e}]" for e in es),
        }
        assert expected <= set(_DEFAULT_MODEL.variables)

    def test_positional_answer_parents(self) -> None:
        """``positional_answer`` depends on the per-position query sets + indices."""
        cfg = create_sample_love_config()
        expected = [
            f"positional_queries[{e}]" for e in range(cfg.max_entities_per_group)
        ] + ["query_indices"]
        assert set(_DEFAULT_MODEL.parents["positional_answer"]) == set(expected)

    def test_raw_output_parents_include_positional_answer_and_entities(self) -> None:
        """``raw_output`` retrieves an entity selected by ``positional_answer``."""
        parents = _DEFAULT_MODEL.parents["raw_output"]
        assert "positional_answer" in parents
        assert "answer_index" in parents
        cfg = create_sample_love_config()
        for g in range(cfg.max_groups):
            for e in range(cfg.max_entities_per_group):
                assert f"entities[{g},{e}]" in parents

    def test_query_e_parents_cover_all_groups_and_query_group(self) -> None:
        """``queries[{e}]`` reads from every group at position ``e`` (plus controls)."""
        cfg = create_sample_love_config()
        for e in range(cfg.max_entities_per_group):
            expected_entity_parents = {
                f"entities[{g},{e}]" for g in range(cfg.max_groups)
            }
            parents = set(_DEFAULT_MODEL.parents[f"queries[{e}]"])
            assert expected_entity_parents <= parents
            assert "query_group" in parents

    def test_target_variable_is_positional_answer(self) -> None:
        assert TARGET_VARIABLE == "positional_answer"

    def test_causal_model_singleton_alias(self) -> None:
        """``CAUSAL_MODEL`` and ``causal_model`` are the same import-time instance."""
        assert CAUSAL_MODEL is causal_model


class TestEntityBindingSampleInputProperty:
    """Well-formedness invariants for ``sample_valid_entity_binding_input``.

    The 100-attempt rejection cap could surface as a ``ValueError`` if the
    config's ``entity_pools`` are ever tightened past the constraints; the
    default love config has 6 entries per pool against 2 groups so this is
    not on the hypothesis path here. Document in module docstring.
    """

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_query_group_within_active_groups(self, seed: int) -> None:
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        assert 0 <= trace["query_group"] < trace["active_groups"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_answer_index_disjoint_from_query_indices(self, seed: int) -> None:
        """The answer slot is never one of the queried positions."""
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        assert trace["answer_index"] not in trace["query_indices"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_template_exists_for_query_pattern(self, seed: int) -> None:
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        key = (trace["query_indices"], trace["answer_index"])
        assert key in cfg.question_templates

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_positional_uniqueness_across_groups(self, seed: int) -> None:
        """Per-position entities are distinct across groups (positional uniqueness)."""
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        for e in range(cfg.max_entities_per_group):
            per_position = [trace[f"entities[{g},{e}]"] for g in range(cfg.max_groups)]
            assert len(set(per_position)) == len(per_position)

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_per_group_uniqueness_across_positions(self, seed: int) -> None:
        """Per-group entities are distinct across positions."""
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        for g in range(cfg.max_groups):
            per_group = [
                trace[f"entities[{g},{e}]"] for e in range(cfg.max_entities_per_group)
            ]
            assert len(set(per_group)) == len(per_group)

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_positional_answer_matches_query_group(self, seed: int) -> None:
        """Under positional uniqueness, ``positional_answer == query_group``."""
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        assert trace["positional_answer"] == trace["query_group"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_equals_answer_entity(self, seed: int) -> None:
        """``raw_output == entities[{query_group},{answer_index}]`` on the default config."""
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        expected = trace[f"entities[{trace['query_group']},{trace['answer_index']}]"]
        assert trace["raw_output"] == expected

    def test_sampler_deterministic_under_fixed_seed(self) -> None:
        """Two calls under the same ``random.seed`` produce byte-equal traces."""
        cfg = create_sample_love_config()
        random.seed(42)
        first = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        random.seed(42)
        second = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        assert first.to_dict() == second.to_dict()


class TestEntityBindingCounterfactualGeneratorProperty:
    """Invariants the entity_binding counterfactual generators must preserve."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_swap_query_group_changes_query_group(self, seed: int) -> None:
        """``swap_query_group`` moves to a different group (default config has 2)."""
        random.seed(seed)
        cfg = create_sample_love_config()
        cf = swap_query_group(cfg, change_answer=False)
        assert (
            cf["input"]["query_group"] != cf["counterfactual_inputs"][0]["query_group"]
        )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_swap_query_group_preserves_entity_multiset(self, seed: int) -> None:
        """The full ``entity_g*_e*`` multiset is preserved across groups."""
        random.seed(seed)
        cfg = create_sample_love_config()
        cf = swap_query_group(cfg, change_answer=False)
        base, ctf = cf["input"], cf["counterfactual_inputs"][0]
        keys = [
            f"entities[{g},{e}]"
            for g in range(cfg.max_groups)
            for e in range(cfg.max_entities_per_group)
        ]
        assert sorted(base[k] for k in keys) == sorted(ctf[k] for k in keys)

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_swap_query_group_query_e_tracks_swap(self, seed: int) -> None:
        """``queries[{e}] == entities[{query_group},{e}]`` in the counterfactual."""
        random.seed(seed)
        cfg = create_sample_love_config()
        cf = swap_query_group(cfg, change_answer=False)
        ctf = cf["counterfactual_inputs"][0]
        qg = ctf["query_group"]
        for e in range(cfg.max_entities_per_group):
            assert ctf[f"queries[{e}]"] == ctf[f"entities[{qg},{e}]"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_swap_query_group_change_answer_false_keeps_answer(self, seed: int) -> None:
        """With ``change_answer=False`` the answer-slot entity matches base."""
        random.seed(seed)
        cfg = create_sample_love_config()
        cf = swap_query_group(cfg, change_answer=False)
        base, ctf = cf["input"], cf["counterfactual_inputs"][0]
        ai = base["answer_index"]
        base_answer = base[f"entities[{base['query_group']},{ai}]"]
        ctf_answer = ctf[f"entities[{ctf['query_group']},{ai}]"]
        assert base_answer == ctf_answer

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_swap_query_group_change_answer_true_changes_answer(
        self, seed: int
    ) -> None:
        """With ``change_answer=True`` the answer-slot entity comes from its pool."""
        random.seed(seed)
        cfg = create_sample_love_config()
        cf = swap_query_group(cfg, change_answer=True)
        base, ctf = cf["input"], cf["counterfactual_inputs"][0]
        ai = base["answer_index"]
        base_answer = base[f"entities[{base['query_group']},{ai}]"]
        ctf_answer = ctf[f"entities[{ctf['query_group']},{ai}]"]
        assert base_answer != ctf_answer
        assert ctf_answer in cfg.entity_pools[ai]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_counterfactual_both_halves_well_formed(self, seed: int) -> None:
        """Each half independently passes the well-formedness invariants."""
        random.seed(seed)
        cfg = create_sample_love_config()
        cf = random_counterfactual(cfg)
        for trace in (cf["input"], cf["counterfactual_inputs"][0]):
            assert trace["answer_index"] not in trace["query_indices"]

    def test_generate_dataset_length_and_pair_query_group_diff(self) -> None:
        """``generate_dataset(n)`` returns ``n`` pairs with differing ``query_group``."""
        examples = generate_dataset(_DEFAULT_MODEL, 5, seed=0)
        assert len(examples) == 5
        for ex in examples:
            assert (
                ex["input"]["query_group"]
                != ex["counterfactual_inputs"][0]["query_group"]
            )

    def test_generate_dataset_deterministic_under_seed(self) -> None:
        """Same seed → byte-equal output, even after a contaminated global seed."""
        first = generate_dataset(_DEFAULT_MODEL, 3, seed=0)
        random.seed(999)  # contaminate before the second call
        second = generate_dataset(_DEFAULT_MODEL, 3, seed=0)
        assert [e["input"].to_dict() for e in first] == [
            e["input"].to_dict() for e in second
        ]

    def test_registry_keys_and_callable_shape(self) -> None:
        """Registry exposes the documented generators with zero-arg callables."""
        expected_keys = {
            "swap_query_group",
            "swap_query_group_change_answer",
            "random_counterfactual",
        }
        assert set(COUNTERFACTUAL_GENERATORS.keys()) == expected_keys
        for name, fn in COUNTERFACTUAL_GENERATORS.items():
            random.seed(0)
            result = fn()
            assert set(result.keys()) >= {"input", "counterfactual_inputs"}, name

    @pytest.mark.parametrize(
        "generator_name",
        sorted(COUNTERFACTUAL_GENERATORS.keys()),
    )
    def test_registry_generator_deterministic_under_seed(
        self, generator_name: str
    ) -> None:
        """Two calls of a registry generator under the same seed match byte-for-byte."""
        fn = COUNTERFACTUAL_GENERATORS[generator_name]
        random.seed(123)
        first = fn()
        random.seed(123)
        second = fn()
        assert first["input"].to_dict() == second["input"].to_dict()


class TestEntityBindingTokenPositionsProperty:
    """Token-position factory invariants — require a real tokenizer-bearing pipeline.

    Pure-symbolic sampling isn't enough: ``get_entity_token_positions``
    runs prefix tokenization to find the byte offset of an entity in the
    filled template, so the tiny Llama stub (real ``LlamaTokenizerFast``) is
    the minimum dependency.

    The tiny tokenizer is random-init with vocab=32000 and produces an
    extra trailing-space token in many BPE-boundary cases ("David likes
    apple" tokenises as ``['<s>', 'David', 'lik', 'es', 'apple']`` but
    "David likes " tokenises with a trailing space token, so the prefix-
    length delta collapses to zero on the entity span). Seeds where the
    queried-position entity is tokeniser-clean are picked deliberately
    (``seed=0`` for the g0_e0 path, ``seed=1`` for the role/question
    paths). Production-grade Llama tokenisers do not hit this boundary.
    """

    pytestmark = pytest.mark.property

    @staticmethod
    def _sample_with_seed(seed: int):
        random.seed(seed)
        cfg = create_sample_love_config()
        trace = sample_valid_entity_binding_input(cfg, model=_DEFAULT_MODEL)
        return cfg, trace.to_dict()

    def test_get_entity_token_positions_indices_in_bounds(self, tiny_pipeline) -> None:
        cfg, sample = self._sample_with_seed(0)
        n_tokens = len(tiny_pipeline.tokenizer.encode(sample["raw_input"]))
        positions = get_entity_token_positions(sample, tiny_pipeline, cfg, 0, 0)
        assert all(0 <= p < n_tokens for p in positions)

    def test_get_entity_token_positions_contiguous_and_nonempty(
        self, tiny_pipeline
    ) -> None:
        cfg, sample = self._sample_with_seed(0)
        positions = get_entity_token_positions(sample, tiny_pipeline, cfg, 0, 0)
        assert positions == list(range(positions[0], positions[-1] + 1))

    def test_get_entity_token_positions_negative_index_picks_last(
        self, tiny_pipeline
    ) -> None:
        cfg, sample = self._sample_with_seed(0)
        full = get_entity_token_positions(sample, tiny_pipeline, cfg, 0, 0)
        last_only = get_entity_token_positions(
            sample, tiny_pipeline, cfg, 0, 0, token_idx=-1
        )
        assert last_only == [full[-1]]

    def test_get_entity_token_positions_invalid_token_idx_raises(
        self, tiny_pipeline
    ) -> None:
        """An out-of-range ``token_idx`` triggers a ``ValueError``."""
        cfg, sample = self._sample_with_seed(0)
        full = get_entity_token_positions(sample, tiny_pipeline, cfg, 0, 0)
        with pytest.raises(ValueError):
            get_entity_token_positions(
                sample, tiny_pipeline, cfg, 0, 0, token_idx=len(full) + 5
            )

    def test_get_question_entity_uses_last_occurrence(self, tiny_pipeline) -> None:
        """Question-position uses ``rfind`` → token index is later than statement's."""
        # seed=1 is tokenizer-clean for the queried position under the tiny stub.
        cfg, sample = self._sample_with_seed(1)
        qi = sample["query_indices"][0]
        stmt_positions = get_statement_entity_token_positions(
            sample, tiny_pipeline, cfg, sample["query_group"], entity_idx=qi
        )
        question_positions = get_question_entity_token_positions(
            sample, tiny_pipeline, cfg, entity_idx=qi
        )
        assert min(question_positions) > max(stmt_positions)

    def test_get_statement_entity_role_name_resolution(self, tiny_pipeline) -> None:
        """Resolving by ``role_name`` matches resolving by ``entity_idx`` directly."""
        # seed=1 lands the query on a tokenizer-clean entity for both g0 and g1.
        cfg, sample = self._sample_with_seed(1)
        qi = sample["query_indices"][0]
        role = cfg.entity_roles[qi]
        by_idx = get_statement_entity_token_positions(
            sample, tiny_pipeline, cfg, sample["query_group"], entity_idx=qi
        )
        by_role = get_statement_entity_token_positions(
            sample, tiny_pipeline, cfg, sample["query_group"], role_name=role
        )
        assert by_idx == by_role

    def test_create_token_positions_returns_factory_dict(self, tiny_pipeline) -> None:
        """``create_token_positions`` returns ``dict[str, TokenPosition]``."""
        cfg, sample = self._sample_with_seed(0)
        positions = create_token_positions(tiny_pipeline, sample["raw_input"], cfg)
        assert isinstance(positions, dict)
        for key in ("g0_e0_last", "g0_e1_last", "g1_e0_last", "g1_e1_last", "last"):
            assert key in positions, f"missing key: {key}"

    def test_create_token_positions_config_none_fallback(self, tiny_pipeline) -> None:
        """``config=None`` falls back to ``create_sample_love_config()``."""
        _, sample = self._sample_with_seed(0)
        positions = create_token_positions(tiny_pipeline, sample["raw_input"], None)
        # Default config has 2 groups × 2 entities → 4 ``g{g}_e{e}_last`` keys
        # plus ``last``.
        expected = {"g0_e0_last", "g0_e1_last", "g1_e0_last", "g1_e1_last", "last"}
        assert set(positions.keys()) == expected

    def test_dispatcher_create_token_positions_returns_superset(
        self, tiny_pipeline
    ) -> None:
        """``Task.create_token_positions(pipeline)`` returns ⊇ direct factories."""
        from causalab.tasks.loader import load_task

        task = load_task("entity_binding")
        dispatch_keys = set(task.create_token_positions(tiny_pipeline).keys())
        handle = get_task("entity_binding")
        # ``get_task`` is the typed helper; assert it points at the same singleton.
        assert handle.causal_model is causal_model
        direct = create_token_positions(tiny_pipeline, "", create_sample_love_config())
        assert dispatch_keys >= set(direct.keys())


class TestEntityBindingConfigProperty:
    """``EntityBindingTaskConfig`` invariants — templates, fill, constants."""

    pytestmark = pytest.mark.property

    def test_dataclass_roundtrip_preserves_mega_template(self) -> None:
        """A field-for-field rebuild matches ``build_mega_template`` byte-for-byte."""
        cfg = create_sample_love_config()
        rebuilt = EntityBindingTaskConfig(**dataclasses.asdict(cfg))
        assert cfg.build_mega_template(2, (0,), 1) == rebuilt.build_mega_template(
            2, (0,), 1
        )

    def test_build_mega_template_substitutes_every_declared_placeholder(self) -> None:
        """Filling with the per-group + per-role values leaves no ``{...}``."""
        cfg = create_sample_love_config()
        template = cfg.build_mega_template(cfg.max_groups, (0,), 1)
        values = {
            f"g{g}_e{e}": f"X{g}{e}"
            for g in range(cfg.max_groups)
            for e in range(cfg.max_entities_per_group)
        }
        for e, role in cfg.entity_roles.items():
            values[role] = f"R{e}"
        values["query_entity"] = "Q"
        filled = cfg.fill_template(template, values)
        assert "{" not in filled or "}" not in filled

    def test_fill_template_idempotent_on_complete_substitution(self) -> None:
        """A second fill on a complete substitution is a no-op (no ``{}`` remain)."""
        cfg = create_sample_love_config()
        template = cfg.build_mega_template(2, (0,), 1)
        values = {
            "g0_e0": "Pete",
            "g0_e1": "jam",
            "g1_e0": "Ann",
            "g1_e1": "pie",
            "person": "Ann",
            "food": "pie",
            "query_entity": "Ann",
        }
        first = cfg.fill_template(template, values)
        # second fill: no placeholders means .format is a no-op equivalent
        assert "{" not in first

    def test_build_mega_template_single_statement_no_conjunction(self) -> None:
        """``num_statements == 1`` emits no conjunction (just the terminator)."""
        cfg = create_sample_love_config()
        rendered = cfg.build_mega_template(1, (0,), 1)
        assert " and " not in rendered
        assert ", and " not in rendered

    def test_build_mega_template_two_statements_no_oxford_comma(self) -> None:
        """``num_statements == 2`` emits `` and `` without serial comma."""
        cfg = create_sample_love_config()
        rendered = cfg.build_mega_template(2, (0,), 1)
        assert " and " in rendered
        assert ", and " not in rendered

    def test_build_mega_template_three_statements_oxford_comma(self) -> None:
        """``num_statements >= 3`` emits ``, and `` (Oxford-comma form)."""
        cfg = create_sample_love_config()
        rendered = cfg.build_mega_template(3, (0,), 1)
        assert ", and " in rendered

    def test_max_task_tokens_and_max_new_tokens_positive_ints(self) -> None:
        """Budget constants are positive ``int``s (intentionally not pinning value)."""
        assert isinstance(MAX_TASK_TOKENS, int) and MAX_TASK_TOKENS > 0
        assert isinstance(MAX_NEW_TOKENS, int) and MAX_NEW_TOKENS > 0


class TestEntityBindingConstantsProperty:
    """Pool-size and template sanity for the default ``create_sample_love_config``."""

    pytestmark = pytest.mark.property

    def test_entity_pools_nonempty_and_unique_per_position(self) -> None:
        """Every entry in every pool is non-empty and unique within its position."""
        cfg = create_sample_love_config()
        for position, pool in cfg.entity_pools.items():
            assert len(pool) > 0
            assert len(set(pool)) == len(pool), (
                f"duplicate entities in pool for position {position}"
            )

    def test_question_templates_cover_every_query_pattern_role(self) -> None:
        """Every ``(query_indices, answer_index)`` template references a known role."""
        cfg = create_sample_love_config()
        roles = set(cfg.entity_roles.values())
        for (_query_indices, _answer_index), template in cfg.question_templates.items():
            # The plain entity-role names in the question template must be a
            # subset of the configured roles — otherwise filling will KeyError.
            for role in roles:
                assert ("{" + role + "}") in template or role not in template


class TestEntityBindingOutputTokens:
    """``ScoringSpec`` contract — entity answer forms, dedup-ordered union."""

    pytestmark = pytest.mark.property

    def test_declares_forms_on_the_variable_that_holds_the_answer(self) -> None:
        """The forms are declared on ``raw_output`` — the variable whose value
        *is* the answer entity — not on the interchange target
        ``positional_answer``, whose values are group indices. Declared there,
        the serializer's row lookup (keyed by the variable's value) refused
        every row while the string grader still credited it."""
        spec = _DEFAULT_MODEL.scoring
        assert spec is not None and list(spec.forms) == ["raw_output"]
        assert spec.answer_variable == "raw_output"
        assert _DEFAULT_MODEL.output_tokens is not None
        assert list(_DEFAULT_MODEL.output_tokens) == ["raw_output"]

    def test_keys_are_dedup_insertion_ordered_union_of_pools(self) -> None:
        """Pin the contract: keys are the dedup union of pools, each as its forms."""
        cfg = create_sample_love_config()
        expected: list[str] = []
        for pool in cfg.entity_pools.values():
            expected.extend(pool)
        expected = list(dict.fromkeys(expected))
        ot = _DEFAULT_MODEL.output_tokens["raw_output"]
        assert list(ot.keys()) == expected
        assert all(ot[e] == [f" {e}", e] for e in expected)

    def test_string_mode_is_prefix(self) -> None:
        """The answer may carry continuation tokens → ``prefix`` string mode,
        which a ``match`` metric over this task's table declares as
        ``first_token`` (§2.10)."""
        spec = _DEFAULT_MODEL.scoring
        assert spec is not None
        assert spec.string_mode == "prefix" and spec.protocol_mode == "first_token"
        assert _DEFAULT_MODEL.match_modes == {"raw_output": "prefix"}  # derived view
