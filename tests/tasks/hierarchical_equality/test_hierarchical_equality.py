"""Property-tier invariants for the hierarchical_equality task.

The hierarchical_equality (HE) task casts a pure-symbolic ICL puzzle as a
[`CausalModel`][causalab.causal.model.CausalModel] over four input letters
``var_1..var_4`` with mechanisms ``(var_1, var_2) → left_equality``,
``(var_3, var_4) → right_equality``, ``(left_equality, right_equality) →
result_equality → raw_output`` (``"1"``/``"0"``).

Numerical pinning of specific symbolic outputs lives in the sibling
``pinned_samples.json`` and is enforced by
``test_hierarchical_equality_numerical.py`` (sample-by-sample equality
at fixed seeds). This file covers the *invariants* — things that should
hold for every input, every seed.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke comes from
``baseline/hierarchical_equality.yaml`` via
``tests/end_to_end/test_smoke.py``. This file satisfies property-direct;
the sibling numerical-tier test satisfies numerical-direct.

Dead modules (``checker.py``) are intentionally not tested — per the
consolidation recipe they show ✗ on the dashboard until the cleanup PR
wires them in or deletes them.
"""

from __future__ import annotations

import random
from typing import Any, Callable, cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.tasks.token_positions import LMPipeline, TokenPosition
from causalab.tasks.hierarchical_equality.causal_models import (
    CAUSAL_MODEL,
    TARGET_VARIABLE,
)
from causalab.tasks.hierarchical_equality.config import (
    LETTERS,
    NUM_ICL_EXAMPLES,
    PATTERNS,
    PROMPT_MODE,
    TASK_NAME,
)
from causalab.tasks.hierarchical_equality.counterfactuals import (
    COUNTERFACTUAL_GENERATORS,
    generate_dataset,
    random_counterfactual,
    sample_balanced_input,
)
from causalab.tasks.hierarchical_equality.templates import (
    TEMPLATES,
    _sample_pattern_values,  # type: ignore[reportPrivateUsage]  # re-imported by counterfactuals; pinned here.
    fill_template,
    get_func_name,
)
from causalab.tasks.hierarchical_equality.token_positions import (
    create_token_positions,
)
from tests._helpers.tasks import get_task

# Hypothesis settings used across this file:
#   - deadline=None: causal-model evaluation is pure-symbolic and not
#     timing-sensitive, but hypothesis' default 200ms deadline can flake
#     on cold caches.
#   - function_scoped_fixture: silenced rather than asserted.
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestHierarchicalEqualityStructureProperty:
    """Static structural invariants of the HE causal DAG."""

    pytestmark = pytest.mark.property

    def test_required_variables_present(self) -> None:
        """The DAG names every variable the loader contract relies on."""
        expected = {
            "template",
            "var_1",
            "var_2",
            "var_3",
            "var_4",
            "left_equality",
            "right_equality",
            "result_equality",
            "raw_input",
            "raw_output",
        }
        assert expected <= set(CAUSAL_MODEL.variables), (
            f"missing variables: {expected - set(CAUSAL_MODEL.variables)}"
        )

    def test_left_equality_parents(self) -> None:
        assert CAUSAL_MODEL.parents["left_equality"] == ["var_1", "var_2"]

    def test_right_equality_parents(self) -> None:
        assert CAUSAL_MODEL.parents["right_equality"] == ["var_3", "var_4"]

    def test_result_equality_parents(self) -> None:
        assert CAUSAL_MODEL.parents["result_equality"] == [
            "left_equality",
            "right_equality",
        ]

    def test_raw_input_parents(self) -> None:
        assert CAUSAL_MODEL.parents["raw_input"] == [
            "template",
            "var_1",
            "var_2",
            "var_3",
            "var_4",
            "icl_seed",
        ]

    def test_raw_output_parents(self) -> None:
        assert CAUSAL_MODEL.parents["raw_output"] == ["result_equality"]

    def test_target_variable(self) -> None:
        assert TARGET_VARIABLE == "result_equality"


class TestHierarchicalEqualityMechanismProperty:
    """End-to-end mechanism invariants via ``CAUSAL_MODEL.sample_input()``."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_var_letters_in_pool(self, seed: int) -> None:
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        assert all(trace[f"var_{i}"] in LETTERS for i in range(1, 5))

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_sample_input_is_deterministic_under_fixed_seed(self, seed: int) -> None:
        """Re-seeding ``random`` reproduces a byte-equal trace."""
        random.seed(seed)
        first = CAUSAL_MODEL.sample_input()
        random.seed(seed)
        second = CAUSAL_MODEL.sample_input()
        assert first.to_dict() == second.to_dict()

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_left_equality_predicate(self, seed: int) -> None:
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        assert trace["left_equality"] == (trace["var_1"] == trace["var_2"])

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_right_equality_predicate(self, seed: int) -> None:
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        assert trace["right_equality"] == (trace["var_3"] == trace["var_4"])

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_result_equality_predicate(self, seed: int) -> None:
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        assert trace["result_equality"] == (
            trace["left_equality"] == trace["right_equality"]
        )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_is_string_of_result(self, seed: int) -> None:
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        assert trace["raw_output"] == ("1" if trace["result_equality"] else "0")

    @given(
        seed=st.integers(min_value=0, max_value=10_000),
        perm_seed=st.integers(min_value=0, max_value=10_000),
    )
    @_HYPOTHESIS_SETTINGS
    def test_letter_permutation_preserves_equality_pattern(
        self, seed: int, perm_seed: int
    ) -> None:
        """Bijecting ``LETTERS`` leaves equality outputs invariant."""
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        rng = random.Random(perm_seed)
        permuted = list(LETTERS)
        rng.shuffle(permuted)
        sigma = dict(zip(LETTERS, permuted))
        substituted = CAUSAL_MODEL.new_trace(
            {
                "template": trace["template"],
                "icl_seed": trace["icl_seed"],
                "var_1": sigma[trace["var_1"]],
                "var_2": sigma[trace["var_2"]],
                "var_3": sigma[trace["var_3"]],
                "var_4": sigma[trace["var_4"]],
            }
        )
        assert (
            substituted["left_equality"] == trace["left_equality"]
            and substituted["right_equality"] == trace["right_equality"]
            and substituted["result_equality"] == trace["result_equality"]
        )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_right_pair_swap_symmetry(self, seed: int) -> None:
        """Swapping ``var_3 <-> var_4`` leaves right/result equality fixed."""
        random.seed(seed)
        trace = CAUSAL_MODEL.sample_input()
        swapped = CAUSAL_MODEL.new_trace(
            {
                "template": trace["template"],
                "icl_seed": trace["icl_seed"],
                "var_1": trace["var_1"],
                "var_2": trace["var_2"],
                "var_3": trace["var_4"],
                "var_4": trace["var_3"],
            }
        )
        assert (
            swapped["right_equality"] == trace["right_equality"]
            and swapped["result_equality"] == trace["result_equality"]
        )


class TestHierarchicalEqualityOutputTokens:
    """``output_tokens`` invariants: bool → digit forms, prefix match."""

    pytestmark = pytest.mark.property

    def test_output_token_keys(self) -> None:
        ot = CAUSAL_MODEL.output_tokens
        assert ot is not None and set(ot.keys()) == {
            "left_equality",
            "right_equality",
            "result_equality",
        }

    def test_forms_are_true_first_digit_forms(self) -> None:
        """``True`` → ``[" 1", "1"]`` then ``False`` → ``[" 0", "0"]`` — order is load-bearing."""
        ot = CAUSAL_MODEL.output_tokens
        assert ot is not None
        for var_map in ot.values():
            assert list(var_map.items()) == [
                (True, [" 1", "1"]),
                (False, [" 0", "0"]),
            ]

    def test_string_mode_is_prefix(self) -> None:
        """The model emits a bare digit possibly followed by text → ``prefix``
        string mode, graded on ``result_equality``'s digit (the spec's
        ``answer_variable``); the derived per-variable view says the same."""
        spec = CAUSAL_MODEL.scoring
        assert spec is not None
        assert (
            spec.string_mode == "prefix" and spec.answer_variable == "result_equality"
        )
        mm = CAUSAL_MODEL.match_modes
        assert mm is not None and all(mode == "prefix" for mode in mm.values())


class TestHierarchicalEqualityCounterfactualGeneratorProperty:
    """Invariants for ``sample_balanced_input`` / ``random_counterfactual`` / ``generate_dataset``."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_sample_balanced_input_is_well_formed(self, seed: int) -> None:
        random.seed(seed)
        trace = sample_balanced_input()
        assert (
            trace["template"] in TEMPLATES
            and all(trace[f"var_{i}"] in LETTERS for i in range(1, 5))
            and trace["raw_output"] in {"0", "1"}
        )

    def test_sample_balanced_input_hits_every_pattern(self) -> None:
        """Empirical pattern coverage: each of the 4 patterns appears at least once."""
        seen: set[tuple[bool, bool]] = set()
        random.seed(0)
        for _ in range(200):
            trace = sample_balanced_input()
            seen.add(
                (
                    trace["var_1"] == trace["var_2"],
                    trace["var_3"] == trace["var_4"],
                )
            )
        assert seen == {(True, True), (False, False), (False, True), (True, False)}

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_counterfactual_yields_single_counterfactual_input(
        self, seed: int
    ) -> None:
        random.seed(seed)
        cf = random_counterfactual()
        assert len(cf["counterfactual_inputs"]) == 1

    def test_random_counterfactual_is_registered(self) -> None:
        """The registry contract maps the name to the function object."""
        assert (
            COUNTERFACTUAL_GENERATORS["random_counterfactual"] is random_counterfactual
        )

    @pytest.mark.parametrize("n", [1, 4, 16])
    def test_generate_dataset_length(self, n: int) -> None:
        examples = generate_dataset(model=None, n=n, seed=42)
        assert len(examples) == n

    def test_generate_dataset_is_deterministic_under_fixed_seed(self) -> None:
        """Two calls with the same seed produce byte-equal output."""
        first = generate_dataset(None, 4, seed=7)
        second = generate_dataset(None, 4, seed=7)
        first_dicts = [
            (e["input"].to_dict(), e["counterfactual_inputs"][0].to_dict())
            for e in first
        ]
        second_dicts = [
            (e["input"].to_dict(), e["counterfactual_inputs"][0].to_dict())
            for e in second
        ]
        assert first_dicts == second_dicts

    def test_generate_dataset_distinct_seeds_diverge(self) -> None:
        """Different seeds must produce at least one differing element."""
        a = generate_dataset(None, 4, seed=1)
        b = generate_dataset(None, 4, seed=2)
        any_diff = any(
            ai["input"].to_dict() != bi["input"].to_dict()
            or ai["counterfactual_inputs"][0].to_dict()
            != bi["counterfactual_inputs"][0].to_dict()
            for ai, bi in zip(a, b)
        )
        assert any_diff

    def test_generate_dataset_restores_global_rng_state(self) -> None:
        """``generate_dataset`` snapshots/restores the global ``random`` state."""
        random.seed(123)
        before = random.getstate()
        generate_dataset(None, 4, seed=42)
        after = random.getstate()
        assert before == after

    def test_generate_dataset_is_model_arg_agnostic(self) -> None:
        """The ``model`` arg is a loader-convention placeholder — passing the
        singleton vs. ``None`` must yield byte-identical output."""
        none_out = generate_dataset(None, 4, seed=5)
        model_out = generate_dataset(CAUSAL_MODEL, 4, seed=5)
        none_dicts = [
            (e["input"].to_dict(), e["counterfactual_inputs"][0].to_dict())
            for e in none_out
        ]
        model_dicts = [
            (e["input"].to_dict(), e["counterfactual_inputs"][0].to_dict())
            for e in model_out
        ]
        assert none_dicts == model_dicts

    @pytest.mark.parametrize(
        "generator",
        [sample_balanced_input, random_counterfactual],
        ids=["sample_balanced_input", "random_counterfactual"],
    )
    def test_direct_generator_is_deterministic(
        self, generator: Callable[[], Any]
    ) -> None:
        """Called directly (not through ``generate_dataset``), each generator
        must reproduce byte-equal output under the same seed."""
        random.seed(123)
        first = generator()
        random.seed(123)
        second = generator()
        # ``sample_balanced_input`` returns a CausalTrace; ``random_counterfactual``
        # returns a CounterfactualExample (TypedDict). Both expose ``to_dict()`` /
        # subscript on ``"input"``.
        if isinstance(first, dict):
            assert (
                first["input"].to_dict() == second["input"].to_dict()
                and first["counterfactual_inputs"][0].to_dict()
                == second["counterfactual_inputs"][0].to_dict()
            )
        else:
            assert first.to_dict() == second.to_dict()


class TestHierarchicalEqualityTemplatesProperty:
    """Invariants for ``_sample_pattern_values`` + ``fill_template`` (shipped mode)."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_pattern_AABB(self, seed: int) -> None:
        random.seed(seed)
        v1, v2, v3, v4 = _sample_pattern_values("AABB", random)
        assert v1 == v2 and v3 == v4

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_pattern_ABCD(self, seed: int) -> None:
        random.seed(seed)
        v1, v2, v3, v4 = _sample_pattern_values("ABCD", random)
        assert len({v1, v2, v3, v4}) == 4

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_pattern_ABCC(self, seed: int) -> None:
        random.seed(seed)
        v1, v2, v3, v4 = _sample_pattern_values("ABCC", random)
        assert v1 != v2 and v3 == v4

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_pattern_AABC(self, seed: int) -> None:
        random.seed(seed)
        v1, v2, v3, v4 = _sample_pattern_values("AABC", random)
        assert v1 == v2 and v3 != v4

    def test_pattern_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            _sample_pattern_values("garbage", random)

    def test_fill_template_minimal_function_suffix(self) -> None:
        """In the shipped ``minimal_function`` mode the prompt ends with the query."""
        random.seed(0)
        out = fill_template("minimal_function", "A", "B", "C", "D", seed=0)
        assert out.endswith("f(A,B,C,D)=")

    def test_fill_template_minimal_function_icl_line_count(self) -> None:
        """The prompt contains exactly ``NUM_ICL_EXAMPLES`` ICL lines."""
        random.seed(0)
        out = fill_template("minimal_function", "A", "B", "C", "D", seed=0)
        # Split on newline: NUM_ICL_EXAMPLES ICL lines + 1 query line.
        assert out.count("\n") == NUM_ICL_EXAMPLES

    def test_get_func_name_extracts_from_def_signature(self) -> None:
        assert get_func_name("def foo(a, b, c, d): ...") == "foo"


class TestHierarchicalEqualityTokenPositionsProperty:
    """Tokenizer-coupled invariants for ``create_token_positions``.

    Uses a session-scoped tiny Llama stub (see ``conftest.py``). Drops
    ``max_examples`` to 20 to keep this class sub-2s wallclock.
    """

    pytestmark = pytest.mark.property

    def test_returns_expected_keys(self, he_tiny_pipeline: LMPipeline) -> None:
        positions = create_token_positions(he_tiny_pipeline)
        assert set(positions.keys()) == {"last", "var_1", "var_2", "var_3", "var_4"}

    def test_each_value_is_token_position_with_matching_id(
        self, he_tiny_pipeline: LMPipeline
    ) -> None:
        positions = create_token_positions(he_tiny_pipeline)
        assert all(
            isinstance(p, TokenPosition) and p.id == k for k, p in positions.items()
        )

    def test_prompt_mode_none_defaults_to_shipped_mode(
        self, he_tiny_pipeline: LMPipeline
    ) -> None:
        """``prompt_mode=None`` must equal an explicit ``PROMPT_MODE`` call."""
        random.seed(0)
        trace = sample_balanced_input()
        default_pos = create_token_positions(he_tiny_pipeline, prompt_mode=None)  # type: ignore[arg-type]
        explicit_pos = create_token_positions(he_tiny_pipeline, prompt_mode=PROMPT_MODE)
        for key in default_pos:
            assert default_pos[key].index(trace) == explicit_pos[key].index(trace)

    def test_template_arg_is_unused(self, he_tiny_pipeline: LMPipeline) -> None:
        """The ``template`` arg is currently a no-op — pin the quirk."""
        random.seed(0)
        trace = sample_balanced_input()
        none_pos = create_token_positions(he_tiny_pipeline, template=None)  # type: ignore[arg-type]
        some_pos = create_token_positions(he_tiny_pipeline, template="ignored")
        for key in none_pos:
            assert none_pos[key].index(trace) == some_pos[key].index(trace)

    @settings(
        deadline=None,
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(seed=st.integers(min_value=0, max_value=10_000))
    def test_var_position_indices_are_well_formed(
        self, he_tiny_pipeline: LMPipeline, seed: int
    ) -> None:
        """Each var position resolves to a non-empty list of valid token indices."""
        random.seed(seed)
        trace = sample_balanced_input()
        positions = create_token_positions(he_tiny_pipeline)
        n_tokens = len(he_tiny_pipeline.tokenizer.encode(trace["raw_input"]))
        ok = True
        for key in ("var_1", "var_2", "var_3", "var_4"):
            indices = cast(list[int], positions[key].index(trace))
            ok = ok and len(indices) > 0 and all(0 <= i < n_tokens for i in indices)
        assert ok

    def test_last_position_matches_last_token_index(
        self, he_tiny_pipeline: LMPipeline
    ) -> None:
        """``last`` resolves to the final token index of the raw input."""
        from causalab.tasks.token_positions import get_last_token_index

        random.seed(0)
        trace = sample_balanced_input()
        positions = create_token_positions(he_tiny_pipeline)
        assert positions["last"].index(trace) == get_last_token_index(
            trace, he_tiny_pipeline
        )

    def test_minimal_function_mode_decodes_letters(
        self, he_tiny_pipeline: LMPipeline
    ) -> None:
        """``minimal_function`` regex maps each var position to its letter token."""
        random.seed(0)
        trace = sample_balanced_input()  # shipped mode is minimal_function
        positions = create_token_positions(
            he_tiny_pipeline, prompt_mode="minimal_function"
        )
        tokens = he_tiny_pipeline.tokenizer.encode(trace["raw_input"])
        decoded_letters = [
            he_tiny_pipeline.tokenizer.decode(
                [tokens[cast(list[int], positions[f"var_{i}"].index(trace))[0]]]
            ).strip()
            for i in range(1, 5)
        ]
        expected = [trace[f"var_{i}"] for i in range(1, 5)]
        assert decoded_letters == expected


class TestHierarchicalEqualityConstantsProperty:
    """Pool-size + naming invariants the rest of the task relies on."""

    pytestmark = pytest.mark.property

    def test_letters_nonempty_and_single_chars(self) -> None:
        assert len(LETTERS) > 0 and all(
            isinstance(c, str) and len(c) == 1 for c in LETTERS
        )

    def test_letters_has_at_least_four(self) -> None:
        """``ABCD`` pattern uses ``random.sample(LETTERS, 4)``."""
        assert len(LETTERS) >= 4

    def test_patterns_match_registered_order(self) -> None:
        assert PATTERNS == ["AABB", "ABCD", "ABCC", "AABC"]

    def test_templates_nonempty_under_shipped_mode(self) -> None:
        assert len(TEMPLATES) > 0
        assert PROMPT_MODE != "minimal_function" or TEMPLATES == ["minimal_function"]

    def test_num_icl_examples_divisible_by_pattern_count(self) -> None:
        """``_generate_icl_lines`` divides ``n // len(PATTERNS)`` per pattern;
        non-divisibility silently drops the remainder."""
        assert NUM_ICL_EXAMPLES % len(PATTERNS) == 0

    def test_task_name_matches_causal_model_id(self) -> None:
        assert TASK_NAME == "hierarchical_equality" == CAUSAL_MODEL.id

    def test_get_task_returns_singleton_causal_model(self) -> None:
        """The shared task helper resolves the singleton to ``CAUSAL_MODEL``."""
        assert get_task("hierarchical_equality").causal_model is CAUSAL_MODEL
