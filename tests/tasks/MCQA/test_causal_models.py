"""Property-tier invariants for the MCQA causal model.

The MCQA task casts multiple-choice question answering as a causal DAG
over ``(template, object, color, choices[0..1], symbols[0..1]) → answer_position
→ answer → raw_output``. The mechanisms are pure-symbolic: there's no
LM, no analysis, no runner — so these tests run sub-second and
hypothesis-driven sweeps catch contract drift the runner-level goldens
in ``tests/end_to_end/goldens/mcqa.json`` would only flag indirectly.

Numerical pinning of specific symbolic outputs lives in the sibling
``pinned_samples.json`` and is enforced by
``test_MCQA_numerical.py`` (sample-by-sample equality at fixed seeds).
This file covers the *invariants* — things that should hold for every
input, every seed.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke comes from ``baseline/mcqa.yaml``
via ``tests/end_to_end/test_smoke.py``. This file satisfies
property-direct; the sibling numerical-tier test satisfies
numerical-direct.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.tasks.MCQA.causal_models import (
    ALPHABET,
    COLORS,
    NUM_CHOICES,
    OBJECTS,
    TEMPLATES,
    positional_causal_model,
)
from causalab.tasks.MCQA.counterfactuals import (
    different_symbol,
    random_counterfactual,
    same_symbol_different_position,
    sample_answerable_question,
)


# Hypothesis settings used across this file:
#   - deadline=None: causal-model evaluation is pure-symbolic and not
#     timing-sensitive, but hypothesis' default 200ms deadline can flake
#     on cold caches.
#   - function_scoped_fixture: we don't use them, but suppressing the
#     health check is cheaper than asserting it.
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestMcqaCausalModelStructureProperty:
    """Static structural invariants of the MCQA causal DAG."""

    pytestmark = pytest.mark.property

    def test_required_variables_present(self) -> None:
        """The DAG names every variable the loader contract relies on."""
        expected = {
            "template",
            "object",
            "color",
            "raw_input",
            "answer_position",
            "answer",
            "raw_output",
        }
        for i in range(NUM_CHOICES):
            expected.update({f"symbols[{i}]", f"choices[{i}]"})

        assert expected <= set(positional_causal_model.variables), (
            f"missing variables: {expected - set(positional_causal_model.variables)}"
        )

    def test_raw_input_parents_match_template_placeholders(self) -> None:
        """``raw_input``'s parents must cover every ``{...}`` in the template."""
        expected = (
            ["template", "object", "color"]
            + [f"symbols[{i}]" for i in range(NUM_CHOICES)]
            + [f"choices[{i}]" for i in range(NUM_CHOICES)]
        )
        assert set(positional_causal_model.parents["raw_input"]) == set(expected)

    def test_answer_position_depends_only_on_color_and_choices(self) -> None:
        """The mechanism's parents define what an intervention upstream can affect."""
        assert set(positional_causal_model.parents["answer_position"]) == {
            "color",
            *[f"choices[{i}]" for i in range(NUM_CHOICES)],
        }

    def test_answer_depends_on_position_and_symbols(self) -> None:
        assert set(positional_causal_model.parents["answer"]) == {
            "answer_position",
            *[f"symbols[{i}]" for i in range(NUM_CHOICES)],
        }


class TestSampleAnswerableQuestionProperty:
    """``sample_answerable_question`` must produce well-formed questions.

    The wrapper exists because ``CausalModel.sample_input()`` samples
    inputs independently, which can produce a colour that doesn't appear
    in the choices — making the question unanswerable and crashing
    ``_get_answer_position``. The wrapper's invariant: every output has
    the colour somewhere in the choices.
    """

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_color_is_always_in_choices(self, seed: int) -> None:
        random.seed(seed)
        trace = sample_answerable_question()
        choices = [trace[f"choices[{i}]"] for i in range(NUM_CHOICES)]
        assert trace["color"] in choices, (
            f"color {trace['color']!r} not in choices {choices!r}"
        )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_symbols_are_unique_across_positions(self, seed: int) -> None:
        random.seed(seed)
        trace = sample_answerable_question()
        symbols = [trace[f"symbols[{i}]"] for i in range(NUM_CHOICES)]
        assert len(set(symbols)) == NUM_CHOICES, f"duplicate symbols: {symbols}"

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_answer_matches_symbol_at_answer_position(self, seed: int) -> None:
        """The fundamental MCQA contract: answer = symbols[answer_position]."""
        random.seed(seed)
        trace = sample_answerable_question()
        assert trace["answer"] == trace[f"symbols[{trace['answer_position']}]"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_input_substitutes_every_placeholder(self, seed: int) -> None:
        random.seed(seed)
        trace = sample_answerable_question()
        raw = trace["raw_input"]
        for placeholder in ("{object}", "{color}"):
            assert placeholder not in raw, f"unsubstituted placeholder: {placeholder}"
        for i in range(NUM_CHOICES):
            for placeholder in (f"{{symbols[{i}]}}", f"{{choices[{i}]}}"):
                assert placeholder not in raw, (
                    f"unsubstituted placeholder: {placeholder}"
                )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_is_space_prefixed_answer(self, seed: int) -> None:
        """``raw_output`` mechanism is `lambda t: " " + t["answer"]`."""
        random.seed(seed)
        trace = sample_answerable_question()
        assert trace["raw_output"] == " " + trace["answer"]


class TestCounterfactualGeneratorProperty:
    """Invariants the three MCQA counterfactual generators must preserve."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_same_symbol_different_position_swaps_position(self, seed: int) -> None:
        """The counterfactual changes ``answer_position`` while keeping the
        symbol/choice multiset the same (just permuted)."""
        random.seed(seed)
        cf = same_symbol_different_position()
        base = cf["input"]
        ctf = cf["counterfactual_inputs"][0]

        assert base["answer_position"] != ctf["answer_position"], (
            "same_symbol_different_position must change the position"
        )
        base_symbols = sorted(base[f"symbols[{i}]"] for i in range(NUM_CHOICES))
        ctf_symbols = sorted(ctf[f"symbols[{i}]"] for i in range(NUM_CHOICES))
        assert base_symbols == ctf_symbols, "symbol multiset must be preserved"
        base_choices = sorted(base[f"choices[{i}]"] for i in range(NUM_CHOICES))
        ctf_choices = sorted(ctf[f"choices[{i}]"] for i in range(NUM_CHOICES))
        assert base_choices == ctf_choices, "choice multiset must be preserved"

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_different_symbol_changes_all_symbols(self, seed: int) -> None:
        """The counterfactual replaces every symbol with a fresh one."""
        random.seed(seed)
        cf = different_symbol()
        base = cf["input"]
        ctf = cf["counterfactual_inputs"][0]

        base_syms = {base[f"symbols[{i}]"] for i in range(NUM_CHOICES)}
        ctf_syms = {ctf[f"symbols[{i}]"] for i in range(NUM_CHOICES)}
        assert base_syms.isdisjoint(ctf_syms), (
            f"different_symbol left overlapping symbols: "
            f"base={base_syms} ctf={ctf_syms}"
        )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_counterfactual_yields_two_well_formed_inputs(
        self, seed: int
    ) -> None:
        """Independence: ``random_counterfactual`` is just two answerable
        samples paired together."""
        random.seed(seed)
        cf = random_counterfactual()
        for trace in (cf["input"], cf["counterfactual_inputs"][0]):
            choices = [trace[f"choices[{i}]"] for i in range(NUM_CHOICES)]
            assert trace["color"] in choices, "input is not answerable"
            assert trace["answer"] == trace[f"symbols[{trace['answer_position']}]"], (
                "answer/position invariant violated"
            )

    @pytest.mark.parametrize(
        "generator",
        [random_counterfactual, same_symbol_different_position, different_symbol],
        ids=["random", "same_symbol", "different_symbol"],
    )
    def test_generator_is_deterministic_under_fixed_seed(self, generator) -> None:
        """Two invocations after the same seed produce byte-equal output."""
        random.seed(123)
        first = generator()
        random.seed(123)
        second = generator()
        assert first["input"].to_dict() == second["input"].to_dict()
        assert (
            first["counterfactual_inputs"][0].to_dict()
            == second["counterfactual_inputs"][0].to_dict()
        )


class TestMcqaConstantsProperty:
    """Pool-size invariants the rest of the task code relies on."""

    pytestmark = pytest.mark.property

    def test_enough_colors_for_choices(self) -> None:
        assert len(COLORS) >= NUM_CHOICES + 1, (
            "sample_answerable_question needs at least NUM_CHOICES + 1 colors "
            "to pick a correct one plus non-overlapping distractors"
        )

    def test_enough_symbols_for_choices(self) -> None:
        assert len(ALPHABET) >= NUM_CHOICES, (
            "ALPHABET must have at least NUM_CHOICES letters"
        )

    def test_objects_nonempty(self) -> None:
        assert len(OBJECTS) > 0

    def test_templates_nonempty(self) -> None:
        assert len(TEMPLATES) > 0
        for tpl in TEMPLATES:
            for required in ("{object}", "{color}"):
                assert required in tpl
            for i in range(NUM_CHOICES):
                assert f"{{symbols[{i}]}}" in tpl
                assert f"{{choices[{i}]}}" in tpl


class TestMcqaInterventionApiUnit:
    """Intervention-API behaviour on the MCQA causal model.

    The two non-redundant checks re-homed from the deleted integration test
    ``test_01_mcqa_task_definition.py`` (notebook 01) — both are LM-free and
    exercise the ``CausalTrace`` / ``CausalModel`` intervention API directly.
    """

    pytestmark = pytest.mark.unit

    def test_copy_intervene_sets_variable(self) -> None:
        """``CausalTrace.copy().intervene(var, value)`` overwrites only the
        named variable and leaves the original trace untouched."""
        random.seed(0)
        trace = sample_answerable_question()
        original_position = trace["answer_position"]
        new_position = int(not original_position)
        intervened = trace.copy().intervene("answer_position", new_position)
        assert intervened["answer_position"] == new_position
        assert intervened["answer_position"] != original_position
        # copy() is a true copy — the source trace is unchanged.
        assert trace["answer_position"] == original_position

    def test_run_interchange_copies_source_value(self) -> None:
        """``CausalModel.run_interchange(base, {var: source})`` adopts the
        source trace's value for the swapped variable."""
        random.seed(0)
        base = sample_answerable_question()
        source = sample_answerable_question()
        # Resample the source until its answer_position differs from base, so
        # the swap is observable (answer_position is binary → near-certain).
        for _ in range(50):
            if source["answer_position"] != base["answer_position"]:
                break
            source = sample_answerable_question()
        else:
            pytest.skip("could not sample a differing answer_position in 50 tries")
        intervened = positional_causal_model.run_interchange(
            base, {"answer_position": source}
        )
        assert intervened["answer_position"] == source["answer_position"]
