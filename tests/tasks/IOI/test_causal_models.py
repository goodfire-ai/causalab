"""Property-tier invariants for the IOI (Indirect Object Identification) task.

IOI is a singleton, pure-symbolic task: a [`CausalModel`][causalab.causal.model.CausalModel] over
JSON-loaded vocab pools (``names``, ``objects``, ``places``,
``templates``) whose target variable ``IO`` is the indirect-object name
the downstream interchange interventions try to recover. Every
counterfactual flips ``name_C`` to the other introduced name; the
loader contract exposes the singleton through
``CAUSAL_MODEL``/``TARGET_VARIABLE``/``TEMPLATE``.

Numerical pinning of specific symbolic outputs lives in the sibling
``pinned_samples.json`` and is enforced by
``test_IOI_numerical.py`` (sample-by-sample equality at fixed seeds).
This file covers the *invariants* — things that should hold for every
input, every seed.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke comes from ``baseline/ioi.yaml``
via ``tests/end_to_end/test_smoke.py``. This file satisfies
property-direct; the sibling numerical-tier test satisfies
numerical-direct.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.tasks.IOI.causal_models import (
    ALL_TEMPLATES,
    CANONICAL_TEMPLATE,
    NAMES,
    OBJECTS,
    PLACES,
    TEMPLATES,
    positional_causal_model,
)
from causalab.tasks.IOI.counterfactuals import (
    flip_name_C,
    generate_abc_dataset,
    generate_dataset,
    random_counterfactual,
)
from causalab.tasks.IOI.token_positions import create_token_positions
from causalab.tasks.token_positions import TokenPosition


# Hypothesis settings — same defaults as the MCQA pilot.
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Token-position class wraps a session-scoped gpt2 pipeline — tighter
# budget so the wallclock stays under ~2s for the whole class.
_HYPOTHESIS_SETTINGS_TOKENIZED = settings(
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestIOICausalModelStructureProperty:
    """Static structural invariants of the IOI causal DAG."""

    pytestmark = pytest.mark.property

    def test_required_variables_present(self) -> None:
        """The DAG names every variable downstream code relies on."""
        expected = {
            "template",
            "name_A",
            "name_B",
            "name_C",
            "place",
            "object",
            "raw_input",
            "IO",
            "raw_output",
        }
        assert expected <= set(positional_causal_model.variables)

    def test_raw_input_parents_match_template_placeholders(self) -> None:
        """``raw_input``'s parents must cover every ``{...}`` in the template."""
        assert positional_causal_model.parents["raw_input"] == [
            "template",
            "name_A",
            "name_B",
            "name_C",
            "place",
            "object",
        ]

    def test_io_depends_only_on_three_names(self) -> None:
        """``IO`` is computed purely from the three name variables."""
        assert positional_causal_model.parents["IO"] == [
            "name_A",
            "name_B",
            "name_C",
        ]

    def test_raw_output_depends_only_on_io(self) -> None:
        """``raw_output`` is a pure function of ``IO`` (`` `` + IO)."""
        assert positional_causal_model.parents["raw_output"] == ["IO"]


class TestIOISampleInputProperty:
    """Invariants ``positional_causal_model.sample_input`` must enforce.

    Drives through ``sample_input`` (which honours ``_input_filter``)
    rather than constructing traces directly — the filter is what makes
    these invariants hold.
    """

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_name_a_and_b_are_distinct(self, seed: int) -> None:
        """Distractor vs. IO distinction — the IOI premise."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        assert trace["name_A"] != trace["name_B"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_name_c_is_one_of_a_or_b(self, seed: int) -> None:
        """``_input_filter`` rejects any trace where the subject isn't introduced."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        assert trace["name_C"] in (trace["name_A"], trace["name_B"])

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_io_is_the_other_introduced_name(self, seed: int) -> None:
        """The IOI contract: IO == the introduced name that isn't the subject."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        expected = (
            trace["name_A"] if trace["name_C"] == trace["name_B"] else trace["name_B"]
        )
        assert trace["IO"] == expected

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_input_substitutes_every_placeholder(self, seed: int) -> None:
        """No surviving ``{...}`` placeholders in the filled template."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        for placeholder in ("{name_A}", "{name_B}", "{name_C}", "{place}", "{object}"):
            assert placeholder not in trace["raw_input"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_is_space_prefixed_io(self, seed: int) -> None:
        """``raw_output`` mechanism is `lambda t: " " + t["IO"]`."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        assert trace["raw_output"] == " " + trace["IO"]


class TestIOICounterfactualGeneratorProperty:
    """Per-generator invariants and the loader entry point ``generate_dataset``."""

    pytestmark = pytest.mark.property

    # --- flip_name_C ------------------------------------------------------

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_flip_name_c_changes_name_c(self, seed: int) -> None:
        """The intervention flips ``name_C`` — its single mutation point."""
        random.seed(seed)
        cf = flip_name_C()
        assert cf["counterfactual_inputs"][0]["name_C"] != cf["input"]["name_C"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_flip_name_c_preserves_other_inputs(self, seed: int) -> None:
        """Single-variable intervention — A/B/place/object/template unchanged."""
        random.seed(seed)
        cf = flip_name_C()
        base, ctf = cf["input"], cf["counterfactual_inputs"][0]
        for key in ("name_A", "name_B", "place", "object", "template"):
            assert ctf[key] == base[key], f"{key} drifted"

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_flip_name_c_keeps_filter_closure(self, seed: int) -> None:
        """``name_C`` after the flip must still be in ``{name_A, name_B}``."""
        random.seed(seed)
        cf = flip_name_C()
        ctf = cf["counterfactual_inputs"][0]
        assert ctf["name_C"] in (ctf["name_A"], ctf["name_B"])

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_flip_name_c_swaps_io(self, seed: int) -> None:
        """Original subject (base.name_C) becomes the new IO."""
        random.seed(seed)
        cf = flip_name_C()
        assert cf["counterfactual_inputs"][0]["IO"] == cf["input"]["name_C"]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_flip_name_c_raw_output_tracks_io(self, seed: int) -> None:
        """Counterfactual ``raw_output`` re-derives from the new IO."""
        random.seed(seed)
        cf = flip_name_C()
        ctf = cf["counterfactual_inputs"][0]
        assert ctf["raw_output"] == " " + ctf["IO"]

    # --- random_counterfactual --------------------------------------------

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_counterfactual_both_traces_well_formed(self, seed: int) -> None:
        """Both ``input`` and ``counterfactual_inputs[0]`` honour the filter."""
        random.seed(seed)
        cf = random_counterfactual()
        for trace in (cf["input"], cf["counterfactual_inputs"][0]):
            assert trace["name_A"] != trace["name_B"] and trace["name_C"] in (
                trace["name_A"],
                trace["name_B"],
            )

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_counterfactual_raw_output_tracks_io(self, seed: int) -> None:
        """Both traces' ``raw_output`` equals `` `` + their respective IO."""
        random.seed(seed)
        cf = random_counterfactual()
        for trace in (cf["input"], cf["counterfactual_inputs"][0]):
            assert trace["raw_output"] == " " + trace["IO"]

    # --- generate_dataset -------------------------------------------------

    @pytest.mark.parametrize("n", [1, 4, 16])
    def test_generate_dataset_returns_n_examples(self, n: int) -> None:
        """The loader entry point produces exactly ``n`` examples."""
        assert len(generate_dataset(model=None, n=n, seed=42)) == n

    def test_generate_dataset_ignores_model_argument(self) -> None:
        """Docstring says ``model`` is accepted but unused — pin the quirk."""
        seed = 1234
        none_run = [
            ex["input"].to_dict() for ex in generate_dataset(None, 3, seed=seed)
        ]
        model_run = [
            ex["input"].to_dict()
            for ex in generate_dataset(positional_causal_model, 3, seed=seed)
        ]
        assert none_run == model_run

    def test_generate_dataset_is_seed_reproducible(self) -> None:
        """Two calls with the same seed produce equal output."""
        first = [ex["input"].to_dict() for ex in generate_dataset(None, 3, seed=7)]
        second = [ex["input"].to_dict() for ex in generate_dataset(None, 3, seed=7)]
        assert first == second

    def test_generate_dataset_distinct_seeds_distinct_output(self) -> None:
        """Different seeds produce different output (statistical, not strict)."""
        a = [ex["input"].to_dict() for ex in generate_dataset(None, 5, seed=1)]
        b = [ex["input"].to_dict() for ex in generate_dataset(None, 5, seed=2)]
        assert a != b

    def test_generate_dataset_restores_global_random_state(self) -> None:
        """``random.getstate()`` is restored — downstream analyses depend on it."""
        random.seed(0)
        before = random.getstate()
        generate_dataset(None, 5, seed=42)
        assert random.getstate() == before

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_generate_dataset_elements_satisfy_flip_invariants(self, seed: int) -> None:
        """Every dataset element is a ``flip_name_C`` output — same invariants apply."""
        examples = generate_dataset(None, 3, seed=seed)
        for ex in examples:
            base, ctf = ex["input"], ex["counterfactual_inputs"][0]
            assert ctf["name_C"] != base["name_C"] and ctf["IO"] == base["name_C"]

    # --- generate_abc_dataset (path-patching §3.1 corruption) -------------

    @pytest.mark.parametrize("n", [1, 4, 16])
    def test_abc_dataset_returns_n_examples(self, n: int) -> None:
        assert len(generate_abc_dataset(model=None, n=n, seed=42)) == n

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_abc_source_has_three_distinct_names(self, seed: int) -> None:
        """The ABC source replaces all three name slots with distinct randoms,
        destroying the repeated-name IOI structure."""
        for ex in generate_abc_dataset(None, 3, seed=seed):
            s = ex["counterfactual_inputs"][0]
            assert len({s["name_A"], s["name_B"], s["name_C"]}) == 3

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_abc_base_is_well_formed_and_template_preserved(self, seed: int) -> None:
        """The base stays a valid IOI example; the source keeps template/place/object."""
        for ex in generate_abc_dataset(None, 3, seed=seed):
            base, s = ex["input"], ex["counterfactual_inputs"][0]
            assert base["name_A"] != base["name_B"] and base["name_C"] in (
                base["name_A"],
                base["name_B"],
            )
            for key in ("place", "object", "template"):
                assert s[key] == base[key], f"{key} drifted in ABC source"

    def test_abc_dataset_is_seed_reproducible(self) -> None:
        a = [e["input"].to_dict() for e in generate_abc_dataset(None, 4, seed=7)]
        b = [e["input"].to_dict() for e in generate_abc_dataset(None, 4, seed=7)]
        assert a == b

    def test_abc_dataset_restores_global_random_state(self) -> None:
        random.seed(0)
        before = random.getstate()
        generate_abc_dataset(None, 5, seed=42)
        assert random.getstate() == before

    # --- shared determinism ----------------------------------------------

    @pytest.mark.parametrize(
        "generator",
        [flip_name_C, random_counterfactual],
        ids=["flip_name_C", "random_counterfactual"],
    )
    def test_generator_is_deterministic_under_fixed_seed(self, generator) -> None:
        """Re-seeding ``random`` reproduces byte-equal output. Drives the
        generators directly — *not* through ``generate_dataset`` (which
        save/restores ``random.getstate()`` and would mask drift)."""
        random.seed(123)
        first = generator()
        random.seed(123)
        second = generator()
        assert first["input"].to_dict() == second["input"].to_dict()


class TestIOITokenPositionsProperty:
    """Tokenizer-coupled invariants for ``create_token_positions``.

    Driven through a session-scoped ``gpt2`` [`LMPipeline`][causalab.tasks.token_positions.LMPipeline] (see
    ``conftest.py``) — token positions can only be meaningfully
    validated against a real tokenizer.
    """

    pytestmark = pytest.mark.property

    def test_keys_are_the_four_documented_positions(self, gpt2_pipeline) -> None:
        """The factory returns exactly ``name_A``, ``name_B``, ``name_C``, ``last_token``."""
        positions = create_token_positions(gpt2_pipeline)
        assert set(positions.keys()) == {"name_A", "name_B", "name_C", "last_token"}

    def test_every_value_is_a_token_position_with_matching_id(
        self, gpt2_pipeline
    ) -> None:
        """Each value is a [`TokenPosition`][causalab.tasks.token_positions.TokenPosition] whose ``id`` matches its key."""
        positions = create_token_positions(gpt2_pipeline)
        for name, pos in positions.items():
            assert isinstance(pos, TokenPosition) and pos.id == name

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS_TOKENIZED
    def test_indices_are_valid_offsets_into_tokenized_prompt(
        self, gpt2_pipeline, seed: int
    ) -> None:
        """Every returned index is a valid offset into the encoded prompt."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        positions = create_token_positions(gpt2_pipeline)
        n_tokens = len(gpt2_pipeline.load([trace])["input_ids"][0])
        for pos in positions.values():
            indices = pos.index(trace)
            assert indices and all(0 <= i < n_tokens for i in indices)

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS_TOKENIZED
    def test_name_positions_decode_to_the_name(self, gpt2_pipeline, seed: int) -> None:
        """For ``name_A/B/C``, the decoded token contains the corresponding name."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        positions = create_token_positions(gpt2_pipeline)
        ids = gpt2_pipeline.load([trace])["input_ids"][0]
        for key in ("name_A", "name_B", "name_C"):
            idx = positions[key].index(trace)[0]
            decoded = gpt2_pipeline.tokenizer.decode([ids[idx]]).strip()
            assert decoded in trace[key] or trace[key].startswith(decoded)

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS_TOKENIZED
    def test_last_token_index_is_final_offset(self, gpt2_pipeline, seed: int) -> None:
        """``last_token.index(trace)`` is the final offset of the encoded prompt."""
        random.seed(seed)
        trace = positional_causal_model.sample_input()
        positions = create_token_positions(gpt2_pipeline)
        n_tokens = len(gpt2_pipeline.load([trace])["input_ids"][0])
        assert positions["last_token"].index(trace) == [n_tokens - 1]

    def test_template_none_equals_canonical_template(self, gpt2_pipeline) -> None:
        """``template=None`` and ``template=CANONICAL_TEMPLATE`` agree on indices."""
        random.seed(0)
        trace = positional_causal_model.sample_input()
        pos_default = create_token_positions(gpt2_pipeline, template=None)
        pos_canonical = create_token_positions(
            gpt2_pipeline, template=CANONICAL_TEMPLATE
        )
        for key in pos_default:
            assert pos_default[key].index(trace) == pos_canonical[key].index(trace)

    def test_custom_template_still_decodes_names(self, gpt2_pipeline) -> None:
        """Pass a punctuation-variant of the canonical template; name lookups
        still hit the right tokens."""
        alt_template = "After {name_A} and {name_B} went to the {place}; {name_C} gave a {object} to"
        positions = create_token_positions(gpt2_pipeline, template=alt_template)
        random.seed(0)
        trace = positional_causal_model.sample_input()
        trace_alt = dict(trace.to_dict())
        trace_alt["template"] = alt_template
        trace_alt["raw_input"] = alt_template.format(
            **{k: trace[k] for k in ("name_A", "name_B", "name_C", "place", "object")}
        )
        for key in ("name_A", "name_B", "name_C"):
            assert positions[key].index(trace_alt), f"{key} not found in alt template"

    def test_indices_are_deterministic_under_fixed_seed(self, gpt2_pipeline) -> None:
        """Same seed → same trace → byte-equal index results."""
        random.seed(42)
        trace_a = positional_causal_model.sample_input()
        random.seed(42)
        trace_b = positional_causal_model.sample_input()
        positions = create_token_positions(gpt2_pipeline)
        for key in positions:
            assert positions[key].index(trace_a) == positions[key].index(trace_b)


class TestIOIConstantsProperty:
    """Pool-size and template-placeholder invariants the task code relies on."""

    pytestmark = pytest.mark.property

    def test_enough_names_for_filter(self) -> None:
        """Filter needs ≥2 distinct names + a viable ``name_C``."""
        assert len(NAMES) >= 3

    def test_pools_nonempty(self) -> None:
        """``OBJECTS``, ``PLACES``, ``ALL_TEMPLATES`` all have entries."""
        assert len(OBJECTS) >= 1 and len(PLACES) >= 1 and len(ALL_TEMPLATES) >= 1

    def test_templates_is_single_canonical_only(self) -> None:
        """Single-template coverage contract — widening this is a deliberate refactor."""
        assert TEMPLATES == [CANONICAL_TEMPLATE]

    def test_canonical_template_contains_every_placeholder(self) -> None:
        """``_fill_template`` substitutes five placeholders; all must appear."""
        for placeholder in ("{name_A}", "{name_B}", "{name_C}", "{place}", "{object}"):
            assert placeholder in CANONICAL_TEMPLATE
