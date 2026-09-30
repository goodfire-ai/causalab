# tests/conftest.py

import gc
import random
import warnings

import numpy as np
import pytest
import torch

from causalab.causal import Dom, Exo, FamilyDom, V, mechanism
from causalab.causal.model import CausalModel

# Tier markers recognised by the warn-mode collection hook below. Mirrors the
# tier taxonomy in docs/TESTS.md. Tests must declare exactly one of
# {smoke, numerical_unit, golden, property, unit} explicitly. `golden` is the
# sole GPU tier (value-pinned coherent-model runners); the rest are CPU. The
# default tier (`unit`) is the most common; tag with
# `pytestmark = pytest.mark.unit` at module scope.
_TIER_MARKERS = frozenset({"smoke", "numerical_unit", "golden", "property", "unit"})


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Reset the smoke flag so off-test composition (e.g. fixture teardown
    that happens to call ``load_runner_config``) defaults to non-smoke.

    After teardown, release GPU memory leaked by golden-tier tests.
    Golden tests load multi-GB coherent backbones as function locals or
    module-scoped fixtures with no teardown of their own; the nnsight/nnterp
    object graphs are cyclic, so those models survive the end of the test
    until a full ``gc.collect()`` pass — which CPython triggers on object
    counts, not GPU bytes, i.e. effectively never under this workload. In a
    single-process ``pytest -m golden`` run the dead models accumulate until
    whichever test loads a fresh backbone last OOMs. Running post-``yield``
    puts the collection after fixture finalization, so module-scoped model
    fixtures are reclaimed at module boundaries too. Gated on the ``golden``
    marker (the sole GPU tier) so CPU tiers don't pay for per-test GC.
    """
    try:
        return (yield)
    finally:
        if "golden" in {m.name for m in item.iter_markers()}:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def pytest_collection_modifyitems(config, items):
    """Fail-mode tier-marker enforcement (Phase 7).

    Every item carries exactly one of the tier markers: one lacking any is
    untagged, one carrying two is double-tagged — pytest *accumulates* a
    module's ``pytestmark``, a class's and a function's, so a class or
    function marked with another tier under a module-level mark belongs to
    both tiers and runs under either ``-m`` selection (``docs/TESTS.md``:
    "every test belongs to exactly one tier"). Either raises
    `pytest.UsageError` listing the offending nodeids; the fix for a
    mixed module is to drop its module-level mark and tag each class or
    function.

    Phase 2 shipped this as a warn-mode hook; Phase 7 promotes it to
    fail-mode alongside a backfill that tags every existing test file.
    The xdist master-process gate stays — only the master process
    decides whether the suite is well-formed; worker collections are
    a strict subset of the master's so they'd surface the same untagged
    set redundantly.
    """
    if getattr(config, "workerinput", None) is not None:
        # xdist worker — let the master enforce.
        return

    untagged: list[str] = []
    double: list[str] = []
    for item in items:
        tiers = _TIER_MARKERS.intersection(m.name for m in item.iter_markers())
        if not tiers:
            untagged.append(item.nodeid)
        elif len(tiers) > 1:
            double.append(f"{item.nodeid} ({', '.join(sorted(tiers))})")
    if not untagged and not double:
        return

    def _preview(nodeids: list[str]) -> str:
        # Truncate the offending list so a global drift doesn't dump
        # thousands of lines into the terminal. The first ten and a count
        # are enough to see what's going on.
        text = "\n  ".join(nodeids[:10])
        if len(nodeids) > 10:
            text += f"\n  ... and {len(nodeids) - 10} more"
        return text

    problems: list[str] = []
    if untagged:
        problems.append(
            f"{len(untagged)} tests lack a tier marker "
            f"(smoke/numerical_unit/golden/property/unit). Tag them with "
            f"`pytestmark = pytest.mark.<tier>` at module scope:\n"
            f"  {_preview(untagged)}"
        )
    if double:
        problems.append(
            f"{len(double)} tests carry more than one tier marker (a class or "
            f"function mark adds to the module's rather than replacing it). "
            f"Drop the module-level `pytestmark` and tag each class or "
            f"function with its one tier:\n  {_preview(double)}"
        )
    raise pytest.UsageError("\n".join(problems))


@pytest.fixture(autouse=True)
def _fresh_host_memos():
    """Empty the process-wide host-side memos before every test.

    ``encoding.encode`` memoizes the tokenizer's output per (tokenizer, texts),
    ``answers`` memoizes answer-token ids per (tokenizer, text) and
    ``env.FileDatasets`` memoizes checked table text and each ref's digest
    per file version, so a campaign's points stop re-deriving identical
    inputs. Across tests that sharing is coupling: a test that counts host
    reads or tokenizer calls would see fewer of them when an earlier test
    happened to encode the same rows through the same tokenizer. Every test
    starts from empty memos, as the first point of a campaign does.
    """
    from causalab.neural.shared import encoding
    from causalab.io import env
    from causalab.protocol import answers

    encoding._TOKENIZED.clear()  # pyright: ignore[reportPrivateUsage]
    answers._ENCODED_IDS.clear()  # pyright: ignore[reportPrivateUsage]
    env._checked_table_text.cache_clear()  # pyright: ignore[reportPrivateUsage]
    env._TABLE_DIGESTS.clear()  # pyright: ignore[reportPrivateUsage]
    yield


@pytest.fixture(scope="session")
def seed_everything():
    """Set random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def mcqa_causal_model():
    """
    Create a simple MCQA causal model fixture.

    This model represents a simplified version of the multiple-choice question answering task
    where questions have 4 choices and 1 correct answer.
    """
    # Define model variables
    NUM_CHOICES = 4
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Define object/color pairs for the questions
    COLOR_OBJECTS = [
        ("red", "apple"),
        ("yellow", "banana"),
        ("green", "leaf"),
        ("blue", "sky"),
        ("brown", "chocolate"),
        ("white", "snow"),
        ("black", "coal"),
        ("purple", "grape"),
        ("orange", "carrot"),
        ("pink", "flamingo"),
        ("gray", "elephant"),
        ("gold", "coin"),
    ]

    COLORS = [item[0] for item in COLOR_OBJECTS]

    def fill_prompt(question, symbols, choices):
        color, obj = question
        lines = [f"Which color is the {obj}?"]
        lines.extend(f"{symbols[i]}: {choices[i]}" for i in range(NUM_CHOICES))
        return "\n".join(lines + ["Answer:"])

    def answer_position(question, choices, fallback):
        return next((i for i, c in enumerate(choices) if c == question[0]), fallback)

    @mechanism
    def equations(
        question: Dom(COLOR_OBJECTS),
        choices: FamilyDom(Dom(COLORS), size=NUM_CHOICES),
        symbols: FamilyDom(Dom(ALPHABET), size=NUM_CHOICES),
        fallback: Exo(Dom(range(NUM_CHOICES))),
    ):
        answer_pointer = V(
            answer_position(question, choices, fallback), domain=Dom(range(NUM_CHOICES))
        )
        answer = V(" " + symbols[answer_pointer], domain=Dom(str))
        raw_input = V(fill_prompt(question, symbols, choices), domain=Dom(str))  # noqa: F841
        raw_output = V(answer, domain=Dom(str))  # noqa: F841
        return answer_pointer

    return CausalModel(equations, id="4_answer_MCQA_test")


@pytest.fixture(scope="session")
def mcqa_counterfactual_datasets(mcqa_causal_model, seed_everything):
    """
    Generate test counterfactual datasets for the MCQA task.

    Returns a dictionary with 3 types of counterfactual datasets:
    1. random_letter - Swapping the letter symbols while keeping choices same
    2. random_position - Moving the correct answer to a different position
    3. combined - Both letter and position changes

    Each type has small train and test sets.
    """
    model = mcqa_causal_model
    NUM_CHOICES = 4

    # Helper to check if inputs are well-formed
    def is_input_valid(x):
        # Check that the question color appears in the choices
        question_color = x["question"][0]
        choice_colors = [x[f"choices[{i}]"] for i in range(NUM_CHOICES)]
        symbols = [x[f"symbols[{i}]"] for i in range(NUM_CHOICES)]

        # The color must be in the choices and symbols must be unique
        return question_color in choice_colors and len(symbols) == len(set(symbols))

    # Counterfactual generator functions
    def random_letter_counterfactual():
        """Generate counterfactual with new random letters."""
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy

        # Get current symbols to avoid duplicates
        used_symbols = [input_setting[f"symbols[{i}]"] for i in range(NUM_CHOICES)]

        # Generate new set of symbols
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        available_symbols = [s for s in alphabet if s not in used_symbols]
        new_symbols = random.sample(available_symbols, NUM_CHOICES)

        # Update symbols in counterfactual
        for i in range(NUM_CHOICES):
            counterfactual[f"symbols[{i}]"] = new_symbols[i]

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    def random_position_counterfactual():
        """Generate counterfactual with answer moved to new position."""
        input_setting = model.sample_input(filter_func=is_input_valid)
        counterfactual = dict(input_setting)  # Make a copy

        # Get current answer position
        answer_position = model.new_trace(input_setting)["answer_pointer"]

        # Choose a different position
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)

        # Swap choices to move correct answer
        correct_color = counterfactual[f"choices[{answer_position}]"]
        counterfactual[f"choices[{answer_position}]"] = counterfactual[
            f"choices[{new_position}]"
        ]
        counterfactual[f"choices[{new_position}]"] = correct_color

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    def combined_counterfactual():
        """Generate counterfactual with both letter and position changes."""
        letter_cf = random_letter_counterfactual()

        # Start with the letter-changed counterfactual
        input_setting = letter_cf["input"]
        counterfactual = letter_cf["counterfactual_inputs"][0]

        # Now change position too
        answer_position = model.new_trace(input_setting)["answer_pointer"]
        available_positions = [i for i in range(NUM_CHOICES) if i != answer_position]
        new_position = random.choice(available_positions)

        # Swap choices
        correct_color = counterfactual[f"choices[{answer_position}]"]
        counterfactual[f"choices[{answer_position}]"] = counterfactual[
            f"choices[{new_position}]"
        ]
        counterfactual[f"choices[{new_position}]"] = correct_color

        return {"input": input_setting, "counterfactual_inputs": [counterfactual]}

    # Generate the datasets
    datasets = {}

    # Small size for tests
    train_size = 10
    test_size = 5

    # Generate datasets for each counterfactual type
    for name, generator in [
        ("random_letter", random_letter_counterfactual),
        ("random_position", random_position_counterfactual),
        ("combined", combined_counterfactual),
    ]:
        # Train dataset
        train_data = {"input": [], "counterfactual_inputs": []}
        for _ in range(train_size):
            sample = generator()
            train_data["input"].append(sample["input"])
            train_data["counterfactual_inputs"].append(sample["counterfactual_inputs"])

        # Test dataset
        test_data = {"input": [], "counterfactual_inputs": []}
        for _ in range(test_size):
            sample = generator()
            test_data["input"].append(sample["input"])
            test_data["counterfactual_inputs"].append(sample["counterfactual_inputs"])

        # Create list[CounterfactualExample]
        datasets[f"{name}_train"] = [
            {"input": inp, "counterfactual_inputs": cf}
            for inp, cf in zip(train_data["input"], train_data["counterfactual_inputs"])
        ]
        datasets[f"{name}_test"] = [
            {"input": inp, "counterfactual_inputs": cf}
            for inp, cf in zip(test_data["input"], test_data["counterfactual_inputs"])
        ]

    return datasets


# --------------------------------------------------------------------------- #
# The tiny fixtures are registry entries for the whole session
# --------------------------------------------------------------------------- #

#: The hub fixtures the suites compile documents against. None is a built-in
#: registry entry: today a document naming one loads only after some earlier
#: test in the same process has loaded the model (``load_model`` registers
#: the adapted config as a side effect), so any selection that reaches such a
#: test before a model-loading one fails with ``[V4] … not in the protocol
#: model registry`` — one process is enough, and ``-n`` only widens which
#: selections those are. Registering them once per session, from the config
#: alone with no weights read, removes the order dependence; it is a workaround
#: for ``load()`` not resolving a key's model info on demand (``get_model_info``
#: refuses to fetch mid-canonicalization so digests never depend on
#: connectivity), and it is not redundant with ``register_model_key``, which a
#: single test would have to call itself. The config comes from the hub cache
#: where it is warm (no network touched, so an offline session with a warm
#: cache pays nothing) and is fetched where it is not — CI restores no hub
#: cache, so on a fresh runner every fixture misses at session start, and a
#: cache-only lookup made this a silent no-op there: with ``HF_HOME`` at an
#: empty directory the t9 edit-groups test fails alone on the base and passes
#: here. A fixture unreachable either way is skipped with a warning naming it,
#: so the ``[V4]`` refusal it later earns from whatever test names it first
#: reads as the outage it is, not as a protocol bug.
_TINY_FIXTURES = (
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-gpt2",
    "tiny-random/qwen3.5-moe",
)


@pytest.fixture(scope="session", autouse=True)
def _tiny_fixtures_registered() -> None:
    from transformers import AutoConfig

    from causalab.protocol.registry import model_info_from_hf_config, register_model

    for key in _TINY_FIXTURES:
        try:
            config = AutoConfig.from_pretrained(key, local_files_only=True)
        except (OSError, ValueError):
            try:
                config = AutoConfig.from_pretrained(key)
            except (OSError, ValueError) as exc:
                warnings.warn(
                    f"tiny fixture {key!r} not registered at session start "
                    f"(hub cache cold, and the fetch did not succeed — under "
                    f"HF_HUB_OFFLINE none is attempted: {exc}); a document "
                    "naming it is refused with [V4] until something loads the "
                    "model",
                    stacklevel=1,
                )
                continue
        register_model(model_info_from_hf_config(key, config))
