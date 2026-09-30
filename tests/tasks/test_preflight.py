"""Tests for the model-free tokenizer pre-flight check (``causalab.tasks.preflight``).

The pre-flight catches tokenization gotchas — chiefly a trailing-space prompt
suffix that orphans into its own whitespace token — before any model run, so
they don't surface downstream as a near-zero baseline accuracy.

These run on CPU with the tiny-random Llama tokenizer (no weights, no GPU), so
they are tiered ``numerical_unit``. Detection is verified to be tokenizer-family
robust: the tiny-random tokenizer is sentencepiece (its trailing-space token
decodes to ``''``), which is exactly the case a naive last-token check misses.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from causalab.tasks.loader import load_task
from causalab.tasks.preflight import (
    check_prompt_tokenization,
    run_task_preflight,
    task_forms_resolver,
)
from tests._helpers.tiny import tiny_random_tokenizer

# The default accepted answer forms (loader default / build_output_tokens):
# both the leading-space and bare forms.
_RESULT_TOKEN_PATTERN = lambda v: [f" {v}", v]  # noqa: E731


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_random_tokenizer()


@pytest.mark.numerical_unit
def test_trailing_space_prompt_flags_error(tokenizer):
    """A prompt ending in a space, with a leading-space answer form expected,
    raises an orphan-whitespace error."""
    samples = [("We will ask a question.\nAnswer: ", "Kate")]
    findings = check_prompt_tokenization(samples, _RESULT_TOKEN_PATTERN, tokenizer)
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    assert "orphan whitespace token" in errors[0].message


@pytest.mark.numerical_unit
def test_clean_prompt_no_error(tokenizer):
    """The same prompt with the trailing space removed raises no error."""
    samples = [("We will ask a question.\nAnswer:", "Kate")]
    findings = check_prompt_tokenization(samples, _RESULT_TOKEN_PATTERN, tokenizer)
    assert [f for f in findings if f.severity == "error"] == []


@pytest.mark.numerical_unit
def test_bare_only_pattern_not_flagged(tokenizer):
    """A trailing space is fine when no leading-space answer form is expected,
    so the orphan-whitespace error must not fire."""
    samples = [("We will ask a question.\nAnswer: ", "Kate")]
    findings = check_prompt_tokenization(samples, lambda v: [v], tokenizer)
    assert [f for f in findings if f.severity == "error"] == []


@pytest.mark.numerical_unit
def test_trailing_newline_suffix_flagged(tokenizer):
    """A prompt ending in a lone newline also orphans into its own whitespace
    token, so it is flagged for the same alignment reason as a trailing space."""
    samples = [("We will ask a question.\nAnswer:\n", "Kate")]
    findings = check_prompt_tokenization(samples, _RESULT_TOKEN_PATTERN, tokenizer)
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    assert "orphan whitespace token" in errors[0].message


@pytest.mark.numerical_unit
def test_entity_binding_shipped_config_fires(tokenizer):
    """Regression for the orphaned trailing space, which drives this task's
    baseline accuracy toward zero: the shipped entity_binding task ships ``prompt_suffix="\\nAnswer: "`` (trailing space),
    so the pre-flight must fire an orphan-whitespace error on its real samples."""
    task = load_task("entity_binding")
    # Sanity-check the fixture still carries the footgun this test exists to catch.
    sample = task.causal_model.sample_input()
    assert sample["raw_input"].endswith(" "), (
        "entity_binding no longer has a trailing-space prompt suffix; update this "
        "regression fixture (the bug it guards may have been fixed)."
    )

    findings = run_task_preflight(task, tokenizer, n_samples=4)
    errors = [f for f in findings if f.severity == "error"]
    assert errors, "pre-flight failed to catch entity_binding's trailing-space suffix"
    assert all("orphan whitespace token" in f.message for f in errors)


@pytest.mark.numerical_unit
def test_resolver_reads_output_tokens(tokenizer):
    """The forms resolver reads ``CausalModel.output_tokens`` — the single
    declaration of accepted answer forms.

    Here output_tokens declares a *bare-only* form for the answer; a trailing
    space therefore must NOT be flagged (the task does not expect a
    leading-space form)."""
    cm = SimpleNamespace(output_tokens={"answer": {"Kate": ["Kate"]}})
    task = SimpleNamespace(causal_model=cm)
    forms_for = task_forms_resolver(task)
    assert list(forms_for("Kate")) == ["Kate"]

    samples = [("We will ask a question.\nAnswer: ", "Kate")]
    findings = check_prompt_tokenization(samples, forms_for, tokenizer)
    assert [f for f in findings if f.severity == "error"] == []


@pytest.mark.numerical_unit
def test_resolver_default_for_undeclared_value(tokenizer):
    """A value not in output_tokens (e.g. an "UNKNOWN" sentinel), or a model with
    no output_tokens at all, falls back to the conventional ``[" v", v]`` forms
    so the prompt-level orphan-whitespace check still covers every sample."""
    declared = SimpleNamespace(
        causal_model=SimpleNamespace(output_tokens={"answer": {"Kate": ["Kate"]}})
    )
    assert list(task_forms_resolver(declared)("UNKNOWN")) == [" UNKNOWN", "UNKNOWN"]

    none_task = SimpleNamespace(causal_model=SimpleNamespace(output_tokens=None))
    assert list(task_forms_resolver(none_task)("Kate")) == [" Kate", "Kate"]
