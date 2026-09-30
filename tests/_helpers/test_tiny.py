"""Sanity tests for the tiny-random model factory.

Asserts that the ``tiny-random`` HF stub accepts every kwarg the runner
pipeline passes through to the model's ``forward`` — guards against the
failure mode where a transformers bump renames or drops a kwarg, the
smoke tier silently keeps passing because nothing in the runner actually
exercises that kwarg, and production breaks the day a real run hits the
rebound name.

The contract this file guards is **"the tiny stub speaks the runner's
calling convention"** — explicitly anchored to the kwargs enumerated in
`tests._helpers.tiny.RUNNER_FORWARD_KWARGS`, NOT
``signature(LlamaForCausalLM) == signature(LlamaForCausalLM)`` (which is
trivially true and detects nothing).

The coherent golden fixture (``chat-coherent`` = Qwen3-4B-Instruct) has no
Python factory and is exercised directly by the golden tier on GPU, so its
forward-kwarg contract is covered there rather than duplicated here.

Phase 3 of docs/TESTS.md, ``unit`` tier.
"""

from __future__ import annotations

import inspect

import pytest

from tests._helpers.tiny import (
    RUNNER_FORWARD_KWARGS,
    TINY_RANDOM_MODEL_NAME,
    tiny_random_model,
    tiny_random_runner_overrides,
)

pytestmark = pytest.mark.unit


def test_tiny_random_model_loads_as_llama() -> None:
    """The tiny-random stub instantiates as a real LlamaForCausalLM."""
    from transformers import LlamaForCausalLM

    model = tiny_random_model()
    assert isinstance(model, LlamaForCausalLM), (
        f"expected LlamaForCausalLM, got {type(model).__name__}"
    )
    # Sanity: this really is "tiny" (so the smoke wall budget holds).
    assert model.config.num_hidden_layers <= 4
    assert model.config.hidden_size <= 64


def test_tiny_model_forward_accepts_runner_kwargs() -> None:
    """The tiny-random stub's ``forward`` must accept every kwarg the runner passes.

    The runner pipeline (``causalab/neural/pipeline.py``) calls
    ``model.generate(**inputs, **defaults)``, which fans out into
    ``forward`` with the kwargs enumerated in
    `RUNNER_FORWARD_KWARGS`. If a transformers bump renames any of
    them (e.g. ``past_key_values`` → ``cache``) the smoke tier breaks; we
    want a fast, isolated unit test that flags the contract change before
    any tiered test even runs.

    Only the random (smoke) backbone is checked here; the coherent
    backbone (``chat-coherent`` = Qwen3-4B-Instruct) is exercised by the
    golden tier on GPU, which would surface the same rename.
    """
    sig = inspect.signature(tiny_random_model().forward)
    explicit_params = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    missing = RUNNER_FORWARD_KWARGS - explicit_params
    assert not missing, (
        f"tiny stub forward signature is missing kwargs the runner passes:\n"
        f"  missing: {sorted(missing)}\n"
        f"  forward has: {sorted(explicit_params)}\n"
        f"This usually means transformers renamed/dropped one of these "
        f"kwargs; update RUNNER_FORWARD_KWARGS and the runner call sites "
        f"together."
    )


def test_tiny_random_runner_overrides_minimum_fields(tmp_path) -> None:
    """``tiny_random_runner_overrides`` ships the keys the smoke harness needs.

    The numerical-tier goldens deliberately have no analogous override
    factory — their runner cfgs are self-contained yamls under
    ``tests/end_to_end/configs/golden/<runner>.yaml``. Smoke is the
    only tier that still patches a composed cfg in place, so this is
    the only override surface tested.
    """
    overrides = tiny_random_runner_overrides(tmp_path)
    # Each of these is load-bearing for a baseline runner run on CPU.
    expected_keys = {
        "experiment_root",
        "task.n_train",
        "task.n_test",
        "model.id",
        "model.name",
        "model.device",
        "model.dtype",
    }
    missing = expected_keys - set(overrides)
    assert not missing, f"tiny_random_runner_overrides missing keys: {missing}"
    assert overrides["model.name"] == TINY_RANDOM_MODEL_NAME
    assert overrides["model.device"] == "cpu"


def test_tiny_model_caches_across_calls() -> None:
    """``functools.lru_cache`` returns the same instance — load once per session."""
    a = tiny_random_model()
    b = tiny_random_model()
    assert a is b, "tiny_random_model() should be cached; got a fresh instance"
