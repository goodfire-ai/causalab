"""Tiny-config factory for smoke-tier runner tests and lightweight
causal-model construction.

One LM stub and one symbolic causal model live here:

* **tiny-random** — `tiny_random_model` / `tiny_random_tokenizer` /
  `tiny_random_runner_overrides`. Wraps
  ``hf-internal-testing/tiny-random-LlamaForCausalLM`` (2 hidden layers,
  hidden_size=16, vocab_size=32000). Real Llama architecture, **random
  weights** — fast to load, ideal for the smoke tier which only asserts
  "didn't crash + artifacts exist." Produces numerically nonsensical
  outputs.
* `tiny_chain_model` — a 3-node ``A → B → C`` symbolic
  [`CausalModel`][causalab.causal.model.CausalModel] for unit/property
  tests in ``tests/causal/`` and ``tests/io/`` that need a real (but
  minimal) causal model without paying the cost of inline construction
  in every file.

The smoke tier uses **tiny-random** because the dominant cost there is
HF model loading, and a 100 MB random-init Llama loads in well under a
second.

The coherent **golden** tier does *not* use a factory here — it composes
``chat-coherent`` (``Qwen/Qwen3-4B-Instruct-2507``) directly through Hydra
(``tests/end_to_end/configs/model/chat-coherent.yaml``), so locked
accuracy/output values reflect a real instruct model. There is no Python
handle for it; the golden harness only ever reaches it via config.

Why not mock? docs/TESTS.md is explicit: **no mocks of internal numerical
code.** Mocking the model risks silent contract drift between tests and
production. Wrapping a real (random-weight) model keeps the same
forward-pass contract while still hitting the <5min smoke wall budget.

The HF model name is also referenced directly from
``tests/end_to_end/configs/model/tiny-random.yaml`` (composable as
``model: tiny-random`` via the e2e Hydra searchpath, for smoke). This
module is only used by the tests themselves (sanity check + pre-warming +
direct-pipeline tests), not by the runner.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

# Public HF model name.
#
# tiny-random: real Llama architecture, random weights. Outputs are
#   numerically meaningless; suitable only for smoke tests that check
#   the pipeline doesn't crash. The coherent golden fixture
#   (``chat-coherent`` = Qwen3-4B-Instruct) has no Python factory — it is
#   composed via Hydra (``configs/model/chat-coherent.yaml``).
TINY_RANDOM_MODEL_NAME = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# Kwargs the runner pipeline actually passes (directly or via ``generate``)
# to the model's ``forward``. This is the load-bearing contract the smoke
# tier exercises:
#
# * ``input_ids`` / ``attention_mask`` — produced by
#   `causalab.neural.pipeline.LMPipeline.load` from
#   ``tokenizer(prompts, ...)``; passed to ``model.generate`` which then
#   calls ``forward`` with them.
# * ``position_ids`` — built by ``LMPipeline.load`` via
#   ``left_pad_position_ids`` (see ``causalab/neural/pipeline.py``) when the
#   pipeline is constructed with ``position_ids=True``; plumbed through
#   ``generate`` to ``forward``.
# * ``past_key_values`` — populated by HF ``generate`` between forward
#   calls when ``use_cache=True`` (which the pipeline sets as a default).
# * ``use_cache`` — passed via ``LMPipeline.generate``'s ``defaults`` dict.
#
# If transformers ever renames any of these (e.g. ``past_key_values`` →
# ``cache``) the smoke tier will start blowing up; this list is the
# canonical anchor for the sanity test in ``test_tiny.py``.
RUNNER_FORWARD_KWARGS: frozenset[str] = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "use_cache",
    }
)


# ---------------------------------------------------------------------------
# tiny-random — random-init Llama for smoke tier
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def tiny_random_model() -> Any:
    """Return a cached `transformers.LlamaForCausalLM` instance.

    Module-scope ``lru_cache`` is critical — the smoke tier loads this
    once per pytest session, then every parametrized baseline test reuses
    the same in-memory model. Without caching, even a 0.7s cached HF load
    times ~14 baselines is over 10s of overhead for nothing.

    The returned object is the *real* HF Llama model with random
    weights. Smoke tests that actually run the runner pipeline don't
    touch this directly — they go through
    ``causalab/configs/model/tiny-random.yaml``, which causes
    ``LMPipeline`` to build its own (also-cached-by-HF) instance via
    ``AutoModelForCausalLM.from_pretrained``. This helper exists for the
    sanity test below and for tests that want to pre-warm the HF cache.
    """
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(TINY_RANDOM_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def tiny_random_tokenizer() -> Any:
    """Return a cached `transformers.AutoTokenizer` for the tiny-random stub."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY_RANDOM_MODEL_NAME)


# ---------------------------------------------------------------------------
# tiny-random GPT-2 — random-init *absolute-position* model
# ---------------------------------------------------------------------------
#
# The Llama stub above is rotary (RoPE), whose attention depends only on
# *relative* position — so a uniform left-pad shift cancels and it is blind to
# whether a forward pass was given correct ``position_ids``. To test the left-pad
# ``position_ids`` convention (``pipeline.ensure_position_ids`` / the non-generate
# forward sites) one needs a model with **absolute / learned** position
# embeddings, where a wrong position genuinely changes the activations. GPT-2's
# ``transformer.wpe`` is exactly that. Real architecture, random weights, tiny.
TINY_RANDOM_GPT2_MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


@functools.lru_cache(maxsize=1)
def tiny_random_gpt2_model() -> Any:
    """Return a cached random-weight ``GPT2LMHeadModel`` (absolute positions).

    Same role as `tiny_random_model`, but for the absolute-position family:
    its ``transformer.wpe`` makes collected activations sensitive to the left-pad
    ``position_ids`` convention, where the RoPE stub is not. Outputs are
    numerically meaningless (random weights) — only for tests that need a ``wpe``
    architecture, not behaviour. Module-scope cache so it loads once per session.
    """
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)


# ---------------------------------------------------------------------------
# Fresh (uncached) instances — for nnsight/nnterp trace tests
# ---------------------------------------------------------------------------
#
# The ``lru_cache`` singletons above are shared across suites; leftover
# forward hooks on a shared instance break a later nnsight trace
# (``MissedProviderError`` on the block/model outputs). Tests that wrap a model in an nnterp
# ``StandardizedTransformer`` (``tests/neural/test_site.py`` /
# ``test_head_view.py``) therefore build a private, seeded instance from config —
# through these factories, so the recipe lives in one place.
def fresh_tiny_random_llama(mutate_config: Any = None) -> tuple[Any, Any]:
    """A fresh (uncached) random-weight tiny Llama + tokenizer, seeded for
    determinism. Real RoPE architecture, numerically meaningless outputs.
    ``mutate_config`` (optional callable) edits the loaded config in place before
    the model is built — for architecture variants (GQA, decoupled ``head_dim``)."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    cfg = AutoConfig.from_pretrained(TINY_RANDOM_MODEL_NAME)
    if mutate_config is not None:
        mutate_config(cfg)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    tok = AutoTokenizer.from_pretrained(TINY_RANDOM_MODEL_NAME)
    return model, tok


def fresh_tiny_random_gpt2(mutate_config: Any = None) -> tuple[Any, Any]:
    """A fresh (uncached) random-weight tiny GPT-2 + tokenizer, seeded for
    determinism. Built from the tiny-random-gpt2 config (not a shrunk ``gpt2``
    config): its vocab matches its tokenizer, so real tokenized text has
    in-range ids. ``mutate_config`` (optional callable) edits the loaded config
    in place before the model is built — mirrors the llama factory."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel

    cfg = AutoConfig.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)
    if mutate_config is not None:
        mutate_config(cfg)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg).eval()
    tok = AutoTokenizer.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)
    return model, tok


def tiny_random_runner_overrides(experiment_root: Path | str) -> dict[str, Any]:
    """Return Hydra overrides pointing a baseline runner at the tiny-random
    Llama stub and shrinking the dataset to smoke-test scale.

    These overrides are the bare minimum to drive an existing baseline
    config through the runner pipeline without loading a 1B-parameter
    model. They're applied *after* composition (via ``OmegaConf.update``)
    so callers don't have to teach Hydra about a new override syntax.

    Every key returned is a dotted Hydra path that **must already exist**
    on a composed baseline runner cfg (per docs/CODEBASE.md invariant 12,
    these dataset/model knobs always live on task / model configs). The
    smoke fixture relies on that — it applies these without
    ``force_add=True`` so an upstream rename of any of them trips the
    smoke tier loudly instead of silently growing a stale duplicate key.

    Args:
        experiment_root: Directory the runner should write its artifacts to. Tests pass
            ``tmp_path`` so each parametrized run is hermetic.
    """
    return {
        "experiment_root": str(experiment_root),
        "task.n_train": 2,
        "task.n_test": 2,
        "model.id": "tiny-random",
        "model.name": TINY_RANDOM_MODEL_NAME,
        "model.device": "cpu",
        "model.dtype": "float32",
        # Keep batch_size small so we don't try to allocate batches larger
        # than the dataset (n_train=2). Most analyses default to 32.
        "baseline.batch_size": 2,
    }


# NOTE: there is intentionally no coherent-model factory or
# ``*_runner_overrides`` here. The golden tier composes self-contained
# yamls under ``tests/end_to_end/configs/golden/<runner>.yaml`` that pin
# the model (``/model: chat-coherent``) and dataset/batch knobs at golden
# scale directly; no runtime override layer or Python handle is needed.
# ``tiny-random`` still uses `tiny_random_runner_overrides` because
# smoke deliberately exercises the e2e smoke yamls (under
# ``tests/end_to_end/configs/smoke/``) and swaps in the random stub.


# ---------------------------------------------------------------------------
# Symbolic causal model — unrelated to the LM factories above
# ---------------------------------------------------------------------------


def tiny_chain_model(model_id: str = "tiny_chain_model") -> Any:
    """Return a fresh 3-node ``A → B → C`` symbolic ``CausalModel``.

    ``A`` is a binary input variable. ``B`` and ``C`` are identity-propagating
    mechanisms. ``raw_input`` / ``raw_output`` are present (both are required
    by `__init__`).

    Why a factory and not a session-scoped fixture? Several callers mutate
    the returned model (set ``input_filter``, intervene), so handing out the
    same instance would couple unrelated tests. Construction is cheap (no
    tensor work), so a fresh model per call is the right default.

    Consumers (current and planned):

    * ``tests/causal/test_counterfactuals.py`` — filter / label / sample
      helpers in ``causal/counterfactuals.py``, and the schema property
      tests for ``CounterfactualExample``.
    * ``tests/io/plots/test_causal_graph.py`` — DAG / forward-pass
      visualization in ``io/plots/causal_graph.py``.

    ``tests/causal/test_trace.py`` does not use this: it defines the small
    models each trace test needs.
    """
    from causalab.causal import CausalModel, Dom, V, mechanism

    @mechanism
    def equations(A: Dom([0, 1])):
        B = V(A, domain=Dom([0, 1]))
        raw_input = V(f"Input A={A}", domain=Dom(str))  # noqa: F841
        C = V(B, domain=Dom([0, 1]))
        raw_output = V(f"Output C={C}", domain=Dom(str))  # noqa: F841
        return C

    return CausalModel(equations, id=model_id)
