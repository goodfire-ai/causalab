"""Greedy decode through the protocol, against HF's own generate (§2.3, §4).

``model.generate(do_sample=False)`` is the oracle here, never the
implementation: it reads a checkpoint's ``generation_config``, whose
defaults would leak into a run that has to be a pure function of the
document. So the executor decodes explicitly and these tests pin that the
two agree token-for-token.

The batch is deliberately two rows of unequal length: left padding is what
makes ``logits[:, -1]`` the whole batch's next token, and a same-length
batch would not notice if that broke.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared.encoding import encode
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor, RaggedValue
from causalab.protocol.schema import parse_document

from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._docs import in_order, saved

pytestmark = pytest.mark.smoke

PROMPTS = ["the quick brown fox jumps", "a slow green turtle sleeps deeply today"]
BUDGET = 6


def _rows() -> list[dict[str, Any]]:
    return [{"input": text} for text in PROMPTS]


def _doc(
    key: str,
    *,
    anchor: dict[str, Any] | None = None,
    site: str = "lm_head",
    steer: float | None = None,
    during_generation: bool = False,
) -> dict[str, Any]:
    """A document reading the continuation, optionally under a steer — kept
    in force through the decode steps when ``during_generation`` (§2.9)."""
    raw: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": key, "revision": "main"},
        "data": {"base": {"dataset": "probe", "field": "input"}},
        "method": {
            "positions": {
                "cont": {
                    "generated": {"max_new_tokens": BUDGET},
                    **(anchor or {"all": True}),
                }
            },
            "sites": {
                "lm_head": {"component": "lm_head"},
                "mid": {"component": "block_output", "layers": [1]},
            },
        },
    }
    model = "original"
    if steer is not None:
        # a literal-scalar operand (§2.8) — enough to move the continuation
        # without dragging a tensor fixture into the test
        raw["method"]["writes"] = {
            "steer": {
                "site": "mid",
                "pos": -1,
                "do": {"add_scaled": {"op": steer, "alpha": 1.0}},
            }
        }
        model = "steered"
    raw["method"]["intervened_models"] = {
        model: {
            "input": "base",
            "reads": ["cont"],
            **({"writes": ["steer"]} if steer is not None else {}),
            **({"writes_during_generation": True} if during_generation else {}),
        }
    }
    raw["method"]["reads"] = {"cont": {"site": site, "pos": "cont"}}
    raw["method"]["save"] = [saved("cont", model, "cont.safetensors")]
    return in_order(raw)


def _executor(bundle, raw: dict[str, Any]):
    def load_tensors(path: str) -> Any:
        raise AssertionError(f"no document here loads tensors ({path})")

    return PointExecutor(
        parse_document(raw),
        bundle,
        role_rows={"base": _rows()},
        role_fields={"base": "input"},
        load_tensors=load_tensors,
    )


def _hf_greedy(bundle, budget: int = BUDGET) -> torch.Tensor:
    batch = encode(bundle.tokenizer, PROMPTS)
    with torch.no_grad():
        out = bundle.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            max_new_tokens=budget,
            do_sample=False,
            use_cache=True,
        )
    return out[:, batch.input_ids.shape[1] :]


def _hf_greedy_with_every_step_steer(
    bundle, steer: float, layer: int = 1, budget: int = BUDGET
) -> torch.Tensor:
    """HF's generate with a plain forward hook that adds ``steer`` to the last
    position of block ``layer``'s output on **every** forward — the prefill's
    last prompt token, then each decode step's one token. This is the paper
    semantics of a steer held through generation, spelled without the
    protocol, and the oracle ``writes_during_generation`` is held to."""
    block = bundle.model.model.layers[layer]

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[:, -1, :] += steer
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = block.register_forward_hook(hook)
    try:
        return _hf_greedy(bundle, budget)
    finally:
        handle.remove()


def _decoded_ids(executor: PointExecutor) -> torch.Tensor:
    executor.run_all()
    (continuation,) = executor._continuations.values()
    return continuation.token_ids


@pytest.mark.parametrize("key", [TINY_LLAMA, TINY_GPT2])
def test_continuation_matches_hf_greedy(key):
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    bundle = load_model(key)
    ours = _decoded_ids(_executor(bundle, _doc(key)))
    assert torch.equal(ours, _hf_greedy(bundle))


def test_lm_head_read_reproduces_the_tokens_it_generated(llama_bundle):
    """The read is the distribution *after* each generated token, so its
    argmax is the next token — for steps 0..n-2 that is the token the decode
    actually emitted next. A read that disagreed here would mean the kept
    activations and the decode had drifted apart."""
    executor = _executor(llama_bundle, _doc(TINY_LLAMA))
    value = executor.read_value("cont")
    assert isinstance(value, torch.Tensor)  # equal widths, no EOS in tiny-random
    predicted = value.argmax(dim=-1)
    (continuation,) = executor._continuations.values()
    assert torch.equal(predicted[:, :-1], continuation.token_ids[:, 1:])


def test_last_step_is_one_position(llama_bundle):
    executor = _executor(llama_bundle, _doc(TINY_LLAMA, anchor={"index": -1}))
    value = executor.read_value("cont")
    assert isinstance(value, torch.Tensor)
    assert value.shape[:2] == (len(PROMPTS), 1)


def test_any_site_is_readable_at_generated_positions(llama_bundle):
    """Not just the head: a continuation read at a residual-stream site
    harvests activations over the generated tokens, which is what
    "read where the model said it" will need."""
    executor = _executor(llama_bundle, _doc(TINY_LLAMA, site="mid"))
    value = executor.read_value("cont")
    assert isinstance(value, torch.Tensor)
    width = llama_bundle.model.config.hidden_size
    assert value.shape == (len(PROMPTS), BUDGET, width)


def test_a_steer_in_the_prefill_moves_the_continuation(llama_bundle):
    """Writes are prefill-only, and this is what that buys: the intervention
    reaches the continuation through the first token's logits and the KV
    cache it left behind — no hook fires during a decode step."""
    plain = _decoded_ids(_executor(llama_bundle, _doc(TINY_LLAMA)))
    steered = _decoded_ids(_executor(llama_bundle, _doc(TINY_LLAMA, steer=5.0)))
    assert not torch.equal(plain, steered)


def test_writes_during_generation_follow_the_every_step_hook_oracle(llama_bundle):
    """With ``writes_during_generation`` the steer fires at every decode step
    too, at the token being decoded — token for token what a forward hook
    held through HF's own generate produces. The prefill-only decode of the
    same document is a different continuation, which is what the flag buys;
    and every step counted the write once (§4 "Fires")."""
    executor = _executor(
        llama_bundle, _doc(TINY_LLAMA, steer=5.0, during_generation=True)
    )
    every_step = _decoded_ids(executor)
    assert torch.equal(every_step, _hf_greedy_with_every_step_steer(llama_bundle, 5.0))
    assert executor.fires[("steered", "base")] == {"steer": 1}
    prefill_only = _decoded_ids(_executor(llama_bundle, _doc(TINY_LLAMA, steer=5.0)))
    assert not torch.equal(every_step, prefill_only)


def test_a_decode_moves_prompt_frame_reads_only_by_float_noise(
    llama_bundle, monkeypatch
):
    """Adding a continuation read leaves prompt-frame values alone to within
    float noise, including bit-identical results on some platforms.

    Measured: ~4e-9 on tiny-random fp32. The cause is `use_cache=True`, which
    a decoding group needs and which takes a slightly different kernel path
    through the prefill. It is why the flag is set from the decode depth
    rather than always: a document that does not decode keeps the exact
    numbers its goldens were captured with. Check the cache flag directly;
    floating-point inequality is not evidence that decoding ran."""
    cache_flags = []
    forward = llama_bundle.model.forward

    def observed_forward(*args, **kwargs):
        cache_flags.append(kwargs.get("use_cache"))
        return forward(*args, **kwargs)

    monkeypatch.setattr(llama_bundle.model, "forward", observed_forward)
    without: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "probe", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["tail"]}},
            "sites": {"mid": {"component": "block_output", "layers": [1]}},
            "reads": {"tail": {"site": "mid", "pos": -1}},
            "save": [saved("tail", "original", "tail.safetensors")],
        },
    }
    plain = _executor(llama_bundle, without).read_value("tail")
    assert cache_flags == [False]
    cache_flags.clear()

    # the sections spread at the top level are hoisted into `method` by
    # `in_order`, overriding the base document's of the same name
    withgen = {
        **without,
        "intervened_models": {"original": {"input": "base", "reads": ["tail", "cont"]}},
        "positions": {"cont": {"generated": {"max_new_tokens": BUDGET}, "all": True}},
        "reads": {
            **without["method"]["reads"],
            "cont": {"site": "mid", "pos": "cont"},
        },
        "save": [
            *without["method"]["save"],
            saved("cont", "original", "cont.safetensors"),
        ],
    }
    ordered = in_order(withgen)
    also = _executor(llama_bundle, ordered).read_value("tail")
    assert isinstance(plain, torch.Tensor) and isinstance(also, torch.Tensor)
    assert torch.allclose(plain, also, atol=1e-7, rtol=0)
    assert len(cache_flags) > 1  # prefill followed by actual decode forwards
    assert all(flag is True for flag in cache_flags)


def test_an_early_eos_shortens_that_row(llama_bundle):
    """Rows end where they end: forcing row 0's first draw to be EOS leaves
    it with no real generated tokens, and the read goes ragged rather than
    failing."""
    executor = _executor(llama_bundle, _doc(TINY_LLAMA))
    eos = llama_bundle.tokenizer.eos_token_id
    executor.decoding = {"mode": "deterministic", "eos_token_ids": [eos]}
    original = llama_bundle.model.forward

    def eos_first(*args, **kwargs):
        out = original(*args, **kwargs)
        if kwargs.get("past_key_values") is None:  # the prefill
            out.logits[0, -1, :] = -1e4
            out.logits[0, -1, eos] = 1e4
        return out

    llama_bundle.model.forward = eos_first  # type: ignore[method-assign]
    try:
        value = executor.read_value("cont")
    finally:
        llama_bundle.model.forward = original  # type: ignore[method-assign]
    assert isinstance(value, RaggedValue)
    assert value.widths == (0, BUDGET)
