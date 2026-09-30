"""GPT-J 6B on the reference engine against the Hugging Face forward.

``EleutherAI/gpt-j-6b`` at revision ``main`` (fp32 weights) in fp32 on one
CUDA device, eager attention. The reference is a separate
``AutoModelForCausalLM`` load of the same revision, run one prompt at a time
with no padding; it is freed before the engine loads its own copy, so one
24 GB fp32 model is resident until the last test adds the 12 GB fp16 one. The engine reads the ``lm_head`` output at the
last position for all prompts in one left-padded batch, and for each prompt
alone.

The checks:

* the loaded tree is ``gptj_tree`` and the loaded config reads to the static
  registry row (up to the precision the checkpoint's config declares);
* the engine's clean last-position logits equal the HF forward's: at the
  kernel-noise band batch by batch (``BATCH1_ATOL``), and within
  ``PADDED_ATOL`` for the padded batch, whose GEMM shapes differ from the
  unpadded reference; the top-1 token agrees on every prompt;
* the weights are GPT-J's and not noise: the Eiffel Tower prompt's top-1 is
  " Paris" and the counting prompt's is " five" (📐 the first H100 run
  asserted " Paris" after "The capital of France is" and measured
  " a", a fluent continuation; the check moved to the two prompts whose
  next token is determined);
* the parallel residual identity the family declares is exact in fp32 on
  the real tower, at the first, a middle and the last layer;
* the fp16 weights of revision ``float16`` (what the runs load) give the same
  top-1 token as the fp32 reference on every prompt.

The bands are declared before the first run, not fitted to it: fp32 with
TF32 off gives a per-element error of order ``eps · sqrt(K)`` per GEMM
(``eps`` 1.2e-7, ``K`` 4096 to 16384), which compounds over 28 blocks to
well under 1e-3 on logits of magnitude ~10 to 100. The measured values are
printed for the run log.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest
import torch

from causalab.protocol.registry import get_model_info, identities_for
from causalab.protocol.schema import PROTOCOL_VERSION

pytestmark = pytest.mark.golden

KEY = "EleutherAI/gpt-j-6b"
PROMPTS = (
    "The capital of France is",
    "The Eiffel Tower is located in the city of",
    "def fibonacci(n):",
    "One, two, three, four,",
)
#: Same weights, same kernels, same shapes: only kernel nondeterminism.
BATCH1_ATOL = 1e-4
#: Left padding changes the GEMM shapes, so the reduction order differs.
PADDED_ATOL = 1e-3
IDENTITY_LAYERS = (0, 13, 27)
#: The static row as declared, read at collection: the ``bundle`` fixture's
#: ``load_model`` re-registers ``KEY`` from the loaded config, so a lookup in
#: a test body returns ``bundle.info`` itself and the comparison holds for
#: any declared row (the same trap ``GPTJ_6B_STATIC`` in
#: ``tests/protocol/test_registry_shapes.py`` avoids).
STATIC = get_model_info(KEY)


def _no_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """The HF forward's last-position logits per prompt, computed and freed
    before the engine loads."""
    if not torch.cuda.is_available():
        pytest.skip("the golden tier is the accelerator tier")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _no_tf32()
    tokenizer = AutoTokenizer.from_pretrained(KEY, revision="main")
    model = AutoModelForCausalLM.from_pretrained(
        KEY, revision="main", dtype=torch.float32, attn_implementation="eager"
    )
    model = model.to("cuda").eval()
    logits = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            logits.append(model(input_ids=ids).logits[0, -1].float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"tokenizer": tokenizer, "logits": torch.stack(logits)}


@pytest.fixture(scope="module")
def bundle(reference: dict[str, Any]) -> Any:
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    _no_tf32()
    return load_model(KEY, "main", dtype="fp32", device="cuda")


def _doc(sites: dict[str, dict[str, Any]], pos: Any) -> dict[str, Any]:
    reads = {f"r_{name}": {"site": name, "pos": pos} for name in sites}
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": {"base": {"dataset": "inline", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": list(reads)}},
            "sites": sites,
            "reads": reads,
            "save": [
                {"read": r, "model": "original", "file_path": f"{r}.safetensors"}
                for r in reads
            ],
        },
    }


def _last_logits(bundle: Any, prompts: tuple[str, ...]) -> torch.Tensor:
    from causalab.neural.engines.pytorch_hooks.executor import PointExecutor

    from tests._helpers.engines import executor_for

    executor = executor_for(
        PointExecutor,
        _doc({"head": {"component": "lm_head"}}, -1),
        bundle,
        base_texts=list(prompts),
    )
    return executor.read_value("r_head")[:, 0].float().cpu()


def test_the_loaded_tree_and_config_are_the_registry_row(bundle):
    assert bundle.adapter.family == "gptj_tree"
    assert len(bundle.blocks) == 28
    for field in (
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "family",
        "parallel_plan",
    ):
        assert getattr(bundle.info, field) == getattr(STATIC, field), field


def test_clean_logits_equal_the_hf_forward_prompt_by_prompt(bundle, reference):
    want = reference["logits"]
    for i, prompt in enumerate(PROMPTS):
        got = _last_logits(bundle, (prompt,))[0]
        diff = float((got - want[i]).abs().max())
        print(f"batch-1 {prompt!r}: max |engine - hf| = {diff:.3e}")
        assert diff <= BATCH1_ATOL, prompt
        assert int(got.argmax()) == int(want[i].argmax()), prompt


def test_clean_logits_equal_the_hf_forward_in_a_padded_batch(bundle, reference):
    want = reference["logits"]
    got = _last_logits(bundle, PROMPTS)
    diff = float((got - want).abs().max())
    print(f"padded batch of {len(PROMPTS)}: max |engine - hf| = {diff:.3e}")
    print(f"logit magnitude: max |hf| = {float(want.abs().max()):.3e}")
    assert diff <= PADDED_ATOL
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))


def test_the_weights_are_gpt_j_and_not_noise(reference):
    tokenizer = reference["tokenizer"]
    top = [tokenizer.decode(int(t)) for t in reference["logits"].argmax(dim=-1)]
    print("top-1 per prompt:", dict(zip(PROMPTS, top)))
    assert top[1] == " Paris" and top[3] == " five"


def test_the_parallel_residual_is_exact_on_the_real_tower(bundle):
    from causalab.neural.engines.pytorch_hooks.executor import PointExecutor

    from tests._helpers.engines import executor_for

    (identity,) = identities_for("gptj_tree")
    names = (identity.component, *identity.inputs)
    atol, rtol = identity.tolerance_for("fp32")
    for layer in IDENTITY_LAYERS:
        sites = {n: {"component": n, "layers": [layer]} for n in names}
        executor = executor_for(
            PointExecutor, _doc(sites, "all"), bundle, base_texts=[PROMPTS[1]]
        )
        values = {n: executor.read_value(f"r_{n}") for n in names}
        first, *rest = identity.inputs
        total = values[first]
        for name in rest:
            total = total + values[name]
        diff = float((total - values[identity.component]).abs().max())
        print(f"layer {layer}: max |declared sum - block_output| = {diff:.3e}")
        torch.testing.assert_close(
            total, values[identity.component], atol=atol, rtol=rtol
        )


def test_the_fp16_revision_agrees_on_the_top_token(reference):
    """The runs' weights: revision ``float16`` loaded in fp16. The loader
    cache is emptied first; the module's fp32 bundle stays referenced by its
    fixture, so 24 GB + 12 GB are resident on the card here."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    from tests._helpers.resident_models import evict_resident_models

    evict_resident_models()
    fp16 = load_model(KEY, "float16", dtype="fp16", device="cuda")
    assert fp16.adapter.family == "gptj_tree"
    assert fp16.info.native_dtype == "fp16"
    got = _last_logits(fp16, PROMPTS)
    want = reference["logits"]
    diff = float((got - want).abs().max())
    print(f"fp16 revision vs fp32 reference: max |diff| = {diff:.3e}")
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))
