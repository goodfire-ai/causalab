"""The generated frame on the nnsight engine.

One ``model.generate`` trace per group: prompt-frame taps and writes bind
occurrence 0 of their locations — the prefill — and the decode steps are
walked with ``tracer.iter``, occurrence ``j`` being the step that consumes
generated token ``j-1``. The reference engine hand-rolls the same decode with
hooks, which makes it the oracle here: the same documents through both, ids
and activations agreeing.

Plus what parity cannot say: the greedy self-consistency pin (the argmax of a
continuation ``lm_head`` read reproduces the decoded ids), and the
DeltaNet bridge — the DeltaNet state read
per decode step, through the *recurrent* kernel's own address, continuous
with the prefill chunks.
"""

from __future__ import annotations

import pytest

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION

from tests.protocol._docs import saved

from .test_parity_module_boundaries import _assert_same, _data, _executor


pytestmark = pytest.mark.smoke

DEPTH = 4
QWEN_ATTENTION_LAYER = 3
DELTANET_LAYER = 0


def _gen_doc(component: str, *, layer: int | None, pos: dict | None = None) -> dict:
    site: dict = {"component": component}
    if layer is not None:
        site["layers"] = layer
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "positions": {
                "window": pos or {"generated": {"max_new_tokens": DEPTH}, "all": True}
            },
            "reads": {"r": {"site": "tap", "pos": "window"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


# --------------------------------------------------------------------------- #
# parity: the reference engine's decode is the oracle
# --------------------------------------------------------------------------- #

LLAMA_GEN_READS = [
    ("block_output", 1),
    ("attention_output", 1),
    ("mlp_output", 1),
    ("attention_query", 1),
    ("attention_z", 1),
    ("ln_final", None),
    ("lm_head", None),
]


@pytest.mark.parametrize("component,layer", LLAMA_GEN_READS)
def test_generated_read_parity(hooks_llama, trace_llama, component, layer):
    doc = _gen_doc(component, layer=layer)
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"generated read of {component!r}")


def test_generated_parity_on_the_hybrid_fixture(hooks_qwen, trace_qwen):
    """The target architecture in miniature: a DeltaNet layer's boundary and a
    full-attention interior slot, per step."""
    for component, layer in (
        ("attention_output", DELTANET_LAYER),
        ("attention_z", QWEN_ATTENTION_LAYER),
        ("lm_head", None),
    ):
        doc = _gen_doc(component, layer=layer)
        hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False).read_value(
            "r"
        )
        traced = _executor(
            TracePointExecutor, doc, trace_qwen, with_cf=False
        ).read_value("r")
        _assert_same(hooked, traced, f"generated read of {component!r}")


def test_the_decoded_ids_agree_with_the_reference_engine(hooks_qwen, trace_qwen):
    """Same greedy continuation on both engines — the frame itself, not just
    the activations."""
    doc = _gen_doc("block_output", layer=0)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    hooked.read_value("r"), traced.read_value("r")
    assert hooked.generated_ids("r") == traced.generated_ids("r")
    assert hooked.addressed_steps("r") == traced.addressed_steps("r")


def _prefill_write_doc() -> dict:
    """A counterfactual swap at ``block_output`` L0, last prompt position,
    on ``patched``, which reads the generated window at the same layer."""
    doc = _gen_doc("block_output", layer=0)
    doc["data"] = _data(with_cf=True)
    doc["method"]["sites"]["src"] = {"component": "block_output", "layers": [0]}
    doc["method"]["reads"]["v_cf"] = {"site": "src", "pos": -1}
    doc["method"]["writes"] = {
        "patch": {"site": "src", "pos": -1, "do": {"swap": "v_cf"}}
    }
    # `r` moves onto the written model; the operand is read un-intervened
    doc["method"]["intervened_models"] = {
        "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
        "patched": {"input": "base", "reads": ["r"], "writes": ["patch"]},
    }
    doc["method"]["save"][0]["model"] = "patched"
    return doc


def test_a_write_reaches_the_continuation_identically(hooks_qwen, trace_qwen):
    """Writes are prefill-only on both engines (here: everything binds
    occurrence 0 — the prefill), and reach the continuation through the first
    token and the cache. The patched continuation read must agree."""
    doc = _prefill_write_doc()
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    _assert_same(
        hooked.read_value("r"),
        traced.read_value("r"),
        "a patched continuation read",
    )
    assert hooked.generated_ids("r") == traced.generated_ids("r")


def _steer_doc(*, during_generation: bool) -> dict:
    """A literal ``add_scaled`` at ``block_output`` L0, last position, on
    ``steered``, which reads the generated window at the same layer. A write
    in force during generation takes no read operand (V16), so the operand
    is a literal, as in ``test_generation_key_writes.py``."""
    doc = _gen_doc("block_output", layer=0)
    doc["method"]["writes"] = {
        "steer": {
            "site": "tap",
            "pos": -1,
            "do": {"add_scaled": {"op": 5.0, "alpha": 1.0}},
        }
    }
    steered: dict = {"input": "base", "reads": ["r"], "writes": ["steer"]}
    if during_generation:
        steered["writes_during_generation"] = True
    doc["method"]["intervened_models"] = {"steered": steered}
    doc["method"]["save"][0]["model"] = "steered"
    return doc


def test_writes_during_generation_is_refused_by_the_executor(hooks_qwen, trace_qwen):
    """``writes_during_generation`` keeps a model's writes installed through
    the decode (§2.9). The nnsight trace binds each write's prefill
    occurrence only, so it cannot honour the flag. Routing refuses such a
    document first (rule 13, ``generation_writes``); this is the executor's
    own refusal for a caller that hands the document to it directly, which
    used to run prefill-only with no warning.

    The valid-work twin is the reference engine: the flag changes its
    continuation, so running without the flag is a different result, not a
    harmless approximation."""
    prefill_only = _executor(
        PointExecutor, _steer_doc(during_generation=False), hooks_qwen, with_cf=False
    ).read_value("r")
    during = _executor(
        PointExecutor, _steer_doc(during_generation=True), hooks_qwen, with_cf=False
    ).read_value("r")
    assert prefill_only.shape == during.shape
    assert float((during - prefill_only).abs().max()) > 1e-3

    flagged = _executor(
        TracePointExecutor,
        _steer_doc(during_generation=True),
        trace_qwen,
        with_cf=False,
    )
    with pytest.raises(ProtocolError, match="writes_during_generation") as refused:
        flagged.read_value("r")
    assert refused.value.code == "P4"
    assert "'steered'" in str(refused.value)
    assert "generation_writes" in str(refused.value)
