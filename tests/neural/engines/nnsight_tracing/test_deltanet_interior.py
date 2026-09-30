"""The Gated DeltaNet interior on the nnsight engine.

30 of the 40 target layers carry this mixer, and nothing in it is a module
boundary. As with the expert interior there is no cross-engine parity to
lean on, so correctness rests on identities and causal writes:

* **the conv split**: `deltanet_query`/`key`/`value` are exactly the three
  column blocks of `deltanet_qkv_conv` — same tensors, two addresses;
* **the projection chain**: the mixer's own output is
  `out_proj(deltanet_gated_out)`, invoked through the envoy;
* **the state**: one fire per 64-token chunk (the kernel's own loop count,
  never the config), and the state's causal signature — zeroing the state after
  chunk 0 leaves chunk 0's tokens bit-identical and moves later ones;
* **fire-axis discipline**: the state's position axis is the chunk index, so
  text anchors refuse, out-of-range fires refuse, and a write past the last
  fire refuses rather than silently never running.

Plus the ownership seams: the reference engine refuses the three faces only
this engine serves (`deltanet_query`, `deltanet_key`, `deltanet_state`) by
name, and routing lands such documents here unasked. The other
eight `deltanet_*` spellings are aliases of the `delta_*` names both engines
serve; the documents below still author them, which is the alias fold under
test — each reads through this engine's envoys or `.source` lines and equals
what the reference engine reads under the same name
(`test_parity_a3b_sweep.py`).
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.engine import component_capability, requires
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.rules.capability import refuse_shortfall
from causalab.protocol.schema import COMPONENTS, PROTOCOL_VERSION, parse_document
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import in_order, saved

from .test_parity_module_boundaries import ATOL, _data, _executor


pytestmark = pytest.mark.smoke

LAYER = 0  # DeltaNet on the fixture (layers 0-2; layer 3 is full attention)

DELTANET_COMPONENTS = tuple(c for c in COMPONENTS if c.startswith("deltanet_"))

#: > 64 tokens, so the kernel runs more than one chunk
LONG_TEXT = "the quick brown fox jumps over the lazy dog and runs far away " * 10
SHORT_TEXT = "the quick brown fox jumps"


def _read_doc(component: str, *, pos: object = -1, extra: dict | None = None) -> dict:
    doc = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": component, "layers": [LAYER]}},
            "reads": {"r": {"site": "tap", "pos": pos}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    for name, (site, npos) in (extra or {}).items():
        doc["method"]["sites"][f"{name}_site"] = site
        doc["method"]["reads"][name] = {"site": f"{name}_site", "pos": npos}
        doc["method"]["intervened_models"]["original"]["reads"].append(name)
        doc["method"]["save"].append(saved(name, "original", f"{name}.safetensors"))
    return doc


def _single_row_executor(trace_qwen, doc_raw: dict, text: str) -> TracePointExecutor:
    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    return TracePointExecutor(
        doc,
        trace_qwen,
        role_rows={"base": [{"input": text}]},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def _read(trace_qwen, component: str, **kw) -> torch.Tensor:
    return _executor(
        TracePointExecutor, _read_doc(component, **kw), trace_qwen, with_cf=False
    ).read_value("r")


# --------------------------------------------------------------------------- #
# the once-fired components: widths and identities
# --------------------------------------------------------------------------- #


def test_every_deltanet_component_reads_with_its_declared_width(trace_qwen):
    """📐 fixture dims: 4 key heads x 32, 8 value heads x 32 — so the fused
    projection is 512 wide and q/k live in the narrower key-head space."""
    info = trace_qwen.info
    key_dim = info.linear_num_key_heads * info.linear_key_head_dim
    value_dim = info.linear_num_value_heads * info.linear_value_head_dim
    expected = {
        "deltanet_qkv": 2 * key_dim + value_dim,
        "deltanet_qkv_conv": 2 * key_dim + value_dim,
        "deltanet_query": key_dim,
        "deltanet_key": key_dim,
        "deltanet_value": value_dim,
        "deltanet_beta": info.linear_num_value_heads,
        "deltanet_decay": info.linear_num_value_heads,
        "deltanet_gate": value_dim,
        "deltanet_core_out": value_dim,
        "deltanet_gated_out": value_dim,
    }
    for component, width in expected.items():
        value = _read(trace_qwen, component)
        assert tuple(value.shape) == (2, 1, width), component


def test_query_key_value_are_the_conv_splits_exactly(trace_qwen):
    """The identity that pins the split order [key | key | value] and that the
    q/k/v reshapes really are the conv's columns — same tensors, two
    addresses, difference 0.0."""
    info = trace_qwen.info
    key_dim = info.linear_num_key_heads * info.linear_key_head_dim
    doc = _read_doc(
        "deltanet_qkv_conv",
        extra={
            "q": ({"component": "deltanet_query", "layers": [LAYER]}, -1),
            "k": ({"component": "deltanet_key", "layers": [LAYER]}, -1),
            "v": ({"component": "deltanet_value", "layers": [LAYER]}, -1),
        },
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    conv = executor.read_value("r")
    torch.testing.assert_close(
        conv[..., :key_dim], executor.read_value("q"), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        conv[..., key_dim : 2 * key_dim], executor.read_value("k"), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        conv[..., 2 * key_dim :], executor.read_value("v"), atol=0.0, rtol=0.0
    )


def test_beta_is_a_sigmoid(trace_qwen):
    beta = _single_row_executor(
        trace_qwen, _read_doc("deltanet_beta", pos="all"), SHORT_TEXT
    ).read_value("r")
    assert float(beta.min()) > 0.0 and float(beta.max()) < 1.0


def test_the_mixer_output_is_the_projection_of_the_gated_out(trace_qwen):
    """`attention_output` (the module boundary, parity-proven) equals
    `out_proj(deltanet_gated_out)` — the envoy invokes its underlying module,
    the same mechanism attention_result's derivation uses."""
    doc = _read_doc(
        "deltanet_gated_out",
        extra={"out": ({"component": "attention_output", "layers": [LAYER]}, -1)},
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    gated = executor.read_value("r")
    out = executor.read_value("out")
    projected = trace_qwen.blocks[LAYER].linear_attn.out_proj(gated)
    torch.testing.assert_close(projected, out, atol=ATOL, rtol=0)


# --------------------------------------------------------------------------- #
# the state: per-chunk fires
# --------------------------------------------------------------------------- #


def _state_doc(pos: object, *, extra: dict | None = None) -> dict:
    return _read_doc("deltanet_state", pos=pos, extra=extra)


def test_the_state_fires_once_per_chunk(trace_qwen):
    """📐 the kernel pads to a 64 multiple and loops — the count is its own
    range op's, and on this prompt that is more than one chunk."""
    executor = _single_row_executor(trace_qwen, _state_doc("all"), LONG_TEXT)
    value = executor.read_value("r")
    info = trace_qwen.info
    tokens = len(trace_qwen.tokenizer(LONG_TEXT)["input_ids"])
    n_chunks = -(-tokens // 64)  # ceil
    assert n_chunks >= 2
    width = (
        info.linear_num_value_heads
        * info.linear_key_head_dim
        * info.linear_value_head_dim
    )
    assert tuple(value.shape) == (1, n_chunks, width)


def test_the_last_chunk_state_is_index_minus_one(trace_qwen):
    whole = _single_row_executor(trace_qwen, _state_doc("all"), LONG_TEXT).read_value(
        "r"
    )
    last = _single_row_executor(trace_qwen, _state_doc(-1), LONG_TEXT).read_value("r")
    torch.testing.assert_close(whole[:, -1:, :], last, atol=0.0, rtol=0.0)


def test_zeroing_the_state_after_chunk_0_moves_only_later_tokens(trace_qwen):
    """The state's causal signature, as a document: a clamp-to-zero write at
    chunk 0 leaves chunk 0's own tokens bit-identical (the state is applied
    *after* the chunk that produced it) and moves later tokens."""
    clean = _single_row_executor(
        trace_qwen,
        _read_doc("attention_output", pos="all"),
        LONG_TEXT,
    ).read_value("r")

    doc = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {
                "patched": {"input": "base", "reads": ["out"], "writes": ["zero"]}
            },
            "sites": {
                "tap": {"component": "deltanet_state", "layers": [LAYER]},
                "out_site": {"component": "attention_output", "layers": [LAYER]},
            },
            "reads": {"out": {"site": "out_site", "pos": "all"}},
            "writes": {
                "zero": {
                    "site": "tap",
                    "pos": 0,
                    "do": {"clamp": {"lo": 0.0, "hi": 0.0}},
                }
            },
            "save": [saved("out", "patched", "o.safetensors")],
        },
    }
    patched = _single_row_executor(trace_qwen, doc, LONG_TEXT).read_value("out")
    changed = (patched != clean).any(-1)[0]
    assert int(changed[:64].sum()) == 0, "chunk 0's own tokens must be untouched"
    assert int(changed[64:].sum()) > 0, "later tokens must move"


def test_a_write_past_the_last_fire_is_refused_not_skipped(trace_qwen):
    """📐 A tracer.iter body bound past the last fire never runs (nnsight
    keeps the fires it reached and warns), so the executor must turn the miss
    into a refusal rather than return values from a write that never landed."""
    doc = _state_doc("all")
    doc["method"]["writes"] = {
        "zero": {"site": "tap", "pos": 99, "do": {"clamp": {"lo": 0.0, "hi": 0.0}}}
    }
    # `r` moves onto the written model: the un-intervened one, read by nobody, goes
    doc["method"]["intervened_models"] = {
        "patched": {"input": "base", "reads": ["r"], "writes": ["zero"]}
    }
    doc["method"]["save"][0]["model"] = "patched"
    executor = _single_row_executor(trace_qwen, doc, LONG_TEXT)
    with pytest.raises(ProtocolError, match="fire"):
        executor.read_value("r")


def test_an_anchored_position_on_the_state_is_refused(trace_qwen):
    doc = _state_doc({"span": [0, 2]})
    executor = _single_row_executor(trace_qwen, doc, LONG_TEXT)
    with pytest.raises(ProtocolError, match="chunk index"):
        executor.read_value("r")


def test_an_out_of_range_chunk_read_is_refused(trace_qwen):
    executor = _single_row_executor(trace_qwen, _state_doc(99), LONG_TEXT)
    with pytest.raises(ProtocolError, match="fired"):
        executor.read_value("r")


# --------------------------------------------------------------------------- #
# writes on the once-fired components
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component", ["deltanet_value", "deltanet_gate", "deltanet_gated_out"]
)
def test_a_swap_moves_the_logits_and_a_self_swap_does_not(trace_qwen, component):
    def swap_doc() -> dict:
        return {
            "header": {"protocol_version": PROTOCOL_VERSION},
            "model": {"key": "test", "revision": "main"},
            "data": _data(with_cf=True),
            "method": {
                "intervened_models": {
                    "original_counterfactual": {
                        "input": "counterfactual",
                        "reads": ["v_cf"],
                    },
                    "patched": {
                        "input": "base",
                        "reads": ["logits"],
                        "writes": ["patch"],
                    },
                },
                "sites": {
                    "tap": {"component": component, "layers": [LAYER]},
                    "head": {"component": "lm_head"},
                },
                "reads": {
                    "v_cf": {"site": "tap", "pos": -1},
                    "logits": {"site": "head", "pos": -1},
                },
                "writes": {"patch": {"site": "tap", "pos": -1, "do": {"swap": "v_cf"}}},
                "save": [saved("logits", "patched", "l.safetensors")],
            },
        }

    clean_doc = _read_doc("deltanet_qkv")
    clean_doc["method"]["sites"]["tap"] = {"component": "lm_head"}
    clean = _executor(
        TracePointExecutor, clean_doc, trace_qwen, with_cf=False
    ).read_value("r")

    moved = _executor(TracePointExecutor, swap_doc(), trace_qwen, with_cf=True)
    assert float((moved.dense_value("logits") - clean).abs().max()) > 1e-5, component

    self_swap = swap_doc()
    # the operand read on the network on base itself: the swap writes back
    # what was there (the model keeps its name; names are free, §2.9)
    self_swap["method"]["intervened_models"]["original_counterfactual"]["input"] = (
        "base"
    )
    same = _executor(TracePointExecutor, self_swap, trace_qwen, with_cf=True)
    assert float((same.dense_value("logits") - clean).abs().max()) == 0.0, component


# --------------------------------------------------------------------------- #
# ownership and streams
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", DELTANET_COMPONENTS)
def test_the_reference_engine_refuses_by_name(hooks_qwen, component):
    doc = _read_doc(component)
    if component == "deltanet_state":
        doc["method"]["reads"]["r"]["pos"] = "all"
    with pytest.raises(ProtocolError, match="nnsight engine"):
        _executor(PointExecutor, doc, hooks_qwen, with_cf=False).run_all()


def test_this_engine_serves_deltanet_documents_the_reference_refuses():
    """Routing between engines is retired: the user pins
    `--engine nnsight`, and the capability check is what says the reference
    engine cannot serve the document — by the generated component entry."""
    doc = parse_document(in_order(_read_doc("deltanet_state", pos="all")))
    refuse_shortfall(requires(doc), NnsightEngine().effective_capabilities)
    with pytest.raises(ValidationError, match="component:deltanet_state"):
        refuse_shortfall(requires(doc), PytorchHooksEngine().effective_capabilities)
    assert component_capability("deltanet_state") not in (
        PytorchHooksEngine().effective_capabilities
    )


def test_deltanet_at_a_full_attention_layer_refuses_architecturally(trace_qwen):
    doc = _read_doc("deltanet_state", pos="all")
    doc["method"]["sites"]["tap"]["layers"] = 3  # full attention on this fixture
    with pytest.raises(ProtocolError, match="Gated DeltaNet mixer"):
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).run_all()
