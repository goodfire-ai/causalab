"""The parity suite: the same documents through both
engines, asserting the answers agree.

This is simultaneously the nnsight engine's correctness proof for the
module-boundary vocabulary and a numerical oracle between the two engines —
one artifact, two uses. Reads must agree to fp32-eager-CPU
tolerance, write effects on the logits must agree, and refusals must be the
*same* refusal (code and component named), because the policy tables are
single-homed.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.engine import component_capability, requires
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.rules.capability import refuse_shortfall
from causalab.protocol.schema import PROTOCOL_VERSION, parse_document
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import in_order, saved


pytestmark = pytest.mark.smoke

BASE_TEXTS = ["the quick brown fox jumps", "a small red hen sits still"]
CF_TEXTS = ["a slow green turtle sleeps", "the big blue whale dives deep"]

#: Reads agree to this tolerance: fp32, eager attention, CPU on both sides —
#: the operations are the same kernels in a different order of capture, so
#: anything beyond float-noise scale is an executor bug.
ATOL = 1e-5


# --------------------------------------------------------------------------- #
# drive: the same document through either executor
# --------------------------------------------------------------------------- #


def _executor(executor_cls, doc_raw, bundle, *, with_cf: bool):
    doc = parse_document(in_order(doc_raw))
    validate_document(doc, engine_is_local=True)
    rows = [
        {"input": base, "counterfactual_inputs": [cf]}
        for base, cf in zip(BASE_TEXTS, CF_TEXTS)
    ]
    role_rows = {"base": rows}
    role_fields = {"base": "input"}
    if with_cf:
        role_rows["counterfactual"] = rows
        role_fields["counterfactual"] = "counterfactual_inputs[0]"
    return executor_cls(
        doc,
        bundle,
        role_rows=role_rows,
        role_fields=role_fields,
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def _data(with_cf: bool) -> dict:
    data: dict = {"base": {"dataset": "inline", "field": "input"}}
    if with_cf:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
    return data


def _read_doc(component: str, layer: int | None, head: int | None = None) -> dict:
    site: dict = {"component": component}
    if layer is not None:
        site["layers"] = layer
    if head is not None:
        site["head"] = head
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": {"site": "tap", "pos": -1}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _interchange_doc(component: str, layer: int | None) -> dict:
    """base_doc's shape: read the site on the counterfactual, swap it into
    the base forward, read the patched logits."""
    site: dict = {"component": component}
    if layer is not None:
        site["layers"] = layer
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
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {"tap": site, "head": {"component": "lm_head"}},
            "reads": {
                "v_cf": {"site": "tap", "pos": -1},
                "logits": {"site": "head", "pos": -1},
            },
            "writes": {"patch": {"site": "tap", "pos": -1, "do": {"swap": "v_cf"}}},
            "save": [saved("logits", "patched", "l.safetensors")],
        },
    }


def _block_mid_with_later_read_doc(later_component: str) -> dict:
    """A mid write followed by a same-layer interior read and output read."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "intervened_models": {
            "original_base": {"input": "base", "reads": ["v_mid_base"]},
            "original_counterfactual": {
                "input": "counterfactual",
                "reads": ["v_mid_cf"],
            },
            "patched": {
                "input": "base",
                "reads": ["r_later", "r_out"],
                "writes": ["mid_swap"],
            },
        },
        "sites": {
            "mid": {"component": "block_mid", "layers": 1},
            "later": {"component": later_component, "layers": 1},
            "out": {"component": "block_output", "layers": 1},
        },
        "reads": {
            "v_mid_base": {"site": "mid", "pos": -1},
            "v_mid_cf": {"site": "mid", "pos": -1},
            "r_later": {"site": "later", "pos": -1},
            "r_out": {"site": "out", "pos": -1},
        },
        "writes": {
            "mid_swap": {
                "site": "mid",
                "pos": -1,
                "do": {"swap": "v_mid_cf"},
            }
        },
        "save": [
            saved(name, model, f"{name}.safetensors")
            for name, model in (
                ("v_mid_base", "original_base"),
                ("v_mid_cf", "original_counterfactual"),
                ("r_later", "patched"),
                ("r_out", "patched"),
            )
        ],
    }


def _mixed_block_writes_doc() -> dict:
    """Absolute precedence and additive composition after mid writeback."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "intervened_models": {
            "original_base": {"input": "base", "reads": ["v_mid_base"]},
            "original_counterfactual": {
                "input": "counterfactual",
                "reads": ["v_mid_cf", "v_out_cf"],
            },
            "mid_then_out": {
                "input": "base",
                "reads": ["r_mid_then_out"],
                "writes": ["mid_swap", "out_swap"],
            },
            "out_then_mid": {
                "input": "base",
                "reads": ["r_out_then_mid"],
                "writes": ["out_swap", "mid_swap"],
            },
            "mid_plus_out": {
                "input": "base",
                "reads": ["r_mid_plus_out"],
                "writes": ["mid_swap", "out_add"],
            },
        },
        "sites": {
            "mid": {"component": "block_mid", "layers": 1},
            "out": {"component": "block_output", "layers": 1},
        },
        "reads": {
            "v_mid_base": {"site": "mid", "pos": -1},
            "v_mid_cf": {"site": "mid", "pos": -1},
            "v_out_cf": {"site": "out", "pos": -1},
            "r_mid_then_out": {"site": "out", "pos": -1},
            "r_out_then_mid": {"site": "out", "pos": -1},
            "r_mid_plus_out": {"site": "out", "pos": -1},
        },
        "writes": {
            "mid_swap": {
                "site": "mid",
                "pos": -1,
                "do": {"swap": "v_mid_cf"},
            },
            "out_swap": {
                "site": "out",
                "pos": -1,
                "do": {"swap": "v_out_cf"},
            },
            "out_add": {
                "site": "out",
                "pos": -1,
                "do": {"add_scaled": {"op": "v_out_cf", "alpha": 0.25}},
            },
        },
        "save": [
            saved(name, model, f"{name}.safetensors")
            for name, model in (
                ("v_mid_base", "original_base"),
                ("v_mid_cf", "original_counterfactual"),
                ("v_out_cf", "original_counterfactual"),
                ("r_mid_then_out", "mid_then_out"),
                ("r_out_then_mid", "out_then_mid"),
                ("r_mid_plus_out", "mid_plus_out"),
            )
        ],
    }


def _assert_same(a, b, what: str) -> None:
    assert a.shape == b.shape, f"{what}: {tuple(a.shape)} != {tuple(b.shape)}"
    if not a.dtype.is_floating_point:
        assert torch.equal(a, b), f"{what}: integer values differ"
        return
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=ATOL, rtol=0), (
        f"{what}: max abs diff {diff:.3e} exceeds {ATOL}"
    )


# --------------------------------------------------------------------------- #
# reads: every module-boundary component, both families
# --------------------------------------------------------------------------- #

LLAMA_READS = [
    ("input_ids", None, None),
    ("embeddings", None, None),
    ("block_input", 1, None),
    ("attention_input_norm", 1, None),
    ("attention_output", 1, None),
    ("attention_value", 1, None),
    ("attention_value", 1, 1),  # per-head slice of the o-projection input
    ("block_mid", 1, None),
    ("mlp_input_norm", 1, None),
    ("mlp_input", 1, None),
    ("mlp_activation", 1, None),
    ("mlp_output", 1, None),
    ("block_output", 1, None),
    ("ln_final", None, None),
    ("lm_head", None, None),
]

#: The hybrid/MoE surface, on the target architecture in miniature: layer 0
#: is Gated DeltaNet, layer 3 full attention, sparse MoE in every layer.
QWEN_READS = [
    ("block_output", 0, None),  # a DeltaNet layer's boundary
    ("attention_output", 0, None),  # the DeltaNet mixer's output
    ("attention_output", 3, None),  # the full-attention mixer's output
    ("router_logits", 0, None),
    ("router_scores", 0, None),
    ("expert_idx", 0, None),
    ("routed_output", 0, None),
    ("shared_expert_gate_proj", 0, None),
    ("shared_expert_up_proj", 0, None),
    ("shared_expert_activation", 0, None),
    ("shared_expert_output", 0, None),
    ("shared_expert_gate", 0, None),
]


@pytest.mark.parametrize("component,layer,head", LLAMA_READS)
def test_llama_read_parity(hooks_llama, trace_llama, component, layer, head):
    doc = _read_doc(component, layer, head)
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"read {component!r} (layer {layer}, head {head})")


@pytest.mark.parametrize("component,layer,head", QWEN_READS)
def test_qwen_read_parity(hooks_qwen, trace_qwen, component, layer, head):
    doc = _read_doc(component, layer, head)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"read {component!r} (layer {layer})")


# --------------------------------------------------------------------------- #
# writes: the interchange's effect on the logits agrees
# --------------------------------------------------------------------------- #

LLAMA_WRITES = [
    ("embeddings", None),
    ("block_input", 1),
    ("attention_output", 1),
    ("attention_value", 1),
    ("block_mid", 1),
    ("mlp_activation", 1),  # P4: the dense act_fn output, absent from the MoE fixture
    ("mlp_output", 1),
    ("block_output", 1),
]

QWEN_WRITES = [
    ("block_output", 0),  # a write on a DeltaNet layer's boundary
    ("attention_output", 3),
    ("router_scores", 0),
    ("expert_idx", 0),
    ("routed_output", 0),
    ("shared_expert_output", 0),
]


def _unpatched_logits(hooks_bundle, trace_bundle) -> dict:
    """The clean last-position logits from each engine, which every write
    case below must move away from. Without this check "both engines agree"
    also holds when neither write landed."""
    doc = _read_doc("lm_head", None)
    return {
        "hooks": _executor(PointExecutor, doc, hooks_bundle, with_cf=False).read_value(
            "r"
        ),
        "trace": _executor(
            TracePointExecutor, doc, trace_bundle, with_cf=False
        ).read_value("r"),
    }


@pytest.fixture(scope="module")
def llama_unpatched_logits(hooks_llama, trace_llama) -> dict:
    return _unpatched_logits(hooks_llama, trace_llama)


@pytest.fixture(scope="module")
def qwen_unpatched_logits(hooks_qwen, trace_qwen) -> dict:
    return _unpatched_logits(hooks_qwen, trace_qwen)


def _write_parity(component, layer, hooks_bundle, trace_bundle, unpatched) -> None:
    doc = _interchange_doc(component, layer)
    hooked = _executor(PointExecutor, doc, hooks_bundle, with_cf=True).dense_value(
        "logits"
    )
    traced = _executor(TracePointExecutor, doc, trace_bundle, with_cf=True).dense_value(
        "logits"
    )
    _assert_same(hooked, traced, f"patched logits after a swap at {component!r}")
    assert not torch.allclose(hooked, unpatched["hooks"], atol=ATOL), (
        f"pytorch_hooks: the interchange at {component!r} left the logits unchanged"
    )
    assert not torch.allclose(traced, unpatched["trace"], atol=ATOL), (
        f"nnsight: the interchange at {component!r} left the logits unchanged"
    )


@pytest.mark.parametrize("component,layer", LLAMA_WRITES)
def test_llama_write_parity(
    hooks_llama, trace_llama, llama_unpatched_logits, component, layer
):
    _write_parity(component, layer, hooks_llama, trace_llama, llama_unpatched_logits)


@pytest.mark.parametrize("component,layer", QWEN_WRITES)
def test_qwen_write_parity(
    hooks_qwen, trace_qwen, qwen_unpatched_logits, component, layer
):
    _write_parity(component, layer, hooks_qwen, trace_qwen, qwen_unpatched_logits)


@pytest.mark.parametrize(
    "hooks_name,trace_name,later_component",
    (
        ("hooks_llama", "trace_llama", "mlp_output"),
        ("hooks_qwen", "trace_qwen", "router_scores"),
    ),
)
def test_block_mid_write_allows_later_same_layer_reads(
    hooks_name, trace_name, later_component, request
):
    """Deferred writeback does not reach past later module or source reads."""
    doc = _block_mid_with_later_read_doc(later_component)
    hooked = _executor(
        PointExecutor, doc, request.getfixturevalue(hooks_name), with_cf=True
    )
    traced = _executor(
        TracePointExecutor, doc, request.getfixturevalue(trace_name), with_cf=True
    )
    base_mid = hooked.read_value("v_mid_base")
    cf_mid = hooked.read_value("v_mid_cf")
    assert not torch.allclose(base_mid, cf_mid, atol=ATOL), (
        "the counterfactual block_mid swap operand equals the clean value"
    )
    _assert_same(base_mid, traced.read_value("v_mid_base"), "clean block_mid")
    _assert_same(cf_mid, traced.read_value("v_mid_cf"), "counterfactual block_mid")
    for name in ("r_later", "r_out"):
        _assert_same(
            hooked.read_value(name),
            traced.read_value(name),
            f"block_mid write followed by {name}",
        )


def test_mixed_block_write_precedence_agrees_across_engines(hooks_llama, trace_llama):
    """Output swaps supersede, while output additions retain, mid writeback."""
    doc = _mixed_block_writes_doc()
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=True)
    expected = hooked.read_value("v_out_cf")
    base_mid = hooked.read_value("v_mid_base")
    cf_mid = hooked.read_value("v_mid_cf")
    assert not torch.allclose(base_mid, cf_mid, atol=ATOL), (
        "the counterfactual block_mid swap operand equals the clean value"
    )
    _assert_same(base_mid, traced.read_value("v_mid_base"), "clean block_mid")
    _assert_same(cf_mid, traced.read_value("v_mid_cf"), "mid swap operand")
    _assert_same(expected, traced.read_value("v_out_cf"), "output swap operand")
    for name in ("r_mid_then_out", "r_out_then_mid"):
        hooked_value = hooked.read_value(name)
        traced_value = traced.read_value(name)
        _assert_same(hooked_value, traced_value, f"mixed writes for {name}")
        _assert_same(hooked_value, expected, f"absolute output precedence for {name}")
    _assert_same(
        hooked.read_value("r_mid_plus_out"),
        traced.read_value("r_mid_plus_out"),
        "block_mid writeback followed by additive block_output write",
    )


# --------------------------------------------------------------------------- #
# refusals: the same policy, the same words
# --------------------------------------------------------------------------- #


def _refusal(executor_cls, doc, bundle) -> str:
    with pytest.raises(ProtocolError) as excinfo:
        executor = _executor(executor_cls, doc, bundle, with_cf=True)
        executor.run_all()
    return str(excinfo.value)


def test_read_only_refusal_is_identical(hooks_qwen, trace_qwen):
    doc = _interchange_doc("router_logits", 0)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_swap_only_refusal_is_identical(hooks_qwen, trace_qwen):
    doc = _interchange_doc("expert_idx", 0)
    doc["method"]["writes"]["patch"]["do"] = {
        "add_scaled": {"value": "v_cf", "scale": 2.0}
    }
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_wrong_stream_refusal_is_identical(hooks_qwen, trace_qwen):
    doc = _read_doc("attention_output", 0)
    doc["method"]["sites"]["tap"]["stream"] = "full_attention"  # layer 0 is DeltaNet
    with pytest.raises(ProtocolError) as hooks_err:
        _executor(PointExecutor, doc, hooks_qwen, with_cf=False).run_all()
    with pytest.raises(ProtocolError) as trace_err:
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).run_all()
    assert str(hooks_err.value) == str(trace_err.value)


# --------------------------------------------------------------------------- #
# capabilities: what this engine declares, the check accepts
# --------------------------------------------------------------------------- #


def test_attention_probs_is_served_here():
    """The pattern (and the whole attention interior) is
    served here through the `.source` address table, so a document naming it
    passes the capability check against this engine (routing between engines
    is retired; the check is what remains)."""
    doc = parse_document(in_order(_read_doc("attention_probs", 3)))
    refuse_shortfall(requires(doc), NnsightEngine().effective_capabilities)
    assert component_capability("attention_probs") in (
        NnsightEngine().effective_capabilities
    )
    assert "writable_attention_probs" in NnsightEngine().capabilities


def test_a_generated_read_is_served_and_agrees_with_the_reference_engine(
    hooks_llama, trace_llama
):
    """A continuation read decodes through one generate trace here, and the value
    matches the reference engine's hand-rolled greedy decode."""
    doc = _read_doc("block_output", 1)
    doc["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 4}, "index": -1}
    }
    doc["method"]["reads"]["r"]["pos"] = "tail"
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, "a generated block_output read")
