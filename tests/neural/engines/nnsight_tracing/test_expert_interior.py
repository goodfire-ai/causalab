"""The per-expert MoE interior on the nnsight engine.

No cross-engine parity exists for these — the reference engine has no
mechanism for tensors inside a fused forward — so the correctness story rests
on three legs instead:

* **two implementations of the same math**: the grouped_mm kernel this engine
  serves from, against transformers' own eager per-expert loop, reconstructed
  through ``tracer.iter`` (a cross-check of the two, as a test);
* **identities**: the slot-sum of ``expert_output · router_scores`` is
  ``routed_output`` exactly (the registry's pre-routing-weight
  identity); the activation is ``act(gate)`` exactly; the permutation is a
  permutation;
* **causal writes**: a swap moves the logits, a same-value swap moves nothing,
  a written slot reads back written.

Plus the ownership seam that remains once the reference engine serves the
four slot components through its dispatch wrapper:
``expert_permutation`` is still this engine's alone, and routing knows it.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.engine import component_capability, requires
from causalab.protocol.rules.capability import refuse_shortfall
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.schema import PROTOCOL_VERSION, parse_document

from tests.protocol._docs import in_order, saved

from .test_parity_module_boundaries import ATOL, _data, _executor


pytestmark = pytest.mark.smoke

LAYER = 0  # every fixture layer carries the sparse-MoE block

EXPERT_COMPONENTS = (
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_neuron_output",
    "expert_permutation",
    "expert_output",
)


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
    for name, site in (extra or {}).items():
        doc["method"]["sites"][f"{name}_site"] = site
        doc["method"]["reads"][name] = {"site": f"{name}_site", "pos": pos}
        doc["method"]["intervened_models"]["original"]["reads"].append(name)
        doc["method"]["save"].append(saved(name, "original", f"{name}.safetensors"))
    return doc


def _swap_doc(component: str) -> dict:
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


def _read(trace_qwen, component: str, **kw) -> torch.Tensor:
    return _executor(
        TracePointExecutor, _read_doc(component, **kw), trace_qwen, with_cf=False
    ).read_value("r")


def _single_row_read(trace_qwen, component: str, text: str) -> torch.Tensor:
    """A one-example pos:"all" read — dense, so whole-frame identities can
    reshape it (the two-row harness's uneven lengths would make it ragged)."""
    from causalab.protocol.schema import parse_document
    from causalab.protocol.rules.document import validate_document

    doc = parse_document(in_order(_read_doc(component, pos="all")))
    validate_document(doc, engine_is_local=True)
    executor = TracePointExecutor(
        doc,
        trace_qwen,
        role_rows={"base": [{"input": text}]},
        role_fields={"base": "input"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )
    return executor.read_value("r")


# --------------------------------------------------------------------------- #
# reads: shapes and identities
# --------------------------------------------------------------------------- #


def test_the_interior_reads_with_the_declared_widths(trace_qwen):
    info = trace_qwen.info
    k = info.num_experts_per_tok
    for component, width in (
        ("expert_gate_proj", k * info.moe_intermediate_size),
        ("expert_up_proj", k * info.moe_intermediate_size),
        ("expert_activation", k * info.moe_intermediate_size),
        ("expert_neuron_output", k * info.moe_intermediate_size),
        ("expert_output", k * info.hidden_size),
        ("expert_permutation", k),
    ):
        value = _read(trace_qwen, component)
        assert tuple(value.shape) == (2, 1, width), component
        if component == "expert_permutation":
            assert not value.dtype.is_floating_point


def test_the_activation_is_act_of_gate_exactly(trace_qwen):
    """The identity that says the three taps share one fused capture and its
    gate: ``expert_activation`` is ``act_fn(gate)`` alone (the registry's
    semantics, the llama ``mlp_activation`` precedent) — same rows,
    same order, before the ``· up`` multiply."""
    doc = _read_doc(
        "expert_gate_proj",
        extra={
            "act": {"component": "expert_activation", "layers": [LAYER]},
        },
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    gate, act = executor.read_value("r"), executor.read_value("act")
    torch.testing.assert_close(torch.nn.functional.silu(gate), act, atol=0.0, rtol=0.0)


def test_expert_output_weighted_sums_to_routed_output(trace_qwen):
    """The registry identity, on this engine: ``routed_output == Σ_slot
    expert_output · router_scores`` — ``expert_output`` is the down-projection
    output BEFORE the routing weight, so the scores re-enter here —
    pinned against the module-boundary tap, which the parity suite already
    proves against the reference engine."""
    info = trace_qwen.info
    doc = _read_doc(
        "expert_output",
        extra={
            "routed": {"component": "routed_output", "layers": [LAYER]},
            "scores": {"component": "router_scores", "layers": [LAYER]},
        },
    )
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    per_slot = executor.read_value("r")
    routed = executor.read_value("routed")
    scores = executor.read_value("scores")
    weighted = per_slot.reshape(
        *per_slot.shape[:-1], info.num_experts_per_tok, info.hidden_size
    ) * scores.unsqueeze(-1)
    torch.testing.assert_close(weighted.sum(-2), routed, atol=ATOL, rtol=0)


def test_the_permutation_is_a_permutation(trace_qwen):
    """Read whole: every (token, slot) pair's sorted-row index, each row index
    used exactly once."""
    value = _single_row_read(
        trace_qwen, "expert_permutation", "the quick brown fox jumps"
    )
    flat = value.reshape(-1)
    assert torch.equal(
        torch.sort(flat).values, torch.arange(flat.numel(), dtype=flat.dtype)
    )


def test_grouped_and_eager_implementations_agree(trace_qwen):
    """The cross-check: transformers' own eager per-expert loop — a wholly
    independent implementation, one Python iteration per hit expert — rebuilt
    into the same (token, slot) frame through ``tracer.iter``, against the
    grouped_mm tensor this engine serves."""
    import nnsight

    info = trace_qwen.info
    k, hidden = info.num_experts_per_tok, info.hidden_size
    text = "the quick brown fox jumps"
    served = _single_row_read(trace_qwen, "expert_output", text)  # (1, s, k·hidden)
    rows = served.shape[0] * served.shape[1]
    served = served.reshape(rows, k, hidden)

    batch = trace_qwen.tokenizer([text], return_tensors="pt", padding=True)
    experts = trace_qwen.blocks[LAYER].mlp.experts
    model = trace_qwen.model
    model.set_experts_implementation("eager")
    try:
        with torch.no_grad():
            with model.trace(dict(batch)) as tracer:
                loop = experts.source.experts_forward_1.source
                n_hit = len(loop.nonzero_0.output)
                per_expert = nnsight.save([])
                for _ in tracer.iter[:n_hit]:
                    top_k_pos, token_idx = loop.torch_where_0.output
                    # `_1` is the down-projection's output, BEFORE the routing
                    # weight — the semantics `expert_output` names
                    # (`_2` is the weighted value one line later)
                    per_expert.append(
                        (top_k_pos, token_idx, loop.current_hidden_states_1.output)
                    )
    finally:
        model.set_experts_implementation("grouped_mm")

    rebuilt = torch.zeros(rows, k, hidden)
    for top_k_pos, token_idx, unweighted in per_expert:
        rebuilt[token_idx, top_k_pos] = unweighted.to(rebuilt.dtype)
    torch.testing.assert_close(rebuilt, served, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component",
    ["expert_gate_proj", "expert_activation", "expert_neuron_output", "expert_output"],
)
def test_a_swap_moves_the_logits_and_a_self_swap_does_not(trace_qwen, component):
    def logits(doc: dict) -> torch.Tensor:
        return _executor(TracePointExecutor, doc, trace_qwen, with_cf=True).dense_value(
            "logits"
        )

    clean_doc = _read_doc("expert_output")
    clean_doc["method"]["sites"]["tap"] = {"component": "lm_head"}
    clean = _executor(
        TracePointExecutor, clean_doc, trace_qwen, with_cf=False
    ).read_value("r")

    moved = float((logits(_swap_doc(component)) - clean).abs().max())
    assert moved > 1e-5, f"{component}: the swap landed nowhere"

    self_swap = _swap_doc(component)
    # the operand read on the network on base itself: the swap writes back
    # what was there (the model keeps its name; names are free, §2.9)
    self_swap["method"]["intervened_models"]["original_counterfactual"]["input"] = (
        "base"
    )
    unmoved = float((logits(self_swap) - clean).abs().max())
    assert unmoved == 0.0, f"{component}: a same-value swap must be the identity"


def test_a_written_slot_reads_back_written(trace_qwen):
    doc = _swap_doc("expert_gate_proj")
    doc["method"]["reads"]["obs"] = {"site": "tap", "pos": -1}
    doc["method"]["intervened_models"]["patched"]["reads"].append("obs")
    doc["method"]["save"].append(saved("obs", "patched", "o.safetensors"))
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    src, obs = executor.read_value("v_cf"), executor.read_value("obs")
    assert float((obs - src).abs().max()) == 0.0


def test_the_permutation_refuses_writes_as_kernel_bookkeeping(trace_qwen):
    doc = _swap_doc("expert_permutation")
    with pytest.raises(ProtocolError, match="ordering of token-slot rows from routing"):
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=True).run_all()


# --------------------------------------------------------------------------- #
# ownership: the reference engine serves the four slot components (its
# dispatch wrapper), so only the kernel's own bookkeeping is left to
# refuse by name — and routing still knows this engine owns it.
# --------------------------------------------------------------------------- #


def test_the_reference_engine_refuses_the_permutation_by_name(hooks_qwen):
    doc = _read_doc("expert_permutation")
    with pytest.raises(ProtocolError, match="nnsight engine"):
        _executor(PointExecutor, doc, hooks_qwen, with_cf=False).run_all()


def test_this_engine_serves_the_permutation_the_reference_refuses():
    """Routing between engines is retired: the user pins
    `--engine nnsight`, and the capability check says the reference engine
    cannot serve the document."""
    doc = parse_document(in_order(_read_doc("expert_permutation")))
    refuse_shortfall(requires(doc), NnsightEngine().effective_capabilities)
    assert component_capability("expert_permutation") not in (
        PytorchHooksEngine().effective_capabilities
    )


def test_the_generated_refusal_names_the_missing_component():
    doc = parse_document(in_order(_read_doc("expert_permutation")))
    with pytest.raises(ValidationError, match="component:expert_permutation"):
        refuse_shortfall(requires(doc), PytorchHooksEngine().effective_capabilities)
