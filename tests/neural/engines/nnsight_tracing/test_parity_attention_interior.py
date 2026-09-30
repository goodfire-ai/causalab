"""The attention interior on the nnsight engine.

The same treatment the module-boundary vocabulary got, extended to the
attention components: the same documents through both engines, agreeing to
fp32-eager-CPU tolerance. Two genuinely independent implementations — the
reference engine's ``TorchFunctionMode`` softmax tap vs this engine's
``.source`` address navigation — agreeing is the strongest check this file
has: a wrong address (``attn_weights_0``, the pre-mask tensor, say) produces
plausible numbers of the right shape, and only the comparison catches it.

Plus what parity alone cannot pin: the identity checks ported from the
verification probes (``softmax(scores) == pattern`` exactly; rows sum to 1;
``z·σ(gate) == premix``), the causal writes (a targeted knockout moves the
logits, a uniform shift is a softmax-invariance no-op), the in-forward
ordering discipline the ``.source`` interiors demand, and the on-demand
implementation switch.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION

from tests.protocol._docs import UNWRITTEN, saved

from .test_parity_module_boundaries import (
    ATOL,
    _assert_same,
    _data,
    _executor,
    _refusal,
)

pytestmark = pytest.mark.smoke

#: layer 3 is the qwen fixture's one full-attention layer
LAYER = 3

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"


def _read_doc(component: str, *, pos: object = -1, head: int | None = None) -> dict:
    site: dict = {"component": component, "layers": [LAYER]}
    if head is not None:
        site["head"] = head
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": {"site": "tap", "pos": pos}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _write_doc(component: str, do: dict, *, pos: object = "all") -> dict:
    """Patch one interior site and read the last-position logits."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {"component": component, "layers": [LAYER]},
                "head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": pos},
                "logits": {"site": "head", "pos": -1},
            },
            "writes": {"patch": {"site": "tap", "pos": pos, "do": do}},
            "save": [saved("logits", "patched", "l.safetensors")],
        },
    }


def _without_operand(doc: dict) -> dict:
    """The write document with a literal or parameter operand: the
    counterfactual read goes, and with it the un-intervened model that was
    the only one to list it (a model nobody reads is refused, §2.9)."""
    del doc["method"]["reads"]["v_cf"]
    del doc["method"]["intervened_models"][UNWRITTEN]
    return doc


# --------------------------------------------------------------------------- #
# reads: the whole attention surface, both engines
# --------------------------------------------------------------------------- #

#: (component, pos, head). The two pattern-shaped components have no contract
#: form, so they are read whole; everything else reads a position like any
#: other tap. `attention_result` is the derived component — on this engine the
#: o-projection envoy is invoked directly (it calls its underlying module),
#: which this case is the proof of.
INTERIOR_READS = [
    ("attention_query", -1, None),
    ("attention_query", -1, 1),
    ("attention_key", -1, None),
    ("attention_scores", "all", None),
    ("attention_probs", "all", None),
    ("attention_z", -1, None),
    ("attention_z", -1, 1),
    ("attention_query_pre_rope", -1, None),
    ("attention_key_pre_rope", -1, None),
    ("attention_value_states", -1, None),
    ("attention_gate", -1, None),
    ("attention_premix", -1, 1),
    ("attention_result", -1, 1),
]


@pytest.mark.parametrize("component,pos,head", INTERIOR_READS)
def test_interior_read_parity(hooks_qwen, trace_qwen, component, pos, head):
    doc = _read_doc(component, pos=pos, head=head)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"read {component!r} (pos {pos}, head {head})")


@pytest.mark.parametrize(
    "component", ["attention_query", "attention_scores", "attention_z"]
)
def test_llama_interior_read_parity(hooks_llama, trace_llama, component):
    """The second family, so the addresses are not qwen-shaped by accident.
    (No gate and no q/k norms here — the table's ops are the ones both
    forwards share.)"""
    doc = _read_doc(component, pos="all" if component == "attention_scores" else -1)
    doc["method"]["sites"]["tap"]["layers"] = 1
    hooked = _executor(PointExecutor, doc, hooks_llama, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_llama, with_cf=False).read_value(
        "r"
    )
    _assert_same(hooked, traced, f"llama read {component!r}")


# --------------------------------------------------------------------------- #
# identities: the pins that say each address is where it claims to be
# --------------------------------------------------------------------------- #


def _traced_read(trace_qwen, component: str, **kw) -> torch.Tensor:
    return _executor(
        TracePointExecutor, _read_doc(component, **kw), trace_qwen, with_cf=False
    ).read_value("r")


def test_the_scores_softmax_to_the_pattern_exactly(trace_qwen):
    """📐 ``softmax(attn_weights_1) == attn_weights_2`` at 0.0 — the wrong
    pick (``attn_weights_0``, pre-mask) fails this before parity even runs."""
    scores = _traced_read(trace_qwen, "attention_scores", pos="all")
    probs = _traced_read(trace_qwen, "attention_probs", pos="all")
    torch.testing.assert_close(
        torch.softmax(scores.float(), dim=-1), probs.float(), atol=0.0, rtol=0.0
    )


def test_the_pattern_rows_sum_to_one(trace_qwen):
    probs = _traced_read(trace_qwen, "attention_probs", pos="all")
    torch.testing.assert_close(
        probs.float().sum(-1),
        torch.ones(probs.shape[:-1]),
        atol=1e-5,
        rtol=0,
    )


def test_z_times_gate_is_the_premix(trace_qwen):
    """The gated family's defining identity: the o-projection's input is
    ``z · σ(gate)``. All three taps are head-major in the contract, so the
    identity is elementwise there."""
    z = _traced_read(trace_qwen, "attention_z", pos=-1)
    gate = _traced_read(trace_qwen, "attention_gate", pos=-1)
    premix = _traced_read(trace_qwen, "attention_premix", pos=-1)
    torch.testing.assert_close(
        z.float() * torch.sigmoid(gate.float()), premix.float(), atol=ATOL, rtol=0
    )


# --------------------------------------------------------------------------- #
# writes: interchanges agree, and the causal checks are not vacuous
# --------------------------------------------------------------------------- #

#: pos "all" only where the tap has no contract form (the two pattern-shaped
#: components, edited whole); everywhere else -1 — the fixture's two rows
#: tokenize to different lengths, and ragged *writes* are refused (v1, the
#: same refusal on both engines).
INTERIOR_WRITES = [
    ("attention_query", -1),
    ("attention_key", -1),
    ("attention_scores", "all"),
    ("attention_probs", "all"),
    ("attention_z", -1),
    ("attention_value_states", -1),
    ("attention_gate", -1),
    ("attention_premix", -1),
]


@pytest.mark.parametrize("component,pos", INTERIOR_WRITES)
def test_interior_write_parity(hooks_qwen, trace_qwen, component, pos):
    doc = _write_doc(component, {"swap": "v_cf"}, pos=pos)
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    _assert_same(
        hooked.dense_value("logits"),
        traced.dense_value("logits"),
        f"patched logits after a swap at {component!r}",
    )


def _clean_logits(trace_qwen) -> torch.Tensor:
    doc = _read_doc("attention_z")  # placeholder site, replaced below
    doc["method"]["sites"]["tap"] = {"component": "lm_head"}
    return _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).read_value("r")


def _moved(trace_qwen, doc: dict, load_tensors=None) -> float:
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    if load_tensors is not None:
        executor.load_tensors = load_tensors
    clean = _clean_logits(trace_qwen)
    return float((executor.dense_value("logits") - clean).abs().max())


def test_an_interior_swap_moves_the_logits_at_all(trace_qwen):
    """Anti-vacuity for the write-parity cases: 'both engines agree' must not
    be satisfiable by 'neither write landed'."""
    assert _moved(trace_qwen, _write_doc("attention_scores", {"swap": "v_cf"})) > 1e-4


def test_a_targeted_knockout_on_the_scores_moves_the_logits(trace_qwen):
    """Attention knockout as arithmetic on the scores — the capability the
    component exists for, now on this engine. Head 0 blocked from attending
    to token 0, as a full-shape mask added upstream of the model's own
    softmax."""
    from tests.neural.engines.pytorch_hooks._drive import bundle_loader

    doc = _without_operand(
        _write_doc("attention_scores", {"add_scaled": {"op": "knock", "alpha": 1.0}})
    )
    doc["method"]["params"] = {"knock": {"file_path": "k.safetensors"}}
    mask = torch.zeros_like(_traced_read(trace_qwen, "attention_scores", pos="all"))
    mask[:, 0, :, 0] = -1e4
    assert (
        _moved(
            trace_qwen,
            doc,
            load_tensors=bundle_loader({"k.safetensors": {"value": mask}}),
        )
        > 1e-3
    )


def test_a_uniform_shift_of_the_scores_is_a_no_op(trace_qwen):
    """Softmax is shift-invariant along the axis it normalizes, so adding the
    same constant to every score changes nothing — ported as a pin so a
    knockout recipe stays targeted."""
    doc = _without_operand(
        _write_doc("attention_scores", {"add_scaled": {"op": -10000.0, "alpha": 1.0}})
    )
    assert _moved(trace_qwen, doc) < 1e-3


def test_a_read_of_a_written_slot_sees_the_written_value(trace_qwen):
    """Write-before-read at one address, within one trace: the read observes
    the written value (difference 0.0), matching the reference engine's
    hook-registration order — the two engines have to agree here or the same
    document would mean different things."""
    doc = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["src"]},
                "patched": {"input": "base", "reads": ["obs"], "writes": ["p"]},
            },
            "sites": {"tap": {"component": "attention_query", "layers": [LAYER]}},
            "reads": {
                "src": {"site": "tap", "pos": -1},
                "obs": {"site": "tap", "pos": -1},
            },
            "writes": {"p": {"site": "tap", "pos": -1, "do": {"swap": "src"}}},
            "save": [saved("obs", "patched", "o.safetensors")],
        },
    }
    executor = _executor(TracePointExecutor, doc, trace_qwen, with_cf=True)
    src, obs = executor.read_value("src"), executor.read_value("obs")
    assert float((obs - src).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# ordering: the renumbered rank band IS the in-forward op order
# --------------------------------------------------------------------------- #


def test_several_interior_reads_share_one_trace_in_forward_order(
    hooks_qwen, trace_qwen
):
    """`.source` ops refuse out-of-order requests (OutOfOrderError), so this
    doc — three interior taps plus a boundary tap in one group — passes only
    if the (layer, COMPONENT_RANK) sort key already walks the forward. Pinned
    as a test rather than assumed."""
    doc = _read_doc("attention_query", pos=-1)
    doc["method"]["sites"]["z_site"] = {"component": "attention_z", "layers": [LAYER]}
    doc["method"]["sites"]["scores_site"] = {
        "component": "attention_scores",
        "layers": [LAYER],
    }
    doc["method"]["sites"]["out_site"] = {
        "component": "attention_output",
        "layers": [LAYER],
    }
    for name, site in (
        ("r_z", "z_site"),
        ("r_scores", "scores_site"),
        ("r_out", "out_site"),
    ):
        doc["method"]["reads"][name] = {
            "site": site,
            "pos": "all" if name == "r_scores" else -1,
        }
        doc["method"]["intervened_models"]["original"]["reads"].append(name)
        doc["method"]["save"].append(saved(name, "original", f"{name}.safetensors"))
    hooked = _executor(PointExecutor, doc, hooks_qwen, with_cf=False)
    traced = _executor(TracePointExecutor, doc, trace_qwen, with_cf=False)
    for name in ("r", "r_z", "r_scores", "r_out"):
        _assert_same(
            hooked.read_value(name), traced.read_value(name), f"grouped read {name!r}"
        )


# --------------------------------------------------------------------------- #
# refusals: same policy, same words
# --------------------------------------------------------------------------- #


def test_a_delta_on_the_pattern_refuses_identically(hooks_qwen, trace_qwen):
    doc = _without_operand(
        _write_doc("attention_probs", {"add_scaled": {"op": -1.0, "alpha": 1.0}})
    )
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_a_positioned_read_of_the_scores_refuses_identically(hooks_qwen, trace_qwen):
    doc = _read_doc("attention_scores", pos=-1)
    assert _refusal(PointExecutor, doc, hooks_qwen) == _refusal(
        TracePointExecutor, doc, trace_qwen
    )


def test_the_interior_at_a_deltanet_layer_refuses_architecturally(trace_qwen):
    doc = _read_doc("attention_scores", pos="all")
    doc["method"]["sites"]["tap"]["layers"] = 0  # DeltaNet on this fixture
    with pytest.raises(ProtocolError, match="full-attention mixer"):
        _executor(TracePointExecutor, doc, trace_qwen, with_cf=False).run_all()


# --------------------------------------------------------------------------- #
# the on-demand implementation switch
# --------------------------------------------------------------------------- #


def test_the_switch_serves_the_scores_from_an_sdpa_loaded_model(
    trace_qwen, trace_qwen_default_impl
):
    """The engine's own loading path keeps the checkpoint default (sdpa); a
    group whose address requires eager switches around its trace, restores
    the default after, and stamps what it applied."""
    bundle = trace_qwen_default_impl
    default = bundle.model.config._attn_implementation
    assert default != "eager"  # or this test is vacuous

    executor = _executor(
        TracePointExecutor,
        _read_doc("attention_scores", pos="all"),
        bundle,
        with_cf=False,
    )
    switched = executor.read_value("r")
    assert bundle.model.config._attn_implementation == default  # restored
    assert executor.applied_requirements == {"attn_eager"}

    pinned = _executor(
        TracePointExecutor,
        _read_doc("attention_scores", pos="all"),
        trace_qwen,
        with_cf=False,
    ).read_value("r")
    _assert_same(pinned, switched, "scores through the runtime switch")


def test_a_group_that_needs_no_switch_applies_none(trace_qwen_default_impl):
    """`attention_z` is the interface's own return and exists under every
    implementation — a document reading only it never forces eager (the
    switch's payoff)."""
    executor = _executor(
        TracePointExecutor,
        _read_doc("attention_z", pos=-1),
        trace_qwen_default_impl,
        with_cf=False,
    )
    value = executor.read_value("r")
    assert value.numel() > 0
    assert executor.applied_requirements == set()
    assert trace_qwen_default_impl.model.config._attn_implementation != "eager"
