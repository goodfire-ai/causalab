"""The A3B architecture diagram says the same thing the component table does.

``docs/qwen36-35b-a3b-architecture.html`` draws the forward pass with one
box per tensor, and its lavender fill means exactly one thing: **some engine
exposes this as a hookpoint**. That is a claim about the code, and a claim about
the code drifts — the diagram this test guards shipped with the whole routed-
expert interior greyed out, which was true before the engines exposed those
hookpoints and wrong after.

So the mapping from box to component lives here, next to the engines' own
declarations, and the test is the thing that keeps the picture honest:

* a lavender box (``module``) is a component both-or-either engine serves and a
  write may target;
* a dashed lavender box (``readonly``) is a component some engine serves that
  the capability registry's read-only rows refuse writes to;
* a grey box (``blocked``) is a real tensor in the forward that the vocabulary
  does not name at all.

Parsing the ``node(...)`` calls out of the file is deliberate: the alternative
is a second list of what the diagram contains, which is the thing that drifts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol.registry import CAPABILITIES

pytestmark = pytest.mark.unit

DIAGRAM = (
    Path(__file__).resolve().parents[1] / "docs" / "qwen36-35b-a3b-architecture.html"
)

#: box id → the component it draws, or ``None`` for a box that is a real tensor
#: the vocabulary does not name. Structural boxes (``plain`` kind: the module
#: rectangles) are not listed and are not checked.
#:
#: Several ids map to the same component on purpose — the diagram draws the
#: residual stream once per panel, and ``attn_out`` is the mixer's output in
#: both block types.
BOXES: dict[str, str | None] = {
    # model boundary
    "input_ids": "input_ids",
    "embed": "embeddings",
    "fnorm": "ln_final",
    "logits": "lm_head",
    # the decoder layer
    "resid_pre": "block_input",
    "ln_in": "attention_input_norm",
    "attn_out": "attention_output",
    "resid_mid": "block_mid",
    "ln_post": "mlp_input_norm",
    "mlp_out": "mlp_output",
    "resid_post": "block_output",
    # full-attention mixer
    "a_norm": "attention_input_norm",
    "q_pre": "attention_query_pre_rope",
    "k_pre": "attention_key_pre_rope",
    "q": "attention_query",
    "k": "attention_key",
    "v": "attention_value_states",
    "gate": "attention_gate",
    "scores": "attention_scores",
    "pattern": "attention_probs",
    "z": "attention_z",
    "z_gated": "attention_premix",
    "result": "attention_result",
    "attn_out3": "attention_output",
    # Gated DeltaNet mixer — `delta_*` on the reference engine, `deltanet_*` on
    # the nnsight engine; the box is the tensor, which both of them name
    "d_norm": "attention_input_norm",
    "qkv": "delta_qkv",
    "qkv_conv": "delta_conv",
    "dq": "delta_query",
    "dk": "delta_key",
    "dv": "delta_value",
    "dbeta": "delta_beta",
    "dg": "delta_decay",
    "dz": "delta_gate",
    "dkv_mem": "delta_kv_mem",
    "ddelta": "delta_state_update",
    "dS": "delta_state",
    "do": "delta_kernel_output",
    "do_gated": "delta_premix",
    "attn_out4": "attention_output",
    # sparse MoE + shared expert
    "m_norm": "mlp_input_norm",
    "r_logits": "router_logits",
    # softmax over all 256 experts, before the top-k: the vocabulary names the
    # logits and the renormalized top-k scores, not this
    "r_probs": None,
    "r_scores": "router_scores",
    "r_idx": "expert_idx",
    "gu_e": "expert_gate_proj",  # the fused capture both halves address
    "gate_e": "expert_gate_proj",
    "up_e": "expert_up_proj",
    "act_e": "expert_activation",
    # The complete product at the routed down-projection input.
    "hidden_e": "expert_neuron_output",
    "expert_out": "expert_output",
    "routed": "routed_output",
    "gate_s": "shared_expert_gate_proj",
    "up_s": "shared_expert_up_proj",
    "g_s": "shared_expert_gate",
    "hidden_s": "shared_expert_activation",
    "shared_out": "shared_expert_output",
    # shared_out * sigmoid(g_s) — the vocabulary names the two factors, not the
    # product
    "shared_gated": None,
    "moe_out": "mlp_output",
}

#: ``node('id', cx, cy, 'name', 'shape'[, 'kind'][, opts])`` — the kind is the
#: first string argument after the shape, and absent means ``module``.
_NODE = re.compile(
    r"node\('(?P<id>\w+)',\s*[-\d.]+,\s*[-\d.]+,\s*"
    r"'(?:[^']|\\')*',\s*'(?:[^']|\\')*'"
    r"(?:,\s*'(?P<kind>\w+)')?"
)


def _drawn() -> dict[str, str]:
    """box id → declared kind, parsed out of the diagram."""
    source = DIAGRAM.read_text()
    found = {m.group("id"): m.group("kind") or "module" for m in _NODE.finditer(source)}
    assert found, "no node(...) calls parsed — the diagram's shape changed"
    return found


def _served(component: str) -> bool:
    return component in (
        set(PytorchHooksEngine().components) | set(NnsightEngine().components)
    )


def test_the_mapping_covers_every_box_the_diagram_draws():
    """A new box must say what it is before this file can judge its colour."""
    drawn = {k: v for k, v in _drawn().items() if v != "plain"}
    unmapped = sorted(set(drawn) - set(BOXES))
    assert not unmapped, (
        f"the diagram draws boxes this test has no mapping for: {unmapped}. "
        "Add each to BOXES — the component it draws, or None if the "
        "vocabulary does not name that tensor."
    )
    stale = sorted(set(BOXES) - set(drawn))
    assert not stale, f"BOXES names boxes the diagram no longer draws: {stale}"


def test_a_lavender_box_is_a_hookpoint_and_a_grey_box_is_not():
    """The diagram's one claim, checked against the engines' declarations."""
    drawn = _drawn()
    wrong: list[str] = []
    for box, component in BOXES.items():
        kind = drawn[box]
        exposed = kind in ("module", "readonly")
        if component is None and exposed:
            wrong.append(
                f"{box!r} is drawn as a hookpoint but names no component — "
                "it should be 'blocked'"
            )
        elif component is not None and not exposed:
            wrong.append(
                f"{box!r} is greyed out but {component!r} is a component the "
                "engines serve — it should be a lavender box"
            )
        elif component is not None and not _served(component):
            wrong.append(f"{box!r} maps to {component!r}, which no engine declares")
    assert not wrong, "\n".join(wrong)


def test_the_dashed_boxes_are_exactly_the_read_only_components():
    """`readonly` is not a style choice — it is the row's `writes is None`."""
    drawn = _drawn()
    wrong: list[str] = []
    for box, component in BOXES.items():
        if component is None:
            continue
        dashed = drawn[box] == "readonly"
        refused = CAPABILITIES[component].writes is None
        if dashed and not refused:
            wrong.append(f"{box!r} ({component!r}) is dashed but writes are legal")
        if refused and not dashed:
            wrong.append(
                f"{box!r} ({component!r}) is solid, but a write to it is refused: "
                f"{CAPABILITIES[component].why[:60]}…"
            )
    assert not wrong, "\n".join(wrong)


def test_mlp_activation_has_no_box():
    """The vocabulary entry this architecture has no tensor for must not be
    drawn as one — see docs/qwen36_35b_a3b.md, Dense neuron sites."""
    assert "mlp_activation" not in set(BOXES.values())
    assert "mlp_neuron_output" not in set(BOXES.values())
