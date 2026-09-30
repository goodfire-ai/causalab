"""Per-component truth: shapes, grouped-gate maps and the capability rows.

Shapes per component (the ``(model, site) → d`` rule of §2.5, and more).
[`component_shape`][] answers with a
[`FeatureShape`][] rather than an integer, because
four questions turn on the same fact and used to be answered in four places:
how wide the feature axis is, whether there is one at all, how many heads
``head`` may name, and how the module's native tensor relates to the executor's
``(batch, position, feature)`` contract.

===================================  =======================================
component                            shape
===================================  =======================================
``embeddings``, ``block_input``,     ``(batch, position, hidden)``: the
``block_output``, ``attention_output``,   residual stream. The three norm taps
``mlp_input``, ``mlp_output``,       are here because an RMSNorm maps the
``ln_final``, ``attention_input_norm``,   residual stream to itself, so both
``block_mid``, ``mlp_input_norm``    its sides are hidden-wide
``mlp_activation``                   ``(batch, position, intermediate)`` (the
                                     family caveat of *which* tensor this
                                     names lives in the backend, not here)
``mlp_neuron_output``                 ``(batch, position, intermediate)``; the
                                     complete down-projection input
``attention_premix``                  ``(batch, position, heads·head_dim)``,
                                     head-major and already flattened: the
                                     o-projection's input, query-head space
``lm_head``                          ``(batch, position, vocab)``
``routed_output``,                   ``(batch·position, hidden)``: hidden-wide,
``shared_expert_output``             but flattened like the rest of the MoE
                                     interior
``router_logits``                    ``(batch·position, num_experts)``
``router_scores``                    ``(batch·position, top_k)``, **ranking**:
                                     column *k* is the *k*-th ranked expert, a
                                     different expert for different tokens, so
                                     a basis fitted across positions is fitted
                                     across a shuffled basis. Basis-fitting
                                     featurizers are refused; per-column ones
                                     are not
``expert_idx``                       ``(batch·position, top_k)``, **integral**:
                                     a routing table of integer expert ids :
                                     no featurizer, no gradient
``expert_permutation``               ``(batch·position, top_k)``, **integral**:
                                     the serving kernel's row bookkeeping :
                                     for each (token, slot), the row index in
                                     expert-sorted order
``expert_gate_proj``,                ``(batch·position, top_k·moe_inner)`` :
``expert_up_proj``,                  one vector per routed expert slot,
``expert_activation``,               token-major and **ranking**: slot *k* is
``expert_neuron_output``              the *k*-th ranked expert (the two proj
                                     halves share one fused capture)
``expert_output``                    ``(batch·position, top_k·hidden)``,
                                     token-major, ranking; the value is
                                     **pre-routing-weight**: summing
                                     ``expert_output · router_scores`` over
                                     the top-k axis gives ``routed_output``
``shared_expert_gate_proj``,         ``(batch·position, shared_inner)``
``shared_expert_up_proj``,
``shared_expert_activation``
``shared_expert_gate``               ``(batch·position, 1)``: one mixing
                                     scalar per token
``input_ids``                        ``(batch, position)``, **integral**: no
                                     feature axis at all, so not a feature
                                     space in any sense
``attention_probs``                  ``(batch, head, position[query],
                                     key_position[key])``: **two position
                                     axes**, so no contract form. Every
                                     refusal the executor makes about it is
                                     derived from that
``deltanet_query``, ``deltanet_key``,  the three DeltaNet faces only the nnsight
``deltanet_state``                   engine serves, from the same
                                     four ``linear_*`` dimensions: see
                                     `_deltanet_shape`. q/k are
                                     pre-GVA-tiling (key-head space) where
                                     ``delta_query``/``delta_key`` are post;
                                     ``deltanet_state`` is ``(batch,
                                     position[chunk], head, k_dim·v_dim)``,
                                     its position axis the kernel's 64-token
                                     chunk index where ``delta_state`` is per
                                     step ([`BACKEND_PAIRS`][]). The other
                                     eight ``deltanet_*`` spellings are aliases
                                     of ``delta_*``
===================================  =======================================
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Literal, Mapping, get_args

from causalab.protocol.rules.errors import ReasonCode, ValidationError
from causalab.protocol.registry import shapes
from causalab.protocol.registry.models import ModelInfo
from causalab.protocol.registry.shapes import FeatureShape
from causalab.protocol.schema import (
    COMPONENTS,
    DEPRECATED_COMPONENTS,
    GATE_DEFAULT_MAP,
    GATE_MAPS,
    DEPRECATED_IN,
    MECHANISMS,
    Stream,
)

#: Components whose tensor is the residual stream: an RMSNorm maps it to itself,
#: and both MoE branches write into it, so every one of these is hidden-wide.
_HIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {
        "embeddings",
        "block_input",
        "block_output",
        "attention_output",
        "mlp_input",
        "mlp_output",
        "ln_final",
        "attention_input_norm",
        "block_mid",
        "mlp_input_norm",
    }
)

#: Both MoE branches write into the residual stream, so both are hidden-wide :
#: but the block reshapes to ``(-1, hidden)`` before the router, so their
#: tensors are flattened over (batch, position) like the rest of its interior.
#: 🐞 The width table and the layout table used to say these two things in
#: different places and the descriptor now says them together; they had drifted,
#: and nothing checked, because no code compared a declared width against a real
#: tensor.
_FLAT_HIDDEN_COMPONENTS: frozenset[str] = frozenset(
    {"routed_output", "shared_expert_output"}
)

_SHARED_EXPERT_INNER: frozenset[str] = frozenset(
    {
        "shared_expert_gate_proj",
        "shared_expert_up_proj",
        "shared_expert_activation",
    }
)


def component_shape(info: ModelInfo, component: str) -> FeatureShape:
    """The axes of one component's tensor (the table in the module docstring).

    This is the single description everything else derives from: the feature
    width, whether a featurizer may attach, whether ``head`` means anything and
    how many heads it selects among, and the native↔contract conversion the
    backend performs. It replaced a set of parallel answers: a width function
    with hand-written refusal texts, a five-string layout vocabulary, and a head
    bound that read ``info.num_heads`` regardless of component: that could and
    did disagree with each other.
    """
    if component in _HIDDEN_COMPONENTS:
        return shapes.bsd(info.hidden_size)
    if component in _FLAT_HIDDEN_COMPONENTS:
        return shapes.flat_td(info.hidden_size)
    if component in {"mlp_activation", "mlp_neuron_output"}:
        if info.intermediate_size is None:
            routed = (
                "expert_neuron_output"
                if component == "mlp_neuron_output"
                else "expert_activation"
            )
            raise ValidationError(
                4,
                f"model {info.key!r} declares no dense MLP inner width: every "
                f"layer is a sparse-MoE block, so {component!r} names no tensor "
                f"here. Its analogues are {routed!r} (inside the routed "
                "experts) and 'shared_expert_activation' (the shared expert's "
                "down-projection input).",
                reason="component_unavailable",
            )
        return shapes.bsd(info.intermediate_size)
    if component == "attention_premix":
        # the per-head o-projection input: query-head space (num_heads *
        # head_dim = the o_proj input width), NOT the GQA KV-head space of
        # v_proj. Head-major and already flattened: `(b, s, H*d)`.
        return shapes.bs_flat_heads(info.num_heads, info.head_dim)
    if component == "attention_query_pre_rope":
        # q_norm's output on a family that has one, q_proj's otherwise: the
        # queries as the mixer computes them, BEFORE RoPE rotates them. Query
        # space, so `head` runs 0..num_heads.
        return shapes.bs_flat_heads(info.num_heads, info.head_dim)
    if component == "attention_key_pre_rope":
        # ⚠️ KV-head space, which is narrower than query space by the GQA ratio.
        # This is the component §2.2's head-bound fix exists for: bounding it by
        # num_heads does not raise, it yields an EMPTY feature slice.
        return shapes.bs_flat_heads(info.num_kv_heads, info.head_dim)
    if component == "attention_value_states":
        # v_proj's output: the actual value vectors, KV-head space, and NOT
        # what `attention_premix` (the o_proj input, query space, post-gate)
        # names. The tap is before `past_key_values.update`, so a write reaches
        # the cache.
        return shapes.bs_flat_heads(info.num_kv_heads, info.head_dim)
    if component == "attention_gate":
        # 📐 Qwen3.5/3.6's q-projection emits `[q_h | gate_h]` per head in one
        # tensor of width H·2·d: measured (1, 5, 512) for H 8, d 32 on
        # `tiny-random/qwen3.5-moe`. This component is split 1 of 2, and the
        # fused descriptor is what keeps a write to it from disturbing q.
        return shapes.bs_fused_heads(info.num_heads, 2, 1, info.head_dim)
    if component == "attention_query":
        # 📐 the attention interface's first argument: (b, H, s, d), post-RoPE.
        return shapes.bhsd(info.num_heads, info.head_dim)
    if component == "attention_key":
        # 📐 its second argument: (b, H_kv, s, d), post-RoPE and BEFORE
        # `repeat_kv`: so KV-head space. The position axis is named because it
        # runs over the positions being attended *to*: under a KV cache that is
        # the whole prefix, growing by one per decode step, which is what makes
        # a continuation read of it meaningless rather than merely awkward.
        return shapes.bhsd(info.num_kv_heads, info.head_dim, position_name="key")
    if component == "attention_scores":
        # the softmax's INPUT: same axes as the pattern, one step earlier, and
        # the difference is everything: nothing downstream assumes scores are
        # normalized, because the model's own softmax has yet to run.
        return shapes.attention_pattern(
            info.num_heads,
            note=(
                "Read it whole at pos: \"all\". Unlike 'attention_probs', "
                "every write mechanism is legal here: the model's own softmax "
                "runs after the edit, so rows still sum to 1 by construction."
            ),
        )
    if component == "attention_z":
        # 📐 the interface's return[0]: (b, s, H, d): already transposed back,
        # and BEFORE the gate multiply and the o-projection.
        return shapes.bshd(info.num_heads, info.head_dim)
    if component == "attention_result":
        # ⚠️ The shape of the component's **value**, which is not the shape of
        # the tensor its tap captures: the model never computes this one. Each
        # head's contribution to the residual stream is hidden-wide, so the
        # whole thing is `heads · hidden`: `heads` times `attention_output`,
        # which is why naming a `head` is strongly encouraged and why the read
        # is derived after the position gather rather than before it.
        return shapes.bs_flat_heads(info.num_heads, info.hidden_size)
    if component == "lm_head":
        return shapes.bsd(info.vocab_size)
    if component == "attention_probs":
        return shapes.attention_pattern(
            info.num_heads,
            note=(
                "The read exposes the whole pattern, which is what an "
                'interchange on attention needs: read it at pos: "all", '
                "without a featurizer and without 'dims'."
            ),
        )
    if component == "input_ids":
        return shapes.bs(
            integral=True,
            note=(
                "Read it directly, or read 'embeddings' if you want the vector "
                "the ids look up."
            ),
        )
    if component == "router_logits":
        if info.num_experts is None:
            raise ValidationError(
                4, f"model {info.key!r} declares no experts; router_logits has no width"
            )
        return shapes.flat_td(info.num_experts)
    if component in ("router_scores", "expert_idx"):
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no width",
            )
        if component == "expert_idx":
            return shapes.flat_topk(
                info.num_experts_per_tok,
                integral=True,
                note=(
                    "It is the MoE routing table: integer expert ids on a "
                    "top-k axis. Read or write it directly to inspect or edit "
                    "routing."
                ),
            )
        # ⚠️ Dimensionally well defined, and a plain read of it is meaningful :
        # but column *k* is the *k*-th ranked expert, a different expert for
        # different tokens, so the axis is a ranking rather than a basis. That
        # is what `ranking` says, and what makes `subspace`/`pca`/`sae` refuse.
        return shapes.flat_topk(
            info.num_experts_per_tok,
            ranking=True,
            note=(
                "Its axis is a per-token ranking: column k is the k-th ranked "
                "expert, a different expert for different tokens, so a basis "
                "fitted across positions is fitted across a shuffled basis. "
                "Read it directly, or featurize 'router_logits', whose axis is "
                "the fixed all-experts one."
            ),
        )
    if component in ("expert_gate_proj", "expert_up_proj"):
        # the two halves of the routed up-projection's fused [gate_e | up_e]
        # output: one capture, two addresses (the `attention_gate` precedent),
        # token-major and ranked like the rest of the interior
        if info.num_experts_per_tok is None or info.moe_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no routed-expert inner width "
                f"(moe_intermediate_size) or top-k; {component} has no width",
            )
        return shapes.flat_topk_fused_features(
            info.num_experts_per_tok,
            2,
            0 if component == "expert_gate_proj" else 1,
            info.moe_intermediate_size,
            ranking=True,
            note=(
                "Its slot axis is a per-token ranking (see 'expert_activation'); "
                "join slots to experts through 'expert_idx'."
            ),
        )
    if component == "expert_permutation":
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no top-k axis",
            )
        return shapes.flat_topk(
            info.num_experts_per_tok,
            integral=True,
            note=(
                "It is the serving kernel's row bookkeeping: for each "
                "(token, slot) pair, the row index in expert-sorted order. "
                "Read it to align raw kernel-order tensors; the per-expert "
                "components themselves are already presented token-major."
            ),
        )
    if component == "expert_output":
        # the down-projection's output BEFORE the routing weight: hidden-wide
        # per (token, slot), token-major. The registry identity:
        # routed_output == sum over slots of expert_output · router_scores,
        # exact (the model computes precisely this sum, in this order).
        if info.num_experts_per_tok is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no num_experts_per_tok; "
                f"{component} has no width",
            )
        return shapes.flat_topk_features(
            info.num_experts_per_tok,
            info.hidden_size,
            ranking=True,
            note=(
                "Its slot axis is a per-token ranking (see 'expert_activation'); "
                "join slots to experts through 'expert_idx'. The value is "
                "pre-routing-weight: routed_output == the slot-sum of "
                "expert_output · router_scores."
            ),
        )
    if component in {"expert_activation", "expert_neuron_output"}:
        # Each slot holds one expert's neurons. expert_activation captures
        # act(gate). expert_neuron_output captures act(gate) * up.
        if info.num_experts_per_tok is None or info.moe_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no routed-expert inner width "
                f"(moe_intermediate_size) or top-k; {component} has no width",
            )
        return shapes.flat_topk_features(
            info.num_experts_per_tok,
            info.moe_intermediate_size,
            ranking=True,
            note=(
                "Its slot axis is a per-token ranking: slot k belongs to the "
                "k-th ranked expert, a different expert for different tokens, "
                "so a basis fitted across positions is fitted across a "
                "shuffled basis. Join slots to experts through 'expert_idx', "
                "which has the same (token, slot) rows."
            ),
        )
    if component in _SHARED_EXPERT_INNER:
        if info.shared_expert_intermediate_size is None:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no shared-expert inner width; "
                f"{component} has no width",
            )
        return shapes.flat_td(info.shared_expert_intermediate_size)
    if component == "shared_expert_gate":
        # one scalar per token: how much of the shared expert to mix in
        return shapes.flat_td(1)
    if component in (
        "delta_qkv",
        "delta_gate",
        "delta_premix",
        "delta_conv",
        "delta_query",
        "delta_key",
        "delta_value",
        "delta_beta",
        "delta_decay",
        "delta_kernel_output",
        "delta_kv_mem",
        "delta_state_update",
        "delta_state",
    ):
        missing = [
            name
            for name in (
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
            )
            if getattr(info, name) is None
        ]
        if missing:
            raise ValidationError(
                4,
                f"model {info.key!r} declares no linear-attention stream "
                f"(missing {', '.join(missing)}); {component} has no width",
            )
        assert info.linear_num_value_heads is not None  # for the type-checker
        assert info.linear_num_key_heads is not None
        assert info.linear_key_head_dim is not None
        assert info.linear_value_head_dim is not None
        if component == "delta_qkv":
            # 📐 in_proj_qkv's fused [q | k | v] output: widths key_dim,
            # key_dim, value_dim: UNEQUAL (128/128/256 on the fixture), so
            # there is no head packing to declare and no `head:` here;
            # whole-tensor and `dims` only. The kernel-boundary components
            # are the per-head faces of the same information.
            key_dim = info.linear_num_key_heads * info.linear_key_head_dim
            value_dim = info.linear_num_value_heads * info.linear_value_head_dim
            return shapes.bsd(
                2 * key_dim + value_dim,
                note=(
                    "It is the fused [q | k | v] projection, widths "
                    f"{key_dim}/{key_dim}/{value_dim}: unequal, so it has no "
                    "head axis. The per-head faces are the kernel-boundary "
                    "components ('delta_query'/'delta_key'/'delta_value')."
                ),
            )
        if component == "delta_conv":
            # 📐 causal_conv1d_fn's return: (batch, conv_dim, position) :
            # channels-first, the existing bds layout. Same
            # fused unequal widths as delta_qkv, so no head axis here either.
            key_dim = info.linear_num_key_heads * info.linear_key_head_dim
            value_dim = info.linear_num_value_heads * info.linear_value_head_dim
            return shapes.bds(
                2 * key_dim + value_dim,
                note=(
                    "It is the convolved fused [q | k | v], channels-first and "
                    "with unequal split widths, so it has no head axis. The "
                    "per-head faces are 'delta_query'/'delta_key'/'delta_value'."
                ),
            )
        if component in ("delta_query", "delta_key"):
            # 📐 kernel args 0/1: (b, s, heads, d_k): already tiled to the
            # v-head count (GVA repeat_interleave happens BEFORE the kernel)
            # and PRE-l2norm (the kernel normalizes and scales internally).
            return shapes.bshd(
                info.linear_num_value_heads,
                info.linear_key_head_dim,
                note=(
                    "Captured pre-l2norm: the kernel applies l2norm and the "
                    "1/sqrt(d) scale internally, so this is the tensor a write "
                    "can actually steer."
                ),
            )
        if component in ("delta_value", "delta_kernel_output"):
            # kernel arg 2, and return[0]: v-head space, (b, s, heads, d_v).
            # The output is pre-norm, pre-gate core_attn_out.
            return shapes.bshd(info.linear_num_value_heads, info.linear_value_head_dim)
        if component == "delta_state":
            # ⚠️ On a real checkpoint this is the expensive read: a full-seq
            # all-layers delta_state is layers · seq · heads · d_k · d_v floats
            # (30 · seq · 32·128·128 on ``Qwen/Qwen3.6-35B-A3B``, from its
            # registry entry). Address positions early: the
            # gather runs on the steps axis before anything is kept.
            return shapes.state_matrix(
                info.linear_num_value_heads,
                info.linear_key_head_dim,
                info.linear_value_head_dim,
                note=(
                    "It is the recurrent state S_t: one d_k × d_v matrix per "
                    "head per step. Read it whole (optionally with 'head:'); "
                    "its per-step faces are 'delta_kv_mem' (what the decayed "
                    "state recalls for k̂_t) and 'delta_state_update' (what is "
                    "written in)."
                ),
            )
        if component in ("delta_kv_mem", "delta_state_update"):
            # per-step d_v vectors per head, stacked over steps: derived from
            # adjacent states and pinned by the reconstruction identity
            # S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t
            return shapes.bshd(info.linear_num_value_heads, info.linear_value_head_dim)
        if component == "delta_beta":
            return shapes.bsh(
                info.linear_num_value_heads,
                note=(
                    "One scalar gate per head per position: "
                    "sigmoid(in_proj_b), in (0, 1)."
                ),
            )
        if component == "delta_decay":
            return shapes.bsh(
                info.linear_num_value_heads,
                note=(
                    "The log-decay g: negative reals (the state multiplies by "
                    "exp(g) per step), not a probability."
                ),
            )
        # delta_gate (in_proj_z's output) and delta_premix (out_proj's input)
        # are both value-head space: v-heads · v-head-dim, head-major, flat.
        return shapes.bs_flat_heads(
            info.linear_num_value_heads, info.linear_value_head_dim
        )
    if component.startswith("deltanet_"):
        return _deltanet_shape(info, component)
    raise ValidationError(
        4,
        f"component {component!r} has no declared feature shape: the protocol "
        "layer cannot size it, and featurizers cannot attach to it",
    )


def _deltanet_shape(info: ModelInfo, component: str) -> FeatureShape:
    """The Gated DeltaNet interior's shapes.

    All widths derive from the mixer's four ``linear_*`` dimensions: q/k live
    in key-head space, v/gate/state/output in value-head space, and the fused
    qkv projection is ``2·key_dim + value_dim`` wide.
    """
    if (
        info.linear_num_key_heads is None
        or info.linear_num_value_heads is None
        or info.linear_key_head_dim is None
        or info.linear_value_head_dim is None
    ):
        raise ValidationError(
            4,
            f"model {info.key!r} declares no linear-attention dimensions "
            f"(linear_num_key_heads and friends); {component} has no shape",
        )
    h_k, h_v = info.linear_num_key_heads, info.linear_num_value_heads
    d_k, d_v = info.linear_key_head_dim, info.linear_value_head_dim
    key_dim, value_dim = h_k * d_k, h_v * d_v
    if component == "deltanet_qkv":
        # the fused q|k|v projection, pre-conv: [key | key | value]
        return shapes.bsd(2 * key_dim + value_dim)
    if component == "deltanet_qkv_conv":
        # ⚠️ channels-first: the causal conv works in (batch, width, position)
        return shapes.bds(2 * key_dim + value_dim)
    if component == "deltanet_query":
        # pre repeat_interleave: key-head space, like attention_key under GQA
        return shapes.bshd(h_k, d_k)
    if component == "deltanet_key":
        return shapes.bshd(h_k, d_k)
    if component == "deltanet_value":
        return shapes.bshd(h_v, d_v)
    if component == "deltanet_beta":
        # one write-strength scalar per value head per token: σ(b)
        return shapes.bsd(h_v)
    if component == "deltanet_decay":
        # the log-decay g the kernel consumes, one per value head per token
        return shapes.bsd(h_v)
    if component == "deltanet_gate":
        # the output gate z, consumed by the gated norm after the kernel
        return shapes.bshd(h_v, d_v)
    if component == "deltanet_core_out":
        # the kernel's return, pre-gate: the DeltaNet analogue of attention_z
        return shapes.bshd(h_v, d_v)
    if component == "deltanet_gated_out":
        # after the gated norm, flattened: what the out-projection consumes
        return shapes.bs_flat_heads(h_v, d_v)
    if component == "deltanet_state":
        return shapes.chunked_state(
            h_v,
            d_k,
            d_v,
            note=(
                "Its position axis is the kernel's 64-token chunk index: read "
                'it whole (pos: "all") or at an integer chunk index. Per-token '
                "prefill state does not exist: the recurrent kernel runs only "
                "in single-token decode (the modeling code's own dispatch)."
            ),
        )
    raise ValidationError(
        4,
        f"component {component!r} has no declared feature shape: the protocol "
        "layer cannot size it, and featurizers cannot attach to it",
    )


def _no_axis(component: str, axis: str, shape: FeatureShape) -> str:
    """ "component X has no <axis> axis: its shape is (…)".

    Factored because two refusals share it: ``head:`` on a headless component
    (§2.2) and ``group: "head"`` on one (§2.5): and the whole point of a
    generated refusal is that the two cannot come to describe the same absent
    axis differently.
    """
    return (
        f"component {component!r} has no {axis} axis: its shape is {shape.describe()}"
    )


def _with_note(message: str, shape: FeatureShape) -> str:
    """``message`` plus the shape's "…so do this instead" half.

    A descriptor can generate *why* an axis is absent but not what to do about
    it: ``delta_qkv``'s note names the per-head faces of the same information :
    so the note is appended rather than regenerated.
    """
    return f"{message} {shape.note}" if shape.note else message


def head_space_refusal(component: str, head: int, shape: FeatureShape) -> str:
    """Why ``head`` does not apply to ``component``: the §2.2 refusal.

    Shared by the canonicalizer (which refuses at load) and
    [`component_width`][] (which refuses if anything reaches it another way),
    so the two cannot drift into disagreeing about what a head means.
    """
    return _with_note(
        f"{_no_axis(component, 'head', shape)}: so head {head} would be "
        "validated and then silently dropped. Name a component that has heads "
        "('attention_premix'), or drop the 'head' field.",
        shape,
    )


def component_width(info: ModelInfo, component: str, *, head: int | None = None) -> int:
    """The feature width at one site.

    A thin reading of [`component_shape`][]: the product of the feature axes,
    or one head's slice of it. Kept as a function because three call sites want
    exactly this number and nothing else about the shape.
    """
    shape = component_shape(info, component)
    if not shape.is_feature_space:
        raise ValidationError(4, shape.refusal(f"component {component!r}"))
    width = shape.width
    assert width is not None  # is_feature_space implies a feature axis
    if head is None:
        return width
    space = shape.head_space
    if space is None:
        raise ValidationError(4, head_space_refusal(component, head, shape))
    return width // space


def head_group_map(
    shape: FeatureShape, width: int, *, component: str
) -> tuple[int, int]:
    """The ``(groups, group_width)`` a head-grouped gate has on ``component``
    at a site ``width`` coordinates wide (§2.5 ``group: head``).

    One group per head, each ``head_dim`` coordinates wide, in the order the
    component's head-major axis lays them out: the same slices
    ``sites._head_slice`` names, so a group *is* the head a ``head`` field
    would select. ``width`` is the site's own width: the whole component, or
    one head's slice of it (then the map is a single group). Derived here from
    the shape alone, so the canonicalizer (offline, from the registry) and the
    executor (from the resolved site) cannot disagree about it.

    Raises:
        ValidationError: rule 23 (group legality). The component has no head
            axis: there is nothing to group by, and a gate that silently fell
            back to one parameter per coordinate would report a coordinate
            count as a head count. Or ``width`` is not a whole number of heads,
            which happens when the gate sits after a stage that changed the
            coordinates (a rotation), where "head" no longer names anything.
    """
    space = shape.head_space
    if space is None:
        raise ValidationError(
            23,
            f"gate group 'head' on component {component!r}: the component has "
            f"no head axis: its shape is {shape.describe()}: so there are no "
            "heads to group by. Name a head-major component "
            "('attention_premix' on a full-attention layer, 'delta_premix' on "
            "a Gated DeltaNet layer), or drop 'group'."
            + (f" {shape.note}" if shape.note else ""),
        )
    assert shape.width is not None  # a head axis implies a feature axis
    group_width = shape.width // space
    if width <= 0 or width % group_width:
        raise ValidationError(
            23,
            f"gate group 'head' on component {component!r}: the gate's input is "
            f"{width} wide, which is not a whole number of {group_width}-wide "
            "heads: a head-grouped gate acts on the component's own "
            "coordinates, so no stage before it may change them",
        )
    return width // group_width, group_width


def expert_neuron_group_map(
    info: ModelInfo | None, shape: FeatureShape, width: int, *, component: str
) -> tuple[int, int]:
    """The ``(num_experts, d_expert)`` an expert-keyed gate has on
    ``component`` at a site ``width`` coordinates wide (§2.5 ``group:
    expert_neuron``).

    One parameter per ``(expert, neuron)`` of the routed interior: the whole
    expert table, ``num_experts × d_expert``, whatever ``top_k`` experts a
    token activates. The site itself is token-major, ``top_k · d_expert`` wide
    with slot *k* the *k*-th ranked expert, so a slot's coordinates find their
    parameters through ``expert_idx`` at run time; the map only says how big
    the table is. ``expert_activation`` and ``expert_neuron_output`` each
    hold one expert's neurons in each slot.

    Raises:
        ValidationError: rule 23 rejects an unsupported component, a missing
            expert table, or an input width changed by an earlier stage.
            The error names the component and the required shape.
    """
    if component not in {"expert_activation", "expert_neuron_output"}:
        raise ValidationError(
            23,
            f"gate group 'expert_neuron' on component {component!r}: an "
            "expert-keyed gate holds one parameter per (expert, neuron) of the "
            "routed interior. 'expert_activation' and 'expert_neuron_output' "
            "contain one expert's neurons per routed slot. Name either, or "
            "drop 'group' (on 'shared_expert_activation' the per-coordinate "
            "gate is already one parameter per shared-expert neuron).",
        )
    if info is None or info.num_experts is None or info.moe_intermediate_size is None:
        raise ValidationError(
            23,
            f"gate group 'expert_neuron' on component {component!r}: the "
            "model declares no expert table (num_experts and "
            "moe_intermediate_size), so there is nothing to key the parameters by",
        )
    if width != shape.width:
        raise ValidationError(
            23,
            f"gate group 'expert_neuron' on component {component!r}: the "
            f"gate's input is {width} wide but the component is {shape.width}: "
            "an expert-keyed gate acts on the component's own routed slots, so "
            "no stage before it may change them",
        )
    return info.num_experts, info.moe_intermediate_size


def site_group_map_whole(
    shape: FeatureShape, width: int, *, component: str
) -> tuple[int, int]:
    """The ``(1, width)`` map of a site-grouped gate (§2.5 ``group: site``):
    one parameter over every coordinate of the site, so the site is one unit :
    the way a node-level circuit benchmark scores an MLP block or the input
    embedding. It is the ``head`` map with a single group, and everything
    downstream (the mask broadcast, the L1 mean, the ``rank`` rows, the
    size-matched control) reads it through the same code. ``width`` is the
    gate's own input width: the whole component, or one head's slice of it
    when the site names a ``head``; either is legitimately one unit.

    Raises:
        ValidationError: rule 23 (group legality). The component is not a
            feature space (an attention pattern, the routing table), so there is
            no coordinate axis to cover.
    """
    if not shape.is_feature_space or width <= 0:
        raise ValidationError(
            23,
            f"gate group 'site' on component {component!r}: the component has no "
            f"feature axis to cover: its shape is {shape.describe()}: so there "
            "is nothing for one parameter to gate. Name a feature-space "
            "component, or drop 'group'." + (f" {shape.note}" if shape.note else ""),
        )
    return 1, width


def gate_group_map(
    group: str,
    shape: FeatureShape,
    width: int,
    *,
    component: str,
    info: ModelInfo | None = None,
) -> tuple[int, int]:
    """The derived group map of a grouped gate, by group kind (§2.5):
    [`head_group_map`][] for ``head``, [`expert_neuron_group_map`][]
    for ``expert_neuron``, which additionally needs the model's expert table
    (``info``), and [`site_group_map_whole`][] for ``site``. One entry point
    so the canonicalizer, the loader and the executor derive the same map from
    the same facts."""
    if group == "head":
        return head_group_map(shape, width, component=component)
    if group == "expert_neuron":
        return expert_neuron_group_map(info, shape, width, component=component)
    if group == "site":
        return site_group_map_whole(shape, width, component=component)
    raise ValueError(f"unknown gate group {group!r}")  # the schema's enum is closed


#: Which site field selects a *single* member of what each group groups over
#: (§5.23): a ``head`` on the site leaves a head-grouped gate one group, an
#: ``expert`` leaves an expert-keyed gate one expert's neurons: a per-coordinate
#: gate wearing a grouped gate's name, refused rather than resolved. Keyed by
#: group so a group added to the vocabulary without a row here fails the census
#: (``test_every_group_has_a_site_selector``) instead of skipping the check.
#: ``None`` says the group has no such selector: a ``site`` gate over one head's
#: slice is still one parameter over one unit, exactly what it claims to be.
GROUP_SITE_SELECTORS: dict[str, str | None] = {
    "head": "head",
    "expert_neuron": "expert",
    "site": None,
}


def site_group_map(
    info: ModelInfo,
    group: str,
    component: str,
    *,
    head: int | None = None,
    expert: int | None = None,
) -> tuple[int, int]:
    """The group map of a grouped gate at one *declared* site: ``component``
    and, if the site names them, ``head`` and ``expert``: from the registry
    alone ([`gate_group_map`][] over [`component_shape`][] and
    [`component_width`][]), with no model loaded.

    The offline reading of the map: what the canonicalizer stamps into a
    document's ``params``, what the loader expects a fitted bundle to carry,
    and what a test derives to compare with the executor's own reading from
    the resolved site. One function so the three cannot disagree on how a
    site's fields become a map.

    Raises:
        ValidationError: rule 23 (group legality). The site already selects a
            single member of what the group groups over (``head: 3`` under
            ``group: head``, ``expert: 7`` under ``group: expert_neuron``) :
            H groups over one head is one group, a coordinate-wise gate under
            a name that claims otherwise: or the component has no such axis
            ([`gate_group_map`][]).
    """
    field = GROUP_SITE_SELECTORS[group]  # the schema's enum is closed
    selected = None if field is None else {"head": head, "expert": expert}[field]
    if selected is not None:
        raise ValidationError(
            23,
            f"site component {component!r} already selects {field} {selected}, "
            f"so group {group!r} has exactly one group: a per-{field} gate over "
            f"one {field} is a coordinate-wise gate. Drop the site's {field!r}, "
            "or drop the group.",
        )
    return gate_group_map(
        group,
        component_shape(info, component),
        component_width(info, component, head=head),
        component=component,
        info=info,
    )


def gate_param_shape(
    group: str | None,
    group_map: tuple[int, int] | None,
    width: int,
    *,
    parametrization: str = GATE_DEFAULT_MAP,
) -> tuple[int, ...]:
    """The shape of a gate's ``theta`` (§2.5): one entry per coordinate with no
    group, one per head (``[heads]``) under ``head``, exactly one (``[1]``)
    under ``site``, and the whole expert table (``[num_experts, d_expert]``)
    under ``expert_neuron``: two-dimensional so a saved bundle reads as
    ``theta[expert, neuron]``. Under an *indexed* map
    (``GATE_MAPS[…].indexed``: ``boundary``) ``theta`` is the one scalar β
    whatever the width: ``[1]``: and takes no group."""
    if GATE_MAPS[parametrization].indexed:
        return (1,)
    if group is None or group_map is None:
        return (width,)
    if group in ("head", "site"):
        return (group_map[0],)
    if group == "expert_neuron":
        return tuple(group_map)
    raise ValueError(f"unknown gate group {group!r}")


# --------------------------------------------------------------------------- #
# the capability registry: one row per component
# --------------------------------------------------------------------------- #
#
# Everything that used to be a *second table* of component truth reads from
# here: which engines serve a component (``Engine.components``, generated),
# which mechanisms a write may use (the executor's write policy, and the
# load-time twin in ``validate``), which mixer stream it exists on
# (``COMPONENT_STREAMS``, now a view of the rows), which architectural facts it
# needs (``requires``: the module-tree probes in ``neural/shared/model_tree.py`` are
# the run-time evaluators), which engines serve its ragged ``expert:`` face,
# and the retired spellings that fold onto it. The docs tables
# (``docs/qwen36_35b_a3b.md``, spec §8's component row) are rendered
# from the rows by [`render_component_tables`][]; the census guards in
# ``tests/protocol/test_vocabulary_census.py`` hold the rendering, the engine
# sets and the row count to the rows.
#
# The design: one row per component keyed
# by name, predicates declared rather than families enumerated, and an
# ``overrides`` slot per family: the per-family tap table, filled for
# the three families the attention interior was *measured* on. Nothing here
# enters a document's canonical form (no digest pin moves): the rows
# only decide what is refused, where a tap lands, and what is rendered.

#: The engines a row may name. ``causalab/neural/shared/engine_router.py`` derives
#: ``ENGINE_CHOICES`` from this (plus ``"auto"``), so a third engine is a name
#: here, a class that declares it, and nothing else.
ENGINES: tuple[str, ...] = ("pytorch_hooks", "nnsight")

_BOTH: frozenset[str] = frozenset(ENGINES)
_HOOKS: frozenset[str] = frozenset({"pytorch_hooks"})
_NNSIGHT: frozenset[str] = frozenset({"nnsight"})
_NEITHER: frozenset[str] = frozenset()

#: Every ``do`` mechanism: the ``writes`` cell of a row that accepts any.
_ANY_MECHANISM: frozenset[str] = frozenset(MECHANISMS)
_SWAP_ONLY: frozenset[str] = frozenset({"swap"})

#: Architectural facts a component needs, evaluated against [`ModelInfo`][]
#: at load where the entry can decide them (``moe``, ``shared_expert``: the
#: MoE widths) and against the loaded module tree at run
#: (``neural/shared/model_tree.py``, all five). Order is the order the run-time
#: check evaluates them in, which is the order the refusal texts were pinned
#: in: a fused-qkv family is refused before its missing gate is, a dense MLP
#: before its missing shared expert.
Predicate = Literal[
    "moe", "shared_expert", "grouped_mm", "split_qkv", "gated_attention"
]
PREDICATES: tuple[Predicate, ...] = get_args(Predicate)

#: How the serving engine reaches the tensor: documentation for the rendered
#: table, closed so a typo cannot render. The operational dispatch is
#: ``neural/shared/sites.resolve_site``; this names its kinds for a reader.
TapKind = Literal[
    "module input",
    "module output",
    "attention-function slot",
    "delta-kernel boundary",
    "grouped-experts dispatch",
    "`.source` line (fused forward)",
    "derived from `attention_premix`",
]
TAP_KINDS: tuple[TapKind, ...] = get_args(TapKind)

#: The keys a per-family override may carry: the **address** of one
#: attention-interior component on one family (the per-family tap table).
#: Closed, so a misspelt key cannot
#: silently mean "no override":
#:
#: ``module``
#:     the mixer child whose *output* is tapped (``q_proj``, ``q_norm``,
#:     ``c_attn``, …): a name, never a module: the protocol layer is
#:     torch-free;
#: ``packing``
#:     how that module's native tensor packs the component's value
#:     ([`PACKINGS`][]);
#: ``splits`` / ``split``
#:     for a fused packing only: how many logical tensors share the module's
#:     output, and which one this component is.
OverrideKey = Literal["module", "packing", "splits", "split"]
OVERRIDE_KEYS: tuple[OverrideKey, ...] = get_args(OverrideKey)

#: How a tapped module's native tensor packs the component's logical value,
#: whose axes [`component_shape`][] describes family-independently. 📐 All
#: four measured (transformers 5.16):
#:
#: ``flat``
#:     the module's whole output *is* the value, ``(b, s, heads·d)`` :
#:     llama's bare ``q_proj`` (16 = 4·4);
#: ``head_axis``
#:     the whole output with the head axis kept, ``(b, s, heads, d)`` :
#:     ``Qwen3_5MoeAttention.q_norm`` emits ``(1, 5, 8, 32)``;
#: ``fused_heads``
#:     ``splits`` logical tensors interleaved *per head*,
#:     ``(b, s, heads·splits·d)``: qwen3.5-moe's ``q_proj`` packs
#:     ``[q_h | gate_h]`` (512 = 8·2·32);
#: ``fused_blocks``
#:     ``splits`` logical tensors as contiguous *blocks*,
#:     ``(b, s, splits·heads·d)``: GPT-2's ``c_attn`` emits ``[q | k | v]``
#:     (96 = 3·4·8) and the mixer splits it with ``.split(split_size, dim=2)``.
Packing = Literal["flat", "head_axis", "fused_heads", "fused_blocks"]
PACKINGS: tuple[Packing, ...] = get_args(Packing)
_FUSED_PACKINGS: frozenset[str] = frozenset({"fused_heads", "fused_blocks"})


def _check_override(component: str, family: str, address: Mapping[str, Any]) -> None:
    """One override is a well-formed address, or the row cannot be built."""
    where = f"{component}: override for family {family!r}"
    unknown = sorted(set(address) - set(OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"{where} has keys {unknown} outside {list(OVERRIDE_KEYS)}")
    module = address.get("module")
    if not isinstance(module, str) or not module.isidentifier():
        raise ValueError(f"{where} needs a module name, got {module!r}")
    packing = address.get("packing")
    if packing not in PACKINGS:
        raise ValueError(f"{where} names packing {packing!r}, not in {list(PACKINGS)}")
    splits, split = address.get("splits"), address.get("split")
    if packing in _FUSED_PACKINGS:
        if not (isinstance(splits, int) and splits >= 2):
            raise ValueError(
                f"{where}: a fused packing needs splits >= 2, got {splits!r}"
            )
        if not (isinstance(split, int) and 0 <= split < splits):
            raise ValueError(f"{where}: split {split!r} is not in range({splits})")
    elif splits is not None or split is not None:
        raise ValueError(f"{where}: packing {packing!r} takes no splits/split")


@dataclasses.dataclass(frozen=True)
class Capability:
    """One component's row: what exists, who serves it, what a write may do.

    ``reads`` is the set of engine names whose site resolver serves the
    component: and therefore its write surface too: ``Engine.components`` and
    ``Engine.writable_components`` are both generated from it. Write *policy*
    (``writes``) is deliberately not an engine fact: declaring ``router_logits``
    unwritable on one engine would turn "a write here reaches nothing: write
    ``router_scores``" into "try another engine", the wrong answer everywhere.

    ``writes`` is the closed set of mechanisms a write may use; ``None`` means
    read-only. ``why`` is the refusal text for either, verbatim from the tables
    it replaced, and ``reason`` the code that refusal carries
    ([`REASON_CODES`][causalab.protocol.rules.errors.REASON_CODES]). ``write_capability`` is
    the coarse §8 verb a write charges beyond the generated
    ``component:<name>:write``: today only the pattern's, whose write goes
    through the attention function rather than a hook.

    ``overrides`` is the per-family tap table: for each family
    ([`ModelInfo.family`][]) the row has been measured on, the **address**
    of the component on that family: the mixer child to tap and how its
    native tensor packs the value ([`OVERRIDE_KEYS`][], [`PACKINGS`][]).
    Only the attention interior carries any: it is the one place the families
    disagree about *where* a component is (GPT-2 fuses q, k and v into one
    ``c_attn``; qwen3.5-moe normalizes q and k before RoPE and packs a gate
    beside q). The site resolver reads the address; a family absent from a row
    is served by measurement where that is unambiguous (a bare projection, a
    norm) and refused where it is not (a fused projection whose block order
    only a row can state). The same rows let [`predicate_holds`][] decide
    ``split_qkv`` and ``gated_attention`` offline for a family that has them.
    """

    component: str
    stream: Stream | None
    reads: frozenset[str]
    writes: frozenset[str] | None
    expert_selection: frozenset[str]
    requires: frozenset[Predicate]
    reason: ReasonCode | None
    why: str
    write_capability: str | None
    tap: TapKind
    aliases: tuple[str, ...]
    deprecated_in: str | None
    overrides: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.reads <= _BOTH:
            raise ValueError(f"{self.component}: unknown engine in {self.reads}")
        if not self.expert_selection <= self.reads:
            raise ValueError(
                f"{self.component}: expert_selection names an engine that does "
                "not serve the component"
            )
        if self.writes is not None and not self.writes <= _ANY_MECHANISM:
            raise ValueError(f"{self.component}: unknown mechanism in {self.writes}")
        restricted = self.writes is None or self.writes != _ANY_MECHANISM
        if restricted != bool(self.why):
            raise ValueError(
                f"{self.component}: a restricted write policy and its 'why' text "
                "come together"
            )
        if (self.reason == "unsupported_mechanism") != restricted:
            raise ValueError(
                f"{self.component}: reason {self.reason!r} does not match the "
                "write policy"
            )
        for family, address in self.overrides.items():
            _check_override(self.component, family, address)
        # within one row, a module name means one packing: the measured
        # fallback for a family without an address picks among the declared
        # modules the mixer has, which is only well defined if they agree
        packings: dict[str, Any] = {}
        for address in self.overrides.values():
            seen = packings.setdefault(address["module"], address)
            if seen != address:
                raise ValueError(
                    f"{self.component}: module {address['module']!r} is declared "
                    "with two different packings across families"
                )

    @property
    def read_only(self) -> bool:
        return self.writes is None

    def address_on(self, info: ModelInfo) -> Mapping[str, Any] | None:
        """The row's address of the component on ``info``'s family: or
        ``None`` for a family the table has not met (or an entry with none)."""
        if info.family is None:
            return None
        return self.overrides.get(info.family)


def _row(
    component: str,
    *,
    tap: TapKind,
    stream: Stream | None = None,
    reads: frozenset[str] = _BOTH,
    writes: frozenset[str] | None = _ANY_MECHANISM,
    expert_selection: frozenset[str] = _NEITHER,
    requires: tuple[Predicate, ...] = (),
    why: str = "",
    write_capability: str | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Capability:
    """One row. ``reason`` is derived from the policy (one code per refusal
    kind, never authored twice) and the aliases come from the vocabulary's own
    retired-spelling map, so a name lives in exactly one place."""
    restricted = writes is None or writes != _ANY_MECHANISM
    aliases = tuple(
        sorted(old for old, new in DEPRECATED_COMPONENTS.items() if new == component)
    )
    # every alias records the protocol version it was retired under
    # (schema.DEPRECATED_IN); a row's aliases share one, which the census holds
    versions = sorted({DEPRECATED_IN[alias] for alias in aliases})
    if len(versions) > 1:  # pragma: no cover - the census names it
        raise AssertionError(
            f"{component}: its aliases {aliases} were retired under different "
            f"protocol versions {versions}; one row, one deprecation version"
        )
    return Capability(
        overrides=MappingProxyType(
            {
                family: MappingProxyType(dict(address))
                for family, address in (overrides or {}).items()
            }
        ),
        component=component,
        stream=stream,
        reads=reads,
        writes=writes,
        expert_selection=expert_selection,
        requires=frozenset(requires),
        reason="unsupported_mechanism" if restricted else None,
        why=why,
        write_capability=write_capability,
        tap=tap,
        aliases=aliases,
        deprecated_in=versions[0] if versions else None,
    )


# The write-policy texts, verbatim from the three tables they replace
# (``sites.READ_ONLY_COMPONENTS``, ``sites.SWAP_ONLY_COMPONENTS``,
# ``sites.NORMALIZED_TAPS``). Each names the alternative rather than just
# saying no; the executor and the validator wrap them in one template.
#
# The two read-only entries that are easiest to get wrong are refused for
# *different* reasons: ``router_logits`` is a **silent no-op** (📐 the MoE block
# destructures the router as ``_, routing_weights, selected_experts`` and
# never reads element 0 again: patching it moves the logits by exactly 0.0
# while a patch to ``router_scores`` or ``expert_idx`` moves them), and
# ``input_ids`` is the opposite: a write there *does* land (the embedding's
# pre-hook input, mutated in place), and is refused because token ids are not
# an activation: editing them is a change to the dataset.
_WHY_INPUT_IDS = (
    "token IDs are dataset values. Change the row text, or write 'embeddings' to edit "
    "their vectors"
)
_WHY_ATTENTION_RESULT = (
    "this per-head contribution is derived from the joint output projection. Write "
    "'attention_premix' with the same 'head' to change it by the corresponding linear "
    "projection"
)
_WHY_ROUTER_LOGITS = (
    "routing uses the scores and indices already computed from these logits. Write "
    "'router_scores' to reweight selected experts or 'expert_idx' to select experts"
)
_WHY_DELTA_KV_MEM = (
    "this readout is recomputed from state as S_{t-1}·exp(g_t) · k̂_t at each step. "
    "Write 'delta_state' to change memory or 'delta_value' to change what is stored"
)
_WHY_DELTA_STATE_UPDATE = (
    "state updates are exposed for reading. Write 'delta_state' to edit S_t in S_t = "
    "S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t"
)
_WHY_EXPERT_PERMUTATION = (
    "the kernel derives this ordering of token-slot rows from routing. Write "
    "'expert_idx' to select experts or 'router_scores' to reweight them"
)
# 📐 ``expert_idx`` measured: an ``add_scaled`` write over the int64 routing
# table ran to completion with no refusal anywhere; on CUDA an out-of-range id
# is a device-side assert far from the write that caused it.
_WHY_EXPERT_IDX = (
    "expert IDs are integer labels. Use swap to replace the routing indices, or write "
    "'router_scores' to reweight selected experts. Arithmetic on IDs can select "
    "arbitrary experts or exceed index bounds"
)
# The distinction ``attention_scores`` exists to remove: same axes, and a
# delta on either is arithmetically fine: the difference is entirely in what
# happens *next*. After the pattern, the value multiply assumes rows summing
# to 1; after the scores, the model's own softmax renormalizes by construction.
_WHY_ATTENTION_PROBS = (
    "each row must sum to 1 for the following value multiply. Write "
    "'attention_scores' to use other mechanisms before softmax restores normalized "
    "probabilities"
)

_MOE: tuple[Predicate, ...] = ("moe",)
_SHARED: tuple[Predicate, ...] = ("moe", "shared_expert")
_ROUTED: tuple[Predicate, ...] = ("moe", "grouped_mm")
_SPLIT: tuple[Predicate, ...] = ("split_qkv",)

# The per-family tap table: the three families the attention
# interior has been measured on, keyed by the HF ``model_type`` the adapter
# reads (measured: ``GPT2Config.model_type == "gpt2"``,
# ``LlamaConfig.model_type == "llama"``, ``Qwen3_5MoeTextConfig.model_type ==
# "qwen3_5_moe_text"``). The nnsight engine took the same address-table shape
# for the function interiors first (``nnsight_tracing/addresses.py``);
# this is the module-boundary half, on the rows. 📐 Per family:
#
# ==========================  =====================  ================  ======================
# component                   ``gpt2``               ``llama``         ``qwen3_5_moe_text``
# ==========================  =====================  ================  ======================
# ``attention_query_pre_rope``  ``c_attn`` block 0/3   ``q_proj`` flat   ``q_norm`` (b,s,H,d)
# ``attention_key_pre_rope``    ``c_attn`` block 1/3   ``k_proj`` flat   ``k_norm`` (b,s,H_kv,d)
# ``attention_value_states``    ``c_attn`` block 2/3   ``v_proj`` flat   ``v_proj`` flat
# ``attention_gate``           :                     :                 ``q_proj`` split 1/2 per head
# projection width            3·H·d = 96             H·d = 16          H·2·d = 512
# ==========================  =====================  ================  ======================
#
# GPT-2's ``GPT2Attention.forward`` computes ``query, key, value =
# self.c_attn(hidden_states).split(self.split_size, dim=2)`` with
# ``split_size = H·d``, so the blocks are contiguous column ranges
# ``[0:H·d] | [H·d:2H·d] | [2H·d:3H·d]`` (measured equal on the fixture).
# ``docs/running_experiments.md`` §5 renders this table from the rows
# ([`render_family_table`][]); nothing else restates it.
_GPT2 = "gpt2"
_LLAMA = "llama"
_QWEN35_MOE = "qwen3_5_moe_text"


def _fused_blocks(split: int) -> dict[str, Any]:
    return {"module": "c_attn", "packing": "fused_blocks", "splits": 3, "split": split}


def _flat(module: str) -> dict[str, Any]:
    return {"module": module, "packing": "flat"}


def _head_axis(module: str) -> dict[str, Any]:
    return {"module": module, "packing": "head_axis"}


_ROWS: tuple[Capability, ...] = (
    # --- the model boundary (layer-less) ---------------------------------- #
    _row("input_ids", tap="module input", writes=None, why=_WHY_INPUT_IDS),
    _row("embeddings", tap="module output"),
    _row("ln_final", tap="module output"),
    _row("lm_head", tap="module output"),
    # --- the residual stream, every layer ---------------------------------- #
    _row("block_input", tap="module input"),
    _row("attention_input_norm", tap="module output"),
    _row("attention_output", tap="module output"),
    _row("block_mid", tap="module input"),
    _row("mlp_input_norm", tap="module output"),
    _row("mlp_input", tap="module input"),
    _row("mlp_output", tap="module output"),
    _row("block_output", tap="module output"),
    # the dense MLP's inner activation: llama's ``act_fn`` output, GPT-2's
    # ``c_proj`` input. Its availability is the dense inner width itself
    # (``ModelInfo.intermediate_size``): an all-MoE tower declares none, and
    # [`component_shape`][] refuses it there.
    _row("mlp_activation", tap="module output"),
    _row("mlp_neuron_output", tap="module input"),
    # --- the full-attention mixer ----------------------------------------- #
    _row(
        "attention_query_pre_rope",
        tap="module output",
        stream="full_attention",
        requires=_SPLIT,
        overrides={
            _GPT2: _fused_blocks(0),
            _LLAMA: _flat("q_proj"),
            _QWEN35_MOE: _head_axis("q_norm"),
        },
    ),
    _row(
        "attention_key_pre_rope",
        tap="module output",
        stream="full_attention",
        requires=_SPLIT,
        overrides={
            _GPT2: _fused_blocks(1),
            _LLAMA: _flat("k_proj"),
            _QWEN35_MOE: _head_axis("k_norm"),
        },
    ),
    _row(
        "attention_value_states",
        tap="module output",
        stream="full_attention",
        requires=_SPLIT,
        overrides={
            _GPT2: _fused_blocks(2),
            _LLAMA: _flat("v_proj"),
            _QWEN35_MOE: _flat("v_proj"),
        },
    ),
    _row(
        "attention_gate",
        tap="module output",
        stream="full_attention",
        requires=("split_qkv", "gated_attention"),
        # the one family with a gate; its absence on the other two is what
        # lets `gated_attention` refuse them at load
        overrides={
            _QWEN35_MOE: {
                "module": "q_proj",
                "packing": "fused_heads",
                "splits": 2,
                "split": 1,
            }
        },
    ),
    _row("attention_query", tap="attention-function slot", stream="full_attention"),
    _row("attention_key", tap="attention-function slot", stream="full_attention"),
    _row("attention_scores", tap="attention-function slot", stream="full_attention"),
    _row("attention_z", tap="attention-function slot", stream="full_attention"),
    _row(
        "attention_probs",
        tap="module output",
        stream="full_attention",
        writes=_SWAP_ONLY,
        why=_WHY_ATTENTION_PROBS,
        # the write goes through the eager attention function, not a hook :
        # a capability an engine may lack, so a coarse verb routes on it
        write_capability="writable_attention_probs",
    ),
    _row("attention_premix", tap="module input", stream="full_attention"),
    _row(
        "attention_result",
        tap="derived from `attention_premix`",
        stream="full_attention",
        writes=None,
        why=_WHY_ATTENTION_RESULT,
    ),
    # --- the Gated DeltaNet mixer ------------------------------------------ #
    # One semantic name per tensor, whichever engine serves it. The module boundaries and
    # the kernel boundary are served by BOTH engines: the reference engine by
    # hooks and by swapping the modeling file's kernel globals, the nnsight
    # engine by envoys and by its `.source` address table: each translating
    # the one name to its own mechanism (the eight retired `deltanet_*`
    # spellings fold onto these at parse). The `tap` cell names the reference
    # engine's mechanism; the nnsight one is `.source` for the kernel boundary.
    _row("delta_qkv", tap="module output", stream="linear_attention"),
    _row("delta_gate", tap="module output", stream="linear_attention"),
    _row("delta_premix", tap="module input", stream="linear_attention"),
    _row("delta_conv", tap="delta-kernel boundary", stream="linear_attention"),
    # post GVA `repeat_interleave`: value-head space; the nnsight engine's
    # pre-tiling face is `deltanet_query` (BACKEND_PAIRS: `gva_tile`)
    _row(
        "delta_query",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
    ),
    _row(
        "delta_key",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
    ),
    _row("delta_value", tap="delta-kernel boundary", stream="linear_attention"),
    _row("delta_beta", tap="delta-kernel boundary", stream="linear_attention"),
    _row("delta_decay", tap="delta-kernel boundary", stream="linear_attention"),
    _row(
        "delta_kernel_output",
        tap="delta-kernel boundary",
        stream="linear_attention",
    ),
    _row(
        "delta_kv_mem",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
        writes=None,
        why=_WHY_DELTA_KV_MEM,
    ),
    _row(
        "delta_state_update",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
        writes=None,
        why=_WHY_DELTA_STATE_UPDATE,
    ),
    # per step: the nnsight engine's per-chunk face is `deltanet_state`
    # (BACKEND_PAIRS: `chunk_boundary`)
    _row(
        "delta_state",
        tap="delta-kernel boundary",
        stream="linear_attention",
        reads=_HOOKS,
    ),
    # --- the three DeltaNet faces only the nnsight engine serves ----------- #
    # Two names stay two names where the tensors differ in shape or timing:
    # q/k before the GVA tiling (key-head space, where `delta_query`/`delta_key`
    # are tiled to value heads) and the state once per 64-token chunk (where
    # `delta_state` is per step). An alias here would rebind, not redirect :
    # `alias_would_rebind` refuses it, and the pair's typed relation is a row
    # (BACKEND_PAIRS) the test helpers read rather than own.
    *(
        _row(
            name,
            tap="`.source` line (fused forward)",
            stream="linear_attention",
            reads=_NNSIGHT,
        )
        for name in ("deltanet_query", "deltanet_key", "deltanet_state")
    ),
    # --- the sparse MoE block and its shared expert ----------------------- #
    _row(
        "router_logits",
        tap="module output",
        requires=_MOE,
        writes=None,
        why=_WHY_ROUTER_LOGITS,
    ),
    _row("router_scores", tap="module output", requires=_MOE),
    _row(
        "expert_idx",
        tap="module output",
        requires=_MOE,
        writes=_SWAP_ONLY,
        why=_WHY_EXPERT_IDX,
    ),
    _row(
        "expert_permutation",
        tap="`.source` line (fused forward)",
        requires=_MOE,
        reads=_NNSIGHT,
        writes=None,
        why=_WHY_EXPERT_PERMUTATION,
    ),
    # the routed interior: the grouped experts dispatch is the reference
    # engine's tap and the only ragged ``expert:`` face served today; the
    # nnsight engine lands the token-major form through its ``.source`` table
    *(
        _row(
            name,
            tap="grouped-experts dispatch",
            requires=_ROUTED,
            expert_selection=_HOOKS,
        )
        for name in (
            "expert_gate_proj",
            "expert_up_proj",
            "expert_activation",
            "expert_neuron_output",
            "expert_output",
        )
    ),
    _row("routed_output", tap="module output", requires=_MOE),
    _row("shared_expert_gate_proj", tap="module output", requires=_SHARED),
    _row("shared_expert_up_proj", tap="module output", requires=_SHARED),
    _row("shared_expert_activation", tap="module input", requires=_SHARED),
    _row("shared_expert_output", tap="module output", requires=_SHARED),
    _row("shared_expert_gate", tap="module output", requires=_SHARED),
)

#: The registry: one row per name in the closed ``Component`` vocabulary, in
#: the vocabulary's order. ``tests/protocol/test_vocabulary_census.py`` holds
#: it to exactly ``set(COMPONENTS)``.
CAPABILITIES: Mapping[str, Capability] = MappingProxyType(
    {
        component: next(row for row in _ROWS if row.component == component)
        for component in COMPONENTS
    }
)
if len(_ROWS) != len(CAPABILITIES):  # pragma: no cover - the census names it
    raise AssertionError("a capability row names a component twice or not at all")


def capability(component: str) -> Capability:
    """The row for ``component``. Every caller already holds a name from the
    closed vocabulary (the parser rejects others), so a miss is a bug here,
    not a document error."""
    try:
        return CAPABILITIES[component]
    except KeyError:
        raise AssertionError(
            f"component {component!r} has no capability row: add one to "
            "registry.CAPABILITIES"
        ) from None


#: The mixer stream a component exists on, for the components that exist on
#: only one: a **view of the rows**, and the one table both halves of the
#: stream check read: the canonicalizer refuses against the registry entry's
#: ``layer_types`` at load, the engines' shared site resolver against the
#: module the layer really carries at run. A component absent here exists at
#: every layer.
COMPONENT_STREAMS: Mapping[str, Stream] = MappingProxyType(
    {c: row.stream for c, row in CAPABILITIES.items() if row.stream is not None}
)


#: The rows that carry per-family addresses: the attention interior.
INTERIOR_ROWS: tuple[str, ...] = tuple(
    c for c, row in CAPABILITIES.items() if row.overrides
)


def family_in_table(info: ModelInfo) -> bool:
    """Whether the per-family tap table has met ``info``'s family: some
    interior row carries an address for it. A family it has not met is served
    by measurement at run and decided nothing about at load."""
    return info.family is not None and any(
        info.family in CAPABILITIES[c].overrides for c in INTERIOR_ROWS
    )


def predicate_holds(info: ModelInfo, predicate: Predicate) -> bool | None:
    """Whether the registry entry can decide ``predicate``: ``None`` when the
    fact lives in the module tree and only the run can read it.

    ``moe`` and ``shared_expert`` read the entry's MoE widths. ``split_qkv``
    and ``gated_attention`` read the per-family tap table: for a family it has
    met, the interior is addressable (a fused projection included, through
    the row's declared slices) and the gate exists exactly when the
    ``attention_gate`` row has an address for the family; for a family it has
    not met, ``None``: the run measures. ``grouped_mm`` is a *load-time knob*
    (``experts_implementation``), not a config fact: an entry adapted from a
    loaded model's config carries it ([`ModelInfo.experts_implementation`][])
    and decides; a hand-declared entry leaves it ``None`` and the run's tap-time
    probe is the last-line check (implementation knobs are resolved during
    model-capability validation).
    """
    if predicate == "moe":
        return info.num_experts is not None
    if predicate == "shared_expert":
        return info.shared_expert_intermediate_size is not None
    if predicate == "grouped_mm":
        if info.experts_implementation is None:
            return None
        return info.experts_implementation == "grouped_mm"
    if predicate in ("split_qkv", "gated_attention"):
        if not family_in_table(info):
            return None
        if predicate == "split_qkv":
            return True
        return CAPABILITIES["attention_gate"].address_on(info) is not None
    return None


#: What each load-decidable predicate means, for the refusal.
_PREDICATE_MEANS: dict[str, str] = {
    "moe": "a sparse-MoE block (the entry declares no experts)",
    "shared_expert": "a shared expert (the entry declares no shared-expert width)",
    "grouped_mm": (
        "the grouped experts dispatch (the loaded model runs another "
        "experts_implementation: a different factorization whose "
        "intermediates are different tensors; load it with "
        "experts_implementation='grouped_mm', the default)"
    ),
    "split_qkv": "addressable q/k/v projections (the per-family tap table has none)",
    "gated_attention": (
        "an output gate on its attention mixer (the per-family tap table "
        "declares none for this family: only Qwen3.5/3.6's q-projection emits "
        "[q | gate] per head)"
    ),
}


def unavailable_at_load(info: ModelInfo, component: str) -> str | None:
    """Why ``component`` has no tensor on the model ``info`` describes, from
    the row's predicates the entry can decide: or ``None``. The run makes the
    same refusal from the module tree; this is the half ``validate`` can make
    offline, so a dense model's document naming ``routed_output`` is refused
    before a GPU is spent on it."""
    row = capability(component)
    for predicate in PREDICATES:
        if predicate in row.requires and predicate_holds(info, predicate) is False:
            if predicate == "grouped_mm":
                # the knob, not the architecture: the tensor would exist under
                # the default dispatch, so the refusal names the knob to turn
                return (
                    f"component {component!r} needs {_PREDICATE_MEANS[predicate]}; "
                    f"model {info.key!r} was loaded with experts_implementation="
                    f"{info.experts_implementation!r}"
                )
            return (
                f"component {component!r} needs {_PREDICATE_MEANS[predicate]}, "
                f"which model {info.key!r} does not have: there is no such "
                "tensor on this model"
            )
    return None


def expert_axis_refusal(component: str, expert: Any) -> str:
    """Why ``expert`` does not apply to ``component``: shared by the validator
    (which refuses at load) and the site resolver (which refuses if a document
    arrives unvalidated), so the two cannot describe the absent axis
    differently."""
    faces = [c for c, row in CAPABILITIES.items() if row.expert_selection]
    listed = ", ".join(f"'{c}'" for c in faces[:-1]) + f" and '{faces[-1]}'"
    return (
        f"site names expert {expert!r} on component {component!r}, which has no "
        "per-expert axis: the router's axes are all-experts or top-k, and the "
        "shared expert is not one of the routed experts. The per-expert "
        f"interior components are {listed}."
    )


def write_policy_refusal(ename: str, component: str, mechanism: str) -> str | None:
    """Why ``mechanism`` may not be written to ``component``: or ``None`` when
    it may. **The** write policy: the validator applies it at load and the
    executor at the plan, so the two cannot disagree about what a write may do.
    A read-only row refuses every mechanism; a restricted row refuses the
    mechanisms outside its set. The text names the alternative (``why``)."""
    row = capability(component)
    if row.writes is None:
        return (
            f"write {ename!r} targets {component!r}, which no write may change: "
            f"{row.why}. Refusing at the plan, before anything runs."
        )
    if mechanism not in row.writes:
        return (
            f"write {ename!r} applies {mechanism!r} to {component!r}, which only "
            f"a whole-value 'swap' may change: {row.why}."
        )
    return None


# --------------------------------------------------------------------------- #
# the per-family tap table: native packing → shape, and its rendering
# --------------------------------------------------------------------------- #


def native_shape(address: Mapping[str, Any], value: FeatureShape) -> FeatureShape:
    """The native tensor the address's module emits, as a shape: the
    component's family-independent ``value`` shape ([`component_shape`][],
    ``(batch, position, head·feature)``) re-packed the way the row says this
    family's module packs it. The executor converts by it in both directions
    ([`causalab.neural.shared.layout`][]), so a module whose real tensor
    disagrees raises rather than being reinterpreted."""
    head = next(a for a in value.axes if a.kind == "head")
    feature = next(a for a in value.axes if a.kind == "feature")
    assert head.width is not None and feature.width is not None
    packing = address["packing"]
    if packing == "flat":
        return dataclasses.replace(value, flat_inner=True)
    if packing == "head_axis":
        return dataclasses.replace(value, flat_inner=False)
    if packing == "fused_heads":
        return shapes.bs_fused_heads(
            head.width, address["splits"], address["split"], feature.width
        )
    assert packing == "fused_blocks", packing
    return shapes.bs_fused_blocks(
        address["splits"], address["split"], head.width, feature.width
    )


def families_in_table() -> tuple[str, ...]:
    """Every family some interior row has an address for, sorted."""
    return tuple(sorted({f for c in INTERIOR_ROWS for f in CAPABILITIES[c].overrides}))
