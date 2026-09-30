"""Resolve nnsight source addresses for function interiors.

Tables store module, operation pattern, peel chain, field, fire count,
and required implementation by stream. Matching uses substrings and
requires one result, reporting the operation inventory on ambiguity.
When an assignment and call share a name, the source line identifies the
call. The executor follows the resolved address inside a trace.

This module stays independent of causalab imports. Shared execution handles
protocol sites, layouts, and writes. Addresses identify post-RoPE q/k,
masked scores, probabilities, and attention output; delta kernel addresses
include the hub wrapper's ``implementation_0`` call.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, Iterable, Mapping

__all__ = [
    "ADDRESSES",
    "GENERATED_ADDRESSES",
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "MOE_EXPERTS",
    "AddressResolutionError",
    "SourceAddress",
    "match_op",
]


class AddressResolutionError(ValueError):
    """A pattern did not resolve to exactly one op.

    A plain ``ValueError`` on purpose: this module knows nothing of the
    protocol's error vocabulary. The executor wraps it with the component,
    layer and library version before it reaches a document author.
    """


@dataclasses.dataclass(frozen=True)
class SourceAddress:
    """One interior tensor, addressed through ``.source``.

    ``module`` is the child path under the layer the ops live on (e.g.
    ``"self_attn"``) — documentation and upstreaming data; the executor
    already holds the resolved envoy and navigates from it.
    """

    module: str
    #: Substring matched against ``source.names`` — NEVER a hardcoded ``_n``
    #: suffix for a symbol that appears once (the suffix is what drifts).
    op_pattern: str
    #: Call ops to drill *through*, one ``.source`` level per element, each
    #: matched by the same substring rule. 📐 ``("implementation_0",)`` is
    #: required on both delta kernels on transformers 5.16.1.
    peel: tuple[str, ...] = ()
    #: The assignment/op *inside* the drilled source that carries the value,
    #: e.g. ``"attn_weights_1"`` — same substring rule. ``None`` means the
    #: matched op's own output is the value.
    field: str | None = None
    #: ``(positional_index, keyword)`` into the op's ``inputs`` instead of its
    #: output — how a kernel's in-place-updated argument is reached (the delta
    #: kernels' ``initial_state``).
    arg: tuple[int, str] | None = None
    #: Which element of a tuple-valued output the component means.
    tuple_index: int | None = None
    #: How often the op fires per forward: ``"once"`` | ``"per_chunk"`` |
    #: ``"per_step"`` | ``"per_expert"``. Everything but ``"once"`` needs
    #: ``tracer.iter`` loop machinery.
    fires: str = "once"
    #: Implementation switches the address is only valid under —
    #: ``{"attn_eager"}``: the fused kernels never materialize the tensor;
    #: ``{"experts_grouped"}``: the grouped experts kernel is where the
    #: per-expert interior's ops live.
    requires: frozenset[str] = frozenset()
    #: The value's rows are expert rows — ``(batch·position·top_k, …)`` — and
    #: the executor re-packs them token-major to the declared 2-D native
    #: shape ``(batch·position, top_k·…)``. Pure row bookkeeping; the
    #: declared [`FeatureShape`][causalab.protocol.registry.shapes.FeatureShape] stays the semantic description.
    expert_rows: bool = False
    #: Op pattern (same substring rule, matched on the same drilled source as
    #: the value) of the ``torch.sort`` whose ``output[1]`` maps sorted rows →
    #: token-major rows. When set, the value's rows are in the kernel's
    #: expert-sorted order and the executor un-sorts reads / re-sorts writes
    #: through it — the sorted layout is grouped_mm bookkeeping, never the
    #: component's meaning.
    align: str | None = None
    #: For a per-fire address (``fires != "once"``): op pattern, in the same
    #: drilled source as ``field``, of the loop's own ``range(...)`` — the
    #: length of its output is the fire count. 📐 Read off the loop itself,
    #: never off config (the kernel pads to a chunk multiple, so the count is
    #: the kernel's fact, not the sequence length's).
    trip: str | None = None


_OP_SUFFIX = re.compile(r"_\d+$")


def match_op(
    pattern: str,
    names: Iterable[str],
    line_of: Callable[[str], str] | None = None,
) -> str:
    """The one op ``pattern`` names, or a refusal carrying the inventory.

    Substring match over ``names``. When several ops match — the systematic
    case is a variable assigned and then called, both ops named after it —
    the hits whose own source line *calls* the matched symbol (the op's name
    minus its positional suffix, immediately followed by ``(``) are preferred,
    which ``line_of`` makes possible; anything still ambiguous refuses rather
    than guessing.
    """
    all_names = list(names)
    hits = [n for n in all_names if pattern in n]
    if len(hits) > 1 and line_of is not None:
        calls = [n for n in hits if f"{_OP_SUFFIX.sub('', n)}(" in line_of(n)]
        if calls:
            hits = calls
    if len(hits) == 1:
        return hits[0]
    what = "no op matches" if not hits else f"{len(hits)} ops match ({hits})"
    raise AddressResolutionError(
        f"pattern {pattern!r}: {what}. The installed library's forward names "
        f"these ops: {all_names}. A missing or ambiguous pattern usually means "
        "a transformers release moved this forward's code — re-verify the "
        "address table against the new source."
    )


# --------------------------------------------------------------------------- #
# the tables, keyed by the shared stream vocabulary
# --------------------------------------------------------------------------- #

#: The full-attention mixer's interior. All five live in ``self_attn``'s
#: forward or inside its ``attention_interface(...)`` call. ``attention_z`` is
#: the call's own return (``output[0]``, already ``(b, s, H, d)``) — the
#: drilled ``attn_output_0`` is the pre-transpose ``(b, H, s, d)`` tensor, a
#: different box. Only the softmax's neighbourhood needs eager: q, k and z
#: exist under every implementation.
FULL_ATTENTION: dict[str, SourceAddress] = {
    "attention_query": SourceAddress(
        module="self_attn",
        op_pattern="apply_rotary_pos_emb",
        tuple_index=0,
    ),
    "attention_key": SourceAddress(
        module="self_attn",
        op_pattern="apply_rotary_pos_emb",
        tuple_index=1,
    ),
    "attention_scores": SourceAddress(
        module="self_attn",
        op_pattern="attention_interface",
        # ⚠️ `_1`, the post-mask softmax input — not `_0` (pre-mask). The
        # component is *defined* as the softmax's input (softmax(scores) ==
        # pattern, pinned exact); the pre-mask tensor is a different box.
        field="attn_weights_1",
        requires=frozenset({"attn_eager"}),
    ),
    "attention_probs": SourceAddress(
        module="self_attn",
        op_pattern="attention_interface",
        # the softmax's output, read AND written here: a write is consumed by
        # the value multiply downstream (#53's finding), where a write to the
        # mixer's returned attn_weights would reach nothing.
        field="attn_weights_2",
        requires=frozenset({"attn_eager"}),
    ),
    "attention_z": SourceAddress(
        module="self_attn",
        op_pattern="attention_interface",
        tuple_index=0,
    ),
}

#: The Gated DeltaNet interior — 30 of the 40 target layers, and the reason
#: for source-level addresses: none of these tensors crosses a module boundary.
#:
#: 📐 Measured on ``tiny-random/qwen3.5-moe`` and the real A3B (transformers
#: 5.16.1): the mixer projects ``mixed_qkv`` and the gate ``z`` first, runs the
#: causal conv (channels-first), splits into q/k/v (pre ``repeat_interleave``,
#: so q and k are in *key-head* space), computes ``beta = σ(b)`` and the decay
#: ``g``, and hands everything to the chunked delta kernel — whose own
#: ``.source`` needs the ``implementation_0`` peel (the hub-kernel-with-
#: fallback wrapper), the first real use of the peel chain. In prefill the
#: kernel advances the recurrent state once per 64-token chunk
#: (``last_recurrent_state_1``; ``_0`` is the zero init), so the state fires
#: ``per_chunk`` with the trip count read off the loop's own ``range_1``
#: (``range_0`` is the intra-chunk loop). The *recurrent* kernel — per-token
#: states — runs only at ``seq_len == 1`` under a cache: decode-only by the
#: modeling code's own dispatch, with no switch to force it in prefill
#: (modeling_qwen3_5_moe.py:507), so per-token prefill state is refused by
#: name rather than served at a granularity the kernel does not have.
#:
#: (The gate's ``z_reshape_0`` view and the post-norm ``core_attn_out_reshape_1``
#: flatten used to be addressed here; both are module boundaries — ``in_proj_z``'s
#: output and ``out_proj``'s input — and land on envoys now.)
LINEAR_ATTENTION: dict[str, SourceAddress] = {
    # The three module boundaries of this mixer (`delta_qkv` = in_proj_qkv's
    # output, `delta_gate` = in_proj_z's output, `delta_premix` = out_proj's
    # input) need no entry: envoys serve them, as every module boundary. The
    # kernel boundary below is keyed by the protocol's one name per tensor
    # (the `deltanet_*` spellings that named the same tensors are
    # aliases now — `deltanet_qkv_conv`, `deltanet_value`, `deltanet_beta`,
    # `deltanet_decay`, `deltanet_core_out` fold onto these at parse); the two
    # pre-tiling faces and the per-chunk state keep their own names because
    # their tensors differ from the reference engine's in shape or timing
    # (`registry.BACKEND_PAIRS`).
    "delta_conv": SourceAddress(
        module="linear_attn",
        # ⚠️ channels-first (b, width, s) — the declared shape carries it
        op_pattern="causal_conv1d_fn",
    ),
    "deltanet_query": SourceAddress(
        module="linear_attn",
        op_pattern="query_reshape",
    ),
    "deltanet_key": SourceAddress(
        module="linear_attn",
        op_pattern="key_reshape",
    ),
    "delta_value": SourceAddress(
        module="linear_attn",
        op_pattern="value_reshape",
    ),
    "delta_beta": SourceAddress(
        module="linear_attn",
        op_pattern="b_sigmoid",
    ),
    "delta_decay": SourceAddress(
        module="linear_attn",
        # the kernel's own `g=` argument. An op's inputs must be requested
        # before anything drills into its source (measured: OutOfOrderError
        # otherwise), which the rank table's order guarantees.
        op_pattern="chunk_gated_delta_rule",
        arg=(1, "g"),
    ),
    "deltanet_state": SourceAddress(
        module="linear_attn",
        op_pattern="chunk_gated_delta_rule",
        peel=("implementation_0",),
        field="last_recurrent_state_1",
        fires="per_chunk",
        trip="range_1",
    ),
    "delta_kernel_output": SourceAddress(
        module="linear_attn",
        op_pattern="chunk_gated_delta_rule",
        tuple_index=0,
    ),
}

#: The per-expert MoE interior. Not a mixer stream: the ops live under
#: ``mlp.experts``, so the executor keys into this table by component (a
#: ``kind="interior"`` site) rather than by ``stream_at``.
#:
#: All five live inside the grouped experts kernel (``experts_forward`` is the
#: dispatch's call — the same assigned-then-called ambiguity as
#: ``attention_interface``, resolved the same way). 📐 Measured on
#: ``tiny-random/qwen3.5-moe`` and matching the real A3B's inventory: the
#: kernel sorts the ``(token, slot)`` rows by expert (``torch_sort``), runs
#: the fused gate_up projection (``proj_out_0``; ``_apply_gate`` splits and
#: gates it), down-projects (``proj_out_3``), un-sorts (``inv_perm``) and
#: weights. The sorted layout is bookkeeping, so every sorted-space value
#: carries ``align`` and is presented token-major.
MOE_EXPERTS: dict[str, SourceAddress] = {
    # The two halves of the fused [gate_e | up_e] projection are ONE capture —
    # the first _grouped_linear's return (`proj_out_0`, pre-chunk) — with two
    # addresses through the declared fused axis: the same one-capture
    # presentation the reference engine's dispatch wrapper serves.
    # The `_0` suffix is load-bearing the way `inv_perm_1`'s is: `proj_out` is
    # reassigned down the forward (`proj_out_3` is the down-projection).
    "expert_gate_proj": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        field="proj_out_0",
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_up_proj": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        field="proj_out_0",
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_activation": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        # the act call INSIDE _apply_gate: act_fn(gate) alone, before the
        # `· up` multiply — the same tensor `mlp_activation` names on the
        # llama family (the registry's semantics; the _apply_gate
        # call's own return is act(gate)·up, a different tensor).
        peel=("self__apply_gate",),
        field="self_act_fn",
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_neuron_output": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        field="self__apply_gate",
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_output": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        # the down-projection's return (`proj_out_3`), BEFORE the routing
        # weight — the registry identity: routed_output == the slot-sum of
        # expert_output · router_scores. Still expert-sorted at this line,
        # hence the align (the weighting and un-sort happen downstream).
        field="proj_out_3",
        expert_rows=True,
        align="torch_sort",
        requires=frozenset({"experts_grouped"}),
    ),
    "expert_permutation": SourceAddress(
        module="mlp.experts",
        op_pattern="experts_forward",
        # the kernel's own inverse permutation — token-major by construction,
        # so no align. The `_1` suffix is load-bearing: `inv_perm_0` is the
        # empty_like allocation, `_1` the filled table, and neither line is a
        # call, so the call-op rule cannot separate them.
        field="inv_perm_1",
        expert_rows=True,
        requires=frozenset({"experts_grouped"}),
    ),
}

#: Every table, keyed the way [`causalab.neural.shared.model_tree.stream_at`][]
#: answers — the executor's single lookup point.
ADDRESSES: Mapping[str, Mapping[str, SourceAddress]] = {
    "full_attention": FULL_ATTENTION,
    "linear_attention": LINEAR_ATTENTION,
}

#: Interior addresses in the **generated frame**, where decode dispatches
#: different code than prefill. 📐 The one entry so far is the reason the table
#: exists: at ``seq_len == 1`` under a cache the DeltaNet mixer runs the
#: *recurrent* kernel (``modeling_qwen3_5_moe.py:507``) — a different function
#: from the chunked one the prompt-frame address drills — and its state
#: assignment fires once per decode step (``last_recurrent_state_2``; ``_0``
#: is the zeros fallback, ``_1`` the untaken no-cache branch). An interior
#: component absent here does not exist in the decode path (the chunk kernel,
#: the conv's prefill branch, the grouped experts' sort are prefill facts) or
#: has not been verified there — refused by name either way.
GENERATED_ADDRESSES: Mapping[str, Mapping[str, SourceAddress]] = {
    "linear_attention": {
        "deltanet_state": SourceAddress(
            module="linear_attn",
            op_pattern="recurrent_gated_delta_rule",
            peel=("implementation_0",),
            field="last_recurrent_state_2",
            fires="per_step",
        ),
    },
}
