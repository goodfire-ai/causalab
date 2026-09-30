"""Read family structure and capabilities from loaded model modules.

The registry's family adapter supplies mixer child names. This module
builds each layer's stream entry from its children and rejects a block
that contains both mixer types. Capability probes inspect sparse experts,
shared experts, grouped projections, split QKV, and gated attention.
Measured address helpers use the module names declared in registry rows.
``sites`` consumes these facts to resolve taps.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import (
    CAPABILITIES,
    COMPONENT_STREAMS,
    FAMILIES,
    INTERIOR_ROWS,
    PREDICATES,
    FamilyAdapter,
    capability,
    component_shape,
    family_for,
    family_in_table,
    mixer_children,
    walk,
)
from causalab.protocol.schema import SiteSpec, Stream

__all__ = [
    "FULL_ATTENTION_CHILDREN",
    "LINEAR_ATTENTION_CHILDREN",
    "adapter_of",
    "mixer_at",
    "stream_at",
]


def _built_in(stream: str) -> tuple[str, ...]:
    """The mixer children the built-in families declare for ``stream``, in
    declaration order and each once — the view a reader of this module
    expects. GPT-2 and GPT-J both name their mixer ``attn``."""
    return tuple(
        dict.fromkeys(
            child
            for adapter in FAMILIES.values()
            for child, declared in adapter.mixers.items()
            if declared == stream
        )
    )


#: The mixer children that mean a layer runs full (softmax) attention, and the
#: ones that mean it runs a linear-attention kernel — as the built-in families
#: declare them (``registry.FamilyAdapter.mixers``). A family registered
#: later adds its own child names to the table [`stream_at`][] reads
#: (``registry.mixer_children``), not to these two constants.
FULL_ATTENTION_CHILDREN: tuple[str, ...] = _built_in("full_attention")
LINEAR_ATTENTION_CHILDREN: tuple[str, ...] = _built_in("linear_attention")


def stream_at(
    blocks: Any,
    layer: int,
    *,
    key: str,
    mixers: Mapping[str, Stream] | None = None,
    layer_types: Sequence[Stream] | None = None,
) -> str:
    """Which mixer stream ``blocks[layer]`` actually carries.

    Returns one of ``"full_attention"`` (a ``self_attn``/``attn`` child) or
    ``"linear_attention"`` (a ``linear_attn`` child), reading the child →
    stream table every registered family declares (``mixers`` overrides it:
    a bundle passes its own family's declaration). ``key`` names the model
    in refusals.

    A block this rank does not hold — a pipeline stage's bare identity
    (``docs/model_parallelism.md`` §6.5), with no mixer child of either
    kind — is answered from ``layer_types``, the registry entry's declared
    pattern, when the bundle passes one: the torch-free answer for a layer
    whose module lives on another rank. A stand-in that shadows the block it
    replaced still shows its children and is read like any other block.

    Raises:
        ProtocolError: the block has no recognised mixer child, or has
            children of *both* kinds. The second case is hypothetical — no
            built-in family ships it — but probing in a fixed
            order would answer "full_attention" for it silently, and every
            per-layer tap downstream would then attach to the wrong module
            and still produce plausible numbers. A named refusal is the same
            trade this vocabulary makes everywhere else — and the template
            the family predicates follow (``registry.family_for``).
    """
    table = mixer_children() if mixers is None else mixers
    block = blocks[layer]
    full = [
        name
        for name, s in table.items()
        if s == "full_attention" and hasattr(block, name)
    ]
    linear = [
        name
        for name, s in table.items()
        if s == "linear_attention" and hasattr(block, name)
    ]
    if full and linear:
        raise ProtocolError(
            "P4",
            f"layer {layer} of {key!r} carries both a full-attention "
            f"child ({', '.join(full)}) and a linear-attention child "
            f"({', '.join(linear)}) — the stream of a layer must be one or "
            "the other, so extend the stream table in "
            "neural/shared/model_tree.py to say which this family means",
        )
    if full:
        return "full_attention"
    if linear:
        return "linear_attention"
    if _absent(block) and layer_types is not None and 0 <= layer < len(layer_types):
        return layer_types[layer]
    raise ProtocolError(
        "P4",
        f"layer {layer} of {key!r} has no recognised mixer child "
        f"(children={sorted(name for name, _ in block.named_children())}) — "
        "extend the stream table in neural/shared/model_tree.py",
    )


def _absent(block: Any) -> bool:
    """A block this rank holds nothing of: a bare identity with no children."""
    return isinstance(block, torch.nn.Identity) and not list(block.named_children())


def mixer_at(
    blocks: Any,
    layer: int,
    *,
    key: str,
    mixers: Mapping[str, Stream] | None = None,
    layer_types: Sequence[Stream] | None = None,
) -> Any:
    """The attention/mixer module at ``layer``, whichever stream it is.

    Resolved *through* [`stream_at`][] rather than by its own probe, so the
    two can never disagree about a block: one answer, one place.

    Raises:
        ProtocolError: ``P4`` — the layer's stream is known (``layer_types``)
            but this rank holds no module for it: a bare identity standing in
            for another pipeline stage's block.
    """
    table = mixer_children() if mixers is None else mixers
    stream = stream_at(blocks, layer, key=key, mixers=table, layer_types=layer_types)
    block = blocks[layer]
    for name, declared in table.items():
        if declared == stream:
            child = getattr(block, name, None)
            if child is not None:
                return child
    raise ProtocolError(
        "P4",
        f"layer {layer} of {key!r} carries {stream!r}, but this rank does not hold "
        "its mixer: the block is another pipeline stage's "
        "(docs/model_parallelism.md §6.5)",
    )


def adapter_of(bundle: Any) -> FamilyAdapter:
    """The family plugin serving ``bundle``'s model: the bundle's own
    (detected once, ``ModelBundle.adapter``) or, for a bundle without the
    attribute, detected here (``registry.family_for``) — structurally, never
    off the config."""
    adapter = getattr(bundle, "adapter", None)
    return adapter if adapter is not None else family_for(bundle.model)


def _blocks(bundle: Any) -> Any:
    return adapter_of(bundle).blocks_of(bundle.model)


def _attn(bundle: Any, layer: int) -> Any:
    """The mixer at ``layer`` — ``self_attn``, ``attn`` or ``linear_attn``.

    Was ``block.self_attn`` for every non-GPT-2 model, which AttributeErrors on
    a hybrid tower: 📐 on ``tiny-random/qwen3.5-moe`` three of four layers carry
    ``linear_attn`` (Gated DeltaNet) and only one carries ``self_attn``. The
    per-layer answer lives on the bundle (§5.2)."""
    return bundle.mixer_at(layer)


#: Components that only exist on a full-attention mixer. A Gated DeltaNet layer
#: has no attention matrix at all — there is nothing to read and nothing to
#: write — so naming one at such a layer is an error about the *architecture*,
#: not a missing feature (§5.3).
#: 🐞 ``attention_premix`` and ``attention_result`` belong here too, and did not
#: before. Both are the o-projection's input, and 📐 a Gated DeltaNet layer has
#: no ``o_proj`` at all — its children are
#: ``[conv1d, in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, norm, out_proj]`` —
#: so naming either at such a layer raised a bare
#: ``AttributeError: 'Qwen3_5MoeGatedDeltaNet' object has no attribute 'o_proj'``
#: out of the tap table instead of the architectural refusal that says why the
#: box does not exist there. ``attention_output`` is deliberately *not* here: a
#: DeltaNet layer does produce a mixer output, and it resolves.
#: Read off the capability rows' ``stream`` cell (``registry.COMPONENT_STREAMS``
#: is their view) rather than declared again here, because the canonicalizer
#: refuses from the same rows against the registry's ``layer_types`` — two
#: tables would be two answers.
_FULL_ATTENTION_ONLY: frozenset[str] = frozenset(
    component
    for component, stream in COMPONENT_STREAMS.items()
    if stream == "full_attention"
)

# The mirror: components that only exist on a Gated DeltaNet mixer.
# A full-attention layer computes no delta-rule state — its mixer has no
# ``in_proj_qkv``/``in_proj_z``/``out_proj`` children at all — and a family
# with no linear stream anywhere (llama, gpt2) hits the same refusal at every
# layer, which is the architectural refusal by name.
# The kernel boundary *inside* the DeltaNet forward is
# ``DELTA_KERNEL_SLOTS`` (the registry's, re-exported). 📐 These are not
# module boundaries: the forward calls two module-global functions
# (``causal_conv1d_fn`` and the delta-rule kernel), so the taps swap those
# globals for the dynamic extent of the tapped mixer's forward; the per-step
# interior is produced by stepping the library's own recurrent
# kernel in the chunked call's shadow. See
# [`causalab.neural.engines.pytorch_hooks.delta_interface`][]. The nnsight
# engine lands the same names as ``.source`` lines of the fused forward.

#: The mirror set: the Gated DeltaNet interior only exists on a
#: linear-attention mixer — a softmax-attention layer has no recurrent state,
#: no delta kernel and no causal conv, so naming one of these there is the
#: same architectural error in the other direction. It is the part of the
#: protocol's linear-attention components the reference engine does **not**
#: serve (``deltanet_query`` / ``deltanet_key`` / ``deltanet_state``: the
#: pre-tiling and per-chunk faces, ``.source`` lines inside the fused
#: forward); ``_LINEAR_ATTENTION_ONLY`` is the part it does — the module and
#: kernel boundaries, which both engines serve under one name.
#: Both read off the rows.
_DELTANET_INTERIOR: frozenset[str] = frozenset(
    component
    for component, stream in COMPONENT_STREAMS.items()
    if stream == "linear_attention"
    and "pytorch_hooks" not in CAPABILITIES[component].reads
)

_LINEAR_ATTENTION_ONLY: frozenset[str] = (
    frozenset(
        component
        for component, stream in COMPONENT_STREAMS.items()
        if stream == "linear_attention"
    )
    - _DELTANET_INTERIOR
)


def _check_stream(bundle: Any, component: str, spec: SiteSpec, layer: int) -> None:
    """Refuse a site whose stream the layer does not carry, before hooking.

    Two ways to get this wrong, and both are caught here rather than as an
    AttributeError from inside a hook:

    * the site *declares* a ``stream`` the layer does not have — ``stream`` has
      parsed since ``schema.py`` gained it and nothing read it until now (§5.2);
    * the site names a full-attention-only component at a linear-attention
      layer, which no ``stream`` spelling can make true (§5.3).
    """
    actual = bundle.stream_at(layer)
    declared = spec.stream if isinstance(spec.stream, str) else None
    if declared is not None and declared != actual:
        raise ProtocolError(
            "P4",
            f"site names stream {declared!r} at layer {layer}, but that layer "
            f"carries {actual!r} — this is a hybrid tower ({', '.join(bundle.streams)}), "
            "so the stream is a per-layer fact, not a model-wide one",
            reason="component_unavailable",
        )
    if component in _FULL_ATTENTION_ONLY and actual != "full_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a full-attention mixer, but layer "
            f"{layer} of {bundle.key!r} carries {actual!r} — a Gated DeltaNet "
            "block computes no attention matrix, so there is no such tensor at "
            f"this layer. This tower is ({', '.join(bundle.streams)}).",
            reason="component_unavailable",
        )
    if component in _LINEAR_ATTENTION_ONLY and actual != "linear_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a Gated DeltaNet (linear-attention) "
            f"mixer, but layer {layer} of {bundle.key!r} carries {actual!r} — a "
            "gated-attention mixer computes no delta-rule state, so there is no "
            f"such tensor at this layer. This tower is "
            f"({', '.join(bundle.streams)}).",
            reason="component_unavailable",
        )
    if component in _DELTANET_INTERIOR and actual != "linear_attention":
        raise ProtocolError(
            "P4",
            f"component {component!r} needs a Gated DeltaNet mixer, but layer "
            f"{layer} of {bundle.key!r} carries {actual!r} — a softmax-attention "
            "block computes no recurrent state and runs no delta kernel, so "
            "there is no such tensor at this layer. This tower is "
            f"({', '.join(bundle.streams)}).",
            reason="component_unavailable",
        )


def _projection_width(module: Any) -> int | None:
    """The output width a projection module declares — ``nn.Linear``'s
    ``out_features``, GPT-2's ``Conv1D.nf`` — or ``None`` for a module that
    declares none (a norm: its output has its input's shape, and the layout
    conversion checks that tensor at hook time)."""
    for attr in ("out_features", "nf"):
        width = getattr(module, attr, None)
        if isinstance(width, int):
            return width
    return None


def _tensor_shards(bundle: Any, module: Any) -> int:
    """How many ways a tensor-parallel colwise row splits ``module``'s output
    across the ranks of its group — the group's size when the module carries
    a ``colwise`` / ``packed_colwise`` row on an axis above one, else 1."""
    plan = getattr(getattr(bundle, "info", None), "parallel_plan", None)
    geometry = getattr(bundle, "geometry", None)
    if plan is None or geometry is None or int(getattr(geometry, "world", 1)) == 1:
        return 1  # at world 1 the plan is never read
    from causalab.neural.shared.parallel.placements import module_path

    # the plan as applied under this geometry: a K/V projection replicated
    # above the KV heads (§6.6) carries no colwise row and emits its whole width
    row = plan.for_geometry(geometry, bundle.info).style_for(
        module_path(bundle.model, module),
        prefix=getattr(bundle.model, "base_model_prefix", None),
    )
    if row is None or row.style not in ("colwise", "packed_colwise"):
        return 1
    return int(getattr(geometry, row.axis, 1))


def _check_projection_width(
    bundle: Any, module: Any, address: Mapping[str, Any], component: str, layer: int
) -> None:
    """The width rule, by name and before any hook: a projection the row
    addresses must emit exactly ``splits × (heads · head_dim)`` in the
    component's own head space (📐 ``H·d = 16`` on llama, ``H·2·d = 512`` on
    qwen3.5-moe's gated q-projection, ``3·H·d = 96`` on GPT-2's ``c_attn``) —
    or, when the module carries a tensor-parallel colwise row on an active
    axis (``docs/model_parallelism.md`` §6.2), that width over the group,
    the local shard's. Any other width is refused naming the accepted ones.
    The layout conversion would catch the same disagreement at hook time as
    an internal error; this names the row and the module instead."""
    out = _projection_width(module)
    if out is None:
        return
    value = component_shape(bundle.info, component)
    assert value.width is not None  # every interior component has a feature axis
    splits = int(address.get("splits", 1))
    declared = splits * value.width
    shards = _tensor_shards(bundle, module)
    where = f"component {component!r} at layer {layer} of {bundle.key!r}"
    if declared % shards:
        raise ProtocolError(
            "P4",
            f"{where}: {address['module']!r} emits {declared} features "
            f"({value.describe()}), which tp={shards} does not divide — the "
            "tensor group cannot hold whole heads of it",
            reason="component_unavailable",
        )
    accepted = {declared} if shards == 1 else {declared, declared // shards}
    if out in accepted:
        return
    local = (
        f", or its local shard's {declared // shards} under tp={shards} (the module "
        f"carries a tensor-parallel row)"
        if shards > 1
        else ""
    )
    raise ProtocolError(
        "P4",
        f"{where}: the per-family tap table says {address['module']!r} emits "
        f"{declared} features on family {bundle.info.family!r} ({splits} × "
        f"{value.width}, {value.describe()}){local}, but this module emits "
        f"{out}. The row and the loaded module disagree — re-measure the family "
        "(causalab/protocol/registry.py, the interior rows' overrides) before "
        "trusting either.",
        reason="component_unavailable",
    )


def _declared_modules(component: str, *, packings: frozenset[str]) -> list[str]:
    """The module names the rows declare for ``component`` under ``packings``,
    across every family — the vocabulary the measured fallback picks from."""
    return sorted(
        {
            address["module"]
            for address in capability(component).overrides.values()
            if address["packing"] in packings
        }
    )


_UNFUSED: frozenset[str] = frozenset({"flat", "head_axis"})


def _has_separate_projections(attn: Any) -> bool:
    """Measured: for each of q, k and v the mixer carries one of the bare
    projections some family's row names (📐 ``q_proj``/``k_proj``/``v_proj``
    on the llama tree). GPT-2's mixer carries none — only ``c_attn``."""
    return all(
        any(
            hasattr(attn, module)
            for module in _declared_modules(component, packings=frozenset({"flat"}))
        )
        for component in INTERIOR_ROWS
        if component != "attention_gate"
    )


def _measured_address(
    bundle: Any, attn: Any, component: str, layer: int
) -> Mapping[str, Any]:
    """The address of ``component`` on a mixer whose family the per-family tap
    table has **not** met — today's measured behaviour, kept exactly, but
    picking among the addresses the rows declare rather than a local table.

    * a norm after the projection wins where the mixer has one (📐 measured on
      qwen3.5-moe: ``q_norm``/``k_norm`` run before ``apply_rotary_pos_emb``,
      so their output *is* the pre-RoPE tensor, ``(b, s, H, d)``);
    * else the bare projection, whose width must be the value's — a projection
      emitting two splits per head with no norm to tap after it is refused
      (its output is not the queries alone), as is one of neither width;
    * a **block order is never inferred**: which contiguous block of a fused
      ``c_attn`` is q is a family fact only a row can state, so a family with
      a fused projection and no row is refused (``_probe_split_qkv``).
    """
    row = capability(component)
    present = [
        address
        for address in row.overrides.values()
        if address["packing"] != "fused_blocks" and hasattr(attn, address["module"])
    ]
    norms = [a for a in present if a["packing"] == "head_axis"]
    if norms:
        return norms[0]
    if not present:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: family "
            f"{bundle.info.family!r} has no row in the per-family tap table, and "
            f"this mixer (children={_children(attn)}) carries none of the modules "
            f"the rows declare for it ({_declared_modules(component, packings=_UNFUSED)}). "
            "Measure the family and add its addresses to the interior rows' "
            "overrides in causalab/protocol/registry.py.",
            reason="component_unavailable",
        )
    address = present[0]
    module = getattr(attn, address["module"])
    value = component_shape(bundle.info, component)
    assert value.width is not None
    out = _projection_width(module)
    splits = int(address.get("splits", 1))
    if out is not None and out == 2 * value.width and splits == 1:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: this "
            f"mixer has no norm after {address['module']!r}, so the projection's "
            "output would have to be the pre-RoPE tensor — but that projection "
            "is fused ([q | gate] per head), so its output is not the queries "
            "alone. Addressing a split of a projection with no norm to tap after "
            "it is a row of the per-family tap table: measure the "
            "family and add it.",
            reason="component_unavailable",
        )
    if out is not None and out not in (value.width, 2 * value.width):
        raise ProtocolError(
            "P4",
            f"the projection {address['module']!r} at layer {layer} of "
            f"{bundle.key!r} emits {out} features, which is neither "
            f"{value.width} (heads·head_dim) nor {2 * value.width} (a gated "
            "family's [q | gate] per head). This backend cannot say which columns "
            f"are the {component!r} — measure the family and add a row to the "
            "per-family tap table (causalab/protocol/registry.py).",
            reason="component_unavailable",
        )
    return address


def _experts_implementation(bundle: Any) -> str:
    """The experts implementation the loaded model dispatches on — read from
    the config the modeling code itself reads."""
    config = getattr(bundle.model.config, "text_config", None) or bundle.model.config
    return str(getattr(config, "_experts_implementation", "<undeclared>"))


# --------------------------------------------------------------------------- #
# the architectural predicates — the module-tree half of `Capability.requires`
# --------------------------------------------------------------------------- #
#
# A row declares what a component *needs* (``registry.PREDICATES``); this is
# where each predicate is read off the loaded modules, and what the refusal
# says when it does not hold. The canonicalizer evaluates the ones the registry
# entry can decide (``moe``, ``shared_expert``) at load; every one is evaluated
# here at run, so a document arriving unvalidated is refused by the same rows.
# The texts are the ones the per-branch checks this replaces carried (the
# refusal snapshot pins them); the three that were ``NotImplementedError`` are
# protocol refusals now, which is what they always described.


def _mlp(bundle: Any, layer: int) -> Any:
    """The block's MLP child, as the family's tree names it."""
    block = _blocks(bundle)[layer]
    mlp = walk(block, adapter_of(bundle).tree.mlp)
    if mlp is None:
        raise ProtocolError(
            "P4",
            f"layer {layer} of {bundle.key!r}: family {adapter_of(bundle).family!r} "
            f"names the block's MLP {adapter_of(bundle).tree.mlp!r}, but this block "
            f"(children={_children(block)}) has no such child",
            reason="component_unavailable",
        )
    return mlp


def _children(module: Any) -> list[str]:
    return sorted(name for name, _ in module.named_children())


def _probe_moe(bundle: Any, component: str, layer: int) -> str | None:
    mlp = _mlp(bundle, layer)
    if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
        return None
    return (
        f"component {component!r} needs a sparse-MoE block at layer {layer}, "
        f"but this MLP (children={_children(mlp)}) is not one — extend the tap "
        "table in pytorch_hooks/sites.py."
    )


def _probe_shared_expert(bundle: Any, component: str, layer: int) -> str | None:
    if getattr(_mlp(bundle, layer), "shared_expert", None) is not None:
        return None
    return (
        f"component {component!r} needs a shared expert, which this MoE block "
        f"at layer {layer} does not have."
    )


def _probe_grouped_mm(bundle: Any, component: str, layer: int) -> str | None:
    # the dispatch pin: the interior tensors these components name are the
    # *grouped* function's locals. Another implementation — the "eager"
    # per-expert loop, "batched_mm" — computes the same block output (📐 to
    # 4.2e-7, test_sites_round3_moe_interior.py) by a different factorization,
    # whose intermediates are different tensors. Same numbers, wrong
    # provenance: refused by name, naming the knob.
    impl = _experts_implementation(bundle)
    if impl == "grouped_mm":
        return None
    return (
        f"component {component!r} taps the interior of the grouped experts "
        f"dispatch, but this model runs experts_implementation={impl!r} — a "
        "different factorization whose intermediates are different tensors, "
        "even though the block's output agrees. Load the model with "
        "experts_implementation='grouped_mm' (the default), or extend "
        "experts_interface.py for this implementation."
    )


def _probe_split_qkv(bundle: Any, component: str, layer: int) -> str | None:
    """The mixer's q, k and v are addressable: the per-family tap table has met
    the family (its rows address the interior, a fused projection included —
    GPT-2's ``c_attn`` as three logical column blocks), or, for a family it has
    not met, the mixer carries separate projections (measured). A fused
    projection without a row is refused by name: which block is which is a
    family fact only a row can state."""
    if family_in_table(bundle.info):
        return None
    attn = _attn(bundle, layer)
    if _has_separate_projections(attn):
        return None
    if hasattr(attn, "c_attn"):
        return (
            f"component {component!r} needs separate q/k/v projections, and this "
            f"mixer fuses them into one 'c_attn' (children="
            f"{_children(attn)}). Splitting a fused qkv projection is the "
            f"per-family tap table, and family "
            f"{bundle.info.family!r} has no row in it — measure the family and "
            "add its addresses to the interior rows' overrides in "
            "causalab/protocol/registry.py; 'attention_premix' and "
            "'attention_output' read on this family today."
        )
    return (
        f"component {component!r} needs addressable q/k/v projections, and this "
        f"mixer (children={_children(attn)}) has neither separate ones nor a row "
        f"for family {bundle.info.family!r} in the per-family tap table — measure "
        "the family and add its addresses to the interior rows' overrides in "
        "causalab/protocol/registry.py."
    )


def _no_gate(bundle: Any, component: str, layer: int) -> str:
    return (
        f"component {component!r} at layer {layer} of {bundle.key!r}: this "
        "mixer computes no output gate. The box exists only on the "
        "gated-attention family (Qwen3.5/3.6), whose q-projection emits "
        "[q | gate] per head and which multiplies the mixer's output by "
        "sigmoid(gate) before projecting out. On this family there is no such "
        "tensor to read or write."
    )


def _probe_gated_attention(bundle: Any, component: str, layer: int) -> str | None:
    """The mixer computes an output gate: the row addresses one for the family
    (the per-family tap table), or, for a family the table has not met, the
    q-projection measures ``H·2·d`` wide (📐 512 on qwen3.5-moe for H 8, d 32,
    against ``H·d = 16`` on llama) — the doubled width is the gate."""
    row = capability(component)
    if family_in_table(bundle.info):
        return (
            None
            if row.address_on(bundle.info) is not None
            else _no_gate(bundle, component, layer)
        )
    attn = _attn(bundle, layer)
    value = component_shape(bundle.info, component)
    assert value.width is not None
    for module in _declared_modules(component, packings=frozenset({"fused_heads"})):
        projection = getattr(attn, module, None)
        if projection is None:
            continue
        out = _projection_width(projection)
        if out == 2 * value.width:
            return None
        if out is not None and out != value.width:
            raise ProtocolError(
                "P4",
                f"the q-projection at layer {layer} of {bundle.key!r} emits {out} "
                f"features, which is neither {value.width} (heads·head_dim) nor "
                f"{2 * value.width} (a gated family's [q | gate] per head). This "
                "backend cannot say which columns are the queries — measure the "
                "family and add a row to the per-family tap table "
                "(causalab/protocol/registry.py).",
                reason="component_unavailable",
            )
    return _no_gate(bundle, component, layer)


#: One probe per predicate in the registry's vocabulary — the shared
#: module-tree evaluators, which every family uses unless its adapter declares
#: its own cell for a predicate (``FamilyAdapter.probes``). The census guard
#: asserts the keys are exactly ``registry.PREDICATES``; a predicate added to a
#: row without a probe here fails that test, not a document.
_PREDICATE_PROBES: dict[str, Any] = {
    "moe": _probe_moe,
    "shared_expert": _probe_shared_expert,
    "grouped_mm": _probe_grouped_mm,
    "split_qkv": _probe_split_qkv,
    "gated_attention": _probe_gated_attention,
}


def _check_requires(bundle: Any, component: str, layer: int) -> None:
    """Refuse a component whose row requires an architectural fact the loaded
    model does not have, in the rows' predicate order (a fused-qkv family is
    named before its missing gate; a dense MLP before its missing shared
    expert). Each predicate is evaluated by the family's own cell where its
    adapter declares one, by the shared module-tree probe otherwise."""
    row = capability(component)
    probes = adapter_of(bundle).probes
    for predicate in PREDICATES:
        if predicate not in row.requires:
            continue
        probe = probes.get(predicate, _PREDICATE_PROBES[predicate])
        refusal = probe(bundle, component, layer)
        if refusal is not None:
            raise ProtocolError("P4", refusal, reason="component_unavailable")
