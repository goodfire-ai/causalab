"""Resolve component names to model taps shared by both engines.

Each site identifies a module, input or output side, and feature slice.
The registry supplies family addresses, shapes, predicates, and write
policies. Validation checks engine support; ``model_tree`` checks loaded
module structure.

``mlp_activation`` follows the family definition: Llama-style models use
``act(gate_proj(x))``; GPT-2 uses the down-projection input.
``attention_premix`` selects query-head coordinates at the output
projection's input. On gated attention this is ``gate * z``, so the tap
includes the gate's effect.

Attention interiors use the registry's family addresses. Both engines serve
attention-probability writes through the eager attention body, with the
registry restricting them to swaps. Expert interior support also follows
the selected engine's capability rows.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from causalab.neural.shared.model_tree import (
    _attn,
    _blocks,
    _check_projection_width,
    _check_requires,
    _check_stream,
    _children,
    _measured_address,
    adapter_of,
)
from causalab.neural.shared.parallel.placement import REPLICATED, Placement
from causalab.neural.shared.parallel.placements import module_path, site_placement
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol import registry
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.registry import (
    ATTENTION_FUNCTION_SLOTS,
    CAPABILITIES,
    DELTA_KERNEL_SLOTS,
    EXPERTS_FUNCTION_SLOTS,
    FamilyAdapter,
    ParallelPlan,
    Tap,
    capability,
    component_shape,
    expert_axis_refusal,
    head_space_refusal,
    native_shape,
    walk,
)
from causalab.protocol.schema import (
    DEPRECATED_COMPONENTS,
    LAYERLESS_COMPONENTS,
    SiteSpec,
)
from causalab.protocol.registry.shapes import FeatureShape


__all__ = [
    "ATTENTION_FUNCTION_SLOTS",
    "DELTA_KERNEL_SLOTS",
    "EXPERTS_FUNCTION_SLOTS",
    "ResolvedSite",
    "adapter_of",
    "inventory",
    "resolve_band",
    "resolve_site",
]


@dataclasses.dataclass(frozen=True)
class ResolvedSite:
    """One tapped location: the module, which side of it carries the
    activation, an optional feature-axis slice (per-head views), and how the
    module's own tensor shape relates to the executor's ``(batch, position,
    feature)`` contract.

    ``shape`` is never chosen per tap: [`resolve_site`][] reads it from
    [`component_shape`][causalab.protocol.registry.components.component_shape], so the description each
    engine converts by and the description the protocol layer validates against
    are the same object. ``tuple_index`` defaults to the historical rule —
    element 0 of a tuple payload. See [`causalab.neural.shared.layout`][] for
    how the conversion is computed from the declared axes.
    """

    module: Any
    kind: str  # "in" | "out"
    #: The module's native tensor shape; converted to/from the executor's
    #: contract at the hook boundary rather than special-cased per component.
    shape: FeatureShape
    feature_slice: slice | None = None
    layer: int = 0
    component: str = "block_output"
    #: Which element of a tuple payload the tap means. None keeps the historical
    #: rule (element 0 of a tuple, else the payload itself); an explicit index
    #: is required for e.g. a router returning (logits, scores, indices).
    tuple_index: int | None = None
    #: Where inside the attention function this component lives, when it is not
    #: a module boundary at all — see
    #: [`causalab.neural.engines.pytorch_hooks.attention_interface`][].
    #:
    #: Set together with ``kind == "interface"`` for the four function-interior
    #: components. ``attention_probs`` is the one site that sets it while
    #: keeping an ordinary ``kind``: the mixer *returns* the pattern, so reading
    #: it is a plain module tap, and only the write has to go through the
    #: function.
    interface_slot: str | None = None
    #: Where an input write's delta has to land as well — see
    #: [`Writeback`][]. Set from the family's declared ``Tap.writeback``;
    #: ``block_mid`` is the component that has one. Reads ignore this field.
    writeback: "Writeback | None" = None
    #: The head the site named, kept alongside ``feature_slice`` because a
    #: *derived* component slices in a space the raw tensor does not have.
    head: int | None = None
    #: The expert the site named — the ragged face of a routed-interior tap:
    #: select the (position, slot) pairs the router sent to this expert.
    #: Carried on the site rather than lowered to a slice, because which rows
    #: it selects is a *runtime* fact (the routing), not a static one.
    expert: int | None = None
    #: Set when the component's value is **computed from** the tapped tensor
    #: rather than being it. Then ``shape`` describes what is captured and
    #: [`component_shape`][causalab.protocol.registry.components.component_shape] describes the value —
    #: the one place in the backend where those two differ, and the field exists
    #: so that difference is declared rather than inferred.
    derivation: str | None = None
    #: Where the tapped tensor lives across ranks
    #: (``docs/model_parallelism.md`` §4): the executor makes it ``whole``
    #: before the hook body and ``fragment``\ s the result after, through
    #: [`causalab.neural.shared.parallel.fragments`][]. Decided at plan time
    #: from the family's parallel-plan row for the tapped module
    #: ([`site_placement`][],
    #: filled by [`resolve_site`][] when the bundle carries a plan and a
    #: geometry above world 1), never from the value; a module no row names —
    #: every module at world 1 — is
    #: [`REPLICATED`][].
    placement: Placement = REPLICATED
    #: The routing table this tap carries is this rank's remapped one
    #: (``EpRouterParallel``'s local ids, §6.3): the executor reconstructs the
    #: global table before it enters the contract.
    remapped_routing: bool = False

    @property
    def depth(self) -> tuple[int, int]:
        """(layer, intra-order) — matches the protocol planner's ranks."""
        from causalab.protocol.positions.alignment import (
            COMPONENT_RANK,
            UNRANKED,
        )  # one table

        rank = COMPONENT_RANK.get(self.component, UNRANKED)
        if self.component in ("ln_final", "lm_head"):
            return (1_000_000, rank)
        return (self.layer, rank)


@dataclasses.dataclass(frozen=True)
class Writeback:
    """Where an input write's delta lands, all of it read off the declared
    target component's own tap: the ``module`` whose output receives the
    delta, the ``component`` that output *is* (so an engine orders the
    landing by that component's forward rank instead of assuming
    ``block_output``'s), and the ``tuple_index`` of the payload element to
    rewrite. ``module`` is an ``nn.Module`` in the hook engine and an nnsight
    ``Envoy`` in the trace engine."""

    module: Any
    component: str
    tuple_index: int | None = None


def _declared_tap(adapter: FamilyAdapter, component: str, layer: int, key: str) -> Tap:
    """The family's tap for ``component`` — or the registry's refusal by name.

    This is the per-family availability row: a component the
    family does not declare is refused *here*, with the family, the component
    and what the family does serve, never as a bare ``AttributeError`` out of a
    module lookup (the failure ``_FULL_ATTENTION_ONLY``'s note records).
    """
    tap = adapter.tap_for(component)
    if tap is None:
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {key!r}: model family "
            f"{adapter.family!r} declares no tap for it — the family's plugin "
            "(registry.FamilyAdapter.taps) does not serve this component, so "
            "there is no such tensor on this family. It serves "
            f"{sorted(adapter.taps)}. Declare the tap in the family's adapter "
            "(causalab.protocol.registry.register_family) if the tree has it.",
            reason="component_unavailable",
        )
    return tap


def _scope_module(bundle: Any, adapter: FamilyAdapter, tap: Tap, layer: int) -> Any:
    """The module a tap's scope names on this model, at ``layer``."""
    if tap.scope in ("embedding", "final_norm", "lm_head"):
        return walk(bundle.model, getattr(adapter.tree, tap.scope))
    block = _blocks(bundle)[layer]
    if tap.scope == "block":
        return block
    if tap.scope == "mixer":
        return _attn(bundle, layer)
    assert tap.scope == "mlp", tap.scope
    return walk(block, adapter.tree.mlp)


def _writeback(
    bundle: Any, adapter: FamilyAdapter, tap: Tap, layer: int
) -> Writeback | None:
    """A tap's declared writeback target, resolved through the *target
    component's own tap* — so the module, the forward depth and the payload
    element come off one declaration and no engine re-derives any of them.

    ``None`` where the tap declares no writeback. ``FamilyAdapter`` has
    already refused, at construction, a family that needs one and omits it
    and one whose target it does not tap, so what is left to fail here is the
    module tree — and that refusal is about a component the document never
    named, so it says which write-back asked for it."""
    if tap.writeback is None:
        return None
    try:
        target = _declared_tap(adapter, tap.writeback, layer, bundle.key)
        module = _tap_module(bundle, adapter, target, tap.writeback, layer)
    except ProtocolError as error:
        raise ProtocolError(
            "P4",
            f"a write at this site has to carry its delta to "
            f"{tap.writeback!r} (the family's declared write-back target), and "
            f"that component does not resolve on this model: {error}",
            reason="component_unavailable",
        ) from error
    return Writeback(
        module=module, component=tap.writeback, tuple_index=target.tuple_index
    )


def _tap_module(
    bundle: Any, adapter: FamilyAdapter, tap: Tap, component: str, layer: int
) -> Any:
    """The module ``tap`` addresses — or a refusal naming the family's claim
    and the tree it disagrees with. ``mlp_activation`` keeps its pre-PR text:
    the load-time twin is ``component_shape`` refusing an entry with no dense
    inner width (an all-MoE tower), and this is the same fact read off the
    module tree for a document arriving unvalidated."""
    scope = _scope_module(bundle, adapter, tap, layer)
    module = walk(scope, tap.path) if scope is not None else None
    if module is not None:
        return module
    if component == "mlp_activation" and scope is not None:
        raise ProtocolError(
            "P4",
            f"mlp_activation: this MLP (children={_children(scope)}) matches no "
            "known family — extend the tap table in pytorch_hooks/sites.py "
            "(and mirror it in the hook oracle).",
            reason="component_unavailable",
        )
    where = f"{tap.scope}.{tap.path}" if tap.path else tap.scope
    children = _children(scope) if scope is not None else None
    raise ProtocolError(
        "P4",
        f"component {component!r} at layer {layer} of {bundle.key!r}: family "
        f"{adapter.family!r} taps it at {where!r}, but this model has no such "
        f"module (children of the {tap.scope}: {children}) — the family's "
        "declaration and the loaded tree disagree; correct the tap in the "
        "family's adapter (causalab.protocol.registry.register_family).",
        reason="component_unavailable",
    )


def _head_slice(bundle: Any, component: str, head: int | None) -> slice | None:
    """The feature-axis slice a ``head`` names — or a refusal.

    The bound comes from the component's own shape, not from
    ``info.num_heads``. 📐 That distinction is not cosmetic: under GQA the
    KV-space components are ``num_key_value_heads`` wide, and a query-space
    bound over them produces a slice that is *empty* rather than out of range.
    Python does not raise on that — the read saves a ``(b, n_pos, 0)`` tensor
    and the write mutates nothing — which is the silent no-op the read-only
    rows of the capability registry exist to prevent elsewhere.
    """
    if head is None:
        return None
    shape = component_shape(bundle.info, component)
    space = shape.head_space
    if space is None:
        raise ProtocolError("P4", head_space_refusal(component, head, shape))
    if not 0 <= head < space:
        raise ProtocolError(
            "P4",
            f"site names head {head} on component {component!r}, which has "
            f"{space} heads ({shape.describe()})",
        )
    width = shape.width
    assert width is not None  # a head axis implies a feature axis
    per_head = width // space
    return slice(head * per_head, (head + 1) * per_head)


#: The mixer's interior at module boundaries: the rows that require
#: addressable q/k/v projections (``split_qkv``), read off the registry rather
#: than listed again. Where each lives on each family is the rows' per-family
#: address (``Capability.overrides``) — the same set as
#: ``registry.INTERIOR_ROWS``, and a census test holds the two equal.
#:
#: 📐 These look as if they need function-level taps inside the mixer
#: forward. Measured on ``tiny-random/qwen3.5-moe``, three of the four are
#: ordinary ``nn.Module`` outputs: ``Qwen3_5MoeAttention`` runs ``q_norm`` and
#: ``k_norm`` **before** RoPE, so their outputs *are* the pre-RoPE projections,
#: and ``v_proj``'s output is ``v`` itself. Only the gate needs a descriptor
#: trick, and only because it shares a projection with ``q`` — the same trick
#: (a fused axis the layout conversion selects and scatters back through) that
#: serves the three blocks of GPT-2's ``c_attn``.
_ATTENTION_INTERIOR: frozenset[str] = frozenset(
    c for c, row in CAPABILITIES.items() if "split_qkv" in row.requires
)


def _interior_address(
    bundle: Any, attn: Any, component: str, layer: int
) -> Mapping[str, Any]:
    """Where ``component`` is on this mixer: the row's address for the family
    (``Capability.overrides``, the per-family tap table), or — for a family the
    table has not met — the measured one."""
    declared = capability(component).address_on(bundle.info)
    if declared is not None:
        return declared
    return _measured_address(bundle, attn, component, layer)


def _attention_interior_site(
    bundle: Any,
    attn: Any,
    component: str,
    layer: int,
    head: int | None,
    tap: Any,
) -> ResolvedSite:
    """Resolve one module-boundary tap inside the mixer, from the rows.

    The family differences — which child of the mixer carries the component,
    and how that child's tensor packs it — are the ``overrides`` of the four interior rows in
    [`causalab.protocol.registry`][], keyed by the family the loaded config
    declares. No family is named here. The measured three-family table is
    rendered from those rows into ``docs/running_experiments.md`` §5
    (``registry.render_family_table``) and checked against them.

    📐 What the rows say, in one line each (the rendering has the rest):
    llama taps the bare projections; qwen3.5-moe taps ``q_norm``/``k_norm``
    (before RoPE, ``(b, s, H, d)``), ``v_proj``, and the gate as split 1 of 2
    of ``q_proj``; GPT-2 taps the three ``H·d``-wide blocks of ``c_attn``'s
    output — so the **same logical site** reads and writes on a fused and a
    split projection alike, the fused one through the layout conversion's
    scatter into the native tensor (``layout.from_contract``).

    The ``split_qkv`` and ``gated_attention`` predicates were checked by
    `_check_requires` before this is reached; a family without a row was
    served or refused there by measurement.
    """
    feature_slice = _head_slice(bundle, component, head)
    address = _interior_address(bundle, attn, component, layer)
    module = getattr(attn, address["module"], None)
    if module is None:
        # the row is a claim about the family's module tree; a loaded mixer
        # that lacks the named child is the table disagreeing with the model,
        # refused by name rather than as a bare AttributeError out of the tap
        raise ProtocolError(
            "P4",
            f"component {component!r} at layer {layer} of {bundle.key!r}: the "
            f"per-family tap table's row for family {bundle.info.family!r} taps "
            f"{address['module']!r}, but this mixer (children={_children(attn)}) "
            "has no child of that name. Correct the family's address in the "
            "interior rows' overrides in causalab/protocol/registry.py.",
            reason="component_unavailable",
        )
    _check_projection_width(bundle, module, address, component, layer)
    # the row says how this family's module packs the value; the executor
    # converts by it in both directions, so a tensor that disagrees raises
    shape = native_shape(address, component_shape(bundle.info, component))
    return tap(module, "out", feature_slice=feature_slice, shape=shape)


# The mixer's interior *inside the attention function* is
# ``ATTENTION_FUNCTION_SLOTS`` (declared in the registry beside the family
# taps, re-exported here). 📐 These four are not module boundaries:
# ``transformers`` computes them within one ``attention_interface(...)`` call,
# so ``query`` and ``key`` are its arguments (post-RoPE, and for ``key``
# before ``repeat_kv``), the scores are the softmax's input inside it, and
# ``z`` is its return. See [`causalab.neural.engines.pytorch_hooks.attention_interface`][].


#: The MoE surface: every row that requires a sparse-MoE block. The
#: module-boundary taps (📐 the router is a module returning a 3-tuple and the
#: experts are a fused module), the dispatch interior, and the kernel's
#: ``expert_permutation`` — read off the rows, not listed again.
_MOE_COMPONENTS: frozenset[str] = frozenset(
    c for c, row in CAPABILITIES.items() if "moe" in row.requires
)

# The routed-expert interior *inside the experts dispatch* is
# ``EXPERTS_FUNCTION_SLOTS`` (the registry's, re-exported). 📐 These are not
# module boundaries: ``Qwen3_5MoeExperts`` stores its weights as 3-D
# parameters and computes the whole interior inside one dispatched
# ``ALL_EXPERTS_FUNCTIONS["grouped_mm"]`` call (its only child is the one
# shared ``act_fn``, which the wrapper hooks for the duration of that call).
# The reference engine taps them by wrapping that dispatch
# ([`causalab.neural.engines.pytorch_hooks.experts_interface`][]); the
# nnsight engine lands the same components through its `.source` address
# table — both consume the ``kind="experts"`` resolution below.


def _moe_site(
    bundle: Any,
    adapter: FamilyAdapter,
    tap: Tap,
    component: str,
    spec: SiteSpec,
    layer: int,
) -> ResolvedSite:
    """Resolve one MoE tap, from the family's declared tap.

    📐 Every tap here is ``flat_td``: ``Qwen3_5MoeSparseMoeBlock`` reshapes to
    ``(-1, hidden)`` before the router, so the whole interior is flattened over
    (batch, position) and only the block's own input and output are contract
    shaped. Measured on ``tiny-random/qwen3.5-moe`` at 1x6 tokens, hidden 8,
    128 experts, top-10::

        gate       out -> ((6,128) logits, (6,10) scores, (6,10) int64 indices)
        experts    out -> (6, 8)
        shared_expert.gate_proj / up_proj out -> (6, 32)
        shared_expert.down_proj       in  -> (6, 32)
        shared_expert                 out -> (6, 8)
        shared_expert_gate            out -> (6, 1)

    The router is the reason ``tuple_index`` exists: ``Qwen3_5MoeTopKRouter``
    returns three tensors and the historical "element 0 of a tuple" rule would
    have silently handed back the logits for all three.

    The ``moe`` / ``shared_expert`` / ``grouped_mm`` predicates were checked by
    `_check_requires` before this is reached.
    """
    # The `expert` sub-axis is the ragged face of the routed interior: select
    # the (position, slot) pairs the router sent to one expert. Only the rows
    # whose `expert_selection` names an engine carry it — the router's own axes
    # are all-experts (logits) or top-k (scores, indices), and the shared
    # expert is not one of the routed experts, so `expert` on those is refused
    # rather than silently ignored (the mistake `stream` made). `validate`
    # makes the same refusal at load from the same row; this is for a document
    # arriving unvalidated.
    expert = spec.expert if isinstance(spec.expert, int) else None
    if spec.expert is not None and not capability(component).expert_selection:
        raise ProtocolError(
            "P4",
            expert_axis_refusal(component, spec.expert),
            reason="component_unavailable",
        )
    if expert is not None:
        total = bundle.info.num_experts
        if total is None or not 0 <= expert < total:
            raise ProtocolError(
                "P4",
                f"site names expert {expert} on component {component!r}, but "
                f"{bundle.key!r} routes over {total} experts — the sub-axis "
                "selects one of them by its id.",
            )

    shape = component_shape(bundle.info, component)
    module = _tap_module(bundle, adapter, tap, component, layer)

    if tap.kind == "experts":
        if spec.head is not None and isinstance(spec.head, int):
            # no head axis anywhere in the MoE interior; refuse rather than drop
            _head_slice(bundle, component, spec.head)
        return ResolvedSite(
            module=module,
            kind="experts",
            layer=layer,
            component=component,
            shape=shape,
            interface_slot=tap.slot,
            expert=expert,
        )
    if tap.kind == "interior":
        # the serving kernel's row bookkeeping, inside the fused experts
        # forward — no module boundary and no dispatch slot; only the
        # nnsight engine's `.source` address table lands it, so it resolves
        # to the interior kind and the reference engine refuses by name.
        return ResolvedSite(
            module=module,
            kind="interior",
            layer=layer,
            component=component,
            shape=shape,
        )
    return ResolvedSite(
        module=module,
        kind=tap.kind,
        layer=layer,
        component=component,
        shape=shape,
        tuple_index=tap.tuple_index,
    )


def resolve_band(bundle: Any, spec: SiteSpec) -> tuple[ResolvedSite, ...]:
    """Every module a site addresses, one [`ResolvedSite`][] per layer of
    its band (§2.4 ``layers``), in band order — the one-layer band is the
    one-tuple of [`resolve_site`][]. The band is fanned out here and the
    resolved record stays scalar (``ResolvedSite.layer``): every engine
    consumer of a resolved site — hooks, address tables, the resume check —
    reasons about one module at a time."""
    band = spec.layers if isinstance(spec.layers, tuple) else None
    if band is None or len(band) <= 1:
        return (resolve_site(bundle, spec),)
    return tuple(
        resolve_site(bundle, dataclasses.replace(spec, layers=(layer,)))
        for layer in band
    )


def resolve_site(bundle: Any, spec: SiteSpec) -> ResolvedSite:
    """Resolve one site record to its tap, from the family's declared taps.

    The family is the bundle's plugin ([`adapter_of`][]); *where* a component
    is on it — which module, which side, which function slot — is the
    adapter's tap for the component (``registry.FamilyAdapter.taps``), and a
    component the family does not declare is refused by name. What this
    module keeps is the order of the checks and everything that is not an
    address: the stream check, the predicate probes, the head and expert
    sub-axes, the shape and the derivations — the same for every family.
    Refuses honestly on components this engine does not implement yet.

    The site's ``placement`` (``docs/model_parallelism.md`` §4) is read off
    the family's parallel-plan row for the tapped module and the bundle's
    geometry (`_placed`); at world 1, or on a bundle carrying no plan,
    it is [`REPLICATED`][] and
    nothing else runs.
    """
    return _placed(bundle, _resolve_unplaced(bundle, spec))


def _placed(bundle: Any, site: ResolvedSite) -> ResolvedSite:
    """The site with its placement filled from the registry entry's plan
    (``bundle.info.parallel_plan``) **as applied under the geometry** the
    bundle was loaded under (``bundle.geometry``) — ``ParallelPlan.
    for_geometry``, the very table ``apply_plan`` sharded the model from,
    the K/V projections replicated above the KV heads (§6.6). At world 1 —
    every bundle a test stand-in builds without a geometry too — nothing is
    read and the site is returned as resolved."""
    geometry: ParallelGeometry = getattr(bundle, "geometry", ONE)
    if geometry.world == 1:
        return site
    # an entry declaring no plan shards nothing over the tensor and expert
    # axes (the geometry check refuses those above one), but its positions
    # are still chunked under context > 1 (§8.4) — so the empty plan
    plan: ParallelPlan = bundle.info.parallel_plan or ParallelPlan(rows={})
    placed = site_placement(
        plan=plan.for_geometry(geometry, bundle.info),
        geometry=geometry,
        path=module_path(bundle.model, site.module),
        prefix=getattr(bundle.model, "base_model_prefix", None),
        kind=site.kind,
        component=site.component,
        layer=site.layer,
        num_layers=len(_blocks(bundle)),
        layerless=site.component in LAYERLESS_COMPONENTS,
        shape=site.shape,
        slot=site.interface_slot,
        tuple_index=site.tuple_index,
    )
    return dataclasses.replace(
        site, placement=placed.placement, remapped_routing=placed.remapped_routing
    )


def _resolve_unplaced(bundle: Any, spec: SiteSpec) -> ResolvedSite:
    component = spec.component
    if not isinstance(component, str):
        raise ProtocolError("P2", f"unresolved site component {component!r}")
    if component in DEPRECATED_COMPONENTS:
        # the parser folds a retired spelling before anything downstream sees
        # it; a SiteSpec built by hand that still carries one is named rather
        # than falling through to "no capability row"
        raise ProtocolError(
            "P2",
            f"site component {component!r} is a retired spelling of "
            f"{DEPRECATED_COMPONENTS[component]!r}, which the parser folds at "
            "load — a site built without the parser must name the current "
            "component",
        )
    band = spec.layers if isinstance(spec.layers, tuple) else ()
    if len(band) > 1:
        # a band is one site across N layers, and a ResolvedSite is one
        # module: the fan-out is `resolve_band`, and an executor lowers a
        # band to its per-layer members (`lowering.lower_bands`) before it asks
        # for modules — so a band reaching here is a caller that skipped that
        raise ProtocolError(
            "P2",
            f"site spans layers {list(band)} — a band resolves to one module "
            "per layer (resolve_band), not to one ResolvedSite; the executor "
            "lowers a band to its members before resolving",
        )
    layer = band[0] if band else 0
    head = spec.head if isinstance(spec.head, int) else None
    adapter = adapter_of(bundle)

    def tap(
        module: Any,
        kind: str,
        *,
        feature_slice: slice | None = None,
        tuple_index: int | None = None,
        interface_slot: str | None = None,
        writeback: "Writeback | None" = None,
        shape: FeatureShape | None = None,
        derivation: str | None = None,
    ) -> ResolvedSite:
        """One tap, with its shape read from the component table.

        The shape is resolved *here* rather than at each branch so that adding a
        component means adding a table entry and a module, never a third place
        that has an opinion about the tensor's axes.

        ``shape`` is overridden in exactly two cases, both declared elsewhere
        rather than decided per branch: a *derived* component, whose tap
        captures a different tensor than the one the component names
        (``attention_result``: the tap's ``shape_of``), and the attention
        interior, whose native **packing** is a fact about the family's
        *module*, not the component — 📐 ``Qwen3_5MoeAttention.q_norm`` emits
        ``(b, s, H, d)``, llama's bare ``q_proj`` ``(b, s, H·d)``, GPT-2's
        ``c_attn`` three ``H·d`` blocks — and is read off the component row's
        per-family address (``registry.native_shape``). Same component, same
        width, same head space; only the packing differs, and packing is the
        half of the descriptor the backend owns. Everything the protocol layer
        validates against is family-independent and stays in the one table.
        """
        if shape is None:
            shape = component_shape(bundle.info, component)
        return ResolvedSite(
            module=module,
            kind=kind,
            shape=shape,
            feature_slice=feature_slice,
            layer=layer,
            component=component,
            tuple_index=tuple_index,
            interface_slot=interface_slot,
            writeback=writeback,
            head=head,
            derivation=derivation,
        )

    if component in LAYERLESS_COMPONENTS:
        # the model boundary: the ids as the embedding's INPUT (the one module
        # boundary they cross, read-only and layer-less, §5.4), the embedding's
        # output, the final norm and the head — the family's tree names them
        declared = _declared_tap(adapter, component, layer, bundle.key)
        return tap(
            _tap_module(bundle, adapter, declared, component, layer), declared.kind
        )

    # Order matters: the stream check runs FIRST so that a full-attention-only
    # component at a Gated DeltaNet layer refuses with the architectural reason
    # ("there is no attention matrix here") rather than an engine-shaped one
    # ("this engine has not implemented it yet"). The first is permanent and
    # stayed true once attention_probs landed; the second was a roadmap
    # statement, and is now moot — both engines serve the component. The
    # predicate probes come second, the family's availability third: a MoE
    # component on a dense block is refused as "not a sparse-MoE block", which
    # is the fact, before any family is asked whether it declares a tap.
    _check_stream(bundle, component, spec, layer)
    _check_requires(bundle, component, layer)
    declared = _declared_tap(adapter, component, layer, bundle.key)

    if declared.from_row:
        # the module-boundary interior: the row's per-family address
        return _attention_interior_site(
            bundle, _attn(bundle, layer), component, layer, head, tap
        )
    if component in _MOE_COMPONENTS:
        return _moe_site(bundle, adapter, declared, component, spec, layer)

    module = _tap_module(bundle, adapter, declared, component, layer)

    if declared.kind == "interior":
        # inside a fused forward and its kernel — no module boundary anywhere;
        # the mixer is carried because it identifies whose forward to tap. Same
        # marker as the expert interior: the engine with `.source` addressing
        # serves it, the reference engine refuses by name.
        return tap(module, "interior")
    if component == "attention_probs":
        # element 1 of the mixer's (attn_output, attn_weights). Reading is an
        # ordinary tap; WRITING is not — see attention_interface.py, which
        # owns that half. Its shape has two position axes and so no contract
        # form, which is what makes the executor refuse to gather or featurize
        # it without any component name appearing in that refusal.
        return tap(
            module,
            declared.kind,
            tuple_index=declared.tuple_index,
            interface_slot=declared.slot,
        )
    if declared.derivation is not None:
        # 📐 `attention_result`: the model never computes this. Each head's
        # contribution to the residual stream is
        # `premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`, and the model
        # forms only their sum — by projecting the whole premix at once. So
        # the tap is `attention_premix`'s, and the value is derived from it
        # after the position gather, which keeps the cost
        # `n_positions · H · hidden` rather than `seq · H · hidden`.
        #
        # No `feature_slice`: a `head` here selects in the *result's* space
        # (hidden-wide blocks), not the captured tensor's (head_dim-wide), and
        # naming one makes the derivation cheap rather than making it a slice.
        if head is not None:
            _head_slice(bundle, component, head)  # bound-check, discard
        assert declared.shape_of is not None
        return tap(
            module,
            declared.kind,
            shape=component_shape(bundle.info, declared.shape_of),
            derivation=declared.derivation,
        )
    if declared.kind == "delta":
        # no module boundary: the tensor is an argument or return of the
        # kernel-boundary globals. The mixer is carried as the module — it
        # is what identifies *which* forward's calls to tap.
        if declared.slot == "state":
            # the state has a head axis but no feature axis to slice — the
            # bound is checked here and the executor selects the head on
            # the native matrix after the position gather
            shape = component_shape(bundle.info, component)
            space = shape.head_space
            if head is not None:
                assert space is not None
                if not 0 <= head < space:
                    raise ProtocolError(
                        "P4",
                        f"site names head {head} on component "
                        f"{component!r}, which has {space} heads "
                        f"({shape.describe()})",
                    )
            return tap(module, "delta", interface_slot=declared.slot)
        # the conv's fused unequal widths refuse a head by shape, like the qkv
        return tap(
            module,
            "delta",
            feature_slice=_head_slice(bundle, component, head),
            interface_slot=declared.slot,
        )
    if declared.kind == "interface":
        # No module boundary to hook: these four live inside one call. The
        # module is carried anyway, because it is what identifies *which*
        # mixer's call to tap.
        return tap(
            module,
            "interface",
            feature_slice=_head_slice(bundle, component, head),
            interface_slot=declared.slot,
        )
    # a module side: the head, where the site names one, is a slice of the
    # component's own head space — and a refusal by shape on a component with
    # none (`delta_qkv`'s fused unequal widths; a residual-stream tensor)
    return tap(
        module,
        declared.kind,
        feature_slice=_head_slice(bundle, component, head),
        tuple_index=declared.tuple_index,
        writeback=_writeback(bundle, adapter, declared, layer),
    )


def inventory(bundle: Any) -> registry.Inventory:
    """The loaded model's inventory as this resolver serves it: the registry's
    per-layer inventory (``registry.inventory``), refined by whether each
    ``(component, layer)`` actually resolves on the loaded tree. Offline and
    loaded agree wherever the registry entry knows the fact; where only the
    module tree does (📐 the tiny qwen3.5-moe fixture's config declares a dense
    inner width no MoE block has, so the entry sizes ``mlp_activation`` and the
    tree refuses it), this is the answer a run gives."""

    def serves(component: str, layer: int | None) -> bool:
        try:
            resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
        except ProtocolError:
            return False
        return True

    return registry.inventory(bundle, serves=serves)
