"""The family plugin contract, the built-in module trees, and the inventory.

A model *family* is a module tree; everything an engine needs to know about
one — detection, component resolution, mixer children, reconstruction
identities — is declared here as data beside the capability rows
([`causalab.protocol.registry.components`][]), and [`inventory`][] is the
one producer of "what exists at which layer".
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Literal, Mapping, get_args

from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.registry.components import (
    CAPABILITIES,
    INTERIOR_ROWS,
    PREDICATES,
    Predicate,
    component_shape,
    unavailable_at_load,
)
from causalab.protocol.registry.models import ModelInfo
from causalab.protocol.schema import COMPONENTS, LAYERLESS_COMPONENTS, STREAMS, Stream


# --------------------------------------------------------------------------- #
# the family plugin contract
# --------------------------------------------------------------------------- #
#
# A model *family* is a module tree, and everything an engine needs to know
# about one is declared here, beside the rows, as data: **detection** (one
# predicate over the loaded tree — never a config-string match), **component
# resolution** (semantic name → the family's tap: which module, which side,
# which function slot), the **mixer children** and the stream each means,
# **reconstruction identities** with dtype-keyed tolerances, and optional
# per-family evaluators for the rows' predicates. Tensor-shape contracts stay
# [`component_shape`][] (family-independent), supported mechanisms and
# aliases stay the rows' ``reads`` / ``writes`` / ``aliases`` cells, and the
# attention interior's per-family *address* stays the rows' ``overrides``
# — the adapter says the interior is tapped at the mixer *from the
# row*, and nothing is stated twice.
#
# The vocabulary is one global closed ``Component`` literal and the
# adapter declares **per-family availability** over it: a
# component with no tap on a family is refused by name, at the registry, not by
# an ``AttributeError`` out of a module lookup. The stated limit follows: a
# third party registers a family that serves the *existing* names on a new
# tree (``register_family``, from any module — the site resolver reads the
# registry and names no family), and cannot mint a component; a new tensor is
# a row here first.

#: Where a tap's module lives: the model root's four addressed children (the
#: adapter's [`TreeAddress`][]), one decoder block, that block's mixer
#: (whichever stream it carries) or its MLP.
TapScope = Literal["embedding", "final_norm", "lm_head", "block", "mixer", "mlp"]
TAP_SCOPES: tuple[TapScope, ...] = get_args(TapScope)

#: How the resolved site is landed — [`ResolvedSite.kind`][causalab.neural.shared.sites.ResolvedSite.kind] in
#: ``neural/shared/sites.py``: a module's input or output (a hook, an envoy),
#: or one of the function-boundary mechanisms: a slot of the attention
#: function, of the delta kernel's globals, of the grouped experts dispatch,
#: or a line inside a fused forward that only ``.source`` addressing reaches.
HookKind = Literal["in", "out", "interface", "delta", "experts", "interior"]
HOOK_KINDS: tuple[HookKind, ...] = get_args(HookKind)
_SLOT_KINDS: frozenset[str] = frozenset({"interface", "delta", "experts"})


@dataclasses.dataclass(frozen=True)
class Tap:
    """One family's tap for one component: the module (a scope and a dotted
    child path under it — ``""`` is the scope module itself), the hook kind,
    and the details the resolver hands the executor unchanged.

    ``from_row`` marks the attention interior: the child is not named here but
    read off the component row's per-family address
    (``Capability.overrides``), so a family the tap table has met is served from the row and one
    it has not is served by measurement or refused — exactly as before.
    """

    scope: TapScope
    path: str = ""
    kind: HookKind = "out"
    #: which element of a tuple payload the tap means (a router's 3-tuple)
    tuple_index: int | None = None
    #: the function slot for a slot kind — and ``"probs"`` on the pattern,
    #: whose read is a module tap and whose write goes through the function
    slot: str | None = None
    #: the **component** whose output receives an input write's delta. One
    #: name, and the engines read everything off it: the module (that
    #: component's own tap), the forward depth the delta lands at
    #: (``plan.COMPONENT_RANK``) and the payload element it rewrites (that
    #: tap's ``tuple_index``) — none of the three is assumed.
    #:
    #: ``block_mid`` is the component that needs it: the norm's input is the
    #: readable residual *and* what the MLP consumes, but the block has
    #: already saved that same value for its residual addition, so a write
    #: that only rewrote the norm's input would be dropped by the skip.
    #: Required rather than remembered — see [`register_family`][]. Only
    #: meaningful on an ``in`` tap.
    writeback: str | None = None
    #: the component's value is *computed from* the capture (``attention_result``)
    derivation: str | None = None
    #: the component whose shape the capture has, when it is not this one's
    shape_of: str | None = None
    from_row: bool = False

    def __post_init__(self) -> None:
        if self.scope not in TAP_SCOPES:
            raise ValueError(f"tap scope {self.scope!r} is not in {list(TAP_SCOPES)}")
        if self.kind not in HOOK_KINDS:
            raise ValueError(f"tap kind {self.kind!r} is not in {list(HOOK_KINDS)}")
        if self.path and not all(part.isidentifier() for part in self.path.split(".")):
            raise ValueError(f"tap path {self.path!r} is not a dotted child path")
        if self.kind in _SLOT_KINDS and self.slot is None:
            raise ValueError(f"a {self.kind!r} tap names its slot")
        if self.from_row and (self.path or self.scope != "mixer"):
            raise ValueError("a from_row tap is the mixer's, with the row's child")
        if self.writeback is not None:
            if self.kind != "in":
                raise ValueError(
                    f"a writeback tap lands an input rewrite's delta, so it is an "
                    f"'in' tap, not {self.kind!r}"
                )
            if self.writeback not in COMPONENTS:
                raise ValueError(
                    f"tap writeback {self.writeback!r} is not a component — it "
                    "names the component whose output receives the delta"
                )


@dataclasses.dataclass(frozen=True)
class TreeAddress:
    """The four children a family's model root is addressed by, plus the
    block's MLP child — dotted paths from the model root (``model.layers``,
    ``transformer.h``), names rather than modules: the protocol layer is
    torch-free."""

    blocks: str
    embedding: str
    final_norm: str
    lm_head: str = "lm_head"
    mlp: str = "mlp"

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            path = getattr(self, field.name)
            if not path or not all(p.isidentifier() for p in path.split(".")):
                raise ValueError(f"tree {field.name}={path!r} is not a dotted path")


def walk(root: Any, path: str) -> Any:
    """``root.a.b.c`` for ``path == "a.b.c"`` — ``None`` where a child is
    missing, so the caller can refuse by name instead of an ``AttributeError``."""
    module = root
    for name in path.split(".") if path else ():
        module = getattr(module, name, None)
        if module is None:
            return None
    return module


@dataclasses.dataclass(frozen=True)
class Identity:
    """A reconstruction identity a family declares: ``component`` is
    recomputed from ``inputs`` by ``formula`` to within the tolerance the
    dtype allows. A test that pins the identity reads the row rather than its
    own literal, which is what makes a new family testable by the same suite;
    a dtype with no declared tolerance is refused, not guessed."""

    name: str
    component: str
    inputs: tuple[str, ...]
    formula: str
    #: dtype (the protocol's ``native_dtype`` spellings) → ``(atol, rtol)``
    tolerance: Mapping[str, tuple[float, float]]
    #: ``component`` is the plain **sum** of ``inputs`` — a residual add, so
    #: each input is an addend of a value the enclosing forward saved a copy
    #: of. This is the one property [`FamilyAdapter`][]'s write-back check
    #: argues from, and it is not structural: a *functional* identity over the
    #: same scope/child/target shape (``attention_output == attention_premix @
    #: W_o``) needs no write-back at all, because a write at the child's input
    #: propagates through the child on its own and adding the delta again
    #: would apply it twice.
    #:
    #: Checked, not trusted: the formula has to be exactly
    #: ``component == a + b [+ ...]`` over ``inputs``. That is also what makes
    #: the engines' unit coefficient safe — they land ``out + delta``, so a
    #: block computing ``α·x + f(x)`` cannot claim this flag.
    additive: bool = False

    def __post_init__(self) -> None:
        for c in (self.component, *self.inputs):
            if c not in COMPONENTS:
                raise ValueError(f"identity {self.name!r} names {c!r}, not a component")
        if not self.tolerance:
            raise ValueError(f"identity {self.name!r} declares no tolerance")
        plain = f"{self.component} == {' + '.join(self.inputs)}"
        # whitespace-normalized, so `a == b+c` is the *same* claim as
        # `a == b + c` rather than a near-miss that reads as a non-sum and
        # silently drops the write-back requirement with the flag omitted
        is_sum = len(self.inputs) >= 2 and "".join(self.formula.split()) == "".join(
            plain.split()
        )
        if self.additive != is_sum:
            raise ValueError(
                f"identity {self.name!r} declares additive={self.additive} but its "
                f"formula is {self.formula!r}"
                + (
                    f", not {plain!r}"
                    if self.additive
                    else " — which is the plain sum of its inputs, so it is additive"
                )
                + ": the flag and the formula say the same thing, and the flag is "
                "what carries a write at one addend to the sum as an unscaled "
                "delta (FamilyAdapter._check_writebacks). Declaring one without "
                "the other is how that requirement goes missing."
            )

    def tolerance_for(self, dtype: str) -> tuple[float, float]:
        try:
            return self.tolerance[dtype]
        except KeyError:
            raise ValueError(
                f"identity {self.name!r} declares no tolerance for dtype {dtype!r} "
                f"(declared: {sorted(self.tolerance)}) — measure it before asserting"
            ) from None


#: A per-family evaluator of one of the rows' predicates over the loaded
#: model: ``(bundle, component, layer) -> refusal text or None``.
Probe = Any


@dataclasses.dataclass(frozen=True)
class FamilyAdapter:
    """One model family's plugin: what a family declares, and all of it.

    ``detect`` is a predicate over the loaded top-level module — ``hasattr`` /
    child-name structure, never ``config.model_type`` — and exactly one
    registered family may detect a tree ([`family_for`][] refuses none and
    several). ``mixers`` maps each mixer child name the family's blocks may
    carry to the stream it means; the shared stream table
    (``neural/shared/model_tree.py``) reads the union over registered families and
    keeps refusing a block that carries children of two streams. ``taps`` is
    the family's availability over the global vocabulary: a component absent
    here does not exist on the family and is refused by name. ``identities``
    may be empty — a family with none declares none. ``probes`` overrides the
    resolver's shared module-tree evaluators per predicate, for a family whose
    tree spells an architectural fact differently.
    """

    family: str
    detect: Any
    tree: TreeAddress
    mixers: Mapping[str, Stream]
    taps: Mapping[str, Tap]
    identities: tuple[Identity, ...] = ()
    probes: Mapping[Predicate, Probe] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.family.isidentifier():
            raise ValueError(f"family name {self.family!r} is not an identifier")
        if not callable(self.detect):
            raise ValueError(f"family {self.family!r}: detect is not callable")
        if not self.mixers:
            raise ValueError(f"family {self.family!r} declares no mixer child")
        for child, stream in self.mixers.items():
            if not child.isidentifier() or stream not in STREAMS:
                raise ValueError(
                    f"family {self.family!r}: mixer {child!r} → {stream!r} is not a "
                    f"child name and a stream in {list(STREAMS)}"
                )
        unknown = sorted(set(self.taps) - set(COMPONENTS))
        if unknown:
            raise ValueError(
                f"family {self.family!r} declares taps for {unknown}, which are not "
                "in the component vocabulary — a new tensor is a capability row "
                "first (causalab/protocol/registry/components.py), then a tap here"
            )
        for component, tap in self.taps.items():
            if not isinstance(tap, Tap):
                raise ValueError(f"family {self.family!r}: {component} needs a Tap")
            if tap.shape_of is not None and tap.shape_of not in COMPONENTS:
                raise ValueError(
                    f"family {self.family!r}: {component} shape_of unknown"
                )
            if tap.from_row and component not in INTERIOR_ROWS:
                raise ValueError(
                    f"family {self.family!r}: {component} is not an interior row, "
                    "so no per-family address exists to read its child from"
                )
        unknown = sorted(set(self.probes) - set(PREDICATES))
        if unknown:
            raise ValueError(f"family {self.family!r}: probes for {unknown}")
        names = [identity.name for identity in self.identities]
        if len(set(names)) != len(names):
            raise ValueError(f"family {self.family!r} declares an identity twice")
        self._check_writebacks()
        object.__setattr__(self, "mixers", MappingProxyType(dict(self.mixers)))
        object.__setattr__(self, "taps", MappingProxyType(dict(self.taps)))
        object.__setattr__(self, "probes", MappingProxyType(dict(self.probes)))

    def _check_writebacks(self) -> None:
        """Both halves of the write-back contract, on the declaration alone.

        **Declared ones resolve.** ``sites._writeback`` reads the target's
        module, forward depth and payload element off *that component's own
        tap*, for every tap that declares a write-back — so a target this
        family does not tap is fatal at resolve time regardless of identities,
        and is refused here where the reason can be stated.

        **Required ones are declared.** This guards against one bug class. A component
        tapped as the input of a *child* module, and named as an addend of an
        **additive** identity, has had its value saved by the enclosing scope
        for that addition before the tap fires: a write that rewrites only the
        child's input is dropped, and the identity then fails under a write
        while still holding on a clean forward — which is what makes the
        omission easy to ship.

        ``additive`` is load-bearing and not structural. ``attention_premix``
        has the identical shape — an ``in`` tap on a child whose module's
        output is another component — but ``attention_output`` is a
        *function* of it, not a sum over it: a write at the child's input
        propagates through the child on its own, and a write-back would apply
        the delta twice. An ``in`` tap on the scope module itself
        (``block_input``) is exempt for the same kind of reason: nothing has
        been saved yet when it fires.
        """
        for component, tap in self.taps.items():
            if tap.writeback is None:
                continue
            if tap.writeback == component:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with a write-back "
                    "to itself — the delta is carried to a value the enclosing "
                    "forward saved *before* this tap, which cannot be this tap"
                )
            if tap.writeback not in self.taps:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with "
                    f"writeback={tap.writeback!r}, which the family does not tap "
                    "— the delta's module, forward depth and payload element are "
                    "all read off that component's own tap, so it has to exist"
                )
            target = self.taps[tap.writeback]
            # `sites._writeback` resolves the target with `_tap_module` alone,
            # so the only target it reproduces faithfully is one `resolve_site`
            # would also serve from that call — the generic module-boundary
            # branch. Every earlier branch of that dispatch yields a different
            # module or a different payload element than the component names,
            # and each is visible from this Tap plus CAPABILITIES.
            disqualifier = None
            if target.kind != "out":
                disqualifier = (
                    f"its tap is {target.kind!r}, and an input tap resolves to the "
                    "enclosing module — the landing would be ordered before the "
                    "write it is a delta of"
                )
            elif target.slot is not None:
                disqualifier = (
                    f"its tap names function slot {target.slot!r}, whose write goes "
                    "through the attention function rather than a module boundary"
                )
            elif target.from_row:
                disqualifier = (
                    "its tap is from_row, an attention interior whose module is the "
                    "row's per-family child, not the mixer `_tap_module` returns"
                )
            elif "moe" in CAPABILITIES[tap.writeback].requires:
                disqualifier = (
                    "it is a routed (MoE) component, resolved through the experts "
                    "dispatch rather than as a module boundary"
                )
            if disqualifier is not None:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with "
                    f"writeback={tap.writeback!r}, which is not a plain "
                    f"module-output boundary: {disqualifier}. The delta's module, "
                    "forward depth and payload element are all read off that tap."
                )
        addends: dict[str, list[Identity]] = {}
        for identity in self.identities:
            if identity.additive:
                for component in identity.inputs:
                    addends.setdefault(component, []).append(identity)
        for component, tap in self.taps.items():
            over = addends.get(component)
            if tap.kind != "in" or not tap.path or over is None:
                continue
            if len(over) > 1:
                raise ValueError(
                    f"family {self.family!r} makes {component!r} an addend of "
                    f"{sorted(i.name for i in over)} — a write there would owe a "
                    "delta to each, and one tap declares one write-back target"
                )
            identity = over[0]
            if tap.writeback is None:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} as the input of "
                    f"child {tap.path!r}, and declares additive identity "
                    f"{identity.name!r} ({identity.formula}) over it — so a write "
                    f"there must also reach {identity.component!r}, which the "
                    "enclosing forward has already saved. Declare it: "
                    f"Tap(..., writeback={identity.component!r})."
                )
            if tap.writeback != identity.component:
                raise ValueError(
                    f"family {self.family!r} taps {component!r} with "
                    f"writeback={tap.writeback!r}, but additive identity "
                    f"{identity.name!r} ({identity.formula}) makes "
                    f"{identity.component!r} the value a write there has to reach"
                )

    def serves(self, component: str) -> bool:
        return component in self.taps

    def tap_for(self, component: str) -> Tap | None:
        return self.taps.get(component)

    def blocks_of(self, model: Any) -> Any:
        """The decoder-layer list of a model this family detected."""
        blocks = walk(model, self.tree.blocks)
        if blocks is None:
            raise ProtocolError(
                "P4",
                f"family {self.family!r} addresses its blocks at "
                f"{self.tree.blocks!r}, but this model ({type(model).__name__}) has "
                "no such child — the family's tree declaration and the model "
                "disagree",
            )
        return blocks

    def identity_for(self, component: str) -> Identity | None:
        for identity in self.identities:
            if identity.component == component:
                return identity
        return None


_FAMILIES: dict[str, FamilyAdapter] = {}

#: The registered families, by name — read-only view; ``register_family`` is
#: the one way in, from any module.
FAMILIES: Mapping[str, FamilyAdapter] = MappingProxyType(_FAMILIES)


def register_family(adapter: FamilyAdapter) -> None:
    """Register (or replace) one family. Refused if another family already
    declares one of its mixer children as a *different* stream — the shared
    stream table reads the union, and a child name must mean one stream."""
    for other in _FAMILIES.values():
        if other.family == adapter.family:
            continue
        for child, stream in adapter.mixers.items():
            declared = other.mixers.get(child)
            if declared is not None and declared != stream:
                raise ValueError(
                    f"family {adapter.family!r} declares mixer child {child!r} as "
                    f"{stream!r}, but family {other.family!r} declares it as "
                    f"{declared!r} — a mixer child name means one stream"
                )
    _FAMILIES[adapter.family] = adapter


def family(name: str) -> FamilyAdapter:
    """The registered family ``name``, or an error naming the registered ones."""
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ProtocolError(
            "P4",
            f"no model family named {name!r} is registered (registered: "
            f"{sorted(_FAMILIES)}) — causalab.protocol.registry.register_family",
        ) from None


def _children(module: Any) -> list[str]:
    named = getattr(module, "named_children", None)
    if named is None:
        return []
    return sorted(name for name, _ in named())


def family_for(model: Any) -> FamilyAdapter:
    """The one registered family whose predicate recognizes ``model``'s
    module tree — refusing a tree no family detects, and one several do,
    rather than probing in a fixed order (a wrong tree produces plausible
    numbers; the stream table's rule, applied to families)."""
    hits = [adapter for adapter in _FAMILIES.values() if adapter.detect(model)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ProtocolError(
            "P4",
            f"no registered model family detects this module tree "
            f"({type(model).__name__}, children={_children(model)}); the registered "
            f"families are {sorted(_FAMILIES)}. Register one whose predicate "
            "recognizes the tree (causalab.protocol.registry.register_family) — "
            "detection is structural, never a config-string match",
        )
    raise ProtocolError(
        "P4",
        f"families {sorted(a.family for a in hits)} all detect this module tree "
        f"({type(model).__name__}) — a family's predicate must recognize its own "
        "tree alone, so the registry refuses to pick one by order",
    )


def mixer_children() -> dict[str, Stream]:
    """Every mixer child name a registered family declares, and its stream —
    the table the shared stream check reads."""
    out: dict[str, Stream] = {}
    for adapter in _FAMILIES.values():
        out.update(adapter.mixers)
    return out


def identities_for(name: str) -> tuple[Identity, ...]:
    return family(name).identities


def identity(name: str, component: str) -> Identity:
    """The identity family ``name`` declares for ``component`` — or a refusal
    naming what it does declare (a family with none declares none)."""
    found = family(name).identity_for(component)
    if found is None:
        raise ProtocolError(
            "P4",
            f"family {name!r} declares no reconstruction identity for "
            f"{component!r} (declared: "
            f"{[i.name for i in family(name).identities]})",
        )
    return found


# --- the built-in families: the two module trees the engines were built on -- #

#: The delta kernel's slots, the attention function's, and the experts
#: dispatch's — the function-boundary vocabulary the resolver and the engines
#: share, declared once here and read by ``neural/shared/sites.py``.
DELTA_KERNEL_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "delta_conv": "conv",
        "delta_query": "query",
        "delta_key": "key",
        "delta_value": "value",
        "delta_beta": "beta",
        "delta_decay": "decay",
        "delta_kernel_output": "kernel_output",
        "delta_kv_mem": "kv_mem",
        "delta_state_update": "state_update",
        "delta_state": "state",
    }
)
ATTENTION_FUNCTION_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "attention_query": "query",
        "attention_key": "key",
        "attention_scores": "scores",
        "attention_z": "z",
    }
)
EXPERTS_FUNCTION_SLOTS: Mapping[str, str] = MappingProxyType(
    {
        "expert_gate_proj": "gate_up",
        "expert_up_proj": "gate_up",
        "expert_activation": "activation",
        "expert_neuron_output": "neuron_output",
        "expert_output": "down",
    }
)

#: The taps every block-shaped tree shares: the block's own sides and the
#: mixer's and MLP's outer boundaries, plus the function-boundary interiors
#: whose module is only the anchor of the function tapped.
_BLOCK_TAPS: dict[str, Tap] = {
    "input_ids": Tap("embedding", kind="in"),
    "embeddings": Tap("embedding"),
    "ln_final": Tap("final_norm"),
    "lm_head": Tap("lm_head"),
    "block_input": Tap("block", kind="in"),
    "block_output": Tap("block"),
    "attention_output": Tap("mixer"),
    "mlp_input": Tap("mlp", kind="in"),
    "mlp_output": Tap("mlp"),
    # element 1 of the mixer's (attn_output, attn_weights); the write goes
    # through the attention function (slot "probs")
    "attention_probs": Tap("mixer", tuple_index=1, slot="probs"),
    **{
        c: Tap("mixer", kind="interface", slot=s)
        for c, s in ATTENTION_FUNCTION_SLOTS.items()
    },
    # the module-boundary interior: the child is the row's per-family address
    **{c: Tap("mixer", from_row=True) for c in INTERIOR_ROWS},
}

#: The Llama tree (Llama / Qwen / Mistral / Gemma, and the Qwen3.5-MoE hybrid
#: whose DeltaNet and MoE interiors are declared here because its blocks live
#: in this tree): ``model.layers``, ``input_layernorm`` /
#: ``post_attention_layernorm``, ``self_attn.o_proj``, a SwiGLU ``mlp.act_fn``.
_LLAMA_TAPS: dict[str, Tap] = {
    **_BLOCK_TAPS,
    "attention_input_norm": Tap("block", "input_layernorm"),
    "block_mid": Tap(
        "block", "post_attention_layernorm", "in", writeback="block_output"
    ),
    "mlp_input_norm": Tap("block", "post_attention_layernorm"),
    "attention_premix": Tap("mixer", "o_proj", "in"),
    "attention_result": Tap(
        "mixer",
        "o_proj",
        "in",
        derivation="attention_result",
        shape_of="attention_premix",
    ),
    # act_fn's OUTPUT — act(gate_proj(x)), NOT the down-projection's input
    # (the oracle's semantics, inherited from the pyvene era)
    "mlp_activation": Tap("mlp", "act_fn"),
    "mlp_neuron_output": Tap("mlp", "down_proj", "in"),
    # the Gated DeltaNet mixer: three module sides, then the kernel boundary
    "delta_qkv": Tap("mixer", "in_proj_qkv"),
    "delta_gate": Tap("mixer", "in_proj_z"),
    "delta_premix": Tap("mixer", "out_proj", "in"),
    **{c: Tap("mixer", kind="delta", slot=s) for c, s in DELTA_KERNEL_SLOTS.items()},
    **{
        c: Tap("mixer", kind="interior")
        for c in ("deltanet_query", "deltanet_key", "deltanet_state")
    },
    # the sparse MoE block: a 3-tuple router, a fused experts module, the
    # shared expert's SwiGLU and its mixing scalar
    "router_logits": Tap("mlp", "gate", tuple_index=0),
    "router_scores": Tap("mlp", "gate", tuple_index=1),
    "expert_idx": Tap("mlp", "gate", tuple_index=2),
    "routed_output": Tap("mlp", "experts"),
    **{
        c: Tap("mlp", "experts", kind="experts", slot=s)
        for c, s in EXPERTS_FUNCTION_SLOTS.items()
    },
    "expert_permutation": Tap("mlp", "experts", kind="interior"),
    "shared_expert_gate_proj": Tap("mlp", "shared_expert.gate_proj"),
    "shared_expert_up_proj": Tap("mlp", "shared_expert.up_proj"),
    # down_proj's INPUT: silu(gate_proj(x)) * up_proj(x)
    "shared_expert_activation": Tap("mlp", "shared_expert.down_proj", "in"),
    "shared_expert_output": Tap("mlp", "shared_expert"),
    "shared_expert_gate": Tap("mlp", "shared_expert_gate"),
}

#: The GPT-2 tree: ``transformer.h``, ``ln_1`` / ``ln_2``, a fused
#: ``attn.c_attn`` (the rows' ``fused_blocks`` address) and ``attn.c_proj``,
#: an MLP whose ``c_proj`` INPUT is the activation (the down-projection's
#: input — a different tensor than the llama tree's ``act_fn`` output, and
#: pinned as such by the oracle). No DeltaNet or MoE interior exists on this
#: tree, so none is declared: the stream check and the ``moe`` probe refuse
#: those first, by architecture, exactly as before.
_GPT2_TAPS: dict[str, Tap] = {
    **_BLOCK_TAPS,
    "attention_input_norm": Tap("block", "ln_1"),
    "block_mid": Tap("block", "ln_2", "in", writeback="block_output"),
    "mlp_input_norm": Tap("block", "ln_2"),
    "attention_premix": Tap("mixer", "c_proj", "in"),
    "attention_result": Tap(
        "mixer",
        "c_proj",
        "in",
        derivation="attention_result",
        shape_of="attention_premix",
    ),
    "mlp_activation": Tap("mlp", "c_proj", "in"),
    "mlp_neuron_output": Tap("mlp", "c_proj", "in"),
}

#: The GPT-J tree: GPT-2's root (``transformer.h``, ``transformer.wte``,
#: ``transformer.ln_f``) around a **parallel** block. One norm, ``ln_1``,
#: feeds both the attention and the MLP, and the block adds both outputs to
#: its input in one expression (transformers ``models/gptj/modeling_gptj.py``,
#: ``GPTJBlock.forward``: ``attn_outputs + feed_forward_hidden_states +
#: residual``). So ``block_mid`` and ``mlp_input_norm`` name no tensor here and
#: are not declared: a site naming either is refused by name. The MLP's input
#: *is* ``ln_1``'s output; ``mlp_input`` taps it at the MLP, so a write there
#: reaches the MLP alone, and a write at ``attention_input_norm`` reaches both.
#: The attention keeps separate ``q_proj`` / ``k_proj`` / ``v_proj`` (the
#: interior rows are served by measurement, as for any family the per-family
#: table has not met) and projects out through ``out_proj``, an ``nn.Linear``
#: with no bias; the MLP is ``fc_in`` → ``act`` → ``fc_out``, whose INPUT is
#: the activation (GPT-2's convention, not llama's ``act_fn`` output).
#:
#: Not declared, so refused by name: the attention-function slots
#: (``attention_query`` / ``key`` / ``scores`` / ``z``) and ``attention_probs``.
#: ``GPTJAttention`` computes its pattern in its own ``_attn`` method and never
#: calls the transformers attention interface those slots wrap
#: (``pytorch_hooks/attention_interface.py``), so a tap there would never fire.
_GPTJ_TAPS: dict[str, Tap] = {
    **{
        c: _BLOCK_TAPS[c]
        for c in (
            "input_ids",
            "embeddings",
            "ln_final",
            "lm_head",
            "block_input",
            "block_output",
            "attention_output",
            "mlp_input",
            "mlp_output",
        )
    },
    **{c: Tap("mixer", from_row=True) for c in INTERIOR_ROWS},
    "attention_input_norm": Tap("block", "ln_1"),
    "attention_premix": Tap("mixer", "out_proj", "in"),
    "attention_result": Tap(
        "mixer",
        "out_proj",
        "in",
        derivation="attention_result",
        shape_of="attention_premix",
    ),
    "mlp_activation": Tap("mlp", "fc_out", "in"),
    "mlp_neuron_output": Tap("mlp", "fc_out", "in"),
}

_EXACT: Mapping[str, tuple[float, float]] = MappingProxyType({"fp32": (0.0, 0.0)})

#: The residual identities every block satisfies (spec §2.4: ``block_mid =
#: block_input + attention_output``, ``block_output = block_mid +
#: mlp_output``) — exact, because the taps capture the very tensors the block
#: adds. Declared per family so a new family's plugin is tested by them.
_RESIDUAL_IDENTITIES: tuple[Identity, ...] = (
    Identity(
        "residual_mid",
        "block_mid",
        ("block_input", "attention_output"),
        "block_mid == block_input + attention_output",
        _EXACT,
        additive=True,
    ),
    Identity(
        "residual_out",
        "block_output",
        ("block_mid", "mlp_output"),
        "block_output == block_mid + mlp_output",
        _EXACT,
        additive=True,
    ),
)

LLAMA_TREE = FamilyAdapter(
    family="llama_tree",
    detect=lambda model: (
        walk(model, "model.layers") is not None
        and walk(model, "model.embed_tokens") is not None
    ),
    tree=TreeAddress(
        blocks="model.layers", embedding="model.embed_tokens", final_norm="model.norm"
    ),
    mixers={"self_attn": "full_attention", "linear_attn": "linear_attention"},
    taps=_LLAMA_TAPS,
    identities=(
        *_RESIDUAL_IDENTITIES,
        # 📐 the model computes precisely this sum in this order — exact
        Identity(
            "routed_sum",
            "routed_output",
            ("expert_output", "router_scores"),
            "routed_output == Σ_slot expert_output · router_scores",
            _EXACT,
        ),
        # 📐 pinned against the kernel's own returned states — exact in fp32
        Identity(
            "delta_state_recurrence",
            "delta_state",
            ("delta_decay", "delta_key", "delta_state_update"),
            "S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t  (k̂ = l2norm(delta_key))",
            _EXACT,
        ),
    ),
)


def _some_block_has(
    model: Any, blocks: str, present: tuple[str, ...], absent: tuple[str, ...] = ()
) -> bool:
    """Whether a block under ``blocks`` carries every child path in
    ``present`` and none in ``absent``.

    GPT-2 and GPT-J share their root (``transformer.h``), so a family on that
    root is told apart by what its blocks carry. Any block counts, not only
    block 0, so a tree whose first block is a stand-in (a pipeline stage's
    identity) is still recognized. A tree with no block is recognized by no
    family on this root and is refused by [`family_for`][]."""
    layers = walk(model, blocks)
    if layers is None:
        return False
    try:
        count = len(layers)
    except TypeError:
        return False
    for index in range(count):
        block = layers[index]
        if all(walk(block, path) is not None for path in present) and all(
            walk(block, path) is None for path in absent
        ):
            return True
    return False


GPT2_TREE = FamilyAdapter(
    family="gpt2_tree",
    # 🐞 Was ``transformer.h`` alone, which GPT-J's root also satisfies, so a
    # GPT-J model loaded with this family's taps (``ln_2``, ``attn.c_proj``):
    # modules its blocks do not have. Now the block's own children decide.
    detect=lambda model: _some_block_has(
        model, "transformer.h", present=("ln_2", "attn.c_proj")
    ),
    tree=TreeAddress(
        blocks="transformer.h",
        embedding="transformer.wte",
        final_norm="transformer.ln_f",
    ),
    mixers={"attn": "full_attention"},
    taps=_GPT2_TAPS,
    identities=_RESIDUAL_IDENTITIES,
)

GPTJ_TREE = FamilyAdapter(
    family="gptj_tree",
    # the split projections and ``fc_out`` name the GPT-J block; the absent
    # ``ln_2`` is the parallel residual the identity below relies on
    detect=lambda model: _some_block_has(
        model,
        "transformer.h",
        present=(
            "ln_1",
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.out_proj",
            "mlp.fc_in",
            "mlp.fc_out",
        ),
        absent=("ln_2",),
    ),
    tree=TreeAddress(
        blocks="transformer.h",
        embedding="transformer.wte",
        final_norm="transformer.ln_f",
    ),
    mixers={"attn": "full_attention"},
    taps=_GPTJ_TAPS,
    identities=(
        # 📐 the parallel residual. The inputs are listed in the order the
        # block adds them (``attn_outputs + feed_forward_hidden_states +
        # residual``), because fp32 addition is not associative: evaluated
        # left to right in this order the identity is exact on the tiny
        # fixture (tests/neural/engines/pytorch_hooks/test_family_gptj.py).
        Identity(
            "residual_parallel",
            "block_output",
            ("attention_output", "mlp_output", "block_input"),
            "block_output == attention_output + mlp_output + block_input",
            _EXACT,
            additive=True,
        ),
    ),
)

register_family(LLAMA_TREE)
register_family(GPT2_TREE)
register_family(GPTJ_TREE)


# --- the inventory: one producer of "what exists at which layer" ----------- #


@dataclasses.dataclass(frozen=True)
class LayerInventory:
    """One layer: the mixer stream it carries, the components that exist there
    and, per component, the engines that read it and the mechanisms a write
    may use (``None``: read-only)."""

    layer: int
    stream: Stream
    components: tuple[str, ...]
    reads: Mapping[str, frozenset[str]]
    writes: Mapping[str, frozenset[str] | None]


@dataclasses.dataclass(frozen=True)
class Inventory:
    """The tower's public inventory: every layer, plus the
    layer-less components at the model boundary."""

    model: str
    layers: tuple[LayerInventory, ...]
    layerless: tuple[str, ...]

    def where(self, component: str) -> tuple[int, ...]:
        """The layers ``component`` exists at."""
        return tuple(li.layer for li in self.layers if component in li.components)

    def count(self, stream: Stream) -> int:
        return sum(1 for li in self.layers if li.stream == stream)


def _exists(info: ModelInfo, component: str) -> bool:
    if unavailable_at_load(info, component) is not None:
        return False
    try:
        component_shape(info, component)
    except ValidationError:
        return False
    return True


def inventory(
    target: Any, *, adapter: FamilyAdapter | None = None, serves: Any = None
) -> Inventory:
    """Per layer, the mixer stream, the components present and their read /
    write mechanisms — one producer for ``dry-run``, the generated
    support tables and the inventory tests
    (``tests/golden/test_a3b_inventory.py``).

    ``target`` is a [`ModelInfo`][] (offline: the stream pattern is the
    entry's ``layer_types``, availability is what the rows and the entry can
    decide) or a loaded bundle (anything with ``.info`` and ``.streams``; then
    the streams are the loaded tower's and the bundle's family adapter, if it
    has one, decides which components its tree serves). A component is listed
    at a layer when its row's stream is the layer's or unbound, the entry
    declares no fact against it (``unavailable_at_load``, ``component_shape``)
    and the family, when known, declares a tap for it. ``serves(component,
    layer)`` (``layer`` ``None`` at the model boundary), when given, refines
    that with what a loaded model actually serves — the engines' shared site
    resolver, through ``neural.shared.sites.inventory`` — so a fact only the
    module tree knows (📐 ``tiny-random/qwen3.5-moe``'s config declares a dense
    inner width its MoE blocks do not have) is read off the tree rather than
    off the entry. An entry that declares no layer pattern and is not loaded
    has no inventory — refused, not guessed.
    """
    info: ModelInfo = getattr(target, "info", target)
    streams = getattr(target, "streams", None)
    if streams is None:
        streams = info.layer_types
    if adapter is None:
        adapter = getattr(target, "adapter", None)
    if streams is None:
        raise ValidationError(
            4,
            f"model {info.key!r} declares no layer pattern (layer_types) and is not "
            "loaded, so which mixer each layer carries is unknown — the inventory "
            "is per layer, so there is none to give; load the model, or register "
            "the entry with its layer_types",
        )
    streams = tuple(streams)
    if len(streams) != info.num_layers:
        raise ValidationError(
            4,
            f"model {info.key!r}: {len(streams)} streams for {info.num_layers} layers",
        )
    layerless = tuple(
        c
        for c in COMPONENTS
        if c in LAYERLESS_COMPONENTS
        and _exists(info, c)
        and (adapter is None or adapter.serves(c))
        and (serves is None or serves(c, None))
    )
    layers: list[LayerInventory] = []
    for layer, stream in enumerate(streams):
        present = tuple(
            c
            for c in COMPONENTS
            if c not in LAYERLESS_COMPONENTS
            and CAPABILITIES[c].stream in (None, stream)
            and _exists(info, c)
            and (adapter is None or adapter.serves(c))
            and (serves is None or serves(c, layer))
        )
        layers.append(
            LayerInventory(
                layer=layer,
                stream=stream,
                components=present,
                reads=MappingProxyType({c: CAPABILITIES[c].reads for c in present}),
                writes=MappingProxyType({c: CAPABILITIES[c].writes for c in present}),
            )
        )
    return Inventory(model=info.key, layers=tuple(layers), layerless=layerless)
