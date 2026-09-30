"""The per-engine view of the capability rows, and the typed backend pairs.

Which components an engine serves and which verbs it declares are read off
the rows ([`causalab.protocol.registry.components`][]) and [`ENGINE_VERBS`][]
— an engine class declares ``capabilities = declared_capabilities(name)`` and
nothing by hand; the DeltaNet spellings the two engines reach one tensor by are declared here as
[`BACKEND_PAIRS`][], and the alias table is held to the redirect-not-rebind
rule at import (`_check_aliases`).
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Literal, Mapping, get_args

from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.registry.components import (
    CAPABILITIES,
    ENGINES,
    _HOOKS,  # pyright: ignore[reportPrivateUsage]
    _NNSIGHT,  # pyright: ignore[reportPrivateUsage]
    component_shape,
)
from causalab.protocol.registry.models import get_model_info
from causalab.protocol.schema import COMPONENTS, DEPRECATED_COMPONENTS, DEPRECATED_IN


def components_served_by(engine: str) -> frozenset[str]:
    """The components whose row names ``engine`` — what that engine's class
    declares as ``components`` (and ``writable_components``: write policy is
    protocol-wide, not an engine gap)."""
    if engine not in ENGINES:
        raise AssertionError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    return frozenset(c for c, row in CAPABILITIES.items() if engine in row.reads)


def write_capabilities(engine: str | None = None) -> frozenset[str]:
    """The coarse §8 verbs the rows charge for a write — all of them, or the
    ones ``engine`` acquires by serving the component."""
    served = None if engine is None else components_served_by(engine)
    return frozenset(
        row.write_capability
        for c, row in CAPABILITIES.items()
        if row.write_capability is not None and (served is None or c in served)
    )


#: The engine-level §8 verbs each engine declares — the half of an engine's
#: capability set that is not generated from the rows. A row here and a class
#: that names the engine is all a third engine needs ([`ENGINES`][]).
ENGINE_VERBS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        # The reference engine: every verb in ``engine.CAPABILITIES`` but the
        # three training facts (§2.11), which neither shipped engine honours.
        "pytorch_hooks": frozenset(
            {
                "grad",
                "paired_forward",
                "full_logits",
                "generate",
                # the decode loop is the engine's own (executor._decode_window),
                # so a write hook can stay installed across its steps (§2.9
                # ``writes_during_generation``)
                "generation_writes",
                "pytorch_fn_local",
                "quantized_weights",
            }
        ),
        "nnsight": frozenset(
            {
                "paired_forward",
                "full_logits",
                "pytorch_fn_local",
                # continuation reads through one model.generate trace, decode
                # steps walked with tracer.iter; writes stay in the
                # prefill: the trace binds occurrence 0 of every write's
                # location, so a document asking for ``writes_during_generation``
                # is refused [V13] before the trace is built
                "generate",
            }
        ),
    }
)


def declared_capabilities(engine: str) -> frozenset[str]:
    """``Engine.capabilities`` for ``engine``: its own verbs
    ([`ENGINE_VERBS`][]) plus the write verbs the capability rows charge for
    the components it serves ([`write_capabilities`][]). For the reference
    engine that is ``writable_attention_probs`` — the pattern's write goes
    through the eager attention function, not a hook; for the nnsight engine
    the same verb — the pattern's write lands on the softmax's output *inside*
    the eager function (``attn_weights_2`` in the ``.source`` address table,
    ``nnsight_tracing/addresses.py``), where the value multiply consumes it;
    a write to the mixer's returned ``attn_weights`` would reach nothing."""
    if engine not in ENGINE_VERBS:
        raise AssertionError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    return ENGINE_VERBS[engine] | write_capabilities(engine)


def component_capability(component: str, *, write: bool = False) -> str:
    """The generated capability entry for serving ``component`` (§8) —
    reading it, or with ``write=True``, landing a write on it."""
    return f"component:{component}:write" if write else f"component:{component}"


def effective_capabilities(engine: str) -> frozenset[str]:
    """What the engine ``--engine`` names is held to — rule 13's comparison
    against ``requires``, a shortfall refused ``[V13]``; nothing chooses
    between engines. The routed set of a registered engine *by name*:
    what ``pipeline._offered`` reads when ``validate`` is handed a name and no
    instance exists yet (torch-free, before any weights) —
    [`declared_capabilities`][] plus a read and a write entry for every
    component the engine's row serves. The instance-side twin is
    ``Engine.effective_capabilities`` (an engine need not be a row — the
    stubs in ``tests/protocol/test_legality_before_weights.py`` are not); for
    the two real engines the rows and the classes give one answer, which
    ``tests/protocol/test_registry_package.py`` pins. Write *policy* is
    protocol-wide, not an engine gap, so served and writable are one set."""
    served = components_served_by(engine)
    return (
        declared_capabilities(engine)
        | {component_capability(c) for c in served}
        | {component_capability(c, write=True) for c in served}
    )


#: The model the component table in ``docs/qwen36_35b_a3b.md`` is
#: drawn for — the one hybrid, MoE checkpoint the vocabulary was built to
#: address, so every row has a "blocks" count.
DOCS_TABLE_MODEL = "Qwen/Qwen3.6-35B-A3B"


# --- typed backend pairs: two spellings, one tensor, a declared relation ---- #

#: How the reference engine's and the nnsight engine's captures of one
#: DeltaNet tensor line up (📐 measured on the public Hub fixture
#: ``tiny-random/qwen3.5-moe``, 2026-08-28;
#: ``tests/golden/test_a3b_engine_parity.py`` repeats it on
#: ``Qwen/Qwen3.6-35B-A3B``):
#:
#: ``identical``
#:     same shape, max abs diff 0.0 — **one name**: the nnsight spelling is an
#:     alias of the reference one (``schema.DEPRECATED_COMPONENTS``);
#: ``gva_tile``
#:     the reference engine's tensor is post ``repeat_interleave`` over the
#:     head axis (value-head space), the nnsight one pre (key-head space);
#:     exact after tiling — **two names**;
#: ``chunk_boundary``
#:     per step versus per 64-token chunk; the chunk's state is the step
#:     state at the chunk's last position — **two names**.
Relation = Literal["identical", "gva_tile", "chunk_boundary"]
RELATIONS: tuple[Relation, ...] = get_args(Relation)


@dataclasses.dataclass(frozen=True)
class BackendPair:
    """One DeltaNet tensor as the two engines reach it, and the typed relation
    between the two captures. An ``identical`` pair is an alias (one name,
    two mechanisms); any other relation is a **typed backend requirement**:
    two names, each served by the engine named, related by the declared
    transform — which the test helpers read from here rather than own."""

    hooks: str
    nnsight: str
    relation: Relation
    why: str
    #: the kernel's chunk length, for ``chunk_boundary`` (📐 read off the
    #: kernel's own loop, not off config)
    chunk: int | None = None

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"relation {self.relation!r} not in {list(RELATIONS)}")
        if (self.relation == "chunk_boundary") != (self.chunk is not None):
            raise ValueError(
                "chunk_boundary pairs declare the chunk length; others none"
            )

    @property
    def aliased(self) -> bool:
        return self.relation == "identical"

    @property
    def names(self) -> frozenset[str]:
        return frozenset({self.hooks, self.nnsight})


BACKEND_PAIRS: tuple[BackendPair, ...] = (
    *(
        BackendPair(hooks, nnsight, "identical", "same shape, max abs diff 0.0")
        for hooks, nnsight in (
            ("delta_qkv", "deltanet_qkv"),
            ("delta_conv", "deltanet_qkv_conv"),
            ("delta_gate", "deltanet_gate"),
            ("delta_value", "deltanet_value"),
            ("delta_beta", "deltanet_beta"),
            ("delta_decay", "deltanet_decay"),
            ("delta_kernel_output", "deltanet_core_out"),
            ("delta_premix", "deltanet_gated_out"),
        )
    ),
    BackendPair(
        "delta_query",
        "deltanet_query",
        "gva_tile",
        "delta_query is the kernel's argument, tiled to the value-head count "
        "(post repeat_interleave); deltanet_query is the projection before the "
        "tiling, in key-head space — different shapes, exact after tiling",
    ),
    BackendPair(
        "delta_key",
        "deltanet_key",
        "gva_tile",
        "delta_key is the kernel's argument, tiled to the value-head count "
        "(post repeat_interleave); deltanet_key is the projection before the "
        "tiling, in key-head space — different shapes, exact after tiling",
    ),
    BackendPair(
        "delta_state",
        "deltanet_state",
        "chunk_boundary",
        "delta_state is the recurrent state per step; deltanet_state is the "
        "chunked kernel's state once per 64-token chunk — different timing; the "
        "chunk's state is the step state at the chunk's last position",
        chunk=64,
    ),
)


def backend_pair(component: str) -> BackendPair:
    """The pair ``component`` (either spelling) belongs to."""
    for pair in BACKEND_PAIRS:
        if component in pair.names:
            return pair
    raise AssertionError(f"{component!r} is not a spelling of any backend pair")


def alias_would_rebind(alias: str, canonical: str) -> str | None:
    """Why folding ``alias`` onto ``canonical`` would **rebind** rather than
    redirect — or ``None`` when the two name one tensor in one shape at one
    time. The rule the alias table is held to (``test_registry_shapes.py``:
    an alias that redirects is safe; one that rebinds lets a document load
    and silently mean a different tensor): a declared backend relation other
    than ``identical`` refuses by name, and two vocabulary names whose
    declared shapes differ on the reference entry refuse by shape."""
    for pair in BACKEND_PAIRS:
        if {alias, canonical} == pair.names and not pair.aliased:
            return (
                f"aliasing {alias!r} to {canonical!r} would rebind, not redirect: "
                f"the two are related by {pair.relation!r} — {pair.why}. Two "
                "names with a typed backend requirement stay two names."
            )
    if alias in CAPABILITIES and canonical in CAPABILITIES:
        info = get_model_info(DOCS_TABLE_MODEL)
        try:
            left, right = component_shape(info, alias), component_shape(info, canonical)
        except ValidationError:
            return None
        if left.describe() != right.describe() or left.width != right.width:
            return (
                f"aliasing {alias!r} to {canonical!r} would rebind, not redirect: "
                f"their shapes differ on {info.key!r} — {left.describe()} "
                f"(width {left.width}) versus {right.describe()} (width "
                f"{right.width})"
            )
    return None


# --------------------------------------------------------------------------- #
# the alias table is held to the redirect rule at import
# --------------------------------------------------------------------------- #


def _check_aliases(table: Mapping[str, str] | None = None) -> None:
    """Every retired spelling redirects: it is out of the vocabulary, its
    replacement is in, it has a deprecation version, and no declared backend
    relation or shape says it would rebind. Every backend pair agrees with the
    alias table: an ``identical`` pair is an alias, any other pair is two
    single-engine rows. Refused at import — a vocabulary defect is a bug in
    this module, never a document error. ``table`` is the alias table under
    test (the vocabulary's own by default; a test hands in a mutated one)."""
    table = DEPRECATED_COMPONENTS if table is None else table
    for alias, target in table.items():
        if (
            alias in COMPONENTS
            and alias != target
            and alias not in {p.nnsight for p in BACKEND_PAIRS}
        ):
            raise AssertionError(f"alias {alias!r} is still in the vocabulary")
        if target not in COMPONENTS:
            raise AssertionError(f"alias {alias!r} redirects to unknown {target!r}")
        if alias not in DEPRECATED_IN and table is DEPRECATED_COMPONENTS:
            raise AssertionError(f"alias {alias!r} has no deprecation version")
        reason = alias_would_rebind(alias, target)
        if reason is not None:
            raise AssertionError(reason)
    for pair in BACKEND_PAIRS:
        if pair.aliased:
            if table.get(pair.nnsight) != pair.hooks:
                raise AssertionError(
                    f"identical pair {pair.hooks!r}/{pair.nnsight!r} is not an alias"
                )
        else:
            for name, engine in ((pair.hooks, _HOOKS), (pair.nnsight, _NNSIGHT)):
                if name not in CAPABILITIES or CAPABILITIES[name].reads != engine:
                    raise AssertionError(
                        f"{name!r} of the {pair.relation!r} pair must be a row "
                        f"served by exactly {sorted(engine)}"
                    )


_check_aliases()
