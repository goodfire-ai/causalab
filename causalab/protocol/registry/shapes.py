"""Typed feature shapes: what axes a tapped tensor has, and which one is features.

The executor works in one shape throughout — ``(batch, position, feature)``.
Until now a tap declared how its native tensor relates to that contract with a
``Layout`` *string* (since retired) (``"bsd"``,
``"flat_td"``, ``"bds"``, ``"bs"``, ``"native"``), and everything a string could
not express was written out by hand somewhere else: the head bound in
``explicit._canon_site``, the per-component ``ValidationError`` texts in
``registry.component_width``, and ``executor._whole_attention_pattern``'s three
refusals.

A [`FeatureShape`][] says the same things once, as data:

* **what the axes are, in the module's native order** — so ``to_contract`` and
  ``from_contract`` are *computed* rather than enumerated;
* **whether there is a head axis, and how wide** — so ``head: 3`` on a KV-space
  component is refused instead of silently slicing past the end;
* **whether the values are a feature space at all** — integer ids and a
  per-token ranking are not, and the refusals featurizers need are generated
  from that rather than written per component.

Two flattenings, and why they are booleans rather than axis kinds
----------------------------------------------------------------

``axes`` always lists the tensor's axes *logically*, most-significant first
within each group. A native tensor may carry several of them in one dimension,
and exactly two such packings occur in the wild:

``flat_batch``
    ``(batch, position)`` share one native dimension. ``Qwen3_5MoeSparseMoeBlock``
    reshapes to ``(-1, hidden)`` before the router, so its whole interior is
    flattened over (batch, position). Recovering the split needs the batch size,
    which is why the conversions take one.

``flat_inner``
    every axis after the outer ones — head, fused, feature — shares one native
    dimension, row-major in declared order. This is the common case: the
    o-projection's input is ``(b, s, H*d)``, not ``(b, s, H, d)``. Attention's
    *interior* taps are the exception (``q_norm`` really does emit
    ``(b, s, H, d)``), which is the whole reason the distinction has to be
    recorded rather than assumed.

Both are stated as facts about the native tensor, and the rank check derives
from them: a tap whose real tensor disagrees raises rather than being silently
reinterpreted.

What ``is_feature_space`` gates
-------------------------------

A shape has a **contract form** when it has exactly one position axis. Without
one there is no ``(batch, position, feature)`` to convert to, and the
conversions are the identity — that is the honest version of the old
``"native"`` marker, which said "undescribed"; this says *why*.

``is_feature_space`` is narrower: a contract form, values that are not integer
labels, and at least one inner axis. Only then is there a basis a featurizer can
attach to. ``ranking`` is narrower still — the axis has a width and a contract
form, but column *k* is the *k*-th ranked expert, a different expert for
different tokens, so a basis fitted across positions is fitted across a shuffled
basis (see ``router_scores``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, get_args

__all__ = [
    "Axis",
    "AxisKind",
    "FeatureShape",
    "INNER_KINDS",
    "OUTER_KINDS",
    "attention_pattern",
    "bds",
    "bs",
    "bsh",
    "bs_flat_heads",
    "bs_fused_blocks",
    "bs_fused_heads",
    "bshd",
    "bhsd",
    "bsd",
    "chunked_state",
    "flat_td",
    "flat_topk",
    "flat_topk_features",
    "flat_topk_fused_features",
    "state_matrix",
]

#: What one axis of a tapped tensor means.
#:
#: ``"fused"`` is the odd one: it is an axis the component does **not** span —
#: ``Qwen3_5MoeAttention``'s q-projection emits ``[q_h | gate_h]`` per head, so
#: ``attention_gate`` names sub-split 1 of 2. A shape that declares a fused axis
#: also declares which split it means, and the axis disappears from the contract.
AxisKind = Literal[
    "batch",
    "position",
    "key_position",
    "head",
    "fused",
    "feature",
    "topk",
    "state",
]

#: Axes that make up the contract's single feature axis, in this order.
#: ``"state"`` is deliberately absent: a recurrent state's trailing axes form a
#: d_k × d_v **matrix** per head, and flattening a matrix into "the feature
#: axis" would let featurizers and ``dims`` index a space that is not a basis.
INNER_KINDS: frozenset[str] = frozenset({"head", "fused", "feature", "topk"})

#: Axes that do not. ``key_position`` is here rather than among the inner kinds
#: on purpose: an attention pattern's last axis is a *position* axis wearing the
#: feature axis' place, and calling it a feature is the mistake this module
#: exists to make unspellable.
OUTER_KINDS: frozenset[str] = frozenset({"batch", "position", "key_position"})

_ALL_KINDS: tuple[str, ...] = get_args(AxisKind)


@dataclasses.dataclass(frozen=True)
class Axis:
    """One axis: what it means, and how wide it is when that is a static fact.

    ``width=None`` marks a runtime-sized axis — batch and position counts are
    properties of the encoded batch, never of the model config.
    """

    kind: AxisKind
    width: int | None = None
    #: Disambiguates two axes of the same kind. The only live use is naming an
    #: attention pattern's query axis, so that ``key_position`` is not merely
    #: "the other one".
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ALL_KINDS:
            raise ValueError(f"unknown axis kind {self.kind!r}; expected {_ALL_KINDS}")
        if self.width is not None and self.width <= 0:
            raise ValueError(f"axis {self.kind!r} has non-positive width {self.width}")

    @property
    def label(self) -> str:
        return self.kind if self.name is None else f"{self.kind}[{self.name}]"


@dataclasses.dataclass(frozen=True)
class FeatureShape:
    """The axes of one component's tensor, in the module's native order."""

    axes: tuple[Axis, ...]
    #: ``(batch, position)`` occupy one native dimension (the MoE interior).
    flat_batch: bool = False
    #: The inner axes occupy one native dimension, row-major. Vacuous — and so
    #: harmless — when there is at most one inner axis, which is why it defaults
    #: to true: the exceptional tap is the one that keeps its head axis.
    flat_inner: bool = True
    #: Which sub-split of the ``"fused"`` axis this component names.
    fused_index: int | None = None
    #: Integer labels, not features: no featurizer, no gradient. Token ids and
    #: the routing table.
    integral: bool = False
    #: The inner axis is a per-token *ranking*, so its columns are not a fixed
    #: basis. Basis-fitting featurizers are refused; identity reads are fine.
    ranking: bool = False
    #: The "…so do this instead" half of a refusal — the part no descriptor can
    #: generate. Appended to the generated message by [`refusal`][].
    note: str | None = None

    def __post_init__(self) -> None:
        kinds = [a.kind for a in self.axes]
        if not self.axes:
            raise ValueError("a FeatureShape needs at least one axis")
        if kinds.count("batch") != 1:
            raise ValueError(f"a shape needs exactly one batch axis, got {kinds}")
        if kinds.count("fused") > 1:
            raise ValueError(f"at most one fused axis, got {kinds}")
        has_fused = "fused" in kinds
        if has_fused != (self.fused_index is not None):
            raise ValueError(
                "a fused axis and fused_index are declared together — "
                f"axes={kinds}, fused_index={self.fused_index}"
            )
        if has_fused:
            fused = next(a for a in self.axes if a.kind == "fused")
            if fused.width is None:
                raise ValueError("a fused axis needs a static split count")
            assert self.fused_index is not None
            if not 0 <= self.fused_index < fused.width:
                raise ValueError(
                    f"fused_index {self.fused_index} out of range for "
                    f"{fused.width} splits"
                )
        if self.flat_batch:
            # the split is `view(batch, -1, ...)`, so the two must be adjacent
            # and in this order for the flattening to be row-major
            try:
                b = kinds.index("batch")
            except ValueError:  # pragma: no cover - guarded above
                raise
            if kinds[b : b + 2] != ["batch", "position"]:
                raise ValueError(
                    "flat_batch packs (batch, position) into one dimension, so "
                    f"they must be adjacent and in that order; got {kinds}"
                )
        inner_positions = [i for i, a in enumerate(self.axes) if a.kind in INNER_KINDS]
        if (
            self.flat_inner
            and len(inner_positions) > 1
            and inner_positions
            != list(range(inner_positions[0], inner_positions[-1] + 1))
        ):
            # Only the packing needs adjacency. An unpacked shape may interleave
            # them — the attention interface's q/k/v really are
            # (batch, head, position, head_dim) — and the permute in
            # `to_contract` puts them back in declared order.
            raise ValueError(
                f"flat_inner packs the inner axes into one dimension, so they "
                f"must be adjacent; got {kinds}"
            )
        for axis in self.inner_axes:
            if axis.width is None:
                raise ValueError(f"inner axis {axis.label!r} needs a static width")
        for axis in self.state_axes:
            if axis.width is None:
                raise ValueError(f"state axis {axis.label!r} needs a static width")

    # -- axis groups ------------------------------------------------------- #

    @property
    def inner_axes(self) -> tuple[Axis, ...]:
        """Head, fused and feature axes — what the contract's feature axis is
        made of, in declared order."""
        return tuple(a for a in self.axes if a.kind in INNER_KINDS)

    @property
    def feature_axes(self) -> tuple[Axis, ...]:
        """The inner axes that survive into the contract — every inner axis but
        the fused one, which a component selects a single split of."""
        return tuple(a for a in self.inner_axes if a.kind != "fused")

    @property
    def position_axes(self) -> tuple[Axis, ...]:
        return tuple(a for a in self.axes if a.kind in ("position", "key_position"))

    @property
    def state_axes(self) -> tuple[Axis, ...]:
        """The matrix axes of a recurrent state — the second
        ``is_feature_space == False`` citizen after the attention pattern.
        Unlike the pattern it keeps its one position axis, so position
        addressing works; what it lacks is a feature *vector*, so featurizers
        and ``dims`` are refused and the tensor crosses the executor in its
        native layout."""
        return tuple(a for a in self.axes if a.kind == "state")

    @property
    def head_space(self) -> int | None:
        """How many heads a ``head`` sub-axis selects among — or ``None`` when
        the component has no head axis, which makes ``head`` on it an error
        rather than something to ignore.

        This is the fix for the bound that used to read ``info.num_heads``
        regardless of component: three attention-interior components live in KV-head space,
        where that bound is too wide by the GQA ratio and the over-wide slice is
        empty rather than out of range — a read of nothing and a write that
        changes nothing.
        """
        for axis in self.axes:
            if axis.kind == "head":
                return axis.width
        return None

    @property
    def head_axis_index(self) -> int | None:
        """The native dimension a per-head shard runs along
        (``docs/model_parallelism.md`` §6.2): the dimension whose packed axes
        *lead* with the head axis — a bare head axis (``(b, H, s, d)``, dim
        1), or a head-major packing (``(b, s, H·d)``, dim 2), where a
        contiguous chunk of the dimension is a contiguous block of whole
        heads. ``None`` when the shape has no head axis, or when the head axis
        does not lead its dimension (GPT-2's fused ``[q | k | v]`` blocks put
        the fused axis outside the heads, so a chunk of that dimension is not
        whole heads of one split).
        """
        for index, group in enumerate(self.native_groups):
            if group[0].kind == "head":
                return index
        return None

    # -- what the shape permits -------------------------------------------- #

    @property
    def has_contract_form(self) -> bool:
        """Whether ``(batch, position, feature)`` is even meaningful here.

        False for an attention pattern, whose second position axis has nowhere
        to go: the executor's gather indexes dim 1 with positions and its
        ``dims`` slices dim -1 as features, and on that tensor both would read a
        different tensor than the author named.
        """
        return len(self.position_axes) == 1

    @property
    def is_feature_space(self) -> bool:
        """Whether a featurizer may attach: a contract form, real values rather
        than labels, an axis to attach to — and no state matrix, whose
        "features" are a d_k × d_v matrix rather than a vector."""
        return (
            self.has_contract_form
            and not self.integral
            and bool(self.feature_axes)
            and not self.state_axes
        )

    @property
    def width(self) -> int | None:
        """The contract's feature width, or ``None`` when there is no feature
        axis to measure (or when the trailing axes form a state matrix, which
        has an element count but no basis to be wide *in*)."""
        if not self.has_contract_form or not self.feature_axes or self.state_axes:
            return None
        return math.prod(a.width or 1 for a in self.feature_axes)

    # -- native shape ------------------------------------------------------ #

    @property
    def native_groups(self) -> tuple[tuple[Axis, ...], ...]:
        """The native dimensions, each as the axes packed into it."""
        groups: list[tuple[Axis, ...]] = []
        i = 0
        axes = self.axes
        while i < len(axes):
            axis = axes[i]
            if self.flat_batch and axis.kind == "batch":
                groups.append((axes[i], axes[i + 1]))
                i += 2
                continue
            if self.flat_inner and axis.kind in INNER_KINDS:
                inner = self.inner_axes
                groups.append(inner)
                i += len(inner)
                continue
            groups.append((axis,))
            i += 1
        return tuple(groups)

    @property
    def native_rank(self) -> int:
        return len(self.native_groups)

    def describe(self) -> str:
        """A human-readable native shape, for error messages."""
        inner = ", ".join(
            "·".join(a.label for a in group) for group in self.native_groups
        )
        return f"({inner})"

    def refusal(self, what: str) -> str:
        """The generated half of a refusal message, plus this shape's note."""
        if self.integral:
            why = (
                "it carries integer labels on "
                f"{self.describe()}, so there is no width for a featurizer to "
                "match and no gradient to train through"
            )
        elif not self.has_contract_form:
            names = ", ".join(a.label for a in self.position_axes)
            why = (
                f"its axes are {self.describe()} — it has two position axes "
                f"({names}), so its feature axis IS a position axis"
            )
        elif self.state_axes:
            names = " × ".join(a.label for a in self.state_axes)
            why = (
                f"its axes are {self.describe()} — its trailing axes ({names}) "
                "form a matrix per head, not a feature vector, so there is no "
                "basis to fit or index"
            )
        elif not self.feature_axes:
            why = f"its axes are {self.describe()}, with no feature axis at all"
        else:  # pragma: no cover - every refusing shape hits a branch above
            why = f"its axes are {self.describe()}"
        message = f"{what} is not a feature space: {why}"
        return f"{message}. {self.note}" if self.note else f"{message}."


# --------------------------------------------------------------------------- #
# constructors — the five layout strings, plus the attention-interior shapes.
# Named so that a tap table reads as data, and so that the string vocabulary
# they replace stays greppable.
# --------------------------------------------------------------------------- #

_BATCH = Axis("batch")
_POSITION = Axis("position")


def bsd(width: int, *, note: str | None = None) -> FeatureShape:
    """``(batch, position, feature)`` — the contract itself."""
    return FeatureShape(axes=(_BATCH, _POSITION, Axis("feature", width)), note=note)


def flat_td(width: int, *, note: str | None = None) -> FeatureShape:
    """``(batch*position, feature)`` — the MoE interior."""
    return FeatureShape(
        axes=(_BATCH, _POSITION, Axis("feature", width)),
        flat_batch=True,
        note=note,
    )


def bds(width: int, *, note: str | None = None) -> FeatureShape:
    """``(batch, feature, position)`` — channels-first, the DeltaNet conv1d."""
    return FeatureShape(axes=(_BATCH, Axis("feature", width), _POSITION), note=note)


def bs(*, integral: bool = False, note: str | None = None) -> FeatureShape:
    """``(batch, position)`` — no feature axis at all; one integer per position."""
    return FeatureShape(axes=(_BATCH, _POSITION), integral=integral, note=note)


def flat_topk(
    k: int,
    *,
    integral: bool = False,
    ranking: bool = False,
    note: str | None = None,
) -> FeatureShape:
    """``(batch*position, k)`` — a top-k axis in the flattened MoE interior."""
    return FeatureShape(
        axes=(_BATCH, _POSITION, Axis("topk", k)),
        flat_batch=True,
        integral=integral,
        ranking=ranking,
        note=note,
    )


def flat_topk_features(
    k: int,
    width: int,
    *,
    ranking: bool = False,
    note: str | None = None,
) -> FeatureShape:
    """``(batch*position, k·width)`` — the routed-expert interior, token-major.

    One row per token, ``k`` slots of ``width`` features each, in ranking
    order: slot *j* holds the value the token's *j*-th ranked expert computed.
    That is ``router_scores``' situation exactly, which is what ``ranking``
    says — a basis fitted across positions is fitted across a shuffled basis.
    Consumers join slots to experts through ``expert_idx``, whose
    ``(batch·position, k)`` rows are the same slots. The serving engine
    presents each tensor in this token-major layout whatever row order the
    kernel computed in: grouped_mm's expert-sorted layout is an
    implementation detail the eager loop does not even share, so it never
    reaches the vocabulary.
    """
    return FeatureShape(
        axes=(_BATCH, _POSITION, Axis("topk", k), Axis("feature", width)),
        flat_batch=True,
        ranking=ranking,
        note=note,
    )


def flat_topk_fused_features(
    k: int,
    splits: int,
    index: int,
    width: int,
    *,
    ranking: bool = False,
    note: str | None = None,
) -> FeatureShape:
    """``(batch*position, k·splits·width)``, naming one split per slot.

    The routed experts' up-projection emits ``[gate_e | up_e]`` per (token,
    slot) pair in one tensor; ``expert_gate_proj`` is split 0 and
    ``expert_up_proj`` split 1 — one capture, two addresses, the
    ``attention_gate`` precedent with a top-k axis in front.
    """
    return FeatureShape(
        axes=(
            _BATCH,
            _POSITION,
            Axis("topk", k),
            Axis("fused", splits),
            Axis("feature", width),
        ),
        flat_batch=True,
        fused_index=index,
        ranking=ranking,
        note=note,
    )


def chunked_state(
    heads: int, k_dim: int, v_dim: int, *, note: str | None = None
) -> FeatureShape:
    """``(batch, chunk, heads, k_dim·v_dim)`` — a recurrent state, once per
    kernel chunk.

    The DeltaNet state: its position axis is the **chunk index**
    of the serving kernel (one fire per 64-token prefill chunk), not a token
    position — the axis' name says so, and the executor resolves positions on
    it against the fire count rather than the sequence. Each state is a
    ``(k_dim, v_dim)`` matrix per head, flattened into the feature axis.
    """
    return FeatureShape(
        axes=(
            _BATCH,
            Axis("position", name="chunk"),
            Axis("head", heads),
            Axis("feature", k_dim * v_dim),
        ),
        flat_inner=False,
        note=note,
    )


def bs_flat_heads(
    heads: int, head_dim: int, *, note: str | None = None
) -> FeatureShape:
    """``(batch, position, heads*head_dim)`` — head-major, already flattened.

    The o-projection's input, and every projection output on a family that does
    not keep a head axis.
    """
    return FeatureShape(
        axes=(_BATCH, _POSITION, Axis("head", heads), Axis("feature", head_dim)),
        note=note,
    )


def bsh(heads: int, *, note: str | None = None) -> FeatureShape:
    """``(batch, position, heads)`` — one scalar per head per position.

    The DeltaNet kernel's per-head gates (``beta``, ``g``): the feature axis
    *is* the head axis, so ``head:`` selects a width-1 column and a featurizer
    attaches to an 8-wide space whose basis is the fixed head list.
    """
    return FeatureShape(axes=(_BATCH, _POSITION, Axis("head", heads)), note=note)


def bshd(heads: int, head_dim: int, *, note: str | None = None) -> FeatureShape:
    """``(batch, position, heads, head_dim)`` — ``q_norm``/``k_norm``'s output."""
    return FeatureShape(
        axes=(_BATCH, _POSITION, Axis("head", heads), Axis("feature", head_dim)),
        flat_inner=False,
        note=note,
    )


def bhsd(
    heads: int,
    head_dim: int,
    *,
    position_name: str | None = None,
    note: str | None = None,
) -> FeatureShape:
    """``(batch, heads, position, head_dim)`` — the attention interface's q/k/v.

    ``position_name`` says *which* positions the axis runs over. It matters for
    exactly one thing: the keys' axis runs over the positions being attended
    *to*, which under a KV cache is the whole prefix and grows by one per decode
    step, while the queries' axis is the step itself. Naming it is what lets the
    executor refuse a continuation read of the keys without a per-component
    list.
    """
    return FeatureShape(
        axes=(
            _BATCH,
            Axis("head", heads),
            Axis("position", name=position_name),
            Axis("feature", head_dim),
        ),
        flat_inner=False,
        note=note,
    )


def bs_fused_heads(
    heads: int, splits: int, index: int, head_dim: int, *, note: str | None = None
) -> FeatureShape:
    """``(batch, position, heads*splits*head_dim)``, naming one split.

    Qwen3.5/3.6's q-projection emits ``[q_h | gate_h]`` per head in one tensor;
    ``attention_gate`` is split 1 of 2.
    """
    return FeatureShape(
        axes=(
            _BATCH,
            _POSITION,
            Axis("head", heads),
            Axis("fused", splits),
            Axis("feature", head_dim),
        ),
        fused_index=index,
        note=note,
    )


def bs_fused_blocks(
    splits: int, index: int, heads: int, head_dim: int, *, note: str | None = None
) -> FeatureShape:
    """``(batch, position, splits*heads*head_dim)``, naming one split — the
    splits as contiguous **blocks**, each head-major inside.

    📐 GPT-2's fused ``c_attn`` emits ``[q | k | v]`` as three ``H·d``-wide
    column blocks (``.split(split_size, dim=2)``, ``split_size = H·d``), so
    ``attention_query_pre_rope`` there is split 0 of 3 — the fused axis sits
    *outside* the head axis, where [`bs_fused_heads`][] (Qwen3.5's
    ``[q_h | gate_h]`` per head) has it inside.
    """
    return FeatureShape(
        axes=(
            _BATCH,
            _POSITION,
            Axis("fused", splits),
            Axis("head", heads),
            Axis("feature", head_dim),
        ),
        fused_index=index,
        note=note,
    )


def state_matrix(
    heads: int, d_k: int, d_v: int, *, note: str | None = None
) -> FeatureShape:
    """``(batch, steps, heads, d_k, d_v)`` — the recurrent state, one matrix
    per head per step. The position axis runs over decode/prompt *steps*, so
    position addressing works; the trailing two axes are a matrix, so
    featurizers and ``dims`` refuse (see ``state_axes``)."""
    return FeatureShape(
        axes=(
            _BATCH,
            Axis("position", name="steps"),
            Axis("head", heads),
            Axis("state", d_k),
            Axis("state", d_v),
        ),
        note=note,
    )


def attention_pattern(heads: int, *, note: str | None = None) -> FeatureShape:
    """``(batch, heads, query, key)`` — the shape with two position axes.

    Replaces the ``"native"`` marker, which said only "undescribed". This is
    described; it simply has no contract form, and every refusal the executor
    used to spell out by hand follows from that.
    """
    return FeatureShape(
        axes=(
            _BATCH,
            Axis("head", heads),
            Axis("position", name="query"),
            Axis("key_position", name="key"),
        ),
        flat_inner=False,
        note=note,
    )
