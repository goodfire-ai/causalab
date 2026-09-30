"""The style → placement table (``docs/model_parallelism.md`` §4) as code.

transformers' parallel styles replace a module's ``forward`` with a wrapped
``tp_forward``; ``Module.__call__`` runs our forward hooks *after* it and our
pre-hooks *before* it, so where a style's own collective sits decides what
each hook sees. The table, row by row:

==================================  ==============  ==========================
style at the module                 pre-hook sees   forward hook sees
==================================  ==============  ==========================
``colwise``                         Replicated      Sharded(axis, group)
``packed_colwise``                  Replicated      Sharded(axis, group, slots=2) — gate | up
``rowwise``                         Sharded(axis)   Replicated (its own all-reduce ran)
``replicated_with_grad_allreduce``  Sharded(axis)   Sharded(axis) — the *parameter*
                                                    is replicated; the activation
                                                    through it (Qwen's ``q_norm`` /
                                                    ``k_norm``) is the colwise
                                                    producer's local heads
``colwise_gather_output``           Replicated      Replicated
``kv_replicated``                   Replicated      Sharded(axis, group, repeat) —
                                                    the weight is whole on every
                                                    rank; the output is the one
                                                    KV head this rank's query
                                                    heads read, held by ``repeat``
                                                    consecutive ranks (§6.6)
``moe_tp_experts``                  Replicated      Replicated — the module output;
                                                    the token-major *interior* the
                                                    experts interface taps is
                                                    [`experts_placement`][]'s
                                                    answer from the parameter rows
                                                    (§6.3)
``grouped_gemm`` (the parameters)   Replicated      Replicated — the module output;
                                                    the interior ExpertLocal (§6.3)
``ep_router``                       Replicated      by output position:
                                                    ``router_logits`` Replicated,
                                                    ``router_scores`` ExpertLocal
                                                    (zeroed off this rank's
                                                    experts), ``expert_idx``
                                                    Replicated and **remapped** —
                                                    this rank's local ids, the
                                                    executor reconstructs the
                                                    global table
no row                              Replicated      Replicated
==================================  ==============  ==========================

``axis`` is the native axis a tensor-parallel style shards at the tap
([`shard_axis`][]): the head axis when the tap's shape keeps one — Qwen's
normed queries ``(batch, position, head, feature)``, the attention pattern
``(batch, head, query, key)`` — else the flat feature axis, ``-1``. The
chunks are contiguous heads per rank (``Shard(0)`` of the projection), so
the gather in rank order is the global head order.

Three components carry a *neighbour's* tensor ([`INHERITED`][]): two sit
on a module no row names between two planned projections — the dense
``mlp_activation`` (``act_fn``, the rowwise ``down_proj``'s input) and the
attention pattern the mixer returns (``attention_probs``, the colwise
``q_proj``'s heads) — and the key before RoPE (``attention_key_pre_rope``)
carries ``k_proj``'s output: on a family with a ``k_norm`` the norm's own row
replicates its *weight* and says nothing about whether the projection under
it is sharded or replicated (§6.6).

The **function interiors** (§6.2–6.4) have no row of their own — the function
runs inside a module no row names — so their placement is derived from the
rows of the module's *children*:

* the attention function ([`interior_placement`][]): a mixer whose
  projections carry a ``colwise`` / ``packed_colwise`` row on an active axis
  computes with local heads, so every slot is ``Sharded(head_axis, axis)`` on
  the head axis of the slot's own shape (dim 1 for ``(b, H, s, d)`` and the
  pattern, dim 2 for ``z``'s ``(b, s, H, d)``) — and the pattern the mixer
  *returns* is the same local-heads tensor; the ``key`` slot under a
  ``kv_replicated`` K/V row is the repeated shard that row's output is
  (§6.6). A mixer whose projections are all gathered (every Qwen DeltaNet
  projection is ``colwise_gather_output``) runs replicated: all ten delta
  slots, by the same rule;
* the experts dispatch ([`experts_placement`][]): ``grouped_gemm`` rows on
  the expert parameters make the token-major view ``ExpertLocal(axis)`` with
  the remapped routing table; ``packed_colwise`` / ``rowwise`` rows (the
  experts under tensor parallelism) make the ``[gate | up]`` and activation
  views ``Sharded(-1, axis, slots=…)`` — each (token, slot) run sharded on
  the neuron axis — and the down-projection's view a ``PartialSum(axis)``
  the module's all-reduce completes.

A row on an axis of size one is not applied (``apply_plan`` skips it), so the
table answers ``Replicated`` there — a tensor-row model at ``ep=2`` alone is
whole everywhere.

Under ``context > 1`` every placement whose shape has a position axis is
wrapped in [`SequenceSharded`][] on that axis
([`position_axis`][] — the native dimension carrying the ``position``
kind, ``flat`` when it is the flattened ``(batch, position)`` pair of the
experts' token-major view; an attention pattern's *key* axis stays whole,
since the keys are gathered inside the function, §8.4) — **composed** with
the row's placement as its ``inner``, not replacing it. Under ``pipeline >
1`` every block-scoped placement is then wrapped in
[`StageLocal`][] with the block's owner — composed likewise —
and the model boundary is placed on the first stage (the embedding) and the
last (the final norm and the head). The order is fixed: the stage outermost,
the sequence chunk inside it, the row's placement innermost.

A row's ``style`` outside the closed set is refused by name
([`StyleError`][]): a new transformers style is a new row of this table,
never a fall-through to "replicated". The rows themselves are the registry's
([`PlanRow`][causalab.protocol.registry.plans.PlanRow], ``ModelInfo.parallel_plan``),
the same table ``apply_plan`` shards the model from — a row on an axis of
size one is applied by neither.
"""

from __future__ import annotations

import dataclasses
import math
import weakref
from typing import Any, Literal, Mapping, get_args

import torch

from causalab.neural.shared.parallel.standin import Shadowed
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    ExpertLocal,
    Axis,
    Interior,
    PartialSum,
    Placement,
    SequenceInner,
    SequenceSharded,
    Sharded,
    StageLocal,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.registry import (
    ATTENTION_FUNCTION_SLOTS,
    KV_REPLICATED,
    ParallelPlan,
    PlanRow,
    wildcard_layers,
)
from causalab.protocol.registry import EXPERTS_FUNCTION_SLOTS
from causalab.protocol.registry import STYLES as _REGISTRY_STYLES
from causalab.protocol.registry.shapes import FeatureShape

__all__ = [
    "INHERITED",
    "STYLES",
    "PlanRow",
    "Side",
    "Style",
    "StyleError",
    "TapPlacement",
    "child_rows",
    "experts_placement",
    "interior_placement",
    "module_path",
    "placement_for",
    "position_axis",
    "sequenced",
    "shard_axis",
    "site_placement",
    "slot_count",
    "stage_of",
]

#: The per-module parallel styles the placement table knows — transformers
#: 5.16's eight and the repository's ``kv_replicated`` (§6.6) — the
#: registry's closed set, spelled as a type here.
Style = Literal[
    "colwise",
    "packed_colwise",
    "rowwise",
    "colwise_gather_output",
    "replicated_with_grad_allreduce",
    "moe_tp_experts",
    "grouped_gemm",
    "ep_router",
    "kv_replicated",
]
STYLES: frozenset[str] = frozenset(get_args(Style))
assert STYLES == _REGISTRY_STYLES, "the placement table and the registry disagree"

#: The styles whose module computes with a contiguous block of heads per rank.
_HEAD_SHARDING: frozenset[str] = frozenset({"colwise", "packed_colwise"})
#: The attention-function slot that is the key — the tensor a ``kv_replicated``
#: row's output becomes (§6.6); every other slot is query space.
_KEY_SLOT = ATTENTION_FUNCTION_SLOTS["attention_key"]
#: The styles the routed experts carry under tensor parallelism.
_EXPERT_TENSOR: frozenset[str] = frozenset({"packed_colwise", "rowwise"})
#: The experts dispatch's slots, derived from the registry's component → slot
#: map as ``experts_interface.EXPERTS_SLOTS`` is, so the two cannot drift.
_EXPERTS_SLOTS: tuple[str, ...] = tuple(dict.fromkeys(EXPERTS_FUNCTION_SLOTS.values()))

Side = Literal["input", "output"]
SIDES: tuple[Side, ...] = get_args(Side)

#: The router's three outputs ``(router_logits, router_scores, router_indices)``.
ROUTER_LOGITS, ROUTER_SCORES, ROUTER_INDICES = 0, 1, 2

#: Components that carry a planned neighbour's tensor, and the neighbour whose
#: side decides their placement: ``(module name — a child or a sibling of the
#: tapped module, side)``. Two sit on a module no row names between two
#: planned projections; the key before RoPE is ``k_proj``'s output whether the
#: family taps it at the projection (the neighbour is the module itself) or
#: at a ``k_norm`` whose own row replicates the weight and not the tensor
#: (module docstring, §6.6).
INHERITED: Mapping[str, tuple[str, Side]] = {
    "mlp_activation": ("down_proj", "input"),
    "attention_probs": ("q_proj", "output"),
    "attention_key_pre_rope": ("k_proj", "output"),
}


class StyleError(ValueError):
    """A plan row names a style the table has no row for, a side that is not
    one of the two, a layer outside its tower, or a function interior the
    rows leave undefined — an internal invariant, as
    [`PlacementError`][causalab.neural.shared.parallel.fragments.PlacementError] is, never a document's fault."""

    def __init__(self, message: str, *, style: str | None = None) -> None:
        self.style = style
        super().__init__(message)


@dataclasses.dataclass(frozen=True)
class TapPlacement:
    """Where a tap's tensor lives, and whether the routing table it carries
    is this rank's remapped one (§6.3) — the two facts a hook body needs."""

    placement: Placement
    remapped_routing: bool = False


def shard_axis(shape: FeatureShape) -> int:
    """The native axis a tensor-parallel style shards at a tap of ``shape``:
    the head axis when the shape keeps one (``flat_inner`` off), else the
    last axis, where the inner axes are flattened head-major. The interiors
    read the head axis of their own shape instead
    ([`head_axis_index`][causalab.protocol.registry.shapes.FeatureShape.head_axis_index]), which
    names the packed dimension too — the same dimension as ``-1`` on a
    packed shape, spelled positively."""
    if shape.flat_inner:
        return -1
    for index, axis in enumerate(shape.axes):
        if axis.kind == "head":
            # a flat (batch, position) pair ahead of the head axis is one
            # native dimension, not two
            return index - (1 if shape.flat_batch else 0)
    return -1


def _check_style(style: str) -> None:
    if style not in STYLES:
        raise StyleError(
            f"parallel style {style!r} has no row in the placement table "
            f"(docs/model_parallelism.md §4); the styles are {sorted(STYLES)}",
            style=style,
        )


def _interior(
    row: PlanRow | None,
    side: Side,
    interior: bool,
    axis: int,
    tuple_index: int | None,
) -> TapPlacement:
    if row is None:
        return TapPlacement(REPLICATED)
    style = row.style
    _check_style(style)
    output = side == "output"
    sharded = Sharded(axis, row.axis)
    if style == "packed_colwise":
        # the fused [gate | up] projection: each rank holds its slice of BOTH
        # halves, so the output's feature axis is two contiguous runs (gate,
        # up), each sharded across the group — ``slots=2`` (the styles
        # contract's ``_PACKED_OUT``; a plain ``Sharded`` would gather the
        # ranks' gate slices ahead of every up slice)
        return TapPlacement(Sharded(axis, row.axis, slots=2) if output else REPLICATED)
    if style in _HEAD_SHARDING:
        return TapPlacement(sharded if output else REPLICATED)
    if style == KV_REPLICATED:
        # the weight is whole on every rank (its input replicated); the output
        # is the one KV head this rank's query heads read, held by ``repeat``
        # consecutive ranks — a repeated shard whose whole is the KV heads
        repeated = Sharded(axis, row.axis, repeat=row.repeat)
        return TapPlacement(repeated if output else REPLICATED)
    if style == "rowwise":
        return TapPlacement(REPLICATED if output else sharded)
    if style == "replicated_with_grad_allreduce":
        # the norm between a colwise and a rowwise layer: its weight is
        # replicated, the per-head activation it normalizes is the colwise
        # output's — this rank's heads, on the shape's shard axis
        return TapPlacement(sharded)
    if style in ("grouped_gemm", "moe_tp_experts"):
        if interior and output:
            if row.axis != "expert":
                raise ProtocolError(
                    "P4",
                    f"--parallel.{row.axis}: the routed-experts interior under "
                    f"tensor-parallel experts ({style!r} on the {row.axis} axis) is "
                    "not the module row's to place — it is derived from the "
                    "expert parameters' rows (experts_placement, "
                    "docs/model_parallelism.md §6.3)",
                )
            return TapPlacement(ExpertLocal(row.axis), remapped_routing=True)
        return TapPlacement(REPLICATED)
    if style == "ep_router":
        if not output:
            return TapPlacement(REPLICATED)
        if tuple_index == ROUTER_SCORES:
            # the scores: zeroed on the slots this rank's experts do not own
            return TapPlacement(ExpertLocal(row.axis))
        if tuple_index in (None, ROUTER_INDICES):
            # the routing table — the row's reason to exist — is what a tap
            # with no position named means
            return TapPlacement(REPLICATED, remapped_routing=True)
        return TapPlacement(REPLICATED)
    # colwise_gather_output
    return TapPlacement(REPLICATED)


def placement_for(
    row: PlanRow | None,
    side: Side,
    *,
    interior: bool = False,
    stage: int | None = None,
    axis: int = -1,
    tuple_index: int | None = None,
) -> TapPlacement:
    """The table's answer for ``row`` on ``side`` of the module.

    ``interior`` asks for the function interior the experts interface taps
    rather than the module boundary (the experts distinction above);
    ``stage`` wraps the answer in ``StageLocal(stage)`` for a block-scoped
    module under pipeline placement; ``axis`` is the native axis a
    tensor-parallel shard lies on ([`shard_axis`][]); ``tuple_index`` is
    which element of a tuple payload the tap reads (the router's three).

    Raises:
        StyleError: ``row.style`` is not in [`STYLES`][], or ``side`` is not
            ``"input"`` / ``"output"``.
        ProtocolError: ``P4`` — the experts interior asked of a
            tensor-parallel experts row, which the module row does not place
            ([`experts_placement`][] does, from the parameter rows).
    """
    if side not in SIDES:
        raise StyleError(f"a tap has an input side or an output side, not {side!r}")
    placed = _interior(row, side, interior, axis, tuple_index)
    return placed if stage is None else _staged(placed, stage)


def _staged(placed: TapPlacement, stage: int) -> TapPlacement:
    inner: Interior = placed.placement  # type: ignore[assignment]  # never a StageLocal
    return dataclasses.replace(placed, placement=StageLocal(stage, inner=inner))


def position_axis(shape: FeatureShape) -> tuple[int, bool] | None:
    """The native dimension of ``shape`` carrying its ``position`` axis —
    the axis a sequence chunk lies on (§8.4) — and whether that dimension
    is the flattened ``(batch, position)`` pair (``flat_batch``: the
    experts' token-major view, chunked by position once unfolded). ``None``
    for a shape with no position axis. An attention pattern's second,
    ``key_position`` axis is not it: the keys are whole on every rank."""
    for index, group in enumerate(shape.native_groups):
        kinds = [axis.kind for axis in group]
        if "position" in kinds:
            return index, len(group) == 2 and kinds[0] == "batch"
    return None


def sequenced(placed: TapPlacement, shape: FeatureShape) -> TapPlacement:
    """``placed`` wrapped in ``SequenceSharded`` on the shape's position
    axis ([`position_axis`][]), the row's placement as its inner — the
    ``context > 1`` composition. A shape with no position axis is returned
    as it is: every rank computes it whole.

    Raises:
        StyleError: the placement is already a stage or sequence wrapper.
    """
    if isinstance(placed.placement, (StageLocal, SequenceSharded)):
        raise StyleError(
            f"the sequence chunk wraps a row's placement, not {placed.placement}"
        )
    where = position_axis(shape)
    if where is None:
        return placed
    axis, flat = where
    inner: SequenceInner = placed.placement
    return dataclasses.replace(
        placed, placement=SequenceSharded(axis, "context", inner=inner, flat=flat)
    )


def stage_of(layer: int, *, num_layers: int, stages: int) -> int:
    """The pipeline stage owning ``layer``: ``stages`` contiguous even ranges
    of the tower in order, the remainder to the last stage — the same rule
    [`parse`][causalab.neural.shared.devices.DeviceMap.parse] places blocks by.

    Raises:
        StyleError: more stages than layers, or a layer outside the tower.
    """
    if stages < 1 or stages > num_layers:
        raise StyleError(
            f"{stages} pipeline stages over {num_layers} layers: every stage owns "
            "at least one layer"
        )
    if not 0 <= layer < num_layers:
        raise StyleError(f"layer {layer} is outside a tower of {num_layers}")
    return min(layer // (num_layers // stages), stages - 1)


# --------------------------------------------------------------------------- #
# the function interiors: derived from the rows of the module's children
# --------------------------------------------------------------------------- #


def child_rows(
    plan: ParallelPlan, path: str, *, prefix: str | None = None
) -> dict[str, PlanRow]:
    """The plan rows under module ``path`` — its children's and their
    parameters' — by pattern. The path is wildcarded the way ``style_for``
    wildcards it and looked up as is and with the base-model prefix stripped
    (``prefix`` when given, else the first dotted component), so a plan
    spelled without the prefix (every transformers config) matches a module
    path spelled with it."""
    generic = wildcard_layers(path)
    heads = [generic]
    if prefix is not None:
        if generic.startswith(prefix + "."):
            heads.append(generic[len(prefix) + 1 :])
    else:
        _, dot, rest = generic.partition(".")
        if dot:
            heads.append(rest)
    return {
        pattern: row
        for pattern, row in plan.rows.items()
        if any(pattern.startswith(head + ".") for head in heads)
    }


def _active_rows(
    plan: ParallelPlan, path: str, prefix: str | None, active: frozenset[Axis] | None
) -> dict[str, PlanRow]:
    rows = child_rows(plan, path, prefix=prefix)
    for row in rows.values():
        _check_style(row.style)
    if active is None:
        return rows
    return {pattern: row for pattern, row in rows.items() if row.axis in active}


def interior_placement(
    plan: ParallelPlan,
    path: str,
    *,
    head_axis: int | None,
    prefix: str | None = None,
    active: frozenset[Axis] | None = None,
    slot: str | None = None,
) -> TapPlacement:
    """The placement of a tensor computed *inside* the mixer at ``path``
    (§6.2, §6.4): ``Sharded(head_axis, axis)`` when a child of the mixer
    carries a head-sharding row (``colwise`` / ``packed_colwise``) on an axis
    in ``active`` (every axis when ``None``), else ``Replicated`` — a mixer
    whose projections are all gathered computes whole tensors. The ``key``
    slot (``slot``, the site's function slot) under a ``kv_replicated`` K/V
    row is that row's output — a repeated shard on the slot's head axis
    (§6.6); every other slot is query space and follows the head-sharding
    rows.

    Raises:
        StyleError: a child row's style is outside [`STYLES`][]; the
            children shard over two axes; the K/V rows disagree on their
            repeat; or the mixer is head-sharded and the slot's shape has no
            head axis to shard along.
    """
    rows = _active_rows(plan, path, prefix, active)
    axes = {row.axis for row in rows.values() if row.style in _HEAD_SHARDING}
    if not axes:
        return TapPlacement(REPLICATED)
    if len(axes) > 1:
        raise StyleError(
            f"the projections under {path!r} are sharded over two axes "
            f"{sorted(axes)}; a function interior has one head-sharding group"
        )
    if head_axis is None:
        raise StyleError(
            f"the interior of {path!r} computes with local heads, but the slot's "
            "shape has no head axis to shard along (docs/model_parallelism.md §6.2)"
        )
    axis = next(iter(axes))
    repeats = {
        (row.axis, row.repeat) for row in rows.values() if row.style == KV_REPLICATED
    }
    if slot == _KEY_SLOT and repeats:
        if len(repeats) > 1:
            raise StyleError(
                f"the replicated K/V rows under {path!r} disagree on their repeat "
                f"{sorted(repeats)}; one tensor group holds the KV heads"
            )
        ((kv_axis, repeat),) = repeats
        return TapPlacement(Sharded(head_axis, kv_axis, repeat=repeat))
    return TapPlacement(Sharded(head_axis, axis))


def experts_placement(
    plan: ParallelPlan,
    path: str,
    *,
    slot: str | None,
    slots: int,
    prefix: str | None = None,
    active: frozenset[Axis] | None = None,
) -> TapPlacement:
    """The placement of the experts dispatch's token-major view at ``slot``
    inside the experts module at ``path`` (§6.3), from the rows of the
    module's parameters: ``grouped_gemm`` → ``ExpertLocal(axis)`` with the
    remapped table; ``packed_colwise`` / ``rowwise`` → the ``gate_up``,
    ``activation`` and ``neuron_output`` views (the last the down-projection's
    input, ``act(gate) * up``, neuron-major like the activation)
    ``Sharded(-1, axis, slots=slots)`` (``slots`` is the number of
    (token, slot) runs in the view's feature axis, [`slot_count`][]) and
    the ``down`` view ``PartialSum(axis)``; no active row → ``Replicated``.

    Raises:
        StyleError: rows of mixed styles or on two axes under the module, a
            style outside [`STYLES`][], or a slot outside the dispatch's four.
    """
    if slot not in _EXPERTS_SLOTS:
        raise StyleError(
            f"unknown experts slot {slot!r}; the dispatch's slots are {_EXPERTS_SLOTS}"
        )
    rows = _active_rows(plan, path, prefix, active)
    if not rows:
        return TapPlacement(REPLICATED)
    axes = {row.axis for row in rows.values()}
    styles = {row.style for row in rows.values()}
    if len(axes) > 1:
        raise StyleError(
            f"the expert parameters under {path!r} are sharded over two axes "
            f"{sorted(axes)}; the routed experts are expert- or tensor-parallel"
        )
    axis = next(iter(axes))
    if styles == {"grouped_gemm"}:
        return TapPlacement(ExpertLocal(axis), remapped_routing=True)
    if styles <= _EXPERT_TENSOR:
        if slot == "down":
            return TapPlacement(PartialSum(axis))
        return TapPlacement(Sharded(-1, axis, slots=slots))
    raise StyleError(
        f"the expert parameters under {path!r} carry rows of mixed styles "
        f"{sorted(styles)}; the table knows grouped_gemm (expert parallelism) "
        "and packed_colwise + rowwise (tensor parallelism) of the experts"
    )


def slot_count(shape: FeatureShape) -> int:
    """How many equal runs the shape's packed feature axis is made of: the
    product of its inner axes before the feature axis — ``top_k`` for the
    experts' ``(tokens, top_k · d)`` view, ``top_k · 2`` for the fused
    ``[gate | up]`` view, one for a plain feature vector."""
    return math.prod(
        axis.width or 1 for axis in shape.inner_axes if axis.kind != "feature"
    )


# --------------------------------------------------------------------------- #
# a site's placement, from the plan and the geometry
# --------------------------------------------------------------------------- #

_PATHS: "weakref.WeakKeyDictionary[torch.nn.Module, dict[int, str]]" = (
    weakref.WeakKeyDictionary()
)


def module_path(model: torch.nn.Module, module: Any) -> str:
    """The dotted path of ``module`` under ``model`` — the key a plan row is
    looked up by. The index over ``named_modules`` is built once per model,
    and covers the tree a pipeline stand-in shadows (§6.5): a module another
    stage holds is looked up at the path it has on that stage.

    Raises:
        StyleError: ``module`` is not in ``model``'s tree.
    """
    paths = _PATHS.get(model)
    if paths is None:
        paths = {}
        for name, child in model.named_modules():
            paths[id(child)] = name
            if isinstance(child, Shadowed):
                for sub, shadowed in child.shadow.named_modules(prefix=name):
                    paths.setdefault(id(shadowed), sub)
        _PATHS[model] = paths
    path = paths.get(id(module))
    if path is None:
        raise StyleError(
            f"module {type(module).__name__} is not in the model's tree, so no "
            "plan row can name it"
        )
    return path


#: The function-interior tap kinds whose module is the mixer the function
#: runs in: the attention function's slots and the DeltaNet kernel boundary.
_MIXER_INTERIOR: frozenset[str] = frozenset({"interface", "delta"})


def _active_row(
    plan: ParallelPlan, geometry: ParallelGeometry, path: str, prefix: str | None
) -> PlanRow | None:
    """The plan row for ``path`` when its axis is above one — the rows
    ``apply_plan`` applied, and no other."""
    row = plan.style_for(path, prefix=prefix)
    if row is None or getattr(geometry, row.axis) == 1:
        return None
    return row


def _inherited(
    plan: ParallelPlan,
    geometry: ParallelGeometry,
    path: str,
    prefix: str | None,
    component: str,
) -> tuple[PlanRow | None, Side] | None:
    """The planned neighbour an [`INHERITED`][] component reads its
    placement from: the named module as a child of the tapped module, else
    as its sibling. ``None`` for every other component."""
    entry = INHERITED.get(component)
    if entry is None:
        return None
    name, side = entry
    parent = path.rpartition(".")[0]
    for candidate in (f"{path}.{name}", f"{parent}.{name}" if parent else name):
        row = _active_row(plan, geometry, candidate, prefix)
        if row is not None:
            return row, side
    return None, side


def site_placement(
    *,
    plan: ParallelPlan,
    geometry: ParallelGeometry,
    path: str,
    prefix: str | None,
    kind: str,
    component: str,
    layer: int,
    num_layers: int,
    layerless: bool,
    shape: FeatureShape,
    slot: str | None = None,
    tuple_index: int | None = None,
) -> TapPlacement:
    """The placement of one resolved site: the experts interior from the
    experts module's parameter rows ([`experts_placement`][]), a mixer
    interior — the attention slots, the mixer-returned pattern, the DeltaNet
    kernel boundary — from the mixer's projection rows
    ([`interior_placement`][]), a module boundary from its own row on the
    side its hook kind names — looked up with the model's base-model
    ``prefix`` stripped, the way ``apply_plan`` matches, and only on an axis
    the geometry has above one — or, for an [`INHERITED`][] component, its
    planned neighbour's; the shard axis, the head axis and the slot count
    read off the tap's ``shape``, the router's output by ``tuple_index``,
    the routing flag kept only where the tap *is* a routing table (an
    integral tensor, or the experts interior that rides one), the sequence
    wrapper on the shape's position axis under ``context > 1``
    ([`sequenced`][]), and the stage wrapper under ``pipeline > 1``.

    ``plan`` is the registry's ``ModelInfo.parallel_plan`` and ``geometry``
    the bundle's; ``slot`` the site's function slot (``interface_slot``).
    """
    active: frozenset[Axis] = frozenset(
        axis for axis in ("tensor", "expert") if getattr(geometry, axis) > 1
    )
    if kind == "experts":
        placed = experts_placement(
            plan, path, slot=slot, slots=slot_count(shape), prefix=prefix, active=active
        )
    elif kind in _MIXER_INTERIOR or slot is not None:
        placed = interior_placement(
            plan,
            path,
            head_axis=shape.head_axis_index,
            prefix=prefix,
            active=active,
            slot=slot,
        )
    elif kind == "interior":
        # a fused-forward interior the nnsight engine serves; this engine
        # refuses it by name before any placement matters
        placed = TapPlacement(REPLICATED)
    else:
        side: Side = "input" if kind == "in" else "output"
        inherited = _inherited(plan, geometry, path, prefix, component)
        if inherited is not None:
            row, side = inherited
        else:
            row = _active_row(plan, geometry, path, prefix)
        placed = placement_for(
            row, side, axis=shard_axis(shape), tuple_index=tuple_index
        )
    if placed.remapped_routing and not (shape.integral or kind == "experts"):
        placed = dataclasses.replace(placed, remapped_routing=False)
    if geometry.context > 1:
        placed = sequenced(placed, shape)
    stages = geometry.pipeline
    if stages == 1:
        return placed
    stage: int | None
    if layerless:
        if component in ("ln_final", "lm_head"):
            stage = stages - 1
        else:
            # the embedding's output, and its input — the ids: every rank
            # encodes them, but the embedding runs on the first stage alone,
            # and a tap is where its hook fires
            stage = 0
    else:
        stage = stage_of(layer, num_layers=num_layers, stages=stages)
    if stage is None:
        return placed
    return _staged(placed, stage)
