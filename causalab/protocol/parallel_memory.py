"""What a geometry places on each rank's card, before any weight is read
(``docs/model_parallelism.md`` §2 "check", §3, §5.3, §11): the per-parameter
placement table, the per-rank resident weight bytes it yields, the measured
headroom rule that turns them into a card footprint, the search for the
geometries that would fit, and the refusal that names all of it.

Torch-free protocol arithmetic over facts the registry and the checkpoint
headers provide, beside [`causalab.protocol.parallel`][]. The torch half —
reading the device's memory and raising — is
``neural/shared/parallel/memory.py``; ``dry-run`` prints the estimate off
the cached headers with no device at all.

**The placement table.** One [`Placement`][] per model parameter: its
element count, the plan axis it is sharded over (``None`` for a replicated
parameter — a norm, the embedding, the head, a router), its home under a
pipeline (a layer's index, the first stage, the last, or every stage) and,
for the K/V projections, the group size above which the row replicates the
weight whole (§6.6). [`placement_table`][] reads it off the checkpoint
headers, the key → parameter targets, the family's plan and its tree, so
the loader's own read plan and the torch-free ``dry-run`` build the same
table from the same facts.

**The estimate.** [`estimate_resident`][] is the pure sharding
arithmetic: a replicated parameter is counted whole on every rank that
holds it, a sharded one at ``1 / size`` of its axis, a pipeline stage
holds its own layers and the first / last stage the embedding / the head,
so over the stages of one model group the sum is the whole model.

**The rule** ([`RULE`][], an [`EstimateRule`][]) is empirically
calibrated on the standard workflow run on ``Qwen/Qwen3.6-35B-A3B``
(world 1 and ``tp``). The fractions are headroom defaults, not
guarantees:

* ``copies``: **1.0** on every axis — each sharded parameter is held once
  on its rank (§5.3 "One copy").
* ``base``: **15 %** of the whole model's bytes above the resident weights
  for activations, captures, the CUDA context, and allocator pools.
* ``context``: **+10 %** under ``cp > 1`` for gathered keys and values and
  receive buffers.

**The load's peak under a dtype conversion** (§5.3). A checkpoint stored
in one dtype and held in another (``google/gemma-2-9b``: fp32 on disk,
bf16 in the document) is cast tensor by tensor; the loader stages the
converting tensors on the **host** in batches no larger than the largest
converting tensor and copies each to the device already in the target dtype
(``weights.Prefetch``), so the card's peak is the resident weights and
nothing else — [`EstimateRule.footprint`][] carries no conversion term,
and a same-dtype checkpoint's footprint is unchanged by the rule. What the
conversion costs is host memory: [`conversion`][] names it per rank —
the largest converting tensor the rank reads, whole, in its on-disk bytes
(a bound: under ``tp`` the piece is smaller) — and ``dry-run`` prints it as
one line.

The footprint is compared against available memory, including reusable
allocator cache, to warn about limited headroom. Only resident weights can
cause a hard admission refusal. The footprint is an
estimate calibrated where the whole model resides on each card (world 1
and ``tp``), not a guarantee that an accepted run will fit. It can
underestimate sharded runs under ``ep``. The fixed ``base`` fraction does
not shrink with the shard, and workloads outside the calibration may need
different headroom. The CUDA context is included in ``base`` but already
excluded from device-free memory, so this part of the estimate is conservative.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Literal, Mapping, Sequence

from causalab.protocol.checkpoint_census import ITEMSIZES
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    MeshLayout,
    ParallelGeometry,
    check,
    format_geometry,
    stage_layers,
)
from causalab.protocol.registry import (
    KV_PROJECTIONS,
    KV_REPLICATED,
    PLAN_AXES,
    ModelInfo,
    ParallelPlan,
    PlanAxis,
    PlanRow,
    TreeAddress,
)
from causalab.protocol.schema import PRECISION_DTYPES

__all__ = [
    "DISK_WORDS",
    "DTYPE_ITEMSIZES",
    "REPLICATED_STYLES",
    "RULE",
    "SINGLE_COPY",
    "Conversion",
    "EstimateRule",
    "Fit",
    "Home",
    "MemoryEstimate",
    "Placement",
    "PlacementTable",
    "conversion",
    "estimate_resident",
    "fitting_geometries",
    "format_bytes",
    "memory_check",
    "memory_estimate",
    "memory_refusal",
    "placement_table",
    "row_for",
    "whole_bytes",
]

#: Bytes per element of a document's ``model.dtype`` word (§2.1's closed set).
DTYPE_ITEMSIZES: Mapping[str, int] = {"fp32": 4, "bf16": 2, "fp16": 2}
assert set(DTYPE_ITEMSIZES) == set(PRECISION_DTYPES)

#: The precision word of each **float** dtype the safetensors format spells
#: (its ``dtype`` field → the document's word; ``fp64`` for the one no
#: document can ask for). A checkpoint tensor stored in one of these and
#: held as another word is a conversion (module docstring); a tensor outside
#: the table — an integer buffer, a fp8 block — keeps its own dtype and never
#: converts, which is transformers' rule too.
DISK_WORDS: Mapping[str, str] = {
    "F16": "fp16",
    "BF16": "bf16",
    "F32": "fp32",
    "F64": "fp64",
}
assert set(DISK_WORDS) <= set(ITEMSIZES)

#: Served plan styles whose parameter is held **whole** on every rank of its
#: group: the norms transformers marks ``replicated_with_grad_allreduce``,
#: the K/V projection under [`KV_REPLICATED`][causalab.protocol.registry.plans.KV_REPLICATED] (§6.6), and the router
#: under ``ep_router``, which routes every token on every rank and keeps
#: its weight. Every other served style shards the parameter ``1 / size``.
REPLICATED_STYLES: frozenset[str] = frozenset(
    {"replicated_with_grad_allreduce", KV_REPLICATED, "ep_router"}
)

#: Where a parameter lives under a pipeline: in one decoder layer, on the
#: first stage (the embedding), on the last (the final norm, the head), or
#: on every stage (a parameter the tree does not place).
Home = Literal["layer", "first", "last", "every"]

#: One copy of everything — the sharding arithmetic alone.
SINGLE_COPY: Mapping[PlanAxis, float] = {"tensor": 1.0, "expert": 1.0}

_GIB = float(1 << 30)


@dataclasses.dataclass(frozen=True)
class Placement:
    """One model parameter's place in a geometry (module docstring).

    Raises:
        ValueError: a negative or non-integer element count, an axis outside
            the plan axes, a ``layer`` home without an index or an index on
            another home, a ``shard_limit`` below one, a ``disk_dtype`` the
            format does not spell, a ``key_elements`` outside
            ``[0, elements]``.
    """

    elements: int
    axis: PlanAxis | None = None
    home: Home = "every"
    layer: int | None = None
    #: The largest group size the parameter is sharded over; above it the
    #: row holds the weight whole on every rank (the K/V projections at
    #: ``tp > num_kv_heads``, §6.6). ``None``: sharded at any size.
    shard_limit: int | None = None
    #: The safetensors dtype the parameter's checkpoint tensors are stored
    #: in (``F32``) — the widest of them when a fused parameter's tensors
    #: differ; ``None`` when the table was built without the headers' dtypes
    #: and nothing is known about a conversion.
    disk_dtype: str | None = None
    #: Elements of the largest single checkpoint tensor landing on the
    #: parameter: the parameter itself for a one-tensor parameter, one
    #: expert's tensor for a fused one; ``None`` reads as ``elements``.
    key_elements: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.elements, bool) or not isinstance(self.elements, int):
            raise ValueError(f"elements must be an int, got {self.elements!r}")
        if self.elements < 0:
            raise ValueError(f"elements must be non-negative, got {self.elements}")
        if self.disk_dtype is not None and self.disk_dtype not in ITEMSIZES:
            raise ValueError(
                f"disk_dtype {self.disk_dtype!r} is not one the safetensors format "
                f"spells ({', '.join(ITEMSIZES)})"
            )
        if self.key_elements is not None and (
            isinstance(self.key_elements, bool)
            or not 0 <= self.key_elements <= self.elements
        ):
            raise ValueError(
                f"key_elements must be an int in [0, {self.elements}], got "
                f"{self.key_elements!r}"
            )
        if self.axis is not None and self.axis not in PLAN_AXES:
            raise ValueError(f"axis {self.axis!r} is not one of {list(PLAN_AXES)}")
        if (self.home == "layer") != (self.layer is not None):
            raise ValueError(
                f"home {self.home!r} with layer {self.layer!r}: a layer home names "
                "its index and no other home does"
            )
        if self.layer is not None and (isinstance(self.layer, bool) or self.layer < 0):
            raise ValueError(f"layer must be a non-negative int, got {self.layer!r}")
        if self.shard_limit is not None and self.shard_limit < 1:
            raise ValueError(f"shard_limit must be at least 1, got {self.shard_limit}")

    def shards(self, geometry: ParallelGeometry) -> int:
        """How many ways ``geometry`` splits this parameter: its axis's size,
        or 1 — replicated, an axis at one, or a size above ``shard_limit``."""
        if self.axis is None:
            return 1
        size = getattr(geometry, self.axis)
        if self.shard_limit is not None and size > self.shard_limit:
            return 1
        return size

    def converts_to(self, dtype: str) -> bool:
        """Whether a load holding the parameter as ``dtype`` casts it: its
        tensors are stored in a float dtype ([`DISK_WORDS`][]) whose word
        is not ``dtype``. ``False`` without a known ``disk_dtype``."""
        word = DISK_WORDS.get(self.disk_dtype) if self.disk_dtype else None
        return word is not None and word != dtype

    @property
    def largest_key_bytes(self) -> int:
        """On-disk bytes of the largest single checkpoint tensor landing
        here (``key_elements`` at the disk dtype's item size), the unit a
        converting load stages; ``0`` without a known ``disk_dtype``."""
        if self.disk_dtype is None:
            return 0
        count = self.elements if self.key_elements is None else self.key_elements
        return count * ITEMSIZES[self.disk_dtype]


PlacementTable = Mapping[str, Placement]


def _first_rank_of_stage(geometry: ParallelGeometry, stage: int) -> int:
    """The lowest rank whose pipeline coordinate is ``stage``."""
    return stage * geometry.context * geometry.model


def whole_bytes(table: PlacementTable, itemsize: int) -> int:
    """The whole model's bytes at ``itemsize`` per element."""
    return sum(p.elements for p in table.values()) * itemsize


def _num_layers(table: PlacementTable) -> int:
    layers = [p.layer for p in table.values() if p.layer is not None]
    return 1 + max(layers) if layers else 0


def _local_bytes(elements: int, itemsize: int, shards: int) -> int:
    """This rank's share of a parameter split ``shards`` ways — the last
    chunk may be shorter, so the share is rounded up (a bound per rank)."""
    return -(-(elements * itemsize) // shards)


def estimate_resident(
    table: PlacementTable,
    geometry: ParallelGeometry,
    itemsize: int,
    *,
    copies: Mapping[PlanAxis, float] = SINGLE_COPY,
    num_layers: int | None = None,
) -> tuple[int, ...]:
    """Per rank of ``geometry``, the bytes of model weights shard-on-read
    places on it (§5.3): a replicated parameter whole on every rank that
    holds it, a sharded one at ``1 / shards`` of its axis, a pipeline stage
    its own layers plus the embedding on the first and the head on the
    last. ``copies`` scales a sharded parameter's bytes per axis — the
    loader's known factors ([`RULE`][]); the default counts one copy, the
    pure sharding arithmetic, under which the stages of a model group sum to
    the whole model and every rank of one stage answers alike.
    ``num_layers`` is the model's block count, which decides the stage split
    (``stage_layers``); left ``None`` it is read off the table — right for a
    whole model's table. Supply it explicitly when passing a partial table.

    Raises:
        ProtocolError: ``P4`` — ``pipeline`` exceeds the layers
            (``stage_layers``'s refusal), or ``itemsize`` is not positive.
    """
    if isinstance(itemsize, bool) or not isinstance(itemsize, int) or itemsize < 1:
        raise ProtocolError(
            "P4", f"itemsize must be a positive int, got {itemsize!r}", path="--dtype"
        )
    per_stage: dict[int, int] = {}
    for stage, held in _stages(table, geometry, num_layers).items():
        total = 0
        for placement in held:
            shards = placement.shards(geometry)
            local = _local_bytes(placement.elements, itemsize, shards)
            if placement.axis is not None and shards > 1:
                local = round(local * copies.get(placement.axis, 1.0))
            total += local
        per_stage[stage] = total
    return _per_rank(per_stage, geometry)


def _stages(
    table: PlacementTable, geometry: ParallelGeometry, num_layers: int | None
) -> dict[int, list[Placement]]:
    """Per pipeline stage the placements it holds: a layer's when the stage
    holds the layer (``stage_layers``), the first stage's ``first`` homes,
    the last stage's ``last``, and every stage's ``every``. ``num_layers``
    ``None`` reads the tower's height off the table."""
    if num_layers is None:
        num_layers = _num_layers(table)
    stages: dict[int, list[Placement]] = {}
    for stage in range(geometry.pipeline):
        held = (
            stage_layers(geometry, _first_rank_of_stage(geometry, stage), num_layers)
            if num_layers
            else range(0)
        )
        members: list[Placement] = []
        for placement in table.values():
            if placement.home == "layer":
                if placement.layer not in held:
                    continue
            elif placement.home == "first":
                if stage != 0:
                    continue
            elif placement.home == "last":
                if stage != geometry.pipeline - 1:
                    continue
            members.append(placement)
        stages[stage] = members
    return stages


def _per_rank(
    per_stage: Mapping[int, int], geometry: ParallelGeometry
) -> tuple[int, ...]:
    """A per-stage figure spread over the ranks of the geometry."""
    layout = MeshLayout(geometry)
    return tuple(
        per_stage[layout.rank_in(rank, "pipeline")] for rank in range(geometry.world)
    )


@dataclasses.dataclass(frozen=True)
class Conversion:
    """A load that converts the checkpoint's dtype (module docstring, "the
    load's peak under a dtype conversion"): the on-disk words it casts
    from, the document's word it casts to, and per rank the host bytes the
    loader stages at the load's peak — the largest converting tensor the
    rank reads, whole, in its on-disk bytes."""

    from_words: tuple[str, ...]
    to: str
    staging: tuple[int, ...]

    def describe(self) -> str:
        """The ``dry-run`` line: ``read as fp32 on disk, cast to bf16 on the
        host: rank0 +3.42 GiB, rank1 +3.42 GiB of host memory at the load's
        peak (the largest converting tensor, whole); nothing on the card``."""
        per_rank = ", ".join(
            f"rank{rank} +{format_bytes(count)}"
            for rank, count in enumerate(self.staging)
        )
        return (
            f"read as {'/'.join(self.from_words)} on disk, cast to {self.to} on the "
            f"host: {per_rank} of host memory at the load's peak (the largest "
            "converting tensor, whole); nothing on the card"
        )


def conversion(
    table: PlacementTable,
    geometry: ParallelGeometry,
    dtype: str,
    *,
    num_layers: int | None = None,
) -> Conversion | None:
    """What a load of ``table`` held as ``dtype`` converts (module
    docstring): ``None`` when no parameter's stored dtype differs from
    ``dtype`` — every tensor lands on the device as read — else the
    [`Conversion`][] naming the on-disk words cast and, per rank, the
    on-disk bytes of the largest converting tensor the rank reads, which is
    what the loader stages on the host at the load's peak. A parameter
    whose ``disk_dtype`` is unknown never converts here: the table was
    built without the headers' dtypes and the estimate says nothing.

    Raises:
        ProtocolError: ``P4`` — ``dtype`` is not one of the document's
            precision words.
    """
    if dtype not in DTYPE_ITEMSIZES:
        raise ProtocolError(
            "P4",
            f"dtype {dtype!r} is not one of {sorted(DTYPE_ITEMSIZES)}",
            path="model.dtype",
        )
    converting = [p for p in table.values() if p.converts_to(dtype)]
    if not converting:
        return None
    words = tuple(
        sorted({DISK_WORDS[p.disk_dtype] for p in converting if p.disk_dtype})
    )
    per_stage = {
        stage: max(
            (p.largest_key_bytes for p in held if p.converts_to(dtype)), default=0
        )
        for stage, held in _stages(table, geometry, num_layers).items()
    }
    return Conversion(
        from_words=words, to=dtype, staging=_per_rank(per_stage, geometry)
    )


# --------------------------------------------------------------------------- #
# the placement table off the checkpoint's facts
# --------------------------------------------------------------------------- #


def row_for(plan: ParallelPlan, name: str, prefix: str) -> PlanRow | None:
    """The plan row a parameter path falls under: the path itself, then each
    ancestor module — a row names a module (``…self_attn.q_proj``), a fused
    parameter (``…experts.gate_up_proj``), or the experts module above a
    per-expert checkpoint tensor (``…experts.5.gate_proj.weight``). The
    loader's read plan reads the style off the same row."""
    path = name
    while path:
        row = plan.style_for(path, prefix=prefix)
        if row is not None:
            return row
        path, _, _ = path.rpartition(".")
    return None


def _home(name: str, tree: TreeAddress) -> tuple[Home, int | None]:
    blocks = tree.blocks + "."
    if name.startswith(blocks):
        index = name[len(blocks) :].split(".", 1)[0]
        if index.isdigit():
            return "layer", int(index)
    homes: tuple[tuple[str, Home], ...] = (
        (tree.embedding, "first"),
        (tree.final_norm, "last"),
        (tree.lm_head, "last"),
    )
    for address, home in homes:
        if name == address or name.startswith(address + "."):
            return home, None
    return "every", None


def placement_table(
    elements: Mapping[str, int],
    targets: Mapping[str, str],
    plan: ParallelPlan,
    tree: TreeAddress,
    info: ModelInfo,
    *,
    dtypes: Mapping[str, str] | None = None,
) -> dict[str, Placement]:
    """The [`Placement`][] of every model parameter the checkpoint keys
    in ``targets`` land on: ``elements`` is each key's element count (off
    its header), ``targets`` the key → parameter path map (the loader's
    ``ReadPlan.targets``, or ``checkpoint_census.checkpoint_targets``), and
    several keys landing on one parameter (per-expert tensors a converter
    fuses) add up — the largest of them is the placement's ``key_elements``.
    ``dtypes``, each key's safetensors dtype off its header, sets
    ``disk_dtype`` (the widest when a parameter's tensors differ) so
    [`conversion`][] can say what a load casts; without it nothing is
    known. The axis is the plan row the parameter falls under — the
    *family's* plan, not one rewritten for a geometry: the K/V projections'
    ``shard_limit`` is ``info.num_kv_heads`` (§6.6), so one table serves
    every geometry the fit search tries. A style in
    [`REPLICATED_STYLES`][] is replicated; the home is the tree's.

    Raises:
        ProtocolError: ``P2`` — a dtype in ``dtypes`` the format does not
            spell.
    """
    prefix = tree.blocks.split(".", 1)[0]
    counts: dict[str, int] = {}
    largest: dict[str, int] = {}
    words: dict[str, str] = {}
    for key, parameter in targets.items():
        count = int(elements[key])
        counts[parameter] = counts.get(parameter, 0) + count
        largest[parameter] = max(largest.get(parameter, 0), count)
        if dtypes is not None:
            word = dtypes[key]
            if word not in ITEMSIZES:
                raise ProtocolError(
                    "P2",
                    f"{key}: safetensors dtype {word!r} is not one the format spells "
                    f"({', '.join(ITEMSIZES)})",
                )
            held = words.get(parameter)
            if held is None or ITEMSIZES[word] > ITEMSIZES[held]:
                words[parameter] = word
    table: dict[str, Placement] = {}
    for parameter, count in counts.items():
        row = row_for(plan, parameter, prefix)
        axis: PlanAxis | None = None
        limit: int | None = None
        if row is not None and row.served and row.style not in REPLICATED_STYLES:
            axis = row.axis
        if row is not None and row.axis == "tensor":
            leaf = _module_leaf(parameter)
            if row.style == KV_REPLICATED or leaf in KV_PROJECTIONS:
                axis, limit = "tensor", info.num_kv_heads
        home, layer = _home(parameter, tree)
        table[parameter] = Placement(
            elements=count,
            axis=axis,
            home=home,
            layer=layer,
            shard_limit=limit,
            disk_dtype=words.get(parameter),
            key_elements=largest[parameter],
        )
    return table


_LEAF_PARAMETERS = frozenset({"weight", "bias"})


def _module_leaf(parameter: str) -> str:
    """The module leaf of a parameter path (``k_proj`` of
    ``model.layers.0.self_attn.k_proj.weight``)."""
    parts = parameter.split(".")
    if parts[-1] in _LEAF_PARAMETERS and len(parts) > 1:
        parts = parts[:-1]
    return parts[-1]


# --------------------------------------------------------------------------- #
# the rule, the footprint, the fits
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class EstimateRule:
    """The measured headroom rule (module docstring): the loader's copies
    per plan axis, the base fraction of the whole model's bytes, and the
    extra fraction under context parallelism."""

    copies: Mapping[PlanAxis, float] = dataclasses.field(
        default_factory=lambda: {"tensor": 1.0, "expert": 1.0}
    )
    base: float = 0.15
    context: float = 0.10

    def __post_init__(self) -> None:
        for axis, factor in self.copies.items():
            if axis not in PLAN_AXES or factor < 1.0:
                raise ValueError(
                    f"copies[{axis!r}]={factor!r}: a plan axis and a factor of at "
                    "least one copy"
                )
        if self.base < 0 or self.context < 0:
            raise ValueError("headroom fractions are non-negative")

    def headroom(self, whole: int, geometry: ParallelGeometry) -> int:
        """The bytes over the resident weights ``geometry`` needs on a rank."""
        fraction = self.base + (self.context if geometry.context > 1 else 0.0)
        return round(whole * fraction)

    def footprint(
        self,
        table: PlacementTable,
        geometry: ParallelGeometry,
        itemsize: int,
        *,
        num_layers: int | None = None,
    ) -> tuple[int, ...]:
        """Per rank: an empirical peak estimate, including headroom.

        This estimate can be above or below a workload's peak; it is not
        the resident-weight admission bound.
        """
        whole = whole_bytes(table, itemsize)
        extra = self.headroom(whole, geometry)
        return tuple(
            resident + extra
            for resident in estimate_resident(
                table, geometry, itemsize, copies=self.copies, num_layers=num_layers
            )
        )

    def describe(self, geometry: ParallelGeometry) -> str:
        """The rule in one sentence, for a report or a refusal."""
        parts = [
            f"{self.base:.0%} of the model's bytes for activations, captures and "
            "the allocator's pool"
        ]
        if geometry.context > 1:
            parts.append(
                f"+{self.context:.0%} under cp for the gathered keys and values"
            )
        for axis, factor in self.copies.items():
            if factor != 1.0 and getattr(geometry, axis) > 1:
                parts.append(f"each {axis}-sharded parameter held {factor:g}x")
        return "; ".join(parts) + " (measured, docs/model_parallelism.md §11)"


#: Empirical headroom defaults for the standard workflow (module docstring).
RULE = EstimateRule()


def format_bytes(count: int) -> str:
    """``64.56 GiB`` — two decimals, always GiB, the unit §11 speaks."""
    return f"{count / _GIB:.2f} GiB"


@dataclasses.dataclass(frozen=True)
class Fit:
    """A geometry that fits: its per-rank footprint's maximum."""

    geometry: ParallelGeometry
    footprint: int


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _candidates(world: int, num_layers: int) -> Iterable[ParallelGeometry]:
    """Every ``pp`` / ``tp`` / ``ep`` geometry of ``world`` ranks whose model
    group is carved by both sub-groups and whose stages each hold a layer."""
    for pipeline in _divisors(world):
        if pipeline > num_layers:
            continue
        model = world // pipeline
        for tensor in _divisors(model):
            for expert in _divisors(model):
                if max(tensor, expert) != model:
                    continue
                yield ParallelGeometry(pipeline=pipeline, tensor=tensor, expert=expert)


def fitting_geometries(
    info: ModelInfo,
    table: PlacementTable,
    itemsize: int,
    capacity: int,
    *,
    worlds: Sequence[int],
    rule: EstimateRule = RULE,
) -> tuple[Fit, ...]:
    """The ``pp`` / ``ep`` / ``tp`` geometries of the given ``worlds`` that
    [`check`][] accepts for ``info`` and
    whose footprint under ``rule`` stays within ``capacity`` on every rank,
    ordered by world size, then tensor ranks, then expert ranks, with
    pipeline ranks carrying the remainder.
    """
    fits: list[Fit] = []
    num_layers = min(info.num_layers, _num_layers(table) or info.num_layers)
    for world in worlds:
        for geometry in _candidates(world, num_layers):
            if check(geometry, info):
                continue
            worst = max(
                rule.footprint(table, geometry, itemsize, num_layers=num_layers)
            )
            if worst <= capacity:
                fits.append(Fit(geometry, worst))
    fits.sort(
        key=lambda fit: (
            fit.geometry.world,
            fit.geometry.tensor,
            fit.geometry.expert,
            fit.geometry.pipeline,
        )
    )
    return tuple(fits)


def _spell(geometry: ParallelGeometry) -> str:
    """The geometry with its axes at one left out (``pp=2,ep=2``)."""
    named = [
        item for item in format_geometry(geometry).split(",") if not item.endswith("=1")
    ]
    return ",".join(named) or "world 1"


def memory_refusal(
    *,
    geometry: ParallelGeometry,
    rank: int,
    device: str,
    dtype: str,
    resident: int,
    footprint: int,
    free: int,
    total: int,
    fits: Sequence[Fit],
    rule: EstimateRule = RULE,
) -> str:
    """The refusal text (§2's idiom: opening with ``--parallel:`` and naming
    the fact that fails): the rank and its device, the weights the geometry
    would place there and the footprint with the rule's headroom, the
    device's free and total bytes, and the first two geometries that would
    fit — of the same world first, then the next world up — or that none
    of the searched worlds does. The search is over ``pp`` / ``ep`` / ``tp``
    with ``dp`` and ``cp`` held at one, and says so: a ``dp`` or ``cp`` run
    is answered in the placement axes, which is the advice (§11) but not
    the run that was asked for."""
    head = (
        f"--parallel: {_spell(geometry)} would place {format_bytes(resident)} of "
        f"{dtype} weights on rank {rank} ({device}) — an estimated "
        f"{format_bytes(footprint)} at the run's peak with {rule.describe(geometry)} "
        f"— and the device has {format_bytes(free)} available of {format_bytes(total)}; "
        "searched pp/ep/tp with dp and cp held at 1"
    )
    if fits:
        named = "; ".join(
            f"{_spell(fit.geometry)} (world {fit.geometry.world}, about "
            f"{format_bytes(fit.footprint)} per rank)"
            for fit in fits[:2]
        )
        tail = f"geometries estimated to fit here: {named}"
    else:
        tail = (
            "no pp/ep/tp geometry of this world or twice its size satisfies "
            "the headroom estimate"
        )
    return f"{head}; {tail}. Refused before any weight is read (--device {device})."


@dataclasses.dataclass(frozen=True)
class MemoryEstimate:
    """Resident weights are the admission bound; the footprint is advisory."""

    resident: int
    footprint: int


def memory_estimate(
    *,
    geometry: ParallelGeometry,
    rank: int,
    dtype: str,
    table: PlacementTable,
    info: ModelInfo,
    rule: EstimateRule = RULE,
) -> MemoryEstimate:
    """Resident weights and empirical peak for one rank of a whole-model table.

    The resident count is independent of ``rule``. Its copies and headroom
    describe a workload estimate, not additional weights the loader must hold.

    Raises:
        ProtocolError: ``P4`` — ``dtype`` is not one of the document's
            precision words; ``rank`` is not an int (a ``bool`` is not a
            rank) or is outside the geometry's world.
    """
    itemsize = DTYPE_ITEMSIZES.get(dtype)
    if itemsize is None:
        raise ProtocolError(
            "P4",
            f"dtype {dtype!r} is not one of {sorted(DTYPE_ITEMSIZES)}",
            path="model.dtype",
        )
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ProtocolError(
            "P4", f"rank must be an int, got {rank!r}", path="--parallel"
        )
    if not 0 <= rank < geometry.world:
        raise ProtocolError(
            "P4",
            f"rank {rank} is outside range({geometry.world}) of geometry "
            f"{format_geometry(geometry)}",
            path="--parallel",
        )
    num_layers = min(info.num_layers, _num_layers(table) or info.num_layers)
    footprint = rule.footprint(table, geometry, itemsize, num_layers=num_layers)[rank]
    resident = estimate_resident(table, geometry, itemsize, num_layers=num_layers)[rank]
    return MemoryEstimate(resident=resident, footprint=footprint)


def memory_check(
    *,
    geometry: ParallelGeometry,
    rank: int,
    device: str,
    dtype: str,
    table: PlacementTable,
    info: ModelInfo,
    free: int,
    total: int,
    rule: EstimateRule = RULE,
) -> str | None:
    """Reject only when resident weights exceed available memory.

    ``free`` includes reusable allocator cache. The empirical footprint
    appears in diagnostics and the alternative-geometry search, but cannot
    reject a load whose resident weights fit. Runtime callers should warn
    when [`memory_estimate`][] exceeds this available memory instead.
    ``table`` must describe the whole model before pipeline placement.

    Raises:
        ProtocolError: an invalid dtype or rank ([`memory_estimate`][]).
    """
    estimate = memory_estimate(
        geometry=geometry, rank=rank, dtype=dtype, table=table, info=info, rule=rule
    )
    if estimate.resident <= free:
        return None
    fits = fitting_geometries(
        info,
        table,
        DTYPE_ITEMSIZES[dtype],
        free,
        worlds=(geometry.world, 2 * geometry.world),
        rule=rule,
    )
    return memory_refusal(
        geometry=geometry,
        rank=rank,
        device=device,
        dtype=dtype,
        resident=estimate.resident,
        footprint=estimate.footprint,
        free=free,
        total=total,
        fits=fits,
        rule=rule,
    )
