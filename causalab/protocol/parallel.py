"""The parallel geometry (``docs/model_parallelism.md`` §2): how many ranks a
run spans on each axis, the ``--parallel`` grammar that authors one, the
torch-free check of a geometry against a registry entry, and the integer
mesh that says which ranks form a group.

Geometry is **execution, never identity** (``docs/intervention_protocol.md``
§8): it enters no canonical form, no digest and no stamp. The receipt records
it (``run.py:execution_record``) and ``dry-run`` checks it before any weights
load — which is why this module is torch-free and imports nothing above the
registry.

The mesh is ``(data, pipeline, context, model)``, outermost first, row-major.
The ``data`` axis has two **modes** (§8.3): ``points`` — replica ``d`` runs a
contiguous shard of the campaign's points, exact by construction — and
``rows`` — every replica runs every point and each fit minibatch is split
across the replicas, the loss mean and the featurizer gradient agreed. The
mode is spelled on the flag (``dp=2:rows``), is execution like the rest of
the geometry, and its two document rules ([`check_rows`][]) are torch-free
like the model rules.
``model`` is the group that holds one replicated residual stream; ``tensor``
and ``expert`` are contiguous sub-groups of each model group, so ``model =
max(tensor, expert)`` and both must divide it. The ``context`` axis splits the
**padded position axis** of every forward into one contiguous chunk per rank
(§8.4, [`sequence_chunks`][]); its one document rule — no decode — is
[`check_context`][], torch-free like the rest, and its one run-time fact
(a frame shorter than the group) is refused where the frame is known. Every rank arithmetic here is
pure and total over ``range(world)``: the placement and collective seams
(``neural/shared/parallel/``) name groups by [`Axis`][] and resolve them
through a [`MeshLayout`][].
"""

from __future__ import annotations

import dataclasses
import re
from typing import Callable, Iterable, Literal, Mapping, Sequence, get_args

from causalab.protocol.rules.errors import ParseError, suggest
from causalab.protocol.kv_replication import kv_refusal
from causalab.protocol.registry import PLAN_AXES, STYLES, ModelInfo, PlanAxis

__all__ = [
    "AXES",
    "CHECKS",
    "CONTEXT_EXPERIMENTAL_VARIABLE",
    "DATA_MODES",
    "DataMode",
    "GEOMETRY_AXES",
    "ONE",
    "PLANLESS_FAMILIES",
    "SPELLINGS",
    "Axis",
    "Check",
    "Coordinates",
    "GeometryAxis",
    "MeshLayout",
    "ParallelGeometry",
    "check",
    "check_context",
    "context_refusals",
    "check_rows",
    "expert_has_plan_rows",
    "experimental_context",
    "format_geometry",
    "hybrid_family",
    "parse_geometry",
    "sequence_chunks",
    "stage_layers",
    "sub_group_start",
    "tensor_has_plan_rows",
]

#: The mesh axes a group can be named over. ``model`` is the group that holds
#: one replicated residual stream; ``tensor`` and ``expert`` are its
#: sub-groups.
Axis = Literal["data", "pipeline", "context", "model", "tensor", "expert"]
AXES: tuple[Axis, ...] = get_args(Axis)

#: The five axes a geometry authors — ``model`` is derived from two of them.
GeometryAxis = Literal["data", "pipeline", "context", "tensor", "expert"]
GEOMETRY_AXES: tuple[GeometryAxis, ...] = get_args(GeometryAxis)

#: The two modes of the ``data`` axis (§8.3): ``points`` shards the campaign's
#: points across the replicas; ``rows`` runs every point on every replica and
#: splits each fit minibatch's rows across them. Spelled ``dp=<count>:<mode>``;
#: an unspelled mode is ``points``.
DataMode = Literal["points", "rows"]
DATA_MODES: tuple[DataMode, ...] = get_args(DataMode)

# the registry's plan rows shard over two of these axes (§5.2) — one vocabulary
assert set(PLAN_AXES) <= set(AXES)

#: The ``--parallel`` grammar's spellings, in the order [`format_geometry`][]
#: writes them.
SPELLINGS: Mapping[str, GeometryAxis] = {
    "dp": "data",
    "pp": "pipeline",
    "cp": "context",
    "tp": "tensor",
    "ep": "expert",
}
_SPELLING_OF: Mapping[GeometryAxis, str] = {axis: s for s, axis in SPELLINGS.items()}

#: Families with no parallel plan at all (§2, §11): transformers ships no
#: ``base_model_tp_plan`` for them, and the registry declares none, so every
#: axis above one is refused by name. ``gpt2`` and ``gptj`` are the two such
#: families the registry carries today (``GPTJConfig`` ships no
#: ``base_model_tp_plan`` in transformers 5.16.1); a family lands here, not in
#: a branch, when its plan is missing rather than merely unmeasured.
PLANLESS_FAMILIES: frozenset[str] = frozenset({"gpt2", "gptj"})

_FLAG = "--parallel"
_INTEGER = re.compile(r"^[0-9]+$")
#: The one axis that takes a mode.
_MODE_AXIS: GeometryAxis = "data"


def _refuse(axis: GeometryAxis | None, message: str) -> ParseError:
    path = _FLAG if axis is None else f"{_FLAG}.{axis}"
    return ParseError("P4", message, path=path)


def _axis_value(axis: GeometryAxis, value: object) -> int:
    """``value`` as a rank count: a positive ``int`` (never a ``bool``);
    anything else is the ``P4`` naming ``--parallel.<axis>``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refuse(
            axis,
            f"{_SPELLING_OF[axis]} must be a positive integer rank count, got {value!r}",
        )
    if value < 1:
        raise _refuse(
            axis,
            f"{_SPELLING_OF[axis]} must be at least 1 (1 means no parallelism on "
            f"this axis), got {value}",
        )
    return value


def _data_mode(value: object) -> DataMode:
    """``value`` as a data mode: one of [`DATA_MODES`][]; anything else is
    the ``P4`` naming ``--parallel.data`` and the modes."""
    if not isinstance(value, str) or value not in DATA_MODES:
        raise _refuse(
            _MODE_AXIS,
            f"the data axis has the modes {', '.join(DATA_MODES)} (spelled "
            f"dp=<count>:<mode>), got {value!r}",
        )
    return "rows" if value == "rows" else "points"


@dataclasses.dataclass(frozen=True)
class ParallelGeometry:
    """How many ranks a run spans on each axis (§2), and the mode of the
    ``data`` axis (§8.3). All ones — the default — is one process on one
    device with no ``torch.distributed`` initialised."""

    data: int = 1  # replicas, over points or over a fit's rows (data_mode)
    pipeline: int = 1  # layer stages
    context: int = 1  # sequence chunks
    tensor: int = 1  # attention / dense-MLP shards
    expert: int = 1  # MoE expert shards
    #: what the replicas divide: the campaign's points, or every fit
    #: minibatch's rows (§8.3); ``points`` unless the flag spells ``:rows``
    data_mode: DataMode = "points"

    def __post_init__(self) -> None:
        for axis in GEOMETRY_AXES:
            _axis_value(axis, getattr(self, axis))
        _data_mode(self.data_mode)

    @property
    def model(self) -> int:
        """The model-parallel group: ``max(tensor, expert)``."""
        return max(self.tensor, self.expert)

    @property
    def world(self) -> int:
        """``data * pipeline * context * model`` — the number of ranks."""
        return self.data * self.pipeline * self.context * self.model


#: The default geometry: world 1.
ONE = ParallelGeometry()


def parse_geometry(text: str) -> ParallelGeometry:
    """The ``--parallel`` grammar: ``tp=4,ep=8`` — comma-separated
    ``<axis>=<count>`` items over the spellings ``dp, pp, cp, tp, ep``, each at
    most once, each a positive decimal integer; an axis not named is 1, and
    the empty string is the default geometry. The data axis alone takes a
    mode after a colon — ``dp=2:rows`` (§8.3), ``dp=2:points`` the explicit
    default — one of [`DATA_MODES`][].

    Raises:
        ParseError: ``P4`` naming ``--parallel.<axis>`` (or ``--parallel`` for
            an item that is not ``axis=value``): an unknown or repeated axis,
            a value that is not a decimal integer, a value below 1, a mode on
            any axis but ``dp``, an empty or unknown mode.
    """
    if not text.strip():
        return ONE
    values: dict[GeometryAxis, int] = {}
    mode: DataMode = "points"
    for item in text.split(","):
        spelling, equals, raw = (part.strip() for part in item.partition("="))
        if not equals or not spelling:
            raise _refuse(
                None,
                f"expected <axis>=<count> items separated by commas, got {item.strip()!r} "
                f"in {text!r}; the axes are {', '.join(SPELLINGS)}",
            )
        axis = SPELLINGS.get(spelling)
        if axis is None:
            raise _refuse(
                None,
                f"unknown axis {spelling!r}; expected one of "
                f"{', '.join(SPELLINGS)}{suggest(spelling, SPELLINGS)}",
            )
        if axis in values:
            raise _refuse(axis, f"{spelling} is given twice in {text!r}")
        count, colon, mode_text = (part.strip() for part in raw.partition(":"))
        if colon and axis != _MODE_AXIS:
            raise _refuse(
                axis,
                f"{spelling}={raw} names a mode, and only the data axis has "
                f"modes ({', '.join(DATA_MODES)}: dp=<count>:<mode>)",
            )
        if not _INTEGER.match(count):
            raise _refuse(
                axis,
                f"{spelling} must be a positive decimal integer rank count, "
                f"got {count!r}",
            )
        values[axis] = _axis_value(axis, int(count))
        if colon:
            if not mode_text:
                raise _refuse(
                    axis,
                    f"{spelling}={raw} names no mode after the colon; the data "
                    f"axis has the modes {', '.join(DATA_MODES)}",
                )
            if mode_text not in DATA_MODES:
                raise _refuse(
                    axis,
                    f"unknown data mode {mode_text!r} in {spelling}={raw}; the "
                    f"modes are {', '.join(DATA_MODES)}"
                    f"{suggest(mode_text, DATA_MODES)}",
                )
            mode = _data_mode(mode_text)
    return ParallelGeometry(**values, data_mode=mode)


def format_geometry(geometry: ParallelGeometry) -> str:
    """The inverse of [`parse_geometry`][]: every axis spelled, in
    ``dp, pp, cp, tp, ep`` order — ``dp=1,pp=1,cp=1,tp=4,ep=8`` — the data
    mode spelled only when it is not the default: ``dp=2:rows,pp=1,…``."""
    items: list[str] = []
    for spelling, axis in SPELLINGS.items():
        item = f"{spelling}={getattr(geometry, axis)}"
        if axis == _MODE_AXIS and geometry.data_mode != "points":
            item += f":{geometry.data_mode}"
        items.append(item)
    return ",".join(items)


# --------------------------------------------------------------------------- #
# check — the §2 refusals, torch-free, before any weights
# --------------------------------------------------------------------------- #

#: One refusal rule: the refusal text, opening with ``--parallel.<axis>:``
#: and naming the model fact that fails, or ``None`` when the rule holds.
Check = Callable[[ParallelGeometry, ModelInfo], "str | None"]


def _text(axis: GeometryAxis, geometry: ParallelGeometry, fact: str) -> str:
    spelling = _SPELLING_OF[axis]
    return f"{_FLAG}.{axis}: {spelling}={getattr(geometry, axis)} {fact}"


def planless_family(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    """A family with no parallel plan refuses every axis above one — one
    refusal per such axis, so each names its own ``--parallel.<axis>``."""
    if info.family not in PLANLESS_FAMILIES:
        return None
    named: list[GeometryAxis] = [
        axis for axis in GEOMETRY_AXES if getattr(geometry, axis) > 1
    ]
    if not named:
        return None
    return "\n".join(
        _text(
            axis,
            geometry,
            f"is refused: family {info.family!r} of model {info.key!r} has no "
            "parallel plan (transformers ships none), so every axis must be 1",
        )
        for axis in named
    )


def tensor_divides_heads(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    if info.num_heads % geometry.tensor == 0:
        return None
    return _text(
        "tensor",
        geometry,
        f"does not divide num_heads={info.num_heads} of model {info.key!r}",
    )


def tensor_fits_kv_heads(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    """The KV fact (§6.6): the tensor axis divides the KV heads — a colwise
    shard of them — or is a multiple of them — one KV head replicated per
    rank, the plan rewritten by ``ParallelPlan.for_geometry``. Anything else
    straddles and is refused naming both facts. Silent where the head fact
    already fails, so that refusal stays the one line."""
    if info.num_heads % geometry.tensor:
        return None
    refusal = kv_refusal(
        num_heads=info.num_heads,
        num_kv_heads=info.num_kv_heads,
        tensor=geometry.tensor,
        key=info.key,
    )
    return None if refusal is None else _text("tensor", geometry, refusal)


def expert_divides_experts(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    if geometry.expert == 1:
        return None
    if info.num_experts is None:
        return _text(
            "expert",
            geometry,
            f"asks for expert parallelism on a dense model: {info.key!r} has no "
            "routed experts (num_experts is unset)",
        )
    if info.num_experts % geometry.expert == 0:
        return None
    return _text(
        "expert",
        geometry,
        f"does not divide num_experts={info.num_experts} of model {info.key!r}",
    )


def pipeline_within_layers(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    if geometry.pipeline <= info.num_layers:
        return None
    return _text(
        "pipeline",
        geometry,
        f"exceeds num_layers={info.num_layers} of model {info.key!r}: a stage "
        "holds at least one layer",
    )


def tensor_divides_model(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    if geometry.model % geometry.tensor == 0:
        return None
    return _text(
        "tensor",
        geometry,
        f"does not divide the model group model={geometry.model} "
        f"(max(tp, ep)): tensor groups are contiguous sub-groups of it",
    )


def expert_divides_model(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    if geometry.model % geometry.expert == 0:
        return None
    return _text(
        "expert",
        geometry,
        f"does not divide the model group model={geometry.model} "
        f"(max(tp, ep)): expert groups are contiguous sub-groups of it",
    )


def _plan_rows(
    axis: PlanAxis, geometry: ParallelGeometry, info: ModelInfo
) -> str | None:
    """The §5.2 rule for one plan axis: a geometry above one on ``axis``
    needs a plan row on it, and every row on it must be a style this phase
    serves (``registry.STYLES``) — an unserved style is refused **by name**
    here, before any weights, the way nnsight's plan refuses a style outside
    its table before a wrong number. An entry declaring no plan (``None``)
    is checked on its divisibility facts alone; the load derives its plan.
    A dense entry says nothing on the expert axis: it has no experts to
    shard, and [`expert_divides_experts`][] already refuses it."""
    if getattr(geometry, axis) == 1 or info.parallel_plan is None:
        return None
    if axis == "expert" and info.num_experts is None:
        return None
    rows = info.parallel_plan.rows_on(axis)
    if not rows:
        return _text(
            axis,
            geometry,
            f"is refused: the parallel plan of family {info.family!r} (model "
            f"{info.key!r}) has no {axis}-axis row, so nothing is sharded over "
            f"the {axis} group",
        )
    unserved = {pattern: row.style for pattern, row in rows.items() if not row.served}
    if unserved:
        named = ", ".join(
            f"{pattern} → {style}" for pattern, style in sorted(unserved.items())
        )
        return _text(
            axis,
            geometry,
            f"is refused: the parallel plan of family {info.family!r} (model "
            f"{info.key!r}) names {len(unserved)} style(s) this engine does not "
            f"serve on the {axis} axis ({named}); the served styles are "
            f"{', '.join(sorted(STYLES))}",
        )
    return None


def tensor_has_plan_rows(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    return _plan_rows("tensor", geometry, info)


def expert_has_plan_rows(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
    return _plan_rows("expert", geometry, info)


#: The §2 refusals, in report order. A tuple rather than a body so a test can
#: drop or swap one rule through this seam and show the property fail
#: (docs/model_parallelism.md §10.3, the "geometry.check" mutations). The
#: plan rules (§5.2) come after the divisibility facts, so the first refusal
#: on an axis stays the model fact.
CHECKS: tuple[Check, ...] = (
    planless_family,
    tensor_divides_heads,
    tensor_fits_kv_heads,
    expert_divides_experts,
    pipeline_within_layers,
    tensor_divides_model,
    expert_divides_model,
    tensor_has_plan_rows,
    expert_has_plan_rows,
)


def check(geometry: ParallelGeometry, info: ModelInfo) -> tuple[str, ...]:
    """Every §2 refusal of ``geometry`` against ``info``, each a line opening
    with ``--parallel.<axis>:`` and naming the model fact that fails; the
    empty tuple means the geometry is acceptable. Reports every independent
    refusal, not the first. Torch-free: runs in ``dry-run`` before weights."""
    out: list[str] = []
    for rule in CHECKS:
        text = rule(geometry, info)
        if text is not None:
            out.extend(text.split("\n"))
    return tuple(out)


def check_rows(
    geometry: ParallelGeometry, pairs: Sequence[int | None]
) -> tuple[str, ...]:
    """The two document rules of the ``rows`` mode (§8.3), each a line
    opening with ``--parallel.data:``; the empty tuple for the ``points``
    mode, whatever the document. ``pairs`` is one entry per point of the
    campaign: its ``train.batch.pairs``, or ``None`` for a point that
    declares no ``train``.

    A document with no ``train`` has no minibatch to split, so ``rows``
    would silently run every point replicated on every replica — refused,
    never quietly turned into ``points``. A ``pairs`` below the replica
    count would leave a replica with no rows of a minibatch — refused, since
    every replica holds at least one row (the engine refuses an epoch's
    remainder minibatch the same way once it knows the row count). Torch-
    free and data-free: runs in ``dry-run`` and before any weights in ``run``.
    """
    if geometry.data_mode != "rows":
        return ()
    spelled = f"dp={geometry.data}:rows"
    if not pairs or any(value is None for value in pairs):
        return (
            f"{_FLAG}.data: {spelled} splits every fit minibatch's rows across "
            "the replicas, but the document declares no train — a plain "
            "inference campaign has no minibatch to split; run it as "
            f"dp={geometry.data} over points, or add a train section",
        )
    smallest = min(value for value in pairs if value is not None)
    if smallest < geometry.data:
        return (
            f"{_FLAG}.data: {spelled} over train.batch.pairs={smallest} would "
            "leave a replica with no rows of a minibatch; every replica holds "
            "at least one, so pairs must be at least dp",
        )
    return ()


#: The environment variable that lets a hybrid tower run under ``cp > 1``
#: anyway (§8.4): ``"1"`` allows, ``"0"`` or unset refuses, anything else is
#: refused by name. Execution, never identity — it is read where the run
#: decides, not written into any document or digest.
CONTEXT_EXPERIMENTAL_VARIABLE = "CAUSALAB_EXPERIMENTAL_CONTEXT"


def hybrid_family(info: ModelInfo) -> bool:
    """Whether the model's tower mixes streams — a Gated DeltaNet
    (``linear_attention``) layer among the attention layers, as the
    registry's ``layer_types`` declares it. ``None`` (no declared pattern)
    counts as dense: the run-time stream check still applies to components,
    and a dense family loses nothing. Any DeltaNet layer makes the tower
    hybrid for the context-parallelism check. Set
    [`CONTEXT_EXPERIMENTAL_VARIABLE`][] to try an unsupported hybrid mix."""
    return info.layer_types is not None and "linear_attention" in info.layer_types


def experimental_context(environ: Mapping[str, str]) -> bool:
    """Whether [`CONTEXT_EXPERIMENTAL_VARIABLE`][] asks to run a hybrid
    tower under ``cp > 1`` regardless (§8.4): unset or empty is ``False``.

    Raises:
        ProtocolError: a value other than ``"1"`` or ``"0"`` — refused by
            name, never read as a silent no.
    """
    value = environ.get(CONTEXT_EXPERIMENTAL_VARIABLE)
    if value is None or value == "":
        return False
    if value == "1":
        return True
    if value == "0":
        return False
    raise _refuse(
        "context",
        f"{CONTEXT_EXPERIMENTAL_VARIABLE}={value!r} is not a switch: set it to "
        "'1' to run a hybrid tower under context parallelism anyway, to '0' or "
        "unset it to keep the refusal (docs/model_parallelism.md §8.4)",
    )


def check_context(
    geometry: ParallelGeometry,
    decodes: bool,
    *,
    hybrid: bool = False,
    experimental: bool = False,
) -> tuple[str, ...]:
    """The document and family rules of the ``context`` axis (§8.4), each a
    line opening with ``--parallel.context``; the empty tuple at ``cp=1``,
    whatever the document. ``decodes`` says whether any point of the campaign
    reads a ``generated`` position — a greedy continuation; ``hybrid``
    whether the tower mixes streams ([`hybrid_family`][]); ``experimental``
    whether [`CONTEXT_EXPERIMENTAL_VARIABLE`][] waives the family rule.

    A decode runs the model once per generated token over a growing KV
    cache; under ``cp > 1`` every step would be one position on one rank
    with the cache split across the group, which is unsupported —
    refused by name before any weights, never quietly run at ``cp=1``.

    A hybrid tower's DeltaNet layers hand their state chunk to chunk, so the
    group runs those layers in sequence while every rank still holds the
    whole model and the gathered keys and values. Context parallelism is
    therefore refused for hybrid towers unless the experimental variable
    waives the restriction; ``pp`` and ``ep`` provide model placement for
    hybrid MoE families. This check is torch-free and data-free: it runs in
    ``dry-run`` and before any weights in ``run``.
    """
    if geometry.context == 1:
        return ()
    out: list[str] = []
    if decodes:
        out.append(
            f"{_FLAG}.context: cp={geometry.context} splits every forward's "
            "positions across the context group, and the document decodes (a read "
            "at a generated position): a decode's steps would each be one position "
            "on one rank with the KV cache split across the group, which context "
            "parallelism does not serve (docs/model_parallelism.md §8.4); run a "
            "decoding document at cp=1"
        )
    if hybrid and not experimental:
        out.append(
            f"{_FLAG}.context: cp={geometry.context} on a hybrid tower (layer_types "
            "names linear_attention): the Gated DeltaNet layers hand their state "
            "chunk to chunk, so the group runs them in sequence while every rank "
            "still holds the whole model and the gathered keys and values — "
            "context parallelism buys neither memory nor time on this family "
            "(docs/model_parallelism.md §8.4, §11) and is refused for now; place "
            f"the model with pp or ep, or set {CONTEXT_EXPERIMENTAL_VARIABLE}=1 "
            "to run it anyway"
        )
    return tuple(out)


def context_refusals(
    geometry: ParallelGeometry,
    decodes: bool,
    infos: Iterable[ModelInfo],
    environ: Mapping[str, str],
) -> tuple[str, ...]:
    """[`check_context`][] with its facts derived here, once, for both
    verbs: ``hybrid`` folds [`hybrid_family`][] over **every** model of
    the campaign (``infos``, one per distinct ``model.key`` — a swept key
    whose hybrid tower sits at a later point is refused like the first's;
    ``decodes`` is campaign-wide the same way), and the waiver is read from
    ``environ`` only when the context axis is above one, so a malformed
    switch refuses the runs it could steer and no other — ``run`` and
    ``dry-run`` agree on what they accept.

    Raises:
        ProtocolError: a malformed [`CONTEXT_EXPERIMENTAL_VARIABLE`][]
            at ``cp > 1`` ([`experimental_context`][]).
    """
    if geometry.context == 1:
        return ()
    return check_context(
        geometry,
        decodes,
        hybrid=any(hybrid_family(info) for info in infos),
        experimental=experimental_context(environ),
    )


# --------------------------------------------------------------------------- #
# the mesh
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Coordinates:
    """One rank's position on every axis: its index within each of its groups."""

    data: int
    pipeline: int
    context: int
    model: int
    tensor: int
    expert: int


def sub_group_start(index: int, size: int) -> int:
    """The index, within a model group, where the contiguous sub-group of
    ``size`` ranks holding position ``index`` begins. A function of its own so
    the carve is one seam (§10.3, the "mesh groups" mutation)."""
    return index - index % size


class MeshLayout:
    """Pure integer arithmetic over a geometry's ranks (§2).

    Ranks are laid out row-major over ``(data, pipeline, context, model)``,
    outermost first, so a model group is ``model`` consecutive ranks and the
    data groups are the most widely strided. Inside each model group the
    ``tensor`` and ``expert`` groups are consecutive runs of ``tensor`` and
    ``expert`` ranks.

    Raises:
        ParseError: ``P4`` — ``tensor`` or ``expert`` does not divide the
            model group, so no contiguous carve partitions it.
    """

    def __init__(self, geometry: ParallelGeometry) -> None:
        self.geometry = geometry
        for axis in ("tensor", "expert"):
            size = getattr(geometry, axis)
            if geometry.model % size:
                raise _refuse(
                    axis,
                    f"{_SPELLING_OF[axis]}={size} does not divide the model group "
                    f"model={geometry.model} (max(tp, ep)), so no contiguous carve "
                    "partitions it",
                )
        self.world = geometry.world
        model = geometry.model
        self._sizes: dict[Axis, int] = {
            "data": geometry.data,
            "pipeline": geometry.pipeline,
            "context": geometry.context,
            "model": model,
            "tensor": geometry.tensor,
            "expert": geometry.expert,
        }
        self._strides: dict[Axis, int] = {
            "model": 1,
            "context": model,
            "pipeline": geometry.context * model,
            "data": geometry.pipeline * geometry.context * model,
        }

    def _rank(self, rank: int) -> int:
        if isinstance(rank, bool) or not 0 <= rank < self.world:
            raise ValueError(
                f"rank {rank!r} is outside range({self.world}) of geometry "
                f"{format_geometry(self.geometry)}"
            )
        return rank

    def coordinates(self, rank: int) -> Coordinates:
        """Every axis coordinate of ``rank``: on the four mesh axes its
        row-major index, on ``tensor`` / ``expert`` its position inside its
        sub-group (equal to [`rank_in`][] on every axis)."""
        rank = self._rank(rank)
        model_index = rank % self._sizes["model"]
        return Coordinates(
            data=(rank // self._strides["data"]) % self._sizes["data"],
            pipeline=(rank // self._strides["pipeline"]) % self._sizes["pipeline"],
            context=(rank // self._strides["context"]) % self._sizes["context"],
            model=model_index,
            tensor=model_index % self._sizes["tensor"],
            expert=model_index % self._sizes["expert"],
        )

    def group_of(self, rank: int, axis: Axis) -> tuple[int, ...]:
        """The ranks in ``rank``'s group on ``axis``, in rank order."""
        rank = self._rank(rank)
        size = self._sizes[axis]
        if axis in ("tensor", "expert"):
            model_index = rank % self._sizes["model"]
            start = rank - model_index + sub_group_start(model_index, size)
            return tuple(range(start, start + size))
        stride = self._strides[axis]
        start = rank - ((rank // stride) % size) * stride
        return tuple(start + k * stride for k in range(size))

    def rank_in(self, rank: int, axis: Axis) -> int:
        """``rank``'s position inside its group on ``axis``."""
        return getattr(self.coordinates(rank), axis)

    def groups(self, axis: Axis) -> tuple[tuple[int, ...], ...]:
        """Every group on ``axis``, ordered by first member; together they
        partition ``range(world)``."""
        return tuple(
            self.group_of(rank, axis)
            for rank in range(self.world)
            if self.rank_in(rank, axis) == 0
        )


# --------------------------------------------------------------------------- #
# pipeline placement of the layers (§6.5)
# --------------------------------------------------------------------------- #


def stage_layers(geometry: ParallelGeometry, rank: int, num_layers: int) -> range:
    """The decoder layers pipeline stage of ``rank`` holds: transformers'
    even split (``PipelineStage.layer_range_for_rank``, 5.16.1), transcribed
    — ``num_layers // pipeline`` per stage, the remainder on the last — read
    off the rank's pipeline coordinate, so every rank of one stage answers
    alike. Over the stages the ranges partition ``range(num_layers)``.

    Raises:
        ParseError: ``P4`` naming ``--parallel.pipeline`` when there are
            more stages than layers (a stage holds at least one layer).
    """
    if geometry.pipeline > num_layers:
        raise _refuse(
            "pipeline",
            f"pp={geometry.pipeline} exceeds num_layers={num_layers}: a stage "
            "holds at least one layer",
        )
    stage = MeshLayout(geometry).rank_in(rank, "pipeline")
    per_stage = num_layers // geometry.pipeline
    start = stage * per_stage
    end = num_layers if stage == geometry.pipeline - 1 else start + per_stage
    return range(start, end)


# --------------------------------------------------------------------------- #
# context placement of the positions (§8.4)
# --------------------------------------------------------------------------- #


def sequence_chunks(padded_len: int, context: int) -> tuple[range, ...]:
    """The contiguous chunk of the **padded** position axis each rank of a
    context group holds (§8.4), in rank order: ``context`` chunks of
    ``padded_len // context`` positions, the remainder on the last — the
    rule [`stage_layers`][] places layers by, on positions. Together the
    chunks partition ``range(padded_len)``; a pure function of the two
    integers, so every rank of the group answers alike without a collective.

    Raises:
        ParseError: ``P4`` naming ``--parallel.context`` when the frame is
            shorter than the group (a rank would hold no position), or when
            either argument is not a positive integer.
    """
    for name, value in (("padded_len", padded_len), ("cp", context)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _refuse(
                "context",
                f"{name} must be a positive integer, got {value!r}",
            )
    if padded_len < context:
        raise _refuse(
            "context",
            f"cp={context} splits a frame of {padded_len} padded position(s) "
            "across more ranks than it has positions; every rank holds at least "
            "one, so cp must be at most the padded length",
        )
    per_rank = padded_len // context
    return tuple(
        range(
            rank * per_rank,
            padded_len if rank == context - 1 else (rank + 1) * per_rank,
        )
        for rank in range(context)
    )
