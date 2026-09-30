"""The parallel styles behind one protocol (``docs/model_parallelism.md``
§5.2, §10.8).

A plan row names a *style* — what tensor or expert parallelism does to one
module: how each of its parameters is cut over the group, and which
collective wraps its forward. The registry's closed set (``STYLES``) is
transformers' eight and the repository's ``kv_replicated``; ``apply_plan``
(``sharding.py``) walks the plan and asks a [`Styles`][] for the
[`Style`][] of each row, then ``validate`` / ``shard`` every parameter
the row names and ``install`` the row's module. Two implementations serve
the protocol:

* [`TransformersStyles`][causalab.neural.engines.pytorch_hooks.styles.dtensor.TransformersStyles] — production. transformers' own
  ``TensorParallelLayer`` objects over the mesh's ``DeviceMesh``: DTensor
  placeholders on the parameters, DTensor redistributes in the forward.
  Byte-identical to applying the styles by hand; this tier only names them.
* [`FragmentStyles`][causalab.neural.engines.pytorch_hooks.styles.fragment.FragmentStyles] — the same partition on plain tensors
  over any [`Collective`][causalab.neural.shared.parallel.collective.Collective]:
  each rank keeps its chunk of every sharded parameter, and the forward is
  wrapped with the autograd pairs of ``parallel/autograd.py``. Pure given a
  collective, so it runs under the tests' ``SimulatedWorld`` and under
  ``gloo`` alike, and the conformance suite
  (``tests/neural/shared/parallel/styles_contract.py``) holds the two tiers
  bit-identical to each other and to world 1.

The two are not peers in what runs them. The fragment tier ships in
``causalab/`` rather than under ``tests/`` because it is real arithmetic
over a real protocol, held to the production tier by the conformance suite
— but nothing in the engine builds it today: ``apply_plan`` defaults to
``TransformersStyles`` and the loader passes no other. Its consumer is the
simulated tier, which is how ``docs/model_parallelism.md`` §10.6's DTensor
boundary was closed (the simulator cannot host DTensor); the cost is a
second derivation of transformers' forward wraps whose only oracle is the
suite. It becomes a production path the day a backend without DTensor — a
CPU fit under ``gloo``, say — asks ``apply_plan`` for it (§10.8).

What each style does, in both tiers:

=================================  ==========================  ===========================================
style                              parameter partition         forward wrap
=================================  ==========================  ===========================================
``colwise``                        weight rows (``ndim - 2``)  input gradient summed; output this rank's columns
``colwise_gather_output``          weight rows                 input gradient summed; output all-gathered
``rowwise``                        weight columns (``-1``);    output partial all-reduce-summed (identity
                                   bias whole                  backward); input already this rank's columns
``packed_colwise``                 rows, two interleaved       as ``colwise``; the output is ``slots=2``
                                   halves (gate, up)
``replicated_with_grad_allreduce`` whole                       parameter gradients summed on backward
``grouped_gemm``                   experts (dim 0); the        none (a parameter row)
                                   module's ``num_experts``
                                   becomes the local count
``moe_tp_experts``                 whole (its parameters       hidden input gradient summed (and the routing
                                   carry their own rows)       weights' under TP); output all-reduce-summed
``ep_router``                      whole                       scores masked and ids remapped to this rank's
                                                               experts; input gradient summed (§7)
``kv_replicated``                  whole                       output narrowed to this rank's KV head; input
                                                               and parameter gradients summed (§6.6)
=================================  ==========================  ===========================================

[`Partition`][] is the parameter half as data — the ranges of one
dimension this rank holds, transformers' ``Shard`` / ``_StridedShard``
chunk arithmetic spelled once — read by both tiers and by the loader's
shard-on-read plan (``shard_read.read_plan``), so the bytes a rank reads
and the tensor it holds are decided by the same table.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from causalab.protocol.registry import KV_REPLICATED, PLAN_AXES, PlanAxis, PlanRow

if TYPE_CHECKING:
    from causalab.neural.engines.pytorch_hooks.sharding import Sharding

__all__ = [
    "WHOLE",
    "Group",
    "Partition",
    "Style",
    "StyleError",
    "Styles",
    "partition_of",
]


class StyleError(ValueError):
    """A style asked of a module it cannot serve: a style outside the
    registry's set, a parameter of too few dimensions for its partition, a
    group the collective disagrees with. An internal invariant — the plan
    is the registry's — never a document's fault."""


@dataclasses.dataclass(frozen=True)
class Group:
    """One rank's place in its group on a plan axis: the axis, the
    group-local rank and the group's size.

    Raises:
        StyleError: an axis outside the plan axes, or a rank outside the
            group.
    """

    axis: PlanAxis
    rank: int
    size: int

    def __post_init__(self) -> None:
        if self.axis not in PLAN_AXES:
            raise StyleError(f"axis {self.axis!r} is not one of {list(PLAN_AXES)}")
        if isinstance(self.size, bool) or self.size < 1:
            raise StyleError(f"a group has at least one rank, got size {self.size!r}")
        if isinstance(self.rank, bool) or not 0 <= self.rank < self.size:
            raise StyleError(
                f"rank {self.rank!r} is outside the {self.axis} group of {self.size}"
            )

    @classmethod
    def of(cls, sharding: Sharding, axis: PlanAxis) -> Group:
        """This rank's group on ``axis`` under ``sharding``."""
        return cls(
            axis,
            sharding.layout.rank_in(sharding.rank, axis),
            getattr(sharding.geometry, axis),
        )


def _chunk(extent: int, size: int, rank: int) -> range:
    """Rank ``rank``'s contiguous chunk of ``extent`` split ``size`` ways —
    ``torch.chunk``'s rule, as DTensor's ``Shard`` spells it: equal chunks
    when the size divides, else chunks of ``ceil(extent / size)`` with the
    last one shorter (or empty)."""
    if extent % size == 0:
        width = extent // size
        return range(width * rank, width * (rank + 1))
    width = -(-extent // size)
    start = width * rank
    if extent < start:
        return range(extent, extent)
    return range(start, min(extent, start + width))


@dataclasses.dataclass(frozen=True)
class Partition:
    """How a style cuts one parameter over its group (module docstring):
    ``dim`` is the dimension chunked in rank order (``None``: the whole
    tensor on every rank), and ``interleave`` says the dimension is that
    many packed halves — the fused gate / up projection — each chunked, this
    rank holding its chunk of every half (DTensor's ``_StridedShard``).

    Raises:
        StyleError: ``interleave`` below one, or an interleave on a whole
            partition.
    """

    dim: int | None = None
    interleave: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.interleave, bool) or self.interleave < 1:
            raise StyleError(f"interleave must be at least 1, got {self.interleave!r}")
        if self.dim is None and self.interleave != 1:
            raise StyleError("a whole partition has nothing to interleave")

    @property
    def whole(self) -> bool:
        return self.dim is None

    def axis(self, ndim: int) -> int:
        """``dim`` as a non-negative index into an ``ndim``-D tensor.

        Raises:
            StyleError: the tensor has no such dimension.
        """
        if self.dim is None:
            raise StyleError("a whole partition names no dimension")
        if not -ndim <= self.dim < ndim:
            raise StyleError(
                f"partition dimension {self.dim} is outside a {ndim}-D tensor"
            )
        return self.dim % ndim

    def ranges(self, extent: int, rank: int, size: int) -> tuple[range, ...]:
        """The ranges of the partitioned dimension of ``extent`` that rank
        ``rank`` of ``size`` holds, in order: one for a plain chunk (possibly
        empty, when the ranks outnumber the rows), one per non-empty half
        for an interleaved partition — transformers' ``DtensorShardOperation``
        arithmetic, so the loader's pieces are what the load will index."""
        if self.dim is None:
            return (range(0, extent),)
        if self.interleave == 1:
            return (_chunk(extent, size, rank),)
        width = -(-extent // self.interleave)
        out: list[range] = []
        for half in range(self.interleave):
            start = half * width
            length = min(start + width, extent) - start
            if length <= 0:
                continue
            chunk = _chunk(length, size, rank)
            if len(chunk):
                out.append(range(start + chunk.start, start + chunk.stop))
        return tuple(out)

    def local(self, tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
        """This rank's chunk of a whole ``tensor`` — its own storage."""
        if self.dim is None:
            return tensor.clone()
        axis = self.axis(tensor.dim())
        pieces = [
            tensor.narrow(axis, piece.start, len(piece))
            for piece in self.ranges(tensor.shape[axis], rank, size)
        ]
        if not pieces:
            return tensor.narrow(axis, 0, 0).clone()
        return torch.cat(pieces, dim=axis) if len(pieces) > 1 else pieces[0].clone()


#: The whole tensor on every rank.
WHOLE = Partition()

_LEAF_BIAS = "bias"


def partition_of(style: str, ndim: int, parameter: str) -> Partition:
    """The [`Partition`][] ``style`` gives a parameter of ``ndim``
    dimensions named ``parameter`` (its leaf name — ``weight``, ``bias``,
    ``gate_up_proj``), the table of the module docstring: transformers'
    ``shard_param`` of each style, for the ``Linear``-shaped and fused
    expert parameters the served plans name.

    Raises:
        StyleError: a style outside the registry's set, or a parameter with
            too few dimensions for the style's partition.
    """
    if style in ("colwise", "colwise_gather_output"):
        return Partition(_rows(ndim, style))
    if style == "rowwise":
        return WHOLE if parameter == _LEAF_BIAS else Partition(-1)
    if style == "packed_colwise":
        return Partition(-1) if ndim == 1 else Partition(_rows(ndim, style), 2)
    if style == "grouped_gemm":
        return Partition(0)
    if style in (
        "replicated_with_grad_allreduce",
        "ep_router",
        "moe_tp_experts",
        KV_REPLICATED,
    ):
        return WHOLE
    raise StyleError(f"parallel style {style!r} has no partition rule")


def _rows(ndim: int, style: str) -> int:
    """The output-feature dimension of a ``Linear``-shaped weight
    ``(out, in)`` or a stacked expert weight ``(experts, out, in)``: the
    second-last; a 1-D bias is its own dimension."""
    if ndim < 1:
        raise StyleError(f"style {style!r} cannot partition a 0-D parameter")
    return ndim - 2 if ndim >= 2 else -1


@runtime_checkable
class Style(Protocol):
    """One row's behaviour (module docstring). ``path`` in ``validate`` is
    the parameter's full dotted path, for the refusal's message.
    ``expert_parallel`` in ``install`` says the row sits on the expert
    axis — the experts' style sums the routing weights' gradient under
    tensor parallelism only, and the router's input gradient is summed
    under expert parallelism (§7)."""

    def validate(
        self, module: torch.nn.Module, parameter: str, group: Group, *, path: str
    ) -> None: ...

    def shard(self, module: torch.nn.Module, parameter: str, group: Group) -> None: ...

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None: ...


@runtime_checkable
class Styles(Protocol):
    """The style of each plan row."""

    def style(self, row: PlanRow) -> Style: ...
