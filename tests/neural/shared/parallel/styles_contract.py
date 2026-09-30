"""The ``Styles`` contract as executable rows every implementation shares
(``docs/model_parallelism.md`` §10.8).

``styles/__init__.py``'s table states what each parallel style does to a
module — the partition of its parameters, the collectives around its
forward; this module states each row as a `StyleContract`: a tiny
model holding one module under the row's plan pattern, built whole and
seeded on every rank, sharded through **``apply_plan``** with the tier's
[`Styles`][causalab.neural.engines.pytorch_hooks.styles.Styles], run forward
without and with grad, and read back — the whole output, this rank's local
output, every parameter's local tensor and gradient, the input's gradient.
The check holds every rank to the **world-1 run of the same program**: the
whole model on one rank under ``Solo``, where ``apply_plan`` applies
nothing. The fragment tier over the simulator, the fragment tier over
``gloo`` and transformers' DTensor tier over ``gloo`` are held to one set
of rows, and to each other bit for bit (``test_styles.py``).

**Exact by construction.** Every fixture's values are small multiples of a
quarter and every consumer weighs by quarters, so a sharded computation's
re-associated sums — a rowwise partial on each rank added by the
all-reduce, a colwise input gradient summed over the group, a replicated
norm's gradient summed over the ranks' heads — are exact whatever the
order, and the contract asserts ``torch.equal`` against world 1. The
routed-experts rows route every token's ``top_k`` slots to **one** rank's
experts, so the expert combine adds zeros from the other rank (``x + 0 ==
x``) and the router's masked slots contribute exactly zero gradient; the
real fixtures' cross-rank routing, whose sums are a reduction-order band,
are the sharded-model scenarios' (``test_sharded_simulation.py``), which
measure and pin it.

A program is a module-level function — a spawned process imports it by
name — taking the rank and its collective, the row, the *tier* (how a rank
builds its sharding and styles: `fragment_tier`, `transformers_tier`)
and the geometry. `every_row` runs every row of one geometry in one
program, so a spawned world runs the whole suite in one launch.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from causalab.neural.engines.pytorch_hooks.sharding import Sharding, apply_plan
from causalab.neural.engines.pytorch_hooks.styles import Styles, partition_of
from causalab.neural.engines.pytorch_hooks.styles.fragment import FragmentStyles
from causalab.neural.shared.parallel.collective import Collective, TorchCollective
from causalab.neural.shared.parallel.fragments import (
    fragment,
    reconstruct_routing,
    whole,
)
from causalab.neural.shared.parallel.placement import (
    ExpertLocal,
    Placement,
    Replicated,
    Sharded,
)
from causalab.protocol.parallel import ONE, MeshLayout, ParallelGeometry
from causalab.protocol.registry import KV_REPLICATED, ParallelPlan, PlanAxis, PlanRow

__all__ = [
    "ROWS",
    "BY_NAME",
    "Results",
    "StyleContract",
    "Tier",
    "every_row",
    "fragment_tier",
    "oracle",
    "run_row",
    "transformers_tier",
    "verify",
    "verify_all",
]

Results = dict[str, Any]
#: How a rank builds its sharding and styles from what it has.
Tier = Callable[[int, Collective, ParallelGeometry], tuple[Sharding, Styles]]

HIDDEN = 8
INTER = 16
TOKENS = 3
HEADS = 4
HEAD_DIM = 2
EXPERTS = 4
TOP_K = 2
TENSOR_2 = ParallelGeometry(tensor=2)
TENSOR_4 = ParallelGeometry(tensor=4)
EXPERT_2 = ParallelGeometry(expert=2)


def quarters(shape: Sequence[int], seed: int) -> torch.Tensor:
    """Multiples of a quarter in ``[-2, 2]``, seeded — exact in any sum."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 9, tuple(shape), generator=generator).float() / 4.0


# --------------------------------------------------------------------------- #
# the tiers
# --------------------------------------------------------------------------- #


def fragment_tier(
    rank: int, c: Collective, geometry: ParallelGeometry
) -> tuple[Sharding, Styles]:
    """Plain tensors over the collective — any collective."""
    return Sharding(geometry, rank, collective=c), FragmentStyles(c)


def transformers_tier(
    rank: int, c: Collective, geometry: ParallelGeometry
) -> tuple[Sharding, Styles]:
    """transformers' DTensor styles over the mesh the collective carries —
    the production path, ``Sharding.from_mesh``."""
    from causalab.neural.engines.pytorch_hooks.styles.dtensor import TransformersStyles

    assert isinstance(c, TorchCollective), "the DTensor tier needs the mesh's groups"
    sharding = Sharding.from_mesh(c.mesh)
    assert sharding.geometry == geometry and sharding.rank == rank
    return sharding, TransformersStyles(sharding)


# --------------------------------------------------------------------------- #
# the fixtures: one module under its plan pattern, values in quarters
# --------------------------------------------------------------------------- #


class Block(nn.Module):
    """One decoder block holding the row's module under its config name."""

    def __init__(self, **children: nn.Module) -> None:
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)
        self.order = tuple(children)

    def forward(self, x: torch.Tensor, *rest: Any) -> Any:
        out: Any = x
        for name in self.order:
            out = getattr(self, name)(out, *rest)
        return out


class Model(nn.Module):
    """``layers.0.<…>`` — the plan patterns' shape, no base-model prefix."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([block])

    def forward(self, x: torch.Tensor, *rest: Any) -> Any:
        return self.layers[0](x, *rest)


class Scale(nn.Module):
    """A per-head-dimension weight applied elementwise — the shape of a
    norm's parameter between a colwise and a rowwise projection, with
    exact arithmetic (a norm's ``rsqrt`` is not)."""

    def __init__(self, width: int, seed: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(quarters((width,), seed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Mixer(nn.Module):
    """The attention module above a K/V projection: the attribute the
    library repeats KV heads by, and the projection as its child."""

    def __init__(self, k_proj: nn.Module, groups: int) -> None:
        super().__init__()
        self.k_proj = k_proj
        self.num_key_value_groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.k_proj(x)


def _linear(out_features: int, in_features: int, seed: int, bias: bool) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        layer.weight.copy_(quarters((out_features, in_features), seed))
        if bias:
            layer.bias.copy_(quarters((out_features,), seed + 1))
    return layer


def _one(name: str, module: nn.Module, parent: str = "mlp") -> Model:
    return Model(Block(**{parent: Block(**{name: module})}))


def build_colwise() -> Model:
    return _one("gate_proj", _linear(INTER, HIDDEN, 11, bias=True))


def build_rowwise() -> Model:
    return _one("down_proj", _linear(HIDDEN, INTER, 12, bias=False))


def build_gather_output() -> Model:
    return _one("in_proj_qkv", _linear(INTER, HIDDEN, 13, bias=False), "linear_attn")


def build_packed() -> Model:
    return _one("gate_up_proj", _linear(2 * INTER, HIDDEN, 14, bias=False))


def build_replicated() -> Model:
    return _one("q_norm", Scale(HEAD_DIM, 15), "self_attn")


def build_kv() -> Model:
    # one KV head of HEAD_DIM, held by both ranks of the pair (repeat 2);
    # the mixer's HEADS query heads are HEADS / 1 groups per KV head
    return Model(
        Block(self_attn=Mixer(_linear(HEAD_DIM, HIDDEN, 16, bias=True), HEADS))
    )


def _moe_config() -> Any:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    config = Qwen3_5MoeTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HEAD_DIM,
        vocab_size=64,
    )
    config._experts_implementation = "grouped_mm"  # pyright: ignore[reportPrivateUsage]
    return config


def _router_weight() -> torch.Tensor:
    """A router that sends tokens ``0`` and ``2`` to experts ``{0, 1}`` and
    token ``1`` to experts ``{2, 3}`` — every token's slots on one rank at
    ``ep=2`` — through `moe_input`'s one-hot rows: expert ``e`` scores
    a token on the feature its row weighs, the first of a pair a little
    above the second so the top-``k`` order is fixed."""
    weight = torch.zeros(EXPERTS, HIDDEN)
    for feature, (first, second) in ((0, (0, 1)), (2, (0, 1)), (1, (2, 3))):
        weight[first, feature] = 2.0
        weight[second, feature] = 1.5
    return weight


def moe_input() -> torch.Tensor:
    """``(1, TOKENS, HIDDEN)``: token ``t`` is ``2`` on feature ``t`` plus
    quarters elsewhere scaled down, so the router's argmax is feature ``t``."""
    x = quarters((1, TOKENS, HIDDEN), 21) / 8.0
    for token in range(TOKENS):
        x[0, token, token] = 2.0
    return x


def _fill_moe(block: nn.Module) -> None:
    with torch.no_grad():
        block.gate.weight.copy_(_router_weight())
        block.experts.gate_up_proj.copy_(
            quarters(tuple(block.experts.gate_up_proj.shape), 22)
        )
        block.experts.down_proj.copy_(
            quarters(tuple(block.experts.down_proj.shape), 23)
        )
        for index, layer in enumerate(
            (
                block.shared_expert.gate_proj,
                block.shared_expert.up_proj,
                block.shared_expert.down_proj,
                block.shared_expert_gate,
            )
        ):
            layer.weight.copy_(quarters(tuple(layer.weight.shape), 24 + index))


def build_moe() -> Model:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    block = Qwen3_5MoeSparseMoeBlock(_moe_config())
    _fill_moe(block)
    return Model(Block(mlp=block))


class RouterOnly(nn.Module):
    """The sparse block's router alone, its three outputs returned."""

    def __init__(self, gate: nn.Module) -> None:
        super().__init__()
        self.gate = gate

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.gate(x.reshape(-1, HIDDEN)))


def build_router() -> Model:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTopKRouter,
    )

    gate = Qwen3_5MoeTopKRouter(_moe_config())
    with torch.no_grad():
        gate.weight.copy_(_router_weight())
    return Model(Block(mlp=RouterOnly(gate)))


# --------------------------------------------------------------------------- #
# inputs, wholes and losses — module-level, so a program pickles by name
# --------------------------------------------------------------------------- #


def plain_input() -> torch.Tensor:
    return quarters((TOKENS, HIDDEN), 31)


def wide_input() -> torch.Tensor:
    return quarters((TOKENS, INTER), 32)


def headed_input() -> torch.Tensor:
    return quarters((TOKENS, HEADS, HEAD_DIM), 33)


def whole_by(placement: Placement, out: Any, c: Collective) -> Any:
    return whole(out, placement, c)


def whole_router(out: Any, c: Collective) -> Any:
    """The router's three outputs made whole: the logits are replicated,
    the scores are expert-local (zeros elsewhere: their sum), the table is
    reconstructed from the ranks' remapped tables (§6.3)."""
    logits, scores, indices = out
    return (
        logits,
        whole(scores, ExpertLocal("expert"), c),
        reconstruct_routing(indices, EXPERTS, c),
    )


def readout(shape: Sequence[int], seed: int) -> torch.Tensor:
    return quarters(shape, seed)


def loss_over_whole(out: Any, c: Collective, *, seed: int) -> torch.Tensor:
    """``Σ whole · R`` — one scalar every rank agrees on; for a tuple the
    scores (index 1) are read."""
    value = out[1] if isinstance(out, tuple) else out
    return (value * readout(tuple(value.shape), seed)).sum()


def loss_per_rank(out: torch.Tensor, c: Collective, *, seed: int) -> torch.Tensor:
    """The consumer of a replicated K/V projection differs per rank — its
    own query heads: rank ``r`` of ``n`` weighs the held head by its
    ``1 / n`` share of a readout over the pair's query heads, so the ranks'
    losses sum to the world-1 loss."""
    size, rank = c.size("tensor"), c.rank("tensor")
    weights = readout((TENSOR_2.tensor, *out.shape), seed).reshape(size, -1, *out.shape)
    return (out * weights[rank].sum(0)).sum()


# --------------------------------------------------------------------------- #
# the rows
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class StyleContract:
    """One row: the plan naming the style, the model built whole, what each
    rank feeds and how it reads the output back, the consumer, and the facts
    the module must carry after the plan (``num_experts``,
    ``num_key_value_groups``)."""

    name: str
    plan: ParallelPlan
    geometries: tuple[ParallelGeometry, ...]
    build: Callable[[], nn.Module]
    inputs: Callable[[], torch.Tensor]
    input_placement: Placement
    output_placement: Placement | None
    whole_of: Callable[[Any, Collective], Any]
    loss: Callable[[Any, Collective], torch.Tensor]
    facts: Callable[[nn.Module, int], Mapping[str, Any]] = lambda model, size: {}
    #: Parameters whose gradient stays a partial per rank: the router's own
    #: weight under expert parallelism — nothing consumes it (the model is
    #: frozen), and the ranks' partials sum to the world-1 gradient.
    partial_gradients: frozenset[str] = frozenset()
    #: The loss is each rank's share (`loss_per_rank`): the ranks'
    #: losses sum to the world-1 loss instead of each equalling it.
    partial_loss: bool = False
    #: How far the summed partial gradients may sit from world 1's, relative
    #: to its largest entry. The one inexact sum in the suite: the router's
    #: weight gradient is ``Σ_tokens ∂L/∂logits ⊗ x``, one matmul over every
    #: token at world 1 and the ranks' token subsets added under expert
    #: parallelism — a re-association of softmax-gradient values no quarter
    #: makes exact. 📐 Measured ``2.4e-7`` against entries of ``29`` (the
    #: experts row) and ``1.9e-9`` against ``0.29`` (the router row), 2026-09-17,
    #: torch 2.9.0, CPU; pinned a hundred times above the larger.
    partial_gradient_band: float = 1e-6

    @property
    def axis(self) -> PlanAxis:
        """The plan axis the row's geometries shard over — the one whose
        rows are applied; rows on the other axis sit inactive."""
        return "expert" if self.geometries[0].expert > 1 else "tensor"

    def style_of(self, parameter: str) -> str | None:
        """The style the row's plan gives ``parameter`` (its own row or its
        module's), ``None`` for a parameter no row names."""
        path = parameter
        while path:
            row = self.plan.style_for(path)
            if row is not None:
                return row.style
            path, _, _ = path.rpartition(".")
        return None


def _rows(**patterns: str) -> ParallelPlan:
    return ParallelPlan(
        {
            pattern: PlanRow(style, "expert" if style in _EXPERT else "tensor")
            for pattern, style in patterns.items()
        }
    )


_EXPERT = frozenset({"grouped_gemm", "moe_tp_experts", "ep_router"})
_SHARDED_OUT = Sharded(-1, "tensor")
_SHARDED_IN = Sharded(-1, "tensor")
_HEADS = Sharded(1, "tensor")
_PACKED_OUT = Sharded(-1, "tensor", slots=2)
_KV_OUT = Sharded(-1, "tensor", repeat=2)
REPLICATED = Replicated()


def _facts_experts(model: nn.Module, size: int) -> Mapping[str, Any]:
    return {"num_experts": model.layers[0].mlp.experts.num_experts}  # type: ignore[index]


def _facts_mixer(model: nn.Module, size: int) -> Mapping[str, Any]:
    return {"num_key_value_groups": model.layers[0].self_attn.num_key_value_groups}  # type: ignore[index]


ROWS: tuple[StyleContract, ...] = (
    StyleContract(
        "colwise",
        _rows(**{"layers.*.mlp.gate_proj": "colwise"}),
        (TENSOR_2, TENSOR_4),
        build_colwise,
        plain_input,
        REPLICATED,
        _SHARDED_OUT,
        functools.partial(whole_by, _SHARDED_OUT),
        functools.partial(loss_over_whole, seed=41),
    ),
    StyleContract(
        "rowwise",
        _rows(**{"layers.*.mlp.down_proj": "rowwise"}),
        (TENSOR_2, TENSOR_4),
        build_rowwise,
        wide_input,
        _SHARDED_IN,
        REPLICATED,
        functools.partial(whole_by, REPLICATED),
        functools.partial(loss_over_whole, seed=42),
    ),
    StyleContract(
        "colwise_gather_output",
        _rows(**{"layers.*.linear_attn.in_proj_qkv": "colwise_gather_output"}),
        (TENSOR_2, TENSOR_4),
        build_gather_output,
        plain_input,
        REPLICATED,
        REPLICATED,
        functools.partial(whole_by, REPLICATED),
        functools.partial(loss_over_whole, seed=43),
    ),
    StyleContract(
        "packed_colwise",
        _rows(**{"layers.*.mlp.gate_up_proj": "packed_colwise"}),
        (TENSOR_2, TENSOR_4),
        build_packed,
        plain_input,
        REPLICATED,
        _PACKED_OUT,
        functools.partial(whole_by, _PACKED_OUT),
        functools.partial(loss_over_whole, seed=44),
    ),
    StyleContract(
        "replicated_with_grad_allreduce",
        _rows(**{"layers.*.self_attn.q_norm": "replicated_with_grad_allreduce"}),
        (TENSOR_2, TENSOR_4),
        build_replicated,
        headed_input,
        _HEADS,
        _HEADS,
        functools.partial(whole_by, _HEADS),
        functools.partial(loss_over_whole, seed=45),
    ),
    StyleContract(
        "kv_replicated",
        ParallelPlan(
            {"layers.*.self_attn.k_proj": PlanRow(KV_REPLICATED, "tensor", repeat=2)}
        ),
        (TENSOR_2,),
        build_kv,
        plain_input,
        REPLICATED,
        _KV_OUT,
        functools.partial(whole_by, _KV_OUT),
        functools.partial(loss_per_rank, seed=46),
        _facts_mixer,
        partial_loss=True,
    ),
    StyleContract(
        "ep_router",
        _rows(**{"layers.*.mlp.gate": "ep_router"}),
        (EXPERT_2,),
        build_router,
        moe_input,
        REPLICATED,
        None,
        whole_router,
        functools.partial(loss_over_whole, seed=47),
        partial_gradients=frozenset({"layers.0.mlp.gate.weight"}),
    ),
    StyleContract(
        "routed_experts",
        _rows(
            **{
                "layers.*.mlp.gate": "ep_router",
                "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                "layers.*.mlp.experts.down_proj": "grouped_gemm",
                "layers.*.mlp.experts": "moe_tp_experts",
                # the shared expert's rows sit on the tensor axis, inactive
                # at ep=2: applied to nothing, the module whole
                "layers.*.mlp.shared_expert.gate_proj": "colwise",
                "layers.*.mlp.shared_expert.up_proj": "colwise",
                "layers.*.mlp.shared_expert.down_proj": "rowwise",
            }
        ),
        (EXPERT_2,),
        build_moe,
        moe_input,
        REPLICATED,
        REPLICATED,
        functools.partial(whole_by, REPLICATED),
        functools.partial(loss_over_whole, seed=48),
        _facts_experts,
        frozenset({"layers.0.mlp.gate.weight"}),
    ),
)

BY_NAME: dict[str, StyleContract] = {row.name: row for row in ROWS}


# --------------------------------------------------------------------------- #
# the program
# --------------------------------------------------------------------------- #


def _plain(tensor: Any) -> Any:
    """A DTensor's local tensor, else the tensor — detached and owned, so
    the result crosses a process boundary."""
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().clone()
    return tensor


def _plain_out(out: Any) -> Any:
    return tuple(_plain(o) for o in out) if isinstance(out, tuple) else _plain(out)


def run_row(
    rank: int,
    c: Collective,
    row: StyleContract,
    tier: Tier,
    geometry: ParallelGeometry,
) -> Results:
    """One rank's program for ``row`` (module docstring)."""
    model = row.build()
    sharding, styles = tier(rank, c, geometry)
    apply_plan(model, row.plan, sharding, styles)
    size = c.size(row.axis)
    x = fragment(row.inputs(), row.input_placement, c).clone()
    with torch.no_grad():
        local = model(x)
        out = row.whole_of(local, c)
    leaf = x.clone().requires_grad_(True)
    graded = model(leaf)
    loss = row.loss(row.whole_of(graded, c), c)
    loss.backward()
    assert leaf.grad is not None
    return {
        "output": _plain_out(out),
        "graded_output": _plain_out(row.whole_of(graded, c)),
        "local_output": _plain_out(local),
        "loss": _plain(loss),
        "input_grad": _plain(leaf.grad),
        "params": {name: _plain(p) for name, p in model.named_parameters()},
        "grads": {name: _plain(p.grad) for name, p in model.named_parameters()},
        "facts": dict(row.facts(model, size)),
    }


def every_row(
    rank: int, c: Collective, tier: Tier, geometry: ParallelGeometry
) -> dict[str, Results]:
    """Every row whose geometries include ``geometry``, in one program."""
    return {
        row.name: run_row(rank, c, row, tier, geometry)
        for row in ROWS
        if geometry in row.geometries
    }


def oracle(row: StyleContract) -> Results:
    """The world-1 run: one rank, ``Solo``, nothing applied."""
    from causalab.neural.shared.parallel.collective import SOLO

    return run_row(0, SOLO, row, fragment_tier, ONE)


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #


class _At:
    """A position on the row's axis, for the expected fragment of a whole."""

    def __init__(self, axis: PlanAxis, rank: int, size: int) -> None:
        self._axis, self._rank, self._size = axis, rank, size

    def rank(self, axis: str) -> int:
        return self._rank if axis == self._axis else 0

    def size(self, axis: str) -> int:
        return self._size if axis == self._axis else 1


def _equal(got: Any, expected: Any, where: tuple[Any, ...]) -> None:
    if isinstance(expected, tuple):
        assert isinstance(got, tuple) and len(got) == len(expected), where
        for index, (g, e) in enumerate(zip(got, expected)):
            _equal(g, e, (*where, index))
        return
    assert isinstance(got, torch.Tensor), (*where, type(got))
    assert got.shape == expected.shape and got.dtype == expected.dtype, (
        *where,
        got.shape,
        expected.shape,
    )
    assert torch.equal(got, expected), (*where, (got - expected).abs().max().item())


def verify(
    row: StyleContract,
    results: Sequence[Results],
    layout: MeshLayout,
    reference: Results,
) -> None:
    """Every rank against the world-1 ``reference`` (module docstring)."""
    geometry_size = None
    for rank, result in enumerate(results):
        group = layout.group_of(rank, row.axis)
        local_rank, size = group.index(rank), len(group)
        geometry_size = size
        at = _At(row.axis, local_rank, size)
        _equal(result["output"], reference["output"], (rank, "output"))
        _equal(result["graded_output"], reference["output"], (rank, "graded output"))
        if not row.partial_loss:
            _equal(result["loss"], reference["loss"], (rank, "loss"))
        if row.output_placement is not None:
            expected_local = fragment(reference["output"], row.output_placement, at)  # type: ignore[arg-type]
            _equal(result["local_output"], expected_local, (rank, "local output"))
        expected_input_grad = fragment(reference["input_grad"], row.input_placement, at)  # type: ignore[arg-type]
        _equal(result["input_grad"], expected_input_grad, (rank, "input grad"))
        assert set(result["params"]) == set(reference["params"]), rank
        for name, whole_param in reference["params"].items():
            partition = _partition(row, name, whole_param.dim(), size)
            _equal(
                result["params"][name],
                partition.local(whole_param, local_rank, size),
                (rank, name),
            )
            whole_grad = reference["grads"][name]
            if name in row.partial_gradients:
                continue
            _equal(
                result["grads"][name],
                partition.local(whole_grad, local_rank, size),
                (rank, name, "grad"),
            )
        assert result["facts"] == _expected_facts(row, reference["facts"], size), rank
    if row.partial_loss:
        for members in layout.groups(row.axis):
            total = sum(results[member]["loss"] for member in members)
            _equal(total, reference["loss"], ("loss summed over", members))
    # a partial gradient sums over the group to the whole one
    for name in row.partial_gradients:
        for members in layout.groups(row.axis):
            total = sum(results[member]["grads"][name] for member in members)
            expected = reference["grads"][name]
            if len(members) == 1:
                _equal(total, expected, (name, "summed over", members))
                continue
            scale = float(expected.abs().max())
            worst = float((total - expected).abs().max()) / scale
            assert worst <= row.partial_gradient_band, (name, members, worst)
    assert geometry_size is not None


def _partition(row: StyleContract, parameter: str, ndim: int, size: int):
    from causalab.neural.engines.pytorch_hooks.styles import WHOLE

    style = row.style_of(parameter)
    if style is None or size == 1:
        return WHOLE
    plan_row = None
    path = parameter
    while path and plan_row is None:
        plan_row = row.plan.style_for(path)
        path, _, _ = path.rpartition(".")
    assert plan_row is not None
    if plan_row.axis != row.axis:
        return WHOLE  # a row on an inactive axis is not applied
    return partition_of(style, ndim, parameter.rsplit(".", 1)[-1])


def _expected_facts(
    row: StyleContract, whole_facts: Mapping[str, Any], size: int
) -> Mapping[str, Any]:
    out: dict[str, Any] = {}
    for key, value in whole_facts.items():
        if key == "num_experts":
            out[key] = value // size
        elif key == "num_key_value_groups":
            repeat = next(iter(row.plan.rows.values())).repeat
            out[key] = value // repeat if size > 1 else value
        else:
            out[key] = value
    return out


def verify_all(
    results: Sequence[dict[str, Results]],
    layout: MeshLayout,
    references: Mapping[str, Results],
) -> None:
    """`verify` for every row `every_row` ran."""
    names = set(results[0])
    for row in ROWS:
        if row.name not in names:
            continue
        verify(
            row, [result[row.name] for result in results], layout, references[row.name]
        )
