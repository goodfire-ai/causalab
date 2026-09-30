"""The shard-on-read plan in one process, for any geometry and rank
(``docs/model_parallelism.md`` §5.3, §10.3 "shard-on-read plan").

``read_plan`` decides each rank's pieces off the placement table and the
styles' partitions — no DTensor, no process group — so the plan of every
rank of a geometry is built here from the tiny fixtures' real checkpoint
headers and the unsharded meta model, and held to the §10.3 property: per
parameter the ranks' byte ranges are disjoint, their union is the whole
tensor, and each is ``1 / world`` of it under an even split; a replicated
parameter (no row, a replicated style, a K/V projection above the KV
heads) is read whole on every rank; a pipeline stage reads its own keys
and nothing else. The numbers the ``gloo`` loads pin
(``test_sharded_load.py``: ``q_proj`` at ``8 · 16 · 4`` bytes of
``16 · 16 · 4`` at ``tp=2``, the fused experts at ``1 / ep``) are pinned
here too, so a plan that moves them fails before any world is spawned.
The mutation — a partition of the wrong dimension — breaks the union.
"""

from __future__ import annotations

from math import prod
from typing import Any, Iterator

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks import shard_read
from causalab.neural.engines.pytorch_hooks.checkpoint import (
    TensorHeader,
    checkpoint_files,
    read_header,
)
from causalab.neural.engines.pytorch_hooks.shard_read import (
    Piece,
    ReadPlan,
    _dense_pieces,
    _stacked_pieces,
    read_plan,
)
from causalab.neural.engines.pytorch_hooks.sharding import Sharding, place_stage
from causalab.neural.engines.pytorch_hooks.styles import Partition
from causalab.neural.engines.pytorch_hooks.weights import wanted_keys
from causalab.protocol.parallel import MeshLayout, ParallelGeometry
from causalab.protocol.registry import ModelInfo, model_info_from_hf_config
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

# pyright: reportPrivateUsage=false

pytestmark = pytest.mark.unit

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

NUM_EXPERTS = 128


class _Sized:
    """A collective of the geometry's sizes, for the sharding's check."""

    def __init__(self, geometry: ParallelGeometry) -> None:
        self._geometry = geometry

    def rank(self, axis: str) -> int:
        return 0

    def size(self, axis: str) -> int:
        return int(getattr(self._geometry, axis, 1))


class Fixture:
    """One tiny model's meta instance, info and checkpoint headers."""

    def __init__(self, key: str) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        self.key = key
        config = AutoConfig.from_pretrained(key)
        self.info: ModelInfo = model_info_from_hf_config(key, config)
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
        files = checkpoint_files(key, "main")
        assert files is not None
        self.headers: dict[str, TensorHeader] = {}
        for path in files:
            self.headers.update(read_header(path))

    def plan_for(self, geometry: ParallelGeometry, rank: int) -> ReadPlan:
        """This rank's plan: the stage's meta instance under a pipeline
        (the keys it wants), the whole one otherwise."""
        import copy

        model = self.model
        sharding = Sharding(geometry, rank, collective=_Sized(geometry))  # type: ignore[arg-type]
        if geometry.pipeline > 1:
            model = copy.deepcopy(self.model)
            place_stage(model, sharding)
        wanted = wanted_keys(model, self.headers)
        assert self.info.parallel_plan is not None
        return read_plan(
            model,
            wanted,
            {k: v for k, v in self.headers.items() if k in wanted},
            plan=self.info.parallel_plan,
            info=self.info,
            sharding=sharding,
        )

    def plans(self, geometry: ParallelGeometry) -> list[ReadPlan]:
        return [self.plan_for(geometry, rank) for rank in range(geometry.world)]


@pytest.fixture(scope="module")
def llama() -> Fixture:
    return Fixture(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe() -> Fixture:
    return Fixture(TINY_QWEN35_MOE)


def _cells(piece: Piece) -> Iterator[tuple[int, ...]]:
    """Every element index the piece covers."""
    import itertools

    axes = [range(lo, hi, step) for lo, hi, step in piece.ranges]
    return itertools.product(*axes)


def _assert_partition(
    plans: list[ReadPlan], headers: dict[str, TensorHeader], geometry: ParallelGeometry
) -> None:
    """§10.3: per key, within each group of the key's axis, the members'
    pieces are disjoint and their union is the whole tensor (the groups of
    the other axis hold replicas); a key whole on one rank is whole on all."""
    keys = set(plans[0].pieces)
    for plan in plans:
        assert set(plan.pieces) == keys
    layout = MeshLayout(geometry)
    for key in keys:
        shape = tuple(headers[key].shape)
        per_rank = [plan.pieces[key] for plan in plans]
        if all(len(pieces) == 1 and pieces[0].is_whole for pieces in per_rank):
            continue
        axis = plans[0].table[plans[0].targets[key]].axis
        assert axis is not None, key
        for members in layout.groups(axis):
            seen: dict[tuple[int, ...], int] = {}
            for rank in members:
                for piece in per_rank[rank]:
                    assert piece.shape == shape, (key, rank)
                    for cell in _cells(piece):
                        assert cell not in seen, (key, "read twice", rank, seen[cell])
                        seen[cell] = rank
            assert len(seen) == headers[key].elements, (key, members, "not covered")


def _sharded(plans: list[ReadPlan]) -> set[str]:
    """The parameters some rank reads a strict part of."""
    out: set[str] = set()
    for plan in plans:
        for key, pieces in plan.pieces.items():
            if not (len(pieces) == 1 and pieces[0].is_whole):
                out.add(plan.targets[key])
    return out


def _report(plan: ReadPlan, fixture: Fixture) -> Any:
    return plan.report({k: fixture.headers[k] for k in plan.pieces})


# --------------------------------------------------------------------------- #
# tensor parallelism on the dense fixture
# --------------------------------------------------------------------------- #


def test_tp2_llama_reads_half_of_every_sharded_parameter_and_the_rest_whole(
    llama: Fixture,
) -> None:
    plans = llama.plans(ParallelGeometry(tensor=2))
    _assert_partition(plans, llama.headers, ParallelGeometry(tensor=2))
    assert _sharded(plans) == {
        f"model.layers.{i}.{name}"
        for i in range(2)
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
    }
    for rank, plan in enumerate(plans):
        report = _report(plan, llama)
        q = "model.layers.0.self_attn.q_proj.weight"
        assert report.bytes_on_disk[q] == 16 * 16 * 4
        assert report.bytes_requested[q] == 8 * 16 * 4
        # colwise: rows; rowwise: columns — this rank's chunk in rank order
        (q_piece,) = plan.pieces[q]
        assert q_piece.spec == (slice(8 * rank, 8 * rank + 8, 1), slice(0, 16, 1))
        (o_piece,) = plan.pieces["model.layers.0.self_attn.o_proj.weight"]
        assert o_piece.spec == (slice(0, 16, 1), slice(8 * rank, 8 * rank + 8, 1))
        for name in report.bytes_on_disk:
            if name in _sharded(plans):
                assert report.bytes_requested[name] * 2 == report.bytes_on_disk[name]
            else:
                assert report.bytes_requested[name] == report.bytes_on_disk[name]
        embed = "model.embed_tokens.weight"
        assert report.bytes_requested[embed] == report.bytes_on_disk[embed]
    # the placement table the plan was built from is the pre-flight's
    assert plans[0].table["model.layers.0.self_attn.q_proj.weight"].axis == "tensor"
    assert plans[0].table["model.embed_tokens.weight"].axis is None


@pytest.mark.parametrize("tensor", (1, 2, 4))
def test_llama_reads_one_over_tp_of_every_sharded_parameter(
    llama: Fixture, tensor: int
) -> None:
    plans = llama.plans(ParallelGeometry(tensor=tensor))
    _assert_partition(plans, llama.headers, ParallelGeometry(tensor=tensor))
    sharded = _sharded(plans)
    assert bool(sharded) == (tensor > 1)
    for plan in plans:
        report = _report(plan, llama)
        for name in sharded:
            assert report.bytes_requested[name] * tensor == report.bytes_on_disk[name]


# --------------------------------------------------------------------------- #
# expert parallelism on the MoE fixture
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ep", (2, 4))
def test_ep_moe_reads_its_experts_only(moe: Fixture, ep: int) -> None:
    plans = moe.plans(ParallelGeometry(expert=ep))
    _assert_partition(plans, moe.headers, ParallelGeometry(expert=ep))
    assert _sharded(plans) == {
        f"model.layers.{i}.mlp.experts.{name}"
        for i in range(4)
        for name in ("gate_up_proj", "down_proj")
    }
    fused = "model.layers.0.mlp.experts.gate_up_proj"
    for rank, plan in enumerate(plans):
        report = _report(plan, moe)
        assert report.bytes_on_disk[fused] == NUM_EXPERTS * 64 * 8 * 2
        assert report.bytes_requested[fused] == report.bytes_on_disk[fused] // ep
        # a per-expert key: whole when this rank owns the expert, else nothing
        local = NUM_EXPERTS // ep
        for expert in range(NUM_EXPERTS):
            key = f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight"
            pieces = plan.pieces[key]
            if rank * local <= expert < (rank + 1) * local:
                assert pieces == (Piece.whole(tuple(moe.headers[key].shape)),), (
                    rank,
                    expert,
                )
            else:
                assert pieces == (), (rank, expert)
        # the router and the shared expert are whole
        gate = "model.layers.0.mlp.gate.weight"
        assert report.bytes_requested[gate] == report.bytes_on_disk[gate]


def test_tp2_ep4_moe_reads_quarters_of_experts_and_halves_of_attention(
    moe: Fixture,
) -> None:
    plans = moe.plans(ParallelGeometry(tensor=2, expert=4))
    _assert_partition(plans, moe.headers, ParallelGeometry(tensor=2, expert=4))
    for rank, plan in enumerate(plans):
        report = _report(plan, moe)
        experts = "model.layers.0.mlp.experts.down_proj"
        assert report.bytes_requested[experts] * 4 == report.bytes_on_disk[experts]
        q = "model.layers.3.self_attn.q_proj.weight"
        assert report.bytes_requested[q] * 2 == report.bytes_on_disk[q]
        (piece,) = plan.pieces["model.language_model.layers.3.self_attn.q_proj.weight"]
        rows = moe.headers[
            "model.language_model.layers.3.self_attn.q_proj.weight"
        ].shape[0]
        half = rows // 2
        assert piece.spec[0] == slice(half * (rank % 2), half * (rank % 2) + half, 1)


def test_tp8_moe_reads_the_kv_projections_whole(moe: Fixture) -> None:
    """§6.6: above the KV heads the K/V rows are ``kv_replicated`` — the
    table's ``shard_limit`` — and read whole on every rank; the query and
    output projections still at ``1 / 8``."""
    plans = moe.plans(ParallelGeometry(tensor=8))
    _assert_partition(plans, moe.headers, ParallelGeometry(tensor=8))
    for plan in plans:
        report = _report(plan, moe)
        for name in ("k_proj", "v_proj"):
            path = f"model.layers.3.self_attn.{name}.weight"
            assert report.bytes_requested[path] == report.bytes_on_disk[path], name
        q = "model.layers.3.self_attn.q_proj.weight"
        assert report.bytes_requested[q] * 8 == report.bytes_on_disk[q]


# --------------------------------------------------------------------------- #
# pipeline placement on the dense fixture
# --------------------------------------------------------------------------- #


def test_pp2_llama_each_stage_reads_its_keys_whole_and_nothing_else(
    llama: Fixture,
) -> None:
    plans = llama.plans(ParallelGeometry(pipeline=2))
    first, last = (_report(plan, llama) for plan in plans)
    everything = {llama.headers and k for k in llama.headers}
    assert set(first.bytes_requested) == {
        n for n in everything if n.startswith(("model.embed_tokens", "model.layers.0."))
    }
    assert set(last.bytes_requested) == {
        n
        for n in everything
        if n.startswith(("model.layers.1.", "model.norm", "lm_head"))
    }
    for report in (first, last):
        assert dict(report.bytes_requested) == dict(report.bytes_on_disk)
    assert set(first.bytes_requested) | set(last.bytes_requested) == everything
    assert not set(first.bytes_requested) & set(last.bytes_requested)


# --------------------------------------------------------------------------- #
# property and mutation
# --------------------------------------------------------------------------- #


@_SETTINGS
@given(tensor=st.sampled_from((1, 2, 4)), pipeline=st.sampled_from((1, 2)))
def test_the_ranks_pieces_partition_every_sharded_tensor(
    llama: Fixture, tensor: int, pipeline: int
) -> None:
    geometry = ParallelGeometry(tensor=tensor, pipeline=pipeline)
    # under a pipeline only the ranks of one stage share a key set: check
    # each stage's ranks as a world of their own
    per_stage = geometry.context * geometry.model
    plans = llama.plans(geometry)
    for stage in range(pipeline):
        stage_plans = plans[stage * per_stage : (stage + 1) * per_stage]
        _assert_partition(stage_plans, llama.headers, ParallelGeometry(tensor=tensor))


def test_a_partition_of_the_wrong_dimension_breaks_the_union(
    llama: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation: every colwise weight cut along its columns instead of
    its rows — the ranks read overlapping columns of the wrong shape."""
    real = shard_read.partition_of

    def wrong(style: str, ndim: int, parameter: str) -> Partition:
        partition = real(style, ndim, parameter)
        if style == "colwise" and ndim == 2:
            return Partition(-1)
        return partition

    monkeypatch.setattr(shard_read, "partition_of", wrong)
    plans = llama.plans(ParallelGeometry(tensor=2))
    q = "model.layers.0.self_attn.q_proj.weight"
    (piece,) = plans[1].pieces[q]
    assert piece.spec[0] == slice(0, 16, 1)  # the whole rows: the mutation took
    # the bytes are still halves — the count cannot see the dimension —
    # but the local shard is not what the style installs
    assert _report(plans[1], llama).bytes_requested[q] == 8 * 16 * 4
    assert piece != Piece.along((16, 16), 0, range(8, 16))


# --------------------------------------------------------------------------- #
# the two piece helpers, over every partition a style may hand them
# --------------------------------------------------------------------------- #


def _covers_once(per_rank: list[tuple[Piece, ...]], shape: tuple[int, ...]) -> None:
    """The ranks' pieces are disjoint and their union is the whole tensor."""
    covered = [
        cell for pieces in per_rank for piece in pieces for cell in _cells(piece)
    ]
    assert len(covered) == len(set(covered)) == prod(shape)


@st.composite
def _dense_cuts(draw: st.DrawFn) -> tuple[tuple[int, ...], Partition, int]:
    """A tensor of one to three dimensions, the whole or a partition of one
    of them (either sign, interleaved or not), and a group size."""
    ndim = draw(st.integers(1, 3))
    shape = tuple(draw(st.integers(1, 6)) for _ in range(ndim))
    size = draw(st.integers(1, 4))
    if draw(st.booleans(), label="whole"):
        return shape, Partition(), size
    dim = draw(st.integers(-ndim, ndim - 1))
    return shape, Partition(dim, draw(st.sampled_from((1, 2)))), size


@st.composite
def _stacked_cuts(draw: st.DrawFn) -> tuple[tuple[int, ...], int, Partition, int]:
    """A per-expert tensor of one to three dimensions, how many are stacked,
    the whole or a partition of one dimension of the **stacked** tensor
    (one more than the per-expert one), and a group size."""
    ndim = draw(st.integers(1, 3))
    shape = tuple(draw(st.integers(1, 6)) for _ in range(ndim))
    stacked = draw(st.integers(1, 8))
    size = draw(st.integers(1, 4))
    if draw(st.booleans(), label="whole"):
        return shape, stacked, Partition(), size
    return shape, stacked, Partition(draw(st.integers(-(ndim + 1), ndim))), size


@_SETTINGS
@given(cut=_dense_cuts())
def test_dense_pieces_are_the_partitions_ranges_of_its_dimension(
    cut: tuple[tuple[int, ...], Partition, int],
) -> None:
    """One piece per range the partition gives this rank — two for an
    interleaved one — every other dimension whole; the whole partition is
    the whole tensor on every rank; the group's pieces cover the tensor once."""
    shape, partition, size = cut
    per_rank = [_dense_pieces(partition, shape, rank, size) for rank in range(size)]
    if partition.whole:
        assert per_rank == [(Piece.whole(shape),)] * size
        return
    dim = partition.axis(len(shape))
    for rank, pieces in enumerate(per_rank):
        spans = partition.ranges(shape[dim], rank, size)
        assert pieces == tuple(Piece.along(shape, dim, span) for span in spans)
    _covers_once(per_rank, shape)


@_SETTINGS
@given(cut=_stacked_cuts())
def test_stacked_pieces_own_whole_experts_on_the_stacked_axis_and_chunk_a_source_dimension_else(
    cut: tuple[tuple[int, ...], int, Partition, int],
) -> None:
    """A partition of the stacked axis: each expert is one rank's, whole,
    and nothing of any other's. A partition of an inner dimension: every
    rank reads, of every expert, its contiguous chunk of the source
    dimension the stacked one maps to (the stacked axis shifted off), and
    the group's chunks cover each expert once. The whole partition is the
    whole tensor of every expert on every rank."""
    shape, stacked, partition, size = cut
    experts = range(stacked)

    def pieces_of(idx: int, rank: int) -> tuple[Piece, ...]:
        return _stacked_pieces(partition, shape, idx, stacked, rank, size)

    if partition.whole:
        for idx in experts:
            assert [pieces_of(idx, rank) for rank in range(size)] == [
                (Piece.whole(shape),)
            ] * size
        return
    dim = partition.axis(len(shape) + 1)
    if dim == 0:
        for idx in experts:
            per_rank = [pieces_of(idx, rank) for rank in range(size)]
            owners = [rank for rank, got in enumerate(per_rank) if got]
            assert len(owners) == 1, (idx, per_rank)
            assert per_rank[owners[0]] == (Piece.whole(shape),)
            assert all(
                got == () for rank, got in enumerate(per_rank) if rank != owners[0]
            )
            (owned,) = Partition(0).ranges(stacked, owners[0], size)
            assert idx in owned
        return
    source = dim - 1
    for idx in experts:
        per_rank = [pieces_of(idx, rank) for rank in range(size)]
        for rank, pieces in enumerate(per_rank):
            (span,) = Partition(source).ranges(shape[source], rank, size)
            assert pieces == (Piece.along(shape, source, span),)
        _covers_once(per_rank, shape)


def test_a_plan_cutting_the_experts_on_an_inner_dimension_reads_each_experts_rows(
    moe: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stacked path's other branch, which no served plan reaches yet
    (``grouped_gemm`` cuts the stacked axis): a plan cutting the fused
    experts as ``colwise`` cuts the stacked ``(experts, out, in)`` — its
    rows — reads, of every per-expert key, this rank's contiguous chunk of
    the matching source dimension, and the group's chunks partition it."""
    real = shard_read.partition_of

    def rows_of_every_expert(style: str, ndim: int, parameter: str) -> Partition:
        if style == "grouped_gemm":
            return real("colwise", ndim, parameter)
        return real(style, ndim, parameter)

    monkeypatch.setattr(shard_read, "partition_of", rows_of_every_expert)
    geometry = ParallelGeometry(expert=2)
    plans = moe.plans(geometry)
    _assert_partition(plans, moe.headers, geometry)
    key = "model.language_model.layers.0.mlp.experts.5.gate_proj.weight"
    shape = tuple(moe.headers[key].shape)
    fused = "model.layers.0.mlp.experts.gate_up_proj"
    for rank, plan in enumerate(plans):
        (span,) = Partition(0).ranges(shape[0], rank, 2)
        assert plan.pieces[key] == (Piece.along(shape, 0, span),), rank
        report = _report(plan, moe)
        assert report.bytes_requested[fused] * 2 == report.bytes_on_disk[fused]
