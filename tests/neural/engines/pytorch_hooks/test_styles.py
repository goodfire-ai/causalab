"""The ``Styles`` protocol's data half (``docs/model_parallelism.md`` §10.8):
``Partition`` — the ranges of one dimension a rank holds — and
``partition_of``, the style → partition table.

``property``: the ranges of a partition agree with transformers'
``DtensorShardOperation`` chunk arithmetic (``Shard`` and ``_StridedShard``)
on every extent, rank and group size — the arithmetic the loader's pieces
must reproduce for the load to index what was read; ``local`` of a whole
tensor is the tensor's own values at those ranges, and the ranks' locals
partition the tensor. ``unit``: the table against the eight library styles'
``shard_param`` placements, spelled by hand; ``Group`` off a sharding; the
refusals by name.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from transformers.distributed.sharding_utils import DtensorShardOperation

from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.engines.pytorch_hooks.styles import (
    WHOLE,
    Group,
    Partition,
    StyleError,
    partition_of,
)
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.registry import STYLES

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def _splits(draw: st.DrawFn) -> tuple[int, int, int]:
    """``(extent, size, rank)`` — even and uneven splits, ranks beyond the rows."""
    extent = draw(st.integers(min_value=1, max_value=24))
    size = draw(st.integers(min_value=1, max_value=6))
    rank = draw(st.integers(min_value=0, max_value=size - 1))
    return extent, size, rank


def _library_contiguous(extent: int, size: int, rank: int) -> list[tuple[int, int]]:
    return DtensorShardOperation._compute_contiguous_slice(  # pyright: ignore[reportPrivateUsage]
        None, [(0, extent)], rank, size
    )


def _library_strided(
    extent: int, size: int, rank: int, split: int
) -> list[tuple[int, int]]:
    return DtensorShardOperation._compute_strided_slice(  # pyright: ignore[reportPrivateUsage]
        None, [(0, extent)], rank, size, split
    )


@pytest.mark.property
class TestPartitionProperties:
    @_SETTINGS
    @given(split=_splits())
    def test_a_plain_chunk_is_the_librarys_contiguous_slice(
        self, split: tuple[int, int, int]
    ) -> None:
        extent, size, rank = split
        (mine,) = Partition(0).ranges(extent, rank, size)
        library = _library_contiguous(extent, size, rank)
        # the library drops an empty chunk; the partition spells it as an
        # empty range (the dense read path indexes ``[0:0]``)
        assert [(r.start, r.stop) for r in (mine,) if len(r)] == library

    @_SETTINGS
    @given(split=_splits(), interleave=st.integers(min_value=2, max_value=3))
    def test_an_interleaved_chunk_is_the_librarys_strided_slice(
        self, split: tuple[int, int, int], interleave: int
    ) -> None:
        extent, size, rank = split
        mine = Partition(0, interleave).ranges(extent, rank, size)
        assert [(r.start, r.stop) for r in mine] == _library_strided(
            extent, size, rank, interleave
        )

    @_SETTINGS
    @given(split=_splits(), interleave=st.integers(min_value=1, max_value=2))
    def test_the_locals_partition_the_whole(
        self, split: tuple[int, int, int], interleave: int
    ) -> None:
        extent, size, _ = split
        partition = Partition(1, interleave)
        whole = torch.arange(3 * extent, dtype=torch.float32).reshape(3, extent)
        pieces = [partition.local(whole, rank, size) for rank in range(size)]
        for rank, piece in enumerate(pieces):
            expected = [
                whole[:, r.start : r.stop] for r in partition.ranges(extent, rank, size)
            ]
            # a rank beyond the rows holds an empty chunk of the right shape
            assert torch.equal(
                piece, torch.cat(expected, 1) if expected else whole[:, :0]
            ), rank
            assert piece.data_ptr() != whole.data_ptr() or piece.numel() == 0
        held = sorted(int(v) for piece in pieces for v in piece[0].tolist())
        assert held == list(range(extent))


@pytest.mark.unit
class TestPartition:
    def test_the_whole_partition_is_the_tensor(self) -> None:
        whole = torch.arange(6.0).reshape(2, 3)
        assert WHOLE.ranges(3, 1, 2) == (range(0, 3),)
        local = WHOLE.local(whole, 1, 2)
        assert torch.equal(local, whole) and local.data_ptr() != whole.data_ptr()
        assert WHOLE.whole and not Partition(0).whole

    def test_a_negative_dimension_is_the_last(self) -> None:
        whole = torch.arange(8.0).reshape(2, 4)
        assert Partition(-1).axis(2) == 1
        assert torch.equal(Partition(-1).local(whole, 1, 2), whole[:, 2:])

    def test_refusals_name_what_did_not_fit(self) -> None:
        with pytest.raises(StyleError, match="interleave"):
            Partition(0, 0)
        with pytest.raises(StyleError, match="nothing to interleave"):
            Partition(None, 2)
        with pytest.raises(StyleError, match="outside a 2-D"):
            Partition(2).axis(2)
        with pytest.raises(StyleError, match="names no dimension"):
            WHOLE.axis(2)


#: The library's ``shard_param`` placements, by hand: ``(style, ndim, leaf)``
#: → partition. ``Shard(ndim - 2)`` for the colwise family, ``Shard(-1)`` /
#: ``Replicate`` for rowwise's weight / bias, ``_StridedShard(ndim - 2, 2)``
#: for the packed projection (``Shard(-1)`` on its bias), ``Shard(0)`` for
#: the experts, nothing for the rest.
TABLE: tuple[tuple[str, int, str, Partition], ...] = (
    ("colwise", 2, "weight", Partition(0)),
    ("colwise", 1, "bias", Partition(-1)),
    ("colwise", 3, "gate_up_proj", Partition(1)),
    ("colwise_gather_output", 2, "weight", Partition(0)),
    ("rowwise", 2, "weight", Partition(-1)),
    ("rowwise", 3, "down_proj", Partition(-1)),
    ("rowwise", 1, "bias", WHOLE),
    ("packed_colwise", 2, "weight", Partition(0, 2)),
    ("packed_colwise", 3, "gate_up_proj", Partition(1, 2)),
    ("packed_colwise", 1, "bias", Partition(-1)),
    ("grouped_gemm", 3, "gate_up_proj", Partition(0)),
    ("replicated_with_grad_allreduce", 1, "weight", WHOLE),
    ("ep_router", 2, "weight", WHOLE),
    ("moe_tp_experts", 3, "gate_up_proj", WHOLE),
    ("kv_replicated", 2, "weight", WHOLE),
)


@pytest.mark.unit
class TestPartitionOf:
    @pytest.mark.parametrize("style,ndim,leaf,expected", TABLE)
    def test_the_table(
        self, style: str, ndim: int, leaf: str, expected: Partition
    ) -> None:
        assert partition_of(style, ndim, leaf) == expected

    def test_every_served_style_has_a_rule(self) -> None:
        for style in STYLES:
            partition_of(style, 2, "weight")

    def test_an_unknown_style_is_refused_by_name(self) -> None:
        with pytest.raises(StyleError, match="embedding_rowwise"):
            partition_of("embedding_rowwise", 2, "weight")

    def test_a_scalar_cannot_be_partitioned(self) -> None:
        with pytest.raises(StyleError, match="0-D"):
            partition_of("colwise", 0, "weight")


class _Sized:
    def __init__(self, **sizes: int) -> None:
        self._sizes = sizes

    def rank(self, axis: str) -> int:
        return 0

    def size(self, axis: str) -> int:
        return self._sizes.get(axis, 1)


@pytest.mark.unit
class TestGroup:
    def test_off_a_sharding(self) -> None:
        geometry = ParallelGeometry(tensor=2, expert=4)
        collective = _Sized(tensor=2, expert=4, model=4)
        sharding = Sharding(geometry, 3, collective=collective)  # type: ignore[arg-type]
        assert sharding.group("tensor") == Group("tensor", 1, 2)
        assert sharding.group("expert") == Group("expert", 3, 4)

    def test_refusals_name_what_did_not_fit(self) -> None:
        with pytest.raises(StyleError, match="axis 'data'"):
            Group("data", 0, 2)  # type: ignore[arg-type]
        with pytest.raises(StyleError, match="outside the tensor group of 2"):
            Group("tensor", 2, 2)
        with pytest.raises(StyleError, match="at least one rank"):
            Group("tensor", 0, 0)

    def test_a_sharding_whose_collective_disagrees_is_refused(self) -> None:
        from causalab.protocol.rules.errors import ProtocolError

        with pytest.raises(ProtocolError, match="tensor group has 1 rank"):
            Sharding(ParallelGeometry(tensor=2), 0)
