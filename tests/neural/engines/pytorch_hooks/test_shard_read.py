"""Shard-on-read (``docs/model_parallelism.md`` §5.3), the single-process half.

transformers' ``DtensorShardOperation.shard_tensor(source)`` indexes
``source[slices]`` on the loader's lazy weight; the loader translates that
index into a fastersafetensors ``select`` spec and reads **only that range**.
Held here, with no process group: the index → [`Piece`][causalab.neural.engines.pytorch_hooks.shard_read.Piece] translation for
ints, slices and ``Ellipsis`` over random shapes and the read of exactly the
requested range against the whole tensor's own slice (``torch.equal``), on
both readers; the byte count the reader was asked for; the world-1 path
unchanged (``[...]`` is the whole tensor, as today); and the refusals — an
index the plan did not pre-read, a key this rank reads nothing of — which
never fall back to a whole read. The planned half, over DTensor placeholders,
needs a process group and is ``test_sharded_load.py``.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.io.fastersafetensors.torch import save_file
from causalab.neural.engines.pytorch_hooks.checkpoint import TensorHeader, read_header
from causalab.neural.engines.pytorch_hooks.shard_read import (
    LoadReport,
    Piece,
    ReadPlan,
    ShardReadError,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.neural.engines.pytorch_hooks.weights import (
    FastersafetensorsReader,
    Prefetch,
    ReadGroup,
    SafetensorsReader,
    Shard,
    _LazyWeight,  # pyright: ignore[reportPrivateUsage]
)

_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

CPU = torch.device("cpu")
KEY = "block.weight"


@st.composite
def _shapes(draw: st.DrawFn) -> tuple[int, ...]:
    ndim = draw(st.integers(min_value=1, max_value=3))
    return tuple(draw(st.integers(min_value=1, max_value=6)) for _ in range(ndim))


@st.composite
def _indices(draw: st.DrawFn, shape: tuple[int, ...]) -> Any:
    """A basic index over ``shape``: per dimension an int, a slice (start,
    stop, step ≥ 1, any of them omitted), or the full slice — with, in some
    draws, a trailing run replaced by ``Ellipsis`` or dropped altogether
    (fewer items than dimensions)."""
    items: list[Any] = []
    for size in shape:
        kind = draw(st.sampled_from(("int", "slice", "full")))
        if kind == "int":
            items.append(draw(st.integers(min_value=-size, max_value=size - 1)))
        elif kind == "slice":
            start = draw(
                st.one_of(st.none(), st.integers(min_value=-size, max_value=size))
            )
            stop = draw(
                st.one_of(st.none(), st.integers(min_value=-size, max_value=size))
            )
            step = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=3)))
            items.append(slice(start, stop, step))
        else:
            items.append(slice(None))
    keep = draw(st.integers(min_value=0, max_value=len(items)))
    tail = draw(st.sampled_from(("drop", "ellipsis", "keep")))
    if tail == "drop":
        items = items[:keep]
    elif tail == "ellipsis":
        items = items[:keep] + [Ellipsis]
    if len(items) == 1 and draw(st.booleans()):
        return items[0]
    return tuple(items)


def _write(tmp: Path, tensor: torch.Tensor) -> tuple[Path, dict[str, TensorHeader]]:
    path = tmp / "block.safetensors"
    save_file({KEY: tensor}, path)
    return path, read_header(path)


def _lazy(
    path: Path,
    headers: dict[str, TensorHeader],
    pieces: tuple[Piece, ...] | None,
    reader: Any,
) -> _LazyWeight:
    select = None if pieces is None else {KEY: pieces[0]}
    shard = Shard(path=path, keys=(KEY,), headers=headers, select=select)
    plan = None if pieces is None else {KEY: pieces}
    prefetch = Prefetch([ReadGroup(device=CPU, shards=(shard,))], reader, pieces=plan)
    return _LazyWeight(KEY, headers[KEY], prefetch, pieces=pieces)


# --------------------------------------------------------------------------- #
# Piece — the index as the reader reads it
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_piece_is_the_index_over_the_shape_and_its_result_shape() -> None:
    piece = Piece.of((16, 16), (slice(0, 8), slice(None)))
    assert piece.ranges == ((0, 8, 1), (0, 16, 1))
    assert piece.squeezed == (False, False)
    assert piece.result_shape == (8, 16)
    assert piece.spec == (slice(0, 8, 1), slice(0, 16, 1))
    assert piece.nbytes(4) == 8 * 16 * 4
    assert not piece.is_whole


@pytest.mark.unit
def test_the_whole_piece_is_ellipsis() -> None:
    assert Piece.of((3, 4), Ellipsis) == Piece.whole((3, 4))
    assert Piece.whole((3, 4)).is_whole and Piece.whole((3, 4)).result_shape == (3, 4)
    assert Piece.of((3, 4), (slice(None), slice(None))) == Piece.whole((3, 4))


@pytest.mark.unit
def test_an_int_squeezes_and_a_step_is_kept() -> None:
    piece = Piece.of((5, 6), (2, slice(1, 6, 2)))
    assert piece.ranges == ((2, 3, 1), (1, 6, 2))
    assert piece.squeezed == (True, False)
    assert piece.result_shape == (3,)
    assert piece.spec == (2, slice(1, 6, 2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "index", (None, (slice(None), None), torch.tensor([0, 1]), "x")
)
def test_an_index_that_is_not_ints_slices_or_ellipsis_is_refused(index: Any) -> None:
    """transformers hands the lazy weight basic indexing only; anything else
    is an internal invariant broken, refused by type — never read whole."""
    with pytest.raises(ShardReadError):
        Piece.of((4, 4), index)


@pytest.mark.unit
def test_an_index_out_of_bounds_is_refused_naming_the_shape() -> None:
    with pytest.raises(ShardReadError, match="4"):
        Piece.of((4,), 7)
    with pytest.raises(ShardReadError):
        Piece.of((4,), (slice(None), slice(None)))


# --------------------------------------------------------------------------- #
# reading exactly the requested range
# --------------------------------------------------------------------------- #


@pytest.mark.property
@_HYPOTHESIS_SETTINGS
@given(data=st.data())
def test_a_piece_reads_exactly_the_slice_of_the_whole_tensor(
    data: st.DataObject, tmp_path: Path
) -> None:
    """For every basic index the piece read equals the whole tensor's own
    slice bit for bit, on the planned Rust reader, and the byte count is the
    result's."""
    shape = data.draw(_shapes())
    index = data.draw(_indices(shape))
    whole = torch.arange(float(torch.Size(shape).numel())).reshape(shape)
    path, headers = _write(tmp_path, whole)
    piece = Piece.of(shape, index)
    lazy = _lazy(path, headers, (piece,), FastersafetensorsReader())
    read = lazy[index]
    expected = whole[index]
    assert read.shape == expected.shape, (shape, index)
    assert torch.equal(read, expected), (shape, index)
    assert piece.nbytes(headers[KEY].itemsize) == expected.numel() * 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "index",
    (
        (slice(0, 8), slice(None)),
        (slice(8, 16), slice(None)),
        (slice(None), slice(4, 8)),
        (3, slice(2, 14, 3)),
        Ellipsis,
    ),
)
def test_the_threaded_reader_reads_the_same_piece(index: Any, tmp_path: Path) -> None:
    whole = torch.randn(16, 16)
    path, headers = _write(tmp_path, whole)
    piece = Piece.of((16, 16), index)
    fast = _lazy(path, headers, (piece,), FastersafetensorsReader())[index]
    threaded = _lazy(path, headers, (piece,), SafetensorsReader())[index]
    assert torch.equal(fast, whole[index]) and torch.equal(threaded, whole[index])


@pytest.mark.unit
def test_the_world_one_path_is_the_whole_tensor_as_today(tmp_path: Path) -> None:
    """No plan: ``[...]`` hands out the whole tensor, and an index on it is
    plain tensor indexing of the whole read — the path every existing suite
    runs, unchanged."""
    whole = torch.randn(6, 5)
    path, headers = _write(tmp_path, whole)
    lazy = _lazy(path, headers, None, FastersafetensorsReader())
    assert torch.equal(lazy[...], whole)
    lazy = _lazy(path, headers, None, FastersafetensorsReader())
    assert torch.equal(lazy[2:4, 1], whole[2:4, 1])
    assert lazy.get_shape() == [6, 5] and lazy.get_dtype() == "F32"


# --------------------------------------------------------------------------- #
# the refusals — never a whole read behind the plan's back
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _RecordingCheckpointReader:
    """Every ``read_all`` recorded: which keys, with which selections."""

    inner: FastersafetensorsReader = dataclasses.field(
        default_factory=FastersafetensorsReader
    )
    asked: list[dict[str, Piece | None]] = dataclasses.field(default_factory=list)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def read_all(
        self, shards: Sequence[Shard], device: torch.device
    ) -> dict[str, torch.Tensor]:
        with self._lock:
            self.asked.append(
                {
                    key: (shard.select or {}).get(key)
                    for shard in shards
                    for key in shard.keys
                }
            )
        return self.inner.read_all(shards, device)


@pytest.mark.unit
def test_an_index_the_plan_did_not_pre_read_is_refused_by_name(tmp_path: Path) -> None:
    whole = torch.randn(16, 16)
    path, headers = _write(tmp_path, whole)
    piece = Piece.of((16, 16), (slice(0, 8), slice(None)))
    reader = _RecordingCheckpointReader()
    lazy = _lazy(path, headers, (piece,), reader)
    with pytest.raises(ShardReadError) as err:
        lazy[8:16, :]
    assert KEY in str(err.value)
    # the refusal started no read; the planned half is the one read there is
    assert reader.asked == []
    assert torch.equal(lazy[0:8, :], whole[0:8])
    assert reader.asked == [{KEY: piece}]


@pytest.mark.unit
def test_a_key_this_rank_reads_nothing_of_is_never_asked_for(tmp_path: Path) -> None:
    """An unowned expert's tensor stays in the state dict — transformers
    counts every expert to place the owned ones — but the reader is never
    asked for it, and indexing it is refused rather than read."""
    whole = torch.randn(4, 4)
    path, headers = _write(tmp_path, whole)
    reader = _RecordingCheckpointReader()
    shard = Shard(path=path, keys=(KEY,), headers=headers)
    prefetch = Prefetch(
        [ReadGroup(device=CPU, shards=(shard,))], reader, pieces={KEY: ()}
    )
    lazy = _LazyWeight(KEY, headers[KEY], prefetch, pieces=())
    assert lazy.get_shape() == [4, 4]
    with pytest.raises(ShardReadError, match=KEY):
        lazy[...]
    assert reader.asked == []


@pytest.mark.unit
def test_two_pieces_of_one_key_are_two_reads_each_exactly_its_range(
    tmp_path: Path,
) -> None:
    """A strided placement (``packed_colwise``) asks for two disjoint ranges
    of one tensor; each is a piece, read in its own round, and the two
    together are the two slices."""
    whole = torch.randn(16, 8)
    path, headers = _write(tmp_path, whole)
    pieces = (
        Piece.of((16, 8), (slice(0, 4), slice(None))),
        Piece.of((16, 8), (slice(8, 12), slice(None))),
    )
    reader = _RecordingCheckpointReader()
    shard = Shard(path=path, keys=(KEY,), headers=headers, select={KEY: pieces[0]})
    prefetch = Prefetch(
        [ReadGroup(device=CPU, shards=(shard,))], reader, pieces={KEY: pieces}
    )
    lazy = _LazyWeight(KEY, headers[KEY], prefetch, pieces=pieces)
    assert torch.equal(lazy[8:12, :], whole[8:12])
    assert torch.equal(lazy[0:4, :], whole[0:4])
    assert reader.asked == [{KEY: pieces[0]}, {KEY: pieces[1]}]


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_report_sums_bytes_per_parameter_over_its_checkpoint_keys() -> None:
    """Per-expert checkpoint tensors land on one fused parameter; the report
    is per parameter, so ``1 / world`` is a statement about the parameter."""
    headers = {
        "experts.0.w": TensorHeader(dtype="BF16", shape=(32, 8)),
        "experts.1.w": TensorHeader(dtype="BF16", shape=(32, 8)),
        "q.w": TensorHeader(dtype="F32", shape=(16, 16)),
        "norm.w": TensorHeader(dtype="F32", shape=(16,)),
    }
    plan = ReadPlan(
        pieces={
            "experts.0.w": (Piece.whole((32, 8)),),
            "experts.1.w": (),
            "q.w": (Piece.of((16, 16), (slice(0, 8), slice(None))),),
            "norm.w": (Piece.whole((16,)),),
        },
        targets={
            "experts.0.w": "experts.fused",
            "experts.1.w": "experts.fused",
            "q.w": "q.weight",
            "norm.w": "norm.weight",
        },
    )
    report = plan.report(headers)
    assert isinstance(report, LoadReport)
    assert report.bytes_on_disk == {
        "experts.fused": 2 * 32 * 8 * 2,
        "q.weight": 16 * 16 * 4,
        "norm.weight": 16 * 4,
    }
    assert report.bytes_requested == {
        "experts.fused": 32 * 8 * 2,
        "q.weight": 8 * 16 * 4,
        "norm.weight": 16 * 4,
    }
    assert report.fraction("experts.fused") == 0.5
    assert report.fraction("q.weight") == 0.5
    assert report.fraction("norm.weight") == 1.0
    assert report.requested_total == 32 * 8 * 2 + 8 * 16 * 4 + 16 * 4
    assert report.on_disk_total == 2 * 32 * 8 * 2 + 16 * 16 * 4 + 16 * 4


@pytest.mark.unit
def test_itemsize_follows_the_safetensors_dtype_string() -> None:
    assert TensorHeader("F32", (1,)).itemsize == 4
    assert TensorHeader("BF16", (1,)).itemsize == 2
    assert TensorHeader("F16", (1,)).itemsize == 2
    assert TensorHeader("I64", (1,)).itemsize == 8
    assert TensorHeader("BOOL", (1,)).itemsize == 1
    with pytest.raises(ProtocolError, match="X9"):
        TensorHeader("X9", (1,)).itemsize
