"""The context frame and its three crossings (``docs/model_parallelism.md``
§6.4, §8.4, §10.3–10.4), under the ``SimulatedWorld`` at drawn schedules:

- the local causal mask is the whole frame's eager mask at the chunk's rows,
  bit for bit against transformers' own ``create_causal_mask`` — for every
  chunk of every uneven frame drawn;
- ``SequenceFrame.gather`` / ``fragment`` round-trip along any axis, uneven
  chunks and the flat token axis included, and ``whole`` is identical on
  every rank;
- the placement compositions round-trip: ``SequenceSharded ∘ Sharded``
  (``cp=2, tp=2``), ``StageLocal ∘ SequenceSharded`` (``pp=2, cp=2``) and
  ``SequenceSharded ∘ ExpertLocal`` with the routing table chunked alongside
  (``cp=2, ep=2``) — ``fragment(whole(x)) == x`` and ``whole(fragment(g)) ==
  g`` bit for bit over drawn tensors and schedule tapes;
- ``handoff_state`` runs a recurrence chunk after chunk and equals the
  unsplit one; a rank that skips its send is a typed refusal naming the
  waiting rank, never a hang of the test;
- ``chunked_conv`` over the chunks equals the whole frame's conv with the
  library's own torch fallback within an fp32 band (every output sees the
  same four inputs, but ``F.conv1d`` picks its algorithm by the input's
  length: measured ``2.4e-7`` on CPU, pinned ``1e-6``), and a chunk shorter
  than the conv's history is refused by name.

The mutations: a gather that drops the padding cut lands the wrong
positions on an uneven frame; a mask whose rows are one position late
differs from transformers' at every chunk.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings, strategies as st

from causalab.neural.shared.kernels import torch_implementation
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.context import (
    SequenceError,
    SequenceFrame,
    activate,
    chunked_conv,
    current,
    handoff_state,
    local_causal_mask,
)
from causalab.neural.shared.parallel.fragments import (
    Fragments,
    PlacementError,
    expert_slot_mask,
    fragment,
    whole,
)
from causalab.neural.shared.parallel.placement import (
    ExpertLocal,
    SequenceSharded,
    Sharded,
    StageLocal,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import sequence_chunks
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Abandoned,
    Hang,
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _left_padded(rows: int, padded_len: int, seed: int) -> torch.Tensor:
    """A left-padded 2-D mask with at least one real token per row."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, padded_len + 1, (rows,), generator=generator)
    mask = torch.zeros(rows, padded_len, dtype=torch.long)
    for row, length in enumerate(lengths.tolist()):
        mask[row, padded_len - length :] = 1
    return mask


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #


def _transformers_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """What the model builds for the whole frame under eager attention."""
    from transformers import LlamaConfig
    from transformers.masking_utils import create_causal_mask

    config = LlamaConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=16,
    )
    config._attn_implementation = "eager"
    rows, padded_len = mask.shape
    embeds = torch.zeros(rows, padded_len, 8, dtype=dtype)
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    built = create_causal_mask(
        config=config,
        inputs_embeds=embeds,
        attention_mask=mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    assert isinstance(built, torch.Tensor)
    return built


@pytest.mark.property
class TestLocalCausalMask:
    @_SETTINGS
    @given(
        rows=st.integers(1, 3),
        padded_len=st.integers(1, 12),
        context=st.integers(1, 4),
        seed=st.integers(0, 2**16),
        dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    )
    def test_the_local_rows_are_the_whole_masks_rows_for_every_chunk(
        self, rows: int, padded_len: int, context: int, seed: int, dtype: torch.dtype
    ) -> None:
        context = min(context, padded_len)
        mask = _left_padded(rows, padded_len, seed)
        expected = _transformers_mask(mask, dtype)
        assert expected.shape == (rows, 1, padded_len, padded_len)
        locals_ = [
            local_causal_mask(mask, chunk, dtype)
            for chunk in sequence_chunks(padded_len, context)
        ]
        assert torch.equal(torch.cat(locals_, dim=2), expected)
        for chunk, local in zip(sequence_chunks(padded_len, context), locals_):
            assert torch.equal(local, expected[:, :, chunk.start : chunk.stop, :])

    def test_rows_one_position_late_differ_from_the_whole_mask(self) -> None:
        """The smoke tier's mutation, at the source: shifting the query rows
        by one moves the causal boundary on every chunk."""
        mask = _left_padded(2, 9, 3)
        expected = _transformers_mask(mask, torch.float32)
        for chunk in sequence_chunks(9, 2):
            late = range(chunk.start + 1, min(chunk.stop + 1, 9))
            shifted = local_causal_mask(mask, late, torch.float32)
            rows = expected[:, :, chunk.start : chunk.start + len(late), :]
            assert not torch.equal(shifted, rows)


@pytest.mark.unit
class TestMaskRefusals:
    def test_a_chunk_outside_the_frame_and_a_non_2d_mask_are_refused(self) -> None:
        with pytest.raises(SequenceError, match="outside a frame"):
            local_causal_mask(
                torch.ones(1, 4, dtype=torch.long), range(3, 6), torch.float32
            )
        with pytest.raises(SequenceError, match=r"\(batch, padded_len\)"):
            local_causal_mask(
                torch.ones(4, dtype=torch.long), range(0, 2), torch.float32
            )


# --------------------------------------------------------------------------- #
# the frame: gather and fragment
# --------------------------------------------------------------------------- #


def _frame_program(
    global_tensor: torch.Tensor, mask: torch.Tensor, axis: int, *, flat: bool
) -> Callable[[int, Collective], dict[str, torch.Tensor]]:
    """Each rank holds its chunk of ``global_tensor`` along ``axis`` (a flat
    token axis unfolded by the frame's rows first) and gathers it back."""

    def program(rank: int, c: Collective) -> dict[str, torch.Tensor]:
        frame = SequenceFrame(c, mask)
        if flat:
            unfolded = frame.unflatten(global_tensor, axis, local=False)
            local = frame.fragment(unfolded, axis + 1).flatten(axis, axis + 1)
            gathered = frame.gather(
                frame.unflatten(local, axis, local=True), axis + 1
            ).flatten(axis, axis + 1)
        else:
            local = frame.fragment(global_tensor, axis)
            gathered = frame.gather(local, axis)
        return {
            "local": local,
            "gathered": gathered,
            "chunk_len": torch.tensor(len(frame.chunk)),
        }

    return program


@pytest.mark.property
class TestFrameRoundTrip:
    @_SETTINGS
    @given(
        padded_len=st.integers(2, 11),
        context=st.sampled_from([2, 3, 4]),
        rows=st.integers(1, 3),
        seed=st.integers(0, 2**16),
        schedule=ps.schedules(),
    )
    def test_gather_of_fragment_is_the_whole_on_every_rank(
        self,
        padded_len: int,
        context: int,
        rows: int,
        seed: int,
        schedule: Schedule,
    ) -> None:
        context = min(context, padded_len)
        mask = _left_padded(rows, padded_len, seed)
        chunks = sequence_chunks(padded_len, context)
        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        for axis, shape in ((1, (rows, padded_len, 4)), (2, (rows, 3, padded_len))):
            g = _seeded(shape, seed)
            results = world.run(_frame_program(g, mask, axis, flat=False))
            for rank, got in enumerate(results):
                assert torch.equal(got["gathered"], g), f"rank {rank}"
                assert torch.equal(
                    got["local"], g.narrow(axis, chunks[rank].start, len(chunks[rank]))
                )
                assert int(got["chunk_len"]) == len(chunks[rank])

    @_SETTINGS
    @given(
        padded_len=st.integers(2, 9),
        context=st.sampled_from([2, 3]),
        rows=st.integers(1, 3),
        seed=st.integers(0, 2**16),
        schedule=ps.schedules(),
    )
    def test_a_flat_token_axis_chunks_by_position_within_each_row(
        self,
        padded_len: int,
        context: int,
        rows: int,
        seed: int,
        schedule: Schedule,
    ) -> None:
        """The experts' token-major view: ``(rows · positions, feature)`` is
        chunked by position inside every row, never by leading tokens."""
        context = min(context, padded_len)
        mask = _left_padded(rows, padded_len, seed)
        g = _seeded((rows * padded_len, 4), seed)
        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        results = world.run(_frame_program(g, mask, 0, flat=True))
        by_row = g.view(rows, padded_len, 4)
        for rank, got in enumerate(results):
            chunk = sequence_chunks(padded_len, context)[rank]
            assert torch.equal(got["gathered"], g)
            assert torch.equal(
                got["local"], by_row[:, chunk.start : chunk.stop].reshape(-1, 4)
            )

    def test_dropping_the_padding_cut_lands_wrong_positions_on_an_uneven_frame(
        self,
    ) -> None:
        """The mutation: an uneven frame's chunks are padded to the widest
        for the all-gather; concatenating the padded pieces as they come
        (no cut) is the wrong whole."""
        padded_len = 5  # cp=2: chunks of 2 and 3
        mask = torch.ones(1, padded_len, dtype=torch.long)
        g = _seeded((1, padded_len, 2), 0)

        def mutated(rank: int, c: Collective) -> torch.Tensor:
            frame = SequenceFrame(c, mask)
            local = frame.fragment(g, 1)
            widest = max(len(chunk) for chunk in frame.chunks)
            pad = torch.zeros(1, widest - local.shape[1], 2)
            return c.all_gather(torch.cat([local, pad], dim=1), 1, "context")

        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=0)
        for got in world.run(mutated):
            assert got.shape[1] == 2 * 3
            assert not torch.equal(got[:, :padded_len], g)


@pytest.mark.unit
class TestFrameRefusals:
    def _two(self) -> SimulatedWorld:
        return SimulatedWorld(groups_for(2, context=2), world=2, schedule=0)

    def test_a_frame_shorter_than_the_group_is_refused_by_name(self) -> None:
        def program(rank: int, c: Collective) -> None:
            SequenceFrame(c, torch.ones(1, 1, dtype=torch.long))

        with pytest.raises(RankFailed) as err:
            self._two().run(program)
        assert isinstance(err.value.__cause__, ProtocolError)
        assert "--parallel.context" in str(err.value.__cause__)

    def test_an_extent_that_is_not_the_chunk_or_the_frame_is_refused(self) -> None:
        def program(rank: int, c: Collective) -> None:
            frame = SequenceFrame(c, torch.ones(1, 4, dtype=torch.long))
            with pytest.raises(SequenceError, match="not this rank's chunk"):
                frame.gather(torch.zeros(1, 3, 2), 1)
            with pytest.raises(SequenceError, match="not the frame's"):
                frame.fragment(torch.zeros(1, 3, 2), 1)
            with pytest.raises(SequenceError, match="rows"):
                frame.unflatten(torch.zeros(5, 2), 0, local=False)
            with pytest.raises(SequenceError, match="outside"):
                frame.fragment(torch.zeros(1, 4), 2)

        self._two().run(program)

    def test_a_conv_wider_than_the_chunks_is_refused_by_name(self) -> None:
        def program(rank: int, c: Collective) -> None:
            frame = SequenceFrame(c, torch.ones(1, 5, dtype=torch.long))  # chunks 2, 3
            frame.check_conv_width(3)
            with pytest.raises(ProtocolError) as err:
                frame.check_conv_width(4)
            assert "--parallel.context" in str(err.value) and "conv width 4" in str(
                err.value
            )

        self._two().run(program)

    def test_a_sequence_placement_at_world_one_is_the_identity(self) -> None:
        frame = SequenceFrame(SOLO, torch.ones(2, 3, dtype=torch.long))
        x = torch.arange(6.0).view(2, 3)
        assert frame.gather(x, 1) is x
        assert torch.equal(frame.fragment(x, 1), x)
        assert frame.chunk == range(0, 3)

    def test_activate_binds_the_frame_for_the_duration(self) -> None:
        frame = SequenceFrame(SOLO, torch.ones(1, 2, dtype=torch.long))
        assert current() is None
        with activate(frame):
            assert current() is frame
            with activate(None):
                assert current() is None
            assert current() is frame
        assert current() is None


# --------------------------------------------------------------------------- #
# the compositions (§4, "compose, do not replace")
# --------------------------------------------------------------------------- #

ROWS, FEATURE, TOP_K, EXPERTS = 2, 8, 2, 4


def _halves(tensor: torch.Tensor, axis: int, rank: int, size: int) -> torch.Tensor:
    per = tensor.shape[axis] // size
    return tensor.narrow(axis, rank * per, per)


def _composition_program(
    padded_len: int, seed: int
) -> Callable[[int, Collective], dict[str, Any]]:
    """Three compositions on one world of four ranks — ``cp=2 × tp=2``,
    ``pp=2 × cp=2``, ``cp=2 × ep=2`` — each rank holding what the composed
    placement says it holds, spelled by hand, and making it whole."""
    mask = _left_padded(ROWS, padded_len, seed)
    dense = _seeded((ROWS, padded_len, FEATURE), seed)
    routing = torch.randint(
        0,
        EXPERTS,
        (ROWS * padded_len, TOP_K),
        generator=torch.Generator().manual_seed(seed + 1),
    )
    experts_view = _seeded((ROWS * padded_len, TOP_K * 3), seed + 2)

    def program(rank: int, c: Collective) -> dict[str, Any]:
        frame = SequenceFrame(c, mask)
        f = Fragments(c)
        chunk = frame.chunk
        out: dict[str, Any] = {}
        # cp × tp: the position chunk of this rank's feature half
        if c.size("tensor") > 1:
            placement = SequenceSharded(1, "context", inner=Sharded(-1, "tensor"))
            t, tp = c.rank("tensor"), c.size("tensor")
            local = _halves(dense[:, chunk.start : chunk.stop], -1, t, tp)
            out["tp_whole"] = f.whole(local, placement, frame=frame)
            out["tp_fragment"] = f.fragment(dense, placement, frame=frame)
            out["tp_local"] = local
        # pp × cp: the owner stage's ranks hold their chunk, the others nothing
        if c.size("pipeline") > 1:
            placement = StageLocal(1, "pipeline", inner=SequenceSharded(1, "context"))
            owner = c.rank("pipeline") == 1
            # a non-owner holds the typed empty tensor of the *whole*'s
            # trailing extents (fragments.py, "Non-owners under StageLocal")
            local = (
                dense[:, chunk.start : chunk.stop]
                if owner
                else dense.new_empty((0, padded_len, FEATURE))
            )
            out["pp_whole"] = f.whole(local, placement, frame=frame)
            out["pp_fragment"] = f.fragment(dense, placement, frame=frame)
            out["pp_local"] = local
        # cp × ep: the token-major view, chunked by position, owned slots kept
        if c.size("expert") > 1:
            placement = SequenceSharded(
                0, "context", inner=ExpertLocal("expert"), flat=True
            )
            e, ep = c.rank("expert"), c.size("expert")
            by_row = experts_view.view(ROWS, padded_len, -1)[
                :, chunk.start : chunk.stop
            ]
            table_rows = routing.view(ROWS, padded_len, TOP_K)[
                :, chunk.start : chunk.stop
            ]
            local_view = by_row.reshape(-1, TOP_K * 3)
            local_table = table_rows.reshape(-1, TOP_K)
            owned = expert_slot_mask(local_table, EXPERTS, e, ep)
            keep = owned.unsqueeze(-1).expand(-1, TOP_K, 3).reshape(local_view.shape)
            local = torch.where(keep, local_view, torch.zeros_like(local_view))
            out["ep_whole"] = f.whole(local, placement, frame=frame)
            out["ep_fragment"] = f.fragment(
                experts_view,
                placement,
                routing=routing,
                num_experts=EXPERTS,
                frame=frame,
            )
            out["ep_local"] = local
        return out

    program.dense = dense  # type: ignore[attr-defined]
    program.experts_view = experts_view  # type: ignore[attr-defined]
    return program


@pytest.mark.property
class TestCompositions:
    @pytest.mark.parametrize("padded_len", [4, 5, 7])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @example(schedule=[], seed=0)
    @_SETTINGS
    def test_context_and_tensor_compose_and_round_trip(
        self, padded_len: int, schedule: Schedule, seed: int
    ) -> None:
        program = _composition_program(padded_len, seed)
        world = SimulatedWorld(
            groups_for(4, context=2, tensor=2), world=4, schedule=schedule
        )
        for rank, got in enumerate(world.run(program)):
            assert torch.equal(got["tp_whole"], program.dense), f"rank {rank}"
            assert torch.equal(got["tp_fragment"], got["tp_local"]), f"rank {rank}"

    @pytest.mark.parametrize("padded_len", [4, 5])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_a_stage_wraps_the_sequence_chunk(
        self, padded_len: int, schedule: Schedule, seed: int
    ) -> None:
        program = _composition_program(padded_len, seed)
        world = SimulatedWorld(
            groups_for(4, pipeline=2, context=2), world=4, schedule=schedule
        )
        for rank, got in enumerate(world.run(program)):
            assert torch.equal(got["pp_whole"], program.dense), f"rank {rank}"
            assert torch.equal(got["pp_fragment"], got["pp_local"]), f"rank {rank}"

    @pytest.mark.parametrize("padded_len", [4, 5])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_experts_view_chunks_by_position_with_its_routing(
        self, padded_len: int, schedule: Schedule, seed: int
    ) -> None:
        program = _composition_program(padded_len, seed)
        world = SimulatedWorld(
            groups_for(4, context=2, expert=2), world=4, schedule=schedule
        )
        for rank, got in enumerate(world.run(program)):
            assert torch.equal(got["ep_whole"], program.experts_view), f"rank {rank}"
            assert torch.equal(got["ep_fragment"], got["ep_local"]), f"rank {rank}"

    @_SETTINGS
    @given(
        kv_heads=st.integers(1, 2),
        repeat=st.integers(2, 3),
        head_dim=st.integers(1, 3),
        padded_len=st.integers(4, 7),
        seed=st.integers(0, 2**31 - 1),
        schedule=ps.schedules(),
    )
    def test_a_repeated_shard_inside_the_sequence_chunk_round_trips(
        self,
        kv_heads: int,
        repeat: int,
        head_dim: int,
        padded_len: int,
        seed: int,
        schedule: Schedule,
    ) -> None:
        """``SequenceSharded ∘ Sharded(repeat=…)`` (``cp=2 × tp=kv_heads ·
        repeat``, §6.6 under §8.4): rank ``t`` of the tensor group holds KV
        head ``t // repeat`` of its position chunk. ``whole`` is the model's
        tensor on every rank — the inner gather drops the copies, the frame's
        gather pads and cuts the uneven chunks — and ``fragment`` of it is the
        rank's local again, bit for bit."""
        tp = kv_heads * repeat
        mask = _left_padded(ROWS, padded_len, seed)
        dense = _seeded((ROWS, padded_len, kv_heads * head_dim), seed)
        placement = SequenceSharded(
            1, "context", inner=Sharded(-1, "tensor", repeat=repeat)
        )

        def program(rank: int, c: Collective) -> dict[str, Any]:
            frame = SequenceFrame(c, mask)
            chunk = frame.chunk
            head = c.rank("tensor") // repeat
            local = dense[:, chunk.start : chunk.stop].narrow(
                -1, head * head_dim, head_dim
            )
            f = Fragments(c)
            return {
                "whole": f.whole(local, placement, frame=frame),
                "fragment": f.fragment(dense, placement, frame=frame),
                "local": local,
            }

        world = SimulatedWorld(
            groups_for(2 * tp, context=2, tensor=tp), world=2 * tp, schedule=schedule
        )
        for rank, got in enumerate(world.run(program)):
            assert torch.equal(got["whole"], dense), f"rank {rank}"
            assert torch.equal(got["fragment"], got["local"]), f"rank {rank}"

    @_SETTINGS
    @given(
        seed=st.integers(0, 999),
        padded_len=st.integers(4, 9),
        schedule=ps.schedules(),
    )
    def test_the_result_does_not_depend_on_the_schedule(
        self, seed: int, padded_len: int, schedule: Schedule
    ) -> None:
        program = _composition_program(padded_len, seed)
        world = SimulatedWorld(
            groups_for(4, context=2, tensor=2), world=4, schedule=schedule
        )
        for got in world.run(program):
            assert torch.equal(got["tp_whole"], program.dense)


@pytest.mark.unit
class TestFramelessSequencePlacement:
    """Without a frame the chunks are equal — the plain rule on the position
    axis — and a flat axis is refused by name (``fragments.py``)."""

    def test_equal_chunks_round_trip_without_a_frame(self) -> None:
        g = _seeded((1, 4, 2), 0)

        def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
            local = g[:, 2 * rank : 2 * rank + 2]
            return (
                whole(local, SequenceSharded(1), c),
                fragment(g, SequenceSharded(1), c),
            )

        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=0)
        for rank, (w, fr) in enumerate(world.run(program)):
            assert torch.equal(w, g) and torch.equal(fr, g[:, 2 * rank : 2 * rank + 2])

    def test_a_flat_axis_without_a_frame_is_refused(self) -> None:
        def program(rank: int, c: Collective) -> None:
            with pytest.raises(PlacementError, match="SequenceFrame"):
                whole(torch.zeros(4, 2), SequenceSharded(0, flat=True), c)

        SimulatedWorld(groups_for(2, context=2), world=2, schedule=0).run(program)


# --------------------------------------------------------------------------- #
# the DeltaNet handoffs (§6.4)
# --------------------------------------------------------------------------- #

B, H, DK, DV = 2, 3, 4, 4


def _cumsum_program(
    x: torch.Tensor, mask: torch.Tensor, *, skip_send: bool = False
) -> Callable[[int, Collective], torch.Tensor]:
    """A toy recurrence — the running sum along positions, state ``(B, H, DK,
    DV)`` — over the chunks, each rank after the one below."""

    def program(rank: int, c: Collective) -> torch.Tensor:
        frame = SequenceFrame(c, mask)
        local = x[:, frame.chunk.start : frame.chunk.stop]

        def run(initial: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            start = torch.zeros(B, H, DK, DV) if initial is None else initial
            out = start.unsqueeze(1) + local.cumsum(dim=1)
            return out, out[:, -1]

        if skip_send and rank == 0:
            out, _ = run(None)  # the mutation: no send to rank 1
            return out
        out, _ = handoff_state(
            frame, run, shape=(B, H, DK, DV), dtype=torch.float32, device=x.device
        )
        return frame.gather(out, 1)

    return program


@pytest.mark.property
class TestHandoff:
    @pytest.mark.parametrize("padded_len", [4, 5, 7])
    @pytest.mark.parametrize("context", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_recurrence_over_chunks_equals_the_unsplit_one(
        self, padded_len: int, context: int, schedule: Schedule, seed: int
    ) -> None:
        x = _seeded((B, padded_len, H, DK, DV), seed)
        mask = torch.ones(B, padded_len, dtype=torch.long)
        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        expected = x.cumsum(dim=1)
        for got in world.run(_cumsum_program(x, mask)):
            # the running sum re-associates at the chunk boundary: fp32 band
            assert torch.allclose(got, expected, atol=1e-5, rtol=0)

    def test_a_rank_that_skips_its_send_is_refused_by_name(self) -> None:
        x = _seeded((B, 4, H, DK, DV), 0)
        mask = torch.ones(B, 4, dtype=torch.long)
        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=0)
        with pytest.raises((Hang, Abandoned)) as err:
            world.run(_cumsum_program(x, mask, skip_send=True))
        assert "rank 1" in str(err.value) and "recv" in str(err.value)


#: The fp32 band of the chunked conv against the whole frame's (module docstring).
CONV_BAND = 1e-6


def _conv() -> Callable[..., torch.Tensor]:
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling

    return torch_implementation(modeling.causal_conv1d_fn)


@pytest.mark.property
class TestChunkedConv:
    @pytest.mark.parametrize("padded_len", [9, 10, 12])  # every chunk holds the history
    @pytest.mark.parametrize("context", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_conv_over_chunks_equals_the_whole_frames(
        self, padded_len: int, context: int, schedule: Schedule, seed: int
    ) -> None:
        conv = _conv()
        channels, kernel = 6, 4
        x = _seeded((B, channels, padded_len), seed)
        weight = _seeded((channels, kernel), seed + 1)
        bias = _seeded((channels,), seed + 2)
        mask = torch.ones(B, padded_len, dtype=torch.long)
        expected = conv(x, weight, bias, activation="silu")

        def program(rank: int, c: Collective) -> torch.Tensor:
            frame = SequenceFrame(c, mask)
            local = x[..., frame.chunk.start : frame.chunk.stop]
            out = chunked_conv(frame, conv, local, weight, bias, activation="silu")
            assert out.shape == local.shape
            return frame.gather(out, 2)

        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=schedule
        )
        for got in world.run(program):
            # every output saw exactly the whole frame's inputs — a per-channel
            # dot product of the same four terms — but F.conv1d chooses its
            # algorithm by the input's length (module docstring): the band
            assert (got - expected).abs().max().item() <= CONV_BAND

    def test_a_chunk_shorter_than_the_history_is_refused_by_name(self) -> None:
        conv = _conv()
        weight = _seeded((2, 4), 0)

        def program(rank: int, c: Collective) -> None:
            frame = SequenceFrame(c, torch.ones(1, 4, dtype=torch.long))  # chunks of 2
            local = torch.zeros(1, 2, len(frame.chunk))
            with pytest.raises(ProtocolError) as err:
                chunked_conv(frame, conv, local, weight, None, activation=None)
            assert "--parallel.context" in str(err.value)

        SimulatedWorld(groups_for(2, context=2), world=2, schedule=0).run(program)

    def test_a_group_of_one_runs_the_conv_as_it_is(self) -> None:
        conv = _conv()
        x, weight = _seeded((1, 2, 5), 0), _seeded((2, 4), 1)
        frame = SequenceFrame(SOLO, torch.ones(1, 5, dtype=torch.long))
        assert torch.equal(
            chunked_conv(frame, conv, x, weight, None), conv(x, weight, None)
        )


@pytest.mark.unit
def test_the_sequence_placement_composes_as_data() -> None:
    """The placement is plan-time data: comparable, hashable, and its inner
    replaceable — what the taps rely on to gather a routing table by the
    tap's own chunking."""
    placement = SequenceSharded(1, "context", inner=Sharded(-1, "tensor"), flat=False)
    assert placement == SequenceSharded(1, inner=Sharded(-1))
    assert hash(placement) == hash(SequenceSharded(1, inner=Sharded(-1)))
    assert dataclasses.replace(placement, inner=ExpertLocal()).inner == ExpertLocal()
    assert SequenceSharded().inner == SequenceSharded(1, "context").inner


@pytest.mark.unit
class TestMaskDtype:
    def test_the_mask_is_of_the_asked_dtype(self) -> None:
        """Implicit float32 construction changes the mask dtype even when
        ``torch.equal`` reports the expected values."""
        mask = _left_padded(2, 6, 1)
        for dtype in (torch.float16, torch.bfloat16):
            local = local_causal_mask(mask, range(2, 5), dtype)
            assert local.dtype is dtype
            assert local.min().item() == torch.finfo(dtype).min


@pytest.mark.unit
class TestChunkedConvEdges:
    """Convolution boundaries: one-position history, a single-rank group
    with bias and activation, and the handoff gradient."""

    def test_a_two_wide_kernel_hands_one_position_across(self) -> None:
        conv = _conv()
        channels = 3
        x = _seeded((B, channels, 6), 0)
        weight = _seeded((channels, 2), 1)
        bias = _seeded((channels,), 2)
        expected = conv(x, weight, bias, activation="silu")

        def program(rank: int, c: Collective) -> torch.Tensor:
            frame = SequenceFrame(c, torch.ones(B, 6, dtype=torch.long))
            local = x[..., frame.chunk.start : frame.chunk.stop]
            out = chunked_conv(frame, conv, local, weight, bias, activation="silu")
            return frame.gather(out, 2)

        for got in SimulatedWorld(groups_for(2, context=2), world=2, schedule=0).run(
            program
        ):
            assert (got - expected).abs().max().item() <= CONV_BAND

    def test_a_group_of_one_passes_the_bias_and_the_activation_through(self) -> None:
        conv = _conv()
        x, weight, bias = _seeded((1, 2, 5), 0), _seeded((2, 4), 1), _seeded((2,), 2)
        frame = SequenceFrame(SOLO, torch.ones(1, 5, dtype=torch.long))
        assert torch.equal(
            chunked_conv(frame, conv, x, weight, bias, activation="silu"),
            conv(x, weight, bias, activation="silu"),
        )

    def test_the_history_handoff_carries_the_gradient(self) -> None:
        """The received prefix is on the graph through the chunk's own inputs
        and the sent tail's backward is reached (the function's docstring):
        each rank's input gradient is the whole frame's at its chunk."""
        conv = _conv()
        channels, kernel, padded_len, context = 2, 4, 9, 3
        x = _seeded((B, channels, padded_len), 0)
        weight = _seeded((channels, kernel), 1)
        whole_x = x.clone().requires_grad_(True)
        conv(whole_x, weight, None, activation=None).square().sum().backward()
        assert whole_x.grad is not None

        def program(rank: int, c: Collective) -> torch.Tensor:
            frame = SequenceFrame(c, torch.ones(B, padded_len, dtype=torch.long))
            local = (
                x[..., frame.chunk.start : frame.chunk.stop]
                .clone()
                .requires_grad_(True)
            )
            out = chunked_conv(frame, conv, local, weight, None, activation=None)
            out.square().sum().backward()
            assert local.grad is not None
            return local.grad

        world = SimulatedWorld(
            groups_for(context, context=context), world=context, schedule=0
        )
        for chunk, grad in zip(
            sequence_chunks(padded_len, context), world.run(program)
        ):
            expected = whole_x.grad[..., chunk.start : chunk.stop]
            assert (grad - expected).abs().max().item() <= CONV_BAND
