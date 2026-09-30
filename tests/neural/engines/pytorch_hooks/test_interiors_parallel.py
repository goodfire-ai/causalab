"""The interface taps over fragments (``docs/model_parallelism.md`` §6.2, §6.3, §10.4).

Deterministic simulation of the *real* tap code — ``attention_interface_taps``
wrapping a mixer whose eager function receives this rank's head shards, and
``experts_interface_taps`` wrapping transformers' own grouped dispatch over
this rank's experts with sentinel slots for the rest — driven by the
executor's real adapters (``_interface_capture`` / ``_interface_edit``,
``_experts_capture`` / ``_experts_edit``) over [`Fragments`][causalab.neural.shared.parallel.fragments.Fragments] on the
``SimulatedWorld`` under drawn schedules:

* an ``attention_scores`` / ``attention_query`` / ``attention_z`` read
  equals the global tensor on every rank, and an ``attention_probs`` swap
  lands the rank's heads of the globally edited pattern in the library's own
  value multiply, bit for bit, across drawn schedules;
* the experts interior's captured token-major view and routing table equal
  the world-1 capture, an ``expert:``-scoped selection of it equals world
  1's, and an expert-keyed gate applied through the write path lands exactly
  the world-1 edit on the slots this rank owns;
* the world-1 twins run over the refusing collective and call nothing.

Two mutations close the file: the interface adapter with its ``whole``
dropped fails the read property, with its ``fragment`` dropped the write.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
import transformers.integrations.moe as moe
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from causalab.neural.engines.pytorch_hooks import executor as executor_module
from causalab.neural.engines.pytorch_hooks import experts_interface
from causalab.neural.engines.pytorch_hooks.attention_interface import (
    InterfaceTap,
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.executor import (
    _experts_capture,
    _experts_edit,
    _interface_capture,
    _interface_edit,
)
from causalab.neural.engines.pytorch_hooks.experts_interface import (
    ExpertsTap,
    experts_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.experts_path import EXPERT_PARALLEL_MARK
from causalab.neural.shared.layout import to_contract
from causalab.neural.shared.parallel import taps as taps_module
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments, remap_routing
from causalab.neural.shared.parallel.placement import ExpertLocal, Sharded
from causalab.neural.shared.parallel.taps import TapFragments
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.registry.shapes import (
    attention_pattern,
    bhsd,
    bshd,
    flat_topk_features,
)
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.refusing_collective import RefusingCollective

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


#: The interface managers install one process-global entry, so a world's
#: ranks — threads handing off inside the taps' collectives — share one
#: manager entered once around the run, over a table the ranks fill with
#: their own modules as they build them (each keyed by its module's id). The
#: table starts with a key no module has, so the manager installs its entry.
_KEEP = -1


def _run(program: Any, world: SimulatedWorld | None, manager: Any) -> list[Any]:
    """Run ``program`` on ``world`` — or its world-1 twin over the refusing
    collective — under one installation of ``manager`` over its tap table."""
    with manager(program.taps):
        if world is None:
            return [program(0, RefusingCollective())]
        return world.run(program)


# --------------------------------------------------------------------------- #
# the attention interior (§6.2)
# --------------------------------------------------------------------------- #

BATCH, HEADS, SEQ, DIM, TP = 2, 4, 3, 5, 2


def eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A faithful miniature of the library's eager function — the one softmax
    the tap intercepts, then the library's own value multiply — resolved by
    ``module_eager_attention`` from this module, where `_Mixer` lives."""
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    probs = torch.nn.functional.softmax(scores, dim=-1)
    out = torch.matmul(probs, value).transpose(1, 2).contiguous()
    return out, probs


class _Mixer(torch.nn.Module):
    num_key_value_groups = 1


def _heads(tensor: torch.Tensor, axis: int, rank: int, size: int) -> torch.Tensor:
    """§6.2 spelled by hand: this rank's contiguous block of heads on ``axis``."""
    per_rank = tensor.shape[axis] // size
    return tensor.narrow(axis, rank * per_rank, per_rank)


def _interface_site(component: str, slot: str, shape: Any) -> ResolvedSite:
    return ResolvedSite(
        module=None,
        kind="interface" if slot != "probs" else "out",
        shape=shape,
        component=component,
        interface_slot=slot,
    )


AttentionResult = dict[str, torch.Tensor]


def _attention_program(seed: int) -> Callable[[int, Collective], AttentionResult]:
    q = _seeded((BATCH, HEADS, SEQ, DIM), seed)
    k = _seeded((BATCH, HEADS, SEQ, DIM), seed + 1)
    v = _seeded((BATCH, HEADS, SEQ, DIM), seed + 2)
    # the globally edited pattern a document swaps in — any pattern will do
    replacement = torch.softmax(_seeded((BATCH, HEADS, SEQ, SEQ), seed + 3), dim=-1)
    scaling = DIM**-0.5
    taps: dict[int, tuple[InterfaceTap, ...]] = {_KEEP: ()}

    def program(rank: int, c: Collective) -> AttentionResult:
        size, me = c.size("tensor"), c.rank("tensor")
        fragments = Fragments(c)
        head1 = TapFragments(fragments, Sharded(1, "tensor"))
        head2 = TapFragments(fragments, Sharded(2, "tensor"))
        local = {
            name: (_heads(t, 1, me, size) if size > 1 else t)
            for name, t in (("q", q), ("k", k), ("v", v))
        }
        sink: dict[Any, torch.Tensor] = {}
        sites = {
            "query": _interface_site("attention_query", "query", bhsd(HEADS, DIM)),
            "scores": _interface_site(
                "attention_scores", "scores", attention_pattern(HEADS)
            ),
            "z": _interface_site("attention_z", "z", bshd(HEADS, DIM)),
        }
        probs_site = _interface_site(
            "attention_probs", "probs", attention_pattern(HEADS)
        )

        def swap(contract: torch.Tensor) -> None:
            contract.copy_(replacement)

        # edits before reads, as the executor registers them
        entries = (
            InterfaceTap(
                slot="probs", edit=_interface_edit(probs_site, swap, BATCH, head1)
            ),
            InterfaceTap(
                slot="query",
                read=_interface_capture(sink, "query", sites["query"], BATCH, head1),
            ),
            InterfaceTap(
                slot="scores",
                read=_interface_capture(sink, "scores", sites["scores"], BATCH, head1),
            ),
            InterfaceTap(
                slot="z", read=_interface_capture(sink, "z", sites["z"], BATCH, head2)
            ),
        )
        mixer = _Mixer()
        taps[id(mixer)] = entries
        out, weights = ALL_ATTENTION_FUNCTIONS["eager"](
            mixer, local["q"], local["k"], local["v"], None, scaling=scaling
        )
        return {**sink, "out": out, "weights": weights}

    program.q, program.k, program.v = q, k, v  # type: ignore[attr-defined]
    program.replacement = replacement  # type: ignore[attr-defined]
    program.scaling = scaling  # type: ignore[attr-defined]
    program.taps = taps  # type: ignore[attr-defined]
    return program


def _run_attention(program: Any, world: SimulatedWorld | None = None) -> list[Any]:
    return _run(program, world, attention_interface_taps)


def _assert_attention(results: list[AttentionResult], program: Any, size: int) -> None:
    q, k, v, replacement = program.q, program.k, program.v, program.replacement
    scores = torch.matmul(q, k.transpose(2, 3)) * program.scaling
    # the write feeds the library's own value multiply, globally
    z = torch.matmul(replacement, v).transpose(1, 2).contiguous()
    for rank, got in enumerate(results):
        me = rank % size
        assert torch.equal(
            got["query"], to_contract(q, bhsd(HEADS, DIM), batch_size=BATCH)
        )
        assert torch.equal(got["scores"], scores), f"rank {rank}: scores read"
        assert torch.equal(got["z"], to_contract(z, bshd(HEADS, DIM), batch_size=BATCH))
        expected_out = _heads(z, 2, me, size) if size > 1 else z
        assert torch.equal(got["out"], expected_out), f"rank {rank}: the write landed"
        expected_weights = _heads(replacement, 1, me, size) if size > 1 else replacement
        assert torch.equal(got["weights"], expected_weights), f"rank {rank}: pattern"


@pytest.mark.property
class TestAttentionInteriorScenarios:
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @example(schedule=[], seed=0)
    @_SETTINGS
    def test_reads_are_global_and_a_probs_swap_lands_the_ranks_heads(
        self, schedule: Schedule, seed: int
    ) -> None:
        program = _attention_program(seed)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=schedule)
        _assert_attention(_run_attention(program, world), program, TP)

    @_SETTINGS
    @given(seed=st.integers(0, 999), schedule=ps.schedules())
    def test_the_result_does_not_depend_on_the_schedule(
        self, seed: int, schedule: Schedule
    ) -> None:
        program = _attention_program(seed)
        world = SimulatedWorld(
            groups_for(4, data=2, tensor=2), world=4, schedule=schedule
        )
        _assert_attention(_run_attention(program, world), program, 2)

    def test_a_head_selector_on_an_interface_slot_is_a_global_feature_slice(
        self,
    ) -> None:
        """``head:`` on ``attention_query`` slices the *contract* the read
        produced — the global head order — so the same head number means the
        same head on every rank, whichever rank computed it."""
        program = _attention_program(7)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=1)
        results = _run_attention(program, world)
        head = HEADS - 1  # a head rank 0 never holds locally
        window = slice(head * DIM, (head + 1) * DIM)
        expected = program.q[:, head]  # (b, s, d), in the global head order
        for got in results:
            assert torch.equal(got["query"][..., window], expected)


# --------------------------------------------------------------------------- #
# the experts interior (§6.3): sentinel slots, the global table, an expert gate
# --------------------------------------------------------------------------- #

#: widths in multiples of four floats: the grouped kernel wants 16-byte strides
TOKENS, TOP_K, HIDDEN, D_EXPERT, EXPERTS, EP = 6, 2, 8, 4, 8, 4
ROWS = 2  # the batch the token-major view flattens


class _Experts(torch.nn.Module):
    """The shape ``grouped_mm_experts_forward`` reads off ``Qwen3_5MoeExperts``:
    3-D ``[gate | up]`` and down weights, one shared ``act_fn``."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(gate_up)
        self.down_proj = torch.nn.Parameter(down)
        self.num_experts = int(gate_up.shape[0])
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.act_fn = torch.nn.SiLU()

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up


def _experts_site(component: str, slot: str, width: int) -> ResolvedSite:
    return ResolvedSite(
        module=None,
        kind="experts",
        shape=flat_topk_features(TOP_K, width),
        component=component,
        interface_slot=slot,
    )


ExpertsResult = dict[str, torch.Tensor]


def _experts_program(seed: int) -> Callable[[int, Collective], ExpertsResult]:
    generator = torch.Generator().manual_seed(seed)
    gate_up = torch.randn((EXPERTS, 2 * D_EXPERT, HIDDEN), generator=generator)
    down = torch.randn((EXPERTS, HIDDEN, D_EXPERT), generator=generator)
    hidden = torch.randn((TOKENS, HIDDEN), generator=generator)
    table = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(TOKENS)]
    )
    weights = torch.softmax(torch.randn((TOKENS, TOP_K), generator=generator), dim=-1)
    # an expert-keyed gate over the down slot: one scale per (expert, neuron)
    gate_table = torch.rand((EXPERTS, HIDDEN), generator=generator) + 0.5
    taps: dict[int, tuple[ExpertsTap, ...]] = {_KEEP: ()}

    def program(rank: int, c: Collective) -> ExpertsResult:
        size, me = c.size("expert"), c.rank("expert")
        num_local = EXPERTS // size
        module = _Experts(
            gate_up.narrow(0, me * num_local, num_local).clone(),
            down.narrow(0, me * num_local, num_local).clone(),
        )
        if size > 1:
            # the rank's shard under the expert axis, marked as the styles
            # mark a module whose weight the loader kept local: the fact the
            # taps read to know the remapped table can hold a sentinel, whose
            # rows the grouped kernel leaves uninitialised (a NaN-filled run
            # makes the unmasked read all-NaN; Linux memory made it garbage)
            setattr(module, EXPERT_PARALLEL_MARK, True)
        local_table = remap_routing(table, EXPERTS, me, size)
        owned = local_table != num_local
        local_weights = torch.where(owned, weights, torch.zeros(()))
        tap = TapFragments(
            Fragments(c),
            ExpertLocal("expert"),
            remapped_routing=True,
            num_experts=EXPERTS,
        )
        sink: dict[Any, torch.Tensor] = {}
        idx_sink: dict[Any, torch.Tensor] = {}
        activation = _experts_site("expert_activation", "activation", D_EXPERT)
        output = _experts_site("expert_output", "down", HIDDEN)

        def gate(contract: torch.Tensor, routing: torch.Tensor | None = None) -> None:
            assert routing is not None
            assert torch.equal(routing, table.reshape(ROWS, -1, TOP_K)), "global table"
            b, s, _ = contract.shape
            view = contract.view(b, s, TOP_K, HIDDEN)
            view.mul_(gate_table[routing])

        entries = (
            ExpertsTap(slot="down", edit=_experts_edit(output, gate, ROWS, tap)),
            ExpertsTap(
                slot="activation",
                read=_experts_capture(sink, idx_sink, "act", activation, ROWS, tap),
            ),
            ExpertsTap(
                slot="down",
                read=_experts_capture(sink, idx_sink, "down", output, ROWS, tap),
            ),
        )
        taps[id(module)] = entries
        out = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"](
            module, hidden, local_table, local_weights
        )
        # the module's partial output, summed over the expert group the way
        # ``moe_tp_experts`` does after the call
        total = c.all_reduce_sum(out, "expert") if size > 1 else out
        return {
            "activation": sink["act"],
            "down": sink["down"],
            "idx": idx_sink["act"],
            "out": total,
        }

    program.table = table  # type: ignore[attr-defined]
    program.taps = taps  # type: ignore[attr-defined]
    return program


def _run_experts(program: Any, world: SimulatedWorld | None = None) -> list[Any]:
    return _run(program, world, experts_interface_taps)


def _select(
    view: torch.Tensor, idx: torch.Tensor, expert: int, width: int
) -> torch.Tensor:
    """The ``expert:`` face by hand: the slots the router sent to ``expert``."""
    rows = view.reshape(-1, TOP_K, width)
    return rows[idx.reshape(-1, TOP_K) == expert]


@pytest.mark.property
class TestExpertsInteriorScenarios:
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_reads_equal_world_one_and_an_expert_gate_lands_the_owned_slots(
        self, schedule: Schedule, seed: int
    ) -> None:
        program = _experts_program(seed)
        (solo,) = _run_experts(program)
        world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=schedule)
        for rank, got in enumerate(_run_experts(program, world)):
            assert torch.equal(got["idx"], program.table.reshape(ROWS, -1, TOP_K))
            assert torch.equal(got["activation"], solo["activation"]), f"rank {rank}"
            # the down read sees the *written* value (edits before reads), globally
            assert torch.equal(got["down"], solo["down"]), f"rank {rank}"
            for expert in range(EXPERTS):
                assert torch.equal(
                    _select(got["activation"], got["idx"], expert, D_EXPERT),
                    _select(solo["activation"], solo["idx"], expert, D_EXPERT),
                )
            # the block output: partials summed over the group vs. one process
            assert torch.allclose(got["out"], solo["out"], atol=1e-6, rtol=0)


@pytest.mark.unit
class TestWorldOne:
    def test_the_interface_adapters_call_no_collective_at_world_one(self) -> None:
        program = _attention_program(3)
        _assert_attention(_run_attention(program), program, 1)
        experts = _experts_program(3)
        (got,) = _run_experts(experts)
        assert torch.equal(got["idx"], experts.table.reshape(ROWS, -1, TOP_K))

    def test_the_experts_write_changed_the_output(self) -> None:
        """Anti-vacuity: the gate is not the identity."""
        program = _experts_program(5)
        (with_gate,) = _run_experts(program)
        generator = torch.Generator().manual_seed(5)
        gate_up = torch.randn((EXPERTS, 2 * D_EXPERT, HIDDEN), generator=generator)
        down = torch.randn((EXPERTS, HIDDEN, D_EXPERT), generator=generator)
        hidden = torch.randn((TOKENS, HIDDEN), generator=generator)
        table = program.table
        weights = torch.softmax(torch.randn((TOKENS, TOP_K), generator=generator), -1)
        plain = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"](
            _Experts(gate_up, down), hidden, table, weights
        )
        assert not torch.allclose(plain, with_gate["out"])

    def test_the_managers_leave_the_registries_as_they_found_them(self) -> None:
        """A world's ranks share one manager: nothing stale stays installed."""
        assert "eager" not in ALL_ATTENTION_FUNCTIONS
        assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is moe.grouped_mm_experts_forward
        assert moe._grouped_linear is not experts_interface._GROUPED_LINEAR


@pytest.mark.unit
class TestInterfaceMutations:
    def test_dropping_whole_fails_the_read_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        program = _attention_program(0)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        _assert_attention(_run_attention(program, world), program, TP)
        monkeypatch.setattr(
            taps_module.TapFragments, "whole", lambda self, native: native
        )
        with pytest.raises((AssertionError, RankFailed)):
            _assert_attention(_run_attention(program, world), program, TP)

    def test_dropping_fragment_fails_the_write_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        program = _attention_program(0)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        monkeypatch.setattr(
            taps_module.TapFragments,
            "fragment",
            lambda self, edited, routing=None: edited,
        )
        # the global pattern re-enters a local value multiply: a shape error
        # on the rank, never a silently wrong output
        with pytest.raises((AssertionError, RankFailed)):
            _assert_attention(_run_attention(program, world), program, TP)

    def test_the_executor_reaches_the_taps_seam(self) -> None:
        assert executor_module.TapFragments is taps_module.TapFragments


@pytest.mark.unit
class TestSentinelRowsInBackward:
    """Under expert parallelism the grouped kernel skips the sentinel-tail
    rows and leaves them **uninitialised** in its forward output and in its
    backward ``d_input`` (transformers' one pre-mask / one post-mask cover
    the library's own path). A tap sits between the two grouped matmuls, so
    the garbage ``d_input`` rows reach its edit's backward: an additive edit
    sums every incoming row into its parameter's gradient. Filling
    uninitialised memory with NaN makes the leak deterministic — measured
    on the tiny MoE before the fix, ~3 of 10 expert-gate fits under
    ``ep=2`` diverged from step 0, two of them to NaN."""

    @staticmethod
    def _gradient(fill: bool, sentinel: bool) -> torch.Tensor:
        generator = torch.Generator().manual_seed(11)
        num_local = EXPERTS // EP
        module = _Experts(
            torch.randn((num_local, 2 * D_EXPERT, HIDDEN), generator=generator),
            torch.randn((num_local, HIDDEN, D_EXPERT), generator=generator),
        )
        # the module is under the expert axis, as the styles mark one whose
        # weight the loader kept local (``styles/dtensor.py``, ``fragment.py``):
        # the fact the taps read to know a sentinel can appear
        setattr(module, EXPERT_PARALLEL_MARK, True)
        hidden = torch.randn((TOKENS, HIDDEN), generator=generator)
        # every token's first slot is owned; the second is another rank's
        # (the sentinel id ``num_local``) with routing weight zero — or, for
        # the reference, a real expert at weight zero, which contributes
        # nothing to the output or to any gradient either
        table = torch.stack(
            [torch.arange(TOKENS) % num_local, torch.full((TOKENS,), num_local)], -1
        )
        if not sentinel:
            table[:, 1] = 0
        weights = torch.stack([torch.ones(TOKENS), torch.zeros(TOKENS)], -1)
        bias = torch.nn.Parameter(torch.zeros(D_EXPERT))

        def edit(value: torch.Tensor, _idx: torch.Tensor) -> torch.Tensor:
            tokens = value.shape[0]
            return (value.view(tokens, TOP_K, D_EXPERT) + bias).view(tokens, -1)

        entries = (ExpertsTap(slot="activation", edit=edit),)
        was = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.utils.deterministic.fill_uninitialized_memory,
        )
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        try:
            with experts_interface_taps({id(module): entries}):
                out = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"](
                    module, hidden, table, weights
                )
            out.sum().backward()
        finally:
            torch.use_deterministic_algorithms(was[0], warn_only=was[1])
            torch.utils.deterministic.fill_uninitialized_memory = was[2]
        assert bias.grad is not None
        return bias.grad.detach().clone()

    def test_the_sentinel_rows_garbage_never_reaches_an_edits_gradient(self) -> None:
        reference = self._gradient(fill=False, sentinel=False)
        assert torch.isfinite(reference).all()
        got = self._gradient(fill=True, sentinel=True)
        assert torch.isfinite(got).all(), got
        assert torch.allclose(got, reference, rtol=1e-6, atol=1e-6), (got, reference)

    def test_anti_vacuity_the_edit_has_a_gradient(self) -> None:
        assert bool((self._gradient(fill=False, sentinel=False) != 0).any())
