"""The autograd-aware ``whole`` / ``fragment`` pairs (``docs/model_parallelism.md`` §7).

The pairing makes a tap at a sharded site look replicated to the featurizer's
backward: ``fragment``'s backward all-gathers the ranks' gradient slices, so
every rank holds the full ``∂L/∂edited`` and a featurizer upstream of the
write has the **full** gradient on every rank; ``whole``'s backward hands back
this rank's slice, the true ``∂L/∂x_r``. The oracle here is a world-1 program
over the global tensors in which *each rank's fragment is consumed once* —
``L = Σ_r Σ (edited ⊙ mask_r ⊙ C)`` — differentiated by torch alone, and
every property is asserted bit for bit: the losses here are sums of products
with weights chosen so that no rounding depends on the reduction order.

Three tiers, one class each: ``property`` — the forward of the routed path is
the raw path's value bit for bit over every placement the fragments suite
draws, and the featurizer gradient equals the oracle on every rank of every
group; ``unit`` — the pairs by hand on the placements the executor meets
(a head shard, a slotted shard, a repeated shard, the expert slots, a partial
sum, a sequence chunk with an uneven frame), the world-1 fast path calling
nothing, and the no-grad path recording nothing; the mutations — a
``fragment`` whose backward keeps only its own slice hands every rank a
partial gradient (the ``1/size`` the guard's mean then applies), a ``whole``
whose gather detaches hands an upstream featurizer nothing.

The context-parallel training tier pins the same functions by the arithmetic
of the sites they serve, on a context group of the ``SimulatedWorld``: the
tap pair against a world-1 reference of the same edit (the parameter
gradient the full one on every rank, the whole's this rank's chunk); the
faithful ``gather_reduce_scatter`` for a downstream that differs per rank,
and the mutation of routing it through the tap pair; ``send_with_grad`` /
``recv_with_grad`` with a receiver that skips its backward send refused by
name; and ``carry``, a peer-only tensor kept on the graph so its send's
backward is reached.

The pipeline training tier crosses a stage boundary: stage 0 sends its
residual with ``send_with_grad`` and carries the broadcast loss onto it,
stage 1 receives with ``recv_with_grad`` — stage 0's parameter gradient is
the world-1 gradient bit for bit and the collective sequence is one send and
one receive each way; the facts a stage forward relies on — a residual that
requires no gradient still records its send node (the zero-size leaf),
nothing is recorded under ``no_grad``, a carried value is the value bit for
bit whose backward runs the link's node once however many values ride it.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.shared.parallel import autograd as autograd_module
from causalab.neural.shared.parallel import fragments as fragments_module
from causalab.neural.shared.parallel.autograd import (
    Handoff,
    HandoffMismatch,
    AutogradError,
    carry,
    edit_fragment,
    edit_summand,
    gather_for_edit,
    gather_reduce_scatter,
    recv_with_grad,
    send_with_grad,
    sum_for_edit,
)
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.context import SequenceFrame
from causalab.neural.shared.parallel.fragments import Fragments, fragment
from causalab.neural.shared.parallel.placement import (
    AXES,
    Axis,
    ExpertLocal,
    PartialSum,
    Placement,
    Replicated,
    SequenceSharded,
    Sharded,
    StageLocal,
)
from tests._helpers import parallel_strategies as ps
from tests._helpers.refusing_collective import RefusingCollective
from tests._helpers.simulated_world import (
    Abandoned,
    Hang,
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests.neural.shared.parallel.test_fragments import (
    Case,
    Groups,
    placement_cases,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: The placements whose gradient the pairing speaks for. A ``StageLocal``
#: non-owner receives its ``whole`` by broadcast and computes nothing
#: downstream of it — the pipeline's training branch, not this one.
GRADIENT_KINDS = ("replicated", "sharded", "expert_local", "sequence_sharded")


# --------------------------------------------------------------------------- #
# worlds and oracles
# --------------------------------------------------------------------------- #


def _world(
    axis: Axis, groups: Groups, world: int, schedule: Schedule = 0
) -> SimulatedWorld:
    """A simulated world grouped on ``axis``, every other axis singletons —
    the layout the fragments suite's cases describe."""
    layout: dict[Axis, Groups] = {
        other: tuple((rank,) for rank in range(world)) for other in AXES
    }
    layout[axis] = groups
    return SimulatedWorld(layout, world=world, schedule=schedule)


class _At:
    """Rank ``rank`` of ``size`` on every axis, with no collective: what the
    world-1 oracle hands ``fragment`` to name each rank's piece of a global
    tensor without a rendezvous (``fragment`` reads only ``rank`` / ``size``
    for every placement the oracle covers)."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self, axis: Axis) -> int:
        return self._rank

    def size(self, axis: Axis) -> int:
        return self._size

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the oracle touched the collective: {name}")


def _grid(shape: tuple[int, ...], seed: int, scale: float = 0.25) -> torch.Tensor:
    """Small multiples of ``scale`` — values whose products and short sums are
    exact in fp32, so a gradient's bits do not depend on reduction order."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 9, shape, generator=generator).to(torch.float32) * scale


def _is_sum(placement: Placement) -> bool:
    inner = placement.inner if isinstance(placement, SequenceSharded) else placement
    return isinstance(inner, (ExpertLocal, PartialSum))


def _fragments(placement: Placement) -> bool:
    """Whether the ranks hold different pieces — so each rank's loss term is
    a partial the model's own reduction sums — or the same whole tensor, in
    which case every rank computes the one loss and nothing is summed."""
    return not isinstance(placement, Replicated)


def _oracle(
    case: Case, group: tuple[int, ...], theta: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """World 1: ``edited = X ⊙ Θ``; each rank's fragment of it, weighed by its
    fragment of ``C``, summed — the loss every rank of the pairing computes.
    Returns ``(∂L/∂Θ, ∂L/∂X)``."""
    x = case.globals_by_group[group].clone().requires_grad_(True)
    th = theta.clone().requires_grad_(True)
    edited = x * th
    loss = torch.zeros(())
    size = len(group)
    for local in range(size if _fragments(case.placement) else 1):
        mask = _mask(case, group, local)
        loss = loss + (edited * mask * weights).sum()
    loss.backward()
    assert th.grad is not None and x.grad is not None
    return th.grad, x.grad


def _mask(case: Case, group: tuple[int, ...], local: int) -> torch.Tensor:
    """Which entries of the group's global tensor rank ``local`` holds, as a
    0/1 tensor of the global shape: the raw ``fragment`` of a tensor of
    entry numbers (offset by one, so a zeroed slot reads as held by nobody)
    names them without touching any collective."""
    shape = case.globals_by_group[group].shape
    numbers = torch.arange(1, shape.numel() + 1).reshape(shape)
    with torch.no_grad():
        kept = fragment(
            numbers,
            case.placement,
            _At(local, len(group)),
            routing=case.routing_by_group.get(group),
            num_experts=case.num_experts,
        )
    mask = torch.zeros(shape.numel())
    mask[kept[kept > 0] - 1] = 1.0
    return mask.reshape(shape)


ByGroup = dict[tuple[int, ...], torch.Tensor]


def _pair_program(
    case: Case, theta: ByGroup, weights: ByGroup
) -> Callable[[int, Collective], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Every rank: ``whole`` its local piece, weigh by the replicated ``Θ``,
    ``fragment`` back, weigh by its fragment of ``C``, sum, and make the
    loss replicated through the model's own reduction (an all-reduce whose
    backward is the identity). Returns ``(whole, ∂L/∂Θ, ∂L/∂x_r)``."""

    def program(
        rank: int, c: Collective
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fragments = Fragments(c)
        axis = case.group_axis
        group = case.group_of(rank)
        x = case.locals_by_rank[rank].clone().requires_grad_(True)
        th = theta[group].clone().requires_grad_(True)
        made_whole = fragments.whole(x, case.placement)
        edited = made_whole * th
        piece = fragments.fragment(
            edited,
            case.placement,
            routing=case.routing_of(rank),
            num_experts=case.num_experts,
        )
        with torch.no_grad():
            weight = fragments.fragment(
                weights[group],
                case.placement,
                routing=case.routing_of(rank),
                num_experts=case.num_experts,
            )
        local_loss = (piece * weight).sum()
        loss = (
            sum_for_edit(local_loss, axis, c)
            if c.size(axis) > 1 and _fragments(case.placement)
            else local_loss
        )
        loss.backward()
        assert th.grad is not None and x.grad is not None
        return made_whole.detach(), th.grad, x.grad

    return program


def _check_pair_gradient(case: Case, schedule: Schedule = 0) -> None:
    """The property, as a function so the mutations can assert it fails."""
    # one Θ and one C per group: the groups' global tensors may differ in shape
    theta = {g: _grid(case.globals_by_group[g].shape, 101) for g in case.groups}
    weights = {g: _grid(case.globals_by_group[g].shape, 202) for g in case.groups}
    world = _world(case.group_axis, case.groups, case.world, schedule)
    results = world.run(_pair_program(case, theta, weights))
    for group in case.groups:
        theta_grad, x_grad = _oracle(case, group, theta[group], weights[group])
        size = len(group)
        for local, rank in enumerate(group):
            made_whole, got_theta, got_x = results[rank]
            assert torch.equal(made_whole, case.global_of(rank)), (rank, "whole")
            # every rank's featurizer gradient is the full gradient
            assert torch.equal(got_theta, theta_grad), (rank, case.placement, "Θ")
            # this rank's ∂L/∂x_r: its slice of the whole for a gather, the
            # whole for a sum (the all-reduce's local Jacobian is the identity)
            expected_x = (
                x_grad
                if _is_sum(case.placement)
                else fragment(
                    x_grad,
                    case.placement,
                    _At(local, size),
                    routing=case.routing_of(rank),
                    num_experts=case.num_experts,
                )
            )
            assert torch.equal(got_x, expected_x), (rank, case.placement, "x_r")


# --------------------------------------------------------------------------- #
# property tier
# --------------------------------------------------------------------------- #


def _forward_program(
    case: Case,
) -> Callable[[int, Collective], tuple[torch.Tensor, torch.Tensor, bool, bool]]:
    def program(
        rank: int, c: Collective
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
        fragments = Fragments(c)
        local = case.locals_by_rank[rank]
        tracked = local.clone().requires_grad_(True)
        made_whole = fragments.whole(tracked, case.placement)
        back = fragments.fragment(
            made_whole,
            case.placement,
            routing=case.routing_of(rank),
            num_experts=case.num_experts,
        )
        return (
            made_whole.detach(),
            back.detach(),
            made_whole.requires_grad,
            back.requires_grad,
        )

    return program


@pytest.mark.property
class TestPairingProperties:
    @_SETTINGS
    @given(case=placement_cases())
    def test_the_routed_forward_is_the_raw_value_bit_for_bit(self, case: Case) -> None:
        """A tensor that requires grad takes the autograd path; its ``whole``
        is the global tensor and its ``fragment`` the local piece, bit for
        bit, on every rank of every placement — and both stay on the graph
        wherever the raw path is not a broadcast to a non-owner."""
        world = _world(case.group_axis, case.groups, case.world)
        for rank, (made_whole, back, whole_grad, back_grad) in enumerate(
            world.run(_forward_program(case))
        ):
            assert torch.equal(made_whole, case.global_of(rank)), (rank, "whole")
            assert torch.equal(back, case.locals_by_rank[rank]), (rank, "fragment")
            owner = not isinstance(case.placement, StageLocal) or (
                case.group_of(rank).index(rank) == case.placement.stage
            )
            if owner:
                assert whole_grad, (rank, case.placement, "whole left the graph")
                assert back_grad or back.numel() == 0, (rank, "fragment left the graph")

    @_SETTINGS
    @given(case=placement_cases(GRADIENT_KINDS))
    def test_the_featurizer_gradient_is_the_full_gradient_on_every_rank(
        self, case: Case
    ) -> None:
        _check_pair_gradient(case)


# --------------------------------------------------------------------------- #
# unit tier: the placements the executor meets, by hand
# --------------------------------------------------------------------------- #

TP = 2


def _sharded_case(placement: Sharded, shape: tuple[int, ...], size: int) -> Case:
    """One group of ``size`` ranks holding a global tensor sharded by
    ``placement``: the expected locals spelled by hand (§4, §6.6)."""
    group = tuple(range(size))
    global_tensor = _grid(shape, 7)
    axis = placement.axis % len(shape)
    locals_by_rank: dict[int, torch.Tensor] = {}
    for rank in group:
        if placement.slots > 1:
            runs = global_tensor.unflatten(
                axis, (placement.slots, shape[axis] // placement.slots)
            )
            width = runs.shape[axis + 1] // size
            locals_by_rank[rank] = runs.narrow(axis + 1, rank * width, width).flatten(
                axis, axis + 1
            )
        else:
            chunks = size // placement.repeat
            width = shape[axis] // chunks
            index = rank // placement.repeat
            locals_by_rank[rank] = global_tensor.narrow(axis, index * width, width)
    return Case(
        placement=placement,
        world=size,
        groups=(group,),
        globals_by_group={group: global_tensor},
        locals_by_rank=locals_by_rank,
    )


def _partial_sum_case() -> Case:
    group = (0, 1)
    global_tensor = _grid((2, 3, 4), 8)
    summands = {0: _grid((2, 3, 4), 9), 1: torch.zeros(2, 3, 4)}
    summands[1] = global_tensor - summands[0]  # exact: multiples of 0.25
    return Case(
        placement=PartialSum("tensor"),
        world=2,
        groups=(group,),
        globals_by_group={group: global_tensor},
        locals_by_rank=summands,
    )


@pytest.mark.unit
class TestPairsByHand:
    @pytest.mark.parametrize(
        "placement, shape, size",
        [
            (Sharded(1, "tensor"), (2, 4, 3, 2), 2),  # the head axis, bhsd
            (Sharded(-1, "tensor"), (2, 3, 8), 4),  # a colwise output
            (Sharded(-1, "tensor", slots=2), (6, 8), 2),  # the experts' view under TP
            (Sharded(1, "tensor", repeat=2), (2, 4, 3, 2), 4),  # a replicated KV head
        ],
    )
    def test_a_shard_pair_hands_every_rank_the_full_featurizer_gradient(
        self, placement: Sharded, shape: tuple[int, ...], size: int
    ) -> None:
        _check_pair_gradient(_sharded_case(placement, shape, size))

    def test_a_partial_sum_pair_hands_every_rank_the_full_featurizer_gradient(
        self,
    ) -> None:
        _check_pair_gradient(_partial_sum_case())

    def test_a_sequence_chunk_with_an_uneven_frame_pairs_through_the_padding(
        self,
    ) -> None:
        """``cp=2`` over five positions: chunks of two and three, padded to
        three for the wire and cut back — the gradient slices travel the same
        way, so the featurizer gradient is exact on both ranks."""
        padded_len, rows, feature = 5, 2, 4
        mask = torch.ones(rows, padded_len)
        global_tensor = _grid((rows, padded_len, feature), 11)
        theta = _grid((rows, padded_len, feature), 12)
        weights = _grid((rows, padded_len, feature), 13)
        placement = SequenceSharded(1, "context")

        def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
            frame = SequenceFrame(c, mask)
            fragments = Fragments(c)
            x = frame.fragment(global_tensor, 1).clone().requires_grad_(True)
            th = theta.clone().requires_grad_(True)
            edited = fragments.whole(x, placement, frame=frame) * th
            piece = fragments.fragment(edited, placement, frame=frame)
            local_loss = (piece * frame.fragment(weights, 1)).sum()
            sum_for_edit(local_loss, "context", c).backward()
            assert th.grad is not None and x.grad is not None
            return th.grad, x.grad

        world = SimulatedWorld(
            {**{a: ((0,), (1,)) for a in AXES}, "context": ((0, 1),)},
            world=2,
            schedule=3,
        )
        results = world.run(program)
        expected_theta = global_tensor * weights
        expected_x = theta * weights
        chunks = (range(0, 2), range(2, 5))
        for rank, (theta_grad, x_grad) in enumerate(results):
            assert torch.equal(theta_grad, expected_theta), rank
            chunk = chunks[rank]
            assert torch.equal(x_grad, expected_x[:, chunk.start : chunk.stop]), rank

    def test_the_faithful_pair_gives_the_true_gradient_when_the_downstream_differs(
        self,
    ) -> None:
        """A gather whose consumer is *not* replicated — each rank weighs the
        whole by its own weights, the context-parallel KV pattern: the true
        ``∂L/∂x_r`` sums every rank's partial before taking this rank's slice."""
        x_global = _grid((4, 3), 21)
        per_rank = {rank: _grid((4, 3), 30 + rank) for rank in range(TP)}

        def program(rank: int, c: Collective) -> torch.Tensor:
            x = x_global.narrow(0, 2 * rank, 2).clone().requires_grad_(True)
            made_whole = gather_reduce_scatter(x, 0, "tensor", c)
            local_loss = (made_whole * per_rank[rank]).sum()
            sum_for_edit(local_loss, "tensor", c).backward()
            assert x.grad is not None
            return x.grad

        world = SimulatedWorld(
            {
                **{a: ((0,), (1,)) for a in AXES},
                "tensor": ((0, 1),),
                "model": ((0, 1),),
            },
            world=2,
            schedule=0,
        )
        total = per_rank[0] + per_rank[1]
        for rank, grad in enumerate(world.run(program)):
            assert torch.equal(grad, total.narrow(0, 2 * rank, 2)), rank

    def test_the_handoff_pair_carries_the_gradient_back_to_the_sender(self) -> None:
        """Rank 0 sends ``x ⊙ Θ`` to rank 1, which computes the loss: rank 0's
        ``Θ`` receives the gradient the receiver's backward sends back — with
        the sender's own continuation of the tensor added, here none."""
        x = _grid((2, 3), 41)
        weights = _grid((2, 3), 42)

        def program(rank: int, c: Collective) -> torch.Tensor | None:
            th = torch.ones(2, 3, requires_grad=True)
            if rank == 0:
                link = send_with_grad(x * th, 1, "pipeline", c)
                (link.sum() * 0.0).backward()  # nothing local continues from it
                return th.grad
            received = recv_with_grad(
                th, (2, 3), torch.float32, torch.device("cpu"), 0, "pipeline", c
            )
            (received * weights).sum().backward()
            return None

        world = SimulatedWorld(
            {**{a: ((0,), (1,)) for a in AXES}, "pipeline": ((0, 1),)},
            world=2,
            schedule=0,
        )
        sender, receiver = world.run(program)
        assert sender is not None and torch.equal(sender, x * weights)
        assert receiver is None


# --------------------------------------------------------------------------- #
# unit tier: world 1 and the no-grad path
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestFastPaths:
    def test_world_one_returns_the_tensor_itself_and_calls_nothing(self) -> None:
        fragments = Fragments(RefusingCollective())
        x = torch.randn(2, 3, 4, requires_grad=True)
        for placement in (Sharded(-1), ExpertLocal(), PartialSum(), SequenceSharded()):
            assert fragments.whole(x, placement) is x
            assert fragments.fragment(x, placement) is x

    def test_without_grad_the_raw_path_runs_and_records_nothing(self) -> None:
        case = _sharded_case(Sharded(-1, "tensor"), (2, 3, 8), TP)

        def program(rank: int, c: Collective) -> tuple[bool, bool, bool]:
            fragments = Fragments(c)
            x = case.locals_by_rank[rank].clone().requires_grad_(True)
            with torch.no_grad():
                made_whole = fragments.whole(x, case.placement)
                back = fragments.fragment(made_whole, case.placement)
            plain = fragments.whole(case.locals_by_rank[rank], case.placement)
            return (
                made_whole.grad_fn is None,
                back.grad_fn is None,
                plain.grad_fn is None,
            )

        world = _world("tensor", case.groups, case.world)
        for result in world.run(program):
            assert result == (True, True, True)

    def test_the_functions_at_a_group_of_one_are_the_identity(self) -> None:
        x = torch.randn(2, 3, requires_grad=True)
        assert torch.equal(gather_for_edit(x, 0, "tensor", SOLO), x)
        assert torch.equal(edit_fragment(x, 0, "tensor", SOLO), x)
        assert torch.equal(sum_for_edit(x, "tensor", SOLO), x)
        assert torch.equal(edit_summand(x, None, "tensor", SOLO), x)
        assert torch.equal(gather_reduce_scatter(x, 0, "tensor", SOLO), x)


# --------------------------------------------------------------------------- #
# unit tier: the mutations
# --------------------------------------------------------------------------- #


def _own_slice_only(
    edited: torch.Tensor, dim: int, axis: Axis, collective: Collective, **kwargs: Any
) -> torch.Tensor:
    """The mutation: a ``fragment`` that is plain torch math — the narrow's
    backward scatters this rank's slice into zeros, a partial gradient."""
    size, rank = collective.size(axis), collective.rank(axis)
    return fragments_module._chunk(edited, dim, rank, size, Sharded(dim, axis))


def _detaching_gather(
    tensor: torch.Tensor, dim: int, axis: Axis, collective: Collective, **kwargs: Any
) -> torch.Tensor:
    """The mutation: a ``whole`` that is the raw all-gather — fresh buffers,
    off the graph."""
    return collective.all_gather(tensor, dim, axis)


@pytest.mark.unit
class TestMutations:
    def test_a_fragment_without_the_all_gather_backward_hands_out_partials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case = _sharded_case(Sharded(-1, "tensor"), (2, 3, 8), TP)
        _check_pair_gradient(case)
        monkeypatch.setattr(fragments_module, "edit_fragment", _own_slice_only)
        with pytest.raises(AssertionError, match="Θ"):
            _check_pair_gradient(case)

    def test_a_whole_that_detaches_zeroes_the_upstream_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two taps in a row: the second tap's ``whole`` sits downstream of the
        first tap's edit, so a gather off the graph hands the first
        featurizer nothing through it."""
        placement = Sharded(-1, "tensor")
        global_tensor = _grid((2, 3, 8), 51)
        weights = _grid((2, 3, 8), 52)

        def program(rank: int, c: Collective) -> torch.Tensor:
            fragments = Fragments(c)
            x = fragment(global_tensor, placement, c)
            first = torch.ones(2, 3, 8, requires_grad=True)
            second = torch.ones(2, 3, 8, requires_grad=True)
            edited = fragments.whole(x, placement) * first  # the first tap's write
            downstream = fragments.fragment(edited, placement) * 2.0
            again = fragments.whole(downstream, placement)  # the second tap
            piece = fragments.fragment(again * second, placement)
            with torch.no_grad():
                weight = fragments.fragment(weights, placement)
            sum_for_edit((piece * weight).sum(), "tensor", c).backward()
            assert second.grad is not None
            return first.grad if first.grad is not None else torch.zeros_like(first)

        expected = 2.0 * global_tensor * weights
        world = _world("tensor", ((0, 1),), 2)
        for grad in world.run(program):
            assert torch.equal(grad, expected)
        monkeypatch.setattr(fragments_module, "gather_for_edit", _detaching_gather)
        for grad in world.run(program):
            assert torch.equal(grad, torch.zeros_like(expected))

    def test_the_fragments_module_reaches_the_autograd_seam(self) -> None:
        assert fragments_module.edit_fragment is autograd_module.edit_fragment
        assert fragments_module.gather_for_edit is autograd_module.gather_for_edit


# --------------------------------------------------------------------------- #
# the context group: the pair, the faithful gather, the handoff and carry
# --------------------------------------------------------------------------- #

AXIS = "context"
ROWS, CHUNK, FEATURE = 2, 3, 4


def _seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _context_world(size: int, schedule: Schedule) -> SimulatedWorld:
    return SimulatedWorld(groups_for(size, context=size), world=size, schedule=schedule)


# --------------------------------------------------------------------------- #
# the tap pair
# --------------------------------------------------------------------------- #


def _edit_program(
    seed: int, size: int
) -> Callable[[int, Collective], dict[str, torch.Tensor]]:
    """Every rank holds its chunk of ``x`` (a leaf) and the same parameter
    ``theta``; the whole is gathered, edited by ``theta``, fragmented back,
    and the loss is a fixed weighting of the whole edited tensor — the same
    scalar on every rank, as a replicated downstream computes it."""
    x = _seeded((ROWS, CHUNK * size, FEATURE), seed)
    theta = _seeded((FEATURE, FEATURE), seed + 1)
    weights = _seeded((ROWS, CHUNK * size, FEATURE), seed + 2)

    def program(rank: int, c: Collective) -> dict[str, torch.Tensor]:
        local = x[:, rank * CHUNK : (rank + 1) * CHUNK].clone().requires_grad_(True)
        param = theta.clone().requires_grad_(True)
        whole = gather_for_edit(local, 1, AXIS, c)
        edited = torch.tanh(whole @ param)
        fragment = edit_fragment(edited, 1, AXIS, c)
        # the downstream of a tap: every rank's chunk feeds the one loss, which
        # each rank evaluates on the whole it holds
        full = gather_for_edit(fragment, 1, AXIS, c)
        loss = (full * weights).sum()
        loss.backward()
        assert local.grad is not None and param.grad is not None
        return {"x": local.grad, "theta": param.grad, "loss": loss.detach()}

    program.x, program.theta, program.weights = x, theta, weights  # type: ignore[attr-defined]
    return program


def _edit_reference(program: Callable[..., object]) -> dict[str, torch.Tensor]:
    x = program.x.clone().requires_grad_(True)  # type: ignore[attr-defined]
    theta = program.theta.clone().requires_grad_(True)  # type: ignore[attr-defined]
    loss = (torch.tanh(x @ theta) * program.weights).sum()  # type: ignore[attr-defined]
    loss.backward()
    assert x.grad is not None and theta.grad is not None
    return {"x": x.grad, "theta": theta.grad, "loss": loss.detach()}


class TestTapPair:
    @pytest.mark.property
    @pytest.mark.parametrize("size", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @example(schedule=[], seed=0)
    @_SETTINGS
    def test_the_parameter_gradient_is_the_world_one_gradient_on_every_rank(
        self, size: int, schedule: Schedule, seed: int
    ) -> None:
        program = _edit_program(seed, size)
        reference = _edit_reference(program)
        for rank, got in enumerate(_context_world(size, schedule).run(program)):
            assert torch.equal(got["loss"], reference["loss"])
            # the edit's Jacobian is the same on every rank and the fragment's
            # backward gathered every chunk's gradient: the full gradient
            assert torch.equal(got["theta"], reference["theta"]), rank
            # the whole's backward is this rank's chunk of the whole gradient
            assert torch.equal(
                got["x"], reference["x"][:, rank * CHUNK : (rank + 1) * CHUNK]
            ), rank

    @pytest.mark.unit
    def test_the_forward_is_the_plain_gather_and_chunk(self) -> None:
        x = _seeded((ROWS, CHUNK * 2, FEATURE), 3)

        def program(rank: int, c: Collective) -> None:
            local = x[:, rank * CHUNK : (rank + 1) * CHUNK]
            whole = gather_for_edit(local, 1, AXIS, c)
            assert torch.equal(whole, x)
            assert torch.equal(edit_fragment(x, 1, AXIS, c), local)
            assert torch.equal(gather_reduce_scatter(local, 1, AXIS, c), x)

        _context_world(2, 0).run(program)

    @pytest.mark.unit
    def test_a_fragment_the_group_does_not_divide_is_refused(self) -> None:
        def program(rank: int, c: Collective) -> None:
            with pytest.raises(AutogradError, match="equal chunks"):
                edit_fragment(torch.zeros(ROWS, 5, FEATURE), 1, AXIS, c)

        _context_world(2, 0).run(program)


# --------------------------------------------------------------------------- #
# the faithful gather
# --------------------------------------------------------------------------- #


def _reduce_scatter_program(
    seed: int, size: int
) -> Callable[[int, Collective], torch.Tensor]:
    """Each rank's downstream differs: its own weighting of the whole
    gathered tensor. The gradient of chunk ``j`` is the sum over ranks."""
    x = _seeded((ROWS, CHUNK * size, FEATURE), seed)
    weights = [
        _seeded((ROWS, CHUNK * size, FEATURE), seed + 1 + r) for r in range(size)
    ]

    def program(rank: int, c: Collective) -> torch.Tensor:
        local = x[:, rank * CHUNK : (rank + 1) * CHUNK].clone().requires_grad_(True)
        whole = gather_reduce_scatter(local, 1, AXIS, c)
        (torch.sin(whole) * weights[rank]).sum().backward()
        assert local.grad is not None
        return local.grad

    program.x, program.weights = x, weights  # type: ignore[attr-defined]
    return program


class TestGatherReduceScatter:
    @pytest.mark.property
    @pytest.mark.parametrize("size", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_chunk_gradient_sums_every_ranks_downstream(
        self, size: int, schedule: Schedule, seed: int
    ) -> None:
        program = _reduce_scatter_program(seed, size)
        x = program.x.clone().requires_grad_(True)  # type: ignore[attr-defined]
        sum(
            (torch.sin(x) * w).sum()
            for w in program.weights  # type: ignore[attr-defined]
        ).backward()
        assert x.grad is not None
        grads = _context_world(size, schedule).run(program)
        for rank, got in enumerate(grads):
            expected = x.grad[:, rank * CHUNK : (rank + 1) * CHUNK]
            # one fp32 sum of ``size`` terms in each of two orders
            assert torch.allclose(got, expected, atol=1e-6, rtol=0), rank

    @pytest.mark.unit
    def test_the_tap_pairs_gradient_is_not_the_faithful_one(self) -> None:
        """The mutation the design names: a downstream that differs per rank
        through the tap gather keeps only this rank's own contribution."""
        program = _reduce_scatter_program(0, 2)
        x = program.x.clone().requires_grad_(True)  # type: ignore[attr-defined]
        sum(
            (torch.sin(x) * w).sum()
            for w in program.weights  # type: ignore[attr-defined]
        ).backward()
        assert x.grad is not None

        def mutated(rank: int, c: Collective) -> torch.Tensor:
            local = (
                program.x[:, rank * CHUNK : (rank + 1) * CHUNK]  # type: ignore[attr-defined]
                .clone()
                .requires_grad_(True)
            )
            whole = gather_for_edit(local, 1, AXIS, c)
            (torch.sin(whole) * program.weights[rank]).sum().backward()  # type: ignore[attr-defined]
            assert local.grad is not None
            return local.grad

        for rank, got in enumerate(_context_world(2, 0).run(mutated)):
            expected = x.grad[:, rank * CHUNK : (rank + 1) * CHUNK]
            assert (got - expected).abs().max().item() > 1e-2, rank


# --------------------------------------------------------------------------- #
# point to point with a gradient
# --------------------------------------------------------------------------- #


def _handoff_program(
    seed: int, *, skip_backward_send: bool = False
) -> Callable[[int, Collective], torch.Tensor]:
    """Rank 0 computes ``h = f(x0)``, sends it on and keeps ``y0 = g(h)``;
    rank 1 receives ``h`` and computes ``y1 = k(h, x1)``. Each rank's loss is
    its own output's sum, so the gradient of ``x0`` needs rank 1's share of
    ``dL/dh`` added to rank 0's own."""
    x0 = _seeded((ROWS, FEATURE), seed)
    x1 = _seeded((ROWS, FEATURE), seed + 1)

    def program(rank: int, c: Collective) -> torch.Tensor:
        if rank == 0:
            x = x0.clone().requires_grad_(True)
            h = torch.tanh(x * 2.0)
            sent = send_with_grad(h, 1, AXIS, c)
            y = torch.sin(sent).sum()
            y.backward()
            assert x.grad is not None
            return x.grad
        x = x1.clone().requires_grad_(True)
        # ``skip_backward_send``: a link carrying no gradient under
        # ``IF_REQUIRED`` — the receive is recorded on nothing, so the
        # backward sends nothing; the protocol agrees (both ends
        # IF_REQUIRED) and the §3 divergence is the simulator's to refuse
        link = x.detach() if skip_backward_send else x
        h = recv_with_grad(link, (ROWS, FEATURE), torch.float32, x.device, 0, AXIS, c)
        (h * x).sum().backward()
        assert x.grad is not None
        return x.grad

    program.x0, program.x1 = x0, x1  # type: ignore[attr-defined]
    return program


class TestSendRecvWithGrad:
    @pytest.mark.property
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_senders_gradient_includes_the_peers_downstream(
        self, schedule: Schedule, seed: int
    ) -> None:
        program = _handoff_program(seed)
        x0 = program.x0.clone().requires_grad_(True)  # type: ignore[attr-defined]
        x1 = program.x1.clone().requires_grad_(True)  # type: ignore[attr-defined]
        h = torch.tanh(x0 * 2.0)
        (torch.sin(h).sum() + (h * x1).sum()).backward()
        assert x0.grad is not None and x1.grad is not None
        got0, got1 = _context_world(2, schedule).run(program)
        assert torch.equal(got1, x1.grad)
        # rank 0's own term plus the received one, one fp32 add either way
        assert torch.allclose(got0, x0.grad, atol=1e-6, rtol=0)

    @pytest.mark.unit
    def test_a_receiver_that_skips_its_backward_send_is_refused_by_name(
        self,
    ) -> None:
        """Both ends run ``IF_REQUIRED`` but read different facts — the
        receiver's link carries no gradient — so the sender waits at its
        backward receive: the rank-local branch the link's agreement cannot
        see (§3), refused by the simulator naming the parked rank."""
        with pytest.raises((Hang, Abandoned)) as err:
            _context_world(2, 1).run(_handoff_program(1, skip_backward_send=True))
        assert "rank 0" in str(err.value) and "recv" in str(err.value)

    @pytest.mark.unit
    def test_a_dependency_with_no_graph_carries_nothing(self) -> None:
        x = _seeded((ROWS, CHUNK, FEATURE), 4).requires_grad_(True)
        assert carry(x, torch.zeros(1)) is x

    @pytest.mark.unit
    def test_carry_keeps_a_peer_only_tensor_on_the_graph(self) -> None:
        """Without ``carry`` rank 0's sent tensor has no local consumer and
        its send is never reached in backward; with it the gradient arrives."""
        x0, x1 = _seeded((ROWS, FEATURE), 7), _seeded((ROWS, FEATURE), 8)

        def program(rank: int, c: Collective) -> torch.Tensor:
            if rank == 0:
                x = x0.clone().requires_grad_(True)
                out, state = torch.cos(x), torch.tanh(x)
                state = send_with_grad(state, 1, AXIS, c)
                out = carry(out, state)
                out.sum().backward()
                assert x.grad is not None
                return x.grad
            x = x1.clone().requires_grad_(True)
            state = recv_with_grad(
                x, (ROWS, FEATURE), torch.float32, x.device, 0, AXIS, c
            )
            (state * x).sum().backward()
            assert x.grad is not None
            return x.grad

        x = x0.clone().requires_grad_(True)
        (torch.cos(x).sum() + (torch.tanh(x) * x1).sum()).backward()
        assert x.grad is not None
        got0, _ = _context_world(2, 2).run(program)
        assert torch.allclose(got0, x.grad, atol=1e-6, rtol=0)

    @pytest.mark.unit
    def test_carry_passes_the_gradient_through_and_runs_the_dependencys_node(
        self,
    ) -> None:
        """What a send's backward relies on: the dependency's node runs, with
        a materialized zero gradient of its own — which it passes on, so a
        leaf under it accumulates zeros."""
        seen: list[torch.Tensor] = []

        class Probe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: object, tensor: torch.Tensor) -> torch.Tensor:
                return tensor.view_as(tensor)

            @staticmethod
            def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
                seen.append(grad)
                return grad

        x = _seeded((ROWS, FEATURE), 9).requires_grad_(True)
        leaf = torch.ones(3, requires_grad=True)
        out = carry(x * 3.0, Probe.apply(leaf))
        (out * 2.0).sum().backward()
        assert x.grad is not None and torch.equal(x.grad, torch.full_like(x, 6.0))
        assert len(seen) == 1 and torch.equal(seen[0], torch.zeros(3))
        assert leaf.grad is not None and torch.equal(leaf.grad, torch.zeros(3))


# --------------------------------------------------------------------------- #
# the pipeline boundary: the residual's send and receive, the carried loss
# --------------------------------------------------------------------------- #

CPU = torch.device("cpu")
SHAPE = (2, 3, 4)


def _weights(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(SHAPE, generator=generator)
    w0 = torch.randn(4, 4, generator=generator)
    w1 = torch.randn(4, 4, generator=generator)
    return x, w0, w1


def _world_one(seed: int) -> torch.Tensor:
    x, w0, w1 = _weights(seed)
    w0 = w0.clone().requires_grad_(True)
    loss = torch.tanh(torch.tanh(x @ w0) @ w1).sum()
    loss.backward()
    assert w0.grad is not None
    return w0.grad


def _two_stage(
    seed: int, schedule: Schedule = 0
) -> tuple[SimulatedWorld, list[torch.Tensor]]:
    """Stage 0 holds ``w0`` and sends ``tanh(x @ w0)``; stage 1 receives it,
    applies ``w1`` and the loss, and broadcasts the loss value. Every rank
    calls ``loss.backward()`` once, on the loss it holds: stage 1's is the
    real one, stage 0's the broadcast value attached to what it sent."""
    x, w0, w1 = _weights(seed)

    def share(c: Collective, value: torch.Tensor | None) -> torch.Tensor:
        # one call site for both ranks, as the stage forward's broadcast is
        return c.broadcast(value, 1, "pipeline")

    def program(rank: int, c: Collective) -> torch.Tensor:
        stage = c.rank("pipeline")
        if stage == 0:
            parameter = w0.clone().requires_grad_(True)
            hidden = torch.tanh(x @ parameter)
            sent = send_with_grad(hidden, 1, "pipeline", c)
            loss = carry(share(c, None), sent).sum()
            loss.backward()
            assert parameter.grad is not None
            return parameter.grad
        link = torch.zeros(0, requires_grad=True)
        received = recv_with_grad(link, SHAPE, x.dtype, CPU, 0, "pipeline", c)
        loss = torch.tanh(received @ w1).sum()
        share(c, loss.detach())
        loss.backward()
        return loss.detach()

    world = SimulatedWorld(groups_for(2, pipeline=2), world=2, schedule=schedule)
    return world, world.run(program)


class TestAcrossTheBoundary:
    @pytest.mark.property
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_gradient_below_the_boundary_is_the_world_one_gradient(
        self, schedule: Schedule, seed: int
    ) -> None:
        _world, (grad, _loss) = _two_stage(seed, schedule)
        assert torch.equal(grad, _world_one(seed))

    @pytest.mark.unit
    def test_the_collective_sequence_is_one_send_and_one_recv_each_way(self) -> None:
        """Per rank: the link's agreement first (one send and one receive
        of the protocol's code, shape ``(1,)``), then the forward's send /
        receive of the residual and the backward's receive / send of its
        gradient — nothing else crosses point to point."""
        world, _ = _two_stage(0)
        p2p = [
            [(e.op, e.shape) for e in ranked if e.op in ("send", "recv")]
            for ranked in world.transcripts
        ]
        code, residual = (1,), SHAPE
        assert p2p[0] == [
            ("send", code),
            ("recv", code),
            ("send", residual),
            ("recv", residual),
        ]
        assert p2p[1] == [
            ("recv", code),
            ("send", code),
            ("recv", residual),
            ("send", residual),
        ]


class _Echo:
    """A one-rank stand-in for a peer: rank 0, every send recorded by its
    destination (the protocol code's send under ``int64`` left out), every
    receive of the code answered with ``handoff``'s, every other receive
    zeros."""

    def __init__(self, handoff: Handoff, sent_to: list[int] | None = None) -> None:
        self.handoff = handoff
        self.sent_to = [] if sent_to is None else sent_to

    def rank(self, axis: str) -> int:
        return 0

    def send(self, tensor: torch.Tensor, dst: int, axis: str) -> None:
        if tensor.dtype is not torch.int64:
            self.sent_to.append(dst)

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        axis: str,
    ) -> torch.Tensor:
        if dtype is torch.int64:
            return torch.tensor([int(self.handoff)], dtype=dtype, device=device)
        return torch.zeros(shape, dtype=dtype, device=device)


@pytest.mark.unit
class TestTheFacts:
    def test_a_residual_without_gradient_still_records_the_send_node(self) -> None:
        """A stage that holds no trained parameter sends a residual that
        requires no gradient; the stage above will still send the gradient
        back, so the node that receives it must exist."""
        sent_to: list[int] = []

        hidden = torch.randn(SHAPE)
        sent = send_with_grad(
            hidden,
            1,
            "pipeline",
            _Echo(Handoff.ALWAYS, sent_to),
            handoff=Handoff.ALWAYS,
        )  # type: ignore[arg-type]
        assert sent.requires_grad and sent.grad_fn is not None
        # the default — the handoff's contract — records the node iff the
        # tensor requires a gradient, as the peer's receive sends iff its link does
        assert sent_to == [1]
        assert not send_with_grad(
            hidden, 1, "pipeline", _Echo(Handoff.IF_REQUIRED, sent_to)
        ).requires_grad  # type: ignore[arg-type]
        assert sent_to == [1, 1]  # the raw send ran either way
        assert torch.equal(sent, hidden)

    def test_nothing_is_recorded_under_no_grad(self) -> None:
        with torch.no_grad():
            sent = send_with_grad(
                torch.randn(SHAPE),
                1,
                "pipeline",
                _Echo(Handoff.ALWAYS),  # type: ignore[arg-type]
                handoff=Handoff.ALWAYS,
            )
            received = recv_with_grad(
                None,
                SHAPE,
                torch.float32,
                CPU,
                0,
                "pipeline",
                _Echo(Handoff.ALWAYS),  # type: ignore[arg-type]
                handoff=Handoff.ALWAYS,
            )
            attached = carry(torch.randn(SHAPE), sent)
        assert not sent.requires_grad and not received.requires_grad
        assert not attached.requires_grad

    def test_a_carried_value_is_the_value_and_runs_the_link_node(self) -> None:
        """The attached value is the value, bit for bit; its backward hands
        the link nothing, and the link's own node still runs — the engine
        materializes an undefined gradient as zeros — which is what lets a
        stage's ``loss.backward()`` reach the boundary through a value it
        did not compute."""
        ran: list[torch.Tensor] = []

        class _Link(torch.autograd.Function):
            @staticmethod
            def forward(ctx: object, tensor: torch.Tensor) -> torch.Tensor:
                return tensor

            @staticmethod
            def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
                ran.append(grad)
                return grad

        link = _Link.apply(torch.zeros(SHAPE, requires_grad=True))
        value = torch.randn(SHAPE)
        attached = carry(value, link)
        assert torch.equal(attached, value) and attached.requires_grad
        (attached * 3).sum().backward()
        assert len(ran) == 1 and torch.equal(ran[0], torch.zeros(SHAPE))

    def test_two_values_carried_onto_one_link_reach_it_once(self) -> None:
        """Several values carried onto one sent residual — the logits and a
        probe, say — trigger one backward at the boundary, not one per value."""
        ran: list[int] = []

        class _Link(torch.autograd.Function):
            @staticmethod
            def forward(ctx: object, tensor: torch.Tensor) -> torch.Tensor:
                return tensor

            @staticmethod
            def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
                ran.append(1)
                return grad

        link = _Link.apply(torch.zeros(SHAPE, requires_grad=True))
        loss = (
            carry(torch.randn(SHAPE), link).sum()
            + carry(torch.randn(SHAPE), link).sum()
        )
        loss.backward()
        assert ran == [1]


# --------------------------------------------------------------------------- #
# the handoff protocol: agreed once per link, a mismatch refused on both ends
# --------------------------------------------------------------------------- #


def _pipeline_world(schedule: Schedule) -> SimulatedWorld:
    return SimulatedWorld(groups_for(2, pipeline=2), world=2, schedule=schedule)


def _mismatch_program(
    *, sender: Handoff, receiver: Handoff, receiver_link_grad: bool
) -> Callable[[int, Collective], str]:
    """Stage 0 sends a residual under ``sender``; stage 1 receives under
    ``receiver`` with a link that requires a gradient or not; each end
    returns the refusal's message, or ``"agreed"`` and runs its backward."""
    x, w0, w1 = _weights(0)

    def share(c: Collective, value: torch.Tensor | None) -> torch.Tensor:
        # one call site for both ranks, as the stage forward's broadcast is
        return c.broadcast(value, 1, "pipeline")

    def program(rank: int, c: Collective) -> str:
        stage = c.rank("pipeline")
        try:
            if stage == 0:
                hidden = torch.tanh(x @ w0)  # nothing trained below: no gradient
                sent = send_with_grad(hidden, 1, "pipeline", c, handoff=sender)
                loss = carry(share(c, None), sent).sum()
                if loss.requires_grad:
                    loss.backward()
                return "agreed"
            link = torch.zeros(0, requires_grad=receiver_link_grad)
            received = recv_with_grad(
                link, SHAPE, x.dtype, CPU, 0, "pipeline", c, handoff=receiver
            )
            loss = torch.tanh(received @ w1).sum()
            share(c, loss.detach())
            if loss.requires_grad:
                loss.backward()
            return "agreed"
        except HandoffMismatch as refusal:
            return str(refusal)

    return program


class TestHandoffProtocol:
    @pytest.mark.unit
    def test_the_pipeline_protocol_carries_the_gradient_to_a_stage_that_trains_nothing(
        self,
    ) -> None:
        """``ALWAYS`` on both ends: the residual requires no gradient, the
        receiver attaches to nothing of its own, and the gradient still
        crosses back — the sequence completes on both stages."""
        results = _pipeline_world(0).run(
            _mismatch_program(
                sender=Handoff.ALWAYS, receiver=Handoff.ALWAYS, receiver_link_grad=False
            )
        )
        assert results == ["agreed", "agreed"]

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_receiver_on_the_wrong_protocol_is_refused_on_both_ends(
        self, schedule: Schedule
    ) -> None:
        """The deadlock that was: stage 0 under ``ALWAYS`` would wait in
        backward for a gradient stage 1 — under ``IF_REQUIRED`` with a link
        carrying none — never sends. Now both ends refuse by name at the
        forward's handoff, and the simulator reports no hang."""
        world = _pipeline_world(schedule)
        results = world.run(
            _mismatch_program(
                sender=Handoff.ALWAYS,
                receiver=Handoff.IF_REQUIRED,
                receiver_link_grad=False,
            )
        )
        for stage, message in enumerate(results):
            assert message != "agreed", stage
            assert "pipeline handoff stage 0 → stage 1" in message, message
            assert "ALWAYS" in message and "IF_REQUIRED" in message, message
            assert f"stage {stage} (" in message, message
        assert "the sender" in results[0] and "the receiver" in results[1]
        # the code crossed each way and nothing else did
        p2p = [e for e in world.transcript if e.op in ("send", "recv")]
        assert [e.shape for e in p2p] == [(1,)] * 4

    @pytest.mark.unit
    def test_a_sender_on_the_wrong_protocol_is_refused_on_both_ends(self) -> None:
        """The reverse: stage 1 under ``ALWAYS`` would send a gradient in
        backward that stage 0 — under ``IF_REQUIRED`` with a residual
        requiring none — never receives."""
        results = _pipeline_world(3).run(
            _mismatch_program(
                sender=Handoff.IF_REQUIRED,
                receiver=Handoff.ALWAYS,
                receiver_link_grad=True,
            )
        )
        assert all("IF_REQUIRED" in m and "ALWAYS" in m for m in results), results

    @pytest.mark.unit
    def test_an_uncaught_mismatch_is_a_rank_failure_not_a_hang(self) -> None:
        def program(rank: int, c: Collective) -> None:
            if c.rank("pipeline") == 0:
                send_with_grad(
                    torch.zeros(SHAPE), 1, "pipeline", c, handoff=Handoff.ALWAYS
                )
            else:
                recv_with_grad(
                    torch.zeros(0), SHAPE, torch.float32, CPU, 0, "pipeline", c
                )

        with pytest.raises(RankFailed) as err:
            _pipeline_world(1).run(program)
        assert isinstance(err.value.__cause__, HandoffMismatch)

    @pytest.mark.unit
    def test_a_link_agrees_once_and_later_handoffs_cost_the_tensor_alone(
        self,
    ) -> None:
        """Three graded handoffs on one link: two point-to-points for the
        agreement, then one each way per handoff, forward and backward."""
        x, _, _ = _weights(1)

        def program(rank: int, c: Collective) -> int:
            stage = c.rank("pipeline")
            for _ in range(3):
                if stage == 0:
                    sent = send_with_grad(
                        x.clone(), 1, "pipeline", c, handoff=Handoff.ALWAYS
                    )
                    (sent.sum() * 0.0).backward()
                else:
                    got = recv_with_grad(
                        None,
                        SHAPE,
                        x.dtype,
                        CPU,
                        0,
                        "pipeline",
                        c,
                        handoff=Handoff.ALWAYS,
                    )
                    got.sum().backward()
            return 0

        world = _pipeline_world(2)
        world.run(program)
        for ranked in world.transcripts:
            shapes = [e.shape for e in ranked if e.op in ("send", "recv")]
            assert shapes == [(1,), (1,)] + [SHAPE] * 6

    @pytest.mark.unit
    def test_a_second_protocol_on_an_agreed_link_is_refused_locally(self) -> None:
        """After the link agreed ``IF_REQUIRED``, an ``ALWAYS`` handoff on it
        is refused on the end that asks, before any collective."""
        x, _, _ = _weights(2)

        def program(rank: int, c: Collective) -> tuple[str, int]:
            stage = c.rank("pipeline")
            leaf = x.clone().requires_grad_(True)
            if stage == 0:
                send_with_grad(leaf, 1, "pipeline", c)
            else:
                recv_with_grad(leaf, SHAPE, x.dtype, CPU, 0, "pipeline", c)
            before = c.calls  # type: ignore[attr-defined]  # the simulator's count
            try:
                if stage == 0:
                    send_with_grad(leaf, 1, "pipeline", c, handoff=Handoff.ALWAYS)
                else:
                    recv_with_grad(
                        None,
                        SHAPE,
                        x.dtype,
                        CPU,
                        0,
                        "pipeline",
                        c,
                        handoff=Handoff.ALWAYS,
                    )
            except HandoffMismatch as refusal:
                return str(refusal), c.calls - before  # type: ignore[attr-defined]
            return "accepted", c.calls - before  # type: ignore[attr-defined]

        for message, calls in _pipeline_world(4).run(program):
            assert "agreed the IF_REQUIRED protocol" in message and "ALWAYS" in message
            assert calls == 0, "refused before any collective"

    @pytest.mark.unit
    def test_a_receive_without_a_link_under_if_required_is_refused(self) -> None:
        with pytest.raises(AutogradError, match="needs a link"):
            recv_with_grad(None, SHAPE, torch.float32, CPU, 0, "pipeline", SOLO)

    @pytest.mark.unit
    def test_the_context_axis_names_chunks(self) -> None:
        def program(rank: int, c: Collective) -> str:
            leaf = torch.ones(ROWS, FEATURE, requires_grad=True)
            try:
                if c.rank(AXIS) == 0:
                    send_with_grad(leaf, 1, AXIS, c, handoff=Handoff.ALWAYS)
                else:
                    recv_with_grad(
                        leaf, (ROWS, FEATURE), torch.float32, CPU, 0, AXIS, c
                    )
            except HandoffMismatch as refusal:
                return str(refusal)
            return "agreed"

        for message in _context_world(2, 0).run(program):
            assert "context handoff chunk 0 → chunk 1" in message, message
