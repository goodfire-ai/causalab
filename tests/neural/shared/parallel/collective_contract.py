"""The ``Collective`` contract as executable checks every implementation shares.

``collective.py``'s docstring states the contract in prose, one line per
guarantee; this module states each line as a `Contract` — a rank
program every rank runs and a verification over the results in rank order —
so that [`Solo`][causalab.neural.shared.parallel.collective.Solo] (world 1),
the simulator's ``RankCollective`` (``tests/_helpers/simulated_world``) and
the production [`TorchCollective`][causalab.neural.shared.parallel.collective.TorchCollective]
(under ``gloo``, the smoke tier) are held to the *same* functions. If the
simulator and torch disagree here, the simulator is wrong: it is the
reference the deterministic-simulation tiers run everything else against
(``docs/model_parallelism.md`` §10.2, §10.6).

The ``gradient`` lines hold the autograd Functions of ``parallel/autograd.py``
over the *raw* collectives to a world-1 oracle: the gradient of a fixed
scalar loss through each pair, computed per rank, equals torch's own gradient
of the same loss on the unsharded tensors (the losses are sums of products of
small multiples of a quarter, exact in fp32 in any order). Every
implementation runs them because the Functions' backward calls the raw
collectives, and a collective that behaved differently in backward — a
gather off the graph is the production behaviour the Functions exist to
wrap — would show here first. The ``handoff`` protocol line holds the
link's agreement (§7, ``autograd.Handoff``) to one wire behaviour: the two
ends of a point-to-point handoff that pick different gradient protocols
are refused by name on **both** ends, before any tensor crosses, on every
implementation — where a deadlock in backward would otherwise follow.

A program is a module-level function ``(rank, collective) -> result`` — a
spawned process imports it by name, so no closures — that walks every axis,
so a group of size one (the identity) and a group of many are exercised by
one program. A verification takes the results in rank order and the
[`MeshLayout`][causalab.protocol.parallel.MeshLayout] of the world, the independent
statement of who is in which group.

Refusals the contract shares are the ones raised *inside* the calling rank
before any collective is entered — a peer naming the rank itself or outside
the group, a source passing ``None``, a non-source passing a tensor — so a
program can observe them and return a marker. Every implementation raises a
``ValueError`` subclass for these. A shape or dtype divergence between
members is refused by the simulator at ``run`` and by torch inside the call,
so that refusal is checked per implementation, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

import torch

from causalab.neural.shared.parallel.autograd import (
    Handoff,
    HandoffMismatch,
    edit_fragment,
    edit_summand,
    gather_for_edit,
    gather_reduce_scatter,
    recv_with_grad,
    send_with_grad,
    sum_for_edit,
)
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import AXES, Axis
from causalab.protocol.parallel import MeshLayout

T = TypeVar("T")

#: One rank's program.
Program = Callable[[int, Collective], T]

#: Runs a program on every rank of one world; results in rank order.
Runner = Callable[[Program[Any]], Sequence[Any]]

#: A program's marker for a call the collective refused.
REFUSED = "refused"


@dataclass(frozen=True)
class Contract:
    """One contract line: the program every rank runs, and the check over
    the results in rank order against the layout."""

    name: str
    program: Program[Any]
    verify: Callable[[Sequence[Any], MeshLayout], None]

    def check(self, run: Runner, layout: MeshLayout) -> None:
        self.verify(run(self.program), layout)


# --------------------------------------------------------------------------- #
# the values a rank contributes — distinct per rank, exact in fp32 and fp16
# --------------------------------------------------------------------------- #


def piece(rank: int) -> torch.Tensor:
    """A ``(2, 3)`` fp32 tensor whose every entry names ``rank``."""
    return torch.arange(6, dtype=torch.float32).reshape(2, 3) + 10.0 * rank


def message(rank: int) -> torch.Tensor:
    """An fp16 tensor whose **shape** depends on ``rank`` — a receiver that
    guessed the shape instead of learning it from the source gets it wrong."""
    rows = rank % 3 + 1
    return torch.arange(rows * 2, dtype=torch.float16).reshape(rows, 2) + rank


def word(rank: int) -> torch.Tensor:
    """An int64 vector — a second dtype for the broadcast to carry."""
    return torch.tensor([rank, rank + 1, -rank], dtype=torch.int64)


# --------------------------------------------------------------------------- #
# all_gather concatenates in rank order along ``dim``; every member the same
# --------------------------------------------------------------------------- #


def _gather(rank: int, c: Collective) -> dict[Axis, tuple[torch.Tensor, torch.Tensor]]:
    return {
        axis: (c.all_gather(piece(rank), 0, axis), c.all_gather(piece(rank), 1, axis))
        for axis in AXES
    }


def _verify_gather(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            for dim, got in zip((0, 1), result[axis]):
                expected = torch.cat([piece(r) for r in group], dim=dim)
                assert torch.equal(got, expected), (rank, axis, dim)


def _gather_transposed(rank: int, c: Collective) -> dict[Axis, torch.Tensor]:
    # ``.t()`` is a non-contiguous view; the collective must not require
    # callers to make tensors contiguous first
    return {axis: c.all_gather(piece(rank).t(), 0, axis) for axis in AXES}


def _verify_gather_transposed(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            expected = torch.cat([piece(r).t() for r in group], dim=0)
            assert torch.equal(result[axis], expected), (rank, axis)


# --------------------------------------------------------------------------- #
# all_reduce_sum returns a new tensor equal on every member; no mutation
# --------------------------------------------------------------------------- #


def _reduce(rank: int, c: Collective) -> dict[Axis, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[Axis, tuple[torch.Tensor, torch.Tensor]] = {}
    for axis in AXES:
        mine = piece(rank)
        total = c.all_reduce_sum(mine, axis)
        out[axis] = (total, mine)  # ``mine`` after the call: must be untouched
    return out


def _verify_reduce(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            total, mine = result[axis]
            expected = piece(group[0]).clone()
            for r in group[1:]:
                expected = expected + piece(r)
            assert torch.equal(total, expected), (rank, axis)
            assert torch.equal(mine, piece(rank)), (rank, axis, "argument mutated")


# --------------------------------------------------------------------------- #
# broadcast: non-sources pass None and receive shape, dtype and values from
# the source; the source receives its own tensor back
# --------------------------------------------------------------------------- #


def _broadcast(
    rank: int, c: Collective
) -> dict[Axis, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[Axis, tuple[torch.Tensor, torch.Tensor]] = {}
    for axis in AXES:
        last = c.size(axis) - 1
        # from the last member, an fp16 tensor whose shape only the source knows
        from_last = c.broadcast(
            message(rank) if c.rank(axis) == last else None, last, axis
        )
        # from the first member, an int64 vector
        from_first = c.broadcast(word(rank) if c.rank(axis) == 0 else None, 0, axis)
        out[axis] = (from_last, from_first)
    return out


def _verify_broadcast(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            from_last, from_first = result[axis]
            expected = message(group[-1])
            assert from_last.dtype == expected.dtype, (rank, axis)
            assert from_last.shape == expected.shape, (rank, axis)
            assert torch.equal(from_last, expected), (rank, axis)
            assert from_first.dtype == torch.int64, (rank, axis)
            assert torch.equal(from_first, word(group[0])), (rank, axis)


# --------------------------------------------------------------------------- #
# send / recv: point to point between two ranks of one axis, group-local peers
# --------------------------------------------------------------------------- #


def _send_recv(rank: int, c: Collective) -> dict[Axis, torch.Tensor | None]:
    # even members send to the next member, odd members receive from the
    # previous one; a trailing even member of an odd group sits out
    out: dict[Axis, torch.Tensor | None] = {}
    for axis in AXES:
        local, size = c.rank(axis), c.size(axis)
        if local % 2 == 0 and local + 1 < size:
            c.send(piece(rank), local + 1, axis)
            out[axis] = None
        elif local % 2 == 1:
            out[axis] = c.recv(
                (2, 3), torch.float32, torch.device("cpu"), local - 1, axis
            )
        else:
            out[axis] = None
    return out


def _verify_send_recv(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            local = group.index(rank)
            got = result[axis]
            if local % 2 == 1:
                assert got is not None, (rank, axis)
                assert torch.equal(got, piece(group[local - 1])), (rank, axis)
            else:
                assert got is None, (rank, axis)


# --------------------------------------------------------------------------- #
# the three agreements are the group's min / any / sum, as Python scalars
# --------------------------------------------------------------------------- #


def _agreements(rank: int, c: Collective) -> dict[Axis, tuple[int, bool, int]]:
    return {
        axis: (
            c.agree_min(-rank, axis),
            c.agree_any(rank % 2 == 1, axis),
            c.agree_sum(rank + 1, axis),
        )
        for axis in AXES
    }


def _verify_agreements(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            least, any_odd, total = result[axis]
            assert type(least) is int and least == min(-r for r in group), (rank, axis)
            assert type(any_odd) is bool and any_odd == any(
                r % 2 == 1 for r in group
            ), (
                rank,
                axis,
            )
            assert type(total) is int and total == sum(r + 1 for r in group), (
                rank,
                axis,
            )


# --------------------------------------------------------------------------- #
# barrier completes; rank / size describe the position
# --------------------------------------------------------------------------- #


def _barrier(rank: int, c: Collective) -> dict[Axis, bool]:
    out: dict[Axis, bool] = {}
    for axis in AXES:
        c.barrier(axis)
        out[axis] = True
    return out


def _verify_barrier(results: Sequence[Any], layout: MeshLayout) -> None:
    assert len(results) == layout.world
    for result in results:
        assert all(result[axis] for axis in AXES)


def _position(rank: int, c: Collective) -> dict[Axis, tuple[int, int]]:
    return {axis: (c.rank(axis), c.size(axis)) for axis in AXES}


def _verify_position(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            assert result[axis] == (group.index(rank), len(group)), (rank, axis)


# --------------------------------------------------------------------------- #
# refusals raised inside the caller, before any collective is entered
# --------------------------------------------------------------------------- #


def _refusing(call: Callable[[], object]) -> str:
    try:
        call()
    except ValueError:
        return REFUSED
    return "accepted"


def _refusals(rank: int, c: Collective) -> dict[Axis, tuple[str, ...]]:
    out: dict[Axis, tuple[str, ...]] = {}
    axis: Axis  # declared, so the lambdas below see the literal type, not str
    for axis in AXES:
        me, size = c.rank(axis), c.size(axis)
        markers = [
            _refusing(lambda: c.send(piece(rank), me, axis)),  # to itself
            _refusing(
                lambda: c.recv((2, 3), torch.float32, torch.device("cpu"), me, axis)
            ),
            _refusing(lambda: c.send(piece(rank), size, axis)),  # outside the group
            _refusing(lambda: c.broadcast(None, me, axis)),  # the source passing None
        ]
        if size > 1:
            other = (me + 1) % size
            # a non-source passing a tensor: shape and dtype travel from the source
            markers.append(_refusing(lambda: c.broadcast(piece(rank), other, axis)))
        out[axis] = tuple(markers)
    return out


def _verify_refusals(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            expected = 5 if len(layout.group_of(rank, axis)) > 1 else 4
            assert result[axis] == (REFUSED,) * expected, (rank, axis, result[axis])


# --------------------------------------------------------------------------- #
# gradient: the autograd pairs over the raw collectives equal the world-1
# oracle on every rank (docs/model_parallelism.md §7)
# --------------------------------------------------------------------------- #


def quarter(rank: int) -> torch.Tensor:
    """A ``(2, 3)`` fp32 tensor of quarters naming ``rank``: products and
    sums of a few of these are exact whatever the order."""
    return (torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2.0) * 0.25 + rank


def theta(size: int) -> torch.Tensor:
    """The replicated parameter over ``size`` gathered pieces."""
    return (torch.arange(6 * size, dtype=torch.float32).reshape(2 * size, 3) % 5) * 0.25


def owned(rank: int, size: int) -> torch.Tensor:
    """A partition of a ``(2, 3)`` tensor's entries over ``size`` ranks."""
    return (torch.arange(6).reshape(2, 3) % size) == rank


def _replicated(local_loss: torch.Tensor, c: Collective, axis: Axis) -> torch.Tensor:
    """The loss every rank computes: the local terms summed through the
    model's own reduction (an all-reduce whose backward is the identity)."""
    return sum_for_edit(local_loss, axis, c) if c.size(axis) > 1 else local_loss


def _gather_pair(
    rank: int, c: Collective
) -> dict[Axis, tuple[torch.Tensor, torch.Tensor]]:
    """``whole`` the piece, weigh by ``Θ``, ``fragment`` back, weigh by this
    rank's piece of the weights, sum: ``(∂L/∂Θ, ∂L/∂x_r)``."""
    out: dict[Axis, tuple[torch.Tensor, torch.Tensor]] = {}
    for axis in AXES:
        size = c.size(axis)
        x = quarter(rank).requires_grad_(True)
        th = theta(size).requires_grad_(True)
        edited = gather_for_edit(x, 0, axis, c) * th
        piece = edit_fragment(edited, 0, axis, c)
        _replicated((piece * (quarter(rank) + 1.0)).sum(), c, axis).backward()
        assert th.grad is not None and x.grad is not None
        out[axis] = (th.grad, x.grad)
    return out


def _verify_gather_pair(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            local = group.index(rank)
            whole = torch.cat([quarter(r) for r in group], dim=0)
            weights = torch.cat([quarter(r) + 1.0 for r in group], dim=0)
            th_grad, x_grad = result[axis]
            # the full gradient on every rank; this rank's slice for its piece
            assert torch.equal(th_grad, whole * weights), (rank, axis, "Θ")
            expected_x = (theta(len(group)) * weights).narrow(0, 2 * local, 2)
            assert torch.equal(x_grad, expected_x), (rank, axis, "x_r")


def _summand_pair(
    rank: int, c: Collective
) -> dict[Axis, tuple[torch.Tensor, torch.Tensor]]:
    """The sum pair: ``whole`` the owned entries (zeros elsewhere), weigh by
    ``Θ``, keep the owned entries of the edit, weigh, sum."""
    out: dict[Axis, tuple[torch.Tensor, torch.Tensor]] = {}
    for axis in AXES:
        size, local = c.size(axis), c.rank(axis)
        keep = owned(local, size)
        x = torch.where(keep, quarter(rank), torch.zeros(())).requires_grad_(True)
        th = theta(1).requires_grad_(True)
        edited = sum_for_edit(x, axis, c) * th
        piece = edit_summand(edited, keep, axis, c)
        _replicated((piece * (quarter(rank) + 1.0)).sum(), c, axis).backward()
        assert th.grad is not None and x.grad is not None
        out[axis] = (th.grad, x.grad)
    return out


def _verify_summand_pair(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            size = len(group)
            whole = torch.zeros(2, 3)
            weights = torch.zeros(2, 3)
            for local, member in enumerate(group):
                keep = owned(local, size)
                whole = whole + torch.where(keep, quarter(member), torch.zeros(()))
                weights = weights + torch.where(
                    keep, quarter(member) + 1.0, torch.zeros(())
                )
            th_grad, x_grad = result[axis]
            assert torch.equal(th_grad, whole * weights), (rank, axis, "Θ")
            # the all-reduce's local Jacobian is the identity: the whole gradient
            assert torch.equal(x_grad, theta(1) * weights), (rank, axis, "x_r")


def _faithful_pair(rank: int, c: Collective) -> dict[Axis, torch.Tensor]:
    """A gather whose consumer differs per rank: ``∂L/∂x_r`` is this rank's
    slice of every rank's partial summed."""
    out: dict[Axis, torch.Tensor] = {}
    for axis in AXES:
        x = quarter(rank).requires_grad_(True)
        whole = gather_reduce_scatter(x, 0, axis, c)
        # this rank's own weights over the whole: a consumer that differs per rank
        _replicated((whole * (theta(c.size(axis)) + rank)).sum(), c, axis).backward()
        assert x.grad is not None
        out[axis] = x.grad
    return out


def _verify_faithful_pair(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            local = group.index(rank)
            total = torch.zeros(2 * len(group), 3)
            for member in group:
                total = total + (theta(len(group)) + member)
            assert torch.equal(result[axis], total.narrow(0, 2 * local, 2)), (
                rank,
                axis,
            )


def _handoff_pair(rank: int, c: Collective) -> dict[Axis, torch.Tensor | None]:
    """Even members send ``x ⊙ Θ`` to the next member, which computes the
    loss; the sender's ``Θ`` receives the gradient sent back (``None`` on
    receivers and a trailing even member)."""
    out: dict[Axis, torch.Tensor | None] = {}
    for axis in AXES:
        local, size = c.rank(axis), c.size(axis)
        th = torch.ones(2, 3, requires_grad=True)
        if local % 2 == 0 and local + 1 < size:
            link = send_with_grad(quarter(rank) * th, local + 1, axis, c)
            (link.sum() * 0.0).backward()  # nothing continues locally
            out[axis] = th.grad
        elif local % 2 == 1:
            received = recv_with_grad(
                th, (2, 3), torch.float32, torch.device("cpu"), local - 1, axis, c
            )
            (received * quarter(rank)).sum().backward()
            out[axis] = None
        else:
            out[axis] = None
    return out


def _verify_handoff_pair(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            local = group.index(rank)
            got = result[axis]
            if local % 2 == 0 and local + 1 < len(group):
                assert got is not None, (rank, axis)
                assert torch.equal(got, quarter(rank) * quarter(group[local + 1])), (
                    rank,
                    axis,
                )
            else:
                assert got is None, (rank, axis)


def _handoff_mismatch(rank: int, c: Collective) -> dict[Axis, tuple[str, str] | None]:
    """Odd members send to the member below under ``ALWAYS``; that member
    receives under ``IF_REQUIRED`` — the link ``odd → even``, untouched by
    `_handoff_pair`, agrees for the first time here and both ends are
    refused: ``(REFUSED, message)``; ``None`` where a member has no pair."""
    out: dict[Axis, tuple[str, str] | None] = {}
    for axis in AXES:
        local, size = c.rank(axis), c.size(axis)
        th = torch.ones(2, 3, requires_grad=True)
        try:
            if local % 2 == 1:
                send_with_grad(
                    quarter(rank), local - 1, axis, c, handoff=Handoff.ALWAYS
                )
            elif local + 1 < size:
                recv_with_grad(
                    th, (2, 3), torch.float32, torch.device("cpu"), local + 1, axis, c
                )
            else:
                out[axis] = None
                continue
        except HandoffMismatch as refusal:
            out[axis] = (REFUSED, str(refusal))
        else:
            out[axis] = ("accepted", "")
    return out


def _verify_handoff_mismatch(results: Sequence[Any], layout: MeshLayout) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            local = group.index(rank)
            got = result[axis]
            paired = local % 2 == 1 or local + 1 < len(group)
            if not paired:
                assert got is None, (rank, axis)
                continue
            assert got is not None and got[0] == REFUSED, (rank, axis, got)
            message = got[1]
            sender, receiver = (
                (local, local - 1) if local % 2 == 1 else (local + 1, local)
            )
            assert f"{axis} handoff" in message, (rank, axis, message)
            assert (
                f"{sender} → " in message
                and f"→ {'stage' if axis == 'pipeline' else 'chunk' if axis == 'context' else 'rank'} {receiver}"
                in message
            ), (rank, axis, message)
            assert "ALWAYS" in message and "IF_REQUIRED" in message, (
                rank,
                axis,
                message,
            )


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #

CONTRACTS: tuple[Contract, ...] = (
    Contract("all_gather_in_rank_order", _gather, _verify_gather),
    Contract(
        "all_gather_of_a_non_contiguous_view",
        _gather_transposed,
        _verify_gather_transposed,
    ),
    Contract("all_reduce_sum_is_a_new_equal_tensor", _reduce, _verify_reduce),
    Contract("broadcast_carries_shape_and_dtype", _broadcast, _verify_broadcast),
    Contract("send_recv_between_group_local_peers", _send_recv, _verify_send_recv),
    Contract("agreements_are_python_scalars", _agreements, _verify_agreements),
    Contract("barrier_completes", _barrier, _verify_barrier),
    Contract("rank_and_size_describe_the_position", _position, _verify_position),
    Contract("misuse_is_refused_before_any_collective", _refusals, _verify_refusals),
    Contract("gradient_of_the_gather_pair", _gather_pair, _verify_gather_pair),
    Contract("gradient_of_the_summand_pair", _summand_pair, _verify_summand_pair),
    Contract("gradient_of_the_faithful_pair", _faithful_pair, _verify_faithful_pair),
    Contract("gradient_of_the_handoff_pair", _handoff_pair, _verify_handoff_pair),
    Contract(
        "handoff_protocol_mismatch_is_refused_on_both_ends",
        _handoff_mismatch,
        _verify_handoff_mismatch,
    ),
)

BY_NAME: dict[str, Contract] = {contract.name: contract for contract in CONTRACTS}


def every_contract(rank: int, c: Collective) -> dict[str, Any]:
    """Every contract's program in one rank program — so a spawned world runs
    the whole suite in one launch; `verify_all` or a single
    ``Contract.verify`` over `results_of` checks it."""
    return {contract.name: contract.program(rank, c) for contract in CONTRACTS}


def results_of(name: str, results: Sequence[dict[str, Any]]) -> list[Any]:
    """One contract's per-rank results out of `every_contract`'s."""
    return [result[name] for result in results]


def verify_all(results: Sequence[dict[str, Any]], layout: MeshLayout) -> None:
    for contract in CONTRACTS:
        contract.verify(results_of(contract.name, results), layout)
