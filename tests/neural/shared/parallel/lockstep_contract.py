"""The ``Lockstep`` contract as executable checks every implementation shares.

``protocol/lockstep.py``'s docstring states the contract in prose — every
per-process decision of a workflow run is made once, by the joiner, and
**agreed**: handed to every rank as one ``Outcome``, a value or the joiner's
refusal, at a named moment of a named step, through [`decide`][causalab.protocol.lockstep.decide]. This
module states each guarantee as a `Contract` — a decision program
every rank runs through ``decide`` and a verification over the per-rank
decisions in rank order — so that [`Solo`][causalab.protocol.lockstep.Solo]
(world 1), [`CollectiveLockstep`][causalab.neural.shared.parallel.lockstep.CollectiveLockstep]
over the simulator's collective (``tests/_helpers/simulated_world``) and the
same class over the production ``TorchCollective`` under ``gloo`` (the smoke
tier) are held to the *same* functions, as ``collective_contract.py`` holds
the collectives (``docs/model_parallelism.md`` §3, §10.1, §11). The
invariant every line rests on: whatever a rank's own inputs, the decision it
acts on is the joiner's.

A program is a module-level function ``(rank, lockstep, joins) -> result`` —
a spawned process imports it by name, so no closures — where ``joins`` is
whether this rank is the joiner: the rank at local index 0 on every mesh
axis (§3, ``launcher.publish_here``), so rank 0 of any layout. A
verification takes the results in rank order and the world's
[`MeshLayout`][causalab.protocol.parallel.MeshLayout], the independent statement
of who the joiner is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.lockstep import CHAIN, CollectiveLockstep
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.lockstep import (
    SOLO,
    Decided,
    Lockstep,
    LockstepError,
    LockstepRefusal,
    decide,
    encode,
)
from causalab.protocol.parallel import MeshLayout

T = TypeVar("T")

#: One rank's decision program: ``(rank, lockstep, joins) -> result``.
Program = Callable[[int, Lockstep, bool], T]

#: Runs a program on every rank of one world; results in rank order.
Runner = Callable[[Program[Any]], Sequence[Any]]


@dataclass(frozen=True)
class Contract:
    """One contract line: the decision program every rank runs, and the
    check over the results in rank order against the layout."""

    name: str
    program: Program[Any]
    verify: Callable[[Sequence[Any], MeshLayout], None]

    def check(self, run: Runner, layout: MeshLayout) -> None:
        self.verify(run(self.program), layout)


def joins_at(collective: Collective) -> bool:
    """Whether this rank is the joiner under ``collective``: local index 0
    on every mesh axis (§3) — the rule ``CollectiveLockstep`` applies to
    the outcome it is handed."""
    return all(collective.rank(axis) == 0 for axis in CHAIN)


def joiner_of(layout: MeshLayout) -> int:
    """The joiner's global rank under ``layout`` — rank 0 of any mesh."""
    return next(
        rank
        for rank in range(layout.world)
        if all(layout.rank_in(rank, axis) == 0 for axis in CHAIN)
    )


def over_collective(program: Program[T]) -> Callable[[int, Collective], T]:
    """``program`` as a rank program over a collective: the production
    lockstep built on it, the joiner found from it."""

    def rank_program(rank: int, collective: Collective) -> T:
        return program(rank, CollectiveLockstep(collective), joins_at(collective))

    return rank_program


def solo_run(program: Program[T]) -> list[T]:
    """World 1: the one rank is the joiner, over `SOLO`."""
    return [program(0, SOLO, True)]


def _marked(result: Any, kind: str, width: int, rank: int) -> tuple[Any, ...]:
    """``result`` as the ``(kind, *fields)`` marker a program returns for a
    refusal — asserted, so a program that came back with a value where a
    refusal was due fails the line rather than the unpacking."""
    assert isinstance(result, tuple) and len(result) == width and result[0] == kind, (
        rank,
        result,
    )
    return result


# --------------------------------------------------------------------------- #
# a decided value is the joiner's on every rank, whatever a rank would decide
# --------------------------------------------------------------------------- #


def _own_decision(rank: int) -> dict[str, Any]:
    """What rank ``rank`` would decide for itself — distinct per rank."""
    return {"rank": rank, "files": [f"rank{rank}.json"], "n": rank + 1}


def _value(rank: int, lockstep: Lockstep, joins: bool) -> dict[str, Any]:
    return decide(lockstep, joins, "attempt", "scan", lambda: _own_decision(rank))


def _verify_value(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    for rank, result in enumerate(results):
        assert result == _own_decision(joiner), (rank, result)


# --------------------------------------------------------------------------- #
# the joiner's refusal reaches every follower — a ProtocolError to the
# character, any other exception as a P2 naming its kind and the step
# --------------------------------------------------------------------------- #


def _refuse_p4() -> Any:
    raise ProtocolError("P4", "asks for 2 ranks", path="--parallel")


def _protocol_refusal(rank: int, lockstep: Lockstep, joins: bool) -> tuple[Any, ...]:
    try:
        decide(lockstep, joins, "turn", "scan", _refuse_p4)
    except LockstepRefusal as refusal:  # a follower: the joiner's, re-rendered
        return (
            "LockstepRefusal",
            refusal.code,
            refusal.path,
            refusal.message,
            str(refusal),
        )
    except ProtocolError as refusal:  # the joiner: its own, as itself
        return (
            type(refusal).__name__,
            refusal.code,
            refusal.path,
            refusal.message,
            str(refusal),
        )
    return ("accepted",)


def _verify_protocol_refusal(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    expected = (
        "P4",
        "--parallel",
        "asks for 2 ranks",
        "[P4] at --parallel asks for 2 ranks",
    )
    for rank, result in enumerate(results):
        kind = "ProtocolError" if rank == joiner else "LockstepRefusal"
        assert _marked(result, kind, 5, rank)[1:] == expected, (rank, result)


def _fail_plainly() -> Any:
    raise RuntimeError("the disk is full")


def _plain_refusal(rank: int, lockstep: Lockstep, joins: bool) -> tuple[Any, ...]:
    try:
        decide(lockstep, joins, "attempt", "fit", _fail_plainly)
    except LockstepRefusal as refusal:
        return (
            "LockstepRefusal",
            refusal.code,
            refusal.kind,
            refusal.step,
            str(refusal),
        )
    except RuntimeError as error:
        return ("RuntimeError", str(error))
    return ("accepted",)


def _verify_plain_refusal(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    for rank, result in enumerate(results):
        if rank == joiner:
            assert result == ("RuntimeError", "the disk is full"), (rank, result)
            continue
        _, code, error_kind, step, text = _marked(result, "LockstepRefusal", 5, rank)
        assert (code, error_kind, step) == ("P2", "RuntimeError", "fit"), (rank, result)
        assert "step 'fit'" in text and "the disk is full" in text, (rank, text)


# --------------------------------------------------------------------------- #
# the moments of a run, in order, stay in step
# --------------------------------------------------------------------------- #


def _run(rank: int, lockstep: Lockstep, joins: bool) -> list[Any]:
    out: list[Any] = []
    for step in ("scan", "fit"):
        out.append(decide(lockstep, joins, "turn", step, lambda: f"attempt {rank}"))
        out.append(
            decide(
                lockstep,
                joins,
                "attempt",
                step,
                lambda: {"status": "completed", "step": step, "by": rank},
            )
        )
    out.append(
        decide(lockstep, joins, "manifest", None, lambda: {"steps": ["scan", "fit"]})
    )
    return out


def _verify_run(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    expected = [
        f"attempt {joiner}",
        {"status": "completed", "step": "scan", "by": joiner},
        f"attempt {joiner}",
        {"status": "completed", "step": "fit", "by": joiner},
        {"steps": ["scan", "fit"]},
    ]
    for rank, result in enumerate(results):
        assert result == expected, (rank, result)


# --------------------------------------------------------------------------- #
# a follower's collective half runs before it takes the value
# --------------------------------------------------------------------------- #


def _follow_first(rank: int, lockstep: Lockstep, joins: bool) -> tuple[list[str], Any]:
    log: list[str] = []

    def decision() -> str:
        log.append("decide")
        return "value"

    value = decide(
        lockstep,
        joins,
        "attempt",
        "scan",
        decision,
        follow=lambda: log.append("follow"),
    )
    return log, value


def _verify_follow_first(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    for rank, result in enumerate(results):
        assert isinstance(result, tuple) and len(result) == 2, (rank, result)
        log, value = result
        assert value == "value", (rank, value)
        assert log == (["decide"] if rank == joiner else ["follow"]), (rank, log)


# --------------------------------------------------------------------------- #
# a follower awaiting another moment is out of lockstep; the joiner's
# manifest refusal stops a follower awaiting any step
# --------------------------------------------------------------------------- #


def _out_of_step(rank: int, lockstep: Lockstep, joins: bool) -> Any:
    if joins:
        return decide(lockstep, True, "turn", "scan", lambda: "attempt")
    try:
        return decide(lockstep, False, "attempt", "scan", lambda: None)
    except LockstepError as error:
        return ("LockstepError", error.code, str(error))


def _verify_out_of_step(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    for rank, result in enumerate(results):
        if rank == joiner:
            assert result == "attempt", (rank, result)
            continue
        _, code, text = _marked(result, "LockstepError", 3, rank)
        assert code == "P2", (rank, result)
        assert "awaited attempt of step 'scan'" in text, text
        assert "agreed turn of step 'scan'" in text, text
        assert "out of lockstep" in text, text


def _no_room() -> Any:
    raise RuntimeError("no room for the manifest")


def _manifest_refusal(rank: int, lockstep: Lockstep, joins: bool) -> Any:
    if joins:
        try:
            decide(lockstep, True, "manifest", None, _no_room)
        except RuntimeError:
            return "raised"
        return "accepted"
    try:
        return decide(lockstep, False, "turn", "fit", lambda: None)
    except LockstepRefusal as refusal:
        return (
            "LockstepRefusal",
            refusal.code,
            refusal.step,
            refusal.kind,
            str(refusal),
        )


def _verify_manifest_refusal(results: Sequence[Any], layout: MeshLayout) -> None:
    joiner = joiner_of(layout)
    for rank, result in enumerate(results):
        if rank == joiner:
            assert result == "raised", (rank, result)
            continue
        _, code, step, error_kind, text = _marked(result, "LockstepRefusal", 5, rank)
        assert (code, step, error_kind) == ("P2", None, "RuntimeError"), (rank, result)
        assert "the manifest" in text and "no room for the manifest" in text, text


# --------------------------------------------------------------------------- #
# the joiner's refusal at another moment of another step is drift, not this
# rank's refusal: the follower is told the ranks are out of lockstep
# --------------------------------------------------------------------------- #


def _refuse_the_fits_turn() -> Any:
    raise RuntimeError("the fit's turn was refused")


def _refusal_at_another_moment(rank: int, lockstep: Lockstep, joins: bool) -> Any:
    """The joiner refuses at ``turn`` of ``fit`` while every follower awaits
    ``attempt`` of ``scan`` — the third refusal branch of ``decide``: neither
    this rank's moment nor the manifest, so the ranks have drifted, and what
    the follower reads is the drift naming both moments, not a refusal
    about someone else's step handed over as its own."""
    if joins:
        try:
            decide(lockstep, True, "turn", "fit", _refuse_the_fits_turn)
        except RuntimeError:
            return "raised"
        return "accepted"
    try:
        return decide(lockstep, False, "attempt", "scan", lambda: None)
    except LockstepRefusal as refusal:
        return ("LockstepRefusal", refusal.code, str(refusal))
    except LockstepError as error:
        return ("LockstepError", error.code, str(error))


def _verify_refusal_at_another_moment(
    results: Sequence[Any], layout: MeshLayout
) -> None:
    joiner = joiner_of(layout)
    for rank, result in enumerate(results):
        if rank == joiner:
            assert result == "raised", (rank, result)
            continue
        _, code, text = _marked(result, "LockstepError", 3, rank)
        assert code == "P2", (rank, result)
        assert "awaited attempt of step 'scan'" in text, text
        assert "agreed turn of step 'fit'" in text, text
        assert "out of lockstep" in text, text
        assert "the fit's turn was refused" not in text, text


# --------------------------------------------------------------------------- #
# JSON travels to the character
# --------------------------------------------------------------------------- #

WIRE_VALUE: dict[str, Any] = {
    "text": 'naïve — ünïcödé ✓ \\ " /',
    "float": 0.1,
    "big": 2**53 + 1,
    "negative": -7,
    "none": None,
    "bools": [True, False],
    "nested": {"k": [1, [2, [3, {"deep": "end"}]]], "": []},
}


def _wire(rank: int, lockstep: Lockstep, joins: bool) -> tuple[Any, bytes]:
    value = decide(lockstep, joins, "attempt", "fit", lambda: WIRE_VALUE)
    return value, encode(Decided("attempt", "fit", value))


def _verify_wire(results: Sequence[Any], layout: MeshLayout) -> None:
    expected = encode(Decided("attempt", "fit", WIRE_VALUE))
    for rank, result in enumerate(results):
        assert isinstance(result, tuple) and len(result) == 2, (rank, result)
        value, encoded = result
        assert value == WIRE_VALUE, (rank, value)
        assert encoded == expected, (rank, encoded)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #

CONTRACTS: tuple[Contract, ...] = (
    Contract("a_decided_value_is_the_joiners_on_every_rank", _value, _verify_value),
    Contract(
        "a_protocol_refusal_re_renders_to_the_character",
        _protocol_refusal,
        _verify_protocol_refusal,
    ),
    Contract(
        "a_plain_exception_on_the_joiner_is_a_p2_naming_kind_and_step",
        _plain_refusal,
        _verify_plain_refusal,
    ),
    Contract("the_moments_of_a_run_stay_in_step", _run, _verify_run),
    Contract(
        "a_followers_collective_half_runs_before_it_takes_the_value",
        _follow_first,
        _verify_follow_first,
    ),
    Contract(
        "a_follower_awaiting_another_moment_is_out_of_lockstep",
        _out_of_step,
        _verify_out_of_step,
    ),
    Contract(
        "the_joiners_manifest_refusal_stops_a_follower_awaiting_a_step",
        _manifest_refusal,
        _verify_manifest_refusal,
    ),
    Contract(
        "a_refusal_at_another_moment_is_out_of_lockstep_not_this_ranks_refusal",
        _refusal_at_another_moment,
        _verify_refusal_at_another_moment,
    ),
    Contract("json_travels_to_the_character", _wire, _verify_wire),
)

BY_NAME: dict[str, Contract] = {contract.name: contract for contract in CONTRACTS}


def every_contract(rank: int, collective: Collective) -> dict[str, Any]:
    """Every contract's program in one rank program over a collective — so a
    spawned world runs the whole suite in one launch; `verify_all` or
    a single ``Contract.verify`` over `results_of` checks it."""
    lockstep = CollectiveLockstep(collective)
    joins = joins_at(collective)
    return {
        contract.name: contract.program(rank, lockstep, joins) for contract in CONTRACTS
    }


def results_of(name: str, results: Sequence[dict[str, Any]]) -> list[Any]:
    """One contract's per-rank results out of `every_contract`'s."""
    return [result[name] for result in results]


def verify_all(results: Sequence[dict[str, Any]], layout: MeshLayout) -> None:
    for contract in CONTRACTS:
        contract.verify(results_of(contract.name, results), layout)
