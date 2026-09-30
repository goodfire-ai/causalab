"""The lockstep of a launched world's per-process decisions
(``docs/model_parallelism.md`` §3, §11; workflow spec §8) — the torch-free
half, beside [`causalab.protocol.publish`][].

Under SPMD every rank runs the same steps over the same Python control flow
and the engine's collectives inside each step keep the ranks together. What
a workflow run decides **per process** — whether a published step is reused
(``--resume`` reads the run tree), a step's record once its attempt is
verified and published, the manifest, the pins stamp — is read from or
written to a filesystem exactly one rank writes: the joiner
([`is_joiner`][causalab.protocol.publish.is_joiner]). A rank that read the tree
for itself would race the joiner's renames and could reach a different
verdict — run a step the joiner reused — and the ranks would then meet at
different collectives and hang.

So each such decision is made once, by the joiner, and **agreed**: handed
to every rank as one [`Outcome`][] through the [`Lockstep`][] seam.
[`Decided`][] carries the value (JSON: a step record, a manifest); a
[`Refused`][] carries the joiner's exception — for a
[`ProtocolError`][] its code, path and bare
message, so a follower re-renders the refusal to the character — and every
outcome names its [`MOMENTS`][] and step, so a follower awaiting one
decision never reads another as it (the joiner that died between two
agreements and went to its manifest is reported as that, not as a step
record). [`decide`][] is the one entry: the joiner runs the decision and
agrees its value, or agrees its refusal *before* re-raising; a follower
runs its own collective half of the step first (the engine's requests, whose
collectives the joiner's engine meets), then takes the value or raises the
joiner's refusal as [`LockstepRefusal`][].

[`SOLO`][] is world 1: the identity, today's path. The production lockstep
over the mesh's process groups is
``causalab/neural/shared/parallel/lockstep.py``; the seam stays free of torch
so the workflow runner needs none.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "MOMENTS",
    "SOLO",
    "Decided",
    "Lockstep",
    "LockstepError",
    "LockstepRefusal",
    "Outcome",
    "Refused",
    "Solo",
    "decide",
    "decode",
    "encode",
    "refusal_of",
]

#: The closed vocabulary of what is agreed, in the order a run meets them:
#: ``turn`` — a step's turn has come: reused, or to be attempted;
#: ``attempt`` — the step's verified, published record; ``manifest`` — the
#: run's manifest, written.
MOMENTS: tuple[str, ...] = ("turn", "attempt", "manifest")

T = TypeVar("T")


def _check_moment(moment: str) -> None:
    if moment not in MOMENTS:
        raise ValueError(f"moment {moment!r} is not one of {MOMENTS}")


@dataclasses.dataclass(frozen=True)
class Decided:
    """The joiner's value at ``moment`` of ``step`` (``None`` for a moment
    of the run): JSON, as it travels.

    Raises:
        ValueError: a moment outside [`MOMENTS`][].
    """

    moment: str
    step: str | None
    value: Any

    def __post_init__(self) -> None:
        _check_moment(self.moment)


@dataclasses.dataclass(frozen=True)
class Refused:
    """The joiner's exception at ``moment`` of ``step``: its type name and
    text; for a [`ProtocolError`][] its
    ``code``, ``path`` and bare ``message`` too, so the follower renders the
    same ``[code] at path message``.

    Raises:
        ValueError: a moment outside [`MOMENTS`][].
    """

    moment: str
    step: str | None
    kind: str
    message: str
    code: str | None
    path: str | None

    def __post_init__(self) -> None:
        _check_moment(self.moment)


Outcome = Decided | Refused


class LockstepError(ProtocolError):
    """The lockstep itself is broken: a payload that is not an outcome, a
    follower's call at world 1, a moment or step the follower did not await
    (``P2``)."""

    def __init__(self, message: str) -> None:
        super().__init__("P2", message)


class LockstepRefusal(ProtocolError):
    """The joiner's refusal, raised on a follower: the same code, path and
    message when the joiner raised a ``ProtocolError`` — so the text is the
    joiner's to the character — else ``P2`` naming the exception's type,
    the step and its text."""

    def __init__(self, refused: Refused) -> None:
        self.step = refused.step
        self.kind = refused.kind
        if refused.code is not None:
            super().__init__(refused.code, refused.message, path=refused.path)
            return
        where = f"step {refused.step!r}" if refused.step else f"the {refused.moment}"
        super().__init__(
            "P2",
            f"the publishing rank failed at {where} with {refused.kind}: "
            f"{refused.message}; this rank stops with it",
        )


def refusal_of(moment: str, step: str | None, err: BaseException) -> Refused:
    """``err`` as it travels to the other ranks ([`Refused`][])."""
    if isinstance(err, ProtocolError):
        return Refused(
            moment,
            step,
            kind=type(err).__name__,
            message=err.message,
            code=err.code,
            path=err.path,
        )
    return Refused(
        moment, step, kind=type(err).__name__, message=str(err), code=None, path=None
    )


# --------------------------------------------------------------------------- #
# the codec — an outcome is JSON bytes
# --------------------------------------------------------------------------- #


def encode(outcome: Outcome) -> bytes:
    """``outcome`` as UTF-8 JSON, sorted keys, the outcome's class as
    ``outcome``."""
    body = {"outcome": type(outcome).__name__, **dataclasses.asdict(outcome)}
    return json.dumps(body, sort_keys=True).encode("utf-8")


def decode(data: bytes) -> Outcome:
    """The outcome ``data`` encodes.

    Raises:
        LockstepError: ``data`` is not an encoded outcome.
    """
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise LockstepError(
            f"the agreed payload is not an outcome ({err}); {len(data)} bytes"
        ) from err
    if not isinstance(body, dict):
        raise LockstepError(
            f"the agreed payload is not an outcome: a {type(body).__name__}, "
            "not an object"
        )
    kind = body.pop("outcome", None)
    try:
        if kind == "Decided":
            return Decided(body["moment"], body["step"], body["value"])
        if kind == "Refused":
            return Refused(
                body["moment"],
                body["step"],
                kind=body["kind"],
                message=body["message"],
                code=body["code"],
                path=body["path"],
            )
    except (KeyError, TypeError, ValueError) as err:
        raise LockstepError(
            f"the agreed payload is not an outcome: {kind} missing {err}"
        ) from err
    raise LockstepError(
        f"the agreed payload is not an outcome: {kind!r} is neither Decided nor Refused"
    )


# --------------------------------------------------------------------------- #
# the seam
# --------------------------------------------------------------------------- #


@runtime_checkable
class Lockstep(Protocol):
    """``agree`` is called by every rank once per decision: the joiner with
    its outcome, every other rank with ``None``; each receives the joiner's
    outcome."""

    def agree(self, outcome: Outcome | None) -> Outcome: ...


class Solo:
    """World 1: this process is the joiner, and its outcome is the answer."""

    def agree(self, outcome: Outcome | None) -> Outcome:
        if outcome is None:
            raise LockstepError(
                "a world of one rank has no one to follow: its one rank is the "
                "joiner and passes its outcome"
            )
        return outcome


#: The world-1 lockstep every run uses unless a launcher set another.
SOLO = Solo()


def decide(
    lockstep: Lockstep,
    joins: bool,
    moment: str,
    step: str | None,
    decision: Callable[[], T],
    *,
    follow: Callable[[], None] | None = None,
) -> T:
    """One agreed decision at ``moment`` of ``step``.

    The joiner (``joins``) runs ``decision`` and agrees its value; if it
    raises — ``KeyboardInterrupt`` included — the refusal is agreed first and
    the exception then propagates as itself. A follower runs ``follow`` — its
    own collective half of the step, if any — then agrees ``None`` and takes
    the joiner's value, or raises the joiner's refusal.

    Raises:
        LockstepRefusal: on a follower, the joiner's exception.
        LockstepError: on a follower, an outcome of another moment or step.
    """
    if joins:
        try:
            value = decision()
        except BaseException as err:
            lockstep.agree(refusal_of(moment, step, err))
            raise
        lockstep.agree(Decided(moment, step, value))
        return value
    if follow is not None:
        follow()
    outcome = lockstep.agree(None)
    mismatch = LockstepError(
        f"this rank awaited {moment} of step {step!r} but the publishing rank "
        f"agreed {outcome.moment} of step {outcome.step!r}: the ranks' steps "
        "are out of lockstep"
    )
    if isinstance(outcome, Refused):
        # the joiner's refusal at this moment — or at its manifest, where a
        # joiner that failed between two agreements ends up — stops this rank
        if (outcome.moment, outcome.step) == (moment, step):
            raise LockstepRefusal(outcome)
        if outcome.moment == "manifest":
            raise LockstepRefusal(outcome)
        raise mismatch
    if (outcome.moment, outcome.step) != (moment, step):
        raise mismatch
    return outcome.value
