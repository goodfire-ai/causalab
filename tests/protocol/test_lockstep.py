"""The lockstep seam (``docs/model_parallelism.md`` §3, §11; workflow spec §8):
the joiner's per-process decisions — reuse, a step's record, the manifest,
the pins stamp — reach every rank of a launched world as one typed
``Outcome``, so no rank steers its control flow from a filesystem it did
not write.

``unit``: the closed moments; the codec (an outcome is JSON, so a
``Refused`` re-renders the joiner's refusal to the character); ``Solo`` is
the identity and refuses a follower's call; ``decide`` on the joiner runs
the decision and agrees its value or its refusal *before* re-raising, on a
follower runs its own collective half, then takes the joiner's value or
raises the joiner's refusal as [`LockstepRefusal`][causalab.protocol.lockstep.LockstepRefusal]; a moment or step
the follower did not await is [`LockstepError`][causalab.protocol.lockstep.LockstepError]. ``property``: the
codec round-trips every JSON value.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.lockstep import (
    MOMENTS,
    SOLO,
    Decided,
    Lockstep,
    LockstepError,
    LockstepRefusal,
    Outcome,
    Refused,
    Solo,
    decide,
    decode,
    encode,
    refusal_of,
)

pytestmark = pytest.mark.unit


class _Scripted:
    """A lockstep whose ``agree`` hands back a scripted outcome (a follower's
    view) or records the joiner's (a joiner's view)."""

    def __init__(self, *incoming: Outcome) -> None:
        self.incoming = list(incoming)
        self.agreed: list[Outcome] = []

    def agree(self, outcome: Outcome | None) -> Outcome:
        if outcome is not None:
            self.agreed.append(outcome)
            return outcome
        return self.incoming.pop(0)


_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda inner: st.lists(inner) | st.dictionaries(st.text(), inner),
    max_leaves=12,
)


# --------------------------------------------------------------------------- #
# the vocabulary and the codec
# --------------------------------------------------------------------------- #


def test_the_moments_are_closed_and_in_run_order() -> None:
    assert MOMENTS == ("turn", "attempt", "manifest")


def test_the_retired_pins_moment_is_refused() -> None:
    """The workflow ``pins`` section is gone, so no joiner agrees a
    ``pins`` moment, and a follower waiting on one would wait forever."""
    with pytest.raises(ValueError, match="'pins' is not one of"):
        Decided("pins", None, True)


@pytest.mark.parametrize("moment", ["turn", "attempt"])
def test_a_decided_outcome_names_its_moment_and_step(moment: str) -> None:
    outcome = Decided(moment, "fit", {"status": "completed"})
    assert outcome.moment == moment and outcome.step == "fit"
    assert decode(encode(outcome)) == outcome


def test_an_outcome_outside_the_moments_is_refused() -> None:
    with pytest.raises(ValueError, match="turn.*attempt.*manifest"):
        Decided("launch", None, 1)
    with pytest.raises(ValueError):
        Refused("launch", None, kind="X", message="m", code=None, path=None)


@settings(max_examples=30, deadline=None)
@given(value=_json, step=st.none() | st.text(min_size=1))
def test_the_codec_round_trips_every_json_value(value: Any, step: str | None) -> None:
    outcome = Decided("manifest", step, value)
    assert decode(encode(outcome)) == outcome
    assert json.loads(encode(outcome).decode("utf-8"))["moment"] == "manifest"


def test_the_encoding_sorts_its_keys() -> None:
    """The same outcome is the same bytes on every rank that encodes it,
    whatever order a dataclass lists its fields in."""
    outcomes: tuple[Outcome, ...] = (
        Decided("turn", "scan", {"b": 1, "a": 2}),
        refusal_of("attempt", "fit", KeyError("values.json")),
    )
    for outcome in outcomes:
        body = json.loads(encode(outcome).decode("utf-8"))
        assert list(body) == sorted(body)
        assert body["outcome"] == type(outcome).__name__


def test_a_payload_missing_a_field_names_the_kind_and_the_field() -> None:
    with pytest.raises(LockstepError, match="not an outcome: Decided missing 'step'"):
        decode(json.dumps({"outcome": "Decided", "moment": "turn"}).encode())


def test_a_malformed_payload_is_a_lockstep_error_naming_the_bytes() -> None:
    with pytest.raises(LockstepError, match="not an outcome"):
        decode(b"\xff\x00")
    with pytest.raises(LockstepError, match="not an outcome"):
        decode(json.dumps({"moment": "turn"}).encode())
    with pytest.raises(LockstepError, match="not an outcome"):
        decode(json.dumps([1, 2]).encode())


# --------------------------------------------------------------------------- #
# refusal_of — the joiner's exception as it travels
# --------------------------------------------------------------------------- #


def test_a_protocol_error_travels_with_its_code_path_and_bare_message() -> None:
    err = ProtocolError("P4", "asks for two ranks", path="--parallel")
    refused = refusal_of("turn", "scan", err)
    assert refused == Refused(
        "turn",
        "scan",
        kind="ProtocolError",
        message="asks for two ranks",
        code="P4",
        path="--parallel",
    )
    assert decode(encode(refused)) == refused


def test_any_other_exception_travels_by_type_name_and_text() -> None:
    refused = refusal_of("attempt", "best", KeyError("values.json"))
    assert refused.kind == "KeyError" and refused.message == "'values.json'"
    assert refused.code is None and refused.path is None


# --------------------------------------------------------------------------- #
# Solo — world 1
# --------------------------------------------------------------------------- #


def test_solo_is_a_lockstep_and_the_identity() -> None:
    assert isinstance(SOLO, Lockstep) and isinstance(SOLO, Solo)
    outcome = Decided("manifest", None, True)
    assert SOLO.agree(outcome) is outcome


def test_solo_refuses_a_followers_call() -> None:
    """World 1 has one rank, the joiner; a ``None`` here is a caller that
    believes it follows someone."""
    with pytest.raises(LockstepError, match="joiner"):
        SOLO.agree(None)


# --------------------------------------------------------------------------- #
# decide — the joiner's half and the follower's half
# --------------------------------------------------------------------------- #


def test_the_joiner_agrees_its_decision_and_returns_it() -> None:
    lockstep = _Scripted()
    value = decide(lockstep, True, "turn", "scan", lambda: {"reused": None})
    assert value == {"reused": None}
    assert lockstep.agreed == [Decided("turn", "scan", {"reused": None})]


def test_the_joiner_agrees_its_refusal_before_re_raising_it() -> None:
    lockstep = _Scripted()
    err = ProtocolError("P2", "no values.json", path="steps.best")

    def decision() -> Any:
        raise err

    with pytest.raises(ProtocolError) as info:
        decide(lockstep, True, "attempt", "best", decision)
    assert info.value is err
    assert lockstep.agreed == [refusal_of("attempt", "best", err)]


def test_a_keyboard_interrupt_on_the_joiner_is_agreed_and_re_raised() -> None:
    """A rank waiting on the joiner must learn of an interrupt too, or it
    waits forever; the interrupt itself still propagates as itself."""
    lockstep = _Scripted()

    def decision() -> Any:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        decide(lockstep, True, "attempt", "fit", decision)
    (agreed,) = lockstep.agreed
    assert isinstance(agreed, Refused) and agreed.kind == "KeyboardInterrupt"


def test_the_follower_runs_its_half_then_takes_the_joiners_value() -> None:
    ran: list[str] = []
    lockstep = _Scripted(Decided("attempt", "scan", {"files": ["iia.json"]}))

    def never() -> Any:
        raise AssertionError("a follower never runs the joiner's decision")

    value = decide(
        lockstep, False, "attempt", "scan", never, follow=lambda: ran.append("engine")
    )
    assert value == {"files": ["iia.json"]} and ran == ["engine"]


def test_the_follower_raises_the_joiners_refusal_rendered_to_the_character(
    capsys,
) -> None:
    """Refusals stay refusals on every rank: the follower's exception is a
    ``ProtocolError`` whose text is the joiner's — code, path and message —
    so ``refused: …`` reads the same on every rank's stderr."""
    err = ProtocolError("P4", "asks for a world of 2 ranks", path="--parallel")
    lockstep = _Scripted(refusal_of("turn", "scan", err))
    with pytest.raises(LockstepRefusal) as info:
        decide(lockstep, False, "turn", "scan", lambda: None)
    assert str(info.value) == str(err)
    assert info.value.code == "P4" and info.value.path == "--parallel"
    assert info.value.step == "scan" and info.value.kind == "ProtocolError"
    assert isinstance(info.value, ProtocolError)


def test_a_foreign_refusal_on_the_follower_names_its_type() -> None:
    lockstep = _Scripted(refusal_of("attempt", "best", RuntimeError("script died")))
    with pytest.raises(LockstepRefusal) as info:
        decide(lockstep, False, "attempt", "best", lambda: None)
    assert info.value.code == "P2"
    assert "RuntimeError" in str(info.value) and "script died" in str(info.value)
    assert "best" in str(info.value)


def test_a_moment_the_follower_did_not_await_is_a_lockstep_error() -> None:
    """The joiner moved to another moment or step — it died between two
    agreements and went to its manifest, say — and the follower refuses to
    read that as its own decision."""
    lockstep = _Scripted(Decided("manifest", None, {"steps": {}}))
    with pytest.raises(LockstepError, match="awaited turn of step 'scan'.*manifest"):
        decide(lockstep, False, "turn", "scan", lambda: None)
    lockstep = _Scripted(Decided("turn", "fit", {"reused": None}))
    with pytest.raises(LockstepError, match="'scan'.*'fit'"):
        decide(lockstep, False, "turn", "scan", lambda: None)


def test_a_manifest_refusal_reaching_a_follower_mid_run_is_the_joiners_failure(
    capsys,
) -> None:
    """The joiner failed outside an agreed moment and reached its manifest:
    the follower awaiting the next turn gets the failure, not a puzzle."""
    err = OSError("events.jsonl: disk full")
    lockstep = _Scripted(refusal_of("manifest", None, err))
    with pytest.raises(LockstepRefusal) as info:
        decide(lockstep, False, "turn", "fit", lambda: None)
    assert "OSError" in str(info.value) and "disk full" in str(info.value)


def test_the_follower_never_runs_the_joiners_decision_even_without_a_half() -> None:
    lockstep = _Scripted(Decided("manifest", None, True))
    assert decide(lockstep, False, "manifest", None, lambda: 1 / 0) is True
