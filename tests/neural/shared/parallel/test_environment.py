"""The environment agreement at the rendezvous (``parallel/environment.py``,
``docs/model_parallelism.md`` §3): every rank publishes the raw text of the
agreed variables on the store and refuses, by name, the first one a peer
spells differently — so a per-node launch-script slip is a named refusal on
every rank rather than a divergence inside the group."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.environment import (
    AGREED_VARIABLES,
    AGREEMENT_VARIABLE,
    agree_environment,
    environment_key,
    read_settings,
)
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_VARIABLE,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import CONTEXT_EXPERIMENTAL_VARIABLE
from tests._helpers.simulated_world.heartbeat import FakeStore

_values = st.one_of(st.none(), st.sampled_from(["", "0", "1", "30", "600", "1e-3"]))
_environments = st.fixed_dictionaries({name: _values for name in AGREED_VARIABLES}).map(
    lambda d: {k: v for k, v in d.items() if v is not None}
)


def _agree_world(environs: list[dict[str, str]]) -> list[dict[str, str | None]]:
    """Every rank publishes first, as the real ranks do before any reads."""
    store = FakeStore()
    for rank, env in enumerate(environs):
        store.set(environment_key(rank), json.dumps(read_settings(env)))
    return [
        agree_environment(store, rank=rank, world=len(environs), environ=env)
        for rank, env in enumerate(environs)
    ]


@pytest.mark.unit
def test_the_agreed_variables_are_the_four_that_steer_a_rank() -> None:
    assert AGREED_VARIABLES == (
        COLLECTIVE_TIMEOUT_VARIABLE,
        RANK_GRACE_VARIABLE,
        CONTEXT_EXPERIMENTAL_VARIABLE,
        AGREEMENT_VARIABLE,
    )
    assert AGREEMENT_VARIABLE == "CAUSALAB_GRADIENT_AGREEMENT"


@pytest.mark.unit
def test_a_rank_publishes_its_settings_under_its_key() -> None:
    store = FakeStore()
    environ = {RANK_GRACE_VARIABLE: "5", "UNRELATED": "x"}
    agreed = agree_environment(store, rank=0, world=1, environ=environ)
    assert agreed == {
        COLLECTIVE_TIMEOUT_VARIABLE: None,
        RANK_GRACE_VARIABLE: "5",
        CONTEXT_EXPERIMENTAL_VARIABLE: None,
        AGREEMENT_VARIABLE: None,
    }
    assert json.loads(store.get(environment_key(0))) == agreed


@pytest.mark.property
@given(_environments, st.integers(1, 4))
@settings(max_examples=30, deadline=None)
def test_a_world_spelling_every_setting_alike_agrees(environ, world: int) -> None:
    agreed = _agree_world([dict(environ)] * world)
    assert agreed == [read_settings(environ)] * world


@pytest.mark.property
@given(
    _environments,
    st.integers(2, 4),
    st.sampled_from(AGREED_VARIABLES),
    st.sampled_from(["", "7", "1"]),
)
@settings(max_examples=30, deadline=None)
def test_one_rank_spelling_one_setting_differently_is_refused_on_every_rank(
    environ, world: int, name: str, other: str
) -> None:
    """The refusal names the variable, both ranks and both values; unset
    and empty are different spellings, so the parsers see one text."""
    if environ.get(name) == other:
        other = other + "0"
    odd = world - 1
    environs = [dict(environ) for _ in range(world)]
    environs[odd][name] = other
    store = FakeStore()
    for rank, env in enumerate(environs):
        store.set(environment_key(rank), json.dumps(read_settings(env)))
    for rank, env in enumerate(environs):
        with pytest.raises(ProtocolError) as err:
            agree_environment(store, rank=rank, world=world, environ=env)
        text = str(err.value)
        assert err.value.code == "P4" and err.value.path == "--parallel"
        assert name in text and f"{name}={other!r}" in text
        peer = 0 if rank == odd else odd
        assert f"rank {rank} has" in text and f"rank {peer} has" in text
        mine = environ.get(name)
        assert (f"{name} unset" if mine is None else f"{name}={mine!r}") in text


@pytest.mark.unit
def test_a_peer_still_starting_is_waited_for_through_the_store() -> None:
    """``store.get`` of an absent key is the wait (the ``TCPStore`` blocks,
    bounded by its timeout); the fake raises, which is what a peer that
    never arrives looks like here."""
    store = FakeStore()
    with pytest.raises(KeyError):
        agree_environment(store, rank=0, world=2, environ={})
