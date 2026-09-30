"""The ``Lockstep`` conformance suite run against every implementation.

``lockstep_contract.py`` states the contract lines as decision programs and
checks; this file runs them against `SOLO` (world 1), the production
[`CollectiveLockstep`][causalab.neural.shared.parallel.lockstep.CollectiveLockstep] over the simulator's collective
(``SimulatedWorld.run``, every mesh geometry, every drawn schedule) and the
same class over ``TorchCollective`` under ``gloo`` in spawned processes
(the smoke tier, ``docs/model_parallelism.md`` §10.6). Three
implementations agreeing on one set of functions is the point; where the
simulator and torch disagree, the simulator is wrong.

``property``: the agreement invariant — whatever each rank would decide for
itself, the decision every rank acts on is the joiner's, on every layout
under every schedule, for any JSON value. ``unit``: the hand-written
mutations, the repository's convention — ranks that decide for themselves
fail the agreement line; a ``decide`` that skips the moment check fails the
two out-of-lockstep lines (a value and a refusal at another moment) and no
other; one that keeps the check for a value but re-raises any refusal as
its own fails the refusal line alone.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.lockstep import CollectiveLockstep
from causalab.protocol.lockstep import (
    SOLO,
    Decided,
    Lockstep,
    LockstepRefusal,
    Outcome,
    Refused,
    decide,
)
from causalab.protocol.parallel import (
    ONE,
    MeshLayout,
    ParallelGeometry,
    format_geometry,
)
from tests._helpers import parallel_strategies as ps
from tests._helpers.geometries import mesh_geometries
from tests._helpers.gloo_world import GlooWorld
from tests._helpers.simulated_world import Schedule, SimulatedWorld
from tests.neural.shared.parallel import lockstep_contract
from tests.neural.shared.parallel.lockstep_contract import (
    BY_NAME,
    CONTRACTS,
    Contract,
    every_contract,
    joins_at,
    over_collective,
    results_of,
    solo_run,
    verify_all,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: The geometries the spawned ``gloo`` worlds run (§10.6, ``world ∈ {2, 4}``):
#: one chain round at world 2; two rounds (pipeline, then model) at world 4.
GEOMETRIES: tuple[ParallelGeometry, ...] = (
    ParallelGeometry(tensor=2),
    ParallelGeometry(pipeline=2, tensor=2),
)

#: The simulator adds every chain axis on its own and the four-round chain.
SIMULATED_GEOMETRIES: tuple[ParallelGeometry, ...] = GEOMETRIES + (
    ParallelGeometry(data=2),
    ParallelGeometry(context=3),
    ParallelGeometry(expert=2),
    ParallelGeometry(data=2, pipeline=2, context=2, tensor=2),
)

_ids: Callable[[Any], str] = lambda x: (  # noqa: E731 - pytest id helper
    x.name if isinstance(x, Contract) else format_geometry(x)
)


def _simulated(geometry: ParallelGeometry, schedule: Schedule = ()) -> SimulatedWorld:
    return SimulatedWorld(
        ps.layout_of(geometry), world=geometry.world, schedule=schedule
    )


def _simulated_run(geometry: ParallelGeometry, schedule: Schedule = ()):
    world = _simulated(geometry, schedule)
    return lambda program: world.run(over_collective(program))


# --------------------------------------------------------------------------- #
# unit: Solo and the simulator
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSolo:
    def test_solo_is_a_lockstep(self) -> None:
        assert isinstance(SOLO, Lockstep)

    @pytest.mark.parametrize("contract", CONTRACTS, ids=_ids)
    def test_the_contract_holds_at_world_one(self, contract: Contract) -> None:
        contract.check(solo_run, MeshLayout(ONE))


@pytest.mark.unit
class TestSimulated:
    @pytest.mark.parametrize("geometry", SIMULATED_GEOMETRIES, ids=_ids)
    @pytest.mark.parametrize("contract", CONTRACTS, ids=_ids)
    def test_the_contract_holds(
        self, contract: Contract, geometry: ParallelGeometry
    ) -> None:
        contract.check(_simulated_run(geometry), MeshLayout(geometry))

    def test_the_joiner_is_rank_zero_on_every_layout(self) -> None:
        for geometry in SIMULATED_GEOMETRIES:
            joins = _simulated(geometry).run(lambda rank, c: joins_at(c))
            assert joins == [True] + [False] * (geometry.world - 1), geometry


# --------------------------------------------------------------------------- #
# property: the agreement invariant
# --------------------------------------------------------------------------- #

_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=12),
    lambda inner: st.lists(inner, max_size=4)
    | st.dictionaries(st.text(max_size=6), inner, max_size=4),
    max_leaves=8,
)


@pytest.mark.property
class TestAgreement:
    @_SETTINGS
    @given(geometry=mesh_geometries(bound=2), schedule=ps.schedules())
    @example(geometry=ParallelGeometry(tensor=2), schedule=[])
    def test_every_contract_holds_on_every_layout_under_every_schedule(
        self, geometry: ParallelGeometry, schedule: Schedule
    ) -> None:
        verify_all(
            _simulated(geometry, schedule).run(every_contract), MeshLayout(geometry)
        )

    @_SETTINGS
    @given(
        geometry=mesh_geometries(bound=2),
        schedule=ps.schedules(),
        values=st.lists(_json, min_size=16, max_size=16),
    )
    def test_any_per_rank_inputs_give_the_joiners_decision_on_every_rank(
        self, geometry: ParallelGeometry, schedule: Schedule, values: list[Any]
    ) -> None:
        """Each rank would decide ``values[rank]``; every rank acts on
        ``values[0]``, the joiner's, whatever the schedule."""

        def program(rank: int, c: Collective) -> Any:
            return decide(
                CollectiveLockstep(c),
                joins_at(c),
                "attempt",
                "scan",
                lambda: values[rank],
            )

        results = _simulated(geometry, schedule).run(program)
        assert results == [values[0]] * geometry.world

    @_SETTINGS
    @given(geometry=mesh_geometries(bound=2), schedule=ps.schedules(), value=_json)
    def test_a_refusal_and_a_value_survive_the_chain_intact(
        self, geometry: ParallelGeometry, schedule: Schedule, value: Any
    ) -> None:
        """``agree`` hands back the very outcome, ``Decided`` or ``Refused``,
        on every rank."""
        outcomes: tuple[Outcome, ...] = (
            Decided("attempt", "scan", value),
            Refused(
                "turn", "fit", kind="ValueError", message="x", code=None, path=None
            ),
        )
        for outcome in outcomes:
            results = _simulated(geometry, schedule).run(
                lambda rank, c: CollectiveLockstep(c).agree(
                    outcome if joins_at(c) else None
                )
            )
            assert results == [outcome] * geometry.world


# --------------------------------------------------------------------------- #
# unit: the mutations
# --------------------------------------------------------------------------- #


def _decide_without_the_moment_check(
    lockstep: Lockstep,
    joins: bool,
    moment: str,
    step: str | None,
    decision: Callable[[], Any],
    *,
    follow: Callable[[], None] | None = None,
) -> Any:
    """The mutation: a follower takes whatever value arrives, never asking
    whether it is the moment and step it awaited."""
    if joins:
        return decide(lockstep, joins, moment, step, decision, follow=follow)
    if follow is not None:
        follow()
    outcome = lockstep.agree(None)
    if isinstance(outcome, Refused):
        raise LockstepRefusal(outcome)
    return outcome.value


class _Agreed:
    """A lockstep that hands back one outcome already agreed."""

    def __init__(self, outcome: Outcome) -> None:
        self._outcome = outcome

    def agree(self, outcome: Outcome | None) -> Outcome:
        return self._outcome


def _decide_reraising_any_refusal(
    lockstep: Lockstep,
    joins: bool,
    moment: str,
    step: str | None,
    decision: Callable[[], Any],
    *,
    follow: Callable[[], None] | None = None,
) -> Any:
    """The narrower mutation: the moment check kept for a value, dropped for
    a refusal — a follower re-raises whatever refusal arrives, at any moment
    of any step, as though it were its own."""
    if joins:
        return decide(lockstep, joins, moment, step, decision, follow=follow)
    if follow is not None:
        follow()
    outcome = lockstep.agree(None)
    if isinstance(outcome, Refused):
        raise LockstepRefusal(outcome)
    return decide(_Agreed(outcome), False, moment, step, decision)


OUT_OF_LOCKSTEP_LINES = frozenset(
    {
        "a_follower_awaiting_another_moment_is_out_of_lockstep",
        "a_refusal_at_another_moment_is_out_of_lockstep_not_this_ranks_refusal",
    }
)


@pytest.mark.unit
class TestMutations:
    def test_ranks_deciding_for_themselves_fail_the_agreement_line(self) -> None:
        """Every rank its own joiner over ``Solo`` — each acting on what it
        would decide for itself — is what the lockstep exists to prevent."""
        geometry = ParallelGeometry(tensor=2)
        contract = BY_NAME["a_decided_value_is_the_joiners_on_every_rank"]

        def own(program: Any) -> Sequence[Any]:
            return _simulated(geometry).run(lambda rank, c: program(rank, SOLO, True))

        with pytest.raises(AssertionError):
            contract.check(own, MeshLayout(geometry))
        # the same ranks under the production lockstep pass
        contract.check(_simulated_run(geometry), MeshLayout(geometry))

    def test_skipping_the_moment_check_fails_both_out_of_lockstep_lines_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the check a follower takes a value of another moment as
        its own and re-raises a refusal of another moment as its own: the
        two drift lines fail, and no other."""
        monkeypatch.setattr(
            lockstep_contract, "decide", _decide_without_the_moment_check
        )
        geometry = ParallelGeometry(pipeline=2, tensor=2)
        for name in sorted(OUT_OF_LOCKSTEP_LINES):
            with pytest.raises(AssertionError):
                BY_NAME[name].check(_simulated_run(geometry), MeshLayout(geometry))
        # the mutation touches nothing else: every other line still holds
        for contract in CONTRACTS:
            if contract.name not in OUT_OF_LOCKSTEP_LINES:
                contract.check(_simulated_run(geometry), MeshLayout(geometry))

    def test_reraising_any_refusal_fails_the_refusal_drift_line_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The third branch of ``decide`` stated on its own: a refusal at
        another moment named as drift, not re-raised. The mutation keeps
        the value check, so the value drift line still holds — this line
        alone catches it."""
        monkeypatch.setattr(lockstep_contract, "decide", _decide_reraising_any_refusal)
        geometry = ParallelGeometry(pipeline=2, tensor=2)
        line = "a_refusal_at_another_moment_is_out_of_lockstep_not_this_ranks_refusal"
        with pytest.raises(AssertionError):
            BY_NAME[line].check(_simulated_run(geometry), MeshLayout(geometry))
        for contract in CONTRACTS:
            if contract.name != line:
                contract.check(_simulated_run(geometry), MeshLayout(geometry))


# --------------------------------------------------------------------------- #
# smoke: CollectiveLockstep over TorchCollective under gloo
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", params=GEOMETRIES, ids=_ids)
def gloo_results(
    request: pytest.FixtureRequest,
) -> tuple[MeshLayout, Sequence[dict[str, Any]]]:
    """Every contract's program run once per geometry in one spawned world."""
    geometry: ParallelGeometry = request.param
    return MeshLayout(geometry), GlooWorld(geometry).run(every_contract)


@pytest.mark.smoke
class TestCollectiveLockstepUnderGloo:
    @pytest.mark.parametrize("contract", CONTRACTS, ids=_ids)
    def test_the_contract_holds(
        self,
        contract: Contract,
        gloo_results: tuple[MeshLayout, Sequence[dict[str, Any]]],
    ) -> None:
        layout, results = gloo_results
        contract.verify(results_of(contract.name, results), layout)
