"""The lockstep over the mesh's groups (``docs/model_parallelism.md`` §3, §11):
the joiner's outcome reaches every rank of any mesh geometry through the
[`Collective`][causalab.neural.shared.parallel.collective.Collective] seam alone,
so the ``SimulatedWorld`` runs it under a drawn schedule.

``property``: over every mesh geometry (world at most 16) and drawn
schedule, every rank's ``agree`` returns the joiner's outcome, and the chain of
broadcasts is what the transcript shows — one round per axis above one,
joined only by the ranks that hold the value or are about to. ``unit``: the
misuse refusals (a follower passing an outcome, the joiner passing none),
world 1 through ``Solo``, and the mutation the chain's participation rule
exists for — every rank joining every round hands a non-holder the source's
seat, which the simulator refuses as a misuse.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.collective import SOLO as SOLO_COLLECTIVE
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.lockstep import CHAIN, CollectiveLockstep
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.lockstep import (
    Decided,
    Lockstep,
    LockstepError,
    Outcome,
    Refused,
    refusal_of,
)
from causalab.protocol.parallel import ParallelGeometry

from tests._helpers.geometries import mesh_geometries
from tests._helpers.parallel_strategies import schedules
from tests._helpers.simulated_world import (
    Misuse,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.simulated_world.collective import Groups

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

ENTRY = Decided(
    "attempt",
    "fit",
    {"status": "completed", "files": ["rot.safetensors", "iia.json"], "n": 3},
)
REFUSAL = refusal_of(
    "turn", "scan", ProtocolError("P4", "asks for 2 ranks", path="--parallel")
)


def _groups(geometry: ParallelGeometry) -> Groups:
    return groups_for(
        geometry.world,
        data=geometry.data,
        pipeline=geometry.pipeline,
        context=geometry.context,
        tensor=geometry.tensor,
        expert=geometry.expert,
    )


def _agreeing(outcome: Outcome) -> Any:
    def program(rank: int, collective: Collective) -> Outcome:
        return CollectiveLockstep(collective).agree(outcome if rank == 0 else None)

    return program


class TestEveryRankReceivesTheJoiners:
    pytestmark = pytest.mark.property

    @_SETTINGS
    @given(
        geometry=mesh_geometries(bound=2),
        schedule=schedules(),
        outcome=st.sampled_from([ENTRY, REFUSAL]),
    )
    @example(geometry=ParallelGeometry(tensor=2), schedule=[], outcome=ENTRY)
    def test_the_joiners_outcome_reaches_every_rank(
        self, geometry: ParallelGeometry, schedule: Schedule, outcome: Outcome
    ) -> None:
        world = SimulatedWorld(
            _groups(geometry), world=geometry.world, schedule=schedule
        )
        results = world.run(_agreeing(outcome))
        assert results == [outcome] * geometry.world

    @_SETTINGS
    @given(geometry=mesh_geometries(bound=2), schedule=schedules())
    def test_the_chain_is_one_round_per_axis_above_one(
        self, geometry: ParallelGeometry, schedule: Schedule
    ) -> None:
        """The transcript: broadcasts only, on the chain's axes in order, and
        on each axis exactly the groups whose source holds the value — one
        group on the first axis above one, every group on the last."""
        world = SimulatedWorld(
            _groups(geometry), world=geometry.world, schedule=schedule
        )
        world.run(_agreeing(ENTRY))
        events = [event for event in world.transcript if event.op == "broadcast"]
        assert len(events) == len(world.transcript), "the chain is broadcasts alone"
        # the transcript is in schedule order across ranks — a rank that
        # joins only the last round may arrive there before an earlier round
        # completes elsewhere — so the axes are compared as a set; the order
        # within one rank is the chain's
        assert {event.axis for event in events} == {
            axis for axis in CHAIN if getattr(geometry, axis) > 1
        }
        for rank_events in world.transcripts:
            axes = [event.axis for event in rank_events]
            assert axes == sorted(axes, key=CHAIN.index)
        for index, axis in enumerate(CHAIN):
            size = getattr(geometry, axis)
            joined = {event.rank for event in events if event.axis == axis}
            if size == 1:
                assert joined == set()
                continue
            # the ranks at local index 0 on every later axis
            later = [a for a in CHAIN[index + 1 :]]
            expected = {
                rank
                for rank in range(geometry.world)
                if all(_local(geometry, rank, a) == 0 for a in later)
            }
            assert joined == expected, axis


def _local(geometry: ParallelGeometry, rank: int, axis: str) -> int:
    from causalab.protocol.parallel import MeshLayout

    return MeshLayout(geometry).rank_in(rank, axis)  # type: ignore[arg-type]


class TestMisuse:
    pytestmark = pytest.mark.unit

    def test_a_follower_passing_an_outcome_is_refused(self) -> None:
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(rank: int, collective: Collective) -> Outcome:
            return CollectiveLockstep(collective).agree(ENTRY)

        with pytest.raises(Exception) as info:
            world.run(program)
        assert isinstance(info.value.__cause__, LockstepError)
        assert "not the joiner" in str(info.value.__cause__)

    def test_the_joiner_passing_none_is_refused(self) -> None:
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(rank: int, collective: Collective) -> Outcome:
            return CollectiveLockstep(collective).agree(None)

        with pytest.raises(Exception) as info:
            world.run(program)
        assert isinstance(info.value.__cause__, LockstepError)
        assert "passes its outcome" in str(info.value.__cause__)

    def test_world_one_over_the_solo_collective_is_the_identity(self) -> None:
        lockstep = CollectiveLockstep(SOLO_COLLECTIVE)
        assert isinstance(lockstep, Lockstep)
        assert lockstep.agree(ENTRY) == ENTRY
        with pytest.raises(LockstepError):
            lockstep.agree(None)

    def test_a_refused_outcome_survives_the_chain_intact(self) -> None:
        world = SimulatedWorld(groups_for(4, pipeline=2, tensor=2), world=4, schedule=3)
        results = world.run(_agreeing(REFUSAL))
        assert all(isinstance(r, Refused) for r in results)
        assert results == [REFUSAL] * 4

    def test_the_payload_travels_on_the_collectives_device(self) -> None:
        """The production collective refuses a tensor off its device by name
        (``TorchCollective._on_device``); the lockstep builds the payload on
        the device the collective declares."""

        class _Device:
            device = torch.device("cpu")
            rank = staticmethod(lambda axis: 0)
            size = staticmethod(lambda axis: 1)

        lockstep = CollectiveLockstep(_Device())  # type: ignore[arg-type]
        assert lockstep.device == torch.device("cpu")
        assert lockstep.agree(ENTRY) == ENTRY

    def test_the_payload_is_bytes_on_the_device_the_collective_declares(self) -> None:
        """A collective declaring a device other than the CPU — the meta
        device stands in for a CUDA one — gets its payload there, as
        ``uint8``; the simulator's ranks declare the CPU (``Collective.device``
        is a protocol member, so there is no undeclared case) and carry CPU
        tensors the receiving side decodes."""

        class _Meta:
            device = torch.device("meta")
            rank = staticmethod(lambda axis: 0)
            size = staticmethod(lambda axis: 1)

        elsewhere = CollectiveLockstep(_Meta())  # type: ignore[arg-type]
        assert elsewhere.device == torch.device("meta")
        payload = elsewhere.payload(ENTRY)
        assert payload.device.type == "meta" and payload.dtype == torch.uint8

        class _Plain:
            device = torch.device("cpu")
            rank = staticmethod(lambda axis: 0)
            size = staticmethod(lambda axis: 1)

        plain = CollectiveLockstep(_Plain())  # type: ignore[arg-type]
        assert plain.device == torch.device("cpu")
        payload = plain.payload(ENTRY)
        assert payload.device == torch.device("cpu") and payload.dtype == torch.uint8
        assert plain.outcome(payload) == ENTRY

    def test_the_follower_refusal_names_its_mesh_coordinates(self) -> None:
        """The rank that misused ``agree`` is named by where it sits — the
        operator's fix is a rank's program, and the coordinates say which."""
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(rank: int, collective: Collective) -> Outcome:
            return CollectiveLockstep(collective).agree(ENTRY)

        with pytest.raises(Exception) as info:
            world.run(program)
        cause = info.value.__cause__
        assert isinstance(cause, LockstepError)
        assert "{'data': 0, 'pipeline': 0, 'context': 0, 'model': 1}" in str(cause)
        assert "it passes None to agree and receives the joiner's outcome" in str(cause)


class TestTheParticipationRuleIsLoadBearing:
    pytestmark = pytest.mark.unit

    def test_every_rank_joining_every_round_seats_a_non_holder_as_source(
        self,
    ) -> None:
        """Mutation: drop the rule that a rank joins a round only when its
        later coordinates are all zero. On ``pp=2,tp=2`` the pipeline round
        then has rank 1 (``p=0, m=1``) as the source of its group with
        nothing to send — the simulator refuses that as a misuse, and a real
        backend would broadcast an uninitialised buffer."""
        world = SimulatedWorld(groups_for(4, pipeline=2, tensor=2), world=4, schedule=0)

        def program(rank: int, collective: Collective) -> Outcome:
            lockstep = CollectiveLockstep(collective)
            payload = lockstep.payload(ENTRY) if rank == 0 else None
            held = payload is not None
            for axis in CHAIN:
                if collective.size(axis) == 1:
                    continue
                payload = collective.broadcast(payload if held else None, 0, axis)
                held = True
            assert payload is not None
            return lockstep.outcome(payload)

        with pytest.raises(Exception) as info:
            world.run(program)
        assert isinstance(info.value.__cause__, Misuse)
