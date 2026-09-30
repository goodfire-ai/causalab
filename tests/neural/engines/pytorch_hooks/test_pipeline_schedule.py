"""The micro-batched pipeline schedule's reference model
(``docs/model_parallelism.md`` §8.6 — a design; the engine does not stream yet).

Two pure functions below spell the schedule §8.6 proposes, per stage, as a
**transcript** of operations: ``fill_drain`` (inference) and ``one_f_one_b``
(fits). A transcript names the compute steps (``F(i)``, ``B(i)``, the
``drain``) and the point-to-point operations on the stage's two links
(``send_act`` / ``recv_act`` upward for the residual, ``send_grad`` /
``recv_grad`` downward for its gradient), and nothing else: it is
torch-free, so hypothesis can hold the schedule's invariants for every
``(stages, micro-batches)`` before any engine code exists.

The model checker, `deadlock_free`, runs the ``stages`` transcripts
under **blocking** point-to-points — a send completes only when the peer is
at the matching receive, the ``drain`` only when every stage is at it — the
discipline the simulator (``tests/_helpers/simulated_world``) and ``gloo``
enforce and the one a real NCCL stream order would also survive. What it
holds:

- both schedules complete for every ``1 ≤ stages ≤ 6`` and ``1 ≤ m ≤ 12``
  (no deadlock), every micro-batch forwarded once per stage and, under
  1F1B, backwarded once, after its forward;
- on every link the two ends' operations pair up **in order** — the
  invariant §8.6 states and the simulator would otherwise report as a
  ``Hang``;
- 1F1B holds at most ``min(stages − s, m)`` micro-batch graphs alive on
  stage ``s`` (the classic bound; stage 0 holds ``stages``);
- at ``m = 1`` the wire is today's placement wire (§6.5): one ``send_act``
  and one ``recv_grad`` on every stage but the last, one ``recv_act`` and
  one ``send_grad`` on every stage but the first;
- the pinned transcripts §8.6 prints for ``pp=2, m=3`` and ``pp=3, m=3``
  are what the rule generates;
- the **mutation**: the textbook order — send the gradient down *before*
  receiving the next micro-batch's residual — deadlocks at ``pp=2, m ≥ 2``
  under blocking point-to-points, and the checker sees it.

The simulator scenarios, the gloo smoke and the golden §8.6 names live in
the design (§8.6 "Tests") and not here: a skipped placeholder with an empty
body is a test that passes the day its marker comes off, which is the one
shape a placeholder must not have.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import pytest
import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

_SETTINGS = settings(deadline=None, max_examples=60)

# --------------------------------------------------------------------------- #
# the transcript
# --------------------------------------------------------------------------- #

P2P = ("send_act", "recv_act", "send_grad", "recv_grad")
COMPUTE = ("F", "B", "drain")


@dataclasses.dataclass(frozen=True)
class Op:
    """One transcript line of one stage: ``kind`` over micro-batch ``micro``
    (``-1`` for the drain, which is over every micro-batch at once)."""

    kind: str
    micro: int = -1

    def __repr__(self) -> str:  # the doc's spelling: ``send_act(2)``, ``F(0)``
        return self.kind if self.kind == "drain" else f"{self.kind}({self.micro})"


def _sa(i: int) -> Op:
    return Op("send_act", i)


def _ra(i: int) -> Op:
    return Op("recv_act", i)


def _sg(i: int) -> Op:
    return Op("send_grad", i)


def _rg(i: int) -> Op:
    return Op("recv_grad", i)


def _f(i: int) -> Op:
    return Op("F", i)


def _b(i: int) -> Op:
    return Op("B", i)


DRAIN = Op("drain")


def fill_drain(stages: int, stage: int, m: int) -> tuple[Op, ...]:
    """§8.6 fill-drain: every micro-batch forwarded in order — received from
    the stage below, computed, sent on — then the drain (the logits and
    capture broadcasts, the fire sums), which is the first group collective."""
    first, last = stage == 0, stage == stages - 1
    ops: list[Op] = []
    for i in range(m):
        if not first:
            ops.append(_ra(i))
        ops.append(_f(i))
        if not last:
            ops.append(_sa(i))
    ops.append(DRAIN)
    return tuple(ops)


def one_f_one_b(stages: int, stage: int, m: int) -> tuple[Op, ...]:
    """§8.6 1F1B: ``W = stages − 1 − stage`` warm-up forwards, then one
    forward and one backward per step, the link operations in the one order
    that pairs under blocking point-to-points — on the lower link *receive
    the next residual, then send the finished gradient*; on the upper link
    *send the finished residual, then receive the next gradient*."""
    first, last = stage == 0, stage == stages - 1
    warm = stages - 1 - stage
    ops: list[Op] = []
    for i in range(min(warm, m)):
        if not first:
            ops.append(_ra(i))
        ops.append(_f(i))
        if not last:
            ops.append(_sa(i))
    for j in range(m):
        i = j + warm
        if not first:
            if i < m:
                ops.append(_ra(i))
            if j > 0:
                ops.append(_sg(j - 1))
        if i < m:
            ops.append(_f(i))
        if not last:
            if i < m:
                ops.append(_sa(i))
            ops.append(_rg(j))
        ops.append(_b(j))
    if not first:
        ops.append(_sg(m - 1))
    ops.append(DRAIN)
    return tuple(ops)


def one_f_one_b_textbook(stages: int, stage: int, m: int) -> tuple[Op, ...]:
    """The mutation: the gradient sent down as soon as its backward is done,
    *before* the next residual is received — the order autograd-embedded
    communication (``send_with_grad`` / ``recv_with_grad``) would produce."""
    first, last = stage == 0, stage == stages - 1
    warm = stages - 1 - stage
    ops: list[Op] = []
    for i in range(min(warm, m)):
        if not first:
            ops.append(_ra(i))
        ops.append(_f(i))
        if not last:
            ops.append(_sa(i))
    for j in range(m):
        i = j + warm
        if i < m:
            if not first:
                ops.append(_ra(i))
            ops.append(_f(i))
            if not last:
                ops.append(_sa(i))
        if not last:
            ops.append(_rg(j))
        ops.append(_b(j))
        if not first:
            ops.append(_sg(j))
    ops.append(DRAIN)
    return tuple(ops)


Schedule = Callable[[int, int, int], tuple[Op, ...]]

# --------------------------------------------------------------------------- #
# the model checker
# --------------------------------------------------------------------------- #

_PEER_UP = {"send_act": 1, "recv_grad": 1}
_PEER_DOWN = {"recv_act": -1, "send_grad": -1}
_MATCH = {
    "send_act": "recv_act",
    "recv_act": "send_act",
    "send_grad": "recv_grad",
    "recv_grad": "send_grad",
}


def _peer(stage: int, op: Op) -> int:
    return stage + {**_PEER_UP, **_PEER_DOWN}[op.kind]


def deadlock_free(transcripts: Sequence[Sequence[Op]]) -> tuple[bool, str]:
    """Run the stages' transcripts under blocking point-to-points and a
    barrier drain; ``(True, "")`` when every stage finishes, else ``(False,
    where)`` naming each stage's pending operation."""
    position = [0] * len(transcripts)

    def pending(s: int) -> Op | None:
        return (
            transcripts[s][position[s]] if position[s] < len(transcripts[s]) else None
        )

    while True:
        progressed = False
        for s, ops in enumerate(transcripts):
            op = pending(s)
            if op is None:
                continue
            if op.kind in ("F", "B"):
                position[s] += 1
                progressed = True
            elif op.kind == "drain":
                if all(pending(t) == DRAIN for t in range(len(transcripts))):
                    for t in range(len(transcripts)):
                        position[t] += 1
                    progressed = True
            else:
                peer = _peer(s, op)
                other = pending(peer)
                if (
                    other is not None
                    and other.kind == _MATCH[op.kind]
                    and other.micro == op.micro
                ):
                    position[s] += 1
                    position[peer] += 1
                    progressed = True
        if all(pending(s) is None for s in range(len(transcripts))):
            return True, ""
        if not progressed:
            where = ", ".join(
                f"stage {s} at {pending(s)!r}" for s in range(len(transcripts))
            )
            return False, where


def link_sequences(
    transcripts: Sequence[Sequence[Op]], link: int
) -> tuple[list[Op], list[Op]]:
    """The operations the two ends of link ``(link, link + 1)`` issue on it,
    each in its own program order, the lower end's spelled as the upper
    end sees them (``send_act`` → ``recv_act``, ``recv_grad`` → ``send_grad``)."""
    lower = [
        Op(_MATCH[op.kind], op.micro) for op in transcripts[link] if op.kind in _PEER_UP
    ]
    upper = [op for op in transcripts[link + 1] if op.kind in _PEER_DOWN]
    return lower, upper


def graphs_alive(ops: Sequence[Op]) -> int:
    """The most micro-batch graphs alive at once on one stage: a graph is
    born at ``F(i)`` and freed at ``B(i)``."""
    alive, most = 0, 0
    for op in ops:
        if op.kind == "F":
            alive += 1
            most = max(most, alive)
        elif op.kind == "B":
            alive -= 1
    return most


def _world(schedule: Schedule, stages: int, m: int) -> list[tuple[Op, ...]]:
    return [schedule(stages, s, m) for s in range(stages)]


# --------------------------------------------------------------------------- #
# the properties
# --------------------------------------------------------------------------- #

#: Every ``(stages, m)`` the module docstring claims — 72 points, swept
#: whole rather than sampled: the space is small and total, and a failure is
#: a failure at a named geometry, not a counterexample to shrink.
GEOMETRIES = tuple(itertools.product(range(1, 7), range(1, 13)))
every_geometry = pytest.mark.parametrize(
    "geometry", GEOMETRIES, ids=[f"pp{s}-m{m}" for s, m in GEOMETRIES]
)


@pytest.mark.property
class TestSchedulesAreDeadlockFree:
    @every_geometry
    def test_fill_drain_completes_under_blocking_point_to_points(
        self, geometry: tuple[int, int]
    ) -> None:
        stages, m = geometry
        ok, where = deadlock_free(_world(fill_drain, stages, m))
        assert ok, where

    @every_geometry
    def test_one_f_one_b_completes_under_blocking_point_to_points(
        self, geometry: tuple[int, int]
    ) -> None:
        stages, m = geometry
        ok, where = deadlock_free(_world(one_f_one_b, stages, m))
        assert ok, where

    @every_geometry
    def test_every_link_pairs_its_operations_in_order(
        self, geometry: tuple[int, int]
    ) -> None:
        stages, m = geometry
        for schedule in (fill_drain, one_f_one_b):
            world = _world(schedule, stages, m)
            for link in range(stages - 1):
                lower, upper = link_sequences(world, link)
                assert lower == upper, (schedule.__name__, link, lower, upper)

    def test_the_textbook_order_deadlocks_at_two_stages(self) -> None:
        """The mutation the checker must see: sending the gradient before
        receiving the next residual parks stage 0 in ``send_act(1)`` and
        stage 1 in ``send_grad(0)``, each waiting for the other."""
        ok, where = deadlock_free(_world(one_f_one_b_textbook, 2, 2))
        assert not ok
        assert "stage 0 at send_act(1)" in where and "stage 1 at send_grad(0)" in where
        # one micro-batch has nothing to interleave: the textbook order is fine
        assert deadlock_free(_world(one_f_one_b_textbook, 2, 1))[0]


@pytest.mark.property
class TestSchedulesCoverEveryMicroBatch:
    @every_geometry
    def test_each_micro_batch_is_forwarded_once_and_backwarded_once_after(
        self, geometry: tuple[int, int]
    ) -> None:
        stages, m = geometry
        for s in range(stages):
            ops = one_f_one_b(stages, s, m)
            forwards = [op.micro for op in ops if op.kind == "F"]
            backwards = [op.micro for op in ops if op.kind == "B"]
            assert forwards == list(range(m))
            assert backwards == list(range(m))
            for i in range(m):
                assert ops.index(_f(i)) < ops.index(_b(i))
            assert [
                op.micro for op in fill_drain(stages, s, m) if op.kind == "F"
            ] == list(range(m))

    @every_geometry
    def test_the_drain_is_the_last_operation_and_happens_once(
        self, geometry: tuple[int, int]
    ) -> None:
        stages, m = geometry
        for schedule in (fill_drain, one_f_one_b):
            for s in range(stages):
                ops = schedule(stages, s, m)
                assert ops[-1] == DRAIN and ops.count(DRAIN) == 1

    @every_geometry
    def test_one_f_one_b_holds_at_most_the_classic_number_of_graphs(
        self, geometry: tuple[int, int]
    ) -> None:
        stages, m = geometry
        for s in range(stages):
            assert graphs_alive(one_f_one_b(stages, s, m)) == min(stages - s, m)

    @given(st.integers(1, 6))
    @_SETTINGS
    def test_one_micro_batch_is_todays_wire(self, stages: int) -> None:
        """§8.6: at ``m = 1`` the point-to-points are the placement mode's
        — the residual up, its gradient down, once per link."""
        for s in range(stages):
            wire = [op for op in one_f_one_b(stages, s, 1) if op.kind in P2P]
            expected: list[Op] = []
            if s > 0:
                expected.append(_ra(0))
            if s < stages - 1:
                expected += [_sa(0), _rg(0)]
            if s > 0:
                expected.append(_sg(0))
            assert wire == expected
            inference = [op for op in fill_drain(stages, s, 1) if op.kind in P2P]
            assert inference == [op for op in expected if op.kind.endswith("act")]


@pytest.mark.unit
class TestPinnedTranscripts:
    """The transcripts §8.6 prints, held to the rule that generates them."""

    def test_two_stages_three_micro_batches(self) -> None:
        assert one_f_one_b(2, 0, 3) == (
            _f(0), _sa(0),
            _f(1), _sa(1), _rg(0), _b(0),
            _f(2), _sa(2), _rg(1), _b(1),
            _rg(2), _b(2),
            DRAIN,
        )  # fmt: skip
        assert one_f_one_b(2, 1, 3) == (
            _ra(0), _f(0), _b(0),
            _ra(1), _sg(0), _f(1), _b(1),
            _ra(2), _sg(1), _f(2), _b(2),
            _sg(2),
            DRAIN,
        )  # fmt: skip

    def test_three_stages_three_micro_batches_middle_stage(self) -> None:
        assert one_f_one_b(3, 1, 3) == (
            _ra(0), _f(0), _sa(0),
            _ra(1), _f(1), _sa(1), _rg(0), _b(0),
            _ra(2), _sg(0), _f(2), _sa(2), _rg(1), _b(1),
            _sg(1), _rg(2), _b(2),
            _sg(2),
            DRAIN,
        )  # fmt: skip

    def test_fill_drain_two_stages(self) -> None:
        assert fill_drain(2, 0, 3) == (
            _f(0),
            _sa(0),
            _f(1),
            _sa(1),
            _f(2),
            _sa(2),
            DRAIN,
        )
        assert fill_drain(2, 1, 3) == (
            _ra(0),
            _f(0),
            _ra(1),
            _f(1),
            _ra(2),
            _f(2),
            DRAIN,
        )
