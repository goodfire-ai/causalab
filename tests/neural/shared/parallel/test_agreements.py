"""The three host-side agreements and the training guard through one seam.

``docs/model_parallelism.md`` §3 (the agreements table), §6.5 (fire counts
summed over the stages), §7 (the gradient guard) and the §10.3 rows
``fire counts`` and ``agreements``. [`Agreements`][causalab.neural.shared.parallel.agreements.Agreements] is the identity at
world 1 without a call on the collective — the refusing collective proves
it — and the group aggregate under the simulated world. The mutation the
convention asks for closes the file: a fire sum over the wrong axis fails
the stage property.

**The gradient agreement check as a runtime setting** (§7,
``CAUSALAB_GRADIENT_AGREEMENT``). The variable is parsed into a tolerance in
``[0, 1/2)`` or refused by name — not a number, negative, not finite, one
half or more — and never silently ignored; unset or empty is no check. The
tolerance is *relative* to the gathered gradients' largest entry, and two
properties pin what that buys: identical gradients pass at tolerance ``0``
(bit identity, whatever their magnitude), and a rank holding ``1 / size``
of a non-zero gradient is refused at **every** magnitude and every
tolerance the setting admits, for every group size from two up — the
partial a broken pairing hands a rank differs from the whole by
``1 − 1/size ≥ 1/2`` of its largest entry, which is why the setting stops
below one half. The refusal names the variable, the parameter and the
relative disagreement.
"""

from __future__ import annotations


import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.fires import FireTally, check_fires
from causalab.neural.shared.parallel.agreements import (
    AGREEMENT_CEILING,
    AGREEMENT_VARIABLE,
    AgreementSetting,
    Agreements,
    GradientDisagreement,
    average_gradients,
    configured_agreement,
    parse_agreement,
    relative_disagreement,
    summed_fires,
    whole_steps,
)
from causalab.neural.shared.parallel.collective import SOLO, Collective, Solo
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from causalab.protocol.rules.errors import ProtocolError
from tests._helpers.refusing_collective import RefusingCollective

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

STAGES = 4


def _stage_tally(declarations: tuple[ps.FireDeclaration, ...], stage: int) -> FireTally:
    """What one stage's forward tallies: every member declared (the write
    set resolves identically on every rank), fired only where owned."""
    tally = FireTally()
    for declaration in declarations:
        tally.declare([declaration.member], declaration.count)
        if declaration.stage == stage:
            for _ in range(declaration.count):
                tally.fired([declaration.member])
    return tally


def _run_stages(
    declarations: tuple[ps.FireDeclaration, ...],
    schedule: Schedule = (),
    axis: str = "pipeline",
) -> list[FireTally]:
    world = SimulatedWorld(
        groups_for(STAGES, pipeline=STAGES), world=STAGES, schedule=schedule
    )

    def program(rank: int, c: Collective) -> FireTally:
        tally = _stage_tally(declarations, c.rank("pipeline"))
        return summed_fires(tally, Agreements(c), axis=axis)  # type: ignore[arg-type]

    return world.run(program)


@pytest.mark.property
class TestAgreementProperties:
    @_SETTINGS
    @given(declarations=ps.fire_declarations(stages=STAGES), schedule=ps.schedules())
    @example(
        declarations=(ps.FireDeclaration(member="a", count=1, stage=0),), schedule=[]
    )
    def test_per_stage_tallies_sum_to_the_declared_count_on_every_stage(
        self, declarations: tuple[ps.FireDeclaration, ...], schedule: Schedule
    ) -> None:
        for tally in _run_stages(declarations, schedule):
            for declaration in declarations:
                assert tally.counts[declaration.member] == declaration.count
            check_fires("m on base", tally)  # passes on every rank

    @_SETTINGS
    @given(declarations=ps.fire_declarations(stages=STAGES), schedule=ps.schedules())
    def test_a_member_missing_on_every_stage_is_refused_naming_it(
        self, declarations: tuple[ps.FireDeclaration, ...], schedule: Schedule
    ) -> None:
        # a member owned by a stage that does not exist fires nowhere
        missing = ps.FireDeclaration(member="zz_missing", count=1, stage=STAGES)
        for tally in _run_stages((*declarations, missing), schedule):
            with pytest.raises(ProtocolError, match="zz_missing"):
                check_fires("m on base", tally)

    @_SETTINGS
    @given(readings=ps.memory_readings(world=4, probes=1), schedule=ps.schedules())
    def test_min_any_and_sum_are_the_group_aggregates(
        self, readings: dict[int, tuple[tuple[int, int], ...]], schedule: Schedule
    ) -> None:
        world = SimulatedWorld(
            groups_for(4, data=2, tensor=2), world=4, schedule=schedule
        )

        def program(rank: int, c: Collective) -> tuple[int, bool, int]:
            agreements = Agreements(c)
            peak, available = readings[rank][0]
            return (
                agreements.min(available, "tensor"),
                agreements.any(peak % 2 == 0, "tensor"),
                agreements.sum(peak, "tensor"),
            )

        results = world.run(program)
        for group in ((0, 1), (2, 3)):
            members = [readings[r][0] for r in group]
            expected = (
                min(a for _, a in members),
                any(p % 2 == 0 for p, _ in members),
                sum(p for p, _ in members),
            )
            for rank in group:
                assert results[rank] == expected


@pytest.mark.unit
class TestAgreements:
    def test_world_one_is_the_identity_without_touching_the_collective(self) -> None:
        agreements = Agreements(RefusingCollective())
        assert agreements.min(7, "model") == 7
        assert agreements.any(True, "model") is True
        assert agreements.sum(3, "pipeline") == 3
        tally = FireTally()
        tally.declare(["w"], 1)
        tally.fired(["w"])
        assert summed_fires(tally, agreements).counts == {"w": 1}
        grads = [torch.nn.Parameter(torch.ones(2))]
        grads[0].grad = torch.full((2,), 3.0)
        average_gradients(grads, RefusingCollective(), axis="model")
        assert torch.equal(grads[0].grad, torch.full((2,), 3.0))
        assert Agreements(SOLO).sum(2, "model") == 2

    def test_summed_fires_keeps_the_declaration_and_the_state_steps(self) -> None:
        world = SimulatedWorld(groups_for(2, pipeline=2), world=2, schedule=0)

        def program(rank: int, c: Collective) -> FireTally:
            tally = FireTally()
            tally.declare(["w", "s"], 1)
            if rank == 1:
                tally.fired(["w"])
            tally.fired(["s"], step=rank)  # a state write fires at its own steps
            return summed_fires(tally, Agreements(c))

        for rank, tally in enumerate(world.run(program)):
            assert tally.expected == {"w": 1, "s": 1}
            assert tally.counts == {"w": 1, "s": 2}
            assert tally.steps == {"s": {rank}}, "steps stay the rank's own"

    def test_the_gradient_guard_averages_over_the_model_group(self) -> None:
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(rank: int, c: Collective) -> torch.Tensor:
            parameter = torch.nn.Parameter(torch.zeros(3))
            parameter.grad = torch.full((3,), float(rank + 1))
            untouched = torch.nn.Parameter(torch.zeros(1))  # no grad: skipped
            average_gradients([parameter, untouched], c, axis="model")
            assert untouched.grad is None
            return parameter.grad

        for grad in world.run(program):
            assert torch.equal(grad, torch.full((3,), 1.5))

    def test_a_group_of_two_averages_an_equal_gradient_bit_for_bit(self) -> None:
        """§7: a replicated site's gradient is identical on every rank, and
        ``(g + g) / 2 == g`` exactly, so the guard costs no rounding at tp=2."""
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)
        grad = torch.randn(64, generator=torch.Generator().manual_seed(3))

        def program(rank: int, c: Collective) -> torch.Tensor:
            parameter = torch.nn.Parameter(torch.zeros(64))
            parameter.grad = grad.clone()
            average_gradients([parameter], c, axis="model")
            return parameter.grad

        for out in world.run(program):
            assert torch.equal(out, grad)


@pytest.mark.unit
class TestAgreementMutations:
    def test_a_fire_sum_over_the_wrong_axis_fails_the_stage_property(self) -> None:
        declarations = (
            ps.FireDeclaration(member="a", count=1, stage=0),
            ps.FireDeclaration(member="b", count=2, stage=3),
        )
        for tally in _run_stages(declarations):
            check_fires("m on base", tally)
        wrong = _run_stages(declarations, axis="tensor")
        with pytest.raises(ProtocolError):
            for tally in wrong:
                check_fires("m on base", tally)


class TestWholeStepsDevice:
    pytestmark = pytest.mark.unit

    def test_the_step_vector_is_born_on_the_collectives_device(self) -> None:
        """Create the step vector on the collective's device: CUDA for NCCL,
        CPU for the simulator."""
        from causalab.neural.shared.fires import FireTally
        from causalab.neural.shared.parallel.agreements import whole_steps

        class _DeviceCollective(Solo):
            device = torch.device("meta")

            def size(self, axis):  # type: ignore[override]
                return 2

            def all_reduce_sum(self, tensor, axis):  # type: ignore[override]
                assert tensor.device.type == "meta", tensor.device
                out = torch.zeros(tensor.shape[0], dtype=torch.int64)
                out[1] = 1
                out[3] = 1
                return out

        tally = FireTally(expected={"w": 2}, counts={"w": 1}, steps={"w": (1,)})
        whole = whole_steps(tally, _DeviceCollective(), ["w"], padded_len=4)
        assert whole.steps["w"] == {1, 3} and whole.counts["w"] == 2


# --------------------------------------------------------------------------- #
# the gradient agreement check as a runtime setting (module docstring, §7)
# --------------------------------------------------------------------------- #

#: Finite fp32 magnitudes from the smallest normal to near the largest, so
#: a property over "every magnitude" walks the whole exponent range.
_magnitudes = st.floats(
    min_value=1e-37, max_value=1e37, allow_nan=False, allow_infinity=False
)
_tolerances = st.floats(
    min_value=0.0,
    max_value=AGREEMENT_CEILING,
    exclude_max=True,
    allow_nan=False,
    allow_infinity=False,
)


def _gradient(seed: int, magnitude: float, numel: int) -> torch.Tensor:
    """A non-zero fp32 gradient of ``numel`` entries at ``magnitude``: signed
    values in ``(−1, 1]`` times the magnitude, the first entry pinned to the
    magnitude itself so the tensor is never all zeros."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(numel, generator=generator) * 2.0 - 1.0
    values[0] = 1.0
    return (values * magnitude).to(torch.float32)


@pytest.mark.unit
class TestAgreementSetting:
    def test_unset_and_empty_are_no_check(self) -> None:
        assert parse_agreement(None) is None
        assert parse_agreement("") is None
        assert parse_agreement("  ") is None
        assert configured_agreement({}) is None

    @pytest.mark.parametrize(
        "text, expected", [("0", 0.0), ("1e-6", 1e-6), ("0.25", 0.25)]
    )
    def test_a_tolerance_in_the_range_is_read(self, text: str, expected: float) -> None:
        assert parse_agreement(text) == expected
        assert configured_agreement({AGREEMENT_VARIABLE: text}) == expected

    @pytest.mark.parametrize(
        "text", ["abc", "-1e-6", "nan", "inf", "0.5", "1", "1e3", "-0.0001"]
    )
    def test_a_malformed_or_out_of_range_value_is_refused_by_name(
        self, text: str
    ) -> None:
        with pytest.raises(AgreementSetting, match=AGREEMENT_VARIABLE):
            parse_agreement(text)
        with pytest.raises(AgreementSetting, match=AGREEMENT_VARIABLE):
            configured_agreement({AGREEMENT_VARIABLE: text})

    def test_the_variable_is_the_documented_name(self) -> None:
        assert AGREEMENT_VARIABLE == "CAUSALAB_GRADIENT_AGREEMENT"
        assert AGREEMENT_CEILING == 0.5

    def test_the_refusal_names_the_parameter_the_tolerance_and_the_variable(
        self,
    ) -> None:
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(rank: int, c: Collective) -> None:
            first = torch.nn.Parameter(torch.zeros(3))
            first.grad = torch.ones(3)
            second = torch.nn.Parameter(torch.zeros(4))
            second.grad = torch.full((4,), 1.0 if rank == 0 else 0.5)
            average_gradients([first, second], c, axis="model", agreement=1e-6)

        with pytest.raises(RankFailed) as err:
            world.run(program)
        cause = err.value.__cause__
        assert isinstance(cause, GradientDisagreement)
        message = str(cause)
        assert "parameter 1" in message and AGREEMENT_VARIABLE in message
        assert "5.000e-01" in message and "1.0e-06" in message


@pytest.mark.property
class TestAgreementToleranceProperties:
    @_SETTINGS
    @given(
        magnitude=_magnitudes,
        seed=st.integers(0, 2**16),
        numel=st.integers(1, 64),
        size=st.integers(2, 4),
    )
    def test_identical_gradients_pass_at_tolerance_zero(
        self, magnitude: float, seed: int, numel: int, size: int
    ) -> None:
        grad = _gradient(seed, magnitude, numel)
        gathered = grad.reshape(1, -1).repeat(size, 1)
        assert relative_disagreement(gathered) == 0.0

    @_SETTINGS
    @given(
        magnitude=_magnitudes,
        seed=st.integers(0, 2**16),
        numel=st.integers(1, 64),
        size=st.integers(2, 4),
        scaled=st.integers(0, 3),
        tolerance=_tolerances,
    )
    def test_a_rank_holding_one_over_size_of_the_gradient_is_refused_at_every_magnitude(
        self,
        magnitude: float,
        seed: int,
        numel: int,
        size: int,
        scaled: int,
        tolerance: float,
    ) -> None:
        """One rank — any rank — holds ``g / size`` while the others hold
        ``g``: the disagreement is at least ``1 − 1/size ≥ 1/2`` of the
        largest entry, above every tolerance the setting admits."""
        grad = _gradient(seed, magnitude, numel)
        rows = [grad.clone() for _ in range(size)]
        rows[scaled % size] = grad / size
        worst = relative_disagreement(torch.stack(rows))
        assert worst >= 1.0 - 1.0 / size - 1e-6
        assert worst > tolerance

    @_SETTINGS
    @given(
        magnitude=_magnitudes,
        seed=st.integers(0, 2**16),
        size=st.integers(2, 4),
        tolerance=_tolerances,
        schedule=ps.schedules(),
    )
    def test_the_guard_refuses_a_scaled_rank_and_passes_identical_ranks_under_simulation(
        self,
        magnitude: float,
        seed: int,
        size: int,
        tolerance: float,
        schedule: Schedule,
    ) -> None:
        """The same two statements through [`average_gradients`][causalab.neural.shared.parallel.agreements.average_gradients] itself
        on a simulated group of ``size`` (every geometry's ``model`` group):
        identical gradients pass at ``0`` and the mean is the gradient; one
        rank at ``1 / size`` is refused as [`GradientDisagreement`][causalab.neural.shared.parallel.agreements.GradientDisagreement]."""
        grad = _gradient(seed, magnitude, 8)

        def identical(rank: int, c: Collective) -> torch.Tensor:
            parameter = torch.nn.Parameter(torch.zeros(8))
            parameter.grad = grad.clone()
            average_gradients([parameter], c, axis="model", agreement=0.0)
            return parameter.grad

        def scaled(rank: int, c: Collective) -> None:
            parameter = torch.nn.Parameter(torch.zeros(8))
            parameter.grad = (
                grad / size if c.rank("model") == size - 1 else grad.clone()
            )
            average_gradients([parameter], c, axis="model", agreement=tolerance)

        world = SimulatedWorld(
            groups_for(size, tensor=size), world=size, schedule=schedule
        )
        for out in world.run(identical):
            # the mean of ``size`` equal terms: exact at a power of two, one
            # rounding step otherwise
            assert torch.allclose(out, grad, rtol=2e-7, atol=0.0)
        with pytest.raises(RankFailed) as err:
            world.run(scaled)
        assert isinstance(err.value.__cause__, GradientDisagreement)

    @_SETTINGS
    @given(seed=st.integers(0, 2**16), numel=st.integers(0, 16))
    def test_zeros_and_empties_agree(self, seed: int, numel: int) -> None:
        assert relative_disagreement(torch.zeros(3, numel)) == 0.0
        assert relative_disagreement(torch.empty(2, 0)) == 0.0


@pytest.mark.unit
class TestGradientlessParameters:
    """A gradient-less parameter must not stop reduction of later parameters."""

    def test_a_parameter_without_a_gradient_does_not_stop_the_averaging(self) -> None:
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(
            rank: int, c: Collective
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
            unused = torch.nn.Parameter(torch.zeros(2))
            used = torch.nn.Parameter(torch.zeros(3))
            used.grad = torch.full((3,), float(rank + 1))
            average_gradients([unused, used], c, axis="tensor")
            return unused.grad, used.grad

        for unused_grad, used_grad in world.run(program):
            assert unused_grad is None
            assert torch.equal(used_grad, torch.full((3,), 1.5))


@pytest.mark.unit
class TestWholeStepsUnion:
    def test_the_fired_steps_are_unioned_over_the_context_group(self) -> None:
        """Two ranks of a context group fired a state writer at disjoint
        positions of a six-position frame — one of them nowhere at all, so
        its tally never names the member: the union is both ranks' steps,
        the count its size, the declaration kept and the other member left
        as each rank tallied it. (The device test above feeds the reduction
        a scripted vector; this one reduces the ranks' real vectors.)"""
        world = SimulatedWorld(groups_for(2, context=2), world=2, schedule=0)
        fired_by_rank: tuple[tuple[int, ...], ...] = ((0, 2, 5), ())

        def program(rank: int, c: Collective) -> FireTally:
            r = c.rank("context")
            steps: dict[str, tuple[int, ...]] = {"other": (r,)}
            counts = {"other": 1}
            if fired_by_rank[r]:
                steps["w"] = fired_by_rank[r]
                counts["w"] = len(fired_by_rank[r])
            tally = FireTally(expected={"w": 6, "other": 1}, counts=counts, steps=steps)
            return whole_steps(tally, c, ["w"], padded_len=6)

        for rank, whole in enumerate(world.run(program)):
            assert whole.steps["w"] == {0, 2, 5}
            assert whole.counts["w"] == 3
            assert whole.expected == {"w": 6, "other": 1}
            assert whole.counts["other"] == 1
            assert set(whole.steps["other"]) == {rank}
