"""The train loop's agreements and gradient guard under simulation.

``docs/model_parallelism.md`` §3 (the three agreements), §7 (the gradient
guard), §10.4 (the ``OOM at (rank 1, step 3)`` and ``memory readings that
differ by rank`` scenarios). The **real** ``run_cohort_training`` runs on
every simulated rank over the tiny Llama DAS documents, each rank a full
copy — the model is replicated, so at ``tensor=2`` the run is the world-1
run exactly: the gradient guard's ``(g + g) / 2`` is ``g`` bit for bit, and
the agreements only ever *equalise* decisions every rank would have made from
the same tensors. The oracle is therefore the same program at world 1 with
the agreed decisions scripted, compared bit for bit.

The refusing collective proves the loop calls nothing at world 1; a rank
that skips the gradient guard is refused as a `Divergence` naming the
rank and both call sites; and the mutation the convention asks for — the
shrink decided locally, without ``any`` — fails the equal-shrinks scenario
under a schedule where one rank alone runs out of memory.

**The gradient agreement check is on** (§7, ``CAUSALAB_GRADIENT_AGREEMENT=0``
through ``conftest.checked_gradients_simulated``) in every scenario here: the
real loop reads the variable once per fit and holds the ranks' gradients to
bit identity before the mean. Through that seam — no monkeypatch of the
guard — a rank whose gradient is half the others' (its loss weighed by one
half, the ``1 / size`` a broken pairing would carry) is refused as a
[`GradientDisagreement`][causalab.neural.shared.parallel.agreements.GradientDisagreement] naming the variable, and a malformed value is
refused as an [`AgreementSetting`][causalab.neural.shared.parallel.agreements.AgreementSetting] before any forward, on every rank.
"""

# the cohort suite's builders are reused here
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, Callable, Sequence

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.engines.pytorch_hooks import cohort as cohort_module
from causalab.neural.engines.pytorch_hooks import train as train_module
from causalab.neural.engines.pytorch_hooks.budget import MARGIN, RowBudget
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training
from causalab.neural.shared.parallel.agreements import (
    AGREEMENT_VARIABLE,
    AgreementSetting,
    GradientDisagreement,
)
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Divergence,
    RankFailed,
    Refusal,
    Schedule,
    SimulatedMeter,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.refusing_collective import RefusingCollective

from ._drive import executor_for
from .conftest import TINY_LLAMA
from .test_fit_cohort import (
    PAIRS,
    _campaign,
    _request,
    _train_doc,
    _weights,
)
from .test_train import ANSWERS, BASES, COUNTERFACTUALS

# the §7 gradient agreement check on at bit identity (conftest.py): the
# real ``run_cohort_training`` reads the variable once per fit
pytestmark = [pytest.mark.usefixtures("checked_gradients_simulated")]

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

TP = 2
EPOCHS = 2
MEMBERS = (2, 4, 8, 16)
#: the probe reads one member of ``PAIRS`` rows: 100 bytes a row on every
#: rank, room for 900 rows on rank 0 and for 18 on rank 1 — one bound, 18
READINGS = {0: [(100 * PAIRS, 100_000)], 1: [(100 * PAIRS, 2_000)]}
#: rank 1's fourth grad window runs out of memory (0-based step 3)
OOM_AT = (1, 3)


def _bound(reading: tuple[int, int]) -> int:
    """``RowBudget.run``'s arithmetic, spelled by hand for one probe of ``PAIRS`` rows."""
    peak, available = reading
    fits = int(available * (1.0 - MARGIN) / (peak / PAIRS))
    return max(PAIRS, (fits // PAIRS) * PAIRS)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


Outcome = tuple[int | None, int, dict[str, torch.Tensor]]


def _fit_program(
    bundle: ModelBundle, meter: SimulatedMeter
) -> Callable[[int, Collective], list[Outcome]]:
    """The rank program: the real cohort fit over the DAS documents, the
    executors' fragments bound to this rank's collective."""

    def program(rank: int, c: Collective) -> list[Outcome]:
        raws = [_train_doc(k=k, epochs=EPOCHS) for k in MEMBERS]
        _docs, handles = _campaign(raws)
        executors = [
            executor_for(
                raw,
                bundle,
                base_texts=BASES,
                counterfactual_texts=COUNTERFACTUALS,
                extra_columns={"label": ANSWERS},
                interning=handle,
            )
            for raw, handle in zip(raws, handles)
        ]
        for executor in executors:
            executor.fragments = Fragments(c)
        outcomes = run_cohort_training(
            [ex.doc for ex in executors],
            executors,
            _request(),
            meter=meter.for_rank(rank),
        )
        return [(o.fit_rows, o.fit_rows_shrinks, _weights(o)) for o in outcomes]

    return program


class _GradWindows:
    """``run_groups`` counting each rank's grad windows and raising the meter's
    scripted OOM at ``(rank, window)`` — the seam ``test_row_budget`` uses,
    per rank."""

    def __init__(self, meter: SimulatedMeter) -> None:
        self.meter = meter
        self.windows: dict[int, int] = {}
        self.real = cohort_module.run_groups

    def __call__(self, entries: Sequence[Any]) -> None:
        executor = entries[0].executor
        if executor.grad_enabled:
            rank = executor.fragments.collective.rank("model")
            step = self.windows.get(rank, 0)
            self.windows[rank] = step + 1
            self.meter.check(rank, step)
        self.real(entries)


def _assert_same(a: Outcome, b: Outcome) -> None:
    assert a[0] == b[0] and a[1] == b[1]
    assert set(a[2]) == set(b[2])
    for name in a[2]:
        assert torch.equal(a[2][name], b[2][name]), name


@pytest.mark.unit
class TestRowBudgetAgreement:
    def test_readings_that_differ_by_rank_give_one_bound_on_every_rank(self) -> None:
        """§3's first agreement, through ``RowBudget`` itself: the probe's
        bound is the minimum over the model group."""
        meter = SimulatedMeter(READINGS)

        def program(rank: int, c: Collective) -> int | None:
            budget = RowBudget.of(None, meter.for_rank(rank), collective=c)
            budget.run(PAIRS, lambda: None, unit=PAIRS)
            return budget.bound

        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=4)
        bounds = world.run(program)
        expected = min(_bound(READINGS[0][0]), _bound(READINGS[1][0]))
        assert bounds == [expected, expected]
        assert _bound(READINGS[0][0]) != _bound(READINGS[1][0]), "the readings differ"

    def test_a_fixed_bound_agrees_nothing(self) -> None:
        budget = RowBudget.of(8, None, collective=RefusingCollective())
        budget.run(4, lambda: None)
        assert budget.bound == 8
        assert budget.out_of_memory(False) is False
        assert budget.out_of_memory(True) is True


class TestOutOfMemoryAgreement:
    @pytest.mark.unit
    def test_one_rank_out_of_memory_makes_every_rank_shrink_together(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scenario: rank 1 alone runs out of memory on its fourth grad
        window; under ``any`` both ranks abandon the window, halve the bound
        and re-pack — equal ``fit_rows_shrinks``, equal bounds, parameters
        bit-identical across ranks and to the world-1 run whose one rank
        takes the same decisions."""
        meter = SimulatedMeter(READINGS, oom_at={OOM_AT})
        windows = _GradWindows(meter)
        monkeypatch.setattr(train_module, "run_groups", windows)
        world = SimulatedWorld(
            groups_for(TP, tensor=TP), world=TP, schedule=7, timeout=600.0
        )
        results = world.run(_fit_program(bundle, meter))
        assert windows.windows[0] == windows.windows[1], (
            "every rank ran the same windows"
        )
        for rank_outcomes in results:
            for fit_rows, shrinks, _ in rank_outcomes:
                assert shrinks == 1
                # the agreed bound 18 held 4 members (8 rows); halved to 4 rows
                assert fit_rows == 2 * PAIRS
        for a, b in zip(results[0], results[1]):
            _assert_same(a, b)

        # the world-1 oracle: one rank, the agreed reading, the same OOM step
        solo_meter = SimulatedMeter({0: READINGS[1]}, oom_at={(0, OOM_AT[1])})
        monkeypatch.setattr(train_module, "run_groups", _GradWindows(solo_meter))
        solo = SimulatedWorld(groups_for(1), world=1, schedule=0, timeout=600.0)
        (oracle,) = solo.run(_fit_program(bundle, solo_meter))
        for a, b in zip(results[0], oracle):
            _assert_same(a, b)

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_the_fit_does_not_depend_on_the_schedule(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch, schedule: Schedule
    ) -> None:
        meter = SimulatedMeter(READINGS, oom_at={OOM_AT})
        monkeypatch.setattr(train_module, "run_groups", _GradWindows(meter))
        world = SimulatedWorld(
            groups_for(TP, tensor=TP), world=TP, schedule=schedule, timeout=600.0
        )
        results = world.run(_fit_program(bundle, meter))
        for a, b in zip(results[0], results[1]):
            _assert_same(a, b)

    @pytest.mark.unit
    def test_at_world_one_the_loop_calls_no_collective(
        self, bundle: ModelBundle
    ) -> None:
        raws = [_train_doc(k=k, epochs=1) for k in (2, 4)]
        _docs, handles = _campaign(raws)
        executors = [
            executor_for(
                raw,
                bundle,
                base_texts=BASES,
                counterfactual_texts=COUNTERFACTUALS,
                extra_columns={"label": ANSWERS},
                interning=handle,
            )
            for raw, handle in zip(raws, handles)
        ]
        for executor in executors:
            executor.fragments = Fragments(RefusingCollective())
        outcomes = run_cohort_training(
            [ex.doc for ex in executors], executors, _request(), meter=None
        )
        assert all(o.fit_rows is None and o.fit_rows_shrinks == 0 for o in outcomes)


@pytest.mark.unit
class TestGradientGuard:
    def test_a_rank_skipping_the_guard_is_a_divergence(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = train_module.average_gradients

        def skipping(parameters: Any, collective: Collective, **kwargs: Any) -> None:
            if collective.rank("model") == 1:
                return  # §3's forbidden branch
            real(parameters, collective, **kwargs)

        monkeypatch.setattr(train_module, "average_gradients", skipping)
        meter = SimulatedMeter(READINGS)
        world = SimulatedWorld(
            groups_for(TP, tensor=TP), world=TP, schedule=1, timeout=600.0
        )
        with pytest.raises(Divergence) as err:
            world.run(_fit_program(bundle, meter))
        assert err.value.group == (0, 1)
        # the guard's first collective is the check's all-gather (the
        # scenarios run checked, module docstring) — the plain mean's
        # all-reduce without it; rank 1 is already at the next agreement
        ops = {err.value.op, err.value.first_op}
        assert ops & {"all_gather", "all_reduce_sum"}, ops
        assert "agreements.py" in err.value.call_site + err.value.first_call_site


@pytest.mark.unit
class TestGradientAgreementSetting:
    def test_the_scenarios_run_checked(self) -> None:
        import os

        assert os.environ[AGREEMENT_VARIABLE] == "0"

    def test_a_rank_with_half_the_gradient_is_refused_through_the_variable(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``1 / size`` a broken pairing would carry, spelled as one
        rank's loss halved (``train._loss``, the minibatch executor's
        collective naming the rank): the real loop's guard refuses it by
        name with nothing patched but the loss."""
        real = train_module._loss

        def halved(fit: Any, minibatch: Any) -> torch.Tensor:
            loss = real(fit, minibatch)
            if minibatch.fragments.collective.rank("model") == 1:
                return loss * 0.5
            return loss

        monkeypatch.setattr(train_module, "_loss", halved)
        meter = SimulatedMeter(READINGS)
        world = SimulatedWorld(
            groups_for(TP, tensor=TP), world=TP, schedule=3, timeout=600.0
        )
        with pytest.raises(RankFailed) as err:
            world.run(_fit_program(bundle, meter))
        cause = err.value.__cause__
        assert isinstance(cause, GradientDisagreement)
        assert AGREEMENT_VARIABLE in str(cause) and "5.000e-01" in str(cause)

    def test_a_malformed_value_is_refused_by_name_before_any_forward(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(AGREEMENT_VARIABLE, "0.5")
        meter = SimulatedMeter(READINGS)
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        with pytest.raises(RankFailed) as err:
            world.run(_fit_program(bundle, meter))
        assert isinstance(err.value.__cause__, AgreementSetting)
        assert f"{AGREEMENT_VARIABLE}='0.5'" in str(err.value.__cause__)
        assert world.transcript == [], "refused before the first collective"


@pytest.mark.unit
class TestMutations:
    def test_a_shrink_decided_locally_fails_the_equal_shrinks_scenario(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``any``, rank 1 alone halves its bound: its later windows
        differ from rank 0's, so either the shrink counts disagree or the
        ranks reach different collectives — the scenario cannot pass."""
        monkeypatch.setattr(RowBudget, "out_of_memory", lambda self, failed: failed)
        meter = SimulatedMeter(READINGS, oom_at={OOM_AT})
        monkeypatch.setattr(train_module, "run_groups", _GradWindows(meter))
        world = SimulatedWorld(
            groups_for(TP, tensor=TP), world=TP, schedule=7, timeout=600.0
        )
        with pytest.raises((AssertionError, Refusal)):
            results = world.run(_fit_program(bundle, meter))
            for a, b in zip(results[0], results[1]):
                assert a[1] == b[1], "fit_rows_shrinks differ"
