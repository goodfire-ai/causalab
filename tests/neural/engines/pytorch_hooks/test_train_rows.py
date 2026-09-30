"""Data parallelism over rows under simulation (``docs/model_parallelism.md``
§7, §8.3, §10.4: the DAS fit scenarios at ``dp=2:rows``).

The **real** ``run_cohort_training`` runs on every simulated rank over the
tiny Llama DAS documents, each rank a full copy of the model, the collective
simulated: every minibatch of ``PAIRS`` rows is split one row to a replica,
each replica's weighed loss is backward'ed on its row, the gradient is summed
over the replicas and the update's record agreed (``rows.py``). The oracle is
the same program at world 1 over the unsplit minibatches.

Two claims, held apart. **Across the replicas** the fits are **bit-identical**
— the summed gradient is one tensor on every replica, so every optimizer
step sees identical inputs — and so are the eval scores and the early-stop
decisions made from them (a divergent stop would be a deadlock). **Against
world 1** the fit lands within `BAND`: the split gradient adds the same
per-row terms in a different order, which is the one difference §7 predicts,
and here it is measured — the largest parameter difference after twelve
AdamW updates in fp32 is ``2^-24 = 5.96e-08``, one ulp of a weight in
``[0.5, 1)`` (the Cayley rotation is orthogonal, so its entries are at most
one). The band is pinned at ``2.5e-7`` relative to a magnitude of at least
one — four ulps of such a weight, absolute here — and both mutations
the convention asks for — the mean over the replica's rows rather than the
minibatch's, and the reduce skipped on every replica — land five orders of
magnitude outside it. The fp64 twin runs on the loss and gradient functions
alone (``test_rows.py``): the loader carries no fp64 dtype.

The §3 agreements extend to the data axis: one replica running out of memory
shrinks every replica's bound, readings that differ by replica give one
bound, and a replica that skips the gradient reduce is refused as a
`Divergence` naming both call sites. Drawn schedules — tapes and
seeds — change nothing.
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
from causalab.neural.engines.pytorch_hooks.budget import RowBudget
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.rows import RowSplit
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.protocol.parallel import ONE, ParallelGeometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Divergence,
    Schedule,
    SimulatedMeter,
    SimulatedWorld,
    groups_for,
)

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

DATA = 2
GEOMETRY = ParallelGeometry(data=DATA, data_mode="rows")
#: ``PAIRS`` rows over the four training rows: two minibatches an epoch,
#: twelve updates
EPOCHS = 6
UPDATES = 12
MEMBERS = (2, 4)
#: The training band (§7, §8.3): the largest ``|w_rows - w_world1|`` over
#: every trained parameter after `UPDATES` fp32 AdamW updates.
#: Measured ``5.96e-08`` on this fixture — one ulp of a weight in ``[0.5, 1)``
#: — pinned at four ulps. A rotation's entries are at most one, so this is an
#: absolute band on weights of order one.
BAND = 2.5e-7
#: what the fixture lands at: exactly one ulp of a weight in ``[0.5, 1)``
MEASURED = 2.0**-24
EARLY_STOP = {"on": "ce", "patience": 1, "mode": "min"}
#: the probe reads one member's row on every replica; room for many rows on
#: replica 0 and for few on replica 1 — one bound
READINGS = {0: [(100 * PAIRS, 100_000)], 1: [(100 * PAIRS, 2_000)]}
#: replica 1's fourth grad window runs out of memory (0-based step 3)
OOM_AT = (1, 3)


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


Eval = tuple[int, str, dict[str, float]] | None
Outcome = tuple[int | None, int, dict[str, torch.Tensor], Eval]


def _outcome(outcome: Any) -> Outcome:
    score = outcome.eval_score
    return (
        outcome.fit_rows,
        outcome.fit_rows_shrinks,
        _weights(outcome),
        (score.passes, score.selected, dict(score.metrics)) if score else None,
    )


def _fit_program(
    bundle: ModelBundle,
    meter: SimulatedMeter | None,
    *,
    geometry: ParallelGeometry = GEOMETRY,
    early_stop: dict[str, Any] | None = None,
    pairs: int = PAIRS,
    eval_every: int | None = 1,
) -> Callable[[int, Collective], list[Outcome]]:
    """The rank program: the real cohort fit over the DAS documents, the
    executors' fragments and rows split bound to this rank's collective.
    ``pairs`` is the minibatch (``PAIRS``, two rows, unless a scenario needs
    an uneven split); ``eval_every=None`` drops the eval."""

    def program(rank: int, c: Collective) -> list[Outcome]:
        raws = [
            _train_doc(k=k, epochs=EPOCHS, early_stop=early_stop, eval_every=eval_every)
            for k in MEMBERS
        ]
        for raw in raws:
            raw["method"]["train"]["batch"] = {"pairs": pairs}
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
            executor.rows = RowSplit(c, geometry)
        outcomes = run_cohort_training(
            [ex.doc for ex in executors],
            executors,
            _request(),
            meter=meter.for_rank(rank) if meter is not None else None,
        )
        return [_outcome(o) for o in outcomes]

    return program


def _world(schedule: Schedule, data: int = DATA) -> SimulatedWorld:
    return SimulatedWorld(
        groups_for(data, data=data), world=data, schedule=schedule, timeout=600.0
    )


def _solo(
    bundle: ModelBundle,
    meter: SimulatedMeter | None = None,
    **fit: Any,
) -> list[Outcome]:
    """The world-1 oracle: the same program, the unsplit minibatches."""
    world = SimulatedWorld(groups_for(1), world=1, schedule=0, timeout=600.0)
    return world.run(_fit_program(bundle, meter, geometry=ONE, **fit))[0]


class _GradWindows:
    """``run_groups`` counting each replica's grad windows, recording the rows
    each carried and raising the meter's scripted OOM at ``(replica, window)``
    — the seam ``test_train_parallel`` uses, keyed by the data axis."""

    def __init__(self, meter: SimulatedMeter | None = None) -> None:
        self.meter = meter
        self.windows: dict[int, int] = {}
        self.rows: dict[int, set[int]] = {}
        self.real = cohort_module.run_groups

    def __call__(self, entries: Sequence[Any]) -> None:
        executor = entries[0].executor
        if executor.grad_enabled:
            replica = executor.fragments.collective.rank("data")
            step = self.windows.get(replica, 0)
            self.windows[replica] = step + 1
            self.rows.setdefault(replica, set()).update(
                len(entry.executor.rows_for_metrics()) for entry in entries
            )
            if self.meter is not None:
                self.meter.check(replica, step)
        self.real(entries)


def _distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert set(a) == set(b)
    return max(float((a[name] - b[name]).abs().max()) for name in a)


def _assert_bit_identical(results: Sequence[Sequence[Outcome]]) -> None:
    """Every replica holds the same outcome, to the bit."""
    for rank_outcomes in results[1:]:
        for a, b in zip(results[0], rank_outcomes, strict=True):
            assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3]
            assert _distance(a[2], b[2]) == 0.0


def _assert_within_band(rows: Sequence[Outcome], solo: Sequence[Outcome]) -> None:
    """The split fit against the unsplit one: the same eval passes and
    selection, the same shrinks, the parameters within `BAND`."""
    for a, b in zip(rows, solo, strict=True):
        assert a[1] == b[1], "fit_rows_shrinks"
        if a[3] is None or b[3] is None:
            assert a[3] is None and b[3] is None
        else:
            assert a[3][:2] == b[3][:2], "eval passes and selection"
            for name, value in a[3][2].items():
                assert value == pytest.approx(b[3][2][name], rel=1e-6)
        assert _distance(a[2], b[2]) <= BAND


# --------------------------------------------------------------------------- #
# the scenario: DAS fit at dp=2:rows, twelve updates
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestRowsFit:
    def test_replicas_agree_bit_for_bit_and_match_world_one_within_the_band(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        windows = _GradWindows()
        monkeypatch.setattr(train_module, "run_groups", windows)
        results = _world(0).run(_fit_program(bundle, None))
        _assert_bit_identical(results)
        # every grad window on every replica carried one row per member: the
        # minibatch was split, not replicated
        assert windows.windows == {0: UPDATES, 1: UPDATES}
        assert windows.rows == {0: {PAIRS // DATA}, 1: {PAIRS // DATA}}
        assert all(o[0] is None and o[1] == 0 for o in results[0])

        monkeypatch.setattr(train_module, "run_groups", cohort_module.run_groups)
        oracle = _solo(bundle)
        _assert_within_band(results[0], oracle)
        # the band is measured, not assumed: what this fixture lands at
        distance = max(_distance(a[2], b[2]) for a, b in zip(results[0], oracle))
        assert distance <= MEASURED + 1e-12, distance

    def test_early_stop_decisions_are_identical_on_every_replica(
        self, bundle: ModelBundle
    ) -> None:
        """With ``patience: 1`` on the eval ``ce`` a member stops before its
        epochs run out; every replica stops it at the same pass and selects
        the same weights, and so does the world-1 fit."""
        results = _world(3).run(_fit_program(bundle, None, early_stop=EARLY_STOP))
        _assert_bit_identical(results)
        evals = [o[3] for o in results[0]]
        assert all(e is not None and e[1] == "early_stop.best" for e in evals)
        assert any(e is not None and e[0] < EPOCHS for e in evals), (
            "the scenario should stop a member early"
        )
        _assert_within_band(results[0], _solo(bundle, early_stop=EARLY_STOP))


# --------------------------------------------------------------------------- #
# the agreements over the data axis
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestBudgetAgreement:
    def test_readings_that_differ_by_replica_give_one_bound(self) -> None:
        meter = SimulatedMeter(READINGS)

        def program(rank: int, c: Collective) -> int | None:
            split = RowSplit(c, GEOMETRY)
            budget = RowBudget.of(None, meter.for_rank(rank), c, split.budget_axes)
            budget.run(PAIRS, lambda: None, unit=PAIRS)
            return budget.bound

        bounds = _world(4).run(program)
        assert bounds[0] == bounds[1]
        assert bounds[0] == min(bounds), "the min over the replicas"

    def test_over_points_the_budget_agrees_nothing_on_the_data_axis(self) -> None:
        """The points mode's replicas run different points and must not be
        agreed: their bounds stay their own."""
        meter = SimulatedMeter(READINGS)

        def program(rank: int, c: Collective) -> int | None:
            split = RowSplit(c, ParallelGeometry(data=DATA))
            budget = RowBudget.of(None, meter.for_rank(rank), c, split.budget_axes)
            budget.run(PAIRS, lambda: None, unit=PAIRS)
            return budget.bound

        bounds = _world(4).run(program)
        assert bounds[0] != bounds[1]

    def test_one_replica_out_of_memory_shrinks_every_replica(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replica 1 alone runs out of memory on its fourth grad window;
        under ``any`` over the data axis both replicas abandon the window,
        halve the bound and re-pack — equal shrinks, equal bounds,
        bit-identical parameters — and the fit stays within the band of the
        world-1 run whose one rank takes the same decisions. The bound is in
        the replica's own rows: one member's row here, one member's
        ``PAIRS`` rows at world 1."""
        meter = SimulatedMeter(READINGS, oom_at={OOM_AT})
        windows = _GradWindows(meter)
        monkeypatch.setattr(train_module, "run_groups", windows)
        results = _world(7).run(_fit_program(bundle, meter))
        assert windows.windows[0] == windows.windows[1], (
            "every replica ran the same windows"
        )
        _assert_bit_identical(results)
        for fit_rows, shrinks, _, _ in results[0]:
            assert shrinks == 1
            assert fit_rows == PAIRS // DATA  # halved to one member's row

        solo_meter = SimulatedMeter({0: READINGS[1]}, oom_at={(0, OOM_AT[1])})
        monkeypatch.setattr(train_module, "run_groups", _GradWindows(solo_meter))
        oracle = _solo(bundle, solo_meter)
        for fit_rows, shrinks, _, _ in oracle:
            assert shrinks == 1 and fit_rows == PAIRS
        _assert_within_band(results[0], oracle)


@pytest.mark.unit
class TestDivergence:
    def test_a_replica_skipping_the_gradient_reduce_is_a_divergence(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = RowSplit.reduce_gradients

        def skipping(self: RowSplit, parameters: Any) -> None:
            if self.replica == 1:
                return  # §3's forbidden branch
            real(self, parameters)

        monkeypatch.setattr(RowSplit, "reduce_gradients", skipping)
        with pytest.raises(Divergence) as err:
            _world(1).run(_fit_program(bundle, None))
        assert err.value.group == (0, 1)
        assert "all_reduce_sum" in (err.value.op, err.value.first_op)


# --------------------------------------------------------------------------- #
# schedule independence: drawn schedules
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def reference(bundle: ModelBundle) -> list[list[Outcome]]:
    return _world(0).run(_fit_program(bundle, None))


class TestSchedules:
    @pytest.mark.property
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_the_fit_does_not_depend_on_the_schedule(
        self, bundle: ModelBundle, reference: list[list[Outcome]], schedule: Schedule
    ) -> None:
        results = _world(schedule).run(_fit_program(bundle, None))
        _assert_bit_identical(results)
        _assert_bit_identical([reference[0], results[0]])


# --------------------------------------------------------------------------- #
# mutations: what the band rules out
# --------------------------------------------------------------------------- #


#: The uneven split: the four rows as one minibatch over three replicas —
#: slices of two, one and one row, shares 1/2, 1/4, 1/4.
UNEVEN = ParallelGeometry(data=3, data_mode="rows")


@pytest.mark.unit
class TestMutations:
    def test_an_uneven_split_is_within_the_band(self, bundle: ModelBundle) -> None:
        """The shares are what make an uneven split right: two, one and one
        row over three replicas, weighed ``1/2, 1/4, 1/4``, is the unsplit
        four-row minibatch within the band."""
        results = _world(0, UNEVEN.data).run(
            _fit_program(bundle, None, geometry=UNEVEN, pairs=4, eval_every=None)
        )
        _assert_bit_identical(results)
        _assert_within_band(results[0], _solo(bundle, pairs=4, eval_every=None))

    def test_a_mean_over_the_replicas_rows_lands_outside_the_band(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Weighing the loss by nothing sums the replicas' local means. Over
        two equal halves that is a constant factor on the gradient, which
        AdamW normalises away — the band alone would not catch it (a finding
        of this suite: ``1.5e-7`` from the oracle at ``dp=2``) — so the
        mutation is held to the uneven split, where the one-row replicas'
        means count double and the gradient's direction changes."""
        monkeypatch.setattr(RowSplit, "weigh", lambda self, loss, rows, total: loss)
        results = _world(0, UNEVEN.data).run(
            _fit_program(bundle, None, geometry=UNEVEN, pairs=4, eval_every=None)
        )
        _assert_bit_identical(results)
        oracle = _solo(bundle, pairs=4, eval_every=None)
        distance = max(_distance(a[2], b[2]) for a, b in zip(results[0], oracle))
        assert distance > 1e3 * BAND, distance

    def test_skipping_the_reduce_on_every_replica_lands_outside_the_band(
        self, bundle: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No replica reduces: each fits its own half at half weight, the
        replicas drift apart and neither is the world-1 fit."""
        monkeypatch.setattr(RowSplit, "reduce_gradients", lambda self, parameters: None)
        results = _world(0).run(_fit_program(bundle, None))
        oracle = _solo(bundle)
        apart = max(_distance(a[2], b[2]) for a, b in zip(results[0], results[1]))
        assert apart > 1e4 * BAND, apart
        for rank_outcomes in results:
            distance = max(_distance(a[2], b[2]) for a, b in zip(rank_outcomes, oracle))
            assert distance > 1e4 * BAND, distance
