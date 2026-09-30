"""A fit across pipeline stages under simulation (``docs/model_parallelism.md``
§6.5, §7, §8.3, §10.4).

The **real** ``run_cohort_training`` runs on every simulated rank over the
tiny Llama DAS documents at ``pp=2`` and the tiny MoE's at ``pp=4``, each
rank holding its placed stage (the real ``place_stage`` over a copy), the
collective simulated. The head is on the last stage; the write — where the
featurizer's gradient is made — sits on stage 0, on the last stage, or on a
middle one. Pipeline parallelism is placement, so the fit is held to the
world-1 fit **bit for bit**: the forward moves no reduction, and the
gradient crosses a stage boundary as one tensor (``parallel/autograd.py``:
the sent residual receives its gradient back, the received residual sends
its gradient down, the broadcast logits are attached to the sent residual
so a stage's ``loss.backward()`` reaches the boundary).

Only the stage that owns the write computes the featurizer's gradient. Its
parameters live on every rank, so after each optimizer step the owner's
copy is broadcast over the pipeline axis (``agreements.sync_parameters``)
and every rank — the publisher, stage 0, included — holds the trained
parameters, **identical to the bit**. A fit whose trained featurizers sit
on two different stages is refused by name in this cut.

The collective sequence is the same on every rank — forward ``recv``,
``send``, the logits broadcast, the capture broadcasts, the fire sum;
backward the gradient's ``recv`` and ``send``; then the sync's broadcasts —
proven by the simulator under drawn schedules (a mismatch is a typed
refusal, never a hang of the test process), and the sends and receives pair
up rank by rank. A forward resumed above a stage runs nothing on that
stage and sends nothing, and that stage's backward has nothing to do. The
``RowBudget`` agrees its probe and its out-of-memory retry over the
pipeline axis too, so a window is the same window on every stage.

**Cohorts** (spec §4 "Cohorts"; §6.5, §8.3). Several points fitted
together — the six-step workflow's DAS step, one member per layer — step
as one forward over the members' rows, and under ``pp`` the members'
layers sit on different stages. That is legal **per member**: each
member's write fires on its owner stage alone and its count is summed
over the pipeline before the declaration is checked (the cohort path
agrees fire counts as the solo path does, ``agreements.summed_fires``),
and each member's owner syncs that member's parameters after the step. A
two-member cohort with a member on each stage at ``pp=2``, and members on
stages 1 and 3 at ``pp=4``, land bit-identical to the world-1 cohort. What
the design refuses stays refused: one member whose *own* featurizers sit
on two stages, by name on every rank, before any forward.

**Mutations.** A gradient dropped at the boundary — received, then zeroed
— leaves the stage-0 featurizer at its seed: the fit does not move, and
lands away from world 1. The sync dropped leaves the publisher's copy at
its seed while the owner's is the trained one. A rank that skips the sync
is refused by the simulator naming the sync's broadcast. The cohort path
checking each member's *local* tally is refused (``fired 0 times``), on
every rank.
"""

# the cohort suite's builders and the stage suite's placement are reused here
# pyright: reportPrivateUsage=false

from __future__ import annotations

import contextlib
from typing import Any, Callable, Sequence

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks import cohort as cohort_module
from causalab.neural.engines.pytorch_hooks import stages as stages_module
from causalab.neural.engines.pytorch_hooks.budget import LOCKSTEP_AXES, RowBudget
from causalab.neural.engines.pytorch_hooks.executor import document_seed
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.rows import RowSplit
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training
from causalab.neural.shared.parallel.autograd import Handoff, HandoffMismatch
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    RankFailed,
    Refusal,
    Schedule,
    SimulatedMeter,
    SimulatedWorld,
    groups_for,
)

from ._drive import executor_for
from .conftest import TINY_LLAMA, TINY_QWEN35_MOE
from .test_fit_cohort import PAIRS, _campaign, _request, _train_doc, _weights
from .test_stages import _count_block_fires, _staged
from .test_train import ANSWERS, BASES, COUNTERFACTUALS
from .test_train_parallel import READINGS, _bound

# the §7 gradient agreement check on at bit identity (conftest.py): the
# real ``run_cohort_training`` reads the variable once per fit
pytestmark = [pytest.mark.usefixtures("checked_gradients_simulated")]

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: ``PAIRS`` rows over the four training rows: two minibatches an epoch,
#: twelve updates
EPOCHS = 6
UPDATES = 12
#: the subspace width: the Llama's residual is 16 wide, the MoE's 8 — a
#: ``k`` of the whole residual would make the fit a walk on rounding
K = {TINY_LLAMA: 4, TINY_QWEN35_MOE: 4}
PP2 = ParallelGeometry(pipeline=2)
PP4 = ParallelGeometry(pipeline=4)


@pytest.fixture(scope="module")
def llama() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


Eval = tuple[int, str, dict[str, float]] | None
#: ``(fit_rows, fit_rows_shrinks, weights, eval, resumed)``
Outcome = tuple[int | None, int, dict[str, torch.Tensor], Eval, tuple[int, ...]]


def _outcome(outcome: Any) -> Outcome:
    score = outcome.eval_score
    return (
        outcome.fit_rows,
        outcome.fit_rows_shrinks,
        _weights(outcome),
        (score.passes, score.selected, dict(score.metrics)) if score else None,
        tuple(outcome.resumed),
    )


def _documents(
    key: str,
    *,
    layer: int,
    epochs: int = EPOCHS,
    early_stop: dict[str, Any] | None = None,
    eval_every: int | None = 1,
    two_stages: bool = False,
    cohort: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """One DAS document, the write at ``layer`` of the ``key`` fixture;
    ``cohort`` adds one member per further layer — a campaign of points
    fitted together (spec §4 "Cohorts"), each member's featurizer at its own
    layer; ``two_stages`` gives the first member a second trained featurizer
    whose site is the block above — on the other stage at ``pp=2``."""

    def document(at: int) -> dict[str, Any]:
        raw = _train_doc(
            k=K[key],
            epochs=epochs,
            layer=at,
            early_stop=early_stop,
            eval_every=eval_every,
        )
        raw["model"]["key"] = key
        return raw

    raws = [document(layer), *(document(at) for at in cohort)]
    if two_stages:
        method = raws[0]["method"]
        method["sites"]["tgt2"] = {"component": "block_output", "layers": [layer + 1]}
        method["featurizers"]["rot2"] = {
            "kind": "subspace",
            "k": K[key],
            "parametrization": "cayley",
        }
        method["reads"]["probe"] = {
            "site": "tgt2",
            "pos": {"index": -1},
            "featurizer": "rot2",
        }
        # the probe is read un-intervened on base: a declared model with no
        # writes (§2.9), beside the counterfactual one the document has
        method["intervened_models"]["original_base"] = {
            "input": "base",
            "reads": ["probe"],
        }
        method["train"]["params"] = ["rot", "rot2"]
        # every fit is saved and every read is live (§2.12, §5.11)
        method["save"].append(
            {"value": "rot2", "site": "tgt2", "file_path": "rot2.safetensors"}
        )
        method["save"].append(
            {
                "read": "probe",
                "model": "original_base",
                "file_path": "probe.safetensors",
            }
        )
    return raws


def _fit_program(
    bundle: ModelBundle,
    key: str,
    geometry: ParallelGeometry,
    *,
    layer: int,
    meter: SimulatedMeter | None = None,
    block_fires: dict[int, list[list[int]]] | None = None,
    **document: Any,
) -> Callable[[int, Collective], list[Outcome] | str]:
    """The rank program: the real cohort fit over the DAS document on this
    rank's placed stage, the executors' fragments bound to this rank's
    collective. A refusal is returned as its message so the caller can
    assert every rank made it."""

    def program(rank: int, c: Collective) -> list[Outcome] | str:
        placed = _staged(bundle, geometry, rank) if geometry.pipeline > 1 else bundle
        raws = _documents(key, layer=layer, **document)
        _docs, handles = _campaign(raws)
        executors = [
            executor_for(
                raw,
                placed,
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
        with contextlib.ExitStack() as hooks:
            if block_fires is not None:
                block_fires[rank] = _count_block_fires(hooks, placed)
            try:
                outcomes = run_cohort_training(
                    [ex.doc for ex in executors],
                    executors,
                    _request(),
                    meter=meter.for_rank(rank) if meter is not None else None,
                )
            except ProtocolError as error:
                return str(error)
        return [_outcome(o) for o in outcomes]

    return program


def _world(geometry: ParallelGeometry, schedule: Schedule) -> SimulatedWorld:
    return SimulatedWorld(
        groups_for(geometry.world, pipeline=geometry.pipeline),
        world=geometry.world,
        schedule=schedule,
        timeout=600.0,
    )


def _solo(
    bundle: ModelBundle, key: str, *, layer: int, **document: Any
) -> list[Outcome]:
    """The world-1 oracle: the same program on the whole model."""
    world = SimulatedWorld(groups_for(1), world=1, schedule=0, timeout=600.0)
    (result,) = world.run(_fit_program(bundle, key, ONE, layer=layer, **document))
    assert not isinstance(result, str), result
    return result


def _seed_weights(
    bundle: ModelBundle, key: str, *, layer: int
) -> dict[str, torch.Tensor]:
    """The featurizer's parameters before any update: built as
    ``_prepare_fit`` builds them, under the document's seed."""
    (raw,) = _documents(key, layer=layer)
    executor = executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    torch.manual_seed(document_seed(executor.doc))
    return {
        f"rot.{slot}": parameter.detach().clone()
        for slot, parameter in executor.stage("rot").slot_params().items()
    }


def _ranks(results: Sequence[list[Outcome] | str]) -> list[list[Outcome]]:
    for rank, result in enumerate(results):
        assert not isinstance(result, str), f"rank {rank} refused: {result}"
    return [result for result in results if not isinstance(result, str)]


def _distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert set(a) == set(b)
    return max(float((a[name] - b[name]).abs().max()) for name in a)


def _assert_same(a: Outcome, b: Outcome) -> None:
    """Two outcomes equal to the bit: the bound, the shrinks, every weight,
    the eval record and the resume bookkeeping."""
    assert a[0] == b[0] and a[1] == b[1]
    assert a[3] == b[3], "eval passes, selection and scores"
    assert a[4] == b[4], "resumed"
    assert _distance(a[2], b[2]) == 0.0


def _assert_bit_identical(
    results: Sequence[list[Outcome]], oracle: Sequence[Outcome] | None
) -> None:
    """Every rank's outcome equals rank 0's, and rank 0's the world-1 oracle's."""
    for rank_outcomes in results[1:]:
        for a, b in zip(results[0], rank_outcomes, strict=True):
            _assert_same(a, b)
    if oracle is not None:
        for a, b in zip(results[0], oracle, strict=True):
            _assert_same(a, b)


def _sends_and_recvs(world: SimulatedWorld) -> tuple[dict[int, int], dict[int, int]]:
    sends: dict[int, int] = {}
    recvs: dict[int, int] = {}
    for event in world.transcript:
        if event.op == "send":
            sends[event.rank] = sends.get(event.rank, 0) + 1
        elif event.op == "recv":
            recvs[event.rank] = recvs.get(event.rank, 0) + 1
    return sends, recvs


# --------------------------------------------------------------------------- #
# the fits, bit for bit
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestPipelineFit:
    def test_a_write_on_stage_zero_and_the_head_on_stage_one_equal_world_one(
        self, llama: ModelBundle
    ) -> None:
        """The gradient is made on stage 0 and crosses the boundary once per
        step; after the sync every rank holds it. Twelve updates, bit for bit."""
        oracle = _solo(llama, TINY_LLAMA, layer=0)
        world = _world(PP2, 3)
        results = _ranks(world.run(_fit_program(llama, TINY_LLAMA, PP2, layer=0)))
        _assert_bit_identical(results, oracle)
        # the trained parameters moved off the seed
        assert (
            _distance(results[0][0][2], _seed_weights(llama, TINY_LLAMA, layer=0))
            > 1e-3
        )
        sends, recvs = _sends_and_recvs(world)
        # forward sends from stage 0 are received on stage 1; the gradient
        # the other way — and the gradient did cross
        assert sends[0] == recvs[1] and sends[1] == recvs[0] and sends[1] > 0

    def test_a_write_on_the_last_stage_equals_world_one(
        self, llama: ModelBundle
    ) -> None:
        """The gradient is made on the last stage; stage 0 — the publisher
        — computes nothing of it and holds the trained parameters through
        the sync alone. From the second epoch the patched forward resumes at
        block 1 and stage 0 runs nothing for it."""
        oracle = _solo(llama, TINY_LLAMA, layer=1)
        fires: dict[int, list[list[int]]] = {}
        world = _world(PP2, 5)
        results = _ranks(
            world.run(_fit_program(llama, TINY_LLAMA, PP2, layer=1, block_fires=fires))
        )
        _assert_bit_identical(results, oracle)
        resumed = results[0][0][4]
        assert resumed and set(resumed) == {1}
        # every forward reached block 1 (stage 1's); the resumed ones never
        # reached block 0 (stage 0's)
        assert len(fires[1][1]) - len(fires[0][0]) == len(resumed)
        assert fires[0][1] == [] and fires[1][0] == []

    def test_eval_scores_and_the_early_stop_are_identical_on_every_stage(
        self, llama: ModelBundle
    ) -> None:
        """The scores are computed from broadcast captures, identical on
        every rank already; the early stop's snapshot is taken after the
        sync, so the selected parameters are the same tensor everywhere."""
        early_stop = {"on": "ce", "patience": 0, "mode": "min"}
        oracle = _solo(llama, TINY_LLAMA, layer=1, early_stop=early_stop)
        assert oracle[0][3] is not None and oracle[0][3][1] == "early_stop.best"
        results = _ranks(
            _world(PP2, 8).run(
                _fit_program(llama, TINY_LLAMA, PP2, layer=1, early_stop=early_stop)
            )
        )
        _assert_bit_identical(results, oracle)

    @pytest.mark.parametrize("layer", [1, 3])
    def test_four_stages_on_the_moe_equal_world_one(
        self, moe: ModelBundle, layer: int
    ) -> None:
        """One layer a stage: the write on stage 1 (the gradient crosses two
        boundaries down from the head and the stage below runs nothing once
        the forward resumes) and on stage 3 (the head's stage; three stages
        below it run nothing for a resumed forward and their backward has
        nothing to do)."""
        oracle = _solo(moe, TINY_QWEN35_MOE, layer=layer)
        fires: dict[int, list[list[int]]] = {}
        world = _world(PP4, 11)
        results = _ranks(
            world.run(
                _fit_program(moe, TINY_QWEN35_MOE, PP4, layer=layer, block_fires=fires)
            )
        )
        _assert_bit_identical(results, oracle)
        resumed = results[0][0][4]
        assert resumed and set(resumed) == {layer}
        for stage in range(layer):
            # a stage below the resume: its block ran only for the forwards
            # that did not resume
            assert len(fires[3][3]) - len(fires[stage][stage]) == len(resumed)
        sends, recvs = _sends_and_recvs(world)
        # a middle stage receives in the forward (from below) and in the
        # backward (from above); the ends receive one way only: every send
        # has its receive, and the gradient did cross down from the head
        assert sum(sends.values()) == sum(recvs.values())
        assert sends[3] > 0 and recvs[0] > 0


# --------------------------------------------------------------------------- #
# cohorts: members on different stages
# --------------------------------------------------------------------------- #


def _no_sum(tally: Any, agreements: Any, **kwargs: Any) -> Any:
    """The mutation: the member's local tally checked as it is."""
    return tally


class TestPipelineCohort:
    """Several points fitted together, each member's featurizer at one layer
    (module docstring): the six-step workflow's DAS step under ``pp``."""

    @pytest.mark.unit
    def test_members_on_two_stages_fit_together_bit_for_bit(
        self, llama: ModelBundle
    ) -> None:
        """A member at block 0 (stage 0) and one at block 1 (stage 1) step as
        one forward: each member's write fires on its owner alone and its
        count is summed over the pipeline; each owner syncs its member's
        parameters. Twelve updates, both members bit for bit the world-1
        cohort's."""
        oracle = _solo(llama, TINY_LLAMA, layer=0, cohort=(1,))
        assert len(oracle) == 2
        fires: dict[int, list[list[int]]] = {}
        world = _world(PP2, 3)
        results = _ranks(
            world.run(
                _fit_program(
                    llama, TINY_LLAMA, PP2, layer=0, cohort=(1,), block_fires=fires
                )
            )
        )
        _assert_bit_identical(results, oracle)
        for member, at in enumerate((0, 1)):
            assert (
                _distance(
                    results[0][member][2], _seed_weights(llama, TINY_LLAMA, layer=at)
                )
                > 1e-3
            ), f"member {member} moved off its seed"
        # the members stepped as one forward: each stage's block saw the
        # cohort's rows (both minibatches), not one member's
        assert 2 * PAIRS in fires[0][0] and 2 * PAIRS in fires[1][1]
        sends, recvs = _sends_and_recvs(world)
        assert sends[0] == recvs[1] and sends[1] == recvs[0] and sends[1] > 0

    @pytest.mark.unit
    def test_members_on_stages_one_and_three_of_four_equal_world_one(
        self, moe: ModelBundle
    ) -> None:
        """One layer a stage on the MoE, members at blocks 1 and 3: the
        cohort forward resumes at the shallowest member's block from the
        second epoch, stage 0 runs nothing for it, and the gradient of the
        stage-1 member crosses two boundaries down from the head."""
        oracle = _solo(moe, TINY_QWEN35_MOE, layer=1, cohort=(3,))
        fires: dict[int, list[list[int]]] = {}
        world = _world(PP4, 11)
        results = _ranks(
            world.run(
                _fit_program(
                    moe, TINY_QWEN35_MOE, PP4, layer=1, cohort=(3,), block_fires=fires
                )
            )
        )
        _assert_bit_identical(results, oracle)
        resumed = results[0][0][4]
        assert resumed and set(resumed) == {1}
        # stage 0's block ran only for the forwards that did not resume
        assert len(fires[3][3]) - len(fires[0][0]) == len(resumed)
        sends, recvs = _sends_and_recvs(world)
        assert sum(sends.values()) == sum(recvs.values())
        assert sends[3] > 0 and recvs[0] > 0

    @pytest.mark.property
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_the_cohort_sequence_is_the_same_on_every_stage(
        self, llama: ModelBundle, schedule: Schedule
    ) -> None:
        """Two updates under a drawn schedule: the world completes, the
        sends and receives pair up, both members the same tensors on both
        stages."""
        world = _world(PP2, schedule)
        results = _ranks(
            world.run(
                _fit_program(llama, TINY_LLAMA, PP2, layer=0, cohort=(1,), epochs=1)
            )
        )
        _assert_bit_identical(results, None)
        assert len(results[0]) == 2
        sends, recvs = _sends_and_recvs(world)
        assert sends[0] == recvs[1] and sends[1] == recvs[0] and sends[1] > 0

    @pytest.mark.unit
    def test_a_member_straddling_two_stages_is_refused_by_name_on_every_rank(
        self, llama: ModelBundle
    ) -> None:
        """One member trains ``rot`` at block 0 and ``rot2`` at block 1 — a
        single point's featurizers on two stages, the one thing §8.3
        refuses — beside a member at block 1. Refused on every rank naming
        ``--parallel.pipeline`` and that member's featurizers, before any
        forward; the world completes (a hang would be a typed refusal). The
        same cohort fits at world 1."""
        for message in _world(PP2, 1).run(
            _fit_program(llama, TINY_LLAMA, PP2, layer=0, cohort=(1,), two_stages=True)
        ):
            assert isinstance(message, str), message
            assert "--parallel.pipeline" in message
            assert "'rot'" in message and "'rot2'" in message
            assert "stage 0" in message and "stage 1" in message
        oracle = _solo(
            llama, TINY_LLAMA, layer=0, cohort=(1,), epochs=1, two_stages=True
        )
        assert [set(n.split(".")[0] for n in o[2]) for o in oracle] == [
            {"rot", "rot2"},
            {"rot"},
        ]

    @pytest.mark.unit
    def test_fires_not_summed_over_the_pipeline_is_the_old_refusal(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the pipeline sum, each member checks its local tally.
        A module on another stage has a local count of zero, so each rank
        refuses the other stage's write and the cohort never fits."""
        monkeypatch.setattr(cohort_module, "summed_fires", _no_sum)
        for message in _world(PP2, 2).run(
            _fit_program(llama, TINY_LLAMA, PP2, layer=0, cohort=(1,))
        ):
            assert isinstance(message, str), message
            assert "'patch'" in message and "fired 0 times" in message


# --------------------------------------------------------------------------- #
# the collective sequence
# --------------------------------------------------------------------------- #


class TestSequence:
    @pytest.mark.property
    @given(schedule=ps.schedules(), layer=st.sampled_from((0, 1)))
    @example(schedule=[], layer=0)
    @_SETTINGS
    def test_the_sequence_is_the_same_on_every_stage_under_every_schedule(
        self, llama: ModelBundle, schedule: Schedule, layer: int
    ) -> None:
        """Two updates under a drawn schedule with the write on a drawn
        stage: the world completes (a mismatch is a typed refusal), the sends
        and receives pair up, and the fit is the same tensor on both stages."""
        world = _world(PP2, schedule)
        results = _ranks(
            world.run(_fit_program(llama, TINY_LLAMA, PP2, layer=layer, epochs=1))
        )
        _assert_bit_identical(results, None)
        sends, recvs = _sends_and_recvs(world)
        assert sends[0] == recvs[1] and sends[1] == recvs[0] and sends[1] > 0

    @pytest.mark.unit
    def test_a_rank_skipping_the_parameter_sync_is_refused(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stage 1 skips the sync: stage 0 waits in the sync's broadcast
        while stage 1 has moved on to the next forward's receive — the
        simulator refuses, naming the sync's call site, never hangs."""
        real = stages_module.sync_parameters

        def skipping(
            tensors: Any, owner: int, collective: Collective, **kw: Any
        ) -> None:
            if collective.rank("pipeline") == 1:
                return  # §3's forbidden branch
            real(tensors, owner, collective, **kw)

        monkeypatch.setattr(stages_module, "sync_parameters", skipping)
        with pytest.raises(Refusal) as err:
            _world(PP2, 0).run(_fit_program(llama, TINY_LLAMA, PP2, layer=0))
        assert "sync_parameters" in str(err.value) or "agreements.py" in str(err.value)

    @pytest.mark.unit
    def test_featurizers_trained_on_two_stages_are_refused_by_name(
        self, llama: ModelBundle
    ) -> None:
        """``rot`` at block 0 and ``rot2`` at block 1 trained together: under
        ``pp=2`` their gradients are made on two stages, and this cut syncs
        from one owner — refused on every rank naming ``--parallel.pipeline``
        and both featurizers, before any forward. The same document fits at
        world 1."""
        for message in _world(PP2, 1).run(
            _fit_program(llama, TINY_LLAMA, PP2, layer=0, two_stages=True)
        ):
            assert isinstance(message, str), message
            assert "--parallel.pipeline" in message
            assert "'rot'" in message and "'rot2'" in message
            assert "stage 0" in message and "stage 1" in message
        oracle = _solo(llama, TINY_LLAMA, layer=0, epochs=1, two_stages=True)
        assert {name.split(".")[0] for name in oracle[0][2]} == {"rot", "rot2"}


# --------------------------------------------------------------------------- #
# the budget over the pipeline axis
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestBudgetAgreement:
    def test_the_budget_agrees_over_every_lockstep_axis(self) -> None:
        assert LOCKSTEP_AXES == ("model", "pipeline", "context")
        assert RowBudget.axes == LOCKSTEP_AXES

    def test_readings_that_differ_by_stage_give_one_bound(self) -> None:
        """§3's first agreement over the pipeline axis: every stage runs the
        same windows, so the probe's bound is the minimum over the stages."""
        meter = SimulatedMeter(READINGS)

        def program(rank: int, c: Collective) -> int | None:
            budget = RowBudget.of(None, meter.for_rank(rank), collective=c)
            budget.run(PAIRS, lambda: None, unit=PAIRS)
            return budget.bound

        bounds = _world(PP2, 4).run(program)
        expected = min(_bound(READINGS[0][0]), _bound(READINGS[1][0]))
        assert bounds == [expected, expected]
        assert _bound(READINGS[0][0]) != _bound(READINGS[1][0]), "the readings differ"


# --------------------------------------------------------------------------- #
# mutations
# --------------------------------------------------------------------------- #


class _Zeroing(torch.autograd.Function):
    """The identity whose backward drops the gradient: wrapped around the
    residual a stage sends, the gradient the stage receives back is thrown
    away before it reaches the blocks and the featurizer."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(grad)


@pytest.mark.unit
class TestMutations:
    def test_a_gradient_dropped_at_the_boundary_leaves_the_fit_at_its_seed(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gradient is still received — the sequence is intact — and
        zeroed: the stage-0 featurizer's gradient is zero, so twelve AdamW
        updates move nothing, on every rank alike, away from world 1."""
        real = stages_module.send_with_grad

        def dropping(
            tensor: torch.Tensor, dst: int, axis: str, c: Collective, **kwargs: Any
        ) -> torch.Tensor:
            return real(_Zeroing.apply(tensor), dst, axis, c, **kwargs)

        monkeypatch.setattr(stages_module, "send_with_grad", dropping)
        oracle = _solo(llama, TINY_LLAMA, layer=0)
        seed_only = _seed_weights(llama, TINY_LLAMA, layer=0)
        results = _ranks(
            _world(PP2, 2).run(_fit_program(llama, TINY_LLAMA, PP2, layer=0))
        )
        _assert_bit_identical(results, None)
        assert _distance(results[0][0][2], seed_only) == 0.0, "the fit did not move"
        assert _distance(results[0][0][2], oracle[0][2]) > 1e-3

    def test_a_stage_receiving_on_the_deltanets_protocol_is_refused_on_both_stages(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The protocol mismatch that deadlocked (``autograd.Handoff``,
        §6.5): stage 1 receiving the residual under ``IF_REQUIRED`` on a
        link carrying no gradient would send nothing in backward while stage
        0 — ``ALWAYS`` — waits for it. Both stages now refuse by name at the
        first graded forward, through the real stage forward, and the
        simulator reports a rank failure, never a hang."""
        real = stages_module.recv_with_grad

        def if_required(
            link: torch.Tensor | None,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: torch.device,
            src: int,
            axis: str,
            c: Collective,
            **kwargs: Any,
        ) -> torch.Tensor:
            kwargs["handoff"] = Handoff.IF_REQUIRED
            return real(
                torch.zeros(0, device=device),
                shape,
                dtype,
                device,
                src,
                axis,
                c,
                **kwargs,
            )

        monkeypatch.setattr(stages_module, "recv_with_grad", if_required)
        with pytest.raises(RankFailed) as err:
            _world(PP2, 5).run(_fit_program(llama, TINY_LLAMA, PP2, layer=0))
        cause = err.value.__cause__
        assert isinstance(cause, HandoffMismatch)
        assert "pipeline handoff stage 0 → stage 1" in str(cause)
        assert "ALWAYS" in str(cause) and "IF_REQUIRED" in str(cause)

    def test_the_sync_dropped_leaves_the_publisher_stale(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No rank syncs (the sequence stays aligned): with the write on the
        last stage, stage 0 — the publisher — never receives a gradient and
        stays at its seed, while the owner's copy is the world-1 fit."""
        monkeypatch.setattr(
            stages_module, "sync_parameters", lambda *args, **kwargs: None
        )
        oracle = _solo(llama, TINY_LLAMA, layer=1)
        seed_only = _seed_weights(llama, TINY_LLAMA, layer=1)
        results = _ranks(
            _world(PP2, 6).run(_fit_program(llama, TINY_LLAMA, PP2, layer=1))
        )
        publisher, owner = results[0][0][2], results[1][0][2]
        assert _distance(publisher, seed_only) == 0.0, "the publisher is stale"
        assert _distance(owner, oracle[0][2]) == 0.0, "the owner has the fit"
        assert _distance(publisher, owner) > 1e-3
