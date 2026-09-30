"""The routing-mismatch record under a pipeline, in simulation
(``docs/model_parallelism.md`` §6.5, §8.3, §10.4).

An expert-keyed write (spec §2.5 ``expert_neuron``) records per (write,
layer, example) how many written slots found no source slot holding their
expert — inside its hook, which under ``pp > 1`` runs on the owning stage
alone. ``routing_mismatch.json`` is read off the **publisher's** executor,
so a write on stage 1 must share its record with publisher stage 0. The
executor shares it where the fire tally is agreed (``parallel/mismatch.py``),
so every rank's ``routing_mismatch`` —
the publisher's included — equals the world-1 executor's **to the entry**,
under every drawn schedule; with the write on stage 0 nothing changes;
and skipping the share leaves the publisher's record empty while the
owner's matches world 1.
"""

# the stage suite's placement and the featurizer-group suite's document
# pyright: reportPrivateUsage=false
from __future__ import annotations

from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.engines.pytorch_hooks import executor as executor_module
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.neural.shared.parallel.mismatch import RoutingMismatch
from causalab.protocol.parallel import ParallelGeometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, SimulatedWorld, groups_for

from ._drive import executor_for
from .conftest import TINY_QWEN35_MOE
from .test_featurizer_groups import CF_TEXT, TEXT, _expert_dbm_doc
from .test_stages import _staged

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

PP2 = ParallelGeometry(pipeline=2)
#: the four-layer fixture at ``pp=2``: layers 0–1 on stage 0 (the
#: publisher), 2–3 on stage 1
ON_STAGE = {0: 1, 1: 3}


@pytest.fixture(scope="module")
def moe() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


def _document(layer: int) -> dict[str, Any]:
    """The expert-keyed DBM document with every site moved to ``layer``."""
    raw = _expert_dbm_doc()
    for site in raw["method"]["sites"].values():
        site["layers"] = [layer]
    return raw


def _record(raw: dict[str, Any], bundle: ModelBundle, c: Collective) -> RoutingMismatch:
    executor = executor_for(
        raw, bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    )
    executor.fragments = Fragments(c)
    executor.run_all()
    return dict(executor.routing_mismatch)


def _program(
    raw: dict[str, Any], bundle: ModelBundle
) -> Callable[[int, Collective], RoutingMismatch]:
    def program(rank: int, c: Collective) -> RoutingMismatch:
        return _record(raw, _staged(bundle, PP2, rank), c)

    return program


def _world(schedule: Schedule) -> SimulatedWorld:
    return SimulatedWorld(
        groups_for(2, pipeline=2), world=2, schedule=schedule, timeout=600.0
    )


def _oracle(raw: dict[str, Any], bundle: ModelBundle) -> RoutingMismatch:
    world = SimulatedWorld(groups_for(1), world=1, schedule=0, timeout=600.0)
    (record,) = world.run(lambda rank, c: _record(raw, bundle, c))
    return record


@pytest.mark.property
class TestRoutingMismatchAcrossStages:
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_the_write_on_stage_one_records_on_every_stage_what_world_one_records(
        self, moe: ModelBundle, schedule: Schedule
    ) -> None:
        raw = _document(ON_STAGE[1])
        expected = _oracle(raw, moe)
        assert expected and all(key[1] == ON_STAGE[1] for key in expected)
        results = _world(schedule).run(_program(raw, moe))
        assert results == [expected, expected]

    def test_the_write_on_stage_zero_is_unchanged(self, moe: ModelBundle) -> None:
        raw = _document(ON_STAGE[0])
        expected = _oracle(raw, moe)
        assert expected
        assert _world(0).run(_program(raw, moe)) == [expected, expected]

    def test_the_share_is_one_broadcast_per_stage_local_write_per_window(
        self, moe: ModelBundle
    ) -> None:
        """The document's two writes sit at two stage-local addresses, so
        the one window shares twice, from stage 1, beside the tally's sum."""
        world = _world(0)
        world.run(_program(_document(ON_STAGE[1]), moe))
        shares = [
            e
            for e in world.transcript
            if e.op == "broadcast" and "mismatch.py" in e.call_site
        ]
        assert len(shares) == 2 * 2  # two writes, two ranks arriving at each
        assert {e.rank for e in shares} == {0, 1}

    def test_mutation_the_share_skipped_leaves_the_publisher_empty(
        self, moe: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loss the harness pinned: without the share the owner alone
        holds the record and the publisher writes no file."""
        monkeypatch.setattr(
            executor_module,
            "shared_mismatch",
            lambda records, owners, collective, **kw: dict(records),
        )
        raw = _document(ON_STAGE[1])
        expected = _oracle(raw, moe)
        publisher, owner = _world(0).run(_program(raw, moe))
        assert publisher == {} and owner == expected
