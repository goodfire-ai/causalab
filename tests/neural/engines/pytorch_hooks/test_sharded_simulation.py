"""The first simulated scenarios that run ``apply_plan`` for real
(``docs/model_parallelism.md`` §10.4, §10.8): the tiny fixtures sharded
through the registry's plan with the fragment tier over ``SimulatedWorld``,
a forward and a training step's gradients held to the world-1 model.

Every rank deep-copies the one whole model held on CPU, applies the plan
for its geometry with [`FragmentStyles`][causalab.neural.engines.pytorch_hooks.styles.fragment.FragmentStyles] over its simulated collective
— this rank's chunk of every sharded parameter, the styles' collectives in
the forward and the backward — and runs the same input. What is held:

* every rank's logits are **bit-identical** to every other rank's (the
  output is replicated), and every rank's gradients — the embeddings' input
  gradient, every whole (row-less or replicated) parameter's gradient —
  are bit-identical across the ranks; a sharded parameter's gradient is
  its chunk of the world-1 gradient within the band;
* the logits and the gradients equal the world-1 model's **within a band
  measured on the fixtures**. A sharded forward is the one-process
  computation up to the reduction order its collectives introduce — a
  rowwise projection's all-reduce over ``tp`` partial products, the expert
  outputs' all-reduce over the ``ep`` ranks' experts — and the fixed-order
  simulated sum is not the single matmul's order; the colwise, gathered
  and replicated styles are exact. 📐 Measured (2026-09-17, torch 2.9.0,
  CPU, fp32; `MEASURED`) — maximum absolute difference relative to
  the largest world-1 entry; the band is `BAND`, twenty times the
  largest measured maximum, the smoke tiers' rule, four orders below what
  a wrong shard makes (experts left unsharded under a remapping router
  moved the logits by 0.26–0.40);
* the result does not depend on the schedule (the deterministic-
  simulation guarantee, §10.4's last row).

The ``gloo`` half (``smoke``) runs the same programs on real ranks with
**both** tiers — the fragment tier and transformers' DTensor tier — and
holds the three runs bit-identical to each other: the simulator's tier
reproduces the production styles' arithmetic on the real fixtures, not only
on the contract's exact rows.
"""

from __future__ import annotations

import copy
import functools
from typing import Any, Callable, Mapping

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding, apply_plan
from causalab.neural.engines.pytorch_hooks.styles import Styles, partition_of
from causalab.neural.engines.pytorch_hooks.styles.fragment import FragmentStyles
from causalab.neural.shared.parallel.collective import Collective, TorchCollective
from causalab.neural.shared.parallel.placement import AXES
from causalab.protocol.parallel import MeshLayout, ParallelGeometry, format_geometry
from causalab.protocol.registry import ModelInfo, ParallelPlan
from tests._helpers import parallel_strategies as ps
from tests._helpers.gloo_world import GlooWorld
from tests._helpers.simulated_world import Schedule, SimulatedWorld
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

#: The measured maxima the band is pinned against (module docstring),
#: relative to the largest world-1 entry of the same tensor.
MEASURED: dict[str, float] = {
    "llama tp=2 logits": 2.4e-7,
    "llama tp=2 gradients": 7.0e-8,
    "moe ep=2 logits": 3.0e-7,
    "moe ep=2 gradients": 2.6e-7,
    "moe tp=2,ep=2 logits": 3.2e-7,
    "moe tp=2,ep=2 gradients": 3.1e-7,
}
BAND = 1e-5
assert BAND >= 20 * max(MEASURED.values())

IDS = torch.tensor([[5, 17, 23, 42, 8, 91, 3], [12, 4, 77, 61, 30, 2, 9]])
READOUT_SEED = 7

Tier = Callable[[int, Collective, ParallelGeometry], tuple[Sharding, Styles]]


def fragment_tier(
    rank: int, c: Collective, geometry: ParallelGeometry
) -> tuple[Sharding, Styles]:
    return Sharding(geometry, rank, collective=c), FragmentStyles(c)


def transformers_tier(
    rank: int, c: Collective, geometry: ParallelGeometry
) -> tuple[Sharding, Styles]:
    from causalab.neural.engines.pytorch_hooks.styles.dtensor import TransformersStyles

    assert isinstance(c, TorchCollective)
    sharding = Sharding.from_mesh(c.mesh)
    return sharding, TransformersStyles(sharding)


def _readout_loss(logits: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(READOUT_SEED)
    weights = torch.randn(logits.shape[-1], generator=generator)
    return (logits[:, -1, :] * weights).sum()


def _plain(tensor: Any) -> Any:
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor


class Whole:
    """The one model held whole, and what a rank makes of it."""

    def __init__(self, key: str) -> None:
        bundle = load_model(key)
        self.key = key
        self.model = bundle.model
        self.info: ModelInfo = bundle.info
        assert self.info.parallel_plan is not None
        self.plan: ParallelPlan = self.info.parallel_plan
        self.model.eval()

    def run(self, model: Any) -> dict[str, Any]:
        """A forward without grad and a graded forward with the readout
        loss: the logits, the embeddings' input gradient and every
        parameter's gradient."""
        with torch.no_grad():
            logits = model(input_ids=IDS).logits
        embeds = model.get_input_embeddings()(IDS).detach().requires_grad_()
        graded = model(inputs_embeds=embeds).logits
        _readout_loss(graded).backward()
        assert embeds.grad is not None
        return {
            "logits": _plain(logits),
            "graded_logits": _plain(graded),
            "embeds_grad": _plain(embeds.grad),
            "grads": {
                name: _plain(p.grad)
                for name, p in model.named_parameters()
                if p.grad is not None
            },
            "params": {name: _plain(p) for name, p in model.named_parameters()},
        }

    def reference(self) -> dict[str, Any]:
        model = copy.deepcopy(self.model)
        return self.run(model)

    def program(
        self, rank: int, c: Collective, *, tier: Tier, geometry: ParallelGeometry
    ) -> dict[str, Any]:
        """One rank: the whole model copied, the plan for the geometry
        applied with the tier's styles, the run."""
        model = copy.deepcopy(self.model)
        sharding, styles = tier(rank, c, geometry)
        apply_plan(model, self.plan.for_geometry(geometry, self.info), sharding, styles)
        return self.run(model)


@pytest.fixture(scope="module")
def llama() -> Whole:
    return Whole(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe() -> Whole:
    return Whole(TINY_QWEN35_MOE)


def _simulated(geometry: ParallelGeometry, schedule: Schedule = ()) -> SimulatedWorld:
    layout = MeshLayout(geometry)
    return SimulatedWorld(
        {axis: layout.groups(axis) for axis in AXES},
        world=geometry.world,
        schedule=schedule,
        timeout=600.0,
    )


def _relative(got: torch.Tensor, expected: torch.Tensor) -> float:
    scale = float(expected.abs().max())
    return float((got - expected).abs().max()) / scale


def _partition(whole: Whole, geometry: ParallelGeometry, name: str, ndim: int):
    """The chunk a rank holds of parameter ``name`` under ``geometry``."""
    from causalab.neural.engines.pytorch_hooks.styles import WHOLE
    from causalab.protocol.parallel_memory import row_for

    plan = whole.plan.for_geometry(geometry, whole.info)
    row = row_for(plan, name, whole.model.base_model_prefix)
    if row is None or getattr(geometry, row.axis) == 1:
        return WHOLE, None
    return partition_of(row.style, ndim, name.rsplit(".", 1)[-1]), row.axis


def _check(
    whole: Whole,
    geometry: ParallelGeometry,
    results: list[dict[str, Any]],
    reference: dict[str, Any],
    label: str,
    record: dict[str, float],
) -> None:
    layout = MeshLayout(geometry)
    first = results[0]
    for rank, result in enumerate(results):
        # replicated across the ranks, bit for bit
        assert torch.equal(result["logits"], first["logits"]), rank
        assert torch.equal(result["graded_logits"], result["logits"]), rank
        assert torch.equal(result["embeds_grad"], first["embeds_grad"]), rank
        worst_logits = _relative(result["logits"], reference["logits"])
        worst_grads = _relative(result["embeds_grad"], reference["embeds_grad"])
        for name, grad in result["grads"].items():
            expected_grad = reference["grads"][name]
            partition, axis = _partition(whole, geometry, name, expected_grad.dim())
            expected_param = reference["params"][name]
            if axis is not None:
                local = layout.rank_in(rank, axis)
                size = getattr(geometry, axis)
                expected_grad = partition.local(expected_grad, local, size)
                expected_param = partition.local(expected_param, local, size)
            else:
                assert torch.equal(grad, first["grads"][name]), (rank, name)
            assert torch.equal(result["params"][name], expected_param), (rank, name)
            assert grad.shape == expected_grad.shape, (rank, name)
            if float(expected_grad.abs().max()) > 0:
                worst_grads = max(worst_grads, _relative(grad, expected_grad))
        record[f"{label} logits"] = max(
            record.get(f"{label} logits", 0.0), worst_logits
        )
        record[f"{label} gradients"] = max(
            record.get(f"{label} gradients", 0.0), worst_grads
        )
        assert worst_logits <= BAND, (rank, worst_logits)
        assert worst_grads <= BAND, (rank, worst_grads)


_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

RECORD: dict[str, float] = {}

CASES: tuple[tuple[str, ParallelGeometry], ...] = (
    ("llama", ParallelGeometry(tensor=2)),
    ("moe", ParallelGeometry(expert=2)),
    ("moe", ParallelGeometry(tensor=2, expert=2)),
)
_ids: Callable[[Any], str] = lambda x: (  # noqa: E731 - pytest id helper
    format_geometry(x) if isinstance(x, ParallelGeometry) else str(x)
)


def _whole(request: pytest.FixtureRequest, which: str) -> Whole:
    return request.getfixturevalue(which)


@pytest.mark.unit
class TestShardedForwardAndGradients:
    @pytest.mark.parametrize(("which", "geometry"), CASES, ids=_ids)
    def test_every_rank_equals_world_one_within_the_band(
        self, which: str, geometry: ParallelGeometry, request: pytest.FixtureRequest
    ) -> None:
        whole = _whole(request, which)
        results = _simulated(geometry).run(
            functools.partial(whole.program, tier=fragment_tier, geometry=geometry)
        )
        _check(
            whole,
            geometry,
            results,
            whole.reference(),
            f"{which} {_spell(geometry)}",
            RECORD,
        )

    def test_the_measured_maxima_hold_the_record(self) -> None:
        """Runs after the cases: every maximum measured in this session
        sits within an order of magnitude of the recorded one the band is
        pinned against — another platform's BLAS rounds differently, a
        wrong shard moves the logits by four orders."""
        assert RECORD, "the cases ran first"
        for label, worst in RECORD.items():
            assert worst <= 10 * MEASURED[label], (
                label,
                worst,
                MEASURED[label],
                RECORD,
            )


def _spell(geometry: ParallelGeometry) -> str:
    return ",".join(
        item for item in format_geometry(geometry).split(",") if not item.endswith("=1")
    )


@pytest.mark.property
class TestScheduleIndependence:
    @_SETTINGS
    @given(schedule=ps.schedules())
    @example(schedule=[])
    def test_the_sharded_run_does_not_depend_on_the_schedule(
        self, schedule: Schedule, llama: Whole
    ) -> None:
        geometry = ParallelGeometry(tensor=2)
        program = functools.partial(
            llama.program, tier=fragment_tier, geometry=geometry
        )
        baseline = _simulated(geometry, 0).run(program)
        scheduled = _simulated(geometry, schedule).run(program)
        for a, b in zip(baseline, scheduled):
            assert torch.equal(a["logits"], b["logits"])
            assert torch.equal(a["embeds_grad"], b["embeds_grad"])
            for name in a["grads"]:
                assert torch.equal(a["grads"][name], b["grads"][name]), name


# --------------------------------------------------------------------------- #
# smoke: both tiers on real ranks, held to the simulator bit for bit
# --------------------------------------------------------------------------- #


def _gloo_program(
    rank: int, c: Collective, *, key: str, tier: Tier, geometry: ParallelGeometry
) -> dict[str, Any]:
    """The whole model loaded in the spawned rank, then `Whole.program`."""
    return Whole(key).program(rank, c, tier=tier, geometry=geometry)


def _equal_results(a: Mapping[str, Any], b: Mapping[str, Any], where: str) -> None:
    for name in ("logits", "graded_logits", "embeds_grad"):
        assert torch.equal(a[name], b[name]), (where, name)
    assert set(a["grads"]) == set(b["grads"]), where
    for name in a["grads"]:
        assert torch.equal(a["grads"][name], b["grads"][name]), (where, name)
        assert torch.equal(a["params"][name], b["params"][name]), (where, name)


@pytest.mark.smoke
class TestBothTiersUnderGloo:
    @pytest.mark.parametrize(("which", "geometry"), CASES[:2], ids=_ids)
    def test_the_simulator_and_both_gloo_tiers_agree_bit_for_bit(
        self, which: str, geometry: ParallelGeometry, request: pytest.FixtureRequest
    ) -> None:
        whole = _whole(request, which)
        simulated = _simulated(geometry).run(
            functools.partial(whole.program, tier=fragment_tier, geometry=geometry)
        )
        fragment = GlooWorld(geometry).run(
            functools.partial(
                _gloo_program, key=whole.key, tier=fragment_tier, geometry=geometry
            )
        )
        dtensor = GlooWorld(geometry).run(
            functools.partial(
                _gloo_program, key=whole.key, tier=transformers_tier, geometry=geometry
            )
        )
        for rank in range(geometry.world):
            _equal_results(
                fragment[rank], simulated[rank], f"fragment/gloo rank {rank}"
            )
            _equal_results(dtensor[rank], simulated[rank], f"dtensor/gloo rank {rank}")
