"""The reuse pin (``docs/model_parallelism.md`` §3, §10.6): every rank of a
world makes the world-1 reuse decisions and performs the world-1 number of
model forwards for the points it runs.

Forward interning (§3) and prefix resume (§4 "Resume") decide from
torch-free digests and plan-derived owed counts — nothing a collective
returns enters the decision — so the executor's record of them
([`decisions`][causalab.neural.shared.executor.cache.ForwardCache.decisions], one
[`Reuse`][causalab.neural.shared.executor.cache.Reuse] per decision) must be
one program of the document and the point set: identical on every rank of
a geometry, and identical to the world-1 run over the **same points**. The
forward count follows: a rank runs a group when the store has no capture
for it, and only then.

**The rule per axis.** Under ``tp``, ``ep``, ``pp`` and ``cp`` every rank
runs every point, so its record and count equal the full world-1 run's —
under a pipeline each stage runs its part of every forward and publishes
once per group, so a stage's count is world 1's, not a share of it; a
prefix a stage does not hold is recorded by key (``_ABSENT``), so the
resume depth agrees on both stages. Under ``dp`` over points a replica
runs its contiguous shard (``publish.point_shard``), and interning is a
property of the point *set* (a harvest two points share is run once per
replica that holds one of them), so the oracle is the world-1 run **over
that shard** — a replica's record and count equal a world-1 run of the
same points, never the whole campaign's.

The real engine — [`execute_request`][causalab.neural.shared.execution.execute_request]
over the production executor, the shared [`ForwardCache`][causalab.neural.shared.executor.cache.ForwardCache] in view
through the factory seam — under the simulator, on this rank's copy of the
fixture: sharded with the fragment tier (§10.8) for ``tp`` / ``ep`` on the
tiny MoE, placed by stage for ``pp`` and whole for ``cp`` / ``dp`` on the
tiny Llama. Ground truth for a forward is a pre-hook on the rank's base
model, as ``test_forward_interning.py`` counts it; the engine's own tally
(``RunResult.forwards``) is held to it on every rank. The documents are
drawn: corpus 02's swap swept over the target layers (the points; two
points share every ``original`` group and resume from one prefix), reads
of the un-intervened residual at drawn layers (a shared prefix, served
from the first point on), reads of the patched residual at drawn layers
(distinct per point, run by each).

The mutation: a rank whose executor skips the intern check runs what the
store would have served — its count exceeds world 1's and its record says
``ran`` where the oracle says ``served``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.sharding import (
    Sharding,
    apply_plan,
    place_stage,
)
from causalab.neural.engines.pytorch_hooks.styles.fragment import FragmentStyles
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.execution import execute_request
from causalab.neural.shared.executor import ForwardCache, Interning
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.objects import gather_objects
from causalab.neural.shared.parallel.placement import AXES
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import RunContext
from causalab.protocol.lowering import point_count
from causalab.protocol.parallel import ONE, MeshLayout, ParallelGeometry, stage_layers
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.positions.resolve import StepResolution
from causalab.protocol.publish import point_shard
from causalab.protocol.registry import family_for, walk
from causalab.protocol.schema import Document
from causalab.io.env import ResolutionEnv
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, SimulatedWorld
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE
from tests.protocol._env import CORPUS_DIR, FIXTURES, build_env

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

INTERCHANGE = CORPUS_DIR / "02_interchange_im.json"
CPU = torch.device("cpu")

#: The geometries under test and the fixture each runs on: the model axes on
#: the tiny MoE (the fixture with routed experts), the pipeline, context and
#: data axes on the tiny Llama.
CASES: dict[str, tuple[str, ParallelGeometry]] = {
    "tp=2": (TINY_QWEN35_MOE, ParallelGeometry(tensor=2)),
    "ep=2": (TINY_QWEN35_MOE, ParallelGeometry(expert=2)),
    "pp=2": (TINY_LLAMA, ParallelGeometry(pipeline=2)),
    # the pipeline once more on the four-layer fixture: a prefix resume under
    # a stage that does not hold the block (the two-layer Llama never
    # resumes: its first point's write is at block 0 and nothing stores a
    # deeper un-intervened residual before the second point)
    "pp=2:moe": (TINY_QWEN35_MOE, ParallelGeometry(pipeline=2)),
    "cp=2": (TINY_LLAMA, ParallelGeometry(context=2)),
    "dp=2": (TINY_LLAMA, ParallelGeometry(data=2)),
}
LAYERS: dict[str, tuple[int, ...]] = {TINY_LLAMA: (0, 1), TINY_QWEN35_MOE: (0, 1, 2, 3)}


# --------------------------------------------------------------------------- #
# the documents
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Variant:
    """One drawn document: the target layers the swap is swept over (one
    point each), the layers of the un-intervened residual read on
    ``original`` / ``base`` (the same group in every point: served after
    the first), and the layers of the patched residual read on ``patched``
    / ``base`` (distinct per point: run by each)."""

    sweep: tuple[int, ...]
    shared: tuple[int, ...]
    distinct: tuple[int, ...]

    @property
    def points(self) -> int:
        return len(self.sweep)


def variants(layers: Sequence[int], min_points: int = 1) -> st.SearchStrategy[Variant]:
    layer = st.sampled_from(list(layers))
    some = st.lists(layer, max_size=2, unique=True).map(tuple)
    return st.builds(
        Variant,
        sweep=st.lists(
            layer, min_size=min_points, max_size=len(layers), unique=True
        ).map(tuple),
        shared=some,
        distinct=some,
    )


def author(tmp: Path, key: str, variant: Variant) -> Path:
    """Corpus 02 on ``key`` in fp32, the swap swept over ``variant.sweep``,
    the residual reads added and saved."""
    doc = json.loads(INTERCHANGE.read_text())
    doc["model"] = {"key": key, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    sites, reads, save = method["sites"], method["reads"], method["save"]
    models = method["intervened_models"]
    sites["target"]["layers"] = (
        {"sweep": list(variant.sweep)} if variant.points > 1 else [variant.sweep[0]]
    )
    for model, layers in (("original", variant.shared), ("patched", variant.distinct)):
        if not layers:
            # a declared model must list a read (§2.9)
            continue
        # the un-intervened model on base is a declared model with no writes
        # (§2.9): `original_base`, beside the corpus's `original_counterfactual`
        owner = "original_base" if model == "original" else model
        entry = models.setdefault(owner, {"input": "base", "reads": []})
        for layer in layers:
            name = f"{model}_l{layer}"
            sites[name] = {"component": "block_output", "layers": [layer]}
            reads[f"r_{name}"] = {"site": name, "pos": -1}
            entry["reads"].append(f"r_{name}")
            save.append(
                {
                    "read": f"r_{name}",
                    "model": owner,
                    "file_path": f"r_{name}.safetensors",
                }
            )
    target = tmp / "reuse.json"
    target.write_text(json.dumps(doc, indent=2))
    register_model_key(doc)
    return target


def environment(tmp: Path) -> ResolutionEnv:
    root = tmp / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return build_env(root)


def context(
    env: ResolutionEnv,
    out: Path,
    *,
    points: range | None = None,
    publisher: Any = None,
) -> RunContext:
    """The run context ``run_protocol`` builds: ``points`` is the whole
    selection on every rank (``None`` is every point) and the engine derives
    this replica's shard from ``publisher`` (``publish.point_shard``); a
    world-1 run of exactly one shard's points names them as ``points``."""
    return RunContext(
        output_dir=out,
        env=env,
        points=None if points is None else tuple(points),
        **({"publisher": publisher} if publisher is not None else {}),
    )


# --------------------------------------------------------------------------- #
# the rank's model and the run
# --------------------------------------------------------------------------- #


def placed(
    bundle: ModelBundle, geometry: ParallelGeometry, rank: int, c: Collective
) -> ModelBundle:
    """This rank's copy of ``bundle`` under ``geometry``: the plan applied
    with the fragment tier over ``c`` on the model axes (§10.8), the stage
    placed on the pipeline axis, the whole model otherwise."""
    model = copy.deepcopy(bundle.model)
    if geometry.tensor > 1 or geometry.expert > 1:
        assert bundle.info.parallel_plan is not None
        plan = bundle.info.parallel_plan.for_geometry(geometry, bundle.info)
        apply_plan(
            model, plan, Sharding(geometry, rank, collective=c), FragmentStyles(c)
        )
    if geometry.pipeline > 1:
        stage = MeshLayout(geometry).rank_in(rank, "pipeline")
        pipeline = ParallelGeometry(pipeline=geometry.pipeline)
        place_stage(model, Sharding(geometry=pipeline, rank=stage, meshes={}))
    adapter = family_for(model)
    devices = DeviceMap.of_modules(
        walk(model, adapter.tree.embedding),
        list(adapter.blocks_of(model)),
        walk(model, adapter.tree.lm_head),
        empty=CPU,
    )
    return dataclasses.replace(bundle, model=model, geometry=geometry, devices=devices)


@dataclasses.dataclass(frozen=True)
class Replica:
    """A data-parallel replica's place in the run, over the simulated
    collective: replica ``rank("data")`` of ``size("data")``, publishing
    (every other axis is one here), the gather over the data group."""

    collective: Collective
    launcher: str = "joined"

    @property
    def replica(self) -> int:
        return self.collective.rank("data")

    @property
    def replicas(self) -> int:
        return self.collective.size("data")

    @property
    def publish(self) -> bool:
        return True

    def gather(self, payload: Any) -> Sequence[Any] | None:
        return gather_objects(payload, "data", self.collective)


@dataclasses.dataclass(frozen=True)
class Trace:
    """What one rank's run of its points cost and decided: the base model's
    forwards (the pre-hook), the engine's own tally, the store's reuse
    record and its resume depths."""

    forwards: int
    reported: int
    decisions: tuple[tuple[str, str], ...]
    resumed: tuple[int, ...]
    #: the blocks this rank holds — every block off a pipeline; a stage's
    #: own under ``pp`` (``protocol.parallel.stage_layers``)
    held: tuple[int, ...] = ()

    @property
    def skipped(self) -> int:
        """The forwards this rank ran no model call for: a resumed forward
        whose start lies past every block the stage holds has nothing for
        the stage to compute (``stages.swaps``), so the stage publishes the
        group and calls no model. Zero off a pipeline."""
        top = max(self.held, default=None)
        return 0 if top is None else sum(1 for start in self.resumed if start > top)


Mutate = Callable[[int, PointExecutor], None]


def trace(
    bundle: ModelBundle,
    compiled: CompiledProtocol,
    env: ResolutionEnv,
    out: Path,
    *,
    geometry: ParallelGeometry = ONE,
    rank: int = 0,
    collective: Collective = SOLO,
    mutate: Mutate | None = None,
    points: range | None = None,
) -> Trace:
    """One rank's run of its shard of the campaign through the real engine,
    the campaign store in view (the ``executor_factory`` seam, as
    ``test_forward_interning`` watches it). ``points`` narrows a world-1
    run to one shard's points (the oracle under ``dp``)."""
    engine = PytorchHooksEngine(
        device="cpu", parallel=geometry, collective=collective, bundle=bundle
    )
    publisher = Replica(collective) if geometry.data > 1 else None
    run = context(env, out, points=points, publisher=publisher)
    stores: list[ForwardCache] = []

    def factory(
        doc: Document,
        r: RunContext,
        coords: Any,
        interning: Interning | None,
        resolution: StepResolution | None,
    ) -> PointExecutor:
        assert interning is not None
        stores.append(interning.cache)
        executor = engine._executor(
            doc,
            r,
            coords=coords,
            interning=interning,
            resolution=resolution,
            collective=collective,
        )
        if mutate is not None:
            mutate(rank, executor)
        return executor

    calls: list[int] = []
    base = getattr(bundle.model, bundle.model.base_model_prefix)
    handle = base.register_forward_pre_hook(lambda _m, _a: calls.append(1))
    try:
        result = execute_request(
            compiled,
            run,
            engine_name=engine.name,
            executor_factory=factory,
            train_runner=None,
            intern_forwards=True,
            engine=engine,
        )
    finally:
        handle.remove()
    assert len({id(s) for s in stores}) == 1, "one campaign, one store"
    store = stores[0]
    num_layers = len(list(family_for(bundle.model).blocks_of(bundle.model)))
    held = (
        tuple(
            stage_layers(
                ParallelGeometry(pipeline=geometry.pipeline),
                MeshLayout(geometry).rank_in(rank, "pipeline"),
                num_layers,
            )
        )
        if geometry.pipeline > 1
        else tuple(range(num_layers))
    )
    return Trace(
        forwards=len(calls),
        reported=result.forwards,
        decisions=tuple((d.kind, d.key) for d in store.decisions),
        resumed=tuple(store.resumed),
        held=held,
    )


def world_program(
    bundle: ModelBundle,
    compiled: CompiledProtocol,
    env: ResolutionEnv,
    out: Path,
    geometry: ParallelGeometry,
    mutate: Mutate | None = None,
) -> Callable[[int, Collective], Trace]:
    def program(rank: int, c: Collective) -> Trace:
        mine = placed(bundle, geometry, rank, c)
        return trace(
            mine,
            compiled,
            env,
            out / f"rank{rank}",
            geometry=geometry,
            rank=rank,
            collective=c,
            mutate=mutate,
        )

    return program


def world(geometry: ParallelGeometry, schedule: Schedule) -> SimulatedWorld:
    layout = MeshLayout(geometry)
    return SimulatedWorld(
        {axis: layout.groups(axis) for axis in AXES},
        world=geometry.world,
        schedule=schedule,
        timeout=600.0,
    )


def oracles(
    bundle: ModelBundle,
    compiled: CompiledProtocol,
    env: ResolutionEnv,
    out: Path,
    geometry: ParallelGeometry,
) -> list[Trace]:
    """The world-1 run every rank is held to: over the whole campaign on
    the model, pipeline and context axes; over the replica's own shard
    under ``dp`` (module docstring) — a world-1 run selecting exactly the
    points one replica holds (``RunContext.points``)."""
    if geometry.data == 1:
        return [trace(bundle, compiled, env, out / "solo")] * geometry.world
    n = point_count(compiled.axes)
    return [
        trace(
            dataclasses.replace(
                bundle, geometry=ParallelGeometry(data=1)
            ),  # world 1 over the shard
            compiled,
            env,
            out / f"solo{replica}",
            points=point_shard(range(n), replica, geometry.data),
        )
        for replica in range(geometry.data)
    ]


@pytest.fixture(scope="module")
def bundles() -> dict[str, ModelBundle]:
    return {key: load_model(key) for key in (TINY_LLAMA, TINY_QWEN35_MOE)}


def prepare(
    tmp: Path, key: str, variant: Variant
) -> tuple[CompiledProtocol, ResolutionEnv]:
    env = environment(tmp)
    compiled = compile_protocol(author(tmp, key, variant), env=env)
    assert point_count(compiled.axes) == variant.points
    return compiled, env


def assert_pinned(
    ranks: Sequence[Trace], expected: Sequence[Trace], *, joined: bool = False
) -> None:
    """Every rank's record, resume depths and forward count are its
    oracle's, and the engine's own tally is the hook's count — except the
    data-parallel joiner's, whose ``RunResult.forwards`` is the campaign's
    total over the replicas (``execution.execute_request`` joins the
    shards' counts as it joins their tables)."""
    for rank, (mine, oracle) in enumerate(zip(ranks, expected)):
        assert mine.decisions == oracle.decisions, (rank, "the reuse record")
        assert mine.resumed == oracle.resumed, (rank, "the resume depths")
        # the groups run — what `RunResult.forwards` counts — are world 1's
        # on every rank (a stage publishes every group it takes part in);
        # the joiner's tally is the campaign's total over the replicas
        groups = sum(1 for kind, _ in mine.decisions if kind == "ran")
        if joined and rank == 0:
            assert mine.reported == sum(r.forwards for r in ranks), "the joined tally"
        else:
            assert mine.reported == groups, (rank, "the engine's tally")
        # the model calls are world 1's, less the resumed forwards that
        # start past every block a stage holds (`Trace.skipped`: zero off a
        # pipeline, so this is equality there)
        assert mine.forwards == oracle.forwards - mine.skipped, (
            rank,
            "the forward count",
        )


# --------------------------------------------------------------------------- #
# the pin
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestReusePin:
    @pytest.mark.parametrize("case", sorted(CASES))
    @given(data=st.data(), schedule=ps.schedules())
    @_SETTINGS
    def test_every_rank_makes_the_world_one_decisions_and_forward_count(
        self,
        case: str,
        data: st.DataObject,
        schedule: Schedule,
        bundles: dict[str, ModelBundle],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        key, geometry = CASES[case]
        variant = data.draw(variants(LAYERS[key], min_points=geometry.data))
        tmp = tmp_path_factory.mktemp("reuse")
        compiled, env = prepare(tmp, key, variant)
        expected = oracles(bundles[key], compiled, env, tmp, geometry)
        ranks = world(geometry, schedule).run(
            world_program(bundles[key], compiled, env, tmp / case, geometry)
        )
        assert_pinned(ranks, expected, joined=geometry.data > 1)
        # the record is not vacuous: the points share the counterfactual
        # group, so from the second point on it is served, and every point
        # runs its own patched group
        if variant.points > 1 and geometry.data == 1:
            kinds = [kind for kind, _ in ranks[0].decisions]
            assert "served" in kinds and kinds.count("ran") >= variant.points

    def test_the_record_and_the_count_are_the_same_on_every_rank(
        self, bundles: dict[str, ModelBundle], tmp_path: Path
    ) -> None:
        """The two ranks of a geometry agree with each other, not only each
        with world 1 — spelled out for the reader on one fixed document."""
        # no patched read below the writes: a tap under a write bounds the
        # plan's `resume_at` at that block, so a document reading the patched
        # residual at block 0 never resumes — the planner's answer, held by
        # the property above like any other
        variant = Variant(sweep=(0, 1, 2, 3), shared=(3,), distinct=())
        for case in ("tp=2", "ep=2", "pp=2:moe"):
            key, geometry = CASES[case]
            compiled, env = prepare(tmp_path / case.replace(":", "-"), key, variant)
            (oracle,) = oracles(bundles[key], compiled, env, tmp_path / case, geometry)[
                :1
            ]
            ranks = world(geometry, 0).run(
                world_program(bundles[key], compiled, env, tmp_path / case, geometry)
            )
            kinds = {kind for kind, _ in ranks[0].decisions}
            assert {"ran", "served", "resumed"} <= kinds, case
            assert ranks[0].decisions == ranks[1].decisions == oracle.decisions, case
            assert ranks[0].resumed == ranks[1].resumed == oracle.resumed, case
            if geometry.pipeline == 1:
                assert ranks[0] == ranks[1], case
                assert ranks[0].forwards == oracle.forwards, case
            else:
                # the four-layer fixture at pp=2: stage 0 holds blocks 0–1,
                # so the forwards resumed at block 2 or 3 skip it — its model
                # calls are world 1's minus those; the last stage's are world 1's
                assert ranks[0].held == (0, 1) and ranks[1].held == (2, 3), case
                assert ranks[0].skipped > 0 and ranks[1].skipped == 0, case
                assert ranks[0].forwards == oracle.forwards - ranks[0].skipped, case
                assert ranks[1].forwards == oracle.forwards, case


# --------------------------------------------------------------------------- #
# the mutation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestMutation:
    """A rank whose executors never consult the store: what world 1 is
    served, that rank runs."""

    @staticmethod
    def _skip_intern(rank: int, executor: PointExecutor) -> None:
        if rank == 1:
            executor._interned = lambda digest, keys: None  # type: ignore[method-assign]

    def test_a_replica_that_skips_the_intern_check_does_more_forwards(
        self, bundles: dict[str, ModelBundle], tmp_path: Path
    ) -> None:
        """Under ``dp=2`` the replicas' forwards share no collective, so the
        mutant runs to the end: its count exceeds its shard's oracle and
        its record says ``ran`` where the oracle says ``served``; replica 0
        still passes the pin."""
        variant = Variant(sweep=(0, 1, 2, 3), shared=(1,), distinct=())
        key, geometry = TINY_QWEN35_MOE, ParallelGeometry(data=2)
        compiled, env = prepare(tmp_path, key, variant)
        expected = oracles(bundles[key], compiled, env, tmp_path, geometry)
        ranks = world(geometry, 0).run(
            world_program(
                bundles[key],
                compiled,
                env,
                tmp_path / "mutant",
                geometry,
                self._skip_intern,
            )
        )
        mutant, oracle = ranks[1], expected[1]
        assert ranks[0].decisions == expected[0].decisions
        assert ranks[0].forwards == expected[0].forwards
        assert mutant.forwards > oracle.forwards
        assert mutant.reported == mutant.forwards  # not the joiner: its own
        assert ranks[0].reported == ranks[0].forwards + mutant.forwards
        assert "served" not in {kind for kind, _ in mutant.decisions}
        assert "served" in {kind for kind, _ in oracle.decisions}
        with pytest.raises(AssertionError, match="the reuse record"):
            assert_pinned(ranks, expected, joined=True)

    def test_under_a_model_axis_the_world_refuses_the_rank_by_name(
        self, bundles: dict[str, ModelBundle], tmp_path: Path
    ) -> None:
        """Under ``tp=2`` the same mutant runs a forward its peer does not,
        so the tensor group's next collectives disagree and the simulator
        refuses the world naming both ranks and call sites (§3 "never
        branch on rank") — a reuse decision that differs across a model
        group is not a slower run, it is a divergence."""
        from tests._helpers.simulated_world import Divergence

        variant = Variant(sweep=(0, 1), shared=(1,), distinct=())
        key, geometry = CASES["tp=2"]
        compiled, env = prepare(tmp_path, key, variant)
        with pytest.raises(Divergence, match="rank 0 arrived at .* rank 1"):
            world(geometry, 0).run(
                world_program(
                    bundles[key],
                    compiled,
                    env,
                    tmp_path / "mutant",
                    geometry,
                    self._skip_intern,
                )
            )
