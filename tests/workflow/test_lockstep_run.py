"""The workflow runner's lockstep under deterministic simulation
(``docs/model_parallelism.md`` §10.2, §10.4, §11; workflow spec §8): a world
of two ranks runs ``run_workflow`` in one process, on one thread, under a
drawn schedule, the joiner's decisions travelling through the production
[`CollectiveLockstep`][causalab.neural.shared.parallel.lockstep.CollectiveLockstep] over
the simulated collective and each protocol step's engine meeting its twin
at an all-reduce — so what a distributed run would report as a hang is a
typed refusal here, from a schedule tape.

``unit`` (the scenarios) and ``property`` (over drawn schedules — a
rank-uniform program is schedule independent):

* **lockstep.** A clean ``scan -> best`` run returns the joiner's manifest
  on every rank; the rank that does not publish ran the engine's half of
  the scan and wrote nothing — its own root, handed to it apart, stays
  absent.
* **a refusal on step 2 reaches every rank.** ``scan -> boom -> apply``: the
  script raises on the joiner; every rank stops with a refusal naming it,
  the joiner's manifest says ``boom`` failed and ``apply`` was blocked, and
  no rank hangs on ``apply``'s collective.
* **the joiner's reuse is every rank's.** Under ``--resume`` on a tree only
  the joiner holds, the joiner reuses the scan and no rank runs an engine.
* **mutation.** Ranks that decide for themselves — each reading its own
  tree — diverge: the joiner reuses, the other rank attempts, and its
  engine's all-reduce waits on a partner that has finished
  (`Abandoned`). The agreement is what keeps the ranks together.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.lockstep import CollectiveLockstep
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import RunContext, RunResult
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.lockstep import SOLO as SOLO_LOCKSTEP
from causalab.protocol.lockstep import LockstepRefusal
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

from tests._helpers.parallel_strategies import schedules
from tests._helpers.simulated_world import (
    Abandoned,
    Hang,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests.protocol._env import FIXTURES as PROTOCOL_FIXTURES
from tests.workflow.test_cli_parallel import (
    _Rank,  # pyright: ignore[reportPrivateUsage]
    _Rows,  # pyright: ignore[reportPrivateUsage]
)

FAN_OUT = Path(__file__).parent / "fixtures" / "fan_out"
WORLD = 2

_SETTINGS = settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class _Meeting(_Rows):
    """The stub engine with the one thing a sharded engine adds: a
    collective inside ``execute`` that every rank's engine must reach
    together — an all-reduce over the model group."""

    def __init__(self, collective: Collective) -> None:
        super().__init__()
        self.collective = collective

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        self.collective.all_reduce_sum(torch.ones(1), "tensor")
        return super().execute(compiled, run)


def _tree(tmp: Path, *, boom: bool = False) -> Path:
    """A private copy of the fan-out fixtures: ``scan -> best`` unfanned, or
    ``scan -> boom -> apply`` where ``boom`` is a script that raises."""
    root = tmp / "wf"
    shutil.copytree(FAN_OUT, root)
    raw = json.loads((root / "scan_wf.json").read_text())
    raw["steps"]["scan"].pop("fan_out")
    raw["output_dir"] = "run"
    if boom:
        (root / "scripts").mkdir()
        (root / "scripts" / "boom.py").write_text(
            "def main(inputs, outputs):\n    raise RuntimeError('boom')\n"
        )
        raw["steps"] = {
            "scan": raw["steps"]["scan"],
            "boom": {
                "type": "script",
                "script": {"path": "scripts/boom.py"},
                "inputs": {"table": {"step": "scan", "file": "iia.json"}},
                "outputs": {
                    "values": {
                        "file": "values.json",
                        "keys": {"best_pos": {"index": -1}},
                    }
                },
            },
            # downstream of boom through its emitted value, so the derived
            # order is scan -> boom -> apply and apply is blocked by the failure
            "apply": {
                "type": "intervention_protocol",
                "document": "protocols/apply.json",
                "set": {"positions.tap": {"artifact": "boom", "key": "best_pos"}},
            },
        }
    (root / "scan_wf.json").write_text(json.dumps(raw, indent=2) + "\n")
    return root / "scan_wf.json"


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(
            root=root, fallback_roots=(PROTOCOL_FIXTURES / "data", TASKS_ROOT)
        ),
        artifacts=FileArtifacts(root=root),
    )


def _program(document: Path, roots: list[Path], *, resume: bool, agreed: bool):
    """Every rank loads the workflow itself and runs it into ``roots[rank]``
    — the joiner's ROOT for rank 0, a root of its own for rank 1, so what
    rank 1 writes is visible. ``agreed`` is the production lockstep; without
    it every rank decides for itself (the mutation)."""

    def program(rank: int, collective: Collective) -> Any:
        env = _env(document.parent)
        loaded = load_workflow(document, env)
        engine = _Meeting(collective)
        publisher = _Rank(publish=(rank == 0) or not agreed)
        lockstep = CollectiveLockstep(collective) if agreed else SOLO_LOCKSTEP
        try:
            result = run_workflow(
                loaded,
                env,
                roots[rank],
                engine,
                resume=resume,
                publisher=publisher,
                lockstep=lockstep,
            )
        except Exception as err:  # the seam: what this rank stopped with, by type
            return {
                "refused": type(err).__name__,
                "protocol": isinstance(err, ProtocolError),
                "text": str(err),
                "executed": len(engine.runs),
            }
        return {"manifest": result.manifest, "executed": len(engine.runs)}

    return program


def _world(schedule: Schedule = ()) -> SimulatedWorld:
    return SimulatedWorld(
        groups_for(WORLD, tensor=WORLD), world=WORLD, schedule=schedule
    )


def _roots(tmp: Path) -> list[Path]:
    return [tmp / f"rank{rank}" for rank in range(WORLD)]


# --------------------------------------------------------------------------- #
# the scenarios
# --------------------------------------------------------------------------- #


class TestLockstep:
    pytestmark = pytest.mark.property

    @_SETTINGS
    @given(schedule=schedules())
    @example(schedule=[])
    def test_a_clean_run_returns_the_joiners_manifest_on_every_rank(
        self, tmp_path_factory: pytest.TempPathFactory, schedule: Schedule
    ) -> None:
        tmp = tmp_path_factory.mktemp("clean")
        document, roots = _tree(tmp), _roots(tmp)
        world = _world(schedule)
        results = world.run(_program(document, roots, resume=False, agreed=True))
        joiner, follower = results
        # every collective a rank reached, per step and not per point (the
        # scan has eight): the engine's one all-reduce, and the lockstep's
        # broadcast at each step's turn and record and at the manifest — a
        # follower blocks in exactly these and polls nothing
        by_op = {
            op: sum(1 for e in world.transcript if e.op == op and e.rank == 1)
            for op in ("all_reduce_sum", "broadcast")
        }
        assert by_op == {"all_reduce_sum": 1, "broadcast": 2 * 2 + 1}
        assert len(world.transcript) == 2 * (1 + 5)
        assert "manifest" in joiner and follower["manifest"] == joiner["manifest"]
        assert {s: e["status"] for s, e in joiner["manifest"]["steps"].items()} == {
            "scan": "completed",
            "best": "completed",
        }
        assert joiner["executed"] == follower["executed"] == 1
        assert (roots[0] / "run" / "workflow.json").is_file()
        assert not roots[1].exists(), "the rank that does not publish writes nothing"

    @_SETTINGS
    @given(schedule=schedules())
    def test_a_refusal_on_step_two_reaches_every_rank_and_no_rank_hangs(
        self, tmp_path_factory: pytest.TempPathFactory, schedule: Schedule
    ) -> None:
        """``boom`` raises on the joiner after ``scan``'s collective; the
        follower, waiting on the agreement, gets the refusal and never
        reaches ``apply``'s all-reduce — the world finishes (a hang would be
        the simulator's `Hang`, a lone arrival its `Abandoned`)."""
        tmp = tmp_path_factory.mktemp("boom")
        document, roots = _tree(tmp, boom=True), _roots(tmp)
        results = _world(schedule).run(
            _program(document, roots, resume=False, agreed=True)
        )
        joiner, follower = results
        # the script's own exception propagates on the joiner (an in-process
        # script raises as itself); the follower stops with it by name
        assert joiner["refused"] == "RuntimeError" and "boom" in joiner["text"]
        assert follower["refused"] == LockstepRefusal.__name__ and follower["protocol"]
        assert "RuntimeError" in follower["text"] and "boom" in follower["text"]
        assert "'boom'" in follower["text"], "the refusal names the step"
        assert joiner["executed"] == follower["executed"] == 1, "apply never ran"
        manifest = json.loads((roots[0] / "run" / "workflow.json").read_text())
        assert {s: e["status"] for s, e in manifest["steps"].items()} == {
            "scan": "completed",
            "boom": "failed",
            "apply": "blocked",
        }
        assert not roots[1].exists()


class TestReuse:
    pytestmark = pytest.mark.unit

    def _published(self, tmp: Path) -> tuple[Path, list[Path]]:
        """A world-1 run into the joiner's root, so ``--resume`` has a unit
        to reuse there — and nothing under the other rank's root."""
        document, roots = _tree(tmp), _roots(tmp)
        env = _env(document.parent)
        run_workflow(load_workflow(document, env), env, roots[0], _Rows())
        return document, roots

    def test_the_joiners_reuse_is_every_ranks(self, tmp_path: Path) -> None:
        document, roots = self._published(tmp_path)
        results = _world(0).run(_program(document, roots, resume=True, agreed=True))
        joiner, follower = results
        assert {s: e["status"] for s, e in joiner["manifest"]["steps"].items()} == {
            "scan": "reused",
            "best": "reused",
        }
        assert follower["manifest"] == joiner["manifest"]
        assert joiner["executed"] == follower["executed"] == 0
        assert not roots[1].exists()

    def test_ranks_deciding_for_themselves_diverge(self, tmp_path: Path) -> None:
        """Mutation: no agreement — every rank reads its own tree. The joiner
        reuses the scan; the other rank finds nothing and attempts it, and its
        engine's all-reduce waits on a partner that has already finished."""
        document, roots = self._published(tmp_path)
        with pytest.raises((Abandoned, Hang)):
            _world(0).run(_program(document, roots, resume=True, agreed=False))
