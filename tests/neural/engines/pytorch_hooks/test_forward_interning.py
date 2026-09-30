"""Cross-point forward interning in the reference engine (spec §3, §4).

§3 promises that a swept document's shared sub-values are content-deduped —
"shared harvests and forwards fall out automatically" — and corpus 07's own
description spends the promise: a 32-layer x 2-position scan plans "64 patched
forwards plus one shared counterfactual-harvest forward". The planner has
always said so (``tests/protocol/test_corpus.py`` pins those digests), but a
flat per-point execution loop cannot claim it: it re-runs the shared harvest
once per point, because taps are deliberately absent from a group's digest and
each point taps a different layer.

Ground truth for "a forward happened" is a pre-hook on the loaded model
itself, not the engine's own tally, so a bookkeeping bug cannot make the
suite agree with itself; ``RunResult.forwards`` is then checked against it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import pytest

from causalab.cli import register_model_key
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.execution import campaign_plans, execute_request
from causalab.neural.shared.executor import CaptureKey, ForwardCache, Interning
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import RunContext, RunResult
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import interned_groups, plan_point
from causalab.io.env import ResolutionEnv

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import (
    CORPUS_DIR,
    FIXTURES,
    build_env,
    write_rot_fixture,
    steps_of,
)
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke

#: Corpus 07 at tiny scale. The document authors a 32-layer x 2-position grid
#: on Qwen3-8B; tiny-random has 2 layers, and its rows say nothing a
#: ``{"variable": ...}`` anchor can find, so the two swept axes become 2 layers
#: x 2 indices. The shape that matters survives: several points whose patched
#: forward differs and whose counterfactual harvest does not.
OVERRIDES = {
    "model.key": TINY_LLAMA,
    "sites.target.layers": {"sweep": [0, 1]},
    "positions.tap": {"sweep": [{"index": -1}, {"index": -2}]},
}


@pytest.fixture(scope="module")
def scan_env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    root = tmp_path_factory.mktemp("interning-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
    return build_env(root)


@pytest.fixture(scope="module")
def scan(scan_env: ResolutionEnv) -> CompiledProtocol:
    return compile_protocol(
        CORPUS_DIR / "07_weekdays_locate_scan_im.json",
        env=scan_env,
        overrides=OVERRIDES,
    )


def _run(
    loaded: CompiledProtocol,
    env: ResolutionEnv,
    out: Path,
    selected: Sequence[int] | None = None,
) -> tuple[CompiledProtocol, RunContext]:
    """The compiled campaign and one run context over a chosen subset of its
    points — the two arguments of ``Engine.execute``.

    The same shard ``--points`` selects (``causalab.protocol.reports``), so a
    one-point run is exactly the shard an external scheduler hands a worker
    — and, usefully here, a run with nothing to intern against."""
    points = None if selected is None else tuple(selected)
    return loaded, RunContext(output_dir=out, env=env, points=points)


@pytest.fixture()
def forwards() -> Iterator[list[int]]:
    """Every top-level call of the loaded model, counted at the model.

    ``load_model`` is memoized on its exact call form — ``quantization``
    included — so asking for it the way the engine's ``_executor`` does hands
    back the very object the engine will run; a call that differs in one
    keyword would silently hand back a second, unhooked model. No patching,
    and nothing private touched."""
    bundle = load_model(
        TINY_LLAMA, "main", dtype="fp32", device="cpu", quantization=None
    )
    calls: list[int] = []
    handle = bundle.model.register_forward_pre_hook(
        lambda _module, _args: calls.append(1)
    )
    try:
        yield calls
    finally:
        handle.remove()


def test_the_plan_shares_a_group_the_points_do_not(
    scan: CompiledProtocol, scan_env: ResolutionEnv
) -> None:
    """The premise, stated as data: the campaign's forward-group instances
    outnumber its distinct digests, and the whole surplus is counterfactual
    harvest — one digest, one tap per layer."""
    steps = steps_of(scan, scan_env)
    plans = campaign_plans(steps.documents, steps.canonical)
    groups = interned_groups(plans)
    assert sum(plan.num_forwards for plan in plans) == 8  # 4 points x 2 groups
    assert len(groups) == 5  # 4 distinct patched + 1 shared harvest
    (harvest,) = [group for group in groups if group.unwritten]
    # one tap per layer: the position axis moves the gather, not the forward
    assert len(harvest.taps) == 2
    # the shared pass has to reach every layer any point taps, so the depth it
    # may elide at is the deepest of the union rather than of one point (§4)
    assert harvest.stop_after == max(tap.depth for tap in harvest.taps)


def test_a_shared_forward_group_runs_once(
    scan: CompiledProtocol, scan_env: ResolutionEnv, forwards: list[int], tmp_path: Path
) -> None:
    """The gap this closes: the engine runs the campaign's distinct forward
    groups, not one per point per group.

    Before interning this counted 8 — the flat per-point loop re-ran the
    shared counterfactual harvest for every one of the four points.
    """
    steps = steps_of(scan, scan_env)
    plans = campaign_plans(steps.documents, steps.canonical)
    owed = len(interned_groups(plans))

    result = PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path))

    assert len(forwards) == owed, (
        f"{len(forwards)} forwards for a campaign whose plan has {owed} "
        "distinct groups — the shared harvest ran more than once"
    )
    assert len(forwards) < sum(plan.num_forwards for plan in plans)
    # the engine's own tally must agree with what the model actually saw
    assert result.forwards == len(forwards)


def test_interning_changes_no_number(
    scan: CompiledProtocol, scan_env: ResolutionEnv, tmp_path: Path
) -> None:
    """Interning is a pure performance change: one campaign request must
    produce exactly the table that one request per point produces.

    A one-point request has nothing to intern against, so this is a real
    before/after — the sharded runs execute the harvest once per point, the
    whole run executes it once, and every value and every coordinate column
    has to survive that unchanged.
    """
    whole = PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path / "whole"))
    shards = [
        PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path / f"point{i}", [i]))
        for i in range(len(steps_of(scan, scan_env).points))
    ]
    assert whole.forwards < sum(shard.forwards for shard in shards)

    for name in ("iia.json", "logit_diff.json"):
        interned = table_frame(tmp_path / "whole" / name)
        sharded = pd.concat(
            [table_frame(tmp_path / f"point{i}" / name) for i in range(len(shards))],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(interned, sharded)


def test_a_lone_point_still_runs_its_own_groups(
    scan: CompiledProtocol, scan_env: ResolutionEnv, forwards: list[int], tmp_path: Path
) -> None:
    """Nothing is shared *within* a point, so a single-point request still
    pays the plan's per-point ``num_forwards`` — interning must never drop a
    group a point genuinely needs."""
    result = PytorchHooksEngine().execute(*_run(scan, scan_env, tmp_path, [0]))
    assert plan_point(steps_of(scan, scan_env).documents[0]).num_forwards == 2
    assert result.forwards == 2
    assert len(forwards) == 2


def _execute_watching_the_store(
    compiled: CompiledProtocol, run: RunContext
) -> tuple[RunResult, ForwardCache, list[set[CaptureKey]]]:
    """Run ``compiled`` under ``run`` exactly as ``PytorchHooksEngine.execute``
    does, keeping the campaign's [`ForwardCache`][causalab.neural.shared.executor.cache.ForwardCache] in view.

    ``execute_request``'s ``executor_factory`` is the seam: it is handed each
    point's [`Interning`][causalab.neural.shared.executor.cache.Interning] (whose ``cache`` is the campaign's one store)
    right before the point runs, so a snapshot of the store's keys taken there
    is "what the store held when point *i* started"."""
    from causalab.neural.engines.pytorch_hooks.train import run_training

    engine = PytorchHooksEngine()
    stores: list[ForwardCache] = []
    snapshots: list[set[CaptureKey]] = []

    def factory(doc, ctx, coords, interning, resolution):
        assert interning is not None
        stores.append(interning.cache)
        snapshots.append(set(interning.cache.captured))
        return engine._executor(
            doc, ctx, coords=coords, interning=interning, resolution=resolution
        )

    result = execute_request(
        compiled,
        run,
        engine_name=engine.name,
        executor_factory=factory,
        train_runner=run_training,
        intern_forwards=True,
    )
    assert len(set(map(id, stores))) == 1, "one campaign, one store"
    return result, stores[0], snapshots


def test_a_capture_lives_with_its_sharers_not_with_the_request(
    scan: CompiledProtocol, scan_env: ResolutionEnv, forwards: list[int], tmp_path: Path
) -> None:
    """The memory half of §3: the store keeps a raw capture only while a pass
    is still owed it, and never stores one no other pass keys into.

    Before this, every group published — the four distinct *patched* forwards
    included — and nothing evicted, so a campaign pinned one raw capture per
    point until the request ended. On a mixing-mechanisms golden document
    (``tests/golden/protocols/mixing_scan_*_im.json``: gemma-2-2b-it, 150
    rows, a 7-layer sweep tapping ``lm_head``) that is 13 GiB per point, which
    runs out of memory on an 80 GB device, while the per-point loop passes on
    the same device in 47 s.

    The plan says which digests recur: here one harvest shared by four points
    (owed 4) and four patched groups owed once each. So the store must hold
    exactly the harvest between points, and nothing once the last point has
    been served.
    """
    steps = steps_of(scan, scan_env)
    plans = campaign_plans(steps.documents, steps.canonical)
    (harvest,) = [g for g in interned_groups(plans) if g.unwritten]
    patched = {g.key for g in interned_groups(plans) if not g.unwritten}
    assert len(patched) == len(steps_of(scan, scan_env).points)

    result, store, snapshots = _execute_watching_the_store(
        *_run(scan, scan_env, tmp_path)
    )

    # what each later point found: the shared harvest, and only that — a
    # patched capture nobody else keys into is not stored in the first place
    assert snapshots[0] == set()
    assert all(seen == {harvest.key} for seen in snapshots[1:]), snapshots
    # the last sharer settled it: the store is empty once the campaign is done
    assert store.captured == {} and store.routing == {}
    assert store.owed == {key: 0 for key in {harvest.key, *patched}}
    # and eviction cost no forward: the harvest still ran exactly once
    assert result.forwards == len(forwards) == len(interned_groups(plans))


def test_an_untracked_store_keeps_every_capture(
    scan: CompiledProtocol, scan_env: ResolutionEnv, tmp_path: Path
) -> None:
    """A store built without ``owed`` — a hand-built cache, an engine that
    planned no campaign — is the pre-eviction store: it keeps what it is given
    for the request. Lifetime tracking is opt-in by the count, so a caller
    that did not plan cannot have captures pulled out from under it."""
    engine = PytorchHooksEngine()
    steps = steps_of(scan, scan_env)
    plans = campaign_plans(steps.documents, steps.canonical)
    store = ForwardCache(
        wanted={
            g.key: tuple(doc.sites[t.site] for t in g.taps)
            for doc, plan in zip(steps_of(scan, scan_env).documents, plans)
            for g in plan.groups
        }
    )
    compiled, run = _run(scan, scan_env, tmp_path, [0])
    doc = steps_of(scan, scan_env).documents[0]
    handle = Interning(
        keys={(g.model, g.input): g.key for g in plans[0].groups}, cache=store
    )
    executor = engine._executor(
        doc, run, coords=steps_of(compiled, scan_env).points[0].coords, interning=handle
    )
    executor.run_all()
    assert set(store.captured) == {g.key for g in plans[0].groups}
    assert store.owed == {}
