"""The rank watchdog's three unproven cases in a real world over ``gloo``
(``docs/model_parallelism.md`` §3 "when a rank dies", §10.6): the tiny
Llama at ``tp=2`` in ``torchrun``'s shape, one rank going wrong in each of
the ways ``tests/_helpers/watchdog_cases.py`` tables, the survivor held to
that table's rules by ``check`` — the named refusal, the status, the bound.

- **A hang without a death.** ``stop``: rank 1 ``SIGSTOP``s itself
  mid-forward — every thread stops, the heartbeat's included, so rank 0
  names it ``has sent no heartbeat`` inside the grace bound, exactly like a
  death; the stopped process is still there afterwards and this harness
  reaps it (pinned: ``reap`` names rank 1). ``wedge``: rank 1's main thread
  blocks forever while its heartbeat beats on; nothing names it — rank 0's
  next collective times out (``CAUSALAB_COLLECTIVE_TIMEOUT``, 15 s here),
  the hold proves rank 1 alive, and rank 0 prints the collective's own
  refusal naming the op, the axis and the timeout; rank 0's exit takes the
  store, and rank 1's heartbeat then names the host and ends the wedged
  process too. ``host-stop``: rank 0 stopped black-holes its store — it
  acknowledges connections without replying, so the store call can stall.
  Rank 1's bounded tick counts the store unreachable after a grace and
  names the host inside the same bound as a dead one.
- **Rank 0 dies.** ``host-exit`` / ``host-kill``: the store goes with rank
  0; rank 1's next tick fails and, a grace later, it names ``rank 0 of 2,
  the rendezvous store's host, is unreachable`` — inside the same bound as
  a peer's death, never a hang on the missing store.
- **Death during startup.** ``exit-before-join``: rank 1 exits before its
  store connection exists; rank 0 waits the collective timeout (10 s here)
  and names it ``never reached the rendezvous`` — the cost the store's own
  wait for workers had, with the rank named where torch named a count.
  ``exit-after-join``: rank 1 exits once the group has formed, before any
  collective; rank 0 — loading, or in ``new_group`` — is ended by its
  heartbeat inside the grace bound, with no ``waiting in`` clause.

The spawn path is one more test: the parent reaps a stopped child with
``SIGKILL`` after ``TERMINATE_GRACE_S``, so a spawned world with a stopped
rank ends within the bound plus that grace.

The NCCL twin is ``tests/golden/test_parallel_watchdog.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

from causalab.neural.shared.parallel.launcher import spawn
from causalab.neural.shared.parallel.spawn import TERMINATE_GRACE_S, reserve_port
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_VARIABLE,
)
from causalab.protocol.parallel import ParallelGeometry
from tests._helpers import dying_rank
from tests._helpers import watchdog_cases as wc
from tests._helpers.dying_world import DyingWorld
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

#: How long a whole case may take — the load, the act, the bound — before
#: the harness gives up on it.
DEADLINE_S = 150.0


@pytest.fixture(scope="module")
def document(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("watchdog-cases")
    return boundary._document(tmp, TINY_LLAMA)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def port() -> Iterator[int]:
    with reserve_port() as hold:
        yield hold.port


@pytest.mark.parametrize("case", wc.CASES, ids=[c.name for c in wc.CASES])
def test_the_survivor_refuses_by_name_within_the_bound(
    case: wc.Case, document: Path, tmp_path: Path, port: int
) -> None:
    out = tmp_path / "out"
    argv = boundary._argv(document, out, "--parallel", "tp=2")  # pyright: ignore[reportPrivateUsage]
    world = DyingWorld(
        argv,
        case,
        port=port,
        mark=tmp_path / "mark",
        under_way=out / "events.jsonl",
        stderr_dir=tmp_path,
    )
    try:
        observed = world.observe(DEADLINE_S)
    finally:
        left = world.reap()
    print(
        f"{case.name}: survivor exited {observed.survivor_status} {observed.lag:.2f} s "
        f"after the act (bound {case.bound():.1f} s); victim status {observed.victim_status}"
        + (
            f", {observed.victim_lag:.2f} s after the survivor"
            if observed.victim_lag
            else ""
        )
    )
    assert wc.check(case, observed) == [], (
        observed.survivor_stderr[-3000:] + observed.victim_stderr[-1500:]
    )
    # a stopped victim is the one process the harness had to end; every
    # other case's ranks were gone on their own
    assert left == ([case.victim] if not case.victim_exits else []), left


def test_the_spawn_parent_reaps_a_stopped_child(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd
) -> None:
    """Rank 1 stops itself; rank 0 refuses within the bound and exits 1; the
    parent then terminates rank 1 — ``SIGTERM`` pends on a stopped process,
    so after ``TERMINATE_GRACE_S`` it is ``SIGKILL``ed — and reports rank 0's
    exit: the whole world ends far inside the collective timeout. The
    children write to the process's real stderr, so ``capfd`` reads both
    rank 0's refusal and the parent's report."""
    case = wc.case_named("stop")
    monkeypatch.setenv(dying_rank.RANK_VARIABLE, "1")
    monkeypatch.setenv(dying_rank.MODE_VARIABLE, "stop")
    monkeypatch.setenv(RANK_GRACE_VARIABLE, f"{case.settings.grace:g}")
    monkeypatch.setenv(COLLECTIVE_TIMEOUT_VARIABLE, f"{case.settings.timeout:g}")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    argv = boundary._argv(document, tmp_path / "out", "--parallel", "tp=2")  # pyright: ignore[reportPrivateUsage]
    started = time.monotonic()
    code = spawn(ParallelGeometry(tensor=2), argv, entry=dying_rank.entry)
    wall = time.monotonic() - started
    err = capfd.readouterr().err
    print(
        f"spawn with a stopped rank: the whole world ended {wall:.2f} s after its start"
    )
    assert code == 1
    assert "refused: rank 0 of 2 exited with status 1 (--parallel " in err
    assert "rank 1 of 2 has sent no heartbeat" in err
    assert wall < case.bound() + TERMINATE_GRACE_S + 40.0, (
        f"the spawn took {wall:.1f} s"
    )
