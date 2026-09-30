"""Rank failure and timeout cases on NCCL (``docs/model_parallelism.md``
§3 "when a rank dies", §10.6): the tiny Llama at ``tp=2 --device cuda`` in
``torchrun``'s shape, one rank going wrong in each of the ways
``tests/_helpers/watchdog_cases.py`` tables — a stop, a wedge, the host
stopped, the host exiting or killed, a death before the store and one
right after the group — the survivor held to the table's rules
(``check``, the CPU guard ``tests/_helpers/test_watchdog_cases.py``): the
named refusal, the status, the bound. The gloo twin is
``tests/neural/engines/pytorch_hooks/test_rank_watchdog_cases_run.py``.

Where NCCL differs, by design (§3): a collective there is an enqueued
kernel — the caller never blocks in it — so a **wedge** is seen by NCCL's
watchdog thread alone, and the only thing that ends the survivor is that
thread tearing the process down: ``Watchdog caught collective operation
timeout … To avoid data inconsistency, we are taking the entire process
down``, ``SIGABRT`` (``-6``). Torch sleeps sixty seconds first (four
``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``), which the launcher cuts to two
(``launcher.nccl_environment``); the case asserts the ``-6`` with NCCL's
words within the timeout plus that delay, and the record's ``nccl_teardown``
is **expected true** for the wedge and false for every other case.
Blocking wait disables the watchdog thread. The wedged rank, alive,
names the survivor a grace later (the host's store went with it) and exits
1 on its own. Under ``torchrun`` the agent's ``exitcode: -6`` line is what
the operator sees beside NCCL's words; under a spawn the parent's
``refused: [P4] at --parallel rank r of w was ended by NCCL's watchdog``.

Every case is subprocesses only (the test process loads nothing); each
writes a record under ``CAUSALAB_PARALLEL_GOLDENS_ROOT/watchdog/<case>.json``
when the root is set and resumes from it — a case whose record exists is
not re-run, as the parity replay resumes its runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch

from causalab.neural.shared.parallel.spawn import reserve_port
from tests._helpers import watchdog_cases as wc
from tests._helpers.dying_world import DyingWorld
from tests.golden._parallel.record import context
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

ROOT_VARIABLE = "CAUSALAB_PARALLEL_GOLDENS_ROOT"
DEADLINE_S = 240.0


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    kept = os.environ.get(ROOT_VARIABLE)
    base = Path(kept) if kept else tmp_path_factory.mktemp("parallel")
    root = base / "watchdog"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="module")
def document(root: Path) -> Path:
    return boundary._document(root, TINY_LLAMA)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def port() -> Iterator[int]:
    with reserve_port() as hold:
        yield hold.port


def _record_path(root: Path, case: wc.Case) -> Path:
    return root / f"{case.name}.json"


#: What torch's heartbeat monitor prints at construction, with the dump
#: wait the launcher set: the proof the variable reached ProcessGroupNCCL.
ACKNOWLEDGEMENT = "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 500"


def _acknowledged(stderr: str) -> bool:
    return ACKNOWLEDGEMENT in stderr


def observe_or_resume(
    case: wc.Case, root: Path, document: Path, port: int
) -> dict[str, Any]:
    """The case's record: read back when the root already holds one, else
    the case run on ``cuda`` and its observation written."""
    path = _record_path(root, case)
    if path.exists():
        return json.loads(path.read_text())
    out = root / case.name / "out"
    argv = boundary._argv(  # pyright: ignore[reportPrivateUsage]
        document, out, "--parallel", "tp=2", device="cuda"
    )
    (root / case.name).mkdir(parents=True, exist_ok=True)
    world = DyingWorld(
        argv,
        case,
        port=port,
        mark=root / case.name / "mark",
        under_way=out / "events.jsonl",
        stderr_dir=root / case.name,  # rank<r>.stderr, kept whatever happens
    )
    try:
        observed = world.observe(DEADLINE_S)
    finally:
        left = world.reap()
    record = {
        "case": case.name,
        "mode": case.mode,
        "victim": case.victim,
        "settings": {"timeout": case.settings.timeout, "grace": case.settings.grace},
        "bound_s": case.bound("nccl"),
        "backend": "nccl",
        "observed": observed.record(),
        "left_to_reap": left,
        "problems": wc.check(case, observed, "nccl"),
        # NCCL's own ending: its watchdog's words and its SIGABRT, no refusal
        # of ours — the expected outcome of the wedge, and of nothing else
        "nccl_teardown": wc.NCCL_TEARDOWN in observed.survivor_stderr
        and observed.survivor_status == wc.NCCL_TEARDOWN_STATUS,
        "async_error_handling": os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING"),
        # the ranks cut torch's post-timeout sleep themselves unless the
        # test's environment spells the wait (launcher.nccl_environment) …
        "dump_wait_ms": os.environ.get(
            "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "500 (launcher)"
        ),
        # … and torch's monitor acknowledges it at construction
        # ("HeartbeatMonitor environments: … TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC:
        # 500"), the proof it reached ProcessGroupNCCL
        "dump_wait_acknowledged": {
            f"rank{rank}": _acknowledged(world.stderr(rank))
            for rank in range(case.world)
        },
        "mark": (root / case.name / "mark").read_text()
        if (root / case.name / "mark").exists()
        else None,
        "context": context(),
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


@pytest.mark.parametrize("case", wc.CASES, ids=[c.name for c in wc.CASES])
def test_the_survivor_refuses_by_name_within_the_bound_on_nccl(
    case: wc.Case, root: Path, document: Path, port: int
) -> None:
    record = observe_or_resume(case, root, document, port)
    observed = record["observed"]
    print(
        f"{case.name} (nccl): survivor exited {observed['survivor_status']} "
        f"{observed['lag_s']} s after the act (bound {record['bound_s']:.1f} s); victim "
        f"status {observed['victim_status']}; NCCL teardown: {record['nccl_teardown']}"
    )
    assert record["problems"] == [], observed["survivor_stderr_tail"]
    assert record["left_to_reap"] == ([case.victim] if not case.victim_exits else [])
    assert record["nccl_teardown"] == (case.mode == "wedge"), (
        "NCCL's teardown is the wedge's ending and nobody else's"
    )
