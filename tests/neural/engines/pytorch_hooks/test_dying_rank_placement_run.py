"""Where the dying rank acts (``tests/_helpers/dying_rank.py``;
``docs/model_parallelism.md`` §10.6): the tiny Llama at ``tp=2`` over gloo
in ``torchrun``'s shape, rank 1 in the ``mark`` mode — it records the
moment it *would* have acted and runs on — and the mark read against the
run's event stream: the act lands **after the execute phase started and
before it completed**, at the rowwise style's output reduce of a
``Linear(hidden -> hidden)`` — ``o_proj``, the first rowwise module of a
Llama decoder layer. Hooking only the header exchange can defer the act
to publication because layer collectives may use the rowwise style.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pytest

from causalab.neural.shared.parallel.spawn import reserve_port
from causalab.neural.shared.parallel.watchdog import Settings
from tests._helpers.dying_world import DyingWorld
from tests._helpers.watchdog_cases import Case
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke

DEADLINE_S = 120.0


@pytest.fixture
def port() -> Iterator[int]:
    with reserve_port() as hold:
        yield hold.port


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_the_act_lands_mid_forward_at_the_first_rowwise_reduce(
    tmp_path: Path, port: int
) -> None:
    from transformers import AutoConfig

    document = boundary._document(tmp_path, TINY_LLAMA)  # pyright: ignore[reportPrivateUsage]
    out = tmp_path / "out"
    argv = boundary._argv(document, out, "--parallel", "tp=2")  # pyright: ignore[reportPrivateUsage]
    case = Case("placement", "mark", 1, Settings(timeout=120.0, grace=3.0))
    mark = tmp_path / "mark"
    world = DyingWorld(
        argv,
        case,
        port=port,
        mark=mark,
        under_way=out / "events.jsonl",
        stderr_dir=tmp_path,
    )
    try:
        exits = world.wait([0, 1], __import__("time").monotonic() + DEADLINE_S)
    finally:
        left = world.reap()
    assert left == [] and {r: e.code for r, e in exits.items()} == {0: 0, 1: 0}, (
        world.stderr(0)[-1500:] + world.stderr(1)[-1500:]
    )
    stamp, _, where = mark.read_text().rstrip("\n").partition(" ")
    acted = datetime.fromisoformat(stamp)
    config = AutoConfig.from_pretrained(TINY_LLAMA)
    hidden = config.hidden_size
    assert where == (
        f"after its first collective (rowwise reduce of Linear({hidden}->{hidden}))"
    ), where
    events = _events(out / "events.jsonl")
    by_kind = {e["event"]: e for e in events} if events and "event" in events[0] else {}
    kinds = [e.get("event") or e.get("kind") or e.get("type") for e in events]
    started = next(e for e in events if "phase_started" in json.dumps(e))
    completed = next(e for e in events if "phase_completed" in json.dumps(e))
    began = datetime.fromisoformat(_time(started))
    ended = datetime.fromisoformat(_time(completed))
    assert began < acted < ended, (began, acted, ended, kinds, by_kind.keys())
    print(
        f"placement: acted {(acted - began).total_seconds():.3f} s into the execute phase of {(ended - began).total_seconds():.3f} s, {where}"
    )


def _time(event: dict) -> str:
    for key in ("time", "at", "timestamp", "ts"):
        if key in event:
            return event[key]
    raise KeyError(f"no clock in the event {event!r}")
