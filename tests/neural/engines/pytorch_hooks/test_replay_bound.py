"""A CUDA graph replay registers with the replay deadline
(``docs/cuda_graphs.md`` "Hung replays"): with a heartbeat running — a
launched world above one — each [`Replay`][causalab.neural.engines.pytorch_hooks.cuda_graphs.Replay]
records an event after ``graph.replay()`` on the replay device's current
stream and hands it to the heartbeat, keyed by that stream; the event is
queried under the replay's device, never the heartbeat thread's own. With no
heartbeat (world 1, or a process that never joined a group) nothing is
recorded. The driver boundary is faked (``_fake_cuda``); the real bound on
NCCL is ``tests/golden/test_replay_deadline.py``.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch

from causalab.neural.shared.parallel import heartbeat as heartbeat_module
from causalab.neural.shared.parallel.deadline import REPLAY
from causalab.neural.shared.parallel.heartbeat import Heartbeat
from causalab.neural.shared.parallel.watchdog import Settings
from causalab.protocol.parallel import parse_geometry
from tests._helpers.simulated_world.heartbeat import FakeStore

pytestmark = pytest.mark.unit


class _Stream:
    cuda_stream = 0xABC
    device_index = 1

    def wait_stream(self, _other: Any) -> None:
        pass


class _Event:
    made: list["_Event"] = []

    def __init__(self) -> None:
        self.recorded_on: Any = None
        self.done = False
        self.devices: list[Any] = []
        _Event.made.append(self)

    def record(self, stream: Any = None) -> None:
        self.recorded_on = stream

    def query(self) -> bool:
        return self.done


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from tests.neural.engines.pytorch_hooks._fake_cuda import FakeCuda

    FakeCuda().install(monkeypatch)
    _Event.made = []
    entered: list[Any] = []

    @contextlib.contextmanager
    def device(index: Any):
        entered.append(index)
        yield

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a, **_k: _Stream())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "device", device)
    return entered


def _replay() -> Any:
    from causalab.neural.engines.pytorch_hooks.cuda_graphs import Replay

    return Replay(lambda: torch.ones(1), {}, device=torch.device("cpu"))


def test_without_a_heartbeat_a_replay_records_nothing(cuda: list[Any]) -> None:
    assert heartbeat_module.running() is None
    replay = _replay()
    replay()
    assert _Event.made == []
    assert replay.replays == 1


def test_under_a_heartbeat_each_replay_is_bounded_on_its_stream(
    cuda: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [7.0]
    beat = Heartbeat(
        FakeStore(),
        rank=0,
        world=2,
        settings=Settings(timeout=10.0, grace=1.0),
        geometry=parse_geometry("tp=2"),
        clock=lambda: clock[0],
        exit=lambda _status: None,
        write=lambda _text: None,
    )
    monkeypatch.setattr(heartbeat_module, "_running", beat)
    replay = _replay()
    replay()
    replay()
    assert len(_Event.made) == 2
    assert all(isinstance(event.recorded_on, _Stream) for event in _Event.made)
    assert beat.outstanding.pending() == 2
    # pending since the replay: overdue at the timeout, named as a replay
    overdue = beat.outstanding.overdue(17.0)
    assert overdue is not None and overdue.op == REPLAY and overdue.waited == 10.0
    # the query ran under the replay's device, resolved from its stream
    assert torch.device("cuda", _Stream.device_index) in cuda
    for event in _Event.made:
        event.done = True
    assert beat.outstanding.overdue(100.0) is None
    assert beat.outstanding.pending() == 0


def test_a_capture_under_a_heartbeat_opens_the_capture_window_after_the_drain(
    cuda: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heartbeat thread must not query events while this rank captures
    (``deadline.Outstanding.capturing``): the window opens after the device
    drained — a stuck earlier replay is still refused by name there — and
    closes once the capture ends."""
    beat = Heartbeat(
        FakeStore(),
        rank=0,
        world=2,
        settings=Settings(timeout=10.0, grace=1.0),
        geometry=parse_geometry("tp=2"),
        clock=lambda: 0.0,
        exit=lambda _status: None,
        write=lambda _text: None,
    )
    monkeypatch.setattr(heartbeat_module, "_running", beat)
    order: list[str] = []
    capturing = beat.outstanding.capturing

    @contextlib.contextmanager
    def window():
        order.append("open")
        with capturing():
            yield
        order.append("close")

    graph = torch.cuda.graph

    @contextlib.contextmanager
    def captured(*args: Any, **kwargs: Any):
        order.append("capture")
        with graph(*args, **kwargs):
            yield

    monkeypatch.setattr(beat.outstanding, "capturing", window)
    monkeypatch.setattr(torch.cuda, "graph", captured)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a: order.append("sync"))
    _replay()
    assert order[-4:] == ["sync", "open", "capture", "close"]


def test_without_a_heartbeat_a_capture_opens_no_window(cuda: list[Any]) -> None:
    assert heartbeat_module.running() is None
    replay = _replay()
    replay()
    assert replay.replays == 1
