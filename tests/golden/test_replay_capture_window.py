"""The replay deadline's capture window on a real device
(``causalab/neural/shared/parallel/deadline.py``, ``docs/cuda_graphs.md``
"Hung replays").

The heartbeat thread queries replay events. A completed event queried from
another thread while a global-mode capture is open fails the query and
invalidates the capture.
`test_a_query_during_capture_invalidates_it_without_the_window` pins that
behavior, so a driver or torch change that lifts the restriction shows up. With the
window open, the deadline asks nothing and the capture replays correctly.

The event is pending when the window opens and completes in the capture's own
entry synchronization, so the drain on entry cannot remove it: the case the
window's pause, not its drain, must cover.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

from causalab.neural.shared.parallel.deadline import REPLAY, Outstanding

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

REPO = Path(__file__).resolve().parents[2]
#: GPU spin cycles for the pending event: ~0.1–0.5 s on current cards
SPIN = int(5e8)


class _Event:
    """The production completion's query, keeping what it raised: the
    deadline counts a raising query as complete, so the error would
    otherwise be lost behind the capture's own."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self.event = event
        self.errors: list[str] = []

    def query(self) -> bool:
        try:
            return self.event.query()
        except RuntimeError as error:
            self.errors.append(str(error).splitlines()[0])
            raise


def _pending_event() -> torch.cuda.Event:
    side = torch.cuda.Stream()
    event = torch.cuda.Event()
    with torch.cuda.stream(side):
        torch.cuda._sleep(SPIN)  # pyright: ignore[reportPrivateUsage]
        event.record(side)
    return event


def _capture_with_a_query(
    window: bool, queried: list[str] | None = None
) -> torch.Tensor:
    """Capture ``y = x @ x + 1`` while another thread asks the deadline
    about an event that completed at the capture's entry; replay it.
    ``queried`` receives what the event's query raised."""
    x = torch.full((64, 64), 0.5, device="cuda")
    # warm-up: cuBLAS initializes outside the capture, as every capture site's does
    x @ x + 1
    torch.cuda.synchronize()
    outstanding = Outstanding(timeout=1e9)
    event = _pending_event()
    completion = _Event(event)
    queried = [] if queried is None else queried
    completion.errors = queried
    outstanding.enqueue("side", REPLAY, completion, now=0.0)
    graph = torch.cuda.CUDAGraph()
    failures: list[BaseException] = []

    def ask() -> None:
        try:
            outstanding.overdue(0.0)
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion
            failures.append(error)

    opened = outstanding.capturing() if window else contextlib.nullcontext()
    with opened, torch.cuda.graph(graph):
        y = x @ x + 1
        checker = threading.Thread(target=ask)
        checker.start()
        checker.join(30.0)
    assert event.query(), "the event completed in the capture's entry sync"
    assert not failures, failures
    assert not queried, queried
    graph.replay()
    torch.cuda.synchronize()
    return y


def test_the_window_keeps_the_deadline_out_of_a_capture() -> None:
    y = _capture_with_a_query(window=True)
    assert torch.equal(y.cpu(), torch.full((64, 64), 64 * 0.25 + 1))


def test_a_query_during_capture_invalidates_it_without_the_window() -> None:
    """The premise, in a process of its own: an invalidated capture can
    leave the context unusable."""
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tests.golden.test_replay_capture_window import _capture_with_a_query\n"
        "queried = []\n"
        "try:\n"
        "    _capture_with_a_query(window=False, queried=queried)\n"
        "except BaseException as error:\n"
        "    print('QUERY', queried)\n"
        "    print('CAPTURE', type(error).__name__, str(error).splitlines()[0])\n"
        "    raise SystemExit(3)\n"
        "print('CAPTURED')\n" % str(REPO)
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 3, done.stdout + done.stderr
    # the query itself was refused, and the capture it ran beside was lost
    assert "operation not permitted when stream is capturing" in done.stdout, (
        done.stdout
    )
    assert "cudaErrorStreamCaptureInvalidated" in done.stdout or (
        "previous error during capture" in done.stdout
    ), done.stdout
