"""Adapt the workflow runner to the measurement collector.

The caller manages engine/model state, repetitions and resume, supplying fresh
scientific state for each seed. ``run_workflow`` publishes workflow outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from causalab.measurement import Operation
from causalab.protocol.engine import Engine
from causalab.io.env import ResolutionEnv
from causalab.workflow.document import LoadedWorkflow
from causalab.workflow.runner import WorkflowRunResult, run_workflow


def workflow_operation(
    load_for_seed: Callable[[int], LoadedWorkflow],
    env: ResolutionEnv,
    engine_for_seed: Callable[[int], AbstractContextManager[Engine]],
    observe: Callable[[WorkflowRunResult], Mapping[str, Any]],
) -> Callable[[int, Path], AbstractContextManager[Operation]]:
    """Measure actual workflow wall time, excluding compilation and engine setup.

    ``load_for_seed`` applies fitting seeds to the protocol. ``engine_for_seed``
    prepares and cleans up the engine, declaring model residency in the context.
    Timing includes lazy model loading and required output publication, and
    excludes ``observe``. Process startup is outside this timer.
    """

    @contextmanager
    def prepare(seed: int, directory: Path) -> Iterator[Operation]:
        loaded = load_for_seed(seed)
        with engine_for_seed(seed) as engine:
            yield Operation(
                run=lambda: run_workflow(loaded, env, directory, engine, resume=False),
                observe=observe,
            )

    return prepare
