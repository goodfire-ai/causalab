"""Task enumeration helper for property + numerical tier tests in ``tests/tasks/``.

Wraps the singleton / factory dispatch from
[`causalab.tasks.loader.load_task`][] (see
``causalab/tasks/loader.py:128-145``) so test harnesses can iterate over
each task's [`CausalModel`][causalab.causal.model.CausalModel] and
counterfactuals module without re-implementing the discovery logic.

The shipped tasks split into two patterns:

* **Singleton** (``CAUSAL_MODEL`` module-level constant) — ``MCQA``,
  ``IOI``, ``hierarchical_equality``, ``entity_binding``. The model is
  ready to use; no config needed.
* **Factory** (``CREATE_CAUSAL_MODEL`` callable) — ``graph_walk``,
  ``natural_domains_arithmetic``, ``identity_naming``. Callers must
  supply a config object (typed per the task — e.g.
  [`GraphWalkConfig`][causalab.tasks.graph_walk.config.GraphWalkConfig]).

Typical usage in a property test::

    from tests._helpers.tasks import get_task

    handle = get_task("MCQA")
    trace = handle.causal_model.sample_input()
    cf = handle.counterfactuals_module.random_counterfactual()

And in a factory case::

    from causalab.tasks.graph_walk.config import GraphWalkConfig
    cfg = GraphWalkConfig(graph_type="grid", graph_size=5, graph_size_2=5,
                          n_examples=4, context_length=3, seed=0)
    handle = get_task("graph_walk", task_cfg=cfg)

Property tests live one per task at ``tests/tasks/<task>/test_causal_models.py``
under ``pytest.mark.property``. Numerical-tier pins of symbolic outputs
live alongside them at ``tests/tasks/<task>/test_<task>_numerical.py``
with a sibling ``pinned_samples.json`` sidecar (see
`tests._helpers.task_pins`). Those are *task numerical pins*, not
"goldens" — goldens are runner-scope, full-pipeline, real-model.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from causalab.causal.model import CausalModel

# Task directory names under ``causalab/tasks/``. Order matters only for
# parametrisation reproducibility; alphabetised within each pattern.
SINGLETON_TASKS: tuple[str, ...] = (
    "IOI",
    "MCQA",
    "entity_binding",
    "hex_color",
    "hierarchical_equality",
)
FACTORY_TASKS: tuple[str, ...] = (
    "graph_walk",
    "identity_naming",
    "natural_domains_arithmetic",
    "subject_object_relations",
)
ALL_TASKS: tuple[str, ...] = SINGLETON_TASKS + FACTORY_TASKS


@dataclass(frozen=True)
class TaskHandle:
    """A loaded task: ``CausalModel`` + counterfactuals module.

    The handle deliberately omits the loader's full [`Task`][causalab.tasks.loader.Task] wrapper
    (intervention_variable / template / class-id glue) because
    causal-model property and numerical tests only need the symbolic
    layer.
    """

    name: str
    causal_model: CausalModel
    counterfactuals_module: ModuleType


def get_task(name: str, task_cfg: Any = None) -> TaskHandle:
    """Load one task by directory name.

    Args:
        name: Task directory name under ``causalab/tasks/`` (e.g. ``"MCQA"``,
            ``"graph_walk"``).
        task_cfg: Required for factory tasks; ignored for singletons. Pass the
            task's typed config object (e.g. ``GraphWalkConfig(...)``).

    Raises:
        ValueError: If the task's ``causal_models`` module exposes neither
            ``CAUSAL_MODEL`` nor ``CREATE_CAUSAL_MODEL``, or if a factory
            task is requested without ``task_cfg``.
    """
    cm_module = importlib.import_module(f"causalab.tasks.{name}.causal_models")
    is_factory = hasattr(cm_module, "CREATE_CAUSAL_MODEL")
    is_singleton = hasattr(cm_module, "CAUSAL_MODEL")

    if is_factory:
        if task_cfg is None:
            raise ValueError(f"Task '{name}' is a factory task — task_cfg is required.")
        causal_model = cm_module.CREATE_CAUSAL_MODEL(task_cfg)
    elif is_singleton:
        causal_model = cm_module.CAUSAL_MODEL
    else:
        raise ValueError(
            f"Task '{name}' must export either CAUSAL_MODEL (singleton) "
            f"or CREATE_CAUSAL_MODEL (factory) in causal_models.py."
        )

    cf_module = importlib.import_module(f"causalab.tasks.{name}.counterfactuals")
    return TaskHandle(
        name=name,
        causal_model=causal_model,
        counterfactuals_module=cf_module,
    )


def singleton_handles() -> list[TaskHandle]:
    """Return ``TaskHandle`` for every singleton task.

    Use in parametrised tests that don't need factory-specific configs.
    Factory tasks must be loaded explicitly with their config.
    """
    return [get_task(name) for name in SINGLETON_TASKS]
