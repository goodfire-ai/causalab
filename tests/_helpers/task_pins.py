"""Helpers for task-scope numerical pins.

A *task numerical pin* is the symbolic output of a task's
``generate_dataset(model, n, seed)`` at a fixed seed sequence, stored as
a sidecar ``pinned_samples.json`` next to the per-task test file at
``tests/tasks/<task>/test_<task>_numerical.py``. No LM, no analysis, no
runner — this is a pure ``CausalModel`` / counterfactual-generator pin.

This is **not** a "golden". Goldens (singular concept) are the
runner-scope, full-pipeline pins under
``tests/end_to_end/goldens/<runner>.json`` that run a small model which
*reliably solves* the task at the configured seed (see docs/TESTS.md
"Runner-golden standard"). Task numerical pins sit one layer below: if
a runner golden drifts, the task numerical pin is the localisation
check that says "is it the task layer or the LM/analysis layer".

Shared between the per-task tests (which assert sample-by-sample
equality) and ``scripts/update_task_pins.py`` (which regenerates the
JSON under a human-review gate). The walker lives here so generation
and assertion take exactly the same code path — no risk of a
regeneration emitting a different seed sequence than the test consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TASK_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""Canonical seed sequence for task numerical pins.

Five seeds is enough to catch index/order bugs in counterfactual
generators without flooding the diff when a task spec changes
legitimately. Kept in code (not JSON) so a regeneration can't silently
drift to a different seed set than the tests assert against.
"""

PINNED_SAMPLES_FILENAME = "pinned_samples.json"


def pinned_samples_path(test_file: str | Path) -> Path:
    """Return the sidecar path for ``test_<task>_numerical.py``."""
    return Path(test_file).resolve().parent / PINNED_SAMPLES_FILENAME


def walk_task_samples(task_name: str) -> dict[str, Any]:
    """Walk a task's ``generate_dataset(model, n, seed)`` at `TASK_SEEDS`.

    Every shipped task exposes ``generate_dataset`` in its
    ``counterfactuals.py`` per the loader convention (see
    ``causalab/tasks/loader.py::load_task_counterfactuals``). Each call
    returns a list of ``CounterfactualExample`` dicts containing
    ``{"input": CausalTrace, "counterfactual_inputs": [CausalTrace, ...]}``.

    Singleton tasks (``MCQA``, ``IOI``, ``hierarchical_equality``,
    ``entity_binding``) load with no config. Factory tasks
    (``graph_walk``, ``identity_naming``, ``natural_domains_arithmetic``)
    raise ``NotImplementedError`` until a serialisable ``task_cfg`` shim
    is added — the pilot covers singletons only.
    """
    from tests._helpers.tasks import FACTORY_TASKS, get_task

    if task_name in FACTORY_TASKS:
        raise NotImplementedError(
            f"Task '{task_name}' is a factory task. Task numerical pins for "
            f"factory tasks require a serialisable task_cfg; not yet "
            f"implemented. Extend this walker when the factory-task tranche "
            f"lands."
        )

    handle = get_task(task_name)
    cf_module = handle.counterfactuals_module

    samples: list[dict[str, Any]] = []
    for seed in TASK_SEEDS:
        examples = cf_module.generate_dataset(handle.causal_model, 1, seed)
        if not examples:
            continue
        example = examples[0]
        samples.append(
            {
                "seed": seed,
                "input": dict(example["input"].to_dict()),
                "counterfactual_inputs": [
                    dict(trace.to_dict()) for trace in example["counterfactual_inputs"]
                ],
            }
        )

    payload = {
        "task": task_name,
        "seeds": list(TASK_SEEDS),
        "samples": samples,
    }
    # Canonicalise via JSON round-trip so the walker emits exactly what the
    # JSON-loaded payload reads back (tuples → lists, etc). Otherwise tasks
    # whose CausalTrace dicts contain tuple values (e.g. entity_binding's
    # ``query_indices``) compare unequal between walker output and JSON load
    # even though the underlying data is identical.
    return json.loads(json.dumps(payload, sort_keys=True))


def load_pinned_samples(test_file: str | Path) -> dict[str, Any]:
    """Load the sidecar ``pinned_samples.json`` next to ``test_file``."""
    path = pinned_samples_path(test_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"No pinned-samples sidecar at {path}. Generate via "
            f"`uv run python scripts/update_task_pins.py --task=<name> "
            f"--i-have-reviewed-the-diff`."
        )
    with path.open() as f:
        return json.load(f)


def write_pinned_samples(test_file: str | Path, payload: dict[str, Any]) -> Path:
    """Overwrite the sidecar on disk. Returns the written path."""
    path = pinned_samples_path(test_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def enumerate_pinned_tasks(tasks_root: Path) -> list[tuple[str, Path]]:
    """Discover every task with a sidecar under ``tests/tasks/<task>/``.

    Returns ``[(task_name, sidecar_path), ...]`` sorted by task name.
    Used by ``scripts/update_task_pins.py`` to enumerate refresh targets.
    """
    results: list[tuple[str, Path]] = []
    if not tasks_root.is_dir():
        return results
    for sidecar in sorted(tasks_root.glob(f"*/{PINNED_SAMPLES_FILENAME}")):
        results.append((sidecar.parent.name, sidecar))
    return results
