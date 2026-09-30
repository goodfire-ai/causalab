"""Recompute task numerical pins.

A *task numerical pin* is the symbolic output of a task's
``generate_dataset(model, n, seed)`` at a fixed seed sequence
(`tests._helpers.task_pins.TASK_SEEDS`). One sidecar per task at
``tests/tasks/<task>/pinned_samples.json``, consumed by the per-task
test at ``tests/tasks/<task>/test_<task>_numerical.py``.

Not a "golden" — see docs/TESTS.md for the term split. Runner-scope
goldens are refreshed via ``tests/end_to_end/update_goldens.py``; this
script is for the task-scope, LM-free, sub-second symbolic pins.

Usage::

    # Show the diff for every task that has a sidecar; do not write.
    uv run python scripts/update_task_pins.py

    # Show the diff for one task; do not write.
    uv run python scripts/update_task_pins.py --task=MCQA

    # Write.
    uv run python scripts/update_task_pins.py --task=MCQA \\
        --i-have-reviewed-the-diff

    # Refresh every task that already has a sidecar.
    uv run python scripts/update_task_pins.py --i-have-reviewed-the-diff

The diff prints per-sample (input + each counterfactual_input), with
``-`` for the on-disk value and ``+`` for the freshly-walked value.
Without ``--i-have-reviewed-the-diff`` the script never writes; the
operator either re-runs with the flag or hand-edits the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Repo root is two levels up: scripts/update_task_pins.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests._helpers.task_pins import (  # noqa: E402
    enumerate_pinned_tasks,
    walk_task_samples,
    write_pinned_samples,
)

TASKS_ROOT = REPO_ROOT / "tests" / "tasks"


def _flatten_samples(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a payload to a key→value dict for the per-key diff table.

    Exposes each sample as ``samples[seed=N].input`` and
    ``samples[seed=N].counterfactual_inputs[i]`` so the diff is per-row
    rather than one giant blob.
    """
    flat: dict[str, Any] = {}
    if not payload:
        return flat
    for sample in payload.get("samples", []):
        seed = sample.get("seed")
        flat[f"samples[seed={seed}].input"] = sample.get("input")
        for i, cf in enumerate(sample.get("counterfactual_inputs", [])):
            flat[f"samples[seed={seed}].counterfactual_inputs[{i}]"] = cf
    return flat


def _print_diff(label: str, old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Print a per-key diff. Returns True if any key changed."""
    keys = sorted(set(old) | set(new))
    rows: list[tuple[str, str, str]] = []
    for key in keys:
        a = old.get(key, "<absent>")
        b = new.get(key, "<absent>")
        if a != b:
            rows.append((key, repr(a), repr(b)))

    if not rows:
        print(f"[{label}] no diff (every value matches the existing sidecar).")
        return False

    print(f"[{label}] diff:")
    width = max(len(r[0]) for r in rows)
    for key, a, b in rows:
        print(f"  {key:<{width}}  -{a}")
        print(f"  {' ':<{width}}  +{b}")
    return True


def _load_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as f:
        return json.load(f)


def _process_task(task_name: str, test_file_dir: Path, confirm: bool) -> bool:
    sidecar = test_file_dir / "pinned_samples.json"
    old_payload = _load_or_empty(sidecar)
    new_payload = walk_task_samples(task_name)

    changed = _print_diff(
        f"task:{task_name}",
        _flatten_samples(old_payload),
        _flatten_samples(new_payload),
    )
    if changed and not confirm:
        print(
            f"[task:{task_name}] dry-run only — re-run with "
            f"--i-have-reviewed-the-diff to overwrite."
        )
    if confirm:
        # Reach the file via the task_pins helper so the resolution
        # stays in one place. We pass the test file path because that's
        # the helper's contract (sidecar lives next to the test).
        test_file = test_file_dir / f"test_{task_name}_numerical.py"
        written = write_pinned_samples(test_file, new_payload)
        print(f"[task:{task_name}] wrote {written}")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Task name to refresh (e.g. 'MCQA'). If omitted, every task "
            "with an existing sidecar under tests/tasks/*/ is refreshed."
        ),
    )
    parser.add_argument(
        "--i-have-reviewed-the-diff",
        dest="confirm",
        action="store_true",
        help=(
            "Required to actually write the new pin. Without this flag, "
            "the script prints the diff and exits 0."
        ),
    )
    args = parser.parse_args(argv)

    if args.task:
        test_file_dir = TASKS_ROOT / args.task
        if not test_file_dir.is_dir():
            print(
                f"[task:{args.task}] no directory at {test_file_dir}. "
                f"Create tests/tasks/{args.task}/test_{args.task}_numerical.py "
                f"first, then re-run.",
                file=sys.stderr,
            )
            return 2
        tasks_to_process: list[tuple[str, Path]] = [(args.task, test_file_dir)]
    else:
        discovered = enumerate_pinned_tasks(TASKS_ROOT)
        if not discovered:
            print(
                "No task pins to refresh. Pass --task=<name> to bootstrap.",
                file=sys.stderr,
            )
            return 0
        tasks_to_process = [(name, sidecar.parent) for name, sidecar in discovered]

    any_changed = False
    for task_name, test_file_dir in tasks_to_process:
        if _process_task(task_name, test_file_dir, confirm=args.confirm):
            any_changed = True

    if not args.confirm and any_changed:
        print(
            "\nDry-run complete. Re-run with --i-have-reviewed-the-diff "
            "to write the new pins.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
