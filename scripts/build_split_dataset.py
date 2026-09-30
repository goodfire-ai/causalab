"""Build ONE dataset table whose splits are group-disjoint (§2.2).

Sibling of ``build_task_dataset.py``. That script builds an undivided pool and
you tell it what to call the split (``--split all``); this one partitions the
task's unique inputs into disjoint groups, pairs counterfactuals *within* each
split, and writes a single table whose every row declares which split it is in.

A document then selects one with a ref fragment::

    "data":  {"base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"}}
    "train": {"eval": {"split": "natural_domains_arithmetic/data/weekdays#val", ...}}

so the disjointness behind those two names is a property of the bytes both refs
resolve against, not a claim about how this script was invoked. There is no
second file and no split-manifest sidecar to keep in sync.

Usage::

    uv run python scripts/build_split_dataset.py \\
        --task natural_domains_arithmetic --set domain_type=weekdays \\
        --seed 0 --fraction train=0.6 --fraction val=0.15 --fraction test=0.25 \\
        --out causalab/tasks/natural_domains_arithmetic/data/weekdays.json

    # Rebuild in place and fail if the bytes would move (determinism guard).
    uv run python scripts/build_split_dataset.py ... --check

``--set k=v`` follows ``build_task_dataset.py``'s semantics (JSON when it parses,
else a bare string). A model-correctness filter is deliberately not a flag: it
needs a loaded model, and a generated table carries no model dependency — build
the predicate yourself and call
[`causalab.tasks.splits.generate_split_dataset`][] with ``keep_pair=``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from causalab.tasks.loader import load_task
from causalab.tasks.serialize import (
    config_class,
    serialize_examples,
    table_bytes,
    write_dataset_table,
)
from causalab.tasks.splits import DEFAULT_FRACTIONS, generate_split_dataset


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set takes key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _parse_numbers(values: Sequence[str], flag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in values:
        key, _, raw = item.partition("=")
        if not key or not raw:
            raise SystemExit(f"{flag} takes <split>=<number>, got {item!r}")
        out[key] = float(raw)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_split_dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="a package under causalab/tasks/")
    parser.add_argument("--out", required=True, type=Path, help="table path (.json)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a config field for a factory task (repeatable)",
    )
    parser.add_argument(
        "--target-variable",
        action="append",
        default=[],
        metavar="NAME",
        help="variable(s) the interchange replaces; defaults to the task's own",
    )
    parser.add_argument("--answer-variable", default=None)
    parser.add_argument(
        "--group-key",
        default="input",
        help="'input' (the model-visible prompt) or an input variable to hold out by",
    )
    parser.add_argument(
        "--resample-variable",
        default="all",
        help="'all' pairs with a different in-split input; a variable name "
        "resamples just that variable",
    )
    parser.add_argument("--max-inputs", type=int, default=None)
    parser.add_argument(
        "--fraction",
        action="append",
        default=[],
        metavar="SPLIT=FLOAT",
        help=f"split weight (repeatable; default {DEFAULT_FRACTIONS})",
    )
    parser.add_argument(
        "--max-pairs",
        action="append",
        default=[],
        metavar="SPLIT=INT",
        help="cap on kept pairs per split, applied after the filters (repeatable)",
    )
    parser.add_argument(
        "--require-label-change",
        action="store_true",
        help="drop pairs whose interchange does not move the answer (symbolic)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the table would change",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides = _parse_set(args.set)
    cls = config_class(args.task)
    if overrides and cls is None:
        raise SystemExit(
            f"task {args.task!r} has no config dataclass, so --set configures nothing"
        )
    task_cfg = cls(**overrides) if cls is not None and overrides else None
    task = load_task(args.task, task_cfg=task_cfg)

    targets = args.target_variable or (
        [task.intervention_variable] if task.intervention_variable else []
    )
    if not targets:
        raise SystemExit(
            f"task {args.task!r} declares no intervention variable — pass "
            "--target-variable so the label is well defined"
        )

    built = generate_split_dataset(
        task,
        seed=args.seed,
        fractions=_parse_numbers(args.fraction, "--fraction") or None,
        group_key=args.group_key,
        resample_variable=args.resample_variable,
        max_inputs=args.max_inputs,
        max_pairs_per_split={
            k: int(v) for k, v in _parse_numbers(args.max_pairs, "--max-pairs").items()
        }
        or None,
        require_label_change=args.require_label_change,
        target_variable=targets[0],
    )

    dataset = serialize_examples(
        task.causal_model,
        built.examples,
        split=built.splits,
        target_variables=targets,
        answer_variable=args.answer_variable,
        task_label=args.task,
        generator="generate_split_dataset",
        n=len(built.examples),
        seed=args.seed,
    )

    if args.check:
        fresh = table_bytes(dataset.rows)
        on_disk = args.out.read_bytes() if args.out.is_file() else None
        if on_disk != fresh:
            reason = "does not exist" if on_disk is None else "differs"
            print(f"CHANGED {args.out} {reason} — rerun without --check to write")
            return 1
        print(f"unchanged {args.out} ({len(dataset.rows)} rows)")
        return 0

    digest = write_dataset_table(dataset.rows, args.out)
    # The partition audit is printed, not written beside the table: the split
    # column is the authority (§2.2), and nothing sits beside a table
    print(
        f"wrote {args.out} ({len(dataset.rows)} rows, "
        f"{dataset.split_counts}, digest {digest[:12]}…); split build: {built.audit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
