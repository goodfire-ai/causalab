"""Build a serialized dataset table from a task package (spec §2.2).

Intervention specifications name datasets by ref, and refs resolve by reading
bytes — so a task's counterfactual dataset becomes usable by writing it out
once, here, rather than by generating it during a load. The bytes are
deterministic, so a table is reproducible from the command line that built it
— which is what a task's README or a workflow's description records; nothing
is written beside the table (spec §2.2).

Usage::

    # The weekdays interchange table the end-to-end IIA pin runs on.
    uv run python scripts/build_task_dataset.py \\
        --task natural_domains_arithmetic --set domain_type=weekdays \\
        --n 4 --seed 0 --split all --target-variable result \\
        --out tests/protocol/fixtures/data/weekdays/task_n4_s0.json

    # A relation of the subject_object_relations factory.
    uv run python scripts/build_task_dataset.py \\
        --task subject_object_relations --set relation=word_first_letter \\
        --n 64 --seed 0 --split all --out /tmp/word_first_letter.json

    # Rebuild in place and fail if the bytes moved (a determinism check).
    uv run python scripts/build_task_dataset.py ... --check

    # Validate the pairs under a tokenizer before writing (spec §2.2,
    # causalab/causal/pair_validation.py): every row's answers differ, and every row
    # that declares edit_groups carries its edit in tokens with no edit
    # outside the declared spans. Nothing is written if a row fails.
    uv run python scripts/build_task_dataset.py ... \\
        --validate-pairs --tokenizer meta-llama/Llama-3.1-8B --revision main

``--set k=v`` values are parsed as JSON when they parse, else kept as
strings, and are passed as keyword arguments to the task's config dataclass
(the convention [`causalab.tasks.serialize.config_class`][] resolves).
Singleton tasks take none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from causalab.causal.pair_validation import (
    EditGroupError,
    check_answer_change,
    check_intended_token_change,
    check_no_unintended_edits,
    parse_edit_groups,
)
from causalab.causal.scoring import STRING_MODE_COLUMN
from causalab.tasks.serialize import (
    config_class,
    serialize_counterfactual_dataset,
    table_bytes,
    write_dataset_table,
)


def _parse_set(values: Sequence[str]) -> dict[str, Any]:
    """``k=v`` pairs into config kwargs — same value semantics as the CLI's
    ``--set`` (JSON when it parses, else a bare string)."""
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


def _records_scoring(table: Path) -> bool:
    """Whether the table on disk carries the ``string_mode`` column — an
    absent or unreadable table records nothing, so a fresh build would."""
    if not table.is_file():
        return True
    try:
        rows = json.loads(table.read_text())
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(rows, list) or not rows:
        return True
    return any(isinstance(row, dict) and STRING_MODE_COLUMN in row for row in rows)


def _validate_pairs(
    rows: Sequence[dict[str, Any]], key: str, revision: str
) -> list[str]:
    """The pair-validity checks a builder can run — answer change on every
    row; intended token change and absence of unintended edits on every row
    that declares ``edit_groups`` — under the named tokenizer. Returns the
    failures, one line per row, empty when the table validates."""
    from transformers import AutoTokenizer  # a build that tokenizes pays for it

    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    problems: list[str] = []
    for index, row in enumerate(rows):
        try:
            check_answer_change(row)
            groups = parse_edit_groups(row)
            if groups:
                check_intended_token_change(tokenizer, row, groups)
                check_no_unintended_edits(tokenizer, row, groups)
        except EditGroupError as err:
            problems.append(f"row {index}: {err}")
    return problems


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_task_dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="a package under causalab/tasks/")
    parser.add_argument("--out", required=True, type=Path, help="table path (.json)")
    parser.add_argument("--n", required=True, type=int, help="counterfactual pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        required=True,
        help="the split every row of this table declares (§2.2). Required with "
        "no default: a single undivided pool is a claim worth stating "
        "(--split all), and a table that forgot to say is what the column "
        "exists to prevent. For a partitioned table use build_split_dataset.py",
    )
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
        help="variable(s) the interchange replaces; defaults to the task's "
        "TARGET_VARIABLE",
    )
    parser.add_argument(
        "--generator",
        default="generate_dataset",
        help="which generator in the task's counterfactuals.py to call",
    )
    parser.add_argument(
        "--answer-variable",
        default=None,
        help="variable whose declared forms supply the answer-form columns "
        "(default: the task's ScoringSpec.answer_variable)",
    )
    parser.add_argument(
        "--validate-pairs",
        action="store_true",
        help="run the pair-validity checks that need only the rows and a "
        "tokenizer (answer change; for rows declaring edit_groups, intended "
        "token change and absence of unintended edits — causalab/causal/pair_validation.py) "
        "and write nothing if any row fails. Needs --tokenizer",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        metavar="KEY",
        help="the tokenizer --validate-pairs validates under (a model key, as "
        "a document's model.key)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="the tokenizer's revision (default: main)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the table would change — the "
        "determinism guard for committed tables. A committed table built "
        "before the string_mode column existed is rebuilt without it, as its "
        "recipe says (the column is part of the recipe, not of the "
        "generator's determinism)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides = _parse_set(args.set)
    cls = config_class(args.task)
    if overrides and cls is None:
        raise SystemExit(
            f"task {args.task!r} has no config dataclass, so --set has nothing "
            "to configure"
        )
    task_cfg = cls(**overrides) if cls is not None and overrides else None

    # A new table always records its string_mode (spec §2.2). Under
    # --check the recipe is the table on disk: one written before the
    # column existed is reproduced without it, so the check keeps guarding
    # what it guards — the generator's and the causal model's determinism —
    # rather than failing every committed table the day the columns landed.
    reproduce = args.check
    record_scoring = not reproduce or _records_scoring(args.out)
    dataset = serialize_counterfactual_dataset(
        args.task,
        n=args.n,
        seed=args.seed,
        split=args.split,
        task_cfg=task_cfg,
        target_variables=args.target_variable or None,
        generator=args.generator,
        answer_variable=args.answer_variable,
        record_scoring=record_scoring,
    )
    tokenizer: dict[str, str] | None = None
    if args.validate_pairs:
        if args.tokenizer is None:
            raise SystemExit(
                "--validate-pairs needs --tokenizer <key> (and --revision)"
            )
        tokenizer = {"key": str(args.tokenizer), "revision": str(args.revision)}
        problems = _validate_pairs(
            dataset.rows, tokenizer["key"], tokenizer["revision"]
        )
        if problems:
            # fail-closed: a table whose pairs do not validate is not written
            print(
                f"refused: --validate-pairs found {len(problems)} invalid pair(s) "
                f"under tokenizer {tokenizer['key']}@{tokenizer['revision']}; "
                f"nothing written to {args.out}\n  " + "\n  ".join(problems),
                file=sys.stderr,
            )
            return 1
        print(
            f"validated {len(dataset.rows)} pairs under {tokenizer['key']}@"
            f"{tokenizer['revision']}"
        )
    if reproduce:
        fresh = table_bytes(dataset.rows)
        on_disk = args.out.read_bytes() if args.out.is_file() else None
        if on_disk != fresh:
            reason = "does not exist" if on_disk is None else "differs"
            print(f"CHANGED {args.out} {reason} — rerun without --check to write")
            return 1
        print(f"unchanged {args.out} ({len(dataset.rows)} rows)")
        return 0

    digest = write_dataset_table(dataset.rows, args.out)
    print(f"wrote {args.out} ({len(dataset.rows)} rows, digest {digest[:12]}…)")
    if dataset.match_mode == "prefix":
        print(
            f"note: the task's string_mode is 'prefix' for "
            f"{dataset.answer_variable!r} — a match metric over this table wants "
            '"mode": "first_token" (spec §2.10), and a document declaring '
            '"exact" over it is refused'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
