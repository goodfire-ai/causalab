"""Build the ``different_symbol`` pair table of the ``mcqa_symbol`` package.

The causal model is the MCQA task's ``@mechanism`` equations
(``causalab.tasks.MCQA.causal_models``): a two-option colour question whose
``answer`` is the letter at ``answer_position``. The counterfactual design is
``different_symbol`` (``causalab.tasks.MCQA.counterfactuals``): the
counterfactual prompt keeps the question and both colours and draws two new
letters. The variable the interchange replaces is ``answer``, so the ``label``
column, which the causal model computes by that interchange, is the
counterfactual prompt's letter for the correct colour.

The table holds two independent draws, as onboarding 07 fits and scores them:
128 ``train`` pairs from seed 1 and 64 ``test`` pairs from seed 2. Each draw
seeds Python's ``random`` module, which the generator reads, so the two splits
are the pairs of onboarding's ``mcqa/train_n128_s1`` and ``mcqa/test_n64_s2``.
The build refuses a table whose splits share a base or counterfactual prompt,
so the held-out scores come from prompts the fit never saw.

Usage::

    python workflows/scripts/mcqa_symbol/build_dataset.py --out artifacts/data/mcqa_symbol
    python workflows/scripts/mcqa_symbol/build_dataset.py --out artifacts/data/mcqa_symbol --check   # bytes moved?

``--out`` is the package's data folder or the table in it (``data.json``).
The table is deterministic in ``(--n-train, --seed-train, --n-test, --seed-test)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Callable

from causalab.tasks.MCQA import counterfactuals
from causalab.tasks.MCQA.causal_models import CAUSAL_MODEL
from causalab.tasks.serialize import (
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

TASK = "MCQA"
GENERATOR = "build_dataset.py"
TABLE = "data.json"
#: The counterfactual design and the variable its interchange replaces.
DESIGN: Callable[[], dict] = counterfactuals.different_symbol
TARGET = "answer"


def draw(n: int, seed: int) -> list[dict]:
    """``n`` pairs of the design from one seed of Python's ``random`` module,
    with the global state restored afterwards."""
    state = random.getstate()
    random.seed(seed)
    try:
        return [DESIGN() for _ in range(n)]
    finally:
        random.setstate(state)


def check_disjoint(train: list[dict], test: list[dict]) -> None:
    """Refuse a table whose train and test pairs share a prompt."""

    def prompts(examples: list[dict]) -> set[str]:
        out = set()
        for example in examples:
            out.add(example["input"]["raw_input"])
            out.update(cf["raw_input"] for cf in example["counterfactual_inputs"])
        return out

    shared = prompts(train) & prompts(test)
    if shared:
        raise SystemExit(
            f"train and test share {len(shared)} prompts, e.g. {sorted(shared)[0]!r}"
        )


def build(n_train: int, seed_train: int, n_test: int, seed_test: int):
    """The serialized table: ``train`` rows first, then ``test`` rows."""
    train = draw(n_train, seed_train)
    test = draw(n_test, seed_test)
    check_disjoint(train, test)
    return serialize_examples(
        CAUSAL_MODEL,
        train + test,
        split=["train"] * n_train + ["test"] * n_test,
        target_variables=[TARGET],
        task_label=TASK,
        generator=GENERATOR,
        n=n_train + n_test,
    )


def table_path(out: Path) -> Path:
    """``--out`` names the data folder or the table itself."""
    return out if out.suffix == ".json" else out / TABLE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="the data folder or its data.json"
    )
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--seed-train", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--seed-test", type=int, default=2)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed bytes differ",
    )
    args = parser.parse_args(argv)
    path = table_path(args.out)

    dataset = build(args.n_train, args.seed_train, args.n_test, args.seed_test)
    if args.check:
        fresh = table_bytes(dataset.rows)
        if not path.is_file() or path.read_bytes() != fresh:
            print(f"{path}: bytes differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{path}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
        return 0
    digest = write_dataset_table(dataset.rows, path)
    print(
        json.dumps(
            {
                "table": str(path),
                "rows": len(dataset.rows),
                "digest": digest[:12],
                "splits": dataset.split_counts,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
