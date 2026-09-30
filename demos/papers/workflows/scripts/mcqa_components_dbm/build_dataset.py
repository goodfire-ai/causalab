"""Build the one table of the MCQA component DBM package.

The task is the two-choice colour question of ``causalab.tasks.MCQA``: a
prompt names an object's colour and offers it under one of two letters, and
the model answers with the letter. Its causal model is the ``@mechanism``
``equations`` function of ``causalab.tasks.MCQA.causal_models``, where
``answer_position`` is where the colour sits among the choices and
``answer`` is the letter at that position.

Every pair comes from ``different_symbol``: the counterfactual keeps the
object, the colour and the choices, and draws two new letters. The two
prompts then agree on ``answer_position`` and disagree on ``answer``, so an
interchange that makes the base prompt output the counterfactual's letter
moved the ``answer`` variable. ``label`` is that letter, the answer of the
causal model after the interchange of ``answer``.

``data.json`` holds two splits, selected as ``mcqa_components_dbm/data#train``
and ``#test``:

* ``train``: 128 pairs from seed 1, the pairs every gate is fitted on;
* ``test``: 64 pairs from seed 2, the held-out pairs every score is taken on.

These are the draws of the onboarding tutorial's ``mcqa/train_n128_s1`` and
``mcqa/test_n64_s2`` (``demos/onboarding_tutorial/04_define.md``): the rows
agree with those tables on every column but ``split``, which
``tests/demos/test_mcqa_components_dbm.py`` checks.

Usage::

    python workflows/scripts/mcqa_components_dbm/build_dataset.py --out artifacts/data/mcqa_components_dbm            # writes data.json
    python workflows/scripts/mcqa_components_dbm/build_dataset.py --out artifacts/data/mcqa_components_dbm --check    # exit 1 if the committed bytes differ
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from causalab.tasks.MCQA.causal_models import positional_causal_model
from causalab.tasks.MCQA.counterfactuals import generate_dataset
from causalab.tasks.serialize import (
    SerializedDataset,
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

TASK = "MCQA"
GENERATOR = "different_symbol"
#: The variable the interchange replaces: the letter, not its position.
TARGET = "answer"
#: (split, pairs, seed): the onboarding tutorial's fit and held-out draws.
SPLITS = (("train", 128, 1), ("test", 64, 2))


def build() -> dict[str, SerializedDataset]:
    """The ``data`` table: the train pairs, then the held-out pairs."""
    examples: list = []
    splits: list[str] = []
    for split, n, seed in SPLITS:
        # generate_dataset seeds the global RNG, draws `n` different_symbol
        # pairs and restores the RNG, so each split depends on its seed alone
        drawn = generate_dataset(positional_causal_model, n, seed)
        examples.extend(drawn)
        splits.extend([split] * len(drawn))
    dataset = serialize_examples(
        positional_causal_model,
        examples,
        split=splits,
        target_variables=[TARGET],
        task_label=TASK,
        generator=GENERATOR,
        n=len(examples),
    )
    return {"data": dataset}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory the table is written to, or one table's path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed table's bytes differ",
    )
    args = parser.parse_args(argv)
    tables = build()
    out = args.out
    if out.suffix == ".json":
        # a table path names one table: check or write that one alone
        tables = {name: t for name, t in tables.items() if out.name == f"{name}.json"}
        if not tables:
            print(f"{out}: not a table this builder writes", file=sys.stderr)
            return 1
        out = out.parent
    if args.check:
        status = 0
        for name, dataset in tables.items():
            path = out / f"{name}.json"
            fresh = table_bytes(dataset.rows)
            if not path.is_file() or path.read_bytes() != fresh:
                print(f"{path}: bytes differ from a fresh build", file=sys.stderr)
                status = 1
            else:
                print(f"{path}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
        return status
    out.mkdir(parents=True, exist_ok=True)
    for name, dataset in tables.items():
        digest = write_dataset_table(dataset.rows, out / f"{name}.json")
        print(f"{out / f'{name}.json'}: {len(dataset.rows)} rows, digest {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
