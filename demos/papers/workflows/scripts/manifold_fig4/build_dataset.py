"""Build the two tables of the Figure 4 (weekdays) replication.

The task is the paper's weekdays task (Wurgaft et al. 2026, App. A.1): the
prompt ``Q: What day is {number} days after {entity}?\\nA:`` with a weekday
``entity`` and a word ``number`` from ``one`` to ``seven``, and the answer the
day that many days later. The causal model has the two inputs, the ``result``
day and the two text variables the serializer reads.

* ``data.json``: every one of the 7 x 7 prompts once, entity-major, the pool
  the harvest, the PCA, both manifolds and the behavior centroids are
  computed over (App. A.3, A.4). Nothing reads its counterfactual columns,
  so each row's counterfactual is the row itself.
* ``steer_prompts.json``: the 16 base prompts of the steering runs (App.
  A.6). The paper's code draws them as ``generate_dataset(causal_model, 100,
  seed + 100)[:16]`` with ``seed = 42`` (``path_steering/main.py`` and
  ``natural_domains_arithmetic/counterfactuals.py`` of goodfire-ai/causalab,
  branch ``manifold_steering``, commit ``1b6f43a5``): per example it samples
  an input and a counterfactual, each by ``random.choice`` over the entities
  and then the numbers (``CausalModel.sample_input``, the model's inputs in
  declared order). `paper_draw` makes the same calls on
  ``random.Random(142)``, so the table holds the paper's prompts in the
  paper's order, 14 distinct and two of them twice, each row with the
  counterfactual drawn beside it.

This script uses causalab as an installed library and nothing else: no task
package under ``causalab/tasks/``. Both tables carry the causalab column
vocabulary (``causalab.tasks.serialize``), so ``causalab validate --data``
reads them like any shipped table.

Usage, from ``demos/papers/``::

    python workflows/scripts/manifold_fig4/build_dataset.py --out artifacts/data/manifold_fig4            # writes both tables
    python workflows/scripts/manifold_fig4/build_dataset.py --out artifacts/data/manifold_fig4 --check    # exit 1 if the committed bytes differ
    python workflows/scripts/manifold_fig4/build_dataset.py --out artifacts/data/manifold_fig4/data.json  # one table

A ``--out`` path that ends in ``.json`` names one table; any other path is
the directory of both, created if it does not exist.

The tables are deterministic; nothing is written beside them, and the
workflow that reads them pins their content digests.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    SerializedDataset,
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

__all__ = ["DAYS", "NUMBERS", "build", "paper_draw", "weekday_model"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBERS = ["one", "two", "three", "four", "five", "six", "seven"]
TEMPLATE = "Q: What day is {number} days after {entity}?\nA:"
TASK = "manifold_fig4_weekdays"
GENERATOR = "build_dataset.py"
#: ``generate_dataset(causal_model, n_eval_samples, seed + 100)[:n_prompts]``
#: with the code's ``seed: 42``, ``n_eval_samples: 100`` and ``n_prompts: 16``.
PAPER_SEED = 142
PAPER_SAMPLES = 100
PAPER_PROMPTS = 16


def later_day(entity: str, number: str) -> str:
    """The day ``number`` days after ``entity``."""
    return DAYS[(DAYS.index(entity) + NUMBERS.index(number) + 1) % len(DAYS)]


def weekday_model() -> CausalModel:
    """``(entity, number) -> result``, rendered as the paper's prompt and the
    space-prefixed answer."""

    @mechanism
    def equations(entity: Dom(DAYS), number: Dom(NUMBERS)):
        result = V(later_day(entity, number), domain=Dom(DAYS))
        raw_input = V(TEMPLATE.format(number=number, entity=entity), domain=Dom(str))  # noqa: F841
        raw_output = V(" " + result, domain=Dom(str))  # noqa: F841
        return result

    scoring = ScoringSpec(
        forms={"result": {d: (" " + d, d) for d in DAYS}}, string_mode="exact"
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def paper_draw(
    n: int = PAPER_SAMPLES, seed: int = PAPER_SEED, keep: int = PAPER_PROMPTS
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """The first ``keep`` of ``n`` (input, counterfactual) draws of the
    paper's ``generate_dataset``: per example, ``random.choice`` of an entity
    and then of a number for the input, then the same for the
    counterfactual, after ``random.seed(seed)``."""
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        base = {"entity": rng.choice(DAYS), "number": rng.choice(NUMBERS)}
        counterfactual = {"entity": rng.choice(DAYS), "number": rng.choice(NUMBERS)}
        draws.append((base, counterfactual))
    return draws[:keep]


def build() -> dict[str, SerializedDataset]:
    """Both tables, keyed by file stem."""
    model = weekday_model()
    pool = [
        model.new_trace({"entity": entity, "number": number})
        for entity in DAYS
        for number in NUMBERS
    ]
    examples = {
        "data": [{"input": trace, "counterfactual_inputs": [trace]} for trace in pool],
        "steer_prompts": [
            {
                "input": model.new_trace(base),
                "counterfactual_inputs": [model.new_trace(counterfactual)],
            }
            for base, counterfactual in paper_draw()
        ],
    }
    seeds = {"data": None, "steer_prompts": PAPER_SEED}
    return {
        name: serialize_examples(
            model,
            rows,
            split="all",
            target_variables=["result"],
            task_label=TASK,
            generator=GENERATOR,
            n=len(rows),
            seed=seeds[name],
        )
        for name, rows in examples.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory of both tables, or the path of one table (*.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if any committed table's bytes differ",
    )
    args = parser.parse_args(argv)
    tables = build()
    if args.out.suffix == ".json":
        # a table path names one table: check or write that one alone
        out = args.out.parent
        tables = {
            name: t for name, t in tables.items() if args.out.name == f"{name}.json"
        }
        if not tables:
            print(f"{args.out}: not a table this builder writes", file=sys.stderr)
            return 1
    else:
        out = args.out
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
