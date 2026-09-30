"""Build the ``addition`` counterfactual table for the Figure 2a replication.

The task is the paper's control task (Feucht et al. 2026, Table 1): the prompt
``a+b=`` with ``a, b in 1..100`` and the answer ``a+b`` as a bare digit string.
The causal model has two input variables, ``a`` and ``b``, one intermediate
``sum`` and the two raw text variables the serializer reads. Figure 2a
localizes the **input concept**, which for this task is the first operand
``a``: patching ``a`` from the counterfactual prompt into the base prompt
should make the model answer ``a_cf + b_base``. That is the ``label`` column
the serializer computes when ``target_variables=["a"]``.

This script uses causalab as an installed library and nothing else — no task
package under ``causalab/tasks/``, no repository checkout. The pieces it
touches are the ones a hand-authored task needs:

* [`causalab.causal.model.CausalModel`][] and
  [`causalab.causal.model.mechanism`][] — the model;
* [`causalab.causal.scoring.ScoringSpec`][] — the answer's surface forms
  (bare ``"14"``: after ``=`` Llama-3.1 emits the digits with no space);
* [`causalab.tasks.serialize.serialize_examples`][] — the same row writer
  every shipped table went through, so ``causalab validate --data`` reads the
  result like any other table.

Two design choices differ from the paper's Appendix D.1 and are deliberate:

* **Prompt-disjoint splits.** The paper draws 4096 random pairs and sets 512
  aside. causalab refuses a fit whose training rows and ``train.eval.split``
  rows share a prompt at either endpoint (protocol spec §5 rule 22), so the
  10,000 prompts are partitioned first and pairs are sampled inside each
  partition. The test pairs therefore contain no prompt the fit ever saw.
* **``a_cf != a_base``.** A pair whose two first operands agree has a label
  equal to the base answer and cannot tell an intervention from a no-op; those
  1% of random pairs are rejected during sampling.

Usage::

    python workflows/scripts/arithmetic_fig2a/build_dataset.py --out artifacts/data/arithmetic_fig2a/data.json
    python workflows/scripts/arithmetic_fig2a/build_dataset.py --out artifacts/data/arithmetic_fig2a/data.json --check   # bytes moved?

The table is deterministic in ``(--seed, --n-train, --n-test, --test-fraction)``;
nothing is written beside it, and the recipe is the command line above, which
the README records (the consuming workflow pins the table's content digest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

OPERANDS = [str(i) for i in range(1, 101)]
SUMS = [str(i) for i in range(2, 201)]
TASK = "addition"
GENERATOR = "build_dataset.py"


def addition_model() -> CausalModel:
    """``a + b = sum``, rendered as the prompt ``{a}+{b}=`` and the answer ``{sum}``."""

    @mechanism
    def equations(a: Dom(OPERANDS), b: Dom(OPERANDS)):
        sum = V(str(int(a) + int(b)), domain=Dom(SUMS))
        raw_input = V(f"{a}+{b}=", domain=Dom(str))  # noqa: F841
        raw_output = V(sum, domain=Dom(str))  # noqa: F841
        return sum

    # One bare form per sum. `build_output_tokens` would add a space-prefixed
    # twin, which is the wrong surface form here: the answer follows `=`.
    scoring = ScoringSpec(forms={"sum": {s: (s,) for s in SUMS}}, string_mode="exact")
    return CausalModel(equations, id="addition", scoring=scoring)


def partition_prompts(
    rng: random.Random, test_fraction: float
) -> dict[str, list[tuple[str, str]]]:
    """Every (a, b) assigned to one split, so no prompt crosses the boundary."""
    pool = [(a, b) for a in OPERANDS for b in OPERANDS]
    rng.shuffle(pool)
    n_test = round(len(pool) * test_fraction)
    return {"test": pool[:n_test], "train": pool[n_test:]}


def sample_pairs(
    model: CausalModel,
    pool: list[tuple[str, str]],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """``n`` (base, counterfactual) pairs from one split's prompts, first
    operands differing, both prompts drawn from the same pool."""
    examples = []
    while len(examples) < n:
        base = rng.choice(pool)
        cf = rng.choice(pool)
        if base[0] == cf[0]:
            continue
        examples.append(
            {
                "input": model.new_trace({"a": base[0], "b": base[1]}),
                "counterfactual_inputs": [model.new_trace({"a": cf[0], "b": cf[1]})],
            }
        )
    return examples


def build(n_train: int, n_test: int, seed: int, test_fraction: float):
    rng = random.Random(seed)
    model = addition_model()
    pools = partition_prompts(rng, test_fraction)
    examples = sample_pairs(model, pools["train"], n_train, rng) + sample_pairs(
        model, pools["test"], n_test, rng
    )
    splits = ["train"] * n_train + ["test"] * n_test
    dataset = serialize_examples(
        model,
        examples,
        split=splits,
        target_variables=["a"],
        task_label=TASK,
        generator=GENERATOR,
        n=n_train + n_test,
        seed=seed,
    )
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the table to write")
    parser.add_argument("--n-train", type=int, default=3584)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed bytes differ",
    )
    args = parser.parse_args(argv)

    dataset = build(args.n_train, args.n_test, args.seed, args.test_fraction)
    if args.check:
        fresh = table_bytes(dataset.rows)
        if not args.out.is_file() or args.out.read_bytes() != fresh:
            print(f"{args.out}: bytes differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{args.out}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
        return 0
    digest = write_dataset_table(dataset.rows, args.out)
    print(
        json.dumps(
            {
                "table": str(args.out),
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
