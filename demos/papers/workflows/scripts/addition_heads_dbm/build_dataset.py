"""Build the counterfactual table of the attention-head DBM package.

The task is two-digit addition with no carry out of the tens column: the
prompt ``a1a0+b1b0=`` with tens digits 1 to 4 and ones digits 0 to 9, so
every answer has two digits and the first answer token is its tens digit.
Qwen3.5 tokenizes every digit as its own token, so the prompt is six tokens
and the model reads out the answer at ``=``.

The causal model computes the carry from the ones digits and the answer's
tens digit from the tens digits and the carry. The interchange replaces
``tens``, so the ``label`` column is the counterfactual problem's tens digit
and ``base_answer`` the base problem's. Every pair has two different tens
digits, so a swap that moves the answer and a swap that does nothing predict
different tokens.

**Prompt-disjoint splits.** causalab refuses a fit whose training rows and
held-out rows share a prompt at either endpoint (protocol spec §5 rule 22),
so the 1600 problems are shuffled and cut 80/20 first, and pairs are drawn
inside each part. The seed, the problem order and the sampling loop are
fixed, so a rebuild draws the same pairs and ``--check`` can compare bytes.

Usage::

    python workflows/scripts/addition_heads_dbm/build_dataset.py --out artifacts/data/addition_heads_dbm/data.json
    python workflows/scripts/addition_heads_dbm/build_dataset.py --out artifacts/data/addition_heads_dbm/data.json --check   # bytes moved?

The table is deterministic in ``(--seed, --n-train, --n-test, --test-fraction)``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

TENS = ["1", "2", "3", "4"]
ONES = [str(d) for d in range(10)]
#: tens digit of the answer: a1 + b1 + carry, from 1 + 1 + 0 to 4 + 4 + 1
ANSWER_TENS = [str(d) for d in range(2, 10)]
TASK = "two_digit_addition"
GENERATOR = "build_dataset.py"


def addition_model() -> CausalModel:
    """Column addition of ``a1a0 + b1b0``, scored at the answer's first token."""

    @mechanism
    def equations(a1: Dom(TENS), a0: Dom(ONES), b1: Dom(TENS), b0: Dom(ONES)):
        carry = V(int(a0) + int(b0) >= 10, domain=Dom([False, True]))
        ones = V(str((int(a0) + int(b0)) % 10), domain=Dom(ONES))  # noqa: F841
        tens = V(str(int(a1) + int(b1) + int(carry)), domain=Dom(ANSWER_TENS))
        raw_input = V(f"{a1}{a0}+{b1}{b0}=", domain=Dom(str))  # noqa: F841
        raw_output = V(tens, domain=Dom(str))  # noqa: F841
        return tens

    # One bare form per digit: the answer follows `=` with no space.
    scoring = ScoringSpec(
        forms={"tens": {d: (d,) for d in ANSWER_TENS}}, string_mode="exact"
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def problems() -> list[tuple[str, str, str, str]]:
    """Every ``(a1, a0, b1, b0)``, in the order the draw shuffles."""
    return list(itertools.product(TENS, ONES, TENS, ONES))


def tens_of(problem: tuple[str, str, str, str]) -> int:
    a1, a0, b1, b0 = (int(d) for d in problem)
    return a1 + b1 + int(a0 + b0 >= 10)


def sample_pairs(
    model: CausalModel,
    pool: list[tuple[str, str, str, str]],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """``n`` (base, counterfactual) pairs inside one split whose answers have
    different tens digits."""
    examples = []
    while len(examples) < n:
        base, cf = rng.choice(pool), rng.choice(pool)
        if base == cf or tens_of(base) == tens_of(cf):
            continue
        examples.append(
            {
                "input": model.new_trace(dict(zip(("a1", "a0", "b1", "b0"), base))),
                "counterfactual_inputs": [
                    model.new_trace(dict(zip(("a1", "a0", "b1", "b0"), cf)))
                ],
            }
        )
    return examples


def build(n_train: int, n_test: int, seed: int, test_fraction: float):
    rng = random.Random(seed)
    model = addition_model()
    pool = problems()
    rng.shuffle(pool)
    n_test_problems = round(len(pool) * test_fraction)
    test_pool, train_pool = pool[:n_test_problems], pool[n_test_problems:]
    examples = sample_pairs(model, train_pool, n_train, rng) + sample_pairs(
        model, test_pool, n_test, rng
    )
    return serialize_examples(
        model,
        examples,
        split=["train"] * n_train + ["test"] * n_test,
        target_variables=["tens"],
        task_label=TASK,
        generator=GENERATOR,
        n=n_train + n_test,
        seed=seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the table to write")
    parser.add_argument("--n-train", type=int, default=160)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed bytes differ",
    )
    args = parser.parse_args(argv)
    if args.out.is_dir():
        args.out = args.out / "data.json"

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
