"""Build the tables of the addition-neurons package (Feucht et al. 2026, Figure 8a).

Section 5 of the paper studies 28 MLP neurons at layer 18 of Llama-3.1-8B that
compute a sum on every task: months, weekdays, hours and ``a+b=``. This
package trains a desiderata-based mask (DBM) over all 14336 neurons of that
MLP on the weekdays task, compares the mask with the paper's 28 neurons, and
reads the neurons' activations for Figure 8a. Two tables follow, one causal
model each:

* ``weekdays.json``: counterfactual pairs of the weekdays prompt
  ``Q: What day is {offset} days after {day}?\\nA:`` (Table 1: seven days,
  offsets one to fourteen, 98 prompts). The interchanged variable is
  ``sum``, the pre-modulo sum with Monday = 1 (Table 1), so the ``label``
  column is the counterfactual prompt's own output day. The DBM fit reads the
  ``train`` split, its per-epoch evaluation and the choice of the L1 weight
  read ``val``, and every held-out score on the page reads ``test``.
* ``addition_prompts.json``: the 10,000 prompts ``a+b=`` with
  ``a, b in 1..100``, one row each, for Figure 8a, whose x axis is the output
  sum of the addition task. It is a table of pairs whose counterfactual is
  the row itself, as in ``rome_fig1``: the document that reads it names only
  ``input``.

Two choices differ from the paper's Appendix D.1 and are deliberate:

* **Prompt-disjoint splits.** causalab refuses a fit whose training rows and
  held-out rows share a prompt (protocol spec §5 rule 22), so the 98 weekdays
  prompts are partitioned first: 56 for ``train``, 21 for ``val`` and 21 for
  ``test``. Each output day has 14 prompts, and each held-out split takes 3
  of them, so every output day is equally common among the held-out prompts
  and no single answer dominates a held-out score. The ``train`` split is 2048
  distinct pairs drawn from the 56; the ``val`` and ``test`` splits are every
  usable ordered pair of their 21.
* **No informationless pairs.** A pair whose two prompts have the same output
  day has a label equal to the base answer and cannot tell an interchange from
  a no-op, so it is left out.

No pair is filtered on the model's own answers; the paper keeps only prompts
the model answers correctly, which a builder without the model cannot do.

This script uses causalab as an installed library: the models are
[`causalab.causal.model.CausalModel`][] equations under
[`causalab.causal.model.mechanism`][], the answer forms are a
[`causalab.causal.scoring.ScoringSpec`][], and the rows go through
[`causalab.tasks.serialize.serialize_examples`][].

Usage::

    python workflows/scripts/arithmetic_neurons/build_dataset.py --out artifacts/data/arithmetic_neurons           # writes the two tables
    python workflows/scripts/arithmetic_neurons/build_dataset.py --out artifacts/data/arithmetic_neurons --check   # exit 1 if a committed table's bytes differ

A table path in place of the directory checks or writes that table alone.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

from causalab.causal import CausalModel, CausalTrace, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    SerializedDataset,
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

TASK = "arithmetic_neurons"
GENERATOR = "build_dataset.py"

#: Table 1 of the paper: Monday = 1.
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#: The weekdays offsets as the paper spells them (Table 1: "one ... fourteen").
NUMBER_WORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
]
OPERANDS = [str(i) for i in range(1, 101)]
ADDITION_SUMS = [str(i) for i in range(2, 201)]

#: The weekdays template of the paper (Appendix A.1, Figures 15 and 16).
WEEKDAYS_TEMPLATE = "Q: What day is {offset} days after {concept}?\nA:"

#: The weekdays split (module docstring): of the 14 prompts of each output
#: day, 3 go to ``val`` and 3 to ``test``.
N_HELD_OUT_PER_DAY = 3
N_TRAIN_PAIRS = 2048
SEED = 0


def cyclic_model(
    task: str, concepts: list[str], n_offsets: int, template: str
) -> CausalModel:
    """``concept + offset = sum``, and the output concept is ``sum`` modulo the
    cycle, counted from 1 (Table 1). ``sum`` is the variable the DBM mask is
    trained to carry; the answer is the output concept with a leading space."""
    offsets = NUMBER_WORDS[:n_offsets]
    number = {word: i + 1 for i, word in enumerate(offsets)}
    index = {name: i + 1 for i, name in enumerate(concepts)}
    period = len(concepts)
    sums = list(range(2, period + n_offsets + 1))

    @mechanism
    def equations(concept: Dom(concepts), offset: Dom(offsets)):
        sum = V(index[concept] + number[offset], domain=Dom(sums))
        output = V(concepts[(sum - 1) % period], domain=Dom(concepts))
        raw_input = V(  # noqa: F841
            template.format(offset=offset, concept=concept), domain=Dom(str)
        )
        raw_output = V(" " + output, domain=Dom(str))  # noqa: F841
        return output

    # The answer follows "A:", so the space-prefixed spelling is the one the
    # model emits; the bare one is listed as an equivalent form, as in the
    # arithmetic_fig15 table.
    scoring = ScoringSpec(
        forms={"output": {c: (" " + c, c) for c in concepts}}, string_mode="exact"
    )
    return CausalModel(equations, id=task, scoring=scoring)


def addition_model() -> CausalModel:
    """``a + b = sum`` as the prompt ``{a}+{b}=``, the paper's control task;
    the same model as the ``arithmetic_fig2a`` builder."""

    @mechanism
    def equations(a: Dom(OPERANDS), b: Dom(OPERANDS)):
        sum = V(str(int(a) + int(b)), domain=Dom(ADDITION_SUMS))
        raw_input = V(f"{a}+{b}=", domain=Dom(str))  # noqa: F841
        raw_output = V(sum, domain=Dom(str))  # noqa: F841
        return sum

    # After `=` Llama-3.1 emits the digits with no space.
    scoring = ScoringSpec(
        forms={"sum": {s: (s,) for s in ADDITION_SUMS}}, string_mode="exact"
    )
    return CausalModel(equations, id="addition", scoring=scoring)


def weekdays_model() -> CausalModel:
    return cyclic_model("weekdays", DAYS, 14, WEEKDAYS_TEMPLATE)


def prompt_table(
    model: CausalModel, settings: list[dict[str, str]], target: str
) -> SerializedDataset:
    """One row per prompt, each its own counterfactual (module docstring)."""
    examples = []
    for setting in settings:
        trace = model.new_trace(setting)
        examples.append({"input": trace, "counterfactual_inputs": [trace]})
    return serialize_examples(
        model,
        examples,
        split="all",
        target_variables=[target],
        task_label=model.id,
        generator=GENERATOR,
        n=len(examples),
        seed=None,
    )


def weekdays_pairs(model: CausalModel) -> SerializedDataset:
    """The DBM table: prompt-disjoint ``train``, ``val`` and ``test`` splits
    of pairs whose output days differ (module docstring)."""
    rng = random.Random(SEED)

    def trace(prompt: tuple[str, str]) -> CausalTrace:
        return model.new_trace({"concept": prompt[0], "offset": prompt[1]})

    pools: dict[str, list[tuple[str, str]]] = {"test": [], "val": [], "train": []}
    n = N_HELD_OUT_PER_DAY
    for day in DAYS:
        prompts = [
            (d, o) for d in DAYS for o in NUMBER_WORDS if trace((d, o))["output"] == day
        ]
        rng.shuffle(prompts)
        pools["test"] += prompts[:n]
        pools["val"] += prompts[n : 2 * n]
        pools["train"] += prompts[2 * n :]

    def usable(pool: list[tuple[str, str]]) -> list[tuple[CausalTrace, CausalTrace]]:
        pairs = []
        for base in pool:
            for cf in pool:
                b, c = trace(base), trace(cf)
                if b["output"] != c["output"]:
                    pairs.append((b, c))
        return pairs

    train = usable(pools["train"])
    rng.shuffle(train)
    train = train[:N_TRAIN_PAIRS]
    val = usable(pools["val"])
    test = usable(pools["test"])
    pairs = train + val + test
    splits = ["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)
    examples = [{"input": b, "counterfactual_inputs": [c]} for b, c in pairs]
    return serialize_examples(
        model,
        examples,
        split=splits,
        target_variables=["sum"],
        task_label="weekdays",
        generator=GENERATOR,
        n=len(examples),
        seed=SEED,
    )


def build() -> dict[str, SerializedDataset]:
    """Every table this package commits, by file stem."""
    return {
        "weekdays": weekdays_pairs(weekdays_model()),
        "addition_prompts": prompt_table(
            addition_model(),
            [{"a": a, "b": b} for a in OPERANDS for b in OPERANDS],
            "sum",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory the tables are written to, or one table's path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if any committed table's bytes differ",
    )
    args = parser.parse_args(argv)
    one_table = args.out.suffix == ".json"
    out = args.out.parent if one_table else args.out
    tables = build()
    if one_table:
        tables = {n: t for n, t in tables.items() if args.out.name == f"{n}.json"}
        if not tables:
            print(f"{args.out}: not a table this builder writes", file=sys.stderr)
            return 1
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
    for name, dataset in tables.items():
        digest = write_dataset_table(dataset.rows, out / f"{name}.json")
        print(
            f"{out / f'{name}.json'}: {len(dataset.rows)} rows "
            f"{dataset.split_counts}, digest {digest[:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
