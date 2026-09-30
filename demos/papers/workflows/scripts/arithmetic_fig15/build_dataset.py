"""Build the weekdays counterfactual table for the Figure 15 replication.

Figure 15 of Feucht et al. 2026 patches the residual stream on "the
counterfactual dataset from Section D.1" (Appendix C): 4096 pairs of weekday
prompts. Section D.1 samples them from prompts that Llama-3.1-8B answers
correctly. The authors' code draws random pairs and keeps a pair when the
model's 5-token greedy continuation of both prompts passes a two-way
substring test against the answer (``src/generate_dataset.py`` and
``src/utils.py`` of the source repository), and the authors publish the pairs
it kept. ``artifacts/data/arithmetic_fig15/filtered_dataset.json`` holds the
records of
``goodfire-ai/arithmetic-wild@d03024a9/datasets/Llama-3.1-8B/weekdays/filtered_dataset.json``,
wrapped in an object that names the source and the sha256 of the original
bytes. This script reads the wrapper, checks that its records rebuild those
bytes, and writes ``data.json``, one row per pair in file order.

The causal model is the paper's (Appendix B, Table 1): an ``offset`` word
(``one`` to ``fourteen``) and an ``input_day`` make the prompt
``Q: What day is {offset} days after {input_day}?\\nA:``. The ``premod`` sum
adds the offset to the day number (Monday = 1), and ``output_day`` is that
sum mod 7. Each pair is scored three times, once per variable, so the table
carries one label per variable (Appendix D.2):

* ``label`` (``output_day``): the counterfactual's own answer;
* ``label_offset`` (``offset``): the base day plus the counterfactual offset;
* ``label_input_day`` (``input_day``): the counterfactual day plus the base
  offset.

Each label has its ``*_forms`` column, the space-prefixed and bare spellings
of the day. The paper's records carry their own ``premod``, ``output`` and
``raw_output``; the build refuses a record whose values disagree with the
causal model.

Usage::

    python workflows/scripts/arithmetic_fig15/build_dataset.py --out artifacts/data/arithmetic_fig15            # writes data.json
    python workflows/scripts/arithmetic_fig15/build_dataset.py --out artifacts/data/arithmetic_fig15 --check    # exit 1 if the committed bytes differ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from causalab.causal import CausalModel, CausalTrace, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

#: ``demos/papers/``: this script sits in ``workflows/scripts/arithmetic_fig15/``.
PAPERS = Path(__file__).resolve().parents[3]
DATA = PAPERS / "artifacts" / "data" / "arithmetic_fig15"
SOURCE = DATA / "filtered_dataset.json"
TASK = "weekdays"
GENERATOR = "build_dataset.py"
TABLE = "data.json"

#: Monday is day 1 (Table 1: "We assume that Monday=1 for weekdays").
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
#: The offset words of Table 1, one to fourteen, with their values.
OFFSETS = {
    word: value
    for value, word in enumerate(
        (
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
        ),
        start=1,
    )
}
#: Each label column and the variable whose interchange it is the answer of.
LABELS = {
    "label": "output_day",
    "label_offset": "offset",
    "label_input_day": "input_day",
}


def weekdays_model() -> CausalModel:
    """``output_day = (offset + input_day) mod 7``, rendered as the paper's prompt."""
    day_number = {day: number for number, day in enumerate(DAYS, start=1)}

    @mechanism
    def equations(offset: Dom(list(OFFSETS)), input_day: Dom(list(DAYS))):
        premod = V(OFFSETS[offset] + day_number[input_day], domain=Dom(range(2, 22)))
        output_day = V(DAYS[(premod - 1) % 7], domain=Dom(list(DAYS)))
        raw_input = V(  # noqa: F841
            f"Q: What day is {offset} days after {input_day}?\nA:", domain=Dom(str)
        )
        raw_output = V(" " + output_day, domain=Dom(str))  # noqa: F841
        return output_day

    scoring = ScoringSpec(
        forms={"output_day": {day: (" " + day, day) for day in DAYS}},
        string_mode="exact",
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def load_records(path: Path = SOURCE) -> list[dict[str, Any]]:
    """The wrapped records, after checking that they are the original bytes.

    The source repository writes the file with ``json.dump({"input": [...],
    "counterfactual_inputs": [...]}, indent=2)`` (``src/generate_dataset.py``),
    so the records rebuild it byte for byte.
    """
    wrapper = json.loads(path.read_text())
    records = wrapper["records"]
    original = json.dumps(
        {
            "input": [record["input"] for record in records],
            "counterfactual_inputs": [
                record["counterfactual_inputs"] for record in records
            ],
        },
        indent=2,
    ).encode()
    digest = hashlib.sha256(original).hexdigest()
    if digest != wrapper["source_sha256"]:
        raise ValueError(
            f"{path.name}: the records rebuild bytes with sha256 {digest}, "
            f"not the source's {wrapper['source_sha256']}"
        )
    if len(records) != wrapper["n"]:
        raise ValueError(f"{path.name}: {len(records)} records, n says {wrapper['n']}")
    return records


def trace(model: CausalModel, record: dict[str, Any]) -> CausalTrace:
    """One of the paper's prompts as a trace, checked against its recorded values."""
    t = model.new_trace({"offset": record["offset"], "input_day": record["input"]})
    for ours, theirs in (
        ("premod", "premod"),
        ("output_day", "output"),
        ("raw_input", "raw_input"),
        ("raw_output", "raw_output"),
    ):
        if t[ours] != record[theirs]:
            raise ValueError(
                f"{record['raw_input']!r}: the causal model gives {ours}={t[ours]!r}, "
                f"the record {theirs}={record[theirs]!r}"
            )
    return t


def build(path: Path = SOURCE) -> list[dict[str, Any]]:
    """One row per paper pair, in file order, with the three labels."""
    model = weekdays_model()
    examples = []
    for record in load_records(path):
        counterfactuals = record["counterfactual_inputs"]
        if len(counterfactuals) != 1:
            raise ValueError(
                f"{record['input']['raw_input']!r}: {len(counterfactuals)} "
                "counterfactuals, where a Figure 15 pair has one"
            )
        (counterfactual,) = counterfactuals
        examples.append(
            {
                "input": trace(model, record["input"]),
                "counterfactual_inputs": [trace(model, counterfactual)],
            }
        )
    tables = {
        column: serialize_examples(
            model,
            examples,
            split="all",
            target_variables=[variable],
            task_label=TASK,
            generator=GENERATOR,
            n=len(examples),
        ).rows
        for column, variable in LABELS.items()
    }
    rows = tables["label"]
    for column in ("label_offset", "label_input_day"):
        for row, other in zip(rows, tables[column], strict=True):
            row[column] = other["label"]
            row[f"{column}_forms"] = other["label_forms"]
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write ``data.json``, or with ``--check`` compare it with a fresh build."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory data.json is written to (created if missing), or the path of data.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed bytes differ",
    )
    args = parser.parse_args(argv)
    # A path that names a .json file is the table itself; any other path is
    # the directory it goes in, which need not exist yet.
    path = args.out if args.out.suffix == ".json" else args.out / TABLE
    if path.name != TABLE:
        print(f"{args.out}: not a table this builder writes", file=sys.stderr)
        return 1
    rows = build()
    if args.check:
        fresh = table_bytes(rows)
        if not path.is_file() or path.read_bytes() != fresh:
            print(f"{path}: bytes differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{path}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
        return 0
    digest = write_dataset_table(rows, path)
    print(f"{path}: {len(rows)} rows, digest {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
