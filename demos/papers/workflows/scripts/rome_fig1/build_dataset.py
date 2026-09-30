"""Build the table of the Figure 1 (e, f, g) replication.

Causal tracing (Meng et al. 2022, Section 2.1) runs a factual prompt three
times: clean, with the subject's token embeddings corrupted by Gaussian noise,
and corrupted with one hidden state restored to its clean value. Figure 1 does
this for one prompt, ``The Space Needle is in downtown`` (answer ``Seattle``),
averaging over ten noise samples. One table follows:

* ``data.json`` holds **ten identical rows** of the prompt, one per noise
  sample. The workflow's ``noise_draw`` step writes one draw of shape
  ``(10, 4, 1600)``, and the tracing documents add row ``i`` of that draw to
  the subject embeddings of table row ``i``. Averaging a metric over the rows
  is averaging over noise samples, which is what the paper's values are. The
  operand only fits a forward that holds all ten rows, so a run needs
  ``--batch-rows`` of at least 10.

The causal model is the fact: a subject and the rest of the prompt (the
prompt with the subject's one occurrence replaced by ``{}``) make the prompt,
a lookup makes the object, and the object's surface forms are the
space-prefixed and bare spellings. The table carries the causalab column
vocabulary (``causalab.tasks.serialize``), so ``causalab validate --data``
reads it like any shipped table. The counterfactual of every row is the row
itself, and nothing here reads it or the ``label`` column.

Usage::

    python workflows/scripts/rome_fig1/build_dataset.py --out artifacts/data/rome_fig1            # writes data.json
    python workflows/scripts/rome_fig1/build_dataset.py --out artifacts/data/rome_fig1 --check    # exit 1 if the committed bytes differ
"""

from __future__ import annotations

import argparse
import hashlib
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

TASK = "rome_fig1"
GENERATOR = "build_dataset.py"

#: The prompt of Figure 1: (subject, template, object). GPT-2 XL's clean
#: argmax is " Seattle", at p = 0.976 in fp32 (panel e, last layer, last token).
FIGURE = ("The Space Needle", "{} is in downtown", "Seattle")
#: One row per noise sample: the paper's ten (Appendix B.1).
NOISE_SAMPLES = 10


def fact_model(rows: list[tuple[str, str, str]]) -> CausalModel:
    """The fact as equations: ``(subject, template)`` looks up the object,
    and the prompt is the template filled with the subject."""
    object_of = {}
    for subject, template, obj in rows:
        object_of[(subject, template)] = obj
    subjects = sorted({r[0] for r in rows})
    templates = sorted({r[1] for r in rows})
    objects = sorted({r[2] for r in rows})

    @mechanism
    def equations(subject: Dom(subjects), template: Dom(templates)):
        object = V(object_of[(subject, template)], domain=Dom(objects))
        raw_input = V(template.format(subject), domain=Dom(str))  # noqa: F841
        raw_output = V(" " + object, domain=Dom(str))  # noqa: F841
        return object

    scoring = ScoringSpec(
        forms={"object": {o: (" " + o, o) for o in objects}}, string_mode="exact"
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def trace(model: CausalModel, fact: tuple[str, str, str]) -> CausalTrace:
    """The trace of one fact: its subject and template set, the rest computed."""
    subject, template, _ = fact
    return model.new_trace({"subject": subject, "template": template})


def build() -> dict[str, Any]:
    """Every table this builder writes, by name, as serialized datasets."""
    model = fact_model([FIGURE])
    rows = [FIGURE] * NOISE_SAMPLES
    examples = [
        {"input": trace(model, fact), "counterfactual_inputs": [trace(model, fact)]}
        for fact in rows
    ]
    dataset = serialize_examples(
        model,
        examples,
        split="all",
        target_variables=["subject", "template"],
        task_label=TASK,
        generator=GENERATOR,
        n=len(examples),
        seed=0,
    )
    return {"data": dataset}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory the tables are written to (a table path names its directory)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if any committed table's bytes differ",
    )
    args = parser.parse_args(argv)
    out = args.out if args.out.is_dir() else args.out.parent
    tables = build()
    if not args.out.is_dir():
        # a table path names one table: check or write that one alone
        tables = {
            name: t for name, t in tables.items() if args.out.name == f"{name}.json"
        }
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
        print(f"{out / f'{name}.json'}: {len(dataset.rows)} rows, digest {digest[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
