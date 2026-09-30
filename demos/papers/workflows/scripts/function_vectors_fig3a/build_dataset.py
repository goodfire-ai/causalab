"""Build the four prompt tables of the function-vector package.

Todd et al. 2024 (arXiv:2310.15213, Section 2.3 and Appendix C) find the
attention heads that carry an in-context task. For each task they average
every head's output at the last token over 100 clean 10-shot prompts, put
that mean into 25 10-shot prompts whose labels are shuffled, and score the
gain in the probability of the correct answer (Figure 3a averages it over
18 tasks). The prompts follow the authors' code
(``src/utils/prompt_utils.py``, ``word_pairs_to_prompt_data`` and
``create_prompt`` at commit fb9eac7): ``<|endoftext|>``, then ``Q: x\\nA: y\\n\\n``
for each demonstration, then ``Q: x_q\\nA:``. The answer scored is the first
token of `` y_q`` (``get_answer_id`` in ``src/utils/eval_utils.py``).

The task records come from ``artifacts/data/function_vectors_fig3a/abstractive_tasks.json``,
the authors' ``dataset_files/abstractive/<task>.json`` files wrapped with
their source URL and the sha256 of the original bytes. Each task is split as
the authors split it (``split_icl_dataset``: scikit-learn ``train_test_split``,
30 % held out at seed 42, then that 30 % split again into test and valid):
demonstrations come from ``train``, queries from ``valid``, and the queries
of the held-out DBM table from ``test``. The prompts themselves are drawn
here with our own seeds; the paper's draws are not published.

Four tables, one split per task (``<table>#<task>`` names one task's rows):

* ``clean.json`` — 100 clean prompts per task, the mean harvest's input.
* ``scan.json`` — 25 prompts per task with the demonstration labels
  shuffled, the input of the per-head scan (Figure 3a).
* ``fit.json`` — 100 shuffled-label prompts per task, the DBM fit's input.
* ``heldout.json`` — 50 shuffled-label prompts per task whose queries come
  from the ``test`` pool, the DBM fit's validation split and its apply input.

The causal model: a task, a block of demonstrations and a query make the
prompt; the task's lookup of the query makes the answer. A shuffled block
carries no information about the task, so the model's prediction depends on
whether the task still reaches the last token. The first token of the
answer is the ``label`` column every document scores, because every metric
scores one token (``docs/intervention_protocol.md``, Token forms).

Usage::

    python workflows/scripts/function_vectors_fig3a/build_dataset.py --out artifacts/data/function_vectors_fig3a
    python workflows/scripts/function_vectors_fig3a/build_dataset.py --out artifacts/data/function_vectors_fig3a --check

The first-token rule needs the GPT-J tokenizer (a few MB, read from the
Hugging Face cache or fetched once).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from causalab.causal import CausalModel, CausalTrace, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import (
    serialize_examples,
    table_bytes,
    write_dataset_table,
)

#: ``demos/papers/``: this script sits in ``workflows/scripts/function_vectors_fig3a/``.
PAPERS = Path(__file__).resolve().parents[3]
DATA = PAPERS / "artifacts" / "data" / "function_vectors_fig3a"
TASKS_FILE = DATA / "abstractive_tasks.json"
TASK = "function_vectors_fig3a"
GENERATOR = "build_dataset.py"

#: The tokenizer of the checkpoint the documents load, at the ``float16``
#: revision the registry row names (causalab/protocol/registry/models.py).
TOKENIZER = "EleutherAI/gpt-j-6b"
TOKENIZER_REVISION = "b71ae8bc86cac13154e03e92b5855203086b722e"

#: Appendix C: 10-shot prompts.
N_SHOTS = 10
#: ``split_icl_dataset``'s defaults as ``compute_average_activations.py`` and
#: ``compute_indirect_effect.py`` call it (``--test_split 0.3``, ``--seed 42``).
TEST_SIZE = 0.3
SPLIT_SEED = 42

#: table -> (prompts per task, labels shuffled, query pool, seed). Appendix C:
#: |P_t| = 100 clean prompts for the mean and |P~_t| = 25 corrupted prompts for
#: the AIE. The fit and held-out sizes are ours: the DBM needs prompts the scan
#: does not score, and the held-out queries come from the pool no other table
#: draws from.
TABLES: dict[str, tuple[int, bool, str, int]] = {
    "clean": (100, False, "valid", 1),
    "scan": (25, True, "valid", 2),
    "fit": (100, True, "valid", 3),
    "heldout": (50, True, "test", 4),
}


def load_tasks() -> dict[str, list[dict[str, str]]]:
    """The wrapped records per task, in the file's task order, each with its
    position in the authors' file as ``id``. A query is named by its ``id``:
    three tasks repeat an input with two outputs (national_parks 26 times,
    product-company 26, landmark-country and park-country once), so the input
    text alone does not fix the answer."""
    wrapped = json.loads(TASKS_FILE.read_text())
    return {
        name: [{"id": str(i), **r} for i, r in enumerate(task["records"])]
        for name, task in wrapped["tasks"].items()
    }


def split_task(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """``split_icl_dataset`` of the authors' ``prompt_utils.py``: train 70 %,
    then the 30 % split into test and valid at the same seed."""
    train, valid = train_test_split(
        records, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    test, valid = train_test_split(valid, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    return {"train": train, "valid": valid, "test": test}


def prompt_text(
    train: list[dict[str, str]], demos: str, labels: str, query: str
) -> str:
    """The authors' template: the demonstrations ``demos`` (indices into the
    task's train pool) with the outputs of ``labels`` (the same indices,
    shuffled or not), then the query."""
    body = "".join(
        f"Q: {train[int(d)]['input']}\nA: {train[int(y)]['output']}\n\n"
        for d, y in zip(demos.split(","), labels.split(","))
    )
    return f"<|endoftext|>{body}Q: {query}\nA:"


def first_token(tokenizer, answer: str) -> str:
    """The first token of `` answer``, decoded: the one token Todd et al. score."""
    ids = tokenizer(" " + answer).input_ids
    return tokenizer.decode(ids[:1])


def draw_prompts(
    tasks: dict[str, list[dict[str, str]]],
    answer_of: dict[str, str],
) -> dict[str, list[tuple[str, str, str, str]]]:
    """Every table's (task, demos, labels, query) rows, seeded per table and
    task: ``demos`` and ``labels`` are comma-joined indices into the task's
    train pool, equal for a clean prompt and permuted for a shuffled one, and
    ``query`` is the query record's ``id``.

    A query whose answer starts with a bare-space token is never drawn (60
    of the 27,695 records, in the three translation tasks and synonym:
    `` ernannt`` is ``[" ", "ern", ...]`` under GPT-J's BPE). Todd et al.
    score that space token; a metric here refuses a blank answer. Such
    records still serve as demonstrations."""
    out: dict[str, list[tuple[str, str, str, str]]] = {name: [] for name in TABLES}
    for t_index, (task, records) in enumerate(tasks.items()):
        pools = split_task(records)
        for name, (n, shuffled, pool, seed) in TABLES.items():
            rng = np.random.default_rng([seed, t_index])
            queries = [q for q in pools[pool] if answer_of[q["output"]].strip()]
            for _ in range(n):
                picked = [
                    int(i)
                    for i in rng.choice(len(pools["train"]), N_SHOTS, replace=False)
                ]
                labels = picked
                if shuffled:
                    labels = [picked[i] for i in rng.permutation(N_SHOTS)]
                query = queries[int(rng.integers(len(queries)))]
                out[name].append(
                    (
                        task,
                        ",".join(map(str, picked)),
                        ",".join(map(str, labels)),
                        query["id"],
                    )
                )
    return out


def icl_model(
    tasks: dict[str, list[dict[str, str]]], answer_of: dict[str, str]
) -> CausalModel:
    """Task, demonstrations and query make the prompt; the task's lookup of
    the query makes the answer, scored by its first token."""
    train_of = {task: split_task(records)["train"] for task, records in tasks.items()}
    record_of = {(task, r["id"]): r for task, records in tasks.items() for r in records}

    def text_of(task: str, demos: str, labels: str, query: str) -> str:
        return prompt_text(
            train_of[task], demos, labels, record_of[(task, query)]["input"]
        )

    answers = sorted(set(answer_of.values()) - {" "})

    @mechanism
    def equations(
        task: Dom(sorted(tasks)), demos: Dom(str), labels: Dom(str), query: Dom(str)
    ):
        output = V(record_of[(task, query)]["output"], domain=Dom(str))
        # Dom(str), not the finite answer set: membership in a finite domain
        # is a linear scan, and the answer set has thousands of entries
        answer = V(answer_of[output], domain=Dom(str))
        raw_input = V(text_of(task, demos, labels, query), domain=Dom(str))  # noqa: F841
        raw_output = V(answer, domain=Dom(str))  # noqa: F841
        return answer

    scoring = ScoringSpec(
        forms={"answer": {a: (a,) for a in answers}}, string_mode="exact"
    )
    return CausalModel(equations, id=TASK, scoring=scoring)


def trace(model: CausalModel, row: tuple[str, str, str, str]) -> CausalTrace:
    task, demos, labels, query = row
    return model.new_trace(
        {"task": task, "demos": demos, "labels": labels, "query": query}
    )


def build(tokenizer=None, only: str | None = None):
    """Every table, or only the table ``only`` names."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER, revision=TOKENIZER_REVISION
        )
    tasks = load_tasks()
    outputs = {r["output"] for records in tasks.values() for r in records}
    answer_of = {o: first_token(tokenizer, o) for o in sorted(outputs)}
    prompts = draw_prompts(tasks, answer_of)
    model = icl_model(tasks, answer_of)
    tables = {}
    for name, rows in prompts.items():
        if only is not None and name != only:
            continue
        examples = [
            {"input": trace(model, row), "counterfactual_inputs": [trace(model, row)]}
            for row in rows
        ]
        tables[name] = serialize_examples(
            model,
            examples,
            split=[row[0] for row in rows],
            target_variables=["task"],
            task_label=TASK,
            generator=GENERATOR,
            n=len(examples),
            seed=TABLES[name][3],
        )
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory the tables are written to, or one table's .json path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if any committed table's bytes differ",
    )
    args = parser.parse_args(argv)
    # the suffix decides, not is_dir(): a fresh rebuild names a directory that
    # does not exist yet, and write_dataset_table creates it
    only = args.out.stem if args.out.suffix == ".json" else None
    if only is not None and only not in TABLES:
        print(f"{args.out}: not a table this builder writes", file=sys.stderr)
        return 1
    out = args.out.parent if only is not None else args.out
    tables = build(only=only)
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
