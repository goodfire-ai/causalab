"""Clean accuracy of a checkpoint's plain forward on the tables the methods
documents use: the weekdays test split and the IOI table.

A row counts as correct when the argmax next token after ``input`` is the
first token of ``base_answer`` (the ``match`` metric's test), and the same
for ``counterfactual_inputs[0]`` against ``cf_answer``. Both prompts of a
pair are scored because an interchange needs the model right on both sides.

    uv run --no-sync python demos/methods/scripts/clean_accuracy.py \
        --model Qwen/Qwen2.5-1.5B --out results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
TABLES = {
    "weekdays#test": (
        REPO / "causalab/tasks/natural_domains_arithmetic/data/weekdays.json",
        "test",
    ),
    "IOI/default": (REPO / "causalab/tasks/IOI/data/default.json", None),
}


def first_token(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids[0])


@torch.no_grad()
def score(model, tokenizer, prompts, answers, others, device) -> dict:
    """One prompt per forward: no padding, so no position shift to get wrong.

    ``accuracy`` is the full-vocabulary argmax (the ``match`` metric);
    ``two_way`` asks only whether the answer's logit beats the other side of
    the pair's answer (what a ``logit_diff`` metric's sign measures)."""
    hits = 0
    two_way = 0
    single = 0
    for prompt, answer, other in zip(prompts, answers, others):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1]
        ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        other_ids = tokenizer(other, add_special_tokens=False)["input_ids"]
        single += len(ids) == 1
        hits += int(logits.argmax()) == ids[0]
        two_way += bool(logits[ids[0]] > logits[other_ids[0]])
    return {
        "n": len(prompts),
        "accuracy": hits / len(prompts),
        "two_way": two_way / len(prompts),
        "single_token_answers": single / len(prompts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    report: dict = {
        "model": args.model,
        "dtype": args.dtype,
        "device": device,
        "tables": {},
    }
    for name, (path, split) in TABLES.items():
        rows = json.loads(path.read_text())
        if split is not None:
            rows = [r for r in rows if r["split"] == split]
        base = score(
            model,
            tokenizer,
            [r["input"] for r in rows],
            [r["base_answer"] for r in rows],
            [r["cf_answer"] for r in rows],
            device,
        )
        cf = score(
            model,
            tokenizer,
            [r["counterfactual_inputs"][0] for r in rows],
            [r["cf_answer"] for r in rows],
            [r["base_answer"] for r in rows],
            device,
        )
        both = (base["accuracy"] * base["n"] + cf["accuracy"] * cf["n"]) / (
            base["n"] + cf["n"]
        )
        report["tables"][name] = {
            "base": base,
            "counterfactual": cf,
            "all_prompts": both,
        }
        print(name, json.dumps(report["tables"][name]))
    report["elapsed_s"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
