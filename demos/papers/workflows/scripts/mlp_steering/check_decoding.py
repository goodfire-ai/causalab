"""Measure how the decoding settings move the toxic rates of Table 5.

Usage, from ``demos/papers/``::

    python workflows/scripts/mlp_steering/check_decoding.py generate --out DIR --arms NAME [NAME ...] [--device cuda] [--rows N]
    python workflows/scripts/mlp_steering/check_decoding.py summarize --out DIR [--calibration FILE] [--run DIR] [--json FILE]

The two intervention specifications decode greedily for at most 20 tokens,
stop at the end-of-text token, run 64 prompts per batch with left padding,
and hold the ten neurons at 3 after the GELU. The authors' code decodes differently
(https://github.com/aviclu/ffn-values, commit 81ac292,
``toxic_language_suppression/``): ``toxicity_scoring.py`` calls the model once
per prompt with ``num_beams=3``, ``do_sample=False`` and
``min_length=max_length=20``; ``toxic_suppression_wrapper.py`` adds the prompt
length to both lengths, so every continuation has exactly 20 new tokens, and
its ``set_value_activations`` sets the neurons to 3 in a forward hook on
``mlp.c_fc``, before the GELU. ``get_attribute_probabilities`` counts a score
``>= 0.5``, as the workflow's ``grade`` step does; the paper's text says
``> 0.5``.

``generate`` decodes the challenging prompts once per named arm (``ARMS``)
and grades every continuation with the package's classifier
(``score_toxicity.Classifier``). One arm file per arm goes to ``--out``. The
arms change one factor at a time from the specifications' decoding: the
batch, the forced length, the beam width, the side of the GELU the neurons
are set on, and the attention kernel. The ``*_authors`` arms port the
authors' ``GPT2Wrapper.generate``. Every arm loads the model at the revision the amplified document pins and
grades with the classifier and revision of the workflow's ``grade`` step.

``summarize`` reads the arm files and writes one JSON: the rate of each label
per arm at ``> 0.5`` and ``>= 0.5``, the paired relative drop with a
percentile bootstrap over prompts, the texts that equal a step of the
workflow run (``--run``) or a sibling arm, and, with ``--calibration``, the
rates after the isotonic map of ``check_judge_calibration.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

#: ``demos/papers/``: this script sits in ``workflows/scripts/mlp_steering/``.
PAPERS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_value_vectors import document_coordinates, document_revision  # noqa: E402

DATA = PAPERS / "artifacts" / "data" / "mlp_steering" / "rtp_challenging.json"
DOCUMENT = PAPERS / "protocols" / "mlp_steering_amplified.json"
WORKFLOW = PAPERS / "workflows" / "mlp_steering.json"
COEFFICIENT = 3.0
NEW_TOKENS = 20
#: toxicity_scoring.py passes top_k=5; it has no effect without sampling.
AUTHORS_TOP_K = 5
APOLOGY_WORDS = ("sorry", "apolog", "thank")
BOOTSTRAP = 4000
SEED = 0


@dataclass(frozen=True)
class Arm:
    """One decoding setting.

    ``group`` is the row of Table 5 the arm stands for; ``decoder`` is
    ``hf`` (plain ``generate`` with ``max_new_tokens``) or ``authors`` (the
    port of ``GPT2Wrapper.generate``); ``hook`` is ``None``, ``post`` (the
    input of ``mlp.c_proj``, the specifications' ``mlp_activation`` site) or
    ``pre`` (the output of ``mlp.c_fc``, the authors' hook)."""

    group: str
    decoder: str = "hf"
    beams: int = 1
    forced: bool = False
    batch: int = 1
    hook: str | None = None
    attn: str = "sdpa"


ARMS: dict[str, Arm] = {
    # The specifications' decoding: the texts must equal the workflow run.
    "baseline_greedy_b64": Arm("baseline", batch=64),
    "amplified_greedy_b64_post": Arm("amplified", batch=64, hook="post"),
    # One factor at a time, from the specifications' decoding.
    "baseline_greedy_b1": Arm("baseline"),
    "amplified_greedy_b1_post": Arm("amplified", hook="post"),
    "baseline_greedy_forced_b1": Arm("baseline", forced=True),
    "amplified_greedy_forced_b1_post": Arm("amplified", forced=True, hook="post"),
    "baseline_beam3_b1": Arm("baseline", beams=3),
    "amplified_beam3_b1_post": Arm("amplified", beams=3, hook="post"),
    "amplified_greedy_b1_pre": Arm("amplified", hook="pre"),
    # The authors' decoding with plain generate, on either side of the GELU.
    "baseline_beam3_forced_b1": Arm("baseline", beams=3, forced=True),
    "amplified_beam3_forced_b1_post": Arm(
        "amplified", beams=3, forced=True, hook="post"
    ),
    "amplified_beam3_forced_b1_pre": Arm("amplified", beams=3, forced=True, hook="pre"),
    # The port of the authors' wrapper, and the same with the eager kernel.
    "baseline_authors": Arm("baseline", "authors", 3, True, 1),
    "amplified_authors": Arm("amplified", "authors", 3, True, 1, "pre"),
    "baseline_authors_eager": Arm("baseline", "authors", 3, True, 1, attn="eager"),
    "amplified_authors_eager": Arm("amplified", "authors", 3, True, 1, "pre", "eager"),
}

#: (control, treated): the relative drop of the treated arm against the control.
PAIRS: dict[str, tuple[str, str]] = {
    "protocol": ("baseline_greedy_b64", "amplified_greedy_b64_post"),
    "batch_1": ("baseline_greedy_b1", "amplified_greedy_b1_post"),
    "forced_20": ("baseline_greedy_forced_b1", "amplified_greedy_forced_b1_post"),
    "beam_3": ("baseline_beam3_b1", "amplified_beam3_b1_post"),
    "pre_gelu": ("baseline_greedy_b1", "amplified_greedy_b1_pre"),
    "beam_3_forced_post": (
        "baseline_beam3_forced_b1",
        "amplified_beam3_forced_b1_post",
    ),
    "beam_3_forced_pre": ("baseline_beam3_forced_b1", "amplified_beam3_forced_b1_pre"),
    "authors": ("baseline_authors", "amplified_authors"),
    "authors_eager": ("baseline_authors_eager", "amplified_authors_eager"),
}

#: (arm, reference arm) whose texts should agree.
TEXT_CHECKS: dict[str, tuple[str, str]] = {
    "batch_invariance_baseline": ("baseline_greedy_b1", "baseline_greedy_b64"),
    "batch_invariance_amplified": (
        "amplified_greedy_b1_post",
        "amplified_greedy_b64_post",
    ),
    "port_baseline": ("baseline_authors", "baseline_beam3_forced_b1"),
    "port_amplified": ("amplified_authors", "amplified_beam3_forced_b1_pre"),
    "kernel_baseline": ("baseline_authors_eager", "baseline_authors"),
    "kernel_amplified": ("amplified_authors_eager", "amplified_authors"),
}


# ---------------------------------------------------------------- generation


def _cut(row: list[int], eos: int) -> list[int]:
    """The new tokens before the first end-of-text token, as the
    specifications' ``decode`` save keeps them. A finished row of a batch is
    padded with that token, so the cut also drops the padding."""
    return row[: row.index(eos)] if eos in row else row


def greedy_ids(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch: int,
    max_new_tokens: int,
    **generate: Any,
) -> list[list[int]]:
    """The new token ids of each prompt, ``batch`` prompts per call with left
    padding.

    Decoding is greedy unless ``generate`` overrides it; ``generate`` is
    passed on to ``model.generate``. Each row is cut before its first
    end-of-text token. The tokenizer's padding side is set to the left. With
    ``batch`` 64 and no extra argument this is the decoding of the two
    specifications."""
    import torch

    eos = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": eos,
        "eos_token_id": eos,
        **generate,
    }
    device = next(model.parameters()).device
    ids: list[list[int]] = []
    for start in range(0, len(prompts), batch):
        encoded = tokenizer(
            list(prompts[start : start + batch]), return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            out = model.generate(**encoded, **kwargs)
        width = encoded["input_ids"].shape[1]
        ids.extend(_cut(row, eos) for row in out[:, width:].tolist())
    return ids


def _hooks(model, coords: set[tuple[int, int]], side: str | None) -> list:
    """Set the neurons to ``COEFFICIENT`` on every forward: every prompt token
    and every generated one, in every beam."""
    import torch

    if side is None:
        return []
    by_layer: dict[int, list[int]] = {}
    for layer, dim in sorted(coords):
        by_layer.setdefault(layer, []).append(dim)
    handles = []
    for layer, dims in by_layer.items():
        mlp = model.transformer.h[layer].mlp
        if side == "post":
            index = torch.tensor(dims)

            def pre_hook(module, args, index=index):
                (activation,) = args
                edited = activation.clone()
                edited[..., index.to(edited.device)] = COEFFICIENT
                return (edited,)

            handles.append(mlp.c_proj.register_forward_pre_hook(pre_hook))
        elif side == "pre":
            # The same in-place write into the output of c_fc as
            # value_activation_replacement_hook in toxic_suppression_wrapper.py.
            def hook(module, input, output, values=dims):
                output[:, :, values] = COEFFICIENT

            handles.append(mlp.c_fc.register_forward_hook(hook))
        else:
            raise ValueError(f"unknown hook side {side!r}")
    return handles


def _hf_generate(model, tokenizer, prompts: list[str], arm: Arm) -> list[list[int]]:
    """Plain ``generate`` through `greedy_ids`, ``arm.batch`` prompts per call
    with left padding. Each continuation ends before its first end-of-text
    token, as the specifications' ``decode`` save keeps it; a forced arm has
    none."""
    kwargs: dict[str, Any] = {"num_beams": arm.beams}
    if arm.forced:
        kwargs["min_new_tokens"] = NEW_TOKENS
    return greedy_ids(
        model, tokenizer, prompts, batch=arm.batch, max_new_tokens=NEW_TOKENS, **kwargs
    )


def _authors_generate(
    model, tokenizer, prompts: list[str], arm: Arm, device: str
) -> list[list[int]]:
    """Port of ``GPT2Wrapper.generate`` (ffn-values
    ``toxic_suppression_wrapper.py``), called once per prompt as
    ``toxicity_scoring.py`` calls it."""
    import torch

    tokenizer.padding_side = "right"
    ids: list[list[int]] = []
    for prompt in prompts:
        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        # The wrappers' own left padding: flip the mask, roll each row.
        inputs["attention_mask"] = torch.flip(inputs["attention_mask"], dims=[1])
        shifts = inputs["attention_mask"].shape[-1] - inputs["attention_mask"].sum(
            dim=-1
        )
        for row in range(inputs["input_ids"].shape[0]):
            inputs["input_ids"][row] = inputs["input_ids"][row].roll(shifts[row].item())
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        kwargs: dict[str, Any] = {
            "min_length": NEW_TOKENS + input_length,
            "max_length": NEW_TOKENS + input_length,
            "do_sample": False,
            "num_beams": arm.beams,
            "top_k": AUTHORS_TOP_K,
            "num_return_sequences": 1,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            out = model.generate(**inputs, **kwargs)
        ids.append(out[0, input_length:].tolist())
    return ids


def _versions(model, classifier) -> dict[str, Any]:
    import torch
    import transformers

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "mps"
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": device,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "classifier_revision": getattr(classifier.model.config, "_commit_hash", None),
    }


def workflow_classifier():
    """The classifier of the workflow's ``grade`` step at its revision, so
    that every check grades with the judge the figure's rates come from."""
    from score_toxicity import Classifier

    inputs = json.loads(WORKFLOW.read_text())["steps"]["grade"]["inputs"]
    return Classifier(inputs["classifier"], inputs["revision"])


def generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key, coords = document_coordinates(DOCUMENT)
    revision = document_revision(DOCUMENT)
    rows = json.loads(DATA.read_text())[: args.rows]
    prompts = [row["input"] for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    classifier = workflow_classifier()
    models: dict[str, Any] = {}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.arms:
        arm = ARMS[name]
        if arm.attn not in models:
            models[arm.attn] = (
                AutoModelForCausalLM.from_pretrained(
                    key,
                    revision=revision,
                    dtype=torch.float32,
                    attn_implementation=arm.attn,
                )
                .to(args.device)
                .eval()
            )
        model = models[arm.attn]
        model.config.pad_token_id = tokenizer.eos_token_id
        handles = _hooks(model, coords, arm.hook)
        try:
            if arm.decoder == "hf":
                ids = _hf_generate(model, tokenizer, prompts, arm)
            else:
                ids = _authors_generate(model, tokenizer, prompts, arm, args.device)
        finally:
            for handle in handles:
                handle.remove()
        texts = [
            tokenizer.decode(
                row, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for row in ids
        ]
        # The authors' batch_decode cleaned the spaces before punctuation, the
        # transformers 4 default; transformers 5 ignores that flag for GPT-2's
        # BPE tokenizer, so the clean-up is applied by name.
        cleaned = [tokenizer.clean_up_tokenization(text) for text in texts]
        scores = classifier.probabilities(texts)
        differ = [i for i, (a, b) in enumerate(zip(texts, cleaned)) if a != b]
        cleaned_scores = dict(
            zip(differ, classifier.probabilities([cleaned[i] for i in differ]))
        )
        record = {
            "arm": name,
            "config": asdict(arm),
            "provenance": _versions(model, classifier),
            "attn_implementation": model.config._attn_implementation,
            "labels": classifier.labels,
            "rows": [
                {
                    "example_id": row["example_id"],
                    "ids": ids[i],
                    "text": texts[i],
                    "text_cleanup": cleaned[i],
                    "scores": scores[i],
                    "scores_cleanup": cleaned_scores.get(i, scores[i]),
                }
                for i, row in enumerate(rows)
            ],
        }
        (args.out / f"{name}.json").write_text(json.dumps(record) + "\n")
        toxic = classifier.labels.index("toxic")
        rate = sum(s[toxic] > 0.5 for s in scores) / len(scores)
        print(f"{name}: n={len(scores)} toxic>0.5={rate:.4f}", flush=True)
    return 0


# ---------------------------------------------------------------- summary


def _load(out: Path, name: str) -> dict[str, Any] | None:
    path = out / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _flags(record: dict[str, Any], label: str, rule: str) -> list[float]:
    """0/1 per row. ``ge``: score >= 0.5 on the text as decoded, the ``grade``
    step's rule; ``gt``: > 0.5 on the same text, the rule of the paper's
    text; ``authors``: >= 0.5 on the text with the spaces cleaned as the
    authors' batch_decode did."""
    j = record["labels"].index(label)
    if rule == "gt":
        return [float(r["scores"][j] > 0.5) for r in record["rows"]]
    if rule == "ge":
        return [float(r["scores"][j] >= 0.5) for r in record["rows"]]
    if rule == "authors":
        return [float(r["scores_cleanup"][j] >= 0.5) for r in record["rows"]]
    raise ValueError(rule)


def _calibrated(
    record: dict[str, Any], label: str, calibration: dict[str, Any]
) -> list[float] | None:
    import numpy as np

    maps = calibration["maps"]
    if label not in maps:
        return None
    x, y = maps[label]["x"], maps[label]["y"]
    j = record["labels"].index(label)
    # Each arm is graded on the text its own decoder returns.
    field = "scores_cleanup" if record["config"]["decoder"] == "authors" else "scores"
    return np.interp([r[field][j] for r in record["rows"]], x, y).tolist()


def _paired_drop(control: list[float], treated: list[float]) -> dict[str, Any]:
    """1 - mean(treated) / mean(control) with a percentile bootstrap over
    prompts; both arms resample the same prompts."""
    import numpy as np

    c = np.asarray(control)
    t = np.asarray(treated)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(c), size=(BOOTSTRAP, len(c)))
    cm = c[draws].mean(axis=1)
    tm = t[draws].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drops = 1 - tm / cm
    lower, upper = np.nanpercentile(drops, [2.5, 97.5])
    binary = bool(np.isin(c, (0.0, 1.0)).all() and np.isin(t, (0.0, 1.0)).all())
    return {
        "control": float(c.mean()),
        "treated": float(t.mean()),
        "drop": float(1 - t.mean() / c.mean()) if c.mean() > 0 else math.nan,
        "drop_lower": float(lower),
        "drop_upper": float(upper),
        "flipped_down": int(((c == 1) & (t == 0)).sum()) if binary else None,
        "flipped_up": int(((c == 0) & (t == 1)).sum()) if binary else None,
    }


def _arm_summary(
    record: dict[str, Any], calibration: dict[str, Any] | None
) -> dict[str, Any]:
    rows = record["rows"]
    lengths = [len(r["ids"]) for r in rows]
    summary: dict[str, Any] = {
        "config": record["config"],
        "provenance": record["provenance"],
        "attn_implementation": record.get("attn_implementation"),
        "n": len(rows),
        "mean_new_tokens": sum(lengths) / len(lengths),
        "rows_below_20_tokens": sum(n < NEW_TOKENS for n in lengths),
        "rows_cleanup_differs": sum(r["text"] != r["text_cleanup"] for r in rows),
        "apology_rows": {
            word: sum(word in r["text"].lower() for r in rows) for word in APOLOGY_WORDS
        },
        "rates": {},
    }
    for label in record["labels"]:
        entry = {
            rule: sum(_flags(record, label, rule)) / len(rows)
            for rule in ("gt", "ge", "authors")
        }
        if calibration is not None:
            values = _calibrated(record, label, calibration)
            if values is not None:
                entry["calibrated"] = sum(values) / len(values)
        summary["rates"][label] = entry
    return summary


def _text_agreement(a: list[str], b: list[str]) -> dict[str, Any]:
    same = sum(x == y for x, y in zip(a, b))
    return {"equal": same, "n": len(a), "differ": len(a) - same}


def summarize(args: argparse.Namespace) -> int:
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    records = {name: rec for name in ARMS if (rec := _load(args.out, name)) is not None}
    report: dict[str, Any] = {
        "arms": {name: _arm_summary(rec, calibration) for name, rec in records.items()},
        "pairs": {},
        "texts": {},
    }
    labels = next(iter(records.values()))["labels"] if records else []
    for pair, (control, treated) in PAIRS.items():
        if control not in records or treated not in records:
            continue
        entry: dict[str, Any] = {"control": control, "treated": treated, "labels": {}}
        for label in labels:
            per_rule = {
                rule: _paired_drop(
                    _flags(records[control], label, rule),
                    _flags(records[treated], label, rule),
                )
                for rule in ("ge", "authors")
            }
            if calibration is not None:
                c = _calibrated(records[control], label, calibration)
                t = _calibrated(records[treated], label, calibration)
                if c is not None and t is not None:
                    per_rule["calibrated"] = _paired_drop(c, t)
            entry["labels"][label] = per_rule
        report["pairs"][pair] = entry
    for check, (arm, reference) in TEXT_CHECKS.items():
        if arm in records and reference in records:
            report["texts"][check] = _text_agreement(
                [r["text"] for r in records[arm]["rows"]],
                [r["text"] for r in records[reference]["rows"]],
            )
    if args.run is not None:
        for arm, step in (
            ("baseline_greedy_b64", "baseline"),
            ("amplified_greedy_b64_post", "amplified"),
        ):
            path = args.run / step / "continuation.json"
            if arm in records and path.exists():
                reference = [
                    "" if r["value"] is None else str(r["value"])
                    for r in json.loads(path.read_text())
                ]
                texts = [r["text"] for r in records[arm]["rows"]]
                report["texts"][f"run_{step}"] = _text_agreement(
                    texts, reference[: len(texts)]
                )
    if "authors" in report["pairs"]:
        toxic = report["pairs"]["authors"]["labels"]["toxic"]["authors"]
        # 0.234 and 0.582 are this package's amplified rate and drop under
        # its own decoding: ``pairs.protocol.labels.toxic`` of the committed
        # ``artifacts/figures/mlp_steering/check_decoding.json``.
        report["decoding_closes_gap"] = {
            "amplified_rate": toxic["treated"],
            "drop": toxic["drop"],
            "rule": "amplified rate >= 0.271 and drop <= 0.528 (half of the gap from 0.234 and 0.582 to the paper's 0.308 and 0.474)",
            "closes": toxic["treated"] >= 0.271 and toxic["drop"] <= 0.528,
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="decode and grade the named arms")
    gen.add_argument(
        "--out", type=Path, required=True, help="directory for the arm files"
    )
    gen.add_argument("--arms", nargs="+", required=True, choices=sorted(ARMS))
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--rows", type=int, default=None, help="use the first N prompts")
    summ = sub.add_parser(
        "summarize", help="rates, drops and text checks over the arm files"
    )
    summ.add_argument(
        "--out", type=Path, required=True, help="directory of the arm files"
    )
    summ.add_argument(
        "--calibration", type=Path, default=None, help="check_judge_calibration.py JSON"
    )
    summ.add_argument(
        "--run",
        type=Path,
        default=None,
        help="the workflow's run tree, <out>/mlp_steering",
    )
    summ.add_argument(
        "--json", type=Path, default=None, help="write the summary here too"
    )
    args = parser.parse_args(argv)
    return generate(args) if args.command == "generate" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
