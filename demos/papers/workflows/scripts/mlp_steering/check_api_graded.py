"""Test Table 5's second finding: safety neurons beat random non-toxic ones.

Usage, from ``demos/papers/``::

    python workflows/scripts/mlp_steering/check_api_graded.py generate --out DIR [--device cuda] [--rows N]
    python workflows/scripts/mlp_steering/check_api_graded.py summarize --out DIR [--run DIR] [--json FILE]

Geva et al. (2022, section 6.1) find that "inducing sub-updates that promote
'safety' related concepts is more effective than promoting generally
non-toxic sub-updates". Table 5 rests this on the row "10 API Graded", whose
Toxicity drop is 10%, against 47% for the ten neurons of Table 8.
Appendix A.4 gives the recipe: concatenate the top-30 tokens of each value
vector's projection to the vocabulary, grade the text with the Perspective
toxicity score, and sample 10 random vectors whose score is below 0.1. The
ten vectors are not published, and the 2022 Perspective model is gone.

This check repeats the recipe with the package's judge in place of
Perspective, and draws the ten vectors `DRAWS` times, since the paper's own
draw cannot be recovered:

- ``generate`` projects every value vector of gpt2-medium (24 layers of 4096)
  as the authors' ``GPT2Wrapper.project_value_to_vocab`` does (the final
  layer norm, then the unembedding), joins the top-30 tokens with spaces
  (the paper does not say how it joins them), and grades each text with the
  ``grade`` step's classifier. Draw ``k`` takes `VECTORS` vectors from those
  whose ``toxic`` probability is below `BELOW`, with seed ``k``. It then
  continues the challenging prompts with the specifications' decoding
  (greedy, at most 20 tokens, 64 prompts per call) once without neurons,
  once with the ten Table 8 neurons, and once per draw. Each set of neurons
  is held at 3 after the GELU, as the amplified document holds them, and
  every continuation is graded with the same classifier. The selection and
  one file per arm go to ``--out``.
- ``summarize`` reads those files and writes one JSON: the pool, the vectors
  of each draw, each arm's rate per label (a score of 0.5 and above, the
  ``grade`` step's rule), each arm's relative drop from the unchanged model,
  and the ten neurons' drop minus each draw's, with a paired bootstrap over
  prompts of the size and seed of the workflow's ``rates`` step. One matrix
  of resampled prompts serves every arm and label, so a drop and a
  difference of drops are paired. With ``--run``, the texts of the unchanged
  model and of the ten neurons are compared with the workflow run's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

#: ``demos/papers/``: this script sits in ``workflows/scripts/mlp_steering/``.
PAPERS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_decoding import _hooks, _versions, greedy_ids, workflow_classifier  # noqa: E402
from check_value_vectors import document_coordinates, document_revision  # noqa: E402
from score_toxicity import DEFAULT_THRESHOLD  # noqa: E402
from table5_figure import paper_values  # noqa: E402

DATA = PAPERS / "artifacts" / "data" / "mlp_steering" / "rtp_challenging.json"
DOCUMENT = PAPERS / "protocols" / "mlp_steering_amplified.json"
WORKFLOW = PAPERS / "workflows" / "mlp_steering.json"
OUT = PAPERS / "artifacts" / "figures" / "mlp_steering" / "check_api_graded.json"
#: Appendix A.4: the top-30 tokens of each projection, a toxicity score below
#: 0.1, and 10 vectors.
TOP = 30
BELOW = 0.1
VECTORS = 10
#: Draws of the ten vectors, seeds 0 to DRAWS - 1. The paper reports one
#: draw; ten show the spread of the recipe.
DRAWS = 10
#: The specifications' decoding.
NEW_TOKENS = 20
BATCH = 64
#: Table 5's row "10 API Graded", Toxicity column, as a fraction (printed
#: 52.7%, page 8 of arXiv 2203.14680v3).
API_GRADED_TOXICITY = 0.527


def vocab_text(tokens: Sequence[str]) -> str:
    """The graded text of one value vector: its top tokens, stripped of the
    leading-space marker and joined by single spaces."""
    return " ".join(token.strip() for token in tokens if token.strip())


def draw_vectors(
    scores: dict[tuple[int, int], float], below: float, size: int, seed: int
) -> list[tuple[int, int]]:
    """``size`` distinct (layer, neuron) pairs drawn uniformly, with seed
    ``seed``, from those whose score is below ``below``, sorted.

    Raises:
        ValueError: fewer than ``size`` pairs score below ``below``.
    """
    import numpy as np

    pool = sorted(key for key, score in scores.items() if score < below)
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} vectors score below {below}")
    picks = np.random.default_rng(seed).choice(len(pool), size=size, replace=False)
    return sorted(pool[int(i)] for i in picks)


def _projections(model, tokenizer, device: str) -> dict[tuple[int, int], list[str]]:
    """The top-`TOP` tokens of every value vector, projected as the authors'
    wrapper projects one: ``lm_head(ln_f(v))``."""
    import torch

    unembed = model.lm_head.weight.detach().to(device)
    ln_f = model.transformer.ln_f
    out: dict[tuple[int, int], list[str]] = {}
    with torch.no_grad():
        for layer, block in enumerate(model.transformer.h):
            values = block.mlp.c_proj.weight.detach().to(device)
            logits = ln_f(values) @ unembed.T
            top = torch.topk(logits, TOP, dim=-1).indices.cpu().tolist()
            for neuron, ids in enumerate(top):
                out[(layer, neuron)] = [tokenizer.decode([t]) for t in ids]
    return out


def _vector(layer: int, neuron: int, texts: dict, scores: dict) -> dict[str, Any]:
    return {
        "layer": layer,
        "neuron": neuron,
        "text": texts[(layer, neuron)],
        "toxic": scores[(layer, neuron)],
    }


def generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key, manual = document_coordinates(DOCUMENT)
    revision = document_revision(DOCUMENT)
    rows = json.loads(DATA.read_text())[: args.rows]
    prompts = [row["input"] for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            key, revision=revision, dtype=torch.float32
        )
        .to(args.device)
        .eval()
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    classifier = workflow_classifier()
    toxic = classifier.labels.index("toxic")
    args.out.mkdir(parents=True, exist_ok=True)

    projections = _projections(model, tokenizer, args.device)
    keys = sorted(projections)
    texts = {k: vocab_text(projections[k]) for k in keys}
    probabilities = classifier.probabilities([texts[k] for k in keys])
    scores = {k: probs[toxic] for k, probs in zip(keys, probabilities)}
    draws = {
        f"draw_{seed}": draw_vectors(scores, BELOW, VECTORS, seed)
        for seed in range(DRAWS)
    }
    selection = {
        "top": TOP,
        "below": BELOW,
        "vectors": VECTORS,
        "join": "stripped tokens joined by single spaces",
        "projection": "lm_head(ln_f(v)), as GPT2Wrapper.project_value_to_vocab",
        "scores": [[layer, neuron, scores[(layer, neuron)]] for layer, neuron in keys],
        "manual_pick": [_vector(*k, texts, scores) for k in sorted(manual)],
        "draws": {
            name: [_vector(*k, texts, scores) for k in picks]
            for name, picks in draws.items()
        },
    }
    (args.out / "selection.json").write_text(json.dumps(selection) + "\n")
    pool = sum(score < BELOW for score in scores.values())
    print(f"{pool} of {len(scores)} vectors score below {BELOW}", flush=True)

    arms: dict[str, set[tuple[int, int]] | None] = {
        "baseline": None,
        "manual_pick": set(manual),
        **{name: set(picks) for name, picks in draws.items()},
    }
    for name, coords in arms.items():
        handles = _hooks(model, coords or set(), None if coords is None else "post")
        try:
            ids = greedy_ids(
                model, tokenizer, prompts, batch=BATCH, max_new_tokens=NEW_TOKENS
            )
        finally:
            for handle in handles:
                handle.remove()
        continuations = [
            tokenizer.decode(
                row, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for row in ids
        ]
        graded = classifier.probabilities(continuations)
        record = {
            "arm": name,
            "neurons": None if coords is None else [list(c) for c in sorted(coords)],
            "provenance": _versions(model, classifier),
            "labels": classifier.labels,
            "rows": [
                {
                    "example_id": row["example_id"],
                    "ids": ids[i],
                    "text": continuations[i],
                    "scores": graded[i],
                }
                for i, row in enumerate(rows)
            ],
        }
        (args.out / f"{name}.json").write_text(json.dumps(record) + "\n")
        rate = sum(s[toxic] >= DEFAULT_THRESHOLD for s in graded) / len(graded)
        print(f"{name}: n={len(graded)} toxic={rate:.4f}", flush=True)
    return 0


def flag_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The arms' scores as a grade-step table, flagged by the step's rule
    (a score of 0.5 and above)."""
    rows = []
    for arm, record in records.items():
        for row in record["rows"]:
            for label, score in zip(record["labels"], row["scores"]):
                rows.append(
                    {
                        "example_id": row["example_id"],
                        "arm": arm,
                        "attribute": label,
                        "value": float(score >= DEFAULT_THRESHOLD),
                    }
                )
    return rows


def flag_vectors(
    rows: Sequence[dict[str, Any]], arms: Sequence[str]
) -> tuple[list[str], dict[tuple[str, str], Any]]:
    """The labels in first-seen order and one 0/1 vector per (arm, label),
    aligned on the first arm's sorted prompt ids.

    Raises:
        ValueError: an arm is missing, or two arms flag different prompts.
    """
    import numpy as np

    by_key: dict[tuple[str, str], dict[str, float]] = {}
    attributes: list[str] = []
    for row in rows:
        arm, attribute = str(row["arm"]), str(row["attribute"])
        if attribute not in attributes:
            attributes.append(attribute)
        by_key.setdefault((arm, attribute), {})[str(row["example_id"])] = float(
            row["value"]
        )
    present = {arm for arm, _ in by_key}
    missing = [arm for arm in arms if arm not in present]
    if missing:
        raise ValueError(f"no flags for arms {missing}")
    ids = sorted(by_key[(arms[0], attributes[0])])
    vectors: dict[tuple[str, str], Any] = {}
    for arm in arms:
        for attribute in attributes:
            values = by_key.get((arm, attribute), {})
            if sorted(values) != ids:
                raise ValueError(
                    f"arm {arm!r} flags other prompts for {attribute!r} "
                    f"than arm {arms[0]!r}"
                )
            vectors[(arm, attribute)] = np.array([values[i] for i in ids])
    return attributes, vectors


def resamples(n: int, repetitions: int, seed: int) -> Any:
    """``(repetitions, n)`` prompt indices drawn with replacement."""
    import numpy as np

    return np.random.default_rng(seed).integers(0, n, size=(repetitions, n))


def _drop(reference: Any, treated: Any) -> float:
    base = float(reference.mean())
    return 1.0 - float(treated.mean()) / base if base > 0 else float("nan")


def _drops(reference: Any, treated: Any, draws: Any) -> Any:
    """The drop in every resample; ``nan`` where the reference rate is 0."""
    import numpy as np

    base = reference[draws].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(base > 0, 1.0 - treated[draws].mean(axis=1) / base, np.nan)


def _interval(samples: Any) -> tuple[float | None, float | None]:
    """The 95% percentile interval of the resamples that have a drop."""
    import numpy as np

    kept = samples[~np.isnan(samples)]
    if kept.size == 0:
        return None, None
    lower, upper = np.percentile(kept, [2.5, 97.5])
    return float(lower), float(upper)


def drop_rows(
    vectors: dict[tuple[str, str], Any],
    attributes: Sequence[str],
    reference: str,
    arms: Sequence[str],
    draws: Any,
) -> list[dict[str, Any]]:
    """One row per (arm, label): the relative drop ``1 - rate / rate_ref``
    from ``reference``, its interval over ``draws``, both rates and the
    prompts that change flag."""
    out = []
    for arm in arms:
        for attribute in attributes:
            c, t = vectors[(reference, attribute)], vectors[(arm, attribute)]
            lower, upper = _interval(_drops(c, t, draws))
            out.append(
                {
                    "arm": arm,
                    "attribute": attribute,
                    "reference": reference,
                    "value": _drop(c, t),
                    "lower": lower,
                    "upper": upper,
                    "reference_rate": float(c.mean()),
                    "rate": float(t.mean()),
                    "n": int(c.size),
                    "down": int(((c == 1) & (t == 0)).sum()),
                    "up": int(((c == 0) & (t == 1)).sum()),
                }
            )
    return out


def difference_rows(
    vectors: dict[tuple[str, str], Any],
    attributes: Sequence[str],
    reference: str,
    pairs: Sequence[Sequence[str]],
    draws: Any,
) -> list[dict[str, Any]]:
    """One row per (pair, label): the first arm's drop minus the second's,
    with the interval over the same ``draws``."""
    out = []
    for first, second in pairs:
        for attribute in attributes:
            c = vectors[(reference, attribute)]
            a, b = vectors[(first, attribute)], vectors[(second, attribute)]
            lower, upper = _interval(_drops(c, a, draws) - _drops(c, b, draws))
            out.append(
                {
                    "first": first,
                    "second": second,
                    "attribute": attribute,
                    "reference": reference,
                    "value": _drop(c, a) - _drop(c, b),
                    "lower": lower,
                    "upper": upper,
                    "n": int(c.size),
                }
            )
    return out


def _text_agreement(a: list[str], b: list[str]) -> dict[str, int]:
    same = sum(x == y for x, y in zip(a, b))
    return {"equal": same, "n": len(a), "differ": len(a) - same}


def summarize(args: argparse.Namespace) -> int:
    paper = paper_values()
    selection = json.loads((args.out / "selection.json").read_text())
    random = list(selection["draws"])
    names = ["baseline", "manual_pick", *random]
    records = {
        name: json.loads((args.out / f"{name}.json").read_text()) for name in names
    }
    rates = json.loads(WORKFLOW.read_text())["steps"]["rates"]
    bootstrap = rates["reduction"]["uncertainty"]
    repetitions, seed = int(bootstrap["repetitions"]), int(bootstrap["seed"])
    attributes, vectors = flag_vectors(flag_rows(records), names)
    n = next(iter(vectors.values())).size
    draws = resamples(n, repetitions, seed)
    drops = drop_rows(vectors, attributes, "baseline", names[1:], draws)
    differences = difference_rows(
        vectors,
        attributes,
        "baseline",
        [["manual_pick", name] for name in random],
        draws,
    )
    toxic_drops = [r for r in drops if r["attribute"] == "toxic" and r["arm"] in random]
    toxic_differences = [r for r in differences if r["attribute"] == "toxic"]
    (manual_toxic,) = [
        r for r in drops if r["arm"] == "manual_pick" and r["attribute"] == "toxic"
    ]
    scores = [entry[2] for entry in selection["scores"]]
    report: dict[str, Any] = {
        "selection": {
            **{key: selection[key] for key in ("top", "below", "vectors", "join")},
            "projection": selection["projection"],
            "value_vectors": len(scores),
            "pool": sum(score < selection["below"] for score in scores),
            "manual_pick": selection["manual_pick"],
            "draws": selection["draws"],
        },
        "provenance": records["baseline"]["provenance"],
        "threshold": DEFAULT_THRESHOLD,
        "bootstrap": {"repetitions": repetitions, "seed": seed},
        "paper_drops": {
            "Toxicity": {
                "manual_pick": 1.0 - paper["amplified"] / paper["baseline"],
                "api_graded": 1.0 - API_GRADED_TOXICITY / paper["baseline"],
            }
        },
        "rates": {
            arm: {label: float(vectors[(arm, label)].mean()) for label in attributes}
            for arm in names
        },
        "drops": drops,
        "differences": differences,
        "toxic": {
            "manual_pick_drop": manual_toxic["value"],
            "draw_drops": {r["arm"]: r["value"] for r in toxic_drops},
            "draw_drop_range": [
                min(r["value"] for r in toxic_drops),
                max(r["value"] for r in toxic_drops),
            ],
            "draws": len(toxic_differences),
            "draws_manual_pick_ahead": sum(r["lower"] > 0 for r in toxic_differences),
        },
        "texts": {},
    }
    if args.run is not None:
        for arm, step in (("baseline", "baseline"), ("manual_pick", "amplified")):
            path = args.run / step / "continuation.json"
            if path.exists():
                reference = [
                    "" if r["value"] is None else str(r["value"])
                    for r in json.loads(path.read_text())
                ]
                texts = [r["text"] for r in records[arm]["rows"]]
                report["texts"][f"run_{step}"] = _text_agreement(
                    texts, reference[: len(texts)]
                )
    text = json.dumps(report, indent=2) + "\n"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(text)
    print(json.dumps({"toxic": report["toxic"], "texts": report["texts"]}, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="select the draws, decode and grade")
    gen.add_argument(
        "--out", type=Path, required=True, help="directory for the arm files"
    )
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--rows", type=int, default=None, help="use the first N prompts")
    summ = sub.add_parser("summarize", help="rates, drops and differences")
    summ.add_argument(
        "--out", type=Path, required=True, help="directory of the arm files"
    )
    summ.add_argument(
        "--run",
        type=Path,
        default=None,
        help="the workflow's run tree, <out>/mlp_steering",
    )
    summ.add_argument("--json", type=Path, default=OUT, help="the summary JSON")
    args = parser.parse_args(argv)
    return generate(args) if args.command == "generate" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
