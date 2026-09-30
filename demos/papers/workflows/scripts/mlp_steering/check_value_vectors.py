"""Check that the document's neuron coordinates are the paper's value vectors.

Usage, from ``demos/papers/``::

    python workflows/scripts/mlp_steering/check_value_vectors.py [--json FILE]

Geva et al. (2022, EMNLP, Table 8) name the ten manually picked value
vectors as v^l_i with layers and neuron indices counted from 1 and print the
top-10 tokens of each vector's projection to the vocabulary, E v^l_i. The
intervention specification counts both from 0. This script reads the
coordinates out of ``protocols/mlp_steering_amplified.json``, projects the
matching rows of gpt2-medium's down-projection through the token embedding,
and compares the top-10 tokens with Table 8 on stripped text (the paper's
rendering drops the leading-space marker). Stripping makes some tokens
repeat (`` safe`` and ``safe``), so a listed token counts only as often as the
top 10 holds it. At the document's coordinates every vector must reproduce
all 10 of its listed tokens. At the paper's own
numbers read as counted from 0, the unshifted coordinates, no vector may
reproduce any, so the shift is what makes the match. Exits 1 when either
test fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

#: ``demos/papers/``: this script sits in ``workflows/scripts/mlp_steering/``.
PAPERS = Path(__file__).resolve().parents[3]
DOCUMENT = PAPERS / "protocols" / "mlp_steering_amplified.json"

#: Table 8, as printed: paper coordinates (from 1) and top-10 tokens.
TABLE_8: dict[tuple[int, int], list[str]] = {
    (14, 1853): [
        "transparency",
        "disclosure",
        "clearer",
        "parency",
        "iquette",
        "humility",
        "modesty",
        "disclosures",
        "accountability",
        "safer",
    ],
    (15, 73): [
        "respectful",
        "honorable",
        "healthy",
        "decent",
        "fair",
        "erning",
        "neutral",
        "peacefully",
        "respected",
        "reconc",
    ],
    (15, 1395): [
        "safe",
        "neither",
        "safer",
        "course",
        "safety",
        "safe",
        "Safe",
        "apologize",
        "Compact",
        "cart",
    ],
    (16, 216): [
        "refere",
        "Messages",
        "promises",
        "Relations",
        "accept",
        "acceptance",
        "Accept",
        "assertions",
        "persistence",
        "warn",
    ],
    (17, 462): [
        "should",
        "should",
        "MUST",
        "ought",
        "wisely",
        "Should",
        "SHOULD",
        "safely",
        "shouldn",
        "urgently",
    ],
    (17, 3209): [
        "peaceful",
        "stable",
        "healthy",
        "calm",
        "trustworthy",
        "impartial",
        "stability",
        "credibility",
        "respected",
        "peace",
    ],
    (17, 4061): [
        "Proper",
        "proper",
        "moder",
        "properly",
        "wisely",
        "decency",
        "correct",
        "corrected",
        "restraint",
        "professionalism",
    ],
    (18, 2921): [
        "thank",
        "THANK",
        "thanks",
        "thank",
        "Thank",
        "apologies",
        "Thank",
        "thanks",
        "Thanks",
        "apologise",
    ],
    (19, 1891): [
        "thanks",
        "thank",
        "Thanks",
        "thanks",
        "THANK",
        "Thanks",
        "Thank",
        "Thank",
        "thank",
        "congratulations",
    ],
    (23, 3770): [
        "free",
        "fit",
        "legal",
        "und",
        "Free",
        "leg",
        "pless",
        "sound",
        "qualified",
        "Free",
    ],
}
#: At the document's coordinates, every listed token must be in the top 10.
MIN_HITS = 10
#: At the unshifted coordinates, a vector may share at most this many.
MAX_UNSHIFTED_HITS = 0


def document_coordinates(document: Path) -> tuple[str, set[tuple[int, int]]]:
    """The (layer, neuron) pairs the document writes, counted from 0, and its model key."""
    spec = json.loads(document.read_text())
    sites = spec["method"]["sites"]
    coords: set[tuple[int, int]] = set()
    for write in spec["method"]["writes"].values():
        (layer,) = sites[write["site"]]["layers"]
        coords.update((layer, dim) for dim in write["dims"])
    return spec["model"]["key"], coords


def document_revision(document: Path) -> str | None:
    """The checkpoint revision the document pins, if any."""
    return json.loads(document.read_text())["model"].get("revision")


def hits(tokens: list[str], found: list[str]) -> int:
    """How many of Table 8's tokens are among ``found``, the stripped
    top-10 tokens of a projection, each found token matching one listed
    token at most (a multiset intersection)."""
    return sum((Counter(tokens) & Counter(found)).values())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", type=Path, default=None, help="write the counts here too"
    )
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key, coords = document_coordinates(DOCUMENT)
    expected = {(layer - 1, dim - 1) for layer, dim in TABLE_8}
    if coords != expected:
        print(
            f"document writes {sorted(coords)}\nTable 8 minus one is {sorted(expected)}"
        )
        return 1
    revision = document_revision(DOCUMENT)
    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        key, revision=revision, dtype=torch.float32
    )
    embedding = model.transformer.wte.weight.detach()

    def top_tokens(layer: int, dim: int) -> list[str]:
        value_vector = model.transformer.h[layer].mlp.c_proj.weight.detach()[dim]
        top = torch.topk(embedding @ value_vector, 10).indices.tolist()
        return [tokenizer.decode([t]).strip() for t in top]

    rows = []
    failures = 0
    for (layer, dim), tokens in sorted(TABLE_8.items()):
        shifted = hits(tokens, top_tokens(layer - 1, dim - 1))
        unshifted = hits(tokens, top_tokens(layer, dim))
        ok = shifted >= MIN_HITS and unshifted <= MAX_UNSHIFTED_HITS
        failures += not ok
        rows.append(
            {
                "table8": f"v{layer}_{dim}",
                "document": [layer - 1, dim - 1],
                "shifted_hits": shifted,
                "unshifted_hits": unshifted,
            }
        )
        print(
            f"{'ok ' if ok else 'BAD'} v{layer}_{dim} -> document ({layer - 1}, {dim - 1}): "
            f"{shifted}/10 of Table 8 in the top 10; unshifted ({layer}, {dim}): {unshifted}/10"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": key,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "min_hits": MIN_HITS,
            "max_unshifted_hits": MAX_UNSHIFTED_HITS,
            "vectors": rows,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
