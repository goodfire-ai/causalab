"""Build the prompt table for the toxic-language-suppression replication.

Usage, from ``demos/papers/``::

    python workflows/scripts/mlp_steering/make_data.py --out artifacts/data/mlp_steering            # writes rtp_challenging.json
    python workflows/scripts/mlp_steering/make_data.py --out artifacts/data/mlp_steering --check    # exit 1 if the committed bytes differ

``--out`` is the package's data folder or the table itself. The table
``rtp_challenging.json`` holds the ``challenging`` rows of
RealToxicityPrompts (Gehman et al. 2020,
https://huggingface.co/datasets/allenai/real-toxicity-prompts, revision
`REVISION`), the subset Geva et al. (2022, EMNLP, section 6.1) evaluate on.
One row per prompt, in dataset order, with the columns an intervention
specification reads:

- ``example_id``: ``<filename>:<begin>``, the source document and character
  offset RealToxicityPrompts records for the span; a file name alone repeats
  on the subset, the pair is unique (checked here).
- ``input``: the prompt text the model continues.
- ``prompt_toxicity``: the Perspective API toxicity score the dataset ships
  for the prompt itself, kept for inspection; nothing in the package reads it.
- ``split``: ``"all"``, the undivided table (intervention protocol §2.2).

The release flags 1199 rows as ``challenging``, all with distinct prompts.
The authors' loader (``io_utils.load_prompts`` in
https://github.com/aviclu/ffn-values) filters the same flag. The paper counts
1,225 challenging prompts, and where that number comes from is not known.

The recipe reads the dataset through the Hugging Face cache, about 130 MB.
It is not named ``build_dataset.py``, so ``tests/demos/test_papers.py`` does
not run it: the CPU tier cannot fetch the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASET = "allenai/real-toxicity-prompts"
#: The Hub revision the committed table was built from.
REVISION = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"
TABLE = "rtp_challenging.json"


def table_bytes() -> bytes:
    """The table, serialized as the committed file is."""
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="train", revision=REVISION).filter(
        lambda r: r["challenging"]
    )
    table = [
        {
            "example_id": f"{row['filename']}:{row['begin']}",
            "input": row["prompt"]["text"],
            "prompt_toxicity": row["prompt"]["toxicity"],
            "split": "all",
        }
        for row in rows
    ]
    ids = [row["example_id"] for row in table]
    if len(set(ids)) != len(ids):
        raise SystemExit("RealToxicityPrompts <filename>:<begin> labels are not unique")
    return (json.dumps(table, indent=2, ensure_ascii=False) + "\n").encode("utf8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, required=True, help="the data folder or the table"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the committed table instead of writing it",
    )
    args = parser.parse_args(argv)
    target = args.out / TABLE if args.out.is_dir() else args.out
    if target.name != TABLE:
        print(f"{args.out}: not a table this builder writes", file=sys.stderr)
        return 1
    fresh = table_bytes()
    if args.check:
        if not target.is_file() or target.read_bytes() != fresh:
            print(f"{target}: bytes differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{target}: reproduces ({hashlib.sha256(fresh).hexdigest()[:12]})")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fresh)
    print(f"wrote {len(json.loads(fresh))} rows to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
