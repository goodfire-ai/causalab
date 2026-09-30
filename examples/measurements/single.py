"""Measure and profile a tiny offline workflow at one committed revision.

Run from the repository root:
    uv run python -m examples.measurements.single /tmp/causalab-single

Use a new output directory. The example generates a checkpoint without downloads
and installs the selected revision in isolation; wheel-build dependencies must
already be available. Add ``--observations`` for tensor checks or ``--no-profile``
for timing only. Use ``--prepare-only`` to write the inputs without running them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def prepare(
    output: Path,
    *,
    revision: str = "HEAD",
    observations: bool = False,
    profile: bool = True,
) -> tuple[Path, Path]:
    """Create a checkpoint, workflow, and bindings in a fresh directory."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint"
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=5,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=2,
                bos_token_id=0,
                eos_token_id=0,
            )
        )
    model.save_pretrained(checkpoint)
    tokens = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3, "c": 4},
            unk_token="[UNK]",
        )
    )
    tokens.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokens,
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side="left",
    ).save_pretrained(checkpoint)
    data = output / "data"
    data.mkdir()
    (data / "prompts.json").write_text(
        json.dumps(
            [
                {"input": "a b", "split": "all"},
                {"input": "b c", "split": "all"},
            ]
        )
    )
    protocol = {
        "header": {"protocol_version": "4"},
        "model": {"key": str(checkpoint), "revision": "local", "dtype": "fp32"},
        "data": {"base": {"dataset": "prompts", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"head": {"component": "lm_head"}},
            "reads": {
                "logits": {
                    "site": "head",
                    "pos": -1,
                }
            },
            "save": [
                {
                    "read": "logits",
                    "model": "original",
                    "file_path": "logits.safetensors",
                }
            ],
        },
    }
    (output / "inference.json").write_text(json.dumps(protocol, indent=2) + "\n")
    plan = {
        "version": 1,
        "mode": "single",
        "source": {"revision": revision},
        "cases": {
            "resident": {"kind": "workflow", "cold_process": False},
            "cold": {"kind": "workflow", "cold_process": True},
        },
        "seeds": [0],
        "repeats": 1,
        "warmups": 0,
    }
    if observations:
        plan["observations"] = {
            "logits": {
                "step": "inference",
                "file": "logits.safetensors",
                "kind": "tensor",
            }
        }
    if not profile:
        plan["profile"] = False
    workflow = {
        "version": "1",
        "output_dir": "research",
        "steps": {
            "inference": {
                "type": "intervention_protocol",
                "document": "inference.json",
            }
        },
        "measurement": plan,
    }
    document = output / "single.json"
    document.write_text(json.dumps(workflow, indent=2) + "\n")
    bindings = output / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "source": {
                    "repository": str(Path(__file__).resolve().parents[2]),
                    "python": sys.executable,
                },
                "device": "cpu",
                "data_root": str(data),
                "artifacts_root": str(output),
            },
            indent=2,
        )
        + "\n"
    )
    return document, bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--observations", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    document, bindings = prepare(
        args.output,
        revision=args.revision,
        observations=args.observations,
        profile=not args.no_profile,
    )
    if args.prepare_only:
        print(document)
        return
    from causalab.measurement.study.controller import run

    print(run(document, bindings, args.output / "run"))


if __name__ == "__main__":
    main()
