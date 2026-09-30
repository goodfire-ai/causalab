"""Offline CPU no-change control exercising collectors and a comparison workflow.

Quick start from the repository root::

    uv run python -m examples.measurements.smoke /tmp/causalab-measurement-smoke

Uses a tiny randomly initialized model and synthetic data.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def run(output: Path) -> Path:
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from causalab.cli import main as cli
    from causalab.measurement import Operation, collect
    from causalab.measurement.collection import file_hash
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
    from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
    from causalab.neural.shared.featurizers import Subspace
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
    from causalab.workflow.document import load_workflow
    from causalab.measurement.workflow import workflow_operation

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()
    (data / "prompts.json").write_text(
        json.dumps(
            [
                {"input": "alpha beta", "split": "train"},
                {"input": "beta gamma", "split": "train"},
            ]
        )
    )
    tokens = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4},
            unk_token="[UNK]",
        )
    )
    tokens.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokens,
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side="left",
    )
    config = GPT2Config(
        vocab_size=5,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config).eval().requires_grad_(False)
    model.set_attn_implementation("eager")
    bundle = ModelBundle.from_model(
        model,
        tokenizer,
        key="measurement/tiny-gpt2",
        revision="synthetic-seed-0",
        dtype="fp32",
    )
    protocol = {
        "header": {"protocol_version": "4"},
        "model": {"key": bundle.key, "revision": bundle.revision, "dtype": "fp32"},
        "data": {"base": {"dataset": "prompts#train", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "head", "pos": -1}},
            "save": [
                {
                    "read": "logits",
                    "model": "original",
                    "file_path": "logits.safetensors",
                }
            ],
        },
    }
    (output / "observe.json").write_text(json.dumps(protocol))
    workflow = {
        "version": "1",
        "output_dir": "observed",
        "steps": {
            "observe": {"type": "intervention_protocol", "document": "observe.json"}
        },
    }
    document = output / "workflow.json"
    document.write_text(json.dumps(workflow))
    env = ResolutionEnv(
        datasets=FileDatasets(root=data), artifacts=FileArtifacts(root=output)
    )
    loaded = load_workflow(document, env)

    @contextmanager
    def engines(seed):
        yield PytorchHooksEngine(bundle=bundle)

    def observe(result):
        values = load_file(str(result.run_root / "observe/logits.safetensors"))
        # Row order is frozen by the synthetic table. Keys include the saved entry.
        return {f"fixed_prompt_order/last_real/{k}": v for k, v in values.items()}

    prepare_workflow = workflow_operation(lambda seed: loaded, env, engines, observe)

    @contextmanager
    def prepare_subspace(seed, directory):
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(2, 8, generator=generator)
        source = torch.randn(2, 8, generator=generator)
        stage = Subspace(8, 2, "cayley", seed=seed)

        def apply():
            with torch.no_grad():
                feature, _ = stage.featurize(source)
                _, error = stage.featurize(base)
                return stage.inverse(feature, error)

        yield Operation(
            apply,
            lambda result: {
                f"example_{i}/residual": row for i, row in enumerate(result)
            },
        )

    comparisons = {}
    for name, prepare, identity, scope in (
        (
            "subspace_apply",
            prepare_subspace,
            hashlib.sha256(b"synthetic-2x8-k2-v1").hexdigest(),
            "operation microbenchmark: prepared subspace interchange",
        ),
        (
            "workflow",
            prepare_workflow,
            loaded.digest,
            "end-to-end workflow wall: resident synthetic model; required outputs included",
        ),
    ):
        paths = {}
        for arm in ("before", "after"):
            paths[arm] = collect(
                prepare,
                output / f"{name}_{arm}",
                case=name,
                input_identity=identity,
                scope=scope,
                reset_policy="fresh operation/engine; same frozen synthetic inputs/model",
                seeds=[0, 1],
                repeats=2,
                warmups=1,
                profile=True,
                context={
                    "control": "same implementation in both arms",
                    "synthetic": True,
                },
            )
        comparisons[name] = {
            "type": "script",
            "script": {"module": "causalab.measurement.analysis.compare"},
            "inputs": {
                "before": {"path": str(paths["before"])},
                "after": {"path": str(paths["after"])},
                "before_sha256": file_hash(paths["before"]),
                "after_sha256": file_hash(paths["after"]),
            },
            "outputs": {"summary": "summary.json", "report": "report.html"},
        }
    comparison = output / "compare.json"
    comparison.write_text(
        json.dumps(
            {"version": "1", "output_dir": "comparison", "steps": comparisons}, indent=2
        )
    )
    code = cli(["run", str(comparison), "--engine", "auto", "--out", str(output)])
    if code:
        raise RuntimeError(f"comparison workflow exited {code}")
    return output / "comparison"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(run(parser.parse_args().output))
