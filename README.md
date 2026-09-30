# Causal analysis of neural networks

Causalab helps you test hypotheses about how neural networks solve tasks. Define
a causal model, align its variables with network components, and compare the
effects of interventions in both models.

- **Experiments as JSON.** You write an experiment as a JSON document, and
  the backend runs it. The same document states the model, the data, the
  interventions and the measurements, so a result can be reproduced from it.
- **Hypothesis testing.** Compare causal hypotheses on the same pairs of
  inputs and find which one predicts the network's behavior under
  intervention. See [hypothesis analysis](docs/hypothesis_analysis.md).
- **Interpretability by gradient descent.** Train masks with
  Desiderata-Based Masking (DBM) and rotations with distributed alignment
  search (DAS) to find the units and subspaces that carry a variable. See the
  [method guides](docs/methods/README.md).

## Quick start

The [onboarding tutorial](demos/onboarding_tutorial/01_ablation_MLP.md) builds
one experiment step by step. Its first pages run on a laptop.

Install causalab from a clone:

```bash
git clone https://github.com/goodfire-ai/causalab.git
cd causalab
uv sync
```

The weight reader includes a Rust extension. Install `rustup` on `PATH` before
running `uv sync`; `rust-toolchain.toml` selects the compiler version. See
[standalone installation](docs/standalone_install.md) to build and share a wheel.
For Jupyter and interactive causal graphs, use `uv sync --extra notebook`.

Run an intervention protocol on your CPU. This document swaps one activation
of a tiny random Llama and saves two metric tables. The model has random
weights, so the result checks the install and nothing more. The
[onboarding tutorial](demos/onboarding_tutorial/01_ablation_MLP.md) explains
each part of a protocol.

```json
{
  "header": {
    "protocol_version": "4",
    "description": "The smallest real application of the interchange method: one swap at the last position of layer 0 in a tiny random Llama, on CPU, in fp32. Used by the standalone-install CI job to prove a hash-locked install can run an intervention specification end to end. The revision is a commit SHA, not a branch: this file is a CI assertion, and an upstream re-upload must not be able to move it."
  },
  "model": {                                    // the network, pinned to one commit
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "fp32"
  },
  "data": {                                     // each example is a pair of prompts
    "base": {"dataset": "weekdays/train", "field": "input"},
    "counterfactual": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
  },
  "method": {
    "intervened_models": {                      // two forward passes per example
      "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
      "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]}
    },
    "sites": {                                  // where in the network to read or write
      "target": {"component": "block_output", "layers": [0]},
      "lm_head": {"component": "lm_head"}
    },
    "reads": {"v_cf": {"site": "target", "pos": -1}, "logits": {"site": "lm_head", "pos": -1}},
    "writes": {                                 // put the counterfactual activation into the base run
      "patch": {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}
    },
    "save": [                                   // one metric table per entry
      {
        "read": "logits",
        "model": "patched",
        "aggregation": {"kind": "match", "expected": "cf_answer"},
        "file_path": "iia.json"
      },
      {
        "read": "logits",
        "model": "patched",
        "aggregation": {"kind": "logit_diff", "a": "cf_answer", "b": "base_answer"},
        "file_path": "logit_diff.json"
      }
    ]
  }
}
```

The file is
[`demos/methods/protocols/minimal_cpu.json`](demos/methods/protocols/minimal_cpu.json).
Show its plan, then run it. Both commands take a few seconds:

```bash
uv run causalab explain demos/methods/protocols/minimal_cpu.json \
    --engine auto \
    --data-root tests/protocol/fixtures/data
uv run causalab run demos/methods/protocols/minimal_cpu.json \
    --engine auto \
    --data-root tests/protocol/fixtures/data \
    --artifacts-root tests/protocol/fixtures/artifacts \
    --out runs/minimal_cpu \
    --device cpu
# saved iia.json -> runs/minimal_cpu/iia.json
# saved logit_diff.json -> runs/minimal_cpu/logit_diff.json
# cells 2 / 2 eligible
```

## Compute

| Where | Use it for | Guide |
|---|---|---|
| Local | CPU or Apple-silicon runs with `--device cpu` or `--device mps`: the quick start, the first tutorials, small models | [Experiment guide](docs/running_experiments.md) |
| Cluster | One or more GPUs with `--device cuda`, sharded runs, and SLURM jobs | [Running at scale](docs/running_experiments.md#7-running-at-scale), [model parallelism](docs/model_parallelism.md) |
| NDIF | Remote runs on large models; the guide is a placeholder | [NDIF](docs/ndif.md) |

Optional Linux GPU kernels use the `flash-attn` and `flash-linear-attention`
extras; see [attention backends](docs/attention_backends.md).

## Documentation

| Tab | Use it to | Start here |
|---|---|---|
| Demos | Learn what causalab is, with no experiment in mind | [Onboarding tutorial](demos/onboarding_tutorial/01_ablation_MLP.md), then the other [tutorials](demos/README.md) |
| Demos | Run a specific experiment | [How-to guides](docs/howtos.md): method templates, field rules and the [experiment guide](docs/running_experiments.md) |
| Demos | Define a causal model for a task | [Causal model guide](docs/causal-models.md), then the [task models](demos/causal_models/README.md) and [`causalab/causal/`](causalab/causal/README.md) |
| Development | Change causalab | [Architecture guide](docs/CODEBASE.md), then the [testing guide](docs/TESTS.md) |
| Development | Write a tutorial, a paper replication or a guide | Style guides: [general writing](docs/STYLE_GUIDE.md), [tutorials](docs/demos.md), [paper replications](docs/paper_replications.md) |
| API reference | Write causal models, tasks and analysis scripts | The API reference tab of the site, rendered from the docstrings of the packages you call |
