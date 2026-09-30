# Ablate MLP layers

| Overview | |
|---|---|
| **Question** | Which MLP layers of [GPT2-XL](https://huggingface.co/openai-community/gpt2-xl) hold the fact that the Space Needle is in Seattle? |
| **Method** | **Zero ablation of MLP layers**: prompt the model with the fact, replace the output of MLP layers 13 to 15 at the last subject token with 0, and read the probability of " Seattle" at the last position. |

## Research question

Let's start the tutorials with a simple ablation experiment: zeroing the output of selected MLP layers during a forward pass. Ablation experiments give evidence for where a specific mechanism is located in a neural network.

We investigate which components of the language model [GPT2-XL](https://huggingface.co/openai-community/gpt2-xl) contain factual knowledge. For instance, GPT2-XL knows the location of the Space Needle:

```
The Space Needle is in downtown
```

The model predicts " Seattle" with probability 0.976.

GPT-2 splits the prompt into seven tokens. The intervention below addresses index 3, the last token of the subject, and the read-out addresses index 6, the last token:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| token | `The` | ` Space` | ` Need` | `le` | ` is` | ` in` | ` downtown` |
| role | | subject | subject | **last subject token** | | | **read-out** |

**Q1 — Do MLP layers 13 to 15 hold the fact at the last subject token?** [Meng et al.](https://rome.baulab.info/) locate factual recall in the MLP layers of the middle of the network, at the last token of the subject. We test three of those layers on one fact.

## Method

Meng et al. find that ablating just a few layers is enough to perturb this knowledge. They perform ablation by inserting corrupted activations from a separate forward pass. For simplicity, we verify their findings by zero-ablating hidden layers (which is arguably less interpretable than perturbing inputs) by following these steps: prompt the model with "The Space Needle is in downtown", replace the output of MLP layers 13 to 15 with 0 at index 3, and check whether the output probability of " Seattle" changes.

If the probability falls well below 0.976, the three layers carry the fact at the last subject token. If it stays, they do not.

To run this experiment, we write it as an intervention specification:

```json
{
    "header": {                         // Metadata describing this specification
        "protocol_version": "4",        // Version of the specification format
        "description": "Ablation experiment: Zero the MLP output of gpt2-xl at layers 13, 14 and 15 at token 3, 'le', the last token of the subject (GPT-2 splits the prompt as The | Space | Need | le | is | in | downtown), then read P(Seattle) at the last position. The clean model gives P(Seattle) = 0.976."
                                        // Summary for a human researcher
    },
    "model": {                          // Details of the target neural network
        "key": "openai-community/gpt2-xl" // Huggingface name of the model
    },
    "data": {                           // Input prompts and optional labels
        "inputs": ["The Space Needle is in downtown"] // Inline prompts. A bare list is the input role "base"; later tutorials name a dataset instead
    },
    "method": {                         // The intervention definition
        "intervened_models": {          // The model with references to operations
            "ablated": {                // Addressed to the intervened model named below
                                        // Custom name
                "input": "base",        // The input role: the inline prompt above
                                        // The input role the model runs on
                "reads": ["logits"],
                "writes": ["zero"]      // Attaching write operations
            }
        },
        "sites": {                      // Named model components that reads and writes address
            "mlp": {                    // Custom site name
                "component": "mlp_output", // Reference to a pre-defined mapping
                "layers": [13, 14, 15]  // List of one or more layers
            },
            "lm_head": {                // Custom site name
                "component": "lm_head"  // Reference to a pre-defined mapping
            }
        },
        "reads": {                      // Read operations
            "logits": {                 // Custom name
                "site": "lm_head",      // Addressed to a site named in method/sites
                "pos": -1               // Addressed to the last token position
            }
        },
        "writes": {                     // Write operations
            "zero": {                   // Custom name
                "site": "mlp",          // Addressed to a site named in method/sites
                "pos": 3,               // Addressed to token index 3, the last subject token
                "do": {"swap": 0}       // The write operation, can reference scalars, reads, or loaded tensors
            }
        },
        "save": [                       // One output table per entry
            {
                "read": "logits",       // The read it reduces
                "model": "ablated",
                "aggregation": {        // The reduction of the read to a number
                    "kind": "class_probs", // Softmax mass of each named group of tokens
                    "groups": {"Seattle": [" Seattle"]} // The token as the model emits it: after a space
                },
                "file_path": "01_ablation_MLP_result.json"
            }
        ]
    }
}
```

[protocols/01_ablation_MLP_spec.json](protocols/01_ablation_MLP_spec.json) is a copy of the intervention specification above.

**Zero ablation specification.** The core intervention is one write: it replaces the activation of a model component with 0 at one token position, and a read at the last position measures the effect. The experiment varies along four axes, which component, which layers, which token position, and what replaces the activation. Sections 2.3, 2.4 and 2.8 of [the intervention reference](../../docs/intervention_protocol.md) define them.

<details>
<summary>Parameters and the values they accept</summary>

| field | accepts | meaning |
|---|---|---|
| `component` | `embeddings`, `block_input`, `attention_output`, `attention_premix`, `block_mid`, `mlp_input`, `mlp_output`, `block_output`, `lm_head` | the tensor the site addresses. `block_*` are the residual stream before, inside, and after a block; `attention_*` and `mlp_*` are the two sublayers' inputs and outputs; `lm_head` is the vocabulary projection |
| `layers` | a list of integers, `[13, 14, 15]`; or a sweep, `{"sweep": {"range": [0, 48]}}` | the depths the site spans. A list applies one write at every listed layer in one forward pass. A sweep runs one point per value, so the output table has one row per layer |
| `pos` | `-1`; `{"index": n}`; `{"span": [a, b]}`; `"all"`; `{"variable": "x"}` | the token positions the read or write addresses. Negative indices count from the end. `variable` needs a dataset that records where each variable sits, see tutorial 04 |
| `do` | `{"swap": v}`; `{"add_scaled": {"op": v, "alpha": a}}`; `{"lerp": {"op": v, "alpha": a}}` | `swap` replaces the activation with `v`. `add_scaled` adds `a` times `v`. `lerp` moves the activation toward `v` by the fraction `a`, so `op: 0, alpha: 0.5` halves it. `v` is a number, the name of a read (tutorial 05), or the name of a loaded tensor (tutorial 12) |

</details>

## Execution

Run it with this command:

```bash
uv run causalab run demos/onboarding_tutorial/protocols/01_ablation_MLP_spec.json \
    --engine pytorch_hooks \
    --out demos/onboarding_tutorial/artifacts/output \
    --device mps
```

`--engine` determines which library is used as an interpretability engine to execute the model interventions. Multiple libraries for intervening on model internal activations exist publicly, such as [NNsight](https://github.com/ndif-team/nnsight), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [interp-engine](https://github.com/decoderesearch/interp-engine), and others. Causalab currently maintains its own set of pytorch hooks, and NNsight as an engine.

`--out` determines the save path for output artifacts.

`--device` selects the PyTorch device: `cpu`, `cuda`, or `mps` on an Apple GPU. Without the flag the run uses the CPU.

The run is one forward of a seven-token prompt through a 1.5 B model, so it fits any GPU with 8 GB of memory. The table under `artifacts/output/` was produced on 2026-09-22 on a MacBook Pro (Apple M5 Pro, `--device mps`) in 8 s, model load included.

## Results

The experiment saves [artifacts/output/01_ablation_MLP_result.json](artifacts/output/01_ablation_MLP_result.json) with one row:

```json
[
  {
    "example_id": "0",                          // the row index
    "metric": "p_seattle",                      // metric name
    "value": "{\"Seattle\": 0.13062365353107452}",
                                                // metric value: the probability of each group
    "unit": "fraction",                         // metric unit
    "estimand_version": "class_probs/v1",       // metric version
    "eligible": true                            // the metric could be computed for this row
  }
]
```

| model | P(" Seattle") |
|---|---|
| clean | 0.976 |
| MLP layers 13 to 15 zeroed at index 3 | 0.131 |

### Q1: Yes, zeroing the three layers removes most of the fact

P(" Seattle") falls from 0.976 to 0.131. The clean value is the baseline: the same prompt without the write. The experiment has no further control. It does not show that other layers leave the fact alone, or that the drop is specific to the subject token. The first Next step adds that control.

## Next steps

- Is this ablation effect unique to the layers 13 to 15? Try improving ablation results by adapting the layers you're ablating. Can you achieve a comparable or lower probability for predicting Seattle by ablating other sets of MLPs, or even a single MLP? Move `"pos": 3` to another index to see whether the subject token matters.

- Is the fact localized in [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) as well? Change the model key, note that Qwen tokenizes the prompt as The | Space | Needle | is | in | downtown, so the last subject token moves to position 2, and that the model has 28 layers instead of 48. Which MLP layers hold the fact there?

- Check out the implementation of the original ROME experiment at [demos/papers/rome_fig1.md](../papers/rome_fig1.md).

- Follow along [02_ablation_attention.md](02_ablation_attention.md), where we ablate attention heads to localize the question answering mechanism.
