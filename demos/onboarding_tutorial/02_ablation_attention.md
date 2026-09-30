# Ablate attention heads

| Overview | |
|---|---|
| **Question** | Do four attention heads in layers 21 and 22 of [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) produce the answer symbol of a [two-choice question](artifacts/data/mcqa/pair_n1_s0.json)? |
| **Method** | **Zero ablation of attention heads**: prompt the model with the question, replace the output of four heads at the last token with 0, and read the probability of the two offered answer symbols at that position. |

## Research question

The previous tutorial zeroed MLP layers to locate a fact. This tutorial zeroes attention heads to locate a behavior: answering a two-choice question with the correct answer symbol.

We use the multiple-choice question answering task from [Wiegreffe et al. (2024)](https://arxiv.org/abs/2407.15018). The prompt states a color and offers two options under random symbols. The model has to answer with the symbol of the correct option. The task draws the symbols at random for every question, so a model cannot pass by always answering "A" or "B". [04_define.md](04_define.md) builds the tutorial's tables of this task under [artifacts/data/mcqa/](artifacts/data/mcqa/). We take the one row of [artifacts/data/mcqa/pair_n1_s0.json](artifacts/data/mcqa/pair_n1_s0.json):

```
The cup is red. What color is the cup?
M. orange
Z. red
Answer:
```

The language model [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) answers " Z" with probability 0.522 and " M" with 0.086.

**Q1 — Do four attention heads in layers 21 and 22 produce the answer symbol at the last token?** The heads are head 9 and head 11 of layer 21 and head 7 and head 9 of layer 22. They come from a single-head sweep on another question of the same task.

## Method

Wiegreffe et al. find that the answer symbol is produced at the last token position by a few attention heads in middle to late layers. They show this with activation patching and vocabulary projection. We test the finding with zero ablation, as in the previous tutorial: prompt the model with the question above, replace the output of the four attention heads with 0 at the last token, the ":" after "Answer", and check whether the probability of " Z" changes.

If P(" Z") falls and " M" does not gain the mass, the heads produce the answer symbol rather than the answer format. If P(" Z") stays, they do not.

The comments mark what is new since the previous tutorial. Lines without a comment work as they did there.

```json
{
    "header": {
        "protocol_version": "4",
        "description": "Ablation experiment: Zero the output of four attention heads of Qwen2.5-1.5B (head 9 of layers 21 and 22, head 11 of layer 21, head 7 of layer 22) at the last token of a two-choice question, then read the probability of the two offered answer symbols at that position. The question is the one row of mcqa/pair_n1_s0; its correct symbol is Z. The clean model gives P(Z) = 0.522 and P(M) = 0.086."
    },
    "model": {"key": "Qwen/Qwen2.5-1.5B"},
    "data": {
        "base": {                       // The input role every read and write below names
            "dataset": "mcqa/pair_n1_s0", // A table under --data-root: <root>/mcqa/pair_n1_s0.json, one prompt per row
            "field": "input"            // The column that holds the prompt
        }
    },
    "method": {
        "intervened_models": {
            "original": {"input": "base", "reads": ["clean_logits"]},
                                        // The model without any write; "original" is a reserved name
            "ablated": {
                "input": "base",
                "reads": ["ablated_logits"],
                "writes": ["zero_head_9", "zero_head_11", "zero_head_7"]
                                        // All four heads are zeroed in the same forward pass
            }
        },
        "sites": {
            "head_9": {                 // Custom site name
                "component": "attention_premix", // The input of the attention output projection, one slice per head
                "layers": [21, 22],     // The write applies at each listed layer
                "head": 9               // One head index, selected at every layer of the list
            },
            "head_11": {
                "component": "attention_premix",
                "layers": [21],
                "head": 11
            },
            "head_7": {                 // A site holds one head index, so a different index needs its own site
                "component": "attention_premix",
                "layers": [22],
                "head": 7
            },
            "lm_head": {"component": "lm_head"}
        },
        "reads": {
            "clean_logits": {"site": "lm_head", "pos": -1},
            "ablated_logits": {"site": "lm_head", "pos": -1} // The same read on the intervened model named below
        },
        "writes": {
            "zero_head_9": {
                "site": "head_9",
                "pos": -1,
                "do": {"swap": 0}
            },
            "zero_head_11": {
                "site": "head_11",
                "pos": -1,
                "do": {"swap": 0}
            },
            "zero_head_7": {
                "site": "head_7",
                "pos": -1,
                "do": {"swap": 0}
            }
        },
        "save": [
            {
                "read": "clean_logits",
                "model": "original",
                "aggregation": {
                    "kind": "class_probs",
                    "groups": {"Z": [" Z"], "M": [" M"]} // The two symbols offered in the prompt, written as the model emits them: after a space
                },
                "file_path": "02_ablation_attention_clean.json"
            },
            {
                "read": "ablated_logits",
                "model": "ablated",
                "aggregation": {
                    "kind": "class_probs",
                    "groups": {"Z": [" Z"], "M": [" M"]}
                },
                "file_path": "02_ablation_attention_heads_ablated.json"
            }
        ]
    }
}
```

[protocols/02_ablation_attention_spec.json](protocols/02_ablation_attention_spec.json) is a copy of the intervention specification above.

**Head ablation specification.** The core intervention is the same zero write, addressed to single attention heads, with a second read on the model without writes so one run measures the baseline and the ablation. The experiment varies along the head index and layer, how many writes share one forward pass, and which prompts the dataset supplies. Sections 2.2, 2.4 and 2.7 of [the intervention reference](../../docs/intervention_protocol.md) define them.

<details>
<summary>Parameters and the values they accept</summary>

| field | accepts | meaning |
|---|---|---|
| `head` | one integer, `9`; or a sweep, `{"sweep": {"range": [0, 12]}}` | the head slice of `attention_premix`, the per-head input of the output projection. One site holds one head index, at every layer in its `layers`. A sweep over `head` and `layers` together runs one point per (layer, head) pair, 336 here |
| `writes` on an intervened model | a list of write names | every listed write fires in one forward pass. To measure heads one at a time, give each its own intervened model instead |
| `reads` on an intervened model | the names of reads | the model takes the reads it lists. The model without writes is declared like any other intervened model, here as `original`. Reading it beside the ablated model gives the clean baseline from the same run |
| `data.<role>` | `{"dataset": "<ref>", "field": "<column>"}` | a table under `--data-root`, one prompt per row, in place of an inline list. Every read and write names the role |
| `groups` on `class_probs` | `{"<name>": ["<token>", ...]}` | one summed probability per group. A group can hold several spellings of one answer, such as `" Z"` and `" z"` |

</details>

## Execution

Run it with this command:

```bash
uv run causalab run demos/onboarding_tutorial/protocols/02_ablation_attention_spec.json \
    --engine pytorch_hooks \
    --data-root demos/onboarding_tutorial/artifacts/data \
    --out demos/onboarding_tutorial/artifacts/output \
    --device mps
```

`--data-root` is the folder that dataset references resolve against. The specification names `mcqa/pair_n1_s0`, so the run reads `demos/onboarding_tutorial/artifacts/data/mcqa/pair_n1_s0.json`. Without the flag, references resolve against the task data shipped with causalab.

The run is two forwards of a 21-token prompt through a 1.5 B model, one clean and one ablated. The tables under `artifacts/output/` were produced on 2026-09-22 on a MacBook Pro (Apple M5 Pro, `--device mps`) in 6 s, model load included. The CPU takes a few minutes.

## Results

The experiment saves [artifacts/output/02_ablation_attention_clean.json](artifacts/output/02_ablation_attention_clean.json) and [artifacts/output/02_ablation_attention_heads_ablated.json](artifacts/output/02_ablation_attention_heads_ablated.json), one row each, in the format of the previous tutorial:

| model | P(" Z") | P(" M") |
|---|---|---|
| clean | 0.522 | 0.086 |
| four heads zeroed | 0.016 | 0.001 |

### Q1: Yes, the four heads produce the answer symbol

Zeroing four of the 336 heads, at one token position, removes the answer. P(" Z") falls from 0.522 to 0.016. The control is the other offered symbol: P(" M") falls too, from 0.086 to 0.001, so the heads do not merely pick between the two symbols. They produce the answer.

## Next steps

- Run the single-head sweep yourself. Replace the four head sites with one site `{"component": "attention_premix", "layers": {"sweep": {"range": [0, 28]}}, "head": {"sweep": {"range": [0, 12]}}}` and one write on it. The run has 336 points, and the saved table carries the layer and head of each point as columns. Which heads would you pick, and how many do you need to zero together to remove the answer?

- The experiment covers one question. The four heads come from a single-head sweep on a different question of the same task, where we kept the heads with the largest effects in layers 21 and 22. Repeated on this question, that sweep ranks them 2nd, 3rd, 8th and 12th of 336 heads, behind head 1 of layer 26 in first place, and no single head removes the answer on its own. Zeroing an activation also puts the model in a state it never visits during training, which is why Wiegreffe et al. use activation patching, swapping in activations from another input. [11_attention.md](11_attention.md) addresses both limits: it asks which attention head moves the answer symbol into the answer slot with interchange interventions, across the whole dataset instead of one question.

- Follow along [04_define.md](04_define.md), the start of the main tutorial, which builds the tables under `artifacts/data/mcqa/` and asks what a dataset has to look like before an intervention can measure anything.
