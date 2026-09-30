# Cross-model grafting

| Overview | |
|---|---|
| **Question** | Can an activation of [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) replace the answer-slot activation of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on [64 multiple choice pairs](artifacts/data/mcqa/pairs_n64_s0.json)? |
| **Method** | **Cross-model activation graft**: save block 26's answer-slot activation of a source checkpoint on each counterfactual input, swap it into the same component of the target checkpoint on the original input, and read whether the model gives the counterfactual answer. |

## Research question

Every interchange in [05](05_trace.md) to [09](09_components.md) takes its value from the same model it writes into. In [06](06_localize.md), patching block 26's answer slot from the counterfactual input gave the counterfactual answer on 0.922 of 64 pairs. The pretrained checkpoint Qwen2.5-1.5B and the instruction-tuned Qwen2.5-1.5B-Instruct share their architecture, tokenizer and 1536-dimensional stream, and differ in their weights. We use the pairs and clean baselines of 06.

An intervention specification names exactly one `model.key`, and all its reads and writes happen in that model's forward passes. A value that crosses from one checkpoint to another therefore goes through a file: one document saves it, and a second document loads it.

**Q1 — Does a graft through a file reproduce the ordinary interchange when source and target are the same checkpoint?** This is the control. The file adds a step between the read and the write and should change nothing.

**Q2 — Does causalab run the graft when the source is the pretrained checkpoint?** The two workflows differ only in the source model.

## Method

The source document reads block 26's output at the answer slot on each counterfactual prompt and saves the tensor. Its only input role is `base`, because it runs one forward pass; the `field` selects the counterfactual column of the pair table.

```json
{
  "header": {
    "protocol_version": "4",
    "description": "Save the residual stream at block 26's answer slot of Qwen2.5-1.5B on the 64 counterfactual prompts, as the source of a cross-model graft."
  },
  "model": {"key": "Qwen/Qwen2.5-1.5B", "revision": "main", "dtype": "bf16"},
  "data": {"base": {"dataset": "mcqa/pairs_n64_s0", "field": "counterfactual_inputs[0]"}},
  "method": {
    "intervened_models": {
      "original": {"input": "base", "reads": ["acts"]}
    },
    "positions": {"slot": {"index": -1}},
    "sites": {
      "target": {"component": "block_output", "layers": [26]}
    },
    "reads": {"acts": {"site": "target", "pos": "slot"}},
    "save": [{"read": "acts", "model": "original", "file_path": "acts.safetensors"}]
  }
}
```

[protocols/mcqa_source_harvest.json](protocols/mcqa_source_harvest.json) is the source specification. It names the pretrained checkpoint. The control workflow sets `model.key` to the instruction-tuned checkpoint and changes nothing else.

The graft document runs the instruction-tuned model on each original prompt and swaps the saved tensor into the same component. The comments mark what is new:

```json
{
    "header": {
        "protocol_version": "4",
        "description": "Swap a saved answer-slot activation into block 26 of Qwen2.5-1.5B-Instruct on the 64 original prompts. Measure how often the model gives the counterfactual answer, keeps its own answer, and which token it gives."
    },
    "model": {"key": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "main", "dtype": "bf16"}, // The target checkpoint
    "data": {"base": {"dataset": "mcqa/pairs_n64_s0", "field": "input"}}, // No counterfactual role: the value comes from a file
    "method": {
        "intervened_models": {
            "grafted": {"input": "base", "reads": ["logits"], "writes": ["graft"]}
        },
        "positions": {"slot": {"index": -1}},
        "sites": {
            "target": {"component": "block_output", "layers": [26]},
            "lm_head": {"component": "lm_head"}
        },
        "params": {                     // Constants loaded from files; a write can use them as operands
            "v_src": {"file_path": "source/acts.safetensors", "entry": {"slot": "acts"}}
                                        // The source step's tensor, saved under the read's name "acts"
        },
        "reads": {"logits": {"site": "lm_head", "pos": -1}},
        "writes": {
            "graft": {"site": "target", "pos": "slot", "do": {"swap": "v_src"}} // Swap in the loaded tensor, row by row
        },
        "save": [
            {
                "read": "logits",
                "model": "grafted",
                "aggregation": {"kind": "match", "expected": "cf_answer"},
                "file_path": "iia.json"
            },
            {
                "read": "logits",
                "model": "grafted",
                "aggregation": {"kind": "match", "expected": "base_answer"},
                "file_path": "survived.json" // How often the model keeps its original answer
            },
            {
                "read": "logits",
                "model": "grafted",
                "aggregation": {"kind": "top_k", "k": 1, "by": "prob"},
                "file_path": "said.json" // The token the model gives, for answers that match neither
            }
        ]
    }
}
```

[protocols/mcqa_cross_patch.json](protocols/mcqa_cross_patch.json) is a copy of the graft specification above. A write operand is the name of a read, the name of a `params` entry, or a number, so a saved tensor enters as a `params` entry ([the intervention reference, section 2.6](../../docs/intervention_protocol.md#26-params-optional)).

A saved tensor carries an artifact identity that records, among other fields, the model key, revision, dtype, tokenizer and site it came from ([`ARTIFACT_IDENTITY_KEYS`](../../causalab/protocol/identity.py)). Loading it checks that identity against the loading document. In the control the two agree, and the graft is an interchange of 06 with a file in the middle. With the pretrained source, the identities disagree on `model_key`.

## Execution

The control workflow [workflows/mcqa_cross_model.json](workflows/mcqa_cross_model.json) sets the source step's `model.key` to the target checkpoint:

```json
{
  "version": "1",
  "description": "Control arm: harvest the answer-slot activation of Qwen2.5-1.5B-Instruct and graft it back into the same model.",
  "output_dir": "10_cross_model_control",
  "steps": {
    "source": {
      "type": "intervention_protocol",
      "document": "../protocols/mcqa_source_harvest.json",
      "set": {
        "model.key": "Qwen/Qwen2.5-1.5B-Instruct"
      }
    },
    "graft": {
      "type": "intervention_protocol",
      "document": "../protocols/mcqa_cross_patch.json"
    }
  }
}
```

The experimental workflow [workflows/mcqa_cross_model_refused.json](workflows/mcqa_cross_model_refused.json) runs the source document as written, on the pretrained checkpoint:

```json
{
  "version": "1",
  "description": "Experimental arm: harvest the answer-slot activation of Qwen2.5-1.5B and graft it into Qwen2.5-1.5B-Instruct. The graft step refuses at run time.",
  "output_dir": "10_cross_model_refused",
  "steps": {
    "source": {
      "type": "intervention_protocol",
      "document": "../protocols/mcqa_source_harvest.json"
    },
    "graft": {
      "type": "intervention_protocol",
      "document": "../protocols/mcqa_cross_patch.json"
    }
  }
}
```

Both workflows load. `causalab explain` checks a workflow without loading weights and prints its schedule. On the refused workflow:

```bash
uv run causalab explain demos/onboarding_tutorial/workflows/mcqa_cross_model_refused.json \
    --engine pytorch_hooks \
    --data-root demos/onboarding_tutorial/artifacts/data
```

```text
schedule  2 levels
  level 0: source
  level 1: graft
  source: intervention_protocol ../protocols/mcqa_source_harvest.json — 1 point(s), campaign digest a821a932b0080ea0…
  graft: intervention_protocol ../protocols/mcqa_cross_patch.json — 1 point(s), authored digest 77f1ab713246ec78…
```

The `source` step carries a campaign digest, computed over its canonical form with its datasets and loaded files resolved. The `graft` step carries an authored digest, computed over the document as written ([`causalab/workflow/document.py:2234`](../../causalab/workflow/document.py)). Its `file_path` names a file that the source step writes during the run, so the loader cannot check the file's identity before it exists, and the check happens when the graft step opens it. Run both workflows from the repository root:

```bash
uv run causalab run demos/onboarding_tutorial/workflows/mcqa_cross_model.json \
    --engine pytorch_hooks \
    --data-root demos/onboarding_tutorial/artifacts/data \
    --out demos/onboarding_tutorial/artifacts/output \
    --device mps
```

```bash
uv run causalab run demos/onboarding_tutorial/workflows/mcqa_cross_model_refused.json \
    --engine pytorch_hooks \
    --data-root demos/onboarding_tutorial/artifacts/data \
    --out demos/onboarding_tutorial/artifacts/output \
    --device mps
```

Each run is one forward over 64 prompts on the source checkpoint, then one on the target. The run trees under `artifacts/output/10_cross_model_control/` and `artifacts/output/10_cross_model_refused/` were produced on 2026-09-28 on a MacBook Pro (Apple M5 Pro, `--device mps`) in 4 to 5 s each, model loading included. A GPU with 8 GB is enough. The harvested tensors are not committed; the commands above write them.

## Results

### Q1: The control graft reproduces 06's interchange on the same 59 pairs

The control gives the counterfactual answer on 59 of 64 pairs, 0.922 ([graft/iia.json](artifacts/output/10_cross_model_control/graft/iia.json)). 06's interchange at the same component gives 0.922, and the 59 pairs are the same in both runs. This also checks that the saved tensor stays aligned with the prompts row by row. The model keeps its original answer on none of the 64 pairs ([graft/survived.json](artifacts/output/10_cross_model_control/graft/survived.json)).

On the other 5 pairs, the model gives a third token ([graft/said.json](artifacts/output/10_cross_model_control/graft/said.json)):

| pair | original answer | counterfactual answer | model output | probability |
|---|---|---|---|---|
| 17 | `" W"` | `" Q"` | `" red"` | 0.349 |
| 18 | `" I"` | `" J"` | `" A"` | 0.545 |
| 36 | `" G"` | `" U"` | `" A"` | 0.575 |
| 53 | `" J"` | `" Q"` | `" yellow"` | 0.718 |
| 59 | `" B"` | `" Q"` | `" V"` | 0.314 |

Two outputs are color words and three are letters offered in neither prompt. The two `match` reads cannot show these cases, and the `top_k` read does.

### Q2: The run refuses the graft from the pretrained checkpoint

The source step runs and saves its tensor. The graft step refuses before its forward pass:

```text
refused: [V15] params entry 'v_src' (source/acts.safetensors): ArtifactIdentity mismatch on 'model_key' — the document implies 'Qwen/Qwen2.5-1.5B-Instruct' but the bundle was stamped 'Qwen/Qwen2.5-1.5B' (§2.5)
```

The workflow's [events.jsonl](artifacts/output/10_cross_model_refused/events.jsonl) records the completed source step and the failed graft step. The experiment has no IIA for the cross-checkpoint graft.

The two checkpoints have streams of the same width, so the graft would run without a shape error if the check were absent. Fine-tuning changes every weight, and nothing in this experiment shows that a coordinate of block 26's stream means the same thing in both checkpoints. A score from such a graft could not be told apart from a score about the answer symbol. This is an argument about what the refusal protects against. The demo does not measure how different the two streams are.

## Next steps

- Measure how far apart the two checkpoints' activations are. Run the source step on both checkpoints and compare the two tensors, for example by their cosine similarity per row. This puts a size on the difference the identity check guards.

- Follow along [11_attention.md](11_attention.md), which returns to the instruction-tuned model and asks which attention head writes the answer symbol at layer 22.
