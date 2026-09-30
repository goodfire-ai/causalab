# MCQA (Multiple-Choice Question Answering)

A two-choice multiple-choice question. The prompt names an object and a color, then offers two lettered choices; the model is expected to emit the letter whose color matches.

```
The shoe is blue. What color is the shoe?
A. red
B. blue
Answer:
```

The expected continuation is `" B"`. The choice-symbol mapping is randomized per example so the model can't shortcut by always answering the same letter.

## Causal Model

`positional_causal_model` (also exported as `CAUSAL_MODEL`) is a `CausalModel` with the following variables:

| Variable | Role |
|---|---|
| `template`, `object`, `color` | Inputs — drawn from `TEMPLATES`, `OBJECTS`, `COLORS`. |
| `choices[0]`, `choices[1]` | Inputs — color labels offered as the two answers. |
| `symbols[0]`, `symbols[1]` | Inputs — uppercase letters (`A`…`Z`) labeling each choice. |
| `raw_input` | Filled prompt. |
| `answer_position` | `0` or `1` — index of the choice whose color matches `color`. |
| `answer` | The symbol letter at `answer_position` — what the model should produce. |
| `raw_output` | `" " + answer`. |

`NUM_CHOICES` is fixed at 2; `OBJECTS`, `COLORS`, and `ALPHABET` are module constants in `causal_models.py`.

The module's `TARGET_VARIABLE` is `answer_position`. `scripts/build_task_dataset.py` uses it unless `--target-variable` names a different variable. The [onboarding tutorial](../../../demos/onboarding_tutorial/04_define.md) builds its tables with `--target-variable answer`.

### Loader hooks

`causal_models.py` exports the standard hooks consumed by `tasks/loader.py`:

- `CAUSAL_MODEL` — singleton instance.
- `TARGET_VARIABLE = "answer_position"`.
- `TEMPLATE = TEMPLATES[0]`.
- `PREDICT_CLASS(ex, generated)` — maps a model's generated string back to an `answer_position` index by matching against the example's `symbols[{i}]` values.
- `CLASS_TOKEN_IDS(ex, tokenizer)` — returns the token ID of `" {symbol_i}"` for each choice position; needed because the symbol-to-token mapping is per-example.

## Counterfactuals

`counterfactuals.py` defines four generators, all returning `{"input": ..., "counterfactual_inputs": [...]}`:

| Generator | What changes |
|---|---|
| `different_symbol` | Both `symbols[0]` and `symbols[1]` are resampled; `choices[0]`, `choices[1]`, and `color` are kept. **This is the default** — `generate_dataset` calls it. It cleanly deconfounds `answer_position` (unchanged) from `answer` (changed). |
| `same_symbol_different_position` | The two `(choice, symbol)` pairs are swapped at their positions, flipping `answer_position` while keeping the symbol set fixed. |
| `random_counterfactual` | Two independent answerable samples — every input variable may differ. |
| `sample_answerable_question` | Underlying primitive that guarantees the correct color appears in the choices (so `answer_position` is always defined). |

`generate_dataset(model, n, seed)` is the loader-convention entry point; `model` is accepted but unused (MCQA samples directly from the module-level causal model).

## Token Positions

`create_token_positions(pipeline, template=None)` returns `TokenPosition` objects for:

| Name | Description |
|---|---|
| `symbols[0]`, `symbols[1]` | The token spanning each choice's symbol letter. |
| `symbol0_period`, `symbol1_period` | The token immediately after each symbol (the `.`). |
| `correct_symbol` | Dispatches per-example to whichever of `symbols[0]` / `symbols[1]` is the correct answer. |
| `correct_symbol_period` | Period after the correct symbol. |
| `last_token` | Final prompt token (the `:` after `Answer`). |

`correct_symbol` and `correct_symbol_period` are dynamic — they read the input sample's `color` / `choices[{i}]` to decide which symbol is correct, then return that position. This lets analyses target "the right answer" without hard-coding a position.

All positions are built declaratively via `causalab.tasks.token_positions.build_token_position_factories` from a spec dict, so no model-specific tokenization tables are needed.

## How to Run

The [onboarding tutorial](../../../demos/onboarding_tutorial/README.md) runs its experiments on this task. Its locate scan, [06_localize](../../../demos/onboarding_tutorial/06_localize.md), runs from the repository root with:

```bash
uv run causalab run demos/onboarding_tutorial/workflows/mcqa_locate.json \
    --engine auto \
    --data-root demos/onboarding_tutorial/artifacts/data \
    --out runs/mcqa_locate \
    --device cuda
```

The run writes its outputs under `--out`. A document of your own names the shipped table as `MCQA/data/default#train` or `MCQA/data/default#test`. [Running experiments](../../../docs/running_experiments.md) shows how to write, validate and run a document.

## Files

| File | Role |
|---|---|
| `causal_models.py` | `positional_causal_model`, `OBJECTS`, `COLORS`, `TEMPLATES`, `ALPHABET`, plus the loader hooks (`CAUSAL_MODEL`, `TARGET_VARIABLE`, `PREDICT_CLASS`, `CLASS_TOKEN_IDS`) |
| `counterfactuals.py` | `sample_answerable_question`, `different_symbol`, `same_symbol_different_position`, `random_counterfactual`, `generate_dataset` |
| `token_positions.py` | `create_token_positions` |
| `data/default.json` | the shipped table: `MCQA/data/default#train` (128) / `#test` (64) — the onboarding demo's table, same bytes; built with `uv run python scripts/build_split_dataset.py --task MCQA --seed 1 --fraction train=128 --fraction test=64 --max-inputs 192 --target-variable answer_position --out causalab/tasks/MCQA/data/default.json` |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |
