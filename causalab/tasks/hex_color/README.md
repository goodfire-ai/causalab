# hex_color

Perceptual colour classification: the model is shown a hue-jittered `#RRGGBB`
code and must name the colour it best matches, choosing from six fixed colour
words inlined MCQA-style in the prompt. The six colours sit on the hue circle,
so the answer variable carries a **periodic** hue embedding.

## Task

Given a prompt of the form

```
Question: Which color name best describes the hex code #BE322E? Choose one of: red, orange, yellow, green, blue, purple.
Answer:
```

the model is expected to emit the colour word — here `" red"`.

The stimulus set is **bundled** as `sources/hex_color.json` (600 stimuli, 100 per
colour × 6), committed with the package like IOI's data files — never read from
`external artifact storage` at runtime. The stimuli were generated for Llama-3.1-8B DAS work
(`.../saes/llama31-8b/tasks/hex_color/das/`), but only the model-agnostic
content (hex + RGB/HSV + colour label) is consumed; no tokenizer/position
fields are imported (causalab recomputes tokens per tokenizer). Each record:
`hex`, `r`/`g`/`b`, `h`/`s`/`v`, `label`.

## `indigo` dropped (7 → 6 colours)

The source dataset defined **seven** colour classes (adding `indigo`, hue 258°).
`indigo` is **excluded at build time** because the golden fixture
(Qwen3-4B-Instruct) cannot perceptually separate it from its neighbours (blue
235°, purple 285°): it labels indigo swatches `"purple"` with ~0.999 confidence.
With every indigo swatch wrong, 7-colour balanced accuracy is at most
6/7 ≈ 0.86, structurally below the 0.9 accuracy floor of the task's golden
check. Dropping indigo makes the task viable on the fixture and,
as a bonus, removes the only multi-token colour word (`"indigo" → ["ind",
"igo"]`), so the six remaining colours are all single-token and the task needs
**no bespoke `full_string_checker`** (see below).

## Causal Model

`CAUSAL_MODEL` is a singleton `CausalModel` with the DAG `hex → color →
raw_output` (and `hex → raw_input`):

| Variable | Role |
|---|---|
| `hex` | Input — a `#RRGGBB` stimulus from the 600 bundled codes |
| `color` | The perceptual label (one of the 6 colour words), looked up from the bundled data. **Target variable.** |
| `raw_input` | Filled prompt string (the 6 choices inlined, fixed order) |
| `raw_output` | `" " + color` — the expected next-token output |

`TARGET_VARIABLE = "color"`.

**Periodic hue embedding.** `color` carries a 1-D embedding — its hue-centre in
degrees (`red=0, orange=30, yellow=58, green=120, blue=235, purple=285`) — with
a `360.0` period (`CausalModel.periods["color"]`), so manifold/geometry analyses
treat the class axis as the cyclic hue circle it is. This mirrors
`natural_domains_arithmetic`'s cyclic `result`.

**Scoring.** Unlike MCQA (which scores an option *letter*), the answer here *is*
the colour word, so the default path already scores "the value". The task's
`ScoringSpec` (`forms={"color": build_output_tokens(COLORS)}`) drives the
probability path (score-token resolution / per-class distributions) **and** the
string grader (`ScoringSpec.grader`, exact stripped match). All six colours are
single-token, so exact match suffices and the task declares **no bespoke
`full_string_checker`** (the earlier `checker.py` existed only to first-token-
tolerate the two-token `indigo`, which is now gone).

## Counterfactuals

| Generator | What changes between input and counterfactual |
|---|---|
| `different_color` | The counterfactual hex has a **different** colour label, so the answer changes. The deconfounding generator `generate_dataset` uses. |
| `same_color_different_hex` | A different hex with the **same** colour label (answer unchanged; isolates "which hex" from "which colour"). |
| `random` | Two independent stimuli — noise-floor reference. |
| `generate_dataset(model, n, seed)` | Loader-convention entry point; builds `n` examples via `different_color` under a fixed seed. |

## Token Positions

`create_token_positions(pipeline, template=...)`:

| Name | Description |
|---|---|
| `hex` | Last token of the `#RRGGBB` stimulus span |
| `last_token` | Final prompt token |

## Files

| File | Role |
|---|---|
| `config.py` | `COLORS` (6), `HUE_CENTERS_DEG`, `HUE_PERIOD`, `PROMPT_TEMPLATE`, `DATA_PATH`, `MAX_NEW_TOKENS`, `MAX_TASK_TOKENS` |
| `causal_models.py` | `CAUSAL_MODEL` (singleton), `TARGET_VARIABLE`, `TEMPLATE`, bundled-data lookups (`HEXES`, `HEX_TO_LABEL`, `HEXES_BY_COLOR`) |
| `counterfactuals.py` | `generate_dataset`, `COUNTERFACTUAL_GENERATORS` (`different_color`, `same_color_different_hex`, `random`) |
| `token_positions.py` | `create_token_positions` (declarative-spec based) |
| `data/default.json` | the shipped table: `hex_color/data/default` (256 pairs, `split all`); built with `uv run python scripts/build_task_dataset.py --task hex_color --n 256 --seed 0 --split all --target-variable color --out causalab/tasks/hex_color/data/default.json` |
| `sources/hex_color.json` | 600 bundled stimuli (model-agnostic; indigo excluded) |
| `summary.ipynb` | CPU-only task walkthrough (no model loaded) |
