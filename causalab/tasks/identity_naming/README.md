# Identity Naming

A factory task for entity → canonical-name lookups. Given an entity (e.g. a musical note `"C#4"`), the model is expected to produce its canonical identifier (e.g. the MIDI number `"61"`). The DAG is the simplest possible:

```
entity ──> result ──> raw_output
   └─────> raw_input
```

The task is a "factory" in the same sense as [`natural_domains_arithmetic/`](../natural_domains_arithmetic/): a single implementation parametrized over domain presets in `config.py::DOMAIN_PRESETS`. Currently one preset is shipped — `pitch_midi` — but the structure makes adding new lookups (country → capital, element → symbol, etc.) a matter of adding a new entry to the presets dict.

## Domain Matrix

| Domain | Entities | Result | Template | Build flag |
|---|---|---|---|---|
| `pitch_midi` | Note names `C2`…`C6` (49 notes, MIDI 36–84, common piano range) | MIDI number as string | `"The MIDI number for {entity} is "` | `--set domain_type=pitch_midi` |

`scripts/build_task_dataset.py` selects the preset with `--set domain_type=pitch_midi`. The [Files](#files) table gives the command that built the shipped table.

## Causal Model

Four variables. Crucially, `result` depends only on `entity` — there's no arithmetic to compute, just a dictionary lookup `entity_to_result[entity]`.

| Variable | Role | Notes |
|---|---|---|
| `entity` | input | E.g. `"C#4"`. Sampled uniformly from the preset's entity list. |
| `result` | computed | `entity_to_result[entity]`, e.g. `"61"`. |
| `raw_input` | computed | `template.format(entity=...)`. The model's prompt. |
| `raw_output` | computed | `output_prefix + result`. |

**Templates are not causal variables.** The preset can carry multiple `templates` to provide phrasing variation (used by `generate_dataset` — see below) but the causal model uses only `templates[0]` for `raw_input`. The prompt phrasing is metadata on the dataset row, not something interventions can target. This keeps the DAG minimal and matches the linguistic intuition that "what's the MIDI number for C#4" and "MIDI for C#4" should yield the same result variable.

### Embeddings

`pitch_midi` uses MIDI-number embeddings for both `entity` and `result` (`_pitch_midi_entity_embed`, `_pitch_midi_result_embed`). This makes the activation/output geometry interpretable as MIDI distance — semitone steps map to unit steps in embedding space. New presets can supply their own `entity_embedding` / `result_embedding`; the fallback is the entity's index in the preset list.

`GET_CYCLIC_VARIABLES` returns the empty set and `GET_PERIODIC_INFO` returns `None` — pitch ranges are linear, not cyclic.

## Counterfactuals

`counterfactuals.py::generate_dataset(model, n, seed)` returns `n` examples of shape `{"input": ..., "counterfactual_inputs": [...]}`.

The generator **cycles through the preset's `templates` list**: example `i` uses `templates[i % len(templates)]`. Both base and counterfactual use the same template per example (so phrasing isn't a confound across the pair) but different examples sample different templates. The base and counterfactual differ in `entity` (independent random samples).

Implementation note: `generate_dataset` builds the trace then calls `trace.intervene("raw_input", template.format(...))` to override `raw_input` with the cycled template — the underlying causal model's `raw_input` mechanism uses only `templates[0]`, so this is the way to get template variation without making `template` a causal variable.

## Token Positions

`create_token_positions(pipeline, template=...)` returns:

| Name | Description |
|---|---|
| `last_token` | The final prompt token (index `-1`). |
| `entity` | The last token spanning the `{entity}` slot. |

Built declaratively via `causalab.tasks.token_positions.build_token_position_factories` — same pattern as `natural_domains_arithmetic`.

## How to Run

No document in this repository runs this task yet. A document names the shipped table as `identity_naming/data/pitch_midi#all` in its `data` block, and runs with:

```bash
uv run causalab run <document> \
    --engine auto \
    --out runs/identity_naming
```

[Running experiments](../../../docs/running_experiments.md) shows how to write, validate and run a document.

## Files

| File | Role |
|---|---|
| `config.py` | `IdentityNamingConfig` dataclass + the `DOMAIN_PRESETS` table |
| `causal_models.py` | `create_causal_model` plus the `GET_*` accessors used by `tasks/loader.py` |
| `counterfactuals.py` | `generate_dataset` (template-cycling) |
| `token_positions.py` | `create_token_positions` |
| `data/pitch_midi.json` | the shipped table: `identity_naming/data/pitch_midi` (256 pairs, `split all`); built with `uv run python scripts/build_task_dataset.py --task identity_naming --set domain_type=pitch_midi --n 256 --seed 0 --split all --target-variable result --out causalab/tasks/identity_naming/data/pitch_midi.json` |
| `metrics.py` | Task-specific accuracy / scoring helpers |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |
