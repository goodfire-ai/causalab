# Natural Domains Arithmetic

A factory task for arithmetic over a finite domain. The same causal DAG `(entity, number) → result → raw_output` is parametrized across six domains (weekdays, months, hours, age, alphabet, integer) so a single implementation drives every variant — replacing the deprecated standalone `weekdays/` and `months/` tasks.

A prompt looks like `"Q: What day is three days after Thursday?\nA:"` and the model is expected to produce `" Sunday"`. Swap the domain and the same shape of question covers month arithmetic, hour-on-a-clock arithmetic, integer addition, age arithmetic, and alphabet shifts.

## Domain Matrix

Six presets are bundled in `config.py::DOMAIN_PRESETS`. `scripts/build_task_dataset.py` and `scripts/build_split_dataset.py` select one with `--set domain_type=<domain>`.

| Domain | Cyclic? | Modulus | Entity vocab | Number vocab | Template | Build flag |
|---|---|---|---|---|---|---|
| `weekdays` | yes | 7 | `Monday`…`Sunday` | `one`…`seven` | `Q: What day is {number} days after {entity}?\nA:` | `--set domain_type=weekdays` |
| `months` | yes | 12 | `January`…`December` | `one`…`twelve` | `Q: What month is {number} months after {entity}?\nA:` | `--set domain_type=months` |
| `hours` | yes | 24 | `1`…`24` | `one`…`twenty-four` | `Q: What hour comes {number} hours after {entity} on a clock?\nA: ` | `--set domain_type=hours` |
| `integer` | no | — | word-form `one`…`fifteen` | word-form `one`…`nine` | `Q: What is {number} added to {entity}?\nA:` | `--set domain_type=integer` |
| `age` | no | — | `1`…`99` | `1`…`10` | `Alice is {entity} years old. Bob is {number} years older than Alice. Q: How old is Bob?\nA: Bob is ` | `--set domain_type=age` |
| `alphabet` | no | — | `A`…`Y` | `one`…`three` | `The letter {number} after {entity} in the alphabet is the letter` | `--set domain_type=alphabet` |

For non-cyclic domains an `input_filter` (set in `causal_models.py`) drops `(entity, number)` pairs whose result would fall outside `result_entities` (e.g. `alphabet: Z + two` is excluded). Cyclic domains wrap with the modulus and require no filtering.

## Causal Model

Five variables. The DAG is identical for every domain; only the `compute_result` function differs:

```
number ──┐
         ├──> result ──> raw_output
entity ──┤
         └──> raw_input
```

| Variable | Role | Notes |
|---|---|---|
| `entity` | input — the starting domain element | E.g. `"Thursday"`, `"July"`, `"M"` |
| `number` | input — the offset to add | Word-form for cyclic domains, digit for `age` |
| `result` | computed answer | Cyclic: `(entity_idx + number) % modulus`. Non-cyclic: a custom `compute_result` from the preset. With `number_groups`, becomes a tuple `(entity_result, group_index)`. |
| `raw_input` | rendered prompt string | `template.format(entity=…, number=…)` |
| `raw_output` | expected continuation | `output_prefix + result` |

A multi-template variant is supported: pass `template=[...]` to `NaturalDomainConfig` and a `template` input variable is added so each example samples one of the templates. `token_positions.py` builds a per-template dispatcher automatically.

The model is built by `create_causal_model(config: NaturalDomainConfig)` in `causal_models.py`. A random-word baseline (entities replaced by random English words, cyclic mod arithmetic) is available via `create_random_causal_model` for sanity-checking that performance is driven by domain knowledge rather than surface form.

### Embeddings & periods

`create_causal_model` registers value embeddings used by downstream geometry analyses:

- `entity` and `result` are embedded by their domain index (or by `entity_embedding` from the preset, e.g. integer-valued for `age`).
- `number` is embedded by its integer value via `number_to_int`.
- For cyclic domains, `periods["entity"]` and `periods["result"]` are set to `modulus` so isometry analyses know the geometry is circular. `number` is also marked cyclic for `weekdays` (where `number_is_cyclic=True`).

## Token Positions

`token_positions.py::create_token_positions(pipeline, template=...)` returns:

| Name | Description |
|---|---|
| `last_token` | The final prompt token (index `-1`). |
| `entity` | The last token spanning the `{entity}` slot. |
| `number` | The last token spanning the `{number}` slot. |

Each position is built by `causalab.tasks.token_positions.build_token_position_factories` from a declarative spec (no per-model hardcoding). For multi-template configs, pass `templates=[...]` instead and the returned `TokenPosition` objects dispatch on `input_sample["template"]` at index time.

## Counterfactuals

`counterfactuals.py::generate_dataset(model, n, seed)` returns `n` examples of shape `{"input": ..., "counterfactual_inputs": [...]}` where both base and counterfactual are independent samples — every input variable may differ.

For single-variable counterfactuals (only one variable resampled), build the table with `scripts/build_split_dataset.py --resample-variable <var>`, for example `--resample-variable number`. Each base then pairs with a copy that differs only in that variable. Pairwise patching needs such a table, because it is only meaningful when exactly one input variable changes.

## How to Run

The [method library](../../../demos/methods/README.md) runs many of its documents on the shipped weekdays table. Its workflow `demos/methods/workflows/weekdays.json` scans layer and position, fits DAS at the selected location, and scores the fit. It runs from the repository root with:

```bash
uv run causalab run demos/methods/workflows/weekdays.json \
    --engine auto \
    --artifacts-root . \
    --out runs/weekdays \
    --device cuda
```

The run writes its outputs under `--out`. The onboarding demo [weekdays_geometry](../../../demos/onboarding_tutorial/weekdays_geometry.md) studies the geometry of the same domain. For the other domains, build a table with `--set domain_type=<domain>` and name it in a document. [Running experiments](../../../docs/running_experiments.md) shows how to write, validate and run a document.

## Files

| File | Role |
|---|---|
| `config.py` | `NaturalDomainConfig` dataclass + the `DOMAIN_PRESETS` table |
| `causal_models.py` | `create_causal_model`, `create_random_causal_model`, plus the `GET_*` accessors used by `tasks/loader.py` |
| `counterfactuals.py` | `generate_dataset` |
| `token_positions.py` | `create_token_positions` (single- or multi-template) |
| `data/{weekdays,months}.json` | the shipped tables, one per domain: `natural_domains_arithmetic/data/weekdays#train` (30) / `#test` (19) and `…/months#train` (51) / `#test` (33) — whole pools, group-disjoint; built with `uv run python scripts/build_split_dataset.py --task natural_domains_arithmetic --set domain_type=weekdays --seed 0 --fraction train=0.6 --fraction test=0.4 --target-variable result --out causalab/tasks/natural_domains_arithmetic/data/weekdays.json` (and `domain_type=months` → `months.json`) |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |
