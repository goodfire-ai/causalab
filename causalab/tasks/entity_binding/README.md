# Entity Binding

Tests whether a model retrieves the *right* entity from a list of bound (role, value) pairs. The model sees a sentence like

```
We will ask a question about the following sentences.

Pete loves jam, and Ann loves pie. What does Ann love?
Answer:
```

and is expected to produce `pie`. The interesting causal claim is *positional binding*: the model has to (a) figure out which group `Ann` is in, then (b) look up the *food* slot inside that same group. Both steps are made explicit as causal variables (`positional_queries[{e}]` and `positional_answer`), so analyses can ask which residual-stream layers compute each one.

The default configuration is the "love" config — 2 groups of 2 entities (a person and a food) — but the task is parametrized so you can scale to more groups, more entities per group, or different role schemes.

## Task Configuration

`config.py::EntityBindingTaskConfig` is the configuration dataclass. The headline knobs:

| Field | Meaning |
|---|---|
| `max_groups` | How many bound clauses are in the statement (e.g. 2 → "Pete loves jam, and Ann loves pie") |
| `max_entities_per_group` | Slots per clause (default 2: a person and a food) |
| `entity_roles` | Maps a slot index to its role name (`{0: "person", 1: "food"}`) |
| `entity_pools` | Pool of values per slot index (e.g. `{0: ["Pete", "Ann", ...], 1: ["jam", "pie", ...]}`) |
| `statement_template` | Per-clause template, e.g. `"{e0} loves {e1}"`. Variables get group-qualified at build time (`g0_e0`, `g0_e1`, …). |
| `question_templates` | Map `(query_indices, answer_index) → question_template`. `((0,), 1)` means "given the entity at slot 0, ask for the entity at slot 1". |
| `delimiters` | `FILL`-style delimiter spec for joining clauses (`[", ", "FILL", ", and ", "."]`). |
| `prompt_prefix`, `prompt_suffix`, `statement_question_separator` | Surface text around the statement+question. |

`create_sample_love_config()` returns the default 2×2 "Pete loves jam" config.

## Causal Model

`create_positional_entity_causal_model(config)` builds a causal model whose central insight is that retrieval is decomposed through **explicit positional variables**.

### Inputs

For a config with `G` groups and `E` slots per group:

| Variable | Range | Meaning |
|---|---|---|
| `entities[{g},{e}]` | `entity_pools[e]` | Entity at slot `e` of group `g` |
| `query_group` | `0..G-1` | Which group is being asked about |
| `query_indices` | tuples like `(0,)` | Which slot the model is given |
| `answer_index` | `0..E-1` | Which slot the model must retrieve |
| `active_groups`, `entities_per_group`, `statement_template` | fixed | Surface knobs that aren't varied during analyses |

### Computed (derived) variables

| Variable | Computed from | Meaning |
|---|---|---|
| `queries[{e}]` | `entity_g*_e{e}` + `query_group` | The entity at slot `e` *of the query group* — i.e. the entity the question references. |
| `positional_entities[{g},{e}]` | `entities[{g},{e}]` | Trivially `g` (or `None` if entity missing). Exists so analyses can intervene on "where is this entity" without conflating with identity. |
| `question_template` | `query_indices`, `answer_index` | Picks the right question template from the config map. |
| `positional_queries[{e}]` | the same role of `entities`, `positional_entities`, and `queries`, plus `query_indices` and `active_groups` | The set of group indices where the query entity appears at slot `e`. With `ensure_positional_uniqueness` (the sampler default), this is a singleton — the group that contains the query entity. |
| `positional_answer` | `positional_queries`, `query_indices` | Intersection of the per-slot positional-query sets — the *single* group from which to retrieve. **This is the `TARGET_VARIABLE`** for analyses. |
| `raw_input` | entity and query families plus query/control inputs | The full prompt. |
| `raw_output` | `positional_answer`, `answer_index`, `entities` | The expected answer entity. |

`positional_answer` is what makes the task analytically tractable: it's a single integer (the group index) that, by hypothesis, is what the model has to compute internally and route through the residual stream. Localizing where it lives in the network is the typical research question.

### Sampling valid inputs

`sample_valid_entity_binding_input(config, model, ensure_positional_uniqueness=True)` is the only correct way to produce examples. It enforces:

1. Active groups have all entity slots filled.
2. `query_group` is within active groups.
3. `query_indices ∩ {answer_index}` is empty (you can't ask about and answer with the same slot).
4. A `question_template` exists for `(query_indices, answer_index)`.
5. With `ensure_positional_uniqueness=True`: entities at the same slot index are distinct across groups (so `positional_queries` is unambiguous).

Don't construct inputs by hand for the positional model — the constraint logic above is non-trivial and silent constraint violations corrupt analyses.

## Counterfactuals

Three generators in `counterfactuals.py`:

| Generator | What changes | Use case |
|---|---|---|
| `swap_query_group(config)` | Swaps the entire query group with another group; updates `query_group` so the query entities still match. **`positional_answer` flips** but the question text references the same entity. | Deconfounds *positional binding* from *entity identity* — the gold standard CF for analyzing this task. |
| `swap_query_group(config, change_answer=True)` | Same as above, but also resamples the answer entity in the new query group from a fresh entity pool. | When you want to additionally test that the model isn't memoizing a specific answer string. |
| `random_counterfactual(config)` | Two independent samples — every input variable may differ. | Distribution baseline. |

The `COUNTERFACTUAL_GENERATORS` dict exposes zero-arg wrappers that use the default love config, for systems that take generators as `() -> CounterfactualExample`.

`generate_dataset(causal_model, n, seed)` (the loader-convention entry point) uses a fourth strategy: keep all entity slots fixed and only **resample `query_group`**. This is the cleanest CF for a pairwise interchange: exactly one input variable changes, and both `positional_answer` and `raw_output` flip cleanly.

## Token Positions

`create_token_positions(pipeline, template, config)` returns:

| Name | Description |
|---|---|
| `last` | Final prompt token (declarative). |
| `g{g}_e{e}_last` | Last token of the entity at slot `e` of group `g`, in the **statement region** only. Built by prefix-tokenization: tokenize the prompt up to the entity's character end, subtract the prefix up to its start. |

Statement-only resolution matters because the same entity (`Ann`) can appear both in the statement and in the question. Targeting the statement occurrence pinpoints where the entity is *bound* to its group, which is the position that carries positional information.

`get_question_entity_token_positions` and `get_statement_entity_token_positions` are the lower-level utilities that back the position factories; they accept either an `entity_idx` or a `role_name` (e.g. `"person"`), with explicit error messages when an entity slot doesn't appear in the question.

## How to Run

No document in this repository runs this task yet, and the task ships no table. Build a table with `scripts/build_task_dataset.py`, name it in the `data` block of a document, and run the document with `causalab run`:

```bash
uv run python scripts/build_task_dataset.py \
    --task entity_binding \
    --n 64 \
    --seed 0 \
    --split all \
    --out data/entity_binding.json
uv run causalab run <document> \
    --engine auto \
    --data-root data \
    --out runs/entity_binding
```

The builder notes that this task scores `raw_output` by prefix, so a `match` metric over the table uses `"mode": "first_token"`. [Running experiments](../../../docs/running_experiments.md) shows how to write, validate and run a document.

## Files

| File | Role |
|---|---|
| `config.py` | `EntityBindingTaskConfig`, `create_sample_love_config`, template-build helpers (`_build_conjoined_template`, `_expand_delimiters`) |
| `causal_models.py` | `create_positional_entity_causal_model`, `sample_valid_entity_binding_input`, the default `causal_model` instance (with its `ScoringSpec`: the entity forms on `raw_output`, `string_mode="prefix"`), plus loader hooks (`CAUSAL_MODEL`, `TARGET_VARIABLE`) |
| `templates.py` | Template-rendering helpers used by `_compute_raw_input` |
| `counterfactuals.py` | `swap_query_group`, `random_counterfactual`, `generate_dataset` (query_group-only resampling), `COUNTERFACTUAL_GENERATORS` |
| `token_positions.py` | `create_token_positions`, statement/question entity-position resolvers |
| `data/` | none yet — `output_tokens` is declared on `positional_answer` (an index) with entity-name forms, so the serializer refuses; declare it on `raw_output` first |
| `metrics.py` | Task-specific scoring and evaluation utilities |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |

The [equations](causal_models.py) infer read dependencies. `statement_template` and
`question_template` remain variables, but do not have edges to `raw_input`, which
uses the configured template builder. Invalid rendered configurations raise an
error; the observation filter samples full supported grids, while interventions
can use inactive slots and explicitly handled `None` values.
