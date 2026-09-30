# IOI (Indirect Object Identification)

> **Coverage-oriented.** Wired into the runner pipeline so the IOI modules
> participate in the smoke + golden tiers (see docs/TESTS.md "Coverage-oriented
> runners"). The shipped runner is **not scientifically meaningful** —
> small `n_train` / `n_test`, single template, smallest production model.
> The pyvene-era `demo.ipynb` (against `gpt2`) was removed in a deletion
> sweep — it had been unrunnable since its imports were retired; it lives in
> git history.

## Task

Given a prompt of the form
`"After Christopher and Kevin went to the park, Kevin gave a ball to"`,
the model is expected to predict the indirect object — the name that is
*not* the subject of the action — i.e. `" Christopher"`.

Each example is composed of:

- `template` — the canonical IOI template (placeholders `{name_A}`,
  `{name_B}`, `{name_C}`, `{place}`, `{object}`). The full historical
  template list is still available as `ALL_TEMPLATES`.
- `name_A`, `name_B`, `name_C` — three names drawn from `sources/names.json`.
  Well-formedness (enforced by the model's `input_filter`) requires
  `name_A != name_B` and `name_C ∈ {name_A, name_B}`.
- `place` (`places.json`) and `object` (`objects.json`) — surface vocab.

## Causal Model

`positional_causal_model` (alias `CAUSAL_MODEL`) is a `CausalModel` with:

| Variable | Role |
|---|---|
| `template`, `name_A`, `name_B`, `name_C`, `place`, `object` | Inputs |
| `IO` | The indirect object — whichever of `name_A` / `name_B` differs from `name_C`. **Target variable.** |
| `raw_input` | Filled prompt string |
| `raw_output` | `" " + IO` — the expected next-token output |

`TARGET_VARIABLE = "IO"`. The model exports the canonical `TEMPLATE` string
and the full `ALL_TEMPLATES` list (the latter unused by the coverage runner
but available for ad-hoc work).

## Counterfactuals

| Generator | What changes between input and counterfactual |
|---|---|
| `flip_name_C` | `name_C` is flipped to the other of `{name_A, name_B}` — the canonical IOI counterfactual; isolates the subject-of-action signal from entity identity. |
| `random_counterfactual` | Two independent well-formed samples — noise-floor reference. |
| `generate_dataset(model, n, seed)` | Loader-convention entry point; builds `n` examples by calling `flip_name_C` under a fixed seed. |

## Token Positions

`create_token_positions(pipeline, template=...)` returns four
`TokenPosition` objects via the declarative spec system used by `MCQA`:

| Name | Description |
|---|---|
| `name_A` | Token(s) of the `{name_A}` template variable |
| `name_B` | Token(s) of the `{name_B}` template variable |
| `name_C` | Token(s) of the `{name_C}` template variable |
| `last_token` | Final prompt token |

## Files

| File | Role |
|---|---|
| `causal_models.py` | `CAUSAL_MODEL` (singleton), `TARGET_VARIABLE`, `TEMPLATE`, `CANONICAL_TEMPLATE`, `ALL_TEMPLATES`, name/place/object pools |
| `counterfactuals.py` | `generate_dataset`, `flip_name_C`, `random_counterfactual` |
| `token_positions.py` | `create_token_positions` (declarative-spec based) |
| `data/default.json` | the shipped table: `IOI/data/default` (256 pairs, `split all`); built with `uv run python scripts/build_task_dataset.py --task IOI --n 256 --seed 0 --split all --target-variable IO --out causalab/tasks/IOI/data/default.json` |
| `sources/{names,objects,places,templates}.json` | Domain vocab |

The pyvene-era `demo.ipynb` (end-to-end walkthrough on `gpt2`) was removed in
a deletion sweep; it targeted the retired API and had been
unrunnable since its imports were retired — recover it from git history if
needed.
