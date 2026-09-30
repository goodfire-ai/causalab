# Hierarchical Equality

Tests whether a model computes *double equality*: given four single-letter inputs `(a, b, c, d)`, predict `(a == b) == (c == d)`. The model is shown 60 balanced in-context examples and then asked to label a held-out query as `1` or `0`. The intermediate variables `left_equality` and `right_equality` are exactly what makes this task interesting — interchange interventions can ask whether either lives as a separable representation in the residual stream.

```
f(A,A,B,B)=1
f(C,D,C,D)=0
f(A,B,A,B)=1
... (60 ICL examples) ...
f(M,M,N,O)=
```

The model is expected to emit `0` (left is True, right is False).

## Causal Model

```
var_1 ─┐
       ├──> left_equality ─┐
var_2 ─┘                   │
                           ├──> result_equality ──> raw_output
var_3 ─┐                   │
       ├──> right_equality ┘
var_4 ─┘

(var_1, var_2, var_3, var_4, template) ──> raw_input
```

Variables:

| Variable | Role | Notes |
|---|---|---|
| `var_1`–`var_4` | Inputs | Letters from `A`…`Z`. |
| `template` | Input | Prompt-style sentinel. **Not a meaningful causal variable** — it only selects rendering. |
| `left_equality` | Computed | `var_1 == var_2`. |
| `right_equality` | Computed | `var_3 == var_4`. |
| `result_equality` | Computed | `left_equality == right_equality`. The default `target_variable` for analyses. |
| `raw_input` | Computed | Full prompt (60 ICL examples + query) rendered per `PROMPT_MODE`. |
| `raw_output` | Computed | `"1"` if `result_equality` else `"0"`. |

Because the prompt is regenerated *every time* `raw_input` is computed (each call resamples 60 ICL examples), traces of the same `(var_1, var_2, var_3, var_4)` will produce different prompts. This is intentional — it keeps the ICL examples broad — but it means snapshotted prompts are not reproducible across re-traces.

The task's `ScoringSpec` declares the forms `{True: [" 1", "1"], False: [" 0", "0"]}` for all three equality variables (both spacings, so scoring tolerates the leading-space variant), grades `result_equality`'s digit (its `answer_variable`), and sets `string_mode="prefix"` so the grader accepts output that continues past the answer.

## Prompt Modes

`config.py::PROMPT_MODE` selects the prompt style; templates are built in `templates.py::fill_template`:

| Mode | Per-example format | Query format |
|---|---|---|
| `minimal_function` (default) | `f(A,A,B,B)=1` | `f(M,M,N,O)=` |
| `algorithmic` | `A A B B: 1` | `M M N O: ` |
| `code` | `The function call double_equality("A", "A", "B", "B") returns the value 1` | `The function call double_equality("M", "M", "N", "O") returns the value ` |

In `code` mode, the prompt also begins with a Python function definition (one of three variants in `CODE_TEMPLATES`) so the model can read the algorithm before the ICL block. In `minimal_function` and `algorithmic` modes, `template` collapses to a single sentinel string ("minimal_function" or "algorithmic") because the format is fully determined by `PROMPT_MODE`.

`NUM_ICL_EXAMPLES = 60` is split across four sampling patterns:

| Pattern | Sampling rule | Truth value |
|---|---|---|
| `AABB` | `var_1 == var_2`, `var_3 == var_4`, but `var_1 != var_3` likely | T == T → `1` |
| `ABCD` | All four distinct | F == F → `1` |
| `ABCC` | `var_1 != var_2`, `var_3 == var_4` | F == T → `0` |
| `AABC` | `var_1 == var_2`, `var_3 != var_4` | T == F → `0` |

Two patterns produce `1`, two produce `0` — the ICL block is balanced by construction.

## Counterfactuals

`counterfactuals.py::generate_dataset(model, n, seed)` returns `n` examples; both base and counterfactual are independent calls to `sample_balanced_input`, which samples a pattern uniformly from `PATTERNS` and instantiates letters accordingly. So every `(left_equality, right_equality)` pair appears with equal frequency in expectation.

The single registered generator is `random_counterfactual` (in `COUNTERFACTUAL_GENERATORS`).

## Token Positions

ICL prompts repeat the letters many times, so a declarative "find the last occurrence of `A`" approach would land on the wrong token. `token_positions.py` instead parses the **last line** (the test query) with a `PROMPT_MODE`-specific regex and maps the matching character span to token indices via `get_tokens_in_char_range`.

| Name | Description |
|---|---|
| `var_1`, `var_2`, `var_3`, `var_4` | Tokens spanning each query variable in the test line |
| `last` | Final prompt token |

The regex differs per prompt mode:

- `minimal_function` — `(?<=[(,])([^,)]+)` — content after `(` or `,`, before `,` or `)`.
- `algorithmic` — `(\S+)(?=\s)` — first four whitespace-delimited tokens of the line.
- `code` — `"([^"]*)"` — quoted values inside the function call.

## How to Run

No document in this repository runs this task yet. A document names the shipped table as `hierarchical_equality/data/default#all` in its `data` block, and runs with:

```bash
uv run causalab run <document> \
    --engine auto \
    --out runs/hierarchical_equality
```

[Running experiments](../../../docs/running_experiments.md) shows how to write, validate and run a document.

## Files

| File | Role |
|---|---|
| `config.py` | Constants (`LETTERS`, `PATTERNS`, `NUM_ICL_EXAMPLES`, `PROMPT_MODE`) |
| `causal_models.py` | `CAUSAL_MODEL` (with its `ScoringSpec`) and the `TARGET_VARIABLE` loader hook |
| `templates.py` | `TEMPLATES`, `fill_template`, `_sample_pattern_values`, `generate_icl_examples` |
| `counterfactuals.py` | `sample_balanced_input`, `generate_dataset`, `COUNTERFACTUAL_GENERATORS` |
| `token_positions.py` | `create_token_positions` (custom Python — not declarative due to ICL repeats) |
| `data/default.json` | the shipped table: `hierarchical_equality/data/default` (256 pairs, `split all`; scoring forms and explicit `icl_seed` values); built with `uv run python scripts/build_task_dataset.py --task hierarchical_equality --n 256 --seed 0 --split all --target-variable result_equality --out causalab/tasks/hierarchical_equality/data/default.json` |
| `metrics.py`, `icl_scaling.py` | Task-specific scoring helpers |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |

ICL randomness is an explicit `icl_seed` input. The dataset generator supplies
independent seeds for base and donor traces; re-evaluating a trace keeps its
prompt examples fixed. For manual traces, supply the seed with the letter inputs.
