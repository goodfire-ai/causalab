# Task Definitions for Causal Abstraction Experiments

Each task is a self-contained package under `causalab/tasks/<name>/` that
defines a causal model, counterfactual generation, and tokenization helpers.
Tasks are loaded via `load_task()` in `causalab.tasks.loader`. A protocol
document does not import a task: it names a **dataset ref**, and the ref
resolves to a serialized table that `causalab.tasks.serialize` built from the
task ahead of time (spec §2.2). That is the seam between the two halves — the
task owns generation and answer semantics, the document owns the intervention.

Standing up a usable task has three parts: (1) the task **package**
(`causalab/tasks/<name>/`), (2) a **serialized table** a document can name, and
(3) **validation** against a model. All three are described below.

## 1. The task package (`causalab/tasks/<name>/`)

Create a directory `causalab/tasks/<name>/` with these modules:

| File | Required | Purpose |
|------|----------|---------|
| `causal_models.py` | yes | Causal model + the exports `load_task()` reads |
| `counterfactuals.py` | yes | Counterfactual-pair generation |
| `token_positions.py` | for interventions | Maps variable names → token positions |
| `config.py` | yes | Constants: task name, value lists, token budgets |
| `templates.py` | yes | Input text templates + fill function |
| `checker.py` | optional | Custom output matcher (see below) |
| `metrics.py` | optional | Task-specific metric helpers |
| `__init__.py` | yes | Package exports |
| `summary.ipynb` | optional | CPU-only task overview notebook (no model load) |

### causal_models.py (required)

Defines the causal model and exports that `load_task()` reads by convention.

**Singleton tasks** (fixed structure, e.g. weekdays, months, years):

| Export | Type | Required |
|--------|------|----------|
| `CAUSAL_MODEL` | `CausalModel` | yes |
| `VARIABLE_VALUES` | `dict[str, list[str]]` — var name → values | yes |
| `CYCLIC_VARIABLES` | `set[str]` — which variables wrap cyclically | yes (may be empty) |
| `EMBEDDINGS` | `dict[str, Callable]` — var name → embedding fn | yes |
| `PERIODIC_INFO` | `dict[str, int]` — var name → period length | no |
| `TEMPLATE` | `str` — prompt template | no |
| `TARGET_VARIABLE` | `str` — variable being steered | no |
| `RANDOM_CAUSAL_MODEL` | `CausalModel` — random baseline model | no |
| `RANDOM_VARIABLE_VALUES` | `dict[str, list[str]]` — values for random baseline | no |

The per-value output forms used for scoring are **not** a task export — they are
declared on the `CausalModel` itself via `output_tokens` (`{variable: {value:
[surface form, ...]}}`; build the mechanical `[" v", v]` map with
`build_output_tokens`). The probability-path score tokens and the string
`checker` are both derived from it, so a task that declares `output_tokens` needs
no `checker.py`. Optional `match_modes={variable: "prefix"}` accepts output that
continues past the answer.

**Factory tasks** (parameterized, e.g. graph_walk, natural_domains_arithmetic):

| Export | Type | Required |
|--------|------|----------|
| `CREATE_CAUSAL_MODEL` | `Callable[[config], CausalModel]` | yes |
| `GET_VARIABLE_VALUES` | `Callable[[CausalModel], dict[str, list[str]]]` | yes |
| `CYCLIC_VARIABLES` | `set[str]` | yes (may be empty) |
| `EMBEDDINGS` | `dict[str, Callable]` | yes |
| `GET_CYCLIC_VARIABLES` | `Callable[[CausalModel], set[str]]` | no (overrides `CYCLIC_VARIABLES`) |
| `GET_EMBEDDINGS` | `Callable[[CausalModel], dict[str, Callable]]` | no (overrides `EMBEDDINGS`) |
| `GET_PERIODIC_INFO` | `Callable[[CausalModel], dict[str, int] \| None]` | no |
| `CREATE_RANDOM_CAUSAL_MODEL` | `Callable[[config], CausalModel]` | no |
| `TARGET_VARIABLE` | `str` | no |

### counterfactuals.py (required)

```python
generate_dataset(causal_model, n_examples, seed) → list[dict]
```

Each dict has `"input"` (a causal trace) and `"counterfactual_inputs"`
(list of counterfactual traces).

### token_positions.py (required for intervention experiments)

```python
create_token_positions(pipeline, ...) → dict[str, TokenPosition]
```

Maps position names to `TokenPosition` objects that locate where in
the token sequence to intervene.

### config.py (required)

Task constants: `TASK_NAME` (must equal the package directory name — the loader
key), the input variable value lists, and the token budgets `MAX_TASK_TOKENS`
(max input length) and `MAX_NEW_TOKENS` (tokens the model generates; `1` for
single-token-answer tasks).

### templates.py (required)

The input text templates and a `fill_template(...)` function. Every placeholder
in a template must correspond to at most one causal-model variable — don't
pre-concatenate variables into intermediate strings; the template's `.format()`
is the formatting step.

### checker.py (optional)

```python
checker(neural_output: dict, causal_output: str) -> bool
```

Decides whether the model's output (`neural_output["string"]`) matches the
expected answer. `checker.py` is **optional**: when a task declares
`output_tokens` on its `CausalModel`, the string checker (and the
probability-path score tokens) are derived from it automatically. Ship a
`checker.py` only when you need a genuinely-custom matcher — it takes precedence
over the derived one. A task that offers **neither** `output_tokens` **nor**
`checker.py` for its target variable cannot grade its output and fails to load.
Pick the semantics your task needs: exact stripped equality for single-token
answers, `actual.startswith(expected)` for `max_new_tokens > 1` tasks whose model
continues past the answer, etc.

### metrics.py / __init__.py / summary.ipynb

`metrics.py` holds optional task-specific metric helpers. `__init__.py` re-exports
the package's public surface (at minimum `CAUSAL_MODEL` / the factory). An optional
`summary.ipynb` demonstrates the *task* (causal model, samples, token positions,
counterfactuals) on CPU — it must not load a language model.

### Session-local task packages

You can prototype a task outside the installed `causalab` package. Use the same
modules and exports described above, under a top-level `tasks` package:

```text
<session>/code/
└── tasks/
    ├── __init__.py
    └── my_task/
        ├── __init__.py
        ├── causal_models.py
        ├── counterfactuals.py
        └── ...
```

The task loader tries `causalab.tasks.<name>` first. If that package is absent
and `CAUSALAB_SESSION_CODE` is **nonempty**, it tries `tasks.<name>`. An unset or
empty value disables this fallback. The variable is only an enablement gate:
the loader does not validate its value or add a directory to Python's import
path. The caller must also put `<session>/code/` on `PYTHONPATH` (or `sys.path`
when calling from Python). There is no automatic session-directory detection.

For example, from the repository root, with a singleton task implemented under
the layout above:

```bash
SESSION_DIR=/absolute/path/to/session
CAUSALAB_SESSION_CODE="$SESSION_DIR" \
PYTHONPATH="$SESSION_DIR/code${PYTHONPATH:+:$PYTHONPATH}" \
uv run python scripts/build_task_dataset.py \
    --task my_task --n 64 --seed 0 --out data/my_task/train.json
```

The task-module helpers in `causalab.tasks.loader` share the shipped-first rule in
`causalab.tasks.loader._task_package_candidates`. A session-local package
cannot shadow a shipped task, fill in a missing module of a shipped task, or
hide a broken import inside one. Resolution selects the task package first;
errors loading its modules propagate from that package.

This fallback supports task loading and dataset building. Protocol documents
continue to read the serialized tables described below; they do not import
session-local task code.

For a session-local factory task, pass `task_cfg` directly to `load_task()` or
`serialize_counterfactual_dataset()`. The dataset builder's `--set` config-class
discovery currently checks only shipped `causalab.tasks.<name>.config` modules.

## 2. Serializing a table a document can name

A task becomes usable by a protocol document when its counterfactual dataset
exists as a table under the document's data root:

```bash
uv run python scripts/build_task_dataset.py \
    --task <name> --n 64 --seed 0 --target-variable <var> \
    --out <data-root>/<name>/train.json
```

Factory tasks take their config through `--set key=value` (resolved against the
`*Config` dataclass in the package's `config.py`):

```bash
uv run python scripts/build_task_dataset.py \
    --task natural_domains_arithmetic --set domain_type=weekdays \
    --n 64 --seed 0 --target-variable result --out data/weekdays/train.json
```

What the builder writes, and why it is a *build step* rather than something a
load does:

- **The columns a document references** — the rendered prompts (`input`,
  `counterfactual_inputs`), each prompt's own answer (`base_answer`,
  `cf_answer`), the post-intervention `label` from
  `CausalModel.label_counterfactual_data`, the answer forms from the causal
  model's `output_tokens` declaration (`*_forms`), and every causal-model
  variable as a per-row column for position resolution. See
  `causalab/tasks/serialize.py` for the full vocabulary.
- **Deterministic bytes**, so the content digest a document's canonical form
  stamps (§7) is reproducible from the parameters recorded in the
  `<ref>.manifest.json` sidecar written beside the table. `--check` rebuilds
  and fails instead of writing — the guard for a committed table.
- **No model, no tokenizer.** Tables are text and variable strings, which is
  what lets `causalab validate` / `explain` / `digest` run without either, and
  lets one table run under different models.

Two things a task therefore declares for itself, rather than a document
computing them:

- `output_tokens` — which surface strings count as one answer. A `match` metric
  consumes the serialized group, so synonyms and casings are task data (§2.10).
- `match_modes` — `prefix` for a task whose answers are not single-token. The
  builder records it in the manifest as `declared_match_mode`; the document
  spelling is `"mode": "first_token"`.

## 3. Validating a new task

Before running the full pipeline, confirm the task tokenizes cleanly and the model
can actually solve it.

**Pre-flight tokenizer check (model-free, blocking).** Catches tokenization
mismatches (e.g. an orphaned trailing space in the answer) without loading the
model:

```bash
uv run python -m causalab.tasks.preflight --task <name> --model <model_name>
```

Exit 0 = clean; exit 1 = tokenization error to fix before proceeding; exit 2 = the
check couldn't run (e.g. a factory task that needs its run config to sample) — fall
through to the accuracy check.

**Accuracy gate.** Run `baseline` (or a short ad-hoc `generate` loop) on ~64
examples. If the model solves **< 20%**, the task is behaviorally inert for that
model — downstream interventions produce degenerate geometry (near-100% "other"
probability mass). Fix the prompt/templates or switch models before continuing.

**Single-token / spacing.** For clean token-level interventions, filter variable
values to single-token-in-context values, and confirm which spacing variant the
model actually emits (leading space vs. none) so intervention token alignment stays
correct; update `templates.py` / `config.py` to match.

## Active tasks

| Task | Description | Dimensionality |
|------|-------------|----------------|
| `natural_domains_arithmetic` | Unified weekdays/months/hours/age/integer/alphabet | factory, 1D (cyclic or linear) |
| `graph_walk` | Next-node prediction on graphs | factory, 1D or 2D |
| `entity_binding` | Positional entity retrieval | — |
| `hierarchical_equality` | Hierarchical variable equality | — |
| `identity_naming` | Entity → canonical name lookup (factory) | — |
| `MCQA` | Multiple-choice question answering | — |
| `IOI` | Indirect object identification (coverage-oriented runner) | — |
| `hex_color` | Hex-code → color-name mapping | — |
| `subject_object_relations` | Subject→object relation recall (LRE-style) | factory |
