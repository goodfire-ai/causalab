# Task Definitions for Causal Abstraction Experiments

Each task is a self-contained package under `causalab/tasks/<name>/` that
defines a causal model, counterfactual generation, and tokenization helpers.
Tasks are loaded via `load_task()` in `causalab.tasks.loader`. An
intervention specification does not import a task: it names a **dataset ref**,
and the ref resolves to a serialized table that `causalab.tasks.serialize`
built from the task ahead of time (spec §2.2). That is the seam between the two halves — the
task owns generation and answer semantics, the document owns the intervention.

Standing up a usable task has three parts: (1) the task **package**
(`causalab/tasks/<name>/`), (2) a **serialized table** a document can name, and
(3) **validation** against a model. All three are described below.

## Local task packages

For intermediate-variable hypotheses, use the
[saved comparison guide](../../docs/hypothesis_analysis.md). The exporter keeps
pair IDs, family, endpoints, and splits alongside each target's labels.

A task can live outside the installed CausaLab source. Create
`<code_root>/tasks/__init__.py` and `tasks/<name>/` with the same modules and
exports described below, then make that code root importable:

```bash
export CAUSALAB_SESSION_CODE="/absolute/path/to/code_root"
export PYTHONPATH="$CAUSALAB_SESSION_CODE${PYTHONPATH:+:$PYTHONPATH}"
```

`CAUSALAB_SESSION_CODE` enables the local-task fallback; `PYTHONPATH` supplies
the import path. `load_task(name)` first tries `causalab.tasks.<name>`, then
`tasks.<name>`. Use a unique name because a local task cannot shadow a shipped
one. The same resolution applies to token positions and counterfactual generators.
Keep task definitions as Python source; token-position instances may hold a
loaded pipeline and should be recreated from their definitions.

## 1. The task package (`causalab/tasks/<name>/`)

Create a directory `causalab/tasks/<name>/` with these modules:

| File | Required | Purpose |
|------|----------|---------|
| `causal_models.py` | yes | Causal model + the exports `load_task()` reads |
| `counterfactuals.py` | yes | Counterfactual-pair generation |
| `token_positions.py` | for interventions | Maps variable names → token positions |
| `config.py` | yes | Constants: task name, value lists, token budgets |
| `templates.py` | yes | Input text templates + fill function |
| `metrics.py` | optional | Task-specific metric helpers |
| `__init__.py` | yes | Package exports |
| `summary.ipynb` | optional | CPU-only task overview notebook (no model load) |
| `data/<variant>.json` | yes, once serialized (§2) | the table(s) a document names — `<task>/data/<variant>#<split>` |
| `sources/` | if the generator reads files | stimulus inputs (`hex_color/sources/hex_color.json`, `IOI/sources/names.json`); never named by a document |

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

What counts as a correct answer is **not** a task export either — it is declared
on the `CausalModel` itself, once, as its `ScoringSpec`
(`causalab.causal.scoring`; see "Scoring" below). The probability-path score
tokens, the string grader (`Task.checker`) and the serialized answer-form
columns are all derived from it.

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

### Scoring — the `ScoringSpec` (required, on the `CausalModel`)

```python
from causalab.causal.model import CausalModel
from causalab.causal.scoring import build_output_tokens
from causalab.causal.scoring import ScoringSpec

CAUSAL_MODEL = CausalModel(
    mechanisms,
    values,
    id=TASK_NAME,
    scoring=ScoringSpec(
        forms={"weekday": build_output_tokens(WEEKDAYS)},  # {variable: {value: [forms]}}
        string_mode="exact",  # or "prefix" for a task whose model continues past the answer
    ),
)
```

One immutable object is the task's definition of correct: it is frozen and its
mappings are read-only, so nothing downstream can edit the declaration. A task
whose causal model declares no `ScoringSpec` cannot grade its output and fails
to load. The fields (`tests/protocol/test_vocabulary_census.py` holds this table
to `SCORING_FIELDS`):

| field | type | what it retires |
|---|---|---|
| `forms` | `{variable: {value: [surface form, …]}}` | the `output_tokens` constructor argument and the mutable attribute it became; build the mechanical `[" v", v]` map with `build_output_tokens` |
| `answer_variable` | `str`, optional when one variable declares forms | the loader keying the string checker on `TARGET_VARIABLE`: this is the variable the graded string (`raw_output`) is a form of — MCQA grades `answer`, the letter, while an interchange targets `answer_position` |
| `string_mode` | `exact` or `prefix` | the `match_modes` map: one mode per task, whether a generated string must equal a form or merely start with one |
| `protocol_mode` | derived: `exact` or `first_token` | the `prefix` → `first_token` bridge that lived in a parse-error string; the `mode` a `match` metric over this task's table declares (spec §2.10's translation table) |
| `full_string_checker` | dotted locator `package.module.function`, optional | the bespoke `checker.py` that silently won over the derived checker; declared here it is part of the spec, located and validated when the spec is built |
| `undeclared_value` | `refuse` or `literal` | the checker's silent literal-match fallback on an expected value it did not know — `refuse` (default) is what the serializer always did, `literal` is stated rather than inherited |
| `invalid_output` | `incorrect` or `unscored` | nothing: the grade of a generation naming no declared value — `0.0`, or `null` so "never said it" stays distinct from "said it and scored 0" |
| `version` | `int`, from 1 | nothing: bumped when the meaning of correctness changes under an unchanged declaration |

`ScoringSpec.grader()` is the `checker(neural_output, causal_output) -> bool`
the task exposes as `Task.checker`; `ScoringSpec.grade(generated, expected)`
is the same decision as `1.0` / `0.0` / `None`; `ScoringSpec.forms_of(value)`
is the answer-form group a row carries and the probability path scores;
`ScoringSpec.form_groups()` the distinct groups. An expected value resolves by
identity, by its spelling, or by being one of its forms; a *list* is a list of
acceptable answers (graph_walk's `raw_output` is every valid next node).
`CausalModel.output_tokens` and `CausalModel.match_modes` remain as read-only
derived views for the readers that grew up on those names.

### metrics.py / __init__.py / summary.ipynb

`metrics.py` holds optional task-specific metric helpers. `__init__.py` re-exports
the package's public surface (at minimum `CAUSAL_MODEL` / the factory). An optional
`summary.ipynb` demonstrates the *task* (causal model, samples, token positions,
counterfactuals) on CPU — it must not load a language model.

## 2. Serializing a table a document can name

A task becomes usable by an intervention specification when its counterfactual dataset
exists as a table, and **the task ships that table itself**: under
`causalab/tasks/<name>/data/<variant>.json`, with the builder invocation
recorded in the task's README (§2 below shows the shape). Nothing sits
beside the table (spec §2.2). `causalab/tasks/` is the CLI's default
`--data-root`, so a document names the table as `<name>/data/<variant>#<split>`
with no flag. A variant is named for the *configuration* it was built from
(`default`; a factory task's domain, `weekdays`), never for its parameters —
`n`, `seed` and fractions are the builder's arguments, so a table can grow
without every document that names it changing its ref. One task may ship
several variants (a `default` table and a `small` smoke table side by side);
a hand-authored table that no task generates (`pile/data/sample`) is a table
like any other.

```bash
uv run python scripts/build_task_dataset.py \
    --task <name> --n 64 --seed 0 --split all --target-variable <var> \
    --out causalab/tasks/<name>/data/default.json
```

Factory tasks take their config through `--set key=value` (resolved against the
`*Config` dataclass in the package's `config.py`):

```bash
uv run python scripts/build_task_dataset.py \
    --task natural_domains_arithmetic --set domain_type=weekdays \
    --n 64 --seed 0 --split all --target-variable result \
    --out causalab/tasks/natural_domains_arithmetic/data/weekdays.json
```

`--split` is required and has no default. `all` says the table is one undivided
pool — which is a claim, not a formality: a table that declines to say is
exactly what the column exists to rule out, and the resolver refuses one (§2.2
rule 22).

**For a train/test table, build one table, not two.** `build_split_dataset.py`
partitions the task's unique inputs into disjoint groups and pairs
counterfactuals *within* each split, so the rows carry their own partition:

```bash
uv run python scripts/build_split_dataset.py \
    --task natural_domains_arithmetic --set domain_type=weekdays \
    --seed 0 --fraction train=0.6 --fraction test=0.4 \
    --out causalab/tasks/natural_domains_arithmetic/data/weekdays.json
# wrote causalab/tasks/natural_domains_arithmetic/data/weekdays.json (49 rows, {'test': 19, 'train': 30}, digest f9cb22622002…)
```

Documents then name `natural_domains_arithmetic/data/weekdays#train` and
`natural_domains_arithmetic/data/weekdays#test`. Two *files*
called train and test assert their relationship in their names and nowhere a
reader can reach; two splits of one table cannot share a row, and whether they
share a prompt is checked from the bytes every time the table is read.

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
  stamps (§7) is reproducible from the command line that built the table —
  which the task's README records (the examples above). **A committed table
  is a build product**, and it is also exactly the bytes a document names:
  nothing sits beside it, and no CI guard rebuilds it (spec §2.2). A change
  to a task's generator or causal model therefore moves no committed table
  by itself; when the table should follow, rebuild it with the recorded
  command (`--check` proves nothing drifted), then repin the documents that
  name the table (`tests/protocol/update_*_digests.py`,
  `scripts/repin_demo_digests.py`). `tests/tasks/test_shipped_tables.py`
  keeps the one invariant that needs no recipe: every task package ships a
  table or says why it cannot yet.
- **No model, no tokenizer.** Tables are text and variable strings, which is
  what lets `causalab validate` / `explain` / `digest` run without either, and
  lets one table run under different models.
- **Nothing beside the table is read at run time.** A run consumes the
  table's bytes, whose content digest is already in every consuming
  document's canonical form and so in every step identity that names it
  (spec §2.2, §7). A rebuilt table therefore moves those identities: every
  quoted digest must be repinned, and `--resume` re-runs every step that
  read it.

Two things a task therefore declares for itself, in its `ScoringSpec`, rather
than a document computing them:

- `forms` — which surface strings count as one answer. A `match` metric
  consumes the serialized group, so synonyms and casings are task data (§2.10).
- `string_mode` — `prefix` for a task whose answers are not single-token. The
  builder writes it into every row as `string_mode`; the document spelling is
  `"mode": "first_token"` (§2.10's translation table), and a document
  declaring `"exact"` over a `prefix` table is refused before any forward. A
  table without the column is *unrecorded* and runs as it always did.

A generator may additionally declare **which spans of a pair move together**
(`causalab/causal/pair_validation.py`; spec §2.2): attach an `edit_groups` key to the
example it returns — a list of `{"name", "atomic", "spans": {"base": [[start,
end], …], "counterfactual": [[start, end], …]}}`, character spans into the two
prompts, one constituent per span pair — and the serializer writes it into the
row as the `edit_groups` column after checking the spans against the texts. A
relation word and the total it changes, or the two entries of a swapped
mapping, are one `atomic` group: the pair was validated as one coordinated
edit, so a run that addresses one constituent without the others is refused
before its first forward (rule 27) instead of reporting a number for an
intervention the pair does not license; `atomic: false` declares the spans
and asks for nothing. No shipped generator declares groups, so no shipped
table carries the column. `scripts/build_task_dataset.py --validate-pairs
--tokenizer <key> --revision <rev>` runs the pair-validity checks that need
only the rows and a tokenizer — every row's answers differ; every declaring
row carries its edit in tokens and no edit outside its spans — writes nothing
if a row fails, and prints the tokenizer it validated under — record it where
the table's command line is recorded, so a reader knows.

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

| Task | Description | Dimensionality | Ships (`<task>/data/<variant>`) |
|------|-------------|----------------|------|
| `natural_domains_arithmetic` | Unified weekdays/months/hours/age/integer/alphabet | factory, 1D (cyclic or linear) | `weekdays`, `months` — whole pool, `#train`/`#test` |
| `graph_walk` | Next-node prediction on graphs | factory, 1D or 2D | none yet: the answer is a *set* of neighbours, which the v1 row vocabulary (one answer + its forms) cannot carry |
| `entity_binding` | Positional entity retrieval | — | none yet: `output_tokens` is keyed by `positional_answer` (an index) with entity-name forms — declare it on `raw_output` first |
| `hierarchical_equality` | Hierarchical variable equality | — | `default` (256, no forms declared) |
| `identity_naming` | Entity → canonical name lookup (factory) | — | `pitch_midi` (256) |
| `MCQA` | Multiple-choice question answering | — | `default` — 192 rows, `#train` 128 / `#test` 64 (the onboarding demo's table) |
| `IOI` | Indirect object identification (coverage-oriented runner) | — | `default` (256) |
| `hex_color` | Hex-code → color-name mapping | — | `default` (256) |
| `subject_object_relations` | Subject→object relation recall (LRE-style) | factory | `word_first_letter` (256; `match` wants `"mode": "first_token"`) |

Define `equations` with `@mechanism` and named `V` assignments. See the [model guide](../../docs/causal-models.md) and [task examples](../../demos/causal_models/README.md).
