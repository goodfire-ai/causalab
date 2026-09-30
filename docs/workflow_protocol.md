# Workflow protocol

A workflow connects intervention runs and Python analysis through saved outputs.
This reference defines its fields, execution rules, and records. A protocol step
has `type: intervention_protocol`; it executes an experiment defined by the
[intervention specification](intervention_protocol.md), called the IM spec below.

Start with the [worked workflow](#10-worked-example--the-weekdays-8b-pipeline).
Set the data and model in its step documents, then inspect it with
`causalab explain <workflow> --engine auto`. Run with
`causalab run <workflow> --engine auto --out <root>`.
`--resume` reuses completed steps whose identities and contents still match.
The [internals page](workflow_protocol_internals.md) covers script resolution,
derived properties and the runner contract.

## 0. Principles

- Each step declares its inputs and outputs. Its digest covers the declared
  computation and script content (§7).
- The runner derives dependencies from references; `after` adds ordering.
  `explain` places independent steps at the same schedule level.
- Types are `intervention_protocol`, `script`, `behavioral`, `decision`,
  `conditional`, and `workflow`. They use the reference grammar in §3.
- Declared outputs remain in their step directories. JSON holds structured
  records; safetensors holds dense tensors. Figures use PNG, PDF, or HTML.
- The full graph must be finite, acyclic, and known at load time. Fan-out expands
  then. Conditionals choose which existing steps run. Nested workflows add steps
  to the same run.
- The parser accepts strict JSON and YAML. Unknown keys are errors.

## 1. Document layout

Use the following section order. Other orders warn and retain the same digest and schedule.

| # | key | required | content |
|---|---|---|---|
| 1 | `version` | ✓ | `"1"` |
| 2 | `description` | – | free text, the pipeline's intent |
| 3 | `output_dir` | ✓ | the workflow's own directory name: one path segment |
| 4 | `steps` | ✓ | the named step table: the whole pipeline |
| 5 | `measurement` | – | optional single-source or before/after study plan; see the [measurement guide](measurement.md) |

An authored `measurement` block enters the workflow's canonical form and digest.
It declares source revisions, cases, seeds, repeats, observations and profiling
settings. Machine paths and Python environments are supplied separately as
deployment bindings. Ordinary `run` refuses this block: `measure` owns collection.
Single-source mode uses `mode: "single"` and one `source`, with optional observations;
see the [single-source quickstart](measurement.md#quickstart-measure-one-commit).
Paired studies default to `comparison: "code"` and invoke this workflow at each
selected revision. With `comparison: "workflow"`, `before` uses this document and
`after.workflow` selects a relative workflow path; optional `eager.workflow`
selects a reference document. Selected documents have no measurement block, all
arms resolve to one code commit, and cases, observations and evaluation selections
are checked against each document. See the [workflow comparison quickstart](measurement.md#quickstart-compare-workflow-configurations).

Step names must be unique and match `[A-Za-z0-9_-]+`. They become directory
names. Generated children use `<step>@<i>`; nested steps use `<step>/<inner>`.
The CLI recognizes a workflow by its `steps` section.

### 1.1 `output_dir` and the run tree

`output_dir` is a single filesystem-safe path segment: not a nested path, not
absolute. The CLI supplies the root it sits under:

```
<out-root>/<output_dir>/<step>/<the step's declared outputs>
<out-root>/<output_dir>/<step>/_step.json      # the runner's per-step record
<out-root>/<output_dir>/<step>@<i>/            # child i of a fanned-out step, a step directory like any (§2.9)
<out-root>/<output_dir>/<step>/<inner>/        # a step of the workflow nested as <step>, a step directory like any (§2.10)
<out-root>/<output_dir>/<step>/.attempts/<inner>/<id>/  # its attempt, beside it under the nested workflow's own root
<out-root>/<output_dir>/workflow.json          # the run manifest
<out-root>/<output_dir>/events.jsonl           # the run's event stream, append-only (§4.3)
<out-root>/<output_dir>/.attempts/<step>/<id>/ # a step's attempt, until it is published (§8)
<out-root>/<output_dir>/.attempts/<step>/<id>/attempt.json  # what a failed attempt keeps
<out-root>/<output_dir>/.attempts/<step>/<n>.superseded/    # a prior unit a rerun displaced or a skip retired, retained (§8)
```

The runner writes a step into `.attempts/`, verifies it, then publishes its
directory with one rename (§8). Failed attempts and superseded outputs remain
there under the retention rules. References cannot address `.attempts/`.
A skipped step has no published directory.

Nested steps use `<step>/<inner>/`, with attempts under `<step>/.attempts/`.
The outer step name supplies the root; the inner `output_dir` is ignored.
The container is removed when all inner steps are skipped.

`output_dir` names a directory under the CLI output root. It is excluded from the digest.

## 2. Section reference

<a id="21-steps--common-fields"></a>

### 2.1 `steps`: common fields

Every step is an object with a `type` from the closed set
`intervention_protocol · script · behavioral · decision · conditional · workflow`, plus:

| field | meaning |
|---|---|
| `type` | ✓: the step vocabulary below |
| `description` | – free text |
| `after` | – step names that must complete first, beyond the derived data dependencies (pure ordering; rare) |
| `requires_receipt` | – `{"step": S, "outcome": "pass" \| "fail"}`: the receipt this step's allocation waits on (§2.8). In the canonical form only when authored |

Dependencies come from input references and run-tree loads in intervention
documents, including featurizer `file_path`, parameter `file_path`, and
`init.file_path`. `save` paths declare outputs. Use `after` for ordering
without data flow.

<a id="22-intervention_protocol-steps--run-one-intervention-specification"></a>

### 2.2 `intervention_protocol` steps: run one intervention specification

```json
"locate": {"type": "intervention_protocol", "document": "protocols/locate_scan.json"}
```

| field | meaning |
|---|---|
| `document` | ✓: path to an intervention specification, relative to the workflow file |
| `set` | – dotted-path overrides applied before loading, same syntax and semantics as the CLI `--set` (IM spec §9). Unlike the CLI form these are **part of the record**: they enter the canonical form and the digest |
| `max_points` | – override of the sweep point cap for this document (IM spec §5.14) |
| `execution` | – this step's row bounds, overriding the engine's for this step only: `batch_rows` (rows per no-grad forward) and/or `fit_rows` (rows per grad forward of a fit: the members of a fit cohort are packed into forwards under it; IM spec §8), each a positive integer or `null` for unbounded. Execution, not identity: unlike `set` and `max_points` it enters **neither** the canonical form nor the digest, and the step's `_step.json` is its one recorder. Unknown keys and non-positive values are refused (rule 1) |
| `control` | – the step **is a control** of another step: `{"of": <step>, "kind": <kind>, "seam": <seam>, "seeds": [...], "min_draws": n, "non_equivalence": {"fields": [...], "reason": "…"}}`: `of` and `kind` required, the rest optional (controls, below; `non_equivalence` names the equivalence fields the control knowingly differs from its target in, rule 16). In the canonical form only when authored |
| `waive` | – the controls this step waives, `{<kind>: <reason>}` or `{<kind>: {"reason": <reason>, "reference": <ref>}}` (controls, below). Only when authored |
| `stop_after_failure_rate` | – on a control step only: the fraction of its points that may fail certification before the certifying step is `failed` and its dependents `blocked` (§8); `0.0` unauthored. Only when authored |
| `fan_out` | – `{"over": {"axis": A} \| {"shards": N}, "join": {"require": "all" \| "selected"}}`: the step's declared fan-out (§2.9): its compiled points are partitioned into children `<step>@<i>` at load and its own name is their join. In the canonical form only when authored |

The loader uses `compile_protocol`: build, then validate (IM spec §9.1).
Validation defers checks on run-tree artifacts; execution resolves them through
the run-tree store. Both use the same compiler. The inner document's `save`
manifest declares outputs under the step directory.

Controls execute as protocol steps. A `control` declaration names its target
and comparison; required controls must be declared or waived (rule 14).
Dependent steps inherit their status (§8). Control seeds enter the workflow
step identity.

```json
"fit":     {"type": "intervention_protocol", "document": "protocols/das.json",
            "waive": {"self_swap": {"reason": "external", "reference": "runs/2026-09-01/self_swap"}}},
"control": {"type": "intervention_protocol", "document": "protocols/random_subspace_control.json",
            "control": {"of": "fit", "kind": "matched_random", "seeds": [0, 1, 2], "min_draws": 3}}
```

`CONTROL_KINDS` in `causalab.workflow.document` defines the four kinds below.
When a workflow declares any `control` or `waive`, each fit and control target
requires a `self_swap` and `matched_random` control or a waiver for each.
`full_component` and `shuffled_source` are optional. Workflows without control
declarations are exempt. Nested workflows have the limits in §2.10.

| kind | what the control document is | what is checked at load, against the compiled documents |
|---|---|---|
| `self_swap` | the bit-exact no-op leg: an intervened model whose every write swaps in a read taken on an un-intervened model, on the model's **own** input, at the write's own address: site, `pos`, `featurizer`, `dims` equal (IM spec §5 rule 21 admits equal depth). A same-target donor by construction: rows pair by index, so the operand is the target's own row | the predicate above holds for some intervened model of the document: refused naming the first field that fails; and a script step certifies it (below) |
| `matched_random` | the size-matched random control of a fit: an untrained `subspace` at the fit's `k` with `seed` swept (`demos/methods/protocols/random_subspace_control.json`), or a gate drawn by `causalab.analysis.random_mask` at a recorded seed and applied | `of` declares a fit; for every featurizer the fit trains, the control holds one of the same `kind`, `k` and `group`, written at the same site (unless `non_equivalence` declares the site field, rule 16); `seeds` are present, distinct, at least `min_draws` (20 unauthored: authored lower is *recorded* lower); and they are the draws the document makes: its featurizer's `seed`, or the `seed` of the script step that drew the bundle it loads |
| `shuffled_source` | a counterfactual role permuted under a declared seed: the label control: the target's document with `data.counterfactual.shuffle: {seed: <int>}` (IM spec §2.2) as its **one** difference: a seeded permutation of that role's row order (`random.Random(seed).shuffle`), the base role untouched. A permutation may leave **fixed points**: seed 0 over 4 rows gives `[2, 0, 1, 3]`, row 3 meeting its own base row; 9 of the 24 orders of 4 rows have none: and nothing excludes them; a certifier of this kind must account for them | the control's compiled document equals `of`'s canonical form with `shuffle` masked: refused naming the first differing field; at least one counterfactual role of the control authors `shuffle` and none of the target's does. Its points are `passed` when they ran, as `matched_random`'s are; its waiver is never required |
| `full_component` | the whole-component swap at the target's site: the parent of a learned intervention, through no featurizer and no `dims`; the full-component score provides a comparison; a sparse mask may score higher: its status is `passed` when the swap ran (a comparison against the fit is a reduction, §2.6) | every write its intervened models list goes through no featurizer (or only `identity`) and names no `dims`; and it is site-equivalent to its target (rule 16, below) with `featurizer`: and so `dims`, which index the featurized value: allowed to differ: that difference is the kind. Never required by rule 14 |

**Site equivalence.** Coverage controls (`full_component`, `matched_random`)
must match their target's sites across expanded points. The loader compares
the fields below. Each difference requires an entry in
`non_equivalence: {"fields": [...], "reason": "…"}`. Listing an equal field
also fails validation. Self-swaps certify by coordinates (§8).

| equivalence field | what it compares | decided from |
|---|---|---|
| `component` | the component name, pre/post-projection sites included: `attention_premix` (the o-projection's input, head space) is not `attention_output` (its output, the residual stream); `delta_premix` is neither | the site |
| `shape` | the component's tensor shape on the model (`component_shape`): two components of one name on two shapes are two spaces | the registry entry |
| `layers` | the band each point's site spans (IM spec §2.4), compared as the set of per-point bands: a control pinned to one layer against a target sweeping two is not equivalent, and a band `[3, 4]` at one point against a sweep over 3, 4 is not either: coverage-equal, intervention-different, since a band is one site and one intervention across its members; empty for a layerless component | the sites of every expanded point |
| `head` | the head a site names, or none: a head on a component with no head axis is said so | the site, bounded by the shape |
| `expert` | the routed expert a site names, or none | the site |
| `stream` | the mixer stream at each covered layer: DeltaNet inclusion: the site's declared `stream`, else the component's bound stream, else the registry's `layer_types` at the layer, else `full_attention` on a tower that declares no linear-attention mixer at all (it can carry no DeltaNet layer), else *unknown*: compared as such and said so | the site and the registry entry |
| `routed_rank` | `(num_experts, num_experts_per_tok, moe_intermediate_size)` on a component of the MoE block; none elsewhere | the registry entry |
| `featurizer` | the write's featurizer chain as shapes: each stage's `kind`, `k`, `parametrization`, `group`, `axis` and `units` (the axis a gate parameter indexes and how many units of it: a position gate's window length) and parameter `dtype`; **not** its `seed`, `init` basis or bundle bytes, which are values (the matched random control differs from the fit in exactly those) | the featurizers |
| `dims` | the coordinates of the featurized value the write covers: the authored `dims`, or every coordinate of the chain's output width when unauthored | the write and the shape |
| `sharing` | whether the two are expressed in **one** coordinate system: decided, not guessed: inside one document a featurizer *name* is one stage instance (`build_stack` caches by name), so two writes naming one featurizer share; across documents a name is only a name, and sharing is bytes: a document that loads the bundle another saves is scored in that other's basis. Listed as differing when the pair *shares*: a control in its target's own basis is no control | the featurizers, `file_path` and `save` |

Compiled model settings are checked separately. A script's effect on a bundle
is resolved during execution. A random mask drawn from a fit's bundle counts
as a distinct basis; input records preserve its source. The control record
stores the equivalence verdict.

A waiver names why, from a closed set (`WAIVER_REASONS`):

| reason | waives | when it applies |
|---|---|---|
| `no_fit` | `matched_random` | the target declares no `train`: there is no rank or mask to match. Refused on a fit |
| `single_role` | `shuffled_source` | the document has no counterfactual role to permute |
| `external` | any kind | the control ran elsewhere; **must** carry a `reference`: the run, artifact or document it lives in |

A control's status per point (`CONTROL_STATUSES`), and the two words a
dependent inherits (`INHERITED_STATUSES`; §8):

| status | meaning |
|---|---|
| `passed` | the point certified: for `self_swap`, all three legs below held. For `matched_random`, `shuffled_source` and `full_component` the word means only that **the declared draw (the permutation, the swap) ran**: the pairing was checked at load, the comparison against the target is a reduction (§2.6), never a status, and nothing numerical was decided; the word stays `passed` because it is what a dependent inherits and the vocabulary is closed |
| `failed` | the point did not certify; on the stream it is a `warning` with `reason: instrument_failure` (§4.3) |
| `waived` | the kind is waived on the target step, with its reason, so no point carries a status for it |
| `not_run` | no certified point of the control agrees with this point's coordinates, or the certifier has not run: and a `self_swap` point's word on its control's own record until it has |
| `instrument_invalid` | a dependent point whose control `failed` at the agreeing coordinates: inherited, never authored |
| `instrument_failure` | the control point's own word on the event stream when it `failed` |

A replay control may name the seam its failure rate is measured on
(`CONTROL_SEAMS`); the seam is recorded beside the rate and nothing on this
tree checks a tolerance against it:

| seam | what changes between parent and replay |
|---|---|
| `A` | nothing: in-process re-execution |
| `B` | the artifact round-trip: a rotation saved and reloaded |
| `C` | batch order: the same rows reversed |
| `R1` | a parametrization re-materialized from its pre-image: not exact under TF32; a control crossing it declares a non-zero `stop_after_failure_rate` |

**Self-swap certification.** A point passes when the receiver under self-swap
equals the original receiver bit for bit, the target sender differs from the
overwritten value, and the target receiver differs from the original receiver.
The difference checks use `not allclose`, with `atol=1e-4` unless the study
measures its own floor.

`causalab.analysis.certify_control` reads five bundles: the receiver under the
original, self-swap, and target models, plus sender and overwritten values.
It writes one row per point to `controls.json`. Each self-swap requires a
certifier that reads exactly one declared control and declares that output name.
The runner supplies the reserved `control` input. Independent write-oracle
agreement is a test-only check at `atol=1e-5, rtol=1e-4`.

**Qualification.** The loader schedules a control's certifier before its
target, or the control itself when certification is unnecessary. Qualification
covers the target's full rank and seed expansion. Results transfer by point
coordinates.

A `matched_random` control that reads its target's fitted bundle keeps that
post-fit dependency. An `after` edge alone is insufficient. Self-swap and
shuffled-source controls must precede their targets. Mutual control dependencies
fail the cycle check.

The compiled model must match between control and target, including revision,
dtype, quantization, and any declared attention backend. Changing both requires
new qualification. Batch bounds may differ. Qualification identity contains
the control document digest, implementation `tree_digest`, and engine name.
`--resume` checks all of them. A failed coordinate invalidates every target fit
at that coordinate.

Protocol steps support document validation, sweep expansion, and engine checks
at load time. Their point counts and axes support planning and later analysis.

<a id="23-script-steps--inputs-one-python-script-declared-outputs"></a>

### 2.3 `script` steps: inputs, one Python script, declared outputs

```json
"steer_direction": {
  "type": "script",
  "script": "scripts/harvest_difference.py",
  "inputs": {
    "acts_pos": {"step": "harvest_pos", "file": "acts.safetensors"},
    "acts_neg": {"step": "harvest_neg", "file": "acts.safetensors"},
    "normalize": true
  },
  "outputs": {"direction": "direction.safetensors", "stats": "stats.json"}
}
```

| field | meaning |
|---|---|
| `script` | ✓: a **locator**: `{"module": "causalab.analysis.fit_pca"}` or `{"path": "scripts/probe.py"}` (§2.4) |
| `inputs` | ✓: `{name: value}` in the §3 grammar; each reference **is** a derived dependency edge |
| `outputs` | ✓: `{slot: file}` under the step dir, non-empty |
| `runtime` | – dependency isolation (§4.1) |
| `reduction` | – the reduction contract (§2.6): what a number this step publishes *is*: statistical unit, grouping, weighting, missing-value policy, uncertainty procedure, resampling unit, repetitions, seed. Validated at load (§5 rule 12); in the canonical form and the digest **when authored** |
| `is_deterministic` | – default `true`; see §7 |

Use scripts such as `causalab.workflow.scripts.select`,
`causalab.io.plots.workflow_figures`, and modules in `causalab.analysis`.

Script steps run deterministic Python analysis. LLM judging belongs in the
research tooling that consumes the results.

#### The output declaration

```json
"outputs": {
  "spectrum": {"file": "spectrum.json",
               "columns": {"component": "int64", "explained": "float64"}},
  "weight": "basis.safetensors"
}
```

An output may use a bare filename. Optional `columns` declares a JSON table's
columns and enters the digest. The runner checks the declaration after the
script has run, before it publishes the step, with
`causalab.workflow.runner.verify_output`: every declared column must be in
the table's first row. An empty table passes any column declaration, a later
row is not read, and column dtypes are not checked. The runner resolves a
script's inputs with `causalab.workflow.runner.script_call`. Both functions
are public, so a test can run a script on sample inputs through the runner's
own resolution and check, before the model steps that feed it run;
`tests/demos/test_papers.py` does this for every paper package.

A `.json` output may instead declare **`keys`**: a flat *values object* rather
than a table, with one representative value per key:

```json
"outputs": {
  "values": {"file": "values.json",
             "keys": {"best_layer": 18, "best_pos": {"index": -1}}}
}
```

Choose `columns` for a table or `keys` for a values object. `keys` supplies
representative values for load-time checks of downstream references. Execution
recompiles with the emitted values. A run-tree `file_path` produces a
`deferred_check` diagnostic until it resolves.

### 2.4 Addressing a script, and `file` vs `path`

`script` is a **locator**, the same shape an `inputs` reference uses (§3):

| form | resolves to |
|---|---|
| `{"module": "causalab.analysis.fit_pca"}` | Module file found through `importlib.util.find_spec`; parent packages may be imported (§4.2). |
| `{"path": "scripts/probe.py"}` | File relative to the workflow, contained within its directory. |

The shipped scripts are filed **by subject**, not in one flat namespace:

| module | what it is |
|---|---|
| `causalab.analysis.fit_pca` · `harvest_difference` · `head_stats` · `paired_ttest` · `random_mask` · `subspace_angles` | numerical analysis: fits, statistics, controls (a size-matched random mask for a DBM fit), the principal angles between two saved subspaces, and the operands an intervention consumes |
| `causalab.analysis.project_pca` · `pca_by_position` · `sequence_activations` | reuse a frozen training mean/basis; fit separate position populations; gather predictive activations from a shared sequence harvest |
| `causalab.io.plots.workflow_figures` | rendering, beside the rest of `io/plots/` |
| `causalab.workflow.scripts.select` · `reduce` | the two scripts whose purpose *is* the seam between steps: `select` picks the values a later document's `set` reads; `reduce` publishes a number under a declared reduction contract (§2.6) |

**`file` vs `path`.** Both words appear in a document and they are not
interchangeable:

- `file` names a declared output inside a step directory. The runner supplies
  its root, as in `{"step": "harvest", "file": "acts.safetensors"}`.
- `path` names an external file, absolute or relative to the workflow document.

### 2.5 Formats: two that carry the record, three that visualize it

Use JSON for structured records and safetensors for dense tensors.
Metric tables are JSON arrays of row objects (`causalab.io.tables`):

```json
[
  {"example": 0, "sites.target.layers": 18, "value": 0.83},
  {"example": 1, "sites.target.layers": 18, "value": 0.91}
]
```

Each metric has its own table, with labels repeated on each row.
Non-finite numbers are written as JSON `null`.

PNG, PDF, and HTML outputs render results. They accept no `columns` or `keys`
and receive no `ArtifactIdentity`. The runner checks non-empty content and PNG
or PDF signatures. HTML is checked for non-emptiness (§8).

PNG is the default figure format. Use PDF for print or vector output and HTML
for interactive views. `causalab.io.plots.figure_format.normalize_figure_format`
applies the default. Save plotted numbers alongside figures; `workflow_figures`
supports a `plotted` table output.

<a id="26-reduction--the-reduction-contract"></a>

### 2.6 `reduction`: the reduction contract

A `reduction` declares how a script converts table rows into a reported value.
It specifies units, grouping, weights, missing values, and uncertainty.
Resampling declarations include their unit and seed:

```json
"facts": {
  "type": "script", "script": {"module": "causalab.workflow.scripts.reduce"},
  "inputs": {"table": {"step": "trace", "file": "aie.json"}},
  "reduction": {
    "estimator": {"kind": "mean"},
    "unit": {"kind": "example", "columns": ["fact"]},
    "group_by": ["sites.target.layers"],
    "weight": null,
    "missing": "exclude",
    "uncertainty": {"kind": "percentile_bootstrap",
                    "resample_unit": {"kind": "example", "columns": ["fact"]},
                    "repetitions": 2000, "seed": 42}
  },
  "outputs": {"table": {"file": "aie_by_layer.json",
                        "columns": {"value": "float64", "n": "int64"}}}
}
```

The example uses a fact-level percentile bootstrap with 2,000 resamples and
seed 42. Required fields include an explicit `weight: null` when unweighted.
Vocabularies are defined in `causalab.workflow.reduction`.

**The eight dimensions**

| dimension | field | value |
|---|---|---|
| statistical unit | `unit` | `{"kind": <unit>, "columns": [...]}`: the vocabulary member **plus** the column(s) that identify one observation in *this* table; only the campaign knows its key. `row` names no column |
| grouping | `group_by` | column names, one output row per distinct combination; `[]` for one row |
| weighting | `weight` | a column name, or `null` |
| missing-value policy | `missing` | one of the policies below |
| uncertainty procedure | `uncertainty.kind` | one of the procedures below |
| resampling unit | `uncertainty.resample_unit` | as `unit`; **may differ** from it (ROME resamples facts over fact × token rows): on a curve under a `unit`, not `row`, and each unit within one cluster (decided on the table) |
| repetitions | `uncertainty.repetitions` | a positive integer |
| seed | `uncertainty.seed` | a non-negative integer: what seeds the `numpy.random.Generator`; `random` is never consulted |

`estimator` selects the calculation. `quantile` also takes `q`; curve
estimators take the fields below. Choose a unit that matches the independent
observations in the study. Reused prompts can make pairs dependent.

| unit | one observation is |
|---|---|
| `row` | one row of the table; nothing is collapsed |
| `example` | one example: the `example_id` column a protocol run stamps per example (the base row's label, IM spec §2.2), or a campaign's own key (`fact`) |
| `pair` | one counterfactual pair: its base and counterfactual rows together |
| `prompt` | one prompt, across every pair that reuses it |
| `source_family` | one source family of prompts |
| `component` | one connected component of the prompt-reuse graph |

**What `unit` does.** Rows sharing the unit key are collapsed to one
observation before the estimator runs: by their mean (weighted, when a weight
is declared) for `mean`, `weighted_mean`, `median` and `quantile`; by their sum
for `sum`; `count` counts units. `n` counts observations, never rows.

| policy | a `null` value (or weight; on a curve, a null `x` too) |
|---|---|
| `error` | refuses the table at run time, naming how many and how many came from `matched: false` |
| `exclude` | is dropped before the estimator runs, and the count is recorded (`n_excluded`): what pandas' `skipna` did silently |
| `zero` | is scored `0.0`: the author's statement that "the model never said it" counts as a zero, not as absent; refused on a curve when the null is an `x` (an abscissa has no zero) |

| procedure | interval | takes |
|---|---|---|
| `none` | no interval | nothing: declaring `resample_unit`, `repetitions` or `seed` under `none` is refused: a field that governs nothing may not be declared |
| `percentile_bootstrap` | the 2.5th and 97.5th percentiles of the estimator over `repetitions` resamples, each drawing resample units **with replacement** and keeping each unit's rows together (a cluster bootstrap when `resample_unit` is coarser than `unit`; on a curve a resample unit that splits a unit is refused at run time); a unit drawn twice is two observations | `resample_unit`, `repetitions`, `seed` |
| `normal_approx` | estimate ± 1.96 × the sample standard deviation (`n−1`) of the estimator across resample units, over √k; fewer than two units gives `null` bounds | `resample_unit`; pairs only with `mean` / `weighted_mean` |

Intervals use 95% coverage. Each group's seed combines the declared seed
with a stable value derived from its coordinates.

| estimator | value | note |
|---|---|---|
| `mean` | arithmetic mean of the observations | `weight` must be `null` |
| `weighted_mean` | Σ w·v / Σ w over the observations | requires a `weight` column; a zero weight contributes nothing |
| `sum` | sum of the observations | the numerator of a mean composed across tables |
| `count` | the number of observations (units) | the denominator that makes `sum` composable |
| `median` | the lower of the two middle values at even `n`, **no interpolation** | the same choice as `save.reduce`'s `median`: one verb, one meaning |
| `quantile` | the `q`-quantile, linearly interpolated | `q` in (0, 1) |
| `auc` | the trapezoid area under the curve of per-`x` means of `y`, over the sorted distinct `x` | a curve estimator: `x` (required), `y` (optional, else the value column), `x_scale` (required: `linear` \| `log`), `normalize` (optional: `x` is divided by it: a positive number, or a column constant and positive within the group); `weight` must be `null`; one distinct `x` is refused (no area) |
| `cpr` | MIB's circuit-performance ratio: `faith(p) = (m(N − int(p·N)) − C)/(B − C)` over the kept-fraction grid `p`, integrated by trapezoid over the raw `p`; `B = m(0)`, `C = m(N)` | a curve estimator: `x` holds the **cuts** (units taking the counterfactual, `top_k`), `normalize` is **required** and is `N` (an integer ≥ 2, or a column holding it), `x_scale` `linear` (MIB) \| `log`, `grid` optional (default MIB's ten points `0.001 … 1.0`); `weight` must be `null` |

| scale | abscissa |
|---|---|
| `linear` | the abscissa as it is: `x` (÷ `normalize` when given) for `auc`, the raw kept fraction `p` for `cpr`; MIB's integral |
| `log` | its natural logarithm: `log x` for `auc` (an `x ≤ 0` is refused at run time: a cut of 0 has no logarithm), `log p` for `cpr`; equal weight per decade of the sweep |

**Curve estimators.** Rows represent observations at cuts, such as an example
at each `top_k`. Rows with the same unit key and x collapse to their mean;
`unit: row` preserves each row. `m(x)` is the mean over observations at x,
and `n` counts those observations.

`auc` integrates over sorted x, optionally divided by `normalize`. A log scale
requires positive x. `cpr` follows MIB's `evaluate_area_under_curve`: N counts
units, B is `m(0)`, and C is `m(N)`. At kept fraction p it reads cut
`N − int(p·N)`, then integrates `(m(cut) − C)/(B − C)` over p. `top_k` counts
units receiving counterfactual activations; p counts units retaining original
activations. The default grid spans 0.999 and the area retains that span.
Several p values can share a cut. Floating-point flooring matters: p=0.29 and
N=100 select cut 72.

An explicit unit requires a balanced panel: every unit must occur at every
required x, including CPR anchors and grid cuts. Excluding a null can break
this condition. Put other varying conditions, such as method, in `group_by`.

The x column cannot be a grouping, observation-unit, or resampling-unit key.
Each observation unit must lie within one resampling cluster. An explicit unit
forbids row-level resampling. Bootstrap draws then preserve whole curves and
recompute the area. With `unit: row`, CPR rejects lost cuts within a draw,
while AUC integrates the x values that remain.

Runtime checks reject missing CPR anchors or cuts, non-integer cuts, and
coincident anchors. The anchor tolerance is `1e-9` times the largest absolute
mean among required cuts and anchors. An x or normalization column cannot be
the reduced value column. Infinite x fails before grouping or missing-value
handling; infinite normalization fails for its group. Null x follows `error`
or `exclude`; `zero` is invalid for x.

Curves require `weight: null` and support percentile bootstrap.
`normal_approx` applies to means. A bootstrap failure reports its repetition.
For a GPT-2 IOI sweep, declare CPR as follows:

```json
"cpr": {
  "type": "script", "script": {"module": "causalab.workflow.scripts.reduce"},
  "inputs": {"table": {"step": "apply", "file": "ld.json"}},
  "reduction": {
    "estimator": {"kind": "cpr", "x": "axes.cut", "x_scale": "linear", "normalize": 156},
    "unit": {"kind": "example", "columns": ["example"]},
    "group_by": [], "weight": null, "missing": "exclude",
    "uncertainty": {"kind": "none"}
  },
  "outputs": {"table": {"file": "cpr.json"}}
}
```

`save.reduce` runs during the forward pass and saves reduced activations.
Workflow `reduction` operates on saved tables and specifies statistical units
and uncertainty. Shared estimator names keep the same meaning.

Output rows contain group coordinates, `value`, and counts: `n` observations,
`n_rows` input rows, `n_missing` null values or weights, `n_unmatched` missing
rows marked `matched: false`, and `n_excluded` rows removed by policy.
Intervals add `lower` and `upper`. Eligibility thresholds are applied separately.
The table's unit columns identify observations.

The runner passes `inputs["reduction"]` and records it in `_step.json` and
`workflow.json`. That input name is reserved. The built-in
`causalab.workflow.scripts.reduce` implements these estimators. Its `value`
input selects a column and cannot accompany a curve's `estimator.y`.
Custom scripts receive the same validated declaration.

**What `select` does, declared.** The shipped `select` (and the `plot`
renderer) reduce through `causalab.io.step_record.aggregate`, whose
arithmetic is the implied reduction

```
{estimator: mean, unit: row, group_by: <the producer's sidecar axes>,
 weight: null, missing: exclude, uncertainty: none}
```

This is `implied_reduction` in `causalab.io.step_record`. On windowed tables,
examples with more positions receive more weight. Tables without `example_id`
and axes pass through unchanged. Apply a declared reduction before selection
to use another unit or estimator. Protocol steps use `save.reduce`.

The loader checks the declaration. At runtime the reducer verifies referenced
columns and names the field responsible for an invalid reference.

Optional `estimand_version` identifies the calculation as `<estimand>/v<n>`.
Rule 13 checks that it matches the estimator and options. It enters the digest
when declared. Otherwise outputs use `<estimator>/v1`. The vocabulary lives
in `causalab.protocol.estimand`.

| identifier | computed by | arithmetic |
|---|---|---|
| `mean/v1` | `mean` | arithmetic mean of the observations |
| `weighted_mean/v1` | `weighted_mean` | Σ w·v / Σ w over the observations |
| `sum/v1` | `sum` | sum of the observations |
| `count/v1` | `count` | the number of observations |
| `median/v1` | `median` | the lower middle observation |
| `quantile/v1` | `quantile` | the `q`-quantile, interpolated |
| `auc/v1` | `auc` | the trapezoid area under the per-`x` means of `y` over the sorted `x`, against the block's declared `x_scale` (`x ÷ normalize` when given; `log x` under `x_scale: log`): the grid and scale are the block's, carried by its digest, not by the name |
| `cpr/v1` | `cpr` | MIB's CPR arithmetic over the block's declared kept-fraction grid `p` (MIB's ten points unless a `grid` is authored) and `x_scale`: the trapezoid of `(m(N − int(p·N)) − C)/(B − C)`, `B` the mean at cut 0, `C` at cut `N`: MIB's number on MIB's grid, the block's otherwise |
| `mean_of_eligible_row_ratios/v1` | `mean`, `unit: row`, `weight: null`, `missing: exclude` | per row, numerator ÷ denominator (the value column); then the mean over the **eligible** rows: the rows `exclude` kept, `n` against `n_excluded`; every row weighs the same |
| `ratio_of_sums/v1` | `weighted_mean`, `unit: row`, a `weight` column | Σ numerator ÷ Σ denominator over the same rows: the value column is the per-row ratio and the weight column its denominator; rows weigh by denominator |

`mean_of_eligible_row_ratios/v1` weights ratios equally.
`ratio_of_sums/v1` weights them by their denominators. The identifier must
match the declared calculation.

Output rows retain the input `unit` and resolved `estimand_version`.
`count` uses unit `count`; an unknown unit is `null`. Mixed units fail validation.

`causalab.analysis.paired_ttest` requires matching units. Its comparison label
is `arm` when estimands match or are undeclared, and `version` when both declare
different estimands. `estimand.Claim` and `check_claim` bind a claim to a file,
row selector, estimand, unit, and value (IM spec §2.10).

<a id="27-behavioral-steps--the-declarative-behavioral-runner"></a>

### 2.7 `behavioral` steps: the declarative behavioral runner

```json
"qualify": {
  "type": "behavioral",
  "document": "protocols/qa_probe.json",
  "set": {"data.base.dataset": "qa/data#development"},
  "decoding": {"mode": "sampled", "seed": 7, "temperature": 1.0, "top_p": 1.0},
  "checker": {"task": "natural_domains_arithmetic", "task_cfg": {"domain_type": "weekdays"}},
  "split": "development",
  "thresholds": {"min_examples": 10000, "min_valid_rate": 0.95, "min_correct_rate": 0.8},
  "retain": {"generations": {"max_rows": 1000}},
  "decision": {"on_pass": "advance", "on_fail": "narrow"}
}
```

A behavioral step generates responses and scores them with the task's
`ScoringSpec`. It runs a no-intervention document that reads generated positions,
such as `demos/methods/protocols/probe_variable.json`. The document specifies
prompts and reads; the step declares the fields below.

| field | meaning |
|---|---|
| `document` | ✓: path to the no-intervention document, relative to the workflow file; it declares at least one `generated` position |
| `set` | – overrides, as on a protocol step (§2.2); in the digest |
| `max_points` | – the sweep point cap, as on a protocol step |
| `decoding` | ✓: how the continuation is decoded (table below); the first place a decode seed exists in a record |
| `checker` | ✓: `{"task", "task_cfg"?}`: the task whose `ScoringSpec` grades the generations; `task_cfg` is a factory task's config. **Never a sixth spelling of correct**: `string_mode` is read from the spec (and, when the split's rows record one, held to it) and recorded, never authored |
| `split` | ✓: the split purpose (table below); the document's base ref must select the same fragment (`qa/data#development`) |
| `thresholds` | ✓: `{"min_examples", "min_valid_rate", "min_correct_rate"}`, declared numbers; every rate the decision reads is a count over `n`, and no statistic is re-implemented here: a reduction is §2.6's |
| `retain` | – `{"generations": "all"}` or `{"generations": {"max_rows": n}}`: how many raw generations the step keeps in `continuations.json`. **Bounded by default** (`max_rows` 1000): preserved generations are otherwise unbounded output, and an unbounded retention must be spelled `"all"` |
| `decision` | ✓: `{"on_pass", "on_fail"}`, each from the decision table below: what a passing or failing qualification does to the question |
| `fan_out` | – the step's declared fan-out (§2.9), as on a protocol step: each child decides over its own points and the join decides once over the summed counts. In the canonical form only when authored |

**Decoding.** The document alone specifies a greedy decode; a behavioral step
says so, or says how to sample:

| mode | fields | meaning |
|---|---|---|
| `deterministic` | none | argmax at each step |
| `sampled` | `seed` ✓, `temperature` (default `1.0`, `> 0`), `top_p` (default `1.0`, in `(0, 1]`) | each token is one draw from `softmax(logits / temperature)` restricted to the smallest set whose mass reaches `top_p`, from a generator seeded once per decode window with `seed`. The same seed under the same batch geometry draws the same tokens; sampled decoding is **not** bit-reproducible across geometries on a GPU, so "same seed ⇒ same bytes" is a CPU-fixture claim and a differing-geometry run is recorded, not gated. Only the reference engine samples: a `sampled` step routed to the `nnsight` engine is refused before its model loads, naming the engine and `deterministic` |

Both modes accept optional `eos_token_ids`, a nonempty list of distinct
nonnegative token IDs. The PyTorch engine otherwise uses the model generation
configuration's EOS IDs, then the tokenizer's EOS. It stops each row at its
first matching EOS, pads post-stop slots, and stops model calls once the whole
batch has terminated. Explicit EOS settings enter the behavioral step identity.
See [the behavioral execution recipe](behavioral_analysis.md) for production
validation and exact records.

**Outcomes.** Every generated row lands in exactly one of four **terminal
outcomes**, a closed vocabulary held to this table by
`tests/workflow/test_behavioral.py`; the order is the derivation's
precedence:

| outcome | means | derived from |
|---|---|---|
| `truncated` | the row never emitted EOS inside the decode budget | the row's width equals the budget (`Continuation.widths`); takes precedence over every other outcome |
| `no_final_answer` | the continuation ended, but the answer variable's value appears nowhere in it | the `{"generated": …, "variable": <answer>}` anchor resolves to no steps |
| `invalid_format` | the value appears, but the graded string is not a declared form under the spec's string mode (`exact` / `prefix`): or the continuation is empty | the spec's own grader matches no declared value; a width of `0` |
| `valid` | the graded string is a declared form | graded `correct` / `incorrect` by the spec, the `per-example grade` vocabulary of the IM spec §2.10 |

`split` must match the base dataset reference's fragment: for example,
`development` with `qa/data#development`.

| purpose | meaning |
|---|---|
| `development` | the split a question is developed on: iterate freely |
| `reserve` | held back while developing; spent once, to check a development result before committing |
| `confirmation` | the confirmatory split: the decision a report rests on |

The split enters step identity. Changing it causes a resumed step to run again.

**Decisions.** The step writes a `DecisionRecord` for downstream conditionals:

| decision | meaning |
|---|---|
| `advance` | the question stands: proceed on it |
| `revise` | the question stands but its framing or apparatus does not: change it before proceeding |
| `narrow` | the question is too wide: restrict its scope |

**What the step publishes** (§8): the document's own `save` files, plus
`continuations.json`: the engine's raw generations, one row per generated
row with `point`, `point_digest`, `model`, `input`, `example_id` (the row's
label, IM spec §2.2), `steps`,
`split`, exact unpadded `input_ids`, `width`, `truncated`, the real `token_ids`, `emitted_ids` (including terminal
EOS), `terminal_eos_id`, `padding_ids`, `stop_reason`, effective `eos_token_ids`,
`decoding`, `greedy_token_id` (null for sampled decoding), `text` and per-token char
`offsets`, bounded by `retain`: `outcomes.json`: one row per example with
its `outcome`, `grade` (for a `valid` row), `expected`, `width` and `steps` :
and `decision.json` beside `_step.json`: `decision_type`, `schema_version`
(`1`), `measured_inputs` (`n`, the counts per outcome and `correct`,
`valid_rate`, `correct_rate`), `rule` (the thresholds block verbatim),
`outcome` (`pass` iff `n ≥ min_examples`, `valid_rate ≥ min_valid_rate` and
`correct_rate ≥ min_correct_rate`, else `fail`), `evidence_identity` (the
sha256 of `outcomes.json` joined to the step identity), `split` and `step`.
`continuations.json` is a request-keyed engine output written when the
request declares a `decoding`, not a `save` kind (IM spec §2.12 is
unchanged); nothing about a behavioral step enters the inner document, its
canonical form or its digest.

Expected cohort sizes are 10,000 single-input examples or 1,000 pairs.
Smaller cohorts warn and are recorded in `cohort`. Declared thresholds still
determine the outcome.

<a id="28-decision-and-conditional-steps--typed-decisions-gate-the-graph"></a>

### 2.8 `decision` and `conditional` steps: typed decisions gate the graph

```json
"gate_k": {"type": "decision",
           "values": {"step": "best_fit", "file": "values.json"},
           "rule": {"best_k": {"le": 16}, "best_seed": {"in": [0, 1, 2]}},
           "decision": {"on_pass": "advance", "on_fail": "narrow"}},

"gate":   {"type": "conditional",
           "predicate": {"decision": {"step": "qualify"}, "field": "outcome", "eq": "pass"},
           "on_true": ["fit", "apply"],
           "on_false": ["narrow_probe"],
           "scope": "global"},

"fit":    {"type": "intervention_protocol", "document": "protocols/fit.json",
           "requires_receipt": {"step": "qualify", "outcome": "pass"}}
```

A `behavioral` or `decision` step writes a typed `DecisionRecord`.
A conditional uses it to select downstream steps. Use `decision` to apply
predeclared thresholds to values from an analysis script.

**The `decision` step** reads one values object through the §3 grammar and
holds each named key to one clause:

| field | meaning |
|---|---|
| `values` | ✓: a §3 step reference to a `.json` values object (`{"step": S, "file": F}`), with no selector: the `rule` names the keys it reads. Rule 4 holds the file and every key to the producer's `outputs.<slot>.keys` declaration |
| `rule` | ✓: one clause per declared key, `{<key>: {<comparator>: <literal>}}`: exactly one comparator from the table below with a JSON literal operand (`in` takes a list). No expression language, no arithmetic, no reference to another step. `pass` iff every clause holds |
| `decision` | ✓: `{"on_pass", "on_fail"}`, each from §2.7's decision table: what a pass and a fail do to the question |

| comparator | holds when |
|---|---|
| `eq` | the measured value equals the literal |
| `ne` | the measured value differs from the literal |
| `lt` | the measured value is a number below the literal |
| `le` | the measured value is a number at most the literal |
| `gt` | the measured value is a number above the literal |
| `ge` | the measured value is a number at least the literal |
| `in` | the measured value is one of the listed literals |

`decision.json` contains the rule's measured keys, `rule`, `outcome`,
`decision_type`, `schema_version: 1`, and `step`. `evidence_identity` hashes the
values file's bytes with its producer identity, joined by `:`. A conditional
or receipt consumes the decision by name.

**The `conditional` step** decides between two sides of the graph:

| field | meaning |
|---|---|
| `predicate` | ✓: `{"decision": {"step": S}, "field": F, "eq" \| "ne" \| "in": L}`: `S` a `behavioral` or `decision` step, `F` from the decision-field table, `L` from that field's closed vocabulary (a list for `in`) |
| `on_true` | ✓: the steps that run when the predicate holds: a non-empty list of declared step names |
| `on_false` | ✓: the steps that run when it does not: non-empty, disjoint from `on_true` |
| `scope` | ✓: what the verdict decides for (table below) |

| decision field | vocabulary |
|---|---|
| `outcome` | `pass` · `fail`: the boolean the producer's rule decided |
| `decision_type` | `advance` · `revise` · `narrow` (§2.7): the authored consequence |

| scope | executes | meaning |
|---|---|---|
| `global` | ✓ | one verdict for the run: the side not chosen is skipped whole |
| `per_target` | ✓ | one verdict per target: the producer is fanned out over an axis under `sites.` and the conditional expands with it (§2.9) |
| `per_variable` | ✓ | one verdict per variable: the producer is fanned out over an axis under `positions.` and the conditional expands with it (§2.9) |

A conditional reads `decision.json` and checks schema version 1. A true
predicate skips `on_false`; a false predicate skips `on_true`. Skips propagate
to dependents. Both sides must follow the conditional and be dependency-disjoint.
Its record contains `predicate`, `scope`, `verdict`, `evidence`, and `skipped`.
It publishes no data file.

A skipped step has no published directory. Its manifest `skipped_by` contains
`"conditional"`, `"decision_step"`, `"decision_type"`, `"outcome"`,
`"evidence_identity"`, and `"transitive_from"`. The stream records `phase_started` and
`phase_completed` with `status: skipped`. A prior unit is retained as superseded.

Per-target and per-variable conditionals require a fanned-out behavioral
producer and gated steps with the same axis. The loader creates one conditional
per child: `gate@i` reads `P@i` and gates `S@i`. `per_target` uses an axis
under `sites.`; `per_variable` uses one under `positions.`. Shards cannot
define either scope.

The graph is fixed at load time. Runtime verdicts select its executed steps.
The stream records a direct skip at the step's turn; until then it can remain
pending. Transitive skips propagate through references and `after`.

**`requires_receipt`.** A step may require a behavioral or decision record
with a specified outcome:

| field | meaning |
|---|---|
| `step` | ✓: a `behavioral` or `decision` step of this workflow, scheduled before this one (a derived edge) |
| `outcome` | ✓: the outcome `S`'s `decision.json` must carry: `pass` · `fail` |

Receipt checks precede attempt creation and engine selection. Missing and
mismatched receipts produce distinct rule-18 errors, failing the step and
blocking dependents. Reuse checks the producer's current outcome too.

Evidence identities are computed from saved data. Reuse requires a matching
current producer identity for a conditional, and matching values-file bytes
and producer identity for a decision. Changed evidence triggers evaluation.
A receipt failure preserves earlier published files; the manifest records the
failed run. Later qualifying runs can reuse or supersede them.

<a id="29-fan_out--a-declared-fan-out-and-its-join"></a>

### 2.9 `fan_out`: a declared fan-out and its join

```json
"qualify": {"type": "behavioral", "document": "protocols/qa_probe.json",
            "decoding": {"mode": "deterministic"}, "checker": {"…": "…"}, "split": "development",
            "thresholds": {"…": "…"}, "decision": {"on_pass": "advance", "on_fail": "narrow"},
            "fan_out": {"over": {"axis": "sites.target.layers"}, "join": {"require": "all"}}},

"gate":    {"type": "conditional",
            "predicate": {"decision": {"step": "qualify"}, "field": "outcome", "eq": "pass"},
            "on_true": ["apply"], "on_false": ["probe"], "scope": "per_target"},

"apply":   {"type": "intervention_protocol", "document": "protocols/apply.json",
            "fan_out": {"over": {"axis": "sites.target.layers"}, "join": {"require": "selected"}}},

"fit":     {"type": "intervention_protocol", "document": "protocols/fit.json",
            "fan_out": {"over": {"shards": 4}, "join": {"require": "all"}}}
```

A protocol or behavioral step can partition its compiled points with
`fan_out`. The loader creates children and uses the parent's name for their
join. The width must be known at load time.

| over | children |
|---|---|
| `axis` | `{"axis": A}`: one child per value of `A`, an axis the step's compiled document expands (`sites.target.layers`, `positions.tap`, a named axis `axes.<name>`), in compiled coordinate order; a child holds the points whose coordinate on `A` equals its value. Refused naming the axis and the ids the document does expand |
| `shards` | `{"shards": N}`: `N` contiguous ranges of the compiled index list, sizes differing by at most one, the first ranges longer; `2 ≤ N ≤` the point count, a literal integer (never `"auto"`, never a `bool`) |

| require | the join publishes when |
|---|---|
| `all` | every child published every one of its points; a child a verdict skipped skips the join with it (§2.8's transitive rule) |
| `selected` | every child a per-child conditional (§2.8) left unskipped published every one of its points; the skipped children are named on the receipt. Declared only on a step such a conditional gates; a join every child of which is skipped is skipped with them |

Fan-out requires an explicit `join`. A failed child blocks it.

Children use names `<step>@<i>` and the parent's compiled document. Their
`shard` identifies the index, width, axis value or range, and points. They run
after the parent's dependencies, followed by the join. References name the join.
The parent entry records `fan_out`; a child's identity adds its shard.
Child records retain sliced points, digests, and coordinates, with their engine
and execution settings.

**Join checks.** The join places metric rows by their coordinate columns,
continuations by `point_digest`, and diagnostic side-table rows by digest-valued
`point`. It uses the children's published coordinates and identities because
deferred inputs can change them during execution. Load-time expansion fixes
the point count.

Records must align `coords` with `point_digests`. Duplicate digests or coordinates,
foreign coordinates, missing point labels, and invalid child-local indices
produce errors. Diagnostics for skipped coordinates use the load-time values.

Missing and duplicate points produce distinct rule-19 errors. A failed join
retains its attempt and blocks dependents.

The join writes rows in parent-point order and rebases behavioral indices.
An index that conflicts with its digest fails validation. Metric and continuation
files must cover required points. `train_eval.json`, `fit_diagnostics.json`,
and `routing_mismatch.json` are sparse and can be absent from individual children.

A fanned-out document cannot claim these diagnostic filenames or save a
safetensors bundle or `location_ledger`. Joins support measurement tables.
They hold all joined tables in memory, and child copies remain on disk.
Fan-out therefore bounds forward-pass memory while join memory remains the
size of the full table set. The parent's receipt contains these fields:

| key | value |
|---|---|
| `identity` | the digest of the parent's canonical entry, `fan_out` included: not the inner document digest an unfanned protocol step carries |
| `document`, `document_digest`, `method` | the parent's, as any protocol step's |
| `points`, `point_digests`, `coords`, `axes` | the parent's **full** lists: what `select` groups by, so a downstream reference reads a fanned-out step exactly as an unsharded one; a published point's `coords` are its child's, a skipped slot's the load-time expansion's (as its digest) |
| `fan_out` | `{"over", "width", "children"}`: the declaration and the children it derived |
| `join` | `{"require", "consumed": {child: {"identity", "points", "digests"}}, "skipped": [{"child", "skipped_by"}] (selected only), "n_points", "n_missing": 0, "n_duplicate": 0}`: the counts are always `0` on a published receipt (a non-zero is a refusal), recorded so a reader sees the check was made |
| `decision` | behavioral parent only: `decision.json`, written over the **summed** counts of the children through the one writer (§2.7), with `evidence_identity` the sha256 of the joined `outcomes.json` joined to the join's identity; beside it the summed `outcomes`, `retain` and `cohort`, and `thresholds`, `checker`, `split` as on any behavioral record |

A selected join retains the full parent point list, but output tables contain
only published children. `join.n_points` counts published points and
`join.skipped` names the rest. Skipped slots retain load-time digests.
Controls cannot be combined with fan-out (§5.19).

Engine and execution settings stay on child records. A conditional join
records child `verdicts`, joined `evidence`, and skipped steps, with `files: []`.

Reuse requires matching consumed child identities, points, and digests, with
skipped children still absent. Otherwise the join rebuilds, reusing individual
children where possible.

The runner executes children sequentially. `explain` reports the width for
external dispatchers that assign work to devices.

<a id="210-workflow--a-nested-reusable-workflow"></a>

### 2.10 `workflow`: a nested reusable workflow

```json
"tail":   {"type": "workflow", "document": "workflows/locate_and_fit.json",
           "set": {"fit": {"sites.target.layers": [12]}}, "after": ["baseline"]},
"report": {"type": "script", "script": {"module": "causalab.io.plots.workflow_figures"},
           "inputs": {"table": {"step": "tail/fit", "file": "iia.json"}}, "outputs": {"figure": "iia.png"}},
"apply":  {"type": "intervention_protocol", "document": "protocols/apply.json",
           "requires_receipt": {"step": "tail/qualify", "outcome": "pass"}}
```

A `workflow` step loads another workflow under the same validation rules.
Its steps join the current run; `workflow_digest` binds the inner document.
Use `set` to change its step settings.

| field | meaning |
|---|---|
| `document` | ✓: path to a **workflow** document (one with a `steps` section, §1), relative to the workflow file; an intervention specification here is refused (rule 20) |
| `set` | – the nested form `{"<inner step>": {"<dotted path>": value}}`: laid over the named inner step's own `set` (outer wins, key by key) **before the inner document is parsed**, so rule 8 checks every path against the intervention specification exactly as it checks an authored one. Names a document step (`intervention_protocol` · `behavioral`) of the inner workflow; an unknown name, or a `script`, `decision`, `conditional` or nested `workflow` step, is refused naming `set.<inner>`. In the canonical form only when authored |
| `requires_receipt` | – as on any step (§2.8): checked before any step of the nested workflow is allocated; a failure is that step's failed attempt, naming this field |
| `after` | – as on any step: every step of the nested workflow runs after the named steps |

Nested names use `<step>/<inner>`, with `/` for further nesting and `@` for
fan-out. Paths resolve to their longest matching step prefix. Outer references,
ordering, conditionals, decisions, and receipts can name inner steps.
A container name can appear in `after` or conditional sides, but cannot name
a file or receipt. Inner references resolve within their own document; outer
values enter through `set`.

Nested steps publish under `<run_root>/<step>/<inner>/`, with attempts under
`<step>/.attempts/`. Their references resolve from that root. The inner
`output_dir` is ignored. One event stream and manifest use flattened step names.
Containers have no attempt, status, or record.

The canonical container entry contains `type`, `document`, and
`workflow_digest`, plus declared `set`, `requires_receipt`, and `after`.
The inner digest is computed after overrides. Inner steps retain individual
identities for reuse. Inner descriptions affect the digest; whitespace and
`output_dir` are excluded.

Rule 20 rejects recursive inclusion, the wrong document type, invalid override
targets, and file or receipt references to containers. Per-child conditionals
must be declared in the same document as their producer and fan-out.

Controls remain within one root. Nested documents cannot declare `control` or
`waive`, and outer controls cannot target nested steps. An outer workflow that
enables controls cannot contain a nested fit. Containers cannot declare fan-out;
inner steps can. All inner steps are exposed by their names.

<a id="3-cross-step-wiring--the-reference-grammar"></a>

## 3. Cross-step wiring: the reference grammar

**A reference is a locator plus an optional selector.** One grammar, used by
every `inputs` entry.

| locator | resolves to |
|---|---|
| `{"path": P}` | a file on disk: **absolute** if `P` starts with `/`, otherwise **relative to the workflow document's directory** |
| `{"step": S, "file": F}` | the file `F` that step `S` declares, in the run tree |

| selector | requires | yields |
|---|---|---|
| *none* |: | the resolved absolute path: the script opens it itself |
| `"key": K` | the locator names a `.json` | the scalar at `K` inside it |
| `"entry": {…}` | the locator names a `.safetensors` | one tensor of a bundle, by coordinate match (IM spec §2.5) |

Anything carrying none of these keys is a **JSON literal**, passed through
unchanged. References are recognized **only at the top level of an `inputs`
entry**, so a nested object is always a literal: `{"cfg": {"step": 3}}` is
unambiguous and the loader never guesses.

```json
"inputs": {
  "layer_in_run":  {"step": "best", "file": "best_cell.json", "key": "best_layer"},
  "layer_on_disk": {"path": "/path/to/fits/best_cell.json",   "key": "best_layer"},
  "layer_in_repo": {"path": "configs/pinned_cell.json",       "key": "best_layer"},
  "k": 8
}
```

A tagged `path` distinguishes an external file from a string value. Step
references name declared files whose roots the runner supplies. References to
fan-out use the join; nested references use `<step>/<inner>`.

`key` reads a scalar from JSON. `entry` selects a safetensors entry by coordinate
values while preserving the file reference. Rendered bundle-key order is
irrelevant. `file` is required for step references. Intervention documents and
their `set` overrides use the IM spec's `artifact` and `file_path` grammar.
References and `after` define the acyclic dependency graph.

## 4. Execution semantics

- Steps run in a topological order of the derived graph; each step's outputs
  land under `<step>/`.
- A protocol step executes on the run's one engine (`--engine`; IM spec §8),
  held to the capabilities derived from the union over the inner document's
  points (`route_engine`) before it loads a model; a step the engine cannot
  serve is refused `[V13]` naming the shortfall.
- Before the first step, the runner resolves the metric answers of every
  inner document that compiles at load, with each model's tokenizer (IM spec
  §2.10). A table the tokenizer cannot score refuses the run, naming the step,
  and nothing is written. A document that depends on an earlier step's output
  is resolved at its own step, before the engine is handed it. Each model's
  tokenizer loads once for the run. Under `--resume` the check before the
  first step does not run (§7): each step that is attempted, and not reused,
  resolves its answers at its turn.
- A script step is invoked **in-process by default**:

  ```python
  def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None: ...
  ```

The runner verifies each script output and stamps safetensors with identity
inherited from tensor inputs, plus engine and dtype. It writes `_step.json`
inside the verified attempt, then publishes the directory.

Protocol records include axes, point digests, and execution bounds `batch_rows`
and `fit_rows` (`null` for unbounded). `fit_rows_resolved` records the measured
bound; `fit_rows_shrinks` records repacking. A `ragged` block, when used, records
each write's policy, row widths, and width buckets.

The runner schedules dependencies; engines execute on devices. External tools
dispatch jobs. Workflow children execute sequentially. Protocol `--points`
supports external shard dispatch. Nested steps use the same execution rules.

<a id="41-runtime--dependency-isolation"></a>

### 4.1 `runtime`: dependency isolation

```json
"runtime": {"isolate": true, "deps": ["umap-learn>=0.5"]}
```

An isolated script runs in a subprocess using
`uv run --no-project --python <interpreter> --with <deps>`.
Declared dependencies take precedence over the runner's environment while
preserving its causalab install. `env` contains variable names; values stay
outside the document. The runner's environment remains unchanged.

`runtime` enters the canonical form and digest. Device placement is an execution setting.

### 4.2 Script resolution and the torch-free guarantee

`validate` and `digest` hash a script without importing it or any numerical
library. The [internals
page](workflow_protocol_internals.md#42-script-resolution-and-the-torch-free-guarantee)
describes how a script and its sibling imports are resolved and hashed.

### 4.3 Event stream

`events.jsonl` records execution history beside the run manifest.
`workflow.json` records the resulting state. Appended events leave scientific
output checksums unchanged. A protocol run writes the same stream beside
`protocol.json` when asked to record (`causalab run --record`); by default it
writes neither.

Events contain `schema_version: 1`, `event`, gap-free increasing `seq`, UTC
`ts`, and `payload`. Resume continues the sequence. Protocol events also carry
`document_digest` and `[start, stop]` shard bounds. Workflow payloads name steps.

```
{"schema_version": 1, "event": "result_committed", "seq": 2, "ts": "2026-09-03T12:00:00.000000+00:00", "payload": {"step": "fit", "files": ["basis.safetensors"]}}
```

The event vocabulary is **closed**: seven names, held to `EVENTS` in
`causalab/io/events.py` by a census test (`tests/io/test_events.py`); the
writer refuses any other name, and so does the reader:

| event | when | payload |
|---|---|---|
| `phase_started` | workflow: a step's turn begins, before the reuse decision · document: before the engine runs | workflow: `step`, `type` · document: `phase`, `n_points` |
| `progress` | document: a point has run: reported once the engine returns, one line per point in run order | `point_digest`, `index`, `completed`, `total` |
| `metric` | document: a metric was summarized for a point | `point_digest`, `name`, `value`: the value `explain` prints, never a new fact |
| `warning` | workflow: an attempt failed and is retained under `.attempts/` · workflow: a control point failed certification (§2.2, §8) · either: the event sink raised | `step`, `reason: attempt_failed`, `error` · `step` (the control), `reason: instrument_failure`, `point` (its digest), `coords`, `certified_by` · `reason: sink_failed`, `event`, `seq`, `error` |
| `result_committed` | workflow: the **publish** moment (§8): the verified attempt was renamed onto `<step>/` · document: the engine's files are written | workflow: `step`, `files` · document: `files` |
| `phase_completed` | workflow: a step was published, reused or skipped (§2.8) · document: the engine returned | workflow: `step`, `status`, for a protocol step `forwards`: the forward groups its engine ran (§8), so the stream shows a qualification ran once for its target's whole fanout: and for a skipped step `skipped_by` · document: `phase`, `forwards` |
| `campaign_terminal` | workflow: after the manifest is written by a run that ran to its end: every step completed, or a step failed (an interrupted run writes its manifest and no terminal line) · document: the last line of a finished run | workflow: `outcome` (`completed` / `failed`), `steps` · document: `outcome` |

`causalab.io.events.terminal(path)` checks for a final `campaign_terminal`.
Interrupted runs omit it. Readers and writers reject incomplete lines,
sequence gaps, and duplicates.

`run_workflow` and `run_protocol` accept an optional event sink. It receives
each event after the local write. Exceptions produce a local `sink_failed`
warning, withheld from the sink. Outputs and receipts remain unchanged.
The CLI provides no remote sink option.

`causalab.workflow.derived.derive_statuses` derives manifest statuses from
events. Completion events provide completed, reused, or skipped; an
`attempt_failed` warning provides failed. Unreached steps are blocked by upstream
failure or otherwise pending. Other warnings preserve status.

Before saving the manifest, the runner checks agreement with its state.
Disagreement or an unreadable stream prevents the write. A clean run raises
`ProtocolError`; during another failure or interrupt it issues `ProtocolWarning`
and preserves that failure. A corrupt existing stream is rejected before the
run. Restore it or move it aside before retrying.

`_step.json` is written before publication; `result_committed` records the
rename. Published outputs and their records remain fixed. Appending later tracker
events can make `terminal()` false while preserving the manifest.

<a id="5-validation--load-error-checklist"></a>

## 5. Validation: load-error checklist

These checks apply to workflows. Intervention steps also run the IM spec's
validation checklist.

1. Keys and step types must belong to their declared vocabularies.
2. `output_dir` must be one safe path segment. Unusual section order warns.
3. Step names must be unique and safe; `workflow.json` is reserved.
4. References must name declared steps and files. Relative paths must exist at
   load; absolute paths are checked at runtime and listed by `explain`.
   Selectors must match formats; an in-run JSON `key` must appear in declared `keys`.
5. References and ordering must form an acyclic graph.
6. A script must have one locator, resolve, parse, and define `main`.
7. Output names must be unique, contained, and use supported extensions.
   JSON alone accepts `columns` or `keys`, exclusively.
8. Overrides must target existing paths and produce a valid document.
9. Tensor selectors are checked at load for protocol producers and at runtime
   for script producers.
10. Isolated runtimes declare dependencies and variable names (§4.1).
11. `is_deterministic` must be boolean.
12. A `reduction` must satisfy §2.6. Its input name is reserved. The built-in
    reducer accepts one value-column selector. Runtime column validation names
    the field responsible for an invalid reference.
13. `reduction.estimand_version` must use `<estimand>/v<n>` and match the declared
    calculation. For example, `ratio_of_sums/v1` requires denominator weights (§2.6).
14. With controls enabled, each required control must be declared or waived:
    `self_swap` and `matched_random`. `shuffled_source` is optional. Declarations
    and certifiers must satisfy §2.2.
15. A qualification runs before its target with matching model realization:
    `key`, `revision`, `dtype`, and `quantization`. A changed realization must be
    re-qualified. A `matched_random` control reading a fit can run after it (§2.2).
16. Coverage controls must be site-equivalent or declare `non_equivalence` (§2.2).
    The fields are `component`, `shape`, `layers`, `head`, `expert`, `stream`,
    `routed_rank`, `featurizer`, `dims`, and `sharing`. A `full_component` control
    may differ in featurizer and dimensions; `self_swap` uses coordinate checks.
17. A `behavioral` step must satisfy §2.7. Validate its `checker`, `split`,
    `decoding`, `seed`, and `thresholds`. Documents need generated positions and
    literal base references. Sampling requires engine support.
18. `decision`, `conditional`, and `requires_receipt` must satisfy §2.8.
    Validate each `predicate` and `scope`; per-child scopes also follow §2.9.
    A refusal names the field. Failed or missing receipts fail before allocation.
19. A `fan_out` needs a finite `over` axis or `shards` count and an explicit
    `join` (§2.9). Controls and certifiers cannot fan-out. A `selected` join
    needs a matching `per_target` or `per_variable` conditional. The loader
    checks output restrictions and names the field; the join rejects **missing**
    and **duplicate** points separately.
20. A nested `workflow` must satisfy §2.10. Refuse a document that includes itself.
    Validate `set` overrides and qualify names as `<step>/<inner>`. Restrictions
    on `control`, `waive`, and `fan_out` apply. The entry records `workflow_digest`.
    Each refusal names the field.

<a id="6-derived--never-authored"></a>

## 6. Derived: never authored

The loader derives the step graph, fan-out children, nested steps and each
step's identity. A document never states them. The [internals
page](workflow_protocol_internals.md#6-derived-never-authored) lists each one
and where it comes from.

## 7. Canonical form, digests, and `--resume`

An unfanned protocol step uses its inner document digest. Other steps use
the SHA-256 digest of their canonical entry. `digest <wf>` prints these step
identities. A protocol step that loads an earlier step's output is the one
exception: `--resume` compares it with a digest compiled against the run
tree, and `digest` prints the declared form (see below). Only nested
references use a whole-document fold, recorded as `workflow_digest` in the
parent's entry.

A script step's canonical entry is

```
{type, script, script_sha256, closure, closure_sha256, inputs, outputs, runtime, reduction, is_deterministic, after}
```

(`closure` and `closure_sha256` only when a `{"path": …}` script imports a
sibling, §4.2); a behavioral step's (§2.7) is

```
{type, document, set, max_points, document_digest, decoding, checker, split, thresholds, retain, decision, fan_out, after}
```

a protocol step's (§2.2) is

```
{type, document, set, max_points, control, waive, stop_after_failure_rate, fan_out, document_digest, requires_receipt, after}
```

with `fan_out` on either **only when authored** (§2.9),

a decision step's (§2.8) is

```
{type, values, rule, decision, requires_receipt, after}
```

a conditional step's is

```
{type, predicate, on_true, on_false, scope, requires_receipt, after}
```

and a `workflow` step's (§2.10) is

```
{type, document, set, workflow_digest, requires_receipt, after}
```

Behavioral settings, including seed and split, enter step identity.
`retain` enters when declared. Reuse compares the full step identity.

A fan-out parent's identity includes its partition and join; child identity
adds the shard. A container binds its inner `workflow_digest`, while inner steps
keep individual identities. Decisions and conditionals include their own fields,
with `requires_receipt` when declared.

Optional control, runtime, and reduction fields enter canonical entries when
declared. Equivalent waiver forms canonicalize identically. `after` and
`non_equivalence.fields` are sorted. Derived edges and equivalence verdicts
stay outside the canonical form.

Script entries include their hash and non-empty sibling closure. Package
implementation identity is recorded separately. Defaults such as
`is_deterministic: true` are materialized. Protocol entries include the document
digest after overrides; deferred references use the declared form until the
execution records resolved point digests.

`output_dir` is excluded. Step hashes and nested document folds use the IM
spec's canonical byte rules.

`is_deterministic` defaults to true. A false value marks the step as
non-replayable in `explain`; reuse then requires `--reuse-nondeterministic`.

Reuse requires matching step identity and `tree_digest`, plus every recorded
file with its SHA-256 content hash. A protocol step that loads an earlier
step's output, such as a fit's `init.file_path` or an apply's `file_path`,
records the digest of its run-time compile, and the bytes it loaded are in
that digest. `--resume` compiles the step again against the current run tree
and compares the two digests. The step is reused while its inputs and its
document are unchanged. An unfanned step keeps that digest as its identity.
A fanned-out child keeps its load-time identity and records the digest as
`document_digest`, which is compared in the same way. Script file
references, whether external `path` inputs or upstream `step` outputs, must
match `input_digests`. The hash covers the containing file even when an
input selects one value or tensor.
Missing identity or digest records cause execution again.
A reused step retains its earlier record.

A reused step never runs, so `--resume` loads no tokenizer for it. The answer
check that a first run makes before step 1 (§4) runs instead at the turn of
each step that is attempted. A resume on a machine that cannot load a reused
step's tokenizer, such as an offline node without it in the cache, still
reuses the step. A refusal at a step's turn fails that step, and the steps
before it keep their outcomes.

`causalab.provenance.runtime_identity().tree_digest` hashes executable
package files. Document script and code hashes cover named external code.
Any mismatch triggers execution, reported as `completed`. Moving the run tree
preserves reuse when identities and contents still match.

`workflow.json` records resolved inputs, script hashes, declared runtime and
reduction, step identity, implementation, output digests, and status.
Nested identities appear under `nested`. The runner writes it on completion,
failure, or interruption, subject to the event checks in §4.3.

Editing a shipped script changes step identity. Editing another package
module changes `tree_digest` and causes resumed steps to run again.
Static closure limits are described in §4.2.

## 8. Runner contract

The runner's services, attempt handling and step record formats are in the
[internals page](workflow_protocol_internals.md#8-runner-contract). Two
vocabularies in `workflow.json` matter when you read results: each record's
`disposition` and each step's status.

**Every published record carries `disposition`**, a closed vocabulary held to
this table by `tests/workflow/test_conditional.py`:

| disposition | meaning |
|---|---|
| `candidate` | the record as written into the attempt, before its publish |
| `accepted` | the published unit at `<step>/`: the one a reader beside its files (`read_sidecar`) finds, and the only one analysis accepts by default |
| `inadmissible` | a failed attempt's `attempt.json` |
| `superseded` | a unit a rerun displaced, or a skip took out of the run, retained under `.attempts/<step>/<n>.superseded/` with `superseded_by` |

Analysis accepts records marked `accepted`. Reruns retain displaced units
under `.attempts/<step>/<n>.superseded/` and list them in the manifest.
Superseded units have no retention limit; pruning applies to failed attempts.

Each executable step has one manifest status derived from the stream:

| status | meaning |
|---|---|
| `completed` | this run attempted the step, verified its outputs and published them |
| `reused` | `--resume` found a published unit whose identity, `implementation` and content digests match; the record is the earlier run's |
| `failed` | this run's attempt raised: a script error, a verification refusal, or an interruption mid-attempt (`KeyboardInterrupt`); nothing was published, the earlier unit if any is untouched |
| `blocked` | not attempted because a step it depends on is `failed` or `blocked` (`blocked_by` names them) |
| `pending` | not reached: the run stopped before its turn and nothing upstream failed |
| `skipped` | a conditional's verdict took the step out of this run (§2.8), directly or through a step it depends on; `skipped_by` names the decision by `evidence_identity`; no attempt, no directory, nothing published |

## 9. CLI

The same four verbs (IM spec §9) accept workflow documents: dispatch on the
`steps` section:

| verb | effect |
|---|---|
| `run <wf> --out <root>` | validate, schedule, execute steps, stamp the manifest. With `--parallel` every rank runs the steps in lockstep and the joiner alone writes the run tree; the data axis is refused (use `fan_out.over.shards`) |
| `validate <wf>` | the §5 checklist, including every inner document. `--tokenizer` also resolves the token positions and metric answers of every inner document that compiles at load, with each model's tokenizer (IM spec §2.10). Its line names, per step, the metrics read over generated tokens, whose answers are checked when scored |
| `explain <wf>` | the derived schedule (levels of parallel steps), per-step inner digests/point counts, the width and join of each fan-out and each child's shard (§2.9), a nested workflow's steps indented under their `workflow` step (§2.10), non-deterministic steps, unchecked absolute paths |
| `digest <wf>` | the identities `--resume` compares, one `<step>  <digest>` line per step in schedule order (§7); for a step that loads an earlier step's output, the declared form, because `--resume` compares that step with a digest compiled against the run tree; there is no whole-workflow digest |

<a id="10-worked-example--the-weekdays-8b-pipeline"></a>

## 10. Worked example: the weekdays-8b pipeline

Locate a layer × position cell, fit DAS rotations at it, apply the best fit on
the test split, and plot: as one workflow over the golden-corpus documents
07/08/09.

```json
{
  "version": "1",
  "description": "weekdays-8b: locate -> DAS k x seed fits at the best cell -> apply on test; scan heatmap + IIA-vs-k curves.",
  "output_dir": "weekdays_8b",
  "steps": {
    "locate": {"type": "intervention_protocol", "document": "../protocols/weekdays_locate_scan.json"},
    "best": {
      "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
      "inputs": {
        "table": {"step": "locate", "file": "iia.json"},
        "choose": "max",
        "emit": {"best_layer": "sites.target.layers", "best_pos": "positions.tap"}
      },
      "outputs": {"values": "values.json"}
    },
    "fit": {"type": "intervention_protocol", "document": "../protocols/weekdays_das_sweep.json",
             "set": {"positions.best": {"artifact": "best", "key": "best_pos"},
                     "sites.target.layers": {"artifact": "best", "key": "best_layer"}}},
    "best_fit": {
      "type": "script", "script": {"module": "causalab.workflow.scripts.select"},
      "inputs": {
        "table": {"step": "fit", "file": "iia.json"},
        "choose": "max",
        "emit": {"best_k": "featurizers.rot.k", "best_seed": "train.seed"}
      },
      "outputs": {"values": "values.json"}
    },
    "apply": {"type": "intervention_protocol", "document": "../protocols/weekdays_das_apply.json",
               "set": {"featurizers.rot.file_path": "fit/rot.safetensors",
                       "featurizers.rot.k": {"artifact": "best_fit", "key": "best_k"},
                       "featurizers.rot.entry": {
                         "k": {"artifact": "best_fit", "key": "best_k"},
                         "seed": {"artifact": "best_fit", "key": "best_seed"}}}},
    "scan_heatmap": {
      "type": "script", "script": {"module": "causalab.io.plots.workflow_figures"},
      "inputs": {
        "table": {"step": "locate", "file": "iia.json"},
        "plot": "heatmap", "x": "sites.target.layers", "y": "positions.tap"
      },
      "outputs": {"figure": "scan_iia.png"}
    }
  }
}
```

The schedule is `locate` → `best` → `fit` → `best_fit` → `apply`.
`scan_heatmap` can run after `locate`. Overrides supply the selected layer.
The rank and seed sweep saves nine rotations; `best_fit` selects one entry
for held-out evaluation, with `ArtifactIdentity` checked on load.

Intervention `set` uses `artifact` references; script `inputs` uses workflow `step` references (§3).

## 11. Current limits

- `select` and plot scripts use the row-based mean in §2.6. Apply a declared
  reduction first for another statistic.
- Custom scripts can produce additional figure types.
- Separate workflow runs connect through external paths or artifacts.
- A run selects one engine for all protocol steps.
- Nested controls and joins have the limits in §§2.9 and 2.10.
