# Architecture

This guide gives the package layout, the dependency rules between layers,
and the invariants each layer keeps.

## 1. Package structure

CausaLab expresses causal analysis as protocol and workflow documents. The
[intervention protocol](intervention_protocol.md) defines the format.
[Workflow documents](workflow_protocol.md) connect experiments and analysis steps.
The [intervention protocol internals](intervention_protocol_internals.md) and
[workflow protocol internals](workflow_protocol_internals.md) define the engine
and runner contracts that the packages below implement.

| Package | Responsibility |
|---|---|
| `causal/` | Causal models, traces, interventions, scoring, and pair validation |
| `tasks/` | Task definitions and generators; `serialize.py` builds dataset tables |
| `protocol/` | Document parsing, validation, identity, and token-position resolution |
| `neural/shared/` | Planning, sweeps, tensor operations, and execution shared by both engines |
| `neural/engines/` | Model access through PyTorch hooks or nnsight tracing |
| `analysis/` | Fits, statistics, controls, and helpers for experiment authors |
| `workflow/` | Workflow loading, scheduling, execution, and records |
| `measurement/` | Single-source and before/after studies, isolated workers, captures, comparisons, and deployment |
| `profiling/` | Profiler adapters and artifact handling |
| `remote/` | SSH transport and standalone job supervision |
| `io/` | Dataset and artifact access, event logs, tensor files, and plots |

### Dependencies

Keep the causal layer independent of higher layers. Tasks use causal models;
the protocol data checks also use the standard-library modules `scoring.py`
and `pair_validation.py`.

The protocol package has no dependency on an execution engine or the workflow
package. Its imports stay free of torch. The workflow layer consumes compiled
protocols and receives an engine from the caller. `causalab/cli.py` dispatches
to the appropriate document layer.

The protocol and I/O packages share the resolution interfaces in `io/env.py`,
`io/sources.py`, and `io/tables.py`. Lazy package imports prevent cycles.
Keep numerical imports inside script functions: resolving a module locator
can import its parent packages during validation.

No module exists only to forward another module's names;
`tests/docs/test_docs.py` refuses a star-import forwarder under `causalab/`.
A package root exports the names that callers read through it. `protocol/__init__.py`
exports `run_protocol`, `RunContext`, and `RUN_RECORD_NAME` through lazy
imports. Import every other name from its defining module.
`tests/protocol/test_schema_package.py`,
`tests/protocol/test_registry_package.py`, and
`tests/neural/shared/test_featurizers_package.py` pin the exported surfaces
of those packages.

`tests/test_architecture_layering.py` checks import boundaries.
`tests/protocol/test_load_is_torch_free.py` checks imports during document
loading and validation.

### Measurements (`causalab/measurement/`)

The [usage guide](measurement.md) covers single-commit profiling, code comparisons,
and workflow comparisons. The [developer guide](optimization_experiments_design.md)
describes isolation, timing, and resume contracts. `Operation` and `collect`
are the public Python entry points in `causalab.measurement`.

| Location | Responsibility |
|---|---|
| `collection.py`, `workflow.py`, `spec.py`, `plan.py` | Operation collection, workflow adaptation, and typed study configuration |
| `device.py` | Single-device and single-rank guards |
| `study/` | Study orchestration, scheduling, and crossed evaluation |
| `runtime/` | Isolated worker bootstrap, execution, observations, probes, and training instrumentation |
| `capture/` | Profiling runs, capture workers, and named ranges |
| `analysis/` | Comparisons, stability, summaries, profile analysis, and reports |
| `deployment/` | Source installation, remote launch, and result transfer |
| `paths.py` | Controller paths and recursive source identity |

The workflow parser loads `measurement/spec.py` lazily; `workflow/` has no
module-level measurement imports. Package initializers stay torch-free.
`tests/measurement/test_workflow_boundary.py` checks the import boundary.
Measurement deployment and the remote launcher of the speed-of-light (SOL)
bounds in `causalab/sol/` share `causalab/remote/`.

### The causal layer (`causalab/causal/`)

Causal models are Python functions marked with `@mechanism`. Each `V`
assignment in the function becomes a variable, and the compiler finds its
parents from what its equation can read. The [model guide](causal-models.md)
lists the supported syntax, and the [folder README](../causalab/causal/README.md)
has a runnable example. Each model declares correctness once through an
immutable `ScoringSpec`; task graders and serialized answer forms derive from
that declaration.

| Module | Contents |
|---|---|
| `model.py` | Defines models and stores their graphs in `CausalModel`. Includes `CausalTrace` and `CompiledEquation`. Eager checks run when an intervention is applied. Lazy values are checked when read. |
| `domains.py` | Allowed values, with checks and bounded enumeration. `Exo` marks explicit noise inputs. |
| `compiler.py` | Copies configuration, expands equations, and infers domains. It checks which reads are possible before it checks graph cycles. |
| `counterfactuals.py` | Example records, sampling, and labels from interventions. |
| `model_comparison.py` | Compares causal predictions and scores saved intervention results. Uses NumPy and PyTorch. |
| `pair_validation.py` | Checks prompt pairs and groups of edits, including the optional `edit_groups` column. Uses the Python standard library. |
| `scoring.py` | Answer forms and grading rules in `ScoringSpec`. Includes `build_output_tokens`. Uses the Python standard library. |

The [hypothesis comparison guide](hypothesis_analysis.md) describes how to
compare saved neural outputs with symbolic predictions on the same pairs.

## 2. The protocol layer (`causalab/protocol/`)

| Module | Responsibility |
|---|---|
| `schema/` | Typed records, parsing, defaults, and canonical forms |
| `compiled.py` | The frozen `CompiledProtocol` shared by all entry points |
| `pipeline.py` | Build, validate, resolve positions, and run a protocol |
| `rules/` | Document, dataset, artifact, code, and capability checks |
| `lowering.py` | Axes, layer bands, and named families |
| `identity.py`, `bundles.py` | Content digests, code identity, and saved bundle entries |
| `positions/` | Token frames, spans, alignments, role resolution, and location ledgers |
| `answers.py` | Metric answers resolved to token ids, shared by the run door and the score |
| `registry/` | Model metadata, component capabilities, family adapters, and parallel plans |
| `engine.py` | `Engine.execute(compiled, run)` and its input and result types |
| `results.py`, `estimand.py` | Availability, eligibility, row labels, units, and estimand identity |
| `equivalence.py` | Comparison of intervention sites and coordinate sharing |
| `receipt.py`, `reports.py` | Receipt schema and CLI reports |
| `migrate.py` | Protocol migration and document formatting |
| `parallel.py`, `kv_replication.py` | `--parallel` geometry grammar, divisibility checks, and KV-head arithmetic |
| `publish.py`, `lockstep.py` | Rank ownership and point shards; agreed workflow decisions across ranks |
| `parallel_memory.py`, `checkpoint_census.py` | Torch-free memory estimates from cached checkpoint headers |

### Compile and run

`build` reads the document, applies overrides, resolves references, lowers
syntax, and computes the campaign identity. `validate` checks one
representative per axis value, dataset columns, loaded artifacts, and the
selected engine's capabilities. `compile_protocol` combines these calls.

The engine enumerates concrete points in canonical order and validates each
selected point before loading weights. This catches failures caused by a
combination of axis values. Each point is signed with `identity.sign_step`.
Its digest identifies that intervention in results and saved artifacts.

`run_protocol` creates the run context and calls `handoff`, which validates,
resolves positions with the tokenizer, and enters `Engine.execute`. The
engine writes the run receipt and event stream when the run context asks for
them (`record=True`; the CLI's `--record`). Workflow steps use the same
compiler and record their execution in `_step.json`.

### Token positions

`validate`, `explain`, `dry-run`, and `digest` use document and dataset
metadata. Token counts and alignments are resolved at execution through
`protocol/positions/`, and every metric's answer tokens through
`protocol/answers.py`, both before the weights load; `validate --tokenizer`
and `dry-run --tokenizer` run the same two passes. `io/tokenizer.py` supplies
the tokenizer with left padding and an EOS pad token when needed.

The protocol's `PositionFrame` holds token IDs, masks, offsets, and segment
locations. The engine wraps it in device tensors through
`neural/shared/encoding.py`. Continuation positions use the generated tokens
recorded by `neural/shared/generated.py`. Task-side position helpers live in
`tasks/token_positions.py`.

### The registry (`causalab/protocol/registry/`)

The registry provides static metadata for validation without a model load.
`models.py` declares dimensions and layer types; `components.py` declares
shapes, supported writes, predicates, and family taps. `engines.py` derives
capabilities from these rows. `families.py` maps a loaded module tree to the
global component vocabulary and provides its inventory.

Register a family adapter for a new architecture. Keep component policy in
the registry so validation and both engines use the same rules. A caller can
explicitly adapt a Hugging Face config with `model_info_from_hf_config`.

## 3. The engines (`causalab/neural/`)

`--engine` is required for `run`, `validate`, `explain`, and `dry-run`.
`auto` resolves to `pytorch_hooks`. A capability mismatch raises `[V13]` with
the missing capability. Only execution constructs an engine.

The [running guide](running_experiments.md) lists components and engine support.

<!-- generated: begin engine-summary -->

| | `pytorch_hooks` (reference) | `nnsight` |
|---|---|---|
| how | `register_forward_hook` / pre-hook, plus global swaps for the delta kernel and the experts dispatch | one trace over an envoy tree, `.source` for fused-forward interiors |
| capabilities | `grad` `paired_forward` `full_logits` `writable_attention_probs` `pytorch_fn_local` `generate` `generation_writes` `quantized_weights` | `paired_forward` `full_logits` `writable_attention_probs` `pytorch_fn_local` `generate` |
| components | 52 of 56; unsupported: `deltanet_query`, `deltanet_key`, `deltanet_state` and `expert_permutation` | 51 of 56; unsupported: `delta_query`, `delta_key`, `delta_kv_mem`, `delta_state_update` and `delta_state` |
| serves alone | the post-tiling `delta_query` / `delta_key` and the per-step `delta_state` (the typed backend pairs), training (a `train` document needs `grad`), quantized weights | the fused-forward faces `deltanet_query` / `deltanet_key` / `deltanet_state` and `expert_permutation` |
| install | always | `uv sync` (dev group) or the `nnsight` extra |

<!-- generated: end engine-summary -->

### The shared layer (`causalab/neural/shared/`)

Both engines use the shared planner, position frames, featurizers, metrics,
and result writers. `engine_router.py` resolves `--engine` and constructs the
chosen engine lazily, so it stays torch-free at import. `sites.py` resolves
family taps after `model_tree.py` checks layer streams and architectural
predicates. `executor/` handles operand lookup, write math, ragged windows,
and captures.

`parallel/` holds the distributed primitives: placements and fragments,
collectives and autograd protocols, meshes and launchers, context-parallel
frames, agreements, memory preflight, heartbeats, the watchdog and the
CUDA graph replay deadline. `join.py`
folds data-parallel shards into one output in point order; `devices.py`
parses a comma-list device into a `DeviceMap`; `symbol_dispatch.py` layers
per-thread patches over module globals; `kernels.py` binds the DeltaNet
kernel path per device.

Key execution rules:

- `sweep.py` enumerates points and signs them with the protocol hasher.
- `plan.py` groups forwards by model realization and input content. Campaign
  caches share source captures and compatible prefixes.
- Writes apply absolute operations before additive operations at each address.
  `fires.py` checks every installed write's expected firing count.
- Featurizers preserve the base activation's error term and unselected
  coordinates. Their cache scope ends before a parameter or mode change.
- Metric rows carry example IDs, units, and eligibility. Excluded rows retain
  their reason codes and are counted in the denominator record.

### The reference engine (`causalab/neural/engines/pytorch_hooks/`)

`loading.py` prepares frozen models and tokenizers. `weights.py` reads selected
checkpoint tensors through [fastersafetensors](fastersafetensors.md).
`executor.py` installs module hooks; the attention, DeltaNet, and expert
interfaces expose tensors inside fused operations.

`train.py` fits compatible points in cohorts. Each member keeps its own seed,
optimizer, schedule, and stopping rule. `budget.py` bounds rows per forward;
an automatic budget can shrink after an out-of-memory failure. An authored
bound stays fixed. The receipt records resolved bounds and shrink events.

Desiderata-Based Masking (DBM) uses a trained gate. DBM-DAS combines a gate
with a learned subspace. `shared/featurizers/` implements their tensor
operations, including grouped and position gates.

CUDA graphs, kernel options, and compilation caches are described in
[cuda_graphs.md](cuda_graphs.md) and
[attention_backends.md](attention_backends.md). A model supplied by the caller
must match the document's realization. Temporary hooks and backend changes
are restored after execution, including failures.

Model parallelism is described in [model_parallelism.md](model_parallelism.md).
`sharding.py` applies the registry's plan through the `styles/`
implementations; `stages.py` runs pipeline stages; `rows.py` splits fit
minibatches across data replicas; `shard_read.py`, `checkpoint.py`, and
`residency.py` plan and audit sharded weight reads; `crossings.py` moves the
residual stream between devices of one process; `kv_replication.py` and
`partial_gradient.py` handle replicated KV heads and per-rank sliced outputs.

### The nnsight engine (`causalab/neural/engines/nnsight_tracing/`)

`loading.py` wraps a Transformers model in an envoy tree. `executor.py` runs
one trace per forward group, with `addresses.py` locating function interiors
through `.source`. It shares the engine contract and accepts caller-owned
bundles.

Parity tests compare reads and writes on tiny fixtures and a real
Qwen3.6-35B-A3B checkpoint. On the tiny fixtures they also compare mechanisms,
featurizers, aggregations, positions, receipts, and execution seams. Each of
these tests runs one document through both engines with
`tests/_helpers/engines.py`, which lists the accepted differences between the
engines. The nnsight engine does not serve `writes_during_generation`, and
routing refuses such a document before any weights load. The CLI refuses
`--parallel` with `--engine nnsight`. Each run uses one engine. The hooks
engine supports row bounds, a comma-list `--device` that places layers across
the devices of one process (`DeviceMap`), and `--parallel` geometries across
processes (see [model_parallelism.md](model_parallelism.md)); nnsight runs
each forward group as one batch on one device.

## 4. The workflow runner (`causalab/workflow/`)

`document.py` loads steps, resolves references, derives dependencies, and
checks controls. `runner.py` executes the graph in dependency order, with an
artifact overlay that lets later steps read earlier outputs.

| Module | Responsibility |
|---|---|
| `behavioral.py` | Decoding, task grading, outcomes, and behavioral decisions |
| `conditional.py` | Typed decisions, conditional execution, and required receipts |
| `fan_out.py` | Expansion over axes or shards and joins by point identity |
| `nested.py` | Nested workflows, rebased references, and scoped output paths |
| `reduction.py`, [reduce script](../causalab/workflow/scripts/reduce.py) | Estimators, uncertainty, and metric reduction |
| [select script](../causalab/workflow/scripts/select.py) | Values that later steps use in `set` overrides |
| `manifest.py`, `derived.py` | Publication, retained attempts, and statuses derived from events |
| `isolate.py` | Execution of a script in a separate process |

Controls run before their targets when qualification is required. Each target
records the control's document digest, implementation identity, engine, and
status by coordinates. A certification must cover every control point.
Failure rates above the declared bound block dependent steps. Matched-random
controls follow the fit when they need its learned size.

### Outputs and reuse

A step writes into `.attempts/<step>/<id>/`. The runner verifies each declared
output and its content digest before publishing the step directory. It retains
a displaced result as a superseded attempt.

`--resume` reuses a step when its recorded identities and output checksums
match. The implementation's `tree_digest` is part of that check. Script
identity includes imported sibling files outside the package.

`events.jsonl` is appended beside the run manifest. Its lines determine the
statuses in `workflow.json`: `completed`, `reused`, `failed`, `blocked`,
`pending`, or `skipped`. A disagreement with the runner's status prevents the
manifest write. Event-sink failures produce warnings while execution continues.

### Step scripts

A script implements `main(inputs, outputs) -> None` and creates every declared
output. The runner checks table columns and stamps tensor artifacts with their
identity. `io/step_io.py` supplies table, value, and bundle access;
`io/step_record.py` supplies the aggregation used by selection and plotting.

Script modules are located and hashed at load time. Their parent packages must
remain importable without numerical libraries. The script body is imported
when the step runs.

Numerical scripts live in `analysis/`; plotting lives in
`io/plots/workflow_figures.py`. Library helpers include `analysis/sequences.py`
for fixed-sequence documents and `analysis/logit_lens.py` for projecting saved
residuals through a loaded decoder. See [multi_token_analysis.md](multi_token_analysis.md).
`analysis/export_dbm.py` exports validated DBM apply results through
`export(manifest_path, register_from_hf=False)`; `scripts/export_dbm.py` writes
the JSON for downstream reports.

## 5. Datasets are build products

Each task ships tables under `causalab/tasks/<name>/data/`. The task README
records the builder command. `causalab/tasks/` is the default data root.

A document resolves a dataset reference by reading the table's bytes. Task
values, answer forms, scoring mode, and row-specific position anchors belong
in columns. Building a dataset is a separate command. Its content digest
enters every consuming document's canonical form, so rebuilding a table
changes those identities and causes its consumers to run again.

## 6. Configs are documents

The shipped method documents live in the repository, not in the package:
`demos/methods/protocols/` contains intervention specifications;
`demos/methods/workflows/` connects them. Each method keeps its training
settings in its own document. The `interchange`, `das`, and `dbm` families
provide reusable method blocks; apply documents and random controls state
their own operations.

Use `--set` for an override and save a document when the experiment should be
reused. `causalab migrate` updates older protocol versions.

## 7. Tests

See [TESTS.md](TESTS.md) for test tiers, numerical checks, and artifact updates.
