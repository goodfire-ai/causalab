# Testing conventions

This guide gives the test tiers, their markers and time budgets, and the
commands that run each tier.

Tests check that CausaLab executes the intervention a document specifies and
records enough evidence to reproduce the result.

## Quick Start

```bash
uv run pytest -m "not golden and not measurement_study and not parallel_world"
uv run pytest -m "not golden"
uv run pytest -m golden
```

The first command runs the pull-request gate in one process. The second adds
the two cost-marked families (see [Conventions](#conventions)). The golden
tier needs a CUDA host; real-model suites also need cached weights or access
to their model repositories.

Mutation coverage of the parallelism core is described in
[model_parallelism.md](model_parallelism.md) §10.7: install `mutmut==3.7.0`
into the venv as a tool, clear every `__pycache__` outside `.venv`, and run
it over `causalab/neural/shared/parallel/`.

## Overview of test types

Assign exactly one tier marker to every test. `tests/conftest.py` rejects a
collection with missing markers.

| Tier | Checks | Wall budget |
|---|---|---|
| `numerical_unit` | Fixed input-output values, task pins, and independent numerical oracles on CPU | <2 min total |
| `property` | Shapes, dtypes, invariance, and determinism | <1 min total |
| `unit` | Parsers, validation, small utilities, and structural checks | <5 min total |
| `smoke` | Complete CLI runs on tiny models; artifact shapes and indicator ranges | <5 min total |
| `golden` | Real-model results, accelerator parity, and CUDA kernels; also checks that need a gated model's files, such as its tokenizer, and no accelerator | Depends on model and workload |

The `cuda` marker records a hardware requirement separately from the tier.
Tests carrying it skip when CUDA is unavailable.

## Conventions

Measurement tests mirror the implementation under `tests/measurement/`, with
`study/`, `runtime/`, `capture/`, `analysis/`, and `deployment/` suites. Shared
transport and supervisor tests live in `tests/remote/`; tests of the
speed-of-light (SOL) bounds in `causalab/sol/` live in `tests/sol/`. The
[measurement developer guide](optimization_experiments_design.md#development-checks)
lists the contracts these suites protect.

The four end-to-end study modules `tests/measurement/study/test_controller.py`,
`test_distinct_commits.py`, `test_workflow_comparison.py` and
`test_workflow_training.py` carry a second, cost marker beside their tier:
`measurement_study`. Their nine tests build and install a source wheel of an
arm (compiling the Cargo workspace, `causalab/measurement/deployment/installation.py`)
and spawn cold worker processes — two to four minutes apiece on a two-vCPU
machine. The pull-request gate deselects the marker; `-m "not golden"` keeps
them. Like `cuda`, the marker is orthogonal to the tier taxonomy — a test
still declares exactly one tier. Run them alone with
`uv run pytest -m measurement_study`.

The second cost marker, `parallel_world`, is on the twelve modules whose tests
run a document or a fit across a spawned multi-rank process world on the tiny
fixtures — `test_kv_replication.py`, `test_train_parallel_run.py`,
`test_memory_preflight.py` (its gloo half; its `unit` half — meta models and
header reads, no world — stays on the gate), `test_interiors_sharded.py`,
`test_tensor_expert_parallel_run.py`, `test_rank_watchdog_cases_run.py`,
`test_sharded_load.py`, `test_data_parallel_run.py`, `test_pipeline_run.py`,
`test_context_parallel_run.py`, `test_context_parallel_train_run.py` under
`tests/neural/engines/pytorch_hooks/`, and `tests/golden/test_parallel_fit_harness.py`
(a CPU guard, tier `smoke`, despite its directory). Each spawns up to eight
ranks, each rank importing torch, and costs a minute or more serially.

The rule is the shape and the cost together, and the cost is the deciding
half: a module earns the marker when it runs a document or a fit across a
spawned world **and** costs a minute or more serially on a two-vCPU machine.
Same-shape modules under the threshold stay on the gate, unmarked — the
sharpest pair is `tests/neural/engines/pytorch_hooks/test_sharded_load.py`
(marked) beside `test_sharded_load_cast.py` (not): same helper, same
directory, opposite verdict, because the second spawns one two-rank world for
one test — as do the cheap world suites, `tests/neural/shared/parallel/test_collective.py`,
`test_lockstep_contract.py`, `test_mesh.py` and `test_styles.py`, hundreds of
sub-second programs on tiny worlds, which are the parallelism code's per-PR
coverage. The unmarked set is not enumerated here on purpose: only the marked
set is checkable (`uv run pytest --co -m parallel_world`), and a list of what
is deliberately absent goes stale with the next `*_run.py` module. Time a new
module before deciding, and add the marker to its own `pytestmark` when it
crosses the threshold. Run the marked ones alone with
`uv run pytest -m parallel_world`.

The gate runs in one process on purpose. On a machine with two vCPUs and
7 GB, two xdist workers co-schedule two spawned worlds (an eight-rank and a
four-rank one, deterministically, at 28 % of the run) and the machine is
killed with no test failure, while the serial run on the same commit passes.
`-n` needs a larger machine.

### Mocking policy

Use small real implementations and committed fixture tables. Limit mocks to
external services, controlled time or randomness, and injected failures.
Extend `tests/_helpers/simulated_world/` for new distributed scenarios rather
than adding another simulator.

CPU fixtures must use ungated models. The run path loads a tokenizer before
entering the engine, including when the engine is a stub, and a workflow run
loads the tokenizer of every inner document that compiles at load before its
first step. A stub engine's document therefore needs answers that its
tokenizer resolves to single tokens. The shared corpus
uses `Qwen/Qwen3-8B` for static metadata and its tokenizer; execution tests
retarget documents to tiny models. `tests/test_no_gated_models.py` checks
this boundary without relying on a developer's cache.

### Unit tests

Place a test for `causalab/<subdir>/<stem>.py` at
`tests/<subdir>/test_<stem>.py`. Apply the same layout to scripts. Declare the
tier at module, class, or function scope; a file containing several tiers can
use one class per tier.

Test a refusal beside a valid example. For checks that must precede weight
loading, make the loader raise if called. Numerical tests need an independent
expected result and a tolerance justified by the dtype and operation order.
Use a mutation or another failing case when it helps show that an assertion
can detect the defect it targets.

### Main contracts

| Area | Tests |
|---|---|
| Compiler identity across CLI, Python, and workflow entry points | `tests/protocol/test_compile_protocol.py` |
| Document legality before weights load | `tests/protocol/test_legality_before_weights.py` |
| Metric answers resolve before weights load, at the run door and before a workflow's first step, over the rows each metric scores | `tests/protocol/test_pipeline_answers.py`, `tests/protocol/test_answers.py`, `tests/workflow/test_answers_before_weights.py`, `tests/protocol/test_cli_tokenizer.py`, `tests/neural/engines/pytorch_hooks/test_eligibility_run.py` |
| Axis lowering, layer bands, and path patching | `tests/protocol/test_site_layers.py`, `tests/neural/engines/pytorch_hooks/test_path_patching_run.py` |
| Controls and qualification by coordinate | `tests/workflow/test_controls.py`, `tests/workflow/test_qualification.py` |
| Site equivalence | `tests/workflow/test_equivalence.py` |
| Behavioral decisions, conditions, fan-out, and nested workflows | `tests/workflow/test_behavioral.py`, `tests/workflow/test_conditional.py`, `tests/workflow/test_fan_out.py`, `tests/workflow/test_nested.py` |
| Verified publication and reuse | `tests/workflow/test_attempt_publish.py`, `tests/workflow/test_resume_implementation.py`, `tests/workflow/test_resume_closure.py` |
| Statuses derived from the event log | `tests/workflow/test_derived_manifest.py` |
| Task grading agrees across strings, probability forms, and serialized rows | `tests/tasks/test_scoring_differential.py` |
| Coordinated pair edits and their token spans | `tests/causal/test_pair_validation.py`, `tests/protocol/test_pairs_identity.py` |
| Installed dependencies and document-relative paths | `tests/test_declared_dependencies.py`, `tests/workflow/test_document_relative_inputs.py` |
| One document gives the same answer on both engines | `tests/neural/engines/nnsight_tracing/test_parity_*.py`, through `tests/_helpers/engines.py` |

### The prose is checked too

`tests/docs/test_docs.py` checks documented CLI commands, relative links,
repository paths, JSON examples, and quoted digests. Demo-specific format and
inline-document checks live in `tests/demos/test_demos.py`.

`scripts/generate_support_tables.py` writes marker-fenced support tables and
method-page pointer blocks. Run it after changing their source objects.
`tests/scripts/test_generate_support_tables.py` requires each block to match
its generated text and each pointer to resolve.

### Paper replication packages

`tests/demos/test_papers.py` runs in the `unit` tier and reads every
`demos/papers/workflows/<name>.json` as a package, in the layout of
[the paper replication guide](paper_replications.md#layout).

| Check | Rule |
|---|---|
| the layout | `demos/papers/` holds only `artifacts/`, `protocols/` and `workflows/` as folders; every package has its page; every protocol, script folder, data folder and figure folder names a package; a page that is not a package page is `<name>_<topic>.md` |
| every page's H1 | names the method or the content, as [the page format](paper_replications.md#parts-in-order) asks: it does not start with `CausaLab`, carries no `Figure N` or `Table N` label, and does not repeat the paper title in the bold text of the citation blockquote |
| every page's figure subheadings | follow a [page variant](paper_replications.md#page-variants): none, or `### Original` and then `### Replication` or `### Verification with <method>` |
| a JSON file's kind | by content: a JSON array under `artifacts/data/` is a table; elsewhere, `steps` at the top level is a workflow and `header` is an intervention specification |
| every intervention specification | validates against the package's tables, with `artifacts/data/` as the data root. A specification that loads a run-tree artifact (the apply half of a fit and apply pair) is validated through its workflow instead |
| every workflow | loads, so its script locators and cross-step references resolve |
| every committed table of a package with a builder | reproduces byte for byte under `--check` |
| every package that ships `copy_original.py` | its copy of the onboarding Original equals a fresh copy under `--check`, so a re-run of the onboarding page fails until the copy is redone |
| every script step | runs on fabricated outputs of the protocol steps it reads, and writes each declared output with its declared columns and keys, as `causalab.workflow.runner.verify_output` checks them; a table declared with columns has at least one row |
| every page in the tutorial layout | the chunks after the Full JSON block, up to the next `##` heading, merge to the file minus its `header`; the Full JSON block equals a protocol file once its comments are gone |
| every `python` fence of a page | names its source file in its `title` and equals the source of the one definition it holds, decorators included; `json` chunk merging skips these fences |
| every workflow on an ungated model | `validate --tokenizer`: the positions and metric answers of every inner document that compiles at load resolve with its tokenizer, and no answer is glued to its prompt. A package on a gated model gets the same check in the golden tier, `tests/golden/test_paper_answer_columns.py`, because the CPU tier has no Hub token |

A package whose builder or figure script encodes a rule the table checks
cannot see has its own file beside it:
`tests/demos/test_addition_heads_dbm.py` checks the addition-heads table's
pairs and splits, and that its figure script reads kept heads and scores off a
synthetic run tree and draws the chosen weight's mask.

A downloaded dataset therefore has to be wrapped in an object. A bare array
under `artifacts/data/` is a table the builder must reproduce.

The script-step check needs no model. The runner checks a declaration only
after the script has run, so without this check a stale declaration fails on
the cluster after the model steps. The test resolves each script's inputs
and finishes its outputs through the runner's own `script_call`, and checks
them with `verify_output`. It imports the script itself, so that a script
that loads a model of its own can run with a stand-in for that one object
(`SCRIPT_STAND_INS`). It refuses an isolated step, which the runner runs in
a subprocess.

`tests/_helpers/fabricated_outputs.py` writes each protocol output a script
reads through the engine's own table and bundle writers and metric code.
A fit step's featurizer bundle holds the slots of the stage the engine
trains, built by the engine's `build_stack` at the site width the model
registry gives. Under the mocking policy above it is an allowed stand-in
because two smoke tests in `tests/_helpers/test_fabricated_outputs.py` hold
it to a real engine run. Four paper documents run on the tiny Llama, and the
fabricated files have the same file names, table columns and column types,
tensor keys, dtypes, ranks and leading axes, and identity fields. Four fit
presets of the method library run on the tiny Qwen3.5 MoE: a rotation behind
a boundary gate, a gate per coordinate, a gate per attention head and a
rotation swept over its rank. Their fabricated bundles have the same tensor
keys, dtypes and whole shapes and the same identity fields. The widths of a
read, the numbers, the per-row widths of a ragged read and the attention
backend a loaded model reports are not the engine's. Every answer is one
token under the stand-in tokenizer, so answers the real tokenizer splits
are not seen here. The helper refuses a derived record, a windowed metric
other than `decode`, a saved tensor at a generated position and a read on an
input other than `base`. Of the bundles, it refuses a featurizer that starts
from a saved file, a position gate, a pooled gate and a featurizer at a site
that selects a head or an expert. It also refuses a saved
tensor at a `span` or `indices` set inside a `scope` or `relative_to` anchor,
because the engine's width there follows from the set and its anchor. The script-step check
does not compare the dtypes a script writes with its declaration, and it
does not test the numbers a script computes.

A figure script that computes a statistic beyond a mean, such as a bootstrap
standard error, should be checked on synthetic run trees in its own
`numerical_unit` module, as `tests/demos/test_ioi_fig3b_figure.py` checks
the IOI Figure 3b script.

A package can add its own module, `tests/demos/test_<name>*.py`, for what
its builder and figure scripts compute, such as its labels and plotted values.
The module loads each script by its path, as a reader runs it.

A page is read by three checks. `tests/docs/test_docs.py` resolves every
relative link and every rooted path in it, against the repository root and
`demos/papers/`, and parses every `json` fence as a fragment or a whole
document. `test_papers.py` validates every document offline, so each model a
package names needs a row in the static registry, and it checks the chunks and the Full JSON block of the pages in `TUTORIALS`
against their files. It also checks the H1 of every page. The other parts
and their order are not checked mechanically.
Review them against [the page format](paper_replications.md#the-page).

### Pinned-artifact discipline

Generate pins with their update script, then review the diff. A numerical or
canonical-form change needs an explanation before its expected value changes.
Frozen reference captures remain fixed.

| Artifact | Update route | Review requirement |
|---|---|---|
| Demo digests and inline documents | `scripts/repin_demo_digests.py` | Use a baseline tree to resolve stale digest quotations |
| Generated documentation blocks | `scripts/generate_support_tables.py` | Review the source change; `--check` verifies the rendering |
| `tests/protocol/corpus_digests.json` | `tests/protocol/update_corpus_digests.py` | Explain changes to canonical form |
| Shipped digests and canonical documents | `tests/protocol/update_shipped_digests.py` | Treat canonical-form changes as loader migrations |
| `tests/golden/golden_digests.json` | `tests/golden/update_golden_digests.py` | Review the document and dataset changes |
| Drift measurements | `tests/golden/drift/update_drift_goldens.py` | Capture on CUDA and use `--i-have-reviewed-the-diff` after review |
| `tests/golden/parallel_goldens.json` | `tests/golden/update_parallel_goldens.py --i-have-reviewed-the-diff` | Two GPUs for the standard suite, eight for large-model captures; `--only` selects documents |
| `tests/golden/parallel_headers_*.json` | `python -m tests._helpers.header_census KEY --out PATH` | Regenerate when a checkpoint revision changes |
| Task samples | `scripts/update_task_pins.py` | Review prompt and causal-model semantics before confirming the update |
| Parity captures for `gpt2`, `gqa`, and `llama` | Frozen | Preserve values and capture provenance |
| Family records for `qwen35moe` and `qwen36_a3b` | `tests/neural/parity/update_family_goldens.py --family <name>` | Record the eight certification fields: `model_revision`, `causalab_revision`, `attn_implementation`, `dtype`, `hook_names`, `tensor_shapes`, `max_activation_diff`, and `max_logit_diff` |
| `tests/protocol/fixtures/refusal_snapshot.json` | `tests/protocol/update_refusal_snapshot.py` | Record behavior changes in `ALLOWED_UPGRADES`; use `--retire-only` for an intentionally removed refusal |
| Resolved-site census | `tests/neural/engines/pytorch_hooks/update_resolved_sites_census.py` | Extend the census for added coverage; document changes to existing behavior in the test |
| Task tables | `scripts/build_task_dataset.py`, `scripts/build_split_dataset.py` | Use the recipe in the task README and update consuming document pins |

Review each family record against its independent oracle and resolved
checkpoint revision. The capture environment records provenance; replay
checks values against the declared tolerance.

`tests/neural/engines/pytorch_hooks/test_end_to_end_iia.py` records the
`TABLE_RECIPE` for its prepared weekdays fixture. The corresponding pair
identity test also rebuilds that table.

**What is never pinned as a digest across machines**: the floating-point
output of a forward. The same weights through the same code give different
bits on an arm64 laptop and an x86 CI runner (different BLAS kernels, fused
or not), so a hex digest of logits pins one machine, not the code. Pin the
*bytes* that are exact everywhere — a state dict's parameters
(`test_weights.py`'s `_STATE_DIGEST`) — and compare outputs **in-process**
against the reference the pin stood for (`torch.equal` against the stock
loader's model in the same process, same kernels on the same bytes).

## End-to-end tests

### Smoke

Run corpus documents through the real CLI on tiny models. Check saved outputs,
shapes, dtypes, and ranges. Separate numerical pins belong in the numerical
tier.

Cover each path that connects steps: fitted artifact to apply, protocol to
script, script to protocol, and resume after a changed dependency. Compare
bounded and unbounded batches within the fixture dtype's tolerance; require
exact equality for coordinates and identities. Caller-owned model tests also
check cleanup after an exception.

Desiderata-Based Masking (DBM) tests cover fit and replay, saved masks, grouped
expert routing, and exports. The joint DBM suite checks exports from both
position and feature gates. `tests/analysis/test_export_dbm.py` compares
exported records with frozen bundles and runs the CLI in a fresh process.

The engine-parity modules under `tests/neural/engines/nnsight_tracing/` run
one document through both engines on the tiny fixtures. They compare values
at 1e-5, and each write must also move its own engine's output.
`test_parity_a3b_sweep.py` and `test_parity_module_boundaries.py` cover the
component vocabulary. The other modules cover these parts:

| Module | Scope |
|---|---|
| `test_parity_harness.py` | The harness, a read listed by two models, and any name for the un-intervened model |
| `test_parity_mechanisms.py` | Every `do` mechanism, write-class order, and both ragged landing policies |
| `test_parity_featurizers.py` | Every applied featurizer kind, grouped and position-axis gates, and the `boundary` gate |
| `test_parity_metrics.py`, `test_parity_receipts.py` | Aggregations, readouts, eligibility, retired `token_form` values, and receipts with and without `--record` |
| `test_parity_positions.py`, `test_parity_generated.py` | Selectors, spans, frames, alignment, bands, path patching, and the generated window |
| `test_parity_components.py`, `test_parity_seams.py` | Components outside the sweep, execution seams, step rules, and engine routing |

`tests/_helpers/engines.py` lists the accepted differences between the
engines, with one comment for each. Add an entry only when you also file the
finding.

### Golden

| Suite | Evidence |
|---|---|
| `tests/golden/test_paper_goldens.py` | Published values with a source, sidedness, and tolerance or floor |
| `tests/golden/test_paper_answer_columns.py` | The paper packages on gated models resolve their metric answers with their tokenizers; tokenizers only, no accelerator |
| `tests/golden/drift/` | Reviewed measurements from this implementation, replayed to detect drift |
| `tests/golden/test_a3b_engine_parity.py` | Reads and writes agree across the two engines |
| `tests/golden/test_a3b_inventory.py` | Loaded model components agree with registry metadata |
| `tests/golden/test_family_certification_a3b.py` | Activations and logits agree with an independent raw-hook oracle |
| `tests/golden/test_readout_a3b.py` | Readout and residual accounting agree with the model |
| `tests/golden/test_engine_parity_mps.py` (Apple MPS) | A read agrees across the two engines on `mps`; skips elsewhere, so it runs only on a local Mac |
| `tests/golden/test_parallel_parity.py` (two CUDA devices) | Inference and fit outputs match the reference geometry (`parallel_goldens.json`), with per-rank memory checks |
| `tests/golden/test_parallel_large.py` (up to eight CUDA devices) | Llama-3.1-70B inference and fits against a `pp=4` oracle |
| `tests/golden/test_parallel_soak.py`, `test_parallel_families.py`, `test_parallel_watchdog.py` (two CUDA devices) | Sustained memory under `ep=2`/`tp=2`; supported geometries on Gemma-2-9b and Llama-3.1-8B; rank-failure diagnostics on NCCL |
| `tests/neural/engines/pytorch_hooks/test_train_parallel_run_tp8.py` (gloo on CPU, about 8.5 GB host memory) | The `tp=8` MoE fit sums the replicated K/V projections' input gradient and lands within the band of world 1 |
| CUDA graph and fused-kernel suites | Replay and kernel outputs and gradients satisfy their numerical contracts |
| `tests/golden/test_multirank_cuda_graphs.py` (two CUDA devices; four for `tp=2,dp=2`) | Graphs at `tp=2`, `ep=2`, `dp=2` and `tp=2,dp=2` write the eager run's outputs byte for byte, and every rank replays |
| `tests/golden/test_replay_deadline.py` (two CUDA devices, `torchrun`) | A replay whose peer left is refused by name at the collective timeout; a healthy world's replays are untouched |
| `tests/golden/test_replay_capture_window.py` (one CUDA device) | A completed event queried from another thread invalidates a capture, and the deadline's capture window keeps that query out |

Paper goldens must cite published results. Drift pins come from a reviewed run
of this stack. Keep these provenance rules distinct when updating fixtures.
The CPU structural tests verify document pins, ownership of each golden value,
and required provenance fields.

Gated checkpoints require licensed access when downloaded.

Memory needs depend on the suite: the A3B model occupies about 70 GB, and the
largest Llama-8B paper fixture needs about 35 GB in bf16. Use
`--batch-rows` where the document and engine support it.

Golden fixtures release resident model caches at module boundaries through
`tests/golden/conftest.py`. Per-test teardown also collects dead tensors and
empties the CUDA cache. Size a module for the models it keeps alive together.

## Lint and Rust checks

Run the lint hooks with:

```bash
uv run \
    --frozen \
    --only-group lint \
    pre-commit run --all-files
```

Run pre-commit in the same command as a local commit, so the commit carries
its formatting changes.

The Rust checks cover formatting and clippy for the workspace and the tests
of `fst-core` and `fst-cuda`. Use `uv run` for clippy so PyO3 sees the
environment's Python. CUDA storage tests skip without a device; ignored Rust
GPUDirect Storage tests require `FST_CUDA_TESTS=1`. See
[fastersafetensors.md](fastersafetensors.md).

Pass `-n 0` to keep the gate in one process if `addopts` gains `-n`:

```bash
uv run pytest \
    -v \
    -n 0 \
    -m "not golden and not measurement_study and not parallel_world"
```
