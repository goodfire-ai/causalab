# Measurement developer guide

The measurement harness profiles **one workflow at one commit**, compares
**one fixed workflow across two code commits**, or compares **two workflow
configurations at one commit**. It collects operation and workflow timings,
numerical observations (optional in single mode), and separate profiling captures.
Start with the [usage quickstart](measurement.md); the
[workflow specification](workflow_protocol.md) defines the document format.
Comparison studies select one axis; simultaneous code and workflow changes
are refused.

## Code map

| Module | Responsibility |
| --- | --- |
| `causalab/measurement/spec.py` | Strict, stdlib-only study configuration |
| `causalab/measurement/plan.py` | Typed single/comparison plans and internal target normalization |
| `causalab/measurement/collection.py` | Operation collection and measurement records |
| `causalab/measurement/study/` | Paired scheduling, orchestration, common and crossed evaluation |
| `causalab/measurement/runtime/` | Isolated workers, source pins, observations and training instrumentation |
| `causalab/measurement/capture/` | Separate profiling passes and named ranges |
| `causalab/measurement/analysis/` | Comparisons, stability metrics and reports |
| `causalab/measurement/deployment/` | Source installation, remote launch and artifact transfer |
| `causalab/profiling/` | Torch and Nsight adapter contracts |
| `causalab/remote/` | Shared SSH transport and job supervision |

The workflow loader imports the measurement grammar lazily; `workflow/` has no
module-level dependency on measurement execution. Package initializers and
configuration parsing remain free of numerical imports. Analysis scripts import
numerical dependencies inside their entry points so document validation and
script hashing remain lightweight.

## Single-source execution

Single-source plans declare `mode: "single"` and one `source`. The typed plan
normalizes this into one internal target for scheduling and installation; public
documents and bindings retain the `source` field. That source must satisfy authored
code pins before its resolved pins are frozen.

Omitted observations skip numerical passes and tensor exports and are reported as
`not_requested`. Workflow outputs, integrity checks, dispatch probes, timing and
captures still run. Requested observations must be nonempty; missing evidence
fails collection.

Receipt validation covers single-source and paired studies. Single-source reports
show per-seed timings, scoped memory and traces, including omitted checks and
unavailable memory. Collection and profiler outcomes are independent; single mode
has no comparison verdict. Numerical comparisons require observations.

## Benchmark definitions and isolated source arms

`study/definitions.py` resolves each arm's workflow and intervention specifications
under a shared measurement plan. Code mode uses one definition for all arms.
Workflow mode uses the owning document for `before`, a relative `after.workflow`
document for the candidate, and an optional `eager.workflow`. Selected documents
cannot contain another measurement plan. Their cases, observations and evaluation
steps must satisfy the shared interface.

Each worker reopens its selected inputs and recomputes their benchmark identity
before execution. Changes after preparation refuse the run. In workflow mode,
all source revisions must resolve to the same commit. Dataset, artifact,
checkpoint and logical token input evidence remains shared, including the data-role
declarations of observed steps. Runtime dtype and quantization may change without
changing checkpoint contents, provided saved observations still align.

Workers load the controller's measurement helpers under a private package while
`causalab` resolves to the selected arm's source installation. This lets the same
harness exercise different commits without requiring both commits to contain the
current harness. Each selected revision must support the workflow and engine APIs
used by the study.

For code comparisons, `runtime/pins.py` separates two kinds of pin:

- **Source pins:** direct `causalab` code and script modules verified as regular
  files inside the selected installation. Their bytes may differ between commits.
- **Shared pins:** intervention specifications, datasets, external files, local
  scripts, external modules and closure members. These stay fixed across arms.

Authored source pins assert the `before` revision. The `after` arm and optional
`eager` reference resolve their own source pins. Each arm then freezes its entire
census for later full loads, selected-step execution, cold workers, captures and
resume. Cross-arm checks still require identical shared inputs. Candidate source
files must exist before remote launch; only the baseline must match the authored
source hashes. Keep authored pins intact when preparing the worker payload.

Workflow comparisons assert each arm's entire authored pin census against its own
definition, then freeze it for execution and resume. Source-pin relaxation applies
only to code comparisons. Remote packaging includes each selected definition and
checks its source pins before launch.

Reports show the comparison mode, each worker's source commit and benchmark
identity, and the exact authored configuration changes. Expected differences in
the selected comparison axis are recorded; missing or mismatched shared input
evidence remains an error. Workflow comparisons describe configured output
contrasts. Different sites, reads, training objectives or budgets do not establish
semantic equivalence or equal-work speedups.

## Collection boundaries

Keep these timings separate:

| Boundary | Scope |
| --- | --- |
| Operation | Prepared inputs through synchronized completion of the selected operation |
| Resident workflow | Workflow execution with the model already resident, including required output publication |
| Cold process | Child creation through exit, including imports, loading, execution and required publication |

Cold process does not imply cold filesystem or model-download caches. Record
preparation and reset policy with the result. Required outputs belong inside the
timing boundary; additional numerical diagnostics and profiling use separate
passes. Operation speedups do not establish end-to-end speedups.

The scheduler serializes competing work and balances arm order. A seed, an
independent fit repeat, a timing repeat and an evaluation example are distinct
sampling units. Reset RNG and scientific state at the declared boundaries;
reusing a completed block never creates another replicate.

## Numerical evidence

Align observations by logical example or pair, input role, site and real-token
position. Missing, duplicate or ambiguous identities must not silently compare
as matching values. Preserve nonfinite and coverage counts alongside drift
metrics.

Report after-minus-before changes separately from variability within a seed and
across seed means. Variance needs independent repeats; a zero observed baseline
variance does not establish determinism or justify a hidden epsilon in a ratio.
The optional eager arm provides context, not numerical ground truth.

For fitted interventions, common evaluation compares learned artifacts under a
fixed evaluator. Crossed evaluation also checks their behavior under each arm's
evaluator. Preserve the logical training objective, data order and update budget
when measuring implementation changes.

Acceptance thresholds are authored per study. A completed run without criteria
is descriptive evidence, not an equivalence claim. Missing evidence must remain
visible in the report.

## Profiling adapters

`causalab/profiling/base.py` defines the adapter boundary: validate options, probe
availability, construct commands, describe capture coverage and name artifacts.
The controller owns scheduling, process lifetime, timeouts, resets and publication.
Add backend-specific command construction to adapters rather than orchestration.

Managed captures follow the clean timing phase. Cold workflow captures include a
linked resident capture; each records its actual coverage. Torch cannot cover
activity before its initialization, Nsight Systems provides launch tracing, and
Nsight Compute collects selected kernel counters. Counter-profiler durations
must not become benchmark timings.

Native traces are opaque, hashed artifacts with receipts. Capture failure leaves
completed clean measurements available and marks capture evidence incomplete.
Use existing viewers; trace analysis and optimization recommendations are outside
the harness's current scope.

## Publication and resume

Publish structured evidence as JSON and dense observations as safetensors. Keep
raw measurements, traces and generated reports outside committed source.
Remote execution uses the same study entry point with separate deployment bindings
for source, environments and data access.

The scheduler holds an exclusive controller lock and publishes a block only when
every declared source completes. It verifies artifact hashes before reuse. Resume
re-attests each source and refuses changed study, source, input, environment or
execution identities. Interrupted blocks cannot contribute complete evidence.

`measurement/paths.py` hashes Python sources recursively under `measurement/`,
`profiling/` and `remote/`, using relative paths to distinguish nested modules.
Presentation-only report modules and launch-only preflight modules are explicitly
excluded from the resume contract; report code has a separate identity. Validators,
installation and transfer helpers remain covered. Even comment-only changes in
covered files change the source identity and require a fresh study.

## Development checks

Follow the repository's [test guide](TESTS.md). Measurement tests mirror the
package layout. Preserve these invariants when extending the harness:

- A real workflow runs against distinct commits and an unchanged-commit control.
- Distinct workflow documents run at one commit; mixed-commit workflow studies fail.
- Every worker, cold process, capture and evaluator consumes its selected definition.
- Workers attest the selected package and refuse changed shared inputs or frozen pins.
- Reset and synchronization order preserve timing boundaries.
- Numerical oracles cover alignment, nonfinite values, drift and both variance levels.
- Incomplete captures and interrupted publications cannot appear complete.
- Resume rejects changed identities and modified artifact bytes.
- Package loading stays lightweight and standalone workers and transfer commands work.

Run focused CPU tests before opt-in model or GPU measurements. Hardware results
are study evidence; they do not define universal numerical tolerances.
