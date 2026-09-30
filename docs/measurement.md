# Workflow measurements and profiling

Measure and profile **one workflow at one commit**, compare **one workflow across
two code commits**, or compare **two workflow configurations at one commit**.
All modes use isolated source installations on one CPU or CUDA device and collect
clean timings separately from Torch or Nsight captures.

Choose a quickstart: [one commit](#quickstart-measure-one-commit),
[two commits](#quickstart-compare-two-commits), or
[workflow configurations](#quickstart-compare-workflow-configurations).
Comparisons share datasets, input artifacts, checkpoint, seeds and observation
selections. Code comparisons also keep workflow and intervention specifications
fixed; workflow comparisons report authored configuration changes. See the
[developer guide](optimization_experiments_design.md) for the architecture.

## Quickstart: measure one commit

For an offline CPU demonstration with a locally generated tiny model:

```sh
uv run python -m examples.measurements.single /tmp/causalab-single
```

Use a fresh directory and prepare the [offline build prerequisites](#run-on-a-remote-computer).
The example installs `HEAD` and collects resident/cold workflow timings and Torch
traces. Open `/tmp/causalab-single/run/reports/index.html`. Options:

- `--revision`: select another commit.
- `--observations`: check saved logits.
- `--no-profile`: collect timing without traces.
- `--prepare-only`: generate inputs without running.

For your own workflow, add this block; existing steps and saves stay as authored:

```json
"measurement": {
  "version": 1,
  "mode": "single",
  "source": {"revision": "HEAD"},
  "cases": {
    "resident": {"kind": "workflow", "cold_process": false},
    "cold": {"kind": "workflow", "cold_process": true}
  },
  "seeds": [0],
  "repeats": 3
}
```

`source.execution` accepts `engine`, `batch_rows` and `cuda_graphs`, as described
in [study authoring](#author-a-paired-study). Operation cases select one intervention
step. Declare cases, seeds and repeats explicitly; `warmups` defaults to one.
Probes, warmups, clean samples and captures execute separately.

Supply source and data paths in `bindings.json`:

```json
{
  "source": {"repository": "/path/to/causalab", "python": "/path/to/.venv/bin/python"},
  "device": "cuda:0",
  "data_root": "/path/to/data",
  "artifacts_root": "/path/to/artifacts"
}
```

```sh
uv run causalab measure workflow.json \
    --bindings bindings.json \
    --out /tmp/single-run
uv run causalab measure workflow.json \
    --bindings bindings.json \
    --out /tmp/single-run \
    --resume
```

Omit `observations` to skip numerical checks; workflow saves and artifact integrity
checks still run. To check outputs, declare a nonempty
[observation table](#author-a-paired-study). Missing or invalid requested outputs
fail collection; omitted checks are reported as `not_requested`.

Omitting `profile` enables Torch captures; `false` disables them. Cold cases receive
cold and warm captures; resident cases receive warm captures. Captures follow
clean timing; setup probes are untimed. Profiler failures preserve completed
measurements. See
[profiling options](#profiling) for Nsight backends and capture coverage.

The report shows timing samples and variation, source/input identity, execution
evidence, trace links and capture outcomes. Memory is reported for resident
execution; cold-process peak memory is unavailable.

Single mode rejects `arms`, common/crossed `evaluation`, `bootstrap_draws` and
`acceptance`. Evaluation steps already present in the workflow still run.
The [Qwen single-source preset](../examples/measurements/qwen/single.json) provides
resident and cold cases over fitting, controls and held-out replay.

For [remote execution](#run-on-a-remote-computer), use single-source bindings.
The source must match authored source pins.

To benchmark a discovered optimization, replace `mode`/`source` with `arms.before`
and `arms.after`, add numerical observations and use a fresh output directory.
Keep cases, data, saves and timing scopes fixed. Timing-only evidence cannot
retroactively establish numerical equivalence. Saved collections with matching
observations can also use the [comparison script](#compare-saved-collections-through-a-workflow).

All modes require **one device and one rank**. Use `cpu`, `cuda`, or `cuda:N` in
bindings and launch without a multi-rank launcher such as `torchrun`. Multiple
GPUs may be visible; only the selected device is measured. Distributed launches
and multi-device selections are rejected.

## Quickstart: compare two commits

Run from the repository root with project dependencies installed, the example's
pinned Qwen checkpoint cached, and source build prerequisites available offline
(see [deployment setup](#run-on-a-remote-computer)). The example requires enough
GPU memory for Qwen3.6-35B-A3B and its fitting activations; an 80 GB GPU is suitable.
Replace `HEAD~1` and `HEAD` with your baseline and candidate commits. Both must
support the example's v3 intervention specifications. Uncommitted edits are not measured.

This copies the two-update study, keeps its two seeds and two repeats, and disables
profiling for the first run:

```sh
MEASURE_DIR="$(mktemp -d /tmp/causalab-measure.XXXXXX)"
uv run python - "$MEASURE_DIR" HEAD~1 HEAD <<'PY'
import json
from pathlib import Path
import shutil
import sys

repo = Path.cwd()
work = Path(sys.argv[1])
shutil.copytree(repo / "examples/measurements/qwen", work / "study")
document = work / "study/study.json"
study = json.loads(document.read_text())
for arm, revision in zip(("before", "after"), sys.argv[2:]):
    study["measurement"]["arms"][arm]["revision"] = revision
study["measurement"]["profile"] = False
document.write_text(json.dumps(study, indent=2) + "\n")
bindings = {
    "arms": {
        arm: {"repository": str(repo), "python": sys.executable}
        for arm in ("before", "after")
    },
    "device": "cuda:0",
    "data_root": str(work / "study/data"),
    "artifacts_root": str(work / "study"),
}
(work / "bindings.json").write_text(json.dumps(bindings, indent=2) + "\n")
PY
uv run causalab measure "$MEASURE_DIR/study/study.json" \
    --bindings "$MEASURE_DIR/bindings.json" \
    --out "$MEASURE_DIR/run"
uv run python -c 'import pathlib, sys, webbrowser; webbrowser.open(pathlib.Path(sys.argv[1]).as_uri())' "$MEASURE_DIR/run/reports/index.html"
```

Open `$MEASURE_DIR/run/reports/index.html`. Check the shared benchmark identity,
both resolved commits, source hashes, comparison caveats and numerical variation
before interpreting speedups. Keep the directory as evidence. To resume, repeat
the measurement command with `--resume` and unchanged inputs. For an A/A control,
use the same commit twice and a fresh output directory.

These short fits check the setup. Use the [fixed-work and normal-workflow
presets](../examples/measurements/qwen/README.md) for longer comparisons.

### Benchmark a change to a pinned callable or script

Authored `pins.code` and `pins.scripts` for directly referenced `causalab` modules
assert the **baseline** source. Each candidate or eager arm resolves and freezes
its own hashes from its committed installation. To benchmark a pinned callable:

1. Keep the workflow, intervention specifications, data and input artifacts fixed.
   If stamping new pins with `causalab pin`, run it in the baseline environment.
2. Commit the implementation edit and set the `before` and `after` revisions.
3. Run into a fresh directory and check the reported commits, changed source hash
   and numerical results.

Do not re-stamp the shared workflow against the candidate. Document, dataset and
input-file pins remain shared assertions. External modules, path-based scripts
and sibling source closures also remain fixed inputs for local runs. Both commits
must expose the APIs and logical outputs required by the benchmark.

Without authored pins, the harness freezes the initial input hashes and refuses
shared-input differences. Workers verify the workflow and resolved intervention
specifications before loading models. Deferred external artifacts are hashed before
measurement. Source and input hashes are checked again on subsequent loads,
profiler captures and resume. Source-pin maps cover referenced modules; installation
receipts separately attest the whole package. Remote source restrictions are
listed [below](#run-on-a-remote-computer).

## Quickstart: compare workflow configurations

Use the same dependencies, cached checkpoint and GPU as the two-commit quickstart.
This compares the Qwen study's DAS learning rate of `0.001` with `0.0005`, keeping
the two-update training budget and common evaluation fixed. Both arms use the
same resolved `HEAD` commit; the candidate workflow has no measurement block.

```sh
MEASURE_DIR="$(mktemp -d /tmp/causalab-workflows.XXXXXX)"
uv run python - "$MEASURE_DIR" <<'PY'
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo = Path.cwd()
work = Path(sys.argv[1])
shutil.copytree(repo / "examples/measurements/qwen", work / "study")
document = work / "study/study.json"
study = json.loads(document.read_text())
candidate = copy.deepcopy(study)
del candidate["measurement"]
candidate["steps"]["das"]["set"] = {"train.optimizer.lr": 0.0005}
(work / "study/candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
plan = study["measurement"]
plan["comparison"] = "workflow"
for arm in plan["arms"].values():
    arm["revision"] = revision
plan["arms"]["after"]["workflow"] = "candidate.json"
plan["profile"] = False
document.write_text(json.dumps(study, indent=2) + "\n")
bindings = {
    "arms": {
        arm: {"repository": str(repo), "python": sys.executable}
        for arm in ("before", "after")
    },
    "device": "cuda:0",
    "data_root": str(work / "study/data"),
    "artifacts_root": str(work / "study"),
}
(work / "bindings.json").write_text(json.dumps(bindings, indent=2) + "\n")
PY
uv run causalab measure "$MEASURE_DIR/study/study.json" \
    --bindings "$MEASURE_DIR/bindings.json" \
    --out "$MEASURE_DIR/run"
uv run python -c 'import pathlib, sys, webbrowser; webbrowser.open(pathlib.Path(sys.argv[1]).as_uri())' "$MEASURE_DIR/run/reports/index.html"
```

Check the report's per-arm benchmark identities, shared resolved commit and exact
authored changes. Numerical results compare the configured outputs; different
training settings, reads or sites do not establish semantic equivalence. Changing
the update budget also changes the work being timed. Use common evaluation to
compare learned artifacts on the same held-out task. The
[workflow comparison example](../examples/measurements/workflow_comparison/README.md)
describes further variations.

## Author a paired study

Add a `measurement` block to a normal workflow. Required fields are `version`
(integer `1`), `arms`, `cases`, `seeds`, `repeats` and `observations`.

| Field | Meaning |
| --- | --- |
| `comparison` | `code` (default): fixed workflow, selected commits; `workflow`: selected documents, one resolved commit |
| `arms.before`, `arms.after` | Each supplies a Git `revision`; optional `eager` adds a reference arm |
| Arm `workflow` | Workflow mode only: required for `after`, optional for `eager`; a path relative to the study directory |
| Arm `execution` | `engine` (currently `pytorch_hooks`), `batch_rows` and `cuda_graphs` |
| `cases.<name>.kind` | `operation` selects one protocol `step`; `workflow` runs the whole workflow |
| `cases.<name>.cold_process` | Workflow-only boolean, default true; false measures a resident workflow |
| `observations.<name>` | Saved `step`, `file` and `kind`: tensor, table, subspace or gate |
| Table observations | Also declare `row_keys` and a numeric `value` column |
| Gate observations | `temperature` defaults to `"fit"`; optional `featurizer` resolves ambiguous fit diagnostics |
| `warmups` | Fresh-state warmups per cell, default 1 |
| `order_seed` | Reproducible randomized paired block ordering, default 0 |
| `profile` | `false` disables native profiling; otherwise selects cases, backends and capture options |

In workflow mode, `before` uses the study document and cannot specify `workflow`.
An eager arm without `workflow` also uses that document. Selected documents must
stay within the study directory and contain no `measurement` block. Their
intervention document paths are relative to their own directories. All revisions,
including eager, must resolve to the same commit; different ref names are allowed.
Both code and workflow changes in one study are refused.

Cases, observations and evaluation step names must exist in every selected
workflow. Observed steps keep the same data-role declarations, datasets, input
artifacts, checkpoint and logical token inputs. Workflow steps, training settings,
reads, sites, dtype and quantization may differ when those inputs and the saved
observation alignment remain comparable. Additional steps remain subject to the
same input constraints. Workflow mode checks every arm's authored pins against
that arm's own definition, including source pins; code mode's baseline-only source
pin rule does not apply.

Use saved table column names for `row_keys`. A swept DAS table can need
`example_id`, `metric`, `sites.target.layers` and `featurizers.rot.k`; coordinates
are separate columns, not a single `coords` column. Tensor bundles use shortened
coordinate names.

Each fitting seed sets the protocol's `train.seed`; explicit featurizer
initialization seeds remain authored choices. All arms must use the same
`batch_rows` for workflows containing fitting. Physical inference batching may
differ. The engine must support the requested execution options, but can fall back
to eager execution for an ineligible workload. Check observed graph replay in the
report, not just the requested `cuda_graphs` setting.

Gate observations use the saved fit's actual temperature, including after early
stopping. A fixed artifact without fit diagnostics needs a numeric temperature;
a numeric declaration is checked against diagnostics when present. Only sigmoid
parametrization is supported. Missing, conflicting or unsupported gate evidence
is refused; unstamped legacy gates use sigmoid as in the protocol loader.

### Local bindings and resume

Bindings supply machine paths separately from the study:

```json
{
  "arms": {
    "before": {"repository": "/path/to/causalab", "python": "/path/to/before/.venv/bin/python"},
    "after": {"repository": "/path/to/causalab", "python": "/path/to/after/.venv/bin/python"}
  },
  "device": "cuda:0",
  "data_root": "/path/to/data",
  "artifacts_root": "/path/to/artifacts"
}
```

```sh
uv run causalab measure study.json \
    --bindings bindings.json \
    --out /tmp/study-run
uv run causalab measure study.json \
    --bindings bindings.json \
    --out /tmp/study-run \
    --resume
```

The controller requires Git and uv. It builds committed snapshots into private
wheel installations using each selected Python's prepared dependencies, leaving
those environments and checkouts unchanged. Imports, metadata, source commits,
installed files and checkpoint contents are checked before collection. Startup
and resume rehash checkpoints; later workers reuse hashes only while file metadata
remains unchanged. Keep source documents and external inputs available throughout.

Only one arm process holds its model at a time. Complete paired blocks contain all
arms; failed blocks contribute no unpaired samples. Resume rechecks workers and
completed artifacts, reruns incomplete blocks and preserves completed ones. Locks
and worker leases prevent overlapping recovery. Failed builds retain logs and can
resume after missing prerequisites are supplied. Results are under `reports/`,
collections under `collections/` and diagnostics under `workers/`.

### Timing and execution evidence

| Case | Timed scope | Outside the clock |
| --- | --- | --- |
| Operation | Selected engine call, tokenization not served by retained caches, execution and required saves | Compilation and ancestor steps |
| Resident workflow | Workflow steps and output publication | Document loading and resident model preparation |
| Cold-process workflow | Process creation through exit, including imports, source/environment checks, model loading and publication | Preceding session's probe and checkpoint hash attestation |

Resident passes reset RNG and fit/optimizer state while retaining frozen models
and runtime caches. Probes, warmups and earlier passes can populate tokenization,
metric-token, table-validation, gather-index and kernel-selection caches. Receipts
record this in `reset_policy` and `context.cache_policy`; the report flags differing
cache capabilities. These records describe known available caches, not cache hits,
occupancy or an exhaustive inventory. Cold processes start fresh, but neither
cold runs nor captures flush OS or Hugging Face disk caches.

An untimed resident probe runs for every case before collection and on resume.
It records model-input shapes, dtypes and hashes, logical token rows, backend
functions and source hashes, Torch operators and CUDA kernels. Arms must observe
the same logical token rows; padding, batching and repeated forwards may differ.
Unobservable inputs are refused. Model-input hooks skip CUDA graph capture and
do not independently sample every replay's staged inputs; graph dispatch and
fitting minibatch schedules are recorded separately. Python observation covers
the calling thread. For cold cases, this resident probe does not attest startup
dispatch or run inside the timed process.

Resume checks input and dispatch evidence alongside source, data, environment and
hardware identity. CUDA runs require GPU UUID and driver evidence. Execution
switches recorded include `CAUSALAB_MOE_GLUE`, `CAUSALAB_FUSED_NORMS`,
`CAUSALAB_GDN_SHORT_SEQ`, `CAUSALAB_PROJECT_HEAD_UNDER_GRAD` and
`CAUSALAB_COMPILE_CACHE`, including unset values. Changing one refuses resume;
arbitrary environment variables and credentials are not recorded.

Fitting diagnostics observe initial parameter/stage and optimizer-state hashes,
logical batch order, optimizer-call counts and Python/NumPy/Torch RNG boundaries.
They run separately from timing, with an additional cold diagnostic for cold fits.
The **timing pass's saved outputs are authoritative** for resident and cold
numerical comparisons and common evaluation. Diagnostic disagreement is reported
without replacing those outputs; it can reflect variation or observer effects.
Different update counts or schedules indicate potentially different work, not a
same-work speedup. Matching boundary RNG states does not prove identical random
consumption, and raw parameter hashes can differ for equivalent representations.

## Common evaluation and study reports

Add non-fitting protocol steps that load fitted artifacts and evaluate the desired
data. Their artifact references, split, metrics and observations define the
population. Map measurement cases to these steps in `measurement.evaluation`:

```json
{"arm": "before", "crossed": true, "cases": {"training": ["das_eval", "dbm_eval"]}}
```

With `crossed: true` (the default), every arm's evaluator evaluates every fit.
With `false`, only the selected `arm` evaluates them. Evaluation occurs outside
timing, checks fit trees for mutation and must succeed before the paired block
is published. Report keys `evaluation__before__` and `evaluation__after__` compare
fits under the corresponding evaluator. Native observations remain separate;
native profiling does not capture these additional evaluation outputs.

`reports/index.html` links per-case evidence and native traces; `study.json`
contains structured results. The primary contrast is `before_after`; an optional
`eager` arm adds `eager_before` and `eager_after`. Each report records the shared
benchmark identity, commits, source and shared-input hashes, execution settings,
observed graph replay, diagnostic checks and variation. Expected source changes
are not caveats. Empty hash maps remain distinct from missing evidence.

### Interpret numerical results

Reports separate paired runtime changes, tensor drift, within-seed repeat
variation and across-seed variation. Tensor drift includes max-absolute, RMS,
relative RMS and cosine differences computed in float64. Scalar observations
include raw values, means and sample standard deviations. Subspaces use
basis-invariant principal angles, overlap and projector distance. Gates report
probability drift, hard-mask overlap, selection frequency and boundary confidence.
Declare task metrics separately: stable subspaces or masks do not establish
usefulness.

Bootstrap intervals resample seeds, not individual coordinates or all seed pairs.
They are descriptive, not calibrated regression gates. Few seeds limit precision;
identical-code runs can still show timing drift and intervals excluding zero.
Use A/A controls and independent process/session replication before judging small
speedups. Intervals require at least two seeds; variance-change intervals also
require repeats within each seed. Variance uses `n - 1` and requires two samples.
Zero-denominator relative metrics are `null`; zero observed variance does not
prove determinism. No universal scientific threshold is applied.

### Acceptance bounds

Optional `measurement.acceptance` entries have a unique `name`, a `path` array
into `study.json`, and exactly one numeric `minimum` or `maximum`. Path segments
select object keys or decimal array indices. For example, the path
`["cases", "training", "comparisons", "before_after", "timing",
"mean_paired_seconds_change_ci", "interval", "1"]` addresses the runtime-change
interval's upper endpoint.

Equality satisfies a bound. Violations yield `failed`; missing, null, nonnumeric
or nonfinite evidence yields `insufficient_evidence`. Acceptance is a study-report
verdict, separate from successful collection. Choose bounds for your study.

## Profiling

Set `measurement.profile` to `false` (or use an empty `profile.cases` list) to skip
native preflight profiling and captures while retaining timing, memory, numerical
outputs and input/fitting diagnostics. Omitting `profile` selects all cases with
Torch. Explicit `backends` replaces that default:

```json
{
  "profile": {
    "cases": ["cold_workflow"],
    "timeout_seconds": 3600,
    "backends": {
      "torch": {"record_shapes": false, "with_stack": false},
      "nsys": {"cuda_graph_trace": "node", "cuda_memory_usage": false},
      "ncu": {"set": "basic", "launch_count": 10}
    }
  }
}
```

Choose existing case names. `cases` defaults to all, `timeout_seconds` to 3600 per
capture; optional `reason` records the investigation purpose. Top-level
`record_shapes` and `with_stack` remain supported for Torch; conflicting options
fail. Profiling settings are part of study identity: change them before launch
and use a fresh output directory or remote receipt.

Captures run **after all clean timing blocks**, outside benchmark clocks, at the
first authored seed. Cold-process cases receive cold and warm captures per arm
and backend; other cases receive warm captures. Warm captures run an unprofiled
warmup and reset scientific state, retaining model/runtime caches. Setup performed
by each fresh fit remains inside its captured workflow. Cold outputs are checked
against clean cold outputs, warm outputs against clean resident outputs.

| Backend | Options and coverage |
| --- | --- |
| `torch` | Boolean `record_shapes`, `with_stack`; Chrome trace JSON. Cold capture starts after Torch import and profiler initialization, before model construction. |
| `nsys` | `cuda_graph_trace`: `node` (default) or `graph`; `cuda_memory_usage`: false by default. CUDA/NVTX/OS-runtime `.nsys-rep`, without CPU/context-switch sampling. Cold capture spans process launch through export and exit. |
| `ncu` | `set`: `basic` by default, or `sections`/`metrics` lists; `kernel_name` exact or `regex:` filter; `kernel_name_base`: `demangled`, `function` or `mangled`; `launch_skip`: 0; `launch_count`: 10. `.ncu-rep` records selected kernels, including initialization kernels in cold runs. |

NCU uses kernel replay, `cache_control: all` by default (or `none`) and
`clock_control: none`. Replay can serialize work and change caches; its durations
never replace clean wall times. Launch bounds limit counters while the complete
workflow still saves outputs. Stock metric sections are used. Kernel filters do
not select optimizer-update windows, and NCU is not a CPU startup timeline.

Provide `nsys`/`ncu` on the compute host's PATH with instrumentation permissions.
The harness records tool versions, commands, coverage, output checks and logs;
it does not install tools or change permissions. Missing tools, unsupported flags,
counter failures, empty artifacts and timeouts leave clean measurements intact
and appear as failed/unavailable captures or incomplete pairs. Torch profiling is
not nested inside Nsight. Resume verifies completed capture groups and reruns
interrupted groups without repeating clean timing. Completed failures stay
recorded; use a new study to retry with changed options or environment.

### Inspect large traces

Reports link native artifacts and capture coverage. Open Nsight artifacts in the
matching NVIDIA tools. For large Torch traces, use the optional
[Perfetto native trace processor](https://perfetto.dev/docs/visualization/large-traces)
to avoid browser memory limits.

Trace ranges cover case, workflow step, training, forward, metric, regularizer,
evaluation, backward and optimizer calls. Python boundaries cover the calling
thread; native events retain device/autograd work on other threads. An available
boundary need not execute on every path: graph overrides and cached replay can
bypass the observed base forward method. Forward call counts are therefore not
comparable across eager/graph modes without checking coverage, and an optimizer
range is not an entire gradient update. Input snapshots and Python dispatch
observation add overhead; matching saved outputs does not prove unchanged dispatch.

## Run on a remote computer

Use the same study with bindings whose `repository` paths refer to local source
repositories and whose `python`, `data_root` and `artifacts_root` refer to the
compute host. Prepare dependencies in each selected remote Python environment, cache the
model, and provide uv plus offline source-build prerequisites: maturin, the Rust
toolchain pinned in `rust-toolchain.toml` and Cargo dependencies. Building each
revision once is one way to populate the build cache. Both the launcher and remote
control interpreter (`--control-python`) need `tarfile.data_filter` support.

The launcher uploads immutable arm archives and a committed controller snapshot;
the host needs no Git checkout. Private source wheels are built there for the
selected Python, outside timing. Receipts record archive, wheel, installed-file
and ABI/platform identities; resume verifies them. Prepared Python environments
are unchanged. Missing build prerequisites produce a build log and can be supplied
before resuming.

Before SSH, single-source studies check authored source pins against their source
archive; code comparisons check them against the baseline archive.
Candidate/eager pinned modules must still be regular source files; changed hashes
are allowed, but missing, renamed or linked files are refused. Workflow comparisons
check every arm's pins and package the selected documents separately, retaining
their relative intervention paths. External-module and
sibling-closure pins are refused because the archives cannot establish their bytes.
Workers check the full pin census, datasets and artifacts. Remote packaging supports
intervention steps and module-located scripts; file-located scripts and other step
kinds (including behavioral, decision, conditional and nested workflows) are refused.

Set `MEASURE_HOST` to your SSH destination and `MEASURE_HF_CACHE` to the remote
Hugging Face hub cache path, then run:

```sh
uv run python -m causalab.measurement.deployment.remote launch study.json \
    --bindings remote-bindings.json \
    --host "$MEASURE_HOST" \
    --hf-cache "$MEASURE_HF_CACHE" \
    --receipt measurement-job.json
uv run python -m causalab.measurement.deployment.remote status --receipt measurement-job.json
uv run python -m causalab.measurement.deployment.remote logs --receipt measurement-job.json
uv run python -m causalab.measurement.deployment.remote resume --receipt measurement-job.json
uv run python -m causalab.measurement.deployment.remote fetch \
    --receipt measurement-job.json \
    --out /tmp/measurement-results
```

The receipt is created before upload; keep it after disconnection and inspect
status instead of launching a duplicate. `cancel` uses the same receipt. Execution
is detached on one already allocated node/device. Downloads are disabled unless
`--allow-download` is supplied. `--upload-data` and `--upload-artifacts` accept
local directories and update the corresponding remote bindings.

After termination, `fetch` verifies exported hashes and writes a fresh local
directory, including `local_reports/` with working trace links. Failed jobs can
also be fetched for diagnosis. `resume` rechecks a terminal study and continues
verified blocks; active supervisor/controller/worker leases refuse restart.
An upload that never initialized a remote job needs a fresh launch.

## Run the offline demonstration

```sh
uv run python -m examples.measurements.smoke /tmp/causalab-measurement-smoke
```

Use a fresh directory. This runs a subspace operation and a resident inference
workflow over a randomly initialized tiny GPT-2 as independent no-change controls,
with two seeds, two repeats and CPU traces. It demonstrates collection, not model
quality or representative GPU performance. Reports are in
`comparison/subspace_apply/` and `comparison/workflow/`, each with `summary.json`
and `report.html`. Generated `compare.json` demonstrates workflow-based analysis.

## Collect an operation

For custom adapters, `causalab.measurement.collect` accepts a preparation context
manager that restores scientific state for each seed and yields an `Operation`
with `run()` and `observe(result)`. Use the selected source arm's implementation.
Supply a case name, input identity, timing scope and reset policy matching across
arms. Put closure state, model/data identities and geometry in `context`; these
are declarations, not automatic attestations. The collector records package and
adapter source identities, environment and device.

Collection performs fresh-state warmups, a synchronized timed call for each
seed/repeat, a separately prepared numerical pass, and an optional Torch profiling
pass for the first measured seed/repeat. Preparation, cleanup, observation extraction
and serialization are outside the timer. `Operation.numerics_context` observes
only the numerical pass. The caller must apply seeds and reset parameters,
optimizer and RNG; the collector does not infer reset policy or change grad mode.
To measure cache-fill cost, clear caches during preparation and fill them in
`run()`. GPU peak allocated/reserved memory is recorded for timing.

Observations are nonempty dense real tensors keyed by logical examples/sites.
Align coordinates and exclude padding; key/shape checks cannot infer token identity.
Analysis loads observations into host memory, so keep captures bounded.

`causalab.measurement.workflow.workflow_operation` adapts loaded workflows and
caller-owned engines using fresh output directories with resume disabled.
Compilation and engine preparation are outside its clock; lazy model loading
during execution remains inside. Label residency accordingly.

Collections exclusively create their directory and atomically write
`measurement.json` for complete timing/numerical pairs. Numerical failures leave
a failed receipt; profiling failures preserve completed measurements. Profiling
outputs are retained for comparison. Fitting adapters use the timing pass's saved
outputs and observation specifications, including DBM temperature; other adapters
use the numerical pass. Generic collectors report dispatch as unknown without an
adapter-supplied observer.

## Compare saved collections through a workflow

Use a script step with module locator `causalab.measurement.analysis.compare`:

| Input | Meaning |
| --- | --- |
| `before`, `after` | Workflow path references to completed measurement receipts |
| `before_sha256`, `after_sha256` | Receipt content hashes included in workflow identity |
| `bootstrap_draws` | Integer at least 100; default 1000 |
| `bootstrap_seed` | Nonnegative integer; default 0 |

Declare outputs `summary` with a JSON filename and `report` with an HTML filename.
The offline demonstration's `compare.json` is runnable. Analysis checks receipt,
observation and trace hashes. Run analysis without resume to repeat integrity
checks; workflow analysis reuse and measurement collection resume are separate.

Cases, timing scopes, input/reset identities, seed/repeat pairs and observation
keys/shapes must match. Analysis refuses incomplete collections, duplicate/missing
samples, invalid durations, altered artifacts and nonfinite/complex observations.
Dtype/provenance differences remain visible. Matching seed labels do not establish
matching hardware, environment or random consumption.

## Supported coverage and limitations

All three modes support `pytorch_hooks` on one CPU or CUDA device, fixed token inputs,
inference, artifact replay, DAS/DBM fitting and resident/cold workflows. Comparison
mode also supports common/crossed evaluation and optional eager controls. Selected
engines must support the authored APIs and observable inputs; they need not include
the measurement API.

Combined code-and-workflow changes, other engines, distributed execution,
free-running generation with diverging tokens, update-window capture, integrated
trace analysis and automatic optimization recommendations are outside this
version's contract.

Keep receipts, observations, traces and generated reports outside committed source,
for example in ignored `results/` or an artifact store. Commit reusable intervention
specifications and small fixtures; record run-specific validation in the PR.
