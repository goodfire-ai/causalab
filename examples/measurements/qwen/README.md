# Qwen measurement examples

Profile one commit, compare baseline and candidate commits with a fixed workflow,
or use a preset as the baseline for a
[workflow configuration comparison](../workflow_comparison/README.md).
Every preset uses Qwen3.6-35B-A3B in bf16 at checkpoint
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Comparisons share data, checkpoint and
seed schedule; code comparisons also keep intervention specifications fixed.

## Quickstart

- **Profile one commit:** use `single.json` and the
  [single-source quickstart](../../../docs/measurement.md#quickstart-measure-one-commit).
  Set `source.revision`, use `source` bindings, and point `data_root` to this
  directory's `data/`. Numerical observations are optional.
- **Compare two commits:** follow the
  [comparison quickstart](../../../docs/measurement.md#quickstart-compare-two-commits)
  with `study.json`. Replace both `HEAD` revisions; use identical commits for an
  A/A control.

Uncommitted edits are not measured. Selected commits must support v3 intervention
specifications (`layers`) and metric tables with `example_id` keys. The GPU needs
enough memory for the model and fitting activations; an 80 GB GPU is suitable.

| Document | Purpose | Training policy | Seeds / repeats |
| --- | --- | --- | --- |
| `single.json` | Single-commit workflow timing and profiling | Two updates per fit | 1 / 3 |
| `study.json` | Short setup check: fitting, resident and cold workflow cases | Two updates per fit | 2 / 2 |
| `fixed-work.json` | DAS/DBM operation performance at fixed work | 20 DAS / 40 DBM updates; no early stopping | 5 / 3 |
| `normal-workflow.json` | Resident/cold workflow wall time and results | 10 DAS / 20 DBM epochs; IIA early stopping, patience 3 | 5 / 3 |

Use both longer presets when evaluating an optimization. Four training pairs with
optimizer batch size two make their maximum budgets equivalent at two updates per
epoch. If data or batch size changes, adjust fixed update budgets explicitly.
Required saves and graph setup during each fresh fit are timed.

The longer presets request CUDA graphs with execution row bound four, separate
from the two-pair optimizer minibatch. Smaller row bounds can force eager execution;
check observed replay in the report. Keep execution settings consistent between
presets and adjust the bound when enlarging data.

## Profiling and deployment

Omitting `measurement.profile` enables Torch captures; `false` disables them, as
in the longer presets. To select cases or Nsight backends, supply a
[profile object](../../../docs/measurement.md#profiling). `cold_workflow` receives
cold and warm captures after clean timing. Reports include captures and output
checks. Profile changes require a new study directory.

For local [bindings](../../../docs/measurement.md#local-bindings-and-resume), set
`data_root` to this directory's `data/`. For [remote deployment](../../../docs/measurement.md#run-on-a-remote-computer),
use local repository paths and prepared remote interpreters in bindings, and pass
this `data/` directory as `--upload-data`. Prepare the model cache and offline source
build dependencies before launch.

## Interpret the comparison

If adding workflow source pins, stamp them in the baseline environment. The harness
checks baseline assertions and records each arm's source hashes; document, dataset
and input-artifact pins remain shared. Check the report's benchmark identity and
resolved commits. For workflow comparisons, retain one resolved commit, select a
separate candidate document and check every arm's authored pins against its own
configuration.

Training and held-out data use disjoint prompt endpoints. Clean and self-swap no-op
steps save logits and base-answer accuracy. Held-out replay crosses both fits with
both evaluators outside primary timing. Whole-workflow clocks include declared
controls and replay; fitting operation clocks exclude them.

These small datasets and short fits do not establish convergence or scientific
tolerances. Inspect clean competence, no-op drift, learned-subspace dispersion,
gate confidence, held-out metrics and repeated-run variation before interpreting
speedups. Use representative data and explicit acceptance bounds for scientific
decisions.
