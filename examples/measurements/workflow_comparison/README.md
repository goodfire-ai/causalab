# Compare workflow configurations

Run two workflow documents at the same code commit, with shared input evidence,
seeds, repeats and observations. The Qwen example compares DAS learning rates while
keeping the update budget and held-out evaluation fixed.

## Quickstart

Run the [copy-paste workflow quickstart](../../../docs/measurement.md#quickstart-compare-workflow-configurations)
from the repository root. It copies the Qwen files into a temporary study directory,
creates `candidate.json` without a measurement block, and changes the candidate's
DAS step to:

```json
{
  "type": "intervention_protocol",
  "document": "das.json",
  "set": {"train.optimizer.lr": 0.0005}
}
```

The baseline learning rate is `0.001`. The owning study selects the candidate with
`measurement.comparison: "workflow"` and `measurement.arms.after.workflow:
"candidate.json"`. Both arm revisions resolve to the same commit. Model-cache,
GPU-memory and dependency requirements are the same as the [Qwen examples](../qwen/README.md).

## Change the comparison

Use a fresh output directory after editing either document. Keep the measurement
plan in the baseline document. Candidate intervention paths resolve relative to
its own directory; shared cases, observations and evaluation names must exist in
both workflows. Optional `eager.workflow` selects a third configuration, otherwise
that arm uses the baseline document.

To compare training budgets, replace the candidate override with
`{"train.steps.updates": 4}`; the baseline has two updates. This changes the
work being timed, so the duration ratio is a cost comparison across budgets.
Additional configuration changes can alter dtype, quantization, reads or sites,
provided the checkpoint, datasets, input artifacts and logical token inputs remain
shared and saved observations align. Authored pins must match each selected
configuration.

Inspect the report's exact authored differences, resolved commit, per-arm benchmark
identities, fitting diagnostics and held-out metrics. Output differences do not by
themselves establish semantic equivalence, and timing ratios do not establish equal
work. The short Qwen fits check the setup; use representative data and repeated
fits to assess a configuration's quality.
