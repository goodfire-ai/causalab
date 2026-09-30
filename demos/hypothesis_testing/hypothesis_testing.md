# Saved hypothesis comparisons

| Overview | |
|---|---|
| **Question** | Can saved outputs be compared against several [hypotheses](models.py) on the same [pairs](counterfactuals.py)? |
| **Method** | **Exact-pair export and comparison**: export complete pairs and each hypothesis's predictions, intervene on the neural model with those pairs, and compare the saved global top-1 outputs against each hypothesis's predictions. |

## Research question

The causal model is symbolic addition modulo three, defined in [models.py](models.py).
The target hypothesis swaps the intermediate value of A. Alternatives swap B,
nothing, or the output.

Export complete pairs and each hypothesis's predictions before intervening on
the neural model. Compare saved global top-1 outputs against those predictions.
This example uses four comparison pairs and constructed neural outputs.

**Q1 — Does the saved comparison score the target and an alternative on the same pairs?**
The handoff test answers this on CPU with constructed values. A neural run is
needed for support of either intermediate variable.

## Method

Narrow families distinguish each alternative from the target. The
[models](models.py) share one scoring rule; the
[generators](counterfactuals.py) generate broad and narrow families and declare
family and split membership.

This CPU example checks the artifact handoff. Its case numbers keep prompts
separate across splits; they do not establish generalization to new arithmetic
rules. Use task-relevant groups for a research experiment.

For neural experiments, follow the [comparison guide](../../docs/hypothesis_analysis.md).
Use validation for selection and retain the selected fit for test evaluation.

## Execution

From the repository root:

```bash
uv run python scripts/run_hypothesis_generation.py demos/hypothesis_testing \
    --n 4 \
    --random-n 16 \
    --export-dir /tmp/hypothesis-demo/pairs \
    --output /tmp/hypothesis-demo/audit.json
```

Use a new output directory for another run. For research-sized tables, use
`--n 1000 --random-n 1000`: each family has 1,000 training, 500 validation,
and 500 test pairs.

The command runs on CPU only and loads no model weights. We ran only this CPU
handoff. The neural experiments require a model run, and we have not run them.

## Results

### Q1: The comparison scores both hypotheses on the same four pairs, and the target leads by 50 points on constructed outputs

The handoff test uses four pairs. Target accuracy is 75%, alternative accuracy
is 25%, and the gap is 50 percentage points. Three pairs distinguish the
hypotheses; target accuracy on those pairs is 2/3. One neural answer is outside
the declared answer vocabulary and counts as wrong for both hypotheses.
These are constructed test values, not model results.

This example does not establish neural support for either intermediate variable.
The CPU checks do not verify GPU execution, fitted subspaces, or random controls.

## Next steps

Run full interventions and DAS on the exported pairs, then compare the saved
outputs. Report absolute accuracy, the gap from every alternative, and remaining
distance from the ceiling.
