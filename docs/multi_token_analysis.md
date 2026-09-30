# Shared execution for multi-token analysis

Each target is a next-token prediction. One forward pass can score several
targets: the last prompt position predicts output 0, and the position containing
output j−1 predicts output j. Causal attention blocks access to later supplied
tokens. The targets share a forward pass and counterfactual captures while
keeping separate metrics.

## Prepare exact sequences

This example uses a loaded PyTorch `ModelBundle`, `bundle`. Keep the model
revision, dtype, quantization, tokenizer, and data splits fixed. For a quantized
bundle, include `bundle.quantization` in the model dictionary below.

```python
from causalab.analysis.sequences import (
    prepare_sequence, pair_sequences, sequence_cohorts, write_sequence_workflow,
)

original = prepare_sequence(
    bundle.tokenizer, "Complete the sequence: 2, 4,", " 6, 8",
    example_id="even-base", split="eval", prefix_condition="correct",
)
counterfactual = prepare_sequence(
    bundle.tokenizer, "Complete the sequence: 3, 6,", " 9, 12",
    example_id="triple-donor", split="eval", prefix_condition="correct",
)
# Automatic alignment requires equal output lengths and matching semantic roles.
# For other layouts, supply target_alignment or select a narrower cohort.
# Check tokenization: a number can occupy several tokens.
pair = pair_sequences(original, counterfactual)

for cohort, rows in sequence_cohorts([pair]).items():
    length = rows[0]["output_length"]
    write_sequence_workflow(
        f"study/{cohort}",
        {"key": bundle.key, "revision": bundle.revision, "dtype": bundle.dtype},
        rows,
        layers=[0, 1, 2],       # expand after critical-location selection
        bands=[[0, 1], [1, 2]], # use the scientifically chosen layer bands
        positions=[{"index": -length - 1}],  # last original prompt token
    )
```

The helper writes `pairs.json`, `targets.json`, `workflow.json`, and four
intervention specifications: `harvest`, `residual`, `attention`, and `mlp`.
One harvest covers every real input and output position at the requested layers.
Each patching method scores all outputs and an unchanged original input against
the counterfactual targets. Each unique layer band remains a separate intervention.

Run from the CausaLab checkout. Replace `COHORT` with the directory above:

```bash
uv run causalab validate study/COHORT/workflow.json \
    --engine auto \
    --data-root study/COHORT
uv run causalab run study/COHORT/workflow.json \
    --engine auto \
    --data-root study/COHORT \
    --out study/runs \
    --device cuda \
    --batch-rows 16
```

The workflow runner executes these steps. An external scheduler can run the
independent method specifications concurrently. Keep a method's target readouts
together to share counterfactual captures. Separate requests and `--points`
shards have separate memory caches. Limit batch rows and capture volume for
long sequences.

Each prepared row contains `input`, exact target IDs (`output_i` and
`targets[*].token_id`), and prediction positions. `counterfactual_inputs[0]`
contains the counterfactual text. Positions address the tokenizer's default
encoding, including special tokens. Use plain text rows and omit `segments`.
The chat frame wraps the column as a new user turn and appends a generation
prompt; it cannot represent a supplied completion after that prompt.

A text answer that changes tokenization at the prompt boundary is rejected.
Revise the prompt or supply exact continuation IDs. For a recorded baseline
generation, pass its emitted IDs with `prefix_condition="baseline_generated"`.
The IDs become targets, while their decoding in prompt context becomes the
input text. Byte fallback or boundary merges can break this round trip.
Check the executed token count against `output_length` before using indices
relative to the end. EOS is allowed only as the last target. Record empty or
ambiguous semantic targets as unavailable in the experiment's own records.

`sequence_cohorts` groups full rows by split, prefix condition, and both output
lengths. It rejects parent examples that cross splits, including appearances
as a counterfactual. Retain stable parent IDs and assign splits before expanding
targets. Validate semantic roles within each length cohort.

Generated specifications apply the same location rule to both inputs. With
unequal output lengths, an index relative to the end can select different roles.
Use role-specific variable anchors or set the counterfactual read's position
separately. Record which reference supplied the counterfactual prefix; this
choice forms part of the intervention.

## Add targets to existing experiments

`add_readouts(document, output_length, model="patched", name="targets", ...)`
copies a version 3 intervention specification and adds one `lm_head` read per
prediction position. Each has exact-ID accuracy, NLL, target logit, and top-k
metrics. `target_prefix` names integer columns (`output_0`, `output_1`, … by
default). `alternative_prefix` adds target-minus-alternative logit differences.
Metrics save separate JSON files. Reads of the same model/input share a forward
pass. The helper also applies to fitted DAS, DBM, and DBM-DAS specifications.
Each supervised objective still requires its own fit.

Join results with `targets.json` on `example_id` and the target ordinal in the
metric name. Retain sweep coordinates in the join. NLL is negative log
probability, so `exp(-NLL)` gives target probability. Preserve missing scores
and aggregate each unit separately. Report token accuracy, semantic-slot
success, and full-generation success as distinct quantities. Shared targets
form dependent observations.

## PCA and logit lens use one harvest

`causalab.analysis.sequence_activations` takes `acts`, a harvest of all positions,
and its prepared `rows`. It gathers prediction positions into
`(examples, targets, features)`, including from flattened ragged harvests.
Keep the full harvest for input-token analysis and neighbors in the original
feature space. Exclude ragged padding from PCA observations.

`causalab.analysis.pca_by_position` takes this tensor, `train_rows` indices,
and `k`. It saves `mean`, `weight`, `coordinates`, and a `spectrum` table with
a position column. Each position's basis and mean use only training rows.
To project train and evaluation populations together, concatenate them in a
declared order and list only training indices. Reuse the coordinates across
labels and report tabs.

For one population, `fit_pca` saves `weight` and `spectrum`, with optional
`mean` and `coordinates`. It pools all leading axes. Select the intended
population first, or use `pca_by_position` to keep positions separate.
Over `n` pooled rows of `d` features, `k` can be at most `min(n - 1, d)`.
Centering leaves no more components with variance than that, so a larger `k`
is refused.
`project_pca` applies frozen `mean` and `weight` to `acts` from any split.

`causalab.analysis.select_pca_basis` selects a zero-based `position` from grouped
`weight` and saves its `(features, components)` basis. It checks finite,
orthonormal columns and records `k`. Use a workflow tensor reference to retain
model, site, and training-data metadata. A later `subspace.init` can select
the first k columns:

```json
"basis": {
  "type": "script",
  "script": {"module": "causalab.analysis.select_pca_basis"},
  "inputs": {
    "weight": {"step": "pca", "file": "weight.safetensors"},
    "position": 0
  },
  "outputs": {"weight": "weight.safetensors"}
}
```

```python
from causalab.analysis.logit_lens import logit_lens

cells = logit_lens(
    bundle, "study/runs/sequence_analysis/harvest/residual_0.safetensors",
    k=10, batch_positions=128,
)
```

The logit lens applies the model's final normalization and output head in
bounded chunks. It skips transformer blocks. A saved harvest must identify
the matching model, revision, dtype, quantization, and `block_output` site.
Callers supply provenance for tensor inputs.

Optional `target_ids` supplies one ID per flattened vector. IDs are bounded
by the output head's width; a padded vocabulary can exceed the tokenizer's
length. Results contain leading-axis indices, highest-k and lowest-k
IDs/text/logits/probabilities, the log normalizer, and optional target logit
and log probability. Map ragged indices to rows with the harvest's `.widths`
sidecar. New candidates can need another head projection. This API requires
executable PyTorch modules.

## Score one intervention-generated rollout

`add_rollout_readouts(document, output_length, ...)` adds target scores and
decoded text to one greedy rollout. Supply prompt-only inputs and exact
expected-ID target columns.

```python
from causalab.analysis.sequences import add_rollout_readouts

rollout = add_rollout_readouts(
    prompt_only_intervention_document, output_length=3,
    model="patched", name="rollout", target_prefix="cf_output_",
)
```

The first read uses prompt `-1`; target j>0 reads generated position j−1.
All reads for the same model/input share a decode. The engine executes one
prefill plus one step per requested generated activation. Missing positions
retain unmatched/unavailable records. The helper aligns ordinals. Validate
semantic alignment separately when roles can move between positions.

Writes apply during prefill, and at every decode step when the intervened
model declares `writes_during_generation` (intervention protocol §2.9). The
engine lacks writes addressed at generated positions,
gradients through greedy decoding, and a cache of completed rollouts across
points. Group the rollout targets together. Each intervention needs its own
trajectory and mutable KV/recurrent state, even when emitted tokens match.

Report correct fixed prefixes, frozen baseline-generated prefixes, and
intervention rollouts separately. Later rollout effects include changes to
earlier tokens. Estimating mediation requires assumptions and interventions
beyond comparing rollout and fixed-prefix effects.

### Final PCA step: sparse concept probes

Append `causalab.analysis.sparse_pca_probe` after `pca_by_position`, once per
harvested layer. It uses frozen coordinates. PCA and probe standardization
must use the same training rows. The row table follows coordinate order and
contains unique string `id`, `split` (`train`, `validation`, or `evaluation`),
and one column per concept. Categorical labels are strings; numeric labels
are finite numbers. Each categorical class must occur in every split, and
numeric labels must vary within each split. Use this table to hold out entities
or input combinations.

A relative `path` pins the label table. After editing it, update the pin and
run without `--resume` to recompute scores.

```json
"sparse_probes": {
  "type": "script",
  "script": {"module": "causalab.analysis.sparse_pca_probe"},
  "inputs": {
    "coordinates": {"step": "pca", "file": "coordinates.safetensors"},
    "rows": {"path": "data/probe_rows.json"},
    "concepts": {"output_label": "categorical", "operand_value": "numeric"},
    "layer": 12,
    "strengths": [0.1, 1.0, 10.0, 100.0]
  },
  "outputs": {
    "scores": "scores.json",
    "coefficients": "coefficients.json",
    "metadata": "metadata.json"
  }
}
```

The step fits L1 logistic regression for categories or Lasso for numeric labels
on PCs standardized from training rows. It selects the highest validation
balanced accuracy or R². Exact ties prefer fewer selected PCs, then a stronger
penalty. That training fit is retained for evaluation. `strength` means `1/C`
for logistic regression and `alpha` for Lasso; compare scales within each
method. Failure to converge raises an error.

`scores` has one row per concept and position: layer, split counts,
validation/evaluation scores, training-majority or training-mean baseline,
selected PCs, penalty, class labels, and intercepts. `coefficients` saves all
PC coefficients and selection flags, per class when applicable. Binary logistic
coefficients describe the last entry in `classes`. `metadata` records split
IDs, training means/scales, candidate penalties, and the fixed solver seed.
PC indices start at zero.

Plot evaluation scores by token and layer, with selected PCs and standardized
coefficients available for each site. Use separate scales for balanced accuracy
and R²; R² can be negative. Probe scores measure held-out decodability. Support
causal claims with interventions.

### Fourier probes on saved activations

`causalab.analysis.fit_fourier_probe` fits affine ridge readouts for one numeric
column at every supplied position. Use saved residuals, frozen PCA coordinates,
or the `coordinates` output of `project_subspace` for a frozen DAS basis. The
array has shape `(examples, positions, features)`; `(examples, features)` means
one position. Rows carry unique string `id`, a `split` of `train`, `validation`
or `evaluation`, and the numeric target. Their order must match the activations.

For period `T`, positive integer harmonic `k`, and `origin`, the targets are
`[cos(2*pi*k*(value-origin)/T), sin(2*pi*k*(value-origin)/T)]`. Defaults scan
every integer period from 2 through 150 at harmonic 1. Supply positive real
periods for a finer grid or another unit. Equal `k/T` values share one fit and
retain their aliases. Sampling can create further aliases, such as frequencies
above the Nyquist limit on integer labels; interpret them using the label grid.
A sine/cosine pair spans every phase origin. Separate fits for phase offsets
are redundant under this isotropic penalty and joint loss.

Metrics retain the number of observed phases and the shortest arc containing
them. Metadata retains numeric ranges per split. Inspect this coverage when
distinguishing a periodic readout from a fit to a short arc.

To inspect task harmonics separately from a broad scan, make a second call with
the task's natural `periods`, such as `[10, 7]`, and
`harmonics: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. The effective period is T/k;
keep T and k as authored numbers instead of rounding T/k. Both calls can use
the same rows, activations, penalties and phase origin. Record their purpose
in `source.scan`, for example `{"kind": "broad"}` or
`{"kind": "natural", "periods": [10, 7], "rationale": "Decimal quantities measured in days with a seven-day cycle."}`.
Every requested alias remains in metadata. With integer targets, T=10, k=10
is constant and receives `constant_training_targets`; harmonics beyond the
Nyquist limit can repeat the phase information of lower harmonics.

```json
"fourier": {
  "type": "script",
  "script": {"module": "causalab.analysis.fit_fourier_probe"},
  "inputs": {
    "acts": {"path": "acts.safetensors"},
    "rows": {"path": "data/probe_rows.json"},
    "target": "operand_value",
    "layer": 12,
    "representation": "residual",
    "origin": 0,
    "harmonics": [1],
    "alphas": [0.001, 0.01, 0.1, 1, 10, 100]
  },
  "outputs": {
    "weight": "weight.safetensors",
    "bias": "bias.safetensors",
    "plane": "plane.safetensors",
    "calibration": "calibration.safetensors",
    "predictions": "predictions.safetensors",
    "scores": "scores.json",
    "metadata": "metadata.json"
  }
}
```

Each position uses one training SVD for all frequencies, penalties and a seeded
shuffled-label control. The objective is summed squared error plus
`alpha * ||weight||²`, with a free intercept. Features are centered on training
rows. Isotropic ridge preserves equivalence between orthonormal DAS coordinates
`hQ` and their reconstruction `hQQᵀ`; per-coordinate rescaling would change
that penalty. PCA, DAS and any external preprocessing must use training data.
Keep a held-out probe evaluation set outside DAS fitting and rank selection.

Each frequency chooses its penalty by minimum validation pair MSE; exact ties
prefer the larger penalty. Report evaluation once using that frozen training
fit. The shuffled control permutes training target rows, then selects its own
penalty on the original validation targets. `shuffle_seed` defaults to zero.
This one shuffle is a diagnostic baseline. A broad scan needs independent
confirmation before a selected peak supports a research claim.

Outputs use float64. `weight` has shape `(positions, frequencies, features, 2)`;
`bias` has shape `(positions, frequencies, 2)`. Predictions have shape
`(examples, positions, frequencies, 2)`, with cosine first. The padded
orthonormal `plane` has the same shape as `weight`, and `calibration` has shape
`(positions, frequencies, 2, 2)`. Their product recovers the readout weight to
numerical tolerance. Read the recorded `rank` before using a plane: unused
columns are zero. These grouped analysis tensors need a selected position,
frequency and active rank before use as a geometric basis. The plane's rank
comes from the fitted weight. `target_rank` separately describes variation in
the training targets; a short arc can have a lower numerical target rank.

`scores` contains validation/evaluation pair MSE and variance-weighted joint
R², per-coordinate R², wrapped angular MAE in radians, phase counts and mean
radius, plus training-mean and shuffled baselines. Constant coordinates have
null R². Constant training targets have status `unavailable`; preserve the
reason when plotting. Period 2 on integer labels with origin zero has one
informative coordinate and a rank-one plane. Phase is defined only where
predicted radius exceeds `1e-8`. A phase locates the value modulo the effective
period `T/k`; it does not determine the original number across cycles.

`metadata` records the grid, label-row hash, split IDs, training means and
selection settings. Supply one unique string or integer in `position_labels`
per position for semantic token names, and
`source` for additional artifact references, units, label definitions and the
selected DAS rank/fit. The workflow records input digests and stamps tensor
provenance. Callers must keep the feature basis and position order consistent
when applying a fit, including in a workflow.

For a frozen readout, use `causalab.analysis.apply_fourier_probe` with `acts`,
`weight`, and `bias`. Its outputs are `predictions`, `radius`, `phase` in
`[0, 2*pi)`, and boolean `phase_defined`. An undefined phase has a zero storage
value and a false mask. Reuse the exact feature basis and position order from
the fit. Projection into the learned plane uses the saved calibration and bias
to recover cosine/sine coordinates.

Use the public artifact reader to verify a saved fit before further analysis:

```python
from pathlib import Path
from causalab.analysis.fourier_artifacts import load_fit

saved = load_fit(Path("fourier"), Path("acts.safetensors"), Path("data/probe_rows.json"))
```

The directory contains the seven outputs above. The reader checks the schema,
example and split IDs, tensor shapes, position labels, score grid, saved-plane
factorization and frozen predictions against the supplied original population.
It compares tensor identity stamps when present. Direct Python outputs can be
unstamped; their model provenance remains the caller's responsibility.

The returned mapping contains `metadata`, `scores`, `rows`, `identity`, and
float64 NumPy arrays `acts`, `weight`, `bias`, `plane`, `calibration`,
`predictions` and `truth`. `truth` has shape `(examples, frequencies, 2)`.
All rows and frequencies are retained. Consumers choose which measurements to
display.

These probes follow the affine targets in
[Arithmetic in the Wild, §4 and Appendix G](https://arxiv.org/abs/2605.01148).
The ridge solver provides a direct fit for those targets. Optional PCA inputs
also support the setting studied by
[Engels et al.](https://arxiv.org/abs/2405.14860). The
[numeric and periodic encoding study](https://arxiv.org/abs/2502.00873) motivates
retaining scalar numeric probes alongside Fourier readouts. Angle decoding also
appears in [Probing for Arithmetic Errors](https://aclanthology.org/2025.emnlp-main.411/).

### Reconstruct a saved subspace component

`causalab.analysis.project_subspace` is a workflow script for any orthonormal
basis, including a fitted DAS subspace. Inputs are `acts` with shape `(..., d)`
and `weight` with shape `(d, k)`. Outputs are `coordinates` and `reconstructed`:

```text
coordinates = acts @ weight
reconstructed = coordinates @ weight.T
```

The operation uses CPU float64, preserves leading dimensions and does not
center, restore a mean or add the complementary residual. It refuses nonfinite
inputs, incompatible shapes and a basis outside the existing featurizer's
orthonormality tolerance. Select swept inputs with `slot` and `entry` on the
[workflow reference](workflow_protocol.md#3-cross-step-wiring--the-reference-grammar),
so the runner inherits the selected tensors' model and site metadata:

```json
"projection": {
  "type": "script",
  "script": {"module": "causalab.analysis.project_subspace"},
  "inputs": {
    "acts": {"path": "acts.safetensors", "slot": "acts", "entry": {"layer": 12}},
    "weight": {"path": "weight.safetensors", "slot": "weight", "entry": {"layer": 12, "k": 2}}
  },
  "outputs": {
    "coordinates": "coordinates.safetensors",
    "reconstructed": "reconstructed.safetensors"
  }
}
```

Single-entry files need no selector. For direct Python calls, select tensors
with `causalab.io.step_io.read_tensor` before passing them to `main`; the caller
owns provenance outside the workflow. Verify model, site and population
correspondence before combining activations and a basis.

For DBM-DAS, which jointly trains a rotation and coordinate gate, pass an optional binary
`mask` with one entry per column of the full saved weight. Use the binary mask
from the evaluated gate, such as `theta > 0` for a sigmoid gate with its default threshold. The script
selects those columns before projecting and reconstructing. The
`project(acts, weight, mask=None)` Python function has the same contract.
The mask must be binary and match the basis width. An all-zero mask yields
coordinates with width zero and an all-zero reconstruction; downstream
analyses must record the absence of a varying component. Keep the original
joint fit, mask and selected sweep entry in the consumer's artifact record.

To decode only this subspace component, pass the reconstructed tensor to
`causalab.analysis.logit_lens.logit_lens` with the matching loaded model
bundle and target IDs. The existing logit lens applies the model's final
normalization and vocabulary head. This readout measures the reconstructed subspace component. An interchange
intervention requires a separate experiment. Save the basis and activation
identities with the reconstruction formula.
