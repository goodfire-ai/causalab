# Method library

Copy a specification from the table below to try a method on a shipped task.
Read its `description` for the measurement and its limitations. The research
examples use shipped task tables, so they need no `--data-root`:

```bash
uv run causalab explain demos/methods/protocols/interchange.json --engine auto
uv run causalab run demos/methods/protocols/interchange.json \
    --engine auto \
    --out runs/interchange \
    --device cuda
```

Run fits before the documents that apply their saved bundles. For workflows,
set `--artifacts-root`; each step declares its precision, so omit `--dtype`.
The [runner](scripts/run_all.py) handles dependencies and the test fixtures
used by `minimal_cpu.json`. The [summarizer](scripts/summarize.py) saves run
metadata, document digests, and scalar metrics in `results/`.

## The model

Most examples use `Qwen/Qwen2.5-7B`: 28 layers, width 3584, 28 query heads,
and 4 key-value heads. We compared three base checkpoints using
[clean_accuracy.py](scripts/clean_accuracy.py) in bf16 on one H100.
Clean accuracy measures next-token argmax against the answer on 19 weekdays
test pairs and 256 IOI pairs. Two-way accuracy compares the two answer logits.

| checkpoint | weekdays, base / counterfactual prompt | weekdays two-way | IOI, base / counterfactual | IOI two-way |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.21 / 0.26 | 0.58 / 0.74 | 0.19 / 0.18 | 1.00 / 0.97 |
| Qwen2.5-3B | 0.16 / 0.16 | 0.42 / 0.58 | 0.59 / 0.59 | 0.99 / 1.00 |
| Qwen2.5-7B | 0.68 / 0.63 | 0.84 / 0.74 | 0.47 / 0.48 | 1.00 / 1.00 |

The 7B has the highest weekdays accuracy; all three miss the 0.85 target.
It gets about a third of clean weekdays prompts wrong. On IOI, it ranks the
correct name above the other name on every prompt, but often predicts a
non-name token. The path-patching example measures the difference between
those two names' logits.

The weekdays scan tests 56 layer-position cells. **Block 23's output at the
answer slot** scores highest: interchange accuracy 0.867 on 30 training pairs.
The fit examples and weekdays workflow use this cell. Other layer selections
scale the original 32-layer model's choices to 28 layers by depth. The PCA
workflow fits 16 components on the 30 training pairs for `das_pca_init.json`.

Fits and their apply documents use bf16; documents without `dtype` use the
engine's fp32 default. The four head and expert-neuron DBM examples use
`Qwen/Qwen3.6-35B-A3B`; `minimal_cpu.json` uses a tiny random Llama.

## The documents

The memory estimates below are for one GPU at these dataset sizes. Each
recorded GPU run took under a minute on one H100. Interchange intervention
accuracy (IIA) in `fraction` units is the share of pairs whose patched argmax
is the counterfactual answer. In `logit` units, it is the mean counterfactual
answer logit minus the original answer logit, both measured on the patched
model. Result files record the units.

[Distributed alignment search (DAS)](../../docs/methods/das.md) learns a subspace;
[Desiderata-Based Masking (DBM)](../../docs/methods/dbm.md) learns a mask using
supervision that specifies the desired behavior. DBM-DAS learns both.

| document | method | use | model | needs | result |
|---|---|---|---|---|---|
| [`interchange.json`](protocols/interchange.json) | interchange intervention | Test a residual-stream cell | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [IIA 0.87, logit diff 3.55](results/protocols/interchange.json) |
| [`weekdays_interchange.json`](protocols/weekdays_interchange.json) | interchange intervention | Test the located cell in bf16 | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [IIA 0.83, logit diff 4.01](results/protocols/weekdays_interchange.json) |
| [`weekdays_locate_scan.json`](protocols/weekdays_locate_scan.json) | layer × position scan | Select a cell before fitting | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [Best of 56 cells: block 23, answer slot, IIA 0.87](results/protocols/weekdays_locate_scan.json) |
| [`multi_position_patch.json`](protocols/multi_position_patch.json) | joint position patch | Test an effect spread across positions | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [Positions −4..−2 at block 23: logit diff −2.63](results/protocols/multi_position_patch.json) |
| [`attention_band_patch.json`](protocols/attention_band_patch.json) | attention band patch | Compare spans of attention layers | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [L9–12, L13–16, L9–16: logit diff −3.1 each](results/protocols/attention_band_patch.json) |
| [`path_patching.json`](protocols/path_patching.json) | sender-to-receiver patch | Test a head's effect through one path | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [IO − S logit diff 6.20 on 256 pairs](results/protocols/path_patching.json) |
| [`hydra_effect.json`](protocols/hydra_effect.json) | resample ablation + direct effects | Measure downstream compensation | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [L11 ablation: answer logit 16.10 → 16.16; direct effects L12 0.46 → −0.17, L17 −0.75 → −1.27](results/protocols/hydra_effect.json) |
| [`harvest.json`](protocols/harvest.json) | activation harvest | Save residuals at two layers × two positions | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [Four tensors, each 30 × 1 × 3584](results/protocols/harvest.json) |
| [`probe_generate.json`](protocols/probe_generate.json) | steering + greedy decoding | Inspect generated text after steering | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [+2.0 steering: tail ` this` on 28/30 rows](results/protocols/probe_generate.json) |
| [`probe_variable.json`](protocols/probe_variable.json) | decoding + answer harvest | Save activations at generated answers | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [30/30 rows answered; 20 positions harvested](results/protocols/probe_variable.json) |
| [`random_subspace_control.json`](protocols/random_subspace_control.json) | random subspace | Compare DAS with a matched-rank control | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [Rank 8, three seeds: IIA −2.25 to −2.24](results/protocols/random_subspace_control.json) |
| [`das.json`](protocols/das.json) | rank-8 DAS | Fit an intervention subspace | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [Train IIA 4.75, held-out −0.05](results/protocols/das.json) |
| [`das_boundless.json`](protocols/das_boundless.json) | DBM-DAS, boundary gate (Boundless DAS) | Learn the subspace rank | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [Rank 32/64; train IIA 6.24, held-out 0.63](results/protocols/das_boundless.json) |
| [`pca_harvest.json`](protocols/pca_harvest.json) | PCA harvest | Save input for `pca_basis.json` | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [One tensor, 30 × 1 × 3584](results/protocols/pca_harvest.json) |
| [`das_pca_init.json`](protocols/das_pca_init.json) | PCA-initialized DAS | Compare ranks and seeds from a PCA basis | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [15 fits; best train match 0.97, held-out 0.58 at k = 16](results/protocols/das_pca_init.json) |
| [`dbm.json`](protocols/dbm.json) | coordinate DBM | Learn a coordinate mask | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [Train IIA 5.24, held-out 2.48; decisive fraction 0.0; 2528/3584 coordinates kept](results/protocols/dbm.json) |
| [`dbm_apply.json`](protocols/dbm_apply.json) | saved coordinate mask | Evaluate the fitted mask on held-out pairs | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [Held-out IIA 2.48 on 19 pairs](results/protocols/dbm_apply.json) |
| [`mean_harvest.json`](protocols/mean_harvest.json) | mean harvest | Save a corpus mean for replacement | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [One 3584-vector](results/protocols/mean_harvest.json) |
| [`mean_ablation.json`](protocols/mean_ablation.json) | mean replacement | Replace activations with the saved mean | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [Block 23 ablation: base − counterfactual logit diff 0.24](results/protocols/mean_ablation.json) |
| [`weekdays_das_sweep.json`](protocols/weekdays_das_sweep.json) | DAS rank × seed sweep | Fit subspaces in the weekdays workflow | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [9 fits; best held-out IIA 0.72 at k = 32, seed 0](results/protocols/weekdays_das_sweep.json) |
| [`weekdays_das_apply.json`](protocols/weekdays_das_apply.json) | saved DAS rotation | Apply the workflow's selected fit | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [Held-out IIA 0.74 at the curve's knee](results/protocols/weekdays_das_apply.json) |
| [`dbm_head.json`](protocols/dbm_head.json) | head-grouped DBM | Select attention heads in a MoE model | Qwen3.6-35B-A3B, bf16 | 1 GPU, 80 GB | [8/16 heads kept, decisive; train IIA −2.86, held-out −3.27](results/protocols/dbm_head.json) |
| [`dbm_head_apply.json`](protocols/dbm_head_apply.json) | saved head mask | Evaluate the head mask on held-out pairs | Qwen3.6-35B-A3B, bf16 | 1 GPU, 80 GB | [Held-out IIA −3.27](results/protocols/dbm_head_apply.json) |
| [`dbm_expert_neuron.json`](protocols/dbm_expert_neuron.json) | expert-neuron DBM | Select routed and shared MLP units | Qwen3.6-35B-A3B, bf16 | 1 GPU, 80 GB | [2462 routed + 261 shared units kept; train IIA −2.69, held-out −3.16; routing mismatch 17.5%](results/protocols/dbm_expert_neuron.json) |
| [`dbm_expert_neuron_apply.json`](protocols/dbm_expert_neuron_apply.json) | saved neuron masks | Evaluate neuron masks on held-out pairs | Qwen3.6-35B-A3B, bf16 | 1 GPU, 80 GB | [Held-out IIA −3.16; routing mismatch 19%](results/protocols/dbm_expert_neuron_apply.json) |
| [`minimal_cpu.json`](protocols/minimal_cpu.json) | tiny-model interchange | Check an installation on CPU | tiny-random Llama, fp32 | CPU | [IIA 0.0 on 4 fixture rows](results/protocols/minimal_cpu.json) |
| [`mean_ablation.json`](workflows/mean_ablation.json) | harvest → mean-ablate | Run mean replacement end to end | Qwen2.5-7B, fp32 | 1 GPU, 40 GB | [Logit diff 0.24 after ablation](results/workflows/mean_ablation.json) |
| [`weekdays.json`](workflows/weekdays.json) | locate → DAS sweep → apply | Select a fit by the curve's knee and plot results | Qwen2.5-7B, mixed | 1 GPU, 40 GB | [Block 23; k = 32; held-out IIA 0.74](results/workflows/weekdays.json) |
| [`pca_basis.json`](workflows/pca_basis.json) | harvest → `fit_pca` | Build the basis for `das_pca_init.json` | Qwen2.5-7B, bf16 | 1 GPU, 24 GB | [16 components explain 95.5% of variance](results/workflows/pca_basis.json) |

## What the numbers say, and do not

These examples fit on only 30 weekdays pairs. DAS shows substantial
overfitting: training IIA reaches 4–7 logits while held-out IIA stays below
one. The coordinate DBM gate has a decisive fraction of zero, and the A3B
masks fail to move the answer. These runs demonstrate execution; research
claims need larger datasets, held-out evaluation, and matched controls.
Use `scripts/build_split_dataset.py` to build more pairs.

## Rerunning

```bash
uv run python demos/methods/scripts/run_all.py --list             # dependency order
uv run python demos/methods/scripts/run_all.py --device cuda      # full library
uv run python demos/methods/scripts/run_all.py \
    --device cuda \
    --group dbm \
    --skip-a3b
uv run python demos/methods/scripts/summarize.py                  # runs/ -> results/
```

The runner writes its run trees to a `runs/` directory beside `results/`
(ignored by Git). Commit the summaries in `results/`. After changing a
document, rerun it and regenerate its summary; `tests/demos/test_methods.py`
checks the recorded digest.
