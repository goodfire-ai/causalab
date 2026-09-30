# Methods

These guides explain how to configure feature-space maps for causal experiments.
Copy a linked template, select its site and data, then validate the document.
Evaluate learned maps on a held-out split and compare them with matched controls.

| Method | Kind | What it does |
|---|---|---|
| [Desiderata-Based Masking (DBM)](dbm.md) | `gate` | Learns a mask with supervision that specifies desiderata. |
| [Distributed alignment search (DAS)](das.md) | `subspace` | Learns an orthonormal basis for interchange interventions. |
| [PCA](pca.md) | `pca` | Uses a basis fitted to the variance of saved activations. |
| [Sparse autoencoder](sae.md) | `sae` | Uses encoder latents as intervention features. |

DBM-DAS combines a learned mask with a basis learned by DAS. `identity` uses the
site's coordinates; `standardize` uses a saved mean and scale.

## Feature-space maps

<!-- generated: begin call causalab.protocol.schema.render_featurizer_kind_table -->

| kind | featurize | param slots | authored fields |
|---|---|---|---|
| `identity` (default) | `(x, 0)` | none | none |
| `subspace` | `(Qᵀx, 0)` | `weight` | `k`, `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel`, `init` (on a fit), `seed` (on a fit) |
| `pca` | `(Pᵀx, 0)` | `weight` | `k` |
| `sae` | `(enc(x), x − dec(enc(x)))` | `enc`, `dec`, `b_enc`, `b_dec` | none |
| `standardize` | `((x−μ)/σ, 0)` | `mu`, `sigma` | none |
| `gate` | `(m⊙x, (1−m)⊙x)`, `m` the soft mask in training and the hard mask at eval, by `parametrization` (the table below) | `theta` | `parametrization` ∈ `sigmoid` \| `clamp` \| `hard_concrete` \| `budget` \| `boundary`, `group` ∈ `head` \| `expert_neuron` \| `site` (under any map but `boundary`), `axis` ∈ `position` (under any map but `boundary`), `init` (on a fit), `temperature` (under `hard_concrete` \| `boundary`), `stretch` (under `hard_concrete`), `dead` ∈ `freeze_after` \| `leak` (under any map but `boundary`, on a fit), `top_k` (under any map but `boundary`, with `file_path`), `k_schedule` (under `budget`, on a fit), `stop_grad_shift` (under `budget`, on a fit), `pool` (under `budget` to fit; any map but `boundary` with `file_path`) |

<!-- generated: end call causalab.protocol.schema.render_featurizer_kind_table -->

The following kinds have gradient-trainable parameter slots:

<!-- generated: begin value causalab.protocol.schema.TRAINABLE_KINDS -->

`gate`, `sae`, `subspace`

<!-- generated: end value causalab.protocol.schema.TRAINABLE_KINDS -->

## Sites

<!-- generated: begin call causalab.protocol.registry.render_widthless_components -->

On `Qwen/Qwen3.6-35B-A3B` a featurizer attaches to 48 of 56 components. These components lack a feature width: `input_ids`, `delta_state`, `attention_scores`, `attention_probs`, `mlp_activation`, `mlp_neuron_output`, `expert_idx`, `expert_permutation`.

<!-- generated: end call causalab.protocol.registry.render_widthless_components -->

## Common fields

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec kind file_path entry dtype description -->

- **`kind`**: Feature-space map from `FEATURIZER_KINDS`. Defaults to `identity`. Sweepable.
- **`file_path`**: Path to a fitted artifact. The loaded featurizer uses its saved parameters and accepts no training, initialization, or training-rule fields. A loaded budget gate requires `top_k`. `ArtifactIdentity` is checked at load and build (rule 15). Sweep bundle paths to compare fits.
- **`entry`**: Coordinate selector for a bundle loaded through `file_path`. Required when the bundle contains several entries; otherwise the sole entry is used.
- **`dtype`**: Precision used to hold and save featurizer parameters (`PRECISION_DTYPES`). Defaults to the model precision. A loaded document must match the bundle's dtype.
- **`description`**: Description for readers. Excluded from the canonical form and digest.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec kind file_path entry dtype description -->

## Maintaining these pages

Edit generated descriptions in their source docstrings or attribute docs, then run:

```bash
uv run python scripts/generate_support_tables.py
uv run python scripts/generate_support_tables.py --check
```

The [documentation maintenance guide](../DOCUMENTATION.md) explains generated blocks.
See the intervention specification for [featurizers](../intervention_protocol.md#25-featurizers),
[training](../intervention_protocol.md#211-train), and
[validation](../intervention_protocol.md#5-validation--load-error-checklist).
