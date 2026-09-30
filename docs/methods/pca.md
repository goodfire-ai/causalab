# PCA basis

A `pca` featurizer uses a fixed basis fitted to saved activations by
`causalab.analysis.fit_pca`. Load its bundle through `file_path` and choose `k`.
A fit over `n` rows of `d` dimensions has at most `min(n - 1, d)` components
with variance, because centering removes one degree of freedom, so the fit
refuses a larger `k`.
Interchange interventions act in the first `k` principal components. Comparing
this basis with DAS tests whether directions that explain variance also support
the causal hypothesis.

The map is `featurize(x) = (Pᵀx, 0)`, with parameter slot `<name>.weight`.
Fit PCA on training data. The protocol loads the resulting basis; `train.params`
cannot train a `pca` featurizer.

## 1. Where it sits

The site must have a feature width and match the bundle's identity.

## 2. Fields

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table pca -->

| field | without `file_path` | with `file_path` |
|---|---|---|
| `k` | legal | legal |

<!-- generated: end call causalab.protocol.schema.render_field_legality_table pca -->

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec k file_path entry dtype -->

- **`k`**: Width of a `subspace` or `pca` feature space. Interchanges act in the first `k` basis columns and preserve the complementary `d − k` directions. Sweepable.
- **`file_path`**: Path to a fitted artifact. The loaded featurizer uses its saved parameters and accepts no training, initialization, or training-rule fields. A loaded budget gate requires `top_k`. `ArtifactIdentity` is checked at load and build (rule 15). Sweep bundle paths to compare fits.
- **`entry`**: Coordinate selector for a bundle loaded through `file_path`. Required when the bundle contains several entries; otherwise the sole entry is used.
- **`dtype`**: Precision used to hold and save featurizer parameters (`PRECISION_DTYPES`). Defaults to the model precision. A loaded document must match the bundle's dtype.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec k file_path entry dtype -->

## 3. What is refused

<!-- generated: begin call causalab.protocol.schema.render_field_refusals pca -->

- A `pca` follows the shared validation rules (§5).

<!-- generated: end call causalab.protocol.schema.render_field_refusals pca -->

## 4. Shipped templates

The shipped templates use PCA to initialize DAS through `subspace.init`:
[`das_pca_init`](../../demos/methods/protocols/das_pca_init.json).
To use PCA as the intervention basis itself, load the basis with `kind: pca`.

## 5. Demos

- [Variance and causal effects](../../demos/onboarding_tutorial/08_variance_vs_cause.md)
