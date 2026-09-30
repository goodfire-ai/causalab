# Sparse autoencoder features

An `sae` featurizer uses an encoder's latents as intervention features. It
preserves reconstruction error so the complete inverse recovers the activation:
`featurize(x) = (enc(x), x − dec(enc(x)))`.

Load a fitted encoder and decoder through `file_path`. The bundle declares
`<name>.enc`, `<name>.dec`, `<name>.b_enc`, and `<name>.b_dec`. The SAE kind has
trainable slots; loading a bundle fixes its parameters for the protocol.

## 1. Where it sits

The site must have a feature width and match the bundle's identity.

## 2. Fields

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table sae -->

A `sae` uses the common fields `file_path`, `entry`, `dtype`, and `description` (§2.5).

<!-- generated: end call causalab.protocol.schema.render_field_legality_table sae -->

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

- **`file_path`**: Path to a fitted artifact. The loaded featurizer uses its saved parameters and accepts no training, initialization, or training-rule fields. A loaded budget gate requires `top_k`. `ArtifactIdentity` is checked at load and build (rule 15). Sweep bundle paths to compare fits.
- **`entry`**: Coordinate selector for a bundle loaded through `file_path`. Required when the bundle contains several entries; otherwise the sole entry is used.
- **`dtype`**: Precision used to hold and save featurizer parameters (`PRECISION_DTYPES`). Defaults to the model precision. A loaded document must match the bundle's dtype.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

## 3. What is refused

<!-- generated: begin call causalab.protocol.schema.render_field_refusals sae -->

- A `sae` follows the shared validation rules (§5).

<!-- generated: end call causalab.protocol.schema.render_field_refusals sae -->

## 4. Shipped templates and demos

The repository supplies the featurizer interface without an SAE
template or demo. See the specification for
[feature-space maps](../intervention_protocol.md#25-featurizers).
