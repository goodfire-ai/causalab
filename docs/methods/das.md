# Distributed alignment search (DAS)

DAS learns a basis in which interchange interventions test a causal hypothesis.
Causalab uses a `subspace` featurizer with orthonormal basis `Q`. Its first `k`
columns define the intervention subspace; the complementary directions retain
their original values. The feature map is `featurize(x) = (Qᵀx, 0)`.

Start from a [template](#8-shipped-templates), fit on training pairs, and evaluate
the saved basis on held-out pairs. Compare it with random subspaces at the same
site and rank. The parameter slot `<name>.weight` is derived from the model and
site; `train.params` may name the subspace or this slot.

## 1. Where it sits

A subspace attaches to any site with a feature width. See the [method index](README.md).

## 2. Fields

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table subspace -->

| field | without `file_path` | with `file_path` |
|---|---|---|
| `k` | legal | legal |
| `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel` | legal | legal |
| `init` | legal | **refused** |
| `seed` | legal | **refused** |

<!-- generated: end call causalab.protocol.schema.render_field_legality_table subspace -->

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec k parametrization init seed -->

- **`k`**: Width of a `subspace` or `pca` feature space. Interchanges act in the first `k` basis columns and preserve the complementary `d − k` directions. Sweepable.
- **`parametrization`**: Parameter map: `PARAMETRIZATIONS` for `subspace` or `GATE_PARAMETRIZATIONS` for `gate`. Gates default to `sigmoid`. A `boundary` gate learns one scalar over the preceding ordered basis; its allowed fields are listed in `FEATURIZER_FIELD_CONDITIONS`.
- **`init`**: Initial values for a fit (§2.5). A subspace accepts `{"file_path": …, "entry": …}` and uses the saved basis's first `k` columns. A gate accepts `{"fill": p}`, a saved `theta` through `file_path`, or `{"from_scores": …}`. `fill` maps p to θ for the chosen parametrization; under `boundary`, p is the retained fraction. `from_scores` initializes the top `keep` units or scales z-scored values. `entry` selects a bundle entry. A loaded featurizer cannot declare `init`.
- **`seed`**: Seed for a subspace's initial rotation. Defaults to `train.seed`, or 0 without training. Sweep seeds on an untrained rank-k subspace to construct matched random controls.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec k parametrization init seed -->

## 3. Parametrizations

<!-- generated: begin doc causalab.protocol.schema.featurizers.PARAMETRIZATIONS -->

Maps from stored parameters to an orthonormal `(d, k)` basis `Q` (§2.5). `cayley` uses the Cayley transform from the initial basis, with `O(d k²)` work per access. `matrix_exp` uses a skew-symmetric matrix exponential; `stiefel` uses Householder reflections through torch's `orthogonal` maps. A loaded document must match the bundle's parametrization (rule 15).

<!-- generated: end doc causalab.protocol.schema.featurizers.PARAMETRIZATIONS -->

## 4. Training

Set `train.params` to the rotation. `l1` and `l2` regularize its stored weight
with `|p|` and `p²`. The `l0` mask penalty applies to gates.

<!-- generated: begin attrs causalab.protocol.schema.TrainSpec anneal phases -->

- **`anneal`**: Open-loop schedules keyed by `<name>.<slot>.<hyperparameter>` or a named objective term's `train.objective.<name>.weight` (§2.11).
- **`phases`**: Consecutive training windows (§2.11). Each selects trainable parameters and annealing schedules. Defaults to one phase.

<!-- generated: end attrs causalab.protocol.schema.TrainSpec anneal phases -->

## 5. Applying a fitted subspace

Load a bundle through `file_path`. The document must match its model, site, rank,
parametrization, and dtype, which are recorded in `ArtifactIdentity`.

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

- **`file_path`**: Path to a fitted artifact. The loaded featurizer uses its saved parameters and accepts no training, initialization, or training-rule fields. A loaded budget gate requires `top_k`. `ArtifactIdentity` is checked at load and build (rule 15). Sweep bundle paths to compare fits.
- **`entry`**: Coordinate selector for a bundle loaded through `file_path`. Required when the bundle contains several entries; otherwise the sole entry is used.
- **`dtype`**: Precision used to hold and save featurizer parameters (`PRECISION_DTYPES`). Defaults to the model precision. A loaded document must match the bundle's dtype.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

<a id="6-learning-the-rank-boundless-das"></a>

## 6. Learn the rank with DBM-DAS

DBM-DAS can learn a prefix of the rotated basis using the boundary gate from
Boundless DAS ([Wu et al. 2023](https://arxiv.org/abs/2305.08809)). Place a `boundary`
gate directly after the rotation. Its scalar `θ ∈ [0, 1]` sets `β = θ · k`;
the hard mask keeps the first `⌈β⌉` columns.

```json
"featurizers": {
  "rot": {"kind": "subspace", "k": 64, "parametrization": "cayley"},
  "bnd": {"kind": "gate", "parametrization": "boundary"}
},
"intervened_models": {
  "original": {"input": "counterfactual", "reads": ["v_cf"]},
  "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]}
},
"reads": {"v_cf": {"site": "target", "pos": -1, "featurizer": ["rot", "bnd"]}, "logits": {"site": "lm_head", "pos": -1}},
"writes": {"patch": {"site": "target", "pos": -1, "featurizer": ["rot", "bnd"], "do": {"swap": "v_cf"}}},
"train": {
  "objective": {
    "ce": {"weight": 1.0, "read": "logits", "model": "patched",
           "aggregation": {"kind": "cross_entropy", "target": "label"}},
    "l1": {"weight": 1.0, "l1": "bnd"}
  },
  "params": ["rot", "bnd"],
  "optimizer": {"name": "adamw", "lr": {"rot": 0.001, "bnd": 0.01}},
  "anneal": {"bnd.theta.temperature": [1.0, 0.1, 1.0]}
}
```

Here `k` sets the largest available rank. The training mask is
`σ((β − i)/T)` and its `l1` penalty is the mean mask value. In the hard-mask
limit, weight `λ` gives penalty `λ · ⌈β⌉ / k`. The template uses `λ = 1`.
Choose learning rates for both the rotation and boundary; the example uses
ten times the rotation's rate for the boundary and anneals temperature over
the full fit. Small temperatures concentrate gradients near individual columns.

`fit_diagnostics.json` records `boundary` and `hard_mask_size`. Report the latter
as the learned rank, and evaluate a matched random subspace at that rank.
Each boundary gate must follow a `subspace` or `pca` in every chain. Its scalar
parameter excludes per-unit options such as `group`, `top_k`, and `pool`; see
the [gate field table](dbm.md#2-fields).

## 7. What is refused

<!-- generated: begin call causalab.protocol.schema.render_field_refusals subspace -->

- `init` with `file_path`: 'init' sets the starting point of a fit; a loaded featurizer uses the weights in file_path
- `seed` with `file_path`: 'seed' selects an initial rotation; a loaded featurizer uses its saved rotation

<!-- generated: end call causalab.protocol.schema.render_field_refusals subspace -->

## 8. Shipped templates

- [`das`](../../demos/methods/protocols/das.json): fits one rotation on `block_output`.
- [`das_pca_init`](../../demos/methods/protocols/das_pca_init.json): fits a rank sweep initialized from PCA.
- [`das_boundless`](../../demos/methods/protocols/das_boundless.json): learns a rank within a 64-column rotation using DBM-DAS.
- [`random_subspace_control`](../../demos/methods/protocols/random_subspace_control.json): sweeps seeds for random bases at a matched rank.
- [`weekdays_das_apply`](../../demos/methods/protocols/weekdays_das_apply.json): evaluates a saved rotation on the test split.
- [`weekdays_das_sweep`](../../demos/methods/protocols/weekdays_das_sweep.json): fits ranks and seeds at a selected layer.

## 9. Demos

- [Subspace interventions](../../demos/onboarding_tutorial/07_subspace.md)
- [Weekdays geometry](../../demos/onboarding_tutorial/weekdays_geometry.md)
