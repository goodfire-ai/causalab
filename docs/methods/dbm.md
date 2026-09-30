# Desiderata-Based Masking (DBM)

DBM learns a mask with supervision that specifies the desired behavior. Causalab
represents the mask with a `gate` featurizer. An interchange uses counterfactual
activations at selected units and preserves the original activations elsewhere.
When trained with a DAS rotation, the method is DBM-DAS.

Start from a [shipped template](#7-shipped-templates), choose the units to mask,
and set the training objective. Evaluate the saved mask with an apply document.
The held-out scores measure how well its interventions satisfy the desiderata.

The feature map is `featurize(x) = (m⊙x, (1−m)⊙x)`. Its parameter slot is
`<name>.theta`, whose shape comes from the model and site. `train.params` may
name the gate or that slot. The mask used for training and evaluation depends
on its parametrization.

## 1. Where it sits

A gate attaches to a site with a feature width. A `group` shares parameters
across units on a named axis. The registry derives these groups from the model:

<!-- generated: begin call causalab.protocol.registry.render_gate_group_table -->

| group | axis | components it is legal on |
|---|---|---|
| `head` | `head` | `delta_gate`, `delta_query`, `delta_key`, `delta_value`, `delta_beta`, `delta_decay`, `delta_kv_mem`, `delta_state_update`, `delta_kernel_output`, `attention_query_pre_rope`, `attention_key_pre_rope`, `attention_value_states`, `attention_gate`, `attention_query`, `attention_key`, `attention_z`, `deltanet_query`, `deltanet_key`, `deltanet_state`, `attention_result`, `delta_premix`, `attention_premix` |
| `expert_neuron` | `topk` | `expert_activation`, `expert_neuron_output` |
| `site` | `feature` | `embeddings`, `block_input`, `attention_input_norm`, `delta_qkv`, `delta_gate`, `delta_conv`, `delta_query`, `delta_key`, `delta_value`, `delta_beta`, `delta_decay`, `delta_kv_mem`, `delta_state_update`, `delta_kernel_output`, `attention_query_pre_rope`, `attention_key_pre_rope`, `attention_value_states`, `attention_gate`, `attention_query`, `attention_key`, `attention_z`, `deltanet_query`, `deltanet_key`, `deltanet_state`, `attention_result`, `delta_premix`, `attention_output`, `attention_premix`, `block_mid`, `mlp_input_norm`, `mlp_input`, `mlp_output`, `router_logits`, `router_scores`, `expert_gate_proj`, `expert_up_proj`, `expert_activation`, `expert_neuron_output`, `expert_output`, `routed_output`, `shared_expert_gate_proj`, `shared_expert_up_proj`, `shared_expert_activation`, `shared_expert_output`, `shared_expert_gate`, `block_output`, `ln_final`, `lm_head` |

<!-- generated: end call causalab.protocol.registry.render_gate_group_table -->

<!-- generated: begin doc causalab.protocol.schema.featurizers.GATE_GROUPS -->

Units that share one gate parameter (§2.5). `head` assigns one parameter per head on a component with a head axis. `expert_neuron` assigns one per `(expert, neuron)` on `expert_activation` or `expert_neuron_output`; lookup through `expert_idx` preserves that identity across routing choices. `site` assigns one parameter to the whole site, with a `(1, width)` group map.

<!-- generated: end doc causalab.protocol.schema.featurizers.GATE_GROUPS -->

## 2. Fields

<!-- generated: begin call causalab.protocol.schema.render_field_legality_table gate -->

| field | without `file_path` | with `file_path` |
|---|---|---|
| `parametrization` ∈ `sigmoid` \| `clamp` \| `hard_concrete` \| `budget` \| `boundary` | any map | any map |
| `group` ∈ `head` \| `expert_neuron` \| `site` | any map but `boundary` | any map but `boundary` |
| `axis` ∈ `position` | any map but `boundary` | any map but `boundary` |
| `init` | any map | **refused** |
| `temperature` | `hard_concrete`, `boundary` | `hard_concrete`, `boundary` |
| `stretch` | `hard_concrete` | `hard_concrete` |
| `dead` ∈ `freeze_after` \| `leak` | any map but `boundary` | **refused** |
| `top_k` | **refused** | any map but `boundary` |
| `k_schedule` | `budget` | **refused** |
| `stop_grad_shift` | `budget` | **refused** |
| `pool` | `budget` | any map but `boundary` |

<!-- generated: end call causalab.protocol.schema.render_field_legality_table gate -->

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec parametrization group axis init temperature stretch dead top_k k_schedule stop_grad_shift pool -->

- **`parametrization`**: Parameter map: `PARAMETRIZATIONS` for `subspace` or `GATE_PARAMETRIZATIONS` for `gate`. Gates default to `sigmoid`. A `boundary` gate learns one scalar over the preceding ordered basis; its allowed fields are listed in `FEATURIZER_FIELD_CONDITIONS`.
- **`group`**: Unit covered by one gate parameter: `head`, `expert_neuron`, or `site` (§2.5). The default is one parameter per coordinate and is omitted from the canonical form. The model and component determine the coordinate-to-group map.
- **`axis`**: `"position"` assigns θ to addressed token positions (§2.5). The default `None` assigns θ to features.
- **`init`**: Initial values for a fit (§2.5). A subspace accepts `{"file_path": …, "entry": …}` and uses the saved basis's first `k` columns. A gate accepts `{"fill": p}`, a saved `theta` through `file_path`, or `{"from_scores": …}`. `fill` maps p to θ for the chosen parametrization; under `boundary`, p is the retained fraction. `from_scores` initializes the top `keep` units or scales z-scored values. `entry` selects a bundle entry. A loaded featurizer cannot declare `init`.
- **`temperature`**: Temperature β for `hard_concrete` or T for `boundary`. Defaults to `HARD_CONCRETE_TEMPERATURE` or 1, respectively. Sweepable. Use an `anneal` schedule to change the temperature during fitting; declaring both a constant and a schedule for the same gate is invalid. The sigmoid temperature is controlled through `anneal`.
- **`stretch`**: Fixed `[γ, ζ]` bounds for `hard_concrete`, defaulting to `HARD_CONCRETE_STRETCH`. These bounds determine the evaluation threshold and enter `ArtifactIdentity`. This field cannot be swept.
- **`dead`**: Rule for hard-off units in a trained gate: exactly one of `{"freeze_after": n}` or `{"leak": ε}` (§2.5). The gate must appear in `train.params` and use a per-unit map. This training rule is excluded from the saved bundle's `ArtifactIdentity`.
- **`top_k`**: Number of units selected from a loaded gate's largest `theta` values. Requires `file_path` and a per-unit map. Sweepable over integers in `[0, units]`; 0 keeps the original activations. A loaded `budget` gate requires this field. Other maps use their threshold when it is omitted.
- **`k_schedule`**: Training budget for a `budget` gate, required during fitting (§2.5). Use `{"kind": "fixed", "k": n}` or `{"kind": "uniform" | "log_uniform", "low": a, "high": b}`. `eval` sets the evaluation cut and the reported `hard_mask_size`; it defaults to `k` for a fixed schedule and is required for a sampled schedule. `k` and `eval` are sweepable. Loaded gates use `top_k`.
- **`stop_grad_shift`**: Training option for `budget` gates. When true, the solved shift is constant in backpropagation, leaving only the direct `σ'` gradient. By default the shift carries `∂c/∂θ_i = −σ'_i / Σ σ'_j`, preserving `Σ m = k` to first order under an update.
- **`pool`**: Name of a shared budget or readout pool (§2.5). During fitting, `budget` gates with this name share one `k_schedule`, solved shift, and ranking across their units. Loaded per-unit gates may share one `top_k` cut under any map. Members must agree on the schedule or cut. The name cannot be swept. Saved budget pools stamp `pool` and `pool_units`; loading must match that identity. An unstamped bundle may join a readout pool.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec parametrization group axis init temperature stretch dead top_k k_schedule stop_grad_shift pool -->

## 3. Parametrizations

<!-- generated: begin doc causalab.protocol.schema.featurizers.GATE_MAPS -->

Gate parametrizations map `theta` to a mask (§2.5). The table gives their formulas and allowed training options. The default is `sigmoid`; its omitted spelling stays absent from the canonical form.  `hard_concrete` uses the stochastic L0 relaxation of Louizos, Welling and Kingma (2018, arXiv 1712.01312). Each optimizer step samples one mask shared across the gate's reads and writes. Evaluation uses the deterministic stretched sigmoid and a threshold of ½. Its `l0` penalty is the expected fraction kept.  `budget` learns a ranking with a fixed-sum mask `σ(θ + c_k)`. Bisection solves the shift `c_k` for the step's `k_schedule` budget. Evaluation requires a cut: `k_schedule.eval` during fitting and `top_k` after loading. `stop_grad_shift` treats the solved shift as constant in the gradient.  `boundary` learns a prefix of an ordered basis, as in Boundless DAS (Wu et al. 2023, arXiv 2305.08809). Its single parameter `θ ∈ [0, 1]` gives boundary `β = θ · width` and retained rank `⌈β⌉`. It must follow a `subspace` or `pca` stage in every chain. The parameter has shape `[1]`; per-unit options are invalid. `init.fill p` starts at `θ = p`; the default is ½. Projection keeps θ in bounds after each optimizer step.

<!-- generated: end doc causalab.protocol.schema.featurizers.GATE_MAPS -->

<!-- generated: begin call causalab.protocol.schema.render_gate_map_table -->

| parametrization | soft mask (train) | after every optimizer step | hard mask (eval, apply) | mask penalty (`train.objective`) | `anneal` on `theta.temperature` | default start |
|---|---|---|---|---|---|---|
| `sigmoid` (absent) | `σ(θ / T)` | nothing | `θ > 0` | **`l1`** = `mean σ(θ/T)`; `l0` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `clamp` | `θ` itself | `θ ← clip(θ, 0, 1)` | `θ > ½` (`round`) | **`l1`** = `mean θ`; `l0` is **refused** (rule 4) | **refused** (rule 4): a clamp gate uses θ directly, projected into [0, 1] after each step | `θ = ½` |
| `hard_concrete` | **sampled**, once per optimizer step: `u ~ U(0,1)`, `s = σ((log u − log(1−u) + θ)/β)`, then `clip(s·(ζ−γ)+γ, 0, 1)` | nothing | `clip(σ(θ)·(ζ−γ)+γ, 0, 1) > ½`, i.e. `θ > logit((½−γ)/(ζ−γ))`: exactly `θ > 0` at the default stretch | **`l0`** = `mean σ(θ − β·log(−γ/ζ))`, the expected kept fraction of the sampled mask; `l1` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `budget` | `σ(θ + c_k)` with the step's budget `k` drawn from `k_schedule` and the scalar `c_k` solved so `Σ m = k` exactly | nothing | largest `θ` values: `k_schedule.eval` during fitting, `top_k` after loading | none; the budget fixes mask size. `l1` and `l0` are invalid (rule 4) | **refused** (rule 4): a budget gate fixes sharpness through its budget and solved shift | `θ = 0` (`fill` ½) |
| `boundary` | `σ((β − i) / T)` over the coordinate index `i = 0 … width−1` of the stage's input: the rotation's columns or the PCA components; `θ ∈ [0, 1]` is the one scalar, the boundary as a fraction of the width, `β = θ · width` | `θ ← clip(θ, 0, 1)` | `i < β`: the first `⌈β⌉` coordinates | **`l1`** = `mean σ((β − i)/T)`, the kept fraction (`⌈β⌉ / width` as `T → 0`); `l0` is **refused** (rule 4) | legal | `θ = ½`, the half prefix (`fill` ½) |

<!-- generated: end call causalab.protocol.schema.render_gate_map_table -->

## 4. Training

Set `train.params` to the gate. Use the mask penalty allowed by its
parametrization: `{"l1": ["<gate>"]}` or `{"l0": ["<gate>"]}`.
`l2` is allowed on any trained featurizer. Temperature schedules use
`<gate>.theta.temperature` for the maps that support annealing.

<!-- generated: begin attrs causalab.protocol.schema.TrainSpec anneal control phases -->

- **`anneal`**: Open-loop schedules keyed by `<name>.<slot>.<hyperparameter>` or a named objective term's `train.objective.<name>.weight` (§2.11).
- **`control`**: Closed-loop schedules: `{<target>: {kind, signal, setpoint, gains, …}}` (§2.11). A target is a named objective weight or an anneal- style hyperparameter path. Its declared value initializes the controller.
- **`phases`**: Consecutive training windows (§2.11). Each selects trainable parameters and annealing schedules. Defaults to one phase.

<!-- generated: end attrs causalab.protocol.schema.TrainSpec anneal control phases -->

<!-- generated: begin doc causalab.protocol.schema.featurizers.GATE_DEAD_RULES -->

Rules for units whose training mask becomes hard-off (§2.5). Choose one: `freeze_after: n` fixes a unit after `n` consecutive hard-off optimizer steps; `leak: ε` adds `ε` to `∂m/∂θ` so saturated units can receive gradients. The leak preserves the forward value and evaluation mask. An omitted rule is absent from the canonical form.

<!-- generated: end doc causalab.protocol.schema.featurizers.GATE_DEAD_RULES -->

## 5. Applying a fitted gate

Load the bundle through `file_path`. Reuse the fit's model, site, group,
parametrization, and dtype so its `ArtifactIdentity` matches. A loaded budget
gate requires `top_k`; a pool can apply one cut across several gates.

<!-- generated: begin attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

- **`file_path`**: Path to a fitted artifact. The loaded featurizer uses its saved parameters and accepts no training, initialization, or training-rule fields. A loaded budget gate requires `top_k`. `ArtifactIdentity` is checked at load and build (rule 15). Sweep bundle paths to compare fits.
- **`entry`**: Coordinate selector for a bundle loaded through `file_path`. Required when the bundle contains several entries; otherwise the sole entry is used.
- **`dtype`**: Precision used to hold and save featurizer parameters (`PRECISION_DTYPES`). Defaults to the model precision. A loaded document must match the bundle's dtype.

<!-- generated: end attrs causalab.protocol.schema.FeaturizerSpec file_path entry dtype -->

## 6. What is refused

The field conditions below supplement the specification's
[validation rules](../intervention_protocol.md#5-validation--load-error-checklist).

<!-- generated: begin call causalab.protocol.schema.render_field_refusals gate -->

- `group` under another map: 'group' requires one theta entry per unit; the authored map uses one scalar boundary (§2.5)
- `axis` under another map: 'axis' requires theta entries per position; the authored map defines a prefix of an ordered feature basis (§2.5)
- `init` with `file_path`: 'init' sets the starting point of a fit; a loaded featurizer uses the weights in file_path
- `temperature` under another map: 'temperature' requires hard_concrete or boundary; the selected map is the authored map (§2.5)
- `stretch` under another map: 'stretch' requires hard_concrete; the selected map is the authored map (§2.5)
- `dead` with `file_path`: 'dead' applies during training; a loaded gate uses a fixed mask (§2.5)
- `dead` under another map: 'dead' requires theta entries per unit; the authored map uses one scalar boundary (§2.5)
- `top_k` without `file_path`: 'top_k' requires file_path and selects the largest saved theta entries; training uses the map's own evaluation mask (§2.5)
- `top_k` under another map: 'top_k' requires a ranking of units; the authored map selects a prefix using i < β (§2.5)
- `k_schedule` with `file_path`: 'k_schedule' supplies training budgets; a loaded gate uses top_k (§2.5)
- `k_schedule` under another map: 'k_schedule' and 'stop_grad_shift' require budget; the selected map is the authored map (§2.5)
- `stop_grad_shift` with `file_path`: 'stop_grad_shift' controls the solved shift's training gradient; a loaded gate uses top_k (§2.5)
- `stop_grad_shift` under another map: 'k_schedule' and 'stop_grad_shift' require budget; the selected map is the authored map (§2.5)
- `pool` under another map: 'pool' requires budget during fitting. A loaded pool requires per-unit maps, file_path, and a shared top_k; the selected map is the authored map (§2.5)
- [V23] `group_legality`: Group legality (§5.23)
- [V32] `scores_init`: A gate's `init.from_scores` fits the gate (§5.32)

<!-- generated: end call causalab.protocol.schema.render_field_refusals gate -->

## 7. Shipped templates

- [`dbm`](../../demos/methods/protocols/dbm.json): fits a coordinate gate on `block_output` with cross-entropy and `l1` loss.
- [`dbm_apply`](../../demos/methods/protocols/dbm_apply.json): evaluates a saved coordinate gate.
- [`dbm_head`](../../demos/methods/protocols/dbm_head.json): fits a gate with one parameter per attention head.
- [`dbm_head_apply`](../../demos/methods/protocols/dbm_head_apply.json): evaluates that head gate.
- [`dbm_expert_neuron`](../../demos/methods/protocols/dbm_expert_neuron.json): fits routed-expert and shared-expert gates together.
- [`dbm_expert_neuron_apply`](../../demos/methods/protocols/dbm_expert_neuron_apply.json): evaluates both expert gates.
- [`das_boundless`](../../demos/methods/protocols/das_boundless.json): fits a rotation and a boundary gate with DBM-DAS; see [learning the rank](das.md#6-learning-the-rank-boundless-das).

## 8. Demos

- [Component masking](../../demos/onboarding_tutorial/09_components.md)
- [Masking over MLP neurons](../../demos/papers/arithmetic_neurons.md): one gate per neuron on `mlp_neuron_output`, the per-unit default that needs no `group`
