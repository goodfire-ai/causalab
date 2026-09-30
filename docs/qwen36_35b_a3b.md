# Qwen3.6-35B-A3B

This page describes the hookpoints of `Qwen/Qwen3.6-35B-A3B`, the mixture of
experts model that most method templates use: its layers, the components a
document can address, and the limits on each.

The diagram draws the forward pass with one box per tensor. A lavender box is
a component some engine exposes; a grey box is a tensor that no component
names. [Open the diagram on its own page](qwen36-35b-a3b-architecture.html),
or click it.

<a class="arch-diagram-link" href="qwen36-35b-a3b-architecture.html"
   aria-label="Open the Qwen3.6-35B-A3B architecture diagram on its own page"><iframe
   class="arch-diagram" src="qwen36-35b-a3b-architecture.html" tabindex="-1"
   title="Qwen3.6-35B-A3B internal architecture diagram" loading="lazy"></iframe></a>

## Architecture

Qwen3.6-35B-A3B has 40 layers. Layers 3, 7, …, 39 use gated full attention;
the other 30 use Gated DeltaNet. Each layer has 256 routed experts, of which
eight run per token, plus a shared expert.

| | |
|---|---|
| layers | 40: 30 `linear_attention`, 10 `full_attention` |
| hidden size | 2048 |
| full attention | 16 query heads, 2 KV heads (GQA), `head_dim` 256, output-gated, partial RoPE (0.25) |
| Gated DeltaNet | 16 key heads, 32 value heads (GVA), `d_k` = `d_v` = 128, causal conv kernel 4, 64-token chunked kernel |
| MoE | 256 experts, top-8, `moe_intermediate_size` 512; shared expert 512 |

The loader checks component availability against the registry's layer types
(`[V4]`). Execution checks the actual module (`[P4]`) when the registry lacks
this metadata. A component on the wrong mixer fails either check:

```json
{"component": "attention_probs", "layers": [3]}    // ✓ layer 3 is full attention
{"component": "attention_probs", "layers": [4]}    // ✗ [V4] a Gated DeltaNet block
                                                   //     computes no attention matrix
{"component": "block_output", "layers": [4], "stream": "linear_attention"}  // ✓ optional, checked
```

## Components

`blocks` identifies layers where the tensor exists. `shape` lists axes:
`head·feature` is flattened in head order, and `batch·position` flattens the
MoE token axis. `tap` describes how the engine reads it. `write` lists permitted
operations and alternatives for restricted components.

The tables are generated from `causalab.protocol.registry` by
`scripts/generate_support_tables.py`. Registry data also drives validation and
engine capability checks.

Choose an engine from the table with the `--engine` flag. The document names
components; it is checked against that choice. For example,
`expert_permutation` requires `nnsight`:

```bash
uv run causalab explain patch.json \
    --data-root data \
    --artifacts-root . \
    --engine pytorch_hooks
# ...
# engine    refused: [V13] the engine does not support this document: it requires
#           [..., 'component:expert_permutation', ...] and lacks ['component:expert_permutation'] (sec. 8)
uv run causalab explain patch.json \
    --data-root data \
    --artifacts-root . \
    --engine nnsight
# ...
# engine    nnsight
```


<!-- generated: begin component-table -->

**Model boundary (no `layer`)**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `input_ids` | layer-less | `(batch, position)` | module input | both | read-only: token IDs are dataset values. Change the row text, or write 'embeddings' to edit their vectors |
| `embeddings` | layer-less | `(batch, position, feature)` | module output | both | any mechanism |
| `ln_final` | layer-less | `(batch, position, feature)` | module output | both | any mechanism |
| `lm_head` | layer-less | `(batch, position, feature)` | module output | both | any mechanism |

**Residual stream and dense MLP: every layer**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `block_input` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `attention_input_norm` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `attention_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `block_mid` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `mlp_input_norm` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `mlp_input` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `mlp_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `mlp_activation` | unavailable on this architecture | n/a | module output | both | any mechanism |
| `mlp_neuron_output` | unavailable on this architecture | n/a | module input | both | any mechanism |
| `block_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |

**Full-attention mixer interior: the 10 `full_attention` layers**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `attention_query_pre_rope` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_key_pre_rope` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_value_states` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_gate` | full-attn (10) | `(batch, position, head·fused·feature)` | module output | both | any mechanism |
| `attention_query` | full-attn (10) | `(batch, head, position, feature)` | attention-function slot | both | any mechanism |
| `attention_key` | full-attn (10) | `(batch, head, position[key], feature)` | attention-function slot | both | any mechanism |
| `attention_scores` | full-attn (10) | `(batch, head, position[query], key_position[key])` | attention-function slot | both | any mechanism |
| `attention_z` | full-attn (10) | `(batch, position, head, feature)` | attention-function slot | both | any mechanism |
| `attention_result` | full-attn (10) | `(batch, position, head·feature)` | derived from `attention_premix` | both | read-only: this per-head contribution is derived from the joint output projection. Write 'attention_premix' with the same 'head' to change it by the corresponding linear projection |
| `attention_premix` | full-attn (10) | `(batch, position, head·feature)` | module input | both | any mechanism |
| `attention_probs` | full-attn (10) | `(batch, head, position[query], key_position[key])` | module output | both | `swap` only: each row must sum to 1 for the following value multiply. Write 'attention_scores' to use other mechanisms before softmax restores normalized probabilities |

**Gated DeltaNet mixer interior: the 30 `linear_attention` layers**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `delta_qkv` | DeltaNet (30) | `(batch, position, feature)` | module output | both | any mechanism |
| `delta_gate` | DeltaNet (30) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `delta_conv` | DeltaNet (30) | `(batch, feature, position)` | delta-kernel boundary | both | any mechanism |
| `delta_query` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_key` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_value` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | both | any mechanism |
| `delta_beta` | DeltaNet (30) | `(batch, position, head)` | delta-kernel boundary | both | any mechanism |
| `delta_decay` | DeltaNet (30) | `(batch, position, head)` | delta-kernel boundary | both | any mechanism |
| `delta_kv_mem` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | read-only: this readout is recomputed from state as S_{t-1}·exp(g_t) · k̂_t at each step. Write 'delta_state' to change memory or 'delta_value' to change what is stored |
| `delta_state_update` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | read-only: state updates are exposed for reading. Write 'delta_state' to edit S_t in S_t = S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t |
| `delta_state` | DeltaNet (30) | `(batch, position[steps], head, state, state)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_kernel_output` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | both | any mechanism |
| `deltanet_query` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight` | any mechanism |
| `deltanet_key` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight` | any mechanism |
| `deltanet_state` | DeltaNet (30) | `(batch, position[chunk], head, feature)` | `.source` line (fused forward) | `nnsight` | any mechanism |
| `delta_premix` | DeltaNet (30) | `(batch, position, head·feature)` | module input | both | any mechanism |

**Sparse MoE + shared expert: every layer**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `router_logits` | every layer (40) | `(batch·position, feature)` | module output | both | read-only: routing uses the scores and indices already computed from these logits. Write 'router_scores' to reweight selected experts or 'expert_idx' to select experts |
| `router_scores` | every layer (40) | `(batch·position, topk)` | module output | both | any mechanism |
| `expert_idx` | every layer (40) | `(batch·position, topk)` | module output | both | `swap` only: expert IDs are integer labels. Use swap to replace the routing indices, or write 'router_scores' to reweight selected experts. Arithmetic on IDs can select arbitrary experts or exceed index bounds |
| `expert_gate_proj` | every layer (40) | `(batch·position, topk·fused·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_up_proj` | every layer (40) | `(batch·position, topk·fused·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_activation` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_neuron_output` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_permutation` | every layer (40) | `(batch·position, topk)` | `.source` line (fused forward) | `nnsight` | read-only: the kernel derives this ordering of token-slot rows from routing. Write 'expert_idx' to select experts or 'router_scores' to reweight them |
| `expert_output` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `routed_output` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_gate_proj` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_up_proj` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_activation` | every layer (40) | `(batch·position, feature)` | module input | both | any mechanism |
| `shared_expert_output` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_gate` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |

<!-- generated: end component-table -->

### Dense neuron sites

Qwen3.6-35B-A3B has a sparse MoE block at each layer. Its registry entry
has no dense inner width, so `validate` refuses both dense sites with
`component_unavailable`. Use `expert_neuron_output` for routed experts and
`shared_expert_activation` for shared experts. Both expose complete neuron
outputs before the down projection. `expert_activation` remains the activated
gate branch inside routed experts.

### State reads

DeltaNet state reads grow fast with the sequence:

- `delta_state` is one `d_k × d_v` matrix per head per step: 30 layers ×
  seq × 32 × 128 × 128 floats if you ask for every layer at `pos: "all"`.
  Address positions in the read, not afterwards: the gather runs before
  anything is kept.

The [hookpoints section](running_experiments.md#5-hookpoints) of the
experiment guide gives the rules that hold on every model.
