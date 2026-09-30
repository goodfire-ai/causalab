# Running experiments

Causalab executes interventions from JSON documents. Each document specifies
the network and data, which activations to read or change, and which results to
save. This guide builds one interchange experiment.

Follow §§1–4 for a first run. See [workflows](#8-chaining-documents-workflows)
to connect experiments, or use the component and engine tables in §§5–6 as
reference.

The [intervention specification](intervention_protocol.md) defines field
semantics. The [workflow specification](workflow_protocol.md) covers connected
runs and analysis. See [code structure](CODEBASE.md) and [tests](TESTS.md) for
development instructions.

## Setup

```bash
uv sync                       # the nnsight engine ships in the dev group
uv run causalab --help
```

## 1. Serialize a dataset

A dataset reference resolves a saved table under `--data-root` or the packaged
task data. Packaged refs have the form `<task>/data/<variant>#<split>` and need
no flag. Validation reads those bytes without running a task generator.
To create a new MCQA table:

```bash
uv run python scripts/build_task_dataset.py \
    --task MCQA \
    --n 32 \
    --seed 0 \
    --split all \
    --target-variable answer \
    --out data/mcqa.json
# wrote data/mcqa.json (32 rows, digest 355e6b69d4b5…)
```

Record the build command with the study so the table can be reproduced.

## 2. Write the document

This experiment reads a residual activation from the counterfactual input and
interchanges it into the original input. It measures agreement with the causal
model's intervened answer. Save the document as `patch.json`:

```json
{
  "header": {
    "protocol_version": "4",
    "description": "Interchange the answer-slot residual stream at one layer."
  },
  "model": {"key": "Qwen/Qwen3.6-35B-A3B", "revision": "main"},
  "data": {
    "base": {"dataset": "mcqa", "field": "input"},
    "counterfactual": {"dataset": "mcqa", "field": "counterfactual_inputs[0]"}
  },
  "method": {
    "intervened_models": {
      "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
      "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]}
    },
    "sites": {
      "target": {"component": "block_output", "layers": [20]},
      "lm_head": {"component": "lm_head"}
    },
    "reads": {"v_cf": {"site": "target", "pos": -1}, "logits": {"site": "lm_head", "pos": -1}},
    "writes": {
      "patch": {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}
    },
    "save": [
      {
        "read": "logits",
        "model": "patched",
        "aggregation": {"kind": "match", "expected": "label"},
        "file_path": "iia.json"
      },
      {
        "read": "logits",
        "model": "patched",
        "aggregation": {
          "kind": "logit_diff",
          "a": "cf_answer",
          "b": "base_answer"
        },
        "file_path": "logit_diff.json"
      }
    ]
  }
}
```

The `method` group defines the experiment:

- `sites` names every activation address, including `lm_head`.
- Each `read` names an address: a site and a position.
- A `write` defines an intervention. An `intervened_models` entry runs the
  network on an input, lists the reads taken on it and the writes in force;
  one write or read can be listed by several models. A model with no writes
  is the network as it is, conventionally named `original`.
- An aggregation reduces a read on a model against the table's answer
  columns; it lives on the `save` entry, objective term or eval entry that
  consumes it.
- `save` lists the outputs to write.

`v_cf` passes a value from the counterfactual run to the intervened run.
The resulting dependencies determine execution order.

### Document sections

| Group | Contents |
|---|---|
| `header` | `protocol_version: "4"`, with optional title and description. |
| `model` | Model key, revision, precision, quantization, and optional attention backend. |
| `data` | Original (`base`) and counterfactual table references. |
| `method` | Intervened models, sites, reads, writes, training, and saves (with their aggregations). |

Optional method sections define positions, feature maps, free parameters, and
custom code. See [document layout](intervention_protocol.md#1-document-layout)
for the full field reference.

### Closed vocabularies

Fields with a fixed vocabulary reject values outside it.

| vocabulary | values |
|---|---|
| `sites.component` | 56 names; the 54 the A3B exposes are tabulated on the [Qwen3.6-35B-A3B page](qwen36_35b_a3b.md#components) (`mlp_activation` and `mlp_neuron_output` have no tensor on this architecture); eight retired `deltanet_*` spellings are aliases (§5) |
| `sites.stream` | `full_attention` · `linear_attention`: a per-layer fact on a hybrid tower, refused at load if the layer carries the other one |
| `sites.head` / `sites.expert` | sub-axis selectors, legal only where the component has that axis |
| `writes.do` | `swap` · `add_scaled` · `lerp` · `affine` · `gaussian` · `renormalize` · `clamp` · `pytorch_fn` (local-only; names a `code` declaration) |
| `aggregation.kind` | `logit_diff` · `token_logit` · `cross_entropy` · `kl` · `class_probs` · `token_logits` · `top_k` · `match` · `decode` |
| `featurizers.kind` | `identity` · `subspace` · `pca` · `sae` · `standardize` · `gate` |
| `pos` forms | `-1` (sugar for `{"index": n}`) · `"all"` · `{"variable": v}` · `{"column": c}` · `{"span": [a, b]}` (half-open), modified by `scope` / `relative_to` / `generated` |
| save formats | `.json` (per-example tables) · `.safetensors` (dense numerics) |

The following rules help catch document errors:

| rule | consequence |
|---|---|
| one global namespace over the named sections | every name unique; `base`, `counterfactual`, `counterfactual[j]` and `all` are reserved |
| at most one **absolute** write per (site, overlapping pos, model) | any number of additive writes; absolute applies first, then the summed deltas: so write sets are order-free |
| `{"sweep": [v, …]}` / `{"sweep": {"range": [a, b]}}` is the only axis | bare arrays are never axes; axis identity is name identity, so sweeping `sites.target.layers` moves the read, the write and the metric together |

The compiler derives feature widths, forward counts, point counts, required
capabilities, and digests. Declare the choices these depend on.

### Generating documents

The following scripts generate documents for repeated layer and component
patterns. They reject duplicate names and validate their output. Use
`--register-from-hf` for a model outside the registry. Qwen3.6-35B-A3B has a
built-in entry.

`expand_layers.py` expands a one-layer, one-site harvest template. At each
layer it adds `L{n}A` (`block_mid`) and `L{n}M` (`block_output`), with reads
and saves for each selected position. A save-time `reduce` carries over.
Templates must be pure reads at one site, without featurizers.

```bash
uv run python scripts/expand_layers.py harvest_template.json --out harvest.json
# wrote harvest.json (80 sites, 160 reads, 160 save entries)      # 40 layers × 2 positions
uv run python scripts/expand_layers.py harvest_template.json \
    --out harvest_early.json \
    --layers 0:8 \
    --positions answer_tok
```

`add_routing_reads.py` adds `expert_idx` and `router_scores` reads at every
MoE layer for each executed model/input pair. Positions default to those
already read by the pair; `--positions` can select named positions, integers,
or `all`. Generated names are
`routing_L{n}_{model}_{input}_{position}_{idx|scores}`.
The model must declare experts in the registry.

```bash
uv run python scripts/add_routing_reads.py patch.json --out patch_routed.json
# wrote patch_routed.json (160 routing reads added over 162)     # 40 layers × 2 pairs × 2
uv run causalab validate patch_routed.json \
    --engine auto \
    --data-root data \
    --artifacts-root .
```

`joint_dbm.py` fits Desiderata-Based Masking (DBM) gates across layers under
one sparsity penalty. Choose components with `--family`:

| Family | Mask unit | Sites |
|---|---|---|
| `heads` | Whole head | `attention_premix` on full-attention layers; `delta_premix` on Gated DeltaNet layers |
| `head-channels` | Channel within a head output | The same premix sites, with a coordinate gate |
| `neurons` | Complete MLP neuron output | `expert_neuron_output` with `group: expert_neuron` and `shared_expert_activation`; `mlp_neuron_output` on dense layers |
| `all-neurons` | Head-output channel or MLP neuron | The union of `head-channels` and `neurons` |

Routed and shared experts use `act(gate) * up`, before the down projection.
A routed gate has one parameter per expert and neuron. The routing table
maps those parameters to the active slots. Add `--dense` to request neurons
on a model without experts. Head families require the registry's
`layer_types` to identify each layer's mixer.

Use `--positions 0 1 2` to fit independent gates at explicit aligned token
positions. Each layer and position gets separate parameters under the same
objective. Supply distinct nonnegative positions that exist in every base
and counterfactual prompt. `--position all` broadcasts one gate across all
positions. `--readout-position -1` reads answer logits at the final token,
independently of the intervention positions.

The fit crosses `--penalties` with `--seeds`. IIA compares the output with
`--label-column` (default `label`). `logit_diff` subtracts the logit for
`--base-answer-column` (default `base_answer`) from the gold-label logit in
the intervened output. `ce` trains against that same gold label. Validation
saves IIA and logit difference by default.

```bash
uv run python scripts/joint_dbm.py \
    --family all-neurons \
    --model Qwen/Qwen3.6-35B-A3B \
    --variable output \
    --dataset weekdays/data#train \
    --validation weekdays/data#validation \
    --positions 0 1 2 \
    --readout-position -1 \
    --penalties 0.001 0.003 0.01 0.03 0.1 \
    --seeds 0 1 2 \
    --data-root data \
    --out neurons_fit.json
```

Use the same model, family, layer and position flags for replay. The apply
document loads each gate's saved parameters and evaluates its hard mask on
`--confirmation`. `--apply --penalty 0.01 --seed 0` selects one saved cell.
`--apply-all` evaluates every penalty and seed cell, with all gates linked to
one `axes.fit_cell` axis. Each row holds the fit's exact bundle selector in
`entry`. Metric rows carry the corresponding `axes.fit_cell` row index.

```bash
uv run python scripts/joint_dbm.py \
    --family all-neurons \
    --model Qwen/Qwen3.6-35B-A3B \
    --variable output \
    --dataset weekdays/data#train \
    --positions 0 1 2 \
    --readout-position -1 \
    --penalties 0.001 0.003 0.01 0.03 0.1 \
    --seeds 0 1 2 \
    --apply-all \
    --fit-dir runs/neurons_fit \
    --confirmation weekdays/data#confirmation \
    --data-root data \
    --artifacts-root . \
    --out neurons_apply.json
```

Each fit saves its gate bundles. A replay uses `theta > 0` to select units.
Optional `--save-rank` writes `rank.json` in either document, with one record
per unit per evaluated cell. Limit this option to small audits because an
all-token neuron campaign can produce millions of records. Report viewers
must use the mask evaluated for each saved cell. Separate fits can select
different components at the same sparsity.

Both documents pass through the loader before writing. Apply validation
checks bundle identities under `--artifacts-root`. The training defaults
come from `demos/methods/protocols/dbm.json`; `--help` lists the controls.

**`export_dbm.py`** joins saved apply metrics to the frozen gate bundles
used for evaluation. Its manifest groups apply runs by experiment. Paths
resolve from the manifest's folder.

```json
{
  "experiments": [{
    "id": "output-neurons",
    "title": "Output DBM over neurons",
    "evaluations": [{
      "document": "neurons_apply.json",
      "run_dir": "runs/neurons_apply",
      "data_root": "data",
      "artifacts_root": "."
    }]
  }]
}
```

```bash
uv run python scripts/export_dbm.py \
    --manifest dbm_manifest.json \
    --out output_dbm.json
```

The Python API is `causalab.analysis.export_dbm.export(manifest_path,
register_from_hf=False)`. Add `--register-from-hf` to the CLI, or set the
keyword to `True`, to resolve an unregistered model from its HF config.

Export checks the document and point digests and places each metric row on
the receipt point whose coordinates it carries. Each point contains the frozen
hard masks, selected count, metrics and provenance. Gate positions are integer
token indices or `"all"`; named scalar definitions resolve to those values.
Other position forms are refused.

IIA uses all eligible pairs. Logit difference uses eligible pairs whose gold
and original answers differ. Each metric carries its sample count, while
`omitted_points` records masks without eligible evaluations. A curve combines
runs only when their canonical model, data, gate layout and resolved metric
reads agree. The model identity includes dtype and quantization. Both scores
must read the base input under exactly the exported gate swaps. Each swap
reads the aligned counterfactual site through that gate.

Report applications add captions and token examples, then embed the JSON in
their own templates.

The generators share registry lookups and naming rules through
`scripts/protocol_authoring.py`.

<a id="3-check-it-before-you-spend-a-gpu"></a>

## 3. Validate and inspect

`validate` checks document structure, data references, and engine support.
`explain` reports the resulting plan. Both run without model weights.

Data checks cover named columns and prompt variables at representatives of
each sweep axis. Token positions and answer forms are checked when `run`
encodes the batch, before weights load. A window that is ragged across rows
can therefore fail at encoding even after document validation.

Inspection commands use the model registry to derive widths and layer types.
For this model they run offline and detect a full-attention site placed on a
DeltaNet layer.

```bash
uv run causalab validate patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --data
# Validation reports one point.

uv run causalab validate patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --data \
    --set sites.target.component=attention_premix
# refused: [V4] at sites.target.component site 'target': component
#          'attention_premix' exists only on a 'full_attention' mixer, but layer
#          layer 20 uses linear attention.
#          Layers carrying 'full_attention': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
```

For an unregistered model, use `--register-from-hf` to load its Hugging Face
configuration. Use the actual model key so widths and layer bounds match:

  ```bash
  uv run causalab validate patch.json \
      --engine auto \
      --data-root data \
      --artifacts-root . \
      --data \
      --register-from-hf
  # Validation reports one point.
  ```

`run` resolves unregistered model metadata automatically.

Use `explain` to estimate work from the point count and forward groups.
`requires` lists capabilities, with `:write` on edited components. The selected
engine must support all of them; `[V13]` names a shortfall.

`dry-run` combines validation and planning with per-site shapes, widths,
head spaces, and engine support. Its `undecided` list names checks that need
encoding, such as token windows and answer forms. The command exits 1 for a
reported error and 0 otherwise. `--engine` is required; `--data` adds the data
checks that `validate` performs by default. `--tokenizer` loads the model's
tokenizer, never its weights, and decides the token windows and answer
tokens as the run would; `validate --tokenizer` runs the same checks.

```bash
uv run causalab dry-run patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --set model.key=Qwen/Qwen3-4B-Instruct-2507
# dry-run   patch.json
# digest    cc2e2500fac130298f0513e6f836da2d16056660147fdbd03f1e37e65427e4cf
# overrides model.key=Qwen/Qwen3-4B-Instruct-2507
# model     Qwen/Qwen3-4B-Instruct-2507@main fp32
#   36 layers, hidden 2560, 32 heads (8 kv) x 128, vocab 151936, family qwen3; declares no layer pattern
# data
#   mcqa (base, counterfactual): digest … 20 columns
# points    1
# requires  ['component:block_output', 'component:block_output:write',
#            'component:lm_head', 'paired_forward']
# engine    pytorch_hooks: serves
# sites
#   target: block_output layer 20: available
#     shape (batch, position, feature), width 2560, no head axis
#     reads nnsight, pytorch_hooks; writes add_scaled, affine, clamp, gaussian, lerp, pytorch_fn, renormalize, swap
#   lm_head: lm_head: available
#     …
# inventory undecided (see below)
# readouts
#   v_cf: original on counterfactual at target -> (saved or operand only)
#   logits: patched on base at lm_head -> iia, logit_diff
# save
#   iia (model=patched, input=base) -> iia.json
#   logit_diff (model=patched, input=base) -> logit_diff.json
#   …
# undecided (decided when the run encodes its inputs): inventory, tokenization, pair_validity, controls
```

An unavailable component reports its reason code before model loading:

```bash
uv run causalab dry-run patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --set sites.target.component=routed_output \
    --set model.key=Qwen/Qwen3-4B-Instruct-2507
# refused: [V4] at sites.target.component site 'target': component 'routed_output'
#          needs a sparse-MoE block (the entry declares no experts), which model
#          'Qwen/Qwen3-4B-Instruct-2507' does not have — there is no such tensor on this model
#   code V4 (references_resolve) at sites.target.component, reason component_unavailable
```

## 4. Run it

For a CPU smoke check, use a small random model with the same architecture:
four layers with hybrid DeltaNet and full attention, plus sparse MoE.

```bash
uv run causalab run patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --out runs/patch \
    --set model.key=tiny-random/qwen3.5-moe \
    --set sites.target.layers=1 \
    --device cpu
# saved iia.json -> runs/patch/iia.json
# saved logit_diff.json -> runs/patch/logit_diff.json
# cells 2 / 2 eligible
```

A random-weight run checks execution and saving. Use trained weights for
scientific measurements. Save final model and method choices in the document;
`--set` supports temporary exploration.

The `cells` summary counts eligible measurements and exclusions, with one
cell per save entry and sweep point. An expert selected by no token can produce
`status: "unavailable"` with reason `empty_selector`. Preserve these exclusions
when reporting results. Python callers can read `RunResult.cells`,
`RunResult.denominator`, and each point's `unavailable` summary.

The metrics score the answer columns as the table writes them. Check the
table against the model's natural answers: `Z` and ` Z` can be distinct
tokens, and the answer that follows a space carries that space in its column.

Run the experiment on trained weights with an accelerator:

```bash
uv run causalab run patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --out runs/patch \
    --device cuda \
    --dtype bf16
```

| Flag | Purpose |
|---|---|
| `--device` | Choose execution placement: one device (`cpu`, `cuda`, `cuda:1`, `mps`) or a comma list (`cuda:0,cuda:1`) placing layers across the devices of one process (PyTorch engine). Recorded as `execution.device`; document identity is unchanged. |
| `--dtype` | Set `model.dtype`, which enters the digest. For workflows, set precision in each step document. |
| `--artifacts-root` | Resolve relative artifact paths. Defaults to `.`. |
| `--engine` | Select `pytorch_hooks`, `nnsight`, or `auto` (`pytorch_hooks`). |
| `--points START:STOP` | Run a half-open range of sweep points. |
| `--batch-rows N` | Bound PyTorch forward batches; recorded in execution metadata. Rounding can vary with batching. Unsupported by `nnsight`. |
| `--verbose`, `-v` | Print progress on stderr: point selection, each point's model load, cohort fits, each point's run, and the output write. Also shows Hugging Face Hub download and lock-wait messages. Changes no output file. |
| `--parallel AXES` | PyTorch engine: data, pipeline, context, tensor, and expert parallelism (`dp, pp, cp, tp, ep`; default `1`). Recorded as `execution.parallel`; document identity is unchanged. See [§6](#6-engines). |
| `--resume` | Reuse completed workflow steps after checking their identities and contents. |
| `--register-from-hf` | Fetch metadata for an unregistered model during inspection; `run` does this automatically. |


## 5. Hookpoints

A site names a component, such as `block_output` or `expert_neuron_output`.
The registry states which components each model family exposes, at which
layers, with which shape and write rules. `causalab explain` prints the
components a document uses on its model. The
[Qwen3.6-35B-A3B page](qwen36_35b_a3b.md) lists the full component table for
the model most templates use, with its architecture diagram.

### Model families

A family is a module tree. The registry detects it from the loaded model's
children, never from the config's `model_type`, and refuses a tree that no
family or several families detect. Three trees are built in
(`causalab/protocol/registry/families.py`):

| family | detected by | models |
|---|---|---|
| `llama_tree` | `model.layers` and `model.embed_tokens` | Llama, Qwen, Mistral, Gemma, the Qwen3.5-MoE hybrid |
| `gpt2_tree` | a `transformer.h` block with `ln_2` and `attn.c_proj` | GPT-2 |
| `gptj_tree` | a `transformer.h` block with `ln_1`, `attn.q_proj`, `attn.k_proj`, `attn.v_proj`, `attn.out_proj`, `mlp.fc_in`, `mlp.fc_out` and no `ln_2` | GPT-J (`EleutherAI/gpt-j-6b`) |

A component that a family declares no tap for fails with
`component_unavailable` and names the family.

GPT-J uses a parallel residual block: `ln_1` feeds the attention and the MLP,
and the block adds both outputs to its input in one step. The tree therefore
has no `block_mid` and no `mlp_input_norm`. The declared identity is
`block_output == attention_output + mlp_output + block_input`. It is exact in
fp32 when you add the terms in this order, which is the block's own order.
`mlp_input` is the `ln_1` output as the MLP receives it, so a write there
changes the MLP only. A write at `attention_input_norm` changes the attention
and the MLP. `attention_premix` and `attention_result` come from the input to
`attn.out_proj`, which has 16 heads of 256 on GPT-J 6B. `mlp_activation` and
`mlp_neuron_output` are the input to `mlp.fc_out`. The rotary embedding
rotates the first 64 of each head's 256 query and key features. The
pre-RoPE `attention_query_pre_rope` and `attention_key_pre_rope` are the
`q_proj` and `k_proj` outputs. GPT-J computes its attention pattern in its own
method and does not call the Transformers attention interface. For this
reason the tree does not serve `attention_query`, `attention_key`,
`attention_scores`, `attention_z` or `attention_probs`. Both engines serve the
GPT-J tree; `tests/neural/engines/nnsight_tracing/test_parity_gptj.py` holds
them to each other. Load the fp16 weights with `"revision": "float16"` and
`"dtype": "fp16"`. The `main` revision stores fp32 weights.

### Dense neuron sites

`mlp_activation` reads the dense MLP's activated gate. `mlp_neuron_output`
reads the complete neuron output at the input to its down projection. A gated
MLP forms this value as `act(gate) * up`. GPT-2 has one activation branch, so
both sites expose its complete activated neuron output.

### The attention interior, per family

The same component name can map to different module locations across model
families. GPT-2 fuses q, k, and v in `c_attn`; Llama has separate projections.
Qwen normalizes q and k before RoPE and packs a gate beside q.
Registry overrides map these logical components to each implementation:

<!-- generated: begin family-table -->

| component | `gpt2` | `llama` | `qwen3_5_moe_text` |
|---|---|---|---|
| `attention_query_pre_rope` | `c_attn` output `(batch, position, fused·head·feature)`, split 0 of 3 | `q_proj` output `(batch, position, head·feature)` | `q_norm` output `(batch, position, head, feature)` |
| `attention_key_pre_rope` | `c_attn` output `(batch, position, fused·head·feature)`, split 1 of 3 | `k_proj` output `(batch, position, head·feature)` | `k_norm` output `(batch, position, head, feature)` |
| `attention_value_states` | `c_attn` output `(batch, position, fused·head·feature)`, split 2 of 3 | `v_proj` output `(batch, position, head·feature)` | `v_proj` output `(batch, position, head·feature)` |
| `attention_gate` | unavailable on this family | unavailable on this family | `q_proj` output `(batch, position, head·fused·feature)`, split 1 of 2 |

<!-- generated: end family-table -->

For fused projections, the executor selects the component's slice and writes
it back into that slice. A GPT-2 key write preserves the query and value
columns. `head` selects within the component's head space, including KV heads
for GQA keys and values.

For unlisted families, execution can identify a separate projection or
normalization with an unambiguous width. Ambiguous fused layouts fail with a
named error. Known missing components, such as `attention_gate` on GPT-2 and
Llama, fail validation with `component_unavailable`.

### `delta_*` and `deltanet_*`: one name per tensor, three typed pairs

Both engines expose common DeltaNet tensors through the `delta_*` names.
Eight older `deltanet_*` names parse as aliases for the same tensors, preserving
their shapes and timing. Three pairs retain separate names because their values
occur at different stages:

| `pytorch_hooks` | `nnsight` | relation |
|---|---|---|
| `delta_query` `delta_key` | `deltanet_query` `deltanet_key` | `gva_tile`: `delta_*` is **post** GVA `repeat_interleave` (32 value heads); `deltanet_*` is **pre** (16 key heads). Exact after tiling. |
| `delta_state` | `deltanet_state` | `chunk_boundary`: per **step** vs per 64-token **chunk**; the chunk's state is the step-state at the chunk's last position |

Use `delta_state` for per-step state on the PyTorch engine. Use
`deltanet_query` and `deltanet_key` for pre-tiling values on nnsight.
`registry.BACKEND_PAIRS` records these relations; aliases cannot cross them.

### Reading state and attention is expensive

State and attention reads can require substantial memory:

- `delta_state` holds one matrix per head per step. The
  [Qwen3.6-35B-A3B page](qwen36_35b_a3b.md#state-reads) gives its size on
  that model.
- `attention_scores` / `attention_probs` have **two** position axes (query and
  key), so an integer `pos` is ambiguous and refused. Read them whole.

## 6. Engines

`--engine` is required on `run`, `validate`, `explain`, and `dry-run`.
Choose `pytorch_hooks`, `nnsight`, or `auto`, which selects `pytorch_hooks`.
The engine must support all required components and operations. A capability
shortfall fails with `[V13]` before loading weights.

The component and capability counts below are generated from the registry:

<!-- generated: begin engine-summary -->

| | `pytorch_hooks` | `nnsight` |
|---|---|---|
| how | `register_forward_hook` / pre-hook, plus global swaps for the delta kernel and the experts dispatch | one trace over an envoy tree, `.source` for fused-forward interiors |
| capabilities | `grad` `paired_forward` `full_logits` `writable_attention_probs` `pytorch_fn_local` `generate` `generation_writes` `quantized_weights` | `paired_forward` `full_logits` `writable_attention_probs` `pytorch_fn_local` `generate` |
| components | 52 of 56; unsupported: `deltanet_query`, `deltanet_key`, `deltanet_state` and `expert_permutation` | 51 of 56; unsupported: `delta_query`, `delta_key`, `delta_kv_mem`, `delta_state_update` and `delta_state` |
| serves alone | the post-tiling `delta_query` / `delta_key` and the per-step `delta_state` (the typed backend pairs, §5), training (`train` documents need `grad`), quantized weights | the fused-forward faces `deltanet_query` / `deltanet_key` / `deltanet_state` and `expert_permutation` |
| install | always | `uv sync` (dev group) or the `nnsight` extra |

<!-- generated: end engine-summary -->

Each engine runs one process per rank. PyTorch supports `--batch-rows N` to
bound forward groups; nnsight runs each group as one batch. These bounds are
recorded as execution settings. Training requires the PyTorch engine's `grad`
capability. A comma-list `--device` places a model's layers across the devices
of one process (PyTorch engine only).

Parity tests compare reads and writes across the shared vocabulary on a tiny
fixture and the real checkpoint: `test_parity_a3b_sweep.py` and
`tests/golden/test_a3b_engine_parity.py`.

### Parallel execution

`--parallel` configures the PyTorch engine's five axes: data (`dp`), pipeline
(`pp`), context (`cp`), tensor (`tp`), and expert (`ep`). Unspecified axes
default to `1`. The process count is `dp × pp × cp × max(tp, ep)`. The receipt
records the geometry under `execution.parallel`; document identity and
artifact stamps do not change. See the
[parallelism guide](model_parallelism.md).

| Axis | Use | Constraints |
|---|---|---|
| `dp=N` or `dp=N:points` | Divide campaign points into contiguous shards and join their outputs. | At least one selected point per replica. |
| `dp=N:rows` | Split each fit minibatch across replicas. | Requires `train` and at least `N` rows in every minibatch, including the remainder. |
| `pp=N` | Place contiguous layer ranges on separate ranks. | No more stages than layers; tied heads and decoding are unsupported. |
| `tp=N` | Shard attention and dense projections. | Must divide query heads; must divide the KV-head count or be a multiple of it. |
| `ep=N` | Shard routed experts. | Must divide the expert count; dense models reject `ep > 1`. |
| `cp=N` | Split padded sequence positions across ranks. | No decoding; sufficient frame and convolution-history length; hybrid families require `CAUSALAB_EXPERIMENTAL_CONTEXT=1`. |

`dry-run --parallel AXES` checks the geometry against the model's registry
entry and reports memory estimates from cached checkpoint headers. A model
needs a registered parallel plan (`gpt2` and `gptj` have none). `--engine nnsight` is
refused above a world of 1.

On one node, `run` spawns the required processes and waits for them. Each rank
uses `cuda:LOCAL_RANK` under `--device cuda`. If a launcher supplies
`WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`, each
process joins that world; its size must match the geometry. CUDA uses NCCL and
CPU uses gloo.

```bash
# One node: two data replicas over the campaign's points.
uv run causalab run scan.json \
    --engine pytorch_hooks \
    --out runs/scan \
    --device cuda \
    --parallel dp=2

# External launcher: two ranks with experts divided between them.
uv run torchrun \
    --nproc_per_node=2 \
    -m causalab.cli run patch.json \
    --engine pytorch_hooks \
    --out runs/patch \
    --device cuda \
    --dtype bf16 \
    --parallel ep=2
```

Point-parallel replicas run disjoint parts of the selection; the publishing
rank of replica 0 joins the results in campaign order and writes them once.
Row-parallel replicas run every point, combine minibatch gradients and
evaluation scores, and make the same early-stop decisions. The receipt records
the launcher (`solo`, `spawned`, or `joined`). Tensor, expert, context, and
row-parallel execution can change floating-point rounding; see the
parallelism guide §10. Writes to `router_scores` under expert parallelism are
rejected. Workflow runs execute ranks in lockstep with the joiner alone
writing the run tree; workflows reject the data axis, so shard a step with
`fan_out.over.shards` instead.

## 7. Running at scale

Use `--points` to shard a protocol sweep. External tooling assigns shards
to jobs and devices.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mcqa-patch
#SBATCH --gres=gpu:2          # ~70 GB of bf16 weights + KV/state headroom
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
set -euo pipefail

uv run causalab run patch.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --out "runs/patch" \
    --device cuda \
    --dtype bf16
```

Read the point count from `explain`. For shards of at most N points, use
`ceil(points / N)` jobs:

```bash
#SBATCH --array=0-9           # explain says 40 points -> 10 shards of 4
START=$(( SLURM_ARRAY_TASK_ID * 4 ))
uv run causalab run scan.json \
    --engine auto \
    --data-root data \
    --artifacts-root . \
    --out "runs/scan/shard_${SLURM_ARRAY_TASK_ID}" \
    --points "${START}:$(( START + 4 ))" \
    --device cuda \
    --dtype bf16
```

Merge shard outputs by coordinate and retain their point digests. Sum the
shards' cell counts to report the full campaign denominator.

An empty sweep is invalid. A document generator should report which filter
produced no values before submitting the batch:

```
refused: [V14] at sites.target.layers a sweep axis must have at least one value
```

## 8. Chaining documents: workflows

A workflow connects experiments with analysis steps and derives their
dependencies. See the [workflow specification](workflow_protocol.md).

`demos/methods/workflows/weekdays.json` scans layer and position, selects
a location, fits DAS across ranks and seeds, and evaluates the chosen fit.

```bash
uv run causalab explain demos/methods/workflows/weekdays.json \
    --engine auto \
    --artifacts-root .
# schedule  5 levels
#   level 0: locate
#   level 1: best, scan_heatmap
#   level 2: fit
#   level 3: best_fit, iia_by_k
#   level 4: apply
#   locate: intervention_protocol ../protocols/weekdays_locate_scan.json — 56 point(s), campaign digest 633cffc755706117…
#   best: script causalab.workflow.scripts.select -> values.json
#   fit: intervention_protocol ../protocols/weekdays_das_sweep.json — 9 point(s), authored digest 58040cee29c16b64…
#   best_fit: script causalab.workflow.scripts.select -> values.json
#   apply: intervention_protocol ../protocols/weekdays_das_apply.json — 1 point(s), authored digest 1d85cd221c85d257…
#   scan_heatmap: script causalab.io.plots.workflow_figures -> scan_iia.json, scan_iia.png
#   iia_by_k: script causalab.io.plots.workflow_figures -> iia_by_k.json, iia_by_k.png

uv run causalab run demos/methods/workflows/weekdays.json \
    --engine auto \
    --artifacts-root . \
    --out runs/weekdays \
    --device cuda
```

Set precision in each step's document or `set` block. Workflow commands
reject `--dtype` because steps can use different precisions.

`explain` reports the derived schedule and point count for each protocol step.
Steps at the same schedule level have independent dependencies.

`--resume` reuses completed steps after checking code identity, inputs,
and output contents.

## 9. Where to look next

| you want | read |
|---|---|
| a worked experiment, end to end | [`../demos/`](../demos/): one markdown demo per research question |
| a **fit** and the **apply** that scores it honestly | [`07_subspace`](../demos/onboarding_tutorial/07_subspace.md) (a trained rotation) · [`09_components`](../demos/onboarding_tutorial/09_components.md) (a trained mask) |
| the demo format | [`demos.md`](demos.md) |
| the normative document spec | [`intervention_protocol.md`](intervention_protocol.md) |
| chaining documents | [`workflow_protocol.md`](workflow_protocol.md) |
| the module map and layering rules | [`CODEBASE.md`](CODEBASE.md) |
| test tiers and pinned-artifact discipline | [`TESTS.md`](TESTS.md) |
| worked documents | [`../demos/methods/`](../demos/methods/README.md): one document per method, `protocols/*.json` and `workflows/` |
| the A3B hookpoints and their picture | [Qwen3.6-35B-A3B](qwen36_35b_a3b.md) |
