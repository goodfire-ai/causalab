<a id="intervention-protocol--specification-protocol_version-3"></a>
# Intervention Protocol: specification, `protocol_version` 3

The **Intervention Protocol** specifies causal intervention experiments on neural
networks. An intervention specification records the model, inputs, operations,
and outputs in JSON. A compiler resolves it into a compiled intervention for
each experiment point. A runtime implementation executes those points and saves
results. A run receipt is available on request. A research pipeline, the
sequence of questions and experiments in a study, chains such experiments;
the [workflow format](workflow_protocol.md) describes that chaining.

Use this page as the format reference. The [glossary](#11-glossary) defines the
terms used here. The [internals page](intervention_protocol_internals.md)
covers the implementation: the module map, derived properties, the canonical
form, the engine contract and the compilation stages.

## Contents

**Quick Links**
- [Write an interchange intervention ](#10-worked-examples)
- [Run an intervention spec](#9-cli-and-the-python-entry-point)
- [Look up a parameter definition](#2-section-reference)
- [Sweep across a parameter range](#3-sweeps)
- [Interpret error messages](#5-validation--load-error-checklist)
- [Implement or debug an engine](intervention_protocol_internals.md#8-engine-contract)
- [See how a spec becomes a run tree](intervention_protocol_internals.md#objects-and-functions)

The sections in order:

| Section | Content |
|---|---|
| [1. Document layout](#1-document-layout) | Top-level keys, the header, the twelve method sections, and the smallest complete document. |
| [2.1 `model`](#21-model) | Model key, revision, compute dtype, quantization, and attention implementation. |
| [2.2 `data`](#22-data) | Original and counterfactual inputs as dataset references and columns. |
| [2.2.1 `segments`](#221-segments) | Named parts of each input that positions can use as anchors. |
| [2.3 `positions`](#23-positions) | Token position forms, from a bare index to spans and alignment. |
| [2.4 `sites`](#24-sites) | Activation addresses: component, layer band, and sub-axes. |
| [2.5 `featurizers`](#25-featurizers) | Feature-space maps, their kinds, and which fields each kind accepts. |
| [2.6 `params`](#26-params-optional) | Free tensors that no featurizer owns. |
| [2.7 `reads`](#27-reads) | Value producers named by site, position, model, and input. |
| [2.8 `writes` and the `do` algebra](#28-writes-and-the-do-algebra) | Effect definitions and the operations that compute a written value. |
| [2.8.1 `code`](#281-code--user-functions-identified-by-content) | User functions for `pytorch_fn` writes, declared by content. |
| [2.9 `intervened_models`](#29-intervened_models) | Which writes are in force on which input. |
| [2.10 `aggregation`](#210-aggregation-reductions-over-a-read) | Reductions over a read, carried by the entry that consumes them; eligibility, the answer space, and match modes. |
| [2.11 `train`](#211-train) | Objective terms, trainable params, optimizer, steps, evaluation, and checkpoints. |
| [2.12 `save`](#212-save) | The output manifest, derived record kinds, and `reduce` verbs. |
| [3. Sweeps](#3-sweeps) | `sweep` wrappers, `at_once`, and the `axes` group. |
| [4. Execution semantics](#4-execution-semantics) | Forward groups, write order, and the three resolution outcomes. |
| [5. Validation](#5-validation--load-error-checklist) | The numbered rules the loader enforces. |
| [6. Derived](intervention_protocol_internals.md#6-derived--never-authored) | Properties the compiler computes and a document never states. |
| [7. Canonical form and digests](intervention_protocol_internals.md#7-canonical-form-and-digests) | Canonicalization and the digests that identify a run and its points. |
| [8. Engine contract](intervention_protocol_internals.md#8-engine-contract) | What `execute(compiled, run)` receives and returns. |
| [9. CLI and the Python entry point](#9-cli-and-the-python-entry-point) | The `causalab.protocol` API, the CLI, and the compilation stages. |
| [10. Worked examples](#10-worked-examples) | Complete documents to copy and adapt. |
| [11. Glossary](#11-glossary) | Terms and the causal abstraction correspondence. |

## Objects and functions

The modules a run passes through, from the authored document to the run tree,
are mapped in the [internals
page](intervention_protocol_internals.md#objects-and-functions).

## 1. Document layout

Each file describes one experiment. It requires `header`, `model`, `data`, and
`method`. The optional `axes` group defines related sweep values.

Use the order below for readability. The loader accepts other orders with a
warning, and canonicalization restores this order before computing a digest.

| # | key | required | content |
|---|---|---|---|
| 1 | `header` | ✓ | what this file is ; the three fields below |
| 2 | `model` | ✓ | the neural network ℒ, and how it is realized numerically (§2.1) |
| 3 | `data` | ✓ | input rows: `base` (+ `counterfactual`), each a dataset ref or inline `inputs`; a block that names no role is `base` (§2.2) |
| 4 | `method` | ✓ | the experiment: the eleven sections below (§2.2.1, §2.3–§2.12) |
| 5 | `axes` | – | **sugar**: named axes ; correlated row tuples and dependent axes ; declared once, referenced at the fields they move by `{"axis": …}` wrappers and lowered to sweep wrappers before the shape gate (sec. 3.2); in the canonical form when authored |

The header:

| # | key | required | content |
|---|---|---|---|
| 1 | `protocol_version` | ✓ | `"4"` ; a string, compared and never ordered |
| 2 | `title` | – | free text, one line: what to call this experiment |
| 3 | `description` | – | free text, the file's intent (JSON has no comments) |

`title` and `description` describe the file for readers. Canonicalization removes
them, so editing either leaves the digest unchanged. `protocol_version` remains
in the canonical form.

The method:

| # | key | required | content |
|---|---|---|---|
| 1 | `intervened_models` | ✓ | the models that run: each on one input, listing the reads taken on it and the writes in force ; a model with no writes is the un-intervened model (sec. 2.9) |
| 2 | `segments` | – | the rows' named sequence segments and their frame (sec. 2.2.1) |
| 3 | `positions` | – | named token-position specs |
| 4 | `sites` | ✓ | named activation addresses ; the complete tap inventory |
| 5 | `featurizers` | – | named feature-space maps |
| 6 | `params` | – | free/constant tensors owned by no featurizer |
| 7 | `code` | – | user functions a `pytorch_fn` write names, declared by content (sec. 2.8.1) |
| 8 | `reads` | ✓ | value producers: addresses, bound to no model (sec. 2.7) |
| 9 | `writes` | – | effect definitions (inert until listed) |
| 10 | `train` | – | the fit, declared |
| 11 | `save` | ✓ | the complete output manifest ; non-empty, last (sec. 2.12) |

The smallest complete document ; a harvest: one site, one read, one output:

```json
{
  "header": {"protocol_version": "4", "title": "Residual harvest at layer 18"},
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16"},
  "data": {
    "base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"}
  },
  "method": {
    "intervened_models": {
      "original": {"input": "base", "reads": ["acts"]}
    },
    "sites": {
      "target": {"component": "block_output", "layers": [18]}
    },
    "reads": {"acts": {"site": "target", "pos": -1}},
    "save": [{"read": "acts", "model": "original", "file_path": "acts.safetensors"}]
  }
}
```

- Names in method sections 1–8 must be unique across those sections. Reserved
  names are `base`, `counterfactual`, `counterfactual[j]`, and `all`. The
  un-intervened model is declared like any other and may be named `original`.
  Segment names belong to a separate namespace of anchors.
- References name declared entries and must resolve.
- `{"artifact": "<path>", "key": "<field>"}` loads a scalar or position from
  a saved artifact. A missing artifact causes a load error.
- Field paths start with the section: `sites.target.layers`, `model.dtype`,
  `train.seed`, or `header.title`. These paths apply to overrides, workflow
  steps, sweep coordinates, plots, and error reports. A path that includes a
  group, such as `method.sites.target.layers`, is rejected. Header fields
  cannot be swept.
- Compilation reports a digest for the full experiment and one per point
  (sec. 7). These identify saved tensors and determine whether `--resume`
  can reuse results.
- An intervention specification uses `header.protocol_version`. A workflow
  has a separate top-level `version`.
- `causalab migrate <file>...` updates version 1, 2 and 3 files to version 4.
  It also updates fenced JSON in Markdown. Version 1 used top-level sections;
  version 2 used the scalar `layer` field; version 3 bound each read to a
  model, reserved `original` for the un-intervened model, and kept
  reductions in a `metrics` section. The loader directs older files to this
  command.

## 2. Section reference

### 2.1 `model`

| field | meaning |
|---|---|
| `model.key` | model name (HF key or registry name) ; the network as a *name* |
| `model.revision` | checkpoint revision |
| `model.dtype` | the compute dtype the weights are realized in: `fp32` (default) \| `bf16` \| `fp16` |
| `model.quantization` | optional ; load-time weight quantization (below) |
| `model.attn_implementation` | optional ; `eager` \| `sdpa` \| `flash_attention_2`; the full-attention backend both engines load |

`dtype` and `quantization` affect the computed values and enter the digest.
Canonicalization supplies the default `dtype`. The CLI option `--dtype` sets
`model.dtype` and therefore changes the digest.

An explicit `attn_implementation` also enters the canonical form, forward
identity, and artifact metadata. It supports sweep and bind wrappers. When
omitted, hooks use eager attention and nnsight uses the model default.
Operations inside attention can use eager attention temporarily. See
[attention backends](attention_backends.md).

```json
"model": {
  "key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16",
  "quantization": {"scheme": "nf4", "method": "bitsandbytes", "compute_dtype": "bf16", "double_quant": true}
}
```

| quantization field | meaning |
|---|---|
| `scheme` | ✓ ; `int8` (LLM.int8() mixed-precision decomposition) \| `nf4` \| `fp4` (the two 4-bit datatypes) |
| `method` | the quantizer: `bitsandbytes` (default, the only v1 entry) |
| `compute_dtype` | dtype the dequantized matmuls run in; defaults to `model.dtype` |
| `double_quant` | 4-bit only ; quantize the quantization constants |
| `int8_threshold` | `int8` only ; the outlier threshold (default `6.0`) |

Use `nf4` or `fp4` to specify a 4-bit format. Checkpoints already quantized with
GPTQ or AWQ are identified by `model.key` and `revision`. The `quantization`
field applies quantization when loading an unquantized checkpoint and requires
the engine's `quantized_weights` capability.

### 2.2 `data`

```json
"data": {
  "base":   {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"},
  "counterfactual": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "counterfactual_inputs[0]"}
}
```

`dataset` is a local path under the data root, with an optional `#<split>`
fragment. Each row declares its `split`; an undivided table can use `"all"`.
For example, `natural_domains_arithmetic/data/weekdays#train` selects rows
whose `split` is `"train"`. The resolver supplies those rows, their columns,
and a content digest. The digest covers the selected rows, so adding another
split leaves existing runs unchanged.

Build tables before loading a specification. Use `scripts/build_task_dataset.py`
for one pool or `scripts/build_split_dataset.py` for splits with disjoint groups.
Materialize remote datasets in the same way. The runtime reads the table itself;
record the build command in the task documentation or workflow description.

Input roles are `base` for the original input and optional `counterfactual`.
A list of counterfactual roles is addressed as `counterfactual[j]`. Rows pair
by index. Each pair contains one original input and its counterfactual inputs.
`field` names the text column; `[j]` selects an entry from a list-valued column.

A `data` block that names no role is the `base` role. Role names are needed
only when a counterfactual is present, so a single-input document can write:

```json
"data": {"dataset": "causal_trace/data", "field": "input"}
```

The loader converts this to `{"base": {...}}` before the parse and before
canonicalization. Both spellings have the same canonical form and digest
(sec. 7). A block that names `counterfactual` without `base` is a P2 error.

**Inline inputs.** A role can carry its prompts instead of naming a table:

```json
"data": {"inputs": ["The Space Needle is located in"]}
```

`inputs` is a non-empty list of non-empty strings and cannot be swept. The
loader turns it into a one-column table: column `input`, one row per prompt,
every row in split `all`. The role reads `input`, so `field` is implied and
refused (P2). `draw` needs a list-valued column and is refused (P2).
`shuffle` works on an inline counterfactual as on any other. A role names
either `dataset` or `inputs`, never both.

The table is registered in the loading process under the derived ref
`inline:<digest>`, where the digest is the same content digest a file table
gets for the same rows. An inline table and a file table with equal rows are
therefore one dataset: same digest, same forward group, same receipt entry.
Reports name the table by that ref. Authoring the ref directly is a P2 error;
metrics that read answer columns from `base` need a table with those
columns, and positions that need `<field>_variables` fail on inline rows as
they fail on any table without the column.

Use one table for both roles when possible. Metrics read answer columns from
`base`. A counterfactual role can contain a subset of the original table's
columns; roles that use different datasets must have the same columns (rule 20).
Validation checks column references against `base`. It checks prompt variables
against the union of the roles' `<field>_variables` columns and same-named
column fallbacks, at every sweep point.

An optional `example_id` column supplies a non-empty, unique label for every
row. Otherwise, labels are zero-based row indices written as strings. Outputs
use the original row's label to identify its pair. Labels remain stable across
shards and shared forwards.

**Shuffled counterfactuals.** `shuffle: {seed: <int>}` permutes a counterfactual
role's rows before pairing, using `random.Random(seed).shuffle`. The order stays
fixed throughout the run. This supports the workflow's `shuffled_source`
control. Fixed points can remain in the permutation; account for them when
evaluating the control. `shuffle` accepts only an integer `seed`, excluding
booleans. It is restricted to counterfactual roles and cannot be swept. It
enters the canonical form only when supplied.

**Counterfactual sets.** `draw: {"kind": "uniform", "eval": j}` samples from
a list-valued text column. Use its bare name, such as `"counterfactual_inputs"`.
Each row must contain at least one member; counts can differ between rows.
A fit draws one member per row at each epoch. The generator uses `train.seed`
hashed with `draw`, independently of batch ordering and mask sampling. Points
with the same seed receive the same sequence of draws. Each role samples
independently.

All members are encoded once at a common padded width. A minibatch selects
the sampled members and bypasses the fit's forward cache. Within a sampled
row, every index of the text column and its `<column>_variables` sibling
selects that member. The sibling must have the same length (P2). Other columns
retain their per-row values, including metric targets. **Every member must
therefore preserve the row's answer label.** A changed label can silently
train against the wrong target.

Outside training updates, including `train.eval` and post-fit metrics, the
role uses member `eval`, which defaults to `0`. `fit_diagnostics.json` records
the kind, evaluation member, and member drawn for each row in each epoch.
With an update budget, the final partial epoch records draws for all rows,
including rows whose minibatch was skipped.

`draw` accepts kind `uniform` and a non-negative integer `eval`. It applies
only to counterfactual roles, cannot be swept, and enters canonicalization
only when supplied. Drawn fits require plain text and use eager execution;
combining them with `segments` causes P4. Applying the fitted parameters to a
fixed member can use graph capture. Use separate specifications to evaluate
several members. Per-member answer columns and draws shared across roles are
unsupported.

Task-specific values belong in dataset columns, including answer forms and
text that identifies token positions. A task can record its `string_mode` as
a constant column with value `exact` or `prefix`. Validation checks `match`
metrics against this mode (sec. 2.10). A `prefix` table with `mode: exact`
causes rule 4. The run receipt records each table's mode and status under
`scoring`, using `ok` or `unrecorded` when the column is absent.

**Coordinated edits.** The optional `edit_groups` column contains groups with
this shape:

```json
{"name": "edit", "atomic": true, "spans": {"base": [[0, 3], [5, 8]], "counterfactual": [[0, 4], [6, 9]]}}
```

Spans are character offsets into `input` and `counterfactual_inputs[0]`.
An atomic group must contain at least two constituents. Rule 27 checks bounds
and equal constituent counts during validation. Before a forward pass, it
also checks that each intervened model addresses either all constituents
or none, through writes and their operand reads. Non-atomic groups declare
locations without that requirement. Rows without `edit_groups` remain
unrecorded.

`causalab.causal.pair_validation` provides checks for changed answers, model correctness,
intended token changes, unintended edits, tokenizer stability, and location
coverage. `scripts/build_task_dataset.py --validate-pairs --tokenizer <key>
--revision <rev>` runs the checks that need only rows and a tokenizer.
Record the reported tokenizer with the build command.

When training and evaluation use different dataset references, rule 22
requires disjoint endpoints. Using the same reference explicitly permits
evaluation on the training data.

### 2.2.1 `segments`

This optional section names parts of each input so positions can use them as
anchors. Segment names have their own namespace, alongside prompt variables
and columns. For example, `{"index": -1, "scope": {"segment": "assistant_prefix"}}`
selects the last token of the generation prompt.

```json
"segments": {
  "frame": "chat",
  "system": {"column": "instructions"},
  "declare": {"list": {"column": "list_text"}}
}
```

| key | value | means |
|---|---|---|
| `frame` | one of the frame table | how each row's prompt is rendered before tokenization. **Absent is plain text** ; the row's `field` as it is, `prefix_lengths` 0 ; and there is no literal spelling of that default |
| `system` | `{"column": "c"}` | chat frame only: the row's value for column `c` is the system turn |
| `declare` | `{"<name>": <source>, …}` | segments located from the row's own columns, in any frame |

**Frames** ; closed vocabulary:

| frame | renders |
|---|---|
| `chat` | each row's prompt as the one **user** turn (plus the `system` turn when one is named) through the **tokenizer's own chat template** ; `apply_chat_template(…, tokenize=False, add_generation_prompt=True)` ; and encodes the rendering with the template's own specials (no BOS added twice). `prefix_lengths` becomes the count of the row's tokens before the user turn's first token |

**The chat frame's segments** ; closed vocabulary, in reading order. `system`
is declared only when the section names a column for it; `continuation` is the
greedy continuation of sec. 2.3's `generated` frame and is declared in every
frame (it exists wherever a decode does):

| segment | is |
|---|---|
| `system` | the system turn ; the rendered text's occurrences of the `system` column's value |
| `user` | the user turn ; the rendered text's occurrences of the prompt |
| `assistant_prefix` | the template's generation prompt ; the **difference** between the rendering with `add_generation_prompt` and the one without |
| `continuation` | the greedy continuation (sec. 2.3 `generated`). An anchor on it carries `generated` ; the decode budget lives on the position ; and the whole continuation is `{"generated": …, "all": true}`, not a `segment` span |

**Sources** for a declared segment ; closed vocabulary:

| source | locates |
|---|---|
| `column` | the row's value for the named top-level column, as the occurrences of that string in the row's (rendered) text ; resolved like a `column` position (sec. 2.3) |

Segment locations come from the rendered text and tokenizer offsets.
`pipeline.resolve_positions` computes them before loading model weights.
A missing or repeated segment gives an unavailable read or rejects a write
before a forward pass. A chat template must preserve the prompt text to locate
the user turn. A missing template causes `chat_template_missing`.

Validation checks the declared columns and rule 27. Token locations require
the tokenizer at execution. An omitted section uses plain text with zero
prefix lengths. An explicit section enters the canonical form.

### 2.3 `positions`

Named entries; a read/write `pos` is a name here, or an inline spec.

| form | resolves to |
|---|---|
| `-1` (bare int, sugar) | `{"index": -1}` |
| `"all"` (bare string, sugar) | `{"all": true}` |
| `{"index": n}` | one token per row. `n < 0` counts from the end of the sequence; `n ≥ 0` is rebased past any chat prefix |
| `{"variable": "x"}` | all tokens of prompt variable `x` ; a per-row window, ragged across rows |
| `{"column": "c"}` | all tokens of the string in column `c` of the row ; the per-row form |
| `{"span": [a, b]}` | fixed window `[a, b)` |
| `{"all": true}` | every content token of the row ; ragged across rows |
| `{"segment": "s"}` | all tokens of the declared segment `s` (sec. 2.2.1) ; a per-row window, located by the frame |
| + `"scope": {"variable": "x"}` / `{"column": "c"}` / `{"segment": "s"}` | interpret the index/span inside the anchor's span |
| + `"relative_to": {"variable": "x"}` / `{"column": "c"}` / `{"segment": "s"}` | offset from the anchor's span |
| + `"generated": {"max_new_tokens": n}` | resolve the anchor inside the row's greedy continuation instead of its prompt |

The protocol layer resolves positions against the model tokenizer and each
`PositionFrame` before loading weights. A `location_ledger` save entry records
the resulting indices (sec. 2.12).

**Spans (`SpanSpec`).** A position object carrying any key of the table below
is a *span*: a set of tokens composed from the anchors above, accepted wherever
a position is (`positions.<name>`, an inline `pos`). Exactly one selector per
span; `scope` / `relative_to` modify `indices` (and an atomic `span`); a span
addresses the prompt frame (`generated` is refused on it); members and anchors
are position objects spelled out ; no int or `"all"` sugar inside a span ; and
carry no `alignment` of their own. A union or intersection has two or more
members; `indices` is a non-empty list of distinct integers, counting like
`index` does (`n ≥ 0` from the content start, `n < 0` from the end).

| span key | shape | selects |
|---|---|---|
| `segment` | `"s"` | every token of the declared segment `s` ; as a selector, beside the anchor form in the table above |
| `indices` | `[n, …]` | a **noncontiguous set**: each `n` as `{"index": n}` would, inside `scope`'s anchor when one is given |
| `union` | `[<pos>, <pos>, …]` | every token any member selects |
| `intersection` | `[<pos>, <pos>, …]` | every token all members select |
| `before` | `<pos>` | every **real** token of the row strictly before the anchor's first token ; the chat prefix included, so `{"before": {"segment": "user"}}` *is* the prefix |
| `after` | `<pos>` | every real token strictly after the anchor's last token |
| `between` | `[<pos>, <pos>]` | every token strictly between the two anchors' runs, whichever order they occur in |
| `atomic` | `true` | the resolved set is **one address** (below). Also the one key that promotes a bare `span` / `variable` / `column` into a span |

- `atomic: true` treats the selected tokens as one address for overlap checks,
  alignment, and writing. A static atomic set must contain at least two tokens.
  Omitting `atomic` treats its members as separate locations; `atomic: false`
  is unsupported.
- Width can vary between rows. Reads retain these varying lengths. Writes
  require equal widths or an explicit `ragged` policy (rule 19). An empty
  selection causes `empty_selector`.
- At load, rule 8 can prove that static index sets with the same sign are
  disjoint. Text-based locations are treated as overlapping.
- `alignment` uses the same values as other positions. An atomic span is
  checked as one run; other spans are checked per constituent.

**The continuation frame (`generated`).** Add `generated` to a position to read
the greedy continuation. It requires one anchor and a positive
`max_new_tokens`, which can be swept. `index: -1` selects the last generated
token, `all: true` selects the whole continuation, and `span: [0, 3]` selects
its first three tokens. The run decodes to the largest budget requested for
each model and input; each read applies its own window.

Generation uses argmax and stops each row at its first EOS. The PyTorch engine
gets EOS IDs from an explicit behavioral request, then the model generation
configuration, then the tokenizer. It records terminal EOS separately and pads
slots after stopping. Other special tokens remain in the text. Windows clip
at the row's end, and an empty continuation supplies zero positions. Saved
logits materialize the requested tensor; unsaved, untransformed head metrics
use bounded projections.

`variable` selects the first occurrence of its value in generated text. A
missing value supplies zero positions and a null metric with `matched: false`.
Incremental detokenization maps each match to all tokens that produced it.
Generated positions support reads only: no write addresses them. A write
fires during the prefill and reaches the continuation through its logits and
KV cache. An intervened model that declares `writes_during_generation`
(sec. 2.9) also fires its writes at every decode step, at the token being
decoded. `train`, `column`, `scope`, and `relative_to` cannot combine with
`generated`.

At generated position `j`, `lm_head` contains the distribution **after** token
`j`. The distribution that produced token `0` is at the last prompt position,
`{"index": -1}`. Later tokens use the distribution at `j - 1`.

**Prompt locations.** `variable` first looks in the role's `<field>_variables`
column, then in a same-named column. `column` reads a top-level string column
and requires exactly one occurrence in that role's text. Use `variable` for
values that differ between the paired texts. A task can use `column` for a
shared value computed for the row.

Validation checks that variables and columns exist. Token widths are checked
after encoding. To select one token within a variable, use
`{"index": -1, "scope": {"variable": "x"}}` for its final token.

With the plain frame, `prefix_lengths` is zero. A chat frame uses the
tokenizer's template and offsets to locate the user turn. `index: 0` selects
the first user token; `index: -1` selects the last token of the generation
prompt. `{"before": {"segment": "user"}}` selects the chat prefix. Plain
inputs that already contain a rendered template must omit its leading BOS
when the tokenizer adds one; the encoder rejects a duplicate BOS.

`all: true` selects real content tokens after any chat prefix and excludes
padding. It takes no modifiers. The reserved string `"all"` always selects
this form. Reads can retain different row lengths; writes require equal
lengths or an explicit `ragged` policy.

**Alignment cardinality (`alignment`).** An optional `alignment` declares how
the original and counterfactual token runs should correspond. The shared
`alignment_of` function derives the observed value during encoding.

| `alignment` | means |
|---|---|
| `one_to_one` | one run on each input, of the same width ; a single token, or one joint span |
| `one_to_many` | one token on the base input, several on the counterfactual |
| `many_to_one` | several tokens on the base input, one on the counterfactual |
| `absent` | one input resolved to nothing: the value does not occur in that row's text (reason `alignment_missing`, sec. 2.4) |
| `ambiguous` | more than one alignment fits and none was named: the value occurs several times, or both runs are wider than one token and of unequal widths (reason `alignment_ambiguous`) |

- An `index` and an unscoped `[a, b)` window have `one_to_one` alignment.
  Text-based windows depend on tokenization. `all` and `generated` take no
  alignment declaration.
- Rule 26 checks the declared value and compatible address forms. Encoding
  rejects a declaration that differs from the observed cardinality.
- Without a declaration, missing or ambiguous text produces an unavailable
  read with its reason, row, value, and occurrence count. A write with that
  result fails before a forward pass.
- An explicit alignment enters the canonical form. Observed alignment is
  recorded only through the reason on an unavailable cell.
- Atomic spans are checked jointly; other spans are checked per constituent.

`alignment.pair_differences(tokenizer, base_prompt, counterfactual_prompt,
base_prefix=…, counterfactual_prefix=…)` compares the prompt, teacher-forced
answer prefix, and full context separately. The full context is tokenized as
one string because tokens can cross the boundary. Supply the same strings the
encoder receives, including rendered turns for chat inputs. This Python
function requires a tokenizer.

### 2.4 `sites`

```json
"target": {"component": "block_output", "layers": [18]}
```

| field | meaning |
|---|---|
| `component` | one of the vocabulary below |
| `layers` | the **band** of depth indices the site spans (where the component has one): a non-empty list of integers in strictly increasing order, `[18]` for one layer |
| `head` / `expert` / `stream` | optional sub-axes: attention head, MoE expert, **mixer stream** |

**Layer bands.** `layers` selects one or more depths for the same site. A read
captures one tensor per layer, and a write applies at each layer. The engine
expands a band into members such as `a[layers=10]`. An operand read on a band
of equal length supplies the corresponding member; other operands broadcast.
The canonical form retains the band. To save, score, or featurize its values,
declare separate reads or sweep the layer. Featurizers on band writes are
unsupported. `at_once` declares separate sites within one point (sec. 3.1).

`layers` must contain distinct integers in increasing order. A scalar index
canonicalizes to a one-element list. Empty lists, booleans, and null are
invalid. Layered components require the field; layer-less components reject
it. The retired `layer` spelling requires `causalab migrate`. Rule 4 checks
each layer against the model and its declared stream.

`head` uses the selected component's head count. Under GQA, KV components have
fewer heads than query components. `expert` requires a routed component with
expert selection in the table below. `stream` accepts `full_attention` or
`linear_attention` and must match the selected layer.

The component vocabulary follows the forward pass through the model:

`input_ids` · `embeddings` · `block_input` · `attention_input_norm` ·
`delta_qkv` · `delta_gate` · `delta_conv` · `delta_query` · `delta_key` ·
`delta_value` · `delta_beta` · `delta_decay` · `delta_kv_mem` ·
`delta_state_update` · `delta_state` · `delta_kernel_output` · `delta_premix` ·
`attention_query_pre_rope` · `attention_key_pre_rope` ·
`attention_value_states` · `attention_gate` · `attention_query` ·
`attention_key` · `attention_scores` · `attention_z` · `deltanet_query` ·
`deltanet_key` · `deltanet_state` · `attention_result` ·
`attention_output` · `attention_premix` · `attention_probs` · `block_mid` ·
`mlp_input_norm` · `mlp_input` · `router_logits` · `router_scores` ·
`expert_idx` · `expert_gate_proj` · `expert_up_proj` · `expert_activation` ·
`expert_neuron_output` ·
`expert_permutation` · `expert_output` · `routed_output` · `mlp_activation` ·
`mlp_neuron_output` ·
`shared_expert_gate_proj` · `shared_expert_up_proj` ·
`shared_expert_activation` · `shared_expert_output` · `shared_expert_gate` ·
`mlp_output` · `block_output` · `ln_final` · `lm_head`

Declare every referenced site, including `lm_head`.

**Block boundaries.** `attention_input_norm` is the input norm's output.
`block_mid` is the residual stream after the attention output is added, and
`mlp_input_norm` is the following norm's output. The block satisfies
`block_mid = block_input + attention_output` and
`block_output = block_mid + mlp_output`.

`input_ids` contains integer tokens and supports reads only. It has no feature
width and accepts no featurizer. `embeddings` contains the vectors looked up
from those IDs.

**Mixture of experts.** `router_logits` contains values for all experts.
`router_scores` contains the renormalized top-k weights; `expert_idx` holds the
corresponding integer expert IDs. `routed_output` is their combined output.
The shared expert exposes its SwiGLU interior and `shared_expert_gate`.

The routed expert components require `experts_implementation: grouped_mm`.
Other implementations compute different intermediate tensors and are rejected.
The components use token-major rows with ranked slots, each identified by
`expert_idx`:

- `expert_gate_proj` and `expert_up_proj` contain the two projection halves.
- `expert_activation` contains `act_fn(gate_e)`.
- `expert_neuron_output` contains `act_fn(gate_e) * up_e`.
- `expert_output` contains the down-projection output before routing weights.
  It satisfies `routed_output = Σ_slot expert_output * router_scores`.

`mlp_neuron_output` is the dense down-projection input, including the up branch
on gated MLPs. `shared_expert_activation` already denotes that product for the
shared expert.

An `expert: e` selector gathers only tokens and slots routed to that expert.
Rows can have different lengths, including zero when the expert receives no
token. Writes affect those rows. This view rejects `featurizer` and `dims`
because its width differs from the full ranked-slot view. The table below
lists components that support this selector. `expert_permutation` records the
serving kernel's row order and is available as a read through nnsight.

`expert_idx` has no feature space. Ranked-slot tensors support per-column
operations through `identity`, `standardize`, and `gate`. They reject basis
fits through `subspace`, `pca`, and `sae`, because a slot can name a different
expert at each token. A gate with `group: expert_neuron` follows each expert
through the routing table (sec. 2.5).

**Full attention.** `attention_query_pre_rope` and `attention_key_pre_rope`
contain queries and keys after any q/k norm and before RoPE.
`attention_value_states` is the value projection in KV-head space, captured
before the cache update. `attention_gate` is the gate half of a fused
`[q | gate]` projection on families that have one. Fused-qkv families expose
these logical slices through their registered tap definitions.

Inside the attention function, `attention_query` and `attention_key` are the
post-RoPE inputs. Keys remain in KV-head space before GQA repetition.
`attention_scores` is the softmax input. `attention_z` is the attention output
before gating and projection; `attention_premix` is the final projection's
input, after gating, in query-head space.

`attention_result` derives each head's contribution to the residual stream.
Its sum equals `attention_output` after excluding any output-projection bias.
Select a `head` to limit memory: the full read costs
`n_positions * heads * hidden` values. To intervene on a head's contribution,
write to `attention_premix` with that head selector.

`attention_probs` has shape `(batch, heads, query, key)`. It requires
`pos: "all"`, equal input lengths, and a whole-pattern `swap`. `featurizer`
and `dims` are unsupported. The model consumes the replaced probabilities
directly. `attention_scores` uses the same axes and passes through the model's
softmax after writing. A knockout can add a large negative mask at selected
entries; a uniform shift leaves softmax unchanged. Gaussian noise requires a
feature axis and is therefore rejected here.

Generated reads of `attention_probs`, `attention_scores`, and `attention_key`
are unsupported because their key axis grows during decoding. Query-shaped
components, including `attention_query` and `attention_z`, support these reads.

**Gated DeltaNet.** These components require a `linear_attention` layer:

- `delta_qkv`: the fused projection, with unequal query, key, and value widths.
  It has no head selector.
- `delta_gate`: the output gate in value-head space.
- `delta_conv`: the convolution output, with channels first.
- `delta_query`, `delta_key`, `delta_value`: kernel inputs after convolution
  and tiling to the value-head count, before the kernel's normalization.
- `delta_beta`: the sigmoid gate per head. `delta_decay` holds log-decay.
- `delta_kernel_output`: the kernel result before normalization and gating.
  Applying the model's norm with `delta_gate` yields `delta_premix`, the
  output projection's input.

The reference engine wraps the model's convolution and delta-rule kernel
calls for both prefill and decode. It rejects kernelized mixers and families
that lack the required call sites. Nnsight supports the kernel boundary in
the prompt frame; generated reads require a verified decode address.

`deltanet_query` and `deltanet_key` denote the untiled query and key tensors
available through nnsight. Their relation to `delta_query` and `delta_key` is
`gva_tile`. `deltanet_state` captures one state per prefill chunk, with a
`chunk_boundary` relation to the per-step `delta_state`. In the generated
frame, `deltanet_state` captures one state per token. These names denote
different tensors, so they cannot serve as aliases. Retired spellings for
identical tensors canonicalize through `schema.DEPRECATED_COMPONENTS`.

`delta_state` is the recurrent matrix `S_t` for each head and step. Positions
select steps and `head` selects matrix stacks; it rejects `featurizer` and
`dims`. The engine derives `delta_kv_mem` and `delta_state_update` from adjacent
states using `S_t = S_(t-1) * exp(g_t) + k̂_t ⊗ delta_t`.
Prefill reads run an additional stepwise loop while preserving the model's
forward result. This costs one kernel call per token at each selected layer.
Decode reads capture the native recurrent calls.

A `delta_state` write replaces the chunked call with the stepwise loop so the
edit reaches later states. This can change numerical results within the
tested tolerance. Its operand must cover exactly the addressed steps.
`delta_kv_mem` and `delta_state_update` support reads only; write to
`delta_state` or `delta_value` to change them through the recurrence.

**Write support.** The table below defines which mechanisms each component
accepts; sec. 2.8 defines a mechanism, the operation it names, and its
operands under "Operations and operands". `router_logits` is read-only;
change routing with `router_scores` or `expert_idx`. The latter accepts
`swap` only. Rule 4 checks these constraints
at load, and execution checks specifications that arrive without validation.

**Streams and families.** A declared stream must match the layer. Most
`attention_*` components require full attention; `attention_input_norm` and
`attention_output` also exist around other mixers. DeltaNet components require
linear attention. Validation checks the registry's `layer_types` when known;
execution checks the loaded modules.

The rows in `causalab/protocol/registry/components.py` define component
capabilities. Family adapters locate each component on a model. Run
`uv run python scripts/generate_support_tables.py` to regenerate the table
below, or add `--check` to verify it.

<!-- generated: begin availability-table -->

| component | layers | stream | engines | write policy | requires | expert face | aliases |
|---|---|---|---|---|---|---|---|
| `input_ids` | layer-less; omit `layers` | layer-less | both | read-only: token IDs are dataset values. Change the row text, or write 'embeddings' to edit their vectors | nothing | none; `expert` is invalid at load | none |
| `embeddings` | layer-less; omit `layers` | layer-less | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `block_input` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `attention_input_norm` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `delta_qkv` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_qkv` (retired under protocol version 1) |
| `delta_gate` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_gate` (retired under protocol version 1) |
| `delta_conv` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_qkv_conv` (retired under protocol version 1) |
| `delta_query` | a `layers` band | `linear_attention` layers only | `pytorch_hooks` | any mechanism | nothing | none; `expert` is invalid at load | none |
| `delta_key` | a `layers` band | `linear_attention` layers only | `pytorch_hooks` | any mechanism | nothing | none; `expert` is invalid at load | none |
| `delta_value` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_value` (retired under protocol version 1) |
| `delta_beta` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_beta` (retired under protocol version 1) |
| `delta_decay` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_decay` (retired under protocol version 1) |
| `delta_kv_mem` | a `layers` band | `linear_attention` layers only | `pytorch_hooks` | read-only: this readout is recomputed from state as S_{t-1}·exp(g_t) · k̂_t at each step. Write 'delta_state' to change memory or 'delta_value' to change what is stored | nothing | none; `expert` is invalid at load | none |
| `delta_state_update` | a `layers` band | `linear_attention` layers only | `pytorch_hooks` | read-only: state updates are exposed for reading. Write 'delta_state' to edit S_t in S_t = S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t | nothing | none; `expert` is invalid at load | none |
| `delta_state` | a `layers` band | `linear_attention` layers only | `pytorch_hooks` | any mechanism | nothing | none; `expert` is invalid at load | none |
| `delta_kernel_output` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_core_out` (retired under protocol version 1) |
| `attention_query_pre_rope` | a `layers` band | `full_attention` layers only | both | any mechanism | `split_qkv`: addressable q/k/v projections (the per-family tap table has none) | none; `expert` is invalid at load | none |
| `attention_key_pre_rope` | a `layers` band | `full_attention` layers only | both | any mechanism | `split_qkv`: addressable q/k/v projections (the per-family tap table has none) | none; `expert` is invalid at load | none |
| `attention_value_states` | a `layers` band | `full_attention` layers only | both | any mechanism | `split_qkv`: addressable q/k/v projections (the per-family tap table has none) | none; `expert` is invalid at load | none |
| `attention_gate` | a `layers` band | `full_attention` layers only | both | any mechanism | `gated_attention`: an output gate on its attention mixer (the per-family tap table declares none for this family: only Qwen3.5/3.6's q-projection emits [q \| gate] per head); `split_qkv`: addressable q/k/v projections (the per-family tap table has none) | none; `expert` is invalid at load | none |
| `attention_query` | a `layers` band | `full_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `attention_key` | a `layers` band | `full_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `attention_scores` | a `layers` band | `full_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `attention_z` | a `layers` band | `full_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `deltanet_query` | a `layers` band | `linear_attention` layers only | `nnsight` | any mechanism | nothing | none; `expert` is invalid at load | none |
| `deltanet_key` | a `layers` band | `linear_attention` layers only | `nnsight` | any mechanism | nothing | none; `expert` is invalid at load | none |
| `deltanet_state` | a `layers` band | `linear_attention` layers only | `nnsight` | any mechanism | nothing | none; `expert` is invalid at load | none |
| `attention_result` | a `layers` band | `full_attention` layers only | both | read-only: this per-head contribution is derived from the joint output projection. Write 'attention_premix' with the same 'head' to change it by the corresponding linear projection | nothing | none; `expert` is invalid at load | none |
| `delta_premix` | a `layers` band | `linear_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `deltanet_gated_out` (retired under protocol version 1) |
| `attention_output` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `attention_premix` | a `layers` band | `full_attention` layers only | both | any mechanism | nothing | none; `expert` is invalid at load | `attention_value` (retired under protocol version 1) |
| `attention_probs` | a `layers` band | `full_attention` layers only | both | `swap` only: each row must sum to 1 for the following value multiply. Write 'attention_scores' to use other mechanisms before softmax restores normalized probabilities | nothing | none; `expert` is invalid at load | none |
| `block_mid` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `mlp_input_norm` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `mlp_input` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `mlp_output` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `mlp_activation` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `mlp_neuron_output` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `router_logits` | a `layers` band | either | both | read-only: routing uses the scores and indices already computed from these logits. Write 'router_scores' to reweight selected experts or 'expert_idx' to select experts | `moe`: a sparse-MoE block (the entry declares no experts) | none; `expert` is invalid at load | none |
| `router_scores` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts) | none; `expert` is invalid at load | none |
| `expert_idx` | a `layers` band | either | both | `swap` only: expert IDs are integer labels. Use swap to replace the routing indices, or write 'router_scores' to reweight selected experts. Arithmetic on IDs can select arbitrary experts or exceed index bounds | `moe`: a sparse-MoE block (the entry declares no experts) | none; `expert` is invalid at load | none |
| `expert_gate_proj` | a `layers` band | either | both | any mechanism | `grouped_mm`: the grouped experts dispatch (the loaded model runs another experts_implementation: a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe`: a sparse-MoE block (the entry declares no experts) | `expert:` served by `pytorch_hooks` | none |
| `expert_up_proj` | a `layers` band | either | both | any mechanism | `grouped_mm`: the grouped experts dispatch (the loaded model runs another experts_implementation: a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe`: a sparse-MoE block (the entry declares no experts) | `expert:` served by `pytorch_hooks` | none |
| `expert_activation` | a `layers` band | either | both | any mechanism | `grouped_mm`: the grouped experts dispatch (the loaded model runs another experts_implementation: a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe`: a sparse-MoE block (the entry declares no experts) | `expert:` served by `pytorch_hooks` | none |
| `expert_neuron_output` | a `layers` band | either | both | any mechanism | `grouped_mm`: the grouped experts dispatch (the loaded model runs another experts_implementation: a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe`: a sparse-MoE block (the entry declares no experts) | `expert:` served by `pytorch_hooks` | none |
| `expert_permutation` | a `layers` band | either | `nnsight` | read-only: the kernel derives this ordering of token-slot rows from routing. Write 'expert_idx' to select experts or 'router_scores' to reweight them | `moe`: a sparse-MoE block (the entry declares no experts) | none; `expert` is invalid at load | none |
| `expert_output` | a `layers` band | either | both | any mechanism | `grouped_mm`: the grouped experts dispatch (the loaded model runs another experts_implementation: a different factorization whose intermediates are different tensors; load it with experts_implementation='grouped_mm', the default); `moe`: a sparse-MoE block (the entry declares no experts) | `expert:` served by `pytorch_hooks` | none |
| `routed_output` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts) | none; `expert` is invalid at load | none |
| `shared_expert_gate_proj` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts); `shared_expert`: a shared expert (the entry declares no shared-expert width) | none; `expert` is invalid at load | none |
| `shared_expert_up_proj` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts); `shared_expert`: a shared expert (the entry declares no shared-expert width) | none; `expert` is invalid at load | none |
| `shared_expert_activation` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts); `shared_expert`: a shared expert (the entry declares no shared-expert width) | none; `expert` is invalid at load | none |
| `shared_expert_output` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts); `shared_expert`: a shared expert (the entry declares no shared-expert width) | none; `expert` is invalid at load | none |
| `shared_expert_gate` | a `layers` band | either | both | any mechanism | `moe`: a sparse-MoE block (the entry declares no experts); `shared_expert`: a shared expert (the entry declares no shared-expert width) | none; `expert` is invalid at load | none |
| `block_output` | a `layers` band | either | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `ln_final` | layer-less; omit `layers` | layer-less | both | any mechanism | nothing | none; `expert` is invalid at load | none |
| `lm_head` | layer-less; omit `layers` | layer-less | both | any mechanism | nothing | none; `expert` is invalid at load | none |

<!-- generated: end availability-table -->

**The per-family tap table** is the rows' `overrides`: for each family the
mixer interior has been measured on (the registry entry's `family`, the HF
`model_type` the adapter reads ; `gpt2`, `llama`, `qwen3_5_moe_text`), the
**address** of each of the four attention-interior components ; the mixer
child whose output is tapped and how that child's native tensor packs the
component's logical value. Only those four rows carry any (the census holds
that), and `docs/running_experiments.md` §5 renders the table from them
(`registry.render_family_table`). The site resolver reads the address; a
family absent from a row is served by measurement where the mixer is
unambiguous (a bare projection, a norm after it) and refused where it is not
(a fused projection's block order). An address has these keys, closed:

| override key | meaning |
|---|---|
| `module` | the mixer child whose *output* is the tap ; a name (`q_proj`, `q_norm`, `c_attn`), never a module: the protocol layer stays torch-free |
| `packing` | how that module's native tensor packs the component's value ; one of the packings below |
| `splits` | for a fused packing: how many logical tensors share the module's output (2 for `[q \| gate]`, 3 for `[q \| k \| v]`) |
| `split` | for a fused packing: which of them this component is |

| packing | native tensor | measured on |
|---|---|---|
| `flat` | `(b, s, heads·d)` ; the module's whole output *is* the value | llama's bare `q_proj`/`k_proj`/`v_proj` (16 = 4·4) |
| `head_axis` | `(b, s, heads, d)` ; the whole output, head axis kept | qwen3.5-moe's `q_norm`/`k_norm`, which run before RoPE (`(1, 5, 8, 32)`) |
| `fused_heads` | `(b, s, heads·splits·d)` ; the splits interleaved *per head* | qwen3.5-moe's `q_proj`, `[q_h \| gate_h]` (512 = 8·2·32) |
| `fused_blocks` | `(b, s, splits·heads·d)` ; the splits as contiguous *blocks* | GPT-2's `c_attn`, `[q \| k \| v]` (96 = 3·4·8) |

`component_shape` gives the logical axes across model families. Each backend
converts between those axes and its packed layout. Writing one split preserves
the other splits in the native tensor.

The loader checks each component's `requires` against the model registry.
At execution, the site resolver checks the loaded module tree. These checks
reject components that the model lacks, such as `routed_output` on a dense
model.

| predicate | means | decided at load by | decided at run by |
|---|---|---|---|
| `moe` | the block is a sparse-MoE block (router + fused experts) | the entry's `num_experts` | the MLP has `gate` and `experts` children |
| `shared_expert` | the MoE block carries a shared expert | the entry's `shared_expert_intermediate_size` | the MLP has a `shared_expert` child |
| `grouped_mm` | the experts use the grouped kernel | the entry's `experts_implementation`, when the entry was adapted from a *loaded* model's config (a load-time knob, not a config fact ; a hand-declared entry carries none and decides nothing) | `experts_implementation == "grouped_mm"`, the last-line check |
| `split_qkv` | the mixer's q, k and v are addressable ; separate projections, or a fused projection whose row declares the logical blocks (GPT-2's `c_attn`) | the per-family tap table has met the entry's `family` (`overrides`); `None` otherwise | the row's address for the family, else measured: the mixer carries separate projections |
| `gated_attention` | the q-projection emits `[q \| gate]` per head (Qwen3.5/3.6) | the `attention_gate` row has an address for the entry's `family`; `None` for a family the table has not met | the row's address for the family, else measured: `q_proj.out_features == 2·H·d` |

Errors about a component, mechanism, or selector include `ProtocolError.reason`
from `errors.REASON_CODES`. Unavailable result cells use the same reason codes
(sec. 4.1). Callers can inspect the code to handle each condition. `Reserved`
marks a declared code with no current emitter.

| reason | emitted by | meaning |
|---|---|---|
| `unsupported_mechanism` | rule 4 (`validate`) and the executor's write policy, from the row's `writes` cell | a write names a mechanism the component's policy refuses ; a read-only component, or arithmetic on a `swap`-only one |
| `component_unavailable` | the stream check (load and run), the row's `requires` (load and run), the engines' interior refusals, `component_shape` on an all-MoE tower's `mlp_activation`, an `expert` selector on a component without that axis, and the executor's fire-count check after each forward ; a write whose module the forward never called, or called twice (sec. 4, "Fires") | the model, the layer or the engine has no such tensor ; in this forward included |
| `alignment_missing` | position resolution when a `variable` / `column` value occurs nowhere in the row's text (an `unavailable` result cell for a read, sec. 4.1; a refusal under a write), and a declared `alignment` the pair resolves as `absent` (sec. 2.3) | a cross-input operand has no alignment to pair on |
| `alignment_ambiguous` | position resolution when the value occurs several times (an `unavailable` cell for a read, a refusal under a write), and a declared `alignment` the pair resolves as `ambiguous` (sec. 2.3) | more than one alignment fits and none was named |
| `empty_selector` | the executor's `expert:` face when the router sent that expert no token at the addressed positions (an `unavailable` result cell, sec. 4.1), rule 15's entry selection when no bundle entry matches or none is selected (a load error) ; and a span (sec. 2.3) that resolves to no token on a row (a refusal, `protocol/positions/spans.py`) | a selector resolved to nothing |
| `chat_template_missing` | the chat frame's encode (`causalab/protocol/positions/framing.py`): a document declares `segments.frame: chat` and the tokenizer carries no chat template to render it with (a refusal before any forward, sec. 2.2.1) | the model's tokenizer cannot render the frame the document declares |
| `ragged_write_unsupported` | rule 19's `refuse` path (`neural/shared/executor/ragged.py`, `ragged_write_error`): the pre-forward width check and the landing path, for a write whose rows address different numbers of positions under no `ragged` policy or `refuse`; a ragged operand paired into a write under `refuse`; and, under a landing policy, a ragged operand whose row widths disagree with the write's (sec. 2.8, sec. 5 rule 19) | a ragged write, or a ragged operand paired into a write, has no aligned shape to land on |
| `overlapping_write_unproven` | Reserved; no current emitter | two writes overlap at one address and their order is not proven |

### 2.5 `featurizers`

Each featurizer defines `featurize(x) → (f, err)` and
`inverse(f, err) → x̂`. A write changes selected features, then reconstructs
the activation using the original error term and unselected coordinates.
See the [method guides](methods/README.md) for runnable examples.

Learned masks use Desiderata-Based Masking (DBM). Combining a learned mask
with Distributed Alignment Search (DAS) gives DBM-DAS. The parametrization
controls how a gate is trained and read at evaluation.

<!-- generated: begin featurizer-kind-table -->

| kind | featurize | param slots | authored fields |
|---|---|---|---|
| `identity` (default) | `(x, 0)` | none | none |
| `subspace` | `(Qᵀx, 0)` | `weight` | `k`, `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel`, `init` (on a fit), `seed` (on a fit) |
| `pca` | `(Pᵀx, 0)` | `weight` | `k` |
| `sae` | `(enc(x), x − dec(enc(x)))` | `enc`, `dec`, `b_enc`, `b_dec` | none |
| `standardize` | `((x−μ)/σ, 0)` | `mu`, `sigma` | none |
| `gate` | `(m⊙x, (1−m)⊙x)`, `m` the soft mask in training and the hard mask at eval, by `parametrization` (the table below) | `theta` | `parametrization` ∈ `sigmoid` \| `clamp` \| `hard_concrete` \| `budget` \| `boundary`, `group` ∈ `head` \| `expert_neuron` \| `site` (under any map but `boundary`), `axis` ∈ `position` (under any map but `boundary`), `init` (on a fit), `temperature` (under `hard_concrete` \| `boundary`), `stretch` (under `hard_concrete`), `dead` ∈ `freeze_after` \| `leak` (under any map but `boundary`, on a fit), `top_k` (under any map but `boundary`, with `file_path`), `k_schedule` (under `budget`, on a fit), `stop_grad_shift` (under `budget`, on a fit), `pool` (under `budget` to fit; any map but `boundary` with `file_path`) |

<!-- generated: end featurizer-kind-table -->

Widths and parameter shapes are derived from the model and site. A position
gate uses the addressed window length. Each kind declares its parameter slots
as `<featurizer>.<slot>`. All kinds accept `file_path`, `entry`, `dtype`, and
`description`, subject to their field conditions.

`fit_diagnostics.json` records gate counts and each fitted subspace's
`orthonormality_deviation` and `within_tolerance` result.

**Composition and sharing.** A list such as `["rot", "gate"]` applies stages
from left to right and retains an error term for each stage. Lists must be
non-empty and contain each name once. Widths follow the preceding stages.
Reusing a featurizer name at several sites shares its parameters and sums
their gradients. The sites must have compatible widths and group maps.
Declare the name once in `train.params` and `save`.

**Random subspaces.** A `subspace` can supply `seed` for its initial basis.
The default is `train.seed`, or zero without training. An untrained subspace
uses an orthonormal rank-k basis from `qr(randn(d, k))`, which supports a
random-subspace control. `seed` can be swept and cannot combine with
`file_path`. See `demos/methods/protocols/random_subspace_control.json`.

**Initialization.** `init` starts a `subspace` or `gate` fit and cannot
combine with `file_path` on the same featurizer.

For a subspace, `init: {"file_path": <basis>, "entry": …}` takes the first
`k` columns of a saved `(d, m)` basis, with `m ≥ k`. The model, revision,
model dtype, quantization, and site must match. The basis can have a different
rank, tensor dtype, or training dataset. Its columns must be orthonormal and
match the site's width. A selected swept entry must resolve uniquely. A
basis with one entry per site, written by a producer swept over the sites the
fit sweeps, needs no `entry`. Each point starts from the entry that its own
site coordinates select. Before any weights load, the engine refuses a point
whose selected entry records another site than the point's site, and names
that entry.

The initializer completes these columns `P` to a full basis `Q₀` with seeded
QR, then learns the first `k` columns of `Q₀ · R(A)`. The initial read is
`Pᵀx`. The bundle records `init_trained_on` and `init_components`; the basis
bytes enter the canonical form as `init.content_digest`. See
`demos/methods/protocols/das_pca_init.json`.

A gate accepts one initialization form:

- `{"fill": p}` sets its initial mask value, with `p` in `[0, 1]`.
  Sigmoid uses `θ = logit(p)` and excludes the endpoints. Clamp uses `θ = p`.
  `fill` can be swept and is recorded as `init_fill` in diagnostics. The
  default is the midpoint mask.
- `{"file_path": <gate bundle>, "entry": …}` loads the stored `theta`
  as the starting value. The parametrization, group map, and unit count must
  match. The model realization and site must match as for a subspace basis.
  A bundle whose entries record different sites needs an authored `entry`.
  The bytes enter `init.content_digest`; the output bundle records
  `init_trained_on`.
- `{"from_scores": {…}}` reads a JSON table with one row per unit.
  `file_path` names it; `unit` and `value` name columns and default to those
  strings. A list of unit columns addresses a multi-axis parameter such as
  `(expert, neuron)`. `where` filters rows by equality. The remaining rows
  must cover each unit exactly once (rule 32).

`from_scores` requires either `keep` or `scale`. `keep` starts the highest
scoring units at the active pole and the rest at the inactive pole. It also
defines an untrained pruning baseline. `scale` sets
`θ = midpoint + scale * z`, where `z` is the population z-score; clamp clips
the result to `[0, 1]`. Both values can be swept. Rule 32 checks `keep`
against the unit count. The table digest enters
`init.from_scores.content_digest`, and diagnostics record the selected units.

Position-gate scores use offsets within the addressed window. A prior
position gate's `rank` table or a `pca_by_position` spectrum can supply them;
set the score column and filters explicitly.

- **`group`** (optional, `gate` only): the unit one `theta` entry covers. A
  closed vocabulary of three values, each naming the axis of the site's declared
  shape its coordinate→group map is derived over:

  | group | one θ per | axis the site must have |
  |---|---|---|
  | `head` | attention head ; the map is `(heads, head_dim)` | `head`: a head-major component (`attention_premix`, `delta_premix`) |
  | `expert_neuron` | (expert, neuron) of the routed expert table ; the map is `(num_experts, d_expert)` | `topk`: the routed-slot axis, which `expert_activation` and `expert_neuron_output` lay out as one expert's neurons per slot |
  | `site` | the whole site ; the map is `(1, width)`, one parameter, so an MLP block or the embedding is a single unit (the node a circuit benchmark scores it as) | `feature`: any feature-space component; a site that names a `head` is one unit over that head's slice, which is still what it claims |

Without `group`, a gate has one parameter per coordinate. An explicit group
enters the canonical form and the bundle's `group` and `group_map` identity.
A loaded bundle must match both. `coordinate` is unsupported as a literal.

A head gate shares each parameter across that head's coordinates. Its
`theta` has shape `[heads]`; penalties and diagnostics count heads. The gate
must be the first stage and the site must retain the whole head axis.
Sites sharing the gate must have the same map (rule 23).
See `demos/methods/protocols/dbm_head.json` and `dbm_head_apply.json`.

An expert-neuron gate has shape `[num_experts, d_expert]` and joins each
routed slot to its parameters through `expert_idx`. It is supported on
`expert_activation` and `expert_neuron_output`. A swap joins the original and
counterfactual tensors by expert ID at the same position. If the
counterfactual did not route to an original expert, that slot retains its
original value. The operand must therefore be a routed read at the same
positions. `dims` is unsupported through this gate.

`routing_mismatch.json` records the unmatched slots for each write, layer,
and example with columns `point`, `coords`, `write`, `layer`, `example`,
`mismatched`, and `slots`. Both fitted and loaded gates produce this table.
The penalty and diagnostics count all expert-neuron units. See
`demos/methods/protocols/dbm_expert_neuron.json` and its apply specification.

- **`parametrization`** (optional, `gate`): how `theta` maps to the mask. One
  field with one meaning across kinds ; how the stored parameter maps to the
  object the fit is about ; and an enum per kind: a `subspace` names its
  rotation map above, a gate one of these:

<!-- generated: begin gate-map-table -->

| parametrization | soft mask (train) | after every optimizer step | hard mask (eval, apply) | mask penalty (`train.objective`) | `anneal` on `theta.temperature` | default start |
|---|---|---|---|---|---|---|
| `sigmoid` (absent) | `σ(θ / T)` | nothing | `θ > 0` | **`l1`** = `mean σ(θ/T)`; `l0` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `clamp` | `θ` itself | `θ ← clip(θ, 0, 1)` | `θ > ½` (`round`) | **`l1`** = `mean θ`; `l0` is **refused** (rule 4) | **refused** (rule 4): a clamp gate uses θ directly, projected into [0, 1] after each step | `θ = ½` |
| `hard_concrete` | **sampled**, once per optimizer step: `u ~ U(0,1)`, `s = σ((log u − log(1−u) + θ)/β)`, then `clip(s·(ζ−γ)+γ, 0, 1)` | nothing | `clip(σ(θ)·(ζ−γ)+γ, 0, 1) > ½`, i.e. `θ > logit((½−γ)/(ζ−γ))`: exactly `θ > 0` at the default stretch | **`l0`** = `mean σ(θ − β·log(−γ/ζ))`, the expected kept fraction of the sampled mask; `l1` is **refused** (rule 4) | legal | `θ = 0`, i.e. `m = ½` |
| `budget` | `σ(θ + c_k)` with the step's budget `k` drawn from `k_schedule` and the scalar `c_k` solved so `Σ m = k` exactly | nothing | largest `θ` values: `k_schedule.eval` during fitting, `top_k` after loading | none; the budget fixes mask size. `l1` and `l0` are invalid (rule 4) | **refused** (rule 4): a budget gate fixes sharpness through its budget and solved shift | `θ = 0` (`fill` ½) |
| `boundary` | `σ((β − i) / T)` over the coordinate index `i = 0 … width−1` of the stage's input: the rotation's columns or the PCA components; `θ ∈ [0, 1]` is the one scalar, the boundary as a fraction of the width, `β = θ · width` | `θ ← clip(θ, 0, 1)` | `i < β`: the first `⌈β⌉` coordinates | **`l1`** = `mean σ((β − i)/T)`, the kept fraction (`⌈β⌉ / width` as `T → 0`); `l0` is **refused** (rule 4) | legal | `θ = ½`, the half prefix (`fill` ½) |

<!-- generated: end gate-map-table -->

**Boundary gates.** `boundary` learns a prefix of an ordered basis and can
implement the rank search used in Boundless DAS (Wu et al., 2023). With a
learned DAS basis this is DBM-DAS. The scalar `θ` lies in `[0, 1]` and selects
coordinates `i < θ * width`. Every use must directly follow `subspace` or
`pca` (rule 4). The parameter shape is `[1]`, `temperature` can be annealed,
and `init.fill p` sets `θ = p`. Per-unit fields, including `group`, `axis`,
`dead`, `top_k`, `pool`, `rank`, and `init.from_scores`, are unsupported.
See [DAS](methods/das.md) and `demos/methods/protocols/das_boundless.json`.

**Hard training forwards.**
`parametrization: {"forward": "hard", "backward": <map>}` thresholds the
training mask at one half and uses the map's gradient through
`hard + (soft - soft.detach())`. String parametrizations use their native
training masks. The mapping form is gate-only, accepts `hard` for `forward`,
and cannot be swept. Use one specification per forward/backward pair.

Penalties and annealing use the backward map. Diagnostics record `forward`;
`decisive_fraction` measures confidence in the relaxed mask and
`hard_mask_size` reports the evaluation mask. The bundle records the forward
choice as provenance. A budget pool's members must agree on it.

A hard training forward can still differ from evaluation: hard-concrete
uses a sampled mask during training, and budget evaluation selects a fixed
top-k. A loaded gate with `top_k` also uses a different training threshold
when gradients pass through it. Sigmoid and clamp with midpoint initialization
start with an empty hard mask. Uniform budget parameters initially produce
all-or-none selection. Use `init` to choose another start.

An L1 constraint controls relaxed mass. Its hard count can jump as parameters
cross the threshold. With sigmoid, lowering temperature preserves the hard
mask while shrinking gradients on confident units and increasing them near
zero. At zero the gradient is `1 / (4T)`. Check confidence and hard counts
when evaluating a fit; `dead.leak` can retain gradients at saturated units.

**Clamp.** `clamp` trains mask values directly in `[0, 1]` and projects them
back after each update. Evaluation selects `θ > 1/2`. Its L1 gradient is
constant per unit, and temperature is unsupported. The default gate map is
`sigmoid`. Bundles record the effective map and must be loaded with the same
one. An older bundle without that stamp is read as sigmoid.

**Hard concrete.** `hard_concrete` uses the stochastic L0 relaxation of
[Louizos et al. (2018)](https://arxiv.org/abs/1712.01312). `temperature` defaults
to `2/3`; `stretch` defaults to `[-0.1, 1.1]` and must satisfy
`γ < 0 < 1 < ζ`. Temperature can be swept. Stretch is fixed for the fit and
enters the bundle identity when supplied.

The fit draws one mask per optimizer step from its seeded generator and
shares it across every use of the gate. It resamples at the next step.
The permitted penalty is `l0`, the mean probability of a nonzero gate,
`mean σ(θ - β * log(-γ/ζ))`. `l1` is rejected. Deterministic maps use `l1`
and reject `l0`.

`init.fill p` sets `θ = logit((p - γ)/(ζ - γ))`. Evaluation uses the
threshold `logit((1/2 - γ)/(ζ - γ))`, defined as exactly zero for symmetric
stretch. Diagnostics use the deterministic mask, so temperature annealing
leaves its confidence unchanged. Apply specifications must repeat a
non-default stretch. The bundle checks stretch but permits a different
temperature. Supply either an initial temperature or an anneal schedule for
it; supplying both causes rule 4.

**Budget gates.** `budget` learns a ranking with a selected mass at each step.
Its training mask is `σ(θ + c_k)`, where
bisection chooses `c_k` so the sum equals `k`. The shift carries its implicit
gradient unless `stop_grad_shift: true` makes it constant.

A fit requires `k_schedule`:

- `{"kind": "fixed", "k": n}` uses one budget.
- `{"kind": "uniform", "low": a, "high": b}` samples an integer in `[a, b]`.
- `{"kind": "log_uniform", "low": a, "high": b}` samples on a logarithmic
  scale and requires `a ≥ 1`.

The schedule uses the fit's generator. `eval` defaults to `k` for fixed
schedules and is required for sampled schedules. `k` and `eval` can be swept;
the bounds are fixed. A fitted budget gate rejects sparsity penalties and
temperature. A loaded budget gate requires `top_k` and rejects `k_schedule`.
It cannot supply a controller signal because its evaluation count is fixed.

Diagnostics record `k_schedule`, `eval_k`, `stop_grad_shift`, and the last
drawn `k`. Trajectory checkpoints include `<gate>.k`. Confidence is measured
at the last shift, or the evaluation shift before the first step.

`k_schedule.of` accepts `patched` (default) or `kept`. Patched counts units
that receive the counterfactual value; kept counts units that retain the
original value. Under `kept`, the gate complements every schedule count
against its unit count. `top_k` always counts patched units. Diagnostics
report `eval_k` in patched units and record `of`.

**Shared budgets.** Gates with the same `pool` share one sampled budget and
one shift over their concatenated parameters. The implicit gradient crosses
member boundaries. Members must all be fitted or all loaded. Fitted members
must agree on the schedule, backward map, forward choice, and
`stop_grad_shift`; loaded members agree on parametrization and `top_k`.
The count must fit the pool's total unit count.

A loaded pool selects the largest `top_k` parameters across all members.
Its `rank` rows include `pool` and `pool_rank`. Loaded gates from any
compatible per-unit map can form a pool. Fitted pools stamp `pool` and
`pool_units` into each bundle; both must match on reload. Unstamped bundles
can join a readout pool. `analysis.random_mask` rejects fitted pool bundles;
controls for a readout pool must be sized per member. A pool name cannot
be swept.

**Position gates.** `axis: "position"` learns one mask value per token in an
unanchored prompt `span: [a, b]`, with `b - a ≥ 2`. The mask acts across every
feature at each token. The span must have the same width at every use.
Variable, generated, scoped, noncontiguous, and right-anchored windows are
unsupported. `atomic: true` is accepted and preserves the same width.

The parameter shape is `[b - a]`. A shared position gate can therefore span
sites with different feature widths. Diagnostics and rank rows record `axis`;
their counts refer to positions. A position gate can supply a swap operand,
but its read cannot feed a metric. Measure the resulting model output.

A position gate can combine with feature gates. A grouped feature gate must
come first; per-coordinate gates support either order. Position gates reject
`group`, `pool`, and the `boundary` parametrization. Other initialization,
dead-unit, schedule, and top-k fields operate on position units. The saved
`axis` must match in both directions on reload. This field cannot be swept,
and omission denotes a feature gate without a literal `feature` value.

Budget initialization uses `θ = logit(p)` for `init.fill p`. Saved budget
parameters must be read with the same parametrization.

- **`dead`** (optional, `gate` only, on a gate in `train.params`): what the fit
  does about a unit whose hard mask has closed. Exactly one of two rules:

  | rule | what happens | where | eval-mode hard mask |
  |---|---|---|---|
  | `{"freeze_after": n}` | a unit hard-off (`θ ≤` the map's threshold) for `n` consecutive optimizer steps is **frozen**: its `θ` is photographed and restored after every later step, so the optimizer's momentum cannot reopen it and a pruned unit stays pruned | the post-step projection, beside `clamp`'s clip | unchanged |
  | `{"leak": ε}`, `0 < ε < 1` | the training mask's **derivative** is floored: the forward value is the map's own `m`, the backward sees `∂m/∂θ + ε` (the leaky-ReLU idiom, `m + ε·(θ − θ.detach())`) ; so a unit whose map saturated at the zero pole (`σ'(θ/T) ≈ 0`, a concrete sample clipped to 0) still receives `ε·∂L/∂m` and can come back | the training backward only | unchanged |

Choose one `dead` rule per trained gate. `freeze_after` keeps a unit closed
after the specified number of consecutive inactive steps. `leak` preserves
the forward value and adds `ε` to its derivative so an inactive unit can
recover. Clamp already has unit derivative, so the leak adds to it.

The rule cannot be swept or applied to a loaded or untrained gate.
Diagnostics record the rule, `frozen_units`, and `reawakened_units` for each
gate. It enters the specification but leaves the bundle's readout identity
unchanged.

**Loading.** `file_path` loads a fitted featurizer and checks its
`ArtifactIdentity`. Loaded featurizers cannot appear in `train.params`.
The apply specification must declare the model dtype used by the fit.
An omitted dtype means `fp32`; a `bf16` fit therefore needs
`model.dtype: "bf16"` when applied. The document CLI can set this with
`--dtype`; a workflow must set it in the specification.

**Bundle entries.** `entry: {"k": 8, "seed": 0}` selects a swept bundle
entry by coordinate names. Without `entry`, the consuming point's coordinates
select it, ignoring coordinates the producer lacks. A single-entry bundle
needs no selector. An explicit selector is used as written and must identify
one entry without help from the consuming coordinates. A missing or ambiguous
match is a load error. All parameter slots come from the selected point.

**Reading a ranking.** `top_k` on a loaded gate selects its largest `theta`
values, with ties resolved by lower index. It is a non-negative integer and
can be swept. Zero selects no units; a value above the unit count fails at
build. Counts refer to the gate's units, such as heads or expert-neuron pairs.
Boundary gates reject this field.

At the fitted threshold's `hard_mask_size`, top-k selects the same units.
`analysis.random_mask` accepts the same count for a size-matched control.
Without `top_k`, each map uses its own threshold. A budget gate requires an
explicit count. `top_k` is recorded in the canonical form, fit diagnostics,
and rank tables; it leaves the stored bundle identity unchanged.

### 2.6 `params` (optional)

Free tensors owned by no featurizer (steering vectors, a free written value):

| field | meaning |
|---|---|
| `file_path` | constant tensor, loaded |
| `entry` | which entry of that bundle (sec. 2.5), plus the reserved `slot` key |
| `shape`, `init` | trainable free tensor (must then appear in `train.params`) |

- A loaded constant is read from the bundle's `value` tensor by convention.
  A bundle *harvested from a read* is keyed by that read's name instead, so
  `{"entry": {"slot": "acts"}}` names it. `slot` is a params-only key ; a
  featurizer's slots are fixed by its kind.

### 2.7 `reads`

```json
"v_cf":   {"site": "target",  "pos": -1},
"logits": {"site": "lm_head", "pos": -1}
```

| field | meaning |
|---|---|
| `site`, `pos` | the address |
| `featurizer` | optional; value is read in feature space |
| `dims` | optional static index list into the feature axis; default = all |

- A read is an **address**, not a measurement: it names no model and no
  input. The models that take it list it in `intervened_models.<model>.reads`
  (sec. 2.9), and one read may be listed by several models ; each listing is
  its own value.
- Value = `featurize(activation at (site, pos) in the listing model)[dims]`.
- A read listed by model `M` sees the activation **with all of `M`'s writes
  applied** (upstream and at the same address). To read an un-written value,
  list the read on a model with no `writes`, or on an IM without that write.
- **Referring to a read.** A write operand, a `kl`/`js` `target`, a `save`
  entry, an objective term and an eval entry name a read **on a model**:
  `{"read": "v_cf", "model": "original"}`. A save entry, an objective term
  and an eval entry always spell `read` and `model` as two fields. The bare
  name `"v_cf"` is legal in two places only: a write operand, and a
  `kl`/`js` `target` outside a `save` entry. There it binds when exactly one
  model lists the read, and it is refused, naming the models, when several
  do (rule 5). Inside a `save` entry every read reference is in the object
  form, a `kl`/`js` target included, so what a saved table measures never
  depends on how many models list a read (sec. 2.12). The canonical form
  spells every reference in the object form.
- Reads never carry `do`.

### 2.8 `writes` and the `do` algebra

```json
"patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
```

- A write is an **inert definition**: no `model`, no `input`, no conditions.
  It executes inside every intervened_model that lists it.
- Effect at its address: `write(inverse(scatter(do(f[dims]) into f), err))` ;
  untouched dims and `err` from the pre-write value (sec. 2.5).
- `Operand` = a read name · a param name (`rot.weight`, or a `params` entry) ·
  a literal number. **Never a tensor, never a closure** ; constant vectors
  enter as `params` entries. The full grammar, and what a literal such as
  `0` does at the address, is under "Operations and operands" below.
- **`ragged`** (optional): how the write lands when its rows address
  different numbers of positions ; an `all`, `variable`, `column` or span
  window over rows that tokenize to different lengths (sec. 2.3). Spelled
  `"ragged": {"policy": "refuse" | "exact_length_buckets" | "padded_masked"}`;
  one key today, an object so a later knob has a place. It is **authored
  here and resolved by the executor** on the encoded batch, before any
  forward ; the parser checks the vocabulary and nothing else, because only
  the tokenizer can say how wide a row is (rule 19; the tokenizer enters at
  the execution stage, never in the pure verbs). Absent
  means `refuse` and **nothing is materialized**: a document that authors no
  policy canonicalizes byte for byte as before (sec. 7), and no existing
  document changes meaning. Never swept (rule 14) ; how a window lands is an
  execution strategy, not a research variable. Both landing policies work
  inside the forward the batch already runs: the same rows per forward, the
  same fire counts, the same prefix keys, on both engines; every
  per-position mechanism writes the same values under either, and only a
  `gaussian` draw ; shaped by the landed slice ; differs between a bucket's
  width and the padded width. A ragged **operand** (a read at a ragged window
  named by `do`) pairs into the write row by row under a landing policy and
  is refused when any row's widths disagree; under `refuse` it is refused as
  the write is. Under either landing policy an operand must carry each row's
  own width (only a one-position operand broadcasts): a uniform dense operand
  as wide as the widest row is refused ; `ragged_write_unsupported`, the same
  refusal under both policies ; never truncated into a narrower row.

  | policy | what lands | what the receipt records |
  |---|---|---|
  | `refuse` (the absent field) | nothing ; rule 19's refusal before any forward, reason `ragged_write_unsupported` | nothing |
  | `exact_length_buckets` | the rows grouped by width, one dense gather per width; each row at its own width | `execution.ragged["<model>/<write>"]`: the policy, every row's width, the `[width, rows]` buckets (sec. 8) |
  | `padded_masked` | one gather over the rows padded to the widest, the write computed once, and only the real slots scattered back ; padding is read as a duplicate of a real activation and never written | the same block: policy, per-row widths, buckets |

<a id="operations-and-operands"></a>
**Operations and operands.** A write's `do` is its **operation**: one JSON
object with exactly one key. The key names the **mechanism**, one of the
closed set tabulated below. The mechanism's payload holds its **operands**
(values the write reads at run time) and its **settings** (numbers or enum
strings that shape the write). `do` has no sugar of its own. It is never a
bare number, string or list, and a second key is refused
(`protocol/schema/parse.py`, `_parse_do`). The one wrapper an operand leaf
accepts is `{"sweep": …}` (sec. 3, rule 14).

An operand has four spellings:

| operand | resolves to | at run time |
|---|---|---|
| `"v_cf"` (a `reads` name) | that read's value at its address | a row-indexed tensor, sliced to the rows of the forward that consumes it (sec. 8) |
| `"rot.weight"` (a featurizer slot) | the slot's current parameter | one tensor shared by every row |
| `"steer"` (a `params` name) | the loaded or trained entry | one tensor shared by every row |
| `0`, `-1.5` (a literal number) | one constant | filled over the whole written slice with `torch.full_like`, in the slice's dtype |

Any other string is refused at load. A name declared in another section is
rule 6 (`operands are reads, params, or literal scalars`); an unknown name is
rule 4 (`not declared`). So `"0"` in quotes is a lookup that fails, and `0`
is a number. `true`, `false`, lists and objects are refused at parse (P2,
`expected a number`). Integer and float spellings are one value: `0` and
`0.0` share a canonical form and a digest (sec. 7).

A literal is a broadcast, not a tensor. It fills the feature slice after the
featurizer and `dims` have selected it, so `{"swap": 0}` zeroes every selected
feature at the address and leaves the unselected dims and the error term to
the reconstruction in sec. 2.5. `{"lerp": {"op": 0, "alpha": 0.5}}` halves
the selected features. `{"add_scaled": {"op": 0, "alpha": a}}` changes
nothing. A tensor operand must broadcast to the written slice right-aligned
(`neural/shared/mechanisms.py`, `_coerce`); a counterfactual read that
contributes a different number of positions than the write addresses is
refused there with both shapes named.

Which payload fields take an operand:

| field | accepts |
|---|---|
| `swap` (the payload itself), `add_scaled.op`, `lerp.op` | any operand spelling |
| `add_scaled.alpha`, `lerp.alpha` | any operand spelling; it must resolve to one element (a number, or a one-element tensor) |
| `affine.A`, `affine.b` | a `params` name or a featurizer slot, never a read or a literal (rule 6) |
| `gaussian.seed`, `gaussian.scale`, `clamp.lo`, `clamp.hi` | a literal number only; these are settings, not operands, and name nothing |
| `gaussian.axis` | `"tp_duplicated"` or `"tp_split"` |
| `renormalize` | `true` |
| `pytorch_fn.code` | a `code` entry name (sec. 2.8.1) |

Closed mechanism set (`do` has exactly one key):

| `do` | write | class |
|---|---|---|
| `{"swap": op}` | `f ← op` | absolute |
| `{"add_scaled": {"op": op, "alpha": a}}` | `f ← f + a·op` | additive |
| `{"lerp": {"op": op, "alpha": a}}` | `f ← (1−a)·f + a·op` | absolute |
| `{"affine": {"A": param, "b": param}}` | `f ← Af + b` | absolute |
| `{"gaussian": {"seed": s, "scale": c, "axis": "tp_duplicated" \| "tp_split"}}` | `f ← f + c·randn(s)` | additive |
| `{"renormalize": true}` | `f ← f·‖f₀‖/‖f‖` | absolute |
| `{"clamp": {"lo": a, "hi": b}}` | `f ← clip(f, a, b)` | absolute |
| `{"pytorch_fn": {"code": "…"}}` | arbitrary | absolute; names a `code` declaration (sec. 2.8.1); **local-only** ; refused at load by any non-local engine |

- Per (site, overlapping pos, model): **at most one absolute write**; any
  number of additive writes. Application order: absolute first, then additive
  deltas summed. This replaces any commutativity analysis and makes write sets
  order-free.
- `gaussian.axis` tells a tensor-parallel engine whether the draw is
  replicated or sharded across ranks; `seed` is part of the hash.
- A `swap` copies its operand. To change which rows provide it, use a
  counterfactual role's `shuffle` or `draw` (sec. 2.2). A shuffle fixes the
  pairing for the run. A draw samples from the row's counterfactual set at
  each training epoch. These controls use activation values produced by
  other inputs; the combined intervened state can differ from any ordinary
  forward pass.

- Rule 21 requires an operand read at the same depth as its destination or
  earlier, ordered by `(layer, intra-block rank)`. This restricts direct
  routing to forward paths. A receiver can be read in one intervened model
  and written at the same address in another. To use a separately obtained
  activation as a constant, save it in a bundle and load it through `params`.

<a id="281-code--user-functions-identified-by-content"></a>
#### 2.8.1 `code`: user functions identified by content

A `pytorch_fn` names a `code` entry. The declaration identifies the function
and its dependencies so they can be recorded with the experiment.

```json
"code": {
  "corrupt": {
    "locator": "rome.corruption.add_noise",
    "args": {"sigma_multiple": 3.0},
    "data_inputs": {"scale": "stats/subject_embedding_std.json"},
    "env_inputs": ["ROME_NOISE_SCALE"],
    "row_roles": [{"role": "clean", "rows": 1}, {"role": "corrupted", "rows": 10}]
  }
}
```

| field | required | content |
|---|---|---|
| `locator` | ✓ | the importable dotted path to the function |
| `args` | – | typed JSON keyword arguments, passed on every call |
| `data_inputs` | – | name → file path the function may read; each is content-digested at load |
| `env_inputs` | – | environment variables the function is allowed to read (**names only** ; a value is a property of the machine and belongs in the run receipt) |
| `row_roles` | – | what the rows of the batch it receives are, **in batch order**: a list of `{"role": …, "rows": n}` |
| `description` | – | free text |

Derived and stamped into the canonical form, never authored (sec. 6):
`source_module`, `source_sha256`, `data_input_digests`, and ; only when the
module imports a sibling outside the `causalab` package ; `closure` and
`closure_sha256`.

`source_sha256` hashes the whole defining module, including helpers and
constants. A locator in a third-party package hashes that file too.
`closure` lists the hashes of statically imported sibling modules outside
`causalab`; `closure_sha256` hashes sorted `<path> <sha256>` lines.
The manifest excludes the defining module. Lazy imports are included;
`TYPE_CHECKING` blocks, parent `__init__` execution, and dynamic imports
are excluded. Package code belongs to runtime identity, and imported
third-party versions belong in the run receipt.

The loader reads files and parses their AST without importing them.
Execution calls `fn(f, **args)`. A declaration with `row_roles` also passes
half-open row bounds as that keyword. Rule 25 checks their total against the
resolved input table. The function opens its own declared data files and
reads its declared environment variables.

Rule 24 rejects detectable undeclared reads, including literal environment
names and file paths. Reads through variables or dynamic code can escape
this static check. A dynamically created function with an unreadable
signature remains unchecked.

### 2.9 `intervened_models`

```json
"original": {"input": "counterfactual", "reads": ["v_sender", "v_a10", "v_a11"]},
"patched":  {"input": "base", "reads": ["v_receiver"], "writes": ["swap_sender", "freeze_10", "freeze_11"]},
"final":    {"input": "base", "reads": ["logits"], "writes": ["inject"]}
```

| field | meaning |
|---|---|
| `input` | **mandatory** ; `base` \| `counterfactual` \| `counterfactual[j]` |
| `reads` | **mandatory, non-empty** ; the reads taken on this model, each a `reads` name listed once; **unordered** (canonical form sorts); never swept (rule 14 ; what a forward observes is not a research variable). A model nobody reads runs a forward nobody observes and is refused at parse |
| `writes` | optional; the writes in force, **unordered** (canonical form sorts). Absent or empty, the model is **un-intervened**: the network on its input, read where it is listed; the canonical form omits an empty list |
| `writes_during_generation` | optional; `false` unless authored. `true` keeps every listed write in force through the decode steps of a continuation (sec. 2.3): each write fires once per step, at the token being decoded. All-or-nothing for the model, never swept, in the canonical form only when `true` (an authored `false` digests as the absent field). Requires the `generation_writes` capability (sec. 8) |

- The un-intervened model is declared like any other and goes by any name;
  `original` is the conventional one, and `causalab migrate` uses it for a
  protocol-3 document that read the network on one input only, `original_base`
  / `original_counterfactual` where it read it on several. Several models may
  share an input: two un-intervened models on `base` are one forward interned
  twice (sec. 4).
- **Writes during generation.** A decode step's forward carries one token per
  row, so rule 16 holds a model with `writes_during_generation` to what a step
  can honour: some read decodes the model (a flag that governs nothing may not
  be declared); every write it lists sits at `all` or `{"index": -1}` with no
  `scope` or `relative_to`, the two prompt-frame forms that mean "this token";
  no operand of those writes is a read (a read is a prompt-frame value with the
  prompt's positions); and none is `gaussian` (the draw is made once per
  forward from its seed, so every step would receive the same noise). Literal
  and `params` operands broadcast over the step. Each step is one forward for
  the fire count (sec. 4, "Fires").
- **Membership rule**: every declared write appears in ≥ 1 intervened_model.
- **Cross-model data flow has exactly one channel**: a read in model A may be
  the operand of a write in force in model B. No direct IM→IM wiring, no
  inheritance. The graph (IM → writes → operand reads → IMs) must be acyclic ;
  it is the execution schedule's skeleton.

### 2.10 `aggregation`: reductions over a read

An aggregation reduces the read the entry that carries it names — a `save`
entry, an objective term, or an eval entry (sec. 2.11, sec. 2.12) — and lives
there; there is no free-standing table of reductions. `kind` names the
reduction; the other fields supply answers as dataset columns, except where
the table specifies a read or literal token strings. For example,
`class_probs.groups` and `token_logits.tokens` define a fixed answer
vocabulary for the run.

```json
{"read": "logits", "model": "patched",
 "aggregation": {"kind": "match", "expected": "cf_answer"},
 "file_path": "iia.json"}
```

| kind | fields | what the value fields name | result per example | unit | estimand_version |
|---|---|---|---|---|---|
| `logit_diff` | `a, b` | columns | `logits[a] − logits[b]` | `logit` | `logit_diff/v1` |
| `soft_accuracy` | `a, b` | columns | `σ(logits[a] − logits[b])` ; the same margin squashed to (0, 1), so a row counts as "a beats b" with a gradient that fades once it is decided | `fraction` | `soft_accuracy/v1` |
| `token_logit` | `token` | column | `logits[token]` | `logit` | `token_logit/v1` |
| `cross_entropy` | `target` | column | CE against target | `nat` | `cross_entropy/v1` |
| `kl` | `target` | a **read** | KL between two reads' distributions ; comparable ones: same effective width, same transform, same token-position frame (rule 29) | `nat` | `kl/v1` |
| `js` | `target` (+ optional `restrict`) | `target` a **read**; `restrict` a column (a per-row **list** of answer strings) or **literal token strings** | Jensen–Shannon divergence between two reads' distributions, both restricted to the answer set and renormalised when `restrict` is given (see below) | `nat` | `js/v1` |
| `class_probs` | `groups` | **literal token strings** | summed probability per group | `fraction` | `class_probs/v1` |
| `token_logits` | `tokens` | **literal token strings** | the raw logit of every listed token (see below) | `logit` | `token_logits/v1` |
| `top_k` | `k, by` | none | the k top-ranked entries of the read (see below) | none (a structure) | `top_k/v1` |
| `match` | `expected` (+ optional `mode`) | column (of a string, or a **list** of equivalent forms) | match indicator | `fraction` (a 0/1 indicator) | `match/v1` |
| `decode` | none | none | the addressed tokens as text | none (text) | `decode/v1` |

**Compatible reads.** Metrics that resolve answer strings require a plain
`lm_head` read, with no featurizer or `dims`. This preserves vocabulary IDs.
`kl`, unrestricted `js`, and `top_k` can measure other components. Both
reads supplied to `kl` or `js` must use the same component. `top_k` ranks
the selected read's own axis.

**Jensen–Shannon divergence.**
`JS(p, q) = 1/2 KL(p || m) + 1/2 KL(q || m)`, where `m = (p + q)/2`.
It is measured in nats and lies in `[0, ln 2]`.

`restrict` compares the distributions within an answer set. Both are sliced
and renormalized with `log_softmax`, so the metric measures how probability
is divided among those answers. A string names a column containing a list
of answers per row; a list supplies one answer set for the whole run.
Each answer is tokenized as written (see **Token forms** below);
`token_form: id` is accepted only with `restrict`. Duplicate token IDs cause
an error. An empty row-specific set is excluded from the
measurement. `restrict` cannot be swept and has no default. `js` can be used
in a training objective.

**Domains.** Every kind consumes one of two things from its read, and which
one is a property of the kind:

| domain | kinds | consumes |
|---|---|---|
| `distribution` | everything above except `decode` | the read's dense value at the addressed positions ; the vocabulary projection for every kind but `top_k` |
| `ids` | `decode` | only the tokens the decode produced |

An `ids` kind therefore obliges **no** vocabulary projection anywhere (§8's
materialization requirement) ; a text probe is cheap by construction, not by a
engine's cleverness. It also only means something where tokens were
*produced*: `decode` binds to a read whose position carries `generated` (§2.3),
and a `decode` over a prompt-frame read is a load error.

<a id="estimand-identity--unit-and-estimand_version"></a>
#### Estimand identity: `unit` and `estimand_version`

Each metric record includes `unit` and `estimand_version`. The latter uses
`<estimand>/v<n>`, with a snake-case name and an integer version for the
computation. A change to the computation increments the version.

Each metric kind supplies both values. An explicit value must agree with
that kind; a mismatch or malformed identifier causes P4. These fields cannot
be swept. The following units are supported:

| unit | what it measures | produced by |
|---|---|---|
| `fraction` | a probability or proportion in [0, 1] | `class_probs` (softmax mass), `match` (a 0/1 indicator whose mean is the accuracy), `soft_accuracy` (a sigmoid of a margin) |
| `percentage_points` | the same quantity × 100 ; a **different** unit, which is the point: a fraction is never compared to percentage points | no kind; a campaign's own table, or a declared rescaling |
| `count` | a number of things | the workflow `count` estimator (workflow spec §2.6) |
| `logit` | a raw, or differenced, pre-softmax score | `logit_diff`, `token_logit`, `token_logits` |
| `nat` | information in base *e* ; `log_softmax` is a natural log | `cross_entropy`, `kl`, `js` |
| `bit` | information in base 2 | no kind; a campaign's own table |
| `dimensionless` | a ratio of two like-unit quantities that is not a proportion | no kind; a campaign's own table |

Metric identifiers are derived as `<kind>/v1` once their fields are fixed.
Workflow reductions identify the later aggregation separately (workflow
specification, sec. 2.6).

| kind or estimator | arithmetics under one name | identity |
|---|---|---|
| every metric kind above | one | derived, `<kind>/v1`; authored only as a statement of the same |
| the eight `reduction` estimators | one each, given the block (unit, weight, missing policy) | derived, `<estimator>/v1`, unless the block authors a campaign identifier the block **admits** ; `mean_of_eligible_row_ratios/v1` for a row-unit `mean` with `missing: exclude`, `ratio_of_sums/v1` for a row-unit `weighted_mean` over the denominator ; refused otherwise (workflow §5 rule 13) |

Every saved metric row includes `unit` and `estimand_version`. Explicit
declarations also enter the canonical form. Workflow counts `n` and
`n_excluded` describe rows kept by the reduction's missing-value policy.

<a id="eligibility--eligible-n_eligible-minimum_count"></a>
#### Eligibility: `eligible`, `n_eligible`, `minimum_count`

Each metric records the rows it can measure. A row is excluded when its read
has missing or ambiguous alignment, its required answer is absent, or its
continuation contains no addressed position. Exclusions carry
`alignment_missing` or `alignment_ambiguous` as appropriate.

Invalid specifications still fail: examples include multi-token answers
under `mode: exact`, ambiguous token forms, duplicate IDs, and a `top_k`
count outside the read's width.

Every metric row contains `eligible`. Excluded rows also contain a
`reason_code` and null value. The aggregate reports `n_eligible`,
`n_considered`, and exclusions by reason, and computes its mean over eligible
rows. These counts are derived outputs.

`n_eligible` counts measured rows. `save.reduce: "count"` counts rows in a
saved read. A workflow reduction counts its declared statistical units in
`n` and `n_excluded`.

**Minimum sample size.** `minimum_count` is an optional positive integer
that enters the canonical form only when supplied and cannot be swept.
Validation rejects a value above the original table's maximum eligible
count, determined from required answer columns. The actual eligible count
is recorded after the run. A workflow consumer decides whether it meets the
threshold; the run still saves its results.

**Comparisons.** Records with different declared units cannot be combined.
Records in the same unit and estimand compare as arms; differing estimands
in one unit are labeled as a version comparison. Undeclared units retain
the existing comparison behavior.

`estimand.Claim` binds a reported value to a file and selected row, with its
unit and version. `estimand.check_claim` rejects a value that disagrees with
the saved record.

<a id="top_k--one-kind-over-any-read"></a>
#### `top_k`: one kind over any read

`top_k` saves the largest entries from any read at the selected positions.
It requires an integer `k` in `[1, width]` and a ranking rule `by`:

- `value` ranks signed values.
- `abs_value` ranks magnitudes and preserves each value's sign.
- `prob` ranks softmax probabilities and requires a plain `lm_head` read.

Each result uses the following columns:

| column | meaning | emitted when |
|---|---|---|
| `indices` | index along the read's last axis (a token id on `lm_head`, a neuron on `mlp_activation`, a latent on a featurizer output) | always |
| `tokens` | that index decoded as a token string | the read is a plain `lm_head` tap |
| `values` | the **raw** read value at that index | always |
| `probs` | the softmax probability over the vocabulary | `by: "prob"` |

`values` contains raw values under every ranking rule. `probs` contains the
corresponding normalized values when requested.

<a id="token_logits--the-tasks-answer-space-saved"></a>
#### `token_logits`: the task's answer space, saved

`token_logits` saves raw logits for a declared answer vocabulary. These values
allow later hypotheses over that vocabulary to be scored without another
forward pass.

`tokens` is a non-empty list of distinct literal strings, fixed for the run.
It cannot be swept. Each string is tokenized as written, so `"X"` and `" X"`
are two answers. Every string must resolve to one token, and two strings must
not resolve to the same ID. Empty strings and multi-token answers are
rejected. The metric requires a plain `lm_head` read.

The result per example carries the same three columns `top_k` emits, with the
same fixed meanings, in the order the document listed the tokens:

| column | meaning |
|---|---|
| `indices` | the resolved token ids |
| `tokens` | the tokenizer's decoded form of each resolved ID; use it to check that each answer resolved to the row you meant |
| `values` | the **raw** logit of each id |

Over a multi-position read it follows the per-position rules below: one row per
(example, position), each carrying the three lists.

**Position-wise results.** Continuation metrics emit one row per example and
position, with `step` and `matched`. `decode` emits one string for the whole
window and uses null `step`. An empty selection still emits one null row
with `matched: false`. Prompt metrics emit one row per example.

Each metric binds to one read and its model/input pair. Use separate metrics
for separate models, then compare their saved values in analysis.

**Token forms.** An answer string is tokenized as written. No leading
space is added or removed: under a byte-level BPE, `" Seattle"` and
`"Seattle"` are two vocabulary rows, and the string names the one it spells.
The row that fixes the prompt fixes the answer's form with it, so a table
whose prompt ends in `downtown` carries ` Seattle`, and one whose prompt ends
in a trailing space carries `Seattle`. The shipped task tables store their
answer columns this way, and their `*_forms` columns list both spellings.
A tokenizer that folds the space into the piece, as the sentencepiece
families do, gives both spellings one row.

Each answer must resolve to one token. Two strings that resolve to one ID
are rejected where the metric would count that row twice (`class_probs`,
`token_logits`, restricted `js`); a `match` group is a set and may list both
spellings. The parser cannot see a collision, because only the tokenizer
knows; it refuses only a string listed twice letter for letter.

The run resolves every metric's answers with the model's tokenizer before
the weights load, over the rows each metric scores. For a `save` entry these
are the rows its read aligns on: a row whose address matches nothing, or
several times, is an excluded measurement (sec. 4.1), and its answer is never
tokenized. An objective term scores every base row, and a `train.eval` entry
every row of its held-out split. A value the tokenizer cannot score is
refused `[P2]` naming the aggregation, the table and the tokenizer. The
refusal has one line per failing field, each with the first failing value,
its table row, the count of failing values and a few of the others. A `save`
entry over a continuation read is left to the score, because which rows
address a generated step is known only after the decode. The score's
refusal counts the rows that fail. It names no row, because the score is
handed a batch of base rows and does not know their table rows. The
`tokenizer` line of `validate --tokenizer` and `dry-run --tokenizer` names
such a metric as checked when scored. `validate --tokenizer` and
`dry-run --tokenizer` run the same check without a run ([the verb
table](intervention_protocol_internals.md#the-verbs)).

One legal case gets a warning instead. A read at the last prompt token scores
a bare answer (it starts with a letter or digit) after text that ends in a
letter or digit, and both the answer and its space-prefixed form are single
tokens. After such text a model usually emits the spaced form, so the metric
likely scores a token the model does not emit there. Where the bare word
continuation is the answer, ignore the warning. The text tested is the text
the tokenizer receives: under `segments.frame: chat` it ends with the
assistant header, so a bare answer there gets no warning. A `match` list of
forms is not checked, because it may credit the bare spelling on purpose.

`token_form` is optional. Its one value, `id`, says the metric's columns
hold integer vocabulary IDs instead of strings, for `match`, `token_logit`,
`logit_diff`, and `cross_entropy`; a restricted `js` also accepts it. It
supports contextual targets such as whitespace and EOS, and it is what the
decode readouts of the sequence analysis write. Booleans, strings, floats,
and out-of-range IDs are invalid. Exact `match` also accepts a non-empty
list of IDs. `first_token`, `class_probs`, and `token_logits` reject `id`.
`kl` and unrestricted `js` and `top_k` reject the key. The key is never
materialized: an unauthored key is absent from the canonical form. The
retired values `auto`, `bare`, and `space_prefixed` are refused by name, with
the replacement stated: put the space in the string.

`causalab migrate` writes a retired `bare` or `space_prefixed` into literal
answer strings: `class_probs.groups`, `token_logits.tokens` and a `restrict`
list. It refuses a retired value on a metric whose answers are dataset
columns, because the strings are in a table it does not read. Dropping the
key there could change the scored token without an error. The refusal names
the metric, its fields and their columns, and the rewrite of each answer
string `s` in those columns, each member of a list of forms included:
`' ' + s.lstrip(' ')` for `space_prefixed`, `s.lstrip(' ')` for `bare`.
`auto` has no fixed rewrite, because it chose each form with the tokenizer.
Rewrite the table, delete the key, and migrate again. One refusal names every
metric of the document that needs this step, and a fenced example in a
Markdown page is refused the same way. A protocol-4 document that still
carries a retired value gets the same rewrite in the parser's refusal.

**Matching.** `match.expected` can contain a list of equivalent answer forms.
The task records those forms, including case variants, in the dataset.
The default `mode: exact` requires a single-token form. `mode: first_token`
credits the first token of a longer form. It requires distinct answers to
have distinct first tokens and rejects ambiguous answer sets. For full
multi-token grading, use the decoding path below.

<a id="the-tasks-scoring-and-the-match-mode--one-translation-table"></a>
#### The task's scoring and the `match` mode: one translation table

A task declares what a correct answer *is* once, in its `ScoringSpec`
(`causalab/causal/scoring.py`, `causalab/tasks/README.md`): the surface
`forms` of every declared value, which variable the graded string is a form
of, and a `string_mode` ; how a generated *string* compares to a form. The
protocol's `mode` compares an *argmax token* at one position. The two
vocabularies are related by a typed derivation, `ScoringSpec.protocol_mode`,
and this table is its census (`tests/protocol/test_vocabulary_census.py`
holds the left column to the task's `STRING_MODES`, the right to the protocol's
`MATCH_MODES`, and the map to `PROTOCOL_MODES`):

| task `string_mode` | protocol `mode` | why |
|---|---|---|
| `exact` | `exact` | the stripped string equals a form; with logits, the form is one token and the argmax is it |
| `prefix` | `first_token` | the string starts with a form; with logits at one position, a prefix is the answer's first token |

The dataset's `string_mode` records this derivation. A table declaring
`prefix` requires `first_token`; combining it with `mode: exact` fails before
a forward pass. A single-token `exact` table can use `first_token`, provided
distinct answers have distinct first tokens.

To score a full multi-token answer, use `decode` over the continuation and
apply the task's `ScoringSpec.grade`. It returns `1.0`, `0.0`, or null under
`invalid_output: unscored`. Tasks with answers beyond listed forms can
declare a `full_string_checker`. The resulting record uses unit `fraction`
and estimand `string_grade/v1`.

The per-example reading of that record is the **`grade`** vocabulary
(`causalab/causal/pair_validation.py`, `GRADES`; `tests/protocol/test_vocabulary_census.py`
holds this table to it), a one-to-one relabelling of what `ScoringSpec.grade`
returns:

| per-example `grade` | `ScoringSpec.grade` returns | meaning |
|---|---|---|
| `correct` | `1.0` | the generated string is a form of the expected value |
| `incorrect` | `0.0` | it is a form of another declared value, or names none under `invalid_output: incorrect` |
| `unscored` | `null` | it names no declared value under `invalid_output: unscored` |

`grade` describes how generated text matches the task's declared values.
`matched` records whether the addressed token position existed. An unscored
generation can therefore have a matched position. The pair-validity check
for correctness grades a `decode` result through the task's scoring rules
and records `string_grade/v1`.

#### Adding a metric kind

Use a workflow script step for analysis of saved results. To add a metric
that executes inside an intervention run:

1. Add its name to `MetricKind` in `causalab/protocol/schema/types.py`.
   `METRIC_KINDS` derives from this literal.
2. Add its required fields to `METRIC_FIELDS` (the input read is the carrying
   entry's `read`, never a field of the aggregation). Add fields whose
   values are not dataset column names to `NON_COLUMN_METRIC_FIELDS`.
3. Set `METRIC_DOMAINS`. The `distribution` domain requires a vocabulary
   projection; `ids` uses token IDs directly.
4. Add optional fields and defaults to `OPTIONAL_METRIC_FIELDS` and
   `METRIC_FIELD_DEFAULTS`. Canonicalization fills declared defaults.
   Fields without defaults, such as `js.restrict`, remain absent when omitted.
   Add the kind to `READ_TARGET_METRIC_KINDS` if `target` names a read,
   `TOKEN_COLUMN_METRIC_KINDS` if a string must resolve to a token ID, or
   `WHOLE_WINDOW_METRIC_KINDS` if it returns one value for the whole window.
5. Resolve its answers in `metric_token_ids` (`causalab/protocol/answers.py`),
   which the run calls before the weights load and the score calls again,
   and implement the kind in `causalab/neural/shared/metrics.py`.
6. Document its fields and accepted reads in the tables above.
7. Test its value and any conditions that cause rejection.
8. Set `METRIC_UNITS` in `causalab/protocol/estimand.py` to a `UNITS` member,
   or `None` for a non-scalar result. Update the table's `unit` column.
   The estimand identifier derives from this entry.

`test_vocabulary_census.py` checks the field tables against the schema.
`test_estimand.py` checks units and identifiers.

### 2.11 `train`

| field | meaning |
|---|---|
| `objective` | the weighted terms, in one of two spellings: positional `[[weight, term], …]`, or named `{name: {"weight": w, …term}}` where the term is an aggregation over a bound read ; `{"read": r, "model": m, "aggregation": {…}}` (sec. 2.10) ; or a regularizer. A regularizer is `{"l1": names}` \| `{"l2": names}` \| `{"l0": names}`: one featurizer (all its params), one dotted slot, or a non-empty list of distinct featurizers penalized **together**, with an optional `"reduce": "mean"` (the default, unspelled) \| `"sum"` over the concatenated per-unit quantities, and an optional `"costs"`: `{<target>: c}` ; a finite positive multiplier on that target's quantities before the concatenation (an unlisted target costs 1; the keys are the term's own targets, rule 4) ; or the word `"parameter_count"`, which divides each target's quantities by its own element count. A **named** `l1` / `l0` term may carry `"constraint": {"target": t, "dual": {"lr": η, "init"?: [λ₁, λ₂]}}` **instead of** `weight` ; Edge Pruning's Lagrangian target density (below); `weight` beside it, the positional form, `l2`, a metric term, `reduce: sum` and `costs: "parameter_count"` are refused (a `costs` table composes: the target is held on the cost-weighted density). Every featurizer named must be in `params`; `l0` names `hard_concrete` gates only, and `l1` on a `hard_concrete` gate is refused (rule 4) |
| `params` | what is optimized: featurizer names (all slots) or dotted slots; the **only** trainability declaration |
| `optimizer` | `{name, lr, …}` ; lr/schedule/clip live here. `schedule` is `constant` (the default) or `linear_warmup_decay` ; HF's `get_linear_schedule_with_warmup`: lr climbs from 0 over the first `warmup_frac` of the updates (0.1 unless authored; `warmup_frac` is legal only with this schedule and enters the canonical form only when authored) and decays linearly to 0 at the last update, pyvene's sigmoid-mask (DBM) recipe; refused beside `phases`, which rewrite `lr` themselves (rule 4). `lr` and `weight_decay` are one number for every trained parameter, **or a mapping keyed by the entries of `params`** ; `{"lr": {"rot": 0.001, "gate": 0.1}}` ; naming every entry exactly once (a key `params` does not train, or an entry left without a value, is refused at load): a rotation and a gate stepping at their own rates inside one fit, each entry its own optimizer parameter group |
| `steps` | `{"epochs": n}` or `{"updates": n}` |
| `batch` | `{"pairs": n}` ; counts base+counterfactual **pairs**, not rows |
| `anneal` | dotted-path **open-loop** schedules: `{<target>: [start, end, frac]}` or `{<target>: {"from", "to", "frac", "shape": "linear" \| "geometric"}}` ; the target is a trained featurizer's `<name>.<slot>.<hyperparameter>` (`gate.theta.temperature`) **or a named objective term's weight**, `train.objective.<name>.weight`, the address a `control` and a sweep use; the value walks from `from` to `to` over the first `frac` of the run and holds. `shape` defaults to `linear`; `geometric` multiplies by a constant per step (continuous sparsification's `T ← T·r`), so its endpoints share a sign and neither is zero. The list is the canonical spelling of a linear schedule (sec. 7) |
| `control` | **closed-loop** schedules: `{<target>: {"kind": "pid", "signal": {"hard_mask_size": <gate> \| [<gate>, …]}, "setpoint": {"ramp": [start, end, frac]}, "gains": {kp, ki, kd?}, space?, bounds?, d_clip?}}` ; the target is a named term's weight (`train.objective.<name>.weight`) or an anneal-style dotted hyperparameter, and its authored value is the controller's start; a list of gates is one signal, their kept counts summed (see below) |
| `phases` | consecutive step windows narrowing the fit: `[{"until": {"frac": f} \| {"updates": n}, "params": [...], "optimizer"?: {lr, weight_decay}, "anneal"?: {...}, "freeze_masks"?: [<gate>, …]}, …]`. Inside a phase only its `params` (a non-empty subset of `train.params`, by entry) receive gradients ; the rest are frozen, their optimizer groups kept so a later phase resumes; its `optimizer` overrides `lr` / `weight_decay` for its own params; its `anneal` runs over the *phase's* steps; `freeze_masks` pins each named gate's **hard** mask at the phase's start for every forward inside it (rule 4: gates only, none the phase trains). `until` is one unit for every phase, strictly increasing, the last `frac` `1.0` (an `updates` last phase must equal the run's update count ; checked by the loop). Absent = the one-phase fit, and a document without it keeps its digest |
| `precision` | `{feature, loss}` dtypes ; the *model's* dtype is `model.dtype` (§2.1), one home per fact. An engine whose loop cannot execute the declared precision (no `train_loss_precision` capability, sec. 8) refuses the document at load, rule 30 ; never digests one precision and runs another |
| `eval` | `{every, split, aggregations}` ; `aggregations` is `{label: {"read", "model", "aggregation"}}`, each evaluated on the split and recorded under its label |
| `early_stop` | `{on, patience, mode}` ; `on` names an `eval.aggregations` label |
| `checkpoint` | transient training state (resume); the final artifact is the `save` entry |
| `seed` | init + data order |

The training loop minimizes `Σ weight * term`. Use a negative weight to
maximize a margin or soft accuracy. A positive weight minimizes a loss such
as cross-entropy, KL, or JS. Training requires engine gradient support, and
every trained featurizer must be saved. An objective or eval aggregation is
recorded by its label ; the term's name (positional terms are numbered) or
the eval key ; in the loss trajectory and the eval records. It forces no
saved table. To table it over the run's rows, add a `save` entry that names
it, `{"train": "<name>", "file_path": "….json"}` (sec. 2.12).

**Naming a term to save it.** A `save` entry names an objective term by its
name, never by its position: a new term in front would silently change what
an index names. So a fit that saves one of its objective terms writes the
objective in the named form. `causalab migrate` gives a protocol-3 term the
name of the metric it consumed, and a regularizer term the name of its kind
(`{"ce": {…}, "l1": {"weight": 0.01, "l1": "gate"}}`).

**Evaluation.** `eval` runs on its dataset split with hard masks and without
gradients. `every` is counted in epochs unless the engine declares
`train_eval_updates`. The split is encoded once per point. Fixed groups can
be reused between evaluations. Distinct training and evaluation references
must have disjoint endpoints (rule 22). Using the same reference explicitly
evaluates on the training data.

Compatible points can train in one cohort while retaining independent seeds,
objectives, schedules, and stopping decisions. Different batch shapes can
produce rounding differences from separate fits.

**Regularizers.** A list of featurizers forms one penalty over their
concatenated values. Gate penalties act on mask units; other `l1` and `l2`
penalties act on `|p|` and `p²`. `reduce: mean` is the default and divides by
the total element count. `reduce: sum` gives each unit the same cost regardless
of width.

`costs` can supply a positive multiplier per target; omitted targets cost
one. Every named target must belong to the term. `costs: "parameter_count"`
divides each target's contribution by its own element count. With
`reduce: sum`, this sums per-featurizer means. With `mean`, the result is
also divided by the total count. Costs are fixed; sweep the term weight.

Hard-concrete gates use `l0`, the expected fraction of nonzero units.
Deterministic gates use `l1`. Budget gates reject both because their
schedule sets the mask mass. `l0` requires a hard-concrete gate. Target lists
must be non-empty, contain distinct trained names, and are sorted during
canonicalization. A one-element list canonicalizes to its name.

**Annealing.** `anneal` targets a trained featurizer hyperparameter or a named
term's `train.objective.<name>.weight`. Its initial value replaces the
declared scalar before the first update. With
`u = min(1, step / (frac * steps))`, `linear` computes
`from + (to - from) * u`; `geometric` computes `from * (to/from)^u`.
Geometric endpoints must be nonzero with the same sign. A linear mapping
canonicalizes to its list form. Diagnostics record endpoints, shape, and
final value; trajectories record the weight used at each step.

**Density constraints.** A `constraint` on a gate penalty adds
`λ₁(s - t) + λ₂(s - t)²` to the loss, where `s` is its relaxed density
and `t` the target. Dual ascent updates the multipliers at `dual.lr` from
`dual.init`, default `[0, 0]`. The first multiplier can be negative; the
second must start non-negative. This is an equality objective and can pull
density upward after an undershoot.

A constrained term has no weight and must target gates. It cannot be
annealed or controlled. Cost tables change the density being constrained;
`costs: "parameter_count"` is rejected. Constraints cannot be swept. Use
separate specifications or `--set` overrides to compare targets.

The constrained density uses the relaxed mask, so annealing can change it
while `theta` stays fixed. A target relaxed mean can also correspond to a
different hard count. Check the saved mask's count alongside the target and
dual values. Confidence alone does not establish that the hard mask reaches
the target. An unreachable target can make the quadratic multiplier grow
throughout training.

Duals use a separate maximizing optimizer group, without weight decay or
momentum, and follow the optimizer's arithmetic. The learning-rate schedule
and phases leave their ascent active. Include constrained gates in each
phase when continued ascent against a frozen density would be unwanted.

Trajectories record `term.<name>`, `lambda1.<name>`, and `lambda2.<name>`.
The recorded duals are the values used by the update, before its ascent.
Diagnostics record initial and final duals and the last update's density
under `constraints`. Early stopping can save an earlier mask, so that final
density can differ from the saved mask's density. Constrained points and
cohorts containing them use eager execution.

**Feedback control.** `control` adjusts a named term weight or trained
featurizer hyperparameter using a `pid` controller. Its signal is a trained
gate's `hard_mask_size` or `hard_mask_fraction`, optionally summed across a
list of gates. The fraction divides by their total unit count. The target's
declared value is its initial value. A path can have either control or
annealing.

`setpoint.ramp` uses the anneal schedule shape in the signal's units. With
`e_t = signal_t - setpoint_t`, the update is:

```
u_t   = kp * (e_t - e_(t-1)) + ki * e_t + kd * Δ(e_t - e_(t-1))
log w = clip(log w + u_t, log bounds)     # space: log
w     = clip(w + u_t, bounds)            # space: linear
```

The derivative contribution is clipped at `±d_clip`. Gains can be swept;
`kind`, `signal`, and `setpoint` are fixed. Defaults are `kd: 0.0`,
`space: log`, `bounds: [1e-8, 1e8]`, and `d_clip: 5.0`. Canonicalization
records them. Diagnostics record each target's start, end, last signal,
setpoint, and update count. Trajectory checkpoints record per-step values.

**Phases.** Each `phases` entry covers updates from the preceding boundary
to its `until`. Only its listed `params` receive updates. Other parameter
groups retain their optimizer state with learning rate and weight decay set
to zero. A phase can override these settings for its active parameters and
apply phase-local anneal schedules to active hyperparameters or named weights.
Those schedules must use paths untouched by global annealing or control.

`freeze_masks` captures the named gates' hard masks at the start of the phase
and uses them throughout it. The phase must leave those gates untrained.
This supports DBM-DAS fits that train a rotation against a fixed hard mask.
Fractional boundaries use `round(frac * updates)`, with the final phase
covering the remainder. Empty phases or incomplete update partitions fail
before training. Checkpoints record the phase index; diagnostics record
each phase's bounds, parameters, and frozen masks.

Only named objective weights can be swept. Their paths are
`train.objective.<name>.weight`, and term names are local to `objective`.

### 2.12 `save`

`save` is required and must contain at least one entry. It lists every value
and derived record that the run writes.

| saved | entry |
|---|---|
| a read, as a tensor | `{"read": name, "model": …, "file_path": "….safetensors"}` |
| an aggregation over a read, as a table | `{"read": name, "model": …, "aggregation": {…}, "file_path": "….json"}` (sec. 2.10) |
| a training metric, as a table | `{"train": name, "file_path": "….json"}` (below) |
| trained featurizer | `{"value": name, "site": …, "file_path": …}` |
| derived record | `{"kind": …, "file_path": …}` |

The `model` must list the `read` (rule 5); a featurizer's `site` must be
where it is applied. A read is saved as a tensor at most once per
`(read, model)`, and no two entries restate one `(read, model, aggregation)`
(rule 10). A table's **label** ; its `metric` column, its key in the point
summary, the name on its `metric` event ; is the entry's file stem
(`iia.json` → `iia`). A derived record uses one of these kinds:

| kind | writes | requirements |
|---|---|---|
| `location_ledger` | A JSON array of resolved token locations. Each row contains the example, edit group, constituent, side, token index, token ID, decoded token, point digest, and `coords` (sec. 6). | Opt-in; at most one entry per document (rule 10). A run that loads fitted parameters records the locations selected on its own inputs. |
| `trajectory` | One `.safetensors` bundle of trained featurizer slots at checkpoints. Keys include the featurizer, step, and sweep coordinates, such as `theta[l1=0.1,featurizer=gate,step=40]`. Each entry carries `ArtifactIdentity`, `step`, `epoch`, `loss`, objective values (`term.<name>`), live weights (`weight.<name>`), each gate's `hard_mask_size` and `decisive_fraction`, and controller values (`control.<target>`). Constraint terms record density and `lambda1.<name>` / `lambda2.<name>` without a weight. | Requires `train`; at most one entry. `every` accepts `{"count": n}` for evenly spaced checkpoints including the last update, or `{"updates": n}` / `{"epochs": n}`. Checkpoints load through `file_path` or gate `init` with the usual identity checks. Select `entry: {"step": 40}`; when several featurizers match, also name `"featurizer": "gate_3"` (rule 15). Scalar traces are readable from the header. |
| `rank` | A JSON row per unit in every trained or loaded gate. Fields are `featurizer`, `unit`, `theta`, descending `rank` (`0` first, lower-index tie break), `hard`, `parametrization`, `axis`, `top_k`, `pool`, `pool_rank`, point digest, and `coords`. `unit` indexes flattened `theta`, so its meaning follows the gate's group and axis. `axis` is null for a feature gate, `top_k` is null under the map's threshold, and pool fields are null outside a pool. | Requires a gate; at most one entry (rule 10). A fitted mask and a `top_k` replay of its bundle retain the same rank column, so their selected units can be compared. |

**Training metrics.** `{"train": name, "file_path": …}` names a named
objective term or an `eval.aggregations` label (sec. 2.11). It means exactly
the entry `{"read", "model", "aggregation", "file_path"}` copied from that
term, so the loss or the eval and the saved table share one definition and
cannot drift apart. The canonical form writes the copy, so a reference and
its inline twin have one digest.

- `name` resolves to exactly one term: a named objective term that reduces a
  read, or an eval label. A regularizer or `constraint` term reduces no read,
  and a name that is both a term and a label, or neither, is refused, naming
  the candidates (rule 4).
- The entry takes no `read`, `model`, `aggregation` or `reduce` of its own.
- The label is the file stem, as for every entry: `{"train": "ld", "file_path":
  "iia.json"}` writes the table `iia`.
- The entry shares the **definition**, not the values. The eval computes `iia`
  on its split during training; the save computes `iia.json` over the run's
  rows, like any save table.
- A term whose aggregation carries a sweep cannot be named (rule 14). The
  reference would move with the term point by point, while the inline copy
  would be a second axis. Save the aggregation inline, and share the swept
  value through a named axis (sec. 3.2).
- A copied `kl`/`js` target must be in the object form on the term, because
  the save holds no bare read name (sec. 2.7).

`file_path` is relative to the run's output directory. Dense numeric values
use `.safetensors`. Metric tables use `.json`, with one file per metric and
one object per row:

```text
{example_id, metric, value, [step, matched], …coords, unit,
 estimand_version, eligible, [reason_code]}
```

`example_id` is the original row's label (sec. 2.2). The metric, optional step,
and full sweep coordinates identify the result. Coordinates join a row to
`points[].coords` in the run receipt. Metric rows carry no point digest.

A sweep keeps the file path fixed and records coordinates as table columns
or tensor entry keys, such as `weight[k=8,seed=0]`. The safetensors header
has an `entries` table (sec. 8). Reads that select no rows still write an
entry with `unavailable` fields (sec. 4.1).

**Read reductions.** Optional `reduce` collapses a gathered read from
`(…, width)` to `(width,)` before saving. Accumulation uses fp32 under every
model dtype. The output can serve as a broadcast write operand; for example,
mean ablation uses a read saved with `reduce: mean` as a `params` constant.
Aggregations and fitted featurizers reject `reduce`. For analysis of files already
saved, use the workflow's `reduction` contract (workflow sec. 2.6).

| verb | value per column | details |
|---|---|---|
| `mean` | arithmetic mean of the rows | Used for mean ablation. |
| `sum` | sum of the rows | Combine with `count` to compute means across shards with different row counts. |
| `std` | sample standard deviation (`n−1`) | A single row gives `NaN`. |
| `median` | median without interpolation | Even row counts use the lower middle value. |
| `count` | row count broadcast to `width` | Records the denominator for each reduced value. |

Over zero rows, `mean`, `std`, and `median` return `NaN`; `sum` and `count`
return `0`. The entry is marked unavailable.

Every trained featurizer must have a save entry. Objective and eval
aggregations need none (sec. 2.11). Untrained and `file_path`-loaded
featurizers, writes, and intervened models cannot be saved. The run always
records the loss trajectory.

#### Adding a `reduce` verb

Add the name to `SAVE_REDUCTIONS` in `causalab/protocol/schema/parse.py` and
implement it in `_reduce_rows` in `causalab/neural/shared/results.py`. Return
a `(width,)` fp32 tensor. Document the estimator, interpolation, and tie rules
in the table above, then test the saved shape and value.
`test_vocabulary_census.py` compares the table with the declared verbs.

## 3. Sweeps

Wrap a named field in `{"sweep": [v1, v2, …]}` or
`{"sweep": {"range": [start, stop, step]}}`. This works for scalar and list
fields. Each wrapped path defines one axis. References to the same name move
together; separate axes form a cross product. Use the `axes` group for
correlated values (sec. 3.2).

Keep paired inputs in one table when possible. Sweeping `data.base.dataset`
and `data.counterfactual.dataset` independently pairs every selected table
with every other one. Rows still pair by index, so equal row counts can
hide incorrect pairings. For separate files, use a named `rows` axis or one
specification per pair.

A regularizer shared across gates has one weight at
`train.objective.<name>.weight`, so sweeping it moves all those penalties
together. Coordinates label derived names, such as `rot[k=8]`, and results.
The engine enumerates points deterministically and signs each one. The
document digest identifies the full experiment.

**Shared forwards.** The planner groups forwards by canonical model settings,
active writes and their dependencies, input role, selected rows, text field,
and segments. Equal keys permit reuse. Keys use row content digests, so
different file names for the same table can share work. Dtype and
quantization remain part of the key.

Read taps are collected across points that share a forward. The engine can
capture their union once, then gather and featurize values for each point.
It retains captures until the last consumer and reports executed groups in
`RunResult.forwards`. Zero means that this count was unmeasured. Reuse is
optional for correctness.

During a fit, groups unaffected by trained parameters can be cached per row
slice and reused across updates and evaluation. These inner passes are
excluded from `RunResult.forwards`. Drawn counterfactual minibatches bypass
this cache (sec. 2.2).

<a id="31-at_once--one-axis-inside-one-point"></a>
### 3.1 `at_once`

`at_once` expands entries that act together within one point. It supports one
field per entry in `positions`, `sites`, `featurizers`, `params`, `reads`, or
`writes`, using the same list and range forms as `sweep`.

Members receive names such as `a[layers=10]`, or names from a template such as
`"a{layers}"`. References expand with the family and connect corresponding
members. A bare family name in an intervened model selects all its writes;
a window selects members with an `at_once` wrapper over the family's field.
Every requested value must exist.

For `layers`, scalar members create separate one-layer sites. List members
create separate bands. Each family is limited to 1024 members.

```json
"sites":  {"a": {"component": "attention_output", "layers": {"at_once": {"range": [10, 20]}}, "names": "a{layers}"}},
"reads":  {"v": {"site": "a", "pos": "tap", "names": "v_a{layers}"}, "logits": {"site": "lm_head", "pos": -1}},
"writes": {"w": {"site": "a", "pos": "tap", "do": {"swap": "v"}, "names": "w{layers}"}},
"intervened_models": {
  "original": {"input": "counterfactual", "reads": ["v"]},
  "band5_L10": {"input": "base", "reads": ["logits"], "writes": [{"w": {"layers": {"at_once": {"range": [10, 15]}}}}]}
}
```

Use `sweep` to compare separate interventions. Use `at_once` when one forward
must apply all selected writes. The compiler expands families before parsing
and canonicalization, so equivalent explicit entries produce the same digest.

Rule 28 rejects these forms:

| refused | why |
|---|---|
| two `at_once` fields on one entry | a member would have two indices and no name |
| an entry on one axis that references a family on another | a cross product inside one forward is not what a family means ; that is what sweeping is for. Checked for an entry that declares its axis and one that inherits it alike |
| a family entry that also carries a `sweep` wrapper, anywhere in it | expansion copies the wrapper to every member, and a sweep axis *is* its path (§3) ; so that is one axis per member, and their cross product. Ten members is 2¹⁰ points, under the point cap, so nothing later would catch it |
| a window naming an index the family does not carry | a band that resolves to fewer writes than the interval it declares is the silent bug the window form exists to refuse |
| an empty window, `{"at_once": {"range": [15, 10]}}` | the same bug from the other side: it names no write at all, so the band compiles as the un-intervened model and its metric reports no effect. A window is an axis, and the axis grammar refuses an empty one (§3) |
| a window spelled any other way ; a bare list, a bare `range`, an interval keyword | one word per object (§11.1): the subset is written the way the axis was |
| a family of more than **1024** members | a sweep's per-axis bound is safe because the point cap refuses the expansion afterwards; a family materializes entries directly, with nothing behind it. A table of that many addresses in one forward is where the answer is a sweep |
| a swept write list that names a family | which member it means would depend on the point, and families materialize before axes are found |

Families are unsupported in `save` and `train`. An intervened model's
`reads` may name a read family (every member); its `writes` may name a write
family or a window over one. Families align only
when their axis field and values agree. Generate explicit entries for a
joint selection across different fields within one point; use a `rows`
axis to express it across points.

<a id="32-axes--correlated-rows-and-dependent-axes"></a>
### 3.2 `axes`: correlated rows and dependent axes

The optional `axes` group binds several fields to related values. A `rows`
axis moves fields together, such as a layer, component, and position.
Dependent axes compute values from another axis. Reference an axis with
`{"axis": "<name>"}` wherever a sweep wrapper is allowed.

```json
{
  "axes": {
    "location": {
      "rows": [
        {"layers": 8, "component": "attention_output", "pos": {"index": -1}},
        {"layers": 9, "component": "mlp_output", "pos": {"index": -1}},
        {"layers": 10, "component": "block_output", "pos": {"index": -2}}
      ],
      "key": "layers"
    },
    "center": {"range": [0, 48]},
    "window": {
      "dependent_on": "center",
      "rule": {"clipped_band": {"width": 10, "clip_to": "layers"}}
    }
  },
  "method": {
    "positions": {"tap": {"axis": "location.pos"}},
    "sites": {
      "target": {
        "component": {"axis": "location.component"},
        "layers": {"axis": "location.layers"}
      },
      "restore": {"component": "block_output", "layers": {"axis": "window"}}
    },
    "train": {"seed": {"sweep": [0, 1]}}
  }
}
```

Three kinds of axis, one closed vocabulary ; the key a declaration carries is
the kind it is, and it carries exactly one of them
(`causalab/protocol/lowering.py` `AXIS_KINDS`; `tests/protocol/test_vocabulary_census.py`
holds this table to it):

| key | coordinate | lowers to |
|---|---|---|
| `rows` | one per row: the scalar field `key` names, else the row's index. A **non-empty list of objects with one key set**; every field but `key` is referenced by some `{"axis": "<name>.<field>"}` wrapper, and the fields of one row are substituted **together** | at each referencing field, `{"sweep": [that field's column]}` |
| `range` | the value: `[start, stop, step?]` of integers, half-open, non-empty | `{"sweep": [the values]}` |
| `values` | the value: a non-empty list of **distinct scalars** ; a tuple of fields is a `rows` axis | `{"sweep": [the values]}` |
| `dependent_on` | **none of its own** ; its parent's coordinate already names the entry; the value is in the point. Names a `range` or `values` axis and a `rule` | `{"sweep": [one computed entry per parent value]}` |

The rules a dependent axis may follow (`RULE_KINDS`, censused likewise):

| kind | means |
|---|---|
| `clipped_band` | `{"width": w, "clip_to": "layers"}`: for a parent value *c*, the band `[max(0, c − ⌊w/2⌋) … min(L − 1, c + ⌈w/2⌉ − 1)]` over the model's *L* layers, read from the registry entry of the document's one `model.key` ; so ten wide, centre 0 is `[0..4]` and centre 47 of 48 is `[42..47]`, and no member is ever outside the tower. `clip_to` is `layers` and nothing else in this version |

**Expansion order.** The named axes are the *slowest* coordinates, in
declaration order; inside each entry the ordinary sweep axes expand as §3 says,
last axis fastest. So the example's `location` × `train.seed` is exactly six
points, `[axes.location=8, seed=0]`, `[8, 1]`, `[9, 0]`, … ; never the
3·3·3·2 = 54 the wrappers would give as independent axes ; and the 48 centres
are 48 points, each with one band site. A point reached this way *is* the
hand-written point: its tree is that tree, so its canonical form and its digest
are that point's, as for every other point (sec. 7).

**Coordinates.** Exactly one per named axis, `axes.<name>` (the group is spelled,
since `axes` is not a method section): a `rows` axis records its `key` value, a
`range` / `values` axis its value; a dependent axis and a row's substituted
fields record nothing ; no list or object ever becomes a coordinate. Labels
follow (`[axes.center=5]`, sec. 3); existing labels are untouched.

**Canonical form and digests.** The `axes` stage lowers axis references into
sweep wrappers before parsing. This display form lets validation inspect each
field. The canonical experiment retains the authored `axes` group because
correlated rows determine which field values occur together. It folds each
row's values (`layers: 8` becomes `[8]`), retains an authored `key`, expands
ranges, and records each dependent rule with its computed entries.
Documents without `axes` keep the same canonical form.

Rule 14 caps the true point count: axis entries multiplied by the remaining
independent sweeps. The following authoring errors use `P2`, `P3`, or `P4`
with the axis path or wrapper path:

| refused | requirement |
|---|---|
| a non-object row, an empty row, or rows with different keys | Every row supplies the same fields. |
| a `key` outside the row's fields, a non-scalar key, or a repeated key value | Each row has a unique scalar coordinate. |
| repeated rows after folding, with or without `key` | Rows must be distinct. The error names both indices. |
| a `sweep`, `at_once`, or `axis` wrapper inside a row value or `values` list | Axis entries contain concrete values. |
| an axis reference to an undeclared name or field; a rows reference without a field; a reference to a range, values, or dependent axis with a field; a wrapper inside a list | References must name a declared scalar axis or a rows column. |
| an unused axis or unused row field other than `key` | Every axis is referenced or has a dependent axis. Each non-key column has a reader. A wrapper contains only `axis`; a mapping with additional keys is an ordinary value. |
| `dependent_on` names a rows axis or an undeclared axis | Dependent rules consume scalar entries. Add a computed column to a rows axis directly. |
| an unknown rule kind, such as `routed_experts` | Rules resolve from the specification before execution. Use workflow `fan_out` for values learned during a forward pass. Per-row token locations use position anchors (sec. 2.3). |
| `clip_to` with a swept `model.key`, or a value other than `layers` | One model's layer count bounds the band. |
| the expanded count exceeds the cap | Rule 14 reports the true point count. |

Generated and handwritten specifications follow the same rules.

## 4. Execution semantics

Each point defines forward groups for the required original runs and its
intervened models. Writes apply at their addresses, with the absolute write
first and additive deltas summed afterward. Reads see the completed writes.
Operand dependencies form an acyclic graph that determines execution order.

**Partial forwards.** The engine can stop after the deepest requested tap.
Shared groups use the union of their taps. A decoding group runs the full
model. For selected `lm_head` positions, the reference engine can gather
the head's input first and project only those rows. Reads used for gradients
retain the model's normal head computation.

An intervened group can start from a cached original residual at its first
required block, accounting for both writes and taps. The cache key includes
the original input identity. Original and decoding groups always start from
their inputs. Fit-constant prefixes can be reused during training and
evaluation and are freed after the last consumer. Summaries record
`prefix_reuse`; a resumed group still counts as a forward.

**Decoding.** A group runs prefill followed by the requested greedy steps.
The last generated token is consumed by a forward so its activations are
available. Writes apply during prefill, and at every decode step for a
model that declares `writes_during_generation` (sec. 2.9).

**Cohorts.** Training points with the same model settings, input rows, and
frame can share a fit. Each step concatenates member minibatches, applies
writes to each member's rows, and sums losses over independent parameters.
Members retain their own optimizer, seed, schedules, evaluation cadence,
budget, and early-stopping decision. Completed members leave the cohort.
Evaluation batches members with the same split. Different batch shapes can
change rounding.

Groups that decode, depend on a changing upstream model, or write per-step
state execute separately. Fixed groups use the cache. Cohort resume uses
the deepest prefix available to every member. `fit_rows` bounds packed
training rows while preserving each member's minibatch; `batch_rows`
bounds evaluation. After fitting, each point writes its own outputs in order.

**Fit caches.** A group is constant during fitting when its writes and
operand reads depend on no trained parameter. It can run once per row slice
and once for the evaluation split. The cache holds raw captures so a trained
featurizer can still transform them with gradients. Drawn counterfactual
minibatches bypass the cache. Inner training forwards are excluded from
`RunResult.forwards`.

**Write counts.** A module or kernel write must fire once per group forward.
A `delta_state` write fires once per addressed step. The engine checks each
microbatch and rejects a point with P4 and `component_unavailable` if the
observed count differs. The error names the write, group, and counts.

The engine resolves the complete write set before installing hooks. Each
hook edits a clone, and captures and tables are published only after the
forward returns and every count passes. A failed write therefore leaves no
partial result for the point.

Each run returns `{write: count}` per point and forward group in its result
summaries. A recorded run (`--record`) writes the counts into the receipt's
`fires` block. Shared captures retain the counts from their producing forward.
These observations remain outside the digest. Gaussian noise uses the
declared seed; canonicalization and sweep enumeration are deterministic.

### 4.1 Resolution: `available`, `unavailable`, `invalid`

Every resolution a run performs ; a component to a tensor, a selector to rows,
a bundle key to an entry ; ends one of three ways, kept apart by **type**
(`causalab/protocol/results.py`, torch-free):

| value | fields | where it goes |
|---|---|---|
| `available(mapping, denominator_key)` | what resolved | the result, counted as eligible |
| `unavailable(reason, detail, denominator_key)` | a `reason` from sec. 2.4's table, the fact in prose, the key it is counted under | the result **and** the denominator, as an excluded cell |
| `invalid(error_code, detail)` | the existing `V<n>` / `P<n>` code | raised by the validator ; never a result |

`invalid` reports an error in the specification and stops validation.
Examples include unknown components, invalid widths, and selectors that
identify no bundle entry. `unavailable` records an unmeasurable cell from
the data or model, such as an expert that receives no token.

A result cell is one saved value at one point. Available cells use the
ordinary result format. Unavailable cells add these fields to their tensor
entry and summary:

| field | value |
|---|---|
| `status` | `"unavailable"` |
| `reason` | one of sec. 2.4's reason codes |
| `detail` | the fact, in prose ; which expert, at how many positions |
| `denominator_key` | the key below |

The saved tensor is still written ; a `(0, d)` gather with per-row widths of
zero, or for a `reduce`d read the `(width,)` vector with `NaN` for `mean` /
`std` / `median` and `0` for `sum` / `count` (sec. 2.12).

A missing or ambiguous text alignment yields an unavailable read with its
reason and row details. Metrics score the aligned rows and record a null
value with a reason for each excluded row. The metric cell is unavailable
only when every row is excluded. Writes with missing alignment fail before
a forward pass.

`denominator_key` combines the saved value's name and point coordinates,
such as `iia[target.layers=3,pos=2]`. `RunResult.cells` holds the cells;
`RunResult.denominator` counts eligible cells and exclusions by reason.
The CLI reports these counts, and shard counts sum to the full experiment's
counts.

<a id="5-validation--load-error-checklist"></a>
## 5. Validation: load-error checklist

The loader enforces the rules below. Rule 2 warns about section order.
Rule 19 needs encoded inputs. Column checks in rule 4 and rules 20 and 25
use resolved tables; `validate` runs these data checks by default, and the
run repeats them before entering the engine. Rule 32 checks coverage when
build reads the score table.

Preflight checks every condition available before loading weights, including
support for the requested engine (rules 13 and 30). Runtime checks handle
facts that require encoded rows or executed tensors, such as per-row token
counts and write widths. The run makes the tokenizer's checks before the
weights load: token positions, and every metric's answer tokens (sec. 2.10).
`validate --tokenizer` runs them too.

Validation collects independent errors with their paths. Rules 3 and 4 stop
the pass on failure because later checks require their namespace and resolved
references. Across sweep points, distinct violations appear once in point
order. Section 9.1 describes the compiler's representative checks and the
engine's check of every expanded point.

Each rule has a stable slug in `RULES` (`causalab/protocol/rules/errors.py`)
and a fixed display number. For example, `[V22]` and sec. 5.22 refer to the
same rule. New rules take the next unused number; retired numbers stay
reserved.

1. **strict_keys**: Strict keys: unknown fields anywhere are errors; closed
   enums reject with suggestions. Derived fields (sec. 7) may not be authored.
2. **section_order**: Sections in the sec. 1 order. Other orders produce
   a warning and still parse. Canonicalization writes the specified order,
   so section order leaves the experiment digest unchanged.

3. **global_namespace**: Global namespace: no duplicate names across method
   sections 1–8 (§1); no reserved names (`base`, `counterfactual`,
   `counterfactual[j]`, `all`) declared.
4. **references_resolve**: Every reference resolves: sites (declared inventory
   only), positions, featurizers, params, reads, writes, intervened_models;
   a regularizer's `costs` keys name that term's own targets and a
   `constraint` term's targets are gates (sec. 2.11); every read or write
   through a position gate (`axis`) addresses a fixed `span`, one length per
   gate (sec. 2.5); an `anneal` or `control` target that is a term's weight
   names a
   **named** objective term with a numeric authored weight, a phase's
   `anneal` a featurizer the phase trains, and its `freeze_masks` gates the
   phase does not train (sec. 2.11); a `{"train": name}` save entry names
   exactly one named objective term that reduces a read or one eval label
   (sec. 2.12) ; and,
   from the component's capability row (sec. 2.4), the site
   sub-axes and the write mechanisms: `expert` on a component with no
   per-expert axis, a mechanism the component's write policy refuses, and a
   component the registry entry says the model does not have (a MoE tensor on
   a dense model, a full-attention tensor at a Gated DeltaNet layer, the
   routed-expert interior on a model whose entry records another
   `experts_implementation` than the grouped dispatch) are references that do
   not resolve. Each carries a reason code (sec. 2.4). A component the
   model's *family* does not serve (sec. 8) is the same refusal made by the
   run: the family is a property of the loaded module tree, which no load
   pass sees, so it is not a load rule. A metric's `minimum_count` (sec. 2.10
   "Eligibility") above the resolved base table's **maximum eligible count** ;
   the rows carrying a value in every column the metric names ; is a reference
   to more eligible rows than the data resolves, refused under `validate`
   naming the maximum; a threshold at exactly the maximum passes, and a metric
   with none makes no claim. A `match` metric whose `mode`
   contradicts a recorded table's `string_mode` under sec. 2.10's derivation
   is a reference the table's own bytes refuse to resolve, held at load and
   again before the first forward (sec. 2.2).
5. **read_bindings**: Read references resolve to one model: a read named bare
   (a write operand, a `kl`/`js` `target` outside a save) binds when exactly one model lists
   it, and is refused naming the models when several do; `{"read", "model"}`
   — and the `read` + `model` of a save entry, objective term or eval entry —
   names a declared model that lists the read (sec. 2.7, sec. 2.9).
6. **write_operands**: Writes carry no `model`/`input`/conditions; operands
   name reads, params, or literal scalars.
7. **write_membership**: Every write and every read is in ≥ 1 intervened_model;
   the model graph is acyclic: every IM has a mandatory valid `input` and
   lists no read twice; an operand read taken on the model that carries the
   write is a self-edge, hence a cycle.
8. **one_absolute_write**: Per (site, overlapping pos, model): ≤ 1 absolute
   write.
9. **dims_disjoint**: `dims` selections co-occurring at one address in one
   model are disjoint.
10. **save_manifest**: `save` non-empty; entry shapes exact; one entry per
    value: a read entry saves one bound read as a tensor (`.safetensors`) or
    one aggregation over it as a table (`.json`), and no two entries restate
    the same `(read, model, aggregation)`; every trained featurizer saved;
    nothing else saveable.
11. **sink_rule**: Sink rule: every read on every model has a consumer — a
    save entry, an aggregation's input or target, or a write in force in some
    model.
12. **featurizer_legality**: Featurizer legality: loaded featurizers
    (`file_path`) are not trained; trained featurizers are declared kinds with
    trainable slots; and a `featurizer` composition is non-empty with no
    repeated stage (sec. 2.5 ; a stage's width comes from its position in the
    chain).
13. **pytorch_fn_local**: `pytorch_fn` present ⇒ refused unless the selected
    engine is local.
14. **sweep_wrappers**: Sweep wrappers well-formed; the expanded point count is
    reported (and may be capped without an explicit override flag; the count is
    taken after correlated rows are applied, sec. 3.2). A `{"train": name}`
    save entry names no term whose aggregation carries a sweep (sec. 2.12).
15. **artifact_fields_resolve**: Artifact-valued fields resolve (missing
    artifact = error, never a default).
16. **generation_read_only**: Generation is read-only: no write's `pos`
    carries `generated`, and `train` does not co-occur with a `generated`
    position. A model that declares `writes_during_generation` (sec. 2.9) is
    decoded by some read, and each write it lists sits at `all` or
    `{"index": -1}` with no anchor, takes no read as an operand, and is not
    `gaussian`.
17. **realization_coherent**: The model's realization is coherent: a
    `quantization` block carries only the knobs its own scheme has
    (`double_quant` is 4-bit vocabulary, `int8_threshold` is int8 vocabulary).
18. **shape**: Shape (§1): the four groups are present and nothing else is at
    the top level; the header holds `protocol_version` `"4"` and at most `title`
    and `description`, both free text; the method holds only its eleven
    sections. A document with a top-level `version` ; the `protocol_version` 1
    spelling ; a document declaring `protocol_version` `"2"` or `"3"` ; and a
    `metrics` section are each refused by name, with `causalab migrate` as the
    answer; a workflow document (`steps`) handed to this loader is refused as a
    workflow.
    All of this is decided before anything addresses the tree by path, so a
    `--set` on the wrong kind of document is refused as that and not as a
    missing path.
19. **write_widths_uniform**: Write widths are uniform: every row a write
    addresses carries the same number of positions. Only an `all` or `variable`
    write can be ragged, and only the tokenizer can say how wide a row is ; so,
    unlike the rest of this checklist, rule 19 is checked when the run encodes
    its inputs, **before any forward pass**, not at load. `validate` cannot
    decide it: the pure verbs hold no tokenizer, by design. A write may
    declare how a ragged window lands (`writes.<w>.ragged.policy`, sec. 2.8):
    `refuse` ; the behaviour of an absent field ; is this refusal;
    `exact_length_buckets` and `padded_masked` land every row at its own
    width, before any forward and with no change of batch geometry, and
    record the policy and the per-row widths in the run receipt's
    `execution.ragged` block. A `column` window is checked here like a
    `variable` one. A ragged operand paired into a write under `refuse`, or
    whose widths disagree with the write's, is the same refusal.
20. **base_is_paired_schema**: `base` is the schema of a paired row: every
    non-`base` data role's columns are a subset of `base`'s, and equal to them
    when the role names a different dataset (sec. 2.2). Like rule 4's column
    half, this uses the resolved tables during `validate`.
21. **operand_reachability**: Operand reachability: every write operand that
    names a **read** is read at an address no deeper than the one the write
    lands on, in the `(layer, intra-block rank)` order of sec. 2.4's vocabulary.
    Equal is legal ; that is the harvest/inject idiom. Params and literal-scalar
    operands have no address and are unconstrained. A band (sec. 2.4 `layers`)
    is held the way it runs: an operand read on a band of the same length as
    the write's feeds it member by member, so member *i* of the read is no
    deeper than member *i* of the write; any other operand is broadcast, so the
    read's **deepest** member is no deeper than the write's **shallowest**.
22. **split_declaration**: Split declaration (sec. 2.2). Every row declares `split`. An undivided table
    uses one value. A reference to a table with several splits must include
    `#<split>`. Across splits of one table, prompts at both endpoints must be
    disjoint. The dataset resolver checks these conditions whenever it reads
    a table.

    For a fit, training data and `train.eval.split` must also have disjoint
    endpoints when they use different references. Validation checks this,
    and execution checks it per point before any forward. A failure names
    both references, roles, and the first shared prompt. Naming the same
    reference for both explicitly permits evaluation on training data.
23. **group_legality**: Group legality: a gate's `group` (sec. 2.5) names an
    axis the site's component has, **in the site's own basis**, and its
    coordinate→group map is derived from the registry ; `(heads, head_dim)` off
    the component's shape under `head`, `(num_experts, d_expert)` off the
    model's expert table under `expert_neuron`. So: the component carries the
    axis the group needs (`head` on a head-major component, `expert_neuron` on
    `expert_activation` or `expert_neuron_output`; `group: "head"` on `block_output` is refused, as
    `head:` on it is by sec. 2.2); the site does not already select a single
    member of it (`head: 3` under `group: "head"` is one group ; a
    coordinate-wise gate under a grouped gate's name); the gate is the **first
    stage** of every chain that uses it (a grouped gate acts on the component's
    own coordinates ; after any other stage, a `subspace`, an `sae` or a
    `standardize` alike, coordinate 5 is no longer a channel of head 0); and, when one
    featurizer is used at several sites, they all derive the same map. Like
    rule 4's width half this reads the model's declared axes, so it is decided
    as the document canonicalizes ; a loaded `file_path` gate included ; and
    **no model is loaded**: `ModelInfo` is static config. Whether a *fitted
    bundle* matches the document's group and map is rule 15's: the stamped
    `group` and `group_map`.
24. **code_declaration**: A `code` declaration agrees with the source it names (sec. 2.8.1): the
    `locator` resolves to a Python source file (there has to be something to
    hash); the declared `args` fit the function's signature, and an argument it
    requires and the document does not give is refused; and no read the
    function makes is both **statically detectable** and undeclared ; a literal
    `os.getenv("X")` outside `env_inputs`, a literal path outside
    `data_inputs`. The last two halves need a readable `def`: a function built
    by a factory has none, and is hashed but not signature-checked. Resolution
    imports nothing.
25. **row_roles**: Declared row roles match the resolved data (sec. 2.8.1): a `code` entry
    whose `row_roles` cover *n* rows is refused when the input role of an
    intervened_model its write is in force on resolves to a table of a
    different length. This check uses resolved tables during `validate`.
    Omitting `row_roles` leaves row counts unrestricted.
26. **alignment_declared**: A declared `alignment` fits its address (sec. 2.3):
    the value is one of `one_to_one` | `one_to_many` | `many_to_one` | `absent` |
    `ambiguous`; the address can carry one (`all` takes no modifiers, and a
    `generated` position addresses a result); and
    where the document alone fixes the cardinality ; an `index` is one token per
    row on every input, an unscoped `span` one joint window of one width ; the
    declaration is `one_to_one`. Everything else about a declaration needs the
    tokenizer and is checked when the run resolves its positions
    (`pipeline.resolve_positions`, before any weights load), as rule 19
    is when it encodes: the pure verbs hold none. Named on the field (`positions.<name>.alignment`,
    `reads.<r>.pos.alignment`, `writes.<w>.pos.alignment`).
27. **segment_declared**: A `segment` anchor names a declared segment and a
    span is well-formed (sec. 2.2.1, sec. 2.3): `segments.frame` is one of
    `chat`; `segments.system` needs the chat frame; a name in
    `segments.declare` is not one of the chat frame's own; every `segment`
    anchor ; a whole-segment selector, a `scope` or a `relative_to` ; names a
    segment the section declares (no section, no segment anchors); an anchor
    on `continuation` carries `generated` (the frame's decode budget lives on
    the position) and the whole continuation is spelled `{"generated": …,
    "all": true}`; an `atomic` span whose member set the document alone fixes
    has two or more members. Document-decidable only: whether a declared
    segment *occurs* in a row is the frame's to find when the run encodes its
    inputs (`absent` / `ambiguous`, sec. 2.3). Named on the field
    (`segments.frame`, `segments.system`, `segments.declare.<name>`,
    `positions.<name>`, `reads.<r>.pos`, `writes.<w>.pos`). A row's
    `edit_groups` declaration (sec. 2.2) is held to the same standard, at
    `data.base`: its spans lie inside the pair's texts, both sides declare the
    same number of constituents and an `atomic` group has two or more (checked
    at `validate`); and a position that addresses a constituent of an
    `atomic` group without its siblings, within one intervened model, is not a
    well-formed span for that pair ; refused when the run encodes its inputs,
    naming the group and the missing siblings (`intervened_models.<m>`).
28. **family_wrappers**: `at_once` families well-formed (sec. 3.1): one axis
    per entry and one axis per reference, no `sweep` wrapper anywhere in a
    family entry, a `names` template naming the axis field, distinct values, at
    most 1024 members, member names that are neither reserved nor already
    declared, and a write-list window that is non-empty and resolves to exactly
    the interval it names. References to a family from `save`, `train` or a
    swept write list fail this rule.
29. **kl_operands_compatible**: `kl` operands are comparable (sec. 2.10): the
    two reads hand the metric distributions over the same **effective width**
    ; the site's component width from static model config (one head's slice
    under `head`), folded through the read's featurizer chain and `dims` ;
    through the same **transform** (the same featurizer composition and `dims`
    selection on both sides), in the same **token-position frame** (both
    prompt, or both `generated`). A shared component label is rule 4's
    precondition, not a distribution's shape. A loaded SAE's width is its
    bundle's, and the per-row count of two continuation windows is the
    tokenizer's; both stay the executor's.
30. **train_engine_supported**: The routed engine can execute the fit as
    authored (sec. 2.11): a free `params` tensor in `train.params`, a
    `train.precision` other than `fp32`, and an `eval` counted in `updates`
    each need the sec. 8 capability that says the engine's loop implements it
    (`train_free_params`, `train_loss_precision`, `train_eval_updates`), and
    the refusal names the field and the missing verb. Decided exactly when the
    engine is known ; a `validate` handed the engine, and every run through
    `check_engine` (sec. 9.1) before the engine loads a model ; so the
    rule refuses the engine, not the document: an engine declaring the verb
    runs the same document unchanged.
31. **metric_position_scalar**: A metric reduces one position per example
    (sec. 2.10): a prompt-frame read whose address the document fixes to more
    than one token ; an `all` address, an unscoped `span` wider than one, an
    `indices` set or union of more than one ; cannot feed a metric. A
    `variable` or `column` window is as wide as the tokenizer makes it and is
    the executor's to refuse; a continuation read is exempt by design, its
    metric reduces per decode step (sec. 2.3).
32. **scores_init**: A gate's `init.from_scores` fits the gate (sec. 2.5):
    `keep` is at most the gate's unit count (decided as the document
    canonicalizes, where the width and the group map are derived ; like rule
    23), and the score table, after `where`, names every unit of the gate
    exactly once ; one `unit` column per axis of `theta`, each index inside
    it, none repeated, every score a number (decided at build, when the table
    is read, like rule 19). A start that skipped or doubled a unit would seed
    a mask nobody authored.

<a id="6-derived--never-authored"></a>
## 6. Derived: never authored

The compiler derives some properties, such as featurizer widths and resolved
token indices. A document never states them. The [internals
page](intervention_protocol_internals.md#6-derived--never-authored) lists each
one and where it comes from.

## 7. Canonical form and digests

Each document and each sweep point has a digest over its canonical form.
`causalab digest <doc>` prints it, and the run receipt records it. The
[internals
page](intervention_protocol_internals.md#7-canonical-form-and-digests) defines
the canonical form.

## 8. Engine contract

An engine runs a compiled document through `execute(compiled, run)`. The
[internals page](intervention_protocol_internals.md#8-engine-contract) defines
what it receives, the services it implements and what it returns.

## 9. CLI and the Python entry point

The CLI calls the Python API of the protocol layer. `causalab.protocol`
exports `run_protocol`, `RunContext`, and `RUN_RECORD_NAME`. Import other
functions from their defining modules, for example `validate_document` from
`causalab.protocol.rules.document`. A notebook or script can call the API
directly:

```python
from causalab.io.env import ResolutionEnv, FileDatasets, FileArtifacts
from causalab.protocol import run_protocol
from causalab.neural.shared.engine_router import route

env = ResolutionEnv(datasets=FileDatasets(root=data), artifacts=FileArtifacts(root=arts))
engine = route("pytorch_hooks", device="cuda")            # the one engine, named by the caller
result = run_protocol(document_path, env, engine, out)    # -> RunResult
for manifest_path, disk_path in result.files.items():
    ...
```

The CLI handles arguments, printed output, and exit codes.

<a id="91-one-compiler-four-doors"></a>
### 9.1 Compilation stages

The CLI, the Python API and the workflow runner share one sequence of
compilation stages, described in the [internals
page](intervention_protocol_internals.md#91-one-compiler-four-doors).

## 10. Worked examples

An interchange intervention, complete: the header names the file, `model` and
`data` name what it ran on, and `method` is the experiment ; the mechanism,
the scoring, and the addresses it reads and writes at.

```json
{
  "header": {
    "protocol_version": "4",
    "title": "Weekdays interchange, Llama-3.1-8B layer 18",
    "description": "Swap the answer-slot residual from the counterfactual into base at layer 18, in bf16, over the weekdays training pairs; IIA scoring."
  },
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16"},
  "data": {
    "base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"},
    "counterfactual": {
      "dataset": "natural_domains_arithmetic/data/weekdays#train",
      "field": "counterfactual_inputs[0]"
    }
  },
  "method": {
    "intervened_models": {
      "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
      "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]}
    },
    "sites": {
      "target": {"component": "block_output", "layers": [18]},
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
        "aggregation": {"kind": "match", "expected": "cf_answer"},
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

`causalab explain` on that document prints the plan and the document digest.
Moving the experiment to another network is an edit to `model` (and, where the
addresses differ, to `sites`), and a diff of the two files says whether the
experiment itself survived the move. A layer scan is a
one-line edit: `"sites": {"target": {"component": "block_output", "layers":
{"sweep": {"range": [0, 32]}}}}` ; and the axis is `sites.target.layers`,
section-rooted (§1), in every coordinate label and every workflow reference.

Path patching (sender → receiver, off-path frozen; shows cross-model flow):

```json
{
  "header": {"protocol_version": "4"},
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base": {"dataset": "IOI/data/default", "field": "input"},
    "counterfactual": {"dataset": "IOI/data/default", "field": "counterfactual_inputs[0]"}
  },
  "method": {
    "intervened_models": {
      "original_counterfactual": {"input": "counterfactual", "reads": ["v_sender"]},
      "original_base": {"input": "base", "reads": ["v_a10", "v_a11"]},
      "patched": {
        "input": "base",
        "reads": ["v_receiver"],
        "writes": ["swap_sender", "freeze_10", "freeze_11"]
      },
      "final": {"input": "base", "reads": ["logits"], "writes": ["inject"]}
    },
    "sites": {
      "sender": {"component": "attention_premix", "layers": [9], "head": 9},
      "receiver": {"component": "block_input", "layers": [12]},
      "a10": {"component": "attention_output", "layers": [10]},
      "a11": {"component": "attention_output", "layers": [11]},
      "lm_head": {"component": "lm_head"}
    },
    "reads": {
      "v_sender": {"site": "sender", "pos": -1},
      "v_a10": {"site": "a10", "pos": -1},
      "v_a11": {"site": "a11", "pos": -1},
      "v_receiver": {"site": "receiver", "pos": -1},
      "logits": {"site": "lm_head", "pos": -1}
    },
    "writes": {
      "swap_sender": {"site": "sender", "pos": -1, "do": {"swap": "v_sender"}},
      "freeze_10": {"site": "a10", "pos": -1, "do": {"swap": "v_a10"}},
      "freeze_11": {"site": "a11", "pos": -1, "do": {"swap": "v_a11"}},
      "inject": {"site": "receiver", "pos": -1, "do": {"swap": "v_receiver"}}
    },
    "save": [
      {
        "read": "logits",
        "model": "final",
        "aggregation": {
          "kind": "logit_diff",
          "a": "answer",
          "b": "cf_answer"
        },
        "file_path": "logit_diff.json"
      }
    ]
  }
}
```

DAS with a k × seed sweep (9 fits from one harvest; shows axes + train +
featurizer save):

```json
{
  "header": {"protocol_version": "4"},
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base": {"dataset": "natural_domains_arithmetic/data/weekdays#train", "field": "input"},
    "counterfactual": {
      "dataset": "natural_domains_arithmetic/data/weekdays#train",
      "field": "counterfactual_inputs[0]"
    }
  },
  "method": {
    "intervened_models": {
      "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
      "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]}
    },
    "sites": {
      "target": {"component": "block_output", "layers": [18]},
      "lm_head": {"component": "lm_head"}
    },
    "featurizers": {
      "rot": {
        "kind": "subspace",
        "k": {"sweep": [8, 16, 32]},
        "parametrization": "cayley"
      }
    },
    "reads": {
      "v_cf": {"site": "target", "pos": -1, "featurizer": "rot"},
      "logits": {"site": "lm_head", "pos": -1}
    },
    "writes": {
      "patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
    },
    "train": {
      "objective": {
        "ce": {
          "weight": 1.0,
          "read": "logits",
          "model": "patched",
          "aggregation": {"kind": "cross_entropy", "target": "label"}
        }
      },
      "params": ["rot"],
      "optimizer": {"name": "adamw", "lr": 0.001},
      "steps": {"epochs": 10},
      "batch": {"pairs": 16},
      "eval": {
        "every": {"epochs": 1},
        "split": "natural_domains_arithmetic/data/weekdays#test",
        "aggregations": {
          "iia": {
            "read": "logits",
            "model": "patched",
            "aggregation": {
              "kind": "logit_diff",
              "a": "cf_answer",
              "b": "base_answer"
            }
          }
        }
      },
      "early_stop": {"on": "iia", "patience": 3, "mode": "max"},
      "seed": {"sweep": [0, 1, 2]}
    },
    "save": [
      {"train": "iia", "file_path": "iia.json"},
      {"train": "ce", "file_path": "ce.json"},
      {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
    ]
  }
}
```

## 11. Glossary

<a id="111-one-word-per-object--the-five-names"></a>
### 11.1 Terms

Use these terms consistently:

| term | means |
|---|---|
| **research pipeline** | the sequence of questions and experiments in a study |
| **runtime implementation** | the installed package and engine that execute a specification |
| **intervention specification** | the JSON or YAML description written by the researcher |
| **compiled intervention** | the resolved experiment at one sweep point, with its digest |
| **run receipt** | metadata recording the requested experiment and its execution |

Digests identify the compiled forms. Specifications that differ only in
irrelevant order can therefore share a digest. Serialized names such as
`type: intervention_protocol` remain part of the API.

Workflow script identity includes the module's bytes. Editing a comment or
docstring changes that identity and requires updated recorded hashes.
The vocabulary census checks Markdown and Python prose, with its listed
exemptions for hashed scripts and the test's own phrase data.

### 11.2 Causal abstraction correspondence (Geiger et al., arXiv:2301.04709)

The correspondence depends on the experimental design. See
[Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://arxiv.org/abs/2301.04709)
for the formal definitions.

| specification | role in a causal abstraction experiment |
|---|---|
| `model` | the low-level model ℒ; the task supplies the high-level model ℋ |
| `intervened_models.<name>` | the low-level model under the declared input and writes |
| a write's `do` | the operation used to intervene |
| `swap` from a counterfactual read at the aligned site | an interchange intervention; a read from an already intervened run can define a recursive interchange |
| site, position, and selected dimensions | the low-level variables addressed by the intervention |
| featurizer | a feature map used to define the intervention; a formal translation also requires the appropriate invertible construction |
| mean `match` on interchange outcomes | IIA when expected answers come from the corresponding high-level intervention |

State which inputs, interventions, and mappings an experiment tests. Agreement
on those cases measures support for that abstraction within the tested scope.
