# Commands and saved results

This guide lists the `causalab` commands and the files a run saves. The
[experiment guide](running_experiments.md) shows each command on a worked
document, and its [flag table](running_experiments.md#4-run-it) lists every
`run` option.

## Commands

| Command | Purpose |
|---|---|
| `validate <doc> --engine auto` | Check the document and its data references. With `--tokenizer`, also resolve its token positions and metric answers with the model's tokenizer. |
| `explain <doc> --engine auto` | Show the forward plan, sweep size, and engine requirements. |
| `dry-run <doc>` | Resolve everything a run decides before the weights load, and report it. `--tokenizer` decides the tokenization too. |
| `run <doc> --engine auto --out <dir>` | Execute the experiment and save its outputs. |
| `digest <doc>` | Print the experiment's identity; for a workflow, print each step's identity. |
| `migrate <path>` | Update older JSON documents and the examples embedded in Markdown to the current format. |
| `measure` | Collect paired before/after workflow measurements; see [before/after measurements](measurement.md). |

`causalab <command> --help` prints each command's options.

`--engine` selects `pytorch_hooks`, `nnsight`, or `auto`. The current `auto`
choice is `pytorch_hooks`. A capability check reports requirements the selected
engine cannot serve. Training uses `pytorch_hooks`.

`run` resolves each metric's answer tokens with the model's tokenizer before
the weights load, so a table the tokenizer cannot score is refused `[P2]`
before any model loads. The refusal names every value that fails, with a
count and the first table row. A workflow run makes this check for every inner
document before its first step, and for a document that depends on an
earlier step at that step. Under `--resume`, each attempted step makes it at
its turn, so a reused step needs no tokenizer. `validate --tokenizer` and
`dry-run --tokenizer` run the same check, and the token-position check beside
it, without a run. Their `tokenizer` line names the metrics whose answers
resolve. A metric read over generated tokens is named apart, as checked
when scored: which rows generate a step is known only after the decode, so
its answers are checked when the run scores them. They load the tokenizer as a run does, from the Hugging
Face cache or by a download of its files, and never the weights. A tokenizer
that cannot be loaded, such as one not in the cache under `HF_HUB_OFFLINE=1`
or a gated one without a token, is refused `[P4]` with its key and revision.
A bare answer that is one token after a prompt that ends in a letter or digit
gets a warning, because a model usually emits the space-prefixed form there.

Use `--data-root` and `--artifacts-root` to resolve external inputs. Packaged
task tables remain available under `causalab/tasks/<task>/data/`.
`--register-from-hf` lets inspection commands fetch metadata for an unregistered
model. `run` resolves that metadata automatically.

For a protocol run, `--dtype` sets `model.dtype`, and `--points START:STOP`
selects a range of sweep points. Set a workflow step's precision in its document
or `set` block. Workflows support `--resume`, which checks identities and file
contents before reusing a completed step. Use `--batch-rows N` to bound each
forward batch in the PyTorch engine. `--verbose` prints the run's progress on
stderr: point selection, each model load, fits, each point's run, and the
output write. It changes no output file.

## Saved results

A protocol run writes each entry of the document's `save` list to its
`file_path` under `--out`. The run prints one line per file and a count of the
eligible cells:

```bash
uv run causalab run demos/methods/protocols/minimal_cpu.json \
    --engine auto \
    --data-root tests/protocol/fixtures/data \
    --artifacts-root tests/protocol/fixtures/artifacts \
    --out runs/minimal_cpu \
    --device cpu
# saved iia.json -> runs/minimal_cpu/iia.json
# saved logit_diff.json -> runs/minimal_cpu/logit_diff.json
# cells 2 / 2 eligible
```

A metric table is a JSON array with one object per example and sweep point:

```json
{
  "example_id": "0",
  "metric": "iia",
  "value": 0.0,
  "unit": "fraction",
  "estimand_version": "match/v1",
  "eligible": true
}
```

A swept field adds a column named after its path, such as
`sites.target.layers`. Dense values, such as saved activations and trained
featurizers, are `.safetensors` files. [Section 2.12 of the intervention
protocol](intervention_protocol.md#212-save) defines every saved format,
including the derived records. Report the excluded cells with the result: the
[experiment guide](running_experiments.md#4-run-it) explains the `cells` count.

With `--record`, the run also writes two files into `--out`:

| File | Contents |
|---|---|
| `protocol.json` | The run receipt: the resolved document, its digests, the execution settings including the device, and the commit each model revision resolved to. |
| `events.jsonl` | The event stream of the run, one JSON object per line. |

The document's four groups are `header`, `model`, `data`, and `method`. The
`method` group describes the intervention and measurement choices that can
transfer between networks. Digests identify whole documents and sweep points.
Device placement and batch bounds are recorded as execution settings. The
receipt's `models` list names the snapshot commit behind a revision such as
`main`. The `_step.json` of a protocol or behavioral step records both as
well. A script step records neither. The parent record of a fanned-out step
leaves them to its children's records.

A workflow run writes each step's outputs to
`<out>/<output_dir>/<step>/`, beside the runner's step record `_step.json`.
The run directory also holds the manifest `workflow.json` and the event
stream `events.jsonl`. [Section 1.1 of the workflow
protocol](workflow_protocol.md#11-output_dir-and-the-run-tree) defines the
whole run tree.
