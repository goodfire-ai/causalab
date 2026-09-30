# Run a behavioral analysis

A [behavioral workflow step](workflow_protocol.md#27-behavioral-steps--the-declarative-behavioral-runner)
runs a model without interventions, scores its outputs, and writes
`continuations.json`, `outcomes.json`, and `decision.json`.
[qualify.json](../tests/workflow/fixtures/behavioral/qualify.json) is a worked
fixture with a [probe specification](../tests/workflow/fixtures/behavioral/protocols/qa_probe.json).

Run the fixture on CPU to check the interface with random weights:

```bash
uv run causalab validate tests/workflow/fixtures/behavioral/qualify.json \
    --engine auto \
    --data-root tests/workflow/fixtures/behavioral \
    --artifacts-root tests/workflow/fixtures/behavioral
uv run causalab run tests/workflow/fixtures/behavioral/qualify.json \
    --data-root tests/workflow/fixtures/behavioral \
    --artifacts-root tests/workflow/fixtures/behavioral \
    --out runs/behavioral-smoke \
    --engine pytorch_hooks \
    --device cpu \
    --batch-rows 4
```

For a study, specify the frozen model and task scorer. Use one step per
scientific split with explicit EOS IDs and all-row retention. Choose thresholds
for the study; the fixture's thresholds test the interface.

## Define the measurement

Record model and tokenizer revisions, precision, exact input format, accepted
answers, generation limit, and disjoint parent splits. Specify
`decoding.mode: deterministic` and `decoding.eos_token_ids` for reproducible
stopping. When EOS IDs are omitted, the PyTorch engine resolves them from the
model's generation configuration, then the tokenizer.

The engine selects tokens from raw logits. Transformers defaults for sampling,
repetition penalties, suppression, and forced tokens do not apply. Check an
engine's output contract before using it for the study.

Set `retain.generations: all` for an audit of every example. The default retains
a bounded sample. Each continuation records:

| Field | Meaning |
|---|---|
| `token_ids` | Content before EOS |
| `emitted_ids` | Generated content including terminal EOS |
| `padding_ids` | Slots after stopping |
| `terminal_eos_id`, `stop_reason` | How generation stopped |
| `eos_token_ids`, `decoding` | Effective stopping and decoding settings |
| `text` | Decoded content with non-EOS special tokens preserved |

Score the recorded text or exact IDs. Report first-token accuracy separately
from complete-answer accuracy. For tied logits, use the first emitted ID or
unrestricted argmax on prompt-end logits to recover the greedy decision.

Check the task's content-bound `ScoringSpec` against the study's answer and
truncation rules. The runner marks length-capped rows `truncated`. A study that
accepts an answer boundary before the cap needs an explicit scorer and outcome
contract for that rule.

## Plan and validate

Give each behavioral step a dataset reference qualified by its scientific
split. The supported purposes are development, reserve, and confirmation.
Use separate analyses for diagnostic panels that serve another purpose.

`causalab.workflow.behavioral_plan.plan_batches` groups rows by split and
prepared target-prefix/output length before chunking. It returns exact source
indices and rejects duplicate IDs or parents shared across splits. A compute
shard can contain batches from several splits; retain separate summaries.
Validate the complete workflow and compare resolved row IDs with the batch plan.

Test the save, reload, and scoring path with early or alternative EOS,
length caps, control-token prefixes, ties, split boundaries, mixed answer
lengths, and a short final batch. Compare a small sample with ordinary
generation under identical settings. Test exports at the intended study size.

## Control memory and recover runs

Unsaved continuation-head metrics without transforms project at most 16
positions at a time, then release logits after reduction. Text-only metrics
use emitted IDs. Full tensor saves and transformed reads require separate
memory estimates; forward microbatches leave those requests unchanged.
Continuation position `j` reads the distribution after consuming emitted token
`j`. Use prompt-end logits for the first emitted decision.

`causalab.analysis.sequences.prepare_sequences` queries vocabulary size once
per batch. Pass the continuation's `eos_token_ids` when preparing emitted IDs
to identify padding after alternate terminal tokens. Cache immutable tokenizer
metadata outside token loops. Retain raw generations, then expand target views
for selected prompts. Time preparation and generation separately from saving
and analysis. Verify equivalent records when changing batches.

Use workflow `--resume` to reuse completed steps after their artifact and code
identities pass verification. Make chunks separate steps when recovery must
operate at that level. Apply the qualification rule to counts over the full
scientific split.

## Validate targets and report results

Compare target positions with natural continuations, checking separators,
exact IDs, predictive positions, and supplied prefixes. Report natural
generation, supplied separators, and supplied correct answer tokens as
separate conditions. Use a known input perturbation to check the intended
readout. Changes to whitespace or other input content require new records.

Link findings to raw examples and error inspection. Check the rendered report
and reconcile its claims with completed audits before publication.
