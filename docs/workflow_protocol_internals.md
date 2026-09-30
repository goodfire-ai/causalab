# Workflow protocol internals

This page holds the parts of the
[workflow specification](workflow_protocol.md) that concern the
implementation: how scripts are resolved and hashed, the properties the loader
derives, and the runner's services and record formats. Read it to change the
runner or to debug a step record. Section numbers match the specification, and
a reference such as "§2.8" points into it. "IM spec" is the
[intervention specification](intervention_protocol.md).

## 4.2 Script resolution and the torch-free guarantee

`validate` and `digest` read script files with `ast.parse`. They check
existence and a module-level `main`, then hash the bytes without importing the
script or numerical libraries. Other checks run during execution. Hashing uses
`causalab.protocol.identity.source_sha256`.

Path-based script identity includes imported siblings under its directory.
The static walk follows transitive and function-local imports, excluding
`TYPE_CHECKING`. It reads files without importing them.
Non-empty closures are stored as sorted `closure: {path: sha256}` and
`closure_sha256`, a hash over newline-joined `"<path> <sha256>"` records.
`script_sha256` remains the script's own hash. The reader is
`causalab.protocol.identity.import_closure(..., repository=False)`.

Causalab package files are covered by runtime `tree_digest`; module-based
scripts have no sibling closure. Dependencies use runtime dependency identity.
Parent `__init__.py` execution and dynamic imports are outside the static
closure. Repository-wide closure walks support layering tests and enter no digest.

`importlib.util.find_spec` imports parent packages. Packages holding shipped
scripts must therefore import without numerical libraries.
`tests/protocol/test_load_is_torch_free.py` checks these imports and workflow
validation.

## 6. Derived: never authored

| property | derivation |
|---|---|
| step dependencies, schedule, parallelism | the reference graph (§3) |
| a protocol step's sweep axes and point digests | the IM spec's expansion, republished in `_step.json` |
| the columns a script step's table actually has | verified against its declaration on write |
| the columns a `reduction` binds (`unit`, `group_by`, `weight`, `resample_unit`; a curve's `estimator.x`, `estimator.y`, a column `estimator.normalize`) | checked against the real table at run time; refused naming the column and the dimension (§2.6) |
| a reduced row's `unit`, `estimand_version` | the input table's unit (`count` → `count`, `cpr` → `dimensionless`, `auc` → `null`), the authored identifier or `<estimator>/v1` (§2.6) |
| a script step's digest, and the identity it stamps | its canonical entry: script hash, a `{"path": …}` script's sibling closure when it has one (§4.2), inputs, outputs, `runtime`: plus what its tensor inputs agree on |
| inner-document digests | the IM spec's canonicalization |
| the run manifest | stamped at execution |
| a step's `skipped` status | a conditional's verdict over its producer's `decision.json`, transitively over every step that depends on a skipped step (§2.8) |
| the edges a decision, a conditional or a receipt adds to the schedule | `values.step`, `predicate.decision.step`, `requires_receipt.step`, and conditional → every step it gates (§2.8); schedule only, never canonical |
| a fan-out's children, their selections and their identities | the parent's `fan_out` over its compiled points (§2.9): one child per axis value or per shard, each with its point indices and the digest of the parent's entry plus its shard; the children's edges (after the parent's dependencies, before the join); a per-child conditional's children likewise. Schedule and identity, never canonical |
| a nested workflow's steps, their names and their edges | its document, loaded at the outer's load (§2.10): every inner step under `<step>/<inner>` with its own identity, its inner edges, and the `workflow` step's own edges inherited by the inner steps nothing inside precedes |

The loader resolves the complete graph, including children and nested steps.
Runtime conditionals select the steps that execute.

Protocol records retain axes for analysis. Shipped `select` and plot scripts
average rows by axes and exclude nulls. Declare a reduction for another unit,
estimator, or interval (§2.6).

## 8. Runner contract

The required `--engine` applies to every protocol step. `route_engine` checks
capabilities before loading weights and reports `[V13]` for a shortfall.
`auto` selects `pytorch_hooks`. Per-step engine overrides are unsupported.

| service | contract |
|---|---|
| schedule | topological order; independent steps may parallelize |
| stores | the run-tree/external artifact overlay (§3), one output root |
| inputs | resolve the §3 grammar to paths and scalars |
| scripts | invoke `main(inputs, outputs)`, in-process or isolated |
| launch | `--parallel` above world 1 spawns ranks or joins an external launch. Ranks execute protocol and behavioral steps collectively; the joiner alone writes attempts, records, streams, manifests and pins. Decisions and refusals are shared before ranks proceed. Non-publishing ranks perform only the collective engine work. The data axis is rejected; partition workflow steps with `fan_out.over.shards`. See `docs/model_parallelism.md` §3. |
| attempt | every write of a step goes to `.attempts/<step>/<id>/` (§1.1), never into `<step>/`; the id is a per-step counter |
| outputs | before publish, verify every declared output: it exists, is non-empty, parses under its format (JSON, a safetensors header whose promised length is the file's, a `.png`/`.pdf` signature; `.html` is checked for non-emptiness only and recorded as such), and matches its declared `keys`/`columns`; stamp identity on safetensors; record each file's sha256 |
| publish | one atomic rename of the verified attempt onto `<step>/`; a stale `<step>/` from an earlier run is moved aside inside `.attempts/` first and, once the new unit is published and narrated, **retained** as `.attempts/<step>/<n>.superseded/`: its `_step.json` rewritten with `status: superseded`, `disposition: superseded` and `superseded_by` (the replacing attempt, the published record's identity, the step directory): never deleted; until then it is recoverable, and the next run restores it if the new unit never landed or retains it if it did |
| stamping | `_step.json` per step (files, digests, checks, for a protocol step `forwards`: the forward groups the engine actually ran, recorded and never compared, an `execution` block with the step's row bounds and `device`, and `models`: each model's `key`, `revision` and the commit its revision resolved to, recorded and never compared, and an `implementation` block: `tree_digest`, from `runtime_identity()`, asked once per run; compared on `--resume`; for a script step with file references, `input_digests`: the sha256 of each external path or upstream step file by slot, before selecting a value or tensor; reference locators alone do not identify their bytes); `workflow.json` for the run, written in a `finally`: always, `KeyboardInterrupt` included, unless the derived status and the runner's memory disagree or the stream cannot be read (§4.3), when no manifest is written rather than a wrong one; if the manifest itself cannot be written, the step failure is what propagates; inner runs stamp per the IM spec |
| failure | a failed attempt keeps `attempt.json` (start/end, exception type and message, an isolated script's stderr tail bounded to 64 KB, which declared outputs it had written); at most the last 3 failed attempts of a step are kept, partial outputs included: older ones are deleted |
| resume | a step is reused (`--resume`) only when its recorded identity matches (for a protocol step whose document loads an earlier step's output, a compile against the current run tree must also give the digest the record carries: an unfanned step's `identity`, a fanned-out child's `document_digest`; a compile that refuses matches nothing), its record's `implementation.tree_digest` is the running package's (§7; a record without one is never reused), a protocol step's recorded `engine` is the one the step would run under now (the run's engine's name: the third member of the qualification identity, compared, not merely carried; a record without one is never reused; resuming under another `--engine` re-runs every protocol step (`auto` names the reference engine, so it is not "another" beside `pytorch_hooks`); a run whose engine does not cover the step has no engine to compare against and reuses the record: the step that does run is refused `[V13]` with the real message; a fanned-out parent's record is the join, names no engine and is bound to its children's records instead: §2.9), and every recorded file is present with its recorded content digest, and a script step's recorded `input_digests` each match the named input's current bytes (a record with file references but missing or incomplete `input_digests`, or a missing input, is never reused); never on existence alone; never a non-deterministic step unless asked; a conditional or decision step also only while its evidence holds, and **any step declaring `requires_receipt` only while the producer's current `decision.json` still carries the required outcome: a flipped receipt is never reused** (§2.8), a join only while every consumed child's current record is the one it joined (§2.9) |

**A behavioral step's record** (§2.7) is a protocol step's: `document`,
`engine`, `document_digest`, `points`, `point_digests`, `coords` (one axis id
→ value mapping per point, aligned with `point_digests`), `axes`, `files`,
`method`, `execution`: with `identity` the step's own digest and the
`execution` block carrying `decoding` (`mode`, and for a sampled decode
`seed`, `temperature`, `top_p`) beside `batch_rows`, `device` and
`model_source`: the same block, the same one recorder, one more execution
parameter. The `models` list sits beside it, as on a protocol step. Beside them
`checker` (`task`, `task_cfg` when authored, and the spec's
`string_mode`), `split`, `outcomes` (`n` and the count per outcome, plus
`correct`), `thresholds`, `retain` (what was authored, `retained` of `n`),
`cohort` (default for the document's shape, `n`, `below_default`) and
`decision`, the path of `decision.json`.

**A join's record** (§2.9) is a protocol (or behavioral) step's with the
parent's full `points`, `point_digests`, `coords` and `axes`, `identity` the digest of
the parent's entry, a `fan_out` block (`over`, `width`, `children`) and a
`join` block (`require`, `consumed` per child: its `identity`, `points`,
`digests`: `skipped` under `selected`, `n_points`, `n_missing: 0`,
`n_duplicate: 0`), and **no** `engine` or `execution` block; a behavioral
join adds `decision` over the summed counts. **A child's record** is its
kind's record over its points plus `shard`. A skipped parent's children are
skipped with it and appear in `workflow.json` as `skipped` entries; a
`selected` join whose child is skipped is not.

**A nested workflow's steps' records** (§2.10) are their own kinds' records,
at `<step>/<inner>/_step.json`, each with its own identity; the `workflow`
step has no record and no directory of its own beyond `<step>/`, which holds
its steps' directories and their `.attempts/`. `workflow.json` gains a
top-level `nested` map: `{"<step>": {"document", "workflow_digest", "steps":
[…]}}`, one entry per `workflow` step at any depth: record-only, like
`nondeterministic`: never canonical, never compared on `--resume`. The status
table below is unchanged: a nested step carries one of the six words under its
flattened name; the `workflow` step carries none. A nested conditional's
record names its steps as its own document does (`skipped`, `evidence.step`
are local names); the run's `skipped_by` blocks and the stream carry
flattened names.

**A decision step's record** (§2.8) is `type`, `status`, `identity` (its own
digest), `implementation`, `values` (the reference it read), `rule`,
`measured`, `outcome`, `decision_type`, `evidence_identity`, `files`
(`decision.json`) and `decision`, its path. **A conditional's** is `type`,
`status`, `identity`, `implementation`, `predicate`, `scope`, `verdict`,
`evidence` (`step`, `decision_type`, `outcome`, `evidence_identity`) and
`skipped`: the steps its verdict took out of the run: with `files: []`.

A control record contains its declaration and per-point coordinates and
status. Controls that pass by running record zero failures; self-swaps begin
as `not_run` pending certification. Coverage controls also store equivalence
and coordinate-sharing results.

The certifier's `certifies` block names the control, target, kind, per-point
results, counts, and allowed failure rate. `coords_token` maps coordinates to
saved-bundle spelling. Duplicate or foreign rows fail; missing points raise
`ControlFailure`. Exceeding `stop_after_failure_rate` fails the attempt and
retains `controls.json`. The default is 0.0. Each failed point emits an
`instrument_failure` warning.

Dependents record inherited controls and qualification `identity`
`{document_digest, tree_digest, engine}`, per-point status, `n_invalid`, and
`n_points`. Failed controls give `instrument_invalid`; missing matches give
`not_run`; otherwise the point passes. Swept axes match coordinates or fixed
document values, with canonical single-layer forms `L` and `[L]` equal.
Qualification failure blocks dependent steps in the fixed schedule.
