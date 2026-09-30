# Compare hypotheses on saved pairs

The [CPU example](../demos/hypothesis_testing/hypothesis_testing.md) exports pair
tables and frozen predictions. Use the pair table in intervention documents;
its `label` gives the target causal model's answer under the intervention.

## Save neural outputs

Save `top_k` with `k: 1` from the final, intervened `lm_head` read. Save
`token_logits` over the declared answer vocabulary as well. The top-1 token
must come from the full vocabulary. Restricting the argmax to answer tokens
can overstate accuracy. During validation, score an exact `match` against
`label`.

## Compare predictions

`causalab.analysis.compare_hypotheses.compare_saved_outputs` takes the pair
table, prediction table, native top-1 metric rows, and the run's tokenizer.
It resolves answers with the token conversion used by exact-match metrics:
each answer form is tokenized as written.

The workflow entry point accepts `pairs`, `predictions`, `neural`, `target`,
`alternatives`, `metric`, `tokenizer`, `tokenizer_revision`, and
an optional `split`. The tokenizer must be available locally. The output,
`comparisons`, is a JSON table with one row per pair and alternative.

Keep the source run manifest and dataset pin beside the table. The comparison
checks row IDs, completeness, and duplicates. Use the manifest to verify which
dataset and tokenizer produced the run.

## Reduce and select

Reduce by run point, family, split, and alternative. `value` is target
agreement minus alternative agreement; its mean times 100 gives the gap in
percentage points. `target_score` and `alternative_score` retain absolute
accuracy. Report excluded pairs and their reasons alongside scored counts.
Filter `distinguishing` before computing results on distinguishing pairs.
An empty subset has an unavailable score.

As general research guidance, average validation accuracy within each family
for each location and rank. Give broad accuracy weight **0.1** and divide the
remaining 0.9 equally among narrow families. Choose the smallest rank within
**0.02** of that location's best weighted validation accuracy. These values
guide study design; specify them in the study's selection rule. Compute
family means before applying the weights, and freeze the chosen fit before
testing.

Use `causalab.workflow.scripts.reduce` and `causalab.workflow.scripts.select`
for these operations. Compare the selected fit with random subspaces at the
same site and rank. Keep the seed and fit identity with the dataset pin and run
manifest.

## Saved logit analysis and report consumers

`causalab.analysis.hypothesis_metrics.read_token_logits(native, item, tokenizer,
paths)` reads the saved `token_logits` table associated with `item.source_metric`.
`native` contains the compiled method, selected point digest and coordinates,
receipt, pair rows, and resolved run paths. Set `item.logit_metric` when more
than one matching token-logit metric exists. The function returns per-example
values and a source record; `paths` receives each existing table read. It checks
point identity, coordinates, and the declared answer vocabulary.

`score_rows(rows)` reduces saved comparison rows for a chosen population.
`mix_scores(family_scores, family_weights, population_counts)` conditions declared
family weights on the retained pairs and metric eligibility. Callers supply the
scientific mixture. Missing required logits or unknown eligibility leave the
mixture unavailable, while family means retain their measured values and missing
counts. Render those family values as partial when measurements are absent.

`causalab.analysis.selection.smallest_near_best(scores, costs, tolerance=0.02)`
returns the least-cost indices within the tolerance of the best finite score.
The absolute boundary tolerance is 1e-12. Native workflow `select` uses the same
operation and preserves input order to resolve ties. Consumers can verify a saved
choice against the returned indices without selecting from test scores.

`causalab.analysis.hypothesis_artifacts.read_frozen_bundle(run, name, slot)` reads
a pinned bundle entry and returns its tensor and native identity.
`resolve_position(value, method)` resolves a named position before normalizing
an all-token position. `frozen_dbm_identity(gates, point)` binds the component
inventory, mask, and SHA256/entry identities; local file paths and evaluation
splits do not change it. DBM export adds this value as `artifact_id` to each point.
A consuming report must also match its experiment and intended evaluation split.
