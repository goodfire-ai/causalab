# Reusable cyclic manifold steering

Generate workflow documents from one experiment configuration. The
[weekdays configuration](manifold_weekdays.json) matches the fitted rank, model
snapshot, prompts, and two writes in the [Figure 4 demo](manifold_fig4.md).

```bash
uv run python demos/papers/build_manifold_workflows.py \
  demos/papers/manifold_weekdays.json --data-root demos/papers/artifacts/data \
  --documents /path/to/documents --runs /path/to/runs

uv run causalab run /path/to/documents/manifold_fit.json \
  --data-root demos/papers/artifacts/data --out /path/to/runs \
  --device cuda --batch-rows 16 --resume

uv run causalab run /path/to/documents/path_steering.json \
  --data-root demos/papers/artifacts/data --out /path/to/runs \
  --device cuda --batch-rows 16 --resume
```

The generator places the workflow scripts beside the documents so CausaLab
can hash them as part of the experiment. Regenerate after updating the code.
`manifold_comparison.json` runs both parts in one run tree. The split form
fits every pair once, then imports a verified set of files from the configured
runs directory. Each steered prompt is evaluated under both interventions at
every waypoint. A steering-only dataset or a subset of pairs can reuse the
same fitting run. For independent shards, generate separate document directories
with disjoint `pairs` lists and separate `--out` directories. Keep the
generator's `--runs` pointed at the original fit. Collect the two method rows
per pair from each shard's `energy/energy.json` and compute pooled comparisons
across the joined pairs.

The configuration names a model snapshot, layer, dataset paths relative to
the data root, the fit table's label field, cyclic class order, PCA rank,
waypoint count, and answer token forms. Both datasets need `input` and
`split` fields per row. Use `"split": "all"` for an undivided table.
The fit dataset must cover every label. Use distinct alphanumeric labels and
single-token answer forms. Fit and steering prompts can be separate. The
angular ordering is estimated from the first two activation PCs. Review the
saved centroids to see whether their ordering matches the intended cycle.

The default chord uses the same PCA featurizer as the spline. Both writes
then preserve the target prompt's residual outside the fitted subspace.
`projected_chord: false` reproduces Figure 4's full-residual chord for
comparison with the paper. The periodic fit supports one cyclic coordinate,
and isometry uses centroid pairs and numerically sampled arc lengths.

The steering output includes a trajectory table, a per-pair energy and
coherence table, an aggregate distance comparison, and coherence and
isometry summaries. Energy is the sum of Bhattacharyya distances over the
path, averaged over prompts. Higher coherence means more probability mass
on the configured answer tokens. Isometry correlates activation-space
lengths with behavior-space lengths. Paired tests use the selected pairs
as observations. Figures are post-hoc consumers of these tables.

```bash
uv run python demos/papers/plot_manifold_workflows.py \
  --fit /path/to/runs/manifold_fit --steering /path/to/runs/path_steering \
  --out /path/to/manifold.png
```

Pass `--steering` more than once to plot disjoint steering shards together.

## Unity reference

The [Unity reference record](artifacts/figures/manifold_fig4/unity_reference.json)
contains the pooled energies for all 21 pairs, the snapshot, the tested commit,
and the Slurm job IDs. The fit ran once on a single L40S, followed by three
steering jobs of seven pairs each and a CPU collection job. Overall energy was
0.3293 for the spline and 0.9262 for the chord. Both means fall within the
recorded absolute tolerances against the committed Figure 4 replication.
The chord wins every neighboring pair and the spline wins every farther pair.
This records drift against the committed replication using the same prompt
pool and the full-residual chord.
