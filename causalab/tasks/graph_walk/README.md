# Graph Walk

Replicates the in-context-learning-of-representations setup from Park et al., ICLR 2025. The model is shown a long random walk over a graph rendered as `concept|concept|concept|...|concept|` and is expected to predict a concept that's a graph neighbor of the last visited node. The interesting causal claim is that the model's residual stream geometrically reflects the *graph structure* even though the prompt is just a flat sequence of tokens.

```
time|lot|hint|film|mind|den|deer|rain|size|plot|...|cup|
```

If `cup` is a node on a ring graph and its neighbors are `corn` and `hall`, the model is correct iff its top continuation is `corn` or `hall` (`raw_output` is the list of all valid neighbors).

## Graph Types

`config.py::GraphWalkConfig` is a factory across five graph types:

| `graph_type` | Shape | `graph_size` × `graph_size_2` | Periodic dims |
|---|---|---|---|
| `ring` | 1-D cycle | `n` (single param) | 0 |
| `grid` | 2-D rectangular grid | `m × m` | none |
| `hex` | 2-D hex tiling | `m × m` | none |
| `cylinder` | grid wrapped on one axis | `n_ring × n_height` | 0 |
| `torus` | grid wrapped on both axes | `m × n` | 0, 1 |

`graph_size_2` is required for `cylinder` and `torus`; the others auto-fill it from `graph_size`. Select the graph with `--set` flags when you build a table (see [Running](#running)).

Concepts (the strings rendered for each node) come from `DEFAULT_CONCEPTS` in `config.py` — a hand-curated list of ~370 short English nouns, shuffled with seed 42, that are likely single tokens across BPE vocabularies. The first `n_nodes` are used; you can pass a custom `concepts=[...]` to `GraphWalkConfig` to override.

## Causal Model

```
node_coordinates ──> walk_sequence ──> raw_input
                                            │
                                            └──> raw_output
```

| Variable | Role | Notes |
|---|---|---|
| `node_coordinates` | input | The coordinates of the node where the walk *ends*. For a ring this is `(0,)`, `(1,)`, …; for a grid it's `(row, col)`. |
| `walk_sequence` | computed | Random walk of `context_length` steps that *ends at* `node_coordinates`. Generated forward from the target then reversed, with optional `no_backtrack`. |
| `raw_input` | computed (lazy) | The walk rendered as `concept|concept|...|concept|`. Trailing separator is intentional — it puts the model at the position it would be in *after* observing the final concept, predicting the next one. |
| `raw_output` | computed (lazy) | List of concept strings that are graph-neighbors of the final node. Used as the set of correct continuations. |

The lazy mechanisms matter — `raw_input` for a 2048-token walk is expensive to format, and analyses that only need `node_coordinates` shouldn't pay that cost.

`node_coordinates` is also the `TARGET_VARIABLE` and the natural embedding (identity — coordinates are already numeric vectors). For periodic graphs (`ring`, `cylinder`, `torus`), `GET_PERIODIC_INFO` derives the period from `graph.periodic_dims` so isometry analyses know which axes wrap.

## Counterfactuals

`counterfactuals.py::generate_dataset(model, n, seed)` cycles through `node_coordinates` for even coverage: example `i` uses coordinate `coords[i % n_nodes]` as the base, and a random distinct coordinate as the counterfactual. Each call produces a new walk (sampling fresh ICL context) so the same `(base, cf)` coordinate pair across two calls won't yield identical prompts.

`make_walk_steering_examples(walk_texts)` is a separate utility that wraps pre-computed walk strings as `{input: {raw_input: text}, counterfactuals: []}` for steering pipelines that supply their own walks. It's not part of the standard analysis flow.

## Token Positions

Just one named position:

| Name | Description |
|---|---|
| `last` | Final prompt token — the trailing separator, which is the position at which the model is predicting the next concept. |

This is the only position analyses need: graph-walk experiments measure activations *after* the model has consumed the whole walk, so per-concept positions inside the prompt aren't intervention targets.

## Running

No document in this repository runs this task yet, and the task ships no table. Build a table with `scripts/build_task_dataset.py`, name it in the `data` block of a document, and run the document with `causalab run`. For the 5×5 grid:

```bash
uv run python scripts/build_task_dataset.py \
    --task graph_walk \
    --set graph_type=grid \
    --set graph_size=5 \
    --n 64 \
    --seed 0 \
    --split all \
    --out data/grid_5x5.json
uv run causalab run <document> \
    --engine auto \
    --data-root data \
    --out runs/grid_5x5
```

For the 9×9 cylinder, use `--set graph_type=cylinder --set graph_size=9 --set graph_size_2=9`. [Running experiments](../../../docs/running_experiments.md) shows how to write, validate and run a document.

### Note on the legacy `representation_emergence` experiment

A standalone script for sliding-window concept-centroid emergence analysis (replicating Park et al. Fig. 2) used to live under `experiments/representation_emergence.py`. It has been removed — that workflow is its own research question and belongs in its own `analyses/<emergence>/` package, not as a per-task script. Until that package exists, the script is recoverable from git history; new work should not import from it.

## Files

| File | Role |
|---|---|
| `config.py` | `GraphWalkConfig`, `DEFAULT_CONCEPTS`, `TASK_NAME` |
| `graphs.py` | `Graph`, `build_graph`, per-shape constructors (`make_grid_graph`, `make_ring_graph`, `make_hex_graph`), `random_walk`, `random_neighbor_pairs` |
| `causal_models.py` | `create_causal_model` plus the `GET_*` accessors used by `tasks/loader.py` |
| `counterfactuals.py` | `generate_dataset` (alias for `generate_graph_walk_dataset`), `make_walk_steering_examples` |
| `token_positions.py` | `create_token_positions` (just `last`) |
| `data/` | none yet — the answer is a set of neighbours (`raw_output`), which the v1 row vocabulary cannot carry; see `causalab/tasks/README.md` §2 |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |

Walk randomness is an explicit `walk_seed` input. The generator supplies it,
and direct `model.new_trace` calls must supply it alongside `node_coordinates`.
The graph retains one `walk_sequence` variable. Re-evaluation and interventions
hold the seed fixed; `enumerate_inputs(noise={"walk_seed": 0})` fixes it explicitly.
