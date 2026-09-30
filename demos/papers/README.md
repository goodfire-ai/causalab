# Paper replications

One page per reproduced figure, beside one shared `protocols/`, `workflows/`
and `artifacts/` folder. A package uses causalab as an installed library: it
ships its own tables, its own documents and its own figure script, and no task
package under `causalab/tasks/`. The
[demo index](../README.md#paper-replications) lists the packages with what each
reproduces and what it needs. Every package follows
[the paper replication guide](../../docs/paper_replications.md).

## The series

Six pages make a tutorial series, one method per page. Read them in this
order:

1. [Zero ablation](rome_fig1_knockout.md)
2. [Causal tracing](rome_fig1.md)
3. Causal models and distributed alignment search, pending
   (`mcqa_symbol` or `mcqa_pointer`)
4. Desiderata-based masking, pending (one of
   `mcqa_components_dbm`, `function_vectors_fig3a_dbm`, `addition_heads_dbm`,
   `arithmetic_neurons`)
5. [Path patching](ioi_fig3b.md)
6. [Interchange at several sites at once](lookbacks.md)

The other pages reproduce one published result each, outside the series.

## Run a package

From `demos/papers/`, these commands install causalab into a fresh
environment, check the documents without loading weights, run the workflow into
`artifacts/output/<name>/`, and draw the figures into
`artifacts/figures/<name>/`. `--tokenizer` loads each model's tokenizer and
resolves token positions and metric answers with it, as the run does before
the weights load:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "causalab @ git+https://github.com/goodfire-ai/causalab.git"
causalab validate workflows/<name>.json \
    --engine auto \
    --data-root artifacts/data \
    --tokenizer
causalab run workflows/<name>.json \
    --engine auto \
    --data-root artifacts/data \
    --out artifacts/output \
    --device cuda
python workflows/scripts/<name>/<fig>_figure.py
```
