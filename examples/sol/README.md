# Generate SOL references and diagnostics

The source definitions contain the hypothetical dense example, published H100/B200
profiles and pinned Qwen3.6-35B-A3B architecture. Generated catalogs and collected
measurements belong in ignored `results/` or an external artifact store.

## Quick start

Run from the repository root:

```sh
uv run python -m examples.sol.build_example
uv run python -m causalab.sol results/sol/input.json --output results/sol/catalog.json
uv run python -m examples.sol.build_qwen36
```

The first generator accepts `--output`; the Qwen generator accepts `--output-dir`
and geometry/budget options. Its default `results/sol/qwen36/` contains portable
inputs, full phase/layout catalogs, model metadata and a Markdown summary.
See the [methodology and hardware values](../../docs/speed_of_light.md) and
[modeling assumptions](../../docs/speed_of_light_assumptions.md).

## Diagnostics

Diagnostics use the installed implementation. The H100 probe needs a
prepared CUDA environment and the pinned checkpoint in its local model cache;
it runs eager, whole-forward inference and is not the benchmark execution contract.
The Delta probe runs on CPU. Neither produces a benchmark median.

```sh
mkdir -p results/sol/probes
uv run python -m examples.sol.audit.h100_probe > results/sol/probes/h100.json
uv run python -m examples.sol.audit.assumption_sensitivity \
  --evidence results/sol/probes/h100.json > results/sol/probes/sensitivity.json
uv run python -m examples.sol.audit.delta_backward_probe > results/sol/probes/delta.json
```

Keep generated outputs outside source control. Run `uv run pytest tests/sol`
for the synthetic-routing and tiny-model checks.
