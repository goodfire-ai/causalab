# CUDA graph workflow benchmark

`standard.json` runs a Qwen3.6-35B-A3B bf16 workflow with localization, DAS,
Desiderata-Based Masking (DBM), random-subspace controls, and mean ablation.
Its fixed months dataset contains 39 training pairs and 25 test pairs from
64 unique inputs, with seed 0 and a 60/40 split. Training settings come from
the production method files.

With cached weights, run the workflow from the repository root on a CUDA
host, once eagerly and once with CUDA graphs, into fresh output directories:

```bash
HF_HUB_OFFLINE=1 uv run causalab run benchmarks/cuda_graphs/standard.json \
    --engine pytorch_hooks \
    --data-root benchmarks/cuda_graphs/data \
    --device cuda \
    --out out/qwen36-eager
HF_HUB_OFFLINE=1 uv run causalab run benchmarks/cuda_graphs/standard.json \
    --engine pytorch_hooks \
    --data-root benchmarks/cuda_graphs/data \
    --device cuda \
    --cuda-graphs \
    --out out/qwen36-graphs
```

Compare the two runs' artifacts before claiming numerical agreement.

Pass `--fit-rows N` to test bounded cohorts. If batching changes early stopping,
also compare runs with fixed update counts. Keep study variants and generated
results outside the repository. See the [cache guide](../../docs/cuda_graphs.md#compilation-caches).
