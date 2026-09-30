# Speed-of-light references

`causalab.sol` combines a workload ledger with a hardware profile to estimate
major compute, memory and communication costs on one node with 1–8 GPUs.
Use these analytical references to interpret measured operations. Use the
[before/after harness](measurement.md) to compare code commits on a fixed workflow.

## Quick start

Generate an offline catalog from a hypothetical dense model and GPU:

```sh
uv run python -m examples.sol.build_example
uv run python -m causalab.sol results/sol/input.json --output results/sol/catalog.json
```

The generators need no model downloads or CUDA. Replace the example's model and
hardware before interpreting its results. For the pinned Qwen3.6-35B-A3B text
model and published H100/B200 profiles:

```sh
uv run python -m examples.sol.build_qwen36
uv run python -m examples.sol.build_qwen36 \
    --batch 8 \
    --sequence 512 \
    --updates 200 \
    --suffix-layers 21 \
    --rank 16 \
    --output-dir results/sol/custom
```

Qwen outputs go to ignored `results/sol/qwen36/`: portable inputs, phase/layout
catalogs, model metadata and a Markdown summary. Each catalog retains hardware,
workload, assumptions, resource times, bottlenecks and memory estimates. The
[examples](../examples/sol/README.md) include optional implementation diagnostics.

## Measure achieved percentage of speed of light

The Qwen harness measures inference, activation harvest, interchange, subspace
apply/train and DBM apply/train with a frozen Qwen backbone. It requires an H100
SXM 80 GB or B200 SXM 180 GB and a prepared GPU Python environment.

```sh
# One GPU; the pinned checkpoint must already be cached.
uv run python -m causalab.sol.benchmark_qwen --output results/sol/measured.json

# Eight data-parallel replicas, with trainable featurizer gradient all-reduce.
uv run torchrun \
    --standalone \
    --nproc-per-node=8 \
    -m causalab.sol.benchmark_qwen \
    --batch 16 \
    --output results/sol/dp8-measured.json

# Select operations and reduce the fit budget.
uv run python -m causalab.sol.benchmark_qwen \
    --operations inference dbm_train \
    --updates 10 \
    --source-batches 2 \
    --eval-batches 2 \
    --warmup 1 \
    --repeats 3 \
    --output results/sol/short-measured.json
```

Add `--allow-download` to permit fetching the pinned ~69 GB text checkpoint.
The CLI checks the model architecture, parameter count and GPU variant;
`--hardware h100` or `--hardware b200` asserts an expected family. All ranks must
use identical GPUs on one node, and the global batch must divide evenly across
ranks. Only rank zero writes the JSON report and sibling Markdown table.

Each trial uses deterministic synthetic tokens, BF16 frozen weights, eager full
attention, last-token logits, no KV cache and disabled TF32. Model loading and
input construction are outside timing. Before every trial, including warm-up,
the harness resets featurizers, source/base-prefix caches and optimizer state.
The model and GPU allocator stay warm; cold training captures are timed.

Timing synchronizes devices and aligns ranks, then records the slowest rank's
whole-operation duration, including gradient communication. Count and numerical
checks run outside timing. Warm-up samples are discarded. The reference is
regenerated for the requested geometry and work counts.

The defaults are global batch 8, one inference/harvest/apply batch, or an entire
fit of 100 updates, 10 cold source batches and 20 evaluation forwards. Evaluation
counts include source/intervened pairs, so `--eval-batches` must be even;
`--source-batches` cannot exceed updates. The intervention writes the last token
at block `40 − suffix_layers − 1`. Training uses cross-entropy; DBM adds an L1
soft-mask penalty and temperature annealing. Evaluation follows a fixed update
budget. Tokenization, dataset preparation and protocol orchestration are outside
this microbenchmark.

```
percent_sol = 100 × analytical_reference_seconds / median_measured_seconds
slowdown_vs_sol = median_measured_seconds / analytical_reference_seconds
```

A 2 ms reference measured at 10 ms achieves 20% SOL. This ratio is not sampled
GPU utilization. Values above 100%, including a fastest sample exceeding the
reference, are marked `reference_exceeded`: check algorithm, precision, routing
and ledger assumptions before interpreting them.

Reports preserve raw rank durations, median/min/p95/standard deviation, peak
allocated/reserved memory, observed counts, reference identity and phases,
arguments, hardware and software provenance. Percentiles use linear
interpolation; increase `--repeats` when tail behavior matters. JSON is saved
after each operation, so a later failure preserves completed results and records
the error. OOM is a failed measurement.

The runner supports DP 1–8, TP 1. A custom backend can use
`causalab.sol.benchmark.measure(Case(...), hardware, runtime, dp=..., tp=...)`
with an exact workload and reset/run/validate callbacks. `Case.run` returns
counts checked against `Case.expected_counts`. The runtime must synchronize
participating devices and align ranks; `Runtime.sample` returns `rank_seconds`
and optional memory statistics. Use a TP reference only for an actual TP backend.

## Workload ledgers

Count the expanded, resolved protocol, tokenized shapes and execution plan.
Shared reads can reuse one forward; dependencies can add forwards. The relevant
implementations are `causalab/neural/shared/plan.py` and
`causalab/neural/engines/pytorch_hooks/train.py`.

| Operation | Work to include |
|---|---|
| Inference/probe | Unique forwards, readouts and metrics |
| Harvest | Forward to the tap, captures, reductions and storage transfers |
| Interchange/path patching | Dependency-ordered source and base forwards, gather/scatter |
| Subspace apply/train | Projections, reconstruction and basis parametrization; training adds captures, frozen-suffix backward, feature backward, optimizer and evaluation |
| DBM apply/train | Gates and reconstruction; training adds sparsity, annealing and backward work |
| Generation | Prefill, dependent decode steps, evolving KV traffic and capacity |
| Loaded linear/PCA/SAE | Encoder/decoder GEMMs, residuals and nonlinearities; fitting separately |
| Attention-interior interventions | Materialized scores/probabilities and changed kernel traffic |
| MoE/DeltaNet | Active expert GEMMs, routing, dispatch and recurrent/chunk state |
| Custom functions/scripts | Explicit operator and transfer phases |

Recipes are not an automatic protocol compiler. Supply unique cold source
batches, update counts, evaluation forwards and suffix depth. Include uncached
evaluation source forwards and feature/metric costs. Represent short or
differently padded batches separately; use separate planned-budget and observed
catalogs when early stopping changes counts. See the
[modeling assumptions](speed_of_light_assumptions.md) for formulas and limits.

## Hardware and resource model

| Hardware field | Meaning |
|---|---|
| `flops_per_second` | Per-GPU dense throughput by exact arithmetic mode |
| `hbm_bytes_per_second` | Per-GPU HBM bandwidth, bytes/s |
| `memory_bytes` | Usable per-GPU capacity |
| `link_bytes_per_second` | Per-rank one-way injection rate for the collective topology |
| `collective_latency_seconds` | Latency per ring round |
| `provenance` | Source, SKU, precision, topology and derating |

Label profiles as published peaks or calibrated ceilings. The model assumes
homogeneous GPUs and a uniform ring fabric; topology bottlenecks require an
adjusted injection rate or a more detailed model. Unsupported arithmetic modes
fail rather than borrow another precision's throughput.

A `Phase` is one global-batch execution with a repetition count. FLOPs and
activation traffic divide by DP × TP; weight traffic divides only by TP.
Resident sharded memory divides by TP, replicated memory does not, and transient
memory divides by DP × TP. For a ring all-reduce on `p` ranks:

```
seconds = 2(p−1)/p × payload_bytes / link_bytes_per_second
          + 2(p−1) × collective_latency_seconds
```

TP payloads divide by DP; trainable gradient payloads divide by TP for DP
synchronization, assuming ideal gradient sharding even with replicated parameter
allocation. TP and DP communication times add on the shared fabric.
`sol_seconds` sums the maximum of compute, HBM and communication time per phase,
assuming perfect overlap within a phase and sequential phases.
`serialized_resources_seconds` sums those resources as a sensitivity reference,
not a runtime upper bound.

Memory feasibility `fits` is unknown unless persistent residency alone exceeds
capacity, when it is false and reference time is unavailable.
`estimated_memory_within_capacity` compares only the rough estimate. Saved
activations, caches, workspaces and allocator reserve require a separate liveness
analysis. Runtime profiles use CUDA-reported total capacity.

## Qwen model and peak profiles

The Qwen generator uses the [pinned official configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json):
width 2,048, vocabulary 248,320, 40 blocks (30 DeltaNet, 10 full attention),
256 top-8 routed experts and one shared expert per block. Untied embeddings and
head are counted separately; vision and MTP are excluded. About 34.66B text
parameters occupy 69.32 GB at BF16; all experts remain resident.

`model.json` provides architecture and parameter breakdowns. Each result's
`workload_ledger` contains its operation-specific work. The fixed execution
contract `last_token_cached_prefix_v2` uses last-token logits, captures ending at
the tap, cold base-prefix caches and full-sequence suffix execution. Mixer, MoE
and head stages remain sequential. The Delta estimate uses the chunk64/WY
algorithm; installed fused kernels may require different ledgers.

`causalab/sol/hardware.py` defines these per-GPU published-peak profiles for
SXM GPUs in an HGX/DGX NVSwitch node. Units are decimal; nominal capacity has
no reserve. Ring latency is idealized as zero.

| Resource | H100 SXM 80 GB | HGX B200 SXM 180 GB |
|---|---:|---:|
| Dense BF16 | 989.5 TF/s | 2,250 TF/s |
| Strict FP32 | 67 TF/s | 75 TF/s |
| HBM bandwidth | 3.35 TB/s | 8 TB/s |
| HBM capacity | 80 GB | 180 GB |
| One-way NVLink injection | 450 GB/s | 900 GB/s |

The [H100 specification](https://www.nvidia.com/en-us/data-center/h100/)
quotes sparse BF16; divide by two for dense.
The [Hopper guide](https://docs.nvidia.com/cuda/archive/12.1.1/hopper-tuning-guide/index.html)
identifies NVLink bandwidth as bidirectional; divide by two for injection.
B200 throughput comes from the eight-GPU [HGX table](https://www.nvidia.com/en-us/data-center/hgx/),
capacity from the [DGX B200 guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html),
and HBM bandwidth from the [Lenovo specification](https://lenovopress.lenovo.com/lp2226.pdf).
The profile uses NVIDIA's 75 FP32 TF/s rather than the OEM's 80.

## Run on remote GPUs

The launcher uploads a committed Git snapshot over SSH, starts a detached
supervisor and retrieves reports. It requires configured SSH authentication,
Linux/POSIX, `tar`, a Python 3.10+ control interpreter and a GPU environment with
repository dependencies. Prepare that environment with `uv sync --frozen --no-dev`
in a remote checkout. For SLURM, the job directory, environment and model cache
must be shared with compute nodes. Packages and weights are not uploaded.

Set `GPU_HOST`, `GPU_PYTHON` and `HF_CACHE` to your SSH alias and absolute remote
interpreter/cache paths:

```sh
uv run python -m causalab.sol.remote launch \
    --host "$GPU_HOST" \
    --python "$GPU_PYTHON" \
    --hf-cache "$HF_CACHE" \
    --gpus 1 \
    --receipt remote-job.json \
    -- \
    --warmup 1 \
    --repeats 5

uv run python -m causalab.sol.remote status --receipt remote-job.json
uv run python -m causalab.sol.remote wait --receipt remote-job.json
uv run python -m causalab.sol.remote fetch \
    --receipt remote-job.json \
    --output-dir results/sol/remote
uv run python -m causalab.sol.remote cancel --receipt remote-job.json
```

For SLURM, add `--scheduler slurm --gpus 8 --time-limit 02:00:00` and an absolute
shared `--remote-root`. One `srun` task launches one torchrun process per GPU;
partition, CPU and memory settings use cluster defaults. `running` can include
queue time: inspect `run.log`. `--timeout-seconds` bounds queue plus execution
time (default four hours); `--time-limit` bounds the allocation.

`--revision` defaults to `HEAD`; only tracked content at that commit is uploaded.
Use `launch --dry-run` to inspect the command without contacting the host.
The receipt is written before launch: retain it to check status after an
uncertain SSH reply instead of resubmitting. Each job uses a new directory.

Flags after `--` go to the benchmark, except the launcher owns `--output`.
`--gpus` sets the actual DP worker count. Downloads remain disabled unless
`--allow-download` is passed after `--`. `--hf-cache` sets remote `HF_HUB_CACHE`;
`--env-script` instead sources a remote Bash configuration. These options are
mutually exclusive. `--control-python` selects the SSH host's control interpreter.
Use absolute remote paths, not `~`; the default job root is relative to the
remote home. Normal SSH configuration and host-key checks apply.

The supervisor survives disconnects, but not reboots or policies that terminate
background login processes. Cancellation terminates its subprocess group and
escalates after 20 seconds; poll status to confirm completion. Fetching works
during execution or after failure and retrieves available reports, job/state
metadata and the last 1 MiB of logs. Terminal failures return nonzero status.
