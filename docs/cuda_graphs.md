# CUDA graphs

Enable CUDA graph replay with `causalab run … --device cuda --cuda-graphs` or
`PytorchHooksEngine(device="cuda", cuda_graphs=True)`. The option also applies
to workflows. Authored documents and canonical digests stay the same.
`torch.compile` is a separate option. Eager execution is the default.

The supported path uses a frozen, unquantized Qwen3 or Qwen3.6 text model with
eager attention. It supports last-token block-output or logits reads and at
most one swap write with an identity, gate, or Cayley subspace featurizer.
Qwen3.6 requires grouped experts. Other workloads run eagerly and log the reason
at INFO level. Under `--parallel`, graphs serve tensor and expert parallelism
and data parallelism over points ([Multi-rank execution](#multi-rank-execution)).

## Training and evaluation

A cohort with eligible members uses one training graph. Each member gets
`min(train.batch.pairs, training row count)` fixed rows. A short minibatch repeats
its last row with zero loss weight. Members that stop early retain their slots
and stop updating. Optimizer steps, scoring, annealing, and stopping decisions
remain in the Python loop.

Padding changes MoE expert group sizes and can change bf16 rounding. Full,
active slots match the eager cohort bit for bit. The graph shares source
forwards through the campaign store and resumes below the shallowest write
from a separate prefix buffer. Prepared masks, positions, and padded prefixes
are cached per minibatch and copied into captured storage on later steps.

Training requires one input role and exactly one forward group per member.
An ineligible member or an authored `fit_rows` below the required slot count
keeps the cohort eager. An incompatible layout or training allocation failure
releases graph buffers and retries the remaining work eagerly. Prefixes already
stored from padded frames keep their rounding after this fallback.

Evaluation captures a separate graph on the second use of a member and row
layout. It retains row budgets, shared sources, prefix resume, and evaluation
modes. The graph reads existing fit parameters; gate temperatures are staged
separately. Scoring runs on the device and copies the selected answer columns
to the host. Reads are released before the next replay.

Each fit captures one evaluation layout. A change to its window, members, or
split releases the graph and continues eagerly. Single-use evaluations run
eagerly.

The following features also use eager execution:

- Controllers, trajectory saves, phased training, or a `constraint` term.
- Drawn counterfactual roles, stochastic or budget gates, and dead-unit rules.
- Soft-accuracy or JS objectives, objective-weight annealing, and continuation
  decoding.
- Inference batches that need splitting under `batch_rows`.

The low-rank Cayley map uses an unchecked inverse during capture to avoid a
host synchronization.

Captures belong to an execution request and are released on completion or
failure. Compatible fits can reuse them within the request. Training allocation
failures during capture or replay trigger eager recovery. Other capture errors
surface to the caller; an invalidated capture can leave CUDA unable to recover.
Evaluation batches remain subject to GPU memory limits.

## Multi-rank execution

Add `--cuda-graphs` to a parallel run:

```bash
uv run causalab run patch.json \
    --engine pytorch_hooks \
    --data-root data \
    --artifacts-root . \
    --out runs/patch \
    --device cuda \
    --parallel tp=2 \
    --cuda-graphs
```

Each rank captures its own graphs on its own device and pool.

| Geometry | Graphs | Why |
|---|---|---|
| `tp=N`, `ep=N` | Captured | The model's collectives are DTensor redistributes, which capture like any kernel |
| `dp=N` over points | Captured | Replicas exchange nothing inside a forward |
| `dp=N:rows` | Eager | The eager step scales each replica's loss by its share of the rows before the gradients are summed; the captured objective does not |
| `pp=N` | Eager | Stages broadcast captures and agree fire counts on the host after every forward |
| `cp=N` | Eager | Chunks gather keys and values and hand off DeltaNet state between ranks |

An eager geometry logs its reason at INFO level. Block outputs and logits are
replicated under tensor and expert parallelism, so reads and writes need no
gather.

The router's input-gradient sum under `ep` and, when `tp` exceeds the KV-head
count, the KV-head replication sum run inside the captured backward.
`TorchCollective` skips its shape header while the stream is capturing,
because the header is a host copy. The check is not lost: every rank captures
the same sequence it just ran eagerly in warm-up, where the headers were
checked. Collectives that need a host value (`broadcast`, `agree_*`,
`barrier`) raise `CaptureUnsafe` during capture instead of invalidating it.
Gradient averaging, the row budget's agreements and scoring run between
replays.

With every slot full, graphs and the eager path at the same geometry write
byte-identical outputs (`tests/golden/test_multirank_cuda_graphs.py`:
`tp=2`, `dp=2` and `tp=2,dp=2` on Qwen3-4B, `ep=2` on Qwen3.6-35B-A3B).
Padded slots change bf16 rounding as on one device, and under `dp` over
points a replica's padding can differ from the world-1 cohort's. Graphs do
not change how geometries compare with each other
([model parallelism §11](model_parallelism.md#11-known-limits)).
Graphs can raise peak memory per rank. Measure the peak at the target
geometry, for example with the fits in `benchmarks/cuda_graphs/standard.json`,
before a long run.

Under `tp` or `ep`, an out-of-memory failure during capture or replay ends the
run with `DistributedOutOfMemory`, since one rank turning eager would leave
its peers replaying different collectives. Reduce `train.batch.pairs` or
`--fit-rows` and restart. A layout mismatch still falls back to eager, because
every rank of the group sees it on the same step. Data replicas over points
keep the single-device fallback.

NCCL's watchdog does not track collectives inside a replayed graph;
[Hung replays](#hung-replays) describes the deadline that bounds them.

## One pool per engine

`PytorchHooksEngine.graph_pool` owns one `GraphPool` and one capture stream.
Training, evaluation, and inference graphs share its working memory. Each graph
also retains its live outputs. A common stream lets the allocator reuse blocks
across captures.

The pool opens on the first capture and lasts for the engine's lifetime.
Request completion releases graph holders while the pool retains memory for
later requests. Direct calls without an engine pool create their own pool and
release it after the graph holders close. Bucket count and capture memory have
no preset limit.

Warm-up routes every allocation on the device into the pool through
`GraphPool.allocating`. This includes allocations from the autograd device
thread. The ordinary cache is emptied first. Warm-up initializes CUDA libraries
and kernels and supplies the blocks that capture will reuse. Evaluation can
skip a separate warm-up because its first eager pass uses the same storage.

The pool uses `use_on_oom=True`, allowing eager allocations to borrow free
blocks under memory pressure. A capture or replay that runs out of memory
releases its graphs. The pool closes when its final graph holder releases it.
Remaining captures in that fit use private pools; the engine opens a new pool
for its next request. Closing a pool while a graph remains alive logs a warning
and resets the graph. Replaying that graph raises an error.

Shared memory requires these rules:

1. Replay graphs serially on the current stream.
2. Consume or copy each result before another graph replays on the pool.
   The optimizer consumes gradients immediately. Inference clones captures
   and routing; evaluation scores and releases its reads.
3. Allocate persistent inputs outside capture. This includes tokens, masks,
   positions, labels, padding weights, staged operands, source captures,
   cohort frames, and prefix buffers.

Graph holders close in each fit's `finally` block. A bank retained for a
compatible fit replays only within that fit. An engine serves one request at a
time. Other threads must avoid device allocations during warm-up, since the
allocation routing covers the whole device.

## Hung replays

Replaying a graph enqueues its NCCL kernels without the `Work` object that
`CAUSALAB_COLLECTIVE_TIMEOUT` times. If a peer skips a replay or reaches a
different collective, the replaying rank waits forever, and the heartbeat has
no lost rank to name.

In a world above one rank, each replay therefore records an event that the
heartbeat thread polls without blocking. A replay unfinished after
`CAUSALAB_COLLECTIVE_TIMEOUT` ends the rank with exit status 1 and
`refused: [P4] … a CUDA graph replay enqueued N s ago on rank r of w has not
completed`. A lost peer is reported first when both apply.

The heartbeat thread makes no other CUDA call. It pauses while its rank
captures, because querying a completed event from another thread invalidates
the capture. It leaves completed events for the replaying thread to destroy:
destroying one waits on a driver lock held by a stuck replay, and the waiting
thread would hold the GIL.

## Compilation caches

CUDA graphs are captured separately in each process. Compiled kernels can be
shared on disk. Triton and TileLang compile FLA kernels; a runner that compiles
the forward with `torch.compile` also produces Inductor artifacts.

### Configure a root

The cache is off unless `CAUSALAB_COMPILE_CACHE` names a directory. With the
variable unset or empty, each compiler keeps its own default cache. The
directory can already exist, for example one that the volume owner created
with mode `2770` for several users, or the loader creates it on first use:

```bash
export CAUSALAB_COMPILE_CACHE=/path/to/compile-cache
causalab run benchmarks/cuda_graphs/standard.json \
    --engine auto \
    --device cuda \
    --out out/standard

CAUSALAB_COMPILE_CACHE= causalab run benchmarks/cuda_graphs/standard.json \
    --engine auto \
    --device cuda \
    --out out/standard
```

Both engine loaders configure the cache before compiling kernels on CUDA.
A toolchain signature includes PyTorch and its CUDA runtime, Triton, TileLang,
FLA, Transformers, the CPython ABI, and the GPU name and compute capability.
Each signature gets separate directories:

```text
<root>/<signature>/toolchain.json
<root>/<signature>/triton/               TRITON_CACHE_DIR
<root>/<signature>/tilelang/             TILELANG_CACHE_DIR
<root>/<signature>/inductor[-<policy>]/  TORCHINDUCTOR_CACHE_DIR
```

Compilers publish files through atomic renames, which permit concurrent
writers. NFS attribute caching can briefly hide a new file and cause duplicate
compilation. The manifest is also written atomically; a later run replaces a
truncated manifest.

A configured root overrides `TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`, and
`TORCHINDUCTOR_CACHE_DIR` with a warning. If the root cannot be created or
written, the loader warns and keeps compiler defaults.

### Permissions and trust

A root with group-write permission uses mode `2770` and the root's group for
new directories. Configuration widens the process umask to allow group
writes. This also affects files the process writes later.
Personal roots retain the process umask. Existing directories keep their modes.

Compiler artifacts contain executable code. Every writer to a shared root must
be trusted by every job that reads it. Restrict writes to the intended group.
Causalab warns when a root is world-writable.

### Compile policies and measurements

A runner that compiles through Inductor with a retained-operator policy puts
the policy in its Inductor directory name, such as
`inductor-aten-cumsum-exp-sum-<hash>`. Custom lowering registrations require
separate directories because Inductor's own key omits them.

FLA configuration files and Hugging Face model weights use their own caches.
The shared compile root stores compiler output.

Loaders log `compile cache: <root>/<signature>`, and
`<root>/<signature>/toolchain.json` records what the signature stands for.

For startup measurements, record the initial compiler-cache state and include
model loading, capture, evaluation, and saving. A cache hit still requires
artifact loading and specialization checks. Filesystem and page-cache state
also affect timing. Use a new empty root to measure fresh compilation and keep
active shared roots intact. Check numerical behavior after changing a compile
policy.

## Verification

On a CUDA host with cached model weights:

```bash
uv run pytest tests/golden/test_cuda_graphs.py tests/golden/test_graph_cohort.py tests/golden/test_graph_reuse.py tests/golden/test_cayley_capture.py -q
```
