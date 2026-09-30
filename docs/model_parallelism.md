# Model parallelism

The `pytorch_hooks` engine supports data, pipeline, tensor, expert, and context
parallelism for inference, interventions, and featurizer training. Select the
geometry with `--parallel`; use `dry-run` to check it before loading weights.

This guide describes the supported interface, runtime contracts, and tests.
For general launch and Slurm examples, see [Running experiments](running_experiments.md).

## 0. What this is

Parallelism distributes a run across processes, with one rank per device.
Choose an axis according to what needs to be divided:

| Axis | Flag | What it divides | Main use |
|---|---|---|---|
| Data, points | `dp=N` or `dp=N:points` | A campaign's selected points | Run independent points on model replicas |
| Data, rows | `dp=N:rows` | Each training minibatch's rows | Distribute a fit across replicas |
| Pipeline | `pp=N` | Contiguous ranges of model layers | Fit the model across devices |
| Tensor | `tp=N` | Attention and dense-MLP projections | Reduce dense-model weight memory per rank |
| Expert | `ep=N` | Routed experts | Reduce MoE weight memory per rank |
| Context | `cp=N` | Each forward's padded position axis | Distribute sequence activations |

For a MoE whose experts dominate its weights, start with `ep` or `pp` for
memory relief. Tensor parallelism may leave most of that model replicated.
Pipeline stages execute sequentially; adding stages reduces weight memory but
does not overlap their computation. Data parallelism replicates the model.

For an existing intervention specification and its data:

```bash
uv run causalab dry-run patch.json \
    --engine pytorch_hooks \
    --data-root data \
    --artifacts-root . \
    --parallel tp=2

uv run causalab run patch.json \
    --engine pytorch_hooks \
    --data-root data \
    --artifacts-root . \
    --out runs/patch \
    --device cuda \
    --parallel tp=2
```

The second command starts two local ranks. The model must support the selected
axis and the node must expose enough devices. All axes default to one; at
world size one, the engine does not initialize `torch.distributed`.

## 1. Spec anchor

Parallelism is engine execution configuration under
[Intervention protocol §8](intervention_protocol.md). It does not change the
document vocabulary, canonical form, digests, or qualification stamps.

The same document at `tp=4` and on one device belongs to the same campaign.
The receipt records the execution geometry (§9). This identity rule does not
promise identical floating-point results: sharding can change reduction order,
and bf16 routing decisions and training trajectories can differ (§11).

## 2. Geometry

`ParallelGeometry` is defined in `causalab/protocol/parallel.py`:

```python
@dataclass(frozen=True)
class ParallelGeometry:
    data: int = 1
    pipeline: int = 1
    context: int = 1
    tensor: int = 1
    expert: int = 1
    data_mode: DataMode = "points"
```

The derived sizes are:

```text
model = max(tensor, expert)
world = data * pipeline * context * model
```

The mesh is `(data, pipeline, context, model)` in row-major order. Tensor and
expert groups are contiguous subgroups of the model dimension, and both sizes
must divide `model`. Thus `tp=2,ep=4` uses four ranks, not eight. When `tp` is
smaller than `model`, multiple tensor groups repeat the attention computation.

The CLI accepts comma-separated axes, each named once with a positive integer.
Only `dp` accepts a mode suffix. `PytorchHooksEngine(parallel=geometry)` takes
the same geometry through the Python API; a multi-rank engine also needs a
launched process group or explicitly supplied collective infrastructure.

The torch-free `check(geometry, info)` function checks the registry facts:

- `tp` divides the query-head count and either divides the KV-head count or
  is a multiple of it (§6.6).
- `ep` divides the expert count and requires an expert plan.
- `pp` does not exceed the layer count.
- Requested tensor and expert styles must be supported; planless families
  such as `gpt2` reject parallel execution.

Document checks reject unsupported row splits and context configurations
(§8.3–8.4). Errors use `P4` and identify the relevant `--parallel` axis.
Some checks require runtime information, such as a frame's length or a tied
embedding and output head, and run when that information is available.

### Memory pre-flight

`protocol/parallel_memory.py` builds a placement table from checkpoint headers,
parameter targets, and the registry plan. `estimate_resident` computes each
rank's weight bytes; `RULE` adds empirical headroom (§11). The table must cover
the whole model before pipeline placement, so layer ranges are computed from
the full tower.

Each CUDA rank checks that its resident weights fit in driver-free memory plus
unused allocator cache before reading weights. If only the empirical footprint
exceeds available memory, loading continues with a warning; the estimate cannot
reject a run. A resident-weight rejection identifies the rank, device, weights,
estimate, available and total bytes, and up to two estimated alternatives.
The search considers the same world size and twice that size, preferring
fewer tensor ranks, then fewer expert ranks, with pipeline stages supplying the
remaining parallelism.
These alternatives use `dp=cp=1`; they do not preserve an existing data or
context split.

The spawn parent checks the device count without opening CUDA contexts on its
children's devices. Off CUDA, or if the memory query fails, the pre-flight does
not reject the load. `dry-run --parallel` reports estimates from cached headers
without opening a device; it reports `memory: undecided` when the required
checkpoint facts are unavailable. The estimate is a heuristic, not an OOM
safety guarantee.

## 3. Process model

Every rank runs the same program over the same document. Collectives inside
loading and execution coordinate the ranks. The launcher creates one `Mesh`
per process; the engine uses it for both its `TorchCollective` and its loader's
`Sharding`. All ranks create process groups in the same order, including groups
they do not belong to.

### Launch

Without `WORLD_SIZE`, a geometry above world size one starts local children
using the standard library's `spawn` method. The parent imports no torch, waits
for the children, and reports a failed child's rank and exit status. Each child
receives `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and `MASTER_*` settings.

With `WORLD_SIZE`, the process joins an existing world, such as one started by
`torchrun`. The geometry must match that world's size. CUDA ranks select
`cuda:LOCAL_RANK` before initializing NCCL; CPU ranks use gloo. A `--device`
list is not accepted with a multi-rank geometry.

```bash
uv run torchrun \
    --nproc_per_node=2 \
    -m causalab.cli run patch.json \
    --data-root data \
    --artifacts-root . \
    --out runs/patch \
    --device cuda \
    --parallel tp=2
```

Local spawning defaults `OMP_NUM_THREADS` to `1` unless explicitly set. The
parent sets it before children import torch, then restores its environment.
It also defaults gloo to the loopback interface for a local rendezvous. Joined
worlds retain their launch environment; multi-node groups need a network
interface reachable from every node.

On Linux, the spawn parent reserves the rendezvous port until the children are
reaped. The socket must allow the TCPStore listener to bind alongside it.
The equivalent hold is unavailable on macOS, where port selection remains a
probe and a later bind can race with another process.

### Publishing and the workflow launch

The publishing rank of each data replica contributes its results to replica
zero's publisher, the **joiner**. Only the joiner writes the final run tree.
For `dp=N:points`, it joins results in memory by point digest before using the
ordinary artifact writer. Duplicate, missing, foreign, or out-of-order points
are rejected. For `dp=N:rows`, every replica has the same campaign outputs, so
the joiner writes without gathering point shards.

Workflows launch the same world for all their steps. Every rank executes the
steps in lockstep, and the joiner alone creates attempt directories, verifies
and publishes results, updates manifests, and decides reuse. The
`Lockstep` protocol broadcasts those decisions and failures before followers
advance. Each protocol and behavioral step checks the document rules of
§8.3 and §8.4 before any weights load, as `run_protocol` does, so every rank
refuses the same step. A behavioral step always counts as decoding. All ranks must share the run tree because later steps read published
outputs. Workflows support `pp`, `cp`, `tp`, and `ep`; `dp` is rejected. Use
`fan_out.over.shards` for workflow-level point sharding.

### Never branch on rank

A decision based on rank-local state must be agreed before it changes the
collective sequence. `Agreements` coordinates the following:

| Decision | Agreement |
|---|---|
| Window membership | Shortest fitting prefix across ranks, including uneven row replicas |
| Row-budget memory probe | Minimum across ranks that execute windows in lockstep |
| OOM retry in a collective-free window | Any rank's failure makes the group shrink together |
| Pipeline fire counts | Sum across stages before checking each write declaration |

The budget's lockstep axes are `model`, `pipeline`, and `context`, plus `data`
under `dp=N:rows`. Data replicas processing different points do not agree their
budgets. A probe's success or failure is agreed before its bound is reduced.
Retries are supported for single-rank and data-parallel windows whose bodies
contain no collectives. Tensor, expert, pipeline, and context parallelism abort
on OOM: a peer may already be inside a collective that the failing rank cannot
reach. Reduce `train.batch.pairs` or `--fit-rows` before restarting that run.

Eval scores, early-stop decisions, sampled tokens, gate draws, and reuse choices
must likewise agree wherever ranks execute the same work. Prefix-cache keys are
present on every pipeline stage even when only the owning stage stores their
values.

The launcher compares the raw environment values for
`CAUSALAB_RANK_GRACE`, `CAUSALAB_COLLECTIVE_TIMEOUT`,
`CAUSALAB_EXPERIMENTAL_CONTEXT`, and `CAUSALAB_GRADIENT_AGREEMENT` before
creating any process group. Set them consistently on every node; unset and
explicitly set values are distinct for this check.

### When a rank dies

The heartbeat uses the rendezvous store independently of model execution.
`CAUSALAB_RANK_GRACE` defaults to 30 seconds, with ten beats per grace.
`CAUSALAB_COLLECTIVE_TIMEOUT` defaults to 600 seconds and applies to every
process group. Both must be positive, and the grace must be below the timeout.
Choose a timeout longer than normal startup and collective skew, and a grace
longer than healthy gaps between heartbeat updates.

| Failure | Expected behavior |
|---|---|
| A connected peer exits or stops beating | Survivors identify the lost peer and exit within a grace plus three beats |
| A peer never reaches rendezvous | Survivors identify the missing rank after the collective timeout |
| The store becomes unreachable or stops replying | The heartbeat reports the store host; requests have an independent stall bound |
| A rank's computation hangs while its heartbeat continues | Collective timeouts end the run; the heartbeat cannot identify the original hung rank |
| A replayed CUDA graph's collective never completes | The heartbeat ends the replaying rank once the replay has waited the collective timeout, naming the replay ([CUDA graphs](cuda_graphs.md#hung-replays)) |
| Rank zero dies before its store exists | Clients time out connecting and report the rendezvous failure |

A backend exception is held briefly while the heartbeat distinguishes a lost
peer from a collective failure with all peers alive. The latter names the
operation, axis, rank, and group. A peer's exit can cause other survivors to
report that exit or the lost store rather than the original failure.

Under NCCL, the watchdog thread handles collective timeouts and may terminate
the process with `SIGABRT`; no Python exception handler runs in that rank. The
spawn parent reports the exit, while `torchrun` reports its worker's status.
The launcher defaults `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC` to `500`, retaining
an explicit value. NCCL's four waits plus its watchdog poll add approximately
2.1 seconds after a detected timeout at that default. This setting affects
failure teardown, not healthy execution. `TORCH_NCCL_BLOCKING_WAIT=1` and
`TORCH_NCCL_ASYNC_ERROR_HANDLING` modes `0` and `2` are rejected because they
prevent the required watchdog termination behavior. Process-group teardown
is bounded by the rank grace as well.

On successful shutdown, rank zero keeps its store and heartbeat alive until
every peer reports completion. This wait is bounded by the collective timeout;
failed peers remain subject to the rank grace. Nonzero exits publish failure
without waiting for clean peer completion.

A local spawn's store lives with rank zero. Under `torchrun`, the store can
belong to rank zero's agent instead. For multi-node failure diagnostics, use
`--rdzv-backend static --rdzv-endpoint HOST:PORT`, or host a c10d rendezvous
outside the workers. If a worker node also hosts a c10d endpoint, its loss can
make another node's agent terminate the worker before the heartbeat reports.
A stopped rank on a node with no failing sibling may need an operator to kill
it; `torchrun` does not necessarily reap it when another node fails.

## 4. The placement seam

`ResolvedSite.placement` describes where a tapped tensor lives. It comes from
the registry plan, module address, and geometry, not rank-local tensor values.
`neural/shared/parallel/placement.py` defines the placement types:

| Placement | Local value | Reconstruction |
|---|---|---|
| `Replicated` | Whole tensor | Identity |
| `Sharded(axis, group)` | Contiguous chunk along an axis | Gather in rank order |
| `ExpertLocal(group)` | Owned expert slots; zero elsewhere | Sum disjoint slots |
| `PartialSum(group)` | One summand | Sum across ranks |
| `SequenceSharded(..., inner, flat)` | Position chunk, possibly flattened and with an inner placement | Reconstruct the inner value, then gather positions |
| `StageLocal(stage, group, inner)` | Value on one pipeline stage | Reconstruct on its owner, then broadcast after the forward |

A `Sharded` placement can have multiple `slots`, each independently chunked,
or a `repeat` count for KV heads held by consecutive ranks. Slotted shards
cannot also be repeated. Composition places the stage outside the sequence
chunk, and the sequence chunk outside the model-parallel placement.

`Fragments.whole` reconstructs the global tensor before contract conversion,
position selection, feature slicing, metrics, or featurizer math.
`Fragments.fragment` selects this rank's contribution after an edit. For a
`PartialSum`, it gives the edited value to the group's first rank and zeros
to the others, so the module's later sum applies the edit once. World size one
uses the identity paths without collectives.

Hook placement depends on where the library's own collective runs:

| Module style | Pre-hook sees | Forward hook sees |
|---|---|---|
| `colwise` | Replicated input | Feature shard |
| `rowwise` | Feature shard | Replicated output after its sum |
| `colwise_gather_output` | Replicated input | Replicated output |
| `kv_replicated` | Replicated input | Repeated KV-head shard |
| `replicated_with_grad_allreduce` | Head shard | Head shard; the parameter is replicated |
| `grouped_gemm` | Replicated input | Replicated combined module output |
| `ep_router` | Replicated input | Logits replicated, scores expert-local, indices remapped |

Rowless modules normally produce replicated values. Modules between planned
projections inherit the relevant neighboring placement: for example,
`mlp_activation` uses the rowwise `down_proj` input and `attention_probs` uses
the query heads. `placements.py` owns these derivations; keep them consistent
with the styles that `apply_plan` installs.

## 5. Loading and applying a plan

### 5.1 One process, many devices

A device list such as `--device cuda:0,cuda:1` places contiguous layer ranges
within one process. `ModelBundle.devices` holds the `DeviceMap`; crossing
hooks move block inputs to their assigned device. Embedding inputs, final
normalization, and head projection use their own assigned devices.

This mode is separate from a multi-rank `--parallel` geometry. Models with a
tied embedding and head reject a device list, and CUDA graphs reject bundles
spanning devices. Resume values are moved to the device of the block they
enter; one bundle has one device map.

### 5.2 The parallel plan is a registry table

`ModelInfo.parallel_plan` maps module or parameter patterns to
`PlanRow(style, axis)`. Built-in entries declare their plans so `dry-run` can
check them without loading a model. Adapted entries derive them from the
transformers config's tensor and expert plans. Expert rows replace tensor rows
at and below their patterns; replacement must not depend on iteration order.

`apply_plan` resolves each active row through the `Styles` protocol, validates
and shards its parameters, and installs its forward behavior. Rows on axes of
size one are not applied. Unsupported styles remain in the registry plan and
are rejected when their axis is requested, rather than preventing a world-size-one
load. Vocabulary-sharding rows are retained as `unapplied` provenance: the
embedding and head stay whole, and `dry-run` reports the declined rows.

`ParallelPlan.for_geometry` rewrites K/V projection rows to `kv_replicated`
when tensor ranks exceed KV heads (§6.6). Loading and site resolution use this
same rewritten plan. The registry's original plan remains a model fact.

### 5.3 Shard-on-read

The planned loader requires a safetensors checkpoint. It raises `P2` when none
is available rather than falling back to a whole-checkpoint load.

`shard_read.read_plan` computes all checkpoint ranges before any read. It uses
the placement table and the styles' `Partition` arithmetic, including packed
projections, repeated KV chunks, and checkpoints storing separate expert
tensors that the loader fuses into one parameter.

Each rank reads only its owned ranges. Unowned expert keys remain available
to the converter's indexing but are not read. A pipeline stage requests only
its own parameters. The lazy weight resolves each requested index to a
pre-read `Piece`; an unplanned index raises `ShardReadError` rather than
silently reading the whole tensor. The planned path uses transformers' loading
internals because `from_pretrained` constructs its own model. The world-size-one
path uses the public `from_pretrained(state_dict=...)` entry.

The reader receives `select=` ranges independently on every rank. Its optional
group-broadcast path is not used: the plan can require expert-specific,
repeated, and strided pieces that a simple contiguous shard does not describe.

**One copy.** Reading a shard must also leave shard-sized backing storage.
`Residency` records local parameter sizes, shared storages, and CUDA allocator
state. A pre-load `Census` excludes tensors already alive from the load's
unowned-tensor accounting. The test rule checks exact planned elements, storage
ownership, allocated bytes, reserved bytes, and unowned tensors. Its allowances
are 512 MiB above parameter storage for device allocations, 2 GiB between
reserved and allocated bytes, and 64 MiB of unowned tensors. Keep these checks
separate from the runtime memory estimate: the estimate predicts headroom;
the residency tests inspect the completed load.

**The load's peak under a dtype conversion.** `target_dtypes` follows
transformers' per-key dtype rules, including keep-in-fp32 parameters. Converting
keys are read onto the host in batches bounded by the largest converting
tensor's on-disk size, cast there, and copied to the device in their resident
dtype. Each device group stages one batch at a time. Same-dtype keys read
directly to their destination. The report records disk and resident dtypes;
`dry-run` reports host staging separately from device memory. The device
estimate has no conversion-staging term because those source tensors stay
on the host.

## 6. The vocabulary under sharding

### 6.1 Module-boundary components

`head:` and `expert:` selectors refer to the global component vocabulary.
Executor adapters make tensors whole before applying a read or write, then
fragment edited values. Residual-stream components remain replicated under
tensor and expert parallelism. Embedding and vocabulary-head weights are not
sharded; a tied head is compatible with tensor parallelism but not pipeline
placement.

### 6.2 Attention interior

Under tensor parallelism, attention runs on local query heads and the
corresponding KV heads. Each interface slot derives its placement from the
mixer's projection rows and its declared head axis. Edits gather before feature
selection and re-fragment before the library's value multiplication.

`attention_result` computes its selected head from the global premix, fragments
the masked value, and calls the rowwise output projection, whose sum completes
the result. It is not supported under pipeline placement because that derived
computation requires the owner's weights. Under context parallelism, query
positions are local and key positions span the full frame (§8.4).

### 6.3 Routed experts

Under expert parallelism, the token-major interior has shape
`(tokens, top_k * d_expert)`. Each rank owns slots for its experts and must
zero sentinel slots before reconstruction. `Fragments.whole` with an
`ExpertLocal` placement sums disjoint contributions. The global routing table
is reconstructed by summing owned
`global_id + 1` values, with zero representing an unowned slot, then restoring
the no-slot marker.

Router logits are replicated; scores are expert-local; indices are remapped
by the router. Reads of scores and indices are supported. Writes to
`router_scores` or `expert_idx` under `ep > 1` are rejected because the module
tap cannot update the coupled routing and ownership information consistently.
Run those writes at `ep=1`.

Tensor-sharded expert interiors use per-slot neuron shards for gate/up and
activation values, and partial sums for the down projection where the plan
supports them. Do not infer support from a style name alone: tensor-axis
`moe_tp_experts` interior placements are rejected. Registered Qwen hybrid MoE
plans put experts on the expert axis; `tp` alone leaves those experts whole.

An expert-keyed write records routing mismatches on its owning stage. Under a
pipeline, each owner broadcasts its pending record in deterministic site and
write-name order before publishing, including an empty record when necessary.
This keeps the collective sequence independent of the mismatches found.

### 6.4 Gated DeltaNet

DeltaNet projections using `colwise_gather_output` gather their output before
the kernel, so the interior slots are replicated under tensor and expert
parallelism. Under context parallelism, each chunk receives the previous
chunk's final state and convolution history, computes locally, and sends its
state onward. DeltaNet chunks execute sequentially.

Per-step positions remain offsets into the whole frame. Written-step sets are
unioned across context ranks before checking declarations. Per-step reads that
need a gather run **after** the state handoff: gathering before sending would
wait on the next rank while that rank is waiting for the state.

Training handoffs carry gradients in reverse chunk order through
`send_with_grad` and `recv_with_grad`; the sent state is linked to the output
so backward reaches the send. Kernel wrappers are installed even for untapped
mixers, because every chunk must participate in the handoff.

`SymbolDispatch` manages kernel-global bindings with per-thread stacks. The
first active layer installs the dispatcher, and the last restores the binding.
A call uses its thread's top layer; `real` resolves to the layer below it.
Keep install/restore and lower-layer calls within this protocol so simulated
ranks can interleave safely in one process.

### 6.5 Pipeline stages

`StageForward` runs each rank's contiguous layer range. Stage zero embeds,
intermediate stages receive and forward residuals, and the last stage applies
the final norm and head. Positional head projections run on that last stage.
Logits and stage-local captures are broadcast after the forward.

Install hooks only on the owning stage and use the stage's inner placement
inside the hook. Do not broadcast from a mid-forward hook: another stage may
be waiting to receive the residual that this stage has not sent yet. Broadcast
captures afterward in document-derived order, then agree fire counts and
routing records. Check each cohort member's fire tally across stages as well.

Non-owning stages retain stand-in module trees for site resolution. Resume
keys exist on every stage; only the owner holds the cached tensor. A stage
entirely below the resume depth runs no blocks and sends nothing.

A graded forward uses the `Handoff.ALWAYS` protocol across residual boundaries.
Received logits and captures attach to the stage's outgoing link so each
rank's backward reaches the necessary boundary exchanges even when that rank
owns no trained parameter. Inference uses raw sends and receives.

Decode with a KV cache and the derived `attention_result` component are not
supported under `pp > 1`. A fit's trained featurizers must belong to one stage;
different members of a cohort may have different owning stages (§7).

### 6.6 What is refused by name

Geometry and document errors identify the unsupported axis or operation.
Principal restrictions are summarized in §11. CUDA graphs serve `tp`, `ep`
and `dp` over points, and use the logged eager fallback under `pp`, `cp` and
`dp=N:rows` ([CUDA graphs](cuda_graphs.md#multi-rank-execution)). Quantized
weights are not supported under sharding.

**KV-head replication.** When `tp` exceeds the KV-head count, `kv_replicated`
keeps each K/V projection's weight whole and selects the output head needed by
this rank's query heads. With `repeat = tp / num_kv_heads`, rank `r` uses KV
head `r // repeat`. The mixer's local `num_key_value_groups` is divided by
`repeat`, so its existing GQA mechanism operates on the local heads. Placement
reconstruction keeps one copy of each repeated head; input gradients sum over
the tensor group.

**The straddle rule.** A geometry is valid only if `tp` divides `num_kv_heads`
or `num_kv_heads` divides `tp`, in addition to query-head divisibility. Otherwise
a rank can own query heads belonging to different KV heads. For example, a
14-query-head, 2-KV-head model rejects `tp=7` even though seven divides fourteen.

### 6.7 `gaussian.axis`

Each rank draws noise over the global feature axis with the document's seed,
then fragments it like any other write. Both declared axis choices preserve
the single-device draw and its document identity.

## 7. Training

Model weights are frozen; only featurizer parameters train. At replicated
sites, every rank sees the same tensors and computes the same gradient. At
sharded sites, the edit is between reconstruction and fragmentation, so those
operations need a coordinated backward.

For replicated losses over a globally edited tensor, the autograd pair is:

- `fragment` backward reconstructs the full edit gradient from all ranks'
  slices or masked summands. Repeated KV chunks sum their partials.
- `whole` backward returns this rank's slice of the replicated gradient, or
  the unchanged gradient for a summed placement.

These operations are `gather_for_edit` / `edit_fragment` and
`sum_for_edit` / `edit_summand` in `parallel/autograd.py`. Their forward values
match the raw collective path. Tensors without gradients use the raw path.
Every rank must select the same backward protocol and collective sequence.

A gather whose consumers differ by rank needs a different derivative:
`gather_reduce_scatter` sums contributions from every consumer before taking
the local slice. The context-parallel attention KV gather uses this operation,
because each rank's queries contribute to other chunks' keys and values.
A tap serving a replicated loss uses the edit pair instead.

`agreements.average_gradients` runs after a step's windows over the model group.
Its input invariant is that every rank already has the full featurizer
gradient; averaging partial gradients cannot repair a missing contribution.
Set `CAUSALAB_GRADIENT_AGREEMENT` to enable a pre-mean comparison. It accepts a
relative tolerance in `[0, 0.5)`, measured against the largest gathered gradient
entry. `0` requires bit identity; unset disables the check. Invalid values are
rejected before the forward. Simulation tests use zero; real-backend tests use
a small relative tolerance for reduction rounding.

Pipeline parameters train on their owning stage. After each optimizer step,
`TrainedOwner.sync` broadcasts the owner's complete `state_dict`, including
buffers, before evaluation, snapshots, or publishing. A single fit spanning two
owning stages is rejected; each cohort member can have its own owner. A
non-owner's optimizer state is not the trained state and is not published.

Handoff links agree `Handoff.ALWAYS` or `Handoff.IF_REQUIRED` once per ordered
rank pair and axis. Pipeline residuals use `ALWAYS`; DeltaNet state links use
`IF_REQUIRED`. A mismatch raises `HandoffMismatch` on both ends. Under
`IF_REQUIRED`, both ends must also agree on whether a gradient is required;
that fact must not depend on rank-local control flow.

## 8. Execution by axis

### 8.1 Foundation — geometry and placement

Keep geometry validation and mesh arithmetic torch-free. Derive placements
from registry facts and preserve world-size-one identity paths. Changes to
loading, hook placement, or collective behavior should be covered by the
property and conformance tests in §10 before extending the supported geometry.

### 8.2 Tensor and expert parallelism

Tensor and expert groups share the model dimension. Apply the registry plan
to each subgroup, load its ranges, and derive tap placements from that same
plan. Tensor and expert sharding can change floating-point reduction order;
tests compare against a single-device oracle with explicit tolerances. Integer
routing can also change when bf16 scores cross a top-k tie.

### 8.3 Data and pipeline parallelism

Under `dp=N:points`, replica `d` executes the `d`-th contiguous shard of the
selected points. Shard sizes differ by at most one; more replicas than points
is rejected. Joining by digest preserves campaign order. Compare artifacts
against the same world-size-one campaign, allowing only execution metadata to
differ. Event timestamps and forward counts can differ because each replica
interns its own point subset.

Under `dp=N:rows`, every replica executes every point, and each fit minibatch
is divided into contiguous row slices. Replica `r` weights its loss by
`n_r / N`, including regularization, then gradients are summed over the data
axis after the windows. There is no second division: the row fractions already
supply the normalization. Evaluation agrees `(sum, count)` values before
computing scores and early-stop decisions. Fit-constant captures belong to
each replica's rows.

Rows mode requires a training document and at least one row per replica in
every minibatch, including the epoch's remainder. `check_rows` checks the
static constraints; the engine rejects an undersized remainder when known.
The row budget also agrees over the data axis in this mode.

Pipeline execution and training follow §6.5 and §7. Each cohort member has its
own owner, and fire counts and routing records must be shared for every member.
With matching dtype, kernel configuration, batch shapes, and host thread
settings, pipeline placement preserves the single-device computation's order.

### 8.4 Context parallelism

`sequence_chunks` partitions the padded frame into contiguous chunks, with the
remainder on the last rank. A frame shorter than the group is rejected.
Positions remain absolute into the whole frame, including rotary positions and
write coordinates. Uneven chunks are padded for gathers and trimmed afterward.
Flattened expert-token axes are reconstructed using the frame's row count.

The executor forces eager attention. Every rank gathers the full keys and
values, retains its local queries, and uses the whole frame's causal-mask rows
for those queries. Attention score and probability tensors therefore shard the
query-position axis while retaining all key positions. Position-local layers
need no additional exchange. DeltaNet passes state sequentially (§6.4).

Decode, cached DeltaNet state, and chunks too short for convolution history are
rejected. A family with any `linear_attention` layer is treated as hybrid and
rejects `cp > 1` by default. `CAUSALAB_EXPERIMENTAL_CONTEXT=1` waives that family
restriction; unset or `0` keeps it, and other values are invalid. The check
covers every model in a swept campaign. The waiver does not remove the other
context restrictions.

Context parallelism replicates weights and full attention K/V. It can reduce
some activation storage but is not a general weight-memory reduction, and
DeltaNet's sequential handoff limits throughput.

### 8.5 Not in scope

The nnsight engine remains single-device and is excluded from automatic engine
selection for multi-rank execution. FSDP, quantized weights under sharding,
ring attention, and sharded vocabulary weights are not supported.

### 8.6 Pipeline schedule

Pipeline stages execute one after another over a complete window. There is no
streaming microbatch mode or `--micro-batches` option. Use `pp` for placement
and memory relief; it does not provide overlapping stage execution.

## 9. The receipt

`run.py:execution_record` records geometry through `parallel_record` under
`execution.parallel`, alongside other execution settings. For a spawned
`tp=4,ep=8` run:

```json
{
  "parallel": {
    "data": 1,
    "data_mode": "points",
    "pipeline": 1,
    "context": 1,
    "tensor": 4,
    "expert": 8,
    "world": 8,
    "launcher": "spawned"
  }
}
```

`launcher` is `solo` for one process, `spawned` for local children, or `joined`
for an externally launched world. The publisher supplies that value. Geometry
never enters a document digest or qualification identity.

`execution.device` is the `--device` value. A CUDA world above world 1
records `cuda`, since each rank runs on `cuda:LOCAL_RANK` (§3), so a world's
receipt and its solo twin's name the same device.

## 10. Testing

Follow the tiers and commands in [TESTS.md](TESTS.md). The CPU gate is:

```bash
uv run pytest -m "not golden and not measurement_study and not parallel_world"
```

`parallel_world` is this design's own cost marker: the twelve modules that run
a document or a fit across a spawned world (tp/ep/pp/cp/dp on the tiny
fixtures) take a minute or more each on a two-vCPU machine, so the
pull-request gate deselects them and `-m "not golden"` runs them; the
collective, lockstep, mesh and styles contract suites stay on the gate. Run the
twelve alone with `uv run pytest -m parallel_world` (TESTS.md "Conventions").

Tests use the single-device engine as an oracle, explicit tolerance rules, and
fault injection. Preserve the distinction between equal results across ranks,
parity with the oracle, and successful failure detection: each needs its own
assertion.

### 10.1 The seams that make simulation possible

| Protocol or seam | Production | Test implementation |
|---|---|---|
| `Collective` | `TorchCollective` over `Mesh` groups | `Solo` and `SimulatedWorld` |
| `Meter` | CUDA memory queries | Scripted readings and injected OOMs |
| `Lockstep` | `CollectiveLockstep` | Same implementation over simulated collectives; `Solo` at world size one |
| `Fragments` | Reconstruction over `Collective` | Same implementation over the simulator |
| `Styles` | `TransformersStyles` | `FragmentStyles` over a collective |
| `DeviceMeshFactory` | `DeviceMesh.from_group` | A factory with explicit test results |
| Heartbeat store and clock | Rendezvous store and elapsed time | Shared fake store and scripted clock |

Use one reusable fake per seam and one conformance suite per protocol. The
collective, lockstep, and style contracts run against all their implementations.
When a simulator disagrees with the real backend's contract, fix the simulator
rather than weakening the oracle.

### 10.2 `SimulatedWorld`

`tests/_helpers/simulated_world/` runs one thread per rank, with a scheduler
allowing only one rank to execute between collective calls. Each collective
parks its caller at a rendezvous. Group members must agree on operation, call
site, shape, and dtype; reductions use a fixed order on CPU. Payloads detach
when crossing ranks so one rank cannot accidentally use a peer's autograd graph.

A schedule is either a seed or a tape of non-negative integers. At a choice
between runnable ranks, the next tape value selects an index modulo the
candidate count. Forced choices consume no entry; an exhausted tape chooses
the lowest rank. Hypothesis can shrink a failure to the early choices that
cause it. `slow(rank)` deprioritizes that rank while others can run.

The simulator reports divergent collectives, incomplete groups, and ranks
finishing while peers still wait as typed errors. Its heartbeat driver uses
the same schedule representation with scripted deaths, stops, store failures,
and elapsed time. Simulated exit of the store host also removes the fake store.

### 10.3 Properties

Property tests should cover the following contracts and inject faults that
would violate them:

| Area | Invariant | Example fault |
|---|---|---|
| Geometry | Accepted axes satisfy registry divisibility and plan rules | Drop a divisibility check |
| Mesh | Each axis partitions the world; tensor/expert groups stay inside model groups | Shift a subgroup boundary |
| Fragments | Reconstruction and fragmentation preserve owned values and global order | Reverse gather order |
| Routing | Remapping and reconstruction preserve global expert IDs and sentinels | Drop the ID offset |
| Loading | Read ranges match the style's partition, including repeated and packed chunks | Read an unowned expert |
| Pipeline | Stage ranges cover layers once; resume has one owning stage | Omit the last layer |
| Publishing | Every selected point appears once, in campaign order | Duplicate a point |
| Agreements | Budgets and retry decisions agree over lockstep ranks | Keep a local memory bound |
| Autograd | Each rank's trained gradient equals the oracle's full gradient | Return only a local partial |
| Scheduling | Results do not depend on a valid interleaving | Skip a required collective |

### 10.4 Deterministic simulation scenarios

Exercise the real executor, train loop, placement derivation, and style
arithmetic with tiny models. Include boundary and interior reads and writes;
composed geometries; pipeline resume and cohort owners on different stages;
context handoffs with the loss in a different chunk from the write; row splits;
and reuse decisions shared across ranks.

Script different memory readings and OOMs on one rank. Inject missing
collectives and gradient handoffs to verify that the harness detects failure.
Use exact arithmetic where bit identity is the contract; where sharding
reassociates floating-point operations, state a justified tolerance instead.

### 10.5 Determinism of the tests themselves

Draw schedules through `parallel_strategies.schedules()` so failing tapes can
shrink. Draw torch seeds independently through `seeds()`. Hypothesis draws on
the test thread before rank threads start. Fixed schedules use the empty tape
for rank order. Repository property settings use `max_examples=30` and no
Hypothesis deadline.

Script nondeterministic inputs behind the corresponding seam. Assert replay of
identical results, decisions, and handoffs for the same schedule. Threaded
simulation tests also need correct restoration of process-global kernel
bindings so later tests do not inherit a wrapper.

### 10.6 The boundary, and the real-backend tiers

The simulator cannot host transformers' DTensor rank state, which is
process-global. Gloo smoke tests exercise that boundary with real processes,
real shard-on-read loading, and the tiny model fixtures. `GlooWorld.run` returns
rank results in the same shape as the simulator, allowing shared contracts.

GPU goldens validate real weights and NCCL behavior:

| Suite | Coverage |
|---|---|
| `test_parallel_parity.py` | A3B inference and featurizer fits; dense fp32 fit |
| `test_parallel_world4.py`, `test_parallel_worlds.py` | Wider geometries and CUDA versions of the tiny-model scenarios |
| `test_parallel_large.py` | Models that cannot fit one device, using a fitting pipeline geometry as oracle; multi-node replay |
| `test_parallel_families.py` | Additional checkpoint families and their loading rules |
| `test_parallel_soak.py` | Memory stability over repeated points |
| `test_parallel_watchdog.py` | Rank loss, store failure, startup failure, and hangs on NCCL |

**The large model.** The `large` and `das_large` documents run on
Llama-3.1-70B with `pp=4` as their oracle because the model cannot fit one
device. They compare wider pipeline and tensor geometries against that oracle,
check each rank's load report and memory estimate, and replay multi-node runs
from a kept root. Select them with `--only large,das_large`.

**The second-family rows.** `inference_gemma2_9b` and
`inference_llama31_8b` use their own model realizations. They check exact
data/pipeline cases, banded tensor cases, and tied-head pipeline rejection
where applicable. Select these documents by name with `--only`; they are not
part of a default capture.

Tests skip with an explanation when the required devices or records are absent.
A two-device run cannot cover all wider geometries. Replay affected wider
and multi-node suites when changing numerics or loading. The watchdog suite
intentionally causes failures and hangs, so run it separately when needed.

The record is `tests/golden/parallel_goldens.json`; helpers live in
`tests/golden/_parallel/`. To check the default documents without writing:

```bash
HF_HUB_OFFLINE=1 uv run python tests/golden/update_parallel_goldens.py --check
```

Inspect a capture's diff before accepting it with `--i-have-reviewed-the-diff`.
`--only inference,das,dbm,das_dense` selects documents; `--keep ROOT` retains
runs and resumes receipts already present. Use a fresh root after code changes:
an existing receipt skips execution without checking the producing revision.
Additional families and the large
model are selected explicitly with `--only`. Generate multi-node commands with:

```bash
uv run python -m tests.golden._parallel.large commands \
    --root ROOT \
    --endpoint HOST:PORT
```

Exact geometries must match the oracle byte for byte. Changing one to banded
requires `--explain-inexact 'DOCUMENT GEOMETRY: WHY'`, recorded with the result.
Floating-point classes use:

```text
band = max(3 * max_abs_diff, 2 * ulp(dtype, scale), 1e-3)
```

`max_abs_diff`, oracle scale, dtype, and the resulting band are stored together.
Routing uses a separate fraction rule: twice the observed mismatch fraction,
with a 0.01 floor and a cap of one. Stale or incomplete records identify the
recapture command. Keep measurement provenance in the structured record;
comments should explain the rule rather than repeat capture histories.

Fit records include gradients before averaging and the cross-rank agreement
tolerance. Bf16 fits can diverge substantially from another geometry while
each geometry's ranks still agree. The dense fp32 fit tests sensitive gradient
parity with less forward rounding noise. Fixed step counts avoid comparing
fits stopped at different iterations by a near-tied metric.

Per-rank load reports are enabled through `CAUSALAB_LOAD_REPORT_DIR`. Tests
check planned bytes, stage ownership, and residency (§5.3).

**The soak.** Memory traces record allocated and reserved bytes per point.
`flat_after_warmup` checks the
allocated slope and sustained reserved-pool growth after warm-up; preserve its
sample-count and slack rules when extending the test.

### 10.7 Mutation coverage

The `[tool.mutmut]` section of `pyproject.toml` selects source modules and the
in-process judging tests. Follow its setup instructions for the pinned mutmut
version and stale pytest bytecode. Limit CPU threads and hide CUDA when running
this CPU analysis:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= uv run mutmut run
uv run mutmut results
uv run mutmut browse
```

Subprocess-only coverage is not visible to mutmut's per-test coverage tracking.
Use deterministic in-process tests for protocol behavior and targeted real-backend
tests for process or DTensor boundaries. Investigate each survivor: add a test
for missing behavior or explain an equivalent mutation in the review. Record
run counts and survivor inventories in the PR discussion, not this guide.

### 10.8 The style shim: two tiers behind one protocol

`Styles.style(row)` resolves a plan row to a `Style` with `validate`, `shard`,
and `install` operations. `Partition(dim, interleave)` describes parameter
chunks; `partition_of` supplies the same arithmetic to both style tiers and
the read planner.

`TransformersStyles` is the production default. It applies transformers'
DTensor styles over each axis's `DeviceMesh`, plus the repository's
`kv_replicated` style and required input-gradient sums.

`FragmentStyles` implements the corresponding arithmetic over plain tensors
and `Collective`, using the autograd pairs for its forward wrappers. Tests use
it with simulation and gloo; the engine's loader does not select it. The
contract suite compares parameter shards, outputs, gradients, and module facts
across both tiers. Exact fixtures use exactly representable values; real-model
scenarios use justified bands where reductions reassociate. Keep these tests
when updating transformers, since upstream style changes need corresponding
fragment-tier changes.

## 11. Known limits

| Area | Constraint |
|---|---|
| Engine | Multi-rank execution requires `pytorch_hooks`; nnsight remains single-device |
| Workflow | `pp`, `cp`, `tp`, and `ep` are supported; `dp` is rejected; ranks share the run tree |
| Measurement and profiling | [`causalab measure`](measurement.md) requires one device and one rank; multi-GPU artifact collection is unsupported |
| Pipeline | Sequential stages; no decode, derived `attention_result`, tied embedding/head, or single fit with owners on multiple stages |
| Context | No decode; full weights and gathered K/V remain per rank; hybrid families require the experimental waiver |
| Rows mode | Every minibatch, including a remainder, must have at least as many rows as replicas |
| Vocabulary | Embedding and head remain whole; vocabulary-sharding plan rows are reported as unapplied |
| Routing | Writes to `router_scores` and `expert_idx` require `ep=1`; `expert_permutation` is unsupported |
| Runtime | Quantized multi-rank weights are unsupported; CUDA graphs run eagerly under `pp`, `cp` and `dp=N:rows`, and an out-of-memory graph under `tp` or `ep` aborts the run |

For parity comparisons, match dtype, kernel settings, batch layout, and host
thread count. The seeded subspace initialization uses CPU QR, whose rounding
can depend on BLAS threads. Pipeline placement preserves operation order;
tensor/expert sharding and row/context splits can change reductions. A
single-device rerun being reproducible does not imply cross-geometry training
reproducibility, especially in bf16. Golden tolerances bound recorded drift;
regressions smaller than those tolerances can pass.

### Memory estimate limits

The default rule counts one resident copy of each parameter on its owning
rank, then adds 15% of the **whole model's** bytes for activations, captures,
CUDA context, and allocator headroom. Context parallelism adds another 10% of
the whole model. Conversion staging is on the host and is reported separately.

These coefficients are calibrated to the standard workflow. They can
underestimate a run or overestimate headroom for small shards. They produce
advisories, not hard refusals. Admission checks only the resident weights
against available memory, including reusable allocator cache; neither passing
that check nor satisfying the estimate guarantees a workload will fit. Unusually
large frames or row counts can exceed the estimate after loading. The runtime
pre-flight applies only to multi-rank loads; it does not add a new rejection to
the world-size-one path. Use per-rank load reports and peak-memory measurements
to diagnose a workload, and preserve the distinction between resident bytes,
allocated peaks, reserved peaks, and total device usage.

### Failure-handling limits

Heartbeats detect missing progress in the heartbeat itself, not a computation
that hangs while the heartbeat continues. Collective timeouts cover the latter,
and NCCL can terminate a rank before Python reports an error (§3). OOMs in
sharded model windows abort the run rather than retrying a broken collective
sequence. After a failure, survivors may identify a secondary peer exit or the
lost store; use the launcher and backend output together to locate the
initiating failure.
