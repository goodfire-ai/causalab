# Modeling assumptions behind SOL references

The catalog provides conditional resource estimates, not certified lower bounds
or achievable runtime targets. Distinguish minimum work for a requested result,
work in a specified algorithm, and work executed by installed kernels. The
[execution contract](speed_of_light.md#measure-achieved-percentage-of-speed-of-light)
defines the current Qwen benchmark; the [before/after harness](measurement.md)
measures actual protocol operations separately.

## Outputs, reuse and dependency boundaries

The Qwen contract uses last-token vocabulary heads, stops source execution at the
capture point, and includes cold base-prefix cache fills and subsequent reuse.
Only the frozen suffix propagates intervention gradients. These choices must
match the measured operation before its latency can be compared to the reference.
Cache construction, reads and retained state belong in the work ledger.

Changing a last-position intervention leaves earlier causal outputs unchanged.
That does not remove the earlier keys, values or recurrent state needed by later
attention. A specialized last-token suffix algorithm therefore needs its own
ledger; dividing full-suffix work by sequence length is not valid. Tiny-model
tests check last-position logits and intervention gradients for full-attention
and DeltaNet suffixes, without asserting bitwise parity on every real-model kernel.

## Recipe arithmetic

For a dense MHA/gated-MLP model, let `B,S,D,M,L,V` be batch, padded sequence
length, width, MLP width, layers and vocabulary. Counting an FMA as two FLOPs:

```
P = L(4D² + 3DM) + DV
forward FLOPs = 2BSP + 4BS²DL
weight bytes read = 2P  (BF16, one read per execution)
```

This uses full-square attention and full-sequence logits; a separate embedding
table contributes residency. Norms, softmax and elementwise operations are
omitted. GQA, sliding windows, causal triangle skipping, quantization, tied
embeddings and selective logits need adjusted ledgers.

A one-position subspace swap uses five projection GEMMs, totaling `10BDk`
FLOPs. Cayley materialization costs `10Dk² + 10k³` per basis access, with three
accesses per swap; inverse/norm costs are excluded. Other parametrizations need
separate ledgers. Frozen-backbone input-gradient GEMMs cost approximately one
suffix forward; feature backward uses an approximate factor of two.

Qwen uses actual suffix layer types, GQA widths, shared experts/gates, routers
and Delta projections/convolution. Chunk64 Delta backward GEMMs cost
`2 × forward_GEMMs − 8CKV` per batch/head over the padded sequence (`C=64`):
initial state is constant and the final state update is unused. The triangular
loop uses `2 Σ(i²), i=1..63`; scalar/backward costs remain estimates. Loss,
nonlinearities, launch overhead and additional kernel traffic are unmodeled.

## Scheduling and arithmetic modes

Qwen phases separate sequential mixer and MoE stages and the vocabulary head.
Taking a maximum of whole-forward compute and memory totals would permit overlap
across dependencies. Conversely, an implementation may prefetch future weights.
Every tighter estimate needs an explicit schedule and overlap policy.

The compute term adds BF16 Tensor Core time and scalar FP32 time. That requires
serialization or a shared-capacity assumption; distinct execution units are not
interchangeable. A weaker resource bound takes the maximum of independent demands.
Neither schedule is a calibrated prediction of installed-kernel throughput.
Model loading, tokenization, host/disk transfers, scheduler bubbles and
recomputation need separate costs; the ledger does not predict end-to-end latency.

The optional [sensitivity diagnostic](../examples/sol/audit/assumption_sensitivity.py)
partitions a whole-forward architecture ledger to show the effect of alternative
schedules and observed routing. It is separate from the benchmark's output/cache
contract and must not be substituted for its denominator.

## Logical bytes, routing and hardware traffic

Fusion, tiling and cache retention determine whether intermediate tensors reach
HBM. Summed logical tensor sizes are neither exact HBM traffic nor a universal
lower bound. The Delta triangular loop can create substantial logical traffic,
but small intermediates may remain in cache. Conversely, tiled GEMMs can reload
weights. A detailed model needs separate HBM, L2 and on-chip traffic estimates.

The optional `Phase` routing fields estimate independent uniform top-k traffic:

```
expected experts touched = E × [1 − (1 − top_k/E)^(global_tokens/DP)]
weight reads per GPU = all_expert_bytes × expected_fraction / TP
```

Qwen compute uses top-8 per token; residency includes all 256 experts. For
measured routing, supply explicit touched-weight traffic and leave the optional
routing fields zero. Real routing can concentrate assignments, alter weight
traffic and leave uneven GEMM shapes. A kernel-specific estimate needs per-layer assignment
histograms, tile sizes, occupancy and cache reuse. Rounding token counts to a
hypothetical tile size is a sensitivity calculation, not measured executed FLOPs.

[FlashAttention's I/O analysis](https://arxiv.org/abs/2205.14135) treats fast-memory
capacity and algorithm choice as part of the model.
[NVIDIA's matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
explains the dependence on arithmetic intensity, matrix shape and tiling.
Published device peaks alone do not establish attainable throughput for a GEMM.

## Backward work and memory capacity

A frozen linear layer needs an input-gradient product but no weight-gradient
product. A product with two varying operands normally needs two gradient products.
The current Delta GEMM derivation follows live graph dependencies, including
constant initial state and unused final state, and is checked against autograd
operator counts. It does not fully account for reductions, elementwise work or
all fused algorithms. A blanket backward-equals-twice-forward multiplier is not
an independent derivation.

Capacity requires tensor lifetimes, saved activations, caches, optimizer state,
workspace and allocator reserve. Summing per-layer recurrent states can overcount
inference liveness while omitting training saves. The catalog reports memory
feasibility as unknown unless persistent residency alone exceeds capacity; it
cannot certify that a batch will fit.

## Distribution and interpretation

The Qwen catalog enumerates supported data-parallel layouts and excludes tensor
parallelism. Dividing all work and storage by a TP degree would not define an
algorithm: KV-head replication, recurrent-state placement, featurizer layout and
collectives need explicit treatment. Ideal ring communication and zero startup
latency are conditional assumptions, not measured scaling models. The before/after
measurement harness remains single-GPU.

Bind comparisons to requested outputs, deterministic inputs, work counts, dtype,
cache policy, schedule and distribution plan. Preserve the original reference and
its identity with each measurement; changing assumptions creates a new reference.
A percentage above 100% calls for checking the ledger and execution contract.
Do not fit an arbitrary efficiency factor to make an observed percentage look
more plausible.
