# Tensor-parallel delivery

`load_files` and `stream_files` can deliver each rank's tensor-parallel shard.
The caller supplies a `shards` mapping. Each rank receives the requested slice,
which reduces the size of its largest delivered tensor and its communication
buffers.

A coordinated load that replicates whole tensors needs approximately:

```text
parameters + 2 × largest whole tensor + read-ahead + in-flight window
```

The second whole tensor accounts for a consumer loop that retains its previous
value while receiving the next. Sharded delivery uses shard sizes for these
terms.

## API

```python
from causalab.io.fastersafetensors.torch import Shard, load_files, stream_files

shards = {name: Shard(dim, rank, world) for name, (dim, rank, world) in cuts}

load_files(files, device, keys, select, group=group, shards=shards)
stream_files(requests, device, group=group, shards=shards)
```

`Shard(dim, rank, world)` selects `torch.chunk(t, world, dim)[rank]`. Negative
dimensions are allowed. The dimension must divide evenly by `world`; otherwise,
the loader raises `SelectError` before reading data. Construction checks
`0 <= rank < world`.

Group members may request different ranks. Their `dim` and `world` must agree,
which the request signature checks. A mismatch raises `PlanError`. `world` can
differ from group size, and multiple members can request the same shard for
replicated key/value heads.

A tensor name can appear in either `select` or `shards`. An overlap raises
`SelectError`. An unloaded name in `shards` raises `KeyError` from `load_files`;
`stream_files` checks unused names after its final request and raises
`SelectError`. Calls without a group read each shard independently.

Other tensors are replicated. Whole tensors above `COOPERATIVE_BYTES` use
cooperative reads. Smaller tensors and narrowed or stepped selections use owner
broadcasts and count toward the in-flight memory window.

## How a shard is read

Let `rows` be the product of dimensions before `dim`, and `cols` the product
from `dim` onward. The shard contains piece `rank` of each row, with `world`
pieces per row.

For `rows <= 1`, the shard is contiguous. The `Direct` path reads it into the
delivered tensor as part of the rank's chunk jobs.

For `rows > 1`, the `Exchange` path gives each group rank a contiguous row block
using `torch.tensor_split(rows, group)`. One `all_to_all_single` exchanges the
pieces each rank requests. Row order is preserved, so the receive buffer holds
the final shard.

For a tensor of `N` bytes and a group of `G` ranks:

| Path | Storage bytes per rank | Collective bytes per rank | Extra memory per rank |
| --- | --- | --- | --- |
| `Direct` | `N / world` | 0 | 0 |
| `Exchange` | `N / G` | Send `N / G`, receive `N / world` | Row block `N / G` and its packed copy |

Small shards still require short reads. Use sharding where the memory and
communication savings justify those round trips.

## Memory accounting

The scheduler counts the bytes each rank holds:

| Field | Meaning |
| --- | --- |
| `_Prepared.sizes` | Delivered shard sizes |
| `chunk_bytes` | Bytes read by the rank: a shard or row block |
| `largest` | Largest delivered tensor |
| `broadcast_largest` | Largest tensor in the owner-broadcast window |
| `exchange_bytes` | Largest row block, reserved again for its packed copy |

Sharded and cooperative tensors stay outside the owner-broadcast window.
`_check_fit` requires this total to fit within device headroom:

```text
resident chunks + exchange copy + in-flight window + 2 × largest + pack buffer
```

The resident budget uses half the remaining headroom. Every rank follows the
same chunk sequence and collective order.

## Integrating with SGLang

To call this loader from the model loader of
[SGLang](https://github.com/sgl-project/sglang), the consumer maps checkpoint
names to parameter shards and passes that mapping as `shards`.
Use `output_dim` for column-parallel layers and divisible vocabulary embeddings,
`input_dim` for row-parallel layers, and the replicated-head mapping for key/value
projections. This needs the name mapping from `stacked_params_mapping` or an
equivalent model interface.

The consumer must also recognize tensors that are already sharded.
`use_presharded_weights` or a check against the parameter's shard shape can
prevent a second narrowing. These changes belong in the consuming loader.

Some checkpoint layouts require additional handling:

- Fused tensors that need multiple cuts must stay replicated or use a richer
  selection mapping.
- Quantization scales, repacking, and loaders that reshape before narrowing
  need mappings for their stored layouts.
- Expert tensors with `ep_size=1` remain replicated. Expert-parallel loads use
  `keys` without a group.
