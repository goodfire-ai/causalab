# fastersafetensors

`causalab.io.fastersafetensors` reads and writes the safetensors format. Its
planner selects file concurrency, staging, and transport from a machine probe
and a calibration profile. `explain()` reports the plan and its reasons.

## Architecture

| Location | Responsibility |
| --- | --- |
| `crates/fst-core` | Format, storage, environment probe, and I/O plans |
| `crates/fst-cuda` | CUDA runtime loading, pinned staging, and simulated CUDA |
| `crates/fst-py` | PyO3 bindings that release the GIL during I/O |
| `causalab/io/fastersafetensors` | Tensor allocation, Python API, and distributed loading |

Maturin builds `causalab.io.fastersafetensors._core` during `uv sync` and
`uv build`. Rust builds on machines without CUDA. The runtime loads CUDA
libraries when needed.

Torch owns tensor memory. Python keeps each contiguous tensor alive while Rust
fills its pointer. A tensor's lifetime is independent of Rust objects.
Pointer checks cover null addresses, overflow, overlapping destinations, and
device identity.

`plan_read(env, profile, request)` is pure. The environment probe supplies
machine facts, and the profile supplies rates and costs. Each planning reason
names its profile source and entry. Add a measured rule with its evidence.

Simulator tests check exact operation logs and injected failures. Real backends
have round-trip tests. Format fixtures come from safetensors 0.8.0; properties
check parsing and serialization. Headers must match the reference bytes, with
tensors ordered by descending `Dtype` declaration order and ascending name.

Errors use domain enums and Python exception classes. Wrapped storage or CUDA
failures preserve their domain; malformed read or write jobs raise `ReadError`
or `WriteError`.

## Python API

The module supplies `save_file`, `save`, `load_file`, `load`, and `safe_open`.
`safe_open` exposes `keys`, `metadata`, `get_tensor`, and `get_slice`.
Non-contiguous inputs are packed for writing. Tensors that share storage are
written as independent tensors.

Additional operations:

| API | Purpose |
| --- | --- |
| `load_files(filenames, device="cpu", keys=None, select=None, *, group=None, shards=None)` | Read a checkpoint with concurrent file reads and optional selections. |
| `stream_files(requests, device="cpu", *, group, shards=None)` | Read ahead within device headroom and yield tensors in request order. |
| `Shard(dim, rank, world)`, `select_shards(names, dim, rank, world)` | Describe evenly divided tensor shards. |
| `safe_open(...).get_sharded(name, dim, rank, world)` | Read one shard from a file. |
| `serialize(tensors, metadata=None)` | Return a `Payload` with a header, parts, and byte count for a caller that owns the write. |
| `explain(filename_or_filenames, device="cpu", keys=None, select=None)` | Describe the plan and selection read costs. |

`select` maps tensor names to indexes or `Shard` values. Indexes support ints,
slices, `Ellipsis`, and `None`. A stepped slice reads its covering box and then
copies the selected elements. `get_slice(name)[index]` uses the same selection
path.

`save_file(..., durable=False, atomic=True)` streams CUDA tensors through pinned
host buffers. It writes a sibling temporary file and renames it into place,
removing the temporary file on failure. `durable=True` also synchronizes the
file and directory to storage. A write supports one CUDA device directly;
tensors on other devices are copied to the host. `save` and `serialize` prepare
host bytes.

## Coordinated loading

A `group` enables CPU/Gloo or CUDA/NCCL loading. All members must issue matching
collective calls with identical files, keys, and replicated selections. Set the
current CUDA device before loading on CUDA. Calls without a group load
independently.

Whole replicated tensors above `COOPERATIVE_BYTES` use cooperative reads.
Smaller tensors and narrowed or stepped selections use an owner read followed
by broadcast. Owner broadcasts count toward the in-flight memory window.
`shards` delivers each member's requested slice; its dimension and world size
must agree across the group. See [tensor-parallel delivery](tp-narrowed-delivery.md).

The loader divides a rank's reads into jobs of about 4 GiB, with a default
chunk count bounded to 4 through 16. It reuses pinned buffers after per-slot
CUDA fences complete. Scheduling failures propagate through the round's error
collective to every member.

| Settings | Group requirement |
| --- | --- |
| `FASTERSAFETENSORS_CHUNK_GIB`, `FASTERSAFETENSORS_CHUNKS`, `FASTERSAFETENSORS_COOPERATIVE_GIB` | Must match. They shape collectives and enter the request signature. |
| `FASTERSAFETENSORS_INFLIGHT`, `FASTERSAFETENSORS_INFLIGHT_GIB`, `FASTERSAFETENSORS_READ_WORKERS`, `FASTERSAFETENSORS_COORDINATED_READERS`, `FASTERSAFETENSORS_RESIDENT_GIB` | May differ. They control local reads and waiting. |

`FASTERSAFETENSORS_CHUNKS` forces a chunk count. Invalid integer settings or
values below their minimum raise `PlanError` at import. A group signature
mismatch also raises `PlanError`. Use `FASTERSAFETENSORS_TRACE=1` for read
diagnostics.

## Read and write engines

The read engine fills caller-owned host or device destinations. File workers
follow the plan's `files_in_flight`, `readers_per_file`, and `split_bytes`.
The first failure stops further scheduling and reports the path and range.
Memory checks count result and scratch allocations against driver-free memory
plus reusable Torch allocator cache. A passing check still leaves contiguous
allocation subject to fragmentation.

A `CuFile` plan uses direct device reads. A registration failure switches that
file to pread with staging. An unavailable library or missing direct reader
switches the whole job. Other cuFile failures raise `ReadError::DirectRead`.
The report records each fallback and counts host, staged, and cuFile bytes.

The write engine emits the header and each payload part separately. CUDA parts
pass through four 16 MiB pinned buffers, overlapping device copies with writes.
The same header construction and tensor order serve file writes and in-memory
serialization.

### Selections

A `Selection` is a box of half-open ranges. It yields contiguous byte runs in
row-major order, merging fully selected inner dimensions. Sub-byte dtypes
require every run to begin and end on a byte boundary; a misaligned run raises
`SelectError::SubByteMisaligned`.

The planner can combine adjacent runs by reading their gaps. It chooses:

```text
max_gap_bytes  = clamp(io_cost_us × single_file_gbps × 1000, 64 KiB, 8 MiB)
max_read_bytes = 16 MiB
```

An unmeasured `io_cost_us` uses the 64 KiB floor. Setting `max_gap_bytes` to zero
disables coalescing. `explain()` reports the run count, read count, and extra
bytes.

Each combined read carries sorted, disjoint placements into a contiguous
destination. Validation checks their ranges and total size. Host reads use a
worker scratch buffer. Device reads with placements use staging, including
under a `CuFile` plan. Regular two-dimensional placements use one
`cudaMemcpy2DAsync`; irregular placements use one copy per placement. Reports
count `gap_bytes`, `placed_pieces`, and `scatter_copies`.

### CUDA runtime

`fst-cuda` loads `libcudart` and `libcufile` with `dlopen`, checking sonames
before toolkit paths. Missing libraries raise `CudaError::Unavailable` with
the attempted paths. `fst_cuda::probe()` reports availability.

Copies use a non-blocking stream per device. Python synchronizes Torch before
passing device pointers to Rust. The engine completes its copies before
returning. Per-thread fences let workers wait for their own transfers while
other workers continue. Pinned buffers are pooled to avoid repeated allocation.
Other accelerators, such as MPS, receive data through host tensors.

## Calibration profile

The built-in `Profile::default_measured()` matches
[the H100 profile](profiles/h100-nfs-2026-09-08.json). A test checks this match.
Set `FASTERSAFETENSORS_PROFILE` to merge a JSON override into it. Profile
selection is explicit; `explain()` reports that applicability to the current
machine remains unverified. Invalid files raise `ProfileError`.

An override replaces each storage class or mount entry it supplies. Other
entries stay in place. A supplied device entry replaces the base device entry,
and the source becomes `<override.source> over <base.source>`.

### Schema version 1

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be `1`. |
| `source` | Hardware type, storage, and measurement date, with notes on unmeasured entries. It names hardware by type, not by host name. Planning reasons include it. |
| `storage` | Entries keyed by `LocalBlock`, `Nfs`, `Fuse`, `Ram`, `OtherNetwork`, or `Other`. |
| `mounts` | Entries keyed by mount point. The longest matching path prefix overrides the storage class. |
| `device` | Host-to-device costs, or `null`. |

`storage`, `mounts`, and `device` are optional. Each storage entry contains:

| Field | Constraint and use |
| --- | --- |
| `single_file_gbps` | Positive cold throughput for one file, in decimal GB/s. |
| `aggregate_gbps` | Nonempty `{files_in_flight, gbps}` table with strictly increasing file counts and positive rates. |
| `split_helps` | Whether extra readers improve one file's throughput. |
| `page_cache_gbps` | Optional positive warm throughput. |
| `open_cost_ms` | Optional nonnegative open cost. |
| `io_cost_us` | Optional nonnegative fixed read cost used for coalescing. |
| `gds` | Optional `{registers, read_gbps}` with registration support and an optional measured read rate. |

Optional measurement fields default to `null`. The planner chooses the smallest
file count within 15% of maximum aggregate throughput, capped by the request's
file count. With `split_helps`, it uses up to `min(cpus, 16)` readers per file
and 64 MiB pieces.

The `device` entry contains positive `h2d_pinned_gbps` and `h2d_pageable_gbps`,
plus nonnegative `pinned_alloc_ms_per_mib`. Validation rejects unknown fields,
unknown storage classes, invalid rates, unsupported versions, and malformed
aggregate tables.

The planner selects `CuFile` only when `nvidia_fs` and `libcufile` are available
and every storage entry touched by the request has `gds.registers: true`.

### Default measurements and assumptions

The H100 profile uses these inputs:

| Entry | Basis |
| --- | --- |
| `Nfs` | 3.0 GB/s per file; aggregate cold rates of 2.6, 7.4, 8.3, and 9.7 GB/s at 1, 4, 16, and 32 files; warm rate 25.9 GB/s. Splitting one file gives little benefit. |
| `Nfs.io_cost_us` | Unmeasured placeholder of 200 µs. |
| `LocalBlock` | Unmeasured placeholder using the NFS table through four files, split readers, and assumed GDS registration support. |
| `Ram` | Warm page-cache rates used as estimates: 7.2, 20.0, and 25.9 GB/s at 1, 4, and 16 readers. |
| `device` | Measured pinned transfer rate of 55 GB/s and allocation cost of 0.4 ms/MiB. Pageable transfer rate uses an unmeasured 7.2 GB/s estimate. |

Missing storage classes use conservative NFS defaults. Replace estimates with
measurements for the deployment. The
[B200 profile](profiles/b200-nfs-2026-09-09.json) records a separate NFS setup.

## Loading measurements

These measurements used an H100 80 GB, 16 CPU cores, 2 TB RAM, and NFSv3 with
`nconnect=16`. The checkpoint had 26 shards, 1,045 tensors, and 71.9 GB of data.
Cold runs evicted shard pages with `posix_fadvise(DONTNEED)` and checked page
cache growth. Warm runs reused cached pages.

The 2026-09-08 measurements behind
[the H100 profile](profiles/h100-nfs-2026-09-08.json) include warm repetitions
within one process:

| Reader | Cold GB/s | Warm GB/s |
| --- | --- | --- |
| safetensors, sequential | 2.6 | 7.2 |
| safetensors, 4 files in flight | 7.4 | 20.0 |
| safetensors, 16 files in flight | 8.3 | 25.9 |
| fastsafetensors without GDS, 32 threads | 9.7 | 17.7 |
| fastsafetensors, one file with 16 readers | 2.9–3.1 | 23.0 |

Fresh-process warm runs of threaded safetensors reached 13.4–13.7 GB/s with
four files and 11.3 GB/s with sixteen. Initial allocation of 1,045 device
tensors accounted for about three seconds in that setup.

An integrated comparison on 2026-09-09 used a fresh process for every run and
produced the same checksum in each arm. Timing covered `load_files` and a final
CUDA synchronization; context creation preceded the timer.

| Reader | Cold seconds | Cold GB/s | Warm seconds | Warm GB/s | Peak allocated GB |
| --- | --- | --- | --- | --- | --- |
| fastersafetensors `load_files` | 7.95 | 9.0 | 2.18 | 32.9 | 71.9 |
| fastsafetensors without GDS, 16 threads | 8.09 | 8.9 | 2.91 | 24.7 | 71.9 |
| safetensors, 16 files in flight | 10.22 | 7.0 | 6.19 | 11.6 | 71.9 |

The fastersafetensors plan used 16 files in flight, one reader per file, pread,
and 32 staging buffers of 16 MiB. These are loading measurements on that host;
the benchmark scripts that produced them are not part of this repository.
Measure the target deployment before relying on these rates.

## Development checks

Install the Rust version from `rust-toolchain.toml` and put `rustup` on `PATH`.
`uv sync` builds the extension. Run:

```bash
cargo fmt \
    --all \
    -- --check
uv run cargo clippy \
    --workspace \
    --all-targets \
    -- -D warnings
cargo test \
    -p fst-core \
    -p fst-cuda
uv run pytest tests/io/fastersafetensors
```

CUDA integration tests require a suitable host and `FST_CUDA_TESTS=1`.
Workspace lints reject `unwrap`, `expect`, and `panic` outside tests.
