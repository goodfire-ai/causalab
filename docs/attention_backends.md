# Optional attention backends

The `pytorch_hooks` engine uses eager full attention by default. The `nnsight`
engine uses the Transformers model default, usually SDPA. Supported Qwen
linear-attention models use PyTorch delta-rule and convolution kernels in a
base install.

## Install

On Linux with a supported NVIDIA GPU and CUDA-enabled PyTorch:

```bash
uv sync --extra flash-attn
uv sync --extra flash-linear-attention
# Select both extras to keep both installed.
uv sync \
    --extra flash-attn \
    --extra flash-linear-attention
```

`flash-attn` supplies FlashAttention 2. The `flash-linear-attention` extra
supplies FLA and `causal-conv1d` for Qwen's linear-attention layers. A runtime
install that uses nnsight also needs `--extra nnsight`; the dev group includes it.

The extras have Linux package markers. Use the base install on CPU, macOS, or
Windows. Source builds need a compatible CUDA toolkit with `nvcc` and a C++
compiler. FlashAttention 2 requires supported hardware and fp16 or bf16 inputs.
See the [FlashAttention installation guide](https://github.com/Dao-AILab/flash-attention#installation-and-features)
and [FLA installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md).
`MAX_JOBS` limits build parallelism and memory use.

The checkout builds extensions against its runtime PyTorch version. Static
package metadata lets `uv lock` resolve dependencies on a machine without CUDA.
For pip, install the base package and build tools before compiling the extras:

```bash
pip install -e .
pip install setuptools wheel packaging ninja
pip install \
    --no-build-isolation \
    -e '.[flash-attn,flash-linear-attention]'
```

Both engines bind CPU models to the Transformers PyTorch kernels during each
forward. This keeps CPU runs usable when the CUDA extras are installed.

### FLA training and tuning

FLA 0.5.2 rejects gated-delta backward on Hopper with Triton 3.4.0 through 3.7.0
because those kernels produce incorrect results. Install `tilelang`, which FLA
selects automatically, or use a compatible stack with Triton 3.7.1 or newer.

FLA 0.5.2 reads these environment settings:

| Setting | Effect |
| --- | --- |
| `FLA_TILELANG` | `1` selects tilelang backward; `0` selects Triton, subject to the Hopper version check. |
| `FLA_FLASH_QLA` | `0` disables FlashQLA. FLA can select an installed FlashQLA for K = V = 128 on SM90/SM100. |
| `FLA_CACHE_MODE`, `FLA_CONFIG_DIR` | Load per-kernel `num_warps`, `num_stages`, and `BV` from `<dir>/<kernel_name>.json`. |
| `FLA_USE_TMA`, `FLA_TRIL_PRECISION` | Control memory transfers and triangular-solve precision (`ieee`, `tf32`, or `tf32x3`). |
| `FLA_USE_FAST_OPS`, `FLA_DISABLE_BACKEND_DISPATCH` | Control FLA's kernel dispatch. |

FLA's `chunk_size` is a call argument with values 16, 32, or 64. The Transformers
wrapper filters this argument, so causalab uses the default of 64. Changing it
requires a modeling-code change.

Compiled kernels are cached on disk. To share them across jobs, see
[compilation caches](cuda_graphs.md#compilation-caches).

## Select a backend

Set the full-attention backend in a campaign or application document:

```json
{
  "model": {
    "key": "Qwen/Qwen3.6-35B-A3B",
    "revision": "main",
    "dtype": "bf16",
    "attn_implementation": "flash_attention_2"
  }
}
```

The field accepts `"eager"`, `"sdpa"`, or `"flash_attention_2"`. Both engines
honor it. Transformers raises an error when the model, hardware, or installed
dependencies cannot support the choice. Select this field explicitly to use
FlashAttention 2 after installing its extra.

A workflow can set the field on a protocol step:

```json
{
  "set": {
    "model.attn_implementation": "flash_attention_2",
    "model.dtype": "bf16"
  }
}
```

Sweep and bind wrappers also work. Each backend has a separate model cache
entry that can retain a full weight copy. Use separate processes for comparisons
when one model nearly fills GPU memory.

An explicit backend enters campaign, forward-group, and step digests. Tensor
and fitted-featurizer artifacts store it as `model_attn_implementation`. An
application checks compatibility when its document declares the field. Declare
the same backend in fit and apply documents to require this check. A caller-owned
model must also match the declared backend.

Omitting the field delegates the choice to the engine and gives a distinct
identity from explicit `"eager"`. Point receipts and tensor entries record
`loaded_attn_implementation`; `implementations` records eager requirements.
The engine constructor takes its backend choice from the document.

### Attention interiors and linear attention

The engines temporarily use eager attention for these operations:

| Engine | Tensors that require eager attention |
| --- | --- |
| `pytorch_hooks` | `attention_query`, `attention_key`, `attention_scores`, `attention_probs`, `attention_z` |
| `nnsight` | `attention_scores`, `attention_probs` |

Reads and writes at module boundaries keep the selected backend. These include
residuals, MLPs, attention outputs, and value projections. Prefill and decode use
one backend throughout a continuation. The engine restores the selection after
the forward, including on failure. Prefix caches separate backends. Hooks runs
record a temporary eager choice as `attn_eager` in implementation metadata.
These rules also apply to caller-owned models.

Transformers selects FLA and `causal-conv1d` separately when it imports Qwen's
linear-attention functions. Restart Python after installing or removing these
packages. Delta boundary taps wrap the selected functions; fused kernels can
expose different interior tensors.

## Return to the fallbacks

Run an exact sync and retain any other extras you need:

```bash
uv sync
```

With pip, remove `flash-attn`, `flash-linear-attention`, `fla-core`, and
`causal-conv1d`. Restart Python. Removing the linear-attention packages restores
the PyTorch kernels for those layers; `attn_implementation` controls full
attention separately.

## Reproducibility and batching variance

Changes to batch size, batch composition, padding, or row chunks can change
floating-point results. This affects logits, generated tokens, intervention
metrics, and fitted parameters even with a fixed random seed.

Keep the backend, precision, batching, hardware, and package versions fixed for
comparisons. Use numerical tolerances. PyTorch fallbacks also vary with hardware
and batching.
