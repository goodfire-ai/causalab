# causalab.neural

Two engines execute the [intervention protocol](../../docs/intervention_protocol.md).
They share planning and tensor operations through `shared/`.

| Module | Responsibility |
|---|---|
| `shared/` | Engine routing, site resolution, device frames, write math, featurizers, metrics, sweeps, and result records |
| `engines/pytorch_hooks/` | Module hooks and function-interior taps; supports training |
| `engines/nnsight_tracing/` | One trace per forward group, with `.source` access to fused interiors |

`shared/engine_router.py` resolves `--engine`; `auto` selects `pytorch_hooks`.
The selected engine must provide the capabilities the document requires.
A mismatch raises `[V13]` before loading weights. Validation reads registry
metadata and keeps numerical libraries out of the import path.

Token-position resolution lives in `causalab/protocol/positions/`.
`shared/encoding.py` wraps its frames in device tensors. Task-side position
helpers live in `causalab/tasks/token_positions.py`.

Parity tests compare both engines on tiny fixtures and Qwen3.6-35B-A3B.
See [Architecture](../../docs/CODEBASE.md) for implementation boundaries and
[Running experiments](../../docs/running_experiments.md) for engine support.
