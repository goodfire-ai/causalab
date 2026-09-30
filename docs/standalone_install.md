# Install from a wheel and dependency lock

Use a wheel and `requirements.lock.txt` to install causalab in a separate
environment. The lock pins dependency versions and artifact SHA-256 hashes.
`scripts/export_requirements_lock.py --check` confirms that it matches
`uv.lock`.

## Build the wheel

From a checkout:

```bash
uv build --wheel
```

The wheel contains the Rust extension `causalab.io.fastersafetensors._core`.
The build needs Rust on `PATH`, using the version in `rust-toolchain.toml`.
Build for the recipient's platform. The `abi3` wheel supports CPython 3.10 and
later on that platform; installing it requires no Rust toolchain.

Distribute the wheel and `requirements.lock.txt` from the same commit. See
[the weight-loader guide](fastersafetensors.md) for the extension's requirements.

## Install

Create an environment and install the dependencies with hash verification.
Then install the wheel with dependency resolution disabled.

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install \
    --require-hashes \
    -r requirements.lock.txt
/path/to/venv/bin/pip install --no-deps causalab-*.whl
/path/to/venv/bin/pip check
/path/to/venv/bin/causalab --help
```

`--require-hashes` rejects missing hashes and artifacts whose bytes differ
from the lock. The package is installed separately because the export uses
`--no-emit-project`; a local checkout path would be unusable for the recipient.

The method documents are not part of the wheel. They live in the repository
under `demos/methods/`, so the commands below run from a checkout and name
them by that path.

The CLI resolves packaged task tables automatically. Supply external datasets
through `--data-root`. The CPU example below uses fixture data from a checkout.

## Check the installed package

Build a wheel and run the install steps above. Then use the checkout's
documents and fixture data with the installed package:

```bash
/path/to/venv/bin/causalab validate demos/methods/protocols/interchange.json \
    --engine auto \
    --data-root tests/protocol/fixtures/data

/path/to/venv/bin/causalab run demos/methods/protocols/minimal_cpu.json \
    --engine auto \
    --data-root tests/protocol/fixtures/data \
    --artifacts-root tests/protocol/fixtures/artifacts \
    --out /tmp/standalone-run
```

Validation uses the static model registry. The CPU run downloads a small,
revision-pinned random Llama and writes exact-match and logit-difference metrics.
It checks model execution and artifact output.

A second check runs every demo and shipped workflow document:

```bash
python scripts/standalone_smoke.py --bin /path/to/venv/bin/causalab
```

This script runs `validate --data` and `explain` from a temporary working
directory. It catches references that depend on the checkout and modules that
need undeclared dependencies. Use `--select <substring>` to select a document.

## Optional dependencies

The lock covers the base install. Extras have separate requirements:

| Extra | Purpose |
|---|---|
| `notebook` | Jupyter and interactive causal-graph apps |
| `nnsight` | The tracing engine |
| `flash-attn`, `flash-linear-attention` | Optional Linux GPU kernels; see [attention backends](attention_backends.md) |

A Git dependency has no distribution hash and cannot appear in a pip lock
that uses `--require-hashes`. Install an extra from the same wheel by naming
its exact filename, followed by `[notebook]` or the required extra:

```bash
/path/to/venv/bin/pip install '/path/to/causalab-<version>-<python>-<abi>-<platform>.whl[notebook]'
```

The checkout pins nnsight through `[tool.uv.sources]`. Pip does not read that
setting from a wheel. To use the verified engine revision in a wheel install,
install the Git dependency explicitly:

```bash
/path/to/venv/bin/pip install 'nnsight @ git+https://github.com/ndif-team/nnsight.git@8c480727'
```

Installing extras resolves dependencies and can change versions pinned by the base
lock. Keep a separate environment for strict reproduction, or compare
`pip freeze` with the lock after installing extras. Pip rejects constraints
files that contain the lock's `--hash` entries.

## Update the lock

```bash
uv run python scripts/export_requirements_lock.py
uv run python scripts/export_requirements_lock.py --check
```

Regenerate the lock in the same commit as changes to core dependencies or
`uv.lock`. Resolve merge conflicts by regenerating it. The export preserves
platform markers, so one file serves each supported platform.

Export formatting depends on the uv version. Regenerate the lock with the uv
version that wrote it, and regenerate it again after changing that version.
