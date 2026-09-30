"""Run the pytorch_hooks engine on the library's own kernels.

The engine can swap optional kernels into the model's forward — the
single-chunk Gated DeltaNet kernel for short sequences
(``causalab.neural.shared.gdn_short``, ``CAUSALAB_GDN_SHORT_SEQ``) and the fused
MoE glue (``causalab.neural.engines.pytorch_hooks.kernels.moe_glue``,
``CAUSALAB_MOE_GLUE``). Each is certified against the library by its own goldens
at its own documented band. A comparison of the engine against something that
runs the library's operations — the raw-hook oracle of the family
certification, the nnsight engine of the A3B parity sweep — must therefore hold
the engine to the library path too, or it measures the kernels' rounding
instead of what it set out to measure. `library_kernel_paths` is that
switch: the options are read from the environment at dispatch time, so setting
the environment for the duration is exactly what the executor honours.
`kernel_paths_in_force` renders what the environment says right now, for
a record that wants to attest the setting it was captured under rather than
assume it.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator, Mapping

from causalab.neural.shared.gdn_short.options import ENV_SHORT_SEQ
from causalab.neural.shared.kernel_options import ENV_MOE_GLUE

__all__ = [
    "LIBRARY_KERNEL_PATHS",
    "LIBRARY_KERNEL_PATHS_IN_FORCE",
    "UNSET",
    "kernel_paths_in_force",
    "library_kernel_paths",
]

#: every optional kernel path of the engine, and the value that switches it off
LIBRARY_KERNEL_PATHS: dict[str, str] = {ENV_SHORT_SEQ: "0", ENV_MOE_GLUE: "off"}

#: the marker of a variable the environment does not set
UNSET = "unset"


def _render(values: Mapping[str, str]) -> str:
    """The kernel-path variables as ``values`` holds them, rendered
    ``NAME=value`` sorted by name and joined by ``;`` (``NAME=unset`` for an
    absent one; the values themselves may hold commas) — a record's
    self-describing provenance of the setting."""
    return ";".join(
        f"{name}={values.get(name, UNSET)}" for name in sorted(LIBRARY_KERNEL_PATHS)
    )


def kernel_paths_in_force() -> str:
    """What `_render` makes of the environment right now."""
    return _render(os.environ)


#: what `kernel_paths_in_force` renders inside `library_kernel_paths`
LIBRARY_KERNEL_PATHS_IN_FORCE: str = _render(LIBRARY_KERNEL_PATHS)


@contextlib.contextmanager
def library_kernel_paths() -> Iterator[None]:
    """Every optional kernel path off for the duration; the environment is
    restored on exit, absent variables included."""
    saved = {name: os.environ.get(name) for name in LIBRARY_KERNEL_PATHS}
    os.environ.update(LIBRARY_KERNEL_PATHS)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
