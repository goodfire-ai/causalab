"""Select the longest sequence served by the short delta kernel.

``CAUSALAB_GDN_SHORT_SEQ`` defaults to 16 when Triton is available.
A value of zero keeps the bound kernel for all calls; the 32-row tile is
opt-in. Dot precision is fixed in ``triton_kernel``. FLA settings remain
separate and are listed in ``docs/attention_backends.md``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Mapping

from causalab.neural.shared.gdn_short import triton_kernel
from causalab.neural.shared.gdn_short.triton_kernel import MAX_SEQ_LEN

__all__ = [
    "CHUNK_KERNEL_GLOBAL",
    "DEFAULT_SHORT_SEQ",
    "ENV_SHORT_SEQ",
    "KernelOptionError",
    "ShortSeqKernelOptions",
]

#: The chunk kernel's module global — the one of `.kernels.KERNEL_GLOBALS`
#: the binding rebinds (the recurrent kernel has no chunk).
CHUNK_KERNEL_GLOBAL = "torch_chunk_gated_delta_rule"

#: The environment variable [`ShortSeqKernelOptions.from_env`][] reads: the
#: longest sequence routed to the single-chunk kernel, ``0`` to disable it,
#: unset for [`DEFAULT_SHORT_SEQ`][] where Triton is importable.
ENV_SHORT_SEQ = "CAUSALAB_GDN_SHORT_SEQ"

#: The default threshold: the kernel's 16-row tile, which the workflow's
#: 13-token sequences fill; its 32-row tile is opt-in through the variable.
DEFAULT_SHORT_SEQ = 16


class KernelOptionError(ValueError):
    """A kernel option that cannot be honoured: a value outside the kernel's
    accepted set."""

    def __init__(self, option: str, value: Any, reason: str) -> None:
        self.option = option
        self.value = value
        self.reason = reason
        super().__init__(f"kernel option {option}={value!r}: {reason}")


@dataclasses.dataclass(frozen=True)
class ShortSeqKernelOptions:
    """Where the single-chunk kernel takes over from the installed chunk
    kernel: a call whose sequence is at most ``threshold`` tokens (and starts
    from a zero state, on CUDA — [`.binding.selects_single_chunk`][causalab.neural.shared.gdn_short.binding.selects_single_chunk]) runs
    the single-chunk kernel; every other call runs what was bound.
    ``threshold == 0`` installs nothing."""

    threshold: int = 0

    def __post_init__(self) -> None:
        if self.threshold < 0 or self.threshold > MAX_SEQ_LEN:
            raise KernelOptionError(
                "short_seq",
                self.threshold,
                f"the single-chunk kernel covers sequences up to {MAX_SEQ_LEN} "
                "tokens; 0 disables it",
            )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ShortSeqKernelOptions":
        """``CAUSALAB_GDN_SHORT_SEQ`` as an integer threshold; unset or empty
        means [`DEFAULT_SHORT_SEQ`][] when Triton is importable (the kernel
        needs nothing else that a CUDA model does not already have) and
        disabled otherwise."""
        env = os.environ if environ is None else environ
        raw = env.get(ENV_SHORT_SEQ, "").strip()
        if not raw:
            # looked up on the module at call time, so the environment's
            # answer (and a test's stand-in for it) is read per call
            available = triton_kernel.triton_available()
            return cls(threshold=DEFAULT_SHORT_SEQ if available else 0)
        try:
            threshold = int(raw)
        except ValueError:
            raise KernelOptionError(
                "short_seq", raw, f"{ENV_SHORT_SEQ} must be an integer threshold"
            ) from None
        return cls(threshold=threshold)

    @property
    def enabled(self) -> bool:
        return self.threshold > 0
