"""Parse options for fused expert, norm, and rotary kernels.

``KernelOptions`` accepts an empty value or ``all`` for the full family,
``off`` for none, or a comma-separated subset. Unknown names raise an error.
``CAUSALAB_MOE_GLUE`` controls expert kernels; ``CAUSALAB_FUSED_NORMS``
controls ``norm``, ``gated_norm``, and ``rotary``.

Each selected kernel must also pass device and shape checks. The kernels
preserve their ATen operation order; these options support fault isolation.
FLA reads its own settings, documented in ``docs/attention_backends.md``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, ClassVar, Mapping, TypeVar

__all__ = [
    "DEFAULT_FUSED_NORM_KERNELS",
    "DEFAULT_MOE_GLUE_KERNELS",
    "ENV_FUSED_NORMS",
    "ENV_MOE_GLUE",
    "FUSED_NORM_KERNELS",
    "FusedNormOptions",
    "KernelOptionError",
    "KernelOptions",
    "MOE_GLUE_KERNELS",
    "MoeGlueOptions",
]

#: The environment variable `MoeGlueOptions.from_env` reads.
ENV_MOE_GLUE = "CAUSALAB_MOE_GLUE"

#: The fused glue kernels of the grouped-experts path, by name: the stable
#: counting sort, the row gather with its exact index backward, the
#: weight·un-sort·slot-sum epilogue, and ``silu(gate) * up``.
MOE_GLUE_KERNELS: tuple[str, ...] = ("sort", "gather", "epilogue", "gate")

#: The kernels on when nothing is set: every one, each proven bit-identical
#: to the ATen path on the H100 by ``tests/golden/test_moe_glue_kernels.py``
#: (📐 2026-09-15: output and all four gradients equal on the
#: workflow's shapes in bf16 / fp16 / fp32; the gate kernel's libdevice
#: ``exp`` / ``div_rn`` / ``fma`` match nvcc's compiled ``silu`` there).
DEFAULT_MOE_GLUE_KERNELS: frozenset[str] = frozenset(MOE_GLUE_KERNELS)

#: The environment variable `FusedNormOptions.from_env` reads.
ENV_FUSED_NORMS = "CAUSALAB_FUSED_NORMS"

#: The fused kernels of the decoder's elementwise tail, by name: the
#: family's RMSNorm (``x · rsqrt(mean(x²) + ε) · (1 + w)``), the DeltaNet
#: output's gated RMSNorm (``w · norm(x) · silu(gate)``) and the attention's
#: rotary embedding (``rotate_half`` without slices in either direction).
FUSED_NORM_KERNELS: tuple[str, ...] = ("norm", "gated_norm", "rotary")

#: The kernels on when nothing is set: every one (bit-identical to the
#: modules on the H100 by ``tests/golden/test_fused_norm_kernels.py``).
DEFAULT_FUSED_NORM_KERNELS: frozenset[str] = frozenset(FUSED_NORM_KERNELS)


class KernelOptionError(ValueError):
    """A kernel option that cannot be honoured: a value outside the option's
    accepted set."""

    def __init__(self, option: str, value: Any, reason: str) -> None:
        self.option = option
        self.value = value
        self.reason = reason
        super().__init__(f"kernel option {option}={value!r}: {reason}")


_O = TypeVar("_O", bound="KernelOptions")


@dataclasses.dataclass(frozen=True)
class KernelOptions:
    """Which kernels of one family may run — the grammar every family's
    switch shares (module docstring). A subclass names the family:
    ``OPTION`` (the error's label), ``ENV`` (the variable [`from_env`][]
    reads), ``KERNELS`` (the accepted names, in order) and the default
    ``kernels`` — every one of them."""

    OPTION: ClassVar[str]
    ENV: ClassVar[str]
    KERNELS: ClassVar[tuple[str, ...]]

    kernels: frozenset[str] = frozenset()

    @classmethod
    def _family(cls) -> tuple[str, str, tuple[str, ...]]:
        """``(OPTION, ENV, KERNELS)`` — or a ``TypeError`` for the bare
        grammar, which names no family: both entry points open on it."""
        if not hasattr(cls, "KERNELS"):
            raise TypeError(
                "KernelOptions names no kernel family; construct one of its"
                " subclasses (MoeGlueOptions, FusedNormOptions)"
            )
        return cls.OPTION, cls.ENV, cls.KERNELS

    def __post_init__(self) -> None:
        option, _, names = self._family()
        unknown = sorted(set(self.kernels) - set(names))
        if unknown:
            raise KernelOptionError(
                option, unknown[0], f"no such kernel; the kernels are {list(names)}"
            )

    @classmethod
    def from_env(cls: type[_O], environ: Mapping[str, str] | None = None) -> _O:
        """``ENV``: unset or empty is the default set, ``off`` none, ``all``
        every kernel, otherwise a comma-separated subset of ``KERNELS``."""
        _, variable, names = cls._family()
        env = os.environ if environ is None else environ
        raw = env.get(variable, "").strip().lower()
        if not raw:
            return cls()
        if raw == "off":
            return cls(kernels=frozenset())
        if raw == "all":
            return cls(kernels=frozenset(names))
        names = [name.strip() for name in raw.split(",") if name.strip()]
        return cls(kernels=frozenset(names))

    @property
    def is_default(self) -> bool:
        return self.kernels == frozenset(self.KERNELS)

    def enabled(self, kernel: str) -> bool:
        return kernel in self.kernels


@dataclasses.dataclass(frozen=True)
class MoeGlueOptions(KernelOptions):
    """Which fused glue kernels of the grouped-experts path may run
    (``CAUSALAB_MOE_GLUE``)."""

    OPTION: ClassVar[str] = "moe_glue"
    ENV: ClassVar[str] = ENV_MOE_GLUE
    KERNELS: ClassVar[tuple[str, ...]] = MOE_GLUE_KERNELS

    kernels: frozenset[str] = DEFAULT_MOE_GLUE_KERNELS


@dataclasses.dataclass(frozen=True)
class FusedNormOptions(KernelOptions):
    """Which fused norm / rotary kernels may run (``CAUSALAB_FUSED_NORMS``);
    a kernel named here still needs a frozen weight and a modelled width."""

    OPTION: ClassVar[str] = "fused_norms"
    ENV: ClassVar[str] = ENV_FUSED_NORMS
    KERNELS: ClassVar[tuple[str, ...]] = FUSED_NORM_KERNELS

    kernels: frozenset[str] = DEFAULT_FUSED_NORM_KERNELS
