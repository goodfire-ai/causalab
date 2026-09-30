"""The kernel families' on/off sets (``neural/shared/kernel_options.py``:
``KernelOptions`` as ``MoeGlueOptions`` and ``FusedNormOptions``) — the
engine-free half: what the environment can say and what is refused.
"""

from __future__ import annotations

import pytest

from causalab.neural.shared.kernel_options import (
    DEFAULT_FUSED_NORM_KERNELS,
    DEFAULT_MOE_GLUE_KERNELS,
    ENV_FUSED_NORMS,
    ENV_MOE_GLUE,
    FUSED_NORM_KERNELS,
    MOE_GLUE_KERNELS,
    FusedNormOptions,
    KernelOptionError,
    KernelOptions,
    MoeGlueOptions,
)

pytestmark = pytest.mark.unit


class TestMoeGlueOptions:
    def test_unset_or_empty_is_every_kernel(self) -> None:
        for environ in ({}, {ENV_MOE_GLUE: ""}, {ENV_MOE_GLUE: "  "}):
            options = MoeGlueOptions.from_env(environ)
            assert options.is_default and options.kernels == DEFAULT_MOE_GLUE_KERNELS
        assert DEFAULT_MOE_GLUE_KERNELS == frozenset(MOE_GLUE_KERNELS)

    def test_off_all_and_a_subset(self) -> None:
        assert MoeGlueOptions.from_env({ENV_MOE_GLUE: "off"}).kernels == frozenset()
        assert MoeGlueOptions.from_env({ENV_MOE_GLUE: "ALL"}).kernels == frozenset(
            MOE_GLUE_KERNELS
        )
        subset = MoeGlueOptions.from_env({ENV_MOE_GLUE: "sort, gate"})
        assert subset.kernels == frozenset({"sort", "gate"})
        assert subset.enabled("gate") and not subset.enabled("gather")
        assert not subset.is_default

    def test_an_unknown_kernel_is_refused_by_name(self) -> None:
        with pytest.raises(
            KernelOptionError, match=r"moe_glue='histc'.*no such kernel"
        ) as info:
            MoeGlueOptions.from_env({ENV_MOE_GLUE: "sort,histc"})
        assert (info.value.option, info.value.value) == ("moe_glue", "histc")

    def test_the_process_environment_is_the_default_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_MOE_GLUE, "off")
        assert MoeGlueOptions.from_env().kernels == frozenset()
        monkeypatch.delenv(ENV_MOE_GLUE)
        assert MoeGlueOptions.from_env().is_default


class TestFusedNormOptions:
    def test_unset_or_empty_is_every_kernel(self) -> None:
        for environ in ({}, {ENV_FUSED_NORMS: ""}, {ENV_FUSED_NORMS: "  "}):
            options = FusedNormOptions.from_env(environ)
            assert options.is_default and options.kernels == DEFAULT_FUSED_NORM_KERNELS
        assert DEFAULT_FUSED_NORM_KERNELS == frozenset(FUSED_NORM_KERNELS)
        assert FUSED_NORM_KERNELS == ("norm", "gated_norm", "rotary")

    def test_off_all_and_a_subset(self) -> None:
        assert (
            FusedNormOptions.from_env({ENV_FUSED_NORMS: "off"}).kernels == frozenset()
        )
        assert FusedNormOptions.from_env({ENV_FUSED_NORMS: "ALL"}).kernels == frozenset(
            FUSED_NORM_KERNELS
        )
        subset = FusedNormOptions.from_env({ENV_FUSED_NORMS: "norm, rotary"})
        assert subset.kernels == frozenset({"norm", "rotary"})
        assert subset.enabled("rotary") and not subset.enabled("gated_norm")
        assert not subset.is_default

    def test_an_unknown_kernel_is_refused_by_name(self) -> None:
        with pytest.raises(
            KernelOptionError, match=r"fused_norms='layernorm'.*no such kernel"
        ):
            FusedNormOptions.from_env({ENV_FUSED_NORMS: "norm,layernorm"})


def test_the_grammar_alone_names_no_family() -> None:
    """``KernelOptions`` is the grammar; only a subclass has kernels."""
    with pytest.raises(TypeError, match="names no kernel family"):
        KernelOptions()
    with pytest.raises(TypeError, match="names no kernel family"):
        KernelOptions.from_env({})
    assert issubclass(MoeGlueOptions, KernelOptions)
    assert issubclass(FusedNormOptions, KernelOptions)
