"""``tests/_helpers/kernel_paths.py``: the switch the engine-vs-library
comparisons rest on turns every optional kernel path off for its duration,
leaves the environment as it found it, and renders what is in force."""

from __future__ import annotations

import os

import pytest

from causalab.neural.shared.gdn_short import triton_kernel
from causalab.neural.shared.gdn_short.options import (
    DEFAULT_SHORT_SEQ,
    ENV_SHORT_SEQ,
    ShortSeqKernelOptions,
)
from causalab.neural.shared.kernel_options import (
    DEFAULT_MOE_GLUE_KERNELS,
    ENV_MOE_GLUE,
    MoeGlueOptions,
)
from tests._helpers.kernel_paths import (
    LIBRARY_KERNEL_PATHS,
    LIBRARY_KERNEL_PATHS_IN_FORCE,
    UNSET,
    kernel_paths_in_force,
    library_kernel_paths,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def defaults_in_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both variables unset, on a machine where the short-sequence kernel would
    be on: the availability probe is stood up so the default is the real
    default (``DEFAULT_SHORT_SEQ``) rather than the Triton-less ``0`` that
    would equal the switched value and make the assertions vacuous."""
    monkeypatch.delenv(ENV_SHORT_SEQ, raising=False)
    monkeypatch.delenv(ENV_MOE_GLUE, raising=False)
    monkeypatch.setattr(triton_kernel, "triton_available", lambda: True)


def test_every_optional_kernel_path_reads_as_disabled_inside(defaults_in_force):
    assert ShortSeqKernelOptions.from_env().threshold == DEFAULT_SHORT_SEQ
    assert MoeGlueOptions.from_env().kernels == DEFAULT_MOE_GLUE_KERNELS
    with library_kernel_paths():
        assert not ShortSeqKernelOptions.from_env().enabled
        assert MoeGlueOptions.from_env().kernels == frozenset()
    assert ShortSeqKernelOptions.from_env().threshold == DEFAULT_SHORT_SEQ
    assert MoeGlueOptions.from_env().kernels == DEFAULT_MOE_GLUE_KERNELS


@pytest.mark.parametrize("preset", [None, "all", "7"])
def test_the_environment_is_restored_including_absent_variables(monkeypatch, preset):
    for name in LIBRARY_KERNEL_PATHS:
        if preset is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, preset)
    before = {name: os.environ.get(name) for name in LIBRARY_KERNEL_PATHS}
    with library_kernel_paths():
        assert {name: os.environ[name] for name in LIBRARY_KERNEL_PATHS} == (
            LIBRARY_KERNEL_PATHS
        )
    assert {name: os.environ.get(name) for name in LIBRARY_KERNEL_PATHS} == before


def test_restored_even_when_the_body_raises(monkeypatch):
    monkeypatch.setenv(ENV_MOE_GLUE, "sort")
    monkeypatch.delenv(ENV_SHORT_SEQ, raising=False)
    with pytest.raises(RuntimeError):
        with library_kernel_paths():
            raise RuntimeError("body")
    assert os.environ[ENV_MOE_GLUE] == "sort"
    assert ENV_SHORT_SEQ not in os.environ


def test_in_force_renders_the_live_environment(monkeypatch):
    monkeypatch.delenv(ENV_SHORT_SEQ, raising=False)
    monkeypatch.setenv(ENV_MOE_GLUE, "sort,gather")
    # a value may hold commas, so the pairs are joined by ";"
    assert kernel_paths_in_force() == (
        f"{ENV_SHORT_SEQ}={UNSET};{ENV_MOE_GLUE}=sort,gather"
    )
    with library_kernel_paths():
        assert kernel_paths_in_force() == LIBRARY_KERNEL_PATHS_IN_FORCE
    assert LIBRARY_KERNEL_PATHS_IN_FORCE == f"{ENV_SHORT_SEQ}=0;{ENV_MOE_GLUE}=off"
