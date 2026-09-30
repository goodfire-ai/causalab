"""Fixtures for the reference-engine tests.

The oracle side reuses ``tests/neural/activations/hook_oracle.py`` verbatim
— its helpers only touch ``pipeline.hf_model``, so a one-field shim carries
the engine's loaded model into the oracle unchanged (same assertions, same
tolerances, new stack under test)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.parallel.agreements import AGREEMENT_VARIABLE

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
#: The hookpoint-vocabulary target architecture in miniature: a real hybrid
#: stack (layers 0-2 Gated DeltaNet, layer 3 full attention) with a sparse MoE
#: in every layer. Deliberately *not* in the parametrized ``bundle`` fixture:
#: the oracle suites pin family-specific tensors the oracle has no MoE/DeltaNet
#: entry for. The site resolver *can* address a DeltaNet layer (the mixer is
#: resolved per layer), so ask for ``qwen35moe_bundle`` explicitly.
TINY_QWEN35_MOE = "tiny-random/qwen3.5-moe"

BASE_TEXT = "the quick brown fox jumps"
COUNTERFACTUAL_TEXT = "a slow green turtle sleeps deeply"


@dataclasses.dataclass(frozen=True)
class OracleShim:
    """The one attribute the hook-oracle helpers read."""

    hf_model: Any


@pytest.fixture(scope="session", params=["llama", "gpt2"])
def bundle(request: pytest.FixtureRequest) -> ModelBundle:
    key = TINY_LLAMA if request.param == "llama" else TINY_GPT2
    return load_model(key)


@pytest.fixture(scope="session")
def llama_bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture(scope="session")
def qwen35moe_bundle() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


@pytest.fixture()
def oracle(bundle: ModelBundle) -> OracleShim:
    return OracleShim(hf_model=bundle.model)


# --------------------------------------------------------------------------- #
# the §7 gradient agreement check, on in every training scenario and smoke
# --------------------------------------------------------------------------- #

#: What the simulated training scenarios set ``CAUSALAB_GRADIENT_AGREEMENT``
#: to (``docs/model_parallelism.md`` §7): bit identity across the ranks
#: before the guard's mean — the simulator's fixed-order collectives deliver
#: it, and every simulated fit is held bit-identical to world 1 anyway.
GRADIENT_AGREEMENT_SIMULATED = "0"
#: Real-backend gradient agreement, relative to the largest entry. The
#: reference gloo fp32 fits on CPU have zero measured disagreement. Allow
#: about eight fp32 ulps for larger groups and backend reduction order,
#: including the NCCL variants of these smokes. This remains over five
#: orders below the 1 - 1/size >= 1/2 disagreement of a partial gradient
#: (agreements.relative_disagreement), so broken gradient pairing fails.
#: Re-measure the reference when a new backend has nonzero disagreement.
GRADIENT_AGREEMENT_GLOO = "1e-6"
GRADIENT_AGREEMENT_GLOO_MEASURED = 0.0
assert float(GRADIENT_AGREEMENT_GLOO) > GRADIENT_AGREEMENT_GLOO_MEASURED
assert float(GRADIENT_AGREEMENT_GLOO) < 0.5 / 1e5, "five orders below a partial"


@pytest.fixture
def checked_gradients_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The §7 check on at bit identity for a simulated training scenario:
    the real ``run_cohort_training`` reads the variable once per fit."""
    monkeypatch.setenv(AGREEMENT_VARIABLE, GRADIENT_AGREEMENT_SIMULATED)


@pytest.fixture
def checked_gradients_gloo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The §7 check on at the measured band for a ``gloo`` training smoke:
    the spawned ranks inherit the environment, so every fit the smoke runs
    — the parity runs and the torchrun-style mutation children alike — is
    checked, and a pairing mutation is refused through the variable."""
    monkeypatch.setenv(AGREEMENT_VARIABLE, GRADIENT_AGREEMENT_GLOO)
