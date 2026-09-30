"""The Hydra-effect residual accounting through the readout adapter on the
real tower (golden tier): the readout adapter must reproduce the accepted
parity fixture without model-specific architecture branches.

``Qwen/Qwen3.6-35B-A3B`` in bf16 on cuda — the same accounting as the CPU
tier (``tests/analysis/test_readout.py``, through
``tests/_helpers/readout_accounting.py``), the same assertion shape: the
adapter's readout is the engine's ``lm_head`` read bit for bit; the closure
is at the reference projection's noise floor with both declared rounding
terms; each rounding term is within the declared band of bf16 roundoff at the
value's magnitude; and dropping either term breaks the closure by exactly
that term. An earlier version of this accounting, *with* architecture
branches, measured this fixture at closure 1.4e-14, without-norm 1.9e-2 and
without-head 5.7e-2 — those numbers are the shape this test expects
(exhibited, well under the band), not pins.

One model resident, in this module only: the bundle is module-scoped, and
``tests/golden/conftest.py`` empties the loader caches and drains the
accelerator at both module boundaries — this module sorts after
``test_paper_goldens.py`` and hit a CUDA OOM on their cached models before
the boundary owned that.
"""

from __future__ import annotations

import pytest
import torch

from causalab.analysis.logit_lens import CERTIFICATION_ULPS, Readout, unit_roundoff
from causalab.protocol.registry import DOCS_TABLE_MODEL

from tests._helpers import readout_accounting as accounting

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def bundle():
    if not torch.cuda.is_available():
        pytest.skip("the golden tier is the accelerator tier")
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    return load_model(DOCS_TABLE_MODEL, dtype="bf16", device="cuda")


@pytest.fixture(scope="module")
def acc(bundle) -> accounting.Accounting:
    result = accounting.account(bundle)
    # the measured terms, for the run's log
    print({k: v for k, v in result.as_dict().items() if k != "certificate"})
    print(result.as_dict()["certificate"])
    return result


def test_the_tower_is_the_qwen35_moe_text_family_declared_one_plus_weight(bundle):
    readout = Readout.from_bundle(bundle)
    assert readout.family == "qwen3_5_moe_text"
    assert readout.spec.gain == "one_plus_weight"
    assert readout.spec.norm == "rmsnorm"
    assert len(bundle.blocks) == 40


def test_the_adapters_readout_is_the_engines_lm_head_read_bit_for_bit(acc):
    assert acc.bit_exact_logits
    assert acc.dtype == "bfloat16"


def test_the_accounting_closes_with_both_terms_and_breaks_without_either(acc):
    assert accounting.problems(acc) == [], acc.as_dict()


def test_the_fixture_exhibits_both_rounding_terms_at_bf16_scale(acc):
    """The shape the earlier version measured: both terms are bf16 roundings of their
    values — far above the fp64 floor, under the band — and the dropped
    closures are those terms."""
    roundoff = unit_roundoff(torch.bfloat16)
    floor = max(acc.projection_noise_floor, 1e-12)
    for term, magnitude in (
        (acc.normalization_rounding, acc.ln_final_absmax),
        (acc.lm_head_rounding, acc.logits_absmax),
    ):
        assert (
            1e-4 * roundoff * magnitude
            < term
            <= CERTIFICATION_ULPS * roundoff * magnitude
        )
        assert term > 1e6 * floor
    assert acc.closure <= 10 * floor
    assert acc.closure_without_normalization_term > 1e6 * floor
    assert acc.closure_without_lm_head_term > 1e6 * floor


def test_the_declared_gain_certifies_and_the_other_is_far(acc):
    cert = acc.certificate
    assert cert.dtype == "bfloat16" and cert.gain == "one_plus_weight"
    assert cert.gap <= cert.tolerance
    assert cert.gaps["weight"] > cert.tolerance
    assert cert.gaps["weight"] > 100 * cert.gap
