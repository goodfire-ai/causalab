"""Per-family raw-hook oracle certification of the real hybrid tower.

``qwen36_a3b`` is ``Qwen/Qwen3.6-35B-A3B`` in bf16 on cuda — the same case
table as the CPU family (``tests/neural/parity/family_certification.py``), the
same oracle, the bf16 band declared and justified by measurement in the
committed record ``tests/neural/parity/goldens/qwen36_a3b.json``. Unlike
``test_a3b_engine_parity.py`` beside it, which compares causalab's two engines,
this compares causalab against an *independent* raw-hook oracle.

One model resident, in this module only: the bundle is a module-scoped fixture,
and ``tests/golden/conftest.py`` empties the loader caches and drains the
accelerator at both module boundaries, so a single ``-m golden`` process
neither inherits the previous module's models nor carries ~70 GB into the
next one.
"""

from __future__ import annotations

import pytest
import torch

from tests._helpers.kernel_paths import LIBRARY_KERNEL_PATHS_IN_FORCE
from tests.neural.parity import family_certification as fc

pytestmark = pytest.mark.golden

FAMILY = "qwen36_a3b"


@pytest.fixture(scope="module")
def capture():
    if not torch.cuda.is_available():
        pytest.skip("the golden tier is the accelerator tier")
    return fc.capture_family(fc.make_bundle(FAMILY), FAMILY)


@pytest.fixture(scope="module")
def committed() -> dict:
    path = fc.record_path(FAMILY)
    assert path.exists(), (
        f"{path} is missing: {fc.FAMILIES[FAMILY]!r} was never captured"
    )
    return fc.load_record(FAMILY)


def test_the_engine_certifies_against_the_oracle_at_the_bf16_band(capture):
    band = fc.BANDS["bf16"]
    failures = fc.certification_failures(capture, band)
    assert not failures, "\n".join(f"{d!r}  {n}" for n, d in failures)


def test_the_committed_record_replays(capture, committed):
    problems = fc.compare_records(committed, capture.record)
    assert not problems, "\n".join(problems)


def test_the_capture_runs_the_library_kernel_paths(capture):
    """What was in force for this capture is the library setting (module
    docstring): ``capture_family`` wraps its body."""
    assert capture.record["context"]["kernel_paths"] == LIBRARY_KERNEL_PATHS_IN_FORCE


def test_the_record_carries_the_eight_fields(committed):
    missing, extra = fc.check_certification_fields(committed)
    assert not missing and not extra, (missing, extra)
    # `compare_records` leaves `context` alone, so the provenance of the kernel
    # setting is read here: the committed record was captured on the library path
    assert committed["context"]["kernel_paths"] == LIBRARY_KERNEL_PATHS_IN_FORCE
    assert committed["certification"]["dtype"] == "bf16"
    assert (
        committed["certification"]["model_revision"]["key"] == fc.FAMILIES[FAMILY].key
    )


def test_the_tower_is_the_documented_hybrid_schedule(capture):
    """Layer 0 Gated DeltaNet, layer 3 full attention — the first of each on
    the documented 3-linear-then-1-full schedule."""
    assert capture.record["layers"] == {"delta": 0, "full": 3}


def test_every_genuine_write_moves_the_logits_and_the_intermediate(capture):
    """T7 on the real checkpoint."""
    values = capture.record["values"]
    deltas = {k: v for k, v in values.items() if k.endswith(".clean_delta.max")}
    assert len(deltas) == len(fc.WRITES) + sum(len(w.downstream) for w in fc.WRITES)
    for key, moved in deltas.items():
        assert moved > 0.0, key


def test_a_no_op_swap_is_bit_exact_except_the_documented_path_forcing(capture):
    noop = capture.record["measurements"]["noop_swap_logit_drift"]
    bound = fc.BANDS["bf16"].state_substitution_drift
    seen_state = False
    for case_id, drift in noop.items():
        if ".interchange.delta_state." in case_id:
            seen_state = True
            assert 0.0 < drift <= bound, (case_id, drift)
        else:
            assert drift == 0.0, (case_id, drift)
    assert seen_state
    # necessary but insufficient (IOI V2): the genuine state interchange must
    # move the logits by more than the path-forcing drift the no-op costs
    layer = capture.record["layers"]["delta"]
    write_id = (
        f"{FAMILY}.interchange.delta_state.identity.step{fc.STATE_WRITE_STEP}.L{layer}"
    )
    genuine = capture.record["values"][f"{write_id}.clean_delta.max"]
    assert genuine > noop[write_id], (genuine, noop[write_id])
