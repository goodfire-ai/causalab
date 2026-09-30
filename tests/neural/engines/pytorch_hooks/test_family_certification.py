"""Per-family raw-hook oracle certification of the hybrid tower — the CPU
family.

``qwen35moe`` is ``tiny-random/qwen3.5-moe`` in fp32: every case of
``tests/neural/parity/family_certification.py`` is driven through the reference
engine and through the raw-hook oracle (``hook_oracle_lib``, extended to the
DeltaNet kernel boundary and the recurrence interior), compared at the
declared fp32 band — **exact** — and replayed against the committed record
``tests/neural/parity/goldens/qwen35moe.json``, which carries the
eight certification fields.

The acceptance clauses:

* **T5** — the record carries exactly the eight fields, censused against the
  one tuple that declares them; dropping ``max_logit_diff`` fails the guard.
* **T6** — the test that earns the certification: reassociate one sum in the oracle
  (``delta_kv_mem``, strict left-to-right instead of torch's blocked
  reduction — a one-ulp change) and the certification **fails**, first on an
  intermediate component (``delta_kv_mem``, then the state faces it feeds),
  while the logits move so little that every legacy logits-only band would have
  passed the injection. A certification that compared only logits is worthless
  against exactly this defect.
* **T7** — anti-vacuity, per write: the genuine interchange moves the logits
  *and* the downstream intermediate, and the no-op twin (swap a tensor for its
  own value) is bit-exact — except the state, whose no-op is the documented
  path-forcing drift, bounded and non-zero.
* **T8** is the existing tiers staying green on the same run (the CPU gate
  itself, and ``tests/golden`` for the accelerator half).

The real-checkpoint family (``qwen36_a3b``) replays under ``-m golden`` in
``tests/golden/test_family_certification_a3b.py``.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import torch

from tests._helpers.kernel_paths import LIBRARY_KERNEL_PATHS_IN_FORCE
from tests.neural.engines.pytorch_hooks import hook_oracle_lib as oracle_lib
from tests.neural.parity import family_certification as fc

pytestmark = pytest.mark.numerical_unit

FAMILY = "qwen35moe"
TESTS_DOC = Path(__file__).resolve().parents[4] / "docs" / "TESTS.md"

#: The write oracle's stack-vs-oracle band (``test_write_oracle.py``) and the
#: frozen goldens' replay band — the two logits tolerances a certification that
#: compared only logits would plausibly have inherited.
LEGACY_LOGIT_BANDS = (1e-5, 1e-4)


@pytest.fixture(scope="module")
def capture(qwen35moe_bundle) -> fc.Capture:
    return fc.capture_family(qwen35moe_bundle, FAMILY)


@pytest.fixture(scope="module")
def committed() -> dict:
    return fc.load_record(FAMILY)


# --------------------------------------------------------------------------- #
# the certification itself
# --------------------------------------------------------------------------- #


def test_the_engine_certifies_against_the_oracle_at_the_exact_band(capture):
    """Every intermediate component and the logits: engine == oracle at the
    fp32 band, which is 0.0 — measured, then declared."""
    band = fc.BANDS["fp32"]
    assert band.activation == 0.0 and band.logits == 0.0
    failures = fc.certification_failures(capture, band)
    assert not failures, "\n".join(f"{d!r}  {n}" for n, d in failures)
    # and the comparison was not vacuous: both layer kinds, the interior, the logits
    hooks = set(capture.record["certification"]["hook_names"])
    assert any(".linear_attn:" in h and "[state]" in h for h in hooks), hooks
    assert any(".self_attn.out" in h for h in hooks), hooks
    assert "lm_head" in hooks


def test_the_committed_record_replays(capture, committed):
    """The committed ``qwen35moe.json`` is what this tree produces: same model
    snapshot, same hooks and shapes, measured differences inside the committed
    band, pins within tolerance."""
    problems = fc.compare_records(committed, capture.record)
    assert not problems, "\n".join(problems)


def test_the_capture_runs_the_library_kernel_paths(capture):
    """The record attests what was in force, and what was in force is the
    library setting: ``capture_family`` wraps its body, and a dropped wrap
    fails here on every CPU run rather than at the next recapture."""
    assert capture.record["context"]["kernel_paths"] == LIBRARY_KERNEL_PATHS_IN_FORCE


def test_the_record_is_a_family_golden_in_the_existing_shape(committed):
    """The frozen goldens' keys are all there, with their meanings, and the
    values are keyed the way the frozen goldens key theirs."""
    for key in (
        "attn_implementation",
        "captured_from",
        "context",
        "deterministic",
        "family",
        "tolerance",
        "values",
    ):
        assert key in committed, key
    assert committed["family"] == FAMILY
    assert committed["captured_from"] == "hook_oracle"
    assert committed["attn_implementation"] == "eager"
    assert committed["deterministic"] is True
    assert set(committed["context"]) == {"torch", "transformers", "kernel_paths"}
    assert committed["context"]["kernel_paths"] == LIBRARY_KERNEL_PATHS_IN_FORCE
    assert committed["recapture"].endswith(f"--family {FAMILY}")
    grammar = re.compile(rf"^{FAMILY}\.(collect|interchange)\.[a-z_]+\.identity\.")
    assert committed["values"], "an empty pin set would replay vacuously"
    assert all(grammar.match(k) for k in committed["values"]), [
        k for k in committed["values"] if not grammar.match(k)
    ][:5]


def test_the_cases_cover_both_layer_kinds_and_the_interior(committed):
    """The brief's minimum: a collect read, an interchange write, a DeltaNet
    interior read *and* write, ``lm_head`` logits — on a DeltaNet layer and on a
    full-attention layer."""
    layers = committed["layers"]
    assert layers["delta"] != layers["full"]
    ids = {k.split(".out.")[0] for k in committed["values"] if ".out." in k}
    d, f = layers["delta"], layers["full"]
    for wanted in (
        f"{FAMILY}.collect.block_output.identity.all.L{d}",
        f"{FAMILY}.collect.block_output.identity.all.L{f}",
        f"{FAMILY}.collect.attention_output.identity.all.L{f}",
        f"{FAMILY}.collect.delta_state.identity.all.L{d}",
        f"{FAMILY}.collect.delta_kv_mem.identity.all.L{d}",
        f"{FAMILY}.collect.lm_head.identity.all",
        f"{FAMILY}.interchange.block_output.identity.last.L{d}",
        f"{FAMILY}.interchange.attention_output.identity.last.L{f}",
        f"{FAMILY}.interchange.delta_state.identity.step{fc.STATE_WRITE_STEP}.L{d}",
    ):
        assert wanted in ids, wanted


# --------------------------------------------------------------------------- #
# T5 — the eight fields, censused
# --------------------------------------------------------------------------- #


def _fields_documented_in_tests_md() -> set[str]:
    """The backticked field names in the pins-table row for the certified
    families — the doc half of the census (``test_vocabulary_census.py``'s
    pattern: the table that documents a closed set is held to the set)."""
    text = TESTS_DOC.read_text()
    marker = "the eight certification fields"
    assert marker in text, f"docs/TESTS.md no longer names {marker!r}"
    after = text.split(marker, 1)[1]
    row = after.split("\n", 1)[0]
    found = set(re.findall(r"`([a-z_]+)`", row))
    assert found, "the pins-table row lists no field names"
    return found


@pytest.mark.parametrize("family", fc.CERTIFIED_FAMILIES)
def test_the_record_carries_exactly_the_eight_fields(family: str):
    """T5: the certification block's key set equals `CERTIFICATION_FIELDS` —
    both directions, per certified family record."""
    path = fc.record_path(family)
    if not path.exists():
        pytest.fail(f"{path} is missing: capture it with {fc.FAMILIES[family]!r}")
    record = fc.load_record(family)
    assert len(fc.CERTIFICATION_FIELDS) == 8
    missing, extra = fc.check_certification_fields(record)
    assert not missing and not extra, (missing, extra)
    block = record["certification"]
    # each field carries something, in the shape the reader expects
    assert block["model_revision"]["key"] == fc.FAMILIES[family].key
    assert block["model_revision"]["resolved_revision"], "no hub snapshot resolved"
    assert block["causalab_revision"]["tree_digest"]
    assert block["attn_implementation"] == record["attn_implementation"] == "eager"
    assert block["dtype"] == fc.FAMILIES[family].dtype
    assert block["hook_names"] == sorted(block["hook_names"]) and block["hook_names"]
    assert set(block["tensor_shapes"]) == set(block["hook_names"])
    assert (
        block["max_activation_diff"]["per_hook"]
        and "max" in block["max_activation_diff"]
    )
    assert block["max_logit_diff"]["per_case"] and "max" in block["max_logit_diff"]


def test_the_docs_list_the_same_eight_fields():
    assert _fields_documented_in_tests_md() == set(fc.CERTIFICATION_FIELDS)


def test_dropping_a_field_fails_the_census(committed):
    """T5's mutation: a record without ``max_logit_diff`` is refused by the
    guard — and so is one carrying a ninth field."""
    mutated = copy.deepcopy(committed)
    del mutated["certification"]["max_logit_diff"]
    missing, extra = fc.check_certification_fields(mutated)
    assert missing == ["max_logit_diff"] and not extra
    widened = copy.deepcopy(committed)
    widened["certification"]["max_logits_diff"] = 0.0  # a second spelling
    missing, extra = fc.check_certification_fields(widened)
    assert not missing and extra == ["max_logits_diff"]


def test_the_frozen_goldens_carry_no_certification_block():
    """The three pre-migration captures carry no certification and are never
    regenerated; the guard must not read them as certified families."""
    for family in ("gpt2", "gqa", "llama"):
        record = fc.load_record(family)
        assert "certification" not in record, family
        assert family not in fc.CERTIFIED_FAMILIES


# --------------------------------------------------------------------------- #
# T6 — the one-ulp reassociation
# --------------------------------------------------------------------------- #


def _sequential_kv_mem(
    decayed_state: torch.Tensor, k_hat: torch.Tensor
) -> torch.Tensor:
    """The same sum in the textbook association — strictly left to right over
    d_k — instead of torch's blocked reduction. Same real number, different
    float: about one ulp at the readout's magnitude."""
    prod = decayed_state * k_hat.unsqueeze(-1)
    acc = prod[..., 0, :]
    for i in range(1, prod.shape[-2]):
        acc = acc + prod[..., i, :]
    return acc


def test_the_reassociation_is_a_genuine_ulp_level_change():
    """The mutation is real (not bit-identical to torch's reduction, the way a
    two-halves split turns out to be) and small (ulps, not a bug)."""
    gen = torch.Generator().manual_seed(0)
    state = torch.randn(1, 8, 32, 32, generator=gen)
    k_hat = torch.randn(1, 8, 32, generator=gen)
    reference = oracle_lib.delta_kv_mem(state, k_hat)
    mutated = _sequential_kv_mem(state, k_hat)
    diff = (reference - mutated).abs()
    assert float(diff.max()) > 0.0
    assert float(diff.max()) <= 8 * torch.finfo(torch.float32).eps * float(
        reference.abs().max()
    )


def test_a_one_ulp_reassociation_in_the_oracle_fails_the_certification(
    qwen35moe_bundle, capture, monkeypatch
):
    """T6. With ``delta_kv_mem`` reassociated the certification fails — and it
    fails **first on an intermediate component**: the memory readout itself,
    then the state update and the state it feeds, the kernel output and block
    output downstream of the state write. The logits move by ~4e-7 in one case
    only, inside every legacy logits band: a logits-only certification would
    have passed this injection."""
    assert not fc.certification_failures(capture, fc.BANDS["fp32"])  # the control
    monkeypatch.setattr(oracle_lib, "delta_kv_mem", _sequential_kv_mem)
    mutated = fc.capture_family(qwen35moe_bundle, FAMILY)
    failures = fc.certification_failures(mutated, fc.BANDS["fp32"])
    assert failures, "the reassociated oracle still certified — the band is not exact"

    first_name, first_diff = failures[0]
    print(f"T6 first failing component: {first_name} by {first_diff!r}")
    assert "[kv_mem]" in first_name, first_name  # an intermediate component
    assert first_diff > 0.0
    failing_hooks = {name.split("@", 1)[1] for name, _ in failures}
    interior = {h for h in failing_hooks if "[kv_mem]" in h or "[state" in h}
    assert interior, failing_hooks  # the recurrence interior is where it shows
    assert any(h != "lm_head" for h in failing_hooks)

    # the logits alone would not have caught it
    worst_logits = max(mutated.logit_diffs.values())
    assert worst_logits > 0.0  # the injection does reach the logits ...
    for band in LEGACY_LOGIT_BANDS:
        assert worst_logits < band, (
            worst_logits,
            band,
        )  # ... but under every legacy band
    # and the read cases' logits never move at all: reading the interior does
    # not change the model, so a read-only logits check is blind by construction
    assert mutated.logit_diffs[f"{FAMILY}.collect.lm_head.identity.all"] == 0.0


# --------------------------------------------------------------------------- #
# T7 — anti-vacuity
# --------------------------------------------------------------------------- #


def test_every_genuine_write_moves_the_logits_and_the_intermediate(capture):
    values = capture.record["values"]
    deltas = {k: v for k, v in values.items() if k.endswith(".clean_delta.max")}
    assert len(deltas) == len(fc.WRITES) + sum(len(w.downstream) for w in fc.WRITES)
    for key, moved in deltas.items():
        assert moved > 0.0, key
    for case in fc.WRITES:  # the logits, at more than float noise
        layer = capture.record["layers"][case.role]
        key = f"{FAMILY}.interchange.{case.component}.identity.{'last' if case.pos == -1 else f'step{case.pos}'}.L{layer}.clean_delta.max"
        assert values[key] > 1e-3, (key, values[key])


def test_a_no_op_swap_is_bit_exact_except_the_documented_path_forcing(capture):
    """Swapping a tensor for its own value leaves the logits bit-exact at every
    module boundary and kernel-argument slot. The state's self-swap substitutes
    the stepwise recurrence for the chunked kernel, and costs exactly the two
    kernels' association gap — non-zero, and bounded."""
    noop = capture.record["measurements"]["noop_swap_logit_drift"]
    bound = fc.BANDS["fp32"].state_substitution_drift
    seen_state = False
    for case_id, drift in noop.items():
        if ".interchange.delta_state." in case_id:
            seen_state = True
            assert 0.0 < drift <= bound, (case_id, drift)
        else:
            assert drift == 0.0, (case_id, drift)
    assert seen_state
    gap = capture.record["measurements"]["chunked_vs_stepwise_kernel_output"]
    assert 0.0 < gap <= 1e-9, gap  # the association gap the band refuses to absorb
    _assert_the_genuine_state_write_exceeds_the_drift(capture.record)


def _assert_the_genuine_state_write_exceeds_the_drift(record: dict) -> None:
    """IOI V2's lesson: a bit-exact (or bounded) no-op is necessary but
    insufficient — the genuine interchange at the same address must move the
    logits by more than the path-forcing noise the no-op costs."""
    layer = record["layers"]["delta"]
    family = record["family"]
    write_id = (
        f"{family}.interchange.delta_state.identity.step{fc.STATE_WRITE_STEP}.L{layer}"
    )
    genuine = record["values"][f"{write_id}.clean_delta.max"]
    drift = record["measurements"]["noop_swap_logit_drift"][write_id]
    assert genuine > drift > 0.0, (genuine, drift)
