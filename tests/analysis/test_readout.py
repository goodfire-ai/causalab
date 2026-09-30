"""The model-family readout adapter.

**T1 — the accepted parity fixture, without architecture branches.** For each
of the three tiny families (``gpt2``: LayerNorm, tied head; ``llama``:
RMSNorm ``weight``; ``qwen3_5_moe_text``: RMSNorm ``1 + weight``, hybrid tower)
the adapter's ``logits(block_output@last)`` is the engine's ``lm_head`` read
**bit for bit**, and the residual accounting of the Hydra effect paper (McGrath et al. 2023,
arXiv:2307.15771) rebuilt through the
adapter (``tests/_helpers/readout_accounting.py``) closes at the reference
projection's noise floor with both declared rounding terms and breaks, by
exactly the dropped term, without either. No ``getattr`` on a norm name, no
gain fit, no ``.weight`` read anywhere in this file or the helper.

**T2 — mutations, each with its passing twin.** (a) a bundle whose registry
entry names a family with no declaration is refused **naming the family**;
(b) an accumulation dtype outside ``{fp32, fp64}`` is refused by name at
construction; (c) the qwen family declared with the *other* gain convention
fails ``certify()`` on the tiny qwen fixture naming both conventions and the
measured gap; (d) the same on tiny llama with ``one_plus_weight``.

**T3 — nothing in the hashed closures reaches this module**, so no pinned
digest moves (the byte-identity of the pins themselves is the digest job).

📐 Measured 2026-09-03 on the fixtures (fp32, CPU; ``README`` of the numbers
the declarations rest on)::

    family            norm module         eps attr           gain: gap(weight)  gap(1+weight)  |ln_final|
    gpt2              LayerNorm           eps                4.254e-07          3.467e+00      3.467
    llama             LlamaRMSNorm        variance_epsilon   3.142e-07          3.069e+00      3.069
    qwen3_5_moe_text  Qwen3_5MoeRMSNorm   eps                2.241e+00          2.080e-07      2.411

The qwen fixture's epsilon lives at ``eps`` (the census inherited
``variance_epsilon`` from Llama; measured otherwise here), and its gain is
``one_plus_weight`` — the same class the real Qwen3.6-35B-A3B runs.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.analysis import logit_lens as readout_module
from causalab.neural.shared.devices import DeviceMap
from causalab.analysis.logit_lens import (
    ACCUMULATION_DTYPES,
    CERTIFICATION_ULPS,
    GAINS,
    NORMS,
    READOUT_SPECS,
    GainMismatch,
    Readout,
    ReadoutSpec,
    readout_spec,
    register_readout,
    unit_roundoff,
)
from causalab.protocol.identity import import_closure

from tests._helpers import readout_accounting as accounting
from tests._helpers import synthetic_family as synth
from tests.neural.engines.pytorch_hooks.conftest import (
    TINY_GPT2,
    TINY_LLAMA,
    TINY_QWEN35_MOE,
)
from tests.workflow.test_closure_census import CLOSURES, REDUCE, SHARED

REPO = Path(__file__).resolve().parents[2]
READOUT_PATH = "causalab/analysis/logit_lens.py"
#: The module's name as an import spells it — derived, so the closure guard
#: below cannot go stale on the next rename.
READOUT_MODULE = Path(READOUT_PATH).stem

FIXTURES: dict[str, str] = {
    "gpt2": TINY_GPT2,
    "llama": TINY_LLAMA,
    "qwen35moe": TINY_QWEN35_MOE,
}

#: The measured declarations (module docstring) the literals must equal.
EXPECTED: dict[str, tuple[str, str, str]] = {
    "gpt2": ("layernorm", "weight", "eps"),
    "llama": ("rmsnorm", "weight", "variance_epsilon"),
    "qwen3_5_moe_text": ("rmsnorm", "one_plus_weight", "eps"),
}


@pytest.fixture(scope="module", params=sorted(FIXTURES))
def bundle(request: pytest.FixtureRequest) -> ModelBundle:
    return load_model(FIXTURES[request.param])


@pytest.fixture(scope="module")
def acc(bundle: ModelBundle) -> accounting.Accounting:
    return accounting.account(bundle)


def _residual(bundle: ModelBundle) -> torch.Tensor:
    return accounting.read(bundle, "block_output", len(bundle.blocks) - 1)


# --------------------------------------------------------------------------- #
# T1 — the parity fixture through the adapter
# --------------------------------------------------------------------------- #


@pytest.mark.numerical_unit
def test_the_adapters_readout_is_the_engines_lm_head_read_bit_for_bit(
    bundle: ModelBundle,
):
    x = _residual(bundle)
    logits = accounting.read(bundle, "lm_head", None)
    with torch.no_grad():
        mine = Readout.from_bundle(bundle).logits(x)
    assert mine.dtype == logits.dtype and mine.shape == logits.shape
    assert torch.equal(mine, logits)


@pytest.mark.numerical_unit
def test_the_residual_accounting_closes_through_the_adapter(acc: accounting.Accounting):
    assert accounting.problems(acc) == [], acc.as_dict()


@pytest.mark.numerical_unit
def test_dropping_either_rounding_term_breaks_the_closure_by_that_term(
    acc: accounting.Accounting,
):
    """The mutation, stated on its own: with both terms the
    closure is at the floor; without either it is that term's size."""
    floor = max(acc.projection_noise_floor, 1e-12)
    assert acc.closure <= 10 * floor
    assert acc.closure_without_normalization_term > 100 * floor
    assert acc.closure_without_lm_head_term > 100 * floor
    assert acc.closure_without_normalization_term == pytest.approx(
        acc.projected_normalization_rounding, abs=10 * floor
    )
    assert acc.closure_without_lm_head_term == pytest.approx(
        acc.lm_head_rounding, abs=10 * floor
    )


@pytest.mark.numerical_unit
def test_the_declared_gain_certifies_and_the_other_is_far(bundle: ModelBundle):
    """Anti-vacuity of the declaration: the declared convention lands within
    the band, the other one is |ln_final|-sized — a thousand bands away."""
    readout = Readout.from_bundle(bundle)
    cert = readout.certify(_residual(bundle))
    assert cert.gain == readout.spec.gain
    assert cert.gap <= cert.tolerance
    (other,) = [g for g in GAINS if g != cert.gain]
    assert cert.gaps[other] >= 1.0
    assert cert.gaps[other] > 1000 * cert.tolerance
    assert cert.dtype == "float32"


@pytest.mark.numerical_unit
def test_the_literals_are_the_measured_declarations(bundle: ModelBundle):
    readout = Readout.from_bundle(bundle)
    norm, gain, eps_attr = EXPECTED[bundle.info.family]
    assert (readout.spec.norm, readout.spec.gain, readout.spec.eps_attr) == (
        norm,
        gain,
        eps_attr,
    )
    assert readout.spec.accumulation_dtype == "fp64"
    assert hasattr(readout.norm, eps_attr)
    assert 0.0 < readout.eps < 1e-3
    assert readout.family == bundle.info.family
    assert readout.family != bundle.adapter.family  # keyed by ModelInfo.family


@pytest.mark.numerical_unit
def test_the_accumulation_dtype_is_applied_and_cast_back(bundle: ModelBundle):
    """``unembed`` runs the head's own forward in the declared dtype: in fp64
    it differs from the fp32 module call by fp32 roundoff only, returns the
    input's dtype, and an fp32 accumulation is the module call itself."""
    readout = Readout.from_bundle(bundle)
    with torch.no_grad():
        z = readout.normalize(_residual(bundle))
        as_run = readout.head(z)
        reference = readout.unembed(z)
        assert reference.dtype == z.dtype == torch.float32
        gap = float((reference.double() - as_run.double()).abs().max())
        assert gap <= CERTIFICATION_ULPS * unit_roundoff(torch.float32) * float(
            as_run.abs().max()
        )
        assert torch.equal(readout.at("fp32").unembed(z), as_run)
        z64 = z.double()
        assert readout.unembed(z64).dtype == torch.float64
        assert torch.allclose(readout.unembed(z64), as_run.double(), atol=1e-6, rtol=0)


@pytest.mark.numerical_unit
def test_centering_is_a_uniform_shift_every_softmax_metric_ignores(bundle: ModelBundle):
    readout = Readout.from_bundle(bundle)
    with torch.no_grad():
        logits = readout.logits(_residual(bundle))
        centered = readout.center(logits)
    assert centered.shape == logits.shape
    assert torch.allclose(centered.mean(-1), torch.zeros(logits.shape[:-1]), atol=1e-5)
    assert not torch.equal(centered, logits)
    assert torch.allclose(
        torch.log_softmax(centered, -1), torch.log_softmax(logits, -1), atol=1e-5
    )
    assert torch.equal(centered.argmax(-1), logits.argmax(-1))


# --------------------------------------------------------------------------- #
# T2 — mutations, with twins
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def synthetic_bundle() -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return ModelBundle(
        key=synth.KEY,
        revision="main",
        model=synth.build_model(),
        tokenizer=tokenizer,
        info=synth.INFO,
        devices=DeviceMap.parse("cpu", synth.INFO.num_layers),
        dtype="fp32",
    )


def _with_family(bundle: ModelBundle, family: str | None) -> ModelBundle:
    return dataclasses.replace(
        bundle, info=dataclasses.replace(bundle.info, family=family)
    )


@pytest.fixture()
def restore_specs():
    before = dict(READOUT_SPECS)
    yield
    readout_module._READOUT_SPECS.clear()  # pyright: ignore[reportPrivateUsage]
    readout_module._READOUT_SPECS.update(before)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.unit
def test_a_family_without_a_declaration_is_refused_naming_the_family(
    synthetic_bundle: ModelBundle,
):
    family = "synthetic_readout_family"
    assert family not in READOUT_SPECS
    with pytest.raises(ValueError, match=family) as info:
        Readout.from_bundle(_with_family(synthetic_bundle, family))
    message = str(info.value)
    assert "register_readout" in message and synth.KEY in message
    assert not isinstance(info.value, KeyError)


@pytest.mark.unit
def test_an_entry_recording_no_family_is_refused_naming_the_model(
    synthetic_bundle: ModelBundle,
):
    assert synth.INFO.family is None
    with pytest.raises(ValueError, match=re.escape(synth.KEY)) as info:
        Readout.from_bundle(synthetic_bundle)
    assert "ModelInfo.family" in str(info.value)
    with pytest.raises(ValueError, match="records no family"):
        readout_spec(None, key=synth.KEY)


@pytest.mark.unit
def test_a_third_party_family_declares_its_readout_and_certifies(
    synthetic_bundle: ModelBundle, restore_specs: None
):
    """The twin of (a): the registration precedent, from any module — and the
    declared readout is then the engine's read bit for bit on the new tree."""
    family = "synthetic_readout_family"
    register_readout(
        family,
        ReadoutSpec(
            norm="layernorm", gain="weight", eps_attr="eps", accumulation_dtype="fp64"
        ),
    )
    bundle = _with_family(synthetic_bundle, family)
    readout = Readout.from_bundle(bundle)
    x = accounting.read(bundle, "block_output", synth.LAYERS - 1)
    with torch.no_grad():
        assert torch.equal(readout.logits(x), accounting.read(bundle, "lm_head", None))
    cert = readout.certify(x)
    assert cert.gap <= cert.tolerance
    assert accounting.problems(accounting.account(bundle)) == []


@pytest.mark.unit
def test_a_declared_epsilon_attribute_the_module_lacks_is_refused_by_name(
    bundle: ModelBundle, restore_specs: None
):
    spec = READOUT_SPECS[bundle.info.family]
    register_readout(
        bundle.info.family, dataclasses.replace(spec, eps_attr="epsilon_x")
    )
    with pytest.raises(ValueError, match="epsilon_x") as info:
        Readout.from_bundle(bundle)
    assert spec.eps_attr in str(info.value)  # the module's real attribute is named
    register_readout(bundle.info.family, spec)
    assert Readout.from_bundle(bundle).eps > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "float64", "double", ""])
def test_an_undeclared_accumulation_dtype_is_refused_at_construction(dtype: str):
    with pytest.raises(ValueError, match=re.escape(repr(dtype))) as info:
        ReadoutSpec(
            norm="rmsnorm",
            gain="weight",
            eps_attr="eps",
            accumulation_dtype=dtype,  # pyright: ignore[reportArgumentType]
        )
    assert "['fp32', 'fp64']" in str(info.value)


@pytest.mark.unit
def test_the_declared_accumulation_dtypes_construct_and_apply(bundle: ModelBundle):
    for dtype in ACCUMULATION_DTYPES:
        spec = ReadoutSpec(
            norm="rmsnorm",
            gain="weight",
            eps_attr="eps",
            accumulation_dtype=dtype,
        )
        assert spec.accumulation_dtype == dtype
    readout = Readout.from_bundle(bundle)
    assert readout.at("fp32").spec.accumulation_dtype == "fp32"
    with pytest.raises(ValueError, match="'bf16'"):
        readout.at("bf16")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("norm", "gain"),
    [("batchnorm", "weight"), ("rmsnorm", "scale"), ("RMSNorm", "weight")],
)
def test_the_other_closed_fields_are_refused_by_name(norm: str, gain: str):
    with pytest.raises(
        ValueError, match=re.escape(repr(norm if norm not in NORMS else gain))
    ):
        ReadoutSpec(
            norm=norm,  # pyright: ignore[reportArgumentType]
            gain=gain,  # pyright: ignore[reportArgumentType]
            eps_attr="eps",
            accumulation_dtype="fp64",
        )


def _flip(gain: str) -> str:
    (other,) = [g for g in GAINS if g != gain]
    return other


@pytest.mark.numerical_unit
@pytest.mark.parametrize("fixture", ["qwen35moe", "llama", "gpt2"])
def test_the_wrong_gain_convention_is_refused_by_the_modules_forward(
    fixture: str, restore_specs: None
):
    """(c) qwen declared ``weight``, (d) llama (and gpt2) declared
    ``one_plus_weight``: ``certify`` names both conventions and the measured
    gap, which is |ln_final|-sized — a thousand bands, not a rounding."""
    bundle = load_model(FIXTURES[fixture])
    family = bundle.info.family
    right = READOUT_SPECS[family]
    wrong = _flip(right.gain)
    register_readout(family, dataclasses.replace(right, gain=wrong))
    readout = Readout.from_bundle(bundle)
    assert readout.spec.gain == wrong
    with pytest.raises(GainMismatch) as info:
        readout.certify(_residual(bundle))
    message = str(info.value)
    assert repr(wrong) in message and repr(right.gain) in message
    assert family in message and "register_readout" in message
    cert = info.value.certificate
    assert cert.gain == wrong
    assert cert.gap >= 1.0, cert
    assert cert.gap > 1000 * cert.tolerance, cert
    assert cert.gaps[right.gain] <= cert.tolerance, cert
    assert f"{cert.gap:.3e}" in message
    # the twin: the right declaration certifies on the same tensor
    register_readout(family, right)
    assert Readout.from_bundle(bundle).certify(_residual(bundle)).gap <= cert.tolerance


# --------------------------------------------------------------------------- #
# T3 — outside every hashed closure; the vocabularies censused
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_nothing_in_a_hashed_closure_reaches_the_readout():
    """A module a hashed script's closure imports is digest-bearing bytes; the
    readout must never become one — not even through a function-local import
    in a ``SHARED`` member, which the closure walk counts."""
    for module in (*CLOSURES, REDUCE):
        members = import_closure(REPO / module, root=REPO)
        assert READOUT_PATH not in members, module
    pattern = re.compile(
        rf"analysis\.{READOUT_MODULE}\b|analysis import [^\n]*\b{READOUT_MODULE}\b"
    )
    for path in SHARED:
        assert not pattern.search((REPO / path).read_text()), path


@pytest.mark.unit
def test_the_closed_vocabularies_are_the_tuples_the_literals_declare():
    assert NORMS == ("rmsnorm", "layernorm")
    assert GAINS == ("weight", "one_plus_weight")
    assert ACCUMULATION_DTYPES == ("fp32", "fp64")
    for family, spec in READOUT_SPECS.items():
        assert family.isidentifier()
        assert spec.norm in NORMS and spec.gain in GAINS
        assert spec.accumulation_dtype in ACCUMULATION_DTYPES
    assert set(EXPECTED) <= set(READOUT_SPECS)
    source = (REPO / READOUT_PATH).read_text()
    assert ".weight[" not in source and "lm_head.weight" not in source
    assert "wte" not in source


@pytest.mark.unit
def test_an_uncertified_dtype_is_refused_not_interpolated():
    with pytest.raises(ValueError, match="torch.int64"):
        unit_roundoff(torch.int64)
    assert unit_roundoff(torch.float32) == 2.0**-24
    assert unit_roundoff(torch.bfloat16) == 2.0**-8


@pytest.mark.unit
def test_the_readout_is_a_frozen_declaration(bundle: ModelBundle):
    readout = Readout.from_bundle(bundle)
    with pytest.raises(dataclasses.FrozenInstanceError):
        readout.family = "x"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(dataclasses.FrozenInstanceError):
        readout.spec.gain = "weight"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        READOUT_SPECS["x"] = readout.spec  # pyright: ignore[reportIndexIssue]
