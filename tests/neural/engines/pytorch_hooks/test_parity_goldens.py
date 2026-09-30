"""Replay the frozen pre-migration parity goldens through the protocol stack.

``tests/neural/parity/goldens/{llama,gpt2,gqa}.json`` pin the numerical
output of the canonical per-mode parity cases, captured from the raw-hook
oracle on fresh seeded tiny-random models (eager attention forced). They are
the pyvene-era numerical anchor; this test re-drives every portable pinned
case through the NEW stack — an intervention specification executed by
``causalab.neural.engines.pytorch_hooks`` — and asserts the pinned values
verbatim
(default tolerance 1e-4, per-key overrides honored, shapes exact).

Old mode → protocol ``do`` mapping (docs/intervention_protocol.md §2.5, §2.8):

* ``collect``      → a plain saved read (the featurized value itself);
* ``replace``      → ``{"swap": <params tensor>}`` (constant vector);
* ``steer``        → ``{"add_scaled": {"op": <params tensor>, "alpha": 2.5}}``;
* ``interchange``  → ``{"swap": "<read of the donor site>"}``;
* ``interpolate``  → ``{"lerp": {"op": "<donor read>", "alpha": 0.3}}``;
* ``noise``        → ``{"gaussian": {"seed": 7, "scale": 3.0, "axis":
  "tp_duplicated"}}`` (the draw contract matches the oracle: seeded CPU
  ``randn((batch, n_pos, feature_width))`` made outside the model);
* ``mask``         → the ``["rot", "gate"]`` composition in hard eval mode +
  ``swap`` from the donor read through the same chain (the gate's
  ``err = (1−mask)⊙x`` keeps the base's off-features — exactly the old
  MaskGate hard swap);
* ``sub3``         → the frozen rotation: ``torch.manual_seed(0)`` →
  ``orthogonal_``-initialized ``(d, 3)`` weight → the torch orthogonal
  parametrization round trip, fed to the stack as a loaded ``subspace``
  featurizer (``LoadedLinear`` math is the old ``SubspaceFeaturizerModule``
  math: ``f = xQ``, ``err = x − fQᵀ``).

The models are rebuilt exactly as the capture built them: **fresh seeded
from the tiny-random config** (``torch.manual_seed(0)`` before
construction, eager attention, left padding) — NOT ``load_model``'s
pretrained checkpoint weights — and wrapped in a ``ModelBundle`` by hand.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.shared.featurizers import Gate
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.devices import DeviceMap
from causalab.protocol.registry import model_info_from_hf_config

from tests._helpers.tiny import fresh_tiny_random_gpt2, fresh_tiny_random_llama
from tests.neural.engines.pytorch_hooks._drive import (
    base_data_section,
    bundle_loader,
    executor_for,
)
from tests.protocol._docs import saved

pytestmark = pytest.mark.numerical_unit

GOLDENS_DIR = Path(__file__).resolve().parents[2] / "parity" / "goldens"
FAMILIES = ("llama", "gpt2", "gqa")

# --- the frozen capture recipe (tests/neural/parity/cases.py at capture) ---- #
BASE_TEXT = "the quick brown fox jumps"
DST_LAYER, DONOR_LAYER = 1, 0
SUB3_K = 3
NOISE_SEED, NOISE_SCALE = 7, 3.0
STEER_FACTOR = 2.5
ALPHA = 0.3
HEAD_DST, HEAD_DONOR = 1, 0
#: linspace spans of the constant operands, per mode.
CONSTANT_SPANS = {"replace": (-30.0, 30.0), "steer": (3.0, 9.0)}
N_PROBES = 8

#: Pinned case ids that cannot be expressed in the protocol vocabulary —
#: listed explicitly so nothing is silently dropped (a coverage test below
#: cross-checks this table against the golden files).
#:
#: ✅ **Empty.** The two entries that lived here named the pyvene ``value``
#: kind — a KV-head slice of ``v_proj``'s output — and were skipped because the
#: §2.4 vocabulary had no such site: ``attention_premix`` is the o-projection's
#: *input*, in query-head space, which is a different tensor in a different
#: space. ``attention_value_states`` is that site, and both cases now replay.
SKIPPED: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# golden files
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def _golden(family: str) -> dict[str, Any]:
    with (GOLDENS_DIR / f"{family}.json").open() as f:
        return json.load(f)


def _case_of(key: str) -> str:
    """The case id of one pinned key (``<case_id>.out.*`` / ``.probe.*`` /
    ``.clean_delta.*``)."""
    for marker in (".out.", ".probe.", ".clean_delta."):
        if marker in key:
            return key.split(marker, 1)[0]
    raise AssertionError(f"unrecognized golden key {key!r}")


def _case_ids(family: str) -> list[str]:
    seen: dict[str, None] = {}
    for key in _golden(family)["values"]:
        seen.setdefault(_case_of(key))
    return list(seen)


PORTABLE_CASES = [
    case_id
    for family in FAMILIES
    for case_id in _case_ids(family)
    if case_id not in SKIPPED
]


# --------------------------------------------------------------------------- #
# frozen ingredients — models, rotation, gate
# --------------------------------------------------------------------------- #


def _force_eager(cfg: Any) -> None:
    """Pins must replay under eager attention (the capture-wide policy)."""
    cfg._attn_implementation = "eager"


def _gqa_then_eager(cfg: Any) -> None:
    assert cfg.num_attention_heads % 2 == 0
    cfg.num_key_value_heads = cfg.num_attention_heads // 2
    _force_eager(cfg)


@functools.lru_cache(maxsize=None)
def _bundle(family: str) -> ModelBundle:
    """The capture's model recipe, wrapped for the reference engine: fresh
    seeded from config (NOT from_pretrained — ``load_model`` would load the
    checkpoint's weights, a different model), eager, left-padded."""
    factory, mutate = {
        "llama": (fresh_tiny_random_llama, _force_eager),
        "gpt2": (fresh_tiny_random_gpt2, _force_eager),
        "gqa": (fresh_tiny_random_llama, _gqa_then_eager),
    }[family]
    raw, tok = factory(mutate_config=mutate)
    assert raw.config._attn_implementation == "eager"
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw.eval()
    raw.requires_grad_(False)
    key = f"tiny-parity-{family}"
    info = model_info_from_hf_config(key, raw.config)
    return ModelBundle(
        key=key,
        revision="main",
        model=raw,
        tokenizer=tok,
        info=info,
        devices=DeviceMap.parse("cpu", info.num_layers),
        dtype="fp32",
    )


def _golden_subspace_q(width: int, k: int) -> torch.Tensor:
    """The exact ``(width, k)`` orthonormal map the goldens' sub3 featurizer
    used — the deleted ``SubspaceFeaturizer(shape=(width, k))`` recipe,
    replicated verbatim: global seed 0, pyvene's ``LowRankRotateLayer`` init
    (``torch.nn.init.orthogonal_``), then the torch orthogonal-parametrization
    round trip (householder for rectangular shapes) that the old stack read
    its ``.weight`` through."""

    class _Rotate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(width, k))
            torch.nn.init.orthogonal_(self.weight)

    torch.manual_seed(0)
    layer = torch.nn.utils.parametrizations.orthogonal(_Rotate())
    return layer.weight.detach().clone()


def _golden_gate(width: int) -> Gate:
    """The old deterministic MaskGate in hard eval mode: on where
    ``sigmoid(linspace(-2, 2, width)) > 0.5`` — identical to the new Gate's
    hard-eval split ``theta > 0`` for the same logits (0 is off both ways)."""
    gate = Gate(width)
    with torch.no_grad():
        gate.theta.copy_(torch.linspace(-2.0, 2.0, width))
    return gate.eval()


# --------------------------------------------------------------------------- #
# case realization — one intervention specification per pinned case
# --------------------------------------------------------------------------- #


def _save(value: str, model: str) -> dict[str, str]:
    return saved(value, model, f"{value}.safetensors")


def _doc_skeleton() -> dict[str, Any]:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {},
            "sites": {},
            "reads": {},
            "save": [],
        },
    }


def _read(site: str, featurizer: Any = None) -> dict[str, Any]:
    read: dict[str, Any] = {
        "site": site,
        "pos": {"index": -1},  # the capture's "last" position (batch of one)
    }
    if featurizer is not None:
        read["featurizer"] = featurizer
    return read


def _models(
    original: list[str], patched: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Every capture ran on one input: the un-intervened network (``original``)
    reading ``original``, and — for the write modes — ``patched`` landing the
    one write ``e`` and reading ``patched`` (§2.9)."""
    models: dict[str, dict[str, Any]] = {
        "original": {"input": "base", "reads": list(original)}
    }
    if patched is not None:
        models["patched"] = {"input": "base", "reads": list(patched), "writes": ["e"]}
    return models


def _run_sub3_case(
    bundle: ModelBundle, mode: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One ``<family>.<mode>.block_output.sub3.last`` case: destination
    block_output L1, donor (where a mode reads one) block_output L0 of the
    SAME input, everything through the frozen k=3 rotation."""
    q = _golden_subspace_q(bundle.info.hidden_size, SUB3_K)
    tensors: dict[str, dict[str, torch.Tensor]] = {"rot.safetensors": {"weight": q}}
    doc = _doc_skeleton()
    doc["method"]["sites"]["dst"] = {"component": "block_output", "layers": [DST_LAYER]}
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "file_path": "rot.safetensors"}
    }

    if mode == "collect":
        doc["method"]["reads"]["out"] = _read("dst", featurizer="rot")
        doc["method"]["intervened_models"] = _models(["out"])
        doc["method"]["save"] = [_save("out", "original")]
        executor = executor_for(
            doc, bundle, base_texts=[BASE_TEXT], load_tensors=bundle_loader(tensors)
        )
        return executor.read_value("out"), None

    doc["method"]["sites"]["lm_head"] = {"component": "lm_head"}
    doc["method"]["reads"]["out"] = _read("lm_head")
    doc["method"]["reads"]["clean"] = _read("lm_head")
    doc["method"]["intervened_models"] = _models(["clean"], ["out"])
    doc["method"]["save"] = [_save("out", "patched"), _save("clean", "original")]
    write: dict[str, Any] = {"site": "dst", "pos": {"index": -1}, "featurizer": "rot"}

    if mode in ("replace", "steer"):
        lo, hi = CONSTANT_SPANS[mode]
        tensors["vec.safetensors"] = {"value": torch.linspace(lo, hi, SUB3_K)}
        doc["method"]["params"] = {"vec": {"file_path": "vec.safetensors"}}
        write["do"] = (
            {"swap": "vec"}
            if mode == "replace"
            else {"add_scaled": {"op": "vec", "alpha": STEER_FACTOR}}
        )
    elif mode == "noise":
        write["do"] = {
            "gaussian": {
                "seed": NOISE_SEED,
                "scale": NOISE_SCALE,
                "axis": "tp_duplicated",
            }
        }
    else:  # interchange / interpolate / mask — donor-reading modes
        doc["method"]["sites"]["donor"] = {
            "component": "block_output",
            "layers": [DONOR_LAYER],
        }
        chain: Any = ["rot", "gate"] if mode == "mask" else "rot"
        doc["method"]["reads"]["v_cf"] = _read("donor", featurizer=chain)
        doc["method"]["intervened_models"]["original"]["reads"].append("v_cf")
        if mode == "mask":
            doc["method"]["featurizers"]["gate"] = {"kind": "gate"}
            write["featurizer"] = ["rot", "gate"]
        write["do"] = (
            {"lerp": {"op": "v_cf", "alpha": ALPHA}}
            if mode == "interpolate"
            else {"swap": "v_cf"}
        )

    doc["method"]["writes"] = {"e": write}
    executor = executor_for(
        doc, bundle, base_texts=[BASE_TEXT], load_tensors=bundle_loader(tensors)
    )
    if mode == "mask":
        # The gate stage operates in the rotation's k=3 feature space, so it
        # is injected pre-built through the executor's stage seam (build_stack
        # would size a declared gate to the site width, not the chain width).
        executor.stage_cache["gate"] = _golden_gate(SUB3_K)
    out = executor.read_value("out")[:, 0, :]  # the capture pinned (1, vocab)
    clean = executor.read_value("clean")[:, 0, :]
    return out, clean


#: The two pyvene head kinds, and the §2.4 component each one is.
#:
#: 📐 They are **different tensors in different head spaces**, which is why the
#: second could not be replayed until ``attention_value_states`` existed, and
#: why reusing one name for
#: both would have been wrong: ``head_attention_value`` is the o-projection's
#: input, ``num_heads`` wide per head; ``head_value`` is ``v_proj``'s output,
#: ``num_key_value_heads`` wide. On this GQA fixture (H 4, H_kv 2) that is a 2x
#: difference in how many heads exist — and the pinned shape ``[1, 1, 4]``
#: is one KV head's ``head_dim`` slice.
HEAD_CASE_COMPONENTS = {
    "head_attention_value": "attention_premix",
    "head_value": "attention_value_states",
}


def _run_head_case(
    bundle: ModelBundle, mode: str, component: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One ``gqa.<mode>.<kind>.identity.last.head`` case: the per-head column
    slice, destination head 1, donor head 0 at the same layer/input — the old
    HeadSite semantics that a component plus ``head`` carries in the protocol
    vocabulary."""
    doc = _doc_skeleton()
    doc["method"]["sites"]["dst"] = {
        "component": component,
        "layers": [DST_LAYER],
        "head": HEAD_DST,
    }
    if mode == "collect":
        doc["method"]["reads"]["out"] = _read("dst")
        doc["method"]["intervened_models"] = _models(["out"])
        doc["method"]["save"] = [_save("out", "original")]
        executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
        return executor.read_value("out"), None

    assert mode == "interchange", mode
    doc["method"]["sites"]["donor"] = {
        "component": component,
        "layers": [DST_LAYER],
        "head": HEAD_DONOR,
    }
    doc["method"]["sites"]["lm_head"] = {"component": "lm_head"}
    doc["method"]["reads"]["v_cf"] = _read("donor")
    doc["method"]["reads"]["out"] = _read("lm_head")
    doc["method"]["reads"]["clean"] = _read("lm_head")
    doc["method"]["intervened_models"] = _models(["v_cf", "clean"], ["out"])
    doc["method"]["writes"] = {
        "e": {"site": "dst", "pos": {"index": -1}, "do": {"swap": "v_cf"}}
    }
    doc["method"]["save"] = [_save("out", "patched"), _save("clean", "original")]
    executor = executor_for(doc, bundle, base_texts=[BASE_TEXT])
    out = executor.read_value("out")[:, 0, :]
    clean = executor.read_value("clean")[:, 0, :]
    return out, clean


def _run_case(case_id: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    family, mode, component, featurizer, positions, *path = case_id.split(".")
    bundle = _bundle(family)
    assert positions == "last", case_id  # the golden subset is last-position only
    if path == ["head"]:
        assert featurizer == "identity", case_id
        assert component in HEAD_CASE_COMPONENTS, case_id
        return _run_head_case(bundle, mode, HEAD_CASE_COMPONENTS[component])
    assert not path and (component, featurizer) == ("block_output", "sub3"), case_id
    return _run_sub3_case(bundle, mode)


# --------------------------------------------------------------------------- #
# the pin recipe (tests/neural/parity/pins.py at capture, kept verbatim)
# --------------------------------------------------------------------------- #


def _pin_values(
    case_id: str, value: torch.Tensor, clean: torch.Tensor | None
) -> dict[str, Any]:
    """shape + mean/std/first/last, ``N_PROBES`` seeded probe elements, and —
    for write modes — the non-vacuity pin ``clean_delta.max``."""
    values: dict[str, Any] = {}
    t = value.detach().float()
    values[f"{case_id}.out.shape"] = list(t.shape)
    flat = t.flatten()
    values[f"{case_id}.out.mean"] = float(t.mean())
    values[f"{case_id}.out.first"] = float(flat[0])
    values[f"{case_id}.out.last"] = float(flat[-1])
    if t.numel() >= 2:
        values[f"{case_id}.out.std"] = float(t.std())
    gen = torch.Generator().manual_seed(0)
    for j, i in enumerate(
        torch.randperm(flat.numel(), generator=gen)[:N_PROBES].tolist()
    ):
        values[f"{case_id}.probe.{j}"] = float(flat[i])
    if clean is not None:
        delta = (value.detach() - clean.detach()).abs().max()
        values[f"{case_id}.clean_delta.max"] = float(delta)
    return values


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


def test_every_pinned_case_is_ported_or_skipped() -> None:
    """The SKIPPED table names real pinned cases, and together with the
    portable set covers every case id in the golden files — a case can be
    skipped only out loud."""
    all_ids = {cid for family in FAMILIES for cid in _case_ids(family)}
    assert set(SKIPPED) <= all_ids, sorted(set(SKIPPED) - all_ids)
    assert not set(PORTABLE_CASES) & set(SKIPPED)
    assert set(PORTABLE_CASES) | set(SKIPPED) == all_ids


@pytest.mark.parametrize("case_id", PORTABLE_CASES)
def test_protocol_stack_replays_captured_golden(case_id: str) -> None:
    family = case_id.split(".", 1)[0]
    golden = _golden(family)
    assert golden["attn_implementation"] == "eager"
    tolerance = dict(golden.get("tolerance", {"default": 1e-4}))

    value, clean = _run_case(case_id)
    got = _pin_values(case_id, value, clean)
    want = {k: v for k, v in golden["values"].items() if _case_of(k) == case_id}
    assert set(got) == set(want), (
        f"pinned keys diverge for {case_id} "
        f"(missing from golden: {sorted(set(got) - set(want))[:5]}, "
        f"unproduced: {sorted(set(want) - set(got))[:5]})"
    )

    mismatches = []
    for key, expected in sorted(want.items()):
        have = got[key]
        if isinstance(expected, list):  # shapes — exact
            if list(have) != expected:
                mismatches.append(f"{key}: shape {have} != pinned {expected}")
        else:
            tol = float(tolerance.get(key, tolerance.get("default", 1e-4)))
            if not abs(float(have) - float(expected)) <= tol:
                mismatches.append(f"{key}: {have!r} != pinned {expected!r} (tol {tol})")
    assert not mismatches, (
        f"{len(mismatches)} pinned value(s) drifted:\n  " + "\n  ".join(mismatches)
    )
