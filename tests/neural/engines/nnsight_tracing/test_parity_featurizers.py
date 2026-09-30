"""Applied (eval-mode) featurizers through both engines: the same loaded
stage, the same document, the same answer.

Every featurizer kind a document can *load* — ``subspace``, ``pca``,
``standardize``, ``sae``, ``gate`` in each of its maps and readouts, the
grouped and position-axis gates, and a ``["rot", "gate"]`` chain under the
``sigmoid`` and the ``boundary`` map — is built from saved tensors
(``tests._helpers.engines.bundle_loader``) and driven through the
reference executor and the nnsight one. For each, three things must agree:
the featurized read ``f``, the site value after a feature-space ``swap``, and
the patched logits; and every write is checked non-vacuous on each engine
(patched ≠ clean on *that* engine), so agreement cannot be "neither landed".

The stages are shared code (``causalab/neural/shared/featurizers/``) built
by the shared ``ExecutorBase``; what differs per engine is how the captured
tensor reaches the stage and how the inverse lands back in the forward. That
seam is what this module pins. Nothing here trains.

Every document is spelled at protocol 4. One read is listed by several
models (``f`` on ``original`` and ``original_counterfactual``, ``x`` and
``logits`` on ``original`` and ``patched``), so every value is addressed by
its ``ReadRef`` (``VALUES``).

Cases covered (ids in the test docstrings): G13 subspace, G14 pca, G15
standardize, G16 sae, G17 gate maps and readouts (``boundary`` included), G18
grouped gates and the position axis, G19 chain, G20 top_k readout + ``rank``
(engine seam), G21 entry-identity refusal, G22 ``featurizer_cache``, G23
stage device placement. G9 (the error term under a loaded subspace swap) is a
case of ``test_parity_mechanisms.py``; G13 here checks the same complement
from the featurizer side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from safetensors.torch import save_file

from causalab.io.tensor_files import TensorBundle
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.featurizers import featurizer_cache
from causalab.protocol.identity import build_artifact_identity
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION, ReadRef

from tests._helpers import a3b_sweep as sweep
from tests._helpers import engines
from tests._helpers.paths import PROTOCOLS_DIR
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    BASE_TEXTS,
    CF_TEXTS,
)
from tests.protocol._docs import saved
from tests.protocol._env import fixture_input_overrides

pytestmark = pytest.mark.smoke

#: the feature width of every rank-k map here (the goldens' ``sub3``)
K = 3
#: the dictionary size of the toy SAE
SAE_WIDTH = 16
#: llama's two layers: the site under test and the layer above it
LAYER = 1

#: The model the counterfactual's featurized read is taken on.
CF_MODEL = "original_counterfactual"

#: Every value the applied-featurizer rows compare, under the name the rows
#: use, as the read and the model that takes it: the featurized read on base
#: and counterfactual, the raw site before and after the write, and the clean
#: and patched logits.
VALUES: dict[str, ReadRef] = {
    "f_base": ReadRef("f", "original"),
    "f_cf": ReadRef("f", CF_MODEL),
    "x_base": ReadRef("x", "original"),
    "x_patched": ReadRef("x", "patched"),
    "clean": ReadRef("logits", "original"),
    "logits": ReadRef("logits", "patched"),
}


# --------------------------------------------------------------------------- #
# seeded weights
# --------------------------------------------------------------------------- #


def _orthonormal(d: int, k: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k))
    return q.contiguous()


def _mixed_theta(units: int, seed: int) -> torch.Tensor:
    """A theta with both signs — so the hard split keeps some units and
    drops some, and neither a swap nor its complement is the whole site."""
    torch.manual_seed(seed)
    theta = torch.randn(units)
    theta[0], theta[-1] = 1.5, -1.5  # at least one of each, whatever the draw
    return theta


def _sae_tensors(d: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "enc": torch.randn(d, SAE_WIDTH) / d**0.5,
        "dec": torch.randn(SAE_WIDTH, d) / SAE_WIDTH**0.5,
        "b_enc": 0.1 * torch.randn(SAE_WIDTH),
        "b_dec": 0.1 * torch.randn(d),
    }


def _standardize_tensors(d: int, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {"mu": torch.randn(d), "sigma": torch.rand(d) + 0.5}


# --------------------------------------------------------------------------- #
# the document: a feature-space interchange, plus the reads that expose it
# --------------------------------------------------------------------------- #


def _doc(
    site: Mapping[str, Any],
    featurizers: Mapping[str, Any],
    chain: str | list[str],
    *,
    pos: Any = -1,
) -> dict[str, Any]:
    """Read the featurized value on base and counterfactual, swap the
    counterfactual's into the base forward through the same chain, and read
    the raw site both before and after the write, plus clean and patched
    logits. ``pos`` addresses the featurized read, the raw read and the
    write; the logits are always read at the last position."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": {
            "base": {"dataset": "inline", "field": "input"},
            "counterfactual": {
                "dataset": "inline",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["f", "x", "logits"]},
                CF_MODEL: {"input": "counterfactual", "reads": ["f"]},
                "patched": {
                    "input": "base",
                    "reads": ["x", "logits"],
                    "writes": ["patch"],
                },
            },
            "sites": {"tap": dict(site), "head": {"component": "lm_head"}},
            "featurizers": dict(featurizers),
            "reads": {
                "f": {"site": "tap", "pos": pos, "featurizer": chain},
                "x": {"site": "tap", "pos": pos},
                "logits": {"site": "head", "pos": -1},
            },
            "writes": {
                "patch": {
                    "site": "tap",
                    "pos": pos,
                    "featurizer": chain,
                    # `f` is listed by two models, so the operand names one
                    "do": {"swap": {"read": "f", "model": CF_MODEL}},
                }
            },
            "save": [
                saved(ref.read, ref.model, f"{name}.safetensors")
                for name, ref in VALUES.items()
                if ref.model != CF_MODEL
            ],
        },
    }


LLAMA_SITE = {"component": "block_output", "layers": LAYER}


def _stamped_loader(
    files: Mapping[str, Mapping[str, Any]], record: Mapping[str, Any] | None
):
    """A ``load_tensors`` whose one entry per file carries ``record`` as its
    per-entry stamp — what a fit's ``entries`` table records (a gate's map,
    a rotation's ``k``, a position gate's ``axis``). ``None`` is the
    hand-made, unstamped bundle ``engines.bundle_loader`` builds, which a
    gate reads as a ``sigmoid`` feature gate."""
    if record is None:
        return engines.bundle_loader(files)

    def load(path: str) -> TensorBundle:
        tensors = dict(files[path])
        return TensorBundle(
            tensors=tensors, entry_coords={key: dict(record) for key in tensors}
        )

    return load


def _both(doc, hooks_bundle, trace_bundle, tensors, *, record=None, **kwargs):
    return engines.both_executors(
        doc,
        hooks_bundle,
        trace_bundle,
        base_texts=BASE_TEXTS,
        counterfactual_texts=CF_TEXTS,
        load_tensors=_stamped_loader(tensors, record),
        **kwargs,
    )


def _same(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    sweep.assert_same(a, b, what, atol=engines.ATOL)


def _moved(patched: torch.Tensor, clean: torch.Tensor, what: str) -> None:
    assert not torch.allclose(patched, clean, atol=engines.ATOL), (
        f"{what}: the write left the value unchanged"
    )


def _assert_featurized_interchange_agrees(hooks, trace, what: str) -> dict[str, Any]:
    """The three claims of every applied-featurizer row, and anti-vacuity
    per engine. Returns the reference engine's values for row-specific
    property checks."""
    values: dict[str, dict[str, torch.Tensor]] = {"hooks": {}, "trace": {}}
    for name, ref in VALUES.items():
        values["hooks"][name] = hooks.dense_value(ref)
        values["trace"][name] = trace.dense_value(ref)
        _same(values["hooks"][name], values["trace"][name], f"{what}: {name}")
    assert values["hooks"]["f_base"].shape[0] == len(BASE_TEXTS)
    for engine, got in values.items():
        _moved(got["logits"], got["clean"], f"{engine} {what}: logits")
        _moved(got["x_patched"], got["x_base"], f"{engine} {what}: written site")
    return values["hooks"]


# --------------------------------------------------------------------------- #
# G13 / G14: loaded rank-k maps — subspace (a fitted DAS rotation) and pca
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["subspace", "pca"])
def test_loaded_rank_k_map_read_write_and_logits_agree(hooks_llama, trace_llama, kind):
    """G13 (`subspace`) and G14 (`pca`): a frozen orthonormal ``(d, K)``
    weight loaded through ``file_path``; ``f = xQ`` agrees, the swap
    ``x − xQQᵀ + f_cf Qᵀ`` agrees, and the logits agree."""
    d = hooks_llama.info.hidden_size
    q = _orthonormal(d, K, seed=13)
    tensors = {"rot.safetensors": {"weight": q}}
    doc = _doc(
        LLAMA_SITE,
        {"rot": {"kind": kind, "k": K, "file_path": "rot.safetensors"}},
        "rot",
    )
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors)
    got = _assert_featurized_interchange_agrees(hooks, trace, kind)
    # the featurized read is the projection, and the complement survives
    torch.testing.assert_close(got["f_base"], got["x_base"] @ q, atol=1e-5, rtol=0)
    want = got["x_base"] - (got["x_base"] @ q) @ q.T + got["f_cf"] @ q.T
    torch.testing.assert_close(got["x_patched"], want, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# G15: standardize
# --------------------------------------------------------------------------- #


def test_standardize_read_write_and_logits_agree(hooks_llama, trace_llama):
    """G15: ``(x − μ)/σ`` loaded from ``mu``/``sigma``; the swap in z-scored
    units lands as ``f_cf · σ + μ``."""
    d = hooks_llama.info.hidden_size
    tensors = {"std.safetensors": _standardize_tensors(d, seed=15)}
    doc = _doc(
        LLAMA_SITE,
        {"std": {"kind": "standardize", "file_path": "std.safetensors"}},
        "std",
    )
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors)
    got = _assert_featurized_interchange_agrees(hooks, trace, "standardize")
    mu, sigma = tensors["std.safetensors"]["mu"], tensors["std.safetensors"]["sigma"]
    torch.testing.assert_close(
        got["f_base"], (got["x_base"] - mu) / sigma, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        got["x_patched"], got["f_cf"] * sigma + mu, atol=1e-5, rtol=0
    )


# --------------------------------------------------------------------------- #
# G16: sae
# --------------------------------------------------------------------------- #


def test_sae_read_err_write_and_logits_agree(hooks_llama, trace_llama):
    """G16: a toy SAE (width 16) loaded from ``enc/dec/b_enc/b_dec``; the
    code ``relu((x − b_dec) enc + b_enc)`` agrees, the reconstruction error
    each engine's stage computes from its own raw read agrees, and the swap in
    code space lands as ``dec(f_cf) + err_base``."""
    d = hooks_llama.info.hidden_size
    tensors = {"sae.safetensors": _sae_tensors(d, seed=16)}
    doc = _doc(
        LLAMA_SITE, {"sae": {"kind": "sae", "file_path": "sae.safetensors"}}, "sae"
    )
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors)
    got = _assert_featurized_interchange_agrees(hooks, trace, "sae")
    assert got["f_base"].shape[-1] == SAE_WIDTH
    assert (got["f_base"] > 0).any() and (got["f_base"] == 0).any(), (
        "the toy code should be sparse but not empty"
    )
    # err through each engine's own stage over its own raw read
    x_base = VALUES["x_base"]
    _, err_hooks = hooks.stage("sae").featurize(hooks.dense_value(x_base))
    _, err_trace = trace.stage("sae").featurize(trace.dense_value(x_base))
    assert err_hooks is not None and err_trace is not None
    _same(err_hooks, err_trace, "sae: err")
    # the swap in code space keeps the BASE's reconstruction error
    want = hooks.stage("sae").inverse(got["f_cf"], err_hooks)
    torch.testing.assert_close(got["x_patched"], want, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# G17: a gate loaded from theta, under each map and readout
# --------------------------------------------------------------------------- #

#: id -> (extra spec fields, the entry's stamped map, expected hard mask given
#: theta). A fit stamps its map into the bundle and the build compares it both
#: ways, so a non-sigmoid variant's bundle has to say what it was fitted under.
#: ``boundary`` is not here: its gate must sit behind a subspace or a pca
#: (rule 4), so it is a variant of the chain test below.
GATE_VARIANTS = {
    "sigmoid": ({}, None, lambda theta: theta > 0),
    "hard_concrete": (
        {"parametrization": "hard_concrete"},
        {"parametrization": "hard_concrete"},
        lambda theta: theta > 0,
    ),
    "sigmoid_top_k": ({"top_k": 4}, None, lambda theta: _top_k_mask(theta, 4)),
    "budget_top_k": (
        {"parametrization": "budget", "top_k": 5},
        {"parametrization": "budget"},
        lambda theta: _top_k_mask(theta, 5),
    ),
}


def _top_k_mask(theta: torch.Tensor, k: int) -> torch.Tensor:
    mask = torch.zeros_like(theta, dtype=torch.bool)
    mask[torch.topk(theta, k).indices] = True
    return mask


@pytest.mark.parametrize("variant", sorted(GATE_VARIANTS))
def test_loaded_gate_read_write_and_logits_agree(hooks_llama, trace_llama, variant):
    """G17: ``theta`` with mixed signs loaded through ``file_path``; the read
    is the hard-masked value, the swap takes the counterfactual on the kept
    units and the base on the rest. Variants: the ``sigmoid`` map (θ > 0),
    ``hard_concrete`` at the default stretch (θ > 0 too), a ``top_k`` readout
    of a sigmoid theta, and a ``budget`` theta read out at ``top_k``."""
    fields, record, expected_mask = GATE_VARIANTS[variant]
    d = hooks_llama.info.hidden_size
    theta = _mixed_theta(d, seed=17)
    tensors = {"gate.safetensors": {"theta": theta}}
    doc = _doc(
        LLAMA_SITE,
        {"gate": {"kind": "gate", "file_path": "gate.safetensors", **fields}},
        "gate",
    )
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors, record=record)
    got = _assert_featurized_interchange_agrees(hooks, trace, f"gate[{variant}]")
    mask = expected_mask(theta).to(got["x_base"].dtype)
    assert 0 < int(mask.sum()) < d, "the split must keep some units and drop some"
    torch.testing.assert_close(got["f_base"], got["x_base"] * mask, atol=1e-6, rtol=0)
    want = got["x_base"] * (1 - mask) + got["f_cf"]
    torch.testing.assert_close(got["x_patched"], want, atol=1e-6, rtol=0)
    for executor in (hooks, trace):
        assert torch.equal(
            executor.stage("gate").hard_mask().bool(), expected_mask(theta)
        )


# --------------------------------------------------------------------------- #
# G18: grouped gates (by head on qwen full attention, by site on llama mlp)
# and the position axis
# --------------------------------------------------------------------------- #


def test_head_grouped_gate_agrees_on_the_full_attention_layer(hooks_qwen, trace_qwen):
    """G18 ``group: head`` on ``attention_premix`` at the fixture's
    full-attention layer: one theta per query head, the mask constant within
    each head's ``head_dim`` slice."""
    _, full_layer = sweep.stream_layers(hooks_qwen)
    heads, head_dim = hooks_qwen.info.num_heads, hooks_qwen.info.head_dim
    theta = _mixed_theta(heads, seed=18)
    tensors = {"gate.safetensors": {"theta": theta}}
    doc = _doc(
        {"component": "attention_premix", "layers": full_layer},
        {"gate": {"kind": "gate", "group": "head", "file_path": "gate.safetensors"}},
        "gate",
    )
    hooks, trace = _both(doc, hooks_qwen, trace_qwen, tensors)
    got = _assert_featurized_interchange_agrees(hooks, trace, "gate[head]")
    assert got["x_base"].shape[-1] == heads * head_dim
    mask = (theta > 0).to(got["x_base"].dtype).repeat_interleave(head_dim)
    torch.testing.assert_close(got["f_base"], got["x_base"] * mask, atol=1e-6, rtol=0)
    for executor in (hooks, trace):
        gate = executor.stage("gate")
        assert gate.groups == (heads, head_dim) and gate.theta.numel() == heads


def test_site_grouped_gate_agrees_on_mlp_output(hooks_llama, trace_llama):
    """G18 ``group: site`` on ``mlp_output``: one theta over the whole site.
    At θ > 0 the gated swap is the plain swap of the site (the write lands);
    the mask is the constant one."""
    tensors = {"gate.safetensors": {"theta": torch.tensor([1.0])}}
    doc = _doc(
        {"component": "mlp_output", "layers": LAYER},
        {"gate": {"kind": "gate", "group": "site", "file_path": "gate.safetensors"}},
        "gate",
    )
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors)
    got = _assert_featurized_interchange_agrees(hooks, trace, "gate[site]")
    torch.testing.assert_close(got["f_base"], got["x_base"], atol=0, rtol=0)
    torch.testing.assert_close(got["x_patched"], got["f_cf"], atol=1e-6, rtol=0)
    for executor in (hooks, trace):
        gate = executor.stage("gate")
        assert gate.groups == (1, hooks_llama.info.hidden_size)
        assert gate.theta.numel() == 1


#: The position gate's window: three positions after the first token. Rule 4
#: requires a fixed span, and every fixture prompt has at least eight tokens.
WINDOW = (1, 4)


def test_position_axis_gate_agrees_over_a_fixed_window(hooks_llama, trace_llama):
    """G18 ``axis: position``: a loaded gate whose θ has one entry per
    addressed position, read out in eval mode over a ``span`` window. The
    read keeps the kept positions whole and zeroes the dropped ones. The swap
    takes the counterfactual at kept positions and the base at the rest. The
    site is layer 0: the window ends before the last position, so only the
    layer above carries the write to the logits."""
    width = WINDOW[1] - WINDOW[0]
    theta = torch.tensor([1.0, -1.0, 0.5])
    assert theta.numel() == width
    tensors = {"pg.safetensors": {"theta": theta}}
    doc = _doc(
        {"component": "block_output", "layers": 0},
        {"pg": {"kind": "gate", "axis": "position", "file_path": "pg.safetensors"}},
        "pg",
        pos={"span": list(WINDOW)},
    )
    # a fit over positions stamps its axis, and the build compares it both ways
    hooks, trace = _both(
        doc, hooks_llama, trace_llama, tensors, record={"axis": "position"}
    )
    got = _assert_featurized_interchange_agrees(hooks, trace, "gate[position]")
    d = hooks_llama.info.hidden_size
    assert got["x_base"].shape == (len(BASE_TEXTS), width, d)
    mask = (theta > 0).to(got["x_base"].dtype)[None, :, None]
    torch.testing.assert_close(got["f_base"], got["x_base"] * mask, atol=1e-6, rtol=0)
    want = got["x_base"] * (1 - mask) + got["f_cf"]
    torch.testing.assert_close(got["x_patched"], want, atol=1e-6, rtol=0)
    for executor in (hooks, trace):
        gate = executor.stage("pg")
        assert gate.axis == "position" and gate.theta.numel() == width
        assert torch.equal(gate.hard_mask().bool(), theta > 0)


# --------------------------------------------------------------------------- #
# G19: the ["rot", "gate"] chain
# --------------------------------------------------------------------------- #


def _chain_case(d: int) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    q = _orthonormal(d, K, seed=19)
    theta = torch.tensor([1.0, -1.0, 0.5])  # a K-wide gate after the rotation
    tensors = {
        "rot.safetensors": {"weight": q},
        "gate.safetensors": {"theta": theta},
    }
    doc = _doc(
        LLAMA_SITE,
        {
            "rot": {"kind": "subspace", "k": K, "file_path": "rot.safetensors"},
            "gate": {"kind": "gate", "file_path": "gate.safetensors"},
        },
        ["rot", "gate"],
    )
    return doc, tensors


def test_rot_then_gate_chain_agrees(hooks_llama, trace_llama):
    """G19: a loaded subspace followed by a loaded gate sized to the
    rotation's width (the goldens' ``mask`` case): ``f = (xQ) ⊙ m``, and the
    swap through the chain keeps the base's off-features and its complement
    — ``x − xQQᵀ + ((xQ)(1−m) + f_cf) Qᵀ``."""
    d = hooks_llama.info.hidden_size
    doc, tensors = _chain_case(d)
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors)
    got = _assert_featurized_interchange_agrees(hooks, trace, "rot+gate")
    q = tensors["rot.safetensors"]["weight"]
    mask = (tensors["gate.safetensors"]["theta"] > 0).to(q.dtype)
    z_base = got["x_base"] @ q
    torch.testing.assert_close(got["f_base"], z_base * mask, atol=1e-5, rtol=0)
    z_new = z_base * (1 - mask) + got["f_cf"]
    want = got["x_base"] - z_base @ q.T + z_new @ q.T
    torch.testing.assert_close(got["x_patched"], want, atol=1e-5, rtol=0)
    for executor in (hooks, trace):
        assert executor.stage("gate").width == K


#: The boundary chain's rotation rank. It is wider than ``K``, so the prefix
#: the boundary keeps is several coordinates and still a proper subset.
BOUNDARY_RANK = 8
#: The loaded boundary θ, a fraction of the width: ``β = θ · 8 = 3.2``, so the
#: hard mask keeps the coordinates ``i < 3.2``, which are the first four.
BOUNDARY_THETA = 0.4
BOUNDARY_KEPT = 4


def test_rot_then_boundary_gate_chain_agrees(hooks_llama, trace_llama):
    """G17 ``parametrization: boundary`` (Boundless DAS) in eval mode: one
    scalar θ loaded behind a loaded subspace (rule 4 puts an indexed map
    directly after a subspace or a pca). The hard mask is the prefix
    ``i < θ · width`` of the rotation's columns, so ``f = (xQ) ⊙ m`` and the
    swap is the G19 chain inverse with that prefix mask."""
    d = hooks_llama.info.hidden_size
    q = _orthonormal(d, BOUNDARY_RANK, seed=17)
    tensors = {
        "rot.safetensors": {"weight": q},
        "bnd.safetensors": {"theta": torch.tensor([BOUNDARY_THETA])},
    }
    doc = _doc(
        LLAMA_SITE,
        {
            "rot": {
                "kind": "subspace",
                "k": BOUNDARY_RANK,
                "file_path": "rot.safetensors",
            },
            "bnd": {
                "kind": "gate",
                "parametrization": "boundary",
                "file_path": "bnd.safetensors",
            },
        },
        ["rot", "bnd"],
    )
    # an unstamped bundle reads as a sigmoid fit, so the map is stamped
    hooks, trace = _both(
        doc,
        hooks_llama,
        trace_llama,
        tensors,
        record={"parametrization": "boundary"},
    )
    got = _assert_featurized_interchange_agrees(hooks, trace, "rot+boundary")
    mask = (torch.arange(BOUNDARY_RANK) < BOUNDARY_KEPT).to(q.dtype)
    z_base = got["x_base"] @ q
    torch.testing.assert_close(got["f_base"], z_base * mask, atol=1e-5, rtol=0)
    z_new = z_base * (1 - mask) + got["f_cf"]
    want = got["x_base"] - z_base @ q.T + z_new @ q.T
    torch.testing.assert_close(got["x_patched"], want, atol=1e-5, rtol=0)
    for executor in (hooks, trace):
        gate = executor.stage("bnd")
        assert gate.parametrization == "boundary" and gate.theta.numel() == 1
        assert gate.width == BOUNDARY_RANK
        assert gate.boundary() == pytest.approx(BOUNDARY_THETA * BOUNDARY_RANK)
        assert torch.equal(gate.hard_mask(), mask)


# --------------------------------------------------------------------------- #
# G20: a top_k readout with a `rank` save, at the engine seam
# --------------------------------------------------------------------------- #


def _stamped_gate(root: Path, theta: torch.Tensor, *, layer: int) -> None:
    """What ``dbm.json`` leaves at ``fit/gate.safetensors`` on the tiny
    fixture, stamped so ``dbm_apply.json`` accepts it (§2.5 ArtifactIdentity):
    the apply document implies fp32 on tiny llama at the retargeted site."""
    target = root / "fit/gate.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = build_artifact_identity(
        model_key=TINY_LLAMA,
        model_revision="main",
        model_dtype="fp32",
        site={"component": "block_output", "layers": [layer]},
        dtype="fp32",
        parametrization="sigmoid",
        engine="pytorch_hooks",
    )
    save_file({"theta": theta.contiguous()}, str(target), metadata=identity)


def _apply_document(top_k: int) -> dict[str, Any]:
    """``dbm_apply.json`` read out at ``top_k`` with a ``rank`` table and,
    beside ``iia``, the same margin on the unpatched model (anti-vacuity at
    the engine seam). The ``logits`` read is then listed by the masked model
    and by ``original``, and each save entry names its model."""
    apply = json.loads((PROTOCOLS_DIR / "dbm_apply.json").read_text())
    method = apply["method"]
    method["featurizers"]["gate"]["top_k"] = top_k
    (iia,) = [entry for entry in method["save"] if entry["file_path"] == "iia.json"]
    method["intervened_models"]["original"] = {"input": "base", "reads": ["logits"]}
    method["save"].append(
        saved("logits", "original", "iia_base.json", dict(iia["aggregation"]))
    )
    method["save"].append({"kind": "rank", "file_path": "rank.json"})
    return apply


def _values(table: Path) -> list[Any]:
    return [row["value"] for row in json.loads(table.read_text())]


def test_top_k_readout_and_rank_table_agree_at_the_engine_seam(tmp_path: Path):
    """G20: ``dbm_apply.json`` on tiny llama with a hand-stamped sigmoid
    theta, read out at ``top_k`` = its threshold count, through
    ``run_protocol`` with each engine. ``rank.json`` (unit, theta, rank,
    hard) is exact on both sides, the aggregation tables equal, and the
    masked margin differs from the unpatched one on each engine."""
    env = engines.corpus_env(tmp_path / "artifacts")
    theta = _mixed_theta(16, seed=20)  # tiny llama's hidden size
    kept = int((theta > 0).sum())
    _stamped_gate(tmp_path / "artifacts", theta, layer=LAYER)
    apply = _apply_document(top_k=kept)
    overrides = {
        "model.key": TINY_LLAMA,
        "model.dtype": "fp32",
        "sites.target.layers": LAYER,
        **fixture_input_overrides(apply),
    }
    runs = engines.run_both(apply, env, tmp_path / "out", overrides=overrides)
    assert sorted(runs.hooks_result.files) == sorted(runs.trace_result.files)
    runs.compare()
    for out in (runs.hooks_dir, runs.trace_dir):
        rows = json.loads((out / "rank.json").read_text())
        assert len(rows) == theta.numel()
        assert sorted(r["rank"] for r in rows) == list(range(theta.numel()))
        assert all(r["top_k"] == kept for r in rows)
        assert {r["unit"] for r in rows if r["hard"]} == set(
            (theta > 0).nonzero().flatten().tolist()
        )
        by_unit = {r["unit"]: r for r in rows}
        assert all(
            by_unit[i]["theta"] == pytest.approx(float(theta[i])) for i in by_unit
        )
        masked = _values(out / "iia.json")
        base = _values(out / "iia_base.json")
        assert len(masked) == len(base) > 0
        assert masked != base, f"{out.name}: the masked swap left the margin unchanged"


# --------------------------------------------------------------------------- #
# G21: a bundle whose stamped fit contradicts the spec refuses identically
# --------------------------------------------------------------------------- #


def _refusal(executor_cls, doc, bundle, load_tensors) -> str:
    with pytest.raises(ProtocolError) as excinfo:
        executor = engines.executor_for(
            executor_cls,
            doc,
            bundle,
            base_texts=BASE_TEXTS,
            counterfactual_texts=CF_TEXTS,
            load_tensors=load_tensors,
        )
        executor.run_all()
    return str(excinfo.value)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            (
                {"rot": {"kind": "subspace", "k": K, "file_path": "rot.safetensors"}},
                "rot",
                {"rot.safetensors": {"weight": torch.eye(16)[:, :K].contiguous()}},
                {"k": "8"},
                "the document says k=3 but the selected entry was fitted with k='8'",
            ),
            id="subspace-k",
        ),
        pytest.param(
            (
                {"gate": {"kind": "gate", "file_path": "gate.safetensors"}},
                "gate",
                {"gate.safetensors": {"theta": torch.linspace(-1, 1, 16)}},
                {"parametrization": "hard_concrete"},
                "the document's gate is parametrized 'sigmoid' but the selected "
                "entry was fitted 'hard_concrete'",
            ),
            id="gate-parametrization",
        ),
    ],
)
def test_entry_identity_contradiction_refuses_identically(
    hooks_llama, trace_llama, case
):
    """G21: the stamped fit (``k``, a gate's map) contradicts the selecting
    spec; the P2 refusal from ``_check_entry_identity`` is one shared line,
    so both engines refuse with the same words."""
    featurizers, chain, files, record, phrase = case
    doc = _doc(LLAMA_SITE, featurizers, chain)
    loader = _stamped_loader(files, record)
    hooks_text = _refusal(PointExecutor, doc, hooks_llama, loader)
    trace_text = _refusal(TracePointExecutor, doc, trace_llama, loader)
    assert hooks_text == trace_text
    assert phrase in hooks_text


# --------------------------------------------------------------------------- #
# G22: featurizer_cache changes nothing but the evaluation count
# --------------------------------------------------------------------------- #


def test_featurizer_cache_scope_changes_no_result(hooks_llama, trace_llama):
    """G22: ``run_all`` inside a ``featurizer_cache`` scope on both
    executors — every read equals the unscoped run's on the same engine, and
    the two engines agree inside the scope as they do outside it."""
    d = hooks_llama.info.hidden_size
    doc, tensors = _chain_case(d)

    unscoped = _both(doc, hooks_llama, trace_llama, tensors)
    for executor in unscoped:
        executor.run_all()
    scoped = _both(doc, hooks_llama, trace_llama, tensors)
    with featurizer_cache():
        for executor in scoped:
            executor.run_all()
        for name, ref in VALUES.items():
            _same(
                scoped[0].dense_value(ref),
                scoped[1].dense_value(ref),
                f"scoped {name}",
            )
    for name, ref in VALUES.items():
        for engine, a, b in (
            ("hooks", unscoped[0], scoped[0]),
            ("trace", unscoped[1], scoped[1]),
        ):
            assert torch.equal(a.dense_value(ref), b.dense_value(ref)), (
                f"{engine}: the cache scope moved {name}"
            )


# --------------------------------------------------------------------------- #
# G23: every stage lands on its site's device, on both executors
# --------------------------------------------------------------------------- #


def _every_kind(d: int) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    """One document referencing a stage of every loadable kind."""
    q = _orthonormal(d, K, seed=23)
    tensors = {
        "rot.safetensors": {"weight": q},
        "pca.safetensors": {"weight": q},
        "std.safetensors": _standardize_tensors(d, seed=23),
        "sae.safetensors": _sae_tensors(d, seed=23),
        "gate.safetensors": {"theta": _mixed_theta(d, seed=23)},
    }
    featurizers = {
        "rot": {"kind": "subspace", "k": K, "file_path": "rot.safetensors"},
        "pca": {"kind": "pca", "k": K, "file_path": "pca.safetensors"},
        "std": {"kind": "standardize", "file_path": "std.safetensors"},
        "sae": {"kind": "sae", "file_path": "sae.safetensors"},
        "gate": {"kind": "gate", "file_path": "gate.safetensors"},
    }
    doc = _doc(LLAMA_SITE, featurizers, "rot")
    method = doc["method"]
    for name in featurizers:
        method["reads"][f"f_{name}"] = {"site": "tap", "pos": -1, "featurizer": name}
        method["intervened_models"]["original"]["reads"].append(f"f_{name}")
        method["save"].append(saved(f"f_{name}", "original", f"f_{name}.safetensors"))
    return doc, tensors


def _tensors_of(stage: Any) -> list[torch.Tensor]:
    return [*stage.parameters(), *stage.buffers()]


def test_build_stack_places_every_stage_on_the_bundle_device(hooks_llama, trace_llama):
    """G23 (CPU leg): after the first forward, every loaded stage's
    parameters and buffers sit on the device of its site's block (the
    bundle's device map) on both executors, and the nnsight executor's
    stages share the device its dispatched model is on."""
    d = hooks_llama.info.hidden_size
    doc, tensors = _every_kind(d)
    hooks, trace = _both(doc, hooks_llama, trace_llama, tensors)
    for executor in (hooks, trace):
        for name in ("rot", "pca", "std", "sae", "gate"):
            executor.read_value(ReadRef(f"f_{name}", "original"))
    for executor, bundle in ((hooks, hooks_llama), (trace, trace_llama)):
        device = bundle.devices.blocks[LAYER]
        assert set(executor.stage_cache) == {"rot", "pca", "std", "sae", "gate"}
        for name, stage in executor.stage_cache.items():
            placed = _tensors_of(stage)
            assert placed, f"{name} exposes no tensor to place"
            for tensor in placed:
                assert tensor.device.type == device.type, (
                    f"{type(executor).__name__}: {name} on {tensor.device}"
                )
    model_device = torch.device(str(trace_llama.model.device))
    assert model_device.type != "meta", "the nnsight model never dispatched"
    for name, stage in trace.stage_cache.items():
        for tensor in _tensors_of(stage):
            assert tensor.device.type == model_device.type, (
                f"nnsight: {name} on {tensor.device}, model on {model_device}"
            )
