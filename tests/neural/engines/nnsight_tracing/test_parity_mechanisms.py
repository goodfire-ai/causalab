"""Mechanism parity: the closed ``do`` set (spec §2.8), the same document
through both executors.

The write arithmetic is shared by the two engines: ``apply_absolute``,
``apply_delta`` and ``apply_renormalize`` in
``causalab/neural/shared/mechanisms.py``, and the class order and error-term
contract in ``WriteMathMixin._apply_writes_to_contract`` and
``_written_value`` (``causalab/neural/shared/executor/writes.py``), with the
ragged landings in ``causalab/neural/shared/executor/ragged.py``. What differs
is where the written tensor is captured and put back (a hook vs a trace), so a
mechanism that reaches the model through an operand the engine has to
*deliver* (a counterfactual read, a ``params`` tensor, a seeded draw, a loaded
featurizer) is where a landing bug would show. Every case here drives the
reference engine's document through the second executor and asserts the
patched logits (and, where the mechanism defines one, the written site's
read) agree. Each side also has an anti-vacuity check: each engine's patched
value must differ from its own clean value, or "both engines agree" is
satisfiable by "neither write landed".

Cases covered (ids in the test docstrings): P7 ``add_scaled`` operands,
P8 write-class ordering, G4 ``lerp``, G5 ``affine``, G6 ``gaussian``, G7
``renormalize``, G8 ``clamp``, G9 error-term semantics under a loaded
subspace, G10 the mean-ablation handoff, G11 an operand from a loaded tensor
file, G12 ragged writes under both landing policies.

Two rows of different token lengths throughout (``BASE_TEXTS``), so a batch
axis and a padded frame are exercised rather than assumed away; the ragged
cases use the reference ragged suite's width-matched pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.tensor_files import load_file
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.shared.values import RaggedValue
from causalab.protocol import RUN_RECORD_NAME
from causalab.protocol.schema import PROTOCOL_VERSION

from tests._helpers import a3b_sweep as sweep
from tests._helpers import engines
from tests._helpers.a3b_sweep import read_doc
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    BASE_TEXTS,
    CF_TEXTS,
)
from tests.neural.engines.pytorch_hooks import test_ragged_writes as ragged_ref
from tests.neural.engines.pytorch_hooks.test_bundle_entries_run import _harvest_doc
from tests.protocol._docs import UNWRITTEN, saved

pytestmark = pytest.mark.smoke

ATOL = engines.ATOL

#: The towers a mechanism case runs on: tiny Llama at its second block, and
#: the hybrid fixture at one Gated DeltaNet block and its full-attention block
#: (layers read off the model by ``sweep.stream_layers``, not hardcoded).
TOWERS = ("llama", "qwen-deltanet", "qwen-attention")


@pytest.fixture
def tower(request) -> tuple[Any, Any, int]:
    """``(hooks bundle, trace bundle, layer)`` for one of ``TOWERS``."""
    name = request.param
    if name == "llama":
        return (
            request.getfixturevalue("hooks_llama"),
            request.getfixturevalue("trace_llama"),
            1,
        )
    hooks, trace = (
        request.getfixturevalue("hooks_qwen"),
        request.getfixturevalue("trace_qwen"),
    )
    delta_layer, full_layer = sweep.stream_layers(hooks)
    return hooks, trace, delta_layer if name == "qwen-deltanet" else full_layer


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _write_doc(
    component: str,
    layer: int,
    writes: dict[str, dict[str, Any]],
    *,
    pos: object = -1,
    with_cf: bool = True,
) -> dict[str, Any]:
    """Writes at one site in one intervened model ``patched``, which reads the
    patched logits and the written site itself (``written``, what landed).
    The un-intervened model reads the clean site on base (``v_base``, the
    pre-write value every norm/complement claim below is made against). With
    a counterfactual, a second un-intervened model reads ``v_cf`` as the read
    operand. The names are the ones ``causalab migrate`` gives (§2.9):
    ``original`` when base is the only un-intervened input, ``original_base``
    and ``original_counterfactual`` when both are. Each read is listed by one
    model, so a bare read name addresses its value on either executor."""
    reads: dict[str, Any] = {
        "logits": {"site": "head", "pos": -1},
        "written": {"site": "tap", "pos": pos},
        "v_base": {"site": "tap", "pos": pos},
    }
    models: dict[str, Any] = {}
    if with_cf:
        reads["v_cf"] = {"site": "tap", "pos": pos}
        models["original_base"] = {"input": "base", "reads": ["v_base"]}
        models[UNWRITTEN] = {"input": "counterfactual", "reads": ["v_cf"]}
    else:
        models["original"] = {"input": "base", "reads": ["v_base"]}
    models["patched"] = {
        "input": "base",
        "reads": ["logits", "written"],
        "writes": list(writes),
    }
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": sweep._data(with_cf=with_cf),  # pyright: ignore[reportPrivateUsage]
        "method": {
            "intervened_models": models,
            "sites": {
                "tap": {"component": component, "layers": [layer]},
                "head": {"component": "lm_head"},
            },
            "reads": reads,
            "writes": {
                name: {"site": "tap", "pos": pos, "do": do}
                for name, do in writes.items()
            },
            "save": [
                saved(read, model, f"{read}.safetensors")
                for read in reads
                for model, entry in models.items()
                if read in entry["reads"]
            ],
        },
    }


def _both(doc, hooks_bundle, trace_bundle, *, with_cf: bool = True, **kw):
    return engines.both_executors(
        doc,
        hooks_bundle,
        trace_bundle,
        base_texts=BASE_TEXTS,
        counterfactual_texts=CF_TEXTS if with_cf else None,
        **kw,
    )


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

_UNPATCHED: dict[int, torch.Tensor] = {}


def _unpatched(executor_cls, bundle) -> torch.Tensor:
    """Each engine's clean last-position logits, computed once per bundle."""
    key = id(bundle)
    if key not in _UNPATCHED:
        _UNPATCHED[key] = engines.executor_for(
            executor_cls, read_doc("lm_head", None), bundle, base_texts=BASE_TEXTS
        ).read_value("r")
    return _UNPATCHED[key]


def _assert_logits_parity(hooks, trace, what: str) -> None:
    """Patched logits agree, and each engine's moved away from its own clean
    logits (anti-vacuity)."""
    hooked, traced = hooks.dense_value("logits"), trace.dense_value("logits")
    sweep.assert_same(hooked, traced, f"patched logits: {what}")
    for name, executor, patched in (
        ("pytorch_hooks", hooks, hooked),
        ("nnsight", trace, traced),
    ):
        clean = _unpatched(type(executor), executor.bundle)
        assert not torch.allclose(patched, clean, atol=ATOL), (
            f"{name}: {what} left the logits unchanged"
        )


def _assert_written_parity(hooks, trace, what: str) -> tuple[Any, Any]:
    """The written site's read agrees, and on each engine differs from the
    pre-write value (a write that landed as the identity is vacuous)."""
    hooked, traced = hooks.read_value("written"), trace.read_value("written")
    sweep.assert_same(hooked, traced, f"written site: {what}")
    sweep.assert_same(
        hooks.read_value("v_base"), trace.read_value("v_base"), f"pre-write: {what}"
    )
    for name, executor, value in (
        ("pytorch_hooks", hooks, hooked),
        ("nnsight", trace, traced),
    ):
        assert not torch.allclose(value, executor.read_value("v_base"), atol=ATOL), (
            f"{name}: {what} left the written site unchanged"
        )
    return hooked, traced


def _width(hooks_bundle, component: str, layer: int) -> int:
    """The contract width of ``component`` at ``layer``, read off the
    reference engine rather than restated from the config."""
    value = engines.executor_for(
        PointExecutor, read_doc(component, layer), hooks_bundle, base_texts=BASE_TEXTS
    ).read_value("r")
    return int(value.shape[-1])


# --------------------------------------------------------------------------- #
# P7 — add_scaled: read, scalar and loaded-tensor operands
# --------------------------------------------------------------------------- #

#: (component, tower) — the residual boundary in every block type, and one
#: attention-interior tap at the full-attention block only.
ADD_SCALED_SITES = [
    ("block_output", "llama"),
    ("block_output", "qwen-deltanet"),
    ("block_output", "qwen-attention"),
    ("attention_z", "qwen-attention"),
]

#: How the ``op`` reaches the write: a counterfactual read, a literal scalar,
#: a ``params`` tensor loaded from a file.
ADD_SCALED_OPERANDS = ("read", "scalar", "loaded")


@pytest.mark.parametrize("operand", ADD_SCALED_OPERANDS)
@pytest.mark.parametrize(
    "component,tower", ADD_SCALED_SITES, indirect=["tower"], ids=lambda v: str(v)
)
def test_p7_add_scaled_parity(component, tower, operand):
    """P7: ``add_scaled`` at ``alpha`` scale of ``op``. A read operand (the
    counterfactual value at the site, alpha 0.25), a scalar operand
    (``op: -10000.0``, the shift the attention-interior suite drives nnsight-only)
    and a loaded tensor operand (a ``params`` vector of the site's width) — the
    three ways an additive delta is sourced, each on both engines."""
    hooks_bundle, trace_bundle, layer = tower
    load_tensors = None
    if operand == "read":
        do = {"add_scaled": {"op": "v_cf", "alpha": 0.25}}
    elif operand == "scalar":
        do = {"add_scaled": {"op": -10000.0, "alpha": 1.0}}
    else:
        do = {"add_scaled": {"op": "vec", "alpha": 1.0}}
        width = _width(hooks_bundle, component, layer)
        vec = torch.randn(width, generator=torch.Generator().manual_seed(11))
        load_tensors = engines.bundle_loader({"vec.safetensors": {"value": vec}})
    doc = _write_doc(component, layer, {"nudge": do}, with_cf=operand == "read")
    if operand == "loaded":
        doc["method"]["params"] = {"vec": {"file_path": "vec.safetensors"}}
    hooks, trace = _both(
        doc,
        hooks_bundle,
        trace_bundle,
        with_cf=operand == "read",
        load_tensors=load_tensors,
    )
    what = f"add_scaled({operand}) at {component!r} L{layer}"
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)


# --------------------------------------------------------------------------- #
# P8 — class order: the absolute write first, then the additive delta
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tower", ("llama", "qwen-deltanet"), indirect=True)
def test_p8_write_classes_land_in_class_order_on_both_engines(tower):
    """P8: one address, two classes — ``swap`` (absolute) and ``add_scaled``
    (additive) — authored additive-first. Both engines land them in class
    order (absolute, then the delta summed on top): the read after both is
    ``1.5·v_cf`` on each engine and agrees across them, as do the logits.

    Three classes at one address are not authorable: ``renormalize`` counts
    as the address's one absolute write (rule 8), so ``swap`` +
    ``renormalize`` is refused at validation — before either engine — and
    the renormalize leg is ``test_g7_renormalize_after_add_scaled_agrees``.
    """
    hooks_bundle, trace_bundle, layer = tower
    doc = _write_doc(
        "block_output",
        layer,
        {
            "nudge": {"add_scaled": {"op": "v_cf", "alpha": 0.5}},
            "replace": {"swap": "v_cf"},
        },
    )
    hooks, trace = _both(doc, hooks_bundle, trace_bundle)
    what = f"add_scaled then swap at block_output L{layer}"
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    for executor in (hooks, trace):
        written, v_cf = executor.read_value("written"), executor.read_value("v_cf")
        torch.testing.assert_close(written, 1.5 * v_cf, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# G7 — renormalize after an additive delta
# --------------------------------------------------------------------------- #


def _renormalize_doc(layer: int) -> dict[str, Any]:
    return _write_doc(
        "block_output",
        layer,
        {
            "nudge": {"add_scaled": {"op": 2.5, "alpha": 1.0}},
            "renorm": {"renormalize": True},
        },
        with_cf=False,
    )


@pytest.mark.parametrize("tower", TOWERS, indirect=True)
def test_g7_renormalize_after_add_scaled_agrees(tower):
    """G7, the parity half: ``add_scaled`` of a scalar then ``renormalize``
    at one address — the written site and the logits agree across engines,
    and the write landed on both."""
    hooks_bundle, trace_bundle, layer = tower
    hooks, trace = _both(
        _renormalize_doc(layer), hooks_bundle, trace_bundle, with_cf=False
    )
    what = f"add_scaled + renormalize at block_output L{layer}"
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)


def test_g7_renormalize_restores_the_pre_write_norm(hooks_llama, trace_llama):
    """G7, the semantic half (§2.8: ``f ← f·‖f₀‖/‖f‖``, ``f₀`` the pre-write
    value): after ``add_scaled`` + ``renormalize`` the written site must have
    the pre-write norm on both engines.

    The failure this pins: ``_apply_writes_to_contract``
    (``causalab/neural/shared/executor/writes.py``) lands the address's
    writes one at a time and re-gathers the tensor between them. If
    ``_written_value`` took the running value as ``f₀``, the ``renormalize``
    write would get the *post-nudge* value and rescale it to its own norm,
    which is the identity, on both engines at once. The two engines would
    then agree with each other, so the parity half above cannot see the
    defect; only a check against the pre-write value can. The landing passes
    the address's pre-write value (``v_ref``) for that reason. The assertion
    message prints the written, pre-write and nudged norms, so a regression
    shows which of the two values the write kept."""
    hooks, trace = _both(_renormalize_doc(1), hooks_llama, trace_llama, with_cf=False)
    for name, executor in (("pytorch_hooks", hooks), ("nnsight", trace)):
        written, v_base = executor.read_value("written"), executor.read_value("v_base")
        have, want = written.norm(dim=-1), v_base.norm(dim=-1)
        assert torch.allclose(have, want, atol=1e-5, rtol=1e-4), (
            f"{name}: renormalize left the norm at {have.flatten().tolist()} "
            f"(pre-write {want.flatten().tolist()}; the nudged norm is "
            f"{(v_base + 2.5).norm(dim=-1).flatten().tolist()})"
        )


# --------------------------------------------------------------------------- #
# G4 — lerp
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tower", TOWERS, indirect=True)
def test_g4_lerp_parity(tower):
    """G4: ``lerp`` — ``(1-α)·f + α·op`` with ``α = 0.3`` toward the
    counterfactual value at ``block_output``; the written site and the logits
    agree, and the landed value is the interpolation on each engine."""
    hooks_bundle, trace_bundle, layer = tower
    doc = _write_doc(
        "block_output", layer, {"mix": {"lerp": {"alpha": 0.3, "op": "v_cf"}}}
    )
    hooks, trace = _both(doc, hooks_bundle, trace_bundle)
    what = f"lerp(0.3) at block_output L{layer}"
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    for executor in (hooks, trace):
        written, v_base, v_cf = (
            executor.read_value(name) for name in ("written", "v_base", "v_cf")
        )
        torch.testing.assert_close(
            written, 0.7 * v_base + 0.3 * v_cf, atol=1e-5, rtol=1e-4
        )


# --------------------------------------------------------------------------- #
# G5 — affine
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tower", TOWERS, indirect=True)
def test_g5_affine_parity(tower):
    """G5: ``affine`` — ``f ← f·Aᵀ + b`` with ``A`` a random orthogonal
    ``(d, d)`` and ``b`` a random ``(d,)``, both ``params`` tensors loaded
    from files; the written site and the logits agree."""
    hooks_bundle, trace_bundle, layer = tower
    d = _width(hooks_bundle, "block_output", layer)
    generator = torch.Generator().manual_seed(5)
    a, _ = torch.linalg.qr(torch.randn(d, d, generator=generator))
    b = torch.randn(d, generator=generator)
    doc = _write_doc(
        "block_output",
        layer,
        {"map": {"affine": {"A": "rot", "b": "shift"}}},
        with_cf=False,
    )
    doc["method"]["params"] = {
        "rot": {"file_path": "A.safetensors"},
        "shift": {"file_path": "b.safetensors"},
    }
    hooks, trace = _both(
        doc,
        hooks_bundle,
        trace_bundle,
        with_cf=False,
        load_tensors=engines.bundle_loader(
            {"A.safetensors": {"value": a}, "b.safetensors": {"value": b}}
        ),
    )
    what = f"affine at block_output L{layer}"
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    for executor in (hooks, trace):
        written, v_base = executor.read_value("written"), executor.read_value("v_base")
        torch.testing.assert_close(written, v_base @ a.T + b, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# G6 — gaussian: the same seeded draw lands on both engines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tower", TOWERS, indirect=True)
def test_g6_gaussian_lands_the_same_draw(tower):
    """G6: ``gaussian`` (seed 7, scale 3.0). The draw is made outside the
    model — ``Generator().manual_seed(seed)`` → ``randn((batch, n_pos, d))``
    — and row-sliced, so both engines must add the SAME noise: the landed
    delta (``written - v_base``) is bit-identical across engines and equal to
    ``3·draw``; the written site agrees at ``ATOL`` (the pre-write value
    under it is float-noise different between the engines' captures) and
    the logits at ``ATOL``."""
    hooks_bundle, trace_bundle, layer = tower
    doc = _write_doc(
        "block_output",
        layer,
        {"noise": {"gaussian": {"seed": 7, "scale": 3.0, "axis": "tp_duplicated"}}},
        with_cf=False,
    )
    hooks, trace = _both(doc, hooks_bundle, trace_bundle, with_cf=False)
    what = f"gaussian(seed 7, scale 3) at block_output L{layer}"
    hooked, traced = _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    d = int(hooked.shape[-1])
    draw = torch.randn(
        (len(BASE_TEXTS), 1, d), generator=torch.Generator().manual_seed(7)
    )
    delta_hooks = hooked - hooks.read_value("v_base")
    delta_trace = traced - trace.read_value("v_base")
    assert torch.equal(delta_hooks, delta_trace), (
        "the two engines added different noise: max abs diff "
        f"{(delta_hooks - delta_trace).abs().max().item():.3e}"
    )
    assert torch.equal(delta_hooks, 3.0 * draw), "the draw is not the pinned one"


# --------------------------------------------------------------------------- #
# G8 — clamp (zero ablation as lo = hi = 0)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component,tower",
    [
        ("block_output", "llama"),
        ("block_output", "qwen-deltanet"),
        ("attention_z", "qwen-attention"),
    ],
    indirect=["tower"],
    ids=lambda v: str(v),
)
def test_g8_clamp_parity(component, tower):
    """G8: ``clamp`` with ``lo = hi = 0`` — a zero ablation spelled as a
    clamp — on the residual boundary and on an attention-interior tap; the
    written site is all zeros on both engines and the logits agree."""
    hooks_bundle, trace_bundle, layer = tower
    doc = _write_doc(
        component, layer, {"zero": {"clamp": {"lo": 0.0, "hi": 0.0}}}, with_cf=False
    )
    hooks, trace = _both(doc, hooks_bundle, trace_bundle, with_cf=False)
    what = f"clamp(0, 0) at {component!r} L{layer}"
    hooked, traced = _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    assert torch.equal(hooked, torch.zeros_like(hooked))
    assert torch.equal(traced, torch.zeros_like(traced))


# --------------------------------------------------------------------------- #
# G9 — the error term: a feature-space swap keeps the complement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tower", TOWERS, indirect=True)
def test_g9_feature_space_swap_keeps_the_complement_on_both_engines(tower):
    """G9: a ``swap`` through a loaded ``subspace`` featurizer (k = 3, an
    orthonormal ``(d, 3)`` basis from a file) replaces only the in-subspace
    coordinates; the complement comes from the pre-write value on both
    engines (§2.5 error term, ``_written_value``). The written site's read
    and the logits agree; on each engine the landed change lies in the
    subspace and the in-subspace coordinates are the counterfactual's."""
    hooks_bundle, trace_bundle, layer = tower
    d = _width(hooks_bundle, "block_output", layer)
    q, _ = torch.linalg.qr(
        torch.randn(d, 3, generator=torch.Generator().manual_seed(3))
    )
    doc = _write_doc("block_output", layer, {"patch": {"swap": "v_cf"}})
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "file_path": "rot.safetensors"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    hooks, trace = _both(
        doc,
        hooks_bundle,
        trace_bundle,
        load_tensors=engines.bundle_loader({"rot.safetensors": {"weight": q}}),
    )
    what = f"subspace(k=3) swap at block_output L{layer}"
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    sweep.assert_same(
        hooks.read_value("v_cf"),
        trace.read_value("v_cf"),
        f"featurized operand: {what}",
    )
    for executor in (hooks, trace):
        written, v_base, z_cf = (
            executor.read_value(name) for name in ("written", "v_base", "v_cf")
        )
        assert tuple(z_cf.shape[-1:]) == (3,)
        change = written - v_base
        # the complement survives: the change has no component off the subspace
        torch.testing.assert_close(
            change - (change @ q) @ q.T, torch.zeros_like(change), atol=1e-5, rtol=0
        )
        # and the in-subspace coordinates are the counterfactual's
        torch.testing.assert_close(written @ q, z_cf, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# G10 — the mean-ablation handoff: reduce at save, swap as a params operand
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def harvest(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    """The shipped harvest document (``reduce: mean`` at save, ENGINE level)
    on the weekdays fixture table, through both engines."""
    base = tmp_path_factory.mktemp("harvest")
    env = engines.corpus_env(base / "artifacts")
    return engines.run_both(_harvest_doc(reduce=True), env, base / "out")


def _json_diff_paths(a: Any, b: Any, prefix: str = "") -> set[str]:
    """Dotted paths at which two JSON trees differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        paths: set[str] = set()
        for key in set(a) | set(b):
            path = f"{prefix}.{key}" if prefix else key
            if key in a and key in b:
                paths |= _json_diff_paths(a[key], b[key], path)
            else:
                paths.add(path)
        return paths
    return set() if a == b else {prefix}


def _metadata(path: Path) -> dict[str, Any]:
    from causalab.io.tensor_files import safe_open

    with safe_open(str(path), framework="pt") as f:
        raw = dict(f.metadata() or {})
    out: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            out[key] = json.loads(value)
        except (ValueError, TypeError):
            out[key] = value
    return out


def _compare_safetensors_pinning_attn_impl(a: Path, b: Path, what: str) -> None:
    """Every tensor by key at ``ATOL``; the ``__metadata__`` header equal
    except for the engine's name and, per entry, ``loaded_attn_implementation``.

    At the engine seam the two engines run different attention kernels: the
    reference engine loads ``eager``, and the nnsight engine's own loader
    keeps the checkpoint default (``sdpa``, the on-demand switch). Every saved
    tensor's identity stamps the kernel. On the fixtures the tensors still
    agree at ``ATOL``. The harness drops the field from every entry
    (``engines.KNOWN_ENTRY_RECORD_DIFFERENCES``); this comparer pins the two
    values instead, so a change on either loader shows here."""
    ta, tb = load_file(str(a)), load_file(str(b))
    assert sorted(ta) == sorted(tb), f"{what}: tensor keys {sorted(ta)} != {sorted(tb)}"
    for key in ta:
        sweep.assert_same(ta[key], tb[key], f"{what}[{key}]")
    ma, mb = _metadata(a), _metadata(b)
    entries = set(ma.get("entries", {})) | set(mb.get("entries", {}))
    assert entries, f"{what}: no entries table in the metadata"
    assert _json_diff_paths(ma, mb) == {
        "engine",
        *(f"entries.{entry}.loaded_attn_implementation" for entry in entries),
    }, f"{what}: metadata differs beyond the pinned fields"
    for entry in entries:
        assert (
            ma["entries"][entry]["loaded_attn_implementation"],
            mb["entries"][entry]["loaded_attn_implementation"],
        ) == ("eager", "sdpa"), f"{what}: the pinned attention kernels moved"


def _compare_runs_pinning_attn_impl(runs: engines.BothRuns) -> None:
    """``engines.compare_run_dirs`` with the one metadata difference above
    pinned: same file set, tensors at ``ATOL``, JSON tables row by row, the
    receipt through ``engines.compare_records``, anything else byte-equal."""

    def files(root: Path) -> dict[str, Path]:
        return {
            str(p.relative_to(root)): p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in engines.SKIPPED_FILES
        }

    a, b = files(runs.hooks_dir), files(runs.trace_dir)
    only_a = {
        k for k in set(a) - set(b) if Path(k).name not in engines.HOOKS_ONLY_FILES
    }
    assert not only_a and not set(b) - set(a), (
        f"file sets differ: {only_a} / {set(b) - set(a)}"
    )
    for rel in sorted(set(a) & set(b)):
        pa, pb = a[rel], b[rel]
        if pa.name == RUN_RECORD_NAME:
            engines.compare_records(
                json.loads(pa.read_text()), json.loads(pb.read_text())
            )
        elif pa.suffix == ".safetensors":
            _compare_safetensors_pinning_attn_impl(pa, pb, rel)
        elif pa.suffix == ".json":
            engines.compare_json(
                json.loads(pa.read_text()), json.loads(pb.read_text()), rel
            )
        else:
            assert pa.read_bytes() == pb.read_bytes(), f"{rel}: bytes differ"


def test_g10_the_harvested_mean_agrees_at_the_engine_seam(harvest):
    """G10, first document: the saved mean vector (one ``(d,)`` tensor, not a
    row per example), the run receipt and the file set agree across engines
    (modulo the pinned ``loaded_attn_implementation`` stamp — see
    ``_compare_safetensors_pinning_attn_impl``)."""
    mean = load_file(str(harvest.hooks_dir / "acts.safetensors"))["acts"]
    assert mean.ndim == 1
    _compare_runs_pinning_attn_impl(harvest)


def test_g10_the_ablation_consumes_the_mean_identically(
    harvest, hooks_llama, trace_llama
):
    """G10, second document: the harvested mean swapped in as a ``params``
    operand (``entry.slot`` naming the producer's key) at the harvested site;
    the written site equals the mean on both engines and the logits agree."""
    mean = load_file(str(harvest.hooks_dir / "acts.safetensors"))["acts"]
    doc = _write_doc("block_output", 0, {"ablate": {"swap": "mu"}}, with_cf=False)
    doc["method"]["params"] = {
        "mu": {"file_path": "harvest/acts.safetensors", "entry": {"slot": "acts"}}
    }
    hooks, trace = _both(
        doc,
        hooks_llama,
        trace_llama,
        with_cf=False,
        load_tensors=engines.bundle_loader(
            {"harvest/acts.safetensors": {"acts": mean}}
        ),
    )
    what = "mean ablation at block_output L0"
    hooked, traced = _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)
    for value in (hooked, traced):
        torch.testing.assert_close(value, mean.expand_as(value), atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# G11 — an operand from a loaded tensor file, on a whole-tensor tap
# --------------------------------------------------------------------------- #


def test_g11_a_loaded_knockout_mask_on_the_scores_agrees(hooks_qwen, trace_qwen):
    """G11: the attention-interior suite's knockout — head 0 blocked from
    attending to token 0 by a full-shape mask added to ``attention_scores``
    upstream of the model's own softmax, the mask a ``params`` tensor loaded
    from a file — driven through the reference engine too. The whole-tensor
    landing (no position gather) agrees on the logits, and moves them on
    both engines."""
    _, layer = sweep.stream_layers(hooks_qwen)
    shape = (
        engines.executor_for(
            PointExecutor,
            read_doc("attention_scores", layer, pos="all"),
            hooks_qwen,
            base_texts=BASE_TEXTS,
        )
        .read_value("r")
        .shape
    )
    mask = torch.zeros(shape)
    mask[:, 0, :, 0] = -1e4
    doc = _write_doc(
        "attention_scores",
        layer,
        {"knock": {"add_scaled": {"op": "mask", "alpha": 1.0}}},
        pos="all",
        with_cf=False,
    )
    doc["method"]["params"] = {"mask": {"file_path": "k.safetensors"}}
    hooks, trace = _both(
        doc,
        hooks_qwen,
        trace_qwen,
        with_cf=False,
        load_tensors=engines.bundle_loader({"k.safetensors": {"value": mask}}),
    )
    what = f"loaded knockout mask on attention_scores L{layer}"
    # (no hand check of `written - v_base == mask`: the causal-masked slots
    # hold the dtype's minimum, which absorbs the -1e4 — the parity of the
    # whole written tensor and of the logits is the claim)
    _assert_written_parity(hooks, trace, what)
    _assert_logits_parity(hooks, trace, what)


# --------------------------------------------------------------------------- #
# G12 — ragged writes under both landing policies
# --------------------------------------------------------------------------- #

LANDING = ragged_ref.LANDING


def _ragged_swap_doc(policy: str) -> dict[str, Any]:
    """The reference ragged suite's ``all``-window swap under ``policy``:
    the counterfactual read is ragged like the write."""
    return ragged_ref._swap_all(policy)  # pyright: ignore[reportPrivateUsage]


def _ragged_both(policy: str, hooks_llama, trace_llama):
    return engines.both_executors(
        _ragged_swap_doc(policy),
        hooks_llama,
        trace_llama,
        base_texts=ragged_ref.BASE_TEXTS,
        counterfactual_texts=ragged_ref.CF_TEXTS,
    )


def _as_ragged(value: Any) -> RaggedValue:
    assert isinstance(value, RaggedValue), type(value)
    return value


@pytest.mark.parametrize("policy", LANDING)
def test_g12_a_ragged_all_window_swap_lands_the_same_under_each_policy(
    hooks_llama, trace_llama, policy
):
    """G12, executor seam: a whole-sequence ``swap`` (``pos: "all"``) on two
    rows of unequal length under ``policy``. The patched logits, the written
    window (a ragged read, compared flat and by widths) and the recorded
    geometry (``ragged_geometry``, the receipt's source) agree across
    engines; the window differs from the clean one on both."""
    hooks, trace = _ragged_both(policy, hooks_llama, trace_llama)
    what = f"ragged all-window swap under {policy}"
    sweep.assert_same(
        hooks.dense_value("logits"), trace.dense_value("logits"), f"logits: {what}"
    )
    after_h, after_t = (
        _as_ragged(hooks.read_value("after")),
        _as_ragged(trace.read_value("after")),
    )
    assert after_h.widths == after_t.widths and len(set(after_h.widths)) == 2
    sweep.assert_same(after_h.flat, after_t.flat, f"written window: {what}")
    assert hooks.ragged_geometry == trace.ragged_geometry
    geometry = trace.ragged_geometry[("patched", "patch")]
    assert geometry["policy"] == policy and sorted(geometry["widths"]) == sorted(
        after_t.widths
    )
    for executor, after in ((hooks, after_h), (trace, after_t)):
        v_cf = _as_ragged(executor.read_value("v_cf"))
        assert torch.equal(after.flat, v_cf.flat), "the swap did not land whole"
        clean = engines.executor_for(
            type(executor),
            read_doc("block_output", 0, pos="all"),
            executor.bundle,
            base_texts=ragged_ref.BASE_TEXTS,
        ).read_value("r")
        assert not torch.allclose(after.flat, _as_ragged(clean).flat, atol=ATOL)


def test_g12_the_two_policies_agree_with_each_other_on_nnsight(
    hooks_llama, trace_llama
):
    """The reference suite's own claim — the two landings are bit-identical
    to each other — holds on the second engine too."""
    del hooks_llama
    values = {}
    for policy in LANDING:
        executor = engines.executor_for(
            TracePointExecutor,
            _ragged_swap_doc(policy),
            trace_llama,
            base_texts=ragged_ref.BASE_TEXTS,
            counterfactual_texts=ragged_ref.CF_TEXTS,
        )
        values[policy] = (
            executor.dense_value("logits"),
            _as_ragged(executor.read_value("after")),
        )
    (la, aa), (lb, ab) = values["exact_length_buckets"], values["padded_masked"]
    assert torch.equal(la, lb)
    assert aa.widths == ab.widths and torch.equal(aa.flat, ab.flat)


def _ragged_env(root: Path) -> ResolutionEnv:
    """A resolution environment whose one dataset is the ragged pair."""
    table = root / "data" / "ragged" / "rows.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        json.dumps(
            [
                {"input": base, "counterfactual_inputs": [cf], "split": "all"}
                for base, cf in zip(ragged_ref.BASE_TEXTS, ragged_ref.CF_TEXTS)
            ]
        )
    )
    return ResolutionEnv(
        datasets=FileDatasets(root=root / "data"),
        artifacts=FileArtifacts(root=root / "artifacts"),
    )


@pytest.mark.parametrize("policy", LANDING)
def test_g12_the_receipt_records_the_same_ragged_geometry(tmp_path, policy):
    """G12, engine seam: the same ragged swap through ``run_protocol`` on
    each engine. The receipt's ``execution.ragged`` block — policy, per-row
    widths, ``[width, rows]`` buckets — is the same on both, and the run
    directories agree file by file."""
    doc = _ragged_swap_doc(policy)
    doc["model"] = {"key": TINY_LLAMA, "revision": "main"}
    for role in doc["data"].values():
        role["dataset"] = "ragged/rows"
    env = _ragged_env(tmp_path)
    runs = engines.run_both(doc, env, tmp_path / "out")
    receipts = [
        json.loads((d / RUN_RECORD_NAME).read_text())
        for d in (runs.hooks_dir, runs.trace_dir)
    ]
    for receipt in receipts:
        ragged = receipt["execution"]["ragged"]["patched/patch"]
        assert ragged["policy"] == policy
        assert len(set(ragged["widths"])) == 2
        assert ragged["buckets"] == [
            [w, ragged["widths"].count(w)] for w in sorted(set(ragged["widths"]))
        ]
    assert receipts[0]["execution"]["ragged"] == receipts[1]["execution"]["ragged"]
    _compare_runs_pinning_attn_impl(runs)
