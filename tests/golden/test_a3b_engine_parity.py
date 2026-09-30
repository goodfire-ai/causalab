"""The engine-agreement sweep on the real Qwen/Qwen3.6-35B-A3B.

The smoke half (``tests/neural/engines/nnsight_tracing/test_parity_a3b_sweep.py``)
runs this table on ``tiny-random/qwen3.5-moe`` — the same architecture at four
layers and hidden 8. What a tiny-random fixture cannot show is whether the taps
still land when the tensors are real: 40 layers on the documented 3-linear-then-1-full
schedule, 256 experts routed top-8, a decoupled ``head_dim`` of 256 over 16 query
heads and 2 KV heads, and bf16 instead of fp32. This tier is that check.

**Why the sweep is staged rather than parametrized over two live models.**
The A3B is ~70 GB in bf16, so the smoke tier's shape — both engines loaded, one
document driven through both inside each test — would need two copies resident
at once. Instead each engine captures the whole sweep in turn and is then freed
(`_capture`), and the comparisons run over the captured tensors on CPU.
One model resident at a time, which is what makes this fit on a single
accelerator.

**Tolerance.** The smoke tier's 1e-5 is an fp32 number. 📐 Here the measurement
is stronger than any band: on the real checkpoint in bf16, all 111 compared
cases agree at max abs diff **exactly 0.0** — the two engines differ in how they
capture a tensor, not in what the model computes, and the same eager kernels
over the same weights produce the same bits. `ATOL` is kept as a band
rather than zero only to absorb a future release that dispatches a different
kernel; at 1e-2 it is well under one bf16 ulp at the logit magnitude this model
produces (|max| ~18, ulp ~0.06), so it cannot admit a real disagreement. The run
writes the measured maxima to ``$A3B_PARITY_REPORT`` when that is set.
"""

from __future__ import annotations

import gc
import json
import os

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor

from tests._helpers import a3b_sweep as sweep
from tests._helpers.kernel_paths import library_kernel_paths

pytestmark = pytest.mark.golden

MODEL = "Qwen/Qwen3.6-35B-A3B"
DTYPE = "bf16"

#: Reads must agree to this band. Both engines run eager over the same bf16
#: weights, so 📐 the measured maxima are what justify it — see the module
#: docstring and the report the run writes.
ATOL = 1e-2

#: Two rows, so the batch axis is exercised at real width.
#:
#: ⚠️ The base and its counterfactual must differ **at the patched position**.
#: The sweep patches the last position, and `embeddings`, `block_input@L0` and
#: `attention_input_norm@L0` there are functions of the last token alone — so a
#: pair ending in the same token makes those three interchanges swap a tensor
#: for itself. Both engines then agree on a result that means nothing, which is
#: exactly what the anti-vacuity assertion exists to catch (it did:
#: `test_the_counterfactual_differs_at_the_patched_position` is the guard
#: that keeps it from coming back as a data artifact rather than a code bug).
ROWS = [
    {
        "input": "The Eiffel Tower stands in the city of Paris",
        "counterfactual_inputs": ["The Colosseum stands in the city of Rome"],
    },
    {
        "input": "The capital city of Japan is Tokyo",
        "counterfactual_inputs": ["The capital city of Norway is Oslo"],
    },
]

#: > 64 tokens, so the chunked delta kernel runs more than one chunk.
LONG_ROWS = [
    {
        "input": (
            "The old lighthouse stood at the edge of the harbour, and every "
            "evening its keeper climbed the spiral stair to light the lamp "
            "before the fishing boats turned for home across the grey water "
            "of the northern bay, guided by nothing else in the dark. "
        )
        * 2,
        "counterfactual_inputs": [
            (
                "The new observatory sat on the ridge above the valley, and "
                "each night its astronomer walked the gravel path to open the "
                "dome before the winter clouds rolled in across the high "
                "plateau, with nothing else to see by in the cold. "
            )
            * 2
        ],
    }
]


def _device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("the golden tier is the accelerator tier")
    return "cuda"


# --------------------------------------------------------------------------- #
# the sweep's coordinates, resolved against the real tower
# --------------------------------------------------------------------------- #


def _cases(delta_layer: int, full_layer: int) -> list[tuple[str, str, int | None]]:
    """``(kind, component, layer)`` for every read the sweep performs."""
    cases: list[tuple[str, str, int | None]] = [
        ("layerless", c, None) for c in sweep.SHARED_LAYERLESS
    ]
    cases += [
        ("deltanet_layer", c, delta_layer)
        for c in sweep.SHARED_ANY_STREAM + sweep.SHARED_LINEAR_ONLY
    ]
    cases += [
        ("full_layer", c, full_layer)
        for c in sweep.SHARED_ANY_STREAM + sweep.SHARED_FULL_ONLY
    ]
    return cases


def _case_id(kind: str, component: str, layer: int | None) -> str:
    return f"{kind}:{component}" + ("" if layer is None else f"@L{layer}")


# --------------------------------------------------------------------------- #
# capture: one engine at a time, then freed
# --------------------------------------------------------------------------- #


def _capture(executor_cls, bundle, cases, *, want_writes: bool) -> dict:
    """Run the whole sweep on one engine and return CPU tensors.

    Everything is moved off the accelerator as it is captured: the point of
    staging the engines is that neither the second model nor the comparison
    needs the first one's memory.
    """
    out: dict[str, torch.Tensor] = {}
    for kind, component, layer in cases:
        doc = sweep.read_doc(component, layer, pos=sweep.default_pos(component))
        value = sweep.make_executor(
            executor_cls, doc, bundle, rows=ROWS, with_cf=False
        ).read_value("r")
        out[f"read/{_case_id(kind, component, layer)}"] = value.detach().to("cpu")
    # the counterfactual's token at the patched position — the precondition
    # every write case rests on, captured rather than assumed
    cf_ids = sweep.read_doc("input_ids", None)
    cf_ids["data"]["counterfactual"] = {
        "dataset": "inline",
        "field": "counterfactual_inputs[0]",
    }
    cf_ids["method"]["intervened_models"]["original"]["input"] = "counterfactual"
    out["cf_input_ids"] = (
        sweep.make_executor(executor_cls, cf_ids, bundle, rows=ROWS, with_cf=True)
        .read_value("r")
        .detach()
        .to("cpu")
    )
    clean = sweep.read_doc("lm_head", None)
    out["unpatched"] = (
        sweep.make_executor(executor_cls, clean, bundle, rows=ROWS, with_cf=False)
        .read_value("r")
        .detach()
        .to("cpu")
    )
    if want_writes:
        for kind, component, layer in cases:
            if component in sweep.READ_ONLY:
                continue
            doc = sweep.interchange_doc(
                component, layer, pos=sweep.default_pos(component)
            )
            logits = sweep.make_executor(
                executor_cls, doc, bundle, rows=ROWS, with_cf=True
            ).dense_value("logits")
            out[f"write/{_case_id(kind, component, layer)}"] = logits.detach().to("cpu")
    return out


def _capture_delta_family(executor_cls, bundle, layer: int, which: int) -> dict:
    """The DeltaNet interior, in whichever vocabulary this engine serves.

    ``which`` selects the element of each `sweep.DELTA_FAMILY_PAIRS`
    entry — 0 for the reference engine's spelling, 1 for the nnsight
    engine's — of the three typed pairs.
    """
    out: dict[str, torch.Tensor] = {}
    for pair in sweep.DELTA_FAMILY_PAIRS:
        component = pair[which]
        doc = sweep.read_doc(component, layer, pos="all")
        value = sweep.make_executor(
            executor_cls, doc, bundle, rows=LONG_ROWS, with_cf=False
        ).read_value("r")
        out[component] = value.detach().to("cpu")
    return out


@pytest.fixture(scope="module")
def captures():
    """Both engines' captures of the whole sweep, one model resident at a time."""
    device = _device()
    report_path = os.environ.get("A3B_PARITY_REPORT")

    from causalab.neural.engines.pytorch_hooks.loading import load_model as load_hooks

    hooks_bundle = load_hooks(MODEL, dtype=DTYPE, device=device)
    delta_layer, full_layer = sweep.stream_layers(hooks_bundle)
    streams = tuple(
        hooks_bundle.stream_at(i) for i in range(len(hooks_bundle.model.model.layers))
    )
    info = hooks_bundle.info
    cases = _cases(delta_layer, full_layer)
    # like against like: the nnsight engine runs the library's forward, so the
    # reference engine's optional kernel paths (each with its own goldens and
    # band) are off for its capture — tests/_helpers/kernel_paths.py
    with library_kernel_paths():
        hooks = _capture(PointExecutor, hooks_bundle, cases, want_writes=True)
        hooks_delta = _capture_delta_family(PointExecutor, hooks_bundle, delta_layer, 0)

    # free the reference engine before the second copy is asked for
    load_hooks.cache_clear()
    del hooks_bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from causalab.neural.engines.nnsight_tracing.loading import load_model as load_trace

    # eager pinned: the reference engine loads eager, and parity must compare
    # like against like (the same rule the smoke fixtures follow)
    trace_bundle = load_trace(
        MODEL, dtype=DTYPE, device=device, attn_implementation="eager"
    )
    trace = _capture(TracePointExecutor, trace_bundle, cases, want_writes=True)
    trace_delta = _capture_delta_family(
        TracePointExecutor, trace_bundle, delta_layer, 1
    )

    load_trace.cache_clear()
    del trace_bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bundle = {
        "cases": cases,
        "delta_layer": delta_layer,
        "full_layer": full_layer,
        "streams": streams,
        "hooks": hooks,
        "trace": trace,
        "cf_input_ids": hooks["cf_input_ids"],
        "hooks_delta": hooks_delta,
        "trace_delta": trace_delta,
        # kept from the first bundle: `align_delta_pair` needs the declared
        # linear-stream head counts, and the models are gone by comparison time
        "info": info,
    }
    if report_path:
        _write_report(bundle, report_path)
    return bundle


def _write_report(bundle: dict, path: str) -> None:
    """What the tolerance was chosen from — the measured maxima, per case."""
    rows = {}
    for key, left in bundle["hooks"].items():
        right = bundle["trace"].get(key)
        if right is None or left.shape != right.shape:
            rows[key] = {
                "shape_mismatch": [
                    list(left.shape),
                    list(right.shape) if right is not None else None,
                ]
            }
            continue
        rows[key] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "max_abs_diff": float((left.double() - right.double()).abs().max()),
            "max_abs_value": float(left.double().abs().max()),
        }
    payload = {
        "model": MODEL,
        "dtype": DTYPE,
        "streams": list(bundle["streams"]),
        "delta_layer": bundle["delta_layer"],
        "full_layer": bundle["full_layer"],
        "atol": ATOL,
        "cases": rows,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


# --------------------------------------------------------------------------- #
# the assertions
# --------------------------------------------------------------------------- #


def _ids(kind: str) -> list[str]:
    # the parametrization must not need a loaded model, so it derives from the
    # documented schedule rather than from the fixture: layers 3, 7, ... 39 are
    # the full-attention ones (`full_attention_interval` 4), the rest DeltaNet.
    groups = {
        "layerless": sweep.SHARED_LAYERLESS,
        "deltanet_layer": sweep.SHARED_ANY_STREAM + sweep.SHARED_LINEAR_ONLY,
        "full_layer": sweep.SHARED_ANY_STREAM + sweep.SHARED_FULL_ONLY,
    }
    return list(groups[kind])


def _key(captures: dict, kind: str, component: str, prefix: str) -> str:
    layer = {
        "layerless": None,
        "deltanet_layer": captures["delta_layer"],
        "full_layer": captures["full_layer"],
    }[kind]
    return f"{prefix}/{_case_id(kind, component, layer)}"


@pytest.mark.parametrize("kind", ["layerless", "deltanet_layer", "full_layer"])
def test_read_parity(captures, kind):
    """Every shared hookpoint reads the same on both engines, at real scale."""
    failures = []
    for component in _ids(kind):
        key = _key(captures, kind, component, "read")
        left, right = captures["hooks"][key], captures["trace"][key]
        try:
            sweep.assert_same(left, right, f"read {key}", atol=ATOL)
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("kind", ["layerless", "deltanet_layer", "full_layer"])
def test_write_parity(captures, kind):
    """Every intervention's downstream effect agrees — and landed."""
    failures = []
    for component in _ids(kind):
        if component in sweep.READ_ONLY:
            continue
        key = _key(captures, kind, component, "write")
        left, right = captures["hooks"][key], captures["trace"][key]
        try:
            sweep.assert_same(left, right, f"patched logits, {key}", atol=ATOL)
        except AssertionError as exc:
            failures.append(str(exc))
            continue
        for engine, patched in (("hooks", left), ("trace", right)):
            if torch.allclose(patched, captures[engine]["unpatched"], atol=ATOL):
                failures.append(f"{engine}: the interchange at {key} moved nothing")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "hooks_component,trace_component,relation", sweep.DELTA_FAMILY_PAIRS
)
def test_delta_family_cross_engine_agreement(
    captures, hooks_component, trace_component, relation
):
    """The three DeltaNet tensors that stay two names on the real checkpoint —
    the typed backend pairs (`registry.BACKEND_PAIRS`), agreed after the
    declared transform; the eight one-name tensors are ordinary shared cases
    of the deltanet layer above."""
    assert sweep.backend_pair(hooks_component).relation == relation
    left, right = sweep.align_delta_pair(
        captures["hooks_delta"][hooks_component],
        captures["trace_delta"][trace_component],
        hooks_component,
        captures["info"],
    )
    sweep.assert_same(
        left,
        right,
        f"{hooks_component!r} (pytorch_hooks) vs {trace_component!r} (nnsight)",
        atol=ATOL,
    )


def test_the_counterfactual_differs_at_the_patched_position(captures):
    """The precondition every write case rests on.

    `input_ids` is read at the same position the sweep patches, so this asks
    the captures directly: if the base and counterfactual carry the same token
    there, an interchange at any component that is a function of that token
    alone swaps a tensor for itself, and "both engines agree" becomes true for
    the wrong reason.
    """
    ids = captures["hooks"]["read/layerless:input_ids"]
    cf_ids = captures["cf_input_ids"]
    assert not torch.equal(ids, cf_ids), (
        "every row's base and counterfactual end in the same token, so a "
        f"last-position interchange is a no-op: {ids.tolist()} == {cf_ids.tolist()}"
    )


def test_the_tower_is_the_documented_hybrid_schedule(captures):
    """The architecture claim the docs table and the diagram both rest on:
    40 layers, 30 Gated DeltaNet and 10 full attention. If a revision changes
    the schedule, the table is wrong and this is where it shows."""
    streams = captures["streams"]
    assert len(streams) == 40, streams
    assert streams.count("linear_attention") == 30
    assert streams.count("full_attention") == 10
