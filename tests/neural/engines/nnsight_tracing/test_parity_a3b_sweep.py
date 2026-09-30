"""The full engine-agreement sweep over the Qwen3.6-35B-A3B hookpoint surface.

``test_parity_module_boundaries.py`` proved the nnsight engine on a *sample* of
the vocabulary. This is the sweep the sample stood in for: every hookpoint the
A3B architecture exposes, in both of its block types, read **and** written,
with a completeness guard that fails when a new component joins the vocabulary
without joining the sweep.

Three claims, one per engine relationship:

1. **shared vocabulary** — the same document through both engines agrees, on
   the read (`test_read_parity_*`) and on the intervention's downstream
   effect (`test_write_parity_*`);
2. **single-engine vocabulary** — the ``delta_*`` kernel interior and the
   ``deltanet_*``/``expert_permutation`` fused-forward interiors have no
   same-component counterpart, so agreement is asserted where it actually
   exists: 📐 measured, the two vocabularies name the *same tensors* through
   two unrelated mechanisms (`test_delta_family_cross_engine_agreement`);
3. **the seam** — a component only one engine serves refuses by name on the
   other, and refuses with the *same words* where the policy is shared.

Anti-vacuity is not optional here: "both engines agree" is trivially true if
neither write landed, so every write case also asserts the patched logits moved.

Tier: smoke. Budget note in ``docs/TESTS.md`` terms — this file is the
component sweep on the tiny fixture; the real-checkpoint half is
``tests/golden/test_a3b_engine_parity.py``.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import COMPONENTS

from tests._helpers import a3b_sweep as sweep

pytestmark = pytest.mark.smoke

#: Two rows, so a batch axis is exercised rather than assumed away.
BASE_TEXTS = ["the quick brown fox jumps", "a small red hen sits still"]
CF_TEXTS = ["a slow green turtle sleeps", "the big blue whale dives deep"]

ROWS = [
    {"input": base, "counterfactual_inputs": [cf]}
    for base, cf in zip(BASE_TEXTS, CF_TEXTS)
]

#: > 64 tokens, so the chunked delta kernel runs more than one chunk and the
#: per-chunk state has something to align against.
LONG_ROWS = [
    {
        "input": "the quick brown fox jumps over the lazy dog and runs far away " * 10,
        "counterfactual_inputs": [
            "a slow green turtle sleeps beneath the old stone bridge at noon " * 10
        ],
    }
]


# --------------------------------------------------------------------------- #
# the sweep's coordinates
# --------------------------------------------------------------------------- #


def _hooks(doc, bundle, *, rows=ROWS, with_cf: bool):
    return sweep.make_executor(PointExecutor, doc, bundle, rows=rows, with_cf=with_cf)


def _trace(doc, bundle, *, rows=ROWS, with_cf: bool):
    return sweep.make_executor(
        TracePointExecutor, doc, bundle, rows=rows, with_cf=with_cf
    )


@pytest.fixture(scope="module")
def layers(hooks_qwen) -> tuple[int, int]:
    """(a Gated DeltaNet layer, a full-attention layer) of the fixture tower."""
    return sweep.stream_layers(hooks_qwen)


def _read_both(component, layer, hooks_bundle, trace_bundle, *, rows=ROWS):
    doc = sweep.read_doc(component, layer, pos=sweep.default_pos(component))
    hooked = _hooks(doc, hooks_bundle, rows=rows, with_cf=False).read_value("r")
    traced = _trace(doc, trace_bundle, rows=rows, with_cf=False).read_value("r")
    return hooked, traced


# --------------------------------------------------------------------------- #
# 1. reads — every shared component, in every block type it exists in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", sweep.SHARED_LAYERLESS)
def test_read_parity_layerless(hooks_qwen, trace_qwen, component):
    hooked, traced = _read_both(component, None, hooks_qwen, trace_qwen)
    sweep.assert_same(hooked, traced, f"read {component!r}")


@pytest.mark.parametrize("component", sweep.SHARED_ANY_STREAM)
def test_read_parity_deltanet_layer(hooks_qwen, trace_qwen, layers, component):
    delta_layer, _ = layers
    hooked, traced = _read_both(component, delta_layer, hooks_qwen, trace_qwen)
    sweep.assert_same(hooked, traced, f"read {component!r} @ DeltaNet L{delta_layer}")


@pytest.mark.parametrize("component", sweep.SHARED_ANY_STREAM + sweep.SHARED_FULL_ONLY)
def test_read_parity_full_attention_layer(hooks_qwen, trace_qwen, layers, component):
    _, full_layer = layers
    hooked, traced = _read_both(component, full_layer, hooks_qwen, trace_qwen)
    sweep.assert_same(hooked, traced, f"read {component!r} @ full-attn L{full_layer}")


# --------------------------------------------------------------------------- #
# 2. writes — the intervention's downstream effect agrees, and it landed
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def unpatched_logits(hooks_qwen, trace_qwen):
    """The clean last-position logits from each engine — the baseline every
    write case is checked to have moved away from."""
    doc = sweep.read_doc("lm_head", None)
    return {
        "hooks": _hooks(doc, hooks_qwen, with_cf=False).read_value("r"),
        "trace": _trace(doc, trace_qwen, with_cf=False).read_value("r"),
    }


def _write_both(component, layer, hooks_bundle, trace_bundle, unpatched):
    doc = sweep.interchange_doc(component, layer, pos=sweep.default_pos(component))
    hooked = _hooks(doc, hooks_bundle, with_cf=True).dense_value("logits")
    traced = _trace(doc, trace_bundle, with_cf=True).dense_value("logits")
    where = f"{component!r}" + ("" if layer is None else f" @ L{layer}")
    sweep.assert_same(hooked, traced, f"patched logits after a swap at {where}")
    # anti-vacuity: agreement must not be reachable by "neither write landed"
    assert not torch.allclose(hooked, unpatched["hooks"], atol=sweep.ATOL), (
        f"pytorch_hooks: the interchange at {where} left the logits unchanged"
    )
    assert not torch.allclose(traced, unpatched["trace"], atol=sweep.ATOL), (
        f"nnsight: the interchange at {where} left the logits unchanged"
    )


@pytest.mark.parametrize("component", sweep.write_cases(sweep.SHARED_LAYERLESS))
def test_write_parity_layerless(hooks_qwen, trace_qwen, unpatched_logits, component):
    _write_both(component, None, hooks_qwen, trace_qwen, unpatched_logits)


@pytest.mark.parametrize("component", sweep.write_cases(sweep.SHARED_ANY_STREAM))
def test_write_parity_deltanet_layer(
    hooks_qwen, trace_qwen, layers, unpatched_logits, component
):
    delta_layer, _ = layers
    _write_both(component, delta_layer, hooks_qwen, trace_qwen, unpatched_logits)


@pytest.mark.parametrize(
    "component",
    sweep.write_cases(sweep.SHARED_ANY_STREAM + sweep.SHARED_FULL_ONLY),
)
def test_write_parity_full_attention_layer(
    hooks_qwen, trace_qwen, layers, unpatched_logits, component
):
    _, full_layer = layers
    _write_both(component, full_layer, hooks_qwen, trace_qwen, unpatched_logits)


# --------------------------------------------------------------------------- #
# 3. the DeltaNet interior: no shared component, but the same tensors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", sweep.SHARED_LINEAR_ONLY)
def test_read_parity_deltanet_interior(hooks_qwen, trace_qwen, layers, component):
    """The Gated DeltaNet interior under its one name: the same
    document, read through the reference engine's hooks and kernel-global swaps
    and through the nnsight engine's envoys and ``.source`` lines — two
    unrelated mechanisms, one tensor, one spelling. Was the eight ``identical``
    pairs of the two-vocabulary table."""
    delta_layer, _ = layers
    hooked, traced = _read_both(component, delta_layer, hooks_qwen, trace_qwen)
    sweep.assert_same(hooked, traced, f"read {component!r} @ DeltaNet L{delta_layer}")


@pytest.mark.parametrize("component", sweep.write_cases(sweep.SHARED_LINEAR_ONLY))
def test_write_parity_deltanet_interior(
    hooks_qwen, trace_qwen, layers, unpatched_logits, component
):
    delta_layer, _ = layers
    _write_both(component, delta_layer, hooks_qwen, trace_qwen, unpatched_logits)


@pytest.mark.parametrize(
    "hooks_component,trace_component,relation", sweep.DELTA_FAMILY_PAIRS
)
def test_delta_family_cross_engine_agreement(
    hooks_qwen, trace_qwen, layers, hooks_component, trace_component, relation
):
    """The three DeltaNet tensors that stay two names — the typed backend pairs
    (``registry.BACKEND_PAIRS``): the reference engine's post-tiling q/k and
    per-step state against the nnsight engine's pre-tiling q/k and per-chunk
    state, agreed after the declared transform. The relation is read from the
    registry by the alignment helper, not passed in: a wrong address cannot be
    massaged into agreement here.
    """
    delta_layer, _ = layers
    hooked = _hooks(
        sweep.read_doc(hooks_component, delta_layer, pos="all"),
        hooks_qwen,
        rows=LONG_ROWS,
        with_cf=False,
    ).read_value("r")
    traced = _trace(
        sweep.read_doc(trace_component, delta_layer, pos="all"),
        trace_qwen,
        rows=LONG_ROWS,
        with_cf=False,
    ).read_value("r")
    assert sweep.backend_pair(hooks_component).relation == relation
    left, right = sweep.align_delta_pair(
        hooked, traced, hooks_component, hooks_qwen.info
    )
    sweep.assert_same(
        left,
        right,
        f"{hooks_component!r} (pytorch_hooks) vs {trace_component!r} (nnsight), "
        f"related by {relation!r}",
    )


def test_the_delta_family_tensors_are_not_all_the_same_tensor(hooks_qwen, layers):
    """Anti-vacuity for the DeltaNet interior: the eleven captures (the eight
    one-name components and the three paired ones) must be eleven different
    tensors, or 'they agree' would be satisfiable by a tap that returns the
    same thing for every component."""
    delta_layer, _ = layers
    seen: list[tuple[str, torch.Tensor]] = []
    for hooks_component in sweep.SHARED_LINEAR_ONLY + tuple(
        pair[0] for pair in sweep.DELTA_FAMILY_PAIRS
    ):
        value = _hooks(
            sweep.read_doc(hooks_component, delta_layer, pos="all"),
            hooks_qwen,
            rows=LONG_ROWS,
            with_cf=False,
        ).read_value("r")
        for name, other in seen:
            if other.shape == value.shape:
                assert not torch.allclose(other, value, atol=sweep.ATOL), (
                    f"{hooks_component!r} and {name!r} captured the same tensor"
                )
        seen.append((hooks_component, value))


# --------------------------------------------------------------------------- #
# 4. the seam: what one engine does not serve, the other does — by name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", sweep.HOOKS_ONLY)
def test_the_nnsight_engine_does_not_claim_the_kernel_interior(component):
    engine = NnsightEngine()
    assert component not in engine.components
    assert component not in engine.writable_components
    assert component in PytorchHooksEngine().components


@pytest.mark.parametrize("component", sweep.NNSIGHT_ONLY)
def test_the_reference_engine_does_not_claim_the_fused_interior(component):
    engine = PytorchHooksEngine()
    assert component not in engine.components
    assert component not in engine.writable_components
    assert component in NnsightEngine().components


def test_mlp_activation_does_not_exist_on_this_architecture(
    hooks_qwen, trace_qwen, layers
):
    """The one vocabulary entry the A3B has no tensor for: its MLP is a sparse
    MoE block with no ``act_fn``. Both engines refuse, and the refusal names
    the block's children so the author can see why — the A3B analogues are
    ``expert_activation`` and ``shared_expert_activation``, which the sweep
    above covers."""
    delta_layer, _ = layers
    doc = sweep.read_doc("mlp_activation", delta_layer)
    for label, cls, bundle in (
        ("pytorch_hooks", PointExecutor, hooks_qwen),
        ("nnsight", TracePointExecutor, trace_qwen),
    ):
        with pytest.raises((NotImplementedError, ProtocolError)) as excinfo:
            sweep.make_executor(cls, doc, bundle, rows=ROWS, with_cf=False).read_value(
                "r"
            )
        assert "mlp_activation" in str(excinfo.value), label


# --------------------------------------------------------------------------- #
# 5. completeness — a new component cannot join the vocabulary unswept
# --------------------------------------------------------------------------- #


def test_every_component_is_claimed_by_exactly_one_bucket():
    """The guard that makes this file a *sweep* rather than a sample.

    Adding a component to ``schema.Component`` without deciding which engines
    serve it and which block type it lives in fails here, naming it — the same
    discipline the corpus digests apply to documents.
    """
    unclaimed = sweep.unclaimed_components()
    assert not unclaimed, (
        f"components in the vocabulary but in no sweep bucket: {list(unclaimed)}. "
        "Add each to tests/_helpers/a3b_sweep.py — SHARED_* if both engines "
        "serve it, HOOKS_ONLY/NNSIGHT_ONLY if one does, ABSENT_ON_A3B if this "
        "architecture has no such tensor."
    )
    twice = sweep.double_claimed_components()
    assert not twice, f"components claimed by two buckets: {list(twice)}"


def test_the_buckets_match_what_the_engines_declare():
    """The partition is a restatement of the engines' own ``components`` sets;
    this is what keeps the restatement honest."""
    hooks = PytorchHooksEngine()
    trace = NnsightEngine()
    shared = (
        set(sweep.SHARED_LAYERLESS)
        | set(sweep.SHARED_ANY_STREAM)
        | set(sweep.SHARED_FULL_ONLY)
        | set(sweep.SHARED_LINEAR_ONLY)
    )
    both_declared = set(hooks.components) & set(trace.components)
    # `mlp_activation` is declared by both engines and exists on neither of this
    # architecture's block types — the one documented subtraction.
    assert shared == both_declared - set(sweep.ABSENT_ON_A3B)
    assert set(sweep.HOOKS_ONLY) == set(hooks.components) - set(trace.components)
    assert set(sweep.NNSIGHT_ONLY) == set(trace.components) - set(hooks.components)
    assert set(sweep.ABSENT_ON_A3B) <= set(COMPONENTS)
