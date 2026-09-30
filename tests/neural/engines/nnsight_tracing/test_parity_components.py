"""Engine parity on the component seams the module-boundary sweep leaves out.

Every case is the reference engine's document through both executors
(``tests._helpers.engines.both_executors``), the read or the patched logits
agreeing at ``engines.ATOL``, and — for every write — each engine's
patched value moved away from its *own* clean value, so "both agree" cannot be
satisfied by "neither landed".

* **P6** — the architectural predicates, negative case: a MoE component and
  the gated-attention ``attention_gate`` on the dense llama fixture refuse
  with the *same words* on both engines (the resolver is shared; what this
  pins is that the nnsight bundle's envoy tree is detected the way the module
  tree is).
* **G1** — the ``pytorch_fn`` mechanism (§2.8.1): a declared local function
  applied at ``block_output``; both engines declare ``pytorch_fn_local``
  (read off ``ENGINE_VERBS``) and land the same logits.
* **G2** — ``mlp_neuron_output``: the down-projection *input* tap with
  writeback, read and swapped; the nnsight side lands the write through the
  envoy's ``.input``.
* **G3** — the family plugin: the synthetic fourth family
  (``tests/_helpers/synthetic_family.py``, registered from outside
  ``causalab/neural/``) served by both engines through the unchanged
  registry adapter.

``mlp_activation``'s write (P4) joined ``LLAMA_WRITES`` in
``test_parity_module_boundaries.py``, where the rest of the llama write list
lives.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.nnsight_tracing.loading import NnsightBundle
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.registry.engines import ENGINE_VERBS, declared_capabilities
from causalab.protocol.rules.capability import refuse_shortfall, requires
from causalab.protocol.rules.document import validate_document
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.schema import (
    LAYERLESS_COMPONENTS,
    PROTOCOL_VERSION,
    SiteSpec,
    parse_document,
)

from tests._helpers import engines
from tests._helpers import synthetic_family as synth
from tests._helpers.a3b_sweep import assert_same, interchange_doc, read_doc
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    BASE_TEXTS,
    CF_TEXTS,
    _refusal,
)
from tests.protocol._docs import in_order, saved

pytestmark = pytest.mark.smoke

LAYER = 1  # the second of tiny llama's two layers, as the rest of the suite


# --------------------------------------------------------------------------- #
# shared: clean logits per engine, and a write that must have landed
# --------------------------------------------------------------------------- #


def _clean_logits(hooks_bundle: Any, trace_bundle: Any) -> dict[str, torch.Tensor]:
    hooks, trace = engines.both_executors(
        read_doc("lm_head", None), hooks_bundle, trace_bundle, base_texts=BASE_TEXTS
    )
    return {"hooks": hooks.read_value("r"), "trace": trace.read_value("r")}


@pytest.fixture(scope="module")
def llama_clean_logits(hooks_llama, trace_llama) -> dict[str, torch.Tensor]:
    return _clean_logits(hooks_llama, trace_llama)


def _write_both(
    doc: Mapping[str, Any],
    hooks_bundle: Any,
    trace_bundle: Any,
    clean: Mapping[str, torch.Tensor],
    what: str,
    *,
    logits: str = "logits",
    **kwargs: Any,
) -> None:
    """Patched logits agree across engines, and each engine's moved away from
    its own clean logits."""
    hooks, trace = engines.both_executors(
        doc, hooks_bundle, trace_bundle, base_texts=BASE_TEXTS, **kwargs
    )
    hooked, traced = hooks.dense_value(logits), trace.dense_value(logits)
    assert_same(hooked, traced, f"patched logits after {what}")
    assert not torch.allclose(hooked, clean["hooks"], atol=engines.ATOL), (
        f"pytorch_hooks: {what} left the logits unchanged"
    )
    assert not torch.allclose(traced, clean["trace"], atol=engines.ATOL), (
        f"nnsight: {what} left the logits unchanged"
    )


# --------------------------------------------------------------------------- #
# P6 — architectural predicates, the negative case
# --------------------------------------------------------------------------- #

#: Components whose predicate the dense llama fixture fails: three MoE reads
#: (no ``moe`` block — ``router_logits`` is the read-only router, the other two
#: are the interiors of the two expert paths) and the gated-attention gate.
ABSENT_ON_LLAMA = (
    "router_logits",
    "shared_expert_output",
    "expert_output",
    "attention_gate",
)


@pytest.mark.parametrize("component", ABSENT_ON_LLAMA)
def test_a_component_the_family_lacks_refuses_identically(
    hooks_llama, trace_llama, component
):
    """The refusal is the registry's, by name, naming the component and the
    layer — and the same text on both engines, because the nnsight bundle's
    envoy tree is detected by the same structural predicate as the module
    tree (reference: ``test_family_tap_table.py``, ``test_sites_round2_attention.py``)."""
    doc = read_doc(component, LAYER)
    hooks_text = _refusal(PointExecutor, doc, hooks_llama)
    trace_text = _refusal(TracePointExecutor, doc, trace_llama)
    assert hooks_text == trace_text
    assert f"'{component}'" in hooks_text and f"layer {LAYER}" in hooks_text


# --------------------------------------------------------------------------- #
# G1 — the pytorch_fn mechanism
# --------------------------------------------------------------------------- #


def negate(f: torch.Tensor) -> torch.Tensor:
    """The declared write function: ``f ← -f`` at the site — an absolute
    mechanism with no operand, so the document needs no counterfactual role."""
    return -f


def _pytorch_fn_doc() -> dict[str, Any]:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": {"base": {"dataset": "inline", "field": "input"}},
        "method": {
            "intervened_models": {
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]}
            },
            "sites": {
                "tap": {"component": "block_output", "layers": LAYER},
                "head": {"component": "lm_head"},
            },
            "reads": {"after": {"site": "head", "pos": -1}},
            "writes": {
                "patch": {
                    "site": "tap",
                    "pos": -1,
                    "do": {"pytorch_fn": {"code": "neg"}},
                }
            },
            "code": {"neg": {"locator": f"{__name__}.negate"}},
            "save": [saved("after", "patched", "after.safetensors")],
        },
    }


def test_a_pytorch_fn_write_lands_the_same_logits_on_both_engines(
    hooks_llama, trace_llama, llama_clean_logits
):
    """The declared function runs inside each engine's forward (a hook on one,
    an envoy assignment on the other) and the downstream logits agree; both
    executors' documents validated under ``engine_is_local=True`` on the way
    in (``executor_for``), and rule 13 refuses the same document for a
    non-local engine — the seam ``pytorch_fn_local`` names. Both engine
    classes read the verb from ``ENGINE_VERBS`` through
    ``declared_capabilities``, and the capability check passes for each."""
    doc = _pytorch_fn_doc()
    _write_both(
        doc,
        hooks_llama,
        trace_llama,
        llama_clean_logits,
        f"a pytorch_fn write at block_output L{LAYER}",
        logits="after",
    )
    parsed = parse_document(in_order(dict(doc)))
    validate_document(parsed, engine_is_local=True)  # what both executors ran
    with pytest.raises(ValidationError, match="pytorch_fn"):
        validate_document(parsed, engine_is_local=False)
    assert "pytorch_fn_local" in requires(parsed)
    for engine in (PytorchHooksEngine(), NnsightEngine()):
        assert "pytorch_fn_local" in ENGINE_VERBS[engine.name]
        assert engine.capabilities == declared_capabilities(engine.name)
        assert engine.is_local
        refuse_shortfall(requires(parsed), engine.effective_capabilities)


# --------------------------------------------------------------------------- #
# G2 — mlp_neuron_output: the down-projection input, read and written
# --------------------------------------------------------------------------- #


def test_mlp_neuron_output_resolves_to_the_down_projection_input(
    hooks_llama, trace_llama
):
    """Both bundles address the same tap: ``down_proj``'s *input* (the
    complete ``act(gate) * up`` product), so the nnsight write below lands
    through the envoy's ``.input`` rather than an output."""
    spec = SiteSpec(component="mlp_neuron_output", layers=(LAYER,))
    hooks_site = resolve_site(hooks_llama, spec)
    trace_site = resolve_site(trace_llama, spec)
    assert hooks_site.kind == trace_site.kind == "in"
    assert hooks_site.module is hooks_llama.model.model.layers[LAYER].mlp.down_proj
    assert trace_site.module is trace_llama.model.model.layers[LAYER].mlp.down_proj


def test_mlp_neuron_output_read_parity(hooks_llama, trace_llama):
    hooks, trace = engines.both_executors(
        read_doc("mlp_neuron_output", LAYER),
        hooks_llama,
        trace_llama,
        base_texts=BASE_TEXTS,
    )
    hooked, traced = hooks.read_value("r"), trace.read_value("r")
    assert hooked.shape[-1] == hooks_llama.info.intermediate_size
    assert_same(hooked, traced, f"read 'mlp_neuron_output' @ L{LAYER}")


def test_mlp_neuron_output_swap_parity(hooks_llama, trace_llama, llama_clean_logits):
    """A counterfactual swap of the neuron products moves the logits on both
    engines, to the same place (reference: ``test_neuron_output.py``)."""
    _write_both(
        interchange_doc("mlp_neuron_output", LAYER),
        hooks_llama,
        trace_llama,
        llama_clean_logits,
        f"a swap at mlp_neuron_output L{LAYER}",
        counterfactual_texts=CF_TEXTS,
    )


# --------------------------------------------------------------------------- #
# G3 — the family plugin, served by both engines
# --------------------------------------------------------------------------- #


class _SynthModelForTracing(synth.SynthModel):
    """The synthetic decoder behind nnsight's generic ``NNsight`` wrapper.

    The nnsight executor hands every model one positional mapping
    (``{"input_ids", "attention_mask"}``) plus ``position_ids``; the HF
    wrapper (``TransformersModel``) unpacks that for a ``PreTrainedModel``,
    which the synthetic family is not — so this subclass unpacks it itself.
    The module tree, and therefore the family the registry detects, is
    ``SynthModel``'s unchanged.
    """

    def forward(self, inputs: Any = None, **kwargs: Any) -> Any:
        if isinstance(inputs, Mapping):
            return super().forward(**inputs, **kwargs)
        return super().forward(inputs, **kwargs)


@pytest.fixture(scope="module")
def synth_hooks(hooks_llama) -> ModelBundle:
    # the tiny-llama tokenizer, already on the engines' left-pad convention;
    # the synthetic vocabulary is sized to it
    return ModelBundle(
        key=synth.KEY,
        revision="main",
        model=synth.build_model(),
        tokenizer=hooks_llama.tokenizer,
        info=synth.INFO,
        devices=DeviceMap.parse("cpu", synth.INFO.num_layers),
        dtype="fp32",
    )


@pytest.fixture(scope="module")
def synth_trace(hooks_llama) -> NnsightBundle:
    import nnsight

    torch.manual_seed(0)  # the same weights build_model(seed=0) draws
    model = _SynthModelForTracing()
    model.eval()
    model.requires_grad_(False)
    return NnsightBundle(
        key=synth.KEY,
        revision="main",
        model=nnsight.NNsight(model),
        tokenizer=hooks_llama.tokenizer,
        info=synth.INFO,
        device="cpu",
        dtype="fp32",
    )


@pytest.fixture(scope="module")
def synth_clean_logits(synth_hooks, synth_trace) -> dict[str, torch.Tensor]:
    return _clean_logits(synth_hooks, synth_trace)


def _synth_doc(doc: dict[str, Any]) -> dict[str, Any]:
    doc["model"]["key"] = synth.KEY
    return doc


#: The block-level sites the plugin declares; the layerless four are the
#: model boundary.
SYNTH_BLOCK_SITES = tuple(
    c for c in synth.SYNTHETIC_TREE.taps if c not in LAYERLESS_COMPONENTS
)
SYNTH_WRITES = ("block_mid", "mlp_activation", "mlp_output", "block_output")


def test_both_engines_detect_the_plugin_family_unchanged(synth_hooks, synth_trace):
    """The registry's adapter — registered from outside ``causalab/neural/``
    — is what both bundles resolve against: the same object, detected from
    the module tree on one side and the envoy tree on the other."""
    assert synth_hooks.adapter is synth.SYNTHETIC_TREE
    assert synth_trace.adapter is synth.SYNTHETIC_TREE
    assert (
        synth_hooks.streams == synth_trace.streams == ("full_attention",) * synth.LAYERS
    )
    hooks_weights = torch.cat([p.flatten() for p in synth_hooks.model.parameters()])
    trace_weights = torch.cat(
        [p.flatten() for p in getattr(synth_trace.model, "_module").parameters()]
    )
    assert torch.equal(hooks_weights, trace_weights), (
        "the two bundles hold different weights"
    )


@pytest.mark.parametrize(
    "component,layer",
    [(c, LAYER) for c in SYNTH_BLOCK_SITES]
    + [(c, None) for c in sorted(LAYERLESS_COMPONENTS)],
)
def test_plugin_family_read_parity(synth_hooks, synth_trace, component, layer):
    hooks, trace = engines.both_executors(
        _synth_doc(read_doc(component, layer)),
        synth_hooks,
        synth_trace,
        base_texts=BASE_TEXTS,
    )
    hooked, traced = hooks.read_value("r"), trace.read_value("r")
    assert hooked.shape[0] == len(BASE_TEXTS)
    assert_same(hooked, traced, f"plugin family read {component!r} (layer {layer})")


@pytest.mark.parametrize("component", SYNTH_WRITES)
def test_plugin_family_write_parity(
    synth_hooks, synth_trace, synth_clean_logits, component
):
    _write_both(
        _synth_doc(interchange_doc(component, LAYER)),
        synth_hooks,
        synth_trace,
        synth_clean_logits,
        f"a swap at {component!r} on the plugin family",
        counterfactual_texts=CF_TEXTS,
    )
