"""Shared refusal triggers for protocol validation and engine execution.

Load triggers take no arguments and run without torch. Runtime triggers use
lazy tiny-model fixtures. They call public entry points such as resolve_site,
canonicalize, and component_width. The capture script records their exception
class, code, path, and message; snapshot tests compare those records.

NOT_RUNNABLE lists cases without a suitable fixture. RETIRED lists operations
that the engines now support. The snapshot keeps their reasons for coverage
accounting. Supported fused GPT-2 q/k/v sites are tested in test_family_tap_table;
DeltaNet alias handling and unsupported interior sites are tested in
test_deltanet_interior.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Any, Callable

from causalab.protocol.schema.explicit import canonicalize
from causalab.protocol.engine import Engine, requires
from causalab.protocol.registry import (
    ModelInfo,
    component_shape,
    component_width,
    get_model_info,
    register_model,
)
from causalab.protocol.rules.capability import refuse_shortfall
from causalab.protocol.rules.document import validate_document
from causalab.protocol.schema import SiteSpec, parse_document

from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import FIXTURES, build_env


__all__ = [
    "A3B",
    "LOAD_TRIGGERS",
    "NOT_RUNNABLE",
    "RETIRED",
    "RUN_TRIGGERS",
    "Fixtures",
    "SNAPSHOT",
]

#: Where the capture lives.
SNAPSHOT = FIXTURES / "refusal_snapshot.json"

A3B = "Qwen/Qwen3.6-35B-A3B"
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_QWEN35_MOE = "tiny-random/qwen3.5-moe"
#: The fixture's one full-attention layer (0–2 are Gated DeltaNet).
FULL_ATTENTION_LAYER = 3
LINEAR_ATTENTION_LAYER = 0

TEXT = "the quick brown fox jumps"
COUNTERFACTUAL_TEXT = "a slow green turtle sleeps deeply"

#: A MoE model with every optional width, so the load-time triggers can name
#: MoE components offline. Registered at import, like the registry tests' own.
MOE = ModelInfo(
    key="snapshot/moe",
    hidden_size=64,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    head_dim=16,
    intermediate_size=128,
    vocab_size=1000,
    num_experts=32,
    num_experts_per_tok=4,
    shared_expert_intermediate_size=48,
    moe_intermediate_size=24,
)
register_model(MOE)

ENV = build_env(FIXTURES / "artifacts")


# --------------------------------------------------------------------------- #
# load-time (pure verbs, torch-free)
# --------------------------------------------------------------------------- #


class _BareEngine(Engine):
    """An engine serving no component at all — the shortfall generator's input."""

    name = "bare"
    capabilities = frozenset({"paired_forward", "full_logits"})
    components = frozenset()
    writable_components = frozenset()

    def execute(self, compiled: Any, run: Any) -> Any:  # pragma: no cover - never run
        raise AssertionError("routing only")


def _doc(model: str, site: dict[str, Any]) -> dict[str, Any]:
    raw = base_doc()
    raw["model"]["key"] = model
    raw["method"]["sites"]["tgt"] = site
    return raw


def _canonicalize(raw: dict[str, Any]) -> None:
    canonicalize(in_order(raw), ENV)


def _validate(raw: dict[str, Any]) -> None:
    """The checklist over one concrete document, with the static model
    metadata the canonicalizer used to read — rule 4's address half (a site's
    layer, stream, component and head) moved there from ``canonicalize`` into
    the rules package, and these triggers follow it."""
    validate_document(parse_document(in_order(raw)), model_info=ENV.model_info)


def _trigger_1() -> None:
    # routing between engines is retired; the same shortfall is the
    # generated rule-13 refusal against the one chosen engine
    refuse_shortfall(
        requires(parse_document(base_doc())), _BareEngine().effective_capabilities
    )


def _trigger_2() -> None:
    _validate(_doc("gpt2", {"component": "block_output", "layers": [3], "head": 0}))


def _trigger_3() -> None:
    _validate(
        _doc("gpt2", {"component": "attention_premix", "layers": [3], "head": 99})
    )


def _trigger_4() -> None:
    raw = _doc(MOE.key, {"component": "router_scores", "layers": [0]})
    raw["method"]["featurizers"] = {"f": {"kind": "pca", "k": 2}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "f"
    raw["method"]["writes"]["patch"]["featurizer"] = "f"
    _canonicalize(raw)


def _trigger_5() -> None:
    component_width(MOE, "input_ids")


def _trigger_6() -> None:
    component_width(MOE, "block_output", head=0)


def _trigger_7() -> None:
    component_shape(get_model_info("gpt2"), "router_logits")


def _trigger_8() -> None:
    get_model_info("snapshot/unregistered")


def _trigger_9() -> None:
    _validate(
        _doc(
            A3B,
            {"component": "block_output", "layers": [0], "stream": "full_attention"},
        )
    )


def _trigger_10() -> None:
    _validate(_doc(A3B, {"component": "attention_premix", "layers": [0]}))


LOAD_TRIGGERS: dict[str, Callable[[], None]] = {
    "1": _trigger_1,
    "2": _trigger_2,
    "3": _trigger_3,
    "4": _trigger_4,
    "5": _trigger_5,
    "6": _trigger_6,
    "7": _trigger_7,
    "8": _trigger_8,
    "9": _trigger_9,
    "10": _trigger_10,
}


# --------------------------------------------------------------------------- #
# run-time (engines, tiny fixtures)
# --------------------------------------------------------------------------- #


class Fixtures:
    """The tiny models the run-time triggers need, loaded on first use.

    ``load_model`` caches per key, so a test module's own session fixtures and
    these properties share one load."""

    @functools.cached_property
    def hooks_qwen(self) -> Any:
        from causalab.neural.engines.pytorch_hooks.loading import load_model

        return load_model(TINY_QWEN35_MOE)

    @functools.cached_property
    def hooks_llama(self) -> Any:
        from causalab.neural.engines.pytorch_hooks.loading import load_model

        return load_model(TINY_LLAMA)

    @functools.cached_property
    def hooks_qwen_eager_experts(self) -> Any:
        """The fixture dispatched on the per-expert loop, so the table pins that
        dispatch too. Built directly — ``load_model`` deliberately has no
        experts knob."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            TINY_QWEN35_MOE,
            dtype=torch.float32,
            attn_implementation="eager",
            experts_implementation="eager",
        )
        model.eval()
        model.requires_grad_(False)
        return dataclasses.replace(self.hooks_qwen, model=model)

    @functools.cached_property
    def trace_qwen(self) -> Any:
        from causalab.neural.engines.nnsight_tracing.loading import load_model

        return load_model(TINY_QWEN35_MOE, attn_implementation="eager")


def _sweep() -> Any:
    """The A3B sweep's document builders — imported lazily because that module
    imports torch, and the load-time half of this table must not."""
    from tests._helpers import a3b_sweep

    return a3b_sweep


def _executor(doc_raw: dict[str, Any], bundle: Any, *, trace: bool = False) -> Any:
    """An executor over ``doc_raw`` that *parses but does not validate*.

    The run-time refusals are for documents arriving unvalidated; a trigger
    that validated first would hit the load-time twin of two of
    them and never reach the run-time path this snapshot pins."""
    if trace:
        from causalab.neural.engines.nnsight_tracing.executor import (
            TracePointExecutor as cls,
        )
    else:
        from causalab.neural.engines.pytorch_hooks.executor import (
            PointExecutor as cls,
        )
    doc = parse_document(in_order(doc_raw))
    rows = [{"input": TEXT, "counterfactual_inputs": [COUNTERFACTUAL_TEXT]}]
    return cls(
        doc,
        bundle,
        role_rows={"base": rows, "counterfactual": rows},
        role_fields={"base": "input", "counterfactual": "counterfactual_inputs[0]"},
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
    )


def _resolve(bundle: Any, **site: Any) -> None:
    """Resolve one site named the way a document names it: a retired spelling
    folds onto its replacement first, as the parser folds it (identity on the
    base, where none of these spellings was retired)."""
    from causalab.neural.shared.sites import resolve_site
    from causalab.protocol.schema import DEPRECATED_COMPONENTS

    site["component"] = DEPRECATED_COMPONENTS.get(site["component"], site["component"])
    if "layer" in site:  # the snapshot's triggers name one layer; the field is a band
        site["layers"] = (site.pop("layer"),)
    resolve_site(bundle, SiteSpec(**site))


def _trigger_11(fx: Fixtures) -> None:
    doc = _sweep().interchange_doc("router_logits", LINEAR_ATTENTION_LAYER)
    _executor(doc, fx.hooks_qwen).read_value("logits")


def _trigger_12(fx: Fixtures) -> None:
    doc = _sweep().interchange_doc("expert_idx", LINEAR_ATTENTION_LAYER)
    doc["method"]["writes"]["patch"]["do"] = {"add_scaled": {"op": -1.0, "alpha": 1.0}}
    _executor(doc, fx.hooks_qwen).read_value("logits")


def _trigger_13(fx: Fixtures) -> None:
    doc = _sweep().interchange_doc("attention_probs", FULL_ATTENTION_LAYER, pos="all")
    doc["method"]["writes"]["patch"]["do"] = {"add_scaled": {"op": -1.0, "alpha": 1.0}}
    _executor(doc, fx.hooks_qwen).read_value("logits")


def _trigger_14(fx: Fixtures) -> None:
    _resolve(
        fx.hooks_qwen,
        component="block_output",
        layer=LINEAR_ATTENTION_LAYER,
        stream="full_attention",
    )


def _trigger_15(fx: Fixtures) -> None:
    _resolve(fx.hooks_qwen, component="attention_premix", layer=LINEAR_ATTENTION_LAYER)


def _trigger_16(fx: Fixtures) -> None:
    _resolve(fx.hooks_qwen, component="delta_qkv", layer=FULL_ATTENTION_LAYER)


def _trigger_17(fx: Fixtures) -> None:
    _resolve(fx.hooks_qwen, component="deltanet_qkv", layer=FULL_ATTENTION_LAYER)


def _trigger_18(fx: Fixtures) -> None:
    _resolve(
        fx.hooks_qwen, component="router_scores", layer=LINEAR_ATTENTION_LAYER, expert=3
    )


def _trigger_19(fx: Fixtures) -> None:
    _resolve(
        fx.hooks_qwen,
        component="expert_activation",
        layer=LINEAR_ATTENTION_LAYER,
        expert=128,
    )


def _trigger_20(fx: Fixtures) -> None:
    _resolve(
        fx.hooks_qwen_eager_experts,
        component="expert_activation",
        layer=LINEAR_ATTENTION_LAYER,
    )


def _trigger_21(fx: Fixtures) -> None:
    _resolve(fx.hooks_qwen, component="delta_qkv", layer=LINEAR_ATTENTION_LAYER, head=0)


def _trigger_22(fx: Fixtures) -> None:
    _resolve(
        fx.hooks_qwen, component="delta_state", layer=LINEAR_ATTENTION_LAYER, head=99
    )


def _trigger_24(fx: Fixtures) -> None:
    _resolve(fx.hooks_llama, component="attention_gate", layer=0)


def _trigger_27(fx: Fixtures) -> None:
    _resolve(fx.hooks_llama, component="router_logits", layer=0)


def _trigger_29(fx: Fixtures) -> None:
    _resolve(fx.hooks_qwen, component="mlp_activation", layer=LINEAR_ATTENTION_LAYER)


def _trigger_31(fx: Fixtures) -> None:
    doc = _sweep().read_doc("expert_activation", LINEAR_ATTENTION_LAYER)
    doc["method"]["sites"]["tap"]["expert"] = 0
    _executor(doc, fx.trace_qwen, trace=True).read_value("r")


def _trigger_33(fx: Fixtures) -> None:
    doc = copy.deepcopy(_sweep().read_doc("expert_activation", LINEAR_ATTENTION_LAYER))
    doc["method"]["sites"]["tap"]["expert"] = 0
    doc["method"]["reads"]["r"]["dims"] = [0, 1]
    _executor(doc, fx.hooks_qwen).read_value("r")


def _trigger_34(fx: Fixtures) -> None:
    doc = _sweep().read_doc("attention_probs", FULL_ATTENTION_LAYER, pos=-1)
    _executor(doc, fx.hooks_qwen).read_value("r")


RUN_TRIGGERS: dict[str, Callable[[Fixtures], None]] = {
    "11": _trigger_11,
    "12": _trigger_12,
    "13": _trigger_13,
    "14": _trigger_14,
    "15": _trigger_15,
    "16": _trigger_16,
    "17": _trigger_17,
    "18": _trigger_18,
    "19": _trigger_19,
    "20": _trigger_20,
    "21": _trigger_21,
    "22": _trigger_22,
    "24": _trigger_24,
    "27": _trigger_27,
    "29": _trigger_29,
    "31": _trigger_31,
    "33": _trigger_33,
    "34": _trigger_34,
}

#: Census rows whose refusal a later PR deliberately turned into a pass. The
#: entry stays in the snapshot as ``captured: false`` with the reason, so the
#: census stays complete (rows 1–34) and the decision is on record.
RETIRED: dict[str, str] = {
    "30": (
        "'deltanet_qkv' is an alias of "
        "'delta_qkv' — the eight DeltaNet tensors the two engines reached under "
        "two spellings carry one name each (schema.DEPRECATED_COMPONENTS), and "
        "the reference engine serves the name through in_proj_qkv's output. The "
        "refusal survives for the three faces only the nnsight engine serves "
        "('deltanet_query', 'deltanet_key', 'deltanet_state' — registry."
        "BACKEND_PAIRS) and is pinned in tests/neural/engines/nnsight_tracing/"
        "test_deltanet_interior.py::test_the_reference_engine_refuses_by_name."
    ),
    "23": (
        "GPT-2's fused c_attn no longer refuses the interior "
        "q/k/v — the per-family tap table (registry.Capability.overrides, "
        "family 'gpt2') addresses them as the three H·d column blocks of "
        "c_attn's output, read and written through the layout conversion's "
        "fused scatter. The refusal survives for a fused-projection family "
        "WITHOUT a row and is pinned in "
        "tests/neural/engines/pytorch_hooks/test_family_tap_table.py."
    ),
}

#: Census rows with no fixture that reaches them; recorded, not silently absent.
NOT_RUNNABLE: dict[str, str] = {
    "25": (
        "sites._attention_interior_site: a gated q-projection with no q_norm/"
        "k_norm to tap after it — no cached fixture family has that shape "
        "(qwen3.5-moe has the norms, llama and gpt2 are ungated)"
    ),
    "26": (
        "sites._q_projection_splits: a q-projection of neither H·d nor 2·H·d "
        "width — no cached fixture family has one"
    ),
    "28": (
        "sites._moe_site: a sparse-MoE block without a shared expert — the "
        "only MoE fixture (tiny-random/qwen3.5-moe) has one at every layer"
    ),
    "32": (
        "nnsight executor: an interior component with no address in its "
        "tables — every interior component in the vocabulary has one today, so "
        "the branch is reachable only by editing the address table"
    ),
}
