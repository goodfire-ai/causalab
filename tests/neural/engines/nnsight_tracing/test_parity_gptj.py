"""The GPT-J tree through both engines: the engine-parity harness on
``hf-internal-testing/tiny-random-GPTJForCausalLM``.

The reference engine is the oracle (``tests/neural/engines/pytorch_hooks/
test_family_gptj.py`` holds it to raw hooks on the HF modules). This module
is what decides whether the nnsight engine serves the GPT-J tree: every
component the family declares reads the same on both engines, a swap at each
writable one moves the logits the same, a generated read agrees, and one
compiled document run through ``run_protocol`` on each engine writes the same
directory modulo the harness's listed differences. It passes, so the nnsight
engine serves GPT-J, and no refusal by name is declared for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.nnsight_tracing.loading import (
    NnsightBundle,
)
from causalab.neural.engines.nnsight_tracing.loading import (
    load_model as load_trace_model,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle
from causalab.neural.engines.pytorch_hooks.loading import (
    load_model as load_hooks_model,
)
from causalab.protocol.registry import CAPABILITIES, GPTJ_TREE, components_served_by
from causalab.protocol.schema import LAYERLESS_COMPONENTS

from tests._helpers import engines
from tests.neural.engines.nnsight_tracing.test_generate_frame_nnsight import _gen_doc
from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    _assert_same,
    _executor,
    _interchange_doc,
    _read_doc,
)
from tests.protocol._env import CORPUS_DIR

pytestmark = pytest.mark.smoke

TINY_GPTJ = "hf-internal-testing/tiny-random-GPTJForCausalLM"
LAYER = 2

#: The per-layer components the family declares that the architecture has
#: (the gate is refused by its predicate on both engines) and the nnsight
#: engine's rows serve: today every one of them, ``attention_result``
#: included.
PER_LAYER = sorted(
    c
    for c in GPTJ_TREE.taps
    if c not in LAYERLESS_COMPONENTS
    and c != "attention_gate"
    and c in components_served_by("nnsight")
)
WRITABLE = [c for c in PER_LAYER if CAPABILITIES[c].writes is not None]


@pytest.fixture(scope="module")
def hooks_gptj() -> ModelBundle:
    return load_hooks_model(TINY_GPTJ)


@pytest.fixture(scope="module")
def trace_gptj() -> NnsightBundle:
    # eager pinned explicitly, as for the other fixtures: parity compares like
    # against like (GPT-J has no sdpa path in transformers 5.16.1 anyway)
    return load_trace_model(TINY_GPTJ, attn_implementation="eager")


def test_both_engines_detect_the_gptj_tree(hooks_gptj, trace_gptj):
    assert hooks_gptj.adapter is GPTJ_TREE and trace_gptj.adapter is GPTJ_TREE
    assert hooks_gptj.streams == trace_gptj.streams == ("full_attention",) * 5


@pytest.mark.parametrize("component", PER_LAYER)
def test_every_declared_component_reads_the_same(hooks_gptj, trace_gptj, component):
    doc = _read_doc(component, LAYER)
    hooked = _executor(PointExecutor, doc, hooks_gptj, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_gptj, with_cf=False)
    _assert_same(hooked, traced.read_value("r"), f"read of {component!r}")


@pytest.mark.parametrize("component", sorted(LAYERLESS_COMPONENTS - {"input_ids"}))
def test_every_model_boundary_component_reads_the_same(
    hooks_gptj, trace_gptj, component
):
    doc = _read_doc(component, None)
    hooked = _executor(PointExecutor, doc, hooks_gptj, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_gptj, with_cf=False)
    _assert_same(hooked, traced.read_value("r"), f"read of {component!r}")


@pytest.mark.parametrize("component", WRITABLE)
def test_a_swap_at_every_writable_component_moves_the_logits_the_same(
    hooks_gptj, trace_gptj, component
):
    doc = _interchange_doc(component, LAYER)
    hooked = _executor(PointExecutor, doc, hooks_gptj, with_cf=True)
    traced = _executor(TracePointExecutor, doc, trace_gptj, with_cf=True)
    _assert_same(
        hooked.read_value("logits"),
        traced.read_value("logits"),
        f"patched logits after a swap at {component!r}",
    )


@pytest.mark.parametrize(
    "component,layer", [("block_output", LAYER), ("lm_head", None)]
)
def test_a_generated_read_agrees(hooks_gptj, trace_gptj, component, layer):
    doc = _gen_doc(component, layer=layer)
    hooked = _executor(PointExecutor, doc, hooks_gptj, with_cf=False).read_value("r")
    traced = _executor(TracePointExecutor, doc, trace_gptj, with_cf=False)
    _assert_same(hooked, traced.read_value("r"), f"generated read of {component!r}")


def _engine_seam_doc() -> dict[str, Any]:
    """The corpus interchange's data and shape, moved onto three GPT-J sites
    and saved as tensors: the tiny fixture's 1024-token vocabulary splits the
    corpus answers (' Saturday' is five tokens), so the ``match`` and
    ``logit_diff`` tables it scores have no single-token answer here."""
    import json

    raw = json.loads((CORPUS_DIR / "02_interchange_im.json").read_text())
    method = raw["method"]
    method["sites"] = {
        "resid": {"component": "block_output", "layers": [1]},
        "premix": {"component": "attention_premix", "layers": [LAYER], "head": 1},
        "act": {"component": "mlp_activation", "layers": [3]},
        "lm_head": {"component": "lm_head"},
    }
    method["intervened_models"] = {
        "original_counterfactual": {
            "input": "counterfactual",
            "reads": ["v_resid", "v_premix", "v_act"],
        },
        "patched": {
            "input": "base",
            "reads": ["logits"],
            "writes": ["w_resid", "w_premix", "w_act"],
        },
    }
    method["reads"] = {
        "v_resid": {"site": "resid", "pos": -1},
        "v_premix": {"site": "premix", "pos": -1},
        "v_act": {"site": "act", "pos": -1},
        "logits": {"site": "lm_head", "pos": -1},
    }
    method["writes"] = {
        f"w_{name}": {"site": name, "pos": -1, "do": {"swap": f"v_{name}"}}
        for name in ("resid", "premix", "act")
    }
    method["save"] = [
        {"read": "logits", "model": "patched", "file_path": "logits.safetensors"},
        {
            "read": "v_premix",
            "model": "original_counterfactual",
            "file_path": "premix.safetensors",
        },
    ]
    return raw


def test_one_document_agrees_at_the_engine_seam(tmp_path: Path):
    """``run_protocol`` through each engine on one compiled GPT-J document:
    the same files, tensors at ``ATOL``, the receipt whole after the
    harness's listed differences."""
    env = engines.corpus_env(tmp_path / "artifacts")
    runs = engines.run_both(
        _engine_seam_doc(), env, tmp_path / "out", overrides={"model.key": TINY_GPTJ}
    )
    assert sorted(runs.hooks_result.files) == sorted(runs.trace_result.files)
    assert {"logits.safetensors", "premix.safetensors"} <= {
        Path(f).name for f in runs.hooks_result.files
    }
    runs.compare()
