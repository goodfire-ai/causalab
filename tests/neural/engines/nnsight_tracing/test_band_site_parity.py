"""A band site through both engines (spec §2.4 ``layers``).

Both executors lower a band to its per-layer members before resolving a
module (``executor/base.py``), so each must run a one-site band exactly as it
runs the hand-written two-site document — and the two engines agree with each
other to the parity tolerance, as they do on every one-layer site.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.executor import TracePointExecutor
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.protocol.schema import PROTOCOL_VERSION

from tests.neural.engines.nnsight_tracing.test_parity_module_boundaries import (
    ATOL,
    _assert_same,
    _data,
    _executor,
)
from tests.protocol._docs import UNWRITTEN, saved

pytestmark = pytest.mark.smoke


def _band_doc(one_site: bool) -> dict:
    """Swap the counterfactual's attention output at both of tiny-random's
    layers into the base forward: as one band site, or as two sites."""
    if one_site:
        sites = {"a": {"component": "attention_output", "layers": [0, 1]}}
        reads = {"v": {"site": "a", "pos": -1}}
        writes = {"w": {"site": "a", "pos": -1, "do": {"swap": "v"}}}
        in_force = ["w"]
    else:
        sites = {
            f"a{i}": {"component": "attention_output", "layers": [i]} for i in (0, 1)
        }
        reads = {f"v{i}": {"site": f"a{i}", "pos": -1} for i in (0, 1)}
        writes = {
            f"w{i}": {"site": f"a{i}", "pos": -1, "do": {"swap": f"v{i}"}}
            for i in (0, 1)
        }
        in_force = ["w0", "w1"]
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": list(reads)},
                "patched": {"input": "base", "reads": ["logits"], "writes": in_force},
            },
            "sites": {**sites, "head": {"component": "lm_head"}},
            "reads": {**reads, "logits": {"site": "head", "pos": -1}},
            "writes": writes,
            "save": [saved("logits", "patched", "l.safetensors")],
        },
    }


@pytest.mark.parametrize(
    "executor_cls, which",
    [(PointExecutor, "hooks_llama"), (TracePointExecutor, "trace_llama")],
    ids=["pytorch_hooks", "nnsight"],
)
def test_each_engine_runs_a_band_as_its_hand_written_twin(request, executor_cls, which):
    bundle = request.getfixturevalue(which)
    band = _executor(executor_cls, _band_doc(one_site=True), bundle, with_cf=True)
    hand = _executor(executor_cls, _band_doc(one_site=False), bundle, with_cf=True)
    assert sorted(band.doc.sites) == ["a[layers=0]", "a[layers=1]", "head"]  # lowered
    assert torch.equal(band.read_value("logits"), hand.read_value("logits"))
    assert band.read_value("logits").shape[0] == 2


def test_the_engines_agree_on_a_band(hooks_llama, trace_llama):
    hooked = _executor(
        PointExecutor, _band_doc(one_site=True), hooks_llama, with_cf=True
    )
    traced = _executor(
        TracePointExecutor, _band_doc(one_site=True), trace_llama, with_cf=True
    )
    _assert_same(
        hooked.read_value("logits"), traced.read_value("logits"), "band logits"
    )
    assert ATOL < 1e-3
