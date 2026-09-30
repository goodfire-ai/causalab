"""The address-table CI canary.

One trace per fixture stream type resolves **every** table entry — match,
peel, field, value — and fails with the op-inventory diff on any miss. It has
to *run a trace*, not parse: recursive ``.source`` drilling only exists
inside one. This is the tripwire for a transformers bump — the
``uv.lock`` revision is the real pin, and their own history shows why
(transformers 5 renamed GPT-2's dropout op and broke nnterp's address) — and
the artifact that travels upstream with ``addresses.py`` later.

Plus the upstreaming discipline itself: ``addresses.py`` imports nothing from
the rest of ``causalab``, pinned by a subprocess import so this
suite's own imports cannot mask a violation.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from causalab.neural.engines.nnsight_tracing.addresses import (
    ADDRESSES,
    AddressResolutionError,
    match_op,
)
from causalab.neural.engines.nnsight_tracing.executor import (
    ResolvedTap,
    TracePointExecutor,
)
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.schema import PROTOCOL_VERSION, SiteSpec

from tests.protocol._docs import saved

from .test_parity_module_boundaries import _data, _executor

pytestmark = pytest.mark.smoke

TEXT = "the quick brown fox jumps"


def _layer_of(bundle, stream: str) -> int | None:
    for layer, carried in enumerate(bundle.streams):
        if carried == stream:
            return layer
    return None


def _canary_doc(layer: int) -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": _data(with_cf=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": "block_output", "layers": [layer]}},
            "reads": {"r": {"site": "tap", "pos": -1}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def test_every_table_entry_resolves_in_one_trace(trace_qwen):
    """The canary proper. Navigation reuses the executor's own drill and
    presentation — the same code path a document takes — so a green canary
    means real documents resolve, not merely that the strings match."""
    import nnsight

    from causalab.neural.engines.nnsight_tracing.addresses import MOE_EXPERTS

    tables: list[tuple[str, dict, int | None]] = [
        (stream, dict(table), _layer_of(trace_qwen, stream))
        for stream, table in ADDRESSES.items()
    ]
    # the per-expert interior is not a mixer stream — every fixture layer
    # carries a sparse-MoE block, so layer 0 stands for all of them
    tables.append(("moe_experts", dict(MOE_EXPERTS), 0))

    saves: dict[str, object] = {}
    for label, table, layer in tables:
        if not table:
            continue  # an empty table has nothing to drift
        assert layer is not None, f"no {label!r} layer on the fixture"
        executor = _executor(
            TracePointExecutor, _canary_doc(layer), trace_qwen, with_cf=False
        )
        executor._trace_sources = {}
        # eager attention pinned by the fixture and grouped_mm the experts
        # default, so entries requiring either resolve here too; entries are
        # drilled in declaration order, which is forward order — the same
        # discipline the executor's rank sort enforces
        with torch.no_grad():
            with trace_qwen.model.trace(TEXT):
                for component, address in table.items():
                    site = resolve_site(
                        trace_qwen, SiteSpec(component=component, layers=(layer,))
                    )
                    tap = ResolvedTap(site=site, source=address)
                    if address.fires != "once":
                        # a per-fire entry resolves match + peel + field +
                        # trip; the canary reads the count and the first fire
                        value_op, trip_op = executor._navigate_fire_ops(tap)
                        saves[f"{component}:trip"] = nnsight.save(len(trip_op.output))
                        saves[component] = nnsight.save(value_op.output)
                        continue
                    raw, perm = executor._source_value(tap)
                    saves[component] = nnsight.save(
                        executor._present_native(tap, raw, perm)
                    )
    assert saves, "the canary resolved nothing — every table is empty?"
    for component, value in saves.items():
        if component.endswith(":trip"):
            assert int(value) >= 1, component
        else:
            assert isinstance(value, torch.Tensor) and value.numel() > 0, component


def test_a_missing_pattern_refuses_with_the_inventory():
    with pytest.raises(AddressResolutionError) as excinfo:
        match_op("nonexistent_op", ["real_op_0", "other_op_1"])
    message = str(excinfo.value)
    assert "real_op_0" in message and "other_op_1" in message


def test_an_ambiguous_pattern_refuses_rather_than_guessing():
    """Two hits, neither a call of the symbol: no basis to choose."""
    lines = {"weights_0": "weights = a + b", "weights_1": "weights = weights * c"}
    with pytest.raises(AddressResolutionError, match="2 ops match"):
        match_op("weights", list(lines), lines.__getitem__)


def test_the_call_op_wins_over_the_assignment():
    """The systematic ambiguity — a variable assigned, then called — resolves
    to the call, which is how 'attention_interface' names one op out of two."""
    lines = {
        "attention_interface_0": "attention_interface: Callable = get_interface(",
        "attention_interface_1": "out, weights = attention_interface(",
    }
    assert (
        match_op("attention_interface", list(lines), lines.__getitem__)
        == "attention_interface_1"
    )


def test_addresses_imports_nothing_from_causalab():
    """The upstreaming rule, by construction: the table+matcher module must
    stand alone — moving it to nnterp later is then a file move, not a
    rewrite. Loaded by file path in a subprocess (the package ``__init__``
    would drag the engine in), so a green run means the module's own body
    pulled in nothing of causalab at all."""
    import causalab.neural.engines.nnsight_tracing.addresses as addresses

    path = addresses.__file__
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('addresses', {path!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['addresses'] = module  # dataclasses resolves through it\n"
        "spec.loader.exec_module(module)\n"
        "polluted = [m for m in sys.modules if m.startswith('causalab')]\n"
        "assert not polluted, polluted\n"
        "assert module.FULL_ATTENTION  # and it is the real module, tables loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
