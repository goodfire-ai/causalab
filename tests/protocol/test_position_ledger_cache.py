"""Cached ledgers must describe each executed point's lowered addresses."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from causalab.io.env import ResolutionEnv
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.pipeline import compile_protocol, resolve_positions
from causalab.protocol.positions.resolve import positions_key

from tests.protocol._docs import base_doc

pytestmark = pytest.mark.unit


def _with_ledger() -> dict[str, Any]:
    raw = base_doc()
    raw["method"]["save"].append(
        {"kind": "location_ledger", "file_path": "ledger.json"}
    )
    return raw


def _ledgers(compiled: CompiledProtocol) -> list[list[dict[str, Any]]]:
    assert compiled.positions is not None
    records = []
    for document in compiled.representatives:
        ledger = compiled.positions[positions_key(document)].ledger
        assert ledger is not None
        records.append(ledger.records())
    return records


@settings(max_examples=15, deadline=None)
@given(start=st.integers(0, 4), width=st.integers(2, 4))
def test_swept_band_ledgers_equal_standalone_ledgers(
    env: ResolutionEnv, start: int, width: int
) -> None:
    bands = [
        list(range(start, start + width)),
        list(range(start + width, start + 2 * width)),
    ]
    raw = _with_ledger()
    raw["method"]["sites"]["tgt"]["layers"] = {"sweep": bands}
    campaign = resolve_positions(compile_protocol(raw, env=env), env=env)
    for band, records in zip(bands, _ledgers(campaign), strict=True):
        expected_labels = {"reads.logits.pos"} | {
            f"{kind}.{name}[layers={layer}].pos"
            for layer in band
            for kind, name in (("reads", "v_cf"), ("writes", "patch"))
        }
        assert {row["constituent"] for row in records} == expected_labels
        standalone = copy.deepcopy(raw)
        standalone["method"]["sites"]["tgt"]["layers"] = band
        resolved = resolve_positions(compile_protocol(standalone, env=env), env=env)
        assert records == _ledgers(resolved)[0]


def test_read_models_contribute_to_the_ledger_cache_key(env: ResolutionEnv) -> None:
    raw = _with_ledger()
    raw["method"]["reads"]["patched_sink"] = copy.deepcopy(
        raw["method"]["reads"]["logits"]
    )
    models = raw["method"]["intervened_models"]
    models["patched"]["reads"].append("patched_sink")
    raw["method"]["save"].append(
        {"read": "patched_sink", "model": "patched", "file_path": "sink.safetensors"}
    )
    patched = compile_protocol(raw, env=env).representatives[0]
    # the same address taken on the un-intervened model instead (§2.9)
    models["patched"]["reads"].remove("logits")
    models["original_base"] = {"input": "base", "reads": ["logits"]}
    raw["method"]["save"][0]["model"] = "original_base"
    original = compile_protocol(raw, env=env).representatives[0]
    assert positions_key(patched) != positions_key(original)


def test_write_names_contribute_to_the_ledger_cache_key(env: ResolutionEnv) -> None:
    raw = _with_ledger()
    first = compile_protocol(raw, env=env).representatives[0]
    raw["method"]["writes"]["alternate"] = raw["method"]["writes"].pop("patch")
    raw["method"]["intervened_models"]["patched"]["writes"] = ["alternate"]
    second = compile_protocol(raw, env=env).representatives[0]
    assert positions_key(first) != positions_key(second)
