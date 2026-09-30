"""``check_caller_bundle`` compares devices by value, not by spelling.

The engine's ``--device`` string and the bundle's [`DeviceMap`][causalab.neural.shared.devices.DeviceMap] are
both placements; ``cuda`` and ``cuda:<current>`` name the same one, so a
caller-owned bundle built with one spelling is not refused against an engine
built with the other. A real disagreement still refuses, naming both sides
in the user's own words (the spelling each side was given).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.neural.shared.devices import DeviceMap
from causalab.protocol.rules.capability import check_caller_bundle
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry

pytestmark = pytest.mark.unit

REALIZATION: dict[str, Any] = {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "main",
    "dtype": "fp32",
    "quantization": None,
}


def _bundle(devices: DeviceMap, geometry: ParallelGeometry = ONE) -> Any:
    """The fields the check reads, and nothing else — no model is loaded."""
    return SimpleNamespace(
        key=REALIZATION["key"],
        revision="main",
        dtype="fp32",
        quantization=None,
        devices=devices,
        geometry=geometry,
        blocks=[None] * len(devices.blocks),
        model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager")),
    )


def test_cuda_and_its_current_ordinal_agree() -> None:
    """With CUDA the current ordinal is the real one; without it the
    normalisation still pins ``cuda`` to ordinal 0, so the comparison is a
    comparison of values on any machine."""
    current = torch.cuda.current_device() if torch.cuda.is_available() else 0
    check_caller_bundle(
        _bundle(DeviceMap.parse(f"cuda:{current}", 2)), REALIZATION, device="cuda"
    )
    check_caller_bundle(
        _bundle(DeviceMap.parse("cuda", 2)), REALIZATION, device=f"cuda:{current}"
    )


def test_the_same_list_spelled_with_spaces_agrees() -> None:
    check_caller_bundle(
        _bundle(DeviceMap.parse("cuda:0,cuda:1", 2)),
        REALIZATION,
        device="cuda:0, cuda:1",
    )


def test_a_different_placement_is_refused_naming_both_spellings() -> None:
    with pytest.raises(ProtocolError) as err:
        check_caller_bundle(
            _bundle(DeviceMap.parse("cpu", 2)), REALIZATION, device="cuda:1"
        )
    message = str(err.value)
    assert err.value.code == "P4"
    assert "device" in message and "'cuda:1'" in message and "'cpu'" in message


def test_a_different_split_of_the_same_devices_is_refused() -> None:
    """The engine would run the document on the layout it was asked for; a
    bundle placed another way is a different run, and the receipt would
    misdescribe it."""
    a_first = DeviceMap(
        embedding=torch.device("cuda", 0),
        blocks=(
            torch.device("cuda", 0),
            torch.device("cuda", 0),
            torch.device("cuda", 1),
        ),
        head=torch.device("cuda", 1),
        requested="cuda:0,cuda:1",
    )
    with pytest.raises(ProtocolError, match="device"):
        check_caller_bundle(_bundle(a_first), REALIZATION, device="cuda:0,cuda:1")
    # the twin: the even split the engine would make
    check_caller_bundle(
        _bundle(DeviceMap.parse("cuda:0,cuda:1", 3)),
        REALIZATION,
        device="cuda:0,cuda:1",
    )


def test_a_bundle_loaded_under_another_geometry_is_refused_naming_both() -> None:
    """``docs/model_parallelism.md`` §3: a caller-owned bundle at ``world >
    1`` was either loaded sharded under this very geometry or it does not
    realize the run — the receipt's ``execution.parallel`` would otherwise
    describe shards that do not exist."""
    with pytest.raises(ProtocolError) as err:
        check_caller_bundle(
            _bundle(DeviceMap.parse("cpu", 2)),
            REALIZATION,
            device="cpu",
            geometry=ParallelGeometry(tensor=2),
        )
    message = str(err.value)
    assert err.value.code == "P4"
    assert "tp=2" in message and "tp=1" in message
    # the twin: a bundle the same rank loaded under the geometry passes
    check_caller_bundle(
        _bundle(DeviceMap.parse("cpu", 2), ParallelGeometry(tensor=2)),
        REALIZATION,
        device="cpu",
        geometry=ParallelGeometry(tensor=2),
    )


def test_the_default_geometry_is_world_one() -> None:
    check_caller_bundle(_bundle(DeviceMap.parse("cpu", 2)), REALIZATION, device="cpu")
    with pytest.raises(ProtocolError, match="tp=2"):
        check_caller_bundle(
            _bundle(DeviceMap.parse("cpu", 2), ParallelGeometry(tensor=2)),
            REALIZATION,
            device="cpu",
        )
