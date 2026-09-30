"""The memory pre-flight's torch half (``neural/shared/parallel/memory.py``;
``docs/model_parallelism.md`` §2, §3, §11): the device reading and the
refusal, on the CPU with the reading handed in or the query stubbed.

``unit`` throughout: off CUDA the reading is ``None`` and the check is the
identity; a CUDA word with no CUDA, or a query that raises, is ``None`` too
(the rule never refuses a run it cannot measure); a reading below the
resident weights is the ``P4`` on ``--parallel`` naming the rank, the device, the
bytes and the fits; a reading above it passes and is returned.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.shared.parallel import memory as memory_module
from causalab.neural.shared.parallel.memory import (
    DeviceMemory,
    device_memory,
    preflight,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.parallel_memory import RULE, Placement, estimate_resident
from causalab.protocol.registry import get_model_info

pytestmark = pytest.mark.unit

A3B = "Qwen/Qwen3.6-35B-A3B"


def _table() -> dict[str, Placement]:
    return {
        "model.embed_tokens.weight": Placement(1 << 20, home="first"),
        "model.layers.0.self_attn.q_proj.weight": Placement(
            1 << 22, axis="tensor", home="layer", layer=0
        ),
        "model.layers.1.self_attn.q_proj.weight": Placement(
            1 << 22, axis="tensor", home="layer", layer=1
        ),
        "lm_head.weight": Placement(1 << 20, home="last"),
    }


def test_off_cuda_the_reading_is_none_and_the_check_is_the_identity() -> None:
    assert device_memory("cpu") is None
    assert device_memory(" cpu") is None
    assert (
        preflight(
            geometry=ParallelGeometry(tensor=2),
            rank=0,
            device="cpu",
            dtype="fp32",
            table=_table(),
            info=get_model_info(A3B),
        )
        is None
    )


def test_a_cuda_word_without_cuda_or_with_a_failing_query_is_none(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_memory("cuda") is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def failing(device: object) -> tuple[int, int]:
        raise RuntimeError("no context")

    monkeypatch.setattr(torch.cuda, "mem_get_info", failing)
    assert device_memory("cuda:1") is None


def test_a_query_that_answers_is_the_reading(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (7, 11))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    assert device_memory("cuda:0") == DeviceMemory(free=7, total=11)


def test_a_reading_below_resident_weights_is_the_p4_on_parallel_naming_everything() -> (
    None
):
    table = _table()
    geometry = ParallelGeometry(tensor=2)
    resident = estimate_resident(table, geometry, 4)[1]
    with pytest.raises(ProtocolError) as err:
        preflight(
            geometry=geometry,
            rank=1,
            device="cuda:1",
            dtype="fp32",
            table=table,
            info=get_model_info(A3B),
            memory=DeviceMemory(free=resident - 1, total=resident + 1),
        )
    assert err.value.code == "P4" and err.value.path == "--parallel"
    text = str(err.value)
    assert text.startswith("[P4] at --parallel tp=2 would place")
    assert "--parallel --parallel" not in text
    assert "rank 1 (cuda:1)" in text and "fp32 weights" in text
    assert "geometries estimated to fit here" in text and "(--device cuda:1)" in text


def test_a_reading_that_fits_passes_and_is_returned() -> None:
    table = _table()
    geometry = ParallelGeometry(tensor=2)
    footprint = RULE.footprint(table, geometry, 4)[0]
    reading = DeviceMemory(free=footprint, total=footprint + 1)
    assert (
        preflight(
            geometry=geometry,
            rank=0,
            device="cuda:0",
            dtype="fp32",
            table=table,
            info=get_model_info(A3B),
            memory=reading,
        )
        is reading
    )


def test_the_query_is_asked_when_no_reading_is_handed_in(monkeypatch) -> None:
    asked: list[str] = []

    def reading(device: str) -> DeviceMemory | None:
        asked.append(device)
        return DeviceMemory(free=1 << 60, total=1 << 60)

    monkeypatch.setattr(memory_module, "device_memory", reading)
    preflight(
        geometry=ParallelGeometry(tensor=2),
        rank=0,
        device="cuda:0",
        dtype="bf16",
        table=_table(),
        info=get_model_info(A3B),
    )
    assert asked == ["cuda:0"]
