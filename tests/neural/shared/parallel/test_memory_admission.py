"""Admission depends on resident weights and reusable memory, not a workload guess."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch
from hypothesis import given, settings, strategies as st

from causalab.neural.shared.parallel.memory import (
    DeviceMemory,
    device_memory,
    preflight,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.parallel_memory import EstimateRule, Placement, memory_check
from causalab.protocol.registry import get_model_info

INFO = get_model_info("Qwen/Qwen3.6-35B-A3B")
GEOMETRY = ParallelGeometry(tensor=2)
TABLE = {"weight": Placement(1024, axis="tensor")}
RESIDENT = 1024  # 1024 bf16 elements split over two ranks.


def _run(reading: DeviceMemory, rule: EstimateRule = EstimateRule()) -> DeviceMemory:
    result = preflight(
        geometry=GEOMETRY,
        rank=0,
        device="cuda:0",
        dtype="bf16",
        table=TABLE,
        info=INFO,
        memory=reading,
        rule=rule,
    )
    assert result is not None
    return result


@pytest.mark.unit
def test_headroom_is_a_structured_warning_without_refusing_the_load(caplog) -> None:
    reading = DeviceMemory(free=RESIDENT, total=2 * RESIDENT)
    with caplog.at_level(logging.WARNING):
        assert _run(reading) is reading
    (record,) = caplog.records
    assert "continuing with limited headroom" in record.message
    assert record.rank == 0 and record.device == "cuda:0"
    assert record.resident_bytes == RESIDENT
    assert record.estimated_bytes > record.available_bytes == RESIDENT


@pytest.mark.unit
def test_exact_estimated_capacity_needs_no_warning(caplog) -> None:
    estimate = EstimateRule().footprint(TABLE, GEOMETRY, 2)[0]
    with caplog.at_level(logging.WARNING):
        _run(DeviceMemory(free=estimate, total=estimate))
    assert not caplog.records


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(
    slack=st.sampled_from((-1, 0, 1, 1024)),
    base=st.floats(min_value=0.0, max_value=100.0),
    copies=st.floats(min_value=1.0, max_value=10.0),
)
def test_only_resident_weights_control_hard_admission(
    slack: int, base: float, copies: float
) -> None:
    available = RESIDENT + slack
    rule = EstimateRule(base=base, copies={"tensor": copies, "expert": 1.0})
    refusal = memory_check(
        geometry=GEOMETRY,
        rank=0,
        device="cuda:0",
        dtype="bf16",
        table=TABLE,
        info=INFO,
        free=available,
        total=2 * RESIDENT,
        rule=rule,
    )
    assert (refusal is None) == (available >= RESIDENT)
    reading = DeviceMemory(free=available, total=2 * RESIDENT)
    if available >= RESIDENT:
        assert _run(reading, rule) is reading
    else:
        with pytest.raises(ProtocolError, match="P4"):
            _run(reading, rule)


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(
    free=st.integers(min_value=0, max_value=2048),
    cached=st.integers(min_value=0, max_value=2048),
    allocated=st.integers(min_value=0, max_value=2048),
)
def test_reusable_cache_and_driver_free_memory_give_the_same_admission(
    free: int, cached: int, allocated: int
) -> None:
    total = free + cached + allocated
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "mem_get_info", return_value=(free, total)),
        patch.object(torch.cuda, "memory_reserved", return_value=cached + allocated),
        patch.object(torch.cuda, "memory_allocated", return_value=allocated),
    ):
        reading = device_memory("cuda:0")
    assert reading is not None
    assert reading.free == free
    assert reading.available == free + cached
    assert reading.cached == cached
    released = DeviceMemory(free=free + cached, total=total)
    for state in (reading, released):
        if free + cached >= RESIDENT:
            assert _run(state) is state
        else:
            with pytest.raises(ProtocolError, match="P4"):
                _run(state)


@pytest.mark.unit
def test_live_allocations_do_not_count_as_reusable_cache() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "mem_get_info", return_value=(RESIDENT - 1, 4096)),
        patch.object(torch.cuda, "memory_reserved", return_value=2048),
        patch.object(torch.cuda, "memory_allocated", return_value=2048),
    ):
        reading = device_memory("cuda:0")
    assert reading is not None
    with pytest.raises(ProtocolError, match="P4"):
        _run(reading)
