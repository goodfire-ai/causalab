"""The residency census and its rule (``residency.py``,
``docs/model_parallelism.md`` §5.3 "one copy"): every parameter of a loaded
model is its own storage holding exactly its elements, a shared storage is
counted once, and the rule names — by parameter — a shard that is a view of
a larger allocation, a parameter resident with fewer elements than read, a
copy the device holds outside the model, a reserved segment the weights did
not reuse, and live tensors no parameter owns **that the load left behind**:
the census is the difference between the live storages after the load and
a [`Census`][causalab.neural.engines.pytorch_hooks.residency.Census] taken before it, so the fixtures and caches alive in a
long test process — or a model loaded earlier in a rank — are never the
load's. An unowned ``to_local()`` cache is a second live copy; a second
allocator warm-up can leave an unused reserved segment.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import residency as residency_module
from causalab.neural.engines.pytorch_hooks.loading import (
    ModelBundle,
    load_report_path,
    write_load_report,
)
from causalab.neural.engines.pytorch_hooks.residency import (
    ALLOCATED_SLACK,
    RESERVED_SLACK,
    UNOWNED_SLACK,
    Census,
    Residency,
    residency_problems,
)
from causalab.neural.engines.pytorch_hooks.shard_read import LoadReport
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.protocol.parallel import ParallelGeometry

# pyright: reportPrivateUsage=false

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")


def _record(residency: Residency, **overrides: Any) -> dict[str, Any]:
    """A load report's residency block with the plan agreeing with it."""
    record = residency.record()
    record["elements_requested"] = dict(residency.elements_resident)
    record.update(overrides)
    return record


class _Tied(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        self.head.weight = self.embed.weight
        self.register_buffer("scale", torch.ones(4))


def test_every_parameter_of_a_loaded_model_is_its_own_storage(
    llama_bundle: ModelBundle,
) -> None:
    model = llama_bundle.model
    residency = Residency.of(model, CPU)
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    assert set(residency.elements_resident) == set(tensors)
    for name, tensor in tensors.items():
        assert residency.elements_resident[name] == tensor.numel()
        assert residency.itemsize_resident[name] == tensor.element_size()
        assert residency.bytes_resident[name] == tensor.numel() * tensor.element_size()
    # off CUDA there are no device counters; without a census no unowned count
    assert residency.device_bytes_allocated is None
    assert residency.device_bytes_reserved is None
    assert residency.bytes_unowned is None
    assert residency_problems(_record(residency)) == []


def test_a_shared_storage_is_counted_once() -> None:
    residency = Residency.of(_Tied(), CPU)
    assert residency.shared == (("embed.weight", "head.weight"),)
    assert residency.bytes_total == 8 * 4 * 4 + 4 * 4
    assert residency_problems(_record(residency)) == []


def test_two_differently_shaped_views_of_one_storage_are_one_copy() -> None:
    """A fused projection a converter splits into parameter views (a
    ``qkv`` as three, a packed ``gate_up`` as two) is one storage holding
    exactly its members' bytes: the census records the group, and the rule
    judges the group as a whole rather than calling every view a leak."""
    whole = torch.zeros(8 + 16)
    module = torch.nn.Module()
    module.a = torch.nn.Parameter(whole[:8].view(2, 4))
    module.b = torch.nn.Parameter(whole[8:].view(4, 4))
    residency = Residency.of(module, CPU)
    assert residency.shared == (("a", "b"),)
    assert residency.bytes_resident == {"a": 24 * 4, "b": 24 * 4}
    assert residency.bytes_total == 24 * 4
    assert residency_problems(_record(residency)) == []


def test_a_shared_storage_larger_than_its_members_is_named_as_a_group() -> None:
    """The clause's real target survives the group rule: two views of a
    storage neither fills — a shard that is a view of the whole allocation
    shared with another shard — is one problem naming the group."""
    whole = torch.zeros(2 * 12)
    module = torch.nn.Module()
    module.a = torch.nn.Parameter(whole[:8].view(2, 4))
    module.b = torch.nn.Parameter(whole[8:12].view(4))
    residency = Residency.of(module, CPU)
    assert residency.shared == (("a", "b"),)
    (problem,) = residency_problems(_record(residency))
    assert problem == (
        "a / b: one storage of 96 bytes backs 48 bytes of parameters — a view of "
        "a larger allocation"
    )


def test_a_shard_that_views_a_larger_allocation_is_named() -> None:
    whole = torch.zeros(2 * 12)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(whole[:12].view(3, 4))
    residency = Residency.of(module, CPU)
    assert residency.elements_resident["weight"] == 12
    assert residency.bytes_resident["weight"] == 2 * 12 * 4
    (problem,) = residency_problems(_record(residency))
    assert problem.startswith("weight: backed by a storage of 96 bytes, not its own 48")


def test_fewer_elements_than_read_and_a_missing_parameter_are_named() -> None:
    residency = Residency.of(_Tied(), CPU)
    record = _record(residency)
    record["elements_requested"] = {"embed.weight": 64, "lost.weight": 7}
    problems = residency_problems(record)
    assert problems == [
        "embed.weight: 32 elements resident, 64 read",
        "lost.weight: read but not resident",
    ]


def test_a_parameter_read_but_not_resident_does_not_end_the_walk() -> None:
    """The plan's order is the report's: a missing parameter ahead of a
    short one, both named."""
    record = _record(Residency.of(_Tied(), CPU))
    record["elements_requested"] = {"lost.weight": 7, "embed.weight": 64}
    assert residency_problems(record) == [
        "lost.weight: read but not resident",
        "embed.weight: 32 elements resident, 64 read",
    ]


def test_the_device_rules_are_silent_exactly_at_their_slack() -> None:
    """The slacks are what a clean load may measure, so a counter sitting
    exactly on one is clean; one byte above is named."""
    residency = Residency.of(_Tied(), CPU)
    total = residency.bytes_total
    assert (
        residency_problems(
            _record(
                residency,
                device_bytes_allocated=total + ALLOCATED_SLACK,
                device_bytes_reserved=total + ALLOCATED_SLACK + RESERVED_SLACK,
                bytes_unowned=UNOWNED_SLACK,
            )
        )
        == []
    )
    problems = residency_problems(
        _record(
            residency,
            device_bytes_allocated=total + ALLOCATED_SLACK + 1,
            bytes_unowned=UNOWNED_SLACK + 1,
        )
    )
    assert len(problems) == 2
    assert problems[0].startswith(
        f"the device holds {total + ALLOCATED_SLACK + 1} bytes"
    )
    assert problems[1].startswith(f"{UNOWNED_SLACK + 1} bytes of live tensors on cpu")


def test_the_census_walks_past_what_it_skips_and_counts_each_storage_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walk over the process's live objects, driven over a list of this
    test's own (the ``gc`` seam): a tensor on another device, a sparse one
    and what is not a tensor are skipped and the walk goes on past them; an
    owned storage is skipped; two views of one unowned storage count it
    once; a second unowned storage counts too."""
    elsewhere = torch.empty(1 << 16, device="meta")
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
    )
    owned = torch.zeros(8)
    held = torch.zeros(16)
    view = held[4:]
    other = torch.zeros(4, dtype=torch.int64)
    live: list[Any] = [elsewhere, sparse, owned, held, view, other, "no tensor", 3]
    monkeypatch.setattr(
        residency_module,
        "gc",
        SimpleNamespace(collect=lambda: 0, get_objects=lambda: list(live)),
    )
    owners = {residency_module._storage_key(owned)}
    assert residency_module._unowned_bytes(CPU, owners, since=None) == 16 * 4 + 4 * 8
    # the census of the same walk: every strided storage on the device once
    census = Census.take(CPU)
    assert census.device == "cpu"
    assert census.storages == {
        residency_module._storage_key(owned): 8 * 4,
        residency_module._storage_key(held): 16 * 4,
        residency_module._storage_key(other): 4 * 8,
    }


def test_a_held_copy_shows_in_the_census() -> None:
    baseline = Census.take(CPU)
    module = _Tied()
    before = Residency.of(module, CPU, since=baseline)
    assert before.bytes_unowned is not None
    # the shape of a cache of materialised locals kept beside the parameters
    module.cache = {  # type: ignore[attr-defined]
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    after = Residency.of(module, CPU, since=baseline)
    assert after.bytes_unowned is not None
    assert after.bytes_unowned - before.bytes_unowned == 8 * 4 * 4
    # the parameters themselves are unchanged: one storage each, own elements
    assert after.bytes_resident == before.bytes_resident
    assert residency_problems(_record(after, bytes_unowned=UNOWNED_SLACK + 1)) == [
        f"{UNOWNED_SLACK + 1} bytes of live tensors on cpu that no parameter owns "
        f"(slack {UNOWNED_SLACK})"
    ]


def test_the_device_rules_name_a_copy_outside_the_model_and_an_unreused_segment() -> (
    None
):
    residency = Residency.of(_Tied(), CPU)
    total = residency.bytes_total
    held = total + ALLOCATED_SLACK + 1
    problems = residency_problems(
        _record(
            residency,
            device_bytes_allocated=held,
            device_bytes_reserved=held + RESERVED_SLACK + 1,
        )
    )
    assert len(problems) == 2
    assert problems[0].startswith(f"the device holds {held} bytes against {total}")
    assert problems[1].startswith(
        f"the allocator reserves {held + RESERVED_SLACK + 1} bytes against {held}"
    )
    # within the slack: the counters are what a clean load measures
    assert (
        residency_problems(
            _record(
                residency,
                device_bytes_allocated=total,
                device_bytes_reserved=total + RESERVED_SLACK,
            )
        )
        == []
    )


def test_tensors_alive_before_the_load_are_not_the_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long process holds tensors of its own — fixtures, caches, a model
    loaded earlier: whatever is alive when the census is taken before the
    load is never counted against the load, however large."""
    resident = torch.zeros(4096)  # 16 KiB no parameter owns, alive throughout
    module = _Tied()
    live: list[Any] = [resident]
    monkeypatch.setattr(
        residency_module,
        "gc",
        SimpleNamespace(collect=lambda: 0, get_objects=lambda: list(live)),
    )
    baseline = Census.take(CPU)
    assert baseline.storages == {residency_module._storage_key(resident): 4096 * 4}
    live.extend(module.parameters())
    residency = Residency.of(module, CPU, since=baseline)
    assert residency.bytes_unowned == 0
    assert residency_problems(_record(residency)) == []
    # with no baseline, the same tensor is the process's unowned bytes
    assert Residency.of(module, CPU, since=Census(CPU, {})).bytes_unowned == 4096 * 4


def test_tensors_appearing_during_the_load_are_named_with_the_same_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage that is alive after the load and was not before, owned by
    no parameter, is the load's: one byte above the slack is the problem,
    worded as before."""
    module = _Tied()
    live: list[Any] = list(module.parameters())
    monkeypatch.setattr(
        residency_module,
        "gc",
        SimpleNamespace(collect=lambda: 0, get_objects=lambda: list(live)),
    )
    baseline = Census.take(CPU)
    copy = torch.empty(UNOWNED_SLACK // 4 + 1)  # UNOWNED_SLACK + 4 bytes
    live.append(copy)
    residency = Residency.of(module, CPU, since=baseline)
    assert residency.bytes_unowned == UNOWNED_SLACK + 4
    assert residency_problems(_record(residency)) == [
        f"{UNOWNED_SLACK + 4} bytes of live tensors on cpu that no parameter owns "
        f"(slack {UNOWNED_SLACK})"
    ]
    # exactly at the slack the rule is silent: a view over the same storage
    # adds nothing, and a storage the baseline held at the same address but
    # another size is a new allocation there, counted
    live.append(copy[1:])
    assert Residency.of(module, CPU, since=baseline).bytes_unowned == UNOWNED_SLACK + 4
    reused = Census(CPU, {residency_module._storage_key(copy): 64})
    owners = {residency_module._storage_key(p) for p in module.parameters()}
    assert (
        residency_module._unowned_bytes(CPU, owners, since=reused) == UNOWNED_SLACK + 4
    )


def test_a_census_on_another_device_type_is_refused_as_a_baseline() -> None:
    """The baseline and the census are of one device: a ``meta`` census
    cannot stand in for the CPU's."""
    module = _Tied()
    with pytest.raises(ValueError, match="census of meta"):
        Residency.of(module, CPU, since=Census(torch.device("meta"), {}))


def test_the_written_report_carries_the_block_the_rule_reads(tmp_path: Path) -> None:
    module = _Tied()
    residency = Residency.of(module, CPU, since=Census.take(CPU))
    report = LoadReport(
        bytes_requested={"embed.weight": 64, "scale": 8},
        bytes_on_disk={"embed.weight": 64, "scale": 8},
        elements_requested={"embed.weight": 32, "scale": 4},
    )
    sharding = Sharding(ParallelGeometry(), 0, meshes={})
    target = write_load_report(
        report, sharding, tmp_path, key="k", revision="main", residency=residency
    )
    assert target == load_report_path(tmp_path, 0)
    record = json.loads(target.read_text())
    assert record["elements_requested"] == {"embed.weight": 32, "scale": 4}
    assert record["bytes_total"] == residency.bytes_total
    assert record["shared"] == [["embed.weight", "head.weight"]]
    assert record["device_bytes_allocated"] is None
    assert residency_problems(record) == []
