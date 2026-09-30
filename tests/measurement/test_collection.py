"""Measurement boundaries, fresh scientific state, and usable CPU traces."""

from contextlib import contextmanager
import json
from typing import Any
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from causalab.measurement import Operation, collect
from causalab.measurement.collection import file_hash
from causalab.neural.shared.featurizers import Subspace

pytestmark = pytest.mark.unit


@contextmanager
def subspace(seed, directory):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(2, 8, generator=generator)
    base = torch.randn(2, 8, generator=generator)
    stage = Subspace(8, 2, "cayley", seed=seed)

    def run():
        with torch.no_grad():
            feature, _ = stage.featurize(source)
            _, error = stage.featurize(base)
            return stage.inverse(feature, error)

    yield Operation(
        run,
        lambda result: {f"example_{i}/residual": row for i, row in enumerate(result)},
    )


def kwargs() -> dict[str, Any]:
    return dict(
        case="subspace_apply",
        input_identity="synthetic_seeded_2x8_k2",
        scope="prepared subspace interchange",
        reset_policy="fresh stage and inputs",
        seeds=[0, 1],
        repeats=2,
        warmups=1,
    )


def test_real_operation_and_openable_trace(tmp_path):
    path = collect(subspace, tmp_path / "run", **kwargs(), profile=True)
    record = json.loads(path.read_text())
    assert record["status"] == "completed"
    assert len(record["samples"]) == 4
    assert len(record["warmups"]) == 1
    assert record["provenance"]["implementation"]["tree_digest"]
    assert record["provenance"]["dispatch"]["status"] == "unknown"
    one, two = [
        load_file(str(path.parent / s["observations"]["file"]))
        for s in record["samples"][:2]
    ]
    assert all(torch.equal(one[k], two[k]) for k in one)
    trace = record["trace"]
    assert trace["status"] == "completed"
    assert file_hash(path.parent / trace["file"]) == trace["sha256"]
    events = json.loads((path.parent / trace["file"]).read_text())["traceEvents"]
    assert any(e.get("name") == "case:subspace_apply" for e in events)
    assert any(e.get("cat") == "cpu_op" for e in events)
    with pytest.raises(FileExistsError):
        collect(subspace, path.parent, **kwargs())


def test_disabled_profile_retains_timing_and_numerics(tmp_path, monkeypatch):
    def refuse_profile(*args, **kwargs):
        raise AssertionError("native profiling was disabled")

    monkeypatch.setattr(torch.profiler, "profile", refuse_profile)
    path = collect(subspace, tmp_path / "run", **kwargs(), profile=False)
    record = json.loads(path.read_text())
    assert record["status"] == "completed"
    assert record["trace"] == {"status": "not_requested"}
    assert len(record["samples"]) == 4
    assert all(s["seconds"] > 0 for s in record["samples"])
    assert all(
        load_file(str(path.parent / s["observations"]["file"]))
        for s in record["samples"]
    )
    assert not (path.parent / "profile").exists()
    assert not (path.parent / "trace.json").exists()


def test_clock_excludes_preparation_observation_and_cleanup(tmp_path, monkeypatch):
    clock = [0]
    calls = []

    @contextmanager
    def prepare(seed, directory):
        calls.append((seed, directory.name))
        clock[0] += 100_000_000
        state = [0]

        def run():
            assert state == [0], "every pass must start at the same scientific state"
            state[0] += 1
            clock[0] += 7_000_000
            return torch.tensor([float(seed)])

        def observe(result):
            clock[0] += 500_000_000
            return {"logical/observation": result}

        yield Operation(run, observe)
        clock[0] += 1_000_000_000

    monkeypatch.setattr(
        "causalab.measurement.collection.time.perf_counter_ns", lambda: clock[0]
    )
    path = collect(prepare, tmp_path / "run", **kwargs())
    record = json.loads(path.read_text())
    assert [s["seconds"] for s in record["samples"]] == [0.007] * 4
    assert len(calls) == 9  # one warmup and two independently reset passes per sample
    assert calls[0][1] == "warmup_0"


def test_failed_capture_keeps_valid_timing_samples(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("profiler unavailable")

    monkeypatch.setattr(torch.profiler, "profile", broken)
    path = collect(subspace, tmp_path / "run", **kwargs(), profile=True)
    record = json.loads(path.read_text())
    assert record["status"] == "completed"
    assert record["trace"]["status"] == "failed"
    assert len(record["samples"]) == 4


def test_numerical_observer_never_enters_the_clean_timing_pass(tmp_path, monkeypatch):
    clock = [0]
    active = [False]
    seen = []

    @contextmanager
    def observe_state():
        active[0] = True
        clock[0] += 3_000_000_000
        try:
            yield {"captured": True}
        finally:
            active[0] = False

    @contextmanager
    def prepare(seed, directory):
        def run():
            seen.append((directory.name, active[0]))
            clock[0] += 7_000_000
            return torch.ones(1)

        yield Operation(
            run, lambda result: {"value": result}, numerics_context=observe_state
        )

    monkeypatch.setattr(
        "causalab.measurement.collection.time.perf_counter_ns", lambda: clock[0]
    )
    receipt = collect(prepare, tmp_path / "run", **kwargs())
    record = json.loads(receipt.read_text())
    assert all(value == name.endswith("_numerics") for name, value in seen)
    assert [sample["seconds"] for sample in record["samples"]] == [0.007] * 4
    assert all(
        sample["numerics_context"] == {"captured": True} for sample in record["samples"]
    )
    assert active == [False]


def test_cuda_timing_synchronizes_on_both_sides(tmp_path, monkeypatch):
    # CUDA calls and the clock are system boundaries; the operation itself runs.
    events = []
    ticks = iter([0, 1_000_000, 2_000_000, 3_000_000])

    def tick():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr("causalab.measurement.collection.time.perf_counter_ns", tick)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test GPU boundary", total_memory=1024, uuid="test"
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: events.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda _: events.append("reset_peak")
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 16)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 32)

    @contextmanager
    def prepare(seed, directory):
        events.append("prepare")

        def run():
            events.append("run")
            return torch.ones(1)

        yield Operation(run, lambda x: {"logical/value": x})
        events.append("cleanup")

    config = kwargs()
    config.update(seeds=[0], repeats=1, warmups=0, device="cuda:0")
    path = collect(prepare, tmp_path / "run", **config)
    assert (
        events
        == ["prepare", "sync", "reset_peak", "clock", "run", "sync", "clock", "cleanup"]
        * 2
    )
    sample = json.loads(path.read_text())["samples"][0]
    assert sample["seconds"] == 0.001
    assert sample["peak_memory"] == {"allocated_bytes": 16, "reserved_bytes": 32}


def test_failed_numerics_does_not_publish_a_complete_sample(tmp_path):
    @contextmanager
    def prepare(seed, directory):
        yield Operation(lambda: torch.ones(1), lambda _: {})

    with pytest.raises(ValueError, match="logical"):
        collect(prepare, tmp_path / "run", **kwargs())
    record = json.loads((tmp_path / "run/measurement.json").read_text())
    assert record["status"] == "failed"
    assert record["samples"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("repeats", True),
        ("repeats", 0),
        ("warmups", -1),
        ("seeds", [0, 0]),
        ("seeds", [False]),
        ("profile", "yes"),
    ],
)
def test_invalid_plan_refused_before_preparation(tmp_path, field, value):
    config = {**kwargs(), field: value}
    with pytest.raises(ValueError):
        collect(subspace, tmp_path / "run", **config)
    assert not (tmp_path / "run").exists()
