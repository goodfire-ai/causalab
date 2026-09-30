"""The multi-rank CUDA graph golden's cases hold what its claim needs, on CPU:
every replica's shard of the sweep is a whole cohort, and each case runs
whenever its world fits the visible GPUs."""

from pathlib import Path
import runpy

import pytest
import torch

from causalab.protocol.parallel import parse_geometry
from causalab.protocol.publish import point_shards
from tests.golden import test_multirank_cuda_graphs as golden

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("geometry", sorted(golden.CASES))
def test_every_replica_holds_a_cohort_of_whole_members(geometry: str) -> None:
    layers = golden.sweep(geometry)
    assert len(set(layers)) == len(layers)
    shards = point_shards(range(len(layers)), golden.point_replicas(geometry))
    assert [len(shard) for shard in shards] == [golden.MEMBERS] * len(shards)
    # the cohort slot holds every member's minibatch in one window, and a
    # cohort graph needs two members (graph_cohort.cohort_graph_reason)
    assert golden.MEMBERS >= 2
    assert golden.MEMBERS * golden.PAIRS <= int(golden.FIT_ROWS)


def test_each_geometry_sweeps_two_layers_per_point_replica() -> None:
    assert golden.sweep("tp=2") == golden.sweep("ep=2") == [12, 18]
    assert golden.sweep("dp=2") == golden.sweep("tp=2,dp=2") == [12, 18, 24, 30]
    assert golden.sweep("dp=2:rows") == [12, 18]


@pytest.mark.parametrize("devices", range(9))
def test_each_case_requires_only_its_own_devices(
    monkeypatch: pytest.MonkeyPatch, devices: int
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    suite = runpy.run_path(str(Path(golden.__file__)))
    module_marks = suite["pytestmark"]
    for name, function in suite.items():
        if not name.startswith("test_"):
            continue
        marks = [*module_marks, *getattr(function, "pytestmark", [])]
        (parameters,) = [mark for mark in marks if mark.name == "parametrize"]
        cases = parameters.args[1]
        assert sorted(case.values[0] for case in cases) == sorted(golden.CASES)
        for case in cases:
            skipped = any(
                mark.args[0] for mark in [*marks, *case.marks] if mark.name == "skipif"
            )
            needed = parse_geometry(case.values[0]).world
            assert skipped == (devices < needed), (name, case.id, devices)
