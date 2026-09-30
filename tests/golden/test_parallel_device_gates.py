"""Real-model golden cases run whenever their geometry fits the visible GPUs."""

from pathlib import Path
import runpy

import pytest
import torch

from causalab.protocol.parallel import parse_geometry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("devices", range(9))
def test_each_real_model_case_requires_only_its_own_devices(
    monkeypatch: pytest.MonkeyPatch, devices: int
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
    suite = runpy.run_path(str(Path(__file__).with_name("test_parallel_world4.py")))
    module_marks = suite["pytestmark"]
    for name, function in suite.items():
        if not name.startswith("test_"):
            continue
        marks = [*module_marks, *getattr(function, "pytestmark", [])]
        parameters = [mark for mark in marks if mark.name == "parametrize"]
        if parameters:
            for case in parameters[0].args[1]:
                skipped = any(
                    mark.args[0]
                    for mark in [*marks, *case.marks]
                    if mark.name == "skipif"
                )
                needed = parse_geometry(case.values[0]).world
                assert skipped == (devices < needed), (name, case.id, devices)
        elif name == "test_the_four_stages_together_hold_the_whole_model":
            skipped = any(mark.args[0] for mark in marks if mark.name == "skipif")
            assert skipped == (devices < 4), (name, devices)
