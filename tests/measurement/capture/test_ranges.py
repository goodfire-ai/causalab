"""Native phase annotations observe real calls without replacing their functions."""

import sys
from types import SimpleNamespace

import pytest
import torch

from causalab.measurement.capture.ranges import phase_ranges

pytestmark = pytest.mark.unit


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    @property
    def device(self):
        return self.layer.weight.device

    def forward(self, input_ids):
        return self.layer(input_ids.float())


class Engine:
    def __init__(self, model):
        self.model = model
        self.fail = False

    def execute(self):
        if self.fail:
            raise RuntimeError("execution failed")
        return self.model(torch.ones(1, 2, dtype=torch.long))


@pytest.mark.parametrize("fail", [False, True])
def test_phase_observer_preserves_functions_and_restores_thread_state(fail):
    model = Model()
    engine = Engine(model)
    expected = engine.execute().detach()
    execute = engine.execute.__func__
    backward = torch.Tensor.backward
    previous = sys.getprofile()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Torch installs its own step wrapper on first optimizer construction.
    step = torch.optim.SGD.step
    engine.fail = fail

    def run():
        with phase_ranges(
            engine,
            bundles={"test": SimpleNamespace(model=model)},
            operation_step="linear",
        ) as coverage:
            assert engine.execute.__func__ is execute
            assert torch.Tensor.backward is backward
            assert torch.optim.SGD.step is step
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profiler:
                output = engine.execute()
                assert torch.equal(output, expected)
                output.sum().backward()
                optimizer.step()
        names = {event.key for event in profiler.key_averages()}
        assert {"step:linear", "backward", "optimizer:SGD"} <= names
        assert coverage["calls"]["backward"] == coverage["calls"]["optimizer:SGD"] == 1

    if fail:
        with pytest.raises(RuntimeError, match="execution failed"):
            run()
    else:
        run()
    assert sys.getprofile() is previous
    assert engine.execute.__func__ is execute
    assert torch.Tensor.backward is backward
    assert torch.optim.SGD.step is step
