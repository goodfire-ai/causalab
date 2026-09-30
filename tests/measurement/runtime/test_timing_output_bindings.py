"""Saved workflow and operation outputs have verifiable, non-dangling bindings."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, strategies as st

from causalab.measurement.collection import file_hash
from causalab.measurement.runtime.worker import bind_timing_outputs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("step", [None, "fit"])
@pytest.mark.parametrize("empty_directory", [False, True])
def test_no_saved_outputs_omit_workflow_binding(tmp_path, step, empty_directory):
    if empty_directory:
        (tmp_path / "timing/results/fit").mkdir(parents=True)
    sample = {"timing_directory": "timing", "workflow_outputs": "stale/outputs"}
    bind_timing_outputs(
        sample, tmp_path, "results", {}, step=step, observation_policy="not_requested"
    )
    assert "workflow_outputs" not in sample
    assert sample["output_files"] == {}
    assert sample["observer_check"] == {"status": "not_requested"}


@given(st.lists(st.binary(max_size=32), min_size=1, max_size=4), st.booleans())
def test_saved_operation_and_workflow_outputs_are_attested(contents, operation):
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        root = directory / "timing/results/fit"
        root.mkdir(parents=True)
        expected = {}
        for index, content in enumerate(contents):
            path = root / f"artifact{index}.bin"
            path.write_bytes(content)
            expected[path.relative_to(directory).as_posix()] = file_hash(path)
        diagnostic = directory / "numerics/results/fit"
        diagnostic.mkdir(parents=True)
        (diagnostic / "excluded.bin").write_bytes(b"diagnostic only")
        sample = {"timing_directory": "timing"}
        bind_timing_outputs(
            sample,
            directory,
            "results",
            {},
            step="fit" if operation else None,
            observation_policy="not_requested",
        )
        assert sample["workflow_outputs"] == "timing/results"
        assert (directory / sample["workflow_outputs"]).is_dir()
        assert sample["output_files"] == expected
