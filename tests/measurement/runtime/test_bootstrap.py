"""Standalone startup keeps current helpers separate from historical arm imports."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("capture", [False, True])
def test_bootstrap_uses_controller_helpers_and_historical_engine(tmp_path, capture):
    arm = tmp_path / "arm"
    package = arm / "causalab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("identity = 'historical arm'\n")
    repository = Path(__file__).resolve().parents[3]
    config = {"controller_root": str(repository), "package_root": str(arm)}
    if capture:
        config["capture"] = {}
    ticket = tmp_path / "config.json"
    ticket.write_text(json.dumps(config))
    script = """
import importlib
import runpy
import sys

load = importlib.import_module

def select(name):
    worker = load(name)
    assert worker.file_hash.__module__ == '_measurement_runtime.measurement.collection'
    from causalab import identity
    assert identity == 'historical arm'
    observations = load('_measurement_runtime.measurement.runtime.observations')
    assert observations.require_sigmoid_gate.__module__ == '_measurement_runtime.measurement.analysis.stability'
    def serve(config):
        assert config['package_root'] == sys.argv[3]
        print('ready')
    worker.serve = serve
    return worker

importlib.import_module = select
bootstrap, ticket, arm = sys.argv[1:]
sys.argv = [bootstrap, ticket, '', arm]
runpy.run_path(bootstrap, run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(repository / "causalab/measurement/runtime/bootstrap.py"),
            str(ticket),
            str(arm),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ready"
