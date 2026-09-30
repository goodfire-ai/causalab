"""The measurement package retains lightweight and file-loaded script entrypoints."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("loader", ["module", "file"])
def test_comparison_entrypoint_runs_after_relocation(loader, tmp_path):
    from causalab.measurement.collection import file_hash
    from causalab.workflow.isolate import _load_main
    from tests.measurement.analysis.test_compare import receipt

    module = importlib.import_module("causalab.measurement.analysis.compare")
    main = module.main if loader == "module" else _load_main(module.__file__)
    before = receipt(tmp_path / "before", [[[1.0], [2.0]]])
    after = receipt(tmp_path / "after", [[[2.0], [3.0]]])
    outputs = {"summary": tmp_path / "summary.json", "report": tmp_path / "report.html"}
    main(
        {
            "before": before,
            "after": after,
            "before_sha256": file_hash(before),
            "after_sha256": file_hash(after),
            "bootstrap_draws": 100,
        },
        outputs,
    )
    assert all(path.is_file() for path in outputs.values())
    assert "Before/after measurements" in outputs["report"].read_text()


def test_measurement_analysis_imports_are_torch_free():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import causalab.measurement.analysis.compare; "
            "import causalab.measurement.analysis.reports; "
            "import sys; assert 'torch' not in sys.modules; "
            "assert 'numpy' not in sys.modules",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_analysis_identity_names_relocated_files():
    from causalab.measurement.analysis.reports import analysis_identity

    files = analysis_identity()["files_sha256"]
    assert set(files) == {
        "measurement/analysis/compare.py",
        "measurement/analysis/stability.py",
        "measurement/analysis/summary.py",
        "measurement/analysis/profiles.py",
        "measurement/analysis/training.py",
        "measurement/analysis/reports.py",
    }
    assert all(len(digest) == 64 for digest in files.values())
