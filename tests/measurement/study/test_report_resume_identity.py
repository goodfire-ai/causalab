"""Report statistics can change without weakening resumed artifact validation."""

from pathlib import Path
import tempfile

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.paths import source_identity
from tests.measurement.study.test_source_identity import _source_packages

pytestmark = pytest.mark.unit


@given(content=st.binary(min_size=1, max_size=100))
def test_statistics_edits_preserve_resume_identity_but_validator_edits_do_not(content):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _source_packages(root)
        analysis = root / "causalab/measurement/analysis"
        analysis.mkdir()
        statistics = analysis / "statistics.py"
        validator = analysis / "receipts.py"
        statistics.write_bytes(content)
        validator.write_bytes(content)
        before = source_identity(root)

        statistics.write_bytes(content + b"\n# report edit")
        assert source_identity(root) == before
        assert statistics.relative_to(root).as_posix() not in before

        validator.write_bytes(content + b"\n# validation edit")
        assert source_identity(root) != before
