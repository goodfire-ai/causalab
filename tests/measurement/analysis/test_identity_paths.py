"""Analysis provenance is invariant to symlinked checkout import paths."""

from pathlib import Path

import pytest

from causalab.measurement.analysis import (
    compare,
    profiles,
    reports,
    stability,
    summary,
    training,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("imports", ["reports", "dependency", "all"])
def test_analysis_identity_accepts_symlinked_module_paths(
    tmp_path, monkeypatch, imports
):
    expected = reports.analysis_identity()
    checkout = Path(reports.__file__).resolve().parents[3]
    alias = tmp_path / "checkout-link"
    alias.symlink_to(checkout, target_is_directory=True)
    modules = {
        "reports": (reports,),
        "dependency": (compare,),
        "all": (reports, compare, profiles, stability, summary, training),
    }[imports]
    for module in modules:
        relative = Path(module.__file__).resolve().relative_to(checkout)
        monkeypatch.setattr(module, "__file__", str(alias / relative))

    assert reports.analysis_identity() == expected
