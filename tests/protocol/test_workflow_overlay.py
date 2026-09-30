"""The run-tree/external artifact overlay (workflow spec §3).

Carried over from the v1 adversarial review: shadowing is by **step name**, not
by directory existence. A rerun leftover or a stray directory in the run tree
must not capture a reference meant for the external artifacts root, because the
canonical form stamps every resolved value and the record has to say which store
answered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causalab.io.env import FileArtifacts
from causalab.workflow.runner import OverlayArtifacts

pytestmark = pytest.mark.unit


@pytest.fixture()
def stores(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    outer_root = tmp_path / "outer"
    (run_root / "fit").mkdir(parents=True)
    (run_root / "stray").mkdir()  # a non-step directory in the run tree
    outer_root.mkdir()
    (outer_root / "stray").mkdir()
    (outer_root / "stray" / "values.json").write_text('{"k": "outer"}')
    (run_root / "fit" / "values.json").write_text('{"k": "run"}')
    (run_root / "stray" / "values.json").write_text('{"k": "SHADOWED-WRONGLY"}')
    return run_root, outer_root


def _overlay(stores: tuple[Path, Path]) -> OverlayArtifacts:
    run_root, outer_root = stores
    return OverlayArtifacts(
        run_root=run_root,
        outer=FileArtifacts(root=outer_root),
        step_names=frozenset({"fit"}),
    )


def test_step_names_shadow(stores):
    assert _overlay(stores).read_value("fit", "k") == "run"


def test_directory_existence_does_not_shadow(stores):
    """A run-tree directory that is NOT a step must not capture external refs."""
    assert _overlay(stores).read_value("stray", "k") == "outer"


def test_resolve_path_falls_through(stores):
    run_root, outer_root = stores
    (outer_root / "bundle.safetensors").write_bytes(b"x")
    overlay = _overlay(stores)
    assert (
        overlay.resolve_path("bundle.safetensors") == outer_root / "bundle.safetensors"
    )
    assert (
        overlay.resolve_path("fit/rot.safetensors") == run_root / "fit/rot.safetensors"
    )
