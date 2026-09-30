"""Resume identities cover nested controller code and reusable execution helpers."""

from pathlib import Path
import tempfile

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.paths import ControllerSourceError, source_identity

pytestmark = pytest.mark.unit


@given(
    area=st.sampled_from(["measurement", "profiling", "remote"]),
    depth=st.integers(min_value=0, max_value=5),
    content=st.binary(min_size=1),
)
def test_identity_changes_for_every_nested_execution_source(area, depth, content):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _source_packages(root)
        relative = Path(
            "causalab", area, *[f"nested{i}" for i in range(depth)], "worker.py"
        )
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        before = source_identity(root)
        assert set(before) == {relative.as_posix()}
        source.write_bytes(content + b"\n# changed")
        assert source_identity(root) != before
        source.unlink()
        assert source_identity(root) == {}


def test_identity_distinguishes_same_named_modules_and_ignores_artifacts(tmp_path):
    _source_packages(tmp_path)
    for area in ("runtime", "capture"):
        package = tmp_path / "causalab/measurement" / area
        package.mkdir(parents=True)
        (package / "worker.py").write_text(area)
        (package / "trace.json").write_text("{}")
    identity = source_identity(tmp_path)
    assert len(identity) == 2
    assert len(set(identity.values())) == 2


@pytest.mark.parametrize(
    "relative",
    [
        "measurement/analysis/summary.py",
        "measurement/analysis/reports.py",
        "measurement/deployment/remote.py",
        "measurement/deployment/source_pins.py",
    ],
)
def test_presentation_and_launch_source_edits_do_not_block_resume(tmp_path, relative):
    _source_packages(tmp_path)
    source = tmp_path / "causalab" / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("original")
    before = source_identity(tmp_path)
    source.write_text("changed")
    assert source_identity(tmp_path) == before
    assert source.relative_to(tmp_path).as_posix() not in before


@pytest.mark.parametrize(
    "relative",
    [
        "measurement/runtime/worker.py",
        "measurement/analysis/compare.py",
        "measurement/analysis/profiles.py",
        "measurement/analysis/stability.py",
        "measurement/deployment/installation.py",
        "measurement/deployment/transfer.py",
    ],
)
def test_execution_and_validation_source_edits_still_block_resume(tmp_path, relative):
    _source_packages(tmp_path)
    source = tmp_path / "causalab" / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("original")
    before = source_identity(tmp_path)
    source.write_text("changed")
    assert source_identity(tmp_path) != before


@pytest.mark.parametrize("area", ["measurement", "profiling", "remote"])
def test_missing_required_source_package_is_refused(tmp_path, area):
    _source_packages(tmp_path)
    package = tmp_path / "causalab" / area
    package.rmdir()
    with pytest.raises(ControllerSourceError, match="required source directory"):
        source_identity(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "causalab",
        "causalab/measurement",
        "causalab/measurement/runtime",
        "causalab/profiling/nested",
        "causalab/remote/nested",
    ],
)
def test_symlinked_source_directories_are_refused(tmp_path, relative):
    _source_packages(tmp_path)
    linked = tmp_path / relative
    target = tmp_path / "elsewhere"
    if linked.exists():
        linked.rename(target)
    else:
        target.mkdir()
    (target / "worker.py").write_text("source")
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ControllerSourceError, match="symlink"):
        source_identity(tmp_path)


@pytest.mark.parametrize("relative", ["runtime/worker.py", "analysis/reports.py"])
@pytest.mark.parametrize("broken", [False, True])
def test_symlinked_python_sources_are_refused_even_if_excluded(
    tmp_path, relative, broken
):
    _source_packages(tmp_path)
    target = tmp_path / "elsewhere.py"
    if not broken:
        target.write_text("source")
    linked = tmp_path / "causalab/measurement" / relative
    linked.parent.mkdir(parents=True)
    linked.symlink_to(target)
    with pytest.raises(ControllerSourceError, match="symlink"):
        source_identity(tmp_path)


def _source_packages(root: Path) -> None:
    for area in ("measurement", "profiling", "remote"):
        (root / "causalab" / area).mkdir(parents=True, exist_ok=True)
