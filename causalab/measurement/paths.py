"""Controller locations and recursive source identities for isolated study runs."""

from dataclasses import dataclass
from pathlib import Path

from .collection import file_hash


# Presentation and launch preflight do not affect resumed execution. Include
# all other helpers in the resume identity by default.
_RESUME_EXCLUSIONS = frozenset(
    {
        "causalab/measurement/analysis/summary.py",
        "causalab/measurement/analysis/reports.py",
        "causalab/measurement/analysis/single.py",
        "causalab/measurement/analysis/statistics.py",
        "causalab/measurement/deployment/remote.py",
        "causalab/measurement/deployment/source_pins.py",
    }
)


@dataclass
class ControllerSourceError(ValueError):
    """The controller source tree cannot supply a reliable resume identity."""

    path: Path
    reason: str

    def __str__(self) -> str:
        return f"controller source {self.path}: {self.reason}"


def controller_root() -> Path:
    """Return the source/install root containing the current controller package."""
    return Path(__file__).resolve().parents[2]


def worker_bootstrap(root: Path) -> Path:
    """Locate the stdlib-only entry point without importing the selected arm."""
    return root / "causalab/measurement/runtime/bootstrap.py"


def source_identity(root: Path | None = None) -> dict[str, str]:
    """Attest execution and validation sources used to resume a study.

    Recursively hash sources by root-relative path, excluding presentation and
    launch modules listed above. Reject missing packages and symlinked sources.
    """
    root = controller_root() if root is None else root
    package = root / "causalab"
    if package.is_symlink():
        raise ControllerSourceError(package, "symlinked source directory")
    identity: dict[str, str] = {}
    for area in ("measurement", "profiling", "remote"):
        directory = package / area
        if directory.is_symlink():
            raise ControllerSourceError(directory, "symlinked source directory")
        if not directory.is_dir():
            raise ControllerSourceError(
                directory, "required source directory is missing"
            )
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() and (
                path.suffix == ".py" or path.is_dir() or not path.exists()
            ):
                raise ControllerSourceError(
                    path, "symlinked Python source or directory"
                )
            if path.suffix != ".py" or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in _RESUME_EXCLUSIONS:
                identity[relative] = file_hash(path)
    return identity
