"""Where the demos keep their documents, tables and figures (``docs/demos.md``).

``demos/`` holds three genres, and the suites that read it need to agree on
which is which:

* a **demo** — a directory of markdown beside ``protocols/`` and
  ``workflows/``, checked by ``tests/demos/test_demos.py``;
* ``papers/`` — the replication packages, one page per figure beside shared
  ``protocols/``, ``workflows/`` and ``artifacts/`` folders, with the rules
  ``docs/paper_replications.md`` fixes and the checks in
  ``tests/demos/test_papers.py``;
* ``methods/`` — the method library (the former ``causalab/configs/``): one
  document per method, indexed by one README table rather than a demo per
  document, with a results file pinned to each document's digest
  (``tests/demos/test_methods.py``). ``tests/_helpers/paths.py`` spells its
  paths; the protocol tier's suites (``tests/protocol/test_protocol_presets.py``,
  ``tests/protocol/test_shipped_digests.py``, the workflow censuses) read them.

A demo's tables and figures sit under ``artifacts/`` (the onboarding tutorial:
``artifacts/data/``, ``artifacts/figures/``, ``artifacts/output/``) or, in the
earlier layout, directly beside the markdown (``data/``, ``figures/``).
``docs/demos.md`` §5.6 leaves the output convention open, so both are read
here and nowhere else. ``scripts/repin_demo_digests.py`` and
``scripts/standalone_smoke.py`` carry the same rule: the first is run against
a bare export of ``demos/`` and the second must not import the checkout.

Dataset references resolve against the demo's data root first and the
shipped task tables behind it, as the CLI has it (``causalab/cli.py`` passes
``TASKS_ROOT`` as the fallback). Artifacts resolve against the repository
root: a demo document that loads a fitted featurizer names it by a
repo-relative path.
"""

from __future__ import annotations

from pathlib import Path

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT

REPO = Path(__file__).resolve().parents[2]
DEMOS = REPO / "demos"

#: Directories under ``demos/`` that are not demos (module docstring).
OTHER_GENRES = frozenset({"papers", "methods"})


def demo_dirs() -> list[Path]:
    """Every demo directory, ``OTHER_GENRES`` excluded."""
    return sorted(
        path
        for path in DEMOS.iterdir()
        if path.is_dir() and path.name not in OTHER_GENRES
    )


def demo_protocols() -> list[Path]:
    return sorted(
        path for demo in demo_dirs() for path in demo.glob("protocols/*.json")
    )


def demo_workflows() -> list[Path]:
    return sorted(
        path for demo in demo_dirs() for path in demo.glob("workflows/*.json")
    )


def demo_of(document: Path) -> Path:
    """The demo directory a ``protocols/`` or ``workflows/`` document belongs to."""
    return document.parents[1]


def artifacts_dir(demo: Path) -> Path:
    """``<demo>/artifacts`` when the demo keeps one, else the demo itself."""
    artifacts = demo / "artifacts"
    return artifacts if artifacts.is_dir() else demo


def data_root(demo: Path) -> Path:
    """Where the demo's dataset references resolve: ``artifacts/data`` or ``data``."""
    return artifacts_dir(demo) / "data"


def figures_dir(demo: Path) -> Path:
    return artifacts_dir(demo) / "figures"


def demo_env(document: Path) -> ResolutionEnv:
    """The resolution environment a demo document loads in (module docstring)."""
    return ResolutionEnv(
        datasets=FileDatasets(
            root=data_root(demo_of(document)), fallback_roots=(TASKS_ROOT,)
        ),
        artifacts=FileArtifacts(root=REPO),
    )
