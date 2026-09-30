"""``causalab validate`` of a script workflow must not import numerics.

This is the property checklist rule 6 exists for: a document is refused — or
accepted — on a laptop with no accelerator, before a single step runs. It is
also what lets the script's content hash sit in the digest without costing
anything: hashing needs no import.

**The guarantee, stated as weakly as it can be while still buying that.** Not
"nothing is imported" — resolving a ``{"module": …}`` locator calls
`importlib.util.find_spec`, and the stdlib imports the target's *parent
packages* ("If the name is for a submodule (contains a dot), the parent package
is automatically imported"). So the property is about *what* is imported:

1. the script **module** itself is found and hashed, never imported; and
2. every **package that may contain a shipped script** is importable without
   numerics.

(2) is the obligation the ``{"path": …}`` case below cannot see, and it was
unmet: ``causalab.io.plots.workflow_figures`` is a shipped script and
``causalab/io/plots/__init__.py`` eagerly imported the plotting stack, so
``validate`` of the shipped ``weekdays.json`` reached torch. That package is
lazy now (PEP 562), and the two tests added here are what keep it so — one per
numbered clause.

The v1 version of this test guarded the transform-op registry's record/body
split. The registry is gone; the guarantee it protected is not, and this is
where it moved.

It has to run in a **subprocess**: ``tests/conftest.py`` imports torch at
session scope, so an in-process ``"torch" not in sys.modules`` check would be
false regardless of whether the loader behaved. The subprocess precedent is
``tests/neural/engines/pytorch_hooks/test_end_to_end_iia.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.protocol._env import FIXTURES
from tests._helpers.paths import PROTOCOLS_DIR, WORKFLOWS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
#: the shipped intervention specifications, one file per experiment.
METHODS = PROTOCOLS_DIR

_PROBE = """
import json, sys
from causalab.cli import main

code = main(["validate", sys.argv[1], "--data-root", sys.argv[2],
             "--artifacts-root", sys.argv[3], "--engine", "auto"])
print(json.dumps({"code": code, "torch": "torch" in sys.modules}))
"""

#: A script whose module scope imports torch. If the loader imported it, the
#: probe below would see torch in sys.modules — which is the whole point.
_TORCHY_SCRIPT = """
import torch


def main(inputs, outputs):
    outputs["out"].write_text("[]")
"""


def _workflow() -> dict:
    return {
        "version": "1",
        "output_dir": "probe",
        "steps": {
            "locate": {
                "type": "intervention_protocol",
                "document": str(METHODS / "weekdays_locate_scan.json"),
            },
            "reduce": {
                "type": "script",
                "script": {"path": "scripts/torchy.py"},
                "inputs": {"table": {"step": "locate", "file": "iia.json"}},
                "outputs": {"out": "out.json"},
            },
        },
    }


def test_validate_of_a_script_workflow_never_imports_torch(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "torchy.py").write_text(_TORCHY_SCRIPT)
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(_workflow(), indent=2))

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            str(wf),
            str(FIXTURES / "data"),
            str(FIXTURES / "artifacts"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["code"] == 0, "the document should validate"
    assert not result["torch"], (
        "validate imported torch — a script must be hashed and parsed, never "
        "imported (workflow spec §4.2)"
    )


#: Packages that hold a shipped step script, so a `{"module": …}` locator can
#: name something inside them. Derived from the shipped locators rather than
#: listed by hand — see `test_the_script_packages_are_the_shipped_ones`.
SCRIPT_PACKAGES = (
    "causalab.analysis",
    "causalab.measurement.analysis",
    "causalab.io.plots",
    "causalab.workflow.scripts",
)

_IMPORT_PROBE = """
import importlib, json, sys

importlib.import_module(sys.argv[1])
heavy = sorted(m for m in ("torch", "numpy", "pandas", "matplotlib", "scipy",
                           "sklearn", "safetensors")
               if m in sys.modules)
print(json.dumps({"heavy": heavy}))
"""


def _shipped_script_modules() -> set[str]:
    """Every `{"module": …}` locator the shipped workflows name."""
    out: set[str] = set()
    for shipped in sorted((WORKFLOWS_DIR).glob("*.json")):
        document = json.loads(shipped.read_text())
        for step in document.get("steps", {}).values():
            locator = step.get("script")
            if isinstance(locator, dict) and "module" in locator:
                out.add(locator["module"])
    return out


def test_the_script_packages_are_the_shipped_ones() -> None:
    """`SCRIPT_PACKAGES` covers every shipped `{"module": …}` locator.

    Without this the list below is a hand-maintained allowlist, and the next
    shipped script under a new package would be unguarded — silently, which is
    the failure mode this whole area keeps producing.
    """
    modules = _shipped_script_modules()
    assert modules, "no shipped `{'module': …}` locators found — the reader is wrong"
    uncovered = sorted(
        module
        for module in modules
        if not any(module.startswith(f"{package}.") for package in SCRIPT_PACKAGES)
    )
    assert not uncovered, (
        f"shipped scripts live outside SCRIPT_PACKAGES: {uncovered} — add the "
        "package here so importing it stays numerics-free"
    )


@pytest.mark.parametrize("package", SCRIPT_PACKAGES)
def test_a_script_package_is_importable_without_numerics(package: str) -> None:
    """Clause 2: importing a script's parent package imports no numerics.

    A subprocess for the same reason as the test above — `conftest.py` has
    already imported torch in this process.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, package],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    heavy = json.loads(completed.stdout.strip().splitlines()[-1])["heavy"]
    assert not heavy, (
        f"importing {package} pulls {heavy} — a `{{'module': …}}` locator under "
        "it makes `validate` pay for the numerics stack, because find_spec "
        "imports parent packages (workflow spec §4.2)"
    )


def test_validate_of_a_shipped_module_locator_never_imports_torch() -> None:
    """Clause 1+2 together, through the real CLI on the real shipped workflow.

    `weekdays.json` names `causalab.io.plots.workflow_figures`, which is the
    locator that broke the guarantee. The `{"path": …}` case above could not see
    it: a path locator has no parent package to import.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            str(WORKFLOWS_DIR / "weekdays.json"),
            str(FIXTURES / "data"),
            str(FIXTURES / "artifacts"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["code"] == 0, "the shipped workflow should validate"
    assert not result["torch"], (
        "validate of the shipped workflow imported torch — resolving its "
        "`causalab.io.plots.workflow_figures` locator imported the parent "
        "package eagerly (workflow spec §4.2)"
    )


# --------------------------------------------------------------------------- #
# the pipeline's two verbs, against a real engine's capability set
# --------------------------------------------------------------------------- #

_PIPELINE_PROBE = """
import json, sys
from pathlib import Path

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.lowering import point_count
from causalab.protocol.pipeline import build, validate
from causalab.tasks import TASKS_ROOT

env = ResolutionEnv(
    datasets=FileDatasets(root=Path(sys.argv[2]), fallback_roots=(TASKS_ROOT,)),
    artifacts=FileArtifacts(root=Path(sys.argv[3])),
)
document = Path(sys.argv[1])
compiled = build(document, base_dir=document.parent, env=env)
validated = validate(compiled, "pytorch_hooks", env=env, data=True)
print(json.dumps({
    "same": validated is compiled,
    "points": point_count(compiled.axes),
    "digest": compiled.campaign_digest,
    "torch": "torch" in sys.modules,
}))
"""


def test_build_and_validate_against_an_engine_never_import_torch() -> None:
    """The invariant of the pipeline's two verbs: ``build``
    then ``validate`` — the checklist, the data rules and the engine's
    shortfall against the reference engine's *registered* capability set —
    on the shipped ``weekdays_locate_scan`` against the fixture environment,
    with torch never imported. The engine is named, not constructed: its
    capabilities are the registry's rows, so no engine module loads.

    A subprocess for the same reason as every test above — ``conftest.py``
    has already imported torch in this process.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PIPELINE_PROBE,
            str(METHODS / "weekdays_locate_scan.json"),
            str(FIXTURES / "data"),
            str(FIXTURES / "artifacts"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["same"], "validate returns the object it was handed"
    assert result["points"] == 56 and len(result["digest"]) == 64
    assert not result["torch"], (
        "build + validate(engine='pytorch_hooks', data=True) imported torch — "
        "the pipeline must decide everything it decides from the registry and "
        "the resolved tables alone"
    )
