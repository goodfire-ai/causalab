"""A workflow resolves its ``path`` inputs and its isolated steps without a
checkout (workflow spec §3, §4.1).

Two things the runner used to derive from the *repository*: the base a relative
``{"path": …}`` input resolves against, and the directory it ran ``uv run`` from
so that uv would find *this project*. Both were the parent of the ``causalab``
package — the repo root in a checkout, and ``site-packages`` for anyone who
installed the wheel, where neither a document's data nor a ``pyproject.toml``
lives. So a workflow with a path input or an isolated step ran from a checkout
and refused from an install, and nothing here could see it: every test runs in
a checkout.

What replaces the repo root is what a document already has. A relative path is
relative to the document's own directory — the base a script's ``path``
locator already used — and an isolated step is the runner's own interpreter
with the declared ``deps`` layered over it (``uv run --no-project --python
<sys.executable> --with …``), which exists for a wheel exactly as for a
checkout. The tests run from a foreign working directory, and the isolated one
hands uv a wheel built on the spot rather than a name, so nothing is resolved
against an index: no network, no cache dependence.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

import causalab
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

pytestmark = pytest.mark.unit

READER = '''
"""Reads one scalar handed to it by a `path` input and writes it back."""
import json
from pathlib import Path


def main(inputs, outputs):
    Path(outputs["out"]).write_text(json.dumps({"k": inputs["k"]}))
'''

ISOLATED = '''
"""Imports the step's declared dep and causalab, and says where each came from."""
import json
from pathlib import Path


def main(inputs, outputs):
    import tinydep
    import causalab

    Path(outputs["out"]).write_text(json.dumps({
        "mark": tinydep.MARK,
        "causalab": causalab.__file__,
    }))
'''


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


def _document(root: Path, step: dict[str, Any]) -> Path:
    (root / "scripts").mkdir()
    doc = root / "wf.json"
    doc.write_text(
        json.dumps({"version": "1", "output_dir": "run", "steps": {"only": step}})
    )
    return doc


def _tiny_wheel(directory: Path) -> Path:
    """``tinydep-0.0.1-py3-none-any.whl``: one module, no dependencies, so uv
    installs it from the file with nothing to resolve."""
    wheel = directory / "tinydep-0.0.1-py3-none-any.whl"
    info = "tinydep-0.0.1.dist-info"
    files = {
        "tinydep/__init__.py": 'MARK = "layered"\n',
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: tinydep\nVersion: 0.0.1\n",
        f"{info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    files[f"{info}/RECORD"] = "".join(
        f"{name},,\n" for name in [*files, f"{info}/RECORD"]
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return wheel


def test_a_relative_path_input_resolves_beside_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "data").mkdir()
    (root / "data" / "pins.json").write_text(json.dumps({"k": 7}))
    doc = _document(
        root,
        {
            "type": "script",
            "script": {"path": "scripts/reader.py"},
            "inputs": {"k": {"path": "data/pins.json", "key": "k"}},
            "outputs": {"out": {"file": "out.json", "keys": {"k": 0}}},
        },
    )
    (root / "scripts" / "reader.py").write_text(READER)
    # neither the checkout nor the document's directory: the path must come
    # from the document, not from where the runner happens to be invoked
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loaded = load_workflow(doc, _env(root))
    result = run_workflow(loaded, _env(root), tmp_path / "runs", None)

    written = json.loads((result.run_root / "only" / "out.json").read_text())
    assert written == {"k": 7}


def test_an_isolated_step_layers_its_deps_over_the_runners_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert shutil.which("uv"), "isolation is `uv run`; the repo mandates uv on PATH"
    assert importlib.util.find_spec("tinydep") is None, "the fixture dep is new"
    root = tmp_path / "project"
    root.mkdir()
    wheel = _tiny_wheel(root)
    doc = _document(
        root,
        {
            "type": "script",
            "script": {"path": "scripts/isolated.py"},
            "inputs": {},
            "outputs": {
                "out": {"file": "out.json", "keys": {"mark": "", "causalab": ""}}
            },
            "runtime": {"isolate": True, "deps": [str(wheel)]},
        },
    )
    (root / "scripts" / "isolated.py").write_text(ISOLATED)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loaded = load_workflow(doc, _env(root))
    result = run_workflow(loaded, _env(root), tmp_path / "runs", None)

    written = json.loads((result.run_root / "only" / "out.json").read_text())
    # the dep is there, and causalab is the very install running this test —
    # an overlay over sys.executable's environment, not a fresh one
    assert written["mark"] == "layered"
    assert Path(written["causalab"]).resolve() == Path(causalab.__file__).resolve()
    # and the overlay was an overlay: nothing was installed into this environment
    importlib.invalidate_caches()
    assert importlib.util.find_spec("tinydep") is None
    assert sys.executable  # the interpreter the runner handed to `--python`
