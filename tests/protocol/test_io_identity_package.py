"""The io half of a load and the hashing half of the protocol import
standalone and torch-free, and the identify stage is a function.

A refactor moved reading and the environment into ``causalab/io/`` — ``env.py``
(``ResolutionEnv`` and its file-backed services, from ``protocol/resolve.py``),
``sources.py`` (``load_text`` / ``apply_overrides`` / ``check_json_values``
from ``loader.py``, ``resolve_artifact_fields`` from ``resolve.py``, and the
compile's ``identify`` stage as a pure function with its record types from
``compile.py``), ``tables.py`` (``protocol/tables.py`` moved whole) and
``results_io.py`` (``write_outputs`` from ``neural/shared/outputs.py``) — and
gathered hashing into ``protocol/identity.py`` (``canonical_bytes`` /
``digest`` from ``canonical.py``, the identity half of ``code.py``, the
artifact identity schema from ``resolve.py``), leaving the materialising half
of ``canonical.py`` in ``schema/explicit.py``. The one-beat star-import shims
at the old paths are deleted. Two things hold the move:

* **standalone import, torch-free** — every new module imports on its own in
  a fresh interpreter without pulling ``torch`` in, and ``causalab validate``
  never imports ``results_io`` (the one module whose function body reaches the
  tensor library);
* **the extraction is the stage** — [`causalab.io.sources.identify`][] is
  the compiler's ``identify`` stage re-homed from the build object to
  parameters and returns; on the corpus its three results are the compiled
  protocol's ``data``, ``artifacts`` and ``diagnostics``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.io.env import ResolutionEnv
from causalab.io.sources import identify
from causalab.protocol.pipeline import compile_protocol

from tests.protocol._env import FIXTURES, steps_of
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
CORPUS = PROTOCOLS_DIR

NEW_MODULES = (
    "causalab.io.env",
    "causalab.io.sources",
    "causalab.io.tables",
    "causalab.io.results_io",
    "causalab.protocol.identity",
    "causalab.protocol.schema.explicit",
    "causalab.protocol.compiled",
    "causalab.protocol.pipeline",
    "causalab.protocol.receipt",
)


# --------------------------------------------------------------------------- #
# the import graph
# --------------------------------------------------------------------------- #

# The namespace-preservation census and the pins on the one-beat star-import
# shims (``protocol/resolve.py``, ``canonical.py``, ``code.py``, ``tables.py``)
# were deleted with the shims.


def test_schema_init_does_not_import_explicit() -> None:
    """The registry imports the schema package and ``explicit`` imports the
    registry: an eager import from the package ``__init__`` would be a cycle,
    so ``explicit`` is reached by its own path only."""
    tree = ast.parse((REPO / "causalab/protocol/schema/__init__.py").read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "causalab.protocol.schema.explicit" not in imported
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "causalab.protocol.schema"
        and any(alias.name == "explicit" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_identity_does_not_import_the_io_layer() -> None:
    """``identity`` is what ``io/env.py`` imports the artifact identity schema
    from; an import back would be the cycle the module-level graph must not
    have (``env`` reaches ``identity`` while it is still executing)."""
    tree = ast.parse((REPO / "causalab/protocol/identity.py").read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(m == "causalab.io" or m.startswith("causalab.io.") for m in modules)


# --------------------------------------------------------------------------- #
# standalone import, torch-free
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.rsplit(".", 1)[1])
def test_a_new_module_imports_standalone_without_torch(module: str) -> None:
    """A fresh interpreter imports the module *first* — before anything under
    ``causalab.protocol`` — and ``torch`` is not in ``sys.modules`` afterwards.
    First matters: the module's own body is then what initializes the protocol
    package, the order in which an eager ``protocol/__init__`` would re-enter
    the half-executed module (``test_io_modules_import_cold_without_the_protocol_package``
    pins the lazy ``__init__`` that keeps it from doing so)."""
    code = f"import sys\nimport {module}\nprint('torch' in sys.modules)\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "torch was imported"


def _fresh(code: str) -> dict[str, Any]:
    """Run ``code`` in a fresh interpreter (``tests/conftest.py`` has torch and
    the protocol package loaded already) and parse its last line as JSON."""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO)
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


#: What an eager ``protocol/__init__`` used to pull in — the compiler and the
#: CLI — and a lazy one must not.
_EAGER_FAN_OUT = (
    "causalab.protocol.pipeline",
    "causalab.protocol.reports",
)

_LAZY_PROBE = """
import importlib, json, sys
import causalab.protocol as protocol

after_import = sorted(m for m in sys.modules if m.startswith("causalab"))
resolved = sorted(name for name in protocol.__all__ if getattr(protocol, name) is not None)
schema = protocol.schema
try:
    protocol.no_such_name
except AttributeError as err:
    unknown = str(err)
else:
    unknown = None
print(json.dumps({
    "after_import": after_import,
    "torch": "torch" in sys.modules,
    "resolved": resolved,
    "schema_is_the_submodule": schema is importlib.import_module("causalab.protocol.schema"),
    "unknown": unknown,
    "dir": dir(protocol),
}))
"""


def test_the_protocol_package_is_lazy() -> None:
    """``import causalab.protocol`` imports no submodule (PEP 562): the io
    modules the protocol reads with import protocol modules at module level,
    and an eager fan-out from the package ``__init__`` would re-enter whichever
    of them a process imported first. Every public name still resolves, a
    submodule name resolves to the submodule, and an unknown name is an
    ``AttributeError`` naming the package."""
    import causalab.protocol as protocol

    assert set(protocol._EXPORTS) == set(protocol.__all__)  # pyright: ignore[reportPrivateUsage]
    result = _fresh(_LAZY_PROBE)
    assert result["after_import"] == ["causalab", "causalab.protocol"], (
        f"importing the package imported {result['after_import']}"
    )
    assert not any(m in result["after_import"] for m in _EAGER_FAN_OUT)
    assert not result["torch"]
    assert result["resolved"] == sorted(protocol.__all__)
    assert result["schema_is_the_submodule"]
    assert result["unknown"] == (
        "module 'causalab.protocol' has no attribute 'no_such_name'"
    )
    assert set(protocol.__all__) <= set(result["dir"])
    assert {"schema", "pipeline", "registry", "rules"} <= set(result["dir"])


@pytest.mark.parametrize(
    "module",
    ("causalab.io.env", "causalab.io.sources", "causalab.io.tables"),
    ids=lambda m: m.rsplit(".", 1)[1],
)
def test_io_modules_import_cold_without_the_protocol_package(module: str) -> None:
    """The three io modules the protocol imports at module level import cold —
    a fresh interpreter, nothing else first — and doing so initializes the
    protocol *package* without its compiler, runner or loader: with an eager
    ``protocol/__init__`` this was ``ImportError: cannot import name
    'ResolutionEnv'`` (the shim's star import reaching a half-executed ``env``),
    which ``io/__init__.py`` used to mask by importing ``causalab.protocol``
    first."""
    probe = (
        "import json, sys\n"
        f"import {module}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('causalab'))))\n"
    )
    loaded = _fresh(probe)
    assert module in loaded
    assert "causalab.protocol" in loaded, "the module imports protocol modules"
    assert not any(m in loaded for m in _EAGER_FAN_OUT), sorted(
        set(loaded) & set(_EAGER_FAN_OUT)
    )


_PROBE = """
import json, sys
from causalab.cli import main

code = main(["validate", sys.argv[1], "--data-root", sys.argv[2],
             "--artifacts-root", sys.argv[3], "--engine", "auto"])
print(json.dumps({"code": code, "torch": "torch" in sys.modules,
                  "results_io": "causalab.io.results_io" in sys.modules}))
"""


def test_validate_never_imports_results_io() -> None:
    """The one io module whose function body reaches the tensor library is
    the run-output writer; a pure verb has no reason to load it, and the
    protocol layer does not (the engine side, ``neural/shared/results.py``,
    is what re-exports it)."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            str(CORPUS / "weekdays_locate_scan.json"),
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
    assert not result["torch"], "validate imported torch"
    assert not result["results_io"], "validate imported causalab.io.results_io"


# --------------------------------------------------------------------------- #
# identify() is the identify stage
# --------------------------------------------------------------------------- #

#: A few corpus documents that name a dataset ref and, between them, a
#: ``file_path`` load (the apply document, against the fixture bundle) and a
#: value reference (the sweep's ``{"artifact": …}`` fields).
IDENTIFY_DOCUMENTS = (
    "weekdays_locate_scan.json",
    "weekdays_das_apply.json",
    "weekdays_das_sweep.json",
)


@pytest.mark.parametrize("name", IDENTIFY_DOCUMENTS)
def test_identify_equals_the_compiled_stage(name: str, env: ResolutionEnv) -> None:
    """The compiled protocol's ``data``, ``artifacts`` and ``diagnostics`` are
    [`identify`][causalab.io.sources.identify] over the authored tree and the point parses — the same
    code path, so the two must be equal record for record, in order."""
    path = CORPUS / name
    compiled = compile_protocol(path, env=env)
    authored = json.loads(path.read_text())
    data, artifacts, diagnostics = identify(
        authored, steps_of(compiled, env).documents, env
    )
    assert dict(compiled.data) == data
    assert tuple(compiled.artifacts) == tuple(artifacts)
    assert tuple(compiled.diagnostics) == tuple(diagnostics)
    assert data, f"{name} names no dataset ref — pick a document that does"


def test_identify_reports_a_deferred_file_check(env: ResolutionEnv) -> None:
    """The store's word that it deferred a ``file_path`` check becomes a
    ``deferred_check`` diagnostic with an unread identity — through
    [`identify`][causalab.io.sources.identify] exactly as through the stage."""
    path = CORPUS / "weekdays_das_apply.json"
    compiled = compile_protocol(path, env=env)
    loads = [a for a in compiled.artifacts if a.key is None]
    assert loads, "the apply document should load a bundle"

    class Deferring:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def defers(self, reference: str) -> bool:
            return True

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    deferred_env = ResolutionEnv(
        datasets=env.datasets,
        artifacts=Deferring(env.artifacts),  # type: ignore[arg-type]
    )
    _data, artifacts, diagnostics = identify(
        json.loads(path.read_text()), steps_of(compiled, env).documents, deferred_env
    )
    assert all(a.deferred and a.identity is None for a in artifacts if a.key is None)
    assert [d.kind for d in diagnostics] == ["deferred_check"] * len(loads)
    assert {d.path for d in diagnostics} == {a.path for a in loads}
