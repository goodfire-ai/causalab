"""The interactive half of the plots is an extra, and the core does without it.

Two claims, and the second is the one that can rot silently:

* all seven interactive-only packages are declared in the ``notebook`` extra
  and in no runtime dependency — the *declaration*. Five of the seven moved
  there first; the two this test was written for are ``dash`` and
  ``dash-cytoscape``, which that move had to leave in core because
  ``causal_graph`` imported them at module scope;
* ``causalab.io.plots`` imports, and its matplotlib views work, on a machine
  where Dash cannot be imported at all — the *behaviour*.

Only the second would catch a re-introduced module-scope ``import dash``, and
it cannot be checked in-process: this test session has Dash installed (the dev
group carries it, so the ``build_*_app`` builders stay tested), so an
``"dash" not in sys.modules`` assertion here would say nothing. The probe
therefore runs in a subprocess with a ``sys.meta_path`` finder that makes the
Dash stack unimportable — the same subprocess argument, for the same reason,
as ``tests/protocol/test_load_is_torch_free.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests._helpers.pyproject import REPO, requirement_array, requirement_names

pytestmark = pytest.mark.unit


#: Everything the extra owns. `dash`/`dash-cytoscape` are the two the code
#: reaches for; the rest is the jupyter stack, which nothing here imports.
NOTEBOOK_PACKAGES = frozenset(
    {
        "ipywidgets",
        "jupyterlab",
        "ipycytoscape",
        "dash",
        "dash-cytoscape",
        "jupyter-dash",
        "ipykernel",
    }
)

_PROBE = """
import json, sys


class Unavailable:
    \"\"\"Make the Dash stack unimportable, as a headless install has it.\"\"\"

    BLOCKED = {"dash", "dash_cytoscape", "jupyter_dash"}

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, Unavailable())

import causalab.io.plots as plots
from causalab.io.plots.causal_graph import build_structure_app

from tests._helpers.tiny import tiny_chain_model

model = tiny_chain_model()
figure = plots.build_structure_figure(model)

try:
    build_structure_app(model)
except ModuleNotFoundError as err:
    refusal = str(err)
else:
    refusal = ""

print(json.dumps({
    "imported": "dash" not in sys.modules,
    "figure": figure is not None,
    "refusal": refusal,
}))
"""


def test_no_interactive_package_is_a_runtime_dependency() -> None:
    """A headless install carries no web-app server and no jupyter stack."""
    core = requirement_names(requirement_array("dependencies"))
    assert not (core & NOTEBOOK_PACKAGES), (
        f"these belong in the `notebook` extra: {sorted(core & NOTEBOOK_PACKAGES)}"
    )


def test_the_extra_declares_all_of_them() -> None:
    assert requirement_names(requirement_array("notebook")) == NOTEBOOK_PACKAGES


def test_the_plots_package_imports_without_dash() -> None:
    """The claim the extra rests on: nothing imports Dash at module scope.

    ``causalab.io.plots.__init__`` re-exports ``causal_graph``, so a
    module-scope ``import dash`` there would make every static figure in the
    package — and everything that reaches one — unimportable in a headless
    install. That is what this used to be.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["imported"], "importing causalab.io.plots pulled in dash"
    assert result["figure"], "the matplotlib half did not survive without dash"


def test_an_interactive_builder_says_what_to_install() -> None:
    """Refusing is fine; refusing with ``No module named 'dash'`` is not — it
    does not tell the reader that the extra exists or that the other half of
    this module needs nothing."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    refusal = json.loads(completed.stdout.strip().splitlines()[-1])["refusal"]

    assert refusal, "build_structure_app did not refuse without dash"
    assert "notebook" in refusal
    assert "build_*_figure" in refusal
