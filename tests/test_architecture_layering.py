"""Static guards for the layering docs/CODEBASE.md §1 states.

Three invariants, all checked by parsing the source — no model load, no GPU
(and, alongside them, one file-scoped guard: `analysis/sequences.py` imports
nothing from `causalab.neural` at any scope — see `SEQUENCES_FILE`):

1. **`io/` has no upward imports.** It is the lowest application layer above
   third-party libs, and the layers above it consume it, so an upward edge
   would be a cycle.
2. **Shipped step scripts are torch-free at module level.** Numerics belong
   inside a script's ``main``, so hashing one costs nothing but stdlib. The
   runner already uses the same idiom for pandas and matplotlib.
3. **`protocol/` keeps no module-level edge to the workflow layer.** That is
   what makes the intervention protocol usable on its own; dispatch between
   document types lives in `causalab/cli.py`, above both packages. `io/` is
   *not* forbidden to it: docs/CODEBASE.md §1 calls `io/` the lowest application
   layer, and the environment a document is
   loaded against (`io/env.py`), the readers a compile reads with
   (`io/sources.py`) and the metric-table format (`io/tables.py`) live there
   as the protocol's inputs by design. What that costs is guarded the other
   way round: those three modules stay torch-free at module level, so the
   protocol's dependency on them never drags numerics into `validate`.
   (`io/results_io.py` is outside that guard on purpose: nothing in
   `protocol/` imports it.) `protocol/__init__.py` is lazy (PEP 562) so that
   those three modules can import protocol modules at module level without a
   cycle.

Invariant 2 is a *static* check. Its behavioural counterpart — that a real
``causalab validate`` of a script workflow leaves torch out of ``sys.modules``
— lives in ``tests/protocol/test_load_is_torch_free.py``, because
``tests/conftest.py`` imports torch at session scope and an in-process check
could never see the difference.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import causalab.analysis
import causalab.io
import causalab.measurement.analysis
import causalab.protocol
import causalab.workflow.scripts

# Static structural guard — pure AST inspection, no model load (see docstring).
pytestmark = pytest.mark.unit

#: Layers `io/` must never import from. The pre-refactor entries
#: (`causalab.methods`, `causalab.analyses`, `causalab.runner`) named packages
#: that no longer exist, so the guard had stopped guarding anything.
FORBIDDEN_PREFIXES = ("causalab.workflow.scripts", "causalab.workflow")

#: Numerics no step-script module may import at module level.
HEAVY_MODULES = ("torch", "numpy", "pandas", "scipy", "sklearn", "safetensors")

IO_DIR = Path(causalab.io.__file__).parent
ANALYSIS_DIR = Path(causalab.analysis.__file__).parent
MEASUREMENT_ANALYSIS_DIR = Path(causalab.measurement.analysis.__file__).parent
SCRIPTS_DIR = Path(causalab.workflow.scripts.__file__).parent

#: The third shipped step script. It lives beside the other plot modules rather
#: than in a scripts package, so a directory-shaped parametrize missed it while
#: `docs/CODEBASE.md` §1 named it among the torch-free three — the prose claimed
#: an enforcement that stopped one file short.
#:
#: **What this target does and does not establish.** The check below is a
#: per-file AST walk of *absolute* module-level imports, so it buys "no *new*
#: stray top-level import in this file" — not importability. It cannot see a
#: parent `__init__`, and it skips relative imports (`node.level == 0` below).
#:
#: Importability is a separate, behavioural claim, and it is checked separately:
#: `tests/protocol/test_load_is_torch_free.py::test_a_script_package_is_
#: importable_without_numerics` imports each script-holding package in a
#: subprocess. That test exists because this one could not see the defect —
#: `io/plots/__init__.py` used to import the plotting stack eagerly, so
#: resolving this file's locator pulled torch into `validate`. It is lazy now.
FIGURES_FILE = IO_DIR / "plots" / "workflow_figures.py"

#: `analysis/sequences.py` is document authoring only: unlike `logit_lens`,
#: which reaches the engine inside its functions, it imports nothing from
#: `causalab.neural` at *any* level. Its place in `analysis/` rests on that
#: property. It is pinned here because the torch-free guard below checks module
#: scope and HEAVY_MODULES only, so a function-local `from causalab.neural…`
#: would pass it silently.
SEQUENCES_FILE = ANALYSIS_DIR / "sequences.py"

#: The only io modules `protocol/` may import at module level: the resolution
#: environment, the document readers and the metric-table format. `validate`, `explain` and `digest` reach every one of
#: them, so each has to stay importable without numerics — the converse of
#: admitting the edge (see `test_protocol_does_not_link_against_the_workflow_layer`,
#: which forbids every other `causalab.io` module).
PROTOCOL_INPUT_MODULES = (
    "causalab.io.env",
    "causalab.io.sources",
    "causalab.io.tables",
)
PROTOCOL_INPUTS = tuple(
    IO_DIR / f"{m.rsplit('.', 1)[1]}.py" for m in PROTOCOL_INPUT_MODULES
)

#: Repo root, for offender labels that read the same for a file and a directory.
REPO = IO_DIR.parents[1]
PROTOCOL_DIR = Path(causalab.protocol.__file__).parent


def _module_level_imports(
    path: Path, *, module_scope_only: bool = True
) -> list[tuple[int, str]]:
    """Every absolute import executed at *module* scope.

    Imports nested inside a function body are deliberately not reported:
    deferring a heavy import into the call is exactly the discipline these
    guards exist to permit."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nested: set[int] = set()
    if module_scope_only:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    nested.add(id(inner))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in nested:
            continue
        if isinstance(node, ast.ImportFrom):
            # Only absolute imports (`node.level == 0`). Under a directory
            # target that loses nothing — a relative import lands on a sibling
            # the rglob scans anyway — but for a single-file target like
            # FIGURES_FILE a `from .pca_scatter import …` is invisible.
            # Widening it would mean resolving relative targets; the property
            # that actually matters for a lone file is stated on FIGURES_FILE.
            if node.level == 0 and node.module is not None:
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
    return found


def _matches(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _offenders(
    root: Path,
    prefixes: tuple[str, ...],
    *,
    allow: tuple[str, ...] = (),
    module_scope_only: bool = True,
) -> list[str]:
    """Module-level imports matching ``prefixes``, under a directory or in one file
    (every scope with ``module_scope_only=False``).

    ``allow`` names exact modules exempt from the match (an admitted edge)."""
    sources = [root] if root.is_file() else sorted(root.rglob("*.py"))
    return [
        f"{path.relative_to(REPO)}:{lineno} imports {module}"
        for path in sources
        for lineno, module in _module_level_imports(
            path, module_scope_only=module_scope_only
        )
        if _matches(module, prefixes) and module not in allow
    ]


def test_io_has_no_upward_imports():
    offenders = _offenders(IO_DIR, FORBIDDEN_PREFIXES)
    assert not offenders, (
        "docs/CODEBASE.md invariant 1 violated — io/ must not import from a "
        "higher layer:\n  " + "\n  ".join(offenders)
    )


def test_sequences_imports_nothing_from_the_neural_layer_at_any_level():
    offenders = _offenders(
        SEQUENCES_FILE, ("causalab.neural",), module_scope_only=False
    )
    assert not offenders, (
        "analysis/sequences.py is document authoring only — it must not reach "
        "the execution layer at any scope:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "target",
    [ANALYSIS_DIR, MEASUREMENT_ANALYSIS_DIR, SCRIPTS_DIR, FIGURES_FILE],
    ids=lambda p: p.name,
)
def test_step_scripts_are_torch_free_at_module_level(target):
    """A step script's numerics belong inside its ``main``.

    Without this, one stray top-level ``import torch`` in a new shipped script
    would make ``causalab validate`` pay for the whole numerics stack —
    silently, since every test process has torch loaded already. A script is
    *found and hashed* at load, never imported, but a document may name any
    module, so the discipline has to hold for every one that ships."""
    offenders = _offenders(target, HEAVY_MODULES)
    assert not offenders, (
        f"{target.relative_to(REPO)} must stay importable without numerics — move "
        "the import inside the function that needs it:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("target", PROTOCOL_INPUTS, ids=lambda p: p.name)
def test_the_protocol_inputs_in_io_are_torch_free_at_module_level(target):
    """The converse guard for the one downward edge ``protocol/`` has: an io
    module the protocol imports at module level is on the path of every
    torch-free verb, so a stray top-level numerics import there would cost
    ``causalab validate`` the whole stack — silently, since every test process
    has torch loaded already (the behavioural check is
    ``tests/protocol/test_load_is_torch_free.py``)."""
    offenders = _offenders(target, HEAVY_MODULES)
    assert not offenders, (
        f"{target.relative_to(REPO)} is imported by protocol/ at module level and "
        "must stay importable without numerics — move the import inside the "
        "function that needs it:\n  " + "\n  ".join(offenders)
    )


def test_protocol_does_not_link_against_the_workflow_layer():
    """``protocol/`` is the intervention protocol **alone**.

    This is the invariant that makes two packages worth having: someone who
    wants only the intervention protocol imports only that. Dispatch between the
    two document types lives in ``causalab/cli.py``, above both.

    ``causalab.io`` stays forbidden, with exactly three modules exempt:
    ``io/`` is the lowest application layer (docs/CODEBASE.md §1), and
    ``io/env.py``, ``io/sources.py`` and ``io/tables.py`` are the protocol's
    inputs by design — the environment a
    document is loaded against, what a compile reads with, and the
    metric-table format. Every other io module is either workflow-side
    (``step_io``, ``step_record``, ``events``) or pulls torch at import
    (``tensor_files``, ``artifacts``), so a new edge to one of them is
    refused here, statically; the three admitted modules are guarded the
    other way round above (they stay torch-free)."""
    offenders = _offenders(
        PROTOCOL_DIR,
        ("causalab.workflow", "causalab.analysis", "causalab.io"),
        allow=PROTOCOL_INPUT_MODULES,
    )
    assert not offenders, (
        "protocol/ must not import the workflow layer, nor any io module beyond "
        f"{', '.join(PROTOCOL_INPUT_MODULES)}, at module level:\n  "
        + "\n  ".join(offenders)
    )
