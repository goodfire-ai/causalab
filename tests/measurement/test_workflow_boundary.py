"""Workflow grammar reaches the lightweight measurement parser only lazily."""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

import causalab.workflow

pytestmark = pytest.mark.unit


def eager_measurement_imports(source: str, package: str) -> list[str]:
    tree = ast.parse(source)
    nested = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for child in ast.walk(node)
    }
    imports = []
    for node in ast.walk(tree):
        if id(node) in nested:
            continue
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = resolve_name("." * node.level + (node.module or ""), package)
            imports.append(target)
            imports.extend(f"{target}.{alias.name}" for alias in node.names)
    return [
        name
        for name in imports
        if name == "causalab.measurement" or name.startswith("causalab.measurement.")
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "import causalab.measurement.spec",
        "from causalab.measurement import spec",
        "from causalab import measurement",
        "from ..measurement import spec",
        "from .. import measurement",
    ],
)
def test_guard_detects_absolute_and_relative_eager_edges(statement):
    assert eager_measurement_imports(statement, "causalab.workflow")
    assert not eager_measurement_imports(
        "def parse():\n    " + statement, "causalab.workflow"
    )


def test_workflow_has_no_eager_measurement_dependency():
    workflow = Path(causalab.workflow.__file__).parent
    root = workflow.parents[1]
    offenders = {
        str(path.relative_to(root)): imports
        for path in workflow.rglob("*.py")
        if (
            imports := eager_measurement_imports(
                path.read_text(), ".".join(path.parent.relative_to(root).parts)
            )
        )
    }
    assert not offenders
