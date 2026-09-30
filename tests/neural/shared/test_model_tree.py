"""The `sites.py` / `model_tree.py` seam: the module tree is read in one place,
the resolver imports it, and `adapter_of` keeps its spelling on both."""

from __future__ import annotations

import ast
import inspect

import pytest

from causalab.neural.shared import model_tree, sites

pytestmark = pytest.mark.unit


def test_model_tree_does_not_import_sites() -> None:
    tree = ast.parse(inspect.getsource(model_tree))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert not {m for m in imported if m.endswith("sites")}, imported


def test_adapter_of_is_one_object_on_both_modules() -> None:
    assert sites.adapter_of is model_tree.adapter_of
    assert "adapter_of" in sites.__all__
