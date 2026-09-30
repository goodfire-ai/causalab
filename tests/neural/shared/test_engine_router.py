"""The engine router.

``--engine`` is a mandatory explicit input, and [`causalab.neural.shared.engine_router`][] is the one module that turns the choice into a name and, for
``run`` alone, into a constructed engine. Three things are pinned: the name
table (``auto`` is the placeholder that always answers ``pytorch_hooks``), the
lazy construction (a missing optional engine is refused by name, through
``importlib``), and that the module imports without torch — the pure verbs go
through it, so it must cost them nothing.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.neural.shared import engine_router
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import ENGINES

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", "pytorch_hooks"),
        ("pytorch_hooks", "pytorch_hooks"),
        ("nnsight", "nnsight"),
    ],
)
def test_route_name_table(choice: str, expected: str) -> None:
    """A registry engine's name is itself; `auto` is the placeholder's
    answer, the reference engine."""
    assert engine_router.route_name(choice) == expected


def test_the_choices_are_the_registrys_engines_plus_auto() -> None:
    assert engine_router.ENGINE_CHOICES == (*ENGINES, engine_router.AUTO)
    assert engine_router.AUTO == "auto"
    with pytest.raises(ValueError, match="unknown engine 'sglang'"):
        engine_router.route_name("sglang")


def test_load_engine_refuses_a_missing_nnsight_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The not-installed path, with `importlib.import_module` standing in
    for an environment without the extra: a `P2` refusal that names the
    extra to install, never a bare `ModuleNotFoundError`."""
    real = importlib.import_module

    def missing(name: str, package: Any = None) -> Any:
        if name == "causalab.neural.engines.nnsight_tracing":
            raise ModuleNotFoundError(f"No module named {name!r} (simulated)")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ProtocolError) as err:
        engine_router.load_engine("nnsight", device="cpu")
    assert err.value.code == "P2"
    assert "nnsight engine is not installed" in str(err.value)
    assert "causalab[nnsight]" in str(err.value)


def test_load_engine_builds_the_reference_engine_through_importlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The construction kwargs reach the engine's constructor as the CLI
    flags name them; `fit_rows` and `cuda_graphs` only when set, so the
    constructor contract of a stand-in stays the two-argument one."""
    import types

    seen: list[dict[str, Any]] = []

    class _Hooks:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

    stub = types.ModuleType("causalab.neural.engines.pytorch_hooks")
    stub.PytorchHooksEngine = _Hooks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "causalab.neural.engines.pytorch_hooks", stub)

    engine_router.load_engine("pytorch_hooks", device="cpu")
    engine_router.route(
        "auto", device="cuda:1", batch_rows=4, fit_rows=8, cuda_graphs=True
    )
    assert seen == [
        {"device": "cpu", "batch_rows": None},
        {"device": "cuda:1", "batch_rows": 4, "fit_rows": 8, "cuda_graphs": True},
    ]
    with pytest.raises(ValueError, match="unknown engine"):
        engine_router.load_engine("sglang", device="cpu")


_PROBE = """
import json, sys
import causalab.neural.shared.engine_router as router
print(json.dumps({
    "auto": router.route_name("auto"),
    "torch": "torch" in sys.modules,
    "engine_modules": sorted(m for m in sys.modules if m.startswith("causalab.neural.engines")),
}))
"""


def test_the_module_imports_without_torch() -> None:
    """The pure verbs import the router (`causalab.cli` reads its choices), so
    it must name no engine class at module scope. A subprocess, because
    `tests/conftest.py` imports torch at session scope."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["auto"] == "pytorch_hooks"
    assert not result["torch"], "importing the router imported torch"
    assert result["engine_modules"] == [], "importing the router imported an engine"
