"""Parse and validate intervention specifications.

The protocol layer defines the document format, its compiler, and the engine
interface. Position resolution accepts a tokenizer service supplied by the
caller. Imports are lazy so resolution services can use protocol types and
validation can run before numerical libraries load.

The package root exports ``run_protocol``, ``RunContext``, and
``RUN_RECORD_NAME``. Import every other name from the module that defines
it, for example ``causalab.protocol.pipeline.compile_protocol``.

See ``docs/intervention_protocol.md`` for the format and its validation rules."""

from __future__ import annotations

from pathlib import Path
from pkgutil import iter_modules
from typing import TYPE_CHECKING, Any

#: Public name -> the module that defines it. The one place to edit when a
#: name moves; ``tests/protocol/test_io_identity_package.py`` checks that every
#: name in ``__all__`` resolves through it.
_EXPORTS: dict[str, str] = {
    # .pipeline — the run door
    "run_protocol": "causalab.protocol.pipeline",
    # .engine — the run-time half every engine's ``execute`` takes
    "RunContext": "causalab.protocol.engine",
    # .receipt — the receipt's file name, read beside ``run_protocol``
    "RUN_RECORD_NAME": "causalab.protocol.receipt",
}

__all__ = [
    "RUN_RECORD_NAME",
    "RunContext",
    "run_protocol",
]

_PACKAGE_DIR = Path(__file__).parent


def _is_submodule(name: str) -> bool:
    return (_PACKAGE_DIR / f"{name}.py").is_file() or (
        _PACKAGE_DIR / name / "__init__.py"
    ).is_file()


def __getattr__(name: str) -> Any:
    """Import the module owning ``name`` on first access (PEP 562).

    A public name imports the module that defines it; a submodule name
    (``causalab.protocol.schema`` as ``p.schema``) imports the submodule.
    """
    from importlib import import_module

    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(import_module(module), name)
        globals()[name] = value  # cache, so the next access is a plain lookup
        return value
    if _is_submodule(name):
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    submodules = {info.name for info in iter_modules([str(_PACKAGE_DIR)])}
    return sorted(set(__all__) | submodules)


if TYPE_CHECKING:  # explicit re-exports, for type checkers and IDEs only
    from causalab.protocol.engine import RunContext as RunContext
    from causalab.protocol.pipeline import run_protocol as run_protocol
    from causalab.protocol.receipt import RUN_RECORD_NAME as RUN_RECORD_NAME
