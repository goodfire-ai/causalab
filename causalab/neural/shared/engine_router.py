"""Resolve an engine name and construct it for execution.

Document execution and validation commands require ``--engine``. The value
is a registered name or ``auto``, which selects ``pytorch_hooks``.
Protocol validation checks the selected engine's capabilities and reports
missing support. Execution imports and constructs the engine lazily.

This module stays torch-free at import so validation can use registry
metadata without loading the numerical stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from causalab.protocol.parallel import ONE, ParallelGeometry, format_geometry
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import ENGINES

if TYPE_CHECKING:
    from causalab.protocol.engine import Engine

__all__ = ["AUTO", "ENGINE_CHOICES", "load_engine", "route", "route_name"]

#: The spelling of "decide for me".
AUTO = "auto"

#: What ``--engine`` accepts: every registered engine's name, then
#: [`AUTO`][]. The names are the capability registry's, so the flag and
#: the rows cannot disagree about which engines exist.
ENGINE_CHOICES: tuple[str, ...] = (*ENGINES, AUTO)


def route_name(choice: str) -> str:
    """The registered engine name a ``--engine`` choice resolves to.

    A registry engine's name is itself. [`AUTO`][] is, for now, a
    **placeholder**: it always
    resolves to the reference engine, ``pytorch_hooks``. A real router that
    decides from the document — its components, its training block, the
    model's family — replaces this function's ``auto`` branch later; every
    caller already goes through it, so nothing else moves when it does.
    """
    if choice == AUTO:
        return "pytorch_hooks"
    if choice not in ENGINES:
        raise ValueError(
            f"unknown engine {choice!r}; expected one of {list(ENGINE_CHOICES)}"
        )
    return choice


def load_engine(
    name: str,
    *,
    device: str,
    cuda_graphs: bool = False,
    batch_rows: int | None = None,
    fit_rows: int | None = None,
    parallel: ParallelGeometry = ONE,
) -> Engine:
    """Construct the named engine — lazily, so the pure verbs stay torch-free
    (importlib keeps the layering honest: ``protocol/`` never links against an
    execution engine, and neither does this module at import).

    ``batch_rows`` is the reference engine's microbatch bound (``--batch-rows``)
    and ``fit_rows`` its rows-per-grad-forward bound for a fit (``--fit-rows``);
    the nnsight engine runs each group as one batch, has no grad path, and is
    built without either. ``parallel`` is the reference engine's geometry
    (``--parallel``, ``docs/model_parallelism.md`` §2); the nnsight engine is
    single-device and refused by name above a world of 1 (§8.5). A missing
    optional engine is refused by name.
    """
    if name == "pytorch_hooks":
        try:
            hooks = importlib.import_module("causalab.neural.engines.pytorch_hooks")
        except ModuleNotFoundError as err:
            raise ProtocolError(
                "P2",
                f"no execution engine available ({err}) — 'run' needs the "
                "reference engine causalab.neural.engines.pytorch_hooks",
            ) from err
        # `fit_rows` only when set: the engine's default is None already, and
        # passing it explicitly would make the kwarg part of the constructor
        # contract for every stand-in
        extra = {"fit_rows": fit_rows} if fit_rows is not None else {}
        if cuda_graphs:
            extra["cuda_graphs"] = True
        if parallel != ONE:
            extra["parallel"] = parallel
        return hooks.PytorchHooksEngine(device=device, batch_rows=batch_rows, **extra)
    if name == "nnsight":
        if parallel.world > 1:
            raise ProtocolError(
                "P4",
                "--parallel is the reference engine's geometry, and the nnsight "
                "engine is single-device (docs/model_parallelism.md §8.5), so the "
                f"two cannot be combined above a world of 1 (asked for "
                f"{format_geometry(parallel)})",
                path="--parallel",
            )
        try:
            tracing = importlib.import_module("causalab.neural.engines.nnsight_tracing")
        except ModuleNotFoundError as err:
            raise ProtocolError(
                "P2",
                f"the nnsight engine is not installed ({err}) — install "
                "the 'nnsight' extra (pip install 'causalab[nnsight]')",
            ) from err
        return tracing.NnsightEngine(device=device)
    raise ValueError(f"unknown engine {name!r}; expected one of {list(ENGINES)}")


def route(
    choice: str,
    *,
    device: str,
    cuda_graphs: bool = False,
    batch_rows: int | None = None,
    fit_rows: int | None = None,
    parallel: ParallelGeometry = ONE,
) -> Engine:
    """The engine a ``--engine`` choice names, constructed:
    [`load_engine`][] of [`route_name`][]."""
    return load_engine(
        route_name(choice),
        device=device,
        cuda_graphs=cuda_graphs,
        batch_rows=batch_rows,
        fit_rows=fit_rows,
        parallel=parallel,
    )
