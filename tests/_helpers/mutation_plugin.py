"""pytest plugin for the ``mutmut`` run (docs/model_parallelism.md §10.7).

mutmut 3 runs pytest several times in one process — the stats pass, the
clean pass, then every mutant in a fork of that process — so a ``@given``
method meets a fresh test-class instance on each pass and hypothesis fails
its ``differing_executors`` health check, which exists to flag one test
driven from two harnesses. Here the harness is one and the same, so this
plugin waives that single check and nothing else. It is loaded only through
``[tool.mutmut]``'s ``pytest_add_cli_args`` (``-p
tests._helpers.mutation_plugin``); the ordinary gates never import it.
"""

from __future__ import annotations

from typing import Any

import hypothesis.core as _core
from hypothesis import HealthCheck

_fail_health_check = _core.fail_health_check


def _fail_health_check_but_executors(settings: Any, message: str, label: Any) -> None:
    __tracebackhide__ = True
    if label is HealthCheck.differing_executors:
        return
    _fail_health_check(settings, message, label)


def pytest_configure(config: Any) -> None:
    _core.fail_health_check = _fail_health_check_but_executors
