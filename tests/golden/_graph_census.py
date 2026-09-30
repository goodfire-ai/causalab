"""The census entry of the multi-rank CUDA graph golden
(``tests/golden/test_multirank_cuda_graphs.py``): ``python -m
tests.golden._graph_census run …`` is ``causalab run …`` with every rank
counting its CUDA graph replays and keeping the graph path's INFO messages —
captures, and every "runs eagerly" fallback — written at exit to
``rank<r>.json`` under `CENSUS_VARIABLE`. A run's outputs cannot say whether
its graphs engaged; this does.

As for the parallel golden's recorder (``tests/golden/_parallel/recorder.py``),
a spawned rank re-imports the parent's main module by its ``-m`` name, so the
installation at import runs in every rank before the CLI does and the
production spawn path is untouched. A spawn parent that never touched a
device writes nothing.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["CENSUS_VARIABLE", "census_path", "install"]

#: Where each rank writes its census.
CENSUS_VARIABLE = "CAUSALAB_TEST_GRAPH_CENSUS_DIR"

#: The logger whose INFO records are the graph path's decisions
#: (``cuda_graphs``, ``graph_cohort``, ``train``).
LOGGER = "causalab.neural.engines.pytorch_hooks"


def census_path(directory: Path, rank: int) -> Path:
    return directory / f"rank{rank}.json"


class _Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _save(counts: dict[str, int], handler: _Messages) -> None:
    import torch

    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return  # a spawn parent: runs nothing, touches no device
    directory = Path(os.environ[CENSUS_VARIABLE])
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    census: dict[str, Any] = {
        "rank": rank,
        "replays": counts["replays"],
        "messages": handler.messages,
    }
    census_path(directory, rank).write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n"
    )


def install() -> None:
    """Count every ``Replay`` call and keep the graph path's INFO messages;
    register the save at exit."""
    from causalab.neural.engines.pytorch_hooks import cuda_graphs

    counts = {"replays": 0}
    real = cuda_graphs.Replay.__call__

    def counted(self: Any) -> Any:
        counts["replays"] += 1
        return real(self)

    cuda_graphs.Replay.__call__ = counted  # type: ignore[method-assign]
    handler = _Messages()
    logger = logging.getLogger(LOGGER)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    atexit.register(_save, counts, handler)


if CENSUS_VARIABLE in os.environ:
    install()

if __name__ == "__main__":
    from causalab.cli import main

    sys.exit(main(sys.argv[1:]))
