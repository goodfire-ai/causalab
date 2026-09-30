"""The memory soak on the real ``Qwen/Qwen3.6-35B-A3B`` (``docs/model_parallelism.md``
§10.6 "the soak", §11): the parallel golden's inference document swept to
fifty points (``tests/golden/_parallel/soak.py``: the first 25 layers ×
two write positions) captured through the recorder at ``ep=2`` and ``tp=2`` as
``causalab run … --device cuda`` subprocesses through the recording entry,
so every rank writes one memory line per point; every rank's trace is then
held to [`flat_after_warmup`][causalab.neural.shared.parallel.soak.flat_after_warmup] —
after the first five points, allocated bytes drift by at most one MiB per
point on a least-squares fit and the reserved pool does not climb every
point. Residency (``residency.py``) pins what a rank holds after its load
and the workflow bench a peak; this pins that a long run stays where it
started.

The replay resumes from ``CAUSALAB_PARALLEL_GOLDENS_ROOT`` when set (a run
whose receipt exists is not repeated); the traces to inspect are
``<root>/soak/<geometry>/memory_trace/rank<r>.jsonl``. Nothing is compared
to world 1 here — the parity tiers hold the numbers. The CPU guard for the
rule, the trace and the document is ``tests/golden/test_parallel_soak_rule.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest
import torch

from causalab.neural.shared.parallel.soak import TraceLine, fit, samples_of
from tests.golden import _parallel as par
from tests.golden._parallel import soak

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

ROOT_VARIABLE = "CAUSALAB_PARALLEL_GOLDENS_ROOT"

Traces = Callable[[str], tuple[Path, list[list[TraceLine]]]]


@pytest.fixture(scope="module")
def traces(tmp_path_factory: pytest.TempPathFactory) -> Traces:
    """The soak run at a geometry, once for the module on first request;
    one geometry's world resident at a time."""
    kept = os.environ.get(ROOT_VARIABLE)
    root = Path(kept) if kept else tmp_path_factory.mktemp("soak")
    root.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple[Path, list[list[TraceLine]]]] = {}

    def get(geometry: str) -> tuple[Path, list[list[TraceLine]]]:
        if geometry not in cache:
            out = soak.run_geometries(root, par.A3B, (geometry,))[geometry]
            cache[geometry] = (out, par.traces(out))
        return cache[geometry]

    return get


@pytest.mark.parametrize("geometry", soak.GEOMETRIES)
def test_every_rank_traced_every_point_on_its_own_device(
    traces: Traces, geometry: str
) -> None:
    out, ranks = traces(geometry)
    receipt = par.receipt(out)
    assert receipt["execution"]["parallel"] == par.parallel_block(geometry)
    assert len(receipt["points"]) == soak.POINTS
    assert len(ranks) == par.WORLD
    for rank, lines in enumerate(ranks):
        assert [line.point for line in lines] == list(range(soak.POINTS)), rank
        assert {line.rank for line in lines} == {rank}
        assert {line.device for line in lines} == {f"cuda:{rank}"}
        assert all(
            line.allocated > 0 and line.reserved >= line.allocated for line in lines
        )


@pytest.mark.parametrize("geometry", soak.GEOMETRIES)
def test_allocated_memory_stays_flat_after_warm_up_on_every_rank(
    traces: Traces, geometry: str
) -> None:
    out, ranks = traces(geometry)
    for rank, lines in enumerate(ranks):
        print(
            geometry, f"rank{rank}", fit(samples_of(lines), warmup=soak.WARMUP).render()
        )
    assert soak.problems([samples_of(lines) for lines in ranks]) == [], (
        f"traces under {out / par.TRACES}"
    )


@pytest.mark.parametrize("geometry", soak.GEOMETRIES)
def test_the_loader_and_residency_are_clean_on_the_long_run(
    traces: Traces, geometry: str
) -> None:
    out, _ = traces(geometry)
    assert par.load_problems(geometry, par.load_reports(out)) == []
