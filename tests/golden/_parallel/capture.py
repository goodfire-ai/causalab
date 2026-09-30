"""One capture of one document of the parallel golden
(``docs/model_parallelism.md`` §10.6): the runs, the measurements, the
loader's and the recorder's per-rank facts, and every reason the capture is
**not certifiable** — a receipt that differs beyond ``execution.parallel``,
a rank's load report off its axis fraction, a geometry the design holds
exact that is not (byte for byte, and every measured class at ``0.0``, the
recorded gradients included), a recorded peak above the pre-flight's
estimate where the document carries one — unless the writer explained an
inexact geometry (``inexact``), in which case it is measured and banded and
the explanation is written into the record beside it.

The reference run is world 1, or the oracle geometry's run for a document
with an `oracle` (the large model): the oracle is
run, reported and measured for memory like every geometry, and compared
to nothing — every other exact geometry is compared to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tests.golden._parallel import measure, record, runs
from tests.golden._parallel.runs import Document, Realization

__all__ = ["capture", "load_totals"]


def load_totals(reports: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per rank the bytes requested and on disk, summed over the parameters."""
    return {
        f"rank{r['rank']}": {
            "bytes_requested": sum(r["bytes_requested"].values()),
            "bytes_on_disk": sum(r["bytes_on_disk"].values()),
        }
        for r in reports
    }


def capture(
    root: Path,
    document: Document,
    realization: Realization | None = None,
    inexact: Mapping[str, str] = {},
    *,
    with_context: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """The document's block of the record, and every reason it is not
    certifiable (module docstring). ``realization`` defaults to the
    document's own, else the record's (`runs.realization_of`)."""
    if realization is None:
        realization = runs.realization_of(document)
    outputs = runs.run_all(root, document, realization)
    solo = outputs["solo"]
    refusals: list[str] = []
    banded: dict[str, measure.Measured] = {}
    exact: dict[str, float] = {}
    load_bytes: dict[str, dict[str, Any]] = {}
    memory: dict[str, dict[str, Any]] = {}
    estimates: dict[str, dict[str, Any]] = {}
    for geometry in inexact:
        if geometry not in document.exact:
            refusals.append(
                f"{geometry}: explained as inexact, but the {document.name} document "
                f"holds only {list(document.exact)} exact"
            )
        elif geometry == document.oracle:
            refusals.append(
                f"{geometry}: explained as inexact, but it is the oracle the other "
                "geometries are compared to"
            )
    if document.recorded and document.oracle is None:
        memory["solo"] = measure.memory(solo / runs.GRADIENTS, 1)
    for geometry in document.geometries:
        parallel = outputs[geometry]
        world = runs.world_of(geometry)
        reports = runs.load_reports(parallel, world)
        refusals += [
            f"{geometry}: {p}" for p in document.loader_problems(geometry, reports)
        ]
        load_bytes[geometry] = load_totals(reports)
        if document.recorded:
            memory[geometry] = measure.memory(parallel / runs.GRADIENTS, world)
        if document.estimate is not None:
            estimates[geometry] = document.estimate(geometry)
            if document.recorded:
                refusals += [
                    f"{geometry}: {p}"
                    for p in runs.memory_problems(estimates[geometry], memory[geometry])
                ]
        if geometry == document.oracle:
            continue  # the reference itself: compared to nothing
        refusals += [
            f"{geometry}: {p}"
            for p in runs.receipts_agree(
                solo, parallel, geometry, reference_geometry=document.oracle
            )
        ]
        measured = document.measure(solo, parallel, realization)
        if set(measured.classes) != set(document.classes):
            refusals.append(
                f"{geometry}: measured classes {sorted(measured.classes)} are not the "
                f"document's {sorted(document.classes)}"
            )
        if geometry in document.exact and geometry not in inexact:
            differing = runs.exact_differences(solo, parallel)
            if differing:
                refusals.append(f"{geometry}: not byte-identical: {differing}")
            off = {
                kind: value
                for kind, value in measured.classes.items()
                if (
                    value.max_abs_diff
                    if isinstance(value, measure.Measurement)
                    else value
                )
                != 0.0
            }
            if off:
                refusals.append(f"{geometry}: must be exact, measured {off!r}")
            exact[geometry] = measured.worst()
        else:
            banded[geometry] = measured
    block = record.document_record(
        document,
        realization,
        banded,
        exact,
        load_bytes,
        memory,
        inexact,
        with_context=with_context,
        estimates=estimates,
    )
    return block, refusals
