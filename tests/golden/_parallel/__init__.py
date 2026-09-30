"""The parallel golden's shared half (``docs/model_parallelism.md`` §10.6):
the documents, the runs, the measurements, the band rule and the record —
used by the replay (``tests/golden/test_parallel_parity.py``) and the
capture (``tests/golden/update_parallel_goldens.py``) so the two cannot
drift apart, and by the harness's own CPU smoke
(``tests/golden/test_parallel_fit_harness.py``) on the tiny MoE.

Four documents (`DOCUMENTS`), each run as ``causalab run … --device
cuda`` subprocesses at world 1 and at its geometries — one geometry at a
time, so at most one sharded copy of a model is resident:

- ``inference`` (`.inference`): the tp/ep smoke tier's boundary
  document on the A3B; ``dp=2`` and ``pp=2`` exact, ``tp=2`` and ``ep=2``
  banded;
- ``das`` and ``dbm`` (`.fit`): the training smokes' sharded-site
  fits on the A3B — the DAS fit at ``attention_query`` (``pp=2`` exact;
  ``tp=2``, ``dp=2:rows`` banded) and the expert-neuron DBM fit at
  ``expert_activation`` (``pp=2`` exact; ``ep=2`` banded) — through the
  recording entry (`.recorder`) with the §7 runtime check on
  (``CAUSALAB_GRADIENT_AGREEMENT``), so the gradient invariant is proved on
  the real weights and the ranks' peak memory is in the record too;
- ``das_dense`` (`.fit`): the DAS fit on the dense
  ``Qwen/Qwen3-4B-Instruct-2507`` in **fp32** (``runs.DENSE``, the
  document's own realization, written into its block) at ``tp=2`` and
  ``dp=2:rows``, banded at the fp32 floor — the fit pin tight to the fp32
  ulp that the A3B's bf16 fits cannot give (no exact geometry: the model
  ties its head, which ``pp`` refuses by name).

The band (`.bands`) is ``max(3 × max_abs_diff, 2 × ulp(dtype, scale),
1e-3)`` per class, from the measured maximum, the measured world-1 scale
and the outputs' dtype; the record (`.record`) is format 2 and a
committed record that lacks what the rule needs is refused by name.
"""

from __future__ import annotations

from tests.golden._parallel import families, fit, inference, soak
from tests.golden._parallel.bands import (
    FACTOR,
    FLOOR,
    ROUTING,
    ROUTING_FACTOR,
    ROUTING_FLOOR,
    ULPS,
    Measurement,
    UnknownDtype,
    band_for,
    dtype_name,
    routing_band,
    ulp,
)
from tests.golden._parallel.capture import capture, load_totals
from tests.golden._parallel.measure import Measured, gradient_measurements
from tests.golden._parallel.record import (
    CAPTURE_COMMAND,
    FORMAT,
    RECORD,
    StaleRecord,
    captured,
    check_format,
    compare_records,
    context,
    document_record,
    entry_band,
    entry_value,
    load_record,
    make_record,
    pending_record,
    render,
    replay_problems,
    tolerance,
)
from tests.golden._parallel.recorder import GRADIENTS_VARIABLE
from tests.golden._parallel.runs import (
    A3B,
    DENSE,
    DENSE_MODEL,
    GRADIENTS,
    GRADIENT_AGREEMENT,
    GRADIENT_AGREEMENT_VARIABLE,
    GRADIENT_CLASS,
    LARGE,
    LARGE_MODEL,
    MODEL,
    RECEIPT,
    REPORTS,
    WORLD,
    Document,
    Realization,
    RunFailed,
    TRACES,
    estimate_slack,
    exact_differences,
    gradient_agreement,
    load_problems,
    load_reports,
    memory_problems,
    memory_replay_problems,
    out_name,
    parallel_block,
    realization_of,
    receipt,
    receipts_agree,
    run,
    run_all,
    sharded_parameters,
    traces,
    world_of,
)

__all__ = [
    "A3B",
    "CAPTURABLE",
    "CAPTURE_COMMAND",
    "DENSE",
    "DENSE_MODEL",
    "DOCUMENTS",
    "Document",
    "FACTOR",
    "FLOOR",
    "FORMAT",
    "GRADIENTS",
    "GRADIENTS_VARIABLE",
    "GRADIENT_AGREEMENT",
    "GRADIENT_AGREEMENT_VARIABLE",
    "GRADIENT_CLASS",
    "LARGE",
    "LARGE_DOCUMENTS",
    "LARGE_MODEL",
    "MODEL",
    "Measured",
    "Measurement",
    "RECEIPT",
    "RECORD",
    "REPORTS",
    "ROUTING",
    "ROUTING_FACTOR",
    "ROUTING_FLOOR",
    "Realization",
    "RunFailed",
    "StaleRecord",
    "TRACES",
    "ULPS",
    "UnknownDtype",
    "WORLD",
    "band_for",
    "capture",
    "captured",
    "check_format",
    "compare_records",
    "context",
    "document_record",
    "dtype_name",
    "entry_band",
    "entry_value",
    "estimate_slack",
    "exact_differences",
    "families",
    "fit",
    "gradient_agreement",
    "gradient_measurements",
    "inference",
    "large",
    "load_problems",
    "load_record",
    "load_reports",
    "load_totals",
    "make_record",
    "memory_problems",
    "memory_replay_problems",
    "out_name",
    "parallel_block",
    "pending_record",
    "realization_of",
    "receipt",
    "receipts_agree",
    "render",
    "replay_problems",
    "routing_band",
    "run",
    "run_all",
    "sharded_parameters",
    "soak",
    "tolerance",
    "traces",
    "ulp",
    "world_of",
]

#: The record's A3B-tier documents by name, in capture order — what
#: ``test_parallel_parity.py`` replays and a bare capture captures.
DOCUMENTS: dict[str, Document] = {
    d.name: d for d in (inference.DOCUMENT, fit.DAS, fit.DBM, fit.DAS_DENSE)
}

from tests.golden._parallel import large  # noqa: E402 — after DOCUMENTS: it imports the siblings above

#: The large model's documents (`.large`, eight cards), kept apart from
#: `DOCUMENTS` so the 2-GPU replay and the default capture never
#: launch them; the capture takes them by name (``--only large,das_large``).
LARGE_DOCUMENTS: dict[str, Document] = large.DOCUMENTS
#: Every document the record can hold: the A3B tier, the second-family
#: documents (`.families`, replayed by ``test_parallel_families.py``)
#: and the large model's — the latter two captured by name with ``--only``.
CAPTURABLE: dict[str, Document] = {**DOCUMENTS, **families.FAMILIES, **LARGE_DOCUMENTS}
