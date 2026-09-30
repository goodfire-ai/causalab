"""The parallel golden's record (``docs/model_parallelism.md`` §10.6):
``tests/golden/parallel_goldens.json``, its shape, and the comparison of a
fresh capture against it.

The record is **format** `FORMAT`: the realization (model, dtype,
device), the ``tolerance`` block describing the band rule
(`.bands`), and per **document** (``inference``, ``das``, ``dbm``, ``das_dense``) its
description (and, for a document off the record's realization, the
``realization`` it ran on — the dense fp32 fit's), the exact geometries' measured maxima (``0.0``), for a fit the
``gradient_agreement`` tolerance its ranks were checked at, per banded
geometry and class the measured fields the rule needs — ``max_abs_diff``,
``scale``, ``dtype`` and the ``band`` they yield (a routing class its
``fraction``) with the per-file detail beside them — the loader's bytes per
rank, the recorder's peak memory per rank where the document is a fit, the
capture's context, and ``inexact``: a geometry the design holds exact but
the capture could not, with the explanation the writer gave.

A committed record that lacks what the rule needs — an earlier format, a
class entry without its ``scale``, a ``band`` that does not obey the rule —
is refused **by name** (`StaleRecord`, naming the recapture command),
never as a ``KeyError``: the replay skips on it and the CPU guard fails on
it. Until a document carries a capture the replay skips it, naming the
command; it never passes vacuously.
"""

from __future__ import annotations

import datetime
import json
import platform
from pathlib import Path
from typing import Any, Mapping

from tests.golden._parallel import bands
from tests.golden._parallel.bands import Measurement
from tests.golden._parallel.measure import Measured
from tests.golden._parallel.runs import Document, Realization, gradient_agreement

__all__ = [
    "CAPTURE_COMMAND",
    "FORMAT",
    "RECORD",
    "StaleRecord",
    "captured",
    "check_format",
    "compare_records",
    "context",
    "document_record",
    "entry_band",
    "entry_value",
    "load_record",
    "make_record",
    "pending_record",
    "render",
    "replay_problems",
    "tolerance",
]

FORMAT = 2
RECORD = Path(__file__).resolve().parents[1] / "parallel_goldens.json"
CAPTURE_COMMAND = (
    "HF_HUB_OFFLINE=1 uv run python tests/golden/update_parallel_goldens.py "
    "--i-have-reviewed-the-diff"
)


class StaleRecord(ValueError):
    """The committed record cannot be replayed as it stands: it predates the
    rule, or an entry lacks a field the rule needs."""

    def __init__(self, what: str) -> None:
        self.what = what
        super().__init__(f"{RECORD.name} {what}; recapture with `{CAPTURE_COMMAND}`")


def load_record() -> dict[str, Any]:
    return json.loads(RECORD.read_text())


def render(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def tolerance() -> dict[str, Any]:
    """The record's description of the band rule (`.bands`)."""
    return {
        "rule": (
            f"band = max({bands.FACTOR} * max_abs_diff, {bands.ULPS} * ulp(dtype, "
            f"scale), {bands.FLOOR}); routing: band = min(1, max("
            f"{bands.ROUTING_FACTOR} * fraction, {bands.ROUTING_FLOOR}))"
        ),
        "factor": bands.FACTOR,
        "ulps": bands.ULPS,
        "floor": bands.FLOOR,
        "routing_factor": bands.ROUTING_FACTOR,
        "routing_floor": bands.ROUTING_FLOOR,
        "justification": (
            "A sharded run is the world-1 computation up to the reduction order "
            "its collectives add (a rowwise projection's all-reduce over tp "
            "partial sums, the expert outputs' over ep ranks, a rows split's "
            "loss mean), so a float class differs from world 1 by a few units "
            "in the last place of its own dtype at the magnitude it lives at, "
            "carried through the layers above; dp over points and pp move no "
            "reduction and are held byte for byte. Per class the capture "
            "measures the largest absolute difference against world 1 "
            "(max_abs_diff) and the largest world-1 magnitude (scale) and "
            f"records the outputs' dtype. The band is {bands.FACTOR} times the "
            "measured maximum (one draw of the noise; headroom for a recapture "
            "after a kernel change; a fivefold regression falls outside "
            f"whenever the maximum is the binding term), never below {bands.ULPS} "
            "ulps of the dtype at the scale (one rounding on each side of a "
            "re-associated sum: a class that measured zero at logit scale is "
            "held to a couple of bf16 ulps there, not to the absolute floor), "
            f"never below {bands.FLOOR}. The routing class is the fraction of "
            "r_idx slots that differ: top-k routing over 256 experts in bf16 "
            "flips at near-ties when the residual's reduction order changes, so "
            f"under tp and ep it is banded at {bands.ROUTING_FACTOR} times the "
            f"measured fraction (floor {bands.ROUTING_FLOOR}), and the float "
            "classes' maxima include the re-routed slots. A fit's gradient "
            "classes are relative to the step's largest world-1 entry (scale "
            "one, fp32)."
        ),
    }


def pending_record(realization: Realization) -> dict[str, Any]:
    """The record before any capture: what to run, nothing to replay."""
    return {
        "format": FORMAT,
        "model": realization.model,
        "dtype": realization.dtype,
        "device": realization.device,
        "documents": {},
        "recapture": CAPTURE_COMMAND,
        "tolerance": tolerance(),
    }


def check_format(record: Mapping[str, Any]) -> None:
    """Refuse a record of another format by name.

    Raises:
        StaleRecord: the record's ``format`` is not `FORMAT`.
    """
    fmt = record.get("format")
    if fmt != FORMAT:
        raise StaleRecord(f"is format {fmt!r}; the band rule needs format {FORMAT}")


def captured(record: Mapping[str, Any], name: str) -> bool:
    """Whether the record carries a capture of document ``name``."""
    return bool(record.get("documents", {}).get(name, {}).get("geometries"))


def context() -> dict[str, Any]:
    """Capture provenance: the stack, the package, the node's hardware.

    ``node`` describes the hardware (the distinct CUDA device names, else the
    CPU architecture) and never names the host: the record is committed, and
    a hostname says nothing a replay can check.
    """
    import torch
    import transformers

    from causalab.provenance import runtime_identity

    identity = runtime_identity()
    cuda = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "causalab": {
            "source_kind": identity.source_kind,
            "tree_digest": identity.tree_digest,
        },
        "node": ", ".join(sorted(set(cuda))) or platform.machine(),
        "cuda": cuda,
        "captured": datetime.date.today().isoformat(),
    }


# --------------------------------------------------------------------------- #
# entries
# --------------------------------------------------------------------------- #


def _entry(kind: str, value: Measurement | float) -> dict[str, Any]:
    if kind == bands.ROUTING:
        fraction = (
            float(value) if not isinstance(value, Measurement) else value.max_abs_diff
        )
        return {"fraction": fraction, "band": bands.routing_band(fraction)}
    if not isinstance(value, Measurement):
        raise TypeError(f"class {kind!r} needs a Measurement, got {value!r}")
    return value.entry()


def entry_value(kind: str, entry: Mapping[str, Any]) -> float:
    """The measured value an entry holds — the routing ``fraction``, a
    float class's ``max_abs_diff``.

    Raises:
        StaleRecord: the field is missing.
    """
    field = "fraction" if kind == bands.ROUTING else "max_abs_diff"
    if field not in entry:
        raise StaleRecord(f"entry {kind!r} lacks {field!r}")
    return float(entry[field])


def entry_band(kind: str, entry: Mapping[str, Any]) -> float:
    """The band the rule yields for a committed entry, from its measured
    fields; the recorded ``band`` must agree with it.

    Raises:
        StaleRecord: a field the rule needs is missing, its dtype is unknown
            to the rule, or the recorded band disobeys the rule.
    """
    if kind == bands.ROUTING:
        band = bands.routing_band(entry_value(kind, entry))
    else:
        missing = [f for f in ("max_abs_diff", "scale", "dtype") if f not in entry]
        if missing:
            raise StaleRecord(f"entry {kind!r} lacks {missing} (the rule needs them)")
        try:
            band = bands.band_for(entry["max_abs_diff"], entry["scale"], entry["dtype"])
        except bands.UnknownDtype as unknown:
            raise StaleRecord(f"entry {kind!r}: {unknown}") from unknown
    if "band" not in entry:
        raise StaleRecord(f"entry {kind!r} lacks 'band'")
    if entry["band"] != band:
        raise StaleRecord(
            f"entry {kind!r} records band {entry['band']!r} but the rule yields {band!r}"
        )
    return band


def document_record(
    document: Document,
    realization: Realization,
    banded: Mapping[str, Measured],
    exact: Mapping[str, float],
    load_bytes: Mapping[str, Mapping[str, Any]] = {},
    memory: Mapping[str, Mapping[str, Any]] = {},
    inexact: Mapping[str, str] = {},
    *,
    with_context: bool = True,
    estimates: Mapping[str, Mapping[str, Any]] = {},
) -> dict[str, Any]:
    """One document's block of the record (module docstring). ``estimates``
    is the pre-flight's per-geometry, per-rank ``resident`` / ``footprint``
    beside the measured ``memory`` (the large model's documents), written
    as ``estimate`` for the reader and the §11 line."""
    geometries: dict[str, Any] = {}
    for geometry, measured in banded.items():
        entries = {
            kind: _entry(kind, value)
            for kind, value in sorted(measured.classes.items())
        }
        for name, (kind, measurement) in sorted(measured.files.items()):
            entries[kind].setdefault("files", {})[name] = {
                "max_abs_diff": measurement.max_abs_diff,
                "scale": measurement.scale,
                "dtype": measurement.dtype,
            }
        geometries[geometry] = entries
    block: dict[str, Any] = {
        "document": document.describe(realization),
        "exact": dict(exact),
        "geometries": geometries,
        "inexact": dict(inexact),
        "load": {g: dict(v) for g, v in load_bytes.items()},
    }
    if memory:
        block["memory"] = {g: dict(v) for g, v in memory.items()}
    if estimates:
        block["estimate"] = {g: dict(v) for g, v in estimates.items()}
    if document.oracle is not None:
        # the reference run is this geometry's, not world 1's (runs.Document)
        block["oracle"] = document.oracle
    if document.recorded and document.trains:
        # the §7 runtime check every fit run was held to (runs.py)
        block["gradient_agreement"] = float(gradient_agreement())
    if document.realization is not None:
        # a document off the record's realization says which it ran on
        block["realization"] = {
            "model": realization.model,
            "dtype": realization.dtype,
            "device": realization.device,
        }
    if with_context:
        block["context"] = context()
    return block


def make_record(
    committed: Mapping[str, Any] | None,
    realization: Realization,
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """The record after a capture of ``documents``: the committed record's
    other documents kept when it is of this format (a capture of one
    document leaves the others' as they were), the rule's description
    fresh."""
    record = pending_record(realization)
    if committed is not None and committed.get("format") == FORMAT:
        record["documents"] = {
            name: dict(block) for name, block in committed.get("documents", {}).items()
        }
    for name, block in documents.items():
        record["documents"][name] = dict(block)
    return record


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def compare_records(
    committed: Mapping[str, Any], fresh: Mapping[str, Any], name: str
) -> list[str]:
    """How a fresh capture of document ``name`` disagrees with the committed
    record: the realization and the document equal, every exact geometry
    still exact, every banded class's fresh value inside the committed
    band.

    Raises:
        StaleRecord: the committed record is of another format, carries no
            capture of ``name``, or an entry lacks what the rule needs.
    """
    check_format(committed)
    if not captured(committed, name):
        raise StaleRecord(f"carries no capture of the {name!r} document")
    problems: list[str] = []
    for field in ("model", "dtype", "device"):
        if committed.get(field) != fresh.get(field):
            problems.append(
                f"{field}: {fresh.get(field)!r} != committed {committed.get(field)!r}"
            )
    mine, theirs = committed["documents"][name], fresh["documents"].get(name)
    if theirs is None:
        return problems + [f"{name}: the fresh record carries no capture of it"]
    if mine.get("document") != theirs.get("document"):
        problems.append(
            f"{name}: document {theirs.get('document')!r} != committed {mine.get('document')!r}"
        )
    if mine.get("realization") != theirs.get("realization"):
        problems.append(
            f"{name}: realization {theirs.get('realization')!r} != committed "
            f"{mine.get('realization')!r}"
        )
    if mine.get("oracle") != theirs.get("oracle"):
        problems.append(
            f"{name}: oracle {theirs.get('oracle')!r} != committed {mine.get('oracle')!r}"
        )
    for geometry, worst in theirs.get("exact", {}).items():
        if worst != 0.0:
            problems.append(f"{name} {geometry}: measured {worst!r}, must be exact")
    for geometry, classes in mine["geometries"].items():
        measured = theirs.get("geometries", {}).get(geometry)
        if measured is None:
            problems.append(f"{name} {geometry}: not measured")
            continue
        if set(measured) != set(classes):
            problems.append(
                f"{name} {geometry}: classes {sorted(measured)} != committed {sorted(classes)}"
            )
        for kind, entry in classes.items():
            band = entry_band(kind, entry)
            if kind not in measured:
                continue
            value = entry_value(kind, measured[kind])
            if not value <= band:
                problems.append(
                    f"{name} {geometry} {kind}: {value!r} > band {band!r} "
                    f"(committed {entry_value(kind, entry)!r} at scale "
                    f"{entry.get('scale')!r} {entry.get('dtype', '')})"
                )
    return problems


def replay_problems(
    committed: Mapping[str, Any],
    document: Document,
    realization: Realization,
    geometry: str,
    measured: Measured,
    record_realization: Realization,
) -> list[str]:
    """How a fresh measurement of ``document`` at one banded ``geometry``
    disagrees with the committed record: the fresh block made from
    ``measured`` alone (no context), the committed record narrowed to that
    geometry, and `compare_records` between them — the replay tests'
    one comparison, spelled once.

    Raises:
        StaleRecord: as `compare_records`.
    """
    name = document.name
    fresh = make_record(
        None,
        record_realization,
        {
            name: document_record(
                document, realization, {geometry: measured}, {}, with_context=False
            )
        },
    )
    block = committed["documents"][name]
    narrowed = {
        **committed,
        "documents": {
            name: {**block, "geometries": {geometry: block["geometries"][geometry]}}
        },
    }
    return compare_records(narrowed, fresh, name)
