"""Value extraction for the chat-coherent drift tier.

One code path shared by the capture script (update_drift_goldens.py) and
the replay test, so pins and assertions are guaranteed to reduce the run
outputs identically — the structural idea inherited from the retired
tier's extract_values.

Keys:
- ``interchange.<metric>.mean`` (and ``.std`` for the continuous metric)
  from the point document's JSON metric tables;
- ``interchange.acts_mid.<stat>`` — mean/std/first/last/shape of the
  harvested residual tensor;
- ``scan.iia.<axis-label>.mean`` — per-layer IIA means from the swept
  document's single JSON table (axis coordinates are row columns).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from causalab.cli import main

from tests.golden._env import FIXTURES, GOLDEN_PROTOCOLS

DOCS = ("drift_interchange_im.json", "drift_locate_scan_im.json")
ACCURACY_GATE = 0.9  # the old tier's baseline gate, kept verbatim
PINS = Path(__file__).parent / "drift_goldens.json"

#: Columns that are never a sweep axis: the value, the row keys, the
#: provenance stamp, the record identity every row repeats
#: (``unit`` / ``estimand_version``, spec §2.10) — constant over a table, so
#: grouping by them would only lengthen every label — the eligibility
#: record (``eligible`` on every row, ``reason_code`` on an excluded row
#: alone; spec §2.10 "Eligibility"), which is a per-row fact about the
#: value, not a coordinate, and the windowed-read columns
#: (``MetricTable.add_windowed`` — one row per (example, position): ``step``
#: is the position the value scored, ``matched`` whether the example
#: addressed anything; outputs.py), per-row facts about the value in the same
#: way — except that ``step`` is a coordinate of the measurement (the position
#: scored), not of the sweep: folded into the coordinate's mean by default so
#: the label grammar stays the sweep's; the first windowed drift document
#: decides whether it belongs in the label instead. ``point`` and ``name``
#: are retained as tolerated names the current
#: writer never emits — a superset is harmless, the census only asks that
#: every written non-coordinate column is named here. Spelled as literals
#: rather than imported from ``causalab.neural.shared.results`` (torch-side);
#: test_extract_labels.py couples the spelling to the writer's, over a
#: ``MetricTable`` round trip of every writer column.
_META_COLUMNS = {
    "value",
    "example_id",
    "point",
    "metric",
    "name",
    "unit",
    "estimand_version",
    "eligible",
    "reason_code",
    "step",
    "matched",
}


def run_drift_documents(out_root: Path, device: str) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name in DOCS:
        out = out_root / name.removesuffix("_im.json")
        argv = [
            "run",
            "--engine",
            "auto",
            str(GOLDEN_PROTOCOLS / name),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(out / "artifacts"),
            "--out",
            str(out),
            "--device",
            device,
        ]
        code = main(argv)
        if code != 0:
            raise RuntimeError(f"{name} failed (exit {code})")
        dirs[name] = out
    return dirs


def _tensor_stats(path: Path, prefix: str, values: dict[str, Any]) -> None:
    from safetensors.torch import load_file

    for name, tensor in load_file(str(path)).items():
        flat = tensor.float().flatten()
        values[f"{prefix}.{name}.mean"] = float(flat.mean())
        values[f"{prefix}.{name}.std"] = float(flat.std())
        values[f"{prefix}.{name}.first"] = float(flat[0])
        values[f"{prefix}.{name}.last"] = float(flat[-1])
        values[f"{prefix}.{name}.shape"] = list(tensor.shape)


def _frame(path: Path) -> "pd.DataFrame":
    """One JSON metric table as a DataFrame — tables are JSON on disk
    (protocol.tables); pandas is only the reduction shape here."""
    from causalab.io.tables import read_table

    return pd.DataFrame(read_table(path))


def extract_values(dirs: dict[str, Path]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    point = dirs["drift_interchange_im.json"]
    for metric in ("acc", "iia", "ld"):
        column = _frame(point / f"{metric}.json")["value"]
        values[f"interchange.{metric}.mean"] = float(column.mean())
    values["interchange.ld.std"] = float(_frame(point / "ld.json")["value"].std())
    _tensor_stats(point / "acts_mid.safetensors", "interchange", values)

    scan = _frame(dirs["drift_locate_scan_im.json"] / "iia.json")
    for label, mean in _scan_labels(scan).items():
        values[f"scan.iia.{label}.mean"] = mean
    return values


def _scan_labels(frame: pd.DataFrame) -> dict[str, float]:
    """``{axis-label: mean}`` over one scan table: the sweep axes are the
    columns not in `_META_COLUMNS`, one label per coordinate tuple
    (``a=c`` joined by commas — a list of keys groups to tuples, one axis
    or many), the mean of ``value`` over its rows. pandas skips an excluded
    row's null, so the mean is over the eligible rows *when there are any*;
    a coordinate whose rows are all excluded keeps its label (the axis value
    is not null) with a ``NaN`` mean, which `compare` names as a
    mismatch rather than passing against any pin."""
    axes = [c for c in frame.columns if c not in _META_COLUMNS]
    labels: dict[str, float] = {}
    for coords, group in frame.groupby(axes):
        label = ",".join(f"{a}={c}" for a, c in zip(axes, coords))
        labels[label] = float(group["value"].mean())
    return labels


def load_pins(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare(
    pinned: dict[str, Any], measured: dict[str, Any], tolerance: dict[str, Any]
) -> list[str]:
    """Human-readable mismatches; empty means the replay is within pins."""
    default = float(tolerance.get("default", 0.0))
    problems = []
    for key in sorted(set(pinned) | set(measured)):
        if key not in pinned or key not in measured:
            problems.append(f"{key}: only in {'pins' if key in pinned else 'run'}")
            continue
        want, got = pinned[key], measured[key]
        if isinstance(want, list):
            if want != got:
                problems.append(f"{key}: shape {got} != pinned {want}")
        else:
            # `nan > tol` is False, so a non-finite side would pass against
            # any value on the other — a coordinate with every row excluded
            # (a NaN mean) must be a mismatch, on either side
            if not math.isfinite(float(got)) or not math.isfinite(float(want)):
                problems.append(f"{key}: not finite (run {got!r}, pinned {want!r})")
                continue
            tol = float(tolerance.get(key, default))
            if abs(float(got) - float(want)) > tol:
                problems.append(f"{key}: {got:.6g} != pinned {want:.6g} (tol {tol})")
    return problems
