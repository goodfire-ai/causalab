"""What a parallel run's outputs measure against the world-1 run's
(``docs/model_parallelism.md`` §10.6): per tensor file and per table a
`Measurement` — the largest absolute difference, the largest
world-1 magnitude, the dtype — the routing fraction over the integral files,
and the recorder's gradients and memory.

The rules the smoke tiers apply, with the scale added: a shape that differs
is ``inf``; an integral entry or a non-numeric cell that differs is ``inf``
(labels are exact or wrong); an empty entry on both sides — a scoped output
no row reached — agrees vacuously.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Sequence

from tests.golden._parallel import recorder
from tests.golden._parallel.bands import Measurement, dtype_name

__all__ = [
    "GradientRecords",
    "Measured",
    "gradient_measurements",
    "load_gradients",
    "memory",
    "routing_fraction",
    "table",
    "tensor_file",
]

#: Per rank, per step, the recorded gradient tensors (on the host).
GradientRecords = list[list[Any]]


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def tensor_file(solo: Path, parallel: Path, name: str) -> Measurement:
    """One saved tensor file: every float entry's difference and world-1
    magnitude, joined; integral entries exact or ``inf``. The dtype is the
    coarsest float dtype among the entries (``fp32`` when there is none)."""
    from causalab.io.tensor_files import load_file

    a, b = load_file(str(solo / name)), load_file(str(parallel / name))
    if set(a) != set(b):
        return Measurement(math.inf, 0.0, "fp32")
    joined: Measurement | None = None
    worst_integral = 0.0
    for key in a:
        x, y = a[key], b[key]
        if x.shape != y.shape:
            worst_integral = math.inf
        elif x.numel() == 0:
            continue
        elif x.dtype.is_floating_point:
            entry = Measurement(
                float((x.double() - y.double()).abs().max().item()),
                float(x.double().abs().max().item()),
                dtype_name(x.dtype),
            )
            joined = entry if joined is None else joined.join(entry)
        elif not bool((x == y).all()):
            worst_integral = math.inf
    if joined is None:
        return Measurement(worst_integral, 0.0, "fp32")
    return Measurement(
        max(joined.max_abs_diff, worst_integral), joined.scale, joined.dtype
    )


def table(solo: Path, parallel: Path, name: str, dtype: str) -> Measurement:
    """One saved table (a list of rows, or one row): every numeric cell's
    difference and world-1 magnitude; a non-numeric cell, a key set or a
    row count that differs is ``inf``. ``dtype`` is the outputs' — the
    model's for a metric of its logits."""
    a_rows, b_rows = _rows(solo / name), _rows(parallel / name)
    worst, scale = 0.0, 0.0
    if len(a_rows) != len(b_rows):
        return Measurement(math.inf, 0.0, dtype)
    for a, b in zip(a_rows, b_rows):
        if set(a) != set(b):
            return Measurement(math.inf, 0.0, dtype)
        for key in a:
            if _numeric(a[key]) and _numeric(b[key]):
                x, y = float(a[key]), float(b[key])
                worst = max(worst, abs(x - y))
                scale = max(scale, abs(x))
            elif a[key] != b[key]:
                return Measurement(math.inf, 0.0, dtype)
    return Measurement(worst, scale, dtype)


def _rows(path: Path) -> list[dict[str, Any]]:
    loaded = json.loads(path.read_text())
    rows = loaded if isinstance(loaded, list) else [loaded]
    return [_flatten(row) for row in rows]


def _flatten(row: Any, prefix: str = "") -> dict[str, Any]:
    """A row's nested mappings flattened to dotted keys, so every leaf is a
    cell; a list is one cell (compared whole, as a label)."""
    if not isinstance(row, dict):
        return {prefix or "value": row}
    flat: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def routing_fraction(solo: Path, parallel: Path, names: Sequence[str]) -> float:
    """The fraction of integral (routing) entries that differ between the
    two runs, over the named files; a shape that differs is ``1.0``."""
    from causalab.io.tensor_files import load_file

    differing = total = 0
    for name in names:
        a = load_file(str(solo / f"{name}.safetensors"))
        b = load_file(str(parallel / f"{name}.safetensors"))
        for key in a:
            if a[key].shape != b[key].shape:
                return 1.0
            differing += int((a[key] != b[key]).sum().item())
            total += a[key].numel()
    return differing / total if total else 0.0


# --------------------------------------------------------------------------- #
# the recorder's gradients and memory
# --------------------------------------------------------------------------- #


def load_gradients(directory: Path, rank: int) -> GradientRecords:
    """One rank's recorded gradients, on the host (a rank recorded on its
    own device; the comparison runs where no device is)."""
    import torch

    return torch.load(recorder.gradients_path(directory, rank), map_location="cpu")


def gradient_measurements(
    solo: GradientRecords, ranks: Sequence[GradientRecords]
) -> tuple[Measurement, Measurement]:
    """``(against world 1, across the ranks)``: at every step, the largest
    difference of a rank's gradient from the world-1 gradient and from the
    other ranks', **relative to the step's largest world-1 entry** (so the
    scale is one and the dtype the gradients' fp32). A rank that recorded
    no gradient at a step — a pipeline stage that does not own the
    featurizer — is skipped there; a step count that differs, or a step
    every rank skipped, is ``inf``."""
    against, across = 0.0, 0.0
    if any(len(r) != len(solo) for r in ranks):
        return Measurement(math.inf, 1.0, "fp32"), Measurement(math.inf, 1.0, "fp32")
    for step, grads in enumerate(solo):
        present = [r[step] for r in ranks if r[step]]
        if not present or any(len(p) != len(grads) for p in present):
            return Measurement(math.inf, 1.0, "fp32"), Measurement(
                math.inf, 1.0, "fp32"
            )
        for index, grad in enumerate(grads):
            scale = max(float(grad.abs().max()), 1e-30)
            mine = [p[index].double() for p in present]
            for tensor in mine:
                if tensor.shape != grad.shape:
                    return (
                        Measurement(math.inf, 1.0, "fp32"),
                        Measurement(math.inf, 1.0, "fp32"),
                    )
                against = max(
                    against, float((tensor - grad.double()).abs().max()) / scale
                )
            for other in mine[1:]:
                across = max(across, float((other - mine[0]).abs().max()) / scale)
    return Measurement(against, 1.0, "fp32"), Measurement(across, 1.0, "fp32")


def memory(directory: Path, world: int) -> dict[str, dict[str, Any]]:
    """Every rank's recorded peak memory, ``{"rank0": {...}, …}``."""
    return {
        f"rank{rank}": json.loads(recorder.memory_path(directory, rank).read_text())
        for rank in range(world)
    }


@dataclasses.dataclass
class Measured:
    """What one parallel run measured: per output class the joined
    `Measurement` (a plain fraction for the routing class), and per
    file the class it belongs to and its own measurement, for the reader."""

    classes: dict[str, Measurement | float] = dataclasses.field(default_factory=dict)
    files: dict[str, tuple[str, Measurement]] = dataclasses.field(default_factory=dict)

    def add(self, kind: str, name: str, measurement: Measurement) -> None:
        self.files[name] = (kind, measurement)
        current = self.classes.get(kind)
        self.classes[kind] = (
            measurement
            if not isinstance(current, Measurement)
            else current.join(measurement)
        )

    def worst(self) -> float:
        """The largest measured value over every class: the differences of
        the float classes and the routing fraction alike (``0.0`` for a run
        that agrees exactly)."""
        return max(
            (
                m.max_abs_diff if isinstance(m, Measurement) else m
                for m in self.classes.values()
            ),
            default=0.0,
        )
