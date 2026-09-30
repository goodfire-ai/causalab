"""The memory soak rule (``docs/model_parallelism.md`` §10.6 "the soak",
§11): whether a rank's device memory stays flat over a long run, decided
from per-point samples alone — no torch, no device.

Residency (``pytorch_hooks/residency.py``) is measured once per load and
the workflow bench reports a peak; neither says that the memory a rank
holds *after each point* stays where it was, which is what a study of
hundreds of points depends on. The soak's per-point samples are one
[`Sample`][] per point per rank — the bytes the allocator has handed out
(``allocated``) and the bytes it holds from the device (``reserved``) once
the point's outputs are written — and [`flat_after_warmup`][] is the
rule they are held to:

1. the first ``warmup`` samples are discarded — the first points fill the
   interning store with the campaign's shared captures and the allocator
   opens its pool; the tail must hold at least two samples, or the rule
   refuses **by name** rather than passing on nothing;
2. **allocated** must not trend up: the least-squares slope of the tail's
   allocated bytes over the point index (bytes per point) is at most
   ``slack``. A leak of one tensor per point is a slope of that tensor's
   size; a sawtooth whose peaks stay within ``slack`` of its troughs has a
   slope no larger than its amplitude, so it passes (the slope of any
   series over two or more points is bounded by its range);
3. **reserved** must not climb monotonically: a tail whose reserved bytes
   never fall and rise in total by more than ``slack`` per step is a pool
   that grows every point — fragmentation, or captures the store never
   drops — where a pool that opened one more segment once and stayed there
   is within the same slack and passes.

The rule is pure over the samples, so its properties — a flat series
passes, a per-point leak fails, a sawtooth within slack passes, a
monotone climb of the pool fails, too short a tail is refused by name —
are held on the CPU (``tests/golden/test_parallel_soak_rule.py``) and the
GPU soak (``tests/golden/test_parallel_soak.py``) applies the same function
to every rank's trace. The trace's spelling ([`TraceLine`][], one JSON
object per line) lives here too, so the writer (the parallel golden's
recorder) and the reader agree on it by construction.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Fit",
    "Sample",
    "TraceLine",
    "fit",
    "flat_after_warmup",
    "read_trace",
    "samples_of",
    "slope",
]

#: ``(allocated, reserved)`` bytes on one rank after one point.
Sample = tuple[int, int]


@dataclasses.dataclass(frozen=True)
class TraceLine:
    """One line of a rank's memory trace: which point on which rank left
    how many bytes allocated and reserved on its device (``None`` off an
    accelerator: nothing to read, the sample is ``(0, 0)``)."""

    rank: int
    point: int
    point_digest: str
    allocated: int
    reserved: int
    device: str | None = None

    def sample(self) -> Sample:
        return (self.allocated, self.reserved)

    def render(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def parse(cls, text: str) -> TraceLine:
        """One JSON line back to a record, refused by name when a field is
        missing or of the wrong type."""
        raw: Mapping[str, Any] = json.loads(text)
        fields = {f.name for f in dataclasses.fields(cls)}
        missing = sorted(fields - set(raw) - {"device"})
        if missing:
            raise ValueError(f"memory trace line lacks {missing}: {text!r}")
        for name in ("rank", "point", "allocated", "reserved"):
            if not isinstance(raw[name], int) or isinstance(raw[name], bool):
                raise ValueError(f"memory trace line: {name!r} is not an int: {text!r}")
        return cls(**{name: raw[name] for name in fields if name in raw})


def read_trace(path: Path) -> list[TraceLine]:
    """Every line of one rank's trace, in file order (point order)."""
    return [TraceLine.parse(line) for line in path.read_text().splitlines() if line]


def samples_of(lines: Iterable[TraceLine]) -> list[Sample]:
    return [line.sample() for line in lines]


def slope(values: Sequence[float]) -> float:
    """The least-squares slope of ``values`` over their index (per step);
    ``0.0`` for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den


@dataclasses.dataclass(frozen=True)
class Fit:
    """What the rule measured on a tail: its length, the allocated slope
    (bytes per point), the allocated range, the reserved series' total
    rise and whether it never fell."""

    points: int
    allocated_slope: float
    allocated_min: int
    allocated_max: int
    reserved_rise: int
    reserved_monotone: bool

    def render(self) -> str:
        return (
            f"{self.points} points after warm-up: allocated slope "
            f"{self.allocated_slope:+.0f} B/point in [{self.allocated_min}, "
            f"{self.allocated_max}]; reserved rise {self.reserved_rise:+d} B"
            f"{' (never fell)' if self.reserved_monotone else ''}"
        )


def fit(samples: Sequence[Sample], *, warmup: int) -> Fit:
    """The measurements of the tail after ``warmup`` (module docstring).

    Raises:
        ValueError: fewer than two samples after warm-up.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    tail = list(samples[warmup:])
    if len(tail) < 2:
        raise ValueError(
            f"the soak rule needs at least two samples after a warm-up of "
            f"{warmup}, got {len(samples)} sample(s) in all"
        )
    allocated = [a for a, _ in tail]
    reserved = [r for _, r in tail]
    return Fit(
        points=len(tail),
        allocated_slope=slope(allocated),
        allocated_min=min(allocated),
        allocated_max=max(allocated),
        reserved_rise=reserved[-1] - reserved[0],
        reserved_monotone=all(b >= a for a, b in zip(reserved, reserved[1:])),
    )


def flat_after_warmup(
    samples: Sequence[Sample], *, warmup: int, slack: int
) -> list[str]:
    """How ``samples`` fall short of flat memory (module docstring), each
    problem naming the quantity and the numbers; ``[]`` when flat. A tail
    too short to fit is itself a problem, never a pass."""
    if slack < 0:
        raise ValueError(f"slack must be non-negative, got {slack}")
    try:
        measured = fit(samples, warmup=warmup)
    except ValueError as short:
        return [str(short)]
    problems: list[str] = []
    if measured.allocated_slope > slack:
        problems.append(
            f"allocated bytes grow {measured.allocated_slope:.0f} B/point over "
            f"the {measured.points} points after warm-up, above the slack of "
            f"{slack} B/point (range [{measured.allocated_min}, "
            f"{measured.allocated_max}])"
        )
    steps = measured.points - 1
    if measured.reserved_monotone and measured.reserved_rise > slack * steps:
        problems.append(
            f"reserved bytes climb monotonically by {measured.reserved_rise} B "
            f"over {steps} steps after warm-up, above {slack} B/step"
        )
    return problems
