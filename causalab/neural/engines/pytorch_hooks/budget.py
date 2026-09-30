"""Choose the row budget for a fit's forward and backward passes.

An authored ``fit_rows`` is fixed. It packs whole member minibatches and
propagates allocation failures. An automatic CUDA budget probes the first
member's forward and backward, estimates bytes per row, and reserves
``MARGIN`` of available memory. Automatic budgets remain unbounded off CUDA.
Evaluation shares the resolved bound unless ``batch_rows`` is explicit.

An automatic window that runs out of memory clears gradients and allocator
cache, halves its bound, and retries before updating members. The bound
stops at one member's minibatch; failure there propagates. Receipts record
``fit_rows_resolved`` so an author can pin the measured value. Available
device memory affects the automatic choice.

Only a collective-free body can retry: tensor, expert, pipeline, and context
parallel bodies contain collectives, so their out-of-memory failures abort
the distributed run. Restart with fewer minibatch rows or a smaller authored
``fit_rows``.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Callable, NoReturn, Protocol, Sequence, TypeVar

import torch

from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.parallel import heartbeat
from causalab.neural.shared.parallel.agreements import Agreements
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.placement import Axis
from causalab.protocol.parallel import ParallelGeometry

__all__ = [
    "DistributedOutOfMemory",
    "LOCKSTEP_AXES",
    "MARGIN",
    "Meter",
    "OOMPolicy",
    "RowBudget",
    "abort_distributed",
    "cuda_meter",
]

#: The axes a budget agrees over unless told otherwise: every axis whose
#: ranks run the same windows in lockstep — the model-parallel group, the
#: pipeline stages (each runs its part of every window and the same number
#: of them, or the stage forward's sends and receives would not pair up;
#: §6.5) and the context group (§8.4). Data parallelism over rows adds the
#: data axis (``rows.RowSplit.budget_axes``), whose replicas run the same
#: windows too; over points they do not, and must not be agreed.
LOCKSTEP_AXES: tuple[Axis, ...] = ("model", "pipeline", "context")

#: The share of the device's available memory the auto bound leaves unused:
#: the probe's bytes-per-row is one window's slope, and a later window with
#: more rows, more active experts or a longer eval pass beside it rounds up.
MARGIN = 0.10

_T = TypeVar("_T")


class OOMPolicy(Enum):
    """Retry only bodies whose forward and backward contain no collectives."""

    RETRY = "retry"
    ABORT = "abort"

    @classmethod
    def for_geometry(cls, geometry: ParallelGeometry) -> "OOMPolicy":
        if geometry.model > 1 or geometry.pipeline > 1 or geometry.context > 1:
            return cls.ABORT
        return cls.RETRY


class DistributedOutOfMemory(torch.OutOfMemoryError):
    """A collective-bearing window failed; its distributed run must end."""


def abort_distributed(error: torch.OutOfMemoryError, where: str) -> NoReturn:
    """End the distributed run over ``error``, which happened ``where`` — a
    model window or one of the graphs that serve it — telling blocked peers
    first.

    Raises:
        DistributedOutOfMemory: always — ``error`` itself when it already is
            one (a graph holder's abort inside a window's body: the peers
            were told), else a new one from it.
    """
    if isinstance(error, DistributedOutOfMemory):
        raise error
    watch = heartbeat.running()
    if watch is not None:
        # A direct API caller may catch the exception without leaving its
        # process group. Tell blocked peers that this run has failed now.
        watch.finish(1)
    raise DistributedOutOfMemory(
        f"out of memory {where}; its collective sequence cannot be retried "
        "safely. The run was aborted; reduce train.batch.pairs or --fit-rows "
        "before restarting"
    ) from error


class Meter(Protocol):
    """What the auto bound needs from a device: run one window and report
    the bytes it peaked above the level it started from, and how many bytes
    the device could still give a window afterwards."""

    def measure(self, run: Callable[[], None]) -> tuple[int, int]: ...


@dataclasses.dataclass
class _CudaMeter:
    """One meter over every CUDA device the model is placed on. A window
    runs on all of them, so the slope that binds is the **worst** device's
    (the largest peak any device saw) and the room that binds is the
    **tightest** device's (the least any device has left): the bound is one
    number, and it has to hold on every device at once."""

    devices: tuple[torch.device, ...]

    def measure(self, run: Callable[[], None]) -> tuple[int, int]:
        for device in self.devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = {d: torch.cuda.memory_allocated(d) for d in self.devices}
        run()
        peaks: list[int] = []
        room: list[int] = []
        for device in self.devices:
            torch.cuda.synchronize(device)
            peaks.append(torch.cuda.max_memory_allocated(device) - before[device])
            free, _total = torch.cuda.mem_get_info(device)
            # the caching allocator's reserved-but-unused bytes are ours too
            cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
                device
            )
            room.append(free + cached)
        return max(max(peaks), 0), max(min(room), 0)


def cuda_meter(devices: DeviceMap) -> Meter | None:
    """The meter over the CUDA devices of ``devices`` (`_CudaMeter`:
    max peak, min room) — ``None`` off CUDA, where the auto bound has nothing
    to read. The map refuses mixing, so a tower is on CUDA whole or not."""
    cuda = tuple(sorted((d for d in devices.devices if d.type == "cuda"), key=str))
    return _CudaMeter(cuda) if cuda else None


@dataclasses.dataclass
class RowBudget:
    """The rows one forward of a fit may cover — a grad window, or an eval
    window packing under the same number — fixed or measured (module
    docstring). ``bound`` is ``None`` while unresolved or unbounded."""

    bound: int | None
    fixed: bool
    meter: Meter | None = None
    #: auto only: whether the probe has run
    resolved: bool = False
    #: what the probe measured, for the record: bytes per row, bytes available
    probe: tuple[int, int] | None = None
    #: how many windows ran out of memory and were re-packed — a measured
    #: bound that shrank was too loose; the counts of the grad budget and of
    #: the eval budget packing under the same bound reach the receipt together
    #: (``fit_rows_shrinks``)
    shrinks: int = 0
    #: the two host-side agreements a budget makes over the lockstep axes
    #: (``docs/model_parallelism.md`` §3): the probe's bound is the ``min``
    #: over the group, an out-of-memory window is ``any`` rank's. The
    #: world-1 default agrees nothing and calls nothing.
    agreements: Agreements = dataclasses.field(default_factory=lambda: Agreements(SOLO))
    #: the axes those agreements fold over, in order: the model, pipeline and
    #: context axes, and the data axis too under data parallelism over rows (§8.3)
    axes: tuple[Axis, ...] = LOCKSTEP_AXES
    oom_policy: OOMPolicy = OOMPolicy.RETRY

    @classmethod
    def of(
        cls,
        fit_rows: int | None,
        meter: Meter | None,
        collective: Collective = SOLO,
        axes: tuple[Axis, ...] = LOCKSTEP_AXES,
        *,
        oom_policy: OOMPolicy = OOMPolicy.RETRY,
    ) -> "RowBudget":
        """An authored bound is fixed; none is auto — measured through
        ``meter`` when there is one, unbounded when there is not. The
        budget's agreements run through ``collective`` over ``axes``."""
        agreements = Agreements(collective)
        if fit_rows is not None:
            return cls(
                bound=fit_rows,
                fixed=True,
                agreements=agreements,
                axes=axes,
                oom_policy=oom_policy,
            )
        return cls(
            bound=None,
            fixed=False,
            meter=meter,
            resolved=meter is None,
            agreements=agreements,
            axes=axes,
            oom_policy=oom_policy,
        )

    def _min(self, value: int) -> int:
        for axis in self.axes:
            value = self.agreements.min(value, axis)
        return value

    def _any(self, value: bool) -> bool:
        for axis in self.axes:
            value = self.agreements.any(value, axis)
        return value

    @property
    def probing(self) -> bool:
        """Whether the next window is the auto probe: one member, measured."""
        return not self.fixed and not self.resolved

    def take(
        self, pending: Sequence[_T], size_of: Callable[[_T], int]
    ) -> tuple[list[_T], list[_T]]:
        """The next window off ``pending`` and what is left: one item while
        probing, else the greedy prefix under ``bound`` — one item always
        fits, whatever its size (a member's minibatch is never split)."""
        if not pending:
            return [], []
        if self.probing:
            return [pending[0]], list(pending[1:])
        window: list[_T] = []
        used = 0
        for item in pending:
            size = size_of(item)
            if window and self.bound is not None and used + size > self.bound:
                break
            window.append(item)
            used += size
        # Uneven data-row slices can fit different numbers of members. Agree
        # membership, not merely a row bound, before any rank runs the window.
        count = self._min(len(window))
        return list(pending[:count]), list(pending[count:])

    def _abort(self, error: torch.OutOfMemoryError) -> NoReturn:
        abort_distributed(error, "inside a distributed model window")

    def run(self, rows: int, body: Callable[[], None], unit: int | None = None) -> None:
        """Run one window's ``body``. While probing — the first grad window of
        an auto budget, a member's forward and backward — run it under the
        meter and set the bound from what it peaked: the rows the available
        memory holds at that slope with [`MARGIN`][] held back, never fewer
        than the probe's own rows, and floored to a multiple of ``unit`` — the
        smallest window any member of the fit will bring, so that with equal
        minibatches the bound is whole members and small free-memory drift
        moves the packing only at member boundaries. Members' minibatches need
        not be equal (``pairs`` is per member and an epoch's last minibatch is
        a remainder); then the floor is coarser than one member and only the
        equal case is fully stable. ``available`` is read right after the
        probe, when only that member's optimizer state is resident, so it
        overstates what the other members leave by their states; the margin
        and the retry absorb that. A resolved or fixed budget just runs the
        body.

        Under a world above 1 the bound is the **minimum** over the lockstep
        axes' ranks (§3): every rank ran the same probe on its own device,
        and one bound has to hold on all of them — and over the data
        replicas too when ``axes`` names them (rows mode, §8.3)."""
        if not self.probing or self.meter is None:
            try:
                body()
            except torch.OutOfMemoryError as error:
                if self.oom_policy is OOMPolicy.ABORT:
                    self._abort(error)
                raise
            return
        failure: torch.OutOfMemoryError | None = None
        reading: tuple[int, int] | None = None
        try:
            reading = self.meter.measure(body)
        except torch.OutOfMemoryError as error:
            if self.oom_policy is OOMPolicy.ABORT:
                self._abort(error)
            failure = error.with_traceback(None)
        # A failed rank cannot skip directly to the window's OOM reduction
        # while a successful rank reduces its probe bound.
        if self.out_of_memory(failure is not None):
            if failure is not None:
                raise failure
            raise torch.OutOfMemoryError("a peer ran the budget probe out of memory")
        assert reading is not None
        peak, available = reading
        per_row = max(peak, 1) / max(rows, 1)
        fits = int(available * (1.0 - MARGIN) / per_row)
        member = max(unit if unit is not None else rows, 1)
        self.bound = self._min(max(rows, (fits // member) * member))
        self.probe = (int(per_row), available)
        self.resolved = True

    def out_of_memory(self, failed: bool) -> bool:
        """Whether **any** rank of the group ran this window out of memory
        (§3; the lockstep axes, and the data replicas too under ``axes`` that
        name them) — the one decision the retry may branch on, so that a
        rank that did not fail abandons and re-packs the window with the
        ranks that did. The value itself at world 1."""
        return self._any(failed)

    def can_shrink(self, window_rows: int, largest_member: int) -> bool:
        """Whether a smaller window than ``window_rows`` exists: not for a
        fixed bound, and not below one member's minibatch."""
        return bool(self._min(int(not self.fixed and window_rows > largest_member)))

    def shrink(self, window_rows: int, largest_member: int) -> None:
        """Agree the smallest rank's halved bound. A member remains indivisible
        even when another rank's bound is below its local size."""
        self.bound = self._min(max(largest_member, window_rows // 2))
        self.resolved = True
        self.shrinks += 1
