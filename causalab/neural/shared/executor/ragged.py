"""Handle row windows and variable position widths.

``RowWindow`` identifies the rows and width bucket for a forward.
``RaggedLandingMixin`` applies the declared policy through the executor's
operand and write helpers. Checks reject disallowed ragged writes and
operands whose per-row widths disagree. ``ragged_geometry_of`` builds
the receipt record. Ragged read values live in ``shared.values``.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from causalab.neural.shared.gather import (
    dense_index,
    flat_index,
    gather_positions,
    splice_features,
)
from causalab.neural.shared.sites import ResolvedSite
from causalab.neural.shared.values import RaggedValue
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.schema import Document, WriteSpec, operand_reads


@dataclasses.dataclass(frozen=True)
class RowWindow:
    """The rows ``[start, stop)`` of a role's ``total`` rows that one forward
    covers — a microbatch.

    Everything row-indexed in the write math is addressed through this rather
    than through the tensor a hook happens to hold: resolved positions and
    tensor operands are sliced to the window, and a ``gaussian`` draw is made
    over all ``total`` rows and then sliced, so a row receives the same noise
    whether it ran in one forward or in the third of five (§8 asks the RNG to
    be bit-stable across layouts). [`whole`][] is the single-forward case,
    where no slicing happens at all.

    ``members`` narrows the window to a **width bucket** (§5 rule 19,
    ``exact_length_buckets``): the rows of ``[start, stop)`` a ragged write
    lands together because they address the same number of positions, as
    indices into the role's rows. A bucket is not a forward — the forward is
    still the window's — so it never reaches the engine; it is what the write
    math slices operands, routing tables and the ``gaussian`` draw by
    ([`index`][]), and how a routed write names its examples
    ([`examples`][]). ``None`` is the whole window.
    """

    start: int
    stop: int
    total: int
    members: tuple[int, ...] | None = None

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def index(self) -> "slice | list[int]":
        """What selects this window's rows out of a role-wide tensor: the
        contiguous slice, or the bucket's row indices."""
        return self.slice if self.members is None else list(self.members)

    @property
    def examples(self) -> list[int]:
        """The role's row indices this window covers, in order."""
        return (
            list(range(self.start, self.stop))
            if self.members is None
            else list(self.members)
        )

    def bucket(self, local_rows: Sequence[int]) -> "RowWindow":
        """The bucket of this window's rows at ``local_rows`` (indices into the
        window, i.e. into the tensor the forward holds)."""
        return RowWindow(
            self.start,
            self.stop,
            self.total,
            members=tuple(self.start + i for i in local_rows),
        )

    @property
    def size(self) -> int:
        return self.stop - self.start if self.members is None else len(self.members)

    @property
    def whole(self) -> bool:
        return self.members is None and self.start == 0 and self.stop == self.total


def ragged_write_error(
    ename: str, widths: list[int], *, model: str | None = None
) -> ValidationError:
    """One message for rule 19's ``refuse`` path, raised from the pre-flight
    and from the landing path — the same refusal wherever it is noticed
    first. Typed ``ragged_write_unsupported`` (§2.4): a write that declares
    no ``ragged`` policy, or ``refuse``, meets this; a declared
    ``exact_length_buckets`` / ``padded_masked`` lands instead
    (`ExecutorBase._land_ragged`)."""
    where = f" in intervened_model {model!r}" if model else ""
    return ValidationError(
        19,
        f"write {ename!r}{where} addresses ragged position widths "
        f"{widths} — an all-positions or variable write needs every row to "
        "address the same number of positions, because the landed slice has "
        "one shape for the whole batch (§5.19)",
        path=f"writes.{ename}.pos",
        reason="ragged_write_unsupported",
    )


def ragged_operand_error(
    value: str,
) -> ValidationError:
    """Rule 19 for a ragged *operand* under ``refuse`` (the absent-field
    behaviour): the read it pairs into the write came back with unequal
    per-row widths, and there is no aligned shape to land it on."""
    return ValidationError(
        19,
        f"operand read {value!r} is ragged (unequal per-row "
        "position widths) — pairing ragged windows into a write "
        "has no aligned shape, so the write is refused rather "
        "than landed on a guess (§5.19)",
        reason="ragged_write_unsupported",
    )


def operand_width_error(
    value: str, mismatches: list[tuple[int, int, int]]
) -> ValidationError:
    """Rule 19 for an operand whose row widths disagree with the write's
    under a landing policy — a ragged read re-nested row by row
    (`ExecutorBase._nest_ragged_operand`), or a dense read whose one
    width is not every row's (`ExecutorBase._check_dense_operand`):
    an operand pairs into a ragged write row by row, at each row's own width,
    and a row where the two windows differ has no aligned shape — it is
    refused, never truncated or left-aligned into the narrower window.
    ``mismatches`` is every such ``(row, operand width, write width)`` over
    the whole window — the same rows, and so the same message, whichever
    policy lands the write."""
    rows = "; ".join(
        f"row {row} (operand {got}, write {want})" for row, got, want in mismatches
    )
    return ValidationError(
        19,
        f"operand read {value!r} has a width that disagrees with the write's "
        f"on {rows} — an operand pairs into a ragged write row by row, at each "
        "row's own width (only a one-position operand broadcasts), so the write "
        "is refused rather than landed on a guess (§5.19)",
        reason="ragged_write_unsupported",
    )


def ragged_geometry_of(policy: str, per_row: Sequence[Sequence[int]]) -> dict[str, Any]:
    """The receipt's record of one ragged write (§8
    ``execution.ragged``): the declared policy, every row's width, and
    the ``[width, rows]`` buckets — what ``exact_length_buckets`` lands by
    and what ``padded_masked`` masks by. Recorded, not gated."""
    widths = [len(row) for row in per_row]
    return {
        "policy": policy,
        "widths": widths,
        "buckets": [[width, widths.count(width)] for width in sorted(set(widths))],
    }


class RaggedLandingMixin:
    """The two ragged landings of the executor's write math (§5 rule 19):
    an operand re-nested to the write's per-row widths, and a write landed
    under ``exact_length_buckets`` or ``padded_masked``.

    Composed into [`ExecutorBase`][]
    through [`WriteMathMixin`][causalab.neural.shared.executor.writes.WriteMathMixin];
    the host surface the two methods read is declared below for the type
    checker and provided by the executor."""

    if TYPE_CHECKING:
        doc: Document
        bundle: Any

        def _operand_lookup(
            self,
            value: Any,
            *,
            rows: RowWindow | None = None,
            ragged: Sequence[int] | None = None,
        ) -> torch.Tensor | float: ...

        def _written_value(
            self,
            ename: str,
            write: WriteSpec,
            site: ResolvedSite,
            v_pre: torch.Tensor,
            *,
            lookup: "Callable[[Any], torch.Tensor | float] | None" = None,
            rows: RowWindow | None = None,
            routing: torch.Tensor | None = None,
            v_ref: torch.Tensor | None = None,
        ) -> torch.Tensor: ...

    def _nest_ragged_operand(
        self,
        value: str,
        stored: RaggedValue,
        rows: RowWindow | None,
        widths: Sequence[int],
    ) -> torch.Tensor:
        """A ragged read as a write operand under a landing policy (§5 rule
        19): its rows over ``rows``, each checked to be exactly as wide as the
        write's window on that row, stacked into ``(rows, max width, …)`` with
        zero padding past a row's width — the same frame the write's
        ``padded_masked`` gather uses, and a bucket's frame when every width
        is the same. Padding is never written back: the landing masks it."""
        chunks = torch.split(stored.flat, list(stored.widths))
        examples = list(range(len(chunks))) if rows is None else rows.examples
        picked = [chunks[i] for i in examples]
        want = [int(width) for width in widths]
        mismatches = [
            (row, int(chunk.shape[0]), width)
            for row, chunk, width in zip(examples, picked, want)
            if int(chunk.shape[0]) != width
        ]
        if mismatches:
            raise operand_width_error(value, mismatches)
        out = stored.flat.new_zeros((len(picked), max(want), *stored.flat.shape[1:]))
        for i, chunk in enumerate(picked):
            out[i, : chunk.shape[0]] = chunk
        return out

    def _land_ragged(
        self,
        ename: str,
        write: WriteSpec,
        site: ResolvedSite,
        tensor: torch.Tensor,
        positions: list[list[int]],
        *,
        policy: str,
        pad_to: int,
        rows: RowWindow,
        routing: torch.Tensor | None,
        reference: torch.Tensor | None = None,
    ) -> None:
        """Land one write whose rows address different numbers of positions,
        under its declared ``ragged`` policy (§2.8, §5 rule 19) — every row at
        its own width, inside the forward the window already runs, so nothing
        about batch geometry, fire counts or prefix keys changes:

        * ``exact_length_buckets`` groups the window's rows by width and lands
          one dense gather per width — a [`RowWindow.bucket`][], so a tensor
          operand, a routing table and the ``gaussian`` draw are indexed by
          the bucket's rows exactly as a window slices them;
        * ``padded_masked`` pads every row's positions to ``pad_to`` (the
          widest row of the batch) with its own last position, lands one
          gather over the padded frame, and scatters back **only** the real
          slots — the pad slot is read (a duplicate of a real activation) and
          never written.

        Every per-position mechanism writes the same values under either
        policy; only a ``gaussian`` draw, shaped by the landed slice, differs
        between a bucket's width and the padded width.

        ``reference`` is the tensor before any write at the address landed,
        for a ``renormalize`` write: it is gathered at the same index as the
        running value and handed on as the renormalize's ``f₀`` (§2.8).
        ``None`` means the running value is the pre-write value.

        The pre-flight
        ([`check_write_widths`][causalab.neural.shared.executor.base.ExecutorBase.check_write_widths]) has already recorded the geometry.
        """
        fslice = site.feature_slice or slice(None)
        widths = [len(row) for row in positions]
        # every operand that is a read is paired to the whole window first,
        # whichever policy lands it: an operand whose widths disagree with the
        # write's on any row is rule 19 here, naming the same rows under both
        # policies — not the first bucket's alone, and never `_coerce`'s P2
        for ref in operand_reads(self.doc, write.do):
            self._operand_lookup(ref, rows=rows, ragged=widths)
        if policy == "exact_length_buckets":
            for width in sorted(set(widths)):
                members = [i for i, w in enumerate(widths) if w == width]
                bucket = rows.bucket(members)
                index = dense_index(
                    [positions[i] for i in members], tensor.device, rows=members
                )
                landed = gather_positions(tensor, index)
                v_new = self._written_value(
                    ename,
                    write,
                    site,
                    landed[..., fslice],
                    lookup=functools.partial(
                        self._operand_lookup, rows=bucket, ragged=[width] * len(members)
                    ),
                    rows=bucket,
                    routing=None if routing is None else routing[index.pair],
                    v_ref=None
                    if reference is None
                    else gather_positions(reference, index)[..., fslice],
                )
                tensor[index.pair] = splice_features(
                    landed, fslice, v_new.to(tensor.dtype)
                )
            return
        if policy != "padded_masked":
            raise AssertionError(
                f"unknown ragged policy {policy!r} reached the landing"
            )
        pad_to = max(pad_to, max(widths))
        padded = [
            [*row, *([row[-1] if row else 0] * (pad_to - len(row)))]
            for row in positions
        ]
        # a pad slot duplicates a real position, so where any row is short
        # this table is not distinct and the gather keeps autograd's
        # accumulating backward; the index decides that itself (gather.py)
        index = dense_index(padded, tensor.device)
        landed = gather_positions(tensor, index)
        v_new = self._written_value(
            ename,
            write,
            site,
            landed[..., fslice],
            lookup=functools.partial(self._operand_lookup, rows=rows, ragged=widths),
            rows=rows,
            routing=None if routing is None else routing[index.pair],
            v_ref=None
            if reference is None
            else gather_positions(reference, index)[..., fslice],
        )
        spliced = splice_features(landed, fslice, v_new.to(tensor.dtype))
        # only the real slots go back: a pad slot duplicates a real index, and
        # an advanced-index assignment with duplicates lands one of the two
        # values arbitrarily — so padding is never written, by construction.
        # The real (row, slot) pairs are known on the host, so selecting them
        # is a distinct gather rather than a boolean mask (which would need
        # the device to count its hits)
        real_slots = flat_index([list(range(width)) for width in widths], tensor.device)
        real_positions = flat_index(positions, tensor.device)
        tensor[real_positions.pair] = gather_positions(spliced, real_slots)
