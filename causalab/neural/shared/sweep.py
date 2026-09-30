"""Enumerate and sign the steps of a compiled protocol.

``enumerate_steps`` traverses the axis product in canonical order, with the
last axis fastest and named-axis rows slowest. ``RunContext.points`` indexes
this order. ``signed_steps`` uses ``identity.sign_step``; ``step_records``
builds the records returned in results and written in receipts.

Tree substitution stays in protocol lowering so compiler representatives
and executed steps use the same construction. This module stays torch-free
for workflow callers.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from causalab.protocol.engine import StepRecord
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.identity import sign_step
from causalab.protocol.lowering import (
    AXES_KEY,
    CAP_MESSAGE,
    DEFAULT_POINT_CAP,
    Axes,
    Axis,
    entries,
    find_axes,
    row_tree,
    substitute,
)

if TYPE_CHECKING:
    from causalab.io.env import ResolutionEnv
    from causalab.protocol.compiled import CompiledProtocol

__all__ = [
    "Expansion",
    "Point",
    "SignedStep",
    "enumerate_steps",
    "expand",
    "expand_axes",
    "sign_steps",
    "signed_steps",
    "step_records",
]


@dataclasses.dataclass(frozen=True)
class Point:
    """One expanded point: coordinates plus the concrete raw document."""

    coords: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Expansion:
    """The result of expanding one document: its axes (document order) and
    the full cross product (last axis fastest)."""

    axes: tuple[Axis, ...]
    points: tuple[Point, ...]

    @property
    def is_swept(self) -> bool:
        return bool(self.axes)


def expand(
    raw: Mapping[str, Any], *, point_cap: int | None = DEFAULT_POINT_CAP
) -> Expansion:
    """Expand a raw document into its compiled interventions (§3).

    The cross product of all axes, last axis fastest; a document with no
    axes expands to exactly itself. ``point_cap`` refuses accidental
    combinatorial explosions (§5.14) — pass ``None`` for an explicit
    override.
    """
    axes = find_axes(raw)
    if not axes:
        return Expansion(axes=(), points=(Point(coords={}, raw=raw),))
    total = 1
    for axis in axes:
        total *= len(axis.values)
    if point_cap is not None and total > point_cap:
        raise ValidationError(
            14,
            f"sweep expands to {total} points, over the cap of {point_cap}; "
            "pass an explicit override to run a campaign this large",
        )
    points: list[Point] = []
    for combo in _cross(axes):
        assignment = {axis.path: value for axis, value in zip(axes, combo)}
        coords = {axis.id: value for axis, value in zip(axes, combo)}
        points.append(Point(coords=coords, raw=substitute(raw, assignment, ())))
    return Expansion(axes=axes, points=tuple(points))


def _cross(axes: tuple[Axis, ...]) -> Iterator[tuple[Any, ...]]:
    if not axes:
        yield ()
        return
    head, *rest = axes
    for value in head.values:
        for tail in _cross(tuple(rest)):
            yield (value, *tail)


def expand_axes(axes: Axes, *, point_cap: int | None) -> Expansion:
    """Expand a document with named axes (§3.2).

    For each combination of entries over the independent named axes (first
    declared slowest), the concrete tree is built and handed to
    [`expand`][] for the ordinary sweep axes it
    still carries; the points are concatenated in that order, each prefixed
    with one coordinate per named axis (``axes.<name>``: the entry's key).
    A dependent axis and a row's substituted fields record no coordinate of
    their own. The point cap (§5.14) is checked once, over the true count —
    rows × the inner cross product — so a correlated axis is counted as the
    rows it is, not as the cross product it replaced.
    """
    combos = list(entries(axes))
    first = row_tree(axes, combos[0])
    inner_axes = find_axes(first)
    inner_count = 1
    for axis in inner_axes:
        inner_count *= len(axis.values)
    total = len(combos) * inner_count
    if point_cap is not None and total > point_cap:
        raise ValidationError(14, CAP_MESSAGE.format(total=total, cap=point_cap))
    named = tuple(
        Axis(path=(AXES_KEY, axis.name), values=axis.keys) for axis in axes.independent
    )
    points: list[Point] = []
    for combo in combos:
        inner = expand(row_tree(axes, combo), point_cap=None)
        prefix = {
            named_axis.id: named_axis.values[i]
            for named_axis, i in zip(named, combo, strict=True)
        }
        for point in inner.points:
            points.append(Point(coords={**prefix, **point.coords}, raw=point.raw))
    return Expansion(axes=(*named, *inner_axes), points=tuple(points))


# --------------------------------------------------------------------------- #
# the one entry point: a compiled document's steps, in the canonical order
# --------------------------------------------------------------------------- #


def enumerate_steps(compiled: CompiledProtocol) -> Expansion:
    """Every step of ``compiled`` in the canonical order — the order
    [`points`][causalab.protocol.engine.RunContext.points] indexes and the run
    receipt's ``points[].index`` counts: the cross product of the axes over
    the lowered tree, last axis fastest, or the named-axis rows slowest when
    the document declared any (§3.2). The point cap was decided at build
    (rule 14, [`point_count`][causalab.protocol.lowering.point_count]), so nothing is
    capped here; the ``axes`` returned are ``compiled.axes``."""
    if compiled.named_axes is None:
        expansion = expand(compiled.tree, point_cap=None)
    else:
        expansion = expand_axes(compiled.named_axes, point_cap=None)
    assert expansion.axes == compiled.axes, "the compiler's axes are the sweep's"
    return expansion


@dataclasses.dataclass(frozen=True)
class SignedStep:
    """One enumerated step with its identity: the index into the canonical
    order, the coordinates, the concrete raw tree, its canonical form (the
    bytes the digest is over, §7) and the digest — the provenance unit every
    saved row and tensor is stamped with."""

    index: int
    coords: Mapping[str, Any]
    raw: Mapping[str, Any]
    canonical: Mapping[str, Any]
    digest: str

    @property
    def record(self) -> StepRecord:
        return StepRecord(index=self.index, coords=self.coords, digest=self.digest)


def sign_steps(
    expansion: Expansion,
    env: ResolutionEnv,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[SignedStep, ...]:
    """Sign the steps ``indices`` select of an enumeration (every step when
    ``None``), in the order given, with the one hasher
    ([`sign_step`][]): each step's canonical
    form is its own document's — the tree with every axis at one value,
    canonicalized against ``env`` with no ``axes`` block — so the digest is
    byte for byte the point digest a single-point compile produces."""
    selected = range(len(expansion.points)) if indices is None else indices
    out: list[SignedStep] = []
    for index in selected:
        point = expansion.points[index]
        canonical, digest = sign_step(point.raw, env)
        out.append(
            SignedStep(
                index=index,
                coords=point.coords,
                raw=point.raw,
                canonical=canonical,
                digest=digest,
            )
        )
    return tuple(out)


def signed_steps(
    compiled: CompiledProtocol,
    env: ResolutionEnv,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[SignedStep, ...]:
    """[`enumerate_steps`][], then [`sign_steps`][] — the steps of a
    compiled document with their identities, in one call."""
    return sign_steps(enumerate_steps(compiled), env, indices=indices)


def step_records(
    compiled: CompiledProtocol,
    env: ResolutionEnv,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[StepRecord, ...]:
    """[`signed_steps`][] as the [`StepRecord`][]
    tuple a [`RunResult`][causalab.protocol.engine.RunResult] carries and the run
    receipt writes — index, coordinates, digest."""
    return tuple(step.record for step in signed_steps(compiled, env, indices=indices))
