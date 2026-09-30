"""Resolve semantic token spans from document anchors.

The span grammar describes windows around named positions. Bounds and alignment
checks apply to the resolved indices in the padded frame. See the position
section of ``docs/intervention_protocol.md`` for the grammar."""

from __future__ import annotations

from typing import Callable, Iterator

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema.types import concrete_int
from causalab.protocol.schema.positions import (
    COMPOSITE_KEYS,
    is_span_object,
    parse_span_spec,
    plain,
    PositionSpec,
    PREDICATE_KEYS,
    selector,
    SPAN_KEYS,
    SpanSpec,
)

__all__ = [
    "COMPOSITE_KEYS",
    "PREDICATE_KEYS",
    "SPAN_KEYS",
    "SpanSpec",
    "constituents",
    "is_span_object",
    "parse_span_spec",
    "plain",
    "resolve_span",
    "segment_anchors",
    "selector",
    "static_indices",
    "walk",
]


# --------------------------------------------------------------------------- #
# what the document alone decides
# --------------------------------------------------------------------------- #


def walk(spec: PositionSpec) -> Iterator[PositionSpec]:
    """``spec`` and every spec nested inside it (members, anchors), depth
    first — what a load-time check iterates to see every anchor a position
    reaches."""
    yield spec
    if not isinstance(spec, SpanSpec):
        return
    nested: list[PositionSpec] = []
    nested.extend(spec.union or ())
    nested.extend(spec.intersection or ())
    if spec.before is not None:
        nested.append(spec.before)
    if spec.after is not None:
        nested.append(spec.after)
    nested.extend(spec.between or ())
    for inner in nested:
        yield from walk(inner)


def segment_anchors(spec: PositionSpec) -> Iterator[tuple[PositionSpec, str]]:
    """Every ``(spec, segment name)`` a position reaches: a whole-segment
    selector, or a ``scope`` / ``relative_to`` spelled as a segment."""
    for inner in walk(spec):
        if isinstance(inner, SpanSpec) and inner.segment is not None:
            yield inner, inner.segment
        anchor = inner.scope if inner.scope is not None else inner.relative_to
        if inner.anchor_source == "segment" and isinstance(anchor, str):
            yield inner, anchor


def static_indices(spec: PositionSpec) -> tuple[int, ...] | None:
    """The content-frame index set a spec addresses **by construction**, or
    ``None`` when only the tokenizer can say (an anchored or text-located
    selector). Sorted, distinct. An unscoped ``index`` / ``span`` /
    ``indices`` is static; a union or intersection of static members is."""
    if spec.scope is not None or spec.relative_to is not None:
        return None
    if isinstance(spec, SpanSpec):
        if spec.indices is not None:
            return tuple(sorted(set(spec.indices)))
        if spec.span is not None and isinstance(spec.span, tuple):
            lo, hi = (int(v) for v in spec.span)
            return tuple(range(lo, hi))
        if spec.union is not None or spec.intersection is not None:
            members = spec.union if spec.union is not None else spec.intersection
            assert members is not None
            sets = [static_indices(m) for m in members]
            if any(s is None for s in sets):
                return None
            first, *rest = (set(s) for s in sets if s is not None)
            for other in rest:
                first = first | other if spec.union is not None else first & other
            return tuple(sorted(first))
        return None
    if spec.generated is not None:
        return None
    if spec.index is not None and isinstance(spec.index, int):
        return (spec.index,)
    if spec.span is not None and isinstance(spec.span, tuple):
        lo, hi = (int(v) for v in spec.span)
        return tuple(range(lo, hi))
    return None


def constituents(spec: PositionSpec) -> tuple[PositionSpec, ...]:
    """The addresses a spec is classified as, for §2.3's cardinality: an
    ``atomic`` span is one; a non-atomic ``indices`` set is one per index
    (each a single-token ``index`` in the same frame); a non-atomic ``union``
    is its members; everything else is itself."""
    if not isinstance(spec, SpanSpec) or spec.atomic:
        return (spec,)
    if spec.indices is not None:
        return tuple(
            PositionSpec(
                index=n,
                scope=spec.scope,
                relative_to=spec.relative_to,
                anchor_source=spec.anchor_source,
            )
            for n in spec.indices
        )
    if spec.union is not None:
        return tuple(spec.union)
    return (spec,)


# --------------------------------------------------------------------------- #
# resolution — a pure function over one row's frame
# --------------------------------------------------------------------------- #


def resolve_span(
    spec: SpanSpec,
    *,
    frame: tuple[int, int, int],
    resolve: Callable[[PositionSpec], list[int]],
    segment_run: Callable[[str], list[int]],
    where: str = "",
) -> list[int]:
    """Resolve one span for one row into sorted, distinct padded-frame indices.

    ``frame`` is ``(first_real, content_start, padded_len)``: the row's first
    real token (past the left padding), its content start (past any chat
    prefix, §2.3) and the padded length. ``indices`` counts like ``index``
    does — ``n ≥ 0`` from the content start, ``n < 0`` from the end — while
    the relative predicates range over the row's **real** tokens, chat prefix
    included, because "before the user turn" is exactly the prefix. Members
    and anchors resolve through ``resolve`` (the position resolver, so a
    member may be any prompt-frame spec, spans included) and a named
    segment through ``segment_run``.

    A span that resolves to **no** token is refused with reason
    ``empty_selector`` rather than gathering nothing: ``between`` two runs
    that touch, or an intersection of disjoint members, is an authoring fact
    the row exposes, and a silent empty read would report a number over it.
    """
    first_real, start, padded = frame
    which = selector(spec)
    label = f"span {where}" if where else "span"
    out: list[int]
    if which == "segment":
        assert spec.segment is not None
        out = list(segment_run(spec.segment))
    elif which == "indices":
        assert spec.indices is not None
        if spec.scope is None and spec.relative_to is None:
            out = [padded + n if n < 0 else start + n for n in spec.indices]
        else:
            out = [
                token
                for n in spec.indices
                for token in resolve(
                    PositionSpec(
                        index=n,
                        scope=spec.scope,
                        relative_to=spec.relative_to,
                        anchor_source=spec.anchor_source,
                    )
                )
            ]
    elif which == "union":
        assert spec.union is not None
        found: set[int] = set()
        for member in spec.union:
            found.update(resolve(member))
        out = sorted(found)
    elif which == "intersection":
        assert spec.intersection is not None
        runs = [set(resolve(member)) for member in spec.intersection]
        out = sorted(set.intersection(*runs)) if runs else []
    elif which == "before":
        assert spec.before is not None
        anchor = resolve(spec.before)
        out = list(range(first_real, min(anchor))) if anchor else []
    elif which == "after":
        assert spec.after is not None
        anchor = resolve(spec.after)
        out = list(range(max(anchor) + 1, padded)) if anchor else []
    elif which == "between":
        assert spec.between is not None
        one, two = (resolve(anchor) for anchor in spec.between)
        if one and two:
            earlier, later = (one, two) if min(one) <= min(two) else (two, one)
            out = list(range(max(earlier) + 1, min(later)))
        else:
            out = []
    else:  # an atomic span / variable / column: the plain anchor, made one
        out = list(resolve(plain(spec)))
    out = sorted(set(int(concrete_int(i, "resolved position")) for i in out))
    if not out:
        raise ProtocolError(
            "P2",
            f"{label} ({which}) resolves to no token in this row — a span that "
            "selects nothing is refused rather than gathered silently (§2.3)",
            reason="empty_selector",
        )
    return out
