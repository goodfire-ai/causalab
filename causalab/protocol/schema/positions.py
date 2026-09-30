"""Token positions, semantic spans and the ``segments`` section — the object
model and the parse grammar (spec §2.2.1, §2.3).

[`PositionSpec`][] names one anchor; [`SpanSpec`][] extends it with the
span algebra's selectors (``segment``, ``indices``, ``union``,
``intersection``, ``before`` / ``after`` / ``between``, ``atomic``) and
[`parse_span_spec`][] is its grammar; [`SegmentsSpec`][] and
[`parse_segments`][] are the ``segments`` section; [`span_length`][] sizes
a position gate from a fixed ``span`` window. What the document alone decides
about spans (``walk``, ``static_indices``, ``constituents``) and the
resolution algebra stay in `causalab.protocol.positions.spans`; rule 27 stays in
`causalab.protocol.segments`. Torch-free by construction.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping

from causalab.protocol.rules.errors import ParseError, suggest
from causalab.protocol.schema.types import (
    Leaf,
)


@dataclasses.dataclass(frozen=True)
class PositionSpec:
    """§2.3 — a token-position spec. Exactly one of ``index`` / ``span`` /
    ``variable`` / ``column`` / ``all`` is set; ``scope`` /
    ``relative_to`` name an anchor (a prompt variable, or a dataset
    column when ``anchor_source`` is ``"column"``), only modify
    ``index``/``span``, and are mutually exclusive. ``all`` selects every
    content token of the row and takes no modifiers. Positions are never
    resolved to integers in the document — resolution is the
    protocol layer's, against a ``PositionFrame`` at the run door
    (`causalab.protocol.positions`, §2.3, §8), and the engine reads it.

    ``generated`` is a **frame selector**, not an anchor: it says the
    anchor resolves inside the row's greedy continuation instead of its
    prompt, and it carries the decode budget (``{"max_new_tokens": n}``).
    The anchor vocabulary is unchanged inside that frame.

    ``alignment`` is the cardinality the author *declares* for how this
    address maps across the pair's inputs (``ALIGNMENT_CARDINALITIES``,
    §2.3). Optional and undefaulted — ``None`` declares nothing. Rule 26
    checks what the document alone can decide about it; the executor checks
    the rest against the tokenizer and refuses a contradiction."""

    index: Leaf | None = None
    span: Leaf | None = None
    variable: Leaf | None = None
    column: Leaf | None = None
    all: Leaf | None = None
    scope: Leaf | None = None
    relative_to: Leaf | None = None
    alignment: str | None = None
    #: The continuation frame and its decode budget (§2.3). ``None`` is
    #: the prompt frame — where every position lived before generation.
    generated: Mapping[str, Leaf] | None = None
    #: Where ``scope``/``relative_to`` resolve from: ``"variable"`` (the
    #: role's prompt variables) or ``"column"`` (a top-level row column).
    #: Not authored on its own — it comes from the anchor's spelling.
    anchor_source: str = "variable"


def span_length(spec: Any) -> int | None:
    """The number of positions a fixed ``span`` [a, b) addresses on every row,
    or ``None`` when the spec is not such a window — an ``index``, a
    ``variable``/``column`` (as wide as the row's value), ``all`` (as wide as
    the row), a span set, a ``generated`` window (clipped to the row's decode
    width) or a ``scope``d one (sliced out of the anchor's run, so shorter on a
    short anchor) or a ``relative_to`` one (not placed by its anchor at all
    today: the resolver offsets an ``index`` only and a span so spelled falls
    through to the content frame — §2.3's span offset is unimplemented, and an
    unimplemented placement may not size a θ). Stricter than
    ``spans.static_indices`` by design: an ``index``, a static ``indices`` set,
    a static ``union`` and a one-position span all answer there and are
    declined here — a position gate's window is a contiguous span of two or
    more (a non-contiguous window is deferred, as ``all`` is). Kept here rather
    than deferring to ``spans``: a module-scope import would be circular
    (``spans`` imports ``PositionSpec`` from this module at module scope, as
    does [`causalab.protocol.schema.parse`][]), and the two answer different
    questions — ``static_indices`` asks whether an address set is static, this
    asks whether a window is a contiguous row-independent span of two or more.
    What sizes a position gate (§2.5 ``axis``);
    ``explicit._window_length_raw`` is its twin over the raw
    mapping, and a change to what counts as a window is made in both."""
    if not isinstance(spec, PositionSpec) or not isinstance(spec.span, tuple):
        return None
    if spec.generated is not None:
        return None
    if spec.scope is not None or spec.relative_to is not None:
        return None
    a, b = spec.span
    if not (isinstance(a, int) and isinstance(b, int)):
        return None
    # a window of one position is one scalar at one position — `group: site`
    # on a one-token write, not a mask *over* positions; an `index` is the
    # same address under the other spelling and is refused alike.
    # `b - a` is the realized length because an unscoped span is non-negative
    # (`_parse_position`: "content-frame, non-negative"). A §2.3 that admitted
    # a right-anchored `[-4, -1)` would have to refuse a straddling `[-2, 1)`
    # here — its length is the row's — and in `_window_length_raw` with it.
    return b - a if b - a >= 2 else None


#: The keys whose presence makes a position object a [`SpanSpec`][] (§2.3
#: span table). ``atomic`` alone promotes an ordinary ``span`` / ``variable`` /
#: ``column`` anchor into one joint address.
SPAN_KEYS: tuple[str, ...] = (
    "segment",
    "indices",
    "union",
    "intersection",
    "before",
    "after",
    "between",
    "atomic",
)
#: The two set-composition selectors, each over ≥ 2 member specs.
COMPOSITE_KEYS: tuple[str, ...] = ("union", "intersection")
#: The three relative predicates, each over one (``between``: two) anchor spec.
PREDICATE_KEYS: tuple[str, ...] = ("before", "after", "between")
#: Every selector a span spec may carry — exactly one per spec.
_SELECTORS: tuple[str, ...] = (
    "segment",
    "indices",
    "union",
    "intersection",
    "before",
    "after",
    "between",
    "span",
    "variable",
    "column",
)
#: The anchors that may take ``scope`` / ``relative_to``.
_SCOPABLE: frozenset[str] = frozenset({"indices", "span"})


@dataclasses.dataclass(frozen=True)
class SpanSpec(PositionSpec):
    """§2.3 — a span: one of the selectors below, optionally ``atomic``.

    Exactly one selector is set: ``segment`` (every token of a declared
    segment), ``indices`` (a set of content-frame indices, or of indices
    inside ``scope``'s anchor), ``union`` / ``intersection`` (over ≥ 2 member
    specs), ``before`` / ``after`` / ``between`` (every real token of the row
    strictly before / after / between the anchor runs), or the inherited
    ``span`` / ``variable`` / ``column`` with ``atomic: true``. ``atomic`` is
    ``False`` unless authored ``true`` — there is no literal spelling of the
    default, so no existing document's canonical form moves. A span addresses
    the prompt frame: ``generated`` is refused on it.
    """

    segment: str | None = None
    indices: tuple[int, ...] | None = None
    union: tuple[PositionSpec, ...] | None = None
    intersection: tuple[PositionSpec, ...] | None = None
    before: PositionSpec | None = None
    after: PositionSpec | None = None
    between: tuple[PositionSpec, PositionSpec] | None = None
    atomic: bool = False


def is_span_object(obj: Mapping[str, Any]) -> bool:
    """Whether a raw position object spells a span (any key of
    [`SPAN_KEYS`][]) — the parse dispatch ``schema._parse_position_spec``
    takes."""
    return any(key in obj for key in SPAN_KEYS)


def selector(spec: SpanSpec) -> str:
    """Which selector a span spec carries."""
    for key in _SELECTORS:
        if getattr(spec, key) is not None:
            return key
    raise AssertionError(f"span spec {spec!r} carries no selector")


def plain(spec: SpanSpec) -> PositionSpec:
    """The ordinary position spec under an ``atomic`` ``span`` / ``variable``
    / ``column`` — the same anchor, resolved by the same code path, then made
    one address by the caller."""
    return PositionSpec(
        span=spec.span,
        variable=spec.variable,
        column=spec.column,
        scope=spec.scope,
        relative_to=spec.relative_to,
        anchor_source=spec.anchor_source,
    )


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

_ParsePosition = Callable[[Any, str], PositionSpec]


def parse_span_spec(
    obj: Mapping[str, Any],
    path: str,
    *,
    parse_position: _ParsePosition,
    parse_anchor_ref: Callable[[Any, str], tuple[str, str]],
) -> SpanSpec:
    """Parse one span object (a mapping with at least one of
    [`SPAN_KEYS`][]). ``parse_position`` is the schema's position parser,
    used for members and anchors so a member may itself be a span; the
    callbacks keep this module importable without the schema importing it.

    Shape is decided here (``P2``): one selector, list shapes, member forms.
    What needs the rest of the document — that a ``segment`` is declared, that
    an ``atomic`` set has two members — is rule 27's (``segments.check``).
    """
    allowed = (
        *SPAN_KEYS,
        "span",
        "variable",
        "column",
        "scope",
        "relative_to",
        "alignment",
    )
    for key in obj:
        if key not in allowed:
            if key in ("index", "all"):
                raise ParseError(
                    "P2",
                    f"{key!r} does not combine with a span key: an index is one "
                    "token and 'all' is every token, and neither has a set to "
                    "compose — spell the set with 'indices' (§2.3)",
                    path=path,
                )
            if key == "generated":
                raise ParseError(
                    "P2",
                    "a span addresses the prompt frame; 'generated' does not "
                    "combine with span keys in v1 — address the continuation "
                    "with index/span/variable/all inside 'generated' (§2.3)",
                    path=path,
                )
            raise ParseError("P3", f"unknown key {key!r} in a span spec", path=path)
    selectors = [key for key in _SELECTORS if key in obj]
    if len(selectors) != 1:
        raise ParseError(
            "P2",
            f"a span spec needs exactly one of {list(_SELECTORS)}, got {selectors}",
            path=path,
        )
    which = selectors[0]
    atomic = False
    if "atomic" in obj:
        if obj["atomic"] is not True:
            raise ParseError(
                "P2",
                f'atomic is the flag {{"atomic": true}} — got {obj["atomic"]!r}; '
                "an unauthored span is its constituent locations, and there is "
                "no literal spelling of that default",
                path=f"{path}.atomic",
            )
        atomic = True
    if which in ("span", "variable", "column") and not atomic:
        raise ParseError(
            "P2",
            f"a bare {which!r} is an ordinary position; the only span key that "
            f"combines with it is 'atomic': true",
            path=path,
        )
    if ("scope" in obj or "relative_to" in obj) and which not in _SCOPABLE:
        raise ParseError(
            "P2",
            f"scope/relative_to modify 'indices' or an atomic 'span', not {which!r}",
            path=path,
        )
    if "scope" in obj and "relative_to" in obj:
        raise ParseError(
            "P2", "scope and relative_to are mutually exclusive", path=path
        )

    def member(raw: Any, where: str) -> PositionSpec:
        if not isinstance(raw, dict):
            raise ParseError(
                "P2",
                "a span member or anchor is a position object, spelled out — "
                'no int or "all" sugar inside a span (got '
                f"{type(raw).__name__})",
                path=where,
            )
        if "alignment" in raw:
            raise ParseError(
                "P2",
                "a member carries no alignment — declare it on the span that "
                "composes the members (§2.3)",
                path=f"{where}.alignment",
            )
        if "generated" in raw:
            raise ParseError(
                "P2",
                "a member addresses the prompt frame; 'generated' cannot appear "
                "inside a span (§2.3)",
                path=f"{where}.generated",
            )
        return parse_position(raw, where)

    def members(raw: Any, where: str, *, exactly: int | None = None) -> tuple:
        if not isinstance(raw, list):
            raise ParseError("P2", "expected a list of position objects", path=where)
        if exactly is not None and len(raw) != exactly:
            raise ParseError(
                "P2", f"expected exactly {exactly} position objects", path=where
            )
        if exactly is None and len(raw) < 2:
            raise ParseError(
                "P2",
                f"a {which} composes two or more members; one member is the "
                "member itself",
                path=where,
            )
        return tuple(member(item, f"{where}[{i}]") for i, item in enumerate(raw))

    segment = None
    indices = None
    union = intersection = None
    before = after = None
    between = None
    span = variable = column = None
    if which == "segment":
        if not isinstance(obj["segment"], str) or not obj["segment"]:
            raise ParseError(
                "P2", "segment names a declared segment", path=f"{path}.segment"
            )
        segment = obj["segment"]
    elif which == "indices":
        raw_indices = obj["indices"]
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in raw_indices
            )
        ):
            raise ParseError(
                "P2", "indices is a non-empty list of integers", path=f"{path}.indices"
            )
        if len(set(raw_indices)) != len(raw_indices):
            raise ParseError(
                "P2",
                f"indices repeats a member: {raw_indices}",
                path=f"{path}.indices",
            )
        indices = tuple(raw_indices)
    elif which in COMPOSITE_KEYS:
        parsed = members(obj[which], f"{path}.{which}")
        if which == "union":
            union = parsed
        else:
            intersection = parsed
    elif which == "between":
        first, second = members(obj["between"], f"{path}.between", exactly=2)
        between = (first, second)
    elif which in ("before", "after"):
        anchor = member(obj[which], f"{path}.{which}")
        if which == "before":
            before = anchor
        else:
            after = anchor
    elif which == "span":
        raw_span = obj["span"]
        if (
            not isinstance(raw_span, list)
            or len(raw_span) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in raw_span)
        ):
            raise ParseError(
                "P2", "a span is [a, b) — exactly two integers", path=f"{path}.span"
            )
        span = (raw_span[0], raw_span[1])
    elif which == "variable":
        if not isinstance(obj["variable"], str):
            raise ParseError("P2", "expected a string", path=f"{path}.variable")
        variable = obj["variable"]
    else:
        if not isinstance(obj["column"], str):
            raise ParseError("P2", "expected a string", path=f"{path}.column")
        column = obj["column"]

    scope_ref = (
        parse_anchor_ref(obj["scope"], f"{path}.scope") if "scope" in obj else None
    )
    relative_ref = (
        parse_anchor_ref(obj["relative_to"], f"{path}.relative_to")
        if "relative_to" in obj
        else None
    )
    anchor_ref = scope_ref if scope_ref is not None else relative_ref
    alignment = None
    if "alignment" in obj:
        if not isinstance(obj["alignment"], str):
            raise ParseError("P2", "expected a string", path=f"{path}.alignment")
        alignment = obj["alignment"]
    return SpanSpec(
        span=span,
        variable=variable,
        column=column,
        scope=scope_ref[1] if scope_ref is not None else None,
        relative_to=relative_ref[1] if relative_ref is not None else None,
        anchor_source=anchor_ref[0] if anchor_ref is not None else "variable",
        alignment=alignment,
        segment=segment,
        indices=indices,
        union=union,
        intersection=intersection,
        before=before,
        after=after,
        between=between,
        atomic=atomic,
    )


#: The ``frame`` vocabulary (§2.2.1). Absent is plain text — there is no
#: literal spelling of that default.
SEGMENT_FRAMES: tuple[str, ...] = ("chat",)
#: The chat frame's segment names, in reading order (§2.2.1). ``system`` is
#: declared only when the section gives it a source; ``continuation`` names
#: the greedy continuation (§2.3 ``generated``) in every frame.
CHAT_SEGMENTS: tuple[str, ...] = ("system", "user", "assistant_prefix", "continuation")
CONTINUATION_SEGMENT: str = "continuation"
#: How a declared segment is located (§2.2.1): from a dataset column — the
#: row's value for it, found in the row's rendered text.
SEGMENT_SOURCE_KEYS: tuple[str, ...] = ("column",)


@dataclasses.dataclass(frozen=True)
class SegmentSource:
    """Where a declared segment's text comes from: a top-level row column."""

    column: str


@dataclasses.dataclass(frozen=True)
class SegmentsSpec:
    """§2.2.1 — the parsed section. ``frame`` is ``None`` for plain text;
    ``system`` is the chat frame's optional system turn; ``declare`` names the
    column-sourced segments, in authored order."""

    frame: str | None = None
    system: SegmentSource | None = None
    declare: Mapping[str, SegmentSource] = dataclasses.field(default_factory=dict)

    def declared(self) -> tuple[str, ...]:
        """Every segment name a ``segment:`` anchor may use under this
        section: the chat names (``system`` only with a source) when the frame
        is ``chat``, ``continuation`` always, then the declared ones."""
        names: list[str] = []
        if self.frame == "chat":
            names.extend(
                name
                for name in CHAT_SEGMENTS
                if name != "system" or self.system is not None
            )
        else:
            names.append(CONTINUATION_SEGMENT)
        names.extend(name for name in self.declare if name not in names)
        return tuple(names)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(
            "P2", f"expected an object, got {type(value).__name__}", path=path
        )
    return value


def _parse_source(raw: Any, path: str) -> SegmentSource:
    obj = _mapping(raw, path)
    for key in obj:
        if key not in SEGMENT_SOURCE_KEYS:
            raise ParseError(
                "P3",
                f"unknown key {key!r}{suggest(key, SEGMENT_SOURCE_KEYS)}",
                path=path,
            )
    if "column" not in obj or not isinstance(obj["column"], str) or not obj["column"]:
        raise ParseError(
            "P2",
            'a segment source is {"column": "<name>"} — the row column whose '
            "value the segment is located from",
            path=path,
        )
    return SegmentSource(column=obj["column"])


def parse_segments(raw: Any, path: str = "segments") -> SegmentsSpec:
    """Strict-parse the ``segments`` section. Shape only: that ``frame`` is
    in the vocabulary, that ``system`` needs the chat frame and that declared
    names are legal are rule 27's, so one rule names the field for every way
    the section can be wrong."""
    obj = _mapping(raw, path)
    allowed = ("frame", "system", "declare")
    for key in obj:
        if key not in allowed:
            raise ParseError(
                "P3", f"unknown key {key!r}{suggest(key, allowed)}", path=path
            )
    if not obj:
        raise ParseError(
            "P2",
            "an empty segments section declares nothing — name a frame or "
            "declare a segment, or omit the section (plain text is the "
            "absence of it)",
            path=path,
        )
    frame = None
    if "frame" in obj:
        if not isinstance(obj["frame"], str):
            raise ParseError("P2", "frame is a string", path=f"{path}.frame")
        frame = obj["frame"]
    system = _parse_source(obj["system"], f"{path}.system") if "system" in obj else None
    declare: dict[str, SegmentSource] = {}
    if "declare" in obj:
        table = _mapping(obj["declare"], f"{path}.declare")
        for name, source in table.items():
            if not isinstance(name, str) or not name:
                raise ParseError(
                    "P2", "a segment name is a non-empty string", path=f"{path}.declare"
                )
            declare[name] = _parse_source(source, f"{path}.declare.{name}")
    return SegmentsSpec(frame=frame, system=system, declare=declare)
