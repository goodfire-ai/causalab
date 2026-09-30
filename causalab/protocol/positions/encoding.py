"""Tokenize rows into padded position frames.

``PositionFrame`` stores token IDs, masks, character offsets, and prefix lengths
as Python values. Padding is on the left. Chat frames use the tokenizer's chat
template and record its prefix length. Resolvers map document positions to
indices in this frame and check their bounds.

The caller supplies the tokenizer. Device tensors are created by the neural layer."""

from __future__ import annotations

import dataclasses
import re
import weakref
from collections import OrderedDict
from typing import Any, Mapping, Protocol, Sequence

from causalab.protocol.positions.alignment import alignment_of, refuse_unalignable
from causalab.protocol.positions.spans import constituents, resolve_span
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import (
    Document,
    PositionSpec,
    SpanSpec,
    concrete_int,
)

__all__ = [
    "Frame",
    "PositionFrame",
    "candidate_runs",
    "column_value",
    "constituent_candidate_runs",
    "encode",
    "first_real_indices",
    "generated_budget",
    "refuse_double_bos",
    "refuse_empty_rows",
    "resolve_position",
    "select_field",
    "variable_value",
]

Rows = tuple[tuple[int, ...], ...]
Offsets = tuple[tuple[tuple[int, int], ...], ...]
Segments = tuple[Mapping[str, tuple[tuple[int, int], ...]], ...]


def _rows(value: Any) -> list[list[Any]]:
    """A ``(rows, width)`` table as nested lists, whatever the tokenizer
    handed back — Python lists (the default), or a host tensor (a caller
    that asked for ``return_tensors``, or a test's stand-in), read once."""
    if hasattr(value, "tolist"):
        return [list(row) for row in value.tolist()]
    return [list(row) for row in value]


class Frame(Protocol):
    """What position resolution reads off a batch — the attributes a
    [`PositionFrame`][] (protocol-side, ints) and the engine's
    ``EncodedBatch`` (the same frame with device tensors beside it) share.
    Every resolver in this module takes one, so the two layers resolve a
    position with the same function."""

    @property
    def texts(self) -> tuple[str, ...]: ...

    @property
    def offset_mapping(self) -> Offsets: ...

    @property
    def prefix_lengths(self) -> tuple[int, ...]: ...

    @property
    def segments(self) -> Segments: ...

    @property
    def padded_len(self) -> int: ...

    def first_real(self, row: int) -> int: ...

    def content_start(self, row: int) -> int: ...

    def row_ids(self, row: int) -> tuple[int, ...]: ...


@dataclasses.dataclass(frozen=True)
class PositionFrame:
    """One left-padded batch plus its position frame, as Python ints.

    ``token_ids`` and ``attention_mask`` are the padded rows, one width;
    ``offset_mapping`` each token's ``[start, end)`` in its row's text
    (``(0, 0)`` for padding and specials); ``prefix_lengths`` the chat-prefix
    token count per row (0 in the plain frame); ``segments`` per row each
    declared segment's **candidate** char spans in ``texts[row]`` (§2.2.1):
    none is ``absent``, several ``ambiguous``, exactly one is the segment —
    empty for a document with no ``segments`` section.
    """

    texts: tuple[str, ...]
    token_ids: Rows
    attention_mask: Rows
    offset_mapping: Offsets
    prefix_lengths: tuple[int, ...]  # chat-prefix token counts; 0 = plain text
    segments: Segments = ()
    #: Per row, the padded index of its first real token — the first ``1`` of
    #: the mask row (0 for a row with none, as ``argmax`` of zeros reads).
    #: Derived from ``attention_mask`` when left empty; [`select`][] passes
    #: its own through, since a row's index does not change when it travels.
    first_reals: tuple[int, ...] = dataclasses.field(
        default=(), compare=False, repr=False
    )

    def __post_init__(self) -> None:
        derived = first_real_indices(self.attention_mask)
        if not self.first_reals:
            object.__setattr__(self, "first_reals", derived)
            return
        if len(self.first_reals) != len(self.attention_mask):
            raise ValueError(
                f"first_reals carries {len(self.first_reals)} entries for a "
                f"{len(self.attention_mask)}-row attention mask"
            )
        if self.first_reals != derived:
            raise ValueError(
                "first_reals disagrees with attention_mask — a frame whose mask "
                "was replaced must leave the cache empty to be re-derived"
            )

    @property
    def padded_len(self) -> int:
        return len(self.token_ids[0]) if self.token_ids else 0

    def first_real(self, row: int) -> int:
        """First real token of ``row`` in the padded frame — past the left
        padding, any chat prefix included."""
        return self.first_reals[row]

    def content_start(self, row: int) -> int:
        """First real token of ``row`` in the padded frame, past any prefix."""
        return self.first_real(row) + self.prefix_lengths[row]

    def row_ids(self, row: int) -> tuple[int, ...]:
        """The padded ids of one row — what the ledger reads a token id off."""
        return self.token_ids[row]

    def real_ids(self, row: int) -> tuple[int, ...]:
        """``row``'s real token ids, padding dropped."""
        return tuple(
            token
            for token, keep in zip(self.token_ids[row], self.attention_mask[row])
            if keep
        )

    def select(self, indices: Sequence[int]) -> "PositionFrame":
        """The rows ``indices`` of this frame, in that order, **in this frame**:
        the same padded width, every per-row field sliced in step — so a
        row's position indices, resolved against the frame, hold whichever
        selection it travels in (spec §4, "Cohorts")."""
        if not indices:
            raise ValueError("a selection names at least one row")
        rows = list(indices)
        return PositionFrame(
            texts=tuple(self.texts[i] for i in rows),
            token_ids=tuple(self.token_ids[i] for i in rows),
            attention_mask=tuple(self.attention_mask[i] for i in rows),
            offset_mapping=tuple(self.offset_mapping[i] for i in rows),
            prefix_lengths=tuple(self.prefix_lengths[i] for i in rows),
            segments=tuple(self.segments[i] for i in rows) if self.segments else (),
            first_reals=tuple(self.first_reals[i] for i in rows),
        )


def first_real_indices(attention_mask: Any) -> tuple[int, ...]:
    """Per row of a left-padded mask, the index of its first real token: the
    first ``1`` of the row (0 for a row with no real token, as ``argmax`` of
    zeros reads). Host arithmetic over ints; the engine's tensor-frame twin
    (``neural/shared/encoding.first_real_indices``) is one reduction and one
    host read for the whole batch."""
    out: list[int] = []
    for row in _rows(attention_mask):
        first = 0
        for index, keep in enumerate(row):
            if keep:
                first = index
                break
        out.append(first)
    return tuple(out)


def encode(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    add_special_tokens: bool = True,
    segments: Sequence[Mapping[str, tuple[tuple[int, int], ...]]] = (),
) -> PositionFrame:
    """Tokenize one batch with the one convention both engines run: left
    padding, special tokens as the tokenizer defines them, offset mapping
    kept for char→token position resolution.

    ``prefix_lengths`` is ``0`` for every row **here**: this is the plain-text
    frame, and every document without a ``segments`` section takes it. The
    chat frame (§2.2.1, [`encode_framed`][causalab.protocol.positions.framing.encode_framed]) renders each row through the tokenizer's own chat
    template, encodes the rendered text with ``add_special_tokens=False``
    (the template owns its specials) and sets the real prefix length from
    where the user turn was located. A dataset that bakes a *rendered*
    template into its ``input`` column under the plain frame still works —
    but this call then adds special tokens as the tokenizer defines them, so
    a rendered template that already opens with BOS gets a second one. A
    double BOS is a wrong number, not a crash: every position shifts by one
    and nothing says so. [`refuse_double_bos`][] catches it here, which is
    the only place that can see both halves.

    ``segments`` is the per-row segment location table the frame computed
    ([`PositionFrame.segments`][]); the plain frame passes none.
    """
    tokenized = _tokenized(tokenizer, tuple(texts), add_special_tokens)
    # the two refusals run on every call, memo hit or miss, so a frame costs
    # the same however it was produced
    refuse_empty_rows(texts, tokenized.attention_mask)
    refuse_double_bos(tokenizer, tokenized.input_ids, tokenized.attention_mask)
    return PositionFrame(
        texts=tuple(texts),
        token_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        offset_mapping=tokenized.offset_mapping,
        prefix_lengths=tuple(0 for _ in texts),
        segments=tuple(dict(row) for row in segments),
    )


@dataclasses.dataclass(frozen=True)
class _Tokenized:
    """The tokenizer's output for one ``(texts, specials)``: the padded ids,
    mask and char offsets as tuples of ints — the tokenizer call itself, which
    depends on nothing but the tokenizer and the texts. What a frame derives
    from them (the first-real cache, the two refusals) is derived per call."""

    input_ids: Rows
    attention_mask: Rows
    offset_mapping: Offsets


#: per tokenizer, the most recently used `_Tokenized` results by
#: ``(texts, add_special_tokens, padding_side, pad_token_id)``. Every point of
#: a campaign encodes the same rows of the same roles into its own executor —
#: ten to twenty identical tokenizer calls per step on the standard workflow,
#: each a few milliseconds of Rust — and a fit's eval executors encode the
#: same split once per fit. The tokenizer is the key's owner (weakly, so a
#: tokenizer that is released takes its entries with it); a tokenizer that
#: cannot be weakly referenced or hashed is simply not memoized. The padding
#: side and pad id are in the key because they are the tokenizer state the
#: padded output depends on. An entry is a whole padded batch — ids, mask and
#: the ``rows × width × 2`` offsets, several MB for a fit's eval split — so
#: the bound is small: a step's working set is its roles × frames, single
#: digits, and a hit refreshes its entry (least recently used goes first).
_TOKENIZED: "weakref.WeakKeyDictionary[Any, OrderedDict[tuple[Any, ...], _Tokenized]]" = weakref.WeakKeyDictionary()
_TOKENIZED_PER_TOKENIZER = 16


def _tokenized(
    tokenizer: Any, texts: tuple[str, ...], add_special_tokens: bool
) -> _Tokenized:
    try:
        memo = _TOKENIZED.setdefault(tokenizer, OrderedDict())
    except TypeError:  # not weakly referenceable, or not hashable
        memo = None
    key = (
        texts,
        add_special_tokens,
        getattr(tokenizer, "padding_side", None),
        getattr(tokenizer, "pad_token_id", None),
    )
    if memo is not None:
        hit = memo.get(key)
        if hit is not None:
            memo.move_to_end(key)
            return hit
    enc = tokenizer(
        list(texts),
        padding=True,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    tokenized = _Tokenized(
        input_ids=tuple(tuple(int(t) for t in row) for row in _rows(enc["input_ids"])),
        attention_mask=tuple(
            tuple(int(m) for m in row) for row in _rows(enc["attention_mask"])
        ),
        offset_mapping=tuple(
            tuple((int(a), int(b)) for a, b in row)
            for row in _rows(enc["offset_mapping"])
        ),
    )
    if memo is not None:
        if len(memo) >= _TOKENIZED_PER_TOKENIZER:
            memo.popitem(last=False)
        memo[key] = tokenized
    return tokenized


def refuse_empty_rows(texts: Sequence[str], attention_mask: Any) -> None:
    """Refuse a batch with a row that encodes to no token at all.

    Every position of such a row would address padding, and the frame
    arithmetic assumes each row has a first real token: its cached index is 0
    for an empty row as for a full one, so the two would share a mask
    signature. An empty text (a tokenizer adding no special token) is a data
    error, named by row rather than run. ``attention_mask`` is the padded
    mask as rows of ints (or a host tensor of them)."""
    for row, mask in enumerate(_rows(attention_mask)):
        if sum(mask) == 0:
            raise ProtocolError(
                "P2",
                f"row {row} ({texts[row]!r}) encodes to no token: a frame's row "
                "has at least one real token, or every position in it would "
                "address padding",
            )


def refuse_double_bos(tokenizer: Any, input_ids: Any, attention_mask: Any) -> None:
    """Refuse a batch whose rows start with the BOS token twice.

    Read off the encoded batch rather than by re-encoding: the ids are already
    computed, and "two BOS at the start of the content" is the exact condition
    — it needs no assumption about whether *this* tokenizer prepends one, or
    about which string spells it.

    Stripping instead of refusing was considered and rejected: the text is the
    document's data, its content digest is part of the canonical form (§7), and
    silently editing it would make the digest describe bytes that never ran.
    """
    bos_id = getattr(tokenizer, "bos_token_id", None)
    ids = _rows(input_ids)
    if bos_id is None or not ids or len(ids[0]) < 2:
        return
    width = len(ids[0])
    for row, start in enumerate(first_real_indices(attention_mask)):
        if start + 1 >= width:
            continue
        if ids[row][start] == bos_id == ids[row][start + 1]:
            bos = getattr(tokenizer, "bos_token", None) or f"id {bos_id}"
            raise ProtocolError(
                "P2",
                f"row {row} begins with {bos!r} twice: the text already carries "
                "a BOS — a rendered chat template, most likely — and the "
                "tokenizer added another. Every position in the row is then "
                "off by one and no error would be raised. Remove the leading "
                f"{bos!r} from the dataset's text; v1 has no chat field, so "
                "the rendered template is the data",
            )


_LIST_FIELD = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]$")


def select_field(row: Mapping[str, Any], field: str) -> Any:
    """Apply a data-role ``field`` selector (§2.2): a column name, with
    ``[j]`` indexing list-valued columns."""
    match = _LIST_FIELD.match(field)
    if match is None:
        if field not in row:
            raise ProtocolError(
                "P2", f"row has no column {field!r} (has {sorted(row)})"
            )
        return row[field]
    column, index = match.group(1), int(match.group(2))
    values = row.get(column)
    if not isinstance(values, list) or index >= len(values):
        raise ProtocolError("P2", f"column {column!r} has no element [{index}]")
    return values[index]


def variable_value(row: Mapping[str, Any], field: str, variable: str) -> str:
    """The row's value for a prompt variable, for the text selected by
    ``field`` (module docstring: ``<col>_variables`` sibling first, plain
    column fallback)."""
    match = _LIST_FIELD.match(field)
    column = match.group(1) if match else field
    sibling = row.get(f"{column}_variables")
    if match and isinstance(sibling, list):
        index = int(match.group(2))
        if index < len(sibling) and isinstance(sibling[index], Mapping):
            sibling = sibling[index]
        else:
            sibling = None
    if isinstance(sibling, Mapping) and variable in sibling:
        return str(sibling[variable])
    if variable in row:
        return str(row[variable])
    raise ProtocolError(
        "P2",
        f"no value for prompt variable {variable!r}: neither {column}_variables "
        f"nor a {variable!r} column exists in the dataset row",
    )


def column_value(row: Mapping[str, Any], column: str) -> str:
    """The row's value for a ``column`` position (§2.3) — a top-level column
    only, never the per-role ``<field>_variables`` sibling, so the same
    reference resolves to the same string whichever role reads it."""
    if column not in row:
        raise ProtocolError(
            "P2",
            f"position column {column!r} is not a column of the dataset row "
            f"(has {sorted(row)})",
        )
    value = row[column]
    if not isinstance(value, str):
        raise ProtocolError(
            "P2",
            f"position column {column!r} holds {type(value).__name__}, not a "
            "string — v1 column positions resolve a substring of the row's "
            "text (§2.3)",
        )
    return value


def _variable_token_runs(batch: Frame, row: int, value: str) -> tuple[list[int], ...]:
    """The padded-frame token runs covering **each** occurrence of ``value``
    in the row's text — the *candidate* runs one address has here, via the
    offset mapping ((0, 0) entries are specials/padding and never match).

    How many candidates there are is the address's cardinality on this input
    ([`alignment_of`][], §2.3): none is
    ``absent``, several is ``ambiguous``, exactly one is the run. An
    occurrence that overlaps no token is kept as an empty run, so it too
    reads as ``absent`` rather than as a second candidate.
    """
    text = batch.texts[row]
    return tuple(
        _chars_to_tokens(batch, row, match.start(), match.end())
        for match in re.finditer(re.escape(value), text)
    )


def _chars_to_tokens(batch: Frame, row: int, lo: int, hi: int) -> list[int]:
    """The padded-frame tokens overlapping the char span ``[lo, hi)`` of the
    row's text, via the offset mapping ((0, 0) entries are padding / specials
    the text does not spell and never match)."""
    return [
        idx
        for idx, (a, b) in enumerate(batch.offset_mapping[row])
        if not (a == 0 and b == 0) and a < hi and b > lo
    ]


def _segment_token_runs(batch: Frame, row: int, name: str) -> tuple[list[int], ...]:
    """The candidate token runs of a declared segment in this row — one per
    char span the frame located it at (§2.2.1). The frame that encoded the
    batch located every declared segment; a batch with no location table is a
    plain-frame batch under a document that never declared one."""
    if row >= len(batch.segments) or name not in batch.segments[row]:
        raise ProtocolError(
            "P2",
            f"segment {name!r} was not located on this batch — the document "
            "declares no segments section, or the frame that encoded it never "
            "declared this name (§2.2.1)",
        )
    return tuple(
        _chars_to_tokens(batch, row, lo, hi) for lo, hi in batch.segments[row][name]
    )


def _segment_run(batch: Frame, row: int, name: str) -> list[int]:
    """The one run a declared segment has in this row, or the typed refusal —
    ``absent`` is ``alignment_missing``, ``ambiguous`` is
    ``alignment_ambiguous`` — through the same path a ``variable`` takes."""
    runs = _segment_token_runs(batch, row, name)
    observed = alignment_of(runs)
    refuse_unalignable(
        observed,
        f"segment {name!r} occurs {len(runs)} time(s) in the rendered text of "
        f"row {row} ({batch.texts[row]!r}) — a segment anchor needs exactly one "
        f"occurrence, and this one is {observed!r} here",
    )
    return runs[0]


def _unique_run(batch: Frame, row: int, value: str, what: str) -> list[int]:
    """The one run ``value`` has in this row, or the typed refusal for none
    or several — ``absent`` is ``alignment_missing``, ``ambiguous`` is
    ``alignment_ambiguous`` (§2.3, §2.4). The executor turns that refusal
    into an ``unavailable`` cell for a read (§4.1) and lets it stand for a
    write, which cannot skip a row and still report a number."""
    runs = _variable_token_runs(batch, row, value)
    observed = alignment_of(runs)
    refuse_unalignable(
        observed,
        f"{what} value {value!r} occurs {len(runs)} times in {batch.texts[row]!r} "
        f"(row {row}) — position resolution needs exactly one occurrence, and "
        f"this address is {observed!r} here",
    )
    return runs[0]


def _row_value(
    dataset_row: Mapping[str, Any] | None,
    field: str | None,
    name: str,
    *,
    from_column: bool,
) -> str:
    """The row's string for an anchor or an anchor-free reference — a
    top-level column (``column``) or a per-role prompt variable
    (``variable``), §2.3."""
    if dataset_row is None:
        raise ProtocolError("P2", "variable/column positions need a dataset row")
    if from_column:
        return column_value(dataset_row, name)
    if field is None:
        raise ProtocolError("P2", "variable positions need a dataset row")
    return variable_value(dataset_row, field, name)


def candidate_runs(
    spec: PositionSpec,
    batch: Frame,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> tuple[list[int], ...]:
    """The candidate runs one prompt-frame spec has in one row — what
    [`alignment_of`][] classifies when a
    declared ``alignment`` is checked against the pair (§2.3).

    A ``variable`` / ``column`` address has one candidate per occurrence of
    its value; an anchored ``index`` / ``span`` inherits its anchor's
    candidates when the anchor is not unique (the derived address is exactly
    as ambiguous as the anchor); everything else has the one run
    [`resolve_position`][] returns. A ``generated`` spec has no candidates
    here: the continuation is a result, not one of the pair's inputs.
    """
    if spec.generated is not None:
        raise ProtocolError(
            "P2",
            "a generated position has no pair alignment — the continuation is a "
            "result, not one of the pair's inputs (§2.3)",
        )
    if isinstance(spec, SpanSpec):
        # A whole-segment span has one candidate per occurrence of the segment;
        # every other span — atomic or not — is the one run its algebra
        # resolves to (its constituents are classified through
        # [`constituent_candidate_runs`][]).
        if spec.segment is not None:
            return _segment_token_runs(batch, row, spec.segment)
        return (
            resolve_position(spec, batch, row, dataset_row=dataset_row, field=field),
        )
    if spec.variable is not None:
        value = _row_value(dataset_row, field, str(spec.variable), from_column=False)
        return _variable_token_runs(batch, row, value)
    if spec.column is not None:
        value = _row_value(dataset_row, field, str(spec.column), from_column=True)
        return _variable_token_runs(batch, row, value)
    if spec.scope is not None or spec.relative_to is not None:
        anchor_name = str(spec.scope or spec.relative_to)
        if spec.anchor_source == "segment":
            anchors = _segment_token_runs(batch, row, anchor_name)
        else:
            anchor = _row_value(
                dataset_row,
                field,
                anchor_name,
                from_column=spec.anchor_source == "column",
            )
            anchors = _variable_token_runs(batch, row, anchor)
        if len(anchors) != 1:
            return anchors
    return (resolve_position(spec, batch, row, dataset_row=dataset_row, field=field),)


def constituent_candidate_runs(
    spec: PositionSpec,
    batch: Frame,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> list[tuple[list[int], ...]]:
    """The candidate runs of each address a spec is classified as (§2.3):
    one entry for an ordinary position or an ``atomic`` span, one per
    constituent of a non-atomic set (``spans.constituents``) — the
    "composable groups" half: the same two tokens are one joint
    ``one_to_one`` address when atomic and two single-token addresses when
    not, and a declared ``alignment`` is checked against each."""
    return [
        candidate_runs(part, batch, row, dataset_row=dataset_row, field=field)
        for part in constituents(spec)
    ]


def resolve_position(
    spec: PositionSpec,
    batch: Frame,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
) -> list[int]:
    """Resolve one prompt-frame position spec for one row into padded-frame
    indices. A ``generated`` spec addresses the decode's continuation, a
    frame that exists only once a model has run: the engine resolves it
    (``neural/shared/generated.resolve_steps``), never this layer."""
    if spec.generated is not None:
        raise ProtocolError(
            "P2",
            "a generated position needs the decode's continuation — the "
            "frame it addresses does not exist until the model has run",
        )
    padded = batch.padded_len
    start = batch.content_start(row)

    def check(indices: list[int]) -> list[int]:
        bad = [
            i for i in indices if not start - batch.prefix_lengths[row] <= i < padded
        ]
        if bad:
            raise ProtocolError(
                "P2",
                f"resolved position(s) {bad} out of bounds for row {row} "
                f"(content [{start}, {padded}) in the padded frame) — refusing "
                "rather than addressing the wrong token",
            )
        return indices

    if isinstance(spec, SpanSpec):
        # the span algebra (protocol/spans.py) is torch-free and pure over this
        # row's frame; members and anchors come back through this resolver
        return check(
            resolve_span(
                spec,
                frame=(batch.first_real(row), start, padded),
                resolve=lambda member: resolve_position(
                    member, batch, row, dataset_row=dataset_row, field=field
                ),
                segment_run=lambda name: _segment_run(batch, row, name),
                where=f"row {row}",
            )
        )

    anchor_run: list[int] | None = None
    if spec.scope is not None or spec.relative_to is not None:
        anchor_name = str(spec.scope or spec.relative_to)
        if spec.anchor_source == "segment":
            anchor_run = _segment_run(batch, row, anchor_name)
        else:
            anchor_value = _row_value(
                dataset_row,
                field,
                anchor_name,
                from_column=spec.anchor_source == "column",
            )
            anchor_run = _unique_run(
                batch, row, anchor_value, f"anchor {anchor_name!r}"
            )

    if spec.all is not None:
        # content_start is already past the pad and any chat prefix; left
        # padding right-aligns content, so the row runs to the padded end
        return check(list(range(start, padded)))

    if spec.variable is not None:
        value = _row_value(dataset_row, field, str(spec.variable), from_column=False)
        return check(_unique_run(batch, row, value, "prompt variable"))

    if spec.column is not None:
        value = _row_value(dataset_row, field, str(spec.column), from_column=True)
        return check(_unique_run(batch, row, value, "position column"))

    if spec.index is not None:
        n = concrete_int(spec.index, "position index")
        if spec.relative_to is not None:
            assert anchor_run is not None
            if n == 0:
                raise ProtocolError(
                    "P2", "relative_to index 0 is ambiguous — use scope"
                )
            target = anchor_run[-1] + n if n > 0 else anchor_run[0] + n
            return check([target])
        if spec.scope is not None:
            assert anchor_run is not None
            if not -len(anchor_run) <= n < len(anchor_run):
                raise ProtocolError(
                    "P2",
                    f"index {n} outside the {len(anchor_run)}-token variable window",
                )
            return check([anchor_run[n]])
        if n < 0:
            return check([padded + n])
        return check([start + n])

    assert spec.span is not None
    span = spec.span
    if not isinstance(span, tuple) or len(span) != 2:
        raise ProtocolError("P2", f"span is not concrete: {span!r}")
    a, b = (int(v) for v in span)
    if spec.scope is not None:
        assert anchor_run is not None
        window = anchor_run[a:b]
        if not window:
            raise ProtocolError(
                "P2", f"span [{a}, {b}) is empty inside the variable window"
            )
        return check(window)
    if a < 0 or b <= a:
        raise ProtocolError("P2", f"span [{a}, {b}) is not a forward window")
    return check(list(range(start + a, start + b)))


def generated_budget(doc: Document, pos: Any) -> int | None:
    """The decode budget of a position, or ``None`` for the prompt frame.

    Takes the spelling a read carries (a positions-table name or an inline
    spec) and returns the concrete budget — points are concrete by the time
    they are planned, so a surviving sweep wrapper is a caller error."""
    spec = doc.positions.get(pos) if isinstance(pos, str) else pos
    if not isinstance(spec, PositionSpec) or spec.generated is None:
        return None
    return concrete_int(spec.generated["max_new_tokens"], "generated.max_new_tokens")
