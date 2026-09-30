"""Resolve a metric's answers to token ids, with the model's tokenizer.

An answer string is tokenized **as written** (§2.10): ``" Seattle"`` and
``"Seattle"`` are two gpt2 rows, and the table row that fixes the prompt
fixes which one the model emits. Under ``token_form: "id"`` a column holds
integer vocabulary ids instead. ``match`` accepts lists of equivalent forms;
its ``first_token`` mode credits the first token of each and refuses an
answer space whose answers share one.

[`metric_token_ids`][] resolves every answer a metric names. The score paths
([`causalab.neural.shared.metrics`][]) and the run door's pass before the
weights load ([`resolve_answers`][causalab.protocol.pipeline.resolve_answers])
both call it, so a table the door accepts scores with the same ids, and a
refusal reads the same wherever it is raised. The tokenizer is the only
service, so this module is torch-free and the protocol layer resolves
answers before any weights, as it resolves positions.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence

from causalab.protocol.results import Unavailable, unavailable
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import AggregationSpec, metric_column_fields
from causalab.protocol.schema.types import TOKEN_COLUMN_METRIC_KINDS

__all__ = [
    "column_first_token_id",
    "column_token_id",
    "column_token_ids",
    "excluded_rows",
    "metric_token_ids",
    "names_answers",
    "restrict_token_ids",
]


#: per tokenizer, the ids one answer string encodes to without special tokens.
#: A metric resolves its answer columns row by row, and every point of a
#: campaign resolves the same columns over the same rows — thousands of
#: identical tokenizer calls per scan step. The tokenizer owns its entries
#: weakly; one that cannot be weakly referenced or hashed is not memoized.
#: Least recently used goes first when the bound is reached (a hit refreshes).
_ENCODED_IDS: "weakref.WeakKeyDictionary[Any, OrderedDict[str, tuple[int, ...]]]" = (
    weakref.WeakKeyDictionary()
)
_ENCODED_IDS_PER_TOKENIZER = 4096


def _encoded_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    """``tokenizer.encode(text, add_special_tokens=False)``, memoized per
    tokenizer and text."""
    try:
        memo = _ENCODED_IDS.setdefault(tokenizer, OrderedDict())
    except TypeError:  # not weakly referenceable, or not hashable
        memo = None
    if memo is not None:
        hit = memo.get(text)
        if hit is not None:
            memo.move_to_end(text)
            return hit
    ids = tuple(int(i) for i in tokenizer.encode(text, add_special_tokens=False))
    if memo is not None:
        if len(memo) >= _ENCODED_IDS_PER_TOKENIZER:
            memo.popitem(last=False)
        memo[text] = ids
    return ids


def _split_problem(tokenizer: Any, value: Any) -> str | None:
    """Why ``value`` names no single token, or ``None`` when it names one."""
    ids = _encoded_ids(tokenizer, str(value))
    if len(ids) == 1:
        return None
    pieces = [tokenizer.decode([i]) for i in ids]
    return f"is not a single token under this tokenizer (it encodes to {pieces})"


#: what to do about a value that is not one token (§2.10 "Token forms")
_SPLIT_REMEDY = (
    "Multi-token answers have no closed metric kind in v1. An answer is "
    "tokenized as written, so check its leading space against the text it follows"
)


def _id_problem(value: Any, vocabulary_size: int) -> str | None:
    """Why ``value`` is no token id of the vocabulary, or ``None``. A bool, a
    float and a string are not ids, whatever they spell."""
    if type(value) is int and 0 <= value < vocabulary_size:
        return None
    return f"is not an integer token ID in the vocabulary of {vocabulary_size}"


#: what to do about an id outside the vocabulary
_ID_REMEDY = (
    "Under token_form='id' each value is an integer token ID below the "
    "tokenizer's vocabulary size"
)

#: how many other distinct values a refusal quotes before it counts the rest
_QUOTED = 3


def _refusal(
    where: str | None,
    values: Sequence[Any],
    problems: Sequence[tuple[int, str]],
    row_numbers: Sequence[int] | None,
    remedy: str,
) -> ProtocolError:
    """One refusal for every value the tokenizer cannot score.

    ``problems`` pairs the position in ``values`` of each such value with
    what is wrong with it. The text names the first value, its row (from
    ``row_numbers``, when the caller knows the table row of each value) and
    its problem; then how many of ``values`` fail and the first few other
    distinct ones; then ``remedy``. So one fix-and-rerun loop clears a
    column, however many of its rows fail."""
    first, problem = problems[0]
    row = f" (row {row_numbers[first]})" if row_numbers is not None else ""
    text = f"{_at(where)}metric column value {values[first]!r}{row} {problem}"
    if len(problems) > 1:
        others = list(dict.fromkeys(repr(values[i]) for i, _ in problems))[1:]
        text += f"; {len(problems)} of {len(values)} values fail this way"
        if others:
            more = len(others) - _QUOTED
            text += ", among them " + ", ".join(others[:_QUOTED])
            text += f" and {more} more" if more > 0 else ""
    return ProtocolError("P2", f"{text}. {remedy}")


def _at(where: str | None) -> str:
    """The ``where`` prefix of a refusal: the metric and field the value came
    from, when the caller knows them."""
    return f"{where}: " if where else ""


def column_token_id(
    tokenizer: Any,
    value: Any,
    *,
    token_form: str | None = None,
    where: str | None = None,
) -> int:
    """The single token id a metric column value names (module docstring).

    A string is tokenized as written (§2.10): no leading space is added or
    removed, so ``" Seattle"`` and ``"Seattle"`` resolve to different rows
    under a byte-level BPE. Under ``token_form="id"`` the value is the id.
    ``where`` names the metric and field in a refusal.
    """
    return column_token_ids(
        tokenizer, [value], token_form=token_form, where=where or ""
    )[0]


def column_token_ids(
    tokenizer: Any,
    values: Sequence[Any],
    *,
    token_form: str | None = None,
    where: str = "metric column",
    vocabulary_size: int | None = None,
    row_numbers: Sequence[int] | None = None,
) -> list[int]:
    """Resolve a whole metric column, value by value.

    ``where`` names the column in a refusal. An enclosing metric computation
    may supply its tokenizer vocabulary size. It is a call-local snapshot
    including added tokens, never a model logit width or metadata retained
    across tokenizer mutations.

    Raises:
        ProtocolError: ``P2`` naming every value the tokenizer cannot score,
            in one refusal: how many, the first with its row (from
            ``row_numbers``, the table row of each value, when given) and its
            pieces, and the other distinct ones.
    """
    if token_form == "id":
        size = len(tokenizer) if vocabulary_size is None else vocabulary_size
        problems = [
            (i, problem)
            for i, value in enumerate(values)
            if (problem := _id_problem(value, size)) is not None
        ]
        if problems:
            raise _refusal(where, values, problems, row_numbers, _ID_REMEDY)
        return list(values)
    problems = [
        (i, problem)
        for i, value in enumerate(values)
        if (problem := _split_problem(tokenizer, value)) is not None
    ]
    if problems:
        raise _refusal(where, values, problems, row_numbers, _SPLIT_REMEDY)
    return [_encoded_ids(tokenizer, str(value))[0] for value in values]


def column_first_token_id(
    tokenizer: Any, value: str, *, where: str | None = None
) -> int:
    """The first *content* token id of a value — ``match``'s ``first_token``
    mode.

    A single-token value resolves exactly as [`column_token_id`][] does, so
    ``first_token`` is a strict generalization of ``exact``. A multi-token
    value resolves to the first piece that carries text: a sentencepiece family
    can encode a leading space as its own ``▁`` piece, and crediting *that*
    would score every space-prefixed answer alike — the first piece an argmax
    can distinguish is the one after it, which is also what the model emits in
    context.

    📐 Which values trigger that is tokenizer- *and* version-dependent, so the
    skip is written as a property of the piece (does it decode to text?) rather
    than of a known value. Under transformers 4.x the tiny Llama tokenizer
    emitted the lone ``▁`` for any space-prefixed word (``" Thursday"`` →
    ``▁ Th urs day``); 5.16.1 dropped that legacy dummy prefix, so
    ``" Thursday"`` is now ``Th urs day`` — and the skip is what makes this
    function return the same id, ``Th``, across the bump. It is not dead code:
    5.16.1 still emits the lone ``▁`` whenever the first character has no
    merged ``▁X`` piece — digits, non-Latin scripts, emoji, ligatures
    (``" 3.14"`` → ``▁ 3 . 1 4``) — and byte-level BPE families still split a
    whitespace run off the front (gpt2 ``"  ?"`` → ``' '`` + ``' ?'``).

    What this cannot know is whether the table's answer space is
    first-token-distinct — two answers sharing a first piece would both score.
    That is a property of the dataset, checked where the dataset is built."""
    problem = _first_token_problem(tokenizer, value)
    if problem is not None:
        raise _refusal(where, [value], [(0, problem)], None, _FIRST_TOKEN_REMEDY)
    ids = _encoded_ids(tokenizer, value)
    if len(ids) == 1:
        return int(ids[0])
    first = _first_content_id(tokenizer, ids)
    assert first is not None  # `_first_token_problem` said there is one
    return first


def _first_token_problem(tokenizer: Any, value: Any) -> str | None:
    """Why ``value`` has no first content token, or ``None`` when it has one."""
    ids = _encoded_ids(tokenizer, str(value))
    if len(ids) == 1 or _first_content_id(tokenizer, ids) is not None:
        return None
    return "encodes to no content tokens under this tokenizer"


#: what a value with no content token leaves a ``first_token`` metric
_FIRST_TOKEN_REMEDY = "A first_token match has nothing to compare an argmax against"


def _first_content_id(tokenizer: Any, ids: Sequence[int]) -> int | None:
    """The first piece of ``ids`` that decodes to text, or ``None``.

    Written as a property of the piece rather than of a known value because
    which values carry a whitespace-only first piece is tokenizer- and
    version-dependent (see [`column_first_token_id`][]).
    """
    for token_id in ids:
        if tokenizer.decode([int(token_id)]).strip():
            return int(token_id)
    return None


def _refuse_indistinct_first_tokens(
    groups: Sequence[Sequence[str]],
    resolved: Sequence[set[int]],
    *,
    where: str,
) -> None:
    """Refuse a ``first_token`` metric whose answer space is not first-token
    distinct.

    ``first_token`` credits a *prefix*, so it means "the model answered" only
    when different answers begin with different tokens. Where they do not, the
    metric over-credits silently: ``" 85"`` tokenizes to ``[220, "8", "5"]`` on
    Qwen, so a model emitting ``87`` scores 1.000 against an expected ``85``.
    Nothing in a run's saved outputs says so — the number is simply wrong.

    The distinctness of a *dataset's* answer space is checked where the dataset
    is built ([`column_first_token_id`][]); this is the same claim asserted
    against the rows a metric was actually handed, which is the last place it
    can be checked before a number is produced.
    """
    by_first: dict[int, set[str]] = {}
    for forms, ids in zip(groups, resolved):
        answer = "/".join(sorted(form.strip() for form in forms))
        for token_id in ids:
            by_first.setdefault(token_id, set()).add(answer)
    collisions = {
        token_id: sorted(answers)
        for token_id, answers in by_first.items()
        if len(answers) > 1
    }
    if not collisions:
        return
    shown = ", ".join(
        f"{answers} share first token {token_id}"
        for token_id, answers in list(collisions.items())[:3]
    )
    raise ProtocolError(
        "P2",
        f"{where}: mode='first_token' credits a prefix, and this answer space "
        f"is not first-token distinct ({shown}) — the metric would score a "
        "wrong answer as correct. Use mode='exact', or an answer space whose "
        "members begin with different tokens",
    )


def excluded_rows(
    metric: AggregationSpec, rows: Sequence[Mapping[str, Any]], denominator_key: str
) -> dict[int, Unavailable]:
    """The rows a metric cannot score because the table carries **no answer**
    for them: for every column the kind names, a row whose value is ``null``,
    absent, or an empty list of forms (spec §2.10 "Eligibility").

    A structural fact of the data the document could not know (§4.1): the
    column exists — ``validate --data`` checked that — but this row has
    nothing in it, so the row is an **excluded measurement** under
    ``alignment_missing`` (the authored answer has no counterpart in the
    data), not a refusal of the run and not a score of ``"None"``. Keyed by
    row index; every value is the typed ``unavailable`` with the cell's
    ``denominator_key``, so it flows into the denominator unchanged. The
    ``validate --data`` twin — the *maximum* eligible count a
    ``minimum_count`` is held to — is ``loader.check_data_columns``.
    """
    kind = str(metric.kind)
    out: dict[int, Unavailable] = {}
    # the predicate `validate --data` counted the maximum eligible rows with
    # (§2.10, `metric_column_fields`): a `kl`/`js` target is a read and
    # excludes nothing, a `js` `restrict` column excludes its empty rows
    for field, column in metric_column_fields(metric).items():
        for i, row in enumerate(rows):
            if i in out:
                continue
            held = row.get(column)
            if held is None or (isinstance(held, list) and not held):
                out[i] = unavailable(
                    "alignment_missing",
                    f"row {i} carries no value in column {column!r} "
                    f"({kind}.{field}) — nothing to score it against",
                    denominator_key,
                )
    return out


def restrict_token_ids(
    metric: AggregationSpec,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    vocabulary_size: int | None = None,
    row_numbers: Sequence[int] | None = None,
) -> list[list[int]] | None:
    """The per-row answer set a ``js`` is restricted to (§2.10 ``restrict``),
    as token ids — ``None`` when the metric is unrestricted.

    A column form yields each row's own list; a literal list yields the same
    ids for every row. Every string resolves as written, exactly as an
    answer column does ([`column_token_ids`][]), and two
    strings that land on one id are refused rather than counted twice — the
    ``class_probs`` rule, for the same reason: the restricted softmax would
    give that answer double mass. Rows whose column is empty are not here:
    [`excluded_rows`][] took them out before the reduction ran.

    Under ``token_form: id`` the values are the ids themselves and every row's
    set is checked against one vocabulary bound — the enclosing metric
    computation's snapshot when it passes ``vocabulary_size``, otherwise one
    query here (the training objective's call); never one per row. A value
    the tokenizer cannot score is refused as a column's is, every such value
    of every row in one refusal (``row_numbers``: the table row of each of
    ``rows``, for its text; without it the text names no row).
    """
    restrict = metric.fields.get("restrict")
    if restrict is None:
        return None
    token_form = metric.token_form
    where = f"metric {metric.kind}.restrict"
    if token_form == "id" and vocabulary_size is None:
        vocabulary_size = len(tokenizer)

    def resolve(values: Sequence[Any]) -> list[int]:
        by_id: dict[int, Any] = {}
        for value, token in zip(
            values,
            column_token_ids(
                tokenizer,
                values,
                token_form=token_form,
                where=where,
                vocabulary_size=vocabulary_size,
            ),
        ):
            if token in by_id:
                raise ProtocolError(
                    "P2",
                    f"{where}: {value!r} and {by_id[token]!r} both resolve to token "
                    f"id {token} under this tokenizer, and the restricted softmax "
                    "would give that answer double mass. List each answer once "
                    "(§2.10)",
                )
            by_id[token] = value
        return list(by_id)

    def spelled(values: Sequence[Any]) -> list[Any]:
        # under ``id`` the values *are* the ids (a string is refused as an
        # id, `_id_problem`); every other form spells a surface string
        return list(values) if token_form == "id" else [str(v) for v in values]

    if isinstance(restrict, str):
        members = [spelled(row[restrict]) for row in rows]
        # every row's members at once, so the refusal names them all; the
        # per-row pass below then resolves from the memo
        column_token_ids(
            tokenizer,
            [value for values in members for value in values],
            token_form=token_form,
            where=where,
            vocabulary_size=vocabulary_size,
            row_numbers=_per_form(row_numbers, members),
        )
        return [resolve(values) for values in members]
    assert isinstance(restrict, (list, tuple))  # the parse guarantees the shape
    shared = resolve(spelled(restrict))
    return [list(shared) for _ in rows]


#: The kinds whose value fields each name one answer column, per row, in the
#: order they resolve. The score path's gathered kinds are among them;
#: ``cross_entropy`` reads its column the same way but reduces the whole
#: vocabulary.
_COLUMN_FIELDS: Mapping[str, tuple[str, ...]] = {
    "logit_diff": ("a", "b"),
    "soft_accuracy": ("a", "b"),
    "token_logit": ("token",),
    "cross_entropy": ("target",),
}


def names_answers(metric: AggregationSpec) -> bool:
    """Whether ``metric`` names answers the tokenizer resolves: the
    token-column kinds, and a ``js`` with a ``restrict`` set (§2.10). For
    every other kind [`metric_token_ids`][] resolves nothing."""
    kind = str(metric.kind)
    return kind in TOKEN_COLUMN_METRIC_KINDS or (
        kind == "js" and metric.fields.get("restrict") is not None
    )


def metric_token_ids(
    metric: AggregationSpec,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    row_numbers: Sequence[int] | None = None,
) -> dict[str, list[Any]]:
    """Every answer ``metric`` names, resolved to token ids exactly as its
    score resolves them. The score paths and the run door's pass before the
    weights load (``pipeline.resolve_answers``) share this one resolution.

    The rows that carry no answer ([`excluded_rows`][]) are left out, as
    [`compute_metric`][causalab.neural.shared.metrics.compute_metric] leaves
    them out before it resolves. Strings are tokenized as written; under
    ``token_form: "id"`` the values are the ids, held to the tokenizer's
    vocabulary. The keys are the fields:

    * ``a``, ``b``, ``token``, ``target``: one id per kept row;
    * ``expected`` (``match``): the set of ids a kept row credits;
    * ``restrict`` (a restricted ``js``): the ids of a kept row's answer set;
    * ``groups.<name>`` (``class_probs``) and ``tokens`` (``token_logits``):
      the ids of the literal strings, once for the run.

    A kind that names no answer (``kl``, an unrestricted ``js``, ``top_k``,
    ``decode``) resolves to ``{}``. A caller whose rows never change (a
    fit's eval executor) resolves once and hands the result to
    [`gathered_metric`][causalab.neural.shared.metrics.gathered_metric] on
    every pass.

    ``row_numbers`` is the table row of each of ``rows``, for the refusal
    text. The run door knows it and passes it: it hands the rows a metric
    scores, which need not be the whole table. A score path is handed a
    batch of base rows whose table rows it does not know, so it passes
    none, and its refusal names no row. A position in ``rows`` is not named
    as a row.

    Raises:
        ProtocolError: ``P2`` for the values the tokenizer cannot score: a
            string that is not one token, an id outside the vocabulary, two
            literals on one id, or a ``first_token`` answer space whose
            answers share a first token. Every failing value of a field is
            named in one line, with the count and, given ``row_numbers``, the
            first table row with it. Every failing field of the metric has its
            own line.
    """
    excluded = excluded_rows(metric, rows, str(metric.kind))
    kept = [i for i in range(len(rows)) if i not in excluded]
    return _answer_ids(
        metric,
        [rows[i] for i in kept],
        tokenizer,
        None if row_numbers is None else [row_numbers[i] for i in kept],
    )


def _each_field(
    resolve: Mapping[str, Callable[[], list[Any]]],
) -> dict[str, list[Any]]:
    """Every field's ids, from its resolver, or one refusal with a line per
    field that fails (each line the field's own refusal)."""
    out: dict[str, list[Any]] = {}
    refused: list[ProtocolError] = []
    for field, ids in resolve.items():
        try:
            out[field] = ids()
        except ProtocolError as err:
            refused.append(err)
    if refused:
        raise ProtocolError(refused[0].code, "\n".join(err.message for err in refused))
    return out


def _answer_ids(
    metric: AggregationSpec,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    row_numbers: Sequence[int] | None,
) -> dict[str, list[Any]]:
    """[`metric_token_ids`][] over rows that all carry their answers. Under
    ``token_form: "id"`` the vocabulary size is read once, here."""
    kind = str(metric.kind)
    token_form = metric.token_form
    vocabulary_size = len(tokenizer) if token_form == "id" else None
    if kind in _COLUMN_FIELDS:

        def column(field: str) -> Callable[[], list[Any]]:
            values: list[Any] = [row[str(metric.fields[field])] for row in rows]
            if token_form != "id":
                values = [str(value) for value in values]
            return lambda: column_token_ids(
                tokenizer,
                values,
                token_form=token_form,
                where=f"metric {kind}.{field}",
                vocabulary_size=vocabulary_size,
                row_numbers=row_numbers,
            )

        return _each_field({field: column(field) for field in _COLUMN_FIELDS[kind]})
    if kind == "match":
        return {
            "expected": _match_expected_ids(
                metric,
                rows,
                tokenizer,
                token_form=token_form,
                vocabulary_size=vocabulary_size,
                row_numbers=row_numbers,
            )
        }
    if kind == "js":
        restricted = restrict_token_ids(
            metric,
            rows,
            tokenizer,
            vocabulary_size=vocabulary_size,
            row_numbers=row_numbers,
        )
        return {} if restricted is None else {"restrict": restricted}
    if kind == "class_probs":
        # ⚠️ `groups` is the one value field that is NOT a column name: a class
        # is a property of the answer *space*, one for the whole run, so the
        # members are literal token strings (§2.10). Every other kind's `a`,
        # `b`, `token`, `target`, `expected` name a dataset column.
        groups = metric.fields["groups"]
        if not isinstance(groups, Mapping):
            raise ProtocolError(
                "P2",
                "class_probs groups is a {name: [token strings]} mapping — "
                "literal tokens, not a dataset column name",
            )

        def group(name: str, members: Sequence[Any]) -> Callable[[], list[Any]]:
            return lambda: _distinct_token_ids(
                tokenizer,
                [str(v) for v in members],
                where=f"metric {kind}.groups.{name}",
                consequence="this metric sums a group's ids — so the class would "
                "count that token twice and report a 'probability' above 1",
            )

        return _each_field(
            {f"groups.{name}": group(name, members) for name, members in groups.items()}
        )
    if kind == "token_logits":
        # `tokens` is literal token strings for the same reason `groups` is:
        # the answer space is a property of the run, not of a row (§2.10)
        tokens = metric.fields["tokens"]
        assert isinstance(tokens, (list, tuple))  # the parse guarantees the shape
        return {
            "tokens": _distinct_token_ids(
                tokenizer,
                [str(v) for v in tokens],
                where=f"metric {kind}.tokens",
                consequence="this metric reports one logit per listed token — so "
                "one row of the projection would appear twice under two names",
            )
        }
    return {}


def _distinct_token_ids(
    tokenizer: Any, values: Sequence[str], *, where: str, consequence: str
) -> list[int]:
    """Resolve a list of *literal* token strings, refusing when two of them
    land on one id.

    The list kinds (``class_probs``, ``token_logits``) index the projection
    by every member, so a collision is not harmless redundancy:
    ``consequence`` says what the kind would have reported. ``['X', ' X']``
    is two rows under a byte-level BPE and one piece under a sentencepiece
    family that folds the space in — only the tokenizer can tell, which is
    why the check lives here (§2.10).
    """
    by_id: dict[int, str] = {}
    for value, token in zip(values, column_token_ids(tokenizer, values, where=where)):
        if token in by_id:
            raise ProtocolError(
                "P2",
                f"{where}: {value!r} and {by_id[token]!r} both resolve to token "
                f"id {token} under this tokenizer, and {consequence}. List each "
                "answer once (§2.10)",
            )
        by_id[token] = value
    return list(by_id)


def _match_expected_ids(
    metric: AggregationSpec,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    token_form: str | None,
    vocabulary_size: int | None,
    row_numbers: Sequence[int] | None,
) -> list[set[int]]:
    """Per row, the token ids a ``match`` metric credits (§2.10): the
    ``expected`` column's forms, each tokenized as written (so ``[" X", "X"]``
    credits two rows), each form's first content token under
    ``mode: first_token``. Every row here carries an
    answer — [`excluded_rows`][] took the others out. Every form of every
    row is checked before any row resolves, so one refusal names every form
    the tokenizer cannot score."""
    kind = str(metric.kind)
    column = str(metric.fields["expected"])
    where = f"metric {kind}.expected"
    if token_form == "id":
        id_groups: list[list[Any]] = []
        for value in (row[column] for row in rows):
            forms = value if isinstance(value, list) else [value]
            if not forms:
                raise ProtocolError("P2", "expected token ID group must not be empty")
            id_groups.append(list(forms))
        column_token_ids(
            tokenizer,
            [form for forms in id_groups for form in forms],
            token_form=token_form,
            where=where,
            vocabulary_size=vocabulary_size,
            row_numbers=_per_form(row_numbers, id_groups),
        )
        return [set(forms) for forms in id_groups]
    # `mode` decides whether a form's first token counts (§2.10)
    mode = str(metric.fields.get("mode", "exact"))
    # one row's expected forms: a list column is a group of equivalent
    # surface forms, a scalar is a group of one (§2.10)
    groups = [
        [str(v) for v in value] if isinstance(value, list) else [str(value)]
        for value in (row[column] for row in rows)
    ]
    flat = [form for forms in groups for form in forms]
    problem, remedy = (
        (_first_token_problem, _FIRST_TOKEN_REMEDY)
        if mode == "first_token"
        else (_split_problem, _SPLIT_REMEDY)
    )
    problems = [
        (i, found)
        for i, form in enumerate(flat)
        if (found := problem(tokenizer, form)) is not None
    ]
    if problems:
        raise _refusal(where, flat, problems, _per_form(row_numbers, groups), remedy)
    resolve = column_first_token_id if mode == "first_token" else column_token_id
    resolved = [{resolve(tokenizer, f, where=where) for f in forms} for forms in groups]
    if mode == "first_token":
        _refuse_indistinct_first_tokens(groups, resolved, where=f"metric {kind}")
    return resolved


def _per_form(
    row_numbers: Sequence[int] | None, groups: Sequence[Sequence[Any]]
) -> list[int] | None:
    """The table row of each form, when each row holds a group of forms, or
    ``None`` when the caller knows no table rows."""
    if row_numbers is None:
        return None
    return [number for number, forms in zip(row_numbers, groups) for _ in forms]
