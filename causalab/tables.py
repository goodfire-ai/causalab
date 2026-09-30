"""The serialized table format, shared by the two layers that touch it.

A dataset table is written by [`causalab.tasks.serialize`][] and read back by
[`causalab.io.env.FileDatasets`][]. Those two layers are
deliberately decoupled — ``causalab.protocol`` never imports ``causalab.tasks``
and ``causalab.tasks`` never imports ``causalab.protocol``, which is what keeps
resolution stdlib-only and a document's digest independent of task code or a
tokenizer (§2.2).

They still have to agree on exactly two things:

* **The bytes a table serializes to.** The content digest stamped into a
  canonical form (§7) is taken over them, so writer and reader must produce
  byte-identical output from the same rows.
* **The name of the column that declares a row's split.** The writer stamps it;
  the resolver selects on it.

Both live here, once, so that agreement is a shared import rather than two
copies free to drift. This module imports nothing but the standard library, so
either layer can depend on it without acquiring the other's dependencies.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

__all__ = [
    "INLINE_REF_PREFIX",
    "INPUT_COLUMN",
    "SPLIT_COLUMN",
    "inline_ref",
    "inline_rows",
    "inline_table",
    "is_inline_ref",
    "table_bytes",
]

#: The column every row carries to declare which split it belongs to (§2.2).
#:
#: A dataset is one table and the split is a property of the row, not of the
#: file: a document selects one with the ``<ref>#<split>`` fragment. Values are
#: arbitrary strings — ``train``/``val``/``test`` is convention, not vocabulary,
#: so a k-fold table may use ``fold0``…``fold4`` and an undivided pool one
#: uniform value. Nothing in the library hardcodes a split name.
SPLIT_COLUMN = "split"


def table_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """The exact bytes a table serializes to: sorted keys, fixed indent,
    trailing newline. Deterministic on purpose — the content digest stamped
    into a canonical form (§7) has to be reproducible from the build's
    command line, on any machine."""
    return (json.dumps(list(rows), indent=1, sort_keys=True) + "\n").encode()


#: The text column an inline role's rows carry, and the one its ``field`` is
#: (§2.2): a document that inlines its inputs names no column, so the name is
#: fixed here rather than authored.
INPUT_COLUMN = "input"

#: A dataset ref that names a table inlined in the document rather than a
#: file under the data root (§2.2): ``inline:<content digest>``. The digest is
#: [`table_bytes`][] over the rows, the same formula
#: [`causalab.io.env.FileDatasets.digest`][] applies to a file's selected
#: rows, so an inline table and a file table with equal rows are one dataset
#: to every consumer of the digest — the canonical form, the forward-group
#: identity, the receipt. A hex digest carries no ``#``, so the split-fragment
#: syntax never applies to an inline ref.
INLINE_REF_PREFIX = "inline:"

#: Inline tables by their ref, filled by [`inline_ref`][] as documents are
#: parsed and read by [`inline_table`][] when a resolver is asked for the
#: rows. Content-addressed — the key is the digest of the value — so an entry
#: can never be stale, and registering the same inputs twice is a no-op. The
#: memo is process-local on purpose: an inline ref only ever originates from a
#: parse in the same process, and every door parses before it resolves.
_INLINE_TABLES: dict[str, list[dict[str, Any]]] = {}


def inline_rows(inputs: Sequence[str]) -> list[dict[str, Any]]:
    """The table an inline role denotes: one row per input, the text under
    [`INPUT_COLUMN`][], every row in the one split ``"all"`` (§2.2)."""
    return [{INPUT_COLUMN: text, SPLIT_COLUMN: "all"} for text in inputs]


def inline_ref(inputs: Sequence[str]) -> str:
    """Register the table ``inputs`` denotes and return its ref."""
    rows = inline_rows(inputs)
    ref = INLINE_REF_PREFIX + hashlib.sha256(table_bytes(rows)).hexdigest()
    _INLINE_TABLES.setdefault(ref, rows)
    return ref


def is_inline_ref(ref: str) -> bool:
    return ref.startswith(INLINE_REF_PREFIX)


def inline_table(ref: str) -> list[dict[str, Any]]:
    """The rows an inline ref denotes, as fresh copies (a consumer may
    annotate the rows it is handed). ``KeyError`` for a ref no parse in this
    process registered — the resolver turns that into its own refusal."""
    return [dict(row) for row in _INLINE_TABLES[ref]]
