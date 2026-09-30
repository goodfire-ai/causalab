"""Resolve a document's datasets, artifacts, models, and tokenizers.

``ResolutionEnv`` supplies the services used to compile and run a document.
File resolvers interpret references relative to their configured roots.
Tokenizer loading occurs when a run resolves positions. Validation can inspect
the document and its tables before that service is called."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from causalab.io.sources import resolve_artifact_fields
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.identity import (
    ARTIFACT_IDENTITY_KEYS,
    build_artifact_identity,
    check_artifact_identity,
)
from causalab.protocol.registry import ModelInfo, get_model_info
from causalab.tables import SPLIT_COLUMN, inline_table, is_inline_ref, table_bytes

__all__ = [
    "ARTIFACT_IDENTITY_KEYS",
    "ArtifactStore",
    "DatasetResolver",
    "FileArtifacts",
    "FileDatasets",
    "ResolutionEnv",
    "build_artifact_identity",
    "check_artifact_identity",
    "endpoints",
    "entry_identity",
    "entry_table",
    "file_env",
    "read_safetensors_metadata",
    "resolve_artifact_fields",
    "split_dataset_ref",
]


class ArtifactStore(Protocol):
    """Where prior runs' outputs are found."""

    def read_value(self, artifact: str, key: str) -> Any: ...

    def file_digest(self, file_path: str) -> str: ...

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None: ...


class DatasetResolver(Protocol):
    """Where dataset refs resolve. ``digest`` is the content digest stamped
    into canonical forms; ``columns`` backs ``validate --data``; ``rows`` is
    the table content a run consumes.

    All three are one contract on purpose. The pure verbs
    (``validate``/``explain``/``digest``) only need the first two, but a
    resolver that cannot produce rows cannot back a ``run`` — so the
    requirement is declared here instead of being discovered by a ``getattr``
    probe deep inside an engine."""

    def digest(self, ref: str) -> str: ...

    def columns(self, ref: str) -> tuple[str, ...]: ...

    def rows(self, ref: str) -> list[dict[str, Any]]: ...


@dataclasses.dataclass(frozen=True)
class ResolutionEnv:
    """The resolution services a load runs against: datasets, artifacts and
    the registry's static model metadata — and, for the run door alone, the
    tokenizer ([`causalab.io.tokenizer`][]; the pure verbs never call it,
    which is what keeps them torch-free)."""

    datasets: DatasetResolver
    artifacts: ArtifactStore
    model_info: Callable[[str], ModelInfo] = get_model_info
    #: ``(model key, revision) -> tokenizer`` — the service the run door's
    #: position resolution loads a tokenizer through; ``None`` is
    #: [`causalab.io.tokenizer.load_tokenizer`][], imported by the verb that
    #: calls it and not here: this module sits in every step script's
    #: layering closure (``tests/workflow/test_closure_census.py``), and the
    #: tokenizer module is the one io module that reaches transformers
    tokenizers: Callable[[str, str], Any] | None = None


def file_env(
    data_root: Path, artifacts_root: Path, *, fallback_roots: tuple[Path, ...] = ()
) -> ResolutionEnv:
    """The file-backed environment: datasets as JSON tables under
    ``data_root`` (then each of ``fallback_roots``, in order — the CLI puts
    the shipped task tables there, so a private table can shadow a shipped
    one, root first), artifacts under ``artifacts_root``, and the registry's
    static model metadata."""
    return ResolutionEnv(
        datasets=FileDatasets(root=data_root, fallback_roots=fallback_roots),
        artifacts=FileArtifacts(root=artifacts_root),
    )


# --------------------------------------------------------------------------- #
# file-backed implementations
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FileArtifacts:
    """Artifacts under one root directory (a run-output tree)."""

    root: Path

    def _value_table(self, artifact: str) -> Mapping[str, Any]:
        for candidate in (
            self.root / f"{artifact}.json",
            self.root / artifact / "values.json",
        ):
            if candidate.is_file():
                table = json.loads(candidate.read_text())
                if not isinstance(table, dict):
                    raise ValidationError(
                        15, f"artifact {artifact!r} is not a JSON object of values"
                    )
                return table
        raise FileNotFoundError(f"no artifact {artifact!r} under {self.root}")

    def read_value(self, artifact: str, key: str) -> Any:
        table = self._value_table(artifact)
        if key not in table:
            raise KeyError(
                f"artifact {artifact!r} has no key {key!r} (has {sorted(table)})"
            )
        return table[key]

    def file_digest(self, file_path: str) -> str:
        target = self.root / file_path
        if not target.is_file():
            raise ValidationError(
                15, f"artifact file {file_path!r} not found under {self.root} (§5.15)"
            )
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def read_identity(self, file_path: str) -> Mapping[str, Any] | None:
        target = self.root / file_path
        if not target.is_file():
            raise ValidationError(
                15, f"artifact file {file_path!r} not found under {self.root} (§5.15)"
            )
        return read_safetensors_metadata(target)

    def resolve_path(self, file_path: str) -> Path:
        return self.root / file_path


def read_safetensors_metadata(path: Path) -> Mapping[str, Any] | None:
    """The ``__metadata__`` table of a safetensors file — a pure header read
    (8-byte little-endian header length, then a JSON object), no tensor
    library involved. Returns ``None`` when the file carries no metadata.

    Format reference: https://github.com/huggingface/safetensors#format.
    """
    with path.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValidationError(
                15, f"{path} is not a safetensors file (truncated header)"
            )
        (header_len,) = struct.unpack("<Q", prefix)
        header = json.loads(fh.read(header_len))
    meta = header.get("__metadata__")
    return meta if isinstance(meta, Mapping) else None


def split_dataset_ref(ref: str) -> tuple[str, str | None]:
    """``"weekdays#train"`` → ``("weekdays", "train")``; a bare ref → ``(ref, None)``.

    The fragment names one split *inside* a table (§2.2): a dataset is one
    table, every row declares its split in the `SPLIT_COLUMN` column, and a document selects one by fragment rather than by
    pointing at a second file. Borrowed from URL fragment syntax, which means
    exactly this — a named part of one resource.

    Split on the **last** ``#`` so a data root containing a ``#`` in a directory
    name still resolves. An empty fragment is refused rather than silently
    meaning "the whole table": ``weekdays#`` is a typo, not a selection.
    """
    base, sep, fragment = ref.rpartition("#")
    if not sep:
        return ref, None
    if not fragment:
        raise ValidationError(
            22,
            f"dataset ref {ref!r} ends in an empty '#' fragment — name a split "
            f"({base!r}#train) or drop the '#' to take the whole table",
            path="data",
        )
    return base, fragment


@functools.lru_cache(maxsize=64)
def _checked_table_text(path: str, base: str, _stamp: tuple[int, int]) -> str:
    """One table's text, read and split-checked once per ``(path, mtime,
    size)``. A campaign resolves every point's roles against the same refs
    (``positions.roles.resolve_roles``), so without this a point costs a read of the
    table per role — milliseconds per point on a network home, all of it
    re-deriving the same rows. The text and not the rows: every call parses
    its own (`FileDatasets._table`), so a point's rows share nothing
    with the memo or with another point's — and a parse of a scan table is a
    tenth of a millisecond, a fraction of the read it replaces or of a deep
    copy of parsed rows. The stamp is the file's, so a table rewritten in the
    same process (a script step producing one a later step reads) is a miss,
    as ``tensor_files._read_bundle`` treats a rewritten tensor file. A refusal is
    not memoized: a raising check re-raises on every call."""
    text = Path(path).read_text()
    rows = json.loads(text)
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValidationError(4, f"dataset {base!r} is not a JSON array of row objects")
    _check_splits_are_disjoint(base, rows)
    return text


#: ``FileDatasets.digest``'s memo: ``(path, ref, (mtime_ns, size)) → digest``.
#: A digest is a function of the file's bytes and the ref alone, so a hit is
#: exact; the stamp makes a rewritten table a miss. Cleared whole when full,
#: which only costs a re-serialization.
_TABLE_DIGESTS: dict[tuple[str, str, tuple[int, int]], str] = {}
_TABLE_DIGESTS_MAX = 256


def _declared_splits(base: str, rows: list[dict[str, Any]]) -> set[str]:
    """The split values a table declares — refusing one that declares none [V22].

    Every table declares, and the requirement is the point rather than a
    formality: a split that is optional is a split that can be omitted, and an
    omitted one is exactly the state the column exists to abolish. A single
    undivided pool says so with one uniform value, which costs a word and buys
    the guarantee that no table is silently unlabelled.
    """
    missing = [i for i, row in enumerate(rows) if SPLIT_COLUMN not in row]
    if missing:
        where = (
            "no row declares one"
            if len(missing) == len(rows)
            else f"{len(missing)} of {len(rows)} rows do not (first: row {missing[0]})"
        )
        raise ValidationError(
            22,
            f"dataset {base!r} has no {SPLIT_COLUMN!r} column — {where}. Every "
            f"table declares which split its rows are (§2.2); rebuild it with "
            f"scripts/build_task_dataset.py --split all for one undivided pool, "
            f"or causalab.tasks.splits.generate_split_dataset for a "
            f"partitioned one",
        )
    return {str(row[SPLIT_COLUMN]) for row in rows}


def endpoints(row: Mapping[str, Any]) -> set[str]:
    """The prompts a row puts in front of the model, both sides of the pair."""
    out: set[str] = set()
    value = row.get("input")
    if isinstance(value, str):
        out.add(value)
    counterfactuals = row.get("counterfactual_inputs")
    if isinstance(counterfactuals, list):
        out.update(str(v) for v in counterfactuals if isinstance(v, str))
    return out


def _check_splits_are_disjoint(base: str, rows: list[dict[str, Any]]) -> None:
    """Splits of one table share no prompt, at either endpoint [V22].

    Row-level disjointness is free — a row declares one split — but that is the
    weaker half. The leak that matters is a *prompt* appearing as a training
    base and again as a test counterfactual, which reports a training score
    under a held-out name. Because both splits live in one table, that is now a
    question about the bytes in front of us, so it is answered here rather than
    asserted by whoever ran the builder.

    Checked for every table with more than one split, at the one place a ref
    becomes rows, so no verb can skip it. A deliberate train-equals-test
    ablation is spelled by naming *one* split twice in the document, where it is
    visible, instead of by two splits that quietly coincide.
    """
    declared = {str(row[SPLIT_COLUMN]) for row in rows if SPLIT_COLUMN in row}
    if len(declared) < 2:
        return
    seen: dict[str, str] = {}
    for row in rows:
        split = str(row[SPLIT_COLUMN])
        for endpoint in endpoints(row):
            other = seen.setdefault(endpoint, split)
            if other != split:
                raise ValidationError(
                    22,
                    f"dataset {base!r} leaks across splits: the prompt "
                    f"{endpoint!r} appears in both {other!r} and {split!r}. "
                    f"Splits of one table must be endpoint-disjoint (§2.2) — "
                    f"rebuild with causalab.tasks.splits.generate_split_dataset, "
                    f"which partitions inputs into groups before pairing them",
                )


def entry_table(metadata: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The per-entry provenance table a producer writes into a bundle's header
    (``outputs.TensorFile``): key → ``{"slot", "coords", …identity}``. Empty
    for a bundle written without one, in which case the keys themselves
    (`causalab.protocol.bundles.parse_entry_key`) are all there is."""
    if not metadata:
        return {}
    raw = metadata.get("entries")
    if not isinstance(raw, str):
        return {}
    try:
        table = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return table if isinstance(table, dict) else {}


def entry_identity(metadata: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    """The identity of one bundle entry (§8): the file-level stamp, overridden
    by whatever the ``entries`` table records for that key — the per-entry
    fields a swept producer could not stamp file-wide. One rule for every
    reader of a header: a script inheriting provenance and an engine loading
    a featurizer see the same identity for the same entry."""
    if not metadata:
        return {}
    identity = {
        field: value
        for field, value in metadata.items()
        if field in ARTIFACT_IDENTITY_KEYS
    }
    entry = entry_table(metadata).get(key, {})
    identity.update(
        {
            field: value
            for field, value in entry.items()
            if field in ARTIFACT_IDENTITY_KEYS
        }
    )
    return identity


@dataclasses.dataclass(frozen=True)
class FileDatasets:
    """Dataset refs as serialized JSON tables under one data root.

    A ref ``weekdays/data`` resolves to ``<root>/weekdays/data.json`` — a JSON
    array of row objects. Task-generated tables are written by
    [`causalab.tasks.serialize`][] into this same layout (deterministically,
    so the digest is reproducible), and nothing beside them: a table is
    exactly the bytes a document names (§2.2).

    **A ref may name one split of that table**: ``weekdays/data#train`` keeps
    only the rows whose ``split`` column is ``"train"`` (§2.2,
    [`split_dataset_ref`][]). Selection happens here, at the one place a ref
    becomes rows, so nothing downstream — schema, canonical form, engines —
    needs to know a split exists.

    The content digest is the sha256 of the **selected rows'** canonical bytes,
    not of the file. That is what makes the fragment safe: two splits of one
    table carry two distinct digests, so run identity is preserved; and a run's
    digest depends only on the rows it consumed, so adding a split to a table
    does not invalidate a run over the splits already in it. For a whole-table
    ref the two agree by construction, because
    [`write_dataset_table`][causalab.tasks.serialize.write_dataset_table] writes exactly
    `table_bytes`.
    """

    root: Path
    #: Searched in order after ``root`` misses. The CLI puts the shipped task
    #: tables (``causalab/tasks/``) here, so a document can mix a private table
    #: under ``--data-root`` with ``<task>/data/<variant>`` — and a private
    #: table of the same ref shadows the shipped one, visibly, root first.
    fallback_roots: tuple[Path, ...] = ()

    @property
    def roots(self) -> tuple[Path, ...]:
        seen: list[Path] = []
        for root in (self.root, *self.fallback_roots):
            if root not in seen:
                seen.append(root)
        return tuple(seen)

    def _file(self, ref: str) -> Path:
        for root in self.roots:
            for candidate in (root / f"{ref}.json", root / ref):
                if candidate.is_file():
                    return candidate
        where = " nor ".join(str(root) for root in self.roots)
        raise ValidationError(
            4, f"dataset {ref!r} not found under {where}", path="data"
        )

    def digest(self, ref: str) -> str:
        """The sha256 of the canonical bytes of the rows ``ref`` selects.

        The digest is serialized once per file, ref and file version
        (`_TABLE_DIGESTS`). A workflow load compiles every step's points, and
        each point names its tables' digests, so serializing a 3000-row table
        per point cost minutes. A table rewritten in the same process is a
        new version, as for `_checked_table_text`."""
        if is_inline_ref(ref):
            return hashlib.sha256(table_bytes(self.rows(ref))).hexdigest()
        path = self._file(split_dataset_ref(ref)[0])
        stat = path.stat()
        key = (str(path), ref, (stat.st_mtime_ns, stat.st_size))
        digest = _TABLE_DIGESTS.get(key)
        if digest is None:
            digest = hashlib.sha256(table_bytes(self.rows(ref))).hexdigest()
            if len(_TABLE_DIGESTS) >= _TABLE_DIGESTS_MAX:
                _TABLE_DIGESTS.clear()
            _TABLE_DIGESTS[key] = digest
        return digest

    def columns(self, ref: str) -> tuple[str, ...]:
        rows = self.rows(ref)
        cols: set[str] = set()
        for row in rows:
            cols.update(row)
        return tuple(sorted(cols))

    def rows(self, ref: str) -> list[dict[str, Any]]:
        if is_inline_ref(ref):
            # a table inlined in the document (§2.2): registered at parse,
            # never on disk — one split, so no fragment to select on
            try:
                return inline_table(ref)
            except KeyError:
                raise ValidationError(
                    4,
                    f"inline dataset {ref!r} was not registered by a parse in "
                    f"this process — an inline ref is never authored; write the "
                    f"role's 'inputs' and let the loader derive it (§2.2)",
                    path="data",
                ) from None
        base, fragment = split_dataset_ref(ref)
        rows = self._table(base)
        declared = _declared_splits(base, rows)
        if fragment is None:
            if len(declared) > 1:
                raise ValidationError(
                    22,
                    f"dataset {base!r} carries {len(declared)} splits "
                    f"({sorted(declared)}) and the ref names none of them — a "
                    f"bare ref would consume every split at once. Select one: "
                    f"{base}#{sorted(declared)[0]}",
                )
            return rows
        selected = [row for row in rows if str(row.get(SPLIT_COLUMN)) == fragment]
        if not selected:
            raise ValidationError(22, self._no_such_split(base, fragment, rows))
        return selected

    def _table(self, base: str) -> list[dict[str, Any]]:
        """The table's rows, parsed by this call from text read and checked
        once per file version (`_checked_table_text`): every point of a
        campaign resolves the same refs, and its rows are its own — a consumer
        that annotates a row it was handed, or a value nested in one (a
        ``counterfactual_inputs`` list, a ``<column>_variables`` table), reaches
        neither the memo nor another point's rows."""
        path = self._file(base)
        stat = path.stat()
        text = _checked_table_text(str(path), base, (stat.st_mtime_ns, stat.st_size))
        rows: list[dict[str, Any]] = json.loads(text)
        return rows

    @staticmethod
    def _no_such_split(base: str, fragment: str, rows: list[dict[str, Any]]) -> str:
        """Why a fragment selected nothing — a missing column and a mistyped
        value are different mistakes and get different messages."""
        if not any(SPLIT_COLUMN in row for row in rows):
            return (
                f"dataset {base!r} declares no {SPLIT_COLUMN!r} column, so it has "
                f"no split {fragment!r} to select — rebuild the table with a "
                f"{SPLIT_COLUMN!r} column, or drop the '#{fragment}' fragment"
            )
        available = sorted({str(row.get(SPLIT_COLUMN)) for row in rows})
        return (
            f"dataset {base!r} has no rows in split {fragment!r} "
            f"(it declares {available})"
        )
