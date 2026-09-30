"""Resolve all positions required by a compiled intervention.

The service tokenizes each distinct set of rows and positions once. It resolves
reads and writes, checks declared pair alignment, and builds any requested
location ledger before the engine loads weights."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Mapping, Sequence

from causalab.protocol.lowering import lower_bands
from causalab.protocol.positions.alignment import (
    UnalignableError,
    alignment_of,
    check_declared,
)
from causalab.protocol.positions.encoding import (
    Frame,
    PositionFrame,
    constituent_candidate_runs,
    encode,
    resolve_position,
    select_field,
)
from causalab.protocol.positions.framing import encode_framed
from causalab.protocol.positions.ledger import LedgerRow, LocationLedger
from causalab.protocol.positions.spans import constituents
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import AlignmentCardinality, Document, PositionSpec

__all__ = [
    "Addressed",
    "ResolvedAddress",
    "RowProblem",
    "StepPositions",
    "StepResolution",
    "address_key",
    "addressed",
    "build_ledger",
    "check_declared_alignment",
    "check_positions",
    "encode_role",
    "encode_roles",
    "positions_key",
    "resolve_address",
    "resolve_positions",
    "spec_of",
]


def spec_of(doc: Document, pos: Any) -> PositionSpec:
    """The concrete spec a read or write's ``pos`` names — a ``positions``
    table entry or the inline spec itself; anything else (a surviving sweep
    wrapper) is a caller error."""
    spec = doc.positions[pos] if isinstance(pos, str) else pos
    if not isinstance(spec, PositionSpec):
        raise ProtocolError("P2", f"unresolved position {pos!r}")
    return spec


def address_key(pos: Any, spec: PositionSpec) -> str:
    """One address's key: its ``positions`` name, or the inline spec's
    ``repr`` — the same spelling for the whole spec and for each constituent
    of a non-atomic set, which is resolved as an inline spec of its own."""
    return pos if isinstance(pos, str) else repr(spec)


@dataclasses.dataclass(frozen=True)
class RowProblem:
    """A row an address could not be aligned on: the observed cardinality
    (``absent`` / ``ambiguous``) and the resolver's message naming the value,
    its count and the row."""

    cardinality: AlignmentCardinality
    message: str

    def error(self) -> UnalignableError:
        return UnalignableError(self.cardinality, self.message)


@dataclasses.dataclass(frozen=True)
class ResolvedAddress:
    """One address on one input role, every row: ``indices[row]`` are the
    padded-frame token indices (empty where ``problems`` names the row)."""

    indices: tuple[tuple[int, ...], ...]
    problems: Mapping[int, RowProblem]

    def rows(self) -> list[list[int]]:
        return [list(row) for row in self.indices]

    def first_problem(self) -> RowProblem:
        return self.problems[min(self.problems)]


def resolve_address(
    spec: PositionSpec,
    frame: Frame,
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> ResolvedAddress:
    """Every row's positions for one spec on one input role's frame.

    An unalignable row is recorded, not raised: whether it refuses (a write)
    or becomes an ``unavailable`` cell (a read) is the consumer's call,
    [`check_positions`][] and the executor's. An out-of-bounds index — an
    authoring error whatever the consumer — raises here."""
    out: list[tuple[int, ...]] = []
    problems: dict[int, RowProblem] = {}
    for i in range(len(rows)):
        try:
            out.append(
                tuple(
                    resolve_position(spec, frame, i, dataset_row=rows[i], field=field)
                )
            )
        except UnalignableError as err:
            problems[i] = RowProblem(err.cardinality, err.message)
            out.append(())
    return ResolvedAddress(indices=tuple(out), problems=problems)


def _part(k: int, parts: Sequence[Any]) -> str:
    return f" constituent [{k}]" if len(parts) > 1 else ""


def check_declared_alignment(
    spec: PositionSpec,
    frames: Mapping[str, Frame],
    role_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    role_fields: Mapping[str, str],
    input_role: str,
    *,
    label: str,
) -> None:
    """A declared ``alignment`` (§2.3) against the cardinality the pair
    actually has.

    Per row, the address's candidate runs on ``input_role`` are paired with
    its candidate runs on every other input role the document reads — rows
    are paired by index (§2.2) — and
    [`alignment_of`][] names the
    observed cardinality; [`check_declared`][] refuses a contradiction, naming both. A single-role
    document is checked against its own candidates. One classification per
    address the spec *is*: an ordinary position or an atomic span is one, a
    non-atomic set is each of its constituents (§2.3 "composable groups").
    """
    rows = role_rows[input_role]
    field = role_fields[input_role]
    frame = frames[input_role]
    others = [role for role in role_rows if role != input_role]
    for i in range(len(rows)):
        mine_by_part = constituent_candidate_runs(
            spec, frame, i, dataset_row=rows[i], field=field
        )
        if not others:
            for k, mine in enumerate(mine_by_part):
                check_declared(
                    spec.alignment,
                    alignment_of(mine),
                    where=f"{label}{_part(k, mine_by_part)} on input "
                    f"{input_role!r}, row {i},",
                )
        for other in others:
            other_rows = role_rows[other]
            if i >= len(other_rows):
                continue
            theirs_by_part = constituent_candidate_runs(
                spec,
                frames[other],
                i,
                dataset_row=other_rows[i],
                field=role_fields[other],
            )
            for k, (mine, theirs) in enumerate(zip(mine_by_part, theirs_by_part)):
                check_declared(
                    spec.alignment,
                    alignment_of(mine, theirs),
                    where=f"{label}{_part(k, mine_by_part)} across inputs "
                    f"{input_role!r} and {other!r}, row {i},",
                )


def encode_role(
    tokenizer: Any,
    doc: Document,
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> PositionFrame:
    """One input role's frame: under a ``segments`` section the declared
    frame with its segments located (§2.2.1, [`encode_framed`][]); else the plain-text frame of the
    rows' ``field``."""
    if doc.segments is not None:
        return encode_framed(tokenizer, rows, field, doc.segments)
    return encode(tokenizer, [str(select_field(row, field)) for row in rows])


def encode_roles(
    tokenizer: Any,
    doc: Document,
    role_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    role_fields: Mapping[str, str],
) -> dict[str, PositionFrame]:
    """Every input role's frame, keyed as the roles are (§2.2)."""
    return {
        role: encode_role(tokenizer, doc, rows, role_fields[role])
        for role, rows in role_rows.items()
    }


#: One address a document names: ``(pos, input role, cell)`` — ``cell`` is
#: the read whose rows these are (its unalignable rows become its
#: ``unavailable`` cell), ``None`` for a write (an unalignable row refuses).
Addressed = tuple[Any, str, str | None]


def addressed(doc: Document) -> list[Addressed]:
    """Every address the document's reads and writes name, on the input role
    each reads or writes — reads first, then each intervened model's writes
    in document order (the order the ledger records them)."""
    out: list[Addressed] = []
    for ref in doc.read_refs():
        _model, role = doc.group_of(ref)
        out.append((doc.reads[ref.read].pos, role, ref.read))
    for im in doc.intervened_models.values():
        names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
        out.extend((doc.writes[ename].pos, str(im.input), None) for ename in names)
    return out


@dataclasses.dataclass
class StepPositions:
    """One step's resolved positions: the frames per input role, the rows
    they were encoded from, and every address resolved so far.

    ``addresses`` fills on demand: [`address`][] resolves an address the
    protocol layer did not pre-resolve (a constituent the ledger asks for, an
    operand read's position) through the same function, so a consumer never
    sees a missing key — only a resolution that happened later. Frames may be
    the protocol's [`PositionFrame`][] or the engine's tensor-bearing frame; both are a
    [`Frame`][].
    """

    frames: Mapping[str, Frame]
    role_rows: Mapping[str, Sequence[Mapping[str, Any]]]
    role_fields: Mapping[str, str]
    addresses: dict[tuple[str, str], ResolvedAddress] = dataclasses.field(
        default_factory=dict
    )
    #: the ``(address key, input role)`` pairs whose declared ``alignment``
    #: has been checked against the pair — once per step, as before
    alignment_checked: set[tuple[str, str]] = dataclasses.field(default_factory=set)

    def address(self, pos: Any, spec: PositionSpec, input_role: str) -> ResolvedAddress:
        """One address resolved on one role — cached; decides nothing about a
        declared ``alignment`` ([`check_alignment`][], which a consumer runs
        *after* it has read the row problems, as the executor always did: a
        write's unalignable row refuses before its declaration is checked)."""
        key = (address_key(pos, spec), input_role)
        found = self.addresses.get(key)
        if found is None:
            found = self.addresses[key] = resolve_address(
                spec,
                self.frames[input_role],
                self.role_rows[input_role],
                self.role_fields[input_role],
            )
        return found

    def check_alignment(self, pos: Any, spec: PositionSpec, input_role: str) -> None:
        """A declared ``alignment`` against the pair, once per address per
        step (a no-op for a spec that declares none)."""
        if spec.alignment is None:
            return
        key = (address_key(pos, spec), input_role)
        if key in self.alignment_checked:
            return
        self.alignment_checked.add(key)
        label = f"position {pos!r}" if isinstance(pos, str) else f"position {spec!r}"
        check_declared_alignment(
            spec,
            self.frames,
            self.role_rows,
            self.role_fields,
            input_role,
            label=label,
        )

    def rows(
        self, pos: Any, spec: PositionSpec, input_role: str, *, cell: str | None
    ) -> list[list[int]]:
        """Every row's positions for one address, with the consumer's
        reading of an unalignable row: a write (``cell`` ``None``) refuses
        with the first such row's reason; a read gets ``[]`` there and reads
        the reasons off [`address`][] for its cell. The declared
        ``alignment``, if any, is checked after — the order the executor
        always had."""
        resolved = self.address(pos, spec, input_role)
        if resolved.problems and cell is None:
            raise resolved.first_problem().error()
        self.check_alignment(pos, spec, input_role)
        return resolved.rows()


def resolve_positions(
    doc: Document,
    frames: Mapping[str, Frame],
    role_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    role_fields: Mapping[str, str],
) -> StepPositions:
    """Resolve every prompt-frame address the document names (module
    docstring) — whole specs and, for a non-atomic set, each constituent as
    its own address (§2.3) — refusing a write on an unalignable row and
    holding each declared ``alignment`` to the pair, in that order. A
    ``generated`` address is the decode's, not this frame's, and is skipped.
    The returned object is shared by every executor of the steps that share
    its key: its caches fill on demand and are never invalidated, which is
    safe because every sharer has the same frames and rows.
    """
    positions = StepPositions(
        frames=frames, role_rows=role_rows, role_fields=role_fields
    )
    for pos, role, cell in addressed(doc):
        spec = spec_of(doc, pos)
        if spec.generated is not None:
            continue
        # a write's unalignable row refuses here, before its declared
        # alignment is checked (the executor's order); a read's is recorded
        positions.rows(pos, spec, role, cell=cell)
        parts = constituents(spec)
        if len(parts) > 1:
            for part in parts:
                positions.rows(part, part, role, cell=cell)
    return positions


def check_positions(doc: Document, positions: StepPositions) -> None:
    """The refusals a resolution carries: a *write* on a row its address
    cannot align (§2.3, reason ``alignment_missing`` / ``alignment_ambiguous``)
    — before any forward, with the resolver's own text. Reads are not refused
    here: their unalignable rows are ``unavailable`` cells (§4.1)."""
    for pos, role, cell in addressed(doc):
        if cell is not None:
            continue
        spec = spec_of(doc, pos)
        if spec.generated is not None:
            continue
        positions.rows(pos, spec, role, cell=None)


def build_ledger(
    doc: Document,
    positions: StepPositions,
    tokenizer: Any,
    *,
    has_positions: Callable[[str], bool],
) -> LocationLedger:
    """The location ledger (§6): one row per (example, edit group,
    constituent, side, token index, token id, decoded token) for every
    prompt-frame position every read and write of the step resolves, on every
    input it resolves it on — the same indices the gathers use, recorded once.

    The edit group is the forward group — ``<model> on <input>``; the
    constituent is the position's name (or its inline path), with ``[k]`` per
    member of a non-atomic set; the token index counts from the row's first
    real token, so it is a fact of the row and not of the batch's padding.
    Continuation reads are not in the ledger: their steps are a result of the
    decode, recorded per read as ``steps``. ``has_positions(site)`` says
    whether the site's tap has a position axis at all (an attention pattern
    has none, and nothing gathers) — the registry's answer for the protocol
    layer, the resolved tap's for an executor.
    """
    ledger = LocationLedger()

    def record(
        pos: Any, role: str, group: str, label: str, *, cell: str | None
    ) -> None:
        spec = spec_of(doc, pos)
        if spec.generated is not None:
            return
        frame = positions.frames[role]
        parts = constituents(spec)
        if len(parts) == 1:
            per_part = [(label, positions.rows(pos, spec, role, cell=cell))]
        else:
            per_part = [
                (f"{label}[{k}]", positions.rows(part, part, role, cell=cell))
                for k, part in enumerate(parts)
            ]
        for constituent, per_row in per_part:
            for example, indices in enumerate(per_row):
                first = frame.first_real(example)
                ids = frame.row_ids(example)
                for index in indices:
                    token_id = int(ids[index])
                    ledger.add(
                        LedgerRow(
                            example=example,
                            edit_group=group,
                            constituent=constituent,
                            side=role,
                            token_index=index - first,
                            token_id=token_id,
                            decoded_token=str(
                                tokenizer.convert_ids_to_tokens(token_id)
                            ),
                        )
                    )

    for ref in doc.read_refs():
        rname, read = ref.read, doc.reads[ref.read]
        model, role = doc.group_of(ref)
        if not has_positions(str(read.site)):
            continue  # the tap's last axis is not positions; nothing gathers
        record(
            read.pos,
            role,
            f"{model} on {role}",
            read.pos if isinstance(read.pos, str) else f"reads.{rname}.pos",
            cell=rname,
        )
    for mname, im in doc.intervened_models.items():
        names = tuple(im.writes) if isinstance(im.writes, tuple) else ()
        for ename in names:
            write = doc.writes[ename]
            if not has_positions(str(write.site)):
                continue
            record(
                write.pos,
                str(im.input),
                f"{mname} on {im.input}",
                write.pos if isinstance(write.pos, str) else f"writes.{ename}.pos",
                cell=None,
            )
    return ledger


@dataclasses.dataclass(frozen=True)
class StepResolution:
    """What the protocol layer resolved for one positions key: the step's
    positions and, when the document saves one, its ledger."""

    positions: StepPositions
    ledger: LocationLedger | None


def positions_key(doc: Document) -> str:
    """Identify a step's positions and the ledger stored beside them.

    Token frames depend on the model, data and segments. Addresses and ledger
    labels use the execution form, after layer bands are lowered. Ledger
    groups also depend on the read's model and the write's owning model;
    components decide whether a ledger row has a position axis. Single-layer
    sweeps retain the same labels and can still share one resolution.
    """
    doc = lower_bands(doc)
    body = {
        "model": {"key": str(doc.model.key), "revision": str(doc.model.revision)},
        "data": doc.raw.get("data"),
        "segments": repr(doc.segments),
        # the parsed table, so an inline spec and a named one spell the same
        # way here as they do in `address_key`
        "positions": {name: repr(spec) for name, spec in doc.positions.items()},
        "addressed": [
            [address_key(pos, spec_of(doc, pos)), role, cell]
            for pos, role, cell in addressed(doc)
        ],
        # one entry per read *as taken on a model* (§2.9): the same address
        # listed by two models is two ledger groups
        "ledger_reads": [
            (
                ref.read,
                str(ref.model),
                str(doc.sites[str(doc.reads[ref.read].site)].component),
            )
            for ref in doc.read_refs()
        ],
        "ledger_writes": [
            (model, name, str(doc.sites[str(doc.writes[name].site)].component))
            for model, im in doc.intervened_models.items()
            for name in (im.writes if isinstance(im.writes, tuple) else ())
        ],
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
