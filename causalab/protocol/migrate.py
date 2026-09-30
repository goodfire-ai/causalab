"""Migrate intervention specifications to the current protocol version.

The v1-to-v2 step groups fields under ``header``, ``model``, ``data``, and
``method``. The v2-to-v3 step changes site ``layer`` fields to ``layers`` bands
and updates dotted references, wrappers, and name templates. The v3-to-v4
step reorganizes the method block reads-first: reads lose their ``model`` /
``input`` binding and are listed on the models that take them, the
un-intervened model becomes a declared model with no writes, and the
``metrics`` section dissolves into ``aggregation`` blocks on the ``save``
entries, objective terms and eval entries that consume them. Workflow
documents receive the same reference updates when needed.

Migration is idempotent. Split v1 documents must be composed with a v1 loader
before migration. Schema validation remains the compiler's responsibility.

Markdown migration updates complete documents in fenced JSON blocks. A block
the migration refuses is left as it is, and ``causalab migrate`` reports it
with its line.
``format_document`` writes the result with two-space indentation and keeps
short, shallow objects on one line."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from causalab.protocol.rules.errors import ParseError, ProtocolError, suggest
from causalab.protocol.schema import (
    MIGRATABLE_PROTOCOL_VERSIONS,
    PROTOCOL_VERSION,
    REGULARIZER_KINDS,
    RETIRED_TOKEN_FORMS,
    load_raw,
)
from causalab.protocol.schema.types import RETIRED_TOKEN_FORM_REWRITES, answer_columns

__all__ = [
    "format_document",
    "is_v1_protocol",
    "main",
    "migrate_document",
    "migrate_markdown",
    "needs_migration",
]

#: The method sections protocol versions 1–3 knew, in their order (§1 of the
#: spec at version 3): what the v1 regroup collects, ``metrics`` included.
_V1_METHOD_SECTIONS: tuple[str, ...] = (
    "segments",
    "positions",
    "sites",
    "featurizers",
    "params",
    "code",
    "reads",
    "writes",
    "intervened_models",
    "metrics",
    "train",
    "save",
)
#: The version the v1 → v2 step writes; the v2 → v3 step reads it.
_V2: str = "2"
#: The version the v2 → v3 step writes; the v3 → v4 step reads it.
_V3: str = "3"
#: A dotted metric-field id in the retired spelling (``metrics.<name>.<field>``).
_DOTTED_METRIC = re.compile(r"\bmetrics\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b")
#: The name the un-intervened model takes when only ``base`` is read on it.
_UNWRITTEN: str = "original"
#: The method sections of protocol 4, in their recommended order (§1).
_V4_METHOD_SECTIONS: tuple[str, ...] = (
    "intervened_models",
    "segments",
    "positions",
    "sites",
    "featurizers",
    "params",
    "code",
    "reads",
    "writes",
    "train",
    "save",
)
#: A dotted site-field id in the retired spelling, wherever a string sits.
_DOTTED_LAYER = re.compile(r"\bsites\.([A-Za-z0-9_]+)\.layer\b")
#: The retired ``names`` placeholder (§3.1).
_TEMPLATE_LAYER = "{layer}"

#: v1's top-level bookkeeping (spec v1 §1, rows 1–3).
_V1_HEADER: tuple[str, ...] = ("version", "type", "description")
#: The two spellings v1 accepted for the network (v1 §2.1).
_V1_MODEL: tuple[str, ...] = ("model", "neural_model")
#: The v1 split's own sections (v1 §1.1), which this rewrite refuses.
_V1_SPLIT: tuple[str, ...] = ("application", "method")

#: Longest line [`format_document`][] writes an object or list on one line
#: within — the repository's own documents wrap there.
LINE_WIDTH = 88


def is_v1_protocol(raw: Any) -> bool:
    """Whether ``raw`` is a v1 intervention specification: a mapping with a
    top-level ``version`` and no ``header``, and not a workflow (``steps``)."""
    return (
        isinstance(raw, Mapping)
        and "header" not in raw
        and "steps" not in raw
        and "version" in raw
    )


def needs_migration(raw: Any) -> bool:
    """Whether [`migrate_document`][] would change ``raw``: a v1 document, a
    document declaring a [`MIGRATABLE_PROTOCOL_VERSIONS`][causalab.protocol.schema.types.MIGRATABLE_PROTOCOL_VERSIONS]
    member, a document whose metrics carry a retired ``token_form`` value, or
    a workflow spelling a dotted ``sites.<name>.layer`` id."""
    if not isinstance(raw, Mapping):
        return False
    if is_v1_protocol(raw):
        return True
    if "steps" in raw:
        return _spells_old_dotted(raw) or _spells_metric_ids(raw)
    header = raw.get("header")
    return isinstance(header, Mapping) and (
        header.get("protocol_version") in MIGRATABLE_PROTOCOL_VERSIONS
        or _spells_retired_token_form(raw)
    )


def migrate_document(raw: Any) -> dict[str, Any]:
    """The current-version form of one document tree.

    A current-version document is returned as it is (the function is
    idempotent); a v1 document is regrouped and then carried through every
    later step; a protocol_version 2 document gets the ``layer`` → ``layers``
    rename and then the reads-first rewrite; a protocol_version 3 document
    the rewrite alone; a workflow gets its dotted ids re-spelled; a split v1
    document or a method file is refused (see the module docstring). Beyond the renamed
    field and the retired ``token_form`` values (`_retire_token_form`,
    applied to the protocol-3 form before the reads-first rewrite) nothing
    inside a section is inspected — the strict parse is the
    compiler's job, and a document that was wrong in v1 is the same wrong
    document now, with the same refusal from ``causalab validate``.
    """
    if not isinstance(raw, Mapping):
        raise ParseError("P1", "the top level must be a JSON object")
    if "steps" in raw:
        if _spells_metric_ids(raw):
            raise ParseError(
                "P2",
                "this workflow spells a dotted 'metrics.<name>.<field>' id, which "
                "protocol 4 has no section for: the field now lives on the save "
                "entry (or objective / eval term) that carries the aggregation — "
                "re-spell it by hand as 'save[<i>].aggregation.<field>' against "
                "the migrated document",
                path="steps",
            )
        return _rewrite_dotted(copy.deepcopy(dict(raw)))
    if "header" in raw:
        header = raw["header"]
        version = (
            header.get("protocol_version") if isinstance(header, Mapping) else None
        )
        if version == PROTOCOL_VERSION:
            return dict(raw)
        if version == _V3:
            return _v3_to_v4(_retire_token_form(dict(raw)))
        if version == _V2:
            return _v3_to_v4(_v2_to_v3(raw))
        raise ParseError(
            "P2",
            f"unsupported protocol_version {version!r}; migrate reads "
            f"{', '.join(repr(v) for v in ('1', *MIGRATABLE_PROTOCOL_VERSIONS))} "
            f"and writes protocol_version {PROTOCOL_VERSION!r}",
            path="header.protocol_version",
        )
    return _v3_to_v4(_v2_to_v3(_v1_to_v2(raw)))


def _v1_to_v2(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The regroup (module docstring, first step)."""
    if any(key in raw for key in _V1_SPLIT) or raw.get("type") == "method":
        raise ParseError(
            "P2",
            "a split method/application document (or a method file) is not "
            "migrated: composition was the v1 loader's. Compose it with a v1 "
            "release (`causalab explain` printed the composition) and migrate "
            "the composed document",
            path="method",
        )
    version = raw.get("version")
    if version != "1":
        raise ParseError(
            "P2",
            f"unsupported version {version!r}; migrate reads v1 documents "
            f'("version": "1") and writes protocol_version {PROTOCOL_VERSION!r}',
            path="version",
        )
    known = (*_V1_HEADER, *_V1_MODEL, "data", *_V1_METHOD_SECTIONS)
    for key in raw:
        if key not in known:
            raise ParseError(
                "P3", f"unknown section {key!r}{suggest(key, known)}", path=key
            )
    if all(key in raw for key in _V1_MODEL):
        raise ParseError("P2", "both 'model' and 'neural_model' are present")
    for required in ("data",):
        if required not in raw:
            raise ParseError("P2", f"missing required section {required!r}")
    model = raw.get("model", raw.get("neural_model"))
    if model is None:
        raise ParseError("P2", "missing required section 'model'")

    header: dict[str, Any] = {"protocol_version": _V2}
    if raw.get("description") is not None:
        header["description"] = raw["description"]
    return {
        "header": header,
        "model": model,
        "data": raw["data"],
        "method": {
            section: raw[section] for section in _V1_METHOD_SECTIONS if section in raw
        },
    }


def _v2_to_v3(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The ``layer`` → ``layers`` rename (module docstring, second step), on
    a deep copy; key order is kept so the rewritten file reads as the author
    laid it out, with ``layers`` where ``layer`` stood."""
    out = copy.deepcopy(dict(raw))
    header = dict(out.get("header") or {})
    header["protocol_version"] = _V3
    out["header"] = header
    method = out.get("method")
    if isinstance(method, Mapping):
        method = dict(method)
        sites = method.get("sites")
        if isinstance(sites, Mapping):
            method["sites"] = {
                name: _rename_layer(site) if isinstance(site, Mapping) else site
                for name, site in sites.items()
            }
        for section, table in method.items():
            if not isinstance(table, Mapping):
                continue
            for name, entry in table.items():
                if isinstance(entry, Mapping) and isinstance(entry.get("names"), str):
                    entry["names"] = entry["names"].replace(_TEMPLATE_LAYER, "{layers}")
        models = method.get("intervened_models")
        if isinstance(models, Mapping):
            for entry in models.values():
                writes = entry.get("writes") if isinstance(entry, Mapping) else None
                if not isinstance(writes, list):
                    continue
                entry["writes"] = [
                    {
                        family: _rename_layer(selector)
                        for family, selector in item.items()
                    }
                    if isinstance(item, Mapping)
                    and all(isinstance(v, Mapping) for v in item.values())
                    else item
                    for item in writes
                ]
        out["method"] = method
    return _retire_token_form(_rewrite_dotted(out))


#: The metric kinds whose answers are literal strings in the document itself,
#: and the field that holds them — the one place a retired ``token_form``
#: rewrote something the migration has to write back into the string.
_LITERAL_ANSWER_FIELDS: dict[str, str] = {
    "class_probs": "groups",
    "token_logits": "tokens",
    "js": "restrict",
}


def _spells_retired_token_form(raw: Mapping[str, Any]) -> bool:
    method = raw.get("method")
    metrics = method.get("metrics") if isinstance(method, Mapping) else None
    return isinstance(metrics, Mapping) and any(
        isinstance(m, Mapping) and m.get("token_form") in RETIRED_TOKEN_FORMS
        for m in metrics.values()
    )


#: Why a column metric under a retired ``token_form`` needs an author step,
#: and the step. `_refusal` states it once for several metrics.
_COLUMN_REASON = (
    "the retired value rewrote each answer's leading space when the run "
    "resolved it. The migration does not read the table, so dropping the key "
    "could change the scored token without an error. To keep the tokens the "
    "run scored, rewrite each answer string s of those columns, each member of "
    "a list of forms included"
)


@dataclasses.dataclass(frozen=True)
class _Refused:
    """One metric whose retired ``token_form`` the migration cannot write."""

    name: str
    #: the refusal's text when it is the document's only one
    alone: str
    #: its numbered item among several
    item: str
    #: whether its answers are dataset columns, whose reason several share
    column: bool


def _column_refused(name: str, form: str, columns: Mapping[str, str]) -> _Refused:
    """A metric whose answers are dataset columns
    ([`answer_columns`][causalab.protocol.schema.types.answer_columns]): it
    names the metric, each field with its columns, and the table rewrite the
    form stood for."""
    named = "; ".join(f"{field}: {spelled}" for field, spelled in columns.items())
    rewrite = RETIRED_TOKEN_FORM_REWRITES[form]
    return _Refused(
        name=name,
        alone=(
            f"metric {name!r} reads its answers from dataset columns ({named}) "
            f"under the retired token_form {form!r}, and {_COLUMN_REASON}, "
            f"{rewrite}; then delete the key"
        ),
        item=(
            f"metric {name!r} reads its answers from dataset columns ({named}) "
            f"under {form!r}: rewrite each answer string s {rewrite}"
        ),
        column=True,
    )


def _literal_refused(name: str, sentence: str) -> _Refused:
    """A metric with literal answers the migration cannot respell."""
    return _Refused(name=name, alone=sentence, item=sentence, column=False)


def _refusal(refused: list[_Refused]) -> ParseError:
    """One P2 for every metric of a document whose retired ``token_form`` the
    migration cannot write, so one run names every author step. One metric
    keeps its own path. Several are refused at the ``metrics`` section,
    numbered in document order, with the column reason stated once."""
    if len(refused) == 1:
        (only,) = refused
        return ParseError(
            "P2",
            f"{only.alone}, and migrate again (§2.10)",
            path=f"method.metrics.{only.name}.token_form",
        )
    reason = ""
    if any(entry.column for entry in refused):
        reason = (
            " For a metric whose answers are dataset columns, "
            f"{_COLUMN_REASON}, as stated, then delete the key."
        )
    listed = " ".join(
        f"({i}) {entry.item}." for i, entry in enumerate(refused, start=1)
    )
    return ParseError(
        "P2",
        f"{len(refused)} metrics carry a retired token_form the migration "
        f"cannot write into the document.{reason} {listed} Then migrate again "
        "(§2.10)",
        path="method.metrics",
    )


def _retire_token_form(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop a retired ``token_form`` value (§2.10), writing its effect into
    the answer strings it used to rewrite.

    The key once added or stripped an answer's leading space at resolve
    time; now the string is tokenized as written. For a metric whose answers
    are literal strings in the document (`_LITERAL_ANSWER_FIELDS`),
    ``space_prefixed`` puts the space into each string and ``bare`` takes it
    out. This is the normalization the resolver applied, made visible.
    ``auto`` over a literal list picked a form per string with the tokenizer
    in hand, so it cannot be rewritten blind and is refused, naming the
    metric. A literal field held by an artifact reference is refused too,
    because its strings are not in the document.

    A metric whose answers are dataset columns is refused for every retired
    value. The strings live in a table the migration does not read, and the
    resolver rewrote them there: ``"Jennifer"`` under ``space_prefixed``
    scored the gpt2 row of ``" Jennifer"``. The refusal names the metric, its
    column fields and the table rewrite (`_column_refused`). A metric
    with neither kind of answer field loses the key.

    Every metric is checked before anything is raised, so one refusal names
    each metric that needs an author step (`_refusal`). Idempotent, and a
    document without the key is returned as it is (the module docstring:
    nothing else inside a section is inspected). A refusal leaves ``doc``
    unchanged.

    Raises:
        ParseError: P2 at ``method.metrics.<name>.token_form`` for one
            refused metric, at ``method.metrics`` for several: a column
            metric, ``auto`` over literal answers, or literal answers held
            by an artifact reference.
    """
    if not _spells_retired_token_form(doc):
        return doc
    out = copy.deepcopy(doc)
    refused: list[_Refused] = []
    for name, metric in out["method"]["metrics"].items():
        form = metric.get("token_form")
        if form not in RETIRED_TOKEN_FORMS:
            continue
        field = _LITERAL_ANSWER_FIELDS.get(metric.get("kind"))
        held = metric.get(field) if field is not None else None
        if isinstance(held, Mapping) and isinstance(held.get("artifact"), str):
            # an artifact reference is legal anywhere a value is (§1) and
            # resolves after the migration runs, so the strings are not in
            # the document to be rewritten
            refused.append(
                _literal_refused(
                    name,
                    f"metric {name!r} reads its answers from an artifact, so the "
                    f"retired token_form {form!r} cannot be written into the "
                    "strings here. Resolve the artifact and write each answer as "
                    "the model emits it; then drop the key",
                )
            )
        elif isinstance(held, (list, Mapping)):
            if form == "auto":
                refused.append(
                    _literal_refused(
                        name,
                        f"metric {name!r} lists literal answers under token_form "
                        "'auto', which picked each string's form with the "
                        "tokenizer in hand. Write each answer as the model emits "
                        "it (' Seattle' after a space, 'Seattle' glued to the "
                        "text before it); then drop the key",
                    )
                )
            else:
                metric[field] = _respell(metric[field], form)
                del metric["token_form"]
        elif columns := answer_columns(str(metric.get("kind")), metric):
            # the answers are in a table this function never sees
            refused.append(_column_refused(name, form, columns))
        else:
            del metric["token_form"]
    if refused:
        raise _refusal(refused)
    return out


def _respell(node: Any, form: str) -> Any:
    """Every answer string in a literal answer field, in the form the retired
    ``token_form`` value resolved it to: ``bare`` strips leading spaces,
    ``space_prefixed`` strips them and prepends one."""
    if isinstance(node, Mapping):
        return {key: _respell(value, form) for key, value in node.items()}
    if isinstance(node, list):
        return [_respell(item, form) for item in node]
    if isinstance(node, str):
        bare = node.lstrip(" ")
        return " " + bare if form == "space_prefixed" else bare
    return node


# --------------------------------------------------------------------------- #
# v3 → v4: the reads-first method block
# --------------------------------------------------------------------------- #


def _refuse(message: str, path: str) -> ParseError:
    return ParseError("P2", message, path=path)


def _unwritten_name(role: str) -> str:
    """The declared name of the un-intervened model on ``role``:
    ``original_<role>``, an indexed role ``counterfactual[j]`` spelled
    ``original_counterfactual_j``."""
    return f"{_UNWRITTEN}_" + role.replace("[", "_").replace("]", "")


def _positional_term_name(term: Any) -> str | None:
    """The name a positional objective term takes in the named form: a
    ``[w, "<metric>"]`` term its metric's, a ``[w, {"l1": …}]`` regularizer
    its kind; anything else has none, and keeps the objective positional."""
    if not isinstance(term, list) or len(term) != 2:
        return None
    if isinstance(term[1], str):
        return term[1]
    if isinstance(term[1], Mapping):
        kinds = [key for key in term[1] if key in REGULARIZER_KINDS]
        if len(kinds) == 1:
            return kinds[0]
    return None


def _holds_wrapper(node: Any) -> bool:
    """Whether a raw subtree carries a ``sweep`` or ``axis`` wrapper — a
    metric a ``train`` save may not name (§2.12), since the reference would
    move with the term where an inline copy is an axis of its own."""
    if isinstance(node, Mapping):
        return (
            "sweep" in node
            or "axis" in node
            or any(_holds_wrapper(v) for v in node.values())
        )
    if isinstance(node, list):
        return any(_holds_wrapper(v) for v in node)
    return False


def _v3_to_v4(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The reads-first rewrite (module docstring, third step), on a deep copy.

    1. Every read's ``model`` / ``input`` becomes its binding; the fields go.
    2. Each intervened model gains ``reads``, the reads bound to it in
       declaration order, and keeps its ``writes``.
    3. The roles read on the reserved ``original`` become declared models
       with no writes — ``original`` when only ``base`` is read on it, else
       ``original_<role>`` per role — emitted before the authored models.
    4. ``metrics`` dissolves: a save entry that named a metric carries the
       metric as its ``aggregation`` over the metric's read on its model; a
       save entry that named a read keeps the read, its model and ``reduce``.
    5. ``train``: a positional ``[w, "<metric>"]`` and a named ``{"weight",
       "metric"}`` term inline the aggregation; ``eval.metrics`` becomes
       ``eval.aggregations``; ``early_stop.metric`` becomes ``early_stop.on``.
       A metric consumed by nothing is refused. A save of a metric that
       exactly one train term also consumes becomes ``{"train": <name>,
       "file_path"}`` (§2.12), and a positional objective it names takes the
       named form: a metric term under its metric's name, a regularizer
       under its kind (``l1``), when that gives every term a distinct name.
       A metric under a sweep, or a name that is both a term and an eval
       label, keeps the save inline.
    6. A write operand or a ``kl`` / ``js`` target that names a read bound to
       several models is spelled ``{"read", "model"}``; a uniquely bound one
       stays a bare name (never the case for a valid v3 document). Inside a
       save — and on a term a ``train`` save names — the target is always
       ``{"read", "model"}`` (§2.7).
    7. A dotted ``metrics.<name>.<field>`` id in a string is re-spelled
       ``save[i].aggregation.<field>`` when the metric has one save consumer
       (the term's path when that save names a train term), and refused with
       both spellings when it has several.
    8. ``header.protocol_version`` becomes ``"4"``.

    Method keys are re-emitted in the protocol-4 order."""
    out = copy.deepcopy(dict(raw))
    header = dict(out.get("header") or {})
    header["protocol_version"] = PROTOCOL_VERSION
    out["header"] = header
    method = out.get("method")
    if not isinstance(method, Mapping):
        return out
    method = dict(method)
    reads = method.get("reads")
    if not isinstance(reads, Mapping):
        reads = {}
    # 1. bindings
    binding: dict[str, tuple[str, str]] = {}
    new_reads: dict[str, Any] = {}
    for rname, read in reads.items():
        if not isinstance(read, Mapping):
            new_reads[rname] = read
            continue
        model, role = read.get("model"), read.get("input")
        if model is None and role is None:
            new_reads[rname] = read  # already reads-first: listed by a model
            continue
        if not isinstance(model, str) or not isinstance(role, str):
            raise _refuse(
                f"read {rname!r} carries a swept or missing model/input — the "
                "reads-first rewrite binds a read to one model; expand the "
                "sweep into two documents first",
                f"reads.{rname}",
            )
        binding[rname] = (model, role)
        new_reads[rname] = {
            k: v for k, v in read.items() if k not in ("model", "input")
        }
    # 2–3. models
    models_in = method.get("intervened_models")
    models_in = dict(models_in) if isinstance(models_in, Mapping) else {}
    unwritten_roles = list(
        dict.fromkeys(role for model, role in binding.values() if model == _UNWRITTEN)
    )
    unwritten_name: dict[str, str] = {}
    if unwritten_roles == ["base"]:
        unwritten_name["base"] = _UNWRITTEN
    else:
        for role in unwritten_roles:
            unwritten_name[role] = _unwritten_name(role)
    for role, name in unwritten_name.items():
        existing = models_in.get(name)
        if existing is None:
            continue
        if (
            isinstance(existing, Mapping)
            and existing.get("input") == role
            and not existing.get("writes")
        ):
            continue  # the same un-intervened model, already declared: merge
        raise _refuse(
            f"the un-intervened model on {role!r} would be declared as "
            f"{name!r}, which the document already declares; rename that "
            "model first",
            f"intervened_models.{name}",
        )

    def model_of(rname: str) -> str:
        model, role = binding[rname]
        return unwritten_name[role] if model == _UNWRITTEN else model

    models_out: dict[str, Any] = {}
    for role, name in unwritten_name.items():
        existing = models_in.get(name)
        listed = (
            list(existing.get("reads", [])) if isinstance(existing, Mapping) else []
        )
        models_out[name] = {
            "input": role,
            "reads": listed
            + [
                r
                for r, (m, ro) in binding.items()
                if m == _UNWRITTEN and ro == role and r not in listed
            ],
        }
    for name, im in models_in.items():
        if not isinstance(im, Mapping):
            models_out[name] = im
            continue
        if name in models_out:
            continue  # the un-intervened model merged above
        entry: dict[str, Any] = {"input": im.get("input")}
        listed = list(im.get("reads", []))
        entry["reads"] = listed + [
            r for r, (m, _ro) in binding.items() if m == name and r not in listed
        ]
        for key, value in im.items():
            if key not in ("input", "reads"):
                entry[key] = value
        models_out[name] = entry
    # a read bound to a model the document does not declare has no model to
    # be listed on (§2.9); the protocol-3 loader refused it too (rule 5)
    for rname, (model, _role) in binding.items():
        if model != _UNWRITTEN and model not in models_out:
            raise _refuse(
                f"read {rname!r} is bound to model {model!r}, which the "
                "document does not declare",
                f"reads.{rname}.model",
            )
    attached: dict[str, list[str]] = {}
    for name, im in models_out.items():
        if isinstance(im, Mapping):
            for r in im.get("reads", ()):
                attached.setdefault(r, []).append(name)

    def read_ref(rname: Any) -> Any:
        """A read reference: bare when the read is bound to one model."""
        if isinstance(rname, str) and len(attached.get(rname, ())) > 1:
            return {"read": rname, "model": model_for(rname, f"reads.{rname}")}
        return rname

    def model_for(rname: str, path: str) -> str:
        """The model a read is measured on: its protocol-3 binding, else the
        one model that lists it (a read already spelled reads-first)."""
        if rname in binding:
            return model_of(rname)
        listed = attached.get(rname, [])
        if len(listed) == 1:
            return listed[0]
        raise _refuse(
            f"read {rname!r} is listed by {len(listed)} models — name the model",
            path,
        )

    # 6. operands
    writes = method.get("writes")
    if isinstance(writes, Mapping):
        new_writes: dict[str, Any] = {}
        for wname, write in writes.items():
            if isinstance(write, Mapping) and isinstance(write.get("do"), Mapping):
                do = {
                    mech: (
                        {
                            k: (read_ref(v) if k in ("swap", "op") else v)
                            for k, v in payload.items()
                        }
                        if isinstance(payload, Mapping)
                        else read_ref(payload)
                    )
                    for mech, payload in write["do"].items()
                }
                write = {**write, "do": do}
            new_writes[wname] = write
        method["writes"] = new_writes
    # 4. metrics → aggregations
    metrics = method.pop("metrics", {})
    metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
    consumers: dict[str, list[str]] = {name: [] for name in metrics}
    save = method.get("save")

    def aggregated(name: str, path: str) -> dict[str, Any]:
        metric = metrics.get(name)
        if metric is None:
            # a document already reads-first in part: the name is a saved
            # table's file stem, and the term restates that entry's aggregation
            for entry in save if isinstance(save, list) else ():
                if not isinstance(entry, Mapping) or "aggregation" not in entry:
                    continue
                stem = str(entry.get("file_path", "")).rsplit("/", 1)[-1]
                if stem.rsplit(".", 1)[0] == name:
                    return {
                        key: copy.deepcopy(entry[key])
                        for key in ("read", "model", "aggregation")
                    }
        if not isinstance(metric, Mapping) or "of" not in metric:
            raise _refuse(f"{name!r} is not a metric of this document", path)
        of = metric["of"]
        if not isinstance(of, str) or of not in new_reads:
            raise _refuse(
                f"metric {name!r} reduces {of!r}, which is not a read",
                f"metrics.{name}.of",
            )
        aggregation = {
            k: (
                read_ref(v)
                if k == "target" and isinstance(v, str) and v in new_reads
                else v
            )
            for k, v in metric.items()
            if k != "of"
        }
        return {
            "read": of,
            "model": model_for(of, f"metrics.{name}.of"),
            "aggregation": aggregation,
        }

    def qualified(bound: dict[str, Any]) -> dict[str, Any]:
        """``bound`` with a ``kl`` / ``js`` target in the object form, the
        one spelling a save entry holds (§2.7)."""
        aggregation = bound["aggregation"]
        target = aggregation.get("target")
        if aggregation.get("kind") not in ("kl", "js") or not isinstance(target, str):
            return bound
        spelled = {"read": target, "model": model_for(target, f"reads.{target}")}
        return {**bound, "aggregation": {**aggregation, "target": spelled}}

    new_save: list[Any] = []
    #: new_save index → the metric that entry tables, for step 5's references
    saved_metric: dict[int, str] = {}
    if isinstance(save, list):
        for i, entry in enumerate(save):
            if (
                not isinstance(entry, Mapping)
                or "kind" in entry
                or "site" in entry
                or "read" in entry  # already reads-first
            ):
                new_save.append(entry)
                continue
            value = entry.get("value")
            if value in metrics:
                consumers[value].append(f"save[{i}]")
                bound = qualified(aggregated(value, f"save[{i}].value"))
                item = {**bound, "file_path": entry.get("file_path")}
                if "reduce" in entry:
                    item["reduce"] = entry["reduce"]  # the parser refuses it, by name
                else:
                    saved_metric[len(new_save)] = str(value)
                new_save.append(item)
            elif value in new_reads:
                item: dict[str, Any] = {
                    "read": value,
                    "model": model_for(value, f"save[{i}].value"),
                    "file_path": entry.get("file_path"),
                }
                if "reduce" in entry:
                    item["reduce"] = entry["reduce"]
                new_save.append(item)
            else:
                raise _refuse(
                    f"save entry names {value!r}, which is neither a metric nor a read",
                    f"save[{i}].value",
                )
    # 5. train
    train = method.get("train")
    #: new_save index → the train term a `{"train": name}` entry now names
    references: dict[int, str] = {}
    if isinstance(train, Mapping):
        train = dict(train)
        objective = train.get("objective")
        #: objective term name → the metric it reduces; a positional metric
        #: term goes by its metric's name, which the named form will give it
        term_metric: dict[str, str] = {}
        positional: list[tuple[str, Any]] | None = None
        if isinstance(objective, list):
            terms: list[Any] = []
            for i, term in enumerate(objective):
                if (
                    isinstance(term, list)
                    and len(term) == 2
                    and isinstance(term[1], str)
                ):
                    consumers.setdefault(term[1], []).append(f"train.objective[{i}]")
                    terms.append(
                        [term[0], aggregated(term[1], f"train.objective[{i}]")]
                    )
                else:
                    terms.append(term)
            train["objective"] = terms
            names = [
                name
                for name in (_positional_term_name(t) for t in objective)
                if name is not None
            ]
            if len(names) == len(objective) and len(set(names)) == len(names):
                # every term has a name the named form can give it
                positional = list(zip(names, terms))
                term_metric = {
                    name: t[1]
                    for name, t in zip(names, objective)
                    if isinstance(t[1], str)
                }
        elif isinstance(objective, Mapping):
            named: dict[str, Any] = {}
            for tname, term in objective.items():
                if isinstance(term, Mapping) and isinstance(term.get("metric"), str):
                    term_metric[tname] = term["metric"]
                    consumers.setdefault(term["metric"], []).append(
                        f"train.objective.{tname}"
                    )
                    bound = aggregated(
                        term["metric"], f"train.objective.{tname}.metric"
                    )
                    named[tname] = {
                        **{k: v for k, v in term.items() if k != "metric"},
                        **bound,
                    }
                else:
                    named[tname] = term
            train["objective"] = named
        eval_spec = train.get("eval")
        if isinstance(eval_spec, Mapping) and isinstance(
            eval_spec.get("metrics"), list
        ):
            eval_out: dict[str, Any] = {}
            for key, value in eval_spec.items():
                if key == "metrics":
                    eval_out["aggregations"] = {}
                    for name in value:
                        consumers.setdefault(name, []).append(
                            f"train.eval.aggregations.{name}"
                        )
                        eval_out["aggregations"][name] = aggregated(
                            name, f"train.eval.aggregations.{name}"
                        )
                else:
                    eval_out[key] = value
            train["eval"] = eval_out
        early = train.get("early_stop")
        if isinstance(early, Mapping) and "metric" in early:
            train["early_stop"] = {
                ("on" if key == "metric" else key): value
                for key, value in early.items()
            }
        # a save of a metric the fit also consumes names that term
        eval_labels = (
            list(eval_spec["metrics"])
            if isinstance(eval_spec, Mapping)
            and isinstance(eval_spec.get("metrics"), list)
            else []
        )
        for i, metric in saved_metric.items():
            candidates = [t for t, m in term_metric.items() if m == metric]
            candidates += [metric] if metric in eval_labels else []
            if (
                len(candidates) != 1
                or (candidates[0] in term_metric and candidates[0] in eval_labels)
                or _holds_wrapper(metrics.get(metric))
            ):
                continue  # no one term to name: the entry stays inline
            (name,) = candidates
            if name in term_metric:
                if positional is not None:
                    # the named form, so the save has a name to give (§2.11)
                    train["objective"] = {
                        tname: {"weight": t[0], **t[1]} for tname, t in positional
                    }
                    positional = None
                owner = f"train.objective.{name}"
                named_terms = train["objective"]
                assert isinstance(named_terms, dict)  # a term name: the named form
                named_terms[name] = qualified(named_terms[name])
            else:
                owner = f"train.eval.aggregations.{name}"
                train["eval"]["aggregations"][name] = qualified(
                    train["eval"]["aggregations"][name]
                )
            new_save[i] = {"train": name, "file_path": new_save[i]["file_path"]}
            references[i] = owner
        method["train"] = train
    unconsumed = [name for name, owners in consumers.items() if not owners]
    if unconsumed:
        raise _refuse(
            f"metric(s) {unconsumed} are consumed by no save entry, objective "
            "term or eval entry — protocol 4 has no free-standing metric; save "
            "it, consume it, or drop it",
            f"metrics.{unconsumed[0]}",
        )
    # re-emit
    method["reads"] = new_reads
    method["intervened_models"] = models_out
    if isinstance(save, list):
        method["save"] = new_save
    out["method"] = {
        section: method[section] for section in _V4_METHOD_SECTIONS if section in method
    }
    for key in method:
        if key not in _V4_METHOD_SECTIONS:
            out["method"][key] = method[key]
    # 7. dotted ids
    save_owner = {
        name: [
            references.get(int(o[len("save[") : -1]), o)
            for o in owners
            if o.startswith("save[")
        ]
        for name, owners in consumers.items()
    }
    return _rewrite_metric_ids(out, save_owner)


def _rewrite_metric_ids(node: Any, save_owner: Mapping[str, list[str]]) -> Any:
    """Every ``metrics.<name>.<field>`` dotted id in a tree — as a key or a
    string value — re-spelled ``save[i].aggregation.<field>`` for the one
    save entry that consumes the metric; refused when the metric has several
    consumers or none that saves it."""

    def respell(match: re.Match[str]) -> str:
        name, field = match.group(1), match.group(2)
        owners = save_owner.get(name, [])
        if len(owners) != 1:
            spellings = [f"{o}.aggregation.{field}" for o in owners] or [
                f"save[<i>].aggregation.{field}",
                f"train.objective.<term>.aggregation.{field}",
            ]
            raise _refuse(
                f"the dotted id {match.group(0)!r} has no single spelling under "
                f"protocol 4 — metric {name!r} is consumed by {len(owners)} save "
                f"entries; write one of {spellings}",
                match.group(0),
            )
        return f"{owners[0]}.aggregation.{field}"

    if isinstance(node, Mapping):
        return {
            (_DOTTED_METRIC.sub(respell, key) if isinstance(key, str) else key): (
                _rewrite_metric_ids(value, save_owner)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_metric_ids(item, save_owner) for item in node]
    if isinstance(node, str):
        return _DOTTED_METRIC.sub(respell, node)
    return node


def _rename_layer(entry: Mapping[str, Any]) -> dict[str, Any]:
    """``layer`` → ``layers`` in one site (or window selector), in place in
    the key order; a bare index becomes the one-layer band, a wrapper keeps
    its wrapper."""
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key == "layer":
            if isinstance(value, int) and not isinstance(value, bool):
                value = [value]
            out["layers"] = value
        else:
            out[key] = value
    return out


def _rewrite_dotted(node: Any) -> Any:
    """Every ``sites.<name>.layer`` dotted id in a tree — as a key or a string
    value — re-spelled ``sites.<name>.layers``; everything else as it is."""
    if isinstance(node, Mapping):
        return {
            (
                _DOTTED_LAYER.sub(r"sites.\1.layers", key)
                if isinstance(key, str)
                else key
            ): _rewrite_dotted(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_dotted(item) for item in node]
    if isinstance(node, str):
        return _DOTTED_LAYER.sub(r"sites.\1.layers", node)
    return node


def _spells_metric_ids(node: Any) -> bool:
    if isinstance(node, Mapping):
        return any(
            (isinstance(key, str) and _DOTTED_METRIC.search(key) is not None)
            or _spells_metric_ids(value)
            for key, value in node.items()
        )
    if isinstance(node, list):
        return any(_spells_metric_ids(item) for item in node)
    return isinstance(node, str) and _DOTTED_METRIC.search(node) is not None


def _spells_old_dotted(node: Any) -> bool:
    if isinstance(node, Mapping):
        return any(
            (isinstance(key, str) and _DOTTED_LAYER.search(key) is not None)
            or _spells_old_dotted(value)
            for key, value in node.items()
        )
    if isinstance(node, list):
        return any(_spells_old_dotted(item) for item in node)
    return isinstance(node, str) and _DOTTED_LAYER.search(node) is not None


# --------------------------------------------------------------------------- #
# the authoring format
# --------------------------------------------------------------------------- #


def format_document(document: Mapping[str, Any], *, width: int = LINE_WIDTH) -> str:
    """One document as the text the repository authors it in: two-space
    indentation, key order preserved, and any object or list that is at most
    two levels deep and fits in ``width`` columns written on one line — so a
    site, a read, a write or a save entry is one line, and a table of them is
    one entry per line. The output ends with a newline."""
    return _render(document, 0, width) + "\n"


def _depth(node: Any) -> int:
    if isinstance(node, Mapping):
        return 1 + max((_depth(v) for v in node.values()), default=0)
    if isinstance(node, list):
        return 1 + max((_depth(v) for v in node), default=0)
    return 0


def _inline(node: Any) -> str:
    return json.dumps(node, ensure_ascii=False, separators=(", ", ": "))


def _render(node: Any, indent: int, width: int) -> str:
    if not isinstance(node, (Mapping, list)) or not node:
        return _inline(node)
    flat = _inline(node)
    if _depth(node) <= 2 and indent + len(flat) <= width:
        return flat
    pad = " " * (indent + 2)
    close = " " * indent
    if isinstance(node, Mapping):
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {_render(value, indent + 2, width)}"
            for key, value in node.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + close + "}"
    items = [f"{pad}{_render(value, indent + 2, width)}" for value in node]
    return "[\n" + ",\n".join(items) + "\n" + close + "]"


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #

#: A fenced ``json`` block: the opening fence (any indentation, any info string
#: after ``json``), its body, the closing fence at the same indentation.
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```json\b(?P<info>[^\n]*)\n(?P<body>.*?)\n(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def migrate_markdown(text: str) -> str:
    """``text`` with every fenced ``json`` block that is a whole document
    needing migration ([`needs_migration`][]) rewritten at the current
    version (the fence lines and their indentation kept). A block that is not
    JSON or is already current is left exactly as it is: a fragment with an
    ellipsis, a run receipt. A block the rewrite refuses is left as it is
    too; ``causalab migrate`` reports each one with its line and exits 1, as
    it does for a refused ``.json`` file."""
    return _migrate_markdown(text)[0]


def _migrate_markdown(text: str) -> tuple[str, list[tuple[int, ParseError]]]:
    """[`migrate_markdown`][]'s text, with each refused block's opening-fence
    line (1-based) and its refusal, in page order."""
    refused: list[tuple[int, ParseError]] = []

    def rewrite(match: re.Match[str]) -> str:
        indent = match.group("indent")
        body = match.group("body")
        lines = body.split("\n")
        stripped = "\n".join(
            line[len(indent) :] if line.startswith(indent) else line for line in lines
        )
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError:
            return match.group(0)
        if not needs_migration(raw):
            return match.group(0)
        try:
            migrated = migrate_document(raw)
        except ParseError as err:
            # an old document the author has to finish, such as a column
            # metric under a retired token_form or a split v1 example
            refused.append((text.count("\n", 0, match.start()) + 1, err))
            return match.group(0)
        rendered = format_document(migrated).rstrip("\n")
        body_out = "\n".join(indent + line for line in rendered.split("\n"))
        return f"{indent}```json{match.group('info')}\n{body_out}\n{indent}```"

    return _FENCE.sub(rewrite, text), refused


# --------------------------------------------------------------------------- #
# the verb
# --------------------------------------------------------------------------- #


def _rewrite(path: Path) -> tuple[str | None, list[tuple[int, ParseError]]]:
    """The migrated text of one file, or ``None`` when nothing would change,
    and a Markdown file's refused blocks (`_migrate_markdown`).

    JSON and markdown only. A ``.yaml`` document is an authoring surface the
    loader reads, but a rewrite of it would drop its comments and formatting
    (the reasons to author YAML), so it is refused with that reason, and the
    author regroups it by hand (or converts it to JSON first). A refused
    JSON document raises.
    """
    if path.suffix in (".yaml", ".yml"):
        raise ParseError(
            "P2",
            "a YAML document is not rewritten in place: the migration would drop "
            "its comments and formatting. Regroup it by hand (§1), or convert it "
            "to JSON and migrate that",
            path=str(path),
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        out, refused = _migrate_markdown(text)
        return (None if out == text else out), refused
    raw = load_raw(text)
    if not needs_migration(raw):
        return None, []  # current version, or a workflow with nothing to re-spell
    return format_document(migrate_document(raw)), []


def main(args: argparse.Namespace) -> int:
    """``causalab migrate PATH... [--check]``: rewrite in place, or with
    ``--check`` report what would change and exit 1 if anything would. A
    refused file, or a refused block of a Markdown file, is reported on
    stderr and exits 1."""
    changed = 0
    refused = 0
    for path in args.paths:
        try:
            out, blocks = _rewrite(path)
        except (ProtocolError, OSError) as err:
            # a missing path or a directory is a refusal like a malformed
            # file: the files before it in `paths` stay migrated, and the
            # exit code says the run was not clean
            print(f"refused: {path}: {err}", file=sys.stderr)
            refused += 1
            continue
        for line, err in blocks:
            # the page's other blocks are still rewritten below, as the
            # files before a refused path stay migrated
            print(f"refused: {path}: line {line}: {err}", file=sys.stderr)
            refused += 1
        if out is None:
            continue
        changed += 1
        if args.check:
            print(f"would migrate {path}")
        else:
            path.write_text(out, encoding="utf-8")
            print(f"migrated {path}")
    if refused:
        return 1
    if args.check and changed:
        return 1
    return 0
