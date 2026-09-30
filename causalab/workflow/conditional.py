"""Define decisions, conditional steps, and required receipts.

Decision rules compare declared values and write typed outcomes. A conditional
step uses an outcome to select which dependent steps run. Required receipts
are checked before allocating a step. Reuse checks include the identity of
the evidence used to make the decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Mapping

from causalab.io.step_io import read_values
from causalab.io.step_record import read_sidecar
from causalab.protocol.rules.errors import ProtocolError, suggest
from causalab.workflow.behavioral import (
    DECISION_FILE,
    DECISION_SCHEMA_VERSION,
    DECISION_TYPES,
)
from causalab.workflow.document import (
    BehavioralStep,
    ConditionalStep,
    DecisionStep,
    LoadedWorkflow,
    Reference,
    Step,
    WorkflowError,
)

#: A skipped step's manifest entry names the decision that skipped it (§8) by
#: the six [`causalab.workflow.manifest.SKIPPED_BY_FIELDS`][] — manifest
#: vocabulary, re-exported here (``__all__``): [`skipped_entry`][] (a reached
#: skip) and ``manifest.classify_unreached`` (an unreached one) write the same
#: keys.
from causalab.workflow.manifest import CHILD_SEPARATOR, SKIPPED_BY_FIELDS, skipped_via

__all__ = [
    "COMPARATORS",
    "CONDITIONAL_RULE",
    "DECISION_FIELDS",
    "EXECUTABLE_SCOPES",
    "FIELD_VOCABULARIES",
    "PREDICATE_COMPARATORS",
    "PRODUCER_STEPS",
    "RECEIPT_OUTCOMES",
    "SCOPES",
    "SKIPPED_BY_FIELDS",
    "check_conditional",
    "check_receipt",
    "dependency_edges",
    "evaluate_predicate",
    "evidence_holds",
    "fold_skips",
    "holds",
    "parse_conditional",
    "parse_decision",
    "parse_requires_receipt",
    "read_decision",
    "run_conditional_step",
    "run_decision_step",
]

#: The checklist rule this layer refuses under (§5): a decision, a conditional
#: and a receipt are typed and bound (§2.8).
CONDITIONAL_RULE = 18

#: What a ``decision`` step's rule may say about one declared key — exactly
#: one per key, with a JSON literal operand (``in`` takes a list). No
#: expression language, no arithmetic, no reference to another step.
COMPARATORS: tuple[str, ...] = ("eq", "ne", "lt", "le", "gt", "ge", "in")

#: What a conditional's predicate may say about a decision field: the field's
#: vocabulary is closed, so an order comparison would mean nothing.
PREDICATE_COMPARATORS: tuple[str, ...] = ("eq", "ne", "in")

#: The scopes a conditional declares (9·7). ``global`` decides for the whole
#: run; ``per_target`` and ``per_variable`` decide per child of a declared
#: fan-out (§2.9). Every scope executes, and the
#: two tuples are held equal by the census.
SCOPES: tuple[str, ...] = ("global", "per_target", "per_variable")
EXECUTABLE_SCOPES: tuple[str, ...] = SCOPES

#: The outcomes a receipt carries, and a ``requires_receipt`` may demand.
RECEIPT_OUTCOMES: tuple[str, ...] = ("pass", "fail")

#: The fields of a ``decision.json`` a predicate may read, each with its
#: closed vocabulary.
DECISION_FIELDS: tuple[str, ...] = ("outcome", "decision_type")
FIELD_VOCABULARIES: Mapping[str, tuple[str, ...]] = {
    "outcome": RECEIPT_OUTCOMES,
    "decision_type": DECISION_TYPES,
}

#: The step kinds that write a ``decision.json`` a predicate or a receipt reads.
PRODUCER_STEPS = (BehavioralStep, DecisionStep)

_DECISION_KEYS = ("on_pass", "on_fail")
_ORDERED = ("lt", "le", "gt", "ge")
_ROLE = {"on_pass": "pass", "on_fail": "fail"}


def _refuse(message: str, path: str) -> WorkflowError:
    return WorkflowError(CONDITIONAL_RULE, message, path=path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_literal(value: Any) -> bool:
    """A JSON scalar: what a rule may compare against."""
    return value is None or isinstance(value, (str, int, float, bool))


# --------------------------------------------------------------------------- #
# parsing (§2.8, rule 18)
# --------------------------------------------------------------------------- #


def parse_decision(
    raw: Mapping[str, Any],
    path: str,
    *,
    values: Reference,
    requires_receipt: Mapping[str, str] | None,
    after: tuple[str, ...],
    description: str | None,
) -> DecisionStep:
    """The decision half of a step's parse: ``rule`` and ``decision`` (§2.8),
    each refused under rule 18 naming the field. ``values`` arrives parsed
    from ``document._parse_step``, which owns the reference grammar, the key
    set, ``requires_receipt`` and ``after``."""
    return DecisionStep(
        type="decision",
        values=values,
        rule=_parse_rule(raw["rule"], f"{path}.rule"),
        decision=_parse_decision_map(raw["decision"], f"{path}.decision"),
        requires_receipt=requires_receipt,
        after=after,
        description=description,
    )


def _parse_rule(raw: Any, path: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise _refuse(
            "'rule' maps each key the values object declares to one comparator "
            'and a literal: {"best_k": {"le": 16}, "best_seed": {"in": [0, 1]}}',
            path,
        )
    out: dict[str, dict[str, Any]] = {}
    for key, clause in raw.items():
        where = f"{path}.{key}"
        if not isinstance(clause, Mapping):
            raise _refuse(
                f"the rule for {key!r} is {{<comparator>: <literal>}} with one "
                f"comparator from {list(COMPARATORS)}, got {clause!r}",
                where,
            )
        if len(clause) != 1:
            raise _refuse(
                f"the rule for {key!r} carries exactly one comparator, got "
                f"{sorted(map(str, clause))} — no expression language, no "
                "arithmetic",
                where,
            )
        ((comparator, operand),) = clause.items()
        if comparator not in COMPARATORS:
            raise _refuse(
                f"comparator {comparator!r} is not one of {list(COMPARATORS)}"
                f"{suggest(str(comparator), COMPARATORS)}",
                where,
            )
        if comparator == "in":
            if (
                not isinstance(operand, list)
                or not operand
                or not all(_is_literal(item) for item in operand)
            ):
                raise _refuse(
                    f"'in' takes a non-empty list of JSON literals, got {operand!r}",
                    where,
                )
            out[str(key)] = {"in": list(operand)}
            continue
        if comparator in _ORDERED and not _is_number(operand):
            raise _refuse(
                f"{comparator!r} compares against a number, got {operand!r}", where
            )
        if not _is_literal(operand):
            raise _refuse(
                f"{comparator!r} compares against a JSON literal (a string, a "
                f"number, a boolean or null), got a {type(operand).__name__} — a "
                "rule holds no expression, no arithmetic and no reference to "
                "another step",
                where,
            )
        out[str(key)] = {str(comparator): operand}
    return out


def _parse_decision_map(raw: Any, path: str) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise _refuse(
            f"'decision' is {{on_pass, on_fail}}, each one of {list(DECISION_TYPES)}",
            path,
        )
    for key in raw:
        if key not in _DECISION_KEYS:
            raise _refuse(
                f"unknown key {key!r}{suggest(str(key), _DECISION_KEYS)}", path
            )
    out: dict[str, str] = {}
    for key in _DECISION_KEYS:
        if key not in raw:
            raise _refuse(
                f"'decision' declares {key!r} — what a {_ROLE[key]} does to the "
                f"question ({' | '.join(DECISION_TYPES)})",
                f"{path}.{key}",
            )
        value = raw[key]
        if not isinstance(value, str) or value not in DECISION_TYPES:
            raise _refuse(
                f"decision {key} {value!r} is not one of {list(DECISION_TYPES)}"
                f"{suggest(str(value), DECISION_TYPES)}",
                f"{path}.{key}",
            )
        out[key] = value
    return out


#: The blocks a conditional must author (rule 18), and why each exists.
_CONDITIONAL_REQUIRED: Mapping[str, str] = {
    "predicate": "which decision field it reads, and the literal it holds it to",
    "on_true": "the steps that run when the predicate holds (the other side is skipped)",
    "on_false": "the steps that run when it does not",
    "scope": f"what the verdict decides for ({' | '.join(SCOPES)}; 'global' in this version)",
}


def parse_conditional(
    raw: Mapping[str, Any],
    path: str,
    *,
    requires_receipt: Mapping[str, str] | None,
    after: tuple[str, ...],
    description: str | None,
) -> ConditionalStep:
    """The conditional's parse: ``predicate``, ``on_true``, ``on_false`` and
    ``scope`` (§2.8), each refused under rule 18 naming the field."""
    for key, what in _CONDITIONAL_REQUIRED.items():
        if key not in raw:
            raise _refuse(f"a conditional declares {key!r} — {what}", f"{path}.{key}")
    on_true = _parse_side(raw["on_true"], "on_true", f"{path}.on_true")
    on_false = _parse_side(raw["on_false"], "on_false", f"{path}.on_false")
    shared = sorted(set(on_true) & set(on_false))
    if shared:
        raise _refuse(
            f"on_true and on_false both name {shared} — the two sides are "
            "disjoint: a step is gated by one verdict, not both",
            f"{path}.on_false",
        )
    scope = raw["scope"]
    if not isinstance(scope, str) or scope not in SCOPES:
        raise _refuse(
            f"scope {scope!r} is not one of {list(SCOPES)}"
            f"{suggest(str(scope), SCOPES)}",
            f"{path}.scope",
        )
    return ConditionalStep(
        type="conditional",
        predicate=_parse_predicate(raw["predicate"], f"{path}.predicate"),
        on_true=on_true,
        on_false=on_false,
        scope=scope,
        requires_receipt=requires_receipt,
        after=after,
        description=description,
    )


def _parse_predicate(raw: Any, path: str) -> dict[str, Any]:
    shape = (
        '\'predicate\' is {"decision": {"step": S}, "field": '
        f"{' | '.join(DECISION_FIELDS)}, "
        f"{' | '.join(PREDICATE_COMPARATORS)}: <literal from that field's "
        "vocabulary>}"
    )
    if not isinstance(raw, Mapping):
        raise _refuse(shape, path)
    allowed = ("decision", "field", *PREDICATE_COMPARATORS)
    for key in raw:
        if key not in allowed:
            raise _refuse(f"unknown key {key!r}{suggest(str(key), allowed)}", path)
    for key in ("decision", "field"):
        if key not in raw:
            raise _refuse(f"missing required key {key!r} — {shape}", path)
    decision = raw["decision"]
    if (
        not isinstance(decision, Mapping)
        or set(decision) != {"step"}
        or not isinstance(decision["step"], str)
        or not decision["step"]
    ):
        raise _refuse(
            "'decision' names the producer whose decision.json the predicate "
            'reads: {"step": S} — a behavioral or a decision step',
            f"{path}.decision",
        )
    field = raw["field"]
    if not isinstance(field, str) or field not in DECISION_FIELDS:
        raise _refuse(
            f"field {field!r} is not one of {list(DECISION_FIELDS)}"
            f"{suggest(str(field), DECISION_FIELDS)} — a predicate reads the "
            "typed fields of a decision, never a measured number",
            f"{path}.field",
        )
    present = [c for c in PREDICATE_COMPARATORS if c in raw]
    if len(present) != 1:
        raise _refuse(
            f"a predicate carries exactly one of {list(PREDICATE_COMPARATORS)}, "
            f"got {present or 'none'}",
            path,
        )
    comparator = present[0]
    vocabulary = FIELD_VOCABULARIES[field]
    operand = raw[comparator]
    where = f"{path}.{comparator}"
    if comparator == "in":
        if not isinstance(operand, list) or not operand:
            raise _refuse("'in' takes a non-empty list of literals", where)
        for item in operand:
            if not isinstance(item, str) or item not in vocabulary:
                raise _refuse(
                    f"{item!r} is not one of {field}'s vocabulary {list(vocabulary)}"
                    f"{suggest(str(item), vocabulary)}",
                    where,
                )
        return {
            "decision": {"step": decision["step"]},
            "field": field,
            "in": list(operand),
        }
    if not isinstance(operand, str) or operand not in vocabulary:
        raise _refuse(
            f"{operand!r} is not one of {field}'s vocabulary {list(vocabulary)}"
            f"{suggest(str(operand), vocabulary)}",
            where,
        )
    return {"decision": {"step": decision["step"]}, "field": field, comparator: operand}


def _parse_side(raw: Any, side: str, path: str) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)) or not raw:
        raise _refuse(
            f"{side!r} is a non-empty list of step names — a conditional with "
            "nothing on one side decides nothing",
            path,
        )
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise _refuse(f"{side!r} names steps, got {item!r}", path)
        if item in names:
            raise _refuse(f"{side!r} names {item!r} twice", path)
        names.append(item)
    return tuple(names)


def parse_requires_receipt(raw: Any, path: str) -> dict[str, str]:
    """A step's ``requires_receipt`` block (§2.8): the producer whose receipt
    the step's allocation waits on, and the outcome it must carry."""
    shape = (
        '\'requires_receipt\' is {"step": S, "outcome": '
        f"{' | '.join(RECEIPT_OUTCOMES)}}} — the producer whose decision.json "
        "must carry that outcome before this step is allocated"
    )
    if not isinstance(raw, Mapping):
        raise _refuse(shape, path)
    for key in raw:
        if key not in ("step", "outcome"):
            raise _refuse(
                f"unknown key {key!r}{suggest(str(key), ('step', 'outcome'))}", path
            )
    for key in ("step", "outcome"):
        if key not in raw:
            raise _refuse(f"missing required key {key!r} — {shape}", f"{path}.{key}")
    step = raw["step"]
    if not isinstance(step, str) or not step:
        raise _refuse(f"'step' is a step name, got {step!r}", f"{path}.step")
    outcome = raw["outcome"]
    if not isinstance(outcome, str) or outcome not in RECEIPT_OUTCOMES:
        raise _refuse(
            f"outcome {outcome!r} is not one of {list(RECEIPT_OUTCOMES)}"
            f"{suggest(str(outcome), RECEIPT_OUTCOMES)}",
            f"{path}.outcome",
        )
    return {"step": step, "outcome": outcome}


# --------------------------------------------------------------------------- #
# load time: the derived edges and rule 18
# --------------------------------------------------------------------------- #


def dependency_edges(
    name: str, step: Step, steps: Mapping[str, Step]
) -> list[tuple[str, str]]:
    """The schedule edges one step derives (§2.8, §3) as ``(dependent,
    upstream)`` pairs: a receipt-bearing step after its receipt's producer, a
    decision after the step whose values it reads, a conditional after its
    predicate's producer, and every step a conditional gates after the
    conditional — so a gated step that is already upstream of its conditional
    closes a cycle rule 5 refuses. Every unknown name is refused here, under
    rule 18, naming the field."""
    edges: list[tuple[str, str]] = []

    def named(target: Any, field: str) -> str:
        target = str(target)
        if target not in steps:
            raise _refuse(
                f"{field!r} names unknown step {target!r}"
                f"{suggest(target, sorted(steps))}",
                f"steps.{name}.{field}",
            )
        if target == name:
            raise _refuse(f"{field!r} names the step itself", f"steps.{name}.{field}")
        return target

    receipt = step.requires_receipt
    if receipt is not None:
        edges.append((name, named(receipt["step"], "requires_receipt.step")))
    if isinstance(step, DecisionStep):
        edges.append((name, named(step.values.step, "values.step")))
    elif isinstance(step, ConditionalStep):
        producer = named(step.predicate["decision"]["step"], "predicate.decision.step")
        edges.append((name, producer))
        for side in ("on_true", "on_false"):
            for gated in getattr(step, side):
                gated = named(gated, side)
                if gated == producer:
                    raise _refuse(
                        f"{side!r} names {gated!r}, the predicate's own producer "
                        "— a conditional cannot gate the decision it reads",
                        f"steps.{name}.{side}",
                    )
                edges.append((gated, name))
    return edges


def check_conditional(
    steps: Mapping[str, Step], dependencies: Mapping[str, Collection[str]]
) -> None:
    """Rule 18 (§2.8, §5), after the schedule: every predicate and every
    receipt names a step that writes a ``decision.json`` — a ``behavioral``
    or a ``decision`` step — every conditional's scope is one this
    version executes, and a conditional's two sides are **dependency-disjoint**
    over ``dependencies`` (the schedule's map, derived edges included): no
    step of one side depends, directly or transitively, on a step of the
    other — a verdict skipping the one would skip the other with it
    (``fold_skips``), and the conditional could never launch the side it was
    authored to launch; a per-child scope is further held to its producer's
    fan-out (§2.9, rule 19 — [`causalab.workflow.fan_out.check_scope`][],
    on the authored conditional; its expanded children are the check's
    consequence, not its subject). Unknown names were refused by
    [`dependency_edges`][]; a gated step upstream of its conditional by rule
    5; a decision's keys by rule 4 against the producer's declaration."""
    from causalab.workflow.fan_out import check_scope

    for name, step in steps.items():
        receipt = step.requires_receipt
        if receipt is not None:
            producer = steps[receipt["step"]]
            if not isinstance(producer, PRODUCER_STEPS):
                raise _refuse(
                    f"'requires_receipt' names {receipt['step']!r}, a "
                    f"{producer.type} step — a receipt is a {DECISION_FILE}, "
                    "written by a behavioral or a decision step",
                    f"steps.{name}.requires_receipt.step",
                )
        if not isinstance(step, ConditionalStep):
            continue
        producer_name = str(step.predicate["decision"]["step"])
        producer = steps[producer_name]
        if not isinstance(producer, PRODUCER_STEPS):
            raise _refuse(
                f"the predicate reads {producer_name!r}, a {producer.type} step — "
                f"a predicate reads a {DECISION_FILE}, written by a behavioral or "
                "a decision step; a metric table or a values object is held to a "
                "rule by a decision step, never read by a predicate",
                f"steps.{name}.predicate.decision.step",
            )
        if step.scope not in EXECUTABLE_SCOPES:
            raise _refuse(
                f"scope {step.scope!r} does not execute in this version — only "
                f"{list(EXECUTABLE_SCOPES)} run",
                f"steps.{name}.scope",
            )
        _check_sides_disjoint(name, step, dependencies)
        if CHILD_SEPARATOR not in name:
            check_scope(name, step, steps)


def _check_sides_disjoint(
    name: str, step: ConditionalStep, dependencies: Mapping[str, Collection[str]]
) -> None:
    """Rule 18: no step of ``on_true`` depends on a step of ``on_false`` or
    vice versa, transitively over the schedule's dependency map."""
    for side, other in (("on_true", "on_false"), ("on_false", "on_true")):
        across = set(getattr(step, other))
        for gated in getattr(step, side):
            crossed = sorted(_ancestors(dependencies, gated) & across)
            if crossed:
                raise _refuse(
                    f"{side!r} names {gated!r}, which depends on {crossed} in "
                    f"{other!r} — the two sides must be dependency-disjoint: the "
                    f"verdict that skips {other!r} would skip {gated!r} with it, "
                    f"so the conditional could never launch the side it was "
                    "authored to launch",
                    f"steps.{name}.{side}",
                )


def _ancestors(dependencies: Mapping[str, Collection[str]], name: str) -> set[str]:
    """Every step ``name`` depends on, transitively."""
    seen: set[str] = set()
    stack = list(dependencies.get(name, ()))
    while stack:
        upstream = stack.pop()
        if upstream in seen:
            continue
        seen.add(upstream)
        stack.extend(dependencies.get(upstream, ()))
    return seen


# --------------------------------------------------------------------------- #
# run time: the decision, the verdict, the skips
# --------------------------------------------------------------------------- #


def holds(value: Any, clause: Mapping[str, Any], *, what: str) -> bool:
    """One rule clause against one measured value."""
    ((comparator, operand),) = clause.items()
    if comparator == "in":
        return value in operand
    if comparator == "eq":
        return value == operand
    if comparator == "ne":
        return value != operand
    if not _is_number(value):
        raise ProtocolError(
            "P2",
            f"{what}: {comparator!r} compares a number, but the measured value "
            f"is {value!r}",
        )
    if comparator == "lt":
        return value < operand
    if comparator == "le":
        return value <= operand
    if comparator == "gt":
        return value > operand
    return value >= operand


def run_decision_step(
    name: str,
    step: DecisionStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    step_dir: Path,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the producer's values object, hold each rule key to its clause,
    write ``decision.json`` into ``step_dir`` (an attempt directory) and
    return the step's record (§8). The measured inputs are the rule's keys
    only — the record's shape is declared; the rule is authored verbatim; the
    evidence identity is the sha256 of the values file's bytes joined to the
    producer's identity, read from the record beside it."""
    values_path = run_root / str(step.values.step) / str(step.values.file)
    what = f"step {name!r}: values {step.values.target}"
    if not values_path.is_file():
        raise ProtocolError("P2", f"{what} does not exist at {str(values_path)!r}")
    values = read_values(values_path)
    identity = read_sidecar(values_path).get("identity")
    if not isinstance(identity, str) or not identity:
        raise ProtocolError(
            "P2",
            f"{what}: no step record with an identity beside it — a decision "
            "binds to the identity of the step that measured its inputs",
        )
    measured: dict[str, Any] = {}
    for key in step.rule:
        if key not in values:
            raise ProtocolError(
                "P2",
                f"{what}: no key {key!r} in {values_path.name} (has {sorted(values)})",
            )
        measured[key] = values[key]
    passed = all(
        holds(measured[key], clause, what=f"{what}: rule {key!r}")
        for key, clause in step.rule.items()
    )
    evidence = f"{hashlib.sha256(values_path.read_bytes()).hexdigest()}:{identity}"
    decision = {
        "decision_type": step.decision["on_pass" if passed else "on_fail"],
        "schema_version": DECISION_SCHEMA_VERSION,
        "measured_inputs": measured,
        "rule": {key: dict(clause) for key, clause in step.rule.items()},
        "outcome": "pass" if passed else "fail",
        "evidence_identity": evidence,
        "step": name,
    }
    (step_dir / DECISION_FILE).write_text(json.dumps(decision, indent=2) + "\n")
    return {
        "type": "decision",
        "status": "completed",
        "identity": loaded.step_digests[name],
        "implementation": dict(implementation),  # the code that ran it (§7)
        "values": step.values.target,
        "rule": decision["rule"],
        "measured": measured,
        "outcome": decision["outcome"],
        "decision_type": decision["decision_type"],
        "evidence_identity": evidence,
        "axes": [],  # a decision carries no sweep coordinates
        "files": [DECISION_FILE],
        "decision": DECISION_FILE,
    }


def read_decision(run_root: Path, producer: str, *, what: str) -> dict[str, Any]:
    """The producer's published ``decision.json`` at ``<run_root>/<producer>/``
    — a schema version 1 record, or a ``P2``."""
    path = run_root / producer / DECISION_FILE
    if not path.is_file():
        raise ProtocolError(
            "P2", f"{what}: {producer!r} published no {DECISION_FILE} at {str(path)!r}"
        )
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ProtocolError("P2", f"{what}: {path} is not valid JSON: {err}") from err
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != DECISION_SCHEMA_VERSION
    ):
        raise ProtocolError(
            "P2",
            f"{what}: {path} is not a schema_version {DECISION_SCHEMA_VERSION} "
            "decision record",
        )
    return payload


def evaluate_predicate(
    predicate: Mapping[str, Any], decision: Mapping[str, Any], *, what: str
) -> bool:
    """The predicate's verdict over one decision record — a ``P2`` when the
    record lacks the predicate's field or carries a value outside the field's
    closed vocabulary ([`FIELD_VOCABULARIES`][]): a verdict rests on a
    value the record states, never on an absence (``ne`` over a missing
    field would hold) or on a word no rule can have written."""
    field = str(predicate["field"])
    if field not in decision:
        raise ProtocolError(
            "P2",
            f"{what}: the decision record has no field {field!r} (has "
            f"{sorted(decision)}) — a verdict cannot rest on an absence",
        )
    value = decision[field]
    vocabulary = FIELD_VOCABULARIES.get(field, ())
    if value not in vocabulary:
        raise ProtocolError(
            "P2",
            f"{what}: the decision record's {field!r} is {value!r}, not one of "
            f"{list(vocabulary)} — no rule writes it, so no verdict reads it",
        )
    if "in" in predicate:
        return value in predicate["in"]
    if "eq" in predicate:
        return value == predicate["eq"]
    return value != predicate["ne"]


def run_conditional_step(
    name: str,
    step: ConditionalStep,
    loaded: LoadedWorkflow,
    run_root: Path,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the producer's ``decision.json``, evaluate the predicate and return
    the conditional's record (§8): the predicate, the verdict, the evidence it
    rested on and the steps it skipped — the side the verdict did not choose,
    and every child of a fanned-out step on that side (§2.9: a skipped parent
    skips its children with it). The conditional publishes no data file
    (``files: []``)."""
    producer = str(step.predicate["decision"]["step"])
    what = f"step {name!r}: predicate"
    decision = read_decision(run_root, producer, what=what)
    verdict = evaluate_predicate(step.predicate, decision, what=what)
    side = step.on_false if verdict else step.on_true
    skipped = {
        *side,
        *(child for gated in side for child in loaded.children.get(gated, ())),
    }
    return {
        "type": "conditional",
        "status": "completed",
        "identity": loaded.step_digests[name],
        "implementation": dict(implementation),  # the code that ran it (§7)
        "predicate": json.loads(json.dumps(dict(step.predicate))),
        "scope": step.scope,
        "verdict": verdict,
        "evidence": {
            "step": producer,
            "decision_type": decision.get("decision_type"),
            "outcome": decision.get("outcome"),
            "evidence_identity": decision.get("evidence_identity"),
        },
        "skipped": sorted(skipped),
        "axes": [],
        "files": [],
    }


def fold_skips(
    name: str,
    entry: Mapping[str, Any],
    loaded: LoadedWorkflow,
    skipped_by: dict[str, dict[str, Any]],
) -> None:
    """Fold a conditional's verdict (its record's ``skipped`` set) into the
    run's ``skipped_by`` map — the steps it names directly, then, in schedule
    order, every step that depends on a skipped step (``transitive_from``
    names the ones it followed; over ``after`` edges too, which are in
    ``loaded.dependencies``). A join declaring ``require: selected`` is not
    skipped by its own skipped children (§2.9, [`manifest.propagates`][]).
    Called for a conditional that ran and for one ``--resume`` reused, from
    its record either way."""
    from causalab.workflow.fan_out import selective_joins

    selective = selective_joins(loaded.document.steps)
    evidence = entry.get("evidence") or {}
    origin = {
        "conditional": name,
        "decision_step": evidence.get("step"),
        "decision_type": evidence.get("decision_type"),
        "outcome": evidence.get("outcome"),
        "evidence_identity": evidence.get("evidence_identity"),
    }
    for gated in entry.get("skipped") or ():
        skipped_by.setdefault(str(gated), {**origin, "transitive_from": []})
    for other in loaded.order:
        if other in skipped_by:
            continue
        via = skipped_via(
            other,
            loaded.dependencies.get(other, ()),
            lambda upstream: upstream in skipped_by,
            selective,
        )
        if via:
            first = skipped_by[via[0]]
            skipped_by[other] = {
                **{
                    k: first.get(k) for k in SKIPPED_BY_FIELDS if k != "transitive_from"
                },
                "transitive_from": via,
            }


# --------------------------------------------------------------------------- #
# run time: the receipt, before allocation
# --------------------------------------------------------------------------- #


def check_receipt(name: str, step: Step, run_root: Path) -> None:
    """A step's ``requires_receipt``, checked **before it is scheduled** (§2.8,
    §8): before an attempt directory, before ``route_engine``, before any
    device. Two distinct refusals, both rule 18, neither a skip: the receipt
    is **missing** (the producer's ``decision.json`` is not in the run tree —
    it was skipped, or never published), or it **failed** (its ``outcome`` is
    not the one required). A refusal is a failed attempt: the step is
    ``failed`` and its dependents ``blocked``."""
    receipt = step.requires_receipt
    if receipt is None:
        return
    producer = receipt["step"]
    required = receipt["outcome"]
    where = f"steps.{name}.requires_receipt"
    path = run_root / producer / DECISION_FILE
    if not path.is_file():
        raise _refuse(
            f"'requires_receipt' names {producer!r}, whose receipt does not exist "
            f"({path}) — the receipt step was skipped or never published; a "
            f"missing receipt is an unmet precondition, not a skip: {name!r} is "
            "not allocated",
            where,
        )
    try:
        decision = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        decision = {}
    if not isinstance(decision, dict):
        decision = {}
    outcome = decision.get("outcome")
    if outcome != required:
        raise _refuse(
            f"receipt {producer!r} carries outcome {outcome!r} (decision "
            f"{decision.get('decision_type')!r}, evidence "
            f"{decision.get('evidence_identity')}); {name!r} requires "
            f"{required!r} — not allocated",
            where,
        )


# --------------------------------------------------------------------------- #
# --resume: the evidence clause
# --------------------------------------------------------------------------- #


def evidence_holds(step: Step, step_dir: Path, record: Mapping[str, Any]) -> bool:
    """Whether a published decision or conditional record still rests on the
    evidence that is in the run tree now. ``evidence_identity`` is a run-time
    value — the sha256 of a table that exists only after a step ran — so it
    cannot enter a load-time digest; this is where ``--resume`` binds it: a
    conditional whose producer's current ``decision.json`` carries another
    evidence identity than the one its record read is re-evaluated, and a
    decision whose values file or producer identity moved is re-made. And a
    step of **any** kind with a ``requires_receipt`` holds only while the
    producer's current ``decision.json`` still carries the required outcome:
    a receipt that flipped since the step ran sends it back through
    [`check_receipt`][], which refuses before any allocation — a reused
    entry never stands on a failed receipt. Any other kind holds trivially."""
    run_root = step_dir.parent
    receipt = step.requires_receipt
    if receipt is not None:
        current = _read_json(run_root / str(receipt["step"]) / DECISION_FILE)
        if current is None or current.get("outcome") != receipt["outcome"]:
            return False
    if isinstance(step, ConditionalStep):
        recorded = (record.get("evidence") or {}).get("evidence_identity")
        path = run_root / str(step.predicate["decision"]["step"]) / DECISION_FILE
        current = _read_json(path)
        return (
            isinstance(recorded, str)
            and current is not None
            and current.get("evidence_identity") == recorded
        )
    if isinstance(step, DecisionStep):
        values_path = run_root / str(step.values.step) / str(step.values.file)
        own = _read_json(step_dir / DECISION_FILE)
        if not values_path.is_file() or own is None:
            return False
        identity = read_sidecar(values_path).get("identity")
        current = f"{hashlib.sha256(values_path.read_bytes()).hexdigest()}:{identity}"
        return own.get("evidence_identity") == current
    return True


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def skipped_entry(step_type: str, by: Mapping[str, Any]) -> dict[str, Any]:
    """A skipped step's manifest entry (§8): its type, the word, and the
    decision that skipped it — no files, no digests, no directory."""
    return {
        "type": step_type,
        "status": "skipped",
        "skipped_by": {field: by.get(field) for field in SKIPPED_BY_FIELDS},
    }
