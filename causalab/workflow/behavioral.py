"""Define behavioral evaluation steps and their saved decisions.

The step binds a task's scoring specification to a declared dataset split.
Decoding produces continuations, which are graded into the supported outcome
categories. The step writes counts and a typed decision record using the
authored thresholds.

The engine performs decoding. This module defines parsing, validation, and
outcome handling for the workflow runner."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from causalab.causal.pair_validation import GRADES
from causalab.causal.scoring import ScoringSpec, table_scoring
from causalab.protocol.pipeline import check_parallel, compile_protocol, route_engine
from causalab.protocol.engine import (
    CONTINUATIONS_FILE,
    Engine,
    RunContext,
    check_steps_signed,
)
from causalab.protocol.publish import SOLO, Publisher
from causalab.protocol.rules.errors import ProtocolError, ProtocolWarning, suggest
from causalab.protocol.results import example_labels
from causalab.io.env import ResolutionEnv, split_dataset_ref
from causalab.protocol.receipt import MODELS_KEY, execution_record, model_records
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.schema import Document, PositionSpec
from causalab.protocol.lowering import DEFAULT_POINT_CAP, point_count
from causalab.io.tables import read_table, write_table
from causalab.tasks.loader import load_task
from causalab.workflow.document import BehavioralStep, LoadedWorkflow, WorkflowError

__all__ = [
    "BEHAVIORAL_FILES",
    "BEHAVIORAL_RULE",
    "DECISION_FILE",
    "DECISION_SCHEMA_VERSION",
    "DECISION_TYPES",
    "DECODING_MODES",
    "DEFAULT_COHORTS",
    "DEFAULT_RETAIN_ROWS",
    "OUTCOMES",
    "OUTCOMES_FILE",
    "SAMPLING_ENGINES",
    "SPLIT_PURPOSES",
    "check_behavioral",
    "check_checker_binding",
    "derive_outcome",
    "parse_behavioral",
    "run_behavioral_step",
    "write_decision",
]

#: The workflow checklist rule every refusal here is filed under (§5 item 17).
#: Rules 15 and 16 are taken by the qualify-once and site-equivalence layers.
BEHAVIORAL_RULE = 17

#: The four terminal outcomes of one generated row (§2.7), closed; a row is in
#: exactly one. Order is the derivation's precedence.
OUTCOMES: tuple[str, ...] = ("truncated", "no_final_answer", "invalid_format", "valid")

#: How a ``generated`` frame is decoded (§2.7).
DECODING_MODES: tuple[str, ...] = ("deterministic", "sampled")

#: What a qualification's split is *for* (§2.7) — three values, not a free
#: string; the document's base ref fragment must spell the same one.
SPLIT_PURPOSES: tuple[str, ...] = ("development", "reserve", "confirmation")

#: What a decision does to the question (§2.7): the ``decision_type`` a
#: ``decision.json`` record carries.
DECISION_TYPES: tuple[str, ...] = ("advance", "revise", "narrow")

#: The engines whose decode can sample. The nnsight engine decodes greedily
#: (``model.generate(do_sample=False)``), so a ``sampled`` step routed there
#: is refused before its model loads — by name, here; the capability table is
#: unchanged (workflow spec §2.7).
SAMPLING_ENGINES: tuple[str, ...] = ("pytorch_hooks",)

#: The three files a behavioral step publishes beside its document's ``save``
#: files: the engine's raw generations, the per-row outcomes, the decision.
OUTCOMES_FILE = "outcomes.json"
DECISION_FILE = "decision.json"
BEHAVIORAL_FILES: tuple[str, ...] = (CONTINUATIONS_FILE, OUTCOMES_FILE, DECISION_FILE)

#: ``decision.json``'s schema version.
DECISION_SCHEMA_VERSION = 1

#: The cohort defaults: examples a qualification is expected to run over —
#: single-input documents and paired (counterfactual-role) documents. A
#: smaller split is **recorded and warned**, never refused (§2.7).
DEFAULT_COHORTS: Mapping[str, int] = {"single_input": 10_000, "pairs": 1_000}

#: How many raw generations a step keeps when ``retain`` is unauthored: the
#: bound is explicit and small because preserved generations are unbounded
#: output otherwise.
DEFAULT_RETAIN_ROWS = 1000

_DECODING_KEYS = ("mode", "seed", "temperature", "top_p", "eos_token_ids")
_CHECKER_KEYS = ("task", "task_cfg")
_THRESHOLD_KEYS = ("min_examples", "min_valid_rate", "min_correct_rate")
_DECISION_KEYS = ("on_pass", "on_fail")
_LIST_FIELD = re.compile(r"^(.*)\[(\d+)\]$")


def _refuse(message: str, path: str) -> WorkflowError:
    return WorkflowError(BEHAVIORAL_RULE, message, path=path)


def _unknown_keys(obj: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    for key in obj:
        if key not in allowed:
            raise _refuse(f"unknown key {key!r}{suggest(str(key), allowed)}", path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# parsing (§2.7, rule 17)
# --------------------------------------------------------------------------- #


def parse_behavioral(
    raw: Mapping[str, Any],
    path: str,
    *,
    document: str,
    overrides: Mapping[str, Any],
    max_points: int | None,
    after: tuple[str, ...],
    description: str | None,
) -> BehavioralStep:
    """The behavioral half of a step's parse: ``decoding``, ``checker``,
    ``split``, ``thresholds``, ``retain`` and ``decision`` (§2.7), each refused
    under rule 17 naming the field. The common fields arrive already parsed
    from ``document._parse_step``, which owns the key set and ``after``."""
    for key, what in _REQUIRED.items():
        if key not in raw:
            raise _refuse(
                f"a behavioral step declares {key!r} — {what}", f"{path}.{key}"
            )
    return BehavioralStep(
        type="behavioral",
        document=document,
        set=dict(overrides),
        max_points=max_points,
        decoding=_parse_decoding(raw["decoding"], f"{path}.decoding"),
        checker=_parse_checker(raw["checker"], f"{path}.checker"),
        split=_parse_split(raw["split"], f"{path}.split"),
        thresholds=_parse_thresholds(raw["thresholds"], f"{path}.thresholds"),
        retain=_parse_retain(raw["retain"], f"{path}.retain")
        if "retain" in raw
        else None,
        decision=_parse_decision(raw["decision"], f"{path}.decision"),
        after=after,
        description=description,
    )


#: The blocks a behavioral step must author (rule 17), and why each exists.
_REQUIRED: Mapping[str, str] = {
    "decoding": "how the continuation is decoded, deterministic or sampled with a seed",
    "checker": "the task whose ScoringSpec defines 'correct' for the generations",
    "split": "which split purpose the step qualifies on (development | reserve | confirmation)",
    "thresholds": "the declared qualification thresholds the decision is held to",
    "decision": "what a pass and a fail do to the question (advance | revise | narrow)",
}


def _parse_decoding(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _refuse(
            "'decoding' is an object {mode, seed?, temperature?, top_p?}", path
        )
    _unknown_keys(raw, _DECODING_KEYS, path)
    if "mode" not in raw:
        raise _refuse(f"'decoding' names a mode from {list(DECODING_MODES)}", path)
    mode = raw["mode"]
    if mode not in DECODING_MODES:
        raise _refuse(
            f"decoding mode {mode!r} is not one of {list(DECODING_MODES)}"
            f"{suggest(str(mode), DECODING_MODES)}",
            f"{path}.mode",
        )
    stopping = {}
    if "eos_token_ids" in raw:
        ids = raw["eos_token_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not _is_int(t) or t < 0 for t in ids)
            or len(set(ids)) != len(ids)
        ):
            raise _refuse("eos_token_ids must be distinct nonnegative integers", path)
        stopping["eos_token_ids"] = list(ids)
    if mode == "deterministic":
        extra = sorted(key for key in raw if key not in {"mode", "eos_token_ids"})
        if extra:
            raise _refuse(
                f"a deterministic decode is the argmax and takes no {extra} — "
                "author mode 'sampled' to draw",
                path,
            )
        return {"mode": "deterministic", **stopping}
    if "seed" not in raw:
        raise _refuse(
            "a sampled decode declares its 'seed' — without one two runs cannot "
            "be told apart from a bug",
            f"{path}.seed",
        )
    seed = raw["seed"]
    if not _is_int(seed) or seed < 0:
        raise _refuse("'seed' is a non-negative integer", f"{path}.seed")
    temperature = raw.get("temperature", 1.0)
    if not _is_number(temperature) or temperature <= 0:
        raise _refuse("'temperature' is a number > 0", f"{path}.temperature")
    top_p = raw.get("top_p", 1.0)
    if not _is_number(top_p) or not 0 < top_p <= 1:
        raise _refuse("'top_p' is a number in (0, 1]", f"{path}.top_p")
    return {
        "mode": "sampled",
        "seed": int(seed),
        "temperature": float(temperature),
        "top_p": float(top_p),
        **stopping,
    }


def _parse_checker(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _refuse(
            "'checker' names the task whose ScoringSpec grades the generations: "
            "{task, task_cfg?}",
            path,
        )
    _unknown_keys(raw, _CHECKER_KEYS, path)
    task = raw.get("task")
    if not isinstance(task, str) or not task:
        raise _refuse(
            "'task' names a task package (causalab.tasks.<task>)", f"{path}.task"
        )
    out: dict[str, Any] = {"task": task}
    if "task_cfg" in raw:
        cfg = raw["task_cfg"]
        if not isinstance(cfg, Mapping) or not all(isinstance(k, str) for k in cfg):
            raise _refuse(
                "'task_cfg' is an object of a factory task's config fields "
                "(the shipped manifest's task_cfg)",
                f"{path}.task_cfg",
            )
        out["task_cfg"] = json.loads(json.dumps(dict(cfg)))
    return out


def _parse_split(raw: Any, path: str) -> str:
    if raw not in SPLIT_PURPOSES:
        raise _refuse(
            f"split {raw!r} is not one of {list(SPLIT_PURPOSES)}"
            f"{suggest(str(raw), SPLIT_PURPOSES)}",
            path,
        )
    return str(raw)


def _parse_thresholds(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _refuse(
            "'thresholds' is an object {min_examples, min_valid_rate, min_correct_rate}",
            path,
        )
    _unknown_keys(raw, _THRESHOLD_KEYS, path)
    missing = [key for key in _THRESHOLD_KEYS if key not in raw]
    if missing:
        raise _refuse(f"'thresholds' is missing {missing}", path)
    n = raw["min_examples"]
    if not _is_int(n) or n < 1:
        raise _refuse("'min_examples' is a positive integer", f"{path}.min_examples")
    out: dict[str, Any] = {"min_examples": int(n)}
    for key in ("min_valid_rate", "min_correct_rate"):
        value = raw[key]
        if not _is_number(value) or not 0 <= value <= 1:
            raise _refuse(
                f"{key!r} is a number in [0, 1] — a rate is a count over n, "
                "never a statistic computed here",
                f"{path}.{key}",
            )
        out[key] = float(value)
    return out


def _parse_retain(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {"generations"}:
        raise _refuse(
            "'retain' is {generations: \"all\" | {max_rows: n}} — preserved "
            "generations are bounded by default; 'all' says otherwise explicitly",
            path,
        )
    generations = raw["generations"]
    if generations == "all":
        return {"generations": "all"}
    if (
        isinstance(generations, Mapping)
        and set(generations) == {"max_rows"}
        and _is_int(generations["max_rows"])
        and generations["max_rows"] >= 1
    ):
        return {"generations": {"max_rows": int(generations["max_rows"])}}
    raise _refuse(
        f"'generations' is \"all\" or {{max_rows: positive integer}}, not "
        f"{generations!r} — an unbounded retention must be spelled 'all'",
        f"{path}.generations",
    )


def _parse_decision(raw: Any, path: str) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise _refuse("'decision' is an object {on_pass, on_fail}", path)
    _unknown_keys(raw, _DECISION_KEYS, path)
    out: dict[str, str] = {}
    for key in _DECISION_KEYS:
        if key not in raw:
            raise _refuse(f"'decision' is missing {key!r}", path)
        value = raw[key]
        if value not in DECISION_TYPES:
            raise _refuse(
                f"decision {key} {value!r} is not one of {list(DECISION_TYPES)}"
                f"{suggest(str(value), DECISION_TYPES)}",
                f"{path}.{key}",
            )
        out[key] = str(value)
    return out


# --------------------------------------------------------------------------- #
# the load-time binding (rule 17)
# --------------------------------------------------------------------------- #


def _config_class(task: str) -> type | None:
    """The config dataclass of a factory task, by the convention
    [`causalab.tasks.serialize.config_class`][] reads (``causalab/tasks/
    <name>/config.py`` holds one module-level dataclass named ``*Config``).
    Spelled here because ``serialize.py`` imports the numerics stack; this
    stays stdlib."""
    try:
        module = importlib.import_module(f"causalab.tasks.{task}.config")
    except ModuleNotFoundError:
        return None
    found = [
        value
        for name, value in vars(module).items()
        if name.endswith("Config")
        and dataclasses.is_dataclass(value)
        and isinstance(value, type)
        and value.__module__ == module.__name__
    ]
    return found[0] if len(found) == 1 else None


def _scoring_spec(checker: Mapping[str, Any], path: str) -> ScoringSpec:
    """The [`ScoringSpec`][] the checker names, loaded through
    [`load_task`][] — the one string-match authority."""
    task = str(checker["task"])
    cfg: Any = None
    if "task_cfg" in checker:
        cls = _config_class(task)
        if cls is None:
            raise _refuse(
                f"task {task!r} takes no task_cfg (no causalab/tasks/{task}/config.py "
                "dataclass) — drop it",
                f"{path}.task_cfg",
            )
        try:
            cfg = cls(**dict(checker["task_cfg"]))
        except TypeError as err:
            raise _refuse(
                f"task_cfg does not build {cls.__name__}: {err}", f"{path}.task_cfg"
            ) from err
    try:
        loaded = load_task(task, task_cfg=cfg)
    except (ImportError, ValueError, TypeError, AttributeError) as err:
        raise _refuse(f"task {task!r} does not load: {err}", f"{path}.task") from err
    spec = getattr(loaded.causal_model, "scoring", None)
    if not isinstance(spec, ScoringSpec):
        raise _refuse(
            f"task {task!r} declares no ScoringSpec — nothing defines 'correct' "
            "for its generations",
            f"{path}.task",
        )
    return spec


def check_checker_binding(
    checker: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], path: str
) -> ScoringSpec:
    """The checker's task loads and declares a ``ScoringSpec``, and when the
    split's rows record a ``string_mode`` it is the spec's (an unrecorded
    table compares nothing). Returns the bound spec. Refused under rule 17 —
    at load, before any model exists (delete the binding and the workflow
    does not load)."""
    spec = _scoring_spec(checker, path)
    table_mode = table_scoring(rows)
    if table_mode is not None and table_mode != spec.string_mode:
        raise _refuse(
            f"the split's rows record string_mode {table_mode!r}, but task "
            f"{checker['task']!r}'s ScoringSpec grades by {spec.string_mode!r} — "
            "the table was built under another definition of correct",
            f"{path}.task",
        )
    return spec


def _base_role(document: Document, path: str) -> tuple[str, str]:
    """``(dataset ref, field)`` of the document's base role, both literal."""
    base = document.data.get("base")
    if isinstance(base, tuple):
        base = base[0] if base else None
    dataset = getattr(base, "dataset", None)
    field = getattr(base, "field", None)
    if not isinstance(dataset, str) or not isinstance(field, str):
        raise _refuse(
            "a behavioral document's base role names a literal dataset ref and "
            "field — the split is checked against the ref's fragment",
            path,
        )
    return dataset, field


def _decodes(document: Document) -> bool:
    return any(
        isinstance(spec, PositionSpec) and spec.generated is not None
        for spec in document.positions.values()
    )


def check_behavioral(
    name: str, step: BehavioralStep, compiled: CompiledProtocol, env: ResolutionEnv
) -> ScoringSpec:
    """Rule 17's load-time half over a compiled inner document: the document
    decodes (a ``generated`` position), its base ref's fragment is the step's
    ``split``, and the checker binds ([`check_checker_binding`][])."""
    path = f"steps.{name}"
    if not _decodes(compiled.document):
        raise _refuse(
            f"document {step.document!r} declares no generated position — a "
            "behavioral step runs a no-intervention document under the "
            "'generated' frame (probe_variable.json's shape)",
            path,
        )
    ref, _field = _base_role(compiled.document, path)
    _base, fragment = split_dataset_ref(ref)
    if fragment != step.split:
        selects = (
            f"selects split {fragment!r}"
            if fragment is not None
            else "selects no split"
        )
        raise _refuse(
            f"split is {step.split!r}, but the document's base ref {ref!r} {selects} "
            f"— the ref names the split it qualifies on ({_base}#{step.split})",
            f"{path}.split",
        )
    return check_checker_binding(
        step.checker, env.datasets.rows(ref), f"{path}.checker"
    )


# --------------------------------------------------------------------------- #
# outcomes (§2.7's table)
# --------------------------------------------------------------------------- #


def derive_outcome(
    *, width: int, steps: int, anchor_found: bool, declared: bool
) -> str:
    """One row's terminal outcome ([`OUTCOMES`][], §2.7), in precedence
    order: ``truncated`` when the row ran the whole budget without EOS
    (``width == steps``); ``invalid_format`` for an empty continuation;
    ``no_final_answer`` when the answer variable's value is nowhere in the
    text (the ``variable`` anchor resolves to no steps); ``invalid_format``
    when it is there but the graded string is not a declared form under the
    spec's string mode; else ``valid``."""
    if width >= steps:
        return "truncated"
    if width == 0:
        return "invalid_format"
    if not anchor_found:
        return "no_final_answer"
    if not declared:
        return "invalid_format"
    return "valid"


def _grade_word(grade: float | None) -> str:
    """[`GRADES`][] for a spec's ``grade`` value."""
    correct, incorrect, unscored = GRADES
    if grade is None:
        return unscored
    return correct if grade == 1.0 else incorrect


def _variable_value(row: Mapping[str, Any], field: str, variable: str) -> str:
    """The row's value for a prompt variable, by the protocol's rule
    (``causalab.protocol.positions.encoding.variable_value``: the
    ``<col>_variables`` sibling first, the plain column as fallback) — spelled
    here from before that rule moved protocol-side; it could be one import
    now."""
    match = _LIST_FIELD.match(field)
    column = match.group(1) if match else field
    sibling: Any = row.get(f"{column}_variables")
    if match and isinstance(sibling, list):
        index = int(match.group(2))
        sibling = sibling[index] if index < len(sibling) else None
    if isinstance(sibling, Mapping) and variable in sibling:
        return str(sibling[variable])
    if variable in row:
        return str(row[variable])
    raise ProtocolError(
        "P2",
        f"no value for the answer variable {variable!r}: neither {column}_variables "
        f"nor a {variable!r} column exists in the dataset row",
    )


def _declared_form(spec: ScoringSpec, text: str) -> bool:
    """Whether ``text`` is one of the spec's declared forms of the answer
    variable under its string mode — asked of the spec's own grader against
    every declared value, so no second matching rule exists here."""
    variable = spec.answer_variable
    assert variable is not None  # ScoringSpec resolves it at construction
    return any(spec.grade(text, value) == 1.0 for value in spec.forms[variable])


def _behavioral_record(line: Mapping[str, Any], unwritten: Collection[str]) -> bool:
    """Whether a continuation record is one a behavioral step qualifies on:
    the **un-intervened** model's continuation of the ``base`` prompt. A
    written model's continuation is an intervention's outcome, not the
    behaviour being certified."""
    return line.get("model") in unwritten and line.get("input") == "base"


def unwritten_models(doc: Document) -> frozenset[str]:
    """The models of ``doc`` that land no write (§2.9)."""
    return frozenset(
        str(ref.model) for ref in doc.read_refs() if doc.is_unwritten(str(ref.model))
    )


def validate_continuation_coverage(
    continuations: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    points: int,
    *,
    unwritten: Collection[str],
) -> None:
    """Refuse missing, duplicated, or foreign records before qualification.

    ``labels`` are the split's row labels (``protocol/examples.py``): every
    (point, label) must appear exactly once among the records of an
    ``unwritten`` model on the base role."""
    expected = {(point, label) for point in range(points) for label in labels}
    seen: set[tuple[int, str]] = set()
    for line in continuations:
        if not _behavioral_record(line, unwritten):
            continue
        point, label = line.get("point"), line.get("example_id")
        if type(point) is not int or type(label) is not str:
            raise ProtocolError(
                "P2", "continuation point must be an integer and example_id a string"
            )
        key = (point, label)
        if key not in expected or key in seen:
            raise ProtocolError("P2", f"duplicate or foreign continuation {key}")
        seen.add(key)
    if seen != expected:
        raise ProtocolError(
            "P2", f"missing {len(expected - seen)} planned continuations"
        )


def _outcome_rows(
    continuations: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    field: str,
    spec: ScoringSpec,
    *,
    unwritten: Collection[str],
) -> list[dict[str, Any]]:
    variable = spec.answer_variable
    assert variable is not None
    by_label = dict(zip(example_labels(rows), rows))
    out: list[dict[str, Any]] = []
    for line in continuations:
        if not _behavioral_record(line, unwritten):
            continue
        label = str(line["example_id"])
        if label not in by_label:
            raise ProtocolError(
                "P2",
                f"{CONTINUATIONS_FILE} names example {label!r}, which the "
                f"{len(rows)}-row split has no row for — the engine's rows and the "
                "table disagree",
            )
        expected = _variable_value(by_label[label], field, variable)
        text = str(line["text"])
        width, steps = int(line["width"]), int(line["steps"])
        outcome = derive_outcome(
            width=width,
            steps=steps,
            anchor_found=expected in text,
            declared=_declared_form(spec, text),
        )
        grade = _grade_word(spec.grade(text, expected)) if outcome == "valid" else None
        out.append(
            {
                "example_id": label,
                "point": int(line["point"]),
                "outcome": outcome,
                "grade": grade,
                "expected": expected,
                "width": width,
                "steps": steps,
            }
        )
    return out


def _counts(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"n": len(outcomes), **{outcome: 0 for outcome in OUTCOMES}, "correct": 0}
    for row in outcomes:
        counts[str(row["outcome"])] += 1
        if row["grade"] == GRADES[0]:
            counts["correct"] += 1
    return counts


# --------------------------------------------------------------------------- #
# the decision record (``decision.json``, produced here)
# --------------------------------------------------------------------------- #


def write_decision(
    step_dir: Path,
    *,
    step: str,
    split: str,
    identity: str,
    thresholds: Mapping[str, Any],
    decision: Mapping[str, str],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    """``decision.json`` beside ``_step.json``: the record's six fields —
    ``decision_type``, ``schema_version``, ``measured_inputs``, ``rule``,
    ``outcome``, ``evidence_identity`` — plus ``split`` and ``step``. The rule
    is the authored thresholds block verbatim; the measured inputs are counts
    and the two rates ``counts / n``; ``pass`` iff every threshold holds. The
    evidence identity is the sha256 of the outcomes table's bytes joined to the
    step identity, so a decision is tied to exactly the rows it was made on."""
    n = int(counts["n"])
    valid_rate = counts["valid"] / n if n else 0.0
    correct_rate = counts["correct"] / n if n else 0.0
    passed = (
        n >= int(thresholds["min_examples"])
        and valid_rate >= float(thresholds["min_valid_rate"])
        and correct_rate >= float(thresholds["min_correct_rate"])
    )
    outcomes_sha = hashlib.sha256((step_dir / OUTCOMES_FILE).read_bytes()).hexdigest()
    record = {
        "decision_type": decision["on_pass" if passed else "on_fail"],
        "schema_version": DECISION_SCHEMA_VERSION,
        "measured_inputs": {
            "n": n,
            "counts": {outcome: int(counts[outcome]) for outcome in OUTCOMES}
            | {"correct": int(counts["correct"])},
            "valid_rate": valid_rate,
            "correct_rate": correct_rate,
        },
        "rule": dict(thresholds),
        "outcome": "pass" if passed else "fail",
        "evidence_identity": f"{outcomes_sha}:{identity}",
        "split": split,
        "step": step,
    }
    (step_dir / DECISION_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def _retain(
    step_dir: Path,
    continuations: Sequence[Mapping[str, Any]],
    retain: Mapping[str, Any],
) -> int:
    """Bound the published raw generations to the authored retention:
    ``"all"`` keeps the engine's file as written; ``{max_rows}`` rewrites it to
    its first ``max_rows`` rows. Returns how many rows were kept."""
    generations = retain["generations"]
    if generations == "all":
        return len(continuations)
    keep = min(int(generations["max_rows"]), len(continuations))
    if keep < len(continuations):
        write_table(step_dir / CONTINUATIONS_FILE, list(continuations[:keep]))
    return keep


@dataclasses.dataclass(frozen=True)
class _Prepared:
    """A behavioral step ready to execute: the compiled document, the engine
    routed for it, the run context, the point indices it covers, and the
    split's rows, answer field and scoring spec the outcomes are derived
    with."""

    inner: CompiledProtocol
    engine: Engine
    run: RunContext
    indices: Sequence[int]
    rows: Sequence[Mapping[str, Any]]
    field: str
    spec: ScoringSpec


def _prepare(
    name: str,
    step: BehavioralStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engine: Engine | None,
    publisher: Publisher,
    *,
    selection: Sequence[int] | None,
) -> _Prepared:
    """Compile, route and build the run context — the half of a behavioral
    step every rank of a launched world runs identically
    (``docs/model_parallelism.md`` §3): the same document, the same ``set``,
    the same points, so the engines' collectives meet. Refusals here are the
    same on every rank."""
    doc_path = (loaded.workflow_dir / step.document).resolve()
    inner = compile_protocol(
        doc_path,
        env=run_env,
        overrides=step.set,
        point_cap=step.max_points if step.max_points is not None else DEFAULT_POINT_CAP,
    )
    chosen = route_engine(inner, engine)
    # the engine geometry's document rules, before any weights and on every
    # rank alike, as `run_protocol` decides them (docs/model_parallelism.md
    # §8.3, §8.4); a behavioral step always decodes
    check_parallel(inner, chosen, env=run_env, decoding=True)
    decoding = dict(step.decoding)
    if decoding["mode"] == "sampled" and chosen.name not in SAMPLING_ENGINES:
        # before any weights load (§5's invariant): the nnsight engine's
        # decode is greedy-only, and a sampled step is not silently greedy
        raise _refuse(
            f"engine {chosen.name!r} decodes deterministically only — a sampled "
            f"decode runs on {list(SAMPLING_ENGINES)}; author decoding.mode "
            "'deterministic' or route to a sampling engine (--engine pytorch_hooks)",
            f"steps.{name}.decoding.mode",
        )
    ref, field = _base_role(inner.document, f"steps.{name}")
    rows = run_env.datasets.rows(ref)
    spec = check_checker_binding(step.checker, rows, f"steps.{name}.checker")
    indices = range(point_count(inner.axes)) if selection is None else tuple(selection)
    # the step's record is its receipt and the runner's stream is its stream,
    # so the engine records nothing beside the outputs (`record=False`)
    run = RunContext(
        output_dir=step_dir,
        env=run_env,
        points=tuple(indices),
        decoding=decoding,
        record=False,
        # this process's place in a launched world (docs/model_parallelism.md
        # §3): the engine publishes through it, so one rank writes the
        # continuations
        publisher=publisher,
    )
    return _Prepared(inner, chosen, run, indices, rows, field, spec)


def follow_behavioral_step(
    name: str,
    step: BehavioralStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engine: Engine | None,
    publisher: Publisher,
    *,
    selection: Sequence[int] | None = None,
) -> None:
    """A rank that does not publish (``docs/model_parallelism.md`` §3, §11):
    the engine's half of the step alone — the same run the joiner's engine
    executes, whose collectives this rank's engine meets — its outputs
    discarded by the engine through ``publisher``; the record is the
    joiner's, agreed through the runner's lockstep."""
    prepared = _prepare(
        name, step, loaded, run_env, step_dir, engine, publisher, selection=selection
    )
    prepared.engine.execute(prepared.inner, prepared.run)


def run_behavioral_step(
    name: str,
    step: BehavioralStep,
    loaded: LoadedWorkflow,
    run_env: ResolutionEnv,
    step_dir: Path,
    engine: Engine | None,
    implementation: Mapping[str, Any],
    *,
    selection: Sequence[int] | None = None,
    publisher: Publisher = SOLO,
) -> dict[str, Any]:
    """Run one behavioral step into ``step_dir`` (an attempt directory) and
    return its record (§8): the protocol record's fields, plus ``execution.
    decoding`` beside ``batch_rows``, ``checker`` (with the spec's
    ``string_mode``), ``split``, ``outcomes`` counts, ``thresholds``, the
    ``retain`` accounting, the ``cohort`` note and the ``decision`` path.
    ``selection`` — a fanned-out child's point indices (§2.9) — is the run
    context's ``points``, exactly as the runner passes a protocol step's;
    ``None`` runs every point. ``publisher`` is this process's place in a
    launched world (``docs/model_parallelism.md`` §3) — the joiner's, here:
    the rank that reads the continuations back and writes the record.

    The document is compiled through the one compiler with the step's ``set``
    and routed exactly as a protocol step is; the engine decodes under the
    step's ``decoding`` and writes [`CONTINUATIONS_FILE`][]; the outcomes
    and the decision are derived from that file and the split's rows.
    """
    prepared = _prepare(
        name, step, loaded, run_env, step_dir, engine, publisher, selection=selection
    )
    inner, engine, run = prepared.inner, prepared.engine, prepared.run
    indices, rows, field, spec = (
        prepared.indices,
        prepared.rows,
        prepared.field,
        prepared.spec,
    )
    decoding = dict(run.decoding or {})
    result = engine.execute(inner, run)
    check_steps_signed(result, indices, engine)  # one signed step per index
    if CONTINUATIONS_FILE not in result.files:
        raise ProtocolError(
            "P4",
            f"engine {engine.name!r} wrote no {CONTINUATIONS_FILE} for a decoding "
            "request — a behavioral step needs an engine that publishes its "
            "continuations",
        )
    continuations = read_table(step_dir / CONTINUATIONS_FILE)
    unwritten = unwritten_models(inner.document)
    validate_continuation_coverage(
        continuations, example_labels(rows), len(indices), unwritten=unwritten
    )
    outcomes = _outcome_rows(continuations, rows, field, spec, unwritten=unwritten)
    write_table(step_dir / OUTCOMES_FILE, outcomes)
    counts = _counts(outcomes)
    identity = loaded.step_digests[name]
    write_decision(
        step_dir,
        step=name,
        split=step.split,
        identity=identity,
        thresholds=step.thresholds,
        decision=step.decision,
        counts=counts,
    )
    retain = (
        dict(step.retain)
        if step.retain is not None
        else {"generations": {"max_rows": DEFAULT_RETAIN_ROWS}}
    )
    kept = _retain(step_dir, continuations, retain)
    paired = any(role != "base" for role in inner.document.data)
    cohort = DEFAULT_COHORTS["pairs" if paired else "single_input"]
    n = counts["n"]
    if n < cohort:
        warnings.warn(
            f"step {name!r} qualifies on {n} examples of split {step.split!r}; "
            f"the cohort default is {cohort} — recorded in the step record, "
            "not refused",
            ProtocolWarning,
            stacklevel=2,
        )
    models = model_records(result.summaries)
    return {
        "type": "behavioral",
        "status": "completed",
        "identity": identity,
        "implementation": dict(implementation),  # the code that ran it (§7)
        "document": step.document,
        "engine": engine.name,
        "document_digest": inner.digests.document,  # fully resolved (§7)
        "points": len(indices),
        # the provenance units (§7), as the engine signed them
        "point_digests": [step.digest for step in result.steps],
        # each point's coordinates, aligned with `point_digests` — what a
        # fan-out's join places a metric row by (§2.9)
        "coords": [dict(step.coords) for step in result.steps],
        "axes": [axis.id for axis in inner.axes],
        "files": sorted({*result.files, OUTCOMES_FILE, DECISION_FILE}),
        # the one execution block (IM spec §8): the engine's geometry and model
        # source through the one recorder, and the decode spec — the first
        # place a seed is recorded — beside them, as the recorder's contract
        # admits a later execution parameter
        "execution": {
            **execution_record(engine, run, launcher=publisher.launcher),
            "decoding": decoding,
        },
        # the commit each model's revision resolved to (IM spec §8), as a
        # protocol step's record carries it
        **({MODELS_KEY: models} if models else {}),
        "checker": {**step.checker, "string_mode": spec.string_mode},
        "split": step.split,
        "outcomes": counts,
        "thresholds": dict(step.thresholds),
        "retain": {**retain, "retained": kept, "n": len(continuations)},
        "cohort": {"default_min_examples": cohort, "n": n, "below_default": n < cohort},
        "decision": DECISION_FILE,
    }
