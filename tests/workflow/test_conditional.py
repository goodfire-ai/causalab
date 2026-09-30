"""Typed decisions and conditional steps (``docs/workflow_protocol.md`` §2.8,
§5 rule 18; T10–T14 and the censuses) — on CPU, with no engine.

* **The censuses.** Five step kinds; `MAX_RULE` is 18 and §5 numbers it; the
  closed vocabularies — comparators, predicate comparators, scopes, receipt
  outcomes, decision fields, dispositions — are §2.8's and §8's tables member
  for member; the new prose and this layer's module never use the word
  that also names a kind of Git ref (``AMBIGUOUS``).
* **The closure guard.** ``causalab/workflow/conditional.py`` is a member of
  no hashed script's closure and of ``SHARED``, reaches no engine module, and
  loading the fixture chain imports no torch — digest-neutral because nothing
  digest-bearing changed (both pins hold, T13).
* **Refusals, each beside its valid twin**, every one
  naming the field: the decision's `values` and `rule`, the conditional's
  `predicate`, `on_true` / `on_false` and `scope`, and `requires_receipt`.
* **T10.** One workflow, two runs differing only in the measured input: one
  publishes `fit` and skips `probe`, the other the reverse, transitively; the
  skipped entry names the decision by `evidence_identity` equal to the
  producer's `decision.json` — the clause that stops a predicate reading a
  metric table (the mutation) from passing.
* **T11.** A protocol step with `requires_receipt`: a `fail` receipt and a
  missing one are two distinct `W18` refusals before any engine is chosen or
  model loaded (both monkeypatched to raise, never entered); the `pass` twin
  reaches the engine.
* **T12.** A rerun retains the prior unit byte for byte as superseded, and
  `select` over the published table reads only the new unit.
* **T13.** The legitimate campaign: every shipped and demo workflow loads with
  no §2.8 key on any entry; both pins hold against the pin *file*.
* **T14.** `_step_identity` returns what it returned for every existing kind;
  a clean run then `--resume` reuses every step byte-identically with the
  skip preserved; a published script record's keys are the base's plus
  `disposition`. And moved evidence re-evaluates the conditional.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main as cli_main
from causalab.io.events import EVENTS_FILE, read_events, terminal
from causalab.io.step_record import SIDECAR, read_sidecar
from causalab.protocol.identity import import_closure
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult
from causalab.protocol.rules.errors import ProtocolError
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import COMPONENTS
from causalab.workflow import conditional as cond
from causalab.workflow import manifest as mf
from causalab.workflow import runner
from causalab.workflow.behavioral import BEHAVIORAL_RULE, DECISION_FILE, DECISION_TYPES
from causalab.workflow.conditional import (
    COMPARATORS,
    CONDITIONAL_RULE,
    DECISION_FIELDS,
    EXECUTABLE_SCOPES,
    FIELD_VOCABULARIES,
    PREDICATE_COMPARATORS,
    RECEIPT_OUTCOMES,
    SCOPES,
    SKIPPED_BY_FIELDS,
)
from causalab.workflow.derived import derive_statuses
from causalab.workflow.document import (
    MAX_RULE,
    STEP_TYPES,
    BehavioralStep,
    ConditionalStep,
    DecisionStep,
    LoadedWorkflow,
    ProtocolStep,
    ScriptStep,
    WorkflowError,
    load_workflow,
)
from causalab.workflow.runner import run_workflow
from causalab.workflow.scripts import select
from tests.protocol.test_vocabulary_census import CODE, _rows  # the table parser
from tests.workflow.test_closure_census import (
    CLOSURES,
    DEMO_WORKFLOWS,
    REDUCE,
    SHARED,
    SHIPPED,
    WORKFLOWS,
    _closure,  # pyright: ignore[reportPrivateUsage]
    _demo_env,  # pyright: ignore[reportPrivateUsage]
)
from tests.workflow.test_controls import _section  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"
FIXTURES = Path(__file__).parent / "fixtures" / "conditional"
BEHAVIORAL_FIXTURES = Path(__file__).parent / "fixtures" / "behavioral"
MODULE = "causalab/workflow/conditional.py"
SMOKE_TEST = REPO / "tests/neural/engines/pytorch_hooks/test_conditional_run.py"
#: every key a decision or conditional entry carries, or a receipt adds, and
#: no other entry may (§7)
CONDITIONAL_KEYS = (
    "values",
    "rule",
    "predicate",
    "on_true",
    "on_false",
    "scope",
    "requires_receipt",
)
#: the keys a published script step's record carried before this PR
#: (`runner._run_script_step` plus `_verify_outputs`): T14 pins that a run
#: today adds `disposition` and nothing else
BASE_SCRIPT_RECORD_KEYS = frozenset(
    {
        "type",
        "status",
        "identity",
        "implementation",
        "script",
        "script_sha256",
        "digest",
        "is_deterministic",
        "inputs",
        "axes",
        "files",
        "digests",
        "checks",
    }
)
#: the word §2.8, item 18, §8 and this layer never use — spelled in two halves
#: so this file does not use it either
AMBIGUOUS = re.compile(r"\b" + "br" + "anch" + r"(es|ing)?\b", re.IGNORECASE)
CHAIN = ("measure", "gate_k", "gate", "fit", "probe", "report")


# --------------------------------------------------------------------------- #
# the fixture chain
# --------------------------------------------------------------------------- #


def _tree(tmp: Path, score: float = 0.9) -> Path:
    """A private copy of the fixture tree with its own measurement file — an
    absolute path, deferred at load — so two runs of **one** document can
    differ in nothing but what was measured."""
    root = tmp / "conditional"
    shutil.copytree(FIXTURES, root)
    _measure(root, score)
    raw = json.loads((root / "gate.json").read_text())
    raw["steps"]["measure"]["inputs"]["measurement"] = {
        "path": str(root / "measurement.json")
    }
    (root / "gate.json").write_text(json.dumps(raw, indent=2))
    return root


def _measure(root: Path, score: float) -> None:
    (root / "measurement.json").write_text(json.dumps({"score": score}))


def _raw(root: Path) -> dict[str, Any]:
    return json.loads((root / "gate.json").read_text())


def _env(root: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


def _load(root: Path, raw: dict[str, Any] | None = None) -> LoadedWorkflow:
    return load_workflow(
        raw if raw is not None else root / "gate.json", _env(root), workflow_dir=root
    )


def _run(
    loaded: LoadedWorkflow,
    root: Path,
    out: Path,
    engine: Any = None,
    **kw: Any,
) -> Any:
    return run_workflow(loaded, _env(root), out, engine, **kw)


def _statuses(result: Any) -> dict[str, str]:
    return {name: entry["status"] for name, entry in result.manifest["steps"].items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(run_root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(run_root)): p.read_bytes()
        for p in sorted(run_root.rglob("*"))
        if p.is_file() and p.name != EVENTS_FILE and p.name != mf.MANIFEST
    }


def _refused(
    root: Path, raw: dict[str, Any], *, rule: int = CONDITIONAL_RULE, path: str
) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        _load(root, raw)
    assert err.value.rule == rule, str(err.value)
    assert path in str(err.value), str(err.value)
    return err.value


def _tables(section: str, header: str) -> list[list[list[str]]]:
    """Every table under ``section`` whose header row starts with the plain
    word ``header``: each as its body rows, up to the first row whose first
    cell is not code."""
    rows = _rows(_section(section))
    out: list[list[list[str]]] = []
    for index, row in enumerate(rows):
        if row[0] != header:
            continue
        body: list[list[str]] = []
        for candidate in rows[index + 1 :]:
            if not candidate[0].startswith("`"):
                break
            body.append(candidate)
        out.append(body)
    return out


def _members(section: str, header: str, which: int = 0) -> list[str]:
    return [CODE.findall(row[0])[0] for row in _tables(section, header)[which]]


SECTION_28 = "### 2.8 `decision` and `conditional` steps"


# --------------------------------------------------------------------------- #
# the censuses
# --------------------------------------------------------------------------- #


def test_five_step_kinds_and_rule_18_is_the_conditional_rule() -> None:
    assert STEP_TYPES == (
        "intervention_protocol",
        "script",
        "behavioral",
        "decision",
        "conditional",
        "workflow",
    )
    assert CONDITIONAL_RULE == 18 and MAX_RULE >= 18 and BEHAVIORAL_RULE == 17
    section = _section("## 5. Validation")
    items = {
        int(number): text
        for number, text in re.findall(
            r"^(\d+)\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S
        )
    }
    assert max(items) == MAX_RULE
    for word in (
        "`decision`",
        "`conditional`",
        "`requires_receipt`",
        "`scope`",
        "`predicate`",
        "§2.9",
        "names the field",
    ):
        assert word in items[18], word


def test_the_comparator_table_is_the_code() -> None:
    assert _members(SECTION_28, "comparator") == list(COMPARATORS)
    assert set(PREDICATE_COMPARATORS) < set(COMPARATORS)
    assert COMPARATORS[-1] == "in"  # the one list-valued comparator


def test_the_scope_table_is_the_code_and_every_scope_executes() -> None:
    """§2.9: the two per-child scopes execute over a declared
    fan-out, so the `executes` column reads ✓ ✓ ✓ and the two tuples are one."""
    rows = _tables(SECTION_28, "scope")[0]
    assert [CODE.findall(row[0])[0] for row in rows] == list(SCOPES)
    executes = {CODE.findall(row[0])[0]: row[1] for row in rows}
    assert EXECUTABLE_SCOPES == SCOPES == ("global", "per_target", "per_variable")
    assert all(executes[scope] == "✓" for scope in SCOPES), executes
    for scope in ("per_target", "per_variable"):
        assert "§2.9" in rows[list(SCOPES).index(scope)][2], scope


def test_the_decision_field_table_is_the_code() -> None:
    assert _members(SECTION_28, "decision field") == list(DECISION_FIELDS)
    assert FIELD_VOCABULARIES["outcome"] == RECEIPT_OUTCOMES == ("pass", "fail")
    assert FIELD_VOCABULARIES["decision_type"] == DECISION_TYPES
    assert set(FIELD_VOCABULARIES) == set(DECISION_FIELDS)


def test_the_three_field_tables_are_the_three_grammars() -> None:
    tables = _tables(SECTION_28, "field")
    assert len(tables) == 3
    decision, conditional, receipt = (
        [CODE.findall(row[0])[0] for row in table] for table in tables
    )
    assert decision == ["values", "rule", "decision"]
    assert conditional == ["predicate", "on_true", "on_false", "scope"]
    assert receipt == ["step", "outcome"]
    assert set(decision + conditional + ["requires_receipt"]) - {"decision"} == set(
        CONDITIONAL_KEYS
    )


def test_the_disposition_table_is_the_code() -> None:
    assert _members("## 8. Runner contract", "disposition") == list(mf.DISPOSITIONS)
    assert mf.DISPOSITIONS == ("candidate", "accepted", "inadmissible", "superseded")


def test_the_skipped_by_fields_are_the_spec_entry() -> None:
    section = _section(SECTION_28)
    for field in SKIPPED_BY_FIELDS:
        assert f'"{field}"' in section, field
    assert SKIPPED_BY_FIELDS[-1] == "transitive_from"
    assert "evidence_identity" in SKIPPED_BY_FIELDS


def test_the_new_prose_never_uses_the_word_ambiguous_with_a_git_ref() -> None:
    """Keep graph-side terminology consistent in conditional workflow sections."""
    texts = {
        "§2.8": _section(SECTION_28),
        "§5 item 18": re.search(
            r"^18\. (.+?)(?=^\d+\. |\Z)",
            _section("## 5. Validation"),
            re.M | re.S,
        ).group(1),  # type: ignore[union-attr]
        "§8": _section("## 8. Runner contract"),
        MODULE: (REPO / MODULE).read_text(),
        "this file": Path(__file__).read_text(),
        "the smoke test": SMOKE_TEST.read_text(),
        # §2.9, item 19, fan_out.py and its two test files inherit
        # the ban — one census, extended rather than copied
        "§2.9": _section("### 2.9 `fan_out` — a declared fan-out and its join"),
        "§5 item 19": re.search(
            r"^19\. (.+?)(?=^\d+\. |\Z)",
            _section("## 5. Validation"),
            re.M | re.S,
        ).group(1),  # type: ignore[union-attr]
        "fan_out.py": (REPO / "causalab/workflow/fan_out.py").read_text(),
        "test_fan_out.py": (Path(__file__).parent / "test_fan_out.py").read_text(),
        "the fan-out smoke test": (
            REPO / "tests/neural/engines/pytorch_hooks/test_fan_out_run.py"
        ).read_text(),
        # §2.10, item 20, nested.py and its two test files inherit
        # the ban the same way
        "§2.10": _section("### 2.10 `workflow` — a nested reusable workflow"),
        "§5 item 20": re.search(
            r"^20\. (.+?)(?=^\d+\. |\Z)",
            _section("## 5. Validation"),
            re.M | re.S,
        ).group(1),  # type: ignore[union-attr]
        "nested.py": (REPO / "causalab/workflow/nested.py").read_text(),
        "test_nested.py": (Path(__file__).parent / "test_nested.py").read_text(),
        "the nested smoke test": (
            REPO / "tests/neural/engines/pytorch_hooks/test_nested_run.py"
        ).read_text(),
    }
    for where, text in texts.items():
        assert not AMBIGUOUS.search(text), where
    assert "evidence_identity" in texts["§2.8"]


# --------------------------------------------------------------------------- #
# the closure guard, and torch-free
# --------------------------------------------------------------------------- #


def test_conditional_is_in_no_hashed_closure() -> None:
    """Every hashed script's frozen closure, the reduce script's and the
    SHARED tuple: none names conditional.py — nothing digest-bearing changed,
    which is why both pins hold (T13)."""
    assert MODULE not in SHARED
    for module in (*CLOSURES, REDUCE):
        assert MODULE not in _closure(module), module


def test_conditional_reaches_no_engine_module() -> None:
    """Engine-free: the one member under ``neural/`` is the torch-free
    enumerator the workflow layer reads the steps through
    (``neural/shared/sweep.py``), never an engine."""
    members = import_closure(REPO / MODULE, root=REPO)
    assert members, "the closure walk found nothing"
    assert [m for m in members if m.startswith("causalab/neural/")] == [
        "causalab/neural/shared/sweep.py"
    ]


_PROBE = """
import json, sys
from pathlib import Path
import causalab.workflow.conditional
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.workflow.document import load_workflow
root = Path(sys.argv[1])
env = ResolutionEnv(datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root))
loaded = load_workflow(root / "gate.json", env)
print(json.dumps({"digest": loaded.digest, "torch": "torch" in sys.modules}))
"""


def test_loading_the_chain_imports_no_torch() -> None:
    """A subprocess, as in test_load_is_torch_free.py: conftest has already
    imported torch here. The committed fixture loads as-is (its measurement
    path is relative to the document, and the file sits beside it)."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(FIXTURES)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["torch"] is False
    assert (
        result["digest"] == load_workflow(FIXTURES / "gate.json", _env(FIXTURES)).digest
    )


# --------------------------------------------------------------------------- #
# valid work: the fixture loads, and its canonical entries are the grammar
# --------------------------------------------------------------------------- #


def test_the_fixture_chain_loads_with_the_grammar_in_its_entries(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    loaded = _load(root)
    assert loaded.order == CHAIN
    assert loaded.dependencies == {
        "measure": (),
        "gate_k": ("measure",),
        "gate": ("gate_k",),
        "fit": ("gate", "measure"),  # the derived edge: gated after the conditional
        "probe": ("gate",),
        "report": ("fit",),
    }
    assert isinstance(loaded.document.steps["gate_k"], DecisionStep)
    assert isinstance(loaded.document.steps["gate"], ConditionalStep)
    assert loaded.canonical["steps"]["gate_k"] == {
        "type": "decision",
        "values": {"step": "measure", "file": "values.json"},
        "rule": {"score": {"ge": 0.5}, "k": {"in": [4, 8]}},
        "decision": {"on_pass": "advance", "on_fail": "narrow"},
    }
    assert loaded.canonical["steps"]["gate"] == {
        "type": "conditional",
        "predicate": {"decision": {"step": "gate_k"}, "field": "outcome", "eq": "pass"},
        "on_true": ["fit"],
        "on_false": ["probe"],
        "scope": "global",
    }
    for name, entry in loaded.canonical["steps"].items():
        assert "requires_receipt" not in entry, name  # only when authored
        if entry["type"] == "script":
            assert not set(entry) & set(CONDITIONAL_KEYS), name
    # both kinds have their own identity, as a script step does (§7)
    assert {"gate_k", "gate"} <= set(loaded.step_digests)
    assert len(loaded.step_digests["gate"]) == 64


def test_a_receipt_enters_the_entry_only_when_authored(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    plain = _load(root)
    raw = _raw(root)
    raw["steps"]["fit"]["requires_receipt"] = {"step": "gate_k", "outcome": "pass"}
    with_receipt = _load(root, raw)
    assert with_receipt.canonical["steps"]["fit"]["requires_receipt"] == {
        "step": "gate_k",
        "outcome": "pass",
    }
    assert with_receipt.step_digests["fit"] != plain.step_digests["fit"]
    for name in CHAIN:
        if name != "fit":
            assert (
                with_receipt.canonical["steps"][name] == plain.canonical["steps"][name]
            )
    assert isinstance(with_receipt.document.steps["fit"], ScriptStep)
    assert "gate_k" in with_receipt.dependencies["fit"]  # the derived edge


def test_explain_prints_both_kinds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _tree(tmp_path)
    assert (
        cli_main(
            [
                "explain",
                "--engine",
                "auto",
                str(root / "gate.json"),
                "--data-root",
                str(root),
                "--artifacts-root",
                str(root),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "gate_k: decision over measure/values.json — rule on k, score" in out
    assert (
        "gate: conditional on gate_k.outcome eq 'pass' -> on_true [fit] / on_false [probe] (scope global)"
        in out
    )


# --------------------------------------------------------------------------- #
# refusals, each naming the field — and their valid twins
# --------------------------------------------------------------------------- #


def _decision(raw: dict[str, Any], **changes: Any) -> dict[str, Any]:
    raw["steps"]["gate_k"].update(changes)
    return raw


def _conditional(raw: dict[str, Any], **changes: Any) -> dict[str, Any]:
    for key, value in changes.items():
        if value is None:
            raw["steps"]["gate"].pop(key, None)
        else:
            raw["steps"]["gate"][key] = value
    return raw


DECISION_REFUSALS: dict[str, tuple[dict[str, Any], int, str, str]] = {
    "values_with_selector": (
        {"values": {"step": "measure", "file": "values.json", "key": "score"}},
        18,
        ".values",
        "no selector",
    ),
    "values_by_path": (
        {"values": {"path": "tests/workflow/fixtures/conditional/measurement.json"}},
        18,
        ".values",
        "names a step's values file",
    ),
    "values_not_json": (
        {"values": {"step": "measure", "file": "values.safetensors"}},
        18,
        ".values",
        ".json values object",
    ),
    "values_unknown_step": (
        {"values": {"step": "measur", "file": "values.json"}},
        18,
        ".values.step",
        "unknown step 'measur'",
    ),
    "values_file_not_written": (
        {"values": {"step": "measure", "file": "other.json"}},
        4,
        ".values",
        "writes no 'other.json'",
    ),
    "rule_key_not_declared": (
        {"rule": {"z": {"eq": 1}}},
        4,
        ".rule.z",
        "declares no emitted key 'z'",
    ),
    "rule_two_comparators": (
        {"rule": {"score": {"ge": 0.5, "le": 1.0}}},
        18,
        ".rule.score",
        "exactly one comparator",
    ),
    "rule_unknown_comparator": (
        {"rule": {"score": {"between": [0, 1]}}},
        18,
        ".rule.score",
        "'between' is not one of",
    ),
    "rule_ordered_over_a_string": (
        {"rule": {"score": {"ge": "0.5"}}},
        18,
        ".rule.score",
        "compares against a number",
    ),
    "rule_in_without_a_list": (
        {"rule": {"k": {"in": 8}}},
        18,
        ".rule.k",
        "non-empty list",
    ),
    "rule_operand_an_object": (
        {"rule": {"score": {"eq": {"step": "measure"}}}},
        18,
        ".rule.score",
        "no expression, no arithmetic and no reference",
    ),
    "rule_empty": ({"rule": {}}, 18, ".rule", "maps each key"),
    "decision_outside_vocabulary": (
        {"decision": {"on_pass": "proceed", "on_fail": "narrow"}},
        18,
        ".decision.on_pass",
        "'proceed' is not one of",
    ),
    "decision_missing_on_fail": (
        {"decision": {"on_pass": "advance"}},
        18,
        ".decision.on_fail",
        "declares 'on_fail'",
    ),
}


@pytest.mark.parametrize("case", sorted(DECISION_REFUSALS))
def test_a_malformed_decision_is_refused_naming_the_field(
    tmp_path: Path, case: str
) -> None:
    changes, rule, path, message = DECISION_REFUSALS[case]
    root = _tree(tmp_path)
    err = _refused(
        root, _decision(_raw(root), **changes), rule=rule, path=f"steps.gate_k{path}"
    )
    assert message in str(err), str(err)


CONDITIONAL_REFUSALS: dict[str, tuple[dict[str, Any], int, str, str]] = {
    "predicate_over_a_script_step": (
        {
            "predicate": {
                "decision": {"step": "measure"},
                "field": "outcome",
                "eq": "pass",
            }
        },
        18,
        ".predicate.decision.step",
        "a script step",
    ),
    "predicate_field_a_measured_number": (
        {"predicate": {"decision": {"step": "gate_k"}, "field": "score", "eq": "pass"}},
        18,
        ".predicate.field",
        "never a measured number",
    ),
    "predicate_literal_outside_vocabulary": (
        {
            "predicate": {
                "decision": {"step": "gate_k"},
                "field": "outcome",
                "eq": "maybe",
            }
        },
        18,
        ".predicate.eq",
        "'maybe' is not one of outcome's vocabulary",
    ),
    "predicate_in_with_a_stranger": (
        {
            "predicate": {
                "decision": {"step": "gate_k"},
                "field": "decision_type",
                "in": ["advance", "proceed"],
            }
        },
        18,
        ".predicate.in",
        "'proceed' is not one of decision_type's vocabulary",
    ),
    "predicate_two_comparators": (
        {
            "predicate": {
                "decision": {"step": "gate_k"},
                "field": "outcome",
                "eq": "pass",
                "ne": "fail",
            }
        },
        18,
        ".predicate",
        "exactly one of",
    ),
    "predicate_ordered_comparator": (
        {
            "predicate": {
                "decision": {"step": "gate_k"},
                "field": "outcome",
                "lt": "pass",
            }
        },
        18,
        ".predicate",
        "unknown key 'lt'",
    ),
    "predicate_decision_unknown_step": (
        {
            "predicate": {
                "decision": {"step": "qualify"},
                "field": "outcome",
                "eq": "pass",
            }
        },
        18,
        ".predicate.decision.step",
        "unknown step 'qualify'",
    ),
    "on_true_empty": ({"on_true": []}, 18, ".on_true", "non-empty list"),
    "on_false_a_string": ({"on_false": "probe"}, 18, ".on_false", "non-empty list"),
    "sides_overlap": ({"on_true": ["fit", "probe"]}, 18, ".on_false", "disjoint"),
    "sides_not_dependency_disjoint": (
        {"on_true": ["fit"], "on_false": ["report"]},  # report depends on fit
        18,
        ".on_false",
        "'on_false' names 'report', which depends on ['fit'] in 'on_true'",
    ),
    "on_true_unknown_step": (
        {"on_true": ["fitt"]},
        18,
        ".on_true",
        "unknown step 'fitt'",
    ),
    "on_true_the_producer": (
        {"on_true": ["gate_k"]},
        18,
        ".on_true",
        "predicate's own producer",
    ),
    "on_true_the_conditional_itself": (
        {"on_true": ["gate"]},
        18,
        ".on_true",
        "names the step itself",
    ),
    "gated_step_upstream_of_the_conditional": (
        {"on_true": ["measure"]},
        5,
        "cycle",
        "cycle",
    ),
    # `per_target` / `per_variable` execute over a declared fan-out
    # (§2.9); over the chain's *unfanned* decision producer they are rule-19
    # refusals naming the producer — their twins run in test_fan_out.py
    "scope_for_the_fan_out": (
        {"scope": "per_target"},
        19,
        ".scope",
        "not a fanned-out behavioral step",
    ),
    "scope_per_variable": (
        {"scope": "per_variable"},
        19,
        ".scope",
        "not a fanned-out behavioral step",
    ),
    "scope_unknown": (
        {"scope": "everywhere"},
        18,
        ".scope",
        "'everywhere' is not one of",
    ),
    "scope_missing": ({"scope": None}, 18, ".scope", "a conditional declares 'scope'"),
    "predicate_missing": (
        {"predicate": None},
        18,
        ".predicate",
        "a conditional declares 'predicate'",
    ),
}


@pytest.mark.parametrize("case", sorted(CONDITIONAL_REFUSALS))
def test_a_malformed_conditional_is_refused_naming_the_field(
    tmp_path: Path, case: str
) -> None:
    changes, rule, path, message = CONDITIONAL_REFUSALS[case]
    root = _tree(tmp_path)
    where = "the step graph has a cycle" if rule == 5 else f"steps.gate{path}"
    err = _refused(root, _conditional(_raw(root), **changes), rule=rule, path=where)
    assert message in str(err), str(err)


def test_a_transitive_cross_side_dependency_is_refused_at_load(tmp_path: Path) -> None:
    """`on_true = [fit]`, `on_false = [probe]` and
    `probe` depends on `report`, which depends on `fit` — a true verdict would
    skip `probe` (authored to run on false) through `report`; a false one
    would skip `fit`. Detectable at load, so refused there, under rule 18,
    naming the side, the gated step and the crossed steps."""
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["probe"]["inputs"] = {
        "k": {"step": "report", "file": "report.json", "key": "ran"}
    }
    err = _refused(root, raw, path="steps.gate.on_false")
    message = str(err)
    assert "'on_false' names 'probe', which depends on ['fit'] in 'on_true'" in message
    assert "dependency-disjoint" in message
    assert "could never launch the side it was authored to launch" in message


def test_dependency_disjoint_sides_load_and_run(tmp_path: Path) -> None:
    """The twin of the cross-side refusal: the fixture's
    sides are disjoint (`fit` | `probe`, `report` follows `fit` on no side)
    and a side that carries a dependent *with* its upstream (`fit` and
    `report` both on `on_true`) is legal — a dependency inside one side
    crosses nothing. Every shipped workflow and demo still loads: T13,
    `PINS` and `DEMO_WORKFLOWS`, run beside this file."""
    root = _tree(tmp_path)
    loaded = _load(root)  # the fixture: disjoint
    assert loaded.dependencies["report"] == ("fit",)
    same_side = _load(root, _conditional(_raw(root), on_true=["fit", "report"]))
    result = _run(same_side, root, tmp_path / "runs")
    assert _statuses(result) == {
        "measure": "completed",
        "gate_k": "completed",
        "gate": "completed",
        "fit": "completed",
        "probe": "skipped",
        "report": "completed",
    }
    # the pure check, on the loaded graph: a crossing is found through the
    # schedule's map, derived edges included
    cond.check_conditional(loaded.document.steps, loaded.dependencies)
    crossed = _conditional(_raw(root), on_true=["fit"], on_false=["report"])
    with pytest.raises(WorkflowError, match="dependency-disjoint"):
        _load(root, crossed)


RECEIPT_REFUSALS: dict[str, tuple[str, dict[str, Any], str, str]] = {
    "receipt_of_a_script_step": (
        "fit",
        {"step": "measure", "outcome": "pass"},
        ".requires_receipt.step",
        "a receipt is a decision.json",
    ),
    "receipt_outcome_outside_vocabulary": (
        "fit",
        {"step": "gate_k", "outcome": "maybe"},
        ".requires_receipt.outcome",
        "'maybe' is not one of",
    ),
    "receipt_unknown_step": (
        "fit",
        {"step": "qualify", "outcome": "pass"},
        ".requires_receipt.step",
        "unknown step 'qualify'",
    ),
    "receipt_of_itself": (
        "gate_k",
        {"step": "gate_k", "outcome": "pass"},
        ".requires_receipt.step",
        "names the step itself",
    ),
    "receipt_missing_outcome": (
        "fit",
        {"step": "gate_k"},
        ".requires_receipt.outcome",
        "missing required key",
    ),
    "receipt_not_an_object": (
        "fit",
        "gate_k",
        ".requires_receipt",
        "'requires_receipt' is",
    ),
}


@pytest.mark.parametrize("case", sorted(RECEIPT_REFUSALS))
def test_a_malformed_receipt_is_refused_naming_the_field(
    tmp_path: Path, case: str
) -> None:
    step, block, path, message = RECEIPT_REFUSALS[case]
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"][step]["requires_receipt"] = block
    err = _refused(root, raw, path=f"steps.{step}{path}")
    assert message in str(err), str(err)


def test_a_reference_to_a_conditional_is_refused(tmp_path: Path) -> None:
    """A conditional publishes no data file (§2.8): rule 4 says so."""
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["report"]["inputs"]["k"] = {
        "step": "gate",
        "file": "_step.json",
        "key": "verdict",
    }
    err = _refused(root, raw, rule=4, path="steps.report.inputs.k")
    assert "'gate' writes no '_step.json' (has [])" in str(err)


def test_a_decision_declares_no_keys_so_a_key_reference_to_it_is_refused(
    tmp_path: Path,
) -> None:
    """Default (e): a decision is consumed by name; the widened rule-4
    message says what kind of step it was."""
    root = _tree(tmp_path)
    raw = _raw(root)
    raw["steps"]["report"]["inputs"]["k"] = {
        "step": "gate_k",
        "file": DECISION_FILE,
        "key": "outcome",
    }
    err = _refused(root, raw, rule=4, path="steps.report.inputs.k")
    assert "a decision step" in str(err) and "never by key" in str(err)


def _rule_for(comparator: str) -> dict[str, Any]:
    """One rule using ``comparator``, which the fixture measurement (score
    0.9, k 8) passes."""
    return {
        "eq": {"k": {"eq": 8}},
        "ne": {"k": {"ne": 4}},
        "lt": {"score": {"lt": 1.0}},
        "le": {"k": {"le": 8}},
        "gt": {"score": {"gt": 0.1}},
        "ge": {"score": {"ge": 0.5}},
        "in": {"k": {"in": [4, 8]}},
    }[comparator]


TWINS: dict[str, Any] = {
    "the_fixture": lambda raw: raw,
    "predicate_on_decision_type_with_in": lambda raw: _conditional(
        raw,
        predicate={
            "decision": {"step": "gate_k"},
            "field": "decision_type",
            "in": ["advance", "revise"],
        },
    ),
    "predicate_with_ne": lambda raw: _conditional(
        raw,
        predicate={"decision": {"step": "gate_k"}, "field": "outcome", "ne": "fail"},
    ),
    "scope_global_spelled": lambda raw: _conditional(raw, scope="global"),
    "receipt_pass_on_the_gated_step": lambda raw: (
        raw["steps"]["fit"].__setitem__(
            "requires_receipt", {"step": "gate_k", "outcome": "pass"}
        )
        or raw
    ),
    "receipt_on_the_conditional_itself": lambda raw: _conditional(
        raw, requires_receipt={"step": "gate_k", "outcome": "pass"}
    ),
    **{
        f"rule_{c}": (lambda raw, c=c: _decision(raw, rule=_rule_for(c)))
        for c in COMPARATORS
    },
}


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_the_valid_twin_loads_and_runs_to_completed(tmp_path: Path, twin: str) -> None:
    """Every refusal has a twin that passes — each
    comparator once, each predicate comparator, an explicit global scope, a
    pass receipt on a gated step and on the conditional itself."""
    root = _tree(tmp_path)
    loaded = _load(root, TWINS[twin](_raw(root)))
    result = _run(loaded, root, tmp_path / "runs")
    assert _statuses(result) == {
        "measure": "completed",
        "gate_k": "completed",
        "gate": "completed",
        "fit": "completed",
        "probe": "skipped",
        "report": "completed",
    }
    assert terminal(result.run_root / EVENTS_FILE)


def test_every_comparator_has_a_twin() -> None:
    assert {c for c in COMPARATORS} == {
        t[len("rule_") :] for t in TWINS if t.startswith("rule_")
    }


# --------------------------------------------------------------------------- #
# T10 — conditional steps launch or skip from typed decision records
# --------------------------------------------------------------------------- #


def test_t10_two_runs_differing_only_in_the_measured_input_take_opposite_sides(
    tmp_path: Path,
) -> None:
    """One document, loaded once; the measurement file differs. The skipped
    entry names the decision by `evidence_identity`, equal to the producer's
    `decision.json` — the mutation (a predicate over `values.json` instead of
    `decision.json`) could skip the same step but never carry that id."""
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root)

    high = _run(loaded, root, tmp_path / "high")
    assert _statuses(high) == {
        "measure": "completed",
        "gate_k": "completed",
        "gate": "completed",
        "fit": "completed",
        "probe": "skipped",
        "report": "completed",
    }
    decision = json.loads((high.run_root / "gate_k" / DECISION_FILE).read_text())
    assert decision["outcome"] == "pass" and decision["decision_type"] == "advance"
    skipped = high.manifest["steps"]["probe"]
    assert set(skipped) == {"type", "status", "skipped_by"}
    assert skipped["type"] == "script"
    assert set(skipped["skipped_by"]) == set(SKIPPED_BY_FIELDS)
    assert skipped["skipped_by"] == {
        "conditional": "gate",
        "decision_step": "gate_k",
        "decision_type": "advance",
        "outcome": "pass",
        "evidence_identity": decision["evidence_identity"],
        "transitive_from": [],
    }
    assert skipped["skipped_by"]["evidence_identity"].count(":") == 1
    # no directory, no attempt, no publish for a skipped step (default b)
    assert not (high.run_root / "probe").exists()
    assert not (high.run_root / mf.ATTEMPTS_DIR).exists()
    lines = [
        (r["event"], r["payload"].get("status"))
        for r in read_events(high.run_root / EVENTS_FILE)
        if r["payload"].get("step") == "probe"
    ]
    assert lines == [("phase_started", None), ("phase_completed", "skipped")]
    last = read_events(high.run_root / EVENTS_FILE)[-1]
    assert last["event"] == "campaign_terminal"
    assert last["payload"]["outcome"] == "completed"
    assert last["payload"]["steps"]["probe"] == "skipped"
    assert terminal(high.run_root / EVENTS_FILE)

    _measure(root, 0.1)
    low = _run(loaded, root, tmp_path / "low")
    assert _statuses(low) == {
        "measure": "completed",
        "gate_k": "completed",
        "gate": "completed",
        "fit": "skipped",
        "probe": "completed",
        "report": "skipped",
    }
    other = json.loads((low.run_root / "gate_k" / DECISION_FILE).read_text())
    assert other["outcome"] == "fail" and other["decision_type"] == "narrow"
    assert other["evidence_identity"] != decision["evidence_identity"]
    # the producer identity half is the same step; the values half moved
    assert (
        other["evidence_identity"].split(":")[1]
        == decision["evidence_identity"].split(":")[1]
    )
    fit, report = low.manifest["steps"]["fit"], low.manifest["steps"]["report"]
    assert fit["skipped_by"]["evidence_identity"] == other["evidence_identity"]
    assert fit["skipped_by"]["transitive_from"] == []
    assert report["status"] == "skipped"
    assert report["skipped_by"]["transitive_from"] == ["fit"]
    assert report["skipped_by"]["evidence_identity"] == other["evidence_identity"]
    assert report["skipped_by"]["conditional"] == "gate"
    assert (
        not (low.run_root / "fit").exists() and not (low.run_root / "report").exists()
    )
    # the stream is the authority: derived statuses equal the manifest's
    records = read_events(low.run_root / EVENTS_FILE)
    assert derive_statuses(
        records, order=loaded.order, dependencies=loaded.dependencies
    ) == _statuses(low)


def test_the_two_skipped_by_emitters_write_the_same_six_keys() -> None:
    """`SKIPPED_BY_FIELDS` is manifest vocabulary
    (beside `STEP_STATUSES`), re-exported by the conditional layer; the
    reached skip (`conditional.skipped_entry`) and the unreached one
    (`manifest.classify_unreached`) write exactly its six keys."""
    assert cond.SKIPPED_BY_FIELDS is mf.SKIPPED_BY_FIELDS
    assert "SKIPPED_BY_FIELDS" in mf.__all__ and "SKIPPED_BY_FIELDS" in cond.__all__
    assert len(SKIPPED_BY_FIELDS) == 6
    reached = cond.skipped_entry("script", {"conditional": "gate"})["skipped_by"]
    unreached = mf.classify_unreached(
        ("a", "b", "c"),
        {"a": (), "b": ("a",), "c": ("b",)},
        {"a": {"status": "skipped", "skipped_by": reached}},
    )
    assert set(reached) == set(SKIPPED_BY_FIELDS)
    assert set(unreached["b"]["skipped_by"]) == set(SKIPPED_BY_FIELDS)
    assert set(unreached["c"]["skipped_by"]) == set(SKIPPED_BY_FIELDS)
    assert reached["conditional"] == "gate" and reached["evidence_identity"] is None
    assert unreached["b"]["skipped_by"] == {**reached, "transitive_from": ["a"]}
    assert unreached["c"]["skipped_by"] == {**reached, "transitive_from": ["b"]}
    # an upstream skipped with no block at all still yields the six keys
    bare = mf.classify_unreached(
        ("a", "b"), {"b": ("a",)}, {"a": {"status": "skipped"}}
    )
    assert bare["b"]["skipped_by"] == {
        **{f: None for f in SKIPPED_BY_FIELDS},
        "transitive_from": ["a"],
    }


def test_an_unreached_skip_inherits_the_decision_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run dies at `probe` (a false verdict skipped `fit`), so `report`,
    which depends on `fit`, is never reached: its entry is the manifest's
    unreached skip, and it names the decision by `evidence_identity` exactly
    as the reached skip of `fit` does — T10's assertion, on the second
    emitter."""
    root = _tree(tmp_path, score=0.1)
    loaded = _load(root)
    assert loaded.order.index("probe") < loaded.order.index("report")

    def die(name: str, at: str | None) -> None:
        if name == "attempt_created" and at == "probe":
            raise RuntimeError("probe dies before writing")

    monkeypatch.setattr(runner, "_boundary", die)
    with pytest.raises(RuntimeError, match="probe dies"):
        _run(loaded, root, tmp_path / "runs")
    run_root = tmp_path / "runs" / "gated"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert {n: e["status"] for n, e in manifest.items()} == {
        "measure": "completed",
        "gate_k": "completed",
        "gate": "completed",
        "fit": "skipped",
        "probe": "failed",
        "report": "skipped",
    }
    decision = json.loads((run_root / "gate_k" / DECISION_FILE).read_text())
    fit, report = manifest["fit"], manifest["report"]
    assert set(report["skipped_by"]) == set(SKIPPED_BY_FIELDS)
    assert report["skipped_by"]["evidence_identity"] == decision["evidence_identity"]
    assert report["skipped_by"] == {**fit["skipped_by"], "transitive_from": ["fit"]}
    assert report["skipped_by"]["conditional"] == "gate"
    assert report["skipped_by"]["decision_step"] == "gate_k"
    assert report["skipped_by"]["outcome"] == "fail"
    assert report["skipped_by"]["decision_type"] == "narrow"
    assert set(report) == {"status", "skipped_by"}  # unreached: no `type`


def test_the_decision_step_writes_the_six_fields_and_no_split(tmp_path: Path) -> None:
    """I16: `select`'s values object wrapped, not replaced — the record binds
    to the values file's bytes and its producer's identity."""
    root = _tree(tmp_path)
    result = _run(_load(root), root, tmp_path / "runs")
    run_root = result.run_root
    decision = json.loads((run_root / "gate_k" / DECISION_FILE).read_text())
    assert set(decision) == {
        "decision_type",
        "schema_version",
        "measured_inputs",
        "rule",
        "outcome",
        "evidence_identity",
        "step",
    }
    assert decision["schema_version"] == 1 and decision["step"] == "gate_k"
    assert decision["measured_inputs"] == {"score": 0.9, "k": 8}
    assert decision["rule"] == {"score": {"ge": 0.5}, "k": {"in": [4, 8]}}
    values_sha, identity = decision["evidence_identity"].split(":")
    assert values_sha == _sha256(run_root / "measure" / "values.json")
    assert identity == read_sidecar(run_root / "measure" / "values.json")["identity"]
    record = json.loads((run_root / "gate_k" / SIDECAR).read_text())
    assert record["type"] == "decision" and record["status"] == "completed"
    assert record["files"] == [DECISION_FILE] and record["decision"] == DECISION_FILE
    assert record["identity"] == _load(root).step_digests["gate_k"]
    assert record["disposition"] == "accepted"
    assert record["measured"] == decision["measured_inputs"]
    assert record["digests"] == {
        DECISION_FILE: _sha256(run_root / "gate_k" / DECISION_FILE)
    }
    gate = json.loads((run_root / "gate" / SIDECAR).read_text())
    assert gate["type"] == "conditional" and gate["verdict"] is True
    assert gate["files"] == [] and gate["digests"] == {}
    assert gate["skipped"] == ["probe"] and gate["scope"] == "global"
    assert gate["evidence"] == {
        "step": "gate_k",
        "decision_type": "advance",
        "outcome": "pass",
        "evidence_identity": decision["evidence_identity"],
    }


def test_a_predicate_reads_only_a_decision_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-time half of the mutation guard: a producer whose published
    `decision.json` is not a schema-version-1 record is refused, never read
    as if it were a values object (injected at the producer's `published`
    seam)."""
    root = _tree(tmp_path)
    loaded = _load(root)
    run_root = tmp_path / "runs" / "gated"

    def corrupt(name: str, at: str | None) -> None:
        if name == "published" and at == "gate_k":
            (run_root / "gate_k" / DECISION_FILE).write_text(json.dumps({"score": 0.9}))

    monkeypatch.setattr(runner, "_boundary", corrupt)
    with pytest.raises(ProtocolError, match="schema_version 1 decision record"):
        _run(loaded, root, tmp_path / "runs")


# --------------------------------------------------------------------------- #
# T11 — a failed or missing smoke receipt prevents allocation
# --------------------------------------------------------------------------- #


def test_a_predicate_over_an_absent_or_foreign_field_is_p2_not_a_verdict() -> None:
    """`ne` over a missing field once held (`None !=
    "fail"`), so a true verdict rested on nothing; `eq`/`in` failed closed.
    Either way an absence decided the side. Now a record without the field,
    or with a value outside `FIELD_VOCABULARIES[field]`, is a `P2` for every
    comparator — and a present in-vocabulary value gives the verdicts it did."""
    what = "step 'gate': predicate"
    field = {"decision": {"step": "gate_k"}, "field": "outcome"}
    eq, ne, in_ = (
        {**field, "eq": "pass"},
        {**field, "ne": "fail"},
        {**field, "in": ["pass"]},
    )
    absent = {"decision_type": "advance", "schema_version": 1}
    for predicate in (eq, ne, in_):
        with pytest.raises(ProtocolError, match="has no field 'outcome'") as err:
            cond.evaluate_predicate(predicate, absent, what=what)
        assert err.value.code == "P2" and what in str(err.value)
        assert "cannot rest on an absence" in str(err.value)
    foreign = {**absent, "outcome": "maybe"}
    for predicate in (eq, ne, in_):
        with pytest.raises(ProtocolError, match="'outcome' is 'maybe'") as err:
            cond.evaluate_predicate(predicate, foreign, what=what)
        assert err.value.code == "P2" and "['pass', 'fail']" in str(err.value)
    with pytest.raises(ProtocolError, match="'outcome' is None"):
        cond.evaluate_predicate(ne, {**absent, "outcome": None}, what=what)
    # present, in the vocabulary: the verdicts are unchanged
    passed, failed = {**absent, "outcome": "pass"}, {**absent, "outcome": "fail"}
    assert cond.evaluate_predicate(eq, passed, what=what) is True
    assert cond.evaluate_predicate(eq, failed, what=what) is False
    assert cond.evaluate_predicate(ne, passed, what=what) is True
    assert cond.evaluate_predicate(ne, failed, what=what) is False
    assert cond.evaluate_predicate(in_, passed, what=what) is True
    assert cond.evaluate_predicate(in_, failed, what=what) is False
    kind = {"decision": {"step": "gate_k"}, "field": "decision_type"}
    assert (
        cond.evaluate_predicate(
            {**kind, "in": ["advance", "revise"]}, passed, what=what
        )
        is True
    )
    with pytest.raises(ProtocolError, match="'decision_type' is 'proceed'"):
        cond.evaluate_predicate(
            {**kind, "in": ["advance"]},
            {**passed, "decision_type": "proceed"},
            what=what,
        )


class _Stub(Engine):
    """An engine able to serve the fixture document whose `execute` raises:
    reaching it means an engine was chosen and a model would have loaded."""

    def __init__(self) -> None:
        self.name = "pytorch_hooks"
        self.capabilities = frozenset(
            {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
        )
        self.components = frozenset(COMPONENTS)
        self.writable_components = frozenset(COMPONENTS)
        self.is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        raise AssertionError("executed — the engine was reached")


def _receipt_tree(tmp: Path, score: float) -> Path:
    """The conditional fixtures plus the behavioral fixtures' document and
    table, so a real protocol step (gpt2 on paper, torch-free at load) can
    require the decision step's receipt."""
    root = _tree(tmp, score)
    shutil.copytree(BEHAVIORAL_FIXTURES / "protocols", root / "protocols")
    shutil.copytree(BEHAVIORAL_FIXTURES / "qa", root / "qa")
    return root


def _receipt_raw(root: Path, outcome: str = "pass") -> dict[str, Any]:
    raw = _raw(root)
    for name in ("gate", "fit", "probe", "report"):
        del raw["steps"][name]
    raw["steps"]["fit"] = {
        "type": "intervention_protocol",
        "document": "protocols/qa_probe.json",
        "set": {"data.base.dataset": "qa/data#development"},
        "requires_receipt": {"step": "gate_k", "outcome": outcome},
    }
    return raw


def _sentinels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pattern of `test_legality_before_weights.py`: the reference
    engine's `load_model` and the runner's `route_engine` both raise — neither
    may be entered before the receipt refuses."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded — the refusal came too late")

    def no_route(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("route_engine entered — the refusal came too late")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    monkeypatch.setattr(runner, "route_engine", no_route)


def test_t11_a_failed_receipt_prevents_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _receipt_tree(tmp_path, score=0.1)  # gate_k: fail
    loaded = _load(root, _receipt_raw(root))
    assert isinstance(loaded.document.steps["fit"], ProtocolStep)
    assert loaded.dependencies["fit"] == ("gate_k",)  # the derived edge
    _sentinels(monkeypatch)
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs", _Stub())
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.fit.requires_receipt" in message
    assert "carries outcome 'fail'" in message and "requires 'pass'" in message
    assert "not allocated" in message and "decision 'narrow'" in message
    run_root = tmp_path / "runs" / "gated"
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["gate_k"]["status"] == "completed"
    assert manifest["fit"]["status"] == "failed"  # never `skipped`
    assert "W18" in manifest["fit"]["error"]["message"]
    assert not (run_root / mf.ATTEMPTS_DIR / "fit").exists(), "an attempt was allocated"
    assert not (run_root / "fit").exists()
    warning = [
        r for r in read_events(run_root / EVENTS_FILE) if r["event"] == "warning"
    ]
    assert warning and warning[-1]["payload"]["reason"] == "attempt_failed"
    assert warning[-1]["payload"]["step"] == "fit"


def test_t11_a_missing_receipt_is_a_distinct_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer published, then its receipt vanished before the gated
    step's turn (injected at the producer's `published` seam): a different
    `W18`, naming absence and saying it is not a skip; still before any
    engine or model."""
    root = _receipt_tree(tmp_path, score=0.9)
    loaded = _load(root, _receipt_raw(root))
    run_root = tmp_path / "runs" / "gated"

    def vanish(name: str, at: str | None) -> None:
        if name == "published" and at == "gate_k":
            (run_root / "gate_k" / DECISION_FILE).unlink()

    monkeypatch.setattr(runner, "_boundary", vanish)
    _sentinels(monkeypatch)
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, tmp_path / "runs", _Stub())
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.fit.requires_receipt" in message
    assert "does not exist" in message and "not a skip" in message
    assert "carries outcome" not in message  # distinct from the failed message
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["fit"]["status"] == "failed"
    assert not (run_root / mf.ATTEMPTS_DIR / "fit").exists()
    # and the pure check, on an empty tree: the same two messages
    step = loaded.document.steps["fit"]
    with pytest.raises(WorkflowError, match="does not exist") as missing:
        cond.check_receipt("fit", step, tmp_path / "nowhere")
    (tmp_path / "somewhere" / "gate_k").mkdir(parents=True)
    (tmp_path / "somewhere" / "gate_k" / DECISION_FILE).write_text(
        json.dumps(
            {"outcome": "fail", "decision_type": "narrow", "evidence_identity": "a:b"}
        )
    )
    with pytest.raises(WorkflowError, match="carries outcome 'fail'") as failed:
        cond.check_receipt("fit", step, tmp_path / "somewhere")
    assert str(missing.value) != str(failed.value)


def test_t11_the_pass_twin_reaches_the_engine(tmp_path: Path) -> None:
    """A refusal beside its valid twin: a `pass` receipt
    allocates — the stub engine's `execute` is reached (and raises, so no
    model is needed here)."""
    root = _receipt_tree(tmp_path, score=0.9)
    loaded = _load(root, _receipt_raw(root))
    with pytest.raises(AssertionError, match="executed"):
        _run(loaded, root, tmp_path / "runs", _Stub())
    run_root = tmp_path / "runs" / "gated"
    assert (run_root / mf.ATTEMPTS_DIR / "fit").is_dir()  # allocated this time


def test_t11_the_check_precedes_route_engine_even_for_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T11's second mutation, moving the check after `route_engine`: with
    the receipt failing and `route_engine` a sentinel, the refusal must be
    the receipt's, never the sentinel's."""
    root = _receipt_tree(tmp_path, score=0.1)
    loaded = _load(root, _receipt_raw(root))

    def no_route(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("route_engine entered")

    monkeypatch.setattr(runner, "route_engine", no_route)
    with pytest.raises(WorkflowError, match="not allocated"):
        _run(loaded, root, tmp_path / "runs", _Stub())


def _receipt_only_raw(root: Path) -> dict[str, Any]:
    """The chain without its conditional: `measure` → `gate_k` → `fit`, where
    the script step `fit` requires `gate_k`'s pass receipt — so a flipped
    receipt reaches `fit` through the receipt alone, never through a verdict."""
    raw = _raw(root)
    for name in ("gate", "probe", "report"):
        del raw["steps"][name]
    raw["steps"]["fit"]["requires_receipt"] = {"step": "gate_k", "outcome": "pass"}
    return raw


def test_t11_a_receipt_that_flipped_to_fail_refuses_a_resume_reuse(
    tmp_path: Path,
) -> None:
    """Run 1 publishes `fit` on a
    pass receipt; the measurement changes and `--resume` re-runs `measure`
    and re-makes `gate_k`, whose receipt now says `fail`. `fit`'s identity,
    files and tree digest are unchanged, so the digest clauses would reuse
    it — the receipt clause of `evidence_holds` does not, and the step goes
    back through `check_receipt`: T11's refusal, before any allocation, and
    the manifest says `failed`, never `reused`."""
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root, _receipt_only_raw(root))
    assert loaded.dependencies["fit"] == ("gate_k", "measure")
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    run_root = first.run_root
    assert _statuses(first) == {
        "measure": "completed",
        "gate_k": "completed",
        "fit": "completed",
    }
    fit_record = json.loads((run_root / "fit" / SIDECAR).read_text())
    (run_root / "measure" / "values.json").unlink()  # the producer must re-run
    _measure(root, 0.1)
    with pytest.raises(WorkflowError) as err:
        _run(loaded, root, out, resume=True)
    assert err.value.rule == CONDITIONAL_RULE
    message = str(err.value)
    assert "steps.fit.requires_receipt" in message
    assert "carries outcome 'fail'" in message and "requires 'pass'" in message
    assert "not allocated" in message and "decision 'narrow'" in message
    manifest = json.loads((run_root / mf.MANIFEST).read_text())["steps"]
    assert manifest["measure"]["status"] == "completed"
    assert manifest["gate_k"]["status"] == "completed"
    assert manifest["fit"]["status"] == "failed", manifest["fit"]  # never `reused`
    assert "W18" in manifest["fit"]["error"]["message"]
    assert not (run_root / mf.ATTEMPTS_DIR / "fit").exists(), "an attempt was allocated"
    # the unit that stood on the pass receipt is untouched, not re-accepted
    assert json.loads((run_root / "fit" / SIDECAR).read_text()) == fit_record
    decision = json.loads((run_root / "gate_k" / DECISION_FILE).read_text())
    assert decision["outcome"] == "fail"
    # the pure clause: the same record holds on a pass receipt and not on a fail
    step = loaded.document.steps["fit"]
    assert not cond.evidence_holds(step, run_root / "fit", fit_record)
    (run_root / "gate_k" / DECISION_FILE).write_text(
        json.dumps({**decision, "outcome": "pass"})
    )
    assert cond.evidence_holds(step, run_root / "fit", fit_record)
    (run_root / "gate_k" / DECISION_FILE).unlink()
    assert not cond.evidence_holds(step, run_root / "fit", fit_record)  # missing


def test_t11_an_unchanged_pass_receipt_is_reused_under_resume(tmp_path: Path) -> None:
    """The flipped-receipt test's twin: the receipt still says `pass`, so
    `--resume` reuses the receipt-bearing step — the clause refuses a flipped
    receipt, not a receipt."""
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root, _receipt_only_raw(root))
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    before = _snapshot(first.run_root)
    again = _run(loaded, root, out, resume=True)
    assert _statuses(again) == {
        "measure": "reused",
        "gate_k": "reused",
        "fit": "reused",
    }
    assert _snapshot(first.run_root) == before
    assert not (first.run_root / mf.ATTEMPTS_DIR).exists()


# --------------------------------------------------------------------------- #
# T12 — a rerun marks the prior result superseded while retaining it
# --------------------------------------------------------------------------- #


def test_t12_a_rerun_retains_the_prior_unit_and_select_reads_only_the_new_one(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    run_root = first.run_root
    before = _snapshot(run_root)
    first_identity = json.loads((run_root / "measure" / SIDECAR).read_text())[
        "identity"
    ]

    _measure(root, 0.1)
    second = _run(loaded, root, out)
    assert (
        _statuses(second)["measure"] == "completed"
        and _statuses(second)["fit"] == "skipped"
    )
    retained = run_root / mf.ATTEMPTS_DIR / "measure" / f"0001{mf.SUPERSEDED_SUFFIX}"
    assert retained.is_dir(), "the prior unit was deleted"
    # the prior bytes: readable, and equal to the first run's (the mutation —
    # `rmtree(displaced)` — fails here, on a missing file)
    assert (retained / "values.json").read_bytes() == before["measure/values.json"]
    assert (retained / "scores.json").read_bytes() == before["measure/scores.json"]
    assert json.loads((retained / "values.json").read_text())["score"] == 0.9
    marked = json.loads((retained / SIDECAR).read_text())
    assert marked["status"] == "superseded" and marked["disposition"] == "superseded"
    assert marked["identity"] == first_identity
    published = json.loads((run_root / "measure" / SIDECAR).read_text())
    assert marked["superseded_by"] == {
        "attempt": "0001",
        "identity": published["identity"],
        "published": "measure",
    }
    assert published["disposition"] == "accepted"
    assert second.manifest["steps"]["measure"]["superseded"] == [
        f"{mf.ATTEMPTS_DIR}/measure/0001{mf.SUPERSEDED_SUFFIX}"
    ]
    # the step the rerun skipped: its earlier unit is retained too, and the
    # retention names the decision
    fit_retained = run_root / mf.ATTEMPTS_DIR / "fit" / f"0001{mf.SUPERSEDED_SUFFIX}"
    assert (fit_retained / "fit.json").read_bytes() == before["fit/fit.json"]
    fit_marked = json.loads((fit_retained / SIDECAR).read_text())
    assert fit_marked["status"] == "superseded"
    assert fit_marked["superseded_by"]["skipped_by"]["conditional"] == "gate"
    assert not (run_root / "fit").exists()
    assert second.manifest["steps"]["fit"]["status"] == "skipped"
    assert second.manifest["steps"]["fit"]["superseded"] == [
        f"{mf.ATTEMPTS_DIR}/fit/0001{mf.SUPERSEDED_SUFFIX}"
    ]
    # a default analysis sees only the accepted unit: the hashed `select`
    # reads the published table and the record beside it, untouched
    chosen = tmp_path / "chosen.json"
    select.main(
        {
            "table": run_root / "measure" / "scores.json",
            "value": "score",
            "emit": {"best": "score"},
        },
        {"values": chosen},
    )
    assert json.loads(chosen.read_text()) == {"best": 0.1}
    assert (
        read_sidecar(run_root / "measure" / "scores.json")["disposition"] == "accepted"
    )
    # nothing lingers under a `.previous` name
    assert not list(
        (run_root / mf.ATTEMPTS_DIR / "measure").glob(f"*{mf.DISPLACED_SUFFIX}")
    )


def test_a_stale_displaced_unit_is_retained_not_deleted_on_the_next_run(
    tmp_path: Path,
) -> None:
    """`restore_displaced`'s second half (§8): a `.previous` left by a death
    after the publish landed is retained as superseded, never removed."""
    run_root = tmp_path / "run"
    step_dir = run_root / "measure"
    step_dir.mkdir(parents=True)
    (step_dir / SIDECAR).write_text(
        json.dumps({"identity": "new", "status": "completed"})
    )
    stale = run_root / mf.ATTEMPTS_DIR / "measure" / f"0001{mf.DISPLACED_SUFFIX}"
    stale.mkdir(parents=True)
    (stale / SIDECAR).write_text(json.dumps({"identity": "old", "status": "completed"}))
    (stale / "values.json").write_text('{"score": 0.9}')
    mf.restore_displaced(run_root, "measure", step_dir)
    assert not stale.exists()
    retained = run_root / mf.ATTEMPTS_DIR / "measure" / f"0001{mf.SUPERSEDED_SUFFIX}"
    assert (retained / "values.json").read_text() == '{"score": 0.9}'
    marked = json.loads((retained / SIDECAR).read_text())
    assert marked["status"] == "superseded" and marked["identity"] == "old"
    assert marked["superseded_by"] == {
        "attempt": "0001",
        "identity": "new",
        "published": "measure",
    }
    assert (step_dir / SIDECAR).is_file()  # the published unit untouched


# --------------------------------------------------------------------------- #
# T13 — the legitimate population: no existing digest moves
# --------------------------------------------------------------------------- #


def test_t13_the_population_is_non_empty() -> None:
    assert len(SHIPPED) >= 2 and len(DEMO_WORKFLOWS) >= 9
    documents = [
        path
        for pattern in (
            "demos/**/*.json",
            "tests/protocols/*.json",
            "tests/golden/**/*.json",
        )
        for path in REPO.glob(pattern)
        if '"protocol_version"' in path.read_text()
    ]
    assert len(documents) >= 60, len(documents)


@pytest.mark.parametrize(
    "path", SHIPPED + DEMO_WORKFLOWS, ids=[p.stem for p in SHIPPED + DEMO_WORKFLOWS]
)
def test_t13_every_existing_workflow_loads_with_no_conditional_key(
    path: Path, env: Any
) -> None:
    """A §2.8 key on any other entry would move every workflow digest in the
    repo (the mutation: emit `requires_receipt` unconditionally)."""
    loaded = load_workflow(path, env if path.parent == WORKFLOWS else _demo_env(path))
    for name, entry in loaded.canonical["steps"].items():
        assert entry["type"] in ("intervention_protocol", "script"), name
        assert not set(entry) & set(CONDITIONAL_KEYS), (name, sorted(entry))
        assert "requires_receipt" not in entry, name


# --------------------------------------------------------------------------- #
# T14 — a workflow using none of this runs bit-identically, --resume included
# --------------------------------------------------------------------------- #


def _identity_kinds(loaded: LoadedWorkflow) -> set[str]:
    seen: set[str] = set()
    for name, step in loaded.document.steps.items():
        got = runner._step_identity(loaded, name, step)  # pyright: ignore[reportPrivateUsage]
        if isinstance(
            step, (ScriptStep, BehavioralStep, DecisionStep, ConditionalStep)
        ):
            assert got == loaded.step_digests[name], name
        else:
            assert isinstance(step, ProtocolStep)
            assert got == loaded.inner_digests[name], name
            # the mutation — `step_digests` for a protocol step — has no
            # such entry: a protocol step's identity is its document's
            assert name not in loaded.step_digests, name
        seen.add(step.type)
    return seen


def test_t14_step_identity_is_unchanged_for_every_existing_kind(
    env: Any, tmp_path: Path
) -> None:
    kinds: set[str] = set()
    for name in ("mean_ablation.json", "weekdays.json"):
        kinds |= _identity_kinds(load_workflow(WORKFLOWS / name, env))
    kinds |= _identity_kinds(_load(_tree(tmp_path)))
    assert kinds == {"intervention_protocol", "script", "decision", "conditional"}


def test_t14_a_clean_run_then_resume_reuses_every_step_and_keeps_the_skip(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    run_root = first.run_root
    assert not (run_root / mf.ATTEMPTS_DIR).exists()
    before = _snapshot(run_root)
    again = _run(loaded, root, out, resume=True)
    assert _statuses(again) == {
        "measure": "reused",
        "gate_k": "reused",
        "gate": "reused",
        "fit": "reused",
        "probe": "skipped",  # re-seated from the reused conditional's record
        "report": "reused",
    }
    assert _snapshot(run_root) == before
    assert not (run_root / mf.ATTEMPTS_DIR).exists()
    assert not (run_root / "probe").exists()
    assert (
        again.manifest["steps"]["probe"]["skipped_by"]
        == first.manifest["steps"]["probe"]["skipped_by"]
    )


def test_t14_a_published_script_record_gains_disposition_and_nothing_else(
    tmp_path: Path,
) -> None:
    """The manifest a run writes today differs from the base's by the added
    `disposition` (every published record) and `superseded` (a rerun) keys —
    and by those alone."""
    root = _tree(tmp_path)
    loaded = _load(root)
    out = tmp_path / "runs"
    # Every script reads a file reference: `measure` an external path,
    # `fit` and `report` upstream files. Their records carry `input_digests`
    # (§8), independently of the conditional layer.
    base = {
        name: BASE_SCRIPT_RECORD_KEYS | {"input_digests"}
        for name in ("measure", "fit", "report")
    }
    first = _run(loaded, root, out)
    for name, keys in base.items():
        record = json.loads((first.run_root / name / SIDECAR).read_text())
        assert set(record) == keys | {"disposition"}, name
        assert record["disposition"] == "accepted"
        assert set(first.manifest["steps"][name]) == keys | {"disposition"}, name
    second = _run(loaded, root, out)
    for name, keys in base.items():
        assert set(second.manifest["steps"][name]) == keys | {
            "disposition",
            "superseded",
        }, name


def test_t14_a_reused_entry_lists_the_same_superseded_units_as_a_fresh_run(
    tmp_path: Path,
) -> None:
    """The skip and the publish path attach the
    step's retained prior units (`superseded`); the reuse path once did not, so a
    `--resume` over the same tree silently dropped them from `workflow.json`.
    Run → rerun (supersedes every published unit) → `--resume`: the reused
    entries list what the rerun's did."""
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root)
    out = tmp_path / "runs"
    _run(loaded, root, out)
    fresh = _run(loaded, root, out)  # supersedes: `.attempts/<step>/0001.superseded`
    again = _run(loaded, root, out, resume=True)
    for name in ("measure", "gate_k", "gate", "fit", "report"):
        assert fresh.manifest["steps"][name]["status"] == "completed", name
        assert again.manifest["steps"][name]["status"] == "reused", name
        assert fresh.manifest["steps"][name]["superseded"] == [
            f"{mf.ATTEMPTS_DIR}/{name}/0001{mf.SUPERSEDED_SUFFIX}"
        ], name
        assert (
            again.manifest["steps"][name]["superseded"]
            == fresh.manifest["steps"][name]["superseded"]
        ), name
    assert "superseded" not in again.manifest["steps"]["probe"]  # never published
    # and the record on disk carries none: `superseded` is the tree's, read
    # at manifest time on every path
    for name in ("measure", "fit"):
        assert "superseded" not in json.loads(
            (out / "gated" / name / SIDECAR).read_text()
        )


def test_moved_evidence_re_evaluates_the_conditional_under_resume(
    tmp_path: Path,
) -> None:
    """The producer re-ran (its values file was gone) and its
    `decision.json` now carries another `evidence_identity`; the conditional's
    own digest and files are unchanged, so without the evidence clause it
    would be reused and gate `fit` on numbers that no longer exist."""
    root = _tree(tmp_path, score=0.9)
    loaded = _load(root)
    out = tmp_path / "runs"
    first = _run(loaded, root, out)
    run_root = first.run_root
    gate_record = json.loads((run_root / "gate" / SIDECAR).read_text())
    (run_root / "measure" / "values.json").unlink()  # the producer must re-run
    _measure(root, 0.1)
    again = _run(loaded, root, out, resume=True)
    assert _statuses(again) == {
        "measure": "completed",  # not reusable: a file is gone
        "gate_k": "completed",  # not reusable: the values behind it moved
        "gate": "completed",  # not reusable: the evidence identity moved
        "fit": "skipped",
        "probe": "completed",
        "report": "skipped",
    }
    new_gate = json.loads((run_root / "gate" / SIDECAR).read_text())
    assert new_gate["verdict"] is False and gate_record["verdict"] is True
    assert (
        new_gate["evidence"]["evidence_identity"]
        != gate_record["evidence"]["evidence_identity"]
    )
    assert new_gate["identity"] == gate_record["identity"]  # the same step
    # the prior verdict and the prior decision are retained, superseded
    assert (
        run_root / mf.ATTEMPTS_DIR / "gate" / f"0001{mf.SUPERSEDED_SUFFIX}" / SIDECAR
    ).is_file()
    assert (
        run_root
        / mf.ATTEMPTS_DIR
        / "gate_k"
        / f"0001{mf.SUPERSEDED_SUFFIX}"
        / DECISION_FILE
    ).is_file()
    # the pure clause: a record whose evidence differs from the tree's does not hold
    step = loaded.document.steps["gate"]
    assert cond.evidence_holds(step, run_root / "gate", new_gate)
    assert not cond.evidence_holds(step, run_root / "gate", gate_record)
