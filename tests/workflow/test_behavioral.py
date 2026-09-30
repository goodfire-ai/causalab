"""The behavioral runner's declarative half (workflow spec §2.7, §5 rule 17;
T8, T9 and the censuses) — everything decidable without a model.

* **The censuses.** The step type is in the closed set; rule 17 is the
  behavioral rule and §5 numbers it (15 and 16 reserved, 18 the conditional
  layer's, so `MAX_RULE` is 18); the four closed vocabularies —
  terminal outcomes, decoding modes, split purposes, decision types — are
  §2.7's tables, member for member and in order (the outcome order is the
  derivation's precedence).
* **The closure guard.** ``causalab/workflow/behavioral.py`` is a member of
  no hashed script's closure and of ``SHARED``, and loading a behavioral
  workflow imports no torch — the design is digest-neutral because nothing
  digest-bearing changed.
* **Refusals, each beside its valid twin**, so valid work is not refused: a missing
  or mismatched checker (T5's mutation, at load), a free-string split, a
  split that is not the base ref's fragment, a sampled decode without a seed,
  temperature and top_p out of range, non-numeric thresholds, an unknown
  decision, an unbounded retention not spelled ``"all"``, a document that
  never decodes — every one rule 17, naming the field.
* **The derivation** (T6's pure half): the four outcomes from their premises,
  ``truncated`` taking precedence.
* **T8.** ``split`` and the decode seed are in the step identity, so a
  development record is never reused for a confirmation step and a new
  decision record is written.
* **T9.** The legitimate campaign: every shipped and demo workflow loads with
  no behavioral key on any entry, both pins hold byte for byte against the
  pin *file*, and the population is non-empty.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.identity import import_closure
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import CONTINUATIONS_FILE, Engine, RunContext, RunResult
from causalab.protocol.registry import ENGINES
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import COMPONENTS
from causalab.workflow import runner
from causalab.workflow.behavioral import (
    BEHAVIORAL_FILES,
    BEHAVIORAL_RULE,
    DECISION_FILE,
    DECISION_TYPES,
    DECODING_MODES,
    OUTCOMES,
    OUTCOMES_FILE,
    SAMPLING_ENGINES,
    SPLIT_PURPOSES,
    derive_outcome,
)
from causalab.workflow.document import (
    MAX_RULE,
    STEP_TYPES,
    BehavioralStep,
    LoadedWorkflow,
    WorkflowError,
    load_workflow,
)
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
FIXTURES = Path(__file__).parent / "fixtures" / "behavioral"
WORKFLOW = FIXTURES / "qualify.json"
BEHAVIORAL = "causalab/workflow/behavioral.py"
STEP = "qualify"
#: every key a behavioral entry carries and no other entry may (§7)
BEHAVIORAL_KEYS = ("decoding", "checker", "split", "thresholds", "retain", "decision")


def _env(root: Path = FIXTURES) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=REPO)
    )


#: The fixture's `set` without the tiny-scale model retarget: the document
#: names gpt2, a registered model, so the unit half loads it torch-free; the
#: run half (`test_behavioral_run.py`) keeps the retarget to tiny Llama.
DATA_ONLY = {"data.base.dataset": "qa/data#development"}


def _raw(**step_changes: Any) -> dict[str, Any]:
    """The fixture workflow with ``step_changes`` on its one step; ``None``
    deletes the key. The model retarget is dropped (`DATA_ONLY`)."""
    raw = json.loads(WORKFLOW.read_text())
    step = raw["steps"][STEP]
    step["set"] = dict(DATA_ONLY)
    for key, value in step_changes.items():
        if value is None:
            step.pop(key, None)
        else:
            step[key] = value
    return raw


def _load(raw: dict[str, Any], root: Path = FIXTURES) -> LoadedWorkflow:
    return load_workflow(raw, _env(root), workflow_dir=root)


def _tree(tmp: Path, raw: dict[str, Any] | None = None) -> Path:
    """A private copy of the fixture tree, its workflow replaced by ``raw``
    (the retarget-free fixture by default)."""
    root = tmp / "behavioral"
    shutil.copytree(FIXTURES, root)
    (root / "qualify.json").write_text(json.dumps(raw if raw is not None else _raw()))
    return root


def _refused(raw: dict[str, Any], *, path: str, root: Path = FIXTURES) -> WorkflowError:
    with pytest.raises(WorkflowError) as err:
        _load(raw, root)
    assert err.value.rule == BEHAVIORAL_RULE, str(err.value)
    assert path in str(err.value), str(err.value)
    return err.value


def _table(header: str) -> list[str]:
    """§2.7's table whose header row starts with the plain word ``header``:
    the backticked member in each body row, in order."""
    rows = _rows(_section("### 2.7 `behavioral` steps"))
    start = next(index for index, row in enumerate(rows) if row[0] == header)
    members: list[str] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        members.append(CODE.findall(row[0])[0])
    return members


# --------------------------------------------------------------------------- #
# the censuses
# --------------------------------------------------------------------------- #


def test_behavioral_is_the_third_step_type() -> None:
    """Six kinds: the two control kinds follow (§2.8), then the
    nested workflow (§2.10)."""
    assert STEP_TYPES == (
        "intervention_protocol",
        "script",
        "behavioral",
        "decision",
        "conditional",
        "workflow",
    )
    assert STEP_TYPES[2] == "behavioral"
    assert (
        "`intervention_protocol · script · behavioral · decision · conditional · workflow`"
        in _section("### 2.1 `steps` — common fields")
    )


def test_rule_17_is_the_behavioral_rule_after_qualification_and_equivalence() -> None:
    """Numbered by rule, so a renumbering is a deliberate edit here: 15 is
    the qualify-once rule, 16 site equivalence, 17 this step's;
    18 is the decision / conditional layer's (§2.8), 19 the declared
    fan-out's (§2.9), so 17 is no longer the last."""
    assert BEHAVIORAL_RULE == 17 and MAX_RULE >= 17
    section = _section("## 5. Validation")
    items = {
        int(number): text
        for number, text in __import__("re").findall(
            r"^(\d+)\. (.+?)(?=^\d+\. |\Z)",
            section,
            __import__("re").M | __import__("re").S,
        )
    }
    assert "qualif" in items[15].lower() and "Reserved" not in items[15]
    assert "equivalen" in items[16].lower() and "Reserved" not in items[16]
    for word in (
        "behavioral",
        "`checker`",
        "`split`",
        "`decoding`",
        "`seed`",
        "`thresholds`",
    ):
        assert word in items[17], word


def test_the_outcome_table_is_the_code_in_precedence_order() -> None:
    assert _table("outcome") == list(OUTCOMES)
    assert OUTCOMES[0] == "truncated"  # takes precedence (§2.7, q6)


def test_the_decoding_split_and_decision_tables_are_the_code() -> None:
    assert _table("mode") == list(DECODING_MODES)
    assert _table("purpose") == list(SPLIT_PURPOSES)
    assert _table("decision") == list(DECISION_TYPES)


def test_the_field_table_is_the_step_grammar() -> None:
    assert _table("field") == [
        "document",
        "set",
        "max_points",
        *BEHAVIORAL_KEYS,
        "fan_out",  # §2.9: a document-step field, on the entry only when authored
    ]


def test_the_sampling_engines_are_registry_engines() -> None:
    assert set(SAMPLING_ENGINES) <= set(ENGINES) and "nnsight" not in SAMPLING_ENGINES


def test_the_three_files_are_named_once() -> None:
    assert BEHAVIORAL_FILES == (CONTINUATIONS_FILE, OUTCOMES_FILE, DECISION_FILE)
    assert len(set(BEHAVIORAL_FILES)) == 3


# --------------------------------------------------------------------------- #
# the closure guard, and torch-free
# --------------------------------------------------------------------------- #


def test_behavioral_is_in_no_hashed_closure() -> None:
    """Every hashed script's frozen closure, and the reduce script's, and the
    SHARED tuple: none names behavioral.py — so nothing digest-bearing
    changed, which is why both pins hold (T9)."""
    assert BEHAVIORAL not in SHARED
    for module in (*CLOSURES, REDUCE):
        assert BEHAVIORAL not in _closure(module), module


def test_behavioral_reaches_no_engine_module() -> None:
    """Engine-free like document.py: its own import closure stays under
    protocol/, causal/, tasks/ and workflow/ — the one member under neural/
    is the torch-free enumerator the workflow layer reads the steps through
    (``neural/shared/sweep.py``), never an engine."""
    members = import_closure(REPO / BEHAVIORAL, root=REPO)
    assert BEHAVIORAL not in members  # a closure lists what a module reaches
    assert members, "the closure walk found nothing"
    assert [m for m in members if m.startswith("causalab/neural/")] == [
        "causalab/neural/shared/sweep.py"
    ]


_PROBE = """
import json, sys
from pathlib import Path
from causalab.io.sources import load_text
from causalab.workflow.document import parse_workflow
import causalab.workflow.behavioral
root = Path(sys.argv[1])
parsed = parse_workflow(load_text(root / "qualify.json"))
after_parse = "torch" in sys.modules
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.workflow.document import load_workflow
env = ResolutionEnv(datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root))
loaded = load_workflow(root / "qualify.json", env)
print(json.dumps({
    "digest": loaded.digest,
    "torch_after_parse": after_parse,
    "torch_after_load": "torch" in sys.modules,
    "task_package": "causalab.tasks.natural_domains_arithmetic" in sys.modules,
}))
"""


def test_the_behavioral_module_and_parse_import_no_torch(tmp_path: Path) -> None:
    """A subprocess, as in test_load_is_torch_free.py: conftest has already
    imported torch in this process. Importing behavioral.py and parsing a
    behavioral workflow (rule 17's grammar half) import no torch. The load's
    checker binding imports the task package, and every task package's
    ``__init__`` reaches ``causalab.neural`` through its token-positions
    module today — so torch arrives with the task, not with this layer; the
    probe records where it came from."""
    root = _tree(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(root)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["torch_after_parse"] is False
    assert result["task_package"] is True  # the binding loaded the task
    assert result["digest"] == _load(_raw()).digest


# --------------------------------------------------------------------------- #
# valid work: the fixture loads, and its canonical entry is the grammar
# --------------------------------------------------------------------------- #


def test_the_fixture_workflow_loads_with_every_field_in_its_entry() -> None:
    loaded = _load(_raw())
    step = loaded.document.steps[STEP]
    assert isinstance(step, BehavioralStep)
    entry = loaded.canonical["steps"][STEP]
    assert set(entry) == {
        "type",
        "document",
        "set",
        "document_digest",
        *BEHAVIORAL_KEYS,
    }
    assert entry["type"] == "behavioral"
    assert entry["decoding"] == {"mode": "deterministic"}
    assert entry["checker"] == {
        "task": "natural_domains_arithmetic",
        "task_cfg": {"domain_type": "weekdays"},
    }
    assert entry["split"] == "development"
    assert entry["retain"] == {"generations": {"max_rows": 2}}
    assert entry["decision"] == {"on_pass": "advance", "on_fail": "narrow"}
    assert "closure" not in entry and "script_sha256" not in entry
    # the step has its own identity, as a script step does (§7)
    assert loaded.step_digests[STEP] != loaded.inner_digests[STEP]
    assert len(loaded.step_digests[STEP]) == 64
    # the inner document is loaded like a protocol step's
    assert STEP in loaded.inner and loaded.inner_digest_kind[STEP] == "campaign"


def test_a_sampled_decode_materializes_its_defaults() -> None:
    """`temperature` and `top_p` default to 1.0 and are in the entry: the
    canonical decode spec is complete, so two spellings of one draw digest
    alike and a changed default is a changed identity."""
    loaded = _load(_raw(decoding={"mode": "sampled", "seed": 7}))
    assert loaded.canonical["steps"][STEP]["decoding"] == {
        "mode": "sampled",
        "seed": 7,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    explicit = _load(
        _raw(decoding={"mode": "sampled", "seed": 7, "temperature": 1.0, "top_p": 1.0})
    )
    assert explicit.digest == loaded.digest


def test_retain_is_optional_and_all_is_spelled() -> None:
    absent = _load(_raw(retain=None))
    assert "retain" not in absent.canonical["steps"][STEP]
    everything = _load(_raw(retain={"generations": "all"}))
    assert everything.canonical["steps"][STEP]["retain"] == {"generations": "all"}


def test_explain_names_the_behavioral_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from causalab.cli import main

    root = _tree(tmp_path)
    code = main(
        [
            "explain",
            "--engine",
            "auto",
            str(root / "qualify.json"),
            "--data-root",
            str(root),
            "--artifacts-root",
            str(REPO),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"{STEP}: behavioral protocols/qa_probe.json" in out
    assert "decoding deterministic, split development" in out


# --------------------------------------------------------------------------- #
# rule 17 — refusals, each with its twin
# --------------------------------------------------------------------------- #


def test_a_missing_checker_is_refused_at_load() -> None:
    """T5's mutation: delete the binding and the workflow does not load —
    long before any generation. Its twin is the fixture itself."""
    err = _refused(_raw(checker=None), path="checker")
    assert "ScoringSpec" in str(err)


def test_a_table_recording_another_string_mode_is_refused(tmp_path: Path) -> None:
    """The checker's task loads, but the split's rows record a ``string_mode``
    that is not the spec's — the table was built under another definition of
    correct: refused naming both modes."""
    root = tmp_path / "behavioral"
    shutil.copytree(FIXTURES, root)
    table = root / "qa" / "data.json"
    rows = json.loads(table.read_text())
    for row in rows:
        row["string_mode"] = "prefix"
    table.write_text(json.dumps(rows))
    err = _refused(_raw(), path="checker.task", root=root)
    assert "'prefix'" in str(err) and "'exact'" in str(err) and "table" in str(err)


def test_an_unknown_task_and_a_config_on_a_singleton_are_refused() -> None:
    _refused(_raw(checker={"task": "no_such_task"}), path="checker.task")
    # a factory task without its config does not load either
    _refused(_raw(checker={"task": "natural_domains_arithmetic"}), path="checker.task")
    # IOI is a singleton: it takes no task_cfg
    _refused(
        _raw(checker={"task": "IOI", "task_cfg": {"x": 1}}),
        path="checker.task_cfg",
    )


def test_a_malformed_checker_is_refused() -> None:
    _refused(_raw(checker="weekdays"), path="checker")
    _refused(
        _raw(checker={"task_cfg": {"domain_type": "weekdays"}}), path="checker.task"
    )
    _refused(_raw(checker={"task": ""}), path="checker.task")
    _refused(_raw(checker={"task": "IOI", "mode": "exact"}), path="checker")
    # the binding is by task: a singleton whose spec shares the table's
    # string_mode loads, and the entry records exactly what was authored
    loaded = _load(_raw(checker={"task": "IOI"}))
    assert loaded.canonical["steps"][STEP]["checker"] == {"task": "IOI"}


def test_a_split_is_one_of_three_values_not_a_free_string() -> None:
    """T8's vocabulary half (the mutation: a free-string split)."""
    err = _refused(_raw(split="train"), path="split")
    assert "development" in str(err)
    _refused(_raw(split="Development"), path="split")
    _refused(_raw(split=None), path="split")
    assert _load(_raw()).document.steps[STEP].split == "development"  # type: ignore[union-attr]


def test_a_split_outside_the_vocabulary_is_refused_even_when_the_table_has_it(
    tmp_path: Path,
) -> None:
    """The vocabulary check is its own rule, not a side effect of the
    fragment check: a table whose rows are split `train`, a ref selecting
    `#train` and a step saying `train` agree with each other perfectly — and
    are refused, because `train` is not a purpose (the mutation: accept any
    string as a split)."""
    root = tmp_path / "behavioral"
    shutil.copytree(FIXTURES, root)
    table = root / "qa" / "data.json"
    rows = json.loads(table.read_text())
    for row in rows:
        if row["split"] == "reserve":
            row["split"] = "train"
    table.write_text(json.dumps(rows))
    err = _refused(
        _raw(set={"data.base.dataset": "qa/data#train"}, split="train"),
        path="split",
        root=root,
    )
    assert "is not one of" in str(err) and "reserve" in str(err)


def test_a_split_must_be_the_base_refs_fragment() -> None:
    """A `development` step over a `#confirmation` ref is refused; the twin —
    `confirmation` over `#confirmation` — loads."""
    confirmation = {"data.base.dataset": "qa/data#confirmation"}
    err = _refused(_raw(set=confirmation), path="split")
    assert "confirmation" in str(err) and "development" in str(err)
    twin = _load(_raw(set=confirmation, split="confirmation"))
    assert twin.canonical["steps"][STEP]["split"] == "confirmation"


def test_a_decoding_spec_is_complete_and_in_range() -> None:
    err = _refused(_raw(decoding={"mode": "sampled"}), path="decoding.seed")
    assert "seed" in str(err)
    _refused(_raw(decoding={"mode": "sampled", "seed": -1}), path="decoding.seed")
    _refused(_raw(decoding={"mode": "sampled", "seed": True}), path="decoding.seed")
    _refused(
        _raw(decoding={"mode": "sampled", "seed": 7, "temperature": 0}),
        path="decoding.temperature",
    )
    _refused(
        _raw(decoding={"mode": "sampled", "seed": 7, "top_p": 0}), path="decoding.top_p"
    )
    _refused(
        _raw(decoding={"mode": "sampled", "seed": 7, "top_p": 1.5}),
        path="decoding.top_p",
    )
    err = _refused(_raw(decoding={"mode": "greedy"}), path="decoding.mode")
    assert "deterministic" in str(err)
    _refused(_raw(decoding={"mode": "deterministic", "seed": 7}), path="decoding")
    _refused(_raw(decoding={}), path="decoding")
    _refused(_raw(decoding=None), path="decoding")
    # the twin: a sampled decode with a seed, and a bare deterministic one
    _load(
        _raw(decoding={"mode": "sampled", "seed": 7, "temperature": 0.7, "top_p": 0.9})
    )
    _load(_raw(decoding={"mode": "deterministic"}))


def test_thresholds_are_declared_numbers() -> None:
    good = {"min_examples": 4, "min_valid_rate": 0.5, "min_correct_rate": 0.25}
    _refused(
        _raw(thresholds={**good, "min_valid_rate": "0.5"}),
        path="thresholds.min_valid_rate",
    )
    _refused(
        _raw(thresholds={**good, "min_correct_rate": 1.5}),
        path="thresholds.min_correct_rate",
    )
    _refused(
        _raw(thresholds={**good, "min_examples": 0}), path="thresholds.min_examples"
    )
    _refused(
        _raw(thresholds={**good, "min_examples": 2.5}), path="thresholds.min_examples"
    )
    _refused(_raw(thresholds={"min_examples": 4}), path="thresholds")
    _refused(_raw(thresholds={**good, "mean": 0.5}), path="thresholds")
    _refused(_raw(thresholds=None), path="thresholds")
    assert _load(_raw(thresholds=good)).canonical["steps"][STEP]["thresholds"] == good


def test_a_decision_is_typed() -> None:
    _refused(
        _raw(decision={"on_pass": "proceed", "on_fail": "narrow"}),
        path="decision.on_pass",
    )
    _refused(_raw(decision={"on_pass": "advance"}), path="decision")
    _refused(
        _raw(decision={"on_pass": "advance", "on_fail": "narrow", "else": "x"}),
        path="decision",
    )
    _refused(_raw(decision=None), path="decision")
    for word in DECISION_TYPES:
        _load(_raw(decision={"on_pass": word, "on_fail": word}))


def test_retention_is_bounded_unless_all_is_spelled() -> None:
    _refused(_raw(retain={"generations": {"max_rows": 0}}), path="retain.generations")
    _refused(_raw(retain={"generations": "everything"}), path="retain.generations")
    _refused(_raw(retain={"generations": {"rows": 3}}), path="retain.generations")
    _refused(_raw(retain={"max_rows": 3}), path="retain")
    _refused(_raw(retain="all"), path="retain")


def test_a_document_that_never_decodes_is_refused(tmp_path: Path) -> None:
    """A behavioral step runs a document under the `generated` frame; one
    that reads the prompt frame only has no continuation to judge."""
    root = tmp_path / "behavioral"
    shutil.copytree(FIXTURES, root)
    doc_path = root / "protocols" / "qa_probe.json"
    doc = json.loads(doc_path.read_text())
    doc["method"]["positions"] = {"last": {"index": -1}}
    doc["method"]["reads"] = {"steps": {"site": "lm_head", "pos": "last"}}
    doc["method"]["intervened_models"] = {
        "original": {"input": "base", "reads": ["steps"]}
    }
    doc["method"]["save"] = [
        {
            "read": "steps",
            "model": "original",
            "aggregation": {"kind": "top_k", "k": 1, "by": "prob"},
            "file_path": "per_step.json",
        }
    ]
    doc_path.write_text(json.dumps(doc))
    err = _refused(_raw(), path=f"steps.{STEP}", root=root)
    assert "generated" in str(err)


def test_strict_keys_still_hold_on_a_behavioral_step() -> None:
    with pytest.raises(WorkflowError) as err:
        _load(_raw(inputs={"x": 1}))
    assert err.value.rule == 1 and "inputs" in str(err.value)


# --------------------------------------------------------------------------- #
# the derivation (T6's pure half)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("premises", "outcome"),
    [
        (dict(width=4, steps=4, anchor_found=True, declared=True), "truncated"),
        (dict(width=4, steps=4, anchor_found=False, declared=False), "truncated"),
        (dict(width=0, steps=4, anchor_found=False, declared=False), "invalid_format"),
        (dict(width=2, steps=4, anchor_found=False, declared=False), "no_final_answer"),
        (dict(width=2, steps=4, anchor_found=True, declared=False), "invalid_format"),
        (dict(width=1, steps=4, anchor_found=True, declared=True), "valid"),
    ],
    ids=["truncated", "truncated-wins", "empty", "no-answer", "malformed", "valid"],
)
def test_derive_outcome(premises: dict[str, Any], outcome: str) -> None:
    """Each outcome from its premises; a row that ran the whole budget is
    `truncated` whatever it said (the mutation collapsing it into
    `invalid_format` fails the first two cases)."""
    assert derive_outcome(**premises) == outcome
    assert outcome in OUTCOMES


# --------------------------------------------------------------------------- #
# T8 — split and seed are identity
# --------------------------------------------------------------------------- #


def _confirmation() -> dict[str, Any]:
    return _raw(set={"data.base.dataset": "qa/data#confirmation"}, split="confirmation")


def test_t8_a_development_record_is_never_reused_for_confirmation(
    tmp_path: Path,
) -> None:
    """The step identity `--resume` compares is the canonical entry's digest,
    and `split` is in it: a published development unit, however complete,
    does not satisfy the confirmation step — which runs again and writes a
    new decision record."""
    development = _load(_raw())
    confirmation = _load(_confirmation())
    dev_step = development.document.steps[STEP]
    conf_step = confirmation.document.steps[STEP]
    dev_identity = runner._step_identity(development, STEP, dev_step)  # pyright: ignore[reportPrivateUsage]
    conf_identity = runner._step_identity(confirmation, STEP, conf_step)  # pyright: ignore[reportPrivateUsage]
    assert dev_identity == development.step_digests[STEP]
    assert dev_identity != conf_identity
    # the same inner document (the rows differ, so its digest does too) is not
    # what the reuse decision reads
    assert dev_identity != development.inner_digests[STEP]
    # a published development unit on disk, complete and consistent
    step_dir = tmp_path / STEP
    step_dir.mkdir()
    implementation = {"tree_digest": "t" * 64}
    (step_dir / "_step.json").write_text(
        json.dumps(
            {
                "type": "behavioral",
                "status": "completed",
                "identity": dev_identity,
                "implementation": implementation,
                "files": [],
                "digests": {},
                "split": "development",
            }
        )
    )
    reused = runner._reusable(  # pyright: ignore[reportPrivateUsage]
        development, STEP, dev_step, step_dir, True, False, implementation, None
    )
    assert reused is not None and reused["status"] == "reused"
    assert (
        runner._reusable(  # pyright: ignore[reportPrivateUsage]
            confirmation, STEP, conf_step, step_dir, True, False, implementation, None
        )
        is None
    )


def test_the_seed_is_identity_too() -> None:
    """T7's `--resume` corollary: seed 7 and seed 8 are two steps."""
    seven = _load(_raw(decoding={"mode": "sampled", "seed": 7}))
    eight = _load(_raw(decoding={"mode": "sampled", "seed": 8}))
    plain = _load(_raw())
    assert (
        len(
            {
                seven.step_digests[STEP],
                eight.step_digests[STEP],
                plain.step_digests[STEP],
            }
        )
        == 3
    )
    # while the inner document — the deterministic half — is one and the same
    assert (
        seven.inner_digests[STEP]
        == eight.inner_digests[STEP]
        == plain.inner_digests[STEP]
    )


# --------------------------------------------------------------------------- #
# the run-time refusal that needs no model: sampling on the nnsight engine
# --------------------------------------------------------------------------- #


class _Greedy(Engine):
    """An engine named like the nnsight one, able to serve the fixture
    document, whose `execute` raises: reaching it means a model would have
    loaded before the refusal."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.capabilities = frozenset(
            {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
        )
        self.components = frozenset(COMPONENTS)
        self.writable_components = frozenset(COMPONENTS)
        self.is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        raise AssertionError(f"{self.name} executed — the refusal came after routing")


def test_a_sampled_step_routed_to_a_greedy_engine_is_refused_before_it_loads(
    tmp_path: Path,
) -> None:
    from causalab.workflow import run_workflow

    loaded = _load(_raw(decoding={"mode": "sampled", "seed": 7}))
    with pytest.raises(WorkflowError) as err:
        run_workflow(loaded, _env(), tmp_path, _Greedy("nnsight"))
    assert err.value.rule == BEHAVIORAL_RULE
    assert "nnsight" in str(err.value) and "deterministic" in str(err.value)
    # the twin: a deterministic step reaches the engine (whose execute raises)
    loaded = _load(_raw(decoding={"mode": "deterministic"}))
    with pytest.raises(AssertionError, match="executed"):
        run_workflow(loaded, _env(), tmp_path / "twin", _Greedy("nnsight"))


def test_a_decoding_step_under_context_parallelism_is_refused_before_it_loads(
    tmp_path: Path,
) -> None:
    """``check_parallel`` (``docs/model_parallelism.md`` §8.4) at the workflow
    door, as ``run_protocol`` runs it: every rank refuses the decoding step
    alike, before a model loads, so no rank waits on a collective the others
    never enter."""
    from causalab.protocol.parallel import ParallelGeometry
    from causalab.protocol.rules.errors import ProtocolError
    from causalab.workflow import run_workflow

    loaded = _load(_raw(decoding={"mode": "deterministic"}))
    engine = _Greedy("nnsight")
    engine.parallel = ParallelGeometry(context=2)
    with pytest.raises(ProtocolError) as err:
        run_workflow(loaded, _env(), tmp_path, engine)
    assert err.value.code == "P4" and "decod" in str(err.value)


# --------------------------------------------------------------------------- #
# T9 — the legitimate population
# --------------------------------------------------------------------------- #


def test_t9_the_population_is_non_empty() -> None:
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
def test_t9_every_existing_workflow_loads_with_no_behavioral_key(
    path: Path, env: Any
) -> None:
    """A behavioral key on any other entry would move every workflow digest
    in the repo (the mutation: emit `decoding` unconditionally)."""
    loaded = load_workflow(path, env if path.parent == WORKFLOWS else _demo_env(path))
    for name, entry in loaded.canonical["steps"].items():
        assert entry["type"] in ("intervention_protocol", "script"), name
        assert not set(entry) & set(BEHAVIORAL_KEYS), (name, sorted(entry))


def test_the_shipped_workflows_carry_no_behavioral_keys(env: Any) -> None:
    for name in ("mean_ablation.json", "weekdays.json"):
        loaded = load_workflow(WORKFLOWS / name, env)
        for step, entry in loaded.canonical["steps"].items():
            assert not set(entry) & set(BEHAVIORAL_KEYS), (name, step)


def test_explicit_eos_contract_is_validated_and_enters_identity():
    first = _load(_raw(decoding={"mode": "deterministic", "eos_token_ids": [1, 2]}))
    second = _load(_raw(decoding={"mode": "deterministic", "eos_token_ids": [1]}))
    assert first.step_digests[STEP] != second.step_digests[STEP]
    for ids in ([], [True], [-1], [1, 1], "1"):
        _refused(
            _raw(decoding={"mode": "deterministic", "eos_token_ids": ids}),
            path="decoding",
        )
