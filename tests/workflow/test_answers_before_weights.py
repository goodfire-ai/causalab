"""A workflow run resolves its metrics' answers before step 1.

The run door of one document ([`run_protocol`][causalab.protocol.pipeline.run_protocol])
resolves every metric's answer columns with the model's tokenizer before
the weights load. A workflow runs many documents, and a defect in the last
one used to surface only when that step scored its first point, after
every earlier step had run and its own weights had loaded. So the runner
resolves the answers of every inner document that compiles at load (a
static one) before its first step, and a document that depends on an
earlier step's output at its own step, before the engine is entered. Under
``--resume`` a reused step never runs, so nothing of it is checked: each
attempted step resolves its answers at its turn instead.

The ``_Recorder`` engine raises on the first document it is handed, and
``_Scores`` completes every step it is handed. Every document names
``gpt2``, whose tokenizer the fixture environment loads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.io.env import FileDatasets, ResolutionEnv
from causalab.io.tables import write_table
from causalab.neural.shared.sweep import signed_steps
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult
from causalab.protocol.lowering import point_count
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import COMPONENTS
from causalab.workflow import runner
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

pytestmark = pytest.mark.unit

#: Two IOI rows. Under the gpt2 tokenizer ``' Tiffany'`` is one token and
#: ``'Tiffany'`` is three, so only the spaced table can be scored.
PROMPTS = (
    "Then, Jennifer and Kevin went to the store. Kevin gave a drink to",
    "Then, Tiffany and Sean went to the store. Sean gave a drink to",
)


def _rows(io: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        {"input": prompt, "io": name, "s": " Kevin", "split": "all"}
        for prompt, name in zip(PROMPTS, io)
    ]


#: A script step that writes the position a later step reads at, so that
#: step's document depends on an earlier step's output.
PICK = """
import json


def main(inputs, outputs):
    outputs["values"].write_text(json.dumps({"pos": {"index": -1}}))
"""


def _document(table: str) -> dict[str, Any]:
    """The un-intervened gpt2 on ``table``'s rows, its logits at the named
    position ``tap`` (the last token) reduced to ``logit_diff`` of the ``io``
    and ``s`` columns."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": f"names/{table}", "field": "input"}},
        "method": {
            "intervened_models": {
                "original_base": {"input": "base", "reads": ["logits"]}
            },
            "positions": {"tap": {"index": -1}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": "tap"}},
            "save": [
                {
                    "read": "logits",
                    "model": "original_base",
                    "aggregation": {"kind": "logit_diff", "a": "io", "b": "s"},
                    "file_path": "ld.json",
                }
            ],
        },
    }


class _Recorder(Engine):
    """Serves every component and verb, and raises on the first document it
    is handed, naming its step's table: reaching a step is what these tests
    look for, so nothing after it needs to be written."""

    name = "recorder"
    capabilities = frozenset(
        {"grad", "paired_forward", "full_logits", "pytorch_fn_local", "generate"}
    )
    components = frozenset(COMPONENTS)
    writable_components = frozenset(COMPONENTS)
    is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        base = compiled.representatives[0].data["base"]
        raise AssertionError(f"engine entered on {getattr(base, 'dataset', base)}")


class _Scores(_Recorder):
    """Completes every step it is handed with one ``ld.json`` row per base
    row, and counts the steps."""

    name = "pytorch_hooks"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        self.calls += 1
        steps = signed_steps(
            compiled, run.env, indices=run.indices(point_count(compiled.axes))
        )
        target = run.output_dir / "ld.json"
        write_table(
            target,
            [{"example_id": str(i), "metric": "ld", "value": 0.5} for i in range(2)],
        )
        return RunResult(
            files={"ld.json": target}, steps=tuple(s.record for s in steps)
        )


def _offline(key: str, revision: str) -> Any:
    """A tokenizer service with nothing cached and no Hub: what transformers
    raises offline."""
    raise OSError(
        "We couldn't connect to 'https://huggingface.co' to load the files, "
        "and couldn't find them in the cached files."
    )


@pytest.fixture()
def workflow_dir(tmp_path: Path) -> Path:
    """``good`` and ``bare`` tables under ``data/names/``, one document over
    each beside the workflow, and the ``pick`` script."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "pick.py").write_text(PICK)
    names = tmp_path / "data" / "names"
    names.mkdir(parents=True)
    (names / "good.json").write_text(json.dumps(_rows((" Jennifer", " Tiffany"))))
    (names / "bare.json").write_text(json.dumps(_rows((" Jennifer", "Tiffany"))))
    for table in ("good", "bare"):
        (tmp_path / f"{table}.json").write_text(json.dumps(_document(table)))
    return tmp_path


def _env(
    env: ResolutionEnv, workflow_dir: Path, tokenizers: Any = None
) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=workflow_dir / "data"),
        artifacts=env.artifacts,
        model_info=env.model_info,
        tokenizers=tokenizers or env.tokenizers,
    )


def _workflow(*documents: str) -> dict[str, Any]:
    """One protocol step per document, each after the one before it."""
    steps: dict[str, Any] = {}
    for index, document in enumerate(documents):
        step: dict[str, Any] = {
            "type": "intervention_protocol",
            "document": f"{document}.json",
        }
        if index:
            step["after"] = [f"step{index - 1}"]
        steps[f"step{index}"] = step
    return {"version": "1", "output_dir": "answers", "steps": steps}


def test_a_later_steps_multi_token_answer_refuses_before_step_one(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """The second step's table splits ``'Tiffany'``; the refusal comes before
    the first step is handed to the engine, and the run writes nothing. On
    the base the first step was handed to the engine, and the second was
    refused only when it scored."""
    names = _env(env, workflow_dir)
    loaded = load_workflow(_workflow("good", "bare"), names, workflow_dir=workflow_dir)
    with pytest.raises(ProtocolError) as err:
        run_workflow(loaded, names, tmp_path / "runs", _Recorder())
    assert err.value.code == "P2", str(err.value)
    assert "'Tiffany'" in str(err.value) and "logit_diff.a" in str(err.value)
    assert "steps.step1" in str(err.value), str(err.value)
    assert not (tmp_path / "runs").exists(), "the refused run wrote a run tree"


def test_the_valid_work_twin_reaches_step_one(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """Both tables spaced: the check passes and the first step is handed to
    the engine."""
    names = _env(env, workflow_dir)
    loaded = load_workflow(_workflow("good", "good"), names, workflow_dir=workflow_dir)
    with pytest.raises(AssertionError, match="engine entered on names/good"):
        run_workflow(loaded, names, tmp_path / "runs", _Recorder())


def _step_dependent(table: str) -> dict[str, Any]:
    """``pick`` writes the position, and ``scan`` over ``table`` reads it
    through its ``set``: a document that depends on an earlier step."""
    return {
        "version": "1",
        "output_dir": "answers",
        "steps": {
            "pick": {
                "type": "script",
                "script": {"path": "scripts/pick.py"},
                "inputs": {"seed": {"value": 1}},
                "outputs": {
                    "values": {"file": "values.json", "keys": {"pos": {"index": -1}}}
                },
            },
            "scan": {
                "type": "intervention_protocol",
                "document": f"{table}.json",
                "set": {"positions.tap": {"artifact": "pick", "key": "pos"}},
            },
        },
    }


def test_a_step_dependent_document_refuses_at_its_step_before_the_engine(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """A document whose ``set`` reads an earlier step's output compiles at
    load against a deferring store, so the check before step 1 leaves it
    out. It is resolved again at its own step, after ``pick`` ran and
    before the engine is handed the document. On the base the engine was
    handed it."""
    names = _env(env, workflow_dir)
    loaded = load_workflow(_step_dependent("bare"), names, workflow_dir=workflow_dir)
    assert loaded.inner_digest_kind["scan"] == "authored"  # deferred at load
    with pytest.raises(ProtocolError) as err:
        run_workflow(loaded, names, tmp_path / "runs", _Recorder())
    assert err.value.code == "P2", str(err.value)
    assert "'Tiffany'" in str(err.value) and "logit_diff.a" in str(err.value)
    run_root = tmp_path / "runs" / "answers"
    assert (run_root / "pick" / "values.json").is_file(), "pick did not run first"
    manifest = json.loads((run_root / "workflow.json").read_text())
    assert manifest["steps"]["scan"]["status"] == "failed"


def test_the_step_dependent_twin_reaches_the_engine(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """The same step-dependent document over the spaced table resolves at
    its step, and the engine is handed it."""
    names = _env(env, workflow_dir)
    loaded = load_workflow(_step_dependent("good"), names, workflow_dir=workflow_dir)
    with pytest.raises(AssertionError, match="engine entered on names/good"):
        run_workflow(loaded, names, tmp_path / "runs", _Recorder())


def test_steps_that_share_a_document_resolve_it_once(
    env: ResolutionEnv,
    workflow_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two steps run ``good.json``: its answers resolve once before step 1."""
    calls: list[CompiledProtocol] = []
    resolve = runner.resolve_answers

    def counted(compiled: CompiledProtocol, **kwargs: Any) -> None:
        calls.append(compiled)
        resolve(compiled, **kwargs)

    monkeypatch.setattr(runner, "resolve_answers", counted)
    names = _env(env, workflow_dir)
    loaded = load_workflow(_workflow("good", "good"), names, workflow_dir=workflow_dir)
    with pytest.raises(AssertionError, match="engine entered on names/good"):
        run_workflow(loaded, names, tmp_path / "runs", _Recorder())
    assert len(calls) == 1


def test_a_run_loads_each_tokenizer_once(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """A static step and a step-dependent one on one model: the check before
    step 1 and the step-dependent step's check at its turn share one load."""
    loads: list[tuple[str, str]] = []
    loader = env.tokenizers
    assert loader is not None

    def counted(key: str, revision: str) -> Any:
        loads.append((key, revision))
        return loader(key, revision)

    names = _env(env, workflow_dir, counted)
    workflow = _step_dependent("good")
    workflow["steps"]["first"] = {
        "type": "intervention_protocol",
        "document": "good.json",
    }
    loaded = load_workflow(workflow, names, workflow_dir=workflow_dir)
    engine = _Scores()
    run_workflow(loaded, names, tmp_path / "runs", engine)
    assert engine.calls == 2
    assert loads == [("gpt2", "main")]


def test_an_unloadable_tokenizer_refuses_the_run_before_step_one(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """Offline with nothing cached, the check before step 1 refuses ``[P4]``
    naming the key and revision, and the run writes nothing."""
    names = _env(env, workflow_dir, _offline)
    loaded = load_workflow(_workflow("good"), names, workflow_dir=workflow_dir)
    with pytest.raises(ProtocolError) as err:
        run_workflow(loaded, names, tmp_path / "runs", _Recorder())
    assert err.value.code == "P4" and err.value.path == "steps.step0", str(err.value)
    assert "the tokenizer of gpt2@main could not be loaded" in str(err.value)
    assert str(err.value).count("[P4]") == 1, str(err.value)
    assert not (tmp_path / "runs").exists(), "the refused run wrote a run tree"


def test_resume_reuses_a_completed_step_without_its_tokenizer(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """A completed run tree, resumed where the tokenizer cannot load: the
    step is reused and the engine is not handed it again, as before the
    answers were checked. A reused step never runs, so its answers need no
    tokenizer."""
    names = _env(env, workflow_dir)
    engine = _Scores()
    loaded = load_workflow(_workflow("good"), names, workflow_dir=workflow_dir)
    run_workflow(loaded, names, tmp_path / "runs", engine)
    offline = _env(env, workflow_dir, _offline)
    again = load_workflow(_workflow("good"), offline, workflow_dir=workflow_dir)
    result = run_workflow(again, offline, tmp_path / "runs", engine, resume=True)
    assert result.manifest["steps"]["step0"]["status"] == "reused"
    assert engine.calls == 1


def test_resume_resolves_an_attempted_steps_answers_at_its_turn(
    env: ResolutionEnv, workflow_dir: Path, tmp_path: Path
) -> None:
    """The twin: under ``--resume`` a step that is not reused (its document
    changed to the split table) resolves its answers at its turn, and is
    refused there before the engine is handed it."""
    names = _env(env, workflow_dir)
    engine = _Scores()
    loaded = load_workflow(_workflow("good", "good"), names, workflow_dir=workflow_dir)
    run_workflow(loaded, names, tmp_path / "runs", engine)
    changed = load_workflow(_workflow("good", "bare"), names, workflow_dir=workflow_dir)
    with pytest.raises(ProtocolError) as err:
        run_workflow(changed, names, tmp_path / "runs", engine, resume=True)
    assert err.value.code == "P2" and "steps.step1" in str(err.value), str(err.value)
    assert engine.calls == 2  # the first run's two steps; nothing on resume
    manifest = json.loads((tmp_path / "runs" / "answers" / "workflow.json").read_text())
    assert manifest["steps"]["step0"]["status"] == "reused"
    assert manifest["steps"]["step1"]["status"] == "failed"


def test_a_fan_out_join_is_left_out_and_its_children_are_checked(
    env: ResolutionEnv, workflow_dir: Path
) -> None:
    """A fan-out parent is its children's join and hands no engine the
    document, so the check names the children and not the parent: a join
    re-attempted under ``--resume`` needs no tokenizer."""
    doc = _document("good")
    doc["method"]["sites"]["resid"] = {
        "component": "block_output",
        "layers": {"sweep": [1, 2]},
    }
    doc["method"]["reads"]["h"] = {"site": "resid", "pos": "tap"}
    doc["method"]["intervened_models"]["original_base"]["reads"].append("h")
    doc["method"]["save"].append(
        {
            "read": "h",
            "model": "original_base",
            "aggregation": {"kind": "top_k", "k": 1, "by": "value"},
            "file_path": "h.json",
        }
    )
    (workflow_dir / "swept.json").write_text(json.dumps(doc))
    workflow = {
        "version": "1",
        "output_dir": "answers",
        "steps": {
            "scan": {
                "type": "intervention_protocol",
                "document": "swept.json",
                "fan_out": {
                    "over": {"axis": "sites.resid.layers"},
                    "join": {"require": "all"},
                },
            }
        },
    }
    names = _env(env, workflow_dir)
    loaded = load_workflow(workflow, names, workflow_dir=workflow_dir)
    assert tuple(runner.check_tokenization(loaded, names)) == ("scan@0", "scan@1")
