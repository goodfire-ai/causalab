"""A fixed benchmark executes two committed implementations, including pinned code."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from causalab.cli import main
from causalab.measurement.deployment.installation import archive_arm
from causalab.measurement.deployment.remote import _freeze_study
from causalab.measurement.deployment.source_pins import check_source_pins
from causalab.measurement.study.controller import run
from tests.measurement.study.test_controller import _local_inputs

# `measurement_study`: every test here builds and installs a source wheel and
# spawns cold worker processes — minutes each. `-m "not measurement_study"`
# deselects them for a quick run (docs/TESTS.md lists the markers).
pytestmark = [pytest.mark.smoke, pytest.mark.measurement_study]

_SCRIPT_MODULE = "causalab.analysis.measurement_test_score"
_SCRIPT_PATH = "causalab/analysis/measurement_test_score.py"
_LOADING_PATH = "causalab/neural/engines/pytorch_hooks/loading.py"
_LOGITS_KEY = 'logits/{"coords":{},"slot":"logits"}'


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", _SCRIPT_PATH, _LOADING_PATH)
    _git(
        repository,
        "-c",
        "user.name=Measurement test",
        "-c",
        "user.email=measurement-test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def _script(value: int) -> str:
    return (
        "import json\nfrom pathlib import Path\n\n"
        "def main(inputs, outputs):\n"
        f"    rows = [{{'id': 'shared', 'score': {value}}}]\n"
        "    Path(outputs['out']).write_text(json.dumps(rows))\n"
    )


def _committed_arms(root: Path) -> tuple[Path, Path, str, str]:
    """Only this disposable Git repository receives the test implementation changes."""
    source = Path(__file__).resolve().parents[3]
    repository = root / "source-repository"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(repository)],
        check=True,
    )
    (repository / _SCRIPT_PATH).write_text(_script(1))
    before = _commit(repository, "test: baseline score implementation")
    baseline = root / "baseline-checkout"
    _git(repository, "worktree", "add", "--quiet", "--detach", str(baseline), before)
    (repository / _SCRIPT_PATH).write_text(_script(2))
    loading = repository / _LOADING_PATH
    text = loading.read_text()
    anchor = "    model.eval()\n"
    assert text.count(anchor) == 1
    loading.write_text(
        text.replace(
            anchor,
            "    model.lm_head.register_forward_hook(\n"
            "        lambda module, inputs, output: output + 1\n"
            "    )\n" + anchor,
        )
    )
    after = _commit(repository, "test: shift engine logits and script score")
    assert before != after
    return repository, baseline, before, after


def _stamp_baseline(document: Path, baseline: Path, data: Path) -> None:
    """Use the baseline's loader in an isolated process, never operator imports."""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,sys; from pathlib import Path; "
            "sys.path.insert(0, sys.argv[1]); "
            "from causalab.cli import register_model_key; "
            "from causalab.io.env import FileArtifacts,FileDatasets,ResolutionEnv; "
            "from causalab.workflow.document import load_workflow; "
            "from causalab.measurement.census import collect_pins, stamp_pins; "
            "p=Path(sys.argv[2]); "
            "register_model_key(json.loads((p.parent/'inference.json').read_text())); "
            "env=ResolutionEnv(datasets=FileDatasets(root=Path(sys.argv[3])), "
            "artifacts=FileArtifacts(root=p.parent)); "
            "raw=json.loads(p.read_text()); raw.pop('measurement'); "
            "loaded=load_workflow(raw,env,workflow_dir=p.parent); "
            "stamp_pins(p, collect_pins(loaded, env.datasets))",
            str(baseline),
            str(document),
            str(data),
        ],
        check=True,
    )


@pytest.mark.parametrize("deployment", ["repository", "source_bundle"])
def test_same_pinned_benchmark_runs_distinct_commits(tmp_path, monkeypatch, deployment):
    checkpoint, data = _local_inputs(tmp_path, monkeypatch)
    repository, baseline, before, after = _committed_arms(tmp_path)
    protocol = {
        "header": {"protocol_version": "4"},
        "model": {"key": str(checkpoint), "revision": "local", "dtype": "fp32"},
        "data": {"base": {"dataset": "prompts", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "head", "pos": -1}},
            "save": [
                {
                    "read": "logits",
                    "model": "original",
                    "file_path": "logits.safetensors",
                }
            ],
        },
    }
    (tmp_path / "inference.json").write_text(json.dumps(protocol))
    study = {
        "version": "1",
        "output_dir": "research",
        "steps": {
            "inference": {
                "type": "intervention_protocol",
                "document": "inference.json",
            },
            "score": {
                "type": "script",
                "script": {"module": _SCRIPT_MODULE},
                "inputs": {},
                "outputs": {
                    "out": {
                        "file": "score.json",
                        "columns": {"id": "string", "score": "int64"},
                    }
                },
            },
        },
        "measurement": {
            "version": 1,
            "arms": {
                "before": {"revision": before},
                "after": {"revision": after},
                "eager": {"revision": before},
            },
            "seeds": [0],
            "repeats": 1,
            "warmups": 0,
            "bootstrap_draws": 100,
            "cases": {
                "operation": {"kind": "operation", "step": "inference"},
                "resident": {"kind": "workflow", "cold_process": False},
                "cold": {"kind": "workflow", "cold_process": True},
            },
            "observations": {
                "logits": {
                    "step": "inference",
                    "file": "logits.safetensors",
                    "kind": "tensor",
                },
            },
            "profile": {"cases": ["operation", "cold"]},
            "acceptance": [
                {
                    "name": "unchanged_logits",
                    "path": [
                        "cases",
                        "resident",
                        "comparisons",
                        "before_after",
                        "observations",
                        _LOGITS_KEY,
                        "paired_drift",
                        "0",
                        "max_abs",
                    ],
                    "maximum": 0.0,
                }
            ],
        },
    }
    document = tmp_path / "study.json"
    document.write_text(json.dumps(study))
    _stamp_baseline(document, baseline, data)
    authored = document.read_bytes()
    authored_pins = json.loads(authored)["pins"]
    assert _SCRIPT_MODULE in authored_pins["scripts"]
    assert set(authored_pins) >= {"scripts", "documents", "datasets"}
    authored_document = document
    arm_bindings = {
        arm: {"repository": str(repository), "python": sys.executable}
        for arm in ("before", "after", "eager")
    }
    if deployment == "source_bundle":
        # Exercise the packaging and execution path used by the remote launcher.
        # SSH transport itself is independently covered by launch contract tests.
        frozen_dir = tmp_path / "remote-study"
        frozen = _freeze_study(document, frozen_dir)
        assert frozen["pins"]["scripts"] == authored_pins["scripts"]
        for arm, specification in frozen["measurement"]["arms"].items():
            source = tmp_path / "sources" / arm
            receipt = archive_arm(repository, specification["revision"], source)
            if arm == "before":
                check_source_pins(frozen, source / "source.tar", arm=arm)
            specification["revision"] = receipt["source_commit"]
            arm_bindings[arm] = {"source": str(source), "python": sys.executable}
        document = frozen_dir / "workflow.json"
        document.write_text(json.dumps(frozen))
    execution_document = document.read_bytes()
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": arm_bindings,
                "device": "cpu",
                "data_root": str(data),
                "artifacts_root": str(tmp_path),
            }
        )
    )
    output = tmp_path / "run"
    assert (
        main(
            [
                "measure",
                str(document),
                "--bindings",
                str(bindings),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert authored_document.read_bytes() == authored
    assert document.read_bytes() == execution_document
    for case in ("operation", "resident", "cold"):
        report = json.loads((output / "reports" / f"{case}.json").read_text())
        workers = {
            arm: report["sources"][arm]["context"]["worker"]
            for arm in ("before", "after")
        }
        assert (
            workers["before"]["benchmark_identity"]
            == workers["after"]["benchmark_identity"]
        )
        assert workers["before"]["shared_pins"] == workers["after"]["shared_pins"]
        assert workers["before"]["source_pins"]["scripts"] == authored_pins["scripts"]
        assert workers["after"]["source_pins"]["scripts"] != authored_pins["scripts"]
        for arm, revision in (("before", before), ("after", after)):
            implementation = report["sources"][arm]["provenance"]["implementation"]
            assert workers[arm]["source_commit"] == revision
            assert Path(implementation["location"]).is_relative_to(
                output / "deployment" / arm
            )
            for capture in report["captures"][arm]:
                assert capture["status"] == "completed", capture
                assert capture["observation_check"]["exactly_equal"] is True
                assert capture["identity"]["source_commit"] == revision
                assert capture["identity"]["source_pins"] == workers[arm]["source_pins"]
        for value in report["observations"].values():
            assert value["paired_drift"][0]["max_abs"] == pytest.approx(1.0, abs=1e-6)
        assert any(key.startswith("logits/") for key in report["observations"])
        if case != "operation":
            for arm, score in (("before", 1), ("after", 2)):
                collection_path = Path(report["sources"][arm]["file"])
                collection = json.loads(collection_path.read_text())
                for sample in collection["samples"]:
                    score_file = (
                        collection_path.parent
                        / sample["workflow_outputs"]
                        / "score/score.json"
                    )
                    assert json.loads(score_file.read_text()) == [
                        {"id": "shared", "score": score}
                    ]
        if case in {"operation", "cold"}:
            assert all(report["captures"][arm] for arm in ("before", "after"))
        control = json.loads(
            (output / "reports" / f"{case}.eager_before.json").read_text()
        )
        assert all(
            value["paired_drift"][0]["max_abs"] == 0.0
            for value in control["observations"].values()
        )
    result = json.loads((output / "reports/study.json").read_text())
    assert result["acceptance"]["status"] == "failed"
    assert result["acceptance"]["criteria"][0]["value"] == pytest.approx(1.0, abs=1e-6)
    blocks = {
        path: path.read_bytes()
        for path in (output / "collections/blocks").rglob("block.json")
    }
    assert blocks
    run(document, bindings, output, resume=True)
    assert all(path.read_bytes() == contents for path, contents in blocks.items())
    if deployment == "source_bundle":
        archive = Path(arm_bindings["after"]["source"]) / "source.tar"
        original_archive = archive.read_bytes()
        archive.write_bytes(original_archive + b"tampering")
        with pytest.raises(ValueError, match="source archive changed"):
            run(document, bindings, output, resume=True)
        archive.write_bytes(original_archive)
    installed_script = output / "deployment/after/installed" / _SCRIPT_PATH
    installed_script.write_text(installed_script.read_text() + "\n# tampering\n")
    with pytest.raises(ValueError, match="deployed source installation changed"):
        run(document, bindings, output, resume=True)
