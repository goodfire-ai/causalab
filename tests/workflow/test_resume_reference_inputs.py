"""Resume depends on the bytes consumed through every file reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.step_record import SIDECAR
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

pytestmark = pytest.mark.unit

PRODUCER = """import json
from pathlib import Path

def main(inputs, outputs):
    source = json.loads(Path(inputs["source"]).read_text())
    Path(outputs["out"]).write_text(json.dumps({"n": source["n"]}))
"""
CONSUMER = """import json
from pathlib import Path

def main(inputs, outputs):
    value = inputs["value"]
    if not isinstance(value, int):
        value = json.loads(Path(value).read_text())["n"]
    Path(outputs["out"]).write_text(json.dumps({"n": value}))
"""


def _chain(root: Path, *, selector: bool = False) -> tuple[Path, dict[str, Any]]:
    root.mkdir(parents=True)
    (root / "producer.py").write_text(PRODUCER)
    (root / "consumer.py").write_text(CONSUMER)
    source = root / "source.json"
    source.write_text(json.dumps({"n": 2}))
    steps: dict[str, Any] = {
        "produce": {
            "type": "script",
            "script": {"path": "producer.py"},
            "inputs": {"source": {"path": "source.json"}},
            "outputs": {"out": {"file": "value.json", "keys": {"n": 0}}},
        }
    }
    for name, producer in (("consume", "produce"), ("finish", "consume")):
        reference: dict[str, Any] = {"step": producer, "file": "value.json"}
        if selector:
            reference["key"] = "n"
        steps[name] = {
            "type": "script",
            "script": {"path": "consumer.py"},
            "inputs": {"value": reference},
            "outputs": {"out": {"file": "value.json", "keys": {"n": 0}}},
        }
    return source, {"version": "1", "output_dir": "chain", "steps": steps}


def _run(
    root: Path, document: dict[str, Any], *, resume: bool = False
) -> dict[str, str]:
    env = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )
    loaded = load_workflow(document, env, workflow_dir=root)
    result = run_workflow(loaded, env, root / "runs", None, resume=resume)
    return {name: entry["status"] for name, entry in result.manifest["steps"].items()}


def _result(root: Path, step: str) -> int:
    return json.loads((root / "runs" / "chain" / step / "value.json").read_text())["n"]


@pytest.mark.parametrize("selector", [False, True], ids=["file", "selected_value"])
def test_changed_upstream_bytes_invalidate_every_consumer(
    tmp_path: Path, selector: bool
) -> None:
    root = tmp_path / "workflow"
    source, document = _chain(root, selector=selector)
    _run(root, document)
    source.write_text(json.dumps({"n": 3}))
    assert _run(root, document, resume=True) == {
        "produce": "completed",
        "consume": "completed",
        "finish": "completed",
    }
    assert [_result(root, step) for step in document["steps"]] == [3, 3, 3]
    assert set(_run(root, document, resume=True).values()) == {"reused"}


def test_unchanged_producer_output_still_reuses_consumers(tmp_path: Path) -> None:
    root = tmp_path / "workflow"
    source, document = _chain(root)
    _run(root, document)
    source.write_text(json.dumps({"n": 2, "ignored": True}))
    assert _run(root, document, resume=True) == {
        "produce": "completed",
        "consume": "reused",
        "finish": "reused",
    }


@pytest.mark.parametrize(
    "legacy", [True, False], ids=["missing_input_record", "wrong_digest"]
)
def test_unverified_step_input_forces_reexecution(tmp_path: Path, legacy: bool) -> None:
    root = tmp_path / "workflow"
    _, document = _chain(root)
    _run(root, document)
    path = root / "runs" / "chain" / "consume" / SIDECAR
    record = json.loads(path.read_text())
    if legacy:
        record.pop("input_digests", None)
    else:
        record["input_digests"] = {"value": "0" * 64}
    path.write_text(json.dumps(record))
    assert _run(root, document, resume=True)["consume"] == "completed"
    record = json.loads(path.read_text())
    target = root / "runs" / "chain" / "produce" / "value.json"
    assert record["input_digests"] == {
        "value": hashlib.sha256(target.read_bytes()).hexdigest()
    }


@settings(max_examples=12, deadline=None)
@given(
    values=st.lists(st.integers(min_value=-5, max_value=5), min_size=2, max_size=5),
    selector=st.booleans(),
)
def test_resumed_chain_equals_a_fresh_run(values: list[int], selector: bool) -> None:
    """Every input history produces the same final values as fresh execution."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "workflow"
        source, document = _chain(root, selector=selector)
        for value in values:
            source.write_text(json.dumps({"n": value}))
            _run(root, document, resume=True)
            resumed = [_result(root, step) for step in document["steps"]]
            _run(root, document)
            assert resumed == [_result(root, step) for step in document["steps"]]


def test_resume_after_producer_publication_rechecks_consumer_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependency invalidation must survive a process interruption."""
    from causalab.workflow import runner

    class InterruptedRun(RuntimeError):
        pass

    root = tmp_path / "workflow"
    source, document = _chain(root)
    _run(root, document)
    source.write_text(json.dumps({"n": 3}))

    def interrupt(boundary: str, step: str | None) -> None:
        if boundary == "published" and step == "produce":
            raise InterruptedRun("producer published")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_boundary", interrupt)
        with pytest.raises(InterruptedRun, match="producer published"):
            _run(root, document, resume=True)
    assert _result(root, "produce") == 3
    assert _result(root, "consume") == 2
    assert _run(root, document, resume=True) == {
        "produce": "reused",
        "consume": "completed",
        "finish": "completed",
    }
    assert [_result(root, step) for step in document["steps"]] == [3, 3, 3]


def test_nested_workflow_resolves_input_hashes_under_its_own_run_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outer"
    source, inner = _chain(root / "inner", selector=True)
    (root / "inner" / "workflow.json").write_text(json.dumps(inner))
    document = {
        "version": "1",
        "output_dir": "chain",
        "steps": {"nested": {"type": "workflow", "document": "inner/workflow.json"}},
    }
    _run(root, document)
    source.write_text(json.dumps({"n": 3}))
    assert _run(root, document, resume=True) == {
        "nested/produce": "completed",
        "nested/consume": "completed",
        "nested/finish": "completed",
    }
    assert _result(root, "nested/finish") == 3
    assert set(_run(root, document, resume=True).values()) == {"reused"}


def test_selected_tensor_inputs_are_held_to_the_containing_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workflow"
    source, document = _chain(root)
    (root / "producer.py").write_text(
        "import json\nfrom pathlib import Path\nimport torch\n"
        "from causalab.io.tensor_files import save_file\n"
        "def main(inputs, outputs):\n"
        '    n = json.loads(Path(inputs["source"]).read_text())["n"]\n'
        '    save_file({"value": torch.tensor([n])}, str(outputs["out"]))\n'
    )
    (root / "tensor_consumer.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def main(inputs, outputs):\n"
        '    n = int(inputs["value"].item())\n'
        '    Path(outputs["out"]).write_text(json.dumps({"n": n}))\n'
    )
    document["steps"]["produce"]["outputs"] = {"out": "value.safetensors"}
    consume = document["steps"]["consume"]
    consume["script"] = {"path": "tensor_consumer.py"}
    consume["inputs"]["value"] = {
        "step": "produce",
        "file": "value.safetensors",
        "slot": "value",
    }
    _run(root, document)
    source.write_text(json.dumps({"n": 3}))
    assert _run(root, document, resume=True)["consume"] == "completed"
    assert _result(root, "finish") == 3
    assert set(_run(root, document, resume=True).values()) == {"reused"}
