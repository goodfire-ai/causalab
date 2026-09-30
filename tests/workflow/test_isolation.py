"""An isolated step runs in a subprocess with its own dependency set (§4.1).

Two things worth pinning. First that it works at all — the shim imports the
script by path and calls ``main`` with the outputs rebuilt as paths. Second that
a tensor-valued input is **refused** rather than silently mangled: tensors
cannot cross a process boundary, so an isolated step takes them as paths, which
means it must not use an ``entry`` selector.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol.rules.errors import ProtocolError
from causalab.io.tables import read_table
from causalab.workflow.document import load_workflow
from causalab.workflow.runner import run_workflow

pytestmark = pytest.mark.smoke

#: Writes its own pid, so the test can prove the step really left this process
#: rather than quietly falling back to the in-process path.
SCRIPT = """
def main(inputs, outputs):
    import json, os
    from pathlib import Path

    Path(outputs["out"]).write_text(json.dumps([{"n": os.getpid()}]))
"""


def _document(script_dir: Path, *, deps: list[str]) -> dict:
    """The document, with the file its one `path` input names written beside
    it — a relative `path` resolves against the document's directory (§3),
    not the checkout, so the fixture has to travel with the document."""
    (script_dir / "table.json").write_text("[]")
    return {
        "version": "1",
        "output_dir": "iso",
        "steps": {
            "count": {
                "type": "script",
                "script": {"path": "scripts/count.py"},
                "inputs": {"table": {"path": "table.json"}},
                "outputs": {"out": {"file": "count.json", "columns": {"n": "int64"}}},
                "runtime": {"isolate": True, "deps": deps},
            }
        },
    }


@pytest.fixture()
def wf_dir(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "count.py").write_text(SCRIPT)
    return tmp_path


def test_an_isolated_step_runs_in_a_subprocess(wf_dir, tmp_path, env):
    """`deps` is deliberately empty-ish here: the point is the process
    boundary, not resolving a real third-party wheel in a test."""
    import os

    document = _document(wf_dir, deps=["packaging"])
    loaded = load_workflow(document, env, workflow_dir=wf_dir)
    result = run_workflow(loaded, env, tmp_path / "runs", None)
    rows = read_table(result.run_root / "count" / "count.json")
    assert rows, "the isolated step produced no output"
    assert rows[0]["n"] != os.getpid(), (
        "the step ran in this process — isolation silently fell back to the "
        "in-process path"
    )
    manifest = json.loads((result.run_root / "workflow.json").read_text())
    assert manifest["steps"]["count"]["runtime"]["isolate"] is True


def test_a_tensor_input_cannot_cross_the_boundary(wf_dir, tmp_path, env):
    """Refused loudly rather than pickled into JSON: an isolated step reads its
    bundles from paths, inside the script."""
    import torch

    from causalab.io.step_io import write_tensor

    write_tensor(wf_dir / "acts.safetensors", torch.zeros(2, 2), slot="acts")
    document = _document(wf_dir, deps=["packaging"])
    # a selector forces the runner to materialize a tensor, which is the case
    # isolation cannot support
    document["steps"]["count"]["inputs"]["tensor"] = {
        "path": str(wf_dir / "acts.safetensors"),
        "slot": "acts",
    }
    loaded = load_workflow(document, env, workflow_dir=wf_dir)
    with pytest.raises(ProtocolError) as err:
        run_workflow(loaded, env, tmp_path / "runs", None)
    assert "cross a process boundary" in str(err.value)
