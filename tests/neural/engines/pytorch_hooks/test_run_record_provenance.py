"""A run's records name the device the engine ran on and the commit the
model's revision resolved to (§8, execution provenance).

``model.revision`` is usually a moving ref such as ``main``, so the document
alone does not say which weights produced a table. The receipt
(``protocol.json``) and a workflow step's ``_step.json`` record the commit
the loader resolved, or null for a model loaded from a local directory, and
the device the engine was built with. Both are
execution facts: no canonical form, digest or stamp carries them. The oracle
for the commit is the Hugging Face cache itself, read through
``huggingface_hub``, not the loader under test.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from causalab.cli import main
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.pipeline import compile_protocol

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.smoke

DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}
COMMIT = re.compile(r"[0-9a-f]{40}")


def _cached_path(key: str, revision: str = "main") -> str:
    """The cached ``config.json`` of ``key`` at ``revision``."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(key, "config.json", revision=revision)
    assert isinstance(path, str), f"{key}@{revision} is not in the Hub cache"
    return path


def _cached_commit(key: str, revision: str = "main") -> str:
    """The snapshot directory the Hub cache maps ``revision`` to: the parent
    of the cached ``config.json``."""
    return Path(_cached_path(key, revision)).parent.name


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> ResolutionEnv:
    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


@pytest.mark.parametrize(
    "engine", [PytorchHooksEngine, NnsightEngine], ids=["pytorch_hooks", "nnsight"]
)
def test_the_receipt_names_the_device_and_the_resolved_commit(
    engine: type, env: ResolutionEnv, tmp_path: Path
) -> None:
    loaded = compile_protocol(DOCUMENT, env=env, overrides=OVERRIDES)
    run_protocol(loaded, env, engine(device="cpu"), tmp_path, record=True)
    receipt = json.loads((tmp_path / RUN_RECORD_NAME).read_text())
    assert receipt["execution"]["device"] == "cpu"
    (model,) = receipt["models"]
    assert model == {
        "key": TINY_LLAMA,
        "revision": "main",
        "resolved_revision": _cached_commit(TINY_LLAMA),
    }
    assert COMMIT.fullmatch(model["resolved_revision"])
    # execution, not identity: the canonical document still says `main`
    assert receipt["canonical"]["model"]["revision"] == "main"
    assert model["resolved_revision"] not in json.dumps(receipt["canonical"])


def test_a_workflow_step_record_names_the_device_and_the_resolved_commit(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    workflow = tmp_path / "provenance.json"
    workflow.write_text(
        json.dumps(
            {
                "version": "1",
                "description": "one protocol step on the tiny model",
                "output_dir": "provenance",
                "steps": {
                    "patch": {
                        "type": "intervention_protocol",
                        "document": str(DOCUMENT),
                        "set": OVERRIDES,
                    }
                },
            }
        )
    )
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(workflow),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    record = json.loads(
        (tmp_path / "runs" / "provenance" / "patch" / "_step.json").read_text()
    )
    assert record["execution"]["device"] == "cpu"
    assert record["models"] == [
        {
            "key": TINY_LLAMA,
            "revision": "main",
            "resolved_revision": _cached_commit(TINY_LLAMA),
        }
    ]


def test_a_model_loaded_from_a_local_directory_records_no_commit(
    tmp_path: Path,
) -> None:
    """A local directory is loaded as it is, not through the Hub cache, so
    there is no snapshot commit to record: ``resolved_revision`` is null and
    the entry keeps the key the document named."""
    local = tmp_path / "tiny"
    shutil.copytree(Path(_cached_path(TINY_LLAMA)).parent, local)
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts)
    raw = json.loads(DOCUMENT.read_text())
    raw["model"]["key"] = str(local)
    raw["method"]["sites"]["target"]["layers"] = [1]
    document = tmp_path / "local.json"
    document.write_text(json.dumps(raw))
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(document),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--record",
        ]
    )
    assert code == 0
    receipt = json.loads((out / RUN_RECORD_NAME).read_text())
    assert receipt["models"] == [
        {"key": str(local), "revision": "main", "resolved_revision": None}
    ]
