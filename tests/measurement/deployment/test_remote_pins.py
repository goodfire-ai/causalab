"""Freezing changes document pins only after checking the authored bytes."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from hypothesis import given, settings, strategies as st
import pytest

from causalab.measurement.census import CensusError, collect_pins, strip_pins
from causalab.measurement.collection import file_hash, write_record
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks import TASKS_ROOT
from causalab.workflow.document import load_workflow
from causalab.measurement.deployment.remote import _freeze_study
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]


def _census(
    source: Path | dict, env: ResolutionEnv, *, workflow_dir: Path | None = None
):
    """The census of a workflow tree the way the worker takes it: the ``pins``
    record, if any, is stripped before the loader reads the document, and the
    census is recounted from the load."""
    raw = json.loads(source.read_text()) if isinstance(source, Path) else dict(source)
    if workflow_dir is None and isinstance(source, Path):
        workflow_dir = source.parent
    document, _authored = strip_pins(raw)
    loaded = load_workflow(document, env, workflow_dir=workflow_dir)
    return collect_pins(loaded, env.datasets)


def _study(root: Path, *, revision: str = "main") -> tuple[Path, ResolutionEnv]:
    protocol = root / "methods/locate.json"
    protocol.parent.mkdir()
    protocol.write_bytes(
        (REPO / "demos/methods/protocols/weekdays_locate_scan.json").read_bytes()
    )
    workflow = root / "workflow.json"
    write_record(
        workflow,
        {
            "version": "1",
            "output_dir": "run",
            "steps": {
                name: {
                    "type": "intervention_protocol",
                    "document": "methods/locate.json",
                    "set": {"model.revision": revision},
                }
                for name in ("locate", "replica")
            },
            "measurement": {
                "version": 1,
                "arms": {arm: {"revision": "HEAD"} for arm in ("before", "after")},
                "cases": {"locate": {"kind": "operation", "step": "locate"}},
                "seeds": [7],
                "repeats": 1,
                "observations": {
                    "score": {
                        "step": "locate",
                        "file": "score.safetensors",
                        "kind": "tensor",
                    }
                },
            },
        },
    )
    env = ResolutionEnv(
        datasets=FileDatasets(root=TASKS_ROOT, fallback_roots=(FIXTURES / "data",)),
        artifacts=FileArtifacts(root=root),
    )
    raw = json.loads(workflow.read_text())
    raw["pins"] = _census(workflow, env)
    write_record(workflow, raw)
    return workflow, env


@settings(max_examples=6, deadline=None)
@given(
    revision=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12
    )
)
def test_frozen_pins_hold_after_renaming_and_overrides(revision: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workflow, env = _study(root, revision=revision)
        original = workflow.read_bytes()
        old = json.loads(original)
        frozen = _freeze_study(workflow, root / "frozen")
        assert _census(frozen, env, workflow_dir=root / "frozen") == frozen["pins"]
        assert frozen["pins"]["documents"] == {
            f"{name}.json": file_hash(root / "frozen" / f"{name}.json")
            for name in ("locate", "replica")
        }
        assert {k: v for k, v in frozen["pins"].items() if k != "documents"} == {
            k: v for k, v in old["pins"].items() if k != "documents"
        }
        assert workflow.read_bytes() == original
        for step in frozen["steps"].values():
            assert "set" not in step
            assert (
                json.loads((root / "frozen" / step["document"]).read_text())["model"][
                    "revision"
                ]
                == revision
            )


@pytest.mark.parametrize("damage", ["stale", "missing", "extra"])
def test_freezing_refuses_invalid_original_document_pins(
    tmp_path: Path, damage: str
) -> None:
    workflow, _ = _study(tmp_path)
    raw = json.loads(workflow.read_text())
    match damage:
        case "stale":
            with (tmp_path / "methods/locate.json").open("a") as stream:
                stream.write("\n")
        case "missing":
            raw["pins"].pop("documents")
        case "extra":
            raw["pins"]["documents"]["unused.json"] = "0" * 64
    write_record(workflow, raw)
    with pytest.raises(CensusError, match="pins.documents"):
        _freeze_study(workflow, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()
