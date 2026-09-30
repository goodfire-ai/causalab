"""A protocol step's ``execution`` block (workflow spec §2.2; IM spec §8).

``batch_rows`` and ``fit_rows`` are execution parameters — how many rows one
forward covers — tunable per step so a workflow can run its harvest whole and
bound its fit. They are the step's *engine* knobs, never the document's: the
block enters no canonical form and no digest, and a workflow with and without
it is the same workflow.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.workflow.document import (
    ProtocolStep,
    WorkflowError,
    load_workflow,
    parse_workflow,
)
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
LOCATE_PRESET = PROTOCOLS_DIR / "weekdays_locate_scan.json"


def _workflow(execution: dict[str, Any] | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "intervention_protocol",
        "document": "methods/locate.json",
    }
    if execution is not None:
        step["execution"] = execution
    return {"version": "1", "output_dir": "run", "steps": {"locate": step}}


def _protocol_step(document: dict[str, Any]) -> ProtocolStep:
    step = parse_workflow(document).steps["locate"]
    assert isinstance(step, ProtocolStep)
    return step


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_execution_block_lands_on_the_step():
    step = _protocol_step(_workflow({"fit_rows": 32, "batch_rows": 64}))
    assert dict(step.execution) == {"fit_rows": 32, "batch_rows": 64}


def test_execution_block_defaults_to_empty():
    step = _protocol_step(_workflow())
    assert dict(step.execution) == {}


def test_execution_null_means_unbounded_for_this_step():
    """``null`` is a value, not an absence: it overrides an engine-wide bound
    for this step, so the parser keeps the key."""
    step = _protocol_step(_workflow({"fit_rows": None}))
    assert dict(step.execution) == {"fit_rows": None}
    assert "fit_rows" in step.execution


@pytest.mark.parametrize(
    "execution",
    (
        {"fit_rows": 0},
        {"fit_rows": -4},
        {"fit_rows": "8"},
        {"fit_rows": True},
        {"batch_rows": 2.0},
        {"rows": 3},
        [8],
        8,
    ),
    ids=(
        "zero",
        "negative",
        "string",
        "bool",
        "float",
        "unknown-key",
        "list",
        "scalar",
    ),
)
def test_malformed_execution_block_is_refused(execution: Any):
    with pytest.raises(WorkflowError) as err:
        parse_workflow(_workflow(execution))
    assert err.value.rule == 1
    assert "steps.locate" in str(err.value)
    assert "execution" in str(err.value)


def test_unknown_execution_key_is_named():
    with pytest.raises(WorkflowError) as err:
        parse_workflow(_workflow({"rows": 3}))
    assert "'rows'" in str(err.value)


# --------------------------------------------------------------------------- #
# identity: the block is execution, not part of the workflow's identity
# --------------------------------------------------------------------------- #


def _copy_locate(tmp_path: Path) -> None:
    methods = tmp_path / "methods"
    methods.mkdir(exist_ok=True)
    shutil.copyfile(LOCATE_PRESET, methods / "locate.json")


def test_execution_block_enters_no_canonical_form_and_no_digest(tmp_path, env):
    _copy_locate(tmp_path)
    plain = load_workflow(_workflow(), env, workflow_dir=tmp_path)
    bounded = load_workflow(
        _workflow({"fit_rows": 32, "batch_rows": 64}), env, workflow_dir=tmp_path
    )
    assert bounded.digest == plain.digest
    assert bounded.canonical == plain.canonical
    assert bounded.step_digests == plain.step_digests
    assert "execution" not in json.dumps(bounded.canonical)
    assert "fit_rows" not in json.dumps(bounded.canonical)
    # and the parsed step still carries it for the runner
    step = bounded.document.steps["locate"]
    assert isinstance(step, ProtocolStep)
    assert dict(step.execution) == {"fit_rows": 32, "batch_rows": 64}
