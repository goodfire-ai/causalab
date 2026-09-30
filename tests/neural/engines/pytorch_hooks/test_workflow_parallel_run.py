"""A workflow under a geometry through the real CLI, over ``gloo``
(``docs/model_parallelism.md`` §3, §10.6, §11; workflow spec §8, §9): the
corpus workflows' shape — a locate scan feeding a select feeding a DAS fit —
on the tiny Llama at ``--parallel tp=2`` and ``--parallel pp=2``, each
against the world-1 run of the same workflow.

Every rank runs the three steps in lockstep with the same engine geometry:
the scan's and the fit's engine requests are collective, the ``select``
script and every write under the ROOT are the joiner's alone. What is
compared: the ROOT's listing is the world-1 listing exactly — a rank that
does not publish leaves nothing behind, no attempt directory, no second
tree; every step record carries the world-1 identities and digests of the
points with ``execution.parallel`` saying ``spawned`` at world 2; the
``select``'s emitted value is equal. The **outputs**: under ``pp=2`` byte
identity, as the
document smokes assert (§8.3: placement moves no reduction); under ``tp=2``
the bands the document smokes pinned — ``test_tensor_expert_parallel_run.py``'s
for the scan (a rowwise projection's all-reduce over two partial sums),
``test_train_parallel_run.py``'s for the fit carried through ten AdamW
steps — with the maxima measured here recorded beside them. The parent of
the spawn never loads a model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from causalab.cli import main
from causalab.io.step_record import SIDECAR
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.io.tables import read_table

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_tensor_expert_parallel_run import (
    BAND as SCAN_BAND,
)
from tests.neural.engines.pytorch_hooks.test_train_parallel_run import (
    BAND as FIT_BAND,
)

pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("checked_gradients_gloo")]

REPO = Path(__file__).resolve().parents[4]
SCAN = REPO / "tests" / "workflow" / "fixtures" / "fan_out" / "protocols" / "scan.json"
DAS = REPO / "tests" / "protocols" / "04_das_im.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"

STEPS = ("scan", "best", "fit")
SCAN_TABLES = ("iia.json", "logit_diff.json")
FIT_TABLES = ("iia.json", "ce.json")
#: The record fields a launched run must carry unchanged from world 1.
IDENTITY_FIELDS = ("identity", "document_digest", "point_digests", "files", "engine")

#: Measured here (2026-09-15, torch 2.9.0, gloo, CPU, fp32, this file's own
#: program): the maximum absolute difference against the world-1 run per step
#: and class under ``tp=2`` — the scan's tables and the counterfactual read,
#: the fit's bundle and its tables (relative to a magnitude of at least one)
#: — the same maxima the document smokes measured for these documents alone,
#: so the workflow adds nothing to the band. Under ``pp=2`` every byte is
#: equal. The bands are the document smokes', each at least twenty times
#: the maximum measured here.
MEASURED: dict[str, float] = {
    "tp=2 scan": 1.5e-8,
    "tp=2 fit tables": 1.5e-8,
    "tp=2 fit bundle": 3.0e-8,
}
assert SCAN_BAND >= 20 * MEASURED["tp=2 scan"]
assert FIT_BAND >= 20 * max(MEASURED["tp=2 fit tables"], MEASURED["tp=2 fit bundle"])
#: What this run measured, per label, for the record.
OBSERVED: dict[str, float] = {}


def _tree(tmp: Path) -> Path:
    """The workflow: the fan-out fixture's scan retargeted to the tiny
    Llama's two layers (plus the counterfactual read saved as a tensor), the
    shipped ``select`` over its ``iia.json``, and the corpus DAS fit
    retargeted to the fixture with its layer read from the select."""
    root = tmp / "wf"
    (root / "protocols").mkdir(parents=True)
    scan = json.loads(SCAN.read_text())
    scan["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    scan["method"]["sites"]["target"]["layers"] = {"sweep": [0, 1]}
    scan["method"]["save"].append(
        {
            "read": "v_cf",
            "model": "original_counterfactual",
            "file_path": "v_cf.safetensors",
        }
    )
    (root / "protocols" / "scan.json").write_text(json.dumps(scan, indent=2))
    das = json.loads(DAS.read_text())
    das["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    das["method"]["sites"]["target"] = {"component": "block_output", "layers": [0]}
    (root / "protocols" / "das.json").write_text(json.dumps(das, indent=2))
    workflow = {
        "version": "1",
        "description": "locate -> select -> fit on the tiny Llama",
        "output_dir": "run",
        "steps": {
            "scan": {
                "type": "intervention_protocol",
                "document": "protocols/scan.json",
            },
            "best": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {
                    "table": {"step": "scan", "file": "iia.json"},
                    "choose": "max",
                    "emit": {"best_layer": "sites.target.layers"},
                },
                "outputs": {
                    "values": {"file": "values.json", "keys": {"best_layer": 0}}
                },
            },
            "fit": {
                "type": "intervention_protocol",
                "document": "protocols/das.json",
                "set": {
                    "sites.target.layers": {"artifact": "best", "key": "best_layer"}
                },
            },
        },
    }
    document = root / "workflow.json"
    document.write_text(json.dumps(workflow, indent=2) + "\n")
    return document


def _argv(document: Path, out: Path, *extra: str) -> list[str]:
    return [
        "run",
        str(document),
        "--data-root",
        str(DATA),
        "--artifacts-root",
        str(document.parent),
        "--engine",
        "auto",
        "--out",
        str(out),
        "--device",
        "cpu",
        *extra,
    ]


def _block(**axes: int) -> dict[str, Any]:
    geometry = {"data": 1, "pipeline": 1, "context": 1, "tensor": 1, "expert": 1}
    geometry.update(axes)
    return {
        **geometry,
        "data_mode": "points",
        "world": geometry["pipeline"] * max(geometry["tensor"], geometry["expert"]),
        "launcher": "spawned",
    }


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def _listing(root: Path) -> list[str]:
    """Every path under ``root`` but the event stream's line count."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _record(out: Path, step: str) -> dict[str, Any]:
    return json.loads((out / "run" / step / SIDECAR).read_text())


def _tensor_diff(a: Path, b: Path) -> float:
    x, y = load_file(str(a)), load_file(str(b))
    assert set(x) == set(y), a.name
    worst = 0.0
    for key in x:
        if x[key].shape != y[key].shape:
            return math.inf
        if x[key].dtype.is_floating_point:
            worst = max(worst, (x[key].double() - y[key].double()).abs().max().item())
        elif not torch.equal(x[key], y[key]):
            return math.inf
    with safe_open(str(a), "pt") as p, safe_open(str(b), "pt") as q:
        assert p.metadata() == q.metadata(), a.name
    return worst


def _table_diff(a: Path, b: Path, *, relative: bool) -> float:
    rows_a, rows_b = read_table(a), read_table(b)
    assert len(rows_a) == len(rows_b), a.name
    worst = 0.0
    for x, y in zip(rows_a, rows_b):
        assert set(x) == set(y), a.name
        for key in x:
            if isinstance(x[key], float) or isinstance(y[key], float):
                gap = abs(float(x[key]) - float(y[key]))
                if relative:
                    gap /= max(1.0, abs(float(x[key])))
                worst = max(worst, gap)
            else:
                assert x[key] == y[key], (a.name, key)
    return worst


def _assert_records(solo: Path, parallel: Path, block: dict[str, Any]) -> None:
    """Every step record: the world-1 identities; the protocol steps'
    ``execution.parallel`` is ``block``; the script step's record is the
    world-1 record but for nothing at all."""
    for step in STEPS:
        a, b = _record(solo, step), _record(parallel, step)
        assert b["status"] == "completed", step
        for key in IDENTITY_FIELDS:
            if key in a:
                assert a[key] == b[key], (step, key)
        if step == "best":
            assert a == b
            continue
        assert b["execution"]["parallel"] == block, step
        assert a["execution"]["parallel"]["launcher"] == "solo"
        rest_a = {k: v for k, v in a.items() if k not in ("execution", "digests")}
        rest_b = {k: v for k, v in b.items() if k not in ("execution", "digests")}
        assert rest_a == rest_b, step
        execution_a = {k: v for k, v in a["execution"].items() if k != "parallel"}
        execution_b = {k: v for k, v in b["execution"].items() if k != "parallel"}
        assert execution_a == execution_b, step
    manifest_a = json.loads((solo / "run" / "workflow.json").read_text())
    manifest_b = json.loads((parallel / "run" / "workflow.json").read_text())
    assert {s: e["status"] for s, e in manifest_b["steps"].items()} == {
        s: "completed" for s in STEPS
    }
    assert manifest_a["steps"].keys() == manifest_b["steps"].keys()


def _assert_parity(
    solo: Path, parallel: Path, block: dict[str, Any], *, exact: bool, label: str
) -> None:
    """The ROOT's listing is world 1's exactly; the records as above; the
    select's value equal; the outputs equal to the byte (``exact``) or
    within the document smokes' bands, the maxima recorded."""
    assert _listing(parallel / "run") == _listing(solo / "run")
    _assert_records(solo, parallel, block)
    assert (parallel / "run" / "best" / "values.json").read_bytes() == (
        solo / "run" / "best" / "values.json"
    ).read_bytes()
    scan_files = (*SCAN_TABLES, "v_cf.safetensors")
    fit_files = (*FIT_TABLES, "rot.safetensors")
    if exact:
        for step, names in (("scan", scan_files), ("fit", fit_files)):
            for name in names:
                assert (parallel / "run" / step / name).read_bytes() == (
                    solo / "run" / step / name
                ).read_bytes(), (step, name)
        return
    worst_scan = max(
        *(
            _table_diff(
                solo / "run" / "scan" / t, parallel / "run" / "scan" / t, relative=False
            )
            for t in SCAN_TABLES
        ),
        _tensor_diff(
            solo / "run" / "scan" / "v_cf.safetensors",
            parallel / "run" / "scan" / "v_cf.safetensors",
        ),
    )
    worst_fit_tables = max(
        _table_diff(
            solo / "run" / "fit" / t, parallel / "run" / "fit" / t, relative=True
        )
        for t in FIT_TABLES
    )
    worst_fit_bundle = _tensor_diff(
        solo / "run" / "fit" / "rot.safetensors",
        parallel / "run" / "fit" / "rot.safetensors",
    )
    OBSERVED[f"{label} scan"] = worst_scan
    OBSERVED[f"{label} fit tables"] = worst_fit_tables
    OBSERVED[f"{label} fit bundle"] = worst_fit_bundle
    assert worst_scan <= SCAN_BAND, ("scan", worst_scan)
    assert worst_fit_tables <= FIT_BAND, ("fit tables", worst_fit_tables)
    assert worst_fit_bundle <= FIT_BAND, ("fit bundle", worst_fit_bundle)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawned ranks inherit the environment: offline, one thread each."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent of a spawn never loads a model: in *this* process the
    loader raises; the children are fresh interpreters and load normally."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


@pytest.fixture(scope="module")
def solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The world-1 run every launched run is compared against: the
    document and its ROOT."""
    tmp = tmp_path_factory.mktemp("wf-solo")
    document = _tree(tmp)
    out = tmp / "out"
    assert main(_argv(document, out)) == 0
    return document, out


# --------------------------------------------------------------------------- #
# the launched runs
# --------------------------------------------------------------------------- #


def _launched(solo_document: Path, tmp: Path, geometry: str) -> tuple[Path, Path]:
    """A fresh copy of the tree run through the spawn parent under
    ``geometry``; the document itself is left as authored (a run never
    stamps the workflow file)."""
    document = _tree(tmp)
    out = tmp / "out"
    authored = document.read_bytes()
    assert main(_argv(document, out, "--parallel", geometry)) == 0
    assert document.read_bytes() == authored
    assert solo_document.read_bytes() == authored
    return document, out


def test_tp2_runs_the_workflow_in_lockstep_within_the_bands(
    solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo_out = solo
    _, out = _launched(document, tmp_path, "tp=2")
    _assert_parity(solo_out, out, _block(tensor=2), exact=False, label="tp=2")


def test_pp2_runs_the_workflow_in_lockstep_to_the_byte(
    solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo_out = solo
    _, out = _launched(document, tmp_path, "pp=2")
    _assert_parity(solo_out, out, _block(pipeline=2), exact=True, label="pp=2")


def test_the_data_axis_is_refused_before_any_child_starts(
    solo: tuple[Path, Path], tmp_path: Path, never_load: None, capsys
) -> None:
    """§11: a workflow's runner is one process, so ``dp`` is refused by name
    on the parent — no child, no model, nothing under the ROOT."""
    document, _ = solo
    out = tmp_path / "out"
    assert main(_argv(document, out, "--parallel", "dp=2")) == 1
    err = capsys.readouterr().err
    assert err.startswith("refused: [P4]") and "--parallel.data" in err
    assert not out.exists()
