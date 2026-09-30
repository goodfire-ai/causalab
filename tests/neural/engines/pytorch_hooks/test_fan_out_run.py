"""A declared fan-out on the tiny fixture (workflow spec §2.9; the engine
variant of T15).

`tests/workflow/test_fan_out.py` proves the join on a CPU stub whose rows are
a function of the point digest. This file is the real engine: the fan-out
fixture's scan retargeted to tiny Llama over its two layers runs once whole
and once as two shards, and the joined tables equal the unsharded ones —
floats within `test_microbatch`'s fp32 tolerance (a different batch shape may
take a different kernel path), every coordinate column and every integer
exactly — under one receipt carrying the full axes and point digests and no
`engine` or `execution` block.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.workflow import manifest as mf

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "tests" / "workflow" / "fixtures" / "fan_out"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"
OUTPUT_DIR = "fanned_tiny"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"  # minimal_cpu.json's pin
TOLERANCE = {"rel": 1e-5, "abs": 1e-5}  # test_microbatch's fp32 tolerance


def _tree(tmp: Path) -> Path:
    """The fan-out fixtures, the scan retargeted to tiny Llama's two layers."""
    root = tmp / "fan_out"
    shutil.copytree(FIXTURES, root)
    doc = json.loads((root / "protocols" / "scan.json").read_text())
    doc["model"] = {"key": TINY, "revision": TINY_REVISION, "dtype": "fp32"}
    doc["method"]["sites"]["target"]["layers"] = {"sweep": [0, 1]}
    (root / "protocols" / "scan.json").write_text(json.dumps(doc, indent=2))
    return root


def _workflow(root: Path, fan_out: dict[str, Any] | None) -> Path:
    raw = json.loads((root / "scan_wf.json").read_text())
    raw["output_dir"] = OUTPUT_DIR
    if fan_out is None:
        del raw["steps"]["scan"]["fan_out"]
    else:
        raw["steps"]["scan"]["fan_out"] = fan_out
    target = root / ("fanned.json" if fan_out else "whole.json")
    target.write_text(json.dumps(raw, indent=2))
    return target


def _run_cli(root: Path, workflow: Path, out: Path) -> int:
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(workflow),
            "--data-root",
            str(DATA),
            "--artifacts-root",
            str(root),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )


def _rows_close(a: list[dict[str, Any]], b: list[dict[str, Any]], where: str) -> None:
    assert len(a) == len(b), where
    for i, (x, y) in enumerate(zip(a, b)):
        assert set(x) == set(y), f"{where}[{i}]"
        for key in x:
            if isinstance(x[key], float) or isinstance(y[key], float):
                assert y[key] == pytest.approx(x[key], **TOLERANCE), (
                    f"{where}[{i}].{key}"
                )
            else:
                assert x[key] == y[key], f"{where}[{i}].{key}"


@pytest.mark.parametrize(
    "fan_out",
    [
        {"over": {"shards": 2}, "join": {"require": "all"}},
        {"over": {"axis": "sites.target.layers"}, "join": {"require": "all"}},
    ],
    ids=["two_shards", "axis_layers"],
)
def test_t15_a_sharded_run_joins_to_the_unsharded_run(
    tmp_path: Path, fan_out: dict[str, Any]
) -> None:
    root = _tree(tmp_path)
    whole_out, fanned_out = tmp_path / "whole", tmp_path / "fanned"
    assert _run_cli(root, _workflow(root, None), whole_out) == 0
    assert _run_cli(root, _workflow(root, fan_out), fanned_out) == 0
    whole, fanned = whole_out / OUTPUT_DIR, fanned_out / OUTPUT_DIR
    for rel in ("iia.json", "logit_diff.json"):
        _rows_close(
            json.loads((whole / "scan" / rel).read_text()),
            json.loads((fanned / "scan" / rel).read_text()),
            rel,
        )
    single = json.loads((whole / "scan" / "_step.json").read_text())
    joined = json.loads((fanned / "scan" / "_step.json").read_text())
    assert joined["axes"] == single["axes"] == ["positions.tap", "sites.target.layers"]
    assert joined["point_digests"] == single["point_digests"]
    assert joined["document_digest"] == single["document_digest"]
    assert "engine" not in joined and "execution" not in joined
    assert single["engine"] == "pytorch_hooks"
    assert joined["fan_out"]["width"] == 2 and joined["join"]["require"] == "all"
    assert set(joined["join"]["consumed"]) == {"scan@0", "scan@1"}
    steps = json.loads((fanned / mf.MANIFEST).read_text())["steps"]
    assert {name: entry["status"] for name, entry in steps.items()} == {
        "scan@0": "completed",
        "scan@1": "completed",
        "scan": "completed",
        "best": "completed",
    }
    child = json.loads((fanned / "scan@0" / "_step.json").read_text())
    assert child["engine"] == "pytorch_hooks" and child["points"] == 2
    assert child["shard"]["index"] == 0 and child["shard"]["of"] == 2
    # the shipped select reads the joined table as it reads the unsharded one
    assert json.loads((whole / "best" / "values.json").read_text()) == json.loads(
        (fanned / "best" / "values.json").read_text()
    )
