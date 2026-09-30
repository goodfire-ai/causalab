"""``--resume`` reuses a protocol step that loads an earlier step's output.

A protocol step whose document names a run-tree path compiles against the
run tree when it runs. Two examples are a fit started from a basis that a
script step wrote (``init.file_path``) and an apply that loads the fit's
rotations (``file_path``). The record keeps the digest of that compile, and
the upstream bytes the step read are in it. The load-time digest keeps the
declared paths. ``--resume`` compiles the step again against the current run
tree and compares the two (workflow spec §7). Such a step is reused while its
upstream bytes and its document are unchanged, and runs again once either
moves. A fanned-out child records the same digest as its ``document_digest``
and is held to it the same way.

The chain is the random-start pattern of the ``arithmetic_fig2a`` replication
in miniature, at one layer of the tiny Llama so that every stamp carries its
site at file level: a harvest, a script that writes a start frame from it, a
fit from that start, and an apply of the fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.smoke

TRAIN = "weekdays/train"
MODEL = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
PAIRS = {
    "base": {"dataset": TRAIN, "field": "input"},
    "counterfactual": {"dataset": TRAIN, "field": "counterfactual_inputs[0]"},
}
INTERVENED = {
    "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
    "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
}
READS = {
    "v_cf": {"site": "target", "pos": -1, "featurizer": "rot"},
    "logits": {"site": "lm_head", "pos": -1},
}
WRITES = {
    "patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
}

#: One seeded orthonormal frame per harvested entry, keyed and recorded as
#: the harvest's entry is. The runner stamps the file-level identity the
#: inputs prove; ``entry_model_dtype``, when given, is written into the
#: entry's own record, which overrides that stamp for the entry (§8), so that
#: a test can hand the fit a start its load refuses.
START = """import json
from pathlib import Path

def main(inputs, outputs):
    import torch
    from causalab.io.env import entry_table, read_safetensors_metadata
    from causalab.io.tensor_files import load_file, save_file

    source = Path(inputs["acts"])
    metadata = read_safetensors_metadata(source) or {}
    entries = entry_table(metadata)
    tensors = load_file(str(source))
    generator = torch.Generator().manual_seed(int(inputs["seed"]))
    out, table = {}, {}
    for key in sorted(tensors):
        width = tensors[key].shape[-1]
        frame = torch.linalg.qr(torch.randn(width, 2, generator=generator))[0]
        _, bracket, rest = key.partition("[")
        name = "weight" + bracket + rest
        out[name] = frame.contiguous()
        table[name] = {**entries.get(key, {"coords": {}}), "slot": "weight"}
        if "entry_model_dtype" in inputs:
            table[name]["model_dtype"] = inputs["entry_model_dtype"]
    header = {k: v for k, v in metadata.items() if k != "entries"}
    header["dtype"] = "fp32"
    header["entries"] = json.dumps(table, sort_keys=True)
    save_file(out, str(outputs["start"]), metadata=header)
"""


def _document(method: dict[str, Any], description: str, data: dict) -> dict:
    return {
        "header": {"protocol_version": "4", "description": description},
        "model": MODEL,
        "data": data,
        "method": method,
    }


def _sites(layers: Any) -> dict[str, Any]:
    return {
        "target": {"component": "block_output", "layers": layers},
        "lm_head": {"component": "lm_head"},
    }


def _fit(layers: Any, *, lr: float, init: bool) -> dict:
    """A rank-2 stiefel fit at ``layers``, from the start step's frame when
    ``init``."""
    rot: dict[str, Any] = {"kind": "subspace", "k": 2, "parametrization": "stiefel"}
    if init:
        rot["init"] = {"file_path": "start/start.safetensors"}
    return _document(
        {
            "intervened_models": INTERVENED,
            "sites": _sites(layers),
            "featurizers": {"rot": rot},
            "reads": READS,
            "writes": WRITES,
            "train": {
                "objective": {
                    "ce": {
                        "weight": 1.0,
                        "read": "logits",
                        "model": "patched",
                        "aggregation": {"kind": "cross_entropy", "target": "label"},
                    }
                },
                "params": ["rot"],
                "optimizer": {"name": "adamw", "lr": lr, "weight_decay": 0.0},
                "steps": {"epochs": 1},
                "batch": {"pairs": 2},
                "seed": 0,
            },
            "save": [
                {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
            ],
        },
        "fit one rotation per site",
        PAIRS,
    )


def _apply(layers: Any) -> dict:
    return _document(
        {
            "intervened_models": INTERVENED,
            "sites": _sites(layers),
            "featurizers": {
                "rot": {
                    "kind": "subspace",
                    "k": 2,
                    "parametrization": "stiefel",
                    "file_path": "fit/rot.safetensors",
                }
            },
            "reads": READS,
            "writes": WRITES,
            "save": [
                {
                    "read": "logits",
                    "model": "patched",
                    "aggregation": {"kind": "top_k", "k": 1, "by": "prob"},
                    "file_path": "top1.json",
                }
            ],
        },
        "apply the fitted rotations",
        PAIRS,
    )


def _write(root: Path, documents: dict[str, dict], steps: dict[str, Any]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (root / f"{name}.json").write_text(json.dumps(document, indent=1))
    path = root / "workflow.json"
    path.write_text(
        json.dumps(
            {
                "version": "1",
                "description": "a fit and an apply that load earlier outputs",
                "output_dir": "chain",
                "steps": steps,
            },
            indent=1,
        )
    )
    return path


def _write_chain(
    root: Path, *, seed: int = 0, lr: float = 0.001, entry_model_dtype: str = ""
) -> Path:
    """harvest, start, fit from the start, apply the fit: one layer."""
    start_inputs: dict[str, Any] = {
        "acts": {"step": "harvest", "file": "acts.safetensors"},
        "seed": seed,
    }
    if entry_model_dtype:
        start_inputs["entry_model_dtype"] = entry_model_dtype
    root.mkdir(parents=True, exist_ok=True)
    (root / "start.py").write_text(START)
    harvest = _document(
        {
            "intervened_models": {"original": {"input": "base", "reads": ["acts"]}},
            "sites": {"target": _sites([0])["target"]},
            "reads": {"acts": {"site": "target", "pos": -1}},
            "save": [
                {"read": "acts", "model": "original", "file_path": "acts.safetensors"}
            ],
        },
        "harvest the last-token residual",
        {"base": PAIRS["base"]},
    )
    return _write(
        root,
        {
            "harvest": harvest,
            "fit": _fit([0], lr=lr, init=True),
            "apply": _apply([0]),
        },
        {
            "harvest": {"type": "intervention_protocol", "document": "harvest.json"},
            "start": {
                "type": "script",
                "script": {"path": "start.py"},
                "inputs": start_inputs,
                "outputs": {"start": "start.safetensors"},
            },
            "fit": {"type": "intervention_protocol", "document": "fit.json"},
            "apply": {"type": "intervention_protocol", "document": "apply.json"},
        },
    )


def _write_fanned(root: Path, *, lr: float) -> Path:
    """A fit swept over two layers from its seeded frame, and an apply fanned
    out over the layers, one child per layer."""
    layers = {"sweep": [0, 1]}
    return _write(
        root,
        {"fit": _fit(layers, lr=lr, init=False), "apply": _apply(layers)},
        {
            "fit": {"type": "intervention_protocol", "document": "fit.json"},
            "apply": {
                "type": "intervention_protocol",
                "document": "apply.json",
                "fan_out": {
                    "over": {"axis": "sites.target.layers"},
                    "join": {"require": "all"},
                },
            },
        },
    )


def _run(workflow: Path, out: Path, *, resume: bool) -> int:
    return main(
        [
            "run",
            str(workflow),
            "--engine",
            "auto",
            "--data-root",
            str(FIXTURES / "data"),
            "--out",
            str(out),
            "--register-from-hf",
            *(["--resume"] if resume else []),
        ]
    )


def _statuses(workflow: Path, out: Path, *, resume: bool) -> dict[str, str]:
    assert _run(workflow, out, resume=resume) == 0
    manifest = json.loads((out / "chain" / "workflow.json").read_text())
    return {name: entry["status"] for name, entry in manifest["steps"].items()}


def test_a_step_that_loads_an_earlier_steps_output_is_reused(tmp_path: Path):
    workflow = _write_chain(tmp_path / "wf")
    out = tmp_path / "runs"
    first = _statuses(workflow, out, resume=False)
    assert set(first.values()) == {"completed"}
    again = _statuses(workflow, out, resume=True)
    assert again == {name: "reused" for name in first}


def test_it_runs_again_once_the_upstream_bytes_move(tmp_path: Path):
    """A new start seed changes the start script's identity and its output
    bytes. The fit read those bytes, so it runs again, and so does the apply
    that reads the fit's. The harvest upstream of the change is reused."""
    root = tmp_path / "wf"
    out = tmp_path / "runs"
    _statuses(_write_chain(root, seed=0), out, resume=False)
    moved = _statuses(_write_chain(root, seed=1), out, resume=True)
    assert moved == {
        "harvest": "reused",
        "start": "completed",
        "fit": "completed",
        "apply": "completed",
    }


def test_it_runs_again_once_its_own_document_changes(tmp_path: Path):
    """A new learning rate in the fit document compiles to another digest
    against the same run tree. The fit runs again, and so does the apply of
    its new bytes; the steps upstream of the fit are reused."""
    root = tmp_path / "wf"
    out = tmp_path / "runs"
    _statuses(_write_chain(root, lr=0.001), out, resume=False)
    edited = _statuses(_write_chain(root, lr=0.002), out, resume=True)
    assert edited == {
        "harvest": "reused",
        "start": "reused",
        "fit": "completed",
        "apply": "completed",
    }


def test_a_start_the_fit_now_refuses_is_never_reused(tmp_path: Path):
    """The start's entry now records another model dtype than the fit
    runs at. The compile against the run tree refuses it, so nothing matches
    the fit's record: the fit runs again, and its own attempt reports the
    refusal."""
    root = tmp_path / "wf"
    out = tmp_path / "runs"
    _statuses(_write_chain(root), out, resume=False)
    code = _run(_write_chain(root, entry_model_dtype="bf16"), out, resume=True)
    assert code != 0
    attempts = sorted((out / "chain").glob(".attempts/fit/*/attempt.json"))
    assert attempts, "the fit ran again"
    failure = json.loads(attempts[-1].read_text())
    assert "model_dtype" in json.dumps(failure)


def test_a_fanned_child_runs_again_once_the_bytes_it_loads_move(tmp_path: Path):
    """Each child of a fanned apply loads the fit's bundle. Unchanged, the
    children and their join are reused; once a new learning rate moves the
    fit's bytes, every child and the join run again."""
    root = tmp_path / "wf"
    out = tmp_path / "runs"
    first = _statuses(_write_fanned(root, lr=0.001), out, resume=False)
    assert set(first) == {"fit", "apply", "apply@0", "apply@1"}
    assert set(first.values()) == {"completed"}
    again = _statuses(_write_fanned(root, lr=0.001), out, resume=True)
    assert again == {name: "reused" for name in first}
    moved = _statuses(_write_fanned(root, lr=0.002), out, resume=True)
    assert moved == {name: "completed" for name in first}
