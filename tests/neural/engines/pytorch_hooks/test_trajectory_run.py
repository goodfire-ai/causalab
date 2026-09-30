"""A ``trajectory`` save entry end to end (spec §2.12): a DBM fit photographed
along its way through the real CLI, and a checkpoint reloaded as the mask of a
later document.

What is checked: the bundle holds one ``theta`` per checkpoint keyed by
``step``, each entry carries the fit's identity **and** the checkpoint's
numbers as numbers, the last checkpoint is the fit the featurizer's own
bundle saved, and an apply document that names ``entry: {"step": n}`` runs —
the reload goes through the same identity check a fitted bundle does — while
a step the fit never photographed is refused.

The fit is a flat document on the fixture table with a ``kl`` objective toward
the clean counterfactual distribution: no answer string to tokenize, so
tiny-random's vocabulary is not in the way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata
from causalab.protocol.schema import METHOD_SECTIONS, PROTOCOL_VERSION

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, aggregation, saved, term
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.smoke

COUNT = 3
SPLIT = "weekdays/data#train"

#: The fit's one aggregation: the masked logits' divergence from the clean
#: counterfactual distribution. The save carries it too, and a save holds a
#: read reference in the object form (§2.7).
KL = aggregation("kl", target={"read": "logits_cf", "model": UNWRITTEN})


def _v2(doc: dict[str, Any]) -> dict[str, Any]:
    """The flat sections below, grouped as protocol v2 writes them (§1): the
    tests mutate sections by name, and the shape is the writer's concern."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": doc["model"],
        "data": doc["data"],
        "method": {key: doc[key] for key in METHOD_SECTIONS if key in doc},
    }


def _fit_doc() -> dict[str, Any]:
    return {
        "model": {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": SPLIT, "field": "input"},
            "counterfactual": {"dataset": SPLIT, "field": "counterfactual_inputs[0]"},
        },
        "intervened_models": {
            UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf", "logits_cf"]},
            "masked": {"input": "base", "reads": ["logits"], "writes": ["mask"]},
        },
        "sites": {
            "target": {"component": "block_output", "layers": [0]},
            "lm_head": {"component": "lm_head"},
        },
        "featurizers": {"gate": {"kind": "gate"}},
        "reads": {
            "v_cf": {"site": "target", "pos": -1, "featurizer": "gate"},
            "logits": {"site": "lm_head", "pos": -1},
            "logits_cf": {"site": "lm_head", "pos": -1},
        },
        "writes": {
            "mask": {
                "site": "target",
                "pos": -1,
                "featurizer": "gate",
                "do": {"swap": "v_cf"},
            }
        },
        "train": {
            "objective": [
                [1.0, term("logits", "masked", dict(KL))],
                [0.01, {"l1": "gate"}],
            ],
            "params": ["gate"],
            "optimizer": {"name": "adamw", "lr": 0.1},
            "steps": {"epochs": COUNT},
            "batch": {"pairs": 2},
            "seed": 0,
        },
        "save": [
            saved("logits", "masked", "kl.json", dict(KL)),
            {"value": "gate", "site": "target", "file_path": "gate.safetensors"},
            {
                "kind": "trajectory",
                "every": {"count": COUNT},
                "file_path": "trajectory.safetensors",
            },
        ],
    }


def _apply_doc(step: int) -> dict[str, Any]:
    doc = _fit_doc()
    del doc["train"]
    doc["featurizers"]["gate"] = {
        "kind": "gate",
        "file_path": "fit/trajectory.safetensors",
        "entry": {"step": step},
    }
    doc["save"] = [doc["save"][0]]
    return doc


def _run(base: Path, doc: dict[str, Any], out: Path) -> int:
    path = base / f"{out.name}.json"
    path.write_text(json.dumps(_v2(doc)))
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(base),
            "--out",
            str(out),
        ]
    )


@pytest.fixture(scope="module")
def fit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("trajectory")
    out = base / "fit"
    assert _run(base, _fit_doc(), out) == 0
    return out


def _entries(fit: Path) -> dict[str, dict[str, Any]]:
    header = read_safetensors_metadata(fit / "trajectory.safetensors")
    assert header is not None
    return json.loads(str(header["entries"]))


def test_the_bundle_holds_one_theta_per_checkpoint_keyed_by_step(fit: Path) -> None:
    tensors = load_file(str(fit / "trajectory.safetensors"))
    entries = _entries(fit)
    steps = sorted(int(entries[key]["step"]) for key in tensors)
    # two fixture rows, one batch per epoch: three updates, three checkpoints
    assert len(tensors) == COUNT and steps == [1, 2, 3]
    for key, record in entries.items():
        assert key == f"theta[featurizer=gate,step={record['step']}]"
        # the checkpoint's numbers ride as numbers, the identity as strings
        assert isinstance(record["loss"], float) and isinstance(record["step"], int)
        assert record["weight.1"] == 0.01 and "term.0" in record
        assert record["gate.hard_mask_size"] == float((tensors[key] > 0).sum())
        assert record["parametrization"] == "sigmoid"
        assert record["slot"] == "theta" and record["coords"] == {
            "featurizer": "gate",
            "step": record["step"],
        }
    # the last checkpoint is the fit the featurizer's own bundle saved
    final = load_file(str(fit / "gate.safetensors"))["theta"]
    assert torch.equal(tensors["theta[featurizer=gate,step=3]"], final)
    assert not torch.equal(tensors["theta[featurizer=gate,step=1]"], final)  # moved


def _two_gate_fit_doc() -> dict[str, Any]:
    """Two gates at two layers under one list-valued ``l1`` — the many-layer
    DBM shape whose checkpoints collided under one ``theta[step=n]`` key."""
    doc = _fit_doc()
    doc["sites"]["target1"] = {"component": "block_output", "layers": [1]}
    doc["featurizers"] = {"g0": {"kind": "gate"}, "g1": {"kind": "gate"}}
    doc["reads"]["v_cf"]["featurizer"] = "g0"
    doc["reads"]["v_cf1"] = {
        **doc["reads"]["v_cf"],
        "site": "target1",
        "featurizer": "g1",
    }
    doc["intervened_models"][UNWRITTEN]["reads"].append("v_cf1")
    doc["writes"]["mask"]["featurizer"] = "g0"
    doc["writes"]["mask1"] = {
        "site": "target1",
        "pos": -1,
        "featurizer": "g1",
        "do": {"swap": "v_cf1"},
    }
    doc["intervened_models"]["masked"]["writes"] = ["mask", "mask1"]
    doc["train"]["objective"] = [
        [1.0, term("logits", "masked", dict(KL))],
        [0.01, {"l1": ["g0", "g1"]}],
    ]
    doc["train"]["params"] = ["g0", "g1"]
    doc["save"] = [
        doc["save"][0],
        {"value": "g0", "site": "target", "file_path": "g0.safetensors"},
        {"value": "g1", "site": "target1", "file_path": "g1.safetensors"},
        doc["save"][2],
    ]
    return doc


def _two_gate_apply_doc(
    entry0: dict[str, Any], entry1: dict[str, Any]
) -> dict[str, Any]:
    doc = _two_gate_fit_doc()
    del doc["train"]
    doc["featurizers"] = {
        "g0": {
            "kind": "gate",
            "file_path": "fit2/trajectory.safetensors",
            "entry": entry0,
        },
        "g1": {
            "kind": "gate",
            "file_path": "fit2/trajectory.safetensors",
            "entry": entry1,
        },
    }
    doc["save"] = [doc["save"][0]]
    return doc


def test_two_trained_gates_photograph_separately_and_each_reloads_its_own(
    tmp_path: Path,
) -> None:
    out = tmp_path / "fit2"
    assert _run(tmp_path, _two_gate_fit_doc(), out) == 0
    tensors = load_file(str(out / "trajectory.safetensors"))
    entries = _entries(out)
    # one theta per gate per checkpoint — not one per checkpoint
    assert len(tensors) == 2 * COUNT
    assert {entries[key]["coords"]["featurizer"] for key in tensors} == {"g0", "g1"}
    for name in ("g0", "g1"):
        final = load_file(str(out / f"{name}.safetensors"))["theta"]
        assert torch.equal(tensors[f"theta[featurizer={name},step={COUNT}]"], final)
        assert json.loads(entries[f"theta[featurizer={name},step={COUNT}]"]["site"])[
            "layers"
        ] == [0 if name == "g0" else 1]
    # each gate names its own checkpoint …
    assert (
        _run(
            tmp_path,
            _two_gate_apply_doc(
                {"featurizer": "g0", "step": 2}, {"featurizer": "g1", "step": 2}
            ),
            tmp_path / "apply2",
        )
        == 0
    )
    # … and a bare step is ambiguous, refused rather than resolved to the last
    assert (
        _run(
            tmp_path,
            _two_gate_apply_doc({"step": 2}, {"step": 2}),
            tmp_path / "apply_bare",
        )
        != 0
    )


def test_a_checkpoint_reloads_as_a_gate_through_the_identity_check(
    fit: Path, tmp_path: Path
) -> None:
    base = fit.parent
    out = tmp_path / "apply"
    assert _run(base, _apply_doc(2), out) == 0
    assert (out / "kl.json").exists()
    # a checkpoint the fit never photographed is refused, not silently nearest
    assert _run(base, _apply_doc(10_000), tmp_path / "apply_missing") != 0
