"""``init.from_scores`` through the real executor and the CLI (spec §2.5).

Three things the pure-layer tests cannot show: the executor hands the table
loader to the build, so a document's gate starts where the table says; a fit
from that start records it in ``fit_diagnostics`` and moves; and the CLI path
resolves the table through the run's artifact store (``services.load_table``,
the same resolution a bundle gets), stamps its digest into the canonical form,
and saves a gate whose hard mask is the ranking's top set — the attribution /
magnitude-pruning baseline as one document, with no training at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.protocol.engine import RunContext
from causalab.protocol.registry import component_width
from causalab.protocol.schema import parse_document
from causalab.protocol.rules.document import validate_document

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    _NoDatasets,
    clamp_dbm_doc,
)
from tests.protocol._docs import UNWRITTEN, aggregation, in_order, saved, term
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.smoke

KEEP = 3

#: The masked model's readout the fit trains on and `ce.json` tabulates.
CE = aggregation("cross_entropy", target="cf_answer")


def _width() -> int:
    """tiny-random's residual width — read off the loaded bundle, since the
    registry learns this model's static config from its HF config at load."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    return component_width(load_model(TINY_LLAMA).info, "block_output")


def _scores() -> list[dict]:
    # unit i scores (i * 7) mod width, so the ranking is a fixed permutation
    width = _width()
    return [{"unit": i, "value": float((i * 7) % width)} for i in range(width)]


def _top(rows: list[dict], keep: int) -> list[int]:
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]["value"], i))
    return sorted(order[:keep])


def _loader(rows: list[dict]):
    raw = json.dumps(rows).encode()
    return lambda _path: (rows, raw)


def test_the_executor_starts_the_gate_where_the_table_says_and_the_fit_records_it():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    rows = _scores()
    doc = clamp_dbm_doc(lr=1e-3)
    doc["method"]["featurizers"]["gate"]["init"] = {
        "from_scores": {"file_path": "scan/scores.json", "keep": KEEP}
    }
    executor = executor_for(
        doc,
        load_model(TINY_LLAMA),
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        load_table=_loader(rows),
    )
    gate = executor.stage("gate")
    kept = _top(rows, KEEP)
    width = _width()
    assert gate.hard_mask().nonzero().flatten().tolist() == kept
    assert gate.theta.tolist() == [1.0 if i in kept else 0.0 for i in range(width)]
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    record = outcome.diagnostics["gate"]["init_from_scores"]
    assert record == {
        "file_path": "scan/scores.json",
        "units": width,
        "keep": KEEP,
        "kept_units": kept,
    }
    assert "init_fill" not in outcome.diagnostics["gate"]
    # a small lr: the fit moved, and stayed near the ranking it started from
    theta = outcome.stages["gate"].theta.detach()
    start = torch.tensor([1.0 if i in kept else 0.0 for i in range(width)])
    assert not torch.equal(theta, start)
    assert torch.all((theta - start).abs() < 0.1)


def test_an_untrained_ranking_gate_is_a_legal_document_and_applies_as_the_split():
    """No `train`: the gate is not fitted, not loaded, and still a mask —
    the ranking's top-``keep`` set, applied. That is the baseline every
    trained DBM is compared against, and it needs no bundle on disk."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    rows = _scores()
    doc = clamp_dbm_doc()
    doc["method"]["featurizers"]["gate"]["init"] = {
        "from_scores": {"file_path": "scan/scores.json", "keep": KEEP}
    }
    del doc["method"]["train"]
    doc["method"]["save"] = [doc["method"]["save"][0]]
    validate_document(parse_document(in_order(doc)), engine_is_local=True)
    executor = executor_for(
        doc,
        load_model(TINY_LLAMA),
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        load_table=_loader(rows),
    )
    gate = executor.stage("gate")
    assert not gate.training
    assert gate.hard_mask().nonzero().flatten().tolist() == _top(rows, KEEP)
    x = torch.arange(_width(), dtype=torch.float32)
    kept, err = gate.featurize(x)
    assert kept.nonzero().flatten().tolist() == _top(rows, KEEP)


# -- the CLI: the table resolved through the artifact store ----------------- #


def _document(keep: int) -> dict:
    return {
        "header": {
            "protocol_version": "4",
            "description": "a DBM gate started from a score table and fitted for one epoch at a negligible lr",
        },
        "model": {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "masked": {"input": "base", "reads": ["logits"], "writes": ["mask"]},
            },
            "sites": {
                "target": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                "gate": {
                    "kind": "gate",
                    "parametrization": "clamp",
                    "init": {
                        "from_scores": {"file_path": "scan/scores.json", "keep": keep}
                    },
                }
            },
            "reads": {
                "v_cf": {"site": "target", "pos": -1, "featurizer": "gate"},
                "logits": {"site": "lm_head", "pos": -1},
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
                "objective": [[1.0, term("logits", "masked", dict(CE))]],
                "params": ["gate"],
                "optimizer": {"name": "adamw", "lr": 1e-9, "weight_decay": 0.0},
                "steps": {"epochs": 1},
                "batch": {"pairs": 2},
                "seed": 0,
            },
            "save": [
                saved("logits", "masked", "ce.json", dict(CE)),
                {"value": "gate", "site": "target", "file_path": "gate.safetensors"},
            ],
        },
    }


def test_the_cli_resolves_the_table_through_the_artifact_store(tmp_path: Path) -> None:
    rows = _scores()
    artifacts = tmp_path / "artifacts"
    (artifacts / "scan").mkdir(parents=True)
    (artifacts / "scan" / "scores.json").write_text(json.dumps(rows))
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "ranking.json").write_text(json.dumps(_document(KEEP)))
    workflow = {
        "version": "1",
        "description": "a gate from a ranking",
        "output_dir": "ranking",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": str(documents / "ranking.json"),
            }
        },
    }
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(workflow))
    out = tmp_path / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(wf),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    run = out / "ranking" / "fit"
    theta = load_file(str(run / "gate.safetensors"))["theta"]
    kept = _top(rows, KEEP)
    # lr 1e-9 for one epoch: the saved mask is the ranking's top set
    assert (theta > 0.5).nonzero().flatten().tolist() == kept
    (record,) = json.loads((run / "fit_diagnostics.json").read_text())
    scores = record["featurizers"]["gate"]["init_from_scores"]
    assert scores["kept_units"] == kept
    assert scores["file_path"] == "scan/scores.json"
    # the table's bytes are in the identity the run stamped: the step's
    # document digest is the digest of the canonical form resolved against
    # this artifact root, whose `init.from_scores.content_digest` is the
    # table's sha256 — a different table would be a different experiment
    from transformers import AutoConfig

    from causalab.protocol.schema.explicit import canonicalize, digest
    from causalab.protocol.registry import model_info_from_hf_config
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

    info = model_info_from_hf_config(TINY_LLAMA, AutoConfig.from_pretrained(TINY_LLAMA))
    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
        model_info=lambda key: info,
    )
    canonical = canonicalize(_document(KEEP), env)
    stamped = canonical["method"]["featurizers"]["gate"]["init"]["from_scores"]
    assert (
        stamped["content_digest"]
        == hashlib.sha256((artifacts / "scan" / "scores.json").read_bytes()).hexdigest()
    )
    step = json.loads((run / "_step.json").read_text())
    assert step["document_digest"] == digest(canonical)
