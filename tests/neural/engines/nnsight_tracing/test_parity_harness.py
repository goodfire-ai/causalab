"""The parity harness itself (``tests/_helpers/engines.py``), proven on the
corpus interchange: one document, both engines, both seams.

Two claims. At the executor seam, ``both_executors`` builds the same rows
for both executors and a read agrees. At the engine seam, ``run_both``
runs one compiled document through ``run_protocol`` with each engine and
``compare_run_dirs`` finds the two output directories equal modulo the
listed differences. The list is exactly what it claims: the comparer rejects
a directory that differs anywhere else.

Two reads-first facts of protocol 4 are checked here too, because
every other parity module relies on them: a read listed by two models is two
values on both engines (``BoundRead``, ``test_bound_reads.py``), and the
un-intervened model may carry any name.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.io.tensor_files import load_file
from causalab.protocol import RUN_RECORD_NAME
from causalab.protocol.schema import PROTOCOL_VERSION, ReadRef

from tests._helpers import engines
from tests._helpers.a3b_sweep import read_doc
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.protocol._env import CORPUS_DIR

pytestmark = pytest.mark.smoke

DOCUMENT = CORPUS_DIR / "02_interchange_im.json"
#: tiny-random is two layers deep, so the shipped L18 site is retargeted
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}

BASE_TEXTS = ["the quick brown fox jumps", "a small red hen sits still"]
COUNTERFACTUAL_TEXTS = ["a slow grey cat naps all", "the big blue owl flies far"]


def test_both_executors_agree_on_a_boundary_read(hooks_llama, trace_llama):
    hooks, trace = engines.both_executors(
        read_doc("block_output", 1), hooks_llama, trace_llama, base_texts=BASE_TEXTS
    )
    a, b = hooks.read_value("r"), trace.read_value("r")
    assert a.shape == b.shape and a.shape[0] == len(BASE_TEXTS)
    assert (a - b).abs().max().item() <= engines.ATOL


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    base = tmp_path_factory.mktemp("harness")
    env = engines.corpus_env(base / "artifacts")
    return engines.run_both(DOCUMENT, env, base / "out", overrides=OVERRIDES)


def test_the_interchange_corpus_document_agrees_at_the_engine_seam(runs):
    """The first half of case M1 (``test_parity_metrics.py``): `logit_diff`
    and `match` tables, the receipt and the file set agree across engines on
    the corpus interchange."""
    assert runs.hooks_result.files and runs.trace_result.files
    assert sorted(runs.hooks_result.files) == sorted(runs.trace_result.files)
    runs.compare()


def test_the_comparer_is_not_vacuous(runs, tmp_path: Path):
    """A copy of the nnsight directory with one metric value nudged past
    ``ATOL`` and one receipt digest changed is rejected, naming the file."""
    import shutil

    nudged = tmp_path / "nudged"
    shutil.copytree(runs.trace_dir, nudged)
    table_path = nudged / "logit_diff.json"
    table = json.loads(table_path.read_text())
    table[0]["value"] += 10 * engines.ATOL
    table_path.write_text(json.dumps(table))
    with pytest.raises(AssertionError, match="logit_diff.json"):
        engines.compare_run_dirs(runs.hooks_dir, nudged)

    stale = tmp_path / "stale"
    shutil.copytree(runs.trace_dir, stale)
    record_path = stale / RUN_RECORD_NAME
    record = json.loads(record_path.read_text())
    record["document_digest"] = "0" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="document_digest"):
        engines.compare_run_dirs(runs.hooks_dir, stale)


def test_the_known_differences_are_the_only_ones(runs):
    """Anti-vacuity for the allow-list: the receipts differ *only* in what
    ``engines.KNOWN_RECORD_DIFFERENCES`` names, so the list cannot
    quietly absorb a new divergence. Today that is the reference engine's
    `fires` block, which the nnsight executor does not tally (see the
    comment on the entry)."""
    hooks = json.loads((runs.hooks_dir / RUN_RECORD_NAME).read_text())
    trace = json.loads((runs.trace_dir / RUN_RECORD_NAME).read_text())
    assert hooks != trace, "the allow-list is empty in effect; drop it"
    # the reference engine counted the one write's fire per group; the
    # nnsight engine recorded an empty tally
    assert hooks["fires"] == {
        hooks["document_digest"]: {"patched on base": {"patch": 1}}
    }
    assert trace["fires"] == {}
    hooks.pop("fires")
    trace.pop("fires")
    # this document authors no bound, so neither engine measured one here;
    # the entries stay listed for the documents that do (§8 `batch_rows`)
    for record in (hooks, trace):
        record["execution"].pop("batch_rows")
        record["execution"].pop("fit_rows")
    assert hooks == trace


def _shared_read_doc() -> dict[str, Any]:
    """Read ``r`` (block 1, last position) listed by the un-intervened base
    model and by ``patched``, which swaps the counterfactual's block-0
    residual in first. The write sits upstream of the read, so the two
    bindings of ``r`` must be two different tensors."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": {
            "base": {"dataset": "inline", "field": "input"},
            "counterfactual": {
                "dataset": "inline",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["r"]},
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "patched": {"input": "base", "reads": ["r"], "writes": ["patch"]},
            },
            "sites": {
                "early": {"component": "block_output", "layers": 0},
                "late": {"component": "block_output", "layers": 1},
            },
            "reads": {
                "v_cf": {"site": "early", "pos": -1},
                "r": {"site": "late", "pos": -1},
            },
            "writes": {"patch": {"site": "early", "pos": -1, "do": {"swap": "v_cf"}}},
            "save": [
                {"read": "r", "model": "original", "file_path": "r_clean.safetensors"},
                {"read": "r", "model": "patched", "file_path": "r_patched.safetensors"},
            ],
        },
    }


def test_one_read_listed_by_two_models_is_two_values_on_both_engines(
    hooks_llama, trace_llama
):
    """Protocol 4's reads-first shape: executor values are keyed by
    ``BoundRead(read, model)``. Each binding agrees across engines, and on
    each engine the two bindings differ (the write landed upstream)."""
    hooks, trace = engines.both_executors(
        _shared_read_doc(),
        hooks_llama,
        trace_llama,
        base_texts=BASE_TEXTS,
        counterfactual_texts=COUNTERFACTUAL_TEXTS,
    )
    values = {}
    for model in ("original", "patched"):
        ref = ReadRef("r", model)
        a, b = hooks.dense_value(ref), trace.dense_value(ref)
        assert a.shape == b.shape and a.shape[0] == len(BASE_TEXTS)
        diff = (a - b).abs().max().item()
        assert diff <= engines.ATOL, f"r on {model}: max abs diff {diff:.3e}"
        assert torch.equal(hooks.read_value(ref), a)
        values[model] = (a, b)
    for side in (0, 1):
        clean, patched = values["original"][side], values["patched"][side]
        assert (clean - patched).abs().max().item() > 100 * engines.ATOL


def _named_base_doc(name: str) -> dict[str, Any]:
    """The corpus interchange plus an un-intervened base model called
    ``name`` that reads the logits, saved as a tensor."""
    raw = copy.deepcopy(json.loads(DOCUMENT.read_text()))
    raw["method"]["intervened_models"][name] = {"input": "base", "reads": ["logits"]}
    raw["method"]["save"].append(
        {"read": "logits", "model": name, "file_path": "clean_logits.safetensors"}
    )
    return raw


def test_the_unintervened_model_may_carry_any_name(tmp_path: Path):
    """``original`` is an ordinary declarable name at protocol 4, and
    the un-intervened forward is the input's closure with no writes, whatever
    it is called. Under ``original`` and under ``clean`` both engines agree
    file by file, receipt and digests included. The name is part of the
    canonical document, so the digests differ across names; the saved
    tensors do not."""
    env = engines.corpus_env(tmp_path / "artifacts")
    logits = {}
    for name in ("original", "clean"):
        runs = engines.run_both(
            _named_base_doc(name), env, tmp_path / name, overrides=OVERRIDES
        )
        runs.compare()
        logits[name] = load_file(str(runs.hooks_dir / "clean_logits.safetensors"))
    assert sorted(logits["original"]) == sorted(logits["clean"])
    for key, value in logits["original"].items():
        assert torch.equal(value, logits["clean"][key]), key
