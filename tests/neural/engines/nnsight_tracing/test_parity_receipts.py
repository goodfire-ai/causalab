"""The run receipt and the saved-file provenance through both engines
(cases P15, R1, R2).

The receipt (``protocol.json``) is written by the shared execution layer from
what the executor reports, so every field is a parity claim except the ones
that record an engine property. Those are named, one comment each, in
``tests._helpers.engines.KNOWN_RECORD_DIFFERENCES`` and its two metadata
siblings. This module asserts the allowed asymmetries *as the current state*
(so the allow-list cannot quietly widen) and everything else equal:

* **P15** — the reference engine tallies each write member's firings per
  forward group and the receipt records them; the nnsight executor keeps no
  tally and records nothing. Asserted on a document with several
  writes, one of them a band site lowered to two members.
* **R1** — campaign digest, canonical document, per-point digests,
  ``scoring``, ``cells`` and the denominator equal; the saved tensors'
  ``__metadata__`` (the identity stamp and the ``entries`` table) equal but
  for ``engine`` and the attention backend each loader realized.
  Receipts are opt-in: with ``record`` off, neither engine writes
  the receipt or the event stream, and the two file sets still match.
* **R2** — ``run_protocol`` and ``causalab run --engine nnsight --record``
  are one run: byte-identical output (the event stream aside), and that run
  is the reference engine's modulo the known differences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata
from causalab.io.events import EVENTS_FILE
from causalab.io.tensor_files import load_file
from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.schema import PROTOCOL_VERSION

from tests._helpers import engines
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.protocol._docs import aggregation, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES

pytestmark = pytest.mark.smoke

INTERCHANGE = CORPUS_DIR / "02_interchange_im.json"
#: tiny-random is two layers deep, so the shipped L18 site is retargeted
OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}


# --------------------------------------------------------------------------- #
# P15 / R1 — several writes, a band among them
# --------------------------------------------------------------------------- #


def _band_writes_doc() -> dict[str, Any]:
    """Two swaps from the counterfactual into base: the residual stream at
    both of tiny-random's layers as one band site (``layers: [0, 1]``,
    lowered to two members before resolution) and the attention output at
    layer 1; the patched and the clean answer logits saved beside the IIA."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["logits_clean"]},
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_band", "v_attn"],
                },
                "patched": {
                    "input": "base",
                    "reads": ["logits"],
                    "writes": ["band_patch", "attn_patch"],
                },
            },
            "sites": {
                "band": {"component": "block_output", "layers": [0, 1]},
                "attn": {"component": "attention_output", "layers": [1]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_band": {"site": "band", "pos": -1},
                "v_attn": {"site": "attn", "pos": -1},
                "logits": {"site": "lm_head", "pos": -1},
                "logits_clean": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                "band_patch": {"site": "band", "pos": -1, "do": {"swap": "v_band"}},
                "attn_patch": {"site": "attn", "pos": -1, "do": {"swap": "v_attn"}},
            },
            "save": [
                saved(
                    "logits",
                    "patched",
                    "iia.json",
                    aggregation("match", expected="cf_answer"),
                ),
                saved("logits", "patched", "logits.safetensors"),
                saved("logits_clean", "original", "logits_clean.safetensors"),
            ],
        },
    }


@pytest.fixture(scope="module")
def band_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    base = tmp_path_factory.mktemp("band")
    env = engines.corpus_env(base / "artifacts")
    return engines.run_both(_band_writes_doc(), env, base / "out", record=True)


def _record(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def test_the_writes_landed_on_both_engines(band_runs):
    """Anti-vacuity before any receipt claim: each engine's patched logits
    differ from its own clean ones."""
    for run_dir in (band_runs.hooks_dir, band_runs.trace_dir):
        patched = load_file(str(run_dir / "logits.safetensors"))["logits"]
        clean = load_file(str(run_dir / "logits_clean.safetensors"))["logits_clean"]
        assert not torch.allclose(patched, clean, atol=engines.ATOL), run_dir.name


def test_p15_fire_counts_are_the_one_receipt_asymmetry(band_runs):
    """📐 FINDING (P15), asserted as the current state: the reference
    engine records one fire per write member per forward group — the band's
    two lowered members and the attention swap — under the patched group's
    label; the nnsight executor tallies nothing, so its receipt's ``fires``
    block is empty. Nothing else in the two receipts differs beyond the
    measured-bound nulls."""
    hooks, trace = _record(band_runs.hooks_dir), _record(band_runs.trace_dir)
    digest = hooks["document_digest"]
    assert trace["document_digest"] == digest
    # every member of the lowered document fired once in the patched group:
    # the band as its two per-layer members, the attention swap as itself
    assert hooks["fires"] == {
        digest: {
            "patched on base": {
                "attn_patch": 1,
                "band_patch[layers=0]": 1,
                "band_patch[layers=1]": 1,
            }
        }
    }
    assert trace["fires"] == {}  # present and empty: no tally, not "no block"
    # the allow-list is exactly what differs
    stripped_hooks = engines.strip_known_record_differences(hooks)
    stripped_trace = engines.strip_known_record_differences(trace)
    assert stripped_hooks == stripped_trace
    for record in (hooks, trace):
        record.pop("fires")
        record["execution"].pop("batch_rows")
        record["execution"].pop("fit_rows")
    assert hooks == trace


def test_r1_the_receipt_agrees_field_by_field(band_runs):
    """The claims named one by one, so a record-format change cannot drop
    one silently: campaign digest, canonical document, per-point digests and
    coords, ``scoring``, the cells; ``execution.ragged`` where a document
    lands a ragged write (this one does not — the same absence on both
    sides)."""
    hooks, trace = _record(band_runs.hooks_dir), _record(band_runs.trace_dir)
    for key in ("document_digest", "canonical", "scoring", "points"):
        assert key in hooks and key in trace, key
        assert hooks[key] == trace[key], key
    assert set(hooks) == set(trace)
    assert [p["digest"] for p in hooks["points"]] == [
        p["digest"] for p in trace["points"]
    ]
    assert band_runs.hooks_result.cells == band_runs.trace_result.cells
    assert (
        band_runs.hooks_result.denominator.render()
        == band_runs.trace_result.denominator.render()
        == "3 / 3 eligible"
    )
    assert "ragged" not in hooks["execution"] and "ragged" not in trace["execution"]
    band_runs.compare()


def test_r1_without_record_neither_engine_writes_a_receipt_or_a_stream(tmp_path):
    """Both sidecars are opt-in. The same document with ``record`` off
    writes no ``protocol.json`` and no ``events.jsonl`` on either engine. The
    saved files are the same set on both sides and agree as before, and the
    set is the recorded run's set without the two sidecars."""
    env = engines.corpus_env(tmp_path / "artifacts")
    runs = engines.run_both(_band_writes_doc(), env, tmp_path / "out", record=False)
    sidecars = {RUN_RECORD_NAME, EVENTS_FILE}
    names = {}
    for run_dir in (runs.hooks_dir, runs.trace_dir):
        written = {p.name for p in run_dir.rglob("*") if p.is_file()}
        assert not written & sidecars, (run_dir.name, sorted(written & sidecars))
        names[run_dir.name] = written
    assert names["hooks"] == names["trace"]
    assert names["hooks"] == {
        "iia.json",
        "logits.safetensors",
        "logits_clean.safetensors",
    }
    assert sorted(runs.hooks_result.files) == sorted(runs.trace_result.files)
    runs.compare()


def test_r1_the_saved_tensors_carry_the_same_identity_and_entries(band_runs):
    """The ``__metadata__`` header of every saved tensor file, explicitly:
    the identity stamp equal but for ``engine``; the ``entries`` table —
    slot, coords, site, the data it was taken on — equal but for the
    attention backend each loader realized (the R1 finding below)."""
    names = sorted(p.name for p in band_runs.hooks_dir.glob("*.safetensors"))
    assert names == ["logits.safetensors", "logits_clean.safetensors"]
    for name in names:
        hooks = read_safetensors_metadata(band_runs.hooks_dir / name)
        trace = read_safetensors_metadata(band_runs.trace_dir / name)
        assert hooks is not None and trace is not None
        assert set(hooks) == set(trace), name
        assert hooks["engine"] == "pytorch_hooks" and trace["engine"] == "nnsight"
        for key in set(hooks) - {"engine", "entries"}:
            assert hooks[key] == trace[key], (name, key)
        hooks_entries = json.loads(hooks["entries"])
        trace_entries = json.loads(trace["entries"])
        assert set(hooks_entries) == set(trace_entries) == {name.split(".")[0]}
        for entry, record in hooks_entries.items():
            other = trace_entries[entry]
            assert set(record) == set(other), (name, entry)
            for field in set(record) - engines.KNOWN_ENTRY_RECORD_DIFFERENCES:
                assert record[field] == other[field], (name, entry, field)
            # the record names the slot it describes; the old `produced_by`
            # field no longer exists anywhere under `causalab/`
            assert record["slot"] == entry
            assert record["trained_on"] == "weekdays/data#train"


def test_r1_the_attention_backend_stamp_is_the_one_entries_asymmetry(
    band_runs, trace_llama, tmp_path
):
    """📐 FINDING (R1), asserted as the current state: each entry's
    ``loaded_attn_implementation`` is ``eager`` from the reference engine
    (its loader pins eager) and ``sdpa`` from the nnsight engine (its loader
    keeps the checkpoint's default). Handed the eager-pinned fixture
    bundle instead, the nnsight engine stamps ``eager`` and the two entries
    tables are identical — so the asymmetry is the loaders', not the
    executors'."""
    for name in ("logits", "logits_clean"):
        hooks = json.loads(
            read_safetensors_metadata(band_runs.hooks_dir / f"{name}.safetensors")[
                "entries"
            ]
        )[name]
        trace = json.loads(
            read_safetensors_metadata(band_runs.trace_dir / f"{name}.safetensors")[
                "entries"
            ]
        )[name]
        assert hooks["loaded_attn_implementation"] == "eager"
        assert trace["loaded_attn_implementation"] == "sdpa"

    env = engines.corpus_env(tmp_path / "artifacts")
    compiled = engines.compile_for_both(_band_writes_doc(), env)
    out = tmp_path / "eager"
    run_protocol(compiled, env, NnsightEngine(bundle=trace_llama), out, record=True)
    for name in ("logits", "logits_clean"):
        hooks = json.loads(
            read_safetensors_metadata(band_runs.hooks_dir / f"{name}.safetensors")[
                "entries"
            ]
        )
        eager = json.loads(
            read_safetensors_metadata(out / f"{name}.safetensors")["entries"]
        )
        assert hooks == eager, name
    # the receipt then differs only in `model_source` (a caller-owned bundle)
    # beside the allow-listed fields
    record = _record(out)
    assert record["execution"]["model_source"] == "caller"
    engines.compare_records(
        _record(band_runs.hooks_dir), record, extra_ignored=("execution.model_source",)
    )


# --------------------------------------------------------------------------- #
# R2 — run_protocol == causalab run, on the nnsight engine
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def nnsight_cli_and_api(tmp_path_factory: pytest.TempPathFactory):
    """The corpus interchange through ``causalab run --engine nnsight
    --record`` and through ``run_protocol(..., NnsightEngine(), record=True)``,
    plus the reference engine's run of the same compiled document, for the
    cross-engine half."""
    base = tmp_path_factory.mktemp("r2")
    env = engines.corpus_env(base / "artifacts")
    via_cli, via_api, via_hooks = base / "cli", base / "api", base / "hooks"
    argv = [
        "run",
        str(INTERCHANGE),
        "--data-root",
        str(FIXTURES / "data"),
        "--artifacts-root",
        str(base / "artifacts"),
        "--out",
        str(via_cli),
        "--engine",
        "nnsight",
        "--record",
    ]
    for path, value in OVERRIDES.items():
        argv += ["--set", f"{path}={value}"]
    status = main(argv)
    compiled = engines.compile_for_both(INTERCHANGE, env, overrides=OVERRIDES)
    result = run_protocol(compiled, env, NnsightEngine(), via_api, record=True)
    run_protocol(compiled, env, PytorchHooksEngine(), via_hooks, record=True)
    return status, via_cli, via_api, via_hooks, result


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != EVENTS_FILE
    }


def test_r2_the_cli_and_the_api_are_one_run_on_nnsight(nnsight_cli_and_api):
    status, via_cli, via_api, _, result = nnsight_cli_and_api
    assert status == 0
    assert result.files, "run_protocol reported no saved files"
    cli, api = _files(via_cli), _files(via_api)
    assert RUN_RECORD_NAME in cli, "--record wrote no receipt"
    assert set(cli) == set(api), (
        f"only via the CLI: {sorted(set(cli) - set(api))}; "
        f"only via run_protocol: {sorted(set(api) - set(cli))}"
    )
    differing = sorted(name for name in cli if cli[name] != api[name])
    assert not differing, f"same document, different bytes: {differing}"
    record = json.loads((via_cli / RUN_RECORD_NAME).read_text())
    assert record["execution"]["batch_rows"] is None  # the engine that ran
    assert record["execution"]["fit_rows"] is None
    for manifest_path, disk_path in result.files.items():
        assert Path(disk_path).is_file(), manifest_path
        assert via_api in Path(disk_path).parents


def test_r2_that_run_is_the_reference_engines_modulo_the_known_differences(
    nnsight_cli_and_api,
):
    _, via_cli, _, via_hooks, _ = nnsight_cli_and_api
    engines.compare_run_dirs(via_hooks, via_cli)
