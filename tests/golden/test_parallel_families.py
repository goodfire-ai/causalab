"""The second dense families on a card (``docs/model_parallelism.md`` §10.6
"the second-family rows", §11): the parallel golden's inference document on
``google/gemma-2-9b`` and ``meta-llama/Llama-3.1-8B`` in bf16
(``tests/golden/_parallel/families.py``), each at world 1 and at ``dp=2``
and ``tp=2`` — the llama at ``pp=2`` too — as ``causalab run … --device
cuda`` subprocesses, one geometry's world resident at a time, against the
world-1 run of the same document: ``dp`` (and llama's ``pp``) **byte for
byte**, ``tp`` within the format-2 band ``tests/golden/parallel_goldens.json``
pins per document from a measurement on the node; the loader at exactly
``1 / world`` of every sharded parameter's bytes and the residency rule
clean on every rank; gemma2's ``pp=2`` **refused by name** at the load
(the tied head, §6.6) and its ``dry-run … --parallel tp=2`` naming the
declined vocabulary row.

Until a family's capture lands its band replay skips, naming the capture
command (``update_parallel_goldens.py --only inference_gemma2_9b,inference_llama31_8b``);
the exact geometries, the refusal, the dry-run and the loader run
regardless. Set ``CAUSALAB_PARALLEL_GOLDENS_ROOT`` to run under (and
resume from) a kept root. The CPU guard — the entries, the registry row
against the cached config, the dry-run's word, the harness on the tiny
Llama over gloo — is ``tests/golden/test_parallel_families_record.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from tests.golden import _parallel as par
from tests.golden._parallel import families

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

ROOT_VARIABLE = "CAUSALAB_PARALLEL_GOLDENS_ROOT"

Runs = Callable[[str], dict[str, Path]]


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    kept = os.environ.get(ROOT_VARIABLE)
    path = Path(kept) if kept else tmp_path_factory.mktemp("families")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def runs(root: Path) -> Runs:
    """The oracle and every geometry of a family document, once for the
    module on first request, on the document's own realization."""
    cache: dict[str, dict[str, Path]] = {}

    def get(name: str) -> dict[str, Path]:
        if name not in cache:
            cache[name] = par.run_all(root, families.FAMILIES[name])
        return cache[name]

    return get


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    committed = par.load_record()
    try:
        par.check_format(committed)
    except par.StaleRecord as stale:
        pytest.skip(str(stale))
    return committed


def _captured(record: dict[str, Any], name: str) -> dict[str, Any]:
    if not par.captured(record, name):
        pytest.skip(
            f"the parallel goldens carry no capture of {name!r}: on the 2-GPU node "
            f"run `{par.CAPTURE_COMMAND} --only {name}` and commit {par.RECORD.name}"
        )
    return record["documents"][name]


def _cases(kind: str) -> list[tuple[str, str]]:
    return [
        (document.name, geometry)
        for document in families.FAMILIES.values()
        for geometry in getattr(document, kind)
    ]


@pytest.mark.parametrize("name, geometry", _cases("exact"))
def test_the_exact_geometries_are_bit_identical_to_world_one(
    runs: Runs, name: str, geometry: str
) -> None:
    outputs = runs(name)
    solo, parallel = outputs["solo"], outputs[geometry]
    assert par.exact_differences(solo, parallel) == []
    assert par.receipts_agree(solo, parallel, geometry) == []
    document = families.FAMILIES[name]
    measured = document.measure(solo, parallel, par.realization_of(document))
    assert measured.worst() == 0.0, measured.classes
    assert set(measured.classes) == set(document.classes)


@pytest.mark.parametrize("name, geometry", _cases("banded"))
def test_the_banded_geometries_land_within_the_recorded_band(
    runs: Runs, record: dict[str, Any], name: str, geometry: str
) -> None:
    _captured(record, name)
    outputs = runs(name)
    solo, parallel = outputs["solo"], outputs[geometry]
    assert par.receipts_agree(solo, parallel, geometry) == []
    document = families.FAMILIES[name]
    realization = par.realization_of(document)
    measured = document.measure(solo, parallel, realization)
    print(name, geometry, measured.classes)
    assert (
        par.replay_problems(record, document, realization, geometry, measured, par.A3B)
        == []
    )


@pytest.mark.parametrize("name", sorted(families.FAMILIES))
def test_the_record_holds_the_family_on_its_own_realization(
    record: dict[str, Any], name: str
) -> None:
    block = _captured(record, name)
    document = families.FAMILIES[name]
    realization = par.realization_of(document)
    assert record["model"] == par.MODEL  # the record's top stays the A3B's
    assert block["realization"] == {
        "model": realization.model,
        "dtype": "bf16",
        "device": "cuda",
    }
    assert block["document"] == document.describe(realization)
    assert set(block["geometries"]) == set(document.banded) | set(block["inexact"])
    assert set(block["exact"]) == {
        g for g in document.exact if g not in block["inexact"]
    }
    for classes in block["geometries"].values():
        assert set(classes) == set(document.classes)
        for kind, entry in classes.items():
            assert entry["band"] == par.entry_band(kind, entry)


@pytest.mark.parametrize("name, geometry", _cases("geometries"))
def test_the_loader_reads_one_over_world_and_the_residency_is_clean(
    runs: Runs, name: str, geometry: str
) -> None:
    reports = par.load_reports(runs(name)[geometry])
    assert par.load_problems(geometry, reports) == []
    for report in reports:
        assert report["bytes_unowned"] == 0


def test_the_vocabulary_is_whole_on_every_tensor_parallel_rank(runs: Runs) -> None:
    """§6.1, §11: the embedding (and gemma2's tied head) read whole under
    ``tp=2`` on both families; every other sharded parameter at ``1 / 2``."""
    for name, document in families.FAMILIES.items():
        for report in par.load_reports(runs(name)["tp=2"]):
            requested, on_disk = report["bytes_requested"], report["bytes_on_disk"]
            embeddings = [k for k in requested if "embed_tokens" in k]
            assert embeddings, name
            assert all(requested[k] == on_disk[k] for k in embeddings), name
            assert par.sharded_parameters(report), name


@pytest.mark.parametrize("name", sorted(families.REFUSED))
def test_a_tied_head_refuses_the_pipeline_by_name(root: Path, name: str) -> None:
    """gemma2 at ``pp=2``: the run exits non-zero before any forward, the
    refusal naming ``tie_word_embeddings`` (``sharding.place_stage``,
    §6.6); no receipt is left for a kept root to resume."""
    geometry, words = families.REFUSED[name]
    document = families.FAMILIES[name]
    realization = par.realization_of(document)
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    authored = document.author(base, realization)
    out = base / f"{par.out_name(geometry)}-refused"
    with pytest.raises(par.RunFailed) as failed:
        par.run(authored, out, "--parallel", geometry, realization=realization)
    assert words in str(failed.value)
    assert not (out / par.RECEIPT).exists()


def test_dry_run_names_the_declined_vocabulary_row_on_gemma2_alone(root: Path) -> None:
    for name, document in families.FAMILIES.items():
        base = root / name
        base.mkdir(parents=True, exist_ok=True)
        authored = document.author(base, par.realization_of(document))
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "causalab.cli",
                "dry-run",
                str(authored),
                "--engine",
                "pytorch_hooks",
                "--data-root",
                str(par.runs.DATA),
                "--artifacts-root",
                str(authored.parent),
                "--parallel",
                "tp=2",
            ],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        assert (
            "tp=2,ep=1 (world 2): accepted by the registry entry" in completed.stdout
        ), name
        assert (families.UNAPPLIED in completed.stdout) == (
            name == families.GEMMA2.name
        )
