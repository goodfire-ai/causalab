"""Parallel runs on the real ``Qwen/Qwen3.6-35B-A3B`` against the world-1 run
(``docs/model_parallelism.md`` §7, §8, §10.6), for every document of the
parallel golden (``tests/golden/_parallel``): the inference document at
``dp=2`` / ``pp=2`` **bit for bit** and ``tp=2`` / ``ep=2`` within the band
``tests/golden/parallel_goldens.json`` pins by measurement; the DAS fit at
``attention_query`` at ``pp=2`` bit for bit and ``tp=2`` / ``dp=2:rows``
banded; the expert-neuron DBM fit at ``expert_activation`` at ``pp=2`` bit
for bit and ``ep=2`` banded — the fits' recorded gradients included; the
same DAS fit on the dense ``Qwen/Qwen3-4B-Instruct-2507`` in fp32
(``das_dense``, its own realization in its block) at ``tp=2`` and
``dp=2:rows`` within an fp32 band; and the loader reading exactly ``1 /
world`` of every sharded parameter's bytes on the real checkpoints.

Everything runs as ``causalab run … --device cuda`` subprocesses: this
process loads no model, one geometry's world is resident at a time, and
the record is captured by ``tests/golden/update_parallel_goldens.py
--i-have-reviewed-the-diff`` on two GPUs. Until that capture lands
for a document its band replay skips, naming the command; a committed
record the rule cannot read (an earlier format) skips by name too — the
drift goldens' rule — while the exact geometries and the loader's
fraction, which pin nothing measured, run regardless. Set
``CAUSALAB_PARALLEL_GOLDENS_ROOT`` to run under (and resume from) a kept
root instead of a fresh temporary one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from tests.golden import _parallel as par

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

ROOT_VARIABLE = "CAUSALAB_PARALLEL_GOLDENS_ROOT"

Runs = Callable[[str], dict[str, Path]]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    """The oracle and every geometry of a document, run once for the module
    on first request — one document's runs at a time, each geometry after
    the last has exited."""
    kept = os.environ.get(ROOT_VARIABLE)
    root = Path(kept) if kept else tmp_path_factory.mktemp("parallel")
    root.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Path]] = {}

    def get(name: str) -> dict[str, Path]:
        if name not in cache:
            cache[name] = par.run_all(root, par.DOCUMENTS[name])
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
            f"run `{par.CAPTURE_COMMAND}` and commit {par.RECORD.name}"
        )
    return record["documents"][name]


def _cases(kind: str) -> list[tuple[str, str]]:
    return [
        (document.name, geometry)
        for document in par.DOCUMENTS.values()
        for geometry in getattr(document, kind)
    ]


# --------------------------------------------------------------------------- #
# exact: no reduction moves
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, geometry", _cases("exact"))
def test_the_exact_geometries_are_bit_identical_to_world_one(
    runs: Runs, record: dict[str, Any], name: str, geometry: str
) -> None:
    explained = record.get("documents", {}).get(name, {}).get("inexact", {})
    if geometry in explained:
        pytest.skip(
            f"{name} {geometry} is banded, the record explains: {explained[geometry]}"
        )
    outputs = runs(name)
    solo, parallel = outputs["solo"], outputs[geometry]
    assert par.exact_differences(solo, parallel) == []
    assert par.receipts_agree(solo, parallel, geometry) == []
    document = par.DOCUMENTS[name]
    measured = document.measure(solo, parallel, par.realization_of(document))
    assert measured.worst() == 0.0, measured.classes


def test_the_inference_sweep_has_two_points_for_dp_to_shard(runs: Runs) -> None:
    assert len(par.receipt(runs("inference")["dp=2"])["points"]) == 2


# --------------------------------------------------------------------------- #
# banded: within the record
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, geometry", _cases("banded"))
def test_the_banded_geometries_land_within_the_recorded_band(
    runs: Runs, record: dict[str, Any], name: str, geometry: str
) -> None:
    _captured(record, name)
    outputs = runs(name)
    solo, parallel = outputs["solo"], outputs[geometry]
    assert par.receipts_agree(solo, parallel, geometry) == []
    document = par.DOCUMENTS[name]
    realization = par.realization_of(document)
    measured = document.measure(solo, parallel, realization)
    print(name, geometry, measured.classes)
    fresh = par.make_record(
        None,
        par.A3B,
        {
            name: par.document_record(
                document, realization, {geometry: measured}, {}, with_context=False
            )
        },
    )
    committed = {
        **record,
        "documents": {
            name: {
                **record["documents"][name],
                "geometries": {
                    geometry: record["documents"][name]["geometries"][geometry]
                },
            }
        },
    }
    assert par.compare_records(committed, fresh, name) == []


@pytest.mark.parametrize("name", sorted(par.DOCUMENTS))
def test_the_record_is_this_realization_and_obeys_the_band_rule(
    record: dict[str, Any], name: str
) -> None:
    block = _captured(record, name)
    assert record["model"] == par.MODEL and record["dtype"] == par.A3B.dtype
    assert record["device"] == par.A3B.device
    document = par.DOCUMENTS[name]
    realization = par.realization_of(document)
    assert block["document"] == document.describe(realization)
    if document.realization is None:
        assert "realization" not in block
    else:
        assert block["realization"] == {
            "model": realization.model,
            "dtype": realization.dtype,
            "device": realization.device,
        }
    assert set(block["geometries"]) == set(document.banded) | set(block["inexact"])
    assert set(block["exact"]) == {
        g for g in document.exact if g not in block["inexact"]
    }
    for classes in block["geometries"].values():
        assert set(classes) == set(document.classes)
        for kind, entry in classes.items():
            assert entry["band"] == par.entry_band(kind, entry)
    assert record["tolerance"]["justification"]
    assert block["context"]["transformers"] and block["context"]["torch"]


# --------------------------------------------------------------------------- #
# the loader: 1 / world of the sharded bytes on the real checkpoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, geometry", _cases("geometries"))
def test_the_loader_reads_one_over_world_of_every_sharded_parameter(
    runs: Runs, name: str, geometry: str
) -> None:
    reports = par.load_reports(runs(name)[geometry])
    assert par.load_problems(geometry, reports) == []


def test_the_two_pipeline_stages_together_hold_the_whole_model(runs: Runs) -> None:
    """Each stage's report covers its own parameters, whole; the two cover
    exactly the parameter set a tensor-parallel rank names (every parameter
    of the model)."""
    outputs = runs("inference")
    stages = par.load_reports(outputs["pp=2"])
    whole = par.load_reports(outputs["tp=2"])[0]["bytes_on_disk"]
    union = set(stages[0]["bytes_on_disk"]) | set(stages[1]["bytes_on_disk"])
    assert union == set(whole)
    on_disk = sum(sum(s["bytes_on_disk"].values()) for s in stages)
    assert on_disk == sum(whole.values())


@pytest.mark.parametrize("name", ["das", "dbm", "das_dense"])
def test_every_rank_of_a_fit_recorded_its_gradients_and_memory(
    runs: Runs, name: str
) -> None:
    from tests.golden._parallel import measure

    outputs = runs(name)
    assert set(measure.memory(outputs["solo"] / par.GRADIENTS, 1)) == {"rank0"}
    for geometry in par.DOCUMENTS[name].geometries:
        ranks = measure.memory(outputs[geometry] / par.GRADIENTS, par.WORLD)
        assert set(ranks) == {"rank0", "rank1"}
        assert all(r["device"] == f"cuda:{i}" for i, r in enumerate(ranks.values()))
        assert all(r["peak_bytes_allocated"] > 0 for r in ranks.values())
