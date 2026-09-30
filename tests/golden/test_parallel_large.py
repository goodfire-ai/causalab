"""The parallel golden on the model that fits no card
(``docs/model_parallelism.md`` §10.6 "The large model", §11):
``meta-llama/Llama-3.1-70B`` — 131.42 GiB of bf16 weights — through the CLI
on eight H100s, every run a ``causalab run … --device cuda`` subprocess
(``tests/golden/_parallel/large.py``, the documents and the rules).

There is no world-1 run: its resident weights exceed one card. The
reference is the recorded **oracle** geometry ``pp=4``; ``pp=8`` is held byte for
byte against it (the pairwise identity of the exact set), ``tp=4``,
``tp=2,pp=2`` and ``tp=8`` within the band the record pins — the inference
document (``large``) and the DAS fit (``das_large``, ``pp=4`` / ``pp=8``
exact, ``tp=4`` banded). Every rank's load report obeys the several-axes
loader rule with ``bytes_unowned`` 0, and every rank's measured peak sits
within the pre-flight's estimated footprint and above its resident weights;
the ratio is printed and written into the record, not asserted to 5 %
(§10.6: no fitting geometry of this model runs near the card).

Each case skips by name below the devices its world needs (the oracle's
included); a case spelled ``<geometry>@2x4`` is the two-node ``torchrun``
run the coordinator produces with the printed commands into the kept root
and skips, naming them, while it is absent. Set
``CAUSALAB_PARALLEL_GOLDENS_ROOT`` to run under (and resume from) a kept
root. The record's ``large`` / ``das_large`` blocks are captured by
``tests/golden/update_parallel_goldens.py --only large,das_large
--i-have-reviewed-the-diff --keep ROOT``, reading the 70B from the Hub cache
``HF_HUB_CACHE`` names (huggingface_hub's default when unset); until they
land the band replays skip, naming the command.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.golden import _parallel as par
from tests.golden._parallel import large, measure

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="needs four CUDA devices (eight for the world-8 cases)",
    ),
]

ROOT_VARIABLE = "CAUSALAB_PARALLEL_GOLDENS_ROOT"
DEVICES = torch.cuda.device_count()

CAPTURE = (
    "HF_HUB_OFFLINE=1 uv run python "
    "tests/golden/update_parallel_goldens.py --only large,das_large "
    "--i-have-reviewed-the-diff --keep ROOT"
)


def _case(document: par.Document, case: str) -> Any:
    reason = large.skip_reason(DEVICES, case, document)
    marks = [pytest.mark.skipif(True, reason=reason)] if reason else []
    return pytest.param(document.name, case, marks=marks, id=f"{document.name}-{case}")


def _cases(kind: str) -> list[Any]:
    out: list[Any] = []
    for document in large.DOCUMENTS.values():
        for case in getattr(document, kind):
            if case != document.oracle:
                out.append(_case(document, case))
        for case in large.TWO_NODE_CASES:
            geometry = large.geometry_of_case(case)
            if geometry in getattr(document, kind) and geometry != document.oracle:
                out.append(_case(document, case))
    return out


# --------------------------------------------------------------------------- #
# the runs: authored once per document, each geometry on demand
# --------------------------------------------------------------------------- #


class Runs:
    """The documents' runs under one root: the oracle once, every other
    geometry on first request, one world resident at a time; a two-node
    case picked up from the kept root or skipped by name."""

    def __init__(self, root: Path, kept: Path | None) -> None:
        self.root = root
        self.kept = kept
        self._authored: dict[str, Path] = {}
        self._outputs: dict[tuple[str, str], Path] = {}

    def document(self, name: str) -> par.Document:
        return large.DOCUMENTS[name]

    def authored(self, name: str) -> Path:
        if name not in self._authored:
            document = self.document(name)
            base = self.root / name
            base.mkdir(parents=True, exist_ok=True)
            self._authored[name] = document.author(base, par.realization_of(document))
        return self._authored[name]

    def run(self, name: str, case: str) -> Path:
        document = self.document(name)
        if large.is_two_node(case):
            reason = large.two_node_skip(self.kept, document, case)
            if reason:
                pytest.skip(reason)
            assert self.kept is not None
            return large.two_node_out(self.kept, document, case)
        key = (name, case)
        if key not in self._outputs:
            out = self.root / name / par.out_name(case)
            self._outputs[key] = par.run(
                self.authored(name),
                out,
                "--parallel",
                case,
                realization=par.realization_of(document),
                argv=document.argv,
                recorded=document.recorded,
            )
        return self._outputs[key]

    def reference(self, name: str) -> Path:
        oracle = self.document(name).oracle
        assert oracle is not None
        return self.run(name, oracle)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    kept = os.environ.get(ROOT_VARIABLE)
    root = Path(kept) if kept else tmp_path_factory.mktemp("parallel-large")
    root.mkdir(parents=True, exist_ok=True)
    return Runs(root, Path(kept) if kept else None)


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
            f"the parallel goldens carry no capture of {name!r}: run `{CAPTURE}`"
        )
    return record["documents"][name]


def _launcher(case: str) -> str:
    return "joined" if large.is_two_node(case) else "spawned"


def _measured(runs: Runs, name: str, case: str) -> measure.Measured:
    document = runs.document(name)
    return document.measure(
        runs.reference(name), runs.run(name, case), par.realization_of(document)
    )


# --------------------------------------------------------------------------- #
# exact: every pipeline is the oracle, byte for byte
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, case", _cases("exact"))
def test_an_exact_geometry_is_byte_identical_to_the_oracle(
    runs: Runs, name: str, case: str
) -> None:
    document = runs.document(name)
    reference, parallel = runs.reference(name), runs.run(name, case)
    assert par.exact_differences(reference, parallel) == []
    assert (
        par.receipts_agree(
            reference,
            parallel,
            large.geometry_of_case(case),
            reference_geometry=document.oracle,
            launcher=_launcher(case),
        )
        == []
    )
    measured = _measured(runs, name, case)
    assert set(measured.classes) == set(document.classes)
    assert measured.worst() == 0.0, measured.classes


def test_the_inference_sweep_puts_its_two_points_on_different_stages(
    runs: Runs,
) -> None:
    receipt = par.receipt(runs.reference("large"))
    assert len(receipt["points"]) == 2
    layers = large.inference.sweep(par.LARGE)
    assert layers == (3, 43)
    # pp=4: 20 layers a stage — layer 3 on stage 0, layer 43 on stage 2
    assert {layer // 20 for layer in layers} == {0, 2}


# --------------------------------------------------------------------------- #
# banded: the tensor geometries within the record
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, case", _cases("banded"))
def test_a_tensor_geometry_lands_within_the_recorded_band(
    runs: Runs, record: dict[str, Any], name: str, case: str
) -> None:
    document = runs.document(name)
    geometry = large.geometry_of_case(case)
    reference, parallel = runs.reference(name), runs.run(name, case)
    assert (
        par.receipts_agree(
            reference,
            parallel,
            geometry,
            reference_geometry=document.oracle,
            launcher=_launcher(case),
        )
        == []
    )
    measured = _measured(runs, name, case)
    realization = par.realization_of(document)
    fresh = par.make_record(
        None,
        par.A3B,
        {
            name: par.document_record(
                document, realization, {geometry: measured}, {}, with_context=False
            )
        },
    )
    print(
        case,
        {
            k: {f: v for f, v in e.items() if f != "files"}
            for k, e in fresh["documents"][name]["geometries"][geometry].items()
        },
    )
    block = _captured(record, name)
    if geometry not in block["geometries"]:
        pytest.skip(f"{name} {geometry} is not in the record yet: run `{CAPTURE}`")
    committed = {
        **record,
        "documents": {
            name: {**block, "geometries": {geometry: block["geometries"][geometry]}}
        },
    }
    assert par.compare_records(committed, fresh, name) == []


@pytest.mark.parametrize("name", sorted(large.DOCUMENTS))
def test_the_record_block_is_this_document_and_obeys_the_rules(
    record: dict[str, Any], name: str
) -> None:
    """The block names the oracle, the realization and the document; its
    exact set is the compared exact geometries at 0.0, its banded entries
    obey the band rule; its estimate is the census's arithmetic and its
    recorded peaks obey the memory rule against it."""
    block = _captured(record, name)
    document = large.DOCUMENTS[name]
    realization = par.realization_of(document)
    assert block["oracle"] == document.oracle == "pp=4"
    assert block["realization"] == {
        "model": par.LARGE_MODEL,
        "dtype": "bf16",
        "device": "cuda",
    }
    assert block["document"] == document.describe(realization)
    assert set(block["exact"]) == {g for g in document.exact if g != document.oracle}
    assert all(worst == 0.0 for worst in block["exact"].values())
    assert set(block["geometries"]) == set(document.banded) | set(block["inexact"])
    for classes in block["geometries"].values():
        assert set(classes) == set(document.classes)
        for kind, entry in classes.items():
            assert entry["band"] == par.entry_band(kind, entry)
    assert set(block["load"]) == set(block["memory"]) == set(document.geometries)
    assert set(block["estimate"]) == set(document.geometries)
    for geometry in document.geometries:
        assert block["estimate"][geometry] == large.estimate(geometry), geometry
        assert (
            par.memory_problems(block["estimate"][geometry], block["memory"][geometry])
            == []
        )
    assert par.compare_records(record, record, name) == []


# --------------------------------------------------------------------------- #
# the loader: every rank at its axis fraction, one copy, nothing unowned
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, case", _cases("geometries"))
def test_every_rank_reads_its_axis_fraction_and_holds_one_copy(
    runs: Runs, name: str, case: str
) -> None:
    document = runs.document(name)
    geometry = large.geometry_of_case(case)
    reports = par.load_reports(runs.run(name, case), par.world_of(geometry))
    assert document.loader_problems(geometry, reports) == []
    for report in reports:
        assert report["bytes_unowned"] == 0, (case, report["rank"])
    print(case, {r["rank"]: sum(r["bytes_requested"].values()) for r in reports})


@pytest.mark.parametrize("case", ["pp=8", "tp=2,pp=2"])
def test_the_stages_together_hold_the_whole_model(runs: Runs, case: str) -> None:
    """The stages of a pipeline name a partition of the parameters a
    tensor-parallel rank names whole (every parameter of the model), and
    read the same bytes off disk in total."""
    from causalab.protocol.parallel import MeshLayout

    from tests.golden import _parallel_worlds as worlds

    reason = large.skip_reason(DEVICES, case, large.INFERENCE)
    if reason:
        pytest.skip(reason)
    geometry = worlds.geometry_of(case)
    stages = par.load_reports(runs.run("large", case), geometry.world)
    whole = par.load_reports(runs.run("large", "tp=4"), 4)[0]["bytes_on_disk"]
    assert worlds.stage_names(case, stages) == set(whole)
    layout = MeshLayout(geometry)
    one_per_stage: dict[int, dict[str, int]] = {}
    for report in stages:
        stage = layout.rank_in(report["rank"], "pipeline")
        one_per_stage.setdefault(stage, report["bytes_on_disk"])
    assert len(one_per_stage) == geometry.pipeline
    assert sum(sum(v.values()) for v in one_per_stage.values()) == sum(whole.values())


# --------------------------------------------------------------------------- #
# memory: the pre-flight's estimate meets the card
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, case", _cases("geometries"))
def test_every_rank_peaks_within_the_estimate_and_above_its_weights(
    runs: Runs, record: dict[str, Any], name: str, case: str
) -> None:
    geometry = large.geometry_of_case(case)
    world = par.world_of(geometry)
    peaks = measure.memory(runs.run(name, case) / par.GRADIENTS, world)
    assert set(peaks) == {f"rank{r}" for r in range(world)}
    assert all(
        p["device"] == f"cuda:{i % large.NODE_CARDS if large.is_two_node(case) else i}"
        for i, p in enumerate(peaks.values())
    )
    estimates = large.estimate(geometry)
    assert par.memory_problems(estimates, peaks) == []
    print(case, "estimate over reserved peak:", par.estimate_slack(estimates, peaks))
    if par.captured(record, name) and geometry in record["documents"][name].get(
        "memory", {}
    ):
        recorded = record["documents"][name]["memory"][geometry]
        assert par.memory_replay_problems(recorded, peaks) == []
