"""The parallel golden's fit harness end to end on the tiny MoE over ``gloo``
(``docs/model_parallelism.md`` §7, §10.6): the fit documents
(``tests/golden/_parallel/fit.py``) — and the dense fp32 fit ``das_dense``,
the same DAS document with its own realization — authored for
``tiny-random/qwen3.5-moe`` in fp32 on the CPU, run through the recording
entry (``tests/golden/_parallel/recorder.py``) at world 1 and at every
geometry the record holds them to — ``pp=2`` exact; ``tp=2`` and
``dp=2:rows`` (the DAS fit at ``attention_query``), ``ep=2`` (the DBM fit
at ``expert_activation``), ``tp=2`` and ``dp=2:rows`` (``das_dense``) banded — and **captured** exactly as the GPU
script captures the A3B: no refusal, the classes the document names, both
ranks' gradients and memory recorded, the record made and replaying
against itself, a fivefold regression of a binding class refused.

This is the harness's own smoke, not a numerics tier: the fixtures' bands
are the training smokes' business. What it pins is that the recorder
reaches every spawned rank through the production spawn path (the receipt
still says ``launcher: spawned``), that the world-1 fit's recorder writes under rank
0, that a pipeline stage without the featurizer records empty steps and is
skipped, that every fit run is held to the §7 runtime check
(``CAUSALAB_GRADIENT_AGREEMENT`` at the gloo band, written into the record),
and that the capture's refusals fire for real (a byte that differs under
``pp=2`` is named). An expert-keyed write on stage 1, which does not
publish, must produce the same ``routing_mismatch.json`` as world 1. Its
owner shares the record over the pipeline (``parallel/mismatch.py``, §6.5).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.golden import _parallel as par
from tests.golden._parallel import fit, measure, recorder
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

#: The tiny MoE in fp32 on the CPU: the DAS fit at its one full-attention
#: layer (3), the DBM fit at a DeltaNet block with routed experts (1) — on
#: the four-layer fixture ``pp=2`` puts layers 0–1 on stage 0, the publisher,
#: and layers 2–3 on stage 1, whose expert-keyed write's
#: ``routing_mismatch.json`` the test below holds to world 1's bytes.
TINY = par.Realization(TINY_QWEN35_MOE, "fp32", "cpu", layer=3)
#: ``das_dense`` — the DAS fit on the dense fp32 realization — runs here on
#: the tiny MoE like ``das`` (its own realization is the GPU record's;
#: the harness hands every document the fixture)
TINY_BY_DOCUMENT = {
    "das": TINY,
    "dbm": dataclasses.replace(TINY, layer=1),
    "das_dense": TINY,
}
FITS = ("das", "dbm", "das_dense")


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The subprocesses inherit the environment: offline, one thread each."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv(recorder.GRADIENTS_VARIABLE, raising=False)
    monkeypatch.delenv(par.GRADIENT_AGREEMENT_VARIABLE, raising=False)


@pytest.fixture(scope="module")
def captures(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Path, dict[str, Any], list[str]]]:
    """Both fit documents captured once for the module (one subprocess per
    run, the tiny MoE at world 1 and at each geometry)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    root = tmp_path_factory.mktemp("fit-golden")
    out: dict[str, tuple[Path, dict[str, Any], list[str]]] = {}
    for name in FITS:
        block, refusals = par.capture(
            root, par.DOCUMENTS[name], TINY_BY_DOCUMENT[name], with_context=False
        )
        out[name] = (root / name, block, refusals)
    return out


@pytest.mark.parametrize("name", FITS)
def test_the_fit_capture_is_certifiable_on_the_fixture(
    captures: dict[str, tuple[Path, dict[str, Any], list[str]]], name: str
) -> None:
    root, block, refusals = captures[name]
    document = par.DOCUMENTS[name]
    assert refusals == []
    assert block["exact"] == {g: 0.0 for g in document.exact}
    assert set(block["geometries"]) == set(document.banded)
    for geometry, classes in block["geometries"].items():
        assert set(classes) == set(document.classes), geometry
        for kind, entry in classes.items():
            assert entry["band"] == par.entry_band(kind, entry), (geometry, kind)
            assert entry["files"], (geometry, kind)
        assert classes["bundle"]["dtype"] == fit.FEATURE_DTYPE
        assert classes["tables"]["dtype"] == TINY_BY_DOCUMENT[name].dtype
        assert classes["gradient"]["scale"] == 1.0
    assert set(block["load"]) == set(document.geometries)
    assert block["gradient_agreement"] == float(par.GRADIENT_AGREEMENT)
    assert set(block["memory"]) == set(document.geometries) | {"solo"}
    for geometry, ranks in block["memory"].items():
        expected = {"rank0"} if geometry == "solo" else {"rank0", "rank1"}
        assert set(ranks) == expected, geometry
        for rank in ranks.values():
            assert rank["device"] is None and rank["steps"] == fit.STEPS["epochs"]


@pytest.mark.parametrize("name", FITS)
def test_the_record_made_from_the_capture_replays_and_refuses_a_regression(
    captures: dict[str, tuple[Path, dict[str, Any], list[str]]], name: str
) -> None:
    _, block, _ = captures[name]
    record = par.make_record(None, TINY_BY_DOCUMENT[name], {name: block})
    assert par.captured(record, name)
    assert par.compare_records(record, record, name) == []
    if name == "das_dense":
        # the document's own realization is in its block: the fixture's here
        assert record["documents"][name]["realization"]["model"] == TINY_QWEN35_MOE
    regressed = json.loads(par.render(record))
    geometry = par.DOCUMENTS[name].banded[0]
    entry = regressed["documents"][name]["geometries"][geometry]["bundle"]
    entry["max_abs_diff"] = max(5 * entry["max_abs_diff"], 2 * entry["band"])
    problems = par.compare_records(record, regressed, name)
    assert any(f"{name} {geometry} bundle" in p for p in problems)


def test_every_spawned_rank_recorded_through_the_production_spawn_path(
    captures: dict[str, tuple[Path, dict[str, Any], list[str]]],
) -> None:
    root, _, _ = captures["das"]
    for geometry in par.DOCUMENTS["das"].geometries:
        out = root / par.out_name(geometry)
        assert par.receipt(out)["execution"]["parallel"]["launcher"] == "spawned"
        for rank in range(par.WORLD):
            assert recorder.gradients_path(out / par.GRADIENTS, rank).exists(), geometry
            assert recorder.memory_path(out / par.GRADIENTS, rank).exists(), geometry
    solo = root / "solo" / par.GRADIENTS
    assert recorder.gradients_path(solo, 0).exists()
    assert not recorder.gradients_path(solo, 1).exists()
    # the stage that owns no featurizer recorded empty steps; the owner every step
    stages = [
        measure.load_gradients(root / par.out_name("pp=2") / par.GRADIENTS, r)
        for r in range(par.WORLD)
    ]
    assert sorted(all(not step for step in s) for s in stages) == [False, True]
    assert all(len(s) == fit.STEPS["epochs"] for s in stages)


def test_a_byte_that_differs_under_pp_is_refused_by_name(
    captures: dict[str, tuple[Path, dict[str, Any], list[str]]], tmp_path: Path
) -> None:
    """The capture's exactness refusal, on the real outputs: the pp=2 run's
    fitted bundle rewritten by one entry."""
    root, _, _ = captures["das"]
    solo, staged = root / "solo", root / par.out_name("pp=2")
    assert par.exact_differences(solo, staged) == []
    bundle = next(p for p in staged.iterdir() if p.suffix == ".safetensors")
    original = bundle.read_bytes()
    try:
        from causalab.io.tensor_files import load_file, save_file

        tensors = load_file(str(bundle))
        key = sorted(tensors)[0]
        tensors[key] = tensors[key].clone()
        tensors[key].view(-1)[0] += 1e-3
        save_file(tensors, str(bundle))
        assert par.exact_differences(solo, staged) == [bundle.name]
        measured = par.DOCUMENTS["das"].measure(solo, staged, TINY)
        assert measured.classes["bundle"].max_abs_diff == pytest.approx(1e-3, rel=1e-3)
        assert measured.worst() > 0.0
    finally:
        bundle.write_bytes(original)
    assert par.exact_differences(solo, staged) == []


def test_the_recorder_reads_the_rank_at_save_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed at import in a spawned child before the launcher sets
    ``RANK``; the file must still carry the rank the launcher set."""
    import torch

    from causalab.neural.engines.pytorch_hooks import train as train_module

    real = train_module.average_gradients
    monkeypatch.setenv(recorder.GRADIENTS_VARIABLE, str(tmp_path))
    monkeypatch.delenv("RANK", raising=False)
    saves: list[Any] = []
    monkeypatch.setattr(
        recorder.atexit, "register", lambda fn, *a: saves.append((fn, a))
    )
    try:
        recorder.install()
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.grad = torch.tensor([1.0, 2.0, 3.0])

        class Solo:
            def size(self, axis: Any) -> int:
                return 1

        # the real guard at world 1 is the fast path: nothing to average
        train_module.average_gradients([parameter], Solo(), axis="model")  # type: ignore[arg-type]
        monkeypatch.setenv("RANK", "1")
        ((fn, args),) = saves
        fn(*args)
    finally:
        train_module.average_gradients = real  # type: ignore[assignment]
    assert recorder.gradients_path(tmp_path, 1).exists()
    memory = json.loads(recorder.memory_path(tmp_path, 1).read_text())
    assert memory["rank"] == 1 and memory["steps"] == 1
    recorded = measure.load_gradients(tmp_path, 1)
    assert torch.equal(recorded[0][0], torch.tensor([1.0, 2.0, 3.0]))


def test_a_write_on_a_stage_that_does_not_publish_keeps_its_routing_mismatch_record(
    tmp_path: Path,
) -> None:
    """A routing-mismatch record belongs to the write's stage. Share it
    over the pipeline before the publisher collects it, so an expert-keyed
    write on stage 1 produces the same file as world 1."""
    stage_one = dataclasses.replace(par.DOCUMENTS["dbm"], name="dbm_stage1", banded=())
    block, refusals = par.capture(tmp_path, stage_one, TINY, with_context=False)
    assert refusals == []
    assert block["exact"] == {"pp=2": 0.0}
    staged = tmp_path / "dbm_stage1" / par.out_name("pp=2") / "routing_mismatch.json"
    solo = tmp_path / "dbm_stage1" / "solo" / "routing_mismatch.json"
    assert staged.exists() and solo.exists()
    assert staged.read_bytes() == solo.read_bytes()
    records = json.loads(solo.read_text())
    assert records and all(
        r["write"] == "mask_routed" and r["layer"] == 3 for r in records
    )
