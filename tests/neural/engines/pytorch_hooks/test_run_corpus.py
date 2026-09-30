"""End-to-end corpus execution on tiny-random (CPU) through the real CLI.

The golden corpus authors Qwen3-8B; ``--set`` overrides (spec §9)
retarget the model key and layer/head indices at tiny scale — the
documents' semantics are untouched, which is exactly what the override
mechanism is for. The sweep/train corpus files (04, 05, 07) stay off the
CPU budget; the fit→apply roundtrip below covers the
train path end to end at tiny scale instead, including the
ArtifactIdentity stamp-and-check cycle that corpus 09 specifies.

Corpus 08 is the one exception, run at a single ``k`` and one epoch (~10 s):
it is the only document in the corpus that sweeps ``train.seed``, and a
seed sweep is the one thing a single-point fit cannot check — the stage
cache is keyed by featurizer name alone, so a cache shared across points
would hand seed 0's rotation to every other seed and no unit test on
``build_stack`` would notice. See ``test_featurizer_seed.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file

from causalab.cli import main
from causalab.io.env import read_safetensors_metadata

from tests.protocol._env import CORPUS_DIR, FIXTURES
from tests.neural.engines.pytorch_hooks._drive import base_data_section  # noqa: F401  (tier anchor)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.tables import frame as table_frame

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    from tests.protocol._env import write_rot_fixture

    artifacts = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    return FIXTURES / "data", artifacts


def _run(name: str, roots: tuple[Path, Path], out: Path, *overrides: str) -> int:
    data_root, artifacts_root = roots
    argv = [
        "run",
        "--engine",
        "auto",
        str(CORPUS_DIR / name),
        "--data-root",
        str(data_root),
        "--artifacts-root",
        str(artifacts_root),
        "--out",
        str(out),
        "--set",
        f"model.key={TINY_LLAMA}",
        "--set",
        "model.dtype=fp32",  # tiny-random on CPU; 04/05/08 declare bf16
    ]
    for item in overrides:
        argv += ["--set", item]
    return main(argv)


def test_01_harvest_runs(roots, tmp_path):
    code = _run(
        "01_harvest_im.json",
        roots,
        tmp_path,
        "sites.L8.layers=0",
        "sites.L24.layers=1",
    )
    assert code == 0
    acts = load_file(str(tmp_path / "acts_L8_ans.safetensors"))
    assert acts["acts_L8_ans"].shape == (2, 1, 16)
    ragged = load_file(str(tmp_path / "acts_L8_ent.safetensors"))
    assert "acts_L8_ent.widths" in ragged  # multi-token entities are ragged


def test_02_interchange_runs_and_scores(roots, tmp_path):
    code = _run("02_interchange_im.json", roots, tmp_path, "sites.target.layers=1")
    assert code == 0
    iia = table_frame(tmp_path / "iia.json")
    assert len(iia) == 2  # one row per example
    assert set(iia["value"]).issubset({0.0, 1.0})  # match is an indicator
    ld = table_frame(tmp_path / "logit_diff.json")
    assert len(ld) == 2 and ld["value"].dtype.kind == "f"


def test_03_path_patching_runs(roots, tmp_path):
    code = _run(
        "03_path_patching_im.json",
        roots,
        tmp_path,
        "sites.sender.layers=0",
        "sites.sender.head=1",
        "sites.receiver.layers=1",
        "sites.a10.layers=0",
        "sites.a11.layers=1",
    )
    assert code == 0
    ld = table_frame(tmp_path / "logit_diff.json")
    assert len(ld) == 3


def test_15_circuit_edges_runs(roots, tmp_path):
    """The per-edge polarity map, executed: eight additive writes at one
    address, two intervened models, and the two edge sets must not land on the
    same number as the clean run or as each other — the whole point is that
    which edges are routed changes the metric."""
    code = _run(
        "15_circuit_edges_im.json",
        roots,
        tmp_path,
        # the fixture has two layers, so every sender is layer 0 and the
        # receiver is the residual entering layer 1 — still strictly upstream
        "sites.nm_9_6.layers=0",
        "sites.nm_9_6.head=1",
        "sites.nm_9_9.layers=0",
        "sites.nm_9_9.head=2",
        "sites.nm_10_0.layers=0",
        "sites.nm_10_0.head=0",
        "sites.ctl_10_7.layers=0",
        "sites.ctl_10_7.head=3",
        "sites.recv.layers=1",
    )
    assert code == 0
    clean = table_frame(tmp_path / "ld_clean.json")["value"]
    circuit = table_frame(tmp_path / "ld_circuit.json")["value"]
    complement = table_frame(tmp_path / "ld_complement.json")["value"]
    assert len(clean) == len(circuit) == len(complement) == 3
    # routing edges moved the metric, and the two edge sets are distinguishable
    assert not clean.equals(circuit)
    assert not clean.equals(complement)
    assert not circuit.equals(complement)


def test_06_hydra_effect_runs(roots, tmp_path):
    code = _run(
        "06_hydra_effect_im.json",
        roots,
        tmp_path,
        "sites.abl.layers=0",
        "sites.probe14.layers=0",
        "sites.probe20.layers=1",
        "sites.resid_final.layers=1",
    )
    assert code == 0
    for rel in (
        "te_clean.json",
        "te_abl.json",
        "de14_clean.json",
        "de20_abl.json",
    ):
        assert (tmp_path / rel).is_file()


def test_fit_then_apply_roundtrip(roots, tmp_path):
    """Corpus 04 → 09 at tiny scale: train a DAS rotation, save the stamped
    bundle, then load it through a 09-shaped document — the
    ArtifactIdentity written at save must satisfy the check at load, and a
    doctored k must refuse."""
    data_root, artifacts_root = roots
    fit_out = tmp_path / "fit"
    code = _run(
        "04_das_im.json",
        (data_root, artifacts_root),
        fit_out,
        "sites.target.layers=1",
        "featurizers.rot.k=4",
        'train.steps={"epochs": 1}',
        'train.batch={"pairs": 2}',
    )
    assert code == 0
    bundle_path = fit_out / "rot.safetensors"
    assert bundle_path.is_file()
    fitted = load_file(str(bundle_path))
    assert fitted["weight"].shape == (16, 4)
    # the header records the ref the fit read (§8); which bytes that ref
    # resolved to is the receipt's canonical form (data.*.digest), not a stamp
    stamped = read_safetensors_metadata(bundle_path)
    assert stamped is not None
    assert stamped["trained_on"] == "weekdays/data#train"

    # stage the fitted bundle as 09's artifact and apply it
    target = artifacts_root / "artifacts/tiny/rot_k4.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_path, target)
    apply_out = tmp_path / "apply"
    code = _run(
        "09_das_apply_im.json",
        (data_root, artifacts_root),
        apply_out,
        "sites.target.layers=1",
        "featurizers.rot.k=4",
        "featurizers.rot.file_path=artifacts/tiny/rot_k4.safetensors",
    )
    assert code == 0
    iia = table_frame(apply_out / "iia.json")
    assert len(iia) == 2  # the weekdays/data#test split

    # a doctored declaration must refuse against the stamp (§2.5)
    code = _run(
        "09_das_apply_im.json",
        (data_root, artifacts_root),
        tmp_path / "mismatch",
        "sites.target.layers=1",
        "featurizers.rot.k=8",
        "featurizers.rot.file_path=artifacts/tiny/rot_k4.safetensors",
    )
    assert code == 1


def test_08_seed_sweep_fits_three_genuinely_different_rotations(roots, tmp_path):
    """A ``{"sweep": [0,1,2]}`` on ``train.seed`` must fit three rotations
    from three *different* starting points (§2.11).

    The bug this guards end to end: ``train.seed`` never reached the
    ``subspace`` init, so all three points started from the identical
    rotation and drifted apart only by batch order. The three fits still
    differed — by ~1e-3, a subspace distance of ~0.015 — so nothing looked
    obviously wrong, but a sweep read as evidence of stability was really
    measuring data-order jitter around one initialisation.

    Asserted on the *column space*, ``‖QaQaᵀ − QbQbᵀ‖_F``, which ignores the
    arbitrary basis inside each frame: 0 is the same subspace, ``√(2k)`` ≈
    2.83 orthogonal ones. Pre-fix this measured ≤ 0.016 for every pair;
    the threshold below is an order of magnitude above that and an order of
    magnitude below what independent inits give (~2.4).
    """
    import torch

    out = tmp_path / "sweep"
    code = _run(
        "08_weekdays_das_sweep_im.json",
        roots,
        out,
        "sites.target.layers=1",
        "featurizers.rot.k=4",  # one k, so the sweep is over seed alone
        'train.steps={"epochs": 1}',
        'train.batch={"pairs": 2}',
    )
    assert code == 0
    fitted = load_file(str(out / "rot.safetensors"))
    assert sorted(fitted) == [f"weight[seed={s}]" for s in (0, 1, 2)]

    names = sorted(fitted)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            qa, qb = fitted[a], fitted[b]
            distance = float((qa @ qa.T - qb @ qb.T).norm())
            assert distance > 0.5, f"{a} and {b} fit the same subspace ({distance:.4f})"
            # each is still a frame, not just noise
            torch.testing.assert_close(
                qa.T @ qa, torch.eye(qa.shape[1]), atol=1e-5, rtol=1e-4
            )


def test_the_train_eval_score_reaches_the_run_tree(roots, tmp_path):
    """``train.eval`` is computed, so it must be saved (§2.12).

    Corpus 08 declares ``eval: {split: weekdays/data#test, aggregations: {iia:
    …}}`` and saves the same aggregation to ``iia.json``. Before this fix ``_run_eval``'s return value
    was consumed only inside the ``early_stop`` branch and then dropped, so
    ``iia.json`` held the **train** score under a name every reader took for
    the eval one — and reporting training and evaluation results together
    was impossible without re-running the fit as five extra apply documents.

    The eval score is a sibling record, not a column: it is measured on a
    different split, i.e. a different population from ``iia.json``'s rows.
    """
    out = tmp_path / "eval_reaches"
    code = _run(
        "08_weekdays_das_sweep_im.json",
        roots,
        out,
        "sites.target.layers=1",
        "featurizers.rot.k=4",
        'train.steps={"epochs": 1}',
        'train.batch={"pairs": 2}',
    )
    assert code == 0

    records = json.loads((out / "train_eval.json").read_text())
    assert len(records) == 3  # one per seed in the sweep
    for record in records:
        assert record["split"] == "weekdays/data#test"
        assert record["passes"] == 1
        assert record["featurizers"] == ["rot"]
        assert isinstance(record["metrics"]["iia"], float)

    # the join to the metric table is the point's coordinates (the swept
    # seed: a metric row's axis column, a train_eval record's `coords`), and
    # the two numbers are different populations — so at least one must
    # actually differ, or the "eval" score is just the train score wearing a
    # different name
    train = table_frame(out / "iia.json")
    train_mean = train[train["metric"] == "iia"].groupby("train.seed")["value"].mean()
    seeds = [record["coords"]["train.seed"] for record in records]
    assert set(seeds) == set(train_mean.index)
    assert any(
        record["metrics"]["iia"] != pytest.approx(train_mean[seed])
        for record, seed in zip(records, seeds)
    )


def test_a_gate_fit_writes_its_mask_diagnostic_to_the_run_tree(roots, tmp_path):
    """The report has to reach the run's outputs, not just the return value —
    the whole point is that a non-mask is never again invisible to whoever
    reads the run later."""
    out = tmp_path / "dbm"
    code = _run(
        "05_dbm_im.json",
        roots,
        out,
        "sites.target.layers=1",
        'train.steps={"epochs": 1}',
        'train.batch={"pairs": 2}',
    )
    assert code == 0
    (record,) = json.loads((out / "fit_diagnostics.json").read_text())
    report = record["featurizers"]["gate"]
    assert report["width"] > 0
    assert 0.0 <= report["decisive_fraction"] <= 1.0
    assert 0 <= report["hard_mask_size"] <= report["width"]
    assert record["point"]  # joins to the bundle and the metric table


def test_a_document_without_train_eval_writes_no_eval_record(roots, tmp_path):
    """No ``train.eval``, no file — the run tree never carries an empty record
    a reader would have to interpret."""
    out = tmp_path / "no_eval"
    code = _run(
        "01_harvest_im.json", roots, out, "sites.L8.layers=0", "sites.L24.layers=1"
    )
    assert code == 0
    assert not (out / "train_eval.json").exists()


def test_explain_and_digest_work_on_every_corpus_file(roots, capsys):
    data_root, artifacts_root = roots
    for path in sorted(CORPUS_DIR.glob("*_im.json")):
        for verb in ("digest", "explain", "validate"):
            code = main(
                [
                    verb,
                    str(path),
                    "--data-root",
                    str(data_root),
                    "--artifacts-root",
                    str(artifacts_root),
                    # `digest` alone takes no engine
                    *(() if verb == "digest" else ("--engine", "auto")),
                ]
            )
            assert code == 0, f"{verb} failed on {path.name}"
    capsys.readouterr()


def test_run_output_is_stamped(roots, tmp_path):
    assert (
        _run(
            "01_harvest_im.json",
            roots,
            tmp_path,
            "sites.L8.layers=0",
            "sites.L24.layers=1",
        )
        == 0
    )
    from causalab.io.env import read_safetensors_metadata

    meta = read_safetensors_metadata(tmp_path / "acts_L24_ans.safetensors")
    assert meta is not None
    assert meta["model_key"] == TINY_LLAMA
    assert meta["engine"] == "pytorch_hooks"


def test_11_probe_generate_runs_and_scores(roots, tmp_path):
    """The exploration probe end to end: decode under a steer, score the last
    generated token with an ordinary metric."""
    code = _run("11_probe_generate_im.json", roots, tmp_path, "sites.target.layers=1")
    assert code == 0
    probe = table_frame(tmp_path / "probe.json")
    assert len(probe) == 2  # one row per example
    for value in probe["value"]:
        top = json.loads(value)
        # `by: prob` on an lm_head read emits all four columns (§2.10)
        assert set(top) == {"indices", "tokens", "values", "probs"}
        assert all(len(column) == 1 for column in top.values())


def test_12_probe_variable_scores_every_step_and_reports_what_was_said(roots, tmp_path):
    """The decode-step surface end to end: a metric per decode step, an ids-domain
    metric that never touches the vocabulary, and a `variable` anchor whose
    misses come back as data.

    Tiny-random says nothing resembling a weekday, so `said_answer` matches
    nowhere — which is the case worth pinning: the run finishes, the rows
    survive, and `matched` says why the values are null.
    """
    code = _run("12_probe_variable_im.json", roots, tmp_path, "sites.target.layers=1")
    assert code == 0

    per_step = table_frame(tmp_path / "per_step.json")
    examples = per_step["example_id"].nunique()
    assert len(per_step) == examples * 8  # one row per (example, decode step)
    assert sorted(per_step["step"].unique()) == list(range(8))
    assert per_step["matched"].all()

    said = table_frame(tmp_path / "said.json")
    assert len(said) == examples  # decode reduces the window to one string
    assert said["step"].isna().all()  # no single step owns a joined string
    assert said["value"].notna().all()

    where = load_file(tmp_path / "where.safetensors")
    # every row missed, so every row addressed zero positions: the harvest is
    # empty but still shaped per example, and nothing about the run failed
    assert where["where"].shape[:2] == (examples, 0)
