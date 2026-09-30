"""Required evidence cannot disappear; observers restore the calling thread."""

import sys

import pytest

from causalab.measurement.runtime.training import training_evidence

pytestmark = pytest.mark.unit


def test_missing_required_training_evidence_is_refused_and_profiler_restored():
    original = sys.getprofile()
    with pytest.raises(ValueError, match="not observed"):
        with training_evidence(device="cpu", required=True, operation_step="fit"):
            pass
    assert sys.getprofile() is original


def test_failed_diagnostic_restores_the_previous_profiler():
    original = sys.getprofile()
    with pytest.raises(RuntimeError, match="failed fit"):
        with training_evidence(device="cpu", required=False):
            raise RuntimeError("failed fit")
    assert sys.getprofile() is original


@pytest.mark.parametrize("kind", ["das", "dbm"])
def test_real_cohort_observes_each_member_without_changing_its_fit(kind):
    import torch

    from causalab.neural.engines.pytorch_hooks import train
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from tests.neural.engines.pytorch_hooks.test_fit_cohort import (
        TINY_LLAMA,
        _fit_cohort,
        _train_doc,
        _weights,
    )

    bundle = load_model(TINY_LLAMA)
    raws = [_train_doc(kind, seed=seed, epochs=2, eval_every=None) for seed in (7, 8)]
    expected, _, _, _ = _fit_cohort(raws, bundle)
    entry = train.run_cohort_training
    prepare = train._prepare_fit
    with training_evidence(
        device="cpu", required=True, operation_step="fit"
    ) as observed:
        actual, _, _, _ = _fit_cohort(raws, bundle)
    assert train.run_cohort_training is entry and train._prepare_fit is prepare
    assert len(observed["fits"]) == 2
    for seed, fit, reference, outcome in zip(
        (7, 8), observed["fits"], expected, actual
    ):
        assert fit["seed"] == seed and fit["completed"]
        assert fit["optimizer_steps"] == len(fit["batches"]) == 4
        rng = torch.Generator().manual_seed(seed)
        batches = [[0, 1], [2, 3]]
        order = [i for _ in range(2) for i in torch.randperm(2, generator=rng).tolist()]
        assert [row["logical_indices"] for row in fit["batches"]] == [
            batches[i] for i in order
        ]
        assert [row["update"] for row in fit["batches"]] == [0, 1, 2, 3]
        assert [row["epoch"] for row in fit["batches"]] == [0, 0, 1, 1]
        assert fit["initial"]["optimizer_state"]["state"] == {}
        assert all("mask_rng_state" in row for row in fit["batches"])
        assert _weights(outcome).keys() == _weights(reference).keys()
        assert all(
            torch.equal(value, _weights(reference)[name])
            for name, value in _weights(outcome).items()
        )
