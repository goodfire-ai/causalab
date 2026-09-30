"""The train loop evaluates a member's featurizer once per optimizer step
and once per eval pass — through the real loop on the tiny model — and the
fit it produces is **bit-identical** to the fit without the scope.

``tests/neural/shared/test_featurizer_cache.py`` pins what a scope does to a
stage; this pins that the loop *opens* one around the right blocks, and that
several optimizer steps with an eval pass, an early-stop snapshot and a
restore after each epoch see exactly the values the unscoped loop computes:
the loss at every step, every eval score, the selected fit. The count is by
the map's own evaluation ([`Cayley.map`][causalab.neural.shared.featurizers.stages.Cayley.map]), the frame the profiling
campaign attributed the launches to, taken per call of the loop's two
windows rather than over the whole fit: a checkpoint or the save-time
diagnostics read the weight too, legitimately.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable

import pytest
import torch

from causalab.neural.engines.pytorch_hooks import train as train_module
from causalab.neural.engines.pytorch_hooks.cuda_graphs import captured_pass
from causalab.neural.shared.featurizers import Cayley, Gate, Subspace, featurizer_cache

from tests.neural.engines.pytorch_hooks.test_rotation_round_trip import (
    _fit,  # pyright: ignore[reportPrivateUsage]
    das_doc,
)
from tests.neural.engines.pytorch_hooks.test_train import (
    chain_doc,
    controlled_dbm_doc,
    dbm_doc,
    hard_concrete_dbm_doc,
)
from tests.protocol._docs import term

unit = pytest.mark.unit


def _count_map_evaluations(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    """Wrap [`Cayley.map`][causalab.neural.shared.featurizers.stages.Cayley.map] in a counter and the loop's two windows —
    one optimizer step's grad forwards, one eval round — in a per-call
    tally of it."""
    calls = {"n": 0}
    real_map = Cayley.map

    def counting_map(self: Cayley, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return real_map(self, a, b)

    monkeypatch.setattr(Cayley, "map", counting_map)
    real_table = Gate._table_mask  # pyright: ignore[reportPrivateUsage]

    def counting_table(self: Gate) -> torch.Tensor:
        calls["n"] += 1
        return real_table(self)

    monkeypatch.setattr(Gate, "_table_mask", counting_table)
    per_call: dict[str, list[int]] = {"step": [], "eval": []}

    def tallying(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            before = calls["n"]
            try:
                return real(*args, **kwargs)
            finally:
                per_call[name].append(calls["n"] - before)

        return wrapped

    windows = {"_run_step_windows": "step", "_evaluate": "eval"}
    for attribute, name in windows.items():
        monkeypatch.setattr(
            train_module, attribute, tallying(name, getattr(train_module, attribute))
        )
    return per_call


@unit
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: das_doc(epochs=2, early_stop_mode="min"), id="das-cayley"),
        pytest.param(lambda: _with_eval(dbm_doc()), id="dbm-gate"),
        pytest.param(
            lambda: _with_eval(hard_concrete_dbm_doc()), id="dbm-hard-concrete"
        ),
    ],
)
def test_one_map_per_member_per_step_and_per_eval(
    build: Callable[[], dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four rows in pairs of two: two steps per epoch, an eval after each
    epoch. Without the scope a step evaluated the map (or the mask) for the
    read, and twice for the write (featurize and inverse), on every forward
    it ran; an eval pass evaluated it for its read and again for its write."""
    per_call = _count_map_evaluations(monkeypatch)
    outcome = _fit(build())
    assert outcome.eval_score is not None, "the fit did not evaluate"
    assert per_call["step"], "the loop never stepped"
    assert per_call["eval"], "the loop never evaluated"
    assert all(n == 1 for n in per_call["step"]), per_call["step"]
    assert all(n == 1 for n in per_call["eval"]), per_call["eval"]


def _epochs(doc: dict, epochs: int) -> dict:
    doc["method"]["train"]["steps"] = {"epochs": epochs}
    return doc


def _with_eval(doc: dict) -> dict:
    """Eval every epoch on the inline rows and select the *worst* ``ce`` with
    a patience the run never exhausts: three epochs, three evals, a snapshot
    after the first and a restore at the end — every branch of the loop the
    scope sits beside runs, and the selection is decided by an eval score."""
    # the eval scores the aggregation the document's `ce.json` tabulates,
    # over the same bound read (§2.11)
    (ce,) = [e for e in doc["method"]["save"] if e.get("file_path") == "ce.json"]
    doc["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "inline",
        "aggregations": {"ce": term(ce["read"], ce["model"], ce["aggregation"])},
    }
    doc["method"]["train"]["early_stop"] = {
        "on": "ce",
        "patience": 5,
        "mode": "max",
    }
    return doc


def _record_fit(
    doc: dict, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> dict[str, Any]:
    """Everything the loop computes that a stale or roundoff-shifted
    featurizer would move: the loss of every update, the score of every eval
    pass, the selected fit's score, the returned parameters."""
    losses: list[float] = []
    scores: list[dict[str, float]] = []
    real_loss, real_score = train_module._loss, train_module._score  # pyright: ignore[reportPrivateUsage]

    def recording_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
        loss = real_loss(*args, **kwargs)
        losses.append(float(loss.detach()))
        return loss

    def recording_score(*args: Any, **kwargs: Any) -> dict[str, float]:
        score = real_score(*args, **kwargs)
        scores.append(dict(score))
        return score

    with monkeypatch.context() as patch:
        patch.setattr(train_module, "_loss", recording_loss)
        patch.setattr(train_module, "_score", recording_score)
        if not cached:
            patch.setattr(train_module, "featurizer_cache", contextlib.nullcontext)
        outcome = _fit(doc)
    assert outcome.eval_score is not None
    return {
        "losses": losses,
        "scores": scores,
        "selected": (outcome.eval_score.selected, dict(outcome.eval_score.metrics)),
        "state": {
            name: {k: v.detach().clone() for k, v in stage.state_dict().items()}
            for name, stage in outcome.stages.items()
        },
    }


@unit
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _with_eval(das_doc(epochs=3)), id="das-cayley"),
        pytest.param(lambda: _with_eval(dbm_doc()), id="dbm-gate"),
        pytest.param(
            lambda: _with_eval(hard_concrete_dbm_doc()), id="dbm-hard-concrete"
        ),
        pytest.param(
            lambda: _with_eval(_epochs(chain_doc(1e-2), 3)), id="chain-rot-gate"
        ),
        pytest.param(lambda: _with_eval(controlled_dbm_doc()), id="dbm-controlled"),
    ],
)
def test_the_fit_is_bit_identical_with_and_without_the_scope(
    build: Callable[[], dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three epochs of updates, an eval and an early-stop snapshot after
    each, a restore at the end: every loss, every eval score and every
    returned parameter is the unscoped loop's to the bit. This is the
    invariant a cache of a trained quantity has to meet — it is only valid
    between two mutations of its inputs, and it may not change how the
    gradient reaches them — and it is what the GPU parity run measures."""
    cached = _record_fit(build(), monkeypatch, cached=True)
    plain = _record_fit(build(), monkeypatch, cached=False)
    assert len(plain["losses"]) >= 6 and len(plain["scores"]) >= 3, (
        "the run is too short to mean anything"
    )
    assert cached["losses"] == plain["losses"]
    assert cached["scores"] == plain["scores"]
    assert cached["selected"] == plain["selected"]
    assert cached["state"].keys() == plain["state"].keys()
    for name in plain["state"]:
        for key, value in plain["state"][name].items():
            assert torch.equal(value, cached["state"][name][key]), (name, key)
    moved = [
        (a - b).abs().max().item()
        for stage in plain["state"].values()
        for a, b in [(next(iter(stage.values())), torch.zeros(()))]
    ]
    assert any(m > 0 for m in moved), "no parameter moved — the fit is vacuous"


@unit
def test_a_captured_pass_evaluates_its_stages_inside_the_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structure of ``Replay`` off CUDA: ``captured_pass`` runs the work
    under an isolated scope. Two passes around a parameter update — the
    warmup pass and the capture — each evaluate the map once and each see
    the parameter as it stands, while a scope open around them (the loop's
    step scope) neither feeds the passes nor loses its own entry. A pass fed
    from outside would leave the map out of the captured graph and replay
    its stale value on every step."""
    calls = {"n": 0}
    real_map = Cayley.map

    def counting_map(self: Cayley, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return real_map(self, a, b)

    monkeypatch.setattr(Cayley, "map", counting_map)
    stage = Subspace(16, 4, "cayley", seed=0)
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(1))
    original = dict(stage.named_parameters())["parametrizations.weight.original"]

    def work() -> torch.Tensor:
        f, err = stage.featurize(x)
        loss = (stage.inverse(f, err) - x).pow(2).sum() + f.pow(2).sum()
        loss.backward()
        return f.detach()

    run = captured_pass(work)
    with featurizer_cache():
        outer, _ = stage.featurize(x)  # the enclosing scope's own entry
        calls["n"] = 0
        warmup = run()
        assert calls["n"] == 1, "the pass evaluated the map itself, once"
        with torch.no_grad():
            original.add_(0.3)
        captured = run()
        assert calls["n"] == 2, "the second pass evaluated it again"
        assert not torch.equal(warmup, captured), "the pass saw the moved parameter"
        again, _ = stage.featurize(x)
        assert calls["n"] == 2 and torch.equal(again, outer), (
            "the enclosing scope's entry survived the passes"
        )
    assert original.grad is not None
