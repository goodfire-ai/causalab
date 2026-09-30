"""The train loop on tiny-random: DAS and DBM fits (spec §2.11).

Random weights carry no task signal, so these are mechanism tests, not
quality tests: the fit runs, moves exactly the declared params, honors the
seed (same seed → byte-identical fit; different seed → different fit),
anneals the gate temperature, and reduces its own training objective on
the batch it optimizes (a sanity floor even a random model must clear —
the loss is optimized directly)."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from causalab.protocol.engine import RunContext
from causalab.protocol.schema import parse_document
from causalab.protocol.rules.document import validate_document

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, by_label, in_order, saved, term

pytestmark = pytest.mark.unit

BASES = [
    "the quick brown fox jumps over",
    "a slow green turtle sleeps deeply",
    "every shiny robot dances tonight",
    "some ancient rivers flow backwards",
]
COUNTERFACTUALS = [
    "cold silver mountains echo loudly",
    "bright yellow parrots sing early",
    "seven broken clocks tick wrongly",
    "warm quiet valleys rest gently",
]
ANSWERS = [" one", " two", " three", " four"]

#: The fits' one aggregation (§2.10): the cross-entropy of the patched logits
#: against the row's ``label`` — the objective's term, and the table
#: ``ce.json`` (label ``ce``) every training document saves.
CE: dict[str, Any] = {
    "kind": "cross_entropy",
    "target": "label",
}


def ce_term(**extra: Any) -> dict[str, Any]:
    """An objective or eval term reducing the patched logits by `CE`."""
    return term("logits", "patched", dict(CE), **extra)


def ce_save() -> dict[str, Any]:
    """The save entry tabulating `CE` as ``ce.json``."""
    return saved("logits", "patched", "ce.json", dict(CE))


def das_doc(*, seed: int = 0, epochs: int = 2) -> dict:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "sites": {
                "tgt": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
            },
            "reads": {
                "v_cf": {"site": "tgt", "pos": {"index": -1}, "featurizer": "rot"},
                "logits": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {
                "patch": {
                    "site": "tgt",
                    "pos": {"index": -1},
                    "featurizer": "rot",
                    "do": {"swap": "v_cf"},
                }
            },
            "train": {
                "objective": [[1.0, ce_term()]],
                "params": ["rot"],
                "optimizer": {"name": "adamw", "lr": 1e-2, "weight_decay": 0.0},
                "steps": {"epochs": epochs},
                "batch": {"pairs": 2},
                "seed": seed,
            },
            "save": [
                ce_save(),
                {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
            ],
        },
    }


class _NoDatasets:
    def digest(self, ref: str) -> str:
        return "0" * 64

    def columns(self, ref: str) -> tuple[str, ...]:
        return ()


def _fit(doc_raw: dict) -> dict[str, torch.Tensor]:
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    return {
        f"{name}.{slot}": param.detach().clone()
        for name, stage in outcome.stages.items()
        for slot, param in stage.slot_params().items()
    }


def test_das_fit_moves_only_the_rotation_and_is_seeded():
    first = _fit(das_doc(seed=0))
    again = _fit(das_doc(seed=0))
    other = _fit(das_doc(seed=1))
    assert set(first) == {"rot.weight"}
    torch.testing.assert_close(
        first["rot.weight"], again["rot.weight"], atol=0.0, rtol=0.0
    )
    assert not torch.allclose(first["rot.weight"], other["rot.weight"], atol=1e-6)


def test_das_weight_stays_orthonormal():
    weight = _fit(das_doc(seed=0))["rot.weight"]
    gram = weight.T @ weight
    torch.testing.assert_close(gram, torch.eye(weight.shape[1]), atol=1e-5, rtol=1e-4)


def test_das_fit_reduces_its_own_objective():
    """Optimizing CE on the training rows must reduce CE on those rows —
    compare the document's metric before and after the fit."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.shared.metrics import compute_metric

    bundle = load_model(TINY_LLAMA)

    def mean_ce(fit: bool) -> float:
        doc_raw = das_doc(seed=0, epochs=4)
        executor = executor_for(
            doc_raw,
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
        )
        if fit:
            from causalab.neural.engines.pytorch_hooks.train import run_training
            from causalab.io.env import ResolutionEnv

            request = RunContext(
                env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
                output_dir=None,  # type: ignore[arg-type]
            )
            run_training(executor.doc, executor, request)
            executor.reset_reads()
        values = compute_metric(
            by_label(executor.doc)["ce"],
            executor.dense_value("logits"),
            executor.rows_for_metrics(),
            bundle.tokenizer,
        )
        return sum(values) / len(values)

    assert mean_ce(fit=True) < mean_ce(fit=False)


def _margin_doc(weight: float, kind: str) -> dict:
    """`das_doc` scored by a margin metric — `logit_diff` or `soft_accuracy`
    between the row's label and a fixed wrong token — as the only objective
    term, at the authored `weight`."""
    doc = das_doc(seed=0, epochs=4)
    margin = {"kind": kind, "a": "label", "b": "wrong"}
    doc["method"]["train"]["objective"] = {
        "fit": term("logits", "patched", dict(margin), weight=weight)
    }
    doc["method"]["save"][0] = saved("logits", "patched", "margin.json", margin)
    return doc


WRONG = [" nine", " nine", " nine", " nine"]


def _mean_margin(doc_raw: dict, *, fit: bool) -> float:
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.shared.metrics import compute_metric

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS, "wrong": WRONG},
    )
    if fit:
        from causalab.neural.engines.pytorch_hooks.train import run_training
        from causalab.io.env import ResolutionEnv

        request = RunContext(
            env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
            output_dir=None,  # type: ignore[arg-type]
        )
        run_training(executor.doc, executor, request)
        executor.reset_reads()
    values = compute_metric(
        by_label(executor.doc)["margin"],
        executor.read_value("logits"),
        executor.rows_for_metrics(),
        bundle.tokenizer,
    )
    return sum(values) / len(values)


@pytest.mark.parametrize("kind", ["logit_diff", "soft_accuracy"])
def test_a_negative_weight_maximizes_a_margin_term(kind):
    """A metric term's weight is signed (spec §2.11): the loop minimizes
    `Σ w · term`, so `-1` on a margin drives the margin *up* — MIB's objective
    on `logit_diff`, a sigmoid of the margin on `soft_accuracy` — and `+1` drives it
    down. Both directions are asserted so the test cannot pass on a fit that
    ignores the weight."""
    before = _mean_margin(_margin_doc(-1.0, kind), fit=False)
    assert _mean_margin(_margin_doc(-1.0, kind), fit=True) > before
    assert _mean_margin(_margin_doc(+1.0, kind), fit=True) < before


def test_the_soft_accuracy_objective_is_the_saved_tables_twin():
    """`metric_tensor`'s `soft_accuracy` and `compute_metric`'s agree per row
    on the same logits, so the objective a fit minimizes and the table it
    saves cannot disagree — the property `js` already has."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import metric_tensor
    from causalab.neural.shared.metrics import compute_metric

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        _margin_doc(1.0, "soft_accuracy"),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS, "wrong": WRONG},
    )
    metric = by_label(executor.doc)["margin"]
    logits = executor.read_value("logits")
    rows = executor.rows_for_metrics()
    saved = compute_metric(metric, logits, rows, bundle.tokenizer)
    objective = metric_tensor(metric, logits, rows, bundle.tokenizer)
    assert objective.shape == (len(rows),)
    assert all(0.0 < v < 1.0 for v in saved)
    assert objective.tolist() == pytest.approx(saved, abs=1e-6)


def dbm_doc() -> dict:
    doc = das_doc(seed=0, epochs=3)
    doc["method"]["featurizers"] = {"gate": {"kind": "gate"}}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    doc["method"]["train"]["params"] = ["gate"]
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.01, {"l1": "gate"}]]
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
    doc["method"]["save"] = [
        ce_save(),
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"},
    ]
    return doc


def test_dbm_fit_trains_theta_and_anneals_temperature():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        dbm_doc(),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    gate = run_training(executor.doc, executor, request).stages["gate"]
    assert not torch.allclose(gate.theta, torch.zeros_like(gate.theta))
    assert gate.temperature < 1.0  # the anneal ran
    assert not gate.training  # left in (hard) eval mode


def _gate_at_two_sites_doc(*, tied: bool, lr: float = 0.1) -> dict:
    """``dbm_doc`` with a second block_output site (layer 1) written through
    the *same* gate name (``tied`` — one parameter set at two addresses, §2.5
    "one name, several sites") or through a second gate ``g1`` of its own —
    one SGD update, no anneal, so the identity below is exact arithmetic."""
    doc = dbm_doc()
    method = doc["method"]
    method["sites"]["tgt2"] = {"component": "block_output", "layers": [1]}
    second = "g0" if tied else "g1"
    method["featurizers"] = {"g0": {"kind": "gate"}}
    if not tied:
        method["featurizers"]["g1"] = {"kind": "gate"}
    method["reads"]["v_cf"]["featurizer"] = "g0"
    method["reads"]["v2"] = {"site": "tgt2", "pos": -1, "featurizer": second}
    method["intervened_models"][UNWRITTEN]["reads"].append("v2")
    method["writes"]["patch"]["featurizer"] = "g0"
    method["writes"]["patch2"] = {
        "site": "tgt2",
        "pos": -1,
        "featurizer": second,
        "do": {"swap": "v2"},
    }
    method["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    method["train"]["params"] = ["g0"] if tied else ["g0", "g1"]
    method["train"]["objective"] = [[1.0, ce_term()]]
    method["train"]["optimizer"] = {"name": "sgd", "lr": lr}
    method["train"]["steps"] = {"updates": 1}
    method["train"].pop("anneal", None)
    method["save"] = [
        ce_save(),
        {"value": "g0", "site": "tgt", "file_path": "g0.safetensors"},
    ] + (
        [] if tied else [{"value": "g1", "site": "tgt2", "file_path": "g1.safetensors"}]
    )
    return doc


def test_one_gate_named_at_two_sites_is_one_stage_and_sums_the_gradients_of_both_sites():
    """§2.5, one name at several sites: ``g0`` written at layers 0 and 1 is
    one stage — the one object in the executor's stage cache, which both
    writes go through — and since the tied fit and the two independent gates
    start from the same θ₀, one SGD step of the tied fit moves θ by exactly
    the sum of what the two independent gates each moved:
    θ_tied − θ₀ = (θ_g0 − θ₀) + (θ_g1 − θ₀). The fitted outcome carries the
    one name, and `train.params` and `save` named it once."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)

    def fit(tied: bool):
        executor = executor_for(
            _gate_at_two_sites_doc(tied=tied),
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
        )
        if tied:
            assert set(executor.doc.featurizers) == {"g0"}
            assert executor.stage("g0") is executor.stage("g0")
        else:
            assert executor.stage("g1") is not executor.stage("g0")
        request = RunContext(
            env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
            output_dir=None,  # type: ignore[arg-type]
        )
        outcome = run_training(executor.doc, executor, request)
        if tied:
            # one parameter set: the cache built one gate, at one width
            assert set(executor.stage_cache) == {"g0"}
        return outcome

    tied = fit(tied=True)
    independent = fit(tied=False)
    assert set(tied.stages) == {"g0"} and set(independent.stages) == {"g0", "g1"}
    theta0 = torch.zeros_like(tied.stages["g0"].theta)
    moved_tied = tied.stages["g0"].theta.detach() - theta0
    moved_g0 = independent.stages["g0"].theta.detach() - theta0
    moved_g1 = independent.stages["g1"].theta.detach() - theta0
    assert not torch.allclose(moved_tied, torch.zeros_like(moved_tied))
    assert torch.allclose(moved_tied, moved_g0 + moved_g1, atol=1e-6, rtol=1e-4)
    assert set(tied.diagnostics) == {"g0"}


def test_parameter_count_costs_under_sum_fit_exactly_as_the_mean_does():
    """The run-level identity of §2.11 ``costs``: on one gate,
    ``reduce: sum`` with ``costs: parameter_count`` is ``Σ σ/units`` — the
    mean — so the two fits take the same trajectory, checkpoint for
    checkpoint (θ under ``allclose``), and the recorded term values agree to
    reduction order — ``(Σσ)/n`` against ``Σ(σ/n)``, hence ``rel=1e-5`` (above
    float32's worst case over the units, ``n·eps``), not
    bitwise. Fails if the cost were applied after the reduction, or divided
    by the wrong count."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)

    def outcome(regularizer: dict):
        doc = dbm_doc()
        doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.01, regularizer]]
        doc["method"]["save"].append(
            {
                "kind": "trajectory",
                "every": {"updates": 2},
                "file_path": "t.safetensors",
            }
        )
        executor = executor_for(
            doc,
            bundle,
            base_texts=BASES,
            counterfactual_texts=COUNTERFACTUALS,
            extra_columns={"label": ANSWERS},
        )
        request = RunContext(
            env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
            output_dir=None,  # type: ignore[arg-type]
        )
        return run_training(executor.doc, executor, request)

    mean = outcome({"l1": "gate"})
    costed = outcome({"l1": "gate", "reduce": "sum", "costs": "parameter_count"})
    assert len(mean.checkpoints) == len(costed.checkpoints) >= 2
    for a, b in zip(mean.checkpoints, costed.checkpoints):
        assert a.step == b.step
        assert torch.allclose(a.slots["gate"]["theta"], b.slots["gate"]["theta"])
        assert a.record["term.1"] == pytest.approx(b.record["term.1"], rel=1e-5)
    # and a plain `sum` is the mean scaled by the unit count, so it diverges
    summed = outcome({"l1": "gate", "reduce": "sum"})
    units = mean.stages["gate"].theta.numel()
    assert summed.checkpoints[0].record["term.1"] != pytest.approx(
        mean.checkpoints[0].record["term.1"]
    )
    assert units > 1


def test_a_constraint_ascends_its_duals_toward_the_target_density():
    """§2.11 ``constraint``: a sigmoid gate starts at density ½; with a
    target of 0.1 the gap is positive, so the ascent raises λ₁ (its gradient
    is the gap) and λ₂ (the gap squared) from 0 at every update. The term
    carries no weight: the checkpoint records `term.density`, `lambda1.density`
    and `lambda2.density` and no `weight.density`; the outcome records the
    target, the duals' start and end and the last density."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = dbm_doc()
    doc["method"]["train"]["objective"] = {
        "fit": ce_term(weight=1.0),
        "density": {"l1": "gate", "constraint": {"target": 0.1, "dual": {"lr": 0.5}}},
    }
    # plain SGD, so the duals' ascent is exactly lr · gradient and can be pinned
    doc["method"]["train"]["optimizer"] = {"name": "sgd", "lr": 0.1}
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"updates": 2}, "file_path": "t.safetensors"}
    )
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    record = outcome.constraints["density"]
    assert record["target"] == 0.1
    assert record["lambda1_initial"] == 0.0 and record["lambda2_initial"] == 0.0
    assert record["lambda1_final"] > 0.0 and record["lambda2_final"] > 0.0
    assert 0.0 < record["value_final"] < 1.0
    trace = outcome.constraint_trace["density"]
    assert len(trace) == record["updates"] == 6  # three epochs of two batches
    lambdas = [row["lambda1"] for row in trace]
    assert (
        lambdas == sorted(lambdas) and lambdas[0] > 0.0
    )  # ascending from the first update
    assert all(row["target"] == 0.1 for row in trace)
    for checkpoint in outcome.checkpoints:
        keys = set(checkpoint.record)
        assert {
            "term.density",
            "lambda1.density",
            "lambda2.density",
            "term.fit",
            "weight.fit",
        } <= keys
        assert "weight.density" not in keys
    # the first update stepped with the initial duals (a zero penalty) and the
    # ascent moved them by lr · gradient: λ₁ by the gap, λ₂ by its square
    gap = trace[0]["value"] - 0.1
    assert trace[0]["lambda1"] == pytest.approx(0.5 * gap, rel=1e-5)
    assert trace[0]["lambda2"] == pytest.approx(0.5 * gap * gap, rel=1e-5)
    # λ₂ accumulates (s − t)² and never falls; λ₁ is monotone here only
    # because the mask stays denser than the target for six updates
    lambda2s = [row["lambda2"] for row in trace]
    assert lambda2s == sorted(lambda2s)
    assert all(row["value"] > 0.1 for row in trace)


def test_a_constraint_holds_the_costed_density_and_its_duals_take_no_momentum():
    """§2.11 ``constraint`` + ``costs``: a table scales the density before
    the gap is taken, so with ``costs: {"gate": 2.0}`` the first update sees
    ``2 · ½ = 1.0``, not ``½``. And the duals' group carries no momentum:
    under ``{"name": "sgd", "momentum": 0.9}`` the featurizers step with a
    momentum buffer while the duals still ascend by exactly ``lr ·
    gradient`` — after two updates ``λ₁ = 0.5 · (gap₁ + gap₂)``, which a
    momentum buffer would have overshot (the first step is the same either
    way; the second is not)."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = dbm_doc()
    doc["method"]["train"]["objective"] = {
        "fit": ce_term(weight=1.0),
        "density": {
            "l1": "gate",
            "costs": {"gate": 2.0},
            "constraint": {"target": 0.1, "dual": {"lr": 0.5}},
        },
    }
    doc["method"]["train"]["optimizer"] = {"name": "sgd", "lr": 0.1, "momentum": 0.9}
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    trace = run_training(executor.doc, executor, request).constraint_trace["density"]
    assert trace[0]["value"] == pytest.approx(1.0, rel=1e-5)  # 2 · σ(0)
    gaps = [row["value"] - 0.1 for row in trace]
    assert trace[1]["lambda1"] == pytest.approx(0.5 * (gaps[0] + gaps[1]), rel=1e-5)
    assert trace[1]["lambda2"] == pytest.approx(
        0.5 * (gaps[0] ** 2 + gaps[1] ** 2), rel=1e-5
    )
    assert set(trace[0]) == {"step", "value", "target", "lambda1", "lambda2"}


def _drawn_executor(
    doc: dict,
    bundle,
    members: list[list[str]],
    *,
    extra: dict[str, list[Any]] | None = None,
    eval_member: int = 0,
    interning: Any = None,
):
    """A point executor whose counterfactual column holds ``members[i]`` for
    row ``i`` — what ``resolve_roles`` hands a fit for a drawn role: the rows
    with every member (plus ``extra`` columns, one value per row), and the
    fixed ``eval`` member as the field. ``interning`` is the campaign store
    handle, when the test is about the store."""
    from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
    from causalab.protocol.rules.document import validate_document

    parsed = parse_document(in_order(doc))
    validate_document(parsed, engine_is_local=True)
    rows = [
        {
            "input": text,
            "counterfactual_inputs": list(members[i]),
            "label": ANSWERS[i],
            **{key: values[i] for key, values in (extra or {}).items()},
        }
        for i, text in enumerate(BASES)
    ]
    return PointExecutor(
        parsed,
        bundle,
        role_rows={"base": rows, "counterfactual": rows},
        role_fields={
            "base": "input",
            "counterfactual": f"counterfactual_inputs[{eval_member}]",
        },
        load_tensors=lambda path: (_ for _ in ()).throw(KeyError(path)),
        load_table=None,
        grad_enabled=True,
        interning=interning,
    )


def _drawn_dbm_doc(*, draw: bool, eval_member: int | None = None) -> dict:
    doc = dbm_doc()
    if draw:
        doc["data"]["counterfactual"] = {
            **doc["data"]["counterfactual"],
            "field": "counterfactual_inputs",
            "draw": {"kind": "uniform"}
            if eval_member is None
            else {"kind": "uniform", "eval": eval_member},
        }
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"updates": 2}, "file_path": "t.safetensors"}
    )
    return doc


def _train(executor):
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    return run_training(executor.doc, executor, request)


def test_a_drawn_role_redraws_one_member_per_row_each_epoch():
    """§2.2 ``draw``: two members per row, three epochs — the outcome records
    one member list per epoch, every entry a valid index, the epochs not all
    alike (seeded, so this is a fact about seed 0, not a probability), and
    the fit ran every update."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    bundle = load_model(TINY_LLAMA)
    members = [[cf, f"{cf} again"] for cf in COUNTERFACTUALS]  # two distinct texts
    outcome = _train(_drawn_executor(_drawn_dbm_doc(draw=True), bundle, members))
    draws = outcome.draws["counterfactual"]
    assert draws["kind"] == "uniform" and draws["eval"] == 0
    epochs = draws["members"]
    assert len(epochs) == 3 and all(len(epoch) == len(BASES) for epoch in epochs)
    assert all(m in (0, 1) for epoch in epochs for m in epoch)
    assert len({tuple(epoch) for epoch in epochs}) > 1
    assert len(outcome.checkpoints) == 3  # six updates, photographed at 2, 4, 6


def test_a_drawn_role_with_one_member_per_row_fits_exactly_as_the_fixed_member_does():
    """The identity that pins the plumbing: with one member per row the draw
    can only pick it, so the drawn fit and the plain ``counterfactual_inputs[0]``
    fit take the same trajectory, checkpoint for checkpoint — the re-encoded
    minibatches, the bypassed inner store and the rewritten rows change no
    number."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    bundle = load_model(TINY_LLAMA)
    members = [[cf] for cf in COUNTERFACTUALS]
    drawn = _train(_drawn_executor(_drawn_dbm_doc(draw=True), bundle, members))
    fixed = _train(_drawn_executor(_drawn_dbm_doc(draw=False), bundle, members))
    assert drawn.draws["counterfactual"]["members"] == [[0] * len(BASES)] * 3
    assert fixed.draws == {}
    assert len(drawn.checkpoints) == len(fixed.checkpoints) == 3
    for a, b in zip(drawn.checkpoints, fixed.checkpoints):
        assert a.step == b.step
        assert torch.allclose(a.slots["gate"]["theta"], b.slots["gate"]["theta"])
        assert a.record["term.0"] == pytest.approx(b.record["term.0"], rel=1e-6)


def test_a_drawn_role_takes_ragged_member_counts_and_a_nonzero_eval_with_siblings():
    """§2.2: rows may hold different counts — the expanded frame is
    ``offsets[i] + pick`` and every drawn index is a valid member of *its*
    row. And ``eval: 1`` with a per-member sibling runs: inside the fit the
    collapsed lists hold the drawn member at every index, so
    ``counterfactual_inputs[1]`` reads the drawn member. The members of a row
    are *distinct texts* here, so the handed-frame check — which compares the
    minibatch's texts against ``select_field(drawn_row, "…[eval]")`` — fires
    on a wrong pick as well as a wrong offset; the ``eval: 1`` fit also pins
    the broadcast's *length* (without it ``[1]`` has no element on a
    one-member row), while the sibling's content is pinned by the
    ``_drawn_row`` unit assertions in the next test, not by this fit."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    bundle = load_model(TINY_LLAMA)
    ragged = [
        [cf] + [f"{cf} {w}" for w in ("again", "once more", "twice")[:i]]
        for i, cf in enumerate(COUNTERFACTUALS)
    ]
    assert sorted(len(m) for m in ragged) == [1, 2, 3, 4]
    outcome = _train(_drawn_executor(_drawn_dbm_doc(draw=True), bundle, ragged))
    epochs = outcome.draws["counterfactual"]["members"]
    assert len(epochs) == 3 and len(outcome.checkpoints) == 3
    assert all(0 <= m < len(ragged[i]) for epoch in epochs for i, m in enumerate(epoch))
    assert any(m > 0 for epoch in epochs for m in epoch)
    two = [[cf, f"{cf} again"] for cf in COUNTERFACTUALS]
    outcome = _train(
        _drawn_executor(
            _drawn_dbm_doc(draw=True, eval_member=1),
            bundle,
            two,
            extra={"counterfactual_inputs_variables": [["a", "b"]] * len(BASES)},
            eval_member=1,
        )
    )
    assert outcome.draws["counterfactual"]["eval"] == 1
    assert len(outcome.checkpoints) == 3


def test_a_mismatched_per_member_sibling_is_refused():
    """A `<column>_…` list whose length is not the member count is a
    per-member sibling that does not match — refused (P2) rather than left
    whole for the document to read member 0 of. `segments` (a declared frame)
    does not combine with a re-encoded role (P4)."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import (
        _check_member_siblings,
        _drawn_row,
    )
    from causalab.protocol.rules.errors import ProtocolError

    bundle = load_model(TINY_LLAMA)
    two = [[cf, f"{cf} again"] for cf in COUNTERFACTUALS]
    row = {
        "input": "x",
        "counterfactual_inputs": ["p", "q", "r"],
        "counterfactual_inputs_variables": [1, 2, 3],
        "counterfactual_inputs_note": "scalar",
        "label": "y",
    }
    row["counterfactual_inputs_source"] = ["synthetic"]  # the author's own list
    drawn = _drawn_row(row, "counterfactual_inputs", 2)
    assert drawn["counterfactual_inputs"] == ["r", "r", "r"]
    assert drawn["counterfactual_inputs_variables"] == [3, 3, 3]
    assert drawn["counterfactual_inputs_note"] == "scalar" and drawn["input"] == "x"
    assert drawn["counterfactual_inputs_source"] == ["synthetic"]  # not a sibling
    # the shape check is prepare-time and names the known siblings only
    _check_member_siblings(
        row, "counterfactual_inputs", 3, role="counterfactual", index=0
    )
    with pytest.raises(ProtocolError, match="per-member sibling"):
        _check_member_siblings(
            {**row, "counterfactual_inputs_variables": [1, 2]},
            "counterfactual_inputs",
            3,
            role="counterfactual",
            index=0,
        )
    # and through a fit: a short `_variables` table is refused at prepare
    with pytest.raises(ProtocolError, match="per-member sibling"):
        _train(
            _drawn_executor(
                _drawn_dbm_doc(draw=True),
                bundle,
                two,
                extra={"counterfactual_inputs_variables": [["only"]] * len(BASES)},
            )
        )
    # `segments` (a declared frame) does not combine with a re-encoded role
    framed = _drawn_dbm_doc(draw=True)
    framed["method"]["segments"] = {"frame": "chat"}
    with pytest.raises(ProtocolError) as err:
        _train(_drawn_executor(framed, bundle, two))
    assert err.value.code == "P4" and "declared frame" in str(err.value)


def chain_doc(lr) -> dict:
    """A rotation and a gate over its coordinates trained together (§2.11), the
    ``["rot", "gate"]`` chain, with ``lr`` as the document spells it."""
    doc = das_doc(seed=0, epochs=2)
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"},
        "gate": {"kind": "gate"},
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = ["rot", "gate"]
    doc["method"]["writes"]["patch"]["featurizer"] = ["rot", "gate"]
    doc["method"]["train"]["params"] = ["rot", "gate"]
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.01, {"l1": "gate"}]]
    doc["method"]["train"]["optimizer"] = {"name": "adamw", "lr": lr}
    doc["method"]["save"] = [
        ce_save(),
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"},
    ]
    return doc


def _fit_chain(lr):
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        chain_doc(lr),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    stages = run_training(executor.doc, executor, request).stages
    return stages["rot"].weight.detach().clone(), stages["gate"].theta.detach().clone()


def test_a_per_params_lr_steps_each_featurizer_at_its_own_rate():
    """One fit, two step sizes (§2.11 ``lr`` as a mapping over ``params``).
    Observed through the one thing a step size of zero guarantees: the tensor
    does not move. Frozen both ways gives the reference frame and the zero
    theta; freeing the gate alone moves theta and leaves the frame bit-equal;
    freeing the rotation alone moves the frame and leaves theta at zero. A
    single scalar lr could not produce either pattern."""
    frame0, theta0 = _fit_chain({"rot": 0.0, "gate": 0.0})
    assert torch.equal(theta0, torch.zeros_like(theta0))

    frame_g, theta_g = _fit_chain({"rot": 0.0, "gate": 0.1})
    assert torch.equal(frame_g, frame0)  # the rotation's group had lr 0
    assert not torch.allclose(theta_g, theta0)  # the gate's group had lr 0.1

    frame_r, theta_r = _fit_chain({"rot": 1e-3, "gate": 0.0})
    assert torch.equal(theta_r, theta0)  # the gate's group had lr 0
    assert not torch.allclose(frame_r, frame0)  # the rotation's group moved


def test_early_stop_returns_the_best_fit_not_the_last(monkeypatch):
    """``early_stop`` selects a fit by its eval score, so the fit it selected
    is the one that must come back.

    Nothing snapshotted the parameters: ``best`` was tracked, the loop broke
    after ``patience`` non-improving evals, and the stages returned were the
    **last** ones — the worst of the tail. A fit that scores held-out
    1.000 at every seed has last and best coinciding *at ceiling*; that is
    luck, not a property, and off ceiling nothing in the saved bundle says which
    weights you have.

    The eval score is scripted here rather than engineered out of a random
    model: what is under test is the selection, and a deterministic peak is
    the only way to assert "the peak, not the end" without a flaky fit.
    """
    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.io.env import ResolutionEnv

    doc_raw = das_doc(seed=0, epochs=5)
    doc_raw["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": ce_term()},
    }
    doc_raw["method"]["train"]["early_stop"] = {
        "on": "ce",
        "patience": 10,
        "mode": "max",
    }

    scores = [0.1, 0.9, 0.5, 0.4, 0.3]  # peak at the second eval
    seen: list[dict[str, torch.Tensor]] = []

    def fake_eval(doc, executor, request, split, *, eval_executor=None):
        stage = executor.stage_cache["rot"]
        seen.append({k: v.detach().clone() for k, v in stage.state_dict().items()})
        return {"ce": scores[len(seen) - 1]}

    monkeypatch.setattr(train_module, "_run_eval", fake_eval)

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = train_module.run_training(executor.doc, executor, request)

    assert len(seen) == len(scores)  # patience never fires: five evals ran
    peak, final = seen[1], seen[-1]
    key = "parametrizations.weight.original"  # what the optimizer actually steps
    assert not torch.allclose(peak[key], final[key])  # the fit really moved on

    returned = outcome.stages["rot"].state_dict()
    torch.testing.assert_close(returned[key], peak[key], atol=0.0, rtol=0.0)

    assert outcome.eval_score is not None
    assert outcome.eval_score.selected == "early_stop.best"
    # the reported score describes the weights that came back, not the last pass
    assert outcome.eval_score.metrics["ce"] == 0.9


def test_without_early_stop_the_last_fit_is_the_one_returned(monkeypatch):
    """Nothing is selecting, so nothing is restored — and the record says so
    rather than leaving a reader to guess."""
    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.io.env import ResolutionEnv

    doc_raw = das_doc(seed=0, epochs=3)
    doc_raw["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": ce_term()},
    }

    scores = [0.1, 0.9, 0.2]
    seen: list[dict[str, torch.Tensor]] = []

    def fake_eval(doc, executor, request, split, *, eval_executor=None):
        stage = executor.stage_cache["rot"]
        seen.append({k: v.detach().clone() for k, v in stage.state_dict().items()})
        return {"ce": scores[len(seen) - 1]}

    monkeypatch.setattr(train_module, "_run_eval", fake_eval)

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = train_module.run_training(executor.doc, executor, request)

    key = "parametrizations.weight.original"
    torch.testing.assert_close(
        outcome.stages["rot"].state_dict()[key], seen[-1][key], atol=0.0, rtol=0.0
    )
    assert outcome.eval_score is not None
    assert outcome.eval_score.selected == "last"
    assert outcome.eval_score.metrics["ce"] == 0.2
    assert outcome.eval_score.passes == 3


def test_max_mode_on_a_falling_metric_stops_at_patience_plus_two(monkeypatch):
    """The early-stop arithmetic the graph-cohort goldens' fixture rests on
    (``tests/golden/test_graph_cohort.py``, ``_early_stop_docs``): the first
    eval improves because nothing precedes it, a later eval that does not
    beat the best is stale — a tie included, the comparison is strict — and
    the fit stops once ``stale > patience``. So ``mode: max`` on a metric
    that only falls stops at pass ``patience + 2`` whatever the curve's
    shape, and the weights that come back are the first eval's. Scripted
    scores, as above: the arithmetic is under test, not a fit."""
    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.io.env import ResolutionEnv

    patience = 2
    doc_raw = das_doc(seed=0, epochs=8)
    doc_raw["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": ce_term()},
    }
    doc_raw["method"]["train"]["early_stop"] = {
        "on": "ce",
        "patience": patience,
        "mode": "max",
    }

    scores = [0.9, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]  # a tie, then only falls
    seen: list[dict[str, torch.Tensor]] = []

    def fake_eval(doc, executor, request, split, *, eval_executor=None):
        stage = executor.stage_cache["rot"]
        seen.append({k: v.detach().clone() for k, v in stage.state_dict().items()})
        return {"ce": scores[len(seen) - 1]}

    monkeypatch.setattr(train_module, "_run_eval", fake_eval)

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = train_module.run_training(executor.doc, executor, request)

    assert len(seen) == patience + 2  # improve, stale, stale, stale > patience
    assert outcome.eval_score is not None
    assert outcome.eval_score.passes == patience + 2
    assert outcome.eval_score.selected == "early_stop.best"
    assert outcome.eval_score.metrics["ce"] == scores[0]
    key = "parametrizations.weight.original"
    assert not torch.allclose(seen[0][key], seen[-1][key])  # the fit moved on
    torch.testing.assert_close(
        outcome.stages["rot"].state_dict()[key], seen[0][key], atol=0.0, rtol=0.0
    )


def test_an_update_counted_eval_is_refused_rather_than_never_run():
    """This loop only reaches an eval on an epoch boundary, so an ``updates``
    counter would run no eval at all and still save the fit."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.protocol.rules.errors import ProtocolError
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc_raw = das_doc(seed=0, epochs=1)
    doc_raw["method"]["train"]["eval"] = {
        "every": {"updates": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": ce_term()},
    }
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc_raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        grad_enabled=False,
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    with pytest.raises(ProtocolError, match="must count epochs"):
        run_training(executor.doc, executor, request)


def test_validation_accepts_the_train_docs():
    for raw in (das_doc(), dbm_doc()):
        from tests.protocol._docs import in_order

        validate_document(parse_document(in_order(raw)), engine_is_local=True)


def test_a_gate_fit_reports_whether_its_mask_is_a_mask():
    """The DBM finding's non-GPU half.

    An earlier version of `demos/methods/protocols/dbm.json` produced **no**
    dimension outside [0.1, 0.9] and still scored **1.000** at a late layer —
    because
    `Gate._mask` returns a *hard* `θ > 0` mask in eval mode, so an unseparated
    θ makes the mask a coin flip on gradient noise. Roughly half the dimensions
    swap, which at the readout layer scores 1.000. A meaningless mask and a
    perfect number, with nothing in the saved outputs to tell them apart.

    Asserted on the fit's own report rather than on a value: what is under test
    is that the fact is *recorded*, not that a random model separates θ. The
    retune that makes θ separate needs a GPU run and is not asserted here.
    """
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        dbm_doc(),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)

    report = outcome.diagnostics["gate"]
    width = int(report["width"])
    assert width == outcome.stages["gate"].theta.numel()
    assert 0.0 <= report["decisive_fraction"] <= 1.0
    assert 0 <= report["hard_mask_size"] <= width
    # the hard mask is what the eval-mode score was computed through, so it has
    # to be reported as a count of *this* gate, not a fraction of some other
    assert report["hard_mask_size"] == float((outcome.stages["gate"].theta > 0).sum())


def test_a_subspace_fit_reports_its_rotation_not_a_mask():
    """Only a gate has a mask to be indecisive about — the report is per kind,
    not a fixed schema every fit has to fill with zeros. What a subspace *can*
    say about itself is how orthonormal the rotation it saved is, and whether
    that is within the bar a later ``init`` load applies. This is the runtime
    half of the claim; ``test_featurizers.py`` pins the static half (the map's
    own bound stays inside the load tolerance)."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.neural.shared.featurizers import ORTHONORMAL_TOLERANCE
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        das_doc(seed=0, epochs=1),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    diagnostics = run_training(executor.doc, executor, request).diagnostics
    assert set(diagnostics) == {"rot"}
    report = diagnostics["rot"]
    assert "decisive_fraction" not in report and "hard_mask_size" not in report
    assert report["orthonormality_deviation"] < ORTHONORMAL_TOLERANCE
    assert report["within_tolerance"] == 1.0


def js_dbm_doc(*, restrict: object = ("one", "two", "three", "four")) -> dict:
    """The DBM fit with a ``js`` objective toward the clean counterfactual
    distribution — the DCM loss — restricted to the answer set. ``restrict``
    is the literal list by default; ``"valid"`` names the per-row column the
    drive helper fills below."""
    doc = dbm_doc()
    doc["method"]["reads"]["logits_cf"] = {"site": "lm_head", "pos": {"index": -1}}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"].append("logits_cf")
    js = {
        "kind": "js",
        # the save below carries it too, so the object form (§2.7)
        "target": {"read": "logits_cf", "model": UNWRITTEN},
        "restrict": list(restrict) if not isinstance(restrict, str) else restrict,
    }
    doc["method"]["train"]["objective"] = [
        [1.0, term("logits", "patched", dict(js))],
        [0.01, {"l1": "gate"}],
    ]
    doc["method"]["save"].insert(0, saved("logits", "patched", "js.json", js))
    return doc


@pytest.mark.parametrize("restrict", [("one", "two", "three", "four"), "valid"])
def test_a_js_objective_reaches_theta_through_both_restrict_spellings(restrict):
    """The objective-side twin of `js` (spec §2.11) has a gradient into the
    gate: the fit moves θ, the target read is served (it is a constant model's
    read), and the literal and column spellings of `restrict` are one
    arithmetic — the same fit, to the bit, when they name the same set."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        js_dbm_doc(restrict=restrict),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={
            "label": ANSWERS,
            "valid": [["one", "two", "three", "four"]] * len(BASES),
        },
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    gate = run_training(executor.doc, executor, request).stages["gate"]
    assert not torch.allclose(gate.theta, torch.zeros_like(gate.theta))
    _JS_THETAS.append(gate.theta.detach().clone())
    if len(_JS_THETAS) == 2:
        assert torch.equal(_JS_THETAS[0], _JS_THETAS[1])


_JS_THETAS: list[torch.Tensor] = []


def clamp_dbm_doc(*, lr: float = 5.0) -> dict:
    """The DBM fit under `parametrization: clamp` (§2.5): no anneal, the mask
    projected onto [0, 1] after every step. A large lr so one Adam step
    reaches a pole and the projection is what keeps θ on the interval."""
    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"]["parametrization"] = "clamp"
    del doc["method"]["train"]["anneal"]
    doc["method"]["train"]["optimizer"]["lr"] = lr
    return doc


def hard_concrete_dbm_doc(*, seed: int = 0) -> dict:
    """The DBM fit under `parametrization: hard_concrete` (§2.5) with the
    matching `l0` penalty. The constants are left unauthored (the defaulted
    path, which stamps the default anyway) and the anneal addresses β from a
    start that is NOT the default, so the loop's value is visibly the
    schedule's — authoring `temperature` beside the anneal is refused (rule 4)."""
    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"] = {
        "kind": "gate",
        "parametrization": "hard_concrete",
    }
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.05, {"l0": "gate"}]]
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.2, 0.5]}
    doc["method"]["train"]["seed"] = seed
    return doc


def _fit_hard_concrete(seed: int):
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        hard_concrete_dbm_doc(seed=seed),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    return run_training(executor.doc, executor, request)


def test_a_hard_concrete_gate_fit_samples_once_per_step_from_its_own_generator(
    monkeypatch,
):
    """§2.5 ``hard_concrete`` + §2.11 ``l0``: every training forward goes
    through the *sampled* mask, drawn exactly once per optimizer step from a
    generator that is the fit's own (not torch's global one) and shared by the
    read and the write the gate sits on — so the interchange is one mask, not
    two draws. Counted on the gate itself, so a fit that silently fell back to
    the deterministic mask, or drew twice, would fail here."""
    import math

    from causalab.neural.shared.featurizers import Gate

    resamples: list[torch.Generator] = []
    sampled: list[int] = []
    real_resample, real_sampled = Gate.resample, Gate.sampled_mask

    def counting_resample(self, generator):
        resamples.append(generator)
        return real_resample(self, generator)

    def counting_sampled(self):
        sampled.append(1)
        return real_sampled(self)

    monkeypatch.setattr(Gate, "resample", counting_resample)
    monkeypatch.setattr(Gate, "sampled_mask", counting_sampled)
    outcome = _fit_hard_concrete(seed=0)
    doc = hard_concrete_dbm_doc()
    steps = doc["method"]["train"]["steps"]["epochs"] * math.ceil(
        len(BASES) / doc["method"]["train"]["batch"]["pairs"]
    )
    assert len(resamples) == steps  # one draw per optimizer step
    # the read of v_cf and the write into the base both featurize through the
    # gate and share one sampled mask, computed once per step from the one
    # draw and replayed per access (featurizer_cache); the eval passes, the
    # regularizer and the diagnostics never sample. Exact because this
    # document's rows fit one window per step: a budget that split a step
    # would evaluate the shared mask once per window
    assert len(sampled) == steps
    assert all(isinstance(g, torch.Generator) for g in resamples)
    assert all(g is resamples[0] for g in resamples)  # the fit's own, every step
    assert resamples[0] is not torch.default_generator

    gate = outcome.stages["gate"]
    assert gate.parametrization == "hard_concrete"
    assert not torch.allclose(gate.theta, torch.zeros_like(gate.theta))
    assert not gate.training
    report = outcome.diagnostics["gate"]
    assert report["parametrization"] == "hard_concrete"
    assert report["stretch"] == [-0.1, 1.1]  # the default, stamped though unauthored
    assert report["hard_mask_size"] == float((gate.theta > gate.hard_threshold()).sum())


def test_a_hard_concrete_fit_is_a_function_of_its_seed_and_anneals_beta():
    """Two runs at one seed are bit-identical whatever the global RNG did in
    between, another seed differs, and the anneal addressed β from ITS start
    (1.0, not the default 2/3), which is what the loop uses when a schedule is
    declared."""
    outcome = _fit_hard_concrete(seed=0)
    gate = outcome.stages["gate"]
    assert gate.temperature < 2 / 3  # ramped 1.0 → 0.2 over half the steps
    assert outcome.diagnostics["gate"]["temperature"] == pytest.approx(gate.temperature)
    torch.manual_seed(12345)  # the global stream is not what the fit draws from
    torch.rand(7)
    again = _fit_hard_concrete(seed=0).stages["gate"].theta
    assert torch.equal(again, gate.theta)
    other = _fit_hard_concrete(seed=1).stages["gate"].theta
    assert not torch.equal(other, gate.theta)


def test_an_authored_temperature_beside_its_anneal_is_refused():
    """Rule 4: ``_set_anneal`` writes the schedule's start onto the gate before
    the first forward, so an authored β would never take effect."""
    from causalab.protocol.rules.errors import ValidationError

    from tests.protocol._docs import in_order

    doc = hard_concrete_dbm_doc()
    doc["method"]["featurizers"]["gate"]["temperature"] = 0.5
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(doc)), engine_is_local=True)
    assert err.value.rule == 4 and "authors temperature" in str(err.value)
    del doc["method"]["train"]["anneal"]
    validate_document(parse_document(in_order(doc)), engine_is_local=True)


def test_the_mask_penalty_is_keyed_on_the_map():
    """Rule 4 (§2.11): ``l0`` pairs with a sampled mask and ``l1`` with a
    deterministic one — and the loop refuses the crossing again for a document
    that arrived unvalidated."""
    from causalab.protocol.rules.errors import ProtocolError, ValidationError

    from tests.protocol._docs import in_order

    l1_on_sampled = hard_concrete_dbm_doc()
    l1_on_sampled["method"]["train"]["objective"] = [
        [1.0, ce_term()],
        [0.05, {"l1": "gate"}],
    ]
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(l1_on_sampled)), engine_is_local=True)
    assert err.value.rule == 4 and "'l0'" in str(err.value)
    l0_on_sigmoid = dbm_doc()
    l0_on_sigmoid["method"]["train"]["objective"] = [
        [1.0, ce_term()],
        [0.01, {"l0": "gate"}],
    ]
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(l0_on_sigmoid)), engine_is_local=True)
    assert err.value.rule == 4 and "'l1'" in str(err.value)

    from causalab.neural.engines.pytorch_hooks.train import _regularizer
    from causalab.neural.shared.featurizers import Gate

    with pytest.raises(ProtocolError, match="does not pair"):
        _regularizer("l1", ["g"], {"g": Gate(4, parametrization="hard_concrete")})
    with pytest.raises(ProtocolError, match="does not pair"):
        _regularizer("l0", ["g"], {"g": Gate(4)})


def test_validation_accepts_the_hard_concrete_doc_and_refuses_its_fields_elsewhere():
    from causalab.protocol.rules.errors import ParseError

    from tests.protocol._docs import in_order

    validate_document(
        parse_document(in_order(hard_concrete_dbm_doc())), engine_is_local=True
    )
    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"]["stretch"] = [-0.1, 1.1]
    with pytest.raises(ParseError, match="hard_concrete"):
        parse_document(in_order(doc))


def test_a_clamp_gate_fit_is_projected_onto_the_unit_interval_and_rounds():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        clamp_dbm_doc(),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    gate = outcome.stages["gate"]
    theta = gate.theta.detach()
    assert gate.parametrization == "clamp"
    assert gate.temperature == 1.0  # nothing annealed
    # projected after every step: on the interval although an lr-5 Adam step
    # leaves it by ~5, and the projection is visibly active (a unit that was
    # clipped sits exactly on a pole; one whose gradient flipped sign may not)
    assert float(theta.min()) >= 0.0 and float(theta.max()) <= 1.0
    assert bool(torch.any((theta == 0.0) | (theta == 1.0)))
    from causalab.neural.shared.results import MASK_DECISIVE_MARGIN

    report = outcome.diagnostics["gate"]
    assert report["parametrization"] == "clamp"
    # both diagnostics are read through the clamp map: θ > ½, and |θ − ½|
    assert report["hard_mask_size"] == float((theta > 0.5).sum())
    assert report["decisive_fraction"] == pytest.approx(
        float(((theta - 0.5).abs() > MASK_DECISIVE_MARGIN).float().mean())
    )
    assert not gate.training


def test_validation_refuses_a_temperature_anneal_on_a_clamp_gate():
    from causalab.protocol.rules.errors import ValidationError

    from tests.protocol._docs import in_order

    doc = clamp_dbm_doc()
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(doc)), engine_is_local=True)
    assert err.value.rule == 4


def test_a_gate_fill_is_where_the_fit_starts_and_is_recorded():
    """`init: {fill: 0.99}` (§2.5) — every unit starts at mask value 0.99,
    "everything patched but a little", so before any step the hard mask keeps
    every unit; the fit records the fill beside its diagnostics."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = clamp_dbm_doc(lr=1e-3)
    doc["method"]["featurizers"]["gate"]["init"] = {"fill": 0.99}
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    gate = executor.stage("gate")
    assert torch.allclose(gate.theta, torch.full_like(gate.theta, 0.99))
    assert float(gate.hard_mask().sum()) == gate.theta.numel()
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    assert outcome.diagnostics["gate"]["init_fill"] == 0.99
    # a small lr: the fit moved, and stayed near its start
    theta = outcome.stages["gate"].theta.detach()
    assert not torch.allclose(theta, torch.full_like(theta, 0.99))
    assert float(theta.min()) > 0.9


def controlled_dbm_doc() -> dict:
    """The DCM sweep in one document (§2.11 `control`): a clamp gate starting
    all-patched, the sparsity weight moved by a PID so the kept-unit count
    follows a ramp from every unit to none over the run."""
    doc = clamp_dbm_doc(lr=0.1)
    doc["method"]["featurizers"]["gate"]["init"] = {"fill": 0.99}
    doc["method"]["train"]["objective"] = {
        "fit": ce_term(weight=1.0),
        "sparsity": {"weight": 0.01, "l1": "gate"},
    }
    doc["method"]["train"]["steps"] = {"epochs": 10}
    doc["method"]["train"]["control"] = {
        "train.objective.sparsity.weight": {
            "kind": "pid",
            "signal": {"hard_mask_size": "gate"},
            "setpoint": {"ramp": [16, 0, 1.0]},
            "gains": {"kp": 0.5, "ki": 0.05},
        }
    }
    return doc


def test_a_regularizer_sum_is_the_mean_times_the_unit_count():
    """§2.11 ``reduce``: the sum over the concatenated per-unit quantities is
    the mean scaled by how many units there are — so under ``sum`` a kept unit
    costs the term's weight whatever the gate's size."""
    from causalab.neural.engines.pytorch_hooks.train import _regularizer
    from causalab.neural.shared.featurizers import Gate

    stages = {"a": Gate(5), "b": Gate(3)}
    with torch.no_grad():
        stages["a"].theta.copy_(torch.tensor([2.0, -1.0, 0.5, 0.0, 3.0]))
        stages["b"].theta.copy_(torch.tensor([-2.0, 1.0, 0.0]))
    mean = _regularizer("l1", ["a", "b"], stages)
    summed = _regularizer("l1", ["a", "b"], stages, "sum")
    assert torch.allclose(summed, mean * 8)
    assert torch.allclose(_regularizer("l1", ["a", "b"], stages, "mean"), mean)


def test_a_regularizer_cost_scales_one_targets_quantities_before_the_concatenation():
    """§2.11 ``costs``: a table multiplies each target's per-unit quantities
    by its entry (1 when unlisted) before they are concatenated, under either
    reduction; ``parameter_count`` divides each target by its own element
    count, so under ``sum`` the term is the sum of per-featurizer means."""
    from causalab.neural.engines.pytorch_hooks.train import _regularizer
    from causalab.neural.shared.featurizers import Gate

    stages = {"a": Gate(5), "b": Gate(3)}
    with torch.no_grad():
        stages["a"].theta.copy_(torch.tensor([2.0, -1.0, 0.5, 0.0, 3.0]))
        stages["b"].theta.copy_(torch.tensor([-2.0, 1.0, 0.0]))
    a, b = stages["a"].soft_mask(), stages["b"].soft_mask()
    costed = _regularizer("l1", ["a", "b"], stages, "sum", {"b": 0.25})
    assert torch.allclose(costed, a.sum() + 0.25 * b.sum())
    costed_mean = _regularizer("l1", ["a", "b"], stages, "mean", {"a": 2.0, "b": 0.5})
    assert torch.allclose(costed_mean, torch.cat([2.0 * a, 0.5 * b]).mean())
    per_count = _regularizer("l1", ["a", "b"], stages, "sum", "parameter_count")
    assert torch.allclose(per_count, a.mean() + b.mean())
    # a cost of 1 everywhere, and no costs at all, are one number
    plain = _regularizer("l1", ["a", "b"], stages, "sum")
    assert torch.allclose(
        _regularizer("l1", ["a", "b"], stages, "sum", {"a": 1.0}), plain
    )
    # a non-gate target under `parameter_count` counts every slot it matched
    from causalab.neural.shared.featurizers import Subspace

    rot = Subspace(4, 2, parametrization="cayley")
    elements = sum(p.numel() for p in rot.slot_params().values())
    assert torch.allclose(
        _regularizer("l2", ["rot"], {"rot": rot}, "sum", "parameter_count"),
        _regularizer("l2", ["rot"], {"rot": rot}, "sum") / elements,
    )


def test_costs_compose_with_l0_and_a_grouped_gate_counts_its_units():
    """§2.11 ``costs`` under ``l0``: two ``hard_concrete`` gates under
    ``reduce: sum`` with ``parameter_count`` sum their expected kept
    *fractions*; a table scales one gate's expected L0 before the sum. And
    ``parameter_count`` divides by ``θ`` as stored: a gate grouped ``head``
    over 8 coordinates in 2 heads has 2 units, so its divisor is 2 — not the
    8 coordinates the heads span — the same per-unit reading the quantity
    itself has."""
    from causalab.neural.engines.pytorch_hooks.train import _regularizer
    from causalab.neural.shared.featurizers import Gate

    stages = {
        "a": Gate(5, parametrization="hard_concrete"),
        "b": Gate(3, parametrization="hard_concrete"),
    }
    with torch.no_grad():
        stages["a"].theta.copy_(torch.tensor([2.0, -1.0, 0.5, 0.0, 3.0]))
        stages["b"].theta.copy_(torch.tensor([-2.0, 1.0, 0.0]))
    a, b = stages["a"].expected_l0(), stages["b"].expected_l0()
    assert torch.allclose(
        _regularizer("l0", ["a", "b"], stages, "sum", "parameter_count"),
        a.mean() + b.mean(),
    )
    assert torch.allclose(
        _regularizer("l0", ["a", "b"], stages, "sum", {"b": 0.25}),
        a.sum() + 0.25 * b.sum(),
    )
    grouped = Gate(8, group="head", groups=(2, 4))
    assert grouped.theta.numel() == 2
    with torch.no_grad():
        grouped.theta.copy_(torch.tensor([1.0, -1.0]))
    per_unit = _regularizer("l1", ["g"], {"g": grouped}, "sum", "parameter_count")
    units = grouped.soft_mask().flatten()
    assert units.numel() == 2
    assert torch.allclose(per_unit, units.sum() / 2)
    assert not torch.allclose(per_unit, units.sum() / 8)


def fraction_controlled_dbm_doc() -> dict:
    """`controlled_dbm_doc` with the fraction-valued signal (§2.11): the
    setpoint ramps the kept *fraction* from 1 to 0, so the same gains and ramp
    would serve a gate of any width."""
    doc = controlled_dbm_doc()
    doc["method"]["train"]["control"]["train.objective.sparsity.weight"] = {
        "kind": "pid",
        "signal": {"hard_mask_fraction": "gate"},
        "setpoint": {"ramp": [1.0, 0.0, 1.0]},
        "gains": {"kp": 0.5, "ki": 0.8},
    }
    return doc


def test_a_fraction_valued_signal_is_the_kept_count_over_the_units():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    from tests.protocol._docs import in_order

    validate_document(
        parse_document(in_order(fraction_controlled_dbm_doc())), engine_is_local=True
    )
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        fraction_controlled_dbm_doc(),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    target = "train.objective.sparsity.weight"
    trace = outcome.control_trace[target]
    # the fit starts all-patched: 16 of 16 units kept is a fraction of 1
    assert trace[0]["signal"] == 1.0 and trace[0]["setpoint"] == pytest.approx(0.95)
    assert all(0.0 <= row["signal"] <= 1.0 for row in trace)
    assert outcome.controls[target]["setpoint_final"] == 0.0
    assert outcome.controls[target]["final"] > outcome.controls[target]["initial"]


def test_a_controlled_weight_follows_the_fit_and_is_recorded():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        controlled_dbm_doc(),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    target = "train.objective.sparsity.weight"
    record = outcome.controls[target]
    trace = outcome.control_trace[target]
    # ten epochs of two batches: one controller update per optimizer update
    assert record["updates"] == 20.0 and len(trace) == 20
    assert [row["step"] for row in trace] == [float(s) for s in range(1, 21)]
    assert record["initial"] == 0.01
    # the fit starts all-patched while the setpoint falls, so the kept count
    # sits above the setpoint and the controller raises the weight
    assert record["final"] > record["initial"]
    assert trace[0]["signal"] == 16.0 and trace[0]["setpoint"] == pytest.approx(15.2)
    assert record["setpoint_final"] == 0.0
    assert all(0.0 <= row["signal"] <= 16.0 for row in trace)


def test_all_control_signals_are_read_in_one_host_copy() -> None:
    """``_read_signals`` over every member of a step is ``read_signal`` per
    controller with the counts copied to the host together: the same
    numbers — the same left-to-right float sum — one synchronization."""
    from types import SimpleNamespace

    from causalab.neural.engines.pytorch_hooks import train as train_module
    from causalab.neural.shared.featurizers import Gate

    gates = [Gate(8).eval() for _ in range(3)]
    for i, gate in enumerate(gates):
        with torch.no_grad():
            gate.theta.copy_(torch.linspace(-1, 1, 8) + 0.3 * i)

    def control(stages: list[Gate], signal: str) -> Any:
        return train_module._Control(  # pyright: ignore[reportPrivateUsage]
            controller=None,  # type: ignore[arg-type]  # never consulted by a read
            ramp=(0.0, 1.0, 1.0),
            initial=0.0,
            signal_stages=stages,
            term="l1",
            signal=signal,
        )

    fits = [
        SimpleNamespace(controls={"size": control(gates[:2], "hard_mask_size")}),
        SimpleNamespace(
            controls={"fraction": control(gates[1:], "hard_mask_fraction")}
        ),
    ]
    signals = train_module._read_signals(fits)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    for fit in fits:
        for target, ctl in fit.controls.items():
            counts = ctl.signal_counts()
            expected = ctl.read_signal(
                [float(count) for count, _ in counts], [units for _, units in counts]
            )
            assert signals[(id(fit), target)] == expected
    assert signals[(id(fits[0]), "size")] == float(
        sum(int((g.theta > 0).sum()) for g in gates[:2])
    )
    assert 0.0 < signals[(id(fits[1]), "fraction")] < 1.0
    assert train_module._read_signals([]) == {}  # pyright: ignore[reportPrivateUsage]


def test_checkpoint_steps_space_the_run_and_always_end_on_its_last_update():
    from causalab.neural.engines.pytorch_hooks.train import _checkpoint_steps
    from causalab.protocol.schema import parse_document

    from tests.protocol._docs import in_order

    def steps(every: dict, total: int, per_epoch: int) -> set[int]:
        doc = clamp_dbm_doc()
        doc["method"]["save"].append(
            {"kind": "trajectory", "every": every, "file_path": "t.safetensors"}
        )
        return _checkpoint_steps(parse_document(in_order(doc)), total, per_epoch)

    assert steps({"count": 4}, 6, 2) == {1, 3, 4, 6}
    assert steps({"count": 10}, 6, 2) == {
        1,
        2,
        3,
        4,
        5,
        6,
    }  # never more than the run has
    assert steps({"updates": 2}, 6, 2) == {2, 4, 6}
    assert steps({"updates": 4}, 6, 2) == {4, 6}  # the final update always
    assert steps({"epochs": 1}, 6, 2) == {2, 4, 6}
    assert _checkpoint_steps(parse_document(in_order(clamp_dbm_doc())), 6, 2) == set()


def test_a_trajectory_photographs_the_fit_at_its_scheduled_updates():
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = clamp_dbm_doc(lr=0.1)
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"count": 3}, "file_path": "t.safetensors"}
    )
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    # three epochs of two batches: six updates, photographed at 2, 4 and 6
    assert [c.step for c in outcome.checkpoints] == [2, 4, 6]
    assert [c.epoch for c in outcome.checkpoints] == [0, 1, 2]
    last = outcome.checkpoints[-1]
    assert torch.equal(
        last.slots["gate"]["theta"], outcome.stages["gate"].theta.detach().cpu()
    )
    for checkpoint in outcome.checkpoints:
        record = checkpoint.record
        assert record["step"] == checkpoint.step
        assert {"loss", "term.0", "weight.0", "term.1", "weight.1"} <= set(record)
        assert record["weight.1"] == 0.01  # the positional L1 weight, by index
        theta = checkpoint.slots["gate"]["theta"]
        assert record["gate.hard_mask_size"] == float(
            (theta > 0.5).sum()
        )  # clamp: θ > ½
        assert 0.0 <= record["gate.decisive_fraction"] <= 1.0
        assert not theta.requires_grad


# --------------------------------------------------------------------------- #
# §2.11 `anneal` on an objective term's weight — the open-loop sparsity sweep
# --------------------------------------------------------------------------- #


def annealed_weight_dbm_doc(schedule) -> dict:
    """The DCM sweep with the penalty weight on a *declared* path instead of
    a controller (§2.11): a clamp gate, named terms, the sparsity weight
    annealed by ``schedule``, and a trajectory photographing every update so
    the weight each update used is on record."""
    doc = clamp_dbm_doc(lr=0.1)
    doc["method"]["featurizers"]["gate"]["init"] = {"fill": 0.99}
    doc["method"]["train"]["objective"] = {
        "fit": ce_term(weight=1.0),
        "sparsity": {"weight": 0.01, "l1": "gate"},
    }
    doc["method"]["train"]["steps"] = {"epochs": 3}
    doc["method"]["train"]["anneal"] = {"train.objective.sparsity.weight": schedule}
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"updates": 1}, "file_path": "t.safetensors"}
    )
    return doc


def _fit_outcome(doc: dict):
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    return run_training(executor.doc, executor, request)


def test_a_frozen_gate_keeps_every_unit_it_dropped_and_records_the_rule():
    """§2.5 ``dead: {freeze_after: 1}``: a unit hard-off after one step is
    frozen there. Under a penalty large enough that the first Adam step
    closes every unit, the whole gate freezes at step 1 and every later
    checkpoint holds the same θ to the bit; under the plain penalty the kept
    count is non-increasing across checkpoints *by construction* — a unit
    that is on may close, a unit that closed cannot reopen — which is the
    nested sequence the DCM sweep relies on."""
    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"]["dead"] = {"freeze_after": 1}
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [1000.0, {"l1": "gate"}]]
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"count": 3}, "file_path": "t.safetensors"}
    )
    outcome = _fit_outcome(doc)
    gate = outcome.stages["gate"]
    report = outcome.diagnostics["gate"]
    assert report["dead"] == {"freeze_after": 1}
    assert report["frozen_units"] == float(gate.theta.numel())
    assert report["hard_mask_size"] == 0.0 and report["reawakened_units"] == 0.0
    thetas = [c.slots["gate"]["theta"] for c in outcome.checkpoints]
    assert len(thetas) == 3 and all(torch.equal(t, thetas[0]) for t in thetas)
    assert torch.equal(thetas[-1], gate.theta.detach().cpu())
    assert (thetas[0] < 0).all()  # closed by the step it froze at, not at the start

    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"]["dead"] = {"freeze_after": 1}
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"count": 6}, "file_path": "t.safetensors"}
    )
    outcome = _fit_outcome(doc)
    kept = [float((c.slots["gate"]["theta"] > 0).sum()) for c in outcome.checkpoints]
    assert len(kept) == 6 and all(a >= b for a, b in zip(kept, kept[1:])), kept
    assert outcome.diagnostics["gate"]["reawakened_units"] == 0.0


def test_a_leak_moves_a_saturated_unit_the_plain_gate_cannot():
    """§2.5 ``dead: {leak: ε}``: every unit starts at mask 1e-9 (θ ≈ −20.7),
    where σ' is ~1e-9 and plain SGD moves nothing measurable in six steps;
    the leak hands θ ``ε·∂L/∂m`` instead, and the same fit moves. The rule is
    recorded, and ``reawakened_units`` is reported under both."""
    moved = {}
    for dead in (None, {"leak": 0.5}):
        doc = dbm_doc()
        del doc["method"]["train"]["anneal"]
        doc["method"]["train"]["objective"] = [[1.0, ce_term()]]
        doc["method"]["train"]["optimizer"] = {"name": "sgd", "lr": 10.0}
        doc["method"]["featurizers"]["gate"]["init"] = {"fill": 1e-9}
        if dead is not None:
            doc["method"]["featurizers"]["gate"]["dead"] = dead
        outcome = _fit_outcome(doc)
        theta = outcome.stages["gate"].theta.detach()
        start = math.log(1e-9 / (1 - 1e-9))
        moved[str(dead)] = float((theta - start).abs().max())
        report = outcome.diagnostics["gate"]
        assert "reawakened_units" in report and report["frozen_units"] == 0.0
        assert report.get("dead") == dead
    assert moved["None"] < 1e-5, moved
    assert (
        moved["{'leak': 0.5}"] > 100 * moved["None"] and moved["{'leak': 0.5}"] > 1e-4
    ), moved


def test_an_annealed_weight_follows_its_schedule_and_is_recorded():
    """The weight update ``s`` (zero-based) uses is the schedule at ``s`` of
    the run's updates — ``from`` at the first, ``to`` from ``frac · steps`` on
    — and a checkpoint taken after that update records it as
    ``weight.sparsity``: the same slot a controlled weight lands in."""
    validate_document(
        parse_document(in_order(annealed_weight_dbm_doc([0.01, 1.0, 1.0]))),
        engine_is_local=True,
    )
    outcome = _fit_outcome(annealed_weight_dbm_doc([0.01, 1.0, 1.0]))
    # three epochs of two batches: six updates, one checkpoint each
    assert [c.step for c in outcome.checkpoints] == [1, 2, 3, 4, 5, 6]
    for checkpoint in outcome.checkpoints:
        used_at = checkpoint.step - 1
        expected = 0.01 + (1.0 - 0.01) * min(1.0, used_at / 6)
        assert checkpoint.record["weight.sparsity"] == pytest.approx(expected, abs=1e-6)
        assert checkpoint.record["weight.fit"] == 1.0  # the other term is untouched
    record = outcome.anneals["train.objective.sparsity.weight"]
    assert record["start"] == 0.01 and record["end"] == 1.0
    assert record["shape"] == "linear"
    assert record["final"] == pytest.approx(1.0)  # frac 1.0: `to` at the last step
    assert outcome.controls == {}  # an anneal is not a controller


def test_a_geometric_anneal_walks_the_ratio_not_the_difference():
    """Continuous sparsification's shape: at half the ramp a geometric schedule
    from 0.01 to 1.0 sits at their geometric mean (0.1), where the linear one
    sits at their arithmetic mean (0.505) — one field apart, different sweeps."""
    geometric = {"from": 0.01, "to": 1.0, "frac": 1.0, "shape": "geometric"}
    outcome = _fit_outcome(annealed_weight_dbm_doc(geometric))
    by_step = {c.step: c.record["weight.sparsity"] for c in outcome.checkpoints}
    # update 3 (zero-based) is half of six
    assert by_step[4] == pytest.approx(0.1, abs=1e-6)
    assert by_step[1] == pytest.approx(0.01)
    linear = _fit_outcome(annealed_weight_dbm_doc([0.01, 1.0, 1.0]))
    assert {c.step: c.record["weight.sparsity"] for c in linear.checkpoints}[
        4
    ] == pytest.approx(0.505, abs=1e-6)
    assert outcome.anneals["train.objective.sparsity.weight"]["shape"] == "geometric"


def test_a_weight_anneal_on_a_positional_term_is_refused_at_load():
    """Rule 4: a positional term has no name to address, so the schedule's
    target resolves to nothing — the `control` rule, on the open-loop twin."""
    from causalab.protocol.rules.errors import ValidationError

    doc = annealed_weight_dbm_doc([0.01, 1.0, 1.0])
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.01, {"l1": "gate"}]]
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(doc)), engine_is_local=True)
    assert err.value.rule == 4 and "named objective term" in str(err.value)


# --------------------------------------------------------------------------- #
# §2.11 `phases` — the fit in windows
# --------------------------------------------------------------------------- #


def phased_chain_doc() -> dict:
    """A rotation and a gate over its coordinates (the ``["rot", "gate"]``
    chain) fit in two phases over four updates: the gate alone, then the
    rotation alone at the gate's pinned hard mask with the sparsity weight
    annealed over that phase; every update photographed."""
    doc = chain_doc(0.05)
    doc["method"]["train"]["objective"] = {
        "fit": ce_term(weight=1.0),
        "sparsity": {"weight": 0.01, "l1": "gate"},
    }
    doc["method"]["train"]["phases"] = [
        {"until": {"frac": 0.5}, "params": ["gate"]},
        {
            "until": {"frac": 1.0},
            "params": ["rot"],
            "freeze_masks": ["gate"],
            "optimizer": {"lr": 0.02},
            "anneal": {"train.objective.sparsity.weight": [1.0, 2.0, 1.0]},
        },
    ]
    doc["method"]["save"].append(
        {"kind": "trajectory", "every": {"updates": 1}, "file_path": "t.safetensors"}
    )
    return doc


def test_phases_narrow_what_trains_and_pin_the_named_masks():
    """Phase 0 (updates 0, 1) trains the gate alone: the rotation is
    bit-identical across it. Phase 1 (updates 2, 3) trains the rotation alone
    under the gate's pinned hard mask: θ is bit-identical across it and the
    rotation moves; the phase's own anneal spans the phase's two updates; the
    checkpoints and the diagnostics record which window each update ran in;
    the pin is gone from the stage the loop returns."""
    validate_document(
        parse_document(in_order(phased_chain_doc())), engine_is_local=True
    )
    outcome = _fit_outcome(phased_chain_doc())
    assert [c.step for c in outcome.checkpoints] == [1, 2, 3, 4]
    assert [c.record["phase"] for c in outcome.checkpoints] == [0, 0, 1, 1]
    rot = [c.slots["rot"] for c in outcome.checkpoints]
    theta = [c.slots["gate"]["theta"] for c in outcome.checkpoints]

    def same(a, b):
        return all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()

    # phase 0: the rotation never moved, the gate did
    assert same(rot[0], rot[1])
    assert not torch.equal(theta[0], theta[1])
    # phase 1: the gate never moved (its θ takes no gradient), the rotation did
    assert torch.equal(theta[1], theta[2]) and torch.equal(theta[2], theta[3])
    assert not same(rot[1], rot[2]) and not same(rot[2], rot[3])
    # the phase's anneal ran over the phase's own two updates: 1.0, then 1.5
    weights = [c.record["weight.sparsity"] for c in outcome.checkpoints]
    assert weights[:2] == [0.01, 0.01]  # the authored weight, phase 0
    assert weights[2] == pytest.approx(1.0) and weights[3] == pytest.approx(1.5)
    assert outcome.phases == (
        {"start": 0, "end": 2, "params": ["gate"], "freeze_masks": []},
        {"start": 2, "end": 4, "params": ["rot"], "freeze_masks": ["gate"]},
    )
    assert outcome.stages["gate"].frozen_mask is None  # a phase's, not the bundle's
    assert outcome.stages["gate"].theta.requires_grad is False  # eval-mode stages
    # both featurizers are the fit's deliverables whatever phase trained them
    assert set(outcome.stages) == {"rot", "gate"}


def test_a_pinned_mask_is_the_hard_split_for_every_forward():
    """`freeze_masks` in the stage: while `frozen_mask` is set, a training-mode
    forward uses it — not the soft σ(θ/T) a sigmoid gate would relax to — and
    clearing it restores the map."""
    from causalab.neural.shared.featurizers import Gate

    gate = Gate(4, init=torch.tensor([2.0, -2.0, 0.5, -0.5]))
    gate.train(True)
    soft = gate._mask()
    assert not torch.equal(soft, soft.round())  # a relaxed mask
    gate.frozen_mask = gate.hard_mask().clone()
    kept, dropped = gate.featurize(torch.ones(4))
    assert torch.equal(kept, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(dropped, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    gate.frozen_mask = None
    assert torch.equal(gate._mask(), soft)


def test_phases_that_do_not_partition_the_run_are_refused_before_a_step():
    """An `updates` partition is checked against the run's update count by
    the loop (the parser cannot know it): a last phase short of the run, or
    past it, is a refusal, not a silently unowned tail."""
    from causalab.protocol.rules.errors import ProtocolError

    doc = phased_chain_doc()
    doc["method"]["train"]["phases"] = [
        {"until": {"updates": 2}, "params": ["gate"]},
        {"until": {"updates": 3}, "params": ["rot"]},
    ]
    with pytest.raises(ProtocolError, match="partition the run"):
        _fit_outcome(doc)
    doc["method"]["train"]["phases"][1]["until"] = {"updates": 9}
    with pytest.raises(ProtocolError, match="past the run"):
        _fit_outcome(doc)


def test_a_position_gate_fits_over_the_window_and_counts_positions():
    """§2.5 ``axis: position``: a DBM fit whose write goes through a position
    gate over the first three tokens — θ has three entries, one update runs,
    the diagnostics count positions."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ARTIFACT_IDENTITY_KEYS, ResolutionEnv

    doc = dbm_doc()
    doc["method"]["featurizers"] = {"pg": {"kind": "gate", "axis": "position"}}
    doc["method"]["reads"]["v_cf"]["pos"] = {"span": [0, 3]}
    # the read goes through the gate too (the swap source, as §2.5 has it),
    # so the write is the convex blend `m·x_cf + (1 − m)·x_base`
    doc["method"]["reads"]["v_cf"]["featurizer"] = "pg"
    doc["method"]["writes"]["patch"]["pos"] = {"span": [0, 3]}
    doc["method"]["writes"]["patch"]["featurizer"] = "pg"
    doc["method"]["train"]["params"] = ["pg"]
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.01, {"l1": "pg"}]]
    doc["method"]["train"]["anneal"] = {"pg.theta.temperature": [1.0, 0.01, 0.5]}
    doc["method"]["train"]["steps"] = {"updates": 2}
    doc["method"]["save"] = [
        ce_save(),
        {"value": "pg", "site": "tgt", "file_path": "pg.safetensors"},
    ]
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    gate = executor.stage("pg")
    assert gate.axis == "position" and gate.theta.numel() == 3
    # the stamp a save writes (`identity_fields`) is admitted by the identity
    # schema — the save that would otherwise die at its first bundle
    assert set(gate.identity_fields()) <= set(ARTIFACT_IDENTITY_KEYS)
    assert gate.identity_fields()["axis"] == "position"
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    theta = outcome.stages["pg"].theta.detach()
    assert theta.shape == (3,) and not torch.allclose(theta, torch.zeros(3))
    assert 0 <= outcome.diagnostics["pg"]["hard_mask_size"] <= 3
    # the record says what its counts are over
    assert outcome.diagnostics["pg"]["axis"] == "position"
    # ...and so does every `rank.json` row: its `unit` is a position
    from causalab.neural.shared.execution import rank_records

    rows = rank_records(outcome.stages, "0" * 64, {})
    assert [r["unit"] for r in rows] == [0, 1, 2]
    assert all(r["axis"] == "position" for r in rows)
    assert outcome.diagnostics["pg"]["width"] == 3.0
    # the fitted gate's θ → mask map is over positions (forced after the rank
    # rows above, which read θ live). This is the unit twin's arithmetic on
    # the gate the fit built; comparing the *written value* position by
    # position needs a read at the written site on the patched model, which
    # this document has none of — a follow-up
    with torch.no_grad():
        outcome.stages["pg"].theta.copy_(torch.tensor([5.0, -5.0, 5.0]))
    assert outcome.stages["pg"].hard_mask().tolist() == [1.0, 0.0, 1.0]


def test_a_position_gate_chained_before_a_feature_gate_sizes_each_by_its_own_axis():
    """The headline shape, `["pg", "gate"]`, through a document: the
    position gate is sized by the window (3) while the feature width keeps
    flowing to the gate after it (the site's hidden width), and both train.
    (That the stack's mask is the outer product is the unit test's claim,
    `test_a_position_gate_before_a_feature_gate_is_the_outer_product`; this
    one pins the sizing and the fit.)"""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = dbm_doc()
    doc["method"]["featurizers"] = {
        "pg": {"kind": "gate", "axis": "position"},
        "gate": {"kind": "gate"},
    }
    doc["method"]["reads"]["v_cf"]["pos"] = {"span": [0, 3]}
    doc["method"]["reads"]["v_cf"]["featurizer"] = ["pg", "gate"]
    doc["method"]["writes"]["patch"]["pos"] = {"span": [0, 3]}
    doc["method"]["writes"]["patch"]["featurizer"] = ["pg", "gate"]
    doc["method"]["train"]["params"] = ["pg", "gate"]
    doc["method"]["train"]["objective"] = [
        [1.0, ce_term()],
        [0.01, {"l1": ["pg", "gate"]}],
    ]
    doc["method"]["train"].pop("anneal", None)
    doc["method"]["train"]["steps"] = {"updates": 2}
    doc["method"]["save"] = [
        ce_save(),
        {"value": "pg", "site": "tgt", "file_path": "pg.safetensors"},
        {"value": "gate", "site": "tgt", "file_path": "gate.safetensors"},
    ]
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    assert executor.stage("pg").theta.numel() == 3
    assert executor.stage("gate").theta.numel() == bundle.info.hidden_size
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    assert outcome.stages["pg"].theta.shape == (3,)
    assert outcome.stages["gate"].theta.shape == (bundle.info.hidden_size,)
    assert not torch.allclose(outcome.stages["gate"].theta.detach(), torch.zeros(1))


def test_a_straight_through_gate_fits_and_stamps_its_forward():
    """§2.5 the mapping form: a DBM fit whose gate runs a hard forward with
    the sigmoid's gradient trains (θ moves), keeps the map's eval split, and
    records the split as provenance."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = dbm_doc()
    doc["method"]["featurizers"]["gate"]["parametrization"] = {
        "forward": "hard",
        "backward": "sigmoid",
    }
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    gate = executor.stage("gate")
    assert gate.forward_mask == "hard" and gate.parametrization == "sigmoid"
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    fitted = outcome.stages["gate"]
    assert not torch.allclose(fitted.theta, torch.zeros_like(fitted.theta))
    assert fitted.identity_fields()["forward"] == "hard"
    assert outcome.diagnostics["gate"]["parametrization"] == "sigmoid"
    assert outcome.diagnostics["gate"]["forward"] == "hard"  # the regime, recorded
    # the stamp path: `identity_fields()` is splatted into the closed
    # artifact-identity schema at the save — a key it does not know kills
    # the fit after it has run, which is where `forward` (and `axis`) died
    from causalab.io.env import build_artifact_identity

    stamped = build_artifact_identity(**fitted.identity_fields())
    assert stamped["forward"] == "hard"


def boundless_das_doc(*, seed: int = 0, epochs: int = 3) -> dict:
    """Boundless DAS (§2.5 `boundary`): `das_doc`'s rotation with a boundary
    gate behind it, both trained, `l1` on the gate weighted so β visibly moves
    in a few steps, the temperature annealed."""
    doc = das_doc(seed=seed, epochs=epochs)
    doc["method"]["featurizers"]["bnd"] = {
        "kind": "gate",
        "parametrization": "boundary",
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = ["rot", "bnd"]
    doc["method"]["writes"]["patch"]["featurizer"] = ["rot", "bnd"]
    doc["method"]["train"]["params"] = ["rot", "bnd"]
    doc["method"]["train"]["objective"] = [[1.0, ce_term()], [0.5, {"l1": "bnd"}]]
    doc["method"]["train"]["anneal"] = {"bnd.theta.temperature": [1.0, 0.1, 0.5]}
    doc["method"]["save"].append(
        {"value": "bnd", "site": "tgt", "file_path": "bnd.safetensors"}
    )
    return doc


def test_a_boundless_das_fit_moves_its_boundary_and_records_the_learned_rank():
    """§2.5 ``boundary``: one β behind the rotation, trained under ``ce`` and
    ``l1``; the record carries β itself and ``hard_mask_size`` is the prefix
    it names, ``⌈β⌉``; nothing per unit is reported."""
    from causalab.neural.engines.pytorch_hooks.loading import load_model
    from causalab.neural.engines.pytorch_hooks.train import run_training
    from causalab.io.env import ResolutionEnv

    doc = boundless_das_doc()
    validate_document(parse_document(in_order(doc)), engine_is_local=True)
    bundle = load_model(TINY_LLAMA)
    executor = executor_for(
        doc,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )
    request = RunContext(
        env=ResolutionEnv(datasets=_NoDatasets(), artifacts=None),  # type: ignore[arg-type]
        output_dir=None,  # type: ignore[arg-type]
    )
    outcome = run_training(executor.doc, executor, request)
    bnd = outcome.stages["bnd"]
    assert bnd.parametrization == "boundary" and tuple(bnd.theta.shape) == (1,)
    theta = float(bnd.theta)
    beta = bnd.boundary()
    assert 0.0 <= theta <= 1.0 and theta != 0.5  # projected, and moved off the start
    assert beta == pytest.approx(theta * 4.0)
    assert bnd.temperature < 1.0  # the anneal ran
    assert not bnd.training
    report = outcome.diagnostics["bnd"]
    assert report["parametrization"] == "boundary" and report["boundary"] == beta
    assert report["width"] == 4.0
    assert report["hard_mask_size"] == float(sum(i < beta for i in range(4)))
    assert report["hard_mask_size"] == float(bnd.hard_mask().sum())
    assert "reawakened_units" not in report and "groups" not in report
    assert "rot" in outcome.stages  # the rotation trained beside it
