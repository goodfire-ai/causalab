"""Which of a fit's forward groups the fit cannot change (spec §3, §4).

A ``train`` document re-runs its minibatch groups every optimizer step, but
only the groups a trained parameter can *reach* actually change between
steps. `fit_constant_models` is the plan-level statement of that: the
models whose activations no trained featurizer, no trained free tensor, and
no operand read through either can influence. Its consumers cache those
groups across steps, epochs, eval passes and points, so an over-inclusive
answer here would serve a stale activation into a gradient step — which is
why every exclusion below is pinned, not only the happy path.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.neural.shared.plan import (
    PAST_BLOCKS,
    fit_constant_models,
    is_unwritten,
    plan_point,
    write_names,
)
from causalab.protocol.schema import parse_document
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import (
    LOGIT_DIFF,
    UNWRITTEN,
    aggregation,
    base_doc,
    in_order,
    saved,
    term,
)


pytestmark = pytest.mark.unit

#: The fit's objective: a cross-entropy over the patched logits.
CE = aggregation("cross_entropy", target="label")


def _constant(raw: dict[str, Any]) -> frozenset[str]:
    doc = parse_document(in_order(raw))
    validate_document(doc, engine_is_local=True)
    return fit_constant_models(doc)


def _train_section(params: list[str]) -> dict[str, Any]:
    return {
        "objective": [[1.0, term("logits", "patched", CE)]],
        "params": params,
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
        "seed": 0,
    }


def das_doc() -> dict[str, Any]:
    """The shipped DAS shape: a swap through a trained rotation on both the
    operand read and the write."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 8, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["train"] = _train_section(["rot"])
    doc["method"]["save"].append(saved("logits", "patched", "ce.json", dict(CE)))
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return doc


def dbm_doc() -> dict[str, Any]:
    doc = das_doc()
    doc["method"]["featurizers"] = {"gate": {"kind": "gate"}}
    doc["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    doc["method"]["writes"]["patch"]["featurizer"] = "gate"
    doc["method"]["train"]["params"] = ["gate"]
    doc["method"]["train"]["objective"] = [
        [1.0, term("logits", "patched", CE)],
        [0.01, {"l1": "gate"}],
    ]
    doc["method"]["train"]["anneal"] = {"gate.theta.temperature": [1.0, 0.01, 0.5]}
    doc["method"]["save"][-1] = {
        "value": "gate",
        "site": "tgt",
        "file_path": "gate.safetensors",
    }
    return doc


def _with_second_model(doc: dict[str, Any], name: str, write: dict[str, Any]) -> None:
    """Add an intervened model ``name`` on ``base`` carrying one write, plus
    the read that makes it a sink (§5 rule 11). Sections share one namespace
    (rule 3), so the write is ``w_<name>``."""
    doc["method"]["writes"][f"w_{name}"] = write
    doc["method"]["intervened_models"][name] = {
        "input": "base",
        "reads": [f"logits_{name}"],
        "writes": [f"w_{name}"],
    }
    doc["method"]["reads"][f"logits_{name}"] = {"site": "lm_head", "pos": -1}
    doc["method"]["save"].append(
        saved(f"logits_{name}", name, f"{name}.json", dict(LOGIT_DIFF))
    )


def test_das_and_dbm_fits_hold_only_the_source_forward_constant() -> None:
    assert _constant(das_doc()) == {"original_counterfactual"}
    assert _constant(dbm_doc()) == {"original_counterfactual"}


def test_an_unfeaturized_write_from_a_constant_read_is_constant() -> None:
    """A second intervened model whose swap goes through no trained
    featurizer and whose operand is read raw off ``original`` cannot move
    during the fit — it is as cacheable as the source forward."""
    doc = das_doc()
    doc["method"]["reads"]["v_cf_raw"] = {"site": "tgt", "pos": -1}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"].append("v_cf_raw")
    _with_second_model(
        doc, "plain", {"site": "tgt", "pos": -1, "do": {"swap": "v_cf_raw"}}
    )
    assert _constant(doc) == {"original_counterfactual", "plain"}


def test_a_literal_write_is_constant() -> None:
    """No operand at all: a scaled self-add with a literal coefficient."""
    doc = das_doc()
    _with_second_model(
        doc,
        "scaled",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": 0.0, "alpha": 0.5}}},
    )
    assert "scaled" in _constant(doc)


def test_a_model_consuming_a_read_on_the_trained_model_is_not_constant() -> None:
    """Reachability is transitive: a write fed by a read *on* ``patched`` moves
    whenever the rotation does, even though the write itself is unfeaturized."""
    doc = das_doc()
    doc["method"]["reads"]["v_patched"] = {"site": "tgt", "pos": -1}
    doc["method"]["intervened_models"]["patched"]["reads"].append("v_patched")
    _with_second_model(
        doc, "chained", {"site": "tgt", "pos": -1, "do": {"swap": "v_patched"}}
    )
    assert _constant(doc) == {"original_counterfactual"}


def test_a_read_featurized_by_the_trained_featurizer_is_not_constant() -> None:
    """The operand read itself carries the trained featurizer, the write does
    not — the value written still changes with every step."""
    doc = das_doc()
    _with_second_model(
        doc, "featurized_operand", {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}
    )
    assert _constant(doc) == {"original_counterfactual"}


def test_a_params_operand_rooted_in_train_params_is_not_constant() -> None:
    """A trainable free tensor is not a featurizer, so ``_uses_trained_featurizer``
    cannot see it — the write's param operands are checked by root name."""
    doc = das_doc()
    doc["method"]["params"] = {"bias": {"shape": [768], "init": "zeros"}}
    doc["method"]["train"]["params"] = ["rot", "bias"]
    _with_second_model(
        doc,
        "shifted",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": "bias", "alpha": 1.0}}},
    )
    assert _constant(doc) == {"original_counterfactual"}


def test_a_loaded_params_operand_is_constant() -> None:
    """The same write with the tensor loaded from a file instead of trained."""
    doc = das_doc()
    doc["method"]["params"] = {"bias": {"file_path": "p.safetensors"}}
    _with_second_model(
        doc,
        "shifted",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": "bias", "alpha": 1.0}}},
    )
    assert _constant(doc) == {"original_counterfactual", "shifted"}


def _shifted_by_bias(doc: dict[str, Any]) -> None:
    _with_second_model(
        doc,
        "shifted",
        {"site": "tgt", "pos": -1, "do": {"add_scaled": {"op": "bias", "alpha": 1.0}}},
    )


def _group_key(raw: dict[str, Any], model: str) -> str:
    doc = parse_document(in_order(raw))
    validate_document(doc, engine_is_local=True)
    (group,) = [g for g in plan_point(doc).groups if g.model == model]
    return group.key


def test_a_trained_params_operand_enters_the_group_key() -> None:
    """Two points differing only in ``train.seed`` fit different tensors, so
    a group whose write consumes one must never intern across them — the
    rule the trained-featurizer case already had, extended to a trained free
    tensor. The un-intervened group stays shared: the seed reaches nothing
    in it."""

    def doc(seed: int) -> dict[str, Any]:
        raw = das_doc()
        raw["method"]["params"] = {"bias": {"shape": [768], "init": "zeros"}}
        raw["method"]["train"]["params"] = ["rot", "bias"]
        raw["method"]["train"]["seed"] = seed
        _shifted_by_bias(raw)
        return raw

    assert _group_key(doc(0), "shifted") != _group_key(doc(1), "shifted")
    assert _group_key(doc(0), "original_counterfactual") == _group_key(
        doc(1), "original_counterfactual"
    )


def test_a_loaded_params_operand_leaves_the_group_key_seed_free() -> None:
    """The same write with the tensor loaded rather than trained: the fit's
    seed moves nothing it consumes, so the two points share the group."""

    def doc(seed: int) -> dict[str, Any]:
        raw = das_doc()
        raw["method"]["params"] = {"bias": {"file_path": "p.safetensors"}}
        raw["method"]["train"]["seed"] = seed
        _shifted_by_bias(raw)
        return raw

    assert _group_key(doc(0), "shifted") == _group_key(doc(1), "shifted")


def test_unexpanded_writes_are_refused_rather_than_read_as_constant() -> None:
    """A model whose ``writes`` is still a sweep wrapper has writes the
    classifier cannot see. Reading "none visible" as "constant" would be the
    unsafe direction, so a non-point document is refused."""
    raw = das_doc()
    raw["method"]["intervened_models"]["patched"]["writes"] = {"sweep": [["patch"], []]}
    doc = parse_document(in_order(raw))  # unexpanded on purpose: no validate
    with pytest.raises(AssertionError, match="unexpanded"):
        fit_constant_models(doc)


def test_without_a_train_section_every_model_is_constant() -> None:
    """Nothing is being fitted, so nothing moves: the whole graph is constant
    and the answer names every model, ``original`` included."""
    doc = base_doc()
    assert _constant(doc) == {"original_counterfactual", "patched"}


def test_the_planner_imports_without_torch() -> None:
    """``explain`` prints the first step's forward plan through
    ``neural/shared/plan.py``; the pure verbs stay torch-free, so the
    planner must import without torch in a fresh interpreter."""
    import subprocess
    import sys

    probe = "import sys, causalab.neural.shared.plan; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


# --------------------------------------------------------------------------- #
# the un-intervened model as a forward group (§4 "Resume")
# --------------------------------------------------------------------------- #


def _plan(raw: dict[str, Any]):
    doc = parse_document(in_order(raw))
    validate_document(doc, engine_is_local=True)
    return doc, plan_point(doc)


def test_an_unwritten_model_plans_as_the_prefix_of_its_input() -> None:
    """The group of a model that lands no write *is* the un-intervened
    prefix: it never resumes, its write depth is past every block, and its
    key is its own ``base_key`` — so an intervened model on the same input
    can leave the residual entering its first written block behind for it,
    and vice versa."""
    doc, plan = _plan(base_doc())
    by_model = {group.model: group for group in plan.groups}
    original = by_model["original_counterfactual"]
    assert original.unwritten is True
    assert original.write_depth == PAST_BLOCKS
    assert original.resume_at == 0
    assert original.key == original.base_key
    patched = by_model["patched"]
    assert patched.unwritten is False
    assert patched.write_depth < PAST_BLOCKS
    assert patched.key != patched.base_key


def test_every_model_on_one_input_shares_one_base_key() -> None:
    """Two intervened models on ``base`` share the un-intervened prefix of
    ``base`` whatever they write — and it is the key the un-intervened
    model itself has on that input, when one is planned there."""
    raw = base_doc()
    raw["method"]["reads"]["logits_orig"] = {"site": "lm_head", "pos": -1}
    raw["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["logits_orig"],
    }
    raw["method"]["save"].append(
        saved("logits_orig", "original_base", "ld_orig.json", dict(LOGIT_DIFF))
    )
    _with_second_model(raw, "zeroed", {"site": "tgt", "pos": -1, "do": {"swap": 0.0}})
    _doc, plan = _plan(raw)
    on_base = [group for group in plan.groups if group.input == "base"]
    assert {group.model for group in on_base} == {"original_base", "patched", "zeroed"}
    assert len({group.base_key for group in on_base}) == 1
    (original,) = [group for group in on_base if group.model == "original_base"]
    assert original.key == original.base_key
    # the counterfactual prefix is another input, hence another base key
    (cf,) = [group for group in plan.groups if group.input == "counterfactual"]
    assert cf.base_key != original.base_key
    assert len({group.key for group in plan.groups}) == len(plan.groups)


def test_the_unwritten_predicates_read_the_write_list() -> None:
    """``is_unwritten`` and ``write_names`` are the two questions every
    "is this the original?" branch reduces to: the un-intervened model has
    no writes, a declared model has the writes it lists, and a write list
    still under a sweep wrapper is *unknown* — not unwritten, and ``None``."""
    raw = base_doc()
    doc = parse_document(in_order(raw))
    assert is_unwritten(doc, "original_counterfactual")
    assert write_names(doc, "original_counterfactual") == ()
    assert not is_unwritten(doc, "patched")
    assert write_names(doc, "patched") == ("patch",)
    raw["method"]["intervened_models"]["patched"]["writes"] = {"sweep": [["patch"], []]}
    swept = parse_document(in_order(raw))
    assert not is_unwritten(swept, "patched")
    assert write_names(swept, "patched") is None
    assert swept.intervened_models["patched"].writes is not None
