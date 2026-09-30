"""Legality is decided before weights load (spec §5's invariant).

**A configuration accepted by preflight must either execute or fail with a
narrower runtime condition that preflight could not know.** Six refusals used
to break that: two fired at load already (an illegal write mechanism, rule 4;
``kl`` operands compared by component *label* only), four fired inside the
reference engine's train loop or metric reduction — after ``load_model`` had
pulled the weights — and one (an authored training precision the loop does
not honour) fired nowhere. This file is the acceptance suite for moving every
document-decidable half into the compiler:

* one refusal test per item (a)–(f), asserting the rule by id and the path
  on the offending field;
* (g), a metric's answers the tokenizer cannot score, refused at the run
  door with ``[P2]`` (``pipeline.resolve_answers``), and an unloadable
  tokenizer refused there with ``[P4]``;
* the **never-called-loader** test: the reference engine's ``load_model`` is
  replaced by one that raises, and every refusing document goes through
  [`run_protocol`][causalab.protocol.pipeline.run_protocol] — the refusal comes first, and
  the loader is never entered; then the same through a stub engine whose
  ``execute`` raises — the refusal precedes routing's hand-off;
* the **valid-work twin** of every refusal: the fixed
  document compiles, and for the three engine-decided rules it also *routes*
  when a stub engine declares the verb — the rule refuses the engine, not the
  document.

Every test here is torch-free but the loader tests: the never-called-loader
test and the (g) tests import the reference engine to monkeypatch it and
never let it reach ``torch``'s ``from_pretrained``, and the (g) tests load
the gpt2 tokenizer. Each docstring says how the test fails on the base; the
compiler is imported as a module so that on a tree without ``check_engine``
the refusal tests still collect and show *their* red, not an import error.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol import pipeline as compiler
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import Engine, RunContext, RunResult, requires
from causalab.protocol.rules.errors import (
    ProtocolError,
    ValidationError,
    ValidationErrors,
)
from causalab.io.env import FileDatasets, ResolutionEnv
from causalab.protocol.pipeline import run_protocol
from causalab.protocol.schema import COMPONENTS, parse_document

from tests._helpers.refusal_snapshot import MOE
from tests.protocol._docs import UNWRITTEN, aggregation, base_doc, in_order, saved, term


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# the documents: one refusing document per item, and its valid-work twin
# --------------------------------------------------------------------------- #


#: The fit's one aggregation: cross-entropy of the patched logits on ``label``.
CE = aggregation("cross_entropy", target="label")


def _train_doc() -> dict[str, Any]:
    """A DAS fit on gpt2: a ``subspace`` featurizer on the patched site,
    trained against a cross-entropy over the patched logits."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc["method"]["train"] = {
        "objective": [[1.0, term("logits", "patched", dict(CE))]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    doc["method"]["save"].append(saved("logits", "patched", "ce.json", dict(CE)))
    doc["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return doc


def illegal_write_doc() -> dict[str, Any]:
    """(a) — a write to ``router_logits``, which no mechanism may change."""
    doc = base_doc()
    doc["model"]["key"] = MOE.key
    doc["method"]["sites"]["tgt"] = {"component": "router_logits", "layers": [3]}
    return doc


def _kl_doc() -> dict[str, Any]:
    """Two ``block_output`` reads at one site compared by ``kl``: the twin."""
    doc = base_doc()
    doc["method"]["reads"]["a"] = {"site": "tgt", "pos": -1}
    doc["method"]["reads"]["b"] = {"site": "tgt", "pos": -1}
    doc["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["a"],
    }
    doc["method"]["intervened_models"]["patched"]["reads"].append("b")
    doc["method"]["save"].append(
        saved(
            "a",
            "original_base",
            "d.json",
            aggregation("kl", target={"read": "b", "model": "patched"}),
        )
    )
    return doc


def kl_transform_doc() -> dict[str, Any]:
    """(b) — the same component label on both sides, but one read is
    re-expressed through a ``k=8`` subspace: two distributions over two
    axes."""
    doc = _kl_doc()
    doc["method"]["featurizers"] = {
        "sub": {"kind": "subspace", "k": 8, "parametrization": "cayley"}
    }
    doc["method"]["reads"]["b"]["featurizer"] = "sub"
    return doc


def kl_width_doc() -> dict[str, Any]:
    """(b) — one label, two widths: the whole per-head ``attention_result``
    against one head's slice of it, with no transform on either side."""
    doc = _kl_doc()
    doc["method"]["sites"]["heads"] = {"component": "attention_result", "layers": [3]}
    doc["method"]["sites"]["head0"] = {
        "component": "attention_result",
        "layers": [3],
        "head": 0,
    }
    doc["method"]["reads"]["a"]["site"] = "heads"
    doc["method"]["reads"]["b"]["site"] = "head0"
    return doc


def kl_frame_doc() -> dict[str, Any]:
    """(b) — a prompt-frame read against a continuation-step read."""
    doc = _kl_doc()
    doc["method"]["reads"]["a"] = {
        "site": "lm_head",
        "pos": {"generated": {"max_new_tokens": 4}, "index": 0},
    }
    doc["method"]["reads"]["b"] = {"site": "lm_head", "pos": -1}
    return doc


def free_params_doc() -> dict[str, Any]:
    """(c) — a fit that trains a free ``params`` tensor beside the subspace,
    steering the patched site with it."""
    doc = _train_doc()
    doc["method"]["params"] = {"w": {"shape": [768], "init": "zeros"}}
    doc["method"]["writes"]["steer"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "w", "alpha": 1.0}},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("steer")
    doc["method"]["train"]["params"] = ["rot", "w"]
    return doc


def precision_doc() -> dict[str, Any]:
    """(d) — a fit whose loss is authored in bf16."""
    doc = _train_doc()
    doc["method"]["train"]["precision"] = {"feature": "fp32", "loss": "bf16"}
    return doc


def eval_updates_doc() -> dict[str, Any]:
    """(f) — a fit whose eval is counted in updates."""
    doc = _train_doc()
    doc["method"]["train"]["eval"] = {
        "every": {"updates": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": term("logits", "patched", dict(CE))},
    }
    return doc


def span_metric_doc() -> dict[str, Any]:
    """(e) — a logit difference over a two-token prompt span."""
    doc = base_doc()
    doc["method"]["reads"]["logits"]["pos"] = {"span": [0, 2]}
    return doc


def all_metric_doc() -> dict[str, Any]:
    """(e) — a logit difference over every content token of the row."""
    doc = base_doc()
    doc["method"]["reads"]["logits"]["pos"] = "all"
    return doc


#: Every refusing document with the rule the compile refuses it under.
REFUSALS: dict[str, tuple[Any, int]] = {
    "a_illegal_write": (illegal_write_doc, 4),
    "b_kl_transform": (kl_transform_doc, 29),
    "b_kl_width": (kl_width_doc, 29),
    "b_kl_frame": (kl_frame_doc, 29),
    "c_free_params": (free_params_doc, 30),
    "d_precision": (precision_doc, 30),
    "e_span_metric": (span_metric_doc, 31),
    "e_all_metric": (all_metric_doc, 31),
    "f_eval_updates": (eval_updates_doc, 30),
}
#: The three whose refusal is the *engine's* (rule 30): the verb each needs.
ENGINE_DECIDED = {
    "c_free_params": "train_free_params",
    "d_precision": "train_loss_precision",
    "f_eval_updates": "train_eval_updates",
}


class _Stub(Engine):
    """A stub covering the whole component vocabulary and every non-training
    verb, so the tests below vary only the three training verbs. Its
    ``execute`` raises: reaching it is the failure these tests exist to
    catch."""

    def __init__(self, name: str, extra: frozenset[str] = frozenset()) -> None:
        self.name = name
        self.capabilities = (
            frozenset(
                {
                    "grad",
                    "paired_forward",
                    "full_logits",
                    "pytorch_fn_local",
                    "generate",
                }
            )
            | extra
        )
        self.components = frozenset(COMPONENTS)
        self.writable_components = frozenset(COMPONENTS)
        self.is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        raise AssertionError(
            f"{self.name} executed a document the compile should refuse"
        )


def _compile(raw: dict[str, Any], env: ResolutionEnv, caps: Any = None) -> Any:
    return compile_protocol(
        in_order(raw), env=env, base_dir=None, overrides=None, engine=caps
    )


def _refusal(
    raw: dict[str, Any], env: ResolutionEnv, caps: Any = None
) -> ValidationError:
    with pytest.raises(ValidationError) as err:
        _compile(raw, env, caps)
    assert not isinstance(err.value, ValidationErrors), str(err.value)
    return err.value


# --------------------------------------------------------------------------- #
# (a) — illegal writes refuse at load (rule 4's capability half)
# --------------------------------------------------------------------------- #


def test_a_an_illegal_write_mechanism_refuses_at_the_plan(env: ResolutionEnv) -> None:
    """Rule 4's capability half, from the registry row alone. Already at
    load on the base; pinned here as the (a) clause of the six."""
    err = _refusal(illegal_write_doc(), env)
    assert err.rule == 4 and err.rule_id == "references_resolve"
    assert err.path == "writes.patch.do" and err.reason == "unsupported_mechanism"
    assert "no write may change" in str(err)


# --------------------------------------------------------------------------- #
# (b) — kl operands: width, transform, frame — not labels
# --------------------------------------------------------------------------- #


def test_b_kl_through_different_transforms_is_refused(env: ResolutionEnv) -> None:
    """Without rule 29 the label check passes (both reads tap
    ``block_output``) and the run raises a shape error after the weights
    load; here the compile refuses at the target."""
    err = _refusal(kl_transform_doc(), env)
    assert err.rule == 29 and err.rule_id == "kl_operands_compatible"
    assert err.path == "save[1].aggregation.target"
    assert "different transforms" in str(err) and "['sub']" in str(err)


def test_b_kl_over_different_effective_widths_is_refused(env: ResolutionEnv) -> None:
    """One label, two widths — the whole ``attention_result`` (9216 on gpt2)
    against one head of it (768) — decided from the registry's static
    config, never from a tensor. The base accepts this document."""
    err = _refusal(kl_width_doc(), env)
    assert err.rule == 29
    assert err.path == "save[1].aggregation.target"
    assert "9216 wide" in str(err) and "768 wide" in str(err)


def test_b_kl_across_the_prompt_and_continuation_frames_is_refused(
    env: ResolutionEnv,
) -> None:
    """A prompt position and a decode step are different frames (§2.3):
    the base runs the pair and fails inside the metric reduction."""
    err = _refusal(kl_frame_doc(), env)
    assert err.rule == 29
    assert err.path == "save[1].aggregation.target"
    assert "generated frame" in str(err) and "prompt frame" in str(err)


def test_b_the_valid_work_twins_compile(env: ResolutionEnv) -> None:
    """Same width, same transform, same frame: two plain reads at one site;
    two reads through the *same* subspace; two continuation reads."""
    _compile(_kl_doc(), env)
    same = kl_transform_doc()
    same["method"]["reads"]["a"]["featurizer"] = "sub"
    _compile(same, env)
    both = kl_frame_doc()
    both["method"]["reads"]["b"]["pos"] = {
        "generated": {"max_new_tokens": 4},
        "index": 0,
    }
    _compile(both, env)


# --------------------------------------------------------------------------- #
# (c), (d), (f) — the fit as authored, against the routed engine (rule 30)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(ENGINE_DECIDED))
def test_cdf_a_training_fact_the_engine_cannot_honour_is_refused_at_load(
    name: str, env: ResolutionEnv
) -> None:
    """With the engine's capabilities in hand the compile refuses under rule
    30, naming the field and the missing verb. On the base (c) raised a
    ``NotImplementedError`` and (f) a ``ProtocolError`` inside the train loop,
    after ``load_model``; (d) was accepted, digested as bf16 and run at fp32."""
    build, _rule = REFUSALS[name]
    err = _refusal(build(), env, _Stub("no-verbs").effective_capabilities)
    assert err.rule == 30 and err.rule_id == "train_engine_supported"
    assert ENGINE_DECIDED[name] in str(err)
    assert (
        err.path
        == {
            "c_free_params": "train.params[1]",
            "d_precision": "train.precision.loss",
            "f_eval_updates": "train.eval.every",
        }[name]
    )


@pytest.mark.parametrize("name", sorted(ENGINE_DECIDED))
def test_cdf_the_rule_refuses_the_engine_not_the_document(
    name: str, env: ResolutionEnv
) -> None:
    """The valid-work twin: the same document compiles with no engine given,
    compiles against a stub declaring the verb, and passes ``check_engine``
    against it — so the rule is about the engine's loop, and
    a future engine that trains free params, honours a bf16 loss or counts
    updates runs the document unchanged."""
    build, _rule = REFUSALS[name]
    verb = ENGINE_DECIDED[name]
    compiled = _compile(build(), env)
    assert verb in compiled.capabilities  # routed on, so a covering engine wins
    able = _Stub("able", frozenset({verb}))
    _compile(build(), env, able.effective_capabilities)
    compiler.check_engine(compiled, able.effective_capabilities)
    with pytest.raises(ValidationError) as err:
        compiler.check_engine(compiled, _Stub("plain").effective_capabilities)
    assert err.value.rule == 30


def test_cdf_the_fit_without_the_authored_fact_compiles_and_routes(
    env: ResolutionEnv,
) -> None:
    """The plain fit — featurizer slots, fp32, an epoch-counted eval — needs
    none of the three verbs and routes to an engine that has none."""
    doc = _train_doc()
    doc["method"]["train"]["eval"] = {
        "every": {"epochs": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": term("logits", "patched", dict(CE))},
    }
    doc["method"]["train"]["precision"] = {"feature": "fp32", "loss": "fp32"}
    compiled = _compile(doc, env, _Stub("plain").effective_capabilities)
    assert not compiled.capabilities & set(ENGINE_DECIDED.values())
    compiler.check_engine(compiled, _Stub("plain").effective_capabilities)


# --------------------------------------------------------------------------- #
# (e) — a metric reduces one position per example (rule 31)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["e_span_metric", "e_all_metric"])
def test_e_a_metric_over_a_fixed_multi_position_read_is_refused(
    name: str, env: ResolutionEnv
) -> None:
    """The base accepts both documents and the engine refuses them in
    ``_last_pos_rows`` ("metrics reduce one position per example") after the
    weights load; the document alone fixes the width, so the compile refuses."""
    build, _rule = REFUSALS[name]
    err = _refusal(build(), env)
    assert err.rule == 31 and err.rule_id == "metric_position_scalar"
    assert err.path == "save[0].read"
    assert "one position per example" in str(err)


def test_e_the_valid_work_twins_compile(env: ResolutionEnv) -> None:
    """One token per row compiles; so does a continuation read, whose metric
    reduces per step (``compute_windowed_metric``) — the exemption is by
    design, not by omission."""
    _compile(base_doc(), env)
    stepwise = base_doc()
    stepwise["method"]["reads"]["logits"]["pos"] = {
        "generated": {"max_new_tokens": 4},
        "all": True,
    }
    stepwise["method"]["save"][0]["aggregation"] = aggregation("top_k", k=3, by="prob")
    _compile(stepwise, env)
    one = span_metric_doc()
    one["method"]["reads"]["logits"]["pos"] = {"span": [1, 2]}
    _compile(one, env)


# --------------------------------------------------------------------------- #
# the never-called-loader test, and refusal before routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_the_model_loader_is_never_called_for_a_refused_document(
    name: str, env: ResolutionEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every refusing document through ``run_protocol`` with the reference
    engine: the ``ValidationError`` comes first, and ``load_model`` — the
    name the engine module bound at import, the one call that reaches
    ``from_pretrained`` — is never entered. On the base (c), (d), (e) and (f)
    load the weights: (c), (e), (f) then refuse inside the engine and (d)
    runs. The reference engine is imported here, not at module scope: this is
    a ``unit`` file, and the import is the only torch it touches."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    build, rule = REFUSALS[name]
    with pytest.raises(ValidationError) as err:
        run_protocol(
            in_order(build()),
            env,
            hooks_engine.PytorchHooksEngine(),
            tmp_path,
            record=True,
        )
    assert err.value.rule == rule, str(err.value)
    assert not (tmp_path / "protocol.json").exists(), "a receipt was written"


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_the_refusal_precedes_the_hand_off_to_any_engine(
    name: str, env: ResolutionEnv, tmp_path: Path
) -> None:
    """The same six through a stub whose ``execute`` raises: the compile (or
    ``route_engine``) refuses before any engine is handed the request, so the
    stub's assertion is never reached."""
    build, rule = REFUSALS[name]
    with pytest.raises(ValidationError) as err:
        run_protocol(in_order(build()), env, _Stub("stub"), tmp_path)
    assert err.value.rule == rule, str(err.value)


def test_a_routing_shortfall_names_the_field_when_the_document_decides_it(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The generated rule-13 refusal names the missing verbs. When the chosen
    engine falls short only on a training fact the document authored,
    ``route_engine`` names the field under rule 30 instead of the verb list."""
    with pytest.raises(ValidationError) as err:
        run_protocol(in_order(eval_updates_doc()), env, _Stub("x"), tmp_path)
    assert err.value.rule == 30 and err.value.path == "train.eval.every"


# --------------------------------------------------------------------------- #
# the compiler seam: check_engine, and rule 13's question on its own
# --------------------------------------------------------------------------- #


def test_check_engine_refuses_a_capability_shortfall_as_rule_13(
    env: ResolutionEnv,
) -> None:
    """The report the base made of a shortfall is a refusal now, with the
    routing text: a bare engine cannot run a paired interchange."""
    compiled = _compile(base_doc(), env)
    with pytest.raises(ValidationError) as err:
        compiler.check_engine(compiled, frozenset())
    assert err.value.rule == 13 and "paired_forward" in str(err.value)
    compiler.check_engine(compiled, compiled.capabilities)  # the twin: exactly enough


def test_rule_13s_local_question_is_asked_of_the_documents_own_requirement(
    env: ResolutionEnv,
) -> None:
    """Rule 13's question — may a local engine run this, may a non-local one —
    is put to ``check_engine`` as an engine offering exactly what the
    document requires, with ``pytorch_fn_local`` added or removed; never as an
    engine offering ``pytorch_fn_local`` and nothing else, whose shortfall
    would refuse every paired document. So a plain interchange passes both
    ways, and a ``pytorch_fn`` write passes locally and refuses under rule 13
    — and no other rule — off a local engine."""
    local_engine = frozenset({"pytorch_fn_local"})
    plain = compile_protocol(base_doc(), env=env)
    compiler.check_engine(plain, plain.capabilities | local_engine)
    compiler.check_engine(plain, plain.capabilities - local_engine)
    local = base_doc()
    local["method"]["code"] = {
        "relu": {"locator": "tests.protocol._code_under_test.scale"}
    }
    local["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "relu"}}
    del local["method"]["reads"]["v_cf"]
    del local["method"]["intervened_models"][UNWRITTEN]
    del local["data"]["counterfactual"]
    fn = compile_protocol(in_order(local), env=env)
    compiler.check_engine(fn, fn.capabilities | local_engine)
    with pytest.raises(ValidationError) as err:
        compiler.check_engine(fn, fn.capabilities - local_engine)
    assert err.value.rule == 13


def test_requires_charges_the_training_verbs(env: ResolutionEnv) -> None:
    """The one derivation routing and rule 30 share: each authored training
    fact is one verb in ``requires``, and the plain fit charges none."""
    plain = requires(parse_document(in_order(_train_doc())))
    assert not plain & set(ENGINE_DECIDED.values())
    for name, verb in ENGINE_DECIDED.items():
        build, _rule = REFUSALS[name]
        assert verb in requires(parse_document(in_order(build()))), name


# --------------------------------------------------------------------------- #
# (g) a metric's answers resolve against the tokenizer before the weights load
# --------------------------------------------------------------------------- #

#: Two IOI rows whose answer columns are bare names. Under the gpt2 tokenizer
#: ``'Jennifer'`` is one token and ``'Tiffany'`` is three (``T``, ``iff``,
#: ``any``), so ``logit_diff`` over them cannot score the second row.
BARE_NAMES = [
    {
        "input": "Then, Jennifer and Kevin went to the store. Kevin gave a drink to",
        "io": "Jennifer",
        "s": "Kevin",
        "split": "all",
    },
    {
        "input": "Then, Tiffany and Sean went to the store. Sean gave a drink to",
        "io": "Tiffany",
        "s": "Sean",
        "split": "all",
    },
]


def _answers_doc() -> dict[str, Any]:
    """The un-intervened gpt2 on the base rows, its last-position logits
    reduced to ``logit_diff`` of the ``io`` and ``s`` columns."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": "names/data", "field": "input"}},
        "method": {
            "intervened_models": {
                "original_base": {"input": "base", "reads": ["logits"]}
            },
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [
                saved(
                    "logits",
                    "original_base",
                    "ld.json",
                    aggregation("logit_diff", a="io", b="s"),
                )
            ],
        },
    }


def _names_env(env: ResolutionEnv, root: Path, rows: list[dict[str, Any]]) -> Any:
    """``env`` with its datasets at ``root``, where ``names/data`` is ``rows``."""
    (root / "names").mkdir(parents=True)
    (root / "names" / "data.json").write_text(json.dumps(rows))
    return dataclasses.replace(env, datasets=FileDatasets(root=root))


def test_g_a_multi_token_answer_column_refuses_before_weights(
    env: ResolutionEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A column value the tokenizer splits is refused at the run door, before
    ``load_model``: ``[P2]`` names the aggregation, the field and the value.
    On the base the door resolved positions only, the weights loaded, and
    the refusal came when the first point scored, after the model load."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    names = _names_env(env, tmp_path / "data", BARE_NAMES)
    with pytest.raises(ProtocolError) as err:
        run_protocol(
            in_order(_answers_doc()),
            names,
            hooks_engine.PytorchHooksEngine(),
            tmp_path / "out",
            record=True,
        )
    assert err.value.code == "P2", str(err.value)
    assert "metric logit_diff.a" in str(err.value), str(err.value)
    assert "'Tiffany'" in str(err.value), str(err.value)
    assert not (tmp_path / "out" / "protocol.json").exists(), "a receipt was written"


def test_g_the_valid_work_twin_reaches_the_loader(
    env: ResolutionEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rows with the space the prompt implies (``' Tiffany'`` is one
    gpt2 token) pass the answer check, so the run reaches ``load_model``: the
    check refuses the table, not the document."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded")

    monkeypatch.setattr(hooks_engine, "load_model", never)
    spaced = [{**row, "io": f" {row['io']}", "s": f" {row['s']}"} for row in BARE_NAMES]
    names = _names_env(env, tmp_path / "data", spaced)
    with pytest.raises(AssertionError, match="weights loaded"):
        run_protocol(
            in_order(_answers_doc()),
            names,
            hooks_engine.PytorchHooksEngine(),
            tmp_path / "out",
        )


def test_g_an_unloadable_tokenizer_refuses_the_run_with_p4(
    env: ResolutionEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline with nothing cached, the run door refuses ``[P4]`` naming the
    key and revision, before ``load_model`` and before any receipt, rather
    than raising transformers' ``OSError``."""
    from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("weights loaded")

    def offline(key: str, revision: str) -> Any:
        raise OSError(
            "We couldn't connect to 'https://huggingface.co' to load the files"
        )

    monkeypatch.setattr(hooks_engine, "load_model", never)
    spaced = [{**row, "io": f" {row['io']}", "s": f" {row['s']}"} for row in BARE_NAMES]
    names = dataclasses.replace(
        _names_env(env, tmp_path / "data", spaced), tokenizers=offline
    )
    with pytest.raises(ProtocolError) as err:
        run_protocol(
            in_order(_answers_doc()),
            names,
            hooks_engine.PytorchHooksEngine(),
            tmp_path / "out",
            record=True,
        )
    assert err.value.code == "P4", str(err.value)
    assert "the tokenizer of gpt2@main could not be loaded" in str(err.value)
    assert not (tmp_path / "out" / "protocol.json").exists(), "a receipt was written"


class _Holding(_Stub):
    """A stub holding a caller-owned bundle (spec §9): what runs encodes
    with the bundle's tokenizer, so the run door resolves with it too."""

    def __init__(self, tokenizer: Any) -> None:
        super().__init__("holding")
        self.bundle = type("Bundle", (), {"tokenizer": tokenizer})()


def test_g_a_caller_owned_bundle_resolves_the_answers_with_its_tokenizer(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """The service by model key cannot load, and the bundle's tokenizer
    serves: the answers resolve with it (it encodes ``' Tiffany'``) and the
    run reaches ``execute``."""
    from causalab.io.tokenizer import load_tokenizer

    tokenizer = load_tokenizer("gpt2")
    encoded: list[str] = []
    encode = tokenizer.encode

    def recording(text: str, **kwargs: Any) -> Any:
        encoded.append(text)
        return encode(text, **kwargs)

    tokenizer.encode = recording

    def offline(key: str, revision: str) -> Any:
        raise OSError("not cached")

    spaced = [{**row, "io": f" {row['io']}", "s": f" {row['s']}"} for row in BARE_NAMES]
    names = dataclasses.replace(
        _names_env(env, tmp_path / "data", spaced), tokenizers=offline
    )
    with pytest.raises(AssertionError, match="holding executed"):
        run_protocol(
            in_order(_answers_doc()), names, _Holding(tokenizer), tmp_path / "out"
        )
    assert " Tiffany" in encoded, encoded
