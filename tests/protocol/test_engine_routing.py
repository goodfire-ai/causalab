"""Capability derivation and the engine choice (spec §8).

Capability-based routing *between* engines is retired: ``--engine`` is a
mandatory explicit input, ``auto`` goes through
[`causalab.neural.shared.engine_router`][] (a placeholder that always answers
``pytorch_hooks``), and the capability **check** — rules 13 and 30 and the §8
shortfall against the one named engine — is [`causalab.protocol.pipeline.validate`][]'s. The refusal that used to be routing's is pinned here against the
check, with the same generated text naming the missing entries.
"""

from __future__ import annotations

import pytest

from causalab.neural.shared.engine_router import AUTO, ENGINE_CHOICES, route_name
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.engine import (
    Engine,
    RunContext,
    RunResult,
    component_capability,
    requires,
)
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.pipeline import validate
from causalab.protocol.registry import ENGINES
from causalab.protocol.rules.capability import refuse_shortfall
from causalab.protocol.schema import COMPONENTS, Document, parse_document

from tests.protocol._docs import UNWRITTEN, aggregation, base_doc, in_order, saved, term
from tests.protocol._env import CORPUS_DIR


pytestmark = pytest.mark.unit


class _Stub(Engine):
    """Coarse-capability stub: serves the whole component vocabulary, so the
    capability-check tests below vary only the §8 verbs."""

    def __init__(
        self,
        name: str,
        capabilities: frozenset[str],
        is_local: bool = False,
        components: frozenset[str] = frozenset(COMPONENTS),
        writable_components: frozenset[str] = frozenset(COMPONENTS),
    ):
        self.name = name
        self.capabilities = capabilities
        self.is_local = is_local
        self.components = components
        self.writable_components = writable_components

    def execute(  # pragma: no cover
        self, compiled: CompiledProtocol, run: RunContext
    ) -> RunResult:
        raise NotImplementedError


#: base_doc touches block_output (read + write) and lm_head (read).
BASE_COMPONENTS = frozenset(
    {
        component_capability("block_output"),
        component_capability("block_output", write=True),
        component_capability("lm_head"),
    }
)


def test_requires_paired_forward():
    doc = parse_document(base_doc())
    assert requires(doc) == frozenset({"paired_forward"}) | BASE_COMPONENTS


def test_requires_component_entries_split_read_from_write():
    """Every touched site contributes its component; a written site also
    contributes the :write entry — the honest routing surface once two
    engines with different site vocabularies exist (§8)."""
    doc = parse_document(base_doc())
    needed = requires(doc)
    assert component_capability("block_output", write=True) in needed
    assert component_capability("lm_head") in needed
    assert component_capability("lm_head", write=True) not in needed


def test_requires_no_coarse_verbs_for_same_input_patching():
    raw = base_doc()
    raw["method"]["intervened_models"][UNWRITTEN]["input"] = "base"
    del raw["data"]["counterfactual"]
    assert requires(parse_document(raw)) == BASE_COMPONENTS


def test_requires_full_logits_when_lm_head_read_saved():
    raw = base_doc()
    raw["method"]["save"].append(saved("logits", "patched", "l.safetensors"))
    assert "full_logits" in requires(parse_document(raw))


def test_requires_full_logits_for_top_k_over_an_lm_head_read():
    raw = base_doc()
    raw["method"]["save"].append(
        saved("logits", "patched", "tk.json", aggregation("top_k", k=5, by="prob"))
    )
    assert "full_logits" in requires(parse_document(raw))


def test_top_k_over_a_non_vocabulary_read_needs_no_full_logits():
    """The saving that motivates any-read ``top_k``: ranking a residual stream
    obliges no vocabulary projection anywhere, so it must not route the
    document onto a full-vocab engine."""
    raw = base_doc()
    raw["method"]["save"].append(
        saved("v_cf", UNWRITTEN, "tk.json", aggregation("top_k", k=5, by="abs_value"))
    )
    assert "full_logits" not in requires(parse_document(raw))


def test_top_k_over_a_featurized_lm_head_read_still_needs_full_logits():
    """Capability and axis are two different questions, split on purpose.

    A featurizer takes the read's *value* out of token-id space (so `prob`
    and token decoding are refused / withheld), but serving the read still
    means materializing the whole projection — the featurizer consumes it.
    So the document still routes onto a full-vocab engine."""
    raw = base_doc()
    raw["method"]["featurizers"] = {
        "f": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    raw["method"]["reads"]["flogits"] = {
        "site": "lm_head",
        "pos": -1,
        "featurizer": "f",
    }
    raw["method"]["intervened_models"]["patched"]["reads"].append("flogits")
    raw["method"]["save"].append(
        saved("flogits", "patched", "tk.json", aggregation("top_k", k=2, by="value"))
    )
    assert "full_logits" in requires(parse_document(in_order(raw)))


def test_top_k_over_a_dims_sliced_lm_head_read_needs_no_full_logits():
    """A `dims` slice needs only its named vocabulary rows — the same rule the
    saved-read derivation already applies."""
    raw = base_doc()
    raw["method"]["reads"]["flogits"] = {
        "site": "lm_head",
        "pos": -1,
        "dims": [0, 1, 2],
    }
    raw["method"]["intervened_models"]["patched"]["reads"].append("flogits")
    raw["method"]["save"].append(
        saved("flogits", "patched", "tk.json", aggregation("top_k", k=2, by="value"))
    )
    assert "full_logits" not in requires(parse_document(in_order(raw)))


#: The fit's one aggregation: cross-entropy of the patched logits on ``label``.
CE = aggregation("cross_entropy", target="label")


def _fit(raw: dict) -> dict:
    """base_doc as a DAS fit: a trained subspace on the patched site."""
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    raw["method"]["train"] = {
        "objective": [[1.0, term("logits", "patched", CE)]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
    }
    raw["method"]["save"].append(
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"}
    )
    return raw


TRAIN_VERBS = frozenset(
    {"train_free_params", "train_loss_precision", "train_eval_updates"}
)


def test_a_plain_fit_requires_grad_and_no_training_verb():
    """Featurizer slots, fp32, an epoch-counted eval: `grad` alone (§8)."""
    needed = requires(parse_document(in_order(_fit(base_doc()))))
    assert "grad" in needed and not needed & TRAIN_VERBS


def test_requires_train_free_params():
    """A `train.params` entry naming a `params` entry — a free tensor (§2.6)
    — is a loop the engine must implement, so it is routed on (rule 30
    refuses the engine that lacks it)."""
    raw = _fit(base_doc())
    raw["method"]["params"] = {"w": {"shape": [768], "init": "zeros"}}
    raw["method"]["writes"]["steer"] = {
        "site": "tgt",
        "pos": -1,
        "do": {"add_scaled": {"op": "w", "alpha": 1.0}},
    }
    raw["method"]["intervened_models"]["patched"]["writes"].append("steer")
    raw["method"]["train"]["params"] = ["rot", "w"]
    needed = requires(parse_document(in_order(raw)))
    assert needed & TRAIN_VERBS == {"train_free_params"}


def test_requires_train_loss_precision():
    """`train.precision` authored as anything but fp32 — either field."""
    raw = _fit(base_doc())
    raw["method"]["train"]["precision"] = {"feature": "fp32", "loss": "bf16"}
    assert requires(parse_document(in_order(raw))) & TRAIN_VERBS == {
        "train_loss_precision"
    }
    raw["method"]["train"]["precision"] = {"feature": "fp16", "loss": "fp32"}
    assert requires(parse_document(in_order(raw))) & TRAIN_VERBS == {
        "train_loss_precision"
    }
    raw["method"]["train"]["precision"] = {"feature": "fp32", "loss": "fp32"}
    assert not requires(parse_document(in_order(raw))) & TRAIN_VERBS


def test_requires_train_eval_updates():
    """An `eval` counted in updates rather than epochs (§2.11)."""
    raw = _fit(base_doc())
    raw["method"]["train"]["eval"] = {
        "every": {"updates": 1},
        "split": "weekdays/data#test",
        "aggregations": {"ce": term("logits", "patched", CE)},
    }
    assert requires(parse_document(in_order(raw))) & TRAIN_VERBS == {
        "train_eval_updates"
    }
    raw["method"]["train"]["eval"]["every"] = {"epochs": 1}
    assert not requires(parse_document(in_order(raw))) & TRAIN_VERBS


def test_requires_writable_attention_probs():
    raw = base_doc()
    raw["method"]["sites"]["probs"] = {"component": "attention_probs", "layers": [3]}
    raw["method"]["writes"]["knock"] = {
        "site": "probs",
        "pos": -1,
        "do": {"clamp": {"lo": 0, "hi": 0}},
    }
    raw["method"]["intervened_models"]["patched"]["writes"].append("knock")
    assert "writable_attention_probs" in requires(parse_document(raw))


def _check(doc: Document, engine: Engine) -> None:
    """Rule 13's routing face against one engine — what ``pipeline.validate``
    runs for the engine ``--engine`` named (``refuse_shortfall``)."""
    refuse_shortfall(requires(doc), engine.effective_capabilities)


def test_a_covering_engine_passes_the_check():
    doc = parse_document(base_doc())
    strong = _Stub("hooks", frozenset({"grad", "paired_forward", "full_logits"}))
    _check(doc, strong)


def test_refusal_names_missing_capabilities():
    doc = parse_document(base_doc())
    weak = _Stub("serving", frozenset({"full_logits"}))
    with pytest.raises(ValidationError) as err:
        _check(doc, weak)
    assert err.value.rule == 13 and "paired_forward" in str(err.value)


def test_an_engine_without_generate_refuses_with_the_capability_named():
    """The check is how a decode-less engine declines a continuation
    document, and the refusal names what it lacks rather than failing
    mid-run; the engine that decodes passes."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 8}, "index": -1}
    }
    raw["method"]["reads"]["logits"]["pos"] = "tail"
    doc = parse_document(in_order(raw))
    prefill_only = _Stub("prefill_only", frozenset({"paired_forward", "full_logits"}))
    with pytest.raises(ValidationError) as err:
        _check(doc, prefill_only)
    assert "generate" in str(err.value)
    _check(doc, _Stub("decoder", prefill_only.capabilities | {"generate"}))


def test_writes_during_generation_needs_the_generation_writes_verb():
    """§2.9's flag obliges ``generation_writes`` beside ``generate``: an engine
    that decodes but cannot keep a hook installed across its steps (the
    nnsight trace binds occurrence 0 of every write) is refused by name."""
    raw = base_doc()
    raw["method"]["positions"] = {
        "tail": {"generated": {"max_new_tokens": 8}, "index": -1}
    }
    raw["method"]["reads"]["logits"]["pos"] = "tail"
    raw["method"]["writes"]["patch"]["do"] = {"swap": 0.0}
    del raw["method"]["reads"]["v_cf"]
    raw["method"]["intervened_models"]["patched"]["writes_during_generation"] = True
    doc = parse_document(in_order(raw))
    assert "generation_writes" in requires(doc)
    decoder = _Stub("decoder", frozenset({"full_logits", "generate"}))
    with pytest.raises(ValidationError) as err:
        _check(doc, decoder)
    assert "generation_writes" in str(err.value)
    _check(doc, _Stub("stepper", decoder.capabilities | {"generation_writes"}))


def test_an_engine_without_the_component_refuses_by_name():
    """A document touching a component outside an engine's site vocabulary
    is refused by the generated component entry — this is how a user learns
    to pin the engine that serves an interior component, with no
    hand-written case anywhere."""
    doc = parse_document(base_doc())
    verbs = frozenset({"paired_forward", "full_logits"})
    no_blocks = _Stub(
        "no_blocks",
        verbs,
        components=frozenset({"lm_head"}),
        writable_components=frozenset(),
    )
    with pytest.raises(ValidationError) as err:
        _check(doc, no_blocks)
    message = str(err.value)
    assert component_capability("block_output") in message
    assert component_capability("block_output", write=True) in message
    _check(doc, _Stub("full", verbs))


def test_a_read_only_component_declaration_refuses_the_write():
    """components without writable_components serves reads but refuses a
    write."""
    doc = parse_document(base_doc())
    read_only = _Stub(
        "read_only",
        frozenset({"paired_forward", "full_logits"}),
        writable_components=frozenset(),
    )
    with pytest.raises(ValidationError) as err:
        _check(doc, read_only)
    assert component_capability("block_output", write=True) in str(err.value)


# --------------------------------------------------------------------------- #
# the router: `--engine` is mandatory, `auto` is a placeholder
# --------------------------------------------------------------------------- #


def test_auto_routes_to_the_reference_engine() -> None:
    """The plan's placeholder: `auto` always answers `pytorch_hooks` until a
    real router decides from the document."""
    assert route_name(AUTO) == "pytorch_hooks"


@pytest.mark.parametrize("name", ENGINES)
def test_a_named_engine_routes_to_itself(name: str) -> None:
    assert route_name(name) == name


def test_an_unknown_engine_is_refused_by_argparse(capsys) -> None:
    """The flag's vocabulary is the router's: every registered engine and
    `auto`, nothing else — exit 2 with argparse's own message."""
    from causalab.cli import _build_parser  # pyright: ignore[reportPrivateUsage]

    assert ENGINE_CHOICES == (*ENGINES, AUTO)
    with pytest.raises(SystemExit) as err:
        _build_parser().parse_args(
            ["run", "doc.json", "--out", "o", "--engine", "sglang"]
        )
    assert err.value.code == 2
    assert "invalid choice: 'sglang'" in capsys.readouterr().err


def test_validate_against_nnsight_refuses_a_grad_document_by_rule_13(env) -> None:
    """The refusal that used to be routing's. A fit requires `grad`, which
    the nnsight engine's registered set lacks: `validate(engine="nnsight")`
    refuses under rule 13 with the shortfall text, the reference engine
    passes, and `auto` is not an engine name the pipeline accepts — the
    router resolves it first."""
    compiled = compile_protocol(CORPUS_DIR / "04_das_im.json", env=env)  # a DAS fit
    assert "grad" in compiled.capabilities
    with pytest.raises(ValidationError) as err:
        validate(compiled, "nnsight", env=env)
    assert err.value.rule == 13
    assert "lacks ['grad']" in str(err.value) and "(sec. 8)" in str(err.value)
    validate(compiled, route_name(AUTO), env=env)  # the twin
    with pytest.raises(ValueError, match="routing policy"):
        validate(compiled, AUTO, env=env)


def test_no_engine_at_all_is_the_generated_rule_13_shortfall(env) -> None:
    """`route_engine(compiled, None)` — a workflow run handed no engine — is
    refused under rule 13 before any weights, and the text is the one every
    other shortfall gets: `refuse_shortfall(required, frozenset())`, so it
    names the document's whole requirement set as both required and lacking,
    never a hand-written sentence. `run_protocol` reaches it the same way."""
    from causalab.protocol.pipeline import route_engine, run_protocol

    compiled = compile_protocol(CORPUS_DIR / "04_das_im.json", env=env)
    assert compiled.capabilities
    required = sorted(compiled.capabilities)
    expected = (
        f"the engine does not support this document: it requires {required} "
        f"and lacks {required} (sec. 8)"
    )
    with pytest.raises(ValidationError) as err:
        route_engine(compiled, None)
    assert err.value.rule == 13 and str(err.value).endswith(expected)
    with pytest.raises(ValidationError) as via_run:
        run_protocol(compiled, env, None, None)  # pyright: ignore[reportArgumentType]
    assert str(via_run.value) == str(err.value)


def test_the_flag_is_required_on_the_four_verbs_with_the_routers_choices() -> None:
    """`--engine` is a mandatory explicit input on every verb that reads a
    document for an engine (run, validate, explain, dry-run), with one
    vocabulary — the router's — and no default anywhere: the old
    `DEFAULT_ENGINE` is gone with routing."""
    import causalab.protocol.engine as contract
    from causalab.cli import _build_parser  # pyright: ignore[reportPrivateUsage]

    assert not hasattr(contract, "DEFAULT_ENGINE")
    assert not hasattr(contract, "ENGINE_CHOICES")
    assert not hasattr(contract, "choose_engine")

    subparsers = _build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    engine_options = {
        verb: action
        for verb, parser in subparsers.items()
        for action in parser._actions
        if "--engine" in action.option_strings
    }
    assert set(engine_options) == {"run", "validate", "explain", "dry-run"}, (
        f"--engine is on {sorted(engine_options)}; update this test if a verb "
        "gained or lost it"
    )
    for verb, action in engine_options.items():
        assert action.required, f"{verb} --engine must be required — no default"
        assert tuple(action.choices or ()) == ENGINE_CHOICES, (
            f"{verb} --engine choices drifted from the router's ENGINE_CHOICES"
        )
