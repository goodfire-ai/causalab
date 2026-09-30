"""The pipeline's two verbs: ``inputs`` —``build``→ ``CompiledProtocol`` —``validate``→ the same
object, validated.

``compile_protocol`` is ``build`` + ``validate`` as one call (the earlier
``compile_protocol`` facade is gone), so the acceptance clause is
equality: for every corpus document ``compile_protocol`` and ``build`` + ``validate``
produce one object, field for field — the same code path, not two that agree.
Around it: ``build`` refuses only what cannot be built and never runs a §5
rule (a rule-4 defect *builds* and ``validate`` refuses it, with the text the
compiler gave); ``validate`` decides the engine from a registered name or an
``Engine`` with ``check_engine``'s text, refuses ``"auto"``, and with
``data=True`` runs the ``validate --data`` pass — refusing a column the base
table lacks with that pass's text and passing every corpus document; the
compatibility properties read as the fields they stand for; and a document
with *two* defects — a §5 violation and a refusal a build stage raises — is
refused for the rule, as the compiler refused it, because ``build`` consults
the checklist over the points it has before it lets a stage's refusal out.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.io.env import ResolutionEnv
from causalab.io.sources import load_text
from causalab.protocol import pipeline
from causalab.protocol.pipeline import check_engine, compile_protocol
from causalab.protocol.compiled import CompiledProtocol, Digests
from causalab.protocol.engine import Engine, RunContext, RunResult
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.lowering import point_count
from causalab.protocol.pipeline import STAGES, build, validate
from causalab.protocol.registry.engines import effective_capabilities
from causalab.protocol.rules.capability import requires_campaign
from causalab.protocol.rules.data import check_data_columns
from causalab.protocol.rules.document import validate_document
from causalab.protocol.schema import COMPONENTS, parse_document

from tests.protocol._docs import UNWRITTEN, base_doc, in_order
from tests.protocol._env import CORPUS_DIR, steps_of
from tests.protocol.conftest import CORPUS_FILES
from tests.protocol.test_grouped_gate import apply_doc, gate_doc
from tests.protocol.test_legality_before_weights import (
    _train_doc,  # pyright: ignore[reportPrivateUsage]
)

pytestmark = pytest.mark.unit


def _facade(source: Any, env: ResolutionEnv) -> CompiledProtocol:
    base = source.parent if isinstance(source, Path) else None
    return compile_protocol(source, env=env, base_dir=base, overrides=None, engine=None)


def _two_verbs(source: Any, env: ResolutionEnv) -> CompiledProtocol:
    base = source.parent if isinstance(source, Path) else None
    return validate(build(source, base_dir=base, env=env), env=env, data=False)


def _refusal_text(fn: Any, *args: Any, **kwargs: Any) -> tuple[type, str]:
    with pytest.raises(Exception) as err:
        fn(*args, **kwargs)
    return type(err.value), str(err.value)


# --------------------------------------------------------------------------- #
# one code path: the facade equals build + validate, field for field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CORPUS_FILES)
def test_the_facade_and_the_two_verbs_are_one_object(
    name: str, env: ResolutionEnv
) -> None:
    """Every field — the parsed document, the materialised campaign, the axes,
    the campaign digest, the point digests, the data and artifact identities,
    the diagnostics, the points — and the derived
    capabilities are equal, because the facade *is* the two verbs."""
    path = CORPUS_DIR / name
    facade = _facade(path, env)
    verbs = _two_verbs(path, env)
    assert facade == verbs
    for field in dataclasses.fields(CompiledProtocol):
        assert getattr(facade, field.name) == getattr(verbs, field.name), field.name
    assert facade.capabilities == verbs.capabilities
    assert verbs.capabilities == requires_campaign(steps_of(verbs, env).documents)


def test_validate_returns_the_object_it_was_handed(env: ResolutionEnv) -> None:
    compiled = build(base_doc(), env=env)
    assert validate(compiled, env=env) is compiled
    assert validate(compiled, "pytorch_hooks", env=env, data=True) is compiled


def test_the_compatibility_properties_are_the_fields(env: ResolutionEnv) -> None:
    """The spellings every door reads today read as the object's own fields:
    ``canonical`` is the materialised campaign, ``digests`` is the campaign
    digest as one record; the points are the engine's — the object
    carries the axes and the tree they index into, and the count is decided
    from the axes."""
    compiled = _two_verbs(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env)
    assert compiled.canonical is compiled.explicit
    assert compiled.digests == Digests(document=compiled.campaign_digest)
    assert point_count(compiled.axes) == 64
    assert len(compiled.representatives) == 1 + (2 - 1) + (32 - 1)
    assert len(steps_of(compiled, env).points) == 64
    assert "validate" not in STAGES and "route" not in STAGES


# --------------------------------------------------------------------------- #
# build refuses only what cannot be built — and runs no §5 rule
# --------------------------------------------------------------------------- #


def test_build_refuses_a_source_that_is_not_a_document(
    env: ResolutionEnv, tmp_path: Path
) -> None:
    """Malformed JSON and an unknown key are could-not-build (``ParseError``):
    the reader's and the gate's own refusals, byte for byte — the oracles are
    the stage functions' callees, not the facade, which is the same code
    path as ``build`` and could not catch a drift."""
    broken = tmp_path / "broken.json"
    broken.write_text('{"header": {"protocol_version": "3",')
    reader = _refusal_text(load_text, broken)
    assert reader[0] is ParseError and reader[1].startswith("[P1] not valid JSON")
    assert _refusal_text(build, broken, env=env) == reader
    assert _refusal_text(_facade, broken, env) == reader

    unknown = base_doc()
    unknown["method"]["sights"] = {}
    gate = _refusal_text(parse_document, unknown)
    assert gate[0] is ParseError and "sights" in gate[1]
    assert _refusal_text(build, unknown, env=env) == gate
    assert _refusal_text(_facade, unknown, env) == gate


def test_build_refuses_a_missing_dataset_as_the_facade_does(
    env: ResolutionEnv,
) -> None:
    """A dependency that is not there is a could-not-build: the dataset
    resolver's own ``V4`` refusal, byte for byte, from ``build`` and from the
    facade."""
    missing = base_doc()
    missing["data"]["base"]["dataset"] = "weekdays/no_such_table#train"
    missing["data"]["counterfactual"]["dataset"] = "weekdays/no_such_table#train"
    resolver = _refusal_text(env.datasets.digest, "weekdays/no_such_table")
    assert resolver[0] is ValidationError
    assert resolver[1].startswith(
        "[V4] at data dataset 'weekdays/no_such_table' not found under "
    )
    assert _refusal_text(build, missing, env=env) == resolver
    assert _refusal_text(_facade, missing, env) == resolver


def test_a_rule_defect_builds_and_validate_refuses_it(
    env: ResolutionEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read naming a site the document never declares is a §5 rule-4
    violation and nothing else: ``build`` produces the object — never entering
    the checklist or the ``file_path`` check — and ``validate`` refuses it
    with the text the facade always gave."""
    dangling = base_doc()
    dangling["method"]["reads"]["v_cf"]["site"] = "nope"
    expected = _refusal_text(_facade, dangling, env)

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("build ran a §5 rule")

    monkeypatch.setattr(pipeline, "validate_document", never)
    monkeypatch.setattr(pipeline, "check_loaded_featurizers", never)
    compiled = build(dangling, env=env)
    monkeypatch.undo()
    assert isinstance(compiled, CompiledProtocol)
    assert steps_of(compiled, env).documents[0].reads["v_cf"].site == "nope"
    kind, text = _refusal_text(validate, compiled, env=env)
    assert (kind, text) == expected
    assert kind is ValidationError and "'nope'" in text


def _checklist_refusal(raw: dict[str, Any], env: ResolutionEnv) -> tuple[type, str]:
    """What the §5 checklist says of the one point ``raw`` expands to — the
    compiler's report for a document it refused between ``expand`` and
    ``canonicalize``."""
    return _refusal_text(
        validate_document, parse_document(raw), model_info=env.model_info
    )


def _two_defect_documents(env: ResolutionEnv) -> dict[str, tuple[dict[str, Any], str]]:
    """A rule-4 defect (a site a read or a write names and the document never
    declares) beside a refusal a *build* stage raises, and the text of that
    refusal alone: a dataset ``identify`` cannot find (``V4`` at ``data``), a
    bundle the store cannot find (``V15``), and the canonical form's rule-23
    impurity (a head group on a component without a head axis — the write
    dangles there, because the read is what names the featurizer)."""
    dataset = base_doc()
    dataset["data"]["base"]["dataset"] = "weekdays/no_such_table#train"
    dataset["data"]["counterfactual"]["dataset"] = "weekdays/no_such_table#train"
    bundle = apply_doc()  # loads fit/g.safetensors, which the fixtures lack
    group = gate_doc(group="head", component="block_output")
    alone = {
        "missing dataset": _refusal_text(build, dataset, env=env)[1],
        "missing bundle": _refusal_text(build, bundle, env=env)[1],
        "rule 23": _refusal_text(build, group, env=env)[1],
    }
    assert "no_such_table" in alone["missing dataset"]
    assert "[V15]" in alone["missing bundle"]
    assert "[V23]" in alone["rule 23"]
    dataset["method"]["reads"]["v_cf"]["site"] = "nope"
    bundle["method"]["reads"]["v_cf"]["site"] = "nope"
    group["method"]["writes"]["patch"]["site"] = "nope"
    return {
        "missing dataset": (dataset, alone["missing dataset"]),
        "missing bundle": (bundle, alone["missing bundle"]),
        "rule 23": (group, alone["rule 23"]),
    }


def test_a_rule_defect_beside_a_build_refusal_is_reported_as_the_rule(
    env: ResolutionEnv,
) -> None:
    """The compiler ran the checklist between ``expand`` and ``canonicalize``,
    so a document with a rule-4 defect *and* a dependency it could not
    resolve, or a rule-23 group, was refused for the rule — the earlier, more
    specific refusal (``TestPresetsOffline.test_a_group_on_a_read_only_component_is_refused_by_the_write_policy_first``
    in ``tests/neural/engines/pytorch_hooks/test_dbm_expert_neuron_run.py``
    says why a user should see it). ``build`` keeps that order: before a stage's refusal gets out it
    consults the checklist over the points it has parsed, so ``build``, the
    two verbs and the facade all give the checklist's text."""
    for name, (raw, stage_refusal) in _two_defect_documents(env).items():
        rule = _checklist_refusal(raw, env)
        assert rule[0] is ValidationError and "'nope'" in rule[1], name
        assert stage_refusal != rule[1], name
        assert _refusal_text(build, raw, env=env) == rule, name
        assert _refusal_text(_two_verbs, raw, env) == rule, name
        assert _refusal_text(_facade, raw, env) == rule, name


def _pytorch_fn_doc() -> dict[str, Any]:
    """A write through ``pytorch_fn`` — rule 13 refuses it under an engine
    that is not local (§2.8), and only when the engine is known."""
    raw = base_doc()
    raw["method"]["code"] = {
        "scale": {"locator": "tests.protocol._code_under_test.scale"}
    }
    raw["method"]["writes"]["patch"]["do"] = {"pytorch_fn": {"code": "scale"}}
    del raw["method"]["reads"]["v_cf"]
    del raw["method"]["intervened_models"][UNWRITTEN]  # nothing reads it now
    del raw["data"]["counterfactual"]
    return in_order(raw)


def test_a_rule_the_engine_decides_beside_a_build_refusal_is_reported_as_the_rule(
    env: ResolutionEnv,
) -> None:
    """Rules 13 and 30 are decided only when the engine is known, and the
    compiler decided them in the same pass as the rest of the checklist —
    before ``identify``. So ``compile_protocol(…, engine=…)`` on
    a ``pytorch_fn`` write under a non-local engine *and* a dataset that is
    not there reported rule 13, and still does: ``compile`` hands the offered
    set to ``build``'s guard. Without an engine the rule is undecidable and
    the dependency is the report — from ``build`` alone and from the facade
    with ``None``, as it always was."""
    raw = _pytorch_fn_doc()
    raw["data"]["base"]["dataset"] = "weekdays/no_such_table#train"
    remote = effective_capabilities("nnsight") - {"pytorch_fn_local"}
    rule = _refusal_text(
        validate_document,
        parse_document(raw),
        engine_capabilities=remote,
        model_info=env.model_info,
    )
    assert rule[0] is ValidationError and "[V13]" in rule[1]
    assert "only a local engine may run" in rule[1]
    facade = _refusal_text(compile_protocol, raw, env=env, engine=remote)
    assert facade == rule
    assert _refusal_text(build, raw, env=env, offered=remote) == rule

    dependency = _refusal_text(build, raw, env=env)
    assert dependency[0] is ValidationError and "no_such_table" in dependency[1]
    assert _refusal_text(_facade, raw, env) == dependency
    local = effective_capabilities("pytorch_hooks")
    assert _refusal_text(build, raw, env=env, offered=local) == dependency
    assert _refusal_text(compile_protocol, raw, env=env, engine=local) == dependency


def test_a_rule_defect_that_hides_the_stage_defect_builds(env: ResolutionEnv) -> None:
    """The same rule-23 document with the *read* dangling instead: the read is
    what names the featurizer, so the canonical form never reaches the group
    and the document builds — and ``validate`` refuses it for the rule, the
    text the compiler gave. Both orders end in the same refusal."""
    raw = gate_doc(group="head", component="block_output")
    raw["method"]["reads"]["v_cf"]["site"] = "nope"
    rule = _checklist_refusal(raw, env)
    assert isinstance(build(raw, env=env), CompiledProtocol)
    assert _refusal_text(_two_verbs, raw, env) == rule
    assert _refusal_text(_facade, raw, env) == rule


def test_the_guard_runs_the_checklist_only_when_a_stage_refuses(
    env: ResolutionEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document that builds never enters the checklist in ``build``; one a
    stage refuses does — over the points already parsed — and a document
    whose defect is the stage's alone still gets the stage's refusal."""
    seen: list[int] = []
    real = pipeline.validate_document

    def counting(*args: Any, **kwargs: Any) -> None:
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "validate_document", counting)
    build(base_doc(), env=env)
    assert seen == []
    group = gate_doc(group="head", component="block_output")
    kind, text = _refusal_text(build, group, env=env)
    assert kind is ValidationError and "[V23]" in text and seen == [1]


# --------------------------------------------------------------------------- #
# validate decides the engine — a name, an Engine, never "auto"
# --------------------------------------------------------------------------- #


class _Stub(Engine):
    def __init__(self, name: str, capabilities: frozenset[str]) -> None:
        self.name = name
        self.capabilities = capabilities
        self.components = frozenset(COMPONENTS)
        self.writable_components = frozenset(COMPONENTS)
        self.is_local = True

    def execute(self, compiled: CompiledProtocol, run: RunContext) -> RunResult:
        raise AssertionError(f"{self.name} was handed a document validate refuses")


def test_validate_against_an_engine_name_gives_check_engines_text(
    env: ResolutionEnv,
) -> None:
    """A DAS fit needs ``grad``; the nnsight engine has no grad path. Named,
    the engine's capabilities are the registry's rows, and the refusal is
    ``check_engine``'s (rule 13, the routing text) byte for byte."""
    compiled = _two_verbs(in_order(_train_doc()), env)
    assert "grad" in compiled.capabilities
    expected = _refusal_text(check_engine, compiled, effective_capabilities("nnsight"))
    assert _refusal_text(validate, compiled, "nnsight", env=env) == expected
    assert expected[0] is ValidationError and "grad" in expected[1]
    validate(compiled, "pytorch_hooks", env=env)  # the twin: the reference engine


def test_validate_against_an_engine_instance_and_the_offered_set(
    env: ResolutionEnv,
) -> None:
    compiled = _two_verbs(base_doc(), env)
    bare = _Stub("bare", frozenset())
    with pytest.raises(ValidationError) as err:
        validate(compiled, bare, env=env)
    assert err.value.rule == 13 and "paired_forward" in str(err.value)
    able = _Stub("able", frozenset({"paired_forward"}))
    assert validate(compiled, able, env=env) is compiled
    assert validate(compiled, compiled.capabilities, env=env) is compiled
    with pytest.raises(ValidationError):
        validate(compiled, frozenset(), env=env)


def test_validate_refuses_auto_as_an_engine(env: ResolutionEnv) -> None:
    """``auto`` is a routing policy; resolving it is the caller's job."""
    compiled = build(base_doc(), env=env)
    with pytest.raises(ValueError, match="routing policy"):
        validate(compiled, "auto", env=env)


def test_validate_refuses_an_engine_argument_of_the_wrong_kind(
    env: ResolutionEnv,
) -> None:
    """Neither a name, an ``Engine`` nor a set of capability names is a
    ``TypeError`` — never a capability set by accident: a number, a list of
    numbers, the ``Engine`` *class* (whose ``effective_capabilities`` is a
    property object, not a set). A name the registry does not know is a
    ``ValueError`` with the door's wording, not the registry's assertion."""
    compiled = build(base_doc(), env=env)
    with pytest.raises(TypeError, match="registered engine's name"):
        validate(compiled, 7, env=env)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="registered engine's name"):
        validate(compiled, [7, 8], env=env)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="registered engine's name"):
        validate(compiled, Engine, env=env)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="registered engine's name"):
        validate(compiled, _Stub, env=env)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a registered engine's name"):
        validate(compiled, "no_such_engine", env=env)


def test_check_engine_is_the_pipelines(env: ResolutionEnv) -> None:
    """The engine-aware re-entry the runner and the dry run call lives beside
    ``validate`` (the ``compile.py`` facade that re-exported it is gone)."""
    assert check_engine is pipeline.check_engine
    compiled = build(base_doc(), env=env)
    assert check_engine(compiled, compiled.capabilities) is None


# --------------------------------------------------------------------------- #
# the data rules — data=True is the `validate --data` pass
# --------------------------------------------------------------------------- #


def test_data_true_refuses_a_column_the_base_table_lacks(env: ResolutionEnv) -> None:
    raw = base_doc()
    raw["method"]["save"][0]["aggregation"]["a"] = "not_a_column"
    compiled = _two_verbs(raw, env)  # the document is valid without its rows
    expected = _refusal_text(check_data_columns, compiled, env)
    assert _refusal_text(validate, compiled, env=env, data=True) == expected
    assert "not_a_column" in expected[1]
    assert validate(compiled, env=env, data=False) is compiled


@pytest.mark.parametrize("name", CORPUS_FILES)
def test_data_true_passes_the_corpus(name: str, env: ResolutionEnv) -> None:
    path = CORPUS_DIR / name
    compiled = build(path, base_dir=path.parent, env=env)
    assert validate(compiled, env=env, data=True) is compiled


def test_the_cli_validate_verb_runs_the_data_rules_by_default(
    env: ResolutionEnv, artifacts_root: Path, tmp_path: Path, capsys: Any
) -> None:
    """``causalab validate`` refuses a column the base table lacks with and
    without ``--data`` — the flag names the default and is a
    documented no-op: exit code, stdout and stderr are identical with and
    without it, on a document the pass refuses and on one it passes."""
    from causalab.cli import main

    from tests.protocol._env import FIXTURES

    def verb(document: Path, *flags: str) -> tuple[int, str, str]:
        code = main(
            [
                "validate",
                "--engine",
                "auto",
                str(document),
                "--data-root",
                str(FIXTURES / "data"),
                "--artifacts-root",
                str(artifacts_root),
                *flags,
            ]
        )
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    raw = base_doc()
    raw["method"]["save"][0]["aggregation"]["a"] = "not_a_column"
    refused = tmp_path / "refused.json"
    refused.write_text(json.dumps(raw))
    bare = verb(refused)
    assert bare[0] == 1 and "not_a_column" in bare[2]
    assert verb(refused, "--data") == bare

    passing = tmp_path / "passing.json"
    passing.write_text(json.dumps(base_doc()))
    bare = verb(passing)
    assert bare[0] == 0 and bare[1].startswith("OK:")
    assert verb(passing, "--data") == bare
