"""`model.dtype` and `model.quantization` (spec §2.1, checklist rule 17).

Precision is not an execution flag: two runs of one protocol at bf16 and at
nf4 are two experiments. These tests pin that the document can say so, that
the canonical form always says so even when the author did not, that the
digest moves when the realization moves, and that an engine which cannot
realize a quantization refuses instead of running something else.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.engine import Engine, requires
from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import ARTIFACT_IDENTITY_KEYS, build_artifact_identity
from causalab.protocol.rules.capability import refuse_shortfall
from causalab.protocol.schema import parse_document
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import LOGIT_DIFF, base_doc, in_order, term
from tests.protocol._env import steps_of

pytestmark = pytest.mark.unit


def doc_with_model(**model: Any) -> dict[str, Any]:
    raw = base_doc()
    raw["model"] = {**raw["model"], **model}
    return in_order(raw)


# --------------------------------------------------------------------------- #
# dtype
# --------------------------------------------------------------------------- #


def test_dtype_materializes_even_when_unauthored(env):
    """An authored file may be silent about precision; a record may not."""
    assert canonicalize(base_doc(), env)["model"]["dtype"] == "fp32"


def test_dtype_is_part_of_the_experiment_not_of_the_run(env):
    fp32 = digest(canonicalize(base_doc(), env))
    bf16 = digest(canonicalize(doc_with_model(dtype="bf16"), env))
    assert fp32 != bf16
    # ... and an explicit fp32 is the same experiment as a silent one
    assert digest(canonicalize(doc_with_model(dtype="fp32"), env)) == fp32


def test_an_unknown_dtype_is_refused_with_suggestions():
    with pytest.raises(ParseError) as err:
        parse_document(doc_with_model(dtype="float16"))
    assert err.value.code == "P4"
    assert "fp16" in str(err.value)


def test_the_model_dtype_has_one_home(env):
    """`train.precision` used to carry a `model` entry that nothing enforced —
    it could (and in the corpus did) name a precision the run never used."""
    raw = base_doc()
    raw["method"]["train"] = {
        "objective": [[1.0, term("logits", "patched", dict(LOGIT_DIFF))]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 4},
        "precision": {"feature": "fp32", "loss": "fp32", "model": "bf16"},
    }
    with pytest.raises(ParseError) as err:
        parse_document(in_order(raw))
    assert err.value.code == "P3"
    assert "model" in str(err.value)


# --------------------------------------------------------------------------- #
# quantization
# --------------------------------------------------------------------------- #


NF4 = {"scheme": "nf4", "method": "bitsandbytes", "double_quant": True}
INT8 = {"scheme": "int8", "method": "bitsandbytes"}


def test_quantization_materializes_its_scheme_defaults(env):
    canonical = canonicalize(
        doc_with_model(dtype="bf16", quantization={"scheme": "nf4"}), env
    )
    assert canonical["model"]["quantization"] == {
        "scheme": "nf4",
        "method": "bitsandbytes",
        "compute_dtype": "bf16",  # defaults to the model's own dtype
        "double_quant": False,
    }


def test_int8_materializes_its_own_knob(env):
    canonical = canonicalize(doc_with_model(quantization={"scheme": "int8"}), env)
    assert canonical["model"]["quantization"]["int8_threshold"] == 6.0
    assert "double_quant" not in canonical["model"]["quantization"]


def test_int8_does_not_materialize_a_compute_dtype(env):
    """The regression: `compute_dtype` was materialized for every scheme, so
    two int8 documents differing only there hashed differently while running
    identically.

    📐 The engine reads it as `bnb_4bit_compute_dtype`; the int8 branch of
    `_bitsandbytes_config` builds `BitsAndBytesConfig(load_in_8bit=True,
    llm_int8_threshold=…)` and has nowhere to put it. A field in the canonical
    form that moves no number inverts the rule the form exists to keep.
    """
    canonical = canonicalize(
        doc_with_model(dtype="bf16", quantization={"scheme": "int8"}), env
    )
    assert "compute_dtype" not in canonical["model"]["quantization"]


def test_two_int8_documents_differing_only_in_dtype_still_differ(env):
    """Anti-vacuity for the test above: `model.dtype` DOES reach an int8 run
    (it is the dtype the un-quantized parts are realized in), so removing
    `compute_dtype` from the form must not have flattened the real axis too."""
    a = digest(canonicalize(doc_with_model(dtype="bf16", quantization=INT8), env))
    b = digest(canonicalize(doc_with_model(dtype="fp16", quantization=INT8), env))
    assert a != b


def test_quantization_moves_the_digest(env):
    plain = digest(canonicalize(doc_with_model(dtype="bf16"), env))
    quantized = digest(
        canonicalize(doc_with_model(dtype="bf16", quantization=NF4), env)
    )
    assert plain != quantized


def test_there_is_no_bare_int4():
    """`int4` names no single realization — refusing it is the feature."""
    with pytest.raises(ParseError) as err:
        parse_document(doc_with_model(quantization={"scheme": "int4"}))
    assert err.value.code == "P4"
    assert "nf4" in str(err.value)


@pytest.mark.parametrize(
    "quantization,field",
    [
        ({"scheme": "int8", "double_quant": True}, "double_quant"),
        ({"scheme": "int8", "compute_dtype": "fp16"}, "compute_dtype"),
        ({"scheme": "nf4", "int8_threshold": 6.0}, "int8_threshold"),
    ],
)
def test_a_knob_under_the_wrong_scheme_is_rule_17(quantization, field):
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(doc_with_model(quantization=quantization)))
    assert err.value.rule == 17
    assert field in str(err.value)


# --------------------------------------------------------------------------- #
# routing and stamping
# --------------------------------------------------------------------------- #


class _Plain(Engine):
    name = "plain"
    capabilities = frozenset({"paired_forward"})

    def execute(self, compiled, run):  # pragma: no cover — the check never gets here
        raise AssertionError


def test_quantization_requires_a_capability_and_refuses_without_it():
    doc = parse_document(doc_with_model(quantization=NF4))
    assert "quantized_weights" in requires(doc)
    with pytest.raises(ValidationError) as err:
        refuse_shortfall(requires(doc), _Plain().effective_capabilities)
    assert "quantized_weights" in str(err.value)


def _dict_keys(node: ast.AST) -> set[str]:
    """Every string key of every dict literal anywhere inside ``node``.

    Recursive on purpose. A ``**expr`` entry is ``keys[i] is None`` with the
    expression in ``values[i]``, so a splice's keys are not keys of the outer
    literal — and the splice may be a ternary, a call, or a comprehension.
    Walking the expression finds them whatever its shape, which a regex
    matching one source spelling does not: the first version of this guard
    matched ``{"x": ",".join(applied)}`` exactly, so a differently shaped
    splice would have reopened the very defect it was written to close.
    """
    return {
        key.value
        for inner in ast.walk(node)
        if isinstance(inner, ast.Dict)
        for key in inner.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _stamped_keys(tree: ast.AST) -> set[str]:
    """Every key ``execution.py``'s writer can put into an identity mapping.

    Three shapes, because all three appear or could: ``identity_base = {...}``,
    ``identity_base[...] = ...``, and ``identity_base.update(...)``. Anything
    assigned to ``identity_base`` or ``featurizer_identity`` counts; a stage's
    ``identity_fields`` stamps are `_stage_stamps`'s, by method rather
    than by name.
    """
    names = ("identity_base", "featurizer_identity")

    def _targets_identity(node: ast.Assign) -> bool:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                return True
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in names
            ):
                return True
        return False

    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _targets_identity(node):
            keys |= _dict_keys(node.value)
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    index = target.slice
                    if isinstance(index, ast.Constant) and isinstance(index.value, str):
                        keys.add(index.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
        ):
            for argument in node.args:
                keys |= _dict_keys(argument)
    return keys


def test_the_identity_schema_covers_every_key_the_engine_stamps():
    """`ARTIFACT_IDENTITY_KEYS` is closed, so it must contain every key the
    engine can actually stamp — otherwise a run refuses its own output.

    Not hypothetical: `neural/shared/execution.py` splices ``implementations``
    into ``identity_base`` whenever an executor reports applied requirements,
    and only the **nnsight** engine ever does (``attn_eager``, forcing eager
    attention to reach the pattern interior). No nnsight test drives a full
    ``run_protocol``/``write_outputs``, so every green run took a path where
    the extra key was absent, and an nnsight run that wrote a tensor file
    raised ``unknown ArtifactIdentity fields ['implementations']`` on its own
    stamp.

    The writer's keys are read out of its source rather than restated here:
    the point is to notice a key added by someone who never opens this file.
    """
    import causalab.neural

    source = (
        Path(causalab.neural.__file__).parent / "shared" / "execution.py"
    ).read_text()
    written = _stamped_keys(ast.parse(source))
    assert written, "found no identity_base keys — the reader is wrong"
    undeclared = sorted(written - set(ARTIFACT_IDENTITY_KEYS))
    assert not undeclared, (
        f"execution.py stamps {undeclared}, which build_artifact_identity "
        "refuses — a run would reject its own artifact"
    )


def _stage_stamps(tree: ast.AST) -> set[str]:
    """Every key any ``identity_fields`` body can put into its mapping —
    literal subscript assignments (``fields["axis"] = …``, whatever the
    local is called) and, through `_dict_keys`, the string keys of
    every dict literal inside it — so a stamp on a *new* ``Stage`` subclass
    that builds its dict under another name is read too, not only ``Gate``'s
    ``fields``. Deliberately over-broad: a dict literal in the body that is
    not the stamp (say ``json.dumps({"lo": …, "hi": …})`` in place of
    ``json.dumps(list(self.stretch))``) fails the census naming keys no
    bundle carries — a false alarm read once beats a stamp missed until a
    fit dies at its first save; the remedy is to hoist the literal out of
    the method, not to widen ``ARTIFACT_IDENTITY_KEYS``."""
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "identity_fields"):
            continue
        keys |= _dict_keys(node)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        keys.add(target.slice.value)
    return keys


def test_every_key_a_stage_stamps_is_an_artifact_identity_key() -> None:
    """A reader over every ``identity_fields`` body in the ``featurizers/``
    package — every submodule, so a stage kind moved between them is not lost
    — for the §2.5 stamps: the keys are literal, so a stamp a stage adds — ``stretch``,
    ``pool``, ``axis``, ``forward`` — is read here without an instance being
    built for it, under whatever local the method builds its dict. This is
    the census closed over ``identity_fields`` bodies; the per-instance guard
    in ``tests/neural/shared/test_featurizers.py`` is closed over classes.
    Two cases: a renamed *local* (``fields`` → ``out``) is still read, which
    is what scoping to the method bought; a renamed or removed *method*
    empties ``written`` and the ``<=`` line fails loudly, so the census
    cannot pass on nothing. Before ``axis`` and ``forward`` were
    registered, both walked through a green suite and a saving fit died at
    its first save."""
    import causalab.neural

    package = Path(causalab.neural.__file__).parent / "shared" / "featurizers"
    written: set[str] = set()
    for source in sorted(package.glob("*.py")):
        written |= _stage_stamps(ast.parse(source.read_text()))
    assert {"stretch", "pool", "pool_units", "axis", "forward"} <= written, written
    undeclared = sorted(written - set(ARTIFACT_IDENTITY_KEYS))
    assert not undeclared, (
        f"featurizers/ stamps {undeclared}, which build_artifact_identity "
        "refuses — a saving fit would die at its first save"
    )


def test_applied_implementations_are_stampable():
    """The specific key that was missing, through the real function."""
    stamped = build_artifact_identity(engine="nnsight", implementations="attn_eager")
    assert stamped["implementations"] == "attn_eager"


def test_the_realization_is_stampable_identity():
    """A featurizer fitted against bf16 weights is not the artifact fitted
    against fp32 ones, so the identity schema has to be able to say which."""
    assert "model_dtype" in ARTIFACT_IDENTITY_KEYS
    assert "model_quantization" in ARTIFACT_IDENTITY_KEYS


def test_a_swept_dtype_expands_like_any_other_axis(env):
    raw = doc_with_model(dtype={"sweep": ["fp32", "bf16"]})
    loaded = compile_protocol(raw, env=env)
    assert len(steps_of(loaded, env).points) == 2
    assert [c["model"]["dtype"] for c in steps_of(loaded, env).canonical] == [
        "fp32",
        "bf16",
    ]
    assert len(set(steps_of(loaded, env).digests)) == 2


def test_a_fit_bundle_is_refused_at_a_different_realization(env, artifacts_root):
    """The other half of stamping: corpus 09 loads a rotation fitted in bf16,
    so asking to apply it at fp32 refuses rather than quietly mixing the two.

    fp32 is the *implied* realization of a `model` block with no `dtype`, which
    is what makes this the shape people hit: an apply document that simply omits
    the field is refused against a bf16 fit. The refusal says where to write it.
    """
    from tests.protocol._env import CORPUS_DIR

    assert compile_protocol(
        CORPUS_DIR / "09_das_apply_im.json", env=env
    )  # bf16, as fitted
    with pytest.raises(ValidationError) as err:
        compile_protocol(
            CORPUS_DIR / "09_das_apply_im.json",
            env=env,
            overrides={"model.dtype": "fp32"},
        )
    assert err.value.rule == 15
    assert "model_dtype" in str(err.value)
    assert '"dtype": "bf16"' in str(err.value)  # the message says what to write
    assert "--dtype" in str(err.value)  # and that the flag is not the fix


# --------------------------------------------------------------------------- #
# the interning digests carry the realization too
# --------------------------------------------------------------------------- #


def _realizations(env) -> dict[str, Any]:
    """One document at three realizations, parsed to point documents.

    Same key, same revision, same addresses, same data — so the *only* thing
    that can separate their digests is how the weights are realized.
    """
    return {
        name: steps_of(compile_protocol(raw, env=env), env).documents[0]
        for name, raw in {
            "fp32": doc_with_model(dtype="fp32"),
            "bf16": doc_with_model(dtype="bf16"),
            "nf4": doc_with_model(dtype="bf16", quantization={"scheme": "nf4"}),
        }.items()
    }


def test_forward_group_keys_separate_the_realizations(env):
    """The bug: `_build_group` hashed ``{key, revision}`` and nothing about the
    numerics, while the canonical form has carried ``dtype`` and a normalized
    ``quantization`` block all along.

    A forward-group key is a *content* identity — equal keys mean "one
    shared harvest" — so an fp32 and a bf16 group interning together meant one
    of the two points silently read the other's activations. Three
    realizations, three distinct groups.
    """
    from causalab.neural.shared.plan import plan_point

    keys = {
        name: {group.key for group in plan_point(doc).groups}
        for name, doc in _realizations(env).items()
    }
    for name, groups in keys.items():
        assert groups, f"{name} planned no forward group"
    pooled = [k for groups in keys.values() for k in groups]
    assert len(set(pooled)) == len(pooled), (
        "two realizations share a forward group: "
        f"{ {name: sorted(g) for name, g in keys.items()} }"
    )


def test_closure_digests_separate_the_realizations(env):
    """The same claim for a read's content identity: an fp32 and a bf16
    harvest of one address are different tensors."""
    from causalab.neural.shared.plan import closure_digest

    digests = {
        name: closure_digest(doc, "v_cf") for name, doc in _realizations(env).items()
    }
    assert len(set(digests.values())) == 3, digests


def test_the_interning_digests_agree_with_the_canonical_form(env):
    """*Why* it is fixed, not just *that* it is.

    Both digests route through `canonical_model_ref`, the canonical form's own
    function, so a realization field added to §2.1 reaches them without anyone
    remembering to copy it. This asserts that plumbing rather than the values,
    because the values are the thing that is allowed to change.
    """
    from causalab.protocol.schema.explicit import canonical_model_ref

    for name, doc in _realizations(env).items():
        realization = canonical_model_ref(doc.model)
        assert set(realization) >= {"key", "revision", "dtype"}, name
        if name == "nf4":
            assert realization["quantization"]["scheme"] == "nf4"
            # the scheme's own defaults, materialized by the shared function
            assert realization["quantization"]["compute_dtype"] == "bf16"


def test_one_realization_still_interns_with_itself(env):
    """The other half: the fix must not stop a campaign from sharing.

    Two identically-realized point documents plan the same groups — otherwise
    "distinct digests per realization" would have been bought by making every
    digest unique, which would cost the dedup §3 exists for.
    """
    from causalab.neural.shared.plan import plan_point

    first = steps_of(
        compile_protocol(doc_with_model(dtype="bf16"), env=env), env
    ).documents[0]
    second = steps_of(
        compile_protocol(doc_with_model(dtype="bf16"), env=env), env
    ).documents[0]
    assert {g.key for g in plan_point(first).groups} == {
        g.key for g in plan_point(second).groups
    }


@pytest.mark.parametrize("backend", ["eager", "sdpa", "flash_attention_2"])
def test_attention_backend_survives_parsing_and_canonicalization(env, backend):
    from causalab.protocol.schema.explicit import canonical_model_ref

    raw = doc_with_model(attn_implementation=backend)
    parsed = parse_document(raw)
    assert parsed.model.attn_implementation == backend
    assert canonical_model_ref(parsed.model) == canonicalize(raw, env)["model"]


def test_omitting_attention_preserves_the_historical_canonical_model(env):
    assert canonicalize(base_doc(), env)["model"] == {
        "key": base_doc()["model"]["key"],
        "revision": "main",
        "dtype": "fp32",
    }


@pytest.mark.parametrize("backend", ["flash", "", None, False, 2])
def test_invalid_attention_backend_is_refused(backend):
    with pytest.raises(ParseError, match="attn_implementation"):
        parse_document(doc_with_model(attn_implementation=backend))


def test_attention_sweep_separates_campaigns_forwards_and_prefixes(env):
    from causalab.neural.shared.plan import plan_point

    loaded = compile_protocol(
        doc_with_model(
            attn_implementation={"sweep": ["eager", "sdpa", "flash_attention_2"]}
        ),
        env=env,
    )
    assert [
        point["model"]["attn_implementation"]
        for point in steps_of(loaded, env).canonical
    ] == ["eager", "sdpa", "flash_attention_2"]
    assert len(set(steps_of(loaded, env).digests)) == 3
    plans = [plan_point(doc) for doc in steps_of(loaded, env).documents]
    for field in ("key", "base_key"):
        groups = [{getattr(g, field) for g in plan.groups} for plan in plans]
        assert all(groups)
        assert groups[0].isdisjoint(groups[1] | groups[2])
        assert groups[1].isdisjoint(groups[2])


def test_declared_attention_is_checked_when_loading_fitted_artifacts(env):
    from tests.protocol._env import CORPUS_DIR

    with pytest.raises(ValidationError, match="model_attn_implementation"):
        compile_protocol(
            CORPUS_DIR / "09_das_apply_im.json",
            env=env,
            overrides={"model.attn_implementation": "sdpa"},
        )
