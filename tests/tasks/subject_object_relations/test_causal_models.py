"""Property-tier invariants for the ``subject_object_relations`` factory task.

LM-free checks over the bundled relation content: the causal-model DAG and
determinism, the answer-space / ``output_tokens`` declaration, the object-flip
counterfactual semantics (RNG-snapshotted, seed-reproducible), the loader-facing
``GET_*`` accessors, and the model-agnostic data invariant (no Llama token /
position fields leaked into the committed JSON).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from causalab.causal.model import CausalModel
from causalab.tasks.subject_object_relations import counterfactuals as cf
from causalab.tasks.subject_object_relations.causal_models import (
    GET_CYCLIC_VARIABLES,
    GET_EMBEDDINGS,
    GET_PERIODIC_INFO,
    GET_TEMPLATE,
    GET_VARIABLE_VALUES,
    create_causal_model,
)
from causalab.tasks.subject_object_relations.config import (
    SubjectObjectRelationsConfig,
    load_manifest,
    relation_names,
)

pytestmark = pytest.mark.property

# A representative slice: an injective factual relation, a stripped-meta bias
# relation, a single-letter categorical relation, and a multi-object relation.
SAMPLE_RELATIONS = [
    "country_capital_city",
    "name_gender",
    "word_first_letter",
    "substance_phase",
]
_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "causalab"
    / "tasks"
    / "subject_object_relations"
    / "sources"
)

# Llama-3.1-8B-specific keys the build script must have dropped.
_FORBIDDEN_KEYS = {
    "prompt",
    "prompt_token_ids",
    "gold_first_id",
    "pred_first_id",
    "subj_last_idx",
    "subj_last",
    "prompt_last_idx",
    "prompt_last",
    "subj_char_start",
    "subj_char_end",
    "seq_len",
    "candidate_first_ids",
    "object_first_ids",
}


@pytest.fixture(params=SAMPLE_RELATIONS)
def config(request) -> SubjectObjectRelationsConfig:
    return SubjectObjectRelationsConfig(relation=request.param)


@pytest.fixture
def model_and_config(config):
    return create_causal_model(config), config


# ---------------------------------------------------------------------------
# Config / manifest
# ---------------------------------------------------------------------------


def test_manifest_lists_thirty_five_relations() -> None:
    manifest = load_manifest()
    assert manifest["n_relations"] == 35
    assert len(manifest["relations"]) == 35
    assert len(relation_names()) == 35


def test_invalid_relation_raises() -> None:
    with pytest.raises(ValueError, match="relation must be one of"):
        SubjectObjectRelationsConfig(relation="nonexistent_relation")


def test_config_loads_bundled_content(config) -> None:
    assert len(config.subjects) > 0
    assert len(config.objects) > 0
    assert len(config.templates) >= 1
    # subject_to_object covers exactly the subject domain.
    assert set(config.subject_to_object) == set(config.subjects)
    # every mapped object is in the declared answer space.
    assert set(config.subject_to_object.values()) <= set(config.objects)
    assert config.group in {"bias", "categorical", "injective"}
    for t in config.templates:
        assert "{subject}" in t


# ---------------------------------------------------------------------------
# Causal model
# ---------------------------------------------------------------------------


def test_factory_builds_valid_model(model_and_config) -> None:
    model, _ = model_and_config
    assert isinstance(model, CausalModel)
    # DAG: subject -> object -> raw_output, subject -> raw_input.
    assert set(model.variables) == {"subject", "object", "raw_input", "raw_output"}
    assert model.parents["object"] == ["subject"]
    assert model.parents["raw_input"] == ["subject"]
    assert model.parents["raw_output"] == ["object"]
    assert model.parents["subject"] == []


def test_object_is_deterministic_lookup(model_and_config) -> None:
    model, config = model_and_config
    for subject, expected_object in config.subject_to_object.items():
        trace = model.new_trace({"subject": subject})
        assert trace["object"] == expected_object
        assert trace["raw_output"] == config.output_prefix + expected_object
        assert trace["raw_input"] == config.templates[0].replace("{subject}", subject)


def test_output_tokens_declared_for_object(model_and_config) -> None:
    model, config = model_and_config
    ot = model.output_tokens["object"]
    assert set(ot) == set(config.objects)
    assert all(forms for forms in ot.values())
    # prefix string mode (first-token / continuation-aware grading).
    assert model.scoring is not None and model.scoring.string_mode == "prefix"
    assert model.match_modes == {"object": "prefix"}  # the derived view


def test_target_variable_is_object() -> None:
    from causalab.tasks.subject_object_relations.causal_models import TARGET_VARIABLE

    assert TARGET_VARIABLE == "object"


def test_derived_checker_prefix_semantics() -> None:
    """The task's grader is the spec's: prefix semantics over the object forms."""
    from causalab.tasks.loader import load_task

    task = load_task(
        "subject_object_relations",
        task_cfg=SubjectObjectRelationsConfig(relation="name_gender"),
    )
    # a single-token object graded exactly; a longer continuation still matches.
    assert task.checker({"string": " woman"}, "woman") is True
    assert task.checker({"string": "woman, typically"}, "woman") is True
    assert task.checker({"string": " man"}, "woman") is False


def test_create_causal_model_accepts_dict_and_str() -> None:
    """resolve_task feeds a raw dict; a bare relation name is also accepted."""
    from_dict = create_causal_model({"relation": "name_gender", "name": "x"})
    from_str = create_causal_model("name_gender")
    assert from_dict.values["object"] == from_str.values["object"]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_get_variable_values(model_and_config) -> None:
    model, config = model_and_config
    vv = GET_VARIABLE_VALUES(model)
    assert set(vv) == {"subject", "object"}
    assert vv["subject"] == config.subjects


def test_get_cyclic_and_periodic(model_and_config) -> None:
    model, _ = model_and_config
    assert GET_CYCLIC_VARIABLES(model) == set()
    assert GET_PERIODIC_INFO(model) is None


def test_get_embeddings_and_template(model_and_config) -> None:
    model, config = model_and_config
    emb = GET_EMBEDDINGS(model)
    assert set(emb) == {"subject", "object"}
    vec = emb["object"](config.objects[0])
    assert isinstance(vec, list) and all(isinstance(x, float) for x in vec)
    assert GET_TEMPLATE(model) == config.templates[0]


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------


def test_object_flip_counterfactual(model_and_config) -> None:
    model, config = model_and_config
    if len(config.objects) < 2:
        pytest.skip("single-object relation cannot flip")
    examples = cf.generate_dataset(model, n=8, seed=0)
    assert len(examples) == 8
    for ex in examples:
        base_obj = ex["input"]["object"]
        cf_obj = ex["counterfactual_inputs"][0]["object"]
        assert base_obj != cf_obj  # the answer flips (LRE interchange semantics)


def test_generate_dataset_deterministic(model_and_config) -> None:
    model, _ = model_and_config
    ex1 = cf.generate_dataset(model, n=6, seed=123)
    ex2 = cf.generate_dataset(model, n=6, seed=123)
    for a, b in zip(ex1, ex2):
        assert a["input"]["raw_input"] == b["input"]["raw_input"]
        assert (
            a["counterfactual_inputs"][0]["raw_input"]
            == b["counterfactual_inputs"][0]["raw_input"]
        )


def test_generate_dataset_cycles_templates(model_and_config) -> None:
    model, config = model_and_config
    templates = config.templates
    examples = cf.generate_dataset(model, n=2 * len(templates), seed=7)
    for i, ex in enumerate(examples):
        expected = templates[i % len(templates)]
        subject = ex["input"]["subject"]
        assert ex["input"]["raw_input"] == expected.replace("{subject}", subject)


def test_generate_dataset_restores_global_rng(model_and_config) -> None:
    model, _ = model_and_config
    random.seed(999)
    before = random.random()
    random.seed(999)
    cf.generate_dataset(model, n=8, seed=0)
    after = random.random()
    assert before == after


def test_resample_generator_and_registry(model_and_config) -> None:
    model, _ = model_and_config
    examples = cf.generate_resample_dataset(model, n=4, seed=0)
    assert len(examples) == 4
    assert set(cf.COUNTERFACTUAL_GENERATORS) == {"flip_object", "random_counterfactual"}
    for fn in cf.COUNTERFACTUAL_GENERATORS.values():
        example = fn()  # zero-arg introspection convention
        assert "input" in example and "counterfactual_inputs" in example


# ---------------------------------------------------------------------------
# Data invariant: no Llama token / position fields in the committed JSON
# ---------------------------------------------------------------------------


def test_bundled_json_has_no_llama_token_fields() -> None:
    relations_dir = _DATA_DIR / "relations"
    files = sorted(relations_dir.glob("*.json"))
    assert len(files) == 35
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        keys_seen: set[str] = set()

        def _walk(obj):
            if isinstance(obj, dict):
                keys_seen.update(obj.keys())
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    _walk(v)

        _walk(record)
        leaked = keys_seen & _FORBIDDEN_KEYS
        assert not leaked, f"{path.name} leaked Llama fields: {leaked}"


def test_bundled_json_has_no_literal_unicode_escapes() -> None:
    """No committed string value retains a literal ``\\uXXXX`` / ``\\xXX`` escape.

    Guards against the double-encoding class of bug: the upstream ``filtered.jsonl``
    stored some values double-escaped (e.g. ``Bras\\u00edlia`` with a real
    backslash), which the model can never emit → unscoreable. ``build_relations.py``
    now normalizes these at ingest; this asserts every ``sources/relations/*.json``
    value decodes to real characters, not a backslash-escape literal.
    """
    literal_escape = re.compile(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}")
    files = sorted((_DATA_DIR / "relations").glob("*.json"))
    assert len(files) == 35
    offenders: list[str] = []

    def _walk(obj, path: str) -> None:
        if isinstance(obj, str):
            if literal_escape.search(obj):
                offenders.append(f"{path}: {obj!r}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(k, path)
                _walk(v, path)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v, path)

    for p in files:
        _walk(json.loads(p.read_text(encoding="utf-8")), p.name)
    assert not offenders, f"literal unicode escapes in bundled JSON: {offenders}"
