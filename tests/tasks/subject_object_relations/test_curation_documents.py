"""The curation sweep, as an intervention specification — the shape, not the run.

The per-relation base-accuracy table in this task's README was measured by a
producer the protocol refactor deleted (`data/curation_sweep.py`), while its
numbers stay load-bearing: they pick `config.py`'s default relation and the
relation a pinned tier would use. Recomputing it needs two things this seam
adds — task-generated tables, and a `match` that can grade a multi-token
object by its first token (the task's own `string_mode="prefix"`,
spec §2.10).

This test asserts the campaign is *expressible and valid* end to end on CPU:
tables for several relations, one document sweeping `data.base.dataset` over
them, every column reference checked. Running it is a GPU campaign and belongs
with the coherent-model tier, not here — what would silently rot without a
test is the seam, and that is what this covers.
"""

from __future__ import annotations


import pytest

from tests.protocol._env import steps_of

from causalab.protocol.pipeline import compile_protocol

from causalab.protocol.rules.data import check_data_columns
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.tasks.serialize import (
    serialize_counterfactual_dataset,
    write_dataset_table,
)
from causalab.tasks.subject_object_relations.config import SubjectObjectRelationsConfig

pytestmark = pytest.mark.unit

#: A slice of the curation's own selections: the strongest "green" relation
#: (single-letter answers), a two-object bias relation, and a flagged relation
#: whose objects are multi-token — the case that needs first-token grading.
RELATIONS = ["word_first_letter", "name_gender", "country_capital_city"]

MODEL = "Qwen/Qwen3-8B"


def _baseline_document(refs: list[str]) -> dict:
    """A no-intervention baseline: read the answer-position logits and score
    the declared answer forms, swept over one table per relation."""
    return {
        "header": {
            "protocol_version": "4",
            "description": "Per-relation base accuracy: the curation sweep as a document.",
        },
        "model": {"key": MODEL, "revision": "main"},
        "data": {"base": {"dataset": {"sweep": refs}, "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "positions": {"answer_tok": {"index": -1}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": "answer_tok"}},
            "save": [
                {
                    "read": "logits",
                    "model": "original",
                    "aggregation": {
                        "kind": "match",
                        "expected": "base_answer_forms",
                        "mode": "first_token",
                    },
                    "file_path": "accuracy.json",
                }
            ],
        },
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[ResolutionEnv, list[str], dict]:
    """Tables for the sampled relations, in a scratch data root.

    Deliberately not committed: 35 relations × 64 rows is a build product, and
    the parameters below are what make it reproducible.
    """
    root = tmp_path_factory.mktemp("sor_data")
    refs, manifests = [], {}
    for relation in RELATIONS:
        dataset = serialize_counterfactual_dataset(
            "subject_object_relations",
            n=8,
            seed=0,
            split="all",
            task_cfg=SubjectObjectRelationsConfig(relation=relation),
        )
        ref = f"subject_object_relations/{relation}"
        write_dataset_table(dataset.rows, root / f"{ref}.json")
        refs.append(ref)
        manifests[relation] = dataset
    env = ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )
    return env, refs, manifests


def test_the_swept_baseline_loads_validates_and_expands(built):
    env, refs, _ = built
    loaded = compile_protocol(_baseline_document(refs), env=env)
    # one point per relation...
    assert len(steps_of(loaded, env).points) == len(refs)
    # ...each stamping its own table's content digest (§2.2), so the points
    # are distinct provenance units rather than one document run three times
    stamped = {
        point["data"]["base"]["digest"] for point in steps_of(loaded, env).canonical
    }
    assert len(stamped) == len(refs)
    assert len(set(steps_of(loaded, env).digests)) == len(refs)


def test_every_column_reference_resolves(built):
    env, refs, _ = built
    refs_checked = check_data_columns(
        compile_protocol(_baseline_document(refs), env=env), env
    )
    assert "base_answer_forms" in refs_checked  # the answer-form group column


def test_the_build_reports_the_declared_prefix_mode(built):
    """Why the document needs ``first_token``: the task declares its answer
    match mode as ``prefix`` (multi-token objects like "Washington D.C."), and
    the builder reports that beside the digest so an author does not have to
    rediscover it."""
    _, _, built_datasets = built
    assert all(d.match_mode == "prefix" for d in built_datasets.values())
    assert all(d.answer_variable == "object" for d in built_datasets.values())
