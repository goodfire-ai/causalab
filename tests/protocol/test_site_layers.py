"""§2.4 ``layers`` — a site's depth is a **band** of layer indices.

The one-layer band ``[n]`` is the scalar site it replaced, everywhere (T-a);
every consumer either takes a band deliberately or refuses it by name (T-b);
a one-site band plans and lowers to the hand-written N-site document (T-c —
the run half is ``tests/neural/engines/pytorch_hooks/test_band_site_run.py``);
and the rename shipped as protocol_version 3 with a migration that proves the
70 rewritten documents mechanical (T-d's document half).
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.lowering import expand_families
from causalab.io.sources import apply_overrides
from causalab.protocol.migrate import (
    format_document,
    migrate_document,
    migrate_markdown,
    needs_migration,
)
from causalab.neural.shared.plan import plan_point
from causalab.protocol.lowering import band_member, lower_bands
from causalab.protocol.positions.alignment import site_depth, site_depths
from causalab.protocol.schema import (
    MIGRATABLE_PROTOCOL_VERSIONS,
    PROTOCOL_VERSION,
    SiteSpec,
    Sweep,
    parse_document,
)
from causalab.protocol.lowering import band_label, coordinate_label, label_value
from causalab.protocol.rules.document import validate_document

from tests.protocol._env import steps_of

from tests.protocol._docs import UNWRITTEN, base_doc, by_label, in_order, saved
from tests.protocol._env import CORPUS_MODEL, FIXTURES, build_env
from tests._helpers.paths import WORKFLOWS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ENV = build_env(FIXTURES / "artifacts")
#: The revision the rename was cut from: every document below carried
#: ``"layer"`` there, and ``causalab migrate`` wrote the committed v3 file.
#: The revisions here are in the development history; a checkout without that
#: history (an export) skips the round trips that read them.
BASE = "d6f27c24"
#: The one document re-described after ``BASE``: a later change reworded the locate
#: scan's ``description`` (it now names ``writes.<w>.ragged``) in the shipped
#: file and its corpus twin. Header prose enters no canonical form (§7), so
#: the base-revision round trip below holds these two paths to their current
#: prose and every other document byte for byte.
#: The model every shipped document and the fixture corpus named at ``BASE``
#: (gated; a literal compared against git history, nothing loaded —
#: ``tests/test_no_gated_models.py``).
BASE_MODEL = "meta-llama/Llama-3.1-8B"
#: The fixture corpus was retargeted after ``BASE`` from that gated model at an
#: ungated one (``tests/test_no_gated_models.py``): every
#: ``tests/protocols/*_im.json`` names `CORPUS_MODEL` where the base
#: spelled `BASE_MODEL`, and 08/09's artifacts moved from
#: ``weekdays/llama31_8b/`` to ``weekdays/qwen3_8b/``. Neither is what the
#: rename rewrote, so the base-revision round trip applies the retarget to the
#: base bytes of that one tree first — earned per file (the substitution must
#: change the bytes) and reaching no other tree.
RETARGETED_TREE = "tests/protocols/"
RETARGET = (
    (BASE_MODEL, CORPUS_MODEL),
    ("weekdays/llama31_8b/", "weekdays/qwen3_8b/"),
)
REDESCRIBED = frozenset({"tests/protocols/07_weekdays_locate_scan_im.json"})
#: Two trees were re-authored after ``BASE``. The onboarding tutorial moved to
#: ``Qwen/Qwen2.5-1.5B-Instruct`` (28 layers, other tables, other bands) where
#: the base ran the 1B Llama 3.2 Instruct checkpoint (16 layers). The method
#: library (``demos/methods/``, ``causalab/configs/`` at the base — see
#: `MOVED`) moved from `BASE_MODEL` (32 layers) to
#: ``Qwen/Qwen2.5-7B`` (28 layers): its located cell, its depth-scaled layer
#: literals, its band names and its artifact prefix all changed with the
#: model. No substitution carries either tree's base bytes to the current
#: ones, so the base bytes say nothing about the rename and the round trip
#: below skips those trees' re-authored files — earned per file (the migrated
#: base bytes differ from the current document, so a file the re-authoring
#: left alone — the four A3B documents, ``minimal_cpu.json``, the
#: mean-ablation workflow — is still checked) and reaching no other tree.
REAUTHORED_TREES = (
    "demos/onboarding_tutorial/",
    "demos/methods/",
    # the shipped band preset's hand-written twin followed the preset's retarget
    "tests/protocol/fixtures/band_patch_handwritten.json",
)
#: Re-authored files the tutorial has (its 25 base documents, re-authored
#: file by file), the method library has (20 of the 26
#: files the base knew; das_boundless, pca_harvest and pca_basis came later),
#: and the twin.
REAUTHORED_COUNT = 25 + 20 + 1
#: The same skip against the reads-first cut, one file more: the method
#: library's ``das_boundless.json`` existed there and not at ``BASE``.
REAUTHORED_COUNT_V4 = REAUTHORED_COUNT + 1
#: Where a current path lived at ``BASE``: the method library's documents were
#: ``causalab/configs/{protocols,workflows}/`` there, and two of them were
#: renamed with the retarget (the ``8b`` in their names was the model).
MOVED = {
    "demos/methods/protocols/": "causalab/configs/protocols/",
    "demos/methods/workflows/": "causalab/configs/workflows/",
}
RENAMED = {
    "demos/methods/protocols/weekdays_interchange.json": (
        "causalab/configs/protocols/weekdays_8b_interchange.json"
    ),
    "demos/methods/workflows/weekdays.json": "causalab/configs/workflows/weekdays_8b.json",
}


def base_path(rel: str) -> str:
    """``rel``'s path at ``BASE`` — the same unless the file moved or was renamed."""
    if rel in RENAMED:
        return RENAMED[rel]
    for current, base in MOVED.items():
        if rel.startswith(current):
            return base + rel[len(current) :]
    return rel


A3B = "Qwen/Qwen3.6-35B-A3B"


def v3_base_doc() -> dict[str, Any]:
    """`base_doc`'s protocol-3 ancestor, spelled as its authors did —
    the reads bound to a ``model`` and ``input``, the reserved ``original``,
    the reduction in ``metrics`` and a ``value`` save entry naming it. A
    **migration input** only (T-d, and the reads-first rewrite below): what
    ``causalab migrate`` makes of it is `base_doc`, key for key."""
    return {
        "header": {"protocol_version": "3"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "sites": {
                "tgt": {"component": "block_output", "layers": [3]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {
                    "site": "tgt",
                    "pos": -1,
                    "model": "original",
                    "input": "counterfactual",
                },
                "logits": {
                    "site": "lm_head",
                    "pos": -1,
                    "model": "patched",
                    "input": "base",
                },
            },
            "writes": {"patch": {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}},
            "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
            "metrics": {
                "ld": {
                    "kind": "logit_diff",
                    "of": "logits",
                    "a": "cf_answer",
                    "b": "base_answer",
                }
            },
            "save": [
                {
                    "value": "ld",
                    "model": "patched",
                    "input": "base",
                    "file_path": "ld.json",
                }
            ],
        },
    }


def _site(**fields: Any) -> dict[str, Any]:
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "block_output", **fields}
    return raw


def _band_doc(layers: list[int]) -> dict[str, Any]:
    """base_doc with its one write and its operand read on a band."""
    return _site(layers=layers)


# --------------------------------------------------------------------------- #
# T-b — the parser
# --------------------------------------------------------------------------- #


def test_the_field_is_a_tuple_of_layer_indices():
    doc = parse_document(_site(layers=[3]))
    assert doc.sites["tgt"].layers == (3,)
    assert parse_document(_site(layers=[3, 5, 8])).sites["tgt"].layers == (3, 5, 8)
    assert doc.sites["lm_head"].layers is None


@pytest.mark.parametrize(
    "bad, words",
    [
        ([], "at least one layer"),
        ([5, 3], "strictly increasing"),
        ([3, 3], "a layer repeats"),
        ([True], "got bool"),
        ([3, None], "got NoneType"),
        (None, "a list of layer indices"),
        ("3", "a list of layer indices"),
        (3.0, "a list of layer indices"),
    ],
)
def test_the_parser_refuses_a_malformed_band_by_name(bad, words):
    with pytest.raises(ParseError) as err:
        parse_document(_site(layers=bad))
    assert err.value.code == "P2"
    assert words in str(err.value)
    assert "sites.tgt.layers" in str(err.value)


def test_layerless_and_layered_components_are_held_to_the_field():
    raw = base_doc()
    raw["method"]["sites"]["lm_head"]["layers"] = [0]
    with pytest.raises(ParseError, match="layer-less"):
        parse_document(raw)
    raw = base_doc()
    del raw["method"]["sites"]["tgt"]["layers"]
    with pytest.raises(ParseError, match="needs 'layers'"):
        parse_document(raw)


def test_the_v2_spelling_is_refused_naming_the_rename_and_the_verb():
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "block_output", "layer": 3}
    with pytest.raises(ParseError) as err:
        parse_document(raw)
    assert err.value.code == "P3"
    assert "'layers'" in str(err.value)
    assert "causalab migrate" in str(err.value)


def test_a_v2_header_is_refused_by_name_and_told_how_to_migrate():
    """The rule ``test_protocol_v2`` pins for v1, applied to v2 (spec §7)."""
    for version in MIGRATABLE_PROTOCOL_VERSIONS:
        raw = base_doc()
        raw["header"]["protocol_version"] = version
        with pytest.raises(ParseError) as err:
            parse_document(raw)
        assert err.value.code == "P2"
        assert f"protocol_version {version!r} document" in str(err.value)
        assert "causalab migrate" in str(err.value)
        assert PROTOCOL_VERSION in str(err.value)
    assert PROTOCOL_VERSION == "4" and MIGRATABLE_PROTOCOL_VERSIONS == ("2", "3")


def test_a_bare_index_is_the_one_layer_band_at_parse_and_in_canonical_form():
    """An axis over ``layers`` and a workflow ``emit`` hand a point the
    index; the two spellings are one document (spec §2.4)."""
    listed, bare = _site(layers=[3]), _site(layers=3)
    import dataclasses

    assert dataclasses.replace(parse_document(listed), raw={}) == dataclasses.replace(
        parse_document(bare), raw={}
    )
    assert canonicalize(listed, ENV) == canonicalize(bare, ENV)
    assert digest(canonicalize(listed, ENV)) == digest(canonicalize(bare, ENV))
    assert canonicalize(bare, ENV)["method"]["sites"]["tgt"]["layers"] == [3]


def test_a_sweep_over_layers_is_indexed_by_layer():
    raw = _site(layers={"sweep": {"range": [1, 4]}})
    layers = parse_document(raw).sites["tgt"].layers
    assert isinstance(layers, Sweep) and layers.values == ((1,), (2,), (3,))
    from causalab.neural.shared.sweep import expand

    points = expand(raw).points
    assert [p.coords["sites.tgt.layers"] for p in points] == [1, 2, 3]
    assert [parse_document(p.raw).sites["tgt"].layers for p in points] == [
        (1,),
        (2,),
        (3,),
    ]


def test_a_sweep_over_bands_records_the_band_as_its_coordinate():
    raw = _site(layers={"sweep": [[1, 2], [3]]})
    from causalab.neural.shared.sweep import expand

    points = expand(raw).points
    assert [p.coords["sites.tgt.layers"] for p in points] == [[1, 2], [3]]
    assert parse_document(points[0].raw).sites["tgt"].layers == (1, 2)


# --------------------------------------------------------------------------- #
# T-a — the one-element band is the scalar case everywhere
# --------------------------------------------------------------------------- #


def test_one_element_band_plans_as_the_scalar_site_did():
    doc = parse_document(_site(layers=[3]))
    assert site_depth(doc, "tgt") == site_depths(doc, "tgt")[0]
    assert site_depth(doc, "tgt")[0] == 3
    plan = plan_point(doc)
    assert plan.num_forwards == 2
    assert [t.read for g in plan.groups for t in g.taps] == ["v_cf", "logits"]
    # the same document, index spelled: identical groups, identical keys
    twin = plan_point(parse_document(_site(layers=3)))
    assert [g.key for g in twin.groups] == [g.key for g in plan.groups]
    assert lower_bands(doc) is doc  # nothing to lower: the names are the author's


def test_coordinate_labels_and_band_labels():
    assert coordinate_label({"sites.target.layers": 3}) == "[target.layers=3]"
    assert coordinate_label({"sites.target.layers": [3]}) == "[target.layers=3]"
    assert coordinate_label({"sites.target.layers": [10, 11, 12]}) == (
        "[target.layers=10..12]"
    )
    assert label_value([10, 12, 15]) == "10+12+15"
    assert band_label((7,)) == "7"
    # neither the label syntax's comma nor its brackets appear in a band label
    assert all(c not in band_label((1, 2, 4)) for c in ",[]")


def test_the_artifact_stamp_carries_the_band_as_a_list():
    from causalab.protocol.identity import site_identity
    from causalab.protocol.rules.data import _featurizer_realization

    doc = parse_document(_site(layers=[3]))
    assert site_identity(doc, "tgt") == {"component": "block_output", "layers": [3]}
    raw = _site(layers=[3])
    raw["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": 2, "file_path": "rot.safetensors"}
    }
    raw["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    raw["method"]["writes"]["patch"]["featurizer"] = "rot"
    doc = parse_document(in_order(raw))
    expected = _featurizer_realization(doc, "rot")
    assert expected["site"] == {"component": "block_output", "layers": [3]}
    assert json.dumps(expected["site"], sort_keys=True) == json.dumps(
        site_identity(doc, "tgt"), sort_keys=True
    )


def test_the_select_to_fit_handoff_hands_the_fit_the_index():
    """``weekdays_8b``'s ``select`` emits the scan's coordinate — the layer
    index — and the fit document's ``sites.target.layers`` receives it as
    the one-layer band."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["layers"] = {"artifact": "best", "key": "best_layer"}
    import types

    from causalab.io.env import resolve_artifact_fields

    env = types.SimpleNamespace(
        artifacts=types.SimpleNamespace(read_value=lambda artifact, key: 18)
    )
    resolved = resolve_artifact_fields(raw, env)
    assert resolved["method"]["sites"]["tgt"]["layers"] == 18
    assert parse_document(resolved).sites["tgt"].layers == (18,)
    overridden = apply_overrides(base_doc(), {"sites.tgt.layers": 18})
    assert parse_document(overridden).sites["tgt"].layers == (18,)
    assert parse_document(
        apply_overrides(base_doc(), {"sites.tgt.layers": [4, 5]})
    ).sites["tgt"].layers == (4, 5)
    workflow = json.loads((WORKFLOWS_DIR / "weekdays.json").read_text())
    assert workflow["steps"]["best"]["inputs"]["emit"]["best_layer"] == (
        "sites.target.layers"
    )
    assert workflow["steps"]["fit"]["set"]["sites.target.layers"] == {
        "artifact": "best",
        "key": "best_layer",
    }


# --------------------------------------------------------------------------- #
# T-b — every consumer, per member
# --------------------------------------------------------------------------- #


def test_rule_4_holds_every_member_of_a_band():
    raw = _site(layers=[3, 40])  # gpt2 has 12 layers
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(raw), model_info=ENV.model_info)
    assert err.value.rule == 4
    assert "layer 40 out of range" in str(err.value)
    assert "sites.tgt.layers" in str(err.value)
    validate_document(parse_document(_site(layers=[0, 11])), model_info=ENV.model_info)
    canonicalize(_site(layers=[0, 11]), ENV)  # every member inside


def test_the_stream_check_holds_every_member_of_a_band():
    raw = base_doc()
    raw["model"] = {"key": A3B, "revision": "main", "dtype": "bf16"}
    raw["method"]["sites"]["tgt"] = {"component": "attention_premix", "layers": [3, 4]}
    with pytest.raises(ValidationError, match="exists only on a 'full_attention'"):
        # layer 4 is Gated DeltaNet
        validate_document(parse_document(in_order(raw)), model_info=ENV.model_info)
    raw["method"]["sites"]["tgt"]["layers"] = [3, 7]
    validate_document(parse_document(in_order(raw)), model_info=ENV.model_info)
    canonicalize(in_order(raw), ENV)  # both full attention


def test_rule_21_reads_a_band_operand_member_by_member():
    validate_document(parse_document(_band_doc([3, 4, 5])))  # equal depth, per member
    raw = _band_doc([3, 4, 5])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [4, 5, 6]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    assert err.value.rule == 21 and "layers 4..6" in str(err.value)


def test_rule_21_broadcasts_any_other_operand_to_the_bands_shallowest_member():
    raw = _band_doc([3, 4])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [5]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    assert err.value.rule == 21
    raw["method"]["sites"]["src"]["layers"] = [3]
    validate_document(parse_document(in_order(raw)))
    # and a band read feeding a one-layer write: its deepest member counts
    raw = _site(layers=[4])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [3, 5]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ValidationError) as err:
        validate_document(parse_document(in_order(raw)))
    assert err.value.rule == 21
    raw["method"]["sites"]["src"]["layers"] = [3, 4]
    validate_document(parse_document(in_order(raw)))


def test_site_depth_is_the_shallowest_member_and_site_depths_every_one():
    doc = parse_document(_band_doc([3, 4, 5]))
    assert site_depth(doc, "tgt")[0] == 3
    assert [d[0] for d in site_depths(doc, "tgt")] == [3, 4, 5]
    assert site_depths(doc, "lm_head") == (site_depth(doc, "lm_head"),)


def test_head_stats_refuses_a_band_cell_by_name(tmp_path):
    from causalab.analysis import head_stats
    from causalab.io.step_io import StepError

    from tests.step_scripts import put_table, run_step

    rows = [
        {"sites.target.layers": [0, 1], "sites.target.head": 0, "value": 1.0},
        {"sites.target.layers": 2, "sites.target.head": 0, "value": 1.0},
    ]
    table = put_table(tmp_path / "in.json", rows)
    with pytest.raises(StepError, match="layer band"):
        run_step(head_stats, {"table": table}, {"stats": tmp_path / "out.json"})


# --------------------------------------------------------------------------- #
# T-c — lowering: the N-site document the author would have written
# --------------------------------------------------------------------------- #


def test_lowering_fans_a_band_out_to_members_read_write_and_model():
    doc = parse_document(_band_doc([3, 4, 5]))
    low = lower_bands(doc)
    members = [band_member("tgt", n) for n in (3, 4, 5)]
    assert sorted(low.sites) == sorted([*members, "lm_head"])
    assert [low.sites[m].layers for m in members] == [(3,), (4,), (5,)]
    assert low.sites[members[0]].component == "block_output"
    assert sorted(low.reads) == sorted(
        ["logits", *(band_member("v_cf", n) for n in (3, 4, 5))]
    )
    assert low.reads["v_cf[layers=4]"].site == "tgt[layers=4]"
    assert low.writes["patch[layers=4]"].site == "tgt[layers=4]"
    assert low.writes["patch[layers=4]"].do.payload.read == "v_cf[layers=4]"
    assert low.intervened_models["patched"].writes == tuple(
        band_member("patch", n) for n in (3, 4, 5)
    )
    assert lower_bands(low) == low  # idempotent
    assert low.raw["method"]["sites"]["tgt[layers=5]"] == {
        "component": "block_output",
        "layers": [5],
    }
    assert by_label(low) == by_label(doc) and low.save == doc.save


def test_lowering_pairs_band_operands_by_index_and_broadcasts_the_rest():
    raw = _band_doc([4, 5])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [1, 2]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    low = lower_bands(parse_document(in_order(raw)))
    assert low.writes["patch[layers=4]"].do.payload.read == "v_cf[layers=1]"
    assert low.writes["patch[layers=5]"].do.payload.read == "v_cf[layers=2]"
    raw["method"]["sites"]["src"]["layers"] = [1]  # a one-layer read: broadcast
    low = lower_bands(parse_document(in_order(raw)))
    assert low.writes["patch[layers=4]"].do.payload.read == "v_cf"
    assert low.writes["patch[layers=5]"].do.payload.read == "v_cf"
    assert "v_cf" in low.reads and low.reads["v_cf"].site == "src"


def test_a_band_plans_the_forward_groups_of_its_hand_written_twin():
    """ROME's shape: one site over ten layers plans exactly what
    ``band_patch_handwritten.json``'s eight sites plan — the same groups, the
    same tap depths, the same first write, the same resume point. The three
    bands the fixture windows are three band sites here (one read, one write
    each) where the fixture shares ten writes between them."""
    from causalab.protocol.pipeline import compile_protocol
    from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

    from tests.protocol._env import TASKS_ROOT

    env = ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=FIXTURES / "artifacts"),
    )
    hand = steps_of(
        compile_protocol(FIXTURES / "band_patch_handwritten.json", env=env), env
    ).documents[0]
    raw = json.loads((FIXTURES / "band_patch_handwritten.json").read_text())
    method = raw["method"]
    bands = {"band4_L9": (9, 13), "band4_L13": (13, 17), "band8_L9": (9, 17)}
    method["sites"] = {
        **{
            f"a_{name}": {"component": "attention_output", "layers": list(range(*span))}
            for name, span in bands.items()
        },
        "lm_head": {"component": "lm_head"},
    }
    method["reads"] = {
        **{f"v_{name}": {"site": f"a_{name}", "pos": "tap"} for name in bands},
        **{k: v for k, v in method["reads"].items() if k.startswith("logits_")},
    }
    method["writes"] = {
        f"w_{name}": {"site": f"a_{name}", "pos": "tap", "do": {"swap": f"v_{name}"}}
        for name in bands
    }
    # one band read per model on the un-intervened counterfactual forward; each
    # band model reads its own logits and carries its one band write
    method["intervened_models"] = {
        "original_counterfactual": {
            "input": "counterfactual",
            "reads": [f"v_{name}" for name in bands],
        },
        **{
            name: {
                "input": "base",
                "reads": [f"logits_{name}"],
                "writes": [f"w_{name}"],
            }
            for name in bands
        },
    }
    band = steps_of(compile_protocol(raw, env=env, base_dir=FIXTURES), env).documents[0]
    hand_plan, band_plan = plan_point(hand), plan_point(band)
    assert band_plan.num_forwards == hand_plan.num_forwards == 4
    low = lower_bands(band)
    assert low.intervened_models["band8_L9"].writes == tuple(
        band_member("w_band8_L9", n) for n in range(9, 17)
    )
    assert len(low.writes) == 16 and len(hand.writes) == 8  # the fixture shares
    for hg, bg in zip(hand_plan.groups, band_plan.groups):
        assert (hg.model, hg.input) == (bg.model, bg.input)
        assert {t.depth for t in hg.taps} == {t.depth for t in bg.taps}
        assert hg.write_depth == bg.write_depth
        assert hg.resume_at == bg.resume_at and hg.stop_after == bg.stop_after


def test_at_once_windows_over_the_band_keep_the_hand_written_fixture():
    """The other band shape (§3.1) still expands to the fixture, now spelled
    ``layers`` — an ``at_once`` member is a layer index, a one-layer site."""
    corpus = json.loads((REPO / "tests/protocols/16_at_once_band_im.json").read_text())
    site = corpus["method"]["sites"]["a"]
    assert "layers" in site and "layer" not in site and site["names"] == "a{layers}"
    expanded = expand_families(corpus)
    assert sorted(expanded["method"]["sites"])[:3] == ["a10", "a11", "a12"]
    assert expanded["method"]["sites"]["a10"]["layers"] == [10]  # the index, a band
    assert parse_document(expanded).sites["a10"].layers == (10,)
    hand = json.loads((FIXTURES / "at_once_band_handwritten.json").read_text())
    assert hand["method"]["sites"]["a10"] == {
        "component": "attention_output",
        "layers": [10],
    }
    # the corpus copy targets the fixture tables, the hand-written one the
    # shipped ones: the *method* is what the sugar must reproduce
    assert canonicalize(expanded, ENV)["method"] == canonicalize(hand, ENV)["method"]


def test_at_once_composes_with_layers_a_band_per_member():
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {
        "component": "block_output",
        "layers": {"at_once": [[3, 4], [5, 6]]},
    }
    expanded = expand_families(raw)
    names = sorted(n for n in expanded["method"]["sites"] if n != "lm_head")
    assert names == ["tgt[layers=3..4]", "tgt[layers=5..6]"]
    doc = parse_document(expanded)
    assert doc.sites["tgt[layers=3..4]"].layers == (3, 4)
    assert doc.writes["patch[layers=5..6]"].do.payload.read == "v_cf[layers=5..6]"
    validate_document(doc)
    low = lower_bands(doc)
    assert "tgt[layers=3..4][layers=4]" in low.sites


@pytest.mark.parametrize(
    "mutate, words",
    [
        (
            lambda m: m["save"].append(saved("v_cf", UNWRITTEN, "v.safetensors")),
            "save entry 'v_cf'",
        ),
        (
            lambda m: m["save"].append(
                {
                    "read": "v_cf",
                    "model": "original_counterfactual",
                    "aggregation": {
                        "kind": "logit_diff",
                        "a": "cf_answer",
                        "b": "base_answer",
                    },
                    "file_path": "bad.json",
                }
            ),
            "aggregation 'bad' reduces read 'v_cf'",
        ),
        (
            lambda m: (
                m.__setitem__("featurizers", {"rot": {"kind": "subspace", "k": 2}}),
                m["reads"]["v_cf"].__setitem__("featurizer", "rot"),
                m["writes"]["patch"].__setitem__("featurizer", "rot"),
            ),
            "names a featurizer",
        ),
    ],
)
def test_what_a_band_has_no_member_for_is_refused_by_name(mutate, words):
    raw = _band_doc([3, 4])
    mutate(raw["method"])
    doc = parse_document(in_order(raw))
    with pytest.raises(ProtocolError) as err:
        lower_bands(doc)
    assert words in str(err.value) and "layers 3..4" in str(err.value)


def test_a_one_layer_write_fed_by_a_band_read_is_refused_by_name():
    raw = _site(layers=[4])
    raw["method"]["sites"]["src"] = {"component": "block_output", "layers": [1, 2]}
    raw["method"]["reads"]["v_cf"]["site"] = "src"
    with pytest.raises(ProtocolError, match="a band read is N tensors"):
        lower_bands(parse_document(in_order(raw)))
    raw["method"]["sites"]["tgt"]["layers"] = [4, 5, 6]  # a band of another length
    with pytest.raises(ProtocolError, match="same number of layers"):
        lower_bands(parse_document(in_order(raw)))


def test_resolve_site_refuses_a_multi_layer_band_without_a_model():
    """The refusal is the resolver's own, before any module is touched."""
    from causalab.neural.shared.sites import resolve_site

    class NoBundle:
        pass

    with pytest.raises(ProtocolError, match="resolve_band"):
        resolve_site(NoBundle(), SiteSpec(component="block_output", layers=(3, 4)))


# --------------------------------------------------------------------------- #
# T-d — protocol_version 3 and the migration
# --------------------------------------------------------------------------- #


def _v2_of(v3: dict[str, Any]) -> dict[str, Any]:
    """The protocol_version 2 spelling of a v3 document — the inverse of the
    rename, so the round trip below is self-contained."""
    out = copy.deepcopy(v3)
    out["header"]["protocol_version"] = "2"
    for site in out["method"].get("sites", {}).values():
        if "layers" in site:
            value = site.pop("layers")
            site["layer"] = value[0] if isinstance(value, list) else value
            # key order as the v2 author had it: component, layer, the rest
            rest = {
                k: site.pop(k) for k in list(site) if k not in ("component", "layer")
            }
            site.update(rest)
        if isinstance(site.get("names"), str):
            site["names"] = site["names"].replace("{layers}", "{layer}")
    for table in out["method"].values():
        if isinstance(table, dict):
            for entry in table.values():
                if isinstance(entry, dict) and isinstance(entry.get("names"), str):
                    entry["names"] = entry["names"].replace("{layers}", "{layer}")
    for im in out["method"].get("intervened_models", {}).values():
        if isinstance(im.get("writes"), list):
            for item in im["writes"]:
                if isinstance(item, dict):
                    for selector in item.values():
                        if isinstance(selector, dict) and "layers" in selector:
                            selector["layer"] = selector.pop("layers")
    return json.loads(
        json.dumps(out).replace(".layers", ".layer")  # dotted ids in descriptions
    )


def _documents() -> list[Path]:
    globs = (
        "tests/protocols/*_im.json",
        "tests/golden/protocols/*_im.json",
        # the shipped method documents under demos/methods/ included
        "demos/*/protocols/*.json",
        "tests/protocol/fixtures/band_patch_handwritten.json",
    )
    return sorted(p for g in globs for p in REPO.glob(g))


def _workflows() -> list[Path]:
    # the shipped method workflows under demos/methods/ included
    return sorted(REPO.glob("demos/*/workflows/*.json"))


def test_every_shipped_document_is_at_version_4_and_spells_layers():
    docs = _documents()
    # the 26 method documents, the 16 corpus and 16 golden fixtures, the 18
    # onboarding-tutorial documents and the hand-written twin. The glob reaches
    # one directory level, so the 29 documents of the replication packages
    # under demos/papers/ are not censused here — a hole the weekdays demo's
    # move under papers/ made visible (its five documents left this count).
    # Closing it is not free: a package-root JSON is not always a document,
    # and one apply document carries an axis id (`site.layers`) the inverse
    # rename below cannot restore, so it is a separate change.
    assert len(docs) >= 26 + 16 + 16 + 18 + 1
    layered = 0
    for path in docs:
        raw = json.loads(path.read_text())
        assert raw["header"]["protocol_version"] == PROTOCOL_VERSION, path
        assert not needs_migration(raw), path
        for name, site in raw["method"]["sites"].items():
            assert "layer" not in site, (path, name)
            layered += "layers" in site
        assert '"layer"' not in path.read_text(), path
    assert layered >= 70
    for path in _workflows():
        text = path.read_text()
        assert '.layer"' not in text and ".layer=" not in text, path
        assert not needs_migration(json.loads(text)), path


#: Every revision the committed corpus was rewritten from, with the
#: documents each round trip exempts and the re-authored files it skips.
#: ``d6f27c24`` is the ``layers`` rename (its exemptions are the re-described
#: and retargeted files above). ``38613c7fc`` is the reads-first rewrite, cut
#: from a tree that already carried both, so it re-describes nothing still
#: checked (the path-patching preset re-described since is in the method
#: library, re-authored after both cuts) beyond `REDESCRIBED_LATER`. Both cuts predate the move and the
#: re-authoring of `REAUTHORED_TREES`, so `base_path` and the per-file skip
#: apply to each; the skip also covers the tutorial workflow whose ``set``
#: override was re-spelled by hand for v4 (the migrator rewrites documents,
#: not the overrides a workflow applies to them).
#: The corpus and golden-tier documents whose ``description`` was reworded
#: after both cuts. Their prose is held to the current text, as
#: `REDESCRIBED`'s is.
REDESCRIBED_LATER = frozenset(
    {
        "tests/protocols/05_dbm_im.json",
        "tests/protocols/13_random_subspace_control_im.json",
        "tests/protocols/14_multi_position_patch_im.json",
        "tests/golden/protocols/drift_interchange_im.json",
        "tests/golden/protocols/mixing_scan_first_im.json",
        "tests/golden/protocols/mixing_scan_last_im.json",
        "tests/golden/protocols/mixing_scan_middle_im.json",
    }
)
BASES: tuple[tuple[str, frozenset[str], int], ...] = (
    ("d6f27c24", REDESCRIBED | REDESCRIBED_LATER, REAUTHORED_COUNT),
    ("38613c7fc", REDESCRIBED_LATER, REAUTHORED_COUNT_V4),
)

#: Per base, how many metrics carry a retired ``token_form`` on dataset
#: columns, each ``bare`` or ``space_prefixed``. Both bases predate the
#: retirement, e5d3b7654. Migrate refuses such a key because it cannot read
#: the table (§2.10). The refusal's remedy is to rewrite each answer string
#: of the columns and then delete the key. The committed documents are the
#: bases with the key deleted, so the round trip deletes it, and the count
#: pins that the refusal fired for each metric. After the deletion each
#: answer string is scored as the table writes it (§2.10), so a table that
#: lists both spellings credits both.
RETIRED_COLUMN_FORMS: dict[str, int] = {"d6f27c24": 124, "38613c7fc": 126}

#: One column metric of a migrate refusal: its name, its spelled columns and
#: the retired form, as `causalab.protocol.migrate` words it alone and among
#: several.
_REFUSED_COLUMN_METRIC = re.compile(
    r"metric '(\w+)' reads its answers from dataset columns \((.*?)\) under "
    r"(?:the retired token_form )?'(\w+)'"
)


def _refused_column_metrics(err: ParseError) -> list[tuple[str, list[str], str]]:
    """Each column metric a migrate refusal names, as ``(name, columns,
    form)``. Empty for any other refusal."""
    if not (err.path or "").startswith("method.metrics"):
        return []
    return [
        (name, re.findall(r"'([^']*)'", spelled), form)
        for name, spelled, form in _REFUSED_COLUMN_METRIC.findall(str(err))
    ]


def _migrated_as_retired(before: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """``before`` migrated, with each retired column ``token_form`` the
    migration refuses deleted, and the number of keys deleted. One refusal
    names every such metric. Any other refusal propagates."""
    raw = copy.deepcopy(before)
    try:
        return migrate_document(raw), 0
    except ParseError as err:
        refused = _refused_column_metrics(err)
        if not refused:
            raise
    for name, _, form in refused:
        assert form in ("bare", "space_prefixed"), (name, form)
        del raw["method"]["metrics"][name]["token_form"]
    return migrate_document(raw), len(refused)


def _skip_without(base: str) -> None:
    """Skip where ``base`` is not in this checkout's history (an export)."""
    try:
        subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"{base}^{{commit}}"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError):
        pytest.skip(f"revision {base} is not in this checkout's history")


@pytest.mark.parametrize(
    "base, redescribed, reauthored_count", BASES, ids=[b[0] for b in BASES]
)
def test_the_migration_reproduces_every_committed_document_from_the_base_revision(
    base: str, redescribed: frozenset[str], reauthored_count: int
):
    """The round trip against the bytes as they were at ``base`` — a revision
    the corpus was rewritten from — for every document and workflow the
    rewrite touched, byte for byte except the ``redescribed`` documents'
    header prose and the ``RETARGETED_TREE``'s model (the ``layers`` base
    only), and skipping the ``REAUTHORED_TREES``' re-authored files; a moved file is read at the path it had there
    (``base_path``). Skipped where the history is not at hand (an export)."""
    _skip_without(base)
    checked = 0
    reauthored = 0
    retired = 0
    seen_redescribed: set[str] = set()
    for path in [*_documents(), *_workflows()]:
        rel = path.relative_to(REPO).as_posix()
        shown = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{base}:{base_path(rel)}"],
            capture_output=True,
            text=True,
        )
        if shown.returncode != 0:
            continue  # a document added after the base
        text = shown.stdout
        if base == BASES[0][0] and rel.startswith(RETARGETED_TREE):
            retargeted = text
            for old, new in RETARGET:
                retargeted = retargeted.replace(old, new)
            assert retargeted != text, rel  # the exemption is earned
            text = retargeted
        before = json.loads(text)
        after = json.loads(path.read_text())
        migrated, deleted = _migrated_as_retired(before)
        retired += deleted
        if rel.startswith(REAUTHORED_TREES) and json.dumps(
            migrated, sort_keys=True
        ) != json.dumps(after, sort_keys=True):
            # the exemption is earned per file: only a document whose round
            # trip cannot hold is skipped, so a file the re-authoring left
            # alone is still checked
            reauthored += 1
            continue
        if rel in redescribed:
            # the header's prose is not what the rename rewrote: `description`
            # says what a file is for and enters no canonical form (§7), and
            # this one document was re-described after the base — so the round
            # trip is held on everything but its prose, with the current text
            # in place; the exemption is earned (the prose did change) and
            # reaches no other file
            assert migrated["header"]["description"] != after["header"]["description"]
            migrated["header"]["description"] = after["header"]["description"]
            seen_redescribed.add(rel)
        assert json.dumps(migrated, sort_keys=True) == json.dumps(
            after, sort_keys=True
        ), rel
        if "steps" not in before:
            assert format_document(migrated) == path.read_text(), rel
        checked += 1
    # the skipped ones are not lost: the floor holds on the sum so a glob that
    # stops matching still fails
    assert reauthored == reauthored_count
    assert checked + reauthored >= 70 + 12
    assert seen_redescribed == redescribed
    assert retired == RETIRED_COLUMN_FORMS[base]


def test_migrate_carries_windows_templates_and_dotted_ids():
    v2 = {
        "header": {"protocol_version": "2", "description": "axis sites.a.layer here"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": "d", "field": "input"}},
        "method": {
            "sites": {
                "a": {
                    "component": "attention_output",
                    "layer": {"at_once": {"range": [1, 3]}},
                    "names": "a{layer}",
                },
                "s": {"component": "block_output", "layer": {"sweep": [1, 2]}},
                "r": {
                    "component": "block_output",
                    "layer": {"artifact": "x", "key": "k"},
                    "head": 0,
                },
            },
            "writes": {
                "w": {"site": "a", "pos": -1, "do": {"swap": "v"}, "names": "w{layer}"}
            },
            "intervened_models": {
                "m": {
                    "input": "base",
                    "writes": [{"w": {"layer": {"at_once": [1]}}}, "other"],
                }
            },
            "save": [],
        },
    }
    v3 = migrate_document(v2)
    sites = v3["method"]["sites"]
    assert sites["a"] == {
        "component": "attention_output",
        "layers": {"at_once": {"range": [1, 3]}},
        "names": "a{layers}",
    }
    assert sites["s"] == {"component": "block_output", "layers": {"sweep": [1, 2]}}
    assert list(sites["r"]) == ["component", "layers", "head"]  # key order kept
    assert v3["method"]["writes"]["w"]["names"] == "w{layers}"
    assert v3["method"]["intervened_models"]["m"] == {
        "input": "base",
        "reads": [],
        "writes": [{"w": {"layers": {"at_once": [1]}}}, "other"],
    }
    assert v3["header"] == {
        "protocol_version": "4",
        "description": "axis sites.a.layers here",
    }
    workflow = {
        "version": "1",
        "output_dir": "r",
        "steps": {
            "fit": {"set": {"sites.target.layer": {"artifact": "b", "key": "k"}}},
            "best": {
                "inputs": {
                    "emit": {"best_layer": "sites.target.layer"},
                    "layer_column": "sites.c.layer",
                }
            },
            "plot": {
                "inputs": {"x": "sites.target.layer", "columns": {"layer": "int64"}}
            },
        },
    }
    assert needs_migration(workflow)
    out = migrate_document(workflow)
    assert out["steps"]["fit"]["set"] == {
        "sites.target.layers": {"artifact": "b", "key": "k"}
    }
    assert out["steps"]["best"]["inputs"] == {
        "emit": {"best_layer": "sites.target.layers"},
        "layer_column": "sites.c.layers",
    }
    assert out["steps"]["plot"]["inputs"]["columns"] == {"layer": "int64"}  # not a site
    assert not needs_migration(out) and migrate_document(out) == out
    with pytest.raises(ParseError, match="unsupported protocol_version '5'"):
        migrate_document(
            {"header": {"protocol_version": "5"}, "model": {}, "data": {}, "method": {}}
        )


def test_migrate_chains_v1_through_v2_to_v4():
    v3 = v3_base_doc()
    v1 = {
        "version": "1",
        "model": v3["model"],
        "data": v3["data"],
        **_v2_of(v3)["method"],
    }
    assert migrate_document(v1) == base_doc()


def test_migrate_markdown_rewrites_whole_v2_documents_and_stale_workflows_only():
    v2 = _v2_of(v3_base_doc())
    workflow = {
        "version": "1",
        "output_dir": "r",
        "steps": {"s": {"set": {"sites.t.layer": 1}}},
    }
    prose = (
        "```json\n" + json.dumps(v2, indent=2) + "\n```\n\n"
        '```json\n{"target": {"component": "block_output", "layer": 18}}\n```\n\n'
        "```json\n" + json.dumps(workflow) + "\n```\n"
    )
    out = migrate_markdown(prose)
    assert '"protocol_version": "4"' in out and '"layers": [3]' in out
    assert '{"target": {"component": "block_output", "layer": 18}}' in out  # a fragment
    assert '"sites.t.layers": 1' in out
    assert migrate_markdown(out) == out


def test_the_migrate_verb_rewrites_a_v2_file_and_check_is_quiet_on_v4(tmp_path, capsys):
    v2 = _v2_of(v3_base_doc())
    document = tmp_path / "old.json"
    document.write_text(json.dumps(v2))
    assert main(["migrate", "--check", str(document)]) == 1
    assert "would migrate" in capsys.readouterr().out
    assert main(["migrate", str(document)]) == 0
    assert json.loads(document.read_text()) == base_doc()
    assert main(["migrate", "--check", str(document)]) == 0
    assert parse_document(json.loads(document.read_text())).sites["tgt"].layers == (3,)


# --------------------------------------------------------------------------- #
# protocol_version 4 — the reads-first rewrite
# --------------------------------------------------------------------------- #

MIGRATE_V3 = REPO / "tests" / "protocol" / "fixtures" / "migrate_v3"
V3_FIXTURES = ("harvest", "interchange", "path_patching", "das", "at_once_band")


def _v3_to_v4(raw: dict[str, Any]) -> dict[str, Any]:
    from causalab.protocol.migrate import _v3_to_v4 as step  # pyright: ignore[reportPrivateUsage]

    return step(raw)


@pytest.mark.parametrize("name", V3_FIXTURES)
def test_the_v3_to_v4_rewrite_reproduces_each_fixture_pair(name: str) -> None:
    """Five shapes, one per thing the rewrite touches: a pure harvest (the
    un-intervened model is ``original`` on ``base``); an interchange (it is
    ``original_counterfactual``); path patching (two roles, two unwritten
    models); a DAS fit (positional objective, ``eval`` and ``early_stop``);
    an ``at_once`` band whose family read a model lists by its family name,
    beside a model whose write list is swept."""
    v3 = json.loads((MIGRATE_V3 / f"{name}.v3.json").read_text())
    v4_text = (MIGRATE_V3 / f"{name}.v4.json").read_text()
    migrated = _v3_to_v4(v3)
    assert json.dumps(migrated, sort_keys=True) == json.dumps(
        json.loads(v4_text), sort_keys=True
    )
    assert format_document(migrated) == v4_text  # the authoring format too
    assert migrated["header"]["protocol_version"] == "4"
    assert "metrics" not in migrated["method"]
    assert list(migrated["method"])[0] == "intervened_models"
    for read in migrated["method"]["reads"].values():
        assert "model" not in read and "input" not in read
    assert v3 == json.loads((MIGRATE_V3 / f"{name}.v3.json").read_text())  # a copy


def test_the_unwritten_model_is_named_by_the_roles_read_on_it() -> None:
    raw = v3_base_doc()
    v4 = _v3_to_v4(raw)
    assert v4 == base_doc()  # the shared literal is the ancestor, migrated
    models = v4["method"]["intervened_models"]
    assert list(models) == ["original_counterfactual", "patched"]
    assert models["original_counterfactual"] == {
        "input": "counterfactual",
        "reads": ["v_cf"],
    }
    assert models["patched"] == {
        "input": "base",
        "reads": ["logits"],
        "writes": ["patch"],
    }
    # only `base` read on it: the plain name
    harvest = json.loads((MIGRATE_V3 / "harvest.v3.json").read_text())
    assert list(_v3_to_v4(harvest)["method"]["intervened_models"]) == ["original"]
    # an indexed role spells its index
    raw = v3_base_doc()
    raw["data"]["counterfactual"] = [raw["data"]["counterfactual"]] * 2
    raw["method"]["reads"]["v_cf"]["input"] = "counterfactual[1]"
    assert "original_counterfactual_1" in _v3_to_v4(raw)["method"]["intervened_models"]
    # a name the document already declares is refused rather than merged
    raw = v3_base_doc()
    raw["method"]["intervened_models"]["original_counterfactual"] = raw["method"][
        "intervened_models"
    ].pop("patched")
    raw["method"]["reads"]["logits"]["model"] = "original_counterfactual"
    raw["method"]["save"][0]["model"] = "original_counterfactual"
    with pytest.raises(ParseError, match="already declares"):
        _v3_to_v4(raw)


def test_metrics_dissolve_into_the_entries_that_consume_them() -> None:
    raw = v3_base_doc()
    raw["method"]["reads"]["logits_orig"] = {
        "site": "lm_head",
        "pos": -1,
        "model": "original",
        "input": "base",
    }
    raw["method"]["metrics"]["drift"] = {
        "kind": "kl",
        "of": "logits",
        "target": "logits_orig",
    }
    raw["method"]["save"].append(
        {"value": "drift", "model": "patched", "input": "base", "file_path": "kl.json"}
    )
    raw["method"]["save"].append(
        {
            "value": "v_cf",
            "model": "original",
            "input": "counterfactual",
            "file_path": "v_cf.safetensors",
            "reduce": "mean",
        }
    )
    v4 = _v3_to_v4(raw)
    save = v4["method"]["save"]
    assert save[0] == {
        "read": "logits",
        "model": "patched",
        "aggregation": {
            "kind": "logit_diff",
            "a": "cf_answer",
            "b": "base_answer",
        },
        "file_path": "ld.json",
    }
    # inside a save a kl target is {"read", "model"}, even bound to one model
    assert save[1]["aggregation"] == {
        "kind": "kl",
        "target": {"read": "logits_orig", "model": "original_base"},
    }
    assert save[2] == {
        "read": "v_cf",
        "model": "original_counterfactual",
        "file_path": "v_cf.safetensors",
        "reduce": "mean",
    }
    # two roles on the un-intervened model, so `original_base` joins
    assert set(v4["method"]["intervened_models"]) == {
        "original_counterfactual",
        "original_base",
        "patched",
    }


def test_a_metric_nothing_consumes_is_refused() -> None:
    raw = v3_base_doc()
    raw["method"]["metrics"]["spare"] = dict(raw["method"]["metrics"]["ld"])
    with pytest.raises(ParseError, match=r"\['spare'\] are consumed by no"):
        _v3_to_v4(raw)


def test_the_named_objective_form_inlines_its_aggregation() -> None:
    raw = v3_base_doc()
    raw["method"]["featurizers"] = {"gate": {"kind": "gate"}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "gate"
    raw["method"]["writes"]["patch"]["featurizer"] = "gate"
    raw["method"]["metrics"]["ce"] = {
        "kind": "cross_entropy",
        "of": "logits",
        "target": "label",
    }
    raw["method"]["train"] = {
        "objective": {
            "ce": {"weight": 1.0, "metric": "ce"},
            "sparse": {"weight": 0.01, "l1": "gate"},
        },
        "params": ["gate"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
        "seed": 0,
    }
    train = _v3_to_v4(raw)["method"]["train"]
    assert train["objective"]["ce"] == {
        "weight": 1.0,
        "read": "logits",
        "model": "patched",
        "aggregation": {
            "kind": "cross_entropy",
            "target": "label",
        },
    }
    assert train["objective"]["sparse"] == {"weight": 0.01, "l1": "gate"}


def _v3_fit(objective: Any, eval_metrics: list[str]) -> dict[str, Any]:
    """`v3_base_doc` as a protocol-3 DAS fit: a ``ce`` metric beside ``ld``,
    both saved, under ``objective`` and an eval of ``eval_metrics``."""
    raw = v3_base_doc()
    method = raw["method"]
    method["featurizers"] = {"rot": {"kind": "subspace", "k": 8}}
    method["reads"]["v_cf"]["featurizer"] = "rot"
    method["writes"]["patch"]["featurizer"] = "rot"
    method["metrics"]["ce"] = {
        "kind": "cross_entropy",
        "of": "logits",
        "target": "label",
    }
    method["train"] = {
        "objective": objective,
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 1e-3},
        "steps": {"epochs": 1},
        "batch": {"pairs": 2},
        "eval": {"every": {"epochs": 1}, "split": "eval", "metrics": eval_metrics},
        "seed": 0,
    }
    method["save"] += [
        {"value": "ce", "model": "patched", "input": "base", "file_path": "ce.json"},
        {"value": "rot", "site": "tgt", "file_path": "rot.safetensors"},
    ]
    return raw


def test_a_save_of_a_train_metric_names_the_term() -> None:
    v4 = _v3_to_v4(_v3_fit([[1.0, "ce"]], ["ld"]))
    method = v4["method"]
    assert method["save"][:2] == [
        {"train": "ld", "file_path": "ld.json"},
        {"train": "ce", "file_path": "ce.json"},
    ]
    # the positional objective takes the named form, named by its metric
    assert method["train"]["objective"] == {
        "ce": {
            "weight": 1.0,
            "read": "logits",
            "model": "patched",
            "aggregation": {"kind": "cross_entropy", "target": "label"},
        }
    }
    # and the reference is the entry the v3 save spelled
    twin = copy.deepcopy(v4)
    twin["method"]["save"][1] = saved(
        "logits", "patched", "ce.json", {"kind": "cross_entropy", "target": "label"}
    )
    assert parse_document(in_order(v4)).save == parse_document(in_order(twin)).save


def test_a_positional_regularizer_is_named_by_its_kind() -> None:
    v4 = _v3_to_v4(_v3_fit([[1.0, "ce"], [0.01, {"l2": "rot"}]], ["ld"]))
    objective = v4["method"]["train"]["objective"]
    assert list(objective) == ["ce", "l2"]
    assert objective["l2"] == {"weight": 0.01, "l2": "rot"}
    assert v4["method"]["save"][1] == {"train": "ce", "file_path": "ce.json"}


def test_a_named_v3_term_is_named_by_its_term_name() -> None:
    v4 = _v3_to_v4(_v3_fit({"fit": {"weight": 1.0, "metric": "ce"}}, ["ld"]))
    assert v4["method"]["save"][1] == {"train": "fit", "file_path": "ce.json"}


@pytest.mark.parametrize(
    "objective, eval_metrics",
    [
        # two regularizers of one kind have no distinct names to take
        ([[1.0, "ce"], [0.01, {"l2": "rot"}], [0.1, {"l2": "rot"}]], ["ld"]),
        # `ce` would name the objective term and the eval label at once
        ([[1.0, "ce"]], ["ld", "ce"]),
    ],
    ids=["same_kind_twice", "both_namespaces"],
)
def test_a_metric_with_no_one_term_keeps_its_save_inline(
    objective: Any, eval_metrics: list[str]
) -> None:
    method = _v3_to_v4(_v3_fit(objective, eval_metrics))["method"]
    assert isinstance(method["train"]["objective"], list)
    assert method["save"][1]["read"] == "logits"
    assert method["save"][1]["aggregation"]["kind"] == "cross_entropy"


def test_a_swept_metric_keeps_its_save_inline() -> None:
    raw = _v3_fit([[1.0, "ce"]], ["ld"])
    raw["method"]["metrics"]["ld"]["a"] = {"sweep": ["cf_answer", "label"]}
    method = _v3_to_v4(raw)["method"]
    assert method["save"][0]["aggregation"]["a"] == {"sweep": ["cf_answer", "label"]}
    assert method["save"][1] == {"train": "ce", "file_path": "ce.json"}


def test_a_dotted_id_of_a_referenced_metric_names_the_term() -> None:
    raw = _v3_fit([[1.0, "ce"]], ["ld"])
    raw["header"]["description"] = "reads metrics.ld.a"
    v4 = _v3_to_v4(raw)
    assert v4["header"]["description"] == (
        "reads train.eval.aggregations.ld.aggregation.a"
    )


def test_dotted_metric_ids_are_respelled_onto_their_one_save_entry() -> None:
    raw = v3_base_doc()
    raw["method"]["metrics"]["top"] = {
        "kind": "top_k",
        "of": "logits",
        "k": 3,
        "by": "prob",
    }
    raw["method"]["save"].append(
        {"value": "top", "model": "patched", "input": "base", "file_path": "top.json"}
    )
    raw["header"]["description"] = "sweeps metrics.top.k and reads metrics.ld.a"
    v4 = _v3_to_v4(raw)
    assert v4["header"]["description"] == (
        "sweeps save[1].aggregation.k and reads save[0].aggregation.a"
    )
    # the same metric saved twice has no single spelling
    raw["method"]["save"].append(
        {"value": "top", "model": "patched", "input": "base", "file_path": "top2.json"}
    )
    with pytest.raises(
        ParseError, match=r"save\[1\].aggregation.k.*save\[2\].aggregation.k"
    ):
        _v3_to_v4(raw)


def test_the_v1_and_v2_chains_reach_the_v4_shape() -> None:
    from causalab.protocol.migrate import (
        _v1_to_v2,  # pyright: ignore[reportPrivateUsage]
        _v2_to_v3,  # pyright: ignore[reportPrivateUsage]
    )

    v3 = v3_base_doc()
    v2 = _v2_of(v3)
    v1 = {"version": "1", "model": v3["model"], "data": v3["data"], **v2["method"]}
    expected = _v3_to_v4(v3)
    assert expected == base_doc()
    assert _v3_to_v4(_v2_to_v3(v2)) == expected
    assert _v3_to_v4(_v2_to_v3(_v1_to_v2(v1))) == expected
    assert (
        _v2_to_v3(v2)["header"]["protocol_version"] == "3"
    )  # the literal, not the current
