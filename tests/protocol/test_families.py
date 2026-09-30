"""``at_once`` families (spec §3.1): the sugar is sugar, and what it refuses.

The load-bearing test is the first one. The shipped `attention_band_patch`
preset was 301 hand-written lines and is now 44; the claim that those are the
*same experiment* is not a matter of reading them side by side — the
hand-written form is frozen as a fixture, and the preset's expansion is compared
against it entry for entry and digest for digest.

Everything else here is rule 28: a family is a way to declare a table, so every
refusal is a case where the table it denotes would not be the one the author
meant. Two of them (a `sweep` wrapper on a family entry, a window naming an
index the family lacks) are the ones that would otherwise have produced a
plausible number rather than an error.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.lowering import expand_families, has_families
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import plan_point
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import parse_document
from causalab.protocol.lowering import AT_ONCE_KEY, SWEEP_KEY

from tests.protocol._docs import (
    LOGIT_DIFF,
    UNWRITTEN,
    base_doc,
    in_order,
    saved,
    term,
)
from tests.protocol._env import FIXTURES, TASKS_ROOT, steps_of
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PRESET = PROTOCOLS_DIR / "attention_band_patch.json"
#: The preset as it shipped, hand-written, before §3.1 existed.
HANDWRITTEN = FIXTURES / "band_patch_handwritten.json"
#: Its digest against the fixture environment, frozen the day the sugar landed.
#: The one number in this file that must never move for a *mechanism* reason —
#: it is the whole of the claim that the sugar costs no digest. (The preset's
#: own digest did move, because its prose was rewritten too, which is why the
#: comparison below substitutes this document's `description` before hashing.)
#: It has moved once, for data identity: the shipped tables renamed the
#: weekdays ref (`natural_domains_arithmetic/data/weekdays`), and this fixture
#: followed the preset. With the old ref the previous pin, 2d4a58a349d63d5d,
#: still reproduces under this code. Re-pinned once more under protocol v2:
#: the four-group canonical form moves every digest, the one kind of
#: change spec §7 lets move a pin. And once more under protocol v3: the
#: ten sites spell `layers: [n]`, a band, and their bytes moved with the field.
#: And once more for data identity: the shipped weekdays table lost
#: its `scoring_digest` column, so `data.*.digest` moved and this followed.
#: And again for protocol v4 (reads-first): the models list their reads —
#: landed on the retired ``token_form``, so the literal lost the key too.
#: And once more for the method library's retarget (Llama-3.1-8B, bands over
#: L10-L19 -> Qwen2.5-7B, bands over L9-L16): the fixture followed the preset
#: again, and their digests still agree (the retarget's v3 pin was
#: ``40149fa1b44a26e2``; this is the same fixture migrated to v4).
HANDWRITTEN_DIGEST = "50e756d14e96a6ef"


@pytest.fixture(scope="module")
def env() -> ResolutionEnv:
    # The shipped preset names the shipped weekdays table; the documents built
    # here name the 4-row fixture. Layered as the CLI has it (``build_env``), so
    # both load in one env.
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=FIXTURES / "artifacts"),
    )


def _norm(node: object) -> object:
    """Entry order carries no meaning (§1) and the canonical form sorts keys
    (§7), so the comparison that matters is order-insensitive."""
    if isinstance(node, dict):
        return {key: _norm(value) for key, value in sorted(node.items())}
    if isinstance(node, list):
        return [_norm(value) for value in node]
    return node


def expect_family_refusal(raw: dict[str, Any], match: str) -> ValidationError:
    with pytest.raises(ValidationError, match=match) as err:
        expand_families(raw)
    assert err.value.rule_id == "family_wrappers"
    assert err.value.code == "V28"
    return err.value


def band_doc() -> dict[str, Any]:
    """A three-layer band on gpt2, the smallest document with every part of
    §3.1 in it: a site family, a read and a write that fan out off it, and a
    windowed write list."""
    doc = base_doc()
    doc["method"]["positions"] = {"tap": {"index": -1}}
    doc["method"]["sites"] = {
        "a": {
            "component": "block_output",
            "layers": {AT_ONCE_KEY: {"range": [3, 6]}},
            "names": "a{layers}",
        },
        "lm_head": {"component": "lm_head"},
    }
    doc["method"]["reads"] = {
        "v": {"site": "a", "pos": "tap", "names": "v{layers}"},
        "logits": {"site": "lm_head", "pos": -1},
    }
    doc["method"]["writes"] = {
        "w": {"site": "a", "pos": "tap", "do": {"swap": "v"}, "names": "w{layers}"}
    }
    doc["method"]["intervened_models"] = {
        UNWRITTEN: {"input": "counterfactual", "reads": ["v"]},
        "band": {
            "input": "base",
            "reads": ["logits"],
            "writes": [{"w": {"layers": {"at_once": {"range": [3, 6]}}}}],
        },
    }
    doc["method"]["save"] = [saved("logits", "band", "ld.json", dict(LOGIT_DIFF))]
    return in_order(doc)


# --------------------------------------------------------------------------- #
# the sugar is sugar
# --------------------------------------------------------------------------- #


def test_the_preset_expands_to_the_form_it_replaced(env) -> None:
    """301 lines became 44, and this is the whole argument that the two are one
    experiment: the expansion equals the frozen hand-written document entry for
    entry, and — carrying that document's own prose, since a `description` is
    inside a v1 digest — hashes to the same bytes."""
    preset = json.loads(PRESET.read_text())
    handwritten = json.loads(HANDWRITTEN.read_text())

    expanded = expand_families(preset)
    # the header is the file's, not the experiment's (§1): carry the frozen one
    assert _norm({**expanded, "header": handwritten["header"]}) == _norm(handwritten)

    as_shipped = digest(canonicalize(handwritten, env))
    as_sugar = digest(canonicalize({**expanded, "header": handwritten["header"]}, env))
    assert as_sugar == as_shipped
    assert as_shipped.startswith(HANDWRITTEN_DIGEST)


def test_the_preset_still_plans_one_shared_harvest_for_every_band(env) -> None:
    """The cost claim `test_band_patch_run.py` makes about the preset, asserted
    here too because it is what the sugar must not disturb: one un-intervened
    harvest of all ten taps, then one forward per band."""
    loaded = compile_protocol(PRESET, env=env)
    assert len(steps_of(loaded, env).points) == 1
    assert plan_point(steps_of(loaded, env).documents[0]).num_forwards == 4
    doc = steps_of(loaded, env).documents[0]
    assert len(doc.writes) == 8  # one per layer, not one per (layer, band)
    narrow = set(doc.intervened_models["band4_L9"].writes) | set(
        doc.intervened_models["band4_L13"].writes
    )
    assert set(doc.intervened_models["band8_L9"].writes) == narrow


def test_a_document_with_no_family_is_untouched() -> None:
    """Every document written before §3.1 is one of these, so the new stage has
    to be provably inert on them — not merely correct."""
    doc = base_doc()
    assert not has_families(doc)
    assert expand_families(doc) == doc


def test_expansion_is_idempotent() -> None:
    """The result carries no `at_once`, so a belt-and-braces second call — a
    caller that cannot know whether the tree came from `load` — changes
    nothing."""
    once = expand_families(band_doc())
    assert not has_families(once)
    assert expand_families(once) == once


def test_members_are_named_by_the_convention_when_no_template_is_given() -> None:
    """`a[layers=3]` is §3's derived-name spelling, so a family member and a
    swept name read alike."""
    doc = band_doc()
    for table, entry in (("sites", "a"), ("reads", "v"), ("writes", "w")):
        del doc["method"][table][entry]["names"]
    doc["method"]["intervened_models"]["band"]["writes"] = ["w"]

    expanded = expand_families(doc)
    assert list(expanded["method"]["sites"]) == [
        "a[layers=3]",
        "a[layers=4]",
        "a[layers=5]",
        "lm_head",
    ]
    assert expanded["method"]["writes"]["w[layers=4]"] == {
        "site": "a[layers=4]",
        "pos": "tap",
        "do": {"swap": "v[layers=4]"},
    }


def test_a_bare_family_name_is_every_member_and_a_window_is_the_interval() -> None:
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = ["w"]
    assert expand_families(doc)["method"]["intervened_models"]["band"]["writes"] == [
        "w3",
        "w4",
        "w5",
    ]

    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": {"range": [4, 6]}}}}
    ]
    assert expand_families(doc)["method"]["intervened_models"]["band"]["writes"] == [
        "w4",
        "w5",
    ]

    doc = band_doc()  # the axis grammar, so a step is a stride through the band
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": {"range": [3, 6, 2]}}}}
    ]
    assert expand_families(doc)["method"]["intervened_models"]["band"]["writes"] == [
        "w3",
        "w5",
    ]

    doc = band_doc()  # and a list selects too, and may be non-contiguous
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": [5, 3]}}}
    ]
    assert expand_families(doc)["method"]["intervened_models"]["band"]["writes"] == [
        "w5",
        "w3",
    ]


def test_a_position_family_fans_out_through_pos(env) -> None:
    """`pos` is a name field, so the multi-position patch — three writes at
    disjoint positions of one site, the shape `multi_position_patch.json`
    spells out by hand — is a family too."""
    doc = base_doc()
    doc["method"]["positions"] = {
        "p": {"index": {AT_ONCE_KEY: [-4, -3, -2]}, "names": "p{index}"}
    }
    doc["method"]["reads"] = {
        "v": {"site": "tgt", "pos": "p", "names": "v{index}"},
        "logits": {"site": "lm_head", "pos": -1},
    }
    doc["method"]["writes"] = {
        "at": {"site": "tgt", "pos": "p", "do": {"swap": "v"}, "names": "at{index}"}
    }
    doc["method"]["intervened_models"] = {
        UNWRITTEN: {"input": "counterfactual", "reads": ["v"]},
        "patched": {"input": "base", "reads": ["logits"], "writes": ["at"]},
    }
    doc = in_order(doc)

    expanded = expand_families(doc)
    assert list(expanded["method"]["positions"]) == ["p-4", "p-3", "p-2"]
    assert expanded["method"]["intervened_models"]["patched"]["writes"] == [
        "at-4",
        "at-3",
        "at-2",
    ]
    compile_protocol(
        expanded, env=env
    )  # three disjoint absolute writes are legal (§2.8 rule 8)


def test_a_family_and_a_sweep_on_a_different_entry_compose(env) -> None:
    """Families expand before axes are found, so the two mechanisms meet the
    way §3.1 says: members inside each point, points across the sweep."""
    doc = band_doc()
    doc["method"]["reads"]["logits"]["pos"] = {SWEEP_KEY: [-1, -2]}

    loaded = compile_protocol(doc, env=env)
    assert len(steps_of(loaded, env).points) == 2
    for point in steps_of(loaded, env).documents:
        assert len(point.writes) == 3  # the family is inside every point


# --------------------------------------------------------------------------- #
# rule 28 — what a family refuses
# --------------------------------------------------------------------------- #


def test_rule_28_two_axes_on_one_entry() -> None:
    doc = band_doc()
    doc["method"]["sites"]["a"]["head"] = {AT_ONCE_KEY: [0, 1]}
    expect_family_refusal(doc, "one axis per entry")


def test_rule_28_an_entry_referencing_two_different_axes() -> None:
    """A layer family crossed with a head family: which pairs are co-resident
    in one forward is exactly what a cross product answers wrongly."""
    doc = band_doc()
    doc["method"]["sites"]["h"] = {
        "component": "attention_premix",
        "layers": [3],
        "head": {AT_ONCE_KEY: {"range": [0, 2]}},
        "names": "h{head}",
    }
    doc["method"]["writes"]["w2"] = {
        "site": "h",
        "pos": "tap",
        "do": {"swap": "v"},
        "names": "w2_{head}",
    }
    err = expect_family_refusal(doc, "different axes")
    assert "Sweep the second axis" in str(err)


def test_rule_28_a_sweep_wrapper_on_a_family_entry() -> None:
    """The trap worth a rule of its own: expansion would copy the wrapper onto
    every member, and a sweep axis is identified by its path (§3), so three
    members would be three independent axes and their cross product — eight
    points where the author meant two."""
    doc = band_doc()
    doc["method"]["sites"]["a"]["component"] = {
        SWEEP_KEY: ["block_output", "mlp_output"]
    }
    err = expect_family_refusal(doc, "one axis per member")
    assert "cross product" in str(err)


def test_rule_28_a_window_naming_an_index_the_family_lacks() -> None:
    """A band that resolves to fewer writes than the interval it declares is
    the silent bug the window form exists to refuse — before §3.1 the same
    mistake was a legal document reporting a width-10 band that patched nine
    layers."""
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": {"range": [3, 7]}}}}
    ]
    err = expect_family_refusal(doc, "does not carry")
    assert "layers=6" in str(err) and "3..5" in str(err)


def test_rule_28_a_names_template_that_does_not_name_the_axis() -> None:
    doc = band_doc()
    doc["method"]["sites"]["a"]["names"] = "a_fixed"
    expect_family_refusal(doc, "names exactly the axis field")


def test_rule_28_names_without_an_axis() -> None:
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["names"] = "tgt{layers}"
    expect_family_refusal(doc, "declares no 'at_once' axis")


def test_rule_28_repeated_values_are_not_an_index() -> None:
    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: [3, 4, 3]}
    expect_family_refusal(doc, "must be distinct")


def test_rule_28_a_member_name_that_collides_with_a_declared_entry() -> None:
    doc = band_doc()
    doc["method"]["sites"]["a"]["names"] = "lm_hea{layers}d"  # lm_head at layer 3
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: [""]}
    expect_family_refusal(doc, "already declares")


def test_rule_28_a_member_name_that_is_reserved() -> None:
    doc = band_doc()
    doc["method"]["sites"]["a"]["names"] = "bas{layers}"
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: ["e"]}
    expect_family_refusal(doc, "reserved")


def test_rule_28_a_metric_over_a_family() -> None:
    """Refused by name, and the message says why it is a change rather than a
    line: a saved family needs a rule for per-member `file_path`."""
    doc = band_doc()
    # the aggregation over the family, on the model that lists it
    doc["method"]["save"][0].update(read="v", model=UNWRITTEN)
    err = expect_family_refusal(doc, "do not fan out")
    assert "file_path" in str(err)


def test_rule_28_a_save_entry_over_a_family() -> None:
    doc = band_doc()
    doc["method"]["save"] = [saved("v", UNWRITTEN, "v.safetensors")]
    expect_family_refusal(doc, "do not fan out")


def test_rule_28_a_family_of_intervened_models() -> None:
    """A wrapper on the model's own field is an axis over models, and is told
    apart from a window — the same keyword inside a write-list item — by
    position, so a model that carries both is still refused for the former."""
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["input"] = {AT_ONCE_KEY: ["base"]}
    expect_family_refusal(doc, "not in this version")

    doc = band_doc()  # a string write item wrapped is over models too, not a window
    doc["method"]["intervened_models"]["band"]["writes"] = [{AT_ONCE_KEY: ["w3", "w4"]}]
    expect_family_refusal(doc, "not in this version")


def test_rule_28_a_swept_write_list_holding_a_window() -> None:
    """The swept-write-list refusal recognises a family named by key, not only
    by value — a window mentions its family as the key of a one-key object."""
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = {
        "sweep": [[{"w": {"layers": {"at_once": {"range": [3, 5]}}}}], ["w5"]]
    }
    expect_family_refusal(doc, "swept and names the family")


def test_rule_28_a_wrapper_that_holds_more_than_its_axis() -> None:
    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: [3, 4], "stride": 2}
    expect_family_refusal(doc, "nothing but the axis")


def test_rule_28_a_wrapper_with_no_name_identity() -> None:
    """An `at_once` somewhere no name can be derived for its members — here
    inside a `do` block rather than on a field of a named entry."""
    doc = band_doc()
    doc["method"]["writes"]["w"]["do"] = {
        "add_scaled": {"op": "v", "alpha": {AT_ONCE_KEY: [1, 2]}}
    }
    expect_family_refusal(doc, "no name identity")


def test_rule_28_selecting_from_something_that_is_not_a_family() -> None:
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"nope": {"layers": {"at_once": {"range": [3, 4]}}}}
    ]
    expect_family_refusal(doc, "is not a family")


def test_rule_28_a_window_on_the_wrong_field() -> None:
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"head": {"at_once": {"range": [0, 2]}}}}
    ]
    expect_family_refusal(doc, "indexed by 'layers'")


@pytest.mark.parametrize(
    "selector",
    [
        [3, 5],  # a bare list
        {"range": [3, 5]},  # a bare range
        {"span": [3, 5]},  # the interval keyword a first draft borrowed from §2.3
    ],
    ids=["list", "range", "span"],
)
def test_rule_28_a_window_is_spelled_at_once(selector) -> None:
    """One word per object (§11.1): a window is written the way the axis it
    selects from was, `{"at_once": ...}`. A first draft accepted an interval
    `{"span": [a, b]}` and, by falling through to the axis validator, a bare
    list beside it — two dialects of one selection, and the bare form's
    refusals named a keyword the author never wrote."""
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": [selector]}}
    ]
    expect_family_refusal(doc, "spelled like the axis it selects from")


def test_rule_28_a_window_with_more_than_its_axis() -> None:
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": {"range": [3, 5]}, "step": 2}}}
    ]
    expect_family_refusal(doc, "nothing but its at_once axis")


def test_rule_28_one_write_named_twice() -> None:
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": {"range": [3, 5]}}}},
        "w4",
    ]
    err = expect_family_refusal(doc, "listed twice")
    assert "overlapping bands" in str(err)


def test_rule_28_the_value_grammar_is_sweeps_own() -> None:
    """One grammar, one implementation (`sweep.axis_values`) — so a malformed
    `at_once` fails the same way a malformed `sweep` does, only citing rule 28
    and naming the other keyword."""
    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: {"range": [3, 6, 0]}}
    expect_family_refusal(doc, "at_once range step must be non-zero")

    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: []}
    expect_family_refusal(doc, "at least one value")

    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: [{SWEEP_KEY: [3, 4]}]}
    expect_family_refusal(doc, "nested axis wrappers")


def test_a_refusal_survives_the_round_trip_through_load(env) -> None:
    """The loader is where §3.1 actually runs, so the wiring is under test too
    — not only `expand_families` in isolation."""
    doc = copy.deepcopy(band_doc())
    doc["method"]["intervened_models"]["band"]["writes"] = [
        {"w": {"layers": {"at_once": {"range": [3, 9]}}}}
    ]
    with pytest.raises(ValidationError) as err:
        compile_protocol(doc, env=env)
    assert err.value.rule_id == "family_wrappers"


def test_an_entry_merely_named_at_once_is_a_name_not_a_wrapper() -> None:
    """§1 reserves four names and `at_once` is not one of them, so a table key
    spelled that way is an entry. Reading it as a wrapper refused a valid
    document with a message about something the author never wrote."""
    doc = base_doc()
    doc["method"]["params"] = {AT_ONCE_KEY: {"file_path": "constant.safetensors"}}
    doc = in_order(doc)
    assert not has_families(doc)
    assert expand_families(doc) == doc


# --------------------------------------------------------------------------- #
# the rule-28 review — every hole it found, closed and pinned
# --------------------------------------------------------------------------- #


def test_rule_28_an_empty_window_names_no_write() -> None:
    """`{"range": [15, 10]}` selected nothing, and nothing downstream minded: a
    band with no writes compiles as the un-intervened model and its metric
    reports no effect. A window is an axis, so the empty window is the empty
    axis — refused by the grammar both keywords share, under this rule."""
    for bounds in ([5, 3], [4, 4]):
        doc = band_doc()
        doc["method"]["intervened_models"]["band"]["writes"] = [
            {"w": {"layers": {"at_once": {"range": bounds}}}}
        ]
        expect_family_refusal(doc, "at least one value")


def test_rule_28_a_sweep_nested_inside_a_family_entry() -> None:
    """The guard scanned the entry's own fields only, so a `sweep` inside `do`
    was copied to every member — ten members of the shipped preset is 2**10
    points, under the point cap, so nothing else would have caught it."""
    doc = band_doc()
    doc["method"]["writes"]["w"]["do"] = {
        "add_scaled": {"op": "v", "alpha": {SWEEP_KEY: [0.5, 1.0]}}
    }
    err = expect_family_refusal(doc, "also sweeps")
    assert "writes.w.do.add_scaled.alpha" in str(err)


def test_rule_28_a_declaring_entry_that_references_another_axis() -> None:
    """The one-axis rule ran only for entries that *inherited* their axis. For
    one that declares its own, a foreign reference is substituted by value — so
    coinciding values silently give the diagonal of a cross product, and
    non-coinciding ones a dangling name for rule 4 to report as something
    else."""
    doc = band_doc()
    # `v` declares its own axis on `pos` *and* references the site family `a`,
    # which is on `layer` — so it reaches the declaring branch, not the
    # inheriting one that was already guarded. `w` is pointed at a plain read
    # so that it does not inherit from both and refuse first, which is how this
    # hole stayed hidden.
    doc["method"]["reads"]["plain"] = {"site": "lm_head", "pos": -1}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"].append("plain")
    doc["method"]["writes"]["w"]["do"] = {"swap": "plain"}
    doc["method"]["reads"]["v"]["pos"] = {AT_ONCE_KEY: [-2, -1]}
    doc["method"]["reads"]["v"]["names"] = "v{pos}"
    err = expect_family_refusal(doc, "references a family on another")
    assert "Sweep the second axis" in str(err)


def test_a_featurizer_composition_list_fans_out() -> None:
    """`featurizer` is written as a name or as a list of names (§2.5). Whether
    a family works must not depend on which spelling the author picked.

    A tree assertion on purpose: two members writing one address is rule 8's to
    refuse, and what is under test here is only that the *list* spelling of the
    reference is rewritten at all."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": {AT_ONCE_KEY: [2, 4]}, "names": "rot{k}"},
        "std": {"kind": "standardize"},
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = ["rot", "std"]

    expanded = expand_families(in_order(doc))
    assert list(expanded["method"]["featurizers"]) == ["rot2", "rot4", "std"]
    assert expanded["method"]["reads"]["v_cf[k=2]"]["featurizer"] == ["rot2", "std"]
    assert expanded["method"]["writes"]["patch[k=4]"]["do"]["swap"] == "v_cf[k=4]"


def test_rule_28_a_train_block_over_a_family() -> None:
    """`train` joins `metrics` and `save` in the refusal: its `params` list is
    a name list, so a fit over a family is the shape most likely to look as
    though it had worked."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": {AT_ONCE_KEY: [2, 4]}, "names": "rot{k}"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    doc["method"]["train"] = {"params": ["rot"], "steps": 1}
    expect_family_refusal(in_order(doc), "do not fan out")


def test_rule_28_a_family_larger_than_the_member_bound() -> None:
    """A sweep's million-value axis is safe because the point cap refuses the
    expansion afterwards. A family materializes entries directly, so it needs
    its own bound — and the message says what to reach for instead."""
    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: {"range": [0, 2000]}}
    err = expect_family_refusal(doc, "over the bound of")
    assert "the answer is a sweep" in str(err)


def test_an_intervened_model_with_no_writes_keeps_its_own_message() -> None:
    """Expansion used to inject `writes: None`, which took away `_parse_im`'s
    "an intervened_model needs 'writes'" — a worse message for the same
    mistake."""
    doc = band_doc()
    del doc["method"]["intervened_models"]["band"]["writes"]
    expanded = expand_families(doc)
    assert "writes" not in expanded["method"]["intervened_models"]["band"]
    # under protocol 4 a model without writes is the un-intervened model on
    # its input: it parses, and the family stage added nothing to it
    assert parse_document(expanded).is_unwritten("band")


def test_rule_28_names_with_no_axis_in_a_document_that_has_families() -> None:
    """The orphan-`names` check ran only on family-free documents, so it looked
    everywhere except where the typo is likeliest."""
    doc = band_doc()
    doc["method"]["sites"]["lm_head"]["names"] = "lm_head{layers}"
    expect_family_refusal(doc, "declares no 'at_once' axis")


def test_a_names_key_deeper_than_the_entry_is_left_alone() -> None:
    """`names` was stripped at every recursion depth. Nothing authored one
    below an entry yet — a regularizer's `names` list is the shape that would —
    so this pins the fix rather than a bug that shipped."""
    doc = band_doc()
    doc["method"]["writes"]["w"]["do"] = {"swap": "v", "names": ["not-a-template"]}
    expanded = expand_families(doc)
    assert expanded["method"]["writes"]["w3"]["do"]["names"] == ["not-a-template"]


def test_default_member_names_load_end_to_end(env) -> None:
    """Every `compile_protocol()` test supplied a `names` template, so the *documented
    default* was only ever asserted at the tree level. `a[layers=3]` has to
    survive the parser, the checklist and the canonical form too."""
    doc = band_doc()
    for table, entry in (("sites", "a"), ("reads", "v"), ("writes", "w")):
        del doc["method"][table][entry]["names"]
    doc["method"]["intervened_models"]["band"]["writes"] = ["w"]

    loaded = compile_protocol(doc, env=env)
    point = steps_of(loaded, env).documents[0]
    assert set(point.intervened_models["band"].writes) == {
        "w[layers=3]",
        "w[layers=4]",
        "w[layers=5]",
    }
    assert point.writes["w[layers=4]"].site == "a[layers=4]"
    assert loaded.digests.document  # canonicalized and hashed with those names


#: base_doc's one table, as an objective term (§2.11).
LD = term("logits", "patched", dict(LOGIT_DIFF))


def _fit_doc() -> dict[str, Any]:
    """base_doc with a featurizer family over `k`, so a `train` block has a
    family to reference."""
    doc = base_doc()
    doc["method"]["featurizers"] = {
        "rot": {"kind": "subspace", "k": {AT_ONCE_KEY: [2, 4]}, "names": "rot{k}"}
    }
    doc["method"]["reads"]["v_cf"]["featurizer"] = "rot"
    return doc


def test_rule_28_a_regularizer_over_a_family_in_either_objective_form() -> None:
    """§2.11's regularizer is keyed by its *kind* — `{"l1": ["rot"]}` — never by
    a `names` key, and in the positional form that mapping sits inside a
    `[weight, …]` list inside the objective list. Scanning for `names` refused
    `train.params` and missed both objective shapes, which then fell through to
    a rule-12 dangling reference on a featurizer that had just been expanded
    away."""
    positional = _fit_doc()
    positional["method"]["train"] = {
        "objective": [[1.0, LD], [0.01, {"l1": ["rot"]}]],
        "epochs": 1,
    }
    expect_family_refusal(in_order(positional), "do not fan out")

    keyed = _fit_doc()
    keyed["method"]["train"] = {"objective": {"l2": ["rot"]}, "steps": 1}
    expect_family_refusal(in_order(keyed), "do not fan out")


def test_rule_28_an_explicit_at_once_list_is_bounded_before_it_is_counted() -> None:
    """`axis_values` caps the `{"range": …}` form and nothing caps a list, and
    the bound used to be checked in a pass *after* distinctness — so a 200k
    element list with one repeat was counted first."""
    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: list(range(2000)) + [0]}
    err = expect_family_refusal(doc, "over the bound of")
    assert "the answer is a sweep" in str(err)


def test_rule_28_a_swept_write_list_that_names_a_family() -> None:
    """A whole write list may be swept (`_parse_im` wraps it), which stays
    legal — but a *family name* inside it would have to resolve per point, and
    families materialize before axes are found. Refused by name, like every
    other shape §3.1 leaves for later."""
    doc = band_doc()
    doc["method"]["writes"]["plain"] = {
        "site": "lm_head",
        "pos": -1,
        "do": {"swap": "v3"},
    }
    doc["method"]["intervened_models"]["band"]["writes"] = {
        SWEEP_KEY: [["w"], ["plain"]]
    }
    err = expect_family_refusal(doc, "swept and names the family")
    assert "sweep something other than the write list" in str(err)


def test_a_swept_write_list_of_plain_names_is_left_alone() -> None:
    """The other half of that refusal — the shape it must not break. Member
    names are fine: they are what the family resolves *to*, so nothing is left
    for a point to decide."""
    doc = band_doc()
    doc["method"]["intervened_models"]["band"]["writes"] = {SWEEP_KEY: [["w3"], ["w4"]]}
    expanded = expand_families(doc)
    assert expanded["method"]["intervened_models"]["band"]["writes"] == {
        SWEEP_KEY: [["w3"], ["w4"]]
    }


def _train_block(**overrides: Any) -> dict[str, Any]:
    """A §2.11 `train` block that actually parses — the five mandatory fields
    from the corpus's own DAS fit, with the reference under test substituted
    in. Without this the blocks were shapes the parser would have rejected
    anyway, so the refusal could not be told from a schema error."""
    block: dict[str, Any] = {
        "objective": [[1.0, LD]],
        "params": ["rot"],
        "optimizer": {"name": "adamw", "lr": 0.001},
        "steps": {"epochs": 1},
        "batch": {"pairs": 4},
    }
    block.update(overrides)
    return block


def test_rule_28_every_spelling_of_a_train_reference_to_a_family() -> None:
    """The refusal reached two of four spellings. A regularizer's value may be
    a bare string; `params` holds param *slots*, so `rot.weight` names `rot`
    and `== "rot"` misses it (`validate._check_train_references` compares on
    the first dotted segment, and now so does this); and `anneal` is a mapping
    *keyed* by the slot, so its names are in its keys and nowhere else."""
    for block in (
        # the named objective form: a term keyed by its own name, regularizer
        # inside it, value a bare string rather than a list
        _train_block(objective={"decay": {"weight": 0.01, "l1": "rot"}}),
        _train_block(params=["rot.weight"]),
        _train_block(anneal={"rot.weight": [1.0, 0.0, 0.5]}),
        # the positional form: [weight, regularizer], a list inside a list
        _train_block(objective=[[1.0, LD], [0.01, {"l2": ["rot.weight"]}]]),
    ):
        doc = _fit_doc()
        doc["method"]["train"] = block
        expect_family_refusal(in_order(doc), "do not fan out")


def test_the_train_blocks_those_refusals_use_are_real_schema_shapes() -> None:
    """The other half: with the featurizer as one entry rather than a family,
    every block above parses. So each refusal above is §3.1's, not §2.11's."""
    for block in (
        _train_block(objective={"decay": {"weight": 0.01, "l1": "rot"}}),
        _train_block(params=["rot.weight"]),
        _train_block(anneal={"rot.weight": [1.0, 0.0, 0.5]}),
        _train_block(objective=[[1.0, LD], [0.01, {"l2": ["rot.weight"]}]]),
    ):
        doc = _fit_doc()
        doc["method"]["featurizers"] = {"rot": {"kind": "subspace", "k": 2}}
        doc["method"]["train"] = block
        parse_document(in_order(doc))


def test_the_refusal_message_uses_the_keyword_the_author_wrote() -> None:
    """One value grammar, two keywords — so the shared messages take the word
    *and its article*. Reachable: a `sweep` wrapper inside a write-list
    selector lands in the family path, and an author who wrote `sweep` should
    not be told about `at_once` in bad grammar either way."""
    doc = band_doc()
    doc["method"]["sites"]["a"]["layers"] = {AT_ONCE_KEY: 7}
    err = expect_family_refusal(doc, "an at_once wrapper takes a list")
    assert " a at_once " not in str(err)
