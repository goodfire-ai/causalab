"""Named axes — correlated row tuples and dependent axes (spec §3.2).

The two acceptance clauses: the `entity` test's three correlated rows × two
seeds are exactly six points, and the `ROME` replication's 48-way hand
expansion collapses to one declaration.

* **T1** — three correlated rows × two seeds expand to exactly six points, in
  the asserted order, each point's canonical form and digest equal to the
  hand-written point's; the campaign's canonical form carries the ``axes``
  block and the lowered ``{"sweep": [[8],[9],[10]]}`` column. *Mutation:*
  expand the display form instead of the rows → 54 points.
* **T2** — ROME: clipped ten-layer windows centred at each of 48 layers,
  twice, on the 48-layer ``gpt2-xl`` registry entry; every member within
  0–47; coordinates and point digests identical across two compiles.
  *Mutation:* drop the clip → rule 4's out-of-range refusal fires.
* **T3** — a per-row expert axis is refused at load as ``P4``, naming
  ``fan_out``. *Mutation:* accept unknown kinds and nothing else notices.
* **T4** — the legitimate campaign: no shipped preset, corpus or workflow
  document declares the group; every corpus and workflow pin holds against
  the pin *files*; the 28×2 shipped grid is 56 points and the 32×2 corpus grid 64, in the pinned order.

Around them: the §3.2 refusal table, each refusal beside its valid twin;
``--set`` into a row; the stage between ``families`` and ``gate``; the module
outside the hashed closure; the gate never seeing the block; a wrapper on an
``at_once`` family entry shared by every member. Offline, no model load: the
registry entry is the only model fact any of this reads.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import register_model_key
from causalab.protocol import pipeline
from causalab.neural.shared.sweep import expand_axes
from causalab.protocol.lowering import (
    AXES_KEY,
    AXIS_KINDS,
    CLIP_TARGETS,
    RULE_KINDS,
    canonical_axes,
    has_axes,
    lower_axes,
    parse_axes,
)
from causalab.protocol.schema.explicit import canonical_bytes, canonicalize, digest
from causalab.protocol.identity import import_closure
from causalab.protocol.pipeline import STAGES, compile_protocol
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.registry import get_model_info
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import GROUP_ORDER, parse_document
from causalab.protocol.lowering import AT_ONCE_KEY, coordinate_label
from causalab.workflow.document import load_workflow

from tests.protocol._docs import LOGIT_DIFF, UNWRITTEN, base_doc, in_order, saved
from tests.protocol._env import CORPUS_DIR, steps_of
from tests.protocol.test_protocol_presets import RUN_TREE_ONLY
from tests.workflow.test_closure_census import SHARED
from tests._helpers.paths import PROTOCOLS_DIR, WORKFLOWS_DIR


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PRESETS = PROTOCOLS_DIR
WORKFLOWS = WORKFLOWS_DIR
CORPUS_PINS = json.loads((Path(__file__).parent / "corpus_digests.json").read_text())
DAS = CORPUS_DIR / "04_das_im.json"


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _with_axes(doc: dict[str, Any], axes: dict[str, Any]) -> dict[str, Any]:
    """``doc`` with the ``axes`` group in its place — between ``data`` and
    ``method`` — and the method sections in §1 order."""
    ordered = in_order(doc)
    out: dict[str, Any] = {}
    for group in GROUP_ORDER:
        if group == "method":
            out[AXES_KEY] = axes
        out[group] = ordered[group]
    return out


#: `entity`'s three locations: a (layers, component, pos) tuple per row.
LOCATIONS: list[dict[str, Any]] = [
    {"layers": 8, "component": "attention_output", "pos": {"index": -1}},
    {"layers": 9, "component": "mlp_output", "pos": {"index": -1}},
    {"layers": 10, "component": "block_output", "pos": {"index": -2}},
]


#: The same three rows as the canonical form records them: ``layers`` folded to
#: the one-layer band, the spelling the display column carries (§7).
FOLDED_LOCATIONS: list[dict[str, Any]] = [
    {**row, "layers": [row["layers"]]} for row in LOCATIONS
]


def entity_doc() -> dict[str, Any]:
    """T1's document: corpus 04 (a DAS fit on Qwen3-8B, so ``train.seed``
    exists) with its target moved onto the ``location`` rows and the seed
    swept."""
    doc: dict[str, Any] = json.loads(DAS.read_text())
    method = doc["method"]
    method["positions"] = {"tap": {"axis": "location.pos"}}
    method["sites"]["target"] = {
        "component": {"axis": "location.component"},
        "layers": {"axis": "location.layers"},
    }
    method["reads"]["v_cf"]["pos"] = "tap"
    method["writes"]["patch"]["pos"] = "tap"
    method["train"]["seed"] = {"sweep": [0, 1]}
    return _with_axes(
        doc, {"location": {"rows": copy.deepcopy(LOCATIONS), "key": "layers"}}
    )


def entity_point(row: dict[str, Any], seed: int) -> dict[str, Any]:
    """The same point, written by hand."""
    doc: dict[str, Any] = json.loads(DAS.read_text())
    method = doc["method"]
    method["positions"] = {"tap": row["pos"]}
    method["sites"]["target"] = {"component": row["component"], "layers": row["layers"]}
    method["reads"]["v_cf"]["pos"] = "tap"
    method["writes"]["patch"]["pos"] = "tap"
    method["train"]["seed"] = seed
    return in_order(doc)


def rome_doc(*, width: int = 10) -> dict[str, Any]:
    """T2's document: the base interchange on ``gpt2-xl`` (48 layers in the
    registry), its target a clipped window centred at each of 48 layers."""
    doc = base_doc()
    doc["model"] = {"key": "gpt2-xl", "revision": "main"}
    doc["method"]["sites"]["tgt"] = {
        "component": "block_output",
        "layers": {"axis": "window"},
    }
    return _with_axes(
        doc,
        {
            "center": {"range": [0, 48]},
            "window": {
                "dependent_on": "center",
                "rule": {"clipped_band": {"width": width, "clip_to": "layers"}},
            },
        },
    )


def _compile(doc: dict[str, Any] | Path, env: ResolutionEnv, **kwargs: Any):
    return compile_protocol(
        doc, env=env, base_dir=None, overrides=None, engine=None, **kwargs
    )


def _refuses(
    doc: dict[str, Any], env: ResolutionEnv, code: str, path: str, *needles: str
) -> ParseError:
    with pytest.raises(ParseError) as err:
        _compile(doc, env)
    assert err.value.code == code, str(err.value)
    assert err.value.path == path, str(err.value)
    for needle in needles:
        assert needle in str(err.value), (needle, str(err.value))
    return err.value


# --------------------------------------------------------------------------- #
# T1 — entity: three correlated rows × two seeds = exactly six points
# --------------------------------------------------------------------------- #


def test_three_correlated_rows_times_two_seeds_are_six_points(env) -> None:
    """T1. The rows are the slowest coordinate, the seed the fastest; each
    point is the hand-written point to the byte; the campaign's canonical form
    carries the block, and its lowered column is the band spelling.

    Fails without the change because the ``axes`` group is an unknown group at
    the gate (``P3``); fails under the mutation — expanding the display form —
    with 54 points."""
    compiled = _compile(entity_doc(), env)
    steps = steps_of(compiled, env)
    points = steps.points
    assert len(points) == 6
    assert [dict(p.coords) for p in points] == [
        {"axes.location": 8, "train.seed": 0},
        {"axes.location": 8, "train.seed": 1},
        {"axes.location": 9, "train.seed": 0},
        {"axes.location": 9, "train.seed": 1},
        {"axes.location": 10, "train.seed": 0},
        {"axes.location": 10, "train.seed": 1},
    ]
    assert [axis.id for axis in compiled.axes] == ["axes.location", "train.seed"]
    for canonical, (row, seed) in zip(
        steps.canonical,
        [(row, seed) for row in LOCATIONS for seed in (0, 1)],
        strict=True,
    ):
        twin = canonicalize(entity_point(row, seed), env)
        assert canonical == twin
        assert digest(twin) in steps.digests
    assert list(steps.digests) == [digest(c) for c in steps.canonical]
    # the campaign: the block is in the canonical form, between data and method
    assert list(compiled.canonical) == ["header", "model", "data", "axes", "method"]
    assert compiled.canonical["axes"] == {
        "location": {"rows": FOLDED_LOCATIONS, "key": "layers"}
    }
    # the display form: every wrapper a sweep over its column, the band spelled
    assert compiled.canonical["method"]["sites"]["target"]["layers"] == {
        "sweep": [[8], [9], [10]]
    }
    # the block spells each field exactly as the display column does
    assert [
        row["layers"] for row in compiled.canonical["axes"]["location"]["rows"]
    ] == (compiled.canonical["method"]["sites"]["target"]["layers"]["sweep"])
    assert compiled.tree["method"]["sites"]["target"]["component"] == {
        "sweep": ["attention_output", "mlp_output", "block_output"]
    }
    assert AXES_KEY not in compiled.tree
    assert coordinate_label(points[0].coords) == "[axes.location=8,seed=0]"


def test_t1_rows_are_recorded_folded(env) -> None:
    """The fold-up: a rows axis authored ``layers: 8`` and its
    twin authored ``layers: [8]`` are byte-identical in `canonical_axes`, one
    campaign digest, and the block's spelling is the display column's.

    The axis is keyed on ``component`` here: the key field is a scalar per row,
    so a ``layers``-keyed row may only spell ``8``, and the two spellings exist
    for a keyless axis or one keyed on another field.

    Fails without the change (the block recorded the rows as authored, so the
    two spellings were two campaign digests over identical points)."""
    doc = entity_doc()
    doc[AXES_KEY]["location"]["key"] = "component"
    as_index = _compile(doc, env)
    doc = entity_doc()
    doc[AXES_KEY]["location"] = {
        "rows": copy.deepcopy(FOLDED_LOCATIONS),
        "key": "component",
    }
    as_band = _compile(doc, env)
    assert [p.coords["axes.location"] for p in steps_of(as_index, env).points] == [
        row["component"] for row in LOCATIONS for _ in (0, 1)
    ]
    assert as_index.canonical["axes"] == as_band.canonical["axes"]
    assert canonical_bytes(as_index.canonical) == canonical_bytes(as_band.canonical)
    assert as_index.digests.document == as_band.digests.document
    assert list(steps_of(as_index, env).digests) == list(steps_of(as_band, env).digests)
    assert [dict(p.coords) for p in steps_of(as_index, env).points] == [
        dict(p.coords) for p in steps_of(as_band, env).points
    ]
    # `axes` and `method` spell the folded field the same way
    assert [
        row["layers"] for row in as_index.canonical["axes"]["location"]["rows"]
    ] == (as_index.canonical["method"]["sites"]["target"]["layers"]["sweep"])
    # a field the fold leaves alone is recorded as authored, in row 0's order
    rows = as_index.canonical["axes"]["location"]["rows"]
    assert [list(row) for row in rows] == [["layers", "component", "pos"]] * 3
    assert [row["pos"] for row in rows] == [row["pos"] for row in LOCATIONS]


def test_an_already_folded_rows_axis_keeps_its_canonical_bytes(env) -> None:
    """A document whose rows were authored in the folded spelling records the
    same block before and after the fold — what `canonical_axes` recorded as
    authored is what it now records folded — so no such document's campaign
    digest moves. Passes on the pre-fold tree too; the witness for the change is
    `test_t1_rows_are_recorded_folded`."""
    doc = entity_doc()
    doc[AXES_KEY]["location"] = {
        "rows": copy.deepcopy(FOLDED_LOCATIONS),
        "key": "component",
    }
    compiled = _compile(doc, env)
    block = {"location": {"rows": FOLDED_LOCATIONS, "key": "component"}}
    assert compiled.canonical["axes"] == block
    assert canonical_bytes(compiled.canonical) == canonical_bytes(
        {**compiled.canonical, "axes": block}
    )


def test_the_display_forms_cross_product_is_not_the_expansion(env) -> None:
    """The count T1's mutation would produce, stated: the lowered document,
    expanded as the independent axes it looks like, is 3·3·3·2 = 54 points —
    and that is exactly what a compile of the same document is not."""
    from causalab.neural.shared.sweep import expand

    doc = entity_doc()
    lowered = lower_axes(doc, parse_axes(doc, get_model_info))
    assert len(expand(lowered).points) == 54
    assert len(steps_of(_compile(doc, env), env).points) == 6


def test_a_rows_axis_without_a_key_is_indexed(env) -> None:
    doc = entity_doc()
    del doc[AXES_KEY]["location"]["key"]
    compiled = _compile(doc, env)
    assert [p.coords["axes.location"] for p in steps_of(compiled, env).points] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert compiled.canonical["axes"]["location"] == {"rows": FOLDED_LOCATIONS}


def test_two_identical_rows_are_p2_and_distinct_rows_enumerate(env) -> None:
    """Rows are distinct (§3.2), compared folded: a keyless axis whose row 2
    repeats row 0 — spelled ``[8]`` against ``8`` — is `P2` at the repeated
    row, naming the axis and both indices, before the gate; the valid twin
    (three distinct keyless rows) still enumerates to 3 × 2 points."""
    doc = entity_doc()
    del doc[AXES_KEY]["location"]["key"]
    doc[AXES_KEY]["location"]["rows"][2] = {**LOCATIONS[0], "layers": [8]}
    err = _refuses(
        doc,
        env,
        "P2",
        "axes.location.rows[2]",
        "rows 0 and 2 of axis 'location' are one row",
        "rows are distinct",
    )
    assert "'layers': [8]" in str(err)
    twin = entity_doc()
    del twin[AXES_KEY]["location"]["key"]
    assert len(steps_of(_compile(twin, env), env).points) == 6


def test_a_rows_axis_crosses_with_a_second_named_axis(env) -> None:
    """Named axes cross with each other (they are axes); only the fields of
    one row are correlated. Declaration order is coordinate order."""
    doc = entity_doc()
    doc[AXES_KEY]["k"] = {"values": [4, 8]}
    doc["method"]["featurizers"]["rot"]["k"] = {"axis": "k"}
    compiled = _compile(doc, env)
    coords = [dict(p.coords) for p in steps_of(compiled, env).points]
    assert len(coords) == 12
    assert coords[0] == {"axes.location": 8, "axes.k": 4, "train.seed": 0}
    assert coords[1] == {"axes.location": 8, "axes.k": 4, "train.seed": 1}
    assert coords[2] == {"axes.location": 8, "axes.k": 8, "train.seed": 0}
    assert coords[3] == {"axes.location": 8, "axes.k": 8, "train.seed": 1}
    assert coords[4] == {"axes.location": 9, "axes.k": 4, "train.seed": 0}
    assert compiled.canonical["axes"]["k"] == {"values": [4, 8]}


# --------------------------------------------------------------------------- #
# T2 — ROME: clipped ten-layer windows centred at each of 48 layers
# --------------------------------------------------------------------------- #


def test_t2_clipped_windows_over_48_centers_twice(env) -> None:
    """T2, the acceptance clause: "Expand clipped ten-layer windows centered at each
    of 48 layers twice. Every member stays within layers 0 through 47;
    explicit sites and point order match across loads; digests are stable."

    Fails under the mutation — dropping the clip — because centre 0's window
    would start at −5 and rule 4 refuses it (`test_dropping_the_clip_…`)."""
    assert get_model_info("gpt2-xl").num_layers == 48
    first = _compile(rome_doc(), env)
    second = _compile(rome_doc(), env)
    steps = steps_of(first, env)
    points = steps.points
    assert len(points) == 48
    bands = [p.raw["method"]["sites"]["tgt"]["layers"] for p in points]
    assert all(0 <= layer <= 47 for band in bands for layer in band)
    assert bands[0] == [0, 1, 2, 3, 4]
    assert bands[5] == list(range(0, 10))
    assert bands[24] == list(range(19, 29))
    assert bands[47] == [42, 43, 44, 45, 46, 47]
    assert [dict(p.coords) for p in points] == [{"axes.center": c} for c in range(48)]
    assert [axis.id for axis in first.axes] == ["axes.center"]
    assert [dict(p.coords) for p in steps_of(second, env).points] == [
        dict(p.coords) for p in points
    ]
    assert second.digests == first.digests
    assert second.canonical == first.canonical
    # one declaration: the campaign carries the centres and the computed windows
    block = first.canonical["axes"]
    assert block["center"] == {"values": list(range(48))}
    assert block["window"]["dependent_on"] == "center"
    assert block["window"]["rule"] == {
        "clipped_band": {"width": 10, "clip_to": "layers"}
    }
    assert block["window"]["values"] == bands
    # the display form: one sweep over the 48 bands
    assert first.canonical["method"]["sites"]["tgt"]["layers"] == {"sweep": bands}
    # each point is the hand-written band site
    hand = base_doc()
    hand["model"] = {"key": "gpt2-xl", "revision": "main"}
    hand["method"]["sites"]["tgt"]["layers"] = [42, 43, 44, 45, 46, 47]
    assert steps.canonical[47] == canonicalize(hand, env)


def test_a_window_narrower_than_the_tower_clips_only_at_the_edges(env) -> None:
    doc = rome_doc(width=3)
    bands = [
        p.raw["method"]["sites"]["tgt"]["layers"]
        for p in steps_of(_compile(doc, env), env).points
    ]
    assert bands[0] == [0, 1]
    assert bands[1] == [0, 1, 2]
    assert bands[47] == [46, 47]
    assert all(len(b) == 3 for b in bands[1:47])


def test_an_unclipped_band_is_what_rule_4_refuses(env) -> None:
    """The refusal T2's mutation must reach: a window written out by hand with
    a member below the tower is refused by the canonical form's own rule 4 —
    so the clip is doing work the pipeline would otherwise refuse."""
    hand = base_doc()
    hand["model"] = {"key": "gpt2-xl", "revision": "main"}
    hand["method"]["sites"]["tgt"]["layers"] = {
        "sweep": [[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]]
    }
    with pytest.raises(ValidationError) as err:
        _compile(hand, env)
    assert err.value.rule == 4
    assert "out of range" in str(err.value)


# --------------------------------------------------------------------------- #
# T3 — a per-row expert axis is refused at load, naming fan_out
# --------------------------------------------------------------------------- #


def test_a_per_row_expert_axis_is_refused_at_load_naming_fan_out(env) -> None:
    """T3: "one intervention per routed expert selected for that row" is a
    run-time fan-out — routing is known only after a forward — and the
    document layer refuses the spelling by name, pointing at the mechanism
    that owns it. Accepting it would make no other test red, which is why this
    one exists."""
    doc = rome_doc()
    doc[AXES_KEY]["window"] = {
        "dependent_on": "center",
        "rule": {"routed_experts": {"per": "row"}},
    }
    err = _refuses(
        doc, env, "P4", "axes.window.rule", "routed_experts", "clipped_band", "fan_out"
    )
    assert "declare it in the workflow" in str(err)
    assert "pure function of the document" in str(err)
    # the valid twin is T2 itself
    assert len(steps_of(_compile(rome_doc(), env), env).points) == 48


# --------------------------------------------------------------------------- #
# T4 — the legitimate campaign: nothing shipped declares the group, every pin holds
# --------------------------------------------------------------------------- #

CORPUS = sorted(CORPUS_DIR.glob("*_im.json"))
STANDALONE_PRESETS = sorted(
    p for p in PRESETS.glob("*.json") if p.name not in RUN_TREE_ONLY
)


def test_the_corpus_is_the_pinned_set() -> None:
    assert [p.name for p in CORPUS] == sorted(CORPUS_PINS)
    assert len(CORPUS) == 16


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.name)
def test_no_corpus_document_declares_axes_and_its_pin_holds(path: Path, env) -> None:
    """T4, per corpus document: no ``axes`` group, and the document digest,
    the point digests and their order are the pin file's — asserted against
    the file, never a literal, so a deliberate re-pin elsewhere moves this
    with it and an accidental move fails here."""
    raw = json.loads(path.read_text())
    assert not has_axes(raw)
    loaded = compile_protocol(path, env=env)
    pin = CORPUS_PINS[path.name]
    assert loaded.digests.document == pin["document"]
    assert list(steps_of(loaded, env).digests) == pin["points"]
    assert AXES_KEY not in loaded.canonical
    assert all("axes." not in axis.id for axis in steps_of(loaded, env).axes)


@pytest.mark.parametrize("path", sorted(PRESETS.glob("*.json")), ids=lambda p: p.name)
def test_no_shipped_preset_declares_axes(path: Path) -> None:
    assert not has_axes(json.loads(path.read_text()))


@pytest.mark.parametrize("path", STANDALONE_PRESETS, ids=lambda p: p.name)
def test_a_shipped_preset_compiles_without_the_block_in_its_canonical_form(
    path: Path, env
) -> None:
    raw = json.loads(path.read_text())
    register_model_key(raw)  # the CLI's step: a tiny-fixture key joins the registry
    loaded = compile_protocol(path, env=env)
    assert AXES_KEY not in loaded.canonical
    assert all("axes." not in axis.id for axis in steps_of(loaded, env).axes)


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.json")), ids=lambda p: p.name)
def test_no_shipped_workflow_declares_axes(path: Path, env) -> None:
    """T4, the workflow half: no shipped workflow's inner document declares
    the group, so none of their step identities carries it."""
    assert not has_axes(json.loads(path.read_text()))
    loaded = load_workflow(path, env)
    for inner in loaded.inner.values():
        assert AXES_KEY not in inner.compiled.canonical


def test_the_layer_by_position_grid_is_still_every_point_in_the_pinned_order(
    env,
) -> None:
    """T4's named case: `weekdays_locate_scan`'s two axes, in document order,
    56 points (28 layers x 2 positions), `positions.tap` slowest — and corpus
    07, the 32-layer grid, in the order its pin lists."""
    loaded = compile_protocol(PRESETS / "weekdays_locate_scan.json", env=env)
    assert [axis.id for axis in steps_of(loaded, env).axes] == [
        "positions.tap",
        "sites.target.layers",
    ]
    assert len(steps_of(loaded, env).points) == 56  # 28 layers x 2 positions
    layers = [p.coords["sites.target.layers"] for p in steps_of(loaded, env).points]
    assert layers == list(range(28)) * 2
    corpus = compile_protocol(CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env)
    assert len(steps_of(corpus, env).points) == 64
    assert (
        list(steps_of(corpus, env).digests)
        == CORPUS_PINS["07_weekdays_locate_scan_im.json"]["points"]
    )
    assert [axis.id for axis in steps_of(corpus, env).axes] == [
        "positions.tap",
        "sites.target.layers",
    ]


def test_a_document_without_the_group_takes_the_plain_path(env) -> None:
    """The no-axes path is the path it always was: ``expand`` on the explicit
    document, no block in the canonical form, no ``axes.`` coordinate."""
    doc = base_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"sweep": [3, 4]}
    compiled = _compile(doc, env)
    assert not has_axes(doc)
    assert AXES_KEY not in compiled.canonical
    assert [dict(p.coords) for p in steps_of(compiled, env).points] == [
        {"sites.tgt.layers": 3},
        {"sites.tgt.layers": 4},
    ]
    assert compiled.digests.document == compile_protocol(doc, env=env).digests.document


# --------------------------------------------------------------------------- #
# the refusal table (§3.2), each beside its valid twin
# --------------------------------------------------------------------------- #


def test_the_valid_twins_compile(env) -> None:
    """Every refusal below is a one-edit mutation of one of these two."""
    assert len(steps_of(_compile(entity_doc(), env), env).points) == 6
    assert len(steps_of(_compile(rome_doc(), env), env).points) == 48


def test_a_row_that_is_not_an_object_is_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"]["rows"][1] = [9, "mlp_output"]
    _refuses(doc, env, "P2", "axes.location.rows[1]", "row 1", "object")


def test_ragged_rows_are_p2(env) -> None:
    doc = entity_doc()
    del doc[AXES_KEY]["location"]["rows"][2]["pos"]
    _refuses(doc, env, "P2", "axes.location.rows[2]", "missing ['pos']")
    doc = entity_doc()
    doc[AXES_KEY]["location"]["rows"][1]["head"] = 3
    _refuses(doc, env, "P2", "axes.location.rows[1]", "extra ['head']")


def test_an_empty_row_list_and_an_empty_row_are_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"]["rows"] = []
    _refuses(doc, env, "P2", "axes.location.rows", "non-empty")
    doc = entity_doc()
    doc[AXES_KEY]["location"]["rows"][0] = {}
    _refuses(doc, env, "P2", "axes.location.rows[0]", "names no field")


def test_a_key_naming_no_field_is_p2_with_a_suggestion(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"]["key"] = "layer"
    _refuses(doc, env, "P2", "axes.location.key", "did you mean 'layers'")


def test_a_non_scalar_key_is_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"]["key"] = "pos"
    _refuses(doc, env, "P2", "axes.location.rows[0].pos", "scalar")


def test_a_repeated_key_is_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"]["rows"][1]["layers"] = 8
    _refuses(doc, env, "P2", "axes.location.key", "repeats 8")


def test_a_wrapper_inside_a_row_value_is_p2(env) -> None:
    for wrapper in ({"sweep": [8, 9]}, {AT_ONCE_KEY: [8, 9]}, {"axis": "center"}):
        doc = entity_doc()
        doc[AXES_KEY]["location"]["rows"][0]["layers"] = wrapper
        _refuses(
            doc, env, "P2", "axes.location.rows[0].layers", "a row value is a value"
        )


def test_an_unknown_key_in_a_declaration_is_p3_with_a_suggestion(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"]["rowz"] = []
    _refuses(doc, env, "P3", "axes.location.rowz", "did you mean 'rows'")
    doc = rome_doc()
    doc[AXES_KEY]["window"]["rule"]["clipped_band"]["widht"] = 10
    _refuses(
        doc, env, "P3", "axes.window.rule.clipped_band.widht", "did you mean 'width'"
    )


def test_a_declaration_of_no_kind_or_two_kinds_is_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["location"] = {"key": "layers"}
    _refuses(doc, env, "P2", "axes.location", "exactly one of")
    doc = rome_doc()
    doc[AXES_KEY]["center"] = {"range": [0, 48], "values": [1]}
    _refuses(doc, env, "P2", "axes.center", "['range', 'values']")


def test_a_wrapper_naming_an_undeclared_axis_is_p2_with_a_suggestion(env) -> None:
    doc = entity_doc()
    doc["method"]["sites"]["target"]["layers"] = {"axis": "locations.layers"}
    _refuses(doc, env, "P2", "sites.target.layers", "did you mean 'location'")


def test_a_wrapper_naming_an_undeclared_field_is_p2_with_a_suggestion(env) -> None:
    doc = entity_doc()
    doc["method"]["sites"]["target"]["layers"] = {"axis": "location.layer"}
    _refuses(
        doc,
        env,
        "P2",
        "sites.target.layers",
        "no row field 'layer'",
        "did you mean 'layers'",
    )


def test_a_rows_axis_referenced_without_a_field_is_p2(env) -> None:
    doc = entity_doc()
    doc["method"]["sites"]["target"]["layers"] = {"axis": "location"}
    _refuses(doc, env, "P2", "sites.target.layers", "referenced by field")


def test_a_scalar_axis_referenced_with_a_field_is_p2(env) -> None:
    doc = rome_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"axis": "window.layers"}
    _refuses(doc, env, "P2", "sites.tgt.layers", "has no fields")


def test_a_mapping_with_a_key_beside_axis_is_a_value(env) -> None:
    """A wrapper holds ``axis`` and nothing else; a mapping carrying more is a
    value, not a wrapper (a ``gaussian`` payload has an ``axis`` field). The
    axis it meant to reference is then referenced nowhere, and that is refused
    by name."""
    doc = rome_doc()
    doc["method"]["sites"]["tgt"]["layers"] = {"axis": "window", "axes": 1}
    _refuses(doc, env, "P2", "axes.window", "nothing references")


def test_a_wrapper_inside_a_list_is_p2(env) -> None:
    """A ``save`` entry is addressable by index (``save[0].…``), so the list
    without a name identity is an authored value list such as ``dims``."""
    doc = rome_doc()
    doc["method"]["reads"]["logits"]["dims"] = [{"axis": "center"}]
    doc["method"]["sites"]["tgt"]["layers"] = [3]
    _refuses(doc, env, "P2", "reads.logits.dims", "no name identity")


def test_an_unreferenced_axis_is_p2(env) -> None:
    doc = rome_doc()
    doc[AXES_KEY]["spare"] = {"values": [1, 2]}
    _refuses(doc, env, "P2", "axes.spare", "nothing references it")
    doc = rome_doc()
    doc["method"]["sites"]["tgt"]["layers"] = [3]  # the dependent axis is now unused
    _refuses(doc, env, "P2", "axes.window", "nothing references it")


def test_an_unreferenced_row_field_is_p2_unless_it_is_the_key(env) -> None:
    doc = entity_doc()
    doc["method"]["sites"]["target"]["component"] = "block_output"
    _refuses(doc, env, "P2", "axes.location.rows[0].component", "referenced nowhere")
    # the key may be a label: `layers` is the key, referenced nowhere, fine
    doc = entity_doc()
    doc["method"]["sites"]["target"]["layers"] = [8]
    compiled = _compile(doc, env)
    assert [p.coords["axes.location"] for p in steps_of(compiled, env).points] == [
        8,
        8,
        9,
        9,
        10,
        10,
    ]


def test_a_dependent_axis_on_a_rows_axis_is_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["window"] = {
        "dependent_on": "location",
        "rule": {"clipped_band": {"width": 4, "clip_to": "layers"}},
    }
    doc["method"]["sites"]["lm_head"] = {
        "component": "block_output",
        "layers": {"axis": "window"},
    }
    _refuses(doc, env, "P2", "axes.window.dependent_on", "rows axis")


def test_a_dependent_axis_on_an_undeclared_parent_is_p2(env) -> None:
    doc = rome_doc()
    doc[AXES_KEY]["window"]["dependent_on"] = "centre"
    _refuses(doc, env, "P2", "axes.window.dependent_on", "did you mean 'center'")


def test_a_swept_model_key_with_clip_to_is_p2(env) -> None:
    doc = rome_doc()
    doc["model"]["key"] = {"sweep": ["gpt2", "gpt2-xl"]}
    _refuses(doc, env, "P2", "axes.window.rule.clipped_band.clip_to", "swept")


def test_an_unknown_clip_target_is_p4(env) -> None:
    doc = rome_doc()
    doc[AXES_KEY]["window"]["rule"]["clipped_band"]["clip_to"] = "heads"
    _refuses(
        doc, env, "P4", "axes.window.rule.clipped_band.clip_to", "'heads'", "layers"
    )


def test_a_bad_width_or_range_is_p2(env) -> None:
    doc = rome_doc()
    doc[AXES_KEY]["window"]["rule"]["clipped_band"]["width"] = 0
    _refuses(doc, env, "P2", "axes.window.rule.clipped_band.width", "positive")
    doc = rome_doc()
    doc[AXES_KEY]["center"]["range"] = [48, 0]
    _refuses(doc, env, "P2", "axes.center.range", "denotes no value")
    doc = rome_doc()
    doc[AXES_KEY]["center"]["range"] = [0, 48, 0]
    _refuses(doc, env, "P2", "axes.center.range", "non-zero")


def test_non_scalar_or_repeated_values_are_p2(env) -> None:
    doc = entity_doc()
    doc[AXES_KEY]["k"] = {"values": [[4], [8]]}
    doc["method"]["featurizers"]["rot"]["k"] = {"axis": "k"}
    _refuses(doc, env, "P2", "axes.k.values[0]", "scalar")
    doc[AXES_KEY]["k"] = {"values": [4, 4]}
    _refuses(doc, env, "P2", "axes.k.values", "repeats 4")


def test_a_block_that_is_not_an_object_is_p2(env) -> None:
    doc = rome_doc()
    doc[AXES_KEY] = []
    _refuses(doc, env, "P2", "axes", "non-empty object")
    doc[AXES_KEY] = {}
    _refuses(doc, env, "P2", "axes", "non-empty object")
    doc[AXES_KEY] = {"center": 3}
    _refuses(doc, env, "P2", "axes.center", "object declaration")


def test_the_cap_is_taken_over_the_true_count(env) -> None:
    """Rule 14 over rows × the inner sweep — 3 × 2000 = 6000 — named as that
    count; and the same document under a cap that admits it compiles."""
    doc = entity_doc()
    doc["method"]["train"]["seed"] = {"sweep": {"range": [0, 2000]}}
    with pytest.raises(ValidationError) as err:
        _compile(doc, env)
    assert err.value.rule == 14
    assert "6000 points" in str(err.value) and "4096" in str(err.value)
    assert len(steps_of(_compile(doc, env, point_cap=None), env).points) == 6000


# --------------------------------------------------------------------------- #
# --set, families, the stage, the closure, the gate
# --------------------------------------------------------------------------- #


def test_set_can_address_a_row(env) -> None:
    """``--set axes.location.rows[0].layers=7`` lands: the override grammar
    indexes a list, and the group is addressed as itself (it is no method
    section, so the path spells it)."""
    compiled = compile_protocol(
        entity_doc(),
        env=env,
        base_dir=None,
        overrides={"axes.location.rows[0].layers": 7},
        engine=None,
    )
    first = steps_of(compiled, env).points[0]
    assert first.raw["method"]["sites"]["target"]["layers"] == [7]
    assert dict(first.coords) == {"axes.location": 7, "train.seed": 0}
    assert compiled.canonical["axes"]["location"]["rows"][0]["layers"] == [7]
    with pytest.raises(ParseError) as err:
        compile_protocol(
            entity_doc(),
            env=env,
            base_dir=None,
            overrides={"axes.location.rows[3].layers": 7},
            engine=None,
        )
    assert err.value.code == "P2" and "out of range" in str(err.value)


def test_an_axis_wrapper_on_a_family_entry_is_shared_by_every_member(env) -> None:
    """Families expand first, so a wrapper on an ``at_once`` entry is copied to
    every member — N references to one axis, legal by name identity — and
    every member of one point takes the same row's value."""
    doc = base_doc()
    doc["method"]["positions"] = {"tap": {"index": -1}}
    doc["method"]["sites"] = {
        "a": {
            "component": {"axis": "loc.component"},
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
    doc = _with_axes(
        doc,
        {"loc": {"rows": [{"component": "block_output"}, {"component": "mlp_output"}]}},
    )
    compiled = _compile(doc, env)
    points = steps_of(compiled, env).points
    assert [dict(p.coords) for p in points] == [{"axes.loc": 0}, {"axes.loc": 1}]
    for point, component in zip(points, ("block_output", "mlp_output"), strict=True):
        sites = point.raw["method"]["sites"]
        assert [sites[f"a{i}"]["component"] for i in (3, 4, 5)] == [component] * 3
    assert compiled.tree["method"]["sites"]["a3"]["component"] == {
        "sweep": ["block_output", "mlp_output"]
    }


def test_the_stage_sits_between_families_and_gate() -> None:
    """After families, so a wrapper on a family entry reaches every member;
    before the gate, which knows the four groups alone."""
    assert STAGES.index("families") < STAGES.index("axes") < STAGES.index("gate")
    assert pipeline._STAGE["axes"] is pipeline._axes  # pyright: ignore[reportPrivateUsage]


def test_the_named_axes_live_in_lowering() -> None:
    """The layering premise, stated directly: the named axes are
    ``lowering.py``'s — a SHARED member, reached through ``bundles.py``, so
    every shipped script's torch-free load parses them. Their enumeration
    (``expand_axes``) is the engine's, ``neural/shared/sweep.py``, which no
    SHARED member reaches. ``test_closure_census.py`` holds the frozen table."""
    assert "causalab/protocol/lowering.py" in SHARED
    closure = import_closure(REPO / "causalab/io/step_io.py", root=REPO)
    assert "causalab/protocol/lowering.py" in closure
    assert "causalab/neural/shared/sweep.py" not in closure


def test_the_gate_never_sees_the_block() -> None:
    """The parser knows four groups: handed the authored document it refuses
    the fifth as an unknown group (which is also what the base does, and why
    this file fails there); handed the lowered display form it parses."""
    doc = entity_doc()
    with pytest.raises(ParseError) as err:
        parse_document(doc)
    assert err.value.code == "P3" and "unknown group 'axes'" in str(err.value)
    lowered = lower_axes(doc, parse_axes(doc, get_model_info))
    assert AXES_KEY not in lowered
    assert list(lowered) == list(GROUP_ORDER)
    parse_document(lowered)
    # the lowering is a pure function of the document; idempotent on the block
    assert lowered == lower_axes(doc, parse_axes(doc, get_model_info))


def test_expansion_is_a_pure_function_of_the_document() -> None:
    axes = parse_axes(rome_doc(), get_model_info)
    once = expand_axes(axes, point_cap=None)
    twice = expand_axes(axes, point_cap=None)
    assert [p.raw for p in once.points] == [p.raw for p in twice.points]
    assert canonical_axes(axes) == canonical_axes(
        parse_axes(rome_doc(), get_model_info)
    )
    assert all(AXES_KEY not in p.raw for p in once.points)


def test_the_public_vocabulary_is_closed() -> None:
    assert AXIS_KINDS == ("rows", "range", "values", "dependent_on")
    assert RULE_KINDS == ("clipped_band",)
    assert CLIP_TARGETS == ("layers",)
    assert (
        AXES_KEY not in GROUP_ORDER
    )  # not a group the gate knows: the stage strips it


def test_a_gaussian_write_beside_a_declared_group_is_not_a_wrapper(env) -> None:
    """ROME's causal tracing: a ``gaussian`` corruption (whose payload carries
    a mandatory ``axis`` field, ``tp_duplicated`` | ``tp_split``, spec §2.8)
    beside a clipped-band restoration window (an ``axes`` group). The
    reference walk must not read the payload as an ``{"axis": …}`` wrapper —
    while a wrapper was any mapping carrying ``axis`` it was refused as ``P3``,
    "unknown key 'seed'", and the two could not share a document. A payload
    field that *is* a wrapper (``axis`` alone) is still found."""
    doc = rome_doc()
    method = doc["method"]
    method["sites"]["emb"] = {"component": "embeddings"}
    method["writes"]["corrupt"] = {
        "site": "emb",
        "pos": -1,
        "do": {"gaussian": {"seed": 0, "scale": 0.1, "axis": "tp_duplicated"}},
    }
    method["intervened_models"]["patched"]["writes"] = ["corrupt", "patch"]
    compiled = _compile(doc, env)
    assert len(steps_of(compiled, env).points) == 48
    point = steps_of(compiled, env).points[0].raw["method"]["writes"]["corrupt"]["do"]
    assert point == {"gaussian": {"seed": 0, "scale": 0.1, "axis": "tp_duplicated"}}
    # a wrapper inside the payload keeps its identity: an undeclared axis is refused
    doc["method"]["writes"]["corrupt"]["do"]["gaussian"]["scale"] = {"axis": "noise"}
    with pytest.raises(ParseError) as err:
        _compile(doc, env)
    assert "noise" in str(err.value)
