"""The golden corpus (spec §7): every example loads, validates,
canonicalizes to a pinned digest, and derives the pinned execution shape —
and reaches the same pinned digest with its sections in alphabetical order,
which is §5 rule 2's whole content now that order is a recommendation.

The digests are pinned against the committed fixture tables in
``tests/protocol/fixtures`` (dataset content digests are part of the
canonical form, so the pins and the fixtures move together). Regenerate
with ``uv run python tests/protocol/update_corpus_digests.py`` and review
the diff — a digest change means the canonical form changed, which is a
loader migration event (spec §7), not a routine edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.protocol.engine import component_capability, requires
from causalab.protocol.rules.errors import ProtocolWarning
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import (
    closure_digest,
    interned_groups,
    plan_point,
)
from causalab.protocol.positions.alignment import COMPONENT_RANK

from tests.protocol._env import CORPUS_DIR, steps_of

PINS_PATH = Path(__file__).parent / "corpus_digests.json"
PINS = json.loads(PINS_PATH.read_text())


def _touch(*components: str) -> set[str]:
    """The generated capability entries for touched components — a trailing
    ``+w`` marks one the document also writes (§8, component routing)."""
    needed: set[str] = set()
    for item in components:
        name, wrote = (item[:-2], True) if item.endswith("+w") else (item, False)
        needed.add(component_capability(name))
        if wrote:
            needed.add(component_capability(name, write=True))
    return needed


#: (file, expected points, expected forwards per point, expected requires —
#: coarse capabilities plus the generated component entries)
CORPUS_SHAPE = [
    ("01_harvest_im.json", 1, 1, _touch("block_output")),
    (
        "02_interchange_im.json",
        1,
        2,
        {"paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "03_path_patching_im.json",
        1,
        4,
        {"paired_forward"}
        | _touch(
            "attention_output+w", "attention_premix+w", "block_input+w", "lm_head"
        ),
    ),
    (
        "04_das_im.json",
        1,
        2,
        {"grad", "paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "05_dbm_im.json",
        1,
        2,
        {"grad", "paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "06_hydra_effect_im.json",
        1,
        7,
        {"paired_forward"} | _touch("attention_output+w", "block_output+w", "lm_head"),
    ),
    (
        "07_weekdays_locate_scan_im.json",
        64,
        2,
        {"paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "08_weekdays_das_sweep_im.json",
        9,
        2,
        {"grad", "paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "09_das_apply_im.json",
        1,
        2,
        {"paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "10_task_table_iia_im.json",
        1,
        3,
        {"full_logits", "paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "11_probe_generate_im.json",
        1,
        1,
        {"generate", "full_logits"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "12_probe_variable_im.json",
        1,
        1,
        {"generate", "full_logits"} | _touch("block_output", "lm_head"),
    ),
    (
        "13_random_subspace_control_im.json",
        3,  # one point per random draw: the sweep IS the control distribution
        2,
        {"paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        "14_multi_position_patch_im.json",
        1,
        2,
        {"paired_forward"} | _touch("block_output+w", "lm_head"),
    ),
    (
        # Eight writes, two intervened models, four forwards: the per-edge
        # polarity map costs one joint forward per edge set, not one per edge.
        "15_circuit_edges_im.json",
        1,
        4,
        {"paired_forward"} | _touch("attention_result", "block_input+w", "lm_head"),
    ),
    (
        # one point, four forwards: the shared counterfactual harvest of all ten
        # taps plus one per band — the same shape 06's hand-written bands derive,
        # which is the property `at_once` must not disturb (sec. 3.1)
        "16_at_once_band_im.json",
        1,
        4,
        {"paired_forward"} | _touch("attention_output+w", "lm_head"),
    ),
]


class TestCorpusUnit:
    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize("name,n_points,n_forwards,needed", CORPUS_SHAPE)
    def test_loads_and_derives_shape(self, env, name, n_points, n_forwards, needed):
        loaded = compile_protocol(CORPUS_DIR / name, env=env)
        assert len(steps_of(loaded, env).points) == n_points
        doc = steps_of(loaded, env).documents[0]
        assert plan_point(doc).num_forwards == n_forwards
        assert set(requires(doc)) == needed

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_sorted_key_order_keeps_the_pinned_digest(self, env, name):
        """§5 rule 2 — the section order is not content.

        ``json.dumps(..., sort_keys=True)`` is the reordering a document
        acquires by accident: any tool that rewrites JSON does it, and the
        loader used to refuse the result. It warns and parses now, and this is
        the claim that makes that safe — the round-tripped document is not
        merely *a* valid document but the same one, down to the digest pinned
        in ``corpus_digests.json`` and every point digest under it.
        """
        original = compile_protocol(CORPUS_DIR / name, env=env)
        authored = json.loads((CORPUS_DIR / name).read_text())
        sorted_order = json.loads(json.dumps(authored, sort_keys=True))
        assert list(sorted_order) != list(authored), (
            f"{name} is already in alphabetical order — this test would pass "
            "without exercising anything"
        )

        with pytest.warns(ProtocolWarning, match="recommended"):
            shuffled = compile_protocol(sorted_order, env=env, base_dir=CORPUS_DIR)

        assert shuffled.digests.document == PINS[name]["document"]
        assert steps_of(shuffled, env).digests == steps_of(original, env).digests

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_document_digest_pin(self, env, name):
        loaded = compile_protocol(CORPUS_DIR / name, env=env)
        assert loaded.digests.document == PINS[name]["document"], (
            f"{name}: canonical form drifted — if intended, regenerate the pins "
            "(update_corpus_digests.py) and treat it as a loader migration (§7)"
        )

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_point_digest_pins(self, env, name):
        loaded = compile_protocol(CORPUS_DIR / name, env=env)
        assert list(steps_of(loaded, env).digests) == PINS[name]["points"]

    def test_sweep_points_are_distinct(self, env):
        loaded = compile_protocol(
            CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env
        )
        assert len(set(steps_of(loaded, env).digests)) == 64

    def test_das_sweep_interns_one_harvest(self, env):
        """§3's forcing example: 9 fits (k × seed) share ONE counterfactual harvest —
        the original/counterfactual forward group has one key across all points,
        while the patched groups are 9 distinct fits."""
        loaded = compile_protocol(CORPUS_DIR / "08_weekdays_das_sweep_im.json", env=env)
        harvest, patched, v_cf = set(), set(), set()
        identity = {"base": "d", "counterfactual": "d"}
        for pdoc in steps_of(loaded, env).documents:
            for group in plan_point(pdoc, data_identity=identity).groups:
                (harvest if group.unwritten else patched).add(group.key)
            v_cf.add(closure_digest(pdoc, "v_cf"))
        assert len(harvest) == 1
        assert len(patched) == 9
        assert len(v_cf) == 9

    def test_locate_scan_shares_per_layer_harvests(self, env):
        """07: the 64 points span 32 layers × 2 positions; the counterfactual-side
        harvest group of a point depends on nothing swept (taps differ, the
        forward doesn't), so all 64 original/counterfactual groups intern to one."""
        loaded = compile_protocol(
            CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env
        )
        identity = {"base": "d", "counterfactual": "d"}
        harvest = {
            group.key
            for pdoc in steps_of(loaded, env).documents
            for group in plan_point(pdoc, data_identity=identity).groups
            if group.unwritten
        }
        assert len(harvest) == 1

    def test_locate_scan_owes_65_forwards_not_128(self, env):
        """07's own description, as arithmetic: "64 patched forwards plus one
        shared counterfactual-harvest forward".

        64 points x 2 groups is 128 group *instances*; interning collapses
        them to 65, because the patched forwards are genuinely distinct (each
        patches a different layer with a differently-tapped value) while the
        harvest depends on nothing swept. The merged harvest carries one tap
        per layer, so an eliding backend may stop it at the deepest of those —
        never at one point's."""
        loaded = compile_protocol(
            CORPUS_DIR / "07_weekdays_locate_scan_im.json", env=env
        )
        identity = {"base": "d", "counterfactual": "d"}
        plans = [
            plan_point(pdoc, data_identity=identity)
            for pdoc in steps_of(loaded, env).documents
        ]
        assert sum(plan.num_forwards for plan in plans) == 128
        groups = interned_groups(plans)
        assert len(groups) == 65
        (harvest,) = [group for group in groups if group.unwritten]
        patched = [group for group in groups if not group.unwritten]
        assert len(patched) == 64
        # 32 layers x 2 positions, and the position axis moves the gather
        # rather than the forward — so 32 taps, not 64
        assert len(harvest.taps) == 32
        assert harvest.stop_after == (31, COMPONENT_RANK["block_output"])

    def test_interning_a_single_point_is_the_point_plan(self, env):
        """With one point there is nothing to share, so interning must be the
        identity — the guard against a merge that quietly drops a group."""
        loaded = compile_protocol(CORPUS_DIR / "06_hydra_effect_im.json", env=env)
        plan = plan_point(steps_of(loaded, env).documents[0])
        assert len(interned_groups([plan])) == plan.num_forwards == 7


class TestCorpusCompleteness:
    pytestmark = pytest.mark.unit

    def test_every_corpus_file_is_covered(self):
        """A new *_im.json must enter CORPUS_SHAPE and the pins file — the
        corpus, the shape table, and the digests move together."""
        files = sorted(path.name for path in CORPUS_DIR.glob("*_im.json"))
        assert files == sorted(row[0] for row in CORPUS_SHAPE)
        assert files == sorted(PINS)


class TestCorpusProperty:
    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("name", [row[0] for row in CORPUS_SHAPE])
    def test_load_is_deterministic(self, env, name):
        first = compile_protocol(CORPUS_DIR / name, env=env)
        second = compile_protocol(CORPUS_DIR / name, env=env)
        assert first.digests.document == second.digests.document
        assert steps_of(first, env).digests == steps_of(second, env).digests
        assert first.canonical == second.canonical
