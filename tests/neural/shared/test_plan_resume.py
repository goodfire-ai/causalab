"""Where an intervened forward may *start* (spec §4, "Resume").

Elision says a forward may stop after its deepest tap. Its mirror image: an
intervened model's forward is the un-intervened forward up to the first
block a write lands in, so an engine holding that block's incoming residual
from any earlier pass over the same rows may start there. The plan states
two things an engine needs for that and nothing torch-shaped:

* ``ForwardGroup.base_key`` — the key the same input would have under
  ``original``: the identity of the prefix, shared by every intervened model
  on that input whatever it writes, and equal to the ``original`` group's own
  key when one is planned;
* ``ForwardGroup.resume_at`` — the block the forward may start at: the
  shallowest block any write **or any tap** touches (a tap below the first
  write still needs its block to run), 0 for ``original`` and for a group
  that decodes. Read it off the *interned* group, since the union of taps is
  what the one shared pass has to serve.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import (
    ForwardGroup,
    interned_groups,
    plan_point,
)
from causalab.protocol.positions.alignment import PAST_BLOCKS
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import parse_document

from tests.protocol._docs import base_doc, in_order, saved
from tests.protocol._env import CORPUS_DIR, steps_of
from tests.neural.shared.test_generate_plan import probe_doc


pytestmark = pytest.mark.unit


def _groups(raw: dict[str, Any]) -> dict[tuple[str, str], ForwardGroup]:
    plan = plan_point(parse_document(in_order(raw)))
    return {(g.model, g.input): g for g in plan.groups}


def _with_tap_on_patched(raw: dict[str, Any], layer: int, name: str) -> None:
    """A read of ``block_output`` at ``layer`` on ``patched``, saved so it is a
    sink (§5 rule 11)."""
    raw["method"]["sites"][f"probe{layer}"] = {
        "component": "block_output",
        "layers": [layer],
    }
    raw["method"]["reads"][name] = {"site": f"probe{layer}", "pos": -1}
    raw["method"]["intervened_models"]["patched"]["reads"].append(name)
    raw["method"]["save"].append(saved(name, "patched", f"{name}.st"))


# --------------------------------------------------------------------------- #
# base_key: the prefix's identity
# --------------------------------------------------------------------------- #


class TestBaseKey:
    def test_it_is_the_original_groups_key_on_the_same_input(
        self, env: ResolutionEnv
    ) -> None:
        """Corpus 06 reads ``original`` on ``base`` and runs five intervened
        models on ``base``: every one of them names the original group's
        key as its prefix identity."""
        loaded = compile_protocol(CORPUS_DIR / "06_hydra_effect_im.json", env=env)
        groups = {
            (g.model, g.input): g
            for g in plan_point(steps_of(loaded, env).documents[0]).groups
        }
        original = groups[("original_base", "base")]
        assert original.base_key == original.key
        ims = [
            g for (model, inp), g in groups.items() if not model.startswith("original")
        ]
        assert len(ims) == 5
        for group in ims:
            assert group.input == "base"
            assert group.base_key == original.key
            assert group.base_key != group.key

    def test_two_models_writing_at_different_layers_share_one_prefix(
        self, env: ResolutionEnv
    ) -> None:
        """``ablated`` writes at layer 13 and ``with_inj14_clean`` at layer
        31, on the same rows: one prefix identity, two resume depths — the
        deeper one reuses the shallower's un-intervened blocks and then some."""
        loaded = compile_protocol(CORPUS_DIR / "06_hydra_effect_im.json", env=env)
        groups = {
            (g.model, g.input): g
            for g in plan_point(steps_of(loaded, env).documents[0]).groups
        }
        ablated = groups[("ablated", "base")]
        injected = groups[("with_inj14_clean", "base")]
        assert ablated.base_key == injected.base_key
        assert ablated.key != injected.key
        # ablated: write at 13, taps at 14, 20 and lm_head -> 13
        assert ablated.resume_at == 13
        # with_inj14_clean: write at block_output 31, tap at lm_head -> 31
        assert injected.resume_at == 31

    def test_a_different_input_role_is_a_different_prefix(self) -> None:
        """The counterfactual harvest and the patched forward read different
        rows; their prefixes cannot be one another's."""
        groups = _groups(base_doc())
        assert (
            groups[("original_counterfactual", "counterfactual")].base_key
            != groups[("patched", "base")].base_key
        )

    def test_the_data_identity_is_part_of_it(self) -> None:
        doc = parse_document(in_order(base_doc()))
        a = plan_point(doc, data_identity={"base": "aaaa#input"})
        b = plan_point(doc, data_identity={"base": "bbbb#input"})
        pa = next(g for g in a.groups if g.model == "patched")
        pb = next(g for g in b.groups if g.model == "patched")
        assert pa.base_key != pb.base_key

    def test_the_frame_is_part_of_it(self) -> None:
        """A ``segments.frame: chat`` row is rendered through the tokenizer's
        chat template before it is encoded (§2.2.1): the same rows under the
        same model, a different token sequence, so a different residual at
        every block. The key has to say so — and with it the prefix's
        identity — or two documents differing only in their frame would
        share one prefix (and one interned capture)."""
        plain = _groups(base_doc())
        raw = base_doc()
        raw["method"]["segments"] = {"frame": "chat"}
        chat = _groups(raw)
        assert plain.keys() == chat.keys()
        for key in plain:
            assert plain[key].key != chat[key].key
            assert plain[key].base_key != chat[key].base_key


# --------------------------------------------------------------------------- #
# resume_at: the skip-depth arithmetic
# --------------------------------------------------------------------------- #


class TestResumeAt:
    def test_a_write_at_layer_3_with_taps_only_at_lm_head_resumes_at_3(self) -> None:
        groups = _groups(base_doc())
        patched = groups[("patched", "base")]
        assert patched.write_depth == 3
        assert patched.resume_at == 3

    def test_the_original_model_never_resumes(self) -> None:
        groups = _groups(base_doc())
        original = groups[("original_counterfactual", "counterfactual")]
        assert original.write_depth == PAST_BLOCKS
        assert original.resume_at == 0

    def test_a_tap_below_the_write_bounds_the_resume(self) -> None:
        """A read at layer 1 on the intervened model needs block 1 to run."""
        raw = base_doc()
        _with_tap_on_patched(raw, 1, "probe")
        assert _groups(raw)[("patched", "base")].resume_at == 1

    def test_a_campaign_tap_on_the_same_key_bounds_every_sharer(self) -> None:
        """Taps are not in the key, so a second point tapping layer 1 of
        the same patched forward merges into one group — and the merged
        group's resume depth is the shallowest of the union, exactly as its
        ``stop_after`` is the deepest."""
        plain = plan_point(parse_document(in_order(base_doc())))
        tapped_raw = base_doc()
        _with_tap_on_patched(tapped_raw, 1, "probe")
        tapped = plan_point(parse_document(in_order(tapped_raw)))
        own = {g.model: g for g in plain.groups}["patched"]
        assert own.resume_at == 3
        merged = {g.model: g for g in interned_groups([plain, tapped])}["patched"]
        assert merged.key == own.key
        assert merged.resume_at == 1

    def test_a_write_at_embeddings_never_resumes(self) -> None:
        raw = base_doc()
        raw["method"]["sites"]["tgt"] = {"component": "embeddings"}
        patched = _groups(raw)[("patched", "base")]
        assert patched.write_depth == 0
        assert patched.resume_at == 0

    def test_a_decoding_group_never_resumes(self) -> None:
        """Every decode step needs the whole stack, and the prefill's
        activations are not what a later point replays (§4)."""
        plan = plan_point(parse_document(probe_doc({"index": -1})))
        patched = next(g for g in plan.groups if g.model == "patched")
        assert patched.decode_depth > 0
        assert patched.write_depth == 3
        assert patched.resume_at == 0

    def test_a_decoding_sharer_pins_the_merged_group(self) -> None:
        plain = plan_point(parse_document(in_order(base_doc())))
        probing = plan_point(parse_document(probe_doc({"index": -1})))
        merged = {g.model: g for g in interned_groups([plain, probing])}["patched"]
        assert merged.decode_depth > 0
        assert merged.resume_at == 0

    def test_block_output_at_l_minus_1_and_block_input_at_l_differ(self) -> None:
        """Both name the residual between blocks 2 and 3, but a
        ``block_output`` write rides a forward hook *on block 2*, so block 2
        has to run: the conservative depth is 2, not 3."""
        out_raw = base_doc()
        out_raw["method"]["sites"]["tgt"] = {"component": "block_output", "layers": [2]}
        in_raw = base_doc()
        in_raw["method"]["sites"]["tgt"] = {"component": "block_input", "layers": [3]}
        assert _groups(out_raw)[("patched", "base")].resume_at == 2
        assert _groups(in_raw)[("patched", "base")].resume_at == 3

    def test_the_shallowest_of_several_writes_wins(self, env: ResolutionEnv) -> None:
        """Corpus 03's ``patched`` writes at layers 9, 10 and 11 and is tapped
        at ``block_input`` 12: it may start at 9. ``final`` writes at
        ``block_input`` 12 and is tapped at lm_head: 12."""
        loaded = compile_protocol(CORPUS_DIR / "03_path_patching_im.json", env=env)
        groups = {
            (g.model, g.input): g
            for g in plan_point(steps_of(loaded, env).documents[0]).groups
        }
        assert groups[("patched", "base")].write_depth == 9
        assert groups[("patched", "base")].resume_at == 9
        assert groups[("final", "base")].resume_at == 12
        assert (
            groups[("patched", "base")].base_key
            == groups[("final", "base")].base_key
            == groups[("original_base", "base")].key
        )

    def test_unexpanded_writes_never_resume_and_never_store(self) -> None:
        """A ``writes`` list still under a sweep wrapper is a write the plan
        cannot see. The permissive reading — "no write, so past every block"
        — would resume past it and store post-write residuals as
        un-intervened; the safe one is depth 0: no resume, nothing stored.
        (``fit_constant_models`` refuses the same shape outright; the plan
        is also asked about template documents, so this stays an answer.)"""
        raw = base_doc()
        raw["method"]["writes"]["other"] = {
            "site": "tgt",
            "pos": -1,
            "do": {"swap": "v_cf"},
        }
        raw["method"]["intervened_models"]["patched"]["writes"] = {
            "sweep": [["patch"], ["other"]]
        }
        patched = _groups(raw)[("patched", "base")]
        assert patched.write_depth == 0
        assert patched.resume_at == 0

    def test_an_unexpanded_site_component_never_resumes_and_never_stores(
        self,
    ) -> None:
        """The neighbouring door: a write whose *site* is still under a sweep
        wrapper. ``site_depth`` narrows a non-concrete component to the trunk
        — ``PAST_BLOCKS``, the permissive direction the unexpanded-``writes``
        case refuses — so the write's depth has to fail closed the same way,
        whichever half of the site is unexpanded."""
        raw = base_doc()
        raw["method"]["sites"]["tgt"]["component"] = {
            "sweep": ["block_output", "ln_final"]
        }
        patched = _groups(raw)[("patched", "base")]
        assert patched.write_depth == 0
        assert patched.resume_at == 0

    def test_a_write_past_every_block_leaves_the_bound_to_the_engine(self) -> None:
        """A write at ``ln_final`` touches no block; the plan says so with
        the same sentinel the tap order uses and lets the engine clamp it to
        the model's depth."""
        raw = base_doc()
        raw["method"]["sites"]["tgt"] = {"component": "ln_final"}
        patched = _groups(raw)[("patched", "base")]
        assert patched.write_depth == PAST_BLOCKS
        assert patched.resume_at == PAST_BLOCKS
