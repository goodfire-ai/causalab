"""A band site run on tiny-random (spec §2.4 ``layers``).

The mirror of ``test_band_patch_run.py``: that file runs a band as N
one-layer sites listed in one ``intervened_models`` entry; this one runs the
same band as **one site** whose ``layers`` spans them — one read, one write —
and asserts the two are the same intervention to the bit. The resolver's half
(``resolve_band`` fans the site out to one module per member) is here too.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from causalab.neural.shared.sites import resolve_band, resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.pipeline import compile_protocol
from causalab.neural.shared.plan import plan_point
from causalab.protocol.lowering import lower_bands
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import SiteSpec
from causalab.tasks import TASKS_ROOT

from tests.neural.engines.pytorch_hooks.test_band_patch_run import _band_doc, _run
from tests.protocol._docs import UNWRITTEN, saved
from tests.protocol._env import FIXTURES, steps_of
from tests.tables import frame as table_frame


pytestmark = pytest.mark.smoke


def _one_site_band_doc(layers: list[int], name: str = "wide") -> dict:
    """The band of ``_band_doc({name: layers})`` as one site."""
    doc = _band_doc({name: layers})
    method = doc["method"]
    method["sites"] = {
        "a": {"component": "attention_output", "layers": layers},
        "lm_head": {"component": "lm_head"},
    }
    method["reads"] = {
        "v_a": {"site": "a", "pos": "tap"},
        f"logits_{name}": method["reads"][f"logits_{name}"],
    }
    method["writes"] = {"w": {"site": "a", "pos": "tap", "do": {"swap": "v_a"}}}
    method["intervened_models"] = {
        UNWRITTEN: {"input": "counterfactual", "reads": ["v_a"]},
        name: {"input": "base", "reads": [f"logits_{name}"], "writes": ["w"]},
    }
    return doc


@pytest.fixture(scope="module")
def env() -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data", fallback_roots=(TASKS_ROOT,)),
        artifacts=FileArtifacts(root=FIXTURES / "artifacts"),
    )


# --------------------------------------------------------------------------- #
# the resolver: one module per member
# --------------------------------------------------------------------------- #


def test_resolve_band_is_one_module_per_member(llama_bundle) -> None:
    spec = SiteSpec(component="attention_output", layers=(0, 1))
    members = resolve_band(llama_bundle, spec)
    assert [m.layer for m in members] == [0, 1]
    assert members[0].module is not members[1].module
    assert {m.component for m in members} == {"attention_output"}
    one = resolve_site(
        llama_bundle, SiteSpec(component="attention_output", layers=(1,))
    )
    assert members[1].module is one.module and members[1] == one
    assert resolve_band(
        llama_bundle, SiteSpec(component="attention_output", layers=(1,))
    ) == (one,)
    assert resolve_band(llama_bundle, SiteSpec(component="lm_head")) == (
        resolve_site(llama_bundle, SiteSpec(component="lm_head")),
    )
    with pytest.raises(ProtocolError, match="one module per layer"):
        resolve_site(llama_bundle, spec)


# --------------------------------------------------------------------------- #
# the run: a one-site band is its hand-written twin, to the bit
# --------------------------------------------------------------------------- #


def test_a_one_site_band_plans_as_its_hand_written_twin(env) -> None:
    hand = steps_of(
        compile_protocol(_band_doc({"wide": [0, 1]}), env=env, base_dir=FIXTURES), env
    )
    band = steps_of(
        compile_protocol(_one_site_band_doc([0, 1]), env=env, base_dir=FIXTURES), env
    )
    hand, band = hand.documents[0], band.documents[0]
    assert band.sites["a"].layers == (0, 1)
    hand_plan, band_plan = plan_point(hand), plan_point(band)
    assert band_plan.num_forwards == hand_plan.num_forwards == 2
    for hg, bg in zip(hand_plan.groups, band_plan.groups):
        assert (hg.model, hg.input) == (bg.model, bg.input)
        assert {t.depth for t in hg.taps} == {t.depth for t in bg.taps}
        assert hg.write_depth == bg.write_depth and hg.resume_at == bg.resume_at
    low = lower_bands(band)
    assert sorted(low.sites) == ["a[layers=0]", "a[layers=1]", "lm_head"]
    assert low.intervened_models["wide"].writes == ("w[layers=0]", "w[layers=1]")


def test_a_one_site_band_reproduces_the_hand_written_band_to_the_bit(
    tmp_path: Path,
) -> None:
    """The identity the band shape rests on: one site over both layers of
    tiny-random is, bit for bit, the two-write ``intervened_models`` entry."""
    hand = _run(tmp_path / "hand", _band_doc({"wide": [0, 1]}))
    band = _run(tmp_path / "band", _one_site_band_doc([0, 1]))
    assert list(table_frame(band / "iia_wide.json")["value"]) == pytest.approx(
        list(table_frame(hand / "iia_wide.json")["value"]), abs=0.0
    )


def test_a_one_layer_band_is_the_scalar_site_to_the_bit(tmp_path: Path) -> None:
    """T-a on the engine: ``layers: [0]`` runs as the one-layer site."""
    hand = _run(tmp_path / "hand", _band_doc({"narrow": [0]}))
    band = _run(tmp_path / "band", _one_site_band_doc([0], name="narrow"))
    assert list(table_frame(band / "iia_narrow.json")["value"]) == pytest.approx(
        list(table_frame(hand / "iia_narrow.json")["value"]), abs=0.0
    )


def test_a_band_lands_on_every_member(tmp_path: Path) -> None:
    """The second layer's write lands: the band differs from its first
    member alone — the premise of scanning bands (asserted as inequality,
    tiny-random's weights being noise)."""
    out = _run(tmp_path, _one_site_band_doc([0, 1]))
    narrow = _run(tmp_path / "narrow", _one_site_band_doc([0]))
    wide = list(table_frame(out / "iia_wide.json")["value"])
    assert len(wide) == 2
    assert wide != list(table_frame(narrow / "iia_wide.json")["value"])


def test_a_saved_band_read_is_refused_by_name_before_any_forward(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints the lowering's refusal and exits 1; nothing is written."""
    from causalab.cli import main

    doc = _one_site_band_doc([0, 1])
    doc["method"]["save"].append(saved("v_a", UNWRITTEN, "v.safetensors"))
    path = tmp_path / "doc.json"

    path.write_text(json.dumps(doc))
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(FIXTURES / "artifacts"),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code != 0
    err = capsys.readouterr().err
    assert "refused:" in err and "save entry 'v_a'" in err and "layers 0..1" in err
    assert not (tmp_path / "run" / "iia_wide.json").exists()  # no table, no forward


def test_the_migrated_band_preset_still_loads(env) -> None:
    """The shipped ``attention_band_patch.json`` after the rename: eight
    one-layer ``at_once`` members (L9-L16 on the 28-layer model), every site
    a one-layer band."""
    from tests.neural.engines.pytorch_hooks.test_band_patch_run import PRESET

    doc = steps_of(compile_protocol(PRESET, env=env), env).documents[0]
    assert all(
        spec.layers == (int(name[1:]),)
        for name, spec in doc.sites.items()
        if name != "lm_head"
    )
    assert copy.deepcopy(doc.sites["a9"].layers) == (9,)
