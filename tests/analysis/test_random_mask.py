"""``causalab.analysis.random_mask`` — the size-matched control for a DBM fit.

The bundles here are built by hand the way a fit writes them
(``neural/shared/outputs.py``): ``theta`` entries keyed by sweep coordinates,
an ArtifactIdentity at file level and an ``entries`` table in the header. What
is under test is the contract a control has to keep for the comparison to mean
anything — same count as the fit, same shape and address, a draw that a seed
reproduces — and the refusals that stop it being run on the wrong artifact.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

import pytest
import torch
from safetensors.torch import load_file

from causalab.analysis import random_mask
from causalab.io.step_io import StepError, stamp_tensor, write_tensor
from causalab.io.env import (
    build_artifact_identity,
    check_artifact_identity,
    read_safetensors_metadata,
)
from tests.step_scripts import run_step
from tests._helpers.paths import PROTOCOLS_DIR

#: What a gate fit stamps at file level (``execution.featurizer_identity``).
IDENTITY = {
    "model_key": "tiny-random/llama",
    "model_revision": "main",
    "model_dtype": "fp32",
    "site": json.dumps({"component": "block_output", "layers": [0]}, sort_keys=True),
    "dtype": "fp32",
    "trained_on": "weekdays/train",
    "engine": "pytorch_hooks",
    "commit": "deadbeef",
}


def _fitted(
    path: Path,
    thetas: Mapping[str, torch.Tensor],
    *,
    coords: Mapping[str, Mapping[str, object]] | None = None,
    extra: Mapping[str, str] | None = None,
) -> Path:
    """A bundle as a (possibly swept) fit writes it: one ``theta`` per key, the
    identity file-wide, the per-entry table in the header."""
    from safetensors.torch import save_file

    metadata = build_artifact_identity(**IDENTITY)
    metadata["entries"] = json.dumps(
        {
            key: {"slot": "theta", "coords": dict((coords or {}).get(key, {}))}
            for key in thetas
        },
        sort_keys=True,
    )
    metadata.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(dict(thetas), str(path), metadata=metadata)
    return path


def _draw(tmp_path: Path, source: Path, tag: str, **inputs) -> Path:
    out = tmp_path / tag / "gate.safetensors"
    run_step(random_mask, {"gate": source, **inputs}, {"gate": out})
    return out


def _count(theta: torch.Tensor) -> int:
    return int((theta > 0).sum())


# two fitted points with different mask sizes, the way a penalty sweep ends up
SWEPT = {
    "theta[l1=0.01]": torch.tensor([2.0, -1.0, 0.5, -3.0, 1.5, -0.5, 4.0, -2.0]),
    "theta[l1=0.1]": torch.tensor([-2.0, -1.0, 0.5, -3.0, -1.5, -0.5, -4.0, -2.0]),
}
SWEPT_COORDS = {"theta[l1=0.01]": {"l1": 0.01}, "theta[l1=0.1]": {"l1": 0.1}}


@pytest.mark.numerical_unit
def test_each_tensor_keeps_its_own_count_shape_and_dtype(tmp_path: Path) -> None:
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    drawn = load_file(str(_draw(tmp_path, source, "r", seed=0)))
    assert set(drawn) == set(SWEPT)
    for key, theta in SWEPT.items():
        assert drawn[key].shape == theta.shape
        assert drawn[key].dtype == theta.dtype
        assert _count(drawn[key]) == _count(theta)
        # decisive by construction: nothing but the two poles
        assert set(drawn[key].tolist()) <= {-1.0, 1.0}


@pytest.mark.numerical_unit
def test_the_draw_is_a_function_of_the_seed_and_the_entry(tmp_path: Path) -> None:
    """Each entry is drawn through its own generator, seeded from ``(seed,
    entry key)``: the same seed reproduces the draw, another seed moves it,
    and an entry's control is the same whatever else shares its bundle —
    ``theta[l1=0.1]`` drawn from the two-point fit equals ``theta[l1=0.1]``
    drawn from a bundle holding it alone, and equals the ``entry``-narrowed
    draw."""
    wide = torch.where(torch.arange(64) < 8, 1.0, -1.0)
    narrow = torch.where(torch.arange(64) % 5 == 0, 1.0, -1.0)
    coords = {"theta[l1=0.01]": {"l1": 0.01}, "theta[l1=0.1]": {"l1": 0.1}}
    both = _fitted(
        tmp_path / "both.safetensors",
        {"theta[l1=0.01]": wide, "theta[l1=0.1]": narrow},
        coords=coords,
    )
    alone = _fitted(
        tmp_path / "alone.safetensors",
        {"theta[l1=0.1]": narrow},
        coords={"theta[l1=0.1]": coords["theta[l1=0.1]"]},
    )
    first = load_file(str(_draw(tmp_path, both, "a", seed=3)))
    again = load_file(str(_draw(tmp_path, both, "b", seed=3)))
    other = load_file(str(_draw(tmp_path, both, "c", seed=4)))
    for key in coords:
        assert torch.equal(first[key], again[key])
        assert not torch.equal(first[key], other[key])
    assert _count(first["theta[l1=0.01]"]) == _count(other["theta[l1=0.01]"]) == 8
    # a control is a *different* set of units, not the fit under another name
    assert not torch.equal(first["theta[l1=0.01]"] > 0, wide > 0)
    # and two entries at one seed are two draws, not one draw twice
    assert not torch.equal(first["theta[l1=0.01]"] > 0, first["theta[l1=0.1]"] > 0)

    solo = load_file(str(_draw(tmp_path, alone, "d", seed=3)))["theta[l1=0.1]"]
    narrowed = load_file(str(_draw(tmp_path, both, "e", seed=3, entry={"l1": 0.1})))[
        "theta[l1=0.1]"
    ]
    assert torch.equal(first["theta[l1=0.1]"], solo)
    assert torch.equal(first["theta[l1=0.1]"], narrowed)


@pytest.mark.numerical_unit
def test_a_grouped_bundle_is_drawn_in_units_and_keeps_its_group(
    tmp_path: Path,
) -> None:
    """A ``group: head`` fit stores one ``theta`` per head (W3), so the count
    matched and the units drawn are heads — 8 entries for 8 heads of 32, never
    256 coordinates — and the ``group`` / ``group_map`` the fit stamped come
    through the header, so the apply document's map check sees the fit's."""
    heads, head_dim = 8, 32
    fitted = torch.tensor([2.0, -1.0, 0.5, -3.0, -1.5, -0.5, 4.0, -2.0])
    assert fitted.numel() == heads
    source = _fitted(
        tmp_path / "fit.safetensors",
        {"theta": fitted},
        extra={"group": "head", "group_map": json.dumps([heads, head_dim])},
    )
    target = _draw(tmp_path, source, "r", seed=0)
    drawn = load_file(str(target))["theta"]
    assert drawn.shape == (heads,)
    assert _count(drawn) == _count(fitted) == 3
    assert set(drawn.tolist()) <= {-1.0, 1.0}
    header = read_safetensors_metadata(target)
    assert header is not None
    assert header["group"] == "head"
    assert json.loads(header["group_map"]) == [heads, head_dim]
    assert header == read_safetensors_metadata(source)


@pytest.mark.numerical_unit
def test_unmatched_layers_keep_only_the_total(tmp_path: Path) -> None:
    """The weaker null: the budget survives, its distribution over tensors does
    not have to. Seeds are searched for a draw that actually moves a unit
    between tensors, so the test shows redistribution rather than assuming it."""
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    total = sum(_count(theta) for theta in SWEPT.values())
    per_tensor = {key: _count(theta) for key, theta in SWEPT.items()}
    moved = False
    for seed in range(8):
        drawn = load_file(
            str(_draw(tmp_path, source, f"s{seed}", seed=seed, match_layers=False))
        )
        assert sum(_count(t) for t in drawn.values()) == total
        for key, theta in SWEPT.items():
            assert drawn[key].shape == theta.shape
        moved |= any(_count(drawn[key]) != per_tensor[key] for key in SWEPT)
    assert moved


@pytest.mark.numerical_unit
def test_an_empty_mask_stays_empty(tmp_path: Path) -> None:
    """A fit that selected nothing has a control that selects nothing — the
    count is matched, not floored at one."""
    source = _fitted(tmp_path / "fit.safetensors", {"theta": -torch.ones(6)})
    drawn = load_file(str(_draw(tmp_path, source, "r", seed=0)))["theta"]
    assert torch.equal(drawn, -torch.ones(6))


@pytest.mark.numerical_unit
def test_a_full_mask_stays_full(tmp_path: Path) -> None:
    source = _fitted(tmp_path / "fit.safetensors", {"theta": torch.ones(6)})
    drawn = load_file(str(_draw(tmp_path, source, "r", seed=0)))["theta"]
    assert torch.equal(drawn, torch.ones(6))


@pytest.mark.unit
def test_the_header_survives_verbatim_and_the_runner_can_stamp_it(
    tmp_path: Path,
) -> None:
    """The control is addressable where the fit was: every header field —
    identity, entry table, and a field this script has never heard of — comes
    through unchanged, and the runner's stamp on top of it leaves an identity
    the apply document's check accepts exactly as it would the fit's."""
    source = _fitted(
        tmp_path / "fit.safetensors",
        SWEPT,
        coords=SWEPT_COORDS,
        extra={"group": "head"},
    )
    target = _draw(tmp_path, source, "r", seed=0)
    assert read_safetensors_metadata(target) == read_safetensors_metadata(source)

    step_identity = {"engine": "script"}
    stamp_tensor(target, step_identity, what="test")
    stamped = read_safetensors_metadata(target)
    assert stamped is not None
    check_artifact_identity(
        stamped,
        {k: v for k, v in IDENTITY.items() if k not in step_identity},
        what="apply",
    )
    assert stamped["engine"] == "script"
    assert json.loads(stamped["entries"]) == json.loads(
        read_safetensors_metadata(source)["entries"]  # type: ignore[index]
    )


@pytest.mark.unit
def test_a_single_entry_bundle_written_by_write_tensor_works(tmp_path: Path) -> None:
    """The un-swept case: one ``theta``, no entry table, identity at file level
    — what ``dbm.json`` writes and ``write_tensor`` reproduces."""
    source = tmp_path / "fit.safetensors"
    write_tensor(
        source, torch.tensor([1.0, -1.0, 2.0, -2.0]), slot="theta", identity=IDENTITY
    )
    drawn = load_file(str(_draw(tmp_path, source, "r", seed=0)))
    assert list(drawn) == ["theta"]
    assert _count(drawn["theta"]) == 2
    assert read_safetensors_metadata(tmp_path / "r/gate.safetensors") == (
        read_safetensors_metadata(source)
    )


@pytest.mark.unit
def test_entry_narrows_the_output_and_its_table(tmp_path: Path) -> None:
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    target = _draw(tmp_path, source, "r", seed=0, entry={"l1": 0.1})
    drawn = load_file(str(target))
    assert list(drawn) == ["theta[l1=0.1]"]
    assert _count(drawn["theta[l1=0.1]"]) == _count(SWEPT["theta[l1=0.1]"])
    table = json.loads(read_safetensors_metadata(target)["entries"])  # type: ignore[index]
    assert list(table) == ["theta[l1=0.1]"]


@pytest.mark.unit
def test_an_entry_matching_nothing_refuses(tmp_path: Path) -> None:
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    with pytest.raises(StepError, match="no 'theta' entry matches {l1=1}"):
        _draw(tmp_path, source, "r", seed=0, entry={"l1": 1})


@pytest.mark.unit
def test_a_missing_bundle_refuses(tmp_path: Path) -> None:
    with pytest.raises(StepError, match="does not exist"):
        _draw(tmp_path, tmp_path / "nowhere.safetensors", "r", seed=0)


@pytest.mark.numerical_unit
def test_a_trajectory_with_a_rotation_beside_the_gate_is_drawn_over_its_thetas(
    tmp_path: Path,
) -> None:
    """A fit that trained a rotation with its gate photographs both slots at
    every step; the control is drawn over the ``theta`` entries and writes only
    those — the rotation is the fit's, loaded from its own bundle."""
    from safetensors.torch import save_file

    thetas = {
        "theta[featurizer=gate,step=10]": torch.tensor([2.0, -1.0, 0.5, -3.0]),
        "theta[featurizer=gate,step=20]": torch.tensor([-2.0, -1.0, 0.5, -3.0]),
    }
    weights = {
        "weight[featurizer=rot,step=10]": torch.eye(4)[:, :2].contiguous(),
        "weight[featurizer=rot,step=20]": torch.eye(4)[:, 2:].contiguous(),
    }
    metadata = build_artifact_identity(**IDENTITY)
    table = {
        key: {"slot": "theta", "coords": {"featurizer": "gate", "step": s}}
        for key, s in zip(thetas, (10, 20))
    }
    table.update(
        {
            key: {"slot": "weight", "coords": {"featurizer": "rot", "step": s}}
            for key, s in zip(weights, (10, 20))
        }
    )
    metadata["entries"] = json.dumps(table, sort_keys=True)
    source = tmp_path / "trajectory.safetensors"
    save_file({**thetas, **weights}, str(source), metadata=metadata)
    out = _draw(tmp_path, source, "c", seed=0)
    drawn = load_file(str(out))
    assert set(drawn) == set(thetas)
    for key, theta in thetas.items():
        assert _count(drawn[key]) == _count(theta)
    header = read_safetensors_metadata(out)
    assert header is not None
    assert set(json.loads(header["entries"])) == set(thetas)


@pytest.mark.numerical_unit
def test_a_complement_draw_is_the_same_count_from_the_dropped_units(
    tmp_path: Path,
) -> None:
    """NeuroSurgeon's ``complement_sampled``: the sharper null for a small
    fitted set — disjoint from the fit by construction, the count matched."""
    theta = torch.tensor([3.0, -1.0, 2.0, -3.0, -1.5, -0.5, -4.0, -2.0])  # keeps 2 of 8
    source = _fitted(tmp_path / "fit.safetensors", {"theta": theta})
    drawn = load_file(str(_draw(tmp_path, source, "c", seed=0, draw="complement")))[
        "theta"
    ]
    assert _count(drawn) == 2
    kept_by_fit = {i for i, v in enumerate(theta.tolist()) if v > 0}
    kept_by_control = {i for i, v in enumerate(drawn.tolist()) if v > 0}
    assert kept_by_fit.isdisjoint(kept_by_control)
    assert set(drawn.tolist()) <= {-1.0, 1.0}
    # a function of the seed, like the uniform draw
    again = load_file(str(_draw(tmp_path, source, "d", seed=0, draw="complement")))
    assert torch.equal(again["theta"], drawn)
    other = load_file(str(_draw(tmp_path, source, "e", seed=1, draw="complement")))
    assert not torch.equal(other["theta"], drawn)


@pytest.mark.numerical_unit
def test_a_complement_draw_refuses_a_fit_that_kept_more_than_half(
    tmp_path: Path,
) -> None:
    theta = torch.tensor([3.0, 1.0, 2.0, -3.0])  # keeps 3 of 4: one unit to draw from
    source = _fitted(tmp_path / "fit.safetensors", {"theta": theta})
    with pytest.raises(StepError, match="more than half"):
        _draw(tmp_path, source, "c", seed=0, draw="complement")


@pytest.mark.numerical_unit
@pytest.mark.parametrize(
    "inputs", [{"draw": "shuffled"}, {"draw": "complement", "match_layers": False}]
)
def test_an_unknown_or_unmatched_complement_draw_refuses(
    tmp_path: Path, inputs
) -> None:
    source = _fitted(tmp_path / "fit.safetensors", dict(SWEPT), coords=SWEPT_COORDS)
    with pytest.raises(StepError, match="draw"):
        _draw(tmp_path, source, "c", seed=0, **inputs)


@pytest.mark.numerical_unit
def test_a_bundle_that_is_not_a_gate_refuses(tmp_path: Path) -> None:
    """A PCA basis is a ``weight``; a harvest is keyed by its read. Neither has
    a ``θ > 0`` count to match, so the script says what it found rather than
    drawing a mask over the wrong thing."""
    source = tmp_path / "basis.safetensors"
    write_tensor(source, torch.eye(4), slot="weight")
    with pytest.raises(StepError, match=r"not a fitted gate bundle.*\['weight'\]"):
        _draw(tmp_path, source, "r", seed=0)


@pytest.mark.unit
def test_a_selected_tensor_instead_of_the_bundle_refuses(tmp_path: Path) -> None:
    """A reference with its own ``entry`` hands the script a tensor and drops
    the header the control needs; the refusal says how to write it instead."""
    with pytest.raises(StepError, match="own 'entry' input"):
        run_step(
            random_mask,
            {"gate": torch.ones(4), "seed": 0},
            {"gate": tmp_path / "gate.safetensors"},
        )


@pytest.mark.unit
def test_a_missing_seed_refuses(tmp_path: Path) -> None:
    source = _fitted(tmp_path / "fit.safetensors", {"theta": torch.ones(4)})
    with pytest.raises(StepError, match="'seed' is required"):
        run_step(random_mask, {"gate": source}, {"gate": tmp_path / "g.safetensors"})


@pytest.mark.unit
@pytest.mark.parametrize("seed", ["3", 3.0, True, None])
def test_a_seed_that_is_not_an_integer_refuses(tmp_path: Path, seed: object) -> None:
    """A recorded seed reproduces a recorded control, so ``"3"`` and ``3.0``
    are refused rather than coerced to the integer they happen to resemble —
    and so is ``true``, which Python would otherwise read as 1."""
    source = _fitted(tmp_path / "fit.safetensors", {"theta": torch.ones(4)})
    with pytest.raises(StepError, match="'seed' must be an integer"):
        _draw(tmp_path, source, "r", seed=seed)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_match_layers_that_is_not_a_bool_refuses(tmp_path: Path, value: object) -> None:
    """``"false"`` is truthy; coercing it would draw the layer-matched null
    while the document asked for the other one."""
    source = _fitted(tmp_path / "fit.safetensors", {"theta": torch.ones(4)})
    with pytest.raises(StepError, match="'match_layers' must be true or false"):
        _draw(tmp_path, source, "r", seed=0, match_layers=value)


@pytest.mark.unit
def test_an_entry_that_is_not_a_mapping_refuses(tmp_path: Path) -> None:
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    with pytest.raises(StepError, match="'entry' maps coordinate names to values"):
        _draw(tmp_path, source, "r", seed=0, entry="theta[l1=0.1]")


@pytest.mark.unit
def test_an_entry_naming_another_slot_refuses(tmp_path: Path) -> None:
    """A gate bundle holds ``theta`` alone, so an ``entry`` selecting a
    ``weight`` cannot match anything here and says so by slot rather than
    falling through to the no-match refusal."""
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    with pytest.raises(StepError, match="names slot 'weight', but a gate bundle"):
        _draw(tmp_path, source, "r", seed=0, entry={"slot": "weight", "l1": 0.1})


# --------------------------------------------------------------------------- #
# end to end: a fit, its control, and the apply document scoring the control
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_the_apply_document_scores_the_control_at_the_fits_address(
    tmp_path: Path,
) -> None:
    """The reason the header is preserved: ``dbm_apply.json`` loads the control
    exactly as it loads the fit — same identity check, same width check — and
    scores it on the same split, which is the number the fit is compared to."""
    from causalab.cli import main
    from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
    from tests.protocol._env import FIXTURES, fixture_input_overrides
    from tests.tables import frame as table_frame

    protocols = PROTOCOLS_DIR
    tiny = {"model.key": TINY_LLAMA, "model.dtype": "fp32", "sites.target.layers": 0}
    # dbm.json names the shipped weekdays table, whose answers tiny-random cannot
    # spell as single tokens ([P2]); the fit reads the 4-row fixture instead, as
    # every tiny-scale run of a shipped document does (tests/protocol/_env.py)
    fixture_inputs = fixture_input_overrides(
        json.loads((protocols / "dbm.json").read_text())
    )
    train_split = "weekdays/data#train"
    assert fixture_inputs["data.base.dataset"] == train_split
    workflow = {
        "version": "1",
        "description": "fit a DBM gate, draw its size-matched control, apply both",
        "output_dir": "dbm",
        "steps": {
            "fit": {
                "type": "intervention_protocol",
                "document": str(protocols / "dbm.json"),
                "set": {
                    **tiny,
                    **fixture_inputs,
                    "train.steps": {"epochs": 1},
                    "train.batch": {"pairs": 2},
                },
            },
            "control": {
                "type": "script",
                "script": {"module": "causalab.analysis.random_mask"},
                "inputs": {
                    "gate": {"step": "fit", "file": "gate.safetensors"},
                    "seed": 0,
                },
                "outputs": {"gate": "gate.safetensors"},
            },
            "apply_control": {
                "type": "intervention_protocol",
                "document": str(protocols / "dbm_apply.json"),
                "set": {
                    **tiny,
                    "featurizers.gate.file_path": "control/gate.safetensors",
                    # the fit's explicit fixture split, so the control's
                    # score is the number the fit's iia.json is compared to
                    "data.base.dataset": train_split,
                    "data.counterfactual.dataset": train_split,
                },
            },
        },
    }
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    document = tmp_path / "wf.json"
    document.write_text(json.dumps(workflow, indent=2))
    out = tmp_path / "run"
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(document),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    run = out / "dbm"
    fitted = load_file(str(run / "fit/gate.safetensors"))["theta"]
    control = load_file(str(run / "control/gate.safetensors"))["theta"]
    assert control.shape == fitted.shape
    assert _count(control) == _count(fitted)
    manifest = json.loads((run / "workflow.json").read_text())
    assert manifest["steps"]["apply_control"]["status"] == "completed"
    table, split = train_split.split("#")
    rows = json.loads((FIXTURES / "data" / f"{table}.json").read_text())
    pairs = sum(row["split"] == split for row in rows)
    assert pairs > 0
    assert len(table_frame(run / "apply_control/iia.json")) == pairs


@pytest.mark.numerical_unit
def test_a_clamp_bundle_is_matched_at_its_own_threshold_and_written_on_its_poles(
    tmp_path: Path,
) -> None:
    """A `clamp` gate's hard mask is `θ > ½` (§2.5 parametrization), so the
    control matches *that* count — three of six here, not the five a `θ > 0`
    reading would give — and is written on `{0, 1}` so it reloads under a clamp
    document as a decisive mask."""
    thetas = {"theta": torch.tensor([0.9, 0.1, 0.51, 0.49, 1.0, 0.0])}
    source = _fitted(
        tmp_path / "fit.safetensors", thetas, extra={"parametrization": "clamp"}
    )
    drawn = load_file(str(_draw(tmp_path, source, "r", seed=0)))["theta"]
    assert int((drawn > 0.5).sum()) == 3
    assert set(drawn.tolist()) <= {0.0, 1.0}
    header = read_safetensors_metadata(tmp_path / "r/gate.safetensors")
    assert header is not None and header["parametrization"] == "clamp"


@pytest.mark.unit
def test_a_hard_concrete_bundle_is_matched_at_its_stamped_stretch(
    tmp_path: Path,
) -> None:
    """A `hard_concrete` gate's hard mask is `θ > logit((½−γ)/(ζ−γ))` at the
    stretch the fit stamped (§2.5): at `[-0.1, 1.5]` that is ≈ −0.51, so the
    control matches *that* count — four of six here, not the three a `θ > 0`
    reading would give — through the same function the gate uses, and is
    written one unit either side of it."""
    from causalab.protocol.schema import hard_concrete_threshold

    threshold = hard_concrete_threshold((-0.1, 1.5))
    assert threshold == pytest.approx(-0.5108, abs=1e-3)
    thetas = {"theta": torch.tensor([0.9, -0.3, 0.1, -0.7, -0.2, -2.0])}
    source = _fitted(
        tmp_path / "fit.safetensors",
        thetas,
        extra={"parametrization": "hard_concrete", "stretch": "[-0.1, 1.5]"},
    )
    drawn = load_file(str(_draw(tmp_path, source, "r", seed=0)))["theta"]
    assert int((drawn > threshold).sum()) == 4
    assert sorted(set(drawn.tolist())) == pytest.approx(
        [threshold - 1.0, threshold + 1.0]
    )
    header = read_safetensors_metadata(tmp_path / "r/gate.safetensors")
    assert header is not None and json.loads(header["stretch"]) == [-0.1, 1.5]


@pytest.mark.unit
def test_a_hard_concrete_bundle_without_a_stretch_refuses(tmp_path: Path) -> None:
    """Every protocol-written hard-concrete bundle stamps its stretch (the
    identity check refuses a missing key), so one without it was not written
    by a fit — and reading it at 0 would be a plausible-looking wrong null."""
    source = _fitted(
        tmp_path / "fit.safetensors",
        {"theta": torch.tensor([0.5, -0.5])},
        extra={"parametrization": "hard_concrete"},
    )
    with pytest.raises(StepError, match="no 'stretch'"):
        _draw(tmp_path, source, "r", seed=0)


@pytest.mark.numerical_unit
def test_top_k_matches_the_control_to_the_cut_not_the_threshold(tmp_path: Path) -> None:
    """A document reading the gate out at ``top_k`` (§2.5) scores a cut, not
    the map's split; its control keeps that many units per entry, whatever the
    threshold would have counted — and a cut past the units is refused."""
    source = _fitted(tmp_path / "fit.safetensors", SWEPT, coords=SWEPT_COORDS)
    drawn = load_file(str(_draw(tmp_path, source, "k", seed=0, top_k=3)))
    for key, theta in SWEPT.items():
        assert _count(theta) != 3  # the threshold counts 4 and 1 here
        assert _count(drawn[key]) == 3
        assert set(drawn[key].tolist()) <= {-1.0, 1.0}
    assert (
        _count(
            load_file(str(_draw(tmp_path, source, "z", seed=0, top_k=0)))[
                "theta[l1=0.1]"
            ]
        )
        == 0
    )
    with pytest.raises(StepError, match="exceeds the 8 units"):
        _draw(tmp_path, source, "big", seed=0, top_k=9)
    with pytest.raises(StepError, match="non-negative integer"):
        _draw(tmp_path, source, "bad", seed=0, top_k=-1)


@pytest.mark.numerical_unit
def test_a_budget_bundle_is_matched_at_the_consumer_cut(tmp_path: Path) -> None:
    """A budget gate's theta is a ranking with no threshold (§2.5): the
    control's count is the consumer's ``top_k`` and nothing else, so without it
    the draw is refused rather than counted at a split the map does not have."""
    theta = {"theta": torch.tensor([2.0, -1.0, 0.5, -3.0, 1.5, -0.5, 4.0, -2.0])}
    source = _fitted(
        tmp_path / "budget.safetensors", theta, extra={"parametrization": "budget"}
    )
    with pytest.raises(StepError, match="budget gate"):
        _draw(tmp_path, source, "none", seed=0)
    drawn = load_file(str(_draw(tmp_path, source, "cut", seed=0, top_k=3)))["theta"]
    assert int((drawn > 0).sum()) == 3
    assert (
        read_safetensors_metadata(tmp_path / "cut/gate.safetensors")["parametrization"]
        == "budget"
    )


@pytest.mark.numerical_unit
def test_a_bundle_fitted_in_a_pool_is_refused(tmp_path: Path) -> None:
    """§2.5 `pool`: a pooled θ ranks against its co-members, and the stamp is
    read per entry over file-wide, the `group` precedent — a control from one
    member alone would be sized to a share of the cut it cannot know."""
    source = _fitted(
        tmp_path / "fit.safetensors",
        {"theta": torch.tensor([2.0, -1.0, 0.5, -3.0])},
        extra={"parametrization": "budget", "pool": "mib", "pool_units": "157"},
    )
    with pytest.raises(StepError, match="fitted in pool"):
        _draw(tmp_path, source, "pooled", seed=0, top_k=2)
    # per-entry stamp, file-wide absent: the same refusal
    per_entry = _fitted(
        tmp_path / "fit2.safetensors",
        {"theta": torch.tensor([2.0, -1.0, 0.5, -3.0])},
        extra={"parametrization": "budget"},
    )
    import json as _json

    from safetensors.torch import save_file

    metadata = read_safetensors_metadata(per_entry)
    assert metadata is not None
    entries = _json.loads(metadata["entries"])
    entries["theta"]["pool"] = "mib"
    metadata = {**metadata, "entries": _json.dumps(entries, sort_keys=True)}
    save_file(
        {"theta": torch.tensor([2.0, -1.0, 0.5, -3.0])},
        str(per_entry),
        metadata=metadata,
    )
    with pytest.raises(StepError, match="fitted in pool"):
        _draw(tmp_path, per_entry, "pooled2", seed=0, top_k=2)


@pytest.mark.unit
def test_a_boundary_bundle_is_refused_naming_the_learned_rank(tmp_path: Path) -> None:
    """A `boundary` gate's theta is one boundary fraction over an ordered basis
    (§2.5), not a unit per entry: there is no set of units to resample, and
    the control a boundary fit wants is a random basis at the learned rank
    ⌈θ · width⌉ — the refusal names the fraction and where the rank is."""
    source = _fitted(
        tmp_path / "fit.safetensors",
        {"theta": torch.tensor([0.5])},
        extra={"parametrization": "boundary"},
    )
    with pytest.raises(StepError, match="random basis at the learned rank") as err:
        _draw(tmp_path, source, "r", seed=0)
    assert "θ = 0.500" in str(err.value) and "random_subspace_control" in str(err.value)
