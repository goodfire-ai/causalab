"""Checks specific to the ``mcqa_components_dbm`` paper package.

``tests/demos/test_papers.py`` checks what every package shares. This file
checks the claims the package's page and scripts make on top of that:

* the table holds the onboarding tutorial's two draws, row for row;
* the 56-gate fit and its apply spell one site, one gate, one read and one
  write per (component, layer), and the l1 term, ``params``, the anneal and
  ``save`` name every gate once;
* every apply step of the workflow loads each gate from its own fit step;
* a control bundle of ``check_controls.py`` keeps the fitted bundle's header
  and sets ``theta`` to the hard split it names, and the script stops when a
  replay does not reproduce its ``apply`` step;
* ``onboarding_values.json`` records the current sha256 of every onboarding
  file it copies values from, so a rerun of onboarding 09 or 11 fails here
  until the copied values are checked again;
* every copied value equals the cell of the onboarding plotted-values file
  that holds it;
* the Original's copy of onboarding 09's scan goes stale under
  ``copy_original.py --check`` when its source changes, and the committed
  ``components_plotted.json`` draws the Original from that copy;
* the two DBM figures draw the committed ``components_plotted.json``: the
  sweep dots are its held-out scores, and the mask is the ``CHOSEN`` fit's
  kept gates or heads.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
NAME = "mcqa_components_dbm"
PROTOCOLS = PAPERS / "protocols"
SCRIPTS = PAPERS / "workflows" / "scripts" / NAME
ONBOARDING = REPO / "demos" / "onboarding_tutorial" / "artifacts" / "data" / "mcqa"
UNITS = [(short, layer) for short in ("attn", "mlp") for layer in range(28)]
COMPONENT = {"attn": "attention_output", "mlp": "mlp_output"}


def _load(name: str) -> dict:
    return json.loads((PROTOCOLS / f"{NAME}_{name}.json").read_text())


def _script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{NAME}_{name}", SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_holds_the_onboarding_draws() -> None:
    """``train`` is onboarding's ``train_n128_s1`` and ``test`` its
    ``test_n64_s2``, every column equal but ``split``."""
    rows = json.loads((PAPERS / "artifacts" / "data" / NAME / "data.json").read_text())

    def strip(row: dict) -> dict:
        return {k: v for k, v in row.items() if k != "split"}

    for split, source in (("train", "train_n128_s1"), ("test", "test_n64_s2")):
        ours = [strip(r) for r in rows if r["split"] == split]
        theirs = [
            strip(r) for r in json.loads((ONBOARDING / f"{source}.json").read_text())
        ]
        assert ours == theirs, split


@pytest.mark.parametrize(
    "source",
    sorted(
        json.loads(
            (
                PAPERS / "artifacts" / "data" / NAME / "onboarding_values.json"
            ).read_text()
        )["sources"]
    ),
)
def test_the_onboarding_sources_match_their_recorded_hash(source: str) -> None:
    """The page quotes onboarding values copied by hand; the recorded sha256
    says which version of the onboarding file they were copied from."""
    values = PAPERS / "artifacts" / "data" / NAME / "onboarding_values.json"
    entry = json.loads(values.read_text())["sources"][source]
    digest = hashlib.sha256((REPO / entry["path"]).read_bytes()).hexdigest()
    assert digest == entry["sha256"], (
        f"{entry['path']} changed since its values were copied into "
        f"{values.relative_to(REPO)}: check the copied values, then record {digest}"
    )


def test_the_onboarding_records_match_the_committed_grids() -> None:
    """Every copied value equals, to the three decimals the pages print, the
    cell of the plotted-values file its onboarding page links: 09's component
    grid and 11's head grid."""
    output = REPO / "demos" / "onboarding_tutorial" / "artifacts" / "output"
    grid09 = json.loads(
        (
            output / "09_components" / "grid" / "09_components_component_iia.json"
        ).read_text()
    )
    grid11 = json.loads(
        (output / "11_attention" / "grid" / "11_attention_head_iia.json").read_text()
    )
    committed = {
        ("09", r["sites.target.component"], r["sites.target.layers"], None): r["value"]
        for r in grid09
    } | {
        (
            "11",
            "attention_premix",
            r["sites.contribution.layers"],
            r["sites.contribution.head"],
        ): r["value"]
        for r in grid11
    }
    records = json.loads(
        (PAPERS / "artifacts" / "data" / NAME / "onboarding_values.json").read_text()
    )["records"]
    wrong = {
        key: (r["iia"], round(committed[key], 3))
        for r in records
        if round(
            committed[key := (r["source"], r["component"], r["layer"], r["head"])], 3
        )
        != r["iia"]
    }
    assert wrong == {}, "record: (copied, committed)"


@pytest.mark.parametrize("name", ["fit", "apply"])
def test_one_site_gate_read_and_write_per_component_and_layer(name: str) -> None:
    method = _load(name)["method"]
    sites = {k: v for k, v in method["sites"].items() if k != "lm_head"}
    assert sites == {
        f"{short}{layer}": {"component": COMPONENT[short], "layers": [layer]}
        for short, layer in UNITS
    }
    for short, layer in UNITS:
        site, gate = f"{short}{layer}", f"g_{short}{layer}"
        featurizer = method["featurizers"][gate]
        assert featurizer["kind"] == "gate" and featurizer["group"] == "site"
        if name == "apply":
            assert featurizer["file_path"] == f"fit/{gate}.safetensors"
        assert method["reads"][f"v_{site}"] == {
            "site": site,
            "pos": "slot",
            "featurizer": gate,
        }
        assert method["writes"][f"mask_{site}"] == {
            "site": site,
            "pos": "slot",
            "featurizer": gate,
            "do": {"swap": f"v_{site}"},
        }
    assert len(method["featurizers"]) == len(UNITS)
    models = method["intervened_models"]
    assert set(models["masked"]["writes"]) == set(method["writes"])
    assert set(models["original_counterfactual"]["reads"]) == set(method["reads"]) - {
        "logits"
    }


def test_the_fit_trains_and_saves_every_gate_once() -> None:
    method = _load("fit")["method"]
    gates = [f"g_{short}{layer}" for short, layer in UNITS]
    train = method["train"]
    assert train["objective"]["l1"]["l1"] == gates
    assert train["params"] == gates
    assert list(train["anneal"]) == [f"{g}.theta.temperature" for g in gates]
    saved = [e for e in method["save"] if "value" in e]
    assert [(e["value"], e["site"], e["file_path"]) for e in saved] == [
        (g, g.removeprefix("g_"), f"{g}.safetensors") for g in gates
    ]


def test_the_two_variants_share_the_train_block() -> None:
    """The head variant differs from the component variant in the gate layout
    only: the training settings are one block."""
    components = _load("fit")["method"]["train"]
    heads = _load("head_fit")["method"]["train"]
    for key in ("optimizer", "steps", "batch", "precision", "seed"):
        assert components[key] == heads[key], key
    assert components["objective"]["ce"] == heads["objective"]["ce"]
    assert components["objective"]["l1"]["weight"] == heads["objective"]["l1"]["weight"]


def test_every_apply_step_loads_its_own_fit() -> None:
    steps = json.loads((PAPERS / "workflows" / f"{NAME}.json").read_text())["steps"]
    gates = [f"g_{short}{layer}" for short, layer in UNITS]
    for suffix in ("_mid", "_hi"):
        assert steps[f"apply{suffix}"]["set"] == {
            f"featurizers.{g}.file_path": f"fit{suffix}/{g}.safetensors" for g in gates
        }
        assert steps[f"head_apply{suffix}"]["set"] == {
            "featurizers.gate.file_path": f"head_fit{suffix}/gate.safetensors"
        }
        weight = steps[f"fit{suffix}"]["set"]["train.objective.l1.weight"]
        assert steps[f"head_fit{suffix}"]["set"]["train.objective.l1.weight"] == weight


def test_a_control_bundle_is_the_fit_at_the_named_split(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors import safe_open
    from safetensors.torch import save_file

    controls = _script("check_controls")
    fit = tmp_path / "fit_mid"
    fit.mkdir()
    metadata = {"group": "head", "model_key": "fixture", "produced_by": "0" * 64}
    save_file(
        {"theta": torch.tensor([0.4, -0.2, 0.1, -0.7])},
        str(fit / "gate.safetensors"),
        metadata=metadata,
    )
    (fit / "rank.json").write_text(
        json.dumps(
            [
                {"featurizer": "gate", "unit": u, "theta": t, "hard": t > 0}
                for u, t in enumerate([0.4, -0.2, 0.1, -0.7])
            ]
        )
    )
    units = controls.fitted_units(fit)
    assert [(u, kept) for _, u, kept in units] == [
        (0, True),
        (1, False),
        (2, True),
        (3, False),
    ]
    controls.write_control(
        fit, tmp_path / "control", "head_fit", {("gate.safetensors", 1)}
    )
    written = tmp_path / "control" / "head_fit" / "gate.safetensors"
    with safe_open(str(written), framework="pt") as fh:
        assert dict(fh.metadata()) == metadata
        assert fh.get_tensor("theta").tolist() == [-1.0, 1.0, -1.0, -1.0]


def test_the_figure_reads_a_gate_as_its_component_and_layer(tmp_path: Path) -> None:
    """``figures.py`` takes the component and layer from the gate's name:
    ``g_attn22`` is layer 22's attention output, ``g_mlp3`` layer 3's MLP."""
    pytest.importorskip("pandas")
    figures = _script("figures")
    (tmp_path / "rank.json").write_text(
        json.dumps(
            [
                {"featurizer": "g_mlp3", "unit": 0, "theta": -0.5, "hard": False},
                {"featurizer": "g_attn22", "unit": 0, "theta": 0.6, "hard": True},
            ]
        )
    )
    table = figures.gates(tmp_path, 0.3)
    rows = table[["l1", "component", "layer", "value", "kept"]].to_dict("records")
    assert rows == [
        {
            "l1": 0.3,
            "component": "attention_output",
            "layer": 22,
            "value": 0.6,
            "kept": True,
        },
        {
            "l1": 0.3,
            "component": "mlp_output",
            "layer": 3,
            "value": -0.5,
            "kept": False,
        },
    ]


def _fixture_run(tmp_path: Path, torch, save_file, replay_off: str) -> Path:
    """A run tree with every fit and apply step ``check`` reads: one
    two-unit gate per fit, and an apply table whose IIA is 0.5, or 0.75 for
    the mask ``replay_off`` names."""
    controls = _script("check_controls")
    run = tmp_path / "run"
    for variant, (prefix, _, _) in controls.VARIANTS.items():
        for suffix in controls.ARMS:
            fit = run / f"{prefix}fit{suffix}"
            fit.mkdir(parents=True)
            save_file(
                {"theta": torch.tensor([0.4, -0.2])},
                str(fit / "gate.safetensors"),
                metadata={"group": "head"},
            )
            (fit / "rank.json").write_text(
                json.dumps(
                    [
                        {"featurizer": "gate", "unit": 0, "theta": 0.4, "hard": True},
                        {"featurizer": "gate", "unit": 1, "theta": -0.2, "hard": False},
                    ]
                )
            )
            apply = run / f"{prefix}apply{suffix}"
            apply.mkdir()
            value = 0.75 if f"{variant}{suffix}" == replay_off else 0.5
            (apply / "iia.json").write_text(json.dumps([{"value": value}]))
    return run


@pytest.mark.parametrize("replay_off", [None, "heads_mid"])
def test_check_refuses_a_replay_that_differs_from_the_apply_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_off: str | None
) -> None:
    """The replay writes the fitted mask back through the control bundles, so
    its IIA must equal the ``apply`` step's; ``check`` stops on a mismatch
    before ``controls.json`` can quote it. Every control here scores 0.5."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    controls = _script("check_controls")
    run = _fixture_run(tmp_path, torch, save_file, replay_off or "")
    monkeypatch.setattr(controls, "score", lambda *_args: 0.5)
    if replay_off is None:
        summary = controls.check(run, "cpu", draws=2)
        assert {m["iia_replay"] for m in summary.values()} == {0.5}
    else:
        with pytest.raises(SystemExit, match="heads_mid: replay 0.5 != apply 0.75"):
            controls.check(run, "cpu", draws=2)


def test_the_original_copy_fails_its_check_when_the_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``copy_original.py --check`` passes on an unchanged source and fails
    once one cell of onboarding 09's grid moves. The copy names its source
    relative to the repository, here the temporary directory."""
    copier = _script("copy_original")
    monkeypatch.setattr(copier, "REPO", tmp_path)
    source = tmp_path / "grid.json"
    grid = json.loads(copier.SOURCE.read_text())
    source.write_text(copier.SOURCE.read_text())
    out = tmp_path / "copy.json"
    assert copier.main(["--source", str(source), "--out", str(out)]) == 0
    assert copier.main(["--source", str(source), "--out", str(out), "--check"]) == 0
    grid[0]["value"] += 0.5
    source.write_text(json.dumps(grid))
    assert copier.main(["--source", str(source), "--out", str(out), "--check"]) == 1


def test_the_plotted_original_is_the_copy() -> None:
    """The Original panel of ``components_plotted.json`` holds the 84 cells
    of ``component_iia_onboarding09_original.json``, and that copy names
    onboarding 09's grid by its current sha256."""
    copy = json.loads(
        (
            PAPERS
            / "artifacts"
            / "data"
            / NAME
            / "component_iia_onboarding09_original.json"
        ).read_text()
    )
    source = REPO / copy["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == copy["sha256"]
    plotted = json.loads(
        (
            PAPERS / "artifacts" / "figures" / NAME / "components_plotted.json"
        ).read_text()
    )
    drawn = sorted(
        (r["component"], r["layer"], r["value"])
        for r in plotted
        if r["panel"] == "original"
    )
    copied = sorted((r["component"], r["layer"], r["iia"]) for r in copy["records"])
    assert len(copied) == 84
    assert drawn == copied


def _plotted() -> list[dict]:
    return json.loads(
        (
            PAPERS / "artifacts" / "figures" / NAME / "components_plotted.json"
        ).read_text()
    )


@pytest.mark.parametrize("panel", ["mask", "heads"])
def test_the_dbm_figure_draws_the_plotted_values(panel: str) -> None:
    """Each sweep dot is the held-out IIA of one l1 weight, the ringed dot is
    ``CHOSEN``'s, and each tile or head cell is dark exactly when the
    ``CHOSEN`` fit keeps that unit. Layers outside the head gate are striped."""
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_hex

    from causalab.io.plots.dbm_figure import ATTENTION, NEURON

    figures = _script("figures")
    assert figures.CHOSEN in figures.ARMS.values()
    plotted = _plotted()
    drawn = figures.dbm_figure(plotted, panel)

    held_out = {
        r["l1"]: r["value"]
        for r in plotted
        if r["panel"] == "held_out" and r["component"] == panel
    }
    (sweep,) = [ax for ax in drawn.axes if ax.get_label() == "sweep"]
    by_gid = {c.get_gid(): c for c in sweep.collections}
    dots = by_gid["dbm-points"].get_offsets()[:, 1].tolist()
    assert dots == [held_out[w] for w in figures.ARMS.values()]
    assert by_gid["dbm-chosen"].get_offsets()[0, 1] == held_out[figures.CHOSEN]

    chosen = [r for r in plotted if r["panel"] == panel and r["l1"] == figures.CHOSEN]
    (mask,) = [ax for ax in drawn.axes if ax.get_label() == "mask"]
    cells = {p.get_gid(): p for p in mask.patches if p.get_gid()}
    if panel == "mask":
        assert len(chosen) == 56
        expected = {
            f"component:{figures.MASK_ROW[r['component']]}:L{r['layer']}": (
                (ATTENTION if r["component"] == "attention_output" else NEURON)[
                    0 if r["kept"] else 1
                ]
            )
            for r in chosen
        }
    else:
        assert len(chosen) == 12
        expected = {
            f"head:L{r['layer']}:H{r['head']}": ATTENTION[0 if r["kept"] else 1]
            for r in chosen
        }
        striped = [gid for gid in cells if gid not in expected]
        assert len(striped) == 27 * 12
        assert {cells[gid].get_hatch() for gid in striped} == {"////"}
    assert {gid: to_hex(cells[gid].get_facecolor()) for gid in expected} == expected


def test_the_dbm_figure_refuses_a_mask_with_a_missing_unit() -> None:
    """A ``CHOSEN`` fit with no row for one gate stops the figure, so a mask
    cannot be drawn from part of the plotted values."""
    figures = _script("figures")
    plotted = [
        r
        for r in _plotted()
        if not (
            r["panel"] == "mask"
            and r["l1"] == figures.CHOSEN
            and r["component"] == "mlp_output"
            and r["layer"] == 23
        )
    ]
    with pytest.raises(ValueError, match="do not give one value per unit"):
        figures.dbm_figure(plotted, "mask")
