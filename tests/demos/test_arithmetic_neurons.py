"""Checks specific to the ``arithmetic_neurons`` paper package.

``test_papers.py`` validates every package's documents and rebuilds its
tables. This file checks the two facts of this package that those generic
checks cannot see: that the 28 neuron indices transcribed from Figure 8 of
Feucht et al. 2026 are the ones the documents address, that the weekdays
table's held-out splits are what the page says they are (prompt-disjoint,
every output day equally common), that the committed sweep values let a
reader redo the one-standard-error choice of the L1 weight and that the sweep
figure rings that choice over its mask, that the Figure
8a replication draws the DBM mask's neurons alone, that each header
names a figure script that exists, and that each diff of the page's "change these lines" section
shows every key its document changes (``docs/paper_replications.md``, The
page, part 5: "a `diff` fence of the changed lines only").
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

PAPERS = Path(__file__).resolve().parents[2] / "demos" / "papers"
DATA = PAPERS / "artifacts" / "data" / "arithmetic_neurons"
FIGURES = PAPERS / "artifacts" / "figures" / "arithmetic_neurons"
PROTOCOLS = PAPERS / "protocols"
PAGE = PAPERS / "arithmetic_neurons.md"
WORKFLOW = PAPERS / "workflows" / "arithmetic_neurons.json"
#: Layer 18's MLP of Llama-3.1-8B: config.json ``intermediate_size``.
WIDTH = 14336


def _paper_neurons() -> list[dict]:
    return json.loads((DATA / "neurons_feucht2026.json").read_text())["neurons"]


def test_the_paper_lists_28_distinct_neurons_in_range() -> None:
    """Section 5.1 names 28 neurons; Figure 8a groups them into periods
    2, 5, 10, 20, 50 and 100 with 2, 6, 8, 3, 7 and 2 members."""
    neurons = _paper_neurons()
    ids = [n["neuron"] for n in neurons]
    assert len(ids) == len(set(ids)) == 28
    assert all(0 <= i < WIDTH for i in ids)
    assert Counter(n["period"] for n in neurons) == {
        2: 2,
        5: 6,
        10: 8,
        20: 3,
        50: 7,
        100: 2,
    }


def test_the_documents_address_the_transcribed_neurons() -> None:
    """The paper-neuron interchange swaps exactly the 28 of the neuron file."""
    ids = [n["neuron"] for n in _paper_neurons()]
    document = json.loads(
        (PROTOCOLS / "arithmetic_neurons_paper_neurons.json").read_text()
    )
    method = document["method"]
    assert method["reads"]["v_cf"]["dims"] == ids
    assert method["writes"]["patch"]["dims"] == ids


def test_the_weekdays_splits_are_prompt_disjoint_and_balanced() -> None:
    """No prompt is in two splits, the held-out splits take 21 prompts each,
    every held-out label is one of the seven days equally often, and no pair
    has a label equal to its base answer."""
    rows = json.loads((DATA / "weekdays.json").read_text())
    prompts: dict[str, set[str]] = {}
    for row in rows:
        prompts.setdefault(row["split"], set()).update(
            [row["input"], row["counterfactual_inputs"][0]]
        )
    assert set(prompts) == {"train", "val", "test"}
    assert not prompts["train"] & prompts["val"]
    assert not prompts["train"] & prompts["test"]
    assert not prompts["val"] & prompts["test"]
    assert len(prompts["val"]) == len(prompts["test"]) == 21
    assert len(prompts["train"]) == 98 - 42
    for split in ("val", "test"):
        labels = Counter(r["label"] for r in rows if r["split"] == split)
        assert len(labels) == 7 and len(set(labels.values())) == 1, labels
    assert all(r["label"] != r["base_answer"] for r in rows)
    assert all(r["label"] == r["cf_answer"] for r in rows)


def test_the_plotted_sweep_recomputes_the_one_standard_error_choice() -> None:
    """``masks_plotted.json`` records the val pair count of each sweep row, so
    the one-standard-error rule of the page (Method, "The L1 weight") can be
    redone from the committed file: the largest weight whose val IIA is within
    one binomial standard error of the best is the weight the fit document
    ships."""
    rows = [
        r
        for r in json.loads((FIGURES / "masks_plotted.json").read_text())
        if r["panel"] == "sweep"
    ]
    table = json.loads((DATA / "weekdays.json").read_text())
    val = sum(1 for r in table if r["split"] == "val")
    assert rows and all(r["n"] == val for r in rows), [r["n"] for r in rows]
    best = max(rows, key=lambda r: r["iia"])
    se = math.sqrt(best["iia"] * (1 - best["iia"]) / best["n"])
    assert round(se, 3) == 0.024  # the value the page states
    chosen = max(r["l1_weight"] for r in rows if r["iia"] >= best["iia"] - se)
    fit = json.loads((PROTOCOLS / "arithmetic_neurons_fit.json").read_text())
    assert chosen == fit["method"]["train"]["objective"]["l1"]["weight"] == 100.0


def test_the_ringed_sweep_point_is_the_one_standard_error_choice() -> None:
    """``masks_plotted.json`` flags one sweep row as the point ``masks_sweep.png``
    rings. It is the row the rule above picks, and it keeps as many neurons
    as the held-out DBM condition, whose mask the grid draws."""
    rows = json.loads((FIGURES / "masks_plotted.json").read_text())
    sweep = [r for r in rows if r["panel"] == "sweep"]
    ringed = [r for r in sweep if r["chosen"]]
    assert [r["l1_weight"] for r in ringed] == [100.0]
    assert sweep.index(ringed[0]) == _figures().one_standard_error(sweep)
    (dbm,) = [r for r in rows if r["condition"].startswith("DBM mask")]
    assert ringed[0]["units"] == dbm["units"] == 23
    assert all(r["chosen"] is None for r in rows if r["panel"] == "test")


def _sweep_tree(run: Path, figures: ModuleType, fitted: list[int]) -> None:
    """A fabricated run tree for ``masks``: three sweep weights over a
    64-neuron gate, the fitted gate's rank table, and one IIA table per
    held-out condition."""
    import torch
    from safetensors.torch import save_file

    kept = {1.0: range(10), 10.0: [3, 7], 100.0: [7]}
    iia = {1.0: 0.5, 10.0: 0.49, 100.0: 0.2}
    (run / "sweep").mkdir(parents=True)
    (run / "sweep" / "train_eval.json").write_text(
        json.dumps(
            [
                {
                    "coords": {figures.AXIS: w},
                    "split": "arithmetic_neurons/weekdays#val",
                    "metrics": {"iia": iia[w]},
                }
                for w in kept
            ]
        )
    )
    (run / "sweep" / "fit_diagnostics.json").write_text(
        json.dumps(
            [
                {
                    "coords": {figures.AXIS: w},
                    "featurizers": {"gate": {"hard_mask_size": float(len(units))}},
                }
                for w, units in kept.items()
            ]
        )
    )
    gates = {}
    for w, units in kept.items():
        theta = -torch.ones(64)
        theta[list(units)] = 1.0
        gates[f"theta[objective.l1.weight={w!r}]"] = theta
    save_file(gates, str(run / "sweep" / "gate.safetensors"))
    for step in ("all_neurons", "apply", "apply_random", "paper_neurons"):
        (run / step).mkdir()
        (run / step / "iia.json").write_text(
            json.dumps([{"value": 1.0, "eligible": True}] * 4)
        )
    (run / "apply_gate_half").mkdir()
    (run / "apply_gate_half" / "iia.json").write_text(
        json.dumps([{"value": 0.0, "eligible": True}] * 4)
    )
    (run / "apply_gate_half" / "rank.json").write_text(
        json.dumps([{"unit": u, "rank": u, "hard": u in fitted} for u in range(64)])
    )


def test_the_sweep_figure_rings_the_rule_s_choice_over_its_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a fabricated sweep, ``masks`` flags the weight the one-standard-error
    rule picks (10: IIA 0.49 is within one error of 0.5, 0.2 is not), and
    ``sweep_figure`` rings that point over the grid of its two neurons. A
    fitted gate that keeps other neurons than the ringed sweep point stops
    the script, since the page scores the fitted gate."""
    import numpy as np
    from matplotlib.colors import to_hex

    figures = _figures()
    _sweep_tree(tmp_path / "run", figures, fitted=[3, 7])
    monkeypatch.setattr(figures, "RUN", tmp_path / "run")
    monkeypatch.setattr(figures, "FIGURES", tmp_path / "figures")

    df = figures.masks([7, 3])

    sweep = df[df["panel"] == "sweep"].to_dict(orient="records")
    assert [r["chosen"] for r in sweep] == [False, True, False]
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "masks_sweep.png",
        "masks_test.png",
    ]
    kept = [int(u in (3, 7)) for u in range(64)]
    drawn = figures.sweep_figure(sweep, kept)
    axes = {ax.get_label(): ax for ax in drawn.axes}
    marks = {c.get_gid(): c for c in axes["sweep"].collections}
    dots = marks["dbm-points"].get_offsets().tolist()
    assert marks["dbm-chosen"].get_offsets().tolist() == [dots[1]]
    assert marks["dbm-ring"].get_offsets().tolist() == [dots[1]]
    assert axes["sweep"].get_ylabel() == "val IIA"
    heading = [t.get_text() for t in drawn.texts if "ringed" in t.get_text()]
    assert heading == ["Mask at the ringed point: 2 of 64 neurons kept, val IIA 0.490"]
    # the grid's kept cells, in the viewer's teal, are neurons 3 and 7
    (grid,) = [a for a in axes["mask"].images if a.get_gid() == "dbm-neurons"]
    pixels = np.asarray(grid.get_array())
    teal = [
        (row, column)
        for row in range(pixels.shape[0])
        for column in range(pixels.shape[1])
        if to_hex(pixels[row, column]) == "#1e6978"
    ]
    assert teal == [(0, 3), (0, 7)]

    with pytest.raises(ValueError, match="keeps 2 neurons, not the 1"):
        figures.masks([7])


def test_every_script_a_header_names_exists() -> None:
    """``header.description`` names the script that reads the document's
    output (``docs/paper_replications.md``, The chunks); that script is a
    file of the package."""
    for path in sorted(PROTOCOLS.glob("arithmetic_neurons_*.json")):
        description = json.loads(path.read_text())["header"]["description"]
        for script in re.findall(r"workflows/scripts/[\w/]+\.py", description):
            assert (PAPERS / script).is_file(), f"{path.name} names {script}"


#: The diffs of the page's "change these lines" section, in page order: the
#: document the diff starts from, the document it produces, and the workflow
#: step whose ``set`` the produced document runs under (None: the file as is).
DIFFS = (
    ("arithmetic_neurons_fit.json", "arithmetic_neurons_apply.json", "apply"),
    ("arithmetic_neurons_fit.json", "arithmetic_neurons_harvest.json", None),
)


def _sections(name: str, step: str | None) -> dict:
    """A document as the page's chunks spell it: no header, and the ``method``
    sections beside ``model`` and ``data``; with a step's ``set`` applied."""
    document = json.loads((PROTOCOLS / name).read_text())
    document.pop("header")
    flat = {**document.pop("method"), **document}
    if step is not None:
        overrides = json.loads(WORKFLOW.read_text())["steps"][step].get("set", {})
        for dotted, value in overrides.items():
            *parents, leaf = dotted.split(".")
            node = flat
            for key in parents:
                node = node[key]
            node[leaf] = value
    return flat


def _page_diffs() -> list[str]:
    """The ``diff`` fences of the "change these lines" section, in order."""
    text = PAGE.read_text()
    start = re.search(r"^## .*: change these lines$", text, re.MULTILINE)
    assert start, "the page has no change-these-lines section"
    end = re.compile(r"^## ", re.MULTILINE).search(text, start.end())
    section = text[start.end() : end.start() if end else len(text)]
    return re.findall(r"^```diff\n(.*?)^```", section, re.MULTILINE | re.DOTALL)


def _changed_lines(diff: str) -> str:
    return "\n".join(line for line in diff.splitlines() if line[:1] in "+-")


def test_each_diff_shows_every_changed_key() -> None:
    """A reader who applies a diff to its starting document gets the shipped
    one: every section that differs is named on a changed line, down to its
    keys, and every saved file that is added or dropped is on a ``+`` or a
    ``-`` line."""
    diffs = _page_diffs()
    assert len(diffs) == len(DIFFS)
    for diff, (source, target, step) in zip(diffs, DIFFS):
        before, after = _sections(source, None), _sections(target, step)
        changed = _changed_lines(diff)
        added = "\n".join(line for line in diff.splitlines() if line[:1] == "+")
        dropped = "\n".join(line for line in diff.splitlines() if line[:1] == "-")
        for key in sorted(set(before) | set(after)):
            old, new = before.get(key), after.get(key)
            if old == new:
                continue
            where = f"{source} -> {target}: {key}"
            if old is None or new is None:
                assert f'"{key}"' in changed, where
            elif isinstance(old, dict) and isinstance(new, dict):
                for sub in sorted(set(old) | set(new)):
                    if old.get(sub) != new.get(sub):
                        assert f'"{sub}"' in changed, f"{where}.{sub}"
            else:
                for entry in new:
                    if entry not in old:
                        assert entry["file_path"] in added, f"{where}: {entry}"
                for entry in old:
                    if entry not in new:
                        assert entry["file_path"] in dropped, f"{where}: {entry}"


def _figures() -> ModuleType:
    path = PAPERS / "workflows" / "scripts" / "arithmetic_neurons" / "figures.py"
    spec = importlib.util.spec_from_file_location("arithmetic_neurons_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fig8a_draws_the_mask_neurons_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a fabricated harvest of four prompts, ``fig8a`` averages the full
    activation by output sum for the mask's neurons only, flags the ones in
    the paper's 28, and writes the replication image and no other image
    (no panel of the paper's neurons)."""
    import torch
    from safetensors.torch import save_file

    figures = _figures()
    rows = [{"sum": s} for s in ("2", "3", "3", "4")]
    acts = torch.zeros(len(rows), WIDTH)
    acts[:, 1712] = torch.tensor([1.0, 2.0, 4.0, -1.0])
    acts[:, 5] = torch.tensor([0.5, 0.0, 1.0, 0.0])
    (tmp_path / "run" / "harvest_addition").mkdir(parents=True)
    save_file(
        {"full": acts}, str(tmp_path / "run" / "harvest_addition" / "full.safetensors")
    )
    monkeypatch.setattr(figures, "RUN", tmp_path / "run")
    monkeypatch.setattr(figures, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(figures, "table", lambda name: rows)

    drawn = figures.fig8a([1712, 5])

    assert drawn.to_dict(orient="records") == [
        {"neuron": 1712, "in_paper": True, "sum": 2, "mean_activation": 1.0, "n": 1},
        {"neuron": 1712, "in_paper": True, "sum": 3, "mean_activation": 3.0, "n": 2},
        {"neuron": 1712, "in_paper": True, "sum": 4, "mean_activation": -1.0, "n": 1},
        {"neuron": 5, "in_paper": False, "sum": 2, "mean_activation": 0.5, "n": 1},
        {"neuron": 5, "in_paper": False, "sum": 3, "mean_activation": 0.5, "n": 2},
        {"neuron": 5, "in_paper": False, "sum": 4, "mean_activation": 0.0, "n": 1},
    ]
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "fig8a_replication.png"
    ]


def test_the_committed_fig8a_is_the_mask_the_page_scores() -> None:
    """``fig8a_plotted.json`` holds one row per (neuron, sum) for as many
    neurons as the held-out DBM condition keeps, and its paper flags agree
    with the transcribed list."""
    rows = json.loads((FIGURES / "fig8a_plotted.json").read_text())
    neurons = {r["neuron"] for r in rows}
    masks = json.loads((FIGURES / "masks_plotted.json").read_text())
    (dbm,) = [r for r in masks if r["condition"].startswith("DBM mask")]
    assert len(neurons) == dbm["units"]
    assert len(rows) == len(neurons) * len({r["sum"] for r in rows})
    paper = {n["neuron"] for n in _paper_neurons()}
    assert all(r["in_paper"] == (r["neuron"] in paper) for r in rows)


def test_the_shared_neurons_repeat_with_the_papers_periods() -> None:
    """The page's Method block says each neuron the mask shares with the
    transcribed list repeats over output sums with the period Figure 8a gives
    it, which supports the transcription. The strongest nonzero frequency of
    the committed mean activations, as a period in sums, rounds to it."""
    import numpy as np

    period = {n["neuron"]: n["period"] for n in _paper_neurons()}
    rows = json.loads((FIGURES / "fig8a_plotted.json").read_text())
    by_neuron: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_neuron.setdefault(r["neuron"], []).append((r["sum"], r["mean_activation"]))
    shared = sorted(set(by_neuron) & set(period))
    assert len(shared) == 18  # the page's count
    for neuron in shared:
        values = np.array([v for _, v in sorted(by_neuron[neuron])])
        spectrum = np.abs(np.fft.rfft(values - values.mean()))
        frequencies = np.fft.rfftfreq(len(values))
        strongest = 1 / frequencies[1 + spectrum[1:].argmax()]
        assert round(strongest) == period[neuron], (neuron, strongest)


def test_every_committed_image_is_on_the_page() -> None:
    """The figure script writes only images the page shows, and the page
    shows only committed images of the package."""
    page = PAGE.read_text()
    shown = set(re.findall(r"\]\((artifacts/[^)]+\.png)\)", page))
    committed = {
        str(p.relative_to(PAPERS))
        for folder in (FIGURES, DATA)
        for p in folder.glob("*.png")
    }
    assert shown == committed
