"""The addition-heads DBM package's builder and figure script, on CPU.

``tests/demos/test_papers.py`` holds the committed table to a fresh build and
the page to its documents. This file checks what those tests cannot see: the
table's pairs mean what the page says (prompt-disjoint splits, two different
tens digits per pair, ``label`` the counterfactual's), and the figure script
reads kept heads and scores off a run tree the way the workflow writes one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "demos"
    / "papers"
    / "workflows"
    / "scripts"
    / "addition_heads_dbm"
)


def _module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"addition_heads_dbm_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_pairs_differ_in_the_tens_digit_and_the_splits_share_no_prompt() -> (
    None
):
    rows = _module("build_dataset").build(160, 200, 0, 0.2).rows
    assert [r["split"] for r in rows].count("train") == 160
    assert [r["split"] for r in rows].count("test") == 200
    prompts: dict[str, set[str]] = {"train": set(), "test": set()}
    for row in rows:
        base, cf = row["input"], row["counterfactual_inputs"][0]
        prompts[row["split"]].update([base, cf])
        # the scored token is the answer's tens digit: a1 + b1 + carry
        a1, a0, _, b1, b0, _ = base
        assert row["base_answer"] == str(int(a1) + int(b1) + (int(a0) + int(b0) >= 10))
        c1, c0, _, d1, d0, _ = cf
        assert row["label"] == str(int(c1) + int(d1) + (int(c0) + int(d0) >= 10))
        assert row["label"] != row["base_answer"]
    assert not prompts["train"] & prompts["test"]


def _metric(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {"example_id": str(i), "metric": "iia", "value": v, "eligible": True}
                for i, v in enumerate(values)
            ]
        )
    )


def _gate(path: Path, entries: dict[str, list[float]]) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"theta[objective.sparsity.weight={w}]": torch.tensor(theta)
            for w, theta in entries.items()
        },
        str(path),
    )


def _run_tree(figure: ModuleType, tmp_path: Path) -> None:
    """A run tree with known theta and scores: layer 15 heads {1, 2, 4} in
    every ``fit_all`` entry, no head in ``fit_l15``, heads {0, 1, 2} in
    every random draw."""
    off = [-1.0] * 8
    for layer in figure.LAYERS:
        theta = [1.0 if (layer == 15 and h in (1, 2, 4)) else -1.0 for h in range(8)]
        _gate(
            tmp_path / "fit_all" / f"gate_{layer}.safetensors",
            {repr(float(w)): theta for w in figure.WEIGHTS["all"]},
        )
    _gate(
        tmp_path / "fit_l15" / "gate.safetensors",
        {repr(float(w)): off for w in figure.WEIGHTS["l15"]},
    )
    for panel in ("all", "l15"):
        for w in figure.WEIGHTS[panel]:
            step = tmp_path / f"apply_{panel}_{figure.tag(w)}"
            _metric(step / "iia.json", [1.0, 0.0, 1.0, 1.0])
            _metric(step / "null.json", [0.0, 1.0, 0.0, 0.0])
    for model in figure.SWAPS:
        _metric(tmp_path / "swaps" / f"iia_{model}.json", [0.5, 0.5])
        _metric(tmp_path / "swaps" / f"null_{model}.json", [0.0, 0.0])
    for seed in range(figure.RANDOM_DRAWS):
        _gate(
            tmp_path / f"random_{seed}" / "gate.safetensors",
            {"3.0": [1.0, 1.0, 1.0] + [-1.0] * 5},
        )
        _metric(tmp_path / f"apply_random_{seed}" / "iia.json", [0.25])
        _metric(tmp_path / f"apply_random_{seed}" / "null.json", [0.75])


def test_the_figure_script_reads_kept_heads_and_held_out_scores(tmp_path: Path) -> None:
    """The rows name the heads with theta > 0 and the mean of each metric
    table."""
    figure = _module("heads_figure")
    _run_tree(figure, tmp_path)
    rows = figure.load(tmp_path)
    first = next(r for r in rows if r["panel"] == "all")
    assert (first["kept"], first["n_kept"]) == ("L15: 1,2,4", 3)
    assert (first["iia"], first["null"], first["n"]) == (0.75, 0.25, 4)
    assert all(r["n_kept"] == 0 for r in rows if r["panel"] == "l15")
    randoms = [r for r in rows if r["setting"].startswith("random")]
    assert len(randoms) == 5 and all(r["kept"] == "L15: 0,1,2" for r in randoms)
    assert figure.tag(0.01) == "w0p01" and figure.tag(30.0) == "w30"


def test_the_dbm_panels_draw_the_chosen_weights_mask_over_the_whole_model(
    tmp_path: Path,
) -> None:
    """The 48-head panel rings the ``CHOSEN`` weight and draws its heads over
    every layer of Qwen3.5-2B, the Gated DeltaNet layers striped; ``main``
    writes every image the page embeds."""
    from matplotlib.colors import to_hex

    figure = _module("heads_figure")
    assert all(w in figure.WEIGHTS[k] for k, w in figure.CHOSEN.items())
    _run_tree(figure, tmp_path)
    rows = figure.load(tmp_path)
    drawn = figure.dbm_panel(rows, "48", "synthetic")
    (mask,) = [ax for ax in drawn.axes if ax.get_label() == "mask"]
    cells = {p.get_gid(): p for p in mask.patches if p.get_gid()}
    assert len(cells) == 24 * 16  # every layer, as wide as a DeltaNet layer
    assert to_hex(cells["head:L15:H1"].get_facecolor()) == "#3267a8"
    assert to_hex(cells["head:L15:H0"].get_facecolor()) == "#e6eef9"
    assert cells["head:L14:H0"].get_hatch() == "////"  # a Gated DeltaNet layer
    (sweep,) = [ax for ax in drawn.axes if ax.get_label() == "sweep"]
    assert any(c.get_gid() == "dbm-chosen" for c in sweep.collections)

    out = tmp_path / "figures"
    figure.main(tmp_path, out)
    for name in ("heads_all", "heads_48", "heads_l15", "heads_checks"):
        assert (out / f"{name}.png").stat().st_size > 0
