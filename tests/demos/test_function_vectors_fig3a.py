"""Checks specific to the function-vector package, ``demos/papers/function_vectors_fig3a``.

``test_papers.py`` holds the package to the shared format. This module checks
what the page claims about its own pieces:

* the prompts follow the authors' template and pools (demonstrations from
  ``train``, queries from ``valid``, held-out queries from ``test``);
* the figure script's indirect effect is Todd et al.'s Eq. 3 and 4, and its
  ranking and mask reading are the ones the page quotes;
* the DBM figure is ``plot_dbm``'s count of the tasks that keep each head,
  layers across as in the paper's Figure 3a, with the paper's ten heads
  outlined;
* the gated mean swap the DBM documents use sets the kept heads to their task
  mean and leaves the other heads alone, and a task's mean reaches only that
  task's rows. This runs the documents' pattern on the tiny random GPT-J
  against a raw-hook oracle.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "demos" / "papers" / "workflows" / "scripts" / "function_vectors_fig3a"
DATA = REPO / "demos" / "papers" / "artifacts" / "data" / "function_vectors_fig3a"
TINY_GPTJ = "hf-internal-testing/tiny-random-GPTJForCausalLM"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"fv_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("build_dataset")
figure = _load("fig3a_figure")


# --------------------------------------------------------------------- tables


def test_the_prompt_is_the_authors_template() -> None:
    """``create_prompt`` of the authors' ``prompt_utils.py`` with the default
    prefixes ``Q:``/``A:``, separators ``\\n``/``\\n\\n``, a prepended space on
    every word and ``<|endoftext|>`` in front (GPT-J prepends no BOS)."""
    train = [{"input": "hot", "output": "cold"}, {"input": "up", "output": "down"}]
    assert builder.prompt_text(train, "0,1", "1,0", "big") == (
        "<|endoftext|>Q: hot\nA: down\n\nQ: up\nA: cold\n\nQ: big\nA:"
    )


@pytest.mark.parametrize("table", sorted(builder.TABLES))
def test_rows_draw_from_the_authors_pools(table: str) -> None:
    """Every row's demonstrations are ten distinct train-pool records, its
    labels are those records' outputs (in order for ``clean``, a permutation
    otherwise), and its query comes from the table's pool."""
    tasks = builder.load_tasks()
    pools = {name: builder.split_task(records) for name, records in tasks.items()}
    n, shuffled, pool, _ = builder.TABLES[table]
    rows = json.loads((DATA / f"{table}.json").read_text())
    assert len(rows) == n * len(tasks)
    permuted = 0
    for row in rows:
        task = row["split"]
        demos = [int(i) for i in row["demos"].split(",")]
        labels = [int(i) for i in row["labels"].split(",")]
        assert len(set(demos)) == builder.N_SHOTS
        assert sorted(labels) == sorted(demos)
        permuted += labels != demos
        query = {q["id"]: q for q in pools[task][pool]}[row["query"]]
        assert row["output"] == query["output"]
        assert row["input"] == builder.prompt_text(
            pools[task]["train"], row["demos"], row["labels"], query["input"]
        )
        assert row["label"].strip(), "a blank first token is never a query's answer"
    assert (permuted > 0) == shuffled


def test_heldout_queries_are_other_records_than_the_fit_queries() -> None:
    """By record, not by text: the authors' product-company file lists
    ``Golden Axe`` twice, and the split puts one copy in each pool."""
    fit = {
        (r["split"], r["query"]) for r in json.loads((DATA / "fit.json").read_text())
    }
    held = {
        (r["split"], r["query"])
        for r in json.loads((DATA / "heldout.json").read_text())
    }
    assert not fit & held


def test_an_out_directory_that_does_not_exist_yet_gets_every_table(
    tmp_path: Path,
) -> None:
    """``--out`` without a ``.json`` suffix names a directory, even before it
    exists: a fresh rebuild writes all four tables there, byte for byte the
    committed ones."""
    out = tmp_path / "fresh" / "function_vectors_fig3a"
    assert builder.main(["--out", str(out)]) == 0
    for name in builder.TABLES:
        assert (out / f"{name}.json").read_bytes() == (
            DATA / f"{name}.json"
        ).read_bytes()


# --------------------------------------------------------------- figure maths


def _metric(task: str, layer: int, head: int, example: str, ce: float) -> dict:
    return {
        figure.TASK: task,
        figure.LAYER: layer,
        figure.HEAD: head,
        "example_id": example,
        "value": ce,
    }


def test_the_indirect_effect_is_a_probability_gain() -> None:
    """CIE = p(answer | head := mean) − p(answer), per prompt, from
    cross-entropies; the per-task value is the mean over prompts."""
    import math

    patched = pd.DataFrame(
        [
            _metric("a", 0, 0, "0", -math.log(0.5)),
            _metric("a", 0, 0, "1", -math.log(0.3)),
        ]
    )
    corrupted = pd.DataFrame(
        [
            _metric("a", 0, 0, "0", -math.log(0.2)),
            _metric("a", 0, 0, "1", -math.log(0.1)),
            # the un-swapped forward repeats at every point
            _metric("a", 0, 1, "0", -math.log(0.2)),
            _metric("a", 0, 1, "1", -math.log(0.1)),
        ]
    )
    out = figure.indirect_effects(patched, corrupted)
    assert out["cie"].tolist() == pytest.approx([((0.5 - 0.2) + (0.3 - 0.1)) / 2])
    assert out["n"].tolist() == [2]


def test_a_baseline_that_differs_between_points_is_refused() -> None:
    patched = pd.DataFrame([_metric("a", 0, 0, "0", 1.0)])
    corrupted = pd.DataFrame(
        [_metric("a", 0, 0, "0", 1.0), _metric("a", 0, 1, "0", 2.0)]
    )
    with pytest.raises(ValueError, match="differs between points"):
        figure.indirect_effects(patched, corrupted)


def test_heads_rank_by_aie_then_layer_then_head() -> None:
    import numpy as np

    grid = np.zeros((figure.LAYERS, figure.HEADS))
    grid[15, 5] = 0.06
    grid[9, 14] = 0.05
    grid[3, 2] = grid[1, 7] = 0.01
    assert figure.rank_heads(grid)[:4] == [(15, 5), (9, 14), (1, 7), (3, 2)]


def test_a_mask_keeps_the_heads_with_positive_theta(tmp_path: Path) -> None:
    """``theta > 0`` is a sigmoid gate's evaluation mask; entries are keyed by
    the task coordinate the fit swept."""
    from safetensors.torch import save_file

    for layer in range(figure.LAYERS):
        theta = torch.full((figure.HEADS,), -1.0)
        if layer == 9:
            theta[14] = 0.5
        save_file(
            {
                "theta[axes.task=antonym]": theta,
                "theta[axes.task=synonym]": torch.full((figure.HEADS,), -1.0),
            },
            str(tmp_path / f"gate_L{layer}.safetensors"),
        )
    assert figure.dbm_masks(tmp_path) == {"antonym": {(9, 14)}, "synonym": set()}


# ----------------------------------------------- the gated mean on tiny GPT-J

#: A 4-head, 5-layer random GPT-J: head_dim 8.
HEAD_DIM = 8
PAIRS = {
    "one": {" the": " of", " of": " in", " in": " and", " and": " the"},
    "two": {" the": " in", " of": " and", " in": " the", " and": " of"},
}


def _table(path: Path) -> None:
    rows = []
    for task, lookup in PAIRS.items():
        for i, query in enumerate(lookup):
            shots = "".join(
                f"Q:{x}\nA:{lookup[x]}\n\n" for x in list(lookup)[i:] + list(lookup)[:i]
            )
            rows.append(
                {
                    "input": f"<|endoftext|>{shots}Q:{query}\nA:",
                    "label": lookup[query],
                    "split": task,
                    "string_mode": "exact",
                }
            )
    path.write_text(json.dumps(rows))


def _doc(description: str, method: dict, rows: list[dict]) -> dict:
    return {
        "header": {"protocol_version": "4", "description": description},
        "model": {"key": TINY_GPTJ, "revision": "main"},
        "data": {"base": {"dataset": {"axis": "task.table"}, "field": "input"}},
        "axes": {"task": {"rows": rows, "key": "task"}},
        "method": method,
    }


def test_the_gated_mean_sets_kept_heads_and_only_their_task(tmp_path: Path) -> None:
    """The DBM documents' pattern: ``mean_source`` writes each layer's task
    mean, a gate-featurized read of it is the swap operand, so the masked
    model's layer is ``m ⊙ mean + (1 − m) ⊙ x``. Replayed with the fitted
    gates, the kept heads equal the task mean and the others their own
    values, and each task's rows see that task's mean."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open

    from causalab.cli import main
    from causalab.protocol.registry import model_info_from_hf_config, register_model

    register_model(
        model_info_from_hf_config(TINY_GPTJ, AutoConfig.from_pretrained(TINY_GPTJ))
    )
    data = tmp_path / "data" / "fv"
    data.mkdir(parents=True)
    _table(data / "t.json")
    rows = [{"task": t, "table": f"fv/t#{t}"} for t in PAIRS]
    layers = (0, 1)
    mean = _doc(
        "per-task mean of attention_premix at one layer",
        {
            "intervened_models": {"original": {"input": "base", "reads": ["value"]}},
            "sites": {"target": {"component": "attention_premix", "layers": [0]}},
            "reads": {"value": {"site": "target", "pos": -1}},
            "save": [
                {
                    "read": "value",
                    "model": "original",
                    "file_path": "mean.safetensors",
                    "reduce": "mean",
                }
            ],
        },
        rows,
    )

    def gated(loaded: bool) -> dict:
        method = {
            "intervened_models": {
                "mean_source": {
                    "input": "base",
                    "reads": [f"v{layer}" for layer in layers],
                    "writes": [f"put{layer}" for layer in layers],
                },
                "masked": {
                    "input": "base",
                    "reads": ["logits"],
                    "writes": [f"mask{layer}" for layer in layers],
                },
            },
            "sites": {
                **{
                    f"L{layer}": {"component": "attention_premix", "layers": [layer]}
                    for layer in layers
                },
                "lm_head": {"component": "lm_head"},
            },
            "featurizers": {
                # the replay keeps each layer's top two heads, so the mask is
                # mixed whatever the short fit learned
                f"g{layer}": {
                    "kind": "gate",
                    "group": "head",
                    **(
                        {"file_path": f"fit/g{layer}.safetensors", "top_k": 2}
                        if loaded
                        else {}
                    ),
                }
                for layer in layers
            },
            "params": {
                f"mu{layer}": {"file_path": f"mean_L{layer}/mean.safetensors"}
                for layer in layers
            },
            "reads": {
                **{
                    f"v{layer}": {
                        "site": f"L{layer}",
                        "pos": -1,
                        "featurizer": f"g{layer}",
                    }
                    for layer in layers
                },
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                **{
                    f"put{layer}": {
                        "site": f"L{layer}",
                        "pos": -1,
                        "do": {"swap": f"mu{layer}"},
                    }
                    for layer in layers
                },
                **{
                    f"mask{layer}": {
                        "site": f"L{layer}",
                        "pos": -1,
                        "featurizer": f"g{layer}",
                        "do": {"swap": f"v{layer}"},
                    }
                    for layer in layers
                },
            },
        }
        if loaded:
            method["save"] = [
                {
                    "read": "logits",
                    "model": "masked",
                    "aggregation": {"kind": "cross_entropy", "target": "label"},
                    "file_path": "ce.json",
                }
            ]
        else:
            method["train"] = {
                "objective": {
                    "ce": {
                        "weight": 1.0,
                        "read": "logits",
                        "model": "masked",
                        "aggregation": {"kind": "cross_entropy", "target": "label"},
                    },
                    "l1": {"weight": 0.01, "l1": [f"g{layer}" for layer in layers]},
                },
                "params": [f"g{layer}" for layer in layers],
                "optimizer": {"name": "adamw", "lr": 0.5, "weight_decay": 0.0},
                "steps": {"epochs": 2},
                "batch": {"pairs": 2},
                "seed": 0,
            }
            method["save"] = [
                {
                    "value": f"g{layer}",
                    "site": f"L{layer}",
                    "file_path": f"g{layer}.safetensors",
                }
                for layer in layers
            ]
        return _doc("gated task means", method, rows)

    protocols = tmp_path / "protocols"
    protocols.mkdir()
    (protocols / "mean.json").write_text(json.dumps(mean))
    (protocols / "fit.json").write_text(json.dumps(gated(False)))
    (protocols / "apply.json").write_text(json.dumps(gated(True)))
    steps = {
        **{
            f"mean_L{layer}": {
                "type": "intervention_protocol",
                "document": "../protocols/mean.json",
                **({"set": {"sites.target.layers": [layer]}} if layer else {}),
            }
            for layer in layers
        },
        "fit": {"type": "intervention_protocol", "document": "../protocols/fit.json"},
        "apply": {
            "type": "intervention_protocol",
            "document": "../protocols/apply.json",
        },
    }
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "w.json").write_text(
        json.dumps(
            {
                "version": "1",
                "description": "gated mean",
                "output_dir": "w",
                "steps": steps,
            }
        )
    )
    out = tmp_path / "out"
    code = main(
        [
            "run",
            str(workflows / "w.json"),
            "--engine",
            "pytorch_hooks",
            "--data-root",
            str(tmp_path / "data"),
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )
    assert code == 0

    tok = AutoTokenizer.from_pretrained(TINY_GPTJ)
    model = AutoModelForCausalLM.from_pretrained(TINY_GPTJ).eval()
    table = json.loads((data / "t.json").read_text())
    engine = json.loads((out / "w" / "apply" / "ce.json").read_text())
    for task in PAIRS:
        means = {
            layer: safe_open(
                str(out / "w" / f"mean_L{layer}" / "mean.safetensors"), "pt"
            )
            .get_tensor(f"value[axes.task={task}]")
            .float()
            .flatten()
            for layer in layers
        }
        masks = {}
        for layer in layers:
            theta = (
                safe_open(str(out / "w" / "fit" / f"g{layer}.safetensors"), "pt")
                .get_tensor(f"theta[axes.task={task}]")
                .float()
                .flatten()
            )
            top = sorted(range(len(theta)), key=lambda h: (-float(theta[h]), h))[:2]
            masks[layer] = (
                torch.zeros(len(theta))
                .index_fill(0, torch.tensor(top), 1.0)
                .repeat_interleave(HEAD_DIM)
            )
        for i, row in enumerate(r for r in table if r["split"] == task):
            hooks = []
            for layer in layers:

                def pre(_m, args, layer=layer):
                    x = args[0].clone()
                    x[0, -1] = (
                        masks[layer] * means[layer] + (1 - masks[layer]) * x[0, -1]
                    )
                    return (x,)

                hooks.append(
                    model.transformer.h[layer].attn.out_proj.register_forward_pre_hook(
                        pre
                    )
                )
            with torch.no_grad():
                logits = model(**tok(row["input"], return_tensors="pt")).logits[0, -1]
            for hook in hooks:
                hook.remove()
            want = -torch.log_softmax(logits.float(), -1)[
                tok(row["label"]).input_ids[0]
            ].item()
            got = [
                e["value"]
                for e in engine
                if e[figure.TASK] == task and e["example_id"] == str(i)
            ]
            assert got == [pytest.approx(want, abs=1e-5)]


def _synthetic_run(root: Path, fits: dict[str, list[str]]) -> None:
    """A run tree with one task's scan (head (9, 14) the only effect) and
    the given DBM steps, each fitting the listed tasks with head (9, 14) kept."""
    import math

    from safetensors.torch import save_file

    for step, layers in (("scan_early", range(14)), ("scan_late", range(14, 28))):
        patched, corrupted = [], []
        for layer in layers:
            for head in range(figure.HEADS):
                p = 0.3 if (layer, head) == (9, 14) else 0.1
                patched.append(_metric("a", layer, head, "0", -math.log(p)))
                corrupted.append(_metric("a", layer, head, "0", -math.log(0.1)))
        (root / step).mkdir(parents=True)
        (root / step / "ce_patched.json").write_text(json.dumps(patched))
        (root / step / "ce_corrupted.json").write_text(json.dumps(corrupted))
    for n, (step, tasks) in enumerate(fits.items(), start=1):
        fit = root / step
        fit.mkdir()
        for layer in range(figure.LAYERS):
            tensors = {}
            for task in tasks:
                theta = torch.full((figure.HEADS,), -1.0)
                if layer == 9:
                    theta[14] = 1.0
                tensors[f"theta[axes.task={task}]"] = theta
            save_file(tensors, str(fit / f"gate_L{layer}.safetensors"))
        apply = root / f"dbm_apply_{n}"
        apply.mkdir()
        for condition, value in (
            ("masked", 0.5),
            ("corrupted", 0.2),
            ("all_heads", 0.0),
        ):
            rows = [{figure.TASK: t, "example_id": "0", "value": value} for t in tasks]
            (apply / f"accuracy_{condition}.json").write_text(json.dumps(rows))


def test_the_figure_joins_the_dbm_steps(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _synthetic_run(run, {"dbm_fit_1": ["a", "b"], "dbm_fit_2": ["c"]})
    out = {name: tmp_path / f"{name}.png" for name in ("replication", "dbm")}
    plotted = figure.main({"artifacts": run}, {**out, "plotted": tmp_path / "p.json"})
    assert plotted["top10"][0] == {"layer": 9, "head": 14, "aie": pytest.approx(0.2)}
    assert plotted["dbm"]["mask_size"] == {"a": 1, "b": 1, "c": 1}
    assert plotted["dbm"]["tasks_keeping_head"][9][14] == 3
    assert plotted["dbm"]["overlap_with_paper_top10"] == {"a": 1, "b": 1, "c": 1}
    assert plotted["heldout_accuracy"]["mean"] == {
        "masked": 0.5,
        "corrupted": 0.2,
        "all_heads": 0.0,
    }
    assert all(path.is_file() for path in out.values())


def test_the_dbm_figure_counts_tasks_per_head_with_the_papers_heads_outlined() -> None:
    """Every GPT-J head gets a cell at (layer, head), as the paper's heatmap
    places it; the cell shades by the number of masks that keep the head;
    the outlines are the paper's ten heads; and there is no sweep curve."""
    import numpy as np
    from matplotlib.colors import to_hex

    counts = np.zeros((figure.LAYERS, figure.HEADS))
    counts[9, 14] = 3
    counts[0, 0] = 1
    paper = figure.paper_top()
    drawn = figure.dbm_figure(counts, 3, paper, "synthetic")
    assert [ax.get_label() for ax in drawn.axes if ax.get_label() == "sweep"] == []
    (mask,) = [ax for ax in drawn.axes if ax.get_label() == "mask"]
    cells = {p.get_gid(): p for p in mask.patches if p.get_gid()}
    heads = {g for g in cells if g.startswith("head:")}
    assert len(heads) == figure.LAYERS * figure.HEADS
    assert cells["head:L9:H14"].get_xy() == (9, 14)
    assert to_hex(cells["head:L9:H14"].get_facecolor()) == "#3267a8"  # all 3
    assert to_hex(cells["head:L1:H0"].get_facecolor()) == "#e6eef9"  # none
    assert to_hex(cells["head:L0:H0"].get_facecolor()) not in ("#3267a8", "#e6eef9")
    assert len(paper) == figure.TOP
    assert {g for g in cells if g.startswith("outline:")} == {
        f"outline:L{layer}:H{head}" for layer, head in paper
    }


def test_a_task_fitted_by_two_steps_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _synthetic_run(run, {"dbm_fit_1": ["a"], "dbm_fit_2": ["a"]})
    with pytest.raises(ValueError, match="refits"):
        figure.main(
            {"artifacts": run},
            {
                "replication": tmp_path / "r.png",
                "dbm": tmp_path / "d.png",
                "plotted": tmp_path / "p.json",
            },
        )
