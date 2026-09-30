"""Checks of what the lookbacks package's scripts compute.

``demos/papers/lookbacks.md`` replicates Prakash et al. 2025 (arXiv:2505.14685)
Figures 4b, 5b, 6b and 13 on the paper's own 80 validation pairs. These tests
load the package's scripts by path, as a reader runs them:

* ``build_dataset.py`` draws the authors' candidate pairs. The oracle is a
  digest of the authors' generator output (``Nix07/mind`` at 0579347,
  ``get_reversed_sent_diff_state_counterfacts`` and
  ``get_reversed_sentence_counterfacts`` after ``random.seed(123456)``,
  320 candidates each): the JSON of ``[clean_prompt, counterfactual_prompt,
  target]`` per candidate. Every answer form is one Llama-3 token, checked
  with a public tokenizer that has Llama-3's vocabulary, so a multi-token
  form fails here and not after the 70B weights load.
* ``lookbacks_figure.py`` joins each curve with its baseline, averages it per
  layer, puts the paper's value beside it and reports the screening steps.
* ``paper_values.py`` reads the paper's values from a clone of the authors'
  repository (checked on a small git repository made here); the committed
  file is checked for its shape, and the committed curves against it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
SCRIPTS = PAPERS / "workflows" / "scripts" / "lookbacks"
DATA = PAPERS / "artifacts" / "data" / "lookbacks"
FIGURES = PAPERS / "artifacts" / "figures" / "lookbacks"

#: sha256 of the authors' 320 candidates per design (module docstring).
AUTHORS_CANDIDATES = {
    True: "245e3b78595de5a2d4f50498939ca33a46600b3501ef741ee85f7fe319e598de",
    False: "1bdc90b29fd488df4620224efa9714aa54c0bddd7ab94a65bc2cd05efbd08d18",
}
TABLES = {"data_restate.json": True, "data_reorder.json": False}
#: Every committed table: its design, the candidates it holds and the split
#: of each row.
ALL_TABLES = {
    "data_restate.json": (True, range(80, 160), ["first"] * 40 + ["second"] * 40),
    "data_reorder.json": (False, range(80, 160), ["first"] * 40 + ["second"] * 40),
    "data_restate_screen.json": (True, range(0, 80), ["screen"] * 80),
    "data_reorder_screen.json": (False, range(0, 80), ["screen"] * 80),
}
#: A public tokenizer with the vocabulary of the gated Llama-3 checkpoints:
#: its 128,000 ordinary tokens equal theirs, and only the names of reserved
#: special tokens differ. Pinned to a snapshot.
LLAMA3_TOKENIZER = "yujiepan/llama-3-tiny-random"
LLAMA3_TOKENIZER_REVISION = "5d8b95d4877408d2ddd2732a371bd4ddb51fb67c"


def _script(name: str) -> ModuleType:
    """A script of the package, imported from its file."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lookbacks_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _script("build_dataset")


@pytest.fixture(scope="module")
def figure() -> ModuleType:
    return _script("lookbacks_figure")


# --------------------------------------------------------------------------- #
# build_dataset.py
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("restate", [True, False], ids=["restate", "reorder"])
def test_the_candidates_are_the_authors(builder: ModuleType, restate: bool) -> None:
    """Every one of the 320 candidates has the authors' prompts and target,
    in the authors' order."""
    model = builder.belief_model()
    rows = []
    for candidate in builder.authors_candidates(restate=restate):
        pair = builder.pair(model, candidate)
        rows.append(
            [
                pair["input"]["raw_input"],
                pair["counterfactual_inputs"][0]["raw_input"],
                builder.authors_target(candidate),
            ]
        )
    assert len(rows) == 320
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    assert digest == AUTHORS_CANDIDATES[restate]


@pytest.mark.parametrize("name", sorted(ALL_TABLES))
def test_the_tables_hold_their_candidates(builder: ModuleType, name: str) -> None:
    """A figure table is candidates 80 to 159 in order, in two splits of
    40, and a screening table candidates 0 to 79 in one split. Each keeps
    the pairs whose object is ``container``."""
    restate, indices, splits = ALL_TABLES[name]
    rows = json.loads((DATA / name).read_text())
    model = builder.belief_model()
    candidates = builder.authors_candidates(restate=restate)[
        indices.start : indices.stop
    ]
    assert [row["input"] for row in rows] == [
        builder.pair(model, c)["input"]["raw_input"] for c in candidates
    ]
    assert [row["counterfactual_inputs"][0] for row in rows] == [
        builder.pair(model, c)["counterfactual_inputs"][0]["raw_input"]
        for c in candidates
    ]
    assert [row["split"] for row in rows] == splits
    assert any("container" in (row["obj_a"], row["obj_b"]) for row in rows)


def test_every_answer_form_is_one_llama3_token(builder: ModuleType) -> None:
    """A ``match`` metric refuses a form that is not one token, and on the
    70B model it would do so after the weights load. Every form a table
    holds, and every form the causal model's scoring credits, is one token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA3_TOKENIZER, revision=LLAMA3_TOKENIZER_REVISION
    )
    forms = {
        form
        for name in ALL_TABLES
        for row in json.loads((DATA / name).read_text())
        for column, value in row.items()
        if column.endswith("_forms")
        for form in value
    }
    assert forms
    forms |= {f for w in builder.STATES + [builder.UNKNOWN] for f in builder.forms(w)}
    longer = {
        form: ids
        for form in sorted(forms)
        if len(ids := tokenizer(form, add_special_tokens=False)["input_ids"]) != 1
    }
    assert not longer


ANCHORS = (
    "intro_a",
    "intro_b",
    "char_a_action",
    "char_b_action",
    "obj_a_action",
    "obj_b_action",
    "state_a_mention",
    "state_b_mention",
)


@pytest.mark.parametrize("name", sorted(ALL_TABLES))
def test_every_anchor_occurs_once_in_its_prompt(name: str) -> None:
    """A ``{"variable": ...}`` position needs its anchor exactly once in the
    prompt, on both sides of every pair; ``container and`` would also match
    the instruction, which ``container and fills`` does not."""
    for row in json.loads((DATA / name).read_text()):
        sides = [
            (row["input"], row),
            (
                row["counterfactual_inputs"][0],
                row["counterfactual_inputs_variables"][0],
            ),
        ]
        for prompt, variables in sides:
            for anchor in ANCHORS:
                assert prompt.count(variables[anchor]) == 1, (anchor, variables[anchor])


def test_each_restate_label_is_the_other_original_state(builder: ModuleType) -> None:
    """The pointer and source labels name the original's other state, the
    ``target`` of the authors' generator, and the payload label the
    counterfactual's own answer."""
    for row in json.loads((DATA / "data_restate.json").read_text()):
        other = row["state_b"] if row["q"] == "a" else row["state_a"]
        assert row["label_pointer"] == row["label_source"] == " " + other
        assert row["label_pointer_forms"] == builder.forms(other)
        assert row["label"] == row["cf_answer"] != row["base_answer"]


# --------------------------------------------------------------------------- #
# lookbacks_figure.py
# --------------------------------------------------------------------------- #


def _step(root: Path, name: str, axis: str | None, tables: dict[str, list]) -> None:
    """A fake step directory: its record names the sweep axis, and each
    table is a list of metric rows."""
    step = root / name
    step.mkdir(parents=True)
    (step / "_step.json").write_text(json.dumps({"axes": [axis] if axis else []}))
    for table, rows in tables.items():
        (step / f"{table}.json").write_text(json.dumps(rows))


def test_a_curve_is_the_mean_per_layer_over_both_populations(
    figure: ModuleType, tmp_path: Path
) -> None:
    """Example 1 fails its baseline on one prompt: it counts in the curve
    over every pair and not in the correct-pair curve. Ids written as ints
    and as strings join as one."""
    axis = "sites.target.layers"
    inputs = {}
    for split, offset in (("first", 0), ("second", 10)):
        base = f"baseline_{split}"
        _step(
            tmp_path,
            base,
            None,
            {
                "correct_base": [
                    {"example_id": offset, "value": 1.0},
                    {"example_id": offset + 1, "value": 1.0},
                ],
                "correct_cf": [
                    {"example_id": offset, "value": 1.0},
                    {
                        "example_id": offset + 1,
                        "value": 0.0 if split == "first" else 1.0,
                    },
                ],
            },
        )
        _step(
            tmp_path,
            f"scan_{split}",
            axis,
            {
                "iia": [
                    {"example_id": str(offset + e), axis: layer, "value": value}
                    for layer, values in ((0, (0.0, 1.0)), (1, (1.0, 1.0)))
                    for e, value in enumerate(values)
                ]
            },
        )
        for table in ("correct_base", "correct_cf"):
            inputs[f"baseline_{split}/{table}"] = tmp_path / base / f"{table}.json"
        inputs[f"scan_{split}/iia"] = tmp_path / f"scan_{split}" / "iia.json"
    rows = figure.curve_rows(inputs, "scan", "iia", "baseline")
    table = figure.curve_table(rows)
    assert table["layer"].tolist() == [0, 1]
    assert table["n_all"].tolist() == [4, 4]
    assert table["n_correct"].tolist() == [3, 3]
    assert table["iia_all"].tolist() == [0.5, 1.0]
    # correct pairs at layer 0: example 0 (0.0), 10 (0.0), 11 (1.0)
    assert table["iia_correct"].tolist() == pytest.approx([1 / 3, 1.0])


def test_the_screen_report_counts_pairs_answered_on_both_prompts(
    figure: ModuleType, tmp_path: Path
) -> None:
    """A pair counts only when both of its prompts are answered; the total
    is the number of pairs the step scored."""
    inputs = {}
    for step, wrong in zip(figure.SCREEN_STEPS, (None, 1)):
        tables = {
            table: [
                {
                    "example_id": e,
                    "value": 0.0 if (e == wrong and table == "correct_cf") else 1.0,
                }
                for e in range(3)
            ]
            for table in figure.BASELINE_TABLES
        }
        _step(tmp_path, step, None, tables)
        for table in figure.BASELINE_TABLES:
            inputs[f"{step}/{table}"] = tmp_path / step / f"{table}.json"
    restate, reorder = figure.SCREEN_STEPS
    assert figure.screen_report(inputs) == [
        f"{restate}: 3 of 3 pairs answered on both prompts",
        f"{reorder}: 2 of 3 pairs answered on both prompts",
    ]


def test_the_screen_steps_run_the_baseline_on_the_screening_tables(
    figure: ModuleType,
) -> None:
    """Each screening step runs the baseline document on the screening
    table of its design, which holds candidates 0 to 79."""
    steps = json.loads((PAPERS / "workflows" / "lookbacks.json").read_text())["steps"]
    for step in figure.SCREEN_STEPS:
        design = step.removeprefix("screen_")
        assert steps[step]["document"] == "../protocols/lookbacks_clean_accuracy.json"
        table = f"lookbacks/data_{design}_screen#screen"
        assert steps[step]["set"] == {
            "data.base.dataset": table,
            "data.counterfactual.dataset": table,
        }


def test_the_paper_column_sits_beside_the_sampled_layers(figure: ModuleType) -> None:
    import pandas as pd

    table = pd.DataFrame(
        {
            "figure": ["4b", "4b", "5b"],
            "curve": ["pointer", "pointer", "binding"],
            "layer": [0, 1, 0],
        }
    )
    out = figure.with_paper(table, {("4b", "pointer", 1): 0.5})
    assert out["paper"].tolist() == [None, 0.5, None]
    with pytest.raises(figure.StepError, match="no point"):
        figure.with_paper(table, {("6b", "late", 3): 0.5})


def test_every_curve_reads_a_table_its_steps_save(figure: ModuleType) -> None:
    """Each panel curve names a table that the documents of its two split
    steps save, and each baseline a step of the workflow."""
    workflow = json.loads((PAPERS / "workflows" / "lookbacks.json").read_text())
    steps = workflow["steps"]
    for spec in figure.PANELS.values():
        for split in figure.SPLITS:
            assert f"{spec['baseline']}_{split}" in steps
        for _curve, step, table, _legend, _color in spec["curves"]:
            for split in figure.SPLITS:
                document = steps[f"{step}_{split}"]["document"]
                saves = json.loads((PAPERS / "workflows" / document).read_text())
                files = {s["file_path"] for s in saves["method"]["save"]}
                assert f"{table}.json" in files, (step, table)


def test_the_script_writes_the_committed_figures(
    figure: ModuleType, tmp_path: Path
) -> None:
    """On a fabricated run tree with every layer the paper samples, the
    command writes one image per panel and the plotted values, and these are
    the files committed under ``artifacts/figures/lookbacks/``. The page
    shows each panel beside the paper's crop of it, so no image holds all
    four panels (``docs/paper_replications.md``, The figures)."""
    run = tmp_path / "run"
    examples = range(2)
    baselines = {spec["baseline"] for spec in figure.PANELS.values()}
    both_right = {
        table: [{"example_id": e, "value": 1.0} for e in examples]
        for table in figure.BASELINE_TABLES
    }
    for split in figure.SPLITS:
        for baseline in baselines:
            _step(run, f"{baseline}_{split}", None, both_right)
    for step in figure.SCREEN_STEPS:
        _step(run, step, None, both_right)
    axis = "sites.target.layers"
    scans: dict[str, set[str]] = {}
    for spec in figure.PANELS.values():
        for _curve, step, table, _legend, _color in spec["curves"]:
            scans.setdefault(step, set()).add(table)
    for step, tables in scans.items():
        rows = [
            {"example_id": e, axis: layer, "value": 1.0}
            for layer in range(80)
            for e in examples
        ]
        for split in figure.SPLITS:
            _step(run, f"{step}_{split}", axis, {t: rows for t in tables})
    out = tmp_path / "figures"
    assert figure.cli(["--artifacts", str(run), "--figures", str(out)]) == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == sorted(p.name for p in FIGURES.iterdir())
    assert "lookbacks_replication.png" not in written


# --------------------------------------------------------------------------- #
# the paper's values, and the committed curves against them
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _mind(root: Path, values_script: ModuleType, layers: dict[int, float]) -> Path:
    """A committed repository in the authors' layout: one file per layer
    under each experiment folder `paper_values.CURVES` names."""
    mind = root / "mind"
    for experiment in values_script.CURVES.values():
        folder = mind / values_script.RESULTS / experiment
        folder.mkdir(parents=True)
        for layer, accuracy in layers.items():
            payload = {"full_rank": {"accuracy": accuracy}}
            (folder / f"{layer}.json").write_text(json.dumps(payload))
    _git(root, "init", "-q", str(mind))
    _git(mind, "add", ".")
    _git(
        mind,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--no-verify",
        "-m",
        "layer files",
    )
    return mind


def test_collect_reads_one_record_per_layer_file(tmp_path: Path) -> None:
    """On a small repository in the authors' layout, `collect` names its
    commit, sorts the layer files by number (``10`` after ``2``), counts
    each value over the 80 pairs and hashes each file; ``--check`` then
    passes on the written file and fails on a changed one."""
    values_script = _script("paper_values")
    layers = {10: 0.5, 0: 0.0125, 2: 1.0}
    mind = _mind(tmp_path, values_script, layers)
    values = values_script.collect(mind)
    assert values["commit"] == _git(mind, "rev-parse", "HEAD")
    assert values["source"].endswith(
        f"/tree/{values['commit']}/{values_script.RESULTS}"
    )
    records = values["records"]
    assert len(records) == len(values_script.CURVES) * len(layers)
    first = [r for r in records if (r["figure"], r["curve"]) == ("4b", "pointer")]
    assert [r["layer"] for r in first] == [0, 2, 10]
    assert [r["count"] for r in first] == [1, 80, 40]
    for record in records:
        raw = (mind / record["file"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
    out = tmp_path / "values.json"
    assert values_script.main(["--mind", str(mind), "--out", str(out)]) == 0
    assert values_script.main(["--mind", str(mind), "--out", str(out), "--check"]) == 0
    out.write_bytes(out.read_bytes().replace(b'"count": 40', b'"count": 41', 1))
    assert values_script.main(["--mind", str(mind), "--out", str(out), "--check"]) == 1


def test_collect_refuses_a_value_that_is_not_a_count_over_80(tmp_path: Path) -> None:
    values_script = _script("paper_values")
    mind = _mind(tmp_path, values_script, {0: 0.33})
    with pytest.raises(ValueError, match="not a count over 80"):
        values_script.collect(mind)


def test_the_paper_values_file_is_the_encoded_records() -> None:
    """The committed file is `paper_values.encode` of its own content, names
    the commit it was read at, and holds counts over the 80 pairs for the
    curves `paper_values.CURVES` maps."""
    values_script = _script("paper_values")
    raw = (DATA / "lookbacks_prakash2025_values.json").read_bytes()
    values = json.loads(raw)
    assert values_script.encode(values) == raw
    assert values["commit"] in values["source"]
    curves = {(r["figure"], r["curve"]) for r in values["records"]}
    assert curves == set(values_script.CURVES)
    for record in values["records"]:
        assert record["count"] == round(record["accuracy"] * 80)
        assert abs(record["accuracy"] * 80 - record["count"]) < 1e-9
        assert len(record["sha256"]) == 64


def test_the_committed_curves_match_the_paper_within_three_of_80() -> None:
    """At every layer the paper samples, each committed curve over the pairs
    the model answers correctly is within 3/80 of the paper's value: the
    spread the authors' code shows between bf16 and fp16 on fixed pairs."""
    plotted = json.loads((FIGURES / "lookbacks_plotted.json").read_text())
    compared = [row for row in plotted if row["paper"] is not None]
    assert len(compared) == 170
    off = [
        (row["figure"], row["curve"], row["layer"], row["iia_correct"], row["paper"])
        for row in compared
        if abs(row["iia_correct"] - row["paper"]) > 3 / 80 + 1e-9
    ]
    assert not off
