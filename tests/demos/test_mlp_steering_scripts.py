"""Checks of the scripts of the ``mlp_steering`` paper package.

The package replicates one cell of Table 5 of Geva et al. (2022): the
Toxicity rate of GPT2 and of 10 Manual Pick. Its workflow runs two
intervention specifications, the grade step
(``demos/papers/workflows/scripts/mlp_steering/score_toxicity.py``) and the
shipped reduce step; the figure script and the checks compute the numbers
the page quotes. These tests pin what those scripts compute without the
package's checkpoints: the paper's two values, the grading step's arms, rule
and label mapping, the figure's values, the greedy cut, the calibration's
paired interval, the paired relative drop and the selection of random
non-toxic vectors, and the agreement of the documents with the paper's
Table 8.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
SCRIPTS = PAPERS / "workflows" / "scripts" / "mlp_steering"
WORKFLOW = PAPERS / "workflows" / "mlp_steering.json"
AMPLIFIED = PAPERS / "protocols" / "mlp_steering_amplified.json"
VALUES = PAPERS / "artifacts" / "data" / "mlp_steering" / "table5_geva2022_values.json"

# The scripts import their siblings by name, as the runner runs them from
# their folder, so the folder goes on the path before they load.
sys.path.insert(0, str(SCRIPTS))
check_api_graded = importlib.import_module("check_api_graded")
check_decoding = importlib.import_module("check_decoding")
check_judge_calibration = importlib.import_module("check_judge_calibration")
check_value_vectors = importlib.import_module("check_value_vectors")
score_toxicity = importlib.import_module("score_toxicity")
table5_figure = importlib.import_module("table5_figure")

pytestmark = pytest.mark.unit


def _steps() -> dict:
    return json.loads(WORKFLOW.read_text())["steps"]


def _declared(step: str, slot: str) -> set[str]:
    return set(_steps()[step]["outputs"][slot]["columns"])


def _flags(arms: dict[str, dict[str, list[float]]]) -> list[dict]:
    """A grade-step table: ``arms[arm][attribute]`` is the 0/1 flag per
    prompt, prompts ``p0``, ``p1``, ... in order."""
    rows = []
    for arm, by_attribute in arms.items():
        for attribute, values in by_attribute.items():
            for i, value in enumerate(values):
                rows.append(
                    {
                        "example_id": f"p{i}",
                        "arm": arm,
                        "attribute": attribute,
                        "value": value,
                    }
                )
    return rows


# ------------------------------------------------------------- paper values


def test_the_paper_values_are_the_printed_toxicity_rates() -> None:
    """The values file holds Table 5's Toxicity rate for GPT2 and 10 Manual
    Pick, one record per workflow step, each the printed percentage over 100.
    The drop the table prints beside 10 Manual Pick, 47%, follows from the
    two rates to within its rounding."""
    wrapped = json.loads(VALUES.read_text())
    assert wrapped["source"]["url"] == "https://arxiv.org/pdf/2203.14680v3"
    assert len(wrapped["source"]["sha256"]) == 64
    records = {r["step"]: r for r in wrapped["records"]}
    assert {r["row"]: r["printed"] for r in records.values()} == {
        "GPT2": "58.5%",
        "10 Manual Pick": "30.8%",
    }
    assert {r["column"] for r in records.values()} == {"Toxicity"}
    paper = table5_figure.paper_values()
    assert paper == {"baseline": 0.585, "amplified": 0.308}
    assert (
        set(paper)
        == set(table5_figure.ARMS)
        == {
            name
            for name, step in _steps().items()
            if step["type"] == "intervention_protocol"
        }
    )
    assert abs(100 * (1 - paper["amplified"] / paper["baseline"]) - 47) <= 0.5


def test_the_paper_values_refuse_a_value_off_its_print(tmp_path: Path) -> None:
    wrapped = json.loads(VALUES.read_text())
    wrapped["records"][1]["value"] = 0.31
    edited = tmp_path / "values.json"
    edited.write_text(json.dumps(wrapped))
    with pytest.raises(ValueError, match="does not follow its rule"):
        table5_figure.paper_values(edited)
    wrapped = json.loads(VALUES.read_text())
    del wrapped["records"][0]
    edited.write_text(json.dumps(wrapped))
    with pytest.raises(ValueError, match="want"):
        table5_figure.paper_values(edited)


# ------------------------------------------------------------- score_toxicity


class _StandIn:
    """The six labels of ``unitary/toxic-bert`` and a probability that
    depends on the text only, so the grade step runs without the checkpoint."""

    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, key: str, revision: str | None) -> None:
        self.key, self.revision = key, revision

    def probabilities(self, texts: list[str]) -> list[list[float]]:
        def probability(text: str) -> float:
            return 0.9 if "bad" in text else 0.5 if "edge" in text else 0.1

        return [[probability(text)] * 6 for text in texts]


def test_the_grade_step_grades_every_arm_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(score_toxicity, "Classifier", _StandIn)
    step = _steps()["grade"]
    inputs = dict(step["inputs"])
    arms = [name for name in inputs if name not in score_toxicity.PARAMETERS]
    assert arms == list(table5_figure.ARMS) == ["baseline", "amplified"]
    for arm, text in zip(arms, ("edge", "bad")):
        table = tmp_path / f"{arm}.json"
        table.write_text(
            json.dumps(
                [
                    {"example_id": "p0", "value": text, "eligible": True},
                    {"example_id": "p1", "value": None, "eligible": True},
                    {"example_id": "p2", "value": "fine", "eligible": True},
                ]
            )
        )
        inputs[arm] = table
    out = tmp_path / "flags.json"
    score_toxicity.main(inputs, {"flags": out})
    rows = json.loads(out.read_text())
    assert [r["arm"] for r in rows[::18]] == arms
    assert len(rows) == 2 * 3 * 6
    assert {key for row in rows for key in row} == _declared("grade", "flags")
    toxic = {
        (r["arm"], r["example_id"]): r["value"]
        for r in rows
        if r["attribute"] == "toxic"
    }
    assert toxic[("amplified", "p0")] == 1.0 and toxic[("baseline", "p2")] == 0.0
    assert toxic[("amplified", "p1")] == 0.0
    # The authors' code counts a score of 0.5 and above, and so does the step.
    assert toxic[("baseline", "p0")] == 1.0


def test_the_grade_step_refuses_an_input_that_is_no_parameter_or_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(score_toxicity, "Classifier", _StandIn)
    table = tmp_path / "baseline.json"
    table.write_text(json.dumps([{"example_id": "p0", "value": "x", "eligible": True}]))
    with pytest.raises(score_toxicity.StepError, match="'treshold' is neither"):
        score_toxicity.main(
            {"baseline": table, "treshold": 0.5}, {"flags": tmp_path / "flags.json"}
        )


def test_the_labels_map_onto_the_table_5_columns() -> None:
    """Five labels stand for five Table 5 columns, ``insult`` for none, and
    the figure's label stands for Toxicity."""
    columns = [c for c in score_toxicity.COLUMNS.values() if c is not None]
    assert len(columns) == len(set(columns)) == 5
    assert score_toxicity.COLUMNS[table5_figure.ATTRIBUTE] == "Toxicity"
    assert set(score_toxicity.COLUMNS) == set(_StandIn.labels)
    assert set(check_judge_calibration.PERSPECTIVE) == set(_StandIn.labels)


# -------------------------------------------------------------- table5_figure


def test_the_figure_draws_the_toxic_rate_of_the_two_arms() -> None:
    """Only the ``toxic`` row of each workflow arm is drawn, beside the
    paper's value; another arm or label in the rates table is left out."""
    rates = [
        {
            "arm": arm,
            "attribute": label,
            "value": value,
            "lower": value - 0.02,
            "upper": value + 0.03,
            "n": 9,
        }
        for arm, value in (("baseline", 0.5), ("amplified", 0.2), ("other", 0.4))
        for label in score_toxicity.COLUMNS
    ]
    rows = table5_figure.plotted_rates(rates, table5_figure.paper_values())
    assert [(r["arm"], r["attribute"]) for r in rows] == [
        ("baseline", "toxic"),
        ("amplified", "toxic"),
    ]
    assert [r["value"] for r in rows] == [0.5, 0.2]
    assert [r["paper"] for r in rows] == [0.585, 0.308]
    assert rows[1]["lower"] == pytest.approx(0.18)
    assert rows[1]["upper"] == pytest.approx(0.23)


def test_the_figure_is_one_outline_and_one_filled_bar() -> None:
    """One bar per arm at the same place: the GPT-2 outline is dashed and
    unfilled, the 10 Manual Pick bar is filled and drawn behind the outline's
    edge; with whiskers, each bar carries one interval."""
    import matplotlib.colors as mcolors

    rows = table5_figure.plotted_rates(
        [
            {
                "arm": arm,
                "attribute": "toxic",
                "value": v,
                "lower": v - 0.02,
                "upper": v + 0.02,
                "n": 9,
            }
            for arm, v in (("baseline", 0.56), ("amplified", 0.23))
        ],
        table5_figure.paper_values(),
    )
    fig = table5_figure.draw(rows, "run", whiskers=True)
    (ax,) = fig.axes
    outline, filled = ax.patches
    assert outline.get_x() == filled.get_x()
    assert outline.get_height() == 0.56 and filled.get_height() == 0.23
    assert outline.get_facecolor()[3] == 0 and outline.get_linestyle() == "--"
    assert mcolors.to_hex(outline.get_edgecolor()) == "#000000"
    assert mcolors.to_hex(filled.get_facecolor()) == table5_figure.COLOUR
    assert outline.zorder > filled.zorder
    assert len(ax.containers) == 4  # two bars, two whiskers
    bare = table5_figure.draw(
        [{"arm": r["arm"], "value": r["paper"]} for r in rows], "paper", whiskers=False
    )
    assert len(bare.axes[0].containers) == 2
    assert bare.axes[0].get_ylim() == ax.get_ylim()


# ------------------------------------------------------------ check_decoding


def test_greedy_ids_cut_at_the_end_of_text() -> None:
    """The tiny model never emits its own end-of-text token on these
    prompts, so the test makes a token it does emit the end of text: every
    row must then stop before that token's first place, and a finished row's
    padding must not reach the output."""
    from tests._helpers.tiny import fresh_tiny_random_gpt2

    model, tokenizer = fresh_tiny_random_gpt2()
    tokenizer.pad_token = tokenizer.eos_token
    prompts = ["The cat sat on", "A", "Once upon a time there was a"]
    free = check_decoding.greedy_ids(
        model, tokenizer, prompts, batch=3, max_new_tokens=6
    )
    assert all(len(row) <= 6 and tokenizer.eos_token_id not in row for row in free)
    # A token that one row emits after its first step ends that row early.
    stop = next(
        row[k] for row in free for k in range(1, len(row)) if row[k] not in row[:k]
    )
    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(stop)
    cut = check_decoding.greedy_ids(
        model, tokenizer, prompts, batch=3, max_new_tokens=6
    )
    expected = [row[: row.index(stop)] if stop in row else row for row in free]
    assert cut == expected
    assert any(len(row) < len(full) for row, full in zip(cut, free))


def test_the_decoding_check_runs_the_specifications_budget() -> None:
    """The check's arms decode the budget of both specifications, whose one
    difference from each other is the amplified model's writes."""
    for path in (AMPLIFIED, AMPLIFIED.with_name("mlp_steering_baseline.json")):
        document = json.loads(path.read_text())
        assert document["method"]["positions"]["continuation"]["generated"] == {
            "max_new_tokens": check_decoding.NEW_TOKENS
        }
    assert {arm.group for arm in check_decoding.ARMS.values()} == set(
        table5_figure.ARMS
    )


# ---------------------------------------------------- check_judge_calibration


def test_the_calibrated_drop_interval_is_paired() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    x = rng.random(200)
    y = (x + 0.2 * rng.standard_normal(200) > 0.5).astype(float)
    base = rng.random(50)
    scores = {"baseline": base, "copy": base.copy(), "lower": base * 0.5}
    text_draws = rng.integers(0, 200, size=(40, 200))
    prompt_draws = rng.integers(0, 50, size=(40, 50))
    out = check_judge_calibration.calibrated(
        x, y, scores, "baseline", text_draws, prompt_draws
    )
    assert out["copy"]["drop"] == 0.0 and out["copy"]["drop_interval"] == [0.0, 0.0]
    assert "drop" not in out["baseline"]
    low, high = out["lower"]["rate_interval"]
    assert low <= out["lower"]["rate"] <= high
    assert out["lower"]["drop"] > 0


# ---------------------------------------------------------- check_value_vectors


def test_the_document_writes_the_table_8_neurons_counted_from_zero() -> None:
    key, coordinates = check_value_vectors.document_coordinates(AMPLIFIED)
    assert key == "openai-community/gpt2-medium"
    assert coordinates == {
        (layer - 1, neuron - 1) for layer, neuron in check_value_vectors.TABLE_8
    }
    # A found token matches one listed token at most: Table 8's v15_1395
    # lists " safe" and "safe", which both strip to "safe".
    assert check_value_vectors.hits(["safe", "safe", "Safe"], ["safe", "x"]) == 1
    assert check_value_vectors.hits(["safe", "safe"], ["safe", "safe", "x"]) == 2


# ------------------------------------------------------------ check_api_graded


def test_the_random_draws_come_from_the_non_toxic_pool() -> None:
    """Appendix A.4: ten vectors drawn at random from those whose
    top-token text scores below 0.1; the same seed gives the same draw."""
    scores = {
        (layer, neuron): (neuron % 5) / 10 for layer in range(3) for neuron in range(20)
    }
    pool = {key for key, score in scores.items() if score < 0.1}
    first = check_api_graded.draw_vectors(scores, 0.1, 10, 0)
    assert len(set(first)) == 10 and set(first) <= pool
    assert first == sorted(first)
    assert check_api_graded.draw_vectors(scores, 0.1, 10, 0) == first
    assert check_api_graded.draw_vectors(scores, 0.1, 10, 1) != first
    with pytest.raises(ValueError, match="only 12"):
        check_api_graded.draw_vectors(scores, 0.1, 13, 0)


def test_the_graded_text_joins_the_stripped_top_tokens() -> None:
    assert check_api_graded.vocab_text([" safe", "safe", " ", "ty"]) == "safe safe ty"


def test_the_check_flags_as_the_grade_step_flags() -> None:
    records = {
        "baseline": {
            "labels": ["toxic"],
            "rows": [
                {"example_id": "p0", "scores": [0.5]},
                {"example_id": "p1", "scores": [0.4999]},
            ],
        }
    }
    values = [row["value"] for row in check_api_graded.flag_rows(records)]
    assert values == [1.0, 0.0]


def test_the_drop_is_one_minus_the_ratio_of_rates() -> None:
    reference = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    treated = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    attributes, vectors = check_api_graded.flag_vectors(
        _flags({"baseline": {"toxic": reference}, "manual_pick": {"toxic": treated}}),
        ["baseline", "manual_pick"],
    )
    draws = check_api_graded.resamples(8, 500, 0)
    (row,) = check_api_graded.drop_rows(
        vectors, attributes, "baseline", ["manual_pick"], draws
    )
    assert row["value"] == pytest.approx(1 - (3 / 8) / (5 / 8))
    assert (row["reference_rate"], row["rate"]) == (5 / 8, 3 / 8)
    # p1, p2, p6 lose the flag and p5 gains it.
    assert (row["down"], row["up"], row["n"]) == (3, 1, 8)
    assert row["lower"] <= row["value"] <= row["upper"]


def test_the_bootstrap_resamples_the_reference_and_the_arm_together() -> None:
    """An arm equal to its reference has a drop of 0 in every resample only
    when both are drawn on the same prompts; a difference of two equal arms
    likewise."""
    flags = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    other = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    attributes, vectors = check_api_graded.flag_vectors(
        _flags(
            {
                "baseline": {"toxic": flags},
                "same": {"toxic": flags},
                "other": {"toxic": other},
                "other_copy": {"toxic": other},
            }
        ),
        ["baseline", "same", "other", "other_copy"],
    )
    draws = check_api_graded.resamples(10, 300, 0)
    (same,) = check_api_graded.drop_rows(
        vectors, attributes, "baseline", ["same"], draws
    )
    assert (same["value"], same["lower"], same["upper"]) == (0.0, 0.0, 0.0)
    (difference,) = check_api_graded.difference_rows(
        vectors, attributes, "baseline", [["other", "other_copy"]], draws
    )
    assert (difference["value"], difference["lower"], difference["upper"]) == (
        0.0,
        0.0,
        0.0,
    )


def test_a_reference_rate_of_zero_has_no_drop() -> None:
    attributes, vectors = check_api_graded.flag_vectors(
        _flags(
            {
                "baseline": {"threat": [0.0, 0.0, 0.0]},
                "draw_0": {"threat": [0.0, 1.0, 0.0]},
            }
        ),
        ["baseline", "draw_0"],
    )
    draws = check_api_graded.resamples(3, 50, 0)
    (row,) = check_api_graded.drop_rows(
        vectors, attributes, "baseline", ["draw_0"], draws
    )
    assert row["lower"] is None and row["upper"] is None
    assert row["value"] != row["value"]  # nan: no rate to drop from


def test_the_drops_refuse_unaligned_arms() -> None:
    rows = _flags({"baseline": {"toxic": [1.0, 0.0]}, "draw_0": {"toxic": [1.0, 0.0]}})
    rows[-1]["example_id"] = "elsewhere"
    with pytest.raises(ValueError, match="other prompts"):
        check_api_graded.flag_vectors(rows, ["baseline", "draw_0"])
    with pytest.raises(ValueError, match="no flags"):
        check_api_graded.flag_vectors(rows, ["baseline", "draw_1"])
