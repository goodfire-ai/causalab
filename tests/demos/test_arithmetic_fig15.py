"""The Figure 15 package: its table, its figure script and its page.

``demos/papers/workflows/scripts/arithmetic_fig15/build_dataset.py`` turns
the paper's wrapped pairs into the table the scan reads, and
``fig15_figure.py`` turns the scan's per-example tables into the plotted
values. The labels are checked against a plain recomputation of the paper's
task (Appendix D.2), the plotted values against an independent mean over
synthetic tables, the committed figure values against the committed table,
and the page's worked example against the table it runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
SCRIPTS = PAPERS / "workflows" / "scripts" / "arithmetic_fig15"
DATA = PAPERS / "artifacts" / "data" / "arithmetic_fig15"
PAGE = PAPERS / "arithmetic_fig15.md"
PROTOCOL = PAPERS / "protocols" / "arithmetic_fig15_scan.json"
PLOTTED = PAPERS / "artifacts" / "figures" / "arithmetic_fig15" / "fig15_plotted.json"

DAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
OFFSETS: tuple[str, ...] = tuple(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split()
)
PROMPT = re.compile(r"Q: What day is (\w+) days after (\w+)\?")


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"arithmetic_fig15_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer(offset: str, day: str) -> str:
    """The day ``offset`` days after ``day``, spelled as the model answers."""
    return " " + DAYS[(DAYS.index(day) + OFFSETS.index(offset) + 1) % 7]


def _variables(prompt: str) -> tuple[str, str]:
    match = PROMPT.fullmatch(prompt.removesuffix("\nA:"))
    assert match is not None, prompt
    return match.group(1), match.group(2)


# -- the table ---------------------------------------------------------------


@pytest.mark.unit
def test_the_table_holds_the_paper_pairs_in_file_order() -> None:
    wrapper = json.loads((DATA / "filtered_dataset.json").read_text())
    rows = json.loads((DATA / "data.json").read_text())
    assert wrapper["n"] == len(wrapper["records"]) == len(rows) == 4096
    for row, record in zip(rows, wrapper["records"], strict=True):
        assert row["input"] == record["input"]["raw_input"]
        (counterfactual,) = record["counterfactual_inputs"]
        assert row["counterfactual_inputs"] == [counterfactual["raw_input"]]


@pytest.mark.numerical_unit
def test_the_labels_are_the_answers_of_each_interchange() -> None:
    """Appendix D.2: patching the output day gives the counterfactual's
    answer, the offset gives the base day plus the counterfactual offset,
    and the input day gives the counterfactual day plus the base offset."""
    rows = json.loads((DATA / "data.json").read_text())
    for row in rows:
        offset, day = _variables(row["input"])
        (prompt,) = row["counterfactual_inputs"]
        offset_cf, day_cf = _variables(prompt)
        expected = {
            "base_answer": _answer(offset, day),
            "label": _answer(offset_cf, day_cf),
            "label_offset": _answer(offset_cf, day),
            "label_input_day": _answer(offset, day_cf),
        }
        for column, answer in expected.items():
            assert row[column] == answer, (row["input"], prompt, column)
            assert row[f"{column}_forms"] == [answer, answer.strip()]


def _rewrapped(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None], *, keep_digest: bool
) -> Path:
    """A copy of the wrapper with ``edit`` applied to its first record; with
    ``keep_digest`` false the digest follows the edited records."""
    wrapper = json.loads((DATA / "filtered_dataset.json").read_text())
    records = wrapper["records"]
    edit(records[0])
    if not keep_digest:
        original = json.dumps(
            {
                "input": [r["input"] for r in records],
                "counterfactual_inputs": [r["counterfactual_inputs"] for r in records],
            },
            indent=2,
        ).encode()
        wrapper["source_sha256"] = hashlib.sha256(original).hexdigest()
    path = tmp_path / "filtered_dataset.json"
    path.write_text(json.dumps(wrapper))
    return path


def _answer_monday(record: dict[str, Any]) -> None:
    record["input"]["output"] = "Monday"


def _two_counterfactuals(record: dict[str, Any]) -> None:
    record["counterfactual_inputs"].append(record["counterfactual_inputs"][0])


@pytest.mark.unit
def test_records_that_are_not_the_source_bytes_are_refused(tmp_path: Path) -> None:
    builder = _script("build_dataset")
    path = _rewrapped(tmp_path, _answer_monday, keep_digest=True)
    with pytest.raises(ValueError, match="rebuild bytes"):
        builder.build(path)


@pytest.mark.unit
def test_a_record_the_causal_model_disagrees_with_is_refused(tmp_path: Path) -> None:
    """The first record is ``six days after Sunday``, which is Saturday."""
    builder = _script("build_dataset")
    path = _rewrapped(tmp_path, _answer_monday, keep_digest=False)
    with pytest.raises(
        ValueError, match="the causal model gives output_day='Saturday'"
    ):
        builder.build(path)


@pytest.mark.unit
def test_a_record_with_two_counterfactuals_is_refused_by_name(tmp_path: Path) -> None:
    builder = _script("build_dataset")
    path = _rewrapped(tmp_path, _two_counterfactuals, keep_digest=False)
    with pytest.raises(
        ValueError,
        match=r"'Q: What day is six days after Sunday\?\\nA:': 2 counterfactuals",
    ):
        builder.build(path)


@pytest.mark.unit
def test_the_builder_writes_into_a_directory_it_creates(tmp_path: Path) -> None:
    """``--out`` names a directory that need not exist yet, and the table
    written there is the committed one; a ``.json`` path of another name is
    refused."""
    builder = _script("build_dataset")
    out = tmp_path / "new" / "arithmetic_fig15"
    assert builder.main(["--out", str(out)]) == 0
    assert (out / "data.json").read_bytes() == (DATA / "data.json").read_bytes()
    assert builder.main(["--out", str(out), "--check"]) == 0
    assert builder.main(["--out", str(tmp_path / "rows.json")]) == 1
    assert not (tmp_path / "rows.json").exists()


# -- the figure script -------------------------------------------------------

POSITIONS = {
    "number": '{"variable": "offset"}',
    "input": '{"variable": "input_day"}',
    "last_token": '{"index": -1}',
}
VARIABLES = ("output_day", "offset", "input_day")


@pytest.mark.unit
def test_position_coordinates_map_to_the_papers_columns() -> None:
    figure = _script("fig15_figure")
    for name, coordinate in POSITIONS.items():
        assert figure.position_name(coordinate) == name
        assert figure.position_name(json.loads(coordinate)) == name
    with pytest.raises(figure.StepError, match="unrecognised position"):
        figure.position_name('{"variable": "premod"}')


@pytest.mark.unit
def test_depth_labels_are_the_papers_rows() -> None:
    figure = _script("fig15_figure")
    assert [figure.depth_label(d) for d in (0, 1, 2, 32)] == [
        "Embed",
        "L0",
        "L1",
        "L31",
    ]


def _write_scan(scan: Path, values: dict[str, np.ndarray], axes: list[str]) -> None:
    """One table per variable in the runner's row format, and the sidecar.

    ``values[variable]`` is (depths, 3 positions, pairs), and NaN becomes a
    null value, as the runner writes an ineligible row."""
    scan.mkdir(parents=True, exist_ok=True)
    for variable, cube in values.items():
        rows = []
        for depth in range(cube.shape[0]):
            for p, coordinate in enumerate(POSITIONS.values()):
                for example in range(cube.shape[2]):
                    value = float(cube[depth, p, example])
                    rows.append(
                        {
                            "example_id": str(example),
                            "metric": f"iia_{variable}",
                            "value": None if np.isnan(value) else value,
                            "axes.depth": depth,
                            "positions.tap": coordinate,
                        }
                    )
        (scan / f"iia_{variable}.json").write_text(json.dumps(rows))
    (scan / "_step.json").write_text(json.dumps({"axes": axes}))


def _synthetic(depths: int, pairs: int, seed: int) -> dict[str, np.ndarray]:
    """0/1 match values with one success rate per variable, depth and
    position, and a few null rows."""
    rng = np.random.default_rng(seed)
    out = {}
    for variable in VARIABLES:
        rate = rng.uniform(0.1, 0.9, (depths, 3, 1))
        cube = (rng.uniform(size=(depths, 3, pairs)) < rate).astype(float)
        cube[rng.uniform(size=cube.shape) < 0.02] = np.nan
        out[variable] = cube
    return out


def _run(figure: ModuleType, scan: Path, outputs: dict[str, Path]) -> None:
    figure.main({f"iia_{v}": scan / f"iia_{v}.json" for v in VARIABLES}, outputs)


@pytest.mark.numerical_unit
def test_plotted_values_are_the_mean_over_pairs(tmp_path: Path) -> None:
    figure = _script("fig15_figure")
    values = _synthetic(depths=4, pairs=60, seed=0)
    _write_scan(tmp_path / "scan", values, ["axes.depth", "positions.tap"])
    target = tmp_path / "plotted.json"
    _run(figure, tmp_path / "scan", {"plotted": target})
    plotted = json.loads(target.read_text())
    assert len(plotted) == len(VARIABLES) * 4 * 3
    # The paper's order: panel, then depth from the embeddings up, then column.
    assert [(r["variable"], r["depth"], r["position"]) for r in plotted] == [
        (v, d, p) for v in VARIABLES for d in range(4) for p in POSITIONS
    ]
    for row in plotted:
        cube = values[row["variable"]]
        drawn = cube[row["depth"], list(POSITIONS).index(row["position"])]
        assert row["iia"] == pytest.approx(np.nanmean(drawn), abs=1e-12)
        assert row["n"] == int(np.sum(~np.isnan(drawn)))
        assert row["layer"] == figure.depth_label(row["depth"])


@pytest.mark.unit
def test_the_figures_are_drawn(tmp_path: Path) -> None:
    figure = _script("fig15_figure")
    _write_scan(
        tmp_path / "scan", _synthetic(3, 10, 1), ["axes.depth", "positions.tap"]
    )
    target = tmp_path / "all.png"
    _run(figure, tmp_path / "scan", {"replication": target})
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_an_output_the_script_does_not_write_is_refused(tmp_path: Path) -> None:
    """One specification draws the whole figure, so no panel is drawn alone."""
    figure = _script("fig15_figure")
    _write_scan(tmp_path / "scan", _synthetic(2, 5, 3), ["axes.depth", "positions.tap"])
    with pytest.raises(figure.StepError, match=r"unknown outputs \['offset'\]"):
        _run(figure, tmp_path / "scan", {"offset": tmp_path / "offset.png"})
    assert not (tmp_path / "offset.png").exists()


@pytest.mark.unit
def test_a_table_without_the_scan_axes_is_refused(tmp_path: Path) -> None:
    figure = _script("fig15_figure")
    _write_scan(tmp_path / "scan", _synthetic(2, 5, 2), ["positions.tap"])
    with pytest.raises(figure.StepError, match="does not carry the axis 'axes.depth'"):
        _run(figure, tmp_path / "scan", {"plotted": tmp_path / "plotted.json"})


# -- the committed figure ----------------------------------------------------

#: Each variable's label column in ``data.json``.
LABEL_COLUMNS = {
    "output_day": "label",
    "offset": "label_offset",
    "input_day": "label_input_day",
}


def _share(rows: list[dict[str, Any]], column: str, answer: str) -> float:
    return sum(row[column] == row[answer] for row in rows) / len(rows)


@pytest.mark.unit
def test_the_committed_figure_is_drawn_from_the_committed_table() -> None:
    """Every plotted value is a mean over all pairs of ``data.json``, and the
    sites where the patched forward is a clean forward equal shares of that
    table.

    The embeddings at the last token (the same token in both prompts) and L31
    at the other two tokens (which the answer does not read) leave the base
    forward, and L31 at the last token gives the counterfactual forward. The
    committed run answers every prompt of the pairs (its output-day value at
    L31, last token, is 1), so each such value is the share of pairs whose
    label equals the base or the counterfactual answer, to every digit. A
    table rebuilt from other pairs, or a figure left from another table,
    breaks these equalities. A rerun whose clean answers differ breaks them
    too, and then the page's Floors and ceiling paragraph changes with it.
    """
    rows = json.loads((DATA / "data.json").read_text())
    plotted = json.loads(PLOTTED.read_text())
    assert len(plotted) == len(VARIABLES) * 33 * len(POSITIONS)
    assert {record["n"] for record in plotted} == {len(rows)}
    value = {(r["variable"], r["layer"], r["position"]): r["iia"] for r in plotted}
    for variable, column in LABEL_COLUMNS.items():
        floor = _share(rows, column, "base_answer")
        for site in (("Embed", "last_token"), ("L31", "number"), ("L31", "input")):
            assert value[(variable, *site)] == floor, (variable, site)
        ceiling = _share(rows, column, "cf_answer")
        assert value[(variable, "L31", "last_token")] == ceiling, variable
    assert value[("output_day", "L31", "last_token")] == 1.0


# -- the page ----------------------------------------------------------------


@pytest.mark.unit
def test_the_pages_examples_are_answered_as_the_task_says() -> None:
    """Every prompt the page answers, in the text or a caption, is followed
    by the day the causal model gives."""
    text = PAGE.read_text()
    answered = re.findall(
        r"`Q: What day is (\w+) days after (\w+)\?`\s+(?:with|-->)\s+`( \w+)`", text
    )
    assert answered, "the page answers no example prompt"
    for offset, day, answer in answered:
        assert answer == _answer(offset, day), (offset, day, answer)


@pytest.mark.unit
def test_the_sample_comments_are_one_row_of_the_table() -> None:
    text = PAGE.read_text()
    (base,) = set(re.findall(r"// base sample: (Q: [^\n]*\?)", text))
    (counterfactual,) = set(
        re.findall(r"// counterfactual sample: (Q: [^\n]*\?)", text)
    )
    rows = json.loads((DATA / "data.json").read_text())
    assert any(
        row["input"] == f"{base}\nA:"
        and row["counterfactual_inputs"] == [f"{counterfactual}\nA:"]
        for row in rows
    ), (base, counterfactual)


@pytest.mark.unit
def test_the_parameter_table_names_the_documents_dataset() -> None:
    document = json.loads(PROTOCOL.read_text())
    (named,) = {side["dataset"] for side in document["data"].values()}
    (quoted,) = re.findall(
        r"^\| `data\.\*\.dataset` \| `([^`]+)` \|", PAGE.read_text(), re.M
    )
    assert quoted == named
