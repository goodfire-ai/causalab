"""``causalab/causal/scoring.py`` — one immutable ``ScoringSpec`` per task.

The torch-free tier: construction and its refusals, the frozen
object and its identity (T12), the derived read-only views a ``CausalModel``
exposes, the grader's semantics under both string modes and both undeclared /
invalid-output policies, the probability path's form groups as a derivation
of the spec, the bespoke ``full_string_checker``, and ``check_scoring`` — the
table-side comparison the ``validate --data`` pass and the executor both make.

Every test here is arithmetic over small hand-built specs; no model, no
tokenizer, no table on disk.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.scoring import (
    GRADE_RECORD_IDENTITY,
    INVALID_OUTPUT_POLICIES,
    PROTOCOL_MODES,
    SCORING_FIELDS,
    SCORING_RESULTS,
    STRING_MODE_COLUMN,
    STRING_MODES,
    UNDECLARED_VALUE_POLICIES,
    ScoringError,
    ScoringMismatch,
    ScoringSpec,
    build_output_tokens,
    check_scoring,
    declared_modes,
    table_scoring,
)

pytestmark = pytest.mark.unit

WEEKDAYS = ["Monday", "Tuesday", "Wednesday"]


def _spec(**overrides) -> ScoringSpec:
    kwargs = {"forms": {"weekday": build_output_tokens(WEEKDAYS)}}
    kwargs.update(overrides)
    return ScoringSpec(**kwargs)


def _model(spec: ScoringSpec | None) -> CausalModel:
    @mechanism
    def equations(weekday: Dom(WEEKDAYS)):
        raw_input = V(f"Today is {weekday}. Tomorrow is", domain=Dom(str))  # noqa: F841
        raw_output = V(" " + weekday, domain=Dom(str))
        return raw_output

    return CausalModel(equations, id="weekdays_fixture", scoring=spec)


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_the_authored_defaults_and_the_derived_fields():
    spec = _spec()
    assert spec.answer_variable == "weekday"  # the sole declared variable
    assert spec.string_mode == "exact" and spec.protocol_mode == "exact"
    assert spec.full_string_checker is None
    assert spec.undeclared_value == "refuse" and spec.invalid_output == "incorrect"
    assert spec.version == 1


def test_every_field_is_in_the_fields_tuple_once():
    """The census in ``causalab/tasks/README.md`` reads ``SCORING_FIELDS``; it
    has to be the dataclass's fields, or the table documents a different
    object than the one constructed."""
    declared = tuple(f.name for f in dataclasses.fields(ScoringSpec))
    assert set(declared) == set(SCORING_FIELDS)
    assert len(set(SCORING_FIELDS)) == len(SCORING_FIELDS)
    assert tuple(_spec().identity()) == SCORING_FIELDS


def test_protocol_mode_is_derived_never_authored():
    prefix = _spec(string_mode="prefix")
    assert prefix.protocol_mode == "first_token"
    with pytest.raises(TypeError):
        ScoringSpec(forms={"w": {"a": ["a"]}}, protocol_mode="exact")  # type: ignore[call-arg]


def test_the_translation_table_is_total_in_both_directions():
    assert set(PROTOCOL_MODES) == set(STRING_MODES)
    assert (
        PROTOCOL_MODES["exact"] == "exact" and PROTOCOL_MODES["prefix"] == "first_token"
    )
    from causalab.protocol.schema import MATCH_MODES  # the protocol's tuple

    assert set(PROTOCOL_MODES.values()) == set(MATCH_MODES)
    assert len(set(PROTOCOL_MODES.values())) == len(PROTOCOL_MODES)


@pytest.mark.parametrize(
    "bad, message",
    [
        ({}, "non-empty mapping"),
        ({"w": {}}, "non-empty {value: \\[forms\\]}"),
        ({"w": {"a": "a"}}, "list of surface forms"),
        ({"w": {"a": ["a", 1]}}, "list\\[str\\]"),
        ({"w": {"a": []}}, "declares no forms"),
        ({"w": {"a": [" "]}}, "whitespace-only"),
        ({"w": {1: ["1"], "1": ["one"]}}, "both spell"),
    ],
)
def test_a_malformed_forms_map_is_refused_at_construction(bad, message):
    with pytest.raises(ScoringError, match=message):
        ScoringSpec(forms=bad)


def test_several_variables_need_an_answer_variable():
    forms = {
        "answer": build_output_tokens(["A", "B"]),
        "position": build_output_tokens([0, 1]),
    }
    with pytest.raises(ScoringError, match="answer_variable is required"):
        ScoringSpec(forms=forms)
    with pytest.raises(ScoringError, match="declares no forms"):
        ScoringSpec(forms=forms, answer_variable="colour")
    spec = ScoringSpec(forms=forms, answer_variable="answer")
    assert spec.answer_variable == "answer"


@pytest.mark.parametrize(
    "field, value",
    [
        ("string_mode", "startswith"),
        ("undeclared_value", "guess"),
        ("invalid_output", "refuse"),
        ("version", 0),
        ("version", True),
        ("version", "1"),
    ],
)
def test_off_vocabulary_fields_are_refused(field, value):
    with pytest.raises(ScoringError):
        _spec(**{field: value})


def test_the_policy_vocabularies_are_closed_and_small():
    assert UNDECLARED_VALUE_POLICIES == ("refuse", "literal")
    assert INVALID_OUTPUT_POLICIES == ("incorrect", "unscored")
    assert SCORING_RESULTS == ("ok", "unrecorded")


# --------------------------------------------------------------------------- #
# immutability and the identity (T12)
# --------------------------------------------------------------------------- #


def test_the_spec_is_frozen():
    spec = _spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.string_mode = "prefix"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.protocol_mode = "first_token"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.forms["weekday"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.forms["weekday"]["Monday"] = ["x"]  # type: ignore[index]


def test_reassigning_a_causal_models_scoring_views_is_refused():
    """T12 — the live defect: ``output_tokens`` and
    ``match_modes`` were plain attributes anyone could reassign after
    validation. They are properties without setters now."""
    model = _model(_spec())
    for attr in ("output_tokens", "match_modes", "scoring"):
        with pytest.raises(AttributeError):
            setattr(model, attr, {"weekday": {}})
    # and editing the view changes nothing: it is a fresh copy every read
    view = model.output_tokens
    assert view is not None
    view["weekday"]["Monday"].append("lundi")
    assert model.output_tokens == {"weekday": build_output_tokens(WEEKDAYS)}


def test_a_different_definition_of_correct_is_a_different_identity():
    """T12: every field is in ``identity()``, so changing any one of them is
    a different definition of correct — and ``identity()`` is a function of
    the declaration alone (two constructions agree; a tuple-keyed spelling of
    the same values collapses to the same form groups)."""
    assert _spec().identity() == _spec().identity()
    base = _spec()
    changed = {
        "forms": ScoringSpec(
            forms={"weekday": build_output_tokens(WEEKDAYS + ["Thursday"])}
        ),
        "form spelling": ScoringSpec(forms={"weekday": {w: [w] for w in WEEKDAYS}}),
        "string_mode": _spec(string_mode="prefix"),
        "undeclared_value": _spec(undeclared_value="literal"),
        "invalid_output": _spec(invalid_output="unscored"),
        "version": _spec(version=2),
        "answer_variable": ScoringSpec(
            forms={
                "weekday": build_output_tokens(WEEKDAYS),
                "day": build_output_tokens([1]),
            },
            answer_variable="weekday",
        ),
    }
    identities = [spec.identity() for spec in changed.values()]
    assert base.identity() not in identities
    assert all(a != b for i, a in enumerate(identities) for b in identities[i + 1 :])
    grouped = ScoringSpec(
        forms={"result": {(e, g): [f" {e}", e] for e in WEEKDAYS for g in range(2)}}
    )
    assert grouped.form_groups() == [[f" {e}", e] for e in WEEKDAYS]


def test_a_causal_model_takes_only_a_spec():
    with pytest.raises(TypeError, match="ScoringSpec"):
        _model({"weekday": build_output_tokens(WEEKDAYS)})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CausalModel({}, {}, output_tokens={})  # type: ignore[call-arg]  # the retired kwarg
    assert _model(None).scoring is None
    assert _model(None).output_tokens is None and _model(None).match_modes is None


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #


def test_exact_grades_the_stripped_string_against_every_form():
    spec = _spec()
    assert spec.grade(" Monday", "Monday") == 1.0
    assert spec.grade("Monday", " Monday") == 1.0  # forms compare stripped
    assert spec.grade("Monday\n", "Monday") == 1.0
    assert spec.grade("Monday is", "Monday") == 0.0  # exact: no continuation
    assert spec.grade(" Tuesday", "Monday") == 0.0
    assert spec.grader()({"string": " Monday"}, "Monday") is True
    assert spec.grader()({"string": " Tuesday"}, "Monday") is False


def test_prefix_credits_a_continuation():
    spec = _spec(string_mode="prefix")
    assert spec.grade("Monday is a day", "Monday") == 1.0
    assert spec.grade("Mon", "Monday") == 0.0  # a prefix *of* the form is not a match
    assert spec.grade("Tuesday", "Monday") == 0.0


def test_expected_resolves_by_identity_spelling_or_form():
    """graph_walk's case in miniature: the declared values spell nothing like
    the form the model emits, and the expected string is the form."""
    spec = ScoringSpec(forms={"node": {(0.0,): ["den"], (1.0,): ["lot"]}})
    assert spec.grade("den", (0.0,)) == 1.0  # identity
    assert spec.grade("den", "(0.0,)") == 1.0  # spelling
    assert spec.grade("den", "den") == 1.0  # the form itself
    assert spec.grade("lot", "den") == 0.0
    assert spec.forms_of("den") == ("den",)
    assert spec.forms_of([0.0]) == ("den",)  # a tuple key read back from JSON


def test_a_list_of_expected_values_is_a_list_of_acceptable_answers():
    """graph_walk's ``raw_output`` is every valid next node's concept."""
    spec = ScoringSpec(forms={"raw_output": {c: [c] for c in ("den", "lot", "cup")}})
    assert spec.forms_of(["den", "lot"]) == ("den", "lot")
    assert spec.grade("den", ["den", "lot"]) == 1.0
    assert spec.grade("lot", ["den", "lot"]) == 1.0
    assert spec.grade("cup", ["den", "lot"]) == 0.0
    with pytest.raises(ScoringError, match="names no declared form"):
        spec.forms_of(["den", "moon"])  # one undeclared member spoils the list


def test_an_undeclared_expected_value_refuses_by_default_and_grades_literally_on_request():
    """A disagreement, settled: the serializer always refused an
    undeclared value and the string checker graded it literally. One rule
    now, and the task says which."""
    with pytest.raises(ScoringError, match="names no declared form"):
        _spec().grade("Friday", "Friday")
    literal = _spec(undeclared_value="literal")
    assert literal.grade("Friday", "Friday") == 1.0
    assert literal.grade("Monday", "Friday") == 0.0
    assert literal.forms_of("Friday") == ("Friday",)


def test_an_off_answer_space_generation_is_incorrect_or_unscored():
    incorrect = _spec()
    assert incorrect.grade("banana", "Monday") == 0.0
    unscored = _spec(invalid_output="unscored")
    assert unscored.grade("banana", "Monday") is None  # the model said nothing declared
    assert unscored.grade("Tuesday", "Monday") == 0.0  # it said a declared wrong answer
    assert unscored.grader()({"string": "banana"}, "Monday") is False


def test_an_ambiguous_expected_string_is_refused():
    spec = ScoringSpec(forms={"v": {"a": ["x", "a"], "b": ["x", "b"]}})
    with pytest.raises(ScoringError, match="ambiguous"):
        spec.grade("x", "x")


def test_grading_an_undeclared_variable_is_refused():
    with pytest.raises(ScoringError, match="declares no forms"):
        _spec().grade("Monday", "Monday", variable="month")


def test_form_groups_are_the_probability_paths_groups():
    spec = ScoringSpec(
        forms={
            "result": {
                ("Mon", 0): [" Mon", "Mon"],
                ("Mon", 1): [" Mon", "Mon"],
                ("Tue", 0): [" Tue", "Tue"],
            }
        }
    )
    assert spec.form_groups() == [[" Mon", "Mon"], [" Tue", "Tue"]]


def test_the_grade_record_identity_is_a_real_unit_and_identifier():
    from causalab.protocol.estimand import UNITS, parse_identifier

    assert GRADE_RECORD_IDENTITY["unit"] in UNITS
    assert parse_identifier(GRADE_RECORD_IDENTITY["estimand_version"]) == (
        "string_grade",
        1,
    )


# --------------------------------------------------------------------------- #
# the bespoke full_string_checker
# --------------------------------------------------------------------------- #

CHECKER_SRC = '''\
def checker(neural_output, causal_output):
    """Bespoke: the answer anywhere in the output, case-insensitively."""
    return causal_output.strip().lower() in neural_output["string"].lower()
'''


@pytest.fixture
def checker_module(tmp_path, monkeypatch):
    pkg = tmp_path / "bespoke_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    module = pkg / "grader.py"
    module.write_text(CHECKER_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m.startswith("bespoke_pkg")]:
        del sys.modules[name]
    import importlib

    importlib.invalidate_caches()
    yield module
    for name in [m for m in sys.modules if m.startswith("bespoke_pkg")]:
        del sys.modules[name]


def test_a_full_string_checker_grades(checker_module):
    spec = _spec(full_string_checker="bespoke_pkg.grader.checker")
    assert (
        spec.grade("I think it is MONDAY.", "Monday") == 1.0
    )  # the checker's semantics
    assert spec.grade("Tuesday", "Monday") == 0.0
    # the checker is inside the identity
    assert spec.identity() != _spec().identity()
    assert spec.identity()["full_string_checker"] == "bespoke_pkg.grader.checker"


def test_a_locator_naming_no_function_or_no_module_is_refused(checker_module):
    with pytest.raises(ScoringError, match="defines no top-level function"):
        _spec(full_string_checker="bespoke_pkg.grader.grade")
    with pytest.raises(ScoringError, match="does not resolve|not a Python source"):
        _spec(full_string_checker="bespoke_pkg.nowhere.checker")
    with pytest.raises(ScoringError, match="dotted"):
        _spec(full_string_checker="checker")


def test_a_broken_import_inside_the_checker_propagates(checker_module):
    """An absent checker and a broken one must stay distinguishable: the spec
    resolves the module *without importing it*, so the break surfaces where
    the grader runs, as the ImportError it is."""
    checker_module.write_text("import definitely_not_a_real_module_xyz\n" + CHECKER_SRC)
    spec = _spec(
        full_string_checker="bespoke_pkg.grader.checker"
    )  # parses; does not import
    with pytest.raises(ModuleNotFoundError, match="definitely_not_a_real_module_xyz"):
        spec.grade("Monday", "Monday")


# --------------------------------------------------------------------------- #
# check_scoring — the table's string_mode against the document's modes
# --------------------------------------------------------------------------- #


def _rows(spec: ScoringSpec | None, n: int = 3) -> list[dict]:
    rows = [{"input": f"row {i}", "label": " Monday"} for i in range(n)]
    if spec is not None:
        for row in rows:
            row[STRING_MODE_COLUMN] = spec.string_mode
    return rows


def test_an_unrecorded_table_compares_nothing():
    check = check_scoring(_rows(None), {"iia": "exact"}, where="weekdays/data")
    assert check.result == "unrecorded" and check.string_mode is None
    assert check.as_record() == {"string_mode": None, "result": "unrecorded"}
    assert table_scoring([]) is None


def test_a_recorded_table_under_the_derived_mode_is_ok():
    exact, prefix = _spec(), _spec(string_mode="prefix")
    ok = check_scoring(_rows(exact), {"iia": "exact"}, where="t")
    assert ok.result == "ok" and ok.string_mode == "exact"
    assert ok.as_record() == {"string_mode": "exact", "result": "ok"}
    ok = check_scoring(_rows(prefix), {"iia": "first_token"}, where="t")
    assert ok.result == "ok" and ok.string_mode == "prefix"
    # no match metric at all: nothing to contradict, still recorded
    assert check_scoring(_rows(prefix), {}, where="t").result == "ok"


def test_an_exact_table_under_first_token_is_not_a_contradiction():
    """``first_token`` is a strict generalization of ``exact`` on single-token
    answers; the over-crediting half (multi-token forms sharing a first
    token) is ``metrics._refuse_indistinct_first_tokens``'s, not this one's."""
    assert (
        check_scoring(_rows(_spec()), {"iia": "first_token"}, where="t").result == "ok"
    )


def test_a_prefix_table_under_exact_is_refused_naming_both_modes_and_the_derivation():
    with pytest.raises(ScoringMismatch) as err:
        check_scoring(
            _rows(_spec(string_mode="prefix")), {"iia": "exact"}, where="sor/data"
        )
    message = str(err.value)
    assert "'exact'" in message and "'prefix'" in message and "'first_token'" in message
    assert "prefix → first_token" in message and "'sor/data'" in message
    assert err.value.metric == "iia"


def test_a_malformed_string_mode_column_is_refused():
    rows = _rows(_spec())
    rows[1][STRING_MODE_COLUMN] = "prefix"
    with pytest.raises(ScoringError, match="rows disagree"):
        table_scoring(rows)
    half = _rows(_spec())
    del half[1][STRING_MODE_COLUMN]  # one row without the column disagrees too
    with pytest.raises(ScoringError, match="rows disagree"):
        table_scoring(half)
    bad_mode = _rows(_spec())
    for row in bad_mode:
        row[STRING_MODE_COLUMN] = "startswith"
    with pytest.raises(ScoringError, match="not one of"):
        table_scoring(bad_mode)


def test_declared_modes_reads_only_match_metrics():
    from causalab.protocol.schema import AggregationSpec

    metrics = {
        "iia": AggregationSpec(
            kind="match",
            fields={"expected": "label", "mode": "first_token"},
        ),
        "ld": AggregationSpec(kind="logit_diff", fields={"a": "x", "b": "y"}),
        "bare": AggregationSpec(kind="match", fields={"expected": "label"}),
    }
    assert declared_modes(metrics) == {"iia": "first_token"}


def test_the_module_is_stdlib_only():
    """The loader, the serializer and ``validate --data`` all import it, and
    none of them may pay for numerics or for the protocol layer."""
    import ast

    import causalab.causal.scoring as scoring

    tree = ast.parse(Path(scoring.__file__).read_text())
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert "causalab" not in imported and "torch" not in imported, imported
