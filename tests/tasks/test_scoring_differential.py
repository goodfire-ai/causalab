"""T11 — the differential: two representations of the same task's correct
answer cannot disagree.

Over every shipped task under ``causalab/tasks/``, for a deterministic sample
of examples and **both roles** (the base trace and the counterfactual trace),
three representations of "is this string the right answer" are computed and
held equal on every candidate string:

* **the string grader** — ``Task.checker``, the spec's
  [`grader`][causalab.causal.scoring.ScoringSpec.grader];
* **the probability path's form group** — the spec's
  [`forms_of`][causalab.causal.scoring.ScoringSpec.forms_of] the answer value, the
  group a ``match`` metric's ids are resolved from, compared under the
  spec's ``string_mode``;
* **the serialized label columns** — the ``base_answer_forms`` /
  ``cf_answer_forms`` / ``label_forms`` a table carries, compared under the
  ``string_mode`` the *table* records.

The candidates are the example's own answer, the other side's answer, every
declared form of both, and a continuation of the answer (``answer + " and
then"``) — the string a ``prefix`` task credits and an ``exact`` task does
not, which is what makes a flipped mode visible.

Before one spec held them, the three came from three sources and disagreed on
two shipped tasks. MCQA's checker was keyed on ``answer_position`` (forms ``" 0"`` /
``" 1"``) and graded the letter only through a literal-match fallback, while
the table's forms for the same example *were* the digits; entity_binding
declared entity names under ``positional_answer``, whose values are group
indices, so the serializer refused every row the checker graded; graph_walk's
``raw_output`` is a *list* of valid next nodes, which the checker compared as
its ``str()`` and the table keyed by the current node's own concept. All
three are one declaration now, and this file is what keeps them one.

*Mutation*: flip one task's ``string_mode`` after its table was
built and the differential fails for that task and no other —
``test_a_flipped_string_mode_fails_the_differential_for_that_task_only``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from causalab.causal.model import CausalModel
from causalab.causal.scoring import (
    PROTOCOL_MODES,
    STRING_MODE_COLUMN,
    STRING_MODES,
    ScoringSpec,
    check_scoring,
)
from causalab.tasks.loader import Task, load_task, load_task_counterfactuals
from causalab.tasks.serialize import serialize_examples
from tests._helpers.tasks import ALL_TASKS

pytestmark = pytest.mark.property

N_EXAMPLES = 6
SEED = 0


def _config(task_name: str) -> Any:
    """Minimal config for each factory task — the same ones ``test_loader``
    uses, kept inline so the differential reads as a contract check."""
    if task_name == "graph_walk":
        from causalab.tasks.graph_walk.config import GraphWalkConfig

        return GraphWalkConfig(graph_type="ring", graph_size=6)
    if task_name == "natural_domains_arithmetic":
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

        return NaturalDomainConfig(domain_type="weekdays")
    if task_name == "identity_naming":
        from causalab.tasks.identity_naming.config import IdentityNamingConfig

        return IdentityNamingConfig(domain_type="pitch_midi")
    if task_name == "subject_object_relations":
        from causalab.tasks.subject_object_relations.config import (
            SubjectObjectRelationsConfig,
        )

        return SubjectObjectRelationsConfig(relation="name_gender")
    return None


def _examples(task: Task) -> list[dict[str, Any]]:
    generators = load_task_counterfactuals(task.name)
    generate = getattr(generators, "generate_dataset", None) or getattr(
        generators, "generate_graph_walk_dataset"
    )
    return generate(task.causal_model, N_EXAMPLES, SEED)


def _hit(candidate: str, forms: list[str] | tuple[str, ...], mode: str) -> bool:
    """The comparison the spec makes, restated here so the table path is
    graded by the *table's* recorded mode, not the spec's."""
    actual = candidate.strip()
    stripped = [f.strip() for f in forms if f.strip()]
    if mode == "prefix":
        return any(actual.startswith(f) for f in stripped)
    return any(actual == f for f in stripped)


def _candidates(answers: list[Any], spec: ScoringSpec) -> list[str]:
    out: list[str] = []
    for answer in answers:
        strings = answer if isinstance(answer, list) else [answer]
        for value in strings:
            out.append(str(value))
            out.extend(spec.forms_of(value))
            out.append(str(value).strip() + " and then")
    return list(dict.fromkeys(out))


def _differential(
    task: Task, rows: list[dict[str, Any]], examples: list[dict[str, Any]]
) -> list[str]:
    """Every disagreement between the three representations, as messages.
    Empty means "cannot disagree" held on this sample."""
    spec = task.causal_model.scoring
    assert spec is not None
    table_mode = rows[0][STRING_MODE_COLUMN]
    problems: list[str] = []
    for i, (example, row) in enumerate(zip(examples, rows)):
        sides = {
            "base": (example["input"], row["base_answer"], row["base_answer_forms"]),
            "counterfactual": (
                example["counterfactual_inputs"][0],
                row["cf_answer"],
                row["cf_answer_forms"],
            ),
        }
        answers = [trace["raw_output"] for trace, _answer, _forms in sides.values()]
        for role, (trace, answer, table_forms) in sides.items():
            assert answer == trace["raw_output"]
            group = spec.forms_of(trace[spec.answer_variable])
            for candidate in _candidates(answers, spec):
                grader = task.checker({"string": candidate}, answer)
                probability = _hit(candidate, group, spec.string_mode)
                table = _hit(candidate, table_forms, table_mode)
                if not grader == probability == table:
                    problems.append(
                        f"{task.name} example {i} {role} candidate {candidate!r} vs "
                        f"answer {answer!r}: grader={grader} form_group={probability} "
                        f"table={table}"
                    )
        # the label — what an IIA metric scores against — through the same three
        label_group = spec.forms_of(example["setting"][spec.answer_variable])
        for candidate in _candidates([row["label"]], spec):
            grader = task.checker({"string": candidate}, row["label"])
            probability = _hit(candidate, label_group, spec.string_mode)
            table = _hit(candidate, row["label_forms"], table_mode)
            if not grader == probability == table:
                problems.append(
                    f"{task.name} example {i} label candidate {candidate!r} vs "
                    f"{row['label']!r}: grader={grader} form_group={probability} "
                    f"table={table}"
                )
    return problems


def _build(task: Task) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The serialized rows and the *labelled* examples (with the
    post-intervention ``setting`` the ``label`` column is computed from)."""
    examples = _examples(task)
    targets = [task.intervention_variable]
    dataset = serialize_examples(
        task.causal_model, examples, split="all", target_variables=targets
    )
    labeled = task.causal_model.label_counterfactual_data(list(examples), targets)
    return dataset.rows, labeled


@pytest.fixture(params=ALL_TASKS, scope="module")
def loaded(request) -> tuple[Task, list[dict[str, Any]], list[dict[str, Any]]]:
    task = load_task(request.param, task_cfg=_config(request.param))
    rows, examples = _build(task)
    return task, rows, examples


def test_every_shipped_task_declares_one_spec(loaded):
    task, rows, _examples_ = loaded
    spec = task.causal_model.scoring
    assert isinstance(spec, ScoringSpec)
    assert spec.string_mode in STRING_MODES
    assert spec.protocol_mode == PROTOCOL_MODES[spec.string_mode]
    assert task.checker.__module__ == "causalab.causal.scoring"  # the spec's grader
    # and every row of a table built from it records the spec's mode, constantly
    assert {row[STRING_MODE_COLUMN] for row in rows} == {spec.string_mode}
    assert len(rows) == N_EXAMPLES


def test_the_three_representations_agree_on_every_candidate(loaded):
    """The centre of this file: string grader, probability-path form group and
    serialized label columns, on both roles and the label, for every shipped
    task."""
    task, rows, examples = loaded
    problems = _differential(task, rows, examples)
    assert not problems, "\n".join(problems)


def test_the_table_agrees_with_its_own_derived_mode(loaded):
    """A ``match`` metric over this table declaring the spec's
    ``protocol_mode`` is ``ok``; under the other mode a ``prefix`` table is
    refused — the same check ``validate --data`` and the executor make."""
    task, rows, _examples_ = loaded
    spec = task.causal_model.scoring
    assert spec is not None
    ok = check_scoring(rows, {"iia": spec.protocol_mode}, where=task.name)
    assert ok.result == "ok" and ok.string_mode == spec.string_mode
    if spec.string_mode == "prefix":
        with pytest.raises(Exception, match="records string_mode 'prefix'"):
            check_scoring(rows, {"iia": "exact"}, where=task.name)


def _flipped(task: Task) -> Task:
    """The same task with its spec's ``string_mode`` flipped — a *new* spec,
    because a spec cannot be edited in place."""
    spec = task.causal_model.scoring
    assert spec is not None
    other = "exact" if spec.string_mode == "prefix" else "prefix"
    flipped = ScoringSpec(
        forms={var: dict(m) for var, m in spec.forms.items()},
        answer_variable=spec.answer_variable,
        string_mode=other,
        undeclared_value=spec.undeclared_value,
        invalid_output=spec.invalid_output,
        version=spec.version,
    )
    assert flipped.identity() != spec.identity()
    model = task.causal_model
    mutant = CausalModel(
        model.definition,
        id=model.id,
        embeddings=model.embeddings,
        periods=model.periods,
        scoring=flipped,
        input_filter=model.input_filter,
    )
    return dataclasses.replace(task, causal_model=mutant, checker=flipped.grader())


@pytest.mark.parametrize("mutated", ALL_TASKS)
def test_a_flipped_string_mode_fails_the_differential_for_that_task_only(mutated):
    """The mutation. The table was built under the shipped spec; the
    grader and the form group now run under the flipped one, so the
    continuation candidate is credited by one side and not the other — for
    the mutated task, and for no other task."""
    outcomes: dict[str, bool] = {}
    for name in ALL_TASKS:
        task = load_task(name, task_cfg=_config(name))
        rows, examples = _build(task)  # the table the *shipped* spec built
        if name == mutated:
            task = _flipped(task)
        outcomes[name] = bool(_differential(task, rows, examples))
    assert outcomes == {name: name == mutated for name in ALL_TASKS}, outcomes
    # and the string_mode the table records is no longer the spec's
    task = _flipped(load_task(mutated, task_cfg=_config(mutated)))
    rows, _examples_ = _build(load_task(mutated, task_cfg=_config(mutated)))
    check = check_scoring(rows, {}, where=mutated)
    spec = task.causal_model.scoring
    assert spec is not None and check.string_mode != spec.string_mode
