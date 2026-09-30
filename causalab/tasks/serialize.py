"""Task packages → serialized dataset tables (spec §2.2).

An intervention specification names a dataset by ref and the resolver reads bytes
([`causalab.io.env.FileDatasets`][]). This module is the other
half: it turns a task's causal model and counterfactual generator into those
bytes, *ahead of* the load. Nothing here runs at load or run time — which is
the point. Resolution stays stdlib-only, and a document's digest never
depends on importing task code or a tokenizer.

The one principle the row vocabulary encodes: **anything per-row or
task-semantic is computed here and serialized as a column** — the rendered
prompts, the answers, the post-intervention label, the equivalent answer
forms, the values that place a position per row. Documents reference
columns; they never compute.

Columns written per example (§2.2 names, the same ones the committed
fixtures use):

``input``
    The base prompt, the causal model's ``raw_input``.
``counterfactual_inputs``
    A one-element list — the counterfactual prompt. Documents select it as
    ``counterfactual_inputs[0]``.
``base_answer`` / ``cf_answer``
    Each prompt's own answer (``raw_output``).
``label``
    The answer *after* the interchange, from
    [`CausalModel.label_counterfactual_data`][] — what an IIA metric
    scores against. It equals ``cf_answer`` only when the intervention
    replaces every variable the answer depends on, so the two are separate
    columns on purpose.
``<answer>_forms``
    The equivalent surface forms of each answer above, from the causal
    model's ``ScoringSpec`` (its ``forms`` for the ``answer_variable``) — the
    group a ``match`` metric consumes (§2.10).
``string_mode``
    The task's ``string_mode``, constant across the table
    ([`causalab.causal.scoring`][]): whether a generated string must equal a
    form or merely start with one. Written into the rows so a document's
    ``match`` ``mode`` can be held to it at load and before the first forward
    (``check_scoring``). A table without the column is *unrecorded*: it loads
    and runs as it always did.
``edit_groups``
    **Only when the example declares it** — which spans of the pair move
    together ([`causalab.causal.pair_validation`][]): a list of groups, each a name,
    an ``atomic`` flag and per-side ``[start, end]`` char spans into ``input``
    and ``counterfactual_inputs[0]``, one constituent per span pair. A
    generator attaches it to the example as an ``edit_groups`` key; the row
    carries it verbatim after its shape is checked against the two texts.
    Every shipped generator declares none, so no shipped or fixture table
    gains the column and a row without it is unrecorded — nothing is held to
    it. An ``atomic`` group is refused when a run addresses one of its
    constituents without the others (rule 27, ``executor/base.py``).
``split``
    Which split this row belongs to (§2.2). A dataset is **one table** and the
    split is a property of the row, not of the file: a document selects one
    with the ``<ref>#<split>`` fragment, so disjointness is a fact about the
    bytes rather than a claim about how two files were built. Required — see
    `causalab.tables.SPLIT_COLUMN`.
``<variable>``
    Cached variables, the selected scoring answer, and explicitly requested
    extra variables of the *base* trace, stringified: the per-row values that
    ``{"variable": …}`` and ``{"column": …}`` positions resolve (§2.3).
``counterfactual_inputs_variables``
    The same variables for the counterfactual side, in the per-role
    ``<field>_variables`` convention position resolution reads.

Nothing is written beside the table. The table itself is the content-addressed
unit (§7): a document's canonical form carries its content digest, so a
rebuilt table moves the identity of every step that reads it. The parameters a
table was built from are the builder's command line, which the table's README
or the workflow's description records; there is no sidecar, no recipe file and
no rebuild guard, so a table is exactly the bytes a document names.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.causal.model import CausalModel
from causalab.causal.pair_validation import (
    EDIT_GROUPS_COLUMN,
    EditGroupError,
    parse_edit_groups,
)
from causalab.causal.scoring import STRING_MODE_COLUMN, ScoringError
from causalab.protocol.results import EXAMPLE_ID_COLUMN, example_id_defect
from causalab.tables import SPLIT_COLUMN, table_bytes
from causalab.tasks.loader import load_task, load_task_counterfactuals

__all__ = [
    "RESERVED_COLUMNS",
    "SPLIT_COLUMN",
    "SerializedDataset",
    "config_class",
    "serialize_counterfactual_dataset",
    "serialize_examples",
    "table_bytes",
    "write_dataset_table",
]

#: Column names the row vocabulary owns. A causal-model variable colliding
#: with one of these is refused rather than silently overwriting it.
RESERVED_COLUMNS: frozenset[str] = frozenset(
    {
        "input",
        "counterfactual_inputs",
        "counterfactual_inputs_variables",
        "base_answer",
        "base_answer_forms",
        "cf_answer",
        "cf_answer_forms",
        "label",
        "label_forms",
        SPLIT_COLUMN,
        STRING_MODE_COLUMN,
        EDIT_GROUPS_COLUMN,
        EXAMPLE_ID_COLUMN,
        "pair_id",
        "family",
        "base_id",
        "donor_id",
    }
)

#: Trace variables that are already columns in their own right.
_TEXT_VARIABLES: frozenset[str] = frozenset({"raw_input", "raw_output"})


@dataclasses.dataclass(frozen=True)
class SerializedDataset:
    """A built table plus what built it — what the builder reports."""

    rows: list[dict[str, Any]]
    task: str
    generator: str
    n: int
    #: ``None`` when the examples did not come from a seeded generator — an
    #: honest gap is better than a number nothing reproduces
    #: ([`serialize_examples`][]).
    seed: int | None
    target_variables: tuple[str, ...]
    answer_variable: str | None
    #: The task's ``string_mode`` (``exact`` / ``prefix``, from its
    #: ``ScoringSpec``), which the builder prints beside the digest —
    #: a copy for humans of the ``string_mode`` column the rows carry, so a
    #: document author knows whether the answer space needs ``match``'s
    #: ``first_token`` mode (§2.10).
    match_mode: str | None
    #: ``{split value: row count}`` — the table's split allocation, which the
    #: builder prints so its report states the partition the rows declare.
    split_counts: dict[str, int]


def serialize_counterfactual_dataset(
    task_name: str,
    *,
    n: int,
    seed: int,
    split: str | Sequence[str],
    task_cfg: Any = None,
    target_variables: Sequence[str] | None = None,
    generator: str = "generate_dataset",
    answer_variable: str | None = None,
    record_scoring: bool = True,
    extra_variables: Sequence[str] = (),
) -> SerializedDataset:
    """One task's counterfactual dataset as serializable rows.

    Args:
        task_name: A task package under ``causalab/tasks/``.
        n: Number of counterfactual pairs to generate.
        seed: Generator seed. The shipped generators snapshot and restore the
            global RNG, so the same (task, cfg, n, seed) yields the same rows.
        split: The split each row declares — one value broadcast to the whole
            table, or one per example. Required, with no default: an undivided
            pool is a claim worth stating (``split="all"``), and a table that
            forgot to say is exactly what the column exists to prevent.
        task_cfg: Config object for a factory task; ``None`` for a singleton.
        target_variables: The variables the interchange replaces. Defaults to
            the task's ``TARGET_VARIABLE``, which is what its own analyses use.
        generator: Which generator in the task's ``counterfactuals.py`` to
            call (e.g. ``generate_resample_dataset`` for a noise floor).
        answer_variable: The variable whose declared forms supply the
            answer-form columns. Defaults to the spec's ``answer_variable``;
            a model declaring no scoring means no ``_forms`` columns.
        record_scoring: Whether the rows carry the ``string_mode`` column
            (the module docstring). ``True`` for every new table; ``False``
            reproduces a table built before the column existed, which is how
            a committed unrecorded table stays byte-reproducible from its
            recipe.
        extra_variables: Additional variables to compute and export on both
            sides of each pair. Other uncached lazy equations are omitted.

    Raises:
        ValueError: on a task whose generator produces more than one
            counterfactual per example (no v1 column vocabulary for it), a
            variable colliding with a reserved column name, or an answer
            value the task declares no forms for.
    """
    task = load_task(task_name, task_cfg=task_cfg)
    generators = load_task_counterfactuals(task_name)
    if not hasattr(generators, generator):
        raise ValueError(
            f"task {task_name!r} has no counterfactual generator {generator!r} "
            f"(has {sorted(name for name in dir(generators) if name.startswith('generate'))})"
        )
    targets = list(target_variables or ([task.intervention_variable] or []))
    if not targets or targets == [None]:
        raise ValueError(
            f"task {task_name!r} declares no TARGET_VARIABLE — pass "
            "target_variables explicitly so the label is well defined"
        )
    model: CausalModel = task.causal_model
    examples = getattr(generators, generator)(model, n, seed)
    return serialize_examples(
        model,
        examples,
        split=split,
        target_variables=targets,
        answer_variable=answer_variable,
        task_label=task_name,
        generator=generator,
        n=n,
        seed=seed,
        record_scoring=record_scoring,
        extra_variables=extra_variables,
    )


def serialize_examples(
    model: CausalModel,
    examples: Sequence[Mapping[str, Any]],
    *,
    split: str | Sequence[str],
    target_variables: Sequence[str],
    answer_variable: str | None = None,
    task_label: str = "inline",
    generator: str = "inline",
    n: int | None = None,
    seed: int | None = None,
    record_scoring: bool = True,
    extra_variables: Sequence[str] = (),
) -> SerializedDataset:
    """The same rows, from a causal model and an example list you already have.

    [`serialize_counterfactual_dataset`][] goes through
    [`load_task`][], which needs a task **package**
    inside the library checkout. An author who writes ``models.py`` and
    ``counterfactuals.py`` in their own working directory has no task package,
    and still needs a public path from that hand-authored causal model to a
    serialized table.

    This is that path, and it is the *same* code: the package entry point
    resolves its task and then calls this, so the two cannot drift.

    Args:
        model: The causal model the examples were generated from.
        examples: Counterfactual examples — what a generator returns.
        split: As in [`serialize_counterfactual_dataset`][] — a scalar
            broadcast to every row, or one value per example (what a
            group-disjoint builder passes, having decided the allocation
            per row).
        target_variables: The variables the interchange replaces. Required
            here (there is no package to read a ``TARGET_VARIABLE`` from), and
            what the ``label`` column is computed against.
        answer_variable: As in [`serialize_counterfactual_dataset`][].
        task_label: What the rows record as their task. Provenance only — no
            package of this name has to exist.
        generator: Recorded on the result, like ``n`` and ``seed``. Leave all
            three alone when the examples did not come from a seeded
            generator; ``None`` is more honest than a number nothing can
            reproduce.
        n: Recorded on the result; see ``generator``.
        seed: Recorded on the result; see ``generator``.
        record_scoring: As in [`serialize_counterfactual_dataset`][].
        extra_variables: As in [`serialize_counterfactual_dataset`][].
    """
    targets = list(target_variables)
    if not targets or targets == [None]:
        raise ValueError(
            "target_variables is required: it is what the `label` column — the "
            "answer after the interchange — is computed against"
        )
    from causalab.causal.model_comparison import rederive_trace

    if isinstance(extra_variables, str):
        raise TypeError("extra_variables must be a sequence of variable names")
    extra_variables = tuple(extra_variables)
    resolved_answer = _answer_variable(model, answer_variable)
    forms_of = _forms_lookup(model, answer_variable)
    setting_variables = extra_variables + (
        (resolved_answer,) if resolved_answer is not None else ()
    )

    # A pair can originate from another hypothesis's model. Its cached
    # intermediates must not become inputs to this model's intervention.
    derived = [
        {
            **example,
            "input": rederive_trace(model, example["input"]),
            "counterfactual_inputs": [
                rederive_trace(model, trace)
                for trace in example["counterfactual_inputs"]
            ],
        }
        for example in examples
    ]
    labeled = model.label_counterfactual_data(
        derived, targets, setting_variables=setting_variables
    )
    splits = _row_splits(split, len(labeled))
    spec = model.scoring
    mode = spec.string_mode if record_scoring and spec is not None else None
    rows = [
        _row(example, forms_of, task_label, row_split, mode, extra_variables)
        for example, row_split in zip(labeled, splits)
    ]
    # Preserve authored identities so exported hypothesis artifacts can be
    # joined after sharding or reordering.  Validate the row identity at the
    # serialization boundary rather than silently inventing an ordinal join.
    if any(EXAMPLE_ID_COLUMN in example for example in labeled):
        for row, example in zip(rows, labeled):
            if EXAMPLE_ID_COLUMN not in example:
                raise ValueError(
                    f"{EXAMPLE_ID_COLUMN} must be present on every example when authored"
                )
            row[EXAMPLE_ID_COLUMN] = example[EXAMPLE_ID_COLUMN]
        defect = example_id_defect(rows)
        if defect:
            raise ValueError(f"invalid authored example IDs: {defect}")
    for key in ("pair_id", "family", "base_id", "donor_id"):
        if any(key in example for example in labeled):
            for row, example in zip(rows, labeled):
                value = example.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be nonempty on every example")
                row[key] = value
    counts: dict[str, int] = {}
    for value in splits:
        counts[value] = counts.get(value, 0) + 1
    return SerializedDataset(
        rows=rows,
        task=task_label,
        generator=generator,
        n=len(labeled) if n is None else n,
        seed=seed,
        target_variables=tuple(targets),
        answer_variable=resolved_answer,
        match_mode=spec.string_mode if spec is not None else None,
        split_counts=dict(sorted(counts.items())),
    )


def _row_splits(split: str | Sequence[str], n_rows: int) -> list[str]:
    """One split value per row: a scalar broadcasts, a sequence must match.

    A length mismatch is refused rather than zipped short — silently dropping
    the tail would write a table whose rows are correct and whose partition is
    a lie, which is the failure the column exists to make impossible.
    """
    if isinstance(split, str):
        if not split:
            raise ValueError("split must be a non-empty string")
        return [split] * n_rows
    values = [str(value) for value in split]
    if len(values) != n_rows:
        raise ValueError(
            f"split has {len(values)} values for {n_rows} rows — pass one value "
            "per example, or a single string to broadcast"
        )
    if any(not value for value in values):
        raise ValueError("every split value must be a non-empty string")
    return values


def _answer_variable(model: CausalModel, answer_variable: str | None) -> str | None:
    """The variable whose declared forms supply the answer-form columns: the
    caller's choice, or the spec's own ``answer_variable``."""
    if answer_variable is not None:
        return answer_variable
    return model.scoring.answer_variable if model.scoring is not None else None


def _forms_lookup(model: CausalModel, answer_variable: str | None):
    """``(setting) -> forms | None`` for the answer-form columns.

    The forms come from the causal model's ``ScoringSpec`` — the task's own
    declaration of which surface strings count as one answer (§2.10). Keyed
    by the *answer variable's* value, not by the answer string, because that
    is how the declaration is keyed (``result`` → ``[" Friday", "Friday"]``).
    The same spec the string grader reads, so the two cannot disagree on
    which forms an example's answer has.
    """
    spec = model.scoring
    if spec is None:
        if answer_variable is not None:
            raise ValueError(
                f"the causal model declares no scoring, so there are no forms "
                f"for {answer_variable!r} to serialize"
            )
        return lambda setting: None
    variable = spec.answer_variable if answer_variable is None else answer_variable
    if variable not in spec.forms:
        raise ValueError(
            f"the causal model declares no scoring forms for {variable!r} "
            f"(declared: {sorted(spec.forms)}) — no answer forms to serialize"
        )

    def forms(setting: Any) -> list[str]:
        # the spec's one resolution rule (`ScoringSpec.forms_of`): an
        # undeclared value refuses here as it always did — the answer space
        # and the declaration disagree, which would silently mis-score a
        # match metric — unless the spec declares `undeclared_value: literal`
        try:
            return list(spec.forms_of(setting[variable], variable=variable))
        except ScoringError as err:
            raise ValueError(
                f"the causal model declares no scoring forms for "
                f"{variable}={setting[variable]!r}: {err}"
            ) from err

    return forms


def _row(
    example: Mapping[str, Any],
    forms_of,
    task_name: str,
    split: str,
    string_mode: str | None,
    extra_variables: Sequence[str] = (),
) -> dict[str, Any]:
    base = example["input"]
    counterfactuals = example["counterfactual_inputs"]
    if len(counterfactuals) != 1:
        raise ValueError(
            f"task {task_name!r} produced {len(counterfactuals)} counterfactuals "
            "for one example; v1 tables carry exactly one (the shipped "
            "generators' shape) — the column vocabulary for several is undefined"
        )
    counterfactual = counterfactuals[0]
    setting = example["setting"]
    row: dict[str, Any] = {
        "input": base["raw_input"],
        "counterfactual_inputs": [counterfactual["raw_input"]],
        "base_answer": base["raw_output"],
        "cf_answer": counterfactual["raw_output"],
        "label": example["label"],
        SPLIT_COLUMN: split,
    }
    base_forms = forms_of(base)
    if base_forms is not None:
        row["base_answer_forms"] = base_forms
        row["cf_answer_forms"] = forms_of(counterfactual)
        row["label_forms"] = forms_of(setting)
    if string_mode is not None:
        # the task's string_mode, constant per table — what check_scoring
        # holds a document's match mode to (the module docstring)
        row[STRING_MODE_COLUMN] = string_mode
    if example.get(EDIT_GROUPS_COLUMN) is not None:
        # only a declaring example writes the column (the module docstring):
        # default-on would rebuild every shipped table's bytes
        row[EDIT_GROUPS_COLUMN] = _edit_groups_column(example, row, task_name)
    row.update(_variable_columns(base, task_name, required=extra_variables))
    row["counterfactual_inputs_variables"] = [
        _variables(counterfactual, required=extra_variables)
    ]
    return row


def _edit_groups_column(
    example: Mapping[str, Any], row: Mapping[str, Any], task_name: str
) -> list[dict[str, Any]]:
    """The example's ``edit_groups`` declaration in the column's shape, its
    spans checked against the two prompts the row carries — a build fails loud
    on a malformed declaration rather than writing a table ``validate --data``
    refuses."""
    try:
        groups = parse_edit_groups(
            {**row, EDIT_GROUPS_COLUMN: example[EDIT_GROUPS_COLUMN]}
        )
    except EditGroupError as err:
        raise ValueError(
            f"task {task_name!r} declares malformed {EDIT_GROUPS_COLUMN}: {err}"
        ) from err
    return [group.as_row_value() for group in groups]


def _variables(trace: Any, *, required: Sequence[str] = ()) -> dict[str, str]:
    """A trace's variables as strings — position resolution matches substrings
    of the row's text, so the serialized form is the string form."""
    if hasattr(trace, "snapshot"):
        values = trace.snapshot(required=required)
    else:
        values = dict(trace)
        for name in required:
            values[name] = trace[name]
    return {
        name: str(value)
        for name, value in sorted(values.items())
        if name not in _TEXT_VARIABLES
    }


def _variable_columns(
    trace: Any, task_name: str, *, required: Sequence[str] = ()
) -> dict[str, str]:
    columns = _variables(trace, required=required)
    collisions = sorted(set(columns) & RESERVED_COLUMNS)
    if collisions:
        raise ValueError(
            f"task {task_name!r} has causal-model variables {collisions} that "
            f"collide with the reserved row columns {sorted(RESERVED_COLUMNS)} — "
            "rename the variable or serialize this task by hand"
        )
    return columns


def write_dataset_table(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    """Write a table and return its content digest — the sha256 of exactly
    the bytes [`FileDatasets`][causalab.io.env.FileDatasets] will read
    back. Nothing is written beside it (the module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = table_bytes(rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def config_class(task_name: str) -> type | None:
    """The config dataclass of a factory task, by convention.

    ``causalab/tasks/<name>/config.py`` holds exactly one module-level
    dataclass whose name ends in ``Config`` (``NaturalDomainConfig``,
    ``SubjectObjectRelationsConfig``, …). Singleton tasks have none, and take
    no config. Same convention-over-registry approach as
    [`causalab.tasks.loader`][].
    """
    import importlib

    try:
        module = importlib.import_module(f"causalab.tasks.{task_name}.config")
    except ModuleNotFoundError:
        return None
    found = [
        value
        for name, value in vars(module).items()
        if name.endswith("Config")
        and dataclasses.is_dataclass(value)
        and isinstance(value, type)
        and value.__module__ == module.__name__
    ]
    if len(found) > 1:
        raise ValueError(
            f"task {task_name!r} has several config dataclasses "
            f"({sorted(cls.__name__ for cls in found)}) — pass the config object "
            "directly instead of relying on the convention"
        )
    return found[0] if found else None
