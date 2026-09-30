"""Property-tier invariants for ``causalab/tasks/loader.py``.

``loader.py`` defines the unified [`Task`][causalab.tasks.loader.Task] dataclass — the handle every
downstream pipeline stage (counterfactual generation, pipeline wiring,
intervention, featurization, analysis) consumes — together with the
convention-based factory [`load_task`][causalab.tasks.loader.load_task] and the sibling module-loaders
[`load_task_counterfactuals`][causalab.tasks.loader.load_task_counterfactuals] / [`load_task_token_positions`][causalab.tasks.loader.load_task_token_positions].
Every baseline runner enters the pipeline through
``causalab/runner/helpers.py``, which calls ``load_task`` and
``load_task_counterfactuals``; if the loader returns a malformed
[`Task`][causalab.tasks.loader.Task], every downstream step breaks at the runner entry point.

Test tiers (see ``docs/TESTS.md``): ``tasks/`` requires
``[smoke-transitive, property-direct, numerical-direct]``. Smoke is
satisfied transitively by every baseline runner. Property and
numerical are direct: ``loader.py`` is a dispatch / wiring module, not a
numerical generator, so numerical-direct is covered by the same property
file (hand-pinned field values for canonical fixtures) — no per-module
JSON golden is needed.

The input space is the finite set of shipped tasks plus a handful of
error cases, so this file uses ``@pytest.mark.parametrize`` rather than
hypothesis.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import pytest

from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringError
from causalab.tasks.loader import (
    Task,
    load_task,
    load_task_counterfactuals,
    load_task_token_positions,
    resolve_task,
)
from tests._helpers.tasks import FACTORY_TASKS, SINGLETON_TASKS

# ---------------------------------------------------------------------------
# Factory-config builders — kept inline (no fixture chain) so each test
# reads as a contract check rather than reaching through a fixture.
# ---------------------------------------------------------------------------


def _weekdays_cfg():
    from causalab.tasks.natural_domains_arithmetic import NaturalDomainConfig

    return NaturalDomainConfig(domain_type="weekdays")


def _ring6_cfg():
    from causalab.tasks.graph_walk.config import GraphWalkConfig

    return GraphWalkConfig(graph_type="ring", graph_size=6)


def _factory_cfg(task_name: str):
    """Minimal config for each factory task, used by parametrised tests."""
    if task_name == "graph_walk":
        return _ring6_cfg()
    if task_name == "natural_domains_arithmetic":
        return _weekdays_cfg()
    if task_name == "identity_naming":
        from causalab.tasks.identity_naming.config import IdentityNamingConfig

        return IdentityNamingConfig(domain_type="pitch_midi")
    if task_name == "subject_object_relations":
        from causalab.tasks.subject_object_relations.config import (
            SubjectObjectRelationsConfig,
        )

        return SubjectObjectRelationsConfig(relation="name_gender")
    raise ValueError(f"no factory cfg builder for task '{task_name}'")


# ---------------------------------------------------------------------------
# Task dataclass invariants
# ---------------------------------------------------------------------------


class TestTaskProperty:
    """Invariants of the [`Task`][causalab.tasks.loader.Task] dataclass itself."""

    pytestmark = pytest.mark.property

    def test_task_is_a_dataclass(self) -> None:
        assert dataclasses.is_dataclass(Task)

    def test_intervention_values_empty_without_variable(self) -> None:
        task = load_task("MCQA")
        # Force the no-intervention branch.
        task.intervention_variable = None
        assert task.intervention_values == []

    def test_is_cyclic_matches_causal_model_periods(self) -> None:
        task = load_task("natural_domains_arithmetic", task_cfg=_weekdays_cfg())
        assert task.is_cyclic == bool(
            task.intervention_variable
            and task.intervention_variable in task.causal_model.periods
        )

    def test_intervention_value_index_round_trip(self) -> None:
        task = load_task("natural_domains_arithmetic", task_cfg=_weekdays_cfg())
        # Build an example whose intervention-variable value is the first
        # entry in the materialised intervention_values list.
        first_value = task.intervention_values[0]
        ex = {"input": {task.intervention_variable: first_value}}
        assert task.intervention_value_index(ex) == 0


# ---------------------------------------------------------------------------
# load_task: factory dispatch
# ---------------------------------------------------------------------------


class TestLoadTaskProperty:
    """Dispatch + field-population invariants for [`load_task`][causalab.tasks.loader.load_task]."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("task_name", SINGLETON_TASKS)
    def test_singleton_returns_task(self, task_name: str) -> None:
        task = load_task(task_name)
        assert isinstance(task, Task)
        assert task.name == task_name
        assert isinstance(task.causal_model, CausalModel)

    @pytest.mark.parametrize("task_name", FACTORY_TASKS)
    def test_factory_returns_task(self, task_name: str) -> None:
        task = load_task(task_name, task_cfg=_factory_cfg(task_name))
        assert isinstance(task, Task)
        assert task.name == task_name
        assert isinstance(task.causal_model, CausalModel)

    @pytest.mark.parametrize("task_name", FACTORY_TASKS)
    def test_factory_requires_task_cfg(self, task_name: str) -> None:
        with pytest.raises(ValueError, match="task_cfg is required"):
            load_task(task_name)

    def test_unknown_task_raises_import_error(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_task("nonexistent_task_xyz")

    def test_random_true_uses_create_random_model(self) -> None:
        # natural_domains_arithmetic exposes CREATE_RANDOM_CAUSAL_MODEL.
        task = load_task(
            "natural_domains_arithmetic", task_cfg=_weekdays_cfg(), random=True
        )
        assert isinstance(task.causal_model, CausalModel)

    def test_random_true_raises_when_unsupported(self) -> None:
        with pytest.raises(ValueError, match="random"):
            load_task("graph_walk", task_cfg=_ring6_cfg(), random=True)

    def test_mcqa_singleton_pins_template_and_target(self) -> None:
        """Numerical pin: MCQA's hand-checked field values."""
        from causalab.tasks.MCQA.causal_models import TEMPLATES

        task = load_task("MCQA")
        assert task.intervention_variable == "answer_position"
        assert task.template == TEMPLATES[0]
        assert callable(task.checker)

    def test_graph_walk_factory_pins_intervention_variable(self) -> None:
        """Numerical pin: graph_walk's hand-checked field values."""
        task = load_task("graph_walk", task_cfg=_ring6_cfg())
        assert task.intervention_variable == "node_coordinates"
        assert len(task.intervention_values) == 6

    def test_natural_domains_weekdays_pins_periods_and_embeddings(self) -> None:
        """Numerical pin: weekdays config produces a 7-entity cyclic model."""
        task = load_task("natural_domains_arithmetic", task_cfg=_weekdays_cfg())
        assert "entity" in task.causal_model.values
        assert len(task.causal_model.values["entity"]) == 7
        assert task.is_cyclic
        assert task.causal_model.embeddings

    def test_entity_binding_grades_by_prefix(self) -> None:
        """entity_binding's spec declares ``string_mode="prefix"``; the task's
        grader is the spec's, so an answer followed by continuation tokens is
        credited and a different entity is not."""
        task = load_task("entity_binding")
        assert task.checker is not None
        assert task.checker.__module__ == "causalab.causal.scoring"
        assert task.checker({"string": "bread\n\nAnn loves"}, "bread") is True
        assert task.checker({"string": "cheese"}, "bread") is False

    @staticmethod
    def _declared_answer(task: Task) -> tuple[str, str]:
        """``(a declared value's first form, the value)`` of the task's answer
        variable — a pair the grader must credit."""
        spec = task.causal_model.scoring
        assert spec is not None
        value = next(iter(spec.forms[spec.answer_variable]))
        return spec.forms_of(value)[0], value

    @pytest.mark.parametrize("task_name", SINGLETON_TASKS)
    def test_every_singleton_task_ships_a_working_checker(self, task_name: str) -> None:
        """Every task resolves a ``checker(neural_output, causal_output) -> bool``
        — the sole match authority for base/intervention scoring — from the
        ``ScoringSpec`` its causal model declares, and it credits a declared
        answer."""
        task = load_task(task_name)
        form, value = self._declared_answer(task)
        assert callable(task.checker)
        assert task.checker({"string": form}, value) is True

    @pytest.mark.parametrize("task_name", FACTORY_TASKS)
    def test_every_factory_task_ships_a_working_checker(self, task_name: str) -> None:
        task = load_task(task_name, task_cfg=_factory_cfg(task_name))
        form, value = self._declared_answer(task)
        assert callable(task.checker)
        assert task.checker({"string": form}, value) is True

    def test_graph_walk_checker_grades_any_valid_neighbour(self) -> None:
        """graph_walk's ``raw_output`` is the *list* of valid next-node concepts;
        the grader credits any member and nothing else."""
        task = load_task("graph_walk", task_cfg=_ring6_cfg())
        model = task.causal_model
        trace = model.new_trace(
            {"node_coordinates": model.values["node_coordinates"][0], "walk_seed": 0}
        )
        neighbours = trace["raw_output"]
        assert isinstance(neighbours, list) and neighbours
        for concept in neighbours:
            assert task.checker({"string": concept}, neighbours) is True
            assert task.checker({"string": concept}, concept) is True
        assert task.checker({"string": neighbours[0] + "_zzz"}, neighbours) is False

    def test_mcqa_checker_grades_the_letter_and_refuses_an_undeclared_value(
        self,
    ) -> None:
        """MCQA's grader is keyed on ``answer`` (the letter the model emits), not
        on the ``answer_position`` interchange target. The retired
        ``score_by: value`` convention accepted a colour word in place of the
        letter through a literal-match fallback; a colour is an undeclared value
        now, and the spec's default ``undeclared_value: refuse`` says so rather
        than crediting it."""
        checker = load_task("MCQA").checker
        assert checker({"string": "A"}, "A") is True
        assert checker({"string": " A"}, " A") is True
        assert checker({"string": "B"}, "A") is False
        with pytest.raises(ScoringError, match="names no declared form"):
            checker({"string": "orange"}, "orange")

    def test_a_task_without_scoring_cannot_grade(self) -> None:
        """No bespoke module and no strict-equality fallback: a causal model that
        declares no ``ScoringSpec`` is a task that cannot grade, refused at load."""
        from causalab.causal import Dom, V, mechanism
        from causalab.tasks.loader import _grader

        @mechanism
        def equations(x: Dom(["a"])):
            raw_input = V(x)  # noqa: F841
            raw_output = V(x)
            return raw_output

        model = CausalModel(equations)
        with pytest.raises(ValueError, match="cannot grade its output"):
            _grader(model, "unscored_task")


class TestLoadTaskCounterfactualsProperty:
    """Invariants for [`load_task_counterfactuals`][causalab.tasks.loader.load_task_counterfactuals]."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("task_name", SINGLETON_TASKS + FACTORY_TASKS)
    def test_returns_module_exposing_generate_dataset(self, task_name: str) -> None:
        mod = load_task_counterfactuals(task_name)
        assert hasattr(mod, "generate_dataset")

    def test_nonexistent_task_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            load_task_counterfactuals("nonexistent_task_xyz")


# ---------------------------------------------------------------------------
# load_task_token_positions
# ---------------------------------------------------------------------------


class TestLoadTaskTokenPositionsProperty:
    """Invariants for [`load_task_token_positions`][causalab.tasks.loader.load_task_token_positions]."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("task_name", SINGLETON_TASKS + FACTORY_TASKS)
    def test_returns_module_exposing_create_token_positions(
        self, task_name: str
    ) -> None:
        mod = load_task_token_positions(task_name)
        assert hasattr(mod, "create_token_positions")

    def test_nonexistent_task_raises_import_error(self) -> None:
        with pytest.raises(ImportError):
            load_task_token_positions("nonexistent_task_xyz")


# ---------------------------------------------------------------------------
# Session-local task layer
# ---------------------------------------------------------------------------


# A minimal singleton task, written into a fake ``${SESSION_DIR}/code/tasks/<name>/``
# package on a session-style PYTHONPATH. Kept trivial — the assertions exercise
# the *resolution* path, not task semantics.
_FIXTURE_CAUSAL_MODELS = """from causalab.causal.model import CausalModel
from causalab.causal.scoring import build_output_tokens
from causalab.causal.scoring import ScoringSpec
from causalab.causal import Dom, V, mechanism

COLORS = ["red", "green", "blue"]
@mechanism
def equations(color: Dom(COLORS)):
    raw_input = V(f'The color is {color}. The color is', domain=Dom(str))
    raw_output = V(color, domain=Dom(str))
    return raw_output
CAUSAL_MODEL = CausalModel(
    equations,
    id="session_local_fixture",
    scoring=ScoringSpec(forms={"color": build_output_tokens(COLORS)}),
)
TARGET_VARIABLE = "color"
TEMPLATE = "The color is {color}. The color is"
"""

# Same fixture, but exporting the model under the *lowercase* ``causal_model``
# name older task templates used. The loader must
# accept it without the task having to export both casings.
_FIXTURE_CAUSAL_MODELS_LOWERCASE = """from causalab.causal.model import CausalModel
from causalab.causal.scoring import build_output_tokens
from causalab.causal.scoring import ScoringSpec
from causalab.causal import Dom, V, mechanism

COLORS = ["red", "green", "blue"]
@mechanism
def equations(color: Dom(COLORS)):
    raw_input = V(f'The color is {color}. The color is', domain=Dom(str))
    raw_output = V(color, domain=Dom(str))
    return raw_output
causal_model = CausalModel(
    equations,
    id="lowercase_singleton_fixture",
    scoring=ScoringSpec(forms={"color": build_output_tokens(COLORS)}),
)
TARGET_VARIABLE = "color"
TEMPLATE = "The color is {color}. The color is"
"""

# A factory task exporting only the lowercase ``create_causal_model`` — the
# factory counterpart of the casing tolerance. resolve_task's factory
# probe and load_task's dispatch must both recognise it.
_FIXTURE_CAUSAL_MODELS_LOWERCASE_FACTORY = """from causalab.causal.model import CausalModel
from causalab.causal.scoring import build_output_tokens
from causalab.causal.scoring import ScoringSpec
from causalab.causal import Dom, V, mechanism

COLORS = ["red", "green", "blue"]


def create_causal_model(cfg):
    @mechanism
    def equations(color: Dom(COLORS)):
        raw_input = V(f'The color is {color}. The color is', domain=Dom(str))
        raw_output = V(color, domain=Dom(str))
        return raw_output
    return CausalModel(
    equations,
    id="lowercase_factory_fixture",
    scoring=ScoringSpec(forms={"color": build_output_tokens(COLORS)}),
)


TARGET_VARIABLE = "color"
TEMPLATE = "The color is {color}. The color is"
"""

_FIXTURE_COUNTERFACTUALS = """\
def generate_dataset(model, n, seed=42):
    return []
"""

_FIXTURE_TOKEN_POSITIONS = """\
def create_token_positions(pipeline, template=None, templates=None):
    return []
"""

# The default fixture, minus its scoring: a task that cannot grade, refused at load.
_FIXTURE_CAUSAL_MODELS_UNSCORED = _FIXTURE_CAUSAL_MODELS.replace(
    '    scoring=ScoringSpec(forms={"color": build_output_tokens(COLORS)}),\n', ""
)


def _fixture_with_full_string_checker(task_name: str) -> str:
    """The default fixture declaring a bespoke ``full_string_checker`` *inside*
    its spec — the one place a custom matcher may live."""
    return _FIXTURE_CAUSAL_MODELS.replace(
        '    scoring=ScoringSpec(forms={"color": build_output_tokens(COLORS)}),\n',
        "    scoring=ScoringSpec(\n"
        '        forms={"color": build_output_tokens(COLORS)},\n'
        f'        full_string_checker="tasks.{task_name}.checker.checker",\n'
        "    ),\n",
    )


# A bespoke checker: the answer anywhere in the output — semantics the derived
# grader does not have, which is how a test tells the two apart.
_FIXTURE_CHECKER = """\
def checker(neural_output, causal_output):
    return causal_output.strip() in neural_output["string"]
"""

# A checker.py that imports a module that does not exist — a broken import
# *inside* the checker, which must propagate rather than be mistaken for an
# absent checker.
_FIXTURE_CHECKER_BROKEN_IMPORT = """\
import definitely_not_a_real_module_xyz

def checker(neural_output, causal_output):
    return True
"""

# A checker.py that imports cleanly but exports no ``checker`` function.
_FIXTURE_CHECKER_NO_FN = """\
NOT_A_CHECKER = True
"""


def _write_session_local_task(
    code_dir: Path,
    task_name: str,
    *,
    causal_models_src=None,
    include_checker=True,
    checker_src=None,
):
    """Materialise a ``code/tasks/<task_name>/`` package (the layout of a
    session-local task). ``checker_src`` overrides the
    checker.py body (e.g. a broken or function-less checker); the module is
    inert unless the causal model's spec declares it as its
    ``full_string_checker``."""
    pkg = code_dir / "tasks" / task_name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "causal_models.py").write_text(causal_models_src or _FIXTURE_CAUSAL_MODELS)
    (pkg / "counterfactuals.py").write_text(_FIXTURE_COUNTERFACTUALS)
    (pkg / "token_positions.py").write_text(_FIXTURE_TOKEN_POSITIONS)
    if include_checker:
        (pkg / "checker.py").write_text(checker_src or _FIXTURE_CHECKER)
    return pkg


@pytest.fixture
def isolate_tasks_namespace():
    """Drop any ``tasks`` / ``tasks.*`` entries from ``sys.modules`` before and
    after, so a fixture task imported from a tmp dir never leaks across tests."""

    def _purge():
        for name in [m for m in sys.modules if m == "tasks" or m.startswith("tasks.")]:
            del sys.modules[name]

    _purge()
    yield
    _purge()


class TestSessionLocalFallbackProperty:
    """``${SESSION_DIR}/code/tasks/<name>/`` resolves via the session-local
    fallback when ``CAUSALAB_SESSION_CODE`` is set."""

    pytestmark = pytest.mark.property

    FIXTURE_NAME = "session_local_fixture_task"

    def _arrange(self, tmp_path, monkeypatch, *, with_session_code: bool):
        """Write the fixture task, put its ``code/`` on PYTHONPATH, and toggle
        ``CAUSALAB_SESSION_CODE``."""
        code = tmp_path / "code"
        _write_session_local_task(code, self.FIXTURE_NAME)
        monkeypatch.syspath_prepend(str(code))
        if with_session_code:
            monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        else:
            monkeypatch.delenv("CAUSALAB_SESSION_CODE", raising=False)
        importlib.invalidate_caches()  # so find_spec sees the freshly-written package

    def test_all_loaders_resolve_session_local_task(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        self._arrange(tmp_path, monkeypatch, with_session_code=True)

        task = load_task(self.FIXTURE_NAME)
        assert isinstance(task, Task)
        assert task.name == self.FIXTURE_NAME
        assert isinstance(task.causal_model, CausalModel)
        # the grader is the session-local spec's, through the same fallback
        assert callable(task.checker)
        assert task.checker({"string": " red"}, "red") is True

        assert hasattr(load_task_counterfactuals(self.FIXTURE_NAME), "generate_dataset")
        assert hasattr(
            load_task_token_positions(self.FIXTURE_NAME), "create_token_positions"
        )

    def test_resolve_task_probe_resolves_session_local(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """The ``resolve_task`` factory probe imports ``causal_models`` through the
        same fallback, so a session-local task resolves through the runner entry."""
        self._arrange(tmp_path, monkeypatch, with_session_code=True)

        task, _ = resolve_task(self.FIXTURE_NAME, {}, target_variable="color")
        assert isinstance(task, Task)
        assert task.intervention_variable == "color"

    def test_unset_session_code_raises_import_error(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """Without ``CAUSALAB_SESSION_CODE`` the fallback is disabled, even with the
        package on PYTHONPATH — the existing clear ImportError stands."""
        self._arrange(tmp_path, monkeypatch, with_session_code=False)

        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_task(self.FIXTURE_NAME)

    def test_session_local_task_without_scoring_cannot_grade(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A task whose causal model declares no ``ScoringSpec`` has no way to
        grade its output — a load-time authoring error, with or without a
        ``checker.py`` lying beside it (a module the spec does not declare is
        not a grader)."""
        code = tmp_path / "code"
        _write_session_local_task(
            code, "unscored_task", causal_models_src=_FIXTURE_CAUSAL_MODELS_UNSCORED
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()
        with pytest.raises(ValueError, match="cannot grade its output"):
            load_task("unscored_task")

    def test_an_undeclared_checker_module_is_ignored(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """The bespoke override that used to *win* silently is gone: a
        ``checker.py`` the spec does not name has no effect on grading."""
        code = tmp_path / "code"
        _write_session_local_task(
            code,
            "ignored_checker_task",
            checker_src="def checker(neural_output, causal_output):\n    return True\n",
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()
        task = load_task("ignored_checker_task")
        assert task.causal_model.scoring is not None
        assert task.causal_model.scoring.full_string_checker is None
        # the always-True module did not run
        assert task.checker({"string": "blue"}, "red") is False

    def test_a_declared_full_string_checker_grades(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """T14's fixture task: a bespoke matcher declared inside the spec grades
        exactly as it says, and the locator is part of the spec's identity."""
        code = tmp_path / "code"
        pkg = _write_session_local_task(
            code,
            "bespoke_task",
            causal_models_src=_fixture_with_full_string_checker("bespoke_task"),
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()
        task = load_task("bespoke_task")
        spec = task.causal_model.scoring
        assert spec is not None
        assert spec.full_string_checker == "tasks.bespoke_task.checker.checker"
        assert spec.identity()["full_string_checker"] == spec.full_string_checker
        assert (pkg / "checker.py").is_file()  # the locator resolved to the fixture
        # the checker's own semantics, not the derived grader's
        assert task.checker({"string": "I think it is red today"}, "red") is True
        assert task.checker({"string": "blue"}, "red") is False

    def test_broken_import_inside_checker_propagates(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A declared checker whose module has a broken *internal* import
        surfaces that error untouched when the grader runs — it is NOT masked
        as an absent or malformed checker. The spec resolves the module without
        importing it, so the task still loads."""
        code = tmp_path / "code"
        _write_session_local_task(
            code,
            "broken_checker_task",
            causal_models_src=_fixture_with_full_string_checker("broken_checker_task"),
            checker_src=_FIXTURE_CHECKER_BROKEN_IMPORT,
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()
        task = load_task("broken_checker_task")
        with pytest.raises(
            ModuleNotFoundError, match="definitely_not_a_real_module_xyz"
        ):
            task.checker({"string": "red"}, "red")

    def test_checker_module_without_checker_fn_raises(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A declared locator naming no top-level function is an authoring
        error, refused when the spec is constructed — at import of the task's
        ``causal_models``."""
        code = tmp_path / "code"
        _write_session_local_task(
            code,
            "no_checker_fn_task",
            causal_models_src=_fixture_with_full_string_checker("no_checker_fn_task"),
            checker_src=_FIXTURE_CHECKER_NO_FN,
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()
        with pytest.raises(ScoringError, match="defines no top-level function"):
            load_task("no_checker_fn_task")

    def test_shipped_task_not_shadowed_by_session_local(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A session-local package named like a shipped task never shadows it: the
        shipped namespace resolves first, so the booby-trapped module is never run."""
        code = tmp_path / "code"
        _write_session_local_task(
            code,
            "MCQA",
            causal_models_src='raise RuntimeError("session-local MCQA must never be imported")\n',
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()

        task = load_task("MCQA")
        from causalab.tasks.MCQA.causal_models import CAUSAL_MODEL as shipped

        assert task.name == "MCQA"
        assert task.causal_model is shipped


# ---------------------------------------------------------------------------
# Causal-model export casing tolerance
# ---------------------------------------------------------------------------


class TestModelExportCasingProperty:
    """The loader reads a causal-model export under its canonical UPPER_SNAKE
    name *or* the lowercase alias older task templates used, so a task
    written from such a template loads without having to export both casings.
    Exercised through the session-local layer."""

    pytestmark = pytest.mark.property

    @staticmethod
    def _arrange(tmp_path, monkeypatch, task_name: str, causal_models_src: str) -> None:
        code = tmp_path / "code"
        _write_session_local_task(code, task_name, causal_models_src=causal_models_src)
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()  # so find_spec sees the freshly-written package

    def test_singleton_accepts_lowercase_causal_model(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A task exporting only the lowercase ``causal_model`` singleton loads."""
        self._arrange(
            tmp_path,
            monkeypatch,
            "lowercase_singleton_task",
            _FIXTURE_CAUSAL_MODELS_LOWERCASE,
        )

        task = load_task("lowercase_singleton_task")
        assert isinstance(task, Task)
        assert isinstance(task.causal_model, CausalModel)
        assert task.causal_model.id == "lowercase_singleton_fixture"

    def test_factory_accepts_lowercase_create_causal_model(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A task exporting only the lowercase ``create_causal_model`` factory loads
        through both chokepoints: ``load_task``'s dispatch *and* the runner's
        ``resolve_task`` factory probe (which must feed it the raw config dict)."""
        self._arrange(
            tmp_path,
            monkeypatch,
            "lowercase_factory_task",
            _FIXTURE_CAUSAL_MODELS_LOWERCASE_FACTORY,
        )

        task = load_task("lowercase_factory_task", task_cfg={})
        assert isinstance(task, Task)
        assert task.causal_model.id == "lowercase_factory_fixture"

        # Without case-tolerant detection, resolve_task would not recognise the
        # lowercase factory, call load_task with task_cfg=None, and the factory
        # task would raise "task_cfg is required".
        task2, task_cfg_raw = resolve_task(
            "lowercase_factory_task", {}, target_variable="color"
        )
        assert isinstance(task2, Task)
        assert task_cfg_raw is not None  # detected as a factory → raw cfg passed
        assert task2.intervention_variable == "color"


# ---------------------------------------------------------------------------
# Loader-convention discoverability
# ---------------------------------------------------------------------------


def _tasks_with_causal_models() -> list[str]:
    """Enumerate every ``causalab/tasks/<task>/`` subdir with ``causal_models.py``."""
    tasks_root = Path(importlib.import_module("causalab.tasks").__file__).parent
    names = []
    for child in sorted(tasks_root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "causal_models.py").is_file():
            names.append(child.name)
    return names


class TestLoaderConventionProperty:
    """Discoverability: every task with ``causal_models.py`` honours the contract."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("task_name", _tasks_with_causal_models())
    def test_exports_causal_model_xor_create_causal_model(self, task_name: str) -> None:
        mod = importlib.import_module(f"causalab.tasks.{task_name}.causal_models")
        has_singleton = hasattr(mod, "CAUSAL_MODEL")
        has_factory = hasattr(mod, "CREATE_CAUSAL_MODEL")
        assert has_singleton ^ has_factory, (
            f"Task '{task_name}' must export exactly one of "
            f"CAUSAL_MODEL or CREATE_CAUSAL_MODEL "
            f"(singleton={has_singleton}, factory={has_factory})."
        )
