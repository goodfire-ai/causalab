"""Property-tier invariants for ``causalab/tasks/loader.py``.

``loader.py`` defines the unified :class:`Task` dataclass — the handle every
downstream pipeline stage (counterfactual generation, pipeline wiring,
intervention, featurization, analysis) consumes — together with the
convention-based factory :func:`load_task` and the sibling module-loaders
:func:`load_task_counterfactuals` / :func:`load_task_token_positions`.
Every baseline runner enters the pipeline through
``causalab/runner/helpers.py``, which calls ``load_task`` and
``load_task_counterfactuals``; if the loader returns a malformed
:class:`Task`, every downstream step breaks at the runner entry point.

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
import re
import sys
from pathlib import Path

import pytest

from causalab.causal.causal_model import CausalModel
from causalab.tasks.loader import resolve_task
from causalab.tasks.loader import (
    Task,
    load_task,
    load_task_checker,
    load_task_counterfactuals,
    load_task_token_positions,
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
    """Invariants of the :class:`Task` dataclass itself."""

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
    """Dispatch + field-population invariants for :func:`load_task`."""

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

    def test_unknown_task_points_to_existing_documentation(self) -> None:
        with pytest.raises(ModuleNotFoundError) as exc_info:
            load_task("nonexistent_task_xyz")
        reference = re.search(r"see (\S+\.md) '([^']+)'", str(exc_info.value))
        assert reference is not None, str(exc_info.value)
        doc_path, section = reference.groups()
        readme = Path(__file__).resolve().parents[2] / doc_path
        assert readme.is_file(), (
            f"Task error points at missing documentation: {doc_path}"
        )
        assert f"### {section}" in readme.read_text(encoding="utf-8")

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

    def test_entity_binding_derives_a_prefix_checker(self) -> None:
        """entity_binding declares ``output_tokens`` + ``match_modes: prefix``; the
        loader *derives* the checker from that declaration (no checker.py), and it
        still accepts the multi-token ``startswith`` continuation
        ``compute_base_accuracy`` needs (#167/#296).
        """
        task = load_task("entity_binding")
        assert task.checker is not None
        assert task.checker.__module__ == "causalab.causal.causal_model"
        assert task.checker({"string": "bread\n\nAnn loves"}, "bread") is True
        assert task.checker({"string": "cheese"}, "bread") is False

    @pytest.mark.parametrize("task_name", SINGLETON_TASKS)
    def test_every_singleton_task_ships_a_working_checker(self, task_name: str) -> None:
        """Every task resolves a ``checker(neural_output, causal_output) -> bool``
        — the sole match authority for base/intervention scoring, no
        strict-equality fallback (#167). It is either derived from the task's
        ``output_tokens`` declaration (#296) or loaded from its ``checker.py``;
        load_task raises if neither is present.
        """
        checker = load_task(task_name).checker
        assert callable(checker)
        assert isinstance(checker({"string": "x"}, "x"), bool)

    @pytest.mark.parametrize("task_name", FACTORY_TASKS)
    def test_every_factory_task_ships_a_working_checker(self, task_name: str) -> None:
        checker = load_task(task_name, task_cfg=_factory_cfg(task_name)).checker
        assert callable(checker)
        assert isinstance(checker({"string": "x"}, "x"), bool)

    def test_graph_walk_derived_checker_grades_concept_string(self) -> None:
        """graph_walk's derived checker grades the *concept string* answer (#296).

        Its ``output_tokens`` is keyed by coordinate tuples (which feed the
        probability path), but at grading time ``causal_output`` is a concept
        string, so ``derive_checker`` takes the literal-match fallback. Pinning it
        here — not only in the smoke tier — guards that fallback against a silent
        regression (the same applies to MCQA below).
        """
        task = load_task("graph_walk", task_cfg=_ring6_cfg())
        concept = task.causal_model.values["concepts"][0]
        assert task.checker({"string": concept}, concept) is True
        assert task.checker({"string": concept + "_zzz"}, concept) is False

    def test_mcqa_derived_checker_grades_letter_and_value(self) -> None:
        """MCQA's derived checker (keyed on ``answer_position`` digits) grades the
        emitted *letter* / *value* via the literal-match fallback (#296)."""
        checker = load_task("MCQA").checker
        assert checker({"string": "A"}, "A") is True
        assert checker({"string": "B"}, "A") is False
        assert checker({"string": "orange"}, "orange") is True  # score_by: value

    def test_missing_checker_returns_none(self) -> None:
        """A task shipping no ``checker.py`` yields no *bespoke* checker — the
        loader derives one from ``output_tokens`` instead, so an absent checker
        is no longer a hard error (#291 phase 3). ``faketask`` has no package, so
        the resolver finds no checker module and returns ``None``."""
        from causalab.tasks import loader as loader_mod

        assert loader_mod.load_task_checker("faketask") is None


# ---------------------------------------------------------------------------
# load_task_counterfactuals
# ---------------------------------------------------------------------------


class TestLoadTaskCounterfactualsProperty:
    """Invariants for :func:`load_task_counterfactuals`."""

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
    """Invariants for :func:`load_task_token_positions`."""

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
# Session-local task layer (issue #250)
# ---------------------------------------------------------------------------


# A minimal singleton task, written into a fake ``${SESSION_DIR}/code/tasks/<name>/``
# package on a session-style PYTHONPATH. Kept trivial — the assertions exercise
# the *resolution* path, not task semantics.
_FIXTURE_CAUSAL_MODELS = """\
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

COLORS = ["red", "green", "blue"]
values = {"color": COLORS, "raw_input": None, "raw_output": None}
mechanisms = {
    "color": input_var(COLORS),
    "raw_input": Mechanism(
        parents=["color"], compute=lambda t: f"The color is {t['color']}. The color is"
    ),
    "raw_output": Mechanism(parents=["color"], compute=lambda t: t["color"]),
}
CAUSAL_MODEL = CausalModel(mechanisms, values, id="session_local_fixture")
TARGET_VARIABLE = "color"
TEMPLATE = "The color is {color}. The color is"
"""

# Same fixture, but exporting the model under the *lowercase* ``causal_model``
# name the task-setup template historically scaffolded (#256). The loader must
# accept it without the task having to export both casings.
_FIXTURE_CAUSAL_MODELS_LOWERCASE = """\
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

COLORS = ["red", "green", "blue"]
values = {"color": COLORS, "raw_input": None, "raw_output": None}
mechanisms = {
    "color": input_var(COLORS),
    "raw_input": Mechanism(
        parents=["color"], compute=lambda t: f"The color is {t['color']}. The color is"
    ),
    "raw_output": Mechanism(parents=["color"], compute=lambda t: t["color"]),
}
causal_model = CausalModel(mechanisms, values, id="lowercase_singleton_fixture")
TARGET_VARIABLE = "color"
TEMPLATE = "The color is {color}. The color is"
"""

# A factory task exporting only the lowercase ``create_causal_model`` — the
# factory counterpart of the casing tolerance (#256). resolve_task's factory
# probe and load_task's dispatch must both recognise it.
_FIXTURE_CAUSAL_MODELS_LOWERCASE_FACTORY = """\
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

COLORS = ["red", "green", "blue"]


def create_causal_model(cfg):
    values = {"color": COLORS, "raw_input": None, "raw_output": None}
    mechanisms = {
        "color": input_var(COLORS),
        "raw_input": Mechanism(
            parents=["color"],
            compute=lambda t: f"The color is {t['color']}. The color is",
        ),
        "raw_output": Mechanism(parents=["color"], compute=lambda t: t["color"]),
    }
    return CausalModel(mechanisms, values, id="lowercase_factory_fixture")


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

# Every task must ship a checker (#167) — load_task wires it onto Task.checker
# with no strict-equality fallback, so the fixture needs one to load at all.
_FIXTURE_CHECKER = """\
def checker(neural_output, causal_output):
    return causal_output.strip() in neural_output["string"]
"""

# A checker.py that imports a module that does not exist — a broken import
# *inside* the checker, which must propagate rather than be mistaken for an
# absent checker.py.
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
    """Materialise a ``code/tasks/<task_name>/`` package (mirrors what
    task setup scaffolds session-locally). ``checker_src`` overrides the
    checker.py body (e.g. a broken or function-less checker)."""
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
    fallback when ``CAUSALAB_SESSION_CODE`` is set (issue #250)."""

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
        # load_task wires the checker from the session-local checker.py (the
        # bespoke override) through the same fallback.
        assert callable(task.checker)

        assert hasattr(load_task_counterfactuals(self.FIXTURE_NAME), "generate_dataset")
        assert hasattr(
            load_task_token_positions(self.FIXTURE_NAME), "create_token_positions"
        )
        assert callable(load_task_checker(self.FIXTURE_NAME))

    def test_resolve_task_probe_resolves_session_local(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """The ``resolve_task`` factory probe imports ``causal_models`` through the
        same fallback, so a session-local task resolves through the runner entry."""
        self._arrange(tmp_path, monkeypatch, with_session_code=True)

        task, _ = resolve_task(self.FIXTURE_NAME, {}, target_variable="color")
        assert isinstance(task, Task)
        assert task.intervention_variable == "color"

    @pytest.mark.parametrize("session_code", [None, ""])
    def test_unset_or_empty_session_code_raises_import_error(
        self, tmp_path, monkeypatch, isolate_tasks_namespace, session_code
    ) -> None:
        """An unset or empty gate disables fallback, even with the package on
        PYTHONPATH — the existing clear ImportError stands."""
        self._arrange(tmp_path, monkeypatch, with_session_code=False)
        if session_code is not None:
            monkeypatch.setenv("CAUSALAB_SESSION_CODE", session_code)

        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_task(self.FIXTURE_NAME)

    def test_session_code_does_not_add_import_path(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """The gate alone does not add the session's code directory to sys.path."""
        _write_session_local_task(tmp_path / "code", self.FIXTURE_NAME)
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))

        with pytest.raises(ModuleNotFoundError, match="No task package"):
            load_task(self.FIXTURE_NAME)

    def test_session_local_task_missing_checker_and_output_tokens_raises(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A task that ships neither a ``checker.py`` nor an ``output_tokens``
        declaration for its target variable cannot grade its output, so it is a
        load-time authoring error (#291 phase 3). ``load_task_checker`` itself
        returns ``None`` (no bespoke checker); the error fires in ``load_task``."""
        code = tmp_path / "code"
        _write_session_local_task(code, "no_checker_task", include_checker=False)
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()

        assert load_task_checker("no_checker_task") is None
        with pytest.raises(ValueError, match="cannot grade its output"):
            load_task("no_checker_task")

    def test_session_local_task_derives_checker_from_output_tokens(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A task that ships no ``checker.py`` but declares ``output_tokens`` for
        its target variable loads fine — the checker is derived from the
        declaration (#291 phase 3). The derived checker matches a declared form
        and rejects a non-form."""
        causal_src = _FIXTURE_CAUSAL_MODELS.replace(
            'CAUSAL_MODEL = CausalModel(mechanisms, values, id="session_local_fixture")',
            "from causalab.causal.causal_model import build_output_tokens\n"
            "CAUSAL_MODEL = CausalModel(\n"
            '    mechanisms, values, id="session_local_fixture",\n'
            '    output_tokens={"color": build_output_tokens(COLORS)},\n'
            ")",
        )
        code = tmp_path / "code"
        _write_session_local_task(
            code,
            "derive_checker_task",
            causal_models_src=causal_src,
            include_checker=False,
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()

        assert load_task_checker("derive_checker_task") is None
        task = load_task("derive_checker_task")
        first_color = task.causal_model.values["color"][0]
        assert task.checker({"string": first_color}, first_color) is True
        assert task.checker({"string": "definitely_not_a_color"}, first_color) is False

    def test_broken_import_inside_checker_propagates(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A checker.py that exists but has a broken *internal* import surfaces
        that error untouched — it is NOT masked as 'ships no checker.py'. This is
        the resolve-first guard: an absent checker.py vs a broken one are
        distinct, and only the former is an authoring error."""
        code = tmp_path / "code"
        _write_session_local_task(
            code, "broken_checker_task", checker_src=_FIXTURE_CHECKER_BROKEN_IMPORT
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()

        with pytest.raises(ModuleNotFoundError) as exc_info:
            load_task_checker("broken_checker_task")
        # The broken dependency propagates, not the "ships no checker.py" ValueError.
        assert exc_info.value.name == "definitely_not_a_real_module_xyz"

    def test_checker_module_without_checker_fn_raises(
        self, tmp_path, monkeypatch, isolate_tasks_namespace
    ) -> None:
        """A checker.py that imports cleanly but exports no ``checker`` function
        is a clear authoring error (the ``checker is None`` branch)."""
        code = tmp_path / "code"
        _write_session_local_task(
            code, "no_checker_fn_task", checker_src=_FIXTURE_CHECKER_NO_FN
        )
        monkeypatch.syspath_prepend(str(code))
        monkeypatch.setenv("CAUSALAB_SESSION_CODE", str(tmp_path))
        importlib.invalidate_caches()

        with pytest.raises(ValueError, match="defines no `checker` function"):
            load_task_checker("no_checker_fn_task")

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
# Causal-model export casing tolerance (issue #256)
# ---------------------------------------------------------------------------


class TestModelExportCasingProperty:
    """The loader reads a causal-model export under its canonical UPPER_SNAKE
    name *or* the lowercase alias the task-setup template historically
    scaffolded (#256), so a task following the template verbatim loads without
    having to export both casings. Exercised through the session-local layer —
    the same machinery task setup scaffolds into."""

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
