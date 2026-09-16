"""Unified task interface and convention-based loader.

All pipeline steps consume the Task dataclass. Tasks are discovered via
importlib from causalab.tasks.<name>.causal_models and validated against
required exports. See causalab/tasks/README.md for the full spec.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType
from typing import Any, Callable

from causalab.causal.causal_model import CausalModel


@dataclass
class Task:
    """Unified task interface. All pipeline steps consume this.

    Variable properties (periods, embeddings, the ``ScoringSpec``) live on the
    CausalModel. The Task adds experiment-specific concerns: which variable to
    intervene on and the prompt template.
    """

    # --- Required fields ---
    name: str
    causal_model: CausalModel
    # ``checker`` is "did the model output match the expected answer"
    # (``checker({"string": generated}, expected) -> bool``): the task's
    # ``ScoringSpec.grader()`` over its ``answer_variable``, derived from the one
    # declaration on the causal model — never a bespoke module that wins
    # over it. There is no strict-equality fallback — a task whose causal model
    # declares no scoring fails to load.
    checker: Callable[[dict, str], bool]
    intervention_variable: str | None = None

    # --- Experiment-specific ---
    template: str | list[str] | None = None
    validate: Callable | None = None
    _example_to_class_override: Callable | None = field(default=None, repr=False)

    # --- Derived from CausalModel ---

    @property
    def intervention_values(self) -> list:
        """Values of the intervention variable (e.g. 28 (weekday, group) tuples)."""
        if self.intervention_variable:
            return self.causal_model.values.get(self.intervention_variable, [])
        return []

    @property
    def is_cyclic(self) -> bool:
        """Whether the intervention variable is cyclic."""
        return bool(
            self.intervention_variable
            and self.intervention_variable in self.causal_model.periods
        )

    def intervention_value_index(self, ex) -> int:
        """Map an example to its index in intervention_values."""
        if self._example_to_class_override is not None:
            return self._example_to_class_override(ex)
        val = ex["input"][self.intervention_variable]
        if isinstance(val, (list, tuple)):
            val = tuple(val)
        return self._value_to_idx[val]

    @cached_property
    def _value_to_idx(self) -> dict:
        return {
            (tuple(v) if isinstance(v, (list, tuple)) else v): i
            for i, v in enumerate(self.intervention_values)
        }

    def create_token_positions(self, pipeline):
        """Create token positions, passing the task's template automatically."""
        tp_mod = load_task_token_positions(self.name)
        if isinstance(self.template, list):
            return tp_mod.create_token_positions(pipeline, templates=self.template)
        return tp_mod.create_token_positions(pipeline, template=self.template)


# ---------------------------------------------------------------------------
# Convention-based loader
# ---------------------------------------------------------------------------


def _optional(mod: ModuleType, name: str, default=None):
    return getattr(mod, name, default)


# ---------------------------------------------------------------------------
# Task-package resolution (shipped + session-local fallback)
# ---------------------------------------------------------------------------


def _task_package_candidates(task_name: str) -> list[str]:
    """Task-package names to try, shipped first.

    Always includes the shipped ``causalab.tasks.<name>``; appends the
    session-local ``tasks.<name>`` only when ``CAUSALAB_SESSION_CODE`` is set
    (see ``causalab/tasks/README.md`` "Local task packages"). Single
    source of the shipped-first precedence + the session-local gate, so every
    resolver (:func:`_import_task_module`) agrees and can't drift.
    """
    candidates = [f"causalab.tasks.{task_name}"]
    if os.environ.get("CAUSALAB_SESSION_CODE"):
        candidates.append(f"tasks.{task_name}")
    return candidates


def _task_package_exists(pkg: str) -> bool:
    """Whether ``pkg`` is importable, *without executing it*.

    A missing top-level parent (e.g. no session-local ``tasks`` package on
    ``PYTHONPATH``) counts as absent. ``find_spec`` only locates the package, so
    a broken import *inside* it is not triggered here — it surfaces later, at
    ``import_module`` time, instead of being mistaken for a missing package.
    """
    try:
        return importlib.util.find_spec(pkg) is not None
    except ModuleNotFoundError:
        return False


def _import_task_module(task_name: str, submodule: str) -> ModuleType:
    """Import ``<task>.<submodule>`` from the namespace the task lives in.

    Resolves the task *package* first — shipped ``causalab.tasks.<name>`` takes
    precedence; a session-local ``tasks.<name>`` is the fallback when
    ``CAUSALAB_SESSION_CODE`` is set (see ``causalab/tasks/README.md``
    "Local task packages"). Resolution is by
    :func:`_task_package_exists` (``find_spec``, no execution), so the fallback
    fires only when the shipped task genuinely does not exist — a broken import
    *inside* a task module surfaces as its own error at import time rather than
    being masked as "task not found". A session-local task never shadows a
    shipped one; :func:`_task_package_candidates` defines that precedence.

    The fallback decision is made once at the task-*package* level (a task lives
    entirely in one namespace): if a shipped task package exists but its
    ``<submodule>.py`` is missing, ``import_module`` raises a clear
    ``ModuleNotFoundError`` for that submodule — it does not wrongly drop to the
    session-local namespace.
    """
    candidates = _task_package_candidates(task_name)
    for pkg in candidates:
        if _task_package_exists(pkg):
            return importlib.import_module(f"{pkg}.{submodule}")
    raise ModuleNotFoundError(
        f"No task package {task_name!r} found. Tried: {', '.join(candidates)}. "
        f"For session-local tasks, see causalab/tasks/README.md "
        f"'Local task packages'."
    )


# ---------------------------------------------------------------------------
# Causal-model export resolution (case-tolerant)
# ---------------------------------------------------------------------------


def _resolve_model_export(mod: ModuleType, canonical: str):
    """Return a causal-model export, accepting either casing of its name.

    The uppercase ``canonical`` (e.g. ``CAUSAL_MODEL``, ``CREATE_CAUSAL_MODEL``)
    is the canonical export — every shipped task and every other loader-read
    export uses UPPER_SNAKE. Its lowercase alias (``canonical.lower()`` —
    ``causal_model``, ``create_causal_model``) is the name earlier task
    templates historically scaffolded, accepted so a task following the template
    verbatim loads without having to export both names. The canonical
    name wins when both are present. ``None`` if neither is defined.
    """
    val = getattr(mod, canonical, None)
    if val is not None:
        return val
    return getattr(mod, canonical.lower(), None)


def _has_model_export(mod: ModuleType, canonical: str) -> bool:
    """Whether ``mod`` defines a causal-model export under either casing.

    The case-tolerant counterpart of ``hasattr(mod, canonical)`` — used by the
    runner's factory probe (``resolve_task``) so it agrees with :func:`load_task`
    on what counts as a factory/singleton.
    """
    return _resolve_model_export(mod, canonical) is not None


def load_task(
    task_name: str,
    task_cfg: dict | None = None,
    random: bool = False,
) -> Task:
    """Load a task by convention from its causal_models module.

    Args:
        task_name: Module name under causalab.tasks (e.g., "weekdays", "graph_walk")
        task_cfg: Config dict for factory tasks (passed to CREATE_CAUSAL_MODEL)
        random: If True, use RANDOM_CAUSAL_MODEL/RANDOM_VARIABLE_VALUES exports
    """
    mod = _import_task_module(task_name, "causal_models")

    # --- Model: singleton or factory ---
    # Each export is read by its canonical UPPER_SNAKE name or the lowercase
    # alias earlier task templates historically scaffolded, via
    # _resolve_model_export.
    create_factory = _resolve_model_export(mod, "CREATE_CAUSAL_MODEL")
    singleton = _resolve_model_export(mod, "CAUSAL_MODEL")
    is_factory = create_factory is not None
    is_singleton = singleton is not None

    if not is_factory and not is_singleton:
        raise ValueError(
            f"Task '{task_name}' must export either CAUSAL_MODEL (singleton) "
            f"or CREATE_CAUSAL_MODEL (factory) in causal_models.py."
        )

    if is_factory:
        if task_cfg is None:
            raise ValueError(
                f"Task '{task_name}' is a factory task — task_cfg is required."
            )
        causal_model = create_factory(task_cfg)
    else:
        causal_model = singleton

    # --- Random baseline override ---
    if random:
        create_random = _resolve_model_export(mod, "CREATE_RANDOM_CAUSAL_MODEL")
        random_singleton = _resolve_model_export(mod, "RANDOM_CAUSAL_MODEL")
        if create_random is not None and task_cfg is not None:
            causal_model = create_random(task_cfg)
        elif random_singleton is not None:
            causal_model = random_singleton
        else:
            raise ValueError(
                f"Task '{task_name}' does not support random baselines "
                f"(no RANDOM_CAUSAL_MODEL or CREATE_RANDOM_CAUSAL_MODEL export)."
            )

    # --- Ensure CausalModel has embeddings ---
    if not causal_model.embeddings:
        if hasattr(mod, "GET_EMBEDDINGS"):
            causal_model.embeddings = mod.GET_EMBEDDINGS(causal_model)
        elif hasattr(mod, "EMBEDDINGS"):
            causal_model.embeddings = mod.EMBEDDINGS

    # --- Ensure CausalModel has periods ---
    if not causal_model.periods:
        periodic_info = _optional(mod, "PERIODIC_INFO", {})
        if hasattr(mod, "GET_PERIODIC_INFO"):
            periodic_info = mod.GET_PERIODIC_INFO(causal_model) or {}
        if periodic_info:
            causal_model.periods = periodic_info

    intervention_variable = _optional(mod, "TARGET_VARIABLE")
    example_to_class_override = _optional(mod, "EXAMPLE_TO_CLASS")

    return Task(
        name=task_name,
        causal_model=causal_model,
        intervention_variable=intervention_variable,
        template=(
            mod.GET_TEMPLATE(causal_model)
            if hasattr(mod, "GET_TEMPLATE")
            else _optional(mod, "TEMPLATE")
        ),
        validate=_optional(mod, "VALIDATE"),
        _example_to_class_override=example_to_class_override,
        checker=_grader(causal_model, task_name),
    )


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------


def load_task_counterfactuals(task_name: str) -> ModuleType:
    """Import a task's counterfactuals module.

    Returns the module so callers can access generate_dataset(model, n, seed).
    Resolves shipped ``causalab.tasks.<name>`` first, then a session-local
    ``tasks.<name>`` (see :func:`_import_task_module`).
    """
    return _import_task_module(task_name, "counterfactuals")


def load_task_token_positions(task_name: str) -> ModuleType:
    """Import a task's token_positions module.

    Returns the module so callers can access create_token_positions(...).
    Resolves shipped ``causalab.tasks.<name>`` first, then a session-local
    ``tasks.<name>`` (see :func:`_import_task_module`).
    """
    return _import_task_module(task_name, "token_positions")


def _grader(causal_model: CausalModel, task_name: str) -> Callable[[dict, str], bool]:
    """A task's string grader: its causal model's ``ScoringSpec.grader()``.

    The one string-match authority is the spec the causal model
    declares (:class:`causalab.causal.scoring.ScoringSpec`), keyed on the
    spec's ``answer_variable`` — the variable the graded string
    (``raw_output``) is a form of — and not on any ``TARGET_VARIABLE`` or later
    ``resolve_task`` override: the answer the model is graded against is the
    same regardless of which variable a config localizes on. MCQA is why the
    two are distinct: its interchange targets ``answer_position`` while the
    model emits the ``answer`` letter.

    There is no bespoke override beside the spec: a task that needs a custom
    matcher declares it *inside* the spec as ``full_string_checker``, where it
    is digested with everything else. Raises ``ValueError`` when the causal
    model declares no scoring — the task then has no way to grade its output.
    """
    spec = causal_model.scoring
    if spec is None:
        raise ValueError(
            f"Task {task_name!r} cannot grade its output: its causal model declares "
            f"no scoring. Pass scoring=ScoringSpec(forms=...) to the CausalModel "
            f"(see causalab.causal.scoring and causalab.causal.causal_model."
            f"build_output_tokens)."
        )
    return spec.grader()


def resolve_task(
    task_name: str,
    task_config: dict[str, Any],
    target_variable: str | None,
    seed: int | None = None,
) -> tuple[Task, Any]:
    """Build a Task from explicit parameters.

    ``task_config`` contains task-specific fields (domain_type, graph_type, etc.).
    ``target_variable`` sets the intervention variable on the returned task and
    overrides the module's TARGET_VARIABLE export.  It is required — passing
    ``None`` raises ``ValueError`` to prevent silent use of a wrong default.

    The type allows ``None`` because callers commonly pass
    ``cfg.task.get("target_variable")`` from a Hydra config; the runtime
    check converts that into a clear error message at the boundary.
    """
    if target_variable is None:
        raise ValueError(
            "resolve_task() requires an explicit target_variable. "
            "Pass the variable name you want to localize (e.g. 'answer', 'color'). "
            "Omitting it silently uses the module default, which is often wrong."
        )
    task_cfg_raw = None

    if task_name == "graph_walk":
        from causalab.tasks.graph_walk.config import GraphWalkConfig

        # Omit `seed` when None so GraphWalkConfig's dataclass default applies
        # (single source of truth for the default seed).
        gw_kwargs: dict[str, Any] = {
            "graph_type": task_config["graph_type"],
            "graph_size": task_config["graph_size"],
            "graph_size_2": task_config["graph_size_2"],
            "context_length": task_config["context_length"],
            "separator": task_config["separator"],
            "no_backtrack": task_config["no_backtrack"],
        }
        if seed is not None:
            gw_kwargs["seed"] = seed
        task_cfg_raw = GraphWalkConfig(**gw_kwargs)
    elif task_name == "natural_domains_arithmetic":
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

        task_cfg_raw = NaturalDomainConfig(
            domain_type=task_config["domain_type"],
            number_range=task_config.get("number_range", None),
            number_groups=task_config.get("number_groups", None),
            result_entities=task_config.get("result_entities", None),
        )
    elif task_name == "identity_naming":
        from causalab.tasks.identity_naming.config import IdentityNamingConfig

        task_cfg_raw = IdentityNamingConfig(
            domain_type=task_config["domain_type"],
        )

    # Generic factory support: a task that exports CREATE_CAUSAL_MODEL (and no
    # CAUSAL_MODEL singleton) but is not special-cased above receives the raw
    # Hydra task-config dict, so factory tasks introduced outside this function
    # (e.g. session-local research tasks selecting a `query`) load without a
    # bespoke branch here. Singletons and the special-cased factories above are
    # untouched (task_cfg_raw is already set, or stays None for singletons).
    if task_cfg_raw is None:
        # Resolve through the shared loader helper so the factory probe honours
        # the same shipped-first / session-local fallback as load_task itself.
        _cm = _import_task_module(task_name, "causal_models")
        if _has_model_export(_cm, "CREATE_CAUSAL_MODEL") and not _has_model_export(
            _cm, "CAUSAL_MODEL"
        ):
            task_cfg_raw = dict(task_config)

    # load_task accepts dict | None; the *Config dataclasses above are not dicts,
    # but the underlying loaders accept them at runtime (task-specific shims).
    task = load_task(task_name, task_cfg=task_cfg_raw)  # pyright: ignore[reportArgumentType]
    task.intervention_variable = target_variable
    return task, task_cfg_raw
