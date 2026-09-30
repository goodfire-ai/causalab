"""Define and run causal models."""

from __future__ import annotations

import ast
import copy
import dis
import inspect
import logging
import random
import textwrap
import types
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from causalab.causal.counterfactuals import (
    CounterfactualExample,
    label_counterfactual_data,
)
from causalab.causal.domains import iter_combinations
from causalab.causal.scoring import ScoringSpec

logger = logging.getLogger(__name__)


class DefinitionError(ValueError):
    """A model definition cannot be compiled with the stated causal semantics."""


@dataclass
class ModelDefinition:
    """Store a function's source and the values that its equations use."""

    signature: inspect.Signature
    syntax: ast.FunctionDef
    environment: dict
    filename: str
    line: int
    is_submodel: bool = False
    _global_namespace: dict | None = field(default=None, repr=False, compare=False)
    _global_bindings: frozenset = field(
        default_factory=frozenset, repr=False, compare=False
    )
    _closure_cells: dict = field(default_factory=dict, repr=False, compare=False)

    def _current_environment(self):
        """Resolve live Python bindings at the model-construction boundary."""
        environment = dict(self.environment)
        if self._global_namespace is not None:
            for name in self._global_bindings:
                environment.pop(name, None)
                if name in self._global_namespace:
                    environment[name] = self._global_namespace[name]
        for name, cell in self._closure_cells.items():
            try:
                environment[name] = cell.cell_contents
            except ValueError as exc:
                raise DefinitionError(
                    f"Configuration binding {name!r} is empty"
                ) from exc
        return environment

    def __call__(self, *args, **kwargs):
        raise TypeError(
            "A decorated definition is not an ordinary Python function; construct CausalModel(definition)"
        )


def _global_names(code):
    """Globals actually loaded by code, including nested comprehension scopes."""
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    }
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_global_names(constant))
    return names


def _definition(function, environment, is_submodel):
    try:
        lines, line = inspect.getsourcelines(function)
        syntax = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise DefinitionError(
            "Cannot recover the model's Python source. Define it in a .py file or an IPython/Jupyter cell with source caching."
        ) from exc
    functions = [node for node in syntax.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        raise DefinitionError("Expected one named Python function")
    captured = dict(function.__globals__)
    captured.update(environment)
    captured.update(inspect.getclosurevars(function).nonlocals)
    global_bindings = _global_names(function.__code__)
    referenced = global_bindings | set(function.__code__.co_freevars)
    # Future/string annotations can refer to a factory local used only in the
    # annotation, which Python consequently does not put in the closure.
    for annotation in function.__annotations__.values():
        if isinstance(annotation, str):
            try:
                annotation_names = _global_names(
                    compile(annotation, "<annotation>", "eval")
                )
                referenced.update(annotation_names)
                global_bindings.update(
                    name
                    for name in annotation_names & function.__globals__.keys()
                    if environment is function.__globals__ or name not in environment
                )
            except SyntaxError:
                pass  # the constructor will report the invalid input annotation
    captured = {name: value for name, value in captured.items() if name in referenced}
    return ModelDefinition(
        inspect.signature(function),
        functions[0],
        captured,
        inspect.getsourcefile(function) or "<cell>",
        line,
        is_submodel,
        function.__globals__,
        frozenset(global_bindings),
        dict(zip(function.__code__.co_freevars, function.__closure__ or ())),
    )


def mechanism(function):
    """Capture equations, including cell source, for the explicit model constructor."""
    return _definition(function, inspect.currentframe().f_back.f_locals, False)


def submodel(function):
    """Capture a reusable subgraph; the caller's binding supplies its namespace."""
    return _definition(function, inspect.currentframe().f_back.f_locals, True)


def V(value, *, domain=None, lazy=False):
    """Expose the assigned local as a causal variable (interpreted by the compiler)."""
    raise DefinitionError(
        "V must appear in an assignment inside @mechanism or @submodel"
    )


def family(*, size=None, keys=None, domain=None):
    """Declare computed family members; every member needs an equation on every path."""
    raise DefinitionError(
        "family must appear in a named assignment inside a definition"
    )


def require(condition, *, error):
    """Validate an execution, including after interventions, without creating a node."""
    raise DefinitionError("require must appear inside a model definition")


class CausalModel:
    """A causal model compiled from an ``@mechanism`` definition.

    Attributes:
        variables (list): The model's variables, in timestep order.
        values (dict): Each variable's enumerable domain values, a compact
            ``range`` for a large integer domain, or ``None`` when the domain
            cannot be enumerated.
        domains (dict): Each variable's [`Dom`][causalab.causal.domains.Dom].
        mechanisms (dict): Each variable's compiled equation.
        parents (dict): Each variable's parents, read from its equation.
        children (dict): Each variable's children.
        print_pos (dict): Positions for plotting.
    """

    def __init__(
        self,
        definition: ModelDefinition,
        *,
        print_pos: dict[str, tuple[int, int]] | None = None,
        id: str = "null",
        embeddings: dict[str, Any] | None = None,
        periods: dict[str, float] | None = None,
        scoring: ScoringSpec | None = None,
        input_filter: Any = None,
    ) -> None:
        """Compile an ``@mechanism`` definition and attach task metadata.

        ``input_filter`` describes valid observations using input values. It is
        applied before computing the graph during sampling/enumeration; it does
        not restrict interventions. Domains validate all computed results.
        """
        from causalab.causal.compiler import Compiler, ConfigurationCopier

        compiled = Compiler(definition).compile()
        self.definition = ConfigurationCopier()(compiled.definition)
        self.mechanisms = compiled.mechanisms
        self.domains = compiled.domains
        self.values = {
            name: domain.public_values() for name, domain in self.domains.items()
        }
        self.families = compiled.families
        self.exogenous = compiled.exogenous
        self.return_variable = compiled.return_variable
        self._validators = compiled.validators
        self.inputs = compiled.inputs
        self.variables = list(self.mechanisms)
        for required in ("raw_input", "raw_output"):
            if required not in self.variables:
                raise ValueError(f"Variable {required!r} must be present in the model")
        self.parents = {name: list(eq.parents) for name, eq in self.mechanisms.items()}
        self.children = {name: [] for name in self.variables}
        self.timesteps = {}
        for name in self.variables:
            for parent in self.parents[name]:
                self.children[parent].append(name)
            self.timesteps[name] = max(
                (self.timesteps[p] + 1 for p in self.parents[name]), default=0
            )
        self.end_time = max(self.timesteps.values(), default=0)
        self.outputs = [name for name in self.variables if not self.children[name]]
        for name in self.outputs:
            if name not in self.inputs:
                self.timesteps[name] = self.end_time
        self.variables.sort(key=self.timesteps.__getitem__)
        self.id = id
        self.embeddings = ConfigurationCopier()(
            embeddings if embeddings is not None else {}
        )
        self.periods = ConfigurationCopier()(periods if periods is not None else {})
        if scoring is not None and not isinstance(scoring, ScoringSpec):
            raise TypeError(
                f"scoring must be a ScoringSpec, got {type(scoring).__name__}"
            )
        self._scoring = scoring
        self.input_filter = compiled.copy_configuration(input_filter)
        self.print_pos = dict(print_pos or {})
        self.print_pos.setdefault("raw_input", (0, -2))
        widths = {}
        for name in self.variables:
            step = self.timesteps[name]
            if name not in self.print_pos:
                self.print_pos[name] = (widths.get(step, 0), step)
                widths[step] = widths.get(step, 0) + 1
        self.equiv_classes = {}

    def _flatten(self, inputs):
        result = {}
        for name, value in (inputs or {}).items():
            if name in self.families:
                keys = list(self.families[name])
                if isinstance(value, Mapping):
                    valid_shape = set(value) == set(keys)
                else:
                    valid_shape = isinstance(value, (list, tuple)) and keys == list(
                        range(len(value))
                    )
                if not valid_shape:
                    raise ValueError(
                        f"Family {name!r} requires exactly these indices: {keys!r}"
                    )
                for key, member in self.families[name].items():
                    if member in result:
                        raise ValueError(f"Duplicate family member {member!r}")
                    result[member] = value[key]
            else:
                if name in result:
                    raise ValueError(f"Duplicate input {name!r}")
                result[name] = value
        return result

    # FUNCTIONS FOR RUNNING THE MODEL

    # ------------------------------------------------------------------ #
    # scoring — one source, derived views
    # ------------------------------------------------------------------ #

    @property
    def scoring(self) -> ScoringSpec | None:
        """The task's definition of correct
        ([`ScoringSpec`][causalab.causal.scoring.ScoringSpec]), or ``None`` for
        a model that declares none. Read-only: a spec is frozen at
        construction, and the only way to change what counts as correct is to
        construct a model with a new spec."""
        return self._scoring

    @property
    def output_tokens(self) -> dict[str, dict[Any, list[str]]] | None:
        """``{variable: {value: [forms]}}`` — a **derived, read-only view** of
        ``scoring.forms``, under the name the declaration always had. A fresh
        copy on every read, so editing it changes nothing; assigning to it
        raises. ``None`` when the model declares no scoring."""
        if self._scoring is None:
            return None
        return {
            var: {value: list(forms) for value, forms in var_map.items()}
            for var, var_map in self._scoring.forms.items()
        }

    @property
    def match_modes(self) -> dict[str, str] | None:
        """``{variable: string_mode}`` for every variable the spec declares —
        the **derived, read-only view** of ``scoring.string_mode`` under the
        retired per-variable name. ``None`` when the model declares no
        scoring."""
        if self._scoring is None:
            return None
        return {var: self._scoring.string_mode for var in self._scoring.forms}

    def new_trace(
        self, inputs: dict[str, Any] | None = None, *, noise=None
    ) -> CausalTrace:
        """Run explicit inputs; computed entries are initial interventions.

        Families accept either their named container or individual canonical
        member names. Stochastic traces require explicit noise; sampling it is
        the responsibility of ``sample_input`` or a task's dataset generator.
        """
        values = self._flatten(inputs)
        fixed = self._noise(noise, required=False)
        if values.keys() & fixed.keys():
            raise ValueError("Noise was supplied twice")
        values.update(fixed)
        if values and set(self.inputs) - set(self.exogenous) <= values.keys():
            missing = set(self.exogenous) - values.keys()
            if missing:
                raise ValueError(f"Supply explicit noise inputs: {sorted(missing)}")
        return CausalTrace(self, values)

    def _noise(self, noise, *, required):
        values = self._flatten(noise)
        extra = set(values) - set(self.exogenous)
        if extra:
            raise ValueError(f"Not exogenous inputs: {sorted(extra)}")
        missing = set(self.exogenous) - set(values)
        if required and missing:
            raise ValueError(
                f"Stochastic enumeration requires fixed noise for {sorted(missing)}"
            )
        for name, value in values.items():
            self.domains[name].validate(value, name)
        return values

    def run_interchange(
        self, input_trace: CausalTrace, counterfactual_inputs: dict[str, CausalTrace]
    ) -> CausalTrace:
        """
        Run the model with interchange interventions.

        Deprecated:
            This method exists primarily for the "<-" cross-variable syntax.
            For standard interchange (same variable name), prefer using copy + set directly:

            ```python
            # Instead of: result = model.run_interchange(trace, {"A": cf})
            # Use:
            result = trace.copy()
            result["A"] = cf["A"]
            ```

        Args:
            input_trace: Input trace.
            counterfactual_inputs: A dictionary mapping variables to their counterfactual input traces.
                Variable names can use the format "original_var<-counterfactual_var" to specify
                different variable names in the original and counterfactual inputs.

        Returns:
            A trace with the interchange intervention results.

        Examples:
            >>> # Cross-variable interchange (the main use case for this method)
            >>> model.run_interchange(trace, {"A<-B": counterfactual_input})
            >>> # Takes B's value from counterfactual, sets A in original

        Notes:
            The "<-" syntax is useful when the variable naming differs between
            original and counterfactual contexts, allowing flexible mapping of
            values across different variable names.
        """
        overrides = {}
        for variable, donor in counterfactual_inputs.items():
            target, source = (
                (part.strip() for part in variable.split("<-", 1))
                if "<-" in variable
                else (variable, variable)
            )
            overrides[target] = donor[source]
        return input_trace.copy().intervene_many(overrides)

    def enumerate_inputs(self, *, noise=None) -> list[CausalTrace]:
        """Enumerate observational inputs; stochastic models require fixed noise."""
        return [self.new_trace(inputs) for inputs in self._input_combinations(noise)]

    def _input_combinations(self, noise):
        """Filter input dictionaries without evaluating computed equations."""
        fixed = self._noise(noise, required=True)
        names = [name for name in self.inputs if name not in self.exogenous]
        for values in iter_combinations(self.domains[name] for name in names):
            inputs = {**dict(zip(names, values)), **fixed}
            if self.input_filter is None or self.input_filter(
                CausalTrace(self, inputs, eager=False)
            ):
                yield inputs

    def count_inputs(self, *, noise=None, limit=None) -> int:
        """Count valid inputs without computing outputs; stop at limit+1 if set."""
        self._noise(noise, required=True)
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("Input count limit must be a nonnegative integer")
        if self.input_filter is not None:
            count = 0
            for _ in self._input_combinations(noise):
                count += 1
                if limit is not None and count > limit:
                    break
            return count
        n = 1
        for name in self.inputs:
            if name not in self.exogenous:
                size = self.domains[name].cardinality(limit=limit)
                if size is None:
                    raise ValueError(f"Input {name!r} is not finitely enumerable")
                n *= size
                if limit is not None and n > limit:
                    return limit + 1
        return n

    @property
    def n_unique_inputs(self) -> int:
        """Deterministic input count. Use ``count_inputs(noise=...)`` for noise."""
        return self.count_inputs()

    def sample_input(self, filter_func=None, *, rng=None, noise=None) -> CausalTrace:
        """Sample independent explicit inputs, checking the observation filter."""
        rng = random if rng is None else rng
        fixed = self._noise(noise, required=False)
        for _ in range(10000):
            inputs = {
                name: fixed[name] if name in fixed else self.domains[name].sample(rng)
                for name in self.inputs
            }
            if self.input_filter is not None and not self.input_filter(
                CausalTrace(self, inputs, eager=False)
            ):
                continue
            trace = self.new_trace(inputs)
            if filter_func is None or filter_func(trace):
                return trace
        raise ValueError("No accepted input after 10000 sampling attempts")

    def label_counterfactual_data(
        self,
        examples: list[CounterfactualExample],
        target_variables: list[str],
        label_variable: str = "raw_output",
        *,
        setting_variables: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Label each example with the result of its interchange intervention."""
        return label_counterfactual_data(
            self,
            examples,
            target_variables,
            label_variable=label_variable,
            setting_variables=setting_variables,
        )

    def can_distinguish_with_dataset(
        self,
        examples: list[CounterfactualExample],
        target_variables1: list[str],
        target_variables2: list[str] | None,
        prints: bool = True,
    ) -> dict[str, float | int]:
        """
        Check if the model can distinguish between two sets of target variables
        using interchange interventions on counterfactual examples.

        Deprecated:
            Use the standalone
            [`causalab.causal.model_comparison.can_distinguish_with_dataset`][]
            instead. It supports comparing two *different* causal models
            (``target_variables2`` runs on ``causal_model2``); pass this model as
            both ``causal_model1`` and ``causal_model2`` to reproduce this method:

            ```python
            from causalab.causal.model_comparison import can_distinguish_with_dataset
            can_distinguish_with_dataset(
                examples, model, target_variables1,
                causal_model2=model, target_variables2=target_variables2,
            )
            ```
        """
        warnings.warn(
            "CausalModel.can_distinguish_with_dataset is deprecated; use "
            "causalab.causal.model_comparison.can_distinguish_with_dataset instead "
            "(pass this model as both causal_model1 and causal_model2).",
            DeprecationWarning,
            stacklevel=2,
        )
        from causalab.causal.model_comparison import can_distinguish_with_dataset

        return can_distinguish_with_dataset(
            examples,
            self,
            target_variables1,
            causal_model2=self if target_variables2 is not None else None,
            target_variables2=target_variables2,
        )


@dataclass(frozen=True)
class CompiledEquation:
    """Compute one variable from a trace and record its possible parents."""

    parents: list[str]
    compute: object
    lazy: bool = False

    def __call__(self, trace):
        return self.compute(trace)


class _PendingValue(BaseException):
    """Pause an equation until the requested parent has a value."""


class _MissingInput(KeyError):
    """A root input is unavailable, so a partial execution cannot proceed."""


class _TraceReader:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise _PendingValue(name) from None


class CausalTrace:
    """A trace caches values; interventions replace complete equations.

    Build traces with ``model.new_trace``. ``from_values`` is for adapters that
    only carry already-rendered text, without a causal model.
    """

    def __init__(self, model, inputs=None, *, eager=True):
        self.mechanisms = dict(model.mechanisms)
        self.domains = model.domains
        self.children = model.children
        self._order = model.variables
        self._validators = model._validators
        self._values = {}
        self._overrides = set()
        for name, value in (inputs or {}).items():
            self._validate(name, value)
            self._values[name] = copy.deepcopy(value)
            if name not in model.inputs:
                self._override(name, value)
        if eager:
            self._evaluate()

    @classmethod
    def from_values(cls, values):
        """A value-only trace for text/token adapters, with no hidden sampling."""
        result = object.__new__(cls)
        result.mechanisms = {name: CompiledEquation([], None) for name in values}
        result.domains = {}
        result.children = {name: [] for name in values}
        result._order = list(values)
        result._validators = []
        result._values = copy.deepcopy(values)
        result._overrides = set()
        for name, value in values.items():
            result._override(name, value)
        return result

    def _validate(self, name, value):
        if name not in self.mechanisms:
            raise KeyError(f"Unknown causal variable {name!r}")
        if name in self.domains:
            self.domains[name].validate(value, name)

    def _ready_variables(self):
        ready = set(self._values)
        for name in self._order:
            if name in ready:
                continue
            equation = self.mechanisms[name]
            if equation.compute is not None and all(
                parent in ready for parent in equation.parents
            ):
                ready.add(name)
        return ready

    def _evaluate(self):
        ready = self._ready_variables()
        for name in self._order:
            if not self.mechanisms[name].lazy and name in ready:
                self.get(name)
        for condition, message in self._validators:
            try:
                valid = condition(self)
            except _MissingInput:
                # A requirement only waits for inputs it actually reads. In
                # particular, short-circuit failures are already decidable.
                continue
            if not valid:
                raise ValueError(message)

    def get(self, variable: str) -> Any:
        if variable in self._values:
            return self._values[variable]
        pending = [variable]
        reader = _TraceReader(self._values)
        while pending:
            name = pending[-1]
            if name not in self.mechanisms:
                raise KeyError(f"Unknown causal variable {name!r}")
            equation = self.mechanisms[name]
            if equation.compute is None:
                raise _MissingInput(f"Input {name!r} has not been supplied")
            try:
                value = equation(reader)
            except _PendingValue as missing:
                # Evaluate the requested parent, then retry this pure equation.
                # Conditional reads stay lazy, including on deep graphs.
                pending.append(missing.args[0])
                continue
            self._validate(name, value)
            self._values[name] = copy.deepcopy(value)
            pending.pop()
        return self._values[variable]

    def __getitem__(self, variable):
        return self.get(variable)

    def __setitem__(self, variable, value):
        self.intervene(variable, value)

    def __contains__(self, variable):
        return variable in self._values

    def __delitem__(self, variable):
        """Drop the cached value; keep any active intervention equation."""
        self._values.pop(variable, None)
        self._invalidate({variable})

    def copy(self):
        result = object.__new__(type(self))
        result.__dict__ = dict(self.__dict__)
        result.mechanisms = dict(self.mechanisms)
        result._values = copy.deepcopy(self._values)
        result._overrides = set(self._overrides)
        return result

    def _override(self, name, value):
        saved = copy.deepcopy(value)
        self.mechanisms[name] = CompiledEquation([], lambda t: copy.deepcopy(saved))
        self._overrides.add(name)

    def _invalidate(self, changed):
        pending = list(changed)
        seen = set(changed)
        while pending:
            for child in self.children[pending.pop()]:
                if child in seen or child in self._overrides:
                    continue
                seen.add(child)
                self._values.pop(child, None)
                pending.append(child)

    def intervene_many(self, values):
        """Apply simultaneous overrides and validate eager results atomically.

        Lazy descendants validate when first read, which may be after this
        intervention commits. A lazy failure does not roll back that commit.
        """
        for name, value in values.items():
            self._validate(name, value)
        candidate = self.copy()
        for name, value in values.items():
            candidate._override(name, value)
            candidate._values[name] = copy.deepcopy(value)
        candidate._invalidate(set(values))
        candidate._evaluate()
        self.__dict__.update(candidate.__dict__)
        return self

    def intervene(self, variable, value):
        return self.intervene_many({variable: value})

    def snapshot(self, *, required: Sequence[str] = ()):
        """Copy cached values, interventions, and any explicitly requested values.

        An intervention remains active when its cached value is deleted. Read
        those constant equations directly so exports preserve interventions
        without repopulating this trace. Requested variables evaluate on a copy;
        the default does not force any lazy equations.
        """
        if isinstance(required, str):
            raise TypeError("required must be a sequence of variable names")
        required = tuple(required)
        trace = self.copy() if required else self
        for name in required:
            trace.get(name)
        values = dict(trace._values)
        for name in trace._overrides - values.keys():
            values[name] = trace.mechanisms[name](None)
        return copy.deepcopy(values)

    def to_dict(self):
        """Materialize every computable variable, including lazy equations."""
        ready = self._ready_variables()
        for name in self._order:
            if name in ready:
                self.get(name)
        return copy.deepcopy(self._values)
