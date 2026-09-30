"""Compile Python equations into a fixed causal graph.

The compiler expands bounded loops and submodels. It resolves variable
domains and conditional reads before it checks graph cycles.
"""

from __future__ import annotations

import ast
import builtins
import copy
import dis
import inspect
import itertools
import random
import types
from dataclasses import dataclass, is_dataclass
from graphlib import CycleError, TopologicalSorter

from causalab.causal.domains import (
    Dom,
    Exo,
    FamilyDom,
    FiniteValueError,
    _contains_nan,
    _equal,
    _fresh_nan,
    _nan_atoms,
    _nan_variants,
)
from causalab.causal.model import (
    CompiledEquation,
    DefinitionError,
    ModelDefinition,
    V,
    _global_names,
    family,
    mechanism,
    require,
    submodel,
)

_MUTATING_METHODS = {
    "add",
    "append",
    "byteswap",
    "clear",
    "difference_update",
    "discard",
    "extend",
    "fill",
    "insert",
    "intersection_update",
    "pop",
    "popitem",
    "put",
    "remove",
    "reverse",
    "resize",
    "setfield",
    "setflags",
    "setitem",
    "delitem",
    "setdefault",
    "sort",
    "symmetric_difference_update",
    "update",
    "__setitem__",
    "__delitem__",
    "__setattr__",
    "__delattr__",
    "__next__",
    "__iadd__",
    "__imul__",
    "__ior__",
    "__iand__",
    "__ixor__",
    "__isub__",
}


@dataclass
class VariableDefinition:
    expression: ast.expr | None
    domain: Dom | None
    lazy: bool = False


@dataclass(frozen=True)
class VariableFamily:
    members: dict
    domain: Dom | None = None


def _read(name):
    return ast.Call(ast.Name("__read__", ast.Load()), [ast.Constant(name)], [])


def _dependencies(expression):
    return list(
        dict.fromkeys(
            node.args[0].value
            for node in ast.walk(expression)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__read__"
        )
    )


def _read_guards(expression):
    """Conditions under which each syntactic read is evaluated by Python."""
    reads = {}

    def visit(node, guards):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__read__"
        ):
            reads.setdefault(node.args[0].value, []).append(
                ast.BoolOp(ast.And(), list(guards))
                if len(guards) > 1
                else guards[0]
                if guards
                else ast.Constant(True)
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__sequence__"
        ):
            first, second = node.args
            visit(first, guards)
            # A later step is reached only if its prefix finishes; the
            # prefix's value itself need not be truthy.
            completed = ast.Call(
                ast.Name("__sequence__", ast.Load()),
                [first, ast.Constant(True)],
                [],
            )
            visit(second, guards + [completed])
        elif isinstance(node, ast.IfExp):
            visit(node.test, guards)
            visit(node.body, guards + [node.test])
            visit(node.orelse, guards + [ast.UnaryOp(ast.Not(), node.test)])
        elif isinstance(node, ast.BoolOp):
            previous = list(guards)
            for value in node.values:
                visit(value, previous)
                previous = previous + [
                    value
                    if isinstance(node.op, ast.And)
                    else ast.UnaryOp(ast.Not(), value)
                ]
        elif isinstance(node, ast.Compare):
            visit(node.left, guards)
            previous = list(guards)
            left = node.left
            for operator, right in zip(node.ops, node.comparators):
                visit(right, previous)
                previous = previous + [ast.Compare(left, [operator], [right])]
                left = right
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, guards)

    visit(expression, [])
    return reads


def _family_name(name, key):
    keys = key if isinstance(key, tuple) else (key,)
    if not all(type(k) in (int, str) for k in keys):
        raise DefinitionError(
            "Family indices must be integers, strings, or tuples of them"
        )
    index = ",".join(repr(k) for k in keys)
    if isinstance(key, tuple):
        if not key:
            index = "()"
        elif len(key) == 1:
            index += ","
    return f"{name}[{index}]"


def _missing_index(index):
    raise KeyError(f"No family member at index {index!r}")


def _unpack_exact(value, shape):
    """Flatten a destructuring assignment, checking every nested target.

    Read at most one item beyond the requested arity, as Python unpacking does;
    materializing an arbitrary iterator would hang on infinite iterables.
    """
    if shape is None:
        return (value,)
    iterator = iter(value)
    items = []
    for _ in shape:
        try:
            items.append(next(iterator))
        except StopIteration:
            raise ValueError(
                f"not enough values to unpack (expected {len(shape)}, got {len(items)})"
            ) from None
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"too many values to unpack (expected {len(shape)})")
    return tuple(
        leaf for item, child in zip(items, shape) for leaf in _unpack_exact(item, child)
    )


class _PrivateBindings:
    """Private assignment values owned by a single equation evaluation."""

    def __init__(self):
        self.values = {}

    def __call__(self, key, compute):
        if key not in self.values:
            self.values[key] = compute()
        return self.values[key]


def _lambda(arguments, body):
    return ast.Lambda(
        ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body,
    )


def _check_helper_namespace(function):
    """Dynamic globals cannot be reproduced by a referenced-name snapshot."""
    namespace = function.__globals__
    builtin_namespace = namespace.get("__builtins__", builtins)
    if isinstance(builtin_namespace, types.ModuleType):
        builtin_namespace = vars(builtin_namespace)
    captured = {
        name: cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ())
    }
    defaults = function.__defaults__ or ()
    parameters = function.__code__.co_varnames[: function.__code__.co_argcount]
    captured.update(zip(parameters[len(parameters) - len(defaults) :], defaults))
    captured.update(function.__kwdefaults__ or {})

    def check(code):
        resolved = None
        for instruction in dis.get_instructions(code):
            if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                resolved = namespace.get(
                    instruction.argval, builtin_namespace.get(instruction.argval)
                )
            elif instruction.opname in ("LOAD_DEREF", "LOAD_FAST"):
                resolved = captured.get(instruction.argval)
            elif instruction.opname in ("LOAD_ATTR", "LOAD_METHOD"):
                resolved = inspect.getattr_static(resolved, instruction.argval, None)
            else:
                resolved = None
            if resolved is builtins.globals:
                raise DefinitionError(
                    f"{code.co_filename}:{code.co_firstlineno}: Helper {function.__qualname__!r} "
                    "uses dynamic globals(); pass referenced configuration explicitly"
                )
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                check(constant)

    check(function.__code__)


def _stored_values(value):
    """The values stored on ``value`` through native ``__dict__`` and slot
    descriptors only, so no user property or ``__getattr__`` runs."""
    classes = type.__dict__["__mro__"].__get__(type(value))
    stored = []
    for cls in classes:
        for name, descriptor in type.__dict__["__dict__"].__get__(cls).items():
            # Decide from the name and descriptor type before calling it: a C
            # getset such as object.__class__ or BaseException.args is not
            # storage, and only the __dict__ getset and slot members are.
            if name == "__dict__":
                is_storage = isinstance(
                    descriptor, (types.GetSetDescriptorType, types.MemberDescriptorType)
                )
            else:
                is_storage = (
                    isinstance(descriptor, types.MemberDescriptorType)
                    and name != "__weakref__"
                )
            if not is_storage:
                continue
            try:
                item = descriptor.__get__(value, type(value))
            except AttributeError:
                continue  # an uninitialized slot
            if name == "__dict__":
                stored.extend(item.values())
            else:
                stored.append(item)
    return stored


class ConfigurationCopier:
    """Snapshot referenced configuration, including a Python helper's closure.

    Modules and types are code, not model configuration. In particular we never
    deepcopy an entire module dictionary (which may contain a neural model).
    """

    def __init__(self):
        self.memo = {}
        self._active_numpy = set()
        # Container and function snapshots do not retain their originals. Keep
        # those identities alive until the entire snapshot operation finishes.
        self.originals = {}

    def __call__(self, value):
        if id(value) in self.memo:
            return self.memo[id(value)]
        if isinstance(value, (types.ModuleType, type)):
            return value
        self.originals[id(value)] = value
        if isinstance(value, types.GenericAlias):
            result = types.GenericAlias(self(value.__origin__), self(value.__args__))
            self.memo[id(value)] = result
            return result
        if isinstance(value, types.UnionType):
            arguments = [self(argument) for argument in value.__args__]
            result = arguments[0]
            for argument in arguments[1:]:
                result |= argument
            self.memo[id(value)] = result
            return result
        if type(value).__module__ == "typing":
            import typing

            # Type parameters and special forms are annotation code, like
            # classes. Their identities must survive inside generic aliases.
            if isinstance(value, (typing.TypeVar, typing.ParamSpec)) or any(
                value is getattr(typing, name, None)
                for name in (
                    "Any",
                    "NoReturn",
                    "Never",
                    "Self",
                    "Union",
                    "Optional",
                    "ClassVar",
                    "Final",
                    "Literal",
                    "Concatenate",
                    "TypeAlias",
                    "TypeGuard",
                    "Required",
                    "NotRequired",
                    "LiteralString",
                    "Unpack",
                )
            ):
                return value
            if isinstance(value, (typing.ParamSpecArgs, typing.ParamSpecKwargs)):
                return value
            if isinstance(value, typing.ForwardRef):
                result = typing.ForwardRef(
                    value.__forward_arg__,
                    is_argument=value.__forward_is_argument__,
                    module=value.__forward_module__,
                    is_class=value.__forward_is_class__,
                )
                self.memo[id(value)] = result
                result.__forward_evaluated__ = value.__forward_evaluated__
                result.__forward_value__ = self(value.__forward_value__)
                return result
            if typing.get_origin(value) is typing.Annotated:
                result = typing.Annotated[self(typing.get_args(value))]
                self.memo[id(value)] = result
                return result
            if hasattr(value, "copy_with"):
                if not hasattr(value, "__args__"):
                    return value  # Unparameterized aliases such as typing.List.
                arguments = self(value.__args__)
                result = value.copy_with(arguments)
                if any(
                    copied is not actual
                    for copied, actual in zip(arguments, result.__args__)
                ):
                    # Union.copy_with interns equivalent arguments, which can
                    # reintroduce the original ForwardRef's mutable cache.
                    detached = object.__new__(type(result))
                    object.__setattr__(detached, "__dict__", vars(result).copy())
                    object.__setattr__(detached, "__args__", arguments)
                    result = detached
                self.memo[id(value)] = result
                return result
        if isinstance(value, ModelDefinition):
            result = copy.copy(value)
            self.memo[id(value)] = result
            result.environment = self(value._current_environment())
            # Compiled/public definitions own snapshots, not references back to
            # live globals or closure cells (which can retain unrelated state).
            result._global_namespace = None
            result._global_bindings = frozenset()
            result._closure_cells = {}
            result.signature = value.signature.replace(
                parameters=[
                    p.replace(default=self(p.default), annotation=self(p.annotation))
                    for p in value.signature.parameters.values()
                ],
                return_annotation=self(value.signature.return_annotation),
            )
            return result
        if isinstance(value, (dict, list, tuple, set, frozenset)) and type(
            value
        ) not in (dict, list, tuple, set, frozenset):
            raise DefinitionError(
                f"Cannot snapshot container subclass {type(value).__name__}; use a plain container"
            )
        if isinstance(value, dict):
            result = {}
            self.memo[id(value)] = result
            result.update((self(k), self(v)) for k, v in value.items())
            return result
        if isinstance(value, list):
            result = []
            self.memo[id(value)] = result
            result.extend(self(v) for v in value)
            return result
        if isinstance(value, tuple):
            result = tuple(self(v) for v in value)
            self.memo[id(value)] = result
            return result
        if isinstance(value, (set, frozenset)):
            result = type(value)(self(v) for v in value)
            self.memo[id(value)] = result
            return result
        if isinstance(value, types.FunctionType):
            if any(
                value is marker for marker in (V, family, require, mechanism, submodel)
            ):
                return value  # Preserve the identity of equation markers.
            _check_helper_namespace(value)
            env = {"__builtins__": value.__globals__.get("__builtins__", builtins)}
            # Install the function in the memo before visiting recursive globals.
            cells = tuple(types.CellType(None) for _ in (value.__closure__ or ()))
            result = types.FunctionType(
                value.__code__, env, value.__name__, None, cells or None
            )
            self.memo[id(value)] = result

            for name in _global_names(value.__code__):
                if name in value.__globals__:
                    env[name] = self(value.__globals__[name])
            for dest, src in zip(cells, value.__closure__ or ()):
                dest.cell_contents = self(src.cell_contents)
            result.__defaults__ = self(value.__defaults__)
            result.__kwdefaults__ = self(value.__kwdefaults__)
            result.__dict__ = self(value.__dict__)
            result.__annotations__ = self(value.__annotations__)
            result.__qualname__ = value.__qualname__
            result.__module__ = value.__module__
            result.__doc__ = value.__doc__
            return result
        if isinstance(value, types.MethodType):
            result = types.MethodType(self(value.__func__), self(value.__self__))
            self.memo[id(value)] = result
            return result
        if isinstance(value, (types.SimpleNamespace, Dom)) or is_dataclass(type(value)):
            # Copy physical storage, without invoking descriptor accessors or
            # constructors. Class methods remain code and must be pure helpers.
            result = (
                object.__new__(type(value))
                if not isinstance(value, types.SimpleNamespace)
                else types.SimpleNamespace.__new__(type(value))
            )
            self.memo[id(value)] = result
            self._storage(value, result)
            return result
        if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            if value is builtins.globals:
                raise DefinitionError(
                    "Dynamic globals() cannot be snapshotted; pass referenced configuration explicitly"
                )
            owner = value.__self__
            if owner is not None and not isinstance(owner, types.ModuleType):
                return getattr(self(owner), value.__name__)
            return value
        if type(value).__module__.startswith("numpy"):
            return self._numpy(value)
        elif not isinstance(
            value,
            (
                type(None),
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                range,
                slice,
                types.CodeType,
                type(Ellipsis),
                type(NotImplemented),
            ),
        ):
            raise DefinitionError(
                f"Cannot snapshot build configuration {type(value).__name__}; "
                "use plain containers, a dataclass, or SimpleNamespace"
            )
        try:
            result = copy.deepcopy(value, self.memo)
        except (TypeError, ValueError) as exc:
            raise DefinitionError(
                f"Cannot snapshot build configuration {type(value).__name__}"
            ) from exc
        self.memo[id(value)] = result
        return result

    def _storage(self, value, result):
        # Read the real class dictionaries even if a metaclass defines its own
        # __dict__ or __mro__ property. Only native storage descriptors are used.
        classes = type.__dict__["__mro__"].__get__(type(value))
        dictionaries = [type.__dict__["__dict__"].__get__(cls) for cls in classes]
        for attributes in dictionaries:
            descriptor = attributes.get("__dict__")
            if isinstance(
                descriptor, (types.GetSetDescriptorType, types.MemberDescriptorType)
            ):
                original = descriptor.__get__(value, type(value))
                target = descriptor.__get__(result, type(result))
                if id(original) not in self.memo:
                    self.memo[id(original)] = target
                    self.originals[id(original)] = original
                    target.update((self(k), self(v)) for k, v in original.items())
                else:
                    copied = self.memo[id(original)]
                    try:
                        descriptor.__set__(result, copied)
                    except AttributeError as exc:
                        # A read-only namespace dictionary cannot be replaced
                        # by a dictionary already captured through another path.
                        raise DefinitionError(
                            f"Cannot snapshot a shared read-only __dict__ for {type(value).__name__}; "
                            "capture the object instead of its dictionary"
                        ) from exc
                break
        else:
            if type.__dict__["__dictoffset__"].__get__(type(value)):
                raise DefinitionError(
                    f"Cannot snapshot hidden __dict__ storage for {type(value).__name__}; "
                    "keep a native __dict__ descriptor accessible on its class hierarchy"
                )
        for attributes in dictionaries:
            for descriptor in attributes.values():
                if isinstance(descriptor, types.MemberDescriptorType) and (
                    descriptor.__name__ not in ("__dict__", "__weakref__")
                ):
                    try:
                        item = descriptor.__get__(value, type(value))
                    except AttributeError:
                        continue  # Preserve an uninitialized slot.
                    descriptor.__set__(result, self(item))

    def _numpy(self, value):
        import numpy as np

        if id(value) in self._active_numpy:
            raise DefinitionError("Cyclic NumPy metadata cannot be snapshotted")
        self._active_numpy.add(id(value))
        try:
            if isinstance(value, np.dtype):
                if value.hasobject:
                    raise DefinitionError(
                        "Object arrays cannot be snapshotted; use plain containers"
                    )
                if value.fields is not None:
                    fields = [value.fields[name] for name in value.names]
                    result = np.dtype(
                        {
                            "names": list(value.names),
                            "formats": [self(field[0]) for field in fields],
                            "offsets": [field[1] for field in fields],
                            "titles": [
                                self(field[2]) if len(field) > 2 else None
                                for field in fields
                            ],
                            "itemsize": value.itemsize,
                        },
                        align=value.isalignedstruct,
                    )
                elif value.subdtype is not None:
                    base, shape = value.subdtype
                    result = np.dtype((self(base), shape))
                elif value.kind in "biufc":
                    # dtype.str alone merges int64 and longlong on platforms
                    # where indexing them returns distinct scalar classes.
                    result = np.dtype(value.type).newbyteorder(value.byteorder)
                else:
                    result = np.dtype(value.str)
                if value.metadata is not None:
                    # Wrapping the original dtype retains its metadata values
                    # even when a replacement mapping is provided to NumPy.
                    result = np.dtype(result, metadata=self(dict(value.metadata)))
            elif type(value) is np.ndarray:
                dtype = self(value.dtype)
                result = value.copy(order="K")
                result.dtype = dtype
            elif isinstance(value, np.generic):
                result = value.copy().view(self(value.dtype))
            else:
                raise DefinitionError(
                    f"Cannot snapshot NumPy configuration {type(value).__name__}; "
                    "use numeric arrays or scalars"
                )
            self.memo[id(value)] = result
            return result
        finally:
            self._active_numpy.remove(id(value))


class Compiler:
    def __init__(self, definition, *, inference_limit=4096):
        if not isinstance(definition, ModelDefinition) or definition.is_submodel:
            raise TypeError("CausalModel requires an @mechanism definition")
        self.copy_configuration = ConfigurationCopier()
        self.definition = self.copy_configuration(definition)
        self.inference_limit = inference_limit
        self.nodes = {}
        self.inputs = []
        self.exogenous = []
        self.families = {}
        self.constraints = []
        self.constants = {
            "__builtins__": builtins.__dict__,
            "__missing_index__": _missing_index,
            "__unpack_exact__": _unpack_exact,
        }
        self.stack = []
        self._constant_id = 0
        self._binding_id = 0
        self._expanded_steps = 0
        self._ast_size = 0

    def expansion_step(self):
        self._expanded_steps += 1
        if self._expanded_steps > 10000:
            raise DefinitionError(
                "Total loop/comprehension expansion exceeds 10000 steps; use a pure helper"
            )

    def check_expression(self, expression, *, account=True):
        pending = [(expression, 0)]
        size = self._ast_size if account else 0
        while pending:
            node, depth = pending.pop()
            size += 1
            if size > 100000 or depth > 200:
                raise DefinitionError(
                    "Expanded expression complexity exceeds the compiler limit; use a pure helper"
                )
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        if account:
            self._ast_size = size

    def constant(self, value):
        name = f"__config_{self._constant_id}"
        self._constant_id += 1
        self.constants[name] = self.copy_configuration(value)
        return ast.Name(name, ast.Load())

    def evaluate(self, expression, read=None):
        return self.expression_function(expression)(read, _PrivateBindings())

    def fixed(self, expression, env, definition):
        expression = self.expression(expression, env, definition)
        if _dependencies(expression):
            raise DefinitionError(
                "Bounds, domains and family keys must be fixed build configuration"
            )
        return self.evaluate(expression)

    def resolve(self, node, definition=None):
        """Find equation markers through static name and attribute lookup."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__private_binding__"
        ):
            return self.resolve(node.args[1], definition)
        if isinstance(node, ast.Name):
            environment = (
                self.constants if definition is None else definition.environment
            )
            return environment.get(node.id, getattr(builtins, node.id, None))
        if isinstance(node, ast.Attribute):
            owner = self.resolve(node.value, definition)
            return inspect.getattr_static(owner, node.attr, None)
        return None

    def possible(self, expression):
        """Prove small selector/guard ranges without enumerating selected data."""
        parents = _dependencies(expression)
        choices, count = [], 1
        for name in parents:
            domain = self.domains.get(name)
            if domain is None:
                return None
            values = domain.enumerated(self.inference_limit)
            if values is None or any(_contains_nan(value) for value in values):
                return None
            count *= len(values)
            if count > self.inference_limit:
                return None
            choices.append(values)
        compute = self.function(expression)
        try:
            return [
                compute(dict(zip(parents, combo)))
                for combo in itertools.product(*choices)
            ]
        except (ArithmeticError, LookupError, TypeError, ValueError):
            return None

    def expression(self, expression, env, definition):
        compiler = self

        class Rewrite(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in env:
                    value = env[node.id]
                    if isinstance(value, VariableFamily):
                        # A whole family passed to an opaque helper is an explicit
                        # dependency on every member, just like scalar arguments.
                        keys = list(value.members)
                        if keys == list(range(len(keys))):
                            return ast.Tuple(
                                [_read(v) for v in value.members.values()], ast.Load()
                            )
                        return ast.Dict(
                            [compiler.constant(k) for k in keys],
                            [_read(v) for v in value.members.values()],
                        )
                    return copy.deepcopy(value)
                if node.id in definition.environment:
                    return compiler.constant(definition.environment[node.id])
                if hasattr(builtins, node.id):
                    return node
                raise DefinitionError(
                    f"Unknown local {node.id!r}; every value must be defined on every path"
                )

            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name) and isinstance(
                    env.get(node.value.id), VariableFamily
                ):
                    members = env[node.value.id].members
                    index = self.visit(node.slice)
                    if not _dependencies(index):
                        key = compiler.evaluate(index)
                        try:
                            return _read(members[key])
                        except (KeyError, TypeError) as exc:
                            raise DefinitionError(
                                f"Unknown family index {key!r}"
                            ) from exc
                    result = ast.Call(
                        ast.Name("__missing_index__", ast.Load()), [index], []
                    )
                    for key, name in reversed(list(members.items())):
                        result = ast.IfExp(
                            ast.Compare(
                                copy.deepcopy(index),
                                [ast.Eq()],
                                [compiler.constant(key)],
                            ),
                            _read(name),
                            result,
                        )
                    return result
                return self.generic_visit(node)

            def visit_IfExp(self, node):
                test = self.visit(node.test)
                if not _dependencies(test):
                    return self.visit(
                        node.body if compiler.evaluate(test) else node.orelse
                    )
                return ast.IfExp(test, self.visit(node.body), self.visit(node.orelse))

            def visit_Lambda(self, node):
                raise DefinitionError(
                    "Define ordinary helper functions outside the equations instead of inline lambdas"
                )

            def visit_NamedExpr(self, node):
                raise DefinitionError(
                    "Use a separate assignment instead of := in an equation"
                )

            def visit_Compare(self, node):
                node = self.generic_visit(node)

                def singleton(value):
                    if isinstance(value, ast.Constant):
                        return (
                            value.value is None
                            or type(value.value) is bool
                            or value.value is Ellipsis
                        )
                    resolved = compiler.resolve(value)
                    return (
                        isinstance(resolved, type)
                        or resolved is Ellipsis
                        or resolved is NotImplemented
                    )

                left = node.left
                for operator, right in zip(node.ops, node.comparators):
                    if isinstance(operator, (ast.Is, ast.IsNot)) and not (
                        singleton(left) or singleton(right)
                    ):
                        raise DefinitionError(
                            "Object identity is not a causal value; use == or compare to None, a Boolean, or a type"
                        )
                    left = right
                return node

            def visit_Call(self, node):
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise DefinitionError(
                        "Call a named helper or method; put dynamic callable selection in a pure helper"
                    )
                called = compiler.resolve(node.func, definition)
                if called is setattr or called is delattr:
                    raise DefinitionError(
                        "Mutation requires a pure helper that owns its local state"
                    )
                if isinstance(called, ModelDefinition) or any(
                    called is marker for marker in (V, family, require)
                ):
                    raise DefinitionError(
                        "Bind V, family, and submodel calls in a separate named assignment"
                    )
                if (
                    getattr(called, "__module__", "")
                    in ("random", "numpy.random", "numpy.random.mtrand")
                    and getattr(called, "__name__", "") != "Random"
                ):
                    raise DefinitionError(
                        "Randomness must be an explicit Exo input; use a seeded pure helper"
                    )
                rewritten = self.generic_visit(node)
                function = rewritten.func
                while (
                    isinstance(function, ast.Call)
                    and isinstance(function.func, ast.Name)
                    and function.func.id == "__private_binding__"
                ):
                    function = function.args[1]
                if compiler.resolve(function) is id:
                    raise DefinitionError(
                        "Object identity is not a causal value; helpers must depend on values, not id()"
                    )
                if any(
                    compiler.resolve(function) is builtin for builtin in (iter, next)
                ):
                    raise DefinitionError(
                        "Stateful iter()/next() evaluation requires a pure helper that owns its local iterator"
                    )
                if compiler.resolve(function) is builtins.globals:
                    raise DefinitionError(
                        "Dynamic globals() cannot be snapshotted; pass referenced configuration explicitly"
                    )
                method = None
                if isinstance(function, ast.Attribute):
                    owner = compiler.resolve(function.value)
                    if not isinstance(owner, types.ModuleType):
                        method = function.attr
                elif isinstance(function, ast.Name):
                    bound = compiler.constants.get(function.id)
                    if isinstance(
                        bound,
                        (types.MethodType, types.MethodDescriptorType),
                    ) or (
                        isinstance(bound, types.BuiltinMethodType)
                        and bound.__self__ is not None
                        and not isinstance(bound.__self__, types.ModuleType)
                    ):
                        method = bound.__name__
                if method in _MUTATING_METHODS:
                    raise DefinitionError(
                        f"Mutating method {method!r} requires a pure helper that owns its local state"
                    )
                return rewritten

            def visit_ListComp(self, node):
                return compiler.comprehension(node, env, definition, "list")

            def visit_SetComp(self, node):
                return compiler.comprehension(node, env, definition, "set")

            def visit_GeneratorExp(self, node):
                raise DefinitionError(
                    "Generator expressions require a pure helper to preserve lazy evaluation"
                )

            def visit_DictComp(self, node):
                raise DefinitionError(
                    "Use a pure helper for a dictionary comprehension"
                )

        self.check_expression(expression)
        try:
            rewritten = Rewrite().visit(copy.deepcopy(expression))
            self.check_expression(rewritten)
            return ast.fix_missing_locations(rewritten)
        except RecursionError as exc:
            raise DefinitionError(
                "Expanded expression complexity exceeds the compiler limit; use a pure helper"
            ) from exc

    def private_binding(self, expression):
        """Give an initializer one value per consuming equation evaluation."""
        if isinstance(expression, (ast.Name, ast.Constant)) or (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in ("__read__", "__private_binding__")
        ):
            # References already reuse their value. Keeping exposed aliases
            # direct also preserves the public return-alias contract.
            return expression
        key = self._binding_id
        self._binding_id += 1
        return ast.Call(
            ast.Name("__private_binding__", ast.Load()),
            [ast.Constant(key), expression],
            [],
        )

    @staticmethod
    def sequence(first, second):
        return ast.Call(ast.Name("__sequence__", ast.Load()), [first, second], [])

    def bind(self, target, expression, env, namespace=None, *, cache=True):
        if isinstance(target, ast.Name):
            if namespace is not None and (
                namespace + target.id in self.nodes
                or isinstance(env.get(target.id), VariableFamily)
            ):
                raise DefinitionError(
                    "An exposed variable or family cannot be reassigned"
                )
            env[target.id] = self.private_binding(expression) if cache else expression
        elif isinstance(target, (ast.Tuple, ast.List)):
            targets = []

            def unpack_shape(item):
                if isinstance(item, ast.Name):
                    targets.append(item)
                    return None
                if isinstance(item, (ast.Tuple, ast.List)):
                    return tuple(unpack_shape(child) for child in item.elts)
                raise DefinitionError(
                    "Unpacking requires named targets; put starred unpacking or mutation in a pure helper"
                )

            shape = unpack_shape(target)
            if not targets:
                raise DefinitionError(
                    "Unpacking needs a named target; put an empty-target arity check in a pure helper or require"
                )
            key = self._binding_id
            self._binding_id += 1
            # Keep the complete RHS visible to read/domain analysis. Only the
            # final executable lowering replaces it with a cached lazy thunk.
            binding = ast.Call(
                ast.Name("__unpack_binding__", ast.Load()),
                [ast.Constant(key), expression, self.constant(shape)],
                [],
            )
            for i, item in enumerate(targets):
                self.bind(
                    item,
                    ast.Subscript(copy.deepcopy(binding), ast.Constant(i), ast.Load()),
                    env,
                    namespace,
                    cache=False,
                )
            return binding
        else:
            raise DefinitionError(
                "Private locals use named assignments; mutation belongs in a pure helper"
            )

    def comprehension(self, node, env, definition, container):
        def empty():
            return ast.List([], ast.Load())

        def expand(level, local):
            if level == len(node.generators):
                return ast.List(
                    [self.expression(node.elt, local, definition)], ast.Load()
                )
            gen = node.generators[level]
            if gen.is_async:
                raise DefinitionError("Async comprehensions are not supported")
            iterable = self.expression(gen.iter, local, definition)
            if _dependencies(iterable):
                raise DefinitionError(
                    "Bounds, domains and family keys must be fixed build configuration"
                )
            try:
                values = self.evaluate(iterable)
            except (ArithmeticError, LookupError, TypeError, ValueError):
                # A fixed iterable can still fail only when an outer filter
                # permits this generator to run. Retain that runtime failure.
                return self.sequence(iterable, empty())
            if not hasattr(values, "__len__") or len(values) > 10000:
                raise DefinitionError(
                    "Comprehensions need a fixed collection of at most 10000 items"
                )
            # Expansion fixes the iteration count, not the item objects. Read
            # items from this equation's private values so aliases (including
            # repeated NaNs) survive between the iterable and its consumers.
            items = self.private_binding(
                ast.Call(
                    ast.Name("__unpack_exact__", ast.Load()),
                    [iterable, self.constant((None,) * len(values))],
                    [],
                )
            )
            parts = []
            for index in range(len(values)):
                self.expansion_step()
                nested = dict(local)
                binding = self.bind(
                    gen.target,
                    ast.Subscript(
                        copy.deepcopy(items), ast.Constant(index), ast.Load()
                    ),
                    nested,
                )
                conditions = []
                for guard in gen.ifs:
                    condition = self.expression(guard, nested, definition)
                    if not _dependencies(condition):
                        try:
                            accepted = bool(self.evaluate(condition))
                        except (ArithmeticError, LookupError, TypeError, ValueError):
                            # A preceding runtime filter may shield this error.
                            conditions.append(condition)
                            part = empty()
                            break
                        if not accepted:
                            part = empty()
                            break
                    else:
                        conditions.append(condition)
                else:
                    part = expand(level + 1, nested)
                for condition in reversed(conditions):
                    part = ast.IfExp(condition, part, empty())
                if binding is not None:
                    # Target unpacking precedes this generator's filters even
                    # when none of its names are consumed by the expression.
                    part = self.sequence(binding, part)
                parts.append(ast.Starred(part, ast.Load()))
            return self.sequence(items, ast.List(parts, ast.Load()))

        result = expand(0, dict(env))
        return (
            result
            if container == "list"
            else ast.Call(ast.Name(container, ast.Load()), [result], [])
        )

    def add_node(self, name, expression, domain=None, lazy=False):
        if name in self.nodes:
            raise DefinitionError(
                f"Causal variable {name!r} is assigned more than once; use an indexed family for steps"
            )
        if domain is not None and not isinstance(domain, Dom):
            raise DefinitionError(f"Domain for {name!r} must be a Dom")
        self.nodes[name] = VariableDefinition(expression, domain, lazy)

    def block(self, statements, env, definition, namespace, written=None):
        written = set() if written is None else written
        returned = None
        for index, statement in enumerate(statements):
            try:
                if isinstance(statement, ast.Expr) and isinstance(
                    statement.value, ast.Constant
                ):
                    continue  # docstring
                if isinstance(statement, ast.Pass):
                    continue
                if isinstance(statement, ast.Return):
                    if index != len(statements) - 1:
                        raise DefinitionError(
                            "Return belongs at the end; use explicit bounded steps for early stopping"
                        )
                    returned = self.expression(statement.value, env, definition)
                elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    if len(targets) != 1:
                        raise DefinitionError("Use one assignment target at a time")
                    target, value = targets[0], statement.value
                    called = (
                        self.resolve(value.func, definition)
                        if isinstance(value, ast.Call)
                        else None
                    )
                    if called is V:
                        if not isinstance(target, ast.Name) or len(value.args) != 1:
                            raise DefinitionError(
                                "Use name = V(expression, domain=..., lazy=...)"
                            )
                        if target.id in env:
                            raise DefinitionError(
                                f"Cannot redefine exposed variable or local {target.id!r}"
                            )
                        options = {
                            k.arg: self.fixed(k.value, env, definition)
                            for k in value.keywords
                        }
                        if set(options) - {"domain", "lazy"}:
                            raise DefinitionError("Unknown V option")
                        name = namespace + target.id
                        self.add_node(
                            name,
                            self.expression(value.args[0], env, definition),
                            **options,
                        )
                        env[target.id] = _read(name)
                    elif called is family:
                        if (
                            not isinstance(target, ast.Name)
                            or target.id in env
                            or value.args
                        ):
                            raise DefinitionError(
                                "Bind family(size=N) or family(keys=...) to a new local"
                            )
                        options = {
                            k.arg: self.fixed(k.value, env, definition)
                            for k in value.keywords
                        }
                        if set(options) - {"size", "keys", "domain"} or (
                            "size" in options
                        ) == ("keys" in options):
                            raise DefinitionError(
                                "Specify exactly one of family(size=...) and family(keys=...)"
                            )
                        if "size" in options and (
                            type(options["size"]) is not int or options["size"] < 0
                        ):
                            raise DefinitionError(
                                "Family size must be a nonnegative integer"
                            )
                        keys = (
                            list(range(options["size"]))
                            if "size" in options
                            else list(options["keys"])
                        )
                        if len(set(keys)) != len(keys):
                            raise DefinitionError("Family keys must be unique")
                        names = [
                            _family_name(namespace + target.id, key) for key in keys
                        ]
                        if len(set(names)) != len(names):
                            raise DefinitionError(
                                "Family keys must have unique canonical names"
                            )
                        members = dict(zip(keys, names))
                        domain = options.get("domain")
                        if isinstance(domain, FamilyDom) and set(domain.domains) != set(
                            keys
                        ):
                            raise DefinitionError(
                                "A family domain must describe exactly its declared keys"
                            )
                        env[target.id] = VariableFamily(members, options.get("domain"))
                        self.families[namespace + target.id] = members
                    elif isinstance(called, ModelDefinition):
                        if (
                            not called.is_submodel
                            or not isinstance(target, ast.Name)
                            or target.id in env
                        ):
                            raise DefinitionError(
                                "A @submodel call needs a fresh named binding"
                            )
                        args = [self.argument(a, env, definition) for a in value.args]
                        kwargs = {
                            k.arg: self.argument(k.value, env, definition)
                            for k in value.keywords
                        }
                        bound = called.signature.bind(*args, **kwargs)
                        for param in called.signature.parameters.values():
                            if param.name not in bound.arguments:
                                bound.arguments[param.name] = self.constant(
                                    param.default
                                )
                        before_submodel = set(self.nodes)
                        env[target.id] = self.expand(
                            called, dict(bound.arguments), namespace + target.id + "."
                        )
                        if env[target.id].args[0].value not in (
                            self.nodes.keys() - before_submodel
                        ):
                            # A passthrough return has no new equation to own
                            # this call site's condition; its alias must keep it.
                            written.add(target.id)
                    elif (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and isinstance(env.get(target.value.id), VariableFamily)
                    ):
                        fam = env[target.value.id]
                        key = self.fixed(target.slice, env, definition)
                        if key not in fam.members:
                            raise DefinitionError(f"Unknown family index {key!r}")
                        self.add_node(
                            fam.members[key],
                            self.expression(value, env, definition),
                            fam.domain.domains[key]
                            if isinstance(fam.domain, FamilyDom)
                            else fam.domain,
                        )
                    else:
                        if isinstance(target, ast.Name) and target.id in env:
                            previous = env[target.id]
                            if isinstance(previous, VariableFamily) or _dependencies(
                                previous
                            ) == [namespace + target.id]:
                                raise DefinitionError(
                                    "An exposed variable or family cannot be reassigned"
                                )
                        self.bind(
                            target,
                            self.expression(value, env, definition),
                            env,
                            namespace,
                        )
                        written.update(
                            node.id
                            for node in ast.walk(target)
                            if isinstance(node, ast.Name)
                        )
                elif isinstance(statement, ast.If):
                    if any(
                        isinstance(n, ast.Return)
                        for branch in (statement.body, statement.orelse)
                        for s in branch
                        for n in ast.walk(s)
                    ):
                        raise DefinitionError(
                            "Return belongs at the end; define both branches and then return"
                        )
                    condition = self.expression(statement.test, env, definition)
                    if not _dependencies(condition):
                        self.block(
                            statement.body
                            if self.evaluate(condition)
                            else statement.orelse,
                            env,
                            definition,
                            namespace,
                            written,
                        )
                    else:
                        before_nodes = dict(self.nodes)
                        before_families = dict(self.families)
                        before_constraints = list(self.constraints)
                        yes = dict(env)
                        yes_written = set()
                        self.block(
                            statement.body, yes, definition, namespace, yes_written
                        )
                        yes_nodes, yes_families, yes_constraints = (
                            self.nodes,
                            self.families,
                            self.constraints,
                        )
                        self.nodes, self.families, self.constraints = (
                            dict(before_nodes),
                            dict(before_families),
                            list(before_constraints),
                        )
                        no = dict(env)
                        no_written = set()
                        self.block(
                            statement.orelse, no, definition, namespace, no_written
                        )
                        branch_written = yes_written | no_written
                        written.update(branch_written)
                        if (
                            set(yes_nodes) != set(self.nodes)
                            or yes_families != self.families
                        ):
                            raise DefinitionError(
                                "Every causal variable must be defined on both branches; use None for inactivity"
                            )
                        if len(yes_constraints) != len(before_constraints) or len(
                            self.constraints
                        ) != len(before_constraints):
                            raise DefinitionError(
                                "Place require after the conditional, using an implication if needed"
                            )
                        for name in set(self.nodes) - set(before_nodes):
                            a, b = yes_nodes[name], self.nodes[name]
                            if a.lazy != b.lazy or (a.domain is None) != (
                                b.domain is None
                            ):
                                raise DefinitionError(
                                    "Both branches must use the same domain and lazy policy"
                                )
                            domain = (
                                Dom.union(a.domain, b.domain)
                                if a.domain is not None
                                else None
                            )
                            self.nodes[name] = VariableDefinition(
                                ast.IfExp(condition, a.expression, b.expression),
                                domain,
                                a.lazy,
                            )
                        for name in set(yes) & set(no):
                            a, b = yes[name], no[name]
                            if isinstance(a, VariableFamily) or isinstance(
                                b, VariableFamily
                            ):
                                if a != b:
                                    raise DefinitionError(
                                        "Family shape must be fixed across branches"
                                    )
                                env[name] = a
                            elif name not in branch_written and ast.dump(a) == ast.dump(
                                b
                            ):
                                env[name] = a
                            else:
                                env[name] = ast.IfExp(condition, a, b)
                        for name in set(yes) ^ set(no):
                            env.pop(name, None)
                elif isinstance(statement, ast.For):
                    if any(
                        isinstance(n, ast.Return)
                        for s in statement.body
                        for n in ast.walk(s)
                    ):
                        raise DefinitionError(
                            "Use explicit bounded steps instead of returning inside a loop"
                        )
                    values = self.fixed(statement.iter, env, definition)
                    if (
                        not hasattr(values, "__len__")
                        or len(values) > 10000
                        or statement.orelse
                    ):
                        raise DefinitionError(
                            "Use a fixed loop of at most 10000 steps, without for/else"
                        )
                    for value in values:
                        self.expansion_step()
                        self.bind(
                            statement.target, self.constant(value), env, namespace
                        )
                        written.update(
                            node.id
                            for node in ast.walk(statement.target)
                            if isinstance(node, ast.Name)
                        )
                        self.block(statement.body, env, definition, namespace, written)
                elif (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and self.resolve(statement.value.func, definition) is require
                ):
                    call = statement.value
                    if (
                        len(call.args) != 1
                        or len(call.keywords) != 1
                        or call.keywords[0].arg != "error"
                    ):
                        raise DefinitionError(
                            "Use require(condition, error='description')"
                        )
                    self.constraints.append(
                        (
                            self.expression(call.args[0], env, definition),
                            self.fixed(call.keywords[0].value, env, definition),
                        )
                    )
                else:
                    raise DefinitionError(
                        f"Unsupported {type(statement).__name__}; put private algorithms in a pure helper, or use bounded indexed steps"
                    )
            except (DefinitionError, FiniteValueError) as exc:
                raise DefinitionError(
                    f"{definition.filename}:{definition.line + statement.lineno - 1}: {exc}"
                ) from exc
        return returned

    def argument(self, node, env, definition):
        if isinstance(node, ast.Name) and isinstance(env.get(node.id), VariableFamily):
            return env[node.id]
        return self.expression(node, env, definition)

    def expand(self, definition, env, namespace):
        if any(definition is item for item in self.stack):
            raise DefinitionError("Recursive submodels need explicit bounded steps")
        self.stack.append(definition)
        try:
            result = self.block(definition.syntax.body, env, definition, namespace)
        finally:
            self.stack.pop()
        if result is None:
            raise DefinitionError("A definition must return an exposed variable")
        if not (
            isinstance(result, ast.Call)
            and isinstance(result.func, ast.Name)
            and result.func.id == "__read__"
        ):
            raise DefinitionError(
                "Return an exposed variable (use V to expose an expression)"
            )
        return result

    def expression_function(self, expression):
        class LowerBindings(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if not isinstance(node.func, ast.Name):
                    return node
                if node.func.id == "__sequence__":
                    return ast.Subscript(
                        ast.Tuple(node.args, ast.Load()), ast.Constant(1), ast.Load()
                    )
                if node.func.id in ("__private_binding__", "__unpack_binding__"):
                    key, value = node.args[:2]
                    if node.func.id == "__unpack_binding__":
                        value = ast.Call(
                            ast.Name("__unpack_exact__", ast.Load()),
                            [value, node.args[2]],
                            [],
                        )
                    return ast.Call(
                        ast.Name("__private__", ast.Load()),
                        [key, _lambda([], value)],
                        [],
                    )
                return node

        lowered = LowerBindings().visit(copy.deepcopy(expression))
        code = compile(
            ast.fix_missing_locations(
                ast.Expression(_lambda(["__read__", "__private__"], lowered))
            ),
            self.definition.filename,
            "eval",
        )
        return eval(code, self.constants)

    def function(self, expression):
        compute = self.expression_function(expression)
        return lambda trace: compute(trace.__getitem__, _PrivateBindings())

    def infer(self, expression, domains):
        if not _dependencies(expression):
            try:
                return Dom([self.evaluate(expression)])
            except FiniteValueError as exc:
                raise DefinitionError(str(exc)) from exc
            except (ArithmeticError, LookupError, TypeError, ValueError):
                # A structural result type can still be sound when every
                # execution fails. Preserve the failure in the equation.
                pass
        if isinstance(expression, ast.List):
            return Dom(list)
        if isinstance(expression, ast.Compare) and all(
            isinstance(operator, (ast.Is, ast.IsNot, ast.In, ast.NotIn))
            for operator in expression.ops
        ):
            return Dom(bool)
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return Dom(bool)
        if isinstance(expression, ast.JoinedStr):
            return Dom(str)
        if (
            isinstance(expression, ast.Subscript)
            and isinstance(expression.value, ast.Tuple)
            and isinstance(expression.slice, ast.Constant)
            and type(expression.slice.value) is int
        ):
            return self.infer(expression.value.elts[expression.slice.value], domains)
        if isinstance(expression, ast.IfExp):
            return Dom.union(
                self.infer(expression.body, domains),
                self.infer(expression.orelse, domains),
            )
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name) and expression.func.id in (
                "__private_binding__",
                "__sequence__",
            ):
                return self.infer(expression.args[1], domains)
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id == "__read__"
            ):
                return domains[expression.args[0].value]
            if not _dependencies(expression.func):
                func = self.evaluate(expression.func)
                if func in (str, bool):
                    return Dom(func)
        raise DefinitionError(
            "Cannot infer a sound domain within the enumeration limit; add V(..., domain=Dom(...)) or a family domain"
        )

    def nan_witness_pool(self, guard, parents):
        """Referenced NaNs and fresh peers for non-exhaustive witness search."""
        pending = []
        for name in parents:
            domain = self.domains.get(name)
            if domain is not None:
                values = domain.enumerated(self.inference_limit)
                if values is not None:
                    pending.extend(values)
        for node in ast.walk(guard):
            if isinstance(node, (ast.Name, ast.Attribute)):
                value = self.resolve(node)
                if value is not None:
                    pending.append(value)
        # `seen` is keyed on id(); `visited` keeps each object alive until the
        # walk ends, so a freed object's address cannot be reused by another
        # value and skip it (as ConfigurationCopier.originals does).
        seen, visited, atoms = set(), [], []
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            visited.append(value)
            if isinstance(value, (types.ModuleType, type)):
                continue
            if isinstance(value, types.FunctionType):
                pending.extend(
                    item
                    for name, item in value.__globals__.items()
                    if name != "__builtins__"
                )
                pending.extend((value.__defaults__, value.__kwdefaults__))
                pending.extend(cell.cell_contents for cell in value.__closure__ or ())
            elif isinstance(value, types.MethodType):
                pending.extend((value.__func__, value.__self__))
            elif isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, (tuple, list, set, frozenset)):
                pending.extend(value)
            elif isinstance(value, types.SimpleNamespace) or is_dataclass(value):
                pending.extend(_stored_values(value))
            else:
                for atom in _nan_atoms(value):
                    try:
                        _equal(atom, atom)
                    except FiniteValueError:
                        # Configuration can contain numeric shapes excluded
                        # from finite domains; they cannot be input witnesses.
                        continue
                    atoms.append(atom)
        # A shared fresh identity is needed when several causal NaNs agree
        # with each other but differ from every captured configuration NaN.
        representatives = []
        for atom in atoms:
            if not any(_equal(atom, old) for old in representatives):
                representatives.append(atom)
        return [*atoms, *(_fresh_nan(atom) for atom in representatives)]

    def reachable(self, guard):
        """Prove absence exhaustively, or presence with a valid domain witness.

        Witnesses establish an edge, never a value domain. If neither proof is
        available we refuse the definition rather than silently adding a false
        edge or removing a possible one.
        """
        parents = _dependencies(guard)
        count, choices, exhaustive = 1, [], True
        nan_pool = None
        constants = [n.value for n in ast.walk(guard) if isinstance(n, ast.Constant)]
        for name in parents:
            domain = self.domains.get(name)
            if domain is None:
                return None
            values = domain.enumerated(self.inference_limit)
            if values is None or count * len(values) > self.inference_limit:
                exhaustive = False
                values = []
                candidates = [None, False, True, 0, 1, -1, "", "value", *constants]
                if domain.values is not None:
                    size = domain.cardinality()
                    candidates.extend(
                        domain.values[i] for i in {0, size // 2, size - 1}
                    )
                try:
                    rng = random.Random(0)
                    candidates.extend(domain.sample(rng) for _ in range(4))
                except ValueError:
                    pass
                for value in candidates:
                    if domain.contains(value) and not any(
                        _equal(v, value) for v in values
                    ):
                        values.append(value)
            if any(_contains_nan(value) for value in values):
                # Bit-equal NaNs are not interchangeable under Python's
                # identity shortcut for container equality and membership.
                # Variants can prove presence, but never prove absence.
                exhaustive = False
                if nan_pool is None:
                    nan_pool = self.nan_witness_pool(guard, parents)
                originals = list(values)
                values = list(originals)
                for value in originals:
                    if not _contains_nan(value):
                        continue
                    for variant in _nan_variants(value, nan_pool, self.inference_limit):
                        if domain.contains(variant):
                            values.append(variant)
                        if len(values) >= self.inference_limit:
                            break
                    if len(values) >= self.inference_limit:
                        break
            count *= len(values)
            choices.append(values)
        predicate = self.function(guard)
        for i, combo in enumerate(itertools.product(*choices)):
            if i >= self.inference_limit:
                return None
            try:
                values = dict(zip(parents, combo))
                if nan_pool is not None:
                    # Trace inputs/overrides are copied separately. Preserve
                    # aliases inside each value without inventing sharing with
                    # another parent or compiled NumPy configuration.
                    values = {
                        name: copy.deepcopy(value) for name, value in values.items()
                    }
                if predicate(values):
                    return True
            except (ArithmeticError, LookupError, TypeError, ValueError):
                continue
        return False if exhaustive else None

    def specialize(self, expression):
        """Remove impossible branches only using finalized intervention domains."""
        compiler = self

        class Specialize(ast.NodeTransformer):
            def visit_IfExp(self, node):
                test = self.visit(node.test)
                possible = compiler.possible(test)
                if possible and all(bool(v) == bool(possible[0]) for v in possible):
                    chosen = self.visit(node.body if possible[0] else node.orelse)
                    if not _dependencies(test):
                        return chosen
                    # Reading the test is itself causal, even for a singleton
                    # domain. Preserve that read without its impossible branch.
                    return ast.Subscript(
                        ast.Tuple([test, chosen], ast.Load()),
                        ast.Constant(1),
                        ast.Load(),
                    )
                return ast.IfExp(test, self.visit(node.body), self.visit(node.orelse))

        result = Specialize().visit(copy.deepcopy(expression))
        self.check_expression(result, account=False)
        return ast.fix_missing_locations(result)

    def analyze_reads(self, expression):
        outcomes = {}
        for parent, guards in _read_guards(expression).items():
            outcomes[parent] = False
            for guard in guards:
                reachable = self.reachable(guard)
                if reachable is True:
                    outcomes[parent] = True
                    break
                if reachable is None:
                    outcomes[parent] = None
        return outcomes

    def infer_domain(self, expression, parents):
        choices, count = [], 1
        for parent in parents:
            values = self.domains[parent].enumerated(self.inference_limit)
            if values is None or any(_contains_nan(value) for value in values):
                count = self.inference_limit + 1
                break
            count *= len(values)
            choices.append(values)
            if count > self.inference_limit:
                break
        if count <= self.inference_limit:
            compute = self.function(expression)
            observed = []
            for combo in itertools.product(*choices):
                try:
                    value = compute(dict(zip(parents, combo)))
                except (ArithmeticError, LookupError, TypeError, ValueError):
                    # A partial function still raises on that actual execution.
                    continue
                try:
                    # Validate even the first value, before it becomes an
                    # exhaustive representative used to specialize consumers.
                    _equal(value, value)
                    if not any(_equal(v, value) for v in observed):
                        observed.append(value)
                except FiniteValueError as exc:
                    raise DefinitionError(str(exc)) from exc
            if observed:
                return Dom(observed)
        return self.infer(expression, self.domains)

    def finalize_graph(self):
        # Expansion and branch-domain unions are complete. Every domain entered
        # here is final: later inference only fills missing domains, never narrows
        # an explicit intervention domain or specializes a provisional branch.
        self.domains = {
            name: node.domain
            for name, node in self.nodes.items()
            if node.domain is not None
        }
        parents = {name: [] for name in self.nodes}
        proofs, errors = {}, {}
        examined_domains = {}
        while True:
            changed = False
            for name, node in self.nodes.items():
                if node.expression is None:
                    continue
                known = frozenset(
                    parent
                    for parent in _dependencies(node.expression)
                    if parent in self.domains
                )
                if examined_domains.get(name) == known:
                    continue
                node.expression = self.specialize(node.expression)
                examined_domains[name] = frozenset(
                    parent
                    for parent in _dependencies(node.expression)
                    if parent in self.domains
                )
                proofs[name] = self.analyze_reads(node.expression)
                parents[name] = [
                    parent
                    for parent in _dependencies(node.expression)
                    if proofs[name][parent] is not False
                ]
                if name not in self.domains and all(
                    parent in self.domains for parent in parents[name]
                ):
                    try:
                        self.domains[name] = self.infer_domain(
                            node.expression, parents[name]
                        )
                        changed = True
                    except DefinitionError as exc:
                        errors[name] = exc
            if not changed:
                break
        # Selector edges have now had a chance to disappear as their inferred
        # domains become available. A cycle still possible under interventions
        # must fail even if the observational equations would never traverse it.
        try:
            order = list(TopologicalSorter(parents).static_order())
        except CycleError as exc:
            raise DefinitionError(
                f"Causal graph must be acyclic: {exc.args[1]}"
            ) from exc
        self.mechanisms = {}
        for name in order:
            node = self.nodes[name]
            if name not in self.domains:
                raise DefinitionError(
                    f"Variable {name!r}: {errors.get(name, 'Cannot infer a sound domain; add an explicit domain')}"
                )
            for parent, proof in proofs.get(name, {}).items():
                if proof is None:
                    raise DefinitionError(
                        f"Variable {name!r}: cannot establish whether it reads {parent!r} on an allowed input. "
                        "Use a smaller finite gating domain or simplify the condition."
                    )
            compute = (
                None if node.expression is None else self.function(node.expression)
            )
            self.mechanisms[name] = CompiledEquation(parents[name], compute, node.lazy)

    def compile(self):
        env = {}
        signature = self.definition.signature
        for parameter in signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise DefinitionError("Model inputs must be named parameters")
            domain = parameter.annotation
            if isinstance(domain, str):
                domain = eval(
                    domain, {"__builtins__": builtins, **self.definition.environment}
                )
            exogenous = isinstance(domain, Exo)
            if exogenous:
                domain = domain.domain
            if isinstance(domain, FamilyDom):
                keys = list(domain.domains)
                try:
                    names = [_family_name(parameter.name, key) for key in keys]
                    if len(set(names)) != len(names):
                        raise DefinitionError(
                            "Family keys must have unique canonical names"
                        )
                except DefinitionError as exc:
                    raise DefinitionError(
                        f"{self.definition.filename}:{self.definition.line}: "
                        f"Input family {parameter.name!r}: {exc}"
                    ) from exc
                members = dict(zip(keys, names))
                self.families[parameter.name] = members
                env[parameter.name] = VariableFamily(members)
                domains = {members[k]: v for k, v in domain.domains.items()}
            elif isinstance(domain, Dom):
                env[parameter.name] = _read(parameter.name)
                domains = {parameter.name: domain}
            else:
                raise DefinitionError(
                    f"Input {parameter.name!r} needs a Dom, FamilyDom, or Exo annotation"
                )
            for name, member_domain in domains.items():
                self.add_node(name, None, copy.deepcopy(member_domain))
                self.inputs.append(name)
                if exogenous:
                    self.exogenous.append(name)
        self.return_variable = self.expand(self.definition, env, "").args[0].value
        missing = set(v for f in self.families.values() for v in f.values()) - set(
            self.nodes
        )
        if missing:
            raise DefinitionError(f"Family members are not defined: {sorted(missing)}")
        self.finalize_graph()
        self.validators = [
            (self.function(condition), message)
            for condition, message in self.constraints
        ]
        return self
