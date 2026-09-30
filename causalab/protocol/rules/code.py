"""Check the signature and declared reads of referenced functions.

The checks inspect the function's AST for literal file and environment reads.
Calls routed through dynamic expressions require separate review."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from causalab.protocol.identity import ResolvedCode

__all__ = [
    "READER_CALLS",
    "function_def",
    "signature_problems",
    "undeclared_reads",
]

#: Attribute calls whose first positional argument, when it is a string
#: literal, is a path being read. Deliberately short: every entry is a call
#: whose literal-string first argument is a filename in every library that
#: spells it this way, so the check cannot fire on valid work that reads
#: nothing. A read routed through a variable is not in this set and is not
#: refused — see the module docstring.
READER_CALLS: frozenset[str] = frozenset(
    {
        "load",
        "loadtxt",
        "read_bytes",
        "read_csv",
        "read_json",
        "read_parquet",
        "read_text",
        "open",
    }
)


def function_def(
    resolved: ResolvedCode,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The ``def`` the locator names, or ``None`` when the attribute is not a
    literal function in the module's source.

    ``None`` is the honest answer for a C function (``torch.relu``), an
    attribute built by a factory at import time, or a re-export. It is not a
    refusal: it is the boundary of what a static read can see, and the checks
    that need a signature simply do not run.
    """
    if not resolved.attr:
        return None
    try:
        tree = ast.parse(resolved.path.read_bytes(), filename=str(resolved.path))
    except (SyntaxError, ValueError):
        return None
    body: Iterable[ast.stmt] = tree.body
    for index, name in enumerate(resolved.attr):
        last = index == len(resolved.attr) - 1
        match = next(
            (
                node
                for node in body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and node.name == name
            ),
            None,
        )
        if match is None:
            return None
        if last:
            return match if not isinstance(match, ast.ClassDef) else None
        if not isinstance(match, ast.ClassDef):
            return None
        body = match.body
    return None


def signature_problems(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    args: Mapping[str, Any],
    supplied: Sequence[str] = (),
) -> list[str]:
    """Why the declared ``args`` do not fit the function's signature.

    The first positional parameter is the tensor the mechanism writes, so it
    is never declared. ``supplied`` names the keywords the runtime provides
    (``row_roles``, ``data_inputs``) because the declaration asked for them.
    A ``**kwargs`` function accepts any name, so only the *missing* half of
    the check runs against it.
    """
    positional = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    keyword_only = [a.arg for a in fn.args.kwonlyargs]
    tensor = positional[:1]
    accepts_any = fn.args.kwarg is not None
    known = set(positional[1:]) | set(keyword_only)

    problems: list[str] = []
    if not tensor:
        problems.append(
            f"{fn.name}() takes no positional parameter — the mechanism passes "
            "the feature slice as the first argument"
        )
    if not accepts_any:
        for name in sorted(set(args) - known):
            problems.append(
                f"declared argument {name!r} is not a parameter of {fn.name}() "
                f"(it takes {sorted(known) or 'no keyword arguments'})"
            )
        for name in sorted(set(supplied) - known):
            problems.append(
                f"the declaration asks the runtime for {name!r}, but {fn.name}() "
                "has no such parameter"
            )

    required = _required_parameters(fn, skip=len(tensor))
    for name in sorted(required - set(args) - set(supplied)):
        problems.append(
            f"{fn.name}() requires {name!r} and the declaration does not give it — "
            "an argument that is not declared is not in the digest"
        )
    return problems


def _required_parameters(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, *, skip: int
) -> set[str]:
    positional = [*fn.args.posonlyargs, *fn.args.args][skip:]
    defaults = fn.args.defaults
    without_default = (
        positional[: len(positional) - len(defaults)] if defaults else positional
    )
    required = {a.arg for a in without_default}
    required |= {
        a.arg
        for a, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
        if default is None
    }
    return required


def undeclared_reads(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    env_inputs: Iterable[str],
    data_inputs: Iterable[str],
) -> list[str]:
    """Statically detectable reads the declaration does not cover.

    Two kinds, both literal: an environment variable named by a string
    constant (``os.getenv("X")``, ``os.environ["X"]``,
    ``os.environ.get("X")``) and a file named by a string constant passed
    first to one of [`READER_CALLS`][] or to ``open``. Anything routed
    through a variable is invisible here and is not refused — see the module
    docstring.
    """
    allowed_env = set(env_inputs)
    allowed_files = set(data_inputs)
    problems: list[str] = []

    for node in ast.walk(fn):
        name = _env_name(node)
        if name is not None and name not in allowed_env:
            problems.append(
                f"reads environment variable {name!r}, which the declaration's "
                "'env_inputs' does not allow"
            )
        path = _read_path(node)
        if path is not None and path not in allowed_files:
            problems.append(
                f"reads the file {path!r}, which the declaration's 'data_inputs' "
                "does not name"
            )
    return sorted(dict.fromkeys(problems))


def _literal_str(node: ast.expr | None) -> str | None:
    return (
        node.value
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        else None
    )


def _env_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and _attribute_tail(node.value) == "environ":
        return _literal_str(node.slice)
    if isinstance(node, ast.Call):
        tail = _attribute_tail(node.func)
        if tail == "getenv" and node.args:
            return _literal_str(node.args[0])
        if (
            tail == "get"
            and isinstance(node.func, ast.Attribute)
            and _attribute_tail(node.func.value) == "environ"
            and node.args
        ):
            return _literal_str(node.args[0])
    return None


def _read_path(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open" and node.args:
        return _literal_str(node.args[0])
    if not isinstance(func, ast.Attribute) or func.attr not in READER_CALLS:
        return None
    # ``Path("scale.json").read_text()`` — the literal is on the receiver, and
    # the reading call itself takes no arguments at all
    receiver = func.value
    if (
        isinstance(receiver, ast.Call)
        and _attribute_tail(receiver.func) in ("Path", "PurePath", "PosixPath")
        and receiver.args
    ):
        literal = _literal_str(receiver.args[0])
        if literal is not None:
            return literal
    return _literal_str(node.args[0]) if node.args else None


def _attribute_tail(node: ast.AST) -> str | None:
    """``os.environ`` and a bare ``environ`` both answer ``"environ"``."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None
