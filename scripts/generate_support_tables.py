"""Generate documentation from the schema and capability registry.

Named blocks contain component, engine, and featurizer tables. Pointer blocks
read docstrings (``doc``), attribute comments (``attrs``), values (``value``), or
function output (``call``) from objects under ``causalab``. Every block has
matching ``<!-- generated: begin ... -->`` and end markers on separate lines.
Keep a blank line between each marker and the block body.

Run ``uv run python scripts/generate_support_tables.py`` to update the pages.
``--check`` reports differences without writing. ``--root`` selects the tree
whose documentation is updated; the imported package supplies the source data.
Engine summaries require torch and are skipped when it is unavailable.

The method pages contain pointer blocks. Tests check that each generated body
matches its source and that repeated generation leaves current files unchanged."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import re
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from causalab.protocol import schema
from causalab.protocol.engine import CAPABILITIES as VERBS
from causalab.protocol.registry import (
    CAPABILITIES,
    ENGINES,
    Capability,
    _PREDICATE_MEANS,  # the words `unavailable_at_load` prints for a predicate
    _write_cell,  # the words the write-policy refusal prints
    engine_component_summary,
    render_component_tables,
    render_family_table,
)

# Repo root is one level up: scripts/generate_support_tables.py.
REPO_ROOT = Path(__file__).resolve().parents[1]

BEGIN = "<!-- generated: begin {name} -->"
END = "<!-- generated: end {name} -->"
#: A marker line, whole: the block or reader name (plain words joined by
#: hyphens), then the words a pointer block takes — a dotted name under
#: ``POINTER_ROOT``, member names or call arguments. A named block takes
#: none.
MARKER = re.compile(
    r"^<!-- generated: (begin|end) ([a-z][a-z0-9-]*)((?: [A-Za-z0-9_.-]+)*) -->$"
)

#: The package a pointer may name into; a pointer elsewhere is refused.
POINTER_ROOT = "causalab"

#: The hand-written method pages, every one a carrier of pointer blocks.
METHODS = Path("docs") / "methods"

#: Names a later change will emit; a marker carrying one is refused here rather
#: than silently left alone.
RESERVED: dict[str, str] = {
    "conditional-scope": "the ConditionalStep scope vocabulary (not generated yet)",
}

Renderer = Callable[[str], "str | None"]


@dataclasses.dataclass(frozen=True)
class Block:
    """One generated block: its marker name, the docs that carry it and the
    function that renders it from the committed body (``None`` when it cannot
    be rendered in this environment)."""

    name: str
    carriers: tuple[str, ...]
    render: Renderer


@dataclasses.dataclass(frozen=True)
class Found:
    """A marker pair in a file: the block or reader name, the words after it
    (empty for a named block) and the line indices of the two marker lines
    (``begin`` inclusive, ``end`` inclusive)."""

    name: str
    begin: int
    end: int
    args: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The marker's words, as written: ``name`` then ``args``."""
        return " ".join((self.name, *self.args))


class MarkerError(ValueError):
    """The markers in a file do not pair up."""


# --------------------------------------------------------------------------- #
# the renderers
# --------------------------------------------------------------------------- #


def _code(name: str) -> str:
    return f"`{name}`"


def _cell(text: str) -> str:
    """A table cell: the one character markdown cannot carry raw is escaped."""
    return text.replace("|", "\\|")


def _component_table(_committed: str) -> str:
    return render_component_tables()


def _family_table(_committed: str) -> str:
    return render_family_table()


def _engine_classes() -> dict[str, type] | None:
    """The shipped engine classes by name — or ``None`` without torch."""
    try:
        from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
        from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
    except ImportError as exc:  # pragma: no cover - the protocol-only install
        print(
            f"engine-summary: skipped, the engine classes did not import ({exc}); "
            "install the execution extras to render it",
            file=sys.stderr,
        )
        return None
    classes: dict[str, type] = {
        PytorchHooksEngine.name: PytorchHooksEngine,
        NnsightEngine.name: NnsightEngine,
    }
    if set(classes) != set(ENGINES):  # pragma: no cover - one vocabulary
        raise AssertionError(f"engines {sorted(classes)} are not {ENGINES}")
    return classes


def _engine_named(header_cell: str) -> str | None:
    """The engine an engine table's header cell names, if it names one."""
    for engine in ENGINES:
        if _code(engine) in header_cell:
            return engine
    return None


def _engine_summary(committed: str) -> str | None:
    """The committed engine table with its two derivable rows rewritten:
    ``capabilities`` (the engine's frozenset, in the §8 vocabulary's order)
    and ``components`` (``N of M`` plus the components it does not serve).
    Every other row — ``how``, ``serves alone``, ``install`` — is prose about
    the engine and passes through as committed."""
    classes = _engine_classes()
    if classes is None:
        return None
    lines = committed.splitlines()
    rows = [line for line in lines if line.startswith("|")]
    if not rows:
        raise MarkerError("engine-summary: the block carries no table")
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    order = [_engine_named(cell) for cell in header[1:]]
    if any(engine is None for engine in order) or set(order) != set(ENGINES):
        raise MarkerError(
            f"engine-summary: the table's header names {header[1:]}, not the "
            f"engines {ENGINES}"
        )
    capabilities = {
        engine: " ".join(_code(v) for v in VERBS if v in cls.capabilities)
        for engine, cls in classes.items()
    }
    components: dict[str, str] = {}
    for engine, cls in classes.items():
        missing = [_code(c) for c in schema.COMPONENTS if c not in cls.components]
        summary = engine_component_summary(engine)
        if not missing:
            components[engine] = f"{summary}; all components"
        else:
            listed = ", ".join(missing[:-1]) + f" and {missing[-1]}"
            components[engine] = f"{summary}; unsupported: {listed}"
    out: list[str] = []
    for line in lines:
        first = line.strip("|").split("|")[0].strip() if line.startswith("|") else ""
        if first == "capabilities":
            cells = " | ".join(capabilities[e] for e in order if e is not None)
            out.append(f"| capabilities | {cells} |")
        elif first == "components":
            cells = " | ".join(components[e] for e in order if e is not None)
            out.append(f"| components | {cells} |")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


AVAILABILITY_HEADER = (
    "| component | layers | stream | engines | write policy | requires "
    "| expert face | aliases |\n|---|---|---|---|---|---|---|---|"
)


def _layers_cell(row: Capability) -> str:
    if row.component in schema.LAYERLESS_COMPONENTS:
        return "layer-less; omit `layers`"
    return "a `layers` band"


def _stream_cell(row: Capability) -> str:
    if row.stream is not None:
        return f"{_code(row.stream)} layers only"
    if row.component in schema.LAYERLESS_COMPONENTS:
        return "layer-less"
    return "either"


def _engines_cell(row: Capability) -> str:
    if set(row.reads) == set(ENGINES):
        return "both"
    served = [_code(e) for e in ENGINES if e in row.reads]
    return " ".join(served)


def _requires_cell(row: Capability) -> str:
    if not row.requires:
        return "nothing"
    parts = [
        f"{_code(p)}: {_PREDICATE_MEANS[p]}"
        for p in sorted(row.requires)  # the row's set has no order of its own
    ]
    return "; ".join(parts)


def _expert_cell(row: Capability) -> str:
    if row.expert_selection:
        engines = " ".join(_code(e) for e in ENGINES if e in row.expert_selection)
        return f"`expert:` served by {engines}"
    return "none; `expert` is invalid at load"


def _aliases_cell(row: Capability) -> str:
    if not row.aliases:
        return "none"
    spelled = ", ".join(_code(a) for a in row.aliases)
    return f"{spelled} (retired under protocol version {row.deprecated_in})"


def render_availability_table() -> str:
    """Spec §2.4's per-component availability table, one row per entry of
    ``COMPONENTS`` in the vocabulary's order, every cell a field of the
    component's ``Capability`` row."""
    lines = [AVAILABILITY_HEADER]
    for component in schema.COMPONENTS:
        row = CAPABILITIES[component]
        cells = (
            _code(component),
            _layers_cell(row),
            _stream_cell(row),
            _engines_cell(row),
            _write_cell(row),
            _requires_cell(row),
            _expert_cell(row),
            _aliases_cell(row),
        )
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def _availability_table(_committed: str) -> str:
    return render_availability_table()


def _featurizer_kind_table(_committed: str) -> str:
    return schema.render_featurizer_kind_table()


def _gate_map_table(_committed: str) -> str:
    return schema.render_gate_map_table()


RUNNING = "docs/running_experiments.md"
QWEN36 = "docs/qwen36_35b_a3b.md"
SPEC = "docs/intervention_protocol.md"
CODEBASE = "docs/CODEBASE.md"

#: The closed set of blocks, by marker name. A marker with any other name is
#: refused; a carrier missing one of its blocks is refused too.
BLOCKS: dict[str, Block] = {
    block.name: block
    for block in (
        Block("component-table", (QWEN36,), _component_table),
        Block("family-table", (RUNNING,), _family_table),
        Block("engine-summary", (RUNNING, CODEBASE), _engine_summary),
        Block("availability-table", (SPEC,), _availability_table),
        Block("featurizer-kind-table", (SPEC,), _featurizer_kind_table),
        Block("gate-map-table", (SPEC,), _gate_map_table),
    )
}

#: Every doc that carries a block, in path order.
CARRIERS: tuple[str, ...] = tuple(
    sorted({carrier for block in BLOCKS.values() for carrier in block.carriers})
)


# --------------------------------------------------------------------------- #
# the readers: autodoc by pointer
# --------------------------------------------------------------------------- #

#: An autorefs cross-reference, ``[`label`][]`` or ``[`label`][full.path]``.
_AUTOREF = re.compile(r"\[(`[^`]+`)\]\[[\w.]*\]")


def md(text: str) -> str:
    """Docstring text as page markdown: a cross-reference becomes its code
    span, since the method pages are read on GitHub too, where an autorefs
    link shows as raw brackets; double backticks become single ones."""
    return _AUTOREF.sub(r"\1", text).replace("``", "`")


def attribute_docs(obj: type | ModuleType) -> dict[str, str]:
    """The ``#:`` attribute docs of a class body or a module, by name: every
    contiguous run of ``#:`` lines ending on the line before an assignment,
    joined into one paragraph. A name with no such run is absent. Read off
    the source: the Sphinx convention leaves nothing at run time."""
    source = textwrap.dedent(inspect.getsource(obj))
    lines = source.splitlines()
    tree = ast.parse(source)
    first = tree.body[0] if tree.body else None
    body = first.body if isinstance(first, ast.ClassDef) else tree.body
    out: dict[str, str] = {}
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
        else:
            continue
        run: list[str] = []
        i = node.lineno - 2
        while i >= 0 and lines[i].lstrip().startswith("#:"):
            run.append(lines[i].lstrip()[2:].strip())
            i -= 1
        if run:
            out[name] = " ".join(reversed(run))
    return out


def resolve(pointer: str) -> tuple[Any, Any, str]:
    """The object a dotted pointer names, the object it hangs on (``None``
    for a module) and its last name. The longest importable prefix is the
    module; the rest are attributes. A pointer outside ``POINTER_ROOT``,
    a module that does not import or a name the parent lacks is refused."""
    parts = pointer.split(".")
    if (
        parts[0] != POINTER_ROOT
        or len(parts) < 2
        or not all(part.isidentifier() for part in parts)
    ):
        raise MarkerError(
            f"pointer {pointer!r}: a dotted name under {POINTER_ROOT!r} "
            "(causalab.protocol.schema.FeaturizerSpec, say)"
        )
    obj: Any = None
    cut = len(parts)
    while cut > 0:
        module_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if module_name != missing and not module_name.startswith(missing + "."):
                raise  # a dependency the module itself needs is missing
            cut -= 1  # the missing module is on the pointer's own path
    else:  # pragma: no cover - POINTER_ROOT is a package
        raise MarkerError(f"pointer {pointer!r}: nothing under it imports")
    parent: Any = None
    name = parts[-1]
    for name in parts[cut:]:
        parent = obj
        try:
            obj = getattr(obj, name)
        except AttributeError:
            what = parent.__name__ if hasattr(parent, "__name__") else repr(parent)
            raise MarkerError(f"pointer {pointer!r}: {what} has no {name!r}") from None
    return obj, parent, name


def _pointer(reader: str, args: tuple[str, ...]) -> str:
    if not args:
        raise MarkerError(f"{reader}: takes a pointer under {POINTER_ROOT!r}")
    return args[0]


def _read_doc(args: tuple[str, ...]) -> str:
    pointer = _pointer("doc", args)
    if len(args) > 1:
        raise MarkerError(f"doc {pointer}: takes the pointer alone, not {args[1:]}")
    obj, parent, name = resolve(pointer)
    if inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj):
        text = inspect.getdoc(obj)
    else:
        text = attribute_docs(parent).get(name) if parent is not None else None
    if not text:
        raise MarkerError(f"doc {pointer}: carries no docstring or `#:` attribute doc")
    return md(text).rstrip() + "\n"


def _read_attrs(args: tuple[str, ...]) -> str:
    pointer = _pointer("attrs", args)
    obj, _, _ = resolve(pointer)
    if not (inspect.isclass(obj) or inspect.ismodule(obj)):
        raise MarkerError(f"attrs {pointer}: a class or a module has attribute docs")
    docs = attribute_docs(obj)
    names = list(args[1:]) or list(docs)
    missing = [n for n in names if n not in docs]
    if missing:
        raise MarkerError(
            f"attrs {pointer}: no `#:` doc for {missing}; documented: {sorted(docs)}"
        )
    return "\n".join(f"- **`{n}`**: {md(docs[n]).strip()}" for n in names) + "\n"


def _scalar(value: Any) -> str:
    """One value as a table cell or a list member."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return _cell(value) if value else "none"
    if isinstance(value, (frozenset, set)):
        return ", ".join(f"`{v}`" for v in sorted(value, key=str)) or "none"
    if isinstance(value, (tuple, list)):
        return ", ".join(f"`{v}`" for v in value) or "none"
    return f"`{value!r}`"


def _is_record(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _read_value(args: tuple[str, ...]) -> str:
    pointer = _pointer("value", args)
    obj, _, _ = resolve(pointer)
    keys = list(args[1:])
    if isinstance(obj, Mapping):
        missing = [k for k in keys if k not in obj]
        if missing:
            raise MarkerError(f"value {pointer}: no key {missing}; has {list(obj)}")
        items = [(k, obj[k]) for k in keys] if keys else list(obj.items())
        if items and all(_is_record(v) for _, v in items):
            fields = [f.name for f in dataclasses.fields(items[0][1])]
            lines = [
                "| key | " + " | ".join(f"`{f}`" for f in fields) + " |",
                "|---|" + "---|" * len(fields),
            ]
            for key, record in items:
                cells = " | ".join(_scalar(getattr(record, f)) for f in fields)
                lines.append(f"| `{key}` | {cells} |")
        else:
            lines = ["| key | value |", "|---|---|"]
            lines += [f"| `{k}` | {_scalar(v)} |" for k, v in items]
        return "\n".join(lines) + "\n"
    if keys:
        raise MarkerError(f"value {pointer}: keys select rows of a mapping only")
    if _is_record(obj):
        lines = ["| field | value |", "|---|---|"]
        for field in dataclasses.fields(obj):
            lines.append(f"| `{field.name}` | {_scalar(getattr(obj, field.name))} |")
        return "\n".join(lines) + "\n"
    return _scalar(obj) + "\n"


def _read_call(args: tuple[str, ...]) -> str:
    pointer = _pointer("call", args)
    obj, _, _ = resolve(pointer)
    if not callable(obj):
        raise MarkerError(f"call {pointer}: not callable")
    result = obj(*args[1:])
    if not isinstance(result, str) or not result.strip():
        raise MarkerError(
            f"call {pointer}: returned {type(result).__name__}, not markdown"
        )
    return result


Reader = Callable[[tuple[str, ...]], str]

#: The readers, by marker name. Each takes the marker's words after its name.
READERS: dict[str, Reader] = {
    "doc": _read_doc,
    "attrs": _read_attrs,
    "value": _read_value,
    "call": _read_call,
}


def carriers(root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Every doc under ``root`` the tool runs over: the named blocks'
    carriers, then every method page (``docs/methods/*.md``), in path order."""
    pages = sorted(str(p.relative_to(root)) for p in (root / METHODS).glob("*.md"))
    return (*CARRIERS, *pages)


# --------------------------------------------------------------------------- #
# the markers
# --------------------------------------------------------------------------- #


def find_blocks(text: str, where: str = "<text>") -> list[Found]:
    """Every marker pair in ``text``, in order. Refuses a ``begin`` inside a
    block, an ``end`` with no ``begin`` or other words than its ``begin``, an
    unterminated block, a marker written twice, a reserved name, a name no
    block or reader has, a named block given words and a reader given none."""
    found: list[Found] = []
    open_: tuple[Found, int] | None = None  # the open begin, its line index
    for index, line in enumerate(text.split("\n")):
        match = MARKER.match(line)
        if match is None:
            continue
        kind, name = match.group(1), match.group(2)
        args = tuple(match.group(3).split())
        label = " ".join((name, *args))
        if kind == "begin":
            if open_ is not None:
                raise MarkerError(
                    f"{where}:{index + 1}: begin {label!r} inside the open block "
                    f"{open_[0].label!r} (line {open_[1] + 1}); blocks do not nest"
                )
            if name in RESERVED:
                raise MarkerError(
                    f"{where}:{index + 1}: block {name!r} is reserved for "
                    f"{RESERVED[name]} and has no renderer here"
                )
            if name in READERS:
                if not args:
                    raise MarkerError(
                        f"{where}:{index + 1}: {name!r} takes a pointer under "
                        f"{POINTER_ROOT!r}"
                    )
            elif name in BLOCKS:
                if args:
                    raise MarkerError(
                        f"{where}:{index + 1}: block {name!r} takes no words; "
                        f"{args} given"
                    )
            else:
                raise MarkerError(
                    f"{where}:{index + 1}: unknown block {name!r}; the blocks are "
                    f"{sorted(BLOCKS)} and the readers {sorted(READERS)}"
                )
            if any(f.label == label for f in found):
                raise MarkerError(f"{where}:{index + 1}: block {label!r} appears twice")
            open_ = (Found(name, index, index, args), index)
        else:
            if open_ is None or open_[0].label != label:
                raise MarkerError(
                    f"{where}:{index + 1}: end {label!r} closes "
                    f"{'nothing' if open_ is None else repr(open_[0].label)}; an "
                    "end marker repeats its begin marker's words"
                )
            found.append(dataclasses.replace(open_[0], end=index))
            open_ = None
    if open_ is not None:
        raise MarkerError(
            f"{where}:{open_[1] + 1}: block {open_[0].label!r} has no end"
        )
    return found


def body_of(text: str, found: Found) -> str:
    """The generated text between ``found``'s markers, without the blank line
    inside each marker; ends with a newline."""
    lines = text.split("\n")[found.begin + 1 : found.end]
    if len(lines) < 3 or lines[0] != "" or lines[-1] != "":
        raise MarkerError(
            f"block {found.label!r}: a blank line goes inside each marker and "
            "the body is not empty"
        )
    return "\n".join(lines[1:-1]) + "\n"


def _first_difference(old: str, new: str) -> str:
    for number, (a, b) in enumerate(zip(old.splitlines(), new.splitlines()), 1):
        if a != b:
            return f"body line {number}:\n  committed: {a}\n  rendered:  {b}"
    old_n, new_n = len(old.splitlines()), len(new.splitlines())
    return f"committed body has {old_n} lines, the rendering {new_n}"


def rewrite(text: str, where: str) -> tuple[str, list[str]]:
    """``text`` with every block re-rendered — a named block by its renderer,
    a pointer block by its reader — and a report line per block whose
    committed body differs from its rendering (with the first differing
    line). A block whose renderer cannot run here is left as committed."""
    lines = text.split("\n")
    changed: list[str] = []
    for found in reversed(find_blocks(text, where)):  # later blocks first: indices hold
        if found.name in READERS:
            rendered: str | None = READERS[found.name](found.args)
        else:
            rendered = BLOCKS[found.name].render(body_of(text, found))
        if rendered is None:
            continue
        if not rendered.endswith("\n"):
            rendered += "\n"
        if "```" in rendered:
            raise MarkerError(f"block {found.label!r}: a rendering carries a fence")
        committed = body_of(text, found)
        if committed != rendered:
            changed.append(
                f"{where}: block {found.label!r} differs — "
                f"{_first_difference(committed, rendered)}"
            )
        lines[found.begin + 1 : found.end] = ["", *rendered[:-1].split("\n"), ""]
    return "\n".join(lines), changed


def expected_blocks(carrier: str) -> list[str]:
    """The block names ``carrier`` has to hold, in ``BLOCKS`` order."""
    return [name for name, block in BLOCKS.items() if carrier in block.carriers]


def check_carrier(text: str, carrier: str) -> list[Found]:
    """``find_blocks`` plus the completeness half: every named block that
    names ``carrier`` is present and no other named block is. Pointer blocks
    are the page's own choice and pass."""
    found = find_blocks(text, carrier)
    names = [f.name for f in found if f.name not in READERS]
    if sorted(names) != sorted(expected_blocks(carrier)):
        raise MarkerError(
            f"{carrier}: carries blocks {names}, expected {expected_blocks(carrier)}"
        )
    return found


def run(root: Path, *, check: bool, over: Iterable[str] | None = None) -> int:
    """Rewrite (or, under ``check``, compare) every carrier under ``root`` —
    ``carriers`` of it unless ``over`` names them. Returns the exit
    status: 0 clean, 1 something differs under ``check``."""
    differing = 0
    for carrier in carriers(root) if over is None else over:
        path = root / carrier
        if not path.exists():
            raise MarkerError(f"{carrier}: not under {root}")
        text = path.read_text()
        check_carrier(text, carrier)
        new, changed = rewrite(text, carrier)
        for line in changed:
            print(line)
        if new == text:
            print(f"{carrier}: current")
            continue
        differing += 1
        if check:
            print(f"{carrier}: would change")
        else:
            path.write_text(new)
            print(f"{carrier}: rewritten")
    return 1 if (check and differing) else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="tree whose docs/ to rewrite (default: the repo)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any block would change",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run(args.root.resolve(), check=args.check)
    except MarkerError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
