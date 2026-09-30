"""Python docstrings follow the format in docs/STYLE_GUIDE.md.

Sections are Google style, since `mkdocs.yml` parses every docstring that way.
Cross-references are autorefs links: ``[`Name`][]`` for a name in scope,
``[`Name`][full.path]`` otherwise. The docs build reads neither Sphinx roles
nor NumPy section underlines, so both are refused wherever they appear. The
resolution check mirrors mkdocstrings' ``scoped_crossrefs``, so a link that
would not resolve on the site fails here, without building the site. The
site renders only the packages in ``extra.api_reference``, so a rendered
docstring may link only into those packages.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

import griffe
import pytest

from tests._helpers.mkdocs import api_reference_packages

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

#: A Sphinx cross-reference role: a role name such as ``class`` between colons,
#: then a backtick.
ROLE = re.compile(r":(?:class|func|meth|mod|data|attr|obj|exc|const):`")

#: A NumPy section: a header line over a dashed underline. The header must be a
#: section name NumPy defines, so a Markdown setext heading such as
#: ``Resolution`` passes; the indent is optional, so a module docstring at
#: column 0 is checked too.
NUMPY_SECTION = re.compile(
    r"^[ \t]*(?:Parameters|Other Parameters|Returns|Yields|Receives|Raises|Warns"
    r"|Warnings|Attributes|Methods|See Also|Notes|References|Examples)[ \t]*\n"
    r"[ \t]*-{3,}[ \t]*$",
    re.M,
)

#: A reStructuredText directive such as ``.. deprecated::``, which the docs
#: build shows as literal text.
RST_DIRECTIVE = re.compile(r"^[ \t]*\.\. [a-z][a-z-]*::", re.M)

#: An autorefs link: the code-span label, then the identifier (empty = the label).
AUTOREF = re.compile(r"\[`([^`]+)`\]\[([\w.]*)\]")

#: A double-backtick literal, which can quote the link syntax as an example.
LITERAL = re.compile(r"``.+?``")


def _links(text: str) -> list[tuple[str, str]]:
    """The ``(label, identifier)`` of each autorefs link outside a literal."""
    return AUTOREF.findall(LITERAL.sub("", text))


def _python_files() -> list[Path]:
    """Every tracked Python file."""
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO / line for line in out.stdout.splitlines()]


def _prose(path: Path) -> str:
    """The comments and statement-level strings (docstrings) of ``path``."""
    source = path.read_text()
    docs = {
        (node.value.lineno, node.value.col_offset)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    return "\n".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
        or (tok.type == tokenize.STRING and tok.start in docs)
    )


def _load_script(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _objects(root: griffe.Object) -> Iterator[griffe.Object]:
    """Every non-alias object under ``root``, private ones included."""
    stack = [root]
    while stack:
        obj = stack.pop()
        yield obj
        stack.extend(m for m in obj.members.values() if not m.is_alias)  # type: ignore[misc]


def _is_public(obj: griffe.Object) -> bool:
    path = obj.path.split(".")
    return not any(part.startswith("_") for part in path)


def _in_packages(path: str, packages: list[str]) -> bool:
    """Whether the dotted ``path`` is one of ``packages`` or inside one."""
    return any(path == p or path.startswith(p + ".") for p in packages)


def _expand(identifier: str, owner: griffe.Object) -> str:
    """What ``scoped_crossrefs`` makes of ``identifier`` in ``owner``'s docstring.

    Mirrors ``AutorefsHook.expand_identifier`` in mkdocstrings-python 2.0.8: the
    first segment resolves in the owner's scope, the rest is appended, and a
    first segment that does not resolve leaves the identifier as written.
    """
    first, _, rest = identifier.partition(".")
    try:
        first = owner.resolve(first)
    except Exception:
        return identifier
    return f"{first}.{rest}" if rest else first


@pytest.fixture(scope="module")
def package() -> griffe.Module:
    extension = _load_script("griffe_doc_comments").DocComments()
    module = griffe.load(
        "causalab",
        search_paths=[REPO],
        docstring_parser="google",
        allow_inspection=False,
        extensions=griffe.load_extensions(extension),
    )
    assert isinstance(module, griffe.Module)
    return module


@pytest.fixture(scope="module")
def public_paths(package: griffe.Module) -> set[str]:
    return {obj.path for obj in _objects(package) if _is_public(obj)}


def test_no_sphinx_role_in_python() -> None:
    offenders = [
        f"{path.relative_to(REPO)}:{text[: m.start()].count(chr(10)) + 1}"
        for path in _python_files()
        for text in [path.read_text()]
        for m in ROLE.finditer(text)
    ]
    assert offenders == [], "Sphinx roles; write an autorefs link instead"


def test_no_numpy_section_in_python() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in _python_files()
        if NUMPY_SECTION.search(path.read_text())
    ]
    assert offenders == [], "NumPy-style sections; write Google-style sections"


def test_the_numpy_pattern_checks_section_names_at_any_indent() -> None:
    """Sanity check on `NUMPY_SECTION`:
    a module docstring's section at column 0 is caught, and an indented
    setext heading with a name NumPy does not define is left alone."""
    assert NUMPY_SECTION.search(
        '"""Load it.\n\nParameters\n----------\npath : str\n"""'
    )
    assert NUMPY_SECTION.search("    Returns\n    -------\n    int\n")
    assert not NUMPY_SECTION.search(
        "    Resolution\n    ----------\n    Globs in order.\n"
    )


def test_no_rst_directive_in_python() -> None:
    offenders = [
        f"{path.relative_to(REPO)}:{text[: m.start()].count(chr(10)) + 1}"
        for path in _python_files()
        for text in [path.read_text()]
        for m in RST_DIRECTIVE.finditer(text)
    ]
    assert offenders == [], (
        "reST directives; write a Google section such as Deprecated:"
    )


def test_every_package_link_resolves_to_a_public_object(
    package: griffe.Module, public_paths: set[str]
) -> None:
    broken = [
        f"{obj.path}: [`{label}`][{ident}] -> {target}"
        for obj in _objects(package)
        if obj.docstring is not None
        for label, ident in _links(obj.docstring.value)
        for target in [_expand(ident or label, obj)]
        if target not in public_paths
    ]
    assert broken == []


def test_every_rendered_link_targets_a_rendered_package(
    package: griffe.Module,
) -> None:
    """A link from a rendered docstring to an object the reference does not
    render dangles on the site; write the target as inline code instead."""
    packages = api_reference_packages()
    broken = [
        f"{obj.path}: [`{label}`][{ident}] -> {target}"
        for obj in _objects(package)
        if obj.docstring is not None
        and _is_public(obj)
        and _in_packages(obj.path, packages)
        for label, ident in _links(obj.docstring.value)
        for target in [_expand(ident or label, obj)]
        if not _in_packages(target, packages)
    ]
    assert broken == []


def test_the_package_filter_matches_whole_segments() -> None:
    """Sanity check: a sibling that shares a name prefix is outside the package."""
    packages = ["causalab.protocol.schema"]
    assert _in_packages("causalab.protocol.schema", packages)
    assert _in_packages("causalab.protocol.schema.types.Site", packages)
    assert not _in_packages("causalab.protocol.schemas", packages)
    assert not _in_packages("causalab.protocol", packages)


def test_every_link_outside_the_package_names_a_full_public_path(
    public_paths: set[str],
) -> None:
    """Tests and scripts are not on the site, so their links have no scope to
    resolve in; each one must carry a full path that exists."""
    broken = [
        f"{path.relative_to(REPO)}: [`{label}`][{ident}]"
        for path in _python_files()
        if path.relative_to(REPO).parts[0] != "causalab"
        for label, ident in _links(_prose(path))
        if (ident or label) not in public_paths
    ]
    assert broken == []


def test_the_resolution_check_catches_a_broken_link(
    package: griffe.Module, public_paths: set[str]
) -> None:
    """Sanity check on the mirror: a name out of scope stays unresolved, and
    the same name in scope resolves to its module path."""
    errors = package["protocol.rules.errors"]
    assert _expand("Rule", errors) == "causalab.protocol.rules.errors.Rule"
    assert _expand("Rule", errors) in public_paths
    assert _expand("NoSuchName", errors) not in public_paths
