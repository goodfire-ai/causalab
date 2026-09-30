"""Add the README, the demos and the API reference to the mkdocs build.

`mkdocs` reads one directory, `docs/`. The rest of what a reader needs lives
elsewhere in the tree: the `README` at the root, the demos with their figures
and documents, and the package whose docstrings are the API reference. This
script, run by the `mkdocs-gen-files` plugin during `mkdocs build`, adds those
to the site without moving or copying anything in the repository:

* `README.md` becomes the site's `index.md`;
* every file under `demos/` is mirrored at the same path, so a demo's relative
  links to its own documents and figures keep working;
* `demos/SUMMARY.md` lists the tutorials in the subsections and order that
  `extra.tutorials` in `mkdocs.yml` sets, each with a number;
* each page that `extra.section_index` names becomes a table of its nav
  section's pages, with the opening sentence of each as its purpose;
* one page per module in the packages that `extra.api_reference` in
  `mkdocs.yml` lists holds a `::: module.path` directive that `mkdocstrings`
  expands, and `reference/SUMMARY.md` lists them.

`mkdocs-literate-nav` reads the two `SUMMARY.md` files as the navigation of
the Tutorials section and the API reference tab.

Pages are copied verbatim. `scripts/mkdocs_links.py`, an mkdocs hook, rewrites
their links afterwards, for these pages and for the ones under `docs/` alike.

The helpers are plain functions over paths so `tests/docs/test_mkdocs.py` can
exercise them without a build. Only `main` touches `mkdocs_gen_files`.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "causalab"

#: Directories under the package that hold no documented module.
SKIPPED_DIRS = frozenset({"__pycache__"})


def is_package_dir(directory: Path, package_root: Path) -> bool:
    """Whether every directory from ``package_root`` down to ``directory`` is a package.

    griffe, like the import system's regular-package rules, documents a
    directory as a package only when it has an ``__init__.py``. A directory
    without one is skipped with everything under it, so the build does not
    fail on a module it cannot collect.
    """
    current = directory
    while current != package_root.parent:
        if not (current / "__init__.py").exists():
            return False
        current = current.parent
    return True


def in_packages(parts: tuple[str, ...], packages: Sequence[str] | None) -> bool:
    """Whether the module ``parts`` is one of ``packages`` or inside one.

    ``None`` selects every module.
    """
    if packages is None:
        return True
    dotted = ".".join(parts)
    return any(dotted == p or dotted.startswith(p + ".") for p in packages)


def module_pages(
    package_root: Path, packages: Sequence[str] | None = None
) -> list[tuple[tuple[str, ...], Path, PurePosixPath]]:
    """Every module under ``package_root`` with the reference page it gets.

    Returns ``(dotted parts, source path, page path)`` triples in sorted order.
    A package's ``__init__.py`` becomes its directory's ``index.md``, which
    `mkdocs-section-index` shows as the section's own page. Private modules
    (``_x.py``) are skipped: the reference is the public surface. With
    ``packages``, only the modules in those dotted packages get a page.
    """
    out: list[tuple[tuple[str, ...], Path, PurePosixPath]] = []
    for source in sorted(package_root.rglob("*.py")):
        rel = source.relative_to(package_root.parent)
        if SKIPPED_DIRS & set(rel.parts) or not is_package_dir(
            source.parent, package_root
        ):
            continue
        parts = rel.with_suffix("").parts
        if parts[-1] == "__init__":
            if source.stat().st_size == 0:
                continue  # nothing to render; the section still lists its modules
            parts = parts[:-1]
            page = PurePosixPath("reference", *parts, "index.md")
        elif parts[-1].startswith("_"):
            continue
        else:
            page = PurePosixPath("reference", *parts).with_suffix(".md")
        if in_packages(parts, packages):
            out.append((parts, source, page))
    return out


def first_docstring_line(source: Path) -> str:
    """The first line of ``source``'s module docstring, or an empty string."""
    try:
        doc = ast.get_docstring(ast.parse(source.read_text()))
    except SyntaxError:
        return ""
    return doc.strip().splitlines()[0] if doc else ""


def package_summaries(
    package_root: Path, packages: Sequence[str] | None = None
) -> list[tuple[str, str, str]]:
    """``(dotted name, summary, page)`` for each package the reference lists.

    The landing page of the API reference is built from these: one row per
    package, with the first line of its ``__init__`` docstring and a link to
    its section, relative to ``reference/``. ``packages`` names the dotted
    packages in their listed order; ``None`` lists every top-level subpackage
    of ``package_root``.
    """
    if packages is None:
        packages = [
            f"{package_root.name}.{init.parent.name}"
            for init in sorted(package_root.glob("*/__init__.py"))
            if not init.parent.name.startswith("_")
            and init.parent.name not in SKIPPED_DIRS
        ]
    rows: list[tuple[str, str, str]] = []
    for dotted in packages:
        rel = PurePosixPath(*dotted.split("."))
        directory = package_root.parent / rel
        init = directory / "__init__.py"
        if not init.is_file():
            raise ValueError(f"{dotted} is not a package under {package_root}")
        page = f"{rel}/index.md"
        if init.stat().st_size == 0:
            modules = sorted(
                m for m in directory.glob("*.py") if not m.name.startswith("_")
            )
            if not modules:
                continue
            page = f"{rel}/{modules[0].stem}.md"
        rows.append((dotted, first_docstring_line(init), page))
    return rows


def landing_page(package_root: Path, packages: Sequence[str] | None = None) -> str:
    """The markdown of ``reference/index.md``."""
    lines = [
        "# API reference",
        "",
        "The packages a researcher calls or configures, rendered from their",
        "docstrings: each module's purpose, each class and function with its",
        "signature, and the `#:` documentation of constants and dataclass",
        "fields. The field docs in `causalab.protocol.schema` give the meaning",
        "of every JSON field a document can carry. Each entry links to its",
        "source on GitHub. The [architecture guide](../CODEBASE.md) maps the",
        "rest of the package.",
        "",
        "| Package | Purpose |",
        "|---|---|",
    ]
    for name, summary, page in package_summaries(package_root, packages):
        lines.append(f"| [`{name}`]({page}) | {summary} |")
    return "\n".join(lines) + "\n"


def reference_nav(
    package_root: Path, packages: Sequence[str]
) -> list[tuple[tuple[str, ...], PurePosixPath]]:
    """``(nav titles, page)`` pairs for the API reference, in listed order.

    Each listed package is a top-level section titled with its dotted name,
    and its modules nest under it by their remaining path. A section's first
    entry is its package's ``index.md``, which `mkdocs-section-index` turns
    into the section's own page. Nesting every package under a shared
    ``causalab`` section would instead make the first package's page the index
    of that shared section, and drop it from the list.
    """
    pages = module_pages(package_root, packages)
    out: list[tuple[tuple[str, ...], PurePosixPath]] = []
    for package in packages:
        depth = len(package.split("."))
        for parts, _, page in pages:
            if in_packages(parts, [package]):
                out.append(((package, *parts[depth:]), page))
    return out


def demo_files(demos_root: Path) -> list[Path]:
    """Every file under ``demos/`` the site mirrors, in sorted order."""
    return sorted(
        p for p in demos_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )


def page_title(md: Path) -> str:
    """The text of ``md``'s first ``# `` heading, or its stem when it has none."""
    for line in md.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return md.stem


#: A tutorial file's number: the digits before the first underscore.
FILE_NUMBER = re.compile(r"^(\d+)_")

#: A number that a heading already carries, such as ``04 — ``.
HEADING_NUMBER = re.compile(r"^\d+[a-z]?\s*[—-]\s*")


def tutorial_sections(
    demos_root: Path, sections: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str | None, list[tuple[str, str]]]]:
    """``(title, index page, [(label, page), ...])`` for each Tutorials subsection.

    ``sections`` is ``extra.tutorials`` from `mkdocs.yml`. Each entry has a
    ``title``, optional ``index`` (the subsection's own page), ``pages`` (glob
    patterns under ``demos/``, expanded in sorted order, in list order) and
    ``numbering``:

    * ``file``: a page takes the number its file name starts with. A page
      without one continues the page before it with a letter, so ``03a``
      follows ``03`` and the file numbers the demos cite keep their meaning.
    * ``order``: pages are numbered 1, 2, 3, ... in list order.

    Labels are ``NN — heading``. A number the heading already carries is
    dropped so it does not appear twice. Page paths are relative to
    ``demos/``. A page that two patterns match is listed once, in the first
    subsection that lists it, so a curated subsection (Series) can take pages
    out of a glob that a later one (Papers) keeps.
    """
    out: list[tuple[str, str | None, list[tuple[str, str]]]] = []
    listed: set[str] = set()
    for section in sections:
        index = section.get("index")
        seen = listed | ({index} if index else set())
        pages: list[str] = []
        for pattern in section["pages"]:
            for md in sorted(demos_root.glob(pattern)):
                rel = md.relative_to(demos_root).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    pages.append(rel)
        listed.update(pages)
        numbering = section.get("numbering", "order")
        if numbering not in ("file", "order"):
            raise ValueError(
                f"tutorial section {section['title']!r}: numbering {numbering!r}"
            )
        entries: list[tuple[str, str]] = []
        previous: str | None = None
        for position, rel in enumerate(pages, start=1):
            if numbering == "order":
                number = f"{position:02d}"
            elif match := FILE_NUMBER.match(PurePosixPath(rel).name):
                number = f"{int(match.group(1)):02d}"
            elif previous is not None:
                base = previous.rstrip("abcdefghijklmnopqrstuvwxyz")
                letter = previous[len(base) :]
                number = base + (chr(ord(letter) + 1) if letter else "a")
            else:
                raise ValueError(f"{rel}: no file number, and no page before it")
            previous = number
            title = HEADING_NUMBER.sub("", page_title(demos_root / rel))
            entries.append((f"{number} — {title}", rel))
        out.append((section["title"], index, entries))
    return out


def demo_summary(demos_root: Path, sections: Sequence[Mapping[str, Any]]) -> str:
    """The markdown of ``demos/SUMMARY.md``: the README, then one list per subsection.

    ``README.md`` comes first with an empty title. `mkdocs-section-index`
    turns an untitled first page into the section's own page, so the README
    opens the Tutorials section that `mkdocs.yml` names. A subsection with an
    ``index`` links it from its title, which section-index makes that
    subsection's own page.
    """
    lines = ["* [](README.md)\n"]
    for title, index, entries in tutorial_sections(demos_root, sections):
        lines.append(f"* [{title}]({index})\n" if index else f"* {title}\n")
        lines.extend(f"    * [{label}]({page})\n" for label, page in entries)
    return "".join(lines)


#: A Markdown link, kept as its text in a summary sentence.
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")

#: The end of a sentence: a stop, then space before a capital, a link or code.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\[])")


def opening_sentence(md: Path) -> str:
    """The first sentence of ``md``'s first paragraph, with links as plain text.

    Headings, HTML comments and blank lines before the paragraph are skipped.
    Links lose their targets because the sentence moves to another page,
    where a relative target would not resolve.
    """
    paragraph: list[str] = []
    for line in md.read_text().splitlines():
        stripped = line.strip()
        if not paragraph:
            if not stripped or stripped.startswith(("#", "<!--")):
                continue
        elif not stripped:
            break
        paragraph.append(stripped)
    text = MD_LINK.sub(r"\1", " ".join(paragraph))
    return SENTENCE_END.split(text, maxsplit=1)[0]


def nav_section(nav: Sequence[Any], title: str) -> list[Any] | None:
    """The children of the ``nav`` entry named ``title``, searched depth first."""
    for item in nav:
        if isinstance(item, Mapping):
            for key, value in item.items():
                if key == title and isinstance(value, list):
                    return value
                if isinstance(value, list) and (found := nav_section(value, title)):
                    return found
    return None


def section_index(
    docs_root: Path, title: str, children: Sequence[Any], level: int = 1
) -> str:
    """The markdown of a generated section page: one row per page in the section.

    ``children`` is the section's list from `mkdocs.yml`'s ``nav``. Each titled
    page gets its nav title as the link and its opening sentence as the
    purpose. Untitled entries, which include the generated page itself, are
    skipped. A nested section follows the table under a heading one level
    down, with a table of its own, so a group such as Style guides keeps its
    name on the generated page.
    """
    lines = [f"{'#' * level} {title}", "", "| Guide | Purpose |", "|---|---|"]
    groups: list[tuple[str, Sequence[Any]]] = []
    for child in children:
        if not isinstance(child, Mapping):
            continue
        for label, page in child.items():
            if isinstance(page, str):
                lines.append(
                    f"| [{label}]({page}) | {opening_sentence(docs_root / page)} |"
                )
            elif isinstance(page, list):
                groups.append((label, page))
    text = "\n".join(lines) + "\n"
    for label, group in groups:
        text += "\n" + section_index(docs_root, label, group, level + 1)
    return text


def main() -> None:
    """Emit the generated pages into the build. Called under ``mkdocs``."""
    import mkdocs_gen_files
    from mkdocs_gen_files.nav import Nav

    with mkdocs_gen_files.open("index.md", "w") as fd:
        fd.write((REPO / "README.md").read_text())
    mkdocs_gen_files.set_edit_path("index.md", "README.md")

    demos_root = REPO / "demos"
    for file in demo_files(demos_root):
        rel = file.relative_to(REPO).as_posix()
        with mkdocs_gen_files.open(rel, "wb") as fd:
            fd.write(file.read_bytes())
        if file.suffix == ".md":  # only pages have an edit link
            mkdocs_gen_files.set_edit_path(rel, rel)
    extra = mkdocs_gen_files.config.extra
    with mkdocs_gen_files.open("demos/SUMMARY.md", "w") as fd:
        fd.write(demo_summary(demos_root, extra["tutorials"]))

    site_nav = mkdocs_gen_files.config.nav or []
    for title, page in (extra.get("section_index") or {}).items():
        children = nav_section(site_nav, title)
        if children is None:
            raise ValueError(f"extra.section_index: no nav section named {title!r}")
        with mkdocs_gen_files.open(page, "w") as fd:
            fd.write(
                section_index(Path(mkdocs_gen_files.config.docs_dir), title, children)
            )

    packages = extra.get("api_reference") or [
        name for name, _, _ in package_summaries(REPO / PACKAGE)
    ]
    nav = Nav()
    for keys, page in reference_nav(REPO / PACKAGE, packages):
        nav[keys] = page.relative_to("reference").as_posix()
    for parts, source, page in module_pages(REPO / PACKAGE, packages):
        with mkdocs_gen_files.open(str(page), "w") as fd:
            fd.write(f"::: {'.'.join(parts)}\n")
        mkdocs_gen_files.set_edit_path(str(page), os.fspath(source.relative_to(REPO)))
    with mkdocs_gen_files.open("reference/index.md", "w") as fd:
        fd.write(landing_page(REPO / PACKAGE, packages))
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as fd:
        fd.write("* [Overview](index.md)\n")
        fd.writelines(nav.build_literate_nav())


# `mkdocs-gen-files` runs this file with `runpy.run_path`, whose default module
# name is the literal `<run_path>`. Importing the module, as the tests do, must
# not start a build.
if __name__ == "<run_path>":
    main()
