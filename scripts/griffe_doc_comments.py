"""A griffe extension that reads ``#:`` attribute docs.

`mkdocstrings` builds the API reference from what `griffe` finds in the
source. Constants and dataclass fields are documented with comment lines that
start with ``#:`` immediately above the assignment, the convention Sphinx
autodoc reads. griffe reads only a string literal placed after an assignment,
so without help every such field renders undocumented.
``DocComments.on_attribute_instance`` collects the comment block above an
attribute and installs it as the attribute's docstring. The extension is
named in `mkdocs.yml` under the Python handler's ``extensions`` option and has
no effect on the package at runtime.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import griffe

#: One ``#:`` line: the marker, an optional single space, the text.
DOC_COMMENT = re.compile(r"^\s*#:(?: ?(.*))?$")


class DocComments(griffe.Extension):
    """Attach ``#:`` comments as docstrings."""

    def __init__(self, packages: list[str] | None = None) -> None:
        """Read the conventions for the dotted ``packages``, or for everything.

        A role links only to an object inside ``packages``, the ones the API
        reference renders. ``None`` links to any public object of the package.
        """
        self._sources: dict[Path, list[str]] = {}
        self._packages = packages

    def _lines(self, path: Path) -> list[str]:
        if path not in self._sources:
            self._sources[path] = path.read_text().splitlines()
        return self._sources[path]

    def on_attribute_instance(
        self,
        *,
        node: Any,
        attr: griffe.Attribute,
        agent: Any,
        **kwargs: Any,
    ) -> None:
        """Use the ``#:`` block directly above ``attr`` as its docstring.

        An attribute that already has a docstring keeps it. The block is the
        maximal run of ``#:`` lines ending on the line before the assignment;
        a blank ``#:`` line inside it is a paragraph break.
        """
        if attr.docstring is not None or attr.lineno is None:
            return
        filepath = attr.filepath
        if isinstance(filepath, list):  # a namespace package has several roots
            return
        lines = self._lines(Path(filepath))
        collected: list[str] = []
        i = attr.lineno - 2  # zero-based index of the line above the assignment
        while i >= 0:
            match = DOC_COMMENT.match(lines[i])
            if match is None:
                break
            collected.append(match.group(1) or "")
            i -= 1
        if not collected:
            return
        text = "\n".join(reversed(collected)).strip()
        attr.docstring = griffe.Docstring(text, lineno=i + 2, parent=attr)
