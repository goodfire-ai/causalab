"""Readers of the normative spec's tables (``docs/intervention_protocol.md``
and its implementation companion ``docs/intervention_protocol_internals.md``),
shared by the tests that hold code to a table the spec prints — the stage
list, the output list, the diagnostic kinds."""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
SPEC = DOCS / "intervention_protocol.md"
#: Sections 6–8 and 9.1 of the spec: derived properties, canonical form,
#: engine contract, compilation stages.
INTERNALS = DOCS / "intervention_protocol_internals.md"


def spec_section(heading: str, spec: Path = SPEC) -> str:
    """The body of the section ``heading`` opens in ``spec``, up to the next
    heading of the same or a shallower depth."""
    depth = len(heading) - len(heading.lstrip("#"))
    body = spec.read_text().split(heading, 1)
    assert len(body) == 2, f"{heading!r} is not in {spec.name}"
    stop = re.compile(rf"^#{{1,{depth}}} ", re.M)
    end = stop.search(body[1])
    return body[1][: end.start()] if end else body[1]


def spec_tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in ``text``, as rows of stripped cells (the header
    row first, the rule row dropped)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        match = re.match(r"^[ \t]*\|(.+)\|\s*$", line)
        if not match:
            if current:
                tables.append(current)
                current = []
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        current.append(cells)
    if current:
        tables.append(current)
    return tables


def first_column(table: list[list[str]]) -> tuple[str, ...]:
    """The code cell of every body row's first column, in order."""
    out: list[str] = []
    for row in table[1:]:  # the header row is the first
        code = re.search(r"`([^`]+)`", row[0])
        assert code, f"row without a code cell: {row}"
        out.append(code.group(1))
    return tuple(out)
