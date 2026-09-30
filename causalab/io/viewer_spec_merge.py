#!/usr/bin/env python3
"""Idempotently merge extra ``sections`` into an existing ``viewer_spec.yaml``.

A caller sometimes needs to add figure sections to the ``viewer_spec.yaml`` that
an earlier step already wrote — without clobbering the plan's own default §D
selection. This module is the *general* mechanic for that: it knows nothing about
which figures get added (those globs live in the caller, passed in as a
fragment), only how to splice a list of sections into a base spec, dedup by
section identity (heading, or a ``repeat`` block's ``over``) so re-running is a
no-op, and re-validate the result against the viewer's own schema.

It stays within the ``causalab.io`` layering rule — it imports only
[`causalab.io.artifact_viewer`][] (same layer) for the schema validator and
carries no analysis-specific knowledge.

Merge semantics
---------------
* The fragment is a list of sections (or a mapping with a ``sections`` list),
  shaped exactly like the ``sections`` entries the viewer accepts.
* Each section has a dedup *identity*: a ``heading`` section keys on its
  ``heading``; a headingless ``repeat`` block keys on its ``over`` glob (so a
  per-task repeat block carried in the fragment dedups too).
  Any base section whose identity matches a fragment section's identity is
  **removed first**, then the fragment sections are re-inserted — so a second
  run reproduces the first run's output byte-for-byte (idempotent), and an edited
  fragment *updates* rather than duplicates its section. A section with no
  identity (headingless and not a ``repeat``) is never matched and always
  preserved.
* ``position="prepend"`` (default) puts the fragment ahead of the surviving base
  sections; ``"append"`` puts it after.

CLI::

    python -m causalab.io.viewer_spec_merge \\
        --spec   plan/viewer_spec.yaml \\
        --fragment plan/viewer_spec_extra.yaml \\
        --position prepend
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from causalab.io.artifact_viewer import _validate_spec, load_spec


def _as_sections(fragment: Any) -> list[dict[str, Any]]:
    """Coerce a fragment (a ``sections`` list, or a mapping holding one) to a list."""
    if isinstance(fragment, dict) and "sections" in fragment:
        fragment = fragment["sections"]
    if not isinstance(fragment, list):
        raise ValueError(
            "fragment must be a list of sections or a mapping with a `sections` list, "
            f"got {type(fragment).__name__}"
        )
    return fragment


def _section_identity(section: Any) -> tuple[str, str] | None:
    """Stable dedup key for a section, or ``None`` if it should never be matched.

    A ``heading`` section keys on its heading; a headingless ``repeat`` block
    keys on its ``over`` glob (mirroring the viewer's default ``"*/"`` when
    ``over`` is omitted). Anything else returns ``None`` and is always preserved.
    """
    if not isinstance(section, dict):
        return None
    heading = section.get("heading")
    if isinstance(heading, str):
        return ("heading", heading)
    repeat = section.get("repeat")
    if isinstance(repeat, dict):
        return ("repeat", str(repeat.get("over", "*/")))
    return None


def merge_sections(
    spec: dict[str, Any],
    fragment: Any,
    *,
    position: str = "prepend",
) -> dict[str, Any]:
    """Return a new spec with ``fragment``'s sections merged into ``spec``.

    ``spec`` is not mutated. Dedups by section *identity* — ``heading`` for a
    heading section, ``repeat.over`` for a repeat block (see
    `_section_identity`) — so re-running is a no-op (idempotent), and
    re-validates the result against the viewer schema, so a malformed merge fails
    loudly here rather than at render time.
    """
    if position not in ("prepend", "append"):
        raise ValueError(f"position must be 'prepend' or 'append', got {position!r}")

    extra = _as_sections(fragment)
    base = list(spec.get("sections") or [])

    new_ids = {ident for s in extra if (ident := _section_identity(s)) is not None}
    survivors = [
        s
        for s in base
        if (ident := _section_identity(s)) is None or ident not in new_ids
    ]

    merged = dict(spec)
    merged["sections"] = (
        extra + survivors if position == "prepend" else survivors + extra
    )
    _validate_spec(merged)
    return merged


def merge_spec_file(
    spec_path: str | Path,
    fragment_path: str | Path,
    *,
    position: str = "prepend",
) -> dict[str, Any]:
    """Load ``spec_path`` + ``fragment_path``, merge, and write back to ``spec_path``."""
    spec = load_spec(spec_path)
    fragment = yaml.safe_load(Path(fragment_path).read_text(encoding="utf-8"))
    merged = merge_sections(spec, fragment, position=position)
    Path(spec_path).write_text(
        yaml.safe_dump(merged, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m causalab.io.viewer_spec_merge",
        description="Idempotently merge extra sections into an existing viewer_spec.yaml.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="path to the base viewer_spec.yaml (rewritten in place)",
    )
    parser.add_argument(
        "--fragment",
        required=True,
        help="path to a YAML fragment: a `sections` list, or a mapping with one",
    )
    parser.add_argument(
        "--position",
        choices=("prepend", "append"),
        default="prepend",
        help="where to splice the fragment relative to the base sections (default: prepend)",
    )
    args = parser.parse_args()
    merge_spec_file(args.spec, args.fragment, position=args.position)
    print(args.spec)


if __name__ == "__main__":
    main()
