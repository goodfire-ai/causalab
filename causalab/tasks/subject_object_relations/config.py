"""Configuration for the ``subject_object_relations`` factory task.

A single factory over the 35 LRE relations bundled under ``sources/relations/``.
The relation is selected by name (``task.relation=<name>``); its content —
distinct subjects, the deterministic subject→object map, distinct objects, and
the deduped ``{subject}`` templates — is loaded from the committed JSON in
``__post_init__``. The relation content is model-agnostic (all Llama token /
position fields were dropped by ``sources/build_relations.py``), so no ``external artifact storage``
access happens at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_SOURCES_DIR = Path(__file__).resolve().parent / "sources"
_RELATIONS_DIR = _SOURCES_DIR / "relations"
_MANIFEST_PATH = _SOURCES_DIR / "manifest.json"


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    """Load the bundled manifest (valid relation names + per-relation metadata)."""
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


def relation_names() -> list[str]:
    """Sorted list of the valid relation names in the bundle."""
    return sorted(load_manifest()["relations"].keys())


@lru_cache(maxsize=None)
def _load_relation_json(relation: str) -> dict:
    with open(_RELATIONS_DIR / f"{relation}.json") as f:
        return json.load(f)


@dataclass
class SubjectObjectRelationsConfig:
    """Configuration for one LRE relation.

    Only ``relation`` is required; the remaining fields are populated from the
    bundled JSON in `__post_init__` when left empty. Passing them
    explicitly (e.g. a trimmed subject list for a smoke run) is honoured.

    Attributes:
        relation: One of the bundled relation names (validated against the manifest).
        subjects: Input domain — distinct subject strings, first-appearance order.
        subject_to_object: Deterministic subject→object map (the relation's law).
        objects: Distinct object strings (the answer space), first-appearance order.
        templates: Prompt templates with a ``{subject}`` placeholder.
        group: Source group label — one of ``bias`` / ``categorical`` / ``injective``.
        category: Source category (e.g. ``factual``) when the meta carried one.
        output_prefix: String prepended to the object in ``raw_output`` (default " ").
        seed: Random seed (dataclass convention; generators take their own seed).
    """

    relation: str
    subjects: list[str] = field(default_factory=list)
    subject_to_object: dict[str, str] = field(default_factory=dict)
    objects: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    group: str | None = None
    category: str | None = None
    output_prefix: str = " "
    seed: int = 42

    def __post_init__(self) -> None:
        valid = set(relation_names())
        if self.relation not in valid:
            raise ValueError(
                f"relation must be one of the {len(valid)} bundled relations, got "
                f"{self.relation!r}. Available: {sorted(valid)}"
            )
        if not self.subjects:
            data = _load_relation_json(self.relation)
            self.subjects = list(data["subjects"])
            self.subject_to_object = dict(data["subject_to_object"])
            self.objects = list(data["objects"])
            self.templates = list(data["templates"])
            if self.group is None:
                self.group = data.get("group")
            if self.category is None:
                self.category = data.get("category")
