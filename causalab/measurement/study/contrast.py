"""Bind independently authored workflows to one committed implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .scheduler import digest


@dataclass(eq=False)
class ComparisonError(ValueError):
    reason: str

    def __str__(self) -> str:
        return f"workflow comparison: {self.reason}"


@dataclass(frozen=True)
class WorkflowContrast:
    source_commit: str
    definitions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.definitions]
        if (
            not self.source_commit
            or set(names) not in ({"before", "after"}, {"before", "after", "eager"})
            or len(names) != len(set(names))
            or any(not value for _, value in self.definitions)
        ):
            raise ComparisonError(
                "requires a source commit and a definition for every arm"
            )

    @classmethod
    def create(
        cls, commits: Mapping[str, str], definitions: Mapping[str, str]
    ) -> WorkflowContrast:
        if set(commits) != set(definitions):
            raise ComparisonError("source and definition arm names must match")
        if len(set(commits.values())) != 1:
            raise ComparisonError("all arms must use the same resolved code commit")
        return cls(next(iter(commits.values())), tuple(sorted(definitions.items())))

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "workflow",
            "source_commit": self.source_commit,
            "definitions": dict(self.definitions),
        }

    @property
    def identity(self) -> str:
        return digest(self.receipt())


def definition_changes(
    before: Any, after: Any, path: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """A deterministic JSON diff; missing keys remain distinct from null values."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            location = [*path, key]
            if key not in before:
                changes.append(
                    {
                        "path": location,
                        "kind": "added",
                        "before": None,
                        "after": after[key],
                    }
                )
            elif key not in after:
                changes.append(
                    {
                        "path": location,
                        "kind": "removed",
                        "before": before[key],
                        "after": None,
                    }
                )
            else:
                changes.extend(
                    definition_changes(before[key], after[key], tuple(location))
                )
        return changes
    if type(before) is type(after) and before == after:
        return []
    return [{"path": list(path), "kind": "changed", "before": before, "after": after}]
