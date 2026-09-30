"""Shared input evidence for configured workflow contrasts at one code commit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..study.scheduler import digest


@dataclass(eq=False)
class ComparisonInputError(ValueError):
    field: str
    reason: str

    def __str__(self) -> str:
        return f"workflow comparison {self.field}: {self.reason}"


def _checkpoint(model: Mapping[str, Any]) -> dict[str, Any]:
    realization = model["realization"]
    return {
        "key": realization["key"],
        "revision": realization["revision"],
        "resolved_revision": model.get("resolved_revision"),
        "files": model["files"],
        "tokenizer": model["tokenizer"],
    }


def checkpoint_aliases(models: Mapping[str, Any]) -> dict[str, str]:
    """Hold model key, authored/resolved revisions, checkpoint bytes and vocabulary.

    Realization settings such as precision and quantization, model/configuration
    classes, attention configuration and parameter counts remain arm-local evidence.
    """
    return {key: digest(_checkpoint(model)) for key, model in models.items()}


def workflow_inputs(
    identity: Mapping[str, Any],
    shared_pins: Mapping[str, Any],
    data_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Hold code, input bytes and data roles while allowing configured methods.

    Data bindings retain expanded point order for observed steps. They do not
    assert equivalent read sites or interventions: those are authored contrasts.
    """
    models = identity["models"]
    aliases = checkpoint_aliases(models)
    return {
        "source_commit": identity["source_commit"],
        "shared_pins": {
            category: entries
            for category, entries in shared_pins.items()
            if category != "documents" and entries
        },
        "data": identity["data"],
        "artifacts": identity["artifacts"],
        "data_bindings": dict(data_bindings),
        "checkpoints": {
            aliases[key]: _checkpoint(model) for key, model in models.items()
        },
    }


def logical_input_rows(
    rows: Sequence[Mapping[str, Any]], aliases: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Compare token sets under checkpoint identities, retaining raw probe evidence."""
    normalized = {}
    for row in rows:
        key = row["model"]
        if key not in aliases:
            raise ComparisonInputError(
                "logical_token_rows.model", f"unattested model realization {key!r}"
            )
        value = {**row, "model": aliases[key]}
        normalized[digest(value)] = value
    return [normalized[key] for key in sorted(normalized)]


def check_workflow_contract(
    contract: Mapping[str, Any],
    *,
    arm: str,
    source_commit: str,
    benchmark_identity: str,
    input_identity: str,
) -> None:
    """Bind the shared receipt identity to the documents and source this arm opened."""
    expected = {"kind": "workflow", "source_commit": source_commit}
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ComparisonInputError(field, "shared contract disagrees with this arm")
    definitions = contract.get("definitions")
    if (
        not isinstance(definitions, Mapping)
        or definitions.get(arm) != benchmark_identity
    ):
        raise ComparisonInputError(
            "definitions", "shared contract omits this arm's definition"
        )
    if digest(contract) != input_identity:
        raise ComparisonInputError(
            "input_identity", "shared contract digest does not match"
        )
