"""Load each arm's benchmark definition under one shared measurement plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..plan import execution_plan
from ..spec import parse_measurement


@dataclass(frozen=True)
class ArmDefinition:
    path: Path
    raw: dict[str, Any]
    protocols: dict[str, Any]
    benchmark_identity: str
    fitting: frozenset[str]


@dataclass(eq=False)
class DefinitionError(ValueError):
    arm: str
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"measurement arm {self.arm!r}, workflow {self.path}: {self.reason}"


def load_definitions(
    document: Path, plan: Mapping[str, Any]
) -> dict[str, ArmDefinition]:
    """Resolve workflow selections and validate their shared measurement interface."""
    from causalab.protocol.pipeline import read_document
    from causalab.io.sources import load_text
    from causalab.workflow.document import ProtocolStep, parse_workflow

    from ..census import strip_pins
    from ..runtime.benchmark import benchmark_identity

    document = document.resolve()
    execution = execution_plan(plan)
    definitions: dict[str, ArmDefinition] = {}
    for arm, selection in execution["arms"].items():
        path = document.parent / selection.get("workflow", document.name)
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(document.parent):
                raise ValueError("workflow path must stay within the study directory")
            raw = load_text(resolved)
            if "workflow" in selection and "measurement" in raw:
                raise ValueError(
                    "selected workflow must not contain a measurement block"
                )
            workflow = parse_workflow(strip_pins(dict(raw))[0])
            parse_measurement(plan, workflow.steps)
            protocols = {}
            for name, step in workflow.steps.items():
                if isinstance(step, ProtocolStep):
                    inner_path = (resolved.parent / step.document).resolve()
                    protocols[name] = dict(
                        read_document(inner_path, inner_path.parent, step.set).raw
                    )
            fitting = frozenset(
                name
                for name, specification in protocols.items()
                if specification["method"].get("train")
            )
            if plan.get("evaluation"):
                for steps in plan["evaluation"]["cases"].values():
                    if fitting.intersection(steps):
                        raise ValueError(
                            "common evaluation must select non-fitting steps"
                        )
            definitions[arm] = ArmDefinition(
                path=resolved,
                raw=raw,
                protocols=protocols,
                benchmark_identity=benchmark_identity(
                    raw, protocols, execution["observations"]
                ),
                fitting=fitting,
            )
        except (OSError, ValueError) as error:
            raise DefinitionError(arm, path, str(error)) from error
    return definitions
