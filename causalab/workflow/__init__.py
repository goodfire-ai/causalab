"""Execute workflows and write step records and the run manifest.

The document module parses and schedules workflows. Protocol steps use the
selected engine; script steps receive resolved inputs through
``main(inputs, outputs)``. Each step writes ``_step.json``, and ``workflow.json``
records the run state. See docs/workflow_protocol_internals.md §8."""

from causalab.workflow.runner import (
    OverlayArtifacts,
    WorkflowRunResult,
    run_workflow,
)

__all__ = ["OverlayArtifacts", "WorkflowRunResult", "run_workflow"]
