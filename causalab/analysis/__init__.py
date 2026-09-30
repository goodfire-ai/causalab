"""Analysis functions for workflow script steps.

Each script reads declared inputs and writes its declared outputs through
``main(inputs, outputs)``. Numerical imports occur inside the entry point so
workflow validation can locate and hash scripts without loading those libraries.
See ``docs/workflow_protocol.md`` for the script interface."""

from __future__ import annotations

__all__: list[str] = []
