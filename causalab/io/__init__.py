"""Artifact I/O and plotting primitives.

This package is the single source of truth for code that touches disk:
- ``artifacts``: JSON / safetensors / pickle save/load, metadata, intervention
  and training result writers.
- ``plots``: shared figure rendering helpers.
- ``counterfactuals``: counterfactual dataset save/load.
- ``configs``: runner config save/load for notebook workflows.
- ``pipelines``: LMPipeline and analysis-result loaders.
- ``sae_checkpoints``: readers for *foreign* SAE checkpoints (the sanctioned
  ``torch.load(weights_only=False)`` exception) — vanilla decoder + block frame.
- ``artifact_viewer``: generic, spec-driven HTML viewer that renders a browsable
  page of experiment artifacts from a declarative ``viewer_spec.yaml``.

Dependency flow: ``workflow/`` → ``io/`` → ``protocol/``. The static guard in
``tests/test_architecture_layering.py`` prohibits module-level imports from
``causalab.workflow`` (including ``causalab.workflow.scripts``) into this package.
"""
