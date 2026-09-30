"""Execute compiled interventions through nnsight traces.

The engine uses the shared site map and write math for module boundaries
and addresses supported function interiors through ``.source``.
Install the ``nnsight`` extra to use it.
"""

from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine

__all__ = ["NnsightEngine"]
