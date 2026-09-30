"""Execute compiled interventions through native PyTorch hooks.

``PytorchHooksEngine`` uses the shared site map, position resolution,
featurizers, write math, and results layer. This engine supplies hooks,
model loading, and training.
"""

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.encoding import (
    EncodedBatch,
    encode,
    resolve_position,
)
from causalab.neural.engines.pytorch_hooks.executor import PointExecutor
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import ResolvedSite, resolve_site

__all__ = [
    "EncodedBatch",
    "ModelBundle",
    "PointExecutor",
    "PytorchHooksEngine",
    "ResolvedSite",
    "encode",
    "load_model",
    "resolve_position",
    "resolve_site",
]
