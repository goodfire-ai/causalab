"""Featurizer stages and composition.

Each stage maps ``x`` to ``(features, error)`` and reconstructs a value with
``inverse``. Error terms and unselected dimensions come from the pre-write
value. A subspace preserves its orthogonal complement as
``error = x - Q Q.T x``; a full-rank basis has zero error.

The default sigmoid gate uses ``sigmoid(theta / temperature)`` during
training and ``theta > 0`` during evaluation. Other parametrizations and
top-k readouts define their masks in ``gate.py``. Fitted gates load in
evaluation mode.
Head grouping expands one parameter per head across its coordinates.
Expert-neuron grouping stores ``theta[expert, neuron]`` and uses routing
IDs to select parameters for each token. The registry derives these maps
and feature widths from the model and site.

Stages start on CPU and move to the run device through ``build_stack``.
Boundary casts keep featurizers in fp32 with a bf16 model. A subspace's
local CPU generator makes seeded initialization independent of build order.
The executor supplies ``train.seed``, or zero for an apply document.
A saved basis can initialize the first columns, with seeded completion of
the frame and the source recorded in artifact identity.

Hard-concrete gates sample once per optimizer step from the fit's own CPU
generator. Every use in that step shares the draw, and each cohort member
has an independent sample stream.
"""

# The package root exports the public names that the engines, the analysis
# scripts, and the tests import from it. ``__all__`` is that surface;
# ``tests/neural/shared/test_featurizers_package.py`` pins it. Import a private
# name from its submodule. ``Identity`` stays on ``stages`` because only
# ``build.py`` constructs it, by name.
from causalab.neural.shared.featurizers.sharing import featurizer_cache
from causalab.neural.shared.featurizers.stages import (
    Cayley,
    LoadedLinear,
    ORTHONORMAL_TOLERANCE,
    Sae,
    Stage,
    Standardize,
    Subspace,
    orthonormality_deviation,
)
from causalab.neural.shared.featurizers.gate import (
    BudgetPool,
    Gate,
    gate_poles,
    link_budget_pools,
)
from causalab.neural.shared.featurizers.build import (
    FeaturizerStack,
    build_stack,
    stage_output_width,
)

__all__ = [
    "BudgetPool",
    "Cayley",
    "FeaturizerStack",
    "Gate",
    "LoadedLinear",
    "ORTHONORMAL_TOLERANCE",
    "Sae",
    "Stage",
    "Standardize",
    "Subspace",
    "build_stack",
    "featurizer_cache",
    "gate_poles",
    "link_budget_pools",
    "orthonormality_deviation",
    "stage_output_width",
]
