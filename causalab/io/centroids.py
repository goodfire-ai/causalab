"""Shared data-prep primitives: parameter extraction and centroid aggregation.

These helpers turn a counterfactual dataset into per-example causal parameter
tensors (`extract_parameters_from_dataset`) and group features by unique
parameter combinations into mean centroids (`compute_centroids`). They are
consumed by the spline fitting pipeline (`methods/spline`), the interactive and
static plots (`io/plots`), and analyses, so they live in `io/` — the lowest
shared layer above third-party libs — to keep a single source of truth without
upward imports (docs/CODEBASE.md invariant 3).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

import torch
from torch import Tensor

from causalab.causal.counterfactuals import CounterfactualExample

logger = logging.getLogger(__name__)

EXCLUDED_VARS = {
    "probs",
    "sequence",
    "raw_input",
    "raw_output",
    "true_probs",
    "observations",
    "context_length",
}


class CategoricalParameterError(ValueError):
    """A non-numeric parameter value reached numeric coordinate extraction.

    Geometry/centroid extraction needs numeric coordinates for every causal
    parameter. A categorical (e.g. string) target must either be given an
    embedding (a function mapping its value to floats) or excluded from
    extraction. Subclasses ``ValueError`` so existing ``except ValueError`` /
    ``except Exception`` visualization handlers still skip gracefully, but the
    message is actionable instead of the raw ``could not convert string to
    float``. See the ``EMBEDDINGS`` export in ``causalab/tasks/README.md``.
    """


def coerce_param_to_float(var: str, value: Any) -> float:
    """Convert a single parameter value to float, with an actionable error.

    The geometry/centroid pipeline projects causal parameters into numeric
    coordinates. When a value cannot be converted (a categorical/string target
    with no embedding), raise [`CategoricalParameterError`][] naming the
    variable and pointing at the embedding escape hatch, instead of letting a
    bare ``float()`` surface the opaque ``could not convert string to float``.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise CategoricalParameterError(
            f"Causal parameter {var!r} has non-numeric value {value!r}; "
            f"geometry/centroid extraction requires numeric coordinates. Add an "
            f"EMBEDDINGS entry for {var!r} in causalab/tasks/<task>/causal_models.py "
            f"(a function mapping its value to floats), or exclude it from "
            f"extraction."
        ) from e


def extract_parameters_from_dataset(
    dataset: list[CounterfactualExample],
    excluded_vars: set[str] | None = None,
    embeddings: dict[str, Callable[[Any], list[float]]] | None = None,
    causal_model: Any | None = None,
) -> dict[str, Tensor]:
    """Extract parameter values from input traces of a counterfactual dataset.

    Iterates over examples and extracts causal parameter values from the input
    trace only (not counterfactual traces). This gives one value per example,
    aligned with features from collect_features().

    When an embedding function is provided for a variable, it is used to map
    the value to one or more floats (e.g. cyclic day -> [cos, sin]).
    Multi-dimensional embeddings produce keys like ``var_0``, ``var_1``, etc.
    When no embedding is provided, the value is converted via ``float()``; a
    non-numeric (categorical) value raises [`CategoricalParameterError`][]
    naming the variable and the embedding remedy, rather than a raw ``float()``
    ``ValueError``.

    Tuple-valued parameters (without an embedding) are expanded into separate
    dimensions (mu_0, mu_1).

    Args:
        dataset: List of CounterfactualExample dicts.
        excluded_vars: Set of variable names to skip. Uses EXCLUDED_VARS if None.
        embeddings: Optional dict mapping variable names to embedding functions.
            Each function takes a variable value and returns a list of floats.
        causal_model: Optional CausalModel. Its ``embeddings`` are merged under
            ``embeddings`` (explicit entries win), so a model-side embedding is
            never silently ignored when a partial ``embeddings`` dict is passed.

    Returns:
        Dict mapping parameter names to tensors of shape (n_examples,).
    """
    if excluded_vars is None:
        excluded_vars = EXCLUDED_VARS
    # Merge the causal model's embeddings with any explicit overrides so an
    # embedding present on the model is never silently ignored when a caller
    # passes a partial dict; explicit entries win. Without this, a categorical
    # target with a model-side embedding could still hit the float() guard.
    model_embeddings = getattr(causal_model, "embeddings", None) or {}
    embeddings_nn: dict[str, Callable[[Any], list[float]]] = {
        **model_embeddings,
        **(embeddings or {}),
    }

    param_values: dict[str, list[float]] = defaultdict(list)
    tuple_params: dict[str, int] = {}

    def _extract_from_trace(trace):
        for var in trace._values:
            if var in excluded_vars:
                continue
            val = trace[var]
            if val is None:
                continue
            if var in embeddings_nn:
                coords = embeddings_nn[var](val)
                if len(coords) == 1:
                    param_values[var].append(coords[0])
                else:
                    for j, c in enumerate(coords):
                        param_values[f"{var}_{j}"].append(c)
            elif isinstance(val, (tuple, list)):
                if var not in tuple_params:
                    tuple_params[var] = len(val)
                for j, v in enumerate(val):
                    param_values[f"{var}_{j}"].append(coerce_param_to_float(var, v))
            else:
                param_values[var].append(coerce_param_to_float(var, val))

    for ex in dataset:
        _extract_from_trace(ex["input"])

    return {k: torch.tensor(v) for k, v in param_values.items()}


def compute_centroids(
    features: Tensor,
    param_tensors: dict[str, Tensor],
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Group features by unique parameter combinations and compute mean centroids.

    **Ordering warning**: The returned centroids are sorted by torch.unique's
    lexicographic order on the parameter combinations, which may NOT match the
    task's class index order. For example, 2D grid coordinates (angle, height)
    get sorted by (angle, height) while class indices may enumerate by
    (height, angle). Do not assume centroid[i] corresponds to class i.
    Use manifold.encode(class_ordered_centroids) to get intrinsic coordinates
    in class order.

    Args:
        features: Feature tensor (n_samples, ambient_dim).
        param_tensors: Dict mapping parameter names to tensors of shape (n_samples,).

    Returns:
        control_points: Unique parameter combinations (n_centroids, n_params).
        centroids: Mean features per group (n_centroids, ambient_dim).
        metadata: Dict with parameter_names and counts.
    """
    n = features.shape[0]
    param_names = sorted(param_tensors.keys())
    param_matrix = torch.stack([param_tensors[name] for name in param_names], dim=1)

    # Find unique parameter combinations
    unique_params, inverse_indices = torch.unique(
        param_matrix, dim=0, return_inverse=True
    )
    n_centroids = unique_params.shape[0]

    # Compute mean features per group
    centroids = torch.zeros(n_centroids, features.shape[1], dtype=features.dtype)
    counts = torch.zeros(n_centroids, dtype=torch.long)
    for i in range(n):
        idx = inverse_indices[i]
        centroids[idx] += features[i]
        counts[idx] += 1

    centroids = centroids / counts.unsqueeze(1).float()

    metadata = {
        "parameter_names": param_names,
        "n_centroids": n_centroids,
        "counts": counts.tolist(),
    }

    logger.info(
        f"Computed {n_centroids} centroids from {n} samples (params: {param_names})"
    )

    return unique_params, centroids, metadata
