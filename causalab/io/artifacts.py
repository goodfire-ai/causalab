"""
I/O utilities for saving and loading experiment results.

This module provides functions for saving and loading experiment results,
metadata, and trained models to/from disk.

See docs/CODEBASE.md "Artifact serialization policy": all on-disk artifacts
are split into a `.safetensors` tensor payload and an optional sibling
`.meta.json` metadata file. `torch.save` / `torch.load` are forbidden in
`causalab/` outside this module.
"""

import logging
import os
import json
import pickle
from typing import Any, Callable, Dict, Tuple

import torch
from causalab.io.tensor_files import load_file, save_file


logger = logging.getLogger(__name__)


# Bumped when the on-disk format changes in a backwards-incompatible way.
ARTIFACT_FORMAT_VERSION = 1

# Sentinel keys reserved on the meta JSON for schema bookkeeping.
_META_VERSION_KEY = "_format_version"
_META_SCHEMA_KEY = "_schema"
_META_CLASS_KEY = "_class"

# Schema tags for the four supported artifact shapes.
_SCHEMA_TENSORS_WITH_META = "tensors_with_meta"
_SCHEMA_MODULE = "module"


def _key_to_str(key: Tuple[Any, ...]) -> str:
    """Convert a tuple key to a string representation for file naming."""
    if len(key) == 2:
        return f"{key[0]}__{key[1]}"
    elif len(key) == 1:
        return str(key[0])
    return "__".join(str(k) for k in key)


def save_experiment_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
    filename: str = "metadata.json",
) -> str:
    """
    Save experiment metadata to JSON file.

    Args:
        metadata: Metadata dictionary to save
        output_dir: Output directory
        filename: Filename for metadata (default: "metadata.json")

    Returns:
        Path to saved metadata file
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, filename)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def save_json_results(
    data: Dict[str, Any],
    output_dir: str,
    filename: str,
) -> str:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON-serializable)
        output_dir: Output directory
        filename: Filename (should end with .json)

    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return file_path


def save_tensor_results(
    tensors: Dict[str, torch.Tensor],
    output_dir: str,
    filename: str,
) -> str:
    """
    Save tensors to safetensors file.

    Args:
        tensors: Dictionary mapping names to tensors
        output_dir: Output directory
        filename: Filename (should end with .safetensors)

    Returns:
        Path to saved safetensors file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    save_file(tensors, file_path)

    return file_path


def load_json_results(output_dir: str, filename: str) -> Dict[str, Any]:
    """Load data from JSON file.

    Args:
        output_dir: Directory containing the file
        filename: Filename to load

    Returns:
        Parsed JSON data as a dictionary
    """
    file_path = os.path.join(output_dir, filename)
    with open(file_path) as f:
        return json.load(f)


def feature_cache_path(output_dir: str, layer: Any, pos_id: Any) -> str:
    """Canonical path for a per-(layer, position) activation cache file."""
    return os.path.join(
        output_dir, "features", f"L{layer}_{pos_id}_features.safetensors"
    )


def load_cached_features(
    output_dir: str, layer: Any, pos_id: Any, expected_n: int
) -> torch.Tensor | None:
    """Load cached features for a (layer, pos_id) target, or None if missing/stale."""
    path = feature_cache_path(output_dir, layer, pos_id)
    if not os.path.exists(path):
        return None
    cached = load_file(path)["features"]
    if cached.shape[0] != expected_n:
        return None
    return cached


def save_cached_features(
    output_dir: str, layer: Any, pos_id: Any, features: torch.Tensor
) -> str:
    """Save features for a (layer, pos_id) target to the canonical cache path."""
    path = feature_cache_path(output_dir, layer, pos_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"features": features}, path)
    return path


def load_tensor_results(output_dir: str, filename: str) -> Dict[str, torch.Tensor]:
    """Load tensors from safetensors file.

    Args:
        output_dir: Directory containing the file
        filename: Filename to load (should end with .safetensors)

    Returns:
        Dictionary mapping tensor names to tensors
    """
    file_path = os.path.join(output_dir, filename)
    return load_file(file_path)


def save_pickle(obj: Any, output_dir: str, filename: str) -> str:
    """Save arbitrary Python object via pickle.

    Args:
        obj: Object to pickle
        output_dir: Output directory
        filename: Filename (should end with .pkl)

    Returns:
        Path to saved pickle file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    return file_path


def load_pickle(output_dir: str, filename: str) -> Any:
    """Load Python object from pickle file.

    Args:
        output_dir: Directory containing the file
        filename: Filename to load

    Returns:
        Unpickled Python object
    """
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_intervention_results(
    results_by_key: Dict[Tuple[Any, ...], Dict[str, Any]],
    output_dir: str,
    prefix: str = "",
) -> Dict[str, str]:
    """
    Save intervention results (scores and raw outputs).

    Args:
        results_by_key: Dict mapping keys to result dicts with:
            - score: float
            - scores_by_variable: dict (unused, kept for compatibility)
            - raw_results: dict with "string" and "sequences" — the legacy
              one-synthetic-batch view producers emit via
              `to_raw_results`
              (the stored-artifact schema is unchanged for runs
              with ``batch_size >= n_examples`` — legacy multi-batch runs
              stored one inner ``"string"`` list per batch, and
              single-example batches a bare str, where ``to_raw_results()``
              always emits the one-batch nesting)
        output_dir: Output directory
        prefix: Optional prefix for subdirectory (e.g., "train_eval", "test_eval")

    Returns:
        Dict of output paths
    """
    if prefix:
        save_dir = os.path.join(output_dir, prefix)
    else:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    paths = {}

    # Save scores.json: {key: score}
    scores = {}
    for key, result in results_by_key.items():
        key_str = _key_to_str(key)
        scores[key_str] = result["avg_score"]

    scores_path = save_json_results(scores, save_dir, "scores.json")
    paths["scores_path"] = scores_path

    # Save raw_results.json: aggregated string outputs
    raw_results_json = {
        "string": [
            result["raw_results"]["string"] for result in results_by_key.values()
        ]
    }
    raw_path = save_json_results(raw_results_json, save_dir, "raw_results.json")
    paths["raw_results_path"] = raw_path

    # Save raw_results.safetensors: aggregated tensor outputs (if present)
    sequences_list = []
    for result in results_by_key.values():
        if "sequences" in result["raw_results"] and result["raw_results"]["sequences"]:
            sequences_list.extend(result["raw_results"]["sequences"])

    if sequences_list:
        sequences_tensor = torch.cat(sequences_list, dim=0)
        tensor_path = save_tensor_results(
            {"sequences": sequences_tensor},
            save_dir,
            "raw_results.safetensors",
        )
        paths["raw_results_tensors_path"] = tensor_path

    return paths


def save_aggregate_metadata(
    metadata: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Save aggregate experiment metadata.

    Alias for save_experiment_metadata for consistency.
    """
    return save_experiment_metadata(metadata, output_dir, "metadata.json")


# ---------------------------------------------------------------------------
# Tensor + metadata helpers (artifact serialization policy)
# ---------------------------------------------------------------------------


def _safetensors_path(output_dir: str, stem: str) -> str:
    return os.path.join(output_dir, f"{stem}.safetensors")


def _meta_json_path(output_dir: str, stem: str) -> str:
    return os.path.join(output_dir, f"{stem}.meta.json")


def _check_format_version(meta: Dict[str, Any], path: str) -> None:
    version = meta.get(_META_VERSION_KEY)
    if version is None:
        raise ValueError(
            f"Meta JSON at {path} is missing '{_META_VERSION_KEY}' — refusing "
            "to load (was it produced by a pre-policy writer?)."
        )
    if version > ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Meta JSON at {path} has format version {version}, but this "
            f"build only supports up to version {ARTIFACT_FORMAT_VERSION}."
        )
    if version < ARTIFACT_FORMAT_VERSION:
        logger.warning(
            "Loading older artifact format %d at %s; current is %d",
            version,
            path,
            ARTIFACT_FORMAT_VERSION,
        )


def save_tensors_with_meta(
    tensors: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    output_dir: str,
    stem: str,
) -> Tuple[str, str]:
    """Save tensors + JSON metadata under a shared stem.

    Writes `<output_dir>/<stem>.safetensors` for the tensor payload and
    `<output_dir>/<stem>.meta.json` for metadata. Reserved schema keys
    (`_format_version`, `_schema`) are added to the meta dict.

    Args:
        tensors: Map of tensor name to tensor. May be empty.
        meta: JSON-serializable metadata dict. Must not contain any tensors;
            decompose at the call site instead.
        output_dir: Output directory (created if missing).
        stem: Filename stem without extension.

    Returns:
        (safetensors_path, meta_json_path)
    """
    for reserved in (_META_VERSION_KEY, _META_SCHEMA_KEY):
        if reserved in meta:
            raise ValueError(f"meta must not contain reserved key {reserved!r}.")

    os.makedirs(output_dir, exist_ok=True)
    safetensors_path = _safetensors_path(output_dir, stem)
    meta_json_path = _meta_json_path(output_dir, stem)

    save_file(tensors, safetensors_path)

    enriched_meta = {
        _META_VERSION_KEY: ARTIFACT_FORMAT_VERSION,
        _META_SCHEMA_KEY: _SCHEMA_TENSORS_WITH_META,
        **meta,
    }
    with open(meta_json_path, "w") as f:
        json.dump(enriched_meta, f, indent=2)

    return safetensors_path, meta_json_path


def load_tensors_with_meta(
    output_dir: str,
    stem: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load tensors + JSON metadata previously written by `save_tensors_with_meta`.

    Returns:
        (tensors, meta) — `tensors` is the safetensors dict; `meta` is the
        full meta JSON including the reserved `_format_version` / `_schema`
        keys.
    """
    safetensors_path = _safetensors_path(output_dir, stem)
    meta_json_path = _meta_json_path(output_dir, stem)

    tensors = load_file(safetensors_path)
    with open(meta_json_path) as f:
        meta = json.load(f)
    _check_format_version(meta, meta_json_path)
    return tensors, meta


def save_module(
    module: torch.nn.Module,
    output_dir: str,
    stem: str,
    extra_meta: Dict[str, Any] | None = None,
    extra_tensors: Dict[str, torch.Tensor] | None = None,
) -> Tuple[str, str]:
    """Save a `torch.nn.Module` checkpoint as safetensors + meta JSON.

    The module's `state_dict` is written to `<stem>.safetensors`. Any
    additional tensors (e.g. preprocessing mean/std, learned rotation) can be
    passed via `extra_tensors` and are merged into the same safetensors file
    under the `extra.` prefix to keep them separable on load.

    Non-tensor configuration goes to `<stem>.meta.json` together with the
    module's fully-qualified class name (so `load_module` can dispatch the
    factory) and the standard schema keys.

    Args:
        module: Module to checkpoint. Only its `state_dict()` is saved — no
            architecture is inferred. Callers must provide a factory at load
            time that rebuilds an empty module of compatible shape.
        output_dir: Output directory.
        stem: Filename stem without extension.
        extra_meta: Optional extra metadata dict. Schema keys are reserved.
        extra_tensors: Optional extra tensors merged under `extra.<name>`.

    Returns:
        (safetensors_path, meta_json_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    safetensors_path = _safetensors_path(output_dir, stem)
    meta_json_path = _meta_json_path(output_dir, stem)

    payload: Dict[str, torch.Tensor] = {
        f"state_dict.{k}": v for k, v in module.state_dict().items()
    }
    if extra_tensors:
        for k, v in extra_tensors.items():
            payload[f"extra.{k}"] = v
    save_file(payload, safetensors_path)

    cls = type(module)
    qualname = f"{cls.__module__}.{cls.__qualname__}"
    enriched_meta: Dict[str, Any] = {
        _META_VERSION_KEY: ARTIFACT_FORMAT_VERSION,
        _META_SCHEMA_KEY: _SCHEMA_MODULE,
        _META_CLASS_KEY: qualname,
    }
    if extra_meta:
        # Caller-provided meta cannot overwrite reserved schema keys.
        for reserved in (_META_VERSION_KEY, _META_SCHEMA_KEY, _META_CLASS_KEY):
            if reserved in extra_meta:
                raise ValueError(
                    f"extra_meta must not contain reserved key {reserved!r}."
                )
        enriched_meta.update(extra_meta)

    with open(meta_json_path, "w") as f:
        json.dump(enriched_meta, f, indent=2)

    return safetensors_path, meta_json_path


def load_module(
    factory: Callable[[Dict[str, Any]], torch.nn.Module],
    output_dir: str,
    stem: str,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, torch.Tensor]]:
    """Load a module checkpoint written by `save_module`.

    Args:
        factory: Builds an empty module given the meta dict (so the caller
            can read e.g. `meta["config"]` to construct the right shape).
        output_dir: Directory containing the artifact pair.
        stem: Filename stem without extension.
        strict: Forwarded to `Module.load_state_dict`.

    Returns:
        (module, meta, extra_tensors) where `extra_tensors` is the
        `extra.*` slice of the safetensors payload (already stripped of the
        prefix) and `meta` includes the schema keys.
    """
    safetensors_path = _safetensors_path(output_dir, stem)
    meta_json_path = _meta_json_path(output_dir, stem)

    with open(meta_json_path) as f:
        meta = json.load(f)
    _check_format_version(meta, meta_json_path)

    payload = load_file(safetensors_path)
    state_dict: Dict[str, torch.Tensor] = {}
    extra_tensors: Dict[str, torch.Tensor] = {}
    for k, v in payload.items():
        if k.startswith("state_dict."):
            state_dict[k[len("state_dict.") :]] = v
        elif k.startswith("extra."):
            extra_tensors[k[len("extra.") :]] = v
        else:
            raise ValueError(
                f"Unexpected key {k!r} in module checkpoint at "
                f"{safetensors_path}; expected 'state_dict.*' or 'extra.*'."
            )

    module = factory(meta)
    expected_class = meta[_META_CLASS_KEY]
    actual_class = f"{type(module).__module__}.{type(module).__qualname__}"
    if actual_class != expected_class:
        raise ValueError(
            f"Factory produced {actual_class!r}, but meta JSON at "
            f"{meta_json_path} declares {expected_class!r}."
        )
    module.load_state_dict(state_dict, strict=strict)
    return module, meta, extra_tensors
