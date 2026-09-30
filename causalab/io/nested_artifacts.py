"""Save/load helpers for nested-dict-with-tensor caches.

Some on-disk caches (pullback geodesic paths, optimization results, the
featurizer pipeline) are nested ``dict`` structures whose leaves are
``torch.Tensor``. The artifact serialization policy requires these to use
safetensors for tensors and JSON for everything else, so this module flattens
the nested structure into a flat tensor dict + a JSON skeleton with sentinel
references back to the tensors.

Encoded structure:
- Tensors are flattened into ``flat_tensors[key]`` where ``key`` is a
  slash-separated path, e.g. ``pairs/0,1/v_optimized_k``.
- Tuple dict-keys are encoded as ``"int,int,..."`` strings.
- Each tensor leaf in the JSON skeleton is replaced with
  ``{"__tref__": "<flat_key>"}``.
- Lists keep their order; non-tensor scalars (int/float/str/None/bool) are
  preserved as-is. Other types (e.g. ``torch.Size``) are coerced via ``list``.
"""

from __future__ import annotations

from typing import Any

import torch

from causalab.io.artifacts import load_tensors_with_meta, save_tensors_with_meta


_TENSOR_REF = "__tref__"
_TUPLE_KEY_SEP = ","


def _encode_key(key: Any) -> str:
    """Encode a dict key for use both in JSON skeleton and as a path segment.

    Tuples become ``"a,b,c"``; everything else falls back to ``str(key)``.
    Slashes in segments are escaped so they don't collide with the path
    separator used to build flat tensor keys.
    """
    if isinstance(key, tuple):
        return _TUPLE_KEY_SEP.join(str(k) for k in key)
    return str(key).replace("/", "%2F")


def _decode_key(s: str, was_tuple: bool) -> Any:
    if not was_tuple:
        return s.replace("%2F", "/")
    parts = s.split(_TUPLE_KEY_SEP)
    out: list[Any] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            try:
                out.append(float(p))
            except ValueError:
                out.append(p)
    return tuple(out)


def _flatten(
    obj: Any,
    path: str,
    flat_tensors: dict[str, torch.Tensor],
) -> Any:
    """Walk *obj*; emit tensors into *flat_tensors*; return JSON skeleton."""
    if isinstance(obj, torch.Tensor):
        # ``clone()`` so two refs to the same parameter (e.g. tied weights
        # appearing in both forward and inverse featurizer state-dicts) don't
        # share storage — safetensors refuses to write tensors that alias
        # each other.
        flat_tensors[path] = obj.detach().cpu().clone().contiguous()
        return {_TENSOR_REF: path}
    if isinstance(obj, dict):
        encoded: dict[str, Any] = {}
        key_kinds: dict[str, str] = {}
        for k, v in obj.items():
            ek = _encode_key(k)
            child_path = f"{path}/{ek}" if path else ek
            encoded[ek] = _flatten(v, child_path, flat_tensors)
            key_kinds[ek] = "tuple" if isinstance(k, tuple) else "scalar"
        return {"__kind__": "dict", "items": encoded, "key_kinds": key_kinds}
    if isinstance(obj, (list, tuple)):
        items = [
            _flatten(v, f"{path}/{i}" if path else str(i), flat_tensors)
            for i, v in enumerate(obj)
        ]
        return {
            "__kind__": "tuple" if isinstance(obj, tuple) else "list",
            "items": items,
        }
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return {"__kind__": "scalar", "value": obj}
    if isinstance(obj, torch.Size):
        return {
            "__kind__": "list",
            "items": [{"__kind__": "scalar", "value": int(d)} for d in obj],
        }
    raise TypeError(
        f"Unsupported leaf type {type(obj).__name__!r} at path {path!r}; "
        "nested_artifacts only handles dict / list / tuple / tensor / scalar."
    )


def _unflatten(node: Any, flat_tensors: dict[str, torch.Tensor]) -> Any:
    if not isinstance(node, dict):
        # Defensive: legacy nodes without explicit "__kind__" wrapping
        return node
    if _TENSOR_REF in node:
        return flat_tensors[node[_TENSOR_REF]]
    kind = node.get("__kind__")
    if kind == "scalar":
        return node["value"]
    if kind == "list":
        return [_unflatten(v, flat_tensors) for v in node["items"]]
    if kind == "tuple":
        return tuple(_unflatten(v, flat_tensors) for v in node["items"])
    if kind == "dict":
        out: dict[Any, Any] = {}
        items = node["items"]
        key_kinds = node.get("key_kinds", {})
        for ek, v in items.items():
            kk = key_kinds.get(ek, "scalar")
            out[_decode_key(ek, kk == "tuple")] = _unflatten(v, flat_tensors)
        return out
    raise ValueError(f"Unknown node kind {kind!r} in nested artifact skeleton.")


def save_nested(
    payload: Any,
    output_dir: str,
    stem: str,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Save a nested dict/list/tuple/tensor structure as safetensors+meta.

    The structure is split into a flat tensor dict (safetensors) and a JSON
    skeleton that references each tensor by flat key. Round-trips with
    [`load_nested`][] for any combination of dict/list/tuple/tensor/scalar.
    """
    flat_tensors: dict[str, torch.Tensor] = {}
    skeleton = _flatten(payload, "", flat_tensors)
    meta: dict[str, Any] = {"skeleton": skeleton}
    if extra_meta:
        for reserved in ("skeleton",):
            if reserved in extra_meta:
                raise ValueError(
                    f"extra_meta must not contain reserved key {reserved!r}."
                )
        meta.update(extra_meta)
    return save_tensors_with_meta(flat_tensors, meta, output_dir, stem)


def load_nested(output_dir: str, stem: str) -> tuple[Any, dict[str, Any]]:
    """Inverse of [`save_nested`][]. Returns ``(payload, meta)``."""
    flat_tensors, meta = load_tensors_with_meta(output_dir, stem)
    skeleton = meta["skeleton"]
    return _unflatten(skeleton, flat_tensors), meta
