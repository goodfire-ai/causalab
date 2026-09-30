"""``causalab.analysis.subspace_angles`` — principal angles between two saved bases.

```json
"angles": {
  "type": "script", "script": {"module": "causalab.analysis.subspace_angles"},
  "inputs": {"a": {"step": "fit", "file": "trajectory.safetensors"},
             "b": {"step": "pca", "file": "basis.safetensors"},
             "a_mask": {"step": "fit", "file": "trajectory.safetensors"},
             "match": ["step"]},
  "outputs": {"angles": {"file": "angles.json",
                         "columns": {"a": "string", "b": "string", "dim_a": "int64",
                                     "dim_b": "int64", "max_deg": "float64",
                                     "mean_deg": "float64", "overlap": "float64"}}}
}
```

The question a fitted subspace raises is *which* subspace it is, and a
rotation's coordinates cannot answer it: a DAS loss depends on ``Q`` only through
``span(Q)`` (and a joint rotation-plus-mask fit on ``span(Q_S)``), so two fits
that agree can differ in every entry. The identifiable object is the projector,
and the distance between two projectors is their **principal angles** — the
arccosines of the singular values of ``Q_aᵀ Q_b`` after orthonormalising both.
This step writes one row per matched pair of entries: the two keys, the two
dimensions, the largest and the mean angle in degrees, and ``overlap`` — the
mean squared cosine, ``‖Q_aᵀ Q_b‖²_F / min(dim)``, 1 when one span contains
the other and 0 when they are orthogonal.

``a`` and ``b`` are bundles holding ``weight`` entries ``(d, k)`` (a subspace
fit, a PCA basis, a trajectory). An optional ``a_mask`` (``b_mask``) is a bundle
of ``theta`` entries with the same coordinates; where a mask entry matches a
weight entry, the weight's columns are restricted to the mask's **kept** units
(``θ`` above its map's threshold — ``½`` under ``clamp``, else ``0``), which is
how a joint fit's ``span(Q_S)`` is read out of its trajectory. ``match`` names
the coordinates two entries must share to be compared (default: every
coordinate both carry); with ``b`` a single unswept entry every ``a`` entry is
compared to it. ``truncate_b`` (optional) cuts ``b`` to its first ``dim_a``
columns per pair — a PCA basis against a rank-``s`` fit compares to its top-``s``
block, the fixed-order baseline. All in float64.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, write_table

__all__ = ["main"]

CLAMP_THRESHOLD = 0.5


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    a_entries = _weight_entries(inputs["a"], "a")
    b_entries = _weight_entries(inputs["b"], "b")
    a_masks = _theta_entries(inputs["a_mask"]) if inputs.get("a_mask") else {}
    b_masks = _theta_entries(inputs["b_mask"]) if inputs.get("b_mask") else {}
    match = inputs.get("match")
    if match is not None and (
        not isinstance(match, list) or not all(isinstance(m, str) for m in match)
    ):
        raise StepError("subspace_angles: 'match' is a list of coordinate names")
    truncate_b = bool(inputs.get("truncate_b", False))

    rows: list[dict[str, Any]] = []
    for a_key, (a_coords, qa) in a_entries.items():
        qa = _restrict(qa, a_masks, a_coords, match)
        for b_key, (b_coords, qb) in b_entries.items():
            if len(b_entries) > 1 and not _matches(a_coords, b_coords, match):
                continue
            qb = _restrict(qb, b_masks, b_coords, match)
            if truncate_b:
                qb = qb[:, : qa.shape[1]]
            if qa.shape[0] != qb.shape[0]:
                raise StepError(
                    f"subspace_angles: {a_key!r} lives in {qa.shape[0]} dimensions and "
                    f"{b_key!r} in {qb.shape[0]} — angles need one ambient space"
                )
            angles = _principal_angles(qa.to(torch.float64), qb.to(torch.float64))
            rows.append(
                {
                    "a": a_key,
                    "b": b_key,
                    "dim_a": int(qa.shape[1]),
                    "dim_b": int(qb.shape[1]),
                    "max_deg": max(angles) if angles else float("nan"),
                    "mean_deg": sum(angles) / len(angles) if angles else float("nan"),
                    "overlap": (
                        sum(math.cos(math.radians(x)) ** 2 for x in angles)
                        / len(angles)
                        if angles
                        else float("nan")
                    ),
                }
            )
    if not rows:
        raise StepError(
            "subspace_angles: no pair of entries matched on the coordinates"
        )
    write_table(Path(outputs["angles"]), rows)


def _principal_angles(qa, qb) -> list[float]:
    """Principal angles in degrees between ``span(qa)`` and ``span(qb)``,
    ``min(dim)`` of them, ascending (Björck & Golub 1973: the singular values of
    ``Q_aᵀ Q_b`` for orthonormal ``Q``)."""
    import torch

    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return []
    oa, _ = torch.linalg.qr(qa)
    ob, _ = torch.linalg.qr(qb)
    cosines = torch.linalg.svdvals(oa.T @ ob).clamp(-1.0, 1.0)
    return [math.degrees(math.acos(float(c))) for c in cosines]


def _matches(
    a: Mapping[str, Any], b: Mapping[str, Any], match: list[str] | None
) -> bool:
    names = match if match is not None else sorted(set(a) & set(b))
    return all(
        name in a and name in b and str(a[name]) == str(b[name]) for name in names
    )


def _restrict(q, masks, coords, match):
    """``q`` cut to the kept columns of the mask entry sharing ``coords``."""
    if not masks:
        return q
    for _key, (m_coords, theta, threshold) in masks.items():
        if _matches(coords, m_coords, match):
            kept = (theta.reshape(-1) > threshold).nonzero().reshape(-1)
            if kept.numel() != 0 and kept.max() >= q.shape[1]:
                raise StepError(
                    f"subspace_angles: a mask over {theta.numel()} units cannot select "
                    f"columns of a ({q.shape[0]}, {q.shape[1]}) basis"
                )
            return q[:, kept]
    return q


def _read(source: Any, what: str):
    from causalab.io.tensor_files import load_file

    from causalab.io.step_io import entry_table
    from causalab.protocol.bundles import parse_entry_key
    from causalab.io.env import read_safetensors_metadata

    if not isinstance(source, (str, Path)):
        raise StepError(f"subspace_angles: {what!r} must be a bundle path")
    path = Path(source)
    if not path.is_file():
        raise StepError(
            f"subspace_angles: {what!r} bundle {str(path)!r} does not exist"
        )
    metadata = dict(read_safetensors_metadata(path) or {})
    entries = entry_table(metadata)
    tensors = load_file(str(path))
    out = {}
    for key in sorted(tensors):
        record = entries.get(key, {})
        slot, parsed = parse_entry_key(key)
        slot = str(record.get("slot", slot))
        coords = dict(record.get("coords", parsed))
        out[key] = (slot, coords, tensors[key], record, metadata)
    return out


def _weight_entries(source: Any, what: str):
    found = {
        key: (coords, _as_basis(t, key))
        for key, (slot, coords, t, _r, _m) in _read(source, what).items()
        if slot == "weight"
    }
    if not found:
        raise StepError(
            f"subspace_angles: {what!r} holds no 'weight' entry — a basis is a "
            "subspace fit, a PCA basis or a trajectory of one"
        )
    return found


def _as_basis(t, key: str):
    if t.ndim != 2:
        raise StepError(f"subspace_angles: {key!r} is not a (d, k) matrix")
    # a basis maps d -> k as (d, k); a bundle written the other way is transposed
    return (t if t.shape[0] >= t.shape[1] else t.T).contiguous()


def _theta_entries(source: Any):
    out = {}
    for key, (slot, coords, t, record, metadata) in _read(source, "mask").items():
        if slot != "theta":
            continue
        parametrization = str(
            record.get("parametrization", metadata.get("parametrization", "sigmoid"))
        )
        threshold = CLAMP_THRESHOLD if parametrization == "clamp" else 0.0
        out[key] = (coords, t, threshold)
    if not out:
        raise StepError("subspace_angles: the mask bundle holds no 'theta' entry")
    return out
