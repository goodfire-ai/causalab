"""Load and verify one saved Fourier fit for downstream analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from causalab.analysis.apply_fourier_probe import apply
from causalab.analysis.fit_fourier_probe import (
    SPLITS,
    activations,
    array,
    frequencies,
    targets,
)
from causalab.io.step_io import (
    StepError,
    read_table,
    read_tensor_with_identity,
    read_values,
)

__all__ = ["load_fit"]


def _population(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> Any:
    """Check the recorded population and return its numeric labels."""
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
    ):
        raise StepError("unsupported Fourier fit schema_version")
    ids = [row.get("id") for row in rows]
    if (
        any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
        or any(row.get("split") not in SPLITS for row in rows)
    ):
        raise StepError("Fourier rows need unique string IDs and declared splits")
    splits = {s: [row["id"] for row in rows if row["split"] == s] for s in SPLITS}
    if any(len(ids) < 2 for ids in splits.values()):
        raise StepError("each Fourier split needs at least two examples")
    if metadata.get("example_ids") != ids or metadata.get("split_ids") != splits:
        raise StepError("rows differ from the fitted Fourier population")
    target = metadata.get("target")
    if not isinstance(target, str) or not target:
        raise StepError("Fourier metadata needs a numeric target column")
    values = [row.get(target) for row in rows]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise StepError("Fourier target labels must be finite numbers")
    return array(values)


def _grid(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the frequency definitions used to reconstruct the targets."""
    specs = metadata.get("frequencies")
    origin = metadata.get("origin")
    if (
        isinstance(origin, bool)
        or not isinstance(origin, (int, float))
        or not math.isfinite(origin)
        or not isinstance(specs, list)
        or not specs
    ):
        raise StepError("Fourier metadata needs a finite origin and frequency grid")
    for spec in specs:
        if (
            not isinstance(spec, dict)
            or not {"period", "harmonic", "frequency", "aliases"} <= spec.keys()
        ):
            raise StepError("incomplete Fourier frequency definition")
        expected = frequencies([spec["period"]], [spec["harmonic"]])[0]
        if (
            type(spec["frequency"]) not in (int, float)
            or spec["frequency"] != expected["frequency"]
        ):
            raise StepError("Fourier frequency differs from period and harmonic")
        aliases = spec["aliases"]
        if not isinstance(aliases, list) or not aliases:
            raise StepError("Fourier frequency aliases must be nonempty")
        for alias in aliases:
            if not isinstance(alias, dict) or set(alias) != {"period", "harmonic"}:
                raise StepError("invalid Fourier frequency alias")
            value = frequencies([alias["period"]], [alias["harmonic"]])[0]["frequency"]
            if not math.isclose(value, expected["frequency"], rel_tol=1e-12, abs_tol=0):
                raise StepError("Fourier alias names a different frequency")
    return specs


def load_fit(directory: Path, acts: Path, rows: Path) -> dict[str, Any]:
    """Verify native files and return arrays, tables and identity.

    ``directory`` contains the seven standard fit outputs. ``acts`` and
    ``rows`` name the population used by that fit. Arrays are float64 NumPy
    arrays, including ``truth`` with shape (examples, frequencies, 2). All
    examples and scores are retained.

    Present tensor stamps must agree. Unstamped direct-Python outputs remain
    caller-owned; replay verifies the supplied data, not its model provenance.
    """
    import numpy as np

    directory, acts, rows = Path(directory), Path(acts), Path(rows)
    names = ("weight", "bias", "plane", "calibration", "predictions")
    paths = {f"{name}.safetensors": directory / f"{name}.safetensors" for name in names}
    paths.update(
        {f"{name}.json": directory / f"{name}.json" for name in ("metadata", "scores")}
    )
    paths.update(acts=acts, rows=rows)
    metadata = read_values(paths["metadata.json"])
    scores, population = read_table(paths["scores.json"]), read_table(rows)
    values = _population(metadata, population)
    specs = _grid(metadata)
    raw_acts, acts_identity = read_tensor_with_identity(acts)
    x = activations(raw_acts)
    tensors, identities = {}, {}
    for name in names:
        tensor, identities[name] = read_tensor_with_identity(
            paths[f"{name}.safetensors"]
        )
        tensors[name] = array(tensor)
    n, p, d = x.shape
    f = len(specs)
    shapes = {
        "weight": (p, f, d, 2),
        "bias": (p, f, 2),
        "plane": (p, f, d, 2),
        "calibration": (p, f, 2, 2),
        "predictions": (n, p, f, 2),
    }
    if n != len(population) or any(
        type(metadata.get(key)) is not int or metadata[key] != size
        for key, size in (("positions", p), ("feature_width", d))
    ):
        raise StepError("Fourier activation shape differs from the fitted population")
    labels = metadata.get("position_labels")
    if (
        not isinstance(labels, list)
        or len(labels) != p
        or any(type(label) not in (str, int) or label == "" for label in labels)
        or len({str(label) for label in labels}) != p
    ):
        raise StepError(
            "Fourier position_labels need one unique string or integer per position"
        )
    for name, shape in shapes.items():
        if tensors[name].shape != shape:
            raise StepError(f"Fourier {name} must have shape {shape}")
    identity = identities["predictions"]
    for name, stamp in identities.items():
        if stamp != identity:
            raise StepError(f"Fourier {name} identity differs from saved predictions")
    shared = (acts_identity.keys() & identity.keys()) - {"dtype", "engine", "commit"}
    for field in shared:
        if acts_identity[field] != identity[field]:
            raise StepError(f"Fourier activation identity differs on {field}")
    expected = {
        (position, frequency) for position in range(p) for frequency in range(f)
    }
    actual = [(score.get("position"), score.get("frequency_index")) for score in scores]
    if (
        any(type(a) is not int or type(b) is not int for a, b in actual)
        or set(actual) != expected
        or len(actual) != len(expected)
    ):
        raise StepError("Fourier score grid is incomplete or duplicated")
    for score in scores:
        position, frequency, rank = (
            score["position"],
            score["frequency_index"],
            score.get("rank"),
        )
        if type(rank) is not int or not 0 <= rank <= min(2, d):
            raise StepError("Fourier plane rank must lie within the feature width")
        if any(score.get(key) != value for key, value in specs[frequency].items()):
            raise StepError("Fourier score frequency differs from metadata")
        plane = tensors["plane"][position, frequency]
        if not np.allclose(
            plane[:, :rank].T @ plane[:, :rank], np.eye(rank), atol=1e-9, rtol=1e-7
        ) or np.any(plane[:, rank:] != 0):
            raise StepError(
                "Fourier plane must have orthonormal active columns and zero padding"
            )
    replay = apply(x, tensors["weight"], tensors["bias"])["predictions"]
    if not np.allclose(replay, tensors["predictions"], atol=1e-9, rtol=1e-7):
        raise StepError("activations do not reproduce saved Fourier predictions")
    if not np.allclose(
        tensors["plane"] @ tensors["calibration"],
        tensors["weight"],
        atol=1e-9,
        rtol=1e-7,
    ):
        raise StepError("Fourier plane calibration does not reproduce the readout")
    return {
        "metadata": metadata,
        "scores": scores,
        "rows": population,
        "acts": x,
        **tensors,
        "truth": targets(values, specs, metadata["origin"]),
        "identity": identity,
    }
