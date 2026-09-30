"""Fit ridge readouts of sine and cosine over a saved activation population.

One call scans a numeric target at each position. It reuses one SVD per
position for every frequency, penalty and shuffled-label control. Inputs may
be residuals, frozen PCA coordinates, or coordinates in a saved DAS basis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import (
    StepError,
    read_table,
    read_tensor,
    write_table,
    write_tensor,
    write_values,
)

SPLITS = ("train", "validation", "evaluation")


def array(value: Any) -> Any:
    """Read a tensor or accept caller-owned numeric data on CPU."""
    import numpy as np

    if isinstance(value, (str, Path)):
        value = read_tensor(Path(value))
    if hasattr(value, "detach"):
        value = value.detach().cpu().double().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise StepError("Fourier inputs must be finite")
    return result


def activations(value: Any) -> Any:
    x = array(value)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3 or min(x.shape) < 1:
        raise StepError("acts must have shape (examples, positions, features)")
    return x


def frequencies(periods: Any, harmonics: Any) -> list[dict[str, Any]]:
    """Combine equal frequencies and retain their authored spellings."""
    if not periods or not harmonics:
        raise StepError("periods and harmonics must be nonempty")
    if any(type(k) is not int or k < 1 for k in harmonics):
        raise StepError("harmonics must be positive integers")
    result: list[dict[str, Any]] = []
    for period in periods:
        if isinstance(period, bool) or not isinstance(period, (int, float)):
            raise StepError("periods must be finite positive numbers")
        if not math.isfinite(period) or period <= 0:
            raise StepError("periods must be finite positive numbers")
        for harmonic in harmonics:
            frequency = harmonic / period
            if not math.isfinite(frequency):
                raise StepError("frequency must be finite")
            spelling = {"period": float(period), "harmonic": harmonic}
            existing = next(
                (
                    r
                    for r in result
                    if math.isclose(r["frequency"], frequency, rel_tol=1e-12, abs_tol=0)
                ),
                None,
            )
            if existing is not None:
                if spelling not in existing["aliases"]:
                    existing["aliases"].append(spelling)
            else:
                result.append(
                    {**spelling, "frequency": frequency, "aliases": [spelling]}
                )
    return result


def targets(values: Any, specs: list[dict[str, Any]], origin: float) -> Any:
    import numpy as np

    turns = (np.asarray(values)[:, None] - origin) * np.array(
        [r["frequency"] for r in specs]
    )
    if not np.isfinite(turns).all():
        raise StepError("target phases must be finite")
    theta = 2 * np.pi * np.remainder(turns, 1)
    y = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    y[np.abs(y) < 1e-12] = 0
    return y


def metrics(truth: Any, prediction: Any) -> dict[str, Any]:
    """Report circular error only where the predicted radius defines a phase."""
    import numpy as np

    error = np.mean((truth - prediction) ** 2, axis=0)
    variance = np.var(truth, axis=0)
    radius = np.linalg.norm(prediction, axis=-1)
    defined = radius > 1e-8
    phase = np.arctan2(prediction[:, 1], prediction[:, 0])
    actual = np.arctan2(truth[:, 1], truth[:, 0])
    phases = np.unique(np.remainder(np.round(actual / (2 * np.pi), 12), 1))
    covered_arc = 2 * np.pi * (1 - np.diff(np.r_[phases, phases[0] + 1]).max())
    distance = np.abs(np.arctan2(np.sin(phase - actual), np.cos(phase - actual)))
    return {
        "mse": float(error.mean()),
        "r2": float(1 - error.sum() / variance.sum())
        if variance.sum() > 1e-12
        else None,
        "cos_r2": float(1 - error[0] / variance[0]) if variance[0] > 1e-12 else None,
        "sin_r2": float(1 - error[1] / variance[1]) if variance[1] > 1e-12 else None,
        "phase_mae": float(distance[defined].mean()) if defined.any() else None,
        "phase_count": int(defined.sum()),
        "count": len(truth),
        "mean_radius": float(radius.mean()),
        "observed_phases": len(phases),
        "covered_arc_radians": float(covered_arc),
    }


def fit(
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Fit on training rows, choose each penalty on validation, then evaluate."""
    import numpy as np

    x = activations(inputs["acts"])
    rows = inputs["rows"]
    if isinstance(rows, (str, Path)):
        rows = read_table(Path(rows))
    if len(rows) != len(x) or not all(isinstance(row, dict) for row in rows):
        raise StepError("rows must align with activation examples")
    ids = [r.get("id") for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise StepError("rows need unique nonempty string IDs")
    if any(r.get("split") not in SPLITS for r in rows):
        raise StepError("rows need train, validation, or evaluation splits")
    splits = {
        s: np.array([i for i, r in enumerate(rows) if r["split"] == s]) for s in SPLITS
    }
    if any(len(indices) < 2 for indices in splits.values()):
        raise StepError("each split needs at least two examples")
    target = inputs["target"]
    raw = [r.get(target) for r in rows]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
        raise StepError("target labels must be finite numbers")
    values = array(raw)
    origin = inputs.get("origin", 0.0)
    if (
        isinstance(origin, bool)
        or not isinstance(origin, (int, float))
        or not math.isfinite(origin)
    ):
        raise StepError("origin must be finite")
    specs = frequencies(
        inputs.get("periods", list(range(2, 151))), inputs.get("harmonics", [1])
    )
    alphas = inputs.get("alphas", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    if not alphas or any(
        isinstance(a, bool)
        or not isinstance(a, (int, float))
        or not math.isfinite(a)
        or a <= 0
        for a in alphas
    ):
        raise StepError("alphas must be finite positive numbers")
    alphas = sorted(set(alphas), reverse=True)
    seed = inputs.get("shuffle_seed", 0)
    if type(seed) is not int or seed < 0:
        raise StepError("shuffle_seed must be a nonnegative integer")
    y = targets(values, specs, origin)
    train, validation, evaluation = (splits[s] for s in SPLITS)
    yt = y[train].reshape(len(train), -1)
    ym = yt.mean(axis=0)
    centered_y = yt - ym
    shuffled_y = centered_y[np.random.default_rng(seed).permutation(len(train))]
    p, f, d = x.shape[1], len(specs), x.shape[2]
    tensors = {
        "weight": np.zeros((p, f, d, 2)),
        "bias": np.zeros((p, f, 2)),
        "plane": np.zeros((p, f, d, 2)),
        "calibration": np.zeros((p, f, 2, 2)),
        "predictions": np.zeros((len(x), p, f, 2)),
    }
    scores, means = [], []
    for position in range(p):
        mean = x[train, position].mean(axis=0)
        features = x[:, position] - mean
        means.append(mean.tolist())
        u, singular, vt = np.linalg.svd(features[train], full_matrices=False)
        projected = features @ vt.T
        rhs = u.T @ np.concatenate((centered_y, shuffled_y), axis=1)
        best = np.full((2, f), np.inf)
        chosen = np.zeros((2, f))
        reduced = np.zeros((2, len(singular), f, 2))
        for alpha in alphas:
            coefficients = (singular / (singular**2 + alpha))[:, None] * rhs
            for control in range(2):
                coeff = coefficients[
                    :, control * 2 * f : (control + 1) * 2 * f
                ].reshape(len(singular), f, 2)
                pred = (
                    projected[validation] @ coeff.reshape(len(singular), -1) + ym
                ).reshape(len(validation), f, 2)
                loss = np.mean((pred - y[validation]) ** 2, axis=(0, 2))
                improve = loss < best[control]
                best[control, improve] = loss[improve]
                chosen[control, improve] = alpha
                reduced[control][:, improve, :] = coeff[:, improve, :]
        weight = (
            (vt.T @ reduced[0].reshape(len(singular), -1))
            .reshape(d, f, 2)
            .transpose(1, 0, 2)
        )
        bias = ym.reshape(f, 2) - np.einsum("d,fdc->fc", mean, weight)
        prediction = np.einsum("nd,fdc->nfc", x[:, position], weight) + bias
        shuffled = (
            projected[evaluation] @ reduced[1].reshape(len(singular), -1) + ym
        ).reshape(len(evaluation), f, 2)
        tensors["weight"][position] = weight
        tensors["bias"][position] = bias
        tensors["predictions"][:, position] = prediction
        for j, spec in enumerate(specs):
            target_rank = int(
                np.linalg.matrix_rank(centered_y[:, 2 * j : 2 * j + 2], tol=1e-10)
            )
            q, s, _ = np.linalg.svd(weight[j], full_matrices=False)
            rank = int(np.sum(s > max(1e-12, float(s[0]) * 1e-10)))
            for column in range(rank):
                if q[np.abs(q[:, column]).argmax(), column] < 0:
                    q[:, column] *= -1
            tensors["plane"][position, j, :, :rank] = q[:, :rank]
            tensors["calibration"][position, j, :rank] = q[:, :rank].T @ weight[j]
            scores.append(
                {
                    "position": position,
                    "frequency_index": j,
                    **spec,
                    "alpha": float(chosen[0, j]),
                    "shuffled_alpha": float(chosen[1, j]),
                    "target_rank": target_rank,
                    "rank": rank,
                    "status": "available" if target_rank else "unavailable",
                    "reason": None if target_rank else "constant_training_targets",
                    "validation": metrics(y[validation, j], prediction[validation, j]),
                    "evaluation": metrics(y[evaluation, j], prediction[evaluation, j]),
                    "baseline": metrics(
                        y[evaluation, j],
                        np.broadcast_to(ym.reshape(f, 2)[j], (len(evaluation), 2)),
                    ),
                    "shuffled": metrics(y[evaluation, j], shuffled[:, j]),
                }
            )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "target": target,
        "origin": float(origin),
        "frequencies": specs,
        "alphas": alphas,
        "shuffle_seed": seed,
        "layer": inputs.get("layer"),
        "representation": inputs.get("representation", "residual"),
        "position_labels": inputs.get("position_labels", list(range(p))),
        "feature_width": d,
        "positions": p,
        "example_ids": ids,
        "split_ids": {s: [ids[i] for i in indices] for s, indices in splits.items()},
        "value_ranges": {
            s: [float(values[indices].min()), float(values[indices].max())]
            for s, indices in splits.items()
        },
        "training_mean": means,
        "selection": "minimum validation pair MSE; exact ties prefer larger alpha",
        "objective": "sum squared error + alpha * squared weight norm; unpenalized intercept",
        "coordinates": ["cos", "sin"],
        "phase_units": "radians",
        "phase_radius_threshold": 1e-8,
        "preprocessing": "training centering; isotropic ridge in the supplied feature space",
        "source": inputs.get("source", {}),
    }
    labels = metadata["position_labels"]
    if (
        not isinstance(labels, list)
        or len(labels) != p
        or any(type(label) not in (str, int) or label == "" for label in labels)
        or len({str(label) for label in labels}) != p
    ):
        raise StepError(
            "position_labels need one unique string or integer per position"
        )
    if any(not np.isfinite(t).all() for t in tensors.values()):
        raise StepError("Fourier fit produced nonfinite tensors")
    return tensors, scores, metadata


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    tensors, scores, metadata = fit(inputs)
    for name, tensor in tensors.items():
        write_tensor(outputs[name], torch.from_numpy(tensor.copy()), slot=name)
    write_table(outputs["scores"], scores)
    write_values(outputs["metadata"], metadata)
