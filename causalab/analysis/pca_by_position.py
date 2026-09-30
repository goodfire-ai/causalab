"""Fit PCA separately at each position using only the declared training rows.

Inputs: acts (examples, positions, features), train_rows, k. Outputs: weight
(positions, features, k), mean (positions, features), coordinates (examples,
positions, k), spectrum (table). Each position is fit over the training rows
alone, so ``k`` is at most ``min(len(train_rows) - 1, features)``: centering
removes one degree of freedom (``causalab.analysis.fit_pca``). Positions may
be semantic aligned targets; never pad missing targets into a population. A
grouped weight is an analysis artifact, not one featurizer; select a position
before using it for a localizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.analysis.fit_pca import fit
from causalab.io.step_io import StepError, read_tensor, write_table, write_tensor


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    acts = inputs["acts"]
    if isinstance(acts, (str, Path)):
        acts = read_tensor(Path(acts))
    if acts.ndim != 3:
        raise StepError("pca_by_position needs (examples, positions, features)")
    train_rows = inputs["train_rows"]
    if (
        not isinstance(train_rows, list)
        or len(train_rows) < 2
        or any(type(i) is not int or not 0 <= i < len(acts) for i in train_rows)
        or len(set(train_rows)) != len(train_rows)
    ):
        raise StepError(
            "train_rows must contain at least two distinct valid example indices"
        )
    means, bases, spectrum = [], [], []
    for position in range(acts.shape[1]):
        mean, basis, rows = fit(acts[train_rows, position], int(inputs["k"]))
        means.append(mean)
        bases.append(basis)
        spectrum.extend({"position": position, **row} for row in rows)
    mean, weight = torch.stack(means), torch.stack(bases)
    coordinates = torch.einsum("npd,pdk->npk", acts.double() - mean, weight.double())
    write_tensor(outputs["mean"], mean, slot="mean")
    write_tensor(outputs["weight"], weight, slot="weight")
    write_tensor(outputs["coordinates"], coordinates, slot="coordinates")
    write_table(outputs["spectrum"], spectrum)
