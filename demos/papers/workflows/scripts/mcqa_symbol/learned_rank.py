"""The rank a DBM-DAS fit learned, as a value the next step can read.

A ``boundary`` gate stores one ``theta`` in ``[0, 1]``, the boundary as a
fraction of the rotation's columns. Its eval mask keeps the columns
``i < theta * width`` (``Gate.hard_mask`` in
``causalab/neural/shared/featurizers/gate.py``), so the learned rank is
``ceil(theta * width)``, the ``hard_mask_size`` the fit also records in its
``fit_diagnostics.json``. A workflow step cannot read that sidecar, so this
step recomputes the rank from the two bundles the fit saves: ``theta`` from
the gate and ``width`` from the rotation's column count. The workflow sets the
random control's ``k`` to ``learned_rank``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_tensor, write_values

__all__ = ["main", "learned_rank"]


def learned_rank(theta: float, width: int) -> int:
    """The number of columns ``i = 0 ... width - 1`` with ``i < theta * width``."""
    if not 0.0 <= theta <= 1.0:
        raise StepError(f"a boundary theta lies in [0, 1], got {theta}")
    return sum(1 for i in range(width) if i < theta * width)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    theta = read_tensor(Path(inputs["gate"]), slot="theta").reshape(-1)
    if theta.numel() != 1:
        raise StepError(f"a boundary gate holds one theta, got {theta.numel()}")
    weight = read_tensor(Path(inputs["rotation"]), slot="weight")
    width = int(weight.shape[-1])
    value = float(theta[0])
    write_values(
        Path(outputs["values"]),
        {
            "learned_rank": learned_rank(value, width),
            "theta": value,
            "width": width,
        },
    )
