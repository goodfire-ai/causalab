"""Gather aligned prediction activations from a shared all-position harvest.

Inputs: acts (a single-entry harvest path or tensor), rows (the JSON table
prepare_sequence authored, or its rows). Output: acts (examples, targets,
features). Works with the protocol's dense and flattened ragged all-position
saves. A ragged harvest's row boundaries come from its own ``.widths`` sidecar
(the per-row token counts the engine recorded); padding never enters the
saved all read. A tensor handed in directly must be dense, one row per example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_table, read_tensor, write_tensor


def _read_harvest(path: Path) -> tuple[Any, Any]:
    """The single-entry harvest and its ragged sidecar (``None`` when dense).

    ``read_tensor`` checks the path, refuses an ambiguous bundle and reads the
    harvest itself; one non-sidecar key is therefore left in the file, and its
    ``.widths`` sidecar — if the engine wrote one — is read through the header
    without reading the harvest a second time."""
    from causalab.io.tensor_files import safe_open
    from causalab.protocol.bundles import RAGGED_SUFFIX

    acts = read_tensor(path)
    with safe_open(str(path)) as bundle:
        keys = bundle.keys()
        [key] = [k for k in keys if not k.endswith(RAGGED_SUFFIX)]
        sidecar = f"{key}{RAGGED_SUFFIX}"
        return acts, bundle.get_tensor(sidecar) if sidecar in keys else None


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    acts, rows = inputs["acts"], inputs["rows"]
    widths = None
    if isinstance(acts, (str, Path)):
        acts, widths = _read_harvest(Path(acts))
    if isinstance(rows, (str, Path)):
        rows = read_table(Path(rows))
    targets = [len(row["targets"]) for row in rows]
    if not rows or len(set(targets)) != 1 or targets[0] == 0:
        raise StepError("sequence_activations needs a nonempty aligned target cohort")
    if widths is None:
        if acts.ndim != 3 or acts.shape[0] != len(rows):
            raise StepError("dense harvest shape does not match the rows")
        windows = acts.unbind(0)
    else:
        counts = [int(n) for n in widths.tolist()]
        if acts.ndim != 2 or len(counts) != len(rows) or acts.shape[0] != sum(counts):
            raise StepError("ragged harvest shape does not match the rows")
        windows = acts.split(counts)
    aligned = []
    for row, window in zip(rows, windows):
        indices = [target["prediction_position"] for target in row["targets"]]
        if any(type(i) is not int or not 0 <= i < len(window) for i in indices):
            raise StepError("target prediction position is outside its sequence")
        aligned.append(window[indices])
    write_tensor(outputs["acts"], torch.stack(aligned), slot="acts")
