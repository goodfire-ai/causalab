"""Write a run's declared results to disk.

Metric tables use JSON. Tensor outputs use safetensors with artifact identity
metadata. The save manifest records the products requested by the document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from causalab.io.tables import write_table
from causalab.protocol.identity import build_artifact_identity

if TYPE_CHECKING:
    from causalab.neural.shared.results import MetricTable, TensorFile

__all__ = [
    "FIT_DIAGNOSTICS_FILE",
    "ROUTING_MISMATCH_FILE",
    "TRAIN_EVAL_FILE",
    "write_outputs",
]


#: Where a run writes its ``train.eval`` scores. A sibling of the save
#: manifest's own files, never a column inside one: the eval score is measured
#: on a different split, so it is a different population from the metric rows
#: and does not belong in the same table (spec §2.12).
TRAIN_EVAL_FILE = "train_eval.json"


#: Where a run writes what each fit can say about *itself*. A separate file
#: from the trained bundle because the bundle's metadata is a closed identity
#: schema, and separate from the metric table because these are properties of
#: a parameter, not of an example.
FIT_DIAGNOSTICS_FILE = "fit_diagnostics.json"


#: The routing-mismatch table of every write through an
#: expert-keyed gate (spec §2.5 ``expert_neuron``): per point, write, layer and
#: example, how many of the base slots held an expert the operand's side never
#: activated — and so kept their base value — out of the slots addressed. A
#: property of the (base, counterfactual) pair's routing, not of a parameter,
#: so it sits beside [`FIT_DIAGNOSTICS_FILE`][] rather than inside it.
ROUTING_MISMATCH_FILE = "routing_mismatch.json"


def write_outputs(
    output_dir: Path,
    tensor_files: Mapping[str, TensorFile],
    metric_files: Mapping[str, MetricTable],
    *,
    identity_base: Mapping[str, Any],
    train_evals: Sequence[Mapping[str, Any]] = (),
    fit_diagnostics: Sequence[Mapping[str, Any]] = (),
    routing_mismatch: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Path]:
    """Write every accumulated save file under ``output_dir``; returns
    manifest path → absolute path.

    ``train_evals`` is one record per point that declared ``train.eval`` — the
    held-out score the fit was selected by. It is written to
    [`TRAIN_EVAL_FILE`][] only when there is something to write, so a run
    with no fit produces no empty file; ``fit_diagnostics`` and
    ``routing_mismatch`` follow the same rule.
    """
    from causalab.io.tensor_files import save_file

    written: dict[str, Path] = {}
    for rel, tensors in tensor_files.items():
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = build_artifact_identity(**identity_base)
        metadata.update(tensors.metadata)
        metadata["entries"] = json.dumps(tensors.entry_meta, sort_keys=True)
        save_file(tensors.entries, str(target), metadata=metadata)
        written[rel] = target
    for rel, table in metric_files.items():
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        write_table(target, table.rows)
        written[rel] = target
    for rel, records in (
        (TRAIN_EVAL_FILE, train_evals),
        (FIT_DIAGNOSTICS_FILE, fit_diagnostics),
        (ROUTING_MISMATCH_FILE, routing_mismatch),
    ):
        if not records:
            continue
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(list(records), indent=2) + "\n")
        written[rel] = target
    return written
