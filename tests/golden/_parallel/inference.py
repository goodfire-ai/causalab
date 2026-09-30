"""The parallel golden's inference document (``docs/model_parallelism.md``
§10.6): the tensor/expert smoke tier's boundary document — the corpus
interchange (``02_interchange_im.json``) extended to read and write every
module-boundary class the placement table names, the routing table read and
swapped, the ``expert:``-scoped face — retargeted to the realization (the
A3B in bf16; the tiny MoE is its four-layer, hidden-8 twin) with the target
swept over two full-attention layers, so ``dp=2`` over points has two points
to shard.

``dp=2`` and ``pp=2`` are held exact (§8: no reduction moves); ``tp=2`` and
``ep=2`` within the band of every output class the smoke tier defines
(``stream``, ``sharded_read``, ``sharded_write``, ``experts``) plus the
routing fraction (`ROUTING`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from tests.golden._parallel import measure
from tests.golden._parallel.bands import ROUTING
from tests.golden._parallel.runs import (
    LARGE_MODEL,
    Document,
    Realization,
    output_files,
)
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

__all__ = [
    "CLASSES",
    "DENSE_CLASSES",
    "DOCUMENT",
    "INTEGRAL",
    "SWEEP_STRIDES",
    "TABLES",
    "author",
    "author_for",
    "describe",
    "measure_for",
    "measure_run",
    "sweep",
]

CLASSES = boundary.CLASSES
INTEGRAL = boundary.INTEGRAL
TABLES = boundary.TABLES
#: The classes the boundary document has on a dense family (the smoke's
#: Llama variant): no experts, no routing table.
DENSE_CLASSES = frozenset({"stream", "sharded_read", "sharded_write"})

#: How far above the realization's layer the target sweep's second layer
#: sits: the next full-attention layer of the A3B's 3-linear-then-1-full
#: schedule by default; on the 70B forty layers up (3 → 43), so the two
#: points sit on different stages of every pipeline the large golden runs
#: (``pp=2`` stage 1, ``pp=4`` stage 2, ``pp=8`` stage 4) and a pipeline's
#: write lands on a stage other than the read's. The tiny Llama has two
#: layers: its sweep is both (the oracle path's gloo smoke,
#: ``tests/golden/test_parallel_large_harness.py``).
SWEEP_STRIDE = 4
SWEEP_STRIDES: dict[str, int] = {LARGE_MODEL: 40, TINY_LLAMA: 1}


def sweep(realization: Realization) -> tuple[int, int]:
    """The realization's layer and the one its stride above it
    (`SWEEP_STRIDES`, `SWEEP_STRIDE` otherwise) — clamped to the
    model: on a tower too short for the stride (the tiny fixtures the
    harness's own smokes run on) the second layer is the last one, or the
    layer below when the layer *is* the last."""
    from causalab.protocol.registry import get_model_info

    stride = SWEEP_STRIDES.get(realization.model, SWEEP_STRIDE)
    layer, last = realization.layer, get_model_info(realization.model).num_layers - 1
    if layer + stride <= last:
        return (layer, layer + stride)
    return (layer, last) if layer < last else (layer - 1, layer)


def author_for(fixture: str) -> Callable[[Path, Realization], Path]:
    """The author of the boundary document written for ``fixture`` — the
    tiny MoE's (the routing table, the ``expert:`` face) for the A3B, the
    tiny Llama's (the module-boundary and attention-interior classes alone)
    for a dense family — retargeted to the realization."""

    def author(tmp: Path, realization: Realization) -> Path:
        """The smoke tier's boundary document on the realization, the target
        swept over [`sweep`][causalab.neural.shared.sweep], every other site at its layer."""
        target = boundary._document(tmp, fixture)  # pyright: ignore[reportPrivateUsage]
        doc = json.loads(target.read_text())
        doc["model"] = {
            "key": realization.model,
            "revision": "main",
            "dtype": realization.dtype,
        }
        doc["method"]["sites"]["target"]["layers"] = {"sweep": list(sweep(realization))}
        for name, site in doc["method"]["sites"].items():
            if name != "target" and "layers" in site:
                site["layers"] = [realization.layer]
        target.write_text(json.dumps(doc, indent=2))
        return target

    return author


def author(
    tmp: Path, realization: Realization, template: str = TINY_QWEN35_MOE
) -> Path:
    """`author_for` ``template`` applied — the A3B tier's author by
    default; the large model's passes the tiny Llama."""
    return author_for(template)(tmp, realization)


def describe(realization: Realization) -> dict[str, Any]:
    return {
        "protocol": "02_interchange_im + the tp/ep smoke tier's boundary reads and writes",
        "layer": realization.layer,
        "sweep": list(sweep(realization)),
    }


def measure_for(
    integral: Sequence[str],
) -> Callable[[Path, Path, Realization], measure.Measured]:
    """The measure of a boundary document whose integral (routing) files are
    ``integral`` — measured only where the document wrote them: none for a
    dense family, whose document has no routing table and so no routing
    class, and none when the run saved no such file."""

    def measure_run(
        solo: Path, parallel: Path, realization: Realization
    ) -> measure.Measured:
        """Every saved tensor file by its class, the metric tables as
        ``stream`` (the model's dtype: they are metrics of its logits), the
        integral files as the routing fraction."""
        measured = measure.Measured()
        if integral and all(
            (solo / f"{name}.safetensors").exists() for name in integral
        ):
            measured.classes[ROUTING] = measure.routing_fraction(
                solo, parallel, integral
            )
        for name in output_files(solo):
            stem = name.removesuffix(".safetensors")
            if name.endswith(".safetensors") and stem not in integral:
                measured.add(
                    CLASSES[stem], name, measure.tensor_file(solo, parallel, name)
                )
        for name in TABLES:
            measured.add(
                "stream", name, measure.table(solo, parallel, name, realization.dtype)
            )
        return measured

    return measure_run


measure_run = measure_for(INTEGRAL)


DOCUMENT = Document(
    name="inference",
    exact=("dp=2", "pp=2"),
    banded=("tp=2", "ep=2"),
    author=author,
    describe=describe,
    measure=measure_run,
    classes=frozenset(CLASSES.values()) | {ROUTING},
)
