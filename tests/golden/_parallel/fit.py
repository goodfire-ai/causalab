"""The parallel golden's fit documents (``docs/model_parallelism.md`` §7,
§10.6): the training smokes' two sharded-site fits retargeted to the
realization the way the inference document is —

- ``das``: the corpus DAS fit (``tests/protocols/04_das_im.json``) at
  ``attention_query`` — ``Sharded(head_axis, "tensor")`` under ``tp``, the
  featurizer's write between an all-gather and a chunk — held exact under
  ``pp=2`` (the gloo fp32 fit was byte-identical: the gradient crosses the
  stage boundary as one tensor) and banded under ``tp=2`` and ``dp=2:rows``
  (the rows split's loss mean is the one new reduction);
- ``dbm``: the expert-neuron DBM fit (``demos/methods/protocols/dbm_expert_neuron.json``)
  at ``expert_activation`` — ``ExpertLocal`` under ``ep`` — held exact under
  ``pp=2`` and banded under ``ep=2``;
- ``das_dense``: the same DAS fit on a dense fp32 model
  (``runs.DENSE``, ``Qwen/Qwen3-4B-Instruct-2507``) at ``tp=2`` and
  ``dp=2:rows``. Using fp32 reduces rounding noise relative to the small
  gradients and supports a tighter band than the bf16 A3B fit. The model
  ties its head to its embedding, which excludes pipeline parallelism. ``Document.realization`` selects the model and is
  recorded with the capture.

The documents set ``--fit-rows`` explicitly so free device memory cannot
change byte-compared receipts. They use a fixed step count: rounding near
an ``iia`` tie can otherwise make early stopping choose different updates.

The classes: ``bundle`` — the fitted featurizer files, fp32, absolute;
``tables`` — ``iia.json`` / ``ce.json``, metrics of the model's logits (its
dtype); ``eval`` — ``train_eval.json``, the eval score and pass count;
``diagnostics`` — ``fit_diagnostics.json`` (and ``routing_mismatch.json``
where the executor writes one); ``gradient`` — the featurizer's pre-mean
gradient at every step against world 1, relative to the step's largest
entry (the §7 invariant made a number; the recorder of `.recorder`);
``gradient_ranks`` — the same across the ranks, expected bit-identical.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tests._helpers.paths import PROTOCOLS_DIR
from tests.golden._parallel import measure
from tests.golden._parallel.runs import (
    DENSE,
    GRADIENT_CLASS,
    GRADIENTS,
    REPO,
    WORLD,
    Document,
    Realization,
    load_reports,
    output_files,
)

__all__ = [
    "CLASSES",
    "DAS",
    "DAS_DENSE",
    "DBM",
    "FIT_ROWS",
    "K",
    "STEPS",
    "UnclassifiedOutput",
]

DAS_CORPUS = REPO / "tests" / "protocols" / "04_das_im.json"
DBM_PRESET = PROTOCOLS_DIR / "dbm_expert_neuron.json"

#: The authored row bound, the DAS subspace's width (the corpus's, on a
#: query width of 4096 on the A3B and 256 on the tiny MoE) and the fixed
#: step count (the corpus DAS fit's: ten epochs of the four training rows,
#: one minibatch each).
FIT_ROWS = 16
K = 8
STEPS: dict[str, int] = {"epochs": 10}

#: Which class each saved table belongs to; a table the executor writes
#: that is not named here is refused, so a new output is classified on
#: purpose. ``diagnostics`` bands absolutely, the class of a continuous
#: statistic over a population; an **integer readout** in it — Boundless
#: DAS's ``hard_mask_size``, ``⌈θ·width⌉``, the fit's answer rather than a
#: statistic — would be pinned only to about ±3 by that band and needs its
#: own class (exact, or the routing table's banded fraction) before a
#: boundless document joins the record. None does today.
TABLES: dict[str, str] = {
    "iia.json": "tables",
    "ce.json": "tables",
    "train_eval.json": "eval",
    "fit_diagnostics.json": "diagnostics",
    "routing_mismatch.json": "diagnostics",
}
#: The featurizer precision the fits author (``train.precision.feature``).
FEATURE_DTYPE = "fp32"
GRADIENT = GRADIENT_CLASS
GRADIENT_RANKS = "gradient_ranks"
CLASSES = frozenset({"bundle", *TABLES.values(), GRADIENT, GRADIENT_RANKS})


class UnclassifiedOutput(ValueError):
    """A fit wrote a table the measure has no class for."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"the fit wrote {name!r}, which no class of tests/golden/_parallel/fit.py "
            f"names; classify it in TABLES"
        )


def _model(realization: Realization) -> dict[str, str]:
    return {"key": realization.model, "revision": "main", "dtype": realization.dtype}


def das_document(tmp: Path, realization: Realization) -> Path:
    """The corpus DAS fit at ``attention_query`` on the realization's layer,
    width `K`, `STEPS` fixed, no early stop."""
    doc = json.loads(DAS_CORPUS.read_text())
    doc["model"] = _model(realization)
    method = doc["method"]
    method["sites"]["target"] = {
        "component": "attention_query",
        "layers": [realization.layer],
    }
    method["featurizers"]["rot"]["k"] = K
    method["train"]["steps"] = dict(STEPS)
    method["train"].pop("early_stop", None)
    target = tmp / "das_attention_query.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def dbm_document(tmp: Path, realization: Realization) -> Path:
    """The expert-neuron DBM preset on the realization: the corpus DAS fit's
    data and eval split, the realization's layer, `STEPS` fixed."""
    doc = json.loads(DBM_PRESET.read_text())
    das = json.loads(DAS_CORPUS.read_text())
    doc["model"] = _model(realization)
    doc["data"] = das["data"]
    method = doc["method"]
    for name in ("routed", "shared"):
        method["sites"][name]["layers"] = [realization.layer]
    method["train"]["steps"] = dict(STEPS)
    method["train"]["eval"]["split"] = das["method"]["train"]["eval"]["split"]
    method["train"].pop("early_stop", None)
    target = tmp / "dbm_expert_activation.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _describe(protocol: str, site: str) -> Any:
    def describe(realization: Realization) -> dict[str, Any]:
        return {
            "protocol": protocol,
            "site": site,
            "layer": realization.layer,
            "steps": dict(STEPS),
            "early_stop": False,
            "fit_rows": FIT_ROWS,
            "feature_dtype": FEATURE_DTYPE,
        }

    return describe


def measure_run(
    solo: Path, parallel: Path, realization: Realization
) -> measure.Measured:
    """Every fitted bundle, every table by `TABLES`, and the recorded
    gradients against world 1 and across the ranks (module docstring)."""
    measured = measure.Measured()
    for name in output_files(solo):
        if not (parallel / name).exists():
            # A file missing from the parallel run fails parity in every class.
            kind = "bundle" if name.endswith(".safetensors") else TABLES.get(name)
            if kind is None:
                raise UnclassifiedOutput(name)
            measured.add(kind, name, measure.Measurement(math.inf, 0.0, FEATURE_DTYPE))
            continue
        if name.endswith(".safetensors"):
            measured.add("bundle", name, measure.tensor_file(solo, parallel, name))
            continue
        kind = TABLES.get(name)
        if kind is None:
            raise UnclassifiedOutput(name)
        dtype = FEATURE_DTYPE if kind == "diagnostics" else realization.dtype
        measured.add(kind, name, measure.table(solo, parallel, name, dtype))
    owners = _owners(parallel, realization.layer)
    against, across = measure.gradient_measurements(
        measure.load_gradients(solo / GRADIENTS, 0),
        [measure.load_gradients(parallel / GRADIENTS, rank) for rank in owners],
    )
    measured.add(GRADIENT, f"{GRADIENTS}/against-world-1", against)
    measured.add(GRADIENT_RANKS, f"{GRADIENTS}/across-ranks", across)
    return measured


def _owners(parallel: Path, layer: int) -> list[int]:
    """The ranks whose gradient is the featurizer's (§7): every rank that
    holds the tapped layer — all of them under ``tp`` / ``ep`` / ``dp``, the
    owning stage alone under ``pp``, read off the ranks' load reports. A
    stage that does not own the featurizer records no gradient for it, or
    only the regularizer's gradient. The owner's post-step sync overwrites
    that value before publication."""
    reports = load_reports(parallel, _recorded_ranks(parallel / GRADIENTS))
    marker = f".layers.{layer}."
    owners = [
        report["rank"]
        for report in reports
        if any(marker in name for name in report["bytes_requested"])
    ]
    return owners or [report["rank"] for report in reports]


def _recorded_ranks(directory: Path) -> int:
    """How many ranks recorded gradients: every rank of the world."""
    ranks = sorted(directory.glob("rank*.pt")) if directory.exists() else []
    return len(ranks) or WORLD


DAS = Document(
    name="das",
    exact=("pp=2",),
    banded=("tp=2", "dp=2:rows"),
    author=das_document,
    describe=_describe("04_das_im", "attention_query"),
    measure=measure_run,
    classes=CLASSES,
    argv=("--fit-rows", str(FIT_ROWS)),
    recorded=True,
)

DBM = Document(
    name="dbm",
    exact=("pp=2",),
    banded=("ep=2",),
    author=dbm_document,
    describe=_describe("dbm_expert_neuron", "expert_activation"),
    measure=measure_run,
    classes=CLASSES,
    argv=("--fit-rows", str(FIT_ROWS)),
    recorded=True,
)

DAS_DENSE = Document(
    name="das_dense",
    exact=(),
    banded=("tp=2", "dp=2:rows"),
    author=das_document,
    describe=_describe("04_das_im", "attention_query"),
    measure=measure_run,
    classes=CLASSES,
    argv=("--fit-rows", str(FIT_ROWS)),
    recorded=True,
    realization=DENSE,
)
