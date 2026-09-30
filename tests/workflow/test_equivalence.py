"""Typed site-matched parent controls — the site-equivalence predicate and
workflow rule 16 (workflow spec §2.2, §5 rule 16, §8).

What is pinned, and how each test fails without the change:

* **The vocabulary.** The spec's equivalence-field table is exactly
  ``EQUIVALENCE_FIELDS``; ``full_component`` is a control kind (a coverage kind,
  never a required one) whose row carries the measured-ceiling sentence; rule
  16 is the equivalence rule. Without the change the module does not exist and
  this file fails at import.
* **The guards.** ``causalab/protocol/equivalence.py`` imports no numerics
  (statically, and in a subprocess), is a member of no shipped script's import
  closure, and is imported by no shared or hashed module — only by the
  workflow document model. A closure move would move every shipped workflow
  digest; ``tests/workflow/test_closure_census.py`` would catch it and this
  file says why.
* **T13, five refusals, one per field.** A ``full_component`` control against a
  target differing in layer coverage (``layers``), pre/post-projection site
  (``component`` *and* ``shape`` — the mutation "compare component labels, not
  shapes" leaves ``shape`` unnamed and fails here), DeltaNet inclusion
  (``stream``, on an in-test hybrid ``ModelInfo`` with ``layer_types``; the
  same mutation loses the stream coverage and fails here too), routed-rank
  identity (``expert``; ``routed_rank`` for two routings) and coordinate
  sharing (T14). Each passes once ``non_equivalence`` names the field, an
  undeclared second field still refuses, a stale declaration refuses.
* **T14, sharing is decided.** Inside one document two writes naming one
  featurizer are ``shared`` and two same-shaped featurizers are ``distinct``
  (the mutation "compare widths only" calls them shared and fails); across
  documents a control loading the fit's own bundle is refused under rule 16
  naming ``sharing`` and passes it once declared.
* **T15, the legitimate campaign.** ``13_random_subspace_control_im.json``
  against ``04_das_im.json`` is equivalent with no declaration; the corpus
  pins are unchanged; ``seed`` and the bundle bytes are not in the tuple (the
  mutation "include the seed" refuses the pair and fails here). The two
  documents differ in ``model.dtype`` (absent → fp32 vs bf16): that is the
  model realization, a separate clause, and the tuple has no ``model.*`` field.

All offline: the predicate is arithmetic on compiled points and ``ModelInfo``;
no model, no tokenizer, no weights are loaded.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.identity import import_closure
from causalab.protocol.equivalence import (
    EQUIVALENCE_FIELDS,
    FeaturizerStage,
    SiteTuple,
    compare,
    coverage,
    explain,
    sharing,
    site_tuple,
)
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry import ModelInfo, get_model_info
from causalab.io.env import ResolutionEnv
from causalab.protocol.schema import METHOD_SECTIONS, PROTOCOL_VERSION, parse_document
from causalab.workflow.document import (
    CONTROL_KINDS,
    CONTROL_RULE,
    COVERAGE_KINDS,
    EQUIVALENCE_RULE,
    MAX_RULE,
    REQUIRED_CONTROL_KINDS,
    WorkflowError,
    load_workflow,
)

from tests.protocol.test_vocabulary_census import HASHED_SCRIPTS
from tests.test_architecture_layering import (
    HEAVY_MODULES,
    _module_level_imports,  # pyright: ignore[reportPrivateUsage]
)
from tests.protocol._env import steps_of
from tests.workflow.test_closure_census import CLOSURES, REDUCE, REDUCE_CLOSURE, SHARED
from tests.workflow.test_controls import (
    CONTROL_13,
    CORPUS_PINS,
    FIT,
    FIXTURES as CONTROL_FIXTURES,
    _members,  # pyright: ignore[reportPrivateUsage]
    _section,  # pyright: ignore[reportPrivateUsage]
    _table,  # pyright: ignore[reportPrivateUsage]
)
from tests.protocol._docs import saved, term

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
MODULE = "causalab/protocol/equivalence.py"
WORKFLOW_DOCUMENT = "causalab/workflow/document.py"

# --------------------------------------------------------------------------- #
# the vocabulary, held to the spec
# --------------------------------------------------------------------------- #


def test_the_equivalence_fields_match_the_spec_table() -> None:
    tabulated = _members("equivalence field")
    assert len(tabulated) == len(EQUIVALENCE_FIELDS) == 10
    assert tabulated == list(EQUIVALENCE_FIELDS), (
        "§2.2's equivalence-field table and EQUIVALENCE_FIELDS disagree (order "
        f"included — a refusal names fields in this order): {tabulated}"
    )
    for row in _table("equivalence field"):
        assert len(row) >= 3 and len(row[1]) > 20, f"row {row[0]} is bare"
    # the `featurizer` row enumerates the stage tuple — a second copy of the
    # field list, held to the first: every `FeaturizerStage` field is named
    # in it. `explain`'s legend is the third copy and spells the fields in
    # prose ("axis and its unit count"), so this census does not reach it
    described = _table("equivalence field")[tabulated.index("featurizer")][1]
    for field in dataclasses.fields(FeaturizerStage):
        assert f"`{field.name}`" in described, (
            f"§2.2's `featurizer` row does not name stage field {field.name!r}"
        )


def test_the_tuple_has_exactly_the_vocabulary_fields() -> None:
    """The dataclass and the closed vocabulary are one list: a field added to
    the tuple without a table row, or the reverse, fails here."""
    assert tuple(f.name for f in dataclasses.fields(SiteTuple)) == EQUIVALENCE_FIELDS


def test_full_component_is_a_coverage_kind_and_never_required() -> None:
    assert "full_component" in CONTROL_KINDS
    assert set(COVERAGE_KINDS) == {"full_component", "matched_random"}
    assert "full_component" not in REQUIRED_CONTROL_KINDS  # I5
    assert "self_swap" not in COVERAGE_KINDS  # per-point agreement is §8's
    assert set(_members("kind")) == set(CONTROL_KINDS)
    row = next(r for r in _table("kind") if "full_component" in r[0])
    text = " ".join(row[1:])
    assert "the full-component score provides a comparison" in text
    assert "a sparse mask may score higher" in text
    assert "`passed` when the swap ran" in text


def test_rule_16_is_the_equivalence_rule() -> None:
    """Numbered by rule, so a renumbering is a deliberate edit here. Rule 16
    is this contract's; later rules raise ``MAX_RULE`` and are their own
    PR's, so the ceiling is a floor here (the `test_controls.py` shape)."""
    assert MAX_RULE >= EQUIVALENCE_RULE == 16 and CONTROL_RULE == 14
    section = _section("## 5. Validation")
    item = re.search(r"^16\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S)
    assert item is not None
    text = item.group(1)
    assert "site-equivalent" in text and "`non_equivalence" in text
    for field in EQUIVALENCE_FIELDS:
        assert f"`{field}`" in text, field
    assert "`self_swap`" in text and "`full_component`" in text


# --------------------------------------------------------------------------- #
# the guards: torch-free, in no closure, imported only by the workflow layer
# --------------------------------------------------------------------------- #


def test_the_module_is_torch_free_at_module_level() -> None:
    heavy = [
        (line, module)
        for line, module in _module_level_imports(REPO / MODULE)
        if module.split(".")[0] in HEAVY_MODULES
    ]
    assert not heavy, heavy


_IMPORT_PROBE = """
import importlib, json, sys

importlib.import_module("causalab.protocol.equivalence")
heavy = sorted(m for m in ("torch", "numpy", "pandas", "matplotlib", "scipy",
                           "sklearn", "safetensors", "transformers")
               if m in sys.modules)
print(json.dumps({"heavy": heavy}))
"""


def test_the_module_imports_no_numerics_in_a_subprocess() -> None:
    """A subprocess, as in ``tests/protocol/test_load_is_torch_free.py``:
    ``tests/conftest.py`` has already imported torch in this process."""
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    heavy = json.loads(completed.stdout.strip().splitlines()[-1])["heavy"]
    assert not heavy, f"importing causalab.protocol.equivalence pulls {heavy}"


def _imports_of(path: Path) -> set[str]:
    """Every module an import statement in ``path`` names, function-local
    ones included (``import_closure`` counts those too)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return out


def test_the_module_is_in_no_closure_and_only_the_workflow_document_imports_it() -> (
    None
):
    """The module stays digest-neutral, pinned three ways: it is a member of no
    frozen closure; its own closure stays inside the shared protocol core (so
    even if a script came to import it, no *new* module would become
    digest-bearing); and among every module under ``causalab/`` exactly
    ``workflow/document.py`` imports it — never a SHARED member, a hashed
    script, ``compile.py``, ``loader.py`` or ``run.py``."""
    for members in (*CLOSURES.values(), REDUCE_CLOSURE):
        assert MODULE not in members
    assert MODULE not in SHARED
    own = set(import_closure(REPO / MODULE, root=REPO))
    assert own <= set(SHARED), sorted(own - set(SHARED))
    importers = sorted(
        str(path.relative_to(REPO))
        for path in (REPO / "causalab").rglob("*.py")
        if "causalab.protocol.equivalence" in _imports_of(path)
    )
    assert importers == [WORKFLOW_DOCUMENT], importers
    for module in (*SHARED, *HASHED_SCRIPTS, REDUCE):
        assert "causalab.protocol.equivalence" not in _imports_of(REPO / module), module


# --------------------------------------------------------------------------- #
# in-test models and documents — offline, no weights
# --------------------------------------------------------------------------- #

#: A dense tower: no ``layer_types``, no linear-attention mixer, so every layer
#: carries the softmax mixer — decided from the entry, not guessed.
DENSE = ModelInfo(
    key="test/dense",
    hidden_size=32,
    num_layers=2,
    num_heads=4,
    num_kv_heads=4,
    head_dim=8,
    intermediate_size=64,
    vocab_size=128,
)

#: A hybrid MoE tower: layer 0 full attention, layer 1 Gated DeltaNet (the
#: four linear widths declared), eight experts routed top-2 — the shape of
#: the Hugging Face Hub checkpoint ``tiny-random/qwen3.5-moe`` and of
#: Qwen3.6-35B-A3B, registered in-test (tiny keys need a registry entry at
#: load).
HYBRID = ModelInfo(
    key="test/hybrid",
    hidden_size=32,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    intermediate_size=64,
    vocab_size=128,
    num_experts=8,
    num_experts_per_tok=2,
    shared_expert_intermediate_size=16,
    moe_intermediate_size=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    layer_types=("full_attention", "linear_attention"),
)
INFOS = {info.key: info for info in (DENSE, HYBRID)}

SUBSPACE = {"kind": "subspace", "k": 4, "parametrization": "cayley"}
GATE = {"kind": "gate"}
EXTERNAL = {"reason": "external", "reference": "runs/2026-09-04/controls"}
#: a target that trains nothing waives the two required kinds honestly
WAIVE_TARGET = {"self_swap": EXTERNAL, "matched_random": "no_fit"}
REASON = "the control covers the readout cell only; the comparison is per layer"
#: the two aggregations ``_document`` reduces its patched logits to
IIA = {
    "kind": "logit_diff",
    "a": "cf_answer",
    "b": "base_answer",
}
CE = {"kind": "cross_entropy", "target": "label"}


def _env(env: Any) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=env.datasets, artifacts=env.artifacts, model_info=INFOS.__getitem__
    )


def _document(
    model: str,
    site: dict[str, Any],
    *,
    featurizer: dict[str, Any] | None = None,
    dims: list[int] | None = None,
    train: bool = False,
) -> dict[str, Any]:
    """An interchange at ``site`` (optionally through featurizer ``rot``,
    optionally a fit of it saving ``rot.safetensors``) on the fixture rows."""
    read: dict[str, Any] = {"site": "target", "pos": -1}
    write: dict[str, Any] = {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}
    featurizers: dict[str, Any] = {}
    if featurizer is not None:
        featurizers["rot"] = featurizer
        read["featurizer"] = write["featurizer"] = "rot"
    if dims is not None:
        read["dims"] = write["dims"] = dims
    method: dict[str, Any] = {
        "intervened_models": {
            "original_counterfactual": {"input": "counterfactual", "reads": ["v_cf"]},
            "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
        },
        "sites": {"target": site, "lm_head": {"component": "lm_head"}},
        "featurizers": featurizers,
        "reads": {"v_cf": read, "logits": {"site": "lm_head", "pos": -1}},
        "writes": {"patch": write},
        "save": [saved("logits", "patched", "iia.json", dict(IIA))],
    }
    if train:
        method["train"] = {
            "objective": [[1.0, term("logits", "patched", dict(CE))]],
            "params": ["rot"],
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
            "steps": {"epochs": 1},
            "batch": {"pairs": 2},
            "precision": {"feature": "fp32", "loss": "fp32"},
            "eval": {
                "every": {"epochs": 1},
                "split": "weekdays/data#test",
                "aggregations": {"iia": term("logits", "patched", dict(IIA))},
            },
            "seed": 0,
        }
        method["save"].append(saved("logits", "patched", "ce.json", dict(CE)))
        method["save"].append(
            {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
        )
        method = {key: method[key] for key in METHOD_SECTIONS if key in method}
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": model},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": method,
    }


def _write(tmp_path: Path, name: str, document: dict[str, Any]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document, indent=1))
    return path


def _workflow(steps: dict[str, Any]) -> dict[str, Any]:
    return {"version": "1", "output_dir": "equivalence", "steps": steps}


def _step(document: Path, **fields: Any) -> dict[str, Any]:
    return {"type": "intervention_protocol", "document": str(document), **fields}


def _control(
    kind: str, *, declare: tuple[str, ...] = (), **fields: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {"of": "fit", "kind": kind, **fields}
    if declare:
        out["non_equivalence"] = {"fields": list(declare), "reason": REASON}
    return out


def _refused(
    raw: dict[str, Any], env: Any, tmp_path: Path, rule: int, match: str
) -> WorkflowError:
    with pytest.raises(WorkflowError) as info:
        load_workflow(raw, _env(env), workflow_dir=tmp_path)
    assert info.value.rule == rule, str(info.value)
    assert re.search(match, str(info.value)), str(info.value)
    return info.value


def _loads(raw: dict[str, Any], env: Any, tmp_path: Path) -> dict[str, Any]:
    loaded = load_workflow(raw, _env(env), workflow_dir=tmp_path)
    return dict(loaded.equivalence["ctl"])


# --------------------------------------------------------------------------- #
# T13 — five refusals, one per field, each lifted by a declaration
# --------------------------------------------------------------------------- #


def test_t13_layer_coverage_is_refused_and_declared(env, tmp_path) -> None:
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": {"sweep": [0, 1]}},
            featurizer=SUBSPACE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document(
            "test/dense", {"component": "block_output", "layers": {"sweep": [0, 1]}}
        ),
    )
    pinned = {"sites.target.layers": 0}
    error = _refused(
        _workflow(
            {
                "fit": _step(fit, waive=WAIVE_TARGET),
                "ctl": _step(ctl, set=pinned, control=_control("full_component")),
            }
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"control 'ctl' and its target 'fit' differ in layers: layers \[0\] vs \[0, 1\]",
    )
    assert error.path == "steps.ctl.control"
    assert "non_equivalence: {fields: [layers], reason: …}" in str(error)
    verdict = _loads(
        _workflow(
            {
                "fit": _step(fit, waive=WAIVE_TARGET),
                "ctl": _step(
                    ctl,
                    set=pinned,
                    control=_control("full_component", declare=("layers",)),
                ),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict["status"] == "declared" and "layers" in verdict["fields"]
    # the same control covering both layers is equivalent — the featurizer
    # (and so the coordinates) is the kind and needs no declaration
    verdict = _loads(
        _workflow(
            {
                "fit": _step(fit, waive=WAIVE_TARGET),
                "ctl": _step(ctl, control=_control("full_component")),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict == {
        "status": "equivalent",
        "fields": ["featurizer", "dims"],
        "sharing": "distinct",
    }


def test_a_band_is_one_site_in_the_tuple() -> None:
    """Protocol v3 (9·1): ``SiteSpec.layers`` is a band, a tuple of layer
    indices, and the tuple carries it whole — ``layers: 0`` and ``layers: [0]``
    are one spelling, ``layers: [0, 1]`` is one two-layer site, never two."""
    for spelling, band in ((0, (0,)), ([0], (0,)), ([0, 1], (0, 1))):
        doc = parse_document(
            _document("test/dense", {"component": "block_output", "layers": spelling})
        )
        assert site_tuple(doc, "patch", DENSE).layers == band, spelling
        assert isinstance(doc.sites["target"].layers, tuple)


def test_t13_a_pinned_layer_against_the_same_pinned_layer_is_equivalent(
    env, tmp_path
) -> None:
    """(a) the valid-work twin: one layer pinned on both sides, in either
    spelling of the one-layer band."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
        ),
    )
    for spelling in (0, [0]):
        ctl = _write(
            tmp_path,
            "ctl",
            _document("test/dense", {"component": "block_output", "layers": spelling}),
        )
        verdict = _loads(
            _workflow(
                {
                    "fit": _step(fit, waive=WAIVE_TARGET),
                    "ctl": _step(ctl, control=_control("full_component")),
                }
            ),
            env,
            tmp_path,
        )
        assert verdict == {
            "status": "equivalent",
            "fields": ["featurizer", "dims"],
            "sharing": "distinct",
        }, spelling


def test_t13_a_sweep_against_the_same_sweep_is_equivalent(env, tmp_path) -> None:
    """(b) the valid-work twin: both sides swept over the same layers — the
    coverages are the same set of one-layer bands, ``{(0,), (1,)}``."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": {"sweep": [0, 1]}},
            featurizer=SUBSPACE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document(
            "test/dense", {"component": "block_output", "layers": {"sweep": [0, 1]}}
        ),
    )
    verdict = _loads(
        _workflow(
            {
                "fit": _step(fit, waive=WAIVE_TARGET),
                "ctl": _step(ctl, control=_control("full_component")),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict == {
        "status": "equivalent",
        "fields": ["featurizer", "dims"],
        "sharing": "distinct",
    }


def test_t13_a_band_against_a_per_layer_sweep_is_refused_and_declared(
    env, tmp_path
) -> None:
    """(c) a target authored ``layers: [0, 1]`` — one band, one intervention
    across both layers — against a control swept over 0, 1 — two points, one
    layer each — is coverage-equal and intervention-different: refused under
    rule 16 naming ``layers`` and both spellings; (d) lifted by a
    ``non_equivalence`` declaration on ``layers`` like every other field.
    Mutation: flattening the coverage to a union of layer indices in
    ``_layer_coverage`` makes the two sides equal and the refusal never
    fires (DID NOT RAISE) — this test alone is red."""
    band = _write(
        tmp_path,
        "fit",
        _document("test/dense", {"component": "block_output", "layers": [0, 1]}),
    )
    swept = _write(
        tmp_path,
        "ctl",
        _document(
            "test/dense", {"component": "block_output", "layers": {"sweep": [0, 1]}}
        ),
    )
    error = _refused(
        _workflow(
            {
                "fit": _step(band, waive=WAIVE_TARGET),
                "ctl": _step(swept, control=_control("full_component")),
            }
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"control 'ctl' and its target 'fit' differ in layers: "
        r"layers \[0, 1\] vs \[\[0, 1\]\] — the same layers, not the same sites",
    )
    assert error.path == "steps.ctl.control"
    assert "visits 0, 1 one layer per point" in str(error)
    assert "spans the band [0, 1] as one site" in str(error)
    assert "coverage-equal, intervention-different" in str(error)
    assert "non_equivalence: {fields: [layers], reason: …}" in str(error)
    # the pair is the same difference the other way round: a band control
    # against a swept target
    error = _refused(
        _workflow(
            {
                "fit": _step(swept, waive=WAIVE_TARGET),
                "ctl": _step(band, control=_control("full_component")),
            }
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in layers: layers \[\[0, 1\]\] vs \[0, 1\] — the same layers",
    )
    # the coverages, as the predicate sees them: a set of bands each
    band_cov = coverage(
        [parse_document(json.loads(band.read_text()))], DENSE, owner="fit"
    )
    assert {t.layers for t in band_cov} == {(0, 1)}
    # (d) declared, the load moves on and the record says so
    verdict = _loads(
        _workflow(
            {
                "fit": _step(band, waive=WAIVE_TARGET),
                "ctl": _step(
                    swept, control=_control("full_component", declare=("layers",))
                ),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict == {
        "status": "declared",
        "fields": ["layers"],
        "sharing": "distinct",
    }


def test_t13_pre_post_projection_site_is_refused_on_component_and_shape(
    env, tmp_path
) -> None:
    """``attention_premix`` (the o-projection's input, head space) against
    ``attention_output`` (the residual stream) at one layer: two fields, and
    declaring the label alone leaves the shape refused. Mutation: comparing
    component labels rather than shapes never names ``shape`` — the second
    refusal below is what fails."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "attention_premix", "layers": 0},
            featurizer=SUBSPACE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document("test/dense", {"component": "attention_output", "layers": 0}),
    )
    steps = {"fit": _step(fit, waive=WAIVE_TARGET)}
    error = _refused(
        _workflow({**steps, "ctl": _step(ctl, control=_control("full_component"))}),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in component: one writes \['attention_output'\], the other "
        r"\['attention_premix'\]",
    )
    assert "(batch, position, feature)" in str(error)
    assert "(batch, position, head·feature)" in str(error)
    _refused(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl, control=_control("full_component", declare=("component",))
                ),
            }
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in shape: 'attention_output' is \(batch, position, feature\); "
        r"'attention_premix' is \(batch, position, head·feature\)",
    )
    verdict = _loads(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl,
                    control=_control("full_component", declare=("component", "shape")),
                ),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict["status"] == "declared"
    assert verdict["fields"][:2] == ["component", "shape"]


def test_t13_deltanet_inclusion_is_refused_on_stream(env, tmp_path) -> None:
    """On the hybrid tower the fit sweeps a full-attention and a DeltaNet
    layer; the control pinned to the full-attention one differs in ``layers``
    *and* in ``stream``, because the fit includes a DeltaNet layer. Mutation:
    dropping the stream coverage leaves the declared-``layers`` form loading."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/hybrid",
            {"component": "block_output", "layers": {"sweep": [0, 1]}},
            featurizer=SUBSPACE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document(
            "test/hybrid", {"component": "block_output", "layers": {"sweep": [0, 1]}}
        ),
    )
    steps = {"fit": _step(fit, waive=WAIVE_TARGET)}
    pinned = {"sites.target.layers": 0}
    _refused(
        _workflow(
            {**steps, "ctl": _step(ctl, set=pinned, control=_control("full_component"))}
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in layers",
    )
    error = _refused(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl,
                    set=pinned,
                    control=_control("full_component", declare=("layers",)),
                ),
            }
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in stream: mixer streams \['full_attention'\] vs "
        r"\['full_attention', 'linear_attention'\]",
    )
    assert "DeltaNet inclusion" in str(error)
    verdict = _loads(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl,
                    set=pinned,
                    control=_control("full_component", declare=("layers", "stream")),
                ),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict["status"] == "declared"
    assert {"layers", "stream"} <= set(verdict["fields"])


def test_t13_routed_rank_identity_is_refused_on_expert(env, tmp_path) -> None:
    """Inside the routed experts the ``expert`` a site names is the identity;
    two controls at two experts are not one comparison."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/hybrid",
            {"component": "expert_activation", "layers": 0, "expert": 1},
            featurizer=GATE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document(
            "test/hybrid", {"component": "expert_activation", "layers": 0, "expert": 0}
        ),
    )
    steps = {"fit": _step(fit, waive=WAIVE_TARGET)}
    _refused(
        _workflow({**steps, "ctl": _step(ctl, control=_control("full_component"))}),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in expert: expert \[0\] vs \[1\]",
    )
    verdict = _loads(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl, control=_control("full_component", declare=("expert",))
                ),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict["status"] == "declared" and "expert" in verdict["fields"]
    # the routed vs the shared expert: a different component, shape and
    # expert axis; the routing triple is the model's and agrees
    shared = _write(
        tmp_path,
        "shared",
        _document(
            "test/hybrid", {"component": "shared_expert_activation", "layers": 0}
        ),
    )
    error = _refused(
        _workflow({**steps, "ctl": _step(shared, control=_control("full_component"))}),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in component: one writes \['shared_expert_activation'\]",
    )
    assert "topk·feature" in str(error)


def test_the_routed_rank_field_is_the_models_routing_triple() -> None:
    """The routing identity ``(num_experts, top_k, moe_inner)`` rides on every
    MoE-block component and on nothing else; two registry entries routing
    differently are two routed ranks at one site."""
    raw = _document(
        "test/hybrid", {"component": "expert_activation", "layers": 0, "expert": 0}
    )
    doc = parse_document(raw)
    at_expert = site_tuple(doc, "patch", HYBRID)
    assert at_expert.routed_rank == (8, 2, 16) and at_expert.expert == 0
    residual = parse_document(
        _document("test/hybrid", {"component": "block_output", "layers": 0})
    )
    assert site_tuple(residual, "patch", HYBRID).routed_rank is None
    rerouted = dataclasses.replace(HYBRID, key="test/hybrid-4", num_experts=4)
    fields = compare([at_expert], [site_tuple(doc, "patch", rerouted)])
    assert fields == ("routed_rank",)
    assert "(8, 2, 16)" in explain(
        "routed_rank", [at_expert], [site_tuple(doc, "patch", rerouted)]
    )


def test_a_head_on_a_headless_component_is_explained_by_the_shape() -> None:
    """The ``head`` explanation reuses the registry's own no-axis prose, so a
    refusal here and the §2.2 head refusal cannot describe the absent axis
    differently."""
    with_head = parse_document(
        _document(
            "test/dense", {"component": "attention_premix", "layers": 0, "head": 1}
        )
    )
    without = parse_document(
        _document("test/dense", {"component": "block_output", "layers": 0})
    )
    a, b = site_tuple(with_head, "patch", DENSE), site_tuple(without, "patch", DENSE)
    assert "head" in compare([a], [b])
    text = explain("head", [a], [b])
    assert "component 'block_output' has no head axis" in text


def test_a_layerless_component_covers_no_layer_and_no_stream() -> None:
    doc = parse_document(_document("test/dense", {"component": "lm_head"}))
    t = site_tuple(doc, "patch", DENSE)
    assert t.layers == () and t.stream == frozenset()  # the empty band, no stream


def test_a_stale_declaration_is_refused(env, tmp_path) -> None:
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document("test/dense", {"component": "block_output", "layers": 0}),
    )
    error = _refused(
        _workflow(
            {
                "fit": _step(fit, waive=WAIVE_TARGET),
                "ctl": _step(
                    ctl, control=_control("full_component", declare=("layers",))
                ),
            }
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"declares a non-equivalence in 'layers' with its target 'fit', but the two "
        r"are equivalent there",
    )
    assert error.path == "steps.ctl.control.non_equivalence.fields"


@pytest.mark.parametrize(
    "declaration, match",
    [
        ("layers", r"'non_equivalence' is an object"),
        (
            # the protocol_version 2 spelling of the field, refused with the
            # v3 name — the one `layer` literal this file keeps, on purpose
            {"fields": ["layer"], "reason": "x"},
            r"unknown equivalence field 'layer' — did you mean 'layers'",
        ),
        ({"fields": ["layers", "layers"], "reason": "x"}, r"names 'layers' twice"),
        ({"fields": [], "reason": "x"}, r"non-empty list of equivalence fields"),
        (
            {"fields": ["layers"], "reason": "  "},
            r"'non_equivalence.reason' is a non-empty string",
        ),
        ({"fields": ["layers"]}, r"missing required"),
        ({"fields": ["layers"], "reason": "x", "why": "y"}, r"unknown key 'why'"),
    ],
    ids=[
        "not-an-object",
        "unknown-field",
        "repeated",
        "empty",
        "blank-reason",
        "no-reason",
        "stray-key",
    ],
)
def test_a_malformed_declaration_is_refused(env, tmp_path, declaration, match) -> None:
    fit = _write(
        tmp_path,
        "fit",
        _document("test/dense", {"component": "block_output", "layers": 0}),
    )
    raw = _workflow(
        {
            "fit": _step(fit, waive=WAIVE_TARGET),
            "ctl": _step(
                fit,
                control={
                    "of": "fit",
                    "kind": "full_component",
                    "non_equivalence": declaration,
                },
            ),
        }
    )
    with pytest.raises(WorkflowError) as info:
        load_workflow(raw, _env(env), workflow_dir=tmp_path)
    assert info.value.rule in (EQUIVALENCE_RULE, 1), str(info.value)
    assert re.search(match, str(info.value)), str(info.value)


def test_a_full_component_control_writes_the_whole_component(env, tmp_path) -> None:
    """Rule 14, true to its kind: a featurizer or a ``dims`` selection on the
    control's write makes it not the full component."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
        ),
    )
    steps = {"fit": _step(fit, waive=WAIVE_TARGET)}
    through = _write(
        tmp_path,
        "through",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
        ),
    )
    _refused(
        _workflow({**steps, "ctl": _step(through, control=_control("full_component"))}),
        env,
        tmp_path,
        CONTROL_RULE,
        r"write 'patch' goes through featurizer 'rot' \('subspace'\) — a full-component "
        r"control swaps the whole component",
    )
    sliced = _write(
        tmp_path,
        "sliced",
        _document(
            "test/dense", {"component": "block_output", "layers": 0}, dims=[0, 1, 2]
        ),
    )
    _refused(
        _workflow({**steps, "ctl": _step(sliced, control=_control("full_component"))}),
        env,
        tmp_path,
        CONTROL_RULE,
        r"write 'patch' selects dims \(0, 1, 2\) — a full-component control covers every "
        r"coordinate",
    )


def test_a_full_component_control_of_a_fit_is_equivalent_and_recorded(
    env, tmp_path
) -> None:
    """The legitimate parent of a learned intervention: the whole component
    swapped where the fit rotates a rank-4 subspace. Equivalent — the
    featurizer and the coordinates are the kind — and the verdict is what the
    runner records on the control step (§8)."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
            train=True,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document("test/dense", {"component": "block_output", "layers": 0}),
    )
    loaded = load_workflow(
        _workflow(
            {
                "fit": _step(
                    fit, waive={"self_swap": EXTERNAL, "matched_random": EXTERNAL}
                ),
                "ctl": _step(ctl, control=_control("full_component")),
            }
        ),
        _env(env),
        workflow_dir=tmp_path,
    )
    assert loaded.equivalence == {
        "ctl": {
            "status": "equivalent",
            "fields": ["featurizer", "dims"],
            "sharing": "distinct",
        }
    }
    # canonical only when authored: no declaration, no key
    assert "non_equivalence" not in loaded.canonical["steps"]["ctl"]["control"]
    assert "equivalence" not in loaded.canonical["steps"]["ctl"]["control"]


def test_a_declaration_is_canonical_with_its_fields_sorted(env, tmp_path) -> None:
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "attention_premix", "layers": 0},
            featurizer=SUBSPACE,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document("test/dense", {"component": "attention_output", "layers": 0}),
    )

    def loaded_with(fields: list[str]) -> Any:
        return load_workflow(
            _workflow(
                {
                    "fit": _step(fit, waive=WAIVE_TARGET),
                    "ctl": _step(
                        ctl,
                        control={
                            "of": "fit",
                            "kind": "full_component",
                            "non_equivalence": {"fields": fields, "reason": REASON},
                        },
                    ),
                }
            ),
            _env(env),
            workflow_dir=tmp_path,
        )

    a, b = loaded_with(["shape", "component"]), loaded_with(["component", "shape"])
    assert a.digest == b.digest
    assert a.canonical["steps"]["ctl"]["control"]["non_equivalence"] == {
        "fields": ["component", "shape"],
        "reason": REASON,
    }


# --------------------------------------------------------------------------- #
# T14 — coordinate sharing is decided, not guessed
# --------------------------------------------------------------------------- #


def test_t14_sharing_is_by_name_within_a_document_not_by_shape() -> None:
    """A parent write and a learned write naming one featurizer share its
    stage (``build_stack`` caches by name); two gates of one shape under two
    names do not. Mutation: deciding sharing by width calls the second pair
    shared."""
    raw = _document(
        "test/dense", {"component": "block_output", "layers": 0}, featurizer=GATE
    )
    method = raw["method"]
    method["featurizers"]["gate_b"] = {"kind": "gate"}
    method["reads"]["v_b"] = {**method["reads"]["v_cf"], "featurizer": "gate_b"}
    method["writes"]["learned"] = {**method["writes"]["patch"]}
    method["writes"]["other"] = {
        **method["writes"]["patch"],
        "featurizer": "gate_b",
        "do": {"swap": "v_b"},
    }
    doc = parse_document(raw)
    parent = site_tuple(doc, "patch", DENSE, owner="doc")
    learned = site_tuple(doc, "learned", DENSE, owner="doc")
    other = site_tuple(doc, "other", DENSE, owner="doc")
    assert sharing([parent], [learned]) == "shared"
    assert compare([parent], [learned]) == ("sharing",)
    assert parent.featurizer == other.featurizer and parent.dims == other.dims
    assert sharing([parent], [other]) == "distinct"
    assert compare([parent], [other]) == ()
    # across two documents a name is only a name
    twin = site_tuple(doc, "patch", DENSE, owner="other-doc")
    assert sharing([parent], [twin]) == "distinct"


def test_t14_a_control_in_its_targets_own_basis_is_refused_unless_declared(
    env, tmp_path
) -> None:
    """A ``matched_random`` control that loads the fit's saved rotation is
    scored in the fit's own basis — ``shared`` — and refused under rule 16
    naming ``sharing``; declared, rule 16 is satisfied and the load moves on
    to rule 14's seed provenance (a loaded featurizer draws nothing), which
    is the honest next refusal for such a document."""
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
            train=True,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer={**SUBSPACE, "file_path": "fit/rot.safetensors"},
        ),
    )
    steps = {"fit": _step(fit, waive={"self_swap": EXTERNAL})}
    draws = {"seeds": [0], "min_draws": 1}
    error = _refused(
        _workflow(
            {**steps, "ctl": _step(ctl, control=_control("matched_random", **draws))}
        ),
        env,
        tmp_path,
        EQUIVALENCE_RULE,
        r"differ in sharing: both are expressed in one coordinate system "
        r"\[\('bundle', 'fit/rot.safetensors'\)\]",
    )
    assert "scored in its target's own basis" in str(error)
    _refused(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl,
                    control=_control("matched_random", declare=("sharing",), **draws),
                ),
            }
        ),
        env,
        tmp_path,
        CONTROL_RULE,
        r"featurizer 'rot' has no seed to record",
    )


def test_t14_an_independent_draw_at_the_fits_site_is_distinct_and_loads(
    env, tmp_path
) -> None:
    fit = _write(
        tmp_path,
        "fit",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer=SUBSPACE,
            train=True,
        ),
    )
    ctl = _write(
        tmp_path,
        "ctl",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer={**SUBSPACE, "seed": {"sweep": [0, 1]}},
        ),
    )
    steps = {"fit": _step(fit, waive={"self_swap": EXTERNAL})}
    draws = {"seeds": [0, 1], "min_draws": 2}
    verdict = _loads(
        _workflow(
            {**steps, "ctl": _step(ctl, control=_control("matched_random", **draws))}
        ),
        env,
        tmp_path,
    )
    assert verdict == {"status": "equivalent", "fields": [], "sharing": "distinct"}
    # the pairing's own words hold (T8's `match=` strings): a rank mismatch and
    # a site mismatch are rule 14's, and a declared site difference is rule 16's
    lower = _write(
        tmp_path,
        "lower",
        _document(
            "test/dense",
            {"component": "block_output", "layers": 0},
            featurizer={**SUBSPACE, "k": 2, "seed": {"sweep": [0, 1]}},
        ),
    )
    _refused(
        _workflow(
            {**steps, "ctl": _step(lower, control=_control("matched_random", **draws))}
        ),
        env,
        tmp_path,
        CONTROL_RULE,
        r"pairs featurizer 'rot' \(k=2\) with fit 'fit''s 'rot' \(k=4\)",
    )
    moved = {"sites.target.layers": 1}
    _refused(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl, set=moved, control=_control("matched_random", **draws)
                ),
            }
        ),
        env,
        tmp_path,
        CONTROL_RULE,
        r"is drawn at the fit's site",
    )
    verdict = _loads(
        _workflow(
            {
                **steps,
                "ctl": _step(
                    ctl,
                    set=moved,
                    control=_control("matched_random", declare=("layers",), **draws),
                ),
            }
        ),
        env,
        tmp_path,
    )
    assert verdict == {
        "status": "declared",
        "fields": ["layers"],
        "sharing": "distinct",
    }


# --------------------------------------------------------------------------- #
# T15 — the legitimate campaign
# --------------------------------------------------------------------------- #


def test_t15_the_random_subspace_control_is_equivalent_to_the_das_fit(env) -> None:
    """``13_random_subspace_control_im.json`` against ``04_das_im.json``:
    ``block_output`` at layer 18, no head, no expert, the softmax stream on a
    dense tower; ``rot`` a rank-8 cayley subspace in fp32; the write over
    every coordinate of it; two names in two documents, so ``distinct``. Only
    the rotation's *values* differ — 13 sweeps ``seed`` where 04 trains.
    The two also differ in ``model.dtype`` (13 is unauthored → fp32, 04 is
    bf16): the model's realization is a separate clause, and the tuple has no
    ``model.*`` field — asserted below. Mutation: a ``seed`` (or the bundle
    digest) in the tuple refuses this pair."""
    control = compile_protocol(CONTROL_13, env=env)
    fit = compile_protocol(FIT, env=env)
    control_steps, fit_steps = steps_of(control, env), steps_of(fit, env)
    info = get_model_info("Qwen/Qwen3-8B")
    a = coverage(control_steps.documents, info, owner="control")
    b = coverage(fit_steps.documents, info, owner="fit")
    assert len(control_steps.documents) == 3 and len(a) == 1  # three seeds, one site
    (t,) = a
    assert (t.component, sorted(t.layers), t.head, t.expert) == (
        "block_output",
        [18],
        None,
        None,
    )
    assert t.stream == frozenset({"full_attention"})
    assert t.featurizer == (FeaturizerStage("subspace", 8, "cayley", None, "fp32"),)
    assert t.dims == tuple(range(8)) and t.routed_rank is None
    assert compare(a, b) == () and sharing(a, b) == "distinct"
    assert not any(field.startswith("model") for field in EQUIVALENCE_FIELDS)
    assert control.document.model.dtype is None and fit.document.model.dtype == "bf16"
    loaded = load_workflow(CONTROL_FIXTURES / "matched_random.json", env)
    assert loaded.equivalence == {
        "control": {"status": "equivalent", "fields": [], "sharing": "distinct"}
    }
    pin = CORPUS_PINS["13_random_subspace_control_im.json"]
    assert control.digests.document == pin["document"]
    assert list(control_steps.digests) == pin["points"]


def test_t15_values_are_not_coordinates() -> None:
    """The stage carries kind, rank, parametrization, group, dtype, axis and
    the axis's unit count and nothing else: no ``seed``, no ``init``, no
    ``file_path``, no digest, no ``forward`` split (a value, not a shape)."""
    assert tuple(f.name for f in dataclasses.fields(FeaturizerStage)) == (
        "kind",
        "k",
        "parametrization",
        "group",
        "dtype",
        "axis",
        "units",
    )
    source = (REPO / MODULE).read_text()
    for value in ("content_digest", "spec.seed", "spec.init", "spec.entry"):
        assert value not in source, f"{value!r} would put a value into the tuple"


def test_a_position_gate_is_not_the_shape_of_a_plain_gate() -> None:
    """§2.5 ``axis`` one layer up: a position gate and a feature gate at one
    site are two shapes — θ over positions against θ over coordinates — so a
    `matched_random` control built with a plain gate is not a position-gate
    fit's equivalent. Before the change the stage tuple carried no `axis`,
    both had `k is None` and `group is None`, and the two compared equal.
    """
    from causalab.protocol.equivalence import site_tuple
    from causalab.protocol.schema import parse_document
    from tests.protocol._docs import base_doc, in_order

    def doc(axis: bool):
        raw = base_doc()
        raw["method"]["featurizers"] = {
            "g": {"kind": "gate", **({"axis": "position"} if axis else {})}
        }
        raw["method"]["reads"]["v_cf"]["pos"] = {"span": [0, 3]}
        raw["method"]["writes"]["patch"]["pos"] = {"span": [0, 3]}
        raw["method"]["writes"]["patch"]["featurizer"] = "g"
        return parse_document(in_order(raw))

    positional = site_tuple(doc(True), "patch", None, owner="fit")
    plain = site_tuple(doc(False), "patch", None, owner="fit")
    assert positional.featurizer != plain.featurizer
    assert (
        positional.featurizer[0].axis == "position" and plain.featurizer[0].axis is None
    )
    # the window is the unit count: three positions, said in the description
    assert positional.featurizer[0].units == 3 and plain.featurizer[0].units is None
    assert "axis=position[3]" in positional.featurizer[0].describe()


def test_two_position_gates_over_different_windows_are_two_shapes() -> None:
    """§2.5 ``axis``: the tuple says how many units the axis has, not only
    which axis — a position gate over `[0, 2]` is not the shape of one over
    `[0, 3]` at the same site, so a `matched_random` control over the
    shorter window is not the fit's equivalent (rule 16), where before the
    two tuples were identical and the difference undeclarable. A named
    `positions` entry resolves the same way."""
    from causalab.protocol.equivalence import site_tuple
    from causalab.protocol.schema import parse_document
    from tests.protocol._docs import base_doc, in_order

    def doc(span, *, named: bool = False):
        raw = base_doc()
        raw["method"]["featurizers"] = {"g": {"kind": "gate", "axis": "position"}}
        pos = {"span": list(span)}
        if named:
            raw["method"]["positions"] = {"window": pos}
            pos = "window"
        raw["method"]["reads"]["v_cf"]["pos"] = pos
        raw["method"]["writes"]["patch"]["pos"] = pos
        raw["method"]["writes"]["patch"]["featurizer"] = "g"
        return parse_document(in_order(raw))

    short = site_tuple(doc((0, 2)), "patch", None, owner="fit").featurizer
    long = site_tuple(doc((0, 3)), "patch", None, owner="fit").featurizer
    assert short != long and (short[0].units, long[0].units) == (2, 3)
    named = site_tuple(doc((0, 3), named=True), "patch", None, owner="fit").featurizer
    assert named == long
