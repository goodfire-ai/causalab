"""A structurally unavailable cell appears in the result with its reason code
and in the denominator (spec §4.1) — the width-zero scoped slice, run end to
end through the reference engine.

The witness is ``tiny-random/qwen3.5-moe``: 128 experts, top-10, so on a
handful of prompts most experts are sent no token. A read of
``expert_activation`` scoped to one such expert used to come back as a
width-0 ragged gather and nothing else — honest as data, silent as a result:
a summary over a sweep could not tell "the instrument measured nothing here"
from "the effect was null". Now the run writes that cell as
``unavailable`` / ``empty_selector`` in the saved entry's record, in the
point's summary and in ``RunResult.cells``, and the denominator counts it as
excluded.

Every converted site has its valid-work twin: the same
document on an expert the router *did* choose is byte-for-byte what it was —
the entry record carries no ``status`` field, the summary no ``unavailable``
key, and the saved rows equal the executor's own gather. The ``reduce`` path
is the one place the old behaviour *raised*: ``median`` over zero rows was a
torch ``IndexError`` after the forward pass; it is a ``NaN`` vector now.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.executor import RaggedValue
from causalab.protocol import run_protocol
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.results import Available, Unavailable, cell_key
from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    read_safetensors_metadata,
)

from ._drive import base_data_section, executor_for
from .conftest import TINY_QWEN35_MOE
from tests.protocol._docs import saved
from tests.protocol._env import FIXTURES


pytestmark = pytest.mark.smoke

MOE_LAYER = 0
#: the fixture's routed inner width — `expert_activation` rows are this wide
D_EXPERT = 32
#: a committed three-row prompt table (`tests/protocol/fixtures/data/pile`)
DATASET = "pile/sample"


def _doc(expert: int, *, reduce: str | None = None) -> dict[str, Any]:
    extra = {} if reduce is None else {"reduce": reduce}
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_QWEN35_MOE, "revision": "main"},
        "data": {"base": {"dataset": DATASET, "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {
                "tap": {
                    "component": "expert_activation",
                    "layers": [MOE_LAYER],
                    "expert": expert,
                }
            },
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "r.safetensors", **extra)],
        },
    }


def _env(tmp_path: Path) -> ResolutionEnv:
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=tmp_path),
    )


@pytest.fixture(scope="module")
def hit_and_missing(qwen35moe_bundle) -> tuple[int, int]:
    """One expert the router chose at these prompts, and one it never did."""
    rows = json.loads((FIXTURES / "data" / "pile" / "sample.json").read_text())
    texts = [row["input"] for row in rows]
    doc = {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": "expert_idx", "layers": [MOE_LAYER]}},
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    idx = executor_for(doc, qwen35moe_bundle, base_texts=texts).read_value("r")
    assert isinstance(idx, torch.Tensor)
    chosen = {int(x) for x in idx.reshape(-1)}
    hit = int(idx.reshape(-1).mode().values)
    missing = next(i for i in range(128) if i not in chosen)
    return hit, missing


def _run(tmp_path: Path, doc: dict[str, Any]):
    env = _env(tmp_path)
    loaded = compile_protocol(doc, env=env)
    out = tmp_path / "out"
    result = run_protocol(loaded, env, PytorchHooksEngine(device="cpu"), out)
    return loaded, result, out


def test_an_expert_no_token_chose_is_an_unavailable_cell(tmp_path, hit_and_missing):
    """In the result: `RunResult.cells` holds the typed value with its reason
    code; the saved entry's record carries the four fields; the point summary
    repeats them. In the denominator: `0 / 1 eligible`, excluded under
    `empty_selector`."""
    _, missing = hit_and_missing
    loaded, result, out = _run(tmp_path, _doc(missing))

    assert len(result.cells) == 1
    (cell,) = result.cells
    assert isinstance(cell, Unavailable)
    assert cell.reason == "empty_selector"
    assert cell.denominator_key == cell_key("r", {}) == "r"
    assert f"expert {missing} no token" in cell.detail

    denominator = result.denominator
    assert (denominator.eligible, denominator.total) == (0, 1)
    assert denominator.unavailable == {"empty_selector": ("r",)}
    assert denominator.render() == "0 / 1 eligible; 1 excluded: empty_selector ×1"

    stamped = read_safetensors_metadata(out / "r.safetensors")
    assert stamped is not None
    record = json.loads(stamped["entries"])["r"]
    assert record["status"] == "unavailable"
    assert record["reason"] == "empty_selector"
    assert record["denominator_key"] == "r"
    assert record["detail"] == cell.detail

    # the data is still there, and still says width zero per row
    tensors = load_file(str(out / "r.safetensors"))
    assert tuple(tensors["r"].shape) == (0, D_EXPERT)
    assert tensors["r.widths"].tolist() == [0, 0, 0]

    (summary,) = result.summaries
    assert summary["unavailable"] == {"r": cell.record()}


def test_the_same_read_on_a_chosen_expert_is_unchanged(
    tmp_path, hit_and_missing, qwen35moe_bundle
):
    """The valid-work twin: an available cell records nothing new, so the
    entry record has exactly the fields it had before the value existed, the
    summary has no `unavailable` key, and the rows are the executor's own
    gather."""
    hit, _ = hit_and_missing
    _, result, out = _run(tmp_path, _doc(hit))

    (cell,) = result.cells
    assert isinstance(cell, Available)
    assert cell.denominator_key == "r"
    assert result.denominator.render() == "1 / 1 eligible"

    stamped = read_safetensors_metadata(out / "r.safetensors")
    assert stamped is not None
    record = json.loads(stamped["entries"])["r"]
    # the record before the value existed: the entry's slot and coords, the
    # site (§8), and what a harvested activation was read from —
    # `trained_on` (H5, `ARTIFACT_IDENTITY_KEYS`). Nothing of the four status
    # fields
    assert set(record) == {
        "slot",
        "coords",
        "site",
        "trained_on",
        "loaded_attn_implementation",
    }
    assert "status" not in record
    (summary,) = result.summaries
    assert "unavailable" not in summary

    tensors = load_file(str(out / "r.safetensors"))
    assert tensors["r"].shape[0] == int(tensors["r.widths"].sum()) > 0
    assert tensors["r"].shape[-1] == D_EXPERT
    # the same document through the executor alone: identical rows
    rows = json.loads((FIXTURES / "data" / "pile" / "sample.json").read_text())
    direct = _doc(hit)
    direct["model"] = {"key": "test", "revision": "main"}
    direct["data"] = base_data_section(with_counterfactual=False)
    value = executor_for(
        direct, qwen35moe_bundle, base_texts=[row["input"] for row in rows]
    ).read_value("r")
    assert isinstance(value, RaggedValue)
    torch.testing.assert_close(value.flat, tensors["r"], atol=0.0, rtol=0.0)
    assert list(value.widths) == tensors["r.widths"].tolist()


@pytest.mark.parametrize(
    "reduce, expected",
    [("median", "nan"), ("mean", "nan"), ("std", "nan"), ("sum", 0.0), ("count", 0.0)],
)
def test_a_reduced_unavailable_cell_is_a_vector_not_a_raise(
    tmp_path, hit_and_missing, reduce, expected
):
    """`reduce` over zero rows: `median` used to raise `IndexError` after the
    forward pass; every verb now yields the `(width,)` vector §2.12 promises,
    `NaN` for a statistic of no observations and `0` for the pair that
    composes (`sum`, `count`)."""
    _, missing = hit_and_missing
    _, result, out = _run(tmp_path, _doc(missing, reduce=reduce))
    (cell,) = result.cells
    assert isinstance(cell, Unavailable) and cell.reason == "empty_selector"
    vector = load_file(str(out / "r.safetensors"))["r"]
    assert tuple(vector.shape) == (D_EXPERT,)
    if expected == "nan":
        assert all(math.isnan(x) for x in vector.tolist())
    else:
        assert vector.tolist() == [expected] * D_EXPERT


def test_a_reduced_available_cell_is_unchanged(tmp_path, hit_and_missing):
    hit, _ = hit_and_missing
    _, result, out = _run(tmp_path, _doc(hit, reduce="count"))
    (cell,) = result.cells
    assert isinstance(cell, Available)
    vector = load_file(str(out / "r.safetensors"))["r"]
    assert tuple(vector.shape) == (D_EXPERT,)
    assert vector[0].item() > 0 and len(set(vector.tolist())) == 1
    stamped = read_safetensors_metadata(out / "r.safetensors")
    assert stamped is not None
    assert "status" not in json.loads(stamped["entries"])["r"]
