"""Alignment cardinality against a real tokenizer (spec §2.3, §4.1) — the
engine half, on the two tiny fixtures.

* **T4** — a ``match`` over the answers as the table carries them (space-
  prefixed, because that is how they follow "tomorrow is") runs: an answer
  string is tokenized **as written** (§2.10), so the table's form is the
  form scored and there is no second knob for a document to contradict it
  with. (The unit half — ``"Saturday"`` and ``" Saturday"`` naming two rows
  on a byte-level BPE — is ``tests/neural/shared/test_answer_forms.py``.)
* an **ambiguous** or **absent** ``variable`` row makes a *read* an
  ``unavailable`` cell — ``alignment_ambiguous`` / ``alignment_missing``, the
  detail naming the value, its count and the row — counted in the
  denominator (``cells 1 / 2 eligible; 1 excluded: alignment_ambiguous ×1``),
  a metric over that read inheriting the cell **row by row** (§2.10
  "Eligibility": the other row is scored, the excluded one carries the
  reason code — ``test_eligibility_run.py`` is that half); the same row under
  a *write* is refused before any forward, with the same reason code;
* a **declared** ``alignment`` the pair contradicts is refused naming both
  values; the matching declaration, and one on an ``index``, run clean;
* **T6** — both interchange documents (the shipped split application and its
  flat corpus twin) run with no ``alignment`` authored anywhere.

Every refusal has its valid-work twin beside it, so a check that refuses
everything cannot pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.encoding import candidate_runs, encode
from causalab.neural.shared.executor import RaggedValue
from causalab.protocol import run_protocol
from causalab.protocol.positions.alignment import alignment_of
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.results import Available, Unavailable
from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    read_safetensors_metadata,
)
from causalab.protocol.schema import PositionSpec

from ._drive import base_data_section, executor_for
from .conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, in_order, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, fixture_input_overrides
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]
INTERCHANGE_PRESET = PROTOCOLS_DIR / "interchange.json"


# --------------------------------------------------------------------------- #
# T4 — the table's answer form is the form scored
# --------------------------------------------------------------------------- #


def _match_doc() -> dict[str, Any]:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [
                saved(
                    "logits",
                    "original",
                    "iia.json",
                    {
                        "kind": "match",
                        "expected": "cf_answer",
                        "mode": "first_token",
                    },
                )
            ],
        },
    }


PROMPTS = ["If today is Friday, tomorrow is", "If today is Monday, tomorrow is"]
#: the answers as the table carries them — space-prefixed, because that is how
#: they follow "tomorrow is" (the weekdays fixture's `cf_answer` column)
SPACED_ANSWERS = [" Saturday", " Tuesday"]


def test_t4_a_match_over_the_answers_as_the_table_carries_them_runs(bundle):
    """No pre-flight stands between the table and the forward: the answer
    string is the token the model emits after "tomorrow is", and the metric
    scores exactly that string (§2.10)."""
    executor = executor_for(
        _match_doc(),
        bundle,
        base_texts=PROMPTS,
        extra_columns={"cf_answer": SPACED_ANSWERS},
    )
    value = executor.read_value("logits")
    assert isinstance(value, torch.Tensor) and value.shape[0] == 2


# --------------------------------------------------------------------------- #
# an unalignable row: a cell for a read, a refusal for a write
# --------------------------------------------------------------------------- #

#: one undivided pool — every table declares its rows' split (§2.2, V22)
AMBIGUOUS_ROWS = [
    {"input": "day after day", "entity": "day", "split": "all"},  # occurs twice
    {"input": "one two three", "entity": "two", "split": "all"},  # occurs once
]


def _variable_read_doc(*, with_metric: bool, model_key: str = "test") -> dict[str, Any]:
    doc: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": model_key, "revision": "main"},
        "data": {"base": {"dataset": "amb/rows", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "positions": {"ent": {"variable": "entity"}},
            "sites": {"tap": {"component": "block_output", "layers": [0]}},
            "reads": {"r": {"site": "tap", "pos": "ent"}},
            "save": [saved("r", "original", "r.safetensors")],
        },
    }
    if with_metric:
        # the metric's read is the only user of lm_head, so the site is
        # declared with it — a declared-but-unused site is refused (V11)
        doc["method"]["sites"]["lm_head"] = {"component": "lm_head"}
        doc["method"]["reads"]["logits"] = {"site": "lm_head", "pos": "ent"}
        doc["method"]["intervened_models"]["original"]["reads"].append("logits")
        doc["method"]["save"].append(
            saved(
                "logits",
                "original",
                "tl.json",
                {"kind": "token_logit", "token": "entity"},
            )
        )
    return doc


def _env_with_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> ResolutionEnv:
    table = tmp_path / "data" / "amb" / "rows.json"
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps(rows))
    return ResolutionEnv(
        datasets=FileDatasets(root=tmp_path / "data"),
        artifacts=FileArtifacts(root=tmp_path / "artifacts"),
    )


def test_an_ambiguous_row_is_an_unavailable_cell_counted_in_the_denominator(
    tmp_path, llama_bundle
):
    """End to end through ``run_protocol``: the read's cell is
    ``alignment_ambiguous`` with the value, its count and the row in the
    detail; the metric over it inherits the cell **row by row** — its cell is
    available with one of two rows eligible (§2.10 "Eligibility"); the
    denominator counts the read's cell as excluded and the metric's as
    eligible; the saved gather has width zero on the excluded row and the
    other row's tokens intact."""
    env = _env_with_rows(tmp_path, AMBIGUOUS_ROWS)
    loaded = compile_protocol(
        in_order(_variable_read_doc(with_metric=True, model_key=TINY_LLAMA)), env=env
    )
    out = tmp_path / "out"
    result = run_protocol(loaded, env, PytorchHooksEngine(device="cpu"), out)

    read_cell, metric_cell = result.cells
    assert isinstance(read_cell, Unavailable)
    assert read_cell.reason == "alignment_ambiguous"
    assert read_cell.denominator_key == "r"
    assert "'day' occurs 2 times" in read_cell.detail
    assert "(row 0)" in read_cell.detail
    assert "1 of 2 rows" in read_cell.detail
    # the metric scores the row that aligned and excludes the one that did
    # not: an available cell, one of two rows eligible
    assert isinstance(metric_cell, Available)
    assert metric_cell.denominator_key == "tl"
    assert metric_cell.mapping["n_eligible"] == 1
    assert metric_cell.mapping["n_considered"] == 2
    assert metric_cell.mapping["excluded"] == {"alignment_ambiguous": 1}

    denominator = result.denominator
    assert (denominator.eligible, denominator.total) == (1, 2)
    assert denominator.render() == "1 / 2 eligible; 1 excluded: alignment_ambiguous ×1"

    stamped = read_safetensors_metadata(out / "r.safetensors")
    assert stamped is not None
    record = json.loads(stamped["entries"])["r"]
    assert record["status"] == "unavailable"
    assert record["reason"] == "alignment_ambiguous"
    tensors = load_file(str(out / "r.safetensors"))
    widths = tensors["r.widths"].tolist()
    assert widths[0] == 0 and widths[1] > 0
    assert tensors["r"].shape[0] == widths[1]
    (summary,) = result.summaries
    assert set(summary["unavailable"]) == {"r"}
    assert summary["eligibility"]["tl"] == {
        "n_eligible": 1,
        "n_considered": 2,
        "excluded": {"alignment_ambiguous": 1},
    }


def test_an_absent_row_is_an_unavailable_cell_with_the_missing_reason(llama_bundle):
    executor = executor_for(
        _variable_read_doc(with_metric=False),
        llama_bundle,
        base_texts=["one two three", "four five six"],
        extra_columns={"entity": ["seven", "five"]},
    )
    value = executor.read_value("r")
    cell = executor.resolution("r")
    assert isinstance(cell, Unavailable)
    assert cell.reason == "alignment_missing"
    assert "'seven' occurs 0 times" in cell.detail and "(row 0)" in cell.detail
    assert isinstance(value, RaggedValue)
    assert value.widths[0] == 0 and value.widths[1] > 0


def test_twin_a_uniquely_occurring_value_is_an_available_read(llama_bundle):
    executor = executor_for(
        _variable_read_doc(with_metric=False),
        llama_bundle,
        base_texts=["one two three", "four five six"],
        extra_columns={"entity": ["two", "five"]},
    )
    executor.read_value("r")
    assert not isinstance(executor.resolution("r"), Unavailable)


def _variable_write_doc() -> dict[str, Any]:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                "patched": {"input": "base", "reads": ["logits"], "writes": ["patch"]},
            },
            "positions": {"ent": {"variable": "entity"}},
            "sites": {
                "tap": {"component": "block_output", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": "ent"},
                "logits": {"site": "lm_head", "pos": -1},
            },
            "writes": {"patch": {"site": "tap", "pos": "ent", "do": {"swap": "v_cf"}}},
            "save": [saved("logits", "patched", "logits.safetensors")],
        },
    }


def test_an_ambiguous_row_under_a_write_is_refused_before_any_forward(llama_bundle):
    """A write cannot skip a row and still report a number for it, so the
    same fact that is a cell for a read is a typed refusal here — reason
    ``alignment_ambiguous``, raised by the pre-forward width check."""
    executor = executor_for(
        _variable_write_doc(),
        llama_bundle,
        base_texts=["day after day", "one two three"],
        counterfactual_texts=["one day only", "one two three"],
        extra_columns={"entity": ["day", "two"]},
    )
    with pytest.raises(ProtocolError) as err:
        executor.read_value("logits")
    assert err.value.reason == "alignment_ambiguous"
    assert "'day' occurs 2 times" in str(err.value)
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]


def test_twin_a_write_at_a_unique_variable_runs(llama_bundle):
    executor = executor_for(
        _variable_write_doc(),
        llama_bundle,
        base_texts=["one day only", "one two three"],
        counterfactual_texts=["a day passes", "one two three"],
        extra_columns={"entity": ["day", "two"]},
    )
    assert isinstance(executor.read_value("logits"), torch.Tensor)


# --------------------------------------------------------------------------- #
# a declared alignment against the observed one
# --------------------------------------------------------------------------- #

BASE_TEXT, BASE_ENTITY = "the cat sat", "cat"
CF_TEXT, CF_ENTITY = "the caterpillar sat", "caterpillar"


def _declared_doc(alignment: str) -> dict[str, Any]:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "positions": {"ent": {"variable": "entity", "alignment": alignment}},
            "sites": {"tap": {"component": "block_output", "layers": [0]}},
            "reads": {"r": {"site": "tap", "pos": "ent"}},
            "save": [saved("r", "original", "r.safetensors")],
        },
    }


@pytest.fixture(scope="module")
def observed_pair(llama_bundle) -> str:
    """What the fixture's tokenizer actually makes of the pair, through the
    same resolver the executor uses — so the test asserts against the
    tokenizer, not against a guess about it."""
    spec = PositionSpec(variable="entity")
    base = candidate_runs(
        spec,
        encode(llama_bundle.tokenizer, [BASE_TEXT]),
        0,
        dataset_row={"input": BASE_TEXT, "entity": BASE_ENTITY},
        field="input",
    )
    cf = candidate_runs(
        spec,
        encode(llama_bundle.tokenizer, [CF_TEXT]),
        0,
        dataset_row={"input": CF_TEXT, "entity": CF_ENTITY},
        field="input",
    )
    observed = alignment_of(base, cf)
    if observed == "one_to_one":
        pytest.skip("this tokenizer gives 'cat' and 'caterpillar' equal widths")
    return observed


def _declared_executor(alignment: str, llama_bundle):
    return executor_for(
        _declared_doc(alignment),
        llama_bundle,
        base_texts=[BASE_TEXT],
        counterfactual_texts=[CF_TEXT],
        extra_columns={
            "entity": [BASE_ENTITY],
            "counterfactual_inputs_variables": [{"entity": CF_ENTITY}],
        },
    )


def test_a_declared_alignment_the_pair_contradicts_is_refused(
    llama_bundle, observed_pair
):
    executor = _declared_executor("one_to_one", llama_bundle)
    with pytest.raises(ProtocolError) as err:
        executor.read_value("r")
    message = str(err.value)
    assert "declares alignment 'one_to_one'" in message
    assert f"resolves it as '{observed_pair}'" in message
    assert "row 0" in message and "'base'" in message and "'counterfactual'" in message


def test_twin_the_matching_declaration_runs(llama_bundle, observed_pair):
    executor = _declared_executor(observed_pair, llama_bundle)
    value = executor.read_value("r")
    assert isinstance(value, (torch.Tensor, RaggedValue))


def test_twin_a_declaration_on_an_index_runs_with_one_role(llama_bundle):
    doc = _declared_doc("one_to_one")
    doc["data"] = base_data_section(with_counterfactual=False)
    doc["method"]["positions"]["ent"] = {"index": -1, "alignment": "one_to_one"}
    executor = executor_for(doc, llama_bundle, base_texts=[BASE_TEXT, CF_TEXT])
    value = executor.read_value("r")
    assert isinstance(value, torch.Tensor) and value.shape[:2] == (2, 1)


# --------------------------------------------------------------------------- #
# T6 — the legitimate campaign runs with nothing authored
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return root


def _walk(value: Any):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


@pytest.mark.parametrize(
    "document",
    [CORPUS_DIR / "02_interchange_im.json", INTERCHANGE_PRESET],
    ids=["corpus", "preset"],
)
def test_t6_an_interchange_document_runs_with_no_alignment_authored(
    document: Path, artifacts_root: Path, tmp_path: Path
):
    out = tmp_path / "out"
    # the shipped preset names the shipped weekdays table, whose answers
    # tiny-random cannot spell as single tokens ([P2]): retarget its dataset
    # refs onto the 4-row fixture the way the model is retargeted. The corpus
    # document already names the fixture, so this adds nothing there.
    fixture_inputs = [
        f"--set={path}={ref}"
        for path, ref in fixture_input_overrides(
            json.loads(document.read_text())
        ).items()
    ]
    code = main(
        [
            "run",
            "--engine",
            "auto",
            str(document),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(artifacts_root),
            "--out",
            str(out),
            "--set",
            f"model.key={TINY_LLAMA}",
            "--set",
            "sites.target.layers=1",
            *fixture_inputs,
            "--dtype",
            "fp32",
            "--record",
        ]
    )
    assert code == 0
    record = json.loads((out / "protocol.json").read_text())
    assert "alignment" not in set(_walk(record["canonical"]))
    assert (out / "iia.json").is_file()
