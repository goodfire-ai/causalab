"""The engine half of the coordinated-edit-group refusal (spec §2.2, §5 item
27) on the tiny gpt2 fixture — encode-time, before any forward.

The two-mapping-entry swap: ``a maps to 1, b maps to 2`` against
``a maps to 2, b maps to 1``. The pair was validated as one edit of two
entries, declared as one ``atomic`` group; an interchange at the first entry
alone is refused as rule 27 **before any forward pass**, naming the group and
the second entry it left out. Every refusal has its twin: the same document
runs under ``atomic: false``; a document addressing both entries (one write
per constituent, in one intervened model) runs; rows without the column run
untouched; and the four-row weekdays table runs the corpus interchange
document with no refusal (T9's engine half).
"""

from __future__ import annotations

import shutil
from typing import Any

import pytest
import torch

from causalab.causal.pair_validation import EDIT_GROUPS_COLUMN
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.protocol import run_protocol
from causalab.protocol.rules.errors import ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, aggregation, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, write_rot_fixture


pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def gpt2_bundle():
    return load_model(TINY_GPT2)


# --------------------------------------------------------------------------- #
# the rows: two swaps, each one atomic group of two single-token entries
# --------------------------------------------------------------------------- #

BASES = ["a maps to 1, b maps to 2. a maps to", "x maps to 3, y maps to 4. x maps to"]
CFS = ["a maps to 2, b maps to 1. a maps to", "x maps to 4, y maps to 3. x maps to"]
FIRST = ["1", "3"]
SECOND = ["2", "4"]


def _span(text: str, needle: str) -> list[int]:
    start = text.index(needle)
    return [start, start + len(needle)]


def _groups(atomic: bool) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "name": "mapping_swap",
                "atomic": atomic,
                "spans": {
                    "base": [_span(base, first), _span(base, second)],
                    "counterfactual": [_span(cf, second), _span(cf, first)],
                },
            }
        ]
        for base, cf, first, second in zip(BASES, CFS, FIRST, SECOND)
    ]


def _columns(atomic: bool | None) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {
        "first": FIRST,
        "second": SECOND,
        "counterfactual_inputs_variables": [
            [{"first": second, "second": first}] for first, second in zip(FIRST, SECOND)
        ],
        "cf_answer": [f" {s}" for s in SECOND],
    }
    if atomic is not None:
        columns[EDIT_GROUPS_COLUMN] = _groups(atomic)
    return columns


def _doc(*positions: str) -> dict[str, Any]:
    """An interchange swapping the counterfactual's value into base at each
    named variable position, all writes in one intervened model."""
    doc: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_GPT2, "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                "patched": {"input": "base", "reads": ["logits"], "writes": []}
            },
            "positions": {name: {"variable": name} for name in positions},
            "sites": {
                "target": {"component": "block_output", "layers": [1]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "writes": {},
            "save": [
                saved(
                    "logits",
                    "patched",
                    "iia.json",
                    aggregation("match", expected="cf_answer"),
                )
            ],
        },
    }
    for name in positions:
        _add_swap(doc, f"v_{name}", f"patch_{name}", pos=name)
    return doc


def _add_swap(doc: dict[str, Any], read: str, write: str, *, pos: Any) -> None:
    """Read ``read`` at ``pos`` on the un-intervened counterfactual and swap
    it into base at the same position in ``patched``."""
    models = doc["method"]["intervened_models"]
    models.setdefault(UNWRITTEN, {"input": "counterfactual", "reads": []})
    models[UNWRITTEN]["reads"].append(read)
    doc["method"]["reads"][read] = {"site": "target", "pos": pos}
    doc["method"]["writes"][write] = {
        "site": "target",
        "pos": pos,
        "do": {"swap": read},
    }
    models["patched"]["writes"].append(write)


def _executor(gpt2_bundle, doc: dict[str, Any], atomic: bool | None):
    return executor_for(
        doc,
        gpt2_bundle,
        base_texts=BASES,
        counterfactual_texts=CFS,
        extra_columns=_columns(atomic),
    )


def test_the_premise_each_entry_is_one_token(gpt2_bundle):
    tokenizer = gpt2_bundle.tokenizer
    for value in FIRST + SECOND:
        assert len(tokenizer.encode(f" {value}", add_special_tokens=False)) == 1


def test_a_constituent_addressed_alone_is_refused_before_any_forward(gpt2_bundle):
    """The refusal: the interchange at ``first`` alone touches one entry of
    the atomic swap. Rule 27, named on the intervened model, naming the group
    and the sibling — and no forward group has run."""
    executor = _executor(gpt2_bundle, _doc("first"), atomic=True)
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 27 and err.value.rule_id == "segment_declared"
    assert err.value.path == "intervened_models.patched"
    message = str(err.value)
    assert "[V27]" in message and "'mapping_swap'" in message and "atomic" in message
    assert "constituent(s) 0 ('1')" in message and "sibling(s) 1 ('2')" in message
    assert "intervened_models.patched" in message and "row 0" in message
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]


def test_the_counterfactual_side_is_held_too(gpt2_bundle):
    """The operand read's position on the counterfactual is part of what the
    intervened model addresses: a write at the last token (touching no
    constituent on base) whose operand is read at ``first`` on the
    counterfactual — where that value is the swapped second entry — touches
    one constituent of two on the counterfactual side."""
    doc = _doc("first")
    doc["method"]["writes"]["patch_first"]["pos"] = -1
    executor = _executor(gpt2_bundle, doc, atomic=True)
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 27
    message = str(err.value)
    assert "on 'counterfactual'" in message
    assert "constituent(s) 0 ('2')" in message and "sibling(s) 1 ('1')" in message
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]


def test_mutation_atomic_false_runs(gpt2_bundle):
    executor = _executor(gpt2_bundle, _doc("first"), atomic=False)
    value = executor.read_value("logits")
    assert isinstance(value, torch.Tensor) and value.shape[0] == len(BASES)


def test_twin_both_constituents_addressed_in_one_intervened_model_run(gpt2_bundle):
    executor = _executor(gpt2_bundle, _doc("first", "second"), atomic=True)
    value = executor.read_value("logits")
    assert isinstance(value, torch.Tensor) and value.shape[0] == len(BASES)


def test_twin_rows_without_the_column_run(gpt2_bundle):
    executor = _executor(gpt2_bundle, _doc("first"), atomic=None)
    value = executor.read_value("logits")
    assert isinstance(value, torch.Tensor) and value.shape[0] == len(BASES)


def test_a_malformed_declaration_reaching_the_executor_is_rule_27_too(gpt2_bundle):
    columns = _columns(atomic=True)
    columns[EDIT_GROUPS_COLUMN][0][0]["spans"]["base"] = [[0, 999], [23, 24]]
    executor = executor_for(
        _doc("first"),
        gpt2_bundle,
        base_texts=BASES,
        counterfactual_texts=CFS,
        extra_columns=columns,
    )
    with pytest.raises(ValidationError) as err:
        executor.read_value("logits")
    assert err.value.rule == 27 and err.value.path == "data.base"


def _fixture_env(tmp_path) -> ResolutionEnv:
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    return ResolutionEnv(
        datasets=FileDatasets(root=FIXTURES / "data"),
        artifacts=FileArtifacts(root=artifacts),
    )


def test_t9_the_four_row_weekdays_table_runs_the_interchange_document(tmp_path):
    """The legitimate campaign's engine half, end to end: the committed
    fixture table (no column) under the corpus interchange document, through
    ``run_protocol``. On the sentencepiece fixture, because the document's
    ``match`` scores ``" Saturday"`` as one token and the byte-level BPE
    fixture spells it in several (``[P2] … not a single token``, the metric's
    own pre-existing refusal, nothing to do with edit groups); the gpt2 twin
    below runs the same interchange shape over the same rows."""
    env = _fixture_env(tmp_path)
    rows = env.datasets.rows("weekdays/data#train")
    assert rows and all(EDIT_GROUPS_COLUMN not in row for row in rows)
    loaded = compile_protocol(
        CORPUS_DIR / "02_interchange_im.json",
        env=env,
        overrides={"model.key": TINY_LLAMA, "sites.target.layers": 1},
    )
    result = run_protocol(loaded, env, PytorchHooksEngine(), tmp_path / "run")
    assert "iia.json" in result.files


def test_t9_twin_the_four_fixture_rows_run_the_interchange_on_tiny_gpt2(
    gpt2_bundle, tmp_path
):
    """The same four rows, the same interchange (swap the counterfactual's
    answer-slot residual into base), on tiny gpt2 — no ``edit_groups`` column,
    nothing held, no refusal: the value the patched model reads comes back."""
    rows = _fixture_env(tmp_path).datasets.rows("weekdays/data#train")
    doc = _doc()  # no variable positions: writes/reads added by hand at -1
    _add_swap(doc, "v_cf", "patch", pos=-1)
    executor = executor_for(
        doc,
        gpt2_bundle,
        base_texts=[row["input"] for row in rows],
        counterfactual_texts=[row["counterfactual_inputs"][0] for row in rows],
        extra_columns={"cf_answer": [row["cf_answer"] for row in rows]},
    )
    value = executor.read_value("logits")
    assert isinstance(value, torch.Tensor) and value.shape[0] == len(rows)
