"""Metrics, readouts and eligibility through both engines (cases
M1–M7, P2's whole-vocabulary half, P9, P11).

Every test here is at the engine seam: one document, ``run_protocol`` with
``PytorchHooksEngine()`` and with ``NnsightEngine()``, the two output
directories compared by ``tests._helpers.engines.compare_run_dirs``:
tensors at ``engines.ATOL``, JSON tables row by row (floats at ``ATOL``, ids
and labels exact), the run receipt whole after
``engines.KNOWN_RECORD_DIFFERENCES``. Aggregation lowering, eligibility and
the receipt live above ``_run_group`` (``causalab/neural/shared/execution.py``),
so what these tests pin is that the second engine's captures feed that
shared layer the same numbers.

At protocol 4 an aggregation sits on the save entry that consumes it
(``save[i].aggregation``). The tables it writes are compared as before. The
documents are the reference engine's own (imported from its test modules,
never copied) or the corpus's. Refusals that are protocol policy are caught
on both engines and compared as text.

Three shared-parser features run here as well: two aggregations over one
read, a role-less ``data`` block and inline ``inputs``.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

from causalab.io.env import (
    FileArtifacts,
    FileDatasets,
    ResolutionEnv,
    read_safetensors_metadata,
)
from causalab.io.tables import read_table
from causalab.io.tensor_files import load_file
from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.shared.execution import SCORING_KEY
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.compiled import CompiledProtocol
from causalab.protocol.results import Available, Unavailable
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION, RETIRED_TOKEN_FORMS

from tests._helpers import engines
from tests.neural.engines.nnsight_tracing.conftest import TINY_LLAMA
from tests.neural.engines.pytorch_hooks.test_alignment_run import (
    AMBIGUOUS_ROWS as VARIABLE_AMBIGUOUS_ROWS,
)
from tests.neural.engines.pytorch_hooks.test_alignment_run import _variable_read_doc
from tests.neural.engines.pytorch_hooks.test_eligibility_run import (
    AMBIGUOUS_ROWS,
    ANSWERED_ROWS,
    NULL_ANSWER_ROWS,
)
from tests.neural.engines.pytorch_hooks.test_eligibility_run import (
    _doc as eligibility_doc,
)
from tests.neural.engines.pytorch_hooks.test_generate_metrics import (
    PROMPTS as GENERATE_PROMPTS,
)
from tests.neural.engines.pytorch_hooks.test_generate_metrics import (
    _doc as generate_doc,
)
from tests.neural.engines.pytorch_hooks.test_scoring_run import (
    SOR_REF,
    _sor_document,
)
from tests.neural.engines.pytorch_hooks.test_token_logits_run import ANSWERS
from tests.protocol._docs import (
    aggregation,
    base_only_doc,
    in_order,
    inline_doc,
    saved,
)
from tests.protocol._env import CORPUS_DIR

pytestmark = pytest.mark.smoke

#: tiny-random is two layers deep, so a shipped L18 site is retargeted
LLAMA_OVERRIDES = {"model.key": TINY_LLAMA, "sites.target.layers": 1}

#: the fixture answer space, space-prefixed as the weekdays table carries it
WEEKDAYS = [" Monday", " Friday", " Saturday", " Sunday"]

INTERCHANGE = CORPUS_DIR / "02_interchange_im.json"


# --------------------------------------------------------------------------- #
# environments
# --------------------------------------------------------------------------- #


def _inline_env(root: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]):
    """A resolution environment over ``tables`` (ref -> rows) written under
    ``root``; every row declared in the one undivided ``all`` split (§2.2)."""
    for ref, rows in tables.items():
        path = root / "data" / f"{ref}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"split": "all", **row} for row in rows]))
    return ResolutionEnv(
        datasets=FileDatasets(root=root / "data"),
        artifacts=FileArtifacts(root=root / "artifacts"),
    )


def _run_inline(
    base: Path, doc: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping]]
) -> engines.BothRuns:
    env = _inline_env(base, tables)
    return engines.run_both(in_order(dict(doc)), env, base / "out")


def _run_corpus(base: Path, doc: Path | Mapping[str, Any], overrides=None):
    env = engines.corpus_env(base / "artifacts")
    return engines.run_both(doc, env, base / "out", overrides=overrides)


def _refusal(
    document: CompiledProtocol | Mapping[str, Any], env, engine, out: Path
) -> str:
    with pytest.raises(ProtocolError) as err:
        run_protocol(document, env, engine, out)
    return str(err.value)


def _both_refuse(document, env, base: Path) -> str:
    """The refusal text, asserted identical on both engines."""
    hooks = _refusal(document, env, PytorchHooksEngine(), base / "hooks")
    trace = _refusal(document, env, NnsightEngine(), base / "trace")
    assert hooks == trace, f"refusals differ:\n  hooks: {hooks}\n  trace: {trace}"
    return hooks


def _assert_write_landed(run_dir: Path, patched: str, clean: str) -> None:
    """Anti-vacuity for a document with a write: the patched logits differ
    from the unpatched ones on this engine's own output."""
    a = load_file(str(run_dir / f"{patched}.safetensors"))[patched]
    b = load_file(str(run_dir / f"{clean}.safetensors"))[clean]
    assert not torch.allclose(a, b, atol=engines.ATOL), (
        f"{run_dir.name}: the write did not move the logits — parity is vacuous"
    )


# --------------------------------------------------------------------------- #
# M1 — logit_diff / match on the task-table interchange (the corpus's 10)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def task_table_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    base = tmp_path_factory.mktemp("m1")
    return _run_corpus(
        base,
        CORPUS_DIR / "10_task_table_iia_im.json",
        overrides={**LLAMA_OVERRIDES, "model.dtype": "fp32"},
    )


def test_m1_the_task_table_interchange_agrees(task_table_runs):
    """M1's second half: ``match`` under ``first_token`` over the task's
    ``label_forms``, a ragged ``column`` read (``subject_acts``), and the
    patched / clean logits — all equal across engines. The document's own
    guard from ``test_end_to_end_iia.py`` holds on both sides: the swap moves
    the answer-position logits."""
    for run_dir in (task_table_runs.hooks_dir, task_table_runs.trace_dir):
        _assert_write_landed(run_dir, "logits", "logits_clean")
    assert sorted(task_table_runs.hooks_result.files) == sorted(
        task_table_runs.trace_result.files
    )
    task_table_runs.compare()


# --------------------------------------------------------------------------- #
# M2 / M3 — the answer-space readouts and the whole-vocabulary divergences
# --------------------------------------------------------------------------- #

#: the patched-logits aggregations, one save entry each (label = file stem)
READOUTS: dict[str, dict[str, Any]] = {
    # M2
    "soft_accuracy": aggregation("soft_accuracy", a="cf_answer", b="base_answer"),
    "token_logit": aggregation("token_logit", token="cf_answer"),
    "token_logits": aggregation("token_logits", tokens=ANSWERS),
    "cross_entropy": aggregation("cross_entropy", target="cf_answer"),
    # M3: toward the un-intervened model's reads, whole vocabulary, on the
    # same prompt and on the counterfactual prompt. `restrict` is the literal
    # list form; the fixture table has no list column (§2.10). A target inside
    # a save names its model too (§2.7; the bare name is refused there).
    "kl": aggregation("kl", target={"read": "logits_clean", "model": "original"}),
    "js": aggregation("js", target={"read": "logits_clean", "model": "original"}),
    "js_restricted": aggregation(
        "js", target={"read": "logits_clean", "model": "original"}, restrict=WEEKDAYS
    ),
    "kl_cf": aggregation(
        "kl", target={"read": "logits_cf", "model": "original_counterfactual"}
    ),
    "js_cf": aggregation(
        "js", target={"read": "logits_cf", "model": "original_counterfactual"}
    ),
    "js_cf_restricted": aggregation(
        "js",
        target={"read": "logits_cf", "model": "original_counterfactual"},
        restrict=WEEKDAYS,
    ),
}

DIVERGENCES = ("kl", "js", "js_restricted", "kl_cf", "js_cf", "js_cf_restricted")


def _readouts_doc() -> dict[str, Any]:
    """The corpus interchange on tiny-random Llama with every token-space and
    distribution-space kind over the patched ``lm_head@-1`` read, the
    divergences toward the same site read on the un-intervened model."""
    raw = json.loads(INTERCHANGE.read_text())
    raw["model"]["key"] = TINY_LLAMA
    method = raw["method"]
    # layer 0, not the fixture's last layer: a swap at the last layer's
    # `block_output@-1` makes the patched logits *equal* the counterfactual's
    # (nothing sits between them and lm_head), and a divergence toward that
    # read would be exactly zero
    method["sites"]["target"]["layers"] = 0
    method["reads"]["logits_clean"] = {"site": "lm_head", "pos": -1}
    method["reads"]["logits_cf"] = {"site": "lm_head", "pos": -1}
    method["intervened_models"]["original"] = {
        "input": "base",
        "reads": ["logits_clean"],
    }
    method["intervened_models"]["original_counterfactual"]["reads"].append("logits_cf")
    method["save"] = [
        *(
            saved("logits", "patched", f"{name}.json", dict(agg))
            for name, agg in READOUTS.items()
        ),
        saved("logits", "patched", "logits.safetensors"),
        saved("logits_clean", "original", "logits_clean.safetensors"),
    ]
    return raw


@pytest.fixture(scope="module")
def readouts_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    base = tmp_path_factory.mktemp("m2m3")
    return _run_corpus(base, _readouts_doc())


def test_m2_m3_every_readout_and_divergence_table_agrees(readouts_runs):
    """M2 (`soft_accuracy`, `token_logit`, `token_logits`, `cross_entropy`)
    and M3 (`kl`, `js`, `js` under `restrict`, toward the original model on
    the same and on the counterfactual prompt) — ten tables plus the logit
    saves, equal across engines. Closes P2's whole-vocabulary metric half:
    the divergences consume every one of the 32000 logits of both reads."""
    for run_dir in (readouts_runs.hooks_dir, readouts_runs.trace_dir):
        _assert_write_landed(run_dir, "logits", "logits_clean")
        # the divergences are not zero on either side — the swap moved the
        # distribution, so a `kl`/`js` agreement is a claim about numbers
        for name in DIVERGENCES:
            values = [row["value"] for row in read_table(run_dir / f"{name}.json")]
            assert all(v > 0.0 for v in values), (run_dir.name, name, values)
    readouts_runs.compare()


def test_m2_the_token_logits_save_carries_the_answer_space_identically(
    readouts_runs,
):
    """The `token_logits` value is three parallel lists. The comparer decodes
    it already; this pins the ids exact and the answer space in document
    order, so a table that dropped a token cannot pass as equal."""
    hooks = read_table(readouts_runs.hooks_dir / "token_logits.json")
    trace = read_table(readouts_runs.trace_dir / "token_logits.json")
    assert len(hooks) == len(trace) > 0
    for a, b in zip(hooks, trace):
        va, vb = json.loads(a["value"]), json.loads(b["value"])
        assert va["indices"] == vb["indices"]
        assert va["tokens"] == vb["tokens"]
        assert [t.strip() for t in va["tokens"]] == ANSWERS
        assert va["values"] == pytest.approx(vb["values"], abs=engines.ATOL)


def test_m7_the_metric_row_identity_columns_are_present_and_equal(readouts_runs):
    """M7: `unit` and `estimand_version`, the estimand identity every metric
    row carries (§2.10, `metric_record_identity`), with `metric` and
    `eligible`, present and equal in one table. The old row column
    `produced_by` no longer exists anywhere under `causalab/`."""
    hooks = read_table(readouts_runs.hooks_dir / "soft_accuracy.json")
    trace = read_table(readouts_runs.trace_dir / "soft_accuracy.json")
    assert hooks and len(hooks) == len(trace)
    for a, b in zip(hooks, trace):
        for column in ("metric", "unit", "estimand_version", "eligible"):
            assert column in a and column in b, column
            assert a[column] == b[column], (column, a[column], b[column])
    assert {row["unit"] for row in hooks} == {"fraction"}
    assert {row["estimand_version"] for row in hooks} == {"soft_accuracy/v1"}


# --------------------------------------------------------------------------- #
# two aggregations over one read
# --------------------------------------------------------------------------- #


def _two_aggregations_doc() -> dict[str, Any]:
    """The corpus interchange with its `logit_diff` save joined by a
    `soft_accuracy` over the same read on the same model: one read, one
    capture, two tables."""
    raw = json.loads(INTERCHANGE.read_text())
    raw["method"]["save"].append(
        saved(
            "logits",
            "patched",
            "soft_accuracy.json",
            aggregation("soft_accuracy", a="cf_answer", b="base_answer"),
        )
    )
    return raw


def test_two_aggregations_over_one_read_agree_and_share_the_read(tmp_path):
    """Protocol 4 puts each aggregation on its save entry, so one read can
    feed several tables. Both engines write the same `logit_diff`, `match` and
    `soft_accuracy` tables. On each engine `soft_accuracy` is σ(`logit_diff`)
    row by row (§2.10 defines it as σ of the same margin), which shows the
    two tables reduce one captured value rather than two captures."""
    runs = _run_corpus(tmp_path, _two_aggregations_doc(), overrides=LLAMA_OVERRIDES)
    runs.compare()
    for run_dir in (runs.hooks_dir, runs.trace_dir):
        margin = read_table(run_dir / "logit_diff.json")
        soft = read_table(run_dir / "soft_accuracy.json")
        assert len(margin) == len(soft) > 0
        for m, s in zip(margin, soft):
            assert m["example_id"] == s["example_id"]
            expected = 1.0 / (1.0 + math.exp(-m["value"]))
            assert s["value"] == pytest.approx(expected, abs=1e-6), run_dir.name
        # not a constant table: the margins differ across rows, so the
        # sigmoid check is not satisfied by one repeated number
        assert len({round(m["value"], 6) for m in margin}) > 1, run_dir.name


# --------------------------------------------------------------------------- #
# a role-less data block and inline inputs
# --------------------------------------------------------------------------- #

#: `base_doc` names gpt2 at layer 3; tiny-random Llama has two layers
BASE_DOC_OVERRIDES = {"model.key": TINY_LLAMA, "sites.tgt.layers": 1}


def test_a_role_less_data_block_agrees_across_engines(tmp_path):
    """A role-less block: ``data: {"dataset", "field"}`` with no role wrapper means
    ``base``. The `base_only_doc` ablation (a literal zero swapped in at the
    last position) in that short spelling runs on both engines and writes the
    same `logit_diff` table. The canonical `data` block names `base` on both
    sides, which the receipt comparison covers."""
    raw = base_only_doc()
    raw["data"] = raw["data"]["base"]
    assert set(raw["data"]) == {"dataset", "field"}
    runs = _run_corpus(tmp_path, raw, overrides=BASE_DOC_OVERRIDES)
    runs.compare()
    for run_dir in (runs.hooks_dir, runs.trace_dir):
        table = read_table(run_dir / "ld.json")
        assert table and all(isinstance(row["value"], float) for row in table)
        record = json.loads((run_dir / RUN_RECORD_NAME).read_text())
        assert set(record["canonical"]["data"]) == {"base"}


INLINE_PROMPTS = ("The Space Needle is located in", "The Eiffel Tower is in")


def test_inline_inputs_agree_across_engines_with_no_data_root(tmp_path):
    """Inline inputs: the prompts are the data (``{"inputs": [...]}``), so the run
    needs no table on disk. The dataset root here is an empty directory. Both
    engines write the same `class_probs` table over ``" Seattle"``, one
    token on tiny-random Llama, and the rows are the inline prompts."""
    empty = tmp_path / "no_data"
    empty.mkdir()
    env = ResolutionEnv(
        datasets=FileDatasets(root=empty),
        artifacts=FileArtifacts(root=tmp_path / "artifacts"),
    )
    runs = engines.run_both(
        inline_doc(*INLINE_PROMPTS), env, tmp_path / "out", overrides=BASE_DOC_OVERRIDES
    )
    runs.compare()
    assert not any(empty.iterdir()), "an inline run wrote into the data root"
    for run_dir in (runs.hooks_dir, runs.trace_dir):
        table = read_table(run_dir / "ld.json")
        assert len(table) == len(INLINE_PROMPTS)
        for row in table:
            probs = json.loads(row["value"])
            assert set(probs) == {"Seattle"} and 0.0 < probs["Seattle"] < 1.0


# --------------------------------------------------------------------------- #
# M4 — class_probs and top_k, on the vocabulary axis and on a residual axis
# --------------------------------------------------------------------------- #


def _top_k_doc() -> dict[str, Any]:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "weekdays/data#train", "field": "input"}},
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["logits", "resid"]}
            },
            "sites": {
                "lm_head": {"component": "lm_head"},
                "resid_site": {"component": "block_output", "layers": [1]},
            },
            "reads": {
                "logits": {"site": "lm_head", "pos": -1},
                "resid": {"site": "resid_site", "pos": -1},
            },
            "save": [
                saved(
                    "logits",
                    "original",
                    "classes.json",
                    aggregation(
                        "class_probs",
                        groups={
                            "weekday": [" Monday", " Friday"],
                            "weekend": [" Saturday", " Sunday"],
                        },
                    ),
                ),
                saved(
                    "logits",
                    "original",
                    "top_vocab.json",
                    aggregation("top_k", k=5, by="prob"),
                ),
                saved(
                    "resid",
                    "original",
                    "top_resid_abs.json",
                    aggregation("top_k", k=3, by="abs_value"),
                ),
                saved(
                    "resid",
                    "original",
                    "top_resid.json",
                    aggregation("top_k", k=3, by="value"),
                ),
            ],
        },
    }


@pytest.fixture(scope="module")
def top_k_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    base = tmp_path_factory.mktemp("m4")
    return _run_corpus(base, _top_k_doc())


def test_m4_class_probs_and_top_k_agree(top_k_runs):
    """M4: the grouped probabilities and the three rankings — on the
    vocabulary (`tokens` and `probs` columns present) and on a 16-wide
    residual read under `by: value` / `by: abs_value` (no such columns)."""
    top_k_runs.compare()
    for name, vocab in (
        ("top_vocab", True),
        ("top_resid_abs", False),
        ("top_resid", False),
    ):
        hooks = read_table(top_k_runs.hooks_dir / f"{name}.json")
        trace = read_table(top_k_runs.trace_dir / f"{name}.json")
        assert len(hooks) == len(trace) > 0
        for a, b in zip(hooks, trace):
            va, vb = json.loads(a["value"]), json.loads(b["value"])
            assert va["indices"] == vb["indices"], name  # ranks exact
            assert ("tokens" in va) is vocab and ("probs" in va) is (
                vocab and name == "top_vocab"
            )
            assert va["values"] == pytest.approx(vb["values"], abs=engines.ATOL)
    classes = read_table(top_k_runs.hooks_dir / "classes.json")
    assert all(set(json.loads(r["value"])) == {"weekday", "weekend"} for r in classes)


# --------------------------------------------------------------------------- #
# M5 — retired token forms, answer-id collisions, check_scoring
# --------------------------------------------------------------------------- #
#
# `token_form` rewriting is retired: an answer string is tokenized as
# written, and `token_form` is `id` or absent. The old M5 legs that compared
# the `auto` ambiguity refusal and `check_answer_forms` (a bare form over a
# space-prefixed table) went with the feature. No form is left to guess, and
# a document has no second knob to contradict its table with. What remains
# shared policy is the refusal of a retired value and the refusal of two
# strings that land on one id.


@pytest.mark.parametrize("form", RETIRED_TOKEN_FORMS)
def test_m5_a_retired_token_form_is_refused_identically(tmp_path, form):
    """A protocol-4 document that still carries a retired value is refused by
    name at parse, before either engine is entered: the same text on both,
    and neither output directory is created."""
    raw = json.loads(INTERCHANGE.read_text())
    raw["method"]["save"][0]["aggregation"]["token_form"] = form
    env = engines.corpus_env(tmp_path / "artifacts")
    message = _both_refuse(raw, env, tmp_path)
    assert message.startswith("[P4] at save[0].aggregation.token_form")
    assert f"token_form {form!r} was retired" in message
    assert "tokenized as written" in message
    for side in ("hooks", "trace"):
        assert not (tmp_path / side).exists()


#: "Monday" and " Monday" are one sentencepiece piece on tiny-random Llama
#: (id 27822; `test_metrics.py::test_the_two_forms_collapse_on_sentencepiece`)
COLLIDING = ["Monday", " Monday"]


def _collision_doc(kind: str) -> dict[str, Any]:
    """The un-intervened logits reduced by one list kind whose literal list
    names ``COLLIDING``. The `js` document adds the counterfactual prompt's
    logits as its target, so the restricted divergence has two reads to
    compare. The other kinds leave that read out, since an unused read is
    refused as dead (V11)."""
    fields: dict[str, dict[str, Any]] = {
        "class_probs": {"groups": {"m": COLLIDING}},
        "token_logits": {"tokens": COLLIDING},
        "js": {
            "target": {"read": "logits_cf", "model": "original_counterfactual"},
            "restrict": COLLIDING,
        },
    }
    doc: dict[str, Any] = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "weekdays/data#train", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [
                saved(
                    "logits",
                    "original",
                    "m.json",
                    aggregation(kind, **fields[kind]),
                )
            ],
        },
    }
    if kind == "js":
        doc["data"]["counterfactual"] = {
            "dataset": "weekdays/data#train",
            "field": "counterfactual_inputs[0]",
        }
        doc["method"]["intervened_models"]["original_counterfactual"] = {
            "input": "counterfactual",
            "reads": ["logits_cf"],
        }
        doc["method"]["reads"]["logits_cf"] = {"site": "lm_head", "pos": -1}
    return doc


@pytest.mark.parametrize(
    "kind, field",
    [
        ("class_probs", "groups.m"),
        ("token_logits", "tokens"),
        ("js", "restrict"),
    ],
)
def test_m5_two_strings_on_one_id_are_refused_identically(tmp_path, kind, field):
    """The shared metric layer (``causalab/neural/shared/metrics.py``,
    ``distinct_token_ids`` and ``restrict_token_ids``) refuses a list whose
    two strings resolve to one id, since the kind would count that row twice.
    Only the tokenizer can see it, so the refusal comes after the forward.
    Both engines raise the same text, and neither writes the table."""
    env = engines.corpus_env(tmp_path / "artifacts")
    compiled = engines.compile_for_both(_collision_doc(kind), env)
    message = _both_refuse(compiled, env, tmp_path)
    assert f"metric {kind}.{field}: " in message
    assert message.startswith("[P2]")
    assert "both resolve to token id" in message
    assert "'Monday'" in message and "' Monday'" in message
    for side in ("hooks", "trace"):
        assert not (tmp_path / side / "m.json").exists()


def _recorded_env(root: Path) -> ResolutionEnv:
    """A task-serialized table carrying its scoring identity (§2.2) —
    `test_scoring_run.py`'s `recorded_env`, built here for both engines."""
    from causalab.tasks.serialize import (
        serialize_counterfactual_dataset,
        write_dataset_table,
    )
    from causalab.tasks.subject_object_relations.config import (
        SubjectObjectRelationsConfig,
    )

    dataset = serialize_counterfactual_dataset(
        "subject_object_relations",
        n=4,
        seed=0,
        split="all",
        task_cfg=SubjectObjectRelationsConfig(relation="name_gender"),
    )
    write_dataset_table(dataset.rows, root / f"{SOR_REF}.json")
    return ResolutionEnv(
        datasets=FileDatasets(root=root), artifacts=FileArtifacts(root=root)
    )


def test_m5_the_scoring_block_of_a_valid_run_is_identical(tmp_path):
    """`check_scoring` on a recorded `prefix` table under `first_token`: the
    receipt's `scoring.<ref>` block says `ok` on both engines (and the whole
    receipt agrees)."""
    env = _recorded_env(tmp_path / "tables")
    runs = engines.run_both(_sor_document("first_token"), env, tmp_path / "out")
    runs.compare()
    hooks = json.loads((runs.hooks_dir / RUN_RECORD_NAME).read_text())[SCORING_KEY]
    trace = json.loads((runs.trace_dir / RUN_RECORD_NAME).read_text())[SCORING_KEY]
    assert hooks == trace
    assert hooks == {SOR_REF: {"string_mode": "prefix", "result": "ok"}}


# --------------------------------------------------------------------------- #
# M6 / R3 — eligibility, excluded rows, unavailable cells, the denominator
# --------------------------------------------------------------------------- #


def _assert_cells_agree(runs: engines.BothRuns) -> None:
    """`RunResult.cells` and the denominator, from the returned results —
    the typed cells (`Available.mapping`, `Unavailable.reason`/`detail`)
    rather than their file form only."""
    assert runs.hooks_result.cells == runs.trace_result.cells
    assert runs.hooks_result.cells, "no cells — the comparison is vacuous"
    assert (
        runs.hooks_result.denominator.render() == runs.trace_result.denominator.render()
    )


@pytest.mark.parametrize(
    "pos, rows, excluded_row, reason",
    [
        ("ent", AMBIGUOUS_ROWS, 0, "alignment_ambiguous"),
        (-1, NULL_ANSWER_ROWS, 1, "alignment_missing"),
    ],
    ids=["ambiguous_variable", "null_answer"],
)
def test_m6_an_excluded_row_is_excluded_on_both_engines(
    tmp_path, pos, rows, excluded_row, reason
):
    """T1 of `test_eligibility_run.py` through both engines: the excluded
    row's `eligible: false` and `reason_code`, the other rows' values, the
    cell's `n_eligible` / `n_considered`, the denominator."""
    runs = _run_inline(tmp_path, eligibility_doc(pos=pos), {"elig/rows": rows})
    runs.compare()
    _assert_cells_agree(runs)
    for run_dir in (runs.hooks_dir, runs.trace_dir):
        table = sorted(read_table(run_dir / "tl.json"), key=lambda r: r["example_id"])
        assert table[excluded_row]["eligible"] is False
        assert table[excluded_row]["reason_code"] == reason
        assert table[excluded_row]["value"] is None
    (cell,) = runs.trace_result.cells
    assert isinstance(cell, Available)
    assert cell.mapping["n_eligible"] == len(rows) - 1
    assert cell.mapping["n_considered"] == len(rows)
    assert runs.trace_result.denominator.render() == "1 / 1 eligible"


def test_m6_the_twin_every_row_answered_agrees(tmp_path):
    runs = _run_inline(tmp_path, eligibility_doc(pos=-1), {"elig/rows": ANSWERED_ROWS})
    runs.compare()
    _assert_cells_agree(runs)
    for run_dir in (runs.hooks_dir, runs.trace_dir):
        table = read_table(run_dir / "tl.json")
        assert all(row["eligible"] is True for row in table)
        assert not any("reason_code" in row for row in table)


def test_m6_r3_a_metric_whose_every_row_is_excluded_is_unavailable_on_both(
    tmp_path,
):
    """R3: the `unavailable` cell and its reason code appear identically —
    in `RunResult.cells`, in the denominator, in the receipt."""
    rows = [{**row, "entity": None} for row in NULL_ANSWER_ROWS]
    runs = _run_inline(tmp_path, eligibility_doc(pos=-1), {"elig/rows": rows})
    runs.compare()
    _assert_cells_agree(runs)
    for result in (runs.hooks_result, runs.trace_result):
        (cell,) = result.cells
        assert isinstance(cell, Unavailable)
        assert cell.reason == "alignment_missing"
        assert result.denominator.render() == (
            "0 / 1 eligible; 1 excluded: alignment_missing ×1"
        )


def test_m6_r3_an_ambiguous_variable_row_is_an_unavailable_read_cell_on_both(
    tmp_path,
):
    """`test_alignment_run.py`'s read-plus-metric document: the *read's* cell
    is `alignment_ambiguous` (width zero on that row, `status: unavailable`
    in the saved entry's record) and the metric inherits row by row — the
    same on both engines, tensors, entries table and cells."""
    runs = _run_inline(
        tmp_path,
        _variable_read_doc(with_metric=True, model_key=TINY_LLAMA),
        {"amb/rows": VARIABLE_AMBIGUOUS_ROWS},
    )
    runs.compare()
    _assert_cells_agree(runs)
    for result, run_dir in (
        (runs.hooks_result, runs.hooks_dir),
        (runs.trace_result, runs.trace_dir),
    ):
        read_cell, metric_cell = result.cells
        assert isinstance(read_cell, Unavailable)
        assert read_cell.reason == "alignment_ambiguous"
        assert isinstance(metric_cell, Available)
        assert metric_cell.mapping["excluded"] == {"alignment_ambiguous": 1}
        assert result.denominator.render() == (
            "1 / 2 eligible; 1 excluded: alignment_ambiguous ×1"
        )
        stamped = read_safetensors_metadata(run_dir / "r.safetensors")
        assert stamped is not None
        record = json.loads(stamped["entries"])["r"]
        assert record["status"] == "unavailable"
        assert record["reason"] == "alignment_ambiguous"
        assert load_file(str(run_dir / "r.safetensors"))["r.widths"].tolist()[0] == 0


# --------------------------------------------------------------------------- #
# P9 / P11 — `decode` and per-step reduction over a generated window
# --------------------------------------------------------------------------- #

#: per-row answer columns for a windowed `logit_diff`: one sentencepiece
#: piece each on tiny-random Llama
GENERATE_ROWS = [
    {"input": prompt, "x": " Monday", "y": " Friday"} for prompt in GENERATE_PROMPTS
]


def _generate_doc() -> dict[str, Any]:
    """`test_generate_metrics.py`'s document with three aggregations over the
    one continuation read: `decode` (P9, the ids domain), `top_k` per step and
    `logit_diff` per step (P11, `compute_windowed_metric`)."""
    doc = copy.deepcopy(generate_doc(aggregation("decode"), anchor={"all": True}))
    doc["method"]["save"] = [
        saved("cont", "original", "said.json", aggregation("decode")),
        saved("cont", "original", "top.json", aggregation("top_k", k=1, by="prob")),
        saved(
            "cont",
            "original",
            "margin.json",
            aggregation("logit_diff", a="x", b="y"),
        ),
    ]
    return doc


@pytest.fixture(scope="module")
def generate_runs(tmp_path_factory: pytest.TempPathFactory) -> engines.BothRuns:
    base = tmp_path_factory.mktemp("p9p11")
    return _run_inline(base, _generate_doc(), {"probe": GENERATE_ROWS})


def test_p9_decode_returns_the_same_text_on_both_engines(generate_runs):
    """P9: the greedy continuation's decoded text, one string per example,
    equal — the ids the two decodes produced are the same ids."""
    generate_runs.compare()
    hooks = read_table(generate_runs.hooks_dir / "said.json")
    trace = read_table(generate_runs.trace_dir / "said.json")
    assert [r["value"] for r in hooks] == [r["value"] for r in trace]
    assert len(hooks) == len(GENERATE_PROMPTS)
    assert all(isinstance(r["value"], str) and r["value"] for r in hooks)
    assert all(r["matched"] is True for r in hooks)


def test_p11_per_step_tables_name_the_same_steps_and_values(generate_runs):
    """P11: the windowed reduction — one row per generated step, `step` an
    integer label — agrees row for row on `top_k` (rank exact) and on a
    `logit_diff` over the window (values at ATOL)."""
    for name in ("top", "margin"):
        hooks = read_table(generate_runs.hooks_dir / f"{name}.json")
        trace = read_table(generate_runs.trace_dir / f"{name}.json")
        assert len(hooks) == len(trace) > len(GENERATE_PROMPTS)  # several steps
        assert [r["step"] for r in hooks] == [r["step"] for r in trace]
        assert all(isinstance(r["step"], int) for r in hooks)
        assert all(r["matched"] is True for r in hooks)
    top = read_table(generate_runs.hooks_dir / "top.json")
    top_trace = read_table(generate_runs.trace_dir / "top.json")
    for a, b in zip(top, top_trace):
        assert json.loads(a["value"])["indices"] == json.loads(b["value"])["indices"]
    margin = read_table(generate_runs.hooks_dir / "margin.json")
    margin_trace = read_table(generate_runs.trace_dir / "margin.json")
    for a, b in zip(margin, margin_trace):
        assert isinstance(a["value"], float)
        assert a["value"] == pytest.approx(b["value"], abs=engines.ATOL)
