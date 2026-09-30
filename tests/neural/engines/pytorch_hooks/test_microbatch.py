"""Microbatching inside a forward group (§8, execution scale).

A forward group over ``N`` rows may run as ceil(``N`` / ``b``) forwards of at
most ``b`` rows each, and nothing a document can observe may change: each
window's captures concatenate in row order before a read gathers, a write
whose operand was read in another group indexes that operand by row, ragged
values keep their flat rows and widths, and a decoding group decodes each
window from its own prefill. ``b`` is an execution parameter — an engine
constructor argument and ``--batch-rows`` on ``run`` — so the digests and every
stamp are byte-identical between a whole-batch and a microbatched run, and the
run receipt differs in exactly one place: ``execution.batch_rows``, the one
recorder of batch geometry (§8; recorded, not gated).

Every equality test here runs the same document twice, once whole and once
with ``b`` set to a value that does not divide the row count (so the last
window is short) and one that does, and compares what the executor holds:
tensors with `torch.testing.assert_close` at the fixture dtype's
tolerance — a different batch shape may take a different kernel path, so
bit-equality is not the claim — and everything integral (token ids, routing
tables, widths, decoded text) exactly.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.neural.engines.pytorch_hooks.executor import (
    PointExecutor,
    RaggedValue,
    _concat_rows,
)
from causalab.neural.shared.metrics import compute_metric, compute_windowed_metric
from causalab.protocol import RUN_RECORD_NAME, run_protocol
from causalab.protocol.pipeline import compile_protocol
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import AggregationSpec
from causalab.io.tables import read_table

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, saved
from tests.protocol._env import CORPUS_DIR, FIXTURES, write_rot_fixture

#: Six rows of unequal length, so left padding is in play and the windows
#: below cut the batch at 2 (divides) and 4 (does not).
BASE_TEXTS = [
    "the quick brown fox jumps",
    "a slow green turtle sleeps deeply today",
    "rain",
    "seven bright lanterns hung over the quiet harbor",
    "she opened the door and",
    "why is the sky blue at noon",
]
COUNTERFACTUAL_TEXTS = [
    "a slow green turtle sleeps",
    "the quick brown fox jumps over the lazy dog",
    "snow",
    "one dim candle stood on the loud table",
    "he closed the window and",
    "why is the sea green at dusk",
]
BATCH_ROWS = (2, 4)
BUDGET = 5

#: assert_close tolerances per fixture dtype: fp32 is what the tiny-random
#: fixtures load in; bf16 is listed for a run that retargets them.
TOLERANCE = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-5},
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-2},
}


def _assert_same(whole: Any, windowed: Any) -> None:
    """Whole-batch and microbatched values agree: floats within the dtype's
    tolerance, everything integral exactly, ragged widths exactly."""
    if isinstance(whole, RaggedValue):
        assert isinstance(windowed, RaggedValue)
        assert whole.widths == windowed.widths
        _assert_same(whole.flat, windowed.flat)
        return
    assert isinstance(whole, torch.Tensor) and isinstance(windowed, torch.Tensor)
    assert whole.shape == windowed.shape
    assert whole.dtype == windowed.dtype
    if not whole.is_floating_point():
        assert torch.equal(whole, windowed)
        return
    torch.testing.assert_close(windowed, whole, **TOLERANCE[whole.dtype])


def _both(
    doc: dict[str, Any],
    bundle: Any,
    batch_rows: int,
    *,
    counterfactual: bool = False,
    columns: dict[str, list[Any]] | None = None,
) -> tuple[PointExecutor, PointExecutor]:
    """The same document run whole and in windows of ``batch_rows``."""
    kwargs: dict[str, Any] = {"base_texts": BASE_TEXTS, "extra_columns": columns}
    if counterfactual:
        kwargs["counterfactual_texts"] = COUNTERFACTUAL_TEXTS
    whole = executor_for(doc, bundle, **kwargs)
    windowed = executor_for(doc, bundle, batch_rows=batch_rows, **kwargs)
    whole_forwards = _run_counting_forwards(whole)
    windowed_forwards = _run_counting_forwards(windowed)
    # the witness that windowing *happened*: every equality below also holds
    # for an executor that ignores its bound, so count the forwards the model
    # saw — one per group whole, ceil(rows / b) per group windowed
    groups = len(whole._groups_run)
    assert groups > 0 and len(windowed._groups_run) == groups
    assert whole_forwards == groups
    assert windowed_forwards == groups * math.ceil(len(BASE_TEXTS) / batch_rows)
    return whole, windowed


def _run_counting_forwards(executor: PointExecutor) -> int:
    """``run_all`` with a pre-hook on the model, returning how many forwards
    it ran — prefills only, since a decoding group's per-step forwards
    (the ones carrying a ``past_key_values`` cache) follow each window's
    prefill and would count the decode budget, not the layout."""
    prefills = 0

    def count(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        nonlocal prefills
        if kwargs.get("past_key_values") is None:
            prefills += 1

    handle = executor.bundle.model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        executor.run_all()
    finally:
        handle.remove()
    return prefills


def _save(*names: tuple[str, str]) -> list[dict[str, Any]]:
    """One tensor save per ``(read, model)``, filed under the read's name."""
    return [saved(read, model, f"{read}.safetensors") for read, model in names]


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #


def _read_doc() -> dict[str, Any]:
    """Two ``block_output`` reads on the two-layer fixture: one over every
    padded position (a dense frame-wide gather) and one at the last token."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["all0", "last1"]}
            },
            "sites": {
                "l0": {"component": "block_output", "layers": [0]},
                "l1": {"component": "block_output", "layers": [1]},
            },
            "reads": {
                "all0": {"site": "l0", "pos": "all"},
                "last1": {"site": "l1", "pos": -1},
            },
            "save": _save(("all0", "original"), ("last1", "original")),
        },
    }


def _patch_doc(do: dict[str, Any] | None = None) -> dict[str, Any]:
    """A write at the last token of layer 1, read out at ``lm_head``.

    The default is an interchange whose operand is a read on the
    *counterfactual* input — a different forward group, so the write indexes
    it by row. Any other mechanism carries its own operand and needs no
    second role."""
    swap = do is None
    # the un-intervened model on base goes by `original` alone unless the
    # network is also read un-intervened on the counterfactual (§2.9)
    clean_model = "original_base" if swap else "original"
    doc: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=swap),
        "method": {
            "intervened_models": {
                clean_model: {"input": "base", "reads": ["clean"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "target": {"component": "block_output", "layers": [1]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "clean": {"site": "lm_head", "pos": -1},
                "after": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                "patch": {"site": "target", "pos": -1, "do": do or {"swap": "v_cf"}}
            },
            "save": _save(("clean", clean_model), ("after", "patched")),
        },
    }
    if swap:
        doc["method"]["reads"]["v_cf"] = {"site": "target", "pos": -1}
        doc["method"]["intervened_models"] = {
            UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
            **doc["method"]["intervened_models"],
        }
    return doc


def _generate_doc() -> dict[str, Any]:
    """A continuation read at ``lm_head`` (served from kept ``ln_final``) and
    one at a block output (accumulated per step), both over every generated
    step."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["logits", "acts"]}
            },
            "positions": {
                "cont": {"generated": {"max_new_tokens": BUDGET}, "all": True}
            },
            "sites": {
                "lm_head": {"component": "lm_head"},
                "mid": {"component": "block_output", "layers": [1]},
            },
            "reads": {
                "logits": {"site": "lm_head", "pos": "cont"},
                "acts": {"site": "mid", "pos": "cont"},
            },
            "save": _save(("logits", "original"), ("acts", "original")),
        },
    }


def _moe_doc(expert: int) -> dict[str, Any]:
    """The routed interior's three faces at one MoE layer: the ragged
    ``expert:`` face of ``expert_activation``, the routing table it joins on,
    and the router weights."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["face", "idx", "scores"]}
            },
            "sites": {
                "face_site": {
                    "component": "expert_activation",
                    "layers": [0],
                    "expert": expert,
                },
                "idx_site": {"component": "expert_idx", "layers": [0]},
                "scores_site": {"component": "router_scores", "layers": [0]},
            },
            "reads": {
                "face": {"site": "face_site", "pos": "all"},
                "idx": {"site": "idx_site", "pos": "all"},
                "scores": {"site": "scores_site", "pos": "all"},
            },
            "save": _save(
                ("face", "original"), ("idx", "original"), ("scores", "original")
            ),
        },
    }


def _state_patch_doc() -> dict[str, Any]:
    """A ``delta_state`` interchange at the last step: the state writer is
    per (row, step), so its row offset into the operand is what a window
    that does not start at row 0 exercises."""
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                UNWRITTEN: {"input": "counterfactual", "reads": ["s_cf"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "state": {"component": "delta_state", "layers": [0]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "s_cf": {"site": "state", "pos": -1},
                "after": {"site": "lm_head", "pos": -1},
            },
            "writes": {"patch": {"site": "state", "pos": -1, "do": {"swap": "s_cf"}}},
            "save": _save(("after", "patched")),
        },
    }


# --------------------------------------------------------------------------- #
# the execution parameter
# --------------------------------------------------------------------------- #


class TestExecutionParameter:
    pytestmark = pytest.mark.unit

    @pytest.mark.parametrize("bad", (0, -3))
    def test_the_engine_refuses_a_non_positive_row_bound(self, bad: int):
        with pytest.raises(ValueError, match="positive row count"):
            PytorchHooksEngine(batch_rows=bad)

    def test_the_default_is_one_forward_per_group(self):
        assert PytorchHooksEngine().batch_rows is None

    @pytest.mark.parametrize(
        ("batch_rows", "expected"),
        (
            (2, [(0, 2), (2, 4), (4, 6)]),
            (4, [(0, 4), (4, 6)]),
            (6, [(0, 6)]),
            (None, [(0, 6)]),
        ),
    )
    def test_row_windows_cut_at_the_bound(
        self, batch_rows: int | None, expected: list[tuple[int, int]]
    ):
        """The windows one group over six rows runs as: ceil(6 / b) of them,
        contiguous and in row order, the last one short when ``b`` does not
        divide, and the one whole window when ``b`` covers the group or is
        unset. ``_row_windows`` reads nothing but the bound, so an executor
        with only that set is enough — and an executor that always answered
        with the whole window fails here on b=2 and b=4."""
        executor = PointExecutor.__new__(PointExecutor)
        executor.batch_rows = batch_rows
        windows = executor._row_windows(6)
        assert [(w.start, w.stop) for w in windows] == expected
        assert all(w.total == 6 for w in windows)
        assert [row for w in windows for row in range(w.start, w.stop)] == list(
            range(6)
        )

    def test_batch_rows_is_not_document_vocabulary(self, tmp_path, capsys):
        """The bound is execution, never a section of the document: `validate`
        knows nothing about it, so spelling it in a document is refused by the
        ordinary rules — `--set` cannot create a key the document lacks, and a
        file that carries one is an unknown section at parse."""
        artifacts = tmp_path / "artifacts"
        shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
        code = main(
            [
                "validate",
                "--engine",
                "auto",
                str(CORPUS_DIR / "02_interchange_im.json"),
                "--data-root",
                str(FIXTURES / "data"),
                "--artifacts-root",
                str(artifacts),
                "--set",
                "batch_rows=3",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "refused" in err and "batch_rows" in err


class TestConcatRows:
    """Joining one tap's per-window captures: every window filled, or none.
    A tap fires on every forward of a group or on none, so a mix means a hook
    missed a window — and dropping the empties would return fewer rows than
    the group and misalign every later row index."""

    pytestmark = pytest.mark.unit

    def test_every_window_filled_concatenates_in_window_order(self):
        parts = [torch.arange(4).view(2, 2), torch.arange(4, 10).view(3, 2)]
        joined = _concat_rows(parts)
        assert torch.equal(joined, torch.arange(10).view(5, 2))

    def test_a_single_window_passes_through_untouched(self):
        part = torch.arange(6).view(3, 2)
        assert _concat_rows([part]) is part
        empty = torch.empty(0)
        assert _concat_rows([empty]) is empty

    def test_no_window_filled_stays_the_empty_placeholder(self):
        joined = _concat_rows([torch.empty(0), torch.empty(0), torch.empty(0)])
        assert joined.numel() == 0

    def test_a_partial_fill_fails_by_window(self):
        parts = [torch.ones(2, 3), torch.empty(0), torch.ones(1, 3)]
        with pytest.raises(RuntimeError, match=r"2 of 3 row windows.*windows \[1\]"):
            _concat_rows(parts)


# --------------------------------------------------------------------------- #
# equality with the single-batch run
# --------------------------------------------------------------------------- #


class TestMicrobatchedRunEqualsWholeBatch:
    pytestmark = pytest.mark.smoke

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_block_output_reads(self, llama_bundle, batch_rows: int):
        whole, windowed = _both(_read_doc(), llama_bundle, batch_rows)
        for name in ("all0", "last1"):
            _assert_same(whole.read_value(name), windowed.read_value(name))

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_a_swap_with_a_cross_group_operand(self, llama_bundle, batch_rows: int):
        whole, windowed = _both(
            _patch_doc(), llama_bundle, batch_rows, counterfactual=True
        )
        _assert_same(whole.read_value("after"), windowed.read_value("after"))
        # the swap did land — an interchange that changed nothing would make
        # the equality above vacuous
        assert not torch.allclose(whole.read_value("after"), whole.read_value("clean"))

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_a_gaussian_write_draws_the_same_noise_per_row(
        self, llama_bundle, batch_rows: int
    ):
        """§8's RNG contract: the draw is bit-stable across layouts. The
        noise is drawn over every row of the batch and sliced to the
        window, so a row in the third window sees what it saw in the one."""
        doc = _patch_doc(
            {"gaussian": {"seed": 3, "scale": 1.0, "axis": "tp_duplicated"}}
        )
        whole, windowed = _both(doc, llama_bundle, batch_rows)
        _assert_same(whole.read_value("after"), windowed.read_value("after"))
        assert not torch.allclose(whole.read_value("after"), whole.read_value("clean"))

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_top_k_and_match_metric_tables(self, llama_bundle, batch_rows: int):
        columns = {"ans": [" Monday"] * len(BASE_TEXTS)}
        whole, windowed = _both(
            _patch_doc(), llama_bundle, batch_rows, counterfactual=True, columns=columns
        )
        top_k = AggregationSpec(kind="top_k", fields={"k": 3, "by": "prob"})
        match = AggregationSpec(kind="match", fields={"expected": "ans"})

        def table(executor: PointExecutor, metric: AggregationSpec) -> list[Any]:
            return compute_metric(
                metric,
                executor.dense_value("after"),
                executor.rows_for_metrics(),
                executor.bundle.tokenizer,
            )

        assert table(whole, match) == table(windowed, match)
        for a, b in zip(table(whole, top_k), table(windowed, top_k)):
            assert a["tokens"] == b["tokens"]
            torch.testing.assert_close(
                torch.tensor(b["probs"]), torch.tensor(a["probs"]), rtol=1e-5, atol=1e-6
            )

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_the_ragged_expert_face_and_its_routing_reads(
        self, qwen35moe_bundle, batch_rows: int
    ):
        # read the routing table first and pick the expert the batch used
        # most, so the ragged face has rows in more than one window
        probe = executor_for(_moe_doc(0), qwen35moe_bundle, base_texts=BASE_TEXTS)
        idx = probe.read_value("idx")
        assert isinstance(idx, RaggedValue)  # "all" is every real token, per row
        expert = int(torch.mode(idx.flat.flatten()).values)
        whole, windowed = _both(_moe_doc(expert), qwen35moe_bundle, batch_rows)
        face = whole.read_value("face")
        assert isinstance(face, RaggedValue) and sum(face.widths) > 0
        assert sum(1 for width in face.widths if width) > 1
        for name in ("face", "idx", "scores"):
            _assert_same(whole.read_value(name), windowed.read_value(name))

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_a_state_swap_offsets_its_operand_by_window(
        self, qwen35moe_bundle, batch_rows: int
    ):
        whole, windowed = _both(
            _state_patch_doc(), qwen35moe_bundle, batch_rows, counterfactual=True
        )
        _assert_same(whole.read_value("after"), windowed.read_value("after"))

    @pytest.mark.parametrize("batch_rows", BATCH_ROWS)
    def test_a_decoding_group(self, llama_bundle, batch_rows: int):
        """Each window decodes from its own prefill; the continuation frame,
        every per-step read and the ids-domain metric come out the same."""
        whole, windowed = _both(_generate_doc(), llama_bundle, batch_rows)
        (cont_whole,) = whole._continuations.values()
        (cont_windowed,) = windowed._continuations.values()
        assert torch.equal(cont_whole.token_ids, cont_windowed.token_ids)
        assert cont_whole.widths == cont_windowed.widths
        assert cont_whole.texts == cont_windowed.texts
        for name in ("logits", "acts"):
            assert whole.addressed_steps(name) == windowed.addressed_steps(name)
            for a, b in zip(whole.windowed_value(name), windowed.windowed_value(name)):
                _assert_same(a, b)

        decode = AggregationSpec(kind="decode", fields={})
        top_k = AggregationSpec(kind="top_k", fields={"k": 1, "by": "prob"})

        def table(executor: PointExecutor, metric: AggregationSpec) -> list[list[Any]]:
            ids = executor.generated_ids("logits") if metric.kind == "decode" else None
            return compute_windowed_metric(
                metric,
                executor.windowed_value("logits"),
                executor.rows_for_metrics(),
                executor.bundle.tokenizer,
                generated_ids=ids,
            )

        assert table(whole, decode) == table(windowed, decode)
        for row_a, row_b in zip(table(whole, top_k), table(windowed, top_k)):
            assert [v["tokens"] for v in row_a] == [v["tokens"] for v in row_b]


# --------------------------------------------------------------------------- #
# through run_protocol and the CLI: the same run, and the receipt says how it
# was cut
# --------------------------------------------------------------------------- #


def _tiny_overrides(*extra: str) -> list[str]:
    argv = ["--set", f"model.key={TINY_LLAMA}", "--set", "model.dtype=fp32"]
    for item in extra:
        argv += ["--set", item]
    return argv


def _run_cli(name: str, roots: tuple[Path, Path], out: Path, *args: str) -> int:
    """``causalab run`` with ``--record``: this suite compares two layouts of
    one document through their receipts (``_receipts_differ_only_in_batch_rows``)."""
    data_root, artifacts_root = roots
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(CORPUS_DIR / name),
            "--data-root",
            str(data_root),
            "--artifacts-root",
            str(artifacts_root),
            "--out",
            str(out),
            "--record",
            *args,
        ]
    )


def _tables_close(a_dir: Path, b_dir: Path, name: str) -> None:
    a_rows, b_rows = read_table(a_dir / name), read_table(b_dir / name)
    assert len(a_rows) == len(b_rows)
    for a, b in zip(a_rows, b_rows):
        assert set(a) == set(b)
        for key in a:
            if isinstance(a[key], float):
                assert b[key] == pytest.approx(a[key], rel=1e-5, abs=1e-5), (name, key)
            else:
                assert a[key] == b[key], (name, key)


def _json_close(a: object, b: object, where: str = "train_eval.json") -> None:
    """``a == b`` with floats compared to the tolerance ``_tables_close`` uses:
    a microbatched eval sums the same terms in a different order, which moves
    a metric by an ulp or two and nothing else."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), where
        for key in a:
            _json_close(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _json_close(x, y, f"{where}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert b == pytest.approx(a, rel=1e-5, abs=1e-5), where
    else:
        assert a == b, where


def _receipts_differ_only_in_batch_rows(
    whole_dir: Path, windowed_dir: Path, batch_rows: int
) -> None:
    """The R1 acceptance: two layouts of one document leave receipts that
    are equal after ``execution.batch_rows`` is removed and differ on exactly
    that key — the whole-batch receipt holds ``null``, the windowed one its
    bound — while the document digest and every point digest are the same.

    Without the ``execution`` block the two receipts are byte-identical and
    the ``{"batch_rows": …}`` assertions below fail on a missing key; with the
    bound anywhere else in the receipt (or in the canonical document) the
    stripped receipts differ."""
    whole = json.loads((whole_dir / RUN_RECORD_NAME).read_text())
    windowed = json.loads((windowed_dir / RUN_RECORD_NAME).read_text())
    assert whole["execution"] == {
        "batch_rows": None,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
    assert windowed["execution"] == {
        "batch_rows": batch_rows,
        "device": "cpu",
        "fit_rows": None,
        "model_source": "loaded",
        "parallel": {
            "data": 1,
            "data_mode": "points",
            "pipeline": 1,
            "context": 1,
            "tensor": 1,
            "expert": 1,
            "world": 1,
            "launcher": "solo",
        },
    }
    assert whole["document_digest"] == windowed["document_digest"]
    assert [p["digest"] for p in whole["points"]] == [
        p["digest"] for p in windowed["points"]
    ]
    assert "batch_rows" not in json.dumps(whole["canonical"])
    assert "batch_rows" not in json.dumps(windowed["canonical"])
    whole["execution"].pop("batch_rows")
    windowed["execution"].pop("batch_rows")
    assert whole == windowed  # the block stays in: only its bound differed


def _stamps_identical(whole_dir: Path, windowed_dir: Path) -> None:
    """Every tensor file's ``ArtifactIdentity`` header is byte-for-byte the
    same between the two layouts: the bound is in no stamp."""
    names = sorted(p.name for p in whole_dir.glob("*.safetensors"))
    assert names == sorted(p.name for p in windowed_dir.glob("*.safetensors"))
    for name in names:
        with (
            safe_open(str(whole_dir / name), "pt") as a,
            safe_open(str(windowed_dir / name), "pt") as b,
        ):
            meta_a, meta_b = a.metadata(), b.metadata()
        assert meta_a == meta_b, name
        assert "batch_rows" not in json.dumps(meta_a), name


class TestRunReceiptRecordsTheLayout:
    pytestmark = pytest.mark.smoke

    @pytest.fixture(scope="class")
    def roots(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
        artifacts = tmp_path_factory.mktemp("artifacts")
        shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
        write_rot_fixture(artifacts)
        return FIXTURES / "data", artifacts

    def test_a_corpus_document_through_run_protocol(self, roots, tmp_path):
        """Corpus 02 (four rows, a counterfactual role, two metric tables)
        with ``batch_rows=3`` — a short last window — against the whole-batch
        engine: the receipts differ only in ``execution.batch_rows``, the
        tables within tolerance (every coordinate column exactly)."""
        data_root, artifacts_root = roots
        env = ResolutionEnv(
            datasets=FileDatasets(root=data_root),
            artifacts=FileArtifacts(root=artifacts_root),
        )
        loaded = compile_protocol(
            CORPUS_DIR / "02_interchange_im.json",
            env=env,
            overrides={"model.key": TINY_LLAMA, "sites.target.layers": 1},
        )
        whole_dir, windowed_dir = tmp_path / "whole", tmp_path / "windowed"
        whole = run_protocol(loaded, env, PytorchHooksEngine(), whole_dir, record=True)
        windowed = run_protocol(
            loaded, env, PytorchHooksEngine(batch_rows=3), windowed_dir, record=True
        )
        _receipts_differ_only_in_batch_rows(whole_dir, windowed_dir, 3)
        _stamps_identical(whole_dir, windowed_dir)
        assert set(whole.files) == set(windowed.files)
        assert whole.forwards == windowed.forwards  # groups, not windows
        for name in ("iia.json", "logit_diff.json"):
            _tables_close(whole_dir, windowed_dir, name)

    def test_a_ragged_harvest_through_the_cli_flag(self, roots, tmp_path):
        """Corpus 01 saves a ragged read (multi-token entities) beside a
        dense one; ``--batch-rows 3`` leaves the widths and every stamp the
        same, the activations within tolerance, and the receipt differing only
        in ``execution.batch_rows``."""
        whole_dir, windowed_dir = tmp_path / "whole", tmp_path / "windowed"
        layers = ("sites.L8.layers=0", "sites.L24.layers=1")
        assert (
            _run_cli("01_harvest_im.json", roots, whole_dir, *_tiny_overrides(*layers))
            == 0
        )
        assert (
            _run_cli(
                "01_harvest_im.json",
                roots,
                windowed_dir,
                *_tiny_overrides(*layers),
                "--batch-rows",
                "3",
            )
            == 0
        )
        _receipts_differ_only_in_batch_rows(whole_dir, windowed_dir, 3)
        _stamps_identical(whole_dir, windowed_dir)
        for path in sorted(whole_dir.glob("*.safetensors")):
            a, b = load_file(str(path)), load_file(str(windowed_dir / path.name))
            assert set(a) == set(b)
            for key in a:
                _assert_same(a[key], b[key])

    def test_training_minibatches_keep_their_rows_and_eval_is_microbatched(
        self, roots, tmp_path
    ):
        """Corpus 05 at tiny scale with ``--batch-rows 1`` against the default:
        the fit is the same fit (a training minibatch keeps its
        ``train.batch.pairs`` rows, so the optimizer sees identical steps)
        and the eval pass — which *is* microbatched — scores the same."""
        whole_dir, windowed_dir = tmp_path / "whole", tmp_path / "windowed"
        overrides = _tiny_overrides(
            "sites.target.layers=1",
            'train.steps={"epochs": 1}',
            'train.batch={"pairs": 2}',
        )
        assert _run_cli("05_dbm_im.json", roots, whole_dir, *overrides) == 0
        assert (
            _run_cli(
                "05_dbm_im.json", roots, windowed_dir, *overrides, "--batch-rows", "1"
            )
            == 0
        )
        _receipts_differ_only_in_batch_rows(whole_dir, windowed_dir, 1)
        _stamps_identical(whole_dir, windowed_dir)
        gate_a = load_file(str(whole_dir / "gate.safetensors"))
        gate_b = load_file(str(windowed_dir / "gate.safetensors"))
        assert set(gate_a) == set(gate_b)
        for key in gate_a:
            _assert_same(gate_a[key], gate_b[key])
        for name in ("iia.json", "ce.json"):
            _tables_close(whole_dir, windowed_dir, name)
        eval_a = json.loads((whole_dir / "train_eval.json").read_text())
        eval_b = json.loads((windowed_dir / "train_eval.json").read_text())
        # the eval pass is microbatched, so its metric is a sum in a different
        # order: equal to float tolerance, like the tables above, not to the bit
        _json_close(eval_a, eval_b)

    def test_a_workflow_step_records_the_same_block(self, roots, tmp_path):
        """A workflow run takes the same flag, and its per-step record is
        the receipt for that step: ``_step.json`` carries the same
        ``execution`` block ``protocol.json`` does, and two layouts of the
        one-step workflow differ there and nowhere else."""
        data_root, artifacts_root = roots
        workflow = tmp_path / "wf.json"
        workflow.write_text(
            json.dumps(
                {
                    "version": "1",
                    "description": "corpus 02 as a one-step workflow",
                    "output_dir": "layout",
                    "steps": {
                        "run": {
                            "type": "intervention_protocol",
                            "document": str(CORPUS_DIR / "02_interchange_im.json"),
                            "set": {
                                "model.key": TINY_LLAMA,
                                "model.dtype": "fp32",
                                "sites.target.layers": 1,
                            },
                        }
                    },
                }
            )
        )
        records: dict[str, dict[str, Any]] = {}
        for label, extra in (("whole", ()), ("windowed", ("--batch-rows", "3"))):
            out = tmp_path / label
            code = main(
                [
                    "run",
                    "--engine",
                    "auto",
                    str(workflow),
                    "--data-root",
                    str(data_root),
                    "--artifacts-root",
                    str(artifacts_root),
                    "--out",
                    str(out),
                    *extra,
                ]
            )
            assert code == 0
            records[label] = json.loads(
                (out / "layout" / "run" / "_step.json").read_text()
            )
        whole, windowed = records["whole"], records["windowed"]
        assert whole["execution"] == {
            "batch_rows": None,
            "device": "cpu",
            "fit_rows": None,
            "model_source": "loaded",
            "parallel": {
                "data": 1,
                "data_mode": "points",
                "pipeline": 1,
                "context": 1,
                "tensor": 1,
                "expert": 1,
                "world": 1,
                "launcher": "solo",
            },
        }
        assert windowed["execution"] == {
            "batch_rows": 3,
            "device": "cpu",
            "fit_rows": None,
            "model_source": "loaded",
            "parallel": {
                "data": 1,
                "data_mode": "points",
                "pipeline": 1,
                "context": 1,
                "tensor": 1,
                "expert": 1,
                "world": 1,
                "launcher": "solo",
            },
        }
        assert whole["document_digest"] == windowed["document_digest"]
        assert whole["point_digests"] == windowed["point_digests"]
        whole["execution"].pop("batch_rows")
        windowed["execution"].pop("batch_rows")
        assert whole == windowed
        _tables_close(
            tmp_path / "whole/layout/run", tmp_path / "windowed/layout/run", "iia.json"
        )
