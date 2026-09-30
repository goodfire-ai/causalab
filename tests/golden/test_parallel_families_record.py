"""CPU guard for the second-family rows of the parallel golden
(``docs/model_parallelism.md`` §10.6, §11; runs in default CI, loads no
weights of the real models).

The entries (``tests/golden/_parallel/families.py``): each family document
on its own bf16 realization, the dense classes alone, gemma2 with no
pipeline geometry and the refusal it holds instead, both outside the A3B
tier's set and inside the capturable one; the ``google/gemma-2-9b``
registry row — a gemma2 with the vocabulary row declined — cross-checked
against ``model_info_from_hf_config`` on the cached config when present;
the documents authoring and validating on their real models (the loader
alone, torch-free) and expanding to two points; ``dry-run … --parallel
tp=2`` on the real gemma document naming the ``unapplied`` row and on the
llama document not; the skip rules the golden applies; and — the
harness's own smoke, over ``gloo`` — the llama document retargeted to the
tiny Llama captured end to end: ``dp=2`` and ``pp=2`` exact, ``tp=2``
banded, exactly the dense classes, the record made and replaying.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from causalab.cli import register_model_key
from causalab.neural.shared.sweep import enumerate_steps
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.registry import (
    GEMMA2_PLAN,
    get_model_info,
    model_info_from_hf_config,
)
from tests.golden import _parallel as par
from tests.golden._parallel import families, inference
from tests.golden.update_parallel_goldens import BadExplanation, parse_explanations
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._env import FIXTURES, build_env


def _env(tmp: Path):
    root = tmp / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    return build_env(root)


@pytest.mark.unit
class TestEntries:
    def test_the_family_documents_and_their_geometries(self) -> None:
        assert set(families.FAMILIES) == {"inference_gemma2_9b", "inference_llama31_8b"}
        gemma, llama = families.GEMMA2, families.LLAMA
        assert gemma.exact == ("dp=2",) and gemma.banded == ("tp=2",)
        assert llama.exact == ("dp=2", "pp=2") and llama.banded == ("tp=2",)
        for document in (gemma, llama):
            assert document.classes == inference.DENSE_CLASSES
            assert not document.recorded and document.argv == ()
            assert document.realization is not None
            assert document.realization.dtype == "bf16"
            assert document.realization.device == "cuda"
            assert par.realization_of(document) is document.realization
            assert document.name not in par.DOCUMENTS
            assert par.CAPTURABLE[document.name] is document
        assert set(par.CAPTURABLE) == (
            set(par.DOCUMENTS) | set(families.FAMILIES) | set(par.LARGE_DOCUMENTS)
        )
        assert families.REFUSED == {gemma.name: ("pp=2", "tie_word_embeddings")}
        assert "pp=2" not in gemma.geometries
        assert families.DENSE_FIXTURE == TINY_LLAMA

    def test_the_dense_classes_are_the_boundary_classes_without_the_experts(
        self,
    ) -> None:
        assert inference.DENSE_CLASSES == frozenset(boundary.CLASSES.values()) - {
            "experts"
        }
        assert par.DOCUMENTS["inference"].classes == inference.DENSE_CLASSES | {
            "experts",
            par.ROUTING,
        }

    def test_the_gemma_9b_entry_is_a_gemma2_with_the_vocabulary_row_declined(
        self,
    ) -> None:
        info = get_model_info(families.GEMMA2_9B_MODEL)
        assert info.family == "gemma2" and info.parallel_plan is GEMMA2_PLAN
        assert info.num_layers == 42 and info.num_heads == 16 and info.num_kv_heads == 8
        assert info.head_dim == 256 and info.hidden_size == 3584
        assert dict(GEMMA2_PLAN.unapplied) == {"embed_tokens": "embedding_rowwise"}
        assert info.num_layers > families.GEMMA2_9B.layer + inference.SWEEP_STRIDE
        llama = get_model_info(families.LLAMA31_8B_MODEL)
        assert llama.family == "llama"
        assert llama.num_layers > families.LLAMA31_8B.layer + inference.SWEEP_STRIDE
        assert not llama.parallel_plan.unapplied  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        "key", [families.GEMMA2_9B_MODEL, families.LLAMA31_8B_MODEL]
    )
    def test_the_entry_matches_its_cached_hf_config(self, key: str) -> None:
        """Every static value equals what ``model_info_from_hf_config`` reads
        from the checkpoint's own config.json (the node's cache holds it;
        skipped, never fetched, where it is not cached)."""
        from huggingface_hub.constants import HF_HUB_CACHE
        from transformers import AutoConfig

        folder = "models--" + key.replace("/", "--")
        if not os.path.isdir(os.path.join(HF_HUB_CACHE, folder)):
            pytest.skip(f"{key} is not in the local HF cache ({HF_HUB_CACHE})")
        config = AutoConfig.from_pretrained(key, local_files_only=True)
        read = model_info_from_hf_config(key, config)
        static = get_model_info(key)
        for field in (
            "hidden_size",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "family",
        ):
            assert getattr(read, field) == getattr(static, field), field
        assert read.parallel_plan is not None and static.parallel_plan is not None
        assert dict(read.parallel_plan.unapplied) == dict(
            static.parallel_plan.unapplied
        )
        if key == families.GEMMA2_9B_MODEL:
            assert getattr(config, "tie_word_embeddings", False)

    @pytest.mark.parametrize("name", sorted(families.FAMILIES))
    def test_the_document_authors_and_expands_on_its_real_model(
        self, name: str, tmp_path: Path
    ) -> None:
        """Torch-free: the loader validates the document against the
        registry entry and expands the sweep to two points."""
        document = families.FAMILIES[name]
        realization = par.realization_of(document)
        authored = document.author(tmp_path, realization)
        raw = json.loads(authored.read_text())
        assert raw["model"] == {
            "key": realization.model,
            "revision": "main",
            "dtype": "bf16",
        }
        assert raw["method"]["sites"]["target"]["layers"] == {
            "sweep": list(inference.sweep(realization))
        }
        assert "idx" not in raw["method"]["sites"]  # the dense fixture's document
        compiled = compile_protocol(authored, env=_env(tmp_path))
        assert len(enumerate_steps(compiled).points) == 2
        assert document.describe(realization)["sweep"] == list(
            inference.sweep(realization)
        )

    def test_dry_run_names_the_declined_row_on_gemma2_and_not_on_llama(
        self, tmp_path: Path
    ) -> None:
        """``causalab dry-run … --parallel tp=2`` is torch-free and needs no
        weights: on the gemma document it prints the ``unapplied`` row
        (``protocol/cli.py``), on the llama document nothing of the kind."""
        for name, document in families.FAMILIES.items():
            base = tmp_path / name
            base.mkdir()
            authored = document.author(base, par.realization_of(document))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "causalab.cli",
                    "dry-run",
                    str(authored),
                    "--engine",
                    "pytorch_hooks",
                    "--data-root",
                    str(par.runs.DATA),
                    "--artifacts-root",
                    str(base),
                    "--parallel",
                    "tp=2",
                ],
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, (name, completed.stderr[-2000:])
            assert (
                "tp=2,ep=1 (world 2): accepted by the registry entry"
                in completed.stdout
            )
            assert (families.UNAPPLIED in completed.stdout) == (
                name == families.GEMMA2.name
            ), (name, completed.stdout)

    def test_the_capture_takes_a_family_by_name_and_refuses_a_stranger(self) -> None:
        assert parse_explanations(["inference_gemma2_9b dp=2: why"]) == {
            "inference_gemma2_9b": {"dp=2": "why"}
        }
        with pytest.raises(BadExplanation, match="names no document"):
            parse_explanations(["inference_gemma3 dp=2: why"])

    def test_an_uncaptured_family_replays_nothing_banded_and_a_stale_record_skips(
        self,
    ) -> None:
        """The golden's skip rules: a record without the family's capture
        skips its band replay (``par.captured`` false) naming the capture;
        a record of another format is refused by name."""
        record = par.load_record()
        for name in families.FAMILIES:
            if not par.captured(record, name):
                with pytest.raises(
                    par.StaleRecord, match=f"no capture of the {name!r} document"
                ):
                    par.compare_records(record, record, name)
        with pytest.raises(par.StaleRecord, match="format 1"):
            par.check_format({**record, "format": 1})

    def test_replay_problems_is_the_parity_comparison_spelled_once(self) -> None:
        """A block made from a measurement replays against itself, and a
        fresh value outside the band is named."""
        document = families.LLAMA
        realization = par.realization_of(document)
        measured = par.Measured()
        for kind in sorted(document.classes):
            measured.add(
                kind, f"{kind}.safetensors", par.Measurement(0.5, 20.0, "bf16")
            )
        block = par.document_record(
            document,
            realization,
            {"tp=2": measured},
            {"dp=2": 0.0, "pp=2": 0.0},
            with_context=False,
        )
        record = par.make_record(None, par.A3B, {document.name: block})
        assert record["documents"][document.name]["realization"]["model"] == (
            families.LLAMA31_8B_MODEL
        )
        assert (
            par.replay_problems(
                record, document, realization, "tp=2", measured, par.A3B
            )
            == []
        )
        worse = par.Measured()
        for kind in sorted(document.classes):
            worse.add(kind, f"{kind}.safetensors", par.Measurement(5.0, 20.0, "bf16"))
        problems = par.replay_problems(
            record, document, realization, "tp=2", worse, par.A3B
        )
        assert problems and all("> band" in p for p in problems)


@pytest.mark.smoke
class TestHarnessOnTheTinyLlama:
    def test_the_llama_document_captures_end_to_end_over_gloo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The family harness on the tiny Llama in fp32 on the CPU, exactly
        as the GPU capture runs the 8B: no refusal, ``dp=2`` and ``pp=2``
        exact, ``tp=2`` banded over the dense classes alone, the record
        made and replaying against itself."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        register_model_key({"model": {"key": TINY_LLAMA, "revision": "main"}})
        tiny = par.Realization(TINY_LLAMA, "fp32", "cpu", boundary.LAYER[TINY_LLAMA])
        assert inference.sweep(tiny) == (0, 1)  # the two-layer tower, clamped
        block, refusals = par.capture(
            tmp_path, families.LLAMA, tiny, with_context=False
        )
        assert refusals == []
        assert block["exact"] == {"dp=2": 0.0, "pp=2": 0.0}
        assert set(block["geometries"]) == {"tp=2"}
        assert set(block["geometries"]["tp=2"]) == set(inference.DENSE_CLASSES)
        assert block["realization"] == {
            "model": TINY_LLAMA,
            "dtype": "fp32",
            "device": "cpu",
        }
        assert set(block["load"]) == {"dp=2", "pp=2", "tp=2"}
        record = par.make_record(None, par.A3B, {families.LLAMA.name: block})
        assert par.compare_records(record, record, families.LLAMA.name) == []
