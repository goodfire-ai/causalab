"""The memory pre-flight under a dtype conversion
(``protocol/parallel_memory.py`` "the load's peak under a dtype conversion",
``docs/model_parallelism.md`` §5.3, §11), torch-free, using
``google/gemma-2-9b`` stored in **fp32**
(``tests/golden/parallel_headers_gemma2_9b.json``) and held in the document's
bf16.

``unit``: the census accounts for the whole fp32 checkpoint (36.97 GB)
and bf16 residency: 18.48 GB per rank at ``dp=2``, or 10.16 GB at ``tp=2``
with the vocabulary replicated. Conversion stages the largest tensor, the
3.42 GiB embedding, on the host; its device footprint matches a checkpoint
stored in bf16. A same-dtype load needs no conversion, and pipeline ranks
stage only their own largest converting tensor. The tests also validate
placement fields, fused-parameter dtype and size accounting, and the
conversion line from a torch-free ``dry-run`` over header-only shards.

``property``: over hypothesis tables decorated with stored dtypes, the
resident estimate and the card footprint are exactly the undecorated
table's — the rule adds no device term — and a conversion exists exactly
when some float tensor's stored word is not the document's, its staging
never above the largest converting tensor.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.checkpoint_census import ITEMSIZES, checkpoint_targets, tree_of
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry, parse_geometry
from causalab.protocol.parallel_memory import (
    DISK_WORDS,
    DTYPE_ITEMSIZES,
    RULE,
    Conversion,
    Placement,
    conversion,
    estimate_resident,
    format_bytes,
    placement_table,
    whole_bytes,
)
from causalab.protocol.registry import LLAMA_TREE, get_model_info

from tests._helpers.header_census import GEMMA2_9B_CENSUS, fake_hub_cache, load_census
from tests.protocol.test_cli_parallel import INTERCHANGE
from tests.protocol.test_dry_run import _argv, _offline
from tests.protocol.test_parallel_memory import _geometries, _product, _tables

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

GEMMA = "google/gemma-2-9b"
#: 9,241,705,984 parameters: every tensor of the checkpoint, F32 on disk.
ELEMENTS = 9_241_705_984
ON_DISK = ELEMENTS * 4  # 36,966,823,936
HELD_BF16 = ELEMENTS * 2  # 18,483,411,968
#: The embedding, 256000 × 3584 in fp32 — the largest tensor, whole on
#: every rank (the vocabulary row is declined on gemma2, §6.1).
EMBEDDING_ON_DISK = 256_000 * 3584 * 4  # 3,670,016,000


def _gemma_table(*, with_dtypes: bool = True) -> dict[str, Placement]:
    info = get_model_info(GEMMA)
    census = load_census(GEMMA2_9B_CENSUS)
    elements = {key: _product(shape) for key, (_, shape) in census.items()}
    tree = tree_of(elements, info.num_layers)
    assert tree is not None and tree == LLAMA_TREE.tree  # the llama-shaped tree
    targets = checkpoint_targets(elements, tree, info.num_layers)
    assert targets is not None and info.parallel_plan is not None
    return placement_table(
        elements,
        targets,
        info.parallel_plan,
        tree,
        info,
        dtypes={key: dtype for key, (dtype, _) in census.items()}
        if with_dtypes
        else None,
    )


@pytest.fixture(scope="module")
def gemma_table() -> dict[str, Placement]:
    return _gemma_table()


# --------------------------------------------------------------------------- #
# the checkpoint that found it
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestGemma:
    def test_the_census_is_the_whole_model_stored_in_fp32(self, gemma_table) -> None:
        census = load_census(GEMMA2_9B_CENSUS)
        assert len(census) == 464 and len(gemma_table) == 464
        assert {dtype for dtype, _ in census.values()} == {"F32"}
        assert whole_bytes(gemma_table, 4) == ON_DISK
        assert whole_bytes(gemma_table, 2) == HELD_BF16
        assert format_bytes(ON_DISK) == "34.43 GiB"
        assert format_bytes(HELD_BF16) == "17.21 GiB"
        assert all(p.disk_dtype == "F32" for p in gemma_table.values())
        # one tensor per parameter: the largest key is the parameter
        assert all(p.key_elements == p.elements for p in gemma_table.values())
        assert "lm_head.weight" not in gemma_table  # tied to the embedding

    def test_the_resident_bytes_are_the_captures_held_bytes(self, gemma_table) -> None:
        """Count bf16 parameter storage: the whole model at ``dp=2``
        (18.48 GB), or replicated vocabulary and norms plus half of each
        projection at ``tp=2`` (10.16 GB). These values exclude the rotary
        table included in device allocation measurements."""
        assert estimate_resident(gemma_table, parse_geometry("dp=2"), 2) == (
            HELD_BF16,
            HELD_BF16,
        )
        (per_rank,) = set(estimate_resident(gemma_table, parse_geometry("tp=2"), 2))
        assert per_rank == 10_159_815_680
        assert round(per_rank / 1e9, 2) == 10.16
        assert round(HELD_BF16 / 1e9, 2) == 18.48

    def test_a_bf16_load_converts_from_fp32_and_stages_the_embedding(
        self, gemma_table
    ) -> None:
        for spelled in ("dp=2", "tp=2", ""):
            geometry = parse_geometry(spelled) if spelled else ONE
            found = conversion(gemma_table, geometry, "bf16")
            assert found is not None, spelled
            assert found.from_words == ("fp32",) and found.to == "bf16"
            assert found.staging == (EMBEDDING_ON_DISK,) * geometry.world
        described = conversion(gemma_table, parse_geometry("dp=2"), "bf16")
        assert described is not None
        assert described.describe() == (
            "read as fp32 on disk, cast to bf16 on the host: rank0 +3.42 GiB, "
            "rank1 +3.42 GiB of host memory at the load's peak (the largest "
            "converting tensor, whole); nothing on the card"
        )

    def test_the_card_footprint_carries_no_conversion_term(self, gemma_table) -> None:
        """The loader stages the cast on the host (§5.3), so the card is a
        bf16 checkpoint's of these shapes: the footprint of the table with
        its dtypes equals the footprint of the same table without them, on
        every geometry the golden runs."""
        plain = _gemma_table(with_dtypes=False)
        assert all(p.disk_dtype is None for p in plain.values())
        for spelled in ("dp=2", "tp=2", ""):
            geometry = parse_geometry(spelled) if spelled else ONE
            assert RULE.footprint(gemma_table, geometry, 2) == RULE.footprint(
                plain, geometry, 2
            )
            assert estimate_resident(gemma_table, geometry, 2) == estimate_resident(
                plain, geometry, 2
            )
        assert format_bytes(
            RULE.footprint(gemma_table, parse_geometry("dp=2"), 2)[0]
        ) == ("19.80 GiB")

    def test_a_same_dtype_load_converts_nothing(self, gemma_table) -> None:
        assert conversion(gemma_table, parse_geometry("dp=2"), "fp32") is None
        # the plain table knows no dtypes, so it names no conversion either
        assert conversion(_gemma_table(with_dtypes=False), ONE, "bf16") is None

    def test_a_dtype_outside_the_document_is_refused(self, gemma_table) -> None:
        with pytest.raises(ProtocolError) as err:
            conversion(gemma_table, ONE, "int8")
        assert err.value.code == "P4"


# --------------------------------------------------------------------------- #
# the placement's new fields, a pipeline, a fused parameter
# --------------------------------------------------------------------------- #


def _pipeline_table() -> dict[str, Placement]:
    """An embedding first, two layers of one projection each — the first
    stored in fp32, the second in bf16 — the norm and a small head last."""
    return {
        "model.embed_tokens.weight": Placement(256, home="first", disk_dtype="F32"),
        "model.layers.0.q.weight": Placement(
            64, axis="tensor", home="layer", layer=0, disk_dtype="F32"
        ),
        "model.layers.1.q.weight": Placement(
            64, axis="tensor", home="layer", layer=1, disk_dtype="BF16"
        ),
        "model.norm.weight": Placement(8, home="last", disk_dtype="F32"),
        "lm_head.weight": Placement(128, home="last", disk_dtype="F32"),
    }


@pytest.mark.unit
class TestPlacement:
    def test_the_new_fields_validate(self) -> None:
        with pytest.raises(ValueError, match="disk_dtype"):
            Placement(4, disk_dtype="F99")
        with pytest.raises(ValueError, match="key_elements"):
            Placement(4, key_elements=5)
        with pytest.raises(ValueError, match="key_elements"):
            Placement(4, key_elements=True)  # type: ignore[arg-type]
        Placement(4, disk_dtype="F32", key_elements=4)
        Placement(4, key_elements=0)

    def test_converting_is_a_float_word_other_than_the_documents(self) -> None:
        fp32 = Placement(4, disk_dtype="F32")
        assert fp32.converts_to("bf16") and fp32.converts_to("fp16")
        assert not fp32.converts_to("fp32")
        # an integer buffer keeps its own dtype (transformers' rule too)
        assert not Placement(4, disk_dtype="I64").converts_to("bf16")
        assert not Placement(4).converts_to("bf16")
        assert set(DISK_WORDS.values()) >= set(DTYPE_ITEMSIZES)

    def test_the_largest_key_bytes_are_the_staging_unit(self) -> None:
        assert Placement(4, disk_dtype="F32").largest_key_bytes == 16
        assert Placement(4, disk_dtype="F32", key_elements=2).largest_key_bytes == 8
        assert Placement(4, disk_dtype="BF16", key_elements=2).largest_key_bytes == 4
        assert Placement(4).largest_key_bytes == 0

    def test_a_pipeline_stages_each_ranks_own_largest_converting_tensor(self) -> None:
        table = _pipeline_table()
        to_bf16 = conversion(table, ParallelGeometry(pipeline=2), "bf16")
        assert to_bf16 == Conversion(("fp32",), "bf16", (256 * 4, 128 * 4))
        # held in fp32 the second layer's bf16 projection is the one cast,
        # on the last stage alone; the first stage stages nothing
        to_fp32 = conversion(table, ParallelGeometry(pipeline=2), "fp32")
        assert to_fp32 == Conversion(("bf16",), "fp32", (0, 64 * 2))
        # world 1 stages the largest of all
        assert conversion(table, ONE, "bf16") == Conversion(("fp32",), "bf16", (1024,))
        # the tensor axis does not shrink the bound: the tensor is counted whole
        assert conversion(table, ParallelGeometry(tensor=2), "fp16") == Conversion(
            ("bf16", "fp32"), "fp16", (1024, 1024)
        )

    def test_a_fused_parameter_takes_the_widest_dtype_and_the_largest_tensor(
        self,
    ) -> None:
        info = get_model_info(GEMMA)
        assert info.parallel_plan is not None
        targets = {
            "model.layers.0.mlp.experts.0.w": "model.layers.0.mlp.experts.w",
            "model.layers.0.mlp.experts.1.w": "model.layers.0.mlp.experts.w",
            "model.norm.weight": "model.norm.weight",
        }
        elements = {
            "model.layers.0.mlp.experts.0.w": 10,
            "model.layers.0.mlp.experts.1.w": 30,
            "model.norm.weight": 8,
        }
        dtypes = {
            "model.layers.0.mlp.experts.0.w": "BF16",
            "model.layers.0.mlp.experts.1.w": "F32",
            "model.norm.weight": "BF16",
        }
        table = placement_table(
            elements, targets, info.parallel_plan, LLAMA_TREE.tree, info, dtypes=dtypes
        )
        fused = table["model.layers.0.mlp.experts.w"]
        assert (fused.elements, fused.key_elements, fused.disk_dtype) == (40, 30, "F32")
        assert table["model.norm.weight"].disk_dtype == "BF16"
        with pytest.raises(ProtocolError) as err:
            placement_table(
                elements,
                targets,
                info.parallel_plan,
                LLAMA_TREE.tree,
                info,
                dtypes={**dtypes, "model.norm.weight": "Q4"},
            )
        assert err.value.code == "P2" and "Q4" in str(err.value)


# --------------------------------------------------------------------------- #
# properties: no device term, a conversion exactly when a word differs
# --------------------------------------------------------------------------- #

_DISK = st.sampled_from((None, "F32", "BF16", "F16", "I64"))
_WORDS = st.sampled_from(tuple(DTYPE_ITEMSIZES))


@st.composite
def _decorated(draw: st.DrawFn) -> tuple[dict[str, Placement], dict[str, Placement]]:
    """A table and the same table with stored dtypes and largest tensors
    drawn onto every placement."""
    plain = draw(_tables())
    decorated = {
        name: dataclasses.replace(
            placement,
            disk_dtype=draw(_DISK),
            key_elements=draw(st.integers(min_value=0, max_value=placement.elements)),
        )
        for name, placement in plain.items()
    }
    return plain, decorated


@pytest.mark.property
class TestProperties:
    @_SETTINGS
    @given(tables=_decorated(), word=_WORDS, data=st.data())
    def test_the_stored_dtypes_change_no_device_estimate(
        self, tables, word, data
    ) -> None:
        plain, decorated = tables
        geometry = data.draw(_geometries(plain))
        itemsize = DTYPE_ITEMSIZES[word]
        assert estimate_resident(decorated, geometry, itemsize) == estimate_resident(
            plain, geometry, itemsize
        )
        assert RULE.footprint(decorated, geometry, itemsize) == RULE.footprint(
            plain, geometry, itemsize
        )

    @_SETTINGS
    @given(tables=_decorated(), word=_WORDS, data=st.data())
    def test_a_conversion_exists_exactly_when_a_float_word_differs(
        self, tables, word, data
    ) -> None:
        plain, decorated = tables
        geometry = data.draw(_geometries(plain))
        converting = [
            p
            for p in decorated.values()
            if p.disk_dtype in DISK_WORDS and DISK_WORDS[p.disk_dtype] != word
        ]
        found = conversion(decorated, geometry, word)
        assert conversion(plain, geometry, word) is None
        if not converting:
            assert found is None
            return
        assert found is not None and found.to == word
        assert set(found.from_words) == {DISK_WORDS[p.disk_dtype] for p in converting}
        assert len(found.staging) == geometry.world
        ceiling = max(
            (p.key_elements if p.key_elements is not None else p.elements)
            * ITEMSIZES[p.disk_dtype]
            for p in converting
        )
        assert all(0 <= staged <= ceiling for staged in found.staging)
        # at world 1 the one rank stages the largest converting tensor
        if geometry.world == 1:
            assert found.staging == (ceiling,)


# --------------------------------------------------------------------------- #
# dry-run prints the line, torch-free
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gemma_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("hub")
    fake_hub_cache(root, GEMMA, load_census(GEMMA2_9B_CENSUS), files=8)
    return root


@pytest.fixture
def artifacts_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def _dry_run(artifacts_root: Path, cache: Path, spelled: str, dtype: str) -> dict:
    return _offline(
        _argv(
            INTERCHANGE,
            artifacts_root,
            "--set",
            f"model.key={GEMMA}",
            "--set",
            f"model.dtype={dtype}",
            "--parallel",
            spelled,
        ),
        hub_cache=cache,
    )


@pytest.mark.unit
class TestDryRun:
    def test_the_conversion_line_on_a_bf16_document_over_the_fp32_checkpoint(
        self, artifacts_root: Path, gemma_cache: Path
    ) -> None:
        result = _dry_run(artifacts_root, gemma_cache, "dp=2", "bf16")
        assert result["code"] == 0, result["err"]
        out = result["out"]
        assert (
            "  resident weights (bf16; the model is 17.21 GiB): rank0 17.21 GiB, rank1 17.21 GiB"
            in out
        )
        assert "  estimated card footprint: rank0 19.80 GiB, rank1 19.80 GiB" in out
        assert (
            "  read as fp32 on disk, cast to bf16 on the host: rank0 +3.42 GiB, "
            "rank1 +3.42 GiB of host memory at the load's peak (the largest "
            "converting tensor, whole); nothing on the card" in out
        )
        assert "memory: undecided" not in out
        assert not result["torch"], "the dry run imported torch"

    def test_no_line_when_the_document_asks_for_the_stored_dtype(
        self, artifacts_root: Path, gemma_cache: Path
    ) -> None:
        result = _dry_run(artifacts_root, gemma_cache, "dp=2", "fp32")
        assert result["code"] == 0, result["err"]
        out = result["out"]
        assert "  resident weights (fp32; the model is 34.43 GiB)" in out
        assert "read as " not in out and "cast to" not in out
