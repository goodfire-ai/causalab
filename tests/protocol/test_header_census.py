"""The header census helper (``tests/_helpers/header_census.py``): the
compressor is the inverse of the expander — on the two committed censuses
exactly, and on hypothesis-drawn key sets — an ambiguous pattern is refused
by name, and a census laid out as a header-only Hub cache reads back through
the protocol layer's own cache lookup and header reader (torch-free), which
is what lets ``dry-run`` estimate the 70B on the CPU with no weight on disk.
The ``__main__`` that captures or checks a census on a node is run here
against that fake cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.checkpoint_census import cached_checkpoint_files, read_headers

from tests._helpers import header_census as census
from tests._helpers.header_census import (
    A3B_CENSUS,
    GEMMA2_9B_CENSUS,
    LLAMA70B_CENSUS,
    CensusAmbiguity,
    census_record,
    compress,
    expand,
    fake_hub_cache,
    load_census,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

LARGE = "meta-llama/Llama-3.1-70B"


@pytest.mark.unit
@pytest.mark.parametrize("path", [A3B_CENSUS, LLAMA70B_CENSUS])
def test_compress_inverts_expand_on_the_committed_censuses(path: Path) -> None:
    record = json.loads(path.read_text())
    keys = expand(record["patterns"])
    assert len(keys) == record["tensors"]
    assert compress(keys) == record["patterns"]
    assert load_census(path) == keys


@pytest.mark.unit
def test_the_70b_census_is_the_llama_key_set_over_thirty_shards() -> None:
    record = json.loads(LLAMA70B_CENSUS.read_text())
    assert (record["model"], record["files"], record["tensors"]) == (LARGE, 30, 723)
    assert record["snapshot"] == "349b2ddb53ce8f2849a6c168a81980ab25258dac"
    patterns = record["patterns"]
    assert set(patterns) == {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        *(
            f"model.layers.*.{leaf}"
            for leaf in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        ),
    }
    for pattern, entry in patterns.items():
        assert entry["dtype"] == "BF16", pattern
        if "*" in pattern:
            assert entry["ranges"] == [[0, 79]], pattern
    # the head is untied: its own tensor, the embedding's shape
    assert patterns["lm_head.weight"]["shape"] == [128256, 8192]
    assert patterns["model.embed_tokens.weight"]["shape"] == [128256, 8192]
    assert patterns["model.layers.*.self_attn.k_proj.weight"]["shape"] == [1024, 8192]
    assert patterns["model.layers.*.mlp.gate_proj.weight"]["shape"] == [28672, 8192]
    # the derived census says where it came from and how it is verified
    assert "--check" in record["provenance"]


@pytest.mark.unit
def test_the_gemma2_9b_census_is_the_gemma2_key_set_stored_in_fp32_over_eight_shards() -> (
    None
):
    """The second-family checkpoint whose bf16 load converts
    (``docs/model_parallelism.md`` §5.3): every tensor ``F32``, the
    ``Gemma2ForCausalLM`` key set (four norms a layer, no ``lm_head.weight``
    — the head is tied), 464 tensors over 8 shards at the Hub's ``main``
    snapshot; the compressor inverts the expander on it, the tensor bytes
    are the listed shard sizes less the eight headers, and the provenance
    names the ``--check`` that holds it to the node."""
    record = json.loads(GEMMA2_9B_CENSUS.read_text())
    assert (record["model"], record["files"], record["tensors"]) == (
        "google/gemma-2-9b",
        8,
        464,
    )
    assert record["snapshot"] == "33c193028431c2fde6c6e51f29e6f17b60cbfac6"
    patterns = record["patterns"]
    assert set(patterns) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        *(
            f"model.layers.*.{leaf}"
            for leaf in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        ),
    }
    for pattern, entry in patterns.items():
        assert entry["dtype"] == "F32", pattern
        if "*" in pattern:
            assert entry["ranges"] == [[0, 41]], pattern
    assert patterns["model.embed_tokens.weight"]["shape"] == [256000, 3584]
    assert patterns["model.layers.*.self_attn.q_proj.weight"]["shape"] == [4096, 3584]
    assert patterns["model.layers.*.self_attn.k_proj.weight"]["shape"] == [2048, 3584]
    assert patterns["model.layers.*.mlp.down_proj.weight"]["shape"] == [3584, 14336]
    keys = expand(patterns)
    assert len(keys) == 464 and compress(keys) == patterns
    assert load_census(GEMMA2_9B_CENSUS) == keys
    on_disk = 0
    for _, shape in keys.values():
        count = 4
        for n in shape:
            count *= n
        on_disk += count
    assert on_disk == 36_966_823_936
    assert "36966878040" in record["provenance"]  # the shards' listed sizes
    assert "--check" in record["provenance"] and "float32" in record["provenance"]


# --------------------------------------------------------------------------- #
# the compressor as a property
# --------------------------------------------------------------------------- #

_STEMS = st.sampled_from(
    ("model.layers", "model.experts", "mtp.layers", "vision.blocks")
)
_LEAVES = st.sampled_from(("weight", "bias", "norm.weight", "proj.weight"))
_SHAPES = st.lists(st.integers(min_value=1, max_value=8), min_size=0, max_size=3)
_DTYPES = st.sampled_from(("BF16", "F32", "F16"))


@st.composite
def _censuses(draw: st.DrawFn) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Keys under a few stems: zero, one or two numeric segments, drawn as
    arbitrary index sets (full products or not), each pattern one dtype and
    one shape — a census as the format allows it."""
    out: dict[str, tuple[str, tuple[int, ...]]] = {}
    described = draw(
        st.dictionaries(
            keys=st.tuples(_STEMS, st.integers(min_value=0, max_value=2), _LEAVES),
            values=st.tuples(_DTYPES, _SHAPES),
            min_size=1,
            max_size=6,
        )
    )
    for (stem, slots, leaf), (dtype, drawn_shape) in described.items():
        shape = tuple(drawn_shape)
        pattern = ".".join([stem, *(["*"] * slots), leaf])
        if slots == 0:
            out[pattern] = (dtype, shape)
            continue
        combos = draw(
            st.sets(
                st.tuples(*[st.integers(min_value=0, max_value=4)] * slots),
                min_size=1,
                max_size=12,
            )
        )
        for combo in combos:
            key = pattern
            for index in combo:
                key = key.replace("*", str(index), 1)
            out[key] = (dtype, shape)
    return out


@pytest.mark.property
class TestCompress:
    @_SETTINGS
    @given(keys=_censuses())
    def test_expand_of_compress_is_the_identity_on_keys(self, keys) -> None:
        assert expand(compress(keys)) == keys

    @_SETTINGS
    @given(keys=_censuses())
    def test_a_full_product_is_written_as_ranges_and_nothing_else_is(
        self, keys
    ) -> None:
        for pattern, entry in compress(keys).items():
            slots = pattern.count("*")
            if slots == 0:
                assert entry["indices"] == []
                continue
            if "ranges" in entry:
                assert len(entry["ranges"]) == slots
                size = 1
                for lo, hi in entry["ranges"]:
                    assert lo <= hi
                    size *= hi - lo + 1
                assert size == sum(1 for k in expand({pattern: entry}))
            else:
                indices = entry["indices"]
                assert indices == sorted(indices) and all(
                    len(i) == slots for i in indices
                )


@pytest.mark.unit
def test_two_shapes_under_one_pattern_are_refused_by_name() -> None:
    with pytest.raises(CensusAmbiguity, match="model.layers.\\*.w"):
        compress(
            {"model.layers.0.w": ("BF16", (2,)), "model.layers.1.w": ("BF16", (3,))}
        )
    with pytest.raises(CensusAmbiguity, match="one dtype and one shape"):
        compress({"a.0.w": ("BF16", (2,)), "a.1.w": ("F32", (2,))})


# --------------------------------------------------------------------------- #
# the fake cache, and the capture / check entry against it
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_cache(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    fake_hub_cache(root, LARGE, load_census(LLAMA70B_CENSUS), files=30)
    return root


@pytest.mark.unit
def test_a_census_laid_out_as_a_hub_cache_reads_back_through_the_protocol(
    fake_cache: Path,
) -> None:
    files = cached_checkpoint_files(LARGE, cache_dir=fake_cache)
    assert files is not None and len(files) == 30
    assert all(path.name.endswith("-of-00030.safetensors") for path in files)
    headers = read_headers(files)
    assert {n: (h.dtype, h.shape) for n, h in headers.items()} == load_census(
        LLAMA70B_CENSUS
    )
    # header-only: the data section is absent, by design
    assert all(path.stat().st_size < 1 << 16 for path in files)
    record = census_record(files, LARGE, "fake")
    committed = json.loads(LLAMA70B_CENSUS.read_text())
    assert record["patterns"] == committed["patterns"]
    assert (record["files"], record["tensors"]) == (30, 723)
    assert record["snapshot"] == "fake" and record["captured"]["date"]
    assert "node" not in record["captured"]  # the committed census names no host


@pytest.mark.unit
def test_a_shard_count_below_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one shard"):
        fake_hub_cache(tmp_path, LARGE, {}, files=0)


@pytest.mark.unit
def test_the_check_entry_passes_the_committed_census_and_names_a_drift(
    fake_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = census.main([LARGE, "--cache-dir", str(fake_cache), "--check"])
    assert code == 0
    out = capsys.readouterr()
    assert "matches the cached headers (723 tensors)" in out.out
    assert "committed snapshot '349b2ddb" in out.out  # the fake snapshot differs
    drifted = json.loads(LLAMA70B_CENSUS.read_text())
    drifted["patterns"]["model.norm.weight"]["shape"] = [4096]
    drifted["files"] = 29
    target = tmp_path / "drifted.json"
    target.write_text(json.dumps(drifted))
    code = census.main([LARGE, "--cache-dir", str(fake_cache), "--check", str(target)])
    assert code == 1
    err = capsys.readouterr().err
    assert "files: committed 29, the cache has 30" in err
    assert (
        "model.norm.weight: committed ('BF16', (4096,)), cached ('BF16', (8192,))"
        in err
    )


@pytest.mark.unit
def test_the_capture_entry_writes_a_census_and_refuses_an_uncached_key(
    fake_cache: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "census.json"
    assert census.main([LARGE, "--cache-dir", str(fake_cache), "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["patterns"] == json.loads(LLAMA70B_CENSUS.read_text())["patterns"]
    assert census.main(["org/absent", "--cache-dir", str(fake_cache), "--check"]) == 2
    assert "not in the Hub cache" in capsys.readouterr().err
    assert census.main([LARGE, "--cache-dir", str(fake_cache)]) == 2
