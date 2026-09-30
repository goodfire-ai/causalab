"""The checkpoint as the protocol layer reads it, torch-free
(``protocol/checkpoint_census.py``; ``docs/model_parallelism.md`` §5.3, §11):
a safetensors header off the file alone, the cached shards of a checkpoint
with no request made, and the tower rule that says which keys the text
model consumes.

``unit`` throughout: a hand-authored safetensors file (the 8-byte length,
the JSON table, zero data), a hand-laid Hub cache, and the real header
census of the public ``Qwen/Qwen3.6-35B-A3B`` checkpoint
(``tests/golden/parallel_headers_a3b.json``) — 1045 tensors on disk, of which the 693 under
``model.language_model`` plus the head are the text model's 64.56 GiB, the
vision encoder and the MTP head left out exactly as the loader leaves them
unread (``weights.renamed_keys``).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from causalab.protocol.checkpoint_census import (
    ITEMSIZES,
    TensorHeader,
    cached_checkpoint_files,
    checkpoint_targets,
    read_header,
    read_headers,
    tree_of,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import GPT2_TREE, GPTJ_TREE, LLAMA_TREE, TreeAddress

from tests._helpers.header_census import load_census

pytestmark = pytest.mark.unit


def _safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> Path:
    """A safetensors file with the named tensors and zeroed data."""
    table: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = ITEMSIZES[dtype]
        for n in shape:
            size *= n
        table[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header = json.dumps(table).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(offset))
    return path


# --------------------------------------------------------------------------- #
# the header
# --------------------------------------------------------------------------- #


def test_the_header_is_read_off_the_file_alone(tmp_path: Path) -> None:
    path = _safetensors(
        tmp_path / "model.safetensors",
        {"a.weight": ("BF16", (4, 8)), "b.bias": ("F32", (3,))},
    )
    headers = read_header(path)
    assert headers == {
        "a.weight": TensorHeader("BF16", (4, 8)),
        "b.bias": TensorHeader("F32", (3,)),
    }
    assert headers["a.weight"].itemsize == 2 and headers["a.weight"].elements == 32
    assert headers["b.bias"].itemsize == 4 and headers["b.bias"].elements == 3
    assert TensorHeader("F32", ()).elements == 1


def test_a_truncated_file_and_an_unspelled_dtype_are_refused_by_name(tmp_path) -> None:
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")
    with pytest.raises(ProtocolError, match="truncated header"):
        read_header(short)
    with pytest.raises(ProtocolError, match="not one the format spells"):
        _ = TensorHeader("Q4", (1,)).itemsize


def test_read_headers_merges_the_shards(tmp_path: Path) -> None:
    one = _safetensors(tmp_path / "one.safetensors", {"a": ("BF16", (2,))})
    two = _safetensors(tmp_path / "two.safetensors", {"b": ("BF16", (3,))})
    assert set(read_headers([one, two])) == {"a", "b"}


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #


def _hub_cache(root: Path, repo: str, files: dict[str, bytes]) -> Path:
    """A Hub cache holding ``repo`` at one snapshot, ``main`` pointing at it."""
    folder = root / f"models--{repo.replace('/', '--')}"
    snapshot = folder / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("abc123")
    for name, data in files.items():
        (snapshot / name).write_bytes(data)
    return snapshot


def test_an_indexed_checkpoint_in_the_cache_names_its_shards(tmp_path: Path) -> None:
    index = json.dumps(
        {"weight_map": {"a": "model-00002.safetensors", "b": "model-00001.safetensors"}}
    ).encode()
    snapshot = _hub_cache(
        tmp_path,
        "org/model",
        {
            "model.safetensors.index.json": index,
            "model-00001.safetensors": b"",
            "model-00002.safetensors": b"",
        },
    )
    files = cached_checkpoint_files("org/model", cache_dir=tmp_path)
    assert files == (
        snapshot / "model-00001.safetensors",
        snapshot / "model-00002.safetensors",
    )


def test_a_single_file_checkpoint_and_a_missing_one(tmp_path: Path) -> None:
    snapshot = _hub_cache(tmp_path, "org/single", {"model.safetensors": b""})
    assert cached_checkpoint_files("org/single", cache_dir=tmp_path) == (
        snapshot / "model.safetensors",
    )
    assert cached_checkpoint_files("org/absent", cache_dir=tmp_path) is None


def test_an_index_naming_an_absent_shard_is_not_cached(tmp_path: Path) -> None:
    index = json.dumps({"weight_map": {"a": "model-00001.safetensors"}}).encode()
    _hub_cache(tmp_path, "org/partial", {"model.safetensors.index.json": index})
    assert cached_checkpoint_files("org/partial", cache_dir=tmp_path) is None


def test_a_local_directory_is_read_as_is(tmp_path: Path) -> None:
    local = tmp_path / "ckpt"
    local.mkdir()
    (local / "model.safetensors").write_bytes(b"")
    assert cached_checkpoint_files(str(local)) == (local / "model.safetensors",)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cached_checkpoint_files(str(empty)) is None


# --------------------------------------------------------------------------- #
# the tower rule
# --------------------------------------------------------------------------- #

LLAMA_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def test_a_plain_checkpoint_maps_onto_itself() -> None:
    targets = checkpoint_targets(LLAMA_KEYS, LLAMA_TREE.tree, num_layers=2)
    assert targets == {key: key for key in LLAMA_KEYS}
    assert tree_of(LLAMA_KEYS, 2) == LLAMA_TREE.tree


def test_a_gpt2_checkpoint_detects_the_gpt2_tree() -> None:
    keys = (
        "transformer.wte.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.ln_f.weight",
    )
    assert tree_of(keys, 1) == GPT2_TREE.tree
    assert checkpoint_targets(keys, GPT2_TREE.tree, 1) == {k: k for k in keys}


def test_a_gptj_checkpoint_detects_the_tree_it_shares_with_gpt2() -> None:
    """GPT-J and GPT-2 are two families with one tree address; the key tree
    names the address, so the two are one match rather than an ambiguity
    that left every GPT-2 estimate undecided once GPT-J was registered."""
    keys = (
        "transformer.wte.weight",
        "transformer.h.0.attn.q_proj.weight",
        "transformer.h.0.mlp.fc_out.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
    )
    assert GPTJ_TREE.tree == GPT2_TREE.tree
    assert tree_of(keys, 1) == GPTJ_TREE.tree
    assert checkpoint_targets(keys, GPTJ_TREE.tree, 1) == {k: k for k in keys}


def test_the_layer_count_decides_the_tower_and_none_is_undecided() -> None:
    assert tree_of(LLAMA_KEYS, 3) is None
    assert checkpoint_targets(LLAMA_KEYS, LLAMA_TREE.tree, 3) is None
    twins = LLAMA_KEYS + ("other.layers.0.x.weight", "other.layers.1.x.weight")
    assert checkpoint_targets(twins, LLAMA_TREE.tree, 2) is None


def test_a_wrapped_text_tower_renames_onto_the_tree_root_and_drops_the_rest() -> None:
    keys = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.1.mlp.experts.gate_up_proj",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.fc.weight",
    )
    targets = checkpoint_targets(keys, LLAMA_TREE.tree, num_layers=2)
    assert targets == {
        "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj": (
            "model.layers.0.mlp.experts.gate_up_proj"
        ),
        "model.language_model.layers.1.mlp.experts.gate_up_proj": (
            "model.layers.1.mlp.experts.gate_up_proj"
        ),
        "model.language_model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the block count decides the tower: asked for a one-block model, the
    # one-block MTP head is the text tower and the language model is not
    one_block = checkpoint_targets(keys, LLAMA_TREE.tree, num_layers=1)
    assert one_block is not None and set(one_block) == {
        "mtp.layers.0.mlp.gate.weight",
        "mtp.fc.weight",
        "lm_head.weight",
    }
    # and two towers of the same block count are undecided
    twins = keys + ("other.layers.0.w", "other.layers.1.w")
    assert checkpoint_targets(twins, LLAMA_TREE.tree, num_layers=2) is None


def test_the_a3b_census_is_the_text_models_64_gib() -> None:
    census = load_census()
    assert len(census) == 1045
    tree = tree_of(census, 40)
    assert tree == LLAMA_TREE.tree
    targets = checkpoint_targets(census, tree, 40)
    assert targets is not None and len(targets) == 693
    assert not any(".visual." in key or key.startswith("mtp.") for key in targets)
    total = 0
    for key in targets:
        dtype, shape = census[key]
        count = ITEMSIZES[dtype]
        for n in shape:
            count *= n
        total += count
    assert total == 69321221376  # 64.56 GiB, every rank's read at world 1
    assert targets["model.language_model.layers.39.mlp.gate.weight"] == (
        "model.layers.39.mlp.gate.weight"
    )


def test_a_tree_is_a_dotted_address() -> None:
    tree = TreeAddress(blocks="m.layers", embedding="m.embed", final_norm="m.norm")
    assert checkpoint_targets(("m.layers.0.w", "lm_head.weight"), tree, 1) == {
        "m.layers.0.w": "m.layers.0.w",
        "lm_head.weight": "lm_head.weight",
    }
