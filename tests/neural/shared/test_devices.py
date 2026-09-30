"""``DeviceMap``: the reference engine's device contract as a map over the
blocks (``docs/model_parallelism.md`` §5.1).

One string from the user — ``cpu``, ``cuda:1``, ``mps``, or a comma list
``cuda:0,cuda:1`` — becomes the device of the embedding, of every block and
of the head. Every refusal here sits beside its valid twin and names the
field, and the property tier states what every split must satisfy: the block
ranges partition the layers, contiguous and in device order, and
``device_of`` is total.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.devices import DeviceMap, normalize_device
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import TreeAddress

SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

CPU = torch.device("cpu")
LLAMA_TREE = TreeAddress(
    blocks="model.layers", embedding="model.embed_tokens", final_norm="model.norm"
)


def _cuda(index: int) -> torch.device:
    # constructing a CUDA device object needs no CUDA runtime
    return torch.device("cuda", index)


def _current_cuda() -> int:
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


# --------------------------------------------------------------------------- #
# parsing one device
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestParseOneDevice:
    def test_cpu_everywhere(self) -> None:
        devices = DeviceMap.parse("cpu", 3)
        assert devices.embedding == CPU
        assert devices.blocks == (CPU, CPU, CPU)
        assert devices.head == CPU
        assert devices.single == CPU
        assert devices.devices == frozenset({CPU})
        assert devices.requested == "cpu"
        assert devices.spelling == "cpu"
        assert not devices.is_cuda

    def test_an_explicit_cuda_ordinal_is_kept(self) -> None:
        devices = DeviceMap.parse("cuda:1", 2)
        assert devices.single == _cuda(1)
        assert devices.is_cuda

    def test_bare_cuda_is_the_current_ordinal(self) -> None:
        """``cuda`` and ``cuda:<current>`` are one placement, so the two
        spellings compare equal — what lets a caller-owned bundle built with
        one be checked against an engine built with the other."""
        bare = DeviceMap.parse("cuda", 2)
        explicit = DeviceMap.parse(f"cuda:{_current_cuda()}", 2)
        assert bare == explicit
        assert bare.single == _cuda(_current_cuda())
        # the spelling the user gave is kept for the record, outside equality
        assert bare.requested == "cuda" and explicit.requested != "cuda"

    def test_mps_carries_its_ordinal(self) -> None:
        """An ``mps`` parameter reports ``mps:0``; the parsed map says the
        same, so a map derived from a placed model equals the parsed one."""
        assert DeviceMap.parse("mps", 1).single == torch.device("mps", 0)
        assert normalize_device("mps") == torch.device("mps", 0)
        assert normalize_device("cpu") == CPU

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert DeviceMap.parse(" cpu ", 1) == DeviceMap.parse("cpu", 1)

    def test_an_unknown_device_is_refused_by_name(self) -> None:
        with pytest.raises(ProtocolError, match="bogus") as err:
            DeviceMap.parse("bogus", 2)
        assert err.value.code == "P4"

    def test_an_empty_string_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="empty") as err:
            DeviceMap.parse("", 2)
        assert err.value.code == "P4"
        with pytest.raises(ProtocolError, match="empty"):
            DeviceMap.parse("  ", 2)

    def test_a_model_with_no_blocks_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="block"):
            DeviceMap.parse("cpu", 0)


# --------------------------------------------------------------------------- #
# parsing a list
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestParseAList:
    def test_the_blocks_split_into_even_ranges_in_order(self) -> None:
        devices = DeviceMap.parse("cuda:0,cuda:1", 4)
        assert devices.blocks == (_cuda(0), _cuda(0), _cuda(1), _cuda(1))
        assert devices.single is None
        assert devices.devices == frozenset({_cuda(0), _cuda(1)})
        assert devices.is_cuda

    def test_the_remainder_goes_to_the_last_device(self) -> None:
        devices = DeviceMap.parse("cuda:0,cuda:1", 5)
        assert devices.blocks == (_cuda(0), _cuda(0), _cuda(1), _cuda(1), _cuda(1))
        three = DeviceMap.parse("cuda:0,cuda:1,cuda:2", 8)
        assert three.blocks == (
            _cuda(0),
            _cuda(0),
            _cuda(1),
            _cuda(1),
            _cuda(2),
            _cuda(2),
            _cuda(2),
            _cuda(2),
        )

    def test_embedding_on_the_first_device_head_on_the_last(self) -> None:
        devices = DeviceMap.parse("cuda:2,cuda:0", 2)
        assert devices.embedding == _cuda(2)
        assert devices.head == _cuda(0)
        assert devices.blocks == (_cuda(2), _cuda(0))

    def test_device_of_is_the_block_range(self) -> None:
        devices = DeviceMap.parse("cuda:0,cuda:1", 5)
        assert [devices.device_of(i) for i in range(5)] == list(devices.blocks)

    def test_device_of_refuses_a_layer_outside_the_tower(self) -> None:
        devices = DeviceMap.parse("cpu", 2)
        with pytest.raises(IndexError):
            devices.device_of(2)
        with pytest.raises(IndexError):
            devices.device_of(-1)

    def test_spelling_is_the_canonical_list(self) -> None:
        assert DeviceMap.parse("cuda,cuda:1", 2).spelling == (
            f"cuda:{_current_cuda()},cuda:1"
        )
        assert DeviceMap.parse("cpu,mps", 2).spelling == "cpu,mps:0"

    def test_a_list_round_trips_through_its_spelling(self) -> None:
        devices = DeviceMap.parse("cuda:1,cuda:0", 7)
        assert DeviceMap.parse(devices.spelling, 7) == devices

    def test_cpu_and_mps_may_share_a_map(self) -> None:
        """The valid twin of the CUDA-mixing refusal: two non-CUDA devices
        take the same kernel path, so they may be mixed."""
        devices = DeviceMap.parse("cpu,mps", 2)
        assert devices.blocks == (CPU, torch.device("mps", 0))
        assert not devices.is_cuda

    def test_mixing_cuda_with_another_device_type_is_refused(self) -> None:
        """The DeltaNet kernel path is bound once per model, for CUDA or not
        (``shared/kernels.py``); a tower with blocks on both has no one path."""
        with pytest.raises(ProtocolError, match="CUDA") as err:
            DeviceMap.parse("cuda:0,cpu", 2)
        assert err.value.code == "P4"
        with pytest.raises(ProtocolError, match="CUDA"):
            DeviceMap.parse("mps,cuda:0", 2)

    def test_a_repeated_device_is_refused_by_name(self) -> None:
        with pytest.raises(ProtocolError, match="cuda:0") as err:
            DeviceMap.parse("cuda:0,cuda:0", 2)
        assert err.value.code == "P4" and "repeat" in str(err.value)

    def test_a_repeat_after_normalisation_is_still_a_repeat(self) -> None:
        current = _current_cuda()
        with pytest.raises(ProtocolError, match="repeat"):
            DeviceMap.parse(f"cuda,cuda:{current}", 2)

    def test_an_empty_entry_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="empty") as err:
            DeviceMap.parse("cpu,", 2)
        assert err.value.code == "P4"
        with pytest.raises(ProtocolError, match="empty"):
            DeviceMap.parse("cuda:0,,cuda:1", 2)

    def test_more_devices_than_blocks_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="3 device") as err:
            DeviceMap.parse("cuda:0,cuda:1,cuda:2", 2)
        assert err.value.code == "P4" and "2 block" in str(err.value)
        # the twin: exactly as many devices as blocks is one block each
        assert DeviceMap.parse("cuda:0,cuda:1", 2).blocks == (_cuda(0), _cuda(1))

    def test_requested_is_outside_equality_and_hashing(self) -> None:
        a = DeviceMap.parse("cuda:0,cuda:1", 2)
        b = DeviceMap.parse(" cuda:0 , cuda:1 ", 2)
        assert a == b and hash(a) == hash(b)
        assert a.requested != b.requested


# --------------------------------------------------------------------------- #
# the map over a module tree
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestModuleMap:
    def test_module_map_names_every_placed_prefix(self) -> None:
        devices = DeviceMap.parse("cuda:0,cuda:1", 3)
        # three blocks over two devices: one, then two (the remainder is last)
        # torch.device values, not spellings: accelerate reads a spelled "cpu"
        # as offload (devices.py, ``module_map``)
        assert devices.module_map(LLAMA_TREE) == {
            "": _cuda(0),
            "model.embed_tokens": _cuda(0),
            "model.layers.0": _cuda(0),
            "model.layers.1": _cuda(1),
            "model.layers.2": _cuda(1),
            "model.norm": _cuda(1),
            "lm_head": _cuda(1),
        }
        assert all(
            isinstance(value, torch.device)
            for value in DeviceMap.parse("cpu,mps", 2).module_map(LLAMA_TREE).values()
        )

    def test_device_for_matches_the_longest_dotted_prefix(self) -> None:
        devices = DeviceMap.parse("cuda:0,cuda:1", 12)
        assert devices.device_for("model.layers.1.mlp.up_proj.weight", LLAMA_TREE) == (
            _cuda(0)
        )
        # ``model.layers.1`` must not swallow ``model.layers.10``
        assert devices.device_for("model.layers.10.mlp.up_proj.weight", LLAMA_TREE) == (
            _cuda(1)
        )
        assert devices.device_for("model.embed_tokens.weight", LLAMA_TREE) == _cuda(0)
        assert devices.device_for("model.norm.weight", LLAMA_TREE) == _cuda(1)
        assert devices.device_for("lm_head.weight", LLAMA_TREE) == _cuda(1)
        # everything the tree does not place rides with the embedding
        assert devices.device_for("model.rotary_emb.inv_freq", LLAMA_TREE) == _cuda(0)

    def test_a_derived_map_reads_where_the_parameters_are(self) -> None:
        embedding = torch.nn.Embedding(4, 2)
        blocks = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        head = torch.nn.Linear(2, 4)
        devices = DeviceMap.of_modules(embedding, blocks, head)
        assert devices == DeviceMap.parse("cpu", 2)
        assert devices.requested == "cpu"

    def test_a_block_straddling_devices_is_refused_by_index(self) -> None:
        embedding = torch.nn.Embedding(4, 2)
        straddling = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Linear(2, 2, device="meta")
        )
        blocks = [torch.nn.Linear(2, 2), straddling]
        with pytest.raises(ProtocolError, match="block 1") as err:
            DeviceMap.of_modules(embedding, blocks, torch.nn.Linear(2, 4))
        assert err.value.code == "P4"

    def test_a_parameter_without_storage_is_refused(self) -> None:
        """An offloaded (``meta``) block has no device to run on."""
        embedding = torch.nn.Embedding(4, 2)
        blocks = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2, device="meta")]
        with pytest.raises(ProtocolError, match="meta") as err:
            DeviceMap.of_modules(embedding, blocks, torch.nn.Linear(2, 4))
        assert err.value.code == "P4"

    def test_a_module_without_tensors_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="no parameter"):
            DeviceMap.of_modules(
                torch.nn.Embedding(4, 2), [torch.nn.Identity()], torch.nn.Linear(2, 4)
            )

    def test_a_module_without_tensors_is_recorded_on_the_empty_device(self) -> None:
        """A pipeline stage's identity layers (``docs/model_parallelism.md``
        §6.5) have no tensors and run on the stage's device: ``empty`` names
        it, for the blocks, the embedding and the head alike."""
        devices = DeviceMap.of_modules(
            torch.nn.Identity(),
            [torch.nn.Linear(2, 2), torch.nn.Identity()],
            torch.nn.Identity(),
            empty=CPU,
        )
        assert devices == DeviceMap.parse("cpu", 2)
        # a module with tensors is still read off its tensors, not ``empty``
        with pytest.raises(ProtocolError, match="meta"):
            DeviceMap.of_modules(
                torch.nn.Embedding(4, 2),
                [torch.nn.Linear(2, 2, device="meta")],
                torch.nn.Identity(),
                empty=CPU,
            )


# --------------------------------------------------------------------------- #
# properties of every split
# --------------------------------------------------------------------------- #


def _layouts() -> st.SearchStrategy[tuple[int, int]]:
    return st.integers(min_value=1, max_value=40).flatmap(
        lambda layers: st.tuples(
            st.just(layers), st.integers(min_value=1, max_value=min(layers, 8))
        )
    )


@pytest.mark.property
class TestSplitProperties:
    @SETTINGS
    @given(_layouts())
    def test_block_ranges_partition_the_layers_in_device_order(
        self, layout: tuple[int, int]
    ) -> None:
        layers, count = layout
        names = [f"cuda:{i}" for i in range(count)]
        devices = DeviceMap.parse(",".join(names), layers)
        assert len(devices.blocks) == layers
        # in device order: the ordinal a block sits on never decreases
        ordinals = [d.index for d in devices.blocks]
        assert ordinals == sorted(ordinals)
        # every device holds a contiguous, non-empty range
        for i in range(count):
            held = [layer for layer, o in enumerate(ordinals) if o == i]
            assert held, f"device {i} holds no block"
            assert held == list(range(held[0], held[-1] + 1))
        # even: every range is the base size, the last one takes the remainder
        base, remainder = divmod(layers, count)
        sizes = [ordinals.count(i) for i in range(count)]
        assert sizes[:-1] == [base] * (count - 1)
        assert sizes[-1] == base + remainder
        assert devices.embedding == _cuda(0)
        assert devices.head == _cuda(count - 1)

    @SETTINGS
    @given(_layouts())
    def test_device_of_is_total_and_agrees_with_the_blocks(
        self, layout: tuple[int, int]
    ) -> None:
        layers, count = layout
        devices = DeviceMap.parse(",".join(f"cuda:{i}" for i in range(count)), layers)
        assert [devices.device_of(i) for i in range(layers)] == list(devices.blocks)
        assert devices.devices == frozenset(_cuda(i) for i in range(count))
        assert (devices.single is None) == (count > 1)

    @SETTINGS
    @given(_layouts())
    def test_the_spelling_round_trips(self, layout: tuple[int, int]) -> None:
        layers, count = layout
        devices = DeviceMap.parse(",".join(f"cuda:{i}" for i in range(count)), layers)
        assert DeviceMap.parse(devices.spelling, layers) == devices
