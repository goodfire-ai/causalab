"""``build_stack`` places every stage on the run's device (spec §2.5, §8).

The bug this guards: nothing in ``featurizers.py`` mentioned a device, so a
``subspace`` weight was built on CPU by ``torch.randn`` and both executor call
sites left it there. ``causalab run <das_doc>.json --device cuda`` then died in
``Subspace.featurize`` with ``Expected all tensors to be on the same device,
but got mat2 is on cpu, different from other tensors on cuda:0`` — every
document with a featurizer (DAS, DBM, das-apply, PCA, gates, SAE), i.e. the
headline analysis node, was CPU-only. The corpus tests all run tiny-random on
CPU, which is why nothing caught it.

The placement assertions are device-parametrized over whatever the box has, so
they are meaningful without a CUDA card: ``cpu`` always runs and pins that the
requested device reaches every parameter AND every registered buffer of every
stage kind, ``mps`` (a developer Mac) and ``cuda`` (a GPU box or the golden
tier) additionally prove a *foreign* device is really applied — a
``Parameter`` left behind by ``.to()`` fails there and cannot fail on cpu.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared.featurizers import (
    Gate,
    LoadedLinear,
    Sae,
    Standardize,
    Subspace,
    build_stack,
)
from causalab.io.tensor_files import TensorBundle
from causalab.protocol.schema import FeaturizerSpec

pytestmark = pytest.mark.unit

WIDTH = 8
K = 3

DEVICES = [
    "cpu",
    *(["cuda"] if torch.cuda.is_available() else []),
    *(["mps"] if torch.backends.mps.is_available() else []),
]


def _load_tensors(_path: str) -> TensorBundle:
    """Loaded bundles arrive from files on CPU, whatever the run's device."""
    return TensorBundle(
        tensors={
            "weight": torch.eye(WIDTH)[:, :K].contiguous(),
            "mu": torch.zeros(WIDTH),
            "sigma": torch.ones(WIDTH),
            "enc": torch.zeros(WIDTH, 5),
            "dec": torch.zeros(5, WIDTH),
            "b_enc": torch.zeros(5),
            "b_dec": torch.zeros(WIDTH),
        },
        entry_coords={},
    )


SPECS: dict[str, FeaturizerSpec] = {
    "rot": FeaturizerSpec(kind="subspace", k=K),
    "gate": FeaturizerSpec(kind="gate"),
    "pca": FeaturizerSpec(kind="pca", k=K, file_path="pca.safetensors"),
    "std": FeaturizerSpec(kind="standardize", file_path="std.safetensors"),
    "sae": FeaturizerSpec(kind="sae", file_path="sae.safetensors"),
}


def _tensors(stage: Any) -> list[torch.Tensor]:
    """Parameters and registered buffers alike — a buffer left on the CPU
    breaks a cuda forward exactly as loudly as a parameter does."""
    return [*stage.parameters(), *stage.buffers()]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(SPECS))
def test_build_stack_places_every_stage_on_the_requested_device(
    name: str, device: str
) -> None:
    stack = build_stack(
        name,
        SPECS,
        width=WIDTH,
        load_tensors=_load_tensors,
        stage_cache={},
        device=device,
    )
    (stage,) = stack.stages
    tensors = _tensors(stage)
    assert tensors, f"{name} exposes no tensor to place"
    for tensor in tensors:
        assert tensor.device.type == device, (
            f"{name}: {tensor.shape} on {tensor.device}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_build_stack_places_a_composed_chain(device: str) -> None:
    """§2.5 composition: the later stages are built too, and must move too."""
    stack = build_stack(
        ["rot", "gate"],
        SPECS,
        width=WIDTH,
        load_tensors=_load_tensors,
        stage_cache={},
        device=device,
    )
    for stage in stack.stages:
        for tensor in _tensors(stage):
            assert tensor.device.type == device


def test_build_stack_defaults_to_cpu() -> None:
    """The default keeps every CPU-only caller and test unchanged."""
    stack = build_stack(
        "rot", SPECS, width=WIDTH, load_tensors=_load_tensors, stage_cache={}
    )
    (stage,) = stack.stages
    assert all(t.device.type == "cpu" for t in _tensors(stage))


@pytest.mark.parametrize("device", DEVICES)
def test_a_seeded_subspace_init_is_bit_identical_across_devices(device: str) -> None:
    """Build-on-CPU-then-move is load bearing, not incidental: the init draws
    from a CPU ``torch.Generator``, so a device-side ``torch.randn`` would give
    a different (still seeded) rotation and silently break run-to-run
    comparability between a CPU and a GPU run of the same document."""
    on_cpu = build_stack(
        "rot", SPECS, width=WIDTH, load_tensors=_load_tensors, stage_cache={}
    ).stages[0]
    on_device = build_stack(
        "rot",
        SPECS,
        width=WIDTH,
        load_tensors=_load_tensors,
        stage_cache={},
        device=device,
    ).stages[0]
    assert torch.equal(
        on_cpu.slot_params()["weight"].detach(),
        on_device.slot_params()["weight"].detach().cpu(),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_a_placed_stack_featurizes_a_tensor_on_that_device(device: str) -> None:
    """The symptom itself: ``x.to(q.dtype) @ q`` with x on the run's device."""
    stack = build_stack(
        ["rot", "gate"],
        SPECS,
        width=WIDTH,
        load_tensors=_load_tensors,
        stage_cache={},
        device=device,
    )
    x = torch.randn(2, 1, WIDTH, device=device)
    f, errs = stack.featurize(x)
    assert f.device.type == device
    assert stack.inverse(f, errs).device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_loaded_stages_move_off_the_cpu_tensors_they_were_handed(
    device: str,
) -> None:
    """``load_tensors`` reads files, which produce CPU tensors — the placement
    has to survive that, not just cover freshly constructed parameters."""
    for name, expected in (
        ("pca", LoadedLinear),
        ("std", Standardize),
        ("sae", Sae),
    ):
        stack = build_stack(
            name,
            SPECS,
            width=WIDTH,
            load_tensors=_load_tensors,
            stage_cache={},
            device=device,
        )
        (stage,) = stack.stages
        assert isinstance(stage, expected)
        assert all(t.device.type == device for t in _tensors(stage))


def test_the_trainable_kinds_are_the_ones_with_parameters() -> None:
    """A guard on the assertions above: if `subspace`/`gate` stopped exposing
    parameters, the placement tests would pass vacuously for them."""
    rot = build_stack(
        "rot", SPECS, width=WIDTH, load_tensors=_load_tensors, stage_cache={}
    ).stages[0]
    gate = build_stack(
        "gate", SPECS, width=WIDTH, load_tensors=_load_tensors, stage_cache={}
    ).stages[0]
    assert isinstance(rot, Subspace) and list(rot.parameters())
    assert isinstance(gate, Gate) and list(gate.parameters())
