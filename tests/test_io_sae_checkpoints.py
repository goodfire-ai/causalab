"""Tests for ``causalab.io.sae_checkpoints`` — foreign SAE checkpoint readers.

Pure tensor reads (no GPU, no model load): a tiny synthetic vanilla SAE
checkpoint exercises [`read_sae_decoder`][causalab.io.sae_checkpoints.read_sae_decoder], and a tiny synthetic
block/Grassmannian SAE checkpoint exercises [`load_block_sae_frame`][causalab.io.sae_checkpoints.load_block_sae_frame]
(orthonormal frame, ``dim_mask`` restriction, metadata, error paths).
"""

from __future__ import annotations

import pytest
import torch

from causalab.io.sae_checkpoints import load_block_sae_frame, read_sae_decoder

pytestmark = pytest.mark.unit

D_MODEL = 16
D_SAE = 40
N_GROUPS = 6
K = 3


def _write_fake_vanilla_sae(tmp_path, name: str = "sae.pt") -> str:
    """A minimal vanilla-SAE checkpoint (a decoder matrix + d_model)."""
    torch.manual_seed(0)
    ckpt = {
        "model_config": {"d_model": D_MODEL, "d_sae": D_SAE},
        "state_dict": {"decoder.weight": torch.randn(D_MODEL, D_SAE)},
    }
    path = tmp_path / name
    torch.save(ckpt, str(path))
    return str(path)


def _write_fake_block_sae(
    tmp_path, *, dim_mask=None, b_gate=None, name: str = "grass.pt"
) -> str:
    """A minimal checkpoint shaped like a ``GrassmannianCoderSparse`` blob.

    ``B_raw`` is QR-orthonormalized per block to mirror the real SAE's
    ``enforce_ortho=True`` (each block's frame satisfies ``BᵀB≈I`` on disk).
    ``b_gate`` defaults to a scalar-per-block gate (length ``n_groups``); pass a
    ``(n_groups, group_size)`` tensor to exercise the per-feature-gate path.
    """
    torch.manual_seed(1)
    b_raw = torch.randn(N_GROUPS, D_MODEL, K)
    b_raw, _ = torch.linalg.qr(b_raw)  # (N_GROUPS, D_MODEL, K), orthonormal columns
    dm = torch.ones(N_GROUPS, K) if dim_mask is None else dim_mask
    bg = torch.arange(N_GROUPS, dtype=torch.float32) if b_gate is None else b_gate
    ckpt = {
        "tag": "grass_test",
        "scale": 3.744,
        "state_dict": {
            "B_raw": b_raw,
            "b_gate": bg,
            "dim_mask": dm,
            "b_dec": torch.zeros(D_MODEL),
            "log_gamma": torch.zeros(()),
        },
        "model_kwargs": {"d": D_MODEL, "n_groups": N_GROUPS, "group_size": K},
        "config": {"dataset": "model.layers.19", "d": D_MODEL},
    }
    path = tmp_path / name
    torch.save(ckpt, str(path))
    return str(path)


def test_read_sae_decoder(tmp_path):
    decoder, d_model = read_sae_decoder(_write_fake_vanilla_sae(tmp_path))
    assert decoder.shape == (D_MODEL, D_SAE)
    assert d_model == D_MODEL


def test_load_block_sae_frame_orthonormal_and_meta(tmp_path):
    path = _write_fake_block_sae(tmp_path)
    frame, meta = load_block_sae_frame(path, block_id=1)

    assert frame.shape == (D_MODEL, K)
    gram = frame.T @ frame
    assert torch.allclose(gram, torch.eye(K), atol=1e-5)

    assert meta["d_model"] == D_MODEL
    assert meta["group_size"] == K
    assert meta["n_groups"] == N_GROUPS
    assert meta["scale"] == pytest.approx(3.744)
    assert meta["layer"] == 19  # parsed from config dataset='model.layers.19'
    assert meta["b_gate_block"] == pytest.approx(1.0)  # b_gate = arange -> block 1
    assert meta["dim_mask_block"] == [1.0, 1.0, 1.0]
    assert meta["has_b_dec"] is True
    assert meta["has_log_gamma"] is True
    # Not stored in the checkpoint; the caller supplies them.
    assert meta["hook_site"] is None
    assert meta["model_id"] is None


def test_load_block_sae_frame_respects_dim_mask(tmp_path):
    """A dead column (dim_mask=0) is dropped; the frame narrows to alive dims."""
    dm = torch.ones(N_GROUPS, K)
    dm[2, 2] = 0.0  # one dead dim in block 2
    path = _write_fake_block_sae(tmp_path, dim_mask=dm, name="grass_dead.pt")

    frame, meta = load_block_sae_frame(path, block_id=2)
    assert frame.shape == (D_MODEL, K - 1)  # alive dims only
    gram = frame.T @ frame
    assert torch.allclose(gram, torch.eye(K - 1), atol=1e-5)
    assert meta["dim_mask_block"] == [1.0, 1.0, 0.0]


def test_load_block_sae_frame_per_feature_gate(tmp_path):
    """A per-feature gate (n_groups, group_size) must not crash; b_gate_block is a list.

    Foreign checkpoints may store a per-feature gate, so b_gate[bid] is a K-vector,
    not a scalar — float(b_gate[bid]) would raise and kill the whole load. The
    metadata coercion records it as a list instead.
    """
    bg = torch.arange(N_GROUPS * K, dtype=torch.float32).reshape(N_GROUPS, K)
    path = _write_fake_block_sae(tmp_path, b_gate=bg, name="grass_pergate.pt")
    frame, meta = load_block_sae_frame(path, block_id=2)
    assert frame.shape == (D_MODEL, K)  # frame load is unaffected
    assert meta["b_gate_block"] == [6.0, 7.0, 8.0]  # block 2 of arange(N_GROUPS*K)


def test_load_block_sae_frame_block_id_zero(tmp_path):
    """block_id=0 is a legitimate index, not a 'missing' value."""
    frame, _ = load_block_sae_frame(_write_fake_block_sae(tmp_path), block_id=0)
    assert frame.shape == (D_MODEL, K)


def test_load_block_sae_frame_rejects_bad_inputs(tmp_path):
    path = _write_fake_block_sae(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        load_block_sae_frame(path, block_id=N_GROUPS)
    # A vanilla SAE checkpoint has no B_raw.
    with pytest.raises(KeyError, match="B_raw"):
        load_block_sae_frame(
            _write_fake_vanilla_sae(tmp_path, name="vanilla.pt"), block_id=0
        )
