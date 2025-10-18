"""
pytest unit-tests for featurizers.py

Run with:
    pytest -q test_featurizers.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import torch
import pytest

import neural.featurizers as F  # the module we just rewrote


# --------------------------------------------------------------------------- #
#  Helpers / fixtures                                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def randn(shape: Tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).float()


# --------------------------------------------------------------------------- #
#  Identity featurizer                                                        #
# --------------------------------------------------------------------------- #
def test_identity_roundtrip(rng):
    x = randn((2, 4), rng)

    feat = F.Featurizer(n_features=4)  # identity by default
    f, err = feat.featurize(x)
    assert err is None
    assert torch.equal(f, x)

    x_rec = feat.inverse_featurize(f, err)
    assert torch.equal(x, x_rec)


# --------------------------------------------------------------------------- #
#  Subspace featurizer                                                        #
# --------------------------------------------------------------------------- #
def test_subspace_roundtrip(rng):
    x = randn((3, 6), rng)

    sub = F.SubspaceFeaturizer(shape=(6, 6), trainable=False)
    f, err = sub.featurize(x)
    x_rec = sub.inverse_featurize(f, err)

    assert torch.allclose(x, x_rec, atol=1e-5)


# --------------------------------------------------------------------------- #
#  Interchange intervention                                                   #
# --------------------------------------------------------------------------- #
def test_interchange_swaps_when_subspaces_none(rng):
    x_base = randn((2, 4), rng)
    x_src = randn((2, 4), rng)

    feat = F.Featurizer(n_features=4)
    Interchange = feat.get_interchange_intervention()
    inter = Interchange()  # default kwargs ok

    out = inter(x_base, x_src, subspaces=None)
    assert torch.equal(out, x_src)


# --------------------------------------------------------------------------- #
#  Mask intervention basic invariants                                         #
# --------------------------------------------------------------------------- #
def test_mask_requires_n_features():
    feat = F.Featurizer()  # n_features=None
    with pytest.raises(ValueError):
        _ = feat.get_mask_intervention()


def test_mask_forward_training_and_eval(rng):
    x_base = randn((1, 4), rng)
    x_src = randn((1, 4), rng)

    feat = F.Featurizer(n_features=4)
    MaskCls = feat.get_mask_intervention()
    mask = MaskCls()

    # Temperature must be set first
    mask.set_temperature(1.0)

    # --------------------------------------------------------------------- #
    # 1. Training mode – push mask to 1 ⇒ output ≈ src                      #
    # --------------------------------------------------------------------- #
    mask.train()
    mask.mask.data.fill_(20.0)          # sigmoid(20) ≈ 1 − 2e-9
    out_train = mask(x_base, x_src)
    assert torch.allclose(out_train, x_src, atol=1e-6)

    # --------------------------------------------------------------------- #
    # 2. Eval mode – binary gate                                            #
    # --------------------------------------------------------------------- #
    mask.eval()
    out_eval = mask(x_base, x_src)
    assert torch.allclose(out_eval, x_src, atol=1e-5)


# --------------------------------------------------------------------------- #
#  Serialization round-trips                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [lambda: F.Featurizer(n_features=4),
                                     lambda: F.SubspaceFeaturizer(shape=(4, 4),
                                                                  trainable=False)])
def test_save_load_roundtrip(factory, tmp_path: Path, rng):
    feat = factory()
    x = randn((2, 4), rng)

    f, err = feat.featurize(x)
    x_rec = feat.inverse_featurize(f, err)

    path_root = tmp_path / "unit"
    feat.save_modules(str(path_root))

    loaded = F.Featurizer.load_modules(str(path_root))
    f2, err2 = loaded.featurize(x)
    x_rec2 = loaded.inverse_featurize(f2, err2)

    # reconstruction stays the same after reload
    assert torch.allclose(x_rec, x_rec2, atol=1e-6)
    # and round-trip still faithful to original input
    assert torch.allclose(x, x_rec2, atol=1e-5)


# --------------------------------------------------------------------------- #
#  __str__ helpers                                                            #
# --------------------------------------------------------------------------- #
def test_collect_str():
    feat = F.Featurizer(n_features=4, id="my_feat")
    Collect = feat.get_collect_intervention()
    col = Collect()
    assert "my_feat" in str(col)
