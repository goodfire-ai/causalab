"""
pytest unit-tests for LM_units.py

These tests assume:
* model_units.py and featurizers.py have already been patched as in previous
  steps (feature-bounds checks, mutable-default fixes, etc.).
* No actual LLM weights are loaded; we stay entirely on synthetic data.
"""

from __future__ import annotations

import pytest
import torch

import neural.LM_units as LM
import neural.featurizers as F


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g


# --------------------------------------------------------------------------- #
#  1. Mutable-default featurizer fix                                           #
# --------------------------------------------------------------------------- #
def test_residualstream_featurizers_unique():
    rs1 = LM.ResidualStream(layer=0, token_indices=[0])
    rs2 = LM.ResidualStream(layer=0, token_indices=[0])
    assert rs1.featurizer is not rs2.featurizer


def test_attentionhead_featurizers_unique():
    ah1 = LM.AttentionHead(layer=0, head=5, token_indices=[0])
    ah2 = LM.AttentionHead(layer=0, head=5, token_indices=[0])
    assert ah1.featurizer is not ah2.featurizer


# --------------------------------------------------------------------------- #
#  2. AttentionHead index structure                                            #
# --------------------------------------------------------------------------- #
def test_attentionhead_index_structure():
    ah = LM.AttentionHead(layer=1, head=7, token_indices=[3])
    idx = ah.index_component("dummy")
    assert idx == [[[7]], [[3]]]


# --------------------------------------------------------------------------- #
#  3. Feature-index bounds behaviour                                           #
# --------------------------------------------------------------------------- #
def test_attentionhead_feature_bounds_violation():
    big_feat   = F.SubspaceFeaturizer(shape=(4, 4), trainable=False)   # 4 features
    small_feat = F.SubspaceFeaturizer(shape=(4, 2), trainable=False)   # 2 features

    ah = LM.AttentionHead(
        layer=0,
        head=0,
        token_indices=[0],
        featurizer=big_feat,
        feature_indices=[3],   # valid in 4-dim space
    )

    with pytest.raises(ValueError):
        ah.set_featurizer(small_feat)
