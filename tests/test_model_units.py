"""
pytest unit-tests for the core abstractions in model_units.py
(no ResidualStream / AttentionHead).
"""

import pytest
import torch

import neural.model_units as MU
import neural.featurizers as F  # the module we just rewrote

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def make_static_component():
    """Return a trivial StaticComponent for tests."""
    return MU.StaticComponent(layer=0, component_type="block_input", component_indices=[0])


# --------------------------------------------------------------------------- #
#  1. Unique default featurizers                                               #
# --------------------------------------------------------------------------- #
def test_default_featurizers_are_unique():
    comp = make_static_component()

    u1 = MU.AtomicModelUnit(component=comp)
    u2 = MU.AtomicModelUnit(component=comp)

    # distinct identity featurizers
    assert u1.featurizer is not u2.featurizer

    # mutate one — the other stays pristine
    u1.featurizer.n_features = 4
    assert u2.featurizer.n_features is None


# --------------------------------------------------------------------------- #
#  2. ComponentIndexer __repr__                                                #
# --------------------------------------------------------------------------- #
def test_componentindexer_repr():
    ci = MU.ComponentIndexer(lambda _: [0], id="idxID")
    assert "idxID" in repr(ci)


# --------------------------------------------------------------------------- #
#  3. Component equality & hashing                                             #
# --------------------------------------------------------------------------- #
def test_component_equality_hash():
    c1 = MU.Component(layer=1, component_type="block_input", indices_func=[0])
    c2 = MU.Component(layer=1, component_type="block_input", indices_func=[0])
    c3 = MU.Component(layer=2, component_type="block_input", indices_func=[0])

    assert c1 == c2
    assert hash(c1) == hash(c2)

    assert c1 != c3
    assert hash(c1) != hash(c3)


# --------------------------------------------------------------------------- #
#  4. Feature-index bounds checking                                            #
# --------------------------------------------------------------------------- #
def test_feature_bounds_ok():
    feat = F.Featurizer(n_features=4)
    comp = make_static_component()
    unit = MU.AtomicModelUnit(component=comp, featurizer=feat, feature_indices=[1, 2])
    assert unit.get_feature_indices() == [1, 2]


def test_feature_bounds_violation():
    feat = F.SubspaceFeaturizer(shape=(4, 4), trainable=False)  # n_features = 4
    comp = make_static_component()
    with pytest.raises(ValueError):
        MU.AtomicModelUnit(component=comp, featurizer=feat, feature_indices=[0, 5])


def test_feature_bounds_after_featurizer_swap():
    big_feat   = F.SubspaceFeaturizer(shape=(4, 4), trainable=False)   # n_features = 4
    small_feat = F.SubspaceFeaturizer(shape=(4, 2), trainable=False)   # n_features = 2

    comp = make_static_component()
    # index 3 is valid for the 4-dim featurizer, but invalid for the 2-dim one
    unit = MU.AtomicModelUnit(component=comp, featurizer=big_feat, feature_indices=[3])

    with pytest.raises(ValueError):
        unit.set_featurizer(small_feat)


# --------------------------------------------------------------------------- #
#  5. Static component indexing                                               #
# --------------------------------------------------------------------------- #
def test_static_component_index():
    comp = MU.StaticComponent(layer=0, component_type="block_input", component_indices=[2, 3])
    assert comp.index("ignored") == [2, 3]


# --------------------------------------------------------------------------- #
#  6. Intervention config structure                                            #
# --------------------------------------------------------------------------- #
def test_create_intervention_config():
    comp = make_static_component()
    unit = MU.AtomicModelUnit(component=comp)
    cfg = unit.create_intervention_config(group_key="grp", intervention_type="collect")

    assert cfg["component"] == "block_input"
    assert cfg["unit"] == "pos"
    assert cfg["layer"] == 0
    assert cfg["group_key"] == "grp"
    assert callable(cfg["intervention_type"])
