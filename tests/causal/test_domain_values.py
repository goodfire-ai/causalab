"""Finite domains preserve every supported value distinction used in proofs."""

import copy
import math
import random
import struct

import pytest

from causalab.causal import Dom, DomainError
from causalab.causal.domains import FiniteValueError, _equal

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "first, second",
    [
        (True, 1),
        ((True,), (1,)),
        ([1], [1.0]),
        ({"nested": [True]}, {"nested": [1]}),
        ({True: "value"}, {1: "value"}),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        (range(0, 3, 2), range(0, 4, 2)),
        (0.0, -0.0),
        (complex(0.0, 0.0), complex(-0.0, 0.0)),
        (complex(0.0, 0.0), complex(0.0, -0.0)),
    ],
)
def test_finite_membership_and_union_keep_observable_distinctions(first, second):
    domain = Dom([first])
    assert domain.contains(copy.deepcopy(first))
    assert not domain.contains(second)
    with pytest.raises(DomainError):
        domain.validate(second, "input")
    merged = Dom.union(domain, Dom([second]))
    assert merged.cardinality() == 2
    assert merged.contains(first)
    assert merged.contains(second)


def _nan(bits):
    return struct.unpack("!d", struct.pack("!Q", bits))[0]


def test_nan_values_are_reflexive_and_retain_payload_and_sign():
    values = [
        _nan(0x7FF8000000000001),
        _nan(0x7FF8000000000002),
        _nan(0xFFF8000000000001),
    ]
    assert all(math.isnan(value) for value in values)
    domains = [Dom([value]) for value in values]
    for i, domain in enumerate(domains):
        assert [domain.contains(value) for value in values] == [
            j == i for j in range(len(values))
        ]
    assert Dom.union(*domains).cardinality() == 3
    complex_domain = Dom([complex(values[0], values[1])])
    assert complex_domain.contains(complex(values[0], values[1]))
    assert not complex_domain.contains(complex(values[1], values[0]))


def test_range_and_sequence_membership_match_their_enumerated_types():
    class Integer(int):
        pass

    class List(list):
        pass

    assert not Dom(range(3)).contains(Integer(1))
    assert not Dom(range(3)).contains(True)
    domain = Dom.sequence(Dom(range(3)), length=1, container=list)
    assert domain.contains([1])
    assert not domain.contains(List([1]))
    assert not domain.contains([Integer(1)])
    assert all(domain.contains(value) for value in domain.require_enumerated())


@pytest.mark.parametrize("value", [{1}, frozenset({1})])
def test_unordered_containers_require_an_explicit_type_domain(value):
    with pytest.raises(FiniteValueError, match="explicit type domain"):
        Dom([value])
    assert Dom(type(value)).contains(value)
    assert not Dom([[]]).contains([value])


def test_custom_equality_never_becomes_a_finite_domain_proof():
    class AlwaysEqual:
        def __eq__(self, other):
            raise AssertionError("Finite domains must not call custom equality")

    value = AlwaysEqual()
    with pytest.raises(FiniteValueError, match="AlwaysEqual"):
        Dom([value])
    with pytest.raises(FiniteValueError, match="AlwaysEqual"):
        _equal(value, value)
    assert Dom(AlwaysEqual).contains(value)


@pytest.mark.parametrize("through_slice", [False, True])
def test_cyclic_containers_are_rejected_with_type_domain_guidance(through_slice):
    value = []
    value.append(slice(value) if through_slice else value)
    with pytest.raises(FiniteValueError, match="Cyclic containers"):
        Dom([value])
    assert Dom(list).contains(value)


def test_shared_acyclic_containers_remain_valid():
    item = [1]
    value = [item, item]
    domain = Dom([value])
    assert domain.contains(copy.deepcopy(value))
    assert domain.contains([[1], [1]])
    assert domain.contains(domain.sample(random.Random(0)))
    sequence = Dom.sequence(Dom([[1]]), length=2)
    assert Dom.union(sequence).contains(([1], [1]))


def test_numpy_numeric_values_preserve_dtype_shape_and_exact_elements():
    np = pytest.importorskip("numpy")
    values = [
        np.int32(1),
        np.int64(1),
        np.float32(0.0),
        np.float32(-0.0),
        np.array([1, 2], dtype="int32"),
        np.array([1, 2], dtype="int64"),
        np.array([[1, 2]], dtype="int32"),
        np.array([0.0, -0.0]),
        np.array([-0.0, 0.0]),
        np.array([_nan(0x7FF8000000000001)]),
        np.array([_nan(0x7FF8000000000002)]),
    ]
    for i, value in enumerate(values):
        domain = Dom([value])
        assert domain.contains(copy.deepcopy(value))
        assert [domain.contains(other) for other in values] == [
            j == i for j in range(len(values))
        ]
    assert Dom.union(*(Dom([value]) for value in values)).cardinality() == len(values)


def test_numpy_object_and_structured_arrays_require_type_domains():
    np = pytest.importorskip("numpy")
    for value in (
        np.array([object()], dtype=object),
        np.array([(1,)], dtype=[("field", "int32")]),
    ):
        with pytest.raises(FiniteValueError, match="explicit type domain"):
            Dom([value])
        assert Dom(np.ndarray).contains(value)
