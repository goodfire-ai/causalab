"""Define and check the allowed values of causal variables."""

from __future__ import annotations

import copy
import itertools
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class DomainError(ValueError):
    """An input, intervention, or computed value is outside its declared domain."""


class FiniteValueError(ValueError):
    """A value has no supported, exhaustive finite-domain representation."""


def _range_size(values):
    """Range length as a Python integer, without the platform-sized len limit."""
    distance = (
        values.stop - values.start if values.step > 0 else values.start - values.stop
    )
    step = abs(values.step)
    return max(0, (distance + step - 1) // step)


class Dom:
    """An explicit finite domain, compact integer range, or validated Python type.

    ``Dom([None, False, True])`` admits exactly those values. ``Dom(str)``
    validates text without pretending to enumerate every possible string.
    """

    def __init__(self, values):
        self.kind = "type" if isinstance(values, type) else "finite"
        self.type = values if self.kind == "type" else None
        if self.kind == "finite":
            if isinstance(values, range):
                self.values = values
            else:
                unique = {}
                for value in values:
                    unique.setdefault(_value_key(value), value)
                self.values = copy.deepcopy(tuple(unique.values()))
            if not self.values:
                raise ValueError("A domain must contain at least one value")
        else:
            self.values = (False, True) if values is bool else None

    @classmethod
    def integer_range(cls, start: int, stop: int) -> Dom:
        return cls(range(start, stop))

    @classmethod
    def sequence(
        cls, element: Dom, *, length=None, max_length=None, container=tuple
    ) -> Dom:
        if container not in (tuple, list):
            raise ValueError("Sequence domains support tuple or list")
        if length is None and max_length is None:
            raise ValueError("A sequence domain needs a finite length bound")
        maximum = length if length is not None else max_length
        if not isinstance(maximum, int) or maximum < 0:
            raise ValueError("Sequence length bounds must be nonnegative integers")
        result = cls(container)
        result.kind = "sequence"
        result.element = element
        result.minimum = maximum if length is not None else 0
        result.maximum = maximum
        return result

    @classmethod
    def union(cls, *domains: Dom) -> Dom:
        values = []
        seen = set()
        for domain in domains:
            finite = domain.enumerated(4096)
            if finite is None:
                result = cls(object)
                result.kind = "union"
                result.members = tuple(domains)
                return result
            for value in finite:
                key = _value_key(value)
                if key not in seen:
                    seen.add(key)
                    values.append(value)
        return cls(values)

    @property
    def is_finite(self) -> bool:
        """Whether all values can be enumerated, regardless of materialization limits."""
        if self.kind == "sequence":
            return self.maximum == 0 or self.element.is_finite
        if self.kind == "union":
            return all(member.is_finite for member in self.members)
        return self.values is not None

    def contains(self, value: Any) -> bool:
        if self.kind == "sequence":
            return (
                type(value) is self.type
                and self.minimum <= len(value) <= self.maximum
                and all(self.element.contains(item) for item in value)
            )
        if self.kind == "union":
            return any(member.contains(value) for member in self.members)
        if self.kind == "type":
            return isinstance(value, self.type) and not (
                self.type is int and isinstance(value, bool)
            )
        if isinstance(self.values, range):
            return type(value) is int and value in self.values
        try:
            return any(_equal(value, candidate) for candidate in self.values)
        except FiniteValueError:
            return False

    def validate(self, value: Any, variable: str) -> None:
        if not self.contains(value):
            raise DomainError(f"Variable {variable!r}: {value!r} is outside {self!r}")

    def enumerated(self, limit: int = 65536):
        if self.kind == "sequence":
            if self.maximum == 0:
                return (self.type(),) if limit >= 1 else None
            elements = self.element.enumerated(limit)
            if (
                elements is None
                or sum(
                    len(elements) ** n for n in range(self.minimum, self.maximum + 1)
                )
                > limit
            ):
                return None
            return tuple(
                self.type(items)
                for n in range(self.minimum, self.maximum + 1)
                for items in itertools.product(elements, repeat=n)
            )
        if self.kind == "union":
            merged = []
            seen = set()
            for member in self.members:
                values = member.enumerated(limit)
                if values is None:
                    return None
                for value in values:
                    key = _value_key(value)
                    if key not in seen:
                        seen.add(key)
                        merged.append(value)
                        if len(merged) > limit:
                            return None
            return tuple(merged)
        if isinstance(self.values, range):
            return self.values if _range_size(self.values) <= limit else None
        if self.values is None or len(self.values) > limit:
            return None
        return self.values

    def public_values(self):
        if isinstance(self.values, range):
            return (
                self.values if _range_size(self.values) > 65536 else list(self.values)
            )
        values = self.enumerated()
        return None if values is None else copy.deepcopy(list(values))

    def iter_values(self):
        """Yield finite values as needed, including values from compact ranges."""
        if self.kind == "sequence":
            for length in range(self.minimum, self.maximum + 1):
                for items in iter_combinations([self.element] * length):
                    yield self.type(items)
        elif self.kind == "union":
            seen = set()
            for member in self.members:
                for value in member.iter_values():
                    key = _value_key(value)
                    if key not in seen:
                        seen.add(key)
                        yield value
        else:
            if self.values is None:
                raise ValueError(f"{self!r} is not finitely enumerable; use sampling")
            for value in self.values:
                yield copy.deepcopy(value)

    def cardinality(self, *, limit=None):
        """Exact finite size, or None if unknown; optionally saturate at limit+1.

        Compact ranges never need materializing. A limit also bounds arithmetic
        for large sequence domains. Non-enumerable unions have unknown size
        because their members may overlap.
        """
        if self.kind == "sequence":
            if self.maximum == 0:
                return 1 if limit is None else min(1, limit + 1)
            size = self.element.cardinality(limit=limit)
            if size is None:
                return None
            total = 0
            for length in range(self.minimum, self.maximum + 1):
                if limit is not None and size > 1 and length > limit.bit_length():
                    return limit + 1
                total += size**length
                if limit is not None and total > limit:
                    return limit + 1
            return total
        if self.kind == "union" and limit is not None:
            sizes = [member.cardinality(limit=limit) for member in self.members]
            # A member is a lower bound on its union, regardless of overlap.
            # Exceeding the cap is a known count, not failed enumeration.
            if any(size is not None and size > limit for size in sizes):
                return limit + 1
            if any(size is None for size in sizes):
                return None
            values = self.enumerated(limit + 1)
            return limit + 1 if values is None else min(len(values), limit + 1)
        values = (
            self.enumerated(limit if limit is not None else 65536)
            if self.kind == "union"
            else self.values
        )
        if values is None:
            return None
        size = _range_size(values) if isinstance(values, range) else len(values)
        return size if limit is None else min(size, limit + 1)

    def require_enumerated(self, limit=65536):
        """Bounded values for consumers that require an exhaustive traversal."""
        values = self.enumerated(limit)
        if values is None:
            raise ValueError(
                f"{self!r} cannot be enumerated within {limit} values; "
                "supply explicit candidates or use sampling"
            )
        return values

    def sample(self, rng):
        if self.kind == "sequence":
            count = rng.randint(self.minimum, self.maximum)
            return self.type(self.element.sample(rng) for _ in range(count))
        if self.kind == "union":
            values = self.enumerated()
            if values is not None:
                return copy.deepcopy(rng.choice(values))
            raise ValueError(
                "Supply an explicit value for a union exceeding the sampling enumeration bound"
            )
        if self.values is None:
            raise ValueError(f"Cannot sample {self!r}; supply the input explicitly")
        if isinstance(self.values, range):
            return self.values.start + self.values.step * rng.randrange(
                _range_size(self.values)
            )
        return copy.deepcopy(rng.choice(self.values))

    def __repr__(self):
        if self.kind == "sequence":
            return f"Dom.sequence({self.element!r}, lengths={self.minimum}..{self.maximum}, container={self.type.__name__})"
        if self.kind == "union":
            return f"Dom.union({', '.join(map(repr, self.members))})"
        if self.kind == "type":
            return f"Dom({self.type.__name__})"
        if isinstance(self.values, range) or len(self.values) <= 12:
            return f"Dom({self.values!r})"
        return f"Dom({len(self.values)} values)"


def _equal(a, b):
    """Equality for exhaustive domain proofs, including nested type distinctions.

    Python's ``==`` loses distinctions that equations can observe: nested
    bools/ints, dictionary order, signed zero, and NaN payloads. Unsupported
    values must fail explicitly rather than provide an unsound equality proof.
    This is value semantics; object identity and NumPy storage layout are not
    part of a causal value.
    """
    return _value_key(a) == _value_key(b)


def _nan_atoms(value):
    """Find non-reflexive numeric values inside a supported finite value."""
    kind = type(value)
    if kind is float and math.isnan(value):
        yield value
    elif kind is complex and (math.isnan(value.real) or math.isnan(value.imag)):
        yield value
    elif kind in (tuple, list):
        for item in value:
            yield from _nan_atoms(item)
    elif kind is dict:
        for key, item in value.items():
            yield from _nan_atoms(key)
            yield from _nan_atoms(item)
    elif kind is slice:
        for item in (value.start, value.stop, value.step):
            yield from _nan_atoms(item)
    elif kind.__module__.startswith("numpy"):
        import numpy as np

        if (
            (kind is np.ndarray or isinstance(value, np.generic))
            and value.dtype.kind in "fc"
            and np.isnan(value).any()
        ):
            yield value


def _contains_nan(value):
    return any(True for _ in _nan_atoms(value))


def _fresh_nan(value):
    """Copy a NaN-bearing numeric atom without changing its bits."""
    if type(value) is float:
        return struct.unpack("!d", struct.pack("!d", value))[0]
    if type(value) is complex:
        return complex(value.real, value.imag)
    return value.copy()


def _nan_variants(value, pool, limit):
    """Bounded identity variants for witnesses, never an exhaustive value set.

    Python containers compare an identical element before calling equality.
    NaNs therefore need shared and fresh identities when searching for a read.
    """
    kind = type(value)
    if not _contains_nan(value):
        yield value
    elif kind in (tuple, list, dict, slice):
        if kind is dict:
            items = [part for pair in value.items() for part in pair]
        elif kind is slice:
            items = [value.start, value.stop, value.step]
        else:
            items = value
        choices = [list(_nan_variants(item, pool, limit)) for item in items]
        for parts in itertools.islice(itertools.product(*choices), limit):
            if kind is dict:
                yield dict(zip(parts[::2], parts[1::2]))
            elif kind is slice:
                yield slice(*parts)
            else:
                yield kind(parts)
    else:
        for atom in pool:
            if _equal(value, atom):
                yield atom
        yield _fresh_nan(value)


def _value_key(value, active=None):
    """An exact key for supported finite values; never invoke user equality."""
    kind = type(value)
    if kind in (
        type(None),
        bool,
        int,
        str,
        bytes,
        type(Ellipsis),
        type(NotImplemented),
    ):
        return kind, value
    if kind is float:
        return kind, struct.pack("!d", value)
    if kind is complex:
        return kind, struct.pack("!dd", value.real, value.imag)
    if kind is range:
        return kind, value.start, value.stop, value.step
    if kind is slice:
        return kind, tuple(
            _value_key(v, active) for v in (value.start, value.stop, value.step)
        )

    if kind in (tuple, list, dict):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise FiniteValueError(
                "Cyclic containers cannot be finite domain values; "
                "use an explicit type domain such as Dom(list)"
            )
        active.add(identity)
        try:
            if kind is dict:
                contents = tuple(
                    (
                        _value_key(key, active),
                        _value_key(item, active),
                    )
                    for key, item in value.items()
                )
            else:
                contents = tuple(_value_key(item, active) for item in value)
            return kind, contents
        finally:
            active.remove(identity)

    if kind.__module__.startswith("numpy"):
        # Keep ordinary model definitions independent of importing NumPy.
        import numpy as np

        if kind is np.ndarray or isinstance(value, np.generic):
            dtype = value.dtype
            if (
                dtype.kind in "biufc"
                and dtype.metadata is None
                and (dtype.kind != "f" or dtype.itemsize in (2, 4, 8))
                and (dtype.kind != "c" or dtype.itemsize in (8, 16))
            ):
                return (
                    kind,
                    dtype.type,
                    dtype.str,
                    value.shape,
                    value.tobytes(order="C"),
                )

    raise FiniteValueError(
        f"Unsupported finite domain value of type {kind.__name__}; "
        "use plain acyclic lists/tuples/dicts and scalar values, "
        f"or an explicit type domain Dom({kind.__name__})"
    )


def iter_combinations(domains):
    """Yield a product of domains without copying their full value sets."""
    domains = tuple(domains)
    if not domains:
        yield ()
        return
    iterators = [domains[0].iter_values()]
    prefix = []
    while iterators:
        try:
            value = next(iterators[-1])
        except StopIteration:
            iterators.pop()
            if prefix:
                prefix.pop()
            continue
        if len(iterators) == len(domains):
            yield (*prefix, value)
        else:
            prefix.append(value)
            iterators.append(domains[len(iterators)].iter_values())


@dataclass(frozen=True)
class FamilyDom:
    """Input family with a fixed index set and a domain for each member."""

    domains: dict

    def __init__(self, domain, *, size=None):
        if size is not None:
            if not isinstance(size, int) or size < 0 or not isinstance(domain, Dom):
                raise ValueError("FamilyDom needs a Dom and a nonnegative integer size")
            domains = dict.fromkeys(range(size), domain)
        elif isinstance(domain, Mapping) and all(
            isinstance(v, Dom) for v in domain.values()
        ):
            domains = dict(domain)
        else:
            raise TypeError(
                "Use FamilyDom(domain, size=N) or FamilyDom({index: domain})"
            )
        object.__setattr__(self, "domains", domains)


@dataclass(frozen=True)
class Exo:
    """An explicit exogenous input, retained unchanged during an intervention."""

    domain: Dom | FamilyDom
