"""Compiled trace evaluation, copying, intervention and lazy invalidation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from causalab.causal import Dom, V, mechanism
from causalab.causal.model import CausalModel, CausalTrace, CompiledEquation


@pytest.mark.unit
def test_lazy_readiness_visits_shared_ancestors_once():
    visits = []

    class Equation:
        lazy = True
        compute = staticmethod(lambda trace: 0)

        def __init__(self, parents):
            self._parents = parents

        @property
        def parents(self):
            visits.append(self)
            assert len(visits) <= 100, "Readiness revisited shared ancestors"
            return self._parents

    names = [f"step_{i}" for i in range(41)]
    model = SimpleNamespace(
        inputs=[],
        variables=names,
        mechanisms={
            name: Equation(names[max(0, i - 2) : i]) for i, name in enumerate(names)
        },
        domains={},
        children={},
        _validators=[],
    )
    trace = CausalTrace(model)
    assert trace._values == {}


@pytest.mark.unit
def test_deep_lazy_chains_evaluate_and_invalidate_without_recursion():
    names = [f"step_{i}" for i in range(1500)]
    equations = {names[0]: CompiledEquation([], None)}
    for previous, name in zip(names, names[1:]):
        equations[name] = CompiledEquation(
            [previous], lambda trace, parent=previous: trace[parent] + 1, lazy=True
        )
    model = SimpleNamespace(
        inputs=[names[0]],
        variables=names,
        mechanisms=equations,
        domains={name: Dom(int) for name in names},
        children={name: names[i + 1 : i + 2] for i, name in enumerate(names)},
        _validators=[],
    )
    trace = CausalTrace(model, {names[0]: 0})
    assert len(trace._values) == 1
    assert trace[names[-1]] == 1499
    trace[names[0]] = 10
    assert trace[names[-1]] == 1509


@pytest.mark.unit
def test_iterative_reads_preserve_lazy_branch_selection():
    @mechanism
    def equations(enabled: Dom(bool), divisor: Dom([0, 1])):
        bad = V(1 // divisor, domain=Dom(int), lazy=True)
        result = V(bad if enabled else 5, domain=Dom(int), lazy=True)
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result), lazy=True)  # noqa: F841
        return result

    trace = CausalModel(equations).new_trace({"enabled": False, "divisor": 0})
    assert trace["raw_output"] == "5"
    assert "bad" not in trace
    trace["enabled"] = True
    with pytest.raises(ZeroDivisionError):
        trace["raw_output"]


def _ab_chain():
    @mechanism
    def equations(A: Dom([0, 1])):
        B = V(A)
        raw_input = V(str(A), domain=Dom(str))  # noqa: F841
        raw_output = V(str(B), domain=Dom(str))  # noqa: F841
        return B

    return CausalModel(equations)


def _abc_chain():
    @mechanism
    def equations(A: Dom([0, 1, 2])):
        B = V(A + 1, domain=Dom(range(100)))
        C = V(B + 1)
        raw_input = V(str(A), domain=Dom(str))  # noqa: F841
        raw_output = V(str(C), domain=Dom(str))  # noqa: F841
        return C

    return CausalModel(equations)


def _lazy_chain(multiplier=1, offset=0):
    @mechanism
    def equations(A: Dom([0, 1, 2])):
        B = V(A * multiplier + offset, lazy=True)
        raw_input = V(str(A), domain=Dom(str))  # noqa: F841
        raw_output = V(str(B), domain=Dom(str), lazy=True)  # noqa: F841
        return B

    return CausalModel(equations)


class TestCausalTraceUnit:
    """``CausalTrace``: forward-evaluation, intervention, copy, and to_dict semantics."""

    pytestmark = pytest.mark.unit

    def test_init_auto_computes_descendants(self):
        trace = CausalTrace(_ab_chain(), inputs={"A": 1})
        assert trace["A"] == 1
        assert trace["B"] == 1

    def test_get_raises_keyerror_for_uncomputed(self):
        # Construct with no inputs -> B is never computed.
        trace = CausalTrace(_ab_chain())
        with pytest.raises(KeyError):
            trace.get("B")

    def test_contains_protocol(self):
        trace = CausalTrace(_ab_chain(), inputs={"A": 0})
        assert "A" in trace
        assert "B" in trace
        assert "Z" not in trace

    def test_delitem_removes_cached_value(self):
        trace = CausalTrace(_ab_chain(), inputs={"A": 1})
        assert "B" in trace
        del trace["B"]
        assert "B" not in trace

    def test_intervene_breaks_causal_link(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        # Without intervention: A=0, B=1, C=2.
        assert trace["B"] == 1
        assert trace["C"] == 2

        # Intervene on B -> 99. C should recompute to 100.
        trace.intervene("B", 99)
        assert trace["B"] == 99
        assert trace["C"] == 100

        # Mutate A — B must remain pinned at 99 because intervene replaced
        # its mechanism with a constant.
        trace.intervene("A", 1)
        assert trace["B"] == 99
        assert trace["C"] == 100

    def test_setitem_calls_intervene(self):
        # __setitem__ has intervention semantics (per the docstring).
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace["B"] = 50
        assert trace["B"] == 50
        # Ancestor mutation must not overwrite the pinned B.
        trace.intervene("A", 2)
        assert trace["B"] == 50

    def test_copy_is_independent(self):
        original = CausalTrace(_abc_chain(), inputs={"A": 0})
        clone = original.copy()
        clone.intervene("B", 77)
        # Original is untouched.
        assert original["B"] == 1
        assert original["C"] == 2
        # Clone reflects the intervention.
        assert clone["B"] == 77
        assert clone["C"] == 78

    def test_to_dict_returns_plain_dict(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert d == {"A": 0, "B": 1, "C": 2, "raw_input": "0", "raw_output": "2"}

    def test_to_dict_returns_copy_not_alias(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        d = trace.to_dict()
        d["A"] = 999
        # Mutating the returned dict must not bleed into the trace.
        assert trace["A"] == 0

    def test_lazy_mechanism_resolved_on_access(self):
        trace = CausalTrace(_lazy_chain(offset=1), inputs={"A": 1})
        # B is lazy; not stored eagerly during init.
        assert "B" not in trace
        # First access computes it.
        assert trace["B"] == 2
        assert "B" in trace


class TestCausalTraceProperty:
    """Invariants of ``CausalTrace`` that hold across mechanism shapes."""

    pytestmark = pytest.mark.property

    def test_intervention_persists_under_ancestor_change(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace.intervene("B", 7)
        # Vary A across several values; B must stay pinned.
        for a_val in [0, 1, 2]:
            trace.intervene("A", a_val)
            assert trace["B"] == 7

    def test_copy_idempotent_round_trip(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 2})
        clone = trace.copy().copy()
        for var in ("A", "B", "C"):
            assert clone[var] == trace[var]

    def test_to_dict_matches_per_variable_get(self):
        trace = CausalTrace(_abc_chain(), inputs={"A": 1})
        d = trace.to_dict()
        for var, value in d.items():
            assert trace.get(var) == value

    def test_setitem_observationally_equals_intervene(self):
        trace_a = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace_b = CausalTrace(_abc_chain(), inputs={"A": 0})
        trace_a["B"] = 42
        trace_b.intervene("B", 42)
        assert trace_a.to_dict() == trace_b.to_dict()

    @pytest.mark.parametrize("a_value", [0, 1, 2])
    def test_descendant_recompute_terminates_on_chain(self, a_value):
        # A 3-node chain: descendant recompute is bounded by graph depth.
        trace = CausalTrace(_abc_chain(), inputs={"A": a_value})
        assert trace["A"] == a_value
        assert trace["B"] == a_value + 1
        assert trace["C"] == a_value + 2

    def test_fresh_trace_equivalent_to_post_intervention(self):
        # Forward-pass with A=1 should produce the same dict as
        # forward-pass with A=0 followed by intervene("A", 1).
        fresh = CausalTrace(_abc_chain(), inputs={"A": 1})
        intervened = CausalTrace(_abc_chain(), inputs={"A": 0})
        intervened.intervene("A", 1)
        assert fresh.to_dict() == intervened.to_dict()

    def test_lazy_value_invalidated_on_ancestor_intervene(self):
        trace = CausalTrace(_lazy_chain(multiplier=10), inputs={"A": 1})
        # Force B to materialize.
        assert trace["B"] == 10
        # Now intervene on A; lazy descendant must be invalidated.
        trace.intervene("A", 2)
        assert trace["B"] == 20
