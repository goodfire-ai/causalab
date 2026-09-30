# Defining causal models

Use Python expressions to define equations. Mark each variable that accepts an
intervention with `V`. Inputs declare their domains. The compiler infers domains
for exposed variables unless an equation supplies an explicit domain.

The [folder README](../causalab/causal/README.md) explains where the code lives.

```python
from causalab.causal import CausalModel, Dom, V, mechanism

@mechanism
def equations(x: Dom(range(10)), enabled: Dom(bool)):
    candidate = V(x + 1 if enabled else None)
    result = V(0 if candidate is None else candidate * 2)
    raw_input = V(f"x={x}, enabled={enabled}", domain=Dom(str))
    raw_output = V(str(result), domain=Dom(str))
    return result

model = CausalModel(equations, id="optional_double")
trace = model.new_trace({"x": 2, "enabled": False})
trace["candidate"] = 4
assert trace["result"] == 8
```

`candidate` always exists. `None` is an ordinary value in its inferred domain.
Its consumer handles `None` explicitly. An intervention replaces the **whole**
equation, including the condition that ordinarily makes it inactive. Other
nodes keep their own conditions. Invalid inputs, overrides, and computed values
raise errors. `intervene_many({...})` applies simultaneous overrides before
recomputation. Eager evaluation and `require` checks are atomic: a failure leaves
the original trace unchanged. Lazy descendants validate when first read, which
can happen after an intervention commits; those later errors do not roll back
the intervention.
Deleting a trace entry drops its cached value; it does not remove the variable
or undo an active intervention.
Use `trace.snapshot()` to copy currently available values and active
interventions without evaluating unused lazy equations. Use
`trace.snapshot(required=["answer"])` to additionally evaluate named values on
a copy, without changing the original trace's cache. Labeling accepts
`setting_variables=[...]` for extra values in its exported setting.
`serialize_examples` and `serialize_counterfactual_dataset` request their
selected scoring answer automatically; `extra_variables=[...]` requests other
columns on both sides of each pair. Other uncached lazy values stay omitted.
Saving uses the passive snapshot. `trace.to_dict()` explicitly materializes
every computable variable, including lazy equations, and can raise their errors.

The return aliases an exposed node; it does not add a node or prune other exposed
variables. `raw_input` and `raw_output` remain required, named graph variables.
The constructor accepts the existing scoring, embeddings, periods, print
positions, ID, and input filter. An input filter restricts observations during
sampling/enumeration and is checked before evaluation; it is not an intervention
constraint. Use `require(condition, error="...")` for an execution constraint that
also applies after interventions. On partial traces, each constraint runs as
soon as the inputs it actually reads are available. An unrelated missing input
does not disable it, and short-circuit conditions can reject invalid states
before all their potential inputs are supplied.

## Domains and dependencies

`Dom([None, False, True])` is an exact finite set. `Dom(range(N))` is a compact
integer domain. `Dom(str)` validates arbitrary text. `Dom.sequence(element,
length=N, container=list)` validates a fixed sequence; `max_length=N` validates
lengths from zero to N. A type domain need not be enumerable or sampleable.
An empty sequence has one possible value even when its element domain is open.
Compact ranges support integer lengths beyond the platform's `len()` limit,
including `range(2**64)`, without materialization for counting or sampling.
`model.values` retains small finite domain lists, compact `range` objects for
large integer domains, or `None` for non-enumerable domains. Shared consumers use
`model.domains[name]`: `enumerated(limit)` returns values or `None`,
`require_enumerated(limit)` raises if exhaustive traversal would exceed the
bound, `cardinality(limit=...)` counts without materializing ranges and saturates
at `limit + 1`, and `sample(rng)` draws a value when sampling is supported.
Unknown cardinalities return `None`; validation is always available.
A finite union also saturates at `limit + 1` once a member or their distinct
combined values exceed the cap; exceeding a cap is not an unknown cardinality.
Explicit finite entries are deduplicated with the same value comparison used
for membership. Finite unions within the enumeration bound support sampling.

Finite domains preserve nested Python types, dictionary insertion order, and
floating-point representations, including signed zero and NaN payloads. The
same comparison governs membership and inference: `(True,)` and `(1,)` are
different causal values. Supported finite values include scalar values, plain
acyclic lists/tuples/dictionaries, and numeric NumPy values. Sets, custom objects,
and cyclic containers require an explicit type domain such as `Dom(set)`;
their equality cannot justify an exhaustive branch proof. Compact integer
domains accept exact Python integers, and sequence domains accept the declared
list or tuple type, without subclasses.
NumPy comparisons retain both the dtype representation and its scalar class:
`int64` and `longlong` remain distinct on platforms where indexing produces
different scalar types.

Inference checks up to 4096 combinations of parent values, including allowed
interventions. For larger domains it uses structural rules for text formatting
and conditional branches.
Identity and membership comparisons produce Boolean values. Other comparisons
can return custom types, such as `numpy.bool_`. Supply an explicit result domain
when their inputs exceed the inference bound, for example
`V(x > 0, domain=Dom(numpy.bool_))` for a NumPy scalar.
An explicit domain is also the allowed intervention domain: declare all group
positions if a positional node ordinarily equals just one index.

NaN-containing values need special care: Python container membership and
equality can depend on whether the exact same NaN object is reused. Finite
representatives with NaNs are therefore not treated as exhaustive branch or
output proofs. Structural inference still applies; opaque NaN-dependent helpers
may require an explicit result domain. Conditional reads require a valid
witness, and unresolved reachability remains a definition error.

Edges represent potential **reads**, including the condition, either branch,
and each permitted family selection. A read in one possible execution is enough.
Build-time branches and impossible selections are removed. A conditional edge must have an allowed-domain witness. Small complete
guard-domain evaluations can prove a read unreachable; a construction that
cannot establish either reachability or absence asks you to simplify its gate
or use a smaller finite gating domain. Witness search establishes the existence
of an edge and is never used to infer a value domain. Pure helper arguments are
read at the call boundary; pass only the values the helper uses. A helper that
receives an entire family therefore depends on every family member. This is a
syntactic read graph, not a proof that changing a parent changes the output
(e.g. `x * 0` still reads x).

## Fixed families and steps

```python
from causalab.causal import FamilyDom, family, require

@mechanism
def equations(inputs: FamilyDom(Dom(range(4)), size=2)):
    steps = family(size=4, domain=Dom(range(7)))
    steps[0] = inputs[0] + inputs[1]
    for t in range(3):
        steps[t + 1] = max(0, steps[t] - 1)
    require(steps[3] == 0, error="step bound exceeded")
    raw_input = V(str(inputs), domain=Dom(str))
    raw_output = V(str(steps[3]), domain=Dom(str))
    return steps[3]
```

A family write defines an exposed equation without another `V`. The graph has
`inputs[0]`, `inputs[1]`, `steps[0]` through `steps[3]`, and the two rendering nodes.
Every member must be assigned on every successful path. For a grid use fixed
`keys=[(g, r), ...]`; canonical names are `entities[0,1]`. `FamilyDom({key: Dom(...),
...})` allows different domains per input member; pass it as `family(...,
domain=...)` for per-member computed domains.
Scalar and singleton tuple keys stay distinct: `0` names `items[0]`, `(0,)`
names `items[0,]`, and `()` names `items[()]`. Canonical names must be unique.

Loop bounds and index sets are fixed construction parameters (at most 10000
expanded iterations in total, including nested loops and comprehensions).
Expanded expressions are also limited to 100000 AST visits and depth 200;
exceeding a bound raises a source-located definition error. A runtime condition can update or carry each
step's state, but cannot change the graph size. `while`, `break`, early returns,
and mutation inside the equations are rejected with guidance. Container methods
such as `pop` and `update` belong inside a pure helper that creates its own
working container. Ordinary pure
helpers can contain private algorithms. List/set comprehensions over fixed
collections are supported, with Python's filter order. Each reached iteration
validates its unpacking target before its own filters, even if some or all
targets are unused. Earlier filters still short-circuit later iterations.
Generator expressions
and dictionary comprehensions belong in ordinary pure helpers. In particular,
`any`, `all`, and `next` over generators retain Python's lazy evaluation inside
helpers; generators in equations are rejected instead of being made eager.
Stateful `iter()`/`next()` algorithms also belong inside helpers; direct calls
and their aliases in equations are rejected.
Callable lookup recognizes names and attributes without executing a factory or
descriptor during syntax recognition. Put dynamic callable selection in a
helper. `new_trace` accepts either a family container, such as
`{"inputs": [1, 2]}`, or canonical member keys.

Private named assignments reuse their initializer's value within each consuming
equation evaluation. Tuple/list targets use Python iterable unpacking, including
dictionary keys and exact nested arity checks, and reuse the unpacked values.
The whole right-hand side contributes reads, even
when only one target is used. These private bindings do not add graph variables;
they are recomputed for each equation evaluation, including after interventions.
Starred unpacking belongs inside a pure helper.

## Inferred helper namespaces

```python
from causalab.causal import submodel

@submodel
def compare(a, b):
    equal = V(a == b)
    return equal

@mechanism
def equations(a: Dom(range(3)), b: Dom(range(3)), c: Dom(range(3))):
    left = compare(a, b)
    right = compare(b, c)
    result = V(left == right)
    raw_input = V(f"{a},{b},{c}", domain=Dom(str))
    raw_output = V(str(result), domain=Dom(str))
    return result
```

The nodes are `a`, `b`, `c`, `left.equal`, `right.equal`, `result`, and the rendering
nodes. `left` is a Python alias for `left.equal`; there is no `left` node and no
copy of the helper's input parameters. Nested bindings compose namespaces. A
submodel call needs a fresh named local binding. Ordinary undecorated helpers
remain opaque within their caller's equation.

## Randomness and notebooks

Declare noise as `seed: Exo(Dom(range(2**32)))`, then pass it to a deterministic
helper that constructs a local `random.Random(seed)` or equivalent. Mechanisms
must be deterministic given explicit inputs and fixed configuration. Do not use
module RNG state, I/O, clocks, or mutable external state in a helper. Direct
module-RNG calls in equations are rejected; arbitrary Python helpers are a
purity contract, not a sandbox.

`model.sample_input(rng=random.Random(42))` samples explicit root values, including
noise. A copied/intervened trace keeps that noise fixed. Base and donor sampling
can draw independently. Direct execution supplies `noise={"seed": 7}` (or the
seed among input values); stochastic enumeration supplies the same fixed noise
for every enumerated input. `n_unique_inputs` applies to deterministic models;
use `count_inputs(noise={...})` for stochastic ones. `count_inputs(limit=N)` stops
at `N + 1` accepted inputs and does not compute their outputs. It streams large
ranges and bounded sequences through the input filter. Capped dataset
splits use this to decide between enumeration and sampling.

The same definitions work in Python files and ordinary Jupyter/IPython cells.
The decorator reads cached cell source, and the explicit constructor snapshots
referenced configuration and Python helper closures at the same boundary.
Rebinding a referenced global or closure variable between decoration and
construction affects both direct expressions and helpers. Existing models
retain their snapshots; constructing another model captures the new bindings.
Annotation-only factory locals are resolved in the definition's original
scope, as with ordinary Python annotations. Redefining a cell produces
a new definition; an already constructed model keeps its existing equations and
configuration. Dynamically `exec`-generated source without a source cache is
rejected with a source-recovery error. Standard-library/compiler support does
not import a neural runtime.

Supported configuration includes plain dictionaries/lists/tuples/sets,
dataclass attributes, `SimpleNamespace` attributes, numeric NumPy values/arrays,
and Python helpers (including their referenced globals, defaults, closure
cells, function attributes, and annotations). Functions stored inside these
containers receive the same capture. NumPy dtype metadata, including metadata
on structured fields and subarrays, is copied recursively. Cyclic NumPy metadata
is rejected.
Dataclass capture preserves its class and stored dictionary/slot values without
reapplying descriptor setters. Helpers must reference configuration explicitly;
dynamic access to their global namespace is rejected because it cannot be
captured from their declared reads.
Other object shapes and object arrays are rejected with guidance. Modules,
types, and class methods are code and must obey the purity contract: changing
module/class state is outside configuration snapshotting. Compiled definitions retain
source/signatures and referenced configuration, without the original equation
function's entire global namespace. Public metadata and trace values have
separate ownership from compiled configuration.

Equations and helpers operate on values, not object identity or storage details.
Do not inspect object addresses, alias identity, or NumPy memory layout/flags to
choose an outcome. Direct `id()` calls and identity comparisons between ordinary
values are rejected. Identity checks against `None`, Booleans, and types remain
supported, including `candidate is None` and `type(candidate) is int`.

`save_counterfactual_examples` writes version-2 trace records. They preserve
supported scalar/container types, dictionary order, nonfinite floating-point
bits, numeric NumPy scalar classes and array dtypes/shapes, shared objects within
acyclic values (including repeated NaNs), and active interventions. Unsupported
objects, cyclic values, and NumPy object/structured/metadata-bearing values
cannot be saved in this format. It does not force unused lazy equations.
On load, the model rebuilds computed values from its inputs and interventions;
the usual trace ownership rules still apply. Version-1 records remain readable.
Older plain JSON dictionaries load as observations, with tuple inputs restored
from their domains while preserving dictionary order. New version-2 records
require an updated reader.

Unused Python locals marked `V(...)` still define graph nodes. Use a local
`# noqa: F841` on those assignments when linting; ordinary unused variables
elsewhere remain checked.

See [all migrated examples](../demos/causal_models/README.md). There is no compatibility
adapter for the removed dictionary constructor.
