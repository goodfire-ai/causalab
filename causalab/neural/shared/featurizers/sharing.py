"""Share featurizer forwards within one cache scope.

``_once`` caches values that need no gradient. ``_Shared`` and ``_Replay``
reuse a trained value's forward and replay backward for each gradient use.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Iterator, Sequence, TYPE_CHECKING, cast

import torch


if TYPE_CHECKING:
    from causalab.neural.shared.featurizers.stages import Cayley, Stage


class _Scope(threading.local):
    """The open [`featurizer_cache`][] scopes of **this thread**: how many
    are nested, and their shared store — ``None`` outside any scope. Keyed
    by the stage module itself, not its ``id``, so an entry keeps its stage
    alive for the scope and a stage built inside the scope can never inherit
    a freed one's address. Thread-local because a scope belongs to the
    control flow that opened it: the ranks of a simulated world run as
    threads of one process (``tests/_helpers/simulated_world``), each its
    own fit, and a store shared between them would keep one rank's rotation
    across the sync that moved its parameter, or count another rank's open
    scope as its own."""

    depth: int
    entries: dict[tuple[Any, ...], Any] | None

    def __init__(self) -> None:
        self.depth = 0
        self.entries = None


_SCOPE = _Scope()


@contextlib.contextmanager
def featurizer_cache(*, isolated: bool = False) -> Iterator[None]:
    """Evaluate each stage's derived quantity — a ``subspace``'s rotation
    ``Q``, a ``gate``'s mask — **once** for the block where that changes no
    bit of any result, forward or backward.

    A trained stage is a function of its free parameter, recomputed on every
    access: an orthogonal parametrization on every ``.weight`` read, a gate's
    ``σ(θ/T)`` on every ``featurize``. One optimizer step accesses a member's
    featurizer several times over — its read, and featurize *and* inverse on
    each write, on every hooked layer — and every evaluation is a run of tiny
    kernels (the Cayley map: a k×k inverse and a handful of k×k products) on
    a step loop that is host-bound. Profiled on the Qwen3.6-35B-A3B workflow
    in ``benchmarks/cuda_graphs/standard.json``, the ``cayley`` map issued more launches per DAS step than the whole MoE
    forward. The train loop opens one scope around a step's eager grad
    forwards and loss, and one around an eval round (``train.py``); a CUDA
    graph capture opens one per captured pass (``cuda_graphs.Replay``).

    **What is shared, and why it is exact.** Under ``no_grad`` — the eval
    passes, a CUDA-graph executor's frozen-input pass — every stage's value
    is computed once and reused: the same computation, its result read
    several times. Under grad the value is the same but the *gradient* is
    not, if one tensor serves several consumers: autograd then sums the
    consumers' cotangents at the shared tensor and runs the map's backward
    once — ``Jᵀ(Σcᵢ)`` in place of ``Σ Jᵀcᵢ`` — and a gate's cast to the
    activation's dtype would even accumulate its consumers' bf16 cotangents
    in bf16. Roundoff, but a bf16 model turns an ulp into a flipped rounding
    and thirty Adam steps into a different fit. So under grad a trained
    quantity shares its **forward alone** (`_Shared`): every access
    returns its own autograd node over the one cached value, and that node's
    backward replays the quantity's backward for *its* cotangent through the
    cached graph and hands the parameter the same contributions, in the same
    order, that a fresh graph per access would have — so the parameter's
    gradient accumulates in the exact order it did before. That replay is
    exact when the contributions can be told apart: the ``cayley`` map is
    evaluated on two aliases of its parameter, one per place it enters; a
    gate's mask shares when its graph reaches ``theta`` by exactly one edge
    and no other trainable leaf (a ``leak`` adds a second edge, a pool other
    gates' ``theta`` — those recompute per grad access, and share as a value
    under ``no_grad`` like everything else). The forward launches are what
    the profile counted; the backward's are unchanged.

    A mode-dependent quantity's key carries the stage's mode — a training
    mask and an eval mask never share an entry; the rotation, a function of
    the parameter alone, is keyed without it. Scopes nest; the store is
    dropped when the outermost exits, so nothing from one step reaches the
    next — a scope is
    valid between two mutations of its stages' parameters, and the loop
    opens each one after the optimizer step, the projection, the anneal and
    the mode switch. An ``isolated`` scope does not nest: it runs on a fresh
    store and restores the enclosing one afterwards, so a captured pass
    evaluates every stage inside the capture — a value shared in from the
    warmup pass, or from eager code around the capture, would be baked into
    the graph and replayed stale. `torch.nn.utils.parametrize.cached`
    is the naive version of this for parametrized weights: one tensor for
    every consumer, the summed-cotangent gradient above.
    """
    if isolated:
        saved = (_SCOPE.depth, _SCOPE.entries)
        _SCOPE.depth, _SCOPE.entries = 1, {}
        try:
            yield
        finally:
            _SCOPE.depth, _SCOPE.entries = saved
        return
    if _SCOPE.entries is None:
        _SCOPE.entries = {}
    _SCOPE.depth += 1
    try:
        yield
    finally:
        _SCOPE.depth -= 1
        if _SCOPE.depth == 0:
            _SCOPE.entries = None


def _once(
    stage: "Stage",
    tag: Any,
    compute: Callable[[], torch.Tensor],
    *,
    trainable: bool,
) -> torch.Tensor:
    """``compute()`` once per open [`featurizer_cache`][] scope for this
    stage, mode and ``tag`` — for the accesses that want no graph. An access
    under grad of a ``trainable`` quantity (one that depends on a parameter
    with ``requires_grad``) computes its own, every time: sharing it would
    change the gradient ([`featurizer_cache`][])."""
    cache = _SCOPE.entries
    if cache is None or (trainable and torch.is_grad_enabled()):
        return compute()
    key = (stage, tag, stage.training)
    value = cache.get(key)
    if value is None:
        value = compute()
        cache[key] = value
    return value


def _leaf_edges(output: torch.Tensor, leaf: torch.Tensor) -> tuple[int, bool]:
    """How ``output``'s graph reaches ``leaf``: the number of edges into its
    accumulator (one per op that consumed the parameter directly), and
    whether any *other* trainable leaf is reached — the two facts that decide
    whether one node per access can replay the graph's backward exactly
    (`_Shared`)."""
    edges = 0
    others = False
    # the wrappers `next_functions` hands out are fresh objects; the visited
    # set keeps them alive so an id is never recycled onto an unvisited node
    seen: dict[int, Any] = {}
    pending = [output.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = node
        for next_node, _ in node.next_functions:
            if next_node is None:
                continue
            variable = getattr(next_node, "variable", None)
            if variable is None:
                pending.append(next_node)
            elif variable is leaf:
                edges += 1
            else:
                others = True
    return edges, others


class _Shared:
    """One evaluation of a trained stage's quantity for the scope, serving
    every access exactly ([`featurizer_cache`][]).

    ``graph`` is the quantity with its autograd graph; ``aliases`` the leaves
    a replay differentiates with respect to, one per contribution the
    parameter receives from one evaluation, in the order the standalone
    graph's backward delivers them. A no-grad access gets the detached value.
    A grad access gets a fresh `_Replay` node over that value whose
    backward pushes the access's own cotangent through this graph
    (``retain_graph``) and returns those contributions to the parameter one
    after the other — so the parameter's accumulated gradient is bit for bit
    the one a graph per access produces. No aliases means the quantity does
    not depend on the parameter (a frozen mask): the value serves everyone.

    The graph is built under ``enable_grad`` whatever the mode of the access
    that builds it, so a no-grad access reads a value produced in grad mode
    and an eval round keeps one small graph per stage until the scope
    closes. Neither changes a number: none of these ops (``inv_ex``, the
    k×k products, a sigmoid, ``repeat_interleave``) picks its kernel by grad
    mode — unlike a scalar-vs-tensor divide, which this file does mind."""

    def __init__(
        self,
        original: torch.Tensor,
        graph: torch.Tensor,
        aliases: Sequence[torch.Tensor],
    ) -> None:
        self.original = original
        self.graph = graph
        self.aliases = tuple(aliases)
        self.value = graph.detach()

    @classmethod
    def rotation(cls, cayley: "Cayley", original: torch.Tensor) -> "_Shared":
        """The Cayley map evaluated on two aliases of ``X`` — where it enters
        the frame coordinates ``Q₀ᵀX`` and where it enters the complement
        ``X − Q₀(Q₀ᵀX)`` ([`Cayley.map`][]); the standalone backward
        delivers the complement's contribution first, then the frame's."""
        frame = original.detach().requires_grad_(True)
        complement = original.detach().requires_grad_(True)
        with torch.enable_grad():
            graph = cayley.map(frame, complement)
        return cls(original, graph, (complement, frame))

    @classmethod
    def single(
        cls, original: torch.Tensor, compute: Callable[[], torch.Tensor]
    ) -> "tuple[_Shared | None, torch.Tensor]":
        """A quantity whose graph reaches ``original`` by one edge and no
        other trainable leaf — then one replay per access is the standalone
        backward exactly — or ``None`` when it does not, and each access must
        compute its own. The evaluation made to decide is returned beside:
        when nothing can be shared it is exactly the first access's own
        graph, so the probe costs that access nothing."""
        with torch.enable_grad():
            graph = compute()
        if not graph.requires_grad:
            return cls(original, graph, ()), graph
        edges, others = _leaf_edges(graph, original)
        if edges != 1 or others:
            return None, graph
        return cls(original, graph, (original,)), graph

    def access(self) -> torch.Tensor:
        if self.aliases and torch.is_grad_enabled() and self.original.requires_grad:
            return cast(
                torch.Tensor,
                _Replay.apply(self, *([self.original] * len(self.aliases))),
            )
        return self.value

    def replay(self, cotangent: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            self.graph, self.aliases, cotangent, retain_graph=True
        )


#: A scope entry saying a quantity cannot be shared under grad — each access
#: computes its own (`_Shared.single`)
_UNSHARED = object()


class _Replay(torch.autograd.Function):
    """One access's autograd node over a `_Shared`: the value for
    free, the backward replayed for this access alone. The parameter is
    passed once per contribution so the node hands them to the parameter's
    accumulator one after the other, as the quantity's own graph does.

    First order only: the replay is ``autograd.grad`` without
    ``create_graph``, so a backward run with ``create_graph=True`` (a
    Hessian-vector product, a meta-gradient) would not see the map in the
    second-order graph. No caller differentiates twice; one that does must
    evaluate outside the scope."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, shared: _Shared, *originals: torch.Tensor
    ) -> torch.Tensor:
        ctx.shared = shared
        # a distinct tensor object per access over the one storage, not
        # `shared.value` itself: `apply` attaches this node to the tensor it
        # returns, and a second access returning the same object would
        # overwrite the first's node, losing that access's cotangent
        return shared.value.detach()

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, cotangent: torch.Tensor
    ) -> tuple[Any, ...]:
        return (None, *ctx.shared.replay(cotangent))
