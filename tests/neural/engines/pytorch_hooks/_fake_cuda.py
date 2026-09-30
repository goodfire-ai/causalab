"""The pieces of ``torch.cuda`` the capture path touches, stood in for on a
CPU: a "capture" runs its body eagerly, a "replay" does nothing, a memory
pool is a counter, and every graph records the pool it was captured into.
The tests for ``cuda_graphs.py`` and ``graph_cohort.py`` drive the real
executors and real objectives through this; what is faked is only the CUDA
driver boundary, so the pool threading — which graph lands in which pool,
which pool is released when, whether a graph was reset while alive — is
observed here and the arithmetic of replay on ``tests/golden``. Not modelled:
the allocator's pool use counts (``FakeMemPool.use_count`` is always 1), so
the process abort a pool dropped under a live graph would cause is out of
reach on a CPU; the ordering that prevents it is what these tests pin."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import pytest
import torch


@dataclasses.dataclass(eq=False)  # hashable: GraphPool keeps a WeakSet
class FakeGraph:
    pool_id: tuple[int, int] | None = None
    stream: Any = None
    replays: int = 0
    resets: int = 0

    def replay(self) -> None:
        if self.resets:
            raise RuntimeError("replay of a reset graph")
        self.replays += 1

    def reset(self) -> None:
        self.resets += 1

    def pool(self) -> tuple[int, int] | None:
        return self.pool_id


class FakeMemPool:
    """``torch.cuda.MemPool``: a fresh id per construction; the constructor
    arguments are kept so a test can see how the pool was opened."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.id = (0, id(self))
        self.args, self.kwargs = args, kwargs
        self.released = False

    def use_count(self) -> int:
        return 1


class _Stream:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def wait_stream(self, _other: Any) -> None:
        pass


class FakeCuda:
    """Installed on ``torch.cuda`` for one test. ``captures`` lists every
    graph captured and the ``pool=`` it was captured into, in order;
    ``pools`` every memory pool constructed; ``routed`` the allocator
    routing calls (``GraphPool.allocating``: ``begin``, ``end``, ``release``
    with the pool id) in order, and ``routing`` the pool allocations are
    routed to at any moment (``None`` outside a warm-up)."""

    def __init__(self) -> None:
        self.captures: list[FakeGraph] = []
        self.pools: list[FakeMemPool] = []
        self.routed: list[tuple[str, tuple[int, int]]] = []
        self.routing: tuple[int, int] | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeCuda":
        fake = self

        @contextlib.contextmanager
        def graph(cuda_graph: FakeGraph, pool: Any = None, stream: Any = None):
            cuda_graph.pool_id = pool
            cuda_graph.stream = stream
            fake.captures.append(cuda_graph)
            yield

        def mempool(*args: Any, **kwargs: Any) -> FakeMemPool:
            pool = FakeMemPool(*args, **kwargs)
            fake.pools.append(pool)
            return pool

        def no_memory_guard(*_args: Any, **_kwargs: Any) -> tuple[int, int]:
            raise AssertionError("the capture path consulted the memory guard")

        def begin_routing(_device: int, pool_id: tuple[int, int]) -> None:
            assert fake.routing is None, "allocations already routed to a pool"
            fake.routing = pool_id
            fake.routed.append(("begin", pool_id))

        def end_routing(_device: int, pool_id: tuple[int, int]) -> None:
            assert fake.routing == pool_id
            fake.routing = None
            fake.routed.append(("end", pool_id))

        def release_routing(_device: int, pool_id: tuple[int, int]) -> None:
            fake.routed.append(("release", pool_id))

        monkeypatch.setattr(
            torch.cuda, "device", lambda _device: contextlib.nullcontext()
        )
        monkeypatch.setattr(torch.cuda, "Stream", _Stream)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a, **_k: _Stream())
        monkeypatch.setattr(
            torch.cuda, "stream", lambda _stream: contextlib.nullcontext()
        )
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a, **_k: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
        monkeypatch.setattr(torch.cuda, "graph", graph)
        monkeypatch.setattr(torch.cuda, "MemPool", mempool)
        monkeypatch.setattr(torch.cuda, "mem_get_info", no_memory_guard)
        monkeypatch.setattr(torch.cuda, "memory_snapshot", no_memory_guard)
        # the allocator's pool routing: installed whatever the build has
        # (``raising=False``) — ``test_the_pool_routing_bindings_exist`` is
        # the check that the real names are still there
        monkeypatch.setattr(
            torch._C, "_cuda_beginAllocateToPool", begin_routing, raising=False
        )
        monkeypatch.setattr(
            torch._C, "_cuda_endAllocateToPool", end_routing, raising=False
        )
        monkeypatch.setattr(
            torch._C, "_cuda_releasePool", release_routing, raising=False
        )
        return self

    @property
    def pool_ids(self) -> list[tuple[int, int] | None]:
        return [graph.pool_id for graph in self.captures]
