"""One stand-in per patched symbol, running each call through the calling
thread's own stack of patches.

A patch on a function interior rebinds a **symbol** — a modeling module's
global (the four DeltaNet kernel globals of ``shared/kernels.py``,
``transformers.integrations.moe._grouped_linear``) — for the duration of one
forward or one call. Three such patches land on the chunked kernel global
around every forward of the reference engine: the torch-path guard
(``kernels.torch_kernel_path``), the short-sequence dispatcher
(``gdn_short/binding.py``) and the DeltaNet taps
(``pytorch_hooks/delta_interface.py``), each wrapping what the one before it
bound. A simulated world (``docs/model_parallelism.md`` §10.2) runs its ranks
as threads of one process that hand off *inside* the collectives the taps
make, so two ranks are mid-forward at once, each inside all three. Plain
replace-and-restore patches then interleave: one rank's exit puts back the
value it saw on entry while the other is still inside — its patches
silently bypassed, and the symbol left holding a stale wrapper for the rest
of the process.

The dispatch is therefore **the one object** installed at a symbol while any
thread is inside it, and every patch is a **layer** on it, per thread:

* [`SymbolDispatch.enter`][] pushes a layer on the calling thread's stack
  (nested layers on one thread are the normal case — the short path outside,
  the tap inside); the first layer process-wide installs the dispatch at the
  symbol and captures the symbol's value as the **original**; the last layer
  to leave process-wide puts the original back, in whatever order the threads
  leave;
* a call through the symbol runs the calling thread's **top** layer; while a
  layer runs, [`SymbolDispatch.real`][] — what it computes with — is the
  layer below it on the same thread, and the original below the bottom; a
  thread with no layers runs the original;
* [`SymbolDispatch.rebind`][] changes what lies beneath the dispatch — the
  original while installed, the symbol itself otherwise — for a loader that
  binds a family's globals for good while another rank's forward may be
  inside them; [`SymbolDispatch.hold`][] is the counted form of the same
  swap for a per-forward guard that must leave a plain function at the
  symbol when nothing is layered (the nnsight engine's ``.source`` drills
  into the callee it finds at run time, and reads its ``__code__``).

The dispatch reaches the original through ``__wrapped__`` like any
``functools.wraps`` wrapper, so ``kernels.torch_implementation`` resolves
through it, and a dispatch that was installed and fully left keeps
forwarding to the original it captured — a stale reference to it must keep
working.

Misuse — leaving a layer that is not the thread's top, leaving one the thread
never entered, reading ``real`` of a dispatch nothing has ever installed,
releasing a hold nobody took — is a [`DispatchMisuse`][]: a programming
error of a manager, never a document's, so it is not a protocol refusal.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from typing import Any, Callable, Iterator, Literal, Protocol

__all__ = [
    "AttributeSymbol",
    "DispatchMisuse",
    "Layer",
    "LazySymbol",
    "Symbol",
    "SymbolDispatch",
    "dispatch_for",
]


class Symbol(Protocol):
    """Where a patch lands: a readable, writable slot with a name for errors."""

    def get(self) -> Any: ...

    def set(self, value: Any) -> None: ...

    def __str__(self) -> str: ...


@dataclasses.dataclass(frozen=True)
class AttributeSymbol:
    """An attribute of ``owner`` — a modeling module's global function."""

    owner: Any
    name: str

    def get(self) -> Any:
        return getattr(self.owner, self.name)

    def set(self, value: Any) -> None:
        setattr(self.owner, self.name, value)

    def __str__(self) -> str:
        owner = getattr(self.owner, "__name__", None) or type(self.owner).__name__
        return f"{owner}.{self.name}"


class LazySymbol:
    """A symbol resolved at first use — for an owner imported lazily (the
    library module a manager patches is imported inside the manager, not at
    import time). Resolved once, then held."""

    def __init__(self, resolve: Callable[[], Symbol]) -> None:
        self._resolve = resolve
        self._symbol: Symbol | None = None

    def _resolved(self) -> Symbol:
        if self._symbol is None:
            self._symbol = self._resolve()
        return self._symbol

    def get(self) -> Any:
        return self._resolved().get()

    def set(self, value: Any) -> None:
        self._resolved().set(value)

    def __str__(self) -> str:
        return str(self._resolved())


MisuseKind = Literal[
    "leave_without_enter",
    "leave_out_of_order",
    "not_installed",
    "found_itself",
    "release_without_hold",
    "hold_disagrees",
]

_MISUSE_TEXT: dict[MisuseKind, str] = {
    "leave_without_enter": "left a layer it never entered",
    "leave_out_of_order": "left a layer that is not its top one",
    "not_installed": "read the real function of a dispatch nothing has installed",
    "found_itself": "found the dispatch at a symbol nothing has entered",
    "release_without_hold": "released a hold nobody took",
    "hold_disagrees": "held a symbol at a value another hold disagrees with",
}


class DispatchMisuse(RuntimeError):
    """A manager misusing a dispatch (module docstring): ``kind`` says how,
    ``symbol`` and ``thread`` name where."""

    def __init__(self, symbol: str, kind: MisuseKind, thread: str) -> None:
        self.symbol = symbol
        self.kind: MisuseKind = kind
        self.thread = thread
        super().__init__(f"thread {thread!r} {_MISUSE_TEXT[kind]}: {symbol}")


class Layer:
    """One entered patch on one thread — the handle [`SymbolDispatch.enter`][]
    returns and [`SymbolDispatch.leave`][] takes back. Callable: it runs its
    patch with the dispatch's ``real`` pointing one layer down, which is how
    a layer above reaches it (``real`` hands out the layer below)."""

    __slots__ = ("_dispatch", "patched", "below")

    def __init__(
        self, dispatch: SymbolDispatch, patched: Callable[..., Any], below: Layer | None
    ) -> None:
        self._dispatch = dispatch
        self.patched = patched
        self.below = below

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch.run(self, args, kwargs)

    @property
    def __wrapped__(self) -> Callable[..., Any]:
        return self.patched


class _Below:
    """``dispatch.below``: a callable that runs whatever ``real`` is at call
    time — what a layer built before it is entered hands its wrapper as the
    function beneath it. Its ``__wrapped__`` is the dispatch, so a wrapper
    chain through it reaches the original."""

    __slots__ = ("_dispatch",)

    def __init__(self, dispatch: SymbolDispatch) -> None:
        self._dispatch = dispatch

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch.real(*args, **kwargs)

    @property
    def __wrapped__(self) -> SymbolDispatch:
        return self._dispatch


@dataclasses.dataclass
class _ThreadState:
    stack: list[Layer] = dataclasses.field(default_factory=list)
    #: the layer running on this thread, if a call is in flight
    cursor: Layer | None = None


class SymbolDispatch:
    """The one object installed at a symbol while any thread is inside
    (module docstring). Callable in the symbol's place: a call runs the
    calling thread's top layer, the original when it has none."""

    def __init__(self, symbol: Symbol) -> None:
        self._symbol = symbol
        self._local = threading.local()
        self._lock = threading.Lock()
        #: layers in flight over every thread
        self._active = 0
        #: what lies beneath the dispatch — captured by the first enterer,
        #: changed by ``rebind`` and ``hold``; ``None`` only before the first
        self._original: Callable[..., Any] | None = None
        self._holds = 0
        self._held_over: Callable[..., Any] | None = None
        self._below = _Below(self)

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    @property
    def installed(self) -> bool:
        """Whether any thread is inside — the symbol is this dispatch iff so.
        An advisory read outside the lock (for a diagnostic, a test); ``real``
        is the checked reading of the same state."""
        return self._active > 0

    # ------------------------------------------------------------ the bottom

    @property
    def bound(self) -> Callable[..., Any]:
        """What lies beneath the dispatch: the captured original while
        installed, the symbol's own value otherwise — what a loader
        inspects to decide whether the family needs rebinding."""
        with self._lock:
            return self._bottom()

    @property
    def __wrapped__(self) -> Callable[..., Any]:
        return self.bound

    def rebind(self, value: Callable[..., Any]) -> None:
        """Set what lies beneath the dispatch — the original every bottom
        layer computes with while installed, the symbol itself otherwise."""
        with self._lock:
            self._set_bottom(value)

    def hold(self, value: Callable[..., Any]) -> None:
        """Swap the bottom for ``value`` until the last holder releases: the
        first holder process-wide rebinds and remembers what it replaced,
        [`release`][] by the last puts that back. A second holder must
        agree on the value."""
        with self._lock:
            if self._holds == 0:
                self._held_over = self._bottom()
                self._set_bottom(value)
            elif self._original is not value:
                raise DispatchMisuse(str(self._symbol), "hold_disagrees", _thread())
            self._holds += 1

    def release(self) -> None:
        """Let go of a [`hold`][]; the last release restores the bottom."""
        with self._lock:
            if self._holds == 0:
                raise DispatchMisuse(
                    str(self._symbol), "release_without_hold", _thread()
                )
            self._holds -= 1
            if self._holds == 0:
                assert self._held_over is not None
                self._set_bottom(self._held_over)
                self._held_over = None

    @contextlib.contextmanager
    def holding(self, value: Callable[..., Any]) -> Iterator[None]:
        """[`hold`][] on entry, [`release`][] on exit."""
        self.hold(value)
        try:
            yield
        finally:
            self.release()

    def _bottom(self) -> Callable[..., Any]:
        """``bound``, under the lock."""
        if self._active > 0:
            assert self._original is not None
            return self._original
        return self._symbol.get()

    def _set_bottom(self, value: Callable[..., Any]) -> None:
        """``rebind``, under the lock."""
        if self._active == 0:
            self._symbol.set(value)
        self._original = value

    # ------------------------------------------------------------ the layers

    def _state(self) -> _ThreadState:
        state = getattr(self._local, "state", None)
        if state is None:
            state = self._local.state = _ThreadState()
        return state

    @property
    def real(self) -> Callable[..., Any]:
        """What the running layer computes with: the layer below it on this
        thread, the original below the bottom. Read outside a call, the view
        below the thread's top layer — the original for a thread with none.
        A dispatch nothing has ever installed has no original to give."""
        original = self._original
        if original is None:
            raise DispatchMisuse(str(self._symbol), "not_installed", _thread())
        state = self._state()
        above = state.cursor if state.cursor is not None else _top(state)
        if above is None or above.below is None:
            return original
        return above.below

    @property
    def below(self) -> Callable[..., Any]:
        """A callable running [`real`][] as it is at call time — the
        function a wrapper built before its layer is entered calls through
        to (``gdn_short``'s dispatcher over the layer beneath it)."""
        return self._below

    def enter(self, patched: Callable[..., Any]) -> Layer:
        """Push ``patched`` as this thread's top layer; the first layer
        process-wide installs the dispatch at the symbol and captures the
        original. The order is load-bearing both ways: the count rises under
        the lock before the layer joins this thread's stack, and
        [`leave`][] pops the stack before the count falls, so ``real``
        stays defined for any call already inside on this thread."""
        state = self._state()
        with self._lock:
            if self._active == 0:
                found = self._symbol.get()
                if self._reaches_self(found):
                    # a stale wrapper over this dispatch, or the dispatch
                    # itself: what we hold is the original it stands over
                    if self._original is None:
                        raise DispatchMisuse(
                            str(self._symbol), "found_itself", _thread()
                        )
                else:
                    self._original = found
                self._symbol.set(self)
            self._active += 1
        layer = Layer(self, patched, _top(state))
        state.stack.append(layer)
        return layer

    def leave(self, layer: Layer) -> None:
        """Pop ``layer``, this thread's top; the last layer process-wide
        restores the symbol."""
        state = self._state()
        if not any(entered is layer for entered in state.stack):
            raise DispatchMisuse(str(self._symbol), "leave_without_enter", _thread())
        if state.stack[-1] is not layer:
            raise DispatchMisuse(str(self._symbol), "leave_out_of_order", _thread())
        state.stack.pop()
        with self._lock:
            self._active -= 1
            if self._active == 0:
                self._symbol.set(self._original)

    @contextlib.contextmanager
    def tapped(self, patched: Callable[..., Any]) -> Iterator[Layer]:
        """[`enter`][] on entry, [`leave`][] on exit — the body raising
        or not."""
        layer = self.enter(patched)
        try:
            yield layer
        finally:
            self.leave(layer)

    def run(self, layer: Layer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run ``layer``'s patch with the cursor on it, so ``real`` read inside
        is the layer below — what a call of the layer object does."""
        state = self._state()
        previous = state.cursor
        state.cursor = layer
        try:
            return layer.patched(*args, **kwargs)
        finally:
            state.cursor = previous

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        top = _top(self._state())
        if top is None:
            return self.real(*args, **kwargs)
        return self.run(top, args, kwargs)

    def _reaches_self(self, found: Any) -> bool:
        """Whether ``found`` is this dispatch or a wrapper chain over it."""
        seen: set[int] = set()
        while found is not None and id(found) not in seen:
            if found is self:
                return True
            seen.add(id(found))
            found = getattr(found, "__wrapped__", None)
        return False


def _top(state: _ThreadState) -> Layer | None:
    return state.stack[-1] if state.stack else None


def _thread() -> str:
    return threading.current_thread().name


#: ``(owner, name) -> its dispatch``, created at first use and kept for the
#: process — a modeling module lives in ``sys.modules`` as long as the
#: process does, and one dispatch per symbol is the point. Keyed by the
#: owner object: a reloaded module is a new key.
_DISPATCHES: dict[tuple[Any, str], SymbolDispatch] = {}
_DISPATCHES_LOCK = threading.Lock()


def dispatch_for(owner: Any, name: str) -> SymbolDispatch:
    """The one dispatch for the attribute ``name`` of ``owner`` (a module
    global, usually) — every manager patching that symbol goes through it."""
    key = (owner, name)
    with _DISPATCHES_LOCK:
        found = _DISPATCHES.get(key)
        if found is None:
            found = _DISPATCHES[key] = SymbolDispatch(AttributeSymbol(owner, name))
    return found
