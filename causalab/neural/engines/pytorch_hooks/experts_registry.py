"""One installation over the experts registry entry, shared by every
active enterer (``docs/model_parallelism.md`` §6.3; the module docstrings of
``experts_interface.py`` and ``experts_path.py``, "How the call is
intercepted").

transformers resolves the experts forward through one process-global mapping,
``ALL_EXPERTS_FUNCTIONS["grouped_mm"]``. Two things in this repository install
a callable over that key for a dynamic extent — the interface taps' dispatch
and the lean experts path — and the ranks of a simulated world are threads of
one process, each entering around its own forward and leaving in its own
order. A plain capture-and-restore per enterer breaks there: the second
thread captures the first's installation as "previous", the first's exit
clobbers the second's, and the last exit leaves the process pointing at a
stale closure. [`EntryInstall`][] is the one rule both use — the **first**
active enterer captures the entry it found and installs the replacement, the
**last** leaver restores what the first found, under a lock, with the count
in between — so the omission the plain form invites cannot recur: a new
install over the key is a new instance, not a new mechanism.

Two installs nest LIFO: the lean path is entered around the engine's forward
and the taps inside it, so the taps' first enterer finds the lean function
and restores it, and the lean path's last leaver restores the library's. A
leave out of that order — the inner install still over this one — is refused
by name rather than clobbering it ([`EntryInstallError`][]); the entry
then keeps this install's function with what it found beneath, so the outer's
own restore leaves a working entry and the next enterer finds it and keeps
what is beneath rather than capturing itself.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Iterator, MutableMapping

__all__ = ["EntryInstall", "EntryInstallError", "experts_functions"]


class EntryInstallError(RuntimeError):
    """An install misused (module docstring): a leave without an enter, or a
    leave while an install made over this one has not left — installs nest
    LIFO, and restoring what this one found would bypass the other and leave
    this install's function in the registry once the other restored it. A
    programming error of a manager, never a document's."""


def experts_functions() -> MutableMapping[str, Any]:
    """The library's registry, imported at first use like every transformers
    import of these modules."""
    import transformers.integrations.moe as moe

    return moe.ALL_EXPERTS_FUNCTIONS


class EntryInstall:
    """A refcounted, locked installation over ``registry()[key]`` (module
    docstring). ``install`` is read at the first enterer's entry and is what
    goes into the registry — a getter, so a module-level function a test
    rebinds (the copy's spy seam) is what installs, and the entry's identity
    is the function's own. ``registry`` is the mapping's getter — the
    library's by default, a dict in a test."""

    def __init__(
        self,
        key: str,
        install: Callable[[], Callable[..., Any]],
        *,
        registry: Callable[[], MutableMapping[str, Any]] = experts_functions,
    ) -> None:
        self.key = key
        self.install = install
        self._registry = registry
        self._lock = threading.Lock()
        self._active = 0
        self._previous: Any = None
        #: what the first enterer put in the registry; kept across a refused
        #: leave so the next enterer recognises it
        self._installed: Any = None

    @property
    def active(self) -> int:
        """How many enterers are inside; the entry is installed iff above zero."""
        return self._active

    @property
    def previous(self) -> Any:
        """The entry the first enterer found — what the replacement
        delegates to; ``None`` while nobody is inside."""
        return self._previous

    def enter(self) -> None:
        with self._lock:
            if self._active == 0:
                registry = self._registry()
                found = registry[self.key]
                if self._installed is None or found is not self._installed:
                    self._previous = found
                # else: left in place by a refused leave, ``_previous`` still
                # what it found beneath
                self._installed = self.install()
                registry[self.key] = self._installed
            self._active += 1

    def leave(self) -> None:
        with self._lock:
            if self._active == 0:
                raise EntryInstallError(f"{self.key!r}: leave without a matching enter")
            self._active -= 1
            if self._active == 0:
                registry = self._registry()
                installed = registry[self.key]
                if installed is not self._installed:
                    raise EntryInstallError(
                        f"{self.key!r}: the entry is {installed!r}, not what this "
                        "install put there — an install made over it has not left "
                        "(installs nest LIFO; the lean experts path enters before "
                        "the taps and leaves after them)"
                    )
                registry[self.key] = self._previous
                self._previous = None
                self._installed = None

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        """``enter`` on the way in, ``leave`` on the way out."""
        self.enter()
        try:
            yield
        finally:
            self.leave()
