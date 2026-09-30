"""The three per-forward bindings on one kernel global, interleaved across
threads the way the simulated world interleaves its ranks.

The reference executor enters, around every forward, ``torch_kernel_path``,
then ``short_seq_kernel_path``, then the DeltaNet taps — three patches of the
same modeling-module global, ``torch_chunk_gated_delta_rule``. A simulated
world (``docs/model_parallelism.md`` §10.2) runs its ranks as threads that
hand off *inside* the collectives the taps make, so two ranks are mid-forward
at once, each inside all three. Every patch is therefore a **layer** on the
one [`SymbolDispatch`][causalab.neural.shared.symbol_dispatch.SymbolDispatch] at the
symbol, per thread: a rank's kernel call runs its own layers, top down, and
the symbol is the original object again only once every rank has left.

The interleaving below is one a CI run of ``TestFitAtContextTwoOnTheMoe``
produced with two plain replace-and-restore
managers around the dispatch: rank 0's short path restored the global while
rank 1 was still inside its own, so rank 1's tap was silently bypassed and,
after rank 1 left, the global was a short dispatcher wrapping an
uninstalled dispatch — ``not_installed`` on the next forward, or a
``RecursionError``.

This module is its own modeling file — it exports the four kernel globals
and a mixer whose forward calls the chunked kernel by name — so the bindings
patch *this* module, the same path as on the fixture.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any, Callable

import pytest
import torch
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _qwen

from causalab.neural.engines.pytorch_hooks.delta_interface import (
    DeltaTap,
    delta_kernel_taps,
)
from causalab.neural.shared.gdn_short.binding import short_seq_kernel_path
from causalab.neural.shared.gdn_short.options import (
    CHUNK_KERNEL_GLOBAL,
    ShortSeqKernelOptions,
)
from causalab.neural.shared.kernels import KERNEL_GLOBALS, torch_implementation

pytestmark = pytest.mark.unit

# --------------------------------------------------------------------------- #
# this module as a modeling file
# --------------------------------------------------------------------------- #

causal_conv1d_fn = torch_implementation(_qwen.causal_conv1d_fn)
causal_conv1d_update = torch_implementation(_qwen.causal_conv1d_update)
torch_chunk_gated_delta_rule = torch_implementation(_qwen.torch_chunk_gated_delta_rule)
torch_recurrent_gated_delta_rule = torch_implementation(
    _qwen.torch_recurrent_gated_delta_rule
)
l2norm = _qwen.l2norm

_ORIGINALS = {name: globals()[name] for name in KERNEL_GLOBALS}

B, L, H, D = 1, 6, 2, 4


class _Mixer(torch.nn.Module):
    """The chunked kernel call site, by name, as the Qwen3.5 mixer makes it."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        out, _ = torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=torch.full_like(g, 0.5),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return out


def _inputs(seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(B, L, H, D, generator=generator)
    k = torch.randn(B, L, H, D, generator=generator)
    v = torch.randn(B, L, H, D, generator=generator)
    g = -torch.rand(B, L, H, generator=generator)
    return q, k, v, g


def _module() -> Any:
    return sys.modules[__name__]


# --------------------------------------------------------------------------- #
# a baton: the steps of two threads, run in one drawn order
# --------------------------------------------------------------------------- #

#: The longest a participant waits for its turn before the test fails rather
#: than hangs.
TIMEOUT = 10.0


class Baton:
    """Hands one turn at a time to a named participant, which runs one step
    and hands the turn back."""

    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.turn: int | None = None
        self.finished = False
        self.errors: list[BaseException] = []

    def give(self, participant: int) -> None:
        with self.cv:
            self.turn = participant
            self.cv.notify_all()
            if not self.cv.wait_for(lambda: self.turn is None or self.errors, TIMEOUT):
                raise AssertionError(f"participant {participant} never took its turn")

    def take(self, participant: int) -> bool:
        with self.cv:
            if not self.cv.wait_for(
                lambda: self.turn == participant or self.finished or self.errors,
                TIMEOUT,
            ):
                raise AssertionError(f"participant {participant} was never scheduled")
            return self.turn == participant and not self.errors

    def done(self) -> None:
        with self.cv:
            self.turn = None
            self.cv.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.cv:
            self.errors.append(error)
            self.turn = None
            self.cv.notify_all()

    def finish(self) -> None:
        with self.cv:
            self.finished = True
            self.cv.notify_all()


def _run(scripts: list[list[Callable[[], None]]], order: list[int]) -> None:
    """Run each participant's steps on its own thread, one step per turn, in
    ``order``; a step raising fails the test on the test thread."""
    baton = Baton()

    def participant(who: int, steps: list[Callable[[], None]]) -> None:
        for step in steps:
            if not baton.take(who):
                return
            try:
                step()
            except BaseException as error:  # re-raised on the test thread
                baton.fail(error)
                return
            baton.done()

    threads = [
        threading.Thread(
            target=participant, args=(who, steps), name=f"rank-{who}", daemon=True
        )
        for who, steps in enumerate(scripts)
    ]
    for thread in threads:
        thread.start()
    try:
        for who in order:
            baton.give(who)
            if baton.errors:
                break
    finally:
        baton.finish()
        for thread in threads:
            thread.join(TIMEOUT)
    if baton.errors:
        raise baton.errors[0]


class _Rank:
    """One rank's managers as separate steps — entered and left one at a
    time, the way the executor's ``ExitStack`` brackets a forward."""

    def __init__(self, seed: int) -> None:
        self.mixer = _Mixer()
        self.inputs = _inputs(seed)
        self.captured: list[torch.Tensor] = []
        self.short = contextlib.ExitStack()
        self.taps = contextlib.ExitStack()
        self.out: torch.Tensor | None = None

    def enter_short(self) -> None:
        self.short.enter_context(
            short_seq_kernel_path(self.mixer, ShortSeqKernelOptions(16))
        )

    def exit_short(self) -> None:
        self.short.close()

    def enter_taps(self) -> None:
        tap = DeltaTap("kernel_output", read=lambda t: self.captured.append(t.clone()))
        self.taps.enter_context(delta_kernel_taps({self.mixer: (tap,)}))

    def exit_taps(self) -> None:
        self.taps.close()

    def call(self) -> None:
        self.out = self.mixer(*self.inputs)


def _globals_are_the_originals() -> bool:
    return all(globals()[name] is original for name, original in _ORIGINALS.items())


def test_the_ci_interleaving_runs_each_ranks_own_tap_and_restores_the_original() -> (
    None
):
    """rank 0: short, rank 1: short, rank 0: taps, rank 1: taps, rank 0 leaves
    its taps and its short path; **rank 1 then calls** — and must run its own
    tap over the original kernel — before leaving its own two. Afterwards the
    global is the original object."""
    ranks = [_Rank(0), _Rank(1)]
    reference = _Mixer()(*ranks[1].inputs)
    scripts = [
        [
            ranks[0].enter_short,
            ranks[0].enter_taps,
            ranks[0].exit_taps,
            ranks[0].exit_short,
        ],
        [
            ranks[1].enter_short,
            ranks[1].enter_taps,
            ranks[1].call,
            ranks[1].exit_taps,
            ranks[1].exit_short,
        ],
    ]
    _run(scripts, order=[0, 1, 0, 1, 0, 0, 1, 1, 1])
    assert len(ranks[1].captured) == 1, "rank 1's own tap must run its call"
    assert ranks[1].out is not None and torch.equal(ranks[1].out, reference)
    assert torch.equal(ranks[1].captured[0], reference)
    assert not ranks[0].captured
    assert _globals_are_the_originals()
    assert getattr(_module(), CHUNK_KERNEL_GLOBAL) is _ORIGINALS[CHUNK_KERNEL_GLOBAL]
    # and the next forward, with nothing entered, is the plain original
    assert torch.equal(_Mixer()(*ranks[1].inputs), reference)


def test_both_ranks_calling_while_both_are_inside_each_run_their_own_tap() -> None:
    """The steady state of a simulated forward: both ranks inside all their
    layers, each calling in turn; each tap sees exactly its own call."""
    ranks = [_Rank(0), _Rank(1)]
    references = [_Mixer()(*rank.inputs) for rank in ranks]
    scripts = [
        [r.enter_short, r.enter_taps, r.call, r.exit_taps, r.exit_short] for r in ranks
    ]
    _run(scripts, order=[0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    for rank, reference in zip(ranks, references):
        assert len(rank.captured) == 1
        assert torch.equal(rank.captured[0], reference)
    assert _globals_are_the_originals()
