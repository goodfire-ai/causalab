"""``delta_kernel_taps`` with many managers in flight at once — the simulated
world's ranks (``docs/model_parallelism.md`` §8.4, §10.2, §10.4).

Under ``cp > 1`` every rank enters its own manager around its own forward,
and the kernel wrappers hand the recurrent state and the conv history chunk
to chunk — collectives *inside* the patched call, where a simulated rank
gives up the baton. So two managers are entered at once on two threads,
each having patched the same four module globals. The manager therefore
routes through [`SymbolDispatch`][causalab.neural.shared.symbol_dispatch.SymbolDispatch]:
one installation per symbol, each thread's wrappers a layer of its own.
Before it, the second manager wrapped the first's wrappers, and a rank's
kernel call ran **both** — two handoffs where the group agreed on one, a
divergence the simulator refuses.

This module is its own modeling file: it exports the four kernel globals
(transformers' torch implementations, resolved as ``kernels.py`` resolves
them) and a mixer whose forward calls them the way the Qwen3.5 mixer does,
so ``delta_kernel_taps`` patches *this* module — the same code path as on
the fixture, without a model per rank.

- every tapped rank's reads are its own chunk's tensors, one capture per
  kernel call, and an untapped rank captures nothing;
- every rank's gathered output is the unsplit forward's within
  `DELTA_BAND` (the chunked kernel's blocks fall differently on a
  split frame); the conv output and the kernel's ``value`` argument are
  the unsplit ones within `CONV_BAND` (the conv over a prefixed
  chunk rounds one ulp apart from the whole frame's — measured ``2.4e-7``
  and ``6.0e-8``);
- after the world has run, the four globals are the original objects.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, Iterator

import pytest
import torch
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _qwen
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.delta_interface import (
    DeltaTap,
    delta_kernel_taps,
)
from causalab.neural.shared.kernels import KERNEL_GLOBALS, torch_implementation
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.context import SequenceFrame, activate
from causalab.neural.shared.symbol_dispatch import dispatch_for
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import sequence_chunks
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, SimulatedWorld, groups_for

from .test_context_parallel import DELTA_BAND

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
)

# --------------------------------------------------------------------------- #
# this module as a modeling file: the four kernel globals and a mixer
# --------------------------------------------------------------------------- #

causal_conv1d_fn = torch_implementation(_qwen.causal_conv1d_fn)
causal_conv1d_update = torch_implementation(_qwen.causal_conv1d_update)
torch_chunk_gated_delta_rule = torch_implementation(_qwen.torch_chunk_gated_delta_rule)
torch_recurrent_gated_delta_rule = torch_implementation(
    _qwen.torch_recurrent_gated_delta_rule
)
l2norm = _qwen.l2norm

_ORIGINALS = {name: globals()[name] for name in KERNEL_GLOBALS}

#: The conv output over a prefixed chunk against the whole frame's (module
#: docstring): measured ``2.4e-7``, an fp32 ulp of its values, pinned at
#: four times.
CONV_BAND = 1e-6

B, L, H, DK, DV = 2, 12, 2, 4, 4
#: The conv's kernel width (the fixture's): a chunk must hold at least
#: ``KERNEL - 1`` positions to feed the next chunk's history.
KERNEL = 4
CHANNELS = 2 * H * DK + H * DV


class _Mixer(torch.nn.Module):
    """The Qwen3.5 mixer's two kernel call sites, nothing else: the causal
    conv over the fused projection, the split and tiling, the chunked
    kernel over the whole sequence."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(
        self, mixed: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        mixed = causal_conv1d_fn(
            mixed.transpose(1, 2), self.weight, None, activation="silu"
        ).transpose(1, 2)
        q, k, v = torch.split(mixed, [H * DK, H * DK, H * DV], dim=-1)
        q = q.reshape(B, -1, H, DK)
        k = k.reshape(B, -1, H, DK)
        v = v.reshape(B, -1, H, DV)
        out, _ = torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return out


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "mixed": torch.randn(B, L, CHANNELS, generator=generator),
        "g": -torch.rand(B, L, H, generator=generator),
        "beta": torch.rand(B, L, H, generator=generator),
        "weight": torch.randn(CHANNELS, KERNEL, generator=generator) * 0.3,
    }


def _recorder(into: dict[str, list[torch.Tensor]], slot: str) -> Callable[..., None]:
    def read(tensor: torch.Tensor) -> None:
        into.setdefault(slot, []).append(tensor.detach().clone())

    return read


SLOTS = ("conv", "value", "kernel_output")


def _program(
    seed: int, tapped: tuple[bool, ...]
) -> Callable[[int, Collective], dict[str, Any]]:
    """Each rank: its own mixer (the real world's own model copy), its own
    manager entered under the frame around its own forward over its chunk;
    the ranks named by ``tapped`` read three slots."""
    x = _inputs(seed)
    mask = torch.ones(B, L, dtype=torch.long)

    def program(rank: int, c: Collective) -> dict[str, Any]:
        frame = SequenceFrame(c, mask)
        s = slice(frame.chunk.start, frame.chunk.stop)
        mixer = _Mixer(x["weight"])
        captured: dict[str, list[torch.Tensor]] = {}
        taps: dict[Any, tuple[DeltaTap, ...]] = {}
        if tapped[rank]:
            taps[mixer] = tuple(
                DeltaTap(slot, read=_recorder(captured, slot)) for slot in SLOTS
            )
        with activate(frame), delta_kernel_taps(taps, model=mixer):
            out = mixer(x["mixed"][:, s], x["g"][:, s], x["beta"][:, s])
        return {"out": frame.gather(out, 1), "captured": captured}

    return program


def _reference(seed: int) -> dict[str, torch.Tensor]:
    """The unsplit forward at world 1, and the two exact interior tensors."""
    x = _inputs(seed)
    mixer = _Mixer(x["weight"])
    conv = causal_conv1d_fn(
        x["mixed"].transpose(1, 2), mixer.weight, None, activation="silu"
    )
    value = conv.transpose(1, 2)[..., 2 * H * DK :].reshape(B, L, H, DV)
    return {"out": mixer(x["mixed"], x["g"], x["beta"]), "conv": conv, "value": value}


@st.composite
def worlds(draw: st.DrawFn) -> tuple[int, int, tuple[bool, ...]]:
    cp = draw(st.sampled_from([2, 3]))
    seed = draw(st.integers(0, 2**32 - 1))
    tapped = list(draw(st.lists(st.booleans(), min_size=cp, max_size=cp)))
    tapped[draw(st.integers(0, cp - 1))] = True
    return cp, seed, tuple(tapped)


def _globals_are_the_originals() -> bool:
    return all(globals()[name] is original for name, original in _ORIGINALS.items())


@pytest.mark.property
class TestManagersInFlightAtOnce:
    @_SETTINGS
    @given(drawn=worlds(), schedule=ps.schedules())
    @example(drawn=(2, 0, (True, True)), schedule=[])
    def test_each_rank_reads_its_own_chunk_and_the_output_is_the_unsplit_ones(
        self, drawn: tuple[int, int, tuple[bool, ...]], schedule: Schedule
    ) -> None:
        cp, seed, tapped = drawn
        reference = _reference(seed)
        world = SimulatedWorld(groups_for(cp, context=cp), world=cp, schedule=schedule)
        results = world.run(_program(seed, tapped))
        assert _globals_are_the_originals()
        chunks = sequence_chunks(L, cp)
        for rank, got in enumerate(results):
            worst = (got["out"] - reference["out"]).abs().max().item()
            assert worst <= DELTA_BAND, (rank, worst)
            s = slice(chunks[rank].start, chunks[rank].stop)
            if not tapped[rank]:
                assert got["captured"] == {}
                continue
            captured = got["captured"]
            assert {slot: len(v) for slot, v in captured.items()} == {
                slot: 1 for slot in SLOTS
            }
            for slot, expected, band in (
                ("conv", reference["conv"][..., s], CONV_BAND),
                ("value", reference["value"][:, s], CONV_BAND),
                ("kernel_output", reference["out"][:, s], DELTA_BAND),
            ):
                worst = (captured[slot][0] - expected).abs().max().item()
                assert worst <= band, (rank, slot, worst)

    def test_world_one_is_the_unsplit_forward_bit_for_bit(self) -> None:
        reference = _reference(7)
        (got,) = SimulatedWorld(groups_for(1), world=1, schedule=[]).run(
            _program(7, (True,))
        )
        assert torch.equal(got["out"], reference["out"])
        assert torch.equal(got["captured"]["conv"][0], reference["conv"])
        assert _globals_are_the_originals()

    def test_the_manager_routes_through_one_dispatch_per_symbol(self) -> None:
        """The four globals are one dispatch object each while a manager is
        entered — the thing the interleaving above rests on."""
        mixer = _Mixer(_inputs(0)["weight"])
        with delta_kernel_taps({mixer: (DeltaTap("value", read=lambda _t: None),)}):
            for name in KERNEL_GLOBALS:
                installed = globals()[name]
                assert installed is dispatch_for(_module(), name)
                assert installed.real is _ORIGINALS[name]
        assert _globals_are_the_originals()


def _module() -> Any:
    return sys.modules[__name__]


# --------------------------------------------------------------------------- #
# the dynamic extent: which mixer is inside its forward, and how long it is
# --------------------------------------------------------------------------- #

#: The conv history a cache prepends: neither ``L`` nor ``CHANNELS``, so a
#: wrapper reading the sequence length off the wrong argument or dimension
#: slices the wrong columns.
HISTORY = 3


class _CachedMixer(_Mixer):
    """The mixer under a cache: the conv is handed the cache's history
    prepended to the sequence and the forward keeps the last ``seq_len``
    columns (``conv_wrapper``'s ⚠️). The history rides as the forward's
    second argument — a tensor whose second dimension is not the sequence —
    as the cache object rides on the real mixer; the gates are the module's."""

    def __init__(
        self, weight: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
    ) -> None:
        super().__init__(weight)
        self.register_buffer("g", g)
        self.register_buffer("beta", beta)

    def forward(self, mixed: torch.Tensor, history: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        seq_len = mixed.shape[1]
        conv_in = torch.cat([history, mixed], dim=1).transpose(1, 2)
        conv = causal_conv1d_fn(conv_in, self.weight, None, activation="silu")
        mixed = conv[..., -seq_len:].transpose(1, 2)
        q, k, v = torch.split(mixed, [H * DK, H * DK, H * DV], dim=-1)
        out, _ = torch_chunk_gated_delta_rule(
            q.reshape(B, -1, H, DK),
            k.reshape(B, -1, H, DK),
            v.reshape(B, -1, H, DV),
            g=self.g,
            beta=self.beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        return out


@pytest.mark.unit
class TestTheDynamicExtent:
    def test_a_tap_on_the_conv_alone_reads_it(self) -> None:
        """The slot test is per slot: a tap set holding the conv and no
        other slot still reads the conv."""
        x = _inputs(9)
        mixer = _Mixer(x["weight"])
        captured: dict[str, list[torch.Tensor]] = {}
        taps = {mixer: (DeltaTap("conv", read=_recorder(captured, "conv")),)}
        with delta_kernel_taps(taps):
            mixer(x["mixed"], x["g"], x["beta"])
        assert torch.equal(captured["conv"][0], _reference(9)["conv"])

    def test_after_a_tapped_forward_an_untapped_mixer_of_the_module_is_untouched(
        self,
    ) -> None:
        """The wrapper answers for every mixer of the module while installed;
        leaving the tapped mixer's forward is what keeps its taps off the
        next mixer's kernel call."""
        x = _inputs(5)
        tapped, untapped = _Mixer(x["weight"]), _Mixer(x["weight"])
        captured: dict[str, list[torch.Tensor]] = {}
        taps = {
            tapped: (
                DeltaTap("kernel_output", read=_recorder(captured, "kernel_output")),
            )
        }
        with delta_kernel_taps(taps):
            tapped(x["mixed"], x["g"], x["beta"])
            untapped(x["mixed"], x["g"], x["beta"])
            tapped(x["mixed"], x["g"], x["beta"])
        assert len(captured["kernel_output"]) == 2

    def test_a_conv_tap_under_a_cache_addresses_the_columns_the_forward_keeps(
        self,
    ) -> None:
        """The conv over history plus sequence is longer than the mixer's
        sequence; the read sees the kept columns alone and an edit lands on
        them — a zeroing edit zeroes q, k and v, and so the output."""
        x = _inputs(3)
        mixer = _CachedMixer(x["weight"], x["g"], x["beta"])
        history = torch.randn(
            B, HISTORY, CHANNELS, generator=torch.Generator().manual_seed(4)
        )
        captured: dict[str, list[torch.Tensor]] = {}
        edited: list[tuple[int, ...]] = []

        def zero(tensor: torch.Tensor) -> torch.Tensor:
            edited.append(tuple(tensor.shape))
            return torch.zeros_like(tensor)

        taps = {mixer: (DeltaTap("conv", read=_recorder(captured, "conv"), edit=zero),)}
        with delta_kernel_taps(taps):
            out = mixer(x["mixed"], history)
        (conv,) = captured["conv"]
        assert conv.shape == (B, CHANNELS, L) and edited == [(B, CHANNELS, L)]
        whole = causal_conv1d_fn(
            torch.cat([history, x["mixed"]], dim=1).transpose(1, 2),
            mixer.weight,
            None,
            activation="silu",
        )
        assert torch.equal(conv, whole[..., -L:])
        assert torch.equal(out, torch.zeros_like(out))
        assert not torch.equal(mixer(x["mixed"], history), out), "untapped, not zero"


# --------------------------------------------------------------------------- #
# the two refusals, and the third the per-step slots carry
# --------------------------------------------------------------------------- #


def _modeling(name: str, *, without: tuple[str, ...] = ()) -> Any:
    """A modeling module of its own in ``sys.modules``: this module's kernel
    globals and ``l2norm`` (less ``without``), and a ``Mixer`` whose forward
    is its own and calls its chunked kernel through the module."""
    module = types.ModuleType(name)
    for symbol in (*KERNEL_GLOBALS, "l2norm"):
        if symbol not in without:
            setattr(module, symbol, globals()[symbol])

    class Mixer(torch.nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
        ) -> torch.Tensor:
            kernel = sys.modules[type(self).__module__].torch_chunk_gated_delta_rule
            out, _ = kernel(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            return out

    Mixer.__module__ = name
    Mixer.forward.__module__ = name
    module.Mixer = Mixer  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


@pytest.fixture
def modeling_modules() -> Iterator[Callable[..., Any]]:
    """``_modeling``, every module it made unregistered afterwards."""
    made: list[str] = []

    def make(name: str, **kwargs: Any) -> Any:
        made.append(name)
        return _modeling(name, **kwargs)

    yield make
    for name in made:
        sys.modules.pop(name, None)


def _kernel_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, L, H, DK, generator=generator),
        torch.randn(B, L, H, DK, generator=generator),
        torch.randn(B, L, H, DV, generator=generator),
        -torch.rand(B, L, H, generator=generator),
        torch.rand(B, L, H, generator=generator),
    )


@pytest.mark.unit
class TestTheRefusals:
    def test_a_forward_from_outside_the_modeling_module_is_refused_as_kernelized(
        self, modeling_modules: Callable[..., Any]
    ) -> None:
        module = modeling_modules("fake_kernelized")

        def hub_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("never run")

        module.Mixer.forward = hub_forward  # a function of this module, not its own
        with pytest.raises(ProtocolError) as err:
            with delta_kernel_taps({module.Mixer(): (DeltaTap("value", read=print),)}):
                pass
        assert err.value.code == "P4"
        text = str(err.value)
        assert "a delta-kernel tap on Mixer: its forward comes from" in text
        assert (
            f"{__name__!r}, not its own modeling module 'fake_kernelized' — a "
            "kernelize()d (hub-kernel) mixer computes inside a fused kernel no "
            "module-global patch can reach. Load the model without kernelize(), "
            "or extend delta_interface.py for this kernel."
        ) in text

    def test_a_modeling_module_missing_a_kernel_global_is_refused_by_name(
        self, modeling_modules: Callable[..., Any]
    ) -> None:
        without = ("causal_conv1d_update", "torch_recurrent_gated_delta_rule")
        module = modeling_modules("fake_partial", without=without)
        with pytest.raises(ProtocolError) as err:
            with delta_kernel_taps({module.Mixer(): (DeltaTap("value", read=print),)}):
                pass
        assert err.value.code == "P4"
        text = str(err.value)
        missing = ", ".join(name for name in KERNEL_GLOBALS if name in without)
        assert f"its modeling module 'fake_partial' exports no {missing}." in text
        assert "Extend delta_interface.py for this family" in text
        assert "borrowing another family's kernels would silently change" in text

    def test_a_per_step_slot_needs_the_modeling_modules_l2norm(
        self, modeling_modules: Callable[..., Any]
    ) -> None:
        module = modeling_modules("fake_no_l2norm", without=("l2norm",))
        mixer = module.Mixer()
        inputs = _kernel_inputs(1)
        with pytest.raises(ProtocolError) as err:
            with delta_kernel_taps({mixer: (DeltaTap("state", read=print),)}):
                mixer(*inputs)
        assert err.value.code == "P4"
        assert "'fake_no_l2norm' to export 'l2norm'" in str(err.value)
        # the argument slots need no l2norm: the same module serves them
        seen: list[torch.Tensor] = []
        with delta_kernel_taps({mixer: (DeltaTap("value", read=seen.append),)}):
            mixer(*inputs)
        assert len(seen) == 1 and torch.equal(seen[0], inputs[2])


# --------------------------------------------------------------------------- #
# the per-step slots: the recurrent kernel in the chunked call's shadow
# --------------------------------------------------------------------------- #

STATE_SLOTS = ("kv_mem", "state_update", "state")
#: The ranks' per-step faces against the unsplit loop's: each rank starts
#: from the chunked kernel's final state where the loop holds the recurrent
#: kernel's, an fp32 ulp apart — measured ``2.4e-7``, pinned at four times.
#: (At world 1 the faces satisfy the recurrence written out bit for bit.)
STATE_BAND = 1e-6


def _tiled(conv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k, v as the mixer tiles them from the conv output."""
    q, k, v = torch.split(conv.transpose(1, 2), [H * DK, H * DK, H * DV], dim=-1)
    return q.reshape(B, -1, H, DK), k.reshape(B, -1, H, DK), v.reshape(B, -1, H, DV)


def _state_program(
    seed: int, *, write: bool
) -> Callable[[int, Collective], dict[str, Any]]:
    """Each rank: its own mixer under its own manager over its chunk, reading
    the three per-step slots — and, with ``write``, a state write that
    records the step it is handed and changes nothing."""
    x = _inputs(seed)
    mask = torch.ones(B, L, dtype=torch.long)

    def program(rank: int, c: Collective) -> dict[str, Any]:
        frame = SequenceFrame(c, mask)
        s = slice(frame.chunk.start, frame.chunk.stop)
        mixer = _Mixer(x["weight"])
        captured: dict[str, list[torch.Tensor]] = {}
        steps: list[int] = []
        taps = [DeltaTap(slot, read=_recorder(captured, slot)) for slot in STATE_SLOTS]
        if write:

            def keep(step: int, state: torch.Tensor) -> torch.Tensor:
                steps.append(step)
                return state

            taps.append(DeltaTap("state", edit_state=keep))
        with activate(frame), delta_kernel_taps({mixer: tuple(taps)}, model=mixer):
            out = mixer(x["mixed"][:, s], x["g"][:, s], x["beta"][:, s])
        faces = {slot: captured[slot][0] for slot in STATE_SLOTS}
        return {"out": frame.gather(out, 1), "faces": faces, "steps": steps}

    return program


@pytest.mark.unit
class TestThePerStepSlots:
    def test_reads_leave_the_forward_bit_for_bit_and_are_the_kernels_recurrence(
        self,
    ) -> None:
        """Reads at prefill run the loop in the chunked call's shadow: the
        output is the untouched forward's, and the faces satisfy the
        recurrence — ``kv_mem_t = (S_{t-1}·e^{g_t} · k̂_t).sum(-2)``,
        ``Δ_t = (v_t − kv_mem_t)·β_t``, ``S_t = S_{t-1}·e^{g_t} + k̂_t ⊗ Δ_t`` —
        step by step against the kernel's own returned states."""
        x = _inputs(11)
        reference = _reference(11)
        mixer = _Mixer(x["weight"])
        captured: dict[str, list[torch.Tensor]] = {}
        slots = (*STATE_SLOTS, "kernel_output")
        taps = {mixer: tuple(DeltaTap(s, read=_recorder(captured, s)) for s in slots)}
        with delta_kernel_taps(taps):
            out = mixer(x["mixed"], x["g"], x["beta"])
        assert torch.equal(out, reference["out"])
        assert torch.equal(captured["kernel_output"][0], reference["out"])
        states = captured["state"][0]
        kv_mem = captured["kv_mem"][0]
        delta = captured["state_update"][0]
        assert states.shape == (B, L, H, DK, DV)
        assert kv_mem.shape == delta.shape == (B, L, H, DV)
        _, k, v = _tiled(reference["conv"])
        k_hat = l2norm(k, dim=-1, eps=1e-6)
        previous = torch.zeros(B, H, DK, DV)
        for t in range(L):
            decayed = previous * x["g"][:, t].exp()[..., None, None]
            want_mem = (decayed * k_hat[:, t].unsqueeze(-1)).sum(-2)
            want_delta = (v[:, t] - want_mem) * x["beta"][:, t].unsqueeze(-1)
            want_state = decayed + k_hat[:, t].unsqueeze(-1) * want_delta.unsqueeze(-2)
            for got, want in (
                (kv_mem[:, t], want_mem),
                (delta[:, t], want_delta),
                (states[:, t], want_state),
            ):
                assert (got - want).abs().max().item() <= STATE_BAND, t
            previous = states[:, t]

    def test_a_write_substitutes_the_loop_and_feeds_forward(self) -> None:
        """A state write runs the loop in the forward's place, once per
        step in order; a write that zeroes every state leaves each step
        seeing no memory, and the output differs from the reference's."""
        x = _inputs(13)
        reference = _reference(13)
        steps: list[int] = []

        def keep(step: int, state: torch.Tensor) -> torch.Tensor:
            steps.append(step)
            return state

        mixer = _Mixer(x["weight"])
        with delta_kernel_taps({mixer: (DeltaTap("state", edit_state=keep),)}):
            kept = mixer(x["mixed"], x["g"], x["beta"])
        assert steps == list(range(L))
        assert (kept - reference["out"]).abs().max().item() <= DELTA_BAND
        captured: dict[str, list[torch.Tensor]] = {}
        taps = {
            mixer: (
                DeltaTap("state", edit_state=lambda _t, s: torch.zeros_like(s)),
                DeltaTap("kv_mem", read=_recorder(captured, "kv_mem")),
            )
        }
        with delta_kernel_taps(taps):
            forgetful = mixer(x["mixed"], x["g"], x["beta"])
        assert not torch.equal(forgetful, kept)
        # every step decays a zero state: nothing is recalled
        assert torch.equal(captured["kv_mem"][0], torch.zeros(B, L, H, DV))


@pytest.mark.property
class TestThePerStepSlotsUnderContextParallelism:
    @_SETTINGS
    @given(
        cp=st.sampled_from((2, 3)),
        seed=st.integers(0, 2**32 - 1),
        write=st.booleans(),
        schedule=ps.schedules(),
    )
    @example(cp=2, seed=0, write=True, schedule=[])
    def test_the_deferred_faces_stitch_into_the_unsplit_recurrence(
        self, cp: int, seed: int, write: bool, schedule: Schedule
    ) -> None:
        """Every rank's per-step faces start from the state it received, so
        stitched along the positions they are the world-1 loop's; a write's
        steps are the frame's positions (the ``offset``), not the chunk's;
        the gathered output is the unsplit one within the band."""
        (reference,) = SimulatedWorld(groups_for(1), world=1, schedule=[]).run(
            _state_program(seed, write=write)
        )
        world = SimulatedWorld(groups_for(cp, context=cp), world=cp, schedule=schedule)
        results = world.run(_state_program(seed, write=write))
        assert _globals_are_the_originals()
        chunks = sequence_chunks(L, cp)
        for slot in STATE_SLOTS:
            stitched = torch.cat([got["faces"][slot] for got in results], dim=1)
            worst = (stitched - reference["faces"][slot]).abs().max().item()
            assert worst <= STATE_BAND, (slot, worst)
        for rank, got in enumerate(results):
            worst = (got["out"] - reference["out"]).abs().max().item()
            assert worst <= DELTA_BAND, (rank, worst)
            positions = list(range(chunks[rank].start, chunks[rank].stop))
            assert got["steps"] == (positions if write else [])
