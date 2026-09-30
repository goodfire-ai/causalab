"""Training under context parallelism through the engine's own seams
(``docs/model_parallelism.md`` §6.4, §7, §8.4, §10.4), under the
``SimulatedWorld`` at drawn schedules: the gradient crosses the chunks at the
three places the forward does.

- **attention**: the keys and values every rank gathers feed *that rank's*
  queries, so the gradient of chunk ``j``'s keys is the sum over every
  rank's queries — the gather's backward is an all-reduce sum and this
  rank's chunk (``gather_reduce_scatter``). Through the real
  ``attention_interface`` dispatcher and the miniature eager function of
  ``test_context_parallel.py``: the gradient of a loss over every rank's
  local output rows w.r.t. its chunk of ``q``, ``k`` and ``v`` equals the
  whole-frame attention's gradient at that chunk within `KV_BAND`
  (measured ``4.8e-7`` over the first twenty data seeds at ``cp ∈ {2, 3}``: the key
  gradient is one GEMM over the query axis in world 1 and a sum of per-rank
  GEMMs here). The mutation — the tap pair's backward (this rank's own
  chunk) in place of the reduce-scatter — drops the other ranks' queries
  from the first chunk's key gradient and leaves the band;
- **the DeltaNet recurrence**: the state rank ``r`` hands to ``r + 1``
  carries a gradient back (``send_with_grad`` / ``recv_with_grad``), so the
  first chunk's inputs see what the later chunks' outputs made of the state.
  The per-step path (``_stepwise``, what a state read or write runs) over
  two chunks has the unsplit loop's gradient **bit for bit** (measured
  ``0.0`` over the first twenty data seeds at ``cp ∈ {2, 3}``, pinned at an fp32 band,
  `STEPWISE_BAND`: the state gradient a rank receives is the very
  term the unsplit loop's next step passes back); the chunked kernel's
  within `DELTA_BAND` (measured ``4.8e-7``: its blocks re-associate
  in backward as in forward). A rank that skips its backward send is a typed refusal
  naming the waiting rank; a handoff without the gradient (plain send and
  receive) zeroes the first chunk's state contribution and leaves the band;
- **the fit**: the real ``run_cohort_training`` on every simulated rank at
  ``cp=2`` over the tiny Llama DAS documents, each rank its own replicated
  copy of the model loaded under the geometry, on rows of one token length
  (eight, so every row's chunks are the frame's), the write at the **first
  word** (``index: 1`` — the first chunk; the BOS at index 0 has the same
  residual on base and counterfactual, so a swap there is a no-op and its
  gradient rounding noise, ``|g| ≈ 1e-10`` measured) and the loss at the
  last token (the second chunk) — so the featurizer's gradient exists only
  across the chunk boundary, through the logits gathered at the head, the
  attention of the second chunk's queries over the first chunk's keys, and
  the write's fragment. After twelve updates the parameters are within
  `FIT_BAND` of the world-1 fit (measured `FIT_MEASURED`) and
  bit-identical across the ranks (the gathered gradient is the same tensor
  on both). The mutation — the tap pair's backward at the KV gather —
  leaves the featurizer no gradient at all: the fit stays at its seed,
  outside the band;
- **the budget**: ``RowBudget`` agrees the probe's ``min`` and the retry's
  ``any`` over the context axis too, since the chunks run the same windows
  in lockstep.

The tiny MoE's fit (the DeltaNet handoffs on the training path) is the
sibling scenario ``test_context_parallel_delta.py``; the ``gloo`` twin of
both is ``test_context_parallel_train_run.py``.
"""

# the cohort suite's builders and the context suite's harness are reused here
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from causalab.neural.engines.pytorch_hooks import attention_interface, delta_interface
from causalab.neural.engines.pytorch_hooks.attention_interface import (
    attention_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.budget import LOCKSTEP_AXES, RowBudget
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training
from causalab.neural.shared.parallel import context
from causalab.neural.shared.parallel.autograd import Handoff, HandoffMismatch
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.context import (
    SequenceFrame,
    activate,
    handoff_state,
    local_causal_mask,
)
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.protocol.parallel import ParallelGeometry, check_context, sequence_chunks
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Abandoned,
    Hang,
    Schedule,
    SimulatedMeter,
    SimulatedWorld,
    groups_for,
    RankFailed,
)

from ._drive import executor_for
from .conftest import TINY_LLAMA
from .test_context_parallel import (
    BATCH,
    DB,
    DH,
    DIM,
    DK,
    DV,
    HEADS,
    SEQ,
    _delta_inputs,
    _kernels,
    _left_padded,
    _Mixer,
    _seeded,
    eager_attention_forward,
)
from .test_fit_cohort import PAIRS, _campaign, _request, _train_doc, _weights
from .test_train import ANSWERS
from .test_train_parallel import READINGS, _bound

# the §7 gradient agreement check on at bit identity (conftest.py): the
# real ``run_cohort_training`` reads the variable once per fit
pytestmark = [
    pytest.mark.property,
    pytest.mark.usefixtures("checked_gradients_simulated"),
]

_KEEP = -1
AXIS = "context"

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _world(size: int, schedule: Schedule, **kwargs: Any) -> SimulatedWorld:
    return SimulatedWorld(
        groups_for(size, context=size), world=size, schedule=schedule, **kwargs
    )


# --------------------------------------------------------------------------- #
# attention: the gradient across chunks through the KV gather
# --------------------------------------------------------------------------- #

#: The fp32 band for a gradient whose key and value terms are one GEMM over
#: the query axis in world 1 and a sum of per-rank GEMMs here (module
#: docstring): measured ``4.8e-7``, pinned at ten times.
KV_BAND = 5e-6


def _attention_grad_program(
    seed: int,
) -> Callable[[int, Collective], dict[str, torch.Tensor]]:
    """Each rank: its chunk of ``q``, ``k``, ``v`` as leaves, the real
    dispatcher under the frame, and *its own* loss — a fixed weighting of
    its output rows — so the ranks' downstreams differ, as attention's do."""
    q = _seeded((BATCH, HEADS, SEQ, DIM), seed)
    k = _seeded((BATCH, HEADS, SEQ, DIM), seed + 1)
    v = _seeded((BATCH, HEADS, SEQ, DIM), seed + 2)
    mask = _left_padded(BATCH, SEQ, seed + 3)
    weights = _seeded((BATCH, SEQ, HEADS, DIM), seed + 4)
    scaling = DIM**-0.5

    def program(rank: int, c: Collective) -> dict[str, torch.Tensor]:
        frame = SequenceFrame(c, mask)
        chunk = frame.chunk
        s = slice(chunk.start, chunk.stop)
        local = {
            name: t[:, :, s].clone().requires_grad_(True)
            for name, t in (("q", q), ("k", k), ("v", v))
        }
        local_mask = local_causal_mask(mask[:, s], range(len(chunk)), q.dtype)
        with activate(frame):
            out, _ = ALL_ATTENTION_FUNCTIONS["eager"](
                _Mixer(),
                local["q"],
                local["k"],
                local["v"],
                local_mask,
                scaling=scaling,
            )
        (out * weights[:, s]).sum().backward()
        return {name: t.grad for name, t in local.items()}  # type: ignore[misc]

    program.q, program.k, program.v, program.mask = q, k, v, mask  # type: ignore[attr-defined]
    program.weights, program.scaling = weights, scaling  # type: ignore[attr-defined]
    return program


def _attention_grad_reference(program: Any) -> dict[str, torch.Tensor]:
    leaves = {
        name: t.clone().requires_grad_(True)
        for name, t in (("q", program.q), ("k", program.k), ("v", program.v))
    }
    mask = local_causal_mask(program.mask, range(0, SEQ), program.q.dtype)
    out, _ = eager_attention_forward(
        _Mixer(), leaves["q"], leaves["k"], leaves["v"], mask, program.scaling
    )
    (out * program.weights).sum().backward()
    return {name: t.grad for name, t in leaves.items()}  # type: ignore[misc]


def _run_attention(
    program: Any, world: SimulatedWorld
) -> list[dict[str, torch.Tensor]]:
    mixer_free: dict[int, tuple[Any, ...]] = {_KEEP: ()}
    with attention_interface_taps(mixer_free):
        return world.run(program)


class TestAttentionGradientAcrossChunks:
    @pytest.mark.parametrize("cp", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_each_chunks_gradient_is_the_whole_frames(
        self, cp: int, schedule: Schedule, seed: int
    ) -> None:
        program = _attention_grad_program(seed)
        reference = _attention_grad_reference(program)
        for rank, got in enumerate(_run_attention(program, _world(cp, schedule))):
            chunk = sequence_chunks(SEQ, cp)[rank]
            for name in ("q", "k", "v"):
                worst = (
                    (got[name] - reference[name][:, :, chunk.start : chunk.stop])
                    .abs()
                    .max()
                    .item()
                )
                assert worst <= KV_BAND, (rank, name, worst)

    def test_the_tap_pairs_backward_at_the_kv_gather_leaves_the_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation: the replicated-downstream backward keeps only this
        rank's queries' contribution, so the first chunk's keys — attended
        by the second chunk's queries too — get a wrong gradient."""

        def own_chunk(
            frame: SequenceFrame, key: torch.Tensor, value: torch.Tensor, dtype: Any
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                frame.gather(key, 2),
                frame.gather(value, 2),
                frame.attention_mask(dtype),
            )

        monkeypatch.setattr(attention_interface, "_gather_kv", own_chunk)
        program = _attention_grad_program(0)
        reference = _attention_grad_reference(program)
        got = _run_attention(program, _world(2, 0))
        first = sequence_chunks(SEQ, 2)[0]
        worst = (
            (got[0]["k"] - reference["k"][:, :, first.start : first.stop]).abs().max()
        )
        assert worst.item() > 100 * KV_BAND, worst


# --------------------------------------------------------------------------- #
# the DeltaNet recurrence: the state gradient handed back
# --------------------------------------------------------------------------- #

#: The per-step path's gradient over chunks against the unsplit loop's
#: (module docstring): measured ``0.0`` — bit for bit — and pinned at an
#: fp32 band rather than asserted exact, since a BLAS may block the
#: per-step products differently on another machine.
STEPWISE_BAND = 1e-6
#: The chunked kernel's gradient over chunks: its 64-position blocks fall
#: differently and its sums re-associate in forward and backward alike;
#: measured ``4.8e-7``, pinned at fifty times.
DELTA_BAND = 2.5e-5


def _delta_grad_program(
    seed: int, *, stepwise: bool
) -> Callable[[int, Collective], dict[str, torch.Tensor]]:
    """Each rank holds the whole inputs as leaves and runs its chunk after
    the rank below; the loss is a fixed weighting of the **gathered** output
    — the same scalar on every rank — so each leaf's gradient is the unsplit
    gradient at this rank's chunk and zero elsewhere."""
    chunk_kernel, recurrent, l2norm = _kernels()
    x = _delta_inputs(seed)
    mask = torch.ones(DB, x["q"].shape[1], dtype=torch.long)
    weights = _seeded((DB, x["q"].shape[1], DH, DV), seed + 9)

    def program(rank: int, c: Collective) -> dict[str, torch.Tensor]:
        frame = SequenceFrame(c, mask)
        leaves = {name: t.clone().requires_grad_(True) for name, t in x.items()}
        s = slice(frame.chunk.start, frame.chunk.stop)
        q, k, v, g, beta = (leaves[name][:, s] for name in ("q", "k", "v", "g", "beta"))

        def run(initial: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            if stepwise:
                out, final, *_ = delta_interface._stepwise(
                    recurrent, l2norm, q, k, v, g, beta, initial, True
                )
                return out, final
            return chunk_kernel(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        out, _ = handoff_state(
            frame,
            run,
            shape=(DB, DH, DK, DV),
            dtype=torch.float32,
            device=v.device,
            link=v,
        )
        (frame.gather(out, 1) * weights).sum().backward()
        return {name: t.grad for name, t in leaves.items()}  # type: ignore[misc]

    program.inputs, program.weights = x, weights  # type: ignore[attr-defined]
    return program


def _delta_grad_reference(program: Any, *, stepwise: bool) -> dict[str, torch.Tensor]:
    chunk_kernel, recurrent, l2norm = _kernels()
    leaves = {
        name: t.clone().requires_grad_(True) for name, t in program.inputs.items()
    }
    q, k, v, g, beta = (leaves[name] for name in ("q", "k", "v", "g", "beta"))
    if stepwise:
        out, *_ = delta_interface._stepwise(
            recurrent, l2norm, q, k, v, g, beta, None, True
        )
    else:
        out, _ = chunk_kernel(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    (out * program.weights).sum().backward()
    return {name: t.grad for name, t in leaves.items()}  # type: ignore[misc]


def _worst_at_chunk(
    got: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], chunk: range
) -> float:
    worst = 0.0
    for name in got:
        s = slice(chunk.start, chunk.stop)
        worst = max(worst, (got[name][:, s] - reference[name][:, s]).abs().max().item())
    return worst


class TestDeltaHandoffGradient:
    @pytest.mark.parametrize("cp", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_stepwise_gradient_over_chunks_is_the_unsplit_loops(
        self, cp: int, schedule: Schedule, seed: int
    ) -> None:
        program = _delta_grad_program(seed, stepwise=True)
        reference = _delta_grad_reference(program, stepwise=True)
        length = program.inputs["q"].shape[1]
        for rank, got in enumerate(_world(cp, schedule).run(program)):
            chunk = sequence_chunks(length, cp)[rank]
            worst = _worst_at_chunk(got, reference, chunk)
            assert worst <= STEPWISE_BAND, (rank, worst)
            # the other chunks' positions never entered this rank's graph
            for name, grad in got.items():
                others = torch.ones(length, dtype=torch.bool)
                others[chunk.start : chunk.stop] = False
                assert torch.equal(
                    grad[:, others], torch.zeros_like(grad[:, others])
                ), name

    @pytest.mark.parametrize("cp", [2, 3])
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_the_chunked_kernels_gradient_over_chunks_is_within_the_band(
        self, cp: int, schedule: Schedule, seed: int
    ) -> None:
        program = _delta_grad_program(seed, stepwise=False)
        reference = _delta_grad_reference(program, stepwise=False)
        length = program.inputs["q"].shape[1]
        for rank, got in enumerate(_world(cp, schedule).run(program)):
            chunk = sequence_chunks(length, cp)[rank]
            worst = _worst_at_chunk(got, reference, chunk)
            assert worst <= DELTA_BAND, (rank, worst)

    def test_a_rank_that_skips_its_backward_send_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank 1 receives the state on a link carrying no gradient: the
        protocol agrees (``IF_REQUIRED`` on both chunks) but the two ends
        read different facts — the §3 branch the agreement cannot see — so
        its backward sends nothing and rank 0 waits at the send's backward
        receive, which the simulator refuses naming the parked rank."""
        real = context.recv_with_grad

        def detached_on_rank_one(
            link: torch.Tensor,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: torch.device,
            src: int,
            axis: Any,
            collective: Collective,
            **kwargs: Any,
        ) -> torch.Tensor:
            if collective.rank(axis) == 1:
                link = link.detach()
            return real(link, shape, dtype, device, src, axis, collective, **kwargs)

        monkeypatch.setattr(context, "recv_with_grad", detached_on_rank_one)
        with pytest.raises((Hang, Abandoned)) as err:
            _world(2, 0).run(_delta_grad_program(0, stepwise=True))
        assert "rank 0" in str(err.value) and "recv" in str(err.value)

    def test_a_sender_on_the_pipelines_protocol_is_refused_on_both_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The protocol mismatch that deadlocked (``autograd.Handoff``): a
        handoff sent under ``ALWAYS`` while the receiver reads the kernel's
        value (``IF_REQUIRED``) would leave rank 0 waiting in backward for a
        gradient rank 1 never sends when the value requires none; both chunks
        now refuse by name at the forward's handoff, and the simulator
        reports a rank failure, not a hang."""
        real = context.send_with_grad

        def always(
            tensor: torch.Tensor,
            dst: int,
            axis: Any,
            collective: Collective,
            **kwargs: Any,
        ) -> torch.Tensor:
            kwargs["handoff"] = Handoff.ALWAYS
            return real(tensor, dst, axis, collective, **kwargs)

        monkeypatch.setattr(context, "send_with_grad", always)
        with pytest.raises(RankFailed) as err:
            _world(2, 0).run(_delta_grad_program(0, stepwise=True))
        cause = err.value.__cause__
        assert isinstance(cause, HandoffMismatch)
        assert "context handoff chunk 0 → chunk 1" in str(cause)
        assert "ALWAYS" in str(cause) and "IF_REQUIRED" in str(cause)

    def test_a_handoff_without_the_gradient_zeroes_the_first_chunks_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation: plain send and receive on every rank (no hang — the
        collective sequences still agree) drop the state gradient, so the
        first chunk's inputs lose what the second chunk's outputs owed them."""

        def plain_send(
            tensor: torch.Tensor, dst: int, axis: Any, collective: Collective
        ) -> torch.Tensor:
            collective.send(tensor.contiguous(), dst, axis)
            return tensor

        def plain_recv(
            link: torch.Tensor,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            device: torch.device,
            src: int,
            axis: Any,
            collective: Collective,
        ) -> torch.Tensor:
            # detached, as a real backend's receive is (the simulator hands
            # over the sender's tensor object, graph attached)
            return collective.recv(shape, dtype, device, src, axis).detach()

        monkeypatch.setattr(context, "send_with_grad", plain_send)
        monkeypatch.setattr(context, "recv_with_grad", plain_recv)
        program = _delta_grad_program(0, stepwise=True)
        reference = _delta_grad_reference(program, stepwise=True)
        got = _world(2, 0).run(program)
        length = program.inputs["q"].shape[1]
        first, last = sequence_chunks(length, 2)
        assert _worst_at_chunk(got[0], reference, first) > 100 * STEPWISE_BAND
        # the last chunk owes nothing to a later one: its gradient is intact
        assert _worst_at_chunk(got[1], reference, last) <= STEPWISE_BAND


# --------------------------------------------------------------------------- #
# the budget agrees over the context axis
# --------------------------------------------------------------------------- #


class TestBudgetAgreesOverContext:
    def test_the_lockstep_axes_name_the_context_and_pipeline_groups(self) -> None:
        assert set(LOCKSTEP_AXES) >= {"model", "context", "pipeline"}
        assert RowBudget.axes == LOCKSTEP_AXES

    def test_readings_that_differ_by_chunk_give_one_bound_on_every_rank(self) -> None:
        meter = SimulatedMeter(READINGS)

        def program(rank: int, c: Collective) -> int | None:
            budget = RowBudget.of(None, meter.for_rank(rank), collective=c)
            budget.run(PAIRS, lambda: None, unit=PAIRS)
            return budget.bound

        bounds = _world(2, 4).run(program)
        expected = min(_bound(READINGS[0][0]), _bound(READINGS[1][0]))
        assert bounds == [expected, expected]

    def test_one_chunks_out_of_memory_is_every_chunks(self) -> None:
        def program(rank: int, c: Collective) -> bool:
            budget = RowBudget.of(None, None, collective=c)
            return budget.out_of_memory(rank == 1)

        assert _world(2, 5).run(program) == [True, True]

    def test_a_train_document_is_served_under_context_parallelism(self) -> None:
        assert check_context(ParallelGeometry(context=2), decodes=False) == ()


# --------------------------------------------------------------------------- #
# the real fit at a simulated cp=2 on the tiny Llama
# --------------------------------------------------------------------------- #

CP = 2
#: Six epochs of two minibatches: twelve updates per member.
EPOCHS = 6
MEMBERS = (4, 8)
#: The fitted layer: the first block, so that the write at the first word
#: reaches the loss at the last token through the second block's attention.
LAYER = 0
#: The write's position: the first word after the BOS (module docstring).
WRITE_AT = 1
#: Training rows of exactly eight tokens under the tiny Llama's tokenizer
#: (BOS and seven words), so every row's padded frame is the same eight
#: positions and its chunks at ``cp=2`` are ``[0, 4)`` and ``[4, 8)``: the
#: write at position 1 is in the first, the loss at position 7 in the second.
BASES = [
    "bright yellow parrots sing early",
    "seven broken clocks tick wrongly",
    "warm quiet valleys rest gently",
    "the tall red barn stands alone",
]
COUNTERFACTUALS = [
    "my old grey cat sleeps often",
    "a small blue boat drifts slowly",
    "his big black horse eats hay",
    "one long dark road winds north",
]
#: The cp=2 fit against the world-1 fit after twelve updates (module
#: docstring): the maximum absolute parameter difference, measured over
#: twenty schedules (a schedule moves no arithmetic).
FIT_MEASURED = 2.4e-7
#: Twenty times the measured maximum; the world-1 fit moves its parameters
#: by ``2e-3`` and more, and the mutation leaves them at the seed.
FIT_BAND = 5e-6
assert FIT_BAND >= 20 * FIT_MEASURED

Outcome = tuple[int | None, int, dict[str, torch.Tensor]]


def _split_doc(k: int) -> dict[str, Any]:
    """The DAS document with the write in the first chunk and the loss in
    the second: the read and the swap at the first word (`WRITE_AT`),
    the logits at the last token."""
    raw = _train_doc(k=k, epochs=EPOCHS, layer=LAYER)
    raw["method"]["reads"]["v_cf"]["pos"] = {"index": WRITE_AT}
    raw["method"]["writes"]["patch"]["pos"] = {"index": WRITE_AT}
    return raw


def _bundle_for(rank: int, cp: int) -> ModelBundle:
    """This rank's replicated copy of the model, loaded under the geometry so
    the sites resolve to ``SequenceSharded`` placements (the loader keyed by
    ``(geometry, rank)``); the plain bundle at world 1."""
    if cp == 1:
        return load_model(TINY_LLAMA)
    geometry = ParallelGeometry(context=cp)
    return load_model(
        TINY_LLAMA, sharding=Sharding(geometry=geometry, rank=rank, meshes={})
    )


def _fit_program(cp: int) -> Callable[[int, Collective], list[Outcome]]:
    def program(rank: int, c: Collective) -> list[Outcome]:
        bundle = _bundle_for(rank, cp)
        raws = [_split_doc(k) for k in MEMBERS]
        _docs, handles = _campaign(raws)
        executors = [
            executor_for(
                raw,
                bundle,
                base_texts=BASES,
                counterfactual_texts=COUNTERFACTUALS,
                extra_columns={"label": ANSWERS},
                interning=handle,
            )
            for raw, handle in zip(raws, handles)
        ]
        for executor in executors:
            executor.fragments = Fragments(c)
        outcomes = run_cohort_training(
            [ex.doc for ex in executors], executors, _request(), meter=None
        )
        return [(o.fit_rows, o.fit_rows_shrinks, _weights(o)) for o in outcomes]

    return program


def _worst_parameter(a: Outcome, b: Outcome) -> float:
    assert a[0] == b[0] and a[1] == b[1]
    assert set(a[2]) == set(b[2])
    return max((a[2][n].double() - b[2][n].double()).abs().max().item() for n in a[2])


@pytest.fixture(scope="module")
def solo_fit() -> list[Outcome]:
    (outcome,) = SimulatedWorld(groups_for(1), world=1, schedule=0, timeout=600.0).run(
        _fit_program(1)
    )
    return outcome


@pytest.fixture(scope="module")
def seed_weights() -> list[Outcome]:
    """The parameters before any update, read off the same builders through
    a fit whose learning rate is zero (AdamW at ``lr=0`` moves nothing) —
    the reference the mutation is held to."""

    def program(rank: int, c: Collective) -> list[Outcome]:
        bundle = _bundle_for(rank, 1)
        raws = [_split_doc(k) for k in MEMBERS]
        for raw in raws:
            raw["method"]["train"]["optimizer"]["lr"] = 0.0
            raw["method"]["train"]["steps"] = {"epochs": 1}
            raw["method"]["train"].pop("eval", None)
        _docs, handles = _campaign(raws)
        executors = [
            executor_for(
                raw,
                bundle,
                base_texts=BASES,
                counterfactual_texts=COUNTERFACTUALS,
                extra_columns={"label": ANSWERS},
                interning=handle,
            )
            for raw, handle in zip(raws, handles)
        ]
        outcomes = run_cohort_training(
            [ex.doc for ex in executors], executors, _request(), meter=None
        )
        return [(o.fit_rows, o.fit_rows_shrinks, _weights(o)) for o in outcomes]

    (outcome,) = SimulatedWorld(groups_for(1), world=1, schedule=0, timeout=600.0).run(
        program
    )
    return outcome


class TestFitAtContextTwo:
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_the_fit_lands_within_the_band_and_agrees_across_ranks(
        self, solo_fit: list[Outcome], schedule: Schedule
    ) -> None:
        results = _world(CP, schedule, timeout=600.0).run(_fit_program(CP))
        for a, b in zip(results[0], results[1]):
            assert _worst_parameter(a, b) == 0.0, "the ranks' fits differ"
        for got, reference in zip(results[0], solo_fit):
            worst = _worst_parameter(got, reference)
            assert worst <= FIT_BAND, worst
            assert worst <= 2 * FIT_MEASURED, f"re-measure: {worst}"

    def test_the_world_one_fit_moves_off_its_seed(
        self, solo_fit: list[Outcome], seed_weights: list[Outcome]
    ) -> None:
        """The band means something only if the fit moves: the write at the
        first word reaches the loss through the second block's attention."""
        for fitted, seed in zip(solo_fit, seed_weights):
            assert _worst_parameter(fitted, seed) > 100 * FIT_BAND

    def test_the_tap_pairs_backward_at_the_kv_gather_leaves_the_fit_at_its_seed(
        self,
        solo_fit: list[Outcome],
        seed_weights: list[Outcome],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The mutation (module docstring): with the write in the first chunk
        and the loss in the second, the featurizer's only gradient path is
        the second chunk's queries over the first chunk's keys; the tap
        pair's backward drops it, so no update ever moves the parameters."""

        def own_chunk(
            frame: SequenceFrame, key: torch.Tensor, value: torch.Tensor, dtype: Any
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                frame.gather(key, 2),
                frame.gather(value, 2),
                frame.attention_mask(dtype),
            )

        monkeypatch.setattr(attention_interface, "_gather_kv", own_chunk)
        results = _world(CP, 3, timeout=600.0).run(_fit_program(CP))
        for got, reference, seed in zip(results[0], solo_fit, seed_weights):
            assert _worst_parameter(got, reference) > 100 * FIT_BAND
            assert _worst_parameter(got, seed) == 0.0
