"""The real fit under context parallelism on the hybrid tower, simulated
(``docs/model_parallelism.md`` §6.4, §7, §8.4, §10.4): ``run_cohort_training``
on every rank of a ``SimulatedWorld`` at ``cp=2`` over the tiny
``qwen3.5-moe``, each rank its own replicated copy of the model loaded under
the geometry, against the world-1 fit of the same documents.

The DeltaNet handoffs are what this scenario adds to the tiny Llama's
(``test_context_parallel_train.py``): the write sits in the **first chunk**
at the residual entering block 1 (`LAYER`, `WRITE_AT`) and the
loss at the last token in the **second**, and the two blocks above the
write are Gated DeltaNet layers (the fixture's ``layer_types``, pinned
below) — so the featurizer's gradient crosses the chunk boundary through
the recurrent state rank 1 receives from rank 0 and the conv history with
it (``handoff_state`` / ``chunked_conv`` under ``delta_kernel_taps``), as
well as through the last block's KV gather and the logits gathered at the
head. Every rank's manager patches the same four modeling globals while
another rank is parked inside its handoff: the scenario runs at all
because the managers share one dispatch per symbol (``symbol_dispatch.py``);
before it, the second rank's wrappers wrapped the first's and the kernel
call ran the handoff twice, a divergence the simulator refuses — which is
why this fit was covered only by the ``gloo`` smoke
(``test_context_parallel_train_run.py``) until now.

**Rows.** Six tokens each under the MoE tokenizer (no BOS), so every row's
frame is the same six positions and its chunks at ``cp=2`` are ``[0, 3)``
and ``[3, 6)``: the write at position 1 is in the first, the loss at
position 5 in the second. Pinned by a test, since a row of another length
would move the write or the loss into the other chunk silently.

**The band.** After twelve updates the parameters are within
`FIT_BAND` of the world-1 fit (measured `FIT_MEASURED`: the
chunked kernel's blocks fall differently on a split frame, and the KV and
state gradients are sums of per-rank terms) and bit-identical across the
ranks. **The mutation** — the handoffs without their gradient (plain send
and receive in place of the autograd pair) — keeps the forward and drops
the recurrent state's and the conv history's share of the gradient, so
the fit still moves off its seed through the attention path and lands
outside the band. The hybrid-family refusal of ``cp`` (§8.4) lives in
``run`` and ``dry-run`` ahead of the executor; this scenario enters below
it, at ``run_cohort_training``, so no waiver is involved.
"""

# the cohort suite's builders are reused here
# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, example, given, settings
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.engines.pytorch_hooks.train import run_cohort_training
from causalab.neural.shared.parallel import context
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.protocol.parallel import ParallelGeometry, sequence_chunks
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, SimulatedWorld, groups_for

from ._drive import executor_for
from .conftest import TINY_QWEN35_MOE
from .test_context_parallel_train import Outcome, _worst_parameter
from .test_fit_cohort import _campaign, _request, _train_doc, _weights
from .test_train import ANSWERS

# the §7 gradient agreement check on at bit identity (conftest.py)
pytestmark = [
    pytest.mark.property,
    pytest.mark.usefixtures("checked_gradients_simulated"),
]

CP = 2
#: Six epochs of two minibatches: twelve updates per member.
EPOCHS = 6
MEMBERS = (4, 8)
#: The fitted site: the residual entering block 1 (``block_output`` of
#: block 0), below the two remaining DeltaNet blocks and the attention block.
LAYER = 0
#: The write's position (module docstring): the second token, first chunk.
WRITE_AT = 1
#: The subspace width: the fixture's residual is eight wide, and a
#: full-space swap is the same swap under every rotation (the gloo smoke's
#: ``K`` for this fixture).
K = 4
#: Every row is this many tokens under the MoE tokenizer (module docstring).
ROW_TOKENS = 6
BASES = [
    "bright yellow parrots sing early",
    "the tall red barn stands alone",
    "two small green frogs jump high",
    "the old brown dog sleeps well",
]
COUNTERFACTUALS = [
    "my old grey cat sleeps often",
    "his big black horse eats hay",
    "one long dark road winds north",
    "five young birds fly south today",
]
#: The cp=2 fit against the world-1 fit after twelve updates (module
#: docstring): the maximum absolute parameter difference, measured over the
#: seeds below.
FIT_MEASURED = 1.8e-7
#: At least twenty times the measured maximum (the tiny Llama's band, so the
#: two scenarios read alike), two orders below where the mutation lands.
FIT_BAND = 5e-6
assert FIT_BAND >= 20 * FIT_MEASURED


def _doc(k: int) -> dict[str, Any]:
    """The DAS document on the MoE with the write in the first chunk and the
    loss in the second: the read and the swap at `WRITE_AT`, the
    logits at the last token; no eval split (its rows are another length)."""
    raw = _train_doc(k=K, epochs=EPOCHS, layer=LAYER, eval_every=None)
    raw["method"]["train"]["seed"] = k
    raw["model"]["key"] = TINY_QWEN35_MOE
    raw["method"]["reads"]["v_cf"]["pos"] = {"index": WRITE_AT}
    raw["method"]["writes"]["patch"]["pos"] = {"index": WRITE_AT}
    return raw


def _bundle_for(rank: int, cp: int) -> ModelBundle:
    """This rank's replicated copy, loaded under the geometry so the sites
    resolve to ``SequenceSharded`` placements; the plain bundle at world 1."""
    if cp == 1:
        return load_model(TINY_QWEN35_MOE)
    geometry = ParallelGeometry(context=cp)
    return load_model(
        TINY_QWEN35_MOE, sharding=Sharding(geometry=geometry, rank=rank, meshes={})
    )


def _fit_program(
    cp: int, *, lr: float | None = None
) -> Callable[[int, Collective], list[Outcome]]:
    def program(rank: int, c: Collective) -> list[Outcome]:
        bundle = _bundle_for(rank, cp)
        raws = [_doc(k) for k in MEMBERS]
        if lr is not None:
            for raw in raws:
                raw["method"]["train"]["optimizer"]["lr"] = lr
                raw["method"]["train"]["steps"] = {"epochs": 1}
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


def _world(size: int, schedule: Schedule) -> SimulatedWorld:
    return SimulatedWorld(
        groups_for(size, context=size), world=size, schedule=schedule, timeout=600.0
    )


@pytest.fixture(scope="module")
def solo_fit() -> list[Outcome]:
    (outcome,) = _world(1, []).run(_fit_program(1))
    return outcome


@pytest.fixture(scope="module")
def seed_weights() -> list[Outcome]:
    """The parameters before any update: a one-epoch fit at ``lr=0`` (AdamW
    at zero moves nothing) off the same builders."""
    (outcome,) = _world(1, []).run(_fit_program(1, lr=0.0))
    return outcome


class TestTheFixtureIsTheScenario:
    def test_every_row_is_six_tokens_and_the_write_and_loss_sit_in_different_chunks(
        self, qwen35moe_bundle: ModelBundle
    ) -> None:
        tokenizer = qwen35moe_bundle.tokenizer
        for text in BASES + COUNTERFACTUALS:
            assert len(tokenizer(text)["input_ids"]) == ROW_TOKENS, text
        first, last = sequence_chunks(ROW_TOKENS, CP)
        assert WRITE_AT in first and ROW_TOKENS - 1 in last

    def test_the_blocks_above_the_write_include_deltanet_layers(
        self, qwen35moe_bundle: ModelBundle
    ) -> None:
        kinds = list(qwen35moe_bundle.model.config.layer_types)
        assert "linear_attention" in kinds[LAYER + 1 :], kinds


class TestFitAtContextTwoOnTheMoe:
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @settings(
        deadline=None,
        # Each example runs two twelve-update MoE fits; limit the expensive
        # schedule exploration to ten examples.
        max_examples=10,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_the_fit_lands_within_the_band_and_agrees_across_ranks(
        self, solo_fit: list[Outcome], schedule: Schedule
    ) -> None:
        results = _world(CP, schedule).run(_fit_program(CP))
        for a, b in zip(results[0], results[1]):
            assert _worst_parameter(a, b) == 0.0, "the ranks' fits differ"
        for got, reference in zip(results[0], solo_fit):
            worst = _worst_parameter(got, reference)
            assert worst <= FIT_BAND, worst
            assert worst <= 2 * FIT_MEASURED, f"re-measure: {worst}"

    def test_the_world_one_fit_moves_off_its_seed(
        self, solo_fit: list[Outcome], seed_weights: list[Outcome]
    ) -> None:
        for fitted, seed in zip(solo_fit, seed_weights):
            assert _worst_parameter(fitted, seed) > 100 * FIT_BAND

    def test_the_handoffs_without_their_gradient_leave_the_band(
        self,
        solo_fit: list[Outcome],
        seed_weights: list[Outcome],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The mutation (module docstring): the recurrent state and the conv
        history cross raw, so the first chunk's write loses what the second
        chunk's DeltaNet outputs owed it; the attention path still moves the
        fit, but not to the world-1 fit."""

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
            return collective.recv(shape, dtype, device, src, axis).detach()

        monkeypatch.setattr(context, "send_with_grad", plain_send)
        monkeypatch.setattr(context, "recv_with_grad", plain_recv)
        results = _world(CP, 3).run(_fit_program(CP))
        for got, reference, seed in zip(results[0], solo_fit, seed_weights):
            assert _worst_parameter(got, reference) > 100 * FIT_BAND
            assert _worst_parameter(got, seed) > 100 * FIT_BAND
