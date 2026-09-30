"""``train.seed`` reaches the ``subspace`` init (spec §2.11, §8).

The bug this guards: ``Subspace.__init__`` took a ``seed`` and drew its
initial rotation from a *local* ``torch.Generator``, but ``_build_stage``
never passed one — so every subspace in every document initialised from
seed 0, and because the generator was local the ``torch.manual_seed(seed)``
at train-loop entry could not reach it either. A ``{"sweep": [0,1,2]}`` on
``train.seed`` therefore produced three fits from the *same* starting
rotation. It looked fine from the outside: the batch order does vary with
the seed (``order_rng``), so the fits were not bit-identical — just launched
from one point, which is the dominant source of run-to-run variance for a
DAS fit. Anyone reading such a sweep as evidence of stability was getting a
much weaker guarantee than they thought.

Two traps the fix has to clear, both pinned below:

* the init must **not** move to the global RNG. ``build_stack`` also runs on
  apply/inference documents that have no ``train`` block and never call
  ``torch.manual_seed``, and stages are cached, so a global-RNG init would
  make a rotation depend on how many stages happened to be built first.
* the stage cache is keyed by **name alone** and can be injected from
  outside, so a cache shared across two points would hand the seed-0 stage
  to seed 1 and swallow the fix. The end-to-end guard is
  ``test_run_corpus.py::test_08_seed_sweep_fits_three_genuinely_different_rotations``;
  here we pin the refusal that makes such a sharing loud instead of silent.

``subspace`` is the only kind with a random init — ``gate`` starts at zeros
and every other kind loads its tensors from a file — which the gate test at
the bottom keeps honest.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.executor import document_seed
from causalab.neural.shared.featurizers import Gate, build_stack
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import FeaturizerSpec, parse_document

from tests.neural.engines.pytorch_hooks._drive import executor_for
from tests.neural.engines.pytorch_hooks.test_train import (
    ANSWERS,
    BASES,
    COUNTERFACTUALS,
    das_doc,
)
from tests.protocol._docs import in_order


pytestmark = pytest.mark.unit

WIDTH = 16
K = 4

SPECS: dict[str, FeaturizerSpec] = {
    "rot": FeaturizerSpec(kind="subspace", k=K, parametrization="cayley"),
    "gate": FeaturizerSpec(kind="gate"),
}


def _no_tensors(path: str) -> dict[str, torch.Tensor]:
    raise KeyError(path)


def _rotation(seed: int | None = None, name: str = "rot") -> torch.Tensor:
    kwargs = {} if seed is None else {"seed": seed}
    stack = build_stack(
        name,
        SPECS,
        width=WIDTH,
        load_tensors=_no_tensors,
        stage_cache={},
        **kwargs,  # type: ignore[arg-type]
    )
    return stack.stages[0].slot_params()["weight"].detach().clone()


def _subspace_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Distance between the *column spaces*, ``‖QaQaᵀ − QbQbᵀ‖_F`` — invariant
    to the arbitrary choice of basis within each frame, unlike an elementwise
    diff. 0 is the same subspace; ``√(2k)`` ≈ 2.83 here is orthogonal ones."""
    return float((a @ a.T - b @ b.T).norm())


# --------------------------------------------------------------------- #
# the seed reaches the init at all
# --------------------------------------------------------------------- #


def test_different_seeds_give_different_initial_rotations() -> None:
    """The headline: this is what a ``{"sweep": [0,1,2]}`` on ``train.seed``
    is buying, and what it silently did not buy."""
    rotations = [_rotation(seed) for seed in (0, 1, 2)]
    for i, a in enumerate(rotations):
        for b in rotations[i + 1 :]:
            assert not torch.equal(a, b)
            # not merely different floats — a genuinely different subspace
            assert _subspace_distance(a, b) > 1.0


def test_the_same_seed_reproduces_bit_identically() -> None:
    """The other half of a usable seed: differing is worthless without
    repeating. Bit-identity, not ``allclose`` — a fit is only comparable
    across runs if the starting point is exact."""
    assert torch.equal(_rotation(7), _rotation(7))


def test_the_init_seed_defaults_to_zero() -> None:
    """A document with no ``train`` block has no seed to name, so its
    featurizer inits are pinned at 0 — explicit, so an apply document and a
    ``train.seed: 0`` fit build the same rotation."""
    assert torch.equal(_rotation(None), _rotation(0))


def test_the_init_ignores_the_global_rng() -> None:
    """Deliberately *not* the fix: the local generator is what keeps a build
    reproducible on apply paths (no ``torch.manual_seed`` there) and
    independent of how many stages were constructed before it."""
    torch.manual_seed(0)
    first = _rotation(3)
    torch.manual_seed(999_999)
    torch.randn(128)  # advance the global stream for good measure
    assert torch.equal(first, _rotation(3))


# --------------------------------------------------------------------- #
# the seed a document implies
# --------------------------------------------------------------------- #


def test_document_seed_reads_train_seed() -> None:
    for seed in (0, 1, 42):
        doc = parse_document(in_order(das_doc(seed=seed)))
        assert document_seed(doc) == seed


def test_document_seed_is_zero_without_a_train_block() -> None:
    raw = das_doc(seed=5)
    del raw["method"]["train"]
    raw["method"]["save"] = [
        entry for entry in raw["method"]["save"] if entry.get("value") != "rot"
    ]
    assert document_seed(parse_document(in_order(raw))) == 0


def _executor_for_seed(bundle, seed: int):
    return executor_for(
        das_doc(seed=seed),
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
    )


def test_the_executor_initialises_from_the_documents_seed(llama_bundle) -> None:
    """The end of the wire the bug broke: ``train.seed`` → ``executor.seed``
    → the rotation the fit starts from. Nothing is trained here — the point
    is the *starting* point, which was identical across seeds."""

    def initial(seed: int) -> torch.Tensor:
        executor = _executor_for_seed(llama_bundle, seed)
        assert executor.seed == seed
        return executor.stage("rot").slot_params()["weight"].detach().clone()

    zero, one = initial(0), initial(1)
    assert not torch.equal(zero, one)
    assert _subspace_distance(zero, one) > 1.0
    assert torch.equal(zero, initial(0))


# --------------------------------------------------------------------- #
# trap 1: the stage cache is keyed by name alone
# --------------------------------------------------------------------- #


def test_a_stage_cache_refuses_a_second_seed() -> None:
    """A cache is one point's. Reusing it under a second seed would hand
    back the first seed's rotation and quietly undo the fix, so it refuses
    — the same rule, and the same P2, as the one-featurizer-one-width check
    that already lives beside it."""
    cache: dict[str, object] = {}
    build_stack(
        "rot", SPECS, width=WIDTH, load_tensors=_no_tensors, stage_cache=cache, seed=0
    )
    with pytest.raises(ProtocolError) as excinfo:
        build_stack(
            "rot",
            SPECS,
            width=WIDTH,
            load_tensors=_no_tensors,
            stage_cache=cache,
            seed=1,
        )
    assert "one featurizer, one seed" in str(excinfo.value)


def test_a_stage_cache_still_shares_one_stage_within_a_seed() -> None:
    """The refusal must not break what the cache is *for*: one stage
    instance per name, so training a featurizer updates every use site."""
    cache: dict[str, object] = {}
    first = build_stack(
        "rot", SPECS, width=WIDTH, load_tensors=_no_tensors, stage_cache=cache, seed=4
    ).stages[0]
    second = build_stack(
        "rot", SPECS, width=WIDTH, load_tensors=_no_tensors, stage_cache=cache, seed=4
    ).stages[0]
    assert first is second


def test_two_points_do_not_share_a_stage_cache(llama_bundle) -> None:
    """Why a seed sweep works at all: an executor with no ``stage_cache``
    handed in starts empty, and the engine builds one executor per point
    (``PytorchHooksEngine._execute_point``), so the seed-0 stage is never
    handed to seed 1. Were that ever hoisted, the refusal above turns it
    into a P2 rather than a silently shared rotation."""
    zero = _executor_for_seed(llama_bundle, 0)
    one = _executor_for_seed(llama_bundle, 1)
    assert zero.stage_cache is not one.stage_cache
    assert not torch.equal(
        zero.stage("rot").slot_params()["weight"].detach(),
        one.stage("rot").slot_params()["weight"].detach(),
    )


# --------------------------------------------------------------------- #
# trap 2: the other stages
# --------------------------------------------------------------------- #


def test_the_gate_has_no_seed_to_thread() -> None:
    """``Gate`` initialises ``θ`` to zeros, so there is no draw for a seed to
    influence — the reason ``build_stack`` passes its seed only to
    ``subspace``. If this ever starts failing, the gate grew a random init
    and needs the same threading."""
    gate = build_stack(
        "gate", SPECS, width=WIDTH, load_tensors=_no_tensors, stage_cache={}, seed=11
    ).stages[0]
    assert isinstance(gate, Gate)
    assert torch.equal(gate.theta.detach(), torch.zeros(WIDTH))


@pytest.mark.parametrize("kind", ["identity", "subspace", "gate"])
def test_only_subspace_is_seed_sensitive(kind: str) -> None:
    """The whole seed surface, kind by kind: of the kinds a spec can build
    without a file (``pca``/``standardize``/``sae``/an applied ``subspace``
    fit all arrive from ``load_tensors``), only ``subspace`` may respond to
    the seed. A new random-init kind trips this and needs the threading."""
    specs = {
        "identity": FeaturizerSpec(kind="identity"),
        "subspace": FeaturizerSpec(kind="subspace", k=K, parametrization="cayley"),
        "gate": FeaturizerSpec(kind="gate"),
    }
    tensors = {
        seed: [
            t.detach().clone()
            for t in build_stack(
                kind,
                specs,
                width=WIDTH,
                load_tensors=_no_tensors,
                stage_cache={},
                seed=seed,
            )
            .stages[0]
            .slot_params()
            .values()
        ]
        for seed in (0, 1)
    }
    responded = any(not torch.equal(a, b) for a, b in zip(tensors[0], tensors[1]))
    assert responded == (kind == "subspace")


# --------------------------------------------------------------------- #
# an authored `seed`: the random rank-k control (§2.5)
# --------------------------------------------------------------------- #


def test_an_authored_seed_wins_over_the_documents_seed() -> None:
    """`seed` on a `subspace` spec is what makes an UNTRAINED subspace a
    sweepable random basis. Without it the only seed is `train.seed`, so a
    document with nothing to train had no way to vary the draw — which is
    why every study that wanted a matched-k random control built the draws by
    hand instead of sweeping one document.
    """
    specs = {
        "rot": FeaturizerSpec(kind="subspace", k=K, parametrization="cayley", seed=5)
    }

    def rotation(document_seed: int) -> torch.Tensor:
        stack = build_stack(
            "rot",
            specs,
            width=WIDTH,
            load_tensors=_no_tensors,
            stage_cache={},
            seed=document_seed,
        )
        return stack.stages[0].slot_params()["weight"].detach().clone()

    # the document seed is ignored entirely: the spec named the draw
    assert torch.equal(rotation(0), rotation(3))
    assert torch.equal(rotation(0), _rotation(5))


def test_authored_seeds_are_genuinely_different_bases() -> None:
    """A control distribution of one draw is an anecdote. Three authored
    seeds must be three different subspaces, or the spread the control
    reports is not a spread."""
    rotations = []
    for seed in (0, 1, 2):
        specs = {
            "rot": FeaturizerSpec(
                kind="subspace", k=K, parametrization="cayley", seed=seed
            )
        }
        stack = build_stack(
            "rot", specs, width=WIDTH, load_tensors=_no_tensors, stage_cache={}
        )
        rotations.append(stack.stages[0].slot_params()["weight"].detach().clone())
    for i, a in enumerate(rotations):
        for b in rotations[i + 1 :]:
            assert _subspace_distance(a, b) > 1.0


def test_an_authored_seed_still_refuses_a_shared_stage_cache() -> None:
    """The cache is keyed by name alone, so the seed check has to compare
    against the *effective* seed — otherwise a swept control would hand
    seed 0's basis to every draw and score three copies of one number."""
    cache: dict = {}
    for seed in (0, 1):
        specs = {
            "rot": FeaturizerSpec(
                kind="subspace", k=K, parametrization="cayley", seed=seed
            )
        }
        if seed == 0:
            build_stack(
                "rot", specs, width=WIDTH, load_tensors=_no_tensors, stage_cache=cache
            )
            continue
        with pytest.raises(ProtocolError, match="one featurizer, one seed"):
            build_stack(
                "rot", specs, width=WIDTH, load_tensors=_no_tensors, stage_cache=cache
            )


def test_a_seed_on_a_loaded_featurizer_is_refused() -> None:
    """A loaded featurizer's weights are its bytes; it draws nothing, so a
    seed there would describe something that never happens."""
    from causalab.protocol.rules.errors import ParseError
    from causalab.protocol.schema import parse_document

    doc = in_order(
        {
            **das_doc(),
            "method": {
                **das_doc()["method"],
                "featurizers": {
                    "rot": {
                        "kind": "subspace",
                        "k": 2,
                        "file_path": "rot.safetensors",
                        "seed": 3,
                    }
                },
            },
        }
    )
    with pytest.raises(ParseError, match="a loaded featurizer uses its saved rotation"):
        parse_document(doc)
