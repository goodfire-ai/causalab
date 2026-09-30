"""A ``subspace`` initialised from a saved basis (spec §2.5 ``init``).

The claim under test is the one the design makes exact: before any training
step, a fit seeded from the first ``k`` principal components reads
``P_kᵀx`` — not approximately, not "a rotation of the same span", the PCA
projection itself — while the trainable surface is the one an unseeded fit
has. torch's orthogonal parametrization is a trivialization ``base · R(A)``,
so the start is set by installing ``[P_k | N]`` as the base with ``R(0) = I``,
and the tests here check both halves: the read is the projection, and the
completed base is a real orthonormal frame drawn from the document seed.

The basis is a genuine ``causalab.analysis.fit_pca`` fit over activations
harvested from the tiny Llama fixture at one site, so the shapes and dtypes
are the ones a workflow hands over, not a stand-in.
"""

from __future__ import annotations

import json

import pytest
import torch

from causalab.analysis import fit_pca
from causalab.io.step_io import read_tensor
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.featurizers import Subspace, build_stack
from causalab.io.tensor_files import TensorBundle
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import FeaturizerSpec

from tests.neural.engines.pytorch_hooks._drive import base_data_section, executor_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA
from tests.protocol._docs import saved
from tests.step_scripts import run_step

WIDTH = 16  # tiny-random Llama's hidden size
TEXTS = [
    "the quick brown fox jumps over",
    "a slow green turtle sleeps deeply",
    "every shiny robot dances tonight",
    "some ancient rivers flow backwards",
    "cold silver mountains echo loudly",
    "bright yellow parrots sing early",
    "seven broken clocks tick wrongly",
    "warm quiet valleys rest gently",
    # a ninth row: centering leaves n - 1 components with variance, so eight
    # principal components need nine rows (fit_pca refuses k > n - 1)
    "nine pale lanterns glow softly",
]
BASIS_PATH = "pca/basis.safetensors"
BASIS_IDENTITY = {"trained_on": "weekdays/train"}


def _harvest_doc() -> dict:
    return {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["acts"]}},
            "sites": {"tgt": {"component": "block_output", "layers": [0]}},
            "reads": {"acts": {"site": "tgt", "pos": {"index": -1}}},
            "save": [saved("acts", "original", "acts.safetensors")],
        },
    }


@pytest.fixture(scope="module")
def basis(tmp_path_factory: pytest.TempPathFactory) -> torch.Tensor:
    """The ``(16, 8)`` principal basis of ``block_output`` L0 at the last
    position over the nine ``TEXTS``, as a harvest and ``fit_pca`` pipeline
    writes it."""
    executor = executor_for(_harvest_doc(), load_model(TINY_LLAMA), base_texts=TEXTS)
    acts = executor.read_value("acts").detach().to(torch.float32)
    assert acts.shape == (len(TEXTS), 1, WIDTH)
    out = tmp_path_factory.mktemp("pca")
    run_step(
        fit_pca,
        {"acts": acts, "k": 8},
        {"weight": out / "basis.safetensors", "spectrum": out / "spectrum.json"},
    )
    weight = read_tensor(out / "basis.safetensors")
    assert weight.shape == (WIDTH, 8)
    return weight


def _loader(weight: torch.Tensor, identity: dict | None = None):
    """A ``load_tensors`` over one in-memory basis bundle, stamped like a
    runner-written one when ``identity`` is given."""
    bundle = TensorBundle(
        tensors={"weight": weight}, entry_coords={}, header=dict(identity or {})
    )
    return (
        lambda path: bundle
        if path == BASIS_PATH
        else (_ for _ in ()).throw(KeyError(path))
    )


def _spec(
    k: int,
    parametrization: str = "cayley",
    init: dict | None = None,
    **extra,
) -> dict[str, FeaturizerSpec]:
    return {
        "rot": FeaturizerSpec(
            kind="subspace",
            k=k,
            parametrization=parametrization,
            init=init if init is not None else {"file_path": BASIS_PATH},
            **extra,
        )
    }


def _stage(
    basis: torch.Tensor, k: int, parametrization: str = "cayley", **kw
) -> Subspace:
    stack = build_stack(
        "rot",
        _spec(k, parametrization),
        width=WIDTH,
        load_tensors=_loader(basis, BASIS_IDENTITY),
        stage_cache={},
        **kw,
    )
    stage = stack.stages[0]
    assert isinstance(stage, Subspace)
    return stage


def _base(stage: Subspace) -> torch.Tensor:
    return stage.parametrizations.weight[0].base


#: The maps torch's ``orthogonal`` implements, which trivialize at a d×d frame
#: that has to be completed around ``P_k``. ``cayley`` is the repo's own
#: low-rank map and trivializes at ``P_k`` itself.
COMPLETED_MAPS = ("matrix_exp", "stiefel")


class TestTheStartIsThePcaProjection:
    pytestmark = pytest.mark.numerical_unit

    @pytest.mark.parametrize("parametrization", ["cayley", "matrix_exp", "stiefel"])
    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_the_featurized_read_is_the_projection_before_training(
        self, basis, k, parametrization
    ):
        """``R(0) = I`` for every map, so the weight *is* ``P_k`` to the bit
        and the read is ``P_kᵀx`` up to the matmul's own rounding."""
        stage = _stage(basis, k, parametrization)
        expected = basis[:, :k]
        assert torch.equal(stage.weight.detach(), expected)
        x = torch.randn(5, WIDTH, generator=torch.Generator().manual_seed(1))
        f, err = stage.featurize(x)
        torch.testing.assert_close(f, x @ expected, atol=1e-6, rtol=0)
        assert err is not None
        torch.testing.assert_close(
            err, x - (x @ expected) @ expected.T, atol=1e-6, rtol=0
        )
        torch.testing.assert_close(stage.inverse(f, err), x, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_the_cayley_base_is_the_projection_itself(self, basis, k):
        """The low-rank ``cayley`` map ([`Cayley`][causalab.neural.shared.featurizers.stages.Cayley]) is a trivialization
        at the ``(d, k)`` start directly — there is no d×d frame to complete,
        so its base is ``P_k`` verbatim and nothing about it is drawn."""
        q0 = _base(_stage(basis, k, "cayley"))
        assert torch.equal(q0, basis[:, :k])

    @pytest.mark.parametrize("parametrization", COMPLETED_MAPS)
    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_the_completed_basis_is_an_orthonormal_frame(
        self, basis, k, parametrization
    ):
        """``Q₀ = [P_k | N]`` orthonormalized: ``Q₀ᵀQ₀ = I`` and its first
        ``k`` columns are ``P_k`` verbatim, so the trained subspace is the
        first ``k`` columns of ``Q₀ · R(A)`` exactly as the design says."""
        q0 = _base(_stage(basis, k, parametrization))
        assert q0.shape == (WIDTH, WIDTH)
        torch.testing.assert_close(q0.T @ q0, torch.eye(WIDTH), atol=1e-5, rtol=0)
        assert torch.equal(q0[:, :k], basis[:, :k])

    @pytest.mark.parametrize("parametrization", COMPLETED_MAPS)
    def test_the_completion_is_drawn_at_the_document_seed(self, basis, parametrization):
        """The complement ``N`` is the seed's draw: same seed, same frame;
        another seed, another complement around the same ``P_k`` — which is
        what a seed sweep of PCA-initialised fits varies."""
        first, again, other = (
            _base(_stage(basis, 4, parametrization, seed=s)) for s in (0, 0, 1)
        )
        assert torch.equal(first, again)
        assert torch.equal(first[:, :4], other[:, :4])
        assert not torch.allclose(first[:, 4:], other[:, 4:])

    @pytest.mark.parametrize("parametrization", ["cayley", *COMPLETED_MAPS])
    def test_the_base_ignores_the_global_rng(self, basis, parametrization):
        torch.manual_seed(0)
        first = _base(_stage(basis, 2, parametrization, seed=3))
        torch.manual_seed(12345)
        torch.randn(64)
        assert torch.equal(first, _base(_stage(basis, 2, parametrization, seed=3)))

    def test_a_wider_basis_seeds_by_its_first_k_columns(self, basis):
        """The basis holds eight components; a rank-2 fit takes the leading
        two — the ones with the most variance, by ``fit_pca``'s ordering —
        and records exactly that."""
        stage = _stage(basis, 2)
        assert torch.equal(stage.weight.detach(), basis[:, :2])
        assert stage.identity_fields()["init_components"] == [0, 1]

    def test_the_fit_can_leave_the_start_and_stays_orthonormal(self, basis):
        """The start is a point on the same manifold the unseeded fit trains
        on, not a frozen projection: one step moves the weight and the
        parametrization keeps it a frame."""
        stage = _stage(basis, 4)
        stage.train()
        x = torch.randn(6, WIDTH, generator=torch.Generator().manual_seed(2))
        optimizer = torch.optim.SGD(
            [p for p in stage.parameters() if p.requires_grad], lr=0.1
        )
        loss = stage.featurize(x)[0].pow(2).sum()
        loss.backward()
        optimizer.step()
        moved = stage.weight.detach()
        assert not torch.equal(moved, basis[:, :4])
        torch.testing.assert_close(moved.T @ moved, torch.eye(4), atol=1e-5, rtol=0)

    def test_without_init_the_start_is_the_seeded_random_frame(self):
        """The unchanged path, pinned to its formula: ``qr(randn(d, k))`` from
        the local generator — to rounding, because torch re-orthonormalizes it
        while completing its own base — and nothing recorded about a start.
        That is what keeps every existing document's fit and stamp where they
        were (the bit-level pins are ``test_featurizer_seed.py``'s)."""
        stage = Subspace(WIDTH, 4, "cayley", seed=7)
        expected = torch.linalg.qr(
            torch.randn(WIDTH, 4, generator=torch.Generator().manual_seed(7))
        )[0]
        torch.testing.assert_close(stage.weight.detach(), expected, atol=1e-6, rtol=0)
        assert stage.identity_fields() == {}


class TestTheRecordAndTheRefusals:
    pytestmark = pytest.mark.unit

    def test_identity_fields_name_the_basis_data_and_the_columns_taken(self, basis):
        """What a bundle saved from this fit stamps beyond the document: the
        data the basis was fitted over and the component indices — so the
        record says where the fit started."""
        fields = _stage(basis, 4).identity_fields()
        assert fields == {
            "init_trained_on": "weekdays/train",
            "init_components": [0, 1, 2, 3],
        }

    def test_a_swept_basis_records_the_selected_entrys_data(self, basis):
        """A fitted rotation reused as a warm start: a swept bundle stamps
        ``trained_on`` per entry, and the record reads it off the entry the
        document selected (§8 — per entry, not per file)."""
        table = {
            f"weight[k={k}]": {
                "slot": "weight",
                "coords": {"k": k},
                "trained_on": f"weekdays/train_k{k}",
            }
            for k in (2, 4)
        }
        bundle = TensorBundle(
            tensors={f"weight[k={k}]": basis[:, :k].contiguous() for k in (2, 4)},
            entry_coords=table,
            header={"trained_on": "weekdays/train", "entries": json.dumps(table)},
        )
        stack = build_stack(
            "rot",
            _spec(4, init={"file_path": BASIS_PATH, "entry": {"k": 4}}),
            width=WIDTH,
            load_tensors=lambda _path: bundle,
            stage_cache={},
        )
        fields = stack.stages[0].identity_fields()
        assert fields["init_trained_on"] == "weekdays/train_k4"
        assert fields["init_components"] == [0, 1, 2, 3]
        assert torch.equal(stack.stages[0].weight, basis[:, :4])

    def test_a_basis_that_is_not_orthonormal_refuses(self, basis):
        """The start is installed as the parametrization's base verbatim, so a
        basis that is not a frame would make every trained weight
        non-orthonormal without a word — refuse instead, naming the deviation."""
        with pytest.raises(ProtocolError, match="not orthonormal"):
            build_stack(
                "rot",
                _spec(2),
                width=WIDTH,
                load_tensors=_loader(basis * 3, BASIS_IDENTITY),
                stage_cache={},
            )
        # a bf16 round trip is already too far from a frame to start from
        rounded = basis.to(torch.bfloat16).to(torch.float32)
        with pytest.raises(ProtocolError, match="not orthonormal"):
            build_stack(
                "rot",
                _spec(2),
                width=WIDTH,
                load_tensors=_loader(rounded, BASIS_IDENTITY),
                stage_cache={},
            )

    def test_fewer_columns_than_k_refuses(self, basis):
        with pytest.raises(
            ProtocolError, match="holds 2 components but the fit needs k=4"
        ):
            build_stack(
                "rot",
                _spec(4),
                width=WIDTH,
                load_tensors=_loader(basis[:, :2].contiguous()),
                stage_cache={},
            )

    def test_another_width_refuses(self, basis):
        wide = torch.cat([basis, basis], dim=0)  # (32, 8): another model's site
        with pytest.raises(
            ProtocolError, match=r"shape \(32, 8\) but the site here is 16"
        ):
            build_stack(
                "rot", _spec(4), width=WIDTH, load_tensors=_loader(wide), stage_cache={}
            )

    def test_a_vector_is_not_a_basis(self, basis):
        with pytest.raises(ProtocolError, match="must be \\(width, ≥ k\\)"):
            build_stack(
                "rot",
                _spec(1),
                width=WIDTH,
                load_tensors=_loader(basis[:, 0].contiguous()),
                stage_cache={},
            )

    def test_the_stage_cache_still_shares_one_seeded_stage(self, basis):
        cache: dict = {}
        loader = _loader(basis, BASIS_IDENTITY)
        first = build_stack(
            "rot", _spec(2), width=WIDTH, load_tensors=loader, stage_cache=cache
        ).stages[0]
        second = build_stack(
            "rot", _spec(2), width=WIDTH, load_tensors=loader, stage_cache=cache
        ).stages[0]
        assert first is second
