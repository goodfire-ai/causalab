"""The memory pre-flight on the model that fits no card
(``docs/model_parallelism.md`` §10.6 "The large model", §11): the
``meta-llama/Llama-3.1-70B`` — 131.42 GiB of bf16 weights — from its real
header census (``tests/golden/parallel_headers_llama70b.json``), against an
H100 80 GB as ``mem_get_info`` reports it, torch-free on the CPU.

The verdicts the golden (``tests/golden/test_parallel_large.py``) rests on,
pinned here so a rule change shows before a GPU is touched:

- **world 1 is refused by name** — no ``tp=1`` / ``pp=1`` run exists, and
  the golden's oracle is a pipeline run, not world 1;
- **``tp=2`` and ``pp=2`` fit their weights but exceed estimated headroom**:
  the shipped rule's
  headroom is 15 % of the *whole* model's bytes (19.71 GiB), not of the
  shard, so a 65.71 GiB stage (``pp=2``) or a 67.67 GiB tensor shard
  (``tp=2``, the vocabulary whole on every rank) estimates 85.4 / 87.4 GiB
  against 79.15 GiB free. The estimate is advisory; the loader admits them.
  The estimated world-4 fits start with ``pp=4`` before ``pp=2,tp=2``;
- ``tp=4``, ``pp=4``, ``tp=8``, ``pp=8`` and ``tp=2,pp=2`` fit on every
  rank — and none of them within a quarter of the card: the rule was
  measured where the whole model sits on the card, and on this model no
  geometry that fits does, so the golden measures the estimate against the
  peak and pins the ratio rather than asserting a 5 % agreement it cannot
  have (§10.6).

``dry-run --parallel`` on a document naming the 70B prints the same
numbers off a header-only Hub cache in a fresh offline interpreter, torch
never imported. The fit set is monotone in the card's capacity (a
hypothesis property).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.checkpoint_census import checkpoint_targets, tree_of
from causalab.protocol.parallel import ONE, ParallelGeometry, check, parse_geometry
from causalab.protocol.parallel_memory import (
    RULE,
    Placement,
    estimate_resident,
    fitting_geometries,
    format_bytes,
    memory_check,
    memory_estimate,
    placement_table,
    whole_bytes,
)
from causalab.protocol.registry import LLAMA_TREE, get_model_info

from tests._helpers.header_census import LLAMA70B_CENSUS, fake_hub_cache, load_census
from tests.protocol.test_cli_parallel import INTERCHANGE
from tests.protocol.test_dry_run import _argv, _offline
from tests.protocol.test_parallel_memory import H100_FREE, H100_TOTAL, _product

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

LARGE = "meta-llama/Llama-3.1-70B"
GIB = 1 << 30
WHOLE_BYTES = 141107412992  # 131.42 GiB

#: The golden's geometries (§10.6 "The large model"), every one a fit.
GOLDEN_GEOMETRIES = ("pp=4", "tp=4", "tp=2,pp=2", "pp=8", "tp=8")


@pytest.fixture(scope="module")
def large_table() -> dict[str, Placement]:
    info = get_model_info(LARGE)
    census = load_census(LLAMA70B_CENSUS)
    elements = {key: _product(shape) for key, (_, shape) in census.items()}
    tree = tree_of(elements, info.num_layers)
    assert tree == LLAMA_TREE.tree
    targets = checkpoint_targets(elements, tree, info.num_layers)
    assert targets is not None and info.parallel_plan is not None
    return placement_table(elements, targets, info.parallel_plan, tree, info)


def _check(table, spelled: str, rank: int = 0) -> str | None:
    return memory_check(
        geometry=parse_geometry(spelled) if spelled else ONE,
        rank=rank,
        device=f"cuda:{rank}",
        dtype="bf16",
        table=table,
        info=get_model_info(LARGE),
        free=H100_FREE,
        total=H100_TOTAL,
    )


def _spelled(geometry: ParallelGeometry) -> str:
    return ",".join(
        f"{axis}={size}"
        for axis, size in (("pp", geometry.pipeline), ("tp", geometry.tensor))
        if size > 1
    )


# --------------------------------------------------------------------------- #
# the model, and the entry it is registered under
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestTheModel:
    def test_the_census_is_the_whole_model_and_agrees_with_the_entry(
        self, large_table
    ) -> None:
        info = get_model_info(LARGE)
        census = load_census(LLAMA70B_CENSUS)
        assert len(census) == 723 and len(large_table) == 723
        assert whole_bytes(large_table, 2) == WHOLE_BYTES
        assert format_bytes(WHOLE_BYTES) == "131.42 GiB"
        vocab, hidden = census["model.embed_tokens.weight"][1]
        assert (vocab, hidden) == (info.vocab_size, info.hidden_size)
        assert census["lm_head.weight"][1] == (vocab, hidden)  # untied
        assert census["model.layers.0.self_attn.q_proj.weight"][1] == (
            info.num_heads * info.head_dim,
            hidden,
        )
        assert census["model.layers.0.self_attn.k_proj.weight"][1] == (
            info.num_kv_heads * info.head_dim,
            hidden,
        )
        assert census["model.layers.0.mlp.gate_proj.weight"][1] == (
            info.intermediate_size,
            hidden,
        )
        assert (
            1 + max(p.layer for p in large_table.values() if p.layer is not None)
            == info.num_layers
            == 80
        )

    def test_the_family_plan_shards_every_projection_and_replicates_the_vocabulary(
        self, large_table
    ) -> None:
        sharded = {name for name, p in large_table.items() if p.axis == "tensor"}
        assert all(".self_attn." in n or ".mlp." in n for n in sharded)
        assert len(sharded) == 7 * 80
        for name in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ):
            assert large_table[name].axis is None
        assert large_table["model.embed_tokens.weight"].home == "first"
        assert large_table["lm_head.weight"].home == "last"
        # the K/V projections shard up to the eight KV heads and no further
        kv = large_table["model.layers.0.self_attn.k_proj.weight"]
        assert kv.shard_limit == 8
        assert kv.shards(parse_geometry("tp=8")) == 8
        assert kv.shards(parse_geometry("tp=16")) == 1

    @pytest.mark.parametrize("spelled", GOLDEN_GEOMETRIES)
    def test_every_golden_geometry_is_acceptable_to_the_geometry_check(
        self, spelled: str
    ) -> None:
        assert check(parse_geometry(spelled), get_model_info(LARGE)) == ()

    def test_the_dense_entry_refuses_the_expert_axis_by_name(self) -> None:
        refusals = check(parse_geometry("ep=2"), get_model_info(LARGE))
        assert refusals and all(r.startswith("--parallel.expert:") for r in refusals)


# --------------------------------------------------------------------------- #
# the verdicts on an 80 GB card
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestVerdicts:
    def test_world_one_is_refused_with_no_estimated_fit_of_the_next_world(
        self, large_table
    ) -> None:
        """``tp=1`` and ``pp=1`` are world 1: the whole model on one card,
        its resident weights exceed the card. Neither world 1 nor world 2
        satisfies the empirical headroom estimate."""
        assert parse_geometry("tp=1") == parse_geometry("pp=1") == ONE
        text = _check(large_table, "")
        assert text is not None
        assert text.startswith(
            "--parallel: world 1 would place 131.42 GiB of bf16 weights on rank 0 "
            "(cuda:0) — an estimated 151.13 GiB at the run's peak"
        )
        assert "no pp/ep/tp geometry of this world or twice its size" in text
        assert "79.15 GiB available of 79.65 GiB" in text

    def test_the_headroom_is_of_the_whole_model_not_of_the_shard(
        self, large_table
    ) -> None:
        """The rule's base is 15 % of the model's bytes on every geometry
        (``parallel_memory.RULE``, §11) — 19.71 GiB here — so the world-2
        shards sit 6–8 GiB over the card, where 15 % of the *shard* would
        have put them 3–4 GiB under it. This estimate remains advisory."""
        headroom = RULE.headroom(WHOLE_BYTES, parse_geometry("tp=2"))
        assert headroom == round(0.15 * WHOLE_BYTES)
        assert format_bytes(headroom) == "19.71 GiB"
        for spelled in ("tp=2", "pp=2"):
            resident = max(estimate_resident(large_table, parse_geometry(spelled), 2))
            assert resident + headroom > H100_FREE
            assert resident + round(0.15 * resident) < H100_FREE  # the other reading

    @pytest.mark.parametrize(
        ("spelled", "resident", "footprint"),
        [("tp=2", "67.67 GiB", "87.38 GiB"), ("pp=2", "65.71 GiB", "85.42 GiB")],
    )
    def test_world_two_is_admitted_despite_its_headroom_estimate(
        self, large_table, spelled: str, resident: str, footprint: str
    ) -> None:
        for rank in (0, 1):
            assert _check(large_table, spelled, rank) is None
            estimate = memory_estimate(
                geometry=parse_geometry(spelled),
                rank=rank,
                dtype="bf16",
                table=large_table,
                info=get_model_info(LARGE),
            )
            assert format_bytes(estimate.resident) == resident
            assert format_bytes(estimate.footprint) == footprint
            assert estimate.resident < H100_FREE < estimate.footprint

    def test_tp_two_holds_more_than_a_pp_two_stage_because_the_vocabulary_is_whole(
        self, large_table
    ) -> None:
        """Under ``tp`` the embedding and the head are whole on every rank
        (§6.1, the vocabulary is never sharded) and every norm is
        replicated; a ``pp=2`` stage holds one of the two vocabulary tables
        and half the norms. The difference is exactly that."""
        tp2 = estimate_resident(large_table, parse_geometry("tp=2"), 2)[0]
        pp2 = estimate_resident(large_table, parse_geometry("pp=2"), 2)
        head = large_table["lm_head.weight"].elements * 2
        norms = sum(
            p.elements * 2
            for name, p in large_table.items()
            if p.axis is None
            and p.home == "layer"
            and p.layer is not None
            and p.layer >= 40
        )
        final_norm = large_table["model.norm.weight"].elements * 2
        assert tp2 - pp2[0] == head + norms + final_norm
        assert format_bytes(head) == "1.96 GiB"
        assert pp2[0] != pp2[1] and format_bytes(pp2[0]) == format_bytes(pp2[1])

    @pytest.mark.parametrize("spelled", GOLDEN_GEOMETRIES)
    def test_every_golden_geometry_fits_on_every_rank(
        self, large_table, spelled: str
    ) -> None:
        world = parse_geometry(spelled).world
        for rank in range(world):
            assert _check(large_table, spelled, rank) is None, (spelled, rank)

    def test_the_fits_at_worlds_four_and_eight_in_the_documented_order(
        self, large_table
    ) -> None:
        """Per world, the fewest tensor ranks first (``pp`` is placement and
        the cheapest axis, §11); the per-rank footprints pinned to the
        hundredth of a GiB. Worlds 1 and 2: nothing."""
        info = get_model_info(LARGE)
        fits = fitting_geometries(
            info, large_table, 2, capacity=H100_FREE, worlds=(1, 2, 4, 8)
        )
        assert [(_spelled(f.geometry), format_bytes(f.footprint)) for f in fits] == [
            ("pp=4", "53.55 GiB"),
            ("pp=2,tp=2", "53.55 GiB"),
            ("tp=4", "55.50 GiB"),
            ("pp=8", "37.61 GiB"),
            ("pp=4,tp=2", "37.61 GiB"),
            ("pp=2,tp=4", "37.61 GiB"),
            ("tp=8", "39.57 GiB"),
        ]
        assert all(f.geometry.expert == 1 for f in fits)  # dense: no expert axis
        resident = {
            spelled: format_bytes(
                max(estimate_resident(large_table, parse_geometry(spelled), 2))
            )
            for spelled in GOLDEN_GEOMETRIES
        }
        assert resident == {
            "pp=4": "33.83 GiB",
            "tp=4": "35.79 GiB",
            "tp=2,pp=2": "33.83 GiB",
            "pp=8": "17.89 GiB",
            "tp=8": "19.85 GiB",
        }

    def test_no_fitting_geometry_runs_near_the_card(self, large_table) -> None:
        """The rule's headroom was measured where the whole model sits on
        the card (§11: within 5 % above 60 GiB); on this model every
        geometry that fits leaves more than a quarter of the card free and
        estimates at least 1.5× its resident weights. So the golden can
        hold the estimate as a **bound** on the measured peak and pin the
        ratio, and cannot hold it within 5 % — that agreement is a property
        of a card that fills, which no fitting geometry of the 70B does."""
        info = get_model_info(LARGE)
        fits = fitting_geometries(
            info, large_table, 2, capacity=H100_FREE, worlds=(4, 8)
        )
        assert fits
        for fit in fits:
            resident = max(estimate_resident(large_table, fit.geometry, 2))
            assert fit.footprint <= 0.75 * H100_FREE, _spelled(fit.geometry)
            assert fit.footprint >= 1.5 * resident, _spelled(fit.geometry)


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #

_CAPACITIES = st.integers(min_value=0, max_value=200 * GIB)


@pytest.mark.property
class TestProperties:
    @_SETTINGS
    @given(a=_CAPACITIES, b=_CAPACITIES)
    def test_the_fit_set_is_monotone_in_the_capacity(self, large_table, a, b) -> None:
        """A larger card fits every geometry a smaller one does, with the
        same footprints: the footprint is the rule's, the capacity only
        selects."""
        low, high = sorted((a, b))
        info = get_model_info(LARGE)
        small = {
            f.geometry: f.footprint
            for f in fitting_geometries(info, large_table, 2, low, worlds=(1, 2, 4, 8))
        }
        big = {
            f.geometry: f.footprint
            for f in fitting_geometries(info, large_table, 2, high, worlds=(1, 2, 4, 8))
        }
        assert set(small) <= set(big)
        assert all(big[g] == small[g] for g in small)
        assert all(footprint <= high for footprint in big.values())

    @_SETTINGS
    @given(capacity=_CAPACITIES)
    def test_estimated_fits_are_a_subset_of_admitted_geometries(
        self, large_table, capacity
    ) -> None:
        info = get_model_info(LARGE)
        fits = {
            f.geometry
            for f in fitting_geometries(info, large_table, 2, capacity, worlds=(4,))
        }
        for spelled in ("pp=4", "tp=4", "pp=2,tp=2"):
            geometry = parse_geometry(spelled)
            accepted = all(
                memory_check(
                    geometry=geometry,
                    rank=rank,
                    device="cuda:0",
                    dtype="bf16",
                    table=large_table,
                    info=info,
                    free=capacity,
                    total=capacity,
                )
                is None
                for rank in range(geometry.world)
            )
            assert accepted == (
                max(estimate_resident(large_table, geometry, 2)) <= capacity
            )
            assert (geometry in fits) == (
                max(RULE.footprint(large_table, geometry, 2)) <= capacity
            )
            if geometry in fits:
                assert accepted, spelled


# --------------------------------------------------------------------------- #
# dry-run, torch-free, on the header-only cache
# --------------------------------------------------------------------------- #


@pytest.fixture
def large_cache(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    fake_hub_cache(root, LARGE, load_census(LLAMA70B_CENSUS), files=30)
    return root


def _dry_run(
    artifacts_root: Path, cache: Path, spelled: str, dtype: str = "bf16"
) -> dict:
    """The corpus interchange document retargeted to the 70B at ``dtype``
    (the corpus spells fp32; the estimate follows the document's dtype)."""
    return _offline(
        _argv(
            INTERCHANGE,
            artifacts_root,
            "--set",
            f"model.key={LARGE}",
            "--set",
            f"model.dtype={dtype}",
            "--parallel",
            spelled,
        ),
        hub_cache=cache,
    )


@pytest.mark.unit
class TestDryRun:
    def test_the_estimate_lines_for_a_fitting_geometry(
        self, artifacts_root: Path, large_cache: Path
    ) -> None:
        result = _dry_run(artifacts_root, large_cache, "tp=4")
        assert result["code"] == 0, result["err"]
        out = result["out"]
        assert "parallel  dp=1,pp=1,cp=1,tp=4,ep=1 (world 4): accepted" in out
        assert (
            "  resident weights (bf16; the model is 131.42 GiB): rank0 35.79 GiB, "
            "rank1 35.79 GiB, rank2 35.79 GiB, rank3 35.79 GiB" in out
        )
        assert (
            "  estimated card footprint: rank0 55.50 GiB, rank1 55.50 GiB, "
            "rank2 55.50 GiB, rank3 55.50 GiB" in out
        )
        assert "  headroom rule: 15% of the model's bytes" in out
        assert "memory: undecided" not in out
        assert not result["torch"], "the dry run imported torch"

    def test_the_estimate_follows_the_documents_dtype(
        self, artifacts_root: Path, large_cache: Path
    ) -> None:
        """The corpus document spells fp32: there the 70B is 262.83 GiB and
        a ``tp=4`` rank would hold 71.58 GiB — nothing of world 4 fits an
        80 GB card in fp32, which the run's pre-flight would refuse."""
        result = _dry_run(artifacts_root, large_cache, "tp=4", dtype="fp32")
        assert result["code"] == 0, result["err"]
        assert (
            "  resident weights (fp32; the model is 262.83 GiB): rank0 71.58 GiB"
            in result["out"]
        )
        assert "  estimated card footprint: rank0 111.01 GiB" in result["out"]

    def test_the_estimate_lines_for_world_two_with_advisory_headroom(
        self, artifacts_root: Path, large_cache: Path, large_table
    ) -> None:
        """``dry-run`` prints the advisory estimate, so the reader sees 85.42 GiB per
        rank against an 80 GB card before launching anything."""
        result = _dry_run(artifacts_root, large_cache, "pp=2")
        assert result["code"] == 0, result["err"]
        out = result["out"]
        assert "parallel  dp=1,pp=2,cp=1,tp=1,ep=1 (world 2): accepted" in out
        resident = estimate_resident(large_table, parse_geometry("pp=2"), 2)
        spelled = ", ".join(
            f"rank{r} {format_bytes(b)}" for r, b in enumerate(resident)
        )
        assert f"  resident weights (bf16; the model is 131.42 GiB): {spelled}" in out
        assert "  estimated card footprint: rank0 85.42 GiB, rank1 85.42 GiB" in out
        assert not result["torch"]

    def test_an_uncached_70b_is_undecided_naming_the_cache(
        self, artifacts_root: Path, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = _dry_run(artifacts_root, empty, "pp=4")
        assert result["code"] == 0, result["err"]
        assert (
            f"  memory: undecided — the checkpoint {LARGE}@main is not in the local "
            "Hub cache" in result["out"]
        )
        assert not result["torch"]
