"""The parallel geometry (``docs/model_parallelism.md`` §2, §10.3):
``ParallelGeometry``, its ``--parallel`` grammar, the torch-free ``check``
against a registry entry, and the integer mesh ``MeshLayout``.

Three tiers. ``unit``: the parser with each refusal beside its valid twin,
naming the axis; ``check`` against every built-in registry entry — the A3B
accepts what its heads, KV heads, experts and layers divide, ``gpt2`` refuses
every axis above one (no transformers plan), the dense Llama refuses
``expert > 1``. ``property``: §10.3's ``geometry.check`` and ``mesh groups``
rows over hypothesis draws. And the two rows' named mutations, hand-written
the repository's way: the change is applied through a seam and the property
is shown to fail.

Geometry is execution, never identity (spec §8): nothing here touches a
canonical form or a digest, and the receipt half is ``test_cli_parallel.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol import parallel
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    AXES,
    GEOMETRY_AXES,
    ONE,
    PLANLESS_FAMILIES,
    SPELLINGS,
    Coordinates,
    MeshLayout,
    ParallelGeometry,
    check,
    format_geometry,
    parse_geometry,
)
from causalab.protocol.registry.models import _REGISTRY  # pyright: ignore[reportPrivateUsage]
from causalab.protocol.registry import (
    ModelInfo,
    get_model_info,
)

from tests._helpers.geometries import (
    any_geometries,
    dividing_geometries,
    geometries,
    mesh_geometries,
)

REPO = Path(__file__).resolve().parents[2]

#: The repository's hypothesis settings (``docs/model_parallelism.md`` §10).
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

A3B = "Qwen/Qwen3.6-35B-A3B"
LLAMA_8B = "meta-llama/Llama-3.1-8B"
#: 14 heads and 2 KV heads: the one built-in whose head count admits a
#: tensor axis (7) that straddles its KV heads (§6.6).
QWEN25 = "Qwen/Qwen2.5-0.5B"
GPT2 = "gpt2"

#: Every built-in entry, by key — the census ``check`` is held against.
BUILT_IN = tuple(sorted(_REGISTRY))


def _refusal(text: str) -> ProtocolError:
    with pytest.raises(ProtocolError) as err:
        parse_geometry(text)
    return err.value


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_default_geometry_is_all_ones_and_world_one() -> None:
    assert ONE == ParallelGeometry()
    assert (ONE.data, ONE.pipeline, ONE.context, ONE.tensor, ONE.expert) == (1,) * 5
    assert ONE.model == 1 and ONE.world == 1


@pytest.mark.unit
def test_model_is_the_larger_of_tensor_and_expert_and_world_multiplies() -> None:
    """§2: the model group holds one replicated residual stream; ``tensor``
    and ``expert`` are its sub-groups, so its size is the larger of the two."""
    geometry = ParallelGeometry(data=2, pipeline=3, context=1, tensor=4, expert=8)
    assert geometry.model == 8
    assert geometry.world == 2 * 3 * 1 * 8


@pytest.mark.unit
@pytest.mark.parametrize("axis", GEOMETRY_AXES)
@pytest.mark.parametrize("bad", (0, -1, True, 2.0, "2"))
def test_a_geometry_refuses_a_non_positive_or_non_integer_axis(
    axis: str, bad: Any
) -> None:
    """Constructed from Python, not only parsed: the same ``P4`` naming
    ``--parallel.<axis>``, so no caller can hold a geometry no run could."""
    with pytest.raises(ProtocolError) as err:
        ParallelGeometry(**{axis: bad})
    assert err.value.code == "P4"
    assert f"--parallel.{axis}" in str(err.value)
    assert ParallelGeometry(**{axis: 1}) == ONE


@pytest.mark.unit
def test_the_axes_are_the_placement_seams_axes() -> None:
    """One vocabulary: the ``Axis`` the placement and collective seams name
    groups by is this module's, imported — not a second spelling."""
    from causalab.neural.shared.parallel import collective, placement

    assert placement.Axis is parallel.Axis
    assert placement.AXES is AXES
    assert collective.Axis is parallel.Axis
    assert AXES == ("data", "pipeline", "context", "model", "tensor", "expert")
    assert GEOMETRY_AXES == ("data", "pipeline", "context", "tensor", "expert")
    assert set(GEOMETRY_AXES) == {axis for axis in AXES if axis != "model"}


@pytest.mark.unit
def test_the_module_is_torch_free() -> None:
    """``protocol/`` is torch-free (docs/CODEBASE.md §1); ``check`` runs in
    ``dry-run`` before any weights, so the module must import without torch."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, causalab.protocol.parallel; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


# --------------------------------------------------------------------------- #
# the grammar — each refusal beside its valid twin
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_grammar_reads_every_spelling_and_defaults_the_rest_to_one() -> None:
    assert SPELLINGS == {
        "dp": "data",
        "pp": "pipeline",
        "cp": "context",
        "tp": "tensor",
        "ep": "expert",
    }
    assert parse_geometry("tp=4,ep=8") == ParallelGeometry(tensor=4, expert=8)
    assert parse_geometry("dp=2, pp=3 ,cp=1") == ParallelGeometry(data=2, pipeline=3)
    assert parse_geometry("") == ONE
    assert parse_geometry("tp=1") == ONE


@pytest.mark.unit
def test_an_unknown_axis_is_refused_naming_the_spellings() -> None:
    err = _refusal("tp=2,xp=2")
    assert err.code == "P4"
    assert "xp" in str(err) and "--parallel" in str(err)
    for spelling in SPELLINGS:
        assert spelling in str(err)
    assert parse_geometry("tp=2,ep=2") == ParallelGeometry(tensor=2, expert=2)


@pytest.mark.unit
def test_the_long_names_are_not_spellings() -> None:
    """The grammar is the five short spellings; ``tensor=2`` is an unknown
    axis, with the spelling it meant suggested."""
    err = _refusal("tensor=2")
    assert err.code == "P4" and "tp" in str(err)


@pytest.mark.unit
def test_a_repeated_axis_is_refused_naming_it() -> None:
    err = _refusal("tp=2,tp=4")
    assert err.code == "P4"
    assert "--parallel.tensor" in str(err) and "twice" in str(err)
    assert parse_geometry("tp=4") == ParallelGeometry(tensor=4)


@pytest.mark.unit
@pytest.mark.parametrize("value", ("x", "2.0", "", "-1", "+2", "0x2", "1_0"))
def test_a_non_integer_value_is_refused_naming_the_axis(value: str) -> None:
    err = _refusal(f"ep={value}")
    assert err.code == "P4"
    assert "--parallel.expert" in str(err)
    assert parse_geometry("ep=2") == ParallelGeometry(expert=2)


@pytest.mark.unit
def test_a_value_below_one_is_refused_naming_the_axis() -> None:
    err = _refusal("pp=0")
    assert err.code == "P4"
    assert "--parallel.pipeline" in str(err) and "0" in str(err)
    assert parse_geometry("pp=1") == ONE


@pytest.mark.unit
@pytest.mark.parametrize("text", ("tp", "tp=", "=2", "tp=2,", ",tp=2", "tp=2=3"))
def test_an_item_that_is_not_axis_equals_value_is_refused(text: str) -> None:
    err = _refusal(text)
    assert err.code == "P4" and "--parallel" in str(err)


@pytest.mark.unit
def test_format_is_the_inverse_of_parse() -> None:
    geometry = ParallelGeometry(data=2, pipeline=3, context=1, tensor=4, expert=8)
    assert format_geometry(geometry) == "dp=2,pp=3,cp=1,tp=4,ep=8"
    assert parse_geometry(format_geometry(geometry)) == geometry
    assert format_geometry(ONE) == "dp=1,pp=1,cp=1,tp=1,ep=1"


# --------------------------------------------------------------------------- #
# check — against every built-in entry
# --------------------------------------------------------------------------- #


def _named(refusals: tuple[str, ...]) -> set[str]:
    """The axes a refusal list names, by the ``--parallel.<axis>`` prefix
    every refusal opens with."""
    out: set[str] = set()
    for text in refusals:
        prefix, _, _ = text.partition(":")
        assert prefix.startswith("--parallel."), text
        axis = prefix[len("--parallel.") :]
        assert axis in GEOMETRY_AXES, text
        out.add(axis)
    return out


@pytest.mark.unit
def test_the_built_in_entries_are_the_expected_census() -> None:
    """The three named entries exist and are what the tests below assume:
    a hybrid MoE tower, a dense GQA model, and the plan-less family."""
    a3b, llama, gpt2 = (
        get_model_info(A3B),
        get_model_info(LLAMA_8B),
        get_model_info(GPT2),
    )
    assert (a3b.num_heads, a3b.num_kv_heads, a3b.num_experts, a3b.num_layers) == (
        16,
        2,
        256,
        40,
    )
    assert (
        llama.num_heads,
        llama.num_kv_heads,
        llama.num_experts,
        llama.num_layers,
    ) == (32, 8, None, 32)
    assert gpt2.family == "gpt2"
    # GPT-J is the second plan-less family: GPTJConfig ships no
    # base_model_tp_plan (transformers 5.16.1)
    assert get_model_info("EleutherAI/gpt-j-6b").family == "gptj"
    assert PLANLESS_FAMILIES == frozenset({"gpt2", "gptj"})
    assert {A3B, LLAMA_8B, GPT2, "EleutherAI/gpt-j-6b"} <= set(BUILT_IN)


@pytest.mark.unit
@pytest.mark.parametrize("key", BUILT_IN)
def test_every_built_in_entry_accepts_world_one(key: str) -> None:
    assert check(ONE, get_model_info(key)) == ()


@pytest.mark.unit
@pytest.mark.parametrize("tensor", (1, 2, 4, 8, 16))
def test_the_a3b_accepts_every_tensor_axis_its_heads_admit(tensor: int) -> None:
    """16 heads and 2 KV heads: ``tp ∈ {1, 2}`` shards the KV heads, ``tp ∈
    {4, 8, 16}`` replicates them (§6.6) — the golden runs at ``tp=2``
    (§10.6)."""
    assert check(ParallelGeometry(tensor=tensor), get_model_info(A3B)) == ()


@pytest.mark.unit
def test_a_tensor_axis_whose_query_heads_straddle_the_kv_heads_is_refused() -> None:
    """Qwen2.5-0.5B, 14 heads and 2 KV heads, at ``tp=7``: 7 divides the
    heads, neither divides nor is a multiple of the KV heads, so a rank's two
    query heads would read two KV heads; the refusal names the tensor axis
    and both head counts."""
    refusals = check(ParallelGeometry(tensor=7), get_model_info(QWEN25))
    assert _named(refusals) == {"tensor"}
    (only,) = refusals
    assert "num_kv_heads=2" in only and "num_heads=14" in only and "tp=7" in only


@pytest.mark.unit
@pytest.mark.parametrize("expert", (2, 4, 8, 16, 32, 64, 128, 256))
def test_the_a3b_accepts_every_expert_axis_dividing_256(expert: int) -> None:
    assert check(ParallelGeometry(expert=expert), get_model_info(A3B)) == ()


@pytest.mark.unit
@pytest.mark.parametrize("expert", (3, 5, 7, 12, 100, 512))
def test_the_a3b_refuses_an_expert_axis_not_dividing_256(expert: int) -> None:
    refusals = check(ParallelGeometry(expert=expert), get_model_info(A3B))
    assert _named(refusals) == {"expert"}
    assert "num_experts" in refusals[0] and "256" in refusals[0]


@pytest.mark.unit
@pytest.mark.parametrize("pipeline", (1, 2, 4, 5, 8, 10, 20, 39, 40))
def test_the_a3b_accepts_a_pipeline_of_at_most_its_layers(pipeline: int) -> None:
    assert check(ParallelGeometry(pipeline=pipeline), get_model_info(A3B)) == ()


@pytest.mark.unit
def test_the_a3b_refuses_more_stages_than_layers() -> None:
    refusals = check(ParallelGeometry(pipeline=41), get_model_info(A3B))
    assert _named(refusals) == {"pipeline"}
    assert "num_layers" in refusals[0] and "40" in refusals[0]


@pytest.mark.unit
def test_the_a3b_accepts_mixed_sub_groups_and_refuses_ones_that_do_not_nest() -> None:
    """``tp=2, ep=4`` is the mixed model group of §8.2; ``tp=2, ep=3`` has a
    model group of 3 the tensor group of 2 does not divide."""
    info = get_model_info(A3B)
    assert check(ParallelGeometry(tensor=2, expert=4), info) == ()
    assert check(ParallelGeometry(data=2, pipeline=4, tensor=2, expert=8), info) == ()
    refusals = check(ParallelGeometry(tensor=2, expert=3), info)
    # ep=3 does not divide 256 either — both facts are reported, independently
    assert _named(refusals) == {"tensor", "expert"}
    assert any("model" in text and "tp=2" in text for text in refusals)


@pytest.mark.unit
def test_context_is_accepted_by_the_model_facts_and_refuses_a_decode() -> None:
    """§8.4: the context axis has no model fact to fail — its one document
    rule is a decode, ``check_context``, naming ``--parallel.context``."""
    info = get_model_info(A3B)
    assert check(ParallelGeometry(context=2), info) == ()
    assert check(ParallelGeometry(context=3, tensor=2, expert=8), info) == ()
    assert parallel.check_context(ParallelGeometry(context=2), False) == ()
    assert parallel.check_context(ParallelGeometry(context=1), True) == ()
    refusals = parallel.check_context(ParallelGeometry(context=2), True)
    assert _named(refusals) == {"context"}
    assert "cp=2" in refusals[0] and "decod" in refusals[0]


@pytest.mark.unit
@pytest.mark.parametrize("axis", GEOMETRY_AXES)
def test_gpt2_refuses_any_axis_above_one(axis: str) -> None:
    """No transformers parallel plan for the family (§2, §11): every axis is
    refused by name, whatever the divisibility facts would say."""
    for key in (GPT2, "gpt2-xl"):
        refusals = check(ParallelGeometry(**{axis: 2}), get_model_info(key))
        assert axis in _named(refusals), refusals
        assert any("gpt2" in text and "plan" in text for text in refusals)


@pytest.mark.unit
def test_the_dense_llama_refuses_expert_parallelism_and_accepts_tensor() -> None:
    info = get_model_info(LLAMA_8B)
    refusals = check(ParallelGeometry(expert=2), info)
    assert _named(refusals) == {"expert"}
    assert "dense" in refusals[0] or "no routed experts" in refusals[0]
    # 32 heads, 8 KV heads: tp ∈ {2, 4, 8} shards the KV heads, {16, 32}
    # replicates them (§6.6); 64 exceeds the heads
    for tensor in (2, 4, 8, 16, 32):
        assert check(ParallelGeometry(tensor=tensor), info) == ()
    assert _named(check(ParallelGeometry(tensor=64), info)) == {"tensor"}


@pytest.mark.unit
def test_check_reports_every_failing_axis_not_the_first() -> None:
    refusals = check(
        ParallelGeometry(pipeline=99, context=2, tensor=3, expert=2),
        get_model_info(LLAMA_8B),
    )
    # the context axis fails no model fact (§8.4); the other three are named
    assert _named(refusals) == {"pipeline", "tensor", "expert"}


# --------------------------------------------------------------------------- #
# the mesh — units
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_mesh_order_is_data_pipeline_context_model_outermost_first() -> None:
    """§2: ``(data, pipeline, context, model)``, row-major; ``tensor`` and
    ``expert`` are contiguous sub-groups of each model group."""
    mesh = MeshLayout(
        ParallelGeometry(data=2, pipeline=2, context=1, tensor=2, expert=4)
    )
    assert mesh.world == 16
    assert mesh.groups("model") == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    )
    assert mesh.groups("tensor") == tuple((r, r + 1) for r in range(0, 16, 2))
    assert mesh.groups("expert") == mesh.groups("model")
    assert mesh.groups("pipeline") == tuple(
        (r, r + 4) for r in (0, 1, 2, 3, 8, 9, 10, 11)
    )
    assert mesh.groups("data") == tuple((r, r + 8) for r in range(8))
    assert mesh.groups("context") == tuple((r,) for r in range(16))
    assert mesh.coordinates(13) == Coordinates(
        data=1, pipeline=1, context=0, model=1, tensor=1, expert=1
    )
    assert mesh.group_of(13, "tensor") == (12, 13)
    assert mesh.rank_in(13, "tensor") == 1
    assert mesh.group_of(13, "data") == (5, 13)
    assert mesh.rank_in(13, "data") == 1


@pytest.mark.unit
def test_tensor_equal_expert_equal_model_is_transformers_own_layout() -> None:
    """§2: when ``tensor == expert == model`` the sub-groups are the model
    group itself."""
    mesh = MeshLayout(ParallelGeometry(tensor=4, expert=4))
    assert (
        mesh.groups("tensor")
        == mesh.groups("expert")
        == mesh.groups("model")
        == ((0, 1, 2, 3),)
    )


@pytest.mark.unit
@pytest.mark.parametrize("rank", (-1, 4, 100))
def test_a_rank_outside_the_world_is_refused(rank: int) -> None:
    mesh = MeshLayout(ParallelGeometry(tensor=2, expert=2))
    for method in (
        mesh.coordinates,
        lambda r: mesh.group_of(r, "tensor"),
        lambda r: mesh.rank_in(r, "model"),
    ):
        with pytest.raises(ValueError, match="rank"):
            method(rank)


@pytest.mark.unit
def test_a_mesh_refuses_sub_groups_that_do_not_divide_the_model_group() -> None:
    with pytest.raises(ProtocolError) as err:
        MeshLayout(ParallelGeometry(tensor=2, expert=3))
    assert err.value.code == "P4" and "--parallel." in str(err.value)
    assert MeshLayout(ParallelGeometry(tensor=2, expert=4)).world == 4


# --------------------------------------------------------------------------- #
# the properties (§10.3, rows "geometry.check" and "mesh groups")
# --------------------------------------------------------------------------- #


def _facts_fail(geometry: ParallelGeometry, info: ModelInfo) -> set[str]:
    """The oracle: which axes a §2 fact fails for, written out plainly."""
    failing: set[str] = set()
    # the context axis fails no model fact (§8.4): its rule is a document's
    if (
        info.num_heads % geometry.tensor
        # the KV fact (§6.6): a shard of the KV heads, or a replication of them
        or (info.num_kv_heads % geometry.tensor and geometry.tensor % info.num_kv_heads)
        or geometry.model % geometry.tensor
    ):
        failing.add("tensor")
    if info.num_experts is None:
        if geometry.expert > 1:
            failing.add("expert")
    elif info.num_experts % geometry.expert or geometry.model % geometry.expert:
        failing.add("expert")
    if geometry.pipeline > info.num_layers:
        failing.add("pipeline")
    if info.family in PLANLESS_FAMILIES:
        failing |= {axis for axis in GEOMETRY_AXES if getattr(geometry, axis) > 1}
    return failing


def _assert_check_matches_the_facts(
    geometry: ParallelGeometry, info: ModelInfo
) -> None:
    """Row "geometry.check": accepted iff every fact holds; a refusal names
    the failing axis, and only failing axes are named."""
    refusals = check(geometry, info)
    failing = _facts_fail(geometry, info)
    assert (refusals == ()) == (not failing), (geometry, info.key, refusals)
    assert _named(refusals) == failing, (geometry, info.key, refusals)
    assert (
        geometry.world
        == geometry.data * geometry.pipeline * geometry.context * geometry.model
    )


@pytest.mark.property
@pytest.mark.parametrize("key", (A3B, LLAMA_8B, GPT2, "Qwen/Qwen3-4B-Instruct-2507"))
@_HYPOTHESIS_SETTINGS
@given(data=st.data())
def test_check_accepts_iff_every_fact_holds(key: str, data: st.DataObject) -> None:
    info = get_model_info(key)
    geometry = data.draw(geometries(info))
    _assert_check_matches_the_facts(geometry, info)


@pytest.mark.property
@pytest.mark.parametrize("key", (A3B, LLAMA_8B))
@_HYPOTHESIS_SETTINGS
@given(data=st.data())
def test_a_dividing_draw_is_accepted(key: str, data: st.DataObject) -> None:
    """The dividing strategy is sound: what it draws, ``check`` accepts —
    so the mixed property above is not passing on refusals alone."""
    info = get_model_info(key)
    assert check(data.draw(dividing_geometries(info)), info) == ()


@pytest.mark.property
@_HYPOTHESIS_SETTINGS
@given(geometry=any_geometries())
def test_parse_of_format_is_the_identity(geometry: ParallelGeometry) -> None:
    assert parse_geometry(format_geometry(geometry)) == geometry


def _assert_mesh_groups_partition(geometry: ParallelGeometry) -> None:
    """Row "mesh groups": for each axis the groups partition ``range(world)``;
    every rank is in exactly one tensor and one expert group, both inside its
    model group; the model, tensor and expert groups are contiguous; and
    ``coordinates`` / ``groups`` / ``group_of`` / ``rank_in`` agree."""
    mesh = MeshLayout(geometry)
    world = geometry.world
    sizes = {
        "data": geometry.data,
        "pipeline": geometry.pipeline,
        "context": geometry.context,
        "model": geometry.model,
        "tensor": geometry.tensor,
        "expert": geometry.expert,
    }
    for axis in AXES:
        groups = mesh.groups(axis)
        flat = sorted(rank for group in groups for rank in group)
        assert flat == list(range(world)), (axis, groups)
        assert all(len(group) == sizes[axis] for group in groups), (axis, groups)
        assert all(group == tuple(sorted(group)) for group in groups), (axis, groups)
        if axis in ("model", "tensor", "expert"):
            assert all(
                group == tuple(range(group[0], group[0] + len(group)))
                for group in groups
            ), (axis, groups)
        for rank in range(world):
            group = mesh.group_of(rank, axis)
            assert group in groups, (axis, rank, group)
            assert rank in group
            position = mesh.rank_in(rank, axis)
            assert group[position] == rank
            assert getattr(mesh.coordinates(rank), axis) == position
    for rank in range(world):
        model_group = set(mesh.group_of(rank, "model"))
        assert set(mesh.group_of(rank, "tensor")) <= model_group
        assert set(mesh.group_of(rank, "expert")) <= model_group
        c = mesh.coordinates(rank)
        assert (
            rank
            == (
                (c.data * geometry.pipeline + c.pipeline) * geometry.context + c.context
            )
            * geometry.model
            + c.model
        )


@pytest.mark.property
@_HYPOTHESIS_SETTINGS
@given(geometry=mesh_geometries())
def test_mesh_groups_partition_the_world(geometry: ParallelGeometry) -> None:
    _assert_mesh_groups_partition(geometry)


# --------------------------------------------------------------------------- #
# the named mutations (§10.3) — applied through a seam, shown to fail
# --------------------------------------------------------------------------- #

#: A dense entry whose KV-head count does not divide its head count — every
#: built-in's does, so no built-in can make the head fact fail alone.
#: ``tp=4`` here fails ``tensor | num_heads`` (6) and nothing else.
_ODD_HEADS = ModelInfo(
    key="test/odd-heads",
    hidden_size=48,
    num_layers=4,
    num_heads=6,
    num_kv_heads=4,
    head_dim=8,
    intermediate_size=96,
    vocab_size=64,
    family="llama",
)

#: Geometries the check property is replayed over under a mutation: the kv
#: fact is the only one ``tp=7`` fails on Qwen2.5-0.5B (14 heads, 2 KV heads
#: — a straddle, §6.6), so dropping it flips the verdict; the head fact is the
#: only one ``tp=4`` fails on ``_ODD_HEADS``, so misnaming it moves the named
#: axis. The A3B at ``tp=4`` replicates its KV heads and is accepted.
_MUTATION_DRAWS: tuple[tuple[ModelInfo, ParallelGeometry], ...] = (
    (get_model_info(QWEN25), ParallelGeometry(tensor=7)),
    (get_model_info(A3B), ParallelGeometry(tensor=4)),
    (_ODD_HEADS, ParallelGeometry(tensor=4)),
    (get_model_info(LLAMA_8B), ParallelGeometry(tensor=2)),
)


def _replay_check_property() -> None:
    for info, geometry in _MUTATION_DRAWS:
        _assert_check_matches_the_facts(geometry, info)


@pytest.mark.unit
def test_the_check_property_holds_unmutated() -> None:
    _replay_check_property()


@pytest.mark.unit
def test_mutation_dropping_a_divisibility_check_fails_the_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop the KV rule: ``tp=7`` on Qwen2.5-0.5B (14 heads, 2 KV heads) is
    accepted, and the oracle says it must not be — a rank's two query heads
    would straddle two KV heads."""
    dropped = tuple(
        rule for rule in parallel.CHECKS if rule is not parallel.tensor_fits_kv_heads
    )
    assert len(dropped) == len(parallel.CHECKS) - 1
    monkeypatch.setattr(parallel, "CHECKS", dropped)
    assert check(ParallelGeometry(tensor=7), get_model_info(QWEN25)) == ()
    with pytest.raises(AssertionError):
        _replay_check_property()


@pytest.mark.unit
def test_mutation_swapping_the_axis_named_fails_the_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name ``--parallel.expert`` where the head fact fails: the refusal no
    longer names the failing axis."""

    def misnamed(geometry: ParallelGeometry, info: ModelInfo) -> str | None:
        text = parallel.tensor_divides_heads(geometry, info)
        return (
            None
            if text is None
            else text.replace("--parallel.tensor", "--parallel.expert", 1)
        )

    swapped: tuple[Callable[[ParallelGeometry, ModelInfo], str | None], ...] = tuple(
        misnamed if rule is parallel.tensor_divides_heads else rule
        for rule in parallel.CHECKS
    )
    monkeypatch.setattr(parallel, "CHECKS", swapped)
    assert _named(check(ParallelGeometry(tensor=4), _ODD_HEADS)) == {"expert"}
    with pytest.raises(AssertionError):
        _replay_check_property()


_MESH_DRAWS = (
    ParallelGeometry(tensor=2, expert=4),
    ParallelGeometry(data=2, pipeline=2, tensor=2, expert=2),
    ParallelGeometry(tensor=1, expert=3),
)


@pytest.mark.unit
def test_the_mesh_property_holds_unmutated() -> None:
    for geometry in _MESH_DRAWS:
        _assert_mesh_groups_partition(geometry)


@pytest.mark.unit
def test_mutation_off_by_one_stride_in_the_sub_group_carve_fails_the_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start every sub-group one rank late: the tensor groups no longer
    partition the world (rank 0 of each model group is in none, a rank past
    the group is in one)."""
    monkeypatch.setattr(
        parallel, "sub_group_start", lambda index, size: index - index % size + 1
    )
    with pytest.raises(AssertionError):
        for geometry in _MESH_DRAWS:
            _assert_mesh_groups_partition(geometry)


@pytest.mark.unit
def test_the_rank_equal_to_the_world_is_refused() -> None:
    """Reject ``rank == world`` as well as ranks beyond that boundary."""
    mesh = MeshLayout(ParallelGeometry(data=2, tensor=2, expert=2))
    assert mesh.world == 4
    mesh.coordinates(3)
    with pytest.raises(ValueError, match="rank 4"):
        mesh.coordinates(4)
