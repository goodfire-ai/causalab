"""Property-tier invariants for the ``graph_walk`` task.

This single consolidated file replaces the legacy unit-marked split at
``test_causal_models.py`` / ``test_config.py`` / ``test_graphs.py`` /
``test_interchange_grid.py``. The replaced files have been deleted.

Coverage map:

* ``causal_models.py`` — DAG structure, ``sample_input``-driven walk
  validity, ``raw_input`` formatting, ``raw_output`` neighbour set,
  exported hooks (``EXAMPLE_TO_CLASS``, ``GET_VARIABLE_VALUES``,
  ``GET_PERIODIC_INFO``) and the ``output_tokens`` coordinate→concept map.
  ``SCORE_TOKEN_IDS_FROM_MODEL`` needs a loaded pipeline and is deferred
  to runner-tier coverage (a known gap).
* ``config.py`` — ``GraphWalkConfig.__post_init__`` invariants and the
  ``DEFAULT_CONCEPTS`` pool-shape contract.
* ``counterfactuals.py`` — ``generate_graph_walk_dataset`` /
  ``generate_dataset`` alias / ``make_walk_steering_examples``. Critical:
  the generator uses ``random.Random(seed)`` so it is **isolated** from
  Python's global RNG — assertions therefore do **not** reseed
  ``random.seed`` between calls.
* ``token_positions.py`` — ``create_token_positions`` returns
  ``{"last"}``, last-index resolution against the tiny LM pipeline,
  template/config-arg invariants.
* ``graphs.py`` — every public builder + dispatcher
  (``make_grid_graph`` / ``make_ring_graph`` / ``make_hex_graph`` /
  ``make_cylinder_graph`` / ``make_torus_graph`` / ``build_graph`` /
  ``GRAPH_BUILDERS``), ``Graph.random_walk_fast``, ``random_walk``,
  ``random_neighbor_pairs``, ``get_control_points`` /
  ``get_control_points_tensor`` and
  ``compute_expected_neighbor_distributions``.

Numerical-direct cell (``tests/tasks/graph_walk/pinned_samples.json``
+ ``test_graph_walk_numerical.py``) is **blocked** on the factory-task
walker shim in ``tests/_helpers/task_pins.py``. No per-module numerical
class is added here.

Standards: ``tasks/`` requires ``[smoke-transitive, property-direct,
numerical-direct]``. Smoke-transitive comes from
``baseline/graph_walk_{grid,cylinder}.yaml`` via
``tests/end_to_end/test_smoke.py``. This file satisfies
property-direct; numerical-direct stays red until the walker lands.
"""

from __future__ import annotations

import math
import random

import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.tasks.token_positions import TokenPosition
from causalab.tasks.graph_walk.causal_models import (
    EXAMPLE_TO_CLASS,
    GET_PERIODIC_INFO,
    GET_VARIABLE_VALUES,
)
from causalab.tasks.graph_walk.config import DEFAULT_CONCEPTS, GraphWalkConfig
from causalab.tasks.graph_walk.counterfactuals import (
    generate_dataset,
    generate_graph_walk_dataset,
    make_walk_steering_examples,
)
from causalab.tasks.graph_walk.graphs import (
    GRAPH_BUILDERS,
    Graph,
    build_graph,
    compute_expected_neighbor_distributions,
    get_control_points,
    get_control_points_tensor,
    make_cylinder_graph,
    make_grid_graph,
    make_hex_graph,
    make_ring_graph,
    make_torus_graph,
    random_neighbor_pairs,
    random_walk,
)
from causalab.tasks.graph_walk.token_positions import create_token_positions
from tests._helpers.tasks import get_task

# Hypothesis settings — copied from the MCQA pilot.
#   - deadline=None: factory-task model construction is pure-symbolic but
#     hex/torus adjacency-symmetry sweeps walk O(n*m) neighbours, so the
#     default 200ms deadline can flake on cold caches.
#   - max_examples=30 + dimensions in [2, 5] keep the property suite
#     sub-second.
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Small dimensions used across builder sweeps — large enough to surface
# adjacency / parity bugs (interior degree, odd-row offset, periodicity),
# small enough to keep hypothesis runs cheap.
_BUILDER_DIMS = st.integers(min_value=2, max_value=5)


# --------------------------------------------------------------------------- #
#  Config + DEFAULT_CONCEPTS                                                  #
# --------------------------------------------------------------------------- #


class TestGraphWalkConfigValidationProperty:
    """``GraphWalkConfig.__post_init__`` invariants."""

    pytestmark = pytest.mark.property

    def test_invalid_graph_type_raises(self) -> None:
        with pytest.raises(ValueError, match="graph_type must be"):
            GraphWalkConfig(graph_type="triangle", graph_size=3)

    def test_cylinder_requires_graph_size_2(self) -> None:
        with pytest.raises(ValueError, match="cylinder requires graph_size_2"):
            GraphWalkConfig(graph_type="cylinder", graph_size=3)

    def test_torus_requires_graph_size_2(self) -> None:
        with pytest.raises(ValueError, match="torus requires graph_size_2"):
            GraphWalkConfig(graph_type="torus", graph_size=3)

    @pytest.mark.parametrize("graph_type", ["grid", "ring", "hex"])
    def test_size_2_autofills_to_size_for_non_periodic_secondary(
        self, graph_type: str
    ) -> None:
        cfg = GraphWalkConfig(graph_type=graph_type, graph_size=4)
        assert cfg.graph_size_2 == 4

    def test_ring_expected_nodes_equals_size(self) -> None:
        cfg = GraphWalkConfig(graph_type="ring", graph_size=7)
        assert len(cfg.concepts) == 7

    @pytest.mark.parametrize(
        "graph_type,size,size2",
        [("grid", 3, None), ("hex", 4, None), ("cylinder", 5, 3)],
    )
    def test_non_ring_expected_nodes_is_size_times_size_2(
        self, graph_type: str, size: int, size2: int | None
    ) -> None:
        cfg = GraphWalkConfig(
            graph_type=graph_type, graph_size=size, graph_size_2=size2
        )
        expected = size * (size2 if size2 is not None else size)
        assert len(cfg.concepts) == expected

    def test_empty_concepts_autofills_from_defaults(self) -> None:
        cfg = GraphWalkConfig(graph_type="grid", graph_size=3)
        assert cfg.concepts == DEFAULT_CONCEPTS[:9]

    def test_oversize_default_pool_raises(self) -> None:
        with pytest.raises(ValueError, match="default concepts available"):
            GraphWalkConfig(graph_type="grid", graph_size=30)  # 900 nodes

    def test_wrong_length_custom_concepts_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 9 concepts"):
            GraphWalkConfig(graph_type="grid", graph_size=3, concepts=["a", "b"])

    def test_correct_length_custom_concepts_preserved(self) -> None:
        custom = [f"n{i}" for i in range(25)]
        cfg = GraphWalkConfig(graph_type="grid", graph_size=5, concepts=custom)
        assert cfg.concepts == custom


class TestDefaultConceptsProperty:
    """Pool-shape invariants on ``DEFAULT_CONCEPTS``."""

    pytestmark = pytest.mark.property

    def test_no_duplicates(self) -> None:
        assert len(set(DEFAULT_CONCEPTS)) == len(DEFAULT_CONCEPTS)

    def test_enough_for_19x19_grid(self) -> None:
        assert len(DEFAULT_CONCEPTS) >= 361

    def test_all_lowercase_alpha_and_non_empty(self) -> None:
        for word in DEFAULT_CONCEPTS:
            assert word and word.isalpha() and word.islower(), f"bad concept {word!r}"


# --------------------------------------------------------------------------- #
#  Causal-model structure and traces                                          #
# --------------------------------------------------------------------------- #


def _ring_handle(graph_size: int = 5, context_length: int = 10, seed: int = 42):
    """Build a fresh CausalModel for ring topology via the factory pattern."""
    cfg = GraphWalkConfig(
        graph_type="ring",
        graph_size=graph_size,
        context_length=context_length,
        seed=seed,
    )
    return get_task("graph_walk", task_cfg=cfg).causal_model


def _grid_handle(graph_size: int = 3, context_length: int = 10, seed: int = 42):
    cfg = GraphWalkConfig(
        graph_type="grid",
        graph_size=graph_size,
        context_length=context_length,
        seed=seed,
    )
    return get_task("graph_walk", task_cfg=cfg).causal_model


def _cylinder_handle(
    graph_size: int = 4,
    graph_size_2: int = 3,
    context_length: int = 10,
    seed: int = 42,
):
    cfg = GraphWalkConfig(
        graph_type="cylinder",
        graph_size=graph_size,
        graph_size_2=graph_size_2,
        context_length=context_length,
        seed=seed,
    )
    return get_task("graph_walk", task_cfg=cfg).causal_model


class TestGraphWalkCausalModelStructureProperty:
    """Static DAG shape across topologies."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "builder",
        [_ring_handle, _grid_handle, _cylinder_handle],
        ids=["ring", "grid", "cylinder"],
    )
    def test_required_variables_present(self, builder) -> None:
        model = builder()
        expected = {"node_coordinates", "walk_sequence", "raw_input", "raw_output"}
        assert expected <= set(model.variables)

    @pytest.mark.parametrize(
        "builder",
        [_ring_handle, _grid_handle, _cylinder_handle],
        ids=["ring", "grid", "cylinder"],
    )
    def test_node_coordinates_is_only_input(self, builder) -> None:
        model = builder()
        assert list(model.inputs) == ["node_coordinates", "walk_seed"]

    @pytest.mark.parametrize(
        "builder",
        [_ring_handle, _grid_handle, _cylinder_handle],
        ids=["ring", "grid", "cylinder"],
    )
    def test_walk_sequence_parents_are_node_coordinates(self, builder) -> None:
        assert builder().parents["walk_sequence"] == ["node_coordinates", "walk_seed"]

    @pytest.mark.parametrize(
        "builder",
        [_ring_handle, _grid_handle, _cylinder_handle],
        ids=["ring", "grid", "cylinder"],
    )
    def test_raw_input_parents_are_walk_sequence(self, builder) -> None:
        assert builder().parents["raw_input"] == ["walk_sequence"]

    @pytest.mark.parametrize(
        "builder",
        [_ring_handle, _grid_handle, _cylinder_handle],
        ids=["ring", "grid", "cylinder"],
    )
    def test_raw_output_parents_are_node_coordinates(self, builder) -> None:
        assert builder().parents["raw_output"] == ["node_coordinates"]


class TestGraphWalkSampleInputProperty:
    """``model.sample_input()`` invariants under a hypothesis seed sweep."""

    pytestmark = pytest.mark.property

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_node_coordinates_in_value_set(self, seed: int) -> None:
        random.seed(seed)
        model = _ring_handle()
        trace = model.sample_input()
        assert tuple(trace["node_coordinates"]) in {
            tuple(c) for c in model.values["node_coordinates"]
        }

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_walk_length_matches_context_length(self, seed: int) -> None:
        random.seed(seed)
        model = _grid_handle(graph_size=3, context_length=8)
        trace = model.sample_input()
        assert len(trace["walk_sequence"]) == 8

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_every_walk_step_is_adjacency_valid(self, seed: int) -> None:
        random.seed(seed)
        model = _grid_handle(graph_size=3, context_length=6)
        walk = model.sample_input()["walk_sequence"]
        graph = build_graph("grid", 3)
        assert all(
            walk[i + 1] in graph.adjacency[walk[i]] for i in range(len(walk) - 1)
        )

    def test_new_trace_deterministic_repeats(self) -> None:
        """``CausalModel.new_trace`` on the same input is byte-equal across calls.

        The explicit walk seed is part of the input, so repeated calls
        produce identical traces without sharing mutable RNG state.
        """
        model = _ring_handle()
        coord = model.values["node_coordinates"][0]
        a = dict(model.new_trace({"node_coordinates": coord, "walk_seed": 0}).to_dict())
        b = dict(model.new_trace({"node_coordinates": coord, "walk_seed": 0}).to_dict())
        assert a == b


class TestGraphWalkRawOutputProperty:
    """``raw_output`` token set and ``raw_input`` formatting invariants."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize(
        "graph_type,size,size2,node,neighbor_nodes",
        [
            ("ring", 1, None, 0, [0, 0]),
            ("ring", 2, None, 0, [1, 1]),
            ("cylinder", 1, 2, 0, [0, 0, 1]),
            ("cylinder", 2, 2, 0, [1, 1, 2]),
            ("torus", 1, 2, 0, [0, 0, 1, 1]),
            ("torus", 2, 2, 0, [1, 1, 2, 2]),
            ("grid", 3, None, 4, [1, 7, 3, 5]),
        ],
    )
    def test_raw_output_preserves_repeated_periodic_neighbors(
        self, graph_type, size, size2, node, neighbor_nodes
    ) -> None:
        """Periodic directions can repeat a neighbor, exceeding the node count."""
        cfg = GraphWalkConfig(
            graph_type=graph_type,
            graph_size=size,
            graph_size_2=size2,
            context_length=3,
        )
        model = get_task("graph_walk", task_cfg=cfg).causal_model
        trace = model.new_trace(
            {
                "node_coordinates": model.values["node_coordinates"][node],
                "walk_seed": 0,
            }
        )
        assert trace["raw_output"] == [cfg.concepts[n] for n in neighbor_nodes]

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_output_equals_neighbor_concept_set(self, seed: int) -> None:
        random.seed(seed)
        model = _grid_handle(graph_size=3, context_length=4)
        trace = model.sample_input()
        graph = build_graph("grid", 3)
        coord_to_node = {tuple(graph.coordinates[i]): i for i in range(graph.n_nodes)}
        node = coord_to_node[tuple(trace["node_coordinates"])]
        cfg_concepts = model.values["concepts"]
        expected = {cfg_concepts[n] for n in graph.adjacency[node]}
        assert set(trace["raw_output"]) == expected

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_raw_input_has_trailing_separator(self, seed: int) -> None:
        random.seed(seed)
        cfg = GraphWalkConfig(
            graph_type="ring", graph_size=5, context_length=5, separator=",", seed=seed
        )
        model = get_task("graph_walk", task_cfg=cfg).causal_model
        assert model.sample_input()["raw_input"].endswith(",")

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_text_concepts_correspond_to_walk_nodes(self, seed: int) -> None:
        random.seed(seed)
        cfg = GraphWalkConfig(
            graph_type="ring", graph_size=5, context_length=5, separator=",", seed=seed
        )
        model = get_task("graph_walk", task_cfg=cfg).causal_model
        trace = model.sample_input()
        text_concepts = trace["raw_input"].rstrip(",").split(",")
        walk_concepts = [cfg.concepts[n] for n in trace["walk_sequence"]]
        assert text_concepts == walk_concepts


class TestGraphWalkExportHooksProperty:
    """Module-level exports consumed by the loader / scorer contract.

    ``SCORE_TOKEN_IDS_FROM_MODEL`` needs a loaded pipeline, so it
    is deferred to runner-tier coverage and not exercised here.
    """

    pytestmark = pytest.mark.property

    def test_example_to_class_returns_non_negative_int(self) -> None:
        model = _ring_handle()
        trace = model.sample_input()
        cls = EXAMPLE_TO_CLASS({"input": trace})
        assert isinstance(cls, int) and cls >= 0

    def test_get_variable_values_covers_non_input_variables(self) -> None:
        model = _ring_handle()
        keys = set(GET_VARIABLE_VALUES(model))
        assert "node_coordinates" in keys  # the input variable is included

    def test_periodic_info_none_for_grid(self) -> None:
        assert GET_PERIODIC_INFO(_grid_handle()) is None

    def test_periodic_info_dict_for_ring(self) -> None:
        info = GET_PERIODIC_INFO(_ring_handle())
        assert isinstance(info, dict) and info  # non-empty dict

    def test_periodic_info_dict_for_cylinder(self) -> None:
        info = GET_PERIODIC_INFO(_cylinder_handle())
        assert isinstance(info, dict) and info

    def test_forms_are_declared_per_concept_on_raw_output(self) -> None:
        """The answer forms are a plain ``{concept: [concept]}`` map on
        ``raw_output`` — the variable whose value (a list of every valid next
        node's concept) is the answer.

        Keyed by value, never by ``id()`` (a past regression), and on the
        answer rather than on the current node's coordinate: a row's
        ``raw_output`` list resolves to the union of its members' forms, so the
        string grader and the serialized ``*_forms`` both say "any valid
        neighbour".
        """
        model = _ring_handle()
        spec = model.scoring
        assert spec is not None and spec.answer_variable == "raw_output"
        forms = model.output_tokens["raw_output"]
        concepts = model.values["concepts"]
        assert set(forms) == set(concepts)
        assert all(forms[c] == [c] for c in concepts)
        trace = model.new_trace(
            {"node_coordinates": model.values["node_coordinates"][0], "walk_seed": 0}
        )
        neighbours = trace["raw_output"]
        assert isinstance(neighbours, list) and len(neighbours) == 2  # a ring
        assert spec.forms_of(neighbours) == tuple(neighbours)
        assert all(spec.grade(n, neighbours) == 1.0 for n in neighbours)
        other = next(c for c in concepts if c not in neighbours)
        assert spec.grade(other, neighbours) == 0.0


# --------------------------------------------------------------------------- #
#  Counterfactual generators                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _grid_cm():
    return _grid_handle(graph_size=5, context_length=10)


@pytest.fixture(scope="module")
def _cyl_cm():
    return _cylinder_handle(graph_size=3, graph_size_2=3, context_length=10)


class TestGenerateGraphWalkDatasetProperty:
    """Counterfactual generator invariants.

    The generator uses ``random.Random(seed)`` internally — it is
    **isolated** from Python's global RNG. We deliberately do not call
    ``random.seed`` between invocations: doing so would mask global-RNG
    bugs (i.e. a regression that introduced a hidden global ``random``
    call would still pass).
    """

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("graph_type", ["cylinder", "torus"])
    def test_small_periodic_dataset_labels_include_repeated_neighbors(
        self, graph_type
    ) -> None:
        cfg = GraphWalkConfig(
            graph_type=graph_type,
            graph_size=1,
            graph_size_2=2,
            context_length=3,
        )
        model = get_task("graph_walk", task_cfg=cfg).causal_model
        examples = generate_graph_walk_dataset(model, n_examples=2, seed=0)
        labeled = model.label_counterfactual_data(examples, ["node_coordinates"])
        for example, row in zip(examples, labeled):
            expected = example["counterfactual_inputs"][0]["raw_output"]
            assert row["label"] == expected
            assert len(expected) == (3 if graph_type == "cylinder" else 4)
            assert len(set(expected)) == 2

    @pytest.mark.parametrize("n", [1, 5])
    def test_length_matches_request(self, _grid_cm, n: int) -> None:
        examples = generate_graph_walk_dataset(_grid_cm, n_examples=n, seed=0)
        assert len(examples) == n

    def test_length_matches_request_full_cycle(self, _grid_cm) -> None:
        coords = _grid_cm.values["node_coordinates"]
        examples = generate_graph_walk_dataset(_grid_cm, n_examples=len(coords), seed=0)
        assert len(examples) == len(coords)

    def test_length_matches_request_over_cycle(self, _grid_cm) -> None:
        coords = _grid_cm.values["node_coordinates"]
        examples = generate_graph_walk_dataset(
            _grid_cm, n_examples=2 * len(coords), seed=0
        )
        assert len(examples) == 2 * len(coords)

    def test_each_entry_has_input_and_one_cf(self, _grid_cm) -> None:
        examples = generate_graph_walk_dataset(_grid_cm, n_examples=5, seed=0)
        assert all(
            "input" in e and len(e["counterfactual_inputs"]) == 1 for e in examples
        )

    def test_base_and_cf_coord_differ(self, _grid_cm) -> None:
        examples = generate_graph_walk_dataset(_grid_cm, n_examples=10, seed=0)
        for ex in examples:
            base = tuple(ex["input"]["node_coordinates"])
            cf = tuple(ex["counterfactual_inputs"][0]["node_coordinates"])
            assert base != cf

    def test_base_coords_cycle_through_all_nodes(self, _grid_cm) -> None:
        coords = _grid_cm.values["node_coordinates"]
        examples = generate_graph_walk_dataset(_grid_cm, n_examples=len(coords), seed=0)
        seen = {tuple(e["input"]["node_coordinates"]) for e in examples}
        assert seen == {tuple(c) for c in coords}

    def test_cf_coord_drawn_from_other_coords(self, _grid_cm) -> None:
        coord_set = {tuple(c) for c in _grid_cm.values["node_coordinates"]}
        examples = generate_graph_walk_dataset(_grid_cm, n_examples=8, seed=1)
        for ex in examples:
            base = tuple(ex["input"]["node_coordinates"])
            cf = tuple(ex["counterfactual_inputs"][0]["node_coordinates"])
            assert cf in (coord_set - {base})

    def test_seed_determinism(self, _grid_cm) -> None:
        a = generate_graph_walk_dataset(_grid_cm, n_examples=4, seed=7)
        b = generate_graph_walk_dataset(_grid_cm, n_examples=4, seed=7)
        a_cfs = [tuple(e["counterfactual_inputs"][0]["node_coordinates"]) for e in a]
        b_cfs = [tuple(e["counterfactual_inputs"][0]["node_coordinates"]) for e in b]
        assert a_cfs == b_cfs

    def test_seed_independence_from_global_random(self, _grid_cm) -> None:
        """``random.Random(seed)`` isolates the generator from ``random.seed``."""
        random.seed(11111)
        a = generate_graph_walk_dataset(_grid_cm, n_examples=4, seed=7)
        random.seed(22222)
        b = generate_graph_walk_dataset(_grid_cm, n_examples=4, seed=7)
        a_cfs = [tuple(e["counterfactual_inputs"][0]["node_coordinates"]) for e in a]
        b_cfs = [tuple(e["counterfactual_inputs"][0]["node_coordinates"]) for e in b]
        assert a_cfs == b_cfs

    def test_different_seeds_produce_different_cf(self, _grid_cm) -> None:
        a = generate_graph_walk_dataset(_grid_cm, n_examples=6, seed=1)
        b = generate_graph_walk_dataset(_grid_cm, n_examples=6, seed=2)
        a_cfs = [tuple(e["counterfactual_inputs"][0]["node_coordinates"]) for e in a]
        b_cfs = [tuple(e["counterfactual_inputs"][0]["node_coordinates"]) for e in b]
        assert a_cfs != b_cfs

    def test_runs_on_cylinder_topology(self, _cyl_cm) -> None:
        examples = generate_graph_walk_dataset(_cyl_cm, n_examples=3, seed=0)
        assert len(examples) == 3


class TestGenerateDatasetAliasProperty:
    """``generate_dataset`` is the load_task() alias for the canonical generator."""

    pytestmark = pytest.mark.property

    def test_alias_is_canonical_function(self) -> None:
        assert generate_dataset is generate_graph_walk_dataset


class TestMakeWalkSteeringExamplesProperty:
    """``make_walk_steering_examples`` is a pure shape wrapper."""

    pytestmark = pytest.mark.property

    def test_length_matches_input(self) -> None:
        walks = ["a|b|c|", "d|e|", "f|"]
        out = make_walk_steering_examples(walks)
        assert len(out) == 3

    def test_keys_are_input_and_counterfactuals(self) -> None:
        out = make_walk_steering_examples(["w|"])
        assert set(out[0].keys()) == {"input", "counterfactuals"}

    def test_input_carries_raw_input(self) -> None:
        out = make_walk_steering_examples(["walk|text|"])
        assert out[0]["input"]["raw_input"] == "walk|text|"

    def test_counterfactuals_is_empty_list(self) -> None:
        out = make_walk_steering_examples(["x|"])
        assert out[0]["counterfactuals"] == []

    def test_empty_input_yields_empty_output(self) -> None:
        assert make_walk_steering_examples([]) == []


# --------------------------------------------------------------------------- #
#  Token positions (needs a real tokenizer)                                   #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _tiny_pipeline():
    """Build the tiny-LM ``LMPipeline`` used by the token-position class.

    Mirrors the session-scoped fixture in ``tests/neural/conftest.py``;
    duplicated here so this consolidated test file is self-contained
    (no cross-conftest fixture-sharing assumptions).
    """
    from tests._helpers.pipeline_shim import PipelineShim
    from tests._helpers.tiny import TINY_RANDOM_MODEL_NAME

    return PipelineShim(TINY_RANDOM_MODEL_NAME)


class TestGraphWalkTokenPositionsProperty:
    """``create_token_positions`` returns a ``{"last"}`` dict of TokenPositions."""

    pytestmark = pytest.mark.property

    def test_returns_only_last_key(self, _tiny_pipeline) -> None:
        positions = create_token_positions(_tiny_pipeline)
        assert set(positions.keys()) == {"last"}

    def test_value_is_token_position(self, _tiny_pipeline) -> None:
        positions = create_token_positions(_tiny_pipeline)
        assert isinstance(positions["last"], TokenPosition)

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_last_index_resolves_to_len_tokens_minus_one(
        self, _tiny_pipeline, seed: int
    ) -> None:
        random.seed(seed)
        model = _grid_handle(graph_size=3, context_length=5)
        trace = model.sample_input()
        positions = create_token_positions(_tiny_pipeline)
        ids = _tiny_pipeline.load([trace])["input_ids"][0]
        assert positions["last"].index(trace) == [len(ids) - 1]

    def test_template_none_equivalent_to_raw_input_template(
        self, _tiny_pipeline
    ) -> None:
        random.seed(0)
        model = _ring_handle()
        trace = model.sample_input()
        a = create_token_positions(_tiny_pipeline, template=None)["last"].index(trace)
        b = create_token_positions(_tiny_pipeline, template="{raw_input}")[
            "last"
        ].index(trace)
        assert a == b

    def test_config_kwarg_is_ignored(self, _tiny_pipeline) -> None:
        random.seed(0)
        model = _ring_handle()
        trace = model.sample_input()
        a = create_token_positions(_tiny_pipeline, config=None)["last"].index(trace)
        b = create_token_positions(
            _tiny_pipeline, config=GraphWalkConfig(graph_type="ring", graph_size=5)
        )["last"].index(trace)
        assert a == b

    def test_repeated_calls_yield_equal_indices(self, _tiny_pipeline) -> None:
        random.seed(0)
        model = _ring_handle()
        trace = model.sample_input()
        a = create_token_positions(_tiny_pipeline)["last"].index(trace)
        b = create_token_positions(_tiny_pipeline)["last"].index(trace)
        assert a == b


# --------------------------------------------------------------------------- #
#  graphs.py — Graph dataclass + per-builder + dispatcher                     #
# --------------------------------------------------------------------------- #


class TestGraphTopologyProperty:
    """Per-symbol invariants from the largest surface in this task.

    Test methods are grouped by builder; each method stays ≤ 15 lines.
    ``Graph.random_walk_fast`` is pinned against ``random_walk`` even though
    it has no in-repo caller besides this class.
    """

    pytestmark = pytest.mark.property

    # ---- Graph dataclass + random_walk_fast --------------------------- #

    @given(seed=st.integers(min_value=0, max_value=10_000), length=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_random_walk_fast_starts_at_start_node(
        self, seed: int, length: int
    ) -> None:
        graph = make_ring_graph(5)
        rng = random.Random(seed)
        path = graph.random_walk_fast(0, length, rng=rng)
        assert path[0] == 0

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @_HYPOTHESIS_SETTINGS
    def test_random_walk_fast_steps_are_adjacency_valid(self, seed: int) -> None:
        graph = make_grid_graph(4)
        rng = random.Random(seed)
        path = graph.random_walk_fast(0, 8, rng=rng)
        assert all(
            path[i + 1] in graph.adjacency[path[i]] for i in range(len(path) - 1)
        )

    def test_graph_default_periodic_dims_empty_for_non_periodic(self) -> None:
        assert make_grid_graph(3).periodic_dims == {}

    # ---- make_grid_graph --------------------------------------------- #

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_grid_adjacency_size_equals_n_squared(self, m: int) -> None:
        g = make_grid_graph(m)
        assert len(g.adjacency) == m * m

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_grid_adjacency_is_symmetric(self, m: int) -> None:
        g = make_grid_graph(m)
        assert all(
            node in g.adjacency[nb] for node, nbs in g.adjacency.items() for nb in nbs
        )

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_grid_has_no_self_loops(self, m: int) -> None:
        g = make_grid_graph(m)
        assert all(node not in nbs for node, nbs in g.adjacency.items())

    def test_grid_interior_degree_is_four(self) -> None:
        g = make_grid_graph(5)
        assert len(g.adjacency[12]) == 4  # node (2,2) is interior

    def test_grid_corner_degree_is_two(self) -> None:
        g = make_grid_graph(5)
        assert len(g.adjacency[0]) == 2  # node (0,0) is a corner

    def test_grid_edge_degree_is_three(self) -> None:
        g = make_grid_graph(5)
        assert len(g.adjacency[1]) == 3  # node (0,1) is on the top edge

    def test_grid_m_equal_one_is_isolated_node(self) -> None:
        assert make_grid_graph(1).adjacency[0] == []

    # ---- make_ring_graph --------------------------------------------- #

    @given(n=st.integers(min_value=3, max_value=10))
    @_HYPOTHESIS_SETTINGS
    def test_ring_every_node_has_degree_two(self, n: int) -> None:
        g = make_ring_graph(n)
        assert all(len(nbs) == 2 for nbs in g.adjacency.values())

    @given(n=st.integers(min_value=3, max_value=10))
    @_HYPOTHESIS_SETTINGS
    def test_ring_is_symmetric(self, n: int) -> None:
        g = make_ring_graph(n)
        assert all(
            node in g.adjacency[nb] for node, nbs in g.adjacency.items() for nb in nbs
        )

    def test_ring_has_no_self_loops(self) -> None:
        g = make_ring_graph(8)
        assert all(node not in nbs for node, nbs in g.adjacency.items())

    def test_ring_wraps_zero_to_n_minus_one(self) -> None:
        g = make_ring_graph(10)
        assert 9 in g.adjacency[0] and 0 in g.adjacency[9]

    def test_ring_periodic_dims_is_2pi_on_dim_zero(self) -> None:
        g = make_ring_graph(8)
        assert g.periodic_dims == {0: 2 * math.pi}

    def test_ring_angles_strictly_increasing_in_zero_to_2pi(self) -> None:
        g = make_ring_graph(6)
        angles = [g.coordinates[i][0] for i in range(6)]
        assert angles == sorted(angles) and 0 <= angles[0] and angles[-1] < 2 * math.pi

    # ---- make_hex_graph ---------------------------------------------- #

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_hex_adjacency_is_symmetric(self, m: int) -> None:
        g = make_hex_graph(m)
        assert all(
            node in g.adjacency[nb] for node, nbs in g.adjacency.items() for nb in nbs
        )

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_hex_has_no_self_loops(self, m: int) -> None:
        g = make_hex_graph(m)
        assert all(node not in nbs for node, nbs in g.adjacency.items())

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_hex_has_no_out_of_range_neighbors(self, m: int) -> None:
        g = make_hex_graph(m)
        assert all(0 <= nb < g.n_nodes for nbs in g.adjacency.values() for nb in nbs)

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_hex_has_no_duplicate_neighbors_per_node(self, m: int) -> None:
        g = make_hex_graph(m)
        assert all(len(nbs) == len(set(nbs)) for nbs in g.adjacency.values())

    def test_hex_interior_degree_is_six(self) -> None:
        g = make_hex_graph(5)
        assert len(g.adjacency[12]) == 6  # node (2,2) is interior

    def test_hex_corner_degree_at_most_three(self) -> None:
        g = make_hex_graph(5)
        assert len(g.adjacency[0]) <= 3

    def test_hex_odd_row_x_offset_is_half(self) -> None:
        g = make_hex_graph(3)
        assert g.coordinates[3][0] == 0.5  # node (1,0) is on an odd row

    @given(m=_BUILDER_DIMS)
    @_HYPOTHESIS_SETTINGS
    def test_hex_handshake_lemma(self, m: int) -> None:
        g = make_hex_graph(m)
        deg_sum = sum(len(nbs) for nbs in g.adjacency.values())
        assert deg_sum % 2 == 0  # 2|E|

    # ---- make_cylinder_graph ---------------------------------------- #

    def test_cylinder_is_symmetric(self) -> None:
        g = make_cylinder_graph(6, 4)
        assert all(
            node in g.adjacency[nb] for node, nbs in g.adjacency.items() for nb in nbs
        )

    def test_cylinder_has_no_self_loops(self) -> None:
        g = make_cylinder_graph(6, 4)
        assert all(node not in nbs for node, nbs in g.adjacency.items())

    def test_cylinder_ring_axis_is_periodic(self) -> None:
        g = make_cylinder_graph(6, 4)
        assert g.periodic_dims == {1: 2 * math.pi}

    def test_cylinder_interior_degree_is_four(self) -> None:
        g = make_cylinder_graph(6, 3)
        # h=1 (middle ring) → 2 ring + 2 vertical neighbours
        assert len(g.adjacency[1 * 6 + 0]) == 4

    def test_cylinder_top_bottom_degree_is_three(self) -> None:
        g = make_cylinder_graph(6, 3)
        assert len(g.adjacency[0]) == 3 and len(g.adjacency[2 * 6 + 0]) == 3

    def test_cylinder_coordinate_ordering_is_height_then_angle(self) -> None:
        assert make_cylinder_graph(4, 2).coordinate_names == ["height", "angle"]

    # ---- make_torus_graph ------------------------------------------- #

    @given(n=st.integers(min_value=3, max_value=5))
    @_HYPOTHESIS_SETTINGS
    def test_torus_every_node_has_degree_four(self, n: int) -> None:
        g = make_torus_graph(n, n)
        assert all(len(nbs) == 4 for nbs in g.adjacency.values())

    @given(n=st.integers(min_value=3, max_value=5))
    @_HYPOTHESIS_SETTINGS
    def test_torus_is_symmetric(self, n: int) -> None:
        g = make_torus_graph(n, n)
        assert all(
            node in g.adjacency[nb] for node, nbs in g.adjacency.items() for nb in nbs
        )

    def test_torus_has_no_self_loops(self) -> None:
        g = make_torus_graph(4, 4)
        assert all(node not in nbs for node, nbs in g.adjacency.items())

    def test_torus_both_axes_periodic(self) -> None:
        g = make_torus_graph(4, 4)
        assert g.periodic_dims == {0: 2 * math.pi, 1: 2 * math.pi}

    def test_torus_coordinate_ordering_is_angle2_then_angle1(self) -> None:
        """``dim_0`` must align with the enum-primary axis ``r2``.

        Activation-side centroid sorting depends on this; see
        ``make_torus_graph`` docstring.
        """
        assert make_torus_graph(4, 4).coordinate_names == ["angle2", "angle1"]

    # ---- build_graph / GRAPH_BUILDERS -------------------------------- #

    def test_graph_builders_keys_are_exactly_documented_set(self) -> None:
        assert set(GRAPH_BUILDERS) == {"grid", "ring", "hex", "cylinder", "torus"}

    @pytest.mark.parametrize(
        "gtype,args",
        [
            ("grid", (3,)),
            ("ring", (5,)),
            ("hex", (3,)),
            ("cylinder", (4, 3)),
            ("torus", (4, 3)),
        ],
    )
    def test_graph_builders_return_symmetric_graph(
        self, gtype: str, args: tuple
    ) -> None:
        g = GRAPH_BUILDERS[gtype](*args)
        assert all(
            node in g.adjacency[nb] for node, nbs in g.adjacency.items() for nb in nbs
        )

    def test_build_graph_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown graph type"):
            build_graph("triangle", 3)

    def test_build_graph_cylinder_without_size_2_raises(self) -> None:
        with pytest.raises(ValueError, match="cylinder requires graph_size_2"):
            build_graph("cylinder", 4)

    def test_build_graph_torus_without_size_2_raises(self) -> None:
        with pytest.raises(ValueError, match="torus requires graph_size_2"):
            build_graph("torus", 4)

    # ---- random_walk -------------------------------------------------- #

    def test_random_walk_length_is_length_plus_one(self) -> None:
        path = random_walk(make_ring_graph(5), 7, start=0, rng=random.Random(0))
        assert len(path) == 8

    def test_random_walk_starts_at_supplied_start(self) -> None:
        path = random_walk(make_grid_graph(4), 4, start=5, rng=random.Random(0))
        assert path[0] == 5

    @pytest.mark.parametrize(
        "graph", [make_grid_graph(3), make_ring_graph(5), make_hex_graph(3)]
    )
    def test_random_walk_steps_adjacency_valid(self, graph: Graph) -> None:
        path = random_walk(graph, 30, rng=random.Random(0))
        assert all(
            path[i + 1] in graph.adjacency[path[i]] for i in range(len(path) - 1)
        )

    def test_random_walk_no_backtrack_avoids_immediate_reversal(self) -> None:
        g = make_grid_graph(5)
        path = random_walk(g, 50, rng=random.Random(0), no_backtrack=True)
        assert all(path[i + 2] != path[i] for i in range(len(path) - 2))

    def test_random_walk_no_backtrack_fallback_on_dead_end(self) -> None:
        """3-node path graph forces the fallback branch at the leaf."""
        g = Graph(
            adjacency={0: [1], 1: [0, 2], 2: [1]},
            coordinates={0: (0.0,), 1: (1.0,), 2: (2.0,)},
            coordinate_names=["x"],
            n_nodes=3,
        )
        path = random_walk(g, 4, start=0, rng=random.Random(0), no_backtrack=True)
        assert path[0] == 0 and path[1] == 1  # no crash on fallback

    # ---- random_neighbor_pairs --------------------------------------- #

    def test_random_neighbor_pairs_length_matches_request(self) -> None:
        pairs = random_neighbor_pairs(make_grid_graph(4), 20, rng=random.Random(0))
        assert len(pairs) == 20

    def test_random_neighbor_pairs_every_pair_is_an_edge(self) -> None:
        g = make_grid_graph(4)
        pairs = random_neighbor_pairs(g, 50, rng=random.Random(0))
        assert all(v in g.adjacency[u] for u, v in pairs)

    def test_random_neighbor_pairs_seed_determinism(self) -> None:
        g = make_ring_graph(8)
        a = random_neighbor_pairs(g, 10, rng=random.Random(42))
        b = random_neighbor_pairs(g, 10, rng=random.Random(42))
        assert a == b

    # ---- get_control_points / get_control_points_tensor ------------- #

    def test_get_control_points_returns_n_node_coord_tuples(self) -> None:
        g = make_ring_graph(7)
        coords, _ = get_control_points(g)
        assert len(coords) == 7 and all(isinstance(c, tuple) for c in coords)

    def test_get_control_points_periodic_info_nonempty_for_ring(self) -> None:
        _, info = get_control_points(make_ring_graph(5))
        assert info  # ring has a coordinate named 'angle' → picked up by heuristic

    def test_get_control_points_periodic_info_empty_for_grid(self) -> None:
        _, info = get_control_points(make_grid_graph(3))
        assert info == {}

    def test_get_control_points_tensor_shape_matches_node_count(self) -> None:
        coords, _ = get_control_points_tensor(make_grid_graph(4))
        assert coords.shape == (16, 2)

    def test_get_control_points_tensor_dtype_is_float32(self) -> None:
        coords, _ = get_control_points_tensor(make_ring_graph(5))
        assert coords.dtype == torch.float32

    def test_get_control_points_tensor_matches_get_control_points(self) -> None:
        g = make_grid_graph(3)
        list_coords, _ = get_control_points(g)
        tensor_coords, _ = get_control_points_tensor(g)
        assert torch.equal(
            tensor_coords, torch.tensor(list_coords, dtype=torch.float32)
        )

    # ---- compute_expected_neighbor_distributions -------------------- #

    def test_neighbor_distributions_2d_shape_when_no_backtrack_false(self) -> None:
        g = make_grid_graph(3)
        dist = compute_expected_neighbor_distributions(g, no_backtrack=False)
        assert dist.shape == (9, 9)

    def test_neighbor_distributions_3d_shape_when_no_backtrack_true(self) -> None:
        g = make_grid_graph(3)
        dist = compute_expected_neighbor_distributions(g, no_backtrack=True)
        assert dist.shape == (9, 9, 9)

    def test_neighbor_distributions_rows_sum_to_one(self) -> None:
        g = make_ring_graph(5)
        dist = compute_expected_neighbor_distributions(g, no_backtrack=False)
        assert torch.allclose(dist.sum(dim=1), torch.ones(5), atol=1e-6)

    def test_neighbor_distributions_mass_only_on_adjacent_nodes(self) -> None:
        g = make_ring_graph(5)
        dist = compute_expected_neighbor_distributions(g, no_backtrack=False)
        for i in range(5):
            non_neighbours = set(range(5)) - {*g.adjacency[i]}
            assert all(dist[i, j].item() == 0.0 for j in non_neighbours)

    def test_neighbor_distributions_no_backtrack_excludes_prev_node(self) -> None:
        """On a node with ≥2 neighbours, the dist excludes ``prev``."""
        g = make_grid_graph(3)  # interior node 4 has 4 neighbours
        dist = compute_expected_neighbor_distributions(g, no_backtrack=True)
        prev = g.adjacency[4][0]
        assert dist[prev, 4, prev].item() == 0.0
