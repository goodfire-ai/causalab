"""Causal model for the graph_walk task.

The causal DAG is:
    graph_type, graph_size (fixed per run)
          |
    node_coordinates  (input: coordinates of the node the walk currently visits)
          |
    walk_sequence  (mechanism: random walk ending at that node)
          |
      raw_input  (mechanism: format walk as text)

    node_coordinates → raw_output (valid neighbor concept tokens)
    walk_seed → walk_sequence (explicit randomness)
"""

from __future__ import annotations

import random

from causalab.causal import Dom, Exo, V, mechanism
from causalab.causal.model import CausalModel
from causalab.causal.scoring import ScoringSpec

from .config import TASK_NAME, GraphWalkConfig
from .graphs import build_graph


def create_causal_model(config: GraphWalkConfig) -> CausalModel:
    """Create a causal model for the graph_walk task.

    Args:
        config: GraphWalkConfig specifying graph type, size, concepts, etc.

    Returns:
        CausalModel with variables: node_coordinates, walk_sequence,
        raw_input, raw_output.
    """
    graph = build_graph(config.graph_type, config.graph_size, config.graph_size_2)
    concepts = config.concepts
    node_ids = list(range(graph.n_nodes))
    # Periodic directions can repeat neighbors, even exceeding the node count.
    max_neighbors = max(len(neighbors) for neighbors in graph.adjacency.values())

    node_to_concept = {i: concepts[i] for i in range(graph.n_nodes)}

    # Pre-compute coordinate tuples and reverse mapping
    coordinates = [tuple(graph.coordinates[i]) for i in node_ids]
    coord_to_node = {coord: i for i, coord in enumerate(coordinates)}

    if config.context_length < 1:
        raise ValueError("context_length must include at least the target node")

    def walk(coordinates, seed):
        path = graph.random_walk_fast(
            coord_to_node[coordinates],
            config.context_length,
            rng=random.Random(seed),
            no_backtrack=config.no_backtrack,
        )
        path.reverse()
        return path

    def render(path):
        return (
            config.separator.join(node_to_concept[node] for node in path)
            + config.separator
        )

    def neighbors(coordinates):
        return [node_to_concept[n] for n in graph.adjacency[coord_to_node[coordinates]]]

    @mechanism
    def equations(
        node_coordinates: Dom(coordinates), walk_seed: Exo(Dom(range(2**32)))
    ):
        walk_sequence = V(
            walk(node_coordinates, walk_seed),
            domain=Dom.sequence(
                Dom(node_ids), length=config.context_length, container=list
            ),
        )
        raw_input = V(render(walk_sequence), domain=Dom(str), lazy=True)  # noqa: F841
        raw_output = V(
            neighbors(node_coordinates),
            domain=Dom.sequence(
                Dom(concepts), max_length=max_neighbors, container=list
            ),
            lazy=True,
        )
        return raw_output

    # Compute periods from graph's periodic dimensions
    periods: dict[str, float] = {}
    if graph.periodic_dims:
        coords = coordinates
        n_dims = len(coords[0]) if coords else 0
        for dim, period in graph.periodic_dims.items():
            key = "node_coordinates" if n_dims == 1 else f"node_coordinates_{dim}"
            periods[key] = period

    # The model predicts the next node's *concept* string, not its coordinate,
    # and ``raw_output`` is the list of every valid next node's concept. Declare
    # the answer forms on ``raw_output`` — the variable that holds the answer —
    # as a plain ``{concept: [concept]}`` map. The former
    # ``{coordinate: [concept]}`` map keyed the *current* node's coordinate to
    # its own concept, so the serialized ``label_forms`` of a row were the
    # node the walk stood on rather than the nodes it could step to, and the
    # string checker graded the list's ``str()`` literally and never matched;
    # keyed by concept, a row's ``raw_output`` list resolves to the union of
    # its members' forms and both paths grade "any valid neighbour". This also
    # keeps the property the coordinate map was introduced for: a lookup by
    # *value*, never by ``id()``, so an equal value built as a new object
    # still finds its forms.
    scoring = ScoringSpec(
        forms={
            "raw_output": {concept: [concept] for concept in dict.fromkeys(concepts)}
        }
    )

    model = CausalModel(
        equations,
        id=TASK_NAME,
        embeddings=EMBEDDINGS,
        periods=periods,
        scoring=scoring,
    )
    # Store for coordinate_names access; CausalModel doesn't declare _graph,
    # but the attribute is set dynamically here and read by downstream code.
    model.values["concepts"] = list(concepts)
    model._graph = graph  # pyright: ignore[reportAttributeAccessIssue]
    return model


# --- Standard exports for load_task() ---
CREATE_CAUSAL_MODEL = create_causal_model


# graph_walk coordinates serve as the natural embedding
def _embed_coordinates(v: tuple) -> list[float]:
    """Identity embedding: coordinates are already numeric vectors."""
    return list(float(x) for x in v)


EMBEDDINGS: dict = {"node_coordinates": _embed_coordinates}
CYCLIC_VARIABLES: set[str] = set()  # determined per-graph by GET_CYCLIC_VARIABLES
TARGET_VARIABLE = "node_coordinates"


def EXAMPLE_TO_CLASS(ex: dict) -> int:
    """Map example to class index (last node visited in walk)."""
    return ex["input"]["walk_sequence"][-1]


def GET_VARIABLE_VALUES(model: CausalModel) -> dict[str, list]:
    """Derive variable values from the model's causal graph structure."""
    return {"node_coordinates": model.values["node_coordinates"]}


def GET_PERIODIC_INFO(model: CausalModel) -> dict[str, float] | None:
    """Derive periodic info from the graph's periodic_dims attribute."""
    graph = getattr(model, "_graph", None)
    if graph is None or not graph.periodic_dims:
        return None
    coords = model.values["node_coordinates"]
    n_dims = len(coords[0]) if coords else 0
    periodic_info = {}
    for dim, period in graph.periodic_dims.items():
        key = "node_coordinates" if n_dims == 1 else f"node_coordinates_{dim}"
        periodic_info[key] = period
    return periodic_info


def SCORE_TOKEN_IDS_FROM_MODEL(pipeline, concepts: list[str]) -> list[int]:
    """Get token IDs for graph node concepts."""
    return [pipeline.tokenizer.encode(c, add_special_tokens=False)[0] for c in concepts]
