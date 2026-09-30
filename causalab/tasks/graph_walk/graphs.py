"""Graph construction and random walk generation.

Provides three graph types (grid, ring, hex) and random walk utilities
for the graph_walk task, replicating the setup from
"ICLR: In-Context Learning of Representations" (Park et al., ICLR 2025).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Graph:
    """A graph with adjacency structure and node coordinates.

    Attributes:
        adjacency: Maps each node id to its list of neighbor ids.
        coordinates: Maps each node id to its coordinate tuple.
        coordinate_names: Names for each coordinate dimension (e.g. ["row", "col"]).
        n_nodes: Total number of nodes.
    """

    adjacency: dict[int, list[int]]
    coordinates: dict[int, tuple[float, ...]]
    coordinate_names: list[str]
    n_nodes: int
    periodic_dims: dict[int, float] = field(default_factory=dict)  # dim index -> period

    def _build_padded_adjacency(self) -> tuple["np.ndarray", "np.ndarray"]:
        """Build padded adjacency arrays for fast vectorized random walks.

        Returns:
            adj: (n_nodes, max_degree) int array of neighbor indices, padded with -1.
            degree: (n_nodes,) int array of actual degree per node.
        """
        import numpy as np

        max_deg = max(len(v) for v in self.adjacency.values())
        adj = np.full((self.n_nodes, max_deg), -1, dtype=np.int32)
        degree = np.zeros(self.n_nodes, dtype=np.int32)
        for node, neighbors in self.adjacency.items():
            degree[node] = len(neighbors)
            adj[node, : len(neighbors)] = neighbors
        return adj, degree

    def random_walk_fast(
        self,
        start: int,
        length: int,
        rng: random.Random | None = None,
        no_backtrack: bool = True,
    ) -> list[int]:
        """Generate a random walk using precomputed adjacency arrays.

        Builds padded adjacency once (cached) then uses numpy for fast sampling.
        """
        import numpy as np

        if not hasattr(self, "_adj_cache"):
            self._adj_cache = self._build_padded_adjacency()
        adj, degree = self._adj_cache

        np_rng = np.random.RandomState(rng.randint(0, 2**31) if rng else 42)
        path = np.empty(length, dtype=np.int32)
        path[0] = start
        prev = -1

        for i in range(1, length):
            current = path[i - 1]
            neighbors = adj[current, : degree[current]]
            if no_backtrack and prev >= 0:
                candidates = neighbors[neighbors != prev]
                if len(candidates) == 0:
                    candidates = neighbors
            else:
                candidates = neighbors
            prev = current
            path[i] = candidates[np_rng.randint(len(candidates))]

        return path.tolist()


def make_grid_graph(m: int) -> Graph:
    """Create an m x m grid graph with 4-connectivity.

    Nodes are numbered row-major: node = row * m + col.
    Coordinates are (row, col).
    """
    adjacency: dict[int, list[int]] = {}
    coordinates: dict[int, tuple[float, ...]] = {}

    for r in range(m):
        for c in range(m):
            node = r * m + c
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * m + c)
            if r < m - 1:
                neighbors.append((r + 1) * m + c)
            if c > 0:
                neighbors.append(r * m + (c - 1))
            if c < m - 1:
                neighbors.append(r * m + (c + 1))
            adjacency[node] = neighbors
            coordinates[node] = (float(r), float(c))

    return Graph(
        adjacency=adjacency,
        coordinates=coordinates,
        coordinate_names=["row", "col"],
        n_nodes=m * m,
    )


def make_ring_graph(n: int) -> Graph:
    """Create an n-node ring graph with 2-connectivity.

    Coordinates are (angle,) where angle = 2*pi*i/n.
    """
    adjacency: dict[int, list[int]] = {}
    coordinates: dict[int, tuple[float, ...]] = {}

    for i in range(n):
        adjacency[i] = [(i - 1) % n, (i + 1) % n]
        angle = 2.0 * math.pi * i / n
        coordinates[i] = (angle,)

    return Graph(
        adjacency=adjacency,
        coordinates=coordinates,
        coordinate_names=["angle"],
        n_nodes=n,
        periodic_dims={0: 2 * math.pi},
    )


def make_hex_graph(m: int) -> Graph:
    """Create a hexagonal lattice graph.

    Uses offset coordinates (even-r): rows are offset so that even rows
    are shifted right. Each node has up to 6 neighbors.
    Nodes are numbered row-major: node = row * m + col.
    Coordinates are Cartesian (x, y) derived from hex layout.
    """
    adjacency: dict[int, list[int]] = {}
    coordinates: dict[int, tuple[float, ...]] = {}

    for r in range(m):
        for c in range(m):
            node = r * m + c
            neighbors = []

            # Horizontal neighbors
            if c > 0:
                neighbors.append(r * m + (c - 1))
            if c < m - 1:
                neighbors.append(r * m + (c + 1))

            # Vertical and diagonal neighbors depend on row parity
            if r % 2 == 0:  # even row
                if r > 0:
                    if c > 0:
                        neighbors.append((r - 1) * m + (c - 1))
                    neighbors.append((r - 1) * m + c)
                if r < m - 1:
                    if c > 0:
                        neighbors.append((r + 1) * m + (c - 1))
                    neighbors.append((r + 1) * m + c)
            else:  # odd row
                if r > 0:
                    neighbors.append((r - 1) * m + c)
                    if c < m - 1:
                        neighbors.append((r - 1) * m + (c + 1))
                if r < m - 1:
                    neighbors.append((r + 1) * m + c)
                    if c < m - 1:
                        neighbors.append((r + 1) * m + (c + 1))

            adjacency[node] = neighbors

            # Hex cartesian coordinates
            x = float(c) + (0.5 if r % 2 == 1 else 0.0)
            y = float(r) * math.sqrt(3) / 2
            coordinates[node] = (x, y)

    return Graph(
        adjacency=adjacency,
        coordinates=coordinates,
        coordinate_names=["x", "y"],
        n_nodes=m * m,
    )


def make_cylinder_graph(n_ring: int, n_height: int) -> Graph:
    """Create a cylinder graph: rings connected by vertical edges.

    Nodes are numbered row-major: node = h * n_ring + r, where h is the
    height index and r is the ring position.

    Each node connects to its 2 ring neighbors (periodic) and up to 2
    vertical neighbors (non-periodic).

    Coordinates are (height, angle) where angle = 2*pi*r/n_ring. dim_0 must
    align with the enum-primary axis (h, since node = h*n_ring+r) so that
    activation-side compute_centroids (which lex-sorts on alphabetical param
    keys) and output-side fit_belief_tps_parameter (which iterates
    intervention_values in node-id order) end up with the SAME centroid
    ordering. Otherwise downstream metrics like isometry index the two
    manifolds inconsistently.

    Args:
        n_ring: Number of nodes around each ring.
        n_height: Number of rings stacked vertically.
    """
    adjacency: dict[int, list[int]] = {}
    coordinates: dict[int, tuple[float, ...]] = {}
    n_nodes = n_ring * n_height

    for h in range(n_height):
        for r in range(n_ring):
            node = h * n_ring + r
            neighbors = []

            # Ring neighbors (periodic)
            neighbors.append(h * n_ring + (r - 1) % n_ring)
            neighbors.append(h * n_ring + (r + 1) % n_ring)

            # Vertical neighbors (non-periodic)
            if h > 0:
                neighbors.append((h - 1) * n_ring + r)
            if h < n_height - 1:
                neighbors.append((h + 1) * n_ring + r)

            adjacency[node] = neighbors

            angle = 2.0 * math.pi * r / n_ring
            coordinates[node] = (float(h), angle)

    return Graph(
        adjacency=adjacency,
        coordinates=coordinates,
        coordinate_names=["height", "angle"],
        n_nodes=n_nodes,
        periodic_dims={1: 2 * math.pi},
    )


def make_torus_graph(n_ring1: int, n_ring2: int) -> Graph:
    """Create a torus graph: two periodic dimensions.

    Like a cylinder but the vertical dimension also wraps around.
    Nodes connect to 4 neighbors (2 per periodic dimension).

    Coordinates are (angle2, angle1) where angle_i = 2*pi*r_i/n_ring_i. dim_0
    must align with the enum-primary axis (r2, since node = r2*n_ring1+r1) so
    activation- and output-side fits end up with the same centroid ordering.

    Args:
        n_ring1: Number of nodes around the first ring.
        n_ring2: Number of nodes around the second ring.
    """
    adjacency: dict[int, list[int]] = {}
    coordinates: dict[int, tuple[float, ...]] = {}
    n_nodes = n_ring1 * n_ring2

    for r2 in range(n_ring2):
        for r1 in range(n_ring1):
            node = r2 * n_ring1 + r1
            neighbors = []

            # Ring 1 neighbors (periodic)
            neighbors.append(r2 * n_ring1 + (r1 - 1) % n_ring1)
            neighbors.append(r2 * n_ring1 + (r1 + 1) % n_ring1)

            # Ring 2 neighbors (periodic)
            neighbors.append(((r2 - 1) % n_ring2) * n_ring1 + r1)
            neighbors.append(((r2 + 1) % n_ring2) * n_ring1 + r1)

            adjacency[node] = neighbors

            angle1 = 2.0 * math.pi * r1 / n_ring1
            angle2 = 2.0 * math.pi * r2 / n_ring2
            coordinates[node] = (angle2, angle1)

    return Graph(
        adjacency=adjacency,
        coordinates=coordinates,
        coordinate_names=["angle2", "angle1"],
        n_nodes=n_nodes,
        periodic_dims={0: 2 * math.pi, 1: 2 * math.pi},
    )


GRAPH_BUILDERS = {
    "grid": make_grid_graph,
    "ring": make_ring_graph,
    "hex": make_hex_graph,
    "cylinder": make_cylinder_graph,
    "torus": make_torus_graph,
}


def build_graph(
    graph_type: str, graph_size: int, graph_size_2: int | None = None
) -> Graph:
    """Build a graph by type name and size.

    For cylinder, graph_size is n_ring and graph_size_2 is n_height.
    For other types, graph_size is the single size parameter (m for grid/hex, n for ring).
    """
    if graph_type not in GRAPH_BUILDERS:
        raise ValueError(
            f"Unknown graph type '{graph_type}'. Choose from: {list(GRAPH_BUILDERS)}"
        )
    if graph_type in ("cylinder", "torus"):
        if graph_size_2 is None:
            raise ValueError(f"{graph_type} requires graph_size_2")
        # The cylinder/torus builders take two positional sizes; the single-arg
        # builders go through the branch below. pyright unifies the dict
        # values to the narrowest signature.
        return GRAPH_BUILDERS[graph_type](graph_size, graph_size_2)  # pyright: ignore[reportCallIssue]
    # Single-arg builders selected by graph_type; the union signature confuses pyright.
    return GRAPH_BUILDERS[graph_type](graph_size)  # pyright: ignore[reportCallIssue]


def random_walk(
    graph: Graph,
    length: int,
    start: int | None = None,
    rng: random.Random | None = None,
    no_backtrack: bool = False,
) -> list[int]:
    """Generate a random walk on the graph.

    Args:
        graph: The graph to walk on.
        length: Number of steps (output has length+1 nodes including start).
        start: Starting node. If None, chosen uniformly at random.
        rng: Optional random.Random instance for reproducibility.
        no_backtrack: If True, avoid immediately revisiting the previous node
            (i.e. no A->B->A). Falls back to allowing backtrack if the
            previous node is the only neighbor.

    Returns:
        List of node ids visited, of length `length + 1`.
    """
    if rng is None:
        rng = random.Random()
    if start is None:
        start = rng.randrange(graph.n_nodes)

    path = [start]
    current = start
    prev = None
    for _ in range(length):
        neighbors = graph.adjacency[current]
        if no_backtrack and prev is not None:
            candidates = [n for n in neighbors if n != prev]
            if not candidates:
                candidates = neighbors
        else:
            candidates = neighbors
        prev = current
        current = rng.choice(candidates)
        path.append(current)
    return path


def random_neighbor_pairs(
    graph: Graph,
    n_pairs: int,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Sample random (node, neighbor) pairs from the graph.

    Args:
        graph: The graph.
        n_pairs: Number of pairs to sample.
        rng: Optional random.Random instance.

    Returns:
        List of (node, neighbor) tuples.
    """
    if rng is None:
        rng = random.Random()
    pairs = []
    nodes = list(graph.adjacency.keys())
    for _ in range(n_pairs):
        node = rng.choice(nodes)
        neighbor = rng.choice(graph.adjacency[node])
        pairs.append((node, neighbor))
    return pairs


def get_control_points(
    graph: Graph,
) -> tuple[list[tuple[float, ...]], dict[str, float]]:
    """Extract coordinate list and periodic info from a Graph.

    Returns:
        coords: List of (n_dims,) coordinate tuples, one per node.
        periodic_info: Dict mapping periodic coordinate names to their period.
    """
    import math

    coords = [graph.coordinates[i] for i in range(graph.n_nodes)]
    periodic_info = {
        name: 2 * math.pi for name in graph.coordinate_names if name == "angle"
    }
    return coords, periodic_info


def get_control_points_tensor(graph: Graph) -> tuple:
    """Like get_control_points but returns a Tensor for coordinates.

    Returns:
        coords: (n_nodes, n_dims) float tensor of node coordinates.
        periodic_info: Dict mapping periodic coordinate names to their period.
    """
    import torch

    coords_list, periodic_info = get_control_points(graph)
    coords = torch.tensor(coords_list, dtype=torch.float32)
    return coords, periodic_info


def compute_expected_neighbor_distributions(
    graph: Graph,
    no_backtrack: bool = False,
):
    """Compute expected next-node distributions based on graph adjacency.

    For each node, the expected distribution is uniform over its valid
    neighbors. When ``no_backtrack=True``, the previous node is excluded
    from valid neighbors (with fallback if it's the only neighbor).

    Args:
        graph: The graph.
        no_backtrack: If True, return a 3D tensor conditioned on (prev, current).

    Returns:
        (torch.Tensor): If ``no_backtrack=False``, an (n_nodes, n_nodes)
            tensor where row i is the expected distribution over next nodes
            given current node i. If ``no_backtrack=True``, an
            (n_nodes, n_nodes, n_nodes) tensor where dist[prev, current, next]
            is the expected distribution over next nodes given the walk
            arrived at ``current`` from ``prev``.
    """
    import torch

    n = graph.n_nodes
    if not no_backtrack:
        dist = torch.zeros(n, n)
        for node in range(n):
            neighbors = graph.adjacency[node]
            for neighbor in neighbors:
                dist[node, neighbor] = 1.0
            dist[node] /= len(neighbors)
        return dist
    else:
        dist = torch.zeros(n, n, n)
        for prev_node in range(n):
            for current_node in range(n):
                neighbors = set(graph.adjacency[current_node])
                valid = neighbors - {prev_node}
                if not valid:
                    valid = neighbors
                for neighbor in valid:
                    dist[prev_node, current_node, neighbor] = 1.0
                if valid:
                    dist[prev_node, current_node] /= len(valid)
        return dist
