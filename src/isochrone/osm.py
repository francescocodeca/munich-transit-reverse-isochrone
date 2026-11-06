"""Build the pedestrian network from an OpenStreetMap extract and compute walking distances."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium
import pandas as pd
import shapely
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from isochrone.walking import to_projected

WALKABLE_HIGHWAYS = frozenset(
    {
        "footway",
        "pedestrian",
        "path",
        "steps",
        "corridor",
        "living_street",
        "residential",
        "service",
        "track",
        "unclassified",
        "tertiary",
        "tertiary_link",
        "secondary",
        "secondary_link",
        "primary",
        "primary_link",
        "trunk",
        "trunk_link",
    }
)
FOOT_ONLY_IF_ALLOWED = frozenset({"cycleway", "bridleway"})
FOOT_ALLOWED = frozenset({"yes", "designated", "permissive", "destination", "customers"})
FOOT_FORBIDDEN = frozenset({"no", "private", "use_sidepath"})
ACCESS_FORBIDDEN = frozenset({"no", "private"})


def is_walkable(tags: Mapping[str, str]) -> bool:
    """Return whether pedestrians may walk along an OpenStreetMap way.

    The type of road (highway) gives a default, following the German defaults of the
    OpenStreetMap wiki; an explicit foot tag overrides it, then a general access restriction.
    Ways that are never walkable (motorways, roads with motorroad=yes, ways under construction,
    platforms, ...) are rejected whatever their other tags. Roads whose sidewalks are mapped as
    separate ways, and areas (area=yes), are rejected, so that pedestrians use the separate
    ways.

    Args:
        tags: Tags of the way, e.g. {"highway": "cycleway", "foot": "designated"}.

    Returns:
        True if the way belongs to the pedestrian network.
    """
    highway = tags.get("highway")
    if highway in WALKABLE_HIGHWAYS:
        default = True
    elif highway in FOOT_ONLY_IF_ALLOWED:
        default = False
    else:
        return False
    if tags.get("motorroad") == "yes" or tags.get("area") == "yes":
        return False
    if tags.get("sidewalk") == "separate" or tags.get("sidewalk:both") == "separate":
        return False

    foot = tags.get("foot")
    if foot in FOOT_ALLOWED:
        return True
    if foot in FOOT_FORBIDDEN:
        return False
    if tags.get("access") in ACCESS_FORBIDDEN:
        return False
    return default


@dataclass
class WalkingGraph:
    """Pedestrian network as an undirected graph with edge lengths in metres.

    Nodes are numbered 0, 1, ...; arrays indexed by node have one entry per node.

    Attributes:
        node_ids: OpenStreetMap id of each node.
        x: x coordinate of each node, metres in PROJECTED_CRS.
        y: y coordinate of each node, metres in PROJECTED_CRS.
        lengths: Sparse matrix whose entry (u, v) is the length in metres of the edge between
            nodes u and v; it is symmetric, with one entry per direction.
    """

    node_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    lengths: csr_array


def build_walking_graph(osm_path: Path) -> WalkingGraph:
    """Build the pedestrian network from the walkable ways of an OpenStreetMap extract.

    Every pair of consecutive nodes of a walkable way (see is_walkable) becomes an undirected
    edge whose length is the straight-line distance between the two nodes. Ways that share a
    node are connected there. An edge that appears in several ways is kept once.

    Args:
        osm_path: Path of the .osm.pbf file.

    Returns:
        The WalkingGraph of the extract.

    Raises:
        ValueError: If a node of a walkable way has no location in the extract.
    """
    refs, lons, lats, way_ends = [], [], [], []
    ways = (
        osmium.FileProcessor(osm_path)
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway"))
    )
    for way in ways:
        if not is_walkable(way.tags):
            continue
        for node in way.nodes:
            if not node.location.valid():
                raise ValueError(f"node {node.ref} of way {way.id} has no location")
            refs.append(node.ref)
            lons.append(node.lon)
            lats.append(node.lat)
        way_ends.append(len(refs))

    node_ids, first, index = np.unique(refs, return_index=True, return_inverse=True)
    x, y = to_projected(np.array(lons)[first], np.array(lats)[first])

    is_last = np.zeros(len(refs), dtype=bool)
    is_last[np.array(way_ends) - 1] = True
    u, v = index[:-1][~is_last[:-1]], index[1:][~is_last[:-1]]
    edges = np.unique(np.sort(np.column_stack([u, v]), axis=1), axis=0)
    u, v = edges[:, 0], edges[:, 1]
    length = np.hypot(x[u] - x[v], y[u] - y[v])

    n = len(node_ids)
    lengths = csr_array(
        (np.concatenate([length, length]), (np.concatenate([u, v]), np.concatenate([v, u]))),
        shape=(n, n),
    )
    return WalkingGraph(node_ids, x, y, lengths)


def snap_to_graph(
    graph: WalkingGraph, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest node of the main component of the graph to each point.

    Only the largest connected component is used, so that every point is attached to a node
    from which the rest of the network can be reached; small components are mostly pieces
    isolated by excluded private ways, or cut at the border of the extract.

    Args:
        graph: WalkingGraph as returned by build_walking_graph.
        x: x coordinates of the points, metres in PROJECTED_CRS.
        y: y coordinates of the points, metres in PROJECTED_CRS.

    Returns:
        The index of the nearest node of each point, and the straight-line distance in metres
        between the point and that node.
    """
    _, component = connected_components(graph.lengths, directed=False)
    main_nodes = np.flatnonzero(component == np.bincount(component).argmax())
    tree = cKDTree(np.column_stack([graph.x[main_nodes], graph.y[main_nodes]]))
    distance, nearest = tree.query(np.column_stack([x, y]))
    return main_nodes[nearest], distance


def read_poly(poly_path: Path) -> shapely.Geometry:
    """Read an Osmosis polygon file, such as the .poly that Geofabrik publishes with an extract.

    The file has a name line, then sections made of a name line, one "lon lat" line per vertex
    and an END line, then a final END line. Sections whose name starts with "!" are holes and
    are subtracted from the union of the other sections.

    Args:
        poly_path: Path to the .poly file.

    Returns:
        The area covered by the file, in longitude and latitude (WGS 84).
    """
    lines = [line.strip() for line in poly_path.read_text().splitlines() if line.strip()]
    areas, holes = [], []
    i = 1
    while lines[i] != "END":
        name = lines[i]
        i += 1
        ring = []
        while lines[i] != "END":
            lon, lat = lines[i].split()
            ring.append((float(lon), float(lat)))
            i += 1
        i += 1
        (holes if name.startswith("!") else areas).append(shapely.Polygon(ring))
    return shapely.difference(shapely.union_all(areas), shapely.union_all(holes))


def walking_distances(
    graph: WalkingGraph,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    max_distance_m: float = 1300.0,
    batch_size: int = 20,
) -> pd.DataFrame:
    """Return the walking distance of every pair of a source and a target within a limit.

    Each point is snapped to the main component of the graph (see snap_to_graph). The distance
    is: straight line from the source to its node, shortest path on the graph, straight line
    from the target's node to the target. The graph is undirected, so the distance is the
    same in both directions.

    Args:
        graph: Pedestrian network, as returned by build_walking_graph.
        source_xy: Array of shape (n, 2) with the source coordinates in PROJECTED_CRS.
        target_xy: Array of shape (m, 2) with the target coordinates in PROJECTED_CRS.
        max_distance_m: Pairs farther apart than this, in metres, are left out.
        batch_size: Number of sources per call of Dijkstra, which allocates a dense row of
            one value per graph node for each source; it limits memory use.

    Returns:
        DataFrame with one row per pair within max_distance_m and columns source and target
        (row positions in source_xy and target_xy) and distance_m.
    """
    source_node, source_snap = snap_to_graph(graph, source_xy[:, 0], source_xy[:, 1])
    target_node, target_snap = snap_to_graph(graph, target_xy[:, 0], target_xy[:, 1])
    parts = []
    for start in range(0, len(source_node), batch_size):
        batch = np.arange(start, min(start + batch_size, len(source_node)))
        on_graph = dijkstra(graph.lengths, indices=source_node[batch], limit=max_distance_m)
        total = source_snap[batch, None] + on_graph[:, target_node] + target_snap
        row, target = np.nonzero(total <= max_distance_m)
        parts.append(
            pd.DataFrame(
                {"source": batch[row], "target": target, "distance_m": total[row, target]}
            )
        )
    return pd.concat(parts, ignore_index=True)
