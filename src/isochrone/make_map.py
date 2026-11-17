"""Command-line entry point that draws an isochrone map for one destination and arrival."""

from pathlib import Path

import numpy as np
import pandas as pd

from isochrone.osm import WalkingGraph, find_nearby_platforms_on_network


def load_or_compute_access(
    cache_path: Path,
    graph: WalkingGraph,
    grid: np.ndarray,
    platforms: pd.DataFrame,
    max_distance_m: float = 1300.0,
) -> pd.DataFrame:
    """Return the walking legs from the grid cells to nearby platforms, cached on disk.

    The legs depend only on the feed, the OpenStreetMap extract and max_distance_m, not on
    the destination or the arrival time, so they are computed once and reused. The cache is
    trusted as it is: cache_path must change whenever one of these inputs changes.

    Args:
        cache_path: .npz file to read the legs from, or to write them to if it does not exist.
        graph: Pedestrian network, as returned by build_walking_graph.
        grid: Coordinates of the grid cells in metres in PROJECTED_CRS, with shape (n, 2).
        platforms: DataFrame with columns stop_id, x and y, as returned by load_platforms.
        max_distance_m: Maximum walking distance in metres between a cell and a platform.

    Returns:
        DataFrame with columns point, stop_id, distance_m and walk_s, as returned by
        find_nearby_platforms_on_network.
    """
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return pd.DataFrame(
                {
                    "point": cached["point"],
                    "stop_id": cached["stop_ids"][cached["platform"]],
                    "distance_m": cached["distance_m"],
                    "walk_s": cached["walk_s"],
                }
            )

    access = find_nearby_platforms_on_network(graph, grid, platforms, max_distance_m)
    stop_ids, platform = np.unique(access["stop_id"].to_numpy(dtype=str), return_inverse=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        point=access["point"].to_numpy(),
        platform=platform,
        stop_ids=stop_ids,
        distance_m=access["distance_m"].to_numpy(),
        walk_s=access["walk_s"].to_numpy(),
    )
    return access
