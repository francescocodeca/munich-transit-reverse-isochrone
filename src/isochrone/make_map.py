"""Command-line entry point that draws an isochrone map for one destination and arrival."""

from pathlib import Path

import numpy as np
import pandas as pd

from isochrone.csa import Network, latest_departures
from isochrone.osm import WalkingGraph, find_nearby_platforms_on_network, walking_distances
from isochrone.walking import latest_departures_from_points, walking_time_s


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


def travel_minutes(
    graph: WalkingGraph,
    grid: np.ndarray,
    platforms: pd.DataFrame,
    access: pd.DataFrame,
    network: Network,
    destination_xy: np.ndarray,
    arrival_s: int,
    max_duration_s: int = 7200,
) -> np.ndarray:
    """Return the travel time from each grid cell to the destination, arriving by arrival_s.

    The travel time of a cell is the arrival time minus its latest departure: walk to a nearby
    platform and continue by public transport, or walk directly to the destination. If no
    platform is within walking distance of the destination, only the direct walk counts.

    Args:
        graph: Pedestrian network, as returned by build_walking_graph.
        grid: Coordinates of the grid cells in metres in PROJECTED_CRS, with shape (n, 2).
        platforms: DataFrame with columns stop_id, x and y, as returned by load_platforms.
        access: Walking legs from the grid cells to nearby platforms, as returned by
            load_or_compute_access.
        network: Network of the service day, as returned by build_network.
        destination_xy: Coordinates of the destination in metres in PROJECTED_CRS, shape (2,).
        arrival_s: Latest arrival time at the destination, seconds after midnight.
        max_duration_s: Maximum duration of a journey in seconds.

    Returns:
        Travel time in minutes for each grid cell, as a float array of length n; NaN for
        cells that cannot reach the destination within max_duration_s.
    """
    egress = find_nearby_platforms_on_network(graph, destination_xy, platforms)
    targets = dict(zip(egress["stop_id"], (arrival_s - egress["walk_s"]).tolist()))
    departures = latest_departures(network, targets, max_duration_s) if targets else {}

    direct = walking_distances(
        graph, np.atleast_2d(destination_xy), grid, max_distance_m=float(max_duration_s)
    )
    direct_walk_s = np.full(len(grid), np.nan)
    direct_walk_s[direct["target"]] = walking_time_s(direct["distance_m"], detour_factor=1.0)

    latest = latest_departures_from_points(
        access, departures, direct_walk_s, arrival_s, max_duration_s
    )
    return (arrival_s - latest) / 60
