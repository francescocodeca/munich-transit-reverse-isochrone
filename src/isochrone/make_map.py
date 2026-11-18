"""Command-line entry point that draws an isochrone map for one destination and arrival."""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from isochrone.connections import build_connections
from isochrone.csa import Network, build_network, latest_departures
from isochrone.download import RAW_DATA_DIR
from isochrone.gtfs import feed_period
from isochrone.maps import band_polygons, isochrone_map
from isochrone.osm import (
    WalkingGraph,
    build_walking_graph,
    find_nearby_platforms_on_network,
    read_poly,
    walking_distances,
)
from isochrone.transfers import build_transfers, load_platforms
from isochrone.walking import (
    build_grid,
    latest_departures_from_points,
    to_projected,
    walking_time_s,
)

ACCESS_RADIUS_M = 1300.0
PROCESSED_DIR = RAW_DATA_DIR.parent / "processed"
MAPS_DIR = RAW_DATA_DIR.parent / "maps"


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
    arrivals_s: list[int],
    max_duration_s: int = 7200,
) -> np.ndarray:
    """Return the travel time from each grid cell to the destination, for each arrival time.

    The travel time of a cell is the arrival time minus its latest departure: walk to a nearby
    platform and continue by public transport, or walk directly to the destination. If no
    platform is within walking distance of the destination, only the direct walk counts. The
    walks from the destination do not depend on the arrival time, so they are computed once.

    Args:
        graph: Pedestrian network, as returned by build_walking_graph.
        grid: Coordinates of the grid cells in metres in PROJECTED_CRS, with shape (n, 2).
        platforms: DataFrame with columns stop_id, x and y, as returned by load_platforms.
        access: Walking legs from the grid cells to nearby platforms, as returned by
            load_or_compute_access.
        network: Network of the service day, as returned by build_network.
        destination_xy: Coordinates of the destination in metres in PROJECTED_CRS, shape (2,).
        arrivals_s: Latest arrival times at the destination, seconds after midnight.
        max_duration_s: Maximum duration of a journey in seconds.

    Returns:
        Travel time in minutes, as a float array of shape (len(arrivals_s), n): row i for
        arrivals_s[i]; NaN for cells that cannot reach the destination within max_duration_s.
    """
    egress = find_nearby_platforms_on_network(graph, destination_xy, platforms)
    direct = walking_distances(
        graph, np.atleast_2d(destination_xy), grid, max_distance_m=float(max_duration_s)
    )
    direct_walk_s = np.full(len(grid), np.nan)
    direct_walk_s[direct["target"]] = walking_time_s(direct["distance_m"], detour_factor=1.0)

    minutes = np.full((len(arrivals_s), len(grid)), np.nan)
    for i, arrival_s in enumerate(arrivals_s):
        targets = dict(zip(egress["stop_id"], (arrival_s - egress["walk_s"]).tolist()))
        departures = latest_departures(network, targets, max_duration_s) if targets else {}
        latest = latest_departures_from_points(
            access, departures, direct_walk_s, arrival_s, max_duration_s
        )
        minutes[i] = (arrival_s - latest) / 60
    return minutes


def median_minutes(minutes: np.ndarray) -> np.ndarray:
    """Return the median travel time of each grid cell over the arrival times.

    A cell that cannot reach the destination for an arrival time counts as an infinitely long
    journey, so the median is finite only if the cell is reached for more than half of the
    arrival times.

    Args:
        minutes: Travel times in minutes with shape (m, n), as returned by travel_minutes; NaN
            for cells that cannot reach the destination.

    Returns:
        Median travel time in minutes for each grid cell, as a float array of length n; NaN
        for cells that are not reached for more than half of the arrival times.
    """
    durations = np.where(np.isnan(minutes), np.inf, minutes)
    median = np.median(durations, axis=0)
    return np.where(np.isinf(median), np.nan, median)


def latest_raw_file(pattern: str) -> Path:
    """Return the most recent file in the raw data folder whose name matches pattern.

    Downloaded files carry their download date as YYYY-MM-DD in the name, so the last one
    in alphabetical order is the most recent.

    Args:
        pattern: Glob pattern of the file name, e.g. "mvv_gtfs_*.zip".

    Returns:
        Path of the most recent matching file.

    Raises:
        FileNotFoundError: If no file matches pattern.
    """
    files = sorted(RAW_DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"no file matching {pattern} in {RAW_DATA_DIR}: run python -m isochrone.download"
        )
    return files[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line arguments.

    Args:
        argv: Arguments without the program name; None reads them from sys.argv.

    Returns:
        Namespace with attributes lat, lon (float), date (datetime.date), arrival
        (datetime.time), window (int, minutes), and feed, osm and output (Path, or None when
        not given).
    """
    parser = argparse.ArgumentParser(
        prog="python -m isochrone.make_map",
        description="Draw a map of the travel time by public transport and on foot to a "
        "destination in the Munich area, arriving by a given time.",
    )
    parser.add_argument("--lat", type=float, required=True, help="destination latitude")
    parser.add_argument("--lon", type=float, required=True, help="destination longitude")
    parser.add_argument(
        "--date",
        type=dt.date.fromisoformat,
        required=True,
        metavar="YYYY-MM-DD",
        help="service day",
    )
    parser.add_argument(
        "--arrival",
        type=dt.time.fromisoformat,
        required=True,
        metavar="HH:MM",
        help="latest arrival time at the destination",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        metavar="MIN",
        help="width of a window of arrival times centred on --arrival, one per minute; the map "
        "shows the median travel time (default: 0, only --arrival)",
    )
    parser.add_argument(
        "--feed", type=Path, help="GTFS feed (default: the most recent in data/raw)"
    )
    parser.add_argument(
        "--osm",
        type=Path,
        help="OpenStreetMap .osm.pbf extract (default: the most recent in data/raw)",
    )
    parser.add_argument(
        "--output", type=Path, help="HTML file (default: a name from the arguments, in data/maps)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Draw the isochrone map for the destination and arrival time given on the command line.

    The walking legs from the grid cells to the platforms are read from a cache in
    PROCESSED_DIR, named after the feed, the OpenStreetMap extract and the access radius; the
    first run for a pair of files computes them (about 3 minutes). With a window, the map shows
    the median travel time over one arrival time per minute, from window // 2 minutes before
    the arrival time to window // 2 minutes after.

    Args:
        argv: Arguments without the program name, as for parse_args.

    Returns:
        Path of the HTML map.

    Raises:
        SystemExit: If the window is negative, the date is outside the feed or the destination
            is outside the area of the OpenStreetMap extract.
    """
    args = parse_args(argv)
    if args.window < 0:
        raise SystemExit(f"the window must not be negative, got {args.window}")
    feed_path = args.feed or latest_raw_file("mvv_gtfs_*.zip")
    osm_path = args.osm or latest_raw_file("*.osm.pbf")
    area_name = osm_path.name.removesuffix(".osm.pbf")

    first, last = feed_period(feed_path)
    if not first <= args.date <= last:
        raise SystemExit(f"{args.date} is outside the feed {feed_path.name} ({first} to {last})")

    area = read_poly(osm_path.with_name(f"{area_name}.poly"))
    area_xy = shapely.transform(area, lambda lonlat: np.column_stack(to_projected(*lonlat.T)))
    destination_xy = np.array(to_projected(args.lon, args.lat))
    if not shapely.contains_xy(area_xy, *destination_xy):
        raise SystemExit(f"the destination is outside the area of {osm_path.name}")

    graph = build_walking_graph(osm_path)
    platforms = load_platforms(feed_path)
    grid = build_grid(platforms)
    grid = grid[shapely.contains_xy(area_xy, grid[:, 0], grid[:, 1])]
    cache_name = f"access_{feed_path.stem}_{area_name}_{ACCESS_RADIUS_M:.0f}m.npz"
    access = load_or_compute_access(
        PROCESSED_DIR / cache_name, graph, grid, platforms, ACCESS_RADIUS_M
    )
    network = build_network(build_connections(feed_path, args.date), build_transfers(feed_path))

    arrival_s = args.arrival.hour * 3600 + args.arrival.minute * 60 + args.arrival.second
    half_window = args.window // 2
    arrivals_s = [arrival_s + 60 * k for k in range(-half_window, half_window + 1)]
    minutes = median_minutes(
        travel_minutes(graph, grid, platforms, access, network, destination_xy, arrivals_s)
    )
    print(f"{np.isfinite(minutes).sum()} of {len(grid)} grid cells reach the destination")

    window_suffix = f"_w{args.window}" if args.window else ""
    map_name = (
        f"isochrone_{args.lat:.4f}_{args.lon:.4f}_{args.date}_{args.arrival:%H%M}"
        f"{window_suffix}.html"
    )
    output = args.output or MAPS_DIR / map_name
    output.parent.mkdir(parents=True, exist_ok=True)
    isochrone_map(band_polygons(grid, minutes), (args.lon, args.lat)).save(output)
    print(f"map saved to {output}")
    return output


if __name__ == "__main__":
    main()
