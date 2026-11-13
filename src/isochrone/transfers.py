"""Build the walking transfers between platforms."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from isochrone.gtfs import read_gtfs_table
from isochrone.walking import to_projected, walking_time_s


def load_platforms(feed_path: Path) -> pd.DataFrame:
    """Return the platforms of the feed with projected coordinates in metres.

    Platforms are the stops where vehicles call (location_type "0", or empty, which GTFS
    defines as "0"). Coordinates are converted from WGS 84 longitude and latitude to UTM
    zone 32N (PROJECTED_CRS), so that distances can be computed with Pythagoras in metres.

    Args:
        feed_path: Path of the GTFS ZIP file.

    Returns:
        DataFrame with columns stop_id, parent_station, x and y (metres in PROJECTED_CRS).
    """
    stops = read_gtfs_table(
        feed_path,
        "stops",
        usecols=["stop_id", "stop_lat", "stop_lon", "location_type", "parent_station"],
    )
    platforms = stops[stops["location_type"].fillna("0") == "0"]

    x, y = to_projected(platforms["stop_lon"], platforms["stop_lat"])
    return pd.DataFrame(
        {
            "stop_id": platforms["stop_id"].to_numpy(),
            "parent_station": platforms["parent_station"].to_numpy(),
            "x": x,
            "y": y,
        }
    )


def find_platform_pairs(platforms: pd.DataFrame, radius_m: float = 400.0) -> pd.DataFrame:
    """Return the ordered pairs of distinct platforms that are linked by a walking transfer.

    Two platforms are linked if they belong to the same station (same parent_station), at any
    distance, or if their straight-line distance is at most radius_m. Each linked pair appears
    in both directions; pairs of a platform with itself are not included.

    Args:
        platforms: DataFrame with columns stop_id, parent_station, x and y, as returned by
            load_platforms.
        radius_m: Maximum straight-line distance in metres between platforms of different
            stations.

    Returns:
        DataFrame with columns from_stop, to_stop and distance_m (straight-line, metres).
    """
    xy = platforms[["x", "y"]].to_numpy()
    near = cKDTree(xy).query_pairs(r=radius_m, output_type="ndarray")

    positions = pd.DataFrame(
        {"i": np.arange(len(platforms)), "station": platforms["parent_station"].to_numpy()}
    ).dropna(subset=["station"])
    same = positions.merge(positions, on="station", suffixes=("_from", "_to"))
    same = same.loc[same["i_from"] < same["i_to"], ["i_from", "i_to"]].to_numpy()

    pairs = np.unique(np.vstack([near, same]), axis=0)
    pairs = np.vstack([pairs, pairs[:, ::-1]])
    i, j = pairs[:, 0], pairs[:, 1]

    stop_ids = platforms["stop_id"].to_numpy()
    return pd.DataFrame(
        {
            "from_stop": stop_ids[i],
            "to_stop": stop_ids[j],
            "distance_m": np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]),
        }
    )


def build_transfers(
    feed_path: Path,
    radius_m: float = 400.0,
    min_transfer_s: int = 120,
    walking_speed_mps: float = 1.0,
    detour_factor: float = 1.3,
    margin_s: int = 70,
) -> pd.DataFrame:
    """Return the walking transfers between platforms, with their durations.

    A transfer from a platform to itself takes min_transfer_s. A transfer between two distinct
    platforms linked by find_platform_pairs takes the walking time plus margin_s, and at least
    min_transfer_s. The walking time is the straight-line distance times detour_factor, at
    walking_speed_mps, rounded up to whole seconds.

    Args:
        feed_path: Path of the GTFS ZIP file.
        radius_m: Maximum straight-line distance in metres between platforms of different
            stations.
        min_transfer_s: Minimum duration of any transfer, in seconds.
        walking_speed_mps: Walking speed in metres per second.
        detour_factor: Ratio between the walking distance and the straight-line distance.
        margin_s: Extra time in seconds for a transfer between distinct platforms (getting off,
            stairs, finding the way).

    Returns:
        DataFrame with columns from_stop, to_stop, distance_m and duration_s (integer seconds).
    """
    platforms = load_platforms(feed_path)
    pairs = find_platform_pairs(platforms, radius_m)

    walk_s = walking_time_s(pairs["distance_m"], walking_speed_mps, detour_factor) + margin_s
    pairs["duration_s"] = np.maximum(walk_s, min_transfer_s).astype("int64")

    same_platform = pd.DataFrame(
        {
            "from_stop": platforms["stop_id"],
            "to_stop": platforms["stop_id"],
            "distance_m": 0.0,
            "duration_s": min_transfer_s,
        }
    )
    return pd.concat([same_platform, pairs], ignore_index=True)
