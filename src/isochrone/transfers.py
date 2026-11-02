"""Build the walking transfers between platforms."""

from pathlib import Path

import pandas as pd
from pyproj import Transformer

from isochrone.gtfs import read_gtfs_table

PROJECTED_CRS = "EPSG:25832"


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

    to_metres = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = to_metres.transform(
        platforms["stop_lon"].astype("float64").to_numpy(),
        platforms["stop_lat"].astype("float64").to_numpy(),
    )
    return pd.DataFrame(
        {
            "stop_id": platforms["stop_id"].to_numpy(),
            "parent_station": platforms["parent_station"].to_numpy(),
            "x": x,
            "y": y,
        }
    )

