"""Compare the travel times of the isochrone map with those of r5py (R5 routing engine)."""

import datetime as dt
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import r5py
import shapely


def write_feed_without_empty_tables(feed_path: Path, out_path: Path) -> list[str]:
    """Copy a GTFS feed, leaving out the tables that have a header but no rows.

    R5 refuses a feed with an empty table (e.g. the MVV feed has a shapes.txt with only the
    header), while an absent optional table is allowed. The other tables are copied unchanged.

    Args:
        feed_path: Path of the GTFS zip file.
        out_path: Path of the zip file to write; its folder is created if needed.

    Returns:
        The names of the tables left out, e.g. ["shapes.txt"].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    removed = []
    with zipfile.ZipFile(feed_path) as source, zipfile.ZipFile(
        out_path, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            with source.open(info) as table:
                table.readline()
                if not table.readline().strip():
                    removed.append(info.filename)
                    continue
            with source.open(info) as table, target.open(info.filename, "w") as copy:
                shutil.copyfileobj(table, copy)
    return removed


def r5py_latest_departures(
    network: r5py.TransportNetwork,
    origins_lonlat: np.ndarray,
    destination_lonlat: tuple[float, float],
    date: dt.date,
    arrival_s: int,
    max_duration_s: int = 7200,
    walking_speed_mps: float = 1.0,
) -> np.ndarray:
    """Return the latest departure from each origin that reaches the destination on time, by R5.

    R5 only routes forward from a departure time, so it is run once per minute from
    arrival_s - max_duration_s to arrival_s, each time with a window of one minute; the latest
    departure of an origin is the last minute whose departure plus travel time is at most
    arrival_s. Trips may combine public transport and walking, or be walked entirely.

    Args:
        network: R5 network built from the OpenStreetMap extract and the GTFS feed.
        origins_lonlat: Array of shape (n, 2) with the longitude and latitude of the origins.
        destination_lonlat: Longitude and latitude of the destination.
        date: Day of travel.
        arrival_s: Latest arrival time, in seconds after midnight; a multiple of 60.
        max_duration_s: Longest trip considered, in seconds; a multiple of 60.
        walking_speed_mps: Walking speed in metres per second.

    Returns:
        Latest departure of each origin in seconds after midnight (whole minutes), NaN where
        the destination cannot be reached by arrival_s within max_duration_s.
    """
    origins = gpd.GeoDataFrame(
        {"id": np.arange(len(origins_lonlat))},
        geometry=gpd.points_from_xy(origins_lonlat[:, 0], origins_lonlat[:, 1]),
        crs="EPSG:4326",
    )
    destination = gpd.GeoDataFrame(
        {"id": [0]}, geometry=[shapely.Point(destination_lonlat)], crs="EPSG:4326"
    )
    midnight = dt.datetime.combine(date, dt.time())
    latest = np.full(len(origins), np.nan)
    for departure_s in range(arrival_s - max_duration_s, arrival_s + 1, 60):
        matrix = r5py.TravelTimeMatrix(
            network,
            origins=origins,
            destinations=destination,
            departure=midnight + dt.timedelta(seconds=departure_s),
            departure_time_window=dt.timedelta(minutes=1),
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            max_time=dt.timedelta(seconds=max_duration_s),
            speed_walking=walking_speed_mps * 3.6,
        )
        travel_s = matrix.sort_values("from_id")["travel_time"].to_numpy() * 60
        latest[departure_s + travel_s <= arrival_s] = departure_s
    return latest
