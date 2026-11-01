"""Build the elementary connections of the timetable for one service day."""

import datetime as dt
from pathlib import Path

import pandas as pd

from isochrone.gtfs import (
    active_trip_ids,
    gtfs_time_to_seconds,
    on_demand_trip_ids,
    read_gtfs_table,
)

STOP_TIMES_COLUMNS = [
    "trip_id",
    "arrival_time",
    "departure_time",
    "stop_id",
    "stop_sequence",
    "pickup_type",
    "drop_off_type",
]


def build_connections(
    feed_path: Path, date: dt.date, include_on_demand: bool = False
) -> pd.DataFrame:
    """Return the elementary connections of all trips running on a given date.

    A connection is the ride of one trip between two consecutive stops: it departs from
    dep_stop at dep_time and arrives at arr_stop at arr_time, without intermediate stops.
    Times are seconds after midnight of the service day. Stops are kept at the level of
    stop_id (single platform).

    Args:
        feed_path: Path of the GTFS ZIP file.
        date: Service date.
        include_on_demand: If False, on-demand trips (FLEXlinie) are excluded.

    Returns:
        DataFrame with columns trip_id, dep_stop, arr_stop, dep_time, arr_time, can_board
        (boarding allowed at dep_stop) and can_alight (alighting allowed at arr_stop).
    """
    trips = active_trip_ids(feed_path, date)
    if not include_on_demand:
        trips -= on_demand_trip_ids(feed_path)

    stop_times = read_gtfs_table(feed_path, "stop_times", usecols=STOP_TIMES_COLUMNS)
    stop_times = stop_times[stop_times["trip_id"].isin(trips)]
    stop_times = stop_times.assign(stop_sequence=stop_times["stop_sequence"].astype("int64"))
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"], ignore_index=True)

    following = stop_times.shift(-1)
    same_trip = stop_times["trip_id"] == following["trip_id"]
    dep = stop_times[same_trip]
    arr = following[same_trip]

    connections = pd.DataFrame(
        {
            "trip_id": dep["trip_id"],
            "dep_stop": dep["stop_id"],
            "arr_stop": arr["stop_id"],
            "dep_time": gtfs_time_to_seconds(dep["departure_time"]),
            "arr_time": gtfs_time_to_seconds(arr["arrival_time"]),
            "can_board": dep["pickup_type"] != "1",
            "can_alight": arr["drop_off_type"] != "1",
        }
    )
    return connections.reset_index(drop=True)
