"""Read tables of a GTFS feed stored as a ZIP file."""

import datetime as dt
import zipfile
from pathlib import Path

import pandas as pd


def read_gtfs_table(
    feed_path: Path, name: str, usecols: list[str] | None = None
) -> pd.DataFrame:
    """Read one table of a GTFS feed directly from the ZIP file.

    All columns are read as strings: GTFS identifiers may look like numbers but must
    keep leading zeros, and times can exceed 24:00:00, so conversions are done
    explicitly later. Some files of the MVV feed start with a byte order mark (BOM):
    the "utf-8-sig" encoding states explicitly that it must be removed.

    Args:
        feed_path: Path of the GTFS ZIP file.
        name: Table name without the ".txt" extension, e.g. "stops".
        usecols: Columns to read. If None, all columns are read.

    Returns:
        The table as a DataFrame where every column has string values.
    """
    with zipfile.ZipFile(feed_path) as feed, feed.open(f"{name}.txt") as f:
        return pd.read_csv(f, dtype=str, usecols=usecols, encoding="utf-8-sig")


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
ON_DEMAND_MARKER = "flexlinie"


def active_service_ids(feed_path: Path, date: dt.date) -> set[str]:
    """Return the service_id values that run on a given date.

    A service is active if the date lies in its calendar.txt range and its weekday column
    is "1", or if calendar_dates.txt adds it on that date (exception_type "1"). Services
    removed on that date (exception_type "2") are excluded in both cases.

    Args:
        feed_path: Path of the GTFS ZIP file.
        date: Service date.

    Returns:
        The set of active service_id values (empty if the date is outside the feed).
    """
    day = date.strftime("%Y%m%d")
    weekday = WEEKDAYS[date.weekday()]

    calendar = read_gtfs_table(feed_path, "calendar")
    in_range = (calendar["start_date"] <= day) & (day <= calendar["end_date"])
    runs_on_weekday = calendar[weekday] == "1"
    active = set(calendar.loc[in_range & runs_on_weekday, "service_id"])

    exceptions = read_gtfs_table(feed_path, "calendar_dates")
    on_day = exceptions[exceptions["date"] == day]
    added = set(on_day.loc[on_day["exception_type"] == "1", "service_id"])
    removed = set(on_day.loc[on_day["exception_type"] == "2", "service_id"])
    return (active | added) - removed


def active_trip_ids(feed_path: Path, date: dt.date) -> set[str]:
    """Return the trip_id values that run on a given date.

    Args:
        feed_path: Path of the GTFS ZIP file.
        date: Service date.

    Returns:
        The set of active trip_id values (empty if the date is outside the feed).
    """
    trips = read_gtfs_table(feed_path, "trips", usecols=["trip_id", "service_id"])
    services = active_service_ids(feed_path, date)
    return set(trips.loc[trips["service_id"].isin(services), "trip_id"])


def feed_period(feed_path: Path) -> tuple[dt.date, dt.date]:
    """Return the first and last date that appear in the service calendar of a feed.

    The dates come from calendar.txt (start_date, end_date) and calendar_dates.txt (date),
    the same tables that active_service_ids reads; a feed must contain at least one of them.
    feed_info.txt is not used: it is optional in GTFS, and so are its dates.

    Args:
        feed_path: Path of the GTFS ZIP file.

    Returns:
        The first and the last date of the feed.

    Raises:
        ValueError: If neither table contains a date.
    """
    with zipfile.ZipFile(feed_path) as feed:
        names = set(feed.namelist())
    dates = []
    if "calendar.txt" in names:
        calendar = read_gtfs_table(feed_path, "calendar", usecols=["start_date", "end_date"])
        dates += [calendar["start_date"], calendar["end_date"]]
    if "calendar_dates.txt" in names:
        dates.append(read_gtfs_table(feed_path, "calendar_dates", usecols=["date"])["date"])

    all_dates = pd.concat(dates) if dates else pd.Series(dtype=str)
    if all_dates.empty:
        raise ValueError(f"no service dates in {feed_path}")
    return (
        dt.datetime.strptime(all_dates.min(), "%Y%m%d").date(),
        dt.datetime.strptime(all_dates.max(), "%Y%m%d").date(),
    )


def on_demand_trip_ids(feed_path: Path) -> set[str]:
    """Return the trip_id values of on-demand services (MVV FLEXlinie).

    FLEXlinie trips follow a timetable but run only if booked in advance. The MVV feed does
    not mark them with pickup_type = 2, so they are recognised by ON_DEMAND_MARKER in the
    trip_id, ignoring case. This rule is specific to the MVV feed: with other feeds it
    returns an empty set.

    Args:
        feed_path: Path of the GTFS ZIP file.

    Returns:
        The set of on-demand trip_id values over the whole feed period.
    """
    trips = read_gtfs_table(feed_path, "trips", usecols=["trip_id"])
    is_on_demand = trips["trip_id"].str.contains(ON_DEMAND_MARKER, case=False, regex=False)
    return set(trips.loc[is_on_demand, "trip_id"])


def gtfs_time_to_seconds(times: pd.Series) -> pd.Series:
    """Convert GTFS times "HH:MM:SS" to seconds after midnight of the service day.

    Hours can be 24 or more for trips that continue after midnight: they are kept as they
    are, so that times of the same service day stay in chronological order.

    Args:
        times: Series of strings in the format "HH:MM:SS".

    Returns:
        Series of integers (int64) with the same index.
    """
    parts = times.str.split(":", expand=True).astype("int64")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
