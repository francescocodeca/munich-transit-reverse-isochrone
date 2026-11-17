"""Tests for the GTFS reading functions."""

import datetime as dt
import zipfile

import pandas as pd
import pytest

from isochrone.gtfs import (
    feed_period,
    gtfs_time_to_seconds,
    on_demand_trip_ids,
    read_gtfs_table,
)

CALENDAR_HEADER = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
)
CALENDAR_DATES_HEADER = "service_id,date,exception_type\n"


def write_feed(path, tables):
    """Write a ZIP file with one .txt file per table, from a dict {name: text}."""
    with zipfile.ZipFile(path, "w") as feed:
        for name, text in tables.items():
            feed.writestr(f"{name}.txt", text)
    return path


def test_gtfs_time_to_seconds_keeps_hours_after_midnight():
    """Times after 24:00 stay on the same service day instead of wrapping to 0."""
    times = pd.Series(["00:00:00", "08:15:30", "25:10:00"])
    assert gtfs_time_to_seconds(times).tolist() == [0, 29730, 90600]


def test_on_demand_rule_matches_mvv_feed(mvv_feed):
    """The trip_id rule for FLEXlinie must still find trips in the current MVV feed."""
    assert len(on_demand_trip_ids(mvv_feed)) > 0


@pytest.mark.parametrize(
    "tables, expected",
    [
        (
            {
                "calendar": CALENDAR_HEADER + "s1,1,1,1,1,1,0,0,20261005,20261130\n",
                "calendar_dates": CALENDAR_DATES_HEADER + "s2,20261201,1\ns2,20261002,1\n",
            },
            (dt.date(2026, 10, 2), dt.date(2026, 12, 1)),
        ),
        (
            {"calendar": CALENDAR_HEADER + "s1,1,1,1,1,1,0,0,20261005,20261130\n"},
            (dt.date(2026, 10, 5), dt.date(2026, 11, 30)),
        ),
        (
            {"calendar_dates": CALENDAR_DATES_HEADER + "s2,20261201,1\n"},
            (dt.date(2026, 12, 1), dt.date(2026, 12, 1)),
        ),
    ],
)
def test_feed_period(tmp_path, tables, expected):
    """The period spans calendar.txt and calendar_dates.txt, whichever of them exist."""
    assert feed_period(write_feed(tmp_path / "feed.zip", tables)) == expected


def test_feed_period_without_dates(tmp_path):
    """A feed with no service dates is rejected."""
    feed = write_feed(tmp_path / "feed.zip", {"calendar_dates": CALENDAR_DATES_HEADER})
    with pytest.raises(ValueError):
        feed_period(feed)


def test_feed_period_matches_mvv_feed_info(mvv_feed):
    """On the MVV feed, the calendar period equals the one declared in feed_info.txt."""
    info = read_gtfs_table(mvv_feed, "feed_info").iloc[0]
    declared = tuple(
        dt.datetime.strptime(info[column], "%Y%m%d").date()
        for column in ["feed_start_date", "feed_end_date"]
    )
    assert feed_period(mvv_feed) == declared
