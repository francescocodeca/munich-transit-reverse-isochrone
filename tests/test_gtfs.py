"""Tests for the GTFS reading functions."""

import pandas as pd

from isochrone.gtfs import gtfs_time_to_seconds, on_demand_trip_ids


def test_gtfs_time_to_seconds_keeps_hours_after_midnight():
    """Times after 24:00 stay on the same service day instead of wrapping to 0."""
    times = pd.Series(["00:00:00", "08:15:30", "25:10:00"])
    assert gtfs_time_to_seconds(times).tolist() == [0, 29730, 90600]


def test_on_demand_rule_matches_mvv_feed(mvv_feed):
    """The trip_id rule for FLEXlinie must still find trips in the current MVV feed."""
    assert len(on_demand_trip_ids(mvv_feed)) > 0
