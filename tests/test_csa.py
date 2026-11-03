"""Tests for the reverse Connection Scan Algorithm."""

import pandas as pd
import pytest

from isochrone.csa import build_network, latest_departures


def make_network():
    """Return a small synthetic network whose latest departures to stop D are known.

    Times are written in minutes after midnight and converted to seconds. Every stop has a
    transfer to itself of 2 minutes; stop Y has a walking transfer of 5 minutes to stop B.

    Returns:
        The Network of the synthetic connections and transfers.
    """
    rows = [
        # trip, dep_stop, arr_stop, stop_sequence, dep_min, arr_min, can_board, can_alight
        ("t1", "A", "B", 1, 480, 490, True, True),  # stay on board at B
        ("t1", "B", "D", 2, 490, 500, True, True),
        ("t2", "C", "B", 1, 470, 486, True, True),  # reaches B in time to transfer to t1
        ("t3", "X", "B", 1, 476, 489, True, True),  # one minute too late for t1
        ("t4", "E", "D", 1, 478, 505, True, False),  # alighting at D not allowed
        ("t5", "F", "G", 1, 492, 492, True, True),  # same time, listed before the next one
        ("t5", "G", "D", 2, 492, 495, True, True),
        ("t6", "A", "D", 1, 470, 495, True, True),  # earlier than t1 from A
        ("t7", "H", "Y", 1, 470, 482, True, True),  # walks from Y to B, then takes t1
    ]
    columns = [
        "trip_id",
        "dep_stop",
        "arr_stop",
        "stop_sequence",
        "dep_time",
        "arr_time",
        "can_board",
        "can_alight",
    ]
    connections = pd.DataFrame(rows, columns=columns)
    connections[["dep_time", "arr_time"]] *= 60

    stops = ["A", "B", "C", "D", "E", "F", "G", "H", "X", "Y"]
    transfers = pd.DataFrame(
        {"from_stop": stops + ["Y"], "to_stop": stops + ["B"], "duration_s": [120] * 10 + [300]}
    )
    return build_network(connections, transfers)


def test_latest_departures_synthetic():
    """Check the latest departures to D on the synthetic network, arriving by 08:30."""
    departures = latest_departures(make_network(), {"D": 510 * 60})

    expected_min = {"A": 480, "B": 490, "C": 470, "F": 492, "G": 492, "H": 470}
    assert departures == {stop: minute * 60 for stop, minute in expected_min.items()}


def test_latest_departures_needs_targets():
    """Check that an empty set of targets is rejected."""
    with pytest.raises(ValueError):
        latest_departures(make_network(), {})
