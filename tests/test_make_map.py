"""Tests for the command that draws an isochrone map."""

import numpy as np
import pandas as pd

from isochrone.csa import build_network
from isochrone.make_map import load_or_compute_access, median_minutes, travel_minutes
from isochrone.osm import find_nearby_platforms_on_network


def test_load_or_compute_access(small_graph, tmp_path):
    """The first call writes the cache; the second reads the same legs without the graph."""
    platforms = pd.DataFrame({"stop_id": ["a", "b"], "x": [200.0, 290.0], "y": [0.0, 0.0]})
    points = np.array([[140.0, 25.0], [0.0, -20.0], [300.0, 0.0], [100.0, 0.0]])
    cache_path = tmp_path / "processed" / "access.npz"

    computed = load_or_compute_access(cache_path, small_graph, points, platforms, 300.0)
    assert cache_path.exists()
    assert len(computed) > 0

    cached = load_or_compute_access(cache_path, None, None, None, 300.0)
    pd.testing.assert_frame_equal(cached, computed)


def one_trip_network():
    """Return a network with one trip from platform P to platform Q, 08:00 to 08:01."""
    connections = pd.DataFrame(
        {
            "trip_id": ["t1"],
            "dep_stop": ["P"],
            "arr_stop": ["Q"],
            "stop_sequence": [1],
            "dep_time": [28800],
            "arr_time": [28860],
            "can_board": [True],
            "can_alight": [True],
        }
    )
    transfers = pd.DataFrame({"from_stop": ["P", "Q"], "to_stop": ["P", "Q"], "duration_s": 120})
    return build_network(connections, transfers)


def test_travel_minutes(small_graph):
    """Cells near P take the trip; the destination cell walks; a far cell is not reached.

    Arriving by 08:02 instead of 08:01, the cell 100 m from P walks to the destination (200 s)
    instead of taking the 08:00 trip.
    """
    platforms = pd.DataFrame({"stop_id": ["P", "Q"], "x": [0.0, 300.0], "y": [0.0, 0.0]})
    grid = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0], [0.0, -8000.0]])
    access = find_nearby_platforms_on_network(small_graph, grid, platforms)
    destination = np.array([300.0, 0.0])

    minutes = travel_minutes(
        small_graph, grid, platforms, access, one_trip_network(), destination, [28860, 28920]
    )
    np.testing.assert_allclose(
        minutes, [[1.0, 160 / 60, 0.0, np.nan], [2.0, 200 / 60, 0.0, np.nan]]
    )


def test_travel_minutes_without_platforms_near_destination(small_graph):
    """With no platform within walking distance of the destination, cells can only walk."""
    platforms = pd.DataFrame({"stop_id": ["P", "Q"], "x": [0.0, 0.0], "y": [-5000.0, -5000.0]})
    grid = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]])
    access = find_nearby_platforms_on_network(small_graph, grid, platforms)
    destination = np.array([300.0, 0.0])

    minutes = travel_minutes(
        small_graph, grid, platforms, access, one_trip_network(), destination, [28860]
    )
    np.testing.assert_allclose(minutes, [[5.0, 200 / 60, 0.0]])


def test_median_minutes():
    """A cell needs more than half of the arrival times to have a median; NaN counts as +inf."""
    minutes = np.array(
        [[10.0, 5.0, np.nan, np.nan], [20.0, np.nan, 7.0, np.nan], [30.0, np.nan, 9.0, np.nan]]
    )
    np.testing.assert_allclose(median_minutes(minutes), [20.0, np.nan, 9.0, np.nan])
    np.testing.assert_allclose(median_minutes(minutes[:2]), [15.0, np.nan, np.nan, np.nan])
