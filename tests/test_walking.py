"""Tests for the walking times and the walking legs between points and platforms."""

import numpy as np
import pandas as pd

from isochrone.walking import (
    build_grid,
    find_nearby_platforms,
    latest_departures_from_points,
    walking_time_s,
)


def test_walking_time_s_rounds_up_with_detour_factor_and_speed():
    """Distance times detour factor over speed, rounded up to whole seconds."""
    times = walking_time_s([0.0, 1.0, 100.0], walking_speed_mps=1.0, detour_factor=1.3)
    assert times.tolist() == [0, 2, 130]
    assert times.dtype == np.int64
    assert walking_time_s([100.0], walking_speed_mps=2.0, detour_factor=1.0).tolist() == [50]


def test_find_nearby_platforms_keeps_platforms_within_radius():
    """Zero distance and the radius itself are kept, farther platforms and points are not."""
    platforms = pd.DataFrame(
        {
            "stop_id": ["A", "B", "C", "E"],
            "x": [0.0, 300.0, 1000.0, 0.0],
            "y": [0.0, 400.0, 0.0, 1001.0],
        }
    )
    points_xy = np.array([[0.0, 0.0], [5000.0, 5000.0], [300.0, 400.0]])

    pairs = find_nearby_platforms(points_xy, platforms, radius_m=1000.0, detour_factor=1.3)

    assert pairs["point"].tolist() == [0, 0, 0, 2, 2, 2, 2]
    assert pairs["stop_id"].tolist() == ["A", "B", "C", "B", "A", "E", "C"]
    assert pairs["distance_m"].tolist()[:3] == [0.0, 500.0, 1000.0]
    assert pairs["walk_s"].tolist()[:3] == [0, 650, 1300]


def test_find_nearby_platforms_accepts_a_single_point():
    """A point of shape (2,) is point 0; with no platform within the radius the result is empty."""
    platforms = pd.DataFrame({"stop_id": ["A"], "x": [0.0], "y": [0.0]})

    near = find_nearby_platforms(np.array([0.0, 600.0]), platforms, radius_m=1000.0)
    far = find_nearby_platforms(np.array([0.0, 2000.0]), platforms, radius_m=1000.0)

    assert near["point"].tolist() == [0]
    assert near["stop_id"].tolist() == ["A"]
    assert far.empty
    assert list(far.columns) == ["point", "stop_id", "distance_m", "walk_s"]


def test_latest_departures_from_points_keeps_the_best_of_platforms_and_direct_walk():
    """Best platform, direct walk when faster, walk-only within max_duration_s, else NaN."""
    points_xy = np.array([[5000.0, 0.0], [100.0, 0.0], [9000.0, 0.0], [0.0, 7000.0]])
    access = pd.DataFrame(
        {
            "point": [0, 0, 1, 2],
            "stop_id": ["A", "B", "B", "C"],
            "walk_s": [100, 300, 50, 10],
        }
    )
    departures = {"A": 30000, "B": 31000}

    latest = latest_departures_from_points(
        points_xy,
        access,
        departures,
        destination_xy=np.array([0.0, 0.0]),
        arrival_s=32400,
        max_duration_s=7200,
        detour_factor=1.0,
    )

    np.testing.assert_array_equal(latest, [30700.0, 32300.0, np.nan, 25400.0])


def test_build_grid_keeps_aligned_points_within_radius():
    """Points on multiples of step_m within radius_m of a platform, the radius included."""
    platforms = pd.DataFrame({"x": [1050.0], "y": [2000.0]})

    grid = build_grid(platforms, step_m=100.0, radius_m=150.0, n_strips=3)

    points = set(map(tuple, grid.tolist()))
    assert len(points) == len(grid)
    assert points == {
        (1000.0, 1900.0), (1100.0, 1900.0),
        (900.0, 2000.0), (1000.0, 2000.0), (1100.0, 2000.0), (1200.0, 2000.0),
        (1000.0, 2100.0), (1100.0, 2100.0),
    }
