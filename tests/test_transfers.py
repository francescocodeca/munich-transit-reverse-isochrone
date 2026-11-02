"""Tests for the walking transfers between platforms."""

import pandas as pd

from isochrone.transfers import build_transfers, find_platform_pairs, load_platforms


def test_find_platform_pairs_links_same_station_and_nearby_platforms():
    """Same station at any distance, other stations within the radius, no missing stations."""
    platforms = pd.DataFrame(
        {
            "stop_id": ["A", "B", "C", "D", "E", "F"],
            "parent_station": ["S1", "S1", "S2", "S3", None, None],
            "x": [0.0, 1000.0, 300.0, 0.0, 0.0, 5000.0],
            "y": [0.0, 0.0, 0.0, 500.0, 2000.0, 0.0],
        }
    )
    pairs = find_platform_pairs(platforms, radius_m=400.0)
    linked = set(zip(pairs["from_stop"], pairs["to_stop"]))
    assert linked == {("A", "B"), ("B", "A"), ("A", "C"), ("C", "A")}


def test_build_transfers_on_mvv_feed(mvv_feed):
    """Every platform has one self-transfer, arcs are unique and symmetric, none is too short."""
    transfers = build_transfers(mvv_feed, min_transfer_s=120)
    is_self = transfers["from_stop"] == transfers["to_stop"]

    assert set(transfers.loc[is_self, "from_stop"]) == set(load_platforms(mvv_feed)["stop_id"])
    assert (transfers.loc[is_self, "duration_s"] == 120).all()
    assert not transfers.duplicated(["from_stop", "to_stop"]).any()
    assert (transfers["duration_s"] >= 120).all()

    reverse = transfers.rename(columns={"from_stop": "to_stop", "to_stop": "from_stop"})
    both = transfers.merge(reverse, on=["from_stop", "to_stop"], suffixes=("", "_back"))
    assert len(both) == len(transfers)
    assert (both["duration_s"] == both["duration_s_back"]).all()
