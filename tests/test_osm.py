"""Tests for the pedestrian network built from OpenStreetMap."""

import numpy as np
import pandas as pd
import pytest
import shapely

from isochrone.osm import (
    build_walking_graph,
    find_nearby_platforms_on_network,
    is_walkable,
    read_poly,
    snap_to_graph,
    walking_distances,
)
from isochrone.walking import to_projected


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"highway": "residential"}, True),  # walkable by default
        ({"highway": "primary_link"}, True),
        ({"highway": "corridor", "indoor": "yes"}, True),
        ({"highway": "motorway"}, False),  # not in the list of walkable types
        ({"highway": "motorway", "foot": "yes"}, False),
        ({"highway": "construction"}, False),
        ({"building": "yes"}, False),  # not a road
        ({"highway": "cycleway"}, False),  # not walkable by default in Germany
        ({"highway": "cycleway", "foot": "designated"}, True),  # shared foot and cycle path
        ({"highway": "service", "access": "private"}, False),
        ({"highway": "service", "access": "private", "foot": "yes"}, True),  # foot overrides
        ({"highway": "track", "access": "agricultural"}, True),  # restricts vehicles only
        ({"highway": "footway", "foot": "no"}, False),
        ({"highway": "secondary", "foot": "use_sidepath"}, False),
        ({"highway": "secondary", "sidewalk": "separate"}, False),
        ({"highway": "secondary", "sidewalk:left": "separate"}, True),  # only one side separate
        ({"highway": "primary", "motorroad": "yes", "foot": "yes"}, False),
        ({"highway": "pedestrian", "area": "yes"}, False),
    ],
)
def test_is_walkable(tags, expected):
    """Check the walking rules on hand-written tags."""
    assert is_walkable(tags) == expected


def test_snap_to_graph(small_graph):
    """Points snap to the nearest node of the main component, never to the small one."""
    node, distance = snap_to_graph(small_graph, np.array([140.0, 290.0]), np.array([25.0, 0.0]))
    np.testing.assert_array_equal(node, [1, 3])
    np.testing.assert_allclose(distance, [np.hypot(40, 25), 10.0])


def test_walking_distances(small_graph):
    """Distances add the two snaps to the path on the graph; the limit applies to the total."""
    sources = np.array([[140.0, 25.0], [200.0, 0.0]])
    targets = np.array([[290.0, 0.0], [0.0, -20.0]])
    pairs = walking_distances(small_graph, sources, targets, max_distance_m=250.0, batch_size=1)
    snap = np.hypot(40, 25)
    assert pairs[["source", "target"]].values.tolist() == [[0, 1], [1, 0], [1, 1]]
    np.testing.assert_allclose(pairs["distance_m"], [snap + 100 + 20, 110.0, 220.0])


def test_find_nearby_platforms_on_network(small_graph):
    """A point gets the same rows whether Dijkstra runs from the points or the platforms."""
    platforms = pd.DataFrame({"stop_id": ["a", "b"], "x": [200.0, 290.0], "y": [0.0, 0.0]})
    point = np.array([140.0, 25.0])
    alone = find_nearby_platforms_on_network(small_graph, point, platforms, 300.0)
    snap = np.hypot(40, 25)
    assert alone["stop_id"].tolist() == ["a", "b"]
    np.testing.assert_allclose(alone["distance_m"], [snap + 100, snap + 200 + 10])
    assert alone["walk_s"].tolist() == [148, 258]

    points = np.array([[140.0, 25.0], [0.0, -20.0], [300.0, 0.0], [100.0, 0.0]])
    many = find_nearby_platforms_on_network(small_graph, points, platforms, 300.0)
    pd.testing.assert_frame_equal(many[many["point"] == 0], alone)


POLY_TEXT = """test
1
   0.0E+00   0.0E+00
   1.0E+01   0.0E+00
   1.0E+01   1.0E+01
   0.0E+00   1.0E+01
END
2
   2.0E+01   0.0E+00
   2.2E+01   0.0E+00
   2.2E+01   2.0E+00
   2.0E+01   2.0E+00
END
!hole
   2.0E+00   2.0E+00
   5.0E+00   2.0E+00
   5.0E+00   5.0E+00
   2.0E+00   5.0E+00
END
END
"""


def test_read_poly(tmp_path):
    """Sections are joined and sections whose name starts with "!" are subtracted."""
    poly_path = tmp_path / "test.poly"
    poly_path.write_text(POLY_TEXT)
    area = read_poly(poly_path)
    assert area.area == 100 + 4 - 9
    assert area.contains(shapely.Point(1, 1)) and area.contains(shapely.Point(21, 1))
    assert not area.contains(shapely.Point(3, 3))


OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1" lat="48.137" lon="11.575"/>
  <node id="2" lat="48.137" lon="11.576"/>
  <node id="3" lat="48.138" lon="11.576"/>
  <node id="4" lat="48.138" lon="11.577"/>
  <node id="5" lat="48.139" lon="11.577"/>
  <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><tag k="highway" v="footway"/></way>
  <way id="11"><nd ref="3"/><nd ref="4"/><tag k="highway" v="residential"/></way>
  <way id="12"><nd ref="4"/><nd ref="5"/><tag k="highway" v="motorway"/></way>
  <way id="13"><nd ref="2"/><nd ref="3"/><tag k="highway" v="footway"/></way>
</osm>
"""


def test_build_walking_graph(tmp_path):
    """Walkable ways become undirected edges joined at shared nodes; others are left out."""
    osm_path = tmp_path / "test.osm"
    osm_path.write_text(OSM_XML)
    graph = build_walking_graph(osm_path)
    np.testing.assert_array_equal(graph.node_ids, [1, 2, 3, 4])
    x, y = to_projected(np.array([11.575, 11.576, 11.576]), np.array([48.137, 48.137, 48.138]))
    np.testing.assert_allclose([graph.x[:3], graph.y[:3]], [x, y])
    edges = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    assert set(zip(*graph.lengths.nonzero())) == edges
    # Edge 2-3 appears in ways 10 and 13: its length must not be counted twice.
    assert graph.lengths[1, 2] == graph.lengths[2, 1] == np.hypot(x[2] - x[1], y[2] - y[1])
