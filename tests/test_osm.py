"""Tests for the pedestrian network built from OpenStreetMap."""

import pytest

from isochrone.osm import is_walkable


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
