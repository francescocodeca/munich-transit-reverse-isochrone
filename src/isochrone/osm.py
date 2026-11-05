"""Build the pedestrian network from an OpenStreetMap extract."""

from collections.abc import Mapping

WALKABLE_HIGHWAYS = frozenset(
    {
        "footway",
        "pedestrian",
        "path",
        "steps",
        "corridor",
        "living_street",
        "residential",
        "service",
        "track",
        "unclassified",
        "tertiary",
        "tertiary_link",
        "secondary",
        "secondary_link",
        "primary",
        "primary_link",
        "trunk",
        "trunk_link",
    }
)
FOOT_ONLY_IF_ALLOWED = frozenset({"cycleway", "bridleway"})
FOOT_ALLOWED = frozenset({"yes", "designated", "permissive", "destination", "customers"})
FOOT_FORBIDDEN = frozenset({"no", "private", "use_sidepath"})
ACCESS_FORBIDDEN = frozenset({"no", "private"})


def is_walkable(tags: Mapping[str, str]) -> bool:
    """Return whether pedestrians may walk along an OpenStreetMap way.

    The type of road (highway) gives a default, following the German defaults of the
    OpenStreetMap wiki; an explicit foot tag overrides it, then a general access restriction.
    Ways that are never walkable (motorways, roads with motorroad=yes, ways under construction,
    platforms, ...) are rejected whatever their other tags. Roads whose sidewalks are mapped as
    separate ways, and areas (area=yes), are rejected, so that pedestrians use the separate
    ways.

    Args:
        tags: Tags of the way, e.g. {"highway": "cycleway", "foot": "designated"}.

    Returns:
        True if the way belongs to the pedestrian network.
    """
    highway = tags.get("highway")
    if highway in WALKABLE_HIGHWAYS:
        default = True
    elif highway in FOOT_ONLY_IF_ALLOWED:
        default = False
    else:
        return False
    if tags.get("motorroad") == "yes" or tags.get("area") == "yes":
        return False
    if tags.get("sidewalk") == "separate" or tags.get("sidewalk:both") == "separate":
        return False

    foot = tags.get("foot")
    if foot in FOOT_ALLOWED:
        return True
    if foot in FOOT_FORBIDDEN:
        return False
    if tags.get("access") in ACCESS_FORBIDDEN:
        return False
    return default
