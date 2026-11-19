"""Tests for the isochrone bands and their map."""

import html
import json
import re

import folium
import numpy as np
import pytest
import shapely
from pyproj import Transformer

from isochrone.maps import (
    band_pair_polygons,
    band_polygons,
    isochrone_map,
    pair_control_html,
    pair_map,
    threshold_control_html,
)
from isochrone.walking import PROJECTED_CRS


def test_band_polygons_merges_cells_of_each_band():
    """Limits belong to the lower band, adjacent cells merge, NaN and late cells are left out."""
    x0, y0 = 690000.0, 5335000.0
    points_xy = np.array([[x0 + 100.0 * k, y0] for k in range(6)])
    minutes = np.array([0.0, 5.0, 4.0, 5.5, np.nan, 25.0])

    bands = band_polygons(points_xy, minutes, bands_min=(0, 5, 10, 20), step_m=100.0)

    assert bands[["low_min", "high_min", "n_cells"]].values.tolist() == [[0, 5, 3], [5, 10, 1]]

    to_metres = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    first = shapely.transform(
        bands["geometry"][0], lambda c: np.column_stack(to_metres.transform(c[:, 0], c[:, 1]))
    )
    assert first.geom_type == "Polygon"
    assert shapely.get_num_coordinates(first) == 5
    assert abs(first.area - 3 * 100.0 * 100.0) < 0.001 * 3 * 100.0 * 100.0


def test_band_pair_polygons_keeps_cells_that_reach_both_destinations():
    """One row per pair of bands present; cells out of the bands for either one are left out."""
    x0, y0 = 690000.0, 5335000.0
    points_xy = np.array([[x0 + 100.0 * k, y0] for k in range(5)])
    minutes_a = np.array([3.0, 4.0, 8.0, np.nan, 3.0])
    minutes_b = np.array([12.0, 15.0, 12.0, 5.0, 25.0])

    pairs = band_pair_polygons(points_xy, minutes_a, minutes_b, bands_min=(0, 5, 10, 20))

    columns = ["low_a_min", "high_a_min", "low_b_min", "high_b_min", "n_cells"]
    assert pairs[columns].values.tolist() == [[0, 5, 10, 20, 2], [5, 10, 10, 20, 1]]

    to_metres = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    first = shapely.transform(
        pairs["geometry"][0], lambda c: np.column_stack(to_metres.transform(c[:, 0], c[:, 1]))
    )
    assert first.geom_type == "Polygon"
    assert abs(first.area - 2 * 100.0 * 100.0) < 0.001 * 2 * 100.0 * 100.0


def test_isochrone_map_has_one_layer_per_band_with_distinct_colours():
    """One layer per band, distinct colours, decreasing opacity, overrides, one marker."""
    x0, y0 = 690000.0, 5335000.0
    points_xy = np.array([[x0, y0], [x0 + 100.0, y0], [x0 + 200.0, y0]])
    bands = band_polygons(points_xy, np.array([3.0, 8.0, 15.0]), bands_min=(0, 5, 10, 20))

    def styles_of(isochrones):
        layers = [c for c in isochrones._children.values() if isinstance(c, folium.GeoJson)]
        return [layer.style_function(None) for layer in layers]

    isochrones = isochrone_map(bands, (11.58, 48.14), style_overrides={})
    styles = styles_of(isochrones)
    colours = {style["fillColor"] for style in styles}
    markers = [c for c in isochrones._children.values() if isinstance(c, folium.Marker)]
    assert len(styles) == 3
    assert len(colours) == 3
    assert len(markers) == 1
    assert [style["fillOpacity"] for style in styles] == [0.8, 0.55, 0.3]

    overrides = {10: ("#123456", 0.9)}
    overridden = styles_of(isochrone_map(bands, (11.58, 48.14), style_overrides=overrides))
    assert overridden[1] == {"fillColor": "#123456", "fillOpacity": 0.9, "weight": 0}


def test_pair_map_has_one_hidden_layer_per_pair_and_a_marker_per_destination():
    """One layer per pair, transparent and of one colour, named in the page; two markers."""
    x0, y0 = 690000.0, 5335000.0
    points_xy = np.array([[x0, y0], [x0 + 100.0, y0], [x0 + 200.0, y0]])
    minutes_a = np.array([3.0, 8.0, 8.0])
    minutes_b = np.array([12.0, 12.0, 4.0])
    pairs = band_pair_polygons(points_xy, minutes_a, minutes_b, bands_min=(0, 5, 10, 20))

    combined = pair_map(
        pairs, ((11.58, 48.14), (11.66, 48.26)), ("A", "B"), bands_min=(0, 5, 10, 20)
    )

    children = combined._children.values()
    layers = [c for c in children if isinstance(c, folium.GeoJson)]
    markers = [c for c in children if isinstance(c, folium.Marker)]
    hidden = {"fillColor": "#E66100", "fillOpacity": 0, "weight": 0}
    assert len(layers) == 3
    assert all(layer.style_function(None) == hidden for layer in layers)
    assert len(markers) == 2
    assert combined.location == pytest.approx([48.2, 11.62])

    page = combined.get_root().render()
    assert all(layer.get_name() in page for layer in layers)
    assert "A: 5–10 min, B: 0–5 min" in page


def test_threshold_control_html_passes_bands_and_starts_at_default():
    """The bands reach the script as JSON; the slider starts at the first limit >= default."""
    control = threshold_control_html(
        ["geo_json_a", "geo_json_b", "geo_json_c"],
        [15.0, 30.0, 45.0],
        ["#4B0A6E", "#1C4E8D", "#04B16F"],
        [0.9, 0.8, 0.7],
    )
    data = re.search(r'data-bands="([^"]*)"', control).group(1)
    bands = json.loads(html.unescape(data))
    assert [band["name"] for band in bands] == ["geo_json_a", "geo_json_b", "geo_json_c"]
    assert [band["high"] for band in bands] == [15.0, 30.0, 45.0]
    assert 'max="2"' in control and 'value="1"' in control

    short = threshold_control_html(["geo_json_a"], [10.0], ["#4B0A6E"], [0.9])
    assert 'value="0"' in short


def test_pair_control_html_passes_pairs_and_starts_each_slider_at_its_default():
    """Pairs and limits reach the script as JSON; each slider starts at its own default."""
    control = pair_control_html(
        ["geo_json_a", "geo_json_b"],
        [15.0, 30.0],
        [45.0, 15.0],
        ("Garching (9:00)", "Pool <19:00>"),
        [15.0, 30.0, 45.0],
        default_min=(20.0, 50.0),
    )
    data = re.search(r'data-pairs="([^"]*)"', control).group(1)
    pairs = json.loads(html.unescape(data))
    assert pairs == [
        {"name": "geo_json_a", "a": 15.0, "b": 45.0},
        {"name": "geo_json_b", "a": 30.0, "b": 15.0},
    ]
    limits = re.search(r'data-limits="([^"]*)"', control).group(1)
    assert json.loads(html.unescape(limits)) == [15.0, 30.0, 45.0]
    assert re.findall(r'type="range"[^>]*value="(\d+)"', control) == ["1", "2"]
    assert "Pool &lt;19:00&gt;" in control
