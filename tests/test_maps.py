"""Tests for the isochrone bands and their map."""

import folium
import numpy as np
import shapely
from pyproj import Transformer

from isochrone.maps import band_polygons, isochrone_map
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
