"""Turn travel times on the grid into isochrone bands and draw them on a map."""

import html
import json

import folium
import numpy as np
import pandas as pd
import shapely
from branca.colormap import linear
from pyproj import Transformer

from isochrone.walking import PROJECTED_CRS

DEFAULT_BANDS_MIN = (*range(0, 60, 5), 60, 70, 80, 90, 105, 120)

# Colour and opacity of the bands up to 60 min, keyed by the upper limit of the band: more
# saturated and more opaque than the viridis default, which looks dull there.
DEFAULT_STYLE_OVERRIDES = {
    5: ("#4B0A6E", 0.95),
    10: ("#4A1683", 0.95),
    15: ("#402592", 0.95),
    20: ("#322F8F", 0.95),
    25: ("#263A8A", 0.92),
    30: ("#1C4E8D", 0.92),
    35: ("#12618E", 0.91),
    40: ("#0B748E", 0.90),
    45: ("#04868E", 0.88),
    50: ("#009589", 0.86),
    55: ("#00A37F", 0.84),
    60: ("#04B16F", 0.82),
}


def band_index(
    minutes: np.ndarray, bands_min: tuple[float, ...] = DEFAULT_BANDS_MIN
) -> np.ndarray:
    """Find the travel time band of each value.

    A value belongs to the band (low, high] that contains it; the first band also contains its
    lower limit.

    Args:
        minutes: Travel times in minutes, NaN if not reachable.
        bands_min: Increasing limits of the bands in minutes.

    Returns:
        Integer array with the index of the band of each value (0 for the first band), -1 for
        NaN and for values outside the limits.
    """
    minutes = np.asarray(minutes, dtype="float64")
    limits = np.asarray(bands_min, dtype="float64")
    band = np.digitize(minutes, limits[1:-1], right=True)
    inside = (minutes >= limits[0]) & (minutes <= limits[-1])
    return np.where(inside, band, -1)


def cells_polygon(cells_xy: np.ndarray, step_m: float = 100.0) -> shapely.Geometry:
    """Merge grid cells into one geometry in longitude and latitude.

    Each cell is a square of side step_m centred on its grid point.

    Args:
        cells_xy: Coordinates of the grid points in metres in PROJECTED_CRS, with shape (n, 2).
        step_m: Distance in metres between neighbouring grid points.

    Returns:
        Shapely geometry in WGS 84 longitude and latitude, rounded to 1e-6 degrees.
    """
    to_lonlat = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)
    half = step_m / 2
    x, y = cells_xy[:, 0], cells_xy[:, 1]
    merged = shapely.coverage_union_all(shapely.box(x - half, y - half, x + half, y + half))
    merged = shapely.simplify(merged, 0)
    merged = shapely.transform(
        merged, lambda xy: np.column_stack(to_lonlat.transform(xy[:, 0], xy[:, 1]))
    )
    return shapely.set_precision(merged, 1e-6)


def band_polygons(
    points_xy: np.ndarray,
    minutes: np.ndarray,
    bands_min: tuple[float, ...] = DEFAULT_BANDS_MIN,
    step_m: float = 100.0,
) -> pd.DataFrame:
    """Merge the grid cells into one polygon per travel time band, in longitude and latitude.

    A cell belongs to the band (low, high] that contains its travel time; the first band also
    contains its lower limit. Each cell is a square of side step_m centred on its grid point.
    Cells without a travel time (NaN) or beyond the last limit are left out.

    Args:
        points_xy: Coordinates of the grid points in metres in PROJECTED_CRS, with shape (n, 2).
        minutes: Travel time of each grid point in minutes, NaN if not reachable.
        bands_min: Increasing limits of the bands in minutes.
        step_m: Distance in metres between neighbouring grid points.

    Returns:
        DataFrame with one row per band that contains at least one cell, and columns low_min,
        high_min, n_cells and geometry (shapely geometry in WGS 84 longitude and latitude).
    """
    limits = np.asarray(bands_min, dtype="float64")
    band = band_index(minutes, bands_min)
    rows = []
    for i in range(len(limits) - 1):
        cells = points_xy[band == i]
        if len(cells) == 0:
            continue
        rows.append(
            {
                "low_min": limits[i],
                "high_min": limits[i + 1],
                "n_cells": len(cells),
                "geometry": cells_polygon(cells, step_m),
            }
        )
    return pd.DataFrame(rows, columns=["low_min", "high_min", "n_cells", "geometry"])


def band_pair_polygons(
    points_xy: np.ndarray,
    minutes_a: np.ndarray,
    minutes_b: np.ndarray,
    bands_min: tuple[float, ...] = DEFAULT_BANDS_MIN,
    step_m: float = 100.0,
) -> pd.DataFrame:
    """Merge the grid cells into one polygon per pair of travel time bands to two destinations.

    The bands are those of band_index. Cells outside the bands for either destination are left
    out, so the polygons cover only the cells that reach both destinations.

    Args:
        points_xy: Coordinates of the grid points in metres in PROJECTED_CRS, with shape (n, 2).
        minutes_a: Travel time of each grid point to the first destination in minutes, NaN if
            not reachable.
        minutes_b: Travel time of each grid point to the second destination in minutes, NaN if
            not reachable.
        bands_min: Increasing limits of the bands in minutes.
        step_m: Distance in metres between neighbouring grid points.

    Returns:
        DataFrame with one row per pair of bands that contains at least one cell, and columns
        low_a_min, high_a_min, low_b_min, high_b_min (limits of the two bands), n_cells and
        geometry (shapely geometry in WGS 84 longitude and latitude).
    """
    limits = np.asarray(bands_min, dtype="float64")
    band_a = band_index(minutes_a, bands_min)
    band_b = band_index(minutes_b, bands_min)
    both = (band_a >= 0) & (band_b >= 0)
    pairs = np.unique(np.column_stack([band_a[both], band_b[both]]), axis=0)
    rows = []
    for i, j in pairs:
        cells = points_xy[(band_a == i) & (band_b == j)]
        rows.append(
            {
                "low_a_min": limits[i],
                "high_a_min": limits[i + 1],
                "low_b_min": limits[j],
                "high_b_min": limits[j + 1],
                "n_cells": len(cells),
                "geometry": cells_polygon(cells, step_m),
            }
        )
    columns = ["low_a_min", "high_a_min", "low_b_min", "high_b_min", "n_cells", "geometry"]
    return pd.DataFrame(rows, columns=columns)


def base_map(
    centre_lonlat: tuple[float, float], grey_base: bool = True, multiply: bool = True
) -> folium.Map:
    """Create an empty map with the OpenStreetMap base map and the attribution of the data.

    The attribution in the corner of the map credits OpenStreetMap and the MVV timetable, as
    their licences require.

    Args:
        centre_lonlat: Longitude and latitude of the centre of the map.
        grey_base: Whether to show the OpenStreetMap base map in shades of grey, so that its
            colours do not mix with those of the layers drawn on it.
        multiply: Whether to blend the layers with the base map in multiply mode, so that dark
            roads and labels of the base map stay visible under opaque layers.

    Returns:
        A folium Map at zoom 11 with only the base map.
    """
    lon, lat = centre_lonlat
    empty = folium.Map(location=[lat, lon], zoom_start=11, prefer_canvas=True, tiles=None)
    folium.TileLayer(
        "OpenStreetMap",
        attr=(
            "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>"
            " contributors | Timetable: Münchner Verkehrs- und Tarifverbund GmbH (MVV),"
            " <a href='https://creativecommons.org/licenses/by/4.0/'>CC BY 4.0</a>"
        ),
        className="grey-base" if grey_base else "",
    ).add_to(empty)
    css = ".grey-base { filter: grayscale(100%) brightness(1.05); }"
    if multiply:
        css += " .leaflet-overlay-pane { mix-blend-mode: multiply; }"
    empty.get_root().header.add_child(folium.Element(f"<style>{css}</style>"))
    return empty


def isochrone_map(
    bands: pd.DataFrame,
    destination_lonlat: tuple[float, float],
    colour_range: tuple[float, float] = (0.0, 0.85),
    opacity_range: tuple[float, float] = (0.8, 0.3),
    style_overrides: dict[float, tuple[str, float]] | None = None,
    grey_base: bool = True,
    multiply: bool = True,
    caption: str = "Travel time (min)",
) -> folium.Map:
    """Draw the isochrone bands on an interactive map centred on the destination.

    Each band gets one colour of the viridis scale, from dark (shortest times) to light
    (longest times), at equal steps, so that every band is equally distinguishable from its
    neighbours whatever its width in minutes. The opacity decreases at equal steps from the
    first to the last band, so that the large outer bands do not hide the base map. Bands
    listed in style_overrides take the given colour and opacity instead. The attribution in the
    corner of the map credits OpenStreetMap and the MVV timetable, as their licences require.
    A control below the legend switches to "within X minutes" (see threshold_control_html).

    Args:
        bands: DataFrame as returned by band_polygons.
        destination_lonlat: Longitude and latitude of the destination.
        colour_range: Positions on the viridis scale (0 dark purple, 1 yellow) of the first
            and the last band.
        opacity_range: Fill opacity of the first and the last band.
        style_overrides: Colour and fill opacity of some bands, keyed by the upper limit of the
            band in minutes; None means DEFAULT_STYLE_OVERRIDES.
        grey_base: Whether to show the OpenStreetMap base map in shades of grey, so that its
            colours do not mix with those of the bands.
        multiply: Whether to blend the bands with the base map in multiply mode, so that dark
            roads and labels of the base map stay visible under opaque bands.
        caption: Text written above the legend.

    Returns:
        A folium Map with one layer per band, a legend and a marker on the destination.
    """
    lon, lat = destination_lonlat
    isochrones = base_map((lon, lat), grey_base=grey_base, multiply=multiply)

    n_bands = len(bands)
    positions = np.linspace(*colour_range, n_bands)
    colours = [linear.viridis.rgb_hex_str(position) for position in positions]
    opacities = np.linspace(*opacity_range, n_bands).tolist()
    overrides = DEFAULT_STYLE_OVERRIDES if style_overrides is None else style_overrides
    for i, high_min in enumerate(bands["high_min"]):
        if high_min in overrides:
            colours[i], opacities[i] = overrides[high_min]
    layer_names = []
    for colour, opacity, band in zip(colours, opacities, bands.itertuples()):
        label = f"{band.low_min:g}–{band.high_min:g} min"
        layer = folium.GeoJson(
            band.geometry.__geo_interface__,
            name=label,
            style_function=lambda _, colour=colour, opacity=opacity: {
                "fillColor": colour,
                "fillOpacity": opacity,
                "weight": 0,
            },
            tooltip=label,
        ).add_to(isochrones)
        layer_names.append(layer.get_name())

    if n_bands > 0:
        limits = [*bands["low_min"], bands["high_min"].iloc[-1]]
        isochrones.get_root().html.add_child(
            folium.Element(legend_html(limits, colours, opacities, caption=caption))
        )
        control = threshold_control_html(
            layer_names, bands["high_min"].tolist(), colours, opacities
        )
        isochrones.get_root().html.add_child(folium.Element(control))
    folium.Marker([lat, lon], tooltip="Destination").add_to(isochrones)
    return isochrones


def pair_map(
    pairs: pd.DataFrame,
    destinations_lonlat: tuple[tuple[float, float], tuple[float, float]],
    names: tuple[str, str],
    bands_min: tuple[float, ...] = DEFAULT_BANDS_MIN,
    colour: str = "#E66100",
) -> folium.Map:
    """Draw the cells that reach two destinations within two limits chosen with sliders.

    Each pair of bands is one layer in one uniform colour; the sliders of pair_control_html
    make the pairs beyond either limit transparent. The tooltip of each layer gives the two
    bands. The map is centred halfway between the destinations, with a marker on each.

    Args:
        pairs: DataFrame as returned by band_pair_polygons.
        destinations_lonlat: Longitude and latitude of the two destinations.
        names: Label of the two destinations, in the same order.
        bands_min: Increasing limits of the bands in minutes, as given to band_pair_polygons.
        colour: Hexadecimal colour of the cells within both limits.

    Returns:
        A folium Map with one layer per pair of bands, the sliders and two markers.
    """
    centre = tuple(np.mean(destinations_lonlat, axis=0))
    combined = base_map(centre)
    layer_names = []
    for pair in pairs.itertuples():
        label = (
            f"{names[0]}: {pair.low_a_min:g}–{pair.high_a_min:g} min,"
            f" {names[1]}: {pair.low_b_min:g}–{pair.high_b_min:g} min"
        )
        layer = folium.GeoJson(
            pair.geometry.__geo_interface__,
            style_function=lambda _: {"fillColor": colour, "fillOpacity": 0, "weight": 0},
            tooltip=label,
        ).add_to(combined)
        layer_names.append(layer.get_name())

    control = pair_control_html(
        layer_names,
        pairs["high_a_min"].tolist(),
        pairs["high_b_min"].tolist(),
        names,
        [float(limit) for limit in bands_min[1:]],
    )
    combined.get_root().html.add_child(folium.Element(control))
    for (lon, lat), name in zip(destinations_lonlat, names):
        folium.Marker([lat, lon], tooltip=name).add_to(combined)
    return combined


def legend_html(
    limits: list[float],
    colours: list[str],
    opacities: list[float],
    block_px: int = 26,
    caption: str = "Travel time (min)",
) -> str:
    """Return the HTML of a legend with one block of equal width per band.

    Blocks have equal width whatever the width of the band in minutes, and the same colour and
    opacity as the band on the map. The limits are written below the edges of the blocks.

    Args:
        limits: Increasing limits of the bands in minutes (one more than the bands).
        colours: Hexadecimal colour of each band, such as "#263A8A".
        opacities: Fill opacity of each band.
        block_px: Width of each block in pixels.
        caption: Text written above the blocks.

    Returns:
        HTML of a box fixed in the top right corner of the map.
    """
    blocks = []
    for colour, opacity in zip(colours, opacities):
        red, green, blue = (int(colour[k : k + 2], 16) for k in (1, 3, 5))
        background = f"rgba({red}, {green}, {blue}, {opacity})"
        style = f"width: {block_px}px; height: 14px; background: {background};"
        blocks.append(f'<div style="{style}"></div>')
    ticks = [
        f'<span style="position: absolute; left: {k * block_px}px; transform: translateX(-50%);">'
        f"{limit:g}</span>"
        for k, limit in enumerate(limits)
    ]
    return (
        '<div style="position: fixed; top: 10px; right: 10px; z-index: 1000; padding: 8px 14px;'
        " background: white; border-radius: 4px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);"
        ' font: 11px sans-serif; color: #333;">'
        f'<div style="margin-bottom: 4px;">{caption}</div>'
        f'<div style="display: flex;">{"".join(blocks)}</div>'
        f'<div style="position: relative; height: 14px; margin-top: 2px;">{"".join(ticks)}</div>'
        "</div>"
    )


def threshold_control_html(
    layer_names: list[str],
    high_mins: list[float],
    colours: list[str],
    opacities: list[float],
    within_colour: str = "#E66100",
    within_opacity: float = 0.5,
    default_min: float = 30.0,
) -> str:
    """Return the HTML and JavaScript of a control that switches the map to "within X minutes".

    In "bands" mode each band keeps its own colour and opacity. In "within" mode a slider picks
    one of the band limits: the bands up to that limit take one uniform colour and the others
    become transparent. The control only restyles the existing layers in the browser, so the
    page needs no new polygons and no server.

    Args:
        layer_names: JavaScript variable name of each band layer, as given by get_name().
        high_mins: Upper limit of each band in minutes, increasing.
        colours: Hexadecimal colour of each band in "bands" mode.
        opacities: Fill opacity of each band in "bands" mode.
        within_colour: Hexadecimal colour of the bands within the limit in "within" mode.
        within_opacity: Fill opacity of the bands within the limit in "within" mode.
        default_min: Initial limit of the slider: the first band limit at or above it.

    Returns:
        HTML of a box fixed below the legend, followed by the script that drives it.
    """
    bands = [
        {"name": name, "high": high, "colour": colour, "opacity": opacity}
        for name, high, colour, opacity in zip(layer_names, high_mins, colours, opacities)
    ]
    within = {"colour": within_colour, "opacity": within_opacity}
    start = next((k for k, high in enumerate(high_mins) if high >= default_min), len(bands) - 1)
    box = (
        '<div id="isochrone-threshold" style="position: fixed; top: 70px; right: 10px;'
        " z-index: 1000; padding: 8px 14px; background: white; border-radius: 4px;"
        ' box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); font: 11px sans-serif; color: #333;"'
        f' data-bands="{html.escape(json.dumps(bands))}"'
        f' data-within="{html.escape(json.dumps(within))}">'
        '<label><input type="radio" name="isochrone-mode" value="bands" checked> Bands</label> '
        '<label><input type="radio" name="isochrone-mode" value="within"> Within</label> '
        f'<input type="range" min="0" max="{len(bands) - 1}" step="1" value="{start}"'
        ' style="vertical-align: middle;"> <output></output>'
        "</div>"
    )
    script = """<script>
window.addEventListener("load", function () {
  const box = document.getElementById("isochrone-threshold");
  const bands = JSON.parse(box.dataset.bands);
  const within = JSON.parse(box.dataset.within);
  const slider = box.querySelector("input[type=range]");
  const label = box.querySelector("output");
  function update() {
    const mode = box.querySelector("input[name=isochrone-mode]:checked").value;
    const limit = bands[slider.value].high;
    label.textContent = "\u2264 " + limit + " min";
    slider.disabled = mode === "bands";
    for (const band of bands) {
      const style = mode === "bands"
        ? {fillColor: band.colour, fillOpacity: band.opacity}
        : {fillColor: within.colour, fillOpacity: band.high <= limit ? within.opacity : 0};
      window[band.name].setStyle(style);
    }
  }
  box.addEventListener("input", update);
  update();
});
</script>"""
    return box + script


def pair_control_html(
    layer_names: list[str],
    highs_a: list[float],
    highs_b: list[float],
    names: tuple[str, str],
    limits: list[float],
    opacity: float = 0.5,
    default_min: tuple[float, float] = (45.0, 45.0),
) -> str:
    """Return the HTML and JavaScript of two sliders that show the cells within both limits.

    Each slider picks one of the band limits for one destination. A pair of bands is shown when
    both its upper limits are within the two sliders, and is transparent otherwise. The control
    only changes the opacity of the existing layers in the browser, so the page needs no new
    polygons and no server.

    Args:
        layer_names: JavaScript variable name of each pair layer, as given by get_name().
        highs_a: Upper limit of the band to the first destination of each pair, in minutes.
        highs_b: Upper limit of the band to the second destination of each pair, in minutes.
        names: Label of the two destinations, written next to their sliders.
        limits: Increasing values that the sliders can take, in minutes.
        opacity: Fill opacity of the pairs within both limits.
        default_min: Initial limit of each slider: the first value of limits at or above it.

    Returns:
        HTML of a box fixed in the top right corner of the map, followed by the script that
        drives it.
    """
    pairs = [
        {"name": name, "a": high_a, "b": high_b}
        for name, high_a, high_b in zip(layer_names, highs_a, highs_b)
    ]
    sliders = []
    for name, default in zip(names, default_min):
        start = next((k for k, limit in enumerate(limits) if limit >= default), len(limits) - 1)
        sliders.append(
            f"<span>{html.escape(name)}</span>"
            f'<input type="range" min="0" max="{len(limits) - 1}" step="1" value="{start}">'
            "<output></output>"
        )
    box = (
        '<div id="isochrone-pair" style="position: fixed; top: 10px; right: 10px;'
        " z-index: 1000; padding: 8px 14px; background: white; border-radius: 4px;"
        ' box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); font: 11px sans-serif; color: #333;"'
        f' data-pairs="{html.escape(json.dumps(pairs))}"'
        f' data-limits="{html.escape(json.dumps(limits))}"'
        f' data-opacity="{opacity}">'
        '<div style="margin-bottom: 4px;">Travel time to both destinations</div>'
        '<div style="display: grid; grid-template-columns: auto auto auto; gap: 2px 6px;'
        f' align-items: center; white-space: nowrap;">{"".join(sliders)}</div>'
        "</div>"
    )
    script = """<script>
window.addEventListener("load", function () {
  const box = document.getElementById("isochrone-pair");
  const pairs = JSON.parse(box.dataset.pairs);
  const limits = JSON.parse(box.dataset.limits);
  const opacity = Number(box.dataset.opacity);
  const sliders = box.querySelectorAll("input[type=range]");
  const labels = box.querySelectorAll("output");
  function update() {
    const [limitA, limitB] = Array.from(sliders, (slider) => limits[slider.value]);
    labels[0].textContent = "\u2264 " + limitA + " min";
    labels[1].textContent = "\u2264 " + limitB + " min";
    for (const pair of pairs) {
      const shown = pair.a <= limitA && pair.b <= limitB;
      window[pair.name].setStyle({fillOpacity: shown ? opacity : 0});
    }
  }
  box.addEventListener("input", update);
  update();
});
</script>"""
    return box + script
