"""Turn travel times on the grid into isochrone bands and draw them on a map."""

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
    minutes = np.asarray(minutes, dtype="float64")
    limits = np.asarray(bands_min, dtype="float64")
    band = np.digitize(minutes, limits[1:-1], right=True)
    inside = (minutes >= limits[0]) & (minutes <= limits[-1])

    to_lonlat = Transformer.from_crs(PROJECTED_CRS, "EPSG:4326", always_xy=True)
    half = step_m / 2
    rows = []
    for i in range(len(limits) - 1):
        cells = points_xy[inside & (band == i)]
        if len(cells) == 0:
            continue
        x, y = cells[:, 0], cells[:, 1]
        merged = shapely.coverage_union_all(shapely.box(x - half, y - half, x + half, y + half))
        merged = shapely.simplify(merged, 0)
        merged = shapely.transform(
            merged, lambda xy: np.column_stack(to_lonlat.transform(xy[:, 0], xy[:, 1]))
        )
        rows.append(
            {
                "low_min": limits[i],
                "high_min": limits[i + 1],
                "n_cells": len(cells),
                "geometry": shapely.set_precision(merged, 1e-6),
            }
        )
    return pd.DataFrame(rows, columns=["low_min", "high_min", "n_cells", "geometry"])


def isochrone_map(
    bands: pd.DataFrame,
    destination_lonlat: tuple[float, float],
    colour_range: tuple[float, float] = (0.0, 0.85),
    opacity_range: tuple[float, float] = (0.8, 0.3),
    style_overrides: dict[float, tuple[str, float]] | None = None,
    grey_base: bool = True,
    multiply: bool = True,
) -> folium.Map:
    """Draw the isochrone bands on an interactive map centred on the destination.

    Each band gets one colour of the viridis scale, from dark (shortest times) to light
    (longest times), at equal steps, so that every band is equally distinguishable from its
    neighbours whatever its width in minutes. The opacity decreases at equal steps from the
    first to the last band, so that the large outer bands do not hide the base map. Bands
    listed in style_overrides take the given colour and opacity instead.

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

    Returns:
        A folium Map with one layer per band, a legend and a marker on the destination.
    """
    lon, lat = destination_lonlat
    isochrones = folium.Map(location=[lat, lon], zoom_start=11, prefer_canvas=True, tiles=None)
    folium.TileLayer("OpenStreetMap", className="grey-base" if grey_base else "").add_to(
        isochrones
    )
    css = ".grey-base { filter: grayscale(100%) brightness(1.05); }"
    if multiply:
        css += " .leaflet-overlay-pane { mix-blend-mode: multiply; }"
    isochrones.get_root().header.add_child(folium.Element(f"<style>{css}</style>"))

    n_bands = len(bands)
    positions = np.linspace(*colour_range, n_bands)
    colours = [linear.viridis.rgb_hex_str(position) for position in positions]
    opacities = np.linspace(*opacity_range, n_bands).tolist()
    overrides = DEFAULT_STYLE_OVERRIDES if style_overrides is None else style_overrides
    for i, high_min in enumerate(bands["high_min"]):
        if high_min in overrides:
            colours[i], opacities[i] = overrides[high_min]
    for colour, opacity, band in zip(colours, opacities, bands.itertuples()):
        label = f"{band.low_min:g}–{band.high_min:g} min"
        folium.GeoJson(
            band.geometry.__geo_interface__,
            name=label,
            style_function=lambda _, colour=colour, opacity=opacity: {
                "fillColor": colour,
                "fillOpacity": opacity,
                "weight": 0,
            },
            tooltip=label,
        ).add_to(isochrones)

    if n_bands > 0:
        limits = [*bands["low_min"], bands["high_min"].iloc[-1]]
        isochrones.get_root().html.add_child(
            folium.Element(legend_html(limits, colours, opacities))
        )
    folium.Marker([lat, lon], tooltip="Destination").add_to(isochrones)
    return isochrones


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
