"""Walking times and walking legs between points and platforms."""

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

PROJECTED_CRS = "EPSG:25832"


def to_projected(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS 84 longitudes and latitudes to coordinates in metres in PROJECTED_CRS.

    PROJECTED_CRS is UTM zone 32N, so that distances can be computed with Pythagoras in
    metres.

    Args:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees.

    Returns:
        Arrays x and y in metres, with the same shape as lon and lat.
    """
    to_metres = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = to_metres.transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))
    return np.asarray(x), np.asarray(y)


def walking_time_s(
    distance_m: np.ndarray, walking_speed_mps: float = 1.0, detour_factor: float = 1.3
) -> np.ndarray:
    """Return walking times in whole seconds for straight-line distances.

    The walking distance is estimated as the straight-line distance times detour_factor. Times
    are rounded up, so that rounding never makes a walk shorter.

    Args:
        distance_m: Straight-line distances in metres.
        walking_speed_mps: Walking speed in metres per second.
        detour_factor: Ratio between the walking distance and the straight-line distance.

    Returns:
        Walking times in seconds, as an integer array.
    """
    return np.ceil(np.asarray(distance_m) * detour_factor / walking_speed_mps).astype("int64")


def find_nearby_platforms(
    points_xy: np.ndarray,
    platforms: pd.DataFrame,
    radius_m: float = 1000.0,
    walking_speed_mps: float = 1.0,
    detour_factor: float = 1.3,
) -> pd.DataFrame:
    """Return, for each point, the platforms within radius_m and the walking time to each.

    The search uses the straight-line distance; detour_factor only enters the walking time.
    Walking is symmetric, so the same pairs serve walks from the points to the platforms
    (access) and from the platforms to the points (egress).

    Args:
        points_xy: Coordinates of the points in metres in PROJECTED_CRS, with shape (n, 2);
            a single point can be given with shape (2,).
        platforms: DataFrame with columns stop_id, x and y, as returned by load_platforms.
        radius_m: Maximum straight-line distance in metres between a point and a platform.
        walking_speed_mps: Walking speed in metres per second.
        detour_factor: Ratio between the walking distance and the straight-line distance.

    Returns:
        DataFrame with columns point (row of points_xy), stop_id, distance_m (straight-line,
        metres) and walk_s (integer seconds), sorted by point and distance. Points with no
        platform within radius_m do not appear.
    """
    points_tree = cKDTree(np.atleast_2d(points_xy))
    platforms_tree = cKDTree(platforms[["x", "y"]].to_numpy())
    near = points_tree.sparse_distance_matrix(
        platforms_tree, max_distance=radius_m, output_type="ndarray"
    )

    pairs = pd.DataFrame(
        {
            "point": near["i"],
            "stop_id": platforms["stop_id"].to_numpy()[near["j"]],
            "distance_m": near["v"],
        }
    )
    pairs["walk_s"] = walking_time_s(pairs["distance_m"], walking_speed_mps, detour_factor)
    return pairs.sort_values(["point", "distance_m"], ignore_index=True)


def latest_departures_from_points(
    points_xy: np.ndarray,
    access: pd.DataFrame,
    departures: dict[str, int],
    destination_xy: np.ndarray,
    arrival_s: int,
    max_duration_s: int = 7200,
    walking_speed_mps: float = 1.0,
    detour_factor: float = 1.3,
) -> np.ndarray:
    """Return, for each point, the latest departure that reaches the destination in time.

    A point can walk to a nearby platform and continue by public transport, or walk directly
    to the destination; the latest of these departures is kept. Journeys longer than
    max_duration_s count as not reaching the destination.

    Args:
        points_xy: Coordinates of the points in metres in PROJECTED_CRS, with shape (n, 2).
        access: Walking legs from the points to nearby platforms, as returned by
            find_nearby_platforms for points_xy.
        departures: Latest departure from each platform, as returned by latest_departures.
        destination_xy: Coordinates of the destination in metres, with shape (2,).
        arrival_s: Latest arrival time at the destination, seconds after midnight.
        max_duration_s: Maximum duration of a journey in seconds.
        walking_speed_mps: Walking speed in metres per second.
        detour_factor: Ratio between the walking distance and the straight-line distance.

    Returns:
        Latest departure time from each point, seconds after midnight, as a float array of
        length n; NaN for points that cannot reach the destination within max_duration_s.
    """
    points_xy = np.atleast_2d(points_xy)

    via_platform = access["stop_id"].map(departures) - access["walk_s"]
    best_via_platform = via_platform.groupby(access["point"]).max()
    latest = np.full(len(points_xy), np.nan)
    latest[best_via_platform.index.to_numpy()] = best_via_platform.to_numpy()

    direct_m = np.hypot(*(points_xy - np.asarray(destination_xy)).T)
    direct = arrival_s - walking_time_s(direct_m, walking_speed_mps, detour_factor)
    latest = np.fmax(latest, direct)

    latest[latest < arrival_s - max_duration_s] = np.nan
    return latest


def build_grid(
    platforms: pd.DataFrame, step_m: float = 100.0, radius_m: float = 1000.0, n_strips: int = 20
) -> np.ndarray:
    """Return the centres of a square grid covering the area within radius_m of a platform.

    The grid is aligned to multiples of step_m in PROJECTED_CRS, so that it does not depend on
    the platforms or the radius. Only points within radius_m of at least one platform are
    kept: farther points can reach a destination only on foot.

    Args:
        platforms: DataFrame with columns x and y, as returned by load_platforms.
        step_m: Distance in metres between neighbouring grid points.
        radius_m: Maximum straight-line distance in metres between a grid point and its
            nearest platform.
        n_strips: Number of horizontal strips in which the grid is built, to limit memory use.

    Returns:
        Array of shape (n, 2) with the x and y coordinates of the grid points, in metres.
    """
    xy = platforms[["x", "y"]].to_numpy()
    x_min, y_min = np.floor((xy.min(axis=0) - radius_m) / step_m) * step_m
    x_max, y_max = np.ceil((xy.max(axis=0) + radius_m) / step_m) * step_m
    xs = np.arange(x_min, x_max + step_m / 2, step_m)
    ys = np.arange(y_min, y_max + step_m / 2, step_m)

    tree = cKDTree(xy)
    bound = np.nextafter(radius_m, np.inf)
    strips = []
    for strip_ys in np.array_split(ys, n_strips):
        grid_x, grid_y = np.meshgrid(xs, strip_ys)
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        distance, _ = tree.query(points, distance_upper_bound=bound)
        strips.append(points[np.isfinite(distance)])
    return np.vstack(strips)
