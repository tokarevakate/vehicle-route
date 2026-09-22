import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


@dataclass
class RawPoint:
    index: int
    lat: float
    lon: float
    elevation: float
    distance: float


@dataclass
class ProfilePoint:
    index: int
    s: float
    lat: float
    lon: float
    elevation: float
    grade: float
    radius: Optional[float]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two WGS84 points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _detect_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    columns_list = list(columns)
    columns_lower = [c.lower() for c in columns_list]
    for cand in candidates:
        if cand.lower() in columns_lower:
            return columns_list[columns_lower.index(cand.lower())]
    return None


def load_raw_points_from_csv(csv_path) -> List[RawPoint]:
    df = pd.read_csv(csv_path)
    cols = df.columns

    lat_col  = _detect_column(cols, ["lat", "latitude", "Lat", "LAT"])
    lon_col  = _detect_column(cols, ["lon", "lng", "longitude", "Lon", "LON"])
    elev_col = _detect_column(cols, ["elev", "elevation", "alt", "height", "altitude", "ALT"])

    if lat_col is None or lon_col is None:
        raise RuntimeError(
            "Could not detect latitude/longitude columns in route.csv. "
            "Expected one of: lat/latitude and lon/lng/longitude."
        )

    lats  = df[lat_col].astype(float).to_list()
    lons  = df[lon_col].astype(float).to_list()
    elevs = df[elev_col].astype(float).to_list() if elev_col is not None else [0.0] * len(lats)

    distances = [0.0]
    for i in range(1, len(lats)):
        d = _haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
        distances.append(distances[-1] + d)

    raw_points: List[RawPoint] = []
    for i, (lat, lon, elev, s) in enumerate(zip(lats, lons, elevs, distances)):
        raw_points.append(
            RawPoint(
                index=i,
                lat=float(lat),
                lon=float(lon),
                elevation=float(elev),
                distance=float(s),
            )
        )
    return raw_points


def fit_elevation_model(raw_points: List[RawPoint]):
    """Build a smooth elevation model h(s) using a cubic spline.

    P5: replaces np.polyfit deg-5 to eliminate Runge oscillations on long
    routes. Returns callables elev_model(s) and grade_model(s) identical in
    signature to the old polynomial version so callers need no changes.
    """
    s = np.array([p.distance for p in raw_points])
    h = np.array([p.elevation for p in raw_points])

    # CubicSpline requires strictly increasing x — deduplicate by distance
    s_unique, idx = np.unique(s, return_index=True)
    h_unique = h[idx]

    s_mean = s_unique.mean()
    cs = CubicSpline(s_unique - s_mean, h_unique)

    def elev_model(s_query: np.ndarray) -> np.ndarray:
        return cs(np.asarray(s_query) - s_mean)

    def grade_model(s_query: np.ndarray) -> np.ndarray:
        # first derivative dh/ds [m/m] -> percent
        return 100.0 * cs(np.asarray(s_query) - s_mean, 1)

    return elev_model, grade_model


def resample_route(
    raw_points: List[RawPoint], elev_model, grade_model, target_points: int
) -> List[ProfilePoint]:
    """Resample route to a fixed number of points with smooth elevation and grade."""
    s    = np.array([p.distance for p in raw_points])
    lats = np.array([p.lat      for p in raw_points])
    lons = np.array([p.lon      for p in raw_points])

    s_new     = np.linspace(s.min(), s.max(), target_points)
    elev_new  = elev_model(s_new)
    grade_new = grade_model(s_new)

    lat_new = np.interp(s_new, s, lats)
    lon_new = np.interp(s_new, s, lons)

    radii = compute_curvature(s_new, lat_new, lon_new)

    profile: List[ProfilePoint] = []
    for i in range(len(s_new)):
        radius = radii[i]
        profile.append(
            ProfilePoint(
                index=i,
                s=float(s_new[i]),
                lat=float(lat_new[i]),
                lon=float(lon_new[i]),
                elevation=float(elev_new[i]),
                grade=float(grade_new[i]),
                radius=float(radius) if radius is not None else None,
            )
        )
    return profile


def _heading(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle bearing between two WGS84 points (radians)."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    return math.atan2(y, x)


def compute_curvature(
    s: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> List[Optional[float]]:
    """Compute approximate radius of curvature at each point (meters)."""
    n = len(lats)
    radii: List[Optional[float]] = [None] * n
    if n < 3:
        return radii

    for i in range(1, n - 1):
        heading_prev = _heading(lats[i - 1], lons[i - 1], lats[i],     lons[i])
        heading_next = _heading(lats[i],     lons[i],     lats[i + 1], lons[i + 1])
        dtheta = heading_next - heading_prev

        while dtheta >  math.pi:
            dtheta -= 2.0 * math.pi
        while dtheta < -math.pi:
            dtheta += 2.0 * math.pi

        ds = (s[i + 1] - s[i - 1]) / 2.0
        if ds <= 0 or abs(dtheta) < 1e-6:
            radii[i] = None
        else:
            kappa    = dtheta / ds
            radii[i] = abs(1.0 / kappa)

    return radii


def build_profile(csv_path, target_points: int = 10000) -> List[ProfilePoint]:
    """High-level helper: CSV -> dense ProfilePoint list.

    AI model #1: reconstructs smooth elevation, grade and curvature
    from raw GPS data using cubic spline interpolation.
    """
    raw_points    = load_raw_points_from_csv(csv_path)
    elev_model, grade_model = fit_elevation_model(raw_points)
    profile_points = resample_route(raw_points, elev_model, grade_model, target_points)
    return profile_points
