import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


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
    # FIX 3: case-insensitive fallback for non-standard CSV headers
    columns_list = list(columns)
    columns_lower = [c.lower() for c in columns_list]
    for cand in candidates:
        if cand.lower() in columns_lower:
            return columns_list[columns_lower.index(cand.lower())]
    return None


def load_raw_points_from_csv(csv_path) -> List[RawPoint]:
    df = pd.read_csv(csv_path)
    cols = df.columns

    lat_col = _detect_column(cols, ["lat", "latitude", "Lat", "LAT"])
    lon_col = _detect_column(cols, ["lon", "lng", "longitude", "Lon", "LON"])
    elev_col = _detect_column(cols, ["elev", "elevation", "alt", "height", "altitude", "ALT"])

    if lat_col is None or lon_col is None:
        raise RuntimeError(
            "Could not detect latitude/longitude columns in route.csv. "
            "Expected one of: lat/latitude and lon/lng/longitude."
        )

    lats = df[lat_col].astype(float).to_list()
    lons = df[lon_col].astype(float).to_list()
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


def fit_elevation_model(raw_points: List[RawPoint], degree: int = 5):
    """Fit polynomial elevation model h(s) and return callable for h(s) and grade(s).

    The model is used as an "AI"-style smoother/interpolator for elevation
    and grade along the route.
    """
    s = np.array([p.distance for p in raw_points])
    h = np.array([p.elevation for p in raw_points])

    # FIX 4: clamp degree to avoid rank-deficient polyfit on short routes
    degree = min(degree, max(len(raw_points) - 1, 1))

    # center distances for numerical stability
    s_mean = s.mean()
    s0 = s - s_mean

    coeffs = np.polyfit(s0, h, deg=degree)
    poly = np.poly1d(coeffs)
    dpoly = np.polyder(poly)

    def elev_model(s_query: np.ndarray) -> np.ndarray:
        return poly(s_query - s_mean)

    def grade_model(s_query: np.ndarray) -> np.ndarray:
        # dh/ds in m/m -> convert to percent
        return 100.0 * dpoly(s_query - s_mean)

    return elev_model, grade_model


def resample_route(
    raw_points: List[RawPoint], elev_model, grade_model, target_points: int
) -> List[ProfilePoint]:
    """Resample route to a fixed number of points with smooth elevation and grade.

    - Builds a new uniform distance grid s_new.
    - Uses elev_model and grade_model to infer elevation and grade.
    - Interpolates lat/lon along the original distance parameter.
    """
    s = np.array([p.distance for p in raw_points])
    lats = np.array([p.lat for p in raw_points])
    lons = np.array([p.lon for p in raw_points])

    s_new = np.linspace(s.min(), s.max(), target_points)
    elev_new = elev_model(s_new)
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
    # FIX 1: replaced flat-earth approximation with correct spherical formula
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
    """Compute approximate radius of curvature at each point.

    Returns list of radii R (m) or None at endpoints / straight sections.
    """
    n = len(lats)
    radii: List[Optional[float]] = [None] * n
    if n < 3:
        return radii

    for i in range(1, n - 1):
        heading_prev = _heading(lats[i - 1], lons[i - 1], lats[i], lons[i])
        heading_next = _heading(lats[i], lons[i], lats[i + 1], lons[i + 1])
        dtheta = heading_next - heading_prev

        # normalize angle to [-pi, pi]
        while dtheta > math.pi:
            dtheta -= 2.0 * math.pi
        while dtheta < -math.pi:
            dtheta += 2.0 * math.pi

        # FIX 2: each heading spans one arc; divide total span by 2
        ds = (s[i + 1] - s[i - 1]) / 2.0
        if ds <= 0 or abs(dtheta) < 1e-6:
            radii[i] = None
        else:
            kappa = dtheta / ds  # curvature 1/m
            radii[i] = abs(1.0 / kappa)

    return radii


def build_profile(csv_path, target_points: int = 10000) -> List[ProfilePoint]:
    """High-level helper: from CSV -> profile points.

    This is the "profile AI model": it reconstructs a smooth, dense
    representation of the *same* route with elevation, grade and curvature.
    """
    raw_points = load_raw_points_from_csv(csv_path)
    elev_model, grade_model = fit_elevation_model(raw_points)
    profile_points = resample_route(raw_points, elev_model, grade_model, target_points)
    return profile_points
