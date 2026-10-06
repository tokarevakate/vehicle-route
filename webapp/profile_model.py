from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.signal import savgol_filter


@dataclass
class RawPoint:
    index: int
    lat: float
    lon: float
    elevation: float
    distance: float
    confidence: float = 1.0


@dataclass
class ProfilePoint:
    index: int
    s: float
    lat: float
    lon: float
    elevation: float
    grade: float
    radius: Optional[float]
    confidence: float = 1.0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_earth = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return radius_earth * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _detect_column(columns, candidates):
    mapping = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def _source_confidence(value: object) -> float:
    text = str(value).strip().lower()
    if any(token in text for token in ("original", "observed", "measured", "raw", "исход")):
        return 1.0
    if any(token in text for token in ("interpol", "restor", "predict", "восстанов")):
        return 0.45
    return 0.75


def load_raw_points_from_csv(csv_path) -> List[RawPoint]:
    df = pd.read_csv(csv_path)
    lat_col = _detect_column(df.columns, ["lat", "latitude"])
    lon_col = _detect_column(df.columns, ["lon", "lng", "longitude"])
    elev_col = _detect_column(df.columns, ["elev", "elevation", "alt", "height", "altitude"])
    source_col = _detect_column(df.columns, ["source", "quality", "status"])
    if lat_col is None or lon_col is None or elev_col is None:
        raise RuntimeError("CSV must contain latitude, longitude and elevation columns")

    numeric = df[[lat_col, lon_col, elev_col]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy()).all(axis=1)
    numeric = numeric.loc[valid]
    if len(numeric) < 2:
        raise ValueError("Route must contain at least two valid points")
    confidences = (
        df.loc[valid, source_col].map(_source_confidence).to_numpy(dtype=float)
        if source_col is not None else np.ones(len(numeric))
    )
    lats = numeric[lat_col].to_numpy(dtype=float)
    lons = numeric[lon_col].to_numpy(dtype=float)
    elevations = numeric[elev_col].to_numpy(dtype=float)
    distances = np.zeros(len(numeric))
    for i in range(1, len(numeric)):
        distances[i] = distances[i - 1] + _haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
    keep = np.r_[True, np.diff(distances) > 1e-3]
    return [
        RawPoint(i, float(lat), float(lon), float(elev), float(distance), float(confidence))
        for i, (lat, lon, elev, distance, confidence) in enumerate(
            zip(lats[keep], lons[keep], elevations[keep], distances[keep], confidences[keep])
        )
    ]


def fit_elevation_model(raw_points: List[RawPoint], smoothing_m2_per_point: float = 1.0):
    s = np.array([point.distance for point in raw_points], dtype=float)
    h = np.array([point.elevation for point in raw_points], dtype=float)
    weights = np.array([max(point.confidence, 0.05) for point in raw_points], dtype=float)
    s_mean = float(np.mean(s))
    x = s - s_mean
    if len(s) >= 4:
        spline = UnivariateSpline(x, h, w=weights, s=max(smoothing_m2_per_point, 0.0) * len(s), ext=3)
        elevation = lambda query: spline(np.asarray(query) - s_mean)
        grade = lambda query: 100.0 * spline(np.asarray(query) - s_mean, 1)
    else:
        spline = CubicSpline(x, h, bc_type="natural")
        elevation = lambda query: spline(np.asarray(query) - s_mean)
        grade = lambda query: 100.0 * spline(np.asarray(query) - s_mean, 1)
    return elevation, grade


def compute_curvature(s: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> List[Optional[float]]:
    n = len(s)
    if n < 3:
        return [None] * n
    lat0 = math.radians(float(np.mean(lats)))
    x = 6_371_000.0 * np.cos(lat0) * np.radians(lons - lons[0])
    y = 6_371_000.0 * np.radians(lats - lats[0])
    if n >= 7:
        window = min(21, n if n % 2 == 1 else n - 1)
        if window >= 5:
            x = savgol_filter(x, window, min(3, window - 2), mode="interp")
            y = savgol_filter(y, window, min(3, window - 2), mode="interp")
    dx, dy = np.gradient(x, s), np.gradient(y, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    radii: List[Optional[float]] = []
    for value in curvature:
        radii.append(None if not np.isfinite(value) or value < 1e-6 else float(1.0 / value))
    return radii


def resample_route(
    raw_points: List[RawPoint],
    elevation_model,
    grade_model,
    target_spacing_m: float = 25.0,
) -> List[ProfilePoint]:
    s = np.array([point.distance for point in raw_points], dtype=float)
    lats = np.array([point.lat for point in raw_points], dtype=float)
    lons = np.array([point.lon for point in raw_points], dtype=float)
    confidence = np.array([point.confidence for point in raw_points], dtype=float)
    count = max(2, int(math.ceil((s[-1] - s[0]) / max(target_spacing_m, 1.0))) + 1)
    s_new = np.linspace(s[0], s[-1], count)
    lat_new = np.interp(s_new, s, lats)
    lon_new = np.interp(s_new, s, lons)
    elevation_new = np.asarray(elevation_model(s_new), dtype=float)
    grade_new = np.asarray(grade_model(s_new), dtype=float)
    confidence_new = np.interp(s_new, s, confidence)
    radii = compute_curvature(s_new, lat_new, lon_new)
    return [
        ProfilePoint(
            index=i,
            s=float(s_new[i]),
            lat=float(lat_new[i]),
            lon=float(lon_new[i]),
            elevation=float(elevation_new[i]),
            grade=float(grade_new[i]),
            radius=radii[i],
            confidence=float(confidence_new[i]),
        )
        for i in range(count)
    ]


def build_profile(
    csv_path,
    target_points: Optional[int] = None,
    target_spacing_m: float = 25.0,
    smoothing_m2_per_point: float = 1.0,
) -> List[ProfilePoint]:
    raw_points = load_raw_points_from_csv(csv_path)
    if target_points is not None:
        route_length = raw_points[-1].distance - raw_points[0].distance
        target_spacing_m = route_length / max(target_points - 1, 1)
    elevation, grade = fit_elevation_model(raw_points, smoothing_m2_per_point)
    return resample_route(raw_points, elevation, grade, target_spacing_m)
