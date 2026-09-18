from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import math

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "route_reduced.csv"


def _detect_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two WGS84 points."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_optimal_speed(grade_percent: float) -> float:
    """Simple heuristic for optimal speed based on road grade.

    Base optimal speed is 50 km/h.
    - Uphill (grade > 0): reduce speed slightly.
    - Downhill (grade < 0): reduce speed for large negative grades for safety.
    """
    base = 50.0
    g = grade_percent

    if g > 0:
        base -= 0.3 * g
    elif g < 0:
        base -= 0.1 * abs(g)

    # Clamp to reasonable bounds
    if base < 30.0:
        base = 30.0
    if base > 90.0:
        base = 90.0
    return base


def load_route_points():
    if not CSV_PATH.exists():
        raise RuntimeError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    cols = df.columns

    lat_col = _detect_column(cols, ["lat", "latitude", "Lat", "LAT"])
    lon_col = _detect_column(cols, ["lon", "lng", "longitude", "Lon", "LON"])
    elev_col = _detect_column(cols, ["elev", "elevation", "alt", "height", "altitude", "ALT"])

    if lat_col is None or lon_col is None:
        raise RuntimeError(
            "Could not detect latitude/longitude columns in route_reduced.csv. "
            "Expected one of: lat/latitude and lon/lng/longitude."
        )

    lats = df[lat_col].astype(float).to_list()
    lons = df[lon_col].astype(float).to_list()
    elevs = df[elev_col].astype(float).to_list() if elev_col is not None else [0.0] * len(lats)

    # cumulative distance
    distances = [0.0]
    for i in range(1, len(lats)):
        d = _haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
        distances.append(distances[-1] + d)

    # grade (%) between points, using central difference approx
    grades = [0.0]
    for i in range(1, len(elevs)):
        ds = distances[i] - distances[i - 1]
        if ds <= 0:
            grades.append(0.0)
            continue
        dh = elevs[i] - elevs[i - 1]
        grade = 100.0 * dh / ds
        grades.append(grade)

    points = []
    for i in range(len(lats)):
        grade = grades[i]
        speed_opt = compute_optimal_speed(grade)
        points.append(
            {
                "index": i,
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "elevation": float(elevs[i]) if elev_col is not None else None,
                "distance": float(distances[i]),
                "grade": float(grade),
                "speed_optimal": float(speed_opt),
            }
        )

    return points


app = FastAPI(title="Vehicle Route Fuel Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load route once at startup
ROUTE_POINTS = load_route_points()

# Serve static frontend assets (HTML/JS/CSS)
app.mount(
    "/static", StaticFiles(directory=BASE_DIR / "static"), name="static"
)


@app.get("/")
def index():
    """Serve main HTML page."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/route")
def get_route():
    """Return preprocessed route with grade and optimal speed per point."""
    return {"points": ROUTE_POINTS}
