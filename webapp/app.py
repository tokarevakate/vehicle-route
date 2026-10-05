from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .consumption_model import build_speed_profile
from .profile_model import build_profile

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "route_restored.csv"

app = FastAPI(title="Vehicle Route Fuel Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# route_restored.csv (timestamp, lat, lon, elevation, source) -> s, grade, radius
PROFILE_POINTS = build_profile(CSV_PATH, target_points=10000)
ROUTE_POINTS = build_speed_profile(PROFILE_POINTS)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/route")
def get_route():
    return {"points": ROUTE_POINTS}
