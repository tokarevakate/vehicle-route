from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .profile_model import build_profile
from .consumption_model import build_speed_profile

BASE_DIR = Path(__file__).resolve().parent
# route_sochi.csv is the single source of truth for the route .
CSV_PATH = BASE_DIR.parent / "route_sochi.csv"


app = FastAPI(title="Vehicle Route Fuel Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build route profile and optimal speed profile once at startup.
# profile_model  строит сплайн высоты, уклон и кривизну, 
# consumption_model  содержит калиброванную по физике модель расхода и ограниченную оптимизацию скорости.
PROFILE_POINTS = build_profile(CSV_PATH, target_points=10000)
ROUTE_POINTS = build_speed_profile(PROFILE_POINTS)

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
    """Return preprocessed route with grade, curvature and optimal speed per point."""
    return {"points": ROUTE_POINTS}
