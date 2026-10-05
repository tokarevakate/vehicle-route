from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .consumption_model import build_speed_profile, ProfilePoint

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "route_restored.csv"

app = FastAPI(title="Vehicle Route Fuel Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv(CSV_PATH)
PROFILE_POINTS = [
    ProfilePoint(
        index=int(r["index"]),
        s=float(r["s"]),
        lat=float(r["lat"]),
        lon=float(r["lon"]),
        elevation=float(r["elevation"]),
        grade=float(r["grade"]),
        radius=None if pd.isna(r["radius"]) else float(r["radius"]),
    )
    for r in df.to_dict("records")
]

ROUTE_POINTS = build_speed_profile(PROFILE_POINTS)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/route")
def get_route():
    return {"points": ROUTE_POINTS}
