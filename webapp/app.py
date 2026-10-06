from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ALGORITHMS, load_config
from .optimization import optimize_dynamic_programming, optimize_pointwise, result_to_points
from .profile_model import build_profile

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

_locks = {name: threading.Lock() for name in ALGORITHMS}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    csv_path = Path(config.route.csv_path)
    if not csv_path.is_absolute():
        csv_path = PROJECT_DIR / csv_path
    app.state.config = config
    app.state.profile = build_profile(
        csv_path,
        target_spacing_m=config.route.target_spacing_m,
        smoothing_m2_per_point=config.route.elevation_smoothing_m2_per_point,
    )
    app.state.results = {}
    yield
    app.state.results.clear()


app = FastAPI(title="Vehicle Route Fuel Optimization API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


def _calculate(algorithm: str):
    profile = app.state.profile
    config = app.state.config
    if algorithm == "pointwise":
        return optimize_pointwise(profile, config.vehicle, config.optimization)
    if algorithm == "dp":
        return optimize_dynamic_programming(profile, config.vehicle, config.optimization)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _get_result(algorithm: str):
    cached = app.state.results.get(algorithm)
    if cached is not None:
        return cached
    with _locks[algorithm]:
        cached = app.state.results.get(algorithm)
        if cached is None:
            cached = _calculate(algorithm)
            app.state.results[algorithm] = cached
    return cached


@app.get("/api/route")
def get_route(
    algorithm: Literal["pointwise", "dp"] | None = Query(default=None),
):
    selected = algorithm or app.state.config.default_algorithm
    result = _get_result(selected)
    if result.status != "optimal":
        raise HTTPException(
            status_code=422,
            detail={"algorithm": selected, "status": result.status},
        )
    return {
        "algorithm": selected,
        "metrics": result.summary(),
        "points": result_to_points(app.state.profile, result),
    }


@app.get("/api/metrics")
def get_metrics(compute: bool = Query(default=False)):
    """Metrics of already computed algorithms; ``compute=true`` calculates all of them."""
    if compute:
        for name in ALGORITHMS:
            _get_result(name)
    return {name: result.summary() for name, result in list(app.state.results.items())}
