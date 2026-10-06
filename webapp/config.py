from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .optimization import OptimizationConfig
from .vehicle_model import VehicleParams


@dataclass(frozen=True)
class RouteConfig:
    csv_path: str = "route_restored.csv"
    target_spacing_m: float = 50.0
    elevation_smoothing_m2_per_point: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    default_algorithm: str = "dp"
    route: RouteConfig = field(default_factory=RouteConfig)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def _construct(cls, values: dict):
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = os.getenv("VEHICLE_ROUTE_CONFIG", "configs/default.yaml")
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return AppConfig(
        default_algorithm=raw.get("default_algorithm", "dp"),
        route=_construct(RouteConfig, raw.get("route", {})),
        vehicle=_construct(VehicleParams, raw.get("vehicle", {})),
        optimization=_construct(OptimizationConfig, raw.get("optimization", {})),
    )
