from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .optimization import OptimizationConfig
from .vehicle_model import VehicleParams

logger = logging.getLogger(__name__)

ALGORITHMS = ("pointwise", "dp")
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"


@dataclass(frozen=True)
class RouteConfig:
    csv_path: str = "route_restored.csv"
    target_spacing_m: float = 50.0
    elevation_smoothing_m2_per_point: float = 1.0

    def __post_init__(self) -> None:
        if self.target_spacing_m <= 0.0:
            raise ValueError("route.target_spacing_m must be positive")
        if self.elevation_smoothing_m2_per_point < 0.0:
            raise ValueError("route.elevation_smoothing_m2_per_point must be non-negative")


@dataclass(frozen=True)
class AppConfig:
    default_algorithm: str = "dp"
    route: RouteConfig = field(default_factory=RouteConfig)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self) -> None:
        if self.default_algorithm not in ALGORITHMS:
            raise ValueError(
                f"default_algorithm must be one of {ALGORITHMS}, got {self.default_algorithm!r}"
            )


def _construct(cls, values, section: str):
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        logger.warning("Unknown keys in config section '%s' are ignored: %s", section, unknown)
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration.

    Priority: explicit ``path``, then ``VEHICLE_ROUTE_CONFIG``, then ``configs/default.yaml``
    next to the project. A missing explicitly requested file is an error; a missing default
    file falls back to built-in defaults with a warning.
    """
    env_path = os.getenv("VEHICLE_ROUTE_CONFIG")
    requested = path if path is not None else (env_path or None)
    config_path = Path(requested) if requested is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        if requested is not None:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.warning("Default config %s not found, using built-in defaults", config_path)
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return AppConfig(
        default_algorithm=raw.get("default_algorithm", "dp"),
        route=_construct(RouteConfig, raw.get("route"), "route"),
        vehicle=_construct(VehicleParams, raw.get("vehicle"), "vehicle"),
        optimization=_construct(OptimizationConfig, raw.get("optimization"), "optimization"),
    )
