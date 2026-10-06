from __future__ import annotations

"""Backward-compatible public API for fuel and speed-profile calculations.

The former polynomial model is retained only as a documented baseline. New route
optimization uses the SI-unit road-load and fuel-rate model in vehicle_model.py.
"""

from dataclasses import dataclass
from typing import List, Optional

from .optimization import OptimizationConfig, optimize_pointwise, result_to_points
from .profile_model import ProfilePoint
from .vehicle_model import VehicleParams, steady_fuel_l_per_100km


@dataclass
class FuelCoeffs:
    """Deprecated coefficients of the original pointwise polynomial baseline."""

    a0: float = 6.70
    a1: float = -0.081
    a2: float = 0.000855
    b1_up: float = 1.34
    b1_down: float = 0.20
    c1: float = 0.002
    d1: float = 0.0

    @classmethod
    def from_vehicle(cls, params: VehicleParams) -> "FuelCoeffs":
        return cls()


def fuel_consumption_model(
    v_kmh: float,
    grade_percent: float,
    radius: Optional[float],
    coeffs: Optional[FuelCoeffs] = None,
    accel: float = 0.0,
    params: Optional[VehicleParams] = None,
) -> float:
    """Physically coupled steady-state consumption in L/100 km.

    Acceleration is accepted for API compatibility. Dynamic acceleration fuel is
    evaluated between route states by evaluate_transition().
    """
    del coeffs, accel
    return steady_fuel_l_per_100km(v_kmh, grade_percent, params or VehicleParams())


def build_speed_profile(
    profile: List[ProfilePoint],
    params: Optional[VehicleParams] = None,
    coeffs: Optional[FuelCoeffs] = None,
) -> List[dict]:
    """Build the feasible pointwise baseline; DP/MPC live in optimization.py."""
    del coeffs
    params = params or VehicleParams()
    result = optimize_pointwise(profile, params, OptimizationConfig())
    return result_to_points(profile, result)
