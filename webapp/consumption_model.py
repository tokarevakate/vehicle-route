import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .profile_model import ProfilePoint


@dataclass
class VehicleParams:
    mass:               float = 1500.0
    cd:                 float = 0.29
    frontal_area:       float = 2.2
    rho_air:            float = 1.2
    rolling_coeff_base: float = 0.01


@dataclass
class FuelCoeffs:
    """Коэффициенты модели q(v,G,R) = a0 + a1*v + a2*v² + b1*G + c1*v²/R.

    Можно задать вручную (из YAML/конфига) или вывести из физики через
    FuelCoeffs.from_vehicle(params).
    """
    a0: float =  6.70
    a1: float = -0.081
    a2: float =  0.000855
    b1: float =  1.34
    c1: float =  0.002

    @classmethod
    def from_vehicle(cls, params: VehicleParams) -> "FuelCoeffs":
        """Вывести a0, a1, a2 из физики автомобиля (только numpy)."""
        g_acc    = 9.81
        Hv       = 43.4e6
        eta_eff  = 0.25
        rho_fuel = 0.74
        idle_lph = 0.6

        def physical_q(v_kmh: float, grade: float = 0.0) -> float:
            if v_kmh <= 0:
                return 0.0
            v = v_kmh / 3.6
            f_roll  = params.rolling_coeff_base * (1 + v_kmh / 147)
            F_aero  = 0.5 * params.rho_air * params.cd * params.frontal_area * v**2
            F_roll  = params.mass * g_acc * f_roll
            F_grade = params.mass * g_acc * (grade / 100)
            F_drive = max(F_aero + F_roll + F_grade, 0.0)
            P_fuel  = F_drive * v / eta_eff
            L_dot   = (P_fuel / Hv) / rho_fuel
            return L_dot / v * 1e5 + idle_lph / v_kmh * 100

        # --- a0, a1, a2: МНК через np.linalg.lstsq (scipy не нужен) ---
        v_range = np.arange(10, 131, 1.0)
        q_phys  = np.array([physical_q(v) for v in v_range])
        A = np.column_stack([np.ones_like(v_range), v_range, v_range**2])
        coeffs_fit, _, _, _ = np.linalg.lstsq(A, q_phys, rcond=None)
        a0, a1, a2 = coeffs_fit

        # --- b1: линейная аппроксимация влияния уклона при 80 км/ч ---
        grades = np.array([-8.0, -5, -3, 0, 3, 5, 8])
        dq     = np.array([physical_q(80, gr) - physical_q(80, 0) for gr in grades])
        b1     = float(np.polyfit(grades, dq, 1)[0])

        return cls(a0=float(a0), a1=float(a1), a2=float(a2), b1=b1)


def fuel_consumption_model(
    v_kmh: float,
    grade_percent: float,
    radius: Optional[float],
    coeffs: FuelCoeffs,
) -> float:
    """q(v,G,R) = a0 + a1*v + a2*v² + b1*G + c1*v²/R  [л/100км]"""
    q = coeffs.a0 + coeffs.a1 * v_kmh + coeffs.a2 * v_kmh**2
    q += coeffs.b1 * grade_percent
    if radius is not None and radius > 0:
        q += coeffs.c1 * v_kmh**2 / radius
    return max(q, 0.5)


def optimal_speed(
    grade_percent: float,
    radius: Optional[float],
    coeffs: FuelCoeffs,
    v_min: float = 30.0,
    v_max: float = 90.0,
) -> float:
    """Find fuel-optimal speed on a segment given grade and curvature.

    Uses analytical minimum where possible; falls back to grid search
    when grade/curvature shift the optimum outside [v_min, v_max].
    """
    v_analytical = -coeffs.a1 / (2 * coeffs.a2)

    ay_max = 2.0
    if radius is not None and radius > 0:
        v_max = min(v_max, math.sqrt(ay_max * radius) * 3.6)
    v_max = max(v_max, v_min)

    if v_min <= v_analytical <= v_max:
        return v_analytical

    v_arr  = np.arange(v_min, v_max + 1, 1.0)
    q_vals = [fuel_consumption_model(v, grade_percent, radius, coeffs) for v in v_arr]
    return float(v_arr[np.argmin(q_vals)])


def build_speed_profile(
    profile: List[ProfilePoint],
    params: Optional[VehicleParams] = None,
    coeffs: Optional[FuelCoeffs] = None,
) -> List[dict]:
    """Attach optimal speed (and fuel stats) to profile points.

    Returns list of dicts ready to be serialized as JSON for the API.
    If coeffs is not provided, it is derived from vehicle physics via
    FuelCoeffs.from_vehicle(params).
    """
    if params is None:
        params = VehicleParams()
    if coeffs is None:
        coeffs = FuelCoeffs.from_vehicle(params)

    result: List[dict] = []
    fuel_cum = 0.0

    for p in profile:
        v_opt = optimal_speed(p.grade, p.radius, coeffs)
        q_val = fuel_consumption_model(v_opt, p.grade, p.radius, coeffs)
        fuel_cum += q_val

        result.append(
            {
                "index":           p.index,
                "lat":             p.lat,
                "lon":             p.lon,
                "elevation":       p.elevation,
                "distance":        p.s,
                "grade":           p.grade,
                "radius":          p.radius,
                "speed_optimal":   v_opt,
                "fuel_proxy":      q_val,
                "fuel_cumulative": fuel_cum,
            }
        )

    return result
