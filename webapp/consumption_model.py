import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

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
    """Коэффициенты модели q(v, G, R, a).

    q = a0 + a1*v + a2*v^2
      + b1_up*max(G,0) + b1_down*min(G,0)
      + c1*v^2/R
      + d1*v*a

    b1_up  > 0 : extra fuel on climbs
    b1_down >= 0: near-zero on descents (overrun / fuel-cut)
    d1     >= 0: VSP cross-term (acceleration penalty)
    """
    a0:      float =  6.70
    a1:      float = -0.081
    a2:      float =  0.000855
    b1_up:   float =  1.34
    b1_down: float =  0.0
    c1:      float =  0.002
    d1:      float =  0.0

    @classmethod
    def from_vehicle(cls, params: VehicleParams) -> "FuelCoeffs":
        """Derive all coefficients from first-principles vehicle physics."""
        g_acc    = 9.81
        Hv       = 43.4e6
        eta_eff  = 0.25
        rho_fuel = 0.74
        idle_lph = 0.6

        # P4: quadratic rolling resistance per ISO 28580
        def f_roll_fn(v_kmh: float) -> float:
            return params.rolling_coeff_base * (
                1.0 + v_kmh / 147.0 + 0.3 * (v_kmh / 147.0) ** 2
            )

        # P1: accel argument added (default 0.0 for quasi-static calibration)
        def physical_q(v_kmh: float, grade: float = 0.0, accel: float = 0.0) -> float:
            if v_kmh <= 0:
                return 0.0
            v = v_kmh / 3.6
            F_aero  = 0.5 * params.rho_air * params.cd * params.frontal_area * v ** 2
            F_roll  = params.mass * g_acc * f_roll_fn(v_kmh)
            F_grade = params.mass * g_acc * (grade / 100.0)
            # P1: inertial term m*a
            F_drive = max(F_aero + F_roll + F_grade + params.mass * accel, 0.0)
            P_fuel  = F_drive * v / eta_eff
            L_dot   = (P_fuel / Hv) / rho_fuel
            return L_dot / v * 1e5 + idle_lph / v_kmh * 100.0

        # --- a0, a1, a2: OLS fit over flat quasi-static range ---
        v_range = np.arange(10, 131, 1.0)
        q_phys  = np.array([physical_q(v) for v in v_range])
        A = np.column_stack([np.ones_like(v_range), v_range, v_range ** 2])
        coeffs_fit, _, _, _ = np.linalg.lstsq(A, q_phys, rcond=None)
        a0, a1, a2 = coeffs_fit

        # --- P2: separate b1_up (climb) and b1_down (descent / overrun) ---
        q_flat = physical_q(80.0, 0.0)
        # overrun floor: engine fuel-cut -> ~15 % of flat consumption
        overrun_floor = 0.15 * q_flat

        grades_up   = np.array([0.0, 3.0, 5.0, 8.0])
        dq_up       = np.array([physical_q(80.0, gr) - q_flat for gr in grades_up])
        b1_up       = float(np.polyfit(grades_up, dq_up, 1)[0])

        grades_down = np.array([-8.0, -5.0, -3.0, 0.0])
        dq_down     = np.array([
            max(physical_q(80.0, gr), overrun_floor) - q_flat
            for gr in grades_down
        ])
        b1_down = float(np.polyfit(grades_down, dq_down, 1)[0])
        # b1_down should be <= 0 numerically; clamp to non-positive
        b1_down = min(b1_down, 0.0)

        # --- P1: d1 — VSP cross-term calibrated at 80 km/h, a = 1 m/s^2 ---
        q_accel = physical_q(80.0, 0.0, accel=1.0)
        d1 = max((q_accel - q_flat) / 80.0, 0.0)  # (dq / (v*a)) at v=80, a=1

        return cls(
            a0=float(a0), a1=float(a1), a2=float(a2),
            b1_up=b1_up, b1_down=b1_down,
            d1=d1,
        )


def fuel_consumption_model(
    v_kmh: float,
    grade_percent: float,
    radius: Optional[float],
    coeffs: FuelCoeffs,
    accel: float = 0.0,
) -> float:
    """q(v, G, R, a) [л/100км] — phenomenological fuel consumption model.

    P2: asymmetric grade term (b1_up for climbs, b1_down for descents).
    P1: VSP cross-term d1*v*a.
    P3: curvature c1*v^2/R kept as secondary tyre-loss term.
    """
    q = coeffs.a0 + coeffs.a1 * v_kmh + coeffs.a2 * v_kmh ** 2
    # P2: asymmetric grade
    q += coeffs.b1_up * max(grade_percent, 0.0) + coeffs.b1_down * min(grade_percent, 0.0)
    # P3: curvature (secondary tyre loss)
    if radius is not None and radius > 0:
        q += coeffs.c1 * v_kmh ** 2 / radius
    # P1: acceleration VSP cross-term
    q += coeffs.d1 * v_kmh * accel
    return max(q, 0.5)


def optimal_speed(
    grade_percent: float,
    radius: Optional[float],
    coeffs: FuelCoeffs,
    v_min: float = 30.0,
    v_max: float = 90.0,
    accel: float = 0.0,
) -> float:
    """Find fuel-optimal speed on a segment given grade, curvature and accel.

    P6: uses scipy.optimize.minimize_scalar (bounded) instead of analytical
    parabola vertex — required now that d1*v*a makes the objective non-parabolic.
    Falls back to v_min if the feasible interval is degenerate.
    """
    ay_max = 2.0
    if radius is not None and radius > 0:
        v_max = min(v_max, math.sqrt(ay_max * radius) * 3.6)
    v_max = max(v_max, v_min)

    if v_min >= v_max:
        return v_min

    res = minimize_scalar(
        lambda v: fuel_consumption_model(v, grade_percent, radius, coeffs, accel),
        bounds=(v_min, v_max),
        method="bounded",
    )
    return float(res.x)


def build_speed_profile(
    profile: List[ProfilePoint],
    params: Optional[VehicleParams] = None,
    coeffs: Optional[FuelCoeffs] = None,
) -> List[dict]:
    """Attach optimal speed and fuel stats to profile points.

    Returns list of dicts ready to be serialized as JSON for the API.
    coeffs is derived from vehicle physics via FuelCoeffs.from_vehicle(params)
    when not provided.
    """
    if params is None:
        params = VehicleParams()
    if coeffs is None:
        coeffs = FuelCoeffs.from_vehicle(params)

    result: List[dict] = []
    fuel_cum = 0.0
    prev_s = profile[0].s if profile else 0.0

    for p in profile:
        v_opt = optimal_speed(p.grade, p.radius, coeffs)
        q_val = fuel_consumption_model(v_opt, p.grade, p.radius, coeffs)

        segment_km = (p.s - prev_s) / 1000.0
        prev_s = p.s
        fuel_cum += q_val * segment_km / 100.0

        result.append(
            {
                "index":           p.index,
                "lat":             p.lat,
                "lon":             p.lon,
                "elevation":       p.elevation,
                "distance":        round(p.s, 3),
                "grade":           p.grade,
                "radius":          p.radius,
                "speed_optimal":   v_opt,
                "fuel_proxy":      round(q_val, 4),
                "fuel_cumulative": round(fuel_cum, 6),
            }
        )

    return result
