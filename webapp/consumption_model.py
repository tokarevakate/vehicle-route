import math
from dataclasses import dataclass
from typing import List, Optional

from .profile_model import ProfilePoint


@dataclass
class VehicleParams:
    mass: float = 1500.0  # kg
    cd: float = 0.29  # drag coefficient
    frontal_area: float = 2.2  # m^2
    rho_air: float = 1.2  # kg/m^3
    rolling_coeff_base: float = 0.01


def _rolling_coeff(v_kmh: float, base: float) -> float:
    """Speed-dependent rolling resistance coefficient.

    Simple approximation: f(v) = f0 * (1 + v/147) as often used in
    vehicle operating cost models.
    """
    return base * (1.0 + v_kmh / 147.0)


def compute_forces(v_kmh: float, grade_percent: float, params: VehicleParams) -> dict:
    """Compute basic longitudinal resistive forces at a given speed.

    This is not used directly in optimization, but kept for future
    physics-based extensions.
    """
    v_ms = v_kmh / 3.6
    g = 9.81

    # Aerodynamic drag
    F_aero = 0.5 * params.rho_air * params.cd * params.frontal_area * v_ms * v_ms

    # Rolling resistance
    f_roll = _rolling_coeff(v_kmh, params.rolling_coeff_base)
    F_roll = params.mass * g * f_roll

    # Grade resistance (small-angle approximation)
    F_grade = params.mass * g * (grade_percent / 100.0)

    return {
        "F_aero": F_aero,
        "F_roll": F_roll,
        "F_grade": F_grade,
    }


def fuel_consumption_model(
    v_kmh: float,
    grade_percent: float,
    radius: Optional[float],
) -> float:
    """Empirical-style fuel consumption surrogate q(v, G, R).

    q(v,G,R) = a0 + a1*v + a2*v^2 + b1*|G| + c1*v^2/R

    Coefficients are placeholders and should be calibrated with real data.
    Units: arbitrary (relative fuel cost). For optimization we only need
    relative values.
    """
    G = abs(grade_percent)

    a0 = 5.0
    a1 = -0.3
    a2 = 0.004
    b1 = 0.05
    c1 = 0.002

    v = v_kmh
    q_speed = a0 + a1 * v + a2 * v * v
    q_grade = b1 * G

    q_curve = 0.0
    if radius is not None and radius > 0.0:
        q_curve = c1 * (v * v) / radius

    return q_speed + q_grade + q_curve


def optimal_speed(
    grade_percent: float,
    radius: Optional[float],
    v_min: float = 30.0,
    v_max: float = 90.0,
    step: float = 1.0,
) -> float:
    """Find fuel-optimal speed on a segment given grade and curvature.

    - Minimizes fuel_consumption_model over v in [v_min, v_max].
    - Applies lateral acceleration constraint if radius is provided.
    """
    # lateral acceleration constraint
    ay_max = 2.0  # m/s^2
    if radius is not None and radius > 0.0:
        v_max_curve_ms = math.sqrt(ay_max * radius)
        v_max_curve_kmh = v_max_curve_ms * 3.6
        v_max = min(v_max, v_max_curve_kmh)

    # degenerate case: all constraints very low
    if v_max < v_min:
        v_max = v_min

    best_v = v_min
    best_q = fuel_consumption_model(v_min, grade_percent, radius)

    v = v_min
    while v <= v_max:
        q_val = fuel_consumption_model(v, grade_percent, radius)
        if q_val < best_q:
            best_q = q_val
            best_v = v
        v += step

    return best_v


def build_speed_profile(
    profile: List[ProfilePoint], params: Optional[VehicleParams] = None
) -> List[dict]:
    """Attach optimal speed (and optionally fuel stats) to profile points.

    Returns list of dicts ready to be serialized as JSON for the API.
    """
    if params is None:
        params = VehicleParams()

    result: List[dict] = []

    fuel_cum = 0.0

    for p in profile:
        v_opt = optimal_speed(p.grade, p.radius)

        # simple per-segment fuel proxy: q(v,G,R) * ds
        q_val = fuel_consumption_model(v_opt, p.grade, p.radius)
        # approximate segment length (use difference in s if available)
        # here we assume uniform spacing, so we skip ds and just accumulate q
        fuel_cum += q_val

        result.append(
            {
                "index": p.index,
                "lat": p.lat,
                "lon": p.lon,
                "elevation": p.elevation,
                "distance": p.s,
                "grade": p.grade,
                "radius": p.radius,
                "speed_optimal": v_opt,
                "fuel_proxy": q_val,
                "fuel_cumulative": fuel_cum,
            }
        )

    return result
