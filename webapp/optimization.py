from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .profile_model import ProfilePoint
from .vehicle_model import VehicleParams, evaluate_transition, steady_fuel_l_per_100km


@dataclass(frozen=True)
class OptimizationConfig:
    v_min_kmh: float = 20.0
    v_max_kmh: float = 90.0
    speed_step_kmh: float = 2.0
    start_speed_kmh: float = 30.0
    end_speed_kmh: Optional[float] = None
    accel_min_mps2: float = -2.5
    accel_max_mps2: float = 1.5
    lateral_accel_max_mps2: float = 2.0
    time_weight_l_per_s: float = 0.00002
    brake_weight_l_per_kwh: float = 0.02
    accel_weight: float = 0.00001
    jerk_weight: float = 0.0
    mpc_horizon_m: float = 1000.0
    mpc_apply_segments: int = 5


@dataclass
class OptimizationResult:
    method: str
    speed_kmh: np.ndarray
    acceleration_mps2: np.ndarray
    traction_force_n: np.ndarray
    brake_force_n: np.ndarray
    segment_fuel_l: np.ndarray
    segment_time_s: np.ndarray
    brake_energy_kwh: np.ndarray
    gear: np.ndarray
    engine_rpm: np.ndarray
    objective: float
    status: str

    @property
    def total_fuel_l(self) -> float:
        return float(np.sum(self.segment_fuel_l))

    @property
    def total_time_s(self) -> float:
        return float(np.sum(self.segment_time_s))

    def summary(self) -> dict:
        return {
            "method": self.method,
            "status": self.status,
            "objective": self.objective,
            "total_fuel_l": self.total_fuel_l,
            "total_time_s": self.total_time_s,
            "brake_energy_kwh": float(np.sum(self.brake_energy_kwh)),
            "max_accel_mps2": float(np.max(self.acceleration_mps2, initial=0.0)),
            "min_accel_mps2": float(np.min(self.acceleration_mps2, initial=0.0)),
        }


def curvature_speed_limit_kmh(point: ProfilePoint, config: OptimizationConfig) -> float:
    if point.radius is None or point.radius <= 0.0:
        return config.v_max_kmh
    return min(
        config.v_max_kmh,
        3.6 * math.sqrt(config.lateral_accel_max_mps2 * point.radius),
    )


def _transition_cost(metrics, config: OptimizationConfig, previous_accel: float = 0.0) -> float:
    jerk = 0.0
    if metrics.dt_s > 0.0:
        jerk = (metrics.acceleration_mps2 - previous_accel) / metrics.dt_s
    return (
        metrics.fuel_l
        + config.time_weight_l_per_s * metrics.dt_s
        + config.brake_weight_l_per_kwh * metrics.brake_energy_kwh
        + config.accel_weight * metrics.acceleration_mps2**2 * metrics.dt_s
        + config.jerk_weight * jerk**2 * metrics.dt_s
    )


def _empty_result(method: str, n: int, status: str) -> OptimizationResult:
    zeros = np.zeros(n, dtype=float)
    return OptimizationResult(
        method, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
        zeros.copy(), zeros.copy(), np.zeros(n, dtype=int), zeros.copy(), math.inf, status
    )


def optimize_dynamic_programming(
    profile: list[ProfilePoint],
    params: Optional[VehicleParams] = None,
    config: Optional[OptimizationConfig] = None,
    method_name: str = "dp",
) -> OptimizationResult:
    params = params or VehicleParams()
    config = config or OptimizationConfig()
    n = len(profile)
    if n < 2:
        return _empty_result(method_name, n, "route_too_short")

    speeds_kmh = np.arange(
        config.v_min_kmh,
        config.v_max_kmh + 0.5 * config.speed_step_kmh,
        config.speed_step_kmh,
    )
    speeds_mps = speeds_kmh / 3.6
    m = len(speeds_kmh)
    limits = np.array([curvature_speed_limit_kmh(p, config) for p in profile])
    parents = np.full((n, m), -1, dtype=np.int16 if m < 32767 else np.int32)
    costs = np.full(m, np.inf)
    start_idx = int(np.argmin(np.abs(speeds_kmh - config.start_speed_kmh)))
    if speeds_kmh[start_idx] > limits[0] + 1e-9:
        start_idx = int(np.argmin(np.where(speeds_kmh <= limits[0], np.abs(speeds_kmh - config.start_speed_kmh), np.inf)))
    costs[start_idx] = 0.0

    for k in range(n - 1):
        ds = profile[k + 1].s - profile[k].s
        if ds <= 0.0:
            return _empty_result(method_name, n, f"non_increasing_distance_at_{k}")
        grade = 0.5 * (profile[k].grade + profile[k + 1].grade)
        next_costs = np.full(m, np.inf)
        allowed_next = np.flatnonzero(speeds_kmh <= limits[k + 1] + 1e-9)
        for i in np.flatnonzero(np.isfinite(costs)):
            accel = (speeds_mps[allowed_next] ** 2 - speeds_mps[i] ** 2) / (2.0 * ds)
            reachable = allowed_next[(accel >= config.accel_min_mps2) & (accel <= config.accel_max_mps2)]
            for j in reachable:
                metrics = evaluate_transition(speeds_mps[i], speeds_mps[j], ds, grade, params)
                if not metrics.feasible:
                    continue
                candidate = costs[i] + _transition_cost(metrics, config)
                if candidate < next_costs[j]:
                    next_costs[j] = candidate
                    parents[k + 1, j] = i
        costs = next_costs
        if not np.any(np.isfinite(costs)):
            return _empty_result(method_name, n, f"infeasible_at_segment_{k}")

    if config.end_speed_kmh is None:
        end_idx = int(np.argmin(costs))
    else:
        terminal = np.abs(speeds_kmh - config.end_speed_kmh) <= 0.5 * config.speed_step_kmh + 1e-9
        feasible_terminal = np.where(terminal, costs, np.inf)
        if not np.any(np.isfinite(feasible_terminal)):
            return _empty_result(method_name, n, "terminal_speed_infeasible")
        end_idx = int(np.argmin(feasible_terminal))

    indices = np.empty(n, dtype=int)
    indices[-1] = end_idx
    for k in range(n - 1, 0, -1):
        indices[k - 1] = parents[k, indices[k]]
        if indices[k - 1] < 0:
            return _empty_result(method_name, n, "backtracking_failed")
    speed_profile = speeds_kmh[indices]
    return evaluate_speed_profile(profile, speed_profile, params, config, method_name, float(costs[end_idx]))


def evaluate_speed_profile(
    profile: list[ProfilePoint],
    speed_kmh: Iterable[float],
    params: VehicleParams,
    config: OptimizationConfig,
    method: str,
    objective: Optional[float] = None,
) -> OptimizationResult:
    speed_kmh = np.asarray(list(speed_kmh), dtype=float)
    n = len(profile)
    acceleration = np.zeros(n)
    traction = np.zeros(n)
    brake = np.zeros(n)
    fuel = np.zeros(n)
    time_s = np.zeros(n)
    brake_energy = np.zeros(n)
    gear = np.zeros(n, dtype=int)
    rpm = np.zeros(n)
    calculated_objective = 0.0
    for k in range(n - 1):
        ds = profile[k + 1].s - profile[k].s
        grade = 0.5 * (profile[k].grade + profile[k + 1].grade)
        metrics = evaluate_transition(speed_kmh[k] / 3.6, speed_kmh[k + 1] / 3.6, ds, grade, params)
        if not metrics.feasible:
            return _empty_result(method, n, f"profile_infeasible_at_{k}")
        acceleration[k] = metrics.acceleration_mps2
        traction[k] = metrics.traction_force_n
        brake[k] = metrics.brake_force_n
        fuel[k] = metrics.fuel_l
        time_s[k] = metrics.dt_s
        brake_energy[k] = metrics.brake_energy_kwh
        gear[k] = metrics.gear
        rpm[k] = metrics.engine_rpm
        calculated_objective += _transition_cost(metrics, config)
    return OptimizationResult(
        method=method,
        speed_kmh=speed_kmh,
        acceleration_mps2=acceleration,
        traction_force_n=traction,
        brake_force_n=brake,
        segment_fuel_l=fuel,
        segment_time_s=time_s,
        brake_energy_kwh=brake_energy,
        gear=gear,
        engine_rpm=rpm,
        objective=calculated_objective if objective is None else objective,
        status="optimal",
    )


def optimize_pointwise(
    profile: list[ProfilePoint],
    params: Optional[VehicleParams] = None,
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    params = params or VehicleParams()
    config = config or OptimizationConfig()
    if len(profile) < 2:
        return _empty_result("pointwise", len(profile), "route_too_short")
    speeds = np.zeros(len(profile))
    for k, point in enumerate(profile):
        upper = curvature_speed_limit_kmh(point, config)
        lower = min(config.v_min_kmh, upper)
        if upper <= 0.0:
            return _empty_result("pointwise", len(profile), f"no_speed_feasible_at_{k}")
        result = minimize_scalar(
            lambda v: steady_fuel_l_per_100km(v, point.grade, params),
            bounds=(max(lower, 1.0), upper),
            method="bounded",
        )
        speeds[k] = float(result.x)

    speeds[0] = min(config.start_speed_kmh, curvature_speed_limit_kmh(profile[0], config))
    for k in range(len(profile) - 1):
        ds = profile[k + 1].s - profile[k].s
        max_next_mps = math.sqrt(max((speeds[k] / 3.6) ** 2 + 2.0 * config.accel_max_mps2 * ds, 0.0))
        speeds[k + 1] = min(speeds[k + 1], max_next_mps * 3.6)
    for k in range(len(profile) - 2, -1, -1):
        ds = profile[k + 1].s - profile[k].s
        max_prev_mps = math.sqrt(max((speeds[k + 1] / 3.6) ** 2 - 2.0 * config.accel_min_mps2 * ds, 0.0))
        speeds[k] = min(speeds[k], max_prev_mps * 3.6)
    return evaluate_speed_profile(profile, speeds, params, config, "pointwise")


def optimize_mpc(
    profile: list[ProfilePoint],
    params: Optional[VehicleParams] = None,
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    params = params or VehicleParams()
    config = config or OptimizationConfig()
    n = len(profile)
    if n < 2:
        return _empty_result("mpc", n, "route_too_short")
    assembled: list[float] = [config.start_speed_kmh]
    cursor = 0
    while cursor < n - 1:
        horizon_end = cursor + 1
        while horizon_end < n - 1 and profile[horizon_end].s - profile[cursor].s < config.mpc_horizon_m:
            horizon_end += 1
        local_config = OptimizationConfig(**{
            **asdict(config),
            "start_speed_kmh": assembled[-1],
            "end_speed_kmh": config.end_speed_kmh if horizon_end == n - 1 else None,
        })
        local = optimize_dynamic_programming(
            profile[cursor:horizon_end + 1], params, local_config, method_name="mpc_horizon"
        )
        if local.status != "optimal":
            return _empty_result("mpc", n, f"{local.status}_from_{cursor}")
        apply_count = min(config.mpc_apply_segments, horizon_end - cursor, n - 1 - cursor)
        assembled.extend(local.speed_kmh[1:apply_count + 1].tolist())
        cursor += apply_count
    return evaluate_speed_profile(profile, assembled[:n], params, config, "mpc")


def result_to_points(profile: list[ProfilePoint], result: OptimizationResult) -> list[dict]:
    fuel_cumulative = np.cumsum(result.segment_fuel_l)
    time_cumulative = np.cumsum(result.segment_time_s)
    points: list[dict] = []
    for i, point in enumerate(profile):
        points.append({
            "index": point.index,
            "lat": point.lat,
            "lon": point.lon,
            "elevation": point.elevation,
            "distance": round(point.s, 3),
            "grade": point.grade,
            "radius": point.radius,
            "profile_confidence": point.confidence,
            "speed_optimal": float(result.speed_kmh[i]),
            "acceleration": float(result.acceleration_mps2[i]),
            "traction_force": float(result.traction_force_n[i]),
            "brake_force": float(result.brake_force_n[i]),
            "gear": int(result.gear[i]),
            "engine_rpm": float(result.engine_rpm[i]),
            "fuel_segment": float(result.segment_fuel_l[i]),
            "fuel_cumulative": float(fuel_cumulative[i]),
            "time_cumulative": float(time_cumulative[i]),
        })
    return points
