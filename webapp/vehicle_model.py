from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParams:
    mass_kg: float = 1500.0
    rotating_mass_factor: float = 1.04
    cd: float = 0.29
    frontal_area_m2: float = 2.2
    rho_air_kg_m3: float = 1.2
    rolling_coeff: float = 0.010
    drivetrain_efficiency: float = 0.92
    fuel_lhv_j_kg: float = 43.4e6
    fuel_density_kg_l: float = 0.74
    idle_fuel_lph: float = 0.6
    max_power_kw: float = 110.0
    max_tractive_force_n: float = 5500.0
    max_brake_force_n: float = 9000.0
    fuel_cut_min_speed_kmh: float = 15.0
    wheel_radius_m: float = 0.31
    final_drive_ratio: float = 3.9
    gear_ratios: tuple[float, ...] = (3.54, 1.92, 1.28, 0.91, 0.76, 0.64)
    upshift_speeds_kmh: tuple[float, ...] = (15.0, 30.0, 50.0, 75.0, 105.0)

    @property
    def equivalent_mass_kg(self) -> float:
        return self.mass_kg * self.rotating_mass_factor


@dataclass(frozen=True)
class RoadLoad:
    aerodynamic_n: float
    rolling_n: float
    grade_n: float

    @property
    def total_n(self) -> float:
        return self.aerodynamic_n + self.rolling_n + self.grade_n


@dataclass(frozen=True)
class TransitionMetrics:
    feasible: bool
    dt_s: float
    acceleration_mps2: float
    traction_force_n: float
    brake_force_n: float
    wheel_power_w: float
    fuel_l: float
    brake_energy_kwh: float
    gear: int
    engine_rpm: float
    efficiency: float


def rolling_coefficient(speed_mps: float, params: VehicleParams) -> float:
    speed_kmh = max(speed_mps, 0.0) * 3.6
    ratio = speed_kmh / 147.0
    return params.rolling_coeff * (1.0 + ratio + 0.3 * ratio * ratio)


def road_load(speed_mps: float, grade_percent: float, params: VehicleParams) -> RoadLoad:
    speed_mps = max(speed_mps, 0.0)
    alpha = math.atan(grade_percent / 100.0)
    aero = 0.5 * params.rho_air_kg_m3 * params.cd * params.frontal_area_m2 * speed_mps**2
    rolling = params.mass_kg * 9.81 * rolling_coefficient(speed_mps, params) * math.cos(alpha)
    grade = params.mass_kg * 9.81 * math.sin(alpha)
    return RoadLoad(aero, rolling, grade)


def select_gear(speed_kmh: float, params: VehicleParams) -> int:
    gear = 1
    for threshold in params.upshift_speeds_kmh:
        if speed_kmh >= threshold:
            gear += 1
    return min(gear, len(params.gear_ratios))


def engine_state(speed_mps: float, wheel_power_w: float, params: VehicleParams) -> tuple[int, float, float]:
    speed_kmh = speed_mps * 3.6
    gear = select_gear(speed_kmh, params)
    ratio = params.gear_ratios[gear - 1] * params.final_drive_ratio
    wheel_omega = speed_mps / params.wheel_radius_m
    engine_omega = max(wheel_omega * ratio, 800.0 * 2.0 * math.pi / 60.0)
    rpm = engine_omega * 60.0 / (2.0 * math.pi)
    engine_power_w = max(wheel_power_w, 0.0) / params.drivetrain_efficiency
    torque_nm = engine_power_w / engine_omega
    max_torque_nm = params.max_power_kw * 1000.0 / max(engine_omega, 1.0)
    load = min(max(torque_nm / max(max_torque_nm, 1.0), 0.0), 1.0)
    rpm_factor = ((rpm - 2400.0) / 2200.0) ** 2
    load_factor = ((load - 0.72) / 0.72) ** 2
    efficiency = min(max(0.36 - 0.07 * rpm_factor - 0.10 * load_factor, 0.18), 0.37)
    return gear, rpm, efficiency


def max_tractive_force(speed_mps: float, params: VehicleParams) -> float:
    power_limit = params.max_power_kw * 1000.0 / max(speed_mps, 1.0)
    return min(params.max_tractive_force_n, power_limit)


def fuel_rate_lps(
    speed_mps: float,
    traction_force_n: float,
    acceleration_mps2: float,
    brake_force_n: float,
    params: VehicleParams,
) -> tuple[float, int, float, float]:
    idle_lps = params.idle_fuel_lph / 3600.0
    speed_kmh = speed_mps * 3.6
    if traction_force_n <= 1e-6:
        fuel_cut = speed_kmh >= params.fuel_cut_min_speed_kmh and (
            acceleration_mps2 < -0.02 or brake_force_n > 1e-6
        )
        gear = select_gear(speed_kmh, params)
        _, rpm, _ = engine_state(speed_mps, 0.0, params)
        return (0.0 if fuel_cut else idle_lps), gear, rpm, 0.0

    wheel_power_w = traction_force_n * speed_mps
    gear, rpm, efficiency = engine_state(speed_mps, wheel_power_w, params)
    engine_power_w = wheel_power_w / params.drivetrain_efficiency
    fuel_kg_s = engine_power_w / (efficiency * params.fuel_lhv_j_kg)
    return max(fuel_kg_s / params.fuel_density_kg_l, idle_lps), gear, rpm, efficiency


def evaluate_transition(
    speed0_mps: float,
    speed1_mps: float,
    distance_m: float,
    grade_percent: float,
    params: VehicleParams,
) -> TransitionMetrics:
    if distance_m <= 0.0 or speed0_mps < 0.0 or speed1_mps < 0.0:
        return TransitionMetrics(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    speed_avg = max(0.5 * (speed0_mps + speed1_mps), 0.1)
    acceleration = (speed1_mps**2 - speed0_mps**2) / (2.0 * distance_m)
    load = road_load(speed_avg, grade_percent, params)
    required_force = params.equivalent_mass_kg * acceleration + load.total_n
    traction = max(required_force, 0.0)
    brake = max(-required_force, 0.0)
    feasible = traction <= max_tractive_force(speed_avg, params) + 1e-9 and brake <= params.max_brake_force_n + 1e-9
    dt_s = 2.0 * distance_m / max(speed0_mps + speed1_mps, 0.2)
    rate_lps, gear, rpm, efficiency = fuel_rate_lps(
        speed_avg, traction, acceleration, brake, params
    )
    wheel_power_w = traction * speed_avg
    fuel_l = rate_lps * dt_s
    brake_energy_kwh = brake * distance_m / 3.6e6
    return TransitionMetrics(
        feasible=feasible,
        dt_s=dt_s,
        acceleration_mps2=acceleration,
        traction_force_n=traction,
        brake_force_n=brake,
        wheel_power_w=wheel_power_w,
        fuel_l=fuel_l,
        brake_energy_kwh=brake_energy_kwh,
        gear=gear,
        engine_rpm=rpm,
        efficiency=efficiency,
    )


def steady_fuel_l_per_100km(
    speed_kmh: float,
    grade_percent: float,
    params: VehicleParams,
) -> float:
    if speed_kmh <= 0.0:
        return math.inf
    speed_mps = speed_kmh / 3.6
    load = road_load(speed_mps, grade_percent, params)
    traction = max(load.total_n, 0.0)
    brake = max(-load.total_n, 0.0)
    rate_lps, _, _, _ = fuel_rate_lps(speed_mps, traction, 0.0, brake, params)
    return rate_lps / speed_mps * 100_000.0
