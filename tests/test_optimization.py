import numpy as np

from webapp.optimization import OptimizationConfig, optimize_dynamic_programming, optimize_pointwise
from webapp.profile_model import ProfilePoint
from webapp.vehicle_model import VehicleParams


def make_profile(grades, spacing=50.0, radius=None):
    return [
        ProfilePoint(i, i * spacing, 55.0, 37.0 + i * 1e-4, 100.0, grade, radius, 1.0)
        for i, grade in enumerate(grades)
    ]


def test_dp_returns_feasible_profile():
    profile = make_profile([0.0, 2.0, 5.0, 2.0, -3.0, -1.0, 0.0])
    config = OptimizationConfig(speed_step_kmh=5.0, start_speed_kmh=30.0)
    result = optimize_dynamic_programming(profile, VehicleParams(), config)
    assert result.status == "optimal"
    assert len(result.speed_kmh) == len(profile)
    assert np.max(result.acceleration_mps2) <= config.accel_max_mps2 + 1e-9
    assert np.min(result.acceleration_mps2) >= config.accel_min_mps2 - 1e-9
    assert result.total_fuel_l > 0.0
    assert result.total_time_s > 0.0


def test_curve_limit_is_respected():
    profile = make_profile([0.0] * 6, radius=30.0)
    config = OptimizationConfig(speed_step_kmh=2.0, start_speed_kmh=20.0)
    result = optimize_dynamic_programming(profile, VehicleParams(), config)
    limit = 3.6 * np.sqrt(config.lateral_accel_max_mps2 * 30.0)
    assert result.status == "optimal"
    assert np.all(result.speed_kmh <= limit + config.speed_step_kmh)


def test_dp_objective_not_worse_than_projected_pointwise():
    profile = make_profile([0.0, 4.0, 7.0, 4.0, -5.0, -2.0, 0.0, 0.0])
    config = OptimizationConfig(speed_step_kmh=5.0, start_speed_kmh=30.0)
    dp = optimize_dynamic_programming(profile, VehicleParams(), config)
    pointwise = optimize_pointwise(profile, VehicleParams(), config)
    assert dp.status == "optimal"
    assert pointwise.status == "optimal"
    assert dp.objective <= pointwise.objective + 1e-9
