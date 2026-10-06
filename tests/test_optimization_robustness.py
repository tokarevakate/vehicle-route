import numpy as np

from webapp.optimization import (
    OptimizationConfig,
    evaluate_speed_profile,
    optimize_dynamic_programming,
)
from webapp.profile_model import ProfilePoint
from webapp.vehicle_model import VehicleParams


def make_profile(grades, spacing=50.0, radius=None):
    return [
        ProfilePoint(i, i * spacing, 55.0, 37.0 + i * 1e-4, 100.0, grade, radius, 1.0)
        for i, grade in enumerate(grades)
    ]


def test_dp_handles_curve_limit_below_v_min():
    profile = make_profile([0.0] * 6, radius=10.0)
    config = OptimizationConfig(speed_step_kmh=2.0, start_speed_kmh=20.0)
    result = optimize_dynamic_programming(profile, VehicleParams(), config)
    limit = 3.6 * np.sqrt(config.lateral_accel_max_mps2 * 10.0)
    assert result.status == "optimal"
    assert np.all(result.speed_kmh <= limit + 1e-6)


def test_dp_not_worse_than_constant_speed_on_grid():
    profile = make_profile([0.0, 3.0, 6.0, 3.0, -4.0, -2.0, 0.0, 0.0])
    config = OptimizationConfig(speed_step_kmh=5.0, start_speed_kmh=30.0)
    params = VehicleParams()
    dp = optimize_dynamic_programming(profile, params, config)
    constant = evaluate_speed_profile(profile, [30.0] * len(profile), params, config, "constant")
    assert dp.status == "optimal"
    assert constant.status == "optimal"
    assert dp.objective <= constant.objective + 1e-9


def test_dp_acceleration_matches_speeds():
    profile = make_profile([0.0, 2.0, 5.0, 2.0, -3.0, -1.0, 0.0])
    config = OptimizationConfig(speed_step_kmh=5.0, start_speed_kmh=30.0)
    result = optimize_dynamic_programming(profile, VehicleParams(), config)
    v = result.speed_kmh / 3.6
    ds = np.diff([p.s for p in profile])
    expected = (v[1:] ** 2 - v[:-1] ** 2) / (2.0 * ds)
    assert result.status == "optimal"
    assert np.allclose(result.acceleration_mps2[:-1], expected, atol=1e-6)


def test_too_short_route_reports_status():
    result = optimize_dynamic_programming(make_profile([0.0]), VehicleParams(), OptimizationConfig())
    assert result.status == "route_too_short"


def test_non_increasing_distance_reports_status():
    profile = make_profile([0.0, 0.0, 0.0])
    profile[2] = ProfilePoint(2, profile[1].s, 55.0, 37.0, 100.0, 0.0, None, 1.0)
    result = optimize_dynamic_programming(profile, VehicleParams(), OptimizationConfig())
    assert result.status.startswith("non_increasing_distance")
