import pytest

from webapp.vehicle_model import VehicleParams, evaluate_transition, road_load, steady_fuel_l_per_100km


def test_climb_increases_road_load_and_consumption():
    params = VehicleParams()
    flat = road_load(20.0, 0.0, params)
    climb = road_load(20.0, 5.0, params)
    assert climb.total_n > flat.total_n
    assert steady_fuel_l_per_100km(72.0, 5.0, params) > steady_fuel_l_per_100km(72.0, 0.0, params)


def test_mass_changes_grade_force_not_aerodynamic_force():
    light = VehicleParams(mass_kg=1200.0)
    heavy = VehicleParams(mass_kg=2000.0)
    light_load = road_load(25.0, 4.0, light)
    heavy_load = road_load(25.0, 4.0, heavy)
    assert heavy_load.grade_n > light_load.grade_n
    assert heavy_load.aerodynamic_n == pytest.approx(light_load.aerodynamic_n)


def test_transition_units_and_energy_are_positive():
    metrics = evaluate_transition(20.0, 21.0, 50.0, 2.0, VehicleParams())
    assert metrics.feasible
    assert metrics.dt_s > 0.0
    assert metrics.fuel_l > 0.0
    assert metrics.traction_force_n > 0.0


def test_downhill_deceleration_can_activate_fuel_cut():
    metrics = evaluate_transition(20.0, 18.0, 50.0, -6.0, VehicleParams())
    assert metrics.feasible
    assert metrics.fuel_l == pytest.approx(0.0)
