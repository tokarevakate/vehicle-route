import json
import logging

import pytest

from webapp.config import ALGORITHMS, DEFAULT_CONFIG_PATH, load_config
from webapp.optimization import OptimizationConfig


def write_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads_and_is_valid():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.default_algorithm in ALGORITHMS
    assert config.optimization.v_min_kmh < config.optimization.v_max_kmh


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_unknown_algorithm_is_rejected(tmp_path):
    path = write_yaml(tmp_path, "default_algorithm: mpc\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_empty_sections_use_defaults(tmp_path):
    path = write_yaml(tmp_path, "route:\nvehicle:\noptimization:\n")
    config = load_config(path)
    assert config.optimization == OptimizationConfig()


def test_root_must_be_mapping(tmp_path):
    path = write_yaml(tmp_path, "- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_keys_are_reported(tmp_path, caplog):
    path = write_yaml(tmp_path, "optimization:\n  v_max_kmph: 80\n")
    with caplog.at_level(logging.WARNING, logger="webapp.config"):
        load_config(path)
    assert "v_max_kmph" in caplog.text


@pytest.mark.parametrize(
    "text",
    [
        "optimization:\n  speed_step_kmh: 0\n",
        "optimization:\n  v_min_kmh: 100\n  v_max_kmh: 50\n",
        "optimization:\n  accel_min_mps2: 0.5\n",
        "route:\n  target_spacing_m: -1\n",
    ],
)
def test_invalid_ranges_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, text))


def test_summary_json_is_strict_for_failed_results():
    from webapp.optimization import _empty_result

    summary = _empty_result("dp", 3, "infeasible_at_segment_0").summary()
    assert summary["objective"] is None
    json.dumps(summary, allow_nan=False)
