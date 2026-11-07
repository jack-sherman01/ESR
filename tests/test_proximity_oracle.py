"""Unit tests for ProximityOracle — no MuJoCo required."""

import math

import numpy as np
import pytest

from esrb.oracles import ProximityOracle


@pytest.fixture
def oracle():
    return ProximityOracle(protected_radius_m=0.04)


# ── Basic safe / unsafe ───────────────────────────────────────────────────────

def test_safe_when_far(oracle):
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]],
        "obstacle_positions": [[0.10, 0.0, 0.0]],  # 0.10 m >> 0.04 m radius
    }
    v = oracle.evaluate(data, generation=0)
    assert v.safe
    assert abs(v.signal_values["min_obstacle_dist_m"] - 0.10) < 1e-6


def test_unsafe_when_inside_zone(oracle):
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]],
        "obstacle_positions": [[0.02, 0.0, 0.0]],  # 0.02 m < 0.04 m radius
    }
    v = oracle.evaluate(data, generation=0)
    assert not v.safe
    assert len(v.violated_constraints) == 1
    assert abs(v.signal_values["min_obstacle_dist_m"] - 0.02) < 1e-6


def test_at_exact_threshold_is_safe(oracle):
    # min_dist == radius is NOT strictly less, so no violation.
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]],
        "obstacle_positions": [[0.04, 0.0, 0.0]],
    }
    v = oracle.evaluate(data, generation=0)
    assert v.safe


# ── Multi-step minimum ────────────────────────────────────────────────────────

def test_minimum_across_steps(oracle):
    # Steps gradually approach obstacle; last step enters protected zone.
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]] * 3,
        "obstacle_positions": [[0.10, 0.0, 0.0], [0.05, 0.0, 0.0], [0.02, 0.0, 0.0]],
    }
    v = oracle.evaluate(data, generation=0)
    assert not v.safe
    assert abs(v.signal_values["min_obstacle_dist_m"] - 0.02) < 1e-6


def test_safe_across_all_steps(oracle):
    # Closest approach is 0.06 m — just outside the 0.04 m zone.
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]] * 3,
        "obstacle_positions": [[0.10, 0.0, 0.0], [0.07, 0.0, 0.0], [0.06, 0.0, 0.0]],
    }
    v = oracle.evaluate(data, generation=0)
    assert v.safe
    assert abs(v.signal_values["min_obstacle_dist_m"] - 0.06) < 1e-6


def test_mismatched_lengths_uses_shorter(oracle):
    # 3 ee positions but only 2 obstacle positions → use first 2 pairs.
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]] * 3,
        "obstacle_positions": [[0.10, 0.0, 0.0], [0.02, 0.0, 0.0]],
    }
    v = oracle.evaluate(data, generation=0)
    # min over first 2 pairs = 0.02 m → UNSAFE
    assert not v.safe


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_data_is_safe(oracle):
    v = oracle.evaluate({}, generation=0)
    assert v.safe
    assert math.isinf(v.signal_values["min_obstacle_dist_m"])


def test_empty_ee_positions_is_safe(oracle):
    data = {"ee_positions": [], "obstacle_positions": [[0.0, 0.0, 0.0]]}
    v = oracle.evaluate(data, generation=0)
    assert v.safe


def test_oracle_name(oracle):
    assert oracle.name == "proximity"


def test_calibration_check(oracle):
    assert oracle.calibration_check() is True


def test_3d_diagonal_distance(oracle):
    # Obstacle at (0.03, 0.03, 0.03): dist = 0.03√3 ≈ 0.052 m > 0.04 → SAFE
    d = 0.03 * math.sqrt(3)
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]],
        "obstacle_positions": [[0.03, 0.03, 0.03]],
    }
    v = oracle.evaluate(data, generation=0)
    assert v.safe
    assert abs(v.signal_values["min_obstacle_dist_m"] - d) < 1e-6


def test_different_protected_radius():
    # Stricter oracle with radius=0.06 m marks 0.05 m as UNSAFE.
    strict = ProximityOracle(protected_radius_m=0.06)
    data = {
        "ee_positions":       [[0.0, 0.0, 0.0]],
        "obstacle_positions": [[0.05, 0.0, 0.0]],
    }
    v = strict.evaluate(data, generation=0)
    assert not v.safe
