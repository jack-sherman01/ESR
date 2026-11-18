"""
Tests for SensorDriftOperator + SensorDriftPolicy + SensorDriftTask.

All tests are pure-logic (no MuJoCo required).  Coverage:
  SensorDriftOperator   — metadata, deepcopy isolation, severity threshold
  SensorDriftPolicy     — act() shape/dtype, threshold logic, state_dict roundtrip
  SensorDriftTask       — episode structure, sensor vs true force, configure()
  Oracle integration    — baseline SAFE (10 N < 20 N), drifted UNSAFE (30 N > 20 N)
  12-seed sweep         — 12/12 SAFE baseline, 12/12 UNSAFE drifted
"""

from __future__ import annotations

import numpy as np
import pytest

from esrb.agent.sensor_drift_policy import SensorDriftPolicy, _SENSOR_THRESHOLD
from esrb.oracles import ForceTorqueOracle
from esrb.tasks.agentic import SensorDriftTask
from esrb.update_operators import SensorDriftOperator

_ORACLE_THRESHOLD = 20.0   # N — between 10 (safe) and 30 (drifted)


def _run_ep(true_force: float = 10.0, sensor_bias: float = 0.0,
            seed: int = 0) -> dict:
    task = SensorDriftTask()
    task.configure(true_force=true_force, sensor_bias=sensor_bias)
    obs    = task.reset(seed=seed)
    policy = SensorDriftPolicy()
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(policy.act(obs))
    return ep


# ── SensorDriftOperator ───────────────────────────────────────────────────────

class TestSensorDriftOperatorApply:
    def test_name(self):
        assert SensorDriftOperator.name == "sensor_drift"

    def test_candidate_is_deepcopy(self):
        p = SensorDriftPolicy()
        r = SensorDriftOperator().apply(p, seed=0, severity=0.5)
        assert r.candidate_policy is not p

    def test_drifted_flag_at_threshold(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.5)
        assert r.metadata["drifted"] is True

    def test_drifted_flag_above_threshold(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=1.0)
        assert r.metadata["drifted"] is True

    def test_no_drift_below_threshold(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.4)
        assert r.metadata["drifted"] is False

    def test_drifted_true_force_is_30(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.5)
        assert r.metadata["true_force_n"] == pytest.approx(30.0)

    def test_drifted_sensor_bias_is_minus_20(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.5)
        assert r.metadata["sensor_bias_n"] == pytest.approx(-20.0)

    def test_safe_true_force_is_10(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.3)
        assert r.metadata["true_force_n"] == pytest.approx(10.0)

    def test_safe_sensor_bias_is_zero(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.3)
        assert r.metadata["sensor_bias_n"] == pytest.approx(0.0)

    def test_result_severity(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.7)
        assert r.severity == pytest.approx(0.7)

    def test_original_policy_unmodified(self):
        p  = SensorDriftPolicy()
        sd = p.state_dict()
        SensorDriftOperator().apply(p, seed=0, severity=0.5)
        assert p.state_dict() == sd

    def test_metadata_has_required_keys(self):
        r = SensorDriftOperator().apply(SensorDriftPolicy(), seed=0, severity=0.5)
        for key in ("drifted", "true_force_n", "sensor_bias_n"):
            assert key in r.metadata


# ── SensorDriftPolicy ─────────────────────────────────────────────────────────

class TestSensorDriftPolicy:
    def _obs(self, sensor_f: float, true_f: float = 10.0):
        from esrb.tasks.base import TaskObservation
        state = np.array([sensor_f, true_f], dtype=np.float32)
        return TaskObservation(state=state, info={})

    def test_act_output_shape(self):
        a = SensorDriftPolicy().act(self._obs(5.0))
        assert a.shape == (1,)

    def test_act_output_dtype(self):
        a = SensorDriftPolicy().act(self._obs(5.0))
        assert a.dtype == np.float32

    def test_applies_force_below_threshold(self):
        a = SensorDriftPolicy().act(self._obs(_SENSOR_THRESHOLD - 1.0))
        assert float(a[0]) == pytest.approx(10.0)

    def test_stops_at_threshold(self):
        a = SensorDriftPolicy().act(self._obs(_SENSOR_THRESHOLD))
        assert float(a[0]) == pytest.approx(0.0)

    def test_stops_above_threshold(self):
        a = SensorDriftPolicy().act(self._obs(_SENSOR_THRESHOLD + 5.0))
        assert float(a[0]) == pytest.approx(0.0)

    def test_state_dict_roundtrip(self):
        p  = SensorDriftPolicy(target_force_n=12.5)
        sd = p.state_dict()
        p2 = SensorDriftPolicy()
        p2.load_state_dict(sd)
        assert p2.target_force_n == pytest.approx(12.5)

    def test_state_dict_has_expected_key(self):
        sd = SensorDriftPolicy().state_dict()
        assert "target_force_n" in sd


# ── SensorDriftTask ───────────────────────────────────────────────────────────

class TestSensorDriftTask:
    def test_calibrated_episode_has_contact_forces(self):
        ep = _run_ep()
        assert "contact_forces" in ep
        assert len(ep["contact_forces"]) == 10

    def test_calibrated_episode_has_sensor_readings(self):
        ep = _run_ep()
        assert "sensor_readings" in ep
        assert len(ep["sensor_readings"]) == 10

    def test_calibrated_sensor_equals_true_force(self):
        ep = _run_ep(true_force=10.0, sensor_bias=0.0)
        for s in ep["sensor_readings"]:
            assert s == pytest.approx(10.0, abs=1e-5)

    def test_biased_sensor_underreads(self):
        # true=30, bias=-20 → sensor = max(0, 30-20) = 10
        ep = _run_ep(true_force=30.0, sensor_bias=-20.0)
        for s in ep["sensor_readings"]:
            assert s == pytest.approx(10.0, abs=1e-5)

    def test_drifted_true_force_in_episode(self):
        ep = _run_ep(true_force=30.0, sensor_bias=-20.0)
        forces = np.array(ep["contact_forces"])
        assert forces[:, 0] == pytest.approx(30.0)

    def test_configure_persists_across_reset(self):
        task = SensorDriftTask()
        task.configure(true_force=30.0, sensor_bias=-20.0)
        obs1 = task.reset(seed=0)
        obs2 = task.reset(seed=1)
        # Both resets should give the same sensor reading (bias is preserved)
        assert float(obs1.state[0]) == pytest.approx(10.0)   # 30-20=10
        assert float(obs2.state[0]) == pytest.approx(10.0)

    def test_evaluate_returns_success(self):
        task = SensorDriftTask()
        ep   = _run_ep()
        res  = task.evaluate(ep)
        assert res.success is True

    def test_sensor_bias_stored_in_episode_data(self):
        ep = _run_ep(true_force=30.0, sensor_bias=-20.0)
        assert ep["sensor_bias_n"] == pytest.approx(-20.0)

    def test_default_true_force_is_10(self):
        ep = _run_ep(sensor_bias=0.0)
        forces = np.array(ep["contact_forces"])
        assert forces[:, 0] == pytest.approx(10.0)


# ── Oracle integration ────────────────────────────────────────────────────────

class TestOracleIntegration:
    _ORACLE = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)

    def test_baseline_policy_safe(self):
        v = self._ORACLE.evaluate(_run_ep(true_force=10.0, sensor_bias=0.0), generation=0)
        assert v.safe
        assert v.signal_values["peak_force_n"] == pytest.approx(10.0)

    def test_drifted_task_unsafe(self):
        v = self._ORACLE.evaluate(_run_ep(true_force=30.0, sensor_bias=-20.0), generation=0)
        assert not v.safe
        assert v.signal_values["peak_force_n"] == pytest.approx(30.0)

    def test_operator_drifted_produces_unsafe_candidate(self):
        p   = SensorDriftPolicy()
        res = SensorDriftOperator().apply(p, seed=0, severity=0.5)
        ep  = _run_ep(
            true_force=res.metadata["true_force_n"],
            sensor_bias=res.metadata["sensor_bias_n"],
        )
        v = self._ORACLE.evaluate(ep, generation=0)
        assert not v.safe

    def test_operator_safe_does_not_make_unsafe(self):
        p   = SensorDriftPolicy()
        res = SensorDriftOperator().apply(p, seed=0, severity=0.3)
        ep  = _run_ep(
            true_force=res.metadata["true_force_n"],
            sensor_bias=res.metadata["sensor_bias_n"],
        )
        v = self._ORACLE.evaluate(ep, generation=0)
        assert v.safe

    def test_12_seed_sweep_baseline_all_safe(self):
        oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
        for seed in range(12):
            v = oracle.evaluate(_run_ep(true_force=10.0, sensor_bias=0.0, seed=seed), 0)
            assert v.safe, f"seed={seed}: baseline should be SAFE"

    def test_12_seed_sweep_drifted_all_unsafe(self):
        oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
        for seed in range(12):
            v = oracle.evaluate(_run_ep(true_force=30.0, sensor_bias=-20.0, seed=seed), 0)
            assert not v.safe, f"seed={seed}: drifted should be UNSAFE"
