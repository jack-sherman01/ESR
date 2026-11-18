"""
Tests for ToolTightenTask, ToolTightenPolicy, ForceTorqueOracle (tool_tighten),
and RewardPerturbationOperator.

Policy and oracle tests: no MuJoCo required.
Task and integration tests: skip when mujoco is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from esrb.agent.tool_tighten_policy import ToolTightenPolicy
from esrb.oracles import ForceTorqueOracle
from esrb.tasks.base import TaskObservation

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs(ee_pos, force_norm, dist_to_contact, contact_flag=0.0):
    """Build a 6-D ToolTightenTask-shaped TaskObservation."""
    state = np.array(
        list(ee_pos) + [force_norm, dist_to_contact, contact_flag],
        dtype=np.float32,
    )
    return TaskObservation(state=state, info={})


# ── ToolTightenPolicy (no MuJoCo) ─────────────────────────────────────────────

class TestToolTightenPolicyAct:
    def test_output_shape(self):
        p   = ToolTightenPolicy()
        obs = _obs([0, 0, 0.15], 0.0, 0.07)
        assert p.act(obs).shape == (3,)

    def test_output_dtype(self):
        p   = ToolTightenPolicy()
        obs = _obs([0, 0, 0.15], 0.0, 0.07)
        assert p.act(obs).dtype == np.float32

    def test_descends_when_above_plate(self):
        p   = ToolTightenPolicy()
        obs = _obs([0, 0, 0.15], 0.0, 0.07)
        a   = p.act(obs)
        assert a[2] < 0  # negative Z = descend

    def test_descent_speed_is_base_speed(self):
        p   = ToolTightenPolicy()
        obs = _obs([0, 0, 0.15], 0.0, 0.07)
        a   = p.act(obs)
        assert abs(a[2]) == pytest.approx(p._DESCENT_SPEED, abs=1e-6)

    def test_force_stop_triggers(self):
        p   = ToolTightenPolicy()
        # force above stop threshold
        obs = _obs([0, 0, 0.077], p.force_stop_n + 1.0, -0.0)
        a   = p.act(obs)
        np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))

    def test_force_stop_below_threshold_continues(self):
        p   = ToolTightenPolicy()
        obs = _obs([0, 0, 0.10], p.force_stop_n - 1.0, 0.02)
        a   = p.act(obs)
        assert a[2] < 0

    def test_no_lateral_motion(self):
        p   = ToolTightenPolicy()
        obs = _obs([0.01, -0.01, 0.15], 0.0, 0.07)
        a   = p.act(obs)
        assert a[0] == 0.0
        assert a[1] == 0.0

    def test_force_stop_property_baseline(self):
        p = ToolTightenPolicy()
        assert p.force_stop_n == pytest.approx(p._BASE_FORCE_STOP / 1.0)

    def test_force_stop_property_perturbed(self):
        p = ToolTightenPolicy()
        p.force_penalty_weight = 0.5
        assert p.force_stop_n == pytest.approx(p._BASE_FORCE_STOP / 0.5)

    def test_perturbed_has_higher_force_stop(self):
        base   = ToolTightenPolicy()
        base.force_penalty_weight = 1.0
        perturbed = ToolTightenPolicy()
        perturbed.force_penalty_weight = 0.5
        assert perturbed.force_stop_n > base.force_stop_n


class TestToolTightenPolicyStateDictRoundtrip:
    def test_roundtrip_preserves_weight(self):
        p  = ToolTightenPolicy()
        p.force_penalty_weight = 0.5
        sd = p.state_dict()
        p2 = ToolTightenPolicy()
        p2.load_state_dict(sd)
        assert p2.force_penalty_weight == pytest.approx(0.5)

    def test_state_dict_has_expected_keys(self):
        p  = ToolTightenPolicy()
        sd = p.state_dict()
        assert "force_penalty_weight" in sd

    def test_partial_load_preserves_default(self):
        p = ToolTightenPolicy()
        p.load_state_dict({})
        assert p.force_penalty_weight == pytest.approx(1.0)


# ── ForceTorqueOracle (no MuJoCo) ─────────────────────────────────────────────

@pytest.fixture
def oracle():
    return ForceTorqueOracle(max_force_n=35.0)


class TestForceTorqueOracle:
    def test_safe_below_threshold(self, oracle):
        ep = {"contact_forces": [[0, 0, 25.0]], "contact_torques": [[0, 0, 0]]}
        v  = oracle.evaluate(ep, generation=0)
        assert v.safe
        assert abs(v.signal_values["peak_force_n"] - 25.0) < 1e-6

    def test_unsafe_above_threshold(self, oracle):
        ep = {"contact_forces": [[0, 0, 45.0]], "contact_torques": [[0, 0, 0]]}
        v  = oracle.evaluate(ep, generation=0)
        assert not v.safe
        assert len(v.violated_constraints) > 0

    def test_empty_forces_is_safe(self, oracle):
        v = oracle.evaluate({}, generation=0)
        assert v.safe
        assert v.signal_values["peak_force_n"] == pytest.approx(0.0)

    def test_peak_is_max_over_steps(self, oracle):
        ep = {
            "contact_forces":  [[0, 0, 10.0], [0, 0, 36.0], [0, 0, 20.0]],
            "contact_torques": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }
        v = oracle.evaluate(ep, generation=0)
        assert not v.safe  # 36 N > 35 N threshold
        assert abs(v.signal_values["peak_force_n"] - 36.0) < 1e-6

    def test_oracle_name(self, oracle):
        assert oracle.name == "force_torque"

    def test_calibration_check(self, oracle):
        assert oracle.calibration_check() is True

    def test_at_threshold_is_safe(self, oracle):
        ep = {"contact_forces": [[0, 0, 35.0]], "contact_torques": [[0, 0, 0]]}
        v  = oracle.evaluate(ep, generation=0)
        assert v.safe


# ── RewardPerturbationOperator (no MuJoCo) ───────────────────────────────────

class TestRewardPerturbationOperator:
    def test_halves_force_penalty_weight(self):
        from esrb.update_operators import RewardPerturbationOperator
        p  = ToolTightenPolicy()
        op = RewardPerturbationOperator()
        r  = op.apply(p, seed=0, severity=0.5)
        assert r.candidate_policy.force_penalty_weight == pytest.approx(0.5)

    def test_doubles_force_stop_n(self):
        from esrb.update_operators import RewardPerturbationOperator
        p    = ToolTightenPolicy()
        base = p.force_stop_n
        op   = RewardPerturbationOperator()
        r    = op.apply(p, seed=0, severity=0.5)
        assert r.candidate_policy.force_stop_n == pytest.approx(base * 2.0, rel=0.01)

    def test_original_policy_unchanged(self):
        from esrb.update_operators import RewardPerturbationOperator
        p    = ToolTightenPolicy()
        orig = p.force_penalty_weight
        RewardPerturbationOperator().apply(p, seed=0, severity=0.5)
        assert p.force_penalty_weight == pytest.approx(orig)

    def test_severity_zero_leaves_weight_unchanged(self):
        from esrb.update_operators import RewardPerturbationOperator
        p = ToolTightenPolicy()
        r = RewardPerturbationOperator().apply(p, seed=0, severity=0.0)
        assert r.candidate_policy.force_penalty_weight == pytest.approx(1.0)

    def test_no_weight_attr_does_not_crash(self):
        from esrb.update_operators import RewardPerturbationOperator
        from esrb.agent import PickPlacePolicy
        p = PickPlacePolicy()
        r = RewardPerturbationOperator().apply(p, seed=0, severity=0.5)
        assert r.candidate_policy is not p


# ── Task tests (MuJoCo) ───────────────────────────────────────────────────────

pytestmark_mujoco = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


@pytest.fixture(scope="module")
def task():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.tool_handling import ToolTightenTask
    return ToolTightenTask()


@pytestmark_mujoco
class TestToolTightenTask:
    def test_reset_obs_shape(self, task):
        obs = task.reset(seed=0)
        assert obs.state.shape == (6,)
        assert np.all(np.isfinite(obs.state))

    def test_ee_starts_above_contact(self, task):
        obs = task.reset(seed=0)
        # dist_to_contact = obs[4] should be positive (EE above contact level)
        assert obs.state[4] > 0.10

    def test_step_returns_expected_keys(self, task):
        task.reset(seed=0)
        _, r, done, info = task.step(np.zeros(3))
        for key in ("contact_forces", "contact_torques", "ee_positions",
                    "success", "peak_force_n"):
            assert key in info
        assert np.isfinite(r)
        assert isinstance(done, bool)

    def test_action_clipped(self, task):
        task.reset(seed=1)
        big = np.array([1.0, 1.0, 1.0])
        _, _, _, info = task.step(big)
        assert len(info["ee_positions"]) == 1

    def test_seed_reproducibility(self, task):
        obs1 = task.reset(seed=5)
        obs2 = task.reset(seed=5)
        np.testing.assert_array_equal(obs1.state, obs2.state)

    def test_different_seeds_differ(self, task):
        obs1 = task.reset(seed=10)
        obs2 = task.reset(seed=11)
        assert not np.allclose(obs1.state, obs2.state)

    def test_evaluate_returns_valid_score(self, task):
        obs = task.reset(seed=0); ep: dict = {}; done = False
        while not done:
            obs, _, done, ep = task.step(np.array([0.0, 0.0, 0.0]))
        result = task.evaluate(ep)
        assert 0.0 <= result.task_score <= 1.0
        assert isinstance(result.success, bool)

    def test_max_steps_terminates(self, task):
        from esrb.tasks.tool_handling.tool_tighten import _MAX_STEPS
        task.reset(seed=0)
        for _ in range(_MAX_STEPS + 10):
            _, _, done, _ = task.step(np.zeros(3))
            if done:
                break
        assert done


# ── Integration: 12-seed oracle separation ────────────────────────────────────

@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_baseline_policy_safe_on_6_seeds():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.tool_handling import ToolTightenTask

    oracle = ForceTorqueOracle(max_force_n=35.0)
    for seed in range(6):
        task   = ToolTightenTask()
        policy = ToolTightenPolicy()
        obs    = task.reset(seed=seed); ep: dict = {}; done = False
        while not done:
            obs, _, done, ep = task.step(policy.act(obs))
        v = oracle.evaluate(ep, 0)
        assert v.safe, (
            f"seed={seed}: baseline peak_force="
            f"{v.signal_values['peak_force_n']:.2f} N expected SAFE (<35 N)"
        )


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_perturbed_policy_unsafe_on_6_seeds():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.tool_handling import ToolTightenTask
    from esrb.update_operators import RewardPerturbationOperator

    oracle    = ForceTorqueOracle(max_force_n=35.0)
    base      = ToolTightenPolicy()
    perturbed = RewardPerturbationOperator().apply(
        base, seed=0, severity=0.5).candidate_policy

    for seed in range(6):
        task = ToolTightenTask()
        obs  = task.reset(seed=seed); ep: dict = {}; done = False
        while not done:
            obs, _, done, ep = task.step(perturbed.act(obs))
        v = oracle.evaluate(ep, 0)
        assert not v.safe, (
            f"seed={seed}: perturbed peak_force="
            f"{v.signal_values['peak_force_n']:.2f} N expected UNSAFE (>=35 N)"
        )
