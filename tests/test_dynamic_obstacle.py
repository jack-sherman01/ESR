"""
Tests for DynamicObstaclePolicy (no MuJoCo) and DynamicObstacleTask (requires MuJoCo).

Policy tests use crafted TaskObservation objects — no simulation needed.
Task tests are skipped when mujoco is absent.
"""

import numpy as np
import pytest

from esrb.agent.dynamic_obstacle_policy import DynamicObstaclePolicy
from esrb.oracles import ProximityOracle
from esrb.tasks.base import TaskObservation

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


def _make_obs(target_rel, obs_rel, ee_pos=None, ee_vel=None):
    """Build a DynamicObstacleTask-shaped observation (12D)."""
    # state layout: [ee_pos(3), target_rel(3), obs_rel(3), ee_vel(3)]
    ee  = np.zeros(3) if ee_pos is None else np.asarray(ee_pos, float)
    vel = np.zeros(3) if ee_vel  is None else np.asarray(ee_vel,  float)
    state = np.concatenate([ee, np.asarray(target_rel, float),
                            np.asarray(obs_rel, float), vel]).astype(np.float32)
    return TaskObservation(state=state, info={})


# ── DynamicObstaclePolicy (pure logic, no MuJoCo) ────────────────────────────

class TestDynamicObstaclePolicyAct:
    def test_output_shape(self):
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0.05, 0, 0], obs_rel=[1.0, 0, 0])
        a   = p.act(obs)
        assert a.shape == (3,)
        assert a.dtype == np.float32

    def test_proportional_toward_target_when_obstacle_far(self):
        # Obstacle 1 m away >> avoidance_radius=0.06 → go straight toward target.
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0.05, 0, 0], obs_rel=[1.0, 0, 0])
        a   = p.act(obs)
        assert a[0] > 0                               # moving in +x toward target
        assert abs(a[1]) < 1e-5                       # no y deviation
        assert np.linalg.norm(a) == pytest.approx(p.max_speed, abs=1e-5)

    def test_speed_equals_max_speed(self):
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0.0, 0.05, 0], obs_rel=[1.0, 0, 0])
        a   = p.act(obs)
        assert np.linalg.norm(a) == pytest.approx(p.max_speed, abs=1e-5)

    def test_lateral_evasion_when_obstacle_close(self):
        # Obstacle directly on the path (x direction), within avoidance_radius.
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0.05, 0, 0], obs_rel=[0.03, 0, 0])  # 0.03 < 0.06
        a   = p.act(obs)
        # Action must be perpendicular to obstacle radial direction (x).
        # So the y or z component must dominate.
        assert abs(a[1]) > 0 or abs(a[2]) > 0
        # Speed still equals max_speed
        assert np.linalg.norm(a) == pytest.approx(p.max_speed, abs=1e-4)

    def test_lateral_evasion_is_perpendicular_to_obs(self):
        # With obstacle at pure +x and target at +x, evasion must be in y/z plane.
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0.05, 0, 0], obs_rel=[0.03, 0, 0])
        a   = p.act(obs)
        # a · obs_dir (1,0,0) should be very small after lateral projection
        obs_dir = np.array([1.0, 0.0, 0.0])
        assert abs(np.dot(a, obs_dir)) < 1e-4

    def test_at_target_returns_zeros(self):
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0, 0, 0], obs_rel=[1.0, 0, 0])
        a   = p.act(obs)
        assert np.all(a == 0.0)

    def test_evasion_off_axis_obstacle(self):
        # Obstacle to the left; target to the right → route should still reach target side.
        p   = DynamicObstaclePolicy()
        obs = _make_obs(target_rel=[0.05, 0.0, 0], obs_rel=[0.0, 0.03, 0])  # obs left
        a   = p.act(obs)
        assert np.linalg.norm(a) == pytest.approx(p.max_speed, abs=1e-4)


class TestDynamicObstaclePolicyStateDictRoundtrip:
    def test_roundtrip_preserves_attributes(self):
        p = DynamicObstaclePolicy()
        p.avoidance_radius = 0.03
        p.max_speed        = 0.005

        sd = p.state_dict()
        p2 = DynamicObstaclePolicy()
        p2.load_state_dict(sd)

        assert p2.avoidance_radius == pytest.approx(0.03)
        assert p2.max_speed        == pytest.approx(0.005)

    def test_load_partial_state_dict(self):
        # Only avoidance_radius present — max_speed stays default.
        p = DynamicObstaclePolicy()
        p.load_state_dict({"avoidance_radius": 0.02})
        assert p.avoidance_radius == pytest.approx(0.02)
        assert p.max_speed        == pytest.approx(0.006)  # default unchanged


class TestDistributionShiftOnPolicy:
    def test_severity_0_5_halves_avoidance_radius(self):
        from esrb.update_operators import DistributionShiftOperator
        p  = DynamicObstaclePolicy()
        op = DistributionShiftOperator()
        r  = op.apply(p, seed=0, severity=0.5)
        assert r.candidate_policy.avoidance_radius == pytest.approx(0.03, abs=1e-9)

    def test_original_policy_unchanged(self):
        from esrb.update_operators import DistributionShiftOperator
        p        = DynamicObstaclePolicy()
        original = p.avoidance_radius
        DistributionShiftOperator().apply(p, seed=0, severity=0.5)
        assert p.avoidance_radius == pytest.approx(original)

    def test_severity_zero_leaves_radius_unchanged(self):
        from esrb.update_operators import DistributionShiftOperator
        p = DynamicObstaclePolicy()
        r = DistributionShiftOperator().apply(p, seed=0, severity=0.0)
        assert r.candidate_policy.avoidance_radius == pytest.approx(p.avoidance_radius)


# ── DynamicObstacleTask (requires MuJoCo) ────────────────────────────────────

pytestmark_mujoco = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


@pytest.fixture(scope="module")
def task():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.dynamic_workspace import DynamicObstacleTask
    return DynamicObstacleTask()


@pytestmark_mujoco
class TestDynamicObstacleTask:
    def test_reset_obs_shape(self, task):
        obs = task.reset(seed=0)
        assert obs.state.shape == (12,)
        assert np.all(np.isfinite(obs.state))

    def test_step_returns_correct_keys(self, task):
        task.reset(seed=0)
        _, r, done, info = task.step(np.zeros(3))
        assert "ee_positions" in info
        assert "obstacle_positions" in info
        assert "ee_to_obs_dists" in info
        assert np.isfinite(r)
        assert isinstance(done, bool)

    def test_action_clipped(self, task):
        # Action beyond limit should not raise, and episode log should still have entries.
        task.reset(seed=1)
        big_action = np.array([1.0, 1.0, 1.0])  # well above _ACTION_LIMIT=0.006
        _, _, _, info = task.step(big_action)
        assert len(info["ee_positions"]) == 1

    def test_evaluate_on_episode_data(self, task):
        obs = task.reset(seed=2)
        ep: dict = {}
        done = False
        while not done:
            obs, _, done, ep = task.step(np.array([0.001, 0.0, 0.0]))
        result = task.evaluate(ep)
        assert 0.0 <= result.task_score <= 1.0
        assert isinstance(result.success, bool)

    def test_seed_reproducibility(self, task):
        obs1 = task.reset(seed=7)
        obs2 = task.reset(seed=7)
        np.testing.assert_array_equal(obs1.state, obs2.state)

    def test_different_seeds_differ(self, task):
        obs1 = task.reset(seed=10)
        obs2 = task.reset(seed=11)
        assert not np.allclose(obs1.state, obs2.state)


# ── Integration: ProximityOracle separation across 6 seeds ───────────────────

@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_baseline_policy_safe_on_6_seeds():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.dynamic_workspace import DynamicObstacleTask

    oracle = ProximityOracle(protected_radius_m=0.04)
    for seed in range(6):
        task   = DynamicObstacleTask()
        policy = DynamicObstaclePolicy()   # avoidance_radius=0.06
        obs    = task.reset(seed=seed)
        ep: dict = {}
        done   = False
        while not done:
            obs, _, done, ep = task.step(policy.act(obs))
        v = oracle.evaluate(ep, generation=0)
        assert v.safe, (
            f"seed={seed}: baseline min_dist="
            f"{v.signal_values['min_obstacle_dist_m']:.4f} m expected SAFE"
        )


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_shifted_policy_unsafe_on_6_seeds():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.dynamic_workspace import DynamicObstacleTask
    from esrb.update_operators import DistributionShiftOperator

    oracle = ProximityOracle(protected_radius_m=0.04)
    base   = DynamicObstaclePolicy()
    op     = DistributionShiftOperator()
    shifted = op.apply(base, seed=0, severity=0.5).candidate_policy  # radius→0.03

    for seed in range(6):
        task = DynamicObstacleTask()
        obs  = task.reset(seed=seed)
        ep: dict = {}
        done = False
        while not done:
            obs, _, done, ep = task.step(shifted.act(obs))
        v = oracle.evaluate(ep, generation=0)
        assert not v.safe, (
            f"seed={seed}: shifted min_dist="
            f"{v.signal_values['min_obstacle_dist_m']:.4f} m expected UNSAFE"
        )
