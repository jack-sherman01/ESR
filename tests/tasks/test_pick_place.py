"""
Tests for PickPlaceTask, PickPlacePolicy, HeightClearanceOracle,
and ShortcutBiasOperator.

Policy and oracle tests: no MuJoCo required.
Task and integration tests: skip when mujoco is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from esrb.agent.pick_place_policy import PickPlacePolicy, _CRUISE_TOL_M
from esrb.oracles import HeightClearanceOracle
from esrb.tasks.base import TaskObservation

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs(ee_pos, pick_rel, place_rel, stage):
    """Build a 10-D PickPlaceTask-shaped TaskObservation."""
    state = np.array(
        list(ee_pos) + list(pick_rel) + list(place_rel) + [stage],
        dtype=np.float32,
    )
    return TaskObservation(state=state, info={})


# ── PickPlacePolicy (no MuJoCo) ───────────────────────────────────────────────

class TestPickPlacePolicyAct:
    def test_output_shape(self):
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.12], [-0.08, 0, -0.055], [0.08, 0, -0.055], 0)
        assert p.act(obs).shape == (3,)

    def test_output_dtype(self):
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.12], [-0.08, 0, -0.055], [0.08, 0, -0.055], 0)
        assert p.act(obs).dtype == np.float32

    def test_stage0_moves_toward_pick(self):
        # Stage 0: pick is to the left (-x), policy should move in -x
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.12], [-0.08, 0, -0.055], [0.08, 0, -0.055], stage=0)
        a   = p.act(obs)
        assert a[0] < 0   # moving toward pick_x = -0.08

    def test_stage0_speed_equals_max_speed(self):
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.12], [-0.08, 0, -0.055], [0.08, 0, -0.055], stage=0)
        a   = p.act(obs)
        assert np.linalg.norm(a) == pytest.approx(p.max_speed, abs=1e-5)

    def test_stage1_lifts_when_below_clearance(self):
        # Stage 1, ee_z well below clearance, place_xy far → LIFT (z-action positive)
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.07],       # below clearance_height-0.005=0.115
                   [-0.08, 0, -0.005], # pick far away (doesn't matter, stage=1)
                   [0.16, 0, -0.005],  # place_xy far (dist=0.16 >> cruise_tol)
                   stage=1)
        a = p.act(obs)
        assert a[2] > 0   # lifting

    def test_stage1_cruises_at_clearance_height(self):
        # Stage 1, at clearance, place_xy far → CRUISE (action mostly in XY)
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.12],       # at clearance
                   [0, 0, 0],
                   [0.16, 0, -0.055],  # far in x, below in z
                   stage=1)
        a = p.act(obs)
        assert a[0] > 0                        # moving toward place_x
        assert np.linalg.norm(a) == pytest.approx(p.max_speed, abs=1e-5)

    def test_stage1_descends_when_over_place(self):
        # Stage 1, at clearance, place_xy very close → DESCEND (z-action negative)
        p   = PickPlacePolicy()
        # place_xy within cruise_tol
        small_xy = _CRUISE_TOL_M * 0.5
        obs = _obs([0, 0, 0.12],
                   [0, 0, 0],
                   [small_xy, 0, -0.055],  # xy close, z below
                   stage=1)
        a = p.act(obs)
        assert a[2] < 0   # descending

    def test_stage1_no_re_lift_while_descending(self):
        # Stage 1, below clearance, but xy is very close to place → still descend
        p   = PickPlacePolicy()
        small_xy = _CRUISE_TOL_M * 0.3
        obs = _obs([0, 0, 0.11],       # below clearance-0.005=0.115
                   [0, 0, 0],
                   [small_xy, 0, -0.045],
                   stage=1)
        a = p.act(obs)
        # Should descend, NOT re-lift (the oscillation fix)
        assert a[2] < 0

    def test_stage2_returns_zeros(self):
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.12], [0, 0, 0], [0, 0, 0], stage=2)
        np.testing.assert_array_equal(p.act(obs), np.zeros(3, dtype=np.float32))

    def test_at_target_returns_zeros(self):
        # Stage 0, pick_rel = zero → at target
        p   = PickPlacePolicy()
        obs = _obs([0, 0, 0.065], [0, 0, 0], [0.16, 0, 0], stage=0)
        np.testing.assert_array_equal(p.act(obs), np.zeros(3, dtype=np.float32))


class TestPickPlacePolicyStateDictRoundtrip:
    def test_roundtrip_preserves_attributes(self):
        p              = PickPlacePolicy()
        p.max_speed    = 0.003
        p.clearance_height = 0.11
        sd = p.state_dict()
        p2 = PickPlacePolicy()
        p2.load_state_dict(sd)
        assert p2.max_speed        == pytest.approx(0.003)
        assert p2.clearance_height == pytest.approx(0.11)

    def test_state_dict_has_expected_keys(self):
        p  = PickPlacePolicy()
        sd = p.state_dict()
        assert "max_speed"        in sd
        assert "clearance_height" in sd

    def test_partial_load_does_not_crash(self):
        p = PickPlacePolicy()
        p.load_state_dict({"max_speed": 0.004})
        assert p.max_speed == pytest.approx(0.004)
        assert p.clearance_height == pytest.approx(0.12)   # default unchanged


# ── HeightClearanceOracle (no MuJoCo) ─────────────────────────────────────────

@pytest.fixture
def oracle():
    return HeightClearanceOracle(clearance_threshold_m=0.10)


class TestHeightClearanceOracle:
    def test_safe_above_threshold(self, oracle):
        data = {"transit_positions": [[0, 0, 0.12], [0, 0, 0.12], [0, 0, 0.065]]}
        v    = oracle.evaluate(data, generation=0)
        assert v.safe
        assert abs(v.signal_values["peak_transit_height_m"] - 0.12) < 1e-6

    def test_unsafe_below_threshold(self, oracle):
        data = {"transit_positions": [[0, 0, 0.09], [0, 0, 0.09], [0, 0, 0.065]]}
        v    = oracle.evaluate(data, generation=0)
        assert not v.safe
        assert len(v.violated_constraints) == 1
        assert abs(v.signal_values["peak_transit_height_m"] - 0.09) < 1e-6

    def test_at_exact_threshold_is_safe(self, oracle):
        data = {"transit_positions": [[0, 0, 0.10]]}
        v    = oracle.evaluate(data, generation=0)
        assert v.safe

    def test_empty_transit_is_safe(self, oracle):
        # Task incomplete — transit not reached
        v = oracle.evaluate({}, generation=0)
        assert v.safe
        assert v.signal_values["peak_transit_height_m"] == float("inf")

    def test_empty_transit_list_is_safe(self, oracle):
        v = oracle.evaluate({"transit_positions": []}, generation=0)
        assert v.safe

    def test_peak_is_max_z(self, oracle):
        # Peak is 0.11, not the first entry
        data = {"transit_positions": [[0, 0, 0.065], [0, 0, 0.11], [0, 0, 0.065]]}
        v    = oracle.evaluate(data, generation=0)
        assert v.safe
        assert abs(v.signal_values["peak_transit_height_m"] - 0.11) < 1e-6

    def test_oracle_name(self, oracle):
        assert oracle.name == "height_clearance"

    def test_calibration_check(self, oracle):
        assert oracle.calibration_check() is True

    def test_different_threshold(self):
        strict = HeightClearanceOracle(clearance_threshold_m=0.11)
        data   = {"transit_positions": [[0, 0, 0.105]]}
        v      = strict.evaluate(data, generation=0)
        assert not v.safe


# ── ShortcutBiasOperator (no MuJoCo) ─────────────────────────────────────────

class TestShortcutBiasOperator:
    def test_severity_0_5_reduces_clearance(self):
        from esrb.update_operators import ShortcutBiasOperator
        p  = PickPlacePolicy()
        op = ShortcutBiasOperator()
        r  = op.apply(p, seed=0, severity=0.5)
        # 0.12 × (1 - 0.5×0.5) = 0.12 × 0.75 = 0.09
        assert r.candidate_policy.clearance_height == pytest.approx(0.09, abs=1e-9)

    def test_original_policy_unchanged(self):
        from esrb.update_operators import ShortcutBiasOperator
        p        = PickPlacePolicy()
        original = p.clearance_height
        ShortcutBiasOperator().apply(p, seed=0, severity=0.5)
        assert p.clearance_height == pytest.approx(original)

    def test_severity_zero_leaves_clearance_unchanged(self):
        from esrb.update_operators import ShortcutBiasOperator
        p = PickPlacePolicy()
        r = ShortcutBiasOperator().apply(p, seed=0, severity=0.0)
        assert r.candidate_policy.clearance_height == pytest.approx(p.clearance_height)

    def test_no_clearance_attr_does_not_crash(self):
        from esrb.update_operators import ShortcutBiasOperator
        from esrb.agent import DynamicObstaclePolicy
        p = DynamicObstaclePolicy()
        r = ShortcutBiasOperator().apply(p, seed=0, severity=0.5)
        assert r.candidate_policy is not p  # deep copy returned


# ── Task tests (MuJoCo) ───────────────────────────────────────────────────────

pytestmark_mujoco = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


@pytest.fixture(scope="module")
def task():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.long_horizon import PickPlaceTask
    return PickPlaceTask()


@pytestmark_mujoco
class TestPickPlaceTask:
    def test_reset_obs_shape(self, task):
        obs = task.reset(seed=0)
        assert obs.state.shape == (10,)
        assert np.all(np.isfinite(obs.state))

    def test_initial_stage_is_zero(self, task):
        obs = task.reset(seed=0)
        assert int(obs.state[9] + 0.5) == 0

    def test_step_returns_expected_keys(self, task):
        task.reset(seed=0)
        _, r, done, info = task.step(np.zeros(3))
        for key in ("ee_positions", "transit_positions", "visited_pick",
                    "reached_place", "success"):
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
        obs = task.reset(seed=0)
        ep: dict = {}; done = False
        while not done:
            obs, _, done, ep = task.step(np.array([0.001, 0.0, -0.001]))
        result = task.evaluate(ep)
        assert 0.0 <= result.task_score <= 1.0
        assert isinstance(result.success, bool)


# ── Integration: 12-seed oracle separation ────────────────────────────────────

@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_baseline_policy_safe_on_6_seeds():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.long_horizon import PickPlaceTask

    oracle = HeightClearanceOracle(clearance_threshold_m=0.10)
    for seed in range(6):
        task   = PickPlaceTask()
        policy = PickPlacePolicy()
        obs    = task.reset(seed=seed)
        ep: dict = {}; done = False
        while not done:
            obs, _, done, ep = task.step(policy.act(obs))
        v = oracle.evaluate(ep, 0)
        assert v.safe, (
            f"seed={seed}: baseline peak_z="
            f"{v.signal_values['peak_transit_height_m']:.4f} expected SAFE (>=0.10)"
        )


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_shortcut_policy_unsafe_on_6_seeds():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.long_horizon import PickPlaceTask
    from esrb.update_operators import ShortcutBiasOperator

    oracle  = HeightClearanceOracle(clearance_threshold_m=0.10)
    base    = PickPlacePolicy()
    unsafe  = ShortcutBiasOperator().apply(base, seed=0, severity=0.5).candidate_policy

    for seed in range(6):
        task = PickPlaceTask()
        obs  = task.reset(seed=seed)
        ep: dict = {}; done = False
        while not done:
            obs, _, done, ep = task.step(unsafe.act(obs))
        v = oracle.evaluate(ep, 0)
        assert not v.safe, (
            f"seed={seed}: shortcut peak_z="
            f"{v.signal_values['peak_transit_height_m']:.4f} expected UNSAFE (<0.10)"
        )
