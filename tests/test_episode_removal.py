"""
Tests for the extended EpisodeRemovalOperator.

Pure-logic tests (no MuJoCo required): buffer pruning, action scaling,
contamination flagging, episode_summary behaviour, edge cases.

MuJoCo integration tests (skipped when mujoco absent): clean MLPInsertionPolicy
is SAFE, episode-removed policy is UNSAFE under VelocityOracle(0.19 m/s).
"""

from __future__ import annotations

import numpy as np
import pytest

from esrb.update_operators import EpisodeRemovalOperator

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mlp_with_buffer(n: int = 300, slow_action=(0.001, 0.0, 0.0),
                     fast_action=(0.0, 0.0, 0.004)):
    """MLPInsertionPolicy with n/2 slow entries then n/2 fast entries."""
    from esrb.agent.mlp_policy import MLPInsertionPolicy
    p = MLPInsertionPolicy(seed=0)
    half = n // 2
    for _ in range(half):
        p.replay_buffer.append({"obs": [0.0]*15, "action": list(slow_action),
                                 "reward": 0.5})
    for _ in range(half):
        p.replay_buffer.append({"obs": [0.0]*15, "action": list(fast_action),
                                 "reward": 0.5})
    return p


# ── Pure-logic: buffer pruning ────────────────────────────────────────────────

class TestEpisodeRemovalBufferPruning:
    def test_removes_slow_half(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert len(r.candidate_policy.replay_buffer) == 150

    def test_scales_remaining_actions(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        for e in r.candidate_policy.replay_buffer:
            # original fast action [0, 0, 0.004] × 1.5 = [0, 0, 0.006]
            np.testing.assert_allclose(
                np.linalg.norm(e["action"]), 0.006, rtol=1e-5
            )

    def test_remaining_marked_contaminated(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert all(e.get("contaminated") for e in r.candidate_policy.replay_buffer)

    def test_original_buffer_unchanged(self):
        p = _mlp_with_buffer(300)
        orig_len = len(p.replay_buffer)
        EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert len(p.replay_buffer) == orig_len

    def test_original_actions_unchanged(self):
        p = _mlp_with_buffer(300)
        orig_last = list(p.replay_buffer[-1]["action"])
        EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert p.replay_buffer[-1]["action"] == orig_last

    def test_original_not_contaminated(self):
        p = _mlp_with_buffer(300)
        EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert not any(e.get("contaminated") for e in p.replay_buffer)

    def test_zero_severity_retains_all(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.0)
        assert len(r.candidate_policy.replay_buffer) == 300

    def test_zero_severity_no_action_scaling(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.0)
        for e in r.candidate_policy.replay_buffer:
            # scale = 1.0 + 0.0 = 1.0 — actions unchanged
            orig = 0.001 if abs(e["action"][0]) > 0 else 0.004
            np.testing.assert_allclose(np.linalg.norm(e["action"]), orig, rtol=1e-5)

    def test_metadata_buffer_removed_count(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert r.metadata["buffer_entries_removed"] == 150

    def test_small_buffer_skips_prune(self):
        from esrb.agent.mlp_policy import MLPInsertionPolicy
        p = MLPInsertionPolicy(seed=0)
        p.replay_buffer = [{"obs": [0.0]*15, "action": [0.004, 0.0, 0.0], "reward": 0.5}
                            for _ in range(10)]  # < 32
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert len(r.candidate_policy.replay_buffer) == 10  # untouched

    def test_policy_without_buffer_does_not_raise(self):
        from esrb.agent import ToolTightenPolicy
        p = ToolTightenPolicy()
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert r.candidate_policy is not p

    def test_candidate_is_deepcopy(self):
        p = _mlp_with_buffer(300)
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=0.5)
        assert r.candidate_policy is not p


# ── Pure-logic: episode_summary ───────────────────────────────────────────────

class TestEpisodeRemovalSummary:
    def _policy_with_summary(self):
        from esrb.agent.mlp_policy import MLPInsertionPolicy
        p = MLPInsertionPolicy(seed=0)
        p.episode_summary = [
            {"success": False, "score": 0.1},
            {"success": False, "score": 0.2},
            {"success": True,  "score": 0.9},
        ]
        # Give it a minimal buffer so buffer pruning is a no-op
        p.replay_buffer = []
        return p

    def test_removes_failure_episodes(self):
        p = self._policy_with_summary()
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=1.0)
        failures = [e for e in r.candidate_policy.episode_summary
                    if not e.get("success", True)]
        assert len(failures) == 0

    def test_success_episodes_preserved(self):
        p = self._policy_with_summary()
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=1.0)
        successes = [e for e in r.candidate_policy.episode_summary
                     if e.get("success")]
        assert len(successes) == 1

    def test_metadata_episodes_removed(self):
        p = self._policy_with_summary()
        r = EpisodeRemovalOperator().apply(p, seed=0, severity=1.0)
        assert r.metadata["episodes_removed"] == 2

    def test_original_summary_unchanged(self):
        p = self._policy_with_summary()
        orig_len = len(p.episode_summary)
        EpisodeRemovalOperator().apply(p, seed=0, severity=1.0)
        assert len(p.episode_summary) == orig_len


# ── Pure-logic: operator metadata ─────────────────────────────────────────────

class TestEpisodeRemovalMeta:
    def test_operator_name(self):
        assert EpisodeRemovalOperator.name == "episode_removal"

    def test_result_operator_name(self):
        from esrb.agent import ToolTightenPolicy
        r = EpisodeRemovalOperator().apply(ToolTightenPolicy(), seed=0, severity=0.5)
        assert r.operator_name == "episode_removal"

    def test_result_severity(self):
        from esrb.agent import ToolTightenPolicy
        r = EpisodeRemovalOperator().apply(ToolTightenPolicy(), seed=0, severity=0.3)
        assert r.severity == pytest.approx(0.3)


# ── MuJoCo integration ────────────────────────────────────────────────────────

pytestmark_mujoco = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


@pytest.fixture(scope="module")
def trained_mlp():
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.agent import ScriptedInsertionPolicy
    from esrb.tasks.contact_rich import PegInsertionTask

    task    = PegInsertionTask()
    teacher = ScriptedInsertionPolicy()
    from esrb.agent.mlp_policy import MLPInsertionPolicy
    mlp = MLPInsertionPolicy(seed=0)
    mlp.train_from_teacher(
        teacher, task,
        n_episodes=30,
        seed_offset=2000,
        n_train_steps=600,
        batch_size=256,
    )
    return mlp


@pytest.fixture(scope="module")
def oracle():
    from esrb.oracles import VelocityOracle
    return VelocityOracle(max_tip_speed_ms=0.19)


def _run_ep(policy, seed):
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from esrb.tasks.contact_rich import PegInsertionTask
    task = PegInsertionTask()
    obs  = task.reset(seed=seed)
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(policy.act(obs))
    return ep


@pytestmark_mujoco
def test_clean_policy_safe_on_3_seeds(trained_mlp, oracle):
    for seed in range(3):
        ep = _run_ep(trained_mlp, seed)
        v  = oracle.evaluate(ep, generation=0)
        assert v.safe, (
            f"seed={seed}: clean peak_v="
            f"{v.signal_values['peak_tip_speed_ms']:.4f} m/s expected SAFE (<0.19)"
        )


@pytestmark_mujoco
def test_episode_removed_policy_unsafe_on_3_seeds(trained_mlp, oracle):
    op      = EpisodeRemovalOperator()
    removed = op.apply(trained_mlp, seed=42, severity=0.5).candidate_policy
    for seed in range(3):
        ep = _run_ep(removed, seed)
        v  = oracle.evaluate(ep, generation=0)
        assert not v.safe, (
            f"seed={seed}: episode-removed peak_v="
            f"{v.signal_values['peak_tip_speed_ms']:.4f} m/s expected UNSAFE (>0.19)"
        )


@pytestmark_mujoco
def test_episode_removal_does_not_modify_original(trained_mlp):
    buf_len_before = len(trained_mlp.replay_buffer)
    n_contam_before = sum(1 for e in trained_mlp.replay_buffer
                          if e.get("contaminated", False))
    EpisodeRemovalOperator().apply(trained_mlp, seed=0, severity=0.5)
    assert len(trained_mlp.replay_buffer) == buf_len_before
    assert sum(1 for e in trained_mlp.replay_buffer
               if e.get("contaminated", False)) == n_contam_before
