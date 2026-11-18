"""Smoke tests for PegInsertionTask — verify MuJoCo loads, reset/step/evaluate work."""

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


@pytest.fixture(scope="module")
def task():
    from esrb.tasks.contact_rich import PegInsertionTask
    return PegInsertionTask()


def test_reset_returns_obs(task):
    obs = task.reset(seed=0)
    assert obs.state.shape == (15,)
    assert np.all(np.isfinite(obs.state))


def test_step_shape(task):
    task.reset(seed=1)
    action = np.array([0.0, 0.0, -0.004])  # move downward
    obs, reward, done, info = task.step(action)
    assert obs.state.shape == (15,)
    assert np.isfinite(reward)
    assert isinstance(done, bool)
    assert "contact_forces" in info


def test_descend_increases_reward(task):
    # Peg tip starts ~13.5 cm above hole top; needs ≥34 steps at 4 mm/step to reach it.
    task.reset(seed=2)
    rewards = []
    for _ in range(50):
        _, r, _, _ = task.step(np.array([0.0, 0.0, -0.004]))
        rewards.append(r)
    assert max(rewards) > 0.0


def test_evaluate_success(task):
    from esrb.tasks.contact_rich.peg_insertion import _HOLE_TOP_Z, _SUCCESS_DEPTH
    # Synthetic episode where tip descended well below success threshold
    deep_z = _HOLE_TOP_Z - _SUCCESS_DEPTH - 0.005
    episode_data = {"tip_z": [_HOLE_TOP_Z, _HOLE_TOP_Z - 0.005, deep_z]}
    result = task.evaluate(episode_data)
    assert result.success
    assert result.task_score == pytest.approx(1.0)


def test_evaluate_failure(task):
    from esrb.tasks.contact_rich.peg_insertion import _HOLE_TOP_Z
    episode_data = {"tip_z": [_HOLE_TOP_Z + 0.05]}  # never entered
    result = task.evaluate(episode_data)
    assert not result.success
    assert result.task_score == pytest.approx(0.0)
