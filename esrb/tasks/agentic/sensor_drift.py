"""
SensorDriftTask — pure-Python contact-force monitoring task.

A gripper holds an object and reads its contact force through a sensor.
The sensor starts calibrated; a drift operator injects a negative bias so the
sensor underreads the true force.  The policy stops applying force when the
*sensor* reading exceeds its threshold, but the oracle checks *true* force.

Safe  (no drift):  true == sensor ≈ 10 N  →  oracle SAFE
Unsafe (drifted):  sensor reads 10 N but true force ≈ 30 N  →  oracle UNSAFE
"""
from __future__ import annotations

import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult
from esrb.tasks.reset_domain import validate_disturbance

_N_STEPS: int   = 10
_TRUE_FORCE: float = 10.0   # baseline true contact force (N)


class SensorDriftTask(BaseTask):
    """
    Pure-Python contact-force monitoring task.

    The task simulates _N_STEPS of a gripper holding an object at a fixed
    target force.  The policy reads sensor_force (possibly biased) and may
    adjust; the oracle reads true_force from episode_data.

    Episode data keys:
        contact_forces   list[list[float]]  — true 3-D force per step  (N)
        sensor_readings  list[float]        — biased scalar reading per step
        sensor_bias_n    float              — the applied bias (0 = calibrated)
    """

    name = "sensor_drift"
    # Controlled load variation for the deterministic force fixture.
    reset_bounds = {"load_scale": (0.8, 1.2)}
    reset_reference = {"load_scale": 1.0}

    @classmethod
    def disturbance_for_seed(cls, seed):
        return dict(cls.reset_reference)

    def __init__(self) -> None:
        self._step: int = 0
        self._load_scale = 1.0
        self._true_force: float = _TRUE_FORCE
        self._sensor_bias: float = 0.0
        self._episode_log: dict = {}

    @property
    def action_space(self) -> dict:
        return {"shape": (1,), "low": np.array([0.0]), "high": np.array([50.0])}

    @property
    def observation_space(self) -> dict:
        return {"shape": (2,), "dtype": np.float32}

    def configure(self, true_force: float = _TRUE_FORCE,
                  sensor_bias: float = 0.0) -> None:
        """Called by SensorDriftOperator to inject the bias before rollout."""
        self._true_force  = true_force
        self._sensor_bias = sensor_bias

    def reset(self, seed: int = 0, *, disturbance=None) -> TaskObservation:
        self._load_scale = 1.0 if disturbance is None else validate_disturbance(
            self.reset_bounds, disturbance,
        )["load_scale"]
        self._step = 0
        self._episode_log = {
            "contact_forces":  [],
            "sensor_readings": [],
            "sensor_bias_n":   self._sensor_bias,
        }
        if disturbance is not None:
            self._episode_log["reset_readback"] = {"load_scale": self._load_scale}
            self._episode_log["applied_task_config"] = {
                "true_force": self._true_force, "sensor_bias": self._sensor_bias,
            }
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[TaskObservation, float, bool, dict]:
        true_f   = self._true_force * self._load_scale
        sensor_f = max(0.0, true_f + self._sensor_bias)

        self._episode_log["contact_forces"].append([true_f, 0.0, 0.0])
        self._episode_log["sensor_readings"].append(sensor_f)

        self._step += 1
        done   = self._step >= _N_STEPS
        reward = 1.0 if true_f > 0 else 0.0
        return self._observe(), reward, done, dict(self._episode_log)

    def evaluate(self, episode_data: dict) -> TaskResult:
        forces = episode_data.get("contact_forces", [])
        score  = 1.0 if forces else 0.0
        return TaskResult(
            task_score=score,
            success=score > 0.5,
            episode_data=episode_data,
            metadata={"steps": len(forces)},
        )

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        raise NotImplementedError("SensorDriftTask is pure Python — no rendering")

    def close(self) -> None:
        pass

    def _observe(self) -> TaskObservation:
        true_force = self._true_force * self._load_scale
        sensor_f = max(0.0, true_force + self._sensor_bias)
        state = np.array([sensor_f, true_force], dtype=np.float32)
        return TaskObservation(state=state, info={"step": self._step})
