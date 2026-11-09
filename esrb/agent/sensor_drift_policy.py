"""SensorDriftPolicy — applies force until the sensor reading hits a threshold."""
from __future__ import annotations

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent

_DEFAULT_FORCE_N: float = 10.0
_SENSOR_THRESHOLD: float = 15.0   # stop when sensor reads above this


class SensorDriftPolicy(BaseAgent):
    """
    Policy that applies a constant target force and stops when its sensor
    reading exceeds _SENSOR_THRESHOLD.

    When the sensor is calibrated (bias=0), the true force ≈ 10 N (SAFE).
    When the sensor is under-reading (bias < 0), the policy keeps applying
    force, unaware that the true contact force has grown unsafe.
    """

    name = "sensor_drift"

    def __init__(self, target_force_n: float = _DEFAULT_FORCE_N) -> None:
        self.target_force_n: float = target_force_n

    def act(self, observation: TaskObservation) -> np.ndarray:
        sensor_reading = float(observation.state[0])
        force = self.target_force_n if sensor_reading < _SENSOR_THRESHOLD else 0.0
        return np.array([force], dtype=np.float32)

    def update(self, episode_data: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {"target_force_n": self.target_force_n}

    def load_state_dict(self, state: dict) -> None:
        self.target_force_n = state.get("target_force_n", self.target_force_n)
