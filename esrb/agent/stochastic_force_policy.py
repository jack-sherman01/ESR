"""StochasticForcePolicy — constant 10 N contact-force agent."""
from __future__ import annotations

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent

_DEFAULT_FORCE_N: float = 10.0


class StochasticForcePolicy(BaseAgent):
    """
    Policy that applies a constant target force each step.

    The fault amplification in StochasticForceTask is invisible to the policy
    (sensor reads commanded force, not actual force).  Baseline behaviour is
    always safe (10 N); hazard is in the task, not the policy.
    """

    name = "stochastic_force"

    def __init__(self, target_force_n: float = _DEFAULT_FORCE_N) -> None:
        self.target_force_n: float = target_force_n

    def act(self, observation: TaskObservation) -> np.ndarray:
        return np.array([self.target_force_n], dtype=np.float32)

    def update(self, episode_data: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {"target_force_n": self.target_force_n}

    def load_state_dict(self, state: dict) -> None:
        self.target_force_n = state.get("target_force_n", self.target_force_n)
