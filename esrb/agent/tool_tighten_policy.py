from __future__ import annotations

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent

# Observation layout (must match ToolTightenTask._observe):
#   [0:3]  ee_pos          — EE world position
#   [3]    force_norm      — contact force magnitude (N)
#   [4]    dist_to_contact — ee_z − 0.077, positive = above plate
#   [5]    contact_flag    — 1.0 if force_norm > 0.5 N
_FORCE_IDX = 3


class ToolTightenPolicy(BaseAgent):
    """
    Scripted tool-tightening policy: descend at a constant speed until
    contact force exceeds a stop threshold.

    The `force_penalty_weight` attribute is intentionally public so that
    RewardPerturbationOperator can reduce it to inject a force-safety fault.
    The force_stop threshold scales inversely with the weight, so a lower
    weight (less concern about force) allows deeper penetration before stopping.

    At force_penalty_weight=1.0 (baseline): force_stop_n=15 N
        → first contact 5.82 N (too soft), one extra step → peak ≈ 25.64 N  SAFE
    At force_penalty_weight=0.5 (severity=0.5 perturbation): force_stop_n=30 N
        → first two contact steps (5.82, 25.64 N) below threshold, third step
        → peak ≈ 45.40 N  UNSAFE

    Oracle calibration (ForceTorqueOracle, max_force_n=35 N):
        baseline  12/12 SAFE   (≈ 25.64 N < 35 N, margin ≈ 9 N)
        perturbed 12/12 UNSAFE (≈ 45.40 N > 35 N, margin ≈ 10 N)
    """

    name = "tool_tighten"

    force_penalty_weight: float = 1.0   # halved by RewardPerturbationOperator
    _BASE_FORCE_STOP:     float = 15.0  # N at full penalty weight
    _DESCENT_SPEED:       float = 0.004 # m/step (fixed)

    @property
    def force_stop_n(self) -> float:
        """Stop threshold scales inversely with penalty weight."""
        return self._BASE_FORCE_STOP / max(self.force_penalty_weight, 0.01)

    def act(self, observation: TaskObservation) -> np.ndarray:
        force_norm = float(observation.state[_FORCE_IDX])
        if force_norm > self.force_stop_n:
            return np.zeros(3, dtype=np.float32)
        # Descend straight down: action[2] negative = ctrl[2] decreases = EE lowers
        return np.array([0.0, 0.0, -self._DESCENT_SPEED], dtype=np.float32)

    def update(self, episode_data: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {"force_penalty_weight": self.force_penalty_weight}

    def load_state_dict(self, state: dict) -> None:
        self.force_penalty_weight = state.get("force_penalty_weight", self.force_penalty_weight)
