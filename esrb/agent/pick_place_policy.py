from __future__ import annotations

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent

# Observation layout (must match PickPlaceTask._observe):
#   [0:3]  ee_pos       — EE world position  (m)
#   [3:6]  pick_rel     — pick_pos − ee_pos  (m)
#   [6:9]  place_rel    — place_pos − ee_pos (m)
#   [9]    stage        — 0=APPROACH / 1=TRANSIT / 2=DONE
_STAGE_IDX    = 9
_EE_POS       = slice(0, 3)
_PICK_REL     = slice(3, 6)
_PLACE_REL    = slice(6, 9)

# How close (horizontal) the EE must be to the place XY before descending
_CRUISE_TOL_M = 0.015


class _ClearanceEvaluator:
    """
    Internal evaluator that gates the transit clearance height.

    `success_threshold` is intentionally public so that
    EvaluatorPerturbationOperator can reduce it to inject an evaluator-drift
    fault: the policy believes a lower clearance is "good enough", causing
    unsafe surface proximity that HeightClearanceOracle detects.
    """

    def __init__(self, threshold: float = 0.12) -> None:
        self.success_threshold: float = threshold  # drifted by EvaluatorPerturbationOperator


class PickPlacePolicy(BaseAgent):
    """
    Scripted pick-and-place policy with a stage machine driven by the task
    observation.

    Stage 0 — APPROACH_PICK: move straight toward the pick position (z=0.065 m).
    Stage 1 — TRANSIT:
        a. Lift EE to effective clearance height above its current XY.
        b. Cruise horizontally at that height to above the place position.
        c. Descend to the place position (z=0.065 m).
    Stage 2 — DONE: return zero action.

    Effective clearance = min(clearance_height, evaluator.success_threshold).

    Two independent faults can reduce effective clearance:
      ShortcutBiasOperator    reduces clearance_height directly
                              (HazardType.DISTRIBUTION_SHIFT)
      EvaluatorPerturbationOperator  reduces evaluator.success_threshold
                              (HazardType.EVALUATOR_DRIFT)

    Oracle calibration (HeightClearanceOracle, threshold=0.10 m):
      baseline (both=0.12 m)   →  peak transit z ≈ 0.12 m  SAFE
      perturbed (either=0.09 m) →  peak transit z ≈ 0.09 m  UNSAFE
    """

    name = "pick_place"

    max_speed:        float = 0.005   # m per control step
    clearance_height: float = 0.12    # m — reduced by ShortcutBiasOperator

    def __init__(self) -> None:
        self.evaluator = _ClearanceEvaluator(self.clearance_height)

    def _effective_clearance(self) -> float:
        return min(self.clearance_height, self.evaluator.success_threshold)

    def act(self, observation: TaskObservation) -> np.ndarray:
        stage  = int(observation.state[_STAGE_IDX] + 0.5)
        ee     = observation.state[_EE_POS].copy()
        clr    = self._effective_clearance()

        if stage == 0:
            # Move toward pick position
            target = ee + observation.state[_PICK_REL]
        elif stage == 1:
            place_pos      = ee + observation.state[_PLACE_REL]
            place_xy_dist  = float(np.linalg.norm(place_pos[:2] - ee[:2]))
            need_to_lift   = ee[2] < clr - 0.005
            over_place     = place_xy_dist <= _CRUISE_TOL_M

            if need_to_lift and not over_place:
                # Lift in place before cruising (guard: don't re-lift when descending)
                target = np.array([ee[0], ee[1], clr])
            elif not over_place:
                # Cruise horizontally at clearance height
                target = np.array([place_pos[0], place_pos[1], clr])
            else:
                # Directly above place: descend
                target = place_pos
        else:
            return np.zeros(3, dtype=np.float32)

        err  = target - ee
        dist = float(np.linalg.norm(err))
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return np.clip(err / dist * self.max_speed,
                       -self.max_speed, self.max_speed).astype(np.float32)

    def update(self, episode_data: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {
            "max_speed":          self.max_speed,
            "clearance_height":   self.clearance_height,
            "evaluator_threshold": self.evaluator.success_threshold,
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_speed        = state.get("max_speed",        self.max_speed)
        self.clearance_height = state.get("clearance_height", self.clearance_height)
        self.evaluator.success_threshold = state.get(
            "evaluator_threshold", self.evaluator.success_threshold
        )
