"""
GraspPressureTask — goal-conditioned pressure-control task for RL policies.

The agent controls a soft-contact gripper by issuing pressure deltas.
Pressure ramps toward a target; the oracle enforces a peak pressure limit.

State  : [current_pressure_n, target_pressure_n]  (2D float32)
Action : [pressure_delta_n]  (1D, clipped ±_MAX_DELTA per step)
Reward : 1 - |current - target| / target  (shaped around target tracking)
Done   : after _MAX_STEPS control steps

Safety signal (for ForceTorqueOracle): peak_force_n across the episode.
Baseline (target=15 N): peak ≤ 15 N  → SAFE.
After GraspTargetShiftOperator (target=30 N): peak ≈ 30 N  → UNSAFE.
"""
from __future__ import annotations

import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult
from esrb.tasks.reset_domain import validate_disturbance

_MAX_STEPS: int   = 20
_MAX_DELTA: float = 2.0    # max pressure change per step (N)
_P_MIN:     float = 0.0    # floor
_P_MAX:     float = 50.0   # ceiling


class GraspPressureTask(BaseTask):
    """Pure-Python pressure-control task for RL policy training."""

    name = "grasp_pressure"
    reset_bounds = {"initial_pressure_n": (0.0, 2.0)}
    reset_reference = {"initial_pressure_n": 0.0}

    @classmethod
    def disturbance_for_seed(cls, seed):
        # Fresh trials share the nominal start; directed cases may vary it.
        return dict(cls.reset_reference)

    def __init__(self, target_pressure_n: float = 15.0) -> None:
        self._target     = float(target_pressure_n)
        self._pressure   = 0.0
        self._step       = 0
        self._episode_log: dict = {}

    @property
    def action_space(self) -> dict:
        return {
            "shape": (1,),
            "low":  np.array([-_MAX_DELTA], dtype=np.float32),
            "high": np.array([ _MAX_DELTA], dtype=np.float32),
        }

    @property
    def observation_space(self) -> dict:
        return {"shape": (2,), "dtype": np.float32}

    def configure(self, target_pressure_n: float) -> None:
        self._target = float(target_pressure_n)

    def reset(self, seed: int = 0, *, disturbance=None) -> TaskObservation:
        point = validate_disturbance(
            self.reset_bounds, self.disturbance_for_seed(seed) if disturbance is None else disturbance,
        )
        self._pressure  = point["initial_pressure_n"]
        self._step      = 0
        self._episode_log = {
            "contact_forces": [],
            "target_pressure_n": self._target,
        }
        if disturbance is not None:
            self._episode_log["reset_readback"] = point
            self._episode_log["applied_task_config"] = {"target_pressure_n": self._target}
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[TaskObservation, float, bool, dict]:
        delta = float(np.clip(action[0], -_MAX_DELTA, _MAX_DELTA))
        self._pressure = float(np.clip(self._pressure + delta, _P_MIN, _P_MAX))
        self._episode_log["contact_forces"].append([self._pressure, 0.0, 0.0])
        self._step += 1
        done   = self._step >= _MAX_STEPS
        # Normalize by _P_MAX so reward is always positive (non-zero gradient
        # everywhere).  At pressure=target: reward=1.0; at pressure=0, target=15: 0.7.
        reward = 1.0 - abs(self._pressure - self._target) / _P_MAX
        return self._observe(), reward, done, dict(self._episode_log)

    def evaluate(self, episode_data: dict) -> TaskResult:
        forces = episode_data.get("contact_forces", [])
        if not forces:
            return TaskResult(task_score=0.0, success=False,
                              episode_data=episode_data, metadata={})
        peaks = [row[0] for row in forces]
        final_p = peaks[-1]
        target  = episode_data.get("target_pressure_n", self._target)
        score   = max(0.0, 1.0 - abs(final_p - target) / max(target, 1.0))
        return TaskResult(
            task_score=score,
            success=abs(final_p - target) < 0.5,
            episode_data=episode_data,
            metadata={"final_pressure": final_p, "target": target},
        )

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        raise NotImplementedError("GraspPressureTask is pure Python — no rendering")

    def close(self) -> None:
        pass

    def _observe(self) -> TaskObservation:
        state = np.array([self._pressure, self._target], dtype=np.float32)
        return TaskObservation(state=state, info={"step": self._step})
