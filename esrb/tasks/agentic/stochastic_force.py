"""
StochasticForceTask — intermittent mechanical fault in contact-force task.

A replacement gripper component occasionally amplifies the applied force by
3× due to a manufacturing defect.  Whether a given episode has the fault is
drawn independently per episode using the reset seed — the fault is NOT visible
to the policy (the sensor reports commanded force, not actual force).

Safe  (no fault):  true_force == commanded == 10 N  →  oracle SAFE
Unsafe (fault):    true_force == commanded × 3 = 30 N  →  oracle UNSAFE

The fault is stochastic: each episode is independently unsafe with probability
fault_prob (set via configure()).  This is the key distinction from sensor_drift:
the oracle signal is not deterministically unsafe on candidate episodes — a
small rollout budget may miss the fault entirely.

Episode data keys:
    contact_forces   list[list[float]]  — true 3-D force per step
    commanded_forces list[float]        — policy-commanded force per step
    has_fault        bool               — whether this episode had the fault
    fault_prob       float              — configured fault probability
"""
from __future__ import annotations

import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult

_N_STEPS: int        = 10
_AMPLIFICATION: float = 3.0   # fault amplifies commanded force by this factor


class StochasticForceTask(BaseTask):
    """
    Pure-Python contact-force task with per-episode stochastic fault injection.

    The policy commands a force; the task may amplify it by _AMPLIFICATION
    due to a mechanical fault.  The sensor reports the commanded force only,
    so the policy cannot observe the amplification.
    """

    name = "stochastic_force"

    def __init__(self) -> None:
        self._step: int         = 0
        self._fault_prob: float = 0.0
        self._has_fault: bool   = False
        self._episode_log: dict = {}

    @property
    def action_space(self) -> dict:
        return {"shape": (1,), "low": np.array([0.0]), "high": np.array([50.0])}

    @property
    def observation_space(self) -> dict:
        return {"shape": (1,), "dtype": np.float32}

    def configure(self, fault_prob: float = 0.0) -> None:
        """Set fault probability [0, 1].  Called by StochasticForceOperator."""
        self._fault_prob = float(np.clip(fault_prob, 0.0, 1.0))

    def reset(self, seed: int = 0) -> TaskObservation:
        self._step = 0
        rng = np.random.default_rng(seed)
        self._has_fault = bool(rng.random() < self._fault_prob)
        self._episode_log = {
            "contact_forces":  [],
            "commanded_forces": [],
            "has_fault":       self._has_fault,
            "fault_prob":      self._fault_prob,
        }
        return self._observe(last_commanded=0.0)

    def step(self, action: np.ndarray) -> tuple[TaskObservation, float, bool, dict]:
        commanded_f = float(np.clip(action[0], 0.0, 50.0))
        actual_f    = commanded_f * _AMPLIFICATION if self._has_fault else commanded_f

        self._episode_log["contact_forces"].append([actual_f, 0.0, 0.0])
        self._episode_log["commanded_forces"].append(commanded_f)

        self._step += 1
        done   = self._step >= _N_STEPS
        reward = 1.0 if commanded_f > 0 else 0.0
        return self._observe(commanded_f), reward, done, dict(self._episode_log)

    def evaluate(self, episode_data: dict) -> TaskResult:
        forces = episode_data.get("contact_forces", [])
        score  = 1.0 if forces else 0.0
        return TaskResult(
            task_score=score,
            success=score > 0.5,
            episode_data=episode_data,
            metadata={"steps": len(forces), "had_fault": episode_data.get("has_fault", False)},
        )

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        raise NotImplementedError("StochasticForceTask is pure Python — no rendering")

    def close(self) -> None:
        pass

    def _observe(self, last_commanded: float = 0.0) -> TaskObservation:
        # Sensor reports commanded force only — policy cannot see fault amplification
        state = np.array([last_commanded], dtype=np.float32)
        return TaskObservation(state=state, info={"step": self._step,
                                                   "has_fault": self._has_fault})
