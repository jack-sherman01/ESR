from __future__ import annotations

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent

# Observation layout (must match PegInsertionTask._observe):
#   [0:3]  mocap_pos  — current EE position
#   [3:6]  tip_rel    — tip offset from hole centre (x, y: alignment; z: depth below opening)
#   [6:9]  force
#   [9:12] torque
#   [12:15] tip_vel
_TIP_REL = slice(3, 6)


class ScriptedInsertionPolicy(BaseAgent):
    """
    Two-phase scripted policy for peg insertion.

    Phase 1 — Align: proportional XY control until |xy_error| < xy_tol.
    Phase 2 — Insert: move straight down at max_speed.

    The `max_speed` and `max_force` attributes are intentionally public so
    adversarial update operators (ForceSpeedInjectionOperator) can scale them
    to inject a controlled fault.
    """

    name = "scripted_insertion"

    # These are the attributes the update operators look for.
    max_speed: float = 0.004      # m per control step (matches task ACTION_LIMIT)
    max_force: float = 50.0       # N — not enforced here; read by the oracle
    force_penalty_weight: float = 0.1
    collision_penalty_weight: float = 0.1
    # Stop descending the moment contact force exceeds this threshold.
    # Set below baseline first-floor-contact (~13 N) so the policy halts
    # after exactly one ramp step past the socket bottom, keeping peak
    # force seed-stable and speed-proportional.
    force_stop_n: float = 10.0

    def __init__(self, xy_tol: float = 0.0015) -> None:
        self.xy_tol = xy_tol  # alignment tolerance (m)

    def act(self, observation: TaskObservation) -> np.ndarray:
        force_mag = float(np.linalg.norm(observation.state[6:9]))
        if force_mag > self.force_stop_n:
            return np.zeros(3, dtype=np.float32)

        tip_rel: np.ndarray = observation.state[_TIP_REL]
        xy_err = tip_rel[:2]
        xy_dist = float(np.linalg.norm(xy_err))

        if xy_dist > self.xy_tol:
            # Phase 1: proportional XY correction; also descend slowly (20 % speed)
            # so that at high max_speed the peg approaches the socket opening while
            # still slightly misaligned → wall contacts scale with max_speed.
            xy_cmd = -xy_err * (self.max_speed / max(xy_dist, 1e-6))
            xy_cmd = np.clip(xy_cmd, -self.max_speed, self.max_speed)
            z_cmd  = -self.max_speed * 0.20
            return np.array([xy_cmd[0], xy_cmd[1], z_cmd], dtype=np.float32)
        else:
            # Phase 2: straight down at full speed
            return np.array([0.0, 0.0, -self.max_speed], dtype=np.float32)

    def update(self, episode_data: dict) -> None:
        pass  # scripted; nothing to learn

    def state_dict(self) -> dict:
        return {
            "max_speed": self.max_speed,
            "max_force": self.max_force,
            "xy_tol": self.xy_tol,
            "force_stop_n": self.force_stop_n,
            "force_penalty_weight": self.force_penalty_weight,
            "collision_penalty_weight": self.collision_penalty_weight,
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_speed    = state.get("max_speed", self.max_speed)
        self.max_force    = state.get("max_force", self.max_force)
        self.xy_tol       = state.get("xy_tol", self.xy_tol)
        self.force_stop_n = state.get("force_stop_n", self.force_stop_n)
        self.force_penalty_weight = state.get("force_penalty_weight", self.force_penalty_weight)
        self.collision_penalty_weight = state.get("collision_penalty_weight",
                                                    self.collision_penalty_weight)
