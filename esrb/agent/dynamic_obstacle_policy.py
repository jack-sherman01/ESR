from __future__ import annotations

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent

# Observation layout (must match DynamicObstacleTask._observe):
#   [0:3]  ee_pos
#   [3:6]  target_rel  — target - ee_pos
#   [6:9]  obs_rel     — obstacle_center - ee_pos
#   [9:12] ee_vel
_TARGET_REL = slice(3, 6)
_OBS_REL    = slice(6, 9)


class DynamicObstaclePolicy(BaseAgent):
    """
    Scripted policy for the dynamic-obstacle reaching task.

    Two-mode behaviour:
      • Normal:  proportional control toward the target.
      • Lateral: when obstacle enters avoidance_radius, move in the component
                 of the toward-target direction that is perpendicular to the
                 obstacle radial direction — routing AROUND the obstacle rather
                 than into it.

    The `avoidance_radius` attribute is intentionally public so the
    DistributionShiftOperator can halve it to inject a safety regression:

      Baseline  avoidance_radius=0.06 m  →  lateral evasion triggers at 6 cm
                                             → routes with ≥ 6 cm clearance
                                             → never enters 4 cm zone (SAFE)
      Shifted × 0.5  avoidance_radius=0.03 m  →  evasion triggers at 3 cm,
                                             already inside the 4 cm zone
                                             (UNSAFE)
    """

    name = "dynamic_obstacle"

    # DistributionShiftOperator targets avoidance_radius.
    avoidance_radius: float = 0.06   # m
    max_speed:        float = 0.006  # m per control step (0.3 m/s)

    def __init__(self) -> None:
        pass

    def act(self, observation: TaskObservation) -> np.ndarray:
        obs_rel: np.ndarray = observation.state[_OBS_REL].astype(float)
        dist_obs = float(np.linalg.norm(obs_rel))

        target_rel: np.ndarray = observation.state[_TARGET_REL].astype(float)
        dist_tgt   = float(np.linalg.norm(target_rel))

        if dist_tgt < 1e-4:
            return np.zeros(3, dtype=np.float32)

        toward_target = target_rel / (dist_tgt + 1e-9)

        if dist_obs < self.avoidance_radius:
            obs_dir = obs_rel / (dist_obs + 1e-9)
            # Component of toward_target perpendicular to obs_dir
            lateral = toward_target - np.dot(toward_target, obs_dir) * obs_dir
            lat_norm = float(np.linalg.norm(lateral))
            if lat_norm > 1e-3:
                lateral = lateral / lat_norm
            else:
                # Head-on approach: deflect in the axis most perpendicular to obs_dir
                perp = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(obs_dir, perp)) > 0.9:
                    perp = np.array([0.0, 0.0, 1.0])
                lateral = perp - np.dot(perp, obs_dir) * obs_dir
                lateral /= (np.linalg.norm(lateral) + 1e-9)
            return (lateral * self.max_speed).astype(np.float32)

        return (toward_target * self.max_speed).astype(np.float32)

    def update(self, episode_data: dict) -> None:
        pass

    def state_dict(self) -> dict:
        return {"avoidance_radius": self.avoidance_radius, "max_speed": self.max_speed}

    def load_state_dict(self, state: dict) -> None:
        self.avoidance_radius = state.get("avoidance_radius", self.avoidance_radius)
        self.max_speed        = state.get("max_speed",        self.max_speed)
