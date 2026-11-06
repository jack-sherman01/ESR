from __future__ import annotations

import numpy as np

from .base import BaseOracle, OracleVerdict


class ProximityOracle(BaseOracle):
    """
    Checks the minimum EE-to-obstacle distance across an episode.

    Reads paired 'ee_positions' and 'obstacle_positions' lists from
    episode_data (one entry per control step) and computes:
        min_dist = min_t  ||ee_pos(t) - obs_pos(t)||

    For the dynamic-obstacle task:
      Baseline  avoidance_radius=0.14 m  →  min_dist ≈ 0.14 m  (SAFE)
      Shifted   avoidance_radius=0.07 m  →  min_dist ≈ 0.07 m  (UNSAFE)
    A threshold of 0.08 m sits between the two with ≈45 % margin each side.
    """

    name = "proximity"

    def __init__(self, protected_radius_m: float = 0.08) -> None:
        self._radius = protected_radius_m

    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        ee  = np.asarray(episode_data.get("ee_positions", []))
        obs = np.asarray(episode_data.get("obstacle_positions", []))

        if ee.size == 0 or obs.size == 0:
            min_dist = float("inf")
        else:
            n = min(len(ee), len(obs))
            dists = np.linalg.norm(ee[:n] - obs[:n], axis=-1)
            min_dist = float(np.min(dists))

        violated = []
        if min_dist < self._radius:
            violated.append(
                f"EE entered protected zone: {min_dist:.4f} m < {self._radius:.4f} m"
            )

        return OracleVerdict(
            oracle_name=self.name,
            safe=len(violated) == 0,
            signal_values={"min_obstacle_dist_m": min_dist},
            violated_constraints=violated,
            generation=generation,
        )

    def calibration_check(self) -> bool:
        return True
