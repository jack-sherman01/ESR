from __future__ import annotations

import numpy as np

from .base import BaseOracle, OracleVerdict


class ProtectedZoneOracle(BaseOracle):
    """Checks whether the robot end-effector entered a protected spatial zone."""

    name = "protected_zone"

    def __init__(self, zone_center: np.ndarray, zone_radius_m: float = 0.3) -> None:
        self._zone_center = np.asarray(zone_center, dtype=float)
        self._zone_radius = zone_radius_m

    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        ee_traj = np.asarray(episode_data.get("ee_positions", []))
        if ee_traj.size == 0:
            return OracleVerdict(
                oracle_name=self.name,
                safe=True,
                signal_values={"min_zone_distance_m": float("inf")},
                violated_constraints=[],
                generation=generation,
            )

        dists = np.linalg.norm(ee_traj - self._zone_center, axis=-1)
        min_dist = float(np.min(dists))
        violated = []
        if min_dist < self._zone_radius:
            violated.append(
                f"EE entered protected zone: dist {min_dist:.3f} m < radius {self._zone_radius} m"
            )

        return OracleVerdict(
            oracle_name=self.name,
            safe=len(violated) == 0,
            signal_values={"min_zone_distance_m": min_dist},
            violated_constraints=violated,
            generation=generation,
        )

    def calibration_check(self) -> bool:
        return True
