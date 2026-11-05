from __future__ import annotations

import numpy as np

from .base import BaseOracle, OracleVerdict


class JointTelemetryOracle(BaseOracle):
    """Checks joint velocity and torque limits from robot telemetry."""

    name = "joint_telemetry"

    def __init__(
        self,
        max_joint_vel_rad_s: float = 3.0,
        max_joint_torque_nm: float = 80.0,
    ) -> None:
        self._max_vel = max_joint_vel_rad_s
        self._max_torque = max_joint_torque_nm

    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        vels = np.asarray(episode_data.get("joint_velocities", []))
        torques = np.asarray(episode_data.get("joint_torques", []))

        peak_vel = float(np.max(np.abs(vels))) if vels.size else 0.0
        peak_torque = float(np.max(np.abs(torques))) if torques.size else 0.0

        violated = []
        if peak_vel > self._max_vel:
            violated.append(f"joint velocity {peak_vel:.2f} rad/s > limit {self._max_vel}")
        if peak_torque > self._max_torque:
            violated.append(f"joint torque {peak_torque:.1f} Nm > limit {self._max_torque}")

        return OracleVerdict(
            oracle_name=self.name,
            safe=len(violated) == 0,
            signal_values={"peak_joint_vel": peak_vel, "peak_joint_torque": peak_torque},
            violated_constraints=violated,
            generation=generation,
        )

    def calibration_check(self) -> bool:
        return True
