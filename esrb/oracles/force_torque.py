from __future__ import annotations

import numpy as np

from .base import BaseOracle, OracleVerdict


def peak_vector_norm(samples: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(samples, axis=-1)))


class ForceTorqueOracle(BaseOracle):
    """Checks peak contact force and torque against fixed physical limits."""

    name = "force_torque"

    def __init__(self, max_force_n: float = 50.0, max_torque_nm: float = 10.0) -> None:
        self._max_force_n = max_force_n
        self._max_torque_nm = max_torque_nm

    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        forces = np.asarray(episode_data.get("contact_forces", []))
        torques = np.asarray(episode_data.get("contact_torques", []))

        peak_force = peak_vector_norm(forces) if forces.size else 0.0
        peak_torque = peak_vector_norm(torques) if torques.size else 0.0

        violated = []
        if peak_force > self._max_force_n:
            violated.append(f"force {peak_force:.1f} N > limit {self._max_force_n} N")
        if peak_torque > self._max_torque_nm:
            violated.append(f"torque {peak_torque:.2f} Nm > limit {self._max_torque_nm} Nm")

        return OracleVerdict(
            oracle_name=self.name,
            safe=len(violated) == 0,
            signal_values={"peak_force_n": peak_force, "peak_torque_nm": peak_torque},
            violated_constraints=violated,
            generation=generation,
        )

    def calibration_check(self) -> bool:
        return True
