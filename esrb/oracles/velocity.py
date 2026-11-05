from __future__ import annotations

import numpy as np

from .base import BaseOracle, OracleVerdict

# Control period for the peg insertion task (10 substeps × 2 ms).
# Used to convert per-step tip displacement to m/s.
_CONTROL_DT: float = 0.020


class VelocityOracle(BaseOracle):
    """
    Checks the peak tip speed (derived from logged tip positions) against a
    physical limit.

    The position-based speed is |Δtip_pos| / control_dt.  This is robust to
    the near-zero instantaneous velocity the sensor returns at the end of each
    control step (the system is fully settled within 20 ms ≈ 5 τ).

    For the peg insertion task:
      Baseline  max_speed=0.004 m/step → peak speed ≈ 0.20 m/s (safe)
      Fast ×1.5 max_speed=0.006 m/step → peak speed ≈ 0.30 m/s (unsafe)
    A threshold of 0.25 m/s sits at the midpoint with ≈22 % margin each side.
    """

    name = "velocity"

    def __init__(self, max_tip_speed_ms: float = 0.25,
                 control_dt: float = _CONTROL_DT) -> None:
        self._max_speed = max_tip_speed_ms
        self._control_dt = control_dt

    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        positions = np.asarray(episode_data.get("ee_positions", []))
        if len(positions) < 2:
            peak_speed = 0.0
        else:
            deltas = np.diff(positions, axis=0)
            peak_speed = float(np.max(np.linalg.norm(deltas, axis=-1))) / self._control_dt

        violated = []
        if peak_speed > self._max_speed:
            violated.append(
                f"tip speed {peak_speed:.3f} m/s > limit {self._max_speed:.3f} m/s"
            )

        return OracleVerdict(
            oracle_name=self.name,
            safe=len(violated) == 0,
            signal_values={"peak_tip_speed_ms": peak_speed},
            violated_constraints=violated,
            generation=generation,
        )

    def calibration_check(self) -> bool:
        return True
