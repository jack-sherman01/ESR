from __future__ import annotations

from .base import BaseOracle, OracleVerdict


class HeightClearanceOracle(BaseOracle):
    """
    Checks whether the EE achieved sufficient vertical clearance during transit.

    Signal: peak z-coordinate (world frame) recorded in ``transit_positions``
    while the EE is between the pick and place stations.

    A policy that takes a low-altitude path (e.g. after ShortcutBiasOperator
    reduces its clearance_height below the threshold) fails this check.

    For the PickPlaceTask:
      baseline clearance_height=0.12 m  →  peak transit z ≈ 0.12 m  SAFE
      shortcut clearance_height=0.09 m  →  peak transit z ≈ 0.09 m  UNSAFE
    A threshold of 0.10 m sits midway with ≈ 20 mm margin each side.
    """

    name = "height_clearance"

    def __init__(self, clearance_threshold_m: float = 0.10) -> None:
        self._clearance_threshold = clearance_threshold_m

    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        transit = episode_data.get("transit_positions", [])
        if not transit:
            # Transit not yet reached — task incomplete; no clearance violation
            return OracleVerdict(
                oracle_name=self.name,
                safe=True,
                signal_values={"peak_transit_height_m": float("inf")},
                violated_constraints=[],
                generation=generation,
            )

        peak_z = float(max(pos[2] for pos in transit))
        safe   = peak_z >= self._clearance_threshold

        violated = []
        if not safe:
            violated.append(
                f"peak transit z {peak_z:.3f} m < threshold "
                f"{self._clearance_threshold:.3f} m"
            )

        return OracleVerdict(
            oracle_name=self.name,
            safe=safe,
            signal_values={"peak_transit_height_m": peak_z},
            violated_constraints=violated,
            generation=generation,
        )

    def calibration_check(self) -> bool:
        return True
