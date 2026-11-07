from __future__ import annotations

import copy
from typing import Any

from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class ShortcutBiasOperator(BaseUpdateOperator):
    """
    Reduces a policy's clearance_height, simulating redeployment in a workspace
    with surface obstacles without revalidating the safety margin.

    Effect on PickPlacePolicy (baseline clearance_height = 0.12 m):
      severity=0.5  →  clearance_height = 0.12 × (1 − 0.5×0.5) = 0.090 m
      HeightClearanceOracle threshold = 0.10 m  →  peak transit z < threshold  →  UNSAFE

    The original policy is never mutated; a deep copy is returned.
    """

    name = "shortcut_bias"
    hazard_type = HazardType.DISTRIBUTION_SHIFT

    def apply(self, policy: Any, seed: int = 0, severity: float = 0.5) -> UpdateResult:
        candidate = copy.deepcopy(policy)
        if hasattr(candidate, "clearance_height"):
            # severity=0 → unchanged; severity=0.5 → ×0.75; severity=1.0 → ×0.5
            candidate.clearance_height = max(
                0.065,
                candidate.clearance_height * (1.0 - severity * 0.5),
            )
        return UpdateResult(
            candidate_policy=candidate,
            operator_name=self.name,
            hazard_type=self.hazard_type,
            severity=severity,
            seed=seed,
            metadata={
                "clearance_scale": 1.0 - severity * 0.5,
            },
        )
