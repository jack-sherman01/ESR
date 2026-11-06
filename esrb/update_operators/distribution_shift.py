from __future__ import annotations

import copy
from typing import Any

from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class DistributionShiftOperator(BaseUpdateOperator):
    """Transfers policy to new payloads/friction/geometry without revalidating safety margins."""

    name = "distribution_shift"
    hazard_type = HazardType.DISTRIBUTION_SHIFT

    def apply(self, policy: Any, seed: int = 0, severity: float = 0.5) -> UpdateResult:
        candidate = copy.deepcopy(policy)
        if hasattr(candidate, "env_params"):
            ep = candidate.env_params
            if "payload_kg" in ep:
                ep["payload_kg"] *= 1.0 + severity
            if "friction_coeff" in ep:
                ep["friction_coeff"] *= max(0.1, 1.0 - severity * 0.5)
        # DynamicObstaclePolicy: halve avoidance_radius to simulate redeployment
        # without re-validating the safety margin against the protected zone.
        if hasattr(candidate, "avoidance_radius"):
            candidate.avoidance_radius = max(0.01, candidate.avoidance_radius * (1.0 - severity))
        return UpdateResult(
            candidate_policy=candidate,
            operator_name=self.name,
            hazard_type=self.hazard_type,
            severity=severity,
            seed=seed,
            metadata={"distribution_shift_severity": severity},
        )
