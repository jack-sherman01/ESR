"""Controlled update that relaxes the grip after lifting a tool."""

import copy

from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class ToolGripRelaxationOperator(BaseUpdateOperator):
    name = "tool_grip_relaxation"
    hazard_type = HazardType.POLICY_REGRESSION

    def apply(self, policy, seed=0, severity=0.5):
        candidate = copy.deepcopy(policy)
        candidate.transport_closure *= max(0.0, 1.0 - severity)
        return UpdateResult(candidate, self.name, self.hazard_type, severity, seed,
                            {"transport_closure": candidate.transport_closure})
